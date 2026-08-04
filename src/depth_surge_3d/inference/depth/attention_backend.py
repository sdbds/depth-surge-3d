"""FlashAttention 2 adapter for Depth Anything V3 attention modules."""

from __future__ import annotations

import importlib
import logging
from functools import lru_cache
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)

DA3_ATTENTION_MODULES = (
    "depth_anything_3.model.utils.attention",
    "depth_anything_3.model.dinov2.layers.attention",
)
DIFFUSERS_ATTENTION_MODULES = ("diffusers.models.attention_processor",)

FlashAttentionFunction = Callable[..., torch.Tensor]
_flash_failure_reported = False
_flash_runtime_disabled = False


@lru_cache(maxsize=1)
def _load_flash_attn_func() -> Optional[FlashAttentionFunction]:
    try:
        from flash_attn import flash_attn_func

        return flash_attn_func
    except Exception:
        return None


def _can_use_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    enable_gqa: bool,
) -> bool:
    tensors = (query, key, value)
    return (
        attn_mask is None
        and not enable_gqa
        and all(tensor.ndim == 4 for tensor in tensors)
        and all(tensor.device.type == "cuda" for tensor in tensors)
        and query.device == key.device == value.device
        and query.dtype == key.dtype == value.dtype
        and query.dtype in (torch.float16, torch.bfloat16)
        and query.shape[0] == key.shape[0] == value.shape[0]
        and query.shape[1] == key.shape[1] == value.shape[1]
        and query.shape[-1] == key.shape[-1] == value.shape[-1]
        and query.shape[-1] <= 256
    )


def _call_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    scale: Optional[float],
    enable_gqa: bool,
) -> torch.Tensor:
    kwargs: dict[str, Any] = {
        "attn_mask": attn_mask,
        "dropout_p": dropout_p,
        "is_causal": is_causal,
    }
    if scale is not None:
        kwargs["scale"] = scale
    if enable_gqa:
        kwargs["enable_gqa"] = True
    return F.scaled_dot_product_attention(query, key, value, **kwargs)


def flash_attention_or_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Run FlashAttention 2 when supported, otherwise preserve SDPA behavior."""
    global _flash_failure_reported, _flash_runtime_disabled

    if not _flash_runtime_disabled and _can_use_flash_attention(
        query, key, value, attn_mask, enable_gqa
    ):
        flash_attn_func = _load_flash_attn_func()
        if flash_attn_func is not None:
            try:
                output = flash_attn_func(
                    query.transpose(1, 2),
                    key.transpose(1, 2),
                    value.transpose(1, 2),
                    dropout_p=dropout_p,
                    softmax_scale=scale,
                    causal=is_causal,
                )
                return output.transpose(1, 2)
            except Exception as exc:
                _flash_runtime_disabled = True
                if not _flash_failure_reported:
                    logger.warning("FlashAttention 2 failed; falling back to PyTorch SDPA: %s", exc)
                    _flash_failure_reported = True

    return _call_sdpa(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        enable_gqa,
    )


class _FunctionalProxy:
    """Delegate torch functional calls while replacing DA3's SDPA entrypoint."""

    def __init__(self, functional: Any):
        self._functional = functional

    def __getattr__(self, name: str) -> Any:
        return getattr(self._functional, name)

    @staticmethod
    def scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        *,
        scale: Optional[float] = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        return flash_attention_or_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )


def _install_flash_attention(module_names: tuple[str, ...]) -> int:
    if _flash_runtime_disabled or _load_flash_attn_func() is None:
        return 0

    enabled_modules = 0
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            functional = module.F
            if not isinstance(functional, _FunctionalProxy):
                module.F = _FunctionalProxy(functional)
            enabled_modules += 1
        except Exception as exc:
            logger.debug("Could not adapt %s for FlashAttention 2: %s", module_name, exc)

    return enabled_modules


def install_da3_flash_attention() -> int:
    """Install the adapter into DA3 modules and return the enabled module count."""
    return _install_flash_attention(DA3_ATTENTION_MODULES)


def install_diffusers_flash_attention() -> int:
    """Prefer FlashAttention 2 in Diffusers processors with transparent SDPA fallback."""
    return _install_flash_attention(DIFFUSERS_ATTENTION_MODULES)
