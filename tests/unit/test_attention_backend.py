"""Tests for the Depth Anything V3 attention backend adapter."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from src.depth_surge_3d.inference.depth import attention_backend


def _qkv() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (2, 3, 5, 8)
    return tuple(torch.randn(shape, dtype=torch.float16) for _ in range(3))


def test_prefers_flash_attention_for_eligible_inputs():
    q, k, v = _qkv()
    flash_output = torch.randn(2, 5, 3, 8, dtype=torch.float16)
    flash_func = MagicMock(return_value=flash_output)
    sdpa = MagicMock()

    with (
        patch.object(attention_backend, "_can_use_flash_attention", return_value=True),
        patch.object(attention_backend, "_load_flash_attn_func", return_value=flash_func),
        patch.object(attention_backend.F, "scaled_dot_product_attention", sdpa),
    ):
        result = attention_backend.flash_attention_or_sdpa(
            q,
            k,
            v,
            dropout_p=0.25,
            is_causal=True,
            scale=0.5,
        )

    flash_q, flash_k, flash_v = flash_func.call_args.args
    assert flash_q.shape == (2, 5, 3, 8)
    assert flash_k.shape == (2, 5, 3, 8)
    assert flash_v.shape == (2, 5, 3, 8)
    assert flash_func.call_args.kwargs == {
        "dropout_p": 0.25,
        "softmax_scale": 0.5,
        "causal": True,
    }
    torch.testing.assert_close(result, flash_output.transpose(1, 2))
    sdpa.assert_not_called()


def test_uses_sdpa_for_inputs_with_attention_mask():
    q, k, v = _qkv()
    mask = torch.ones(5, 5, dtype=torch.bool)
    expected = object()
    flash_func = MagicMock()
    sdpa = MagicMock(return_value=expected)

    with (
        patch.object(attention_backend, "_load_flash_attn_func", return_value=flash_func),
        patch.object(attention_backend.F, "scaled_dot_product_attention", sdpa),
    ):
        result = attention_backend.flash_attention_or_sdpa(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.1,
            scale=0.25,
        )

    assert result is expected
    flash_func.assert_not_called()
    sdpa.assert_called_once_with(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=0.1,
        is_causal=False,
        scale=0.25,
    )


def test_falls_back_to_sdpa_when_flash_attention_fails():
    q, k, v = _qkv()
    expected = object()
    flash_func = MagicMock(side_effect=RuntimeError("unsupported kernel"))
    sdpa = MagicMock(return_value=expected)

    with (
        patch.object(attention_backend, "_can_use_flash_attention", return_value=True),
        patch.object(attention_backend, "_load_flash_attn_func", return_value=flash_func),
        patch.object(attention_backend.F, "scaled_dot_product_attention", sdpa),
        patch.object(attention_backend, "_flash_runtime_disabled", False),
    ):
        result = attention_backend.flash_attention_or_sdpa(q, k, v)
        second_result = attention_backend.flash_attention_or_sdpa(q, k, v)

    assert result is expected
    assert second_result is expected
    flash_func.assert_called_once()
    assert sdpa.call_count == 2


def test_installs_adapter_into_both_da3_attention_modules_idempotently():
    functional_one = SimpleNamespace(scaled_dot_product_attention=MagicMock(), relu=object())
    functional_two = SimpleNamespace(scaled_dot_product_attention=MagicMock(), relu=object())
    modules = {
        "da3.attention.one": SimpleNamespace(F=functional_one),
        "da3.attention.two": SimpleNamespace(F=functional_two),
    }

    with (
        patch.object(attention_backend, "DA3_ATTENTION_MODULES", tuple(modules)),
        patch.object(attention_backend, "_load_flash_attn_func", return_value=MagicMock()),
        patch.object(
            attention_backend.importlib,
            "import_module",
            side_effect=lambda name: modules[name],
        ),
    ):
        assert attention_backend.install_da3_flash_attention() == 2
        first_proxies = tuple(module.F for module in modules.values())
        assert attention_backend.install_da3_flash_attention() == 2

    assert all(
        isinstance(module.F, attention_backend._FunctionalProxy) for module in modules.values()
    )
    assert tuple(module.F for module in modules.values()) == first_proxies
    assert modules["da3.attention.one"].F.relu is functional_one.relu
    assert modules["da3.attention.two"].F.relu is functional_two.relu


def test_does_not_patch_da3_when_flash_attention_is_unavailable():
    import_module = MagicMock()

    with (
        patch.object(attention_backend, "_load_flash_attn_func", return_value=None),
        patch.object(attention_backend.importlib, "import_module", import_module),
    ):
        assert attention_backend.install_da3_flash_attention() == 0

    import_module.assert_not_called()


def test_installs_adapter_into_diffusers_attention_processor():
    functional = SimpleNamespace(scaled_dot_product_attention=MagicMock(), gelu=object())
    module = SimpleNamespace(F=functional)

    with (
        patch.object(attention_backend, "_load_flash_attn_func", return_value=MagicMock()),
        patch.object(attention_backend.importlib, "import_module", return_value=module) as load,
    ):
        assert attention_backend.install_diffusers_flash_attention() == 1

    load.assert_called_once_with("diffusers.models.attention_processor")
    assert isinstance(module.F, attention_backend._FunctionalProxy)
    assert module.F.gelu is functional.gelu
