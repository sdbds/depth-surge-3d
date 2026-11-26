"""Optional resident CUDA offsets for prevalidated relative geometry."""

from __future__ import annotations

import math
import struct
from typing import Literal

import numpy as np
import torch

try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]
    from triton.language.extra.cuda import libdevice  # type: ignore[import-untyped]
except (ImportError, OSError):
    triton = None
    tl = None
    libdevice = None


_DISABLED_CUDA_DEVICES: set[int] = set()
_DISABLED_SINGLE_EYE_CUDA_DEVICES: set[int] = set()
_OFFSET_BLOCK_PIXELS = 256
_MAX_OFFSET_PIXELS = 2**31 - 1


if triton is not None:

    @triton.jit(do_not_specialize=["convergence_bits", "scale_bits"])
    def _relative_offsets_kernel(
        near_score,
        left,
        right,
        pixel_count,
        convergence_bits,
        scale_bits,
        BLOCK_PIXELS: tl.constexpr,
        DIRECTION: tl.constexpr = 0,
    ):
        pixel = tl.program_id(0) * BLOCK_PIXELS + tl.arange(0, BLOCK_PIXELS)
        score = tl.load(near_score + pixel, mask=pixel < pixel_count, other=0.0).to(tl.float64)
        convergence = convergence_bits.to(tl.uint64).to(tl.float64, bitcast=True)
        scale = scale_bits.to(tl.uint64).to(tl.float64, bitcast=True)
        # Preserve each legacy float64 rounding step, including the half-lane tie.
        shift = score - convergence
        shift = shift * scale
        shift = shift * tl.full((), 16.0, tl.float64)
        half = tl.full((), 0.5, tl.float64)
        if DIRECTION >= 0:
            left_value = libdevice.ceil(shift - half).to(tl.int32)
            tl.store(left + pixel, left_value, mask=pixel < pixel_count)
        if DIRECTION <= 0:
            right_value = libdevice.ceil(-shift - half).to(tl.int32)
            tl.store(right + pixel, right_value, mask=pixel < pixel_count)


def triton_relative_offsets_prevalidated(
    near_score: torch.Tensor,
    *,
    stereo_strength: float,
    convergence: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Consume the unchanged CPU bilinear resize of validated [0, 1] scores.

    The float32 resize stays within a conservative two-unit distance of a
    validated convergence. With strength <= 5 and fewer than 2**31 pixels,
    abs(fine_shift) < width, so both offsets and kernel indexes fit
    int32. Other layouts/sizes retain the original checked host builder.
    """
    return _relative_offsets_prevalidated(
        near_score, stereo_strength=stereo_strength, convergence=convergence, direction=0
    )


def triton_relative_eye_offsets_prevalidated(
    near_score: torch.Tensor,
    *,
    stereo_strength: float,
    convergence: float,
    eye: Literal["left", "right"],
) -> torch.Tensor | None:
    """Use the same bounded arithmetic with only one resident output plane."""
    if eye not in ("left", "right"):
        return None
    result = _relative_offsets_prevalidated(
        near_score,
        stereo_strength=stereo_strength,
        convergence=convergence,
        direction=1 if eye == "left" else -1,
    )
    return None if result is None else result[0]


def _relative_offsets_prevalidated(
    near_score: torch.Tensor,
    *,
    stereo_strength: float,
    convergence: float,
    direction: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if triton is None or near_score.device.type != "cuda" or not torch.cuda.is_available():
        return None
    device_index = near_score.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    disabled_devices = (
        _DISABLED_CUDA_DEVICES if direction == 0 else _DISABLED_SINGLE_EYE_CUDA_DEVICES
    )
    if device_index in disabled_devices:
        return None
    if (
        near_score.dtype != torch.float32
        or near_score.ndim != 2
        or not near_score.is_contiguous()
        or not 0 < near_score.numel() <= _MAX_OFFSET_PIXELS
        or not math.isfinite(stereo_strength)
        or not 0.0 <= stereo_strength <= 5.0
        or not math.isfinite(convergence)
        or not 0.0 <= convergence <= 1.0
    ):
        return None
    scale64 = np.float64(near_score.shape[1]) * np.float64(stereo_strength)
    scale64 = scale64 / np.float64(200.0)
    # Python float arguments would otherwise be narrowed to float32 by Triton.
    convergence_bits = struct.unpack("<Q", struct.pack("<d", float(np.float64(convergence))))[0]
    scale_bits = struct.unpack("<Q", struct.pack("<d", float(scale64)))[0]
    left = torch.empty(near_score.shape, dtype=torch.int32, device=near_score.device)
    right = (
        torch.empty(near_score.shape, dtype=torch.int32, device=near_score.device)
        if direction == 0
        else left
    )
    try:
        _relative_offsets_kernel[(triton.cdiv(near_score.numel(), _OFFSET_BLOCK_PIXELS),)](
            near_score,
            left,
            right,
            near_score.numel(),
            convergence_bits,
            scale_bits,
            BLOCK_PIXELS=_OFFSET_BLOCK_PIXELS,
            DIRECTION=direction,
            num_warps=4,
            enable_fp_fusion=False,
        )
    except triton.errors.TritonError:
        disabled_devices.add(device_index)
        return None
    except RuntimeError as error:
        if str(error) == "Triton Error [CUDA]: out of memory":
            raise torch.cuda.OutOfMemoryError(str(error)) from error
        raise
    return left, right
