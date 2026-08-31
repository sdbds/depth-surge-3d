"""Optional bounded CUDA output packing for Fast stereo bands."""

from __future__ import annotations

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
_OUTPUT_BLOCK_PIXELS = 128
_MAX_COUNT_BLOCKS = 8192


if triton is not None:

    @triton.jit
    def _fast_band_payload_kernel(
        colour,
        prefill,
        filled,
        unresolved,
        payload,
        block_counts,
        pixel_count,
        BACKGROUND: tl.constexpr,
        INCLUDE_COVERAGE_FLAGS: tl.constexpr,
        BLOCK_PIXELS: tl.constexpr,
    ):
        block = tl.program_id(0)
        pixel = block * BLOCK_PIXELS + tl.arange(0, BLOCK_PIXELS)
        in_bounds = pixel < pixel_count
        lane = tl.arange(0, 16)
        sample = pixel[:, None] * 16 + lane[None, :]
        valid = tl.load(prefill + sample, mask=in_bounds[:, None], other=0).to(tl.int1)
        was_filled = tl.load(filled + sample, mask=in_bounds[:, None], other=0).to(tl.int1)
        is_hole = tl.load(unresolved + sample, mask=in_bounds[:, None], other=0).to(tl.int1)
        valid_mask = tl.sum(tl.where(valid, 1 << lane[None, :], 0), axis=1)
        mask_pointer = payload.to(tl.pointer_type(tl.int16)) + 12
        tl.store(mask_pointer + pixel, valid_mask ^ 0xFFFF, mask=in_bounds)
        repair = tl.max(was_filled.to(tl.int32), axis=1) * 0x04
        hole_bits: tl.constexpr = 0x48 if BACKGROUND else 0x40
        repair |= tl.max(is_hole.to(tl.int32), axis=1) * hole_bits
        if INCLUDE_COVERAGE_FLAGS:
            repair |= tl.where(valid_mask == 0, 0x02, tl.where(valid_mask != 0xFFFF, 0x01, 0))
        tl.store(payload + 24 + pixel_count * 5 + pixel, repair, mask=in_bounds)

        nonfinite_count = tl.full((), 0, tl.int32)
        for channel in tl.static_range(3):
            samples = tl.load(colour + sample * 3 + channel, mask=in_bounds[:, None], other=0.0)
            # Keep Torch's adjacent-pair tree, including each float32 rounding step.
            pairs = tl.sum(tl.reshape(samples, (BLOCK_PIXELS, 8, 2)), axis=2)
            quads = tl.sum(tl.reshape(pairs, (BLOCK_PIXELS, 4, 2)), axis=2)
            octets = tl.sum(tl.reshape(quads, (BLOCK_PIXELS, 2, 2)), axis=2)
            mean = tl.sum(octets, axis=1) * 0.0625
            finite = (mean == mean) & (tl.abs(mean) != float("inf"))
            nonfinite_count += tl.sum((in_bounds & ~finite).to(tl.int32), axis=0)
            rounded = libdevice.nearbyint(tl.where(mean == mean, mean, 0.0))
            rgb = tl.minimum(tl.maximum(rounded, 0.0), 255.0).to(tl.uint8)
            tl.store(
                payload + 24 + pixel_count * 2 + pixel * 3 + channel,
                rgb,
                mask=in_bounds,
            )
        filled_count = tl.sum(tl.sum(was_filled.to(tl.int32), axis=1), axis=0)
        hole_count = tl.sum(tl.sum(is_hole.to(tl.int32), axis=1), axis=0)
        tl.store(block_counts + block * 3, filled_count)
        tl.store(block_counts + block * 3 + 1, hole_count)
        tl.store(block_counts + block * 3 + 2, nonfinite_count)

    @triton.jit
    def _write_lane_counts_kernel(
        block_counts,
        payload,
        block_count,
        REDUCTION_SIZE: tl.constexpr,
    ):
        component = tl.program_id(0)
        block = tl.arange(0, REDUCTION_SIZE)
        values = tl.load(
            block_counts + block * 3 + component, mask=block < block_count, other=0
        ).to(tl.int64)
        total = tl.sum(values, axis=0)
        tl.store(payload.to(tl.pointer_type(tl.int64)) + component, total)


def _cuda_device_index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else device.index


def triton_output_available(device: torch.device) -> bool:
    return bool(
        triton is not None
        and device.type == "cuda"
        and torch.cuda.is_available()
        and _cuda_device_index(device) not in _DISABLED_CUDA_DEVICES
    )


def _inputs_are_supported(
    colour: torch.Tensor,
    prefill: torch.Tensor,
    filled: torch.Tensor,
    unresolved: torch.Tensor,
) -> bool:
    if not triton_output_available(colour.device):
        return False
    if not all(value.device == colour.device for value in (prefill, filled, unresolved)):
        return False
    if not all(value.is_contiguous() for value in (colour, prefill, filled, unresolved)):
        return False
    if colour.dtype != torch.float32 or colour.ndim != 3 or colour.shape[2] != 3:
        return False
    if prefill.ndim != 3 or prefill.shape[2] != 16:
        return False
    height, width = prefill.shape[:2]
    return bool(
        height > 0
        and width > 0
        and colour.shape[:2] == (height, width * 16)
        and all(
            value.dtype == torch.bool and value.shape == prefill.shape
            for value in (prefill, filled, unresolved)
        )
        and height * width <= _OUTPUT_BLOCK_PIXELS * _MAX_COUNT_BLOCKS
    )


def triton_fast_band_payload(
    colour: torch.Tensor,
    prefill: torch.Tensor,
    filled: torch.Tensor,
    unresolved: torch.Tensor,
    *,
    background: bool,
    include_coverage_flags: bool = False,
) -> torch.Tensor | None:
    """Pack RGB/diagnostics, optionally finalizing coverage flags in the repair byte."""
    if not _inputs_are_supported(colour, prefill, filled, unresolved):
        return None
    pixel_count = prefill.numel() // 16
    block_count = triton.cdiv(pixel_count, _OUTPUT_BLOCK_PIXELS)
    payload = torch.empty(24 + 6 * pixel_count, dtype=torch.uint8, device=colour.device)
    # At most 96 KiB; large custom bands retain the existing Torch path.
    block_counts = torch.empty((block_count, 3), dtype=torch.int32, device=colour.device)
    try:
        _fast_band_payload_kernel[(block_count,)](
            colour,
            prefill,
            filled,
            unresolved,
            payload,
            block_counts,
            pixel_count,
            BACKGROUND=background,
            INCLUDE_COVERAGE_FLAGS=include_coverage_flags,
            BLOCK_PIXELS=_OUTPUT_BLOCK_PIXELS,
            enable_fp_fusion=False,
        )
        _write_lane_counts_kernel[(3,)](
            block_counts,
            payload,
            block_count,
            REDUCTION_SIZE=triton.next_power_of_2(block_count),
        )
    except triton.errors.TritonError:
        _DISABLED_CUDA_DEVICES.add(_cuda_device_index(colour.device))
        return None
    except RuntimeError as error:
        # The Triton driver does not use Torch's OOM exception type.
        if str(error) == "Triton Error [CUDA]: out of memory":
            raise torch.cuda.OutOfMemoryError(str(error)) from error
        raise
    return payload
