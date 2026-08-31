"""Optional fused CUDA selection and background fill for Fast stereo bands."""

from __future__ import annotations

import torch

try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]
except (ImportError, OSError):
    triton = None
    tl = None


_DISABLED_CUDA_DEVICES: set[int] = set()
_FILL_BLOCK_SIZE = 128


if triton is not None:

    @triton.jit
    def _background_fill_kernel(
        image,
        valid,
        projected,
        left_indexes,
        right_indexes,
        output,
        holes,
        fillable_output,
        total,
        WIDTH: tl.constexpr,
        MAX_GAP_SAMPLES: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        in_bounds = offset < total
        column = offset % WIDTH
        row = offset // WIDTH
        is_valid = tl.load(valid + offset, mask=in_bounds, other=0).to(tl.int1)
        left = tl.load(left_indexes + offset, mask=in_bounds, other=-1)
        right = tl.load(right_indexes + offset, mask=in_bounds, other=WIDTH)
        left_exists = left >= 0
        right_exists = right < WIDTH
        both_exist = left_exists & right_exists
        safe_left = tl.maximum(0, tl.minimum(left, WIDTH - 1))
        safe_right = tl.maximum(0, tl.minimum(right, WIDTH - 1))
        row_base = row * WIDTH
        left_disparity = tl.load(
            projected + row_base + safe_left,
            mask=in_bounds,
            other=0.0,
        )
        right_disparity = tl.load(
            projected + row_base + safe_right,
            mask=in_bounds,
            other=0.0,
        )
        left_distance = column - left
        right_distance = right - column
        nearer_left = left_disparity < right_disparity
        equal_prefers_left = (left_disparity == right_disparity) & (left_distance <= right_distance)
        choose_left = (left_exists & ~right_exists) | (
            both_exist & (nearer_left | equal_prefers_left)
        )
        selected = tl.where(choose_left, left, right)
        safe_selected = tl.maximum(0, tl.minimum(selected, WIDTH - 1))
        has_candidate = left_exists | right_exists
        run_width = right - left - 1
        fillable = (~is_valid) & has_candidate & (run_width <= MAX_GAP_SAMPLES)
        hole = (~is_valid) & (~fillable)

        for channel in tl.static_range(3):
            source_value = tl.load(
                image + offset * 3 + channel,
                mask=in_bounds,
                other=0.0,
            )
            candidate_value = tl.load(
                image + (row_base + safe_selected) * 3 + channel,
                mask=in_bounds,
                other=0.0,
            )
            value = tl.where(fillable, candidate_value, source_value)
            value = tl.where(hole, 0.0, value)
            tl.store(output + offset * 3 + channel, value, mask=in_bounds)
        tl.store(holes + offset, hole, mask=in_bounds)
        tl.store(fillable_output + offset, fillable, mask=in_bounds)


def _cuda_device_index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else device.index


def triton_fill_available(device: torch.device | None = None) -> bool:
    if triton is None or not torch.cuda.is_available():
        return False
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type != "cuda":
        return False
    return _cuda_device_index(device) not in _DISABLED_CUDA_DEVICES


def _inputs_are_supported(
    image: torch.Tensor,
    valid: torch.Tensor,
    projected: torch.Tensor,
    left_indexes: torch.Tensor,
    right_indexes: torch.Tensor,
) -> bool:
    if not triton_fill_available(image.device):
        return False
    if not (
        image.is_cuda
        and valid.is_cuda
        and projected.is_cuda
        and left_indexes.is_cuda
        and right_indexes.is_cuda
    ):
        return False
    if not all(
        value.device == image.device for value in (valid, projected, left_indexes, right_indexes)
    ):
        return False
    if not all(
        value.is_contiguous() for value in (image, valid, projected, left_indexes, right_indexes)
    ):
        return False
    if image.dtype != torch.float32 or image.ndim != 3 or image.shape[2] != 3:
        return False
    shape = image.shape[:2]
    return bool(
        shape[0] > 0
        and shape[1] > 0
        and valid.dtype == torch.bool
        and projected.dtype == torch.float32
        and left_indexes.dtype == torch.int32
        and right_indexes.dtype == torch.int32
        and valid.shape == shape
        and projected.shape == shape
        and left_indexes.shape == shape
        and right_indexes.shape == shape
    )


def triton_background_fill(
    image: torch.Tensor,
    valid: torch.Tensor,
    projected: torch.Tensor,
    left_indexes: torch.Tensor,
    right_indexes: torch.Tensor,
    *,
    max_gap_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Fuse candidate selection and fill, or return ``None`` for fallback."""

    if not _inputs_are_supported(
        image,
        valid,
        projected,
        left_indexes,
        right_indexes,
    ):
        return None
    output = torch.empty_like(image)
    holes = torch.empty_like(valid)
    fillable = torch.empty_like(valid)
    total = valid.numel()
    try:
        _background_fill_kernel[(triton.cdiv(total, _FILL_BLOCK_SIZE),)](
            image,
            valid,
            projected,
            left_indexes,
            right_indexes,
            output,
            holes,
            fillable,
            total,
            WIDTH=valid.shape[1],
            MAX_GAP_SAMPLES=max_gap_samples,
            BLOCK_SIZE=_FILL_BLOCK_SIZE,
        )
    except triton.errors.TritonError:
        _DISABLED_CUDA_DEVICES.add(_cuda_device_index(image.device))
        return None
    except RuntimeError as error:
        # The Triton driver does not use Torch's OOM exception type.
        if str(error) == "Triton Error [CUDA]: out of memory":
            raise torch.cuda.OutOfMemoryError(str(error)) from error
        raise
    return output, holes, fillable
