"""Optional value-only CUDA scans for exact Fast background-fill boundaries."""

from __future__ import annotations

import torch

try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]
except (ImportError, OSError):
    triton = None
    tl = None


_DISABLED_CUDA_DEVICES: set[int] = set()
_MAX_SCAN_WIDTH = 65536
_SCAN_BLOCK_SIZE = 2048


if triton is not None:

    @triton.jit
    def _scan_maximum(left, right):
        return tl.maximum(left, right)

    @triton.jit
    def _scan_minimum(left, right):
        return tl.minimum(left, right)

    @triton.jit
    def _nearest_valid_indexes_kernel(
        valid,
        left_output,
        right_output,
        summaries,
        WIDTH: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCKS_PER_ROW: tl.constexpr,
    ):
        block = tl.program_id(1)
        column = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offset = tl.program_id(0) * WIDTH + column
        in_bounds = column < WIDTH
        is_valid = tl.load(valid + offset, mask=in_bounds, other=0).to(tl.int1)
        left_seed = tl.where(is_valid, column, -1)
        left = tl.associative_scan(left_seed, 0, _scan_maximum)
        tl.store(left_output + offset, left, mask=in_bounds)
        right_seed = tl.where(is_valid, column, WIDTH)
        right = tl.associative_scan(right_seed, 0, _scan_minimum, reverse=True)
        tl.store(right_output + offset, right, mask=in_bounds)
        summary_offset = (tl.program_id(0).to(tl.int64) * BLOCKS_PER_ROW + block) * 2
        tl.store(summaries + summary_offset, tl.max(left_seed, 0))
        tl.store(summaries + summary_offset + 1, tl.min(right_seed, 0))

    @triton.jit
    def _propagate_boundaries_kernel(
        left_output,
        right_output,
        summaries,
        WIDTH: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCKS_PER_ROW: tl.constexpr,
        SUMMARY_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        neighbor = tl.arange(0, SUMMARY_SIZE)
        summary_offset = (row.to(tl.int64) * BLOCKS_PER_ROW + neighbor) * 2
        left_carry = tl.max(tl.load(summaries + summary_offset, mask=neighbor < block, other=-1), 0)
        right_carry = tl.min(
            tl.load(
                summaries + summary_offset + 1,
                mask=(neighbor > block) & (neighbor < BLOCKS_PER_ROW),
                other=WIDTH,
            ),
            0,
        )
        column = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offset = row * WIDTH + column
        in_bounds = column < WIDTH
        left = tl.load(left_output + offset, mask=in_bounds, other=-1)
        right = tl.load(right_output + offset, mask=in_bounds, other=WIDTH)
        tl.store(left_output + offset, tl.maximum(left, left_carry), mask=in_bounds)
        tl.store(right_output + offset, tl.minimum(right, right_carry), mask=in_bounds)


def _cuda_device_index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else device.index


def triton_nearest_valid_indexes(
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return exact inclusive row boundaries, or None for the Torch fallback."""
    if not (
        triton is not None
        and valid.is_cuda
        and valid.dtype == torch.bool
        and valid.ndim == 2
        and valid.is_contiguous()
        and valid.shape[0] > 0
        and 0 < valid.shape[1] <= _MAX_SCAN_WIDTH
        and valid.numel() <= torch.iinfo(torch.int32).max
        and _cuda_device_index(valid.device) not in _DISABLED_CUDA_DEVICES
    ):
        return None
    left = torch.empty(valid.shape, dtype=torch.int32, device=valid.device)
    right = torch.empty_like(left)
    block_size = min(_SCAN_BLOCK_SIZE, triton.next_power_of_2(valid.shape[1]))
    block_count = triton.cdiv(valid.shape[1], block_size)
    summaries = torch.empty(
        (valid.shape[0], block_count, 2), dtype=torch.int32, device=valid.device
    )
    grid = (valid.shape[0], block_count)
    try:
        # Bounded tiles avoid whole-row scan register spills on wide fine rasters.
        _nearest_valid_indexes_kernel[grid](
            valid,
            left,
            right,
            summaries,
            WIDTH=valid.shape[1],
            BLOCK_SIZE=block_size,
            BLOCKS_PER_ROW=block_count,
            num_warps=4,
        )
        _propagate_boundaries_kernel[grid](
            left,
            right,
            summaries,
            WIDTH=valid.shape[1],
            BLOCK_SIZE=block_size,
            BLOCKS_PER_ROW=block_count,
            SUMMARY_SIZE=triton.next_power_of_2(block_count),
            num_warps=4,
        )
    except triton.errors.TritonError:
        _DISABLED_CUDA_DEVICES.add(_cuda_device_index(valid.device))
        return None
    except RuntimeError as error:
        # The Triton driver does not use Torch's OOM exception type.
        if str(error) == "Triton Error [CUDA]: out of memory":
            raise torch.cuda.OutOfMemoryError(str(error)) from error
        raise
    return left, right
