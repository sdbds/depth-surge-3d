"""Optional CUDA winner-key construction without expanded candidate tensors."""

from __future__ import annotations

import torch

try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]
except (ImportError, OSError):
    triton = None
    tl = None


_HORIZONTAL_SUBPIXELS = 16
_DISABLED_CUDA_DEVICES: set[int] = set()


if triton is not None:

    @triton.jit
    def _winner_keys_kernel(
        near_score,
        sample_offsets,
        source_valid,
        winners,
        candidate_count,
        width,
        fine_width,
        source_index_offset,
        BLOCK_SIZE: tl.constexpr,
    ):
        candidate = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        candidate_in_bounds = candidate < candidate_count
        source_local = (candidate // 16).to(tl.int64)
        lane = candidate % 16
        source_column = source_local % width
        source_row = source_local // width
        sample_offset = tl.load(
            sample_offsets + source_local,
            mask=candidate_in_bounds,
            other=0,
        ).to(tl.int64)
        target_column = source_column * 16 + sample_offset + lane.to(tl.int64)
        target_in_bounds = (target_column >= 0) & (target_column < fine_width)
        valid = candidate_in_bounds & target_in_bounds
        valid &= tl.load(source_valid + source_local, mask=candidate_in_bounds, other=0)

        score = tl.load(near_score + source_local, mask=candidate_in_bounds, other=0.0)
        score = tl.where(score == 0.0, 0.0, score)
        depth_bits = score.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
        source_index = source_local + source_index_offset
        key = (depth_bits << 32) | (0xFFFFFFFF - source_index)
        target_index = source_row * fine_width + target_column
        tl.atomic_max(winners + target_index, key, mask=valid)

    @triton.jit
    def _gather_winners_kernel(
        source,
        near_score,
        winners,
        colour,
        disparity,
        valid_output,
        target_count,
        pixel_count,
        source_index_offset,
        BLOCK_SIZE: tl.constexpr,
    ):
        target = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        target_in_bounds = target < target_count
        winner = tl.load(winners + target, mask=target_in_bounds, other=-1)
        valid = winner >= 0
        source_index = 0xFFFFFFFF - (winner & 0xFFFFFFFF)
        local_index = source_index - source_index_offset
        local_index = tl.maximum(0, tl.minimum(local_index, pixel_count - 1))
        source_offset = local_index * 3

        red = tl.load(source + source_offset, mask=target_in_bounds, other=0.0)
        green = tl.load(source + source_offset + 1, mask=target_in_bounds, other=0.0)
        blue = tl.load(source + source_offset + 2, mask=target_in_bounds, other=0.0)
        score = tl.load(near_score + local_index, mask=target_in_bounds, other=0.0)
        red = tl.where(valid, red, 0.0)
        green = tl.where(valid, green, 0.0)
        blue = tl.where(valid, blue, 0.0)
        score = tl.where(valid, score, -float("inf"))

        colour_offset = target * 3
        tl.store(colour + colour_offset, red, mask=target_in_bounds)
        tl.store(colour + colour_offset + 1, green, mask=target_in_bounds)
        tl.store(colour + colour_offset + 2, blue, mask=target_in_bounds)
        tl.store(disparity + target, score, mask=target_in_bounds)
        tl.store(valid_output + target, valid, mask=target_in_bounds)

    @triton.jit
    def _compact_quality_winners_kernel(
        source,
        winners,
        colour,
        packed_missing,
        output_pixel_count,
        source_pixel_count,
        source_index_offset,
        BLOCK_SIZE: tl.constexpr,
    ):
        pixel = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        pixel_in_bounds = pixel < output_pixel_count
        blue_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        green_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        red_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        missing = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)

        for lane in range(16):
            target = pixel * 16 + lane
            winner = tl.load(winners + target, mask=pixel_in_bounds, other=-1)
            valid = winner >= 0
            source_index = 0xFFFFFFFF - (winner & 0xFFFFFFFF)
            local_index = source_index - source_index_offset
            local_index = tl.maximum(0, tl.minimum(local_index, source_pixel_count - 1))
            source_offset = local_index * 3
            load_mask = pixel_in_bounds & valid
            blue = tl.load(source + source_offset, mask=load_mask, other=0.0)
            green = tl.load(source + source_offset + 1, mask=load_mask, other=0.0)
            red = tl.load(source + source_offset + 2, mask=load_mask, other=0.0)
            blue_sum += blue.to(tl.float32)
            green_sum += green.to(tl.float32)
            red_sum += red.to(tl.float32)
            missing |= tl.where(valid, 0, 1 << lane)

        colour_offset = pixel * 3
        tl.store(colour + colour_offset, blue_sum * 0.0625, mask=pixel_in_bounds)
        tl.store(colour + colour_offset + 1, green_sum * 0.0625, mask=pixel_in_bounds)
        tl.store(colour + colour_offset + 2, red_sum * 0.0625, mask=pixel_in_bounds)
        tl.store(packed_missing + pixel, missing, mask=pixel_in_bounds)


def _cuda_device_index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else device.index


def triton_splat_available(device: torch.device | None = None) -> bool:
    if triton is None or not torch.cuda.is_available():
        return False
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type != "cuda":
        return False
    return _cuda_device_index(device) not in _DISABLED_CUDA_DEVICES


def triton_winner_keys(
    near_score: torch.Tensor,
    sample_offsets: torch.Tensor,
    source_index_offset: int,
    source_valid: torch.Tensor,
) -> torch.Tensor | None:
    """Return packed CUDA winners, or ``None`` when Triton cannot be used."""

    if not triton_splat_available(near_score.device) or not (
        near_score.is_cuda
        and near_score.is_contiguous()
        and sample_offsets.is_contiguous()
        and source_valid.is_contiguous()
    ):
        return None
    pixel_count = near_score.numel()
    candidate_count = pixel_count * _HORIZONTAL_SUBPIXELS
    fine_width = near_score.shape[1] * _HORIZONTAL_SUBPIXELS
    winners = torch.full(
        (candidate_count,),
        -1,
        dtype=torch.int64,
        device=near_score.device,
    )
    block_size = 256
    try:
        _winner_keys_kernel[(triton.cdiv(candidate_count, block_size),)](
            near_score,
            sample_offsets,
            source_valid,
            winners,
            candidate_count,
            near_score.shape[1],
            fine_width,
            source_index_offset,
            BLOCK_SIZE=block_size,
        )
    except triton.errors.TritonError:
        _DISABLED_CUDA_DEVICES.add(_cuda_device_index(near_score.device))
        return None
    return winners


def triton_gather_winners(
    source: torch.Tensor,
    near_score: torch.Tensor,
    winners: torch.Tensor,
    source_index_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Gather packed CUDA winners without materializing decoded index tensors."""

    if not triton_splat_available(source.device) or not (
        source.is_cuda
        and source.is_contiguous()
        and near_score.is_contiguous()
        and winners.is_contiguous()
    ):
        return None
    height, width = near_score.shape
    target_count = winners.numel()
    colour = torch.empty((target_count, 3), dtype=torch.float32, device=source.device)
    disparity = torch.empty(target_count, dtype=torch.float32, device=source.device)
    valid = torch.empty(target_count, dtype=torch.bool, device=source.device)
    block_size = 256
    try:
        _gather_winners_kernel[(triton.cdiv(target_count, block_size),)](
            source,
            near_score,
            winners,
            colour,
            disparity,
            valid,
            target_count,
            near_score.numel(),
            source_index_offset,
            BLOCK_SIZE=block_size,
        )
    except triton.errors.TritonError:
        _DISABLED_CUDA_DEVICES.add(_cuda_device_index(source.device))
        return None
    return (
        colour.reshape(height, width * _HORIZONTAL_SUBPIXELS, 3),
        disparity.reshape(height, width * _HORIZONTAL_SUBPIXELS),
        valid.reshape(height, width * _HORIZONTAL_SUBPIXELS),
    )


def triton_compact_quality_winners(
    source: torch.Tensor,
    winners: torch.Tensor,
    source_index_offset: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Reduce packed CUDA winners directly to RGB means and missing masks."""

    if not triton_splat_available(source.device) or not (
        source.is_cuda
        and source.is_contiguous()
        and source.ndim == 3
        and source.shape[2] == 3
        and source.dtype == torch.uint8
        and winners.is_cuda
        and winners.is_contiguous()
        and winners.dtype == torch.int64
        and winners.numel() == source.shape[0] * source.shape[1] * _HORIZONTAL_SUBPIXELS
    ):
        return None
    output_pixel_count = source.shape[0] * source.shape[1]
    colour = torch.empty(
        (source.shape[0], source.shape[1], 3),
        dtype=torch.float32,
        device=source.device,
    )
    packed_missing_flat = torch.empty(
        (output_pixel_count,),
        dtype=torch.int16,
        device=source.device,
    )
    block_size = 256
    try:
        _compact_quality_winners_kernel[(triton.cdiv(output_pixel_count, block_size),)](
            source,
            winners,
            colour,
            packed_missing_flat,
            output_pixel_count,
            output_pixel_count,
            source_index_offset,
            BLOCK_SIZE=block_size,
        )
    except triton.errors.TritonError:
        _DISABLED_CUDA_DEVICES.add(_cuda_device_index(source.device))
        return None
    return colour, packed_missing_flat.reshape(source.shape[0], source.shape[1])
