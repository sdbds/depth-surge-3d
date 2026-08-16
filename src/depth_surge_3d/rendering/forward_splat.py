"""Deterministic horizontal subpixel visibility for one stereo row band."""

from __future__ import annotations

from dataclasses import dataclass

import torch


HORIZONTAL_SUBPIXELS = 16
_UINT32_MASK = 0xFFFFFFFF
_SOURCE_INDEX_LIMIT = 2**32


@dataclass(frozen=True)
class SubpixelSplatResult:
    """Fine-grid colour, depth, and pre-fill validity for one row band."""

    colour: torch.Tensor
    disparity: torch.Tensor
    valid: torch.Tensor


def _validate_inputs(
    image: torch.Tensor,
    near_score: torch.Tensor,
    sample_offsets: torch.Tensor,
    source_index_offset: int,
    source_valid: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if image.ndim != 3 or near_score.ndim != 2 or sample_offsets.ndim != 2:
        raise ValueError("Splat inputs must be unbatched [H,W,3], [H,W], and [H,W]")
    if image.shape[-1] != 3:
        raise ValueError("Image must have exactly three channels")
    if near_score.shape != image.shape[:2] or sample_offsets.shape != near_score.shape:
        raise ValueError("Near score and sample offsets must match the image")
    if image.device != near_score.device or image.device != sample_offsets.device:
        raise ValueError("Image, near score, and offsets must use the same device")
    _validate_near_score(near_score)
    if sample_offsets.dtype != torch.int32:
        raise ValueError("Sample offsets must use int32")
    if image.is_floating_point() and not torch.isfinite(image).all():
        raise ValueError("Image values must be finite")
    source_valid = _validate_source_valid(image, near_score, source_valid)
    pixel_count = near_score.numel()
    if source_index_offset < 0 or source_index_offset + pixel_count > _SOURCE_INDEX_LIMIT:
        raise ValueError("Full-frame source indexes must fit in 32 bits")
    return image.to(dtype=torch.float32), source_valid


def _validate_near_score(near_score: torch.Tensor) -> None:
    if near_score.dtype != torch.float32:
        raise ValueError("Near score must use float32")
    if not torch.isfinite(near_score).all():
        raise ValueError("Near score must be finite")
    if (near_score < 0.0).any():
        raise ValueError("Near score must be nonnegative")


def _validate_source_valid(
    image: torch.Tensor,
    near_score: torch.Tensor,
    source_valid: torch.Tensor | None,
) -> torch.Tensor:
    if source_valid is None:
        return torch.ones_like(near_score, dtype=torch.bool)
    if source_valid.shape != near_score.shape:
        raise ValueError("Source validity must match the near score")
    if source_valid.dtype != torch.bool:
        raise ValueError("Source validity must use bool")
    if source_valid.device != image.device:
        raise ValueError("Source validity must use the same device as the image")
    return source_valid


def _pack_depth_source_key(
    near_score: torch.Tensor,
    source_index: torch.Tensor,
) -> torch.Tensor:
    """Pack nonnegative float32 depth and lowest-source tie order into int64."""

    positive_zero = torch.zeros((), dtype=torch.float32, device=near_score.device)
    normalized = torch.where(near_score == 0.0, positive_zero, near_score)
    depth_bits = normalized.contiguous().view(torch.int32).to(torch.int64)
    depth_bits.bitwise_and_(_UINT32_MASK)
    indexes = source_index.to(dtype=torch.int64)
    return (depth_bits << 32) | (_UINT32_MASK - indexes)


def _decode_source_index(packed_key: torch.Tensor) -> torch.Tensor:
    """Recover the full-frame source index from a valid packed winner key."""

    return _UINT32_MASK - (packed_key & _UINT32_MASK)


def _expanded_targets(sample_offsets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = sample_offsets.shape
    device = sample_offsets.device
    fine_width = width * HORIZONTAL_SUBPIXELS
    source_columns = torch.arange(width, dtype=torch.int64, device=device).view(1, width)
    first_columns = source_columns * HORIZONTAL_SUBPIXELS + sample_offsets.to(torch.int64)
    lanes = torch.arange(HORIZONTAL_SUBPIXELS, dtype=torch.int64, device=device)
    target_columns = first_columns.unsqueeze(-1) + lanes
    in_bounds = (target_columns >= 0) & (target_columns < fine_width)
    row_offsets = (
        torch.arange(height, dtype=torch.int64, device=device).view(height, 1, 1) * fine_width
    )
    safe_columns = target_columns.clamp(0, fine_width - 1)
    return row_offsets + safe_columns, in_bounds


def _winner_keys(
    near_score: torch.Tensor,
    sample_offsets: torch.Tensor,
    source_index_offset: int,
    source_valid: torch.Tensor,
) -> torch.Tensor:
    pixel_count = near_score.numel()
    target_count = pixel_count * HORIZONTAL_SUBPIXELS
    source_indexes = torch.arange(
        source_index_offset,
        source_index_offset + pixel_count,
        dtype=torch.int64,
        device=near_score.device,
    ).reshape(near_score.shape)
    source_keys = _pack_depth_source_key(near_score, source_indexes)
    target_indexes, in_bounds = _expanded_targets(sample_offsets)
    candidate_keys = source_keys.unsqueeze(-1).expand_as(target_indexes)
    candidate_valid = in_bounds & source_valid.unsqueeze(-1)
    candidate_keys = candidate_keys.masked_fill(~candidate_valid, -1)
    winners = torch.full(
        (target_count,),
        -1,
        dtype=torch.int64,
        device=near_score.device,
    )
    winners.scatter_reduce_(
        0,
        target_indexes.reshape(-1),
        candidate_keys.reshape(-1),
        reduce="amax",
        include_self=True,
    )
    return winners


def _gather_winners(
    source: torch.Tensor,
    near_score: torch.Tensor,
    winners: torch.Tensor,
    source_index_offset: int,
) -> SubpixelSplatResult:
    height, width = near_score.shape
    fine_width = width * HORIZONTAL_SUBPIXELS
    valid = winners >= 0
    local_indexes = _decode_source_index(winners) - source_index_offset
    local_indexes.clamp_(0, near_score.numel() - 1)
    colour = source.reshape(-1, 3)[local_indexes]
    disparity = near_score.reshape(-1)[local_indexes]
    colour = torch.where(valid.unsqueeze(1), colour, 0.0)
    disparity = torch.where(valid, disparity, -torch.inf)
    return SubpixelSplatResult(
        colour=colour.reshape(height, fine_width, 3),
        disparity=disparity.reshape(height, fine_width),
        valid=valid.reshape(height, fine_width),
    )


def forward_splat_band(
    image: torch.Tensor,
    near_score: torch.Tensor,
    sample_offsets: torch.Tensor,
    *,
    source_index_offset: int = 0,
    source_valid: torch.Tensor | None = None,
) -> SubpixelSplatResult:
    """Resolve 16 horizontal samples per source pixel with one integer max."""

    source, source_valid = _validate_inputs(
        image,
        near_score,
        sample_offsets,
        source_index_offset,
        source_valid,
    )
    winners = _winner_keys(near_score, sample_offsets, source_index_offset, source_valid)
    return _gather_winners(source, near_score, winners, source_index_offset)
