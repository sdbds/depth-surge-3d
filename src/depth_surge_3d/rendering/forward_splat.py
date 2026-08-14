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
    canonical: torch.Tensor,
    sample_offsets: torch.Tensor,
    source_index_offset: int,
) -> torch.Tensor:
    if image.ndim != 3 or canonical.ndim != 2 or sample_offsets.ndim != 2:
        raise ValueError("Splat inputs must be unbatched [H,W,3], [H,W], and [H,W]")
    if image.shape[-1] != 3:
        raise ValueError("Image must have exactly three channels")
    if canonical.shape != image.shape[:2] or sample_offsets.shape != canonical.shape:
        raise ValueError("Canonical disparity and sample offsets must match the image")
    if image.device != canonical.device or image.device != sample_offsets.device:
        raise ValueError("Image, canonical disparity, and offsets must use the same device")
    if canonical.dtype != torch.float32:
        raise ValueError("Canonical disparity must use float32")
    if sample_offsets.dtype != torch.int32:
        raise ValueError("Sample offsets must use int32")
    if image.is_floating_point() and not torch.isfinite(image).all():
        raise ValueError("Image values must be finite")
    if not torch.isfinite(canonical).all() or (canonical < 0.0).any() or (canonical > 1.0).any():
        raise ValueError("Canonical disparity must be finite and lie within [0, 1]")
    pixel_count = canonical.numel()
    if source_index_offset < 0 or source_index_offset + pixel_count > _SOURCE_INDEX_LIMIT:
        raise ValueError("Full-frame source indexes must fit in 32 bits")
    return image.to(dtype=torch.float32)


def _pack_depth_source_key(
    canonical: torch.Tensor,
    source_index: torch.Tensor,
) -> torch.Tensor:
    """Pack nonnegative float32 depth and lowest-source tie order into int64."""

    positive_zero = torch.zeros((), dtype=torch.float32, device=canonical.device)
    normalized = torch.where(canonical == 0.0, positive_zero, canonical)
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
    canonical: torch.Tensor,
    sample_offsets: torch.Tensor,
    source_index_offset: int,
) -> torch.Tensor:
    pixel_count = canonical.numel()
    target_count = pixel_count * HORIZONTAL_SUBPIXELS
    source_indexes = torch.arange(
        source_index_offset,
        source_index_offset + pixel_count,
        dtype=torch.int64,
        device=canonical.device,
    ).reshape(canonical.shape)
    source_keys = _pack_depth_source_key(canonical, source_indexes)
    target_indexes, in_bounds = _expanded_targets(sample_offsets)
    candidate_keys = source_keys.unsqueeze(-1).expand_as(target_indexes)
    candidate_keys = candidate_keys.masked_fill(~in_bounds, -1)
    winners = torch.full(
        (target_count,),
        -1,
        dtype=torch.int64,
        device=canonical.device,
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
    canonical: torch.Tensor,
    winners: torch.Tensor,
    source_index_offset: int,
) -> SubpixelSplatResult:
    height, width = canonical.shape
    fine_width = width * HORIZONTAL_SUBPIXELS
    valid = winners >= 0
    local_indexes = _decode_source_index(winners) - source_index_offset
    local_indexes.clamp_(0, canonical.numel() - 1)
    colour = source.reshape(-1, 3)[local_indexes]
    disparity = canonical.reshape(-1)[local_indexes]
    colour = torch.where(valid.unsqueeze(1), colour, 0.0)
    disparity = torch.where(valid, disparity, -torch.inf)
    return SubpixelSplatResult(
        colour=colour.reshape(height, fine_width, 3),
        disparity=disparity.reshape(height, fine_width),
        valid=valid.reshape(height, fine_width),
    )


def forward_splat_band(
    image: torch.Tensor,
    canonical: torch.Tensor,
    sample_offsets: torch.Tensor,
    *,
    source_index_offset: int = 0,
) -> SubpixelSplatResult:
    """Resolve 16 horizontal samples per source pixel with one integer max."""

    source = _validate_inputs(image, canonical, sample_offsets, source_index_offset)
    winners = _winner_keys(canonical, sample_offsets, source_index_offset)
    return _gather_winners(source, canonical, winners, source_index_offset)
