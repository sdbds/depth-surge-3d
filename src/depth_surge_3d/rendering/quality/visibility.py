"""Winner-aware banded visibility used only by the Quality renderer."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..forward_splat import (
    SubpixelSplatResult,
    _decode_source_index,
    _gather_winners,
    _validate_inputs,
    _winner_keys,
)


@dataclass(frozen=True)
class QualitySplatBand:
    """Fine-grid winners plus the unchanged colour/depth splat."""

    splat: SubpixelSplatResult
    winner_source_index: torch.Tensor


def _quality_splat_band(
    image: torch.Tensor,
    near_score: torch.Tensor,
    sample_offsets: torch.Tensor,
    *,
    source_index_offset: int,
    source_valid: torch.Tensor,
    validate_values: bool,
) -> QualitySplatBand:
    source, checked_valid = _validate_inputs(
        image,
        near_score,
        sample_offsets,
        source_index_offset,
        source_valid,
        validate_values=validate_values,
    )
    winners = _winner_keys(
        near_score,
        sample_offsets,
        source_index_offset,
        checked_valid,
    )
    splat = _gather_winners(source, near_score, winners, source_index_offset)
    winner_indexes = _decode_source_index(winners)
    invalid = torch.full_like(winner_indexes, 0xFFFFFFFF)
    winner_indexes = torch.where(splat.valid.reshape(-1), winner_indexes, invalid)
    return QualitySplatBand(
        splat=splat,
        winner_source_index=winner_indexes.reshape(splat.valid.shape),
    )


def quality_splat_band(
    image: torch.Tensor,
    near_score: torch.Tensor,
    sample_offsets: torch.Tensor,
    *,
    source_index_offset: int,
    source_valid: torch.Tensor,
) -> QualitySplatBand:
    """Resolve visibility and retain full-frame winner source identities."""

    return _quality_splat_band(
        image,
        near_score,
        sample_offsets,
        source_index_offset=source_index_offset,
        source_valid=source_valid,
        validate_values=True,
    )


def _quality_splat_band_prevalidated(
    image: torch.Tensor,
    near_score: torch.Tensor,
    sample_offsets: torch.Tensor,
    *,
    source_index_offset: int,
    source_valid: torch.Tensor,
) -> QualitySplatBand:
    return _quality_splat_band(
        image,
        near_score,
        sample_offsets,
        source_index_offset=source_index_offset,
        source_valid=source_valid,
        validate_values=False,
    )
