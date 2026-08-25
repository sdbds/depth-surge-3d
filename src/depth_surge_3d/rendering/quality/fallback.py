"""Full-frame same-region safe-donor fallback prototype."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from . import local_strip, native_kd
from .arena import QUALITY_REPAIR_ARENA_BYTES, QualityByteArena
from .implicit_kd import (
    ImplicitRegionKdIndex,
    QualityRepairFallbackQueryBudgetError,
)
from .local_strip import QualityRepairBudgets, QualityVisibilityAnalysis
from .repair_records import REPAIR_RECORD_DTYPE


QUALITY_REPAIR_FALLBACK_INDEX_BYTES = 48 * 1024 * 1024
QUALITY_REPAIR_FALLBACK_VISITS_BYTES = 16 * 1024 * 1024


class QualityRepairNoSafeFallbackError(RuntimeError):
    """A repair target has no same-region safe donor within the limit."""


@dataclass(frozen=True)
class FallbackPrecomputeResult:
    """Provisional colours plus fixed-arena query diagnostics."""

    records: np.ndarray
    visited_nodes: np.ndarray
    indexed_donor_count: int
    index_bytes: int
    query_count: int
    retained_safe_donor_mask_bytes: int
    anchor_fallback_count: int


def is_safe_donor(
    analysis: QualityVisibilityAnalysis,
    *,
    row: int,
    column: int,
    region_id: int,
) -> bool:
    """Evaluate the clipped full-support safe-donor predicate."""

    if not isinstance(analysis, QualityVisibilityAnalysis):
        raise TypeError("analysis must be QualityVisibilityAnalysis")
    height, width = analysis.coverage_count.shape
    if not 0 <= row < height or not 0 <= column < width:
        raise ValueError("donor coordinate lies outside analysis")
    if region_id <= 0 or region_id == int(np.iinfo(np.uint32).max):
        raise ValueError("region_id must be a positive canonical region")
    radius = max(1, math.floor(height / 1080.0 + 0.5))
    for sample_row in range(max(0, row - radius), min(height, row + radius + 1)):
        for sample_column in range(
            max(0, column - radius),
            min(width, column + radius + 1),
        ):
            if (
                int(analysis.coverage_count[sample_row, sample_column]) != 16
                or int(analysis.pure_region_id[sample_row, sample_column]) != region_id
            ):
                return False
    return True


def precompute_native_fallbacks(
    records: np.ndarray,
    analysis: QualityVisibilityAnalysis,
    *,
    render_shape: tuple[int, int],
    region_count: int,
    budgets: QualityRepairBudgets,
    in_place: bool = False,
    arena: QualityByteArena | None = None,
) -> FallbackPrecomputeResult:
    """Store one provisional colour for every still-unplanned record."""

    height, width, count = _validate_precompute_inputs(
        records,
        analysis,
        render_shape,
        region_count,
        budgets,
    )
    if arena is None:
        arena = QualityByteArena(
            np.empty(QUALITY_REPAIR_ARENA_BYTES, dtype=np.uint8),
            "Quality repair",
        )
    arena.reset()
    if in_place:
        output = records
        output.setflags(write=True)
    else:
        output = np.array(records, copy=True, order="C")
    index_partition = arena.allocate(
        QUALITY_REPAIR_FALLBACK_INDEX_BYTES,
        np.uint8,
    )
    visits_storage = arena.allocate(
        QUALITY_REPAIR_FALLBACK_VISITS_BYTES // np.dtype(np.uint32).itemsize,
        np.uint32,
    )
    query_count = int(np.count_nonzero(output["backend"] == np.uint8(0)))
    if query_count == 0:
        output.setflags(write=False)
        visits = visits_storage[:0]
        visits.setflags(write=False)
        return FallbackPrecomputeResult(output, visits, 0, 0, 0, 0, 0)
    offsets_bytes = (count + 1) * np.dtype(np.uint32).itemsize
    use_full_bytes = count + 1
    referenced_bytes = count + 1
    member_offset = (offsets_bytes + use_full_bytes + referenced_bytes + 63) // 64 * 64
    if member_offset > QUALITY_REPAIR_FALLBACK_INDEX_BYTES:
        raise QualityRepairFallbackQueryBudgetError(
            "safe-donor descriptors exceed their 48 MiB partition"
        )
    offsets = np.ndarray(
        (count + 1,),
        dtype=np.uint32,
        buffer=index_partition,
        offset=0,
    )
    use_full = np.ndarray(
        (count + 1,),
        dtype=np.bool_,
        buffer=index_partition,
        offset=offsets_bytes,
    )
    referenced = np.ndarray(
        (count + 1,),
        dtype=np.bool_,
        buffer=index_partition,
        offset=offsets_bytes + use_full_bytes,
    )
    referenced.fill(False)
    for record in output:
        if int(record["backend"]) == 0:
            referenced[int(record["region_id"])] = True

    native = native_kd._require_native()
    member_capacity = (QUALITY_REPAIR_FALLBACK_INDEX_BYTES - member_offset) // np.dtype(
        np.uint32
    ).itemsize
    member_storage = np.ndarray(
        (member_capacity,),
        dtype=np.uint32,
        buffer=index_partition,
        offset=member_offset,
    )
    donor_count, index_bytes = native.build_safe_donor_index_into(
        analysis.coverage_count,
        analysis.pure_region_id,
        referenced,
        True,
        member_storage,
        offsets,
        use_full,
    )
    members = member_storage[: int(donor_count)]
    members.setflags(write=False)
    offsets.setflags(write=False)
    index = ImplicitRegionKdIndex(height, width, count, members, offsets)
    if index.index_bytes > QUALITY_REPAIR_FALLBACK_INDEX_BYTES:
        raise QualityRepairFallbackQueryBudgetError(
            "safe-donor index exceeds its 48 MiB partition"
        )
    if (
        query_count * np.dtype(np.uint32).itemsize
        > QUALITY_REPAIR_FALLBACK_VISITS_BYTES
    ):
        raise QualityRepairFallbackQueryBudgetError(
            "fallback visit counts exceed their 16 MiB partition"
        )

    budget_state = np.asarray(
        [
            budgets.repair_fallback_visits_consumed,
            budgets.repair_fallback_visit_cap,
        ],
        dtype=np.uint64,
    )
    raw_records = output.view(np.uint8).reshape(output.size, 16)
    fallback_limit = max(1, math.floor((256 * height) / 1080.0 + 0.5))
    try:
        query_count_written, anchor_fallback_count = native.query_fallback_records(
            index.member_index,
            index.region_offsets,
            height,
            width,
            raw_records,
            analysis.pure_bgr[0],
            analysis.pure_bgr[1],
            analysis.pure_bgr[2],
            fallback_limit,
            budget_state,
            analysis.anchor_fallbacks_seeded,
            visits_storage,
        )
    except RuntimeError as error:
        message = str(error)
        if "quality repair fallback visit cap exceeded" in message:
            raise QualityRepairFallbackQueryBudgetError(
                f"implicit k-d visit cap {budgets.repair_fallback_visit_cap} exceeded"
            ) from error
        if "no safe donor" in message:
            raise QualityRepairNoSafeFallbackError(message) from error
        raise
    finally:
        budgets.repair_fallback_visits_consumed = int(budget_state[0])
    if int(query_count_written) != query_count:
        raise RuntimeError("fallback query count changed during fixed-arena execution")
    visits = visits_storage[:query_count]
    output.setflags(write=False)
    visits.setflags(write=False)
    return FallbackPrecomputeResult(
        records=output,
        visited_nodes=visits,
        indexed_donor_count=index.indexed_sample_count,
        index_bytes=int(index_bytes),
        query_count=query_count,
        retained_safe_donor_mask_bytes=0,
        anchor_fallback_count=int(anchor_fallback_count),
    )


def finalize_safe_fallbacks(
    records: np.ndarray, *, in_place: bool = False
) -> np.ndarray:
    """Commit every residual provisional colour as backend 3."""

    if (
        not isinstance(records, np.ndarray)
        or records.dtype != REPAIR_RECORD_DTYPE
        or records.ndim != 1
        or not records.flags.c_contiguous
    ):
        raise TypeError("records must be a C-contiguous repair record vector")
    if in_place:
        output = records
        output.setflags(write=True)
    else:
        output = np.array(records, copy=True, order="C")
    output["backend"][output["backend"] == np.uint8(0)] = np.uint8(3)
    output.setflags(write=False)
    return output


def _validate_precompute_inputs(
    records: np.ndarray,
    analysis: QualityVisibilityAnalysis,
    render_shape: tuple[int, int],
    region_count: int,
    budgets: QualityRepairBudgets,
) -> tuple[int, int, int]:
    _validate_record_vector(records)
    height, width = _validate_render_shape(render_shape)
    _validate_precompute_context(analysis, budgets, render_shape)
    count = _validate_region_count(region_count)
    _validate_unplanned_records(records, height, width, count)
    return height, width, count


def _validate_record_vector(records: np.ndarray) -> None:
    if (
        not isinstance(records, np.ndarray)
        or records.dtype != REPAIR_RECORD_DTYPE
        or records.ndim != 1
        or not records.flags.c_contiguous
    ):
        raise TypeError("records must be a C-contiguous repair record vector")


def _validate_precompute_context(
    analysis: QualityVisibilityAnalysis,
    budgets: QualityRepairBudgets,
    render_shape: tuple[int, int],
) -> None:
    if not isinstance(analysis, QualityVisibilityAnalysis):
        raise TypeError("analysis must be QualityVisibilityAnalysis")
    if not isinstance(budgets, QualityRepairBudgets):
        raise TypeError("budgets must be QualityRepairBudgets")
    height, width = render_shape
    local_strip._validate_analysis(analysis, height, width)


def _validate_render_shape(render_shape: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(render_shape, tuple)
        or len(render_shape) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in render_shape
        )
    ):
        raise TypeError("render_shape must be an integer (height, width) tuple")
    height, width = render_shape
    if height <= 0 or width <= 0:
        raise ValueError("render_shape must be positive")
    return height, width


def _validate_region_count(region_count: int) -> int:
    if isinstance(region_count, (bool, np.bool_)) or not isinstance(
        region_count, (int, np.integer)
    ):
        raise TypeError("region_count must be an integer")
    count = int(region_count)
    if count <= 0 or count + 1 > int(np.iinfo(np.uint32).max):
        raise ValueError("region_count plus sentinel must fit uint32")
    return count


def _validate_unplanned_records(
    records: np.ndarray,
    height: int,
    width: int,
    region_count: int,
) -> None:
    for record in records:
        if int(record["backend"]) != 0:
            continue
        region = int(record["region_id"])
        pixel = int(record["pixel_index"])
        if not 0 < region <= region_count:
            raise ValueError("unplanned record region exceeds region_count")
        if not 0 <= pixel < height * width:
            raise ValueError("record pixel lies outside render_shape")
