"""Global bounded coordinator for sparse Quality repair records."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import exemplar, fallback
from .local_strip import (
    LocalStripPlanResult,
    QualityRepairBudgets,
    QualityVisibilityAnalysis,
    plan_local_strips,
)
from .repair_records import (
    NO_REGION,
    QUALITY_REPAIR_RECORD_CAP,
    REPAIR_RECORD_DTYPE,
)


QUALITY_EXEMPLAR_MIN_TARGET_PIXELS = 49


@dataclass(frozen=True)
class QualityRepairPlan:
    """Final record colours and deterministic planning diagnostics."""

    records: np.ndarray
    record_component_ids: np.ndarray
    component_count: int
    local_evaluated_slot_count: int
    local_physical_sample_read_count: int
    local_filled_run_count: int
    local_budget_skipped_run_count: int
    fallback_indexed_donor_count: int
    fallback_index_bytes: int
    fallback_query_count: int
    fallback_anchor_count: int
    fallback_visited_nodes_total: int
    fallback_visited_nodes_max: int
    fallback_visited_nodes_p95: float
    exemplar_processed_target_count: int
    exemplar_unfinished_target_count: int
    exemplar_patch_iteration_count: int
    exemplar_donor_evaluation_count: int


@dataclass(frozen=True)
class QualityRepairPlanSummary:
    """Fixed-size diagnostics retained after an eye's repair arenas are released."""

    segment_record_count: int
    segment_table_bytes: int
    backend_lane_counts: tuple[int, int, int]
    backend_pixel_counts: tuple[int, int, int]
    backend_component_counts: tuple[int, int, int]
    local_evaluated_slot_count: int
    local_physical_sample_read_count: int
    local_budget_skipped_run_count: int
    fallback_indexed_donor_count: int
    fallback_query_count: int
    fallback_anchor_count: int
    fallback_visited_nodes_total: int
    fallback_visited_nodes_max: int
    fallback_visited_nodes_p95: float
    exemplar_donor_evaluation_count: int


def _maximum_referenced_region(records: np.ndarray) -> int:
    """Return the largest non-sentinel region without Python object expansion."""

    regions = records["region_id"]
    valid = regions != NO_REGION
    if not bool(np.any(valid)):
        return 0
    return int(np.max(regions, where=valid, initial=np.uint32(0)))


def summarize_quality_repair_plan(
    plan: QualityRepairPlan,
    repair_bits: np.ndarray,
) -> QualityRepairPlanSummary:
    """Collapse variable repair tables to bounded scalar diagnostics."""

    records = plan.records
    backends = records["backend"]
    lane_counts = tuple(
        sum(int(mask).bit_count() for mask in records["lane_mask"][backends == np.uint8(backend)])
        for backend in (1, 2, 3)
    )
    backend_lane_counts = (lane_counts[0], lane_counts[1], lane_counts[2])
    pixel_counts = tuple(
        int(np.count_nonzero(repair_bits & np.uint8(bit))) for bit in (0x04, 0x10, 0x20)
    )
    backend_pixel_counts = (pixel_counts[0], pixel_counts[1], pixel_counts[2])
    component_backend_bits = np.zeros(plan.component_count, dtype=np.uint8)
    if records.size:
        np.bitwise_or.at(
            component_backend_bits,
            plan.record_component_ids,
            np.left_shift(np.uint8(1), backends),
        )
    component_counts = tuple(
        int(np.count_nonzero(component_backend_bits & np.uint8(1 << backend)))
        for backend in (1, 2, 3)
    )
    backend_component_counts = (
        component_counts[0],
        component_counts[1],
        component_counts[2],
    )
    return QualityRepairPlanSummary(
        segment_record_count=int(records.size),
        segment_table_bytes=int(records.nbytes),
        backend_lane_counts=backend_lane_counts,
        backend_pixel_counts=backend_pixel_counts,
        backend_component_counts=backend_component_counts,
        local_evaluated_slot_count=plan.local_evaluated_slot_count,
        local_physical_sample_read_count=plan.local_physical_sample_read_count,
        local_budget_skipped_run_count=plan.local_budget_skipped_run_count,
        fallback_indexed_donor_count=plan.fallback_indexed_donor_count,
        fallback_query_count=plan.fallback_query_count,
        fallback_anchor_count=plan.fallback_anchor_count,
        fallback_visited_nodes_total=plan.fallback_visited_nodes_total,
        fallback_visited_nodes_max=plan.fallback_visited_nodes_max,
        fallback_visited_nodes_p95=plan.fallback_visited_nodes_p95,
        exemplar_donor_evaluation_count=plan.exemplar_donor_evaluation_count,
    )


def _least_set_bit(mask: int) -> int:
    return (mask & -mask).bit_length() - 1


def _find(parent: np.ndarray, item: int) -> int:
    root = item
    while int(parent[root]) != root:
        root = int(parent[root])
    while item != root:
        next_item = int(parent[item])
        parent[item] = np.uint32(root)
        item = next_item
    return root


def _union(parent: np.ndarray, rank: np.ndarray, left: int, right: int) -> None:
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root == right_root:
        return
    left_rank = int(rank[left_root])
    right_rank = int(rank[right_root])
    if left_rank < right_rank or (left_rank == right_rank and right_root < left_root):
        left_root, right_root = right_root, left_root
        left_rank, right_rank = right_rank, left_rank
    parent[right_root] = np.uint32(left_root)
    if left_rank == right_rank:
        if left_rank == np.iinfo(np.uint8).max:
            raise RuntimeError("repair component union rank overflow")
        rank[left_root] = np.uint8(left_rank + 1)


def _union_group_pairs(
    records: np.ndarray,
    parent: np.ndarray,
    rank: np.ndarray,
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
    relation: str,
) -> None:
    for left in range(left_start, left_end):
        left_region = int(records[left]["region_id"])
        left_mask = int(records[left]["lane_mask"])
        for right in range(right_start, right_end):
            if left_region != int(records[right]["region_id"]):
                continue
            right_mask = int(records[right]["lane_mask"])
            if relation == "same":
                adjacent = (
                    (((left_mask << 1) & right_mask) | ((right_mask << 1) & left_mask)) & 0xFFFF
                ) != 0
            elif relation == "horizontal":
                adjacent = bool(left_mask & 0x8000) and bool(right_mask & 0x0001)
            else:
                adjacent = bool(left_mask & right_mask)
            if adjacent:
                _union(parent, rank, left, right)


def build_record_component_ids(  # noqa: C901
    records: np.ndarray,
    *,
    render_shape: tuple[int, int],
) -> tuple[np.ndarray, int]:
    """Assign component IDs by ascending minimum canonical fine index."""

    if (
        not isinstance(records, np.ndarray)
        or records.dtype != REPAIR_RECORD_DTYPE
        or records.ndim != 1
        or not records.flags.c_contiguous
    ):
        raise TypeError("records must be a C-contiguous repair record vector")
    if len(render_shape) != 2 or any(
        isinstance(value, bool) or not isinstance(value, int) for value in render_shape
    ):
        raise TypeError("render_shape must be an integer (height, width) tuple")
    height, width = render_shape
    if height <= 0 or width <= 0 or height * width * 16 > np.iinfo(np.uint32).max:
        raise ValueError("render_shape must be positive and fit uint32 fine indexes")
    if records.size > QUALITY_REPAIR_RECORD_CAP:
        raise ValueError("repair record count exceeds the fixed arena")
    count = int(records.size)
    if count == 0:
        result = np.empty(0, dtype=np.uint32)
        result.setflags(write=False)
        return result, 0
    raw_records = records.view(np.uint8).reshape(count, REPAIR_RECORD_DTYPE.itemsize)
    (
        component_ids,
        component_count,
    ) = fallback.native_kd._require_native().build_record_component_ids(
        raw_records,
        height,
        width,
    )
    component_ids.setflags(write=False)
    return component_ids, int(component_count)


def _component_member_order(
    component_ids: np.ndarray,
    component_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    members, offsets = fallback.native_kd._require_native().build_component_member_order(
        component_ids,
        component_count,
    )
    offsets.setflags(write=False)
    members.setflags(write=False)
    return members, offsets


def _materialize_safe_donor_mask(
    records: np.ndarray,
    analysis: QualityVisibilityAnalysis,
    region_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    referenced = np.zeros(region_count + 1, dtype=np.bool_)
    for record in records:
        if int(record["backend"]) == 0:
            referenced[int(record["region_id"])] = True
    native = fallback.native_kd._require_native()
    safe, region_has_safe_donor = native.build_safe_donor_mask(
        analysis.coverage_count,
        analysis.pure_region_id,
        referenced,
    )
    safe.setflags(write=False)
    region_has_safe_donor.setflags(write=False)
    return safe, region_has_safe_donor


def _component_has_exemplar_candidate(
    records: np.ndarray,
    candidate_mask: np.ndarray,
    pure_region_id: np.ndarray,
    *,
    region_id: int,
    render_shape: tuple[int, int],
) -> bool:
    height, width = render_shape
    pixels = records["pixel_index"]
    rows = pixels // np.uint32(width)
    columns = pixels % np.uint32(width)
    y0 = max(0, int(rows.min()) - 128)
    y1 = min(height, int(rows.max()) + 129)
    x0 = max(0, int(columns.min()) - 128)
    x1 = min(width, int(columns.max()) + 129)
    return bool(
        np.any(
            candidate_mask[y0:y1, x0:x1] & (pure_region_id[y0:y1, x0:x1] == np.uint32(region_id))
        )
    )


def _linear_p95(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    ordered = np.sort(values.astype(np.uint32, copy=True))
    position = 0.95 * (ordered.size - 1)
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    fraction = position - lower
    return float(ordered[lower]) * (1.0 - fraction) + float(ordered[upper]) * fraction


def plan_quality_repairs(
    analysis: QualityVisibilityAnalysis,
    *,
    records: np.ndarray,
    render_shape: tuple[int, int],
    local_limit_px: int,
    budgets: QualityRepairBudgets,
) -> QualityRepairPlan:
    """Plan local, exemplar, and exact same-region fallback repair."""

    skipped_before = budgets.local_budget_skipped_run_count
    local: LocalStripPlanResult = plan_local_strips(
        records,
        analysis,
        render_shape=render_shape,
        local_limit_px=local_limit_px,
        budgets=budgets,
    )
    component_ids, component_count = build_record_component_ids(
        local.records,
        render_shape=render_shape,
    )
    if local.records.size == 0:
        return QualityRepairPlan(
            records=local.records,
            record_component_ids=component_ids,
            component_count=0,
            local_evaluated_slot_count=local.evaluated_slot_count,
            local_physical_sample_read_count=local.physical_sample_read_count,
            local_filled_run_count=local.local_filled_run_count,
            local_budget_skipped_run_count=(
                budgets.local_budget_skipped_run_count - skipped_before
            ),
            fallback_indexed_donor_count=0,
            fallback_index_bytes=0,
            fallback_query_count=0,
            fallback_anchor_count=0,
            fallback_visited_nodes_total=0,
            fallback_visited_nodes_max=0,
            fallback_visited_nodes_p95=0.0,
            exemplar_processed_target_count=0,
            exemplar_unfinished_target_count=0,
            exemplar_patch_iteration_count=0,
            exemplar_donor_evaluation_count=0,
        )
    member_order, component_offsets = _component_member_order(
        component_ids,
        component_count,
    )
    region_count = _maximum_referenced_region(local.records)
    provisional = fallback.precompute_native_fallbacks(
        local.records,
        analysis,
        render_shape=render_shape,
        region_count=region_count,
        budgets=budgets,
    )
    safe_donor_mask, region_has_safe_donor = _materialize_safe_donor_mask(
        provisional.records,
        analysis,
        region_count,
    )
    exemplar_candidate_mask = fallback.native_kd._require_native().build_exemplar_candidate_mask(
        safe_donor_mask,
        analysis.pure_region_id,
    )
    exemplar_candidate_mask.setflags(write=False)
    working = np.array(provisional.records, copy=True, order="C")
    exemplar_processed = 0
    exemplar_unfinished = 0
    exemplar_iterations = 0
    exemplar_evaluations = 0
    for component in range(component_count):
        start = int(component_offsets[component])
        end = int(component_offsets[component + 1])
        indexes = member_order[start:end]
        if not any(int(working[int(index)]["backend"]) == 0 for index in indexes):
            continue
        component_records = np.ascontiguousarray(working[indexes])
        region_id = int(component_records[0]["region_id"])
        if not bool(region_has_safe_donor[region_id]):
            continue
        target_pixels = np.unique(component_records["pixel_index"])
        if target_pixels.size < QUALITY_EXEMPLAR_MIN_TARGET_PIXELS:
            continue
        if not _component_has_exemplar_candidate(
            component_records,
            exemplar_candidate_mask,
            analysis.pure_region_id,
            region_id=region_id,
            render_shape=render_shape,
        ):
            continue
        if not fallback.native_kd._require_native().component_may_score_exemplar(
            analysis.coverage_count,
            analysis.pure_region_id,
            safe_donor_mask,
            target_pixels,
            region_id,
        ):
            continue
        repaired = exemplar.repair_component_exemplar(
            component_records,
            analysis,
            safe_donor_mask,
            region_id=region_id,
            render_shape=render_shape,
            budgets=budgets,
        )
        working[indexes] = repaired.records
        exemplar_processed += repaired.processed_target_count
        exemplar_unfinished += repaired.unfinished_target_count
        exemplar_iterations += repaired.patch_iteration_count
        exemplar_evaluations += repaired.donor_evaluation_count
    final_records = fallback.finalize_safe_fallbacks(working)
    visited = provisional.visited_nodes
    return QualityRepairPlan(
        records=final_records,
        record_component_ids=component_ids,
        component_count=component_count,
        local_evaluated_slot_count=local.evaluated_slot_count,
        local_physical_sample_read_count=local.physical_sample_read_count,
        local_filled_run_count=local.local_filled_run_count,
        local_budget_skipped_run_count=(budgets.local_budget_skipped_run_count - skipped_before),
        fallback_indexed_donor_count=provisional.indexed_donor_count,
        fallback_index_bytes=provisional.index_bytes,
        fallback_query_count=provisional.query_count,
        fallback_anchor_count=provisional.anchor_fallback_count,
        fallback_visited_nodes_total=int(np.sum(visited, dtype=np.uint64)),
        fallback_visited_nodes_max=int(visited.max(initial=np.uint32(0))),
        fallback_visited_nodes_p95=_linear_p95(visited),
        exemplar_processed_target_count=exemplar_processed,
        exemplar_unfinished_target_count=exemplar_unfinished,
        exemplar_patch_iteration_count=exemplar_iterations,
        exemplar_donor_evaluation_count=exemplar_evaluations,
    )
