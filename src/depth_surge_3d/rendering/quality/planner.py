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
    reconstruct_repair_runs,
)


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
    fallback_visited_nodes_total: int
    fallback_visited_nodes_max: int
    fallback_visited_nodes_p95: float
    exemplar_processed_target_count: int
    exemplar_unfinished_target_count: int
    exemplar_patch_iteration_count: int
    exemplar_donor_evaluation_count: int


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
    reconstruct_repair_runs(records, render_shape=render_shape)
    count = int(records.size)
    if count == 0:
        result = np.empty(0, dtype=np.uint32)
        result.setflags(write=False)
        return result, 0

    parent = np.arange(count, dtype=np.uint32)
    rank = np.zeros(count, dtype=np.uint8)
    previous_starts = np.full(width, -1, dtype=np.int32)
    previous_ends = np.full(width, -1, dtype=np.int32)
    cursor = 0
    for row in range(height):
        current_starts = np.full(width, -1, dtype=np.int32)
        current_ends = np.full(width, -1, dtype=np.int32)
        row_end_pixel = (row + 1) * width
        while cursor < count and int(records[cursor]["pixel_index"]) < row_end_pixel:
            pixel = int(records[cursor]["pixel_index"])
            if pixel < row * width:
                raise ValueError("records are not ordered by output pixel")
            column = pixel - row * width
            start = cursor
            while cursor < count and int(records[cursor]["pixel_index"]) == pixel:
                cursor += 1
            end = cursor
            current_starts[column] = np.int32(start)
            current_ends[column] = np.int32(end)
            _union_group_pairs(records, parent, rank, start, end, start, end, "same")
            if column > 0 and int(current_starts[column - 1]) >= 0:
                _union_group_pairs(
                    records,
                    parent,
                    rank,
                    int(current_starts[column - 1]),
                    int(current_ends[column - 1]),
                    start,
                    end,
                    "horizontal",
                )
            if int(previous_starts[column]) >= 0:
                _union_group_pairs(
                    records,
                    parent,
                    rank,
                    int(previous_starts[column]),
                    int(previous_ends[column]),
                    start,
                    end,
                    "vertical",
                )
        previous_starts = current_starts
        previous_ends = current_ends
    if cursor != count:
        raise ValueError("record pixel lies outside render_shape")

    for index in range(count):
        parent[index] = np.uint32(_find(parent, index))
    unassigned = np.uint32(np.iinfo(np.uint32).max)
    root_component = np.full(count, unassigned, dtype=np.uint32)
    component_ids = np.empty(count, dtype=np.uint32)
    component_count = 0
    for index in range(count):
        root = int(parent[index])
        component = int(root_component[root])
        if component == int(unassigned):
            component = component_count
            root_component[root] = np.uint32(component)
            component_count += 1
        component_ids[index] = np.uint32(component)
    component_ids.setflags(write=False)
    return component_ids, component_count


def _component_member_order(
    component_ids: np.ndarray,
    component_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(component_count + 1, dtype=np.uint32)
    for component in component_ids:
        offsets[int(component) + 1] += np.uint32(1)
    running = 0
    for index in range(1, offsets.size):
        running += int(offsets[index])
        offsets[index] = np.uint32(running)
    cursors = np.array(offsets[:-1], copy=True)
    members = np.empty(component_ids.size, dtype=np.uint32)
    for record_index, component_value in enumerate(component_ids):
        component = int(component_value)
        position = int(cursors[component])
        members[position] = np.uint32(record_index)
        cursors[component] = np.uint32(position + 1)
    offsets.setflags(write=False)
    members.setflags(write=False)
    return members, offsets


def _materialize_safe_donor_mask(
    records: np.ndarray,
    analysis: QualityVisibilityAnalysis,
    region_count: int,
) -> np.ndarray:
    referenced = np.zeros(region_count + 1, dtype=np.bool_)
    for record in records:
        if int(record["backend"]) == 0:
            referenced[int(record["region_id"])] = True
    native = fallback.native_kd._require_native()
    members, _offsets = native.build_safe_donor_index(
        analysis.coverage_count,
        analysis.pure_region_id,
        referenced,
    )
    safe = np.zeros(analysis.coverage_count.shape, dtype=np.bool_)
    safe.reshape(-1)[members] = True
    return safe


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
    referenced_regions = [
        int(value) for value in local.records["region_id"] if int(value) != int(NO_REGION)
    ]
    region_count = max(referenced_regions, default=0)
    provisional = fallback.precompute_native_fallbacks(
        local.records,
        analysis,
        render_shape=render_shape,
        region_count=region_count,
        budgets=budgets,
    )
    safe_donor_mask = _materialize_safe_donor_mask(
        provisional.records,
        analysis,
        region_count,
    )
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
        local_budget_skipped_run_count=(
            budgets.local_budget_skipped_run_count - skipped_before
        ),
        fallback_indexed_donor_count=provisional.indexed_donor_count,
        fallback_index_bytes=provisional.index_bytes,
        fallback_query_count=provisional.query_count,
        fallback_visited_nodes_total=int(np.sum(visited, dtype=np.uint64)),
        fallback_visited_nodes_max=int(visited.max(initial=np.uint32(0))),
        fallback_visited_nodes_p95=_linear_p95(visited),
        exemplar_processed_target_count=exemplar_processed,
        exemplar_unfinished_target_count=exemplar_unfinished,
        exemplar_patch_iteration_count=exemplar_iterations,
        exemplar_donor_evaluation_count=exemplar_evaluations,
    )
