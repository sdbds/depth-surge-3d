"""Global bounded coordinator for sparse Quality repair records."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import exemplar, fallback
from .arena import QualityByteArena, QualityFixedArenas
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
    QualityRepairBudgetError,
)


QUALITY_EXEMPLAR_MIN_TARGET_PIXELS = 49
_MIB = 1024 * 1024
_GRAPH_PARENT_OFFSET = 0
_GRAPH_RANK_OFFSET = 16 * _MIB
_GRAPH_MEMBERS_OFFSET = 20 * _MIB
_GRAPH_KEYS_OFFSET = 36 * _MIB
_GRAPH_WORKSPACE_OFFSET = 52 * _MIB
_EXEMPLAR_MASK_PREFLIGHT_BYTES = 4 * (512 * 512 + 256 * 256 + 128 * 128)


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
    *,
    backend_lane_counts: tuple[int, int, int] | None = None,
) -> QualityRepairPlanSummary:
    """Collapse variable repair tables to bounded scalar diagnostics."""

    records = plan.records
    backends = records["backend"]
    if backend_lane_counts is None:
        lane_counts = tuple(
            sum(
                int(mask).bit_count()
                for mask in records["lane_mask"][backends == np.uint8(backend)]
            )
            for backend in (1, 2, 3)
        )
        backend_lane_counts = (lane_counts[0], lane_counts[1], lane_counts[2])
    elif len(backend_lane_counts) != 3 or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in backend_lane_counts
    ):
        raise ValueError("backend_lane_counts must contain three nonnegative integers")
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
                    (((left_mask << 1) & right_mask) | ((right_mask << 1) & left_mask))
                    & 0xFFFF
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
    members, offsets = (
        fallback.native_kd._require_native().build_component_member_order(
            component_ids,
            component_count,
        )
    )
    offsets.setflags(write=False)
    members.setflags(write=False)
    return members, offsets


def _fixed_graph_components(
    records: np.ndarray,
    *,
    render_shape: tuple[int, int],
    arenas: QualityFixedArenas,
) -> tuple[np.ndarray, np.ndarray, int]:
    height, width = render_shape
    count = int(records.size)
    graph_arena = arenas.graph
    if graph_arena is None:
        raise RuntimeError("Quality graph arena was released before planning")
    storage = graph_arena.storage
    parent = np.ndarray(
        (count,), dtype=np.uint32, buffer=storage, offset=_GRAPH_PARENT_OFFSET
    )
    rank = np.ndarray(
        (count,), dtype=np.uint8, buffer=storage, offset=_GRAPH_RANK_OFFSET
    )
    members = np.ndarray(
        (count,), dtype=np.uint32, buffer=storage, offset=_GRAPH_MEMBERS_OFFSET
    )
    keys = np.ndarray(
        (count,), dtype=np.uint32, buffer=storage, offset=_GRAPH_KEYS_OFFSET
    )
    workspace_count = (storage.nbytes - _GRAPH_WORKSPACE_OFFSET) // np.dtype(
        np.int32
    ).itemsize
    workspace = np.ndarray(
        (workspace_count,),
        dtype=np.int32,
        buffer=storage,
        offset=_GRAPH_WORKSPACE_OFFSET,
    )
    raw_records = records.view(np.uint8).reshape(count, REPAIR_RECORD_DTYPE.itemsize)
    native = fallback.native_kd._require_native()
    try:
        component_count = native.build_record_components_into(
            raw_records,
            height,
            width,
            parent,
            rank,
            members,
            keys,
            workspace,
        )
    except native.QualityBudgetError as error:
        raise QualityRepairBudgetError("64 MiB graph arena exceeded") from error
    component_ids = parent
    component_ids.setflags(write=False)
    members.setflags(write=False)
    return component_ids, members, int(component_count)


def _component_target_pixels(
    records: np.ndarray,
    record_indexes: np.ndarray,
    *,
    region_id: int,
    render_shape: tuple[int, int],
    arena: QualityByteArena,
) -> np.ndarray:
    output = arena.allocate(int(record_indexes.size), np.uint32)
    raw_records = records.view(np.uint8).reshape(
        records.size,
        REPAIR_RECORD_DTYPE.itemsize,
    )
    height, width = render_shape
    count = int(
        fallback.native_kd._require_native().collect_exemplar_targets_into(
            raw_records,
            record_indexes,
            region_id,
            height,
            width,
            output,
        )
    )
    result = output[:count]
    result.setflags(write=False)
    return result


def _linear_p95(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    ordered = np.sort(values.astype(np.uint32, copy=True))
    position = 0.95 * (ordered.size - 1)
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    fraction = position - lower
    return float(ordered[lower]) * (1.0 - fraction) + float(ordered[upper]) * fraction


def _linear_p95_in_place(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    values.setflags(write=True)
    values.sort(kind="heapsort")
    values.setflags(write=False)
    position = 0.95 * (values.size - 1)
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    fraction = position - lower
    return float(values[lower]) * (1.0 - fraction) + float(values[upper]) * fraction


def plan_quality_repairs(
    analysis: QualityVisibilityAnalysis,
    *,
    records: np.ndarray,
    render_shape: tuple[int, int],
    local_limit_px: int,
    budgets: QualityRepairBudgets,
    arenas: QualityFixedArenas | None = None,
) -> QualityRepairPlan:
    """Plan local, exemplar, and exact same-region fallback repair."""

    if arenas is None:
        arenas = QualityFixedArenas.allocate()
        arenas.reset_eye()
        arena_records = arenas.records()[: records.size]
        arena_records[:] = records
        records = arena_records
    repair_arena = arenas.repair
    if repair_arena is None:
        raise RuntimeError("Quality repair arena was released before planning")
    skipped_before = budgets.local_budget_skipped_run_count
    local: LocalStripPlanResult = plan_local_strips(
        records,
        analysis,
        render_shape=render_shape,
        local_limit_px=local_limit_px,
        budgets=budgets,
    )
    local_evaluated_slot_count = local.evaluated_slot_count
    local_physical_sample_read_count = local.physical_sample_read_count
    local_filled_run_count = local.local_filled_run_count
    if local.records is not records:
        records.setflags(write=True)
        records[:] = local.records
        records.setflags(write=False)
    del local
    component_ids, member_order, component_count = _fixed_graph_components(
        records,
        render_shape=render_shape,
        arenas=arenas,
    )
    if records.size == 0:
        return QualityRepairPlan(
            records=records,
            record_component_ids=component_ids,
            component_count=0,
            local_evaluated_slot_count=local_evaluated_slot_count,
            local_physical_sample_read_count=local_physical_sample_read_count,
            local_filled_run_count=local_filled_run_count,
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
    region_count = _maximum_referenced_region(records)
    provisional = fallback.precompute_native_fallbacks(
        records,
        analysis,
        render_shape=render_shape,
        region_count=region_count,
        budgets=budgets,
        in_place=True,
        arena=repair_arena,
    )
    visited = provisional.visited_nodes
    fallback_visited_nodes_total = int(np.sum(visited, dtype=np.uint64))
    fallback_visited_nodes_max = int(visited.max(initial=np.uint32(0)))
    fallback_visited_nodes_p95 = _linear_p95_in_place(visited)
    fallback_indexed_donor_count = provisional.indexed_donor_count
    fallback_index_bytes = provisional.index_bytes
    fallback_query_count = provisional.query_count
    fallback_anchor_count = provisional.anchor_fallback_count
    working = provisional.records
    del visited
    safe_donor_mask, region_has_safe_donor = fallback.materialize_safe_donor_mask(
        provisional,
        analysis,
        region_count=region_count,
        arena=repair_arena,
    )
    del provisional
    working.setflags(write=True)
    exemplar_processed = 0
    exemplar_unfinished = 0
    exemplar_iterations = 0
    exemplar_evaluations = 0
    member_cursor = 0
    component_arena_mark = repair_arena.mark()
    for component in range(component_count):
        repair_arena.rewind(component_arena_mark)
        start = member_cursor
        while (
            member_cursor < member_order.size
            and int(component_ids[int(member_order[member_cursor])]) == component
        ):
            member_cursor += 1
        end = member_cursor
        if start == end:
            raise RuntimeError("fixed graph member order omitted a component")
        indexes = member_order[start:end]
        region_id = int(working[int(indexes[0])]["region_id"])
        if not bool(region_has_safe_donor[region_id]):
            continue
        preflight_mark = repair_arena.mark()
        native = fallback.native_kd._require_native()
        try:
            target_pixels = _component_target_pixels(
                working,
                indexes,
                region_id=region_id,
                render_shape=render_shape,
                arena=repair_arena,
            )
            if target_pixels.size < QUALITY_EXEMPLAR_MIN_TARGET_PIXELS:
                continue
            mask_workspace = repair_arena.allocate(
                _EXEMPLAR_MASK_PREFLIGHT_BYTES,
                np.uint8,
            )
            may_score = native.component_may_score_exemplar(
                analysis.coverage_count,
                analysis.pure_region_id,
                safe_donor_mask,
                target_pixels,
                region_id,
                mask_workspace,
            )
        except native.QualityBudgetError as error:
            raise QualityRepairBudgetError(
                "64 MiB repair arena exceeded during exemplar preflight"
            ) from error
        finally:
            repair_arena.rewind(preflight_mark)
        if not may_score:
            continue
        repaired = exemplar.repair_component_exemplar(
            working,
            analysis,
            safe_donor_mask,
            region_id=region_id,
            render_shape=render_shape,
            budgets=budgets,
            in_place=True,
            arena=repair_arena,
            record_indexes=indexes,
        )
        exemplar_processed += repaired.processed_target_count
        exemplar_unfinished += repaired.unfinished_target_count
        exemplar_iterations += repaired.patch_iteration_count
        exemplar_evaluations += repaired.donor_evaluation_count
    final_records = fallback.finalize_safe_fallbacks(working, in_place=True)
    return QualityRepairPlan(
        records=final_records,
        record_component_ids=component_ids,
        component_count=component_count,
        local_evaluated_slot_count=local_evaluated_slot_count,
        local_physical_sample_read_count=local_physical_sample_read_count,
        local_filled_run_count=local_filled_run_count,
        local_budget_skipped_run_count=(
            budgets.local_budget_skipped_run_count - skipped_before
        ),
        fallback_indexed_donor_count=fallback_indexed_donor_count,
        fallback_index_bytes=fallback_index_bytes,
        fallback_query_count=fallback_query_count,
        fallback_anchor_count=fallback_anchor_count,
        fallback_visited_nodes_total=fallback_visited_nodes_total,
        fallback_visited_nodes_max=fallback_visited_nodes_max,
        fallback_visited_nodes_p95=fallback_visited_nodes_p95,
        exemplar_processed_target_count=exemplar_processed,
        exemplar_unfinished_target_count=exemplar_unfinished,
        exemplar_patch_iteration_count=exemplar_iterations,
        exemplar_donor_evaluation_count=exemplar_evaluations,
    )
