"""Bounded three-level deterministic exemplar prototype."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import local_strip, native_kd
from .arena import QualityByteArena
from .local_strip import QualityRepairBudgets, QualityVisibilityAnalysis
from .repair_records import REPAIR_RECORD_DTYPE


_CORE_INTERIOR_SIZE = 384
_CORE_HALO_SIZE = 64
_COMPONENT_READ_MARGIN = 128
_MIN_COARSE_SIDE = 32
_PATCH_RADIUS = 3
_DONOR_SUPPORT_RADIUS = 4
_PATCH_ITERATION_CAP = 8_192
_NO_REGION = int(np.iinfo(np.uint32).max)


@dataclass(frozen=True)
class ExemplarCore:
    """One row-major target owner and its bounded working domain."""

    origin: tuple[int, int]
    interior_bbox: tuple[int, int, int, int]
    working_bbox: tuple[int, int, int, int]
    read_bbox: tuple[int, int, int, int]


@dataclass
class ExemplarLevel:
    """The five exact arrays owned by one exemplar pyramid level."""

    working_bgr: np.ndarray
    donor_mask: np.ndarray
    target_mask: np.ndarray
    processed_mask: np.ndarray
    barrier_mask: np.ndarray

    def __post_init__(self) -> None:
        if (
            not isinstance(self.working_bgr, np.ndarray)
            or self.working_bgr.dtype != np.uint8
            or self.working_bgr.ndim != 3
            or self.working_bgr.shape[2] != 3
            or not self.working_bgr.flags.c_contiguous
        ):
            raise TypeError("working_bgr must be a C-contiguous uint8 [h,w,3] array")
        shape = self.working_bgr.shape[:2]
        if shape[0] <= 0 or shape[1] <= 0:
            raise ValueError("an exemplar level cannot be empty")
        for name in (
            "donor_mask",
            "target_mask",
            "processed_mask",
            "barrier_mask",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.bool_
                or value.shape != shape
                or not value.flags.c_contiguous
            ):
                raise TypeError(f"{name} must be a matching C-contiguous bool array")


@dataclass(frozen=True)
class ExemplarRepairResult:
    """Completed full-level records and deterministic work counters."""

    records: np.ndarray
    processed_target_count: int
    unfinished_target_count: int
    patch_iteration_count: int
    donor_evaluation_count: int
    donor_mask_write_count: int = 0


@dataclass
class _ComponentWork:
    target_pixels: np.ndarray
    target_rows: np.ndarray
    target_columns: np.ndarray
    target_colours: np.ndarray
    completed: np.ndarray
    completed_colours: np.ndarray
    component_bbox: tuple[int, int, int, int]
    evaluation_count: int = 0
    patch_iteration_count: int = 0


def partition_exemplar_cores(
    component_bbox: tuple[int, int, int, int],
    *,
    frame_shape: tuple[int, int],
) -> tuple[ExemplarCore, ...]:
    """Partition a target bbox into unique 384-pixel owners with 64-pixel halos."""

    height, width = _validate_shape(frame_shape)
    y0, y1, x0, x1 = _validate_bbox(component_bbox, height, width)
    read_bbox = (
        max(0, y0 - _COMPONENT_READ_MARGIN),
        min(height, y1 + _COMPONENT_READ_MARGIN),
        max(0, x0 - _COMPONENT_READ_MARGIN),
        min(width, x1 + _COMPONENT_READ_MARGIN),
    )
    read_y0, read_y1, read_x0, read_x1 = read_bbox
    result: list[ExemplarCore] = []
    for origin_y in range(y0, y1, _CORE_INTERIOR_SIZE):
        interior_y1 = min(origin_y + _CORE_INTERIOR_SIZE, y1)
        for origin_x in range(x0, x1, _CORE_INTERIOR_SIZE):
            interior_x1 = min(origin_x + _CORE_INTERIOR_SIZE, x1)
            interior = (origin_y, interior_y1, origin_x, interior_x1)
            working = (
                max(read_y0, origin_y - _CORE_HALO_SIZE),
                min(read_y1, interior_y1 + _CORE_HALO_SIZE),
                max(read_x0, origin_x - _CORE_HALO_SIZE),
                min(read_x1, interior_x1 + _CORE_HALO_SIZE),
            )
            result.append(
                ExemplarCore(
                    origin=(origin_y, origin_x),
                    interior_bbox=interior,
                    working_bbox=working,
                    read_bbox=read_bbox,
                )
            )
    return tuple(result)


def downsample_exemplar_level(level: ExemplarLevel) -> ExemplarLevel:
    """Reduce clipped 2x2 children using the frozen masks and ties-to-even mean."""

    if not isinstance(level, ExemplarLevel):
        raise TypeError("level must be ExemplarLevel")
    child_height, child_width = level.working_bgr.shape[:2]
    parent_height = (child_height + 1) // 2
    parent_width = (child_width + 1) // 2
    totals = np.zeros((parent_height, parent_width, 3), dtype=np.uint16)
    counts = np.zeros((parent_height, parent_width), dtype=np.uint8)
    donor = np.ones((parent_height, parent_width), dtype=np.bool_)
    target = np.zeros((parent_height, parent_width), dtype=np.bool_)
    processed = np.ones((parent_height, parent_width), dtype=np.bool_)
    barrier = np.zeros((parent_height, parent_width), dtype=np.bool_)
    for delta_row, delta_column in ((0, 0), (0, 1), (1, 0), (1, 1)):
        child = level.working_bgr[delta_row::2, delta_column::2]
        rows, columns = child.shape[:2]
        if rows == 0 or columns == 0:
            continue
        destination = (slice(0, rows), slice(0, columns))
        totals[destination] += child.astype(np.uint16)
        counts[destination] += np.uint8(1)
        donor[destination] &= level.donor_mask[delta_row::2, delta_column::2]
        target[destination] |= level.target_mask[delta_row::2, delta_column::2]
        processed[destination] &= level.processed_mask[delta_row::2, delta_column::2]
        barrier[destination] |= level.barrier_mask[delta_row::2, delta_column::2]
    divisors = counts[..., None].astype(np.uint16)
    quotients = totals // divisors
    remainders = totals % divisors
    doubled = remainders * np.uint16(2)
    rounded = quotients + (
        (doubled > divisors) | ((doubled == divisors) & ((quotients & 1) != 0))
    )
    working = np.ascontiguousarray(rounded.astype(np.uint8))
    working[barrier] = np.uint8(0)
    target &= ~barrier
    donor &= ~barrier
    processed &= ~target & ~barrier
    return ExemplarLevel(working, donor, target, processed, barrier)


def build_exemplar_pyramid(full_level: ExemplarLevel) -> tuple[ExemplarLevel, ...]:
    """Build full, half, and quarter levels while retaining a 32-pixel side."""

    if not isinstance(full_level, ExemplarLevel):
        raise TypeError("full_level must be ExemplarLevel")
    levels = [full_level]
    while len(levels) < 3:
        current_height, current_width = levels[-1].working_bgr.shape[:2]
        next_shape = ((current_height + 1) // 2, (current_width + 1) // 2)
        if min(next_shape) < _MIN_COARSE_SIDE:
            break
        levels.append(downsample_exemplar_level(levels[-1]))
    return tuple(levels)


def _downsample_exemplar_level_fixed(
    level: ExemplarLevel,
    arena: QualityByteArena,
) -> ExemplarLevel:
    child_height, child_width = level.target_mask.shape
    parent_shape = ((child_height + 1) // 2, (child_width + 1) // 2)
    working = arena.allocate((*parent_shape, 3), np.uint8)
    donor = arena.allocate(parent_shape, np.bool_)
    target = arena.allocate(parent_shape, np.bool_)
    processed = arena.allocate(parent_shape, np.bool_)
    barrier = arena.allocate(parent_shape, np.bool_)
    native_kd._require_native().downsample_exemplar_level_into(
        level.working_bgr,
        level.donor_mask,
        level.target_mask,
        level.processed_mask,
        level.barrier_mask,
        working,
        donor,
        target,
        processed,
        barrier,
    )
    return ExemplarLevel(working, donor, target, processed, barrier)


def _build_exemplar_pyramid_fixed(
    full_level: ExemplarLevel,
    arena: QualityByteArena,
) -> tuple[ExemplarLevel, ...]:
    levels = [full_level]
    while len(levels) < 3:
        current_height, current_width = levels[-1].target_mask.shape
        next_shape = ((current_height + 1) // 2, (current_width + 1) // 2)
        if min(next_shape) < _MIN_COARSE_SIDE:
            break
        levels.append(_downsample_exemplar_level_fixed(levels[-1], arena))
    return tuple(levels)


def select_frontier_target(level: ExemplarLevel) -> tuple[int, int, int] | None:
    """Return the canonical eligible frontier centre and its known count."""

    if not isinstance(level, ExemplarLevel):
        raise TypeError("level must be ExemplarLevel")
    selected = native_kd._require_native().select_exemplar_frontier(
        level.target_mask,
        level.processed_mask,
        level.barrier_mask,
    )
    if selected is None:
        return None
    row, column, known_count = selected
    return int(row), int(column), int(known_count)


def enumerate_donor_centres(donor_mask: np.ndarray) -> np.ndarray:
    """Enumerate row-major centres whose complete 9x9 support is donor-safe."""

    if (
        not isinstance(donor_mask, np.ndarray)
        or donor_mask.dtype != np.bool_
        or donor_mask.ndim != 2
        or not donor_mask.flags.c_contiguous
    ):
        raise TypeError("donor_mask must be a C-contiguous bool raster")
    height, width = donor_mask.shape
    if height < 2 * _DONOR_SUPPORT_RADIUS + 1 or width < 2 * _DONOR_SUPPORT_RADIUS + 1:
        return np.empty((0, 2), dtype=np.uint32)
    integral = np.zeros((height + 1, width + 1), dtype=np.uint32)
    integral[1:, 1:] = donor_mask
    np.cumsum(integral, axis=0, dtype=np.uint32, out=integral)
    np.cumsum(integral, axis=1, dtype=np.uint32, out=integral)
    support = 2 * _DONOR_SUPPORT_RADIUS + 1
    counts = (
        integral[support:, support:]
        - integral[:-support, support:]
        - integral[support:, :-support]
        + integral[:-support, :-support]
    )
    rows, columns = np.nonzero(counts == support * support)
    centres = np.empty((rows.size, 2), dtype=np.uint32)
    centres[:, 0] = rows.astype(np.uint32, copy=False) + np.uint32(
        _DONOR_SUPPORT_RADIUS
    )
    centres[:, 1] = columns.astype(np.uint32, copy=False) + np.uint32(
        _DONOR_SUPPORT_RADIUS
    )
    return centres


def _enumerate_donor_centres_fixed(
    donor_mask: np.ndarray,
    arena: QualityByteArena,
) -> np.ndarray:
    height, width = donor_mask.shape
    if height < 9 or width < 9:
        return arena.allocate((0, 2), np.uint32)
    integral = arena.allocate((height + 1, width + 1), np.uint32)
    capacity = (height - 8) * (width - 8)
    centres = arena.allocate((capacity, 2), np.uint32)
    count = int(
        native_kd._require_native().enumerate_exemplar_donor_centres_into(
            donor_mask,
            integral,
            centres,
        )
    )
    result = centres[:count]
    result.setflags(write=False)
    return result


def reflect_101_index(index: int, length: int) -> int:
    """Map any integer coordinate with OpenCV-style reflect-101 semantics."""

    index = _require_integer("index", index)
    length = _require_integer("length", length)
    if length <= 0:
        raise ValueError("length must be positive")
    if length == 1:
        return 0
    period = 2 * length - 2
    reflected = index % period
    return reflected if reflected < length else period - reflected


def subsample_candidate_indexes(
    *,
    candidate_count: int,
    evaluation_limit: int,
) -> np.ndarray:
    """Return the exact unique floor(k*N/M) candidate indexes."""

    candidate_count = _require_integer("candidate_count", candidate_count)
    evaluation_limit = _require_integer("evaluation_limit", evaluation_limit)
    if candidate_count < 0 or evaluation_limit < 0:
        raise ValueError("candidate_count and evaluation_limit must be nonnegative")
    selected_count = min(candidate_count, evaluation_limit)
    if selected_count == 0:
        return np.empty(0, dtype=np.uint32)
    if selected_count == candidate_count:
        return np.arange(candidate_count, dtype=np.uint32)
    result = np.empty(selected_count, dtype=np.uint32)
    for ordinal in range(selected_count):
        result[ordinal] = np.uint32((ordinal * candidate_count) // selected_count)
    return result


def score_exemplar_patch(
    level: ExemplarLevel,
    *,
    target_center: tuple[int, int],
    donor_center: tuple[int, int],
) -> int:
    """Score one legal 7x7 target/donor alignment for the scalar oracle."""

    if not isinstance(level, ExemplarLevel):
        raise TypeError("level must be ExemplarLevel")
    target_row, target_column = _validate_center("target_center", target_center)
    donor_row, donor_column = _validate_center("donor_center", donor_center)
    height, width = level.target_mask.shape
    if not _complete_support(
        target_row,
        target_column,
        _PATCH_RADIUS,
        height,
        width,
    ):
        raise ValueError("target_center must have complete 7x7 support")
    if not _complete_support(
        donor_row,
        donor_column,
        _DONOR_SUPPORT_RADIUS,
        height,
        width,
    ):
        raise ValueError("donor_center must have complete 9x9 support")
    return _score_patch(
        level,
        target_row,
        target_column,
        donor_row,
        donor_column,
    )


def repair_component_exemplar(
    records: np.ndarray,
    analysis: QualityVisibilityAnalysis,
    safe_donor_mask: np.ndarray,
    *,
    region_id: int,
    render_shape: tuple[int, int],
    budgets: QualityRepairBudgets,
    in_place: bool = False,
    arena: QualityByteArena | None = None,
    record_indexes: np.ndarray | None = None,
) -> ExemplarRepairResult:
    """Run the exact bounded exemplar state machine for one component."""

    height, width, region = _validate_repair_inputs(
        records,
        analysis,
        safe_donor_mask,
        region_id,
        render_shape,
        budgets,
        record_indexes,
    )
    if record_indexes is not None and arena is None:
        raise ValueError("record_indexes require the fixed-arena exemplar path")
    if in_place:
        output = records
        output.setflags(write=True)
    else:
        output = np.array(records, copy=True, order="C")
    if arena is None:
        residual_indexes = np.flatnonzero(output["backend"] == np.uint8(0))
        if residual_indexes.size == 0:
            output.setflags(write=False)
            return ExemplarRepairResult(output, 0, 0, 0, 0)
        work = _prepare_component_work(output, residual_indexes, region, width)
        selected_record_indexes = None
    else:
        selected_record_indexes = record_indexes
        if selected_record_indexes is None:
            selected_record_indexes = arena.allocate(int(output.size), np.uint32)
            for index in range(output.size):
                selected_record_indexes[index] = np.uint32(index)
        work = _prepare_component_work_fixed(
            output,
            selected_record_indexes,
            region,
            height,
            width,
            arena,
        )
        if work is None:
            output.setflags(write=False)
            return ExemplarRepairResult(output, 0, 0, 0, 0)
    _run_component_cores(
        work,
        analysis,
        safe_donor_mask,
        region,
        render_shape,
        budgets,
        arena=arena,
    )
    _commit_completed_records(
        output,
        work.target_pixels,
        work.completed,
        work.completed_colours,
        record_indexes=selected_record_indexes,
    )
    output.setflags(write=False)
    processed_count = int(np.count_nonzero(work.completed))
    return ExemplarRepairResult(
        records=output,
        processed_target_count=processed_count,
        unfinished_target_count=int(work.completed.size) - processed_count,
        patch_iteration_count=work.patch_iteration_count,
        donor_evaluation_count=work.evaluation_count,
    )


def _prepare_component_work(
    output: np.ndarray,
    residual_indexes: np.ndarray,
    region: int,
    width: int,
) -> _ComponentWork:
    target_pixels, target_colours = _collect_component_targets(
        output,
        residual_indexes,
        region,
    )
    target_rows = target_pixels // np.uint32(width)
    target_columns = target_pixels % np.uint32(width)
    component_bbox = (
        int(target_rows.min()),
        int(target_rows.max()) + 1,
        int(target_columns.min()),
        int(target_columns.max()) + 1,
    )
    return _ComponentWork(
        target_pixels=target_pixels,
        target_rows=target_rows,
        target_columns=target_columns,
        target_colours=target_colours,
        completed=np.zeros(target_pixels.size, dtype=np.bool_),
        completed_colours=np.array(target_colours, copy=True, order="C"),
        component_bbox=component_bbox,
    )


def _prepare_component_work_fixed(
    output: np.ndarray,
    record_indexes: np.ndarray,
    region: int,
    height: int,
    width: int,
    arena: QualityByteArena,
) -> _ComponentWork | None:
    target_capacity = int(record_indexes.size)
    target_pixels_storage = arena.allocate(target_capacity, np.uint32)
    target_colours_storage = arena.allocate((target_capacity, 3), np.uint8)
    raw_records = output.view(np.uint8).reshape(output.size, REPAIR_RECORD_DTYPE.itemsize)
    target_count = int(
        native_kd._require_native().collect_exemplar_targets_into(
            raw_records,
            record_indexes,
            region,
            height,
            width,
            target_pixels_storage,
            target_colours_storage,
        )
    )
    if target_count == 0:
        return None
    target_pixels = target_pixels_storage[:target_count]
    target_colours = target_colours_storage[:target_count]
    target_rows = arena.allocate(target_count, np.uint32)
    target_columns = arena.allocate(target_count, np.uint32)
    np.floor_divide(target_pixels, np.uint32(width), out=target_rows)
    np.remainder(target_pixels, np.uint32(width), out=target_columns)
    completed = arena.allocate(target_count, np.bool_, zero=True)
    completed_colours = arena.allocate((target_count, 3), np.uint8)
    completed_colours[:] = target_colours
    component_bbox = (
        int(target_rows.min()),
        int(target_rows.max()) + 1,
        int(target_columns.min()),
        int(target_columns.max()) + 1,
    )
    return _ComponentWork(
        target_pixels=target_pixels,
        target_rows=target_rows,
        target_columns=target_columns,
        target_colours=target_colours,
        completed=completed,
        completed_colours=completed_colours,
        component_bbox=component_bbox,
    )


def _run_component_cores(
    work: _ComponentWork,
    analysis: QualityVisibilityAnalysis,
    safe_donor_mask: np.ndarray,
    region: int,
    render_shape: tuple[int, int],
    budgets: QualityRepairBudgets,
    *,
    arena: QualityByteArena | None = None,
) -> None:
    if arena is None:
        cores = partition_exemplar_cores(work.component_bbox, frame_shape=render_shape)
        for core in cores:
            if not _component_budget_available(work, budgets):
                break
            _repair_core(work, core, analysis, safe_donor_mask, region, budgets)
        return
    height, width = render_shape
    y0, y1, x0, x1 = work.component_bbox
    read_bbox = (
        max(0, y0 - _COMPONENT_READ_MARGIN),
        min(height, y1 + _COMPONENT_READ_MARGIN),
        max(0, x0 - _COMPONENT_READ_MARGIN),
        min(width, x1 + _COMPONENT_READ_MARGIN),
    )
    read_y0, read_y1, read_x0, read_x1 = read_bbox
    core_mark = arena.mark()
    for origin_y in range(y0, y1, _CORE_INTERIOR_SIZE):
        interior_y1 = min(origin_y + _CORE_INTERIOR_SIZE, y1)
        for origin_x in range(x0, x1, _CORE_INTERIOR_SIZE):
            if not _component_budget_available(work, budgets):
                return
            arena.rewind(core_mark)
            interior_x1 = min(origin_x + _CORE_INTERIOR_SIZE, x1)
            core = ExemplarCore(
                origin=(origin_y, origin_x),
                interior_bbox=(origin_y, interior_y1, origin_x, interior_x1),
                working_bbox=(
                    max(read_y0, origin_y - _CORE_HALO_SIZE),
                    min(read_y1, interior_y1 + _CORE_HALO_SIZE),
                    max(read_x0, origin_x - _CORE_HALO_SIZE),
                    min(read_x1, interior_x1 + _CORE_HALO_SIZE),
                ),
                read_bbox=read_bbox,
            )
            _repair_core(
                work,
                core,
                analysis,
                safe_donor_mask,
                region,
                budgets,
                arena=arena,
            )


def _repair_core(
    work: _ComponentWork,
    core: ExemplarCore,
    analysis: QualityVisibilityAnalysis,
    safe_donor_mask: np.ndarray,
    region: int,
    budgets: QualityRepairBudgets,
    *,
    arena: QualityByteArena | None = None,
) -> None:
    if arena is None:
        full_level, local_target_indexes = _build_full_level(
            core,
            analysis,
            safe_donor_mask,
            work.target_pixels,
            work.target_rows,
            work.target_columns,
            work.target_colours,
            work.completed,
            work.completed_colours,
            region,
        )
    else:
        full_level = _build_full_level_fixed(
            core,
            analysis,
            safe_donor_mask,
            work,
            region,
            arena,
        )
        local_target_indexes = None
    if not np.any(full_level.target_mask):
        return
    levels = (
        build_exemplar_pyramid(full_level)
        if arena is None
        else _build_exemplar_pyramid_fixed(full_level, arena)
    )
    core_evaluations = 0
    for level_index in range(len(levels) - 1, -1, -1):
        if level_index < len(levels) - 1:
            if arena is None:
                _replicate_coarse_targets(
                    levels[level_index + 1],
                    levels[level_index],
                )
            else:
                _replicate_coarse_targets_fixed(
                    levels[level_index + 1],
                    levels[level_index],
                )
        scratch_mark = None if arena is None else arena.mark()
        try:
            iteration_count, evaluation_count = _repair_level(
                levels[level_index],
                budgets,
                core_evaluations=core_evaluations,
                component_evaluations=work.evaluation_count,
                arena=arena,
            )
        finally:
            if scratch_mark is not None:
                arena.rewind(scratch_mark)
        core_evaluations += evaluation_count
        work.evaluation_count += evaluation_count
        work.patch_iteration_count += iteration_count
        if not _core_budget_available(core_evaluations, work, budgets):
            break
    if local_target_indexes is None:
        _record_full_level_completions_fixed(
            levels[0],
            work,
            core.working_bbox,
            int(analysis.coverage_count.shape[1]),
        )
    else:
        _record_full_level_completions(
            levels[0],
            local_target_indexes,
            work.target_rows,
            work.target_columns,
            work.completed,
            work.completed_colours,
            core.working_bbox,
        )


def _component_budget_available(
    work: _ComponentWork,
    budgets: QualityRepairBudgets,
) -> bool:
    return (
        work.evaluation_count < budgets.exemplar_component_evaluation_cap
        and budgets.exemplar_eye_evaluations_consumed
        < budgets.exemplar_eye_evaluation_cap
    )


def _core_budget_available(
    core_evaluations: int,
    work: _ComponentWork,
    budgets: QualityRepairBudgets,
) -> bool:
    return (
        core_evaluations < budgets.exemplar_core_evaluation_cap
        and _component_budget_available(work, budgets)
    )


def _repair_level(
    level: ExemplarLevel,
    budgets: QualityRepairBudgets,
    *,
    core_evaluations: int,
    component_evaluations: int,
    arena: QualityByteArena | None = None,
) -> tuple[int, int]:
    iterations = 0
    evaluations = 0
    if (
        _remaining_evaluations(
            budgets,
            core_evaluations,
            component_evaluations,
            evaluations,
        )
        <= 0
    ):
        return iterations, evaluations
    candidates = (
        enumerate_donor_centres(level.donor_mask)
        if arena is None
        else _enumerate_donor_centres_fixed(level.donor_mask, arena)
    )
    if candidates.size == 0:
        return iterations, evaluations
    native = native_kd._require_native()
    if arena is None:
        donor_gradient_x, donor_gradient_y = native.build_exemplar_sobel_planes(
            level.working_bgr
        )
        selected_storage = None
    else:
        donor_gradient_x = arena.allocate(level.target_mask.shape, np.int16)
        donor_gradient_y = arena.allocate(level.target_mask.shape, np.int16)
        native.build_exemplar_sobel_planes(
            level.working_bgr,
            donor_gradient_x,
            donor_gradient_y,
        )
        selected_storage = arena.allocate(int(candidates.shape[0]), np.uint32)
    donor_gradient_x.setflags(write=False)
    donor_gradient_y.setflags(write=False)
    while iterations < _PATCH_ITERATION_CAP:
        remaining = _remaining_evaluations(
            budgets,
            core_evaluations,
            component_evaluations,
            evaluations,
        )
        if remaining <= 0:
            break
        frontier = select_frontier_target(level)
        if frontier is None:
            break
        if selected_storage is None:
            selected = subsample_candidate_indexes(
                candidate_count=int(candidates.shape[0]),
                evaluation_limit=remaining,
            )
        else:
            selected_count = min(int(candidates.shape[0]), remaining)
            native.subsample_exemplar_indexes_into(
                int(candidates.shape[0]),
                selected_count,
                selected_storage,
            )
            selected = selected_storage[:selected_count]
        if selected.size == 0:
            break
        target_row, target_column, _ = frontier
        selected_best = native.select_exemplar_donor(
            level.working_bgr,
            level.processed_mask,
            level.barrier_mask,
            target_row,
            target_column,
            candidates,
            selected,
            donor_gradient_x,
            donor_gradient_y,
        )
        best = tuple(int(value) for value in selected_best)
        charge = int(selected.size)
        evaluations += charge
        budgets.exemplar_eye_evaluations_consumed += charge
        _copy_patch(level, target_row, target_column, best[1], best[2])
        iterations += 1
    return iterations, evaluations


def _remaining_evaluations(
    budgets: QualityRepairBudgets,
    core_evaluations: int,
    component_evaluations: int,
    level_evaluations: int,
) -> int:
    return min(
        budgets.exemplar_core_evaluation_cap - core_evaluations - level_evaluations,
        budgets.exemplar_component_evaluation_cap
        - component_evaluations
        - level_evaluations,
        budgets.exemplar_eye_evaluation_cap - budgets.exemplar_eye_evaluations_consumed,
    )


def _score_patch(
    level: ExemplarLevel,
    target_row: int,
    target_column: int,
    donor_row: int,
    donor_column: int,
) -> int:
    height, width = level.target_mask.shape
    known_offsets: list[tuple[int, int]] = []
    gradient_offsets: list[tuple[int, int]] = []
    score = 0
    for delta_row in range(-_PATCH_RADIUS, _PATCH_RADIUS + 1):
        for delta_column in range(-_PATCH_RADIUS, _PATCH_RADIUS + 1):
            row = target_row + delta_row
            column = target_column + delta_column
            if not level.processed_mask[row, column]:
                continue
            known_offsets.append((delta_row, delta_column))
            for channel in range(3):
                score += 2 * abs(
                    int(level.working_bgr[row, column, channel])
                    - int(
                        level.working_bgr[
                            donor_row + delta_row,
                            donor_column + delta_column,
                            channel,
                        ]
                    )
                )
            if _target_gradient_is_known(level, row, column, height, width):
                gradient_offsets.append((delta_row, delta_column))
    for delta_row, delta_column in gradient_offsets:
        target_gradient = _target_sobel(
            level,
            target_row + delta_row,
            target_column + delta_column,
        )
        donor_gradient = _direct_sobel(
            level.working_bgr,
            donor_row + delta_row,
            donor_column + delta_column,
        )
        score += abs(target_gradient[0] - donor_gradient[0])
        score += abs(target_gradient[1] - donor_gradient[1])
    return score


def _target_gradient_is_known(
    level: ExemplarLevel,
    row: int,
    column: int,
    height: int,
    width: int,
) -> bool:
    for delta_row in (-1, 0, 1):
        mapped_row = reflect_101_index(row + delta_row, height)
        for delta_column in (-1, 0, 1):
            mapped_column = reflect_101_index(column + delta_column, width)
            if (
                level.barrier_mask[mapped_row, mapped_column]
                or not level.processed_mask[mapped_row, mapped_column]
            ):
                return False
    return True


def _target_sobel(level: ExemplarLevel, row: int, column: int) -> tuple[int, int]:
    height, width = level.target_mask.shape
    luma = np.empty((3, 3), dtype=np.int32)
    for kernel_row, delta_row in enumerate((-1, 0, 1)):
        mapped_row = reflect_101_index(row + delta_row, height)
        for kernel_column, delta_column in enumerate((-1, 0, 1)):
            mapped_column = reflect_101_index(column + delta_column, width)
            luma[kernel_row, kernel_column] = _integer_luma(
                level.working_bgr[mapped_row, mapped_column]
            )
    return _sobel_from_luma(luma)


def _direct_sobel(working_bgr: np.ndarray, row: int, column: int) -> tuple[int, int]:
    luma = np.empty((3, 3), dtype=np.int32)
    for kernel_row, delta_row in enumerate((-1, 0, 1)):
        for kernel_column, delta_column in enumerate((-1, 0, 1)):
            luma[kernel_row, kernel_column] = _integer_luma(
                working_bgr[row + delta_row, column + delta_column]
            )
    return _sobel_from_luma(luma)


def _sobel_from_luma(luma: np.ndarray) -> tuple[int, int]:
    gx = (
        -int(luma[0, 0])
        + int(luma[0, 2])
        - 2 * int(luma[1, 0])
        + 2 * int(luma[1, 2])
        - int(luma[2, 0])
        + int(luma[2, 2])
    )
    gy = (
        -int(luma[0, 0])
        - 2 * int(luma[0, 1])
        - int(luma[0, 2])
        + int(luma[2, 0])
        + 2 * int(luma[2, 1])
        + int(luma[2, 2])
    )
    return gx, gy


def _integer_luma(bgr: np.ndarray) -> int:
    return (29 * int(bgr[0]) + 150 * int(bgr[1]) + 77 * int(bgr[2]) + 128) >> 8


def _copy_patch(
    level: ExemplarLevel,
    target_row: int,
    target_column: int,
    donor_row: int,
    donor_column: int,
) -> None:
    native_kd._require_native().copy_exemplar_patch_into(
        level.working_bgr,
        level.target_mask,
        level.processed_mask,
        target_row,
        target_column,
        donor_row,
        donor_column,
    )


def _build_full_level(
    core: ExemplarCore,
    analysis: QualityVisibilityAnalysis,
    safe_donor_mask: np.ndarray,
    target_pixels: np.ndarray,
    target_rows: np.ndarray,
    target_columns: np.ndarray,
    target_colours: np.ndarray,
    completed: np.ndarray,
    completed_colours: np.ndarray,
    region: int,
) -> tuple[ExemplarLevel, np.ndarray]:
    y0, y1, x0, x1 = core.working_bbox
    interior_y0, interior_y1, interior_x0, interior_x1 = core.interior_bbox
    shape = (y1 - y0, x1 - x0)
    coverage = analysis.coverage_count[y0:y1, x0:x1]
    regions = analysis.pure_region_id[y0:y1, x0:x1]
    same_proxy = np.ascontiguousarray((coverage > 0) & (regions == np.uint32(region)))
    working = np.stack(
        tuple(plane[y0:y1, x0:x1] for plane in analysis.pure_bgr),
        axis=2,
    )
    donor = np.ascontiguousarray(
        safe_donor_mask[y0:y1, x0:x1] & same_proxy,
        dtype=np.bool_,
    )
    target = np.zeros(shape, dtype=np.bool_)
    processed = same_proxy.copy()
    barrier = ~same_proxy

    in_working = (
        (target_rows >= y0)
        & (target_rows < y1)
        & (target_columns >= x0)
        & (target_columns < x1)
    )
    local_indexes = np.flatnonzero(in_working)
    for target_index in local_indexes:
        index = int(target_index)
        row = int(target_rows[index]) - y0
        column = int(target_columns[index]) - x0
        donor[row, column] = False
        if (
            interior_y0 <= int(target_rows[index]) < interior_y1
            and interior_x0 <= int(target_columns[index]) < interior_x1
        ):
            target[row, column] = True
            processed[row, column] = False
            barrier[row, column] = False
            working[row, column] = target_colours[index]
        elif completed[index]:
            processed[row, column] = True
            barrier[row, column] = False
            working[row, column] = completed_colours[index]
        else:
            processed[row, column] = False
            barrier[row, column] = True
            working[row, column] = np.uint8(0)
    working[barrier] = np.uint8(0)
    level = ExemplarLevel(
        np.ascontiguousarray(working),
        donor,
        target,
        np.ascontiguousarray(processed),
        np.ascontiguousarray(barrier),
    )
    return level, local_indexes


def _build_full_level_fixed(
    core: ExemplarCore,
    analysis: QualityVisibilityAnalysis,
    safe_donor_mask: np.ndarray,
    work: _ComponentWork,
    region: int,
    arena: QualityByteArena,
) -> ExemplarLevel:
    y0, y1, x0, x1 = core.working_bbox
    interior_y0, interior_y1, interior_x0, interior_x1 = core.interior_bbox
    shape = (y1 - y0, x1 - x0)
    working = arena.allocate((*shape, 3), np.uint8)
    processed = arena.allocate(shape, np.bool_)
    donor = arena.allocate(shape, np.bool_)
    target = arena.allocate(shape, np.bool_)
    barrier = arena.allocate(shape, np.bool_)
    native_kd._require_native().prepare_exemplar_full_level_into(
        analysis.coverage_count,
        analysis.pure_region_id,
        analysis.pure_bgr[0],
        analysis.pure_bgr[1],
        analysis.pure_bgr[2],
        safe_donor_mask,
        work.target_pixels,
        work.target_colours,
        work.completed,
        work.completed_colours,
        region,
        y0,
        x0,
        interior_y0,
        interior_y1,
        interior_x0,
        interior_x1,
        working,
        donor,
        target,
        processed,
        barrier,
    )
    return ExemplarLevel(working, donor, target, processed, barrier)


def _record_full_level_completions(
    full_level: ExemplarLevel,
    local_target_indexes: np.ndarray,
    target_rows: np.ndarray,
    target_columns: np.ndarray,
    completed: np.ndarray,
    completed_colours: np.ndarray,
    working_bbox: tuple[int, int, int, int],
) -> None:
    y0, _, x0, _ = working_bbox
    for target_index in local_target_indexes:
        index = int(target_index)
        local_row = int(target_rows[index]) - y0
        local_column = int(target_columns[index]) - x0
        if (
            full_level.target_mask[local_row, local_column]
            and full_level.processed_mask[local_row, local_column]
        ):
            completed[index] = True
            completed_colours[index] = full_level.working_bgr[local_row, local_column]


def _record_full_level_completions_fixed(
    full_level: ExemplarLevel,
    work: _ComponentWork,
    working_bbox: tuple[int, int, int, int],
    frame_width: int,
) -> None:
    y0, _, x0, _ = working_bbox
    native_kd._require_native().record_exemplar_completions_into(
        full_level.working_bgr,
        full_level.target_mask,
        full_level.processed_mask,
        work.target_pixels,
        frame_width,
        y0,
        x0,
        work.completed,
        work.completed_colours,
    )


def _replicate_coarse_targets(coarse: ExemplarLevel, fine: ExemplarLevel) -> None:
    height, width = fine.target_mask.shape
    for row in range(height):
        for column in range(width):
            if not fine.target_mask[row, column]:
                continue
            parent_row = row // 2
            parent_column = column // 2
            if (
                coarse.target_mask[parent_row, parent_column]
                and not coarse.barrier_mask[parent_row, parent_column]
            ):
                fine.working_bgr[row, column] = coarse.working_bgr[
                    parent_row,
                    parent_column,
                ]


def _replicate_coarse_targets_fixed(
    coarse: ExemplarLevel,
    fine: ExemplarLevel,
) -> None:
    native_kd._require_native().replicate_exemplar_targets_into(
        coarse.working_bgr,
        coarse.target_mask,
        coarse.barrier_mask,
        fine.working_bgr,
        fine.target_mask,
    )


def _commit_completed_records(
    output: np.ndarray,
    target_pixels: np.ndarray,
    completed: np.ndarray,
    completed_colours: np.ndarray,
    *,
    record_indexes: np.ndarray | None = None,
) -> None:
    if record_indexes is not None:
        raw_records = output.view(np.uint8).reshape(
            output.size,
            REPAIR_RECORD_DTYPE.itemsize,
        )
        native_kd._require_native().commit_exemplar_records_into(
            raw_records,
            record_indexes,
            target_pixels,
            completed,
            completed_colours,
        )
        return
    for record in output:
        if int(record["backend"]) != 0:
            continue
        target_index = int(
            np.searchsorted(target_pixels, np.uint32(record["pixel_index"]))
        )
        if target_index < target_pixels.size and completed[target_index]:
            record["fill_bgr"] = completed_colours[target_index]
            record["backend"] = np.uint8(2)


def _collect_component_targets(
    records: np.ndarray,
    residual_indexes: np.ndarray,
    region: int,
) -> tuple[np.ndarray, np.ndarray]:
    residual = records[residual_indexes]
    if np.any(residual["region_id"] != np.uint32(region)):
        raise ValueError("every residual component record must match region_id")
    target_pixels = np.unique(residual["pixel_index"])
    target_colours = np.empty((target_pixels.size, 3), dtype=np.uint8)
    for target_index, pixel in enumerate(target_pixels):
        matching = residual[residual["pixel_index"] == pixel]
        target_colours[target_index] = matching[0]["fill_bgr"]
        if np.any(matching["fill_bgr"] != target_colours[target_index]):
            raise ValueError(
                "records for one target pixel disagree on provisional colour"
            )
    return np.ascontiguousarray(target_pixels), target_colours


def _validate_repair_inputs(
    records: np.ndarray,
    analysis: QualityVisibilityAnalysis,
    safe_donor_mask: np.ndarray,
    region_id: int,
    render_shape: tuple[int, int],
    budgets: QualityRepairBudgets,
    record_indexes: np.ndarray | None,
) -> tuple[int, int, int]:
    height, width = _validate_shape(render_shape)
    if (
        not isinstance(records, np.ndarray)
        or records.dtype != REPAIR_RECORD_DTYPE
        or records.ndim != 1
        or not records.flags.c_contiguous
    ):
        raise TypeError("records must be a C-contiguous repair record vector")
    if not isinstance(analysis, QualityVisibilityAnalysis):
        raise TypeError("analysis must be QualityVisibilityAnalysis")
    local_strip._validate_analysis(analysis, height, width)
    if (
        not isinstance(safe_donor_mask, np.ndarray)
        or safe_donor_mask.dtype != np.bool_
        or safe_donor_mask.shape != render_shape
        or not safe_donor_mask.flags.c_contiguous
    ):
        raise TypeError("safe_donor_mask must be a matching C-contiguous bool raster")
    if not isinstance(budgets, QualityRepairBudgets):
        raise TypeError("budgets must be QualityRepairBudgets")
    region = _require_integer("region_id", region_id)
    if not 0 < region < _NO_REGION:
        raise ValueError("region_id must be a positive canonical region")
    if record_indexes is None:
        for record in records:
            pixel = int(record["pixel_index"])
            if not 0 <= pixel < height * width:
                raise ValueError("record pixel lies outside render_shape")
    elif (
        not isinstance(record_indexes, np.ndarray)
        or record_indexes.dtype != np.uint32
        or record_indexes.ndim != 1
        or not record_indexes.flags.c_contiguous
    ):
        raise TypeError("record_indexes must be a C-contiguous uint32 vector")
    return height, width, region


def _has_processed_four_neighbour(
    level: ExemplarLevel,
    row: int,
    column: int,
) -> bool:
    for delta_row, delta_column in ((-1, 0), (0, -1), (0, 1), (1, 0)):
        neighbour_row = row + delta_row
        neighbour_column = column + delta_column
        if (
            level.processed_mask[neighbour_row, neighbour_column]
            and not level.barrier_mask[neighbour_row, neighbour_column]
        ):
            return True
    return False


def _child_coordinates(
    parent_row: int,
    parent_column: int,
    child_height: int,
    child_width: int,
) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    for delta_row, delta_column in ((0, 0), (0, 1), (1, 0), (1, 1)):
        row = 2 * parent_row + delta_row
        column = 2 * parent_column + delta_column
        if row < child_height and column < child_width:
            result.append((row, column))
    return tuple(result)


def _divide_ties_to_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    doubled = 2 * remainder
    if doubled < denominator:
        return quotient
    if doubled > denominator:
        return quotient + 1
    return quotient if quotient % 2 == 0 else quotient + 1


def _validate_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
    ):
        raise TypeError("shape must be an integer (height, width) tuple")
    height, width = shape
    if height <= 0 or width <= 0 or height * width > int(np.iinfo(np.uint32).max):
        raise ValueError("shape must be positive and fit uint32 pixels")
    return height, width


def _validate_bbox(
    bbox: tuple[int, int, int, int],
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    if (
        not isinstance(bbox, tuple)
        or len(bbox) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in bbox)
    ):
        raise TypeError("bbox must be an integer (y0, y1, x0, x1) tuple")
    y0, y1, x0, x1 = bbox
    if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
        raise ValueError("bbox must be nonempty and lie inside the frame")
    return y0, y1, x0, x1


def _validate_center(name: str, center: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(center, tuple)
        or len(center) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in center
        )
    ):
        raise TypeError(f"{name} must be an integer (row, column) tuple")
    return center


def _complete_support(
    row: int,
    column: int,
    radius: int,
    height: int,
    width: int,
) -> bool:
    return radius <= row < height - radius and radius <= column < width - radius


def _require_integer(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)
