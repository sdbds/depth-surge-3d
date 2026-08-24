"""Bounded direction-preserving local-strip prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .repair_records import (
    REPAIR_RECORD_DTYPE,
    QualityRepairBudgetError,
    reconstruct_repair_runs,
)


QUALITY_LOCAL_SLOT_CAP = 16_777_216
QUALITY_LOCAL_NEIGHBOR_SAMPLE_CAP = 268_435_456
QUALITY_REPAIR_FALLBACK_VISIT_CAP = 268_435_456
QUALITY_EXEMPLAR_CORE_EVALUATION_CAP = 2_000_000
QUALITY_EXEMPLAR_COMPONENT_EVALUATION_CAP = 8_000_000
QUALITY_EXEMPLAR_EYE_EVALUATION_CAP = 32_000_000
_UINT64_MAX = int(np.iinfo(np.uint64).max)
_ROW_DELTAS = (0, -1, 1, -2, 2)
_Bgr = tuple[int, int, int]
_Context = tuple[_Bgr, _Bgr, _Bgr]
_Donors = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class QualityVisibilityAnalysis:
    """Dense per-eye context retained by visibility Pass A."""

    coverage_count: np.ndarray
    pure_region_id: np.ndarray
    pure_bgr: tuple[np.ndarray, np.ndarray, np.ndarray]
    anchor_fallbacks_seeded: bool = False


@dataclass
class QualityRepairBudgets:
    """Checked semantic counters shared by left eye then right eye."""

    local_slot_cap: int = QUALITY_LOCAL_SLOT_CAP
    local_neighbor_sample_cap: int = QUALITY_LOCAL_NEIGHBOR_SAMPLE_CAP
    repair_fallback_visit_cap: int = QUALITY_REPAIR_FALLBACK_VISIT_CAP
    exemplar_core_evaluation_cap: int = QUALITY_EXEMPLAR_CORE_EVALUATION_CAP
    exemplar_component_evaluation_cap: int = QUALITY_EXEMPLAR_COMPONENT_EVALUATION_CAP
    exemplar_eye_evaluation_cap: int = QUALITY_EXEMPLAR_EYE_EVALUATION_CAP
    local_slots_consumed: int = field(default=0, init=False)
    local_neighbor_samples_consumed: int = field(default=0, init=False)
    repair_fallback_visits_consumed: int = field(default=0, init=False)
    exemplar_eye_evaluations_consumed: int = field(default=0, init=False)
    local_budget_skipped_run_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        for name in (
            "local_slot_cap",
            "local_neighbor_sample_cap",
            "repair_fallback_visit_cap",
            "exemplar_core_evaluation_cap",
            "exemplar_component_evaluation_cap",
            "exemplar_eye_evaluation_cap",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            if not 0 <= int(value) <= _UINT64_MAX:
                raise ValueError(f"{name} must fit uint64")
            setattr(self, name, int(value))

    def reserve_local(self, slot_charge: int, sample_charge: int) -> bool:
        """Atomically reserve a complete run or leave both counters unchanged."""

        _require_uint64("slot_charge", slot_charge)
        _require_uint64("sample_charge", sample_charge)
        next_slots = _checked_add(self.local_slots_consumed, slot_charge)
        next_samples = _checked_add(
            self.local_neighbor_samples_consumed,
            sample_charge,
        )
        if next_slots > self.local_slot_cap or next_samples > self.local_neighbor_sample_cap:
            self.local_budget_skipped_run_count += 1
            return False
        self.local_slots_consumed = next_slots
        self.local_neighbor_samples_consumed = next_samples
        return True


@dataclass(frozen=True)
class LocalStripPlanResult:
    """Planned local records plus physical-work diagnostics."""

    records: np.ndarray
    evaluated_slot_count: int
    physical_sample_read_count: int
    local_filled_run_count: int
    eligible_slot_count: int
    unsafe_donor_slot_count: int


@dataclass
class _LocalDiagnostics:
    evaluated_slots: int = 0
    physical_sample_reads: int = 0
    filled_runs: int = 0
    eligible_slots: int = 0
    unsafe_donor_slots: int = 0


def plan_local_strips(
    records: np.ndarray,
    analysis: QualityVisibilityAnalysis,
    *,
    render_shape: tuple[int, int],
    local_limit_px: int,
    budgets: QualityRepairBudgets,
) -> LocalStripPlanResult:
    """Plan complete repair runs with the frozen scalar strip enumeration."""

    height, width = _validate_inputs(
        records,
        analysis,
        render_shape,
        local_limit_px,
        budgets,
    )
    runs = reconstruct_repair_runs(records, render_shape=render_shape)
    output = np.array(records, copy=True, order="C")
    diagnostics = _LocalDiagnostics()
    search_limit = max(1, math.floor((64 * height) / 1080.0 + 0.5))
    safe_radius = max(1, math.floor(height / 1080.0 + 0.5))
    support = _checked_square(_checked_add(_checked_product(2, safe_radius), 1))

    for run in runs:
        start_fine = int(run["start_fine"])
        end_fine = int(run["end_fine"])
        lane_count = end_fine - start_fine + 1
        if lane_count > 16 * local_limit_px:
            continue
        first_pixel = start_fine // 16
        last_pixel = end_fine // 16
        touched_pixels = last_pixel - first_pixel + 1
        slot_charge = _checked_product(5, search_limit)
        per_slot = _checked_add(3, _checked_product(touched_pixels, support))
        sample_charge = _checked_add(3, _checked_product(slot_charge, per_slot))
        if not budgets.reserve_local(slot_charge, sample_charge):
            continue
        diagnostics.evaluated_slots = _checked_add(
            diagnostics.evaluated_slots,
            slot_charge,
        )
        if _plan_one_run(
            output,
            run,
            analysis,
            height=height,
            width=width,
            search_limit=search_limit,
            safe_radius=safe_radius,
            diagnostics=diagnostics,
        ):
            diagnostics.filled_runs += 1

    output.setflags(write=False)
    return LocalStripPlanResult(
        records=output,
        evaluated_slot_count=diagnostics.evaluated_slots,
        physical_sample_read_count=diagnostics.physical_sample_reads,
        local_filled_run_count=diagnostics.filled_runs,
        eligible_slot_count=diagnostics.eligible_slots,
        unsafe_donor_slot_count=diagnostics.unsafe_donor_slots,
    )


def _validate_inputs(
    records: np.ndarray,
    analysis: QualityVisibilityAnalysis,
    render_shape: tuple[int, int],
    local_limit_px: int,
    budgets: QualityRepairBudgets,
) -> tuple[int, int]:
    if not isinstance(analysis, QualityVisibilityAnalysis):
        raise TypeError("analysis must be QualityVisibilityAnalysis")
    if not isinstance(budgets, QualityRepairBudgets):
        raise TypeError("budgets must be QualityRepairBudgets")
    if isinstance(local_limit_px, (bool, np.bool_)) or not isinstance(
        local_limit_px, (int, np.integer)
    ):
        raise TypeError("local_limit_px must be an integer")
    if int(local_limit_px) <= 0:
        raise ValueError("local_limit_px must be positive")
    if (
        not isinstance(render_shape, tuple)
        or len(render_shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in render_shape)
    ):
        raise TypeError("render_shape must be an integer (height, width) tuple")
    height, width = render_shape
    if height <= 0 or width <= 0:
        raise ValueError("render_shape must be positive")
    if not isinstance(records, np.ndarray) or records.dtype != REPAIR_RECORD_DTYPE:
        raise TypeError("records must use REPAIR_RECORD_DTYPE")
    if records.ndim != 1 or not records.flags.c_contiguous:
        raise ValueError("records must be a C-contiguous vector")
    _validate_analysis(analysis, height, width)
    return height, width


def _validate_analysis(
    analysis: QualityVisibilityAnalysis,
    height: int,
    width: int,
) -> None:
    shape = (height, width)
    if (
        not isinstance(analysis.coverage_count, np.ndarray)
        or analysis.coverage_count.dtype != np.uint8
        or analysis.coverage_count.shape != shape
        or not analysis.coverage_count.flags.c_contiguous
    ):
        raise TypeError("coverage_count must be a matching C-contiguous uint8 raster")
    if (
        not isinstance(analysis.pure_region_id, np.ndarray)
        or analysis.pure_region_id.dtype != np.uint32
        or analysis.pure_region_id.shape != shape
        or not analysis.pure_region_id.flags.c_contiguous
    ):
        raise TypeError("pure_region_id must be a matching C-contiguous uint32 raster")
    if not isinstance(analysis.pure_bgr, tuple) or len(analysis.pure_bgr) != 3:
        raise TypeError("pure_bgr must contain independent B, G, and R planes")
    for plane in analysis.pure_bgr:
        if (
            not isinstance(plane, np.ndarray)
            or plane.dtype != np.uint8
            or plane.shape != shape
            or not plane.flags.c_contiguous
        ):
            raise TypeError("pure_bgr planes must be matching C-contiguous uint8 rasters")
    for left in range(3):
        for right in range(left + 1, 3):
            if np.shares_memory(analysis.pure_bgr[left], analysis.pure_bgr[right]):
                raise ValueError("pure_bgr planes must be independently owned")


def _plan_one_run(
    output: np.ndarray,
    run: np.void,
    analysis: QualityVisibilityAnalysis,
    *,
    height: int,
    width: int,
    search_limit: int,
    safe_radius: int,
    diagnostics: _LocalDiagnostics,
) -> bool:
    row = int(run["row"])
    start_fine = int(run["start_fine"])
    end_fine = int(run["end_fine"])
    region = int(run["region_id"])
    far_side = int(run["far_side"])
    direction = -1 if far_side == 0 else 1
    anchor_fine = start_fine - 1 if far_side == 0 else end_fine + 1
    boundary_column = anchor_fine // 16
    first_pixel = start_fine // 16
    last_pixel = end_fine // 16
    touched_pixel_count = last_pixel - first_pixel + 1
    context_columns = (
        boundary_column + 2 * direction,
        boundary_column + direction,
        boundary_column,
    )
    actual_context = _read_proxy_context(
        analysis,
        row,
        context_columns,
        region,
        height,
        width,
        diagnostics,
    )
    if actual_context is None:
        return False
    best_donors = _find_best_local_donors(
        actual_context,
        analysis,
        row=row,
        boundary_column=boundary_column,
        direction=direction,
        donor_count=touched_pixel_count,
        region=region,
        height=height,
        width=width,
        search_limit=search_limit,
        safe_radius=safe_radius,
        diagnostics=diagnostics,
    )
    if best_donors is None:
        return False
    _write_local_fill(output, run, analysis, best_donors, far_side)
    return True


def _find_best_local_donors(
    actual_context: _Context,
    analysis: QualityVisibilityAnalysis,
    *,
    row: int,
    boundary_column: int,
    direction: int,
    donor_count: int,
    region: int,
    height: int,
    width: int,
    search_limit: int,
    safe_radius: int,
    diagnostics: _LocalDiagnostics,
) -> _Donors | None:
    best: tuple[int, int, _Donors] | None = None
    for offset in range(1, search_limit + 1):
        for row_index, row_delta in enumerate(_ROW_DELTAS):
            ordinal = (offset - 1) * 5 + row_index
            candidate = _evaluate_local_slot(
                actual_context,
                analysis,
                candidate_row=row + row_delta,
                candidate_start=boundary_column + direction * offset,
                boundary_column=boundary_column,
                direction=direction,
                donor_count=donor_count,
                region=region,
                height=height,
                width=width,
                safe_radius=safe_radius,
                ordinal=ordinal,
                diagnostics=diagnostics,
            )
            if candidate is not None and (best is None or candidate[:2] < best[:2]):
                best = candidate
    return None if best is None else best[2]


def _evaluate_local_slot(
    actual_context: _Context,
    analysis: QualityVisibilityAnalysis,
    *,
    candidate_row: int,
    candidate_start: int,
    boundary_column: int,
    direction: int,
    donor_count: int,
    region: int,
    height: int,
    width: int,
    safe_radius: int,
    ordinal: int,
    diagnostics: _LocalDiagnostics,
) -> tuple[int, int, _Donors] | None:
    candidate_columns = (
        candidate_start + 3 * direction,
        candidate_start + 2 * direction,
        candidate_start + direction,
    )
    donor_columns = tuple(candidate_start - index * direction for index in range(donor_count))
    if not _slot_coordinates_are_legal(
        candidate_row,
        candidate_columns,
        donor_columns,
        boundary_column,
        direction,
        height,
        width,
    ):
        return None
    candidate_context = _read_proxy_context(
        analysis,
        candidate_row,
        candidate_columns,
        region,
        height,
        width,
        diagnostics,
    )
    if candidate_context is None:
        return None
    donors = tuple((candidate_row, column) for column in donor_columns)
    for donor_row, donor_column in donors:
        if _is_safe_donor(
            analysis,
            donor_row,
            donor_column,
            region,
            safe_radius,
            diagnostics,
        ):
            continue
        diagnostics.unsafe_donor_slots = _checked_add(
            diagnostics.unsafe_donor_slots,
            1,
        )
        return None
    diagnostics.eligible_slots = _checked_add(diagnostics.eligible_slots, 1)
    return _context_score(actual_context, candidate_context), ordinal, donors


def _write_local_fill(
    output: np.ndarray,
    run: np.void,
    analysis: QualityVisibilityAnalysis,
    best_donors: _Donors,
    far_side: int,
) -> None:
    record_start = int(run["record_start"])
    record_end = int(run["record_end"])
    record_indexes = range(record_start, record_end)
    if far_side == 1:
        record_indexes = range(record_end - 1, record_start - 1, -1)
    for record_index, donor in zip(
        record_indexes,
        best_donors,
        strict=True,
    ):
        donor_row, donor_column = donor
        for channel in range(3):
            output[record_index]["fill_bgr"][channel] = analysis.pure_bgr[channel][
                donor_row,
                donor_column,
            ]
        output[record_index]["backend"] = np.uint8(1)


def _read_proxy_context(
    analysis: QualityVisibilityAnalysis,
    row: int,
    columns: tuple[int, int, int],
    region: int,
    height: int,
    width: int,
    diagnostics: _LocalDiagnostics,
) -> _Context | None:
    if not 0 <= row < height or any(not 0 <= column < width for column in columns):
        return None
    for column in columns:
        diagnostics.physical_sample_reads += 1
        if (
            int(analysis.coverage_count[row, column]) == 0
            or int(analysis.pure_region_id[row, column]) != region
        ):
            return None
    return (
        _read_bgr(analysis, row, columns[0]),
        _read_bgr(analysis, row, columns[1]),
        _read_bgr(analysis, row, columns[2]),
    )


def _read_bgr(analysis: QualityVisibilityAnalysis, row: int, column: int) -> _Bgr:
    return (
        int(analysis.pure_bgr[0][row, column]),
        int(analysis.pure_bgr[1][row, column]),
        int(analysis.pure_bgr[2][row, column]),
    )


def _slot_coordinates_are_legal(
    row: int,
    context_columns: tuple[int, int, int],
    donor_columns: tuple[int, ...],
    boundary_column: int,
    direction: int,
    height: int,
    width: int,
) -> bool:
    if not 0 <= row < height:
        return False
    if any(not 0 <= column < width for column in context_columns):
        return False
    if any(not 0 <= column < width for column in donor_columns):
        return False
    return all((column - boundary_column) * direction > 0 for column in donor_columns)


def _is_safe_donor(
    analysis: QualityVisibilityAnalysis,
    row: int,
    column: int,
    region: int,
    radius: int,
    diagnostics: _LocalDiagnostics,
) -> bool:
    height, width = analysis.coverage_count.shape
    for sample_row in range(max(0, row - radius), min(height, row + radius + 1)):
        for sample_column in range(
            max(0, column - radius),
            min(width, column + radius + 1),
        ):
            diagnostics.physical_sample_reads += 1
            if (
                int(analysis.coverage_count[sample_row, sample_column]) != 16
                or int(analysis.pure_region_id[sample_row, sample_column]) != region
            ):
                return False
    return True


def _context_score(
    actual: tuple[tuple[int, int, int], ...],
    candidate: tuple[tuple[int, int, int], ...],
) -> int:
    bgr_l1 = 0
    actual_luma = [0, 0, 0]
    candidate_luma = [0, 0, 0]
    for index in range(3):
        for channel in range(3):
            bgr_l1 += abs(actual[index][channel] - candidate[index][channel])
        actual_luma[index] = _integer_luma(actual[index])
        candidate_luma[index] = _integer_luma(candidate[index])
    difference_l1 = 0
    for index in range(2):
        actual_delta = actual_luma[index + 1] - actual_luma[index]
        candidate_delta = candidate_luma[index + 1] - candidate_luma[index]
        difference_l1 += abs(actual_delta - candidate_delta)
    return 2 * bgr_l1 + difference_l1


def _integer_luma(bgr: tuple[int, int, int]) -> int:
    return (29 * bgr[0] + 150 * bgr[1] + 77 * bgr[2] + 128) >> 8


def _require_uint64(name: str, value: int) -> None:
    if not 0 <= value <= _UINT64_MAX:
        raise QualityRepairBudgetError(f"{name} exceeds uint64")


def _checked_add(left: int, right: int) -> int:
    result = left + right
    _require_uint64("checked sum", result)
    return result


def _checked_product(left: int, right: int) -> int:
    result = left * right
    _require_uint64("checked product", result)
    return result


def _checked_square(value: int) -> int:
    return _checked_product(value, value)
