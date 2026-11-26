"""Canonical per-frame Quality coverage statistics."""

from __future__ import annotations

import math
from collections import Counter
from numbers import Integral
from typing import Any, Callable, Iterable

import numpy as np

from ...io.streamed_canonical_json import ReplayableJsonArray

from ..forward_splat import HORIZONTAL_SUBPIXELS
from .types import (
    CompactStereoRenderResult,
    DenseHistogramPairs,
    FastEyeDiagnostics,
    HoleRunDiagnostics,
    QualityRepairPlanSummary,
)


_STATE_BITS = {
    "prefill_partial": 0x01,
    "prefill_full": 0x02,
    "local_filled": 0x04,
    "post_local_residual": 0x08,
    "exemplar_filled": 0x10,
    "fallback_filled": 0x20,
    "final_unresolved": 0x40,
}
_U64_MAX = 2**64 - 1


class DiagnosticsCardinalityError(ValueError):
    """A persisted diagnostics integer exceeded the frozen uint64 domain."""

    def __init__(self, field: str, operands: tuple[object, ...]) -> None:
        self.field = field
        self.operands = operands
        super().__init__(f"diagnostics uint64 overflow for {field}: {operands!r}")


def _u64_operand(field: str, value: object) -> int:
    if type(value) is int:
        normalized = value
    else:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise DiagnosticsCardinalityError(field, (value,))
        normalized = int(value)
    if normalized < 0 or normalized > _U64_MAX:
        raise DiagnosticsCardinalityError(field, (value,))
    return normalized


def checked_u64_add(field: str, *operands: object) -> int:
    """Add nonnegative integer operands without entering Python-bigint output."""

    total = 0
    normalized: list[int] = []
    for operand in operands:
        value = _u64_operand(field, operand)
        normalized.append(value)
        if value > _U64_MAX - total:
            raise DiagnosticsCardinalityError(field, tuple(normalized))
        total += value
    return total


def checked_u64_mul(field: str, *operands: object) -> int:
    """Multiply nonnegative integer operands in frozen left-to-right order."""

    total = 1
    normalized: list[int] = []
    for operand in operands:
        value = _u64_operand(field, operand)
        normalized.append(value)
        if value and total > _U64_MAX // value:
            raise DiagnosticsCardinalityError(field, tuple(normalized))
        total *= value
    return total


def validate_diagnostics_cardinality_dimensions(
    *,
    frame_count: object,
    height: object,
    width: object,
) -> int:
    """Prove the fixed diagnostics dimensions and their root counter bounds."""

    count = _u64_operand("frame_count", frame_count)
    normalized_height = _u64_operand("render_shape.height", height)
    normalized_width = _u64_operand("render_shape.width", width)
    if count == 0 or normalized_height == 0 or normalized_width == 0:
        raise DiagnosticsCardinalityError(
            "diagnostics_dimensions",
            (count, normalized_height, normalized_width),
        )
    pixels = checked_u64_mul(
        "render_shape.pixel_count",
        normalized_height,
        normalized_width,
    )
    lane_count = checked_u64_mul("frame_lane_count", 16, pixels)
    checked_u64_mul("root_pixel_count", count, pixels)
    checked_u64_mul("root_lane_count", count, lane_count)
    return pixels


def _histogram_cumulative(histogram: np.ndarray) -> np.ndarray:
    values = np.asarray(histogram)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise TypeError("histogram must be a one-dimensional integer array")
    if np.issubdtype(values.dtype, np.signedinteger) and np.any(values < 0):
        raise DiagnosticsCardinalityError("histogram_total", (int(values.min()),))
    normalized = values.astype(np.uint64, copy=False)
    cumulative = np.cumsum(normalized, dtype=np.uint64)
    if cumulative.size > 1 and np.any(cumulative[1:] < cumulative[:-1]):
        raise DiagnosticsCardinalityError(
            "histogram_total",
            (int(normalized.max()), int(normalized.size)),
        )
    return cumulative


def _histogram_metrics(
    histogram: np.ndarray,
    percentile: float,
) -> tuple[int, int, float]:
    cumulative = _histogram_cumulative(histogram)
    if cumulative.size == 0 or int(cumulative[-1]) == 0:
        return 0, 0, 0.0
    total = int(cumulative[-1])
    values = np.asarray(histogram)
    maximum = int(values.size - 1 - np.argmax(values[::-1] != 0))
    position = percentile * (total - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    lower = int(np.searchsorted(cumulative, np.uint64(lower_index), side="right"))
    upper = int(np.searchsorted(cumulative, np.uint64(upper_index), side="right"))
    interpolated = float(lower) + (float(upper) - float(lower)) * (position - lower_index)
    return total, maximum, interpolated


def _histogram_total(histogram: np.ndarray) -> int:
    cumulative = _histogram_cumulative(histogram)
    return 0 if cumulative.size == 0 else int(cumulative[-1])


def _histogram_max(histogram: np.ndarray) -> int:
    values = np.asarray(histogram)
    if not values.size or not np.any(values):
        return 0
    return int(values.size - 1 - np.argmax(values[::-1] != 0))


def _linear_percentile_from_histogram(histogram: np.ndarray, percentile: float) -> float:
    return _histogram_metrics(histogram, percentile)[2]


def _prefill_run_histograms(
    plan: QualityRepairPlanSummary | None,
    width: int,
    streamed: HoleRunDiagnostics | None,
) -> tuple[np.ndarray, np.ndarray]:
    del plan
    lane_histogram = np.zeros(16 * (width + 1), dtype=np.uint64)
    span_histogram = np.zeros(width + 1, dtype=np.uint64)
    if streamed is not None:
        if (
            streamed.lane_count_histogram.shape != lane_histogram.shape
            or streamed.touched_pixel_span_histogram.shape != span_histogram.shape
        ):
            raise ValueError("streamed hole-run histogram width changed")
        lane_histogram[:] = streamed.lane_count_histogram
        span_histogram[:] = streamed.touched_pixel_span_histogram
    return lane_histogram, span_histogram


def _histogram_pairs(histogram: np.ndarray) -> DenseHistogramPairs:
    return DenseHistogramPairs(histogram)


def _backend_counts(
    plan: QualityRepairPlanSummary | None,
) -> tuple[dict[str, int], dict[str, int], dict[str, Any]]:
    ordered = ("local", "exemplar", "fallback")
    if isinstance(plan, QualityRepairPlanSummary):
        lane_counts = dict(zip(ordered, plan.backend_lane_counts, strict=True))
        pixel_counts = dict(zip(ordered, plan.backend_pixel_counts, strict=True))
        component_count_payload: dict[str, Any] = {
            "availability": "quality_segment_graph",
            **dict(zip(ordered, plan.backend_component_counts, strict=True)),
        }
        return lane_counts, pixel_counts, component_count_payload
    lane_counts = {name: 0 for name in ordered}
    pixel_counts = {name: 0 for name in ordered}
    component_count_payload = {
        "availability": "unavailable_quality_none_no_repair_graph",
        "local": None,
        "exemplar": None,
        "fallback": None,
    }
    return lane_counts, pixel_counts, component_count_payload


def _repair_state_counts(repair_bits: np.ndarray) -> dict[str, int]:
    # Match the native dtype identity check; equivalent metadata dtypes stay on NumPy.
    if (
        type(repair_bits) is np.ndarray
        and repair_bits.dtype is np.dtype(np.uint8)
        and repair_bits.ndim == 2
        and repair_bits.flags.c_contiguous
    ):
        from . import native_kd

        native = native_kd._require_native() if native_kd.native_available() else None
        counter = getattr(native, "count_repair_state_bits", None)
        if counter is not None:
            counts = counter(repair_bits)
            return {name: int(counts[bit.bit_length() - 1]) for name, bit in _STATE_BITS.items()}
    return {
        name: int(np.count_nonzero(repair_bits & np.uint8(bit)))
        for name, bit in _STATE_BITS.items()
    }


def _eye_stats(
    coverage: np.ndarray,
    repair_bits: np.ndarray,
    plan: QualityRepairPlanSummary | None,
    *,
    quality_limits: dict[str, int | float] | None,
    streamed_hole_runs: HoleRunDiagnostics | None,
    render_mode: str,
    fast_diagnostics: FastEyeDiagnostics | None,
) -> dict[str, Any]:
    pixel_count = int(coverage.size)
    state_counts = _repair_state_counts(repair_bits)
    state_ratios = {
        name: (float(count) / float(pixel_count) if pixel_count else 0.0)
        for name, count in state_counts.items()
    }
    lane_histogram, span_histogram = _prefill_run_histograms(
        plan,
        coverage.shape[1],
        streamed_hole_runs,
    )
    lane_count, lane_max, lane_p95 = _histogram_metrics(lane_histogram, 0.95)
    _span_count, span_max, span_p95 = _histogram_metrics(span_histogram, 0.95)
    quality_budgets: dict[str, Any] | None
    if render_mode == "fast":
        if fast_diagnostics is None:
            raise ValueError("Fast diagnostics require producer-attested counters")
        backend_lanes = {
            "local": int(fast_diagnostics.local_filled_lane_count),
            "exemplar": 0,
            "fallback": 0,
        }
        backend_pixels = {
            "local": state_counts["local_filled"],
            "exemplar": 0,
            "fallback": 0,
        }
        backend_components: dict[str, Any] = {
            "availability": "unavailable_fast_no_segment_graph",
            "local": None,
            "exemplar": None,
            "fallback": None,
        }
        quality_budgets = None
        final_unresolved_lane_count = int(fast_diagnostics.final_unresolved_lane_count)
    else:
        backend_lanes, backend_pixels, backend_components = _backend_counts(plan)
        final_unresolved_lane_count = (
            int(np.sum(HORIZONTAL_SUBPIXELS - coverage, dtype=np.uint64)) if plan is None else 0
        )
    if render_mode == "quality" and plan is None:
        quality_budgets = {
            "availability": "unavailable_quality_none_no_repair_plan",
            "segment_record_count": None,
            "segment_table_bytes": None,
            "local_slot_charge": None,
            "local_neighborhood_sample_charge": None,
            "local_budget_skipped_run_count": None,
            "exemplar_evaluations": None,
            "fallback_indexed_donor_count": None,
            "fallback_query_count": None,
            "fallback_visited_nodes_total": None,
            "fallback_visited_nodes_max": None,
            "fallback_visited_nodes_p95": None,
        }
    elif render_mode == "quality":
        assert plan is not None
        segment_record_count = plan.segment_record_count
        segment_table_bytes = plan.segment_table_bytes
        quality_budgets = {
            "availability": "quality_repair_plan",
            "segment_record_count": segment_record_count,
            "segment_table_bytes": segment_table_bytes,
            "local_slot_charge": int(plan.local_evaluated_slot_count),
            "local_neighborhood_sample_charge": int(plan.local_physical_sample_read_count),
            "local_budget_skipped_run_count": int(plan.local_budget_skipped_run_count),
            "exemplar_evaluations": int(plan.exemplar_donor_evaluation_count),
            "fallback_indexed_donor_count": int(plan.fallback_indexed_donor_count),
            "fallback_query_count": int(plan.fallback_query_count),
            "fallback_visited_nodes_total": int(plan.fallback_visited_nodes_total),
            "fallback_visited_nodes_max": int(plan.fallback_visited_nodes_max),
            "fallback_visited_nodes_p95": float(plan.fallback_visited_nodes_p95),
        }
    return {
        "pixel_count": pixel_count,
        "state_pixel_counts": state_counts,
        "state_pixel_ratios": state_ratios,
        "hole_runs": {
            "count": lane_count,
            "lane_count_histogram": _histogram_pairs(lane_histogram),
            "touched_pixel_span_histogram": _histogram_pairs(span_histogram),
            "lane_count_max": lane_max,
            "lane_count_p95": lane_p95,
            "touched_pixel_span_max": span_max,
            "touched_pixel_span_p95": span_p95,
            "physical_width_px_max": float(lane_max) / 16.0,
            "physical_width_px_p95": lane_p95 / 16.0,
        },
        "backend_lane_counts": backend_lanes,
        "backend_pixel_counts": backend_pixels,
        "backend_component_counts": backend_components,
        "final_unresolved_lane_count": final_unresolved_lane_count,
        "quality_limits": quality_limits,
        "quality_budgets": quality_budgets,
    }


def _metric_stats(value: Any | None) -> dict[str, int | float] | None:
    if value is None:
        return None
    return {
        "valid_pixel_count": int(value.valid_pixel_count),
        "clamped_pixel_count": int(value.clamped_pixel_count),
        "clamped_fraction": float(value.clamped_fraction),
    }


def build_quality_frame_stats(
    compact: CompactStereoRenderResult,
    *,
    frame_name: str,
    geometry_mode: str,
    occlusion_fill: str,
    render_mode: str = "quality",
) -> dict[str, Any]:
    """Build the closed frame statistics object from compact state."""

    validate_diagnostics_cardinality_dimensions(
        frame_count=1,
        height=int(compact.left_image.shape[0]),
        width=int(compact.left_image.shape[1]),
    )

    if geometry_mode not in {"relative", "metric"}:
        raise ValueError("diagnostic geometry_mode must be relative or metric")
    limits = getattr(compact, "quality_limits", None)
    left_limits = limits.get("left") if isinstance(limits, dict) else None
    right_limits = limits.get("right") if isinstance(limits, dict) else None
    if render_mode not in {"fast", "quality"}:
        raise ValueError("diagnostic render_mode must be fast or quality")
    geometry_nearest = getattr(compact, "geometry_nearest", None)
    if render_mode == "quality" and geometry_nearest is None:
        geometry_nearest = {
            "indexed_sample_count": 0,
            "query_count": 0,
            "visited_nodes_total": 0,
            "visited_nodes_max": 0,
        }
    return {
        "schema_version": 1,
        "frame_name": frame_name,
        "render_mode": render_mode,
        "geometry_mode": geometry_mode,
        "occlusion_fill": occlusion_fill,
        "render_shape": [int(compact.left_image.shape[0]), int(compact.left_image.shape[1])],
        "eyes": {
            "left": _eye_stats(
                compact.left_coverage_count,
                compact.left_repair_bits,
                compact.left_plan,
                quality_limits=left_limits,
                streamed_hole_runs=compact.left_hole_runs,
                render_mode=render_mode,
                fast_diagnostics=compact.left_fast_diagnostics,
            ),
            "right": _eye_stats(
                compact.right_coverage_count,
                compact.right_repair_bits,
                compact.right_plan,
                quality_limits=right_limits,
                streamed_hole_runs=compact.right_hole_runs,
                render_mode=render_mode,
                fast_diagnostics=compact.right_fast_diagnostics,
            ),
        },
        "metric_projection": _metric_stats(compact.metric_stats),
        "geometry_nearest": geometry_nearest,
    }


def _merge_histogram_pairs(
    eyes: list[dict[str, Any]],
    field: str,
) -> np.ndarray:
    maximum = 0
    for eye in eyes:
        pairs = eye["hole_runs"][field]
        if isinstance(pairs, DenseHistogramPairs):
            maximum = max(maximum, int(pairs.bins.size) - 1)
        else:
            for value, _count in pairs:
                maximum = max(maximum, int(value))
    merged = np.zeros(maximum + 1, dtype=np.uint64)
    for eye in eyes:
        pairs = eye["hole_runs"][field]
        source = enumerate(pairs.bins) if isinstance(pairs, DenseHistogramPairs) else pairs
        for value, count in source:
            index = int(value)
            merged[index] = np.uint64(checked_u64_add(field, int(merged[index]), int(count)))
    return merged


def _add_histogram_pairs(
    target: np.ndarray,
    pairs: DenseHistogramPairs | Iterable[tuple[int, int]] | list[list[int]],
) -> None:
    if isinstance(pairs, DenseHistogramPairs):
        if pairs.bins.shape != target.shape:
            raise ValueError("dense histogram width changed between frames")
        source: Iterable[Iterable[int]] = enumerate(pairs.bins)
    else:
        source = pairs
    for bucket, count in source:
        index = int(bucket)
        if index < 0 or index >= target.size:
            raise ValueError("histogram bucket is outside the render width")
        target[index] = np.uint64(
            checked_u64_add("diagnostic_histogram_bin", int(target[index]), int(count))
        )


def _sum_named_counts(
    values: list[dict[str, Any]],
    field: str,
    names: tuple[str, ...],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for name in names:
        total = 0
        for value in values:
            total = checked_u64_add(f"{field}.{name}", total, value[field][name])
        result[name] = total
    return result


def _aggregate_eye(eyes: list[dict[str, Any]], render_mode: str) -> dict[str, Any]:
    state_names = tuple(_STATE_BITS)
    backend_names = ("local", "exemplar", "fallback")
    pixel_count = 0
    for eye in eyes:
        pixel_count = checked_u64_add("eye.pixel_count", pixel_count, eye["pixel_count"])
    state_counts = _sum_named_counts(eyes, "state_pixel_counts", state_names)
    lane_histogram = _merge_histogram_pairs(eyes, "lane_count_histogram")
    span_histogram = _merge_histogram_pairs(eyes, "touched_pixel_span_histogram")
    lane_p95 = _linear_percentile_from_histogram(lane_histogram, 0.95)
    component_availability = {eye["backend_component_counts"]["availability"] for eye in eyes}
    if len(component_availability) != 1:
        raise ValueError("backend component availability changed between frames")
    availability = next(iter(component_availability))
    if availability == "quality_segment_graph":
        component_counts: dict[str, Any] = {
            "availability": availability,
            **_sum_named_counts(eyes, "backend_component_counts", backend_names),
        }
    else:
        component_counts = {
            "availability": availability,
            **{name: None for name in backend_names},
        }

    limits = [eye["quality_limits"] for eye in eyes]
    if all(value is None for value in limits):
        limit_ranges = None
    elif any(value is None for value in limits):
        raise ValueError("quality limit availability changed between frames")
    else:
        concrete_limits = [value for value in limits if value is not None]
        limit_ranges = {
            "max_neighbour_abs_q_jump_px_min": float(
                min(value["max_neighbour_abs_q_jump_px"] for value in concrete_limits)
            ),
            "max_neighbour_abs_q_jump_px_max": float(
                max(value["max_neighbour_abs_q_jump_px"] for value in concrete_limits)
            ),
            "predicted_gap_px_min": int(
                min(value["predicted_gap_px"] for value in concrete_limits)
            ),
            "predicted_gap_px_max": int(
                max(value["predicted_gap_px"] for value in concrete_limits)
            ),
            "local_limit_px_min": int(min(value["local_limit_px"] for value in concrete_limits)),
            "local_limit_px_max": int(max(value["local_limit_px"] for value in concrete_limits)),
        }

    budgets = [eye["quality_budgets"] for eye in eyes]
    budget_totals: dict[str, Any] | None
    if render_mode == "fast":
        budget_totals = None
    elif all(
        value["availability"] == "unavailable_quality_none_no_repair_plan" for value in budgets
    ):
        budget_totals = {
            "availability": "unavailable_quality_none_no_repair_plan",
            "segment_record_count": None,
            "segment_table_bytes_max": None,
            "local_slot_charge": None,
            "local_neighborhood_sample_charge": None,
            "local_budget_skipped_run_count": None,
            "exemplar_evaluations": None,
            "fallback_indexed_donor_count": None,
            "fallback_query_count": None,
            "fallback_visited_nodes_total": None,
            "fallback_visited_nodes_max": None,
            "fallback_visited_nodes_frame_p95_max": None,
        }
    elif all(value["availability"] == "quality_repair_plan" for value in budgets):
        budget_totals = {
            "availability": "quality_repair_plan",
            "segment_record_count": checked_u64_add(
                "quality_budget_totals.segment_record_count",
                *(value["segment_record_count"] for value in budgets),
            ),
            "segment_table_bytes_max": max(int(value["segment_table_bytes"]) for value in budgets),
            "local_slot_charge": checked_u64_add(
                "quality_budget_totals.local_slot_charge",
                *(value["local_slot_charge"] for value in budgets),
            ),
            "local_neighborhood_sample_charge": checked_u64_add(
                "quality_budget_totals.local_neighborhood_sample_charge",
                *(value["local_neighborhood_sample_charge"] for value in budgets),
            ),
            "local_budget_skipped_run_count": checked_u64_add(
                "quality_budget_totals.local_budget_skipped_run_count",
                *(value["local_budget_skipped_run_count"] for value in budgets),
            ),
            "exemplar_evaluations": checked_u64_add(
                "quality_budget_totals.exemplar_evaluations",
                *(value["exemplar_evaluations"] for value in budgets),
            ),
            "fallback_indexed_donor_count": checked_u64_add(
                "quality_budget_totals.fallback_indexed_donor_count",
                *(value["fallback_indexed_donor_count"] for value in budgets),
            ),
            "fallback_query_count": checked_u64_add(
                "quality_budget_totals.fallback_query_count",
                *(value["fallback_query_count"] for value in budgets),
            ),
            "fallback_visited_nodes_total": checked_u64_add(
                "quality_budget_totals.fallback_visited_nodes_total",
                *(value["fallback_visited_nodes_total"] for value in budgets),
            ),
            "fallback_visited_nodes_max": max(
                int(value["fallback_visited_nodes_max"]) for value in budgets
            ),
            "fallback_visited_nodes_frame_p95_max": float(
                max(value["fallback_visited_nodes_p95"] for value in budgets)
            ),
        }
    else:
        raise ValueError("quality budget availability changed between frames")

    return {
        "pixel_count": pixel_count,
        "state_pixel_counts": state_counts,
        "state_pixel_ratios": {
            name: float(state_counts[name]) / float(pixel_count) for name in state_names
        },
        "hole_runs": {
            "count": _histogram_total(lane_histogram),
            "lane_count_histogram": _histogram_pairs(lane_histogram),
            "touched_pixel_span_histogram": _histogram_pairs(span_histogram),
            "lane_count_max": _histogram_max(lane_histogram),
            "lane_count_p95": lane_p95,
            "touched_pixel_span_max": _histogram_max(span_histogram),
            "touched_pixel_span_p95": _linear_percentile_from_histogram(span_histogram, 0.95),
            "physical_width_px_max": float(_histogram_max(lane_histogram)) / 16.0,
            "physical_width_px_p95": lane_p95 / 16.0,
        },
        "backend_lane_counts": _sum_named_counts(eyes, "backend_lane_counts", backend_names),
        "backend_pixel_counts": _sum_named_counts(eyes, "backend_pixel_counts", backend_names),
        "backend_component_counts": component_counts,
        "final_unresolved_lane_count": checked_u64_add(
            "eye.final_unresolved_lane_count",
            *(eye["final_unresolved_lane_count"] for eye in eyes),
        ),
        "quality_limit_ranges": limit_ranges,
        "quality_budget_totals": budget_totals,
    }


def build_diagnostics_summary(  # noqa: C901
    frame_stats: Iterable[dict[str, Any]],
    *,
    frame_names: tuple[str, ...] | None = None,
    frame_name_replay: Callable[[], Iterable[str]] | None = None,
    frame_count: int | None = None,
    clamped_fraction_replay: Callable[[], Iterable[float]] | None = None,
    render_shape: tuple[int, int],
    render_mode: str,
    geometry_mode: str,
    occlusion_fill: str,
    ordered_frame_manifest_fingerprint: str,
    frames_jsonl_sha256: str,
) -> dict[str, Any]:
    """Aggregate validated source-ordered frame statistics."""

    if frame_name_replay is None:
        if not frame_names:
            raise ValueError("diagnostics frame names must be nonempty")
        expected_frame_count = len(frame_names)

        def replay_frame_names() -> Iterable[str]:
            return iter(frame_names)

        frame_name_replay = replay_frame_names
        summary_frame_names: object = list(frame_names)
    else:
        if frame_names is not None:
            raise ValueError("diagnostics frame names require exactly one replay source")
        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
            raise ValueError("streamed diagnostics frame count must be positive")
        expected_frame_count = frame_count
        summary_frame_names = ReplayableJsonArray(frame_name_replay)
    validate_diagnostics_cardinality_dimensions(
        frame_count=expected_frame_count,
        height=render_shape[0],
        width=render_shape[1],
    )
    expected_names = iter(frame_name_replay())

    state_names = tuple(_STATE_BITS)
    backend_names = ("local", "exemplar", "fallback")

    def eye_accumulator() -> dict[str, Any]:
        return {
            "pixel_count": 0,
            "state_counts": Counter(),
            "lane_histogram": np.zeros(
                16 * (int(render_shape[1]) + 1),
                dtype=np.uint64,
            ),
            "span_histogram": np.zeros(int(render_shape[1]) + 1, dtype=np.uint64),
            "backend_lane_counts": Counter(),
            "backend_pixel_counts": Counter(),
            "component_availability": None,
            "component_counts": Counter(),
            "final_unresolved_lane_count": 0,
            "limit_mode": None,
            "limit_ranges": {},
            "budget_availability": None,
            "budget_sums": Counter(),
            "budget_maxima": {},
        }

    eye_accumulators = {name: eye_accumulator() for name in ("left", "right")}
    valid_count = 0
    clamped_count = 0
    clamped_fractions: list[float] | None = [] if clamped_fraction_replay is None else None
    clamped_fraction_sum = 0.0
    clamped_fraction_max = 0.0
    affected_frame_count = 0
    nearest = {
        "indexed_sample_count": 0,
        "query_count": 0,
        "visited_nodes_total": 0,
        "visited_nodes_max": 0,
    }
    frame_count = 0

    for value in frame_stats:
        try:
            expected_frame_name = next(expected_names)
        except StopIteration as error:
            raise ValueError("frame statistics exceed the source sequence") from error
        if value["frame_name"] != expected_frame_name:
            raise ValueError("frame statistics are not in exact source order")
        frame_count = checked_u64_add("frame_count", frame_count, 1)
        for eye_name in ("left", "right"):
            eye = value["eyes"][eye_name]
            accumulator = eye_accumulators[eye_name]
            accumulator["pixel_count"] = checked_u64_add(
                f"{eye_name}.pixel_count",
                accumulator["pixel_count"],
                eye["pixel_count"],
            )
            for name in state_names:
                accumulator["state_counts"][name] = checked_u64_add(
                    f"{eye_name}.state_pixel_counts.{name}",
                    accumulator["state_counts"][name],
                    eye["state_pixel_counts"][name],
                )
            for histogram_name, target_name in (
                ("lane_count_histogram", "lane_histogram"),
                ("touched_pixel_span_histogram", "span_histogram"),
            ):
                _add_histogram_pairs(
                    accumulator[target_name],
                    eye["hole_runs"][histogram_name],
                )
            for name in backend_names:
                accumulator["backend_lane_counts"][name] = checked_u64_add(
                    f"{eye_name}.backend_lane_counts.{name}",
                    accumulator["backend_lane_counts"][name],
                    eye["backend_lane_counts"][name],
                )
                accumulator["backend_pixel_counts"][name] = checked_u64_add(
                    f"{eye_name}.backend_pixel_counts.{name}",
                    accumulator["backend_pixel_counts"][name],
                    eye["backend_pixel_counts"][name],
                )
            components = eye["backend_component_counts"]
            availability = components["availability"]
            previous_availability = accumulator["component_availability"]
            if previous_availability not in {None, availability}:
                raise ValueError("backend component availability changed between frames")
            accumulator["component_availability"] = availability
            if availability == "quality_segment_graph":
                for name in backend_names:
                    accumulator["component_counts"][name] = checked_u64_add(
                        f"{eye_name}.backend_component_counts.{name}",
                        accumulator["component_counts"][name],
                        components[name],
                    )
            accumulator["final_unresolved_lane_count"] = checked_u64_add(
                f"{eye_name}.final_unresolved_lane_count",
                accumulator["final_unresolved_lane_count"],
                eye["final_unresolved_lane_count"],
            )

            limits = eye["quality_limits"]
            limit_mode = "none" if limits is None else "value"
            if accumulator["limit_mode"] not in {None, limit_mode}:
                raise ValueError("quality limit availability changed between frames")
            accumulator["limit_mode"] = limit_mode
            if limits is not None:
                ranges = accumulator["limit_ranges"]
                for field, target_prefix, convert in (
                    ("max_neighbour_abs_q_jump_px", "max_neighbour_abs_q_jump_px", float),
                    ("predicted_gap_px", "predicted_gap_px", int),
                    ("local_limit_px", "local_limit_px", int),
                ):
                    current = convert(limits[field])
                    minimum = f"{target_prefix}_min"
                    maximum = f"{target_prefix}_max"
                    ranges[minimum] = min(ranges.get(minimum, current), current)
                    ranges[maximum] = max(ranges.get(maximum, current), current)

            if render_mode != "fast":
                budgets = eye["quality_budgets"]
                budget_availability = budgets["availability"]
                if accumulator["budget_availability"] not in {
                    None,
                    budget_availability,
                }:
                    raise ValueError("quality budget availability changed between frames")
                accumulator["budget_availability"] = budget_availability
                if budget_availability == "quality_repair_plan":
                    for field in (
                        "segment_record_count",
                        "local_slot_charge",
                        "local_neighborhood_sample_charge",
                        "local_budget_skipped_run_count",
                        "exemplar_evaluations",
                        "fallback_indexed_donor_count",
                        "fallback_query_count",
                        "fallback_visited_nodes_total",
                    ):
                        accumulator["budget_sums"][field] = checked_u64_add(
                            f"{eye_name}.quality_budgets.{field}",
                            accumulator["budget_sums"][field],
                            budgets[field],
                        )
                    for field in (
                        "segment_table_bytes",
                        "fallback_visited_nodes_max",
                        "fallback_visited_nodes_p95",
                    ):
                        current = float(budgets[field])
                        accumulator["budget_maxima"][field] = max(
                            accumulator["budget_maxima"].get(field, current),
                            current,
                        )

        projection = value["metric_projection"]
        if geometry_mode == "metric":
            if projection is None:
                raise ValueError("metric diagnostics require projection statistics")
            valid_count = checked_u64_add(
                "metric_projection.valid_pixel_count",
                valid_count,
                projection["valid_pixel_count"],
            )
            clamped_count = checked_u64_add(
                "metric_projection.clamped_pixel_count",
                clamped_count,
                projection["clamped_pixel_count"],
            )
            fraction = float(projection["clamped_fraction"])
            if clamped_fractions is not None:
                clamped_fractions.append(fraction)
            clamped_fraction_sum += fraction
            clamped_fraction_max = max(clamped_fraction_max, fraction)
            affected_frame_count = checked_u64_add(
                "metric_projection.affected_frame_count",
                affected_frame_count,
                int(fraction > 0.0),
            )
        elif projection is not None:
            raise ValueError("relative diagnostics cannot contain metric statistics")

        nearest_value = value["geometry_nearest"]
        if render_mode == "quality":
            if nearest_value is None:
                raise ValueError("Quality diagnostics require geometry-nearest statistics")
            for field in ("indexed_sample_count", "query_count", "visited_nodes_total"):
                nearest[field] = checked_u64_add(
                    f"geometry_nearest.{field}",
                    nearest[field],
                    nearest_value[field],
                )
            nearest["visited_nodes_max"] = max(
                nearest["visited_nodes_max"],
                int(nearest_value["visited_nodes_max"]),
            )
        del value

    try:
        next(expected_names)
    except StopIteration:
        pass
    else:
        raise ValueError("frame statistics ended before the source sequence")
    if frame_count != expected_frame_count:
        raise ValueError("frame statistics ended before the source sequence")

    eyes: dict[str, Any] = {}
    for eye_name, accumulator in eye_accumulators.items():
        pixel_count = int(accumulator["pixel_count"])
        lane_histogram = accumulator["lane_histogram"]
        span_histogram = accumulator["span_histogram"]
        lane_p95 = _linear_percentile_from_histogram(lane_histogram, 0.95)
        availability = accumulator["component_availability"]
        if availability == "quality_segment_graph":
            component_counts = {
                "availability": availability,
                **{name: int(accumulator["component_counts"][name]) for name in backend_names},
            }
        else:
            component_counts = {
                "availability": availability,
                **{name: None for name in backend_names},
            }
        budget_totals: dict[str, Any] | None
        if render_mode == "fast":
            budget_totals = None
        elif accumulator["budget_availability"] == "unavailable_quality_none_no_repair_plan":
            budget_totals = {
                "availability": "unavailable_quality_none_no_repair_plan",
                "segment_record_count": None,
                "segment_table_bytes_max": None,
                "local_slot_charge": None,
                "local_neighborhood_sample_charge": None,
                "local_budget_skipped_run_count": None,
                "exemplar_evaluations": None,
                "fallback_indexed_donor_count": None,
                "fallback_query_count": None,
                "fallback_visited_nodes_total": None,
                "fallback_visited_nodes_max": None,
                "fallback_visited_nodes_frame_p95_max": None,
            }
        elif accumulator["budget_availability"] == "quality_repair_plan":
            sums = accumulator["budget_sums"]
            maxima = accumulator["budget_maxima"]
            budget_totals = {
                "availability": "quality_repair_plan",
                **{field: int(sums[field]) for field in sums},
                "segment_table_bytes_max": int(maxima["segment_table_bytes"]),
                "fallback_visited_nodes_max": int(maxima["fallback_visited_nodes_max"]),
                "fallback_visited_nodes_frame_p95_max": float(maxima["fallback_visited_nodes_p95"]),
            }
        else:
            raise ValueError("quality budget availability changed between frames")
        state_counts = {name: int(accumulator["state_counts"][name]) for name in state_names}
        eyes[eye_name] = {
            "pixel_count": pixel_count,
            "state_pixel_counts": state_counts,
            "state_pixel_ratios": {
                name: float(state_counts[name]) / float(pixel_count) for name in state_names
            },
            "hole_runs": {
                "count": _histogram_total(lane_histogram),
                "lane_count_histogram": _histogram_pairs(lane_histogram),
                "touched_pixel_span_histogram": _histogram_pairs(span_histogram),
                "lane_count_max": _histogram_max(lane_histogram),
                "lane_count_p95": lane_p95,
                "touched_pixel_span_max": _histogram_max(span_histogram),
                "touched_pixel_span_p95": _linear_percentile_from_histogram(span_histogram, 0.95),
                "physical_width_px_max": float(_histogram_max(lane_histogram)) / 16.0,
                "physical_width_px_p95": lane_p95 / 16.0,
            },
            "backend_lane_counts": {
                name: int(accumulator["backend_lane_counts"][name]) for name in backend_names
            },
            "backend_pixel_counts": {
                name: int(accumulator["backend_pixel_counts"][name]) for name in backend_names
            },
            "backend_component_counts": component_counts,
            "final_unresolved_lane_count": int(accumulator["final_unresolved_lane_count"]),
            "quality_limit_ranges": (
                dict(accumulator["limit_ranges"]) if accumulator["limit_mode"] == "value" else None
            ),
            "quality_budget_totals": budget_totals,
        }

    if geometry_mode == "metric":
        streamed_fractions: object
        if clamped_fractions is None:
            assert clamped_fraction_replay is not None
            streamed_fractions = ReplayableJsonArray(clamped_fraction_replay)
        else:
            streamed_fractions = clamped_fractions
        metric_projection = {
            "valid_pixel_count": valid_count,
            "clamped_pixel_count": clamped_count,
            "clamped_fraction": (float(clamped_count) / float(valid_count) if valid_count else 0.0),
            "clamped_fractions": streamed_fractions,
            "affected_frame_count": affected_frame_count,
            "mean_clamped_fraction": (
                clamped_fraction_sum / float(expected_frame_count) if expected_frame_count else 0.0
            ),
            "max_clamped_fraction": clamped_fraction_max,
        }
    else:
        metric_projection = None
    geometry_nearest = nearest if render_mode == "quality" else None
    return {
        "schema_version": 1,
        "algorithm_version": "stereo-coverage-sidecar-v1",
        "availability": "available",
        "frame_names": summary_frame_names,
        "frame_count": expected_frame_count,
        "render_shape": [int(render_shape[0]), int(render_shape[1])],
        "render_mode": render_mode,
        "geometry_mode": geometry_mode,
        "occlusion_fill": occlusion_fill,
        "ordered_frame_manifest_fingerprint": ordered_frame_manifest_fingerprint,
        "frames_jsonl_sha256": frames_jsonl_sha256,
        "eyes": eyes,
        "metric_projection": metric_projection,
        "geometry_nearest": geometry_nearest,
    }
