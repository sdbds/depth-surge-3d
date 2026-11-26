"""Canonical per-frame Quality coverage statistics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..forward_splat import HORIZONTAL_SUBPIXELS
from .types import (
    CompactStereoRenderResult,
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


def _histogram_metrics(
    histogram: np.ndarray,
    percentile: float,
) -> tuple[int, int, float]:
    values = np.asarray(histogram)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise TypeError("histogram must be a one-dimensional integer array")
    if np.issubdtype(values.dtype, np.signedinteger) and np.any(values < 0):
        raise ValueError("histogram counts must be nonnegative")
    cumulative = np.cumsum(values.astype(np.uint64, copy=False), dtype=np.uint64)
    if cumulative.size == 0 or int(cumulative[-1]) == 0:
        return 0, 0, 0.0
    total = int(cumulative[-1])
    maximum = int(values.size - 1 - np.argmax(values[::-1] != 0))
    position = percentile * (total - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    lower = int(np.searchsorted(cumulative, np.uint64(lower_index), side="right"))
    upper = int(np.searchsorted(cumulative, np.uint64(upper_index), side="right"))
    interpolated = float(lower) + (float(upper) - float(lower)) * (position - lower_index)
    return total, maximum, interpolated


def _prefill_run_histograms(
    width: int,
    streamed: HoleRunDiagnostics | None,
) -> tuple[np.ndarray, np.ndarray]:
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


def _histogram_pairs(histogram: np.ndarray) -> list[list[int]]:
    nonzero = np.flatnonzero(histogram)
    return [[int(index), int(histogram[index])] for index in nonzero]


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
        if plan is None:
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
        else:
            quality_budgets = {
                "availability": "quality_repair_plan",
                "segment_record_count": plan.segment_record_count,
                "segment_table_bytes": plan.segment_table_bytes,
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
    """Build per-frame statistics from a compact stereo render result."""

    height = int(compact.left_image.shape[0])
    width = int(compact.left_image.shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("diagnostics render dimensions must be positive")
    if geometry_mode not in {"relative", "metric"}:
        raise ValueError("diagnostic geometry_mode must be relative or metric")
    if render_mode not in {"fast", "quality"}:
        raise ValueError("diagnostic render_mode must be fast or quality")
    limits = getattr(compact, "quality_limits", None)
    left_limits = limits.get("left") if isinstance(limits, dict) else None
    right_limits = limits.get("right") if isinstance(limits, dict) else None
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
        "render_shape": [height, width],
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
