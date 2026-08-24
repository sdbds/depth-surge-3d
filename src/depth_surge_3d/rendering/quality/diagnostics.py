"""Canonical per-frame Quality coverage statistics."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from ..forward_splat import HORIZONTAL_SUBPIXELS
from .planner import QualityRepairPlan
from .types import CompactStereoRenderResult, FastEyeDiagnostics, HoleRunDiagnostics


_STATE_BITS = {
    "prefill_partial": 0x01,
    "prefill_full": 0x02,
    "local_filled": 0x04,
    "post_local_residual": 0x08,
    "exemplar_filled": 0x10,
    "fallback_filled": 0x20,
    "final_unresolved": 0x40,
}


def _linear_percentile_from_histogram(histogram: Counter[int], percentile: float) -> float:
    total = sum(histogram.values())
    if total == 0:
        return 0.0
    position = percentile * (total - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))

    def value_at(index: int) -> int:
        seen = 0
        for value, count in sorted(histogram.items()):
            seen += count
            if index < seen:
                return value
        raise RuntimeError("histogram index is outside its total")

    lower = value_at(lower_index)
    upper = value_at(upper_index)
    return float(lower) + (float(upper) - float(lower)) * (position - lower_index)


def _prefill_run_histograms(
    plan: QualityRepairPlan | None,
    width: int,
    streamed: HoleRunDiagnostics | None,
) -> tuple[Counter, Counter]:
    lane_histogram: Counter[int] = Counter()
    span_histogram: Counter[int] = Counter()
    if streamed is not None:
        lane_histogram.update(dict(streamed.lane_count_histogram))
        span_histogram.update(dict(streamed.touched_pixel_span_histogram))
        return lane_histogram, span_histogram
    if plan is None or plan.records.size == 0:
        return lane_histogram, span_histogram
    active_row = -1
    active_start = -1
    active_end = -1
    for record in plan.records:
        pixel = int(record["pixel_index"])
        row, column = divmod(pixel, width)
        mask = int(record["lane_mask"])
        start = column * 16 + (mask & -mask).bit_length() - 1
        end = column * 16 + mask.bit_length() - 1
        if row == active_row and start == active_end + 1:
            active_end = end
            continue
        if active_row >= 0:
            length = active_end - active_start + 1
            lane_histogram[length] += 1
            span_histogram[active_end // 16 - active_start // 16 + 1] += 1
        active_row, active_start, active_end = row, start, end
    length = active_end - active_start + 1
    lane_histogram[length] += 1
    span_histogram[active_end // 16 - active_start // 16 + 1] += 1
    return lane_histogram, span_histogram


def _histogram_pairs(histogram: Counter[int]) -> list[list[int]]:
    return [[int(value), int(count)] for value, count in sorted(histogram.items())]


def _backend_counts(
    plan: QualityRepairPlan | None,
) -> tuple[dict[str, int], dict[str, int], dict[str, Any]]:
    names = {1: "local", 2: "exemplar", 3: "fallback"}
    lane_counts = {name: 0 for name in names.values()}
    pixels = {name: set() for name in names.values()}
    components = {name: set() for name in names.values()}
    if plan is not None:
        for index, record in enumerate(plan.records):
            backend = names[int(record["backend"])]
            lane_counts[backend] += int(record["lane_mask"]).bit_count()
            pixels[backend].add(int(record["pixel_index"]))
            components[backend].add(int(plan.record_component_ids[index]))
        component_counts: dict[str, Any] = {
            "availability": "quality_segment_graph",
            **{name: len(values) for name, values in components.items()},
        }
    else:
        component_counts = {
            "availability": "unavailable_quality_none_no_repair_graph",
            "local": None,
            "exemplar": None,
            "fallback": None,
        }
    return (
        lane_counts,
        {name: len(values) for name, values in pixels.items()},
        component_counts,
    )


def _eye_stats(
    coverage: np.ndarray,
    repair_bits: np.ndarray,
    plan: QualityRepairPlan | None,
    *,
    quality_limits: dict[str, int | float] | None,
    streamed_hole_runs: HoleRunDiagnostics | None,
    render_mode: str,
    fast_diagnostics: FastEyeDiagnostics | None,
) -> dict[str, Any]:
    pixel_count = int(coverage.size)
    state_counts = {
        name: int(np.count_nonzero(repair_bits & np.uint8(bit)))
        for name, bit in _STATE_BITS.items()
    }
    state_ratios = {
        name: (float(count) / float(pixel_count) if pixel_count else 0.0)
        for name, count in state_counts.items()
    }
    lane_histogram, span_histogram = _prefill_run_histograms(
        plan,
        coverage.shape[1],
        streamed_hole_runs,
    )
    lane_p95 = _linear_percentile_from_histogram(lane_histogram, 0.95)
    if render_mode == "fast":
        if fast_diagnostics is None:
            raise ValueError("Fast diagnostics require producer-attested counters")
        backend_lanes = {
            "local": int(fast_diagnostics.local_filled_lane_count),
            "exemplar": 0,
            "fallback": 0,
        }
        backend_pixels = {
            "local": int(np.count_nonzero(repair_bits & np.uint8(0x04))),
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
        quality_budgets: dict[str, Any] = {
            "availability": "unavailable_quality_none_no_repair_plan",
            "segment_record_count": None,
            "segment_table_bytes": None,
            "local_slot_charge": None,
            "local_neighborhood_sample_charge": None,
            "local_budget_skipped_run_count": None,
            "exemplar_evaluations": None,
            "fallback_indexed_donor_count": None,
            "fallback_query_count": None,
            "fallback_anchor_count": None,
            "fallback_visited_nodes_total": None,
            "fallback_visited_nodes_max": None,
            "fallback_visited_nodes_p95": None,
        }
    elif render_mode == "quality":
        assert plan is not None
        quality_budgets = {
            "availability": "quality_repair_plan",
            "segment_record_count": int(plan.records.size),
            "segment_table_bytes": int(plan.records.nbytes),
            "local_slot_charge": int(plan.local_evaluated_slot_count),
            "local_neighborhood_sample_charge": int(plan.local_physical_sample_read_count),
            "local_budget_skipped_run_count": int(plan.local_budget_skipped_run_count),
            "exemplar_evaluations": int(plan.exemplar_donor_evaluation_count),
            "fallback_indexed_donor_count": int(plan.fallback_indexed_donor_count),
            "fallback_query_count": int(plan.fallback_query_count),
            "fallback_anchor_count": int(plan.fallback_anchor_count),
            "fallback_visited_nodes_total": int(plan.fallback_visited_nodes_total),
            "fallback_visited_nodes_max": int(plan.fallback_visited_nodes_max),
            "fallback_visited_nodes_p95": float(plan.fallback_visited_nodes_p95),
        }
    return {
        "pixel_count": pixel_count,
        "state_pixel_counts": state_counts,
        "state_pixel_ratios": state_ratios,
        "hole_runs": {
            "count": int(sum(lane_histogram.values())),
            "lane_count_histogram": _histogram_pairs(lane_histogram),
            "touched_pixel_span_histogram": _histogram_pairs(span_histogram),
            "lane_count_max": int(max(lane_histogram, default=0)),
            "lane_count_p95": lane_p95,
            "touched_pixel_span_max": int(max(span_histogram, default=0)),
            "touched_pixel_span_p95": _linear_percentile_from_histogram(span_histogram, 0.95),
            "physical_width_px_max": float(max(lane_histogram, default=0)) / 16.0,
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
) -> Counter[int]:
    merged: Counter[int] = Counter()
    for eye in eyes:
        for value, count in eye["hole_runs"][field]:
            merged[int(value)] += int(count)
    return merged


def _sum_named_counts(
    values: list[dict[str, Any]],
    field: str,
    names: tuple[str, ...],
) -> dict[str, int]:
    return {name: sum(int(value[field][name]) for value in values) for name in names}


def _aggregate_eye(eyes: list[dict[str, Any]], render_mode: str) -> dict[str, Any]:
    state_names = tuple(_STATE_BITS)
    backend_names = ("local", "exemplar", "fallback")
    pixel_count = sum(int(eye["pixel_count"]) for eye in eyes)
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
            "fallback_anchor_count": None,
            "fallback_visited_nodes_total": None,
            "fallback_visited_nodes_max": None,
            "fallback_visited_nodes_frame_p95_max": None,
        }
    elif all(value["availability"] == "quality_repair_plan" for value in budgets):
        budget_totals = {
            "availability": "quality_repair_plan",
            "segment_record_count": sum(int(value["segment_record_count"]) for value in budgets),
            "segment_table_bytes_max": max(int(value["segment_table_bytes"]) for value in budgets),
            "local_slot_charge": sum(int(value["local_slot_charge"]) for value in budgets),
            "local_neighborhood_sample_charge": sum(
                int(value["local_neighborhood_sample_charge"]) for value in budgets
            ),
            "local_budget_skipped_run_count": sum(
                int(value["local_budget_skipped_run_count"]) for value in budgets
            ),
            "exemplar_evaluations": sum(int(value["exemplar_evaluations"]) for value in budgets),
            "fallback_indexed_donor_count": sum(
                int(value["fallback_indexed_donor_count"]) for value in budgets
            ),
            "fallback_query_count": sum(int(value["fallback_query_count"]) for value in budgets),
            "fallback_anchor_count": sum(int(value["fallback_anchor_count"]) for value in budgets),
            "fallback_visited_nodes_total": sum(
                int(value["fallback_visited_nodes_total"]) for value in budgets
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
            "count": int(sum(lane_histogram.values())),
            "lane_count_histogram": _histogram_pairs(lane_histogram),
            "touched_pixel_span_histogram": _histogram_pairs(span_histogram),
            "lane_count_max": int(max(lane_histogram, default=0)),
            "lane_count_p95": lane_p95,
            "touched_pixel_span_max": int(max(span_histogram, default=0)),
            "touched_pixel_span_p95": _linear_percentile_from_histogram(span_histogram, 0.95),
            "physical_width_px_max": float(max(lane_histogram, default=0)) / 16.0,
            "physical_width_px_p95": lane_p95 / 16.0,
        },
        "backend_lane_counts": _sum_named_counts(eyes, "backend_lane_counts", backend_names),
        "backend_pixel_counts": _sum_named_counts(eyes, "backend_pixel_counts", backend_names),
        "backend_component_counts": component_counts,
        "final_unresolved_lane_count": sum(int(eye["final_unresolved_lane_count"]) for eye in eyes),
        "quality_limit_ranges": limit_ranges,
        "quality_budget_totals": budget_totals,
    }


def build_diagnostics_summary(
    frame_stats: list[dict[str, Any]],
    *,
    frame_names: tuple[str, ...],
    render_shape: tuple[int, int],
    render_mode: str,
    geometry_mode: str,
    occlusion_fill: str,
    ordered_frame_manifest_fingerprint: str,
    frames_jsonl_sha256: str,
) -> dict[str, Any]:
    """Aggregate validated source-ordered frame statistics."""

    if [value["frame_name"] for value in frame_stats] != list(frame_names):
        raise ValueError("frame statistics are not in exact source order")
    eyes = {
        name: _aggregate_eye([value["eyes"][name] for value in frame_stats], render_mode)
        for name in ("left", "right")
    }
    if geometry_mode == "metric":
        projections = [value["metric_projection"] for value in frame_stats]
        if any(value is None for value in projections):
            raise ValueError("metric diagnostics require projection statistics")
        concrete = [value for value in projections if value is not None]
        valid_count = sum(int(value["valid_pixel_count"]) for value in concrete)
        clamped_count = sum(int(value["clamped_pixel_count"]) for value in concrete)
        fractions = [float(value["clamped_fraction"]) for value in concrete]
        metric_projection = {
            "valid_pixel_count": valid_count,
            "clamped_pixel_count": clamped_count,
            "clamped_fraction": (float(clamped_count) / float(valid_count) if valid_count else 0.0),
            "clamped_fractions": fractions,
            "affected_frame_count": sum(value > 0.0 for value in fractions),
            "mean_clamped_fraction": (
                sum(fractions, 0.0) / float(len(fractions)) if fractions else 0.0
            ),
            "max_clamped_fraction": max(fractions, default=0.0),
        }
    else:
        if any(value["metric_projection"] is not None for value in frame_stats):
            raise ValueError("relative diagnostics cannot contain metric statistics")
        metric_projection = None

    nearest_values = [value["geometry_nearest"] for value in frame_stats]
    if render_mode == "quality":
        if any(value is None for value in nearest_values):
            raise ValueError("Quality diagnostics require geometry-nearest statistics")
        concrete_nearest = [value for value in nearest_values if value is not None]
        geometry_nearest = {
            "indexed_sample_count": sum(
                int(value["indexed_sample_count"]) for value in concrete_nearest
            ),
            "query_count": sum(int(value["query_count"]) for value in concrete_nearest),
            "visited_nodes_total": sum(
                int(value["visited_nodes_total"]) for value in concrete_nearest
            ),
            "visited_nodes_max": max(
                (int(value["visited_nodes_max"]) for value in concrete_nearest),
                default=0,
            ),
        }
    else:
        geometry_nearest = None
    return {
        "schema_version": 1,
        "algorithm_version": "stereo-coverage-sidecar-v1",
        "availability": "available",
        "frame_names": list(frame_names),
        "frame_count": len(frame_names),
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
