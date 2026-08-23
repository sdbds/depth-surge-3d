"""Compact two-pass offline Quality stereo renderer."""

from __future__ import annotations

import gc
import math
from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from ..forward_splat import HORIZONTAL_SUBPIXELS
from ..stereo_renderer import (
    StereoRenderResult,
    StereoRenderer,
    StereoRenderSettings,
    _convert_image_band,
    _downsample_subpixel_band,
    calculate_band_height,
)
from .geometry import (
    QualityGeometryResult,
    build_geometry_eye_offsets_into,
    build_metric_quality_geometry,
    build_relative_eye_offsets_into,
    build_relative_quality_geometry,
)
from .local_strip import QualityRepairBudgets, QualityVisibilityAnalysis
from .native_repair import build_native_repair_records
from .planner import QualityRepairPlan, plan_quality_repairs
from .repair_records import (
    NO_REGION,
    QUALITY_REPAIR_RECORD_CAP,
    REPAIR_RECORD_DTYPE,
    QualityRepairBudgetError,
    validate_record_coverage,
)
from .types import (
    CompactStereoRenderResult,
    HoleRunDiagnostics,
    MetricStereoPrimitiveInput,
    QualityStereoControls,
)
from .visibility import QualitySplatBand, quality_splat_band


@dataclass(frozen=True)
class _EyeAnalysisResult:
    analysis: QualityVisibilityAnalysis
    records: np.ndarray


@dataclass(frozen=True)
class _CompactEyeResult:
    image: np.ndarray
    coverage_count: np.ndarray
    repair_bits: np.ndarray
    plan: QualityRepairPlan | None
    hole_runs: HoleRunDiagnostics | None = None


@dataclass
class _FrameBandState:
    initial_height: int
    retry_height: int
    current_height: int
    retry_used: bool = False


def _release_after_oom(renderer: StereoRenderer) -> None:
    renderer._release_after_oom()
    gc.collect()


def _retry_after_oom(
    owner: StereoRenderer,
    state: _FrameBandState,
    operation,
    **kwargs,
):
    try:
        return operation(band_height=state.current_height, **kwargs)
    except torch.cuda.OutOfMemoryError as error:
        if state.retry_used:
            width = int(kwargs["source"].shape[1])
            height = int(kwargs["source"].shape[0])
            _release_after_oom(owner)
            raise RuntimeError(
                "CUDA Quality stereo rendering ran out of memory for frame "
                f"{width}x{height}; attempted band heights "
                f"{state.initial_height} and {state.retry_height}"
            ) from error
        _release_after_oom(owner)
        state.retry_used = True
        state.current_height = state.retry_height
        return operation(band_height=state.current_height, **kwargs)


def _quality_splat(
    renderer: StereoRenderer,
    source: np.ndarray,
    geometry: QualityGeometryResult,
    offsets: np.ndarray,
    start_row: int,
    end_row: int,
) -> QualitySplatBand:
    width = source.shape[1]
    source_band = torch.from_numpy(np.ascontiguousarray(source[start_row:end_row])).to(
        renderer.device
    )
    near_score_band = torch.from_numpy(
        np.array(geometry.near_score[start_row:end_row], copy=True, order="C")
    ).to(renderer.device)
    source_valid_band = torch.from_numpy(
        np.array(geometry.source_valid[start_row:end_row], copy=True, order="C")
    ).to(renderer.device)
    offset_band = renderer._transfer_offset_band(offsets[start_row:end_row])
    return quality_splat_band(
        source_band,
        near_score_band,
        offset_band,
        source_index_offset=start_row * width,
        source_valid=source_valid_band,
    )


def _write_analysis_band(
    analysis: QualityVisibilityAnalysis,
    band: QualitySplatBand,
    source_regions: np.ndarray,
    start_row: int,
    end_row: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_count = end_row - start_row
    width = analysis.coverage_count.shape[1]
    valid = np.ascontiguousarray(band.splat.valid.detach().cpu().numpy())
    winners = np.ascontiguousarray(
        band.winner_source_index.detach().cpu().numpy().astype(np.uint32, copy=False)
    )
    safe_winners = np.where(valid, winners, np.uint32(0))
    winner_regions = source_regions.reshape(-1)[safe_winners]
    winner_regions = np.where(valid, winner_regions, NO_REGION).astype(np.uint32, copy=False)
    lane_valid = valid.reshape(row_count, width, HORIZONTAL_SUBPIXELS)
    lane_regions = winner_regions.reshape(row_count, width, HORIZONTAL_SUBPIXELS)
    coverage = np.sum(lane_valid, axis=2, dtype=np.uint16).astype(np.uint8)
    first_lane = np.argmax(lane_valid, axis=2)
    first_region = np.take_along_axis(
        lane_regions,
        first_lane[..., None],
        axis=2,
    )[..., 0]
    pure = (coverage > 0) & np.all(
        (~lane_valid) | (lane_regions == first_region[..., None]),
        axis=2,
    )
    analysis.coverage_count[start_row:end_row] = coverage
    region_output = analysis.pure_region_id[start_row:end_row]
    region_output.fill(NO_REGION)
    region_output[pure] = first_region[pure]

    colours = band.splat.colour.detach().cpu().numpy()
    lane_colours = colours.reshape(row_count, width, HORIZONTAL_SUBPIXELS, 3)
    sums = np.sum(lane_colours, axis=2, dtype=np.int32)
    nonzero = coverage > 0
    rounded = np.zeros((row_count, width, 3), dtype=np.uint8)
    if nonzero.any():
        means = sums[nonzero].astype(np.float64)
        means /= coverage[nonzero, None].astype(np.float64)
        rounded[nonzero] = np.rint(means).astype(np.uint8)
    for channel, plane in enumerate(analysis.pure_bgr):
        output = plane[start_row:end_row]
        output.fill(0)
        output[pure] = rounded[..., channel][pure]
    near_score = np.ascontiguousarray(band.splat.disparity.detach().cpu().numpy())
    return valid, near_score, np.ascontiguousarray(winner_regions)


def _analyze_eye_once(
    *,
    renderer: StereoRenderer,
    source: np.ndarray,
    geometry: QualityGeometryResult,
    offsets: np.ndarray,
    band_height: int,
) -> _EyeAnalysisResult:
    height, width = source.shape[:2]
    coverage = np.empty((height, width), dtype=np.uint8)
    pure_regions = np.empty((height, width), dtype=np.uint32)
    pure_bgr = tuple(np.empty((height, width), dtype=np.uint8) for _ in range(3))
    analysis = QualityVisibilityAnalysis(coverage, pure_regions, pure_bgr)
    record_arena = np.zeros(QUALITY_REPAIR_RECORD_CAP, dtype=REPAIR_RECORD_DTYPE)
    record_count = 0
    for start_row in range(0, height, band_height):
        end_row = min(height, start_row + band_height)
        band = _quality_splat(renderer, source, geometry, offsets, start_row, end_row)
        valid, winner_near_score, winner_regions = _write_analysis_band(
            analysis,
            band,
            geometry.final_region_map,
            start_row,
            end_row,
        )
        if not valid.all():
            built = build_native_repair_records(valid, winner_near_score, winner_regions)
            next_count = record_count + int(built.records.size)
            if next_count > QUALITY_REPAIR_RECORD_CAP:
                raise QualityRepairBudgetError("64 MiB repair record arena exceeded")
            destination = record_arena[record_count:next_count]
            destination[:] = built.records
            destination["pixel_index"] += np.uint32(start_row * width)
            record_count = next_count
        del band, valid, winner_near_score, winner_regions
    records = record_arena[:record_count]
    records.setflags(write=False)
    return _EyeAnalysisResult(analysis=analysis, records=records)


def _none_repair_bits(coverage: np.ndarray) -> np.ndarray:
    result = np.zeros(coverage.shape, dtype=np.uint8)
    partial = (coverage > 0) & (coverage < HORIZONTAL_SUBPIXELS)
    full = coverage == 0
    result[partial] |= np.uint8(0x01)
    result[full] |= np.uint8(0x02)
    result[coverage < HORIZONTAL_SUBPIXELS] |= np.uint8(0x40)
    return result


def _render_none_eye_once(
    *,
    renderer: StereoRenderer,
    source: np.ndarray,
    geometry: QualityGeometryResult,
    offsets: np.ndarray,
    band_height: int,
) -> _CompactEyeResult:
    height, width = source.shape[:2]
    output = np.empty_like(source)
    coverage = np.empty((height, width), dtype=np.uint8)
    lane_histogram: Counter[int] = Counter()
    span_histogram: Counter[int] = Counter()
    for start_row in range(0, height, band_height):
        end_row = min(height, start_row + band_height)
        band = _quality_splat(renderer, source, geometry, offsets, start_row, end_row)
        lane_valid = band.splat.valid.reshape(
            end_row - start_row,
            width,
            HORIZONTAL_SUBPIXELS,
        )
        lane_valid_host = np.ascontiguousarray(lane_valid.detach().cpu().numpy())
        coverage[start_row:end_row] = lane_valid_host.sum(axis=2).astype(np.uint8)
        for row in lane_valid_host:
            missing = np.flatnonzero(~row.reshape(-1))
            cursor = 0
            while cursor < missing.size:
                start = int(missing[cursor])
                end_cursor = cursor + 1
                while (
                    end_cursor < missing.size
                    and int(missing[end_cursor]) == int(missing[end_cursor - 1]) + 1
                ):
                    end_cursor += 1
                end = int(missing[end_cursor - 1])
                lane_histogram[end - start + 1] += 1
                span_histogram[end // HORIZONTAL_SUBPIXELS - start // HORIZONTAL_SUBPIXELS + 1] += 1
                cursor = end_cursor
        downsampled = _downsample_subpixel_band(band.splat.colour)
        output[start_row:end_row] = _convert_image_band(downsampled, source.dtype)
        del band, lane_valid, lane_valid_host, downsampled
    hole_runs = HoleRunDiagnostics(
        lane_count_histogram=tuple(sorted(lane_histogram.items())),
        touched_pixel_span_histogram=tuple(sorted(span_histogram.items())),
    )
    return _CompactEyeResult(
        output,
        coverage,
        _none_repair_bits(coverage),
        None,
        hole_runs,
    )


def _background_repair_bits(
    coverage: np.ndarray,
    records: np.ndarray,
) -> np.ndarray:
    result = np.zeros(coverage.shape, dtype=np.uint8)
    result[(coverage > 0) & (coverage < HORIZONTAL_SUBPIXELS)] |= np.uint8(0x01)
    result[coverage == 0] |= np.uint8(0x02)
    flat = result.reshape(-1)
    for record in records:
        pixel = int(record["pixel_index"])
        backend = int(record["backend"])
        if backend == 1:
            flat[pixel] |= np.uint8(0x04)
        elif backend == 2:
            flat[pixel] |= np.uint8(0x08 | 0x10)
        elif backend == 3:
            flat[pixel] |= np.uint8(0x08 | 0x20)
        else:
            raise RuntimeError("Quality repair plan contains an unplanned record")
    return result


def _band_records(
    records: np.ndarray,
    *,
    start_pixel: int,
    end_pixel: int,
) -> np.ndarray:
    pixels = records["pixel_index"]
    start = int(np.searchsorted(pixels, np.uint32(start_pixel), side="left"))
    end = int(np.searchsorted(pixels, np.uint32(end_pixel), side="left"))
    selected = np.array(records[start:end], copy=True, order="C")
    selected["pixel_index"] -= np.uint32(start_pixel)
    return selected


def _scatter_repair_records(
    colour: torch.Tensor,
    records: np.ndarray,
    *,
    width: int,
) -> int:
    lane_count = sum(int(record["lane_mask"]).bit_count() for record in records)
    if lane_count == 0:
        return 0
    indexes = np.empty(lane_count, dtype=np.int64)
    colours = np.empty((lane_count, 3), dtype=np.float32)
    cursor = 0
    for record in records:
        pixel = int(record["pixel_index"])
        row, column = divmod(pixel, width)
        mask = int(record["lane_mask"])
        for lane in range(HORIZONTAL_SUBPIXELS):
            if mask & (1 << lane):
                indexes[cursor] = row * width * HORIZONTAL_SUBPIXELS + column * 16 + lane
                colours[cursor] = record["fill_bgr"]
                cursor += 1
    if cursor != lane_count:
        raise RuntimeError("Quality repair scatter count changed")
    device_indexes = torch.from_numpy(indexes).to(colour.device)
    device_colours = torch.from_numpy(colours).to(colour.device)
    colour.reshape(-1, 3).index_copy_(0, device_indexes, device_colours)
    return lane_count


def _replay_background_eye_once(
    *,
    renderer: StereoRenderer,
    source: np.ndarray,
    geometry: QualityGeometryResult,
    offsets: np.ndarray,
    plan: QualityRepairPlan,
    band_height: int,
) -> np.ndarray:
    height, width = source.shape[:2]
    output = np.empty_like(source)
    scattered = 0
    planned = sum(int(value).bit_count() for value in plan.records["lane_mask"])
    for start_row in range(0, height, band_height):
        end_row = min(height, start_row + band_height)
        band = _quality_splat(renderer, source, geometry, offsets, start_row, end_row)
        valid = np.ascontiguousarray(band.splat.valid.detach().cpu().numpy())
        records = _band_records(
            plan.records,
            start_pixel=start_row * width,
            end_pixel=end_row * width,
        )
        validate_record_coverage(records, ~valid)
        scattered += _scatter_repair_records(
            band.splat.colour,
            records,
            width=width,
        )
        downsampled = _downsample_subpixel_band(band.splat.colour)
        output[start_row:end_row] = _convert_image_band(downsampled, source.dtype)
        del band, valid, records, downsampled
    if scattered != planned:
        raise RuntimeError("Quality Pass B scatter total does not match the repair plan")
    return output


def _render_background_eye(
    *,
    renderer: StereoRenderer,
    source: np.ndarray,
    geometry: QualityGeometryResult,
    offsets: np.ndarray,
    controls_limit_px: int,
    budgets: QualityRepairBudgets,
    band_state: _FrameBandState,
) -> _CompactEyeResult:
    analyzed = _retry_after_oom(
        renderer,
        band_state,
        _analyze_eye_once,
        renderer=renderer,
        source=source,
        geometry=geometry,
        offsets=offsets,
    )
    safe_limit = max(1, math.floor(controls_limit_px * source.shape[0] / 1080.0 + 0.5))
    local_limit = min(safe_limit, geometry.predicted_gap_px)
    plan = plan_quality_repairs(
        analyzed.analysis,
        records=analyzed.records,
        render_shape=source.shape[:2],
        local_limit_px=local_limit,
        budgets=budgets,
    )
    coverage = analyzed.analysis.coverage_count
    repair_bits = _background_repair_bits(coverage, plan.records)
    image = _retry_after_oom(
        renderer,
        band_state,
        _replay_background_eye_once,
        renderer=renderer,
        source=source,
        geometry=geometry,
        offsets=offsets,
        plan=plan,
    )
    return _CompactEyeResult(image, coverage, repair_bits, plan)


def _build_eye_offsets(
    geometry: QualityGeometryResult,
    settings: StereoRenderSettings,
    eye: Literal["left", "right"],
) -> np.ndarray:
    output = np.empty(geometry.near_score.shape, dtype=np.int32)
    build_relative_eye_offsets_into(
        geometry.near_score,
        stereo_strength=settings.stereo_strength,
        convergence=settings.convergence,
        eye=eye,
        output_int32=output,
        row_float64_scratch=np.empty(geometry.near_score.shape[1], dtype=np.float64),
    )
    return output


def _build_metric_eye_offsets(
    geometry: QualityGeometryResult,
    eye: Literal["left", "right"],
) -> np.ndarray:
    output = np.empty(geometry.near_score.shape, dtype=np.int32)
    build_geometry_eye_offsets_into(
        geometry.total_disparity_fraction,
        eye=eye,
        output_int32=output,
        row_float64_scratch=np.empty(geometry.near_score.shape[1], dtype=np.float64),
    )
    return output


def _quality_limits(
    geometry: QualityGeometryResult,
    *,
    height: int,
    occlusion_fill: Literal["none", "background"],
    occlusion_fill_max_px: int,
) -> dict[str, dict[str, int | float]] | None:
    if occlusion_fill != "background":
        return None
    safe_limit = max(
        1,
        math.floor(occlusion_fill_max_px * height / 1080.0 + 0.5),
    )
    local_limit = min(safe_limit, geometry.predicted_gap_px)
    return {
        eye: {
            "max_neighbour_abs_q_jump_px": float(
                geometry.max_neighbour_abs_q_jump_px
            ),
            "predicted_gap_px": int(geometry.predicted_gap_px),
            "local_limit_px": int(local_limit),
        }
        for eye in ("left", "right")
    }


def _render_metric_quality_eyes(
    *,
    renderer: StereoRenderer,
    source: np.ndarray,
    geometry: QualityGeometryResult,
    controls: QualityStereoControls,
) -> tuple[_CompactEyeResult, _CompactEyeResult]:
    height, width = source.shape[:2]
    initial_height = calculate_band_height(
        width,
        height,
        renderer.temporary_budget_bytes,
    )
    band_state = _FrameBandState(
        initial_height=initial_height,
        retry_height=max(1, initial_height // 2),
        current_height=initial_height,
    )
    budgets = QualityRepairBudgets()
    eyes: list[_CompactEyeResult] = []
    for eye in ("left", "right"):
        offsets = _build_metric_eye_offsets(geometry, eye)
        if controls.occlusion_fill == "none":
            result = _retry_after_oom(
                renderer,
                band_state,
                _render_none_eye_once,
                renderer=renderer,
                source=source,
                geometry=geometry,
                offsets=offsets,
            )
        else:
            result = _render_background_eye(
                renderer=renderer,
                source=source,
                geometry=geometry,
                offsets=offsets,
                controls_limit_px=controls.occlusion_fill_max_px,
                budgets=budgets,
                band_state=band_state,
            )
        eyes.append(result)
        del offsets
    return eyes[0], eyes[1]


def render_relative_quality_compact(
    *,
    renderer: StereoRenderer,
    frame: np.ndarray,
    canonical: np.ndarray,
    settings: StereoRenderSettings,
) -> CompactStereoRenderResult:
    """Render relative primitives with sequential compact Quality eyes."""

    if not isinstance(renderer, StereoRenderer):
        raise TypeError("renderer must be StereoRenderer")
    if not isinstance(settings, StereoRenderSettings) or settings.stereo_render_mode != "quality":
        raise ValueError("Quality rendering requires StereoRenderSettings in quality mode")
    source = np.ascontiguousarray(np.asarray(frame))
    if source.dtype != np.uint8:
        raise TypeError("Quality rendering requires uint8 BGR input")
    canonical_values = np.ascontiguousarray(np.asarray(canonical, dtype=np.float32))
    StereoRenderer._validate_inputs(source, canonical_values, settings)
    height, width = source.shape[:2]
    geometry = build_relative_quality_geometry(
        source,
        canonical_values,
        render_shape=(height, width),
        stereo_strength=settings.stereo_strength,
        convergence=settings.convergence,
    )
    initial_height = calculate_band_height(
        width,
        height,
        renderer.temporary_budget_bytes,
    )
    band_state = _FrameBandState(
        initial_height=initial_height,
        retry_height=max(1, initial_height // 2),
        current_height=initial_height,
    )
    budgets = QualityRepairBudgets()
    eyes: list[_CompactEyeResult] = []
    for eye in ("left", "right"):
        offsets = _build_eye_offsets(geometry, settings, eye)
        if settings.occlusion_fill == "none":
            result = _retry_after_oom(
                renderer,
                band_state,
                _render_none_eye_once,
                renderer=renderer,
                source=source,
                geometry=geometry,
                offsets=offsets,
            )
        else:
            result = _render_background_eye(
                renderer=renderer,
                source=source,
                geometry=geometry,
                offsets=offsets,
                controls_limit_px=settings.occlusion_fill_max_px,
                budgets=budgets,
                band_state=band_state,
            )
        eyes.append(result)
        del offsets
    left, right = eyes
    safe_limit = max(
        1,
        math.floor(settings.occlusion_fill_max_px * height / 1080.0 + 0.5),
    )
    local_limit = min(safe_limit, geometry.predicted_gap_px)
    quality_limits = (
        {
            eye: {
                "max_neighbour_abs_q_jump_px": float(
                    geometry.max_neighbour_abs_q_jump_px
                ),
                "predicted_gap_px": int(geometry.predicted_gap_px),
                "local_limit_px": int(local_limit),
            }
            for eye in ("left", "right")
        }
        if settings.occlusion_fill == "background"
        else None
    )
    return CompactStereoRenderResult(
        left_image=left.image,
        right_image=right.image,
        left_coverage_count=left.coverage_count,
        right_coverage_count=right.coverage_count,
        left_repair_bits=left.repair_bits,
        right_repair_bits=right.repair_bits,
        left_plan=left.plan,
        right_plan=right.plan,
        geometry_nearest={
            "indexed_sample_count": int(canonical_values.size),
            "query_count": int(geometry.geometry_query_count),
            "visited_nodes_total": int(geometry.geometry_visited_nodes_total),
            "visited_nodes_max": int(geometry.geometry_visited_nodes_max),
        },
        quality_limits=quality_limits,
        left_hole_runs=left.hole_runs,
        right_hole_runs=right.hole_runs,
    )


def render_metric_primitives_compact(
    *,
    renderer: StereoRenderer,
    frame: np.ndarray,
    primitives: MetricStereoPrimitiveInput,
    controls: QualityStereoControls,
) -> CompactStereoRenderResult:
    """Render native metric primitives after edge-aware Quality resampling."""

    if not isinstance(renderer, StereoRenderer):
        raise TypeError("renderer must be StereoRenderer")
    if not isinstance(primitives, MetricStereoPrimitiveInput):
        raise TypeError("primitives must be MetricStereoPrimitiveInput")
    if not isinstance(controls, QualityStereoControls):
        raise TypeError("controls must be QualityStereoControls")
    source = np.ascontiguousarray(np.asarray(frame))
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
        raise TypeError("Quality rendering requires uint8 BGR input")
    metric = primitives.metric
    for name in ("inverse_depth", "valid", "focal_x_normalized"):
        if not hasattr(metric, name):
            raise TypeError("metric primitive is missing native geometry fields")
    height, width = source.shape[:2]
    geometry = build_metric_quality_geometry(
        source,
        metric.inverse_depth,
        metric.valid,
        metric.focal_x_normalized,
        render_shape=(height, width),
        virtual_baseline_mm=float(primitives.virtual_baseline_mm),
        convergence_distance_m=float(primitives.convergence_distance_m),
        max_disparity_percent=float(primitives.max_disparity_percent),
        retained_crop_width=primitives.retained_crop_width,
    )
    left, right = _render_metric_quality_eyes(
        renderer=renderer,
        source=source,
        geometry=geometry,
        controls=controls,
    )
    if geometry.metric_stats is None:
        raise RuntimeError("Metric Quality geometry did not produce projection statistics")
    return CompactStereoRenderResult(
        left_image=left.image,
        right_image=right.image,
        left_coverage_count=left.coverage_count,
        right_coverage_count=right.coverage_count,
        left_repair_bits=left.repair_bits,
        right_repair_bits=right.repair_bits,
        left_plan=left.plan,
        right_plan=right.plan,
        metric_stats=geometry.metric_stats,
        geometry_nearest={
            "indexed_sample_count": int(metric.inverse_depth.size),
            "query_count": int(geometry.geometry_query_count),
            "visited_nodes_total": int(geometry.geometry_visited_nodes_total),
            "visited_nodes_max": int(geometry.geometry_visited_nodes_max),
        },
        quality_limits=_quality_limits(
            geometry,
            height=height,
            occlusion_fill=controls.occlusion_fill,
            occlusion_fill_max_px=controls.occlusion_fill_max_px,
        ),
        left_hole_runs=left.hole_runs,
        right_hole_runs=right.hole_runs,
    )


def render_relative_quality_public(
    *,
    renderer: StereoRenderer,
    frame: np.ndarray,
    canonical: np.ndarray,
    settings: StereoRenderSettings,
) -> StereoRenderResult:
    """Materialize the four legacy boolean masks for the public API."""

    compact = render_relative_quality_compact(
        renderer=renderer,
        frame=frame,
        canonical=canonical,
        settings=settings,
    )
    left_valid = np.array(compact.left_coverage_count > 0, copy=True, order="C")
    right_valid = np.array(compact.right_coverage_count > 0, copy=True, order="C")
    if settings.occlusion_fill == "background":
        left_hole = np.zeros(left_valid.shape, dtype=np.bool_)
        right_hole = np.zeros(right_valid.shape, dtype=np.bool_)
    else:
        left_hole = np.array(compact.left_coverage_count == 0, copy=True, order="C")
        right_hole = np.array(compact.right_coverage_count == 0, copy=True, order="C")
    return StereoRenderResult(
        left_image=compact.left_image,
        right_image=compact.right_image,
        left_valid_mask=left_valid,
        right_valid_mask=right_valid,
        left_hole_mask=left_hole,
        right_hole_mask=right_hole,
    )
