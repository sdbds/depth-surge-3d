"""Compact one-pass offline Quality stereo renderer."""

from __future__ import annotations

import gc
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal

import cv2
import numpy as np
import torch

from ...core.resource_errors import QualityHostBudgetError
from ..forward_splat import HORIZONTAL_SUBPIXELS
from ..stereo_renderer import (
    StereoRenderResult,
    StereoRenderer,
    StereoRenderSettings,
    _convert_image_band,
    _downsample_subpixel_band,
)
from . import native_kd
from .arena import QualityFixedArenas
from .geometry import (
    QualityGeometryResult,
    build_geometry_eye_offsets_into,
    build_metric_quality_geometry,
    build_relative_eye_offsets_into,
    build_relative_quality_geometry,
)
from .local_strip import QualityVisibilityAnalysis
from .planner import plan_quality_repairs  # noqa: F401 - retained as the v9 test seam.
from .planner import QualityRepairPlanSummary
from .repair_records import (
    NO_REGION,
    QUALITY_REPAIR_RECORD_CAP,
    REPAIR_RECORD_DTYPE,
    RepairRecordBuildResult,
    QualityRepairBudgetError,
    validate_record_coverage,
)
from .types import (
    CompactStereoRenderResult,
    HoleRunDiagnostics,
    MetricStereoPrimitiveInput,
    QualityStereoControls,
)
from .visibility import QualitySplatBand, _quality_splat_band_prevalidated


QUALITY_HOST_BUDGET_BYTES = 512 * 1024 * 1024
QUALITY_RUNTIME_OVERHEAD_BYTES = 16 * 1024 * 1024
QUALITY_SPLAT_BYTES_PER_PIXEL = 1536
QUALITY_SCATTER_WORKSPACE_BYTES = 12 * 1024 * 1024
QUALITY_SCATTER_CHUNK_PIXELS = QUALITY_SCATTER_WORKSPACE_BYTES // (HORIZONTAL_SUBPIXELS * 4)


def _calculate_quality_band_height(
    render_width: int,
    render_height: int,
    temporary_budget_bytes: int,
) -> int:
    if render_width <= 0 or render_height <= 0:
        raise ValueError("Quality render width and height must be positive")
    if temporary_budget_bytes <= 0:
        raise ValueError("Quality temporary GPU budget must be positive")
    rows = temporary_budget_bytes // (render_width * QUALITY_SPLAT_BYTES_PER_PIXEL)
    return min(render_height, max(1, rows))


def _quality_public_memory_preflight(
    *,
    render_shape: tuple[int, int],
    native_shape: tuple[int, int],
    occlusion_fill: Literal["none", "background"],
    materialize_public_masks: bool,
) -> dict[str, int]:
    height, width = render_shape
    native_height, native_width = native_shape
    if min(height, width, native_height, native_width) <= 0:
        raise ValueError("Quality render and native shapes must be positive")
    output_pixels = height * width
    native_pixels = native_height * native_width
    if 16 * output_pixels > np.iinfo(np.uint32).max:
        raise QualityHostBudgetError.for_limit(
            "quality-index-range",
            16 * output_pixels,
            int(np.iinfo(np.uint32).max),
        )
    if native_pixels + 1 > np.iinfo(np.uint32).max:
        raise QualityHostBudgetError.for_limit(
            "quality-native-index-range",
            native_pixels + 1,
            int(np.iinfo(np.uint32).max),
        )
    histogram_bytes = 272 * (width + 1)
    runtime = QUALITY_RUNTIME_OVERHEAD_BYTES
    phases = {
        "quality-lowres-region": 3 * output_pixels + 22 * native_pixels + runtime,
        "quality-geometry-nearest": 64 * output_pixels + 17 * native_pixels + runtime,
        "quality-region": 52 * output_pixels + 18 * native_pixels + runtime,
        "quality-geometry-matte": 68 * output_pixels + 18 * native_pixels + runtime,
    }
    if occlusion_fill == "background":
        visibility = 40 * output_pixels + 9 * native_pixels + histogram_bytes + runtime
        proxy = 100 * output_pixels + 9 * native_pixels + histogram_bytes + runtime
        phases["quality-background-visibility"] = visibility
        phases["quality-background-proxy"] = proxy
        if materialize_public_masks:
            phases["quality-public"] = max(
                proxy,
                14 * output_pixels + histogram_bytes + runtime,
            )
    else:
        visibility = 40 * output_pixels + 9 * native_pixels + histogram_bytes + runtime
        phases["quality-none-visibility"] = visibility
        if materialize_public_masks:
            phases["quality-none-public"] = max(
                visibility,
                14 * output_pixels + histogram_bytes + runtime,
            )
    phase, required = max(phases.items(), key=lambda item: item[1])
    if required > QUALITY_HOST_BUDGET_BYTES:
        label = "quality-public" if materialize_public_masks else phase
        raise QualityHostBudgetError.for_limit(
            label,
            required,
            QUALITY_HOST_BUDGET_BYTES,
        )
    return phases


@dataclass(frozen=True)
class _EyeAnalysisResult:
    analysis: QualityVisibilityAnalysis
    records: np.ndarray


@dataclass(frozen=True)
class _CompactEyeResult:
    image: np.ndarray
    coverage_count: np.ndarray
    repair_bits: np.ndarray
    plan: QualityRepairPlanSummary | None
    hole_runs: HoleRunDiagnostics | None = None


@dataclass(frozen=True)
class _UnrepairedEyeResult:
    image: np.ndarray
    coverage_count: np.ndarray
    hole_runs: HoleRunDiagnostics


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
    while True:
        try:
            return operation(band_height=state.current_height, **kwargs)
        except torch.cuda.OutOfMemoryError as error:
            _release_after_oom(owner)
            if state.retry_used:
                width = int(kwargs["source"].shape[1])
                height = int(kwargs["source"].shape[0])
                raise RuntimeError(
                    "CUDA Quality stereo rendering ran out of memory for frame "
                    f"{width}x{height}; attempted band heights "
                    f"{state.initial_height} and {state.retry_height}"
                ) from error
            state.retry_used = True
            state.current_height = state.retry_height


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
    return _quality_splat_band_prevalidated(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return (
        valid,
        near_score,
        np.ascontiguousarray(winner_regions),
        np.ascontiguousarray(
            lane_colours.reshape(row_count, width * HORIZONTAL_SUBPIXELS, 3),
            dtype=np.float32,
        ),
    )


def _write_analysis_band_native(
    analysis: QualityVisibilityAnalysis,
    band: QualitySplatBand,
    source_bgr: np.ndarray,
    source_near_score: np.ndarray,
    source_regions: np.ndarray,
    start_row: int,
    end_row: int,
    record_output: np.ndarray,
) -> RepairRecordBuildResult:
    winner_sources_i32 = np.ascontiguousarray(
        band.winner_source_index.to(dtype=torch.int32).detach().cpu().numpy()
    )
    winner_sources = winner_sources_i32.view(np.uint32)
    native = native_kd._require_native()
    raw_output = record_output.view(np.uint8).reshape(
        record_output.size,
        REPAIR_RECORD_DTYPE.itemsize,
    )
    try:
        record_count, statistics = native.analyze_splat_band(
            winner_sources,
            source_bgr,
            source_near_score,
            source_regions,
            analysis.coverage_count[start_row:end_row],
            analysis.pure_region_id[start_row:end_row],
            analysis.pure_bgr[0][start_row:end_row],
            analysis.pure_bgr[1][start_row:end_row],
            analysis.pure_bgr[2][start_row:end_row],
            raw_output,
        )
    except native.QualityBudgetError as error:
        raise QualityRepairBudgetError("64 MiB repair record arena exceeded") from error
    records = record_output[: int(record_count)]
    records.setflags(write=False)
    return RepairRecordBuildResult(
        records=records,
        prefill_run_count=int(statistics["prefill_run_count"]),
        repair_run_count=int(statistics["repair_run_count"]),
        invalid_lane_count=int(statistics["invalid_lane_count"]),
    )


def _analyze_eye_once(
    *,
    renderer: StereoRenderer,
    source: np.ndarray,
    geometry: QualityGeometryResult,
    offsets: np.ndarray,
    arenas: QualityFixedArenas,
    band_height: int,
) -> _EyeAnalysisResult:
    height = int(source.shape[0])
    width = int(source.shape[1])
    coverage = np.empty((height, width), dtype=np.uint8)
    pure_regions = np.empty((height, width), dtype=np.uint32)
    pure_bgr = (
        np.empty((height, width), dtype=np.uint8),
        np.empty((height, width), dtype=np.uint8),
        np.empty((height, width), dtype=np.uint8),
    )
    analysis = QualityVisibilityAnalysis(
        coverage,
        pure_regions,
        pure_bgr,
        anchor_fallbacks_seeded=True,
    )
    record_arena = arenas.records()
    record_count = 0
    for start_row in range(0, height, band_height):
        end_row = min(height, start_row + band_height)
        band = _quality_splat(renderer, source, geometry, offsets, start_row, end_row)
        built = _write_analysis_band_native(
            analysis,
            band,
            source,
            geometry.near_score,
            geometry.final_region_map,
            start_row,
            end_row,
            record_arena[record_count:],
        )
        if built.records.size:
            next_count = record_count + int(built.records.size)
            if next_count > QUALITY_REPAIR_RECORD_CAP:
                raise QualityRepairBudgetError("64 MiB repair record arena exceeded")
            built.records.setflags(write=True)
            built.records["pixel_index"] += np.uint32(start_row * width)
            built.records.setflags(write=False)
            record_count = next_count
        del band, built
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


def _accumulate_hole_run_histograms(
    lane_valid: np.ndarray,
    lane_histogram: np.ndarray,
    span_histogram: np.ndarray,
    coverage_output: np.ndarray | None = None,
) -> None:
    native_kd._require_native().accumulate_lane_hole_run_histograms_into(
        lane_valid,
        lane_histogram,
        span_histogram,
        coverage_output,
    )


def _lane_mask_weights(device: torch.device) -> torch.Tensor:
    return torch.bitwise_left_shift(
        torch.ones(HORIZONTAL_SUBPIXELS, dtype=torch.int32, device=device),
        torch.arange(HORIZONTAL_SUBPIXELS, dtype=torch.int32, device=device),
    )


def _pack_missing_lane_masks(
    lane_valid: torch.Tensor,
    lane_weights: torch.Tensor,
) -> torch.Tensor:
    valid_masks = torch.sum(
        lane_valid * lane_weights,
        dim=2,
        dtype=torch.int32,
    )
    valid_masks.bitwise_xor_(0xFFFF)
    return valid_masks.to(dtype=torch.int16)


def _render_unrepaired_eye_once(
    *,
    renderer: StereoRenderer,
    source: np.ndarray,
    geometry: QualityGeometryResult,
    offsets: np.ndarray,
    band_height: int,
) -> _UnrepairedEyeResult:
    height, width = source.shape[:2]
    output = np.empty_like(source)
    coverage = np.empty((height, width), dtype=np.uint8)
    lane_histogram = np.zeros(16 * (width + 1), dtype=np.uint64)
    span_histogram = np.zeros(width + 1, dtype=np.uint64)
    native_extension = None
    lane_weights = None
    if renderer.device.type == "cuda":
        native_extension = native_kd._require_native()
        lane_weights = _lane_mask_weights(renderer.device)
    for start_row in range(0, height, band_height):
        end_row = min(height, start_row + band_height)
        band = _quality_splat(renderer, source, geometry, offsets, start_row, end_row)
        lane_valid = band.splat.valid.reshape(
            end_row - start_row,
            width,
            HORIZONTAL_SUBPIXELS,
        )
        if native_extension is None:
            lane_valid_host = np.ascontiguousarray(lane_valid.detach().numpy())
            _accumulate_hole_run_histograms(
                lane_valid_host,
                lane_histogram,
                span_histogram,
                coverage[start_row:end_row],
            )
            del lane_valid_host
        else:
            if lane_weights is None:
                raise AssertionError("CUDA lane-mask weights are unavailable")
            packed_missing = np.ascontiguousarray(
                _pack_missing_lane_masks(lane_valid, lane_weights).cpu().numpy().view(np.uint16)
            )
            native_extension.accumulate_packed_lane_hole_run_histograms_into(
                packed_missing,
                lane_histogram,
                span_histogram,
                coverage[start_row:end_row],
            )
            del packed_missing
        downsampled = _downsample_subpixel_band(band.splat.colour)
        output[start_row:end_row] = _convert_image_band(downsampled, source.dtype)
        del band, lane_valid, downsampled
    hole_runs = HoleRunDiagnostics(
        lane_count_histogram=lane_histogram,
        touched_pixel_span_histogram=span_histogram,
    )
    return _UnrepairedEyeResult(
        output,
        coverage,
        hole_runs,
    )


def _render_none_eye_once(
    *,
    renderer: StereoRenderer,
    source: np.ndarray,
    geometry: QualityGeometryResult,
    offsets: np.ndarray,
    band_height: int,
) -> _CompactEyeResult:
    unrepaired = _render_unrepaired_eye_once(
        renderer=renderer,
        source=source,
        geometry=geometry,
        offsets=offsets,
        band_height=band_height,
    )
    return _CompactEyeResult(
        unrepaired.image,
        unrepaired.coverage_count,
        _none_repair_bits(unrepaired.coverage_count),
        None,
        unrepaired.hole_runs,
    )


def _build_background_proxy_mask(
    coverage: np.ndarray,
    *,
    far_side: Literal["left", "right"],
    foreground_exclusion_px: int,
) -> np.ndarray:
    repair_core = np.ascontiguousarray(coverage <= np.uint8(HORIZONTAL_SUBPIXELS // 2))
    exclusion_zone = np.array(repair_core, copy=True, order="C")
    far_margin = 2
    if far_side == "right":
        for distance in range(1, foreground_exclusion_px + 1):
            exclusion_zone[:, :-distance] |= repair_core[:, distance:]
        for distance in range(1, far_margin + 1):
            exclusion_zone[:, distance:] |= repair_core[:, :-distance]
    else:
        for distance in range(1, foreground_exclusion_px + 1):
            exclusion_zone[:, distance:] |= repair_core[:, :-distance]
        for distance in range(1, far_margin + 1):
            exclusion_zone[:, :-distance] |= repair_core[:, distance:]
    vertical = np.array(exclusion_zone, copy=True, order="C")
    exclusion_zone[1:] |= vertical[:-1]
    exclusion_zone[:-1] |= vertical[1:]
    return np.ascontiguousarray(exclusion_zone)


def _composite_background_proxy(  # noqa: C901 - validation stays at this boundary.
    pre_repair: np.ndarray,
    coverage: np.ndarray,
    source: np.ndarray,
    *,
    radius_px: int,
    far_side: Literal["left", "right"],
    foreground_exclusion_px: int,
) -> np.ndarray:
    if (
        not isinstance(pre_repair, np.ndarray)
        or pre_repair.dtype != np.uint8
        or pre_repair.ndim != 3
        or pre_repair.shape[2] != 3
        or not pre_repair.flags.c_contiguous
    ):
        raise TypeError("pre_repair must be a contiguous uint8 BGR raster")
    shape = (int(pre_repair.shape[0]), int(pre_repair.shape[1]))
    if (
        not isinstance(coverage, np.ndarray)
        or coverage.dtype != np.uint8
        or coverage.shape != shape
        or not coverage.flags.c_contiguous
    ):
        raise TypeError("coverage must be a matching contiguous uint8 raster")
    if (
        not isinstance(source, np.ndarray)
        or source.dtype != np.uint8
        or source.shape != pre_repair.shape
        or not source.flags.c_contiguous
    ):
        raise TypeError("source must match the pre-repair BGR raster")
    if np.any(coverage > np.uint8(HORIZONTAL_SUBPIXELS)):
        raise ValueError("coverage exceeds the horizontal subpixel count")
    if isinstance(radius_px, (bool, np.bool_)) or not isinstance(radius_px, (int, np.integer)):
        raise TypeError("radius_px must be an integer")
    radius = int(radius_px)
    if radius < 1 or radius > 3:
        raise ValueError("radius_px must be within 1..3")
    if far_side not in {"left", "right"}:
        raise ValueError("far_side must be 'left' or 'right'")
    if isinstance(foreground_exclusion_px, (bool, np.bool_)) or not isinstance(
        foreground_exclusion_px,
        (int, np.integer),
    ):
        raise TypeError("foreground_exclusion_px must be an integer")
    exclusion = int(foreground_exclusion_px)
    if exclusion < 1 or exclusion > 32:
        raise ValueError("foreground_exclusion_px must be within 1..32")

    missing = np.ascontiguousarray(coverage < np.uint8(HORIZONTAL_SUBPIXELS))
    if not bool(np.any(missing)):
        return np.array(pre_repair, copy=True, order="C")

    native = native_kd._require_native()
    proxy_source = native.normalize_background_proxy_source(pre_repair, coverage)

    mask = _build_background_proxy_mask(
        coverage,
        far_side=far_side,
        foreground_exclusion_px=exclusion,
    )
    if bool(np.all(mask)):
        proxy = np.array(source, copy=True, order="C")
    else:
        proxy = cv2.inpaint(
            proxy_source,
            np.ascontiguousarray(mask, dtype=np.uint8) * np.uint8(255),
            float(radius),
            cv2.INPAINT_TELEA,
        )

    return native.composite_background_proxy(pre_repair, coverage, proxy)


def _background_proxy_repair_state(
    coverage: np.ndarray,
) -> tuple[np.ndarray, QualityRepairPlanSummary]:
    repair_bits = np.empty_like(coverage)
    missing = np.empty_like(coverage)
    missing_lanes, missing_pixels = native_kd._require_native().build_background_proxy_state_into(
        coverage,
        repair_bits,
        missing,
    )
    missing_lanes = int(missing_lanes)
    missing_pixels = int(missing_pixels)
    component_count = 0
    if missing_pixels:
        component_count = int(cv2.connectedComponents(missing, connectivity=4)[0]) - 1
    summary = QualityRepairPlanSummary(
        segment_record_count=component_count,
        segment_table_bytes=component_count * REPAIR_RECORD_DTYPE.itemsize,
        backend_lane_counts=(0, 0, missing_lanes),
        backend_pixel_counts=(0, 0, missing_pixels),
        backend_component_counts=(0, 0, component_count),
        local_evaluated_slot_count=0,
        local_physical_sample_read_count=0,
        local_budget_skipped_run_count=0,
        fallback_indexed_donor_count=0,
        fallback_query_count=0,
        fallback_anchor_count=0,
        fallback_visited_nodes_total=0,
        fallback_visited_nodes_max=0,
        fallback_visited_nodes_p95=0.0,
        exemplar_donor_evaluation_count=0,
    )
    return repair_bits, summary


def _background_repair_bits(
    coverage: np.ndarray,
    records: np.ndarray,
    *,
    out: np.ndarray,
    backend_lane_counts_out: np.ndarray | None = None,
) -> np.ndarray:
    if (
        not isinstance(out, np.ndarray)
        or out.dtype != np.uint8
        or out.shape != coverage.shape
        or not out.flags.c_contiguous
    ):
        raise TypeError("repair-bit output must be the contiguous Pass A B plane")
    if backend_lane_counts_out is None:
        backend_lane_counts_out = np.empty(3, dtype=np.uint64)
    if (
        not isinstance(backend_lane_counts_out, np.ndarray)
        or backend_lane_counts_out.dtype != np.uint64
        or backend_lane_counts_out.shape != (3,)
        or not backend_lane_counts_out.flags.c_contiguous
        or not backend_lane_counts_out.flags.writeable
    ):
        raise TypeError("backend lane counts must be a writable uint64[3] vector")
    raw_records = records.view(np.uint8).reshape(records.size, REPAIR_RECORD_DTYPE.itemsize)
    native_kd._require_native().build_background_repair_bits_into(
        raw_records,
        coverage,
        out,
        backend_lane_counts_out,
    )
    return out


def _plan_hole_runs(records: np.ndarray, width: int) -> HoleRunDiagnostics:
    lane_histogram = np.zeros(16 * (width + 1), dtype=np.uint64)
    span_histogram = np.zeros(width + 1, dtype=np.uint64)
    raw_records = records.view(np.uint8).reshape(records.size, REPAIR_RECORD_DTYPE.itemsize)
    native_kd._require_native().build_hole_run_histograms_into(
        raw_records,
        width,
        lane_histogram,
        span_histogram,
    )
    return HoleRunDiagnostics(
        lane_count_histogram=lane_histogram,
        touched_pixel_span_histogram=span_histogram,
    )


@contextmanager
def _relative_band_records(
    records: np.ndarray,
    *,
    start_pixel: int,
    end_pixel: int,
) -> Iterator[np.ndarray]:
    pixels = records["pixel_index"]
    start = int(np.searchsorted(pixels, np.uint32(start_pixel), side="left"))
    end = int(np.searchsorted(pixels, np.uint32(end_pixel), side="left"))
    selected = records[start:end]
    selected.setflags(write=True)
    selected["pixel_index"] -= np.uint32(start_pixel)
    try:
        yield selected
    finally:
        selected["pixel_index"] += np.uint32(start_pixel)
        selected.setflags(write=False)


def _scatter_repair_records(
    colour: torch.Tensor,
    records: np.ndarray,
    *,
    width: int,
    host_colours: np.ndarray | None,
    host_mask: np.ndarray | None,
) -> int:
    if records.size == 0:
        return 0
    raw_records = records.view(np.uint8).reshape(records.size, REPAIR_RECORD_DTYPE.itemsize)
    native = native_kd._require_native()
    if colour.device.type == "cpu":
        if host_colours is not None or host_mask is not None:
            raise AssertionError("CPU compact scatter received a host staging workspace")
        output = colour.detach().numpy().reshape(-1, 3)
        return int(native.scatter_repair_records_float_into(raw_records, 0, output))
    if host_colours is None or host_mask is None:
        raise AssertionError("CUDA compact scatter has no bounded host workspace")
    total_pixels = int(colour.shape[0]) * width
    flat_colour = colour.reshape(-1, 3)
    record_pixels = records["pixel_index"]
    scattered = 0
    for pixel_start in range(0, total_pixels, QUALITY_SCATTER_CHUNK_PIXELS):
        pixel_end = min(total_pixels, pixel_start + QUALITY_SCATTER_CHUNK_PIXELS)
        record_start = int(np.searchsorted(record_pixels, np.uint32(pixel_start), side="left"))
        record_end = int(np.searchsorted(record_pixels, np.uint32(pixel_end), side="left"))
        if record_start == record_end:
            continue
        lane_capacity = (pixel_end - pixel_start) * HORIZONTAL_SUBPIXELS
        colour_staging = host_colours[:lane_capacity]
        mask_staging = host_mask[:lane_capacity]
        chunk_raw = raw_records[record_start:record_end]
        scattered += int(
            native.materialize_repair_scatter_u8_into(
                chunk_raw,
                pixel_start,
                colour_staging,
                mask_staging,
            )
        )
        device_colours = torch.from_numpy(colour_staging).to(
            device=colour.device,
            dtype=colour.dtype,
        )
        device_mask = torch.from_numpy(mask_staging).to(
            device=colour.device,
            dtype=torch.bool,
        )
        target = flat_colour[pixel_start * HORIZONTAL_SUBPIXELS : pixel_end * HORIZONTAL_SUBPIXELS]
        target.copy_(torch.where(device_mask[:, None], device_colours, target))
        del device_colours, device_mask, target
    return scattered


def _replay_background_eye_once(
    *,
    renderer: StereoRenderer,
    source: np.ndarray,
    geometry: QualityGeometryResult,
    offsets: np.ndarray,
    records: np.ndarray,
    band_height: int,
    planned_lane_count: int,
) -> np.ndarray:
    height, width = source.shape[:2]
    output = np.empty_like(source)
    scattered = 0
    if renderer.device.type == "cpu":
        host_colours = None
        host_mask = None
    else:
        staging_pixels = min(
            band_height * width,
            QUALITY_SCATTER_CHUNK_PIXELS,
        )
        staging_lanes = staging_pixels * HORIZONTAL_SUBPIXELS
        host_colours = np.empty((staging_lanes, 3), dtype=np.uint8)
        host_mask = np.empty(staging_lanes, dtype=np.bool_)
    for start_row in range(0, height, band_height):
        end_row = min(height, start_row + band_height)
        band = _quality_splat(renderer, source, geometry, offsets, start_row, end_row)
        valid = np.ascontiguousarray(band.splat.valid.detach().cpu().numpy())
        with _relative_band_records(
            records,
            start_pixel=start_row * width,
            end_pixel=end_row * width,
        ) as band_records:
            validate_record_coverage(band_records, ~valid)
            scattered += _scatter_repair_records(
                band.splat.colour,
                band_records,
                width=width,
                host_colours=host_colours,
                host_mask=host_mask,
            )
        downsampled = _downsample_subpixel_band(band.splat.colour)
        output[start_row:end_row] = _convert_image_band(downsampled, source.dtype)
        del band, valid, downsampled
    if scattered != planned_lane_count:
        raise RuntimeError("Quality Pass B scatter total does not match the repair plan")
    return output


def _render_background_eye(
    *,
    renderer: StereoRenderer,
    source: np.ndarray,
    geometry: QualityGeometryResult,
    offsets: np.ndarray,
    eye: Literal["left", "right"],
    controls_limit_px: int,
    band_state: _FrameBandState,
) -> _CompactEyeResult:
    pre_repair = _retry_after_oom(
        renderer,
        band_state,
        _render_unrepaired_eye_once,
        renderer=renderer,
        source=source,
        geometry=geometry,
        offsets=offsets,
    )
    safe_limit = max(1, math.floor(controls_limit_px * source.shape[0] / 1080.0 + 0.5))
    local_limit = min(safe_limit, geometry.predicted_gap_px)
    image = _composite_background_proxy(
        pre_repair.image,
        pre_repair.coverage_count,
        source,
        radius_px=min(3, max(1, local_limit)),
        far_side=eye,
        foreground_exclusion_px=local_limit,
    )
    repair_bits, summary = _background_proxy_repair_state(
        pre_repair.coverage_count,
    )
    return _CompactEyeResult(
        image,
        pre_repair.coverage_count,
        repair_bits,
        summary,
        pre_repair.hole_runs,
    )


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
            "max_neighbour_abs_q_jump_px": float(geometry.max_neighbour_abs_q_jump_px),
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
    initial_height = _calculate_quality_band_height(
        width,
        height,
        renderer.temporary_budget_bytes,
    )
    band_state = _FrameBandState(
        initial_height=initial_height,
        retry_height=max(1, initial_height // 2),
        current_height=initial_height,
    )
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
                eye=eye,
                controls_limit_px=controls.occlusion_fill_max_px,
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
    frame_values = np.asarray(frame)
    canonical_input = np.asarray(canonical)
    if frame_values.ndim != 3 or frame_values.shape[2] != 3 or canonical_input.ndim != 2:
        raise TypeError("Quality rendering requires BGR frames and a 2D canonical map")
    _quality_public_memory_preflight(
        render_shape=(int(frame_values.shape[0]), int(frame_values.shape[1])),
        native_shape=(int(canonical_input.shape[0]), int(canonical_input.shape[1])),
        occlusion_fill=settings.occlusion_fill,
        materialize_public_masks=False,
    )
    source = np.ascontiguousarray(frame_values)
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
    indexed_sample_count = int(canonical_values.size)
    del canonical, canonical_input, canonical_values, frame, frame_values
    initial_height = _calculate_quality_band_height(
        width,
        height,
        renderer.temporary_budget_bytes,
    )
    band_state = _FrameBandState(
        initial_height=initial_height,
        retry_height=max(1, initial_height // 2),
        current_height=initial_height,
    )
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
                eye=eye,
                controls_limit_px=settings.occlusion_fill_max_px,
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
                "max_neighbour_abs_q_jump_px": float(geometry.max_neighbour_abs_q_jump_px),
                "predicted_gap_px": int(geometry.predicted_gap_px),
                "local_limit_px": int(local_limit),
            }
            for eye in ("left", "right")
        }
        if settings.occlusion_fill == "background"
        else None
    )
    result = CompactStereoRenderResult(
        left_image=left.image,
        right_image=right.image,
        left_coverage_count=left.coverage_count,
        right_coverage_count=right.coverage_count,
        left_repair_bits=left.repair_bits,
        right_repair_bits=right.repair_bits,
        left_plan=left.plan,
        right_plan=right.plan,
        geometry_nearest={
            "indexed_sample_count": indexed_sample_count,
            "query_count": int(geometry.geometry_query_count),
            "visited_nodes_total": int(geometry.geometry_visited_nodes_total),
            "visited_nodes_max": int(geometry.geometry_visited_nodes_max),
        },
        quality_limits=quality_limits,
        left_hole_runs=left.hole_runs,
        right_hole_runs=right.hole_runs,
    )
    del source, geometry, eyes, left, right
    return result


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
    frame_values = np.asarray(frame)
    if frame_values.ndim != 3 or frame_values.shape[2] != 3:
        raise TypeError("Quality rendering requires uint8 BGR input")
    metric = primitives.metric
    for name in ("inverse_depth", "valid", "focal_x_normalized"):
        if not hasattr(metric, name):
            raise TypeError("metric primitive is missing native geometry fields")
    metric_shape = np.asarray(metric.inverse_depth).shape
    if len(metric_shape) != 2:
        raise TypeError("metric inverse depth must be two-dimensional")
    _quality_public_memory_preflight(
        render_shape=(int(frame_values.shape[0]), int(frame_values.shape[1])),
        native_shape=(int(metric_shape[0]), int(metric_shape[1])),
        occlusion_fill=controls.occlusion_fill,
        materialize_public_masks=False,
    )
    source = np.ascontiguousarray(frame_values)
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
        raise TypeError("Quality rendering requires uint8 BGR input")
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
    indexed_sample_count = int(metric.inverse_depth.size)
    del frame, frame_values, primitives, metric
    left, right = _render_metric_quality_eyes(
        renderer=renderer,
        source=source,
        geometry=geometry,
        controls=controls,
    )
    if geometry.metric_stats is None:
        raise RuntimeError("Metric Quality geometry did not produce projection statistics")
    metric_stats = geometry.metric_stats
    geometry_nearest = {
        "indexed_sample_count": indexed_sample_count,
        "query_count": int(geometry.geometry_query_count),
        "visited_nodes_total": int(geometry.geometry_visited_nodes_total),
        "visited_nodes_max": int(geometry.geometry_visited_nodes_max),
    }
    quality_limits = _quality_limits(
        geometry,
        height=height,
        occlusion_fill=controls.occlusion_fill,
        occlusion_fill_max_px=controls.occlusion_fill_max_px,
    )
    result = CompactStereoRenderResult(
        left_image=left.image,
        right_image=right.image,
        left_coverage_count=left.coverage_count,
        right_coverage_count=right.coverage_count,
        left_repair_bits=left.repair_bits,
        right_repair_bits=right.repair_bits,
        left_plan=left.plan,
        right_plan=right.plan,
        metric_stats=metric_stats,
        geometry_nearest=geometry_nearest,
        quality_limits=quality_limits,
        left_hole_runs=left.hole_runs,
        right_hole_runs=right.hole_runs,
    )
    del source, geometry, left, right
    return result


def render_relative_quality_public(
    *,
    renderer: StereoRenderer,
    frame: np.ndarray,
    canonical: np.ndarray,
    settings: StereoRenderSettings,
) -> StereoRenderResult:
    """Materialize the four legacy boolean masks for the public API."""

    frame_values = np.asarray(frame)
    canonical_values = np.asarray(canonical)
    if frame_values.ndim != 3 or canonical_values.ndim != 2:
        raise TypeError("Quality rendering requires BGR frames and a 2D canonical map")
    _quality_public_memory_preflight(
        render_shape=(int(frame_values.shape[0]), int(frame_values.shape[1])),
        native_shape=(int(canonical_values.shape[0]), int(canonical_values.shape[1])),
        occlusion_fill=settings.occlusion_fill,
        materialize_public_masks=True,
    )
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
