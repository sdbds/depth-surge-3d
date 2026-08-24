"""Deterministic one-sided Quality geometry and lifetime accounting."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterator, Literal

import numpy as np

from . import native_kd, native_region_solver, region_solver
from .implicit_kd import ImplicitRegionKdIndex, VisitBudget
from ..stereo_geometry import (
    MetricProjectionStats,
    _resize_float32_bilinear,
)


QUALITY_GEOMETRY_NEAREST_VISIT_CAP = 268_435_456
_UINT32_MAX = int(np.iinfo(np.uint32).max)
_INT32_MIN = int(np.iinfo(np.int32).min)
_INT32_MAX = int(np.iinfo(np.int32).max)


@dataclass(frozen=True)
class GeometryLifetimeEvent:
    """One ordered allocation, release, or semantic checkpoint."""

    ordinal: int
    step: int
    action: Literal["allocate", "release", "checkpoint"]
    name: str
    byte_count: int
    live_bytes: int


@dataclass(frozen=True)
class GeometryLifetimeTrace:
    """Frozen allocation trace and its measured semantic live-byte peak."""

    events: tuple[GeometryLifetimeEvent, ...]
    peak_live_bytes: int


@dataclass(frozen=True)
class QualityGeometryResult:
    """Final geometry plus region/index evidence from the ten-step prototype."""

    near_score: np.ndarray
    total_disparity_fraction: np.ndarray
    source_valid: np.ndarray
    source_region_map: np.ndarray
    final_region_map: np.ndarray
    metric_valid: np.ndarray | None
    metric_stats: MetricProjectionStats | None
    edge_band_pixel_count: int
    selected_region_mismatch_count: int
    geometry_query_count: int
    geometry_visited_nodes_total: int
    geometry_visited_nodes_max: int
    geometry_index_bytes: int
    max_neighbour_abs_q_jump_px: float
    predicted_gap_px: int
    final_region_hash_before_resample: str
    final_region_hash_after_resample: str
    lifetime: GeometryLifetimeTrace


@dataclass(frozen=True)
class _InterpolationWeights:
    coordinates: tuple[tuple[int, int], ...]
    retained_weights: tuple[np.float64, ...]
    retained: np.float64
    source_y: np.float64
    source_x: np.float64


@dataclass
class _MetricBaseline:
    native_weight: np.ndarray
    native_weighted_inverse: np.ndarray
    resized_weight: np.ndarray
    resized_weighted_inverse: np.ndarray
    near_score: np.ndarray
    resized_valid: np.ndarray
    total_disparity_fraction: np.ndarray
    stats: MetricProjectionStats


@dataclass(frozen=True)
class _RegionState:
    source_region_map: np.ndarray
    final_region_map: np.ndarray
    region_count: int
    radius: int
    edge_band_pixel_count: int
    final_region_hash: str


@dataclass(frozen=True)
class _ResampleDiagnostics:
    query_count: int
    visited_nodes_total: int
    visited_nodes_max: int
    index_bytes: int


class _LifetimeRecorder:
    def __init__(self) -> None:
        self._events: list[GeometryLifetimeEvent] = []
        self._live: dict[str, int] = {}
        self._live_bytes = 0
        self._peak_live_bytes = 0

    def allocate(self, step: int, name: str, byte_count: int) -> None:
        if name in self._live:
            raise RuntimeError(f"allocation {name!r} is already live")
        size = _require_nonnegative_integer("byte_count", byte_count)
        self._live[name] = size
        self._live_bytes += size
        self._peak_live_bytes = max(self._peak_live_bytes, self._live_bytes)
        self._append(step, "allocate", name, size)

    def release(self, step: int, name: str) -> None:
        if name not in self._live:
            raise RuntimeError(f"allocation {name!r} is not live")
        size = self._live.pop(name)
        self._live_bytes -= size
        self._append(step, "release", name, size)

    def checkpoint(self, step: int, name: str) -> None:
        self._append(step, "checkpoint", name, 0)

    def freeze(self) -> GeometryLifetimeTrace:
        return GeometryLifetimeTrace(tuple(self._events), self._peak_live_bytes)

    def _append(
        self,
        step: int,
        action: Literal["allocate", "release", "checkpoint"],
        name: str,
        byte_count: int,
    ) -> None:
        self._events.append(
            GeometryLifetimeEvent(
                ordinal=len(self._events),
                step=step,
                action=action,
                name=name,
                byte_count=byte_count,
                live_bytes=self._live_bytes,
            )
        )


def one_sided_resample_scalar(
    values: np.ndarray,
    source_regions: np.ndarray,
    *,
    selected_region: int,
    render_shape: tuple[int, int],
    output_coordinate: tuple[int, int],
) -> np.float32 | None:
    """Apply the exact ordered binary64 oracle or report retained zero."""

    _validate_native_primitive("values", values)
    _validate_source_regions(
        source_regions,
        (int(values.shape[0]), int(values.shape[1])),
    )
    region = _validate_region(selected_region, int(source_regions.max()))
    destination_shape = _validate_shape("render_shape", render_shape)
    output_row, output_column = _validate_output_coordinate(
        output_coordinate,
        destination_shape,
    )
    weights = _one_sided_weights(
        source_regions,
        region,
        destination_shape,
        output_row,
        output_column,
    )
    if weights.retained == np.float64(0.0):
        return None
    return _apply_one_sided_weights(values, weights)


def materialize_streamed_edge_band(
    source_regions: np.ndarray,
    *,
    render_shape: tuple[int, int],
    radius: int,
) -> np.ndarray:
    """Materialize the bounded-row classifier only for oracle comparison."""

    _validate_source_regions(source_regions)
    destination_shape = _validate_shape("render_shape", render_shape)
    dilation_radius = _require_nonnegative_integer("radius", radius)
    result = np.zeros(destination_shape, dtype=np.bool_)
    for row, edge_row in _iter_streamed_edge_rows(
        source_regions,
        destination_shape,
        dilation_radius,
    ):
        result[row] = edge_row
    result.setflags(write=False)
    return result


def build_relative_eye_offsets_into(
    near_score: np.ndarray,
    *,
    stereo_strength: float,
    convergence: float,
    eye: Literal["left", "right"],
    output_int32: np.ndarray,
    row_float64_scratch: np.ndarray,
) -> None:
    """Build one relative eye map with the exact current Fast scalar order."""

    _validate_offset_buffers(near_score, output_int32, row_float64_scratch)
    strength = _validate_float("stereo_strength", stereo_strength, 0.0, 5.0)
    convergence64 = _validate_float("convergence", convergence, 0.0, 1.0)
    direction = _validate_eye(eye)
    width = near_score.shape[1]
    scale = np.float64(np.float64(width) * strength)
    scale = np.float64(scale / np.float64(200.0))
    for row in range(near_score.shape[0]):
        np.copyto(row_float64_scratch, near_score[row], casting="unsafe")
        np.subtract(
            row_float64_scratch,
            convergence64,
            out=row_float64_scratch,
        )
        np.multiply(row_float64_scratch, scale, out=row_float64_scratch)
        np.multiply(
            row_float64_scratch,
            np.float64(16.0),
            out=row_float64_scratch,
        )
        if direction < 0:
            np.negative(row_float64_scratch, out=row_float64_scratch)
        np.subtract(
            row_float64_scratch,
            np.float64(0.5),
            out=row_float64_scratch,
        )
        np.ceil(row_float64_scratch, out=row_float64_scratch)
        _narrow_offset_row(row_float64_scratch, output_int32[row])


def build_geometry_eye_offsets_into(
    total_disparity_fraction: np.ndarray,
    *,
    eye: Literal["left", "right"],
    output_int32: np.ndarray,
    row_float64_scratch: np.ndarray,
) -> None:
    """Build one metric/common-geometry eye map in the frozen operation order."""

    _validate_geometry_offset_buffers(
        total_disparity_fraction,
        output_int32,
        row_float64_scratch,
    )
    direction = _validate_eye(eye)
    width = total_disparity_fraction.shape[1]
    for row in range(total_disparity_fraction.shape[0]):
        np.multiply(
            total_disparity_fraction[row],
            np.float64(width),
            out=row_float64_scratch,
        )
        np.multiply(
            row_float64_scratch,
            np.float64(0.5),
            out=row_float64_scratch,
        )
        np.multiply(
            row_float64_scratch,
            np.float64(16.0),
            out=row_float64_scratch,
        )
        if direction < 0:
            np.negative(row_float64_scratch, out=row_float64_scratch)
        np.subtract(
            row_float64_scratch,
            np.float64(0.5),
            out=row_float64_scratch,
        )
        np.ceil(row_float64_scratch, out=row_float64_scratch)
        _narrow_offset_row(row_float64_scratch, output_int32[row])


def build_relative_quality_geometry(
    guide_bgr: np.ndarray,
    canonical: np.ndarray,
    *,
    render_shape: tuple[int, int],
    stereo_strength: float,
    convergence: float,
    geometry_visit_cap: int = QUALITY_GEOMETRY_NEAREST_VISIT_CAP,
) -> QualityGeometryResult:
    """Execute the ten frozen Quality steps for a relative primitive."""

    destination_shape = _validate_geometry_inputs(
        guide_bgr,
        canonical,
        render_shape,
    )
    if np.any(canonical < np.float32(0.0)) or np.any(canonical > np.float32(1.0)):
        raise ValueError("canonical must lie within [0, 1]")
    strength = _validate_float("stereo_strength", stereo_strength, 0.0, 5.0)
    convergence64 = _validate_float("convergence", convergence, 0.0, 1.0)
    visit_cap = _require_nonnegative_integer("geometry_visit_cap", geometry_visit_cap)
    recorder = _LifetimeRecorder()

    low_displacement = _relative_displacement(
        canonical,
        destination_shape[1],
        strength,
        convergence64,
    )
    low_regions = region_solver.build_low_resolution_regions(
        low_displacement,
        np.ones(canonical.shape, dtype=np.bool_),
        canonical,
    )
    recorder.checkpoint(1, "low_resolution_regions_complete")

    near_score = _resize_float32_bilinear(canonical, destination_shape)
    near_score = np.array(near_score, copy=True, order="C")
    total = _derive_relative_total(
        near_score,
        strength,
        convergence64,
    )
    source_valid = np.ones(destination_shape, dtype=np.bool_)
    recorder.allocate(2, "fast_baseline", near_score.nbytes + total.nbytes + source_valid.nbytes)
    recorder.checkpoint(2, "fast_baseline_complete")

    regions = _solve_final_regions(
        guide_bgr,
        low_regions,
        destination_shape,
        recorder,
    )
    index = native_kd.build_native_implicit_region_kd(
        regions.source_region_map,
        region_count=regions.region_count,
    )
    recorder.allocate(6, "retained_zero_index", index.index_bytes)
    recorder.checkpoint(7, "final_geodesic_region_selection")
    budget = VisitBudget(visit_cap)
    diagnostics = _resample_primitives_in_place(
        (canonical,),
        (near_score,),
        regions,
        index,
        budget,
        recorder,
    )
    recorder.checkpoint(8, "primitive_resample_complete")
    recorder.release(9, "retained_zero_index")
    del index

    _update_relative_total_inside_band(
        total,
        near_score,
        regions.source_region_map,
        regions.radius,
        strength,
        convergence64,
    )
    recorder.checkpoint(10, "derived_geometry_complete")
    displacement = _relative_displacement(
        near_score,
        destination_shape[1],
        strength,
        convergence64,
    )
    result = _freeze_quality_result(
        near_score=near_score,
        total=total,
        source_valid=source_valid,
        region_state=regions,
        metric_valid=None,
        metric_stats=None,
        diagnostics=diagnostics,
        displacement=displacement,
        recorder=recorder,
    )
    return result


def build_metric_quality_geometry(
    guide_bgr: np.ndarray,
    inverse_depth: np.ndarray,
    valid: np.ndarray,
    focal_x_normalized: np.float32,
    *,
    render_shape: tuple[int, int],
    virtual_baseline_mm: float,
    convergence_distance_m: float,
    max_disparity_percent: float,
    retained_crop_width: int,
    geometry_visit_cap: int = QUALITY_GEOMETRY_NEAREST_VISIT_CAP,
) -> QualityGeometryResult:
    """Execute primitive resampling before metric projection and clamp stats."""

    destination_shape = _validate_metric_geometry_inputs(
        guide_bgr,
        inverse_depth,
        valid,
        focal_x_normalized,
        render_shape,
    )
    focal = np.float64(focal_x_normalized)
    baseline_mm = _validate_float("virtual_baseline_mm", virtual_baseline_mm, 0.0, 100.0)
    convergence_m = _validate_positive_float(
        "convergence_distance_m",
        convergence_distance_m,
    )
    disparity_percent = _validate_float(
        "max_disparity_percent",
        max_disparity_percent,
        0.0,
        100.0,
    )
    crop_width = _validate_crop_width(retained_crop_width, destination_shape[1])
    visit_cap = _require_nonnegative_integer("geometry_visit_cap", geometry_visit_cap)
    recorder = _LifetimeRecorder()

    native_weight = np.ascontiguousarray(valid, dtype=np.float32)
    native_weighted = np.ascontiguousarray(inverse_depth * native_weight, dtype=np.float32)
    low_displacement, low_near = _metric_low_resolution_fields(
        native_weighted,
        valid,
        focal,
        destination_shape[1],
        baseline_mm,
        convergence_m,
        disparity_percent,
        crop_width,
    )
    low_regions = region_solver.build_low_resolution_regions(
        low_displacement,
        valid,
        low_near,
    )
    recorder.checkpoint(1, "low_resolution_regions_complete")

    baseline = _build_metric_baseline(
        native_weight,
        native_weighted,
        destination_shape,
        focal,
        baseline_mm,
        convergence_m,
        disparity_percent,
        crop_width,
    )
    baseline_bytes = sum(
        values.nbytes
        for values in (
            baseline.resized_weight,
            baseline.resized_weighted_inverse,
            baseline.near_score,
            baseline.resized_valid,
            baseline.total_disparity_fraction,
        )
    )
    recorder.allocate(2, "fast_baseline", baseline_bytes)
    recorder.checkpoint(2, "fast_baseline_complete")

    regions = _solve_final_regions(
        guide_bgr,
        low_regions,
        destination_shape,
        recorder,
    )
    index = native_kd.build_native_implicit_region_kd(
        regions.source_region_map,
        region_count=regions.region_count,
    )
    recorder.allocate(6, "retained_zero_index", index.index_bytes)
    recorder.checkpoint(7, "final_geodesic_region_selection")
    budget = VisitBudget(visit_cap)
    diagnostics = _resample_primitives_in_place(
        (baseline.native_weight, baseline.native_weighted_inverse),
        (baseline.resized_weight, baseline.resized_weighted_inverse),
        regions,
        index,
        budget,
        recorder,
    )
    recorder.checkpoint(8, "primitive_resample_complete")
    recorder.release(9, "retained_zero_index")
    del index

    _update_metric_derived_inside_band(
        baseline,
        regions,
        focal,
        baseline_mm,
        convergence_m,
        disparity_percent,
        crop_width,
    )
    baseline.stats = _recount_metric_stats(
        baseline.near_score,
        baseline.resized_valid,
        focal,
        baseline_mm,
        convergence_m,
        disparity_percent,
        crop_width,
    )
    recorder.checkpoint(10, "derived_geometry_complete")
    displacement = _geometry_displacement(
        baseline.total_disparity_fraction,
        destination_shape[1],
    )
    return _freeze_quality_result(
        near_score=baseline.near_score,
        total=baseline.total_disparity_fraction,
        source_valid=np.ones(destination_shape, dtype=np.bool_),
        region_state=regions,
        metric_valid=baseline.resized_valid,
        metric_stats=baseline.stats,
        diagnostics=diagnostics,
        displacement=displacement,
        recorder=recorder,
    )


def _solve_final_regions(
    guide_bgr: np.ndarray,
    low_regions: region_solver.LowResolutionRegionResult,
    render_shape: tuple[int, int],
    recorder: _LifetimeRecorder,
) -> _RegionState:
    source_region_map = low_regions.region_map
    region_count = int(low_regions.region_rank_bits.size) - 1
    initial = region_solver.nearest_label_upsample(
        source_region_map,
        render_height=render_shape[0],
        render_width=render_shape[1],
    )
    radius = region_solver.edge_band_radius(
        geometry_height=source_region_map.shape[0],
        geometry_width=source_region_map.shape[1],
        render_height=render_shape[0],
        render_width=render_shape[1],
    )
    edge_band = region_solver.build_four_connected_edge_band(initial, radius=radius)
    seeds = region_solver.build_sparse_region_seeds(initial, edge_band, radius=radius)
    morphology_bytes = initial.nbytes + edge_band.nbytes + seeds.nbytes
    recorder.allocate(3, "morphology_scratch", morphology_bytes)
    recorder.checkpoint(3, "nearest_label_map_complete")
    geodesic = native_region_solver.solve_native_geodesic_regions(
        guide_bgr,
        edge_band,
        seeds,
        low_regions.region_rank_bits,
    )
    solver_bytes = (
        geodesic.distance.nbytes + geodesic.heap_position.nbytes + geodesic.heap_pixel_bytes
    )
    recorder.allocate(4, "geodesic_solver_scratch", solver_bytes)
    recorder.checkpoint(4, "final_geodesic_region_map_complete")
    final_region_map = geodesic.region_map
    recorder.allocate(4, "final_region_map", final_region_map.nbytes)
    edge_count = int(np.count_nonzero(edge_band))
    final_hash = _array_sha256(final_region_map)
    del initial, edge_band, seeds, geodesic, low_regions
    recorder.release(5, "geodesic_solver_scratch")
    recorder.release(5, "morphology_scratch")
    recorder.checkpoint(5, "region_scratch_released")
    return _RegionState(
        source_region_map=source_region_map,
        final_region_map=final_region_map,
        region_count=region_count,
        radius=radius,
        edge_band_pixel_count=edge_count,
        final_region_hash=final_hash,
    )


def _resample_primitives_in_place(
    source_primitives: tuple[np.ndarray, ...],
    output_primitives: tuple[np.ndarray, ...],
    regions: _RegionState,
    index: ImplicitRegionKdIndex,
    budget: VisitBudget,
    recorder: _LifetimeRecorder,
) -> _ResampleDiagnostics:
    if len(source_primitives) != len(output_primitives) or not source_primitives:
        raise ValueError("source and output primitive tuples must match")
    render_shape = (
        int(regions.final_region_map.shape[0]),
        int(regions.final_region_map.shape[1]),
    )
    query_capacity = regions.edge_band_pixel_count
    recorder.allocate(7, "retained_zero_query_scratch", 40 * query_capacity)
    query_pixels = np.empty(query_capacity, dtype=np.uint32)
    query_regions = np.empty(query_capacity, dtype=np.uint32)
    query_y = np.empty(query_capacity, dtype=np.float64)
    query_x = np.empty(query_capacity, dtype=np.float64)
    query_count, observed_edge_count = _collect_one_sided_queries(
        source_primitives,
        output_primitives,
        regions,
        query_pixels,
        query_regions,
        query_y,
        query_x,
    )
    if observed_edge_count != regions.edge_band_pixel_count:
        raise RuntimeError("streamed edge classifier disagrees with the solver band")

    visits = _resolve_retained_zero_queries(
        source_primitives,
        output_primitives,
        render_shape,
        index,
        budget,
        query_pixels,
        query_regions,
        query_y,
        query_x,
        query_count,
    )
    recorder.release(8, "retained_zero_query_scratch")
    return _ResampleDiagnostics(
        query_count=query_count,
        visited_nodes_total=int(visits.sum(dtype=np.uint64)),
        visited_nodes_max=int(visits.max()) if visits.size else 0,
        index_bytes=index.index_bytes,
    )


def _collect_one_sided_queries(
    source_primitives: tuple[np.ndarray, ...],
    output_primitives: tuple[np.ndarray, ...],
    regions: _RegionState,
    query_pixels: np.ndarray,
    query_regions: np.ndarray,
    query_y: np.ndarray,
    query_x: np.ndarray,
) -> tuple[int, int]:
    query_count, edge_count = native_kd._require_native().collect_one_sided_queries(
        source_primitives,
        output_primitives,
        regions.source_region_map,
        regions.final_region_map,
        regions.radius,
        query_pixels,
        query_regions,
        query_y,
        query_x,
    )
    return int(query_count), int(edge_count)


def _resolve_retained_zero_queries(
    source_primitives: tuple[np.ndarray, ...],
    output_primitives: tuple[np.ndarray, ...],
    render_shape: tuple[int, int],
    index: ImplicitRegionKdIndex,
    budget: VisitBudget,
    query_pixels: np.ndarray,
    query_regions: np.ndarray,
    query_y: np.ndarray,
    query_x: np.ndarray,
    query_count: int,
) -> np.ndarray:
    if query_count == 0:
        return np.empty(0, dtype=np.uint32)
    result = native_kd.query_native_geometry_batch(
        index,
        np.ascontiguousarray(query_regions[:query_count]),
        np.ascontiguousarray(query_y[:query_count]),
        np.ascontiguousarray(query_x[:query_count]),
        budget=budget,
    )
    if not np.all(result.found):
        raise RuntimeError("selected source region has no retained-zero member")
    for query_index in range(query_count):
        output_pixel = int(query_pixels[query_index])
        output_row, output_column = divmod(output_pixel, render_shape[1])
        source_pixel = int(result.sample_index[query_index])
        for source, output in zip(
            source_primitives,
            output_primitives,
            strict=True,
        ):
            output[output_row, output_column] = source.flat[source_pixel]
    return result.visited_nodes


def _one_sided_weights(
    source_regions: np.ndarray,
    selected_region: int,
    render_shape: tuple[int, int],
    output_row: int,
    output_column: int,
) -> _InterpolationWeights:
    source_height, source_width = source_regions.shape
    render_height, render_width = render_shape
    source_x = _source_coordinate(output_column, source_width, render_width)
    source_y = _source_coordinate(output_row, source_height, render_height)
    clipped_x = min(max(source_x, np.float64(0.0)), np.float64(source_width - 1))
    clipped_y = min(max(source_y, np.float64(0.0)), np.float64(source_height - 1))
    x0 = int(math.floor(float(clipped_x)))
    x1 = min(x0 + 1, source_width - 1)
    y0 = int(math.floor(float(clipped_y)))
    y1 = min(y0 + 1, source_height - 1)
    weight_x = np.float64(clipped_x - np.float64(x0))
    weight_y = np.float64(clipped_y - np.float64(y0))
    opposite_x = np.float64(np.float64(1.0) - weight_x)
    opposite_y = np.float64(np.float64(1.0) - weight_y)
    weights = (
        np.float64(opposite_y * opposite_x),
        np.float64(opposite_y * weight_x),
        np.float64(weight_y * opposite_x),
        np.float64(weight_y * weight_x),
    )
    coordinates = ((y0, x0), (y0, x1), (y1, x0), (y1, x1))
    retained_weights = tuple(
        np.float64(
            weight * np.float64(1.0 if int(source_regions[row, column]) == selected_region else 0.0)
        )
        for weight, (row, column) in zip(weights, coordinates, strict=True)
    )
    retained = np.float64(retained_weights[0] + retained_weights[1])
    retained = np.float64(retained + retained_weights[2])
    retained = np.float64(retained + retained_weights[3])
    return _InterpolationWeights(
        coordinates=coordinates,
        retained_weights=retained_weights,
        retained=retained,
        source_y=source_y,
        source_x=source_x,
    )


def _apply_one_sided_weights(
    values: np.ndarray,
    weights: _InterpolationWeights,
) -> np.float32:
    terms = tuple(
        np.float64(np.float64(values[row, column]) * weight)
        for weight, (row, column) in zip(
            weights.retained_weights,
            weights.coordinates,
            strict=True,
        )
    )
    numerator = np.float64(terms[0] + terms[1])
    numerator = np.float64(numerator + terms[2])
    numerator = np.float64(numerator + terms[3])
    return np.float32(np.float64(numerator / weights.retained))


def _iter_streamed_edge_rows(
    source_regions: np.ndarray,
    render_shape: tuple[int, int],
    radius: int,
) -> Iterator[tuple[int, np.ndarray]]:
    render_height, render_width = render_shape
    source_columns = _mapped_source_columns(source_regions.shape[1], render_width)
    label_cache: dict[int, np.ndarray] = {}
    for row in range(render_height):
        edge = np.zeros(render_width, dtype=np.bool_)
        for boundary_y in range(max(0, row - radius), min(render_height, row + radius + 1)):
            horizontal_radius = radius - abs(boundary_y - row)
            boundary = _streamed_boundary_row(
                source_regions,
                source_columns,
                label_cache,
                boundary_y,
                render_shape,
            )
            _dilate_boundary_row(edge, boundary, horizontal_radius)
        _evict_label_rows(label_cache, row - radius - 2)
        yield row, edge


def _mapped_source_columns(source_width: int, render_width: int) -> np.ndarray:
    result = np.empty(render_width, dtype=np.uint32)
    for column in range(render_width):
        result[column] = np.uint32(_nearest_source_index(column, source_width, render_width))
    return result


def _cached_label_row(
    source_regions: np.ndarray,
    source_columns: np.ndarray,
    cache: dict[int, np.ndarray],
    row: int,
    render_height: int,
) -> np.ndarray:
    if row not in cache:
        source_row = _nearest_source_index(row, source_regions.shape[0], render_height)
        cache[row] = np.ascontiguousarray(
            source_regions[source_row, source_columns],
            dtype=np.uint32,
        )
    return cache[row]


def _streamed_boundary_row(
    source_regions: np.ndarray,
    source_columns: np.ndarray,
    cache: dict[int, np.ndarray],
    row: int,
    render_shape: tuple[int, int],
) -> np.ndarray:
    render_height, render_width = render_shape
    labels = _cached_label_row(
        source_regions,
        source_columns,
        cache,
        row,
        render_height,
    )
    boundary = np.zeros(render_width, dtype=np.bool_)
    different = labels[:-1] != labels[1:]
    boundary[:-1] |= different
    boundary[1:] |= different
    if row:
        boundary |= labels != _cached_label_row(
            source_regions,
            source_columns,
            cache,
            row - 1,
            render_height,
        )
    if row + 1 < render_height:
        boundary |= labels != _cached_label_row(
            source_regions,
            source_columns,
            cache,
            row + 1,
            render_height,
        )
    return boundary


def _dilate_boundary_row(
    edge: np.ndarray,
    boundary: np.ndarray,
    horizontal_radius: int,
) -> None:
    for delta in range(-horizontal_radius, horizontal_radius + 1):
        if delta < 0:
            edge[:delta] |= boundary[-delta:]
        elif delta > 0:
            edge[delta:] |= boundary[:-delta]
        else:
            edge |= boundary


def _evict_label_rows(cache: dict[int, np.ndarray], minimum_live_row: int) -> None:
    for cached_row in tuple(cache):
        if cached_row < minimum_live_row:
            del cache[cached_row]


def _source_coordinate(
    output_index: int,
    source_size: int,
    destination_size: int,
) -> np.float64:
    coordinate = np.float64(output_index)
    coordinate = np.float64(coordinate + np.float64(0.5))
    coordinate = np.float64(coordinate * np.float64(source_size))
    coordinate = np.float64(coordinate / np.float64(destination_size))
    return np.float64(coordinate - np.float64(0.5))


def _nearest_source_index(
    output_index: int,
    source_size: int,
    destination_size: int,
) -> int:
    coordinate = _source_coordinate(output_index, source_size, destination_size)
    nearest = int(math.floor(float(np.float64(coordinate + np.float64(0.5)))))
    return min(max(nearest, 0), source_size - 1)


def _relative_displacement(
    near_score: np.ndarray,
    render_width: int,
    strength: np.float64,
    convergence: np.float64,
) -> np.ndarray:
    scale = np.float64(np.float64(render_width) * strength)
    scale = np.float64(scale / np.float64(200.0))
    displacement = near_score.astype(np.float64)
    np.subtract(displacement, convergence, out=displacement)
    np.multiply(displacement, scale, out=displacement)
    return np.ascontiguousarray(displacement)


def _derive_relative_total(
    near_score: np.ndarray,
    strength: np.float64,
    convergence: np.float64,
) -> np.ndarray:
    total = near_score.astype(np.float64)
    np.subtract(total, convergence, out=total)
    np.multiply(total, strength, out=total)
    np.divide(total, np.float64(100.0), out=total)
    return np.ascontiguousarray(total)


def _update_relative_total_inside_band(
    total: np.ndarray,
    near_score: np.ndarray,
    source_regions: np.ndarray,
    radius: int,
    strength: np.float64,
    convergence: np.float64,
) -> None:
    render_shape = (int(near_score.shape[0]), int(near_score.shape[1]))
    for row, edge_row in _iter_streamed_edge_rows(source_regions, render_shape, radius):
        if not np.any(edge_row):
            continue
        values = near_score[row, edge_row].astype(np.float64)
        np.subtract(values, convergence, out=values)
        np.multiply(values, strength, out=values)
        np.divide(values, np.float64(100.0), out=values)
        total[row, edge_row] = values


def _build_metric_baseline(
    native_weight: np.ndarray,
    native_weighted_inverse: np.ndarray,
    render_shape: tuple[int, int],
    focal: np.float64,
    baseline_mm: np.float64,
    convergence_m: np.float64,
    disparity_percent: np.float64,
    crop_width: int,
) -> _MetricBaseline:
    resized_weight = np.array(
        _resize_float32_bilinear(native_weight, render_shape),
        copy=True,
        order="C",
    )
    resized_weighted = np.array(
        _resize_float32_bilinear(native_weighted_inverse, render_shape),
        copy=True,
        order="C",
    )
    resized_valid = np.ascontiguousarray(
        resized_weight >= np.float32(0.5),
        dtype=np.bool_,
    )
    near_score = np.zeros(render_shape, dtype=np.float32)
    np.divide(
        resized_weighted,
        resized_weight,
        out=near_score,
        where=resized_weight > np.float32(0.0),
    )
    near_score[~resized_valid] = np.float32(0.0)
    total, stats = _derive_metric_geometry(
        near_score,
        resized_valid,
        focal,
        baseline_mm,
        convergence_m,
        disparity_percent,
        crop_width,
    )
    return _MetricBaseline(
        native_weight=native_weight,
        native_weighted_inverse=native_weighted_inverse,
        resized_weight=resized_weight,
        resized_weighted_inverse=resized_weighted,
        near_score=near_score,
        resized_valid=resized_valid,
        total_disparity_fraction=total,
        stats=stats,
    )


def _metric_low_resolution_fields(
    weighted_inverse: np.ndarray,
    valid: np.ndarray,
    focal: np.float64,
    render_width: int,
    baseline_mm: np.float64,
    convergence_m: np.float64,
    disparity_percent: np.float64,
    crop_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    near_score = np.array(weighted_inverse, copy=True, order="C")
    total, _ = _derive_metric_geometry(
        near_score,
        valid,
        focal,
        baseline_mm,
        convergence_m,
        disparity_percent,
        crop_width,
        render_width=render_width,
    )
    return _geometry_displacement(total, render_width), near_score


def _derive_metric_geometry(
    near_score: np.ndarray,
    resized_valid: np.ndarray,
    focal: np.float64,
    baseline_mm: np.float64,
    convergence_m: np.float64,
    disparity_percent: np.float64,
    crop_width: int,
    *,
    render_width: int | None = None,
) -> tuple[np.ndarray, MetricProjectionStats]:
    width = near_score.shape[1] if render_width is None else render_width
    retained_fraction = np.float64(crop_width) / np.float64(width)
    baseline_m = np.float64(baseline_mm / np.float64(1000.0))
    limit = np.float64(disparity_percent / np.float64(100.0))
    raw = (
        focal
        / retained_fraction
        * baseline_m
        * (near_score.astype(np.float64) - np.float64(1.0 / convergence_m))
    )
    clamped = np.clip(raw, -limit, limit)
    total = np.ascontiguousarray(clamped * retained_fraction, dtype=np.float64)
    valid_count = int(np.count_nonzero(resized_valid))
    clamped_count = int(np.count_nonzero(resized_valid & ((raw < -limit) | (raw > limit))))
    fraction = clamped_count / valid_count if valid_count else 0.0
    return total, MetricProjectionStats(valid_count, clamped_count, fraction)


def _update_metric_derived_inside_band(
    baseline: _MetricBaseline,
    regions: _RegionState,
    focal: np.float64,
    baseline_mm: np.float64,
    convergence_m: np.float64,
    disparity_percent: np.float64,
    crop_width: int,
) -> None:
    width = baseline.near_score.shape[1]
    retained_fraction = np.float64(crop_width) / np.float64(width)
    baseline_m = np.float64(baseline_mm / np.float64(1000.0))
    limit = np.float64(disparity_percent / np.float64(100.0))
    for row, edge_row in _iter_streamed_edge_rows(
        regions.source_region_map,
        (int(baseline.near_score.shape[0]), int(baseline.near_score.shape[1])),
        regions.radius,
    ):
        columns = np.flatnonzero(edge_row)
        for column_value in columns:
            column = int(column_value)
            weight = baseline.resized_weight[row, column]
            is_valid = bool(weight >= np.float32(0.5))
            baseline.resized_valid[row, column] = is_valid
            if weight > np.float32(0.0) and is_valid:
                inverse = np.float32(baseline.resized_weighted_inverse[row, column] / weight)
            else:
                inverse = np.float32(0.0)
            baseline.near_score[row, column] = inverse
            raw = np.float64(focal / retained_fraction)
            raw = np.float64(raw * baseline_m)
            raw = np.float64(
                raw * np.float64(np.float64(inverse) - np.float64(1.0 / convergence_m))
            )
            clamped = min(max(raw, -limit), limit)
            baseline.total_disparity_fraction[row, column] = np.float64(clamped * retained_fraction)


def _recount_metric_stats(
    near_score: np.ndarray,
    resized_valid: np.ndarray,
    focal: np.float64,
    baseline_mm: np.float64,
    convergence_m: np.float64,
    disparity_percent: np.float64,
    crop_width: int,
) -> MetricProjectionStats:
    _, stats = _derive_metric_geometry(
        near_score,
        resized_valid,
        focal,
        baseline_mm,
        convergence_m,
        disparity_percent,
        crop_width,
    )
    return stats


def _geometry_displacement(total: np.ndarray, render_width: int) -> np.ndarray:
    displacement = np.array(total, copy=True, order="C")
    np.multiply(displacement, np.float64(render_width), out=displacement)
    np.multiply(displacement, np.float64(0.5), out=displacement)
    return displacement


def _freeze_quality_result(
    *,
    near_score: np.ndarray,
    total: np.ndarray,
    source_valid: np.ndarray,
    region_state: _RegionState,
    metric_valid: np.ndarray | None,
    metric_stats: MetricProjectionStats | None,
    diagnostics: _ResampleDiagnostics,
    displacement: np.ndarray,
    recorder: _LifetimeRecorder,
) -> QualityGeometryResult:
    max_jump = _max_four_neighbour_jump(displacement)
    predicted_gap = int(math.ceil(max_jump)) + 2
    after_hash = _array_sha256(region_state.final_region_map)
    arrays: tuple[np.ndarray | None, ...] = (
        near_score,
        total,
        source_valid,
        region_state.source_region_map,
        region_state.final_region_map,
        metric_valid,
    )
    for values in arrays:
        if values is not None:
            values.setflags(write=False)
    return QualityGeometryResult(
        near_score=near_score,
        total_disparity_fraction=total,
        source_valid=source_valid,
        source_region_map=region_state.source_region_map,
        final_region_map=region_state.final_region_map,
        metric_valid=metric_valid,
        metric_stats=metric_stats,
        edge_band_pixel_count=region_state.edge_band_pixel_count,
        selected_region_mismatch_count=0,
        geometry_query_count=diagnostics.query_count,
        geometry_visited_nodes_total=diagnostics.visited_nodes_total,
        geometry_visited_nodes_max=diagnostics.visited_nodes_max,
        geometry_index_bytes=diagnostics.index_bytes,
        max_neighbour_abs_q_jump_px=max_jump,
        predicted_gap_px=predicted_gap,
        final_region_hash_before_resample=region_state.final_region_hash,
        final_region_hash_after_resample=after_hash,
        lifetime=recorder.freeze(),
    )


def _max_four_neighbour_jump(displacement: np.ndarray) -> float:
    maximum = np.float64(0.0)
    if displacement.shape[1] > 1:
        maximum = max(maximum, np.max(np.abs(displacement[:, 1:] - displacement[:, :-1])))
    if displacement.shape[0] > 1:
        maximum = max(maximum, np.max(np.abs(displacement[1:, :] - displacement[:-1, :])))
    return float(maximum)


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _validate_geometry_inputs(
    guide_bgr: np.ndarray,
    primitive: np.ndarray,
    render_shape: tuple[int, int],
) -> tuple[int, int]:
    destination_shape = _validate_shape("render_shape", render_shape)
    _validate_guide(guide_bgr, destination_shape)
    _validate_native_primitive("canonical", primitive)
    return destination_shape


def _validate_metric_geometry_inputs(
    guide_bgr: np.ndarray,
    inverse_depth: np.ndarray,
    valid: np.ndarray,
    focal_x_normalized: np.float32,
    render_shape: tuple[int, int],
) -> tuple[int, int]:
    destination_shape = _validate_geometry_inputs(
        guide_bgr,
        inverse_depth,
        render_shape,
    )
    if np.any(inverse_depth < np.float32(0.0)):
        raise ValueError("inverse_depth must be nonnegative")
    if (
        not isinstance(valid, np.ndarray)
        or valid.dtype != np.bool_
        or valid.shape != inverse_depth.shape
        or not valid.flags.c_contiguous
    ):
        raise TypeError("valid must be a matching C-contiguous bool raster")
    if not isinstance(focal_x_normalized, np.float32):
        raise TypeError("focal_x_normalized must use float32")
    if not np.isfinite(focal_x_normalized) or focal_x_normalized <= np.float32(0.0):
        raise ValueError("focal_x_normalized must be finite and positive")
    return destination_shape


def _validate_guide(guide_bgr: np.ndarray, shape: tuple[int, int]) -> None:
    if (
        not isinstance(guide_bgr, np.ndarray)
        or guide_bgr.dtype != np.uint8
        or guide_bgr.shape != (*shape, 3)
        or not guide_bgr.flags.c_contiguous
    ):
        raise TypeError("guide_bgr must be a matching C-contiguous uint8 BGR image")


def _validate_native_primitive(name: str, values: np.ndarray) -> None:
    if (
        not isinstance(values, np.ndarray)
        or values.dtype != np.float32
        or values.ndim != 2
        or values.shape[0] <= 0
        or values.shape[1] <= 0
        or not values.flags.c_contiguous
    ):
        raise TypeError(f"{name} must be a nonempty C-contiguous float32 raster")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")


def _validate_source_regions(
    source_regions: np.ndarray,
    expected_shape: tuple[int, int] | None = None,
) -> None:
    if (
        not isinstance(source_regions, np.ndarray)
        or source_regions.dtype != np.uint32
        or source_regions.ndim != 2
        or source_regions.shape[0] <= 0
        or source_regions.shape[1] <= 0
        or not source_regions.flags.c_contiguous
    ):
        raise TypeError("source_regions must be a nonempty C-contiguous uint32 raster")
    if expected_shape is not None and source_regions.shape != expected_shape:
        raise ValueError("source_regions must match the primitive shape")
    if np.any(source_regions == 0) or np.any(source_regions == np.uint32(_UINT32_MAX)):
        raise ValueError("source_regions must contain positive canonical IDs")


def _validate_shape(name: str, shape: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
    ):
        raise TypeError(f"{name} must be an integer (height, width) tuple")
    height, width = shape
    if height <= 0 or width <= 0 or height * width + 1 > _UINT32_MAX:
        raise ValueError(f"{name} must be positive and fit the uint32 contract")
    return height, width


def _validate_output_coordinate(
    coordinate: tuple[int, int],
    render_shape: tuple[int, int],
) -> tuple[int, int]:
    if (
        not isinstance(coordinate, tuple)
        or len(coordinate) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in coordinate)
    ):
        raise TypeError("output_coordinate must be an integer (row, column) tuple")
    row, column = coordinate
    if not 0 <= row < render_shape[0] or not 0 <= column < render_shape[1]:
        raise ValueError("output_coordinate lies outside render_shape")
    return row, column


def _validate_region(value: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("selected_region must be an integer")
    region = int(value)
    if not 0 < region <= maximum:
        raise ValueError("selected_region must be present in source_regions")
    return region


def _validate_offset_buffers(
    near_score: np.ndarray,
    output: np.ndarray,
    scratch: np.ndarray,
) -> None:
    _validate_native_primitive("near_score", near_score)
    _validate_common_offset_buffers(
        (int(near_score.shape[0]), int(near_score.shape[1])),
        output,
        scratch,
    )


def _validate_geometry_offset_buffers(
    total: np.ndarray,
    output: np.ndarray,
    scratch: np.ndarray,
) -> None:
    if (
        not isinstance(total, np.ndarray)
        or total.dtype != np.float64
        or total.ndim != 2
        or not total.flags.c_contiguous
        or not np.isfinite(total).all()
    ):
        raise TypeError("total_disparity_fraction must be a finite C-contiguous float64 raster")
    _validate_common_offset_buffers(
        (int(total.shape[0]), int(total.shape[1])),
        output,
        scratch,
    )


def _validate_common_offset_buffers(
    shape: tuple[int, int],
    output: np.ndarray,
    scratch: np.ndarray,
) -> None:
    if (
        not isinstance(output, np.ndarray)
        or output.dtype != np.int32
        or output.shape != shape
        or not output.flags.c_contiguous
        or not output.flags.writeable
    ):
        raise TypeError("output_int32 must be a writable matching C-contiguous int32 raster")
    if (
        not isinstance(scratch, np.ndarray)
        or scratch.dtype != np.float64
        or scratch.shape != (shape[1],)
        or not scratch.flags.c_contiguous
        or not scratch.flags.writeable
    ):
        raise TypeError("row_float64_scratch must be a writable width-sized float64 vector")
    if np.shares_memory(output, scratch):
        raise ValueError("output and row scratch must not share storage")


def _narrow_offset_row(values: np.ndarray, output: np.ndarray) -> None:
    if values.size and (float(values.min()) < _INT32_MIN or float(values.max()) > _INT32_MAX):
        raise ValueError("projected fine-sample offsets exceed int32 range")
    output[:] = values.astype(np.int32)


def _validate_eye(eye: str) -> int:
    if eye == "left":
        return 1
    if eye == "right":
        return -1
    raise ValueError("eye must be 'left' or 'right'")


def _validate_float(
    name: str,
    value: object,
    minimum: float,
    maximum: float,
) -> np.float64:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be numeric")
    result = np.float64(value)
    if not np.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be finite and within [{minimum}, {maximum}]")
    return result


def _validate_positive_float(name: str, value: object) -> np.float64:
    result = _validate_float(name, value, 0.0, float(np.finfo(np.float64).max))
    if result <= np.float64(0.0):
        raise ValueError(f"{name} must be positive")
    return result


def _validate_crop_width(value: int, render_width: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("retained_crop_width must be an integer")
    width = int(value)
    if not 1 <= width <= render_width:
        raise ValueError("retained_crop_width must lie within the render width")
    return width


def _require_nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result
