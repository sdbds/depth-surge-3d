"""Native-backed Quality region primitives and input validation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from . import native_kd


class QualityRegionQueueBudgetError(RuntimeError):
    """Geodesic priority queue crossed its fixed capacity or uint64 distance bound."""


@dataclass(frozen=True)
class LowResolutionRegionResult:
    """Canonical low-resolution labels and their deterministic tie ranks."""

    region_map: np.ndarray
    canonical_keys: np.ndarray
    region_rank_bits: np.ndarray


@dataclass(frozen=True)
class StableSurfaceSupportResult:
    """Fused stable-sample, surface-region, and interior support outputs."""

    surface_regions: LowResolutionRegionResult
    stable_samples: np.ndarray
    stable_surfaces: np.ndarray
    stable_sample_include: np.ndarray


@dataclass(frozen=True)
class GeodesicSolveResult:
    """Native RGB geodesic solve output and queue accounting."""

    region_map: np.ndarray
    distance: np.ndarray
    heap_position: np.ndarray
    max_live_entries: int
    settled_count: int
    queue_insertions: int
    queue_decrease_keys: int
    queue_pops: int
    heap_pixel_bytes: int
    heap_position_bytes: int
    distance_bytes: int


def build_low_resolution_regions(
    one_eye_displacement_px: np.ndarray,
    metric_validity: np.ndarray,
    near_score: np.ndarray,
) -> LowResolutionRegionResult:
    """Build stable four-neighbour regions from final one-eye displacement."""

    _validate_low_resolution_inputs(
        one_eye_displacement_px,
        metric_validity,
        near_score,
    )
    height, width = one_eye_displacement_px.shape
    if height * width + 1 > int(np.iinfo(np.uint32).max):
        raise ValueError("low-resolution sample count exceeds uint32 contract")
    (
        region_map,
        canonical_keys,
        region_rank_bits,
    ) = native_kd._require_native().build_low_resolution_regions(
        one_eye_displacement_px,
        metric_validity,
        near_score,
    )
    for values in (region_map, canonical_keys, region_rank_bits):
        values.setflags(write=False)
    return LowResolutionRegionResult(
        region_map=region_map,
        canonical_keys=canonical_keys,
        region_rank_bits=region_rank_bits,
    )


def build_stable_surface_support(
    initial_region_map: np.ndarray,
    stable_regions: np.ndarray,
    one_eye_displacement_px: np.ndarray,
    near_score: np.ndarray,
) -> StableSurfaceSupportResult:
    """Build stable surface support without synthetic displacement rasters."""

    _validate_region_map(initial_region_map)
    shape = (int(initial_region_map.shape[0]), int(initial_region_map.shape[1]))
    if (
        not isinstance(stable_regions, np.ndarray)
        or stable_regions.dtype != np.bool_
        or stable_regions.ndim != 1
        or not stable_regions.flags.c_contiguous
    ):
        raise TypeError("stable_regions must be a C-contiguous bool vector")
    if (
        not isinstance(one_eye_displacement_px, np.ndarray)
        or one_eye_displacement_px.dtype != np.float64
        or one_eye_displacement_px.shape != shape
        or not one_eye_displacement_px.flags.c_contiguous
    ):
        raise TypeError("one_eye_displacement_px must be a matching C-contiguous float64 raster")
    if (
        not isinstance(near_score, np.ndarray)
        or near_score.dtype != np.float32
        or near_score.shape != shape
        or not near_score.flags.c_contiguous
    ):
        raise TypeError("near_score must be a matching C-contiguous float32 raster")
    (
        region_map,
        canonical_keys,
        region_rank_bits,
        stable_samples,
        stable_surfaces,
        stable_sample_include,
    ) = native_kd._require_native().build_stable_surface_support(
        initial_region_map,
        stable_regions,
        one_eye_displacement_px,
        near_score,
    )
    for values in (
        region_map,
        canonical_keys,
        region_rank_bits,
        stable_samples,
        stable_surfaces,
        stable_sample_include,
    ):
        values.setflags(write=False)
    return StableSurfaceSupportResult(
        surface_regions=LowResolutionRegionResult(
            region_map=region_map,
            canonical_keys=canonical_keys,
            region_rank_bits=region_rank_bits,
        ),
        stable_samples=stable_samples,
        stable_surfaces=stable_surfaces,
        stable_sample_include=stable_sample_include,
    )


def nearest_label_upsample(
    low_resolution_regions: np.ndarray,
    *,
    render_height: int,
    render_width: int,
) -> np.ndarray:
    """Upsample labels with the native half-pixel nearest-neighbour kernel."""

    _validate_region_map(low_resolution_regions)
    destination_height = _validate_positive_dimension("render_height", render_height)
    destination_width = _validate_positive_dimension("render_width", render_width)
    output = native_kd._require_native().nearest_label_upsample(
        low_resolution_regions,
        destination_height,
        destination_width,
    )
    output.setflags(write=False)
    return output


def edge_band_radius(
    *,
    geometry_height: int,
    geometry_width: int,
    render_height: int,
    render_width: int,
) -> int:
    """Return the canonical mapped-boundary dilation radius."""

    source_height = _validate_positive_dimension("geometry_height", geometry_height)
    source_width = _validate_positive_dimension("geometry_width", geometry_width)
    destination_height = _validate_positive_dimension("render_height", render_height)
    destination_width = _validate_positive_dimension("render_width", render_width)
    scale_x = float(destination_width) / float(source_width)
    scale_y = float(destination_height) / float(source_height)
    return int(math.ceil(max(scale_x, scale_y))) + 1


def build_four_connected_edge_band(
    mapped_regions: np.ndarray,
    *,
    radius: int,
) -> np.ndarray:
    """Mark every valid two-sided label boundary and cross-dilate it."""

    _validate_region_map(mapped_regions)
    dilation_radius = _validate_nonnegative_integer("radius", radius)
    boundary = native_kd._require_native().build_four_connected_edge_band(
        mapped_regions,
        dilation_radius,
    )
    boundary.setflags(write=False)
    return boundary


def build_sparse_region_seeds(
    mapped_regions: np.ndarray,
    edge_band: np.ndarray,
    *,
    radius: int,
) -> np.ndarray:
    """Keep immutable outside markers and deterministic sparse band seeds."""

    _validate_region_map(mapped_regions)
    shape = (int(mapped_regions.shape[0]), int(mapped_regions.shape[1]))
    _validate_bool_raster("edge_band", edge_band, shape=shape)
    erosion_radius = _validate_nonnegative_integer("radius", radius)
    seeds = native_kd._require_native().build_sparse_region_seeds(
        mapped_regions,
        edge_band,
        erosion_radius,
    )
    seeds.setflags(write=False)
    return seeds


def build_sparse_stable_region_seeds(
    mapped_regions: np.ndarray,
    edge_band: np.ndarray,
    stable_initial: np.ndarray,
    stable_surfaces: np.ndarray,
    *,
    radius: int,
) -> np.ndarray:
    """Build sparse seeds and discard unstable owners in one native pass."""

    _validate_region_map(mapped_regions)
    shape = (int(mapped_regions.shape[0]), int(mapped_regions.shape[1]))
    _validate_bool_raster("edge_band", edge_band, shape=shape)
    _validate_bool_raster("stable_initial", stable_initial, shape=shape)
    if (
        not isinstance(stable_surfaces, np.ndarray)
        or stable_surfaces.dtype != np.bool_
        or stable_surfaces.ndim != 1
        or not stable_surfaces.shape[0]
        or not stable_surfaces.flags.c_contiguous
    ):
        raise TypeError("stable_surfaces must be a non-empty C-contiguous bool vector")
    erosion_radius = _validate_nonnegative_integer("radius", radius)
    seeds = native_kd._require_native().build_sparse_stable_region_seeds(
        mapped_regions,
        edge_band,
        stable_initial,
        stable_surfaces,
        erosion_radius,
    )
    seeds.setflags(write=False)
    return seeds


def select_stable_region_support(
    region_map: np.ndarray,
    *,
    region_count: int,
    area_threshold: int,
) -> np.ndarray:
    """Select stable low-resolution regions with known label cardinality."""

    _validate_region_map(region_map)
    count = _validate_positive_dimension("region_count", region_count)
    threshold = _validate_positive_dimension("area_threshold", area_threshold)
    stable = native_kd._require_native().select_stable_region_support(
        region_map,
        count,
        threshold,
    )
    stable.setflags(write=False)
    return stable


def _validate_low_resolution_inputs(
    one_eye_displacement_px: np.ndarray,
    metric_validity: np.ndarray,
    near_score: np.ndarray,
) -> None:
    if (
        not isinstance(one_eye_displacement_px, np.ndarray)
        or one_eye_displacement_px.dtype != np.float64
        or one_eye_displacement_px.ndim != 2
        or not one_eye_displacement_px.flags.c_contiguous
    ):
        raise TypeError("one_eye_displacement_px must be a C-contiguous float64 raster")
    shape = (
        int(one_eye_displacement_px.shape[0]),
        int(one_eye_displacement_px.shape[1]),
    )
    if not shape[0] or not shape[1]:
        raise ValueError("low-resolution rasters must be non-empty")
    _validate_bool_raster("metric_validity", metric_validity, shape=shape)
    if (
        not isinstance(near_score, np.ndarray)
        or near_score.dtype != np.float32
        or near_score.shape != shape
        or not near_score.flags.c_contiguous
    ):
        raise TypeError("near_score must be a matching C-contiguous float32 raster")


def _validate_region_map(region_map: np.ndarray) -> None:
    if (
        not isinstance(region_map, np.ndarray)
        or region_map.dtype != np.uint32
        or region_map.ndim != 2
        or not region_map.flags.c_contiguous
    ):
        raise TypeError("region map must be a C-contiguous uint32 raster")
    if not region_map.shape[0] or not region_map.shape[1]:
        raise ValueError("region map must be non-empty")


def _validate_bool_raster(
    name: str,
    values: np.ndarray,
    *,
    shape: tuple[int, int] | None = None,
) -> None:
    if (
        not isinstance(values, np.ndarray)
        or values.dtype != np.bool_
        or values.ndim != 2
        or (shape is not None and values.shape != shape)
        or not values.flags.c_contiguous
    ):
        qualifier = "matching " if shape is not None else ""
        raise TypeError(f"{name} must be a {qualifier}C-contiguous bool raster")


def _validate_positive_dimension(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _validate_solver_inputs(
    guide: np.ndarray,
    band_mask: np.ndarray,
    seed_region_map: np.ndarray,
    region_rank_bits: np.ndarray,
) -> None:
    if (
        not isinstance(guide, np.ndarray)
        or guide.dtype != np.uint8
        or guide.ndim != 3
        or guide.shape[-1] != 3
        or not guide.flags.c_contiguous
    ):
        raise TypeError("guide must be a C-contiguous three-channel uint8 BGR raster")
    shape = (int(guide.shape[0]), int(guide.shape[1]))
    _validate_bool_raster("band_mask", band_mask, shape=shape)
    if (
        not isinstance(seed_region_map, np.ndarray)
        or seed_region_map.dtype != np.uint32
        or seed_region_map.shape != shape
        or not seed_region_map.flags.c_contiguous
    ):
        raise TypeError("seed_region_map must be a matching C-contiguous uint32 raster")
    if (
        not isinstance(region_rank_bits, np.ndarray)
        or region_rank_bits.dtype != np.uint32
        or region_rank_bits.ndim != 1
        or region_rank_bits.size < 2
        or not region_rank_bits.flags.c_contiguous
    ):
        raise TypeError("region_rank_bits must be a C-contiguous uint32 vector")
    maximum_region = int(seed_region_map.max(initial=np.uint32(0)))
    if maximum_region >= region_rank_bits.size:
        raise ValueError("seed region exceeds region_rank_bits")
