"""Scalar contracts for one-sided Quality geometry and lifetime order."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

from scripts.quality_task0 import region_solver as reference_solver
from src.depth_surge_3d.rendering.stereo_geometry import (
    build_metric_geometry,
    build_relative_geometry,
    resize_metric_geometry,
)
from src.depth_surge_3d.rendering.stereo_renderer import (
    StereoRenderSettings,
    calculate_eye_sample_offsets,
    calculate_geometry_eye_sample_offsets,
)


def test_quality_geometry_module_is_importable() -> None:
    assert importlib.util.find_spec("scripts.quality_task0.quality_geometry") is not None


def _scalar_coordinate(
    output_index: int,
    source_size: int,
    destination_size: int,
) -> np.float64:
    coordinate = np.float64(output_index)
    coordinate = np.float64(coordinate + np.float64(0.5))
    coordinate = np.float64(coordinate * np.float64(source_size))
    coordinate = np.float64(coordinate / np.float64(destination_size))
    return np.float64(coordinate - np.float64(0.5))


def _scalar_one_sided(
    values: np.ndarray,
    regions: np.ndarray,
    *,
    selected_region: int,
    render_shape: tuple[int, int],
    output_coordinate: tuple[int, int],
) -> np.float32 | None:
    source_height, source_width = values.shape
    render_height, render_width = render_shape
    output_row, output_column = output_coordinate
    sx = _scalar_coordinate(output_column, source_width, render_width)
    sy = _scalar_coordinate(output_row, source_height, render_height)
    cx = min(max(sx, np.float64(0.0)), np.float64(source_width - 1))
    cy = min(max(sy, np.float64(0.0)), np.float64(source_height - 1))
    x0 = int(np.floor(cx))
    x1 = min(x0 + 1, source_width - 1)
    y0 = int(np.floor(cy))
    y1 = min(y0 + 1, source_height - 1)
    wx = np.float64(cx - np.float64(x0))
    wy = np.float64(cy - np.float64(y0))
    ox = np.float64(np.float64(1.0) - wx)
    oy = np.float64(np.float64(1.0) - wy)
    weights = (
        np.float64(oy * ox),
        np.float64(oy * wx),
        np.float64(wy * ox),
        np.float64(wy * wx),
    )
    coordinates = ((y0, x0), (y0, x1), (y1, x0), (y1, x1))
    retained_weights = tuple(
        np.float64(weight if int(regions[row, column]) == selected_region else 0.0)
        for weight, (row, column) in zip(weights, coordinates, strict=True)
    )
    retained = np.float64(retained_weights[0] + retained_weights[1])
    retained = np.float64(retained + retained_weights[2])
    retained = np.float64(retained + retained_weights[3])
    if retained == np.float64(0.0):
        return None
    terms = tuple(
        np.float64(np.float64(values[row, column]) * weight)
        for weight, (row, column) in zip(retained_weights, coordinates, strict=True)
    )
    numerator = np.float64(terms[0] + terms[1])
    numerator = np.float64(numerator + terms[2])
    numerator = np.float64(numerator + terms[3])
    return np.float32(np.float64(numerator / retained))


@pytest.mark.parametrize("mask", range(16))
@pytest.mark.parametrize("output_coordinate", [(0, 0), (1, 1), (2, 2)])
def test_one_sided_scalar_matches_independent_binary64_oracle(
    quality_geometry,
    mask: int,
    output_coordinate: tuple[int, int],
) -> None:
    resample = quality_geometry.one_sided_resample_scalar
    values = np.array([[0.125, 0.375], [0.625, 0.875]], dtype=np.float32)
    regions = np.full((2, 2), 2, dtype=np.uint32)
    for index in range(4):
        if mask & (1 << index):
            regions.flat[index] = np.uint32(1)
    expected = _scalar_one_sided(
        values,
        regions,
        selected_region=1,
        render_shape=(3, 3),
        output_coordinate=output_coordinate,
    )

    actual = resample(
        values,
        regions,
        selected_region=1,
        render_shape=(3, 3),
        output_coordinate=output_coordinate,
    )

    if expected is None:
        assert actual is None
    else:
        assert np.asarray(actual, dtype=np.float32).tobytes() == expected.tobytes()


def _offset_for_value(quality_geometry, value: np.float32) -> int:
    build_offsets = quality_geometry.build_relative_eye_offsets_into
    near_score = np.full((1, 1920), np.float32(0.5), dtype=np.float32)
    near_score[0, 0] = value
    output = np.empty(near_score.shape, dtype=np.int32)
    scratch = np.empty(near_score.shape[1], dtype=np.float64)
    build_offsets(
        near_score,
        stereo_strength=2.0,
        convergence=0.5,
        eye="left",
        output_int32=output,
        row_float64_scratch=scratch,
    )
    return int(output[0, 0])


def test_one_ulp_one_sided_primitive_change_crosses_a_relative_fine_lane(quality_geometry) -> None:
    resample = quality_geometry.one_sided_resample_scalar
    boundary = np.float32(0.9052734375)
    above = np.nextafter(boundary, np.float32(1.0))
    regions = np.array([[1, 2], [2, 2]], dtype=np.uint32)
    base_values = np.full((2, 2), boundary, dtype=np.float32)
    above_values = np.full((2, 2), above, dtype=np.float32)

    base = resample(
        base_values,
        regions,
        selected_region=1,
        render_shape=(3, 3),
        output_coordinate=(1, 1),
    )
    changed = resample(
        above_values,
        regions,
        selected_region=1,
        render_shape=(3, 3),
        output_coordinate=(1, 1),
    )

    assert base == boundary
    assert changed == above
    assert _offset_for_value(quality_geometry, base) == 124
    assert _offset_for_value(quality_geometry, changed) == 125


def test_streamed_edge_rows_equal_the_full_cross_dilated_oracle(quality_geometry) -> None:
    materialize = quality_geometry.materialize_streamed_edge_band
    low_regions = np.array(
        [[1, 1, 2], [1, 3, 2], [4, 3, 2]],
        dtype=np.uint32,
    )
    render_shape = (11, 13)
    mapped = reference_solver.nearest_label_upsample(
        low_regions,
        render_height=render_shape[0],
        render_width=render_shape[1],
    )
    radius = reference_solver.edge_band_radius(
        geometry_height=low_regions.shape[0],
        geometry_width=low_regions.shape[1],
        render_height=render_shape[0],
        render_width=render_shape[1],
    )
    expected = reference_solver.build_four_connected_edge_band(mapped, radius=radius)

    actual = materialize(low_regions, render_shape=render_shape, radius=radius)

    np.testing.assert_array_equal(actual, expected)


def _relative_edge_fixture():
    geometry_height, geometry_width = 4, 24
    render_shape = (16, 64)
    canonical = np.zeros((geometry_height, geometry_width), dtype=np.float32)
    canonical[:, geometry_width // 2 :] = np.float32(1.0)
    guide = np.zeros((*render_shape, 3), dtype=np.uint8)
    guide[:, 34:] = np.uint8(255)
    return guide, canonical, render_shape


def test_relative_ten_step_geometry_uses_final_geodesic_regions_and_exact_baseline() -> None:
    # The Task 0 trace always builds an index; production can use a bounded local query.
    from scripts.quality_task0 import quality_geometry

    build = quality_geometry.build_relative_quality_geometry
    guide, canonical, render_shape = _relative_edge_fixture()
    result = build(
        guide,
        canonical,
        render_shape=render_shape,
        stereo_strength=5.0,
        convergence=0.5,
    )
    baseline = build_relative_geometry(
        canonical,
        render_shape,
        stereo_strength=5.0,
        convergence=0.5,
    )
    initial = reference_solver.nearest_label_upsample(
        result.source_region_map,
        render_height=render_shape[0],
        render_width=render_shape[1],
    )
    radius = reference_solver.edge_band_radius(
        geometry_height=canonical.shape[0],
        geometry_width=canonical.shape[1],
        render_height=render_shape[0],
        render_width=render_shape[1],
    )
    edge_band = reference_solver.build_four_connected_edge_band(initial, radius=radius)

    np.testing.assert_array_equal(result.near_score[~edge_band], baseline.near_score[~edge_band])
    np.testing.assert_array_equal(
        result.total_disparity_fraction[~edge_band],
        baseline.total_disparity_fraction[~edge_band],
    )
    assert np.any(result.final_region_map != initial)
    assert set(np.unique(result.near_score[edge_band]).tolist()) <= {0.0, 1.0}
    changed_coordinate = (8, 32)
    assert result.final_region_map[changed_coordinate] != initial[changed_coordinate]
    final_selected = quality_geometry.one_sided_resample_scalar(
        canonical,
        result.source_region_map,
        selected_region=int(result.final_region_map[changed_coordinate]),
        render_shape=render_shape,
        output_coordinate=changed_coordinate,
    )
    initial_selected = quality_geometry.one_sided_resample_scalar(
        canonical,
        result.source_region_map,
        selected_region=int(initial[changed_coordinate]),
        render_shape=render_shape,
        output_coordinate=changed_coordinate,
    )
    assert result.near_score[changed_coordinate] == final_selected == np.float32(0.0)
    assert initial_selected == np.float32(1.0)
    assert result.selected_region_mismatch_count == 0
    assert result.geometry_query_count > 0
    assert result.final_region_hash_before_resample == result.final_region_hash_after_resample
    assert not result.final_region_map.flags.writeable

    event_keys = [(event.action, event.name) for event in result.lifetime.events]
    release_solver = event_keys.index(("release", "geodesic_solver_scratch"))
    release_morphology = event_keys.index(("release", "morphology_scratch"))
    allocate_index = event_keys.index(("allocate", "retained_zero_index"))
    release_index = event_keys.index(("release", "retained_zero_index"))
    derive = event_keys.index(("checkpoint", "derived_geometry_complete"))
    assert max(release_solver, release_morphology) < allocate_index < release_index < derive


def _metric_stats_oracle(
    near_score: np.ndarray,
    resized_valid: np.ndarray,
    *,
    focal: float,
    baseline_mm: float,
    convergence_m: float,
    max_disparity_percent: float,
    retained_crop_width: int,
) -> tuple[np.ndarray, tuple[int, int, float]]:
    source_width = near_score.shape[1]
    retained_fraction = np.float64(retained_crop_width) / np.float64(source_width)
    raw = (
        np.float64(focal)
        / retained_fraction
        * (np.float64(baseline_mm) / np.float64(1000.0))
        * (near_score.astype(np.float64) - np.float64(1.0 / convergence_m))
    )
    limit = np.float64(max_disparity_percent) / np.float64(100.0)
    total = np.clip(raw, -limit, limit) * retained_fraction
    valid_count = int(np.count_nonzero(resized_valid))
    clamped_count = int(np.count_nonzero(resized_valid & ((raw < -limit) | (raw > limit))))
    fraction = clamped_count / valid_count if valid_count else 0.0
    return total, (valid_count, clamped_count, fraction)


def test_metric_geometry_resamples_primitives_then_recounts_clamps(quality_geometry) -> None:
    build = quality_geometry.build_metric_quality_geometry
    inverse = np.array(
        [[0.1, 0.1, 10.0, 10.0], [0.1, 0.1, 10.0, 10.0]],
        dtype=np.float32,
    )
    valid = np.ones(inverse.shape, dtype=np.bool_)
    render_shape = (6, 12)
    guide = np.zeros((*render_shape, 3), dtype=np.uint8)
    guide[:, 7:] = np.uint8(255)
    result = build(
        guide,
        inverse,
        valid,
        focal_x_normalized=np.float32(1.0),
        render_shape=render_shape,
        virtual_baseline_mm=100.0,
        convergence_distance_m=2.0,
        max_disparity_percent=100.0,
        retained_crop_width=12,
    )
    fast_geometry, _fast_stats = build_metric_geometry(
        inverse,
        valid,
        np.float32(1.0),
        render_shape,
        virtual_baseline_mm=100.0,
        convergence_distance_m=2.0,
        max_disparity_percent=100.0,
        retained_crop_width=12,
    )
    _fast_inverse, fast_valid = resize_metric_geometry(inverse, valid, render_shape)
    initial = reference_solver.nearest_label_upsample(
        result.source_region_map,
        render_height=render_shape[0],
        render_width=render_shape[1],
    )
    radius = reference_solver.edge_band_radius(
        geometry_height=inverse.shape[0],
        geometry_width=inverse.shape[1],
        render_height=render_shape[0],
        render_width=render_shape[1],
    )
    edge_band = reference_solver.build_four_connected_edge_band(initial, radius=radius)
    expected_total, expected_stats = _metric_stats_oracle(
        result.near_score,
        result.metric_valid,
        focal=1.0,
        baseline_mm=100.0,
        convergence_m=2.0,
        max_disparity_percent=100.0,
        retained_crop_width=12,
    )

    np.testing.assert_array_equal(
        result.near_score[~edge_band],
        fast_geometry.near_score[~edge_band],
    )
    np.testing.assert_array_equal(
        result.total_disparity_fraction[~edge_band],
        fast_geometry.total_disparity_fraction[~edge_band],
    )
    np.testing.assert_array_equal(result.metric_valid[~edge_band], fast_valid[~edge_band])
    np.testing.assert_array_equal(result.total_disparity_fraction, expected_total)
    assert (
        result.metric_stats.valid_pixel_count,
        result.metric_stats.clamped_pixel_count,
        result.metric_stats.clamped_fraction,
    ) == expected_stats
    assert result.source_valid.all()


@pytest.mark.parametrize(
    "geometry_shape",
    [(4, 4), (4, 8), (8, 8), (8, 16)],
    ids=["G_over_Q_0_5", "G_over_Q_1", "G_over_Q_2", "G_over_Q_4"],
)
def test_relative_offsets_and_metric_clamps_match_fast_at_all_frozen_ratios(
    quality_geometry,
    geometry_shape: tuple[int, int],
) -> None:
    build_relative = quality_geometry.build_relative_quality_geometry
    build_metric = quality_geometry.build_metric_quality_geometry
    build_offsets = quality_geometry.build_relative_eye_offsets_into
    build_geometry_offsets = quality_geometry.build_geometry_eye_offsets_into
    render_shape = (4, 8)
    guide = np.zeros((*render_shape, 3), dtype=np.uint8)
    canonical = np.full(geometry_shape, np.float32(0.9052734375), dtype=np.float32)
    relative = build_relative(
        guide,
        canonical,
        render_shape=render_shape,
        stereo_strength=2.0,
        convergence=0.5,
    )
    settings = StereoRenderSettings(stereo_strength=2.0, convergence=0.5)
    fast_left, fast_right = calculate_eye_sample_offsets(
        torch.from_numpy(relative.near_score.copy()),
        settings,
    )
    scratch = np.empty(render_shape[1], dtype=np.float64)
    for eye, expected in (("left", fast_left), ("right", fast_right)):
        actual = np.empty(render_shape, dtype=np.int32)
        build_offsets(
            relative.near_score,
            stereo_strength=2.0,
            convergence=0.5,
            eye=eye,
            output_int32=actual,
            row_float64_scratch=scratch,
        )
        np.testing.assert_array_equal(actual, expected)

    inverse = np.full(geometry_shape, np.float32(1.25), dtype=np.float32)
    valid = np.ones(geometry_shape, dtype=np.bool_)
    metric = build_metric(
        guide,
        inverse,
        valid,
        focal_x_normalized=np.float32(0.8),
        render_shape=render_shape,
        virtual_baseline_mm=63.0,
        convergence_distance_m=2.0,
        max_disparity_percent=2.0,
        retained_crop_width=8,
    )
    fast_geometry, fast_stats = build_metric_geometry(
        inverse,
        valid,
        np.float32(0.8),
        render_shape,
        virtual_baseline_mm=63.0,
        convergence_distance_m=2.0,
        max_disparity_percent=2.0,
        retained_crop_width=8,
    )
    np.testing.assert_array_equal(metric.near_score, fast_geometry.near_score)
    np.testing.assert_array_equal(
        metric.total_disparity_fraction,
        fast_geometry.total_disparity_fraction,
    )
    assert metric.metric_stats == fast_stats

    expected_metric_offsets = calculate_geometry_eye_sample_offsets(metric.total_disparity_fraction)
    for eye, expected in zip(
        ("left", "right"),
        expected_metric_offsets,
        strict=True,
    ):
        actual = np.empty(render_shape, dtype=np.int32)
        build_geometry_offsets(
            metric.total_disparity_fraction,
            eye=eye,
            output_int32=actual,
            row_float64_scratch=scratch,
        )
        np.testing.assert_array_equal(actual, expected)
