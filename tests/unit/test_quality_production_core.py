from __future__ import annotations

import importlib

import numpy as np
import pytest

from scripts.quality_task0 import quality_geometry as prototype_geometry


def _result_signature(result) -> tuple[object, ...]:
    return (
        result.near_score.tobytes(),
        result.total_disparity_fraction.tobytes(),
        result.source_valid.tobytes(),
        result.source_region_map.tobytes(),
        result.final_region_map.tobytes(),
        result.edge_band_pixel_count,
        result.selected_region_mismatch_count,
        result.geometry_query_count,
        result.geometry_visited_nodes_total,
        result.geometry_visited_nodes_max,
        result.geometry_index_bytes,
        result.max_neighbour_abs_q_jump_px,
        result.predicted_gap_px,
        result.final_region_hash_before_resample,
        result.final_region_hash_after_resample,
    )


def test_production_relative_geometry_matches_gate_prototype() -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.geometry")
    guide = np.array(
        [
            [[0, 0, 0], [0, 0, 0], [220, 220, 220], [220, 220, 220], [20, 40, 80]],
            [[0, 0, 0], [0, 0, 0], [220, 220, 220], [220, 220, 220], [20, 40, 80]],
            [[10, 10, 10], [10, 10, 10], [230, 230, 230], [230, 230, 230], [30, 50, 90]],
            [[10, 10, 10], [10, 10, 10], [230, 230, 230], [230, 230, 230], [30, 50, 90]],
        ],
        dtype=np.uint8,
    )
    canonical = np.array([[0.1, 0.9, 0.4], [0.2, 0.8, 0.5]], dtype=np.float32)
    arguments = {
        "guide_bgr": guide,
        "canonical": canonical,
        "render_shape": guide.shape[:2],
        "stereo_strength": 2.0,
        "convergence": 0.5,
    }

    expected = prototype_geometry.build_relative_quality_geometry(**arguments)
    actual = production.build_relative_quality_geometry(**arguments)

    assert _result_signature(actual) == _result_signature(expected)


def test_production_single_eye_offset_builders_match_gate_prototype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.geometry")
    near_score = np.array(
        [[0.9052734375, np.nextafter(np.float32(0.5), np.float32(1.0))]],
        dtype=np.float32,
    )
    expected = np.empty(near_score.shape, dtype=np.int32)
    actual = np.empty(near_score.shape, dtype=np.int32)

    prototype_geometry.build_relative_eye_offsets_into(
        near_score,
        stereo_strength=2.0,
        convergence=0.5,
        eye="left",
        output_int32=expected,
        row_float64_scratch=np.empty(near_score.shape[1], dtype=np.float64),
    )
    monkeypatch.setattr(
        production.np,
        "copyto",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("production entered the scalar row offset builder")
        ),
    )
    production.build_relative_eye_offsets_into(
        near_score,
        stereo_strength=2.0,
        convergence=0.5,
        eye="left",
        output_int32=actual,
        row_float64_scratch=np.empty(near_score.shape[1], dtype=np.float64),
    )

    assert np.array_equal(actual, expected)
    assert actual[0, 0] == 0


def test_production_native_module_uses_frozen_prebuilt_algorithm() -> None:
    native = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")

    assert native.native_available()
    build_info = native.native_build_info()
    assert build_info["algorithm"] == "quality-implicit-region-kd-hybrid-nth-element-v2"
    assert build_info["index_build_parallelism"] == "deterministic-region-std-thread-v1"
    assert 1 <= build_info["index_build_worker_cap"] <= 32
    assert build_info["geometry_query_parallelism"] == "deterministic-std-thread-v1"
    assert 1 <= build_info["geometry_query_worker_cap"] <= 32
    assert build_info["native_spawned_worker_cap"] == 7
    assert build_info["kd_stack_capacity"] == 64
    assert build_info["runtime_jit"] is False
    assert len(build_info["source_sha256"]) == 64
    int(build_info["source_sha256"], 16)


def test_stable_region_support_rejects_low_area_transition_slivers() -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.geometry")
    regions = np.ones((8, 10), dtype=np.uint32)
    regions[1:3, 1:3] = np.uint32(2)  # area 4, span 2: transient
    regions[4, 1:5] = np.uint32(3)  # area 4, span 4: transition sliver
    regions[5:7, 6:9] = np.uint32(4)  # area 6: stable surface

    stable = production._select_stable_region_support(
        regions,
        area_threshold=5,
    )
    included = production._stable_interior_support(regions, stable[regions])

    assert stable.tolist() == [False, True, False, False, True]
    assert not included[np.isin(regions, (2, 3))].any()
    for region in (1, 4):
        assert included[regions == np.uint32(region)].any()


def test_native_foreground_matte_matches_hand_checked_cross_copy() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    near_score = np.array([[0.1, 0.2, 0.8, 0.9, 0.91]], dtype=np.float32)
    total = np.array([[-0.4, -0.3, 0.3, 0.4, 0.41]], dtype=np.float64)
    source_valid = np.array([[True, False, True, True, True]], dtype=np.bool_)
    regions = np.array([[1, 1, 2, 2, 3]], dtype=np.uint32)
    displacement = np.array([[0.0, 0.0, 4.0, 4.0, 4.1]], dtype=np.float64)
    metric_valid = np.array([[True, False, True, True, True]], dtype=np.bool_)

    expanded = native_kd._require_native().expand_foreground_matte_one_pixel(
        near_score,
        total,
        source_valid,
        regions,
        displacement,
        metric_valid,
    )

    assert expanded.tolist() == [[False, True, False, False, False]]
    np.testing.assert_array_equal(
        near_score,
        np.array([[0.1, 0.8, 0.8, 0.9, 0.91]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        total,
        np.array([[-0.4, 0.3, 0.3, 0.4, 0.41]], dtype=np.float64),
    )
    assert source_valid.tolist() == [[True, True, True, True, True]]
    assert regions.tolist() == [[1, 2, 2, 2, 3]]
    assert metric_valid.tolist() == [[True, True, True, True, True]]


def test_foreground_matte_expands_only_a_large_nearer_surface_jump(monkeypatch) -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.geometry")
    near_score = np.array([[0.1, 0.2, 0.8, 0.9, 0.91]], dtype=np.float32)
    total = np.array([[-0.4, -0.3, 0.3, 0.4, 0.41]], dtype=np.float64)
    source_valid = np.ones((1, 5), dtype=np.bool_)
    regions = np.array([[1, 1, 2, 2, 3]], dtype=np.uint32)
    displacement = np.array([[0.0, 0.0, 4.0, 4.0, 4.1]], dtype=np.float64)

    with monkeypatch.context() as patch:
        patch.setattr(
            production.np,
            "copyto",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("production entered NumPy foreground-matte copying")
            ),
        )
        expanded = production._expand_foreground_matte_one_pixel(
            near_score,
            total,
            source_valid,
            regions,
            displacement,
            metric_valid=None,
        )

    assert expanded.tolist() == [[False, True, False, False, False]]
    np.testing.assert_array_equal(
        near_score,
        np.array([[0.1, 0.8, 0.8, 0.9, 0.91]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        total,
        np.array([[-0.4, 0.3, 0.3, 0.4, 0.41]], dtype=np.float64),
    )
    np.testing.assert_array_equal(regions, np.array([[1, 2, 2, 2, 3]], dtype=np.uint32))


def test_stable_surface_snaps_jagged_transition_to_persistent_depth_labels() -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.geometry")
    native_height, native_width = 48, 64
    render_shape = (96, 128)
    canonical = np.zeros((native_height, native_width), dtype=np.float32)
    canonical[:, native_width // 2 :] = np.float32(1.0)
    for row in range(native_height):
        canonical[row, native_width // 2 - 1 + row % 2] = np.float32(0.5)
    guide = np.zeros((*render_shape, 3), dtype=np.uint8)
    guide[:, render_shape[1] // 2 :] = np.uint8(255)

    result = production.build_relative_quality_geometry(
        guide,
        canonical,
        render_shape=render_shape,
        stereo_strength=5.0,
        convergence=0.5,
    )

    assert set(np.unique(result.near_score).tolist()) <= {0.0, 1.0}
    assert result.geometry_query_count == result.edge_band_pixel_count


def test_stable_surface_breaks_a_transitively_connected_depth_ramp() -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.geometry")
    region_solver = importlib.import_module("src.depth_surge_3d.rendering.quality.region_solver")
    native_height, native_width = 8, 64
    render_shape = (16, 1920)
    canonical = np.zeros((native_height, native_width), dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, 17, dtype=np.float32)
    canonical[:, 24:41] = ramp
    canonical[:, 41:] = np.float32(1.0)
    guide = np.zeros((*render_shape, 3), dtype=np.uint8)
    guide_edge = 1104
    guide[:, guide_edge:] = np.uint8(255)

    raw_regions = region_solver.build_low_resolution_regions(
        production._relative_displacement(
            canonical,
            render_shape[1],
            np.float64(1.2),
            np.float64(0.5),
        ),
        np.ones(canonical.shape, dtype=np.bool_),
        canonical,
    )
    assert set(np.unique(raw_regions.region_map).tolist()) == {1}

    result = production.build_relative_quality_geometry(
        guide,
        canonical,
        render_shape=render_shape,
        stereo_strength=1.2,
        convergence=0.5,
    )

    assert set(np.unique(result.near_score).tolist()) <= {0.0, 1.0}
    assert result.geometry_query_count > 0
    transitions = np.flatnonzero(np.diff(result.near_score[render_shape[0] // 2]))
    assert transitions.size == 1
    assert abs(int(transitions[0]) + 1 - guide_edge) <= 2
