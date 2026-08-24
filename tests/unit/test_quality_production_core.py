from __future__ import annotations

import importlib

import numpy as np

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


def test_production_single_eye_offset_builders_match_gate_prototype() -> None:
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
    assert native.native_build_info() == {
        "algorithm": "quality-implicit-region-kd-median-of-medians-v1",
        "kd_stack_capacity": 64,
        "runtime_jit": False,
    }
