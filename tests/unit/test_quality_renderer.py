from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from src.depth_surge_3d.rendering.stereo_renderer import (
    SPLAT_BYTES_PER_PIXEL,
    StereoRenderer,
    StereoRenderSettings,
)


def _quality_settings(**values: object) -> StereoRenderSettings:
    return StereoRenderSettings(
        stereo_render_mode="quality",
        **values,  # type: ignore[arg-type]
    )


def test_quality_strength_zero_is_exact_and_has_no_repair_bits() -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    generator = np.random.default_rng(20260824)
    frame = generator.integers(0, 256, size=(5, 13, 3), dtype=np.uint8)
    canonical = generator.random((3, 7), dtype=np.float32)

    result = quality.render_relative_quality_compact(
        renderer=StereoRenderer(device="cpu"),
        frame=frame,
        canonical=canonical,
        settings=_quality_settings(stereo_strength=0.0),
    )

    assert np.array_equal(result.left_image, frame)
    assert np.array_equal(result.right_image, frame)
    assert np.all(result.left_coverage_count == 16)
    assert np.all(result.right_coverage_count == 16)
    assert not result.left_repair_bits.any()
    assert not result.right_repair_bits.any()


def test_quality_none_records_partial_and_full_holes_without_repair() -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    frame = np.full((5, 32, 3), 100, dtype=np.uint8)
    canonical = np.ones((2, 8), dtype=np.float32)

    result = quality.render_relative_quality_compact(
        renderer=StereoRenderer(device="cpu"),
        frame=frame,
        canonical=canonical,
        settings=_quality_settings(
            stereo_strength=5.0,
            convergence=0.0,
            occlusion_fill="none",
        ),
    )

    left_missing = result.left_coverage_count < 16
    assert left_missing.any()
    assert np.all(result.left_repair_bits[left_missing] & np.uint8(0x40))
    assert not np.any(result.left_repair_bits & np.uint8(0x3C))
    assert np.any(result.left_repair_bits & np.uint8(0x01))


def test_quality_background_finishes_every_lane_and_public_holes_are_clear() -> None:
    generator = np.random.default_rng(29)
    frame = generator.integers(0, 256, size=(9, 48, 3), dtype=np.uint8)
    canonical = np.tile(
        np.concatenate(
            [
                np.full(12, 0.15, dtype=np.float32),
                np.full(12, 0.9, dtype=np.float32),
                np.full(24, 0.2, dtype=np.float32),
            ]
        ),
        (9, 1),
    )
    renderer = StereoRenderer(device="cpu")

    result = renderer.render(
        frame,
        canonical,
        _quality_settings(stereo_strength=5.0, convergence=0.45),
    )

    assert not result.left_hole_mask.any()
    assert not result.right_hole_mask.any()
    assert result.left_valid_mask.any() and result.right_valid_mask.any()
    assert result.left_image.dtype == frame.dtype
    assert result.right_image.dtype == frame.dtype


def test_quality_band_height_does_not_change_rgb_or_compact_masks() -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    generator = np.random.default_rng(37)
    frame = generator.integers(0, 256, size=(8, 41, 3), dtype=np.uint8)
    canonical = generator.random((4, 17), dtype=np.float32)
    settings = _quality_settings(
        stereo_strength=3.0,
        convergence=0.4,
        occlusion_fill="none",
    )

    full = quality.render_relative_quality_compact(
        renderer=StereoRenderer(device="cpu"),
        frame=frame,
        canonical=canonical,
        settings=settings,
    )
    one_row = quality.render_relative_quality_compact(
        renderer=StereoRenderer(
            device="cpu",
            temporary_budget_bytes=41 * SPLAT_BYTES_PER_PIXEL,
        ),
        frame=frame,
        canonical=canonical,
        settings=settings,
    )

    for name in (
        "left_image",
        "right_image",
        "left_coverage_count",
        "right_coverage_count",
        "left_repair_bits",
        "right_repair_bits",
    ):
        assert np.array_equal(getattr(full, name), getattr(one_row, name))


def test_quality_pass_retries_once_from_the_start_after_cuda_oom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    frame = np.zeros((4, 9, 3), dtype=np.uint8)
    canonical = np.full((2, 4), 0.5, dtype=np.float32)
    renderer = StereoRenderer(
        device="cpu",
        temporary_budget_bytes=9 * SPLAT_BYTES_PER_PIXEL * 4,
    )
    original = quality._render_none_eye_once
    attempted_heights: list[int] = []

    def fail_once(*args, **kwargs):
        attempted_heights.append(int(kwargs["band_height"]))
        if len(attempted_heights) == 1:
            raise torch.cuda.OutOfMemoryError("injected")
        return original(*args, **kwargs)

    monkeypatch.setattr(quality, "_render_none_eye_once", fail_once)

    result = quality.render_relative_quality_compact(
        renderer=renderer,
        frame=frame,
        canonical=canonical,
        settings=_quality_settings(occlusion_fill="none"),
    )

    assert attempted_heights[:2] == [4, 2]
    assert result.left_image.shape == frame.shape


def test_quality_controls_reject_boolean_limit() -> None:
    types = importlib.import_module("src.depth_surge_3d.rendering.quality.types")

    with pytest.raises(ValueError, match="non-boolean integer"):
        types.QualityStereoControls("background", True)


def test_native_analysis_band_matches_the_existing_scalar_boundary() -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    forward_splat = importlib.import_module("src.depth_surge_3d.rendering.forward_splat")
    local_strip = importlib.import_module("src.depth_surge_3d.rendering.quality.local_strip")
    native_repair = importlib.import_module("src.depth_surge_3d.rendering.quality.native_repair")
    visibility = importlib.import_module("src.depth_surge_3d.rendering.quality.visibility")
    height, width = 2, 3
    fine_width = width * 16
    valid = np.ones((height, fine_width), dtype=np.bool_)
    valid[0, 5:9] = False
    valid[0, 22:25] = False
    valid[1, :2] = False
    winner_sources = np.empty((height, fine_width), dtype=np.int64)
    for column in range(width):
        winner_sources[:, column * 16 : (column + 1) * 16] = (0, 2, 1)[column]
    winner_sources[0, 43] = 2
    winner_sources[~valid] = 0xFFFFFFFF
    source_regions = np.array([[1, 1, 2], [1, 2, 2]], dtype=np.uint32)
    near_score = np.zeros((height, fine_width), dtype=np.float32)
    colour = np.zeros((height, fine_width, 3), dtype=np.float32)
    source_near = np.array([0.25, 0.5, 0.75, 0.3, 0.7, 0.8], dtype=np.float32)
    source_bgr = np.array(
        [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
            [15, 25, 35],
            [45, 55, 65],
            [75, 85, 95],
        ],
        dtype=np.float32,
    )
    for sample in np.flatnonzero(valid):
        source_index = int(winner_sources.flat[int(sample)])
        near_score.flat[int(sample)] = source_near[source_index]
        colour.reshape(-1, 3)[int(sample)] = source_bgr[source_index]
    band = visibility.QualitySplatBand(
        splat=forward_splat.SubpixelSplatResult(
            colour=torch.from_numpy(colour),
            disparity=torch.from_numpy(near_score),
            valid=torch.from_numpy(valid),
        ),
        winner_source_index=torch.from_numpy(winner_sources),
    )

    def empty_analysis():
        return local_strip.QualityVisibilityAnalysis(
            np.empty((height, width), dtype=np.uint8),
            np.empty((height, width), dtype=np.uint32),
            tuple(np.empty((height, width), dtype=np.uint8) for _ in range(3)),
            anchor_fallbacks_seeded=True,
        )

    expected_analysis = empty_analysis()
    expected_inputs = quality._write_analysis_band(
        expected_analysis,
        band,
        source_regions,
        0,
        height,
    )
    expected_records = native_repair.build_native_repair_records(
        expected_inputs[0],
        expected_inputs[1],
        expected_inputs[2],
        winner_bgr=expected_inputs[3],
    ).records

    actual_analysis = empty_analysis()
    actual_records = quality._write_analysis_band_native(
        actual_analysis,
        band,
        source_bgr.astype(np.uint8).reshape(height, width, 3),
        source_near.reshape(source_regions.shape),
        source_regions,
        0,
        height,
    ).records

    assert np.array_equal(actual_analysis.coverage_count, expected_analysis.coverage_count)
    assert np.array_equal(actual_analysis.pure_region_id, expected_analysis.pure_region_id)
    for actual, expected in zip(
        actual_analysis.pure_bgr,
        expected_analysis.pure_bgr,
        strict=True,
    ):
        assert np.array_equal(actual, expected)
    assert actual_records.tobytes() == expected_records.tobytes()


def test_metric_quality_renders_native_primitives_and_returns_final_stats() -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    types = importlib.import_module("src.depth_surge_3d.rendering.quality.types")
    metric_geometry = importlib.import_module(
        "src.depth_surge_3d.processing.frames.metric_geometry"
    )
    frame = np.arange(6 * 11 * 3, dtype=np.uint8).reshape(6, 11, 3)
    metric = metric_geometry.MetricGeometryFrame(
        np.array([[1.0, 0.8, 0.0], [0.7, 0.5, 0.3]], dtype=np.float32),
        np.array([[True, True, False], [True, True, True]], dtype=np.bool_),
        np.float32(0.5),
    )
    primitives = types.MetricStereoPrimitiveInput(
        metric=metric,
        virtual_baseline_mm=np.float64(63.0),
        convergence_distance_m=np.float64(2.0),
        max_disparity_percent=np.float64(2.0),
        retained_crop_width=11,
    )

    result = quality.render_metric_primitives_compact(
        renderer=StereoRenderer(device="cpu"),
        frame=frame,
        primitives=primitives,
        controls=types.QualityStereoControls("none", 8),
    )

    assert result.left_image.shape == frame.shape
    assert result.right_image.shape == frame.shape
    assert result.metric_stats is not None
    assert result.metric_stats.valid_pixel_count > 0
    assert result.geometry_nearest == {
        "indexed_sample_count": 6,
        "query_count": result.geometry_nearest["query_count"],
        "visited_nodes_total": result.geometry_nearest["visited_nodes_total"],
        "visited_nodes_max": result.geometry_nearest["visited_nodes_max"],
    }
