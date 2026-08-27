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


def test_missing_lane_background_proxy_preserves_valid_pixel_contribution() -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    source = np.zeros((9, 9, 3), dtype=np.uint8)
    source[:, :, 1] = np.uint8(100)
    pre_repair = source.copy()
    coverage = np.full((9, 9), 16, dtype=np.uint8)
    coverage[4, 4] = np.uint8(8)
    pre_repair[4, 4] = np.asarray([50, 0, 0], dtype=np.uint8)
    coverage[4, 5] = np.uint8(0)
    pre_repair[4, 5] = np.uint8(0)

    repaired = quality._composite_background_proxy(
        pre_repair,
        coverage,
        source,
        radius_px=3,
        far_side="right",
        foreground_exclusion_px=3,
    )

    np.testing.assert_array_equal(repaired[coverage == 16], pre_repair[coverage == 16])
    np.testing.assert_array_equal(repaired[4, 4], np.asarray([50, 50, 0], dtype=np.uint8))
    np.testing.assert_allclose(
        repaired[4, 5],
        np.asarray([0, 100, 0], dtype=np.uint8),
        atol=2,
    )


def test_background_proxy_excludes_the_foreground_side_from_inpaint_donors() -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    source = np.zeros((9, 15, 3), dtype=np.uint8)
    source[:, :, 1] = np.uint8(100)
    pre_repair = source.copy()
    pre_repair[:, 4] = np.asarray([0, 0, 200], dtype=np.uint8)
    pre_repair[:, 5:10] = np.uint8(0)
    coverage = np.full((9, 15), 16, dtype=np.uint8)
    coverage[:, 5:10] = np.uint8(0)

    repaired = quality._composite_background_proxy(
        pre_repair,
        coverage,
        source,
        radius_px=3,
        far_side="right",
        foreground_exclusion_px=4,
    )

    assert int(repaired[:, 5:10, 2].max()) <= 5
    assert int(repaired[:, 5:10, 1].min()) >= 95


def test_background_proxy_mask_only_expands_low_coverage_holes() -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    coverage = np.full((7, 15), 16, dtype=np.uint8)
    coverage[3, 7] = np.uint8(0)
    coverage[1, 13] = np.uint8(15)
    coverage[5, 13] = np.uint8(8)

    mask = quality._build_background_proxy_mask(
        coverage,
        far_side="right",
        foreground_exclusion_px=4,
    )

    assert not mask[1, 13]
    assert not mask[0, 13]
    assert not mask[2, 13]
    assert mask[3, 3:10].all()
    assert mask[2, 3:10].all()
    assert mask[4, 3:10].all()
    assert not mask[3, 2]
    assert not mask[3, 10]
    assert mask[5, 13]


def test_hole_run_histograms_are_row_bounded_and_exact() -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    lane_valid = np.ones((2, 2, 16), dtype=np.bool_)
    missing = lane_valid.reshape(2, -1)
    missing[0, [0, 1, 15, 16, 17, 31]] = False
    missing[1, [0, 31]] = False
    lane_histogram = np.zeros(16 * (2 + 1), dtype=np.uint64)
    span_histogram = np.zeros(2 + 1, dtype=np.uint64)

    quality._accumulate_hole_run_histograms(
        lane_valid,
        lane_histogram,
        span_histogram,
    )

    assert lane_histogram[1] == 3
    assert lane_histogram[2] == 1
    assert lane_histogram[3] == 1
    assert int(lane_histogram.sum()) == 5
    assert span_histogram.tolist() == [0, 4, 1]


def test_quality_background_avoids_legacy_global_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    frame = np.full((9, 48, 3), 100, dtype=np.uint8)
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
    real_splat = quality._quality_splat
    splat_count = 0

    def count_splat(*args, **kwargs):
        nonlocal splat_count
        splat_count += 1
        return real_splat(*args, **kwargs)

    monkeypatch.setattr(quality, "_quality_splat", count_splat)
    monkeypatch.setattr(
        quality.QualityFixedArenas,
        "allocate",
        lambda: (_ for _ in ()).throw(AssertionError("allocated legacy repair arenas")),
    )
    monkeypatch.setattr(
        quality,
        "plan_quality_repairs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("entered legacy global repair planner")
        ),
    )
    monkeypatch.setattr(
        quality,
        "_replay_background_eye_once",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("replayed a second splat")),
    )

    result = quality.render_relative_quality_compact(
        renderer=StereoRenderer(device="cpu"),
        frame=frame,
        canonical=canonical,
        settings=_quality_settings(stereo_strength=5.0, convergence=0.45),
    )

    assert splat_count == 2
    for coverage, bits, plan in (
        (result.left_coverage_count, result.left_repair_bits, result.left_plan),
        (result.right_coverage_count, result.right_repair_bits, result.right_plan),
    ):
        assert plan is not None
        missing_lanes = int(np.sum(16 - coverage, dtype=np.uint64))
        assert plan.backend_lane_counts == (0, 0, missing_lanes)
        assert plan.fallback_indexed_donor_count == 0
        assert not np.any(bits & np.uint8(0x40))


def test_quality_compact_result_releases_dense_eye_plans_before_return() -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    diagnostics = importlib.import_module("src.depth_surge_3d.rendering.quality.diagnostics")
    frame = np.arange(7 * 33 * 3, dtype=np.uint8).reshape(7, 33, 3)
    canonical = np.tile(
        np.concatenate(
            (
                np.full(11, 0.1, dtype=np.float32),
                np.full(11, 0.9, dtype=np.float32),
                np.full(11, 0.2, dtype=np.float32),
            )
        ),
        (7, 1),
    )

    compact = quality.render_relative_quality_compact(
        renderer=StereoRenderer(device="cpu"),
        frame=frame,
        canonical=canonical,
        settings=_quality_settings(stereo_strength=4.0, convergence=0.45),
    )

    assert compact.left_plan is not None and compact.right_plan is not None
    assert not hasattr(compact.left_plan, "records")
    assert not hasattr(compact.right_plan, "records")
    stats = diagnostics.build_quality_frame_stats(
        compact,
        frame_name="frame_000001",
        geometry_mode="relative",
        occlusion_fill="background",
    )
    assert stats["eyes"]["left"]["quality_budgets"]["availability"] == ("quality_repair_plan")


def test_quality_public_budget_rejects_before_materializing_broadcast_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    content_manifest = importlib.import_module("src.depth_surge_3d.io.content_manifest")
    frame = np.broadcast_to(np.zeros((1, 1, 3), dtype=np.uint8), (10_000, 10_000, 3))
    canonical = np.zeros((1, 1), dtype=np.float32)
    monkeypatch.setattr(
        quality,
        "build_relative_quality_geometry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("geometry allocation preceded host budget preflight")
        ),
    )

    with pytest.raises(content_manifest.QualityHostBudgetError, match="quality-public"):
        quality.render_relative_quality_public(
            renderer=StereoRenderer(device="cpu"),
            frame=frame,
            canonical=canonical,
            settings=_quality_settings(),
        )


def test_quality_background_preflight_models_the_one_pass_proxy() -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    render_shape = (10, 20)
    native_shape = (5, 10)
    pixels = 200
    native_pixels = 50
    histogram_bytes = 272 * (render_shape[1] + 1)

    phases = quality._quality_public_memory_preflight(
        render_shape=render_shape,
        native_shape=native_shape,
        occlusion_fill="background",
        materialize_public_masks=False,
    )

    assert "quality-planning" not in phases
    assert "quality-pass-b" not in phases
    assert phases["quality-geometry-matte"] == (
        68 * pixels + 18 * native_pixels + quality.QUALITY_RUNTIME_OVERHEAD_BYTES
    )
    assert phases["quality-background-visibility"] == (
        40 * pixels + 9 * native_pixels + histogram_bytes + quality.QUALITY_RUNTIME_OVERHEAD_BYTES
    )
    assert phases["quality-background-proxy"] == (
        100 * pixels + 9 * native_pixels + histogram_bytes + quality.QUALITY_RUNTIME_OVERHEAD_BYTES
    )


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
        temporary_budget_bytes=9 * quality.QUALITY_SPLAT_BYTES_PER_PIXEL * 4,
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


def test_quality_band_height_uses_the_quality_device_coefficient() -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")

    assert quality.QUALITY_SPLAT_BYTES_PER_PIXEL == 1536
    assert quality._calculate_quality_band_height(3840, 2160, 256 * 1024 * 1024) == (
        256 * 1024 * 1024 // (3840 * 1536)
    )


def test_quality_second_oom_releases_device_state_and_reports_both_heights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    state = quality._FrameBandState(initial_height=8, retry_height=4, current_height=8)
    attempted_heights: list[int] = []
    releases: list[object] = []

    def always_oom(*, band_height: int, **_kwargs: object) -> None:
        attempted_heights.append(band_height)
        raise torch.cuda.OutOfMemoryError("injected")

    monkeypatch.setattr(quality, "_release_after_oom", lambda owner: releases.append(owner))
    source = np.zeros((8, 11, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="attempted band heights 8 and 4"):
        quality._retry_after_oom(
            object(),
            state,
            always_oom,
            source=source,
        )

    assert attempted_heights == [8, 4]
    assert len(releases) == 2


def test_cuda_oom_release_synchronizes_before_collecting_and_clearing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    renderer = StereoRenderer(device="cpu")
    renderer.device = torch.device("cuda")
    events: list[str] = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty"))
    monkeypatch.setattr(
        "src.depth_surge_3d.rendering.stereo_renderer.gc.collect",
        lambda: events.append("collect"),
    )

    renderer._release_after_oom()

    assert events == ["sync", "collect", "empty"]


def test_quality_none_retains_dense_uint64_hole_histograms() -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    frame = np.full((3, 17, 3), 80, dtype=np.uint8)
    canonical = np.tile(np.linspace(0.0, 1.0, 17, dtype=np.float32), (3, 1))

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

    for hole_runs in (result.left_hole_runs, result.right_hole_runs):
        assert hole_runs is not None
        assert hole_runs.lane_count_histogram.dtype == np.uint64
        assert hole_runs.lane_count_histogram.shape == (16 * (17 + 1),)
        assert hole_runs.touched_pixel_span_histogram.dtype == np.uint64
        assert hole_runs.touched_pixel_span_histogram.shape == (17 + 1,)


def test_quality_background_marks_only_missing_coverage_as_proxy_repaired() -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    frame = np.full((3, 17, 3), 80, dtype=np.uint8)
    canonical = np.tile(np.linspace(0.0, 1.0, 17, dtype=np.float32), (3, 1))

    result = quality.render_relative_quality_compact(
        renderer=StereoRenderer(device="cpu"),
        frame=frame,
        canonical=canonical,
        settings=_quality_settings(stereo_strength=5.0, convergence=0.0),
    )

    for coverage, bits in (
        (result.left_coverage_count, result.left_repair_bits),
        (result.right_coverage_count, result.right_repair_bits),
    ):
        missing = coverage < np.uint8(16)
        assert np.all((bits[missing] & np.uint8(0x28)) == np.uint8(0x28))
        assert not np.any(bits[missing] & np.uint8(0x40))
        assert not bits[~missing].any()


def test_quality_renderer_uses_the_frame_level_value_validation_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forward_splat = importlib.import_module("src.depth_surge_3d.rendering.forward_splat")
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    monkeypatch.setattr(
        forward_splat,
        "_validate_near_score",
        lambda _value: (_ for _ in ()).throw(
            AssertionError("Quality repeated a validated full-frame value scan")
        ),
    )
    frame = np.full((3, 17, 3), 80, dtype=np.uint8)
    canonical = np.tile(np.linspace(0.0, 1.0, 17, dtype=np.float32), (3, 1))

    result = quality.render_relative_quality_compact(
        renderer=StereoRenderer(device="cpu"),
        frame=frame,
        canonical=canonical,
        settings=_quality_settings(stereo_strength=5.0, convergence=0.0),
    )

    assert result.left_image.shape == frame.shape


def test_native_background_diagnostics_match_canonical_record_semantics() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    repair_records = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
    native = native_kd._require_native()
    records = np.zeros(3, dtype=repair_records.REPAIR_RECORD_DTYPE)
    records["pixel_index"] = (1, 1, 2)
    records["lane_mask"] = (0x0003, 0x000C, 0x00F0)
    records["backend"] = (1, 2, 3)
    raw_records = records.view(np.uint8).reshape(3, 16)
    coverage = np.asarray([[16, 8, 0, 16]], dtype=np.uint8)
    repair_bits = np.empty_like(coverage)
    backend_lane_counts = np.empty(3, dtype=np.uint64)
    lane_histogram = np.empty(16 * (coverage.shape[1] + 1), dtype=np.uint64)
    span_histogram = np.empty(coverage.shape[1] + 1, dtype=np.uint64)

    repair_lane_count = native.build_background_repair_bits_into(
        raw_records,
        coverage,
        repair_bits,
        backend_lane_counts,
    )
    histogram_lane_count = native.build_hole_run_histograms_into(
        raw_records,
        coverage.shape[1],
        lane_histogram,
        span_histogram,
    )

    assert repair_lane_count == histogram_lane_count == 8
    assert backend_lane_counts.tolist() == [2, 2, 4]
    assert repair_bits.tolist() == [[0, 0x01 | 0x04 | 0x18, 0x02 | 0x28, 0]]
    assert np.flatnonzero(lane_histogram).tolist() == [4]
    assert int(lane_histogram[4]) == 2
    assert np.flatnonzero(span_histogram).tolist() == [1]
    assert int(span_histogram[1]) == 2


def test_quality_background_proxy_never_expands_compact_records_to_host_lane_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    native = native_kd._require_native()
    monkeypatch.setattr(
        native,
        "expand_repair_scatter",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Pass B expanded compact records into per-lane host arrays")
        ),
    )
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

    result = quality.render_relative_quality_compact(
        renderer=StereoRenderer(device="cpu"),
        frame=frame,
        canonical=canonical,
        settings=_quality_settings(stereo_strength=5.0, convergence=0.45),
    )

    assert np.any(result.left_repair_bits & np.uint8(0x3C))


def test_quality_controls_reject_boolean_limit() -> None:
    types = importlib.import_module("src.depth_surge_3d.rendering.quality.types")

    with pytest.raises(ValueError, match="non-boolean integer"):
        types.QualityStereoControls("background", True)


def test_native_analysis_band_matches_the_existing_scalar_boundary() -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    forward_splat = importlib.import_module("src.depth_surge_3d.rendering.forward_splat")
    local_strip = importlib.import_module("src.depth_surge_3d.rendering.quality.local_strip")
    native_repair = importlib.import_module("src.depth_surge_3d.rendering.quality.native_repair")
    repair_records = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
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
    record_output = np.empty(
        height * fine_width,
        dtype=repair_records.REPAIR_RECORD_DTYPE,
    )
    actual_records = quality._write_analysis_band_native(
        actual_analysis,
        band,
        source_bgr.astype(np.uint8).reshape(height, width, 3),
        source_near.reshape(source_regions.shape),
        source_regions,
        0,
        height,
        record_output,
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
