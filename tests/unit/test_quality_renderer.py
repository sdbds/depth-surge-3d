from __future__ import annotations

import importlib
import threading

import cv2
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
    assert set(result.performance_seconds) == {
        "geometry_baseline",
        "geometry_region_solve",
        "geometry_index_build",
        "geometry_nearest_query",
        "geometry_finalize",
        "left_eye",
        "right_eye",
        "total",
    }
    assert all(value >= 0.0 for value in result.performance_seconds.values())
    assert result.performance_seconds["total"] > 0.0


def test_prepared_relative_quality_matches_monolithic_render() -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    generator = np.random.default_rng(20260828)
    frame = generator.integers(0, 256, size=(9, 32, 3), dtype=np.uint8)
    canonical = generator.random((5, 16), dtype=np.float32)
    settings = _quality_settings(
        stereo_strength=3.5,
        convergence=0.4,
        occlusion_fill="background",
    )

    monolithic = quality.render_relative_quality_compact(
        renderer=StereoRenderer(device="cpu"),
        frame=frame,
        canonical=canonical,
        settings=settings,
    )
    prepared = quality.prepare_relative_quality_frame(
        frame=frame,
        canonical=canonical,
        settings=settings,
    )
    split = quality.render_prepared_relative_quality_compact(
        renderer=StereoRenderer(device="cpu"),
        prepared=prepared,
    )

    for name in (
        "left_image",
        "right_image",
        "left_coverage_count",
        "right_coverage_count",
        "left_repair_bits",
        "right_repair_bits",
    ):
        assert np.array_equal(getattr(split, name), getattr(monolithic, name)), name
    assert split.left_plan == monolithic.left_plan
    assert split.right_plan == monolithic.right_plan
    assert split.geometry_nearest == monolithic.geometry_nearest
    assert split.quality_limits == monolithic.quality_limits


def test_quality_background_overlaps_left_repair_with_right_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    render_thread = threading.get_ident()
    finish_threads: list[int] = []
    original_finish = quality._finish_background_eye

    def record_finish(**kwargs):
        finish_threads.append(threading.get_ident())
        return original_finish(**kwargs)

    monkeypatch.setattr(quality, "_finish_background_eye", record_finish)
    quality.render_relative_quality_compact(
        renderer=StereoRenderer(device="cpu"),
        frame=np.full((8, 24, 3), 100, dtype=np.uint8),
        canonical=np.linspace(0.0, 1.0, 48, dtype=np.float32).reshape(4, 12),
        settings=_quality_settings(
            stereo_strength=4.0,
            occlusion_fill="background",
        ),
    )

    assert len(finish_threads) == 2
    assert render_thread in finish_threads
    assert any(thread != render_thread for thread in finish_threads)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_quality_compact_cpu_cuda_outputs_are_byte_exact() -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    generator = np.random.default_rng(20260828)
    frame = generator.integers(0, 256, size=(9, 48, 3), dtype=np.uint8)
    canonical = generator.random((5, 24), dtype=np.float32)
    settings = _quality_settings(stereo_strength=4.0, convergence=0.45)

    cpu = quality.render_relative_quality_compact(
        renderer=StereoRenderer(device="cpu"),
        frame=frame,
        canonical=canonical,
        settings=settings,
    )
    cuda = quality.render_relative_quality_compact(
        renderer=StereoRenderer(device="cuda"),
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
        assert np.array_equal(getattr(cuda, name), getattr(cpu, name))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_quality_cuda_avoids_dense_lane_validity_host_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    generator = np.random.default_rng(20260829)
    frame = generator.integers(0, 256, size=(9, 48, 3), dtype=np.uint8)
    canonical = generator.random((5, 24), dtype=np.float32)
    monkeypatch.setattr(
        quality,
        "_accumulate_hole_run_histograms",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("CUDA transferred the dense lane-validity raster")
        ),
    )

    result = quality.render_relative_quality_compact(
        renderer=StereoRenderer(device="cuda"),
        frame=frame,
        canonical=canonical,
        settings=_quality_settings(stereo_strength=4.0, convergence=0.45),
    )

    assert result.left_coverage_count.shape == frame.shape[:2]
    assert result.right_coverage_count.shape == frame.shape[:2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_quality_cuda_compact_gather_matches_hand_checked_lanes() -> None:
    fast_splat = importlib.import_module("src.depth_surge_3d.rendering.fast_splat_triton")
    operation = getattr(fast_splat, "triton_compact_quality_winners", None)
    assert operation is not None, "compact Quality CUDA gather is unavailable"

    source = torch.tensor(
        [[[16, 32, 48], [160, 176, 192]]],
        dtype=torch.uint8,
        device="cuda",
    )
    source_index_offset = 10
    source_zero = (1 << 32) | (0xFFFFFFFF - source_index_offset)
    source_one = (1 << 32) | (0xFFFFFFFF - source_index_offset - 1)
    winners = torch.tensor(
        [source_zero] * 8 + [source_one] * 4 + [-1] * 4 + [-1] * 16,
        dtype=torch.int64,
        device="cuda",
    )

    compact = operation(source, winners, source_index_offset)

    assert compact is not None
    colour, packed_missing = compact
    assert colour.cpu().tolist() == [[[48.0, 60.0, 72.0], [0.0, 0.0, 0.0]]]
    assert packed_missing.cpu().numpy().view(np.uint16).tolist() == [[0xF000, 0xFFFF]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_quality_cuda_compact_visibility_skips_general_splat_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    fast_splat = importlib.import_module("src.depth_surge_3d.rendering.fast_splat_triton")
    if not fast_splat.triton_splat_available(torch.device("cuda")):
        pytest.skip("Triton splat is unavailable")
    generator = np.random.default_rng(20260830)
    frame = generator.integers(0, 256, size=(9, 48, 3), dtype=np.uint8)
    canonical = generator.random((5, 24), dtype=np.float32)

    def reject_general_band(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("CUDA Quality materialized the general 16x splat band")

    monkeypatch.setattr(quality, "_quality_splat", reject_general_band)

    result = quality.render_relative_quality_compact(
        renderer=StereoRenderer(device="cuda"),
        frame=frame,
        canonical=canonical,
        settings=_quality_settings(stereo_strength=4.0, convergence=0.45),
    )

    assert result.left_image.shape == frame.shape
    assert result.right_image.shape == frame.shape
    assert result.left_coverage_count.shape == frame.shape[:2]
    assert result.right_coverage_count.shape == frame.shape[:2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_quality_cuda_compact_compile_failure_falls_back_byte_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    fast_splat = importlib.import_module("src.depth_surge_3d.rendering.fast_splat_triton")
    generator = np.random.default_rng(20260830)
    frame = generator.integers(0, 256, size=(7, 31, 3), dtype=np.uint8)
    canonical = generator.random((4, 16), dtype=np.float32)
    settings = _quality_settings(stereo_strength=4.0, convergence=0.45)
    expected = quality.render_relative_quality_compact(
        renderer=StereoRenderer(device="cpu"),
        frame=frame,
        canonical=canonical,
        settings=settings,
    )

    class BrokenKernel:
        def __getitem__(self, _grid):
            def launch(*_args: object, **_kwargs: object) -> None:
                raise fast_splat.triton.errors.TritonError("simulated compact compile failure")

            return launch

    monkeypatch.setattr(fast_splat, "_DISABLED_CUDA_DEVICES", set())
    monkeypatch.setattr(fast_splat, "_compact_quality_winners_kernel", BrokenKernel())

    actual = quality.render_relative_quality_compact(
        renderer=StereoRenderer(device="cuda"),
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
        assert np.array_equal(getattr(actual, name), getattr(expected, name)), name
    assert not fast_splat.triton_splat_available(torch.device("cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_quality_cuda_does_not_allocate_full_frame_lane_mask_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    generator = np.random.default_rng(20260830)
    frame = generator.integers(0, 256, size=(9, 48, 3), dtype=np.uint8)
    canonical = generator.random((5, 24), dtype=np.float32)
    original_empty = quality.torch.empty

    def guarded_empty(*args: object, **kwargs: object) -> torch.Tensor:
        size = args[0] if args else kwargs.get("size")
        device_value = kwargs.get("device")
        if (
            size is not None
            and tuple(size) == frame.shape[:2]  # type: ignore[arg-type]
            and kwargs.get("dtype") == torch.int16
            and device_value is not None
            and torch.device(device_value).type == "cuda"  # type: ignore[arg-type]
        ):
            raise AssertionError("CUDA allocated a full-frame packed lane mask")
        return original_empty(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(quality.torch, "empty", guarded_empty)

    result = quality.render_relative_quality_compact(
        renderer=StereoRenderer(device="cuda"),
        frame=frame,
        canonical=canonical,
        settings=_quality_settings(stereo_strength=4.0, convergence=0.45),
    )

    assert result.left_coverage_count.shape == frame.shape[:2]
    assert result.right_coverage_count.shape == frame.shape[:2]


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


def test_native_background_proxy_normalization_matches_float32_rounding() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    values = np.arange(256, dtype=np.uint8)
    pre_repair = np.empty((17, 256, 3), dtype=np.uint8)
    pre_repair[:, :, 0] = values
    pre_repair[:, :, 1] = values[::-1]
    pre_repair[:, :, 2] = np.bitwise_xor(values, np.uint8(0x55))
    coverage_shape = (17, 256)
    coverage = np.broadcast_to(
        np.arange(17, dtype=np.uint8)[:, None],
        coverage_shape,
    ).copy()
    expected = np.zeros_like(pre_repair)
    visible = coverage > np.uint8(0)
    normalized = pre_repair.astype(np.float32)
    normalized *= np.float32(16.0)
    np.divide(normalized, coverage[:, :, None], out=normalized, where=visible[:, :, None])
    expected[visible] = np.clip(np.rint(normalized[visible]), 0, 255).astype(np.uint8)

    actual = native_kd._require_native().normalize_background_proxy_source(
        pre_repair,
        coverage,
    )

    assert actual.shape[:2] == coverage_shape
    assert actual.dtype == np.uint8
    assert actual.flags.c_contiguous
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize(("far_side", "direction"), [("left", -1), ("right", 1)])
@pytest.mark.parametrize("foreground_exclusion_px", [1, 8, 32])
def test_native_background_proxy_preparation_matches_frozen_composed_oracle(
    far_side: str,
    direction: int,
    foreground_exclusion_px: int,
) -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    rng = np.random.default_rng(2026083105 + foreground_exclusion_px + direction)
    pre_repair = rng.integers(0, 256, size=(37, 53, 3), dtype=np.uint8)
    coverage = rng.integers(0, 17, size=pre_repair.shape[:2], dtype=np.uint8)
    native = native_kd._require_native()
    expected_proxy_source = native.normalize_background_proxy_source(
        pre_repair,
        coverage,
    )
    expected_mask = np.ascontiguousarray(
        quality._build_background_proxy_mask(
            coverage,
            far_side=far_side,
            foreground_exclusion_px=foreground_exclusion_px,
        ),
        dtype=np.uint8,
    )
    expected_mask *= np.uint8(255)
    expected_repair_bits = np.empty_like(coverage)
    expected_missing = np.empty_like(coverage)
    expected_missing_lanes, expected_missing_pixels = native.build_background_proxy_state_into(
        coverage,
        expected_repair_bits,
        expected_missing,
    )

    (
        proxy_source,
        mask,
        repair_bits,
        missing,
        missing_lanes,
        missing_pixels,
        masked_pixels,
        mask_bounds,
    ) = native.prepare_background_proxy(
        pre_repair,
        coverage,
        direction,
        foreground_exclusion_px,
    )

    assert proxy_source.tobytes() == expected_proxy_source.tobytes()
    assert mask.tobytes() == expected_mask.tobytes()
    assert repair_bits.tobytes() == expected_repair_bits.tobytes()
    assert missing.tobytes() == expected_missing.tobytes()
    assert int(missing_lanes) == int(expected_missing_lanes)
    assert int(missing_pixels) == int(expected_missing_pixels)
    assert int(masked_pixels) == int(np.count_nonzero(expected_mask))
    x, y, width, height = cv2.boundingRect(expected_mask)
    assert tuple(int(value) for value in mask_bounds) == (
        y,
        y + height,
        x,
        x + width,
    )
    for values in (proxy_source, mask, repair_bits, missing):
        assert values.dtype == np.uint8
        assert values.flags.c_contiguous


def test_native_background_proxy_preparation_rejects_invalid_coverage() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")

    with pytest.raises(ValueError, match="coverage exceeds"):
        native_kd._require_native().prepare_background_proxy(
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.asarray([[17]], dtype=np.uint8),
            1,
            8,
        )


def test_native_background_proxy_composition_matches_float32_rounding() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    rng = np.random.default_rng(160255)
    pre_repair = rng.integers(0, 256, size=(71, 257, 3), dtype=np.uint8)
    proxy = rng.integers(0, 256, size=pre_repair.shape, dtype=np.uint8)
    coverage = rng.integers(0, 17, size=pre_repair.shape[:2], dtype=np.uint8)
    coverage[:, :17] = np.arange(17, dtype=np.uint8)
    missing_lanes = np.float32(16.0) - coverage.astype(np.float32)
    repaired = pre_repair.astype(np.float32)
    repaired += proxy.astype(np.float32) * (missing_lanes[:, :, None] / np.float32(16.0))
    expected = np.clip(np.rint(repaired), 0, 255).astype(np.uint8)
    expected[coverage == np.uint8(16)] = pre_repair[coverage == np.uint8(16)]

    actual = native_kd._require_native().composite_background_proxy(
        pre_repair,
        coverage,
        proxy,
    )

    assert actual.dtype == np.uint8
    assert actual.flags.c_contiguous
    assert actual.tobytes() == expected.tobytes()


def test_background_proxy_bypasses_full_frame_float32_normalization(monkeypatch) -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    source = np.zeros((9, 9, 3), dtype=np.uint8)
    source[:, :, 1] = np.uint8(100)
    pre_repair = source.copy()
    coverage = np.full((9, 9), 16, dtype=np.uint8)
    coverage[4, 4] = np.uint8(8)
    pre_repair[4, 4] = np.asarray([50, 0, 0], dtype=np.uint8)
    coverage[4, 5] = np.uint8(0)
    pre_repair[4, 5] = np.uint8(0)
    monkeypatch.setattr(
        quality.np,
        "divide",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("production entered full-frame float32 normalization")
        ),
    )

    repaired = quality._composite_background_proxy(
        pre_repair,
        coverage,
        source,
        radius_px=3,
        far_side="right",
        foreground_exclusion_px=3,
    )

    assert repaired[4, 4].tolist() == [50, 50, 0]
    np.testing.assert_allclose(repaired[4, 5], [0, 100, 0], atol=2)


def test_background_finish_uses_fused_native_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    rng = np.random.default_rng(2026083107)
    source = rng.integers(0, 256, size=(11, 19, 3), dtype=np.uint8)
    pre_repair = source.copy()
    coverage = np.full(source.shape[:2], 16, dtype=np.uint8)
    coverage[2:9, 7:12] = rng.integers(0, 16, size=(7, 5), dtype=np.uint8)
    pre_repair[coverage < np.uint8(16)] //= np.uint8(2)
    expected_image = quality._composite_background_proxy(
        pre_repair,
        coverage,
        source,
        radius_px=1,
        far_side="right",
        foreground_exclusion_px=1,
    )
    expected_bits, expected_summary = quality._background_proxy_repair_state(coverage)
    hole_runs = quality.HoleRunDiagnostics(
        lane_count_histogram=np.zeros(16 * 20, dtype=np.uint64),
        touched_pixel_span_histogram=np.zeros(20, dtype=np.uint64),
    )
    unrepaired = quality._UnrepairedEyeResult(pre_repair, coverage, hole_runs)

    def old_path_error(*_args, **_kwargs):
        raise AssertionError("production entered the composed background proxy path")

    monkeypatch.setattr(quality, "_composite_background_proxy", old_path_error)
    monkeypatch.setattr(quality, "_background_proxy_repair_state", old_path_error)
    monkeypatch.setattr(quality, "_build_background_proxy_mask", old_path_error)

    actual = quality._finish_background_eye(
        pre_repair=unrepaired,
        source=source,
        eye="right",
        controls_limit_px=4,
        predicted_gap_px=4,
    )

    assert actual.image.tobytes() == expected_image.tobytes()
    assert actual.repair_bits.tobytes() == expected_bits.tobytes()
    assert actual.plan == expected_summary
    assert actual.coverage_count is coverage
    assert actual.hole_runs is hole_runs


@pytest.mark.parametrize("radius_px", [1, 2, 3])
def test_background_proxy_roi_inpaint_matches_full_frame_exactly(
    radius_px: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    rng = np.random.default_rng(2026083111 + radius_px)
    proxy_source = rng.integers(0, 256, size=(97, 151, 3), dtype=np.uint8)
    mask = np.zeros(proxy_source.shape[:2], dtype=np.uint8)
    mask[21:58, 39:91] = np.uint8(255)
    mask[32:47, 50:77] = np.uint8(0)
    bounds = (21, 58, 39, 91)
    expected = cv2.inpaint(
        proxy_source,
        mask,
        float(radius_px),
        cv2.INPAINT_TELEA,
    )
    original_inpaint = quality.cv2.inpaint
    inpaint_shapes: list[tuple[int, int]] = []

    def recording_inpaint(image, inpaint_mask, radius, method):
        inpaint_shapes.append(tuple(image.shape[:2]))
        return original_inpaint(image, inpaint_mask, radius, method)

    monkeypatch.setattr(quality.cv2, "inpaint", recording_inpaint)
    actual = quality._inpaint_background_proxy(
        np.array(proxy_source, copy=True, order="C"),
        mask,
        radius_px=radius_px,
        mask_bounds=bounds,
    )

    assert inpaint_shapes == [
        (
            bounds[1] - bounds[0] + 2 * (radius_px + 1),
            bounds[3] - bounds[2] + 2 * (radius_px + 1),
        )
    ]
    assert actual.tobytes() == expected.tobytes()


def test_background_proxy_roi_inpaint_skips_empty_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    proxy_source = np.arange(9 * 13 * 3, dtype=np.uint8).reshape(9, 13, 3)
    mask = np.zeros((9, 13), dtype=np.uint8)
    monkeypatch.setattr(
        quality.cv2,
        "inpaint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("empty background proxy mask entered OpenCV inpaint")
        ),
    )

    actual = quality._inpaint_background_proxy(
        proxy_source,
        mask,
        radius_px=3,
        mask_bounds=(0, 0, 0, 0),
    )

    assert actual is proxy_source


def test_native_lane_analysis_fuses_coverage_and_row_bounded_histograms() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    lane_valid = np.ones((2, 2, 16), dtype=np.bool_)
    missing = lane_valid.reshape(2, -1)
    missing[0, [0, 1, 15, 16, 17, 31]] = False
    missing[1, [0, 31]] = False
    coverage = np.empty((2, 2), dtype=np.uint8)
    lane_histogram = np.zeros(16 * (2 + 1), dtype=np.uint64)
    span_histogram = np.zeros(2 + 1, dtype=np.uint64)

    native_kd._require_native().accumulate_lane_hole_run_histograms_into(
        lane_valid,
        lane_histogram,
        span_histogram,
        coverage,
    )

    assert coverage.tolist() == [[13, 13], [15, 15]]
    assert lane_histogram[1] == 3
    assert lane_histogram[2] == 1
    assert lane_histogram[3] == 1
    assert int(lane_histogram.sum()) == 5
    assert span_histogram.tolist() == [0, 4, 1]


def test_native_lane_analysis_matches_scalar_randomized_oracle() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    rng = np.random.default_rng(20260828)
    lane_valid = np.ascontiguousarray(rng.random((7, 19, 16)) > 0.18)
    expected_coverage = lane_valid.sum(axis=2, dtype=np.uint16).astype(np.uint8)
    expected_lane_histogram = np.zeros(16 * 20, dtype=np.uint64)
    expected_span_histogram = np.zeros(20, dtype=np.uint64)
    expected_run_count = 0
    for row in lane_valid:
        active_start: int | None = None
        for lane, valid in enumerate(row.reshape(-1)):
            if not valid and active_start is None:
                active_start = lane
            if valid and active_start is not None:
                lane_length = lane - active_start
                pixel_span = (lane - 1) // 16 - active_start // 16 + 1
                expected_lane_histogram[lane_length] += np.uint64(1)
                expected_span_histogram[pixel_span] += np.uint64(1)
                expected_run_count += 1
                active_start = None
        if active_start is not None:
            lane_length = row.size - active_start
            pixel_span = (row.size - 1) // 16 - active_start // 16 + 1
            expected_lane_histogram[lane_length] += np.uint64(1)
            expected_span_histogram[pixel_span] += np.uint64(1)
            expected_run_count += 1

    coverage = np.empty(expected_coverage.shape, dtype=np.uint8)
    lane_histogram = np.zeros_like(expected_lane_histogram)
    span_histogram = np.zeros_like(expected_span_histogram)
    run_count = native_kd._require_native().accumulate_lane_hole_run_histograms_into(
        lane_valid,
        lane_histogram,
        span_histogram,
        coverage,
    )

    assert run_count == expected_run_count
    np.testing.assert_array_equal(coverage, expected_coverage)
    np.testing.assert_array_equal(lane_histogram, expected_lane_histogram)
    np.testing.assert_array_equal(span_histogram, expected_span_histogram)


def test_native_packed_lane_analysis_matches_bool_input() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    native = native_kd._require_native()
    rng = np.random.default_rng(20260829)
    lane_valid = np.ascontiguousarray(rng.random((7, 19, 16)) > 0.18)
    lane_valid[0, 0] = True
    lane_valid[0, 1] = False
    lane_valid[0, 2] = True
    lane_valid[0, 2, 15] = False
    lane_weights = np.left_shift(
        np.uint32(1),
        np.arange(16, dtype=np.uint32),
    )
    packed_missing = np.sum(
        np.logical_not(lane_valid).astype(np.uint32) * lane_weights,
        axis=2,
        dtype=np.uint32,
    ).astype(np.uint16)

    expected_coverage = np.empty((7, 19), dtype=np.uint8)
    expected_lane_histogram = np.zeros(16 * 20, dtype=np.uint64)
    expected_span_histogram = np.zeros(20, dtype=np.uint64)
    expected_run_count = native.accumulate_lane_hole_run_histograms_into(
        lane_valid,
        expected_lane_histogram,
        expected_span_histogram,
        expected_coverage,
    )
    coverage = np.empty_like(expected_coverage)
    lane_histogram = np.zeros_like(expected_lane_histogram)
    span_histogram = np.zeros_like(expected_span_histogram)

    run_count = native.accumulate_packed_lane_hole_run_histograms_into(
        packed_missing,
        lane_histogram,
        span_histogram,
        coverage,
    )

    assert run_count == expected_run_count
    np.testing.assert_array_equal(coverage, expected_coverage)
    np.testing.assert_array_equal(lane_histogram, expected_lane_histogram)
    np.testing.assert_array_equal(span_histogram, expected_span_histogram)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_lane_mask_packing_is_bit_exact() -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    rng = np.random.default_rng(20260829)
    lane_valid_host = np.ascontiguousarray(rng.random((7, 19, 16)) > 0.18)
    lane_valid_host[0, 0] = True
    lane_valid_host[0, 1] = False
    lane_valid_host[0, 2] = True
    lane_valid_host[0, 2, 15] = False
    lane_valid = torch.as_tensor(lane_valid_host, device="cuda")
    lane_weights = torch.bitwise_left_shift(
        torch.ones(16, dtype=torch.int32, device="cuda"),
        torch.arange(16, dtype=torch.int32, device="cuda"),
    )
    expected = np.sum(
        np.logical_not(lane_valid_host).astype(np.uint32)
        * np.left_shift(np.uint32(1), np.arange(16, dtype=np.uint32)),
        axis=2,
        dtype=np.uint32,
    ).astype(np.uint16)

    actual = quality._pack_missing_lane_masks(lane_valid, lane_weights)

    assert actual.dtype == torch.int16
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.cpu().numpy().view(np.uint16), expected)


def test_native_background_proxy_state_fuses_bits_mask_and_counts() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    coverage = np.asarray([[16, 15, 1, 0]], dtype=np.uint8)
    repair_bits = np.empty_like(coverage)
    missing_mask = np.empty_like(coverage)

    missing_lanes, missing_pixels = native_kd._require_native().build_background_proxy_state_into(
        coverage,
        repair_bits,
        missing_mask,
    )

    assert repair_bits.tolist() == [[0, 0x29, 0x29, 0x2A]]
    assert missing_mask.tolist() == [[0, 1, 1, 1]]
    assert missing_lanes == 32
    assert missing_pixels == 3


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


def test_quality_background_avoids_intermediate_none_repair_bits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    frame = np.full((9, 48, 3), 100, dtype=np.uint8)
    canonical = np.tile(
        np.linspace(0.0, 1.0, 24, dtype=np.float32),
        (5, 1),
    )
    monkeypatch.setattr(
        quality,
        "_none_repair_bits",
        lambda _coverage: (_ for _ in ()).throw(
            AssertionError("background path materialized intermediate None repair bits")
        ),
    )

    quality.render_relative_quality_compact(
        renderer=StereoRenderer(device="cpu"),
        frame=frame,
        canonical=canonical,
        settings=_quality_settings(
            stereo_strength=5.0,
            convergence=0.45,
            occlusion_fill="background",
        ),
    )


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


@pytest.mark.parametrize("occlusion_fill", ["none", "background"])
@pytest.mark.parametrize("materialize_public_masks", [False, True])
def test_quality_geodesic_preflight_charges_cached_priorities_without_raising_peak(
    occlusion_fill, materialize_public_masks
) -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    pixels = 1080 * 1920
    native_pixels = 608 * 1080
    phases = quality._quality_public_memory_preflight(
        render_shape=(1080, 1920),
        native_shape=(608, 1080),
        occlusion_fill=occlusion_fill,
        materialize_public_masks=materialize_public_masks,
    )
    previous_region_bytes = (
        52 * pixels
        + 18 * native_pixels
        + quality.QUALITY_RUNTIME_OVERHEAD_BYTES
        + quality.QUALITY_AUXILIARY_THREAD_BYTES
    )

    assert phases["quality-region"] == previous_region_bytes + 16 * pixels
    previous_phases = dict(phases, **{"quality-region": previous_region_bytes})
    assert max(phases.values()) == max(previous_phases.values())


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
        68 * pixels
        + 18 * native_pixels
        + quality.QUALITY_RUNTIME_OVERHEAD_BYTES
        + quality.QUALITY_AUXILIARY_THREAD_BYTES
    )
    assert phases["quality-background-visibility"] == (
        40 * pixels
        + 9 * native_pixels
        + histogram_bytes
        + quality.QUALITY_RUNTIME_OVERHEAD_BYTES
        + quality.QUALITY_AUXILIARY_THREAD_BYTES
    )
    assert phases["quality-background-proxy"] == (
        100 * pixels
        + 9 * native_pixels
        + histogram_bytes
        + quality.QUALITY_RUNTIME_OVERHEAD_BYTES
        + quality.QUALITY_AUXILIARY_THREAD_BYTES
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


def test_quality_cuda_budget_uses_only_a_bounded_share_of_live_free_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    renderer = StereoRenderer(device="cpu")
    renderer.device = torch.device("cuda")

    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda _device: (8 * 1024**3, 24 * 1024**3),
    )
    assert quality._quality_temporary_budget_bytes(renderer) == 512 * 1024 * 1024

    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda _device: (2 * 1024**3, 24 * 1024**3),
    )
    assert quality._quality_temporary_budget_bytes(renderer) == 256 * 1024 * 1024

    explicitly_smaller = StereoRenderer(
        device="cpu",
        temporary_budget_bytes=256 * 1024 * 1024,
    )
    explicitly_smaller.device = torch.device("cuda")
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda _device: (8 * 1024**3, 24 * 1024**3),
    )
    assert quality._quality_temporary_budget_bytes(explicitly_smaller) == 256 * 1024 * 1024

    explicitly_larger = StereoRenderer(
        device="cpu",
        temporary_budget_bytes=768 * 1024 * 1024,
    )
    explicitly_larger.device = torch.device("cuda")
    assert quality._quality_temporary_budget_bytes(explicitly_larger) == 768 * 1024 * 1024


def test_quality_preloaded_device_frame_preserves_banded_eye_and_uploads_offsets_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    generator = np.random.default_rng(20260829)
    source = generator.integers(0, 256, size=(6, 19, 3), dtype=np.uint8)
    near_score = generator.random((6, 19), dtype=np.float32)
    source_valid = generator.random((6, 19)) > 0.1
    offsets = generator.integers(-24, 25, size=(6, 19), dtype=np.int32)

    class Geometry:
        pass

    geometry = Geometry()
    geometry.near_score = np.ascontiguousarray(near_score)
    geometry.source_valid = np.ascontiguousarray(source_valid)
    renderer = StereoRenderer(device="cpu")
    transfer_shapes: list[tuple[int, ...]] = []

    def record_transfer(values: np.ndarray) -> torch.Tensor:
        transfer_shapes.append(values.shape)
        return torch.from_numpy(values)

    monkeypatch.setattr(renderer, "_transfer_offset_band", record_transfer)
    baseline = quality._render_unrepaired_eye_once(
        renderer=renderer,
        source=source,
        geometry=geometry,
        offsets=offsets,
        band_height=2,
    )
    assert transfer_shapes == [(2, 19), (2, 19), (2, 19)]

    transfer_shapes.clear()
    preloaded = quality._render_unrepaired_eye_once(
        renderer=renderer,
        source=source,
        geometry=geometry,
        offsets=offsets,
        band_height=2,
        device_frame=quality._QualityDeviceFrame(
            source=torch.from_numpy(source),
            near_score=torch.from_numpy(geometry.near_score),
            source_valid=torch.from_numpy(geometry.source_valid),
        ),
    )

    assert transfer_shapes == [(6, 19)]
    assert np.array_equal(preloaded.image, baseline.image)
    assert np.array_equal(preloaded.coverage_count, baseline.coverage_count)
    assert np.array_equal(
        preloaded.hole_runs.lane_count_histogram,
        baseline.hole_runs.lane_count_histogram,
    )
    assert np.array_equal(
        preloaded.hole_runs.touched_pixel_span_histogram,
        baseline.hole_runs.touched_pixel_span_histogram,
    )


def test_quality_preloaded_frame_falls_back_when_full_offset_upload_ooms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    source = np.zeros((4, 7, 3), dtype=np.uint8)
    offsets = np.zeros((4, 7), dtype=np.int32)

    class Geometry:
        pass

    geometry = Geometry()
    geometry.near_score = np.zeros((4, 7), dtype=np.float32)
    geometry.source_valid = np.ones((4, 7), dtype=np.bool_)
    renderer = StereoRenderer(device="cpu")
    transfers: list[tuple[int, ...]] = []

    def transfer(values: np.ndarray) -> torch.Tensor:
        transfers.append(values.shape)
        if values.shape == offsets.shape:
            raise torch.cuda.OutOfMemoryError("simulated full offset upload OOM")
        return torch.from_numpy(values)

    monkeypatch.setattr(renderer, "_transfer_offset_band", transfer)
    result = quality._render_unrepaired_eye_once(
        renderer=renderer,
        source=source,
        geometry=geometry,
        offsets=offsets,
        band_height=2,
        device_frame=quality._QualityDeviceFrame(
            source=torch.from_numpy(source),
            near_score=torch.from_numpy(geometry.near_score),
            source_valid=torch.from_numpy(geometry.source_valid),
        ),
    )

    assert transfers == [(4, 7), (2, 7), (2, 7)]
    assert np.array_equal(result.image, source)


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
