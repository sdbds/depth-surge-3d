"""Fast relative preparation must keep only the raster data its renderer uses."""

from dataclasses import fields

import numpy as np
import pytest
import torch

from src.depth_surge_3d.rendering import stereo_renderer as renderer_module
from src.depth_surge_3d.rendering.stereo_geometry import build_relative_geometry


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return request.param


def _legacy_compact(renderer, source, canonical, settings):
    geometry = build_relative_geometry(
        canonical,
        source.shape[:2],
        stereo_strength=settings.stereo_strength,
        convergence=settings.convergence,
    )
    near_score = torch.from_numpy(np.array(geometry.near_score, copy=True, order="C"))
    offsets = renderer_module.calculate_eye_sample_offsets(near_score, settings)
    return renderer._render_fast_compact_core(
        np.ascontiguousarray(source),
        geometry,
        offsets,
        renderer_module.StereoSplatSettings(
            max_eye_shift_fraction=settings.stereo_strength / 200.0,
            occlusion_fill=settings.occlusion_fill,
        ),
    )


def _assert_equal(actual, expected):
    for name in (
        "left_image",
        "right_image",
        "left_coverage_count",
        "right_coverage_count",
        "left_repair_bits",
        "right_repair_bits",
    ):
        actual_array = getattr(actual, name)
        expected_array = getattr(expected, name)
        assert actual_array.dtype == expected_array.dtype
        np.testing.assert_array_equal(actual_array, expected_array)
    for name in ("left_fast_diagnostics", "right_fast_diagnostics"):
        actual_eye = getattr(actual, name)
        expected_eye = getattr(expected, name)
        assert actual_eye.local_filled_lane_count == expected_eye.local_filled_lane_count
        assert actual_eye.final_unresolved_lane_count == expected_eye.final_unresolved_lane_count
        for histogram in ("lane_count_histogram", "touched_pixel_span_histogram"):
            np.testing.assert_array_equal(
                getattr(actual_eye.hole_runs, histogram),
                getattr(expected_eye.hole_runs, histogram),
            )


def test_fast_relative_renders_without_unused_common_disparity(monkeypatch, device):
    rng = np.random.default_rng(20260831)
    source = rng.integers(0, 256, size=(7, 23, 3), dtype=np.uint8)
    canonical = rng.random((5, 13), dtype=np.float32)
    settings = renderer_module.StereoRenderSettings(stereo_strength=3.75, convergence=0.42)
    renderer = renderer_module.StereoRenderer(device=device)
    expected = _legacy_compact(renderer, source, canonical, settings)

    def reject_unused_plane(*_args, **_kwargs):
        raise AssertionError("Fast relative projection must not build unused common disparity")

    monkeypatch.setattr(renderer_module, "build_relative_geometry", reject_unused_plane)
    actual = renderer.render_relative_fast_compact(source, canonical, settings)
    _assert_equal(actual, expected)


@pytest.mark.parametrize("layout", ["contiguous", "readonly", "negative", "fortran"])
def test_fast_relative_retains_only_owned_near_score_and_validity(monkeypatch, layout):
    source = np.full((3, 11, 3), 123, dtype=np.uint8)
    canonical = np.linspace(0.0, 1.0, 33, dtype=np.float32).reshape(3, 11)
    if layout == "readonly":
        canonical.setflags(write=False)
    elif layout == "negative":
        canonical = canonical[:, ::-1]
    elif layout == "fortran":
        canonical = np.asfortranarray(canonical)
    before = canonical.copy()
    renderer = renderer_module.StereoRenderer(device="cpu")
    settings = renderer_module.StereoRenderSettings()
    expected = _legacy_compact(renderer, source, canonical, settings)
    original_core = renderer._render_fast_compact_core
    observed = []

    def inspect_prepared(frame, geometry, offsets, splat_settings, **kwargs):
        retained = sum(
            value.nbytes
            for field in fields(geometry)
            if isinstance(value := getattr(geometry, field.name), np.ndarray)
        )
        assert retained <= source.shape[0] * source.shape[1] * 5
        assert not np.shares_memory(geometry.near_score, canonical)
        assert geometry.near_score.dtype == np.float32
        assert geometry.near_score.flags.c_contiguous
        np.testing.assert_array_equal(geometry.near_score, before)
        assert geometry.source_valid.dtype == np.bool_
        assert geometry.source_valid.all()
        observed.append(retained)
        return original_core(frame, geometry, offsets, splat_settings, **kwargs)

    monkeypatch.setattr(renderer, "_render_fast_compact_core", inspect_prepared)
    actual = renderer.render_relative_fast_compact(source, canonical, settings)
    assert observed
    np.testing.assert_array_equal(canonical, before)
    _assert_equal(actual, expected)


@pytest.mark.parametrize("fill", ["none", "background"])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.float32])
@pytest.mark.parametrize("depth_shape", [(7, 23), (3, 11), (13, 29)])
def test_fast_relative_preparation_preserves_pixels_and_diagnostics(
    device, fill, dtype, depth_shape
):
    rng = np.random.default_rng(20260901)
    source = rng.integers(0, 256, size=(7, 23, 3), dtype=np.uint8).astype(dtype)
    if dtype == np.float32:
        source += np.float32(0.375)
    canonical = rng.random(depth_shape, dtype=np.float32)
    source_before = source.copy()
    canonical_before = canonical.copy()
    settings = renderer_module.StereoRenderSettings(
        stereo_strength=4.731,
        convergence=0.3719,
        occlusion_fill=fill,
    )
    renderer = renderer_module.StereoRenderer(
        device=device,
        temporary_budget_bytes=23 * renderer_module.SPLAT_BYTES_PER_PIXEL * 2,
    )
    expected = _legacy_compact(renderer, source, canonical, settings)
    actual = renderer.render_relative_fast_compact(source, canonical, settings)
    _assert_equal(actual, expected)
    np.testing.assert_array_equal(source, source_before)
    np.testing.assert_array_equal(canonical, canonical_before)


@pytest.mark.parametrize("dtype", [np.bool_, np.float16, np.float32, np.float64])
@pytest.mark.parametrize("layout", ["readonly", "negative", "fortran"])
def test_fast_relative_preparation_keeps_canonical_conversion_and_layout(dtype, layout):
    source = np.arange(3 * 7 * 3, dtype=np.uint8).reshape(3, 7, 3)
    canonical = np.linspace(0.0, 1.0, 15).reshape(3, 5).astype(dtype)
    if layout == "readonly":
        source.setflags(write=False)
        canonical.setflags(write=False)
    elif layout == "negative":
        source = source[::-1, ::-1]
        canonical = canonical[::-1, ::-1]
    else:
        source = np.asfortranarray(source)
        canonical = np.asfortranarray(canonical)
    before_source, before_canonical = source.copy(), canonical.copy()
    settings = renderer_module.StereoRenderSettings()
    renderer = renderer_module.StereoRenderer(device="cpu")
    expected = _legacy_compact(renderer, source, canonical, settings)
    actual = renderer.render_relative_fast_compact(source, canonical, settings)
    _assert_equal(actual, expected)
    np.testing.assert_array_equal(source, before_source)
    np.testing.assert_array_equal(canonical, before_canonical)


def test_fast_relative_offsets_reuse_the_owned_near_score(monkeypatch):
    source = np.full((3, 13, 3), 128, dtype=np.uint8)
    canonical = np.linspace(0.0, 1.0, 39, dtype=np.float32).reshape(3, 13)
    renderer = renderer_module.StereoRenderer(device="cpu")
    settings = renderer_module.StereoRenderSettings()
    expected = _legacy_compact(renderer, source, canonical, settings)
    original_offsets = renderer_module.calculate_eye_sample_offsets
    original_core = renderer._render_fast_compact_core
    owned_near = []

    def observe_offsets(near, controls):
        owned_near.append(near.numpy())
        return original_offsets(near, controls)

    def inspect_shared_raster(frame, geometry, offsets, controls, **kwargs):
        assert len(owned_near) == 1
        assert np.shares_memory(owned_near[0], geometry.near_score)
        assert not np.shares_memory(owned_near[0], canonical)
        return original_core(frame, geometry, offsets, controls, **kwargs)

    monkeypatch.setattr(renderer_module, "calculate_eye_sample_offsets", observe_offsets)
    monkeypatch.setattr(renderer, "_render_fast_compact_core", inspect_shared_raster)
    _assert_equal(renderer.render_relative_fast_compact(source, canonical, settings), expected)


def test_fast_relative_compact_preserves_the_legacy_half_lane_boundary(monkeypatch):
    width = 8950
    source = np.full((1, width, 3), 123, dtype=np.uint8)
    canonical = np.full((1, width), np.float32(0.23064221441745758), dtype=np.float32)
    settings = renderer_module.StereoRenderSettings(
        stereo_strength=0.473053267491137,
        convergence=0.05202130106440961,
    )
    renderer = renderer_module.StereoRenderer(device="cpu")
    expected = _legacy_compact(renderer, source, canonical, settings)
    original_offsets = renderer_module.calculate_eye_sample_offsets
    offsets_seen = []

    def observe_offsets(near, controls):
        offsets = original_offsets(near, controls)
        offsets_seen.append(int(offsets[0][0, 0]))
        return offsets

    monkeypatch.setattr(renderer_module, "calculate_eye_sample_offsets", observe_offsets)
    actual = renderer.render_relative_fast_compact(source, canonical, settings)
    assert offsets_seen == [61]
    _assert_equal(actual, expected)


@pytest.mark.parametrize(
    "canonical,message",
    [
        (np.zeros((0, 2), dtype=np.float32), "non-empty 2D"),
        (np.zeros((2, 0), dtype=np.float32), "non-empty 2D"),
        (np.zeros((1, 2, 3), dtype=np.float32), "non-empty 2D"),
        (np.array([[np.nan]], dtype=np.float32), "must be finite"),
        (np.array([[np.inf]], dtype=np.float32), "must be finite"),
        (np.array([[-np.inf]], dtype=np.float32), "must be finite"),
        (np.array([[-0.01]], dtype=np.float32), "within \\[0, 1\\]"),
        (np.array([[1.01]], dtype=np.float32), "within \\[0, 1\\]"),
    ],
)
def test_fast_relative_rejects_bad_depth_before_preparation(monkeypatch, canonical, message):
    source = np.zeros((2, 3, 3), dtype=np.uint8)

    def reject_allocation(*_args, **_kwargs):
        raise AssertionError("Invalid input must fail before raster preparation")

    monkeypatch.setattr(renderer_module, "_resize_float32_bilinear", reject_allocation)
    with pytest.raises(ValueError, match=message):
        renderer_module.StereoRenderer(device="cpu").render_relative_fast_compact(
            source,
            canonical,
            renderer_module.StereoRenderSettings(),
        )


@pytest.mark.parametrize(
    "name,value",
    [
        ("stereo_strength", np.nan),
        ("stereo_strength", np.inf),
        ("stereo_strength", -np.inf),
        ("stereo_strength", -0.01),
        ("stereo_strength", np.nextafter(5.0, np.inf)),
        ("convergence", np.nan),
        ("convergence", np.inf),
        ("convergence", -np.inf),
        ("convergence", -0.01),
        ("convergence", np.nextafter(1.0, np.inf)),
    ],
)
def test_fast_relative_retains_control_revalidation(monkeypatch, name, value):
    source = np.full((3, 7, 3), 123, dtype=np.uint8)
    canonical = np.full((3, 7), 0.5, dtype=np.float32)
    settings = renderer_module.StereoRenderSettings()
    object.__setattr__(settings, name, value)
    renderer = renderer_module.StereoRenderer(device="cpu")
    message = f"{name} must be finite and within"
    with pytest.raises(ValueError, match=message):
        _legacy_compact(renderer, source, canonical, settings)

    def reject_allocation(*_args, **_kwargs):
        raise AssertionError("Invalid controls must fail before raster preparation")

    monkeypatch.setattr(renderer_module, "_resize_float32_bilinear", reject_allocation)
    with pytest.raises(ValueError, match=message):
        renderer.render_relative_fast_compact(source, canonical, settings)


@pytest.mark.parametrize(
    "source,error,message",
    [
        (np.zeros((2, 3), dtype=np.uint8), ValueError, "shape"),
        (np.zeros((0, 3, 3), dtype=np.uint8), ValueError, "dimensions"),
        (np.zeros((2, 0, 3), dtype=np.uint8), ValueError, "dimensions"),
        (np.zeros((2, 3, 3), dtype=np.bool_), TypeError, "Unsupported frame dtype"),
        (np.zeros((2, 3, 3), dtype=np.complex64), TypeError, "Unsupported frame dtype"),
        (np.full((2, 3, 3), np.nan, dtype=np.float32), ValueError, "Frame values must be finite"),
    ],
)
def test_fast_relative_rejects_bad_source_before_preparation(monkeypatch, source, error, message):
    canonical = np.full((2, 3), 0.5, dtype=np.float32)

    def reject_allocation(*_args, **_kwargs):
        raise AssertionError("Invalid input must fail before raster preparation")

    monkeypatch.setattr(renderer_module, "_resize_float32_bilinear", reject_allocation)
    with pytest.raises(error, match=message):
        renderer_module.StereoRenderer(device="cpu").render_relative_fast_compact(
            source,
            canonical,
            renderer_module.StereoRenderSettings(),
        )


@pytest.mark.parametrize("stage", ["preload", "band"])
def test_fast_relative_preparation_survives_cuda_oom_retry(monkeypatch, stage):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if stage == "band" and renderer_module._native_lane_analyzer() is None:
        pytest.skip("Native lane analysis is unavailable")
    rng = np.random.default_rng(20260901)
    source = rng.integers(0, 256, size=(7, 23, 3), dtype=np.uint8)
    canonical = rng.random((5, 13), dtype=np.float32)
    settings = renderer_module.StereoRenderSettings()
    renderer = renderer_module.StereoRenderer(
        device="cuda",
        temporary_budget_bytes=23 * renderer_module.TRITON_SPLAT_BYTES_PER_PIXEL * 2,
    )
    expected = _legacy_compact(renderer, source, canonical, settings)
    injected = []
    if stage == "preload":
        from src.depth_surge_3d.rendering import fast_offsets_triton

        # This case covers offset-upload OOM on the original host fallback.
        monkeypatch.setattr(fast_offsets_triton, "triton", None)
        original = renderer._transfer_offset_band

        def fail_once(values):
            if not injected:
                injected.append(True)
                raise torch.cuda.OutOfMemoryError("injected relative offset preload failure")
            return original(values)

        monkeypatch.setattr(renderer, "_transfer_offset_band", fail_once)
    else:
        original = renderer_module._download_fast_cuda_payload

        def fail_once(*args, **kwargs):
            if not injected:
                injected.append(True)
                raise torch.cuda.OutOfMemoryError("injected relative band failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(renderer_module, "_download_fast_cuda_payload", fail_once)
    actual = renderer.render_relative_fast_compact(source, canonical, settings)
    assert injected == [True]
    _assert_equal(actual, expected)
