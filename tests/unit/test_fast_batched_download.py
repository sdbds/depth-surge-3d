"""Fast CUDA bands retain identical pixels and diagnostics with one download."""

from __future__ import annotations

from concurrent.futures import Future

import numpy as np
import pytest
import torch

from src.depth_surge_3d.rendering import stereo_renderer as renderer_module
from src.depth_surge_3d.rendering import fast_output_triton as output_module
from src.depth_surge_3d.rendering.quality.diagnostic_transaction import canonical_json_bytes
from src.depth_surge_3d.rendering.quality.diagnostics import build_quality_frame_stats


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")


@pytest.fixture(params=["torch", "fused"])
def output_backend(request, monkeypatch):
    if request.param == "torch":
        monkeypatch.setattr(output_module, "triton_output_available", lambda _device: False)
    elif output_module.triton is None or renderer_module._native_lane_analyzer() is None:
        pytest.skip("Fused output dependencies are unavailable")


def _inputs(dtype=np.uint8):
    rng = np.random.default_rng(20260831)
    frame = rng.integers(0, 256, size=(7, 31, 3), dtype=np.uint8).astype(dtype)
    if dtype == np.uint16:
        frame *= np.uint16(257)
    elif dtype == np.float32:
        frame = (frame - np.float32(96.0)) * np.float32(0.25)
    canonical = rng.random((5, 17), dtype=np.float32)
    return frame, canonical


def _renderer(device):
    return renderer_module.StereoRenderer(
        device=device,
        temporary_budget_bytes=31 * renderer_module.TRITON_SPLAT_BYTES_PER_PIXEL * 2,
    )


@pytest.mark.parametrize("fill", ["none", "background"])
def test_fast_cuda_downloads_one_same_size_payload_per_band(monkeypatch, fill):
    if renderer_module._native_lane_analyzer() is None:
        pytest.skip("Native lane analysis is unavailable")
    frame, canonical = _inputs()
    downloads = []
    original_cpu = torch.Tensor.cpu

    def observe_cpu(tensor, *args, **kwargs):
        if tensor.is_cuda:
            downloads.append(tensor.numel() * tensor.element_size())
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", observe_cpu)
    renderer = _renderer("cuda")
    settings = renderer_module.StereoRenderSettings(
        stereo_strength=4.25,
        convergence=0.37,
        occlusion_fill=fill,
    )
    actual = renderer.render_relative_fast_compact(frame, canonical, settings)

    bytes_per_pixel = (
        renderer_module.TRITON_SPLAT_BYTES_PER_PIXEL
        if renderer_module._triton_splat_available(renderer.device)
        else renderer_module.SPLAT_BYTES_PER_PIXEL
    )
    band_height = renderer_module.calculate_band_height(
        31, 7, renderer.temporary_budget_bytes, bytes_per_pixel=bytes_per_pixel
    )
    band_pixels = [min(band_height, 7 - start) * 31 for start in range(0, 7, band_height)]
    assert downloads == [6 * pixels + 24 for pixels in band_pixels] * 2
    assert actual.left_image.shape == frame.shape
    assert actual.right_image.shape == frame.shape


@pytest.mark.parametrize("fill", ["none", "background"])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_fast_cuda_compact_pixels_and_diagnostics_match_cpu(fill, dtype):
    frame, canonical = _inputs(dtype)
    original_frame = frame.copy()
    original_canonical = canonical.copy()
    settings = renderer_module.StereoRenderSettings(
        stereo_strength=4.25,
        convergence=0.37,
        occlusion_fill=fill,
    )
    expected = _renderer("cpu").render_relative_fast_compact(frame, canonical, settings)
    actual = _renderer("cuda").render_relative_fast_compact(frame, canonical, settings)

    for name in (
        "left_image",
        "right_image",
        "left_coverage_count",
        "right_coverage_count",
        "left_repair_bits",
        "right_repair_bits",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    identity = dict(
        frame_name="frame_000001",
        geometry_mode="relative",
        occlusion_fill=fill,
        render_mode="fast",
    )
    assert canonical_json_bytes(build_quality_frame_stats(actual, **identity)) == (
        canonical_json_bytes(build_quality_frame_stats(expected, **identity))
    )
    np.testing.assert_array_equal(frame, original_frame)
    np.testing.assert_array_equal(canonical, original_canonical)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_fast_cuda_download_retains_nonfinite_rejection(monkeypatch, invalid, output_backend):
    frame, canonical = _inputs()
    original_fill = renderer_module._fill_background_band_state

    def invalid_fill(*args, **kwargs):
        result, holes, filled = original_fill(*args, **kwargs)
        result[-1, -1, -1] = invalid
        return result, holes, filled

    monkeypatch.setattr(renderer_module, "_fill_background_band_state", invalid_fill)
    with pytest.raises(ValueError, match="Rendered image contains non-finite values"):
        _renderer("cuda").render_relative_fast_compact(
            frame, canonical, renderer_module.StereoRenderSettings()
        )


@pytest.mark.parametrize("width", [1, 3, 257])
def test_fast_cuda_packet_preserves_rounding_masks_and_int64_counts(width):
    pixels = 3 * width
    colour = np.resize(
        np.array([-3.0, -0.5, 0.5, 1.5, 2.5, 3.5, 254.5, 255.5, 300.0], dtype=np.float32),
        (3, width, 3),
    )
    masks = np.arange(pixels, dtype=np.uint16).reshape(3, width)
    masks[0, 0] = np.uint16(0xFFFF)
    masks[-1, -1] = np.uint16(0xAAAA)
    lanes = (masks[..., None] & (1 << np.arange(16, dtype=np.uint16))) != 0
    repairs = (np.arange(pixels) % 128).astype(np.uint8).reshape(3, width)
    counts = np.array([2**40 + 17, 2**60 + 3, 0], dtype=np.int64)

    image, missing, repair, actual_counts = renderer_module._download_fast_cuda_band(
        torch.from_numpy(colour).to("cuda"),
        torch.from_numpy(lanes).to("cuda"),
        renderer_module._lane_mask_weights(torch.device("cuda")),
        torch.from_numpy(repairs).to("cuda"),
        torch.from_numpy(counts).to("cuda"),
    )

    np.testing.assert_array_equal(image, np.clip(np.rint(colour), 0, 255).astype(np.uint8))
    np.testing.assert_array_equal(missing, masks ^ np.uint16(0xFFFF))
    np.testing.assert_array_equal(repair, repairs)
    np.testing.assert_array_equal(actual_counts, counts)
    for array in (image, missing, repair, actual_counts):
        assert array.flags.c_contiguous and array.flags.aligned


def test_fast_cuda_without_native_keeps_fallback(monkeypatch):
    frame, canonical = _inputs()
    settings = renderer_module.StereoRenderSettings()
    monkeypatch.setattr(renderer_module, "_native_lane_analyzer", lambda: None)

    def unexpected_batch(*_args, **_kwargs):
        raise AssertionError("Native-free Fast rendering cannot use the packed analyzer")

    monkeypatch.setattr(renderer_module, "_download_fast_cuda_band", unexpected_batch)
    expected = _renderer("cpu").render_relative_fast_compact(frame, canonical, settings)
    actual = _renderer("cuda").render_relative_fast_compact(frame, canonical, settings)
    for name in ("left_image", "right_image", "left_coverage_count", "right_repair_bits"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def test_fast_cuda_batch_oom_retries_without_changing_output(monkeypatch, output_backend):
    if renderer_module._native_lane_analyzer() is None:
        pytest.skip("Native lane analysis is unavailable")
    frame, canonical = _inputs()
    settings = renderer_module.StereoRenderSettings()
    expected = _renderer("cpu").render_relative_fast_compact(frame, canonical, settings)
    original_download = renderer_module._download_fast_cuda_payload
    row_counts = []

    def fail_first(payload, image_shape, *args, **kwargs):
        row_counts.append(image_shape[0])
        if len(row_counts) == 1:
            raise torch.cuda.OutOfMemoryError("injected output-packet allocation failure")
        return original_download(payload, image_shape, *args, **kwargs)

    monkeypatch.setattr(renderer_module, "_download_fast_cuda_payload", fail_first)
    actual = _renderer("cuda").render_relative_fast_compact(frame, canonical, settings)
    assert len(row_counts) == 15
    assert all(rows == 1 for rows in row_counts[1:])
    for name in (
        "left_image",
        "right_image",
        "left_coverage_count",
        "right_coverage_count",
        "left_repair_bits",
        "right_repair_bits",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def test_fast_cuda_async_analysis_retains_downloaded_mask_ownership(monkeypatch):
    if renderer_module._native_lane_analyzer() is None:
        pytest.skip("Native lane analysis is unavailable")
    frame, canonical = _inputs()
    settings = renderer_module.StereoRenderSettings(stereo_strength=5.0)
    expected = _renderer("cpu").render_relative_fast_compact(frame, canonical, settings)
    original_queue = renderer_module._FastLaneAnalysisQueue
    monkeypatch.setattr(
        renderer_module,
        "_FastLaneAnalysisQueue",
        lambda native: original_queue(native, async_threshold_seconds=0.0),
    )
    actual = _renderer("cuda").render_relative_fast_compact(frame, canonical, settings)
    identity = dict(
        frame_name="frame_000001",
        geometry_mode="relative",
        occlusion_fill="background",
        render_mode="fast",
    )
    assert canonical_json_bytes(build_quality_frame_stats(actual, **identity)) == (
        canonical_json_bytes(build_quality_frame_stats(expected, **identity))
    )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_fast_cuda_native_failure_precedes_nonfinite_rejection(
    monkeypatch, invalid, output_backend
):
    frame, canonical = _inputs()
    original_fill = renderer_module._fill_background_band_state

    class FailingNative:
        def accumulate_packed_lane_hole_run_histograms_into(self, *_args):
            raise RuntimeError("injected native histogram failure")

    def invalid_fill(*args, **kwargs):
        result, holes, filled = original_fill(*args, **kwargs)
        result[-1, -1, -1] = invalid
        return result, holes, filled

    native = FailingNative()
    monkeypatch.setattr(renderer_module, "_native_lane_analyzer", lambda: native)
    monkeypatch.setattr(renderer_module, "_fill_background_band_state", invalid_fill)
    with pytest.raises(RuntimeError, match="injected native histogram failure"):
        _renderer("cuda").render_relative_fast_compact(
            frame, canonical, renderer_module.StereoRenderSettings()
        )


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_fast_cuda_finishes_queued_histograms_before_freezing_them(monkeypatch, dtype):
    if renderer_module._native_lane_analyzer() is None:
        pytest.skip("Native lane analysis is unavailable")
    frame, canonical = _inputs(dtype)
    settings = renderer_module.StereoRenderSettings(stereo_strength=5.0)
    expected = _renderer("cpu").render_relative_fast_compact(frame, canonical, settings)

    class DeferredFuture(Future):
        def __init__(self, function, args):
            super().__init__()
            self.function = function
            self.args = args

        def result(self, timeout=None):
            if not self.done():
                try:
                    self.set_result(self.function(*self.args))
                except BaseException as error:
                    self.set_exception(error)
            return super().result(timeout=timeout)

    class DeferredExecutor:
        def __init__(self, **_kwargs):
            pass

        def submit(self, function, *args):
            return DeferredFuture(function, args)

        def shutdown(self, **_kwargs):
            pass

    original_queue = renderer_module._FastLaneAnalysisQueue
    monkeypatch.setattr(renderer_module, "ThreadPoolExecutor", DeferredExecutor)
    monkeypatch.setattr(
        renderer_module,
        "_FastLaneAnalysisQueue",
        lambda native: original_queue(native, async_threshold_seconds=0.0),
    )
    actual = _renderer("cuda").render_relative_fast_compact(frame, canonical, settings)
    identity = dict(
        frame_name="frame_000001",
        geometry_mode="relative",
        occlusion_fill="background",
        render_mode="fast",
    )
    assert canonical_json_bytes(build_quality_frame_stats(actual, **identity)) == (
        canonical_json_bytes(build_quality_frame_stats(expected, **identity))
    )
