"""Finalize Fast coverage flags where the packed lane bitmap already exists."""

import numpy as np
import pytest
import torch

from src.depth_surge_3d.rendering import fast_output_triton as output_module
from src.depth_surge_3d.rendering import stereo_renderer as renderer_module


@pytest.fixture
def cuda_output(monkeypatch):
    if not torch.cuda.is_available() or output_module.triton is None:
        pytest.skip("Triton CUDA is unavailable")
    monkeypatch.setattr(output_module, "_DISABLED_CUDA_DEVICES", set())
    return output_module


def _assert_render_equal(actual, expected):
    for name in (
        "left_image",
        "right_image",
        "left_coverage_count",
        "right_coverage_count",
        "left_repair_bits",
        "right_repair_bits",
    ):
        np.testing.assert_array_equal(np.asarray(getattr(actual, name)), getattr(expected, name))
    for name in ("left_fast_diagnostics", "right_fast_diagnostics"):
        result = getattr(actual, name)
        wanted = getattr(expected, name)
        assert result.local_filled_lane_count == wanted.local_filled_lane_count
        assert result.final_unresolved_lane_count == wanted.final_unresolved_lane_count
        for histogram in ("lane_count_histogram", "touched_pixel_span_histogram"):
            np.testing.assert_array_equal(
                getattr(result.hole_runs, histogram), getattr(wanted.hole_runs, histogram)
            )


def _frame(fill):
    rng = np.random.default_rng(20260831)
    frame = rng.integers(0, 256, size=(7, 23, 3), dtype=np.uint8)
    canonical = rng.random((5, 13), dtype=np.float32)
    settings = renderer_module.StereoRenderSettings(
        stereo_strength=5.0, convergence=0.2, occlusion_fill=fill
    )
    return frame, canonical, settings


def _gpu_renderer():
    return renderer_module.StereoRenderer(
        device="cuda", temporary_budget_bytes=23 * renderer_module.TRITON_SPLAT_BYTES_PER_PIXEL * 2
    )


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("pattern", ["none", "filled", "unresolved", "mixed"])
def test_complete_gpu_repair_matches_every_lane_bitmap(cuda_output, background, pattern):
    bitmaps = np.arange(65536, dtype=np.uint16)
    bits = (bitmaps[:, None] & (np.uint16(1) << np.arange(16, dtype=np.uint16))) != 0
    filled = np.zeros_like(bits)
    unresolved = np.zeros_like(bits)
    if pattern == "filled":
        filled = ~bits
    elif pattern == "unresolved":
        unresolved = ~bits
    elif pattern == "mixed":
        filled[:, ::2] = ~bits[:, ::2]
        unresolved[:, 1::2] = ~bits[:, 1::2]
    masks = [
        torch.from_numpy(value.reshape(1, -1, 16)).cuda() for value in (bits, filled, unresolved)
    ]
    colour = torch.full((1, bitmaps.size * 16, 3), 127.5, device="cuda")

    payload = cuda_output.triton_fast_band_payload(
        colour, *masks, background=background, include_coverage_flags=True
    )

    assert payload is not None and payload.numel() == 24 + 6 * bitmaps.size
    image, missing, repair, counts = renderer_module._download_fast_cuda_payload(
        payload, (1, bitmaps.size, 3), (1, bitmaps.size)
    )
    coverage = bits.sum(axis=1)
    expected = np.where(coverage == 0, 2, np.where(coverage < 16, 1, 0)).astype(np.uint8)
    expected |= filled.any(axis=1).astype(np.uint8) * np.uint8(0x04)
    expected |= unresolved.any(axis=1).astype(np.uint8) * np.uint8(0x48 if background else 0x40)
    np.testing.assert_array_equal(repair[0], expected)
    np.testing.assert_array_equal(missing[0], bitmaps ^ np.uint16(65535))
    assert np.all(image == 128)
    assert counts.tolist() == [int(filled.sum()), int(unresolved.sum()), 0]
    for device, original in zip(masks, (bits, filled, unresolved)):
        np.testing.assert_array_equal(device.cpu().numpy()[0], original)


def test_coverage_opt_in_retains_legacy_packet_default_and_other_fields(cuda_output):
    colour = torch.full((1, 48, 3), 9.0, device="cuda")
    valid = torch.tensor([[[True] * 16, [True] * 8 + [False] * 8, [False] * 16]], device="cuda")
    filled = torch.zeros_like(valid)
    filled[0, 1, 8:] = True
    holes = torch.zeros_like(valid)
    holes[0, 2] = True
    legacy = cuda_output.triton_fast_band_payload(colour, valid, filled, holes, background=True)
    complete = cuda_output.triton_fast_band_payload(
        colour, valid, filled, holes, background=True, include_coverage_flags=True
    )
    assert legacy is not None and complete is not None
    assert torch.equal(legacy[:-3], complete[:-3])
    assert legacy[-3:].cpu().tolist() == [0, 4, 0x48]
    assert complete[-3:].cpu().tolist() == [0, 5, 0x4A]
    assert complete[:24].view(torch.int64).cpu().tolist() == [8, 16, 0]


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), np.finfo(np.float32).max]
)
def test_complete_coverage_keeps_nonfinite_counts_and_source_unchanged(cuda_output, value):
    colour = torch.full((1, 129 * 16, 3), 2.5, device="cuda")
    colour[0, :16, 0] = float(value)
    colour[0, -16:, 2] = float(value)
    before = colour.clone()
    valid = torch.ones((1, 129, 16), dtype=torch.bool, device="cuda")
    valid[0, 0, 1:] = False
    valid[0, -1] = False
    empty = torch.zeros_like(valid)
    payload = cuda_output.triton_fast_band_payload(
        colour, valid, empty, empty, background=False, include_coverage_flags=True
    )
    assert payload is not None
    _, _, repair, counts = renderer_module._download_fast_cuda_payload(
        payload, (1, 129, 3), (1, 129)
    )
    assert counts.tolist() == [0, 0, 2]
    np.testing.assert_array_equal(repair[0], [1] + [0] * 127 + [2])
    torch.testing.assert_close(colour, before, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("fill", ["none", "background"])
def test_fused_renderer_does_not_rescan_cpu_coverage(cuda_output, monkeypatch, fill):
    if renderer_module._native_lane_analyzer() is None:
        pytest.skip("Native lane analysis is unavailable")
    frame, canonical, settings = _frame(fill)
    expected = renderer_module.StereoRenderer(device="cpu").render_relative_fast_compact(
        frame, canonical, settings
    )
    original_empty = np.empty

    class CoverageWithoutRescanning(np.ndarray):
        def __gt__(self, _other):
            raise AssertionError("Fused output must not rescan CPU coverage")

        __lt__ = __eq__ = __gt__

    def guarded_empty(shape, *args, **kwargs):
        result = original_empty(shape, *args, **kwargs)
        if result.shape == frame.shape[:2] and result.dtype == np.uint8:
            return result.view(CoverageWithoutRescanning)
        return result

    monkeypatch.setattr(np, "empty", guarded_empty)
    actual = _gpu_renderer().render_relative_fast_compact(frame, canonical, settings)
    _assert_render_equal(actual, expected)
    assert not cuda_output._DISABLED_CUDA_DEVICES
    assert np.any(expected.left_repair_bits & np.uint8(3))


@pytest.mark.parametrize("fill", ["none", "background"])
@pytest.mark.parametrize("failure", ["first", "middle", "compile"])
def test_cpu_coverage_repair_is_retained_for_each_unfused_band(
    cuda_output, monkeypatch, fill, failure
):
    if renderer_module._native_lane_analyzer() is None:
        pytest.skip("Native lane analysis is unavailable")
    frame, canonical, settings = _frame(fill)
    expected = renderer_module.StereoRenderer(device="cpu").render_relative_fast_compact(
        frame, canonical, settings
    )
    original = cuda_output.triton_fast_band_payload
    calls = []

    def sometimes_unavailable(*args, **kwargs):
        calls.append(args[0].shape[0])
        if len(calls) == (1 if failure == "first" else 2):
            return None
        return original(*args, **kwargs)

    if failure == "compile":

        class UnavailableKernel:
            def __getitem__(self, _grid):
                def fail(*_args, **_kwargs):
                    raise cuda_output.triton.errors.TritonError(
                        "injected output compilation failure"
                    )

                return fail

        monkeypatch.setattr(cuda_output, "_fast_band_payload_kernel", UnavailableKernel())
    else:
        monkeypatch.setattr(cuda_output, "triton_fast_band_payload", sometimes_unavailable)
    actual = _gpu_renderer().render_relative_fast_compact(frame, canonical, settings)
    _assert_render_equal(actual, expected)
    if failure != "compile":
        assert calls == [2, 2, 2, 1] * 2


@pytest.mark.parametrize("kernel_name", ["_fast_band_payload_kernel", "_write_lane_counts_kernel"])
def test_output_driver_oom_retries_smaller_bands_with_exact_repair(
    cuda_output, monkeypatch, kernel_name
):
    if renderer_module._native_lane_analyzer() is None:
        pytest.skip("Native lane analysis is unavailable")
    frame, canonical, settings = _frame("background")
    expected = renderer_module.StereoRenderer(device="cpu").render_relative_fast_compact(
        frame, canonical, settings
    )
    original_kernel = getattr(cuda_output, kernel_name)
    original_output = cuda_output.triton_fast_band_payload
    heights = []

    def observe_output(colour, *args, **kwargs):
        heights.append(colour.shape[0])
        return original_output(colour, *args, **kwargs)

    class FailSecondBand:
        def __getitem__(self, grid):
            def call(*args, **kwargs):
                if len(heights) == 2:
                    raise RuntimeError("Triton Error [CUDA]: out of memory")
                return original_kernel[grid](*args, **kwargs)

            return call

    monkeypatch.setattr(cuda_output, kernel_name, FailSecondBand())
    monkeypatch.setattr(cuda_output, "triton_fast_band_payload", observe_output)
    actual = _gpu_renderer().render_relative_fast_compact(frame, canonical, settings)
    assert heights == [2, 2] + [1] * 14
    assert not cuda_output._DISABLED_CUDA_DEVICES
    _assert_render_equal(actual, expected)


@pytest.mark.parametrize("kernel_name", ["_fast_band_payload_kernel", "_write_lane_counts_kernel"])
def test_output_driver_oom_retains_its_original_cause(cuda_output, monkeypatch, kernel_name):
    colour = torch.zeros((1, 16, 3), device="cuda")
    valid = torch.ones((1, 1, 16), dtype=torch.bool, device="cuda")
    empty = torch.zeros_like(valid)
    failure = RuntimeError("Triton Error [CUDA]: out of memory")

    class FailingKernel:
        def __getitem__(self, _grid):
            def call(*_args, **_kwargs):
                raise failure

            return call

    monkeypatch.setattr(cuda_output, kernel_name, FailingKernel())
    with pytest.raises(torch.cuda.OutOfMemoryError) as captured:
        cuda_output.triton_fast_band_payload(
            colour, valid, empty, empty, background=False, include_coverage_flags=True
        )
    assert captured.value.__cause__ is failure
    assert not cuda_output._DISABLED_CUDA_DEVICES


@pytest.mark.parametrize("kernel_name", ["_fast_band_payload_kernel", "_write_lane_counts_kernel"])
@pytest.mark.parametrize(
    "message",
    [
        "Triton Error [CUDA]: an illegal memory access was encountered",
        "Triton Error [CUDA]: invalid argument",
        "out of memory",
    ],
)
def test_output_unrelated_driver_errors_remain_visible(
    cuda_output, monkeypatch, kernel_name, message
):
    colour = torch.zeros((1, 16, 3), device="cuda")
    valid = torch.ones((1, 1, 16), dtype=torch.bool, device="cuda")
    empty = torch.zeros_like(valid)
    failure = RuntimeError(message)

    class FailingKernel:
        def __getitem__(self, _grid):
            def call(*_args, **_kwargs):
                raise failure

            return call

    monkeypatch.setattr(cuda_output, kernel_name, FailingKernel())
    with pytest.raises(RuntimeError) as captured:
        cuda_output.triton_fast_band_payload(
            colour, valid, empty, empty, background=False, include_coverage_flags=True
        )
    assert captured.value is failure
    assert not cuda_output._DISABLED_CUDA_DEVICES
