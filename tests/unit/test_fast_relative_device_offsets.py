"""Relative Fast CUDA offsets stay on device without changing rendered output."""

import gc
import threading
import weakref

import numpy as np
import pytest
import torch

from src.depth_surge_3d.rendering import stereo_renderer as renderer_module
from src.depth_surge_3d.rendering import fast_offsets_triton as offsets_module


requires_triton_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or offsets_module.triton is None,
    reason="Triton CUDA is unavailable",
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
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    for name in ("left_fast_diagnostics", "right_fast_diagnostics"):
        actual_eye, expected_eye = getattr(actual, name), getattr(expected, name)
        assert actual_eye.local_filled_lane_count == expected_eye.local_filled_lane_count
        assert actual_eye.final_unresolved_lane_count == expected_eye.final_unresolved_lane_count
        for histogram in ("lane_count_histogram", "touched_pixel_span_histogram"):
            np.testing.assert_array_equal(
                getattr(actual_eye.hole_runs, histogram),
                getattr(expected_eye.hole_runs, histogram),
            )


def _fixture(fill="background"):
    rng = np.random.default_rng(20260831)
    source = rng.integers(0, 256, size=(7, 31, 3), dtype=np.uint8)
    canonical = rng.random((5, 17), dtype=np.float32)
    settings = renderer_module.StereoRenderSettings(
        stereo_strength=4.731, convergence=0.3719, occlusion_fill=fill
    )
    return source, canonical, settings


@requires_triton_cuda
@pytest.mark.parametrize("fill", ["none", "background"])
def test_relative_fast_cuda_never_materializes_or_transfers_host_offsets(monkeypatch, fill):
    source, canonical, settings = _fixture(fill)
    before_source, before_canonical = source.copy(), canonical.copy()
    expected = renderer_module.StereoRenderer(device="cpu").render_relative_fast_compact(
        source, canonical, settings
    )
    renderer = renderer_module.StereoRenderer(
        device="cuda",
        profile=True,
        temporary_budget_bytes=31 * renderer_module.TRITON_SPLAT_BYTES_PER_PIXEL * 2,
    )

    def reject_host_offsets(*_args, **_kwargs):
        raise AssertionError("Supported relative CUDA rendering must keep offsets on device")

    original_cpu = torch.Tensor.cpu

    def reject_offset_readback(tensor, *args, **kwargs):
        if tensor.dtype == torch.int32 and tensor.shape == source.shape[:2]:
            raise AssertionError("Resident CUDA offsets must not round-trip through the host")
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(renderer_module, "calculate_eye_sample_offsets", reject_host_offsets)
    monkeypatch.setattr(renderer, "_transfer_offset_band", reject_host_offsets)
    monkeypatch.setattr(torch.Tensor, "cpu", reject_offset_readback)
    actual = renderer.render_relative_fast_compact(source, canonical, settings)

    _assert_equal(actual, expected)
    np.testing.assert_array_equal(source, before_source)
    np.testing.assert_array_equal(canonical, before_canonical)
    assert renderer._offset_transfer_bytes == 0
    assert 0 < actual.performance_seconds["fast_offset_build"]
    assert actual.performance_seconds["fast_offset_build"] < actual.performance_seconds["total"]


@requires_triton_cuda
@pytest.mark.parametrize("reason", ["no_triton", "disabled", "size"])
def test_relative_fast_cuda_falls_back_to_one_host_offset_pair(monkeypatch, reason):
    source, canonical, settings = _fixture()
    expected = renderer_module.StereoRenderer(device="cpu").render_relative_fast_compact(
        source, canonical, settings
    )
    renderer = renderer_module.StereoRenderer(device="cuda", profile=True)
    if reason == "no_triton":
        monkeypatch.setattr(offsets_module, "triton", None)
    elif reason == "disabled":
        monkeypatch.setattr(offsets_module, "_DISABLED_CUDA_DEVICES", {torch.cuda.current_device()})
    else:
        monkeypatch.setattr(
            offsets_module, "_MAX_OFFSET_PIXELS", source.shape[0] * source.shape[1] - 1
        )
    original_offsets = renderer_module.calculate_eye_sample_offsets
    host_calls = []

    def observe_host_offsets(near, controls):
        host_calls.append(near.device.type)
        return original_offsets(near, controls)

    monkeypatch.setattr(renderer_module, "calculate_eye_sample_offsets", observe_host_offsets)
    actual = renderer.render_relative_fast_compact(source, canonical, settings)
    _assert_equal(actual, expected)
    assert host_calls == ["cpu"]
    assert renderer._offset_transfer_bytes == source.shape[0] * source.shape[1] * 8


def _observe_resident_allocations(monkeypatch, source, stage, injected, resident_refs):
    original_to, original_empty = torch.Tensor.to, torch.empty
    offset_allocations = 0

    def observe_transfer(tensor, *args, **kwargs):
        result = original_to(tensor, *args, **kwargs)
        if result.device.type == "cuda" and tuple(result.shape[:2]) == source.shape[:2]:
            resident_refs.append(weakref.ref(result))
        return result

    def observe_allocation(*args, **kwargs):
        nonlocal offset_allocations
        is_offset = args and args[0] == source.shape[:2] and kwargs.get("dtype") == torch.int32
        if is_offset:
            offset_allocations += 1
            if stage == "allocation" and offset_allocations == 2:
                injected.append(stage)
                raise torch.cuda.OutOfMemoryError("injected second offset allocation failure")
        result = original_empty(*args, **kwargs)
        if is_offset:
            resident_refs.append(weakref.ref(result))
        return result

    monkeypatch.setattr(torch.Tensor, "to", observe_transfer)
    monkeypatch.setattr(torch, "empty", observe_allocation)


@requires_triton_cuda
@pytest.mark.parametrize("stage", ["allocation", "kernel", "first_band", "later_band"])
def test_relative_fast_oom_releases_resident_planes_before_host_fallback(monkeypatch, stage):
    source, canonical, settings = _fixture()
    expected = renderer_module.StereoRenderer(device="cpu").render_relative_fast_compact(
        source, canonical, settings
    )
    renderer = renderer_module.StereoRenderer(
        device="cuda",
        temporary_budget_bytes=31 * renderer_module.TRITON_SPLAT_BYTES_PER_PIXEL * 2,
    )
    original_release = renderer._release_after_oom
    original_host = renderer_module.calculate_eye_sample_offsets
    original_splat = renderer_module._forward_splat_band_prevalidated
    original_kernel = offsets_module._relative_offsets_kernel
    resident_refs, injected, host_calls, releases = [], [], [], []
    splat_calls = 0

    _observe_resident_allocations(monkeypatch, source, stage, injected, resident_refs)

    class ObservedKernel:
        def __getitem__(self, grid):
            launch = original_kernel[grid]

            def observed(*args, **kwargs):
                if stage == "kernel" and not injected:
                    injected.append(stage)
                    raise torch.cuda.OutOfMemoryError("injected offset kernel failure")
                return launch(*args, **kwargs)

            return observed

    def observe_splat(*args, **kwargs):
        nonlocal splat_calls
        splat_calls += 1
        fail_at = 1 if stage == "first_band" else 3
        if stage.endswith("band") and splat_calls == fail_at:
            injected.append(stage)
            raise torch.cuda.OutOfMemoryError("injected resident-offset band failure")
        return original_splat(*args, **kwargs)

    def assert_released():
        gc.collect()
        assert resident_refs, "The failed attempt must actually allocate resident inputs"
        assert all(ref() is None for ref in resident_refs), "OOM traceback retains CUDA planes"
        releases.append(True)
        original_release()

    def observe_host(near, controls):
        host_calls.append(near.device.type)
        return original_host(near, controls)

    monkeypatch.setattr(offsets_module, "_relative_offsets_kernel", ObservedKernel())
    monkeypatch.setattr(renderer_module, "_forward_splat_band_prevalidated", observe_splat)
    monkeypatch.setattr(renderer_module, "calculate_eye_sample_offsets", observe_host)
    monkeypatch.setattr(renderer, "_release_after_oom", assert_released)
    actual = renderer.render_relative_fast_compact(source, canonical, settings)

    assert injected == [stage] and releases == [True]
    assert host_calls == ["cpu"]
    assert not any(thread.name.startswith("fast-lane-analysis") for thread in threading.enumerate())
    _assert_equal(actual, expected)


@requires_triton_cuda
def test_relative_fast_repeated_oom_keeps_the_public_error_and_allows_a_new_frame(monkeypatch):
    source, canonical, settings = _fixture()
    renderer = renderer_module.StereoRenderer(
        device="cuda",
        temporary_budget_bytes=31 * renderer_module.TRITON_SPLAT_BYTES_PER_PIXEL * 2,
    )

    def fail_band(*_args, **_kwargs):
        raise torch.cuda.OutOfMemoryError("injected repeated band failure")

    with monkeypatch.context() as patch:
        patch.setattr(renderer_module, "_forward_splat_band_prevalidated", fail_band)
        with pytest.raises(RuntimeError, match="attempted band heights 2 and 1") as caught:
            renderer.render_relative_fast_compact(source, canonical, settings)
        assert isinstance(caught.value.__cause__, torch.cuda.OutOfMemoryError)
    assert not any(thread.name.startswith("fast-lane-analysis") for thread in threading.enumerate())
    expected = renderer_module.StereoRenderer(device="cpu").render_relative_fast_compact(
        source, canonical, settings
    )
    _assert_equal(renderer.render_relative_fast_compact(source, canonical, settings), expected)


@requires_triton_cuda
def test_relative_fast_resident_offsets_follow_the_active_nondefault_stream():
    source, canonical, settings = _fixture()
    expected = renderer_module.StereoRenderer(device="cpu").render_relative_fast_compact(
        source, canonical, settings
    )
    stream = torch.cuda.Stream()
    renderer = renderer_module.StereoRenderer(device="cuda")
    with torch.cuda.stream(stream):
        actual = renderer.render_relative_fast_compact(source, canonical, settings)
    stream.synchronize()
    _assert_equal(actual, expected)
