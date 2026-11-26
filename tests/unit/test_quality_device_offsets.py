"""Quality projection keeps exact single-eye offsets on CUDA when supported."""

import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.depth_surge_3d.rendering import fast_offsets_triton as offsets_module
from src.depth_surge_3d.rendering.quality import renderer as quality
from src.depth_surge_3d.rendering.stereo_renderer import StereoRenderer, StereoRenderSettings


requires_triton_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or offsets_module.triton is None,
    reason="Triton CUDA is unavailable",
)


def _fixture(fill="background"):
    rng = np.random.default_rng(20260906)
    source = rng.integers(0, 256, (9, 48, 3), dtype=np.uint8)
    canonical = rng.random((5, 24), dtype=np.float32)
    settings = StereoRenderSettings(
        stereo_strength=4.731,
        convergence=0.3719,
        stereo_render_mode="quality",
        occlusion_fill=fill,
    )
    return source, canonical, settings


def _render(renderer, fixture):
    source, canonical, settings = fixture
    return quality.render_relative_quality_compact(
        renderer=renderer, frame=source, canonical=canonical, settings=settings
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
    for name in ("left_plan", "right_plan", "geometry_nearest", "quality_limits"):
        assert getattr(actual, name) == getattr(expected, name)
    for name in ("left_hole_runs", "right_hole_runs"):
        for histogram in ("lane_count_histogram", "touched_pixel_span_histogram"):
            np.testing.assert_array_equal(
                getattr(getattr(actual, name), histogram),
                getattr(getattr(expected, name), histogram),
            )


@requires_triton_cuda
@pytest.mark.parametrize("fill", ["none", "background"])
def test_quality_cuda_avoids_host_offset_construction_and_upload(monkeypatch, fill):
    fixture = _fixture(fill)
    before = fixture[0].tobytes(), fixture[1].tobytes()
    expected = _render(StereoRenderer(device="cpu"), fixture)
    renderer = StereoRenderer(
        device="cuda", temporary_budget_bytes=48 * quality.QUALITY_SPLAT_BYTES_PER_PIXEL * 2
    )

    def reject_host_offsets(*_args, **_kwargs):
        raise AssertionError("Quality CUDA offsets must stay on device")

    original_cpu = torch.Tensor.cpu

    def reject_offset_readback(tensor, *args, **kwargs):
        if tensor.dtype == torch.int32 and tuple(tensor.shape) == fixture[0].shape[:2]:
            raise AssertionError("Quality CUDA offsets must not round-trip through the host")
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(quality, "build_relative_eye_offsets_into", reject_host_offsets)
    monkeypatch.setattr(renderer, "_transfer_offset_band", reject_host_offsets)
    monkeypatch.setattr(torch.Tensor, "cpu", reject_offset_readback)
    actual = _render(renderer, fixture)

    _assert_equal(actual, expected)
    assert before == (fixture[0].tobytes(), fixture[1].tobytes())


@requires_triton_cuda
@pytest.mark.parametrize(
    "eye, expected", [("left", [0, 0, 0, 1, 1]), ("right", [0, 0, -1, -1, -1])]
)
def test_single_eye_cuda_offsets_preserve_asymmetric_half_lane_ties(eye, expected):
    operation = getattr(offsets_module, "triton_relative_eye_offsets_prevalidated", None)
    assert callable(operation), "Quality needs an exact single-plane CUDA offset builder"
    near = np.full((1, 100), 0.5, dtype=np.float32)
    near[0, :5] = [
        0,
        np.nextafter(np.float32(0.5), np.float32(0)),
        0.5,
        np.nextafter(np.float32(0.5), np.float32(1)),
        1,
    ]
    result = operation(
        torch.from_numpy(near).cuda(), stereo_strength=0.125, convergence=0.0, eye=eye
    )
    assert result is not None and result.dtype == torch.int32 and result.is_cuda
    actual = result[0, :5].cpu().tolist()
    assert [actual[0], actual[2], actual[4]] == [expected[0], expected[2], expected[4]]
    assert abs(actual[1] - expected[1]) <= 1 and abs(actual[3] - expected[3]) <= 1


@requires_triton_cuda
@pytest.mark.parametrize("eye", ["left", "right"])
def test_single_eye_cuda_offsets_allocate_one_plane_and_respect_the_tail(monkeypatch, eye):
    operation = getattr(offsets_module, "triton_relative_eye_offsets_prevalidated", None)
    assert callable(operation), "Quality must not allocate both full-eye planes"
    near = torch.full((3, 257), 0.5, device="cuda")
    original_empty = torch.empty
    allocations = []

    def guarded_empty(shape, *, dtype, device):
        assert tuple(shape) == tuple(near.shape) and dtype == torch.int32
        full = original_empty(near.numel() + 19, dtype=dtype, device=device).fill_(123456789)
        allocations.append(full)
        return full[: near.numel()].view(shape)

    monkeypatch.setattr(torch, "empty", guarded_empty)
    result = operation(near, stereo_strength=2.0, convergence=0.5, eye=eye)
    assert result is not None and len(allocations) == 1
    assert torch.count_nonzero(result).item() == 0
    assert allocations[0][near.numel() :].cpu().tolist() == [123456789] * 19


@requires_triton_cuda
@pytest.mark.parametrize("eye", ["left", "right"])
@pytest.mark.parametrize("width", [1, 257, 8950])
@pytest.mark.parametrize(
    "strength, convergence",
    [(5.0, 0.0), (4.731, 0.3719), (0.473053267491137, 0.05202130106440961)],
)
def test_single_eye_cuda_matches_native_rounding_across_the_safe_range(
    eye, width, strength, convergence
):
    rng = np.random.default_rng(width)
    values = rng.uniform(-1, 2, (3, width)).astype(np.float32)
    values.flat[0] = -1
    values.flat[-1] = 2
    near = torch.from_numpy(values).cuda()
    settings = StereoRenderSettings(stereo_strength=strength, convergence=convergence)
    expected = quality._build_eye_offsets(SimpleNamespace(near_score=values), settings, eye)

    actual = offsets_module.triton_relative_eye_offsets_prevalidated(
        near, stereo_strength=strength, convergence=convergence, eye=eye
    )

    assert actual is not None
    # Native float32 Triton kernel stays within 1 subpixel lane (1/16 px) of float64.
    np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=0.0, atol=1)
    np.testing.assert_array_equal(near.cpu().numpy().view(np.uint32), values.view(np.uint32))


@requires_triton_cuda
def test_quality_releases_each_eye_offset_before_allocating_the_next(monkeypatch):
    fixture = _fixture()
    original_empty = torch.empty
    references = []

    def observe_empty(*args, **kwargs):
        is_offset = args and args[0] == fixture[0].shape[:2] and kwargs.get("dtype") == torch.int32
        if is_offset:
            assert all(reference() is None for reference in references)
        result = original_empty(*args, **kwargs)
        if is_offset:
            references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(torch, "empty", observe_empty)
    _render(StereoRenderer(device="cuda"), fixture)
    assert len(references) == 2
    assert all(reference() is None for reference in references)


@requires_triton_cuda
@pytest.mark.parametrize("reason", ["no_triton", "disabled", "size", "preload"])
def test_quality_device_offset_unavailability_uses_the_original_host_path(monkeypatch, reason):
    fixture = _fixture()
    expected = _render(StereoRenderer(device="cpu"), fixture)
    renderer = StereoRenderer(device="cuda", profile=True)
    original_build = quality.build_relative_eye_offsets_into
    host_eyes = []

    def observe_build(*args, **kwargs):
        host_eyes.append(kwargs["eye"])
        return original_build(*args, **kwargs)

    if reason == "no_triton":
        monkeypatch.setattr(offsets_module, "triton", None)
    elif reason == "disabled":
        monkeypatch.setattr(
            offsets_module, "_DISABLED_SINGLE_EYE_CUDA_DEVICES", {torch.cuda.current_device()}
        )
    elif reason == "size":
        monkeypatch.setattr(offsets_module, "_MAX_OFFSET_PIXELS", 1)
    else:
        monkeypatch.setattr(quality, "_preload_quality_device_frame", lambda *_args: None)
    monkeypatch.setattr(quality, "build_relative_eye_offsets_into", observe_build)
    actual = _render(renderer, fixture)

    _assert_equal(actual, expected)
    assert host_eyes == ["left", "right"]
    assert renderer._offset_transfer_bytes == fixture[0].shape[0] * fixture[0].shape[1] * 8


@requires_triton_cuda
@pytest.mark.parametrize("stage", ["allocation", "kernel"])
@pytest.mark.parametrize("eye", ["left", "right"])
def test_quality_offset_oom_releases_failed_buffers_before_host_fallback(monkeypatch, stage, eye):
    fixture = _fixture()
    expected = _render(StereoRenderer(device="cpu"), fixture)
    original_empty = torch.empty
    original_kernel = offsets_module._relative_offsets_kernel
    original_build = quality.build_relative_eye_offsets_into
    original_release = quality._release_after_oom
    allocation_count = 0
    references, host_eyes, injected = [], [], []
    failing_allocation = 1 if eye == "left" else 2

    def observe_empty(*args, **kwargs):
        nonlocal allocation_count
        is_offset = args and args[0] == fixture[0].shape[:2] and kwargs.get("dtype") == torch.int32
        if is_offset:
            allocation_count += 1
        result = original_empty(*args, **kwargs)
        if is_offset and allocation_count == failing_allocation:
            references.append(weakref.ref(result))
            if stage == "allocation":
                injected.append(stage)
                raise torch.cuda.OutOfMemoryError("injected partial offset allocation")
        return result

    class ObservedKernel:
        def __getitem__(self, grid):
            launch = original_kernel[grid]

            def observed(*args, **kwargs):
                if stage == "kernel" and allocation_count == failing_allocation and not injected:
                    injected.append(stage)
                    raise RuntimeError("Triton Error [CUDA]: out of memory")
                return launch(*args, **kwargs)

            return observed

    def observe_build(*args, **kwargs):
        assert all(reference() is None for reference in references)
        host_eyes.append(kwargs["eye"])
        return original_build(*args, **kwargs)

    def release(renderer):
        assert all(reference() is None for reference in references)
        return original_release(renderer)

    monkeypatch.setattr(torch, "empty", observe_empty)
    monkeypatch.setattr(offsets_module, "_relative_offsets_kernel", ObservedKernel())
    monkeypatch.setattr(offsets_module, "_DISABLED_SINGLE_EYE_CUDA_DEVICES", set())
    monkeypatch.setattr(quality, "build_relative_eye_offsets_into", observe_build)
    monkeypatch.setattr(quality, "_release_after_oom", release)
    actual = _render(StereoRenderer(device="cuda"), fixture)

    _assert_equal(actual, expected)
    assert injected == [stage] and host_eyes == [eye]
    assert not offsets_module._DISABLED_SINGLE_EYE_CUDA_DEVICES


@requires_triton_cuda
def test_quality_offset_compiler_failure_does_not_disable_fast_offsets(monkeypatch):
    fixture = _fixture()
    expected = _render(StereoRenderer(device="cpu"), fixture)

    class FailedKernel:
        def __getitem__(self, _grid):
            def launch(*_args, **_kwargs):
                raise offsets_module.triton.errors.TritonError(
                    "injected single-eye compile failure"
                )

            return launch

    monkeypatch.setattr(offsets_module, "_relative_offsets_kernel", FailedKernel())
    monkeypatch.setattr(offsets_module, "_DISABLED_SINGLE_EYE_CUDA_DEVICES", set())
    monkeypatch.setattr(offsets_module, "_DISABLED_CUDA_DEVICES", set())
    actual = _render(StereoRenderer(device="cuda"), fixture)

    _assert_equal(actual, expected)
    assert offsets_module._DISABLED_SINGLE_EYE_CUDA_DEVICES == {torch.cuda.current_device()}
    assert not offsets_module._DISABLED_CUDA_DEVICES


@requires_triton_cuda
def test_quality_offset_non_oom_runtime_errors_are_not_hidden_by_fallback(monkeypatch):
    class FailedKernel:
        def __getitem__(self, _grid):
            def launch(*_args, **_kwargs):
                raise RuntimeError("CUDA error: an illegal memory access was encountered")

            return launch

    def reject_fallback(*_args, **_kwargs):
        raise AssertionError("A non-OOM CUDA failure must propagate, not trigger host rendering")

    monkeypatch.setattr(offsets_module, "_relative_offsets_kernel", FailedKernel())
    monkeypatch.setattr(offsets_module, "_DISABLED_SINGLE_EYE_CUDA_DEVICES", set())
    monkeypatch.setattr(quality, "build_relative_eye_offsets_into", reject_fallback)
    with pytest.raises(RuntimeError, match="illegal memory access"):
        _render(StereoRenderer(device="cuda"), _fixture())
    assert not offsets_module._DISABLED_SINGLE_EYE_CUDA_DEVICES


@requires_triton_cuda
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 3.0, -2.0, 1e20])
def test_quality_scores_outside_the_device_proof_keep_host_validation(monkeypatch, value):
    values = np.full((2, 3), value, dtype=np.float32)
    geometry = SimpleNamespace(near_score=values)
    settings = _fixture()[2]
    device_frame = SimpleNamespace(near_score=torch.from_numpy(values).cuda())

    def reject_device(*_args, **_kwargs):
        raise AssertionError("Unproven Quality values must retain native host validation")

    monkeypatch.setattr(offsets_module, "triton_relative_eye_offsets_prevalidated", reject_device)
    if np.isfinite(value) and abs(value) < 10:
        expected = quality._build_eye_offsets(geometry, settings, "left")
        actual = quality._build_eye_offsets(geometry, settings, "left", device_frame)
        np.testing.assert_array_equal(actual, expected)
    else:
        with pytest.raises(ValueError) as baseline:
            quality._build_eye_offsets(geometry, settings, "left")
        with pytest.raises(ValueError) as candidate:
            quality._build_eye_offsets(geometry, settings, "left", device_frame)
        assert str(candidate.value) == str(baseline.value)
