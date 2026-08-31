from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from src.depth_surge_3d.rendering.forward_splat import SubpixelSplatResult
import src.depth_surge_3d.rendering.stereo_renderer as stereo_renderer


def _fast_fill_module():
    return importlib.import_module("src.depth_surge_3d.rendering.fast_fill_triton")


@pytest.mark.skipif(
    not torch.cuda.is_available() or _fast_fill_module().triton is None,
    reason="CUDA and Triton are required",
)
def test_triton_background_fill_matches_hand_checked_boundary_selection() -> None:
    fast_fill = _fast_fill_module()
    width = 7
    columns = torch.arange(width, dtype=torch.float32, device="cuda")
    image = columns.view(1, width, 1).expand(2, width, 3).contiguous()
    projected = torch.zeros((2, width), dtype=torch.float32, device="cuda")
    projected[1, 1] = 0.25
    projected[1, 5] = 0.25
    valid = torch.zeros((2, width), dtype=torch.bool, device="cuda")
    valid[1, 1] = True
    valid[1, 5] = True
    _, left_index, right_index = stereo_renderer._nearest_valid_indexes(valid)

    actual = fast_fill.triton_background_fill(
        image,
        valid,
        projected,
        left_index,
        right_index,
        max_gap_samples=8,
    )

    assert actual is not None
    rendered, holes, fillable = actual
    assert rendered[0].count_nonzero().item() == 0
    assert holes[0].tolist() == [True] * width
    assert fillable[0].tolist() == [False] * width
    assert rendered[1, :, 0].tolist() == [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0]
    assert holes[1].tolist() == [False] * width
    assert fillable[1].tolist() == [True, False, True, True, True, False, True]


@pytest.mark.skipif(
    not torch.cuda.is_available() or _fast_fill_module().triton is None,
    reason="CUDA and Triton are required",
)
@pytest.mark.parametrize("max_gap_samples", [0, 1, 17, 513])
def test_triton_background_fill_matches_torch_fallback_exactly(
    monkeypatch: pytest.MonkeyPatch,
    max_gap_samples: int,
) -> None:
    fast_fill = _fast_fill_module()
    generator = torch.Generator(device="cuda").manual_seed(20260830 + max_gap_samples)
    image = torch.randn((5, 513, 3), generator=generator, device="cuda")
    projected = torch.rand((5, 513), generator=generator, device="cuda")
    valid = torch.rand((5, 513), generator=generator, device="cuda") > 0.37
    valid[0] = False
    splat = SubpixelSplatResult(colour=image, disparity=projected, valid=valid)
    _, left_index, right_index = stereo_renderer._nearest_valid_indexes(valid)
    with monkeypatch.context() as fallback:
        fallback.setattr(
            fast_fill,
            "triton_background_fill",
            lambda *_args, **_kwargs: None,
        )
        expected = stereo_renderer._fill_background_band_state(
            splat,
            max_gap_samples=max_gap_samples,
        )

    actual = fast_fill.triton_background_fill(
        image,
        valid,
        projected,
        left_index,
        right_index,
        max_gap_samples=max_gap_samples,
    )

    assert actual is not None
    for actual_values, expected_values in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_values, expected_values, rtol=0, atol=0)


def test_triton_background_fill_rejects_host_tensors() -> None:
    fast_fill = _fast_fill_module()
    image = torch.zeros((1, 3, 3), dtype=torch.float32)
    valid = torch.zeros((1, 3), dtype=torch.bool)
    projected = torch.zeros((1, 3), dtype=torch.float32)
    indexes = torch.zeros((1, 3), dtype=torch.int32)

    actual = fast_fill.triton_background_fill(
        image,
        valid,
        projected,
        indexes,
        indexes,
        max_gap_samples=1,
    )

    assert actual is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_fill_kernel_failure_does_not_disable_triton_splat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fast_fill = _fast_fill_module()
    fast_splat = importlib.import_module("src.depth_surge_3d.rendering.fast_splat_triton")
    if fast_fill.triton is None or fast_splat.triton is None:
        pytest.skip("Triton is unavailable")

    class BrokenKernel:
        def __getitem__(self, _grid):
            def fail(*_args, **_kwargs):
                raise fast_fill.triton.errors.TritonError("simulated fill failure")

            return fail

    image = torch.zeros((1, 3, 3), dtype=torch.float32, device="cuda")
    valid = torch.zeros((1, 3), dtype=torch.bool, device="cuda")
    projected = torch.zeros((1, 3), dtype=torch.float32, device="cuda")
    indexes = torch.zeros((1, 3), dtype=torch.int32, device="cuda")
    monkeypatch.setattr(fast_fill, "_DISABLED_CUDA_DEVICES", set(), raising=False)
    monkeypatch.setattr(fast_fill, "_background_fill_kernel", BrokenKernel())

    actual = fast_fill.triton_background_fill(
        image,
        valid,
        projected,
        indexes,
        indexes,
        max_gap_samples=1,
    )

    assert actual is None
    assert not fast_fill.triton_fill_available(image.device)
    assert fast_splat.triton_splat_available(image.device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("stage", ["output", "holes", "fillable", "kernel", "driver_kernel"])
def test_fill_oom_is_not_swallowed_or_cached(monkeypatch, stage):
    fast_fill = _fast_fill_module()
    if fast_fill.triton is None:
        pytest.skip("Triton is unavailable")
    image = torch.zeros((2, 7, 3), dtype=torch.float32, device="cuda")
    valid = torch.ones((2, 7), dtype=torch.bool, device="cuda")
    projected = torch.zeros((2, 7), dtype=torch.float32, device="cuda")
    indexes = torch.zeros((2, 7), dtype=torch.int32, device="cuda")
    failure = (
        RuntimeError("Triton Error [CUDA]: out of memory")
        if stage == "driver_kernel"
        else torch.cuda.OutOfMemoryError("injected fill allocation failure")
    )
    monkeypatch.setattr(fast_fill, "_DISABLED_CUDA_DEVICES", set())
    if stage in {"output", "holes", "fillable"}:
        original = torch.empty_like
        allocations = 0

        def fail_allocation(*args, **kwargs):
            nonlocal allocations
            allocations += 1
            if allocations == {"output": 1, "holes": 2, "fillable": 3}[stage]:
                raise failure
            return original(*args, **kwargs)

        monkeypatch.setattr(torch, "empty_like", fail_allocation)
    else:

        class BrokenKernel:
            def __getitem__(self, _grid):
                def fail(*_args, **_kwargs):
                    raise failure

                return fail

        monkeypatch.setattr(fast_fill, "_background_fill_kernel", BrokenKernel())
    with pytest.raises(torch.cuda.OutOfMemoryError) as captured:
        fast_fill.triton_background_fill(
            image, valid, projected, indexes, indexes, max_gap_samples=3
        )
    if stage == "driver_kernel":
        assert captured.value.__cause__ is failure
    else:
        assert captured.value is failure
    assert not fast_fill._DISABLED_CUDA_DEVICES


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    "message",
    [
        "Triton Error [CUDA]: an illegal memory access was encountered",
        "Triton Error [CUDA]: invalid argument",
        "out of memory",
    ],
)
def test_unrelated_fill_driver_errors_are_not_retried_or_silenced(monkeypatch, message):
    fast_fill = _fast_fill_module()
    if fast_fill.triton is None:
        pytest.skip("Triton is unavailable")
    image = torch.zeros((1, 7, 3), dtype=torch.float32, device="cuda")
    valid = torch.ones((1, 7), dtype=torch.bool, device="cuda")
    projected = torch.zeros((1, 7), dtype=torch.float32, device="cuda")
    indexes = torch.zeros((1, 7), dtype=torch.int32, device="cuda")
    failure = RuntimeError(message)

    class BrokenKernel:
        def __getitem__(self, _grid):
            def fail(*_args, **_kwargs):
                raise failure

            return fail

    monkeypatch.setattr(fast_fill, "_DISABLED_CUDA_DEVICES", set())
    monkeypatch.setattr(fast_fill, "_background_fill_kernel", BrokenKernel())
    with pytest.raises(RuntimeError) as captured:
        fast_fill.triton_background_fill(
            image, valid, projected, indexes, indexes, max_gap_samples=3
        )
    assert captured.value is failure
    assert not fast_fill._DISABLED_CUDA_DEVICES


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("compact", [False, True])
def test_fill_driver_oom_retries_smaller_bands_without_changing_output(monkeypatch, compact):
    fast_fill = _fast_fill_module()
    if fast_fill.triton is None or not stereo_renderer._triton_splat_available(
        torch.device("cuda")
    ):
        pytest.skip("Triton Fast kernels are unavailable")
    rng = np.random.default_rng(20260901)
    source = rng.integers(0, 256, (7, 73, 3), dtype=np.uint8)
    depth = rng.random((7, 73), dtype=np.float32)
    settings = stereo_renderer.StereoRenderSettings(
        stereo_strength=4.25, convergence=0.37, occlusion_fill="background"
    )
    renderer = stereo_renderer.StereoRenderer(device="cuda", temporary_budget_bytes=73 * 1088 * 2)
    render = renderer.render_relative_fast_compact if compact else renderer.render
    with monkeypatch.context() as fallback:
        fallback.setattr(fast_fill, "triton_background_fill", lambda *_args, **_kwargs: None)
        expected = render(source, depth, settings)
    original = fast_fill._background_fill_kernel
    observed_rows = []

    class FailsOnce:
        def __getitem__(self, grid):
            def call(*args, **kwargs):
                observed_rows.append(args[1].shape[0])
                if len(observed_rows) == 1:
                    raise RuntimeError("Triton Error [CUDA]: out of memory")
                return original[grid](*args, **kwargs)

            return call

    monkeypatch.setattr(fast_fill, "_DISABLED_CUDA_DEVICES", set())
    monkeypatch.setattr(fast_fill, "_background_fill_kernel", FailsOnce())
    actual = render(source, depth, settings)
    assert observed_rows[0] == 2 and all(rows == 1 for rows in observed_rows[1:])
    assert not fast_fill._DISABLED_CUDA_DEVICES
    if compact:
        for field in (
            "left_image",
            "right_image",
            "left_coverage_count",
            "right_coverage_count",
            "left_repair_bits",
            "right_repair_bits",
        ):
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
        for field in ("left_fast_diagnostics", "right_fast_diagnostics"):
            actual_eye, expected_eye = getattr(actual, field), getattr(expected, field)
            assert actual_eye.local_filled_lane_count == expected_eye.local_filled_lane_count
            assert (
                actual_eye.final_unresolved_lane_count == expected_eye.final_unresolved_lane_count
            )
            np.testing.assert_array_equal(
                actual_eye.hole_runs.lane_count_histogram,
                expected_eye.hole_runs.lane_count_histogram,
            )
            np.testing.assert_array_equal(
                actual_eye.hole_runs.touched_pixel_span_histogram,
                expected_eye.hole_runs.touched_pixel_span_histogram,
            )
    else:
        for field in stereo_renderer.StereoRenderResult.__dataclass_fields__:
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
