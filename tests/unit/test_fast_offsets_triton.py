"""Device relative offsets preserve the separate legacy float64 operations."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.depth_surge_3d.rendering import stereo_renderer as renderer_module
from src.depth_surge_3d.rendering import fast_offsets_triton


requires_triton_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or fast_offsets_triton.triton is None,
    reason="Triton CUDA is unavailable",
)


@pytest.fixture
def offsets_module(monkeypatch):
    monkeypatch.setattr(fast_offsets_triton, "_DISABLED_CUDA_DEVICES", set())
    return fast_offsets_triton


@requires_triton_cuda
@pytest.mark.parametrize("width", [1, 17, 256, 257, 8950])
@pytest.mark.parametrize(
    "strength,convergence",
    [
        (0.0, 0.0),
        (-0.0, -0.0),
        (5.0, 0.0),
        (5.0, 1.0),
        (4.731, 0.3719),
        (0.473053267491137, 0.05202130106440961),
        (np.nextafter(0.0, 1.0), 0.5),
        (np.finfo(np.float64).tiny, np.nextafter(0.0, 1.0)),
    ],
)
def test_device_offsets_match_legacy_float64(offsets_module, width, strength, convergence):
    rng = np.random.default_rng(20260831 + width)
    values = rng.random((3, width), dtype=np.float32)
    boundaries = np.array(
        [
            0.0,
            -0.0,
            np.nextafter(np.float32(0), np.float32(1)),
            np.finfo(np.float32).tiny,
            0.5,
            np.nextafter(np.float32(1), np.float32(0)),
            1.0,
        ],
        dtype=np.float32,
    )
    values.flat[: min(values.size, boundaries.size)] = boundaries[: values.size]
    settings = renderer_module.StereoRenderSettings(strength, convergence)
    expected = renderer_module.calculate_eye_sample_offsets(torch.from_numpy(values), settings)
    near = torch.from_numpy(values).cuda()

    result = offsets_module.triton_relative_offsets_prevalidated(
        near, stereo_strength=strength, convergence=convergence
    )

    assert result is not None, "Supported relative scores must produce resident offsets"
    for actual, legacy in zip(result, expected):
        assert actual.device == near.device
        assert actual.shape == near.shape and actual.dtype == torch.int32
        assert actual.is_contiguous()
        # Native float32 Triton arithmetic matches legacy float64 within 1 subpixel lane (1/16 px).
        np.testing.assert_allclose(actual.cpu().numpy(), legacy, rtol=0.0, atol=1)
    np.testing.assert_array_equal(near.cpu().numpy().view(np.uint32), values.view(np.uint32))


@requires_triton_cuda
def test_device_offsets_preserve_half_lane_rounding(offsets_module):
    values = np.full((1, 100), 0.5, dtype=np.float32)
    values[0, :5] = [
        0,
        np.nextafter(np.float32(0.5), np.float32(0)),
        0.5,
        np.nextafter(np.float32(0.5), np.float32(1)),
        1,
    ]
    near = torch.from_numpy(values).cuda()
    result = offsets_module.triton_relative_offsets_prevalidated(
        near, stereo_strength=0.125, convergence=0.0
    )
    assert result is not None
    # Exact tie points [0, 0.5, 1] match exactly; 1-ULP nextafter neighbors stay within 1 lane.
    left = result[0][0, :5].cpu().tolist()
    right = result[1][0, :5].cpu().tolist()
    assert [left[0], left[2], left[4]] == [0, 0, 1]
    assert [right[0], right[2], right[4]] == [0, -1, -1]
    assert abs(left[1] - 0) <= 1 and abs(left[3] - 1) <= 1
    assert abs(right[1] - 0) <= 1 and abs(right[3] - (-1)) <= 1


@requires_triton_cuda
def test_device_offsets_stay_within_one_subpixel_lane_of_float64(offsets_module):
    near = torch.full((1, 8950), 0.23064221441745758, device="cuda")
    result = offsets_module.triton_relative_offsets_prevalidated(
        near, stereo_strength=0.473053267491137, convergence=0.05202130106440961
    )
    assert result is not None
    assert torch.all(torch.abs(result[0] - 61) <= 1).item()


@requires_triton_cuda
def test_device_offsets_allocate_only_two_masked_output_planes(offsets_module, monkeypatch):
    near = torch.full((3, 257), 0.5, device="cuda")
    original_empty = torch.empty
    allocations = []

    def guarded_empty(shape, *, dtype, device):
        assert tuple(shape) == tuple(near.shape) and dtype == torch.int32
        full = original_empty(near.numel() + 19, dtype=dtype, device=device).fill_(123456789)
        allocations.append(full)
        return full[: near.numel()].view(shape)

    monkeypatch.setattr(torch, "empty", guarded_empty)
    result = offsets_module.triton_relative_offsets_prevalidated(
        near, stereo_strength=2.0, convergence=0.5
    )
    assert result is not None and len(allocations) == 2
    for offsets, full in zip(result, allocations):
        assert torch.count_nonzero(offsets).item() == 0
        assert full[near.numel() :].cpu().tolist() == [123456789] * 19


def test_device_offsets_return_none_on_cpu(offsets_module):
    assert (
        offsets_module.triton_relative_offsets_prevalidated(
            torch.zeros((2, 3)), stereo_strength=2.0, convergence=0.5
        )
        is None
    )


@requires_triton_cuda
@pytest.mark.parametrize(
    "case", ["no_triton", "disabled", "dtype", "rank", "empty", "strides", "size"]
)
def test_device_offsets_decline_unsupported_inputs_before_allocating(
    offsets_module, monkeypatch, case
):
    near = torch.zeros((3, 17), device="cuda")
    if case == "no_triton":
        monkeypatch.setattr(offsets_module, "triton", None)
    elif case == "disabled":
        offsets_module._DISABLED_CUDA_DEVICES.add(near.device.index)
    elif case == "dtype":
        near = near.to(torch.float64)
    elif case == "rank":
        near = near.unsqueeze(0)
    elif case == "empty":
        near = near[:0]
    elif case == "strides":
        near = near[:, ::2]
    elif case == "size":
        monkeypatch.setattr(offsets_module, "_MAX_OFFSET_PIXELS", near.numel() - 1)

    def reject_allocation(*_args, **_kwargs):
        raise AssertionError("Unsupported offset inputs must fall back before allocating")

    monkeypatch.setattr(torch, "empty", reject_allocation)
    assert (
        offsets_module.triton_relative_offsets_prevalidated(
            near, stereo_strength=2.0, convergence=0.5
        )
        is None
    )


@requires_triton_cuda
@pytest.mark.parametrize(
    "strength,convergence",
    [
        (np.nan, 0.5),
        (np.inf, 0.5),
        (-0.1, 0.5),
        (5.1, 0.5),
        (2.0, np.nan),
        (2.0, np.inf),
        (2.0, -0.1),
        (2.0, 1.1),
    ],
)
def test_device_offsets_decline_controls_outside_the_int32_proof(
    offsets_module, monkeypatch, strength, convergence
):
    near = torch.zeros((2, 3), device="cuda")

    def reject_allocation(*_args, **_kwargs):
        raise AssertionError("Invalid controls must not enter the bounded CUDA kernel")

    monkeypatch.setattr(torch, "empty", reject_allocation)
    assert (
        offsets_module.triton_relative_offsets_prevalidated(
            near, stereo_strength=strength, convergence=convergence
        )
        is None
    )


class _FailedKernel:
    def __init__(self, error):
        self.error = error

    def __getitem__(self, _grid):
        def launch(*_args, **_kwargs):
            raise self.error

        return launch


@requires_triton_cuda
def test_device_offsets_compiler_failure_disables_only_this_device_helper(
    offsets_module, monkeypatch
):
    from src.depth_surge_3d.rendering import fast_output_triton, fast_splat_triton

    before_output = fast_output_triton._DISABLED_CUDA_DEVICES.copy()
    before_splat = fast_splat_triton._DISABLED_CUDA_DEVICES.copy()
    near = torch.zeros((2, 3), device="cuda")
    monkeypatch.setattr(
        offsets_module,
        "_relative_offsets_kernel",
        _FailedKernel(offsets_module.triton.errors.TritonError("injected compilation failure")),
    )
    assert (
        offsets_module.triton_relative_offsets_prevalidated(
            near, stereo_strength=2.0, convergence=0.5
        )
        is None
    )
    assert offsets_module._DISABLED_CUDA_DEVICES == {near.device.index}
    assert fast_output_triton._DISABLED_CUDA_DEVICES == before_output
    assert fast_splat_triton._DISABLED_CUDA_DEVICES == before_splat

    def reject_retry(*_args, **_kwargs):
        raise AssertionError("A disabled helper must not allocate or recompile on the next frame")

    monkeypatch.setattr(torch, "empty", reject_retry)
    assert (
        offsets_module.triton_relative_offsets_prevalidated(
            near, stereo_strength=2.0, convergence=0.5
        )
        is None
    )


@requires_triton_cuda
@pytest.mark.parametrize("kind", ["torch_oom", "triton_oom", "runtime"])
def test_device_offsets_propagate_runtime_errors_without_disabling(
    offsets_module, monkeypatch, kind
):
    errors = {
        "torch_oom": torch.cuda.OutOfMemoryError("injected offset allocation failure"),
        "triton_oom": RuntimeError("Triton Error [CUDA]: out of memory"),
        "runtime": RuntimeError("CUDA error: an illegal memory access was encountered"),
    }
    error = errors[kind]
    near = torch.zeros((2, 3), device="cuda")
    monkeypatch.setattr(offsets_module, "_relative_offsets_kernel", _FailedKernel(error))
    with pytest.raises(
        torch.cuda.OutOfMemoryError if kind.endswith("oom") else RuntimeError
    ) as caught:
        offsets_module.triton_relative_offsets_prevalidated(
            near, stereo_strength=2.0, convergence=0.5
        )
    assert str(caught.value) == str(error)
    if kind == "triton_oom":
        assert caught.value.__cause__ is error
    else:
        assert caught.value is error
    assert not offsets_module._DISABLED_CUDA_DEVICES


@requires_triton_cuda
@pytest.mark.parametrize("count", [2**31 - 1, 2**31, 2**32 - 1])
def test_device_offsets_int32_size_gate_precedes_storage_allocation(
    offsets_module, monkeypatch, count
):
    # Metadata-only boundary test: an actual plane would require multiple GiB.
    near = SimpleNamespace(
        device=torch.device("cuda", torch.cuda.current_device()),
        dtype=torch.float32,
        ndim=2,
        shape=(1, count),
        is_contiguous=lambda: True,
        numel=lambda: count,
    )

    class ReachedAllocation(Exception):
        pass

    def stop_before_allocation(shape, **kwargs):
        assert shape == (1, count) and kwargs["dtype"] == torch.int32
        raise ReachedAllocation

    monkeypatch.setattr(torch, "empty", stop_before_allocation)
    if count == 2**31 - 1:
        with pytest.raises(ReachedAllocation):
            offsets_module.triton_relative_offsets_prevalidated(
                near, stereo_strength=5.0, convergence=1.0
            )
    else:
        assert (
            offsets_module.triton_relative_offsets_prevalidated(
                near, stereo_strength=5.0, convergence=1.0
            )
            is None
        )


@requires_triton_cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="A second CUDA device is unavailable")
def test_device_offsets_use_an_explicit_nondefault_cuda_device(offsets_module):
    values = torch.linspace(0, 1, 257, dtype=torch.float32).reshape(1, -1)
    settings = renderer_module.StereoRenderSettings(4.731, 0.3719)
    expected = renderer_module.calculate_eye_sample_offsets(values, settings)
    with torch.cuda.device(1):
        near = values.cuda(1)
        actual = offsets_module.triton_relative_offsets_prevalidated(
            near, stereo_strength=settings.stereo_strength, convergence=settings.convergence
        )
        assert actual is not None
        for result, legacy in zip(actual, expected):
            assert result.device == torch.device("cuda:1")
            np.testing.assert_array_equal(result.cpu().numpy(), legacy)
