"""Nearest-boundary scans retain values, not unused Torch arg-index rasters."""

import numpy as np
import pytest
import torch

from src.depth_surge_3d.rendering import fast_scan_triton, stereo_renderer


@pytest.fixture
def cuda_scan(monkeypatch):
    if not torch.cuda.is_available() or fast_scan_triton.triton is None:
        pytest.skip("CUDA Triton is unavailable")
    monkeypatch.setattr(fast_scan_triton, "_DISABLED_CUDA_DEVICES", set())
    return fast_scan_triton


def _expected_boundaries(valid):
    columns = np.broadcast_to(np.arange(valid.shape[1], dtype=np.int32), valid.shape)
    left = np.maximum.accumulate(np.where(valid, columns, -1), axis=1)
    right = np.minimum.accumulate(np.where(valid, columns, valid.shape[1])[:, ::-1], axis=1)[
        :, ::-1
    ]
    return columns, left, right


def _assert_boundaries(actual, expected):
    assert len(actual) == len(expected)
    for result, wanted in zip(actual, expected):
        assert result.dtype == torch.int32
        np.testing.assert_array_equal(result.cpu().numpy(), wanted)


def test_cuda_nearest_boundaries_use_only_bounded_value_storage(cuda_scan):
    rng = np.random.default_rng(20260831)
    host = rng.random((64, 4096)) > 0.7
    valid = torch.from_numpy(host).cuda()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    columns, left, right = stereo_renderer._nearest_valid_indexes(valid)
    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated() - before

    _assert_boundaries((columns, left, right), _expected_boundaries(host))
    assert allocated <= 12 * host.size + 4 * host.shape[1] + 32768


@pytest.mark.parametrize("width", [30720, 61440, 65536])
def test_cuda_scan_has_no_unaccounted_driver_local_storage(cuda_scan, monkeypatch, width):
    if torch.version.hip is not None:
        pytest.skip("This check uses NVIDIA compiled-kernel allocation metadata")
    valid = torch.ones((3, width), dtype=torch.bool, device="cuda")
    compiled = []

    class ObserveKernel:
        def __init__(self, original):
            self.original = original

        def __getitem__(self, grid):
            def call(*args, **kwargs):
                kernel = self.original[grid](*args, **kwargs)
                compiled.append(kernel)
                return kernel

            return call

    for name in ("_nearest_valid_indexes_kernel", "_propagate_boundaries_kernel"):
        monkeypatch.setattr(cuda_scan, name, ObserveKernel(getattr(cuda_scan, name)))
    result = cuda_scan.triton_nearest_valid_indexes(valid)
    assert result is not None and compiled
    torch.cuda.synchronize()
    assert all(kernel.n_spills == 0 for kernel in compiled)
    assert all(kernel.metadata.global_scratch_size == 0 for kernel in compiled)


def test_cuda_nearest_boundaries_keep_inclusive_indexes_and_row_sentinels(cuda_scan):
    valid = torch.tensor(
        [[False, True, False, False, True, False], [False] * 6, [True] * 6],
        dtype=torch.bool,
        device="cuda",
    )
    result = cuda_scan.triton_nearest_valid_indexes(valid)
    assert result is not None
    _assert_boundaries(
        result,
        (
            [[-1, 1, 1, 1, 4, 4], [-1] * 6, [0, 1, 2, 3, 4, 5]],
            [[1, 1, 4, 4, 4, 6], [6] * 6, [0, 1, 2, 3, 4, 5]],
        ),
    )


@pytest.mark.parametrize(
    "width",
    [
        1,
        2,
        15,
        16,
        17,
        31,
        32,
        33,
        255,
        256,
        257,
        1023,
        1024,
        1025,
        2047,
        2048,
        2049,
        4095,
        4096,
        4097,
        30719,
        30720,
        61440,
        65535,
        65536,
    ],
)
@pytest.mark.parametrize("pattern", ["empty", "full", "isolated", "edges", "alternating", "random"])
def test_cuda_nearest_boundaries_match_exact_prefix_and_suffix(cuda_scan, width, pattern):
    host = np.zeros((3, width), dtype=np.bool_)
    if pattern == "full":
        host[:] = True
    elif pattern == "isolated":
        host[:, width // 2] = True
    elif pattern == "edges":
        host[:, 0] = True
        host[:, -1] = True
    elif pattern == "alternating":
        host[:, ::2] = True
    elif pattern == "random":
        host[:] = np.random.default_rng(width).random(host.shape) > 0.6
    valid = torch.from_numpy(host).cuda()
    result = cuda_scan.triton_nearest_valid_indexes(valid)
    assert result is not None
    assert all(value.shape == valid.shape and value.is_contiguous() for value in result)
    _assert_boundaries(result, _expected_boundaries(host)[1:])
    np.testing.assert_array_equal(valid.cpu().numpy(), host)


def test_cuda_nearest_boundaries_propagate_across_empty_tiles_without_crossing_rows(cuda_scan):
    host = np.zeros((4, 8193), dtype=np.bool_)
    host[1, [2047, 6144]] = True
    host[2, [0, 2048, 4095, 4096, 8192]] = True
    host[3, 2049] = True
    result = cuda_scan.triton_nearest_valid_indexes(torch.from_numpy(host).cuda())
    assert result is not None
    _assert_boundaries(result, _expected_boundaries(host)[1:])


@pytest.mark.parametrize(
    "layout",
    [
        "rows",
        "columns",
        "transpose",
        "zero_rows",
        "zero_width",
        "oversized",
        "uint8",
        "float32",
        "three_dimensional",
    ],
)
def test_unsupported_cuda_scans_fall_back_without_allocating(cuda_scan, monkeypatch, layout):
    valid = torch.zeros((3, 10), dtype=torch.bool, device="cuda")
    if layout == "rows":
        valid = valid[::2]
    elif layout == "columns":
        valid = valid[:, ::2]
    elif layout == "transpose":
        valid = valid.T
    elif layout == "zero_rows":
        valid = valid[:0]
    elif layout == "zero_width":
        valid = valid[:, :0]
    elif layout == "oversized":
        valid = torch.zeros((1, 65537), dtype=torch.bool, device="cuda")
    elif layout == "uint8":
        valid = valid.to(dtype=torch.uint8)
    elif layout == "float32":
        valid = valid.to(dtype=torch.float32)
    else:
        valid = valid.unsqueeze(0)

    def reject_allocation(*_args, **_kwargs):
        raise AssertionError("Unsupported scan inputs must not allocate outputs")

    monkeypatch.setattr(torch, "empty", reject_allocation)
    monkeypatch.setattr(torch, "empty_like", reject_allocation)
    assert cuda_scan.triton_nearest_valid_indexes(valid) is None


@pytest.mark.parametrize("layout", ["rows", "columns", "transpose", "oversized"])
def test_renderer_retains_exact_torch_fallback_for_unsupported_layouts(cuda_scan, layout):
    host = np.random.default_rng(29).random((5, 12)) > 0.4
    valid = torch.from_numpy(host).cuda()
    if layout == "rows":
        valid = valid[::2]
    elif layout == "columns":
        valid = valid[:, ::2]
    elif layout == "transpose":
        valid = valid.T
    else:
        valid = torch.zeros((2, 65537), dtype=torch.bool, device="cuda")
        valid[:, 63::127] = True
    _assert_boundaries(
        stereo_renderer._nearest_valid_indexes(valid),
        _expected_boundaries(valid.cpu().numpy()),
    )


def test_cpu_nearest_boundaries_do_not_require_triton(monkeypatch):
    monkeypatch.setattr(fast_scan_triton, "triton", None)
    host = np.array([[False, True, False], [False, False, False]], dtype=np.bool_)
    valid = torch.from_numpy(host)
    assert fast_scan_triton.triton_nearest_valid_indexes(valid) is None
    _assert_boundaries(stereo_renderer._nearest_valid_indexes(valid), _expected_boundaries(host))


def test_missing_triton_retains_exact_cuda_fallback(cuda_scan, monkeypatch):
    monkeypatch.setattr(cuda_scan, "triton", None)
    host = np.random.default_rng(57).random((3, 71)) > 0.5
    valid = torch.from_numpy(host).cuda()
    _assert_boundaries(stereo_renderer._nearest_valid_indexes(valid), _expected_boundaries(host))


@pytest.mark.parametrize(
    "kernel_name", ["_nearest_valid_indexes_kernel", "_propagate_boundaries_kernel"]
)
def test_cuda_scan_failure_disables_only_scan_and_retains_boundaries(
    cuda_scan, monkeypatch, kernel_name
):
    from src.depth_surge_3d.rendering import fast_fill_triton, fast_output_triton, fast_splat_triton

    other_modules = (fast_fill_triton, fast_output_triton, fast_splat_triton)
    original_flags = [set(module._DISABLED_CUDA_DEVICES) for module in other_modules]
    calls = []

    class BrokenKernel:
        def __getitem__(self, grid):
            def fail(*_args, **_kwargs):
                calls.append(grid)
                raise cuda_scan.triton.errors.TritonError("injected scan compilation failure")

            return fail

    monkeypatch.setattr(cuda_scan, kernel_name, BrokenKernel())
    host = np.array([[False, True, False, True], [False] * 4], dtype=np.bool_)
    valid = torch.from_numpy(host).cuda()
    for _ in range(2):
        _assert_boundaries(
            stereo_renderer._nearest_valid_indexes(valid), _expected_boundaries(host)
        )
    assert len(calls) == 1
    assert valid.device.index in cuda_scan._DISABLED_CUDA_DEVICES
    assert [set(module._DISABLED_CUDA_DEVICES) for module in other_modules] == original_flags


@pytest.mark.parametrize(
    "stage",
    [
        "first_output",
        "second_output",
        "summaries",
        "kernel",
        "propagation_kernel",
        "driver_kernel",
        "driver_propagation_kernel",
    ],
)
def test_cuda_scan_oom_is_not_swallowed_or_cached(cuda_scan, monkeypatch, stage):
    valid = torch.ones((2, 31), dtype=torch.bool, device="cuda")
    failure = (
        RuntimeError("Triton Error [CUDA]: out of memory")
        if stage.startswith("driver_")
        else torch.cuda.OutOfMemoryError("injected scan allocation failure")
    )

    def fail(*_args, **_kwargs):
        raise failure

    if stage == "first_output":
        monkeypatch.setattr(torch, "empty", fail)
    elif stage == "second_output":
        monkeypatch.setattr(torch, "empty_like", fail)
    elif stage == "summaries":
        original_empty = torch.empty
        allocations = 0

        def fail_summary(*args, **kwargs):
            nonlocal allocations
            allocations += 1
            if allocations == 2:
                raise failure
            return original_empty(*args, **kwargs)

        monkeypatch.setattr(torch, "empty", fail_summary)
    else:

        class BrokenKernel:
            def __getitem__(self, _grid):
                return fail

        kernel_name = (
            "_propagate_boundaries_kernel"
            if "propagation" in stage
            else "_nearest_valid_indexes_kernel"
        )
        monkeypatch.setattr(cuda_scan, kernel_name, BrokenKernel())
    with pytest.raises(torch.cuda.OutOfMemoryError) as captured:
        cuda_scan.triton_nearest_valid_indexes(valid)
    if stage.startswith("driver_"):
        assert captured.value.__cause__ is failure
    else:
        assert captured.value is failure
    assert not cuda_scan._DISABLED_CUDA_DEVICES


@pytest.mark.parametrize(
    "message",
    [
        "Triton Error [CUDA]: an illegal memory access was encountered",
        "Triton Error [CUDA]: invalid argument",
        "out of memory",
    ],
)
def test_unrelated_driver_errors_are_not_retried_or_silenced(cuda_scan, monkeypatch, message):
    valid = torch.ones((2, 31), dtype=torch.bool, device="cuda")
    failure = RuntimeError(message)

    class BrokenKernel:
        def __getitem__(self, _grid):
            def fail(*_args, **_kwargs):
                raise failure

            return fail

    monkeypatch.setattr(cuda_scan, "_nearest_valid_indexes_kernel", BrokenKernel())
    with pytest.raises(RuntimeError) as captured:
        cuda_scan.triton_nearest_valid_indexes(valid)
    assert captured.value is failure
    assert not cuda_scan._DISABLED_CUDA_DEVICES


def test_cuda_scan_uses_the_calling_stream_and_nonzero_storage_offset(cuda_scan):
    stream = torch.cuda.Stream()
    host = np.random.default_rng(81).random((7, 4097)) > 0.7
    with torch.cuda.stream(stream):
        valid = torch.from_numpy(host).cuda()[2:5]
        assert valid.storage_offset() > 0 and valid.is_contiguous()
        result = cuda_scan.triton_nearest_valid_indexes(valid)
        assert result is not None
        _assert_boundaries(result, _expected_boundaries(host[2:5])[1:])
    stream.synchronize()


@pytest.mark.parametrize("driver_error", [False, True])
@pytest.mark.parametrize(
    "kernel_name", ["_nearest_valid_indexes_kernel", "_propagate_boundaries_kernel"]
)
def test_fast_render_retries_a_scan_oom_with_identical_pixels_and_diagnostics(
    cuda_scan, monkeypatch, driver_error, kernel_name
):
    rng = np.random.default_rng(83)
    source = rng.integers(0, 256, size=(7, 23, 3), dtype=np.uint8)
    canonical = rng.random((5, 13), dtype=np.float32)
    renderer = stereo_renderer.StereoRenderer(
        device="cuda", temporary_budget_bytes=23 * stereo_renderer.TRITON_SPLAT_BYTES_PER_PIXEL * 2
    )
    settings = stereo_renderer.StereoRenderSettings()
    with monkeypatch.context() as fallback:
        fallback.setattr(cuda_scan, "triton_nearest_valid_indexes", lambda _valid: None)
        expected = renderer.render_relative_fast_compact(source, canonical, settings)
    original = getattr(cuda_scan, kernel_name)
    observed_rows = []

    class FailsOnce:
        def __getitem__(self, grid):
            def call(*args, **kwargs):
                observed_rows.append(grid[0])
                if len(observed_rows) == 1:
                    if driver_error:
                        raise RuntimeError("Triton Error [CUDA]: out of memory")
                    raise torch.cuda.OutOfMemoryError("injected boundary scan OOM")
                return original[grid](*args, **kwargs)

            return call

    monkeypatch.setattr(cuda_scan, kernel_name, FailsOnce())
    actual = renderer.render_relative_fast_compact(source, canonical, settings)
    assert observed_rows[0] == 2 and all(rows == 1 for rows in observed_rows[1:])
    assert not cuda_scan._DISABLED_CUDA_DEVICES
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
        actual_eye = getattr(actual, name)
        expected_eye = getattr(expected, name)
        assert actual_eye.local_filled_lane_count == expected_eye.local_filled_lane_count
        assert actual_eye.final_unresolved_lane_count == expected_eye.final_unresolved_lane_count
        for histogram in ("lane_count_histogram", "touched_pixel_span_histogram"):
            np.testing.assert_array_equal(
                getattr(actual_eye.hole_runs, histogram), getattr(expected_eye.hole_runs, histogram)
            )
