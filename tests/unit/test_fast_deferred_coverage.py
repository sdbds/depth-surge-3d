"""Fused bands can let bounded native work publish coverage before the eye joins."""

from concurrent.futures import Future
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.depth_surge_3d.rendering import fast_output_triton as output_module
from src.depth_surge_3d.rendering import stereo_renderer as renderer_module
from src.depth_surge_3d.rendering.quality.diagnostic_transaction import canonical_json_bytes
from src.depth_surge_3d.rendering.quality.diagnostics import build_quality_frame_stats


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
        self.shutdown_called = False

    def submit(self, function, *args):
        return DeferredFuture(function, args)

    def shutdown(self, *, wait, cancel_futures):
        assert wait is True
        self.shutdown_called = True


@pytest.fixture
def native():
    result = renderer_module._native_lane_analyzer()
    if result is None:
        pytest.skip("Native lane analysis is unavailable")
    return result


@pytest.fixture
def deferred_executor(monkeypatch):
    monkeypatch.setattr(renderer_module, "ThreadPoolExecutor", DeferredExecutor)


def buffers(packed):
    height, width = packed.shape
    return (
        np.zeros(16 * (width + 1), dtype=np.uint64),
        np.zeros(width + 1, dtype=np.uint64),
        np.full((height, width), 255, dtype=np.uint8),
    )


def direct_result(native, packed):
    result = buffers(packed)
    native.accumulate_packed_lane_hole_run_histograms_into(packed, *result)
    return result


def assert_same(actual, expected):
    for value, oracle in zip(actual, expected):
        np.testing.assert_array_equal(value, oracle)


@pytest.mark.parametrize("word", [0, 0xFFFF, 0xA55A])
def test_deferred_coverage_is_written_by_native_work_at_join(
    native, deferred_executor, monkeypatch, word
):
    packed = np.full((5, 7), word, dtype=np.uint16)
    expected = direct_result(native, packed)
    lane, span, coverage = actual = buffers(packed)

    class ForbiddenCoverageLookup:
        def __getitem__(self, _key):
            raise AssertionError("deferred coverage was recomputed on the caller")

    monkeypatch.setattr(renderer_module, "_PACKED_LANE_COVERAGE", ForbiddenCoverageLookup())
    with renderer_module._FastLaneAnalysisQueue(native, async_threshold_seconds=0.0) as queue:
        queue.submit(packed[:2], lane, span, coverage[:2], defer_coverage=True)
        queue.submit(packed[2:], lane, span, coverage[2:], defer_coverage=True)
        assert np.all(coverage[2:] == 255)
        queue.finish_pending()
        assert_same(actual, expected)
    assert queue._closed and not queue._pending
    assert queue._executor.shutdown_called


@pytest.mark.parametrize("explicit", [False, True])
def test_default_coverage_remains_available_before_waiting(native, deferred_executor, explicit):
    packed = np.array([[0, 1, 0xFFFF], [0xAAAA, 0xFF, 0]], dtype=np.uint16)
    expected = direct_result(native, packed)
    lane, span, coverage = actual = buffers(packed)
    keywords = {"defer_coverage": False} if explicit else {}
    with renderer_module._FastLaneAnalysisQueue(native, async_threshold_seconds=0.0) as queue:
        queue.submit(packed[:1], lane, span, coverage[:1], **keywords)
        queue.submit(packed[1:], lane, span, coverage[1:], **keywords)
        assert coverage.tolist() == [[16, 15, 0], [8, 8, 16]]
        queue.finish_pending()
        assert_same(actual, expected)


@pytest.mark.parametrize("deferred", [(True, False, True, False), (False, True, False, True)])
def test_mixed_coverage_contracts_keep_bounded_ordered_native_work(
    native, deferred_executor, deferred
):
    rng = np.random.default_rng(20260831)
    packed = rng.integers(0, 65536, size=(8, 13), dtype=np.uint16)
    expected = direct_result(native, packed)
    lane, span, coverage = actual = buffers(packed)
    with renderer_module._FastLaneAnalysisQueue(native, async_threshold_seconds=0.0) as queue:
        for index, defer in enumerate(deferred):
            rows = slice(2 * index, 2 * index + 2)
            queue.submit(packed[rows], lane, span, coverage[rows], defer_coverage=defer)
            assert len(queue._pending) <= 2
            if not defer:
                np.testing.assert_array_equal(coverage[rows], expected[2][rows])
        queue.finish_pending()
        assert_same(actual, expected)
    with pytest.raises(RuntimeError, match="queue is closed"):
        queue.submit(packed, lane, span, coverage, defer_coverage=True)


@pytest.mark.parametrize(
    "duration,defer,override,worker_expected",
    [
        (0.0001, True, None, False),
        (0.000499, True, None, False),
        (0.0005, True, None, True),
        (0.00075, True, None, True),
        (0.00075, False, None, False),
        (0.002, False, None, True),
        (0.0025, False, None, True),
        (0.00075, True, float("inf"), False),
        (0.0001, True, 0.0, True),
        (0.0001, False, 0.0, True),
        (0.0015, True, 0.002, False),
    ],
)
def test_worker_creation_matches_measured_cost_and_coverage_contract(
    native, deferred_executor, monkeypatch, duration, defer, override, worker_expected
):
    clock = iter((0.0, duration))
    monkeypatch.setattr(renderer_module, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    packed = np.zeros((1, 3), dtype=np.uint16)
    lane, span, coverage = buffers(packed)
    options = {} if override is None else {"async_threshold_seconds": override}
    with renderer_module._FastLaneAnalysisQueue(native, **options) as queue:
        queue.submit(packed, lane, span, coverage, defer_coverage=defer)
        assert (queue._executor is not None) is worker_expected
        assert coverage.tolist() == [[16, 16, 16]]


def test_real_worker_keeps_coverage_private_until_completion(native):
    packed = np.array([[0, 1, 0xFFFF], [0xAAAA, 0xFF, 0]], dtype=np.uint16)
    expected = direct_result(native, packed)
    lane, span, coverage = actual = buffers(packed)
    entered = threading.Event()
    release = threading.Event()

    class PausedNative:
        calls = 0

        def accumulate_packed_lane_hole_run_histograms_into(self, *args):
            self.calls += 1
            if self.calls == 2:
                entered.set()
                assert release.wait(5), "native worker did not receive its release"
            native.accumulate_packed_lane_hole_run_histograms_into(*args)

    with renderer_module._FastLaneAnalysisQueue(
        PausedNative(), async_threshold_seconds=0.0
    ) as queue:
        try:
            queue.submit(packed[:1], lane, span, coverage[:1], defer_coverage=True)
            queue.submit(packed[1:], lane, span, coverage[1:], defer_coverage=True)
            assert entered.wait(5)
            assert coverage[1].tolist() == [255, 255, 255]
        finally:
            release.set()
        queue.finish_pending()
        assert_same(actual, expected)
    assert all(not thread.is_alive() for thread in queue._executor._threads)


@pytest.mark.parametrize("foreground_failure", [False, True])
def test_deferred_failure_is_drained_without_replacing_a_foreground_error(
    native, deferred_executor, foreground_failure
):
    packed = np.zeros((2, 3), dtype=np.uint16)
    lane, span, coverage = buffers(packed)
    failure = RuntimeError("injected native coverage failure")
    foreground = ValueError("injected foreground failure")

    class FailingNative:
        calls = 0

        def accumulate_packed_lane_hole_run_histograms_into(self, *args):
            self.calls += 1
            if self.calls > 1:
                raise failure
            native.accumulate_packed_lane_hole_run_histograms_into(*args)

    expected = foreground if foreground_failure else failure
    with pytest.raises(type(expected)) as captured:
        with renderer_module._FastLaneAnalysisQueue(
            FailingNative(), async_threshold_seconds=0.0
        ) as queue:
            queue.submit(packed[:1], lane, span, coverage[:1], defer_coverage=True)
            queue.submit(packed[1:], lane, span, coverage[1:], defer_coverage=True)
            if foreground_failure:
                raise foreground
    assert captured.value is expected
    assert queue._closed and not queue._pending and queue._executor.shutdown_called


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("fill", ["none", "background"])
@pytest.mark.parametrize("pattern", [(True,), (False,), (True, False), (False, True)])
def test_only_actual_fused_bands_defer_coverage_and_both_eyes_finish(
    native, deferred_executor, monkeypatch, fill, pattern
):
    if output_module.triton is None:
        pytest.skip("Triton output is unavailable")
    rng = np.random.default_rng(20260831)
    frame = rng.integers(0, 256, size=(7, 31, 3), dtype=np.uint8)
    canonical = rng.random((5, 17), dtype=np.float32)
    settings = renderer_module.StereoRenderSettings(
        stereo_strength=5.0, convergence=0.37, occlusion_fill=fill
    )
    budget = 31 * renderer_module.TRITON_SPLAT_BYTES_PER_PIXEL * 2
    expected = renderer_module.StereoRenderer(
        device="cpu", temporary_budget_bytes=budget
    ).render_relative_fast_compact(frame, canonical, settings)
    original_queue = renderer_module._FastLaneAnalysisQueue
    original_submit = original_queue.submit
    original_payload = output_module.triton_fast_band_payload
    fused_bands, deferred_bands, queues = [], [], []

    def make_queue(extension):
        queue = original_queue(extension, async_threshold_seconds=0.0)
        queues.append(queue)
        return queue

    def observe_submit(self, *args, **kwargs):
        deferred_bands.append(kwargs.get("defer_coverage", False))
        return original_submit(self, *args, **kwargs)

    def mixed_payload(*args, **kwargs):
        fused = pattern[len(fused_bands) % len(pattern)]
        fused_bands.append(fused)
        if not fused:
            return None
        result = original_payload(*args, **kwargs)
        assert result is not None
        return result

    monkeypatch.setattr(renderer_module, "_FastLaneAnalysisQueue", make_queue)
    monkeypatch.setattr(original_queue, "submit", observe_submit)
    monkeypatch.setattr(output_module, "triton_fast_band_payload", mixed_payload)
    actual = renderer_module.StereoRenderer(
        device="cuda", temporary_budget_bytes=budget
    ).render_relative_fast_compact(frame, canonical, settings)
    assert len(fused_bands) == 8 and deferred_bands == fused_bands
    fields = dict(
        frame_name="frame_000001", geometry_mode="relative", occlusion_fill=fill, render_mode="fast"
    )
    assert canonical_json_bytes(
        build_quality_frame_stats(actual, **fields)
    ) == canonical_json_bytes(build_quality_frame_stats(expected, **fields))
    for name in (
        "left_image",
        "right_image",
        "left_coverage_count",
        "right_coverage_count",
        "left_repair_bits",
        "right_repair_bits",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert len(queues) == 1 and queues[0]._closed and not queues[0]._pending


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("retry_fails", [False, True])
def test_late_oom_drains_deferred_coverage_before_retry_or_failure(
    native, deferred_executor, monkeypatch, retry_fails
):
    if output_module.triton is None:
        pytest.skip("Triton output is unavailable")
    rng = np.random.default_rng(20260831)
    frame = rng.integers(0, 256, size=(7, 31, 3), dtype=np.uint8)
    canonical = rng.random((5, 17), dtype=np.float32)
    settings = renderer_module.StereoRenderSettings(stereo_strength=5.0)
    budget = 31 * renderer_module.TRITON_SPLAT_BYTES_PER_PIXEL * 2
    expected = renderer_module.StereoRenderer(
        device="cpu", temporary_budget_bytes=budget
    ).render_relative_fast_compact(frame, canonical, settings)
    original_queue = renderer_module._FastLaneAnalysisQueue
    original_download = renderer_module._download_fast_cuda_payload
    original_release = renderer_module.StereoRenderer._release_after_oom
    queues, row_counts, released = [], [], []

    def make_queue(extension):
        queue = original_queue(extension, async_threshold_seconds=0.0)
        queues.append(queue)
        return queue

    def fail_with_pending_work(payload, image_shape, *args, **kwargs):
        row_counts.append(image_shape[0])
        if len(row_counts) == 3 or (retry_fails and len(row_counts) == 6):
            assert len(queues[-1]._pending) == 1
            future = queues[-1]._pending[0]
            assert not future.done() and isinstance(future.args[3], np.ndarray)
            raise torch.cuda.OutOfMemoryError("injected OOM after queued coverage")
        return original_download(payload, image_shape, *args, **kwargs)

    def release_after_join(self):
        queue = queues[-1]
        assert queue._closed and not queue._pending and queue._executor.shutdown_called
        released.append(queue)
        return original_release(self)

    monkeypatch.setattr(renderer_module, "_FastLaneAnalysisQueue", make_queue)
    monkeypatch.setattr(renderer_module, "_download_fast_cuda_payload", fail_with_pending_work)
    monkeypatch.setattr(renderer_module.StereoRenderer, "_release_after_oom", release_after_join)
    renderer = renderer_module.StereoRenderer(device="cuda", temporary_budget_bytes=budget)
    if retry_fails:
        with pytest.raises(RuntimeError, match="attempted band heights 2 and 1") as captured:
            renderer.render_relative_fast_compact(frame, canonical, settings)
        assert isinstance(captured.value.__cause__, torch.cuda.OutOfMemoryError)
        assert row_counts == [2, 2, 2, 1, 1, 1] and len(released) == 2
    else:
        actual = renderer.render_relative_fast_compact(frame, canonical, settings)
        assert row_counts == [2, 2, 2] + [1] * 14 and len(released) == 1
        for name in (
            "left_image",
            "right_image",
            "left_coverage_count",
            "right_coverage_count",
            "left_repair_bits",
            "right_repair_bits",
        ):
            np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
        fields = dict(
            frame_name="frame_000001",
            geometry_mode="relative",
            occlusion_fill="background",
            render_mode="fast",
        )
        assert canonical_json_bytes(
            build_quality_frame_stats(actual, **fields)
        ) == canonical_json_bytes(build_quality_frame_stats(expected, **fields))
    assert len(queues) == 2
    assert all(queue._closed and not queue._pending for queue in queues)
    assert all(queue._executor.shutdown_called for queue in queues)
