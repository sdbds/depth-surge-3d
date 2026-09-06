"""Bounded source prefetch must not change raw-depth ownership or payloads."""

import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.inference.depth.types import (
    DepthBatch,
    DepthRepresentation,
    PinholeCameraBatch,
)
from src.depth_surge_3d.processing.frames import depth_processor
from src.depth_surge_3d.processing.frames.depth_processor import DepthMapProcessor
from src.depth_surge_3d.processing.frames.depth_storage import RawDepthStore


class InputStampedEstimator:
    exact_input_batch_reuse = False

    def __init__(self, representation=DepthRepresentation.INVERSE_DEPTH):
        self.representation = representation
        self.calls = []
        self.on_inference = None

    def estimate_depth_batch(self, frames, **kwargs):
        self.calls.append((frames.copy(), kwargs))
        if self.on_inference is not None:
            self.on_inference(len(self.calls))
        values = frames[:, 0, 0, 0].astype(np.float32)[:, None, None]
        values = values + np.arange(1, 7, dtype=np.float32).reshape(1, 2, 3)
        if len(self.calls) == 2:
            values *= np.float32(70000)
        camera = (
            PinholeCameraBatch(np.full(len(frames), 0.8, dtype=np.float32))
            if self.representation is DepthRepresentation.METRIC_DEPTH
            else None
        )
        return DepthBatch(values, self.representation, camera=camera)


@pytest.fixture
def raw_case(tmp_path, monkeypatch):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_files = []
    for index in range(5):
        path = frames_dir / f"frame_{index:04d}.png"
        assert cv2.imwrite(str(path), np.full((8, 6, 3), index * 10, np.uint8))
        frame_files.append(path)

    def run(estimator, name="raw", raw_store=None):
        processor = DepthMapProcessor(estimator)
        monkeypatch.setattr(
            processor,
            "_prepare_raw_depth_stage",
            lambda *args: (2, 8, 30, "auto", (2, 3), 2),
        )
        camera_model = (
            "pinhole_fx" if estimator.representation is DepthRepresentation.METRIC_DEPTH else "none"
        )
        return processor._complete_raw_depth_stage(
            frame_files,
            {"keep_intermediates": False},
            tmp_path / name,
            tmp_path / "metric",
            {"camera_model": camera_model},
            {},
            raw_store,
            None,
            (6, 8),
        )

    return frame_files, run


def _prefetch_threads():
    return [thread for thread in threading.enumerate() if thread.name.startswith("depth-prefetch")]


def test_only_next_batch_decodes_while_inference_and_writes_stay_on_caller(raw_case, monkeypatch):
    _, run = raw_case
    caller = threading.get_ident()
    loaded_second = threading.Event()
    loaded = []
    writes = []
    original_load = DepthMapProcessor._load_chunk_frames
    original_write = RawDepthStore.write_batch

    def load(self, files, settings):
        result = original_load(self, files, settings)
        loaded.append((files[0].name, threading.get_ident()))
        if files[0].name == "frame_0002.png":
            loaded_second.set()
        return result

    def infer(call):
        assert threading.get_ident() == caller
        if call == 1:
            assert loaded_second.wait(5), "next batch was not prefetched during inference"
            assert [name for name, _ in loaded] == ["frame_0000.png", "frame_0002.png"]

    def write(self, names, batch):
        assert threading.get_ident() == caller
        writes.append(list(names))
        if len(writes) == 1:
            assert [name for name, _ in loaded] == ["frame_0000.png", "frame_0002.png"]
        return original_write(self, names, batch)

    estimator = InputStampedEstimator()
    estimator.on_inference = infer
    monkeypatch.setattr(DepthMapProcessor, "_load_chunk_frames", load)
    monkeypatch.setattr(RawDepthStore, "write_batch", write)
    store = run(estimator)

    assert loaded[0][1] == caller
    assert all(worker != caller for _, worker in loaded[1:])
    assert writes == [
        ["frame_0000.png", "frame_0001.png"],
        ["frame_0002.png", "frame_0003.png"],
        ["frame_0004.png"],
    ]
    assert store.metadata["completed_count"] == 5
    assert not _prefetch_threads()


@pytest.mark.parametrize("representation", list(DepthRepresentation))
def test_prefetch_preserves_payloads_camera_contract_promotion_and_short_tail(
    raw_case, monkeypatch, representation
):
    files, run = raw_case
    monkeypatch.setattr(depth_processor, "RAW_DEPTH_PREFETCH_BUDGET_BYTES", 0, raising=False)
    serial_estimator = InputStampedEstimator(representation)
    serial = run(serial_estimator, "serial")
    monkeypatch.setattr(depth_processor, "RAW_DEPTH_PREFETCH_BUDGET_BYTES", 1024 * 1024)
    prefetched_estimator = InputStampedEstimator(representation)
    prefetched = run(prefetched_estimator, "prefetched")

    assert serial.metadata == prefetched.metadata
    assert prefetched.metadata["selected_dtype"] == "float32"
    for path in files:
        assert (
            serial.path_for(path.name).read_bytes() == prefetched.path_for(path.name).read_bytes()
        )
    for (old_frames, old_kwargs), (new_frames, new_kwargs) in zip(
        serial_estimator.calls, prefetched_estimator.calls, strict=True
    ):
        np.testing.assert_array_equal(old_frames, new_frames, strict=True)
        assert old_kwargs == new_kwargs
    assert [len(frames) for frames, _ in prefetched_estimator.calls] == [2, 2, 1]


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_primary_inference_failure_wins_over_prefetch_failure_and_worker_is_joined(
    raw_case, monkeypatch, failure
):
    files, run = raw_case
    second_finished = threading.Event()
    original_load = DepthMapProcessor._load_chunk_frames

    def load(self, paths, settings):
        if paths[0] == files[2]:
            second_finished.set()
            raise OSError("later decode failure")
        return original_load(self, paths, settings)

    def infer(_call):
        assert second_finished.wait(5), "next batch was not prefetched during inference"
        raise failure("primary inference failure")

    estimator = InputStampedEstimator()
    estimator.on_inference = infer
    monkeypatch.setattr(DepthMapProcessor, "_load_chunk_frames", load)
    with pytest.raises(failure, match="primary inference failure"):
        run(estimator)
    assert not _prefetch_threads()


def test_later_decode_failure_keeps_prior_raw_writes_and_joins_worker(raw_case, monkeypatch):
    files, run = raw_case
    original_load = DepthMapProcessor._load_chunk_frames

    def load(self, paths, settings):
        if paths[0] == files[2]:
            raise OSError("source read failed")
        return original_load(self, paths, settings)

    monkeypatch.setattr(DepthMapProcessor, "_load_chunk_frames", load)
    with pytest.raises(OSError, match="source read failed"):
        run(InputStampedEstimator())
    raw_dir = files[0].parent.parent / "raw"
    assert (raw_dir / "frame_0000.npz").is_file()
    assert (raw_dir / "frame_0001.npz").is_file()
    assert not (raw_dir / "frame_0002.npz").exists()
    assert not _prefetch_threads()


@pytest.mark.parametrize("stage", ["inference", "write"])
@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_failure_waits_for_a_still_running_decoder(  # noqa: C901
    raw_case, monkeypatch, stage, failure
):
    files, run = raw_case
    started = threading.Event()
    release = threading.Event()
    stopped = threading.Event()
    joining = threading.Event()
    finished = threading.Event()
    failures = []
    original_load = DepthMapProcessor._load_chunk_frames

    class ObservedExecutor(ThreadPoolExecutor):
        def shutdown(self, wait=True, *, cancel_futures=False):
            joining.set()
            return super().shutdown(wait=wait, cancel_futures=cancel_futures)

    def load(self, paths, settings):
        if paths[0] == files[2]:
            started.set()
            try:
                assert release.wait(10)
                return original_load(self, paths, settings)
            finally:
                stopped.set()
        return original_load(self, paths, settings)

    def infer(_call):
        assert started.wait(5)
        if stage == "inference":
            raise failure("primary failure")

    def write(*_args):
        raise failure("primary failure")

    def process():
        try:
            run(estimator)
        except BaseException as error:
            failures.append(error)
            assert stopped.is_set()
        finally:
            finished.set()

    estimator = InputStampedEstimator()
    estimator.on_inference = infer
    monkeypatch.setattr(depth_processor, "ThreadPoolExecutor", ObservedExecutor)
    monkeypatch.setattr(DepthMapProcessor, "_load_chunk_frames", load)
    if stage == "write":
        monkeypatch.setattr(RawDepthStore, "write_batch", write)
    pipeline = threading.Thread(target=process)
    pipeline.start()
    try:
        assert joining.wait(5)
        assert not finished.is_set()
    finally:
        release.set()
        pipeline.join(10)
    assert not pipeline.is_alive()
    assert len(failures) == 1
    assert type(failures[0]) is failure
    assert str(failures[0]) == "primary failure"
    assert not _prefetch_threads()


@pytest.mark.parametrize("budget", [0, 2 * 2 * 8 * 6 * 3 - 1])
def test_batches_exceeding_decode_and_stack_budget_load_serially(raw_case, monkeypatch, budget):
    _, run = raw_case
    caller = threading.get_ident()
    original_load = DepthMapProcessor._load_chunk_frames
    loaded = []

    def load(self, files, settings):
        loaded.append((len(files), threading.get_ident()))
        return original_load(self, files, settings)

    monkeypatch.setattr(depth_processor, "RAW_DEPTH_PREFETCH_BUDGET_BYTES", budget, raising=False)
    monkeypatch.setattr(DepthMapProcessor, "_load_chunk_frames", load)
    run(InputStampedEstimator())
    assert all(worker == caller for count, worker in loaded if count == 2)
    if budget == 0:
        assert all(worker == caller for _, worker in loaded)
    assert not _prefetch_threads()


def test_unknown_png_header_falls_back_without_changing_decode_behavior(raw_case, monkeypatch):
    _, run = raw_case
    caller = threading.get_ident()
    original_load = DepthMapProcessor._load_chunk_frames
    loaded_threads = []

    def load(self, files, settings):
        loaded_threads.append(threading.get_ident())
        return original_load(self, files, settings)

    monkeypatch.setattr(depth_processor, "read_png_header", lambda _path: None, raising=False)
    monkeypatch.setattr(DepthMapProcessor, "_load_chunk_frames", load)
    run(InputStampedEstimator())
    assert loaded_threads == [caller] * 3


def test_prefetch_memory_pressure_retries_on_caller_and_disables_further_prefetch(
    raw_case, monkeypatch
):
    _, run = raw_case
    caller = threading.get_ident()
    original_load = DepthMapProcessor._load_chunk_frames
    failed = threading.Event()
    loads_after_failure = []
    failed_buffers = []

    def load(self, files, settings):
        worker = threading.get_ident()
        if worker != caller:
            partial_batch = np.zeros((2, 8, 6, 3), dtype=np.uint8)
            failed_buffers.append(weakref.ref(partial_batch))
            failed.set()
            raise MemoryError("background allocation failed")
        if failed.is_set():
            assert all(buffer() is None for buffer in failed_buffers)
            loads_after_failure.append(worker)
        return original_load(self, files, settings)

    monkeypatch.setattr(DepthMapProcessor, "_load_chunk_frames", load)
    store = run(InputStampedEstimator())
    assert failed.is_set(), "test did not exercise background memory pressure"
    assert loads_after_failure == [caller, caller]
    assert store.metadata["completed_count"] == 5
    assert not _prefetch_threads()
