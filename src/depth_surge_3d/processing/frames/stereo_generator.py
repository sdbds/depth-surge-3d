"""Bounded stereo generation over canonical disparity maps."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from ...core.constants import PREVIEW_FRAME_SAMPLE_RATE
from ...rendering.stereo_renderer import (
    StereoRenderer,
    StereoRenderSettings,
)
from .depth_storage import canonical_json_hash


CANONICAL_DEPTH_SCHEMA_VERSION = 1
CANONICAL_DEPTH_ALGORITHM_VERSION = "scene-percentile-v1"
CANONICAL_METADATA_REQUIRED_FIELDS = {
    "source_raw_fingerprint",
    "source_model_fingerprint",
    "scene_manifest_fingerprint",
    "depth_bounds_fingerprint",
    "native_shape",
}
STEREO_HOST_BUDGET = 512 * 1024 * 1024
HOST_STEREO_BYTES_PER_PIXEL = 16
HOST_SLOT_OVERHEAD = 1024 * 1024
MIN_STEREO_IO_WORKERS = 1
MAX_STEREO_IO_WORKERS = 16


def validate_stereo_io_workers(workers: int) -> int:
    """Validate the bounded decode/write worker count."""

    if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)):
        raise ValueError("stereo_io_workers must be an integer in 1..16")
    value = int(workers)
    if not MIN_STEREO_IO_WORKERS <= value <= MAX_STEREO_IO_WORKERS:
        raise ValueError("stereo_io_workers must be an integer in 1..16")
    return value


def _default_stereo_io_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return min(4, max(1, cpu_count - 2))


def calculate_stereo_pipeline_capacity(
    render_width: int,
    render_height: int,
    stereo_io_workers: int,
) -> int:
    """Derive queue and lifecycle-permit capacity from the fixed host budget."""

    workers = validate_stereo_io_workers(stereo_io_workers)
    if render_width <= 0 or render_height <= 0:
        raise ValueError("Render width and height must be positive")
    slot_bytes = render_width * render_height * HOST_STEREO_BYTES_PER_PIXEL + HOST_SLOT_OVERHEAD
    memory_slots = STEREO_HOST_BUDGET // slot_bytes
    capacity = min(2 * workers, memory_slots)
    if capacity < 1:
        raise MemoryError(
            f"Stereo frame {render_width}x{render_height} cannot fit one lifecycle "
            f"slot: required {slot_bytes} host bytes; budget is "
            f"{STEREO_HOST_BUDGET} bytes"
        )
    return capacity


@dataclass(frozen=True)
class StereoPipelineStats:
    """Observable host-memory and I/O backpressure accounting."""

    permit_count: int
    queue_capacity: int
    permits_acquired: int
    permits_released: int
    active_permits: int
    max_active_permits: int
    permit_wait_seconds: float
    queue_wait_seconds: float
    decoded_frames: int
    rendered_frames: int
    written_frames: int


class _PipelineMetrics:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.lock = threading.Lock()
        self.permits_acquired = 0
        self.permits_released = 0
        self.active_permits = 0
        self.max_active_permits = 0
        self.permit_wait_seconds = 0.0
        self.queue_wait_seconds = 0.0
        self.decoded_frames = 0
        self.rendered_frames = 0
        self.written_frames = 0

    def permit_acquired(self, waited: float) -> None:
        with self.lock:
            self.permits_acquired += 1
            self.active_permits += 1
            self.max_active_permits = max(
                self.max_active_permits,
                self.active_permits,
            )
            self.permit_wait_seconds += waited

    def permit_released(self) -> None:
        with self.lock:
            self.permits_released += 1
            self.active_permits -= 1

    def queue_waited(self, waited: float) -> None:
        with self.lock:
            self.queue_wait_seconds += waited

    def increment(self, field: str) -> None:
        with self.lock:
            setattr(self, field, getattr(self, field) + 1)

    def snapshot(self) -> StereoPipelineStats:
        with self.lock:
            return StereoPipelineStats(
                permit_count=self.capacity,
                queue_capacity=self.capacity,
                permits_acquired=self.permits_acquired,
                permits_released=self.permits_released,
                active_permits=self.active_permits,
                max_active_permits=self.max_active_permits,
                permit_wait_seconds=self.permit_wait_seconds,
                queue_wait_seconds=self.queue_wait_seconds,
                decoded_frames=self.decoded_frames,
                rendered_frames=self.rendered_frames,
                written_frames=self.written_frames,
            )


class _LifecyclePermits:
    def __init__(self, capacity: int, metrics: _PipelineMetrics) -> None:
        self._semaphore = threading.BoundedSemaphore(capacity)
        self._metrics = metrics

    def acquire(self) -> None:
        started = time.perf_counter()
        self._semaphore.acquire()
        self._metrics.permit_acquired(time.perf_counter() - started)

    def release(self) -> None:
        self._semaphore.release()
        self._metrics.permit_released()


@dataclass(frozen=True)
class _FileWorkItem:
    index: int
    frame_path: Path
    depth_path: Path
    frame_name: str
    left_path: Path
    right_path: Path


@dataclass(frozen=True)
class _DecodedMessage:
    work: _FileWorkItem
    frame: np.ndarray | None = None
    canonical: np.ndarray | None = None
    error: Exception | None = None


@dataclass(frozen=True)
class _WriteItem:
    work: _FileWorkItem
    left_image: np.ndarray
    right_image: np.ndarray


@dataclass(frozen=True)
class _WriteCompletion:
    work: _FileWorkItem
    error: Exception | None = None


_QUEUE_SENTINEL = object()


def _timed_put(
    target: queue.Queue[object],
    item: object,
    metrics: _PipelineMetrics,
) -> None:
    started = time.perf_counter()
    target.put(item)
    metrics.queue_waited(time.perf_counter() - started)


def _atomic_write_png(path: Path, image: np.ndarray) -> None:
    """Encode a PNG and atomically replace its destination from the same directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded, payload = cv2.imencode(".png", image)
    if not encoded:
        raise OSError(f"Could not encode PNG: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload.tobytes())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_pair(item: _WriteItem) -> None:
    try:
        _atomic_write_png(item.work.left_path, item.left_image)
        _atomic_write_png(item.work.right_path, item.right_image)
    except Exception:
        item.work.left_path.unlink(missing_ok=True)
        item.work.right_path.unlink(missing_ok=True)
        raise


def _decode_work_item(
    work: _FileWorkItem,
    *,
    encoding_scale: float,
    render_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    frame = cv2.imread(str(work.frame_path), cv2.IMREAD_COLOR)
    encoded = cv2.imread(str(work.depth_path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise OSError(f"Could not load frame: {work.frame_path}")
    if encoded is None:
        raise OSError(f"Could not load canonical disparity: {work.depth_path}")
    if frame.shape[:2] != render_shape:
        raise ValueError(
            f"Frame shape changed at {work.frame_path}: "
            f"expected {render_shape}, got {frame.shape[:2]}"
        )
    if encoded.dtype != np.uint16 or encoded.ndim != 2:
        raise TypeError(f"Canonical disparity must be uint16: {work.depth_path}")
    canonical = encoded.astype(np.float32)
    canonical *= np.float32(1.0 / encoding_scale)
    return frame, canonical


class _StereoFilePipeline:
    """Own the bounded queues, lifecycle permits, and I/O worker threads."""

    def __init__(
        self,
        *,
        renderer: StereoRenderer,
        work_items: list[_FileWorkItem],
        encoding_scale: float,
        render_shape: tuple[int, int],
        render_settings: StereoRenderSettings,
        workers: int,
        capacity: int,
        completed: int,
        total_frames: int,
        report_progress: Callable[..., None],
    ) -> None:
        self.renderer = renderer
        self.work_items = work_items
        self.encoding_scale = encoding_scale
        self.render_shape = render_shape
        self.render_settings = render_settings
        self.workers = workers
        self.completed = completed
        self.total_frames = total_frames
        self.report_progress = report_progress
        self.metrics = _PipelineMetrics(capacity)
        self.permits = _LifecyclePermits(capacity, self.metrics)
        self.decode_requests: queue.Queue[object] = queue.Queue(maxsize=capacity)
        self.decoded_frames: queue.Queue[object] = queue.Queue(maxsize=capacity)
        self.write_requests: queue.Queue[object] = queue.Queue(maxsize=capacity)
        self.write_completions: queue.Queue[object] = queue.Queue(maxsize=capacity)
        self.first_error: tuple[str, _FileWorkItem, Exception] | None = None
        self.writes_enqueued = 0
        self.completions_seen = 0
        self.successful_completions = 0
        self.feeder = threading.Thread(
            target=self._feed_decoder,
            name="stereo-decode-feeder",
            daemon=True,
        )
        self.decoder_threads = [
            threading.Thread(
                target=self._decode_worker,
                name=f"stereo-decode-{index}",
                daemon=True,
            )
            for index in range(workers)
        ]
        self.writer_threads = [
            threading.Thread(
                target=self._write_worker,
                name=f"stereo-write-{index}",
                daemon=True,
            )
            for index in range(workers)
        ]

    def run(self) -> StereoPipelineStats:
        for thread in self.decoder_threads + self.writer_threads:
            thread.start()
        self.feeder.start()

        for _ in self.work_items:
            message = self.decoded_frames.get()
            try:
                assert isinstance(message, _DecodedMessage)
                write_item = self._render_decoded(message)
                del message
                if write_item is not None:
                    _timed_put(self.write_requests, write_item, self.metrics)
                    self.writes_enqueued += 1
                self._drain_ready_completions()
            finally:
                self.decoded_frames.task_done()

        self._wait_for_writes()
        for _ in range(self.workers):
            _timed_put(self.write_requests, _QUEUE_SENTINEL, self.metrics)
        self.feeder.join()
        for thread in self.decoder_threads + self.writer_threads:
            thread.join()

        stats = self.metrics.snapshot()
        if self.first_error is not None:
            stage, work, error = self.first_error
            raise RuntimeError(f"Stereo {stage} failed for {work.frame_name}: {error}") from error
        return stats

    def _feed_decoder(self) -> None:
        for work in self.work_items:
            _timed_put(self.decode_requests, work, self.metrics)
        for _ in range(self.workers):
            _timed_put(self.decode_requests, _QUEUE_SENTINEL, self.metrics)

    def _decode_worker(self) -> None:
        while True:
            work = self.decode_requests.get()
            try:
                if work is _QUEUE_SENTINEL:
                    return
                assert isinstance(work, _FileWorkItem)
                self.permits.acquire()
                message = self._decode(work)
                _timed_put(self.decoded_frames, message, self.metrics)
            finally:
                self.decode_requests.task_done()

    def _decode(self, work: _FileWorkItem) -> _DecodedMessage:
        try:
            frame, canonical = _decode_work_item(
                work,
                encoding_scale=self.encoding_scale,
                render_shape=self.render_shape,
            )
        except Exception as error:
            self.permits.release()
            return _DecodedMessage(work=work, error=error)
        self.metrics.increment("decoded_frames")
        return _DecodedMessage(
            work=work,
            frame=frame,
            canonical=canonical,
        )

    def _render_decoded(self, message: _DecodedMessage) -> _WriteItem | None:
        if message.error is not None:
            self._record_error("decode", message.work, message.error)
            return None
        if self.first_error is not None:
            self.permits.release()
            return None
        assert message.frame is not None and message.canonical is not None
        try:
            result = self.renderer.render(
                message.frame,
                message.canonical,
                self.render_settings,
            )
        except Exception as error:
            self.permits.release()
            self._record_error("render", message.work, error)
            return None
        self.metrics.increment("rendered_frames")
        return _WriteItem(
            work=message.work,
            left_image=result.left_image,
            right_image=result.right_image,
        )

    def _write_worker(self) -> None:
        while True:
            item = self.write_requests.get()
            try:
                if item is _QUEUE_SENTINEL:
                    return
                assert isinstance(item, _WriteItem)
                completion = self._write(item)
                _timed_put(self.write_completions, completion, self.metrics)
            finally:
                self.write_requests.task_done()

    def _write(self, item: _WriteItem) -> _WriteCompletion:
        error = None
        try:
            _write_pair(item)
        except Exception as write_error:
            error = write_error
        else:
            self.metrics.increment("written_frames")
        finally:
            self.permits.release()
        return _WriteCompletion(work=item.work, error=error)

    def _drain_ready_completions(self) -> None:
        while True:
            try:
                completion = self.write_completions.get_nowait()
            except queue.Empty:
                return
            try:
                assert isinstance(completion, _WriteCompletion)
                self._handle_completion(completion)
            finally:
                self.write_completions.task_done()

    def _wait_for_writes(self) -> None:
        while self.completions_seen < self.writes_enqueued:
            completion = self.write_completions.get()
            try:
                assert isinstance(completion, _WriteCompletion)
                self._handle_completion(completion)
            finally:
                self.write_completions.task_done()

    def _handle_completion(self, completion: _WriteCompletion) -> None:
        self.completions_seen += 1
        if completion.error is not None:
            self._record_error("write", completion.work, completion.error)
            return
        self.successful_completions += 1
        try:
            self.report_progress(
                processed=self.completed + self.successful_completions,
                total=self.total_frames,
                work_index=completion.work.index,
                left_path=completion.work.left_path,
            )
        except Exception as error:
            self._record_error("progress", completion.work, error)

    def _record_error(
        self,
        stage: str,
        work: _FileWorkItem,
        error: Exception,
    ) -> None:
        if self.first_error is None:
            self.first_error = (stage, work, error)


class StereoPairGenerator:
    """Generate stereo pairs with main-thread rendering and bounded threaded I/O."""

    def __init__(
        self,
        verbose: bool = False,
        *,
        renderer: StereoRenderer | None = None,
    ) -> None:
        self.verbose = verbose
        self.renderer = renderer if renderer is not None else StereoRenderer()
        self.last_pipeline_stats: StereoPipelineStats | None = None

    @staticmethod
    def _render_settings(settings: dict[str, Any]) -> StereoRenderSettings:
        return StereoRenderSettings(
            stereo_strength=float(settings.get("stereo_strength", 2.0)),
            convergence=float(settings.get("convergence", 0.5)),
            occlusion_fill=str(settings.get("occlusion_fill", "background")),
        )

    def create_stereo_pairs(
        self,
        frames: np.ndarray,
        depth_maps: np.ndarray,
        frame_files: list[Path],
        directories: dict[str, Path],
        settings: dict[str, Any],
        progress_tracker=None,
    ) -> bool:
        """Render a caller-owned in-memory batch without spawning CUDA processes."""

        try:
            if not (len(frames) == len(depth_maps) == len(frame_files)):
                raise ValueError("Frame, canonical disparity, and name counts must match")
            render_settings = self._render_settings(settings)
            keep = bool(settings.get("keep_intermediates", False))
            for index, (frame, canonical, frame_file) in enumerate(
                zip(frames, depth_maps, frame_files)
            ):
                result = self.renderer.render(frame, canonical, render_settings)
                left_path = directories.get("left_frames", Path()) / f"{frame_file.stem}.png"
                right_path = directories.get("right_frames", Path()) / f"{frame_file.stem}.png"
                if keep:
                    _write_pair(
                        _WriteItem(
                            work=_FileWorkItem(
                                index=index,
                                frame_path=frame_file,
                                depth_path=frame_file,
                                frame_name=frame_file.stem,
                                left_path=left_path,
                                right_path=right_path,
                            ),
                            left_image=result.left_image,
                            right_image=result.right_image,
                        )
                    )
                self._report_progress(
                    progress_tracker,
                    processed=index + 1,
                    total=len(frame_files),
                    work_index=index,
                    left_path=left_path if keep else None,
                )
            return True
        except Exception as error:
            print(f"Error creating stereo pairs: {error}")
            traceback.print_exc()
            return False

    def create_stereo_pairs_from_files(
        self,
        frame_files: list[Path],
        depth_files: list[Path],
        directories: dict[str, Path],
        settings: dict[str, Any],
        progress_tracker=None,
    ) -> bool:
        """Decode, render, and atomically write file-backed stereo pairs."""

        self.last_pipeline_stats = None
        if len(frame_files) != len(depth_files):
            print(
                f"Error: Frame/depth count mismatch: {len(frame_files)} frames, "
                f"{len(depth_files)} depth maps"
            )
            return False
        try:
            left_dir = directories["left_frames"]
            right_dir = directories["right_frames"]
            left_dir.mkdir(parents=True, exist_ok=True)
            right_dir.mkdir(parents=True, exist_ok=True)
            metadata = self._get_canonical_metadata(depth_files, frame_files)
            work_items, completed = self._build_file_work_items(
                frame_files,
                depth_files,
                left_dir,
                right_dir,
            )
            if not work_items:
                print(f"  Reusing {completed} existing stereo pairs")
                return True

            render_shape = self._read_render_shape(frame_files[0])
            workers = validate_stereo_io_workers(
                settings.get("stereo_io_workers", _default_stereo_io_workers())
            )
            capacity = calculate_stereo_pipeline_capacity(
                render_shape[1],
                render_shape[0],
                workers,
            )
            render_settings = self._render_settings(settings)
            print(
                f"  Using {workers} stereo I/O workers with " f"{capacity} bounded frame slots..."
            )
            self._run_file_pipeline(
                work_items,
                encoding_scale=float(metadata["encoding_scale"]),
                render_shape=render_shape,
                render_settings=render_settings,
                workers=workers,
                capacity=capacity,
                completed=completed,
                total_frames=len(frame_files),
                progress_tracker=progress_tracker,
            )
            return True
        except Exception as error:
            print(f"Error creating stereo pairs: {error}")
            traceback.print_exc()
            return False

    def _run_file_pipeline(
        self,
        work_items: list[_FileWorkItem],
        *,
        encoding_scale: float,
        render_shape: tuple[int, int],
        render_settings: StereoRenderSettings,
        workers: int,
        capacity: int,
        completed: int,
        total_frames: int,
        progress_tracker,
    ) -> None:
        pipeline = _StereoFilePipeline(
            renderer=self.renderer,
            work_items=work_items,
            encoding_scale=encoding_scale,
            render_shape=render_shape,
            render_settings=render_settings,
            workers=workers,
            capacity=capacity,
            completed=completed,
            total_frames=total_frames,
            report_progress=lambda **values: self._report_progress(
                progress_tracker,
                **values,
            ),
        )
        try:
            self.last_pipeline_stats = pipeline.run()
        finally:
            self.last_pipeline_stats = pipeline.metrics.snapshot()

    @staticmethod
    def _report_progress(
        progress_tracker,
        *,
        processed: int,
        total: int,
        work_index: int,
        left_path: Path | None,
    ) -> None:
        if progress_tracker and (processed % 5 == 0 or processed == total):
            progress_tracker.update_progress(
                "Creating stereo pairs",
                phase="stereo_generation",
                frame_num=processed,
                step_name="Stereo Pair Creation",
                step_progress=processed,
                step_total=total,
            )
        if (
            progress_tracker
            and left_path is not None
            and hasattr(progress_tracker, "send_preview_frame")
            and (work_index % PREVIEW_FRAME_SAMPLE_RATE == 0 or processed == total)
        ):
            progress_tracker.send_preview_frame(left_path, "stereo_left", processed)

    @staticmethod
    def _read_render_shape(frame_path: Path) -> tuple[int, int]:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise OSError(f"Could not load frame: {frame_path}")
        return int(frame.shape[0]), int(frame.shape[1])

    @staticmethod
    def _build_file_work_items(
        frame_files: list[Path],
        depth_files: list[Path],
        left_dir: Path,
        right_dir: Path,
    ) -> tuple[list[_FileWorkItem], int]:
        work_items: list[_FileWorkItem] = []
        completed = 0
        for index, (frame_file, depth_file) in enumerate(zip(frame_files, depth_files)):
            frame_name = frame_file.stem
            left_path = left_dir / f"{frame_name}.png"
            right_path = right_dir / f"{frame_name}.png"
            if left_path.is_file() and right_path.is_file():
                completed += 1
                continue
            left_path.unlink(missing_ok=True)
            right_path.unlink(missing_ok=True)
            work_items.append(
                _FileWorkItem(
                    index=index,
                    frame_path=frame_file,
                    depth_path=depth_file,
                    frame_name=frame_name,
                    left_path=left_path,
                    right_path=right_path,
                )
            )
        return work_items, completed

    @staticmethod
    def _get_canonical_metadata(
        depth_files: list[Path],
        frame_files: list[Path],
    ) -> dict[str, Any]:
        """Load and validate the required local canonical disparity contract."""

        if not depth_files:
            raise ValueError("Canonical disparity files are required")
        metadata_file = depth_files[0].parent / "metadata.json"
        if not metadata_file.is_file():
            raise ValueError(f"Canonical disparity metadata is missing: {metadata_file}")
        try:
            metadata = json.loads(metadata_file.read_text())
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Canonical disparity metadata is invalid: {metadata_file}") from error

        fingerprint = metadata.get("fingerprint")
        unhashed = {key: value for key, value in metadata.items() if key != "fingerprint"}
        valid = (
            CANONICAL_METADATA_REQUIRED_FIELDS.issubset(metadata)
            and metadata.get("schema_version") == CANONICAL_DEPTH_SCHEMA_VERSION
            and metadata.get("algorithm_version") == CANONICAL_DEPTH_ALGORITHM_VERSION
            and metadata.get("representation") == "relative_disparity"
            and metadata.get("near_value") == 1.0
            and metadata.get("far_value") == 0.0
            and metadata.get("encoding") == "uint16_png"
            and metadata.get("encoding_scale") == 65535.0
            and metadata.get("num_frames") == len(depth_files)
            and metadata.get("frame_names") == [path.name for path in frame_files]
            and isinstance(fingerprint, str)
            and fingerprint == canonical_json_hash(unhashed)
        )
        if not valid:
            raise ValueError("Canonical disparity metadata does not match this render input")

        expected_paths = [
            metadata_file.parent / f"{Path(frame_name).stem}.png"
            for frame_name in metadata["frame_names"]
        ]
        if [path.resolve() for path in depth_files] != [path.resolve() for path in expected_paths]:
            raise ValueError("Canonical disparity files do not match the metadata path manifest")

        native_shape = tuple(int(value) for value in metadata["native_shape"])
        if len(native_shape) != 2 or any(value < 1 for value in native_shape):
            raise ValueError("Canonical disparity metadata has an invalid native shape")
        for path in depth_files:
            payload = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if (
                payload is None
                or payload.dtype != np.uint16
                or payload.ndim != 2
                or payload.shape != native_shape
            ):
                raise ValueError(f"Canonical disparity payload does not match metadata: {path}")
        return metadata
