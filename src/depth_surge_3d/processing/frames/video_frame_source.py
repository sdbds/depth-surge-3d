"""Zero-PNG streaming video frame source backed by sequential FFmpeg pipes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import hashlib
from pathlib import Path
import struct
import subprocess
import threading
from typing import Any

import cv2
import numpy as np

from ...core.file_identity import file_sample_fingerprint
from ...core.video_timing import selected_frame_range
from ...utils.path_utils import ffmpeg_frame_interval


STREAMING_SOURCE_ALGORITHM_VERSION = "ffmpeg-rawvideo-stream-v1"


def _compute_luma_histogram(gray_frame: np.ndarray) -> np.ndarray:
    """Compute the exact 32-bin L1-normalized luma histogram used by scene analysis."""

    histogram = cv2.calcHist([gray_frame], [0], None, [32], [0, 256]).astype(np.float32)
    normalized = np.empty_like(histogram)
    cv2.normalize(histogram, normalized, alpha=1.0, norm_type=cv2.NORM_L1)
    return normalized.reshape(-1)


def _read_exact(stream: Any, target: bytearray) -> int:
    """Read up to len(target) bytes from a binary pipe into target without extra copies."""

    view = memoryview(target)
    total = 0
    needed = len(target)
    while total < needed:
        chunk = stream.readinto(view[total:])
        if not chunk:
            break
        total += chunk
    return total


def _terminate_ffmpeg_process(proc: subprocess.Popen[bytes]) -> None:
    if proc.stdout is not None and not proc.stdout.closed:
        try:
            proc.stdout.close()
        except OSError:
            pass
    try:
        if proc.poll() is None:
            proc.terminate()
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5.0)
        except (OSError, subprocess.SubprocessError):
            pass
    except (OSError, subprocess.SubprocessError):
        pass


def _resolve_clip_interval(
    video_properties: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[tuple[int, int], str, str] | None:
    fps = float(video_properties.get("fps", 0.0))
    frame_range = selected_frame_range(video_properties, settings)
    start_frame, end_frame = frame_range
    expected_frames = end_frame - start_frame
    if expected_frames <= 0:
        return None
    if video_properties.get("frame_rate"):
        start_time, duration = ffmpeg_frame_interval(
            start_frame, end_frame, Fraction(video_properties["frame_rate"])
        )
    else:
        start_time = str(start_frame / fps if fps > 0 else 0)
        duration = str(expected_frames / fps if fps > 0 else 0)
    return frame_range, start_time, duration


def _scan_gray_stream(
    video_path: Path,
    *,
    start_time: str,
    duration: str,
    width: int,
    height: int,
    compute_histograms: bool,
) -> tuple[int, tuple[np.ndarray, ...] | None] | None:
    gray_nbytes = width * height
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-threads",
        "0",
        "-ss",
        start_time,
        "-i",
        str(video_path),
        "-t",
        duration,
        "-pix_fmt",
        "gray",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=max(gray_nbytes * 4, 1024 * 1024),
        )
    except OSError:
        return None

    buf = bytearray(gray_nbytes)
    frame_count = 0
    histograms: list[np.ndarray] = []
    partial_trailing = False
    try:
        assert proc.stdout is not None
        while True:
            got = _read_exact(proc.stdout, buf)
            if got == 0:
                break
            if got < gray_nbytes:
                partial_trailing = True
                break
            frame_count += 1
            if compute_histograms:
                gray = np.frombuffer(buf, dtype=np.uint8).reshape(height, width)
                histograms.append(_compute_luma_histogram(gray))
        proc.stdout.close()
        returncode = proc.wait(timeout=30.0)
    except (OSError, subprocess.SubprocessError):
        _terminate_ffmpeg_process(proc)
        return None

    if returncode != 0 or partial_trailing or frame_count <= 0:
        return None
    return frame_count, (tuple(histograms) if compute_histograms else None)


def _probe_ffmpeg_decoded_dimensions(
    video_path: Path,
    start_time: str,
) -> tuple[int, int] | None:
    """Decode one BMP frame header to confirm FFmpeg's exact rotated output dimensions."""

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-ss",
        start_time,
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "bmp",
        "pipe:1",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdout is not None
        header = proc.stdout.read(26)
        proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.read()
            proc.stderr.close()
        proc.wait(timeout=15.0)
    except (OSError, subprocess.SubprocessError):
        return None
    if len(header) < 26 or header[:2] != b"BM":
        return None
    raw_width, raw_height = struct.unpack_from("<ii", header, 18)
    width = abs(int(raw_width))
    height = abs(int(raw_height))
    if width <= 0 or height <= 0:
        return None
    return width, height


@dataclass(frozen=True)
class VideoFrameSource:
    """Immutable descriptor for a zero-PNG streaming video frame sequence."""

    video_path: Path
    start_time: str
    duration: str
    frame_range: tuple[int, int]
    width: int
    height: int
    frame_count: int
    frame_names: tuple[str, ...]
    source_fingerprint: str
    luma_histograms: tuple[np.ndarray, ...] | None
    virtual_dir: Path

    def virtual_frame_files(self) -> list[Path]:
        """Return canonical virtual frame paths matching extract_frames naming."""

        return [self.virtual_dir / name for name in self.frame_names]

    def index_for_path(self, path: Path) -> int:
        """Resolve a canonical virtual frame path to its 0-based stream index."""

        stem = path.stem
        if not stem.startswith("frame_"):
            raise ValueError(f"Noncanonical streaming frame name: {path.name}")
        index = int(stem[6:]) - 1
        if not 0 <= index < self.frame_count or self.frame_names[index] != path.name:
            raise KeyError(f"Frame name not found in streaming source: {path.name}")
        return index

    def open_reader(
        self,
        *,
        window_size: int = 40,
        consume_once: bool = False,
    ) -> "SequentialVideoFrameReader":
        """Open a thread-safe sequential BGR24 FFmpeg reader over this source."""

        return SequentialVideoFrameReader(
            self,
            window_size=window_size,
            consume_once=consume_once,
        )


class SequentialVideoFrameReader:
    """Thread-safe sequential BGR24 FFmpeg reader with bounded sliding-window cache."""

    def __init__(
        self,
        source: VideoFrameSource,
        *,
        window_size: int = 40,
        consume_once: bool = False,
    ) -> None:
        self.source = source
        self.window_size = max(1, int(window_size))
        self.consume_once = bool(consume_once)
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_lines: deque[str] = deque(maxlen=20)
        self._next_stream_index = 0
        self._cache: dict[int, np.ndarray] = {}
        self._frame_nbytes = source.height * source.width * 3
        self._buffer = bytearray(self._frame_nbytes)
        self._closed = False

    def __enter__(self) -> "SequentialVideoFrameReader":
        return self

    def __exit__(self, _exc_type: object, _exc_val: object, _exc_tb: object) -> None:
        self.close()

    def _drain_stderr(self, stream: Any) -> None:
        try:
            for raw_line in stream:
                text = raw_line.decode("utf-8", errors="replace").strip()
                if text:
                    self._stderr_lines.append(text)
        except OSError:
            pass

    def _start_process(self) -> None:
        self._stop_process()
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "0",
            "-ss",
            self.source.start_time,
            "-i",
            str(self.source.video_path),
            "-t",
            self.source.duration,
            "-pix_fmt",
            "bgr24",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        self._stderr_lines.clear()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=max(self._frame_nbytes * 2, 1024 * 1024),
        )
        self._process = proc
        self._next_stream_index = 0
        if proc.stderr is not None:
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr,
                args=(proc.stderr,),
                daemon=True,
            )
            self._stderr_thread.start()

    def _stop_process(self) -> None:
        proc = self._process
        self._process = None
        if proc is not None:
            _terminate_ffmpeg_process(proc)
        thread = self._stderr_thread
        self._stderr_thread = None
        if thread is not None:
            thread.join(timeout=2.0)

    def _read_next_frame_locked(self) -> np.ndarray:
        if self._process is None:
            self._start_process()
        assert self._process is not None and self._process.stdout is not None
        got = _read_exact(self._process.stdout, self._buffer)
        if got < self._frame_nbytes:
            tail = "; ".join(self._stderr_lines)
            self._stop_process()
            raise OSError(
                f"Streaming video frame {self._next_stream_index} truncated "
                f"({got}/{self._frame_nbytes} bytes)" + (f": {tail}" if tail else "")
            )
        frame = np.frombuffer(self._buffer, dtype=np.uint8).reshape(
            self.source.height, self.source.width, 3
        )
        return frame.copy()

    def get_frame(self, index: int) -> np.ndarray:
        """Return contiguous uint8 BGR frame at 0-based stream index."""

        if not 0 <= index < self.source.frame_count:
            raise IndexError(
                f"Streaming frame index {index} out of range [0, {self.source.frame_count})"
            )
        with self._lock:
            if self._closed:
                raise RuntimeError("SequentialVideoFrameReader is closed")
            cached = self._cache.get(index)
            if cached is not None:
                return self._cache.pop(index) if self.consume_once else cached

            if index < self._next_stream_index:
                # Backward request beyond sliding window (e.g. V2 OOM full-stage retry).
                self._cache.clear()
                self._start_process()

            while self._next_stream_index <= index:
                cur_idx = self._next_stream_index
                frame = self._read_next_frame_locked()
                self._next_stream_index += 1
                if self.consume_once:
                    if cur_idx == index:
                        return frame
                    self._cache[cur_idx] = frame
                else:
                    self._cache[cur_idx] = frame
                    evict_before = cur_idx - self.window_size
                    if evict_before >= 0:
                        for old_idx in [k for k in self._cache if k <= evict_before]:
                            del self._cache[old_idx]

            result = self._cache[index]
            return self._cache.pop(index) if self.consume_once else result

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._cache.clear()
            self._stop_process()


def build_video_frame_source(
    video_path: str | Path,
    frames_dir: Path,
    video_properties: dict[str, Any],
    settings: dict[str, Any],
    *,
    compute_histograms: bool = True,
) -> VideoFrameSource | None:
    """Index a video clip in one fast gray rawvideo FFmpeg pass (zero PNGs on disk)."""

    source_path = Path(video_path)
    if not source_path.is_file():
        return None
    interval = _resolve_clip_interval(video_properties, settings)
    if interval is None:
        return None
    frame_range, start_time, duration = interval
    probed = _probe_ffmpeg_decoded_dimensions(source_path, start_time)
    if probed is None:
        return None
    width, height = probed
    scanned = _scan_gray_stream(
        source_path,
        start_time=start_time,
        duration=duration,
        width=width,
        height=height,
        compute_histograms=compute_histograms,
    )
    if scanned is None:
        return None
    frame_count, histograms = scanned
    start_frame, end_frame = frame_range
    frame_names = tuple(f"frame_{idx:06d}.png" for idx in range(1, frame_count + 1))
    video_fp = file_sample_fingerprint(source_path)
    fp_payload = (
        f"{STREAMING_SOURCE_ALGORITHM_VERSION}\0{video_fp}\0"
        f"{start_frame}:{end_frame}\0{frame_count}\0{width}x{height}"
    ).encode("ascii")
    source_fingerprint = hashlib.sha256(fp_payload).hexdigest()

    return VideoFrameSource(
        video_path=source_path,
        start_time=start_time,
        duration=duration,
        frame_range=frame_range,
        width=width,
        height=height,
        frame_count=frame_count,
        frame_names=frame_names,
        source_fingerprint=source_fingerprint,
        luma_histograms=histograms,
        virtual_dir=Path(frames_dir),
    )
