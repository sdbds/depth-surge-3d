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
from typing import IO, Any

import cv2
import numpy as np

from ...core.file_identity import file_sample_fingerprint
from ...core.video_timing import selected_frame_range
from ...io.durable_publish import atomic_write_json
from ...utils.path_utils import ffmpeg_frame_interval
from .scene_analyzer import luma_histogram
from .source_frame_manifest import SOURCE_FRAME_METADATA_NAME, read_source_frame_manifest


STREAMING_SOURCE_ALGORITHM_VERSION = "ffmpeg-rawvideo-stream-v2"


def streaming_frame_names(count: int) -> tuple[str, ...]:
    return tuple(f"frame_{idx:06d}.png" for idx in range(1, count + 1))


def stream_index_for_clip(
    metadata: dict[str, Any] | None, video_fp: str | None, frame_range: tuple[int, int]
) -> dict[str, Any] | None:
    if (
        metadata
        and metadata.get("algorithm_version") == STREAMING_SOURCE_ALGORITHM_VERSION
        and video_fp
        and metadata.get("source_video_fingerprint") == video_fp
        and (metadata.get("start_frame"), metadata.get("end_frame_exclusive")) == frame_range
        and all(
            type(metadata.get(k)) is int and metadata[k] > 0
            for k in ("width", "height", "num_frames")
        )
        and type(metadata.get("use_cuda")) is bool
    ):
        return metadata
    return None


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


def _read_png_histogram(stream: IO[bytes]) -> np.ndarray | None:
    frame = bytearray(stream.read(8))
    if not frame:
        return None
    # image2pipe concatenates PNGs; keep each frame's color chunks for OpenCV's gray decode.
    while True:
        header = stream.read(8)
        length, kind = struct.unpack(">I4s", header)
        frame.extend(header)
        payload = stream.read(length + 4)  # Includes the chunk CRC.
        if len(payload) != length + 4:
            raise OSError("Incomplete PNG from FFmpeg")
        frame.extend(payload)
        if kind == b"IEND":
            break
    gray = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise OSError("Could not decode PNG from FFmpeg")
    return luma_histogram(gray)


def _scan_video_stream(
    video_path: Path,
    *,
    start_time: str,
    duration: str,
    width: int,
    height: int,
    compute_histograms: bool,
    use_cuda: bool,
) -> tuple[int, tuple[np.ndarray, ...] | None] | None:
    frame_nbytes = width * height
    output = (
        ["-pix_fmt", "rgb24", "-c:v", "png", "-compression_level", "0", "-f", "image2pipe"]
        if compute_histograms
        else ["-pix_fmt", "gray", "-f", "rawvideo"]
    )
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-threads",
        "0",
        *(["-hwaccel", "cuda"] if use_cuda else []),
        "-ss",
        start_time,
        "-i",
        str(video_path),
        "-t",
        duration,
        "-fps_mode",
        "passthrough",
        *output,
        "pipe:1",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=max(frame_nbytes, 1024 * 1024),
        )
    except OSError:
        return None

    frame_count = 0
    histograms: list[np.ndarray] = []
    try:
        assert proc.stdout is not None
        if compute_histograms:
            while (histogram := _read_png_histogram(proc.stdout)) is not None:
                histograms.append(histogram)
            frame_count = len(histograms)
        else:
            buf = bytearray(frame_nbytes)
            while got := _read_exact(proc.stdout, buf):
                if got != frame_nbytes:
                    raise OSError("Incomplete gray frame from FFmpeg")
                frame_count += 1
        proc.stdout.close()
        returncode = proc.wait(timeout=30.0)
    except (OSError, struct.error, cv2.error, subprocess.SubprocessError):
        _terminate_ffmpeg_process(proc)
        return None

    if returncode != 0 or frame_count <= 0:
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
    use_cuda: bool = False

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
            *(["-hwaccel", "cuda"] if self.source.use_cuda else []),
            "-ss",
            self.source.start_time,
            "-i",
            str(self.source.video_path),
            "-t",
            self.source.duration,
            "-pix_fmt",
            "rgb24",
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
            decoder = "CUDA" if self.source.use_cuda else "CPU"
            raise OSError(
                f"Streaming video frame {self._next_stream_index} truncated using {decoder} "
                f"({got}/{self._frame_nbytes} bytes)" + (f": {tail}" if tail else "")
            )
        frame = np.frombuffer(self._buffer, dtype=np.uint8).reshape(
            self.source.height, self.source.width, 3
        )
        return frame[:, :, ::-1].copy()

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
                self._cache.pop(cur_idx - self.window_size, None)
                self._cache[cur_idx] = frame

            return self._cache.pop(index) if self.consume_once else self._cache[index]

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
    reuse_index: bool = False,
) -> VideoFrameSource | None:
    """Index the selected clip without writing source PNGs."""

    source_path = Path(video_path)
    if not source_path.is_file():
        return None
    interval = _resolve_clip_interval(video_properties, settings)
    if interval is None:
        return None
    frame_range, start_time, duration = interval
    start_frame, end_frame = frame_range
    video_fp = file_sample_fingerprint(source_path)
    saved = stream_index_for_clip(read_source_frame_manifest(frames_dir), video_fp, frame_range)
    if reuse_index:
        if saved is None:
            return None
        width, height = saved["width"], saved["height"]
        frame_count, use_cuda = saved["num_frames"], saved["use_cuda"]
        histograms = None
    else:
        probed = _probe_ffmpeg_decoded_dimensions(source_path, start_time)
        if probed is None:
            return None
        width, height = probed
        # Prefer the original decoder when resuming; CPU and CUDA can yield different RGB.
        decoders = (False,) if saved and not saved["use_cuda"] else (True, False)
        for use_cuda in decoders:
            scanned = _scan_video_stream(
                source_path,
                start_time=start_time,
                duration=duration,
                width=width,
                height=height,
                compute_histograms=compute_histograms,
                use_cuda=use_cuda,
            )
            if scanned is not None:
                break
        else:
            return None
        frame_count, histograms = scanned
        atomic_write_json(
            Path(frames_dir) / SOURCE_FRAME_METADATA_NAME,
            {
                "algorithm_version": STREAMING_SOURCE_ALGORITHM_VERSION,
                "source_video_fingerprint": video_fp,
                "start_frame": start_frame,
                "end_frame_exclusive": end_frame,
                "num_frames": frame_count,
                "width": width,
                "height": height,
                "use_cuda": use_cuda,
            },
        )
    fp_payload = (
        f"{STREAMING_SOURCE_ALGORITHM_VERSION}\0{use_cuda}\0{video_fp}\0"
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
        frame_names=streaming_frame_names(frame_count),
        source_fingerprint=source_fingerprint,
        luma_histograms=histograms,
        virtual_dir=Path(frames_dir),
        use_cuda=use_cuda,
    )
