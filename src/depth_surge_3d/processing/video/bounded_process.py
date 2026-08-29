"""Bounded, deadlock-free subprocess adapters for final-media validation."""

from __future__ import annotations

from dataclasses import dataclass
import subprocess
import threading
import time
from collections.abc import Sequence
from typing import Callable, Protocol


_READ_BYTES = 64 * 1024


class BoundedProcessError(RuntimeError):
    """Base failure for a bounded child-process lifecycle."""


class ProcessOutputBudgetError(BoundedProcessError):
    """A child stream exceeded its fixed resident-byte allowance."""


class ProcessTimeoutError(BoundedProcessError):
    """A child exceeded its wall or semantic-progress deadline."""


class ProcessPipeError(BoundedProcessError):
    """A child pipe could not be drained to completion."""


@dataclass(frozen=True)
class BoundedProcessResult:
    stdout: bytes
    stderr: bytes
    returncode: int


@dataclass(frozen=True)
class ProgressProcessResult:
    stderr: bytes
    returncode: int


@dataclass(frozen=True)
class StreamingProcessResult:
    byte_count: int
    stderr: bytes
    returncode: int


class ProgressLineParser(Protocol):
    def feed_line(self, raw_line: bytes) -> bool: ...

    def finish(self) -> None: ...


@dataclass
class _CapturedStream:
    cap: int
    name: str
    payload: bytearray
    overflow: bool = False
    error: BaseException | None = None


def _terminate_and_reap(
    process: subprocess.Popen[bytes],
    *,
    grace_seconds: float,
) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    else:
        process.wait()


def _drain_capture(
    stream,
    state: _CapturedStream,
    wake: threading.Event,
) -> None:
    try:
        while True:
            chunk = stream.read(_READ_BYTES)
            if not chunk:
                return
            available = max(state.cap - len(state.payload), 0)
            if available:
                state.payload.extend(chunk[:available])
            if len(chunk) > available:
                state.overflow = True
                wake.set()
    except BaseException as error:  # pragma: no cover - OS pipe failures are platform-specific
        state.error = error
        wake.set()


class BoundedProcessCaptureV1:
    """Capture two finite child streams concurrently under exact byte caps."""

    def __init__(
        self,
        *,
        stdout_cap: int,
        stderr_cap: int,
        timeout_seconds: float,
        terminate_grace_seconds: float = 5.0,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        if stdout_cap < 0 or stderr_cap < 0:
            raise ValueError("process stream caps must be nonnegative")
        if timeout_seconds <= 0 or terminate_grace_seconds <= 0:
            raise ValueError("process deadlines must be positive")
        self.stdout_cap = stdout_cap
        self.stderr_cap = stderr_cap
        self.timeout_seconds = timeout_seconds
        self.terminate_grace_seconds = terminate_grace_seconds
        self.cancelled = cancelled

    def run(self, arguments: Sequence[str]) -> BoundedProcessResult:  # noqa: C901
        if not arguments:
            raise ValueError("bounded process arguments cannot be empty")
        process = subprocess.Popen(
            list(arguments),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        assert process.stdout is not None and process.stderr is not None
        wake = threading.Event()
        stdout = _CapturedStream(self.stdout_cap, "stdout", bytearray())
        stderr = _CapturedStream(self.stderr_cap, "stderr", bytearray())
        threads = [
            threading.Thread(
                target=_drain_capture,
                args=(process.stdout, stdout, wake),
                name="bounded-process-stdout",
                daemon=True,
            ),
            threading.Thread(
                target=_drain_capture,
                args=(process.stderr, stderr, wake),
                name="bounded-process-stderr",
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + self.timeout_seconds
        timed_out = False
        was_cancelled = False
        try:
            while process.poll() is None:
                if stdout.overflow or stderr.overflow or stdout.error or stderr.error:
                    break
                if self.cancelled is not None and self.cancelled():
                    was_cancelled = True
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                wake.wait(min(remaining, 0.05))
                wake.clear()
        finally:
            if process.poll() is None:
                _terminate_and_reap(
                    process,
                    grace_seconds=self.terminate_grace_seconds,
                )
            else:
                process.wait()
            for thread in threads:
                thread.join(timeout=self.terminate_grace_seconds)
            process.stdout.close()
            process.stderr.close()
        if any(thread.is_alive() for thread in threads):
            raise ProcessPipeError("child pipe drain did not terminate")
        if was_cancelled:
            raise InterruptedError("bounded child stopped by cancellation request")
        if timed_out:
            raise ProcessTimeoutError("bounded child exceeded its wall deadline")
        for state in (stdout, stderr):
            if state.error is not None:
                raise ProcessPipeError(f"{state.name} drain failed") from state.error
            if state.overflow:
                raise ProcessOutputBudgetError(f"{state.name} exceeded its {state.cap}-byte cap")
        assert process.returncode is not None
        return BoundedProcessResult(
            stdout=bytes(stdout.payload),
            stderr=bytes(stderr.payload),
            returncode=process.returncode,
        )


@dataclass
class _ProgressState:
    parser: ProgressLineParser
    line_cap: int
    last_semantic_progress: float
    error: BaseException | None = None


@dataclass
class _StreamingCountState:
    byte_count: int
    last_progress: float
    error: BaseException | None = None


def _drain_progress(  # noqa: C901
    stream,
    state: _ProgressState,
    wake: threading.Event,
) -> None:
    pending = bytearray()
    discarding = False
    try:
        while True:
            chunk = stream.read(_READ_BYTES)
            if not chunk:
                if pending and state.error is None:
                    state.error = ProcessPipeError(
                        "progress stream ended with an unterminated line"
                    )
                    wake.set()
                return
            for byte in chunk:
                if discarding:
                    if byte == 0x0A:
                        discarding = False
                    continue
                pending.append(byte)
                if len(pending) > state.line_cap:
                    state.error = ProcessOutputBudgetError("progress line exceeded its byte cap")
                    pending.clear()
                    discarding = True
                    wake.set()
                    continue
                if byte == 0x0A:
                    if state.error is None and state.parser.feed_line(bytes(pending)):
                        state.last_semantic_progress = time.monotonic()
                        wake.set()
                    pending.clear()
    except BaseException as error:
        state.error = error
        wake.set()


def _drain_stream_count(
    stream,
    state: _StreamingCountState,
    wake: threading.Event,
    *,
    read_bytes: int,
    maximum_byte_count: int,
) -> None:
    buffer = bytearray(read_bytes)
    try:
        while True:
            count = stream.readinto(buffer)
            if not count:
                return
            if state.byte_count > maximum_byte_count - count:
                state.error = ProcessOutputBudgetError("stream byte count exceeds its limit")
                wake.set()
                continue
            state.byte_count += count
            state.last_progress = time.monotonic()
            wake.set()
    except BaseException as error:  # pragma: no cover - OS pipe failures are platform-specific
        state.error = error
        wake.set()


def run_ffmpeg_progress_v1(  # noqa: C901
    arguments: Sequence[str],
    *,
    parser: ProgressLineParser,
    diagnostic_cap: int,
    line_cap: int,
    stall_seconds: float,
    terminate_grace_seconds: float = 5.0,
    cancelled: Callable[[], bool] | None = None,
) -> ProgressProcessResult:
    """Drain FFmpeg progress and diagnostics without cumulative stdout storage."""

    if not arguments:
        raise ValueError("progress process arguments cannot be empty")
    if diagnostic_cap < 0 or line_cap <= 0:
        raise ValueError("progress process caps are invalid")
    if stall_seconds <= 0 or terminate_grace_seconds <= 0:
        raise ValueError("progress process deadlines must be positive")
    process = subprocess.Popen(
        list(arguments),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    assert process.stdout is not None and process.stderr is not None
    wake = threading.Event()
    progress = _ProgressState(parser, line_cap, time.monotonic())
    diagnostics = _CapturedStream(diagnostic_cap, "stderr", bytearray())
    threads = [
        threading.Thread(
            target=_drain_progress,
            args=(process.stdout, progress, wake),
            name="ffmpeg-progress-stdout",
            daemon=True,
        ),
        threading.Thread(
            target=_drain_capture,
            args=(process.stderr, diagnostics, wake),
            name="ffmpeg-progress-stderr",
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    timed_out = False
    was_cancelled = False
    try:
        while process.poll() is None:
            if progress.error or diagnostics.error or diagnostics.overflow:
                break
            if cancelled is not None and cancelled():
                was_cancelled = True
                break
            remaining = stall_seconds - (time.monotonic() - progress.last_semantic_progress)
            if remaining <= 0:
                timed_out = True
                break
            wake.wait(min(remaining, 0.05))
            wake.clear()
    finally:
        if process.poll() is None:
            _terminate_and_reap(process, grace_seconds=terminate_grace_seconds)
        else:
            process.wait()
        for thread in threads:
            thread.join(timeout=terminate_grace_seconds)
        process.stdout.close()
        process.stderr.close()
    if any(thread.is_alive() for thread in threads):
        raise ProcessPipeError("progress pipe drain did not terminate")
    if was_cancelled:
        raise InterruptedError("progress child stopped by cancellation request")
    if timed_out:
        raise ProcessTimeoutError("FFmpeg made no semantic progress before its deadline")
    if progress.error is not None:
        if isinstance(progress.error, BoundedProcessError):
            raise progress.error
        raise ProcessPipeError("progress parser failed") from progress.error
    if diagnostics.error is not None:
        raise ProcessPipeError("FFmpeg diagnostic drain failed") from diagnostics.error
    if diagnostics.overflow:
        raise ProcessOutputBudgetError(f"stderr exceeded its {diagnostics.cap}-byte cap")
    parser.finish()
    assert process.returncode is not None
    return ProgressProcessResult(
        stderr=bytes(diagnostics.payload),
        returncode=process.returncode,
    )


class FfmpegInputPipeProcessV1:
    """Feed a child stdin while bounded threads drain FFmpeg output pipes."""

    def __init__(
        self,
        *,
        parser: ProgressLineParser,
        diagnostic_cap: int,
        line_cap: int,
        stall_seconds: float,
        terminate_grace_seconds: float = 5.0,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        if diagnostic_cap < 0 or line_cap <= 0:
            raise ValueError("input-pipe process caps are invalid")
        if stall_seconds <= 0 or terminate_grace_seconds <= 0:
            raise ValueError("input-pipe process deadlines must be positive")
        self.parser = parser
        self.diagnostic_cap = diagnostic_cap
        self.line_cap = line_cap
        self.stall_seconds = stall_seconds
        self.terminate_grace_seconds = terminate_grace_seconds
        self.cancelled = cancelled
        self._process: subprocess.Popen[bytes] | None = None
        self._progress: _ProgressState | None = None
        self._diagnostics: _CapturedStream | None = None
        self._drain_threads: list[threading.Thread] = []
        self._watchdog_thread: threading.Thread | None = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._operation_active = threading.Event()
        self._state_lock = threading.Lock()
        self._termination_lock = threading.Lock()
        self._last_input_progress = 0.0
        self._failure: BaseException | None = None
        self._closed = False

    def start(self, arguments: Sequence[str]) -> None:
        if not arguments:
            raise ValueError("input-pipe process arguments cannot be empty")
        if self._process is not None:
            raise RuntimeError("input-pipe process has already been started")
        process = subprocess.Popen(
            list(arguments),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        started = time.monotonic()
        self._process = process
        self._last_input_progress = started
        self._progress = _ProgressState(self.parser, self.line_cap, started)
        self._diagnostics = _CapturedStream(
            self.diagnostic_cap,
            "stderr",
            bytearray(),
        )
        self._drain_threads = [
            threading.Thread(
                target=_drain_progress,
                args=(process.stdout, self._progress, self._wake),
                name="ffmpeg-input-progress-stdout",
                daemon=True,
            ),
            threading.Thread(
                target=_drain_capture,
                args=(process.stderr, self._diagnostics, self._wake),
                name="ffmpeg-input-progress-stderr",
                daemon=True,
            ),
        ]
        for thread in self._drain_threads:
            thread.start()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog,
            name="ffmpeg-input-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def _set_failure(self, error: BaseException) -> BaseException:
        with self._state_lock:
            if self._failure is None:
                self._failure = error
            selected = self._failure
        self._wake.set()
        return selected

    def _current_failure(self) -> BaseException | None:
        progress = self._progress
        diagnostics = self._diagnostics
        if progress is not None and progress.error is not None:
            error = progress.error
            if not isinstance(error, BoundedProcessError):
                error = ProcessPipeError("progress parser failed")
            return self._set_failure(error)
        if diagnostics is not None and diagnostics.error is not None:
            return self._set_failure(ProcessPipeError("FFmpeg diagnostic drain failed"))
        if diagnostics is not None and diagnostics.overflow:
            return self._set_failure(
                ProcessOutputBudgetError(f"stderr exceeded its {diagnostics.cap}-byte cap")
            )
        with self._state_lock:
            return self._failure

    def _terminate(self) -> None:
        process = self._process
        if process is None:
            return
        with self._termination_lock:
            _terminate_and_reap(
                process,
                grace_seconds=self.terminate_grace_seconds,
            )

    def _watchdog(self) -> None:
        process = self._process
        progress = self._progress
        assert process is not None and progress is not None
        while not self._stop.is_set():
            error = self._current_failure()
            if error is None and self.cancelled is not None and self.cancelled():
                error = self._set_failure(
                    InterruptedError("input-pipe child stopped by cancellation request")
                )
            if error is None and self._operation_active.is_set():
                last_progress = max(
                    self._last_input_progress,
                    progress.last_semantic_progress,
                )
                if time.monotonic() - last_progress >= self.stall_seconds:
                    error = self._set_failure(
                        ProcessTimeoutError(
                            "FFmpeg input pipe made no progress before its deadline"
                        )
                    )
            if error is not None:
                try:
                    self._terminate()
                except BaseException as terminate_error:  # pragma: no cover - OS-specific
                    failure = self._set_failure(
                        ProcessPipeError("input-pipe child could not be reaped")
                    )
                    failure.__cause__ = terminate_error
                return
            if process.poll() is not None:
                self._wake.set()
                return
            self._wake.wait(0.05)
            self._wake.clear()

    def _require_open_process(self) -> subprocess.Popen[bytes]:
        if self._process is None:
            raise RuntimeError("input-pipe process has not been started")
        if self._closed:
            raise RuntimeError("input-pipe process is already closed")
        return self._process

    def _close_runtime(self, *, terminate: bool) -> None:  # noqa: C901
        process = self._process
        if process is None or self._closed:
            return
        if terminate or process.poll() is None:
            self._terminate()
        else:
            process.wait()
        self._stop.set()
        self._wake.set()
        watchdog = self._watchdog_thread
        if watchdog is not None and watchdog is not threading.current_thread():
            watchdog.join(timeout=self.terminate_grace_seconds)
        for thread in self._drain_threads:
            thread.join(timeout=self.terminate_grace_seconds)
        streams = (process.stdin, process.stdout, process.stderr)
        for stream in streams:
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError:
                    pass
        self._closed = True
        if watchdog is not None and watchdog.is_alive():
            raise ProcessPipeError("input-pipe watchdog did not terminate")
        if any(thread.is_alive() for thread in self._drain_threads):
            raise ProcessPipeError("input-pipe drain did not terminate")

    def _raise_after_cleanup(self, fallback: BaseException) -> None:
        failure = self._current_failure()
        try:
            self._close_runtime(terminate=True)
        except BaseException as cleanup_error:
            if failure is None:
                raise cleanup_error from fallback
        if failure is not None:
            raise failure from fallback
        raise fallback

    def write(self, payload: bytes | bytearray | memoryview) -> None:
        process = self._require_open_process()
        assert process.stdin is not None
        try:
            view = memoryview(payload).cast("B")
        except (TypeError, ValueError) as error:
            raise ValueError("input-pipe payload must be a contiguous byte buffer") from error
        self._operation_active.set()
        self._last_input_progress = time.monotonic()
        try:
            offset = 0
            while offset < len(view):
                failure = self._current_failure()
                if failure is not None:
                    self._raise_after_cleanup(failure)
                if process.poll() is not None:
                    self._raise_after_cleanup(
                        ProcessPipeError("input-pipe child exited before consuming its input")
                    )
                chunk = view[offset : offset + _READ_BYTES]
                try:
                    written = process.stdin.write(chunk)
                except (BrokenPipeError, OSError):
                    self._raise_after_cleanup(
                        ProcessPipeError("input-pipe child stopped accepting input")
                    )
                if written is None or written <= 0:
                    self._raise_after_cleanup(
                        ProcessPipeError("input-pipe child accepted no input bytes")
                    )
                offset += written
                self._last_input_progress = time.monotonic()
                self._wake.set()
        finally:
            self._operation_active.clear()

    def finish(self) -> ProgressProcessResult:
        process = self._require_open_process()
        assert process.stdin is not None
        self._operation_active.set()
        self._last_input_progress = time.monotonic()
        try:
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError):
                self._raise_after_cleanup(
                    ProcessPipeError("input-pipe child stopped before end-of-input")
                )
            while process.poll() is None and self._current_failure() is None:
                self._wake.wait(0.05)
                self._wake.clear()
            failure = self._current_failure()
            self._close_runtime(terminate=failure is not None)
            if failure is not None:
                raise failure
            self.parser.finish()
            assert process.returncode is not None
            diagnostics = self._diagnostics
            assert diagnostics is not None
            return ProgressProcessResult(
                stderr=bytes(diagnostics.payload),
                returncode=process.returncode,
            )
        finally:
            self._operation_active.clear()

    def abort(self) -> None:
        if self._process is None or self._closed:
            return
        self._set_failure(InterruptedError("input-pipe child was aborted"))
        self._close_runtime(terminate=True)


def run_streaming_byte_count_v1(  # noqa: C901
    arguments: Sequence[str],
    *,
    diagnostic_cap: int,
    stall_seconds: float,
    stream_buffer_bytes: int,
    maximum_byte_count: int = 0xFFFFFFFFFFFFFFFF,
    terminate_grace_seconds: float = 5.0,
    cancelled: Callable[[], bool] | None = None,
) -> StreamingProcessResult:
    """Drain stdout through one reusable buffer while retaining bounded stderr."""

    if not arguments:
        raise ValueError("streaming process arguments cannot be empty")
    if diagnostic_cap < 0 or stream_buffer_bytes <= 0 or maximum_byte_count < 0:
        raise ValueError("streaming process caps are invalid")
    if stall_seconds <= 0 or terminate_grace_seconds <= 0:
        raise ValueError("streaming process deadlines must be positive")
    process = subprocess.Popen(
        list(arguments),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    assert process.stdout is not None and process.stderr is not None
    wake = threading.Event()
    counted = _StreamingCountState(0, time.monotonic())
    diagnostics = _CapturedStream(diagnostic_cap, "stderr", bytearray())
    threads = [
        threading.Thread(
            target=_drain_stream_count,
            args=(process.stdout, counted, wake),
            kwargs={
                "read_bytes": stream_buffer_bytes,
                "maximum_byte_count": maximum_byte_count,
            },
            name="stream-count-stdout",
            daemon=True,
        ),
        threading.Thread(
            target=_drain_capture,
            args=(process.stderr, diagnostics, wake),
            name="stream-count-stderr",
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    timed_out = False
    was_cancelled = False
    try:
        while process.poll() is None:
            if counted.error or diagnostics.error or diagnostics.overflow:
                break
            if cancelled is not None and cancelled():
                was_cancelled = True
                break
            remaining = stall_seconds - (time.monotonic() - counted.last_progress)
            if remaining <= 0:
                timed_out = True
                break
            wake.wait(min(remaining, 0.05))
            wake.clear()
    finally:
        if process.poll() is None:
            _terminate_and_reap(process, grace_seconds=terminate_grace_seconds)
        else:
            process.wait()
        for thread in threads:
            thread.join(timeout=terminate_grace_seconds)
        process.stdout.close()
        process.stderr.close()
    if any(thread.is_alive() for thread in threads):
        raise ProcessPipeError("streaming pipe drain did not terminate")
    if was_cancelled:
        raise InterruptedError("streaming child stopped by cancellation request")
    if timed_out:
        raise ProcessTimeoutError("streaming child made no progress before its deadline")
    if counted.error is not None:
        if isinstance(counted.error, BoundedProcessError):
            raise counted.error
        raise ProcessPipeError("streaming stdout drain failed") from counted.error
    if diagnostics.error is not None:
        raise ProcessPipeError("streaming diagnostic drain failed") from diagnostics.error
    if diagnostics.overflow:
        raise ProcessOutputBudgetError(f"stderr exceeded its {diagnostics.cap}-byte cap")
    assert process.returncode is not None
    return StreamingProcessResult(
        byte_count=counted.byte_count,
        stderr=bytes(diagnostics.payload),
        returncode=process.returncode,
    )
