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
