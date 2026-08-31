"""Strict bounded validation for a generated final MP4 candidate."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import os
from pathlib import Path
import stat

from .bounded_process import (
    BoundedProcessCaptureV1,
    BoundedProcessResult,
    ProcessOutputBudgetError,
    ProcessTimeoutError,
    ProgressProcessResult,
    run_ffmpeg_progress_v1,
)
from .encoding_provider import (
    FINAL_DECODE_DIAGNOSTIC_BYTES,
    FINAL_ENCODING_LINE_BYTES,
    FINAL_PROBE_STDERR_BYTES,
    FINAL_PROBE_STDOUT_BYTES,
    FinalContainerValidationError,
    FinalDecodeProgressParser,
    FfprobeJsonBudgetError,
    FinalContainerExpectation,
    parse_final_ffprobe_json,
    validate_final_probe_streams,
)


FINAL_PROBE_TIMEOUT_BASE_SECONDS = 30
FINAL_PROBE_TIMEOUT_PER_1000_WORK_UNITS_SECONDS = 2
FINAL_PROBE_TIMEOUT_MAX_SECONDS = 21_600
FINAL_PROBE_BYTES_PER_WORK_UNIT = 64 * 1024
FINAL_PROBE_VIDEO_THREADS_MAX = 8
FINAL_DECODE_STALL_SECONDS = 120.0
_U64_MAX = 0xFFFFFFFFFFFFFFFF


class FinalContainerProcessError(RuntimeError):
    """FFprobe or the complete decode returned an invalid process result."""


class FinalContainerProbeTimeoutError(FinalContainerProcessError):
    """The bounded final-container metadata probe exceeded its deadline."""


class FinalContainerProbeBudgetError(FinalContainerProcessError):
    """The final-container probe exceeded its output or parser-state budget."""


class FinalContainerDecodeTimeoutError(FinalContainerProcessError):
    """The final full decode stopped making semantic progress."""


class FinalContainerDecodeBudgetError(FinalContainerProcessError):
    """The final full decode exceeded its bounded diagnostic state."""


FiniteRunner = Callable[..., BoundedProcessResult]
ProgressRunner = Callable[..., ProgressProcessResult]


def _checked_u64(value: int, label: str) -> int:
    if type(value) is not int or not 0 <= value <= _U64_MAX:
        raise FinalContainerValidationError(f"{label} exceeds u64")
    return value


def _checked_u64_add(left: int, right: int, label: str) -> int:
    _checked_u64(left, label)
    _checked_u64(right, label)
    if left > _U64_MAX - right:
        raise FinalContainerValidationError(f"{label} exceeds u64")
    return left + right


def _checked_u64_mul(left: int, right: int, label: str) -> int:
    _checked_u64(left, label)
    _checked_u64(right, label)
    if left and right > _U64_MAX // left:
        raise FinalContainerValidationError(f"{label} exceeds u64")
    return left * right


def _ceil_div_u64(value: int, divisor: int, label: str) -> int:
    _checked_u64(value, label)
    if divisor <= 0:
        raise ValueError("ceil divisor must be positive")
    quotient, remainder = divmod(value, divisor)
    return _checked_u64_add(quotient, int(bool(remainder)), label)


def derive_final_probe_timeout_seconds(
    candidate_byte_count: int,
    expectation: FinalContainerExpectation,
) -> int:
    """Derive the bounded count-frames deadline from frozen work units."""

    byte_count = _checked_u64(candidate_byte_count, "final candidate byte count")
    stream_units = _checked_u64_add(
        expectation.frame_count,
        expectation.audio_packet_upper_bound,
        "final probe stream work units",
    )
    byte_units = _ceil_div_u64(
        byte_count,
        FINAL_PROBE_BYTES_PER_WORK_UNIT,
        "final probe byte work units",
    )
    work_units = max(1, stream_units, byte_units)
    work_batches = _ceil_div_u64(work_units, 1000, "final probe work batches")
    increment = _checked_u64_mul(
        work_batches,
        FINAL_PROBE_TIMEOUT_PER_1000_WORK_UNITS_SECONDS,
        "final probe timeout increment",
    )
    timeout = _checked_u64_add(
        FINAL_PROBE_TIMEOUT_BASE_SECONDS,
        increment,
        "final probe timeout",
    )
    return min(FINAL_PROBE_TIMEOUT_MAX_SECONDS, timeout)


def _run_finite(
    arguments: Sequence[str],
    *,
    stdout_cap: int,
    stderr_cap: int,
    timeout_seconds: float,
    cancelled: Callable[[], bool] | None = None,
) -> BoundedProcessResult:
    return BoundedProcessCaptureV1(
        stdout_cap=stdout_cap,
        stderr_cap=stderr_cap,
        timeout_seconds=timeout_seconds,
        cancelled=cancelled,
    ).run(arguments)


def validate_final_container_v4(
    path: Path,
    expectation: FinalContainerExpectation,
    *,
    ffprobe: str = "ffprobe",
    ffmpeg: str = "ffmpeg",
    finite_runner: FiniteRunner = _run_finite,
    progress_runner: ProgressRunner = run_ffmpeg_progress_v1,
    cancelled: Callable[[], bool] | None = None,
) -> bool:
    """Probe scalar media facts, then decode every accepted stream to null."""

    candidate = Path(path)
    try:
        metadata = candidate.stat(follow_symlinks=False)
    except OSError as error:
        raise FinalContainerProcessError("final video candidate is unavailable") from error
    if not stat.S_ISREG(metadata.st_mode) or candidate.is_symlink() or metadata.st_size <= 0:
        raise FinalContainerProcessError("final video candidate is not an ordinary file")
    probe_timeout_seconds = derive_final_probe_timeout_seconds(
        metadata.st_size,
        expectation,
    )

    # FFprobe defaults to one decoder thread; cap parallelism instead of using auto.
    probe_video_threads = min(
        FINAL_PROBE_VIDEO_THREADS_MAX, os.cpu_count() or 1, expectation.frame_count
    )
    probe_arguments = (
        ffprobe,
        "-v",
        "error",
        "-threads:v",
        str(probe_video_threads),
        "-count_frames",
        "-show_entries",
        "stream=codec_type,codec_name,width,height,pix_fmt,sample_aspect_ratio,display_aspect_ratio,avg_frame_rate,nb_read_frames",
        "-of",
        "json",
        str(candidate),
    )
    try:
        finite_limits: dict[str, object] = {
            "stdout_cap": FINAL_PROBE_STDOUT_BYTES,
            "stderr_cap": FINAL_PROBE_STDERR_BYTES,
            "timeout_seconds": probe_timeout_seconds,
        }
        if cancelled is not None:
            finite_limits["cancelled"] = cancelled
        probe = finite_runner(probe_arguments, **finite_limits)
    except ProcessTimeoutError as error:
        raise FinalContainerProbeTimeoutError("final FFprobe exceeded its deadline") from error
    except ProcessOutputBudgetError as error:
        raise FinalContainerProbeBudgetError("final FFprobe exceeded its output budget") from error
    if probe.returncode != 0 or probe.stderr:
        raise FinalContainerProcessError("final FFprobe emitted diagnostics or failed")
    try:
        streams = parse_final_ffprobe_json(probe.stdout)
    except FfprobeJsonBudgetError as error:
        raise FinalContainerProbeBudgetError(
            "final FFprobe JSON exceeded its parser-state budget"
        ) from error
    validate_final_probe_streams(streams, expectation)
    del streams, probe

    decode_arguments = (
        ffmpeg,
        "-v",
        "error",
        "-xerror",
        "-nostats",
        "-progress",
        "pipe:1",
        "-i",
        str(candidate),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-f",
        "null",
        "-",
    )
    progress = FinalDecodeProgressParser()
    try:
        progress_limits: dict[str, object] = {
            "parser": progress,
            "diagnostic_cap": FINAL_DECODE_DIAGNOSTIC_BYTES,
            "line_cap": FINAL_ENCODING_LINE_BYTES,
            "stall_seconds": FINAL_DECODE_STALL_SECONDS,
        }
        if cancelled is not None:
            progress_limits["cancelled"] = cancelled
        decoded = progress_runner(decode_arguments, **progress_limits)
    except ProcessTimeoutError as error:
        raise FinalContainerDecodeTimeoutError(
            "final full decode stopped making progress"
        ) from error
    except ProcessOutputBudgetError as error:
        raise FinalContainerDecodeBudgetError(
            "final full decode exceeded its diagnostic budget"
        ) from error
    if decoded.returncode != 0 or decoded.stderr:
        raise FinalContainerProcessError("final full decode emitted diagnostics or failed")
    if not progress.ended:
        raise FinalContainerProcessError("final full decode has no terminal progress state")
    return True
