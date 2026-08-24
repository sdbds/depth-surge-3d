"""Strict bounded validation for a generated final MP4 candidate."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
import stat

from .bounded_process import (
    BoundedProcessCaptureV1,
    BoundedProcessResult,
    ProgressProcessResult,
    run_ffmpeg_progress_v1,
)
from .encoding_provider import (
    FINAL_DECODE_DIAGNOSTIC_BYTES,
    FINAL_ENCODING_LINE_BYTES,
    FINAL_PROBE_STDERR_BYTES,
    FINAL_PROBE_STDOUT_BYTES,
    FfmpegProgressParser,
    FinalContainerExpectation,
    parse_final_ffprobe_json,
    validate_final_probe_streams,
)


FINAL_PROBE_TIMEOUT_SECONDS = 30.0
FINAL_DECODE_STALL_SECONDS = 120.0


class FinalContainerProcessError(RuntimeError):
    """FFprobe or the complete decode returned an invalid process result."""


FiniteRunner = Callable[..., BoundedProcessResult]
ProgressRunner = Callable[..., ProgressProcessResult]


def _run_finite(
    arguments: Sequence[str],
    *,
    stdout_cap: int,
    stderr_cap: int,
    timeout_seconds: float,
) -> BoundedProcessResult:
    return BoundedProcessCaptureV1(
        stdout_cap=stdout_cap,
        stderr_cap=stderr_cap,
        timeout_seconds=timeout_seconds,
    ).run(arguments)


def validate_final_container_v4(
    path: Path,
    expectation: FinalContainerExpectation,
    *,
    ffprobe: str = "ffprobe",
    ffmpeg: str = "ffmpeg",
    finite_runner: FiniteRunner = _run_finite,
    progress_runner: ProgressRunner = run_ffmpeg_progress_v1,
) -> bool:
    """Probe scalar media facts, then decode every accepted stream to null."""

    candidate = Path(path)
    try:
        metadata = candidate.stat(follow_symlinks=False)
    except OSError as error:
        raise FinalContainerProcessError("final video candidate is unavailable") from error
    if not stat.S_ISREG(metadata.st_mode) or candidate.is_symlink() or metadata.st_size <= 0:
        raise FinalContainerProcessError("final video candidate is not an ordinary file")

    probe_arguments = (
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        "stream=codec_type,codec_name,width,height,pix_fmt,sample_aspect_ratio,display_aspect_ratio,avg_frame_rate,nb_read_frames",
        "-of",
        "json",
        str(candidate),
    )
    probe = finite_runner(
        probe_arguments,
        stdout_cap=FINAL_PROBE_STDOUT_BYTES,
        stderr_cap=FINAL_PROBE_STDERR_BYTES,
        timeout_seconds=FINAL_PROBE_TIMEOUT_SECONDS,
    )
    if probe.returncode != 0 or probe.stderr:
        raise FinalContainerProcessError("final FFprobe emitted diagnostics or failed")
    streams = parse_final_ffprobe_json(probe.stdout)
    validate_final_probe_streams(streams, expectation)

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
    progress = FfmpegProgressParser()
    decoded = progress_runner(
        decode_arguments,
        parser=progress,
        diagnostic_cap=FINAL_DECODE_DIAGNOSTIC_BYTES,
        line_cap=FINAL_ENCODING_LINE_BYTES,
        stall_seconds=FINAL_DECODE_STALL_SECONDS,
    )
    if decoded.returncode != 0 or decoded.stderr:
        raise FinalContainerProcessError("final full decode emitted diagnostics or failed")
    if not progress.ended:
        raise FinalContainerProcessError("final full decode has no terminal progress state")
    return True
