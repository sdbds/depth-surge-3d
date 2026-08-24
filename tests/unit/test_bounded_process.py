from __future__ import annotations

import sys
import time

import pytest

from src.depth_surge_3d.processing.video.bounded_process import (
    BoundedProcessCaptureV1,
    ProcessOutputBudgetError,
    ProcessTimeoutError,
    run_ffmpeg_progress_v1,
    run_streaming_byte_count_v1,
)
from src.depth_surge_3d.processing.video.encoding_provider import FfmpegProgressParser


def test_bounded_capture_drains_stdout_and_stderr_concurrently() -> None:
    result = BoundedProcessCaptureV1(
        stdout_cap=4096,
        stderr_cap=4096,
        timeout_seconds=5.0,
    ).run(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(b'o'*2048); " "sys.stderr.buffer.write(b'e'*2048)",
        ]
    )

    assert result.returncode == 0
    assert result.stdout == b"o" * 2048
    assert result.stderr == b"e" * 2048


def test_bounded_capture_reports_overflow_after_reaping_child() -> None:
    with pytest.raises(ProcessOutputBudgetError, match="stdout"):
        BoundedProcessCaptureV1(
            stdout_cap=127,
            stderr_cap=1024,
            timeout_seconds=5.0,
        ).run([sys.executable, "-c", "print('x'*512)"])


def test_bounded_capture_timeout_terminates_and_reaps_child() -> None:
    started = time.monotonic()
    with pytest.raises(ProcessTimeoutError):
        BoundedProcessCaptureV1(
            stdout_cap=1024,
            stderr_cap=1024,
            timeout_seconds=0.2,
            terminate_grace_seconds=0.2,
        ).run([sys.executable, "-c", "import time; time.sleep(30)"])
    assert time.monotonic() - started < 3.0


def test_progress_runner_requires_semantic_progress_and_terminal_end() -> None:
    parser = FfmpegProgressParser()
    result = run_ffmpeg_progress_v1(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('frame=1\\nprogress=end\\n'); " "sys.stdout.flush()",
        ],
        parser=parser,
        diagnostic_cap=1024,
        line_cap=8192,
        stall_seconds=1.0,
    )

    assert result.returncode == 0
    assert result.stderr == b""
    assert parser.ended is True


def test_streaming_runner_counts_stdout_without_retaining_it() -> None:
    result = run_streaming_byte_count_v1(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(b'x'*131071); sys.stdout.flush()",
        ],
        diagnostic_cap=1024,
        stall_seconds=2.0,
        stream_buffer_bytes=4096,
    )

    assert result.returncode == 0
    assert result.byte_count == 131071
    assert result.stderr == b""


def test_streaming_runner_drains_then_rejects_diagnostics() -> None:
    with pytest.raises(ProcessOutputBudgetError, match="stderr"):
        run_streaming_byte_count_v1(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'x'*4096); "
                "sys.stderr.buffer.write(b'e'*2048)",
            ],
            diagnostic_cap=128,
            stall_seconds=2.0,
            stream_buffer_bytes=512,
        )
