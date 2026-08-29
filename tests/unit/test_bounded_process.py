from __future__ import annotations

import sys
import time

import pytest

from src.depth_surge_3d.processing.video.bounded_process import (
    BoundedProcessCaptureV1,
    FfmpegInputPipeProcessV1,
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


def test_bounded_capture_cancellation_terminates_and_reaps_child() -> None:
    started = time.monotonic()
    with pytest.raises(InterruptedError, match="cancellation"):
        BoundedProcessCaptureV1(
            stdout_cap=1024,
            stderr_cap=1024,
            timeout_seconds=30,
            terminate_grace_seconds=0.2,
            cancelled=lambda: True,
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


def test_progress_runner_accepts_ffmpeg_negative_audio_priming_time() -> None:
    parser = FfmpegProgressParser()
    progress = "frame=1\n" "out_time_us=-66667\n" "out_time_ms=-66667\n" "progress=end\n"

    result = run_ffmpeg_progress_v1(
        [
            sys.executable,
            "-c",
            f"import sys; sys.stdout.write({progress!r}); sys.stdout.flush()",
        ],
        parser=parser,
        diagnostic_cap=1024,
        line_cap=8192,
        stall_seconds=1.0,
    )

    assert result.returncode == 0
    assert result.stderr == b""
    assert parser.ended is True


def test_progress_runner_accepts_temporarily_unavailable_ffmpeg_times() -> None:
    parser = FfmpegProgressParser()
    progress = (
        "frame=0\n"
        "total_size=44\n"
        "out_time_us=N/A\n"
        "out_time_ms=N/A\n"
        "progress=continue\n"
        "frame=1\n"
        "out_time_us=0\n"
        "out_time_ms=0\n"
        "progress=end\n"
    )

    result = run_ffmpeg_progress_v1(
        [
            sys.executable,
            "-c",
            f"import sys; sys.stdout.write({progress!r}); sys.stdout.flush()",
        ],
        parser=parser,
        diagnostic_cap=1024,
        line_cap=8192,
        stall_seconds=1.0,
    )

    assert result.returncode == 0
    assert result.stderr == b""
    assert parser.ended is True


def test_progress_runner_cancellation_terminates_and_reaps_child() -> None:
    started = time.monotonic()
    with pytest.raises(InterruptedError, match="cancellation"):
        run_ffmpeg_progress_v1(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            parser=FfmpegProgressParser(),
            diagnostic_cap=1024,
            line_cap=8192,
            stall_seconds=30,
            terminate_grace_seconds=0.2,
            cancelled=lambda: True,
        )
    assert time.monotonic() - started < 3.0


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


def test_streaming_runner_cancellation_terminates_and_reaps_child() -> None:
    started = time.monotonic()
    with pytest.raises(InterruptedError, match="cancellation"):
        run_streaming_byte_count_v1(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            diagnostic_cap=1024,
            stall_seconds=30,
            stream_buffer_bytes=512,
            terminate_grace_seconds=0.2,
            cancelled=lambda: True,
        )
    assert time.monotonic() - started < 3.0


def test_input_pipe_streams_bytes_while_draining_progress_and_diagnostics() -> None:
    parser = FfmpegProgressParser()
    pipe = FfmpegInputPipeProcessV1(
        parser=parser,
        diagnostic_cap=4096,
        line_cap=8192,
        stall_seconds=2.0,
    )
    pipe.start(
        [
            sys.executable,
            "-c",
            "import sys; "
            "sys.stderr.buffer.write(b'e'*2048); sys.stderr.flush(); "
            "payload=sys.stdin.buffer.read(); "
            "sys.stdout.write(f'frame={len(payload)}\\nprogress=end\\n'); "
            "sys.stdout.flush()",
        ]
    )

    pipe.write(b"x" * 131071)
    result = pipe.finish()

    assert result.returncode == 0
    assert result.stderr == b"e" * 2048
    assert parser.maximums["frame"] == 131071
    assert parser.ended is True


def test_input_pipe_output_overflow_terminates_a_blocked_writer() -> None:
    pipe = FfmpegInputPipeProcessV1(
        parser=FfmpegProgressParser(),
        diagnostic_cap=127,
        line_cap=8192,
        stall_seconds=5.0,
        terminate_grace_seconds=0.2,
    )
    pipe.start(
        [
            sys.executable,
            "-c",
            "import sys, time; "
            "sys.stderr.buffer.write(b'e'*4096); sys.stderr.flush(); "
            "time.sleep(30)",
        ]
    )

    started = time.monotonic()
    with pytest.raises(ProcessOutputBudgetError, match="stderr"):
        pipe.write(b"x" * (8 * 1024 * 1024))
    assert time.monotonic() - started < 3.0


def test_input_pipe_cancellation_terminates_a_blocked_writer() -> None:
    started = time.monotonic()
    pipe = FfmpegInputPipeProcessV1(
        parser=FfmpegProgressParser(),
        diagnostic_cap=1024,
        line_cap=8192,
        stall_seconds=30.0,
        terminate_grace_seconds=0.2,
        cancelled=lambda: time.monotonic() - started >= 0.1,
    )
    pipe.start([sys.executable, "-c", "import time; time.sleep(30)"])

    with pytest.raises(InterruptedError, match="cancellation"):
        pipe.write(b"x" * (8 * 1024 * 1024))
    assert time.monotonic() - started < 3.0
