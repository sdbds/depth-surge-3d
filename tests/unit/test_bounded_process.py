from __future__ import annotations

import sys
import threading
import time

import pytest

from src.depth_surge_3d.processing.video.bounded_process import (
    BoundedProcessCaptureV1,
    FfmpegInputPipeProcessV1,
    ProcessOutputBudgetError,
    ProcessPipeError,
    ProcessTimeoutError,
    run_ffmpeg_progress_v1,
    run_streaming_byte_count_v1,
)
from src.depth_surge_3d.processing.video.encoding_provider import FfmpegProgressParser


def test_input_pipe_does_not_accept_trailing_bytes_reported_during_final_drain():
    tail_pending = threading.Event()

    class DelayedTailParser(FfmpegProgressParser):
        def feed_line(self, raw_line):
            if self.ended:
                tail_pending.set()
                # A pipe may deliver a child's final bytes after exit was observed.
                assert pipe._stop.wait(5)
            return super().feed_line(raw_line)

    pipe = FfmpegInputPipeProcessV1(
        parser=DelayedTailParser(), diagnostic_cap=256, line_cap=128, stall_seconds=10
    )
    try:
        pipe.start([sys.executable, "-c", "print('progress=end'); print('frame=2')"])
        assert tail_pending.wait(5)
        assert pipe._process.wait(timeout=5) == 0
        with pytest.raises(ProcessPipeError, match="parser failed"):
            pipe.finish()
        assert all(not thread.is_alive() for thread in pipe._drain_threads)
    finally:
        pipe.abort()


@pytest.mark.parametrize(
    ("payload", "error", "reason"),
    [
        (b"frame=3", ProcessPipeError, "unterminated"),
        (b"a" * 60 + b"\nprogress=end\n", ProcessOutputBudgetError, "line"),
        (b"progress=end\nframe=2\n", ProcessPipeError, "parser failed"),
    ],
)
def test_progress_child_rejects_truncated_oversized_or_post_terminal_output(payload, error, reason):
    with pytest.raises(error, match=reason):
        run_ffmpeg_progress_v1(
            [sys.executable, "-c", f"import sys;sys.stdout.buffer.write({payload!r})"],
            parser=FfmpegProgressParser(),
            diagnostic_cap=128,
            line_cap=32,
            stall_seconds=4,
        )


def test_streaming_byte_limit_accepts_the_exact_last_byte_and_rejects_one_more():
    args = dict(diagnostic_cap=0, stream_buffer_bytes=3, maximum_byte_count=17, stall_seconds=5)
    command = [sys.executable, "-c", "import sys;sys.stdout.buffer.write(bytes(int(sys.argv[1])))"]
    assert run_streaming_byte_count_v1([*command, "17"], **args).byte_count == 17
    with pytest.raises(ProcessOutputBudgetError, match="byte count"):
        run_streaming_byte_count_v1([*command, "18"], **args)


def test_input_pipe_validates_state_and_can_finish_after_a_rejected_buffer():
    pipe = FfmpegInputPipeProcessV1(
        parser=FfmpegProgressParser(), diagnostic_cap=64, line_cap=64, stall_seconds=5
    )
    with pytest.raises(RuntimeError, match="not been started"):
        pipe.write(b"x")
    with pytest.raises(ValueError, match="empty"):
        pipe.start([])
    command = [
        sys.executable,
        "-c",
        "import sys; print('progress=end'); sys.exit(sys.stdin.buffer.read()!=b'ok')",
    ]
    pipe.start(command)
    try:
        with pytest.raises(RuntimeError, match="already been started"):
            pipe.start(command)
        with pytest.raises(ValueError, match="contiguous"):
            pipe.write(memoryview(b"abcde")[::2])
        pipe.write(b"ok")
        assert pipe.finish().returncode == 0
        with pytest.raises(RuntimeError, match="already closed"):
            pipe.finish()
    finally:
        pipe.abort()


def test_input_pipe_early_child_exit_reaps_and_closes_all_streams():
    pipe = FfmpegInputPipeProcessV1(
        parser=FfmpegProgressParser(), diagnostic_cap=64, line_cap=64, stall_seconds=5
    )
    pipe.start([sys.executable, "-c", "import sys;sys.exit(7)"])
    try:
        process = pipe._process
        assert process.wait(timeout=5) == 7
        with pytest.raises(ProcessPipeError, match="exited before"):
            pipe.write(b"remaining frame")
        assert process.stdin.closed and process.stdout.closed and process.stderr.closed
    finally:
        pipe.abort()


def test_child_that_ignores_termination_is_still_reaped_on_cancel(tmp_path):
    from src.depth_surge_3d.processing.video import bounded_process

    ready = tmp_path / "ready"
    script = (
        "import sys,time,pathlib,signal;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "pathlib.Path(sys.argv[1]).touch();time.sleep(20)"
    )
    with pytest.raises(InterruptedError, match="cancellation"):
        bounded_process.run_ffmpeg_progress_v1(
            [sys.executable, "-c", script, str(ready)],
            parser=FfmpegProgressParser(),
            diagnostic_cap=64,
            line_cap=64,
            stall_seconds=6,
            terminate_grace_seconds=0.25,
            cancelled=ready.exists,
        )
    assert ready.exists()


@pytest.mark.parametrize("reader", ["progress", "stdin-pipe"])
def test_nonsemantic_ffmpeg_heartbeats_cannot_reset_an_encoding_stall(reader):
    script = (
        "import time;\nfor i in range(1000):\n"
        " print('progress=continue',flush=True)\n time.sleep(.03)"
    )
    options = dict(
        parser=FfmpegProgressParser(),
        diagnostic_cap=128,
        line_cap=128,
        stall_seconds=0.3,
        terminate_grace_seconds=1.0,
    )
    command = [sys.executable, "-c", script]
    if reader == "progress":
        with pytest.raises(ProcessTimeoutError, match="no semantic progress"):
            run_ffmpeg_progress_v1(command, **options)
    else:
        pipe = FfmpegInputPipeProcessV1(**options)
        pipe.start(command)
        try:
            with pytest.raises(ProcessTimeoutError, match="no progress"):
                pipe.finish()
            assert pipe._process.returncode is not None
            assert all(not thread.is_alive() for thread in pipe._drain_threads)
        finally:
            pipe.abort()


@pytest.mark.parametrize(
    "runner",
    [
        lambda: BoundedProcessCaptureV1(stdout_cap=-1, stderr_cap=1, timeout_seconds=1),
        lambda: BoundedProcessCaptureV1(stdout_cap=1, stderr_cap=1, timeout_seconds=0),
        lambda: BoundedProcessCaptureV1(stdout_cap=1, stderr_cap=1, timeout_seconds=1).run([]),
        lambda: run_ffmpeg_progress_v1(
            [], parser=FfmpegProgressParser(), diagnostic_cap=1, line_cap=1, stall_seconds=1
        ),
        lambda: run_ffmpeg_progress_v1(
            ["ffmpeg"], parser=FfmpegProgressParser(), diagnostic_cap=1, line_cap=0, stall_seconds=1
        ),
        lambda: run_ffmpeg_progress_v1(
            ["ffmpeg"], parser=FfmpegProgressParser(), diagnostic_cap=1, line_cap=1, stall_seconds=0
        ),
        lambda: FfmpegInputPipeProcessV1(
            parser=FfmpegProgressParser(), diagnostic_cap=-1, line_cap=1, stall_seconds=1
        ),
        lambda: FfmpegInputPipeProcessV1(
            parser=FfmpegProgressParser(), diagnostic_cap=1, line_cap=1, stall_seconds=0
        ),
        lambda: run_streaming_byte_count_v1(
            [], diagnostic_cap=1, stall_seconds=1, stream_buffer_bytes=1
        ),
        lambda: run_streaming_byte_count_v1(
            ["ffmpeg"], diagnostic_cap=1, stall_seconds=1, stream_buffer_bytes=0
        ),
        lambda: run_streaming_byte_count_v1(
            ["ffmpeg"], diagnostic_cap=1, stall_seconds=0, stream_buffer_bytes=1
        ),
    ],
    ids=[
        "capture-cap",
        "capture-timeout",
        "capture-argv",
        "progress-argv",
        "progress-cap",
        "progress-stall",
        "pipe-cap",
        "pipe-stall",
        "counter-argv",
        "counter-buffer",
        "counter-stall",
    ],
)
def test_refuses_invalid_process_limits_before_spawning_an_external_tool(monkeypatch, runner):
    from src.depth_surge_3d.processing.video import bounded_process

    def forbidden_spawn(*args, **kwargs):
        pytest.fail("invalid limits reached child creation")

    monkeypatch.setattr(bounded_process.subprocess, "Popen", forbidden_spawn)
    with pytest.raises(ValueError, match="cap|deadline|empty"):
        runner()


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
