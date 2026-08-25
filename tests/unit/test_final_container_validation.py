from __future__ import annotations

import json
import pytest

from src.depth_surge_3d.processing.video.bounded_process import (
    BoundedProcessResult,
    ProcessOutputBudgetError,
    ProcessTimeoutError,
    ProgressProcessResult,
)
from src.depth_surge_3d.processing.video.encoding_provider import (
    FinalContainerExpectation,
)
from src.depth_surge_3d.processing.video.final_container_validation import (
    FinalContainerDecodeBudgetError,
    FinalContainerDecodeTimeoutError,
    FinalContainerProbeBudgetError,
    FinalContainerProbeTimeoutError,
    FinalContainerProcessError,
    derive_final_probe_timeout_seconds,
    validate_final_container_v4,
)


def _probe_raw() -> bytes:
    return json.dumps(
        {
            "programs": [],
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 8,
                    "height": 4,
                    "pix_fmt": "yuv420p",
                    "sample_aspect_ratio": "1:1",
                    "display_aspect_ratio": "2:1",
                    "avg_frame_rate": "24/1",
                    "nb_read_frames": "1",
                }
            ],
            "stream_groups": [],
        },
        separators=(",", ":"),
    ).encode("ascii")


def _expectation() -> FinalContainerExpectation:
    return FinalContainerExpectation(
        width=8,
        height=4,
        fps="24/1",
        frame_count=1,
        resolved_encoder="libx264",
        audio_expected=False,
        audio_packet_upper_bound=0,
    )


def test_final_container_validator_runs_exact_probe_and_full_decode(tmp_path) -> None:
    video = tmp_path / "candidate.mp4"
    video.write_bytes(b"candidate")
    calls: list[tuple[str, ...]] = []

    def finite_runner(arguments, **caps):
        calls.append(tuple(arguments))
        assert caps == {
            "stdout_cap": 64 * 1024,
            "stderr_cap": 64 * 1024,
            "timeout_seconds": 32,
        }
        return BoundedProcessResult(_probe_raw(), b"", 0)

    def progress_runner(arguments, *, parser, **limits):
        calls.append(tuple(arguments))
        assert limits["diagnostic_cap"] == 256 * 1024
        parser.feed_line(b"frame=1\n")
        parser.feed_line(b"progress=end\n")
        parser.finish()
        return ProgressProcessResult(b"", 0)

    assert validate_final_container_v4(
        video,
        _expectation(),
        finite_runner=finite_runner,
        progress_runner=progress_runner,
    )
    assert calls[0] == (
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        "stream=codec_type,codec_name,width,height,pix_fmt,sample_aspect_ratio,display_aspect_ratio,avg_frame_rate,nb_read_frames",
        "-of",
        "json",
        str(video),
    )
    assert calls[1] == (
        "ffmpeg",
        "-v",
        "error",
        "-xerror",
        "-nostats",
        "-progress",
        "pipe:1",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-f",
        "null",
        "-",
    )


def test_final_probe_timeout_uses_stream_audio_and_byte_work_units() -> None:
    assert derive_final_probe_timeout_seconds(1, _expectation()) == 32
    assert (
        derive_final_probe_timeout_seconds(
            1,
            FinalContainerExpectation(
                width=8,
                height=4,
                fps="24/1",
                frame_count=1,
                resolved_encoder="libx264",
                audio_expected=True,
                audio_packet_upper_bound=1001,
            ),
        )
        == 34
    )
    assert derive_final_probe_timeout_seconds(1001 * 64 * 1024, _expectation()) == 34


def test_final_probe_timeout_is_capped_and_checks_u64_sums() -> None:
    capped = FinalContainerExpectation(
        width=8,
        height=4,
        fps="24/1",
        frame_count=20_000_000,
        resolved_encoder="libx264",
        audio_expected=False,
        audio_packet_upper_bound=0,
    )
    assert derive_final_probe_timeout_seconds(1, capped) == 21_600

    overflowing = FinalContainerExpectation(
        width=8,
        height=4,
        fps="24/1",
        frame_count=2**64 - 1,
        resolved_encoder="libx264",
        audio_expected=True,
        audio_packet_upper_bound=1,
    )
    with pytest.raises(ValueError, match="u64"):
        derive_final_probe_timeout_seconds(1, overflowing)


@pytest.mark.parametrize(
    "result",
    [
        BoundedProcessResult(_probe_raw(), b"warning", 0),
        BoundedProcessResult(_probe_raw(), b"", 1),
    ],
)
def test_final_container_validator_rejects_probe_stderr_or_failure(tmp_path, result) -> None:
    video = tmp_path / "candidate.mp4"
    video.write_bytes(b"candidate")

    with pytest.raises(FinalContainerProcessError):
        validate_final_container_v4(
            video,
            _expectation(),
            finite_runner=lambda *_args, **_kwargs: result,
            progress_runner=lambda *_args, **_kwargs: ProgressProcessResult(b"", 0),
        )


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ProcessTimeoutError("timeout"), FinalContainerProbeTimeoutError),
        (ProcessOutputBudgetError("overflow"), FinalContainerProbeBudgetError),
    ],
)
def test_final_container_maps_probe_process_budget_failures(tmp_path, error, expected) -> None:
    video = tmp_path / "candidate.mp4"
    video.write_bytes(b"candidate")

    def fail_probe(*_args, **_kwargs):
        raise error

    with pytest.raises(expected):
        validate_final_container_v4(video, _expectation(), finite_runner=fail_probe)


def test_final_container_maps_probe_parser_state_overflow(tmp_path) -> None:
    video = tmp_path / "candidate.mp4"
    video.write_bytes(b"candidate")
    raw = b'{"streams":[{"codec_type":"' + b"x" * 129 + b'"}]}'

    with pytest.raises(FinalContainerProbeBudgetError, match="parser-state"):
        validate_final_container_v4(
            video,
            _expectation(),
            finite_runner=lambda *_args, **_kwargs: BoundedProcessResult(raw, b"", 0),
        )


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ProcessTimeoutError("timeout"), FinalContainerDecodeTimeoutError),
        (ProcessOutputBudgetError("overflow"), FinalContainerDecodeBudgetError),
    ],
)
def test_final_container_maps_decode_process_budget_failures(tmp_path, error, expected) -> None:
    video = tmp_path / "candidate.mp4"
    video.write_bytes(b"candidate")

    def fail_decode(*_args, **_kwargs):
        raise error

    with pytest.raises(expected):
        validate_final_container_v4(
            video,
            _expectation(),
            finite_runner=lambda *_args, **_kwargs: BoundedProcessResult(
                _probe_raw(), b"", 0
            ),
            progress_runner=fail_decode,
        )
