"""Unit tests for final video encoding."""

import gc
import inspect
import io
import json
import subprocess
import weakref
from pathlib import Path
from pathlib import PurePosixPath
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.core.file_identity import file_sample_fingerprint
from src.depth_surge_3d.processing.video.final_audio import (
    OpenFileIdentityV1,
    SelectedAudioIdentityV1,
)
from src.depth_surge_3d.processing.video.bounded_process import BoundedProcessResult
from src.depth_surge_3d.processing.video.video_encoder import VideoEncoder
from src.depth_surge_3d.processing.frames.source_frame_manifest import (
    write_source_frame_manifest,
)
from src.depth_surge_3d.utils.imaging.png_header import PngHeader
from src.depth_surge_3d.utils.path_utils import generate_output_filename


class _FakeProcess:
    def __init__(self, output: str, returncode: int = 0):
        self.stdout = io.StringIO(output)
        self.returncode = returncode
        self.waited = False
        self.terminated = False
        self.killed = False
        self.communicate_timeouts: list[float | None] = []

    def wait(self, timeout=None):
        self.waited = True
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def communicate(self, timeout=None):
        self.communicate_timeouts.append(timeout)
        self.waited = True
        return ("", None)


class _BrokenProgressStream:
    def __iter__(self):
        yield "diagnostic before read failure\n"
        raise OSError("progress pipe failed")


class _TimeoutProcess(_FakeProcess):
    def communicate(self, timeout=None):
        self.communicate_timeouts.append(timeout)
        if timeout is not None:
            raise subprocess.TimeoutExpired(["ffmpeg"], timeout)
        self.waited = True
        return ("diagnostic after kill\n", None)


def _write_eye_sequence(
    directory: Path,
    indices: list[int],
    *,
    shape: tuple[int, int] = (4, 6),
) -> list[Path]:
    directory.mkdir()
    files = []
    for index in indices:
        path = directory / f"frame_{index:06d}.png"
        image = np.full((*shape, 3), index % 255, dtype=np.uint8)
        assert cv2.imwrite(str(path), image)
        files.append(path)
    return files


def _direct_settings(**overrides):
    settings = {
        "target_fps": "24000/1001",
        "vr_format": "side_by_side",
        "vr_resolution": "custom",
        "per_eye_width": 6,
        "per_eye_height": 4,
        "preserve_audio": False,
        "video_encoder": "libx264",
    }
    settings.update(overrides)
    return settings


def _usable_audio(path: Path) -> SelectedAudioIdentityV1:
    return SelectedAudioIdentityV1(
        selection="original_video",
        selected_source_path=path,
        source_open_identity=OpenFileIdentityV1("windows", "0" * 16, "1" * 32),
        source_byte_count=5,
        source_raw_sha256="a" * 64,
        resolved_clip_arguments=("-ss", "2"),
        disposition="usable",
        sample_rate=48_000,
        channels=2,
        decoded_pcm_bytes=384_000,
        audio_end_sample=96_096,
    )


@pytest.mark.parametrize("start", [0, 1, 1_000_000])
def test_validate_direct_sequence_accepts_canonical_consecutive_names(tmp_path, start):
    left = _write_eye_sequence(tmp_path / "left", [start, start + 1])
    right = _write_eye_sequence(tmp_path / "right", [start, start + 1])
    header = PngHeader(width=6, height=4, bit_depth=8, color_type=2, channels=3)

    with patch(
        "src.depth_surge_3d.processing.video.video_encoder.read_png_header",
        side_effect=[header, header],
    ) as read_header:
        sequence = VideoEncoder()._validate_direct_stereo_sequence(
            list(reversed(left)), list(reversed(right)), 2
        )

    assert sequence.start_number == start
    assert sequence.frame_count == 2
    assert sequence.left_pattern == tmp_path / "left" / "frame_%06d.png"
    assert sequence.right_pattern == tmp_path / "right" / "frame_%06d.png"
    assert sequence.left_header == header
    assert sequence.right_header == header
    assert read_header.call_count == 2
    assert [call.args for call in read_header.call_args_list] == [
        (tmp_path / "left" / f"frame_{start:06d}.png",),
        (tmp_path / "right" / f"frame_{start:06d}.png",),
    ]


@pytest.mark.parametrize(
    "left_names,right_names,total",
    [
        ([], [], 0),
        (["frame_000001.png"], ["frame_000002.png"], 1),
        (
            ["frame_000001.png", "frame_000003.png"],
            ["frame_000001.png", "frame_000003.png"],
            2,
        ),
        (["frame_1.png"], ["frame_1.png"], 1),
        (["frame_0000001.png"], ["frame_0000001.png"], 1),
    ],
)
def test_validate_direct_sequence_rejects_invalid_manifests(
    tmp_path, left_names, right_names, total
):
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    left_files = []
    right_files = []
    for directory, names, destination in (
        (left_dir, left_names, left_files),
        (right_dir, right_names, right_files),
    ):
        for name in names:
            path = directory / name
            assert cv2.imwrite(str(path), np.zeros((4, 6, 3), np.uint8))
            destination.append(path)

    with pytest.raises(ValueError):
        VideoEncoder()._validate_direct_stereo_sequence(left_files, right_files, total)


def test_validate_direct_sequence_rejects_multiple_parent_directories(tmp_path):
    left = _write_eye_sequence(tmp_path / "left", [1])
    left.extend(_write_eye_sequence(tmp_path / "other_left", [2]))
    right = _write_eye_sequence(tmp_path / "right", [1, 2])

    with pytest.raises(ValueError):
        VideoEncoder()._validate_direct_stereo_sequence(left, right, 2)


def test_validate_direct_sequence_rejects_positive_total_mismatch(tmp_path):
    left = _write_eye_sequence(tmp_path / "left", [1])
    right = _write_eye_sequence(tmp_path / "right", [1])

    with pytest.raises(ValueError):
        VideoEncoder()._validate_direct_stereo_sequence(left, right, 2)


def test_validate_direct_sequence_rejects_unreadable_first_png(tmp_path):
    left = _write_eye_sequence(tmp_path / "left", [1])
    right = _write_eye_sequence(tmp_path / "right", [1])

    with patch(
        "src.depth_surge_3d.processing.video.video_encoder.read_png_header",
        return_value=None,
    ):
        with pytest.raises(ValueError):
            VideoEncoder()._validate_direct_stereo_sequence(left, right, 1)


def test_direct_command_omits_scale_for_exact_size_and_uses_hstack(tmp_path):
    left = _write_eye_sequence(tmp_path / "left", [1, 2])
    right = _write_eye_sequence(tmp_path / "right", [1, 2])
    encoder = VideoEncoder()
    sequence = encoder._validate_direct_stereo_sequence(left, right, 2)

    command = encoder._build_direct_stereo_command(
        sequence,
        tmp_path / ".output.direct.tmp.mp4",
        "source.mp4",
        _direct_settings(),
    )

    graph = command[command.index("-filter_complex") + 1]
    assert graph == (
        "[0:v]setsar=1[left];[1:v]setsar=1[right];"
        "[left][right]hstack=inputs=2:shortest=1[stacked];"
        "[stacked]setsar=1[vr]"
    )
    assert "scale=" not in graph
    assert command.count("-framerate") == 2
    assert command.count("24000/1001") == 2
    assert command.count("-start_number") == 2
    starts = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "-start_number"
    ]
    assert starts == ["1", "1"]
    assert command[command.index("-frames:v") + 1] == "2"
    assert command[command.index("-map") + 1] == "[vr]"
    assert command[:4] == ["ffmpeg", "-y", "-loglevel", "error"]
    assert command[4:6] == ["-nostats", "-framerate"]
    assert command[-9:] == [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-preset",
        "medium",
        str(tmp_path / ".output.direct.tmp.mp4"),
    ]


def test_direct_command_scales_both_eyes_and_uses_vstack(tmp_path):
    left = _write_eye_sequence(tmp_path / "left", [0], shape=(6, 8))
    right = _write_eye_sequence(tmp_path / "right", [0], shape=(5, 8))
    encoder = VideoEncoder()
    sequence = encoder._validate_direct_stereo_sequence(left, right, 1)

    command = encoder._build_direct_stereo_command(
        sequence,
        tmp_path / ".output.direct.tmp.mp4",
        "source.mp4",
        _direct_settings(vr_format="over_under", per_eye_width=8, per_eye_height=6),
    )

    graph = command[command.index("-filter_complex") + 1]
    assert graph.count("scale=8:6:flags=bicubic+accurate_rnd,setsar=1") == 2
    assert "[left][right]vstack=inputs=2:shortest=1[stacked]" in graph
    assert graph.endswith("[stacked]setsar=1[vr]")


def test_direct_command_maps_audio_from_third_input(tmp_path):
    left = _write_eye_sequence(tmp_path / "left", [1])
    right = _write_eye_sequence(tmp_path / "right", [1])
    (tmp_path / "original_audio.flac").write_bytes(b"audio")
    encoder = VideoEncoder()
    sequence = encoder._validate_direct_stereo_sequence(left, right, 1)

    command = encoder._build_direct_stereo_command(
        sequence,
        tmp_path / ".output.direct.tmp.mp4",
        "source.mp4",
        _direct_settings(preserve_audio=True),
    )

    audio_map = command.index("-map", command.index("-map") + 1)
    assert command[audio_map : audio_map + 5] == [
        "-map",
        "2:a:0?",
        "-c:a",
        "aac",
        "-shortest",
    ]
    assert max(
        index for index, value in enumerate(command) if value == "-i"
    ) < command.index("-filter_complex")


def test_schema5_direct_command_uses_preflighted_audio_without_shortest(tmp_path):
    left = _write_eye_sequence(tmp_path / "left", [1])
    right = _write_eye_sequence(tmp_path / "right", [1])
    source = tmp_path / "source.mp4"
    source.write_bytes(b"audio")
    encoder = VideoEncoder()
    sequence = encoder._validate_direct_stereo_sequence(left, right, 1)

    command = encoder._build_direct_stereo_command(
        sequence,
        tmp_path / ".output.tmp.mp4",
        str(source),
        _direct_settings(preserve_audio=True),
        selected_audio=_usable_audio(source.resolve()),
    )

    assert command.count("-i") == 3
    assert command[command.index("-ss") : command.index("-filter_complex")] == [
        "-ss",
        "2",
        "-i",
        str(source.resolve()),
    ]
    audio_map = command.index("-map", command.index("-map") + 1)
    assert command[audio_map : audio_map + 6] == [
        "-map",
        "2:a:0",
        "-filter:a",
        "atrim=end_sample=96096,asetpts=PTS-STARTPTS",
        "-c:a",
        "aac",
    ]
    assert "-shortest" not in command


def test_normalized_arguments_use_frozen_audio_identity_without_rehash(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"audio")
    runtime = tmp_path / ".depth-surge-final-v4-a.tmp.mp4"
    command = [
        "ffmpeg",
        "-i",
        "frames/frame_%06d.png",
        "-i",
        str(source.resolve()),
        str(runtime),
    ]

    with patch.object(VideoEncoder, "_hash_input") as hash_input:
        normalized = VideoEncoder._normalize_publication_arguments(
            command=command,
            image_inputs={"frames/frame_%06d.png": "frames"},
            selected_audio=_usable_audio(source.resolve()),
            runtime_output=runtime,
            input_stage_fingerprint="b" * 64,
            final_relative_path="output.mp4",
        )

    assert normalized[4] == f"@audio:sha256={'a' * 64}:bytes=5:stream=a:0"
    hash_input.assert_not_called()


@pytest.mark.parametrize(
    "settings",
    [
        _direct_settings(per_eye_width=0),
        _direct_settings(per_eye_height=-1),
        _direct_settings(vr_format="anaglyph"),
        _direct_settings(vr_format=None),
    ],
)
def test_direct_command_rejects_invalid_settings(tmp_path, settings):
    left = _write_eye_sequence(tmp_path / "left", [1])
    right = _write_eye_sequence(tmp_path / "right", [1])
    encoder = VideoEncoder()
    sequence = encoder._validate_direct_stereo_sequence(left, right, 1)

    with pytest.raises(ValueError):
        encoder._build_direct_stereo_command(
            sequence, tmp_path / ".output.direct.tmp.mp4", "source.mp4", settings
        )


def test_direct_command_omits_audio_arguments_when_disabled(tmp_path):
    left = _write_eye_sequence(tmp_path / "left", [1])
    right = _write_eye_sequence(tmp_path / "right", [1])
    encoder = VideoEncoder()
    sequence = encoder._validate_direct_stereo_sequence(left, right, 1)

    command = encoder._build_direct_stereo_command(
        sequence,
        tmp_path / ".output.direct.tmp.mp4",
        "source.mp4",
        _direct_settings(preserve_audio=False),
    )

    assert command.count("-i") == 2
    assert "-c:a" not in command
    assert "-shortest" not in command


def test_direct_command_uses_explicit_nvenc_cq18_quality_contract(tmp_path):
    left = _write_eye_sequence(tmp_path / "left", [1])
    right = _write_eye_sequence(tmp_path / "right", [1])
    encoder = VideoEncoder()
    sequence = encoder._validate_direct_stereo_sequence(left, right, 1)

    with patch.object(encoder, "_check_nvenc_available", return_value=True):
        command = encoder._build_direct_stereo_command(
            sequence,
            tmp_path / ".output.direct.tmp.mp4",
            "source.mp4",
            _direct_settings(video_encoder="nvenc"),
        )

    assert command[-15:] == [
        "-c:v",
        "hevc_nvenc",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "p7",
        "-tune",
        "hq",
        "-rc",
        "vbr",
        "-cq",
        "18",
        "-b:v",
        "0",
        str(tmp_path / ".output.direct.tmp.mp4"),
    ]


def test_nvenc_capability_probe_is_bounded_and_cancellable() -> None:
    def cancelled() -> bool:
        return False
    with patch(
        "src.depth_surge_3d.processing.video.video_encoder.BoundedProcessCaptureV1"
    ) as capture:
        capture.return_value.run.return_value = BoundedProcessResult(
            b" V..... hevc_nvenc NVIDIA NVENC hevc encoder\n",
            b"",
            0,
        )

        assert VideoEncoder()._check_nvenc_available(cancelled=cancelled)

    capture.assert_called_once_with(
        stdout_cap=1024 * 1024,
        stderr_cap=64 * 1024,
        timeout_seconds=10.0,
        cancelled=cancelled,
    )
    capture.return_value.run.assert_called_once_with(
        ("ffmpeg", "-hide_banner", "-encoders")
    )


def test_direct_runner_merges_stderr_tracks_frames_and_waits():
    process = _FakeProcess("frame=1\nnoise\nframe=2\n")
    tracker = Mock()

    with patch(
        "src.depth_surge_3d.processing.video.video_encoder.subprocess.Popen",
        return_value=process,
    ) as popen:
        returncode, diagnostics = VideoEncoder()._run_ffmpeg_with_progress(
            ["ffmpeg"], 2, tracker
        )

    assert returncode == 0
    assert diagnostics == ("noise",)
    assert process.waited is True
    assert [
        call.kwargs["frame_num"] for call in tracker.update_progress.call_args_list
    ] == [1, 2]
    assert popen.call_args.kwargs == {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
    }


def test_direct_runner_retains_only_last_fifty_diagnostic_lines():
    process = _FakeProcess("".join(f"line-{index}\n" for index in range(75)), 1)

    with patch(
        "src.depth_surge_3d.processing.video.video_encoder.subprocess.Popen",
        return_value=process,
    ):
        returncode, diagnostics = VideoEncoder()._run_ffmpeg_with_progress(
            ["ffmpeg"], 1, None
        )

    assert returncode == 1
    assert len(diagnostics) == 50
    assert diagnostics[0] == "line-25"
    assert diagnostics[-1] == "line-74"


def test_direct_runner_ignores_tracker_errors_and_uses_wait_returncode():
    process = _FakeProcess("frame=4\n", returncode=7)
    tracker = Mock()
    tracker.update_progress.side_effect = RuntimeError("tracker unavailable")

    with patch(
        "src.depth_surge_3d.processing.video.video_encoder.subprocess.Popen",
        return_value=process,
    ):
        returncode, diagnostics = VideoEncoder()._run_ffmpeg_with_progress(
            ["ffmpeg"], 4, tracker
        )

    assert returncode == 7
    assert diagnostics == ()
    assert process.waited is True


def test_direct_runner_terminates_drains_and_reaps_after_read_error():
    process = _FakeProcess("")
    process.stdout = _BrokenProgressStream()

    with patch(
        "src.depth_surge_3d.processing.video.video_encoder.subprocess.Popen",
        return_value=process,
    ):
        with pytest.raises(OSError, match="progress pipe failed"):
            VideoEncoder()._run_ffmpeg_with_progress(["ffmpeg"], 1, None)

    assert process.terminated is True
    assert process.communicate_timeouts == [5]
    assert process.waited is True


def test_direct_runner_surfaces_diagnostic_tail_after_read_error(capsys):
    process = _FakeProcess("")
    process.stdout = _BrokenProgressStream()

    with patch(
        "src.depth_surge_3d.processing.video.video_encoder.subprocess.Popen",
        return_value=process,
    ):
        with pytest.raises(OSError, match="progress pipe failed"):
            VideoEncoder()._run_ffmpeg_with_progress(["ffmpeg"], 1, None)

    output = capsys.readouterr().out
    assert "FFmpeg diagnostic tail:" in output
    assert "diagnostic before read failure" in output


def test_direct_runner_kills_and_reaps_when_bounded_drain_times_out():
    process = _TimeoutProcess("")
    process.stdout = _BrokenProgressStream()

    with patch(
        "src.depth_surge_3d.processing.video.video_encoder.subprocess.Popen",
        return_value=process,
    ):
        with pytest.raises(OSError, match="progress pipe failed"):
            VideoEncoder()._run_ffmpeg_with_progress(["ffmpeg"], 1, None)

    assert process.terminated is True
    assert process.killed is True
    assert process.communicate_timeouts == [5, None]
    assert process.waited is True


def test_direct_create_video_atomically_replaces_final_output(tmp_path):
    left = _write_eye_sequence(tmp_path / "left", [1])
    right = _write_eye_sequence(tmp_path / "right", [1])
    settings = _direct_settings()
    final = tmp_path / generate_output_filename(
        "source.mp4", settings["vr_format"], settings["vr_resolution"]
    )
    temporary = tmp_path / f".{final.stem}.direct.tmp.mp4"
    unrelated = tmp_path / f".{final.stem}.direct.tmp.mp4.backup"
    final.write_bytes(b"old-valid-video")
    temporary.write_bytes(b"stale-partial-video")
    unrelated.write_bytes(b"keep-me")
    left_before = left[0].read_bytes()
    right_before = right[0].read_bytes()

    def launch(command, **_kwargs):
        assert Path(command[-1]) == temporary
        Path(command[-1]).write_bytes(b"new-valid-video")
        return _FakeProcess("frame=1\nprogress=end\n")

    with (
        patch(
            "src.depth_surge_3d.processing.video.video_encoder.verify_ffmpeg_installation",
            return_value=True,
        ),
        patch(
            "src.depth_surge_3d.processing.video.video_encoder.subprocess.Popen",
            side_effect=launch,
        ),
    ):
        assert VideoEncoder().create_video_from_stereo_sequences(
            left,
            right,
            tmp_path,
            "source.mp4",
            settings,
            total_frames=1,
        )

    assert final.read_bytes() == b"new-valid-video"
    assert not temporary.exists()
    assert unrelated.read_bytes() == b"keep-me"
    assert left[0].read_bytes() == left_before
    assert right[0].read_bytes() == right_before


def test_direct_create_video_failure_preserves_old_final_and_removes_temp(
    tmp_path, capsys
):
    left = _write_eye_sequence(tmp_path / "left", [1])
    right = _write_eye_sequence(tmp_path / "right", [1])
    settings = _direct_settings()
    final = tmp_path / generate_output_filename(
        "source.mp4", settings["vr_format"], settings["vr_resolution"]
    )
    temporary = tmp_path / f".{final.stem}.direct.tmp.mp4"
    final.write_bytes(b"old-valid-video")
    left_before = left[0].read_bytes()
    right_before = right[0].read_bytes()

    def launch(command, **_kwargs):
        Path(command[-1]).write_bytes(b"partial")
        output = "".join(f"line-{index}\n" for index in range(75))
        return _FakeProcess(output, returncode=1)

    with (
        patch(
            "src.depth_surge_3d.processing.video.video_encoder.verify_ffmpeg_installation",
            return_value=True,
        ),
        patch(
            "src.depth_surge_3d.processing.video.video_encoder.subprocess.Popen",
            side_effect=launch,
        ),
    ):
        assert not VideoEncoder().create_video_from_stereo_sequences(
            left,
            right,
            tmp_path,
            "source.mp4",
            settings,
            total_frames=1,
        )

    output = capsys.readouterr().out
    assert "return code 1" in output
    assert "line-25" in output
    assert "line-74" in output
    assert "line-24" not in output
    assert final.read_bytes() == b"old-valid-video"
    assert not temporary.exists()
    assert left[0].read_bytes() == left_before
    assert right[0].read_bytes() == right_before


def test_direct_cancellation_reaps_ffmpeg_and_preserves_inputs_and_old_final(tmp_path):
    import app as web_app

    left = _write_eye_sequence(tmp_path / "left", [1])
    right = _write_eye_sequence(tmp_path / "right", [1])
    settings = _direct_settings()
    final = tmp_path / generate_output_filename(
        "source.mp4", settings["vr_format"], settings["vr_resolution"]
    )
    temporary = tmp_path / f".{final.stem}.direct.tmp.mp4"
    final.write_bytes(b"old-valid-video")
    left_before = left[0].read_bytes()
    right_before = right[0].read_bytes()
    process = _FakeProcess("frame=1\n")

    def launch(command, **_kwargs):
        Path(command[-1]).write_bytes(b"partial")
        return process

    callback = web_app.ProgressCallback("test-session", total_frames=1)
    web_app.current_processing["stop_requested"] = True
    try:
        with (
            patch(
                "src.depth_surge_3d.processing.video.video_encoder.verify_ffmpeg_installation",
                return_value=True,
            ),
            patch(
                "src.depth_surge_3d.processing.video.video_encoder.subprocess.Popen",
                side_effect=launch,
            ),
        ):
            with pytest.raises(InterruptedError, match="stopped by user request"):
                VideoEncoder().create_video_from_stereo_sequences(
                    left,
                    right,
                    tmp_path,
                    "source.mp4",
                    settings,
                    total_frames=1,
                    progress_tracker=callback,
                )
    finally:
        web_app.current_processing["stop_requested"] = False

    assert process.terminated is True
    assert process.communicate_timeouts == [5]
    assert process.waited is True
    assert final.read_bytes() == b"old-valid-video"
    assert not temporary.exists()
    assert left[0].read_bytes() == left_before
    assert right[0].read_bytes() == right_before


def test_direct_create_video_validates_before_launch_and_cleans_stale_temp(tmp_path):
    settings = _direct_settings()
    final = tmp_path / generate_output_filename(
        "source.mp4", settings["vr_format"], settings["vr_resolution"]
    )
    temporary = tmp_path / f".{final.stem}.direct.tmp.mp4"
    final.write_bytes(b"old-valid-video")
    temporary.write_bytes(b"stale-partial-video")

    with (
        patch(
            "src.depth_surge_3d.processing.video.video_encoder.verify_ffmpeg_installation",
            return_value=True,
        ),
        patch(
            "src.depth_surge_3d.processing.video.video_encoder.subprocess.Popen"
        ) as popen,
    ):
        assert not VideoEncoder().create_video_from_stereo_sequences(
            [],
            [],
            tmp_path,
            "source.mp4",
            settings,
            total_frames=0,
        )

    popen.assert_not_called()
    assert final.read_bytes() == b"old-valid-video"
    assert not temporary.exists()


def test_direct_create_video_preflight_failure_cleans_only_exact_stale_temp(tmp_path):
    left = _write_eye_sequence(tmp_path / "left", [1])
    right = _write_eye_sequence(tmp_path / "right", [1])
    settings = _direct_settings()
    final = tmp_path / generate_output_filename(
        "source.mp4", settings["vr_format"], settings["vr_resolution"]
    )
    temporary = tmp_path / f".{final.stem}.direct.tmp.mp4"
    unrelated = tmp_path / f".{final.stem}.direct.tmp.mp4.backup"
    final.write_bytes(b"old-valid-video")
    temporary.write_bytes(b"stale-partial-video")
    unrelated.write_bytes(b"keep-me")
    left_before = left[0].read_bytes()
    right_before = right[0].read_bytes()

    with (
        patch(
            "src.depth_surge_3d.processing.video.video_encoder.verify_ffmpeg_installation",
            return_value=False,
        ),
        patch(
            "src.depth_surge_3d.processing.video.video_encoder.subprocess.Popen"
        ) as popen,
    ):
        assert not VideoEncoder().create_video_from_stereo_sequences(
            left,
            right,
            tmp_path,
            "source.mp4",
            settings,
            total_frames=1,
        )

    popen.assert_not_called()
    assert final.read_bytes() == b"old-valid-video"
    assert not temporary.exists()
    assert unrelated.read_bytes() == b"keep-me"
    assert left[0].read_bytes() == left_before
    assert right[0].read_bytes() == right_before


def test_direct_create_video_cleans_temp_after_launch_exception(tmp_path):
    left = _write_eye_sequence(tmp_path / "left", [1])
    right = _write_eye_sequence(tmp_path / "right", [1])
    settings = _direct_settings()
    final = tmp_path / generate_output_filename(
        "source.mp4", settings["vr_format"], settings["vr_resolution"]
    )
    temporary = tmp_path / f".{final.stem}.direct.tmp.mp4"
    final.write_bytes(b"old-valid-video")

    with (
        patch(
            "src.depth_surge_3d.processing.video.video_encoder.verify_ffmpeg_installation",
            return_value=True,
        ),
        patch(
            "src.depth_surge_3d.processing.video.video_encoder.subprocess.Popen",
            side_effect=OSError("launch failed"),
        ),
    ):
        assert not VideoEncoder().create_video_from_stereo_sequences(
            left,
            right,
            tmp_path,
            "source.mp4",
            settings,
            total_frames=1,
        )

    assert final.read_bytes() == b"old-valid-video"
    assert not temporary.exists()


@pytest.mark.parametrize("output_state", ["missing", "empty"])
def test_direct_create_video_reports_output_validation_failure(
    tmp_path, capsys, output_state
):
    left = _write_eye_sequence(tmp_path / "left", [1])
    right = _write_eye_sequence(tmp_path / "right", [1])
    settings = _direct_settings()
    final = tmp_path / generate_output_filename(
        "source.mp4", settings["vr_format"], settings["vr_resolution"]
    )
    temporary = tmp_path / f".{final.stem}.direct.tmp.mp4"
    final.write_bytes(b"old-valid-video")

    def launch(command, **_kwargs):
        if output_state == "empty":
            Path(command[-1]).touch()
        return _FakeProcess("frame=1\ndiagnostic from zero exit\n")

    with (
        patch(
            "src.depth_surge_3d.processing.video.video_encoder.verify_ffmpeg_installation",
            return_value=True,
        ),
        patch(
            "src.depth_surge_3d.processing.video.video_encoder.subprocess.Popen",
            side_effect=launch,
        ),
    ):
        assert not VideoEncoder().create_video_from_stereo_sequences(
            left,
            right,
            tmp_path,
            "source.mp4",
            settings,
            total_frames=1,
        )

    assert final.read_bytes() == b"old-valid-video"
    assert not temporary.exists()
    output = capsys.readouterr().out
    assert "Direct FFmpeg output validation failed" in output
    assert "temporary output is missing or empty" in output
    assert "return code 0" not in output
    assert "diagnostic from zero exit" in output


def test_direct_create_video_cleans_temp_when_atomic_replace_fails(tmp_path):
    left = _write_eye_sequence(tmp_path / "left", [1])
    right = _write_eye_sequence(tmp_path / "right", [1])
    settings = _direct_settings()
    final = tmp_path / generate_output_filename(
        "source.mp4", settings["vr_format"], settings["vr_resolution"]
    )
    temporary = tmp_path / f".{final.stem}.direct.tmp.mp4"
    final.write_bytes(b"old-valid-video")

    def launch(command, **_kwargs):
        Path(command[-1]).write_bytes(b"new-valid-video")
        return _FakeProcess("frame=1\n")

    with (
        patch(
            "src.depth_surge_3d.processing.video.video_encoder.verify_ffmpeg_installation",
            return_value=True,
        ),
        patch(
            "src.depth_surge_3d.processing.video.video_encoder.subprocess.Popen",
            side_effect=launch,
        ),
        patch.object(Path, "replace", side_effect=OSError("replace failed")) as replace,
    ):
        assert not VideoEncoder().create_video_from_stereo_sequences(
            left,
            right,
            tmp_path,
            "source.mp4",
            settings,
            total_frames=1,
        )

    replace.assert_called_once_with(final)
    assert final.read_bytes() == b"old-valid-video"
    assert not temporary.exists()


def test_extract_frames_writes_content_bound_stage_manifest(tmp_path):
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"source-video")
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    def extract(_command, **_kwargs):
        for index in range(1, 3):
            frame = np.full((2, 3, 3), index, dtype=np.uint8)
            assert cv2.imwrite(str(frames_dir / f"frame_{index:06d}.png"), frame)
        return MagicMock(returncode=0)

    with patch(
        "src.depth_surge_3d.processing.video.video_encoder.subprocess.run",
        side_effect=extract,
    ):
        frame_files = VideoEncoder().extract_frames(
            str(source_video),
            {"base": tmp_path, "frames": frames_dir},
            {"frame_count": 2, "fps": 30.0},
            {"start_time": None, "end_time": None},
        )

    metadata = json.loads((frames_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["frame_names"] == [path.name for path in frame_files]
    assert metadata["source_video_fingerprint"] == file_sample_fingerprint(source_video)
    assert metadata["source_frame_fingerprint"]
    assert metadata["fingerprint"]


def test_extract_frames_cuda_command_allows_ffmpeg_to_negotiate_10_bit_frames(tmp_path):
    """CUDA decoding must not force 10-bit hardware frames through 8-bit NV12."""
    source_video = tmp_path / "source.mkv"
    source_video.write_bytes(b"10-bit-av1")
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    commands = []

    def extract(command, **_kwargs):
        commands.append(command)
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        assert cv2.imwrite(str(frames_dir / "frame_000001.png"), frame)
        return MagicMock(returncode=0)

    with patch(
        "src.depth_surge_3d.processing.video.video_encoder.subprocess.run",
        side_effect=extract,
    ):
        frame_files = VideoEncoder().extract_frames(
            str(source_video),
            {"base": tmp_path, "frames": frames_dir},
            {"frame_count": 1, "fps": 30.0},
            {"start_time": None, "end_time": None},
        )

    assert len(frame_files) == 1
    assert len(commands) == 1
    cuda_command = commands[0]
    assert cuda_command[cuda_command.index("-hwaccel") + 1] == "cuda"
    assert "-hwaccel_output_format" not in cuda_command
    assert not any("hwdownload" in argument for argument in cuda_command)
    assert "nv12" not in cuda_command


def test_extract_frames_uses_passthrough_fps_mode_for_cuda_and_cpu_fallback(tmp_path):
    """Both extraction paths must use FFmpeg's supported per-output FPS mode."""
    source_video = tmp_path / "source.mkv"
    source_video.write_bytes(b"source-video")
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    commands = []

    def extract(command, **_kwargs):
        commands.append(command)
        if len(commands) == 1:
            return MagicMock(returncode=1)
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        assert cv2.imwrite(str(frames_dir / "frame_000001.png"), frame)
        return MagicMock(returncode=0)

    with patch(
        "src.depth_surge_3d.processing.video.video_encoder.subprocess.run",
        side_effect=extract,
    ):
        frame_files = VideoEncoder().extract_frames(
            str(source_video),
            {"base": tmp_path, "frames": frames_dir},
            {"frame_count": 1, "fps": 30.0},
            {"start_time": None, "end_time": None},
        )

    assert len(frame_files) == 1
    assert len(commands) == 2
    for command in commands:
        assert "-fps_mode" in command
        assert command[command.index("-fps_mode") + 1] == "passthrough"
        assert "-vsync" not in command


def test_create_video_preserves_fractional_source_fps_when_target_is_original(tmp_path):
    """Original FPS must use the source stream rate instead of the 30 FPS fallback."""
    encoder = VideoEncoder()
    settings = {
        "target_fps": None,
        "source_fps": 23.98,
        "vr_format": "side_by_side",
        "vr_resolution": "16x9-1080p",
        "preserve_audio": False,
        "video_encoder": "libx265",
    }
    probe_info = {
        "streams": [
            {
                "codec_type": "video",
                "avg_frame_rate": "24000/1001",
                "r_frame_rate": "24000/1001",
            }
        ]
    }
    completed = MagicMock(returncode=0)

    with (
        patch(
            "src.depth_surge_3d.processing.video.video_encoder.verify_ffmpeg_installation",
            return_value=True,
        ),
        patch(
            "src.depth_surge_3d.processing.video.video_encoder.get_video_info_ffprobe",
            return_value=probe_info,
        ),
        patch(
            "src.depth_surge_3d.processing.video.video_encoder.subprocess.run",
            return_value=completed,
        ) as run,
    ):
        result = encoder.create_video(
            tmp_path / "frames",
            tmp_path,
            str(tmp_path / "source.mkv"),
            settings,
        )

    assert result is True
    command = run.call_args.args[0]
    frame_rate_index = command.index("-framerate") + 1
    assert command[frame_rate_index] == "24000/1001"


def test_create_video_trims_preextracted_audio_to_selected_clip(tmp_path):
    """A clipped frame sequence must not be muxed with audio from source time zero."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    audio_file = tmp_path / "original_audio.flac"
    audio_file.write_bytes(b"full-source-audio")
    settings = {
        "target_fps": 24,
        "vr_format": "side_by_side",
        "vr_resolution": "16x9-1080p",
        "preserve_audio": True,
        "video_encoder": "libx265",
        "start_time": "3:16",
        "end_time": "3:20",
    }

    with (
        patch(
            "src.depth_surge_3d.processing.video.video_encoder.verify_ffmpeg_installation",
            return_value=True,
        ),
        patch(
            "src.depth_surge_3d.processing.video.video_encoder.subprocess.run",
            return_value=MagicMock(returncode=0),
        ) as run,
    ):
        result = VideoEncoder().create_video(
            frames_dir,
            tmp_path,
            str(tmp_path / "source.mkv"),
            settings,
        )

    assert result is True
    command = run.call_args.args[0]
    audio_path_index = command.index(str(audio_file))
    assert command[audio_path_index - 5 : audio_path_index + 1] == [
        "-ss",
        "196",
        "-t",
        "4",
        "-i",
        str(audio_file),
    ]


def test_extract_frames_reuses_complete_existing_sequence(tmp_path):
    """Resuming a stopped job does not decode a complete frame sequence again."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    expected = []
    for i in range(1, 4):
        path = frames_dir / f"frame_{i:06d}.png"
        cv2.imwrite(str(path), np.zeros((2, 2, 3), dtype=np.uint8))
        expected.append(path)
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"source-video")
    write_source_frame_manifest(expected, source_video)

    with patch(
        "src.depth_surge_3d.processing.video.video_encoder.subprocess.run"
    ) as run:
        result = VideoEncoder().extract_frames(
            str(source_video),
            {"base": tmp_path, "frames": frames_dir},
            {"frame_count": 3, "fps": 30.0},
            {"start_time": None, "end_time": None},
        )

    assert result == expected
    run.assert_not_called()


def test_extract_frames_tolerates_small_vfr_metadata_count_drift(tmp_path):
    """Rounded FPS metadata must not force a complete VFR sequence to be decoded again."""
    frames_dir = tmp_path / "frames"
    depth_dir = tmp_path / "depth"
    frames_dir.mkdir()
    depth_dir.mkdir()
    expected = []
    for i in range(1, 100):
        path = frames_dir / f"frame_{i:06d}.png"
        path.write_bytes(b"png")
        expected.append(path)
    (depth_dir / "frame_000001.png").write_bytes(b"depth")
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"source-video")
    write_source_frame_manifest(expected, source_video)

    with patch(
        "src.depth_surge_3d.processing.video.video_encoder.subprocess.run"
    ) as run:
        result = VideoEncoder().extract_frames(
            str(source_video),
            {"base": tmp_path, "frames": frames_dir, "disparity_maps": depth_dir},
            {"frame_count": 100, "fps": 23.98},
            {"start_time": None, "end_time": None},
        )

    assert result == expected
    run.assert_not_called()


def test_metric_geometry_npz_counts_as_downstream_progress(tmp_path: Path) -> None:
    metric_dir = tmp_path / "metric"
    metric_dir.mkdir()
    (metric_dir / "frame_000001.npz").write_bytes(b"metric")

    assert VideoEncoder._has_downstream_progress({"metric_geometry": metric_dir})


def test_extract_frames_reextracts_approximate_count_without_downstream_progress(
    tmp_path,
):
    """A near-complete frame count alone cannot prove that extraction finished."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    stale_frames = []
    for i in range(1, 100):
        path = frames_dir / f"frame_{i:06d}.png"
        path.write_bytes(b"partial")
        stale_frames.append(path)

    completed = MagicMock(returncode=0)
    with patch(
        "src.depth_surge_3d.processing.video.video_encoder.subprocess.run",
        return_value=completed,
    ) as run:
        result = VideoEncoder().extract_frames(
            "source.mp4",
            {"base": tmp_path, "frames": frames_dir},
            {"frame_count": 100, "fps": 23.98},
            {"start_time": None, "end_time": None},
        )

    assert run.call_count == 1
    assert result == []
    assert not any(path.exists() for path in stale_frames)


def test_extract_frames_does_not_reuse_empty_sequence_with_stale_downstream_files(
    tmp_path,
):
    """Downstream leftovers cannot turn an empty extraction directory into a cache hit."""
    frames_dir = tmp_path / "frames"
    depth_dir = tmp_path / "depth"
    frames_dir.mkdir()
    depth_dir.mkdir()
    (depth_dir / "frame_000001.png").write_bytes(b"stale")

    completed = MagicMock(returncode=0)
    with patch(
        "src.depth_surge_3d.processing.video.video_encoder.subprocess.run",
        return_value=completed,
    ) as run:
        result = VideoEncoder().extract_frames(
            "source.mp4",
            {"base": tmp_path, "frames": frames_dir, "disparity_maps": depth_dir},
            {"frame_count": 2, "fps": 30.0},
            {"start_time": None, "end_time": None},
        )

    assert run.call_count == 1
    assert result == []


def test_schema5_assembled_encode_publishes_bound_video_and_manifests(
    tmp_path, monkeypatch
):
    from src.depth_surge_3d.core.settings import validate_settings
    from src.depth_surge_3d.io.operations import save_processing_settings
    from src.depth_surge_3d.io.settings_artifact import ProcessingSettingsArtifactV5

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    job = tmp_path / "job"
    frames = job / "99_vr_frames"
    frames.mkdir(parents=True)
    image = np.zeros((4, 8, 3), dtype=np.uint8)
    assert cv2.imwrite(str(frames / "frame_000001.png"), image)
    settings = validate_settings(
        {
            "preserve_audio": False,
            "video_encoder": "libx264",
            "target_fps": "original",
            "vr_resolution": "custom:4x4",
            "per_eye_width": 4,
            "per_eye_height": 4,
            "vr_output_width": 8,
            "vr_output_height": 4,
        },
        source="explicit",
    )
    settings_file = save_processing_settings(
        job,
        "job",
        settings,
        {
            "width": 4,
            "height": 4,
            "fps": 24.0,
            "frame_count": 1,
            "duration": 1.0 / 24.0,
            "codec": 0,
            "sample_aspect_ratio_numerator": 1,
            "sample_aspect_ratio_denominator": 1,
            "sample_aspect_ratio": "1:1",
        },
        str(source),
        transactional_v5=True,
    )

    launches: list[list[str]] = []

    def fake_bounded_run(self, command, total_frames, progress_tracker=None):
        launches.append(command)
        Path(command[-1]).write_bytes(b"validated-mp4")
        return 0, ()

    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder.verify_ffmpeg_installation",
        lambda: True,
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder.subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("schema-5 encoding used unbounded subprocess.run")
        ),
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder.get_video_info_ffprobe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("schema-5 assembled encoding reprobed source FPS")
        ),
    )
    monkeypatch.setattr(
        VideoEncoder, "_run_schema5_ffmpeg_with_progress", fake_bounded_run
    )
    monkeypatch.setattr(VideoEncoder, "_validate_final_container", lambda *_args: True)
    monkeypatch.setattr(
        io,
        "BytesIO",
        lambda: (_ for _ in ()).throw(AssertionError("manifest was materialized")),
    )

    encoder = VideoEncoder()
    assert encoder.create_video(frames, job, str(source), settings)
    assert encoder.pending_final_encoding_reservation is not None
    artifact = ProcessingSettingsArtifactV5.parse(settings_file.read_bytes())
    expected = artifact.to_object()["output_info"]["expected_final_relative_path"]
    assert (job / expected).read_bytes() == b"validated-mp4"
    assert (job / "encoding_input_manifest.json").is_file()
    assert (job / "final_video_manifest.json").is_file()
    encoder.pending_final_encoding_reservation.validate_authenticated_publication(
        container_validator=lambda _path: True
    )
    assert (
        job / ".depth-surge-reservations-v1/final-encoding-reservation-v1.json"
    ).is_file()
    assert launches[0][launches[0].index("-start_number") + 1] == "1"
    assert launches[0][launches[0].index("-framerate") + 1] == "24/1"
    assert launches[0][launches[0].index("-vf") + 1] == "setsar=1"

    resumed = VideoEncoder()
    assert resumed.create_video(frames, job, str(source), settings)
    assert len(launches) == 1
    assert resumed.pending_final_encoding_reservation is not None
    from src.depth_surge_3d.io.operations import update_processing_status
    from src.depth_surge_3d.io.reservation_protocol import ReservationConflictError

    with pytest.raises(ReservationConflictError, match="terminal settings"):
        resumed.retire_final_encoding_reservation(success=True)
    assert update_processing_status(
        settings_file,
        "completed",
        {"frames_processed": 1},
    )
    resumed.retire_final_encoding_reservation(success=True)
    assert not (
        job / ".depth-surge-reservations-v1/final-encoding-reservation-v1.json"
    ).exists()


def test_schema5_provider_count_must_equal_durable_settings_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.processing.video.encoding_provider import (
        EncodingSequenceProvider,
    )

    job = tmp_path / "job"
    job.mkdir()
    provider = EncodingSequenceProvider.freeze(
        mode="assembled_vr",
        frame_count=1,
        first_index=0,
        frames_parent=PurePosixPath("99_vr_frames"),
    )
    artifact = Mock()
    artifact.to_object.return_value = {
        "metadata": {"final_media_publication_generation": None},
        "video_properties": {"frame_count": 2},
    }
    reserve = Mock(side_effect=AssertionError("reservation crossed frame-count gate"))
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder."
        "prepare_final_encoding_reservation_v1",
        reserve,
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder."
        "recover_processing_control_prefix_v1",
        lambda _root: artifact,
    )

    with pytest.raises(ValueError, match="frame count"):
        VideoEncoder()._prepare_schema5_publication_context(
            provider=provider,
            output_dir=job,
            settings=_direct_settings(),
        )

    reserve.assert_not_called()


def test_schema5_prelaunch_staged_manifest_mismatch_discards_and_rebuilds_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.processing.video.encoding_provider import (
        EncodingSequenceProvider,
        FinalEncodingInputChangedError,
    )

    job = tmp_path / "job"
    job.mkdir()
    active_index = (
        job / ".depth-surge-reservations-v1/final-encoding-reservation-v1.json"
    )
    active_index.parent.mkdir()
    active_index.write_bytes(b"active")
    provider = EncodingSequenceProvider.freeze(
        mode="assembled_vr",
        frame_count=1,
        first_index=0,
        frames_parent=PurePosixPath("99_vr_frames"),
    )
    artifact = Mock()
    artifact.to_object.return_value = {
        "metadata": {"final_media_publication_generation": None},
        "video_properties": {"frame_count": 1, "fps": 24.0},
        "processing_settings": {"target_fps": None, "source_fps": 24.0},
        "output_info": {"expected_final_relative_path": "result.mp4"},
    }
    old = Mock()
    old.job_root = job
    old.index.fingerprint = "11" * 32
    old.index.publication_generation = "44" * 16
    old.index.sibling_video_temporary_relative_path = ".old.tmp.mp4"
    old.index.final_video_relative_path = "result.mp4"
    old.recover_publication_prefix.return_value = Mock(phase="ready_pre_video")
    old.input_publication_source.return_value = Mock(
        fingerprint="a" * 64,
        raw_sha256="b" * 64,
        raw_byte_count=10,
    )
    old.discard_prelaunch_invocation.side_effect = lambda: active_index.unlink()
    fresh = Mock(sibling_path=job / ".fresh.tmp.mp4", job_root=job)
    fresh.index.fingerprint = "22" * 32
    fresh.index.sibling_video_temporary_relative_path = ".fresh.tmp.mp4"
    fresh.index.final_video_relative_path = "result.mp4"
    staged = Mock(fingerprint="c" * 64)
    source = Mock()
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder."
        "load_final_encoding_reservation_v1",
        lambda *_args, **_kwargs: old,
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder."
        "recover_processing_control_prefix_v1",
        lambda _root: artifact,
    )
    monkeypatch.setattr(
        VideoEncoder,
        "_recover_encoding_manifest_source",
        lambda *_args, **_kwargs: old.input_publication_source.return_value,
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder."
        "FinalEncodingReservationCapabilityV1.discard_prelaunch_invocation",
        lambda _capability: old.discard_prelaunch_invocation(),
    )

    def stage_after_rebuild(*_args, **_kwargs):
        preparation_frames = sum(
            frame.function == "_prepare_schema5_publication_context"
            for frame in inspect.stack()
        )
        assert preparation_frames == 1
        return staged, source

    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder."
        "audit_encoding_input_manifest",
        Mock(side_effect=FinalEncodingInputChangedError("staged input differs")),
    )
    prepare = Mock(return_value=fresh)
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder."
        "prepare_final_encoding_reservation_v1",
        prepare,
    )
    monkeypatch.setattr(
        VideoEncoder,
        "_stage_encoding_manifest_file",
        stage_after_rebuild,
    )
    monkeypatch.setattr(
        VideoEncoder,
        "_encoding_manifest_extent_authority",
        staticmethod(lambda _reservation: Mock()),
    )

    context = VideoEncoder()._prepare_schema5_publication_context(
        provider=provider,
        output_dir=job,
        settings=_direct_settings(),
    )

    old.discard_prelaunch_invocation.assert_called_once_with()
    prepare.assert_called_once()
    assert context.reservation.index_fingerprint == "22" * 32
    assert context.staged_manifest is staged


def test_schema5_manifest_stage_releases_settings_and_prepared_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.processing.video.encoding_provider import (
        EncodingSequenceProvider,
    )

    job = tmp_path / "job"
    job.mkdir()
    provider = EncodingSequenceProvider.freeze(
        mode="assembled_vr",
        frame_count=1,
        first_index=0,
        frames_parent=PurePosixPath("99_vr_frames"),
    )
    references: dict[str, weakref.ReferenceType[object]] = {}

    class Artifact:
        def to_object(self):
            return {
                "metadata": {"final_media_publication_generation": None},
                "video_properties": {"frame_count": 1, "fps": 24.0},
                "processing_settings": {"target_fps": None, "source_fps": 24.0},
                "output_info": {"expected_final_relative_path": "result.mp4"},
            }

    class Descriptor:
        payload_raw_max = 4096
        logical_byte_count = 8192

    class Transaction:
        descriptor = Descriptor()
        source_path = job / "manifest.reserved"

    class Index:
        fingerprint = "11" * 32
        sibling_video_temporary_relative_path = ".fresh.tmp.mp4"
        final_video_relative_path = "result.mp4"

    class Reservation:
        job_root = job
        index = Index()
        input_manifest_transaction = Transaction()

    def recover(_root):
        return Artifact()

    def prepare(*_args, settings, **_kwargs):
        reservation = Reservation()
        references["settings"] = weakref.ref(settings)
        references["reservation"] = weakref.ref(reservation)
        return reservation

    staged = Mock(fingerprint="a" * 64)
    source = Mock()

    def stage(*_args, **_kwargs):
        gc.collect()
        assert references["settings"]() is None
        assert references["reservation"]() is None
        return staged, source

    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder."
        "recover_processing_control_prefix_v1",
        recover,
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder."
        "prepare_final_encoding_reservation_v1",
        prepare,
    )
    monkeypatch.setattr(
        VideoEncoder,
        "_encoding_manifest_extent_authority",
        staticmethod(lambda _reservation: Mock()),
    )
    monkeypatch.setattr(VideoEncoder, "_stage_encoding_manifest_file", stage)

    context = VideoEncoder()._prepare_schema5_publication_context(
        provider=provider,
        output_dir=job,
        settings=_direct_settings(),
    )

    assert context.staged_manifest is staged


def _manifest_extent_authority_fixture(
    job: Path,
    source: Path,
):
    from src.depth_surge_3d.io.file_identity import (
        capture_directory_identity,
        capture_file_identity,
    )
    from src.depth_surge_3d.processing.video.video_encoder import (
        _EncodingManifestExtentAuthority,
    )

    return _EncodingManifestExtentAuthority(
        job_root=job,
        job_root_identity=capture_directory_identity(job),
        source_path=source,
        source_parent_identity=capture_directory_identity(source.parent),
        source_identity=capture_file_identity(source),
        payload_raw_max=4096,
        logical_byte_count=8192,
        indexed_source=True,
    )


def test_indexed_manifest_recovery_rejects_source_symlink_before_read(
    tmp_path: Path,
) -> None:
    from src.depth_surge_3d.io.reservation_protocol import ReservationConflictError

    job = tmp_path / "job"
    job.mkdir()
    source = job / "manifest.reserved"
    source.write_bytes(b"original")
    authority = _manifest_extent_authority_fixture(job, source)
    victim = job / "victim.bin"
    victim.write_bytes(b"victim-must-not-be-read")
    source.unlink()
    try:
        source.symlink_to(victim)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(ReservationConflictError, match="without following"):
        VideoEncoder._recover_encoding_manifest_source(
            authority,
            expected_frame_count=1,
        )

    assert victim.read_bytes() == b"victim-must-not-be-read"


def test_indexed_manifest_staging_rejects_ordinary_source_replacement_before_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.io.reservation_protocol import ReservationConflictError

    job = tmp_path / "job"
    job.mkdir()
    source = job / "manifest.reserved"
    source.write_bytes(b"original")
    authority = _manifest_extent_authority_fixture(job, source)
    source.unlink()
    source.write_bytes(b"replacement-must-not-be-written")
    stage = Mock()
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder."
        "stage_encoding_input_manifest",
        stage,
    )

    with pytest.raises(ReservationConflictError, match="identity changed"):
        VideoEncoder()._stage_encoding_manifest_file(
            provider=Mock(frame_count=1),
            output_dir=job,
            settings=_direct_settings(),
            reservation=Mock(),
            authority=authority,
        )

    stage.assert_not_called()
    assert source.read_bytes() == b"replacement-must-not-be-written"


def test_schema5_manifest_audit_releases_settings_and_prepared_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.processing.video.encoding_provider import (
        EncodingSequenceProvider,
    )

    job = tmp_path / "job"
    job.mkdir()
    active_index = (
        job / ".depth-surge-reservations-v1/final-encoding-reservation-v1.json"
    )
    active_index.parent.mkdir()
    active_index.write_bytes(b"active")
    provider = EncodingSequenceProvider.freeze(
        mode="assembled_vr",
        frame_count=1,
        first_index=0,
        frames_parent=PurePosixPath("99_vr_frames"),
    )
    references: dict[str, weakref.ReferenceType[object]] = {}

    class Artifact:
        def to_object(self):
            return {
                "metadata": {"final_media_publication_generation": "44" * 16},
                "video_properties": {"frame_count": 1, "fps": 24.0},
                "processing_settings": {"target_fps": None, "source_fps": 24.0},
                "output_info": {"expected_final_relative_path": "result.mp4"},
            }

    class Index:
        fingerprint = "11" * 32
        publication_generation = "44" * 16
        sibling_video_temporary_relative_path = ".old.tmp.mp4"
        final_video_relative_path = "result.mp4"

    class Reservation:
        job_root = job
        index = Index()
        input_manifest_transaction = None
        sibling_path = job / ".old.tmp.mp4"

        @staticmethod
        def recover_publication_prefix():
            return Mock(phase="ready_pre_video")

        @staticmethod
        def reset_incomplete_manifest_sources():
            return None

    def recover(_root):
        return Artifact()

    def load(_root, settings):
        reservation = Reservation()
        references["settings"] = weakref.ref(settings)
        references["reservation"] = weakref.ref(reservation)
        return reservation

    source = Mock(
        fingerprint="a" * 64,
        raw_sha256="b" * 64,
        raw_byte_count=10,
    )
    staged = Mock()

    def audit(*_args, **_kwargs):
        gc.collect()
        assert references["settings"]() is None
        assert references["reservation"]() is None
        return staged

    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder."
        "recover_processing_control_prefix_v1",
        recover,
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder."
        "load_final_encoding_reservation_v1",
        load,
    )
    monkeypatch.setattr(
        VideoEncoder,
        "_recover_encoding_manifest_source",
        lambda *_args, **_kwargs: source,
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder."
        "audit_encoding_input_manifest",
        audit,
    )

    context = VideoEncoder()._prepare_schema5_publication_context(
        provider=provider,
        output_dir=job,
        settings=_direct_settings(),
    )

    assert context.staged_manifest is staged


def test_schema5_assembled_launch_uses_provider_path_and_one_final_launch_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.processing.video.encoding_provider import (
        EncodingSequenceProvider,
    )

    job = tmp_path / "job"
    job.mkdir()
    caller_path = tmp_path / "caller_alias"
    sibling = job / ".depth-surge-final-v4-fixture.tmp.mp4"
    provider = EncodingSequenceProvider.freeze(
        mode="assembled_vr",
        frame_count=1,
        first_index=7,
        frames_parent=PurePosixPath("99_vr_frames"),
    )
    staged = Mock(provider_audit=Mock())
    events: list[str] = []
    reservation = Mock()
    reservation.commit_producer_settings.side_effect = lambda: (
        events.append("commit") or Mock()
    )
    context = Mock(
        already_authenticated=False,
        reservation=reservation,
        staged_manifest=staged,
        manifest_source=Mock(),
        sibling_path=sibling,
        expected_relative_path="result.mp4",
        provider=provider,
        settings=Mock(),
    )
    selected_audio = Mock(audio_end_sample=None)
    selected_audio.input_arguments.return_value = []
    selected_audio.output_arguments.return_value = []
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder."
        "recover_processing_control_prefix_v1",
        lambda _root: Mock(),
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder.verify_ffmpeg_installation",
        lambda: True,
    )
    monkeypatch.setattr(
        VideoEncoder,
        "_prepare_schema5_assembled_context",
        lambda *_args, **_kwargs: context,
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder.prepare_selected_audio_v1",
        lambda **_kwargs: events.append("audio") or selected_audio,
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder.validate_encoding_provider",
        lambda *_args, **_kwargs: events.append("validate"),
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder.audit_encoding_input_manifest",
        lambda *_args, **_kwargs: events.append("content") or staged,
    )
    launches: list[list[str]] = []

    def fail_launch(self, command, total_frames, progress_tracker=None):
        events.append("launch")
        launches.append(command)
        return 1, ()

    monkeypatch.setattr(VideoEncoder, "_run_schema5_ffmpeg_with_progress", fail_launch)

    assert not VideoEncoder().create_video(
        caller_path,
        job,
        str(tmp_path / "source.mp4"),
        {
            "preserve_audio": False,
            "video_encoder": "libx264",
            "vr_format": "side_by_side",
            "vr_resolution": "custom",
            "per_eye_width": 4,
            "per_eye_height": 4,
        },
    )

    assert events == ["audio", "commit", "validate", "content", "launch"]
    input_index = launches[0].index("-i") + 1
    assert launches[0][input_index] == str(job / "99_vr_frames" / "frame_%06d.png")


def test_schema5_direct_launch_commits_producer_after_all_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.processing.frames.vr_assembler import DirectStereoSource

    job = tmp_path / "job"
    left = job / "left"
    right = job / "right"
    left.mkdir(parents=True)
    right.mkdir()
    sibling = job / ".depth-surge-final-v4-fixture.tmp.mp4"
    events: list[str] = []
    reservation = Mock()
    reservation.commit_producer_settings.side_effect = lambda: (
        events.append("commit") or Mock()
    )
    staged = Mock(provider_audit=Mock(), first_headers=((6, 4), (6, 4)))
    context = Mock(
        already_authenticated=False,
        reservation=reservation,
        staged_manifest=staged,
        manifest_source=Mock(),
        sibling_path=sibling,
        expected_relative_path="result.mp4",
        settings=Mock(),
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder."
        "recover_processing_control_prefix_v1",
        lambda _root: Mock(),
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder.verify_ffmpeg_installation",
        lambda: True,
    )
    monkeypatch.setattr(
        VideoEncoder,
        "_prepare_schema5_direct_provider_context",
        lambda *_args, **_kwargs: context,
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder.prepare_selected_audio_v1",
        lambda **_kwargs: events.append("audio") or Mock(),
    )
    monkeypatch.setattr(
        VideoEncoder,
        "_build_schema5_direct_provider_command",
        lambda *_args, **_kwargs: events.append("command") or ["ffmpeg", str(sibling)],
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder.validate_encoding_provider",
        lambda *_args, **_kwargs: events.append("validate"),
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder.audit_encoding_input_manifest",
        lambda *_args, **_kwargs: events.append("content") or staged,
    )
    monkeypatch.setattr(
        VideoEncoder,
        "_run_schema5_ffmpeg_with_progress",
        lambda *_args, **_kwargs: events.append("launch") or (1, ()),
    )

    assert not VideoEncoder().create_video_from_stereo_source(
        DirectStereoSource(left, right, 1, 1),
        job,
        str(tmp_path / "source.mp4"),
        _direct_settings(),
        total_frames=1,
    )

    assert events == ["audio", "command", "commit", "validate", "content", "launch"]


def test_schema5_launch_boundary_rehashes_images_after_producer_commit(
    tmp_path: Path,
) -> None:
    from src.depth_surge_3d.processing.video.encoding_provider import (
        EncodingProviderChangedError,
        EncodingSequenceProvider,
        stage_encoding_input_manifest,
    )

    job = tmp_path / "job"
    frames = job / "99_vr_frames"
    frames.mkdir(parents=True)
    frame = frames / "frame_000000.png"
    assert cv2.imwrite(str(frame), np.zeros((4, 8, 3), np.uint8))
    provider = EncodingSequenceProvider.freeze(
        mode="assembled_vr",
        frame_count=1,
        first_index=0,
        frames_parent=PurePosixPath("99_vr_frames"),
    )
    staged = stage_encoding_input_manifest(
        provider,
        job_root=job,
        sink=io.BytesIO(),
        per_eye_size=(4, 4),
        vr_format="side_by_side",
    )
    reservation = Mock()

    def commit_then_change():
        assert cv2.imwrite(str(frame), np.ones((4, 8, 3), np.uint8))
        return Mock()

    reservation.commit_producer_settings.side_effect = commit_then_change
    context = Mock(
        provider=provider,
        staged_manifest=staged,
        reservation=reservation,
    )

    with pytest.raises(EncodingProviderChangedError, match="changed"):
        VideoEncoder._commit_schema5_launch_boundary(
            context,
            job_root=job,
            settings={
                "per_eye_width": 4,
                "per_eye_height": 4,
                "vr_format": "side_by_side",
            },
        )

    reservation.discard_prelaunch_invocation.assert_called_once_with()


def test_schema5_rejects_the_eager_sequence_api_before_path_list_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    (job / "processing-settings-v5.json").write_bytes(b"schema-5-marker")
    encoder = VideoEncoder()
    validate = Mock(
        side_effect=AssertionError("schema-5 entered eager Path-list validation")
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder."
        "recover_processing_control_prefix_v1",
        lambda _root: Mock(),
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder.verify_ffmpeg_installation",
        lambda: True,
    )
    monkeypatch.setattr(encoder, "_validate_direct_stereo_sequence", validate)

    assert not encoder.create_video_from_stereo_sequences(
        [job / "left" / "frame_000001.png"],
        [job / "right" / "frame_000001.png"],
        job,
        str(tmp_path / "source.mp4"),
        _direct_settings(),
        total_frames=1,
    )

    validate.assert_not_called()


def test_schema5_failed_encode_keeps_reservation_available_for_terminal_retirement(
    tmp_path, monkeypatch
):
    from src.depth_surge_3d.core.settings import validate_settings
    from src.depth_surge_3d.io.operations import save_processing_settings
    from src.depth_surge_3d.processing.video.final_publication import (
        FINAL_ENCODING_RESERVATION_RELATIVE_PATH,
    )

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    job = tmp_path / "job"
    frames = job / "99_vr_frames"
    frames.mkdir(parents=True)
    assert cv2.imwrite(str(frames / "frame_000007.png"), np.zeros((4, 8, 3), np.uint8))
    settings = validate_settings(
        {
            "preserve_audio": False,
            "video_encoder": "libx264",
            "vr_resolution": "custom:4x4",
            "per_eye_width": 4,
            "per_eye_height": 4,
            "vr_output_width": 8,
            "vr_output_height": 4,
        },
        source="explicit",
    )
    save_processing_settings(
        job,
        "job",
        settings,
        {
            "width": 4,
            "height": 4,
            "fps": 24.0,
            "frame_count": 1,
            "duration": 1.0 / 24.0,
            "codec": 0,
            "sample_aspect_ratio_numerator": 1,
            "sample_aspect_ratio_denominator": 1,
            "sample_aspect_ratio": "1:1",
        },
        str(source),
        transactional_v5=True,
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder.verify_ffmpeg_installation",
        lambda: True,
    )
    monkeypatch.setattr(
        VideoEncoder,
        "_run_schema5_ffmpeg_with_progress",
        lambda *_args, **_kwargs: (1, ("forced failure",)),
    )

    encoder = VideoEncoder(defer_final_reservation_retirement=True)
    assert not encoder.create_video(frames, job, str(source), settings)
    assert encoder._pending_final_reservation is not None
    assert (job / FINAL_ENCODING_RESERVATION_RELATIVE_PATH).is_file()

    from src.depth_surge_3d.io.operations import update_processing_status
    from src.depth_surge_3d.io.reservation_protocol import ReservationConflictError

    with pytest.raises(ReservationConflictError, match="durable failed"):
        encoder.retire_final_encoding_reservation(success=False)
    assert encoder._pending_final_reservation is not None
    assert (job / FINAL_ENCODING_RESERVATION_RELATIVE_PATH).is_file()
    assert update_processing_status(
        job / "processing-settings-v5.json",
        "failed",
        {"error": "forced failure"},
    )
    encoder.retire_final_encoding_reservation(success=False)
    assert encoder._pending_final_reservation is None
    assert not (job / FINAL_ENCODING_RESERVATION_RELATIVE_PATH).exists()


def test_schema5_manifest_staging_failure_keeps_new_reservation_for_terminal_retirement(
    tmp_path, monkeypatch
):
    from src.depth_surge_3d.core.settings import validate_settings
    from src.depth_surge_3d.io.operations import save_processing_settings
    from src.depth_surge_3d.processing.video.final_publication import (
        FINAL_ENCODING_RESERVATION_RELATIVE_PATH,
    )

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    job = tmp_path / "job"
    frames = job / "99_vr_frames"
    frames.mkdir(parents=True)
    assert cv2.imwrite(str(frames / "frame_000007.png"), np.zeros((4, 8, 3), np.uint8))
    settings = validate_settings(
        {
            "preserve_audio": False,
            "video_encoder": "libx264",
            "vr_resolution": "custom:4x4",
            "per_eye_width": 4,
            "per_eye_height": 4,
            "vr_output_width": 8,
            "vr_output_height": 4,
        },
        source="explicit",
    )
    save_processing_settings(
        job,
        "job",
        settings,
        {
            "width": 4,
            "height": 4,
            "fps": 24.0,
            "frame_count": 1,
            "duration": 1.0 / 24.0,
            "codec": 0,
            "sample_aspect_ratio_numerator": 1,
            "sample_aspect_ratio_denominator": 1,
            "sample_aspect_ratio": "1:1",
        },
        str(source),
        transactional_v5=True,
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder.verify_ffmpeg_installation",
        lambda: True,
    )
    monkeypatch.setattr(
        VideoEncoder,
        "_stage_encoding_manifest_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("forced staging failure")
        ),
    )

    encoder = VideoEncoder(defer_final_reservation_retirement=True)

    assert not encoder.create_video(frames, job, str(source), settings)
    assert encoder._pending_final_reservation is not None
    assert (job / FINAL_ENCODING_RESERVATION_RELATIVE_PATH).is_file()

    from src.depth_surge_3d.io.operations import update_processing_status
    from src.depth_surge_3d.io.reservation_protocol import ReservationConflictError

    with pytest.raises(ReservationConflictError, match="durable failed"):
        encoder.retire_final_encoding_reservation(success=False)
    assert encoder._pending_final_reservation is not None
    assert (job / FINAL_ENCODING_RESERVATION_RELATIVE_PATH).is_file()
    assert update_processing_status(
        job / "processing-settings-v5.json",
        "failed",
        {"error": "forced staging failure"},
    )
    encoder.retire_final_encoding_reservation(success=False)
    assert encoder._pending_final_reservation is None
    assert not (job / FINAL_ENCODING_RESERVATION_RELATIVE_PATH).exists()


def test_schema5_incomplete_reservation_stays_retryable_instead_of_becoming_failed(
    tmp_path, monkeypatch
):
    from src.depth_surge_3d.io.reservation_protocol import ReservationIncompleteError

    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder.verify_ffmpeg_installation",
        lambda: True,
    )
    monkeypatch.setattr(
        VideoEncoder,
        "_prepare_schema5_assembled_context",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ReservationIncompleteError("persistent ENOSPC after final index")
        ),
    )

    with pytest.raises(ReservationIncompleteError, match="persistent ENOSPC"):
        VideoEncoder(defer_final_reservation_retirement=True).create_video(
            tmp_path,
            tmp_path,
            str(tmp_path / "source.mp4"),
            {},
        )


def test_schema5_direct_encode_uses_provider_and_same_publication_contract(
    tmp_path, monkeypatch
):
    from src.depth_surge_3d.core.settings import validate_settings
    from src.depth_surge_3d.io.operations import save_processing_settings
    from src.depth_surge_3d.io.settings_artifact import ProcessingSettingsArtifactV5
    from src.depth_surge_3d.processing.frames.vr_assembler import DirectStereoSource

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    job = tmp_path / "job"
    left_dir = job / "04_left_frames"
    right_dir = job / "04_right_frames"
    left_dir.mkdir(parents=True)
    right_dir.mkdir(parents=True)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    left = left_dir / "frame_000001.png"
    right = right_dir / "frame_000001.png"
    assert cv2.imwrite(str(left), image)
    assert cv2.imwrite(str(right), image)
    settings = validate_settings(
        {
            "preserve_audio": False,
            "direct_vr_encode": True,
            "video_encoder": "libx264",
            "target_fps": "original",
            "vr_resolution": "custom:4x4",
            "per_eye_width": 4,
            "per_eye_height": 4,
            "vr_output_width": 8,
            "vr_output_height": 4,
        },
        source="explicit",
    )
    settings_file = save_processing_settings(
        job,
        "job",
        settings,
        {
            "width": 4,
            "height": 4,
            "fps": 24.0,
            "frame_count": 1,
            "duration": 1.0 / 24.0,
            "codec": 0,
            "sample_aspect_ratio_numerator": 1,
            "sample_aspect_ratio_denominator": 1,
            "sample_aspect_ratio": "1:1",
        },
        str(source),
        transactional_v5=True,
    )

    def fake_progress(self, command, total_frames, progress_tracker):
        Path(command[-1]).write_bytes(b"validated-direct-mp4")
        return 0, ()

    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder.verify_ffmpeg_installation",
        lambda: True,
    )
    monkeypatch.setattr(
        VideoEncoder, "_run_schema5_ffmpeg_with_progress", fake_progress
    )
    monkeypatch.setattr(VideoEncoder, "_validate_final_container", lambda *_args: True)

    encoder = VideoEncoder()
    assert encoder.create_video_from_stereo_source(
        DirectStereoSource(left_dir, right_dir, 1, 1),
        job,
        str(source),
        settings,
        total_frames=1,
    )
    artifact = ProcessingSettingsArtifactV5.parse(settings_file.read_bytes())
    expected = artifact.to_object()["output_info"]["expected_final_relative_path"]
    assert (job / expected).read_bytes() == b"validated-direct-mp4"
    assert encoder.pending_final_encoding_reservation is not None
    encoder.pending_final_encoding_reservation.validate_authenticated_publication(
        container_validator=lambda _path: True
    )
    assert (
        job / ".depth-surge-reservations-v1/final-encoding-reservation-v1.json"
    ).is_file()


def test_schema5_direct_provider_revalidates_immediately_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.processing.frames.vr_assembler import DirectStereoSource

    job = tmp_path / "job"
    left = job / "left"
    right = job / "right"
    left.mkdir(parents=True)
    right.mkdir()
    sibling = job / ".depth-surge-final-v4-fixture.tmp.mp4"
    reservation = Mock()
    staged = Mock(provider_audit=Mock())
    context = Mock(
        already_authenticated=False,
        reservation=reservation,
        staged_manifest=staged,
        manifest_source=Mock(),
        sibling_path=sibling,
    )
    encoder = VideoEncoder()
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder."
        "recover_processing_control_prefix_v1",
        lambda _root: Mock(),
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder.verify_ffmpeg_installation",
        lambda: True,
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder.get_video_info_ffprobe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("schema-5 direct encoding reprobed source FPS")
        ),
    )
    monkeypatch.setattr(
        encoder,
        "_prepare_schema5_direct_provider_context",
        lambda **_kwargs: context,
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder.prepare_selected_audio_v1",
        lambda **_kwargs: Mock(),
    )
    monkeypatch.setattr(
        encoder,
        "_build_schema5_direct_provider_command",
        lambda *_args: ["ffmpeg", str(sibling)],
    )
    validate = Mock(side_effect=RuntimeError("provider replay changed"))
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder.validate_encoding_provider",
        validate,
    )
    launch = Mock(side_effect=AssertionError("FFmpeg launch crossed provider gate"))
    monkeypatch.setattr(encoder, "_run_schema5_ffmpeg_with_progress", launch)

    assert not encoder.create_video_from_stereo_source(
        DirectStereoSource(left, right, 1, 1),
        job,
        str(tmp_path / "source.mp4"),
        _direct_settings(),
        total_frames=1,
    )

    validate.assert_called_once()
    assert validate.call_args.kwargs == {
        "launch_started": False,
        "cancelled": None,
    }
    launch.assert_not_called()
    reservation.discard_nonsemantic_sibling.assert_called_once_with()


def test_schema5_direct_provider_preserves_unowned_sibling_occupant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.processing.frames.vr_assembler import DirectStereoSource

    job = tmp_path / "job"
    left = job / "left"
    right = job / "right"
    left.mkdir(parents=True)
    right.mkdir()
    sibling = job / ".depth-surge-final-v4-fixture.tmp.mp4"
    sibling.write_bytes(b"unowned-occupant")
    reservation = Mock()
    context = Mock(
        already_authenticated=False,
        reservation=reservation,
        staged_manifest=Mock(provider_audit=Mock()),
        manifest_source=Mock(),
        sibling_path=sibling,
    )
    encoder = VideoEncoder()
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder."
        "recover_processing_control_prefix_v1",
        lambda _root: Mock(),
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder.verify_ffmpeg_installation",
        lambda: True,
    )
    monkeypatch.setattr(
        encoder,
        "_prepare_schema5_direct_provider_context",
        lambda **_kwargs: context,
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder.prepare_selected_audio_v1",
        lambda **_kwargs: Mock(),
    )
    launch = Mock(side_effect=AssertionError("FFmpeg launched with sibling collision"))
    monkeypatch.setattr(encoder, "_run_schema5_ffmpeg_with_progress", launch)

    assert not encoder.create_video_from_stereo_source(
        DirectStereoSource(left, right, 1, 1),
        job,
        str(tmp_path / "source.mp4"),
        _direct_settings(),
        total_frames=1,
    )

    assert sibling.read_bytes() == b"unowned-occupant"
    launch.assert_not_called()


def test_direct_source_recovers_schema5_before_any_legacy_path_materialization(
    tmp_path,
    monkeypatch,
):
    from src.depth_surge_3d.processing.frames.vr_assembler import DirectStereoSource

    job = tmp_path / "job"
    left = job / "left"
    right = job / "right"
    left.mkdir(parents=True)
    right.mkdir()
    encoder = VideoEncoder()
    context = Mock(already_authenticated=True, reservation=Mock())
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder."
        "recover_processing_control_prefix_v1",
        lambda _root: Mock(),
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder.verify_ffmpeg_installation",
        lambda: True,
    )
    monkeypatch.setattr(
        encoder,
        "_prepare_schema5_direct_provider_context",
        lambda **_kwargs: context,
    )
    monkeypatch.setattr(
        encoder,
        "create_video_from_stereo_sequences",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("schema-5 recovery entered the legacy Path-list API")
        ),
    )

    assert encoder.create_video_from_stereo_source(
        DirectStereoSource(left, right, 1, 4_194_304),
        job,
        str(tmp_path / "source.mp4"),
        {},
        total_frames=4_194_304,
    )


def test_maximum_encoding_manifest_bound_is_a_disk_extent_not_a_resident_cap() -> None:
    from pathlib import PurePosixPath

    from src.depth_surge_3d.processing.video.encoding_provider import (
        EncodingSequenceProvider,
    )

    provider = EncodingSequenceProvider.freeze(
        mode="direct_stereo",
        frame_count=4_194_304,
        first_index=1,
        left_parent=PurePosixPath("07_left_upscaled"),
        right_parent=PurePosixPath("07_right_upscaled"),
    )

    raw_bound = VideoEncoder._encoding_manifest_raw_bound(provider)

    assert raw_bound > 512 * 1024 * 1024
    assert raw_bound < 1 << 64


@pytest.mark.parametrize(
    ("crash_point", "expected_launches"),
    [
        ("after-video", 2),
        ("after-input-manifest", 2),
        ("after-final-intent-sync", 1),
    ],
)
def test_schema5_post_boundary_restart_reencodes_without_rewriting_input_manifest(
    tmp_path, monkeypatch, crash_point, expected_launches
):
    from src.depth_surge_3d.core.settings import validate_settings
    from src.depth_surge_3d.io.operations import save_processing_settings
    from src.depth_surge_3d.io.settings_artifact import ProcessingSettingsArtifactV5
    from src.depth_surge_3d.processing.video.final_publication import (
        FINAL_ENCODING_RESERVATION_RELATIVE_PATH,
        FinalPublicationTransactionV4,
    )

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    job = tmp_path / "job"
    frames = job / "99_vr_frames"
    frames.mkdir(parents=True)
    assert cv2.imwrite(str(frames / "frame_000001.png"), np.zeros((4, 8, 3), np.uint8))
    settings = validate_settings(
        {
            "preserve_audio": False,
            "video_encoder": "libx264",
            "vr_resolution": "custom:4x4",
            "per_eye_width": 4,
            "per_eye_height": 4,
            "vr_output_width": 8,
            "vr_output_height": 4,
        },
        source="explicit",
    )
    settings_file = save_processing_settings(
        job,
        "job",
        settings,
        {
            "width": 4,
            "height": 4,
            "fps": 24.0,
            "frame_count": 1,
            "duration": 1.0 / 24.0,
            "codec": 0,
            "sample_aspect_ratio_numerator": 1,
            "sample_aspect_ratio_denominator": 1,
            "sample_aspect_ratio": "1:1",
        },
        str(source),
        transactional_v5=True,
    )
    launches = 0

    def fake_bounded_run(self, command, total_frames, progress_tracker=None):
        nonlocal launches
        launches += 1
        Path(command[-1]).write_bytes(f"validated-{launches}".encode())
        return 0, ()

    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.video_encoder.verify_ffmpeg_installation",
        lambda: True,
    )
    monkeypatch.setattr(
        VideoEncoder, "_run_schema5_ffmpeg_with_progress", fake_bounded_run
    )
    monkeypatch.setattr(VideoEncoder, "_validate_final_container", lambda *_args: True)
    original_publish = FinalPublicationTransactionV4.publish

    def crash_after_input(self):
        return original_publish(self, crash_at=crash_point)

    monkeypatch.setattr(FinalPublicationTransactionV4, "publish", crash_after_input)
    assert not VideoEncoder(defer_final_reservation_retirement=True).create_video(
        frames, job, str(source), settings
    )
    committed_input = (
        (job / "encoding_input_manifest.json").read_bytes()
        if crash_point != "after-video"
        else None
    )
    assert (job / FINAL_ENCODING_RESERVATION_RELATIVE_PATH).is_file()

    monkeypatch.setattr(FinalPublicationTransactionV4, "publish", original_publish)
    resumed_encoder = VideoEncoder()
    assert resumed_encoder.create_video(frames, job, str(source), settings)

    artifact = ProcessingSettingsArtifactV5.parse(settings_file.read_bytes())
    assert artifact.status == "in_progress"
    assert launches == expected_launches
    if committed_input is not None:
        assert (job / "encoding_input_manifest.json").read_bytes() == committed_input
    assert resumed_encoder.pending_final_encoding_reservation is not None
    resumed_encoder.pending_final_encoding_reservation.validate_authenticated_publication(
        container_validator=lambda _path: True
    )
    assert (job / FINAL_ENCODING_RESERVATION_RELATIVE_PATH).is_file()


@pytest.mark.parametrize("discard_result", [False, OSError("forward recovery pending")])
def test_schema5_attempt_cleanup_never_unlinks_a_nondiscardable_sibling(
    tmp_path,
    discard_result,
) -> None:
    sibling = tmp_path / ".depth-surge-final-v4-fixture.tmp.mp4"
    sibling.write_bytes(b"only-forward-recovery-copy")
    reservation = Mock()
    if isinstance(discard_result, Exception):
        reservation.discard_nonsemantic_sibling.side_effect = discard_result
    else:
        reservation.discard_nonsemantic_sibling.return_value = discard_result
    context = Mock(reservation=reservation)

    VideoEncoder._discard_schema5_sibling_after_attempt(context)

    assert sibling.read_bytes() == b"only-forward-recovery-copy"
    reservation.discard_nonsemantic_sibling.assert_called_once_with()
