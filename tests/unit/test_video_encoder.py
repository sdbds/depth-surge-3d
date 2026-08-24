"""Unit tests for final video encoding."""

import io
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.core.file_identity import file_sample_fingerprint
from src.depth_surge_3d.processing.video.final_audio import (
    OpenFileIdentityV1,
    SelectedAudioIdentityV1,
)
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
    starts = [command[index + 1] for index, value in enumerate(command) if value == "-start_number"]
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
    assert max(index for index, value in enumerate(command) if value == "-i") < command.index(
        "-filter_complex"
    )


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


def test_direct_command_reuses_nvenc_arguments_unchanged(tmp_path):
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

    assert command[-9:] == [
        "-c:v",
        "hevc_nvenc",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "p7",
        "-tune",
        "hq",
        str(tmp_path / ".output.direct.tmp.mp4"),
    ]


def test_direct_runner_merges_stderr_tracks_frames_and_waits():
    process = _FakeProcess("frame=1\nnoise\nframe=2\n")
    tracker = Mock()

    with patch(
        "src.depth_surge_3d.processing.video.video_encoder.subprocess.Popen",
        return_value=process,
    ) as popen:
        returncode, diagnostics = VideoEncoder()._run_ffmpeg_with_progress(["ffmpeg"], 2, tracker)

    assert returncode == 0
    assert diagnostics == ("noise",)
    assert process.waited is True
    assert [call.kwargs["frame_num"] for call in tracker.update_progress.call_args_list] == [1, 2]
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
        returncode, diagnostics = VideoEncoder()._run_ffmpeg_with_progress(["ffmpeg"], 1, None)

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
        returncode, diagnostics = VideoEncoder()._run_ffmpeg_with_progress(["ffmpeg"], 4, tracker)

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


def test_direct_create_video_failure_preserves_old_final_and_removes_temp(tmp_path, capsys):
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
        patch("src.depth_surge_3d.processing.video.video_encoder.subprocess.Popen") as popen,
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
        patch("src.depth_surge_3d.processing.video.video_encoder.subprocess.Popen") as popen,
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
def test_direct_create_video_reports_output_validation_failure(tmp_path, capsys, output_state):
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

    with patch("src.depth_surge_3d.processing.video.video_encoder.subprocess.run") as run:
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

    with patch("src.depth_surge_3d.processing.video.video_encoder.subprocess.run") as run:
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


def test_extract_frames_reextracts_approximate_count_without_downstream_progress(tmp_path):
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


def test_extract_frames_does_not_reuse_empty_sequence_with_stale_downstream_files(tmp_path):
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


def test_schema5_assembled_encode_publishes_bound_video_and_manifests(tmp_path, monkeypatch):
    from src.depth_surge_3d.core.settings import validate_settings
    from src.depth_surge_3d.io.operations import save_processing_settings
    from src.depth_surge_3d.io.settings_artifact import ProcessingSettingsArtifactV5
    from src.depth_surge_3d.processing.video.final_publication import audit_final_media_v1

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
    monkeypatch.setattr(VideoEncoder, "_run_schema5_ffmpeg_with_progress", fake_bounded_run)
    monkeypatch.setattr(VideoEncoder, "_validate_final_container", lambda *_args: True)
    monkeypatch.setattr(
        io,
        "BytesIO",
        lambda: (_ for _ in ()).throw(AssertionError("manifest was materialized")),
    )

    encoder = VideoEncoder(defer_final_reservation_retirement=True)
    assert encoder.create_video(frames, job, str(source), settings)
    artifact = ProcessingSettingsArtifactV5.parse(settings_file.read_bytes())
    expected = artifact.to_object()["output_info"]["expected_final_relative_path"]
    assert (job / expected).read_bytes() == b"validated-mp4"
    assert (job / "encoding_input_manifest.json").is_file()
    assert (job / "final_video_manifest.json").is_file()
    assert audit_final_media_v1(job, artifact).final_video_authenticated is True
    assert (job / ".depth-surge-reservations-v1/final-encoding-reservation-v1.json").is_file()
    assert launches[0][launches[0].index("-start_number") + 1] == "1"
    assert launches[0][launches[0].index("-vf") + 1] == "setsar=1"

    resumed = VideoEncoder(defer_final_reservation_retirement=True)
    assert resumed.create_video(frames, job, str(source), settings)
    assert len(launches) == 1
    resumed.retire_final_encoding_reservation(success=True)
    assert not (job / ".depth-surge-reservations-v1/final-encoding-reservation-v1.json").exists()


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
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("forced staging failure")),
    )

    encoder = VideoEncoder(defer_final_reservation_retirement=True)

    assert not encoder.create_video(frames, job, str(source), settings)
    assert encoder._pending_final_reservation is not None
    assert (job / FINAL_ENCODING_RESERVATION_RELATIVE_PATH).is_file()

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


def test_schema5_direct_encode_uses_provider_and_same_publication_contract(tmp_path, monkeypatch):
    from src.depth_surge_3d.core.settings import validate_settings
    from src.depth_surge_3d.io.operations import save_processing_settings
    from src.depth_surge_3d.io.settings_artifact import ProcessingSettingsArtifactV5
    from src.depth_surge_3d.processing.video.final_publication import audit_final_media_v1

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
    monkeypatch.setattr(VideoEncoder, "_run_schema5_ffmpeg_with_progress", fake_progress)
    monkeypatch.setattr(VideoEncoder, "_validate_final_container", lambda *_args: True)

    assert VideoEncoder().create_video_from_stereo_sequences(
        [left],
        [right],
        job,
        str(source),
        settings,
        total_frames=1,
    )
    artifact = ProcessingSettingsArtifactV5.parse(settings_file.read_bytes())
    expected = artifact.to_object()["output_info"]["expected_final_relative_path"]
    assert (job / expected).read_bytes() == b"validated-direct-mp4"
    assert audit_final_media_v1(job, artifact).final_video_authenticated is True


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
        audit_final_media_v1,
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
    monkeypatch.setattr(VideoEncoder, "_run_schema5_ffmpeg_with_progress", fake_bounded_run)
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
    assert VideoEncoder().create_video(frames, job, str(source), settings)

    artifact = ProcessingSettingsArtifactV5.parse(settings_file.read_bytes())
    assert launches == expected_launches
    if committed_input is not None:
        assert (job / "encoding_input_manifest.json").read_bytes() == committed_input
    assert audit_final_media_v1(job, artifact).final_video_authenticated is True
    assert not (job / FINAL_ENCODING_RESERVATION_RELATIVE_PATH).exists()
