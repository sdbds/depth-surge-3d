"""Unit tests for final video encoding."""

import json
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from src.depth_surge_3d.core.file_identity import file_sample_fingerprint
from src.depth_surge_3d.processing.video.video_encoder import VideoEncoder
from src.depth_surge_3d.processing.frames.source_frame_manifest import (
    write_source_frame_manifest,
)


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
