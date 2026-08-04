"""Unit tests for final video encoding."""

from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from src.depth_surge_3d.processing.video.video_encoder import VideoEncoder


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


def test_extract_frames_reuses_complete_existing_sequence(tmp_path):
    """Resuming a stopped job does not decode a complete frame sequence again."""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    expected = []
    for i in range(1, 4):
        path = frames_dir / f"frame_{i:06d}.png"
        cv2.imwrite(str(path), np.zeros((2, 2, 3), dtype=np.uint8))
        expected.append(path)

    with patch("src.depth_surge_3d.processing.video.video_encoder.subprocess.run") as run:
        result = VideoEncoder().extract_frames(
            "source.mp4",
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

    with patch("src.depth_surge_3d.processing.video.video_encoder.subprocess.run") as run:
        result = VideoEncoder().extract_frames(
            "source.mp4",
            {"base": tmp_path, "frames": frames_dir, "depth_maps": depth_dir},
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
            {"base": tmp_path, "frames": frames_dir, "depth_maps": depth_dir},
            {"frame_count": 2, "fps": 30.0},
            {"start_time": None, "end_time": None},
        )

    assert run.call_count == 1
    assert result == []
