"""Real FFmpeg coverage for clip selection in the source-frame cache."""

import shutil
import subprocess
from pathlib import Path

import cv2
import pytest

from src.depth_surge_3d.processing.video.video_encoder import VideoEncoder


@pytest.mark.integration
def test_switching_equal_length_clips_matches_a_fresh_extraction(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg is required")
    source = tmp_path / "red-then-blue.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=red:s=32x16:r=2:d=2",
            "-f",
            "lavfi",
            "-i",
            "color=blue:s=32x16:r=2:d=2",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0",
            "-c:v",
            "ffv1",
            str(source),
        ],
        check=True,
        capture_output=True,
    )

    def extract_colors(directory: Path, start: str, end: str) -> list[tuple[int, ...]]:
        directory.mkdir(exist_ok=True)
        frames = VideoEncoder().extract_frames(
            str(source),
            {"frames": directory},
            {"frame_count": 8, "fps": 2.0},
            {"start_time": start, "end_time": end},
        )
        assert len(frames) == 2
        return [tuple(cv2.imread(str(frame))[0, 0].tolist()) for frame in frames]

    cached_dir = tmp_path / "cached"
    red = extract_colors(cached_dir, "0:00", "0:01")
    blue = extract_colors(tmp_path / "fresh", "0:02", "0:03")
    assert all(r > 200 and b < 10 for b, _, r in red)
    assert all(b > 200 and r < 10 for b, _, r in blue)
    assert extract_colors(cached_dir, "0:02", "0:03") == blue
