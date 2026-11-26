"""Lightweight FFprobe final container validation tests."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import pytest

from src.depth_surge_3d.processing.video.final_container_validation import (
    FinalContainerExpectation,
    FinalContainerProcessError,
    FinalContainerValidationError,
    validate_final_container_v4,
    verify_existing_final_video,
)


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg/ffprobe unavailable")
def test_validate_final_container_accepts_valid_and_rejects_mismatches(tmp_path: Path) -> None:
    mp4 = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=size=128x64:rate=24:duration=0.25",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(mp4),
        ],
        check=True,
    )

    ok = validate_final_container_v4(
        mp4,
        FinalContainerExpectation(
            width=128,
            height=64,
            fps="24/1",
            frame_count=6,
            resolved_encoder="libx264",
            audio_expected=False,
        ),
    )
    assert ok is True
    assert verify_existing_final_video(mp4, expected_width=128, expected_height=64) is True
    assert verify_existing_final_video(mp4, expected_width=64, expected_height=64) is False

    with pytest.raises(FinalContainerValidationError, match="dimensions"):
        validate_final_container_v4(
            mp4,
            FinalContainerExpectation(
                width=256,
                height=64,
                fps="24/1",
                frame_count=6,
                resolved_encoder="libx264",
                audio_expected=False,
            ),
        )

    with pytest.raises(FinalContainerValidationError, match="audio"):
        validate_final_container_v4(
            mp4,
            FinalContainerExpectation(
                width=128,
                height=64,
                fps="24/1",
                frame_count=6,
                resolved_encoder="libx264",
                audio_expected=True,
            ),
        )


def test_validate_final_container_rejects_empty_or_missing(tmp_path: Path) -> None:
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    exp = FinalContainerExpectation(
        width=64, height=64, fps="30/1", frame_count=1, resolved_encoder="libx264"
    )
    with pytest.raises(FinalContainerProcessError):
        validate_final_container_v4(empty, exp)
    with pytest.raises(FinalContainerProcessError):
        validate_final_container_v4(tmp_path / "missing.mp4", exp)
