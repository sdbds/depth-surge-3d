"""Fast FFprobe audio selection and clip-duration alignment tests."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import pytest

from src.depth_surge_3d.processing.video.final_audio import prepare_selected_audio_v1


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def test_prepare_selected_audio_disabled(tmp_path: Path) -> None:
    video = tmp_path / "in.mp4"
    video.write_bytes(b"unused")
    res = prepare_selected_audio_v1(
        output_dir=tmp_path,
        original_video=str(video),
        settings={"preserve_audio": False},
        frame_count=10,
        fps="30/1",
    )
    assert res.disposition == "disabled"
    assert not res.usable
    assert res.input_arguments() == ()
    assert res.output_arguments(1) == ()


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg/ffprobe unavailable")
def test_prepare_selected_audio_probes_real_video_and_computes_end_sample(tmp_path: Path) -> None:
    with_audio = tmp_path / "with_audio.mp4"
    silent = tmp_path / "silent.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=size=64x64:rate=30:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(with_audio),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=size=64x64:rate=30:duration=0.5",
            "-c:v",
            "libx264",
            str(silent),
        ],
        check=True,
    )

    audio = prepare_selected_audio_v1(
        output_dir=tmp_path,
        original_video=str(with_audio),
        settings={"preserve_audio": True},
        frame_count=15,
        fps="30/1",
    )
    assert audio.usable is True
    assert audio.sample_rate == 48000
    assert audio.audio_end_sample == 24000
    assert "-i" in audio.input_arguments()
    assert "atrim=end_sample=24000,asetpts=PTS-STARTPTS" in audio.output_arguments(1)

    no_audio = prepare_selected_audio_v1(
        output_dir=tmp_path,
        original_video=str(silent),
        settings={"preserve_audio": True},
        frame_count=15,
        fps="30/1",
    )
    assert no_audio.disposition == "no_a0"
    assert not no_audio.usable
