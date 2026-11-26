"""Real FFmpeg video encoder tests: streamed BGR24, direct stereo, assembled VR, extract."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.io.operations import save_processing_settings
from src.depth_surge_3d.processing.frames.vr_assembler import DirectStereoSource
from src.depth_surge_3d.processing.video.final_container_validation import (
    verify_existing_final_video,
)
from src.depth_surge_3d.processing.video.video_encoder import VideoEncoder


def _has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _make_source_video(path: Path, *, with_audio: bool = True) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=64x64:rate=24:duration=0.25",
    ]
    if with_audio:
        cmd.extend(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=0.25"])
    cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
    if with_audio:
        cmd.extend(["-c:a", "aac"])
    cmd.append(str(path))
    subprocess.run(cmd, check=True)


def _base_settings(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "vr_format": "side_by_side",
        "vr_resolution": "custom:64x64",
        "per_eye_width": 64,
        "per_eye_height": 64,
        "vr_output_width": 128,
        "vr_output_height": 64,
        "video_encoder": "libx264",
        "preserve_audio": True,
        "source_fps": 24.0,
        "target_fps": None,
        "start_time": None,
        "end_time": None,
        "keep_intermediates": True,
    }
    data.update(overrides)
    return data


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg/ffprobe unavailable")
def test_streamed_bgr24_session_encodes_and_publishes_atomically(tmp_path: Path) -> None:
    src_video = tmp_path / "source.mp4"
    _make_source_video(src_video, with_audio=True)
    out_dir = tmp_path / "job"
    out_dir.mkdir()
    settings = _base_settings(video_encoder="libx264")

    encoder = VideoEncoder()
    session = encoder.open_streamed_vr_session(
        output_dir=out_dir,
        original_video=str(src_video),
        settings=settings,
        frame_count=6,
    )
    frame = np.full((64, 128, 3), 120, dtype=np.uint8)
    for i in range(6):
        session.write(f"frame_{i:06d}", frame)
    assert session.finish_and_publish() is True
    assert session.final_output_path.is_file()
    assert verify_existing_final_video(
        session.final_output_path, expected_width=128, expected_height=64
    )
    assert not any(p.name.endswith(".tmp.mp4") for p in out_dir.iterdir())


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg/ffprobe unavailable")
def test_direct_stereo_source_and_assembled_vr_encode(tmp_path: Path) -> None:
    src_video = tmp_path / "clip.mp4"
    _make_source_video(src_video, with_audio=False)
    out_dir = tmp_path / "job"
    left_dir = out_dir / "06_left_cropped"
    right_dir = out_dir / "06_right_cropped"
    vr_dir = out_dir / "99_vr_frames"
    for d in (left_dir, right_dir, vr_dir):
        d.mkdir(parents=True)

    eye = np.full((64, 64, 3), 90, dtype=np.uint8)
    vr = np.full((64, 128, 3), 140, dtype=np.uint8)
    for idx in range(1, 5):
        cv2.imwrite(str(left_dir / f"frame_{idx:06d}.png"), eye)
        cv2.imwrite(str(right_dir / f"frame_{idx:06d}.png"), eye)
        cv2.imwrite(str(vr_dir / f"frame_{idx:06d}.png"), vr)

    encoder = VideoEncoder()
    settings = _base_settings(preserve_audio=False, video_encoder="libx265")
    source = DirectStereoSource(
        left_parent=left_dir, right_parent=right_dir, first_index=1, frame_count=4
    )
    assert (
        encoder.create_video_from_stereo_source(
            source, out_dir, str(src_video), settings, total_frames=4
        )
        is True
    )
    mp4s = list(out_dir.glob("*.mp4"))
    assert len(mp4s) == 1
    assert verify_existing_final_video(mp4s[0], expected_width=128, expected_height=64)

    # Now test pre-assembled VR encode path with libx264
    mp4s[0].unlink()
    settings2 = _base_settings(preserve_audio=False, video_encoder="libx264")
    assert encoder.create_video(vr_dir, out_dir, str(src_video), settings2) is True
    mp4s2 = list(out_dir.glob("*.mp4"))
    assert len(mp4s2) == 1
    assert verify_existing_final_video(mp4s2[0], expected_width=128, expected_height=64)


@pytest.mark.skipif(not _has_ffmpeg(), reason="ffmpeg/ffprobe unavailable")
def test_extract_frames_extracts_and_reuses(tmp_path: Path) -> None:
    src_video = tmp_path / "src.mp4"
    _make_source_video(src_video, with_audio=False)
    out_dir = tmp_path / "job"
    frames_dir = out_dir / "00_original_frames"
    frames_dir.mkdir(parents=True)

    encoder = VideoEncoder()
    props = {"fps": 24.0, "frame_count": 6, "width": 64, "height": 64, "frame_rate": "24/1"}
    dirs = {"base": out_dir, "frames": frames_dir}
    extracted = encoder.extract_frames(
        str(src_video), dirs, props, {"start_time": None, "end_time": None}
    )
    assert len(extracted) == 6
    reused = encoder.extract_frames(
        str(src_video), dirs, props, {"start_time": None, "end_time": None}
    )
    assert reused == extracted


def test_resolve_encoder_name_respects_libx264_and_libx265(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = VideoEncoder()
    monkeypatch.setattr(encoder, "_check_nvenc_available", lambda **_kw: False)
    assert encoder._resolve_encoder_name("libx265") == "libx265"
    assert encoder._resolve_encoder_name("libx264") == "libx264"
    assert encoder._resolve_encoder_name("auto") == "libx264"
    monkeypatch.setattr(encoder, "_check_nvenc_available", lambda **_kw: True)
    assert encoder._resolve_encoder_name("auto") == "hevc_nvenc"
    assert encoder._resolve_encoder_name("libx265") == "libx265"


def test_schema5_settings_output_name_honored(tmp_path: Path) -> None:
    src = tmp_path / "demo.mp4"
    src.write_bytes(b"fake")
    job = tmp_path / "job"
    from src.depth_surge_3d.core.settings import validate_settings

    settings = validate_settings(
        {
            "vr_format": "side_by_side",
            "vr_resolution": "16x9-1080p",
            "per_eye_width": 1920,
            "per_eye_height": 1080,
            "vr_output_width": 3840,
            "vr_output_height": 1080,
        },
        source="explicit",
    )
    save_processing_settings(
        job,
        "demo_1",
        settings,
        {
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "frame_count": 10,
            "duration": 1 / 3,
            "codec": 0,
            "sample_aspect_ratio": "1:1",
            "sample_aspect_ratio_numerator": 1,
            "sample_aspect_ratio_denominator": 1,
        },
        str(src),
        transactional_v5=True,
    )
    out_path, _tl = VideoEncoder._resolve_output_and_timeline(job, str(src), settings)
    assert out_path.parent == job
    assert out_path.suffix == ".mp4"
