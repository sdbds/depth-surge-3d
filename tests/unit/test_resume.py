"""Stage-level resume inspection and legacy migration tests."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.core.settings import validate_settings
from src.depth_surge_3d.io.operations import save_processing_settings
from src.depth_surge_3d.io.resume import (
    apply_legacy_migration,
    build_resume_report,
    resolve_resume_depth_model_version,
    resolve_resume_source_video,
)
from src.depth_surge_3d.processing.frames.source_frame_manifest import write_source_frame_manifest


def _v5_props(width: int = 64, height: int = 64, frame_count: int = 2) -> dict[str, object]:
    return {
        "width": width,
        "height": height,
        "fps": 30.0,
        "frame_count": frame_count,
        "duration": frame_count / 30.0,
        "codec": 0,
        "sample_aspect_ratio": "1:1",
        "sample_aspect_ratio_numerator": 1,
        "sample_aspect_ratio_denominator": 1,
    }


def test_resolve_resume_depth_model_version_and_source_video(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video-content-0123456789")
    job = tmp_path / "job"
    settings = validate_settings(
        {
            "depth_model_version": "v3",
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
        "clip_1",
        settings,
        _v5_props(),
        str(video),
        transactional_v5=True,
    )

    assert resolve_resume_source_video(job) == video.resolve()
    assert (
        resolve_resume_depth_model_version({"depth_model_version": "v2"}, job, default="v3") == "v2"
    )
    assert (
        resolve_resume_depth_model_version({"model_path": "see_through_model"}, job, default="v3")
        == "see_through"
    )
    assert resolve_resume_depth_model_version({}, job, default="v3") == "v3"


def test_build_resume_report_and_legacy_migration_archive_and_delete(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"sample-video-bytes-xyz")
    job = tmp_path / "job"
    frames_dir = job / "00_original_frames"
    frames_dir.mkdir(parents=True)
    legacy_dir = job / "02_depth_maps"
    legacy_dir.mkdir()
    (legacy_dir / "frame_000001.png").write_bytes(b"old")

    frame_files: list[Path] = []
    for idx in (1, 2):
        fp = frames_dir / f"frame_{idx:06d}.png"
        cv2.imwrite(str(fp), np.zeros((16, 16, 3), dtype=np.uint8))
        frame_files.append(fp)
    write_source_frame_manifest(frame_files, str(video), (0, 2))

    # Write legacy v1 settings file with removed key
    legacy_settings = job / "clip-settings.json"
    legacy_settings.write_text(
        json.dumps(
            {
                "metadata": {
                    "batch_name": "clip",
                    "source_video": str(video),
                    "source_video_name": video.name,
                    "processing_status": "in_progress",
                },
                "video_properties": {
                    "width": 16,
                    "height": 16,
                    "fps": 30.0,
                    "frame_count": 2,
                    "duration": 2 / 30.0,
                },
                "processing_settings": {
                    "vr_format": "side_by_side",
                    "vr_resolution": "16x9-1080p",
                    "baseline": 0.065,
                },
                "output_info": {
                    "output_directory": str(job),
                    "expected_output_filename": "clip_3D_side-by-side_16x9-1080p.mp4",
                },
            }
        ),
        encoding="utf-8",
    )

    current = validate_settings(
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
    report = build_resume_report(job, current, source_video=video)
    assert report.stage("frames").disposition == "preserve"
    assert report.stage("legacy_depth_maps").disposition == "invalidate"
    assert "baseline" in report.removed_settings
    assert report.settings_backup_required is True

    apply_legacy_migration(report, "archive")
    assert (job / "legacy_v1" / "02_depth_maps").is_dir()
    assert (job / "settings.legacy.json").is_file()
    assert (job / "processing-settings-v5.json").is_file()
    assert not legacy_dir.exists()


def test_build_resume_report_rejects_mismatched_source_video(tmp_path: Path) -> None:
    video1 = tmp_path / "v1.mp4"
    video2 = tmp_path / "v2.mp4"
    video1.write_bytes(b"first-video-payload")
    video2.write_bytes(b"second-different-video")
    job = tmp_path / "job"
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
        "v1",
        settings,
        _v5_props(),
        str(video1),
        transactional_v5=True,
    )
    with pytest.raises(ValueError, match="Cannot resume"):
        build_resume_report(job, settings, source_video=video2)
