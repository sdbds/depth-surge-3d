"""Schema-5 processing settings artifact validation and migration tests."""

from __future__ import annotations

from pathlib import Path
import pytest

from src.depth_surge_3d.core.settings import validate_settings
from src.depth_surge_3d.io.settings_artifact import (
    ProcessingSettingsArtifactV5,
    SettingsArtifactSchemaError,
    create_processing_settings_artifact_v5,
    migrate_processing_settings_artifact_v5,
    validate_portable_final_component_v1,
)


def _sample_artifact(tmp_path: Path) -> ProcessingSettingsArtifactV5:
    settings = validate_settings(
        {
            "vr_format": "side_by_side",
            "vr_resolution": "16x9-1080p",
            "per_eye_width": 1920,
            "per_eye_height": 1080,
            "vr_output_width": 3840,
            "vr_output_height": 1080,
            "video_encoder": "libx265",
            "keep_intermediates": True,
        },
        source="explicit",
    )
    return create_processing_settings_artifact_v5(
        batch_name="clip_1",
        source_video=str(tmp_path / "clip.mp4"),
        source_video_name="clip.mp4",
        source_video_fingerprint="a" * 32,
        video_properties={
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "frame_count": 30,
            "duration": 1.0,
            "codec": 0,
            "sample_aspect_ratio": "1:1",
            "sample_aspect_ratio_numerator": 1,
            "sample_aspect_ratio_denominator": 1,
        },
        processing_settings=settings,
        output_directory=str(tmp_path / "job"),
        project_version="0.9.0",
        creation_now_usec="2026-11-26T00:00:00.000000Z",
    )


def test_settings_artifact_v5_roundtrip_and_transitions(tmp_path: Path) -> None:
    artifact = _sample_artifact(tmp_path)
    raw = artifact.encode()
    reparsed = ProcessingSettingsArtifactV5.parse(raw)
    assert reparsed.to_object() == artifact.to_object()
    assert reparsed.status == "in_progress"
    assert reparsed.selected_frame_count == 30

    completed = reparsed.completed(
        terminal_at="2026-11-26T00:01:00.000000Z",
        frames_processed=30,
        cleanup_status="not_requested",
        metric_clamp_summary=None,
    )
    assert completed.status == "completed"
    assert completed.revision == reparsed.revision + 1

    restarted = completed.restart(
        started_at="2026-11-26T00:02:00.000000Z",
        processing_settings=completed.to_object()["processing_settings"],
    )
    assert restarted.status == "in_progress"


def test_validate_portable_final_component_rejects_traversal_and_reserved() -> None:
    assert validate_portable_final_component_v1("clip_sbs_1080p.mp4") == "clip_sbs_1080p.mp4"
    for bad in ("../escape.mp4", "a/b.mp4", "CON.mp4", "clip.mp4 ", ""):
        with pytest.raises((ValueError, SettingsArtifactSchemaError)):
            validate_portable_final_component_v1(bad)


def test_migrate_legacy_settings_to_v5(tmp_path: Path) -> None:
    migrated_settings = validate_settings(
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
    legacy = {
        "metadata": {
            "batch_name": "legacy_clip",
            "source_video": str(tmp_path / "legacy.mp4"),
            "source_video_name": "legacy.mp4",
            "processing_status": "in_progress",
        },
        "video_properties": {
            "width": 1280,
            "height": 720,
            "fps": 24.0,
            "frame_count": 24,
            "duration": 1.0,
        },
        "processing_settings": {"vr_format": "side_by_side", "vr_resolution": "16x9-1080p"},
        "output_info": {
            "output_directory": str(tmp_path / "job"),
            "expected_output_filename": "legacy_3D_side-by-side_16x9-1080p.mp4",
        },
    }
    artifact = migrate_processing_settings_artifact_v5(
        legacy_settings=legacy,
        migrated_processing_settings=migrated_settings,
        migration_now_usec="2026-11-26T00:00:00.000000Z",
        job_id="1" * 32,
        job_reservation_generation="2" * 32,
        job_terminal_reservation_id="3" * 32,
    )
    assert artifact.status == "in_progress"
    assert artifact.selected_frame_count == 24
