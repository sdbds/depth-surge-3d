"""Stage-aware resume reporting and one-way legacy migration tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.core.settings import validate_settings
from src.depth_surge_3d.processing.frames.depth_storage import RawDepthStore
from src.depth_surge_3d.processing.frames.scene_analyzer import (
    SCENE_ALGORITHM_VERSION,
    SCENE_SCHEMA_VERSION,
)


def _write_frames(output_dir: Path, count: int = 2) -> tuple[list[Path], str]:
    frames_dir = output_dir / "00_original_frames"
    frames_dir.mkdir(parents=True)
    frame_files = []
    hasher = hashlib.sha256()
    for index in range(count):
        path = frames_dir / f"frame_{index + 1:06d}.png"
        image = np.full((4, 6, 3), index * 40, dtype=np.uint8)
        assert cv2.imwrite(str(path), image)
        frame_files.append(path)
        hasher.update(path.name.encode("utf-8"))
        hasher.update(path.read_bytes())
    return frame_files, hasher.hexdigest()


def _write_settings(
    output_dir: Path,
    processing_settings: dict,
    *,
    current_schema: bool = False,
) -> Path:
    metadata = {
        "batch_name": "job",
        "source_video": str(output_dir / "source.mp4"),
        "processing_status": "in_progress",
    }
    if current_schema:
        metadata["settings_schema_version"] = 2
    payload = {
        "metadata": metadata,
        "video_properties": {"frame_count": 2, "width": 6, "height": 4, "fps": 30.0},
        "processing_settings": processing_settings,
    }
    path = output_dir / "job-settings.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _current_settings(**overrides) -> dict:
    values = {
        "depth_model_version": "v3",
        "model_size": "large",
        "depth_resolution": "auto",
        "device": "cpu",
        "target_fps": 30,
    }
    values.update(overrides)
    return validate_settings(values, source="explicit")


def _write_candidate_manifest(output_dir: Path, frame_files: list[Path], fingerprint: str) -> None:
    scene_dir = output_dir / "01_scene_data"
    scene_dir.mkdir()
    manifest = {
        "schema_version": SCENE_SCHEMA_VERSION,
        "algorithm_version": SCENE_ALGORITHM_VERSION,
        "status": "candidate",
        "frame_names": [path.name for path in frame_files],
        "scene_ids": [0] * len(frame_files),
        "candidate_cuts": [],
        "settings": {"enabled": True, "threshold": 0.55, "min_frames": 8},
        "source_frame_fingerprint": fingerprint,
    }
    (scene_dir / "scene_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_raw_metadata(
    output_dir: Path,
    frame_files: list[Path],
    source_fingerprint: str,
    *,
    model_size: str,
) -> None:
    raw_dir = output_dir / "02_depth_raw"
    raw_dir.mkdir()
    semantic = {
        "backend": "test.Estimator",
        "model_info": {"revision": "immutable"},
        "depth_settings": {
            "depth_model_version": "v3",
            "model_size": model_size,
            "depth_resolution": "auto",
            "device": "cpu",
        },
        "weight_sha256": None,
        "artifact_identity": "immutable",
        "source_frame_fingerprint": source_fingerprint,
        "preprocessing_algorithm": "native-depth-adapter-v2",
    }
    metadata = {
        "schema_version": 1,
        "storage_status": "ready",
        "representation": "relative_depth",
        "frame_names": [path.name for path in frame_files],
        "native_shape": [4, 6],
        "requested_dtype": "auto",
        "selected_dtype": "float16",
        "storage_provenance": "native_float16",
        "compression": "npz_deflate",
        "semantic_fingerprint": semantic,
        "completed_count": 0,
    }
    metadata["fingerprint"] = RawDepthStore._fingerprint(metadata)
    (raw_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _legacy_job(tmp_path: Path):
    frame_files, _ = _write_frames(tmp_path)
    settings_file = _write_settings(
        tmp_path,
        {
            "baseline": 0.065,
            "focal_length": 1000,
            "hole_fill_quality": "fast",
            "vr_format": "side_by_side",
        },
    )
    legacy_depth = tmp_path / "02_depth_maps"
    legacy_depth.mkdir()
    (legacy_depth / "frame_000001.png").write_bytes(b"legacy-depth")
    for name in ("04_left_frames", "04_right_frames", "08_left_final", "08_right_final"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / frame_files[0].name).write_bytes(b"legacy-stereo")
    return settings_file, legacy_depth


def test_report_preserves_original_frames_and_lists_legacy_stages(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    _legacy_job(tmp_path)

    report = build_resume_report(tmp_path, _current_settings())

    assert report.stage("frames").disposition == "preserve"
    assert "frames" in report.preserved_stage_names
    assert "legacy_depth_maps" in report.invalidated_stage_names
    assert "stereo" in report.invalidated_stage_names
    assert "legacy_final" in report.invalidated_stage_names
    assert set(report.removed_settings) == {"baseline", "focal_length", "hole_fill_quality"}


def test_candidate_manifest_resumes_finalization_but_cannot_reuse_canonical(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    frame_files, fingerprint = _write_frames(tmp_path)
    _write_settings(tmp_path, _current_settings(), current_schema=True)
    _write_candidate_manifest(tmp_path, frame_files, fingerprint)
    canonical_dir = tmp_path / "03_disparity_maps"
    canonical_dir.mkdir()
    (canonical_dir / "metadata.json").write_text("{}", encoding="utf-8")

    report = build_resume_report(tmp_path, _current_settings())

    assert report.stage("scene_data").disposition == "resume"
    assert report.stage("disparity_maps").disposition == "invalidate"
    assert "candidate" in report.stage("disparity_maps").reason


def test_raw_model_fingerprint_mismatch_invalidates_raw_and_downstream(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    frame_files, fingerprint = _write_frames(tmp_path)
    _write_settings(tmp_path, _current_settings(), current_schema=True)
    _write_candidate_manifest(tmp_path, frame_files, fingerprint)
    _write_raw_metadata(tmp_path, frame_files, fingerprint, model_size="base")

    report = build_resume_report(tmp_path, _current_settings(model_size="large"))

    assert report.stage("depth_raw").disposition == "invalidate"
    assert "model_size" in report.stage("depth_raw").reason


def test_default_migration_archives_invalid_generated_data_and_keeps_frames(tmp_path):
    from src.depth_surge_3d.io.resume import apply_legacy_migration, build_resume_report

    _, legacy_depth = _legacy_job(tmp_path)
    report = build_resume_report(tmp_path, _current_settings())

    apply_legacy_migration(report, "archive")

    assert not legacy_depth.exists()
    assert (tmp_path / "legacy_v1" / "02_depth_maps" / "frame_000001.png").is_file()
    assert (tmp_path / "legacy_v1" / "08_left_final" / "frame_000001.png").is_file()
    assert (tmp_path / "00_original_frames" / "frame_000001.png").is_file()


def test_explicit_delete_removes_invalid_generated_data_only(tmp_path):
    from src.depth_surge_3d.io.resume import apply_legacy_migration, build_resume_report

    _, legacy_depth = _legacy_job(tmp_path)
    report = build_resume_report(tmp_path, _current_settings(migrate_legacy="delete"))

    apply_legacy_migration(report, "delete")

    assert not legacy_depth.exists()
    assert not (tmp_path / "legacy_v1").exists()
    assert (tmp_path / "00_original_frames" / "frame_000001.png").is_file()


def test_archive_collision_fails_before_mutating_any_source(tmp_path):
    from src.depth_surge_3d.io.resume import apply_legacy_migration, build_resume_report

    _, legacy_depth = _legacy_job(tmp_path)
    collision = tmp_path / "legacy_v1" / "02_depth_maps"
    collision.mkdir(parents=True)

    report = build_resume_report(tmp_path, _current_settings())

    with pytest.raises(FileExistsError, match="02_depth_maps"):
        apply_legacy_migration(report, "archive")

    assert legacy_depth.is_dir()
    assert (tmp_path / "04_left_frames").is_dir()


def test_legacy_settings_are_backed_up_and_rewritten_without_removed_names(tmp_path):
    from src.depth_surge_3d.io.resume import (
        PROCESSING_SETTINGS_SCHEMA_VERSION,
        apply_legacy_migration,
        build_resume_report,
    )

    settings_file, _ = _legacy_job(tmp_path)
    original_bytes = settings_file.read_bytes()
    report = build_resume_report(tmp_path, _current_settings(stereo_strength=3.0))

    apply_legacy_migration(report, "archive")

    assert (tmp_path / "settings.legacy.json").read_bytes() == original_bytes
    migrated = json.loads(settings_file.read_text(encoding="utf-8"))
    assert migrated["metadata"]["settings_schema_version"] == PROCESSING_SETTINGS_SCHEMA_VERSION
    assert migrated["processing_settings"]["stereo_strength"] == 3.0
    assert "baseline" not in migrated["processing_settings"]
    assert "focal_length" not in migrated["processing_settings"]
    assert "hole_fill_quality" not in migrated["processing_settings"]


def test_source_video_fingerprint_mismatch_invalidates_original_frames(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    _write_frames(tmp_path)
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"first source payload")
    settings_file = _write_settings(tmp_path, _current_settings(), current_schema=True)
    settings_data = json.loads(settings_file.read_text(encoding="utf-8"))
    settings_data["metadata"]["source_video_sha256"] = hashlib.sha256(
        source_video.read_bytes()
    ).hexdigest()
    settings_file.write_text(json.dumps(settings_data), encoding="utf-8")
    source_video.write_bytes(b"different source payload")

    report = build_resume_report(tmp_path, _current_settings())

    assert report.stage("frames").disposition == "invalidate"
    assert "source video fingerprint" in report.stage("frames").reason


def test_matching_settings_backup_allows_crash_resume(tmp_path):
    from src.depth_surge_3d.io.resume import apply_legacy_migration, build_resume_report

    settings_file, _ = _legacy_job(tmp_path)
    (tmp_path / "settings.legacy.json").write_bytes(settings_file.read_bytes())
    report = build_resume_report(tmp_path, _current_settings())

    apply_legacy_migration(report, "archive")

    migrated = json.loads(settings_file.read_text(encoding="utf-8"))
    assert "baseline" not in migrated["processing_settings"]
