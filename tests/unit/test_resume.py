"""Stage-aware resume reporting and one-way legacy migration tests."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.core.file_identity import (
    FILE_IDENTITY_ALGORITHM_VERSION,
    file_sample_fingerprint,
)
from src.depth_surge_3d.core.settings import PROCESSING_SETTINGS_SCHEMA_VERSION, validate_settings
from src.depth_surge_3d.processing.frames.depth_storage import (
    RAW_DEPTH_READABLE_SCHEMA_VERSIONS,
    RAW_DEPTH_SCHEMA_VERSION,
    RawDepthStore,
    canonical_json_hash,
    depth_preprocessing_algorithm,
    select_depth_model_settings,
)
from src.depth_surge_3d.processing.frames.depth_processor import (
    CANONICAL_DEPTH_ALGORITHM_VERSION,
    CANONICAL_DEPTH_SCHEMA_VERSION,
    DEPTH_BOUNDS_SCHEMA_VERSION,
)
from src.depth_surge_3d.processing.frames.scene_analyzer import (
    SCENE_ALGORITHM_VERSION,
    SCENE_SCHEMA_VERSION,
)
from src.depth_surge_3d.processing.frames.stereo_generator import (
    STEREO_STAGE_ALGORITHM_VERSION,
    STEREO_STAGE_SCHEMA_VERSION,
)
from src.depth_surge_3d.processing.frames.source_frame_manifest import (
    frame_sequence_fingerprint,
    write_source_frame_manifest,
)


def _write_frames(output_dir: Path, count: int = 2) -> tuple[list[Path], str]:
    frames_dir = output_dir / "00_original_frames"
    frames_dir.mkdir(parents=True)
    frame_files = []
    for index in range(count):
        path = frames_dir / f"frame_{index + 1:06d}.png"
        image = np.full((4, 6, 3), index * 40, dtype=np.uint8)
        assert cv2.imwrite(str(path), image)
        frame_files.append(path)
    return frame_files, frame_sequence_fingerprint(frame_files)


def _write_settings(
    output_dir: Path,
    processing_settings: dict,
    *,
    current_schema: bool = False,
    include_source_fingerprint: bool = True,
) -> Path:
    source_video = output_dir / "source.mp4"
    if not source_video.exists():
        source_video.write_bytes(b"source-video")
    metadata = {
        "batch_name": "job",
        "source_video": str(source_video),
        "processing_status": "in_progress",
    }
    if include_source_fingerprint:
        metadata["source_video_fingerprint_algorithm"] = FILE_IDENTITY_ALGORITHM_VERSION
        metadata["source_video_fingerprint"] = file_sample_fingerprint(source_video)
    if current_schema:
        metadata["settings_schema_version"] = PROCESSING_SETTINGS_SCHEMA_VERSION
    payload = {
        "metadata": metadata,
        "video_properties": {"frame_count": 2, "width": 6, "height": 4, "fps": 30.0},
        "processing_settings": processing_settings,
    }
    path = output_dir / "job-settings.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    frame_files = sorted((output_dir / "00_original_frames").glob("frame_*.png"))
    if frame_files:
        write_source_frame_manifest(frame_files, source_video)
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
    settings: dict | None = None,
    schema_version: int = RAW_DEPTH_SCHEMA_VERSION,
    camera_model: str = "none",
) -> None:
    raw_dir = output_dir / "02_depth_raw"
    raw_dir.mkdir()
    selected_settings = settings or _current_settings(model_size=model_size)
    semantic = {
        "backend": "test.Estimator",
        "model_info": {"revision": "immutable"},
        "depth_settings": select_depth_model_settings(selected_settings),
        "weight_sha256": None,
        "artifact_identity": "immutable",
        "source_frame_fingerprint": source_fingerprint,
        "preprocessing_algorithm": depth_preprocessing_algorithm(selected_settings),
    }
    if schema_version == 3:
        semantic["camera_model"] = camera_model
    metadata = {
        "schema_version": schema_version,
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
        "promoted_frame_count": 0,
    }
    if schema_version == 3:
        metadata["camera_model"] = camera_model
    metadata["fingerprint"] = RawDepthStore._fingerprint(metadata)
    (raw_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _write_current_depth_pipeline(
    output_dir: Path,
    frame_files: list[Path],
    source_fingerprint: str,
    *,
    settings: dict | None = None,
    schema_version: int = RAW_DEPTH_SCHEMA_VERSION,
    camera_model: str = "none",
) -> tuple[dict, dict, dict]:
    _write_candidate_manifest(output_dir, frame_files, source_fingerprint)
    _write_raw_metadata(
        output_dir,
        frame_files,
        source_fingerprint,
        model_size="large",
        settings=settings,
        schema_version=schema_version,
        camera_model=camera_model,
    )
    raw_dir = output_dir / "02_depth_raw"
    raw_metadata_path = raw_dir / "metadata.json"
    raw_metadata = json.loads(raw_metadata_path.read_text(encoding="utf-8"))
    scene_dir = output_dir / "01_scene_data"
    manifest_path = scene_dir / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bounds = {
        "schema_version": DEPTH_BOUNDS_SCHEMA_VERSION,
        "algorithm_version": CANONICAL_DEPTH_ALGORITHM_VERSION,
        "source_raw_fingerprint": raw_metadata["fingerprint"],
        "sample_fingerprint": "samples",
        "scenes": {"0": {"low": 0.0, "high": 1.0}},
    }
    bounds["fingerprint"] = canonical_json_hash(bounds)
    (scene_dir / "depth_bounds.json").write_text(json.dumps(bounds), encoding="utf-8")
    manifest.update(
        {
            "status": "final",
            "sample_fingerprint": "samples",
            "bounds_fingerprint": bounds["fingerprint"],
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    for index, frame in enumerate(frame_files):
        with (raw_dir / f"{frame.stem}.npz").open("wb") as handle:
            arrays = {"values": np.full((4, 6), index, dtype=np.float16)}
            if camera_model == "pinhole_fx":
                arrays["focal_x_normalized"] = np.array(0.8, dtype=np.float32)
            np.savez_compressed(handle, **arrays)
    raw_metadata = json.loads(raw_metadata_path.read_text(encoding="utf-8"))
    raw_metadata["completed_count"] = len(frame_files)
    raw_metadata_path.write_text(json.dumps(raw_metadata), encoding="utf-8")

    canonical_dir = output_dir / "03_disparity_maps"
    canonical_dir.mkdir()
    canonical = {
        "schema_version": CANONICAL_DEPTH_SCHEMA_VERSION,
        "algorithm_version": CANONICAL_DEPTH_ALGORITHM_VERSION,
        "representation": "relative_disparity",
        "near_value": 1.0,
        "far_value": 0.0,
        "encoding": "uint16_png",
        "encoding_scale": 65535.0,
        "num_frames": len(frame_files),
        "frame_names": [path.name for path in frame_files],
        "native_shape": [4, 6],
        "source_raw_fingerprint": raw_metadata["fingerprint"],
        "source_model_fingerprint": canonical_json_hash(raw_metadata["semantic_fingerprint"]),
        "scene_manifest_fingerprint": canonical_json_hash(manifest),
        "depth_bounds_fingerprint": bounds["fingerprint"],
    }
    canonical["fingerprint"] = canonical_json_hash(canonical)
    (canonical_dir / "metadata.json").write_text(json.dumps(canonical), encoding="utf-8")
    for index, frame in enumerate(frame_files):
        assert cv2.imwrite(
            str(canonical_dir / f"{frame.stem}.png"),
            np.full((4, 6), index * 1000, dtype=np.uint16),
        )
    return manifest, bounds, canonical


def _write_current_stereo_pipeline(
    output_dir: Path,
    frame_files: list[Path],
    canonical: dict,
    *,
    algorithm_version: str = STEREO_STAGE_ALGORITHM_VERSION,
) -> None:
    left_dir = output_dir / "04_left_frames"
    right_dir = output_dir / "04_right_frames"
    left_dir.mkdir()
    right_dir.mkdir()
    settings = _current_settings()
    metadata = {
        "schema_version": STEREO_STAGE_SCHEMA_VERSION,
        "algorithm_version": algorithm_version,
        "source_canonical_fingerprint": canonical["fingerprint"],
        "frame_names": [path.name for path in frame_files],
        "render_shape": [4, 6],
        "render_settings": {
            "stereo_strength": settings["stereo_strength"],
            "convergence": settings["convergence"],
            "occlusion_fill": settings["occlusion_fill"],
        },
        "renderer_device_type": "cpu",
        "encoding": "uint8_png",
    }
    metadata["fingerprint"] = canonical_json_hash(metadata)
    (left_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for index, frame in enumerate(frame_files):
        image = np.full((4, 6, 3), index * 20, dtype=np.uint8)
        assert cv2.imwrite(str(left_dir / f"{frame.stem}.png"), image)
        assert cv2.imwrite(str(right_dir / f"{frame.stem}.png"), image)


def _write_downstream_placeholders(output_dir: Path, frame_name: str) -> None:
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    for name in (
        "05_left_distorted",
        "05_right_distorted",
        "06_left_cropped",
        "06_right_cropped",
        "07_left_upscaled",
        "07_right_upscaled",
        "99_vr_frames",
    ):
        directory = output_dir / name
        directory.mkdir()
        assert cv2.imwrite(str(directory / frame_name), image)


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


def test_raw_mismatch_accepts_schema_v2_for_camera_free_backend(tmp_path):
    from src.depth_surge_3d.io.resume import _raw_mismatch_reason

    frame_files, fingerprint = _write_frames(tmp_path)
    settings = _current_settings()
    _write_raw_metadata(
        tmp_path,
        frame_files,
        fingerprint,
        model_size="large",
        settings=settings,
        schema_version=2,
    )
    metadata = RawDepthStore.read_metadata(tmp_path / "02_depth_raw")

    assert RAW_DEPTH_READABLE_SCHEMA_VERSIONS == frozenset({2, 3})
    assert metadata is not None
    assert _raw_mismatch_reason(metadata, settings, frame_files, fingerprint, None) is None


def test_resume_preserves_valid_schema_v2_metadata_and_payload_bytes(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    frame_files, fingerprint = _write_frames(tmp_path)
    settings = _current_settings()
    _write_settings(tmp_path, settings, current_schema=True)
    _write_current_depth_pipeline(
        tmp_path,
        frame_files,
        fingerprint,
        settings=settings,
        schema_version=2,
    )
    raw_dir = tmp_path / "02_depth_raw"
    metadata_before = (raw_dir / "metadata.json").read_bytes()
    payloads_before = [path.read_bytes() for path in sorted(raw_dir.glob("*.npz"))]

    report = build_resume_report(tmp_path, settings)

    assert report.stage("depth_raw").disposition == "preserve"
    assert (raw_dir / "metadata.json").read_bytes() == metadata_before
    assert [path.read_bytes() for path in sorted(raw_dir.glob("*.npz"))] == payloads_before


@pytest.mark.parametrize(
    ("schema_version", "camera_model"),
    [(2, "none"), (3, "none")],
)
def test_moge_resume_requires_v3_pinhole_camera_metadata(tmp_path, schema_version, camera_model):
    from src.depth_surge_3d.io.resume import _raw_mismatch_reason

    frame_files, fingerprint = _write_frames(tmp_path)
    settings = _current_settings()
    settings["depth_model_version"] = "moge2"
    _write_raw_metadata(
        tmp_path,
        frame_files,
        fingerprint,
        model_size="large",
        settings=settings,
        schema_version=schema_version,
        camera_model=camera_model,
    )
    metadata = RawDepthStore.read_metadata(tmp_path / "02_depth_raw")

    assert metadata is not None
    reason = _raw_mismatch_reason(metadata, settings, frame_files, fingerprint, None)
    assert reason is not None
    assert "pinhole_fx" in reason


def test_moge_resume_accepts_v3_pinhole_camera_metadata(tmp_path):
    from src.depth_surge_3d.io.resume import _raw_mismatch_reason

    frame_files, fingerprint = _write_frames(tmp_path)
    settings = _current_settings()
    settings["depth_model_version"] = "moge2"
    _write_raw_metadata(
        tmp_path,
        frame_files,
        fingerprint,
        model_size="large",
        settings=settings,
        schema_version=3,
        camera_model="pinhole_fx",
    )
    metadata = RawDepthStore.read_metadata(tmp_path / "02_depth_raw")

    assert metadata is not None
    assert _raw_mismatch_reason(metadata, settings, frame_files, fingerprint, None) is None


def test_raw_model_fingerprint_mismatch_recomputes_depth_derived_scene_data(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    frame_files, fingerprint = _write_frames(tmp_path)
    _write_settings(tmp_path, _current_settings(), current_schema=True)
    _write_current_depth_pipeline(tmp_path, frame_files, fingerprint)

    report = build_resume_report(tmp_path, _current_settings(model_size="base"))

    assert report.stage("depth_raw").disposition == "invalidate"
    assert report.stage("scene_data").disposition == "resume"
    assert "raw depth" in report.stage("scene_data").reason
    assert report.stage("disparity_maps").disposition == "invalidate"


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


def test_source_video_fingerprint_mismatch_aborts_resume(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    _write_frames(tmp_path)
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"first source payload")
    _write_settings(tmp_path, _current_settings(), current_schema=True)
    source_video.write_bytes(b"different source payload")

    with pytest.raises(ValueError, match="source video fingerprint mismatch"):
        build_resume_report(tmp_path, _current_settings())


def test_same_shape_frame_tampering_invalidates_original_frames(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    frame_files, _ = _write_frames(tmp_path)
    _write_settings(tmp_path, _current_settings(), current_schema=True)
    replacement = np.full((4, 6, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(frame_files[0]), replacement)

    report = build_resume_report(tmp_path, _current_settings())

    assert report.stage("frames").disposition == "invalidate"
    assert "frame fingerprint" in report.stage("frames").reason


def test_invalid_original_frames_never_enter_migration_set(tmp_path):
    from src.depth_surge_3d.io.resume import apply_legacy_migration, build_resume_report

    frame_files, _ = _write_frames(tmp_path)
    _write_settings(tmp_path, _current_settings(), current_schema=True)
    replacement = np.full((4, 6, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(frame_files[0]), replacement)

    report = build_resume_report(tmp_path, _current_settings(migrate_legacy="delete"))

    assert report.stage("frames").disposition == "invalidate"
    assert (tmp_path / "00_original_frames") not in report.migration_paths
    apply_legacy_migration(report, "delete")
    assert frame_files[0].is_file()


def test_matching_settings_backup_allows_crash_resume(tmp_path):
    from src.depth_surge_3d.io.resume import apply_legacy_migration, build_resume_report

    settings_file, _ = _legacy_job(tmp_path)
    (tmp_path / "settings.legacy.json").write_bytes(settings_file.read_bytes())
    report = build_resume_report(tmp_path, _current_settings())

    apply_legacy_migration(report, "archive")

    migrated = json.loads(settings_file.read_text(encoding="utf-8"))
    assert "baseline" not in migrated["processing_settings"]


def test_settings_rewrite_failure_rolls_back_archived_stages(tmp_path, monkeypatch):
    import src.depth_surge_3d.io.resume as resume

    settings_file, legacy_depth = _legacy_job(tmp_path)
    original_settings = settings_file.read_bytes()
    report = resume.build_resume_report(tmp_path, _current_settings())

    def fail_write(*_args, **_kwargs):
        raise OSError("simulated settings write failure")

    monkeypatch.setattr(resume, "_atomic_write_json", fail_write)

    with pytest.raises(OSError, match="simulated settings"):
        resume.apply_legacy_migration(report, "archive")

    assert legacy_depth.is_dir()
    assert (tmp_path / "04_left_frames").is_dir()
    assert not (tmp_path / "legacy_v1" / "02_depth_maps").exists()
    assert settings_file.read_bytes() == original_settings


def test_missing_source_fingerprint_invalidates_frames_instead_of_aborting(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    _write_frames(tmp_path)
    _write_settings(
        tmp_path,
        _current_settings(),
        current_schema=True,
        include_source_fingerprint=False,
    )

    report = build_resume_report(tmp_path, _current_settings())

    assert report.stage("frames").disposition == "invalidate"
    assert "source video fingerprint is missing" in report.stage("frames").reason


def test_resume_source_resolver_accepts_legacy_metadata_without_fingerprint(tmp_path):
    from src.depth_surge_3d.io.resume import resolve_resume_source_video

    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"legacy-source")
    settings_file = tmp_path / "job-settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "metadata": {
                    "source_video": str(source_video),
                    "source_video_name": source_video.name,
                },
                "processing_settings": {},
            }
        ),
        encoding="utf-8",
    )

    assert (
        resolve_resume_source_video(
            tmp_path,
            settings_file=settings_file,
        )
        == source_video.resolve()
    )


def test_actual_resume_source_must_match_saved_source(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    _write_frames(tmp_path)
    _write_settings(tmp_path, _current_settings(), current_schema=True)
    other_source = tmp_path / "other.mp4"
    other_source.write_bytes(b"different-source")

    with pytest.raises(ValueError, match="source video fingerprint mismatch"):
        build_resume_report(
            tmp_path,
            _current_settings(),
            source_video=other_source,
        )


def test_noncontiguous_frame_manifest_is_invalid(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    frame_files, _ = _write_frames(tmp_path)
    frame_files[1].rename(frame_files[1].with_name("frame_000003.png"))
    _write_settings(tmp_path, _current_settings(), current_schema=True)

    report = build_resume_report(tmp_path, _current_settings())

    assert report.stage("frames").disposition == "invalidate"
    assert "manifest" in report.stage("frames").reason


def test_scene_manifest_must_match_current_scene_settings(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    frame_files, fingerprint = _write_frames(tmp_path)
    _write_settings(tmp_path, _current_settings(), current_schema=True)
    _write_candidate_manifest(tmp_path, frame_files, fingerprint)

    report = build_resume_report(tmp_path, _current_settings(scene_cut_threshold=0.7))

    assert report.stage("scene_data").disposition == "invalidate"
    assert "settings" in report.stage("scene_data").reason


def test_scene_manifest_requires_one_scene_id_per_frame(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    frame_files, fingerprint = _write_frames(tmp_path)
    _write_settings(tmp_path, _current_settings(), current_schema=True)
    _write_candidate_manifest(tmp_path, frame_files, fingerprint)
    manifest_path = tmp_path / "01_scene_data" / "scene_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["scene_ids"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = build_resume_report(tmp_path, _current_settings())

    assert report.stage("scene_data").disposition == "invalidate"
    assert "scene IDs" in report.stage("scene_data").reason


def test_corrupt_completed_raw_payload_invalidates_raw_stage(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    frame_files, fingerprint = _write_frames(tmp_path)
    _write_settings(tmp_path, _current_settings(), current_schema=True)
    _write_candidate_manifest(tmp_path, frame_files, fingerprint)
    _write_raw_metadata(tmp_path, frame_files, fingerprint, model_size="large")
    metadata_path = tmp_path / "02_depth_raw" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["completed_count"] = 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / "02_depth_raw" / "frame_000001.npz").write_bytes(b"corrupt")

    report = build_resume_report(tmp_path, _current_settings())

    assert report.stage("depth_raw").disposition == "invalidate"
    assert "payload" in report.stage("depth_raw").reason


def test_current_model_fingerprint_is_compared_outside_user_settings(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    frame_files, fingerprint = _write_frames(tmp_path)
    _write_settings(tmp_path, _current_settings(), current_schema=True)
    _write_candidate_manifest(tmp_path, frame_files, fingerprint)
    _write_raw_metadata(tmp_path, frame_files, fingerprint, model_size="large")
    expected_model = {
        "backend": "test.Estimator",
        "model_info": {"revision": "different"},
        "depth_settings": {
            "depth_model_version": "v3",
            "model_size": "large",
            "depth_resolution": "auto",
            "device": "cpu",
        },
        "weight_sha256": None,
        "artifact_identity": "different",
    }

    report = build_resume_report(
        tmp_path,
        _current_settings(),
        model_fingerprint=expected_model,
    )

    assert report.stage("depth_raw").disposition == "invalidate"
    assert "model fingerprint" in report.stage("depth_raw").reason


def test_final_bounds_must_cover_exact_manifest_scene_set(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    frame_files, fingerprint = _write_frames(tmp_path)
    _write_settings(tmp_path, _current_settings(), current_schema=True)
    manifest, bounds, _ = _write_current_depth_pipeline(tmp_path, frame_files, fingerprint)
    bounds["scenes"] = {"1": {"low": 0.0, "high": 1.0}}
    bounds["fingerprint"] = canonical_json_hash(
        {key: value for key, value in bounds.items() if key != "fingerprint"}
    )
    (tmp_path / "01_scene_data" / "depth_bounds.json").write_text(
        json.dumps(bounds), encoding="utf-8"
    )
    manifest["bounds_fingerprint"] = bounds["fingerprint"]
    (tmp_path / "01_scene_data" / "scene_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    report = build_resume_report(tmp_path, _current_settings())

    assert report.stage("scene_data").disposition == "invalidate"
    assert "bounds" in report.stage("scene_data").reason


def test_corrupt_canonical_payload_resumes_disparity_stage(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    frame_files, fingerprint = _write_frames(tmp_path)
    _write_settings(tmp_path, _current_settings(), current_schema=True)
    _write_current_depth_pipeline(tmp_path, frame_files, fingerprint)
    (tmp_path / "03_disparity_maps" / "frame_000002.png").write_bytes(b"corrupt")

    report = build_resume_report(tmp_path, _current_settings())

    assert report.stage("disparity_maps").disposition == "resume"
    assert "payload" in report.stage("disparity_maps").reason


def test_corrupt_stereo_payload_resumes_stereo_stage(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    frame_files, fingerprint = _write_frames(tmp_path)
    _write_settings(tmp_path, _current_settings(), current_schema=True)
    _, _, canonical = _write_current_depth_pipeline(tmp_path, frame_files, fingerprint)
    _write_current_stereo_pipeline(tmp_path, frame_files, canonical)
    (tmp_path / "04_right_frames" / "frame_000001.png").write_bytes(b"corrupt")

    report = build_resume_report(tmp_path, _current_settings())

    assert report.stage("stereo").disposition == "resume"
    assert "payload" in report.stage("stereo").reason


def test_v1_stereo_metadata_preserves_upstream_and_invalidates_frame_stages(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    frame_files, fingerprint = _write_frames(tmp_path)
    settings = _current_settings()
    _write_settings(tmp_path, settings, current_schema=True)
    _, _, canonical = _write_current_depth_pipeline(tmp_path, frame_files, fingerprint)
    _write_current_stereo_pipeline(
        tmp_path,
        frame_files,
        canonical,
        algorithm_version="torch-forward-splat-v1",
    )
    _write_downstream_placeholders(tmp_path, frame_files[0].name)

    report = build_resume_report(tmp_path, settings)

    assert report.stage("frames").disposition == "preserve"
    assert report.stage("depth_raw").disposition == "preserve"
    assert report.stage("disparity_maps").disposition == "preserve"
    assert report.stage("stereo").disposition == "invalidate"
    for stage_name in ("distortion", "crop", "upscale", "vr_frames"):
        assert report.stage(stage_name).disposition == "invalidate"


def test_float32_promotion_resumes_raw_and_invalidates_downstream(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    frame_files, fingerprint = _write_frames(tmp_path)
    _write_settings(tmp_path, _current_settings(), current_schema=True)
    _, _, canonical = _write_current_depth_pipeline(tmp_path, frame_files, fingerprint)
    _write_current_stereo_pipeline(tmp_path, frame_files, canonical)

    report = build_resume_report(
        tmp_path,
        _current_settings(raw_storage_dtype="float32"),
    )

    assert report.stage("depth_raw").disposition == "resume"
    assert "promotion" in report.stage("depth_raw").reason
    assert report.stage("disparity_maps").disposition == "invalidate"
    assert report.stage("stereo").disposition == "invalidate"


def test_delete_cleanup_failure_does_not_roll_back_committed_migration(tmp_path, monkeypatch):
    import src.depth_surge_3d.io.resume as resume

    settings_file, legacy_depth = _legacy_job(tmp_path)
    report = resume.build_resume_report(
        tmp_path,
        _current_settings(migrate_legacy="delete"),
    )
    original_rmtree = resume.shutil.rmtree

    def partially_remove_staging(path):
        staging = Path(path)
        first_entry = sorted(staging.iterdir())[0]
        original_rmtree(first_entry)
        raise OSError("simulated cleanup interruption")

    monkeypatch.setattr(resume.shutil, "rmtree", partially_remove_staging)

    with pytest.warns(RuntimeWarning, match="committed.*cleanup"):
        resume.apply_legacy_migration(report, "delete")

    assert not legacy_depth.exists()
    assert not (tmp_path / "04_left_frames").exists()
    assert (tmp_path / ".resume_delete_staging").is_dir()
    migrated = json.loads(settings_file.read_text(encoding="utf-8"))
    assert "baseline" not in migrated["processing_settings"]


def test_generated_stage_without_completion_manifest_is_invalidated(tmp_path):
    from src.depth_surge_3d.io.resume import _validate_generated_stage

    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    payload = np.zeros((4, 6, 3), dtype=np.uint8)
    assert cv2.imwrite(str(left / "frame_000001.png"), payload)
    assert cv2.imwrite(str(right / "frame_000001.png"), payload)

    stage = _validate_generated_stage(
        "crop",
        (left, right),
        upstream_reusable=True,
        changed_setting=None,
        reusable_reason="crop settings match",
    )

    assert stage.disposition == "invalidate"
    assert "manifest" in stage.reason


@pytest.mark.parametrize(
    "manifest",
    [None, {"source_frame_fingerprint": 7}],
)
def test_frame_stage_explicitly_invalidates_malformed_manifest(tmp_path, monkeypatch, manifest):
    import src.depth_surge_3d.io.resume as resume

    _write_frames(tmp_path)
    settings_data = {"metadata": {"source_video_fingerprint": "source-fingerprint"}}
    monkeypatch.setattr(resume, "read_source_frame_manifest", lambda _directory: manifest)
    monkeypatch.setattr(
        resume,
        "source_frame_manifest_mismatch_reason",
        lambda *_arguments: None,
    )

    stage, _frame_files, fingerprint = resume._validate_frame_stage(
        tmp_path,
        settings_data,
        {},
    )

    assert stage.disposition == "invalidate"
    assert "manifest" in stage.reason
    assert fingerprint is None
