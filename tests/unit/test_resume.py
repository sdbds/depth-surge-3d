"""Stage-aware resume reporting and one-way legacy migration tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.core.file_identity import (
    FILE_IDENTITY_ALGORITHM_VERSION,
    file_sample_fingerprint,
)
from src.depth_surge_3d.core.settings import (
    PROCESSING_SETTINGS_SCHEMA_VERSION,
    UnsupportedSettingsSchemaError,
    validate_settings,
)
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
from src.depth_surge_3d.processing.frames.metric_geometry import (
    ClipConvergence,
    MetricGeometryFrame,
    MetricGeometryStore,
)
from src.depth_surge_3d.processing.frames.scene_analyzer import (
    SCENE_ALGORITHM_VERSION,
    SCENE_SCHEMA_VERSION,
)
from src.depth_surge_3d.processing.frames.stereo_generator import (
    METRIC_PROJECTION_ALGORITHM_VERSION,
    STEREO_STAGE_ALGORITHM_VERSION,
    STEREO_STAGE_SCHEMA_VERSION,
)
from src.depth_surge_3d.processing.frames.source_frame_manifest import (
    frame_sequence_fingerprint,
    write_source_frame_manifest,
)
from src.depth_surge_3d.utils.imaging.image_processing import (
    CENTER_CROP_ALGORITHM_VERSION,
    calculate_center_crop_dimensions,
)
from src.depth_surge_3d.inference.depth.vdpp_contract import (
    build_vdpp_execution_plan,
    vdpp_model_identity,
)
from src.depth_surge_3d.processing.frames.temporal_storage import StabilizedDepthStore


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


def _write_v2_raw_metadata(
    output_dir: Path,
    frame_files: list[Path],
    source_fingerprint: str,
) -> dict:
    settings = _current_settings(
        depth_model_version="v2",
        model_size="base",
        depth_resolution="518",
    )
    _write_raw_metadata(
        output_dir,
        frame_files,
        source_fingerprint,
        model_size="base",
    )
    metadata_path = output_dir / "02_depth_raw" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    semantic = metadata["semantic_fingerprint"]
    semantic.update(
        {
            "model_info": {
                "revision": "immutable",
                "inference_algorithm": "vda-offline-shot-v1",
            },
            "depth_settings": select_depth_model_settings(settings),
            "scene_algorithm_version": SCENE_ALGORITHM_VERSION,
            "execution_plan": {
                "requested_input_size": 518,
                "effective_input_size": 384,
                "precision": "fp16",
                "fallback_policy": "v2-uniform-halving-v1",
            },
        }
    )
    metadata["fingerprint"] = RawDepthStore._fingerprint(metadata)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return {
        key: semantic[key]
        for key in (
            "backend",
            "model_info",
            "depth_settings",
            "weight_sha256",
            "artifact_identity",
        )
    }


def _mutate_raw_semantic(output_dir: Path, mutate) -> None:
    metadata_path = output_dir / "02_depth_raw" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mutate(metadata["semantic_fingerprint"])
    metadata["fingerprint"] = RawDepthStore._fingerprint(metadata)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


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


def _write_complete_stabilized_stage(
    output_dir: Path,
    frame_files: list[Path],
    canonical: dict,
) -> StabilizedDepthStore:
    shot_plan = [{"shot_id": 0, "start": 0, "end": len(frame_files)}]
    semantic = {
        "frame_names": [path.name for path in frame_files],
        "native_shape": canonical["native_shape"],
        "source_canonical_fingerprint": canonical["fingerprint"],
        "scene_manifest_fingerprint": canonical["scene_manifest_fingerprint"],
        "postprocessor_settings": {"temporal_postprocessor": "vdpp"},
        "model_identity": vdpp_model_identity(),
        "execution_plan": build_vdpp_execution_plan(tuple(canonical["native_shape"])),
        "shot_plan": shot_plan,
    }
    store = StabilizedDepthStore(
        output_dir / "03_disparity_stabilized",
        frame_files=frame_files,
        semantic_identity=semantic,
        runtime_identity={"runtime": "test"},
        execution_provenance={"runtime": "test"},
    )
    store.prepare(store.audit())
    calibration = {
        "mode": "base_fallback",
        "pair_count": 0,
        "midpoint_count": 0,
        "midpoint_fraction": 0.0,
        "flat_frame_count": len(frame_files),
        "source_mean": None,
        "source_variance": None,
        "source_std": None,
        "raw_mean": None,
        "raw_variance": None,
        "raw_std": None,
        "covariance": None,
        "correlation": None,
        "scale": None,
        "shift": None,
        "candidate_mean": None,
        "candidate_std": None,
        "postclip_contrast_ratio": None,
        "postclip_mean_drift": None,
        "preclip_low_fraction": None,
        "preclip_high_fraction": None,
        "fallback_reason": "source_no_range",
    }
    store.commit_shot(
        0,
        (
            (index, np.full((4, 6), index * 1000, dtype=np.uint16))
            for index in range(len(frame_files))
        ),
        calibration=calibration,
    )
    store.finalize()
    return store


def _publish_building_stabilized_metadata(store: StabilizedDepthStore) -> dict:
    metadata = json.loads(store.metadata_path.read_text(encoding="utf-8"))
    metadata["status"] = "building"
    metadata["payload_fingerprint"] = None
    metadata["artifact_fingerprint"] = None
    metadata["state_fingerprint"] = canonical_json_hash(
        {
            "status": metadata["status"],
            "completed_shots": metadata["completed_shots"],
        }
    )
    metadata["metadata_fingerprint"] = canonical_json_hash(
        {key: value for key, value in metadata.items() if key != "metadata_fingerprint"}
    )
    store.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


def _rehash_first_stabilized_manifest(store: StabilizedDepthStore) -> None:
    manifest_path = store.manifest_dir / "shot_000000.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_fingerprint"] = canonical_json_hash(
        {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    metadata = json.loads(store.metadata_path.read_text(encoding="utf-8"))
    metadata["completed_shots"][0]["manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    metadata["state_fingerprint"] = canonical_json_hash(
        {
            "status": metadata["status"],
            "completed_shots": metadata["completed_shots"],
        }
    )
    metadata["metadata_fingerprint"] = canonical_json_hash(
        {key: value for key, value in metadata.items() if key != "metadata_fingerprint"}
    )
    store.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


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


def _write_metric_geometry_pipeline(
    output_dir: Path,
    frame_files: list[Path],
    source_fingerprint: str,
    manifest: dict,
) -> dict:
    raw_metadata = RawDepthStore.read_metadata(output_dir / "02_depth_raw")
    assert raw_metadata is not None
    candidate_scene_fingerprint = canonical_json_hash(
        {
            "schema_version": manifest["schema_version"],
            "algorithm_version": manifest["algorithm_version"],
            "frame_names": manifest["frame_names"],
            "scene_ids": manifest["scene_ids"],
        }
    )
    store = MetricGeometryStore.open_or_create(
        output_dir / "03_metric_geometry",
        frame_names=[path.name for path in frame_files],
        native_shape=(4, 6),
        source_raw_fingerprint=raw_metadata["fingerprint"],
        source_frame_fingerprint=source_fingerprint,
        candidate_scene_fingerprint=candidate_scene_fingerprint,
        preflight_required_bytes=0,
    )
    for frame in frame_files:
        store.write_frame(
            frame.name,
            MetricGeometryFrame(
                inverse_depth=np.ones((4, 6), dtype=np.float32),
                valid=np.ones((4, 6), dtype=np.bool_),
                focal_x_normalized=np.float32(0.8),
            ),
        )
    return store.finalize(
        ClipConvergence(
            distance_m=np.float32(2.0),
            selected_frame_indexes=(0,),
            sample_count=24,
        )
    )


def _write_metric_stereo_pipeline(
    output_dir: Path,
    frame_files: list[Path],
    metric_metadata: dict,
    settings: dict,
) -> None:
    left_dir = output_dir / "04_left_frames"
    right_dir = output_dir / "04_right_frames"
    left_dir.mkdir(exist_ok=True)
    right_dir.mkdir(exist_ok=True)
    retained_width, _ = calculate_center_crop_dimensions(6, 4, settings["crop_factor"])
    metadata = {
        "schema_version": STEREO_STAGE_SCHEMA_VERSION,
        "algorithm_version": STEREO_STAGE_ALGORITHM_VERSION,
        "geometry_mode": "metric_camera",
        "frame_names": [path.name for path in frame_files],
        "render_shape": [4, 6],
        "occlusion_fill": settings["occlusion_fill"],
        "renderer_device_type": "cpu",
        "encoding": "uint8_png",
        "source_metric_fingerprint": metric_metadata["fingerprint"],
        "projection_algorithm_version": METRIC_PROJECTION_ALGORITHM_VERSION,
        "source_width": 6,
        "retained_crop_width": retained_width,
        "center_crop_algorithm_version": CENTER_CROP_ALGORITHM_VERSION,
        "sample_aspect_ratio": "1:1",
        "virtual_baseline_mm": settings["virtual_baseline_mm"],
        "requested_convergence_distance": settings["metric_convergence_distance"],
        "effective_convergence_distance_m": 2.0,
        "max_disparity_percent": settings["max_disparity_percent"],
    }
    metadata["fingerprint"] = canonical_json_hash(metadata)
    (left_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for frame in frame_files:
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        assert cv2.imwrite(str(left_dir / f"{frame.stem}.png"), image)
        assert cv2.imwrite(str(right_dir / f"{frame.stem}.png"), image)


def _metric_job(output_dir: Path, *, include_relative: bool = True) -> tuple[list[Path], dict]:
    frame_files, source_fingerprint = _write_frames(output_dir)
    settings = _current_settings(
        depth_model_version="moge2",
        stereo_geometry_mode="metric_camera",
        apply_distortion=False,
    )
    settings_file = _write_settings(output_dir, settings, current_schema=True)
    settings_data = json.loads(settings_file.read_text(encoding="utf-8"))
    settings_data["video_properties"].update(
        {
            "sample_aspect_ratio_numerator": 1,
            "sample_aspect_ratio_denominator": 1,
            "sample_aspect_ratio": "1:1",
        }
    )
    settings_file.write_text(json.dumps(settings_data), encoding="utf-8")
    manifest, _, _ = _write_current_depth_pipeline(
        output_dir,
        frame_files,
        source_fingerprint,
        settings=settings,
        camera_model="pinhole_fx",
    )
    metric_metadata = _write_metric_geometry_pipeline(
        output_dir, frame_files, source_fingerprint, manifest
    )
    if not include_relative:
        disparity_dir = output_dir / "03_disparity_maps"
        for path in disparity_dir.iterdir():
            path.unlink()
    _write_metric_stereo_pipeline(output_dir, frame_files, metric_metadata, settings)
    return frame_files, settings


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


def _write_generated_stage(output_dir: Path, names: tuple[str, ...], frame_name: str) -> None:
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    metadata = {"schema_version": 1, "stage": names[0]}
    metadata["fingerprint"] = canonical_json_hash(metadata)
    for index, name in enumerate(names):
        directory = output_dir / name
        directory.mkdir(exist_ok=True)
        assert cv2.imwrite(str(directory / frame_name), image)
        if index == 0:
            (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _directory_bytes(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


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
    for name in (
        "04_left_frames",
        "04_right_frames",
        "08_left_final",
        "08_right_final",
    ):
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
    assert set(report.removed_settings) == {
        "baseline",
        "focal_length",
        "hole_fill_quality",
    }


def test_resume_rejects_future_schema_before_any_mutation(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    _write_frames(tmp_path)
    settings_file = _write_settings(tmp_path, _current_settings(), current_schema=True)
    payload = json.loads(settings_file.read_text(encoding="utf-8"))
    payload["metadata"]["settings_schema_version"] = PROCESSING_SETTINGS_SCHEMA_VERSION + 1
    payload["processing_settings"]["future_knob"] = True
    settings_file.write_text(json.dumps(payload), encoding="utf-8")
    original_bytes = settings_file.read_bytes()

    with pytest.raises(UnsupportedSettingsSchemaError, match="newer settings schema"):
        build_resume_report(tmp_path, _current_settings())

    assert settings_file.read_bytes() == original_bytes
    assert not (tmp_path / "settings.legacy.json").exists()


def test_resume_rejects_unknown_field_in_current_schema(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    _write_frames(tmp_path)
    settings_file = _write_settings(tmp_path, _current_settings(), current_schema=True)
    payload = json.loads(settings_file.read_text(encoding="utf-8"))
    payload["processing_settings"]["future_knob"] = True
    settings_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown setting.*future_knob"):
        build_resume_report(tmp_path, _current_settings())


def test_resume_migrates_v2_missing_temporal_postprocessor_to_off(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    _write_frames(tmp_path)
    settings = _current_settings()
    settings.pop("temporal_postprocessor")
    settings_file = _write_settings(tmp_path, settings)
    payload = json.loads(settings_file.read_text(encoding="utf-8"))
    payload["metadata"]["settings_schema_version"] = 2
    settings_file.write_text(json.dumps(payload), encoding="utf-8")

    report = build_resume_report(tmp_path, _current_settings())

    assert report.migrated_settings["temporal_postprocessor"] == "off"
    assert report.settings_backup_required is True


def test_vdpp_resume_selects_complete_content_addressed_stabilized_stage(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    frame_files, fingerprint = _write_frames(tmp_path)
    settings = _current_settings(temporal_postprocessor="vdpp")
    _write_settings(tmp_path, settings, current_schema=True)
    _manifest, _bounds, canonical = _write_current_depth_pipeline(
        tmp_path,
        frame_files,
        fingerprint,
    )
    _write_complete_stabilized_stage(tmp_path, frame_files, canonical)

    report = build_resume_report(tmp_path, settings)

    assert report.stage("disparity_maps").disposition == "preserve"
    assert report.stage("disparity_stabilized").disposition == "preserve"


@pytest.mark.parametrize("changed_identity", ["model", "execution"])
def test_vdpp_resume_rejects_internally_valid_but_obsolete_semantic_identity(
    tmp_path,
    changed_identity,
):
    from src.depth_surge_3d.io.resume import build_resume_report

    frame_files, fingerprint = _write_frames(tmp_path)
    settings = _current_settings(temporal_postprocessor="vdpp")
    _write_settings(tmp_path, settings, current_schema=True)
    _manifest, _bounds, canonical = _write_current_depth_pipeline(
        tmp_path,
        frame_files,
        fingerprint,
    )
    store = _write_complete_stabilized_stage(tmp_path, frame_files, canonical)
    metadata = json.loads(store.metadata_path.read_text(encoding="utf-8"))
    semantic = metadata["semantic_identity"]
    if changed_identity == "model":
        semantic["model_identity"]["checkpoint_sha256"] = "0" * 64
    else:
        semantic["execution_plan"]["downsize"] = False
    metadata["semantic_fingerprint"] = canonical_json_hash(semantic)
    metadata["artifact_fingerprint"] = canonical_json_hash(
        {
            "semantic_fingerprint": metadata["semantic_fingerprint"],
            "payload_fingerprint": metadata["payload_fingerprint"],
        }
    )
    metadata.pop("metadata_fingerprint")
    metadata["metadata_fingerprint"] = canonical_json_hash(metadata)
    store.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = build_resume_report(tmp_path, settings)

    stage = report.stage("disparity_stabilized")
    assert stage.disposition == "invalidate"
    assert "identity" in stage.reason


def test_vdpp_resume_rejects_tampered_building_metadata(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    frame_files, fingerprint = _write_frames(tmp_path)
    settings = _current_settings(temporal_postprocessor="vdpp")
    _write_settings(tmp_path, settings, current_schema=True)
    _manifest, _bounds, canonical = _write_current_depth_pipeline(
        tmp_path,
        frame_files,
        fingerprint,
    )
    store = _write_complete_stabilized_stage(tmp_path, frame_files, canonical)
    metadata = json.loads(store.metadata_path.read_text(encoding="utf-8"))
    metadata["status"] = "building"
    store.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = build_resume_report(tmp_path, settings)

    stage = report.stage("disparity_stabilized")
    assert stage.disposition == "invalidate"
    assert "metadata fingerprint" in stage.reason


def test_vdpp_building_resume_reports_invalid_diagnostics_as_regeneration_work(
    tmp_path,
):
    from src.depth_surge_3d.io.resume import build_resume_report

    frame_files, fingerprint = _write_frames(tmp_path)
    settings = _current_settings(temporal_postprocessor="vdpp")
    _write_settings(tmp_path, settings, current_schema=True)
    _manifest, _bounds, canonical = _write_current_depth_pipeline(
        tmp_path,
        frame_files,
        fingerprint,
    )
    store = _write_complete_stabilized_stage(tmp_path, frame_files, canonical)
    _publish_building_stabilized_metadata(store)
    manifest_path = store.manifest_dir / "shot_000000.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["calibration"]["midpoint_fraction"] = -0.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _rehash_first_stabilized_manifest(store)

    report = build_resume_report(tmp_path, settings)

    stage = report.stage("disparity_stabilized")
    assert stage.disposition == "resume"
    assert "record-valid=0" in stage.reason
    assert "invalid-to-regenerate=1" in stage.reason
    assert "pending=0" in stage.reason


def test_vdpp_building_resume_rejects_structural_completed_record_corruption(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    frame_files, fingerprint = _write_frames(tmp_path)
    settings = _current_settings(temporal_postprocessor="vdpp")
    _write_settings(tmp_path, settings, current_schema=True)
    _manifest, _bounds, canonical = _write_current_depth_pipeline(
        tmp_path,
        frame_files,
        fingerprint,
    )
    store = _write_complete_stabilized_stage(tmp_path, frame_files, canonical)
    metadata = _publish_building_stabilized_metadata(store)
    metadata["completed_shots"].append(dict(metadata["completed_shots"][0]))
    metadata["state_fingerprint"] = canonical_json_hash(
        {"status": "building", "completed_shots": metadata["completed_shots"]}
    )
    metadata["metadata_fingerprint"] = canonical_json_hash(
        {key: value for key, value in metadata.items() if key != "metadata_fingerprint"}
    )
    store.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = build_resume_report(tmp_path, settings)

    stage = report.stage("disparity_stabilized")
    assert stage.disposition == "invalidate"
    assert "sorted and unique" in stage.reason


def test_off_resume_leaves_stabilized_stage_dormant_and_out_of_report(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    frame_files, fingerprint = _write_frames(tmp_path)
    settings = _current_settings(temporal_postprocessor="off")
    _write_settings(tmp_path, settings, current_schema=True)
    _manifest, _bounds, canonical = _write_current_depth_pipeline(
        tmp_path,
        frame_files,
        fingerprint,
    )
    _write_complete_stabilized_stage(tmp_path, frame_files, canonical)

    report = build_resume_report(tmp_path, settings)

    assert "disparity_stabilized" not in [stage.name for stage in report.stages]
    assert (tmp_path / "03_disparity_stabilized").is_dir()


def test_corrupt_stabilized_payload_invalidates_only_derived_render_source(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    frame_files, fingerprint = _write_frames(tmp_path)
    settings = _current_settings(temporal_postprocessor="vdpp")
    _write_settings(tmp_path, settings, current_schema=True)
    _manifest, _bounds, canonical = _write_current_depth_pipeline(
        tmp_path,
        frame_files,
        fingerprint,
    )
    store = _write_complete_stabilized_stage(tmp_path, frame_files, canonical)
    store.depth_files[0].write_bytes(b"corrupt")

    report = build_resume_report(tmp_path, settings)

    assert report.stage("depth_raw").disposition == "preserve"
    assert report.stage("disparity_maps").disposition == "preserve"
    assert report.stage("disparity_stabilized").disposition == "invalidate"


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


@pytest.mark.parametrize(
    "execution_plan",
    [
        None,
        {
            "requested_input_size": 518,
            "effective_input_size": 500,
            "precision": "fp16",
            "fallback_policy": "v2-uniform-halving-v1",
        },
        {
            "requested_input_size": 518,
            "effective_input_size": 384,
            "fallback_policy": "v2-uniform-halving-v1",
        },
        {
            "requested_input_size": 518,
            "effective_input_size": 384,
            "precision": [],
            "fallback_policy": "v2-uniform-halving-v1",
        },
        {
            "requested_input_size": 720,
            "effective_input_size": 384,
            "precision": "fp16",
            "fallback_policy": "v2-uniform-halving-v1",
        },
        {
            "requested_input_size": 518,
            "effective_input_size": 384,
            "precision": "fp32",
            "fallback_policy": "v2-uniform-halving-v1",
        },
    ],
)
def test_v2_resume_invalidates_missing_or_incompatible_execution_plan(
    tmp_path,
    execution_plan,
):
    from src.depth_surge_3d.io.resume import build_resume_report

    settings = _current_settings(
        depth_model_version="v2",
        model_size="base",
        depth_resolution="518",
    )
    frame_files, fingerprint = _write_frames(tmp_path)
    _write_settings(tmp_path, settings, current_schema=True)
    _write_candidate_manifest(tmp_path, frame_files, fingerprint)
    model_fingerprint = _write_v2_raw_metadata(tmp_path, frame_files, fingerprint)

    def replace_plan(semantic):
        if execution_plan is None:
            semantic.pop("execution_plan")
        else:
            semantic["execution_plan"] = execution_plan

    _mutate_raw_semantic(tmp_path, replace_plan)

    report = build_resume_report(
        tmp_path,
        settings,
        model_fingerprint=model_fingerprint,
    )

    assert report.stage("depth_raw").disposition == "invalidate"
    assert "execution plan" in report.stage("depth_raw").reason


@pytest.mark.parametrize("scene_version", [None, "scene-analysis-v0"])
def test_v2_resume_invalidates_missing_or_incompatible_scene_algorithm(
    tmp_path,
    scene_version,
):
    from src.depth_surge_3d.io.resume import build_resume_report

    settings = _current_settings(
        depth_model_version="v2",
        model_size="base",
        depth_resolution="518",
    )
    frame_files, fingerprint = _write_frames(tmp_path)
    _write_settings(tmp_path, settings, current_schema=True)
    _write_candidate_manifest(tmp_path, frame_files, fingerprint)
    model_fingerprint = _write_v2_raw_metadata(tmp_path, frame_files, fingerprint)

    def replace_scene_version(semantic):
        if scene_version is None:
            semantic.pop("scene_algorithm_version")
        else:
            semantic["scene_algorithm_version"] = scene_version

    _mutate_raw_semantic(tmp_path, replace_scene_version)

    report = build_resume_report(
        tmp_path,
        settings,
        model_fingerprint=model_fingerprint,
    )

    assert report.stage("depth_raw").disposition == "invalidate"
    assert "scene algorithm" in report.stage("depth_raw").reason


def test_v2_resume_accepts_valid_execution_contract(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    settings = _current_settings(
        depth_model_version="v2",
        model_size="base",
        depth_resolution="518",
    )
    frame_files, fingerprint = _write_frames(tmp_path)
    _write_settings(tmp_path, settings, current_schema=True)
    _write_candidate_manifest(tmp_path, frame_files, fingerprint)
    model_fingerprint = _write_v2_raw_metadata(tmp_path, frame_files, fingerprint)

    report = build_resume_report(
        tmp_path,
        settings,
        model_fingerprint=model_fingerprint,
    )

    assert report.stage("depth_raw").disposition == "resume"


def test_loaded_model_depth_settings_override_backend_name_heuristic(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    v2_settings = _current_settings(
        depth_model_version="v2",
        model_size="base",
        depth_resolution="518",
    )
    frame_files, fingerprint = _write_frames(tmp_path)
    _write_settings(tmp_path, v2_settings, current_schema=True)
    _write_candidate_manifest(tmp_path, frame_files, fingerprint)
    model_fingerprint = _write_v2_raw_metadata(tmp_path, frame_files, fingerprint)

    report = build_resume_report(
        tmp_path,
        _current_settings(
            depth_model_version="v3",
            model_size="large",
            depth_resolution="518",
        ),
        model_fingerprint=model_fingerprint,
    )

    assert report.stage("depth_raw").disposition == "resume"


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


def test_resume_reports_both_valid_stage3_directories_without_deleting_inactive_one(
    tmp_path,
):
    from src.depth_surge_3d.io.resume import build_resume_report

    _, settings = _metric_job(tmp_path)
    relative_before = _directory_bytes(tmp_path / "03_disparity_maps")
    metric_before = _directory_bytes(tmp_path / "03_metric_geometry")

    report = build_resume_report(tmp_path, settings)

    assert report.stage("disparity_maps").disposition == "preserve"
    assert report.stage("metric_geometry").disposition == "preserve"
    assert report.stage("stereo").disposition == "preserve"
    assert _directory_bytes(tmp_path / "03_disparity_maps") == relative_before
    assert _directory_bytes(tmp_path / "03_metric_geometry") == metric_before


def test_mode_switch_preserves_both_valid_stage3_formats_and_invalidates_only_stereo(
    tmp_path,
):
    from src.depth_surge_3d.io.resume import build_resume_report

    _, settings = _metric_job(tmp_path)
    relative_before = _directory_bytes(tmp_path / "03_disparity_maps")
    metric_before = _directory_bytes(tmp_path / "03_metric_geometry")

    report = build_resume_report(tmp_path, {**settings, "stereo_geometry_mode": "relative"})

    assert report.stage("depth_raw").disposition == "preserve"
    assert report.stage("disparity_maps").disposition == "preserve"
    assert report.stage("metric_geometry").disposition == "preserve"
    assert report.stage("stereo").disposition == "invalidate"
    assert _directory_bytes(tmp_path / "03_disparity_maps") == relative_before
    assert _directory_bytes(tmp_path / "03_metric_geometry") == metric_before


@pytest.mark.parametrize(
    ("setting_name", "new_value"),
    [
        ("virtual_baseline_mm", 70.0),
        ("metric_convergence_distance", 3.0),
        ("max_disparity_percent", 1.5),
    ],
)
def test_metric_stereo_settings_invalidate_only_stereo(tmp_path, setting_name, new_value):
    from src.depth_surge_3d.io.resume import build_resume_report

    _, settings = _metric_job(tmp_path)

    report = build_resume_report(tmp_path, {**settings, setting_name: new_value})

    assert report.stage("depth_raw").disposition == "preserve"
    assert report.stage("metric_geometry").disposition == "preserve"
    assert report.stage("stereo").disposition == "invalidate"


def test_depth_identity_change_invalidates_metric_geometry_and_stereo(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    _, settings = _metric_job(tmp_path)

    report = build_resume_report(tmp_path, {**settings, "depth_resolution": 512})

    assert report.stage("depth_raw").disposition == "invalidate"
    assert "depth_resolution" in report.stage("depth_raw").reason
    assert report.stage("metric_geometry").disposition == "invalidate"
    assert report.stage("stereo").disposition == "invalidate"


def test_completed_metric_geometry_reuses_compatible_ready_raw_identity_without_payloads(
    tmp_path,
):
    from src.depth_surge_3d.io.resume import build_resume_report

    _, settings = _metric_job(tmp_path)
    raw_dir = tmp_path / "02_depth_raw"
    for path in raw_dir.glob("*.npz"):
        path.unlink()
    metadata_path = raw_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["completed_count"] = 0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = build_resume_report(tmp_path, settings)

    assert report.stage("depth_raw").disposition == "resume"
    assert report.stage("metric_geometry").disposition == "preserve"
    assert report.stage("stereo").disposition == "preserve"


def test_corrupt_raw_payload_does_not_invalidate_completed_metric_stage(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    _, settings = _metric_job(tmp_path)
    (tmp_path / "02_depth_raw" / "frame_000001.npz").write_bytes(b"corrupt raw payload")

    report = build_resume_report(tmp_path, settings)

    assert report.stage("depth_raw").disposition == "invalidate"
    assert "payload validation failed" in report.stage("depth_raw").reason
    assert report.stage("metric_geometry").disposition == "preserve"
    assert report.stage("stereo").disposition == "preserve"


def test_crop_change_invalidates_metric_stereo_but_not_relative_stereo(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    metric_dir = tmp_path / "metric"
    metric_dir.mkdir()
    frame_files, metric_settings = _metric_job(metric_dir)
    _write_generated_stage(
        metric_dir,
        ("06_left_cropped", "06_right_cropped"),
        frame_files[0].name,
    )
    metric_report = build_resume_report(metric_dir, {**metric_settings, "crop_factor": 0.8})

    relative_dir = tmp_path / "relative"
    relative_dir.mkdir()
    frame_files, fingerprint = _write_frames(relative_dir)
    relative_settings = _current_settings(apply_distortion=False)
    _write_settings(relative_dir, relative_settings, current_schema=True)
    _, _, canonical = _write_current_depth_pipeline(relative_dir, frame_files, fingerprint)
    _write_current_stereo_pipeline(relative_dir, frame_files, canonical)
    _write_generated_stage(
        relative_dir,
        ("06_left_cropped", "06_right_cropped"),
        frame_files[0].name,
    )
    relative_report = build_resume_report(relative_dir, {**relative_settings, "crop_factor": 0.8})

    assert metric_report.stage("metric_geometry").disposition == "preserve"
    assert metric_report.stage("stereo").disposition == "invalidate"
    assert relative_report.stage("stereo").disposition == "preserve"
    assert relative_report.stage("crop").disposition == "invalidate"


def test_final_output_width_change_reuses_metric_stereo(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    frame_files, settings = _metric_job(tmp_path)
    _write_generated_stage(
        tmp_path,
        ("06_left_cropped", "06_right_cropped"),
        frame_files[0].name,
    )
    _write_generated_stage(tmp_path, ("99_vr_frames",), frame_files[0].name)

    report = build_resume_report(
        tmp_path,
        {**settings, "per_eye_width": 2560, "vr_output_width": 5120},
    )

    assert report.stage("metric_geometry").disposition == "preserve"
    assert report.stage("stereo").disposition == "preserve"
    assert report.stage("crop").disposition == "invalidate"
    assert report.stage("vr_frames").disposition == "invalidate"


def test_no_raw_missing_selected_mode_reports_reinference_without_deletion(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    _, settings = _metric_job(tmp_path, include_relative=False)
    raw_dir = tmp_path / "02_depth_raw"
    for path in raw_dir.glob("*.npz"):
        path.unlink()
    metadata_path = raw_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["completed_count"] = 0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    frames_before = _directory_bytes(tmp_path / "00_original_frames")
    metric_before = _directory_bytes(tmp_path / "03_metric_geometry")

    report = build_resume_report(tmp_path, {**settings, "stereo_geometry_mode": "relative"})

    assert (
        "MoGe inference is required to build the selected geometry stage"
        in report.stage("disparity_maps").reason
    )
    assert report.stage("metric_geometry").disposition == "preserve"
    assert _directory_bytes(tmp_path / "00_original_frames") == frames_before
    assert _directory_bytes(tmp_path / "03_metric_geometry") == metric_before


def test_resume_report_keeps_raw_and_metric_temporary_files(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    _, settings = _metric_job(tmp_path)
    raw_temporary = tmp_path / "02_depth_raw" / "sentinel.npz.tmp"
    metric_temporary = tmp_path / "03_metric_geometry" / "sentinel.tmp"
    raw_temporary.write_bytes(b"raw temporary")
    metric_temporary.write_bytes(b"metric temporary")
    before = {
        "raw": _directory_bytes(tmp_path / "02_depth_raw"),
        "metric": _directory_bytes(tmp_path / "03_metric_geometry"),
    }

    report = build_resume_report(tmp_path, settings)

    assert report.stage("depth_raw").disposition == "preserve"
    assert report.stage("metric_geometry").disposition == "preserve"
    assert _directory_bytes(tmp_path / "02_depth_raw") == before["raw"]
    assert _directory_bytes(tmp_path / "03_metric_geometry") == before["metric"]


def test_metric_stereo_manifest_requires_explicit_occlusion_fill(tmp_path):
    from src.depth_surge_3d.io.resume import build_resume_report

    _, settings = _metric_job(tmp_path)
    metadata_path = tmp_path / "04_left_frames" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("occlusion_fill")
    metadata["fingerprint"] = canonical_json_hash(
        {key: value for key, value in metadata.items() if key != "fingerprint"}
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = build_resume_report(tmp_path, settings)

    assert report.stage("metric_geometry").disposition == "preserve"
    assert report.stage("stereo").disposition == "invalidate"


def test_legacy_metric_resume_reprobes_sar_before_report_construction(tmp_path, monkeypatch):
    import src.depth_surge_3d.io.resume as resume

    frame_files, fingerprint = _write_frames(tmp_path)
    settings = _current_settings(depth_model_version="moge2", stereo_geometry_mode="relative")
    settings_file = _write_settings(tmp_path, settings, current_schema=True)
    _write_current_depth_pipeline(tmp_path, frame_files, fingerprint)
    source = tmp_path / "source.mp4"
    calls = []

    def probe(path):
        calls.append(path)
        return {
            "sample_aspect_ratio_numerator": 1,
            "sample_aspect_ratio_denominator": 1,
            "sample_aspect_ratio": "1:1",
        }

    monkeypatch.setattr(resume, "get_video_properties", probe)

    report = resume.build_resume_report(
        tmp_path,
        {**settings, "stereo_geometry_mode": "metric_camera"},
        source_video=source,
        settings_file=settings_file,
    )

    assert calls == [str(source)]
    assert report.stage("frames").disposition == "preserve"


def test_legacy_metric_resume_without_source_fails_without_invalidating_relative_data(
    tmp_path,
):
    from src.depth_surge_3d.io.resume import build_resume_report

    frame_files, fingerprint = _write_frames(tmp_path)
    settings = _current_settings(depth_model_version="moge2")
    _write_settings(tmp_path, settings, current_schema=True)
    _write_current_depth_pipeline(
        tmp_path,
        frame_files,
        fingerprint,
        settings=settings,
        camera_model="pinhole_fx",
    )
    (tmp_path / "source.mp4").unlink()
    relative_before = _directory_bytes(tmp_path / "03_disparity_maps")

    with pytest.raises(ValueError, match="re-probe.*sample aspect ratio"):
        build_resume_report(
            tmp_path,
            {**settings, "stereo_geometry_mode": "metric_camera"},
            source_video=tmp_path / "missing.mp4",
        )

    assert _directory_bytes(tmp_path / "03_disparity_maps") == relative_before
    relative = build_resume_report(
        tmp_path,
        {**settings, "stereo_geometry_mode": "relative"},
        source_video=None,
    )
    assert relative.stage("disparity_maps").disposition == "preserve"
