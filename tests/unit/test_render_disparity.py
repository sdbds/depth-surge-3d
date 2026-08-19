"""Shared base/stabilized render-disparity validation tests."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.core.depth_contract import canonical_json_hash
from src.depth_surge_3d.core.render_disparity import (
    STABILIZED_DEPTH_ALGORITHM_VERSION,
    audit_stabilized_shot_records,
    validate_render_disparity_input,
)


def _write_png(path: Path, value: int = 32768) -> None:
    assert cv2.imwrite(str(path), np.full((3, 5), value, dtype=np.uint16))


def _write_base(
    directory: Path, frame_name: str = "frame_000001.png"
) -> tuple[list[Path], list[Path]]:
    directory.mkdir()
    depth_path = directory / frame_name
    _write_png(depth_path)
    metadata = {
        "schema_version": 1,
        "algorithm_version": "scene-percentile-v1",
        "representation": "relative_disparity",
        "near_value": 1.0,
        "far_value": 0.0,
        "encoding": "uint16_png",
        "encoding_scale": 65535.0,
        "num_frames": 1,
        "frame_names": [frame_name],
        "native_shape": [3, 5],
        "source_raw_fingerprint": "raw",
        "source_model_fingerprint": "model",
        "scene_manifest_fingerprint": "scene",
        "depth_bounds_fingerprint": "bounds",
    }
    metadata["fingerprint"] = canonical_json_hash(metadata)
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return [depth_path], [Path(frame_name)]


def _write_stabilized(
    directory: Path,
    frame_name: str = "frame_000001.png",
    *,
    calibration: dict | None = None,
) -> tuple[list[Path], list[Path], dict]:
    directory.mkdir()
    depth_path = directory / frame_name
    _write_png(depth_path)
    file_digest = hashlib.sha256(depth_path.read_bytes()).hexdigest()
    file_records = [{"name": frame_name, "sha256": file_digest}]
    shot_payload = canonical_json_hash(file_records)
    if calibration is None:
        calibration = {
            "mode": "all_midpoint",
            "pair_count": 0,
            "midpoint_count": 15,
            "midpoint_fraction": 1.0,
            "flat_frame_count": 1,
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
            "fallback_reason": None,
        }
    shot_manifest = {
        "schema_version": 2,
        "shot_id": 0,
        "start": 0,
        "end": 1,
        "calibration": calibration,
        "files": file_records,
        "shot_payload_sha256": shot_payload,
    }
    shot_manifest["manifest_fingerprint"] = canonical_json_hash(shot_manifest)
    manifest_dir = directory / "shot_manifests"
    manifest_dir.mkdir()
    manifest_path = manifest_dir / "shot_000000.json"
    manifest_path.write_text(
        json.dumps(shot_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    completed_shots = [
        {
            "shot_id": 0,
            "manifest": "shot_manifests/shot_000000.json",
            "manifest_sha256": manifest_digest,
            "shot_payload_sha256": shot_payload,
        }
    ]
    semantic_identity = {
        "frame_names": [frame_name],
        "native_shape": [3, 5],
        "source_canonical_fingerprint": "base",
        "scene_manifest_fingerprint": "scene",
        "postprocessor_settings": {"temporal_postprocessor": "vdpp"},
        "model_identity": {"name": "vdpp", "vendor_port_version": 1},
        "execution_plan": {"window_size": 32, "overlap": 4, "stride": 28},
        "shot_plan": [{"shot_id": 0, "start": 0, "end": 1}],
    }
    metadata = {
        "schema_version": 1,
        "algorithm_version": STABILIZED_DEPTH_ALGORITHM_VERSION,
        "status": "complete",
        "representation": "relative_disparity",
        "near_value": 1.0,
        "far_value": 0.0,
        "encoding": "uint16_png",
        "encoding_scale": 65535.0,
        "num_frames": 1,
        "semantic_identity": semantic_identity,
        "semantic_fingerprint": canonical_json_hash(semantic_identity),
        "execution_provenance": {"device_name": "test"},
        "partial_resume_runtime_fingerprint": "runtime",
        "completed_shots": completed_shots,
        "state_fingerprint": canonical_json_hash(
            {"status": "complete", "completed_shots": completed_shots}
        ),
        "payload_fingerprint": canonical_json_hash(
            [{"shot_id": 0, "shot_payload_sha256": shot_payload}]
        ),
    }
    metadata["artifact_fingerprint"] = canonical_json_hash(
        {
            "semantic_fingerprint": metadata["semantic_fingerprint"],
            "payload_fingerprint": metadata["payload_fingerprint"],
        }
    )
    metadata["metadata_fingerprint"] = canonical_json_hash(metadata)
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return [depth_path], [Path(frame_name)], metadata


def _ols_calibration() -> dict[str, object]:
    return {
        "mode": "ols",
        "pair_count": 15,
        "midpoint_count": 0,
        "midpoint_fraction": 0.0,
        "flat_frame_count": 0,
        "source_mean": 0.5,
        "source_variance": 0.04,
        "source_std": math.sqrt(0.04),
        "raw_mean": 0.5,
        "raw_variance": 0.04,
        "raw_std": math.sqrt(0.04),
        "covariance": 0.04,
        "correlation": 1.0,
        "scale": 1.0,
        "shift": 0.0,
        "candidate_mean": 0.5,
        "candidate_std": 0.2,
        "postclip_contrast_ratio": 1.0,
        "postclip_mean_drift": 0.0,
        "preclip_low_fraction": 0.0,
        "preclip_high_fraction": 0.0,
        "fallback_reason": None,
    }


def _rehash_stabilized(directory: Path) -> None:
    manifest_path = directory / "shot_manifests/shot_000000.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["manifest_fingerprint"] = canonical_json_hash(
        {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
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
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def test_accepts_historical_base_canonical_artifact(tmp_path: Path) -> None:
    depth_files, frame_files = _write_base(tmp_path / "03_disparity_maps")

    artifact = validate_render_disparity_input(depth_files, frame_files)

    assert artifact.producer == "base"
    assert artifact.fingerprint == artifact.metadata["fingerprint"]
    assert artifact.native_shape == (3, 5)


def test_accepts_content_addressed_stabilized_artifact(tmp_path: Path) -> None:
    depth_files, frame_files, metadata = _write_stabilized(tmp_path / "03_disparity_stabilized")

    artifact = validate_render_disparity_input(depth_files, frame_files)

    assert artifact.producer == "stabilized"
    assert artifact.fingerprint == metadata["artifact_fingerprint"]


def test_rejects_unknown_producer_even_with_common_fields(tmp_path: Path) -> None:
    depth_files, frame_files = _write_base(tmp_path / "depth")
    metadata_path = depth_files[0].parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["algorithm_version"] = "future-magic-v1"
    metadata["fingerprint"] = canonical_json_hash(
        {key: value for key, value in metadata.items() if key != "fingerprint"}
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported render-disparity producer"):
        validate_render_disparity_input(depth_files, frame_files)


def test_rejects_stabilized_payload_byte_change(tmp_path: Path) -> None:
    depth_files, frame_files, _ = _write_stabilized(tmp_path / "stable")
    _write_png(depth_files[0], value=123)

    with pytest.raises(ValueError, match="not complete and valid"):
        validate_render_disparity_input(depth_files, frame_files)


def test_rejects_complete_v1_stabilized_artifact_after_v2_identity_bump(
    tmp_path: Path,
) -> None:
    depth_files, frame_files, _ = _write_stabilized(tmp_path / "stable")
    metadata_path = depth_files[0].parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["algorithm_version"] = "vdpp-canonical-shot-v1"
    metadata["metadata_fingerprint"] = canonical_json_hash(
        {key: value for key, value in metadata.items() if key != "metadata_fingerprint"}
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported render-disparity producer"):
        validate_render_disparity_input(depth_files, frame_files)


def test_rejects_stabilized_metadata_self_hash_change(tmp_path: Path) -> None:
    depth_files, frame_files, _ = _write_stabilized(tmp_path / "stable")
    metadata_path = depth_files[0].parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["execution_provenance"]["device_name"] = "tampered"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata fingerprint"):
        validate_render_disparity_input(depth_files, frame_files)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("correlation", 0.9),
        ("scale", 2.0),
        ("shift", -0.5),
        ("postclip_contrast_ratio", 0.9),
        ("postclip_mean_drift", 0.001),
    ],
)
def test_rehashed_derived_diagnostic_tamper_is_still_invalid(
    tmp_path: Path,
    field: str,
    value: float,
) -> None:
    root = tmp_path / "stable"
    depth_files, frame_files, _ = _write_stabilized(
        root,
        calibration=_ols_calibration(),
    )
    manifest_path = root / "shot_manifests/shot_000000.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["calibration"][field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _rehash_stabilized(root)

    with pytest.raises(ValueError, match="not complete and valid"):
        validate_render_disparity_input(depth_files, frame_files)

    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    semantic = metadata["semantic_identity"]
    audit = audit_stabilized_shot_records(
        root,
        metadata=metadata,
        frame_names=semantic["frame_names"],
        shot_plan=semantic["shot_plan"],
        native_shape=tuple(semantic["native_shape"]),
    )
    assert audit.invalid_shot_ids == (0,)


def test_rehashed_negative_zero_diagnostic_is_invalid(tmp_path: Path) -> None:
    root = tmp_path / "stable"
    depth_files, frame_files, _ = _write_stabilized(
        root,
        calibration=_ols_calibration(),
    )
    manifest_path = root / "shot_manifests/shot_000000.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["calibration"]["shift"] = -0.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _rehash_stabilized(root)

    with pytest.raises(ValueError, match="not complete and valid"):
        validate_render_disparity_input(depth_files, frame_files)


def test_shared_audit_rejects_duplicate_completed_record_as_structural(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stable"
    _, _, metadata = _write_stabilized(root)
    metadata["completed_shots"].append(dict(metadata["completed_shots"][0]))
    semantic = metadata["semantic_identity"]

    with pytest.raises(ValueError, match="sorted and unique"):
        audit_stabilized_shot_records(
            root,
            metadata=metadata,
            frame_names=semantic["frame_names"],
            shot_plan=semantic["shot_plan"],
            native_shape=tuple(semantic["native_shape"]),
        )
