"""Shared base/stabilized render-disparity validation tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.core.depth_contract import canonical_json_hash
from src.depth_surge_3d.core.render_disparity import (
    STABILIZED_DEPTH_ALGORITHM_VERSION,
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
) -> tuple[list[Path], list[Path], dict]:
    directory.mkdir()
    depth_path = directory / frame_name
    _write_png(depth_path)
    file_digest = hashlib.sha256(depth_path.read_bytes()).hexdigest()
    file_records = [{"name": frame_name, "sha256": file_digest}]
    shot_payload = canonical_json_hash(file_records)
    shot_manifest = {
        "schema_version": 1,
        "shot_id": 0,
        "start": 0,
        "end": 1,
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

    with pytest.raises(ValueError, match="SHA-256"):
        validate_render_disparity_input(depth_files, frame_files)


def test_rejects_stabilized_metadata_self_hash_change(tmp_path: Path) -> None:
    depth_files, frame_files, _ = _write_stabilized(tmp_path / "stable")
    metadata_path = depth_files[0].parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["execution_provenance"]["device_name"] = "tampered"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata fingerprint"):
        validate_render_disparity_input(depth_files, frame_files)
