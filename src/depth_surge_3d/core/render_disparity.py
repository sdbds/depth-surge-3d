"""Strict validation for every producer accepted by stereo rendering."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .depth_contract import (
    CANONICAL_DEPTH_ALGORITHM_VERSION,
    CANONICAL_DEPTH_SCHEMA_VERSION,
    CANONICAL_METADATA_REQUIRED_FIELDS,
    canonical_json_hash,
)
from ..utils.imaging.png_header import png_header_matches


STABILIZED_DEPTH_SCHEMA_VERSION = 1
STABILIZED_DEPTH_ALGORITHM_VERSION = "vdpp-canonical-shot-v1"


@dataclass(frozen=True)
class RenderDisparityArtifact:
    """A completely validated file-backed disparity source."""

    producer: Literal["base", "stabilized"]
    files: tuple[Path, ...]
    metadata: dict[str, Any]
    fingerprint: str
    native_shape: tuple[int, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"Could not hash render-disparity payload: {path}") from exc
    return digest.hexdigest()


def _read_json(path: Path, description: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        decoded = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is invalid: {path}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{description} is invalid: {path}")
    return decoded, payload


def _native_shape(value: object) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in value)
    ):
        raise ValueError("Render-disparity metadata has an invalid native shape")
    return int(value[0]), int(value[1])


def _validate_common(
    metadata: dict[str, Any],
    *,
    depth_files: list[Path],
    frame_files: list[Path],
    frame_names: object,
    native_shape_value: object,
) -> tuple[tuple[int, int], list[str]]:
    expected_names = [path.name for path in frame_files]
    valid_common = (
        metadata.get("representation") == "relative_disparity"
        and metadata.get("near_value") == 1.0
        and metadata.get("far_value") == 0.0
        and metadata.get("encoding") == "uint16_png"
        and metadata.get("encoding_scale") == 65535.0
        and metadata.get("num_frames") == len(depth_files)
        and frame_names == expected_names
    )
    if not valid_common:
        raise ValueError("Render-disparity metadata does not match this render input")

    shape = _native_shape(native_shape_value)
    root = depth_files[0].parent
    expected_paths = [root / f"{Path(name).stem}.png" for name in expected_names]
    if [path.resolve() for path in depth_files] != [path.resolve() for path in expected_paths]:
        raise ValueError("Render-disparity files do not match the metadata path manifest")
    for path in depth_files:
        if not png_header_matches(path, shape=shape, bit_depth=16):
            raise ValueError(f"Render-disparity payload does not match metadata: {path}")
    return shape, expected_names


def _validate_base(
    metadata: dict[str, Any],
    depth_files: list[Path],
    frame_files: list[Path],
) -> RenderDisparityArtifact:
    fingerprint = metadata.get("fingerprint")
    unhashed = {key: value for key, value in metadata.items() if key != "fingerprint"}
    if not (
        CANONICAL_METADATA_REQUIRED_FIELDS.issubset(metadata)
        and metadata.get("schema_version") == CANONICAL_DEPTH_SCHEMA_VERSION
        and isinstance(fingerprint, str)
        and fingerprint == canonical_json_hash(unhashed)
    ):
        raise ValueError("Base canonical disparity metadata does not match this render input")
    shape, _ = _validate_common(
        metadata,
        depth_files=depth_files,
        frame_files=frame_files,
        frame_names=metadata.get("frame_names"),
        native_shape_value=metadata.get("native_shape"),
    )
    return RenderDisparityArtifact(
        producer="base",
        files=tuple(depth_files),
        metadata=metadata,
        fingerprint=fingerprint,
        native_shape=shape,
    )


def _validate_shot_plan(shot_plan: object, num_frames: int) -> list[dict[str, int]]:
    if not isinstance(shot_plan, list) or not shot_plan:
        raise ValueError("Stabilized disparity shot plan is invalid")
    normalized: list[dict[str, int]] = []
    next_start = 0
    for expected_id, shot in enumerate(shot_plan):
        if not isinstance(shot, dict):
            raise ValueError("Stabilized disparity shot plan is invalid")
        shot_id = shot.get("shot_id")
        start = shot.get("start")
        end = shot.get("end")
        if (
            isinstance(shot_id, bool)
            or isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(shot_id, int)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or shot_id != expected_id
            or start != next_start
            or end <= start
            or end > num_frames
        ):
            raise ValueError("Stabilized disparity shot plan is invalid")
        normalized.append({"shot_id": shot_id, "start": start, "end": end})
        next_start = end
    if next_start != num_frames:
        raise ValueError("Stabilized disparity shot plan does not cover every frame")
    return normalized


def _validate_stabilized_shots(
    root: Path,
    metadata: dict[str, Any],
    frame_names: list[str],
    shot_plan: list[dict[str, int]],
) -> str:
    completed = metadata.get("completed_shots")
    if not isinstance(completed, list) or len(completed) != len(shot_plan):
        raise ValueError("Stabilized disparity is not complete")

    payload_records: list[dict[str, Any]] = []
    for shot, record in zip(shot_plan, completed):
        if not isinstance(record, dict) or record.get("shot_id") != shot["shot_id"]:
            raise ValueError("Stabilized disparity completed-shot state is invalid")
        relative_manifest = f"shot_manifests/shot_{shot['shot_id']:06d}.json"
        if record.get("manifest") != relative_manifest:
            raise ValueError("Stabilized disparity shot manifest path is invalid")
        manifest_path = root / Path(relative_manifest)
        manifest, manifest_bytes = _read_json(
            manifest_path,
            "Stabilized disparity shot manifest",
        )
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        if record.get("manifest_sha256") != manifest_sha:
            raise ValueError(f"Stabilized disparity manifest SHA-256 mismatch: {manifest_path}")
        manifest_fingerprint = manifest.get("manifest_fingerprint")
        manifest_unhashed = {
            key: value for key, value in manifest.items() if key != "manifest_fingerprint"
        }
        if not isinstance(manifest_fingerprint, str) or manifest_fingerprint != canonical_json_hash(
            manifest_unhashed
        ):
            raise ValueError(
                f"Stabilized disparity shot manifest self-hash mismatch: {manifest_path}"
            )
        if (
            manifest.get("schema_version") != 1
            or manifest.get("shot_id") != shot["shot_id"]
            or manifest.get("start") != shot["start"]
            or manifest.get("end") != shot["end"]
        ):
            raise ValueError(f"Stabilized disparity shot manifest range mismatch: {manifest_path}")

        expected_names = frame_names[shot["start"] : shot["end"]]
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != len(expected_names):
            raise ValueError(f"Stabilized disparity shot file manifest is invalid: {manifest_path}")
        normalized_files: list[dict[str, str]] = []
        for name, file_record in zip(expected_names, files):
            if (
                not isinstance(file_record, dict)
                or file_record.get("name") != name
                or not isinstance(file_record.get("sha256"), str)
            ):
                raise ValueError(
                    f"Stabilized disparity shot file manifest is invalid: {manifest_path}"
                )
            payload_path = root / f"{Path(name).stem}.png"
            digest = _sha256(payload_path)
            if file_record["sha256"] != digest:
                raise ValueError(f"Stabilized disparity payload SHA-256 mismatch: {payload_path}")
            normalized_files.append({"name": name, "sha256": digest})

        shot_payload = canonical_json_hash(normalized_files)
        if (
            manifest.get("shot_payload_sha256") != shot_payload
            or record.get("shot_payload_sha256") != shot_payload
        ):
            raise ValueError(f"Stabilized disparity shot payload SHA-256 mismatch: {manifest_path}")
        payload_records.append({"shot_id": shot["shot_id"], "shot_payload_sha256": shot_payload})

    payload_fingerprint = canonical_json_hash(payload_records)
    if metadata.get("payload_fingerprint") != payload_fingerprint:
        raise ValueError("Stabilized disparity payload fingerprint is invalid")
    return payload_fingerprint


def _validate_stabilized(
    metadata: dict[str, Any],
    root: Path,
    depth_files: list[Path],
    frame_files: list[Path],
) -> RenderDisparityArtifact:
    metadata_fingerprint = metadata.get("metadata_fingerprint")
    metadata_unhashed = {
        key: value for key, value in metadata.items() if key != "metadata_fingerprint"
    }
    if not isinstance(metadata_fingerprint, str) or metadata_fingerprint != canonical_json_hash(
        metadata_unhashed
    ):
        raise ValueError("Stabilized disparity metadata fingerprint is invalid")
    if (
        metadata.get("schema_version") != STABILIZED_DEPTH_SCHEMA_VERSION
        or metadata.get("status") != "complete"
    ):
        raise ValueError("Stabilized disparity metadata is not complete")

    semantic = metadata.get("semantic_identity")
    if not isinstance(semantic, dict):
        raise ValueError("Stabilized disparity semantic identity is invalid")
    semantic_fingerprint = metadata.get("semantic_fingerprint")
    if not isinstance(semantic_fingerprint, str) or semantic_fingerprint != canonical_json_hash(
        semantic
    ):
        raise ValueError("Stabilized disparity semantic fingerprint is invalid")

    shape, frame_names = _validate_common(
        metadata,
        depth_files=depth_files,
        frame_files=frame_files,
        frame_names=semantic.get("frame_names"),
        native_shape_value=semantic.get("native_shape"),
    )
    shot_plan = _validate_shot_plan(semantic.get("shot_plan"), len(depth_files))
    state = {
        "status": metadata.get("status"),
        "completed_shots": metadata.get("completed_shots"),
    }
    if metadata.get("state_fingerprint") != canonical_json_hash(state):
        raise ValueError("Stabilized disparity state fingerprint is invalid")
    payload_fingerprint = _validate_stabilized_shots(
        root,
        metadata,
        frame_names,
        shot_plan,
    )
    expected_artifact = canonical_json_hash(
        {
            "semantic_fingerprint": semantic_fingerprint,
            "payload_fingerprint": payload_fingerprint,
        }
    )
    if metadata.get("artifact_fingerprint") != expected_artifact:
        raise ValueError("Stabilized disparity artifact fingerprint is invalid")
    return RenderDisparityArtifact(
        producer="stabilized",
        files=tuple(depth_files),
        metadata=metadata,
        fingerprint=expected_artifact,
        native_shape=shape,
    )


def validate_render_disparity_input(
    depth_files: list[Path],
    frame_files: list[Path],
) -> RenderDisparityArtifact:
    """Validate one of the two explicitly supported render-disparity producers."""

    if not depth_files:
        raise ValueError("Render-disparity files are required")
    if len(depth_files) != len(frame_files):
        raise ValueError("Render-disparity and frame counts do not match")
    root = depth_files[0].parent
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"Render-disparity metadata is missing: {metadata_path}")
    metadata, _ = _read_json(metadata_path, "Render-disparity metadata")
    algorithm = metadata.get("algorithm_version")
    if algorithm == CANONICAL_DEPTH_ALGORITHM_VERSION:
        return _validate_base(metadata, depth_files, frame_files)
    if algorithm == STABILIZED_DEPTH_ALGORITHM_VERSION:
        return _validate_stabilized(metadata, root, depth_files, frame_files)
    raise ValueError(f"unsupported render-disparity producer: {algorithm!r}")
