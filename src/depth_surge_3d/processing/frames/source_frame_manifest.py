"""Content-bound metadata for extracted source frames."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ...core.file_identity import (
    FILE_IDENTITY_ALGORITHM_VERSION,
    file_sample_fingerprint,
)

SOURCE_FRAME_SCHEMA_VERSION = 2
SOURCE_FRAME_ALGORITHM_VERSION = "png-stat-v1"
SOURCE_FRAME_METADATA_NAME = "metadata.json"


def _metadata_fingerprint(metadata: dict[str, Any]) -> str:
    encoded = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def frame_sequence_fingerprint(frame_files: list[Path]) -> str:
    """Hash frame names and cheap file metadata in sequence order."""

    hasher = hashlib.sha256()
    for path in frame_files:
        stat = path.stat()
        hasher.update(path.name.encode("utf-8"))
        hasher.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode("ascii"))
    return hasher.hexdigest()


def build_source_frame_manifest(
    frame_files: list[Path], source_video: Path | str
) -> dict[str, Any]:
    """Build immutable metadata for one completed frame extraction."""

    if not frame_files:
        raise ValueError("Cannot build a source-frame manifest for an empty sequence")
    source_path = Path(source_video)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source video does not exist: {source_path}")
    metadata: dict[str, Any] = {
        "schema_version": SOURCE_FRAME_SCHEMA_VERSION,
        "algorithm_version": SOURCE_FRAME_ALGORITHM_VERSION,
        "source_video_fingerprint_algorithm": FILE_IDENTITY_ALGORITHM_VERSION,
        "source_video_fingerprint": file_sample_fingerprint(source_path),
        "num_frames": len(frame_files),
        "frame_names": [path.name for path in frame_files],
        "source_frame_fingerprint": frame_sequence_fingerprint(frame_files),
    }
    metadata["fingerprint"] = _metadata_fingerprint(metadata)
    return metadata


def write_source_frame_manifest(frame_files: list[Path], source_video: Path | str) -> Path:
    """Atomically persist a completed source-frame manifest."""

    metadata = build_source_frame_manifest(frame_files, source_video)
    metadata_path = frame_files[0].parent / SOURCE_FRAME_METADATA_NAME
    temporary = metadata_path.with_name(f"{metadata_path.name}.tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(metadata_path)
    return metadata_path


def read_source_frame_manifest(frames_dir: Path) -> dict[str, Any] | None:
    """Read a source-frame manifest, returning None for malformed payloads."""

    try:
        metadata = json.loads((frames_dir / SOURCE_FRAME_METADATA_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return metadata if isinstance(metadata, dict) else None


def source_frame_manifest_mismatch_reason(
    metadata: dict[str, Any] | None,
    frame_files: list[Path],
    source_video_fingerprint: str,
) -> str | None:
    """Return why persisted source frames are not the completed extraction."""

    if metadata is None:
        return "source frame manifest is missing"
    fingerprint = metadata.get("fingerprint")
    unhashed = {key: value for key, value in metadata.items() if key != "fingerprint"}
    checks = (
        metadata.get("schema_version") == SOURCE_FRAME_SCHEMA_VERSION,
        metadata.get("algorithm_version") == SOURCE_FRAME_ALGORITHM_VERSION,
        metadata.get("source_video_fingerprint_algorithm") == FILE_IDENTITY_ALGORITHM_VERSION,
        metadata.get("source_video_fingerprint") == source_video_fingerprint,
        metadata.get("num_frames") == len(frame_files),
        metadata.get("frame_names") == [path.name for path in frame_files],
        isinstance(fingerprint, str) and fingerprint == _metadata_fingerprint(unhashed),
    )
    if not all(checks):
        return "source frame manifest fingerprint mismatch"
    if metadata.get("source_frame_fingerprint") != frame_sequence_fingerprint(frame_files):
        return "source frame fingerprint mismatch"
    return None
