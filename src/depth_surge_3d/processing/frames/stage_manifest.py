"""Cheap completion manifests for generated PNG stages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from ...core.depth_contract import canonical_json_hash
from .frame_stage_parallelism import png_headers_match


FRAME_STAGE_SCHEMA_VERSION = 1
FRAME_STAGE_METADATA_NAME = "metadata.json"


def _source_stat_fingerprint(source_files: Sequence[Path]) -> str:
    hasher = hashlib.sha256()
    for path in source_files:
        stat = path.stat()
        hasher.update(path.parent.name.encode("utf-8"))
        hasher.update(b"/")
        hasher.update(path.name.encode("utf-8"))
        hasher.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode("ascii"))
    return hasher.hexdigest()


def build_stage_identity(
    *,
    stage: str,
    algorithm_version: str,
    frame_names: Sequence[str],
    source_files: Sequence[Path],
    settings: dict[str, Any],
) -> dict[str, Any]:
    if not frame_names or not source_files:
        raise ValueError("A generated frame stage requires non-empty source files")
    return {
        "schema_version": FRAME_STAGE_SCHEMA_VERSION,
        "stage": stage,
        "algorithm_version": algorithm_version,
        "frame_names": list(frame_names),
        "source_fingerprint": _source_stat_fingerprint(source_files),
        "settings": dict(settings),
    }


def _read_metadata(directory: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads((directory / FRAME_STAGE_METADATA_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def stage_is_reusable(identity: dict[str, Any], output_directories: Sequence[Path]) -> bool:
    if not output_directories:
        return False
    metadata = _read_metadata(output_directories[0])
    if metadata is None or any(metadata.get(key) != value for key, value in identity.items()):
        return False
    fingerprint = metadata.get("fingerprint")
    unhashed = {key: value for key, value in metadata.items() if key != "fingerprint"}
    shape = metadata.get("output_shape")
    bit_depth = metadata.get("bit_depth")
    if (
        not isinstance(fingerprint, str)
        or fingerprint != canonical_json_hash(unhashed)
        or not isinstance(shape, list)
        or len(shape) not in {2, 3}
        or not all(isinstance(value, int) and value > 0 for value in shape)
        or not isinstance(bit_depth, int)
    ):
        return False
    expected_names = identity.get("frame_names")
    if not isinstance(expected_names, list):
        return False
    output_files: list[Path] = []
    for directory in output_directories:
        if not directory.is_dir():
            return False
        if sorted(path.name for path in directory.glob("*.png")) != expected_names:
            return False
        output_files.extend(directory / name for name in expected_names)
    return png_headers_match(output_files, shape=shape, bit_depth=bit_depth)


def clear_stage_outputs(output_directories: Sequence[Path]) -> None:
    for directory in output_directories:
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.glob("*.png"):
            path.unlink(missing_ok=True)
    if output_directories:
        (output_directories[0] / FRAME_STAGE_METADATA_NAME).unlink(missing_ok=True)


def complete_stage(
    identity: dict[str, Any],
    output_directories: Sequence[Path],
    *,
    shape: Sequence[int],
    bit_depth: int = 8,
) -> bool:
    if not output_directories:
        return False
    expected_names = identity.get("frame_names")
    if not isinstance(expected_names, list):
        return False
    normalized_shape = [int(value) for value in shape]
    output_files: list[Path] = []
    for directory in output_directories:
        if sorted(path.name for path in directory.glob("*.png")) != expected_names:
            return False
        output_files.extend(directory / name for name in expected_names)
    if not png_headers_match(output_files, shape=normalized_shape, bit_depth=bit_depth):
        return False
    metadata = {
        **identity,
        "output_shape": normalized_shape,
        "bit_depth": bit_depth,
    }
    metadata["fingerprint"] = canonical_json_hash(metadata)
    metadata_path = output_directories[0] / FRAME_STAGE_METADATA_NAME
    temporary = metadata_path.with_name(f"{metadata_path.name}.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(metadata_path)
    return True
