"""Authoritative metadata contract for canonical disparity payloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


CANONICAL_DEPTH_SCHEMA_VERSION = 1
CANONICAL_DEPTH_ALGORITHM_VERSION = "scene-percentile-v1"
CANONICAL_METADATA_REQUIRED_FIELDS = {
    "schema_version",
    "algorithm_version",
    "representation",
    "near_value",
    "far_value",
    "encoding",
    "encoding_scale",
    "num_frames",
    "frame_names",
    "native_shape",
    "source_raw_fingerprint",
    "source_model_fingerprint",
    "scene_manifest_fingerprint",
    "depth_bounds_fingerprint",
    "fingerprint",
}


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return repr(value)


def canonical_json_hash(payload: Any) -> str:
    """Hash metadata independently of dictionary insertion order."""

    encoded = json.dumps(
        jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()
