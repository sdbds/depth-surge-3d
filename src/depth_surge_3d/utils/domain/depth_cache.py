"""
Depth map caching system for faster re-processing.

Caches depth maps globally (not tied to specific output batches) so users
can experiment with different stereo/VR settings without re-computing depth.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from ...core.depth_contract import (
    CANONICAL_DEPTH_ALGORITHM_VERSION,
    CANONICAL_DEPTH_SCHEMA_VERSION,
    CANONICAL_METADATA_REQUIRED_FIELDS,
    canonical_json_hash,
)
from ...core.file_identity import file_sample_fingerprint

DEPTH_CACHE_SETTING_KEYS = (
    "depth_model_version",
    "model_path",
    "model_size",
    "depth_resolution",
    "use_metric_depth",
    "device",
    "start_time",
    "end_time",
    "super_sample",
    "temporal_window_size",
    "temporal_window_overlap",
    "denoising_steps",
    "seed",
    "scene_detection",
    "scene_cut_threshold",
    "min_scene_frames",
    "raw_storage_dtype",
    "model_fingerprint",
)


def _is_valid_canonical_metadata(metadata: dict[str, Any], num_frames: int) -> bool:
    if metadata.get("schema_version") != CANONICAL_DEPTH_SCHEMA_VERSION:
        return False
    if metadata.get("algorithm_version") != CANONICAL_DEPTH_ALGORITHM_VERSION:
        return False
    if not CANONICAL_METADATA_REQUIRED_FIELDS.issubset(metadata):
        return False
    if metadata.get("representation") != "relative_disparity":
        return False
    if metadata.get("near_value") != 1.0 or metadata.get("far_value") != 0.0:
        return False
    if metadata.get("encoding") != "uint16_png" or metadata.get("encoding_scale") != 65535.0:
        return False
    if metadata.get("num_frames") != num_frames:
        return False
    if len(metadata.get("frame_names", [])) != num_frames:
        return False
    fingerprint = metadata.get("fingerprint")
    unhashed = {key: value for key, value in metadata.items() if key != "fingerprint"}
    return isinstance(fingerprint, str) and fingerprint == canonical_json_hash(unhashed)


def get_cache_dir() -> Path:
    """Get the global depth cache directory."""
    # Use XDG_CACHE_HOME if available, otherwise ~/.cache
    import os

    cache_home = os.environ.get("XDG_CACHE_HOME")
    if cache_home:
        cache_dir = Path(cache_home) / "depth-surge-3d" / "depth_cache"
    else:
        cache_dir = Path.home() / ".cache" / "depth-surge-3d" / "depth_cache"

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def compute_cache_key(video_path: str, depth_settings: dict[str, Any]) -> str:
    """
    Compute cache key for depth maps.

    The cache key is based on:
    - The shared bounded-cost source video content fingerprint
    - Settings that affect depth: model version, model size, depth resolution,
      metric vs relative

    Args:
        video_path: Path to input video
        depth_settings: Depth-related settings

    Returns:
        32-character hex cache key
    """
    hasher = hashlib.blake2b(digest_size=16)  # 16 bytes = 32 hex chars

    hasher.update(file_sample_fingerprint(video_path).encode("ascii"))

    # Hash depth-relevant settings (sorted for consistency)
    settings_for_hash = {k: depth_settings.get(k) for k in DEPTH_CACHE_SETTING_KEYS}
    settings_json = json.dumps(settings_for_hash, sort_keys=True)
    hasher.update(settings_json.encode())

    return hasher.hexdigest()


def get_cached_depth_map_files(
    video_path: str,
    depth_settings: dict[str, Any],
    num_frames: int,
) -> list[Path] | None:
    """Return validated cache paths without decoding the depth images."""
    expected_model_fingerprint = depth_settings.get("model_fingerprint")
    if not isinstance(expected_model_fingerprint, str) or not expected_model_fingerprint:
        return None

    cache_entry_dir = get_cache_dir() / compute_cache_key(video_path, depth_settings)
    metadata_file = cache_entry_dir / "metadata.json"
    if not metadata_file.exists():
        return None

    try:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        if not _is_valid_canonical_metadata(metadata, num_frames):
            return None
        if metadata.get("source_model_fingerprint") != expected_model_fingerprint:
            return None

        depth_files = [cache_entry_dir / f"depth_{i:06d}.png" for i in range(num_frames)]
        if not all(path.is_file() for path in depth_files):
            return None
        return depth_files
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _load_cacheable_metadata(
    depth_files: list[Path], expected_model_fingerprint: str
) -> dict[str, Any] | None:
    source_metadata_path = depth_files[0].parent / "metadata.json"
    try:
        with source_metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    checks = (
        _is_valid_canonical_metadata(metadata, len(depth_files)),
        metadata.get("source_model_fingerprint") == expected_model_fingerprint,
        metadata.get("frame_names") == [path.name for path in depth_files],
    )
    return metadata if all(checks) else None


def save_depth_map_files_to_cache(
    video_path: str, depth_settings: dict[str, Any], depth_files: list[Path]
) -> bool:
    """Copy disk-backed uint16 depth maps into the global cache with bounded memory."""
    metadata_tmp: Path | None = None
    try:
        if not depth_files:
            return False
        expected_model_fingerprint = depth_settings.get("model_fingerprint")
        if not isinstance(expected_model_fingerprint, str) or not expected_model_fingerprint:
            return False
        metadata = _load_cacheable_metadata(depth_files, expected_model_fingerprint)
        if metadata is None:
            return False
        cache_entry_dir = get_cache_dir() / compute_cache_key(video_path, depth_settings)
        cache_entry_dir.mkdir(parents=True, exist_ok=True)

        for i, source in enumerate(depth_files):
            destination = cache_entry_dir / f"depth_{i:06d}.png"
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)

        metadata_file = cache_entry_dir / "metadata.json"
        metadata_tmp = cache_entry_dir / "metadata.json.tmp"
        with open(metadata_tmp, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        metadata_tmp.replace(metadata_file)
        return True
    except (OSError, TypeError, ValueError):
        if metadata_tmp and metadata_tmp.exists():
            metadata_tmp.unlink(missing_ok=True)
        return False


def clear_cache() -> int:
    """
    Clear all cached depth maps.

    Returns:
        Number of cache entries removed
    """
    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        return 0

    count = 0
    for entry_dir in cache_dir.iterdir():
        if entry_dir.is_dir():
            try:
                import shutil

                shutil.rmtree(entry_dir)
                count += 1
            except Exception:
                pass

    return count


def get_cache_size() -> tuple[int, int]:
    """
    Get cache statistics.

    Returns:
        (number_of_entries, total_size_bytes)
    """
    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        return (0, 0)

    num_entries = 0
    total_size = 0

    for entry_dir in cache_dir.iterdir():
        if entry_dir.is_dir():
            num_entries += 1
            for file in entry_dir.rglob("*"):
                if file.is_file():
                    total_size += file.stat().st_size

    return (num_entries, total_size)
