"""Deterministic RGB scene pre-pass and raw-depth scene finalization."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ...inference.depth.types import DepthRepresentation
from .depth_normalizer import SceneDepthBounds, depth_to_score


SCENE_SCHEMA_VERSION = 1
SCENE_ALGORITHM_VERSION = "luma-bhattacharyya-v1"
SCENE_MERGE_RATIO = 0.10
MAX_SCENE_HISTOGRAM_WORKERS = 8


@dataclass
class _SceneBlock:
    source_ids: list[int]
    start: int
    stop: int
    sample_count: int
    bounds: SceneDepthBounds


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _luma_histogram(path: Path) -> np.ndarray:
    frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise OSError(f"Could not read source frame for scene analysis: {path}")
    histogram = cv2.calcHist([frame], [0], None, [32], [0, 256]).astype(np.float32)
    normalized = np.empty_like(histogram)
    cv2.normalize(histogram, normalized, alpha=1.0, norm_type=cv2.NORM_L1)
    return normalized.reshape(-1)


def _load_luma_histograms(frame_files: list[Path]) -> list[np.ndarray]:
    worker_count = min(
        len(frame_files),
        MAX_SCENE_HISTOGRAM_WORKERS,
        max(1, (os.cpu_count() or 1) - 2),
    )
    if worker_count == 1:
        return [_luma_histogram(path) for path in frame_files]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(_luma_histogram, frame_files))


def analyze_scenes(
    frame_files: list[Path],
    output_dir: Path,
    *,
    enabled: bool = True,
    threshold: float = 0.55,
    min_frames: int = 8,
) -> dict[str, Any]:
    """Persist provisional frame-to-scene IDs before depth inference."""

    if not frame_files:
        raise ValueError("Scene analysis requires at least one frame")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Scene cut threshold must be in [0,1]")
    if min_frames < 1:
        raise ValueError("Minimum scene length must be positive")

    cuts: list[int] = []
    if enabled:
        histograms = _load_luma_histograms(frame_files)
        previous = histograms[0]
        last_cut = 0
        for index, current in enumerate(histograms[1:], start=1):
            distance = float(cv2.compareHist(previous, current, cv2.HISTCMP_BHATTACHARYYA))
            if distance >= threshold and index - last_cut >= min_frames:
                cuts.append(index)
                last_cut = index
            previous = current

        while cuts and len(frame_files) - cuts[-1] < min_frames:
            cuts.pop()

    scene_ids: list[int] = []
    scene_id = 0
    cut_set = set(cuts)
    for index in range(len(frame_files)):
        if index in cut_set:
            scene_id += 1
        scene_ids.append(scene_id)

    manifest: dict[str, Any] = {
        "schema_version": SCENE_SCHEMA_VERSION,
        "algorithm_version": SCENE_ALGORITHM_VERSION,
        "status": "candidate",
        "frame_names": [path.name for path in frame_files],
        "scene_ids": scene_ids,
        "candidate_cuts": cuts,
        "settings": {
            "enabled": bool(enabled),
            "threshold": float(threshold),
            "min_frames": int(min_frames),
        },
    }
    _atomic_write_json(output_dir / "scene_manifest.json", manifest)
    return manifest


def _even_indexes(length: int, limit: int) -> np.ndarray:
    count = min(length, limit)
    if count == length:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, length - 1, count)).astype(np.int64))


def _load_raw_map(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if "values" not in payload:
            raise ValueError(f"Raw depth file has no values array: {path}")
        values = np.asarray(payload["values"], dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Raw depth file must contain one [H,W] map: {path}")
    return values


def sample_scene_depths(
    raw_files: list[Path],
    manifest: dict[str, Any],
    representation: DepthRepresentation,
) -> dict[int, np.ndarray]:
    """Collect bounded deterministic disparity-score samples per candidate scene."""

    scene_ids = [int(value) for value in manifest.get("scene_ids", [])]
    if manifest.get("status") != "candidate":
        raise ValueError("Depth sampling requires a candidate scene manifest")
    if len(raw_files) != len(scene_ids):
        raise ValueError("Raw file count must match scene manifest frame count")

    samples: dict[int, np.ndarray] = {}
    for scene_id in dict.fromkeys(scene_ids):
        frame_indexes = [index for index, value in enumerate(scene_ids) if value == scene_id]
        selected = _even_indexes(len(frame_indexes), 32)
        pieces: list[np.ndarray] = []
        for local_index in selected:
            values = _load_raw_map(raw_files[frame_indexes[int(local_index)]])
            score, valid = depth_to_score(values, representation)
            y_indexes = _even_indexes(values.shape[0], 64)
            x_indexes = _even_indexes(values.shape[1], 64)
            sampled_score = score[np.ix_(y_indexes, x_indexes)].reshape(-1)
            sampled_valid = valid[np.ix_(y_indexes, x_indexes)].reshape(-1)
            pieces.append(sampled_score[sampled_valid])
        samples[scene_id] = (
            np.concatenate(pieces).astype(np.float32, copy=False)
            if pieces and any(piece.size for piece in pieces)
            else np.empty(0, dtype=np.float32)
        )
    return samples


def _bounds(samples: np.ndarray) -> SceneDepthBounds:
    finite = np.asarray(samples, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return SceneDepthBounds(0.0, 0.0)
    return SceneDepthBounds(
        float(np.percentile(finite, 2)),
        float(np.percentile(finite, 98)),
    )


def _can_merge(left: _SceneBlock, right: _SceneBlock) -> bool:
    if left.sample_count == 0 or right.sample_count == 0:
        return left.sample_count == right.sample_count
    combined_span = max(left.bounds.high, right.bounds.high) - min(
        left.bounds.low, right.bounds.low
    )
    bound_difference = max(
        abs(left.bounds.low - right.bounds.low),
        abs(left.bounds.high - right.bounds.high),
    )
    if combined_span == 0.0:
        return bound_difference == 0.0
    return bound_difference <= SCENE_MERGE_RATIO * combined_span


def finalize_scenes(
    candidate_manifest: dict[str, Any],
    samples: dict[int, np.ndarray],
) -> tuple[dict[str, Any], dict[int, SceneDepthBounds]]:
    """Merge provisional scenes to a deterministic fixed point and recompute bounds."""

    if candidate_manifest.get("status") != "candidate":
        raise ValueError("Scene finalization requires a candidate manifest")
    source_scene_ids = [int(value) for value in candidate_manifest.get("scene_ids", [])]
    ordered_ids = list(dict.fromkeys(source_scene_ids))
    if set(ordered_ids) != set(samples):
        raise ValueError("Candidate samples must cover every manifest scene")

    ordered_samples = [
        np.asarray(samples[scene_id], dtype=np.float32).reshape(-1) for scene_id in ordered_ids
    ]
    pooled_samples = (
        np.concatenate(ordered_samples) if ordered_samples else np.empty(0, dtype=np.float32)
    )
    blocks: list[_SceneBlock] = []
    offset = 0
    for scene_id, values in zip(ordered_ids, ordered_samples):
        stop = offset + values.size
        blocks.append(
            _SceneBlock(
                source_ids=[scene_id],
                start=offset,
                stop=stop,
                sample_count=values.size,
                bounds=_bounds(values),
            )
        )
        offset = stop

    while len(blocks) > 1:
        changed = False
        index = 0
        while index < len(blocks) - 1:
            left = blocks[index]
            right = blocks[index + 1]
            if _can_merge(left, right):
                left.source_ids.extend(right.source_ids)
                left.stop = right.stop
                left.sample_count += right.sample_count
                left.bounds = _bounds(pooled_samples[left.start : left.stop])
                del blocks[index + 1]
                changed = True
            else:
                index += 1
        if not changed:
            break

    old_to_new: dict[int, int] = {}
    final_bounds: dict[int, SceneDepthBounds] = {}
    for final_id, block in enumerate(blocks):
        final_bounds[final_id] = block.bounds
        for source_id in block.source_ids:
            old_to_new[int(source_id)] = final_id

    final_scene_ids = [old_to_new[scene_id] for scene_id in source_scene_ids]
    final_cuts = [
        index
        for index in range(1, len(final_scene_ids))
        if final_scene_ids[index] != final_scene_ids[index - 1]
    ]
    final_manifest = dict(candidate_manifest)
    final_manifest.update(
        {
            "status": "final",
            "scene_ids": final_scene_ids,
            "final_cuts": final_cuts,
        }
    )
    return final_manifest, final_bounds
