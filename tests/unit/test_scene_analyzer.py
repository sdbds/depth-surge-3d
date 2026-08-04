"""Deterministic scene pre-pass and finalization tests."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.inference.depth.types import DepthRepresentation
from src.depth_surge_3d.processing.frames.scene_analyzer import (
    analyze_scenes,
    finalize_scenes,
    sample_scene_depths,
)


def _write_frames(directory: Path, values: list[int]) -> list[Path]:
    directory.mkdir()
    paths = []
    for index, value in enumerate(values):
        path = directory / f"frame_{index:06d}.png"
        assert cv2.imwrite(str(path), np.full((8, 12, 3), value, dtype=np.uint8))
        paths.append(path)
    return paths


def _candidate_manifest(scene_ids: list[int]) -> dict:
    return {
        "schema_version": 1,
        "algorithm_version": "luma-bhattacharyya-v1",
        "status": "candidate",
        "frame_names": [f"frame_{i:06d}" for i in range(len(scene_ids))],
        "scene_ids": scene_ids,
        "candidate_cuts": [i for i in range(1, len(scene_ids)) if scene_ids[i] != scene_ids[i - 1]],
        "settings": {"enabled": True, "threshold": 0.55, "min_frames": 3},
    }


def test_scene_prepass_writes_deterministic_candidate_manifest(tmp_path: Path) -> None:
    frame_files = _write_frames(tmp_path / "frames", [0] * 4 + [255] * 4 + [0] * 4)
    output_dir = tmp_path / "scene"

    manifest = analyze_scenes(
        frame_files,
        output_dir,
        enabled=True,
        threshold=0.55,
        min_frames=3,
    )

    assert manifest["status"] == "candidate"
    assert manifest["candidate_cuts"] == [4, 8]
    assert manifest["scene_ids"] == [0] * 4 + [1] * 4 + [2] * 4
    assert json.loads((output_dir / "scene_manifest.json").read_text()) == manifest


def test_scene_prepass_can_be_disabled_and_drops_too_short_final_scene(tmp_path: Path) -> None:
    frames = _write_frames(tmp_path / "frames", [0] * 4 + [255] * 2)

    disabled = analyze_scenes(frames, tmp_path / "disabled", enabled=False)
    guarded = analyze_scenes(frames, tmp_path / "guarded", min_frames=3)

    assert disabled["scene_ids"] == [0] * 6
    assert disabled["candidate_cuts"] == []
    assert guarded["scene_ids"] == [0] * 6
    assert guarded["candidate_cuts"] == []


def test_scene_sampling_uses_representation_score_and_ignores_invalid_values(
    tmp_path: Path,
) -> None:
    raw_files = []
    for index, values in enumerate(
        [
            np.array([[0.0, 0.5], [1.0, np.nan]], dtype=np.float32),
            np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32),
        ]
    ):
        path = tmp_path / f"frame_{index:06d}.npz"
        np.savez_compressed(path, values=values)
        raw_files.append(path)

    samples = sample_scene_depths(
        raw_files,
        _candidate_manifest([0, 0]),
        DepthRepresentation.RELATIVE_DEPTH,
    )

    np.testing.assert_allclose(samples[0], [0.0, -0.5, -1.0, -0.2, -0.4, -0.6, -0.8])


def test_finalization_merges_left_to_right_from_pooled_sample_unions() -> None:
    manifest = _candidate_manifest([0, 0, 1, 1, 2, 2])
    samples = {
        0: np.linspace(0.0, 10.0, 101, dtype=np.float32),
        1: np.linspace(0.5, 10.5, 101, dtype=np.float32),
        2: np.linspace(1.0, 11.0, 101, dtype=np.float32),
    }

    final_manifest, bounds = finalize_scenes(manifest, samples)

    pooled = np.concatenate([samples[0], samples[1], samples[2]])
    assert final_manifest["status"] == "final"
    assert final_manifest["scene_ids"] == [0] * 6
    assert list(bounds) == [0]
    assert bounds[0].low == pytest.approx(float(np.percentile(pooled, 2)))
    assert bounds[0].high == pytest.approx(float(np.percentile(pooled, 98)))


def test_equal_flat_scenes_merge_without_division_by_zero() -> None:
    manifest = _candidate_manifest([0, 1])
    samples = {
        0: np.full(16, 2.0, dtype=np.float32),
        1: np.full(16, 2.0, dtype=np.float32),
    }

    final_manifest, bounds = finalize_scenes(manifest, samples)

    assert final_manifest["scene_ids"] == [0, 0]
    assert bounds[0].low == bounds[0].high == 2.0


def test_dissimilar_candidate_scenes_remain_separate() -> None:
    manifest = _candidate_manifest([0, 0, 1, 1])
    samples = {
        0: np.linspace(0.0, 1.0, 101, dtype=np.float32),
        1: np.linspace(10.0, 11.0, 101, dtype=np.float32),
    }

    final_manifest, bounds = finalize_scenes(manifest, samples)

    assert final_manifest["scene_ids"] == [0, 0, 1, 1]
    assert list(bounds) == [0, 1]


def test_finalization_rejects_non_candidate_manifest() -> None:
    manifest = _candidate_manifest([0])
    manifest["status"] = "final"

    with pytest.raises(ValueError, match="candidate"):
        finalize_scenes(manifest, {0: np.array([1.0], dtype=np.float32)})
