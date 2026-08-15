"""Shot-aware file-backed V2 temporal inference tests."""

from __future__ import annotations

import shutil
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.inference.depth.types import DepthBatch, DepthRepresentation
from src.depth_surge_3d.processing.frames.depth_processor import DepthMapProcessor
from src.depth_surge_3d.processing.frames.scene_analyzer import (
    SCENE_ALGORITHM_VERSION,
    SCENE_SCHEMA_VERSION,
)


class FakeSequenceEstimator:
    model_path = None

    def __init__(self, *, fail_shot_length: int | None = None) -> None:
        self.fail_shot_length = fail_shot_length
        self.shot_lengths: list[int] = []
        self.loader_requests: list[list[int]] = []
        self.loaded_markers: list[list[int]] = []

    @staticmethod
    def get_model_info() -> dict[str, object]:
        return {
            "family": "fake-v2",
            "revision": "immutable-test-model",
            "inference_algorithm": "vda-offline-shot-v1",
        }

    def estimate_depth_batch(self, *_args, **_kwargs):
        raise AssertionError("V2 file-backed inference must use iter_sequence_depth")

    def iter_sequence_depth(
        self,
        frame_count,
        load_frames,
        *,
        target_fps,
        input_size,
        fp32,
    ):
        del target_fps, input_size, fp32
        self.shot_lengths.append(frame_count)
        requested = list(range(frame_count))
        frames = load_frames(requested)
        markers = [int(frame[0, 0, 0]) for frame in frames]
        self.loader_requests.append(requested)
        self.loaded_markers.append(markers)
        values = np.asarray(markers, dtype=np.float32).reshape(frame_count, 1, 1)
        split = min(1, frame_count)
        yield 0, DepthBatch(values[:split].copy(), DepthRepresentation.INVERSE_DEPTH)
        if self.fail_shot_length == frame_count:
            raise RuntimeError(f"failed shot length {frame_count}")
        if split < frame_count:
            yield split, DepthBatch(values[split:].copy(), DepthRepresentation.INVERSE_DEPTH)


@pytest.fixture
def v2_frames(tmp_path: Path) -> list[Path]:
    frame_dir = tmp_path / "00_original_frames"
    frame_dir.mkdir()
    paths = []
    for index in range(9):
        frame = np.full((2, 3, 3), index + 1, dtype=np.uint8)
        path = frame_dir / f"frame_{index:06d}.png"
        assert cv2.imwrite(str(path), frame)
        paths.append(path)
    return paths


@pytest.fixture
def v2_directories(tmp_path: Path) -> dict[str, Path]:
    return {
        "base": tmp_path,
        "scene_data": tmp_path / "01_scene_data",
        "depth_raw": tmp_path / "02_depth_raw",
        "disparity_maps": tmp_path / "03_disparity_maps",
    }


@pytest.fixture
def v2_settings() -> dict[str, object]:
    return {
        "depth_model_version": "v2",
        "depth_resolution": "6",
        "target_fps": 30,
        "keep_intermediates": True,
        "super_sample": "none",
        "scene_detection": True,
        "scene_cut_threshold": 0.55,
        "min_scene_frames": 1,
        "raw_storage_dtype": "float32",
        "temporal_window_size": 32,
        "temporal_window_overlap": 10,
    }


def _candidate_manifest(frame_files: list[Path]) -> dict[str, object]:
    return {
        "schema_version": SCENE_SCHEMA_VERSION,
        "algorithm_version": SCENE_ALGORITHM_VERSION,
        "status": "candidate",
        "frame_names": [path.name for path in frame_files],
        "scene_ids": [0, 0, 0, 1, 1, 1, 1, 2, 2],
        "candidate_cuts": [3, 7],
        "settings": {"enabled": True, "threshold": 0.55, "min_frames": 1},
        "source_frame_fingerprint": "fixture-source",
    }


def _run_v2_pipeline(
    estimator: FakeSequenceEstimator,
    frame_files: list[Path],
    settings: dict[str, object],
    directories: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> list[Path]:
    processor = DepthMapProcessor(estimator)
    manifest = _candidate_manifest(frame_files)
    monkeypatch.setattr(processor, "_determine_chunk_params", lambda *_args: (2, 6))
    monkeypatch.setattr(processor, "_clear_gpu_memory", lambda: None)
    monkeypatch.setattr(processor, "_load_or_analyze_scenes", lambda *_args: manifest)
    result = processor.generate_depth_map_files(
        frame_files,
        settings,
        directories,
        progress_tracker=None,
    )
    assert result is not None
    return result


def test_v2_pipeline_partitions_candidate_shots_and_maps_local_indexes(
    v2_frames: list[Path],
    v2_directories: dict[str, Path],
    v2_settings: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimator = FakeSequenceEstimator()

    outputs = _run_v2_pipeline(
        estimator,
        v2_frames,
        v2_settings,
        v2_directories,
        monkeypatch,
    )

    assert len(outputs) == len(v2_frames)
    assert estimator.shot_lengths == [3, 4, 2]
    assert estimator.loader_requests == [[0, 1, 2], [0, 1, 2, 3], [0, 1]]
    assert estimator.loaded_markers == [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
    raw_files = sorted(v2_directories["depth_raw"].glob("*.npz"))
    assert len(raw_files) == len(v2_frames)
    metadata = json.loads(
        (v2_directories["depth_raw"] / "metadata.json").read_text(encoding="utf-8")
    )
    semantic = metadata["semantic_fingerprint"]
    assert semantic["model_info"]["inference_algorithm"] == "vda-offline-shot-v1"
    assert semantic["scene_algorithm_version"] == SCENE_ALGORITHM_VERSION
    assert semantic["depth_settings"]["scene_cut_threshold"] == 0.55


def test_v2_resume_reuses_complete_shots_and_recomputes_the_whole_partial_shot(
    v2_frames: list[Path],
    v2_directories: dict[str, Path],
    v2_settings: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_v2_pipeline(
        FakeSequenceEstimator(),
        v2_frames,
        v2_settings,
        v2_directories,
        monkeypatch,
    )
    raw_dir = v2_directories["depth_raw"]
    for index in (4, 5, 6):
        (raw_dir / f"frame_{index:06d}.npz").unlink()
    shutil.rmtree(v2_directories["disparity_maps"])
    estimator = FakeSequenceEstimator()

    _run_v2_pipeline(
        estimator,
        v2_frames,
        v2_settings,
        v2_directories,
        monkeypatch,
    )

    assert estimator.shot_lengths == [4]
    assert estimator.loaded_markers == [[4, 5, 6, 7]]
    assert len(list(raw_dir.glob("*.npz"))) == len(v2_frames)


def test_v2_failure_leaves_partial_shot_and_next_run_restarts_that_shot(
    v2_frames: list[Path],
    v2_directories: dict[str, Path],
    v2_settings: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failing = FakeSequenceEstimator(fail_shot_length=4)

    with pytest.raises(RuntimeError, match="failed shot length 4"):
        _run_v2_pipeline(
            failing,
            v2_frames,
            v2_settings,
            v2_directories,
            monkeypatch,
        )

    raw_dir = v2_directories["depth_raw"]
    assert sorted(path.stem for path in raw_dir.glob("*.npz")) == [
        "frame_000000",
        "frame_000001",
        "frame_000002",
        "frame_000003",
    ]
    assert not (v2_directories["disparity_maps"] / "metadata.json").exists()

    resumed = FakeSequenceEstimator()
    _run_v2_pipeline(
        resumed,
        v2_frames,
        v2_settings,
        v2_directories,
        monkeypatch,
    )

    assert resumed.shot_lengths == [4, 2]
    assert resumed.loaded_markers == [[4, 5, 6, 7], [8, 9]]
    assert len(list(raw_dir.glob("*.npz"))) == len(v2_frames)
