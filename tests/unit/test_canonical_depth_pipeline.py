"""Integration tests for the file-backed canonical depth pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.inference.depth.types import DepthBatch, DepthRepresentation
from src.depth_surge_3d.processing.frames import depth_processor as depth_processor_module
from src.depth_surge_3d.processing.frames.depth_normalizer import canonicalize_depth
from src.depth_surge_3d.processing.frames.depth_processor import DepthMapProcessor
from src.depth_surge_3d.processing.frames.depth_storage import canonical_json_hash


class FakeRelativeDepthEstimator:
    """Small native-resolution estimator used to exercise orchestration."""

    def __init__(self) -> None:
        self.calls = 0
        self.fail_on_inference = False
        self.revision = "test-revision-1"
        self.batch_lengths: list[int] = []

    def get_model_info(self) -> dict[str, str]:
        return {"family": "fake-relative", "revision": self.revision}

    def estimate_depth_batch(self, frames: np.ndarray, **_kwargs) -> DepthBatch:
        if self.fail_on_inference:
            raise AssertionError("resume must not repeat completed inference")
        self.calls += 1
        self.batch_lengths.append(len(frames))
        native = np.array(
            [[0.0, 0.25, 0.5], [0.5, 0.75, 1.0]],
            dtype=np.float32,
        )
        values = np.repeat(native[None, ...], len(frames), axis=0)
        return DepthBatch(values, DepthRepresentation.RELATIVE_DEPTH)


@pytest.fixture
def source_frames(tmp_path: Path) -> list[Path]:
    frame_dir = tmp_path / "00_original_frames"
    frame_dir.mkdir()
    result = []
    for index in range(4):
        frame = np.full((6, 8, 3), index * 30, dtype=np.uint8)
        path = frame_dir / f"frame_{index:06d}.png"
        assert cv2.imwrite(str(path), frame)
        result.append(path)
    return result


@pytest.fixture
def stage_directories(tmp_path: Path) -> dict[str, Path]:
    return {
        "base": tmp_path,
        "frames": tmp_path / "00_original_frames",
        "scene_data": tmp_path / "01_scene_data",
        "depth_raw": tmp_path / "02_depth_raw",
        "disparity_maps": tmp_path / "03_disparity_maps",
    }


@pytest.fixture
def pipeline_settings() -> dict[str, object]:
    return {
        "depth_resolution": "6",
        "target_fps": 30,
        "keep_intermediates": True,
        "super_sample": "none",
        "scene_detection": False,
        "scene_cut_threshold": 0.55,
        "min_scene_frames": 8,
        "raw_storage_dtype": "auto",
    }


def _run_pipeline(
    estimator: FakeRelativeDepthEstimator,
    source_frames: list[Path],
    stage_directories: dict[str, Path],
    settings: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> list[Path]:
    processor = DepthMapProcessor(estimator)
    monkeypatch.setattr(processor, "_determine_chunk_params", lambda *_args: (2, 6))
    monkeypatch.setattr(processor, "_clear_gpu_memory", lambda: None)
    result = processor.generate_depth_map_files(
        source_frames,
        settings,
        stage_directories,
        progress_tracker=None,
    )
    assert result is not None
    return result


def test_native_frame_preprocessing_has_a_new_fingerprint_version(
    source_frames: list[Path],
    pipeline_settings: dict[str, object],
) -> None:
    processor = DepthMapProcessor(FakeRelativeDepthEstimator())

    fingerprint = processor._raw_semantic_fingerprint(source_frames, pipeline_settings)

    assert fingerprint["preprocessing_algorithm"] == "native-depth-adapter-v2"


def test_file_pipeline_persists_native_raw_before_canonicalization(
    source_frames: list[Path],
    stage_directories: dict[str, Path],
    pipeline_settings: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = stage_directories["depth_raw"]
    real_canonicalize = canonicalize_depth

    def guarded_canonicalize(*args, **kwargs):
        assert len(list(raw_dir.glob("*.npz"))) == len(source_frames)
        return real_canonicalize(*args, **kwargs)

    monkeypatch.setattr(
        depth_processor_module,
        "canonicalize_depth",
        guarded_canonicalize,
        raising=False,
    )
    estimator = FakeRelativeDepthEstimator()

    output_files = _run_pipeline(
        estimator,
        source_frames,
        stage_directories,
        pipeline_settings,
        monkeypatch,
    )

    assert estimator.calls == 2
    raw_files = sorted(raw_dir.glob("*.npz"))
    assert len(raw_files) == len(source_frames)
    with np.load(raw_files[0], allow_pickle=False) as payload:
        assert payload["values"].shape == (2, 3)
        assert payload["values"].dtype == np.float16

    scene_dir = stage_directories["scene_data"]
    manifest = json.loads((scene_dir / "scene_manifest.json").read_text())
    assert manifest["status"] == "final"
    assert (scene_dir / "depth_samples.npz").is_file()
    assert (scene_dir / "depth_bounds.json").is_file()

    canonical_dir = stage_directories["disparity_maps"]
    assert output_files == [canonical_dir / f"{path.stem}.png" for path in source_frames]
    assert all(
        cv2.imread(str(path), cv2.IMREAD_UNCHANGED).dtype == np.uint16 for path in output_files
    )
    metadata = json.loads((canonical_dir / "metadata.json").read_text())
    assert metadata["representation"] == "relative_disparity"
    assert metadata["near_value"] == 1.0
    assert metadata["far_value"] == 0.0
    assert metadata["encoding_scale"] == 65535.0
    assert metadata["source_raw_fingerprint"]
    assert metadata["scene_manifest_fingerprint"]
    assert metadata["depth_bounds_fingerprint"]


def test_resume_from_candidate_manifest_reuses_raw_and_is_byte_identical(
    source_frames: list[Path],
    stage_directories: dict[str, Path],
    pipeline_settings: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimator = FakeRelativeDepthEstimator()
    first_files = _run_pipeline(
        estimator,
        source_frames,
        stage_directories,
        pipeline_settings,
        monkeypatch,
    )
    expected_bytes = [path.read_bytes() for path in first_files]

    scene_manifest = stage_directories["scene_data"] / "scene_manifest.json"
    manifest = json.loads(scene_manifest.read_text())
    manifest["status"] = "candidate"
    manifest["scene_ids"] = [0] * len(source_frames)
    manifest.pop("bounds_fingerprint", None)
    scene_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    for path in first_files:
        path.unlink()
    (stage_directories["disparity_maps"] / "metadata.json").unlink()

    estimator.fail_on_inference = True
    resumed_files = _run_pipeline(
        estimator,
        source_frames,
        stage_directories,
        pipeline_settings,
        monkeypatch,
    )

    assert [path.read_bytes() for path in resumed_files] == expected_bytes


def test_source_frame_change_invalidates_scene_and_downstream_stages(
    source_frames: list[Path],
    stage_directories: dict[str, Path],
    pipeline_settings: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimator = FakeRelativeDepthEstimator()
    _run_pipeline(
        estimator,
        source_frames,
        stage_directories,
        pipeline_settings,
        monkeypatch,
    )
    manifest_path = stage_directories["scene_data"] / "scene_manifest.json"
    first_manifest = json.loads(manifest_path.read_text())
    first_source_fingerprint = first_manifest["source_frame_fingerprint"]

    changed = np.full((6, 8, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(source_frames[0]), changed)
    estimator.calls = 0
    estimator.batch_lengths.clear()
    _run_pipeline(
        estimator,
        source_frames,
        stage_directories,
        pipeline_settings,
        monkeypatch,
    )

    second_manifest = json.loads(manifest_path.read_text())
    assert second_manifest["source_frame_fingerprint"] != first_source_fingerprint
    assert estimator.calls == 2


def test_model_fingerprint_change_rebuilds_raw_stage_instead_of_aborting(
    source_frames: list[Path],
    stage_directories: dict[str, Path],
    pipeline_settings: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimator = FakeRelativeDepthEstimator()
    _run_pipeline(
        estimator,
        source_frames,
        stage_directories,
        pipeline_settings,
        monkeypatch,
    )
    raw_metadata_path = stage_directories["depth_raw"] / "metadata.json"
    first_raw_metadata = json.loads(raw_metadata_path.read_text())
    first_raw_fingerprint = first_raw_metadata["fingerprint"]
    estimator.revision = "test-revision-2"
    estimator.calls = 0
    estimator.batch_lengths.clear()

    _run_pipeline(
        estimator,
        source_frames,
        stage_directories,
        pipeline_settings,
        monkeypatch,
    )

    assert estimator.calls == 2
    second_raw_metadata = json.loads(raw_metadata_path.read_text())
    second_raw_fingerprint = second_raw_metadata["fingerprint"]
    assert second_raw_fingerprint != first_raw_fingerprint

    scene_dir = stage_directories["scene_data"]
    bounds = json.loads((scene_dir / "depth_bounds.json").read_text())
    assert bounds["source_raw_fingerprint"] == second_raw_fingerprint
    with np.load(scene_dir / "depth_samples.npz", allow_pickle=False) as payload:
        assert str(payload["source_raw_fingerprint"].item()) == second_raw_fingerprint


def test_partial_temporal_chunk_replays_original_context_without_overwriting_completed_raw(
    source_frames: list[Path],
    stage_directories: dict[str, Path],
    pipeline_settings: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimator = FakeRelativeDepthEstimator()
    first_files = _run_pipeline(
        estimator,
        source_frames,
        stage_directories,
        pipeline_settings,
        monkeypatch,
    )
    completed_raw = stage_directories["depth_raw"] / f"{source_frames[0].stem}.npz"
    completed_bytes = completed_raw.read_bytes()
    missing_raw = stage_directories["depth_raw"] / f"{source_frames[1].stem}.npz"
    missing_raw.unlink()
    for path in first_files:
        path.unlink()
    (stage_directories["disparity_maps"] / "metadata.json").unlink()
    estimator.calls = 0
    estimator.batch_lengths.clear()

    _run_pipeline(
        estimator,
        source_frames,
        stage_directories,
        pipeline_settings,
        monkeypatch,
    )

    assert estimator.batch_lengths == [2]
    assert completed_raw.read_bytes() == completed_bytes


def test_completed_canonical_stage_resumes_without_raw_payloads(
    source_frames: list[Path],
    stage_directories: dict[str, Path],
    pipeline_settings: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = dict(pipeline_settings, keep_intermediates=False)
    estimator = FakeRelativeDepthEstimator()
    first_files = _run_pipeline(
        estimator,
        source_frames,
        stage_directories,
        settings,
        monkeypatch,
    )
    expected_bytes = [path.read_bytes() for path in first_files]
    assert not list(stage_directories["depth_raw"].glob("*.npz"))

    estimator.fail_on_inference = True
    resumed_files = _run_pipeline(
        estimator,
        source_frames,
        stage_directories,
        settings,
        monkeypatch,
    )
    assert [path.read_bytes() for path in resumed_files] == expected_bytes


def test_global_cache_restore_copies_metadata_into_local_canonical_stage(
    tmp_path: Path,
    source_frames: list[Path],
    stage_directories: dict[str, Path],
    pipeline_settings: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "global_cache_entry"
    cache_dir.mkdir()
    cached_files = []
    for index in range(len(source_frames)):
        path = cache_dir / f"depth_{index:06d}.png"
        assert cv2.imwrite(str(path), np.full((2, 3), index, dtype=np.uint16))
        cached_files.append(path)
    metadata = {
        "schema_version": 1,
        "algorithm_version": "scene-percentile-v1",
        "representation": "relative_disparity",
        "near_value": 1.0,
        "far_value": 0.0,
        "encoding": "uint16_png",
        "encoding_scale": 65535.0,
        "num_frames": len(source_frames),
        "frame_names": [path.name for path in source_frames],
        "native_shape": [2, 3],
        "source_raw_fingerprint": "raw-fingerprint",
        "source_model_fingerprint": "model-fingerprint",
        "scene_manifest_fingerprint": "scene-fingerprint",
        "depth_bounds_fingerprint": "bounds-fingerprint",
    }
    metadata["fingerprint"] = canonical_json_hash(metadata)
    (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    monkeypatch.setattr(
        depth_processor_module,
        "get_cached_depth_map_files",
        lambda *_args, **_kwargs: cached_files,
    )
    estimator = FakeRelativeDepthEstimator()
    estimator.fail_on_inference = True
    settings = dict(pipeline_settings, video_path=str(tmp_path / "input.mp4"))

    restored = _run_pipeline(
        estimator,
        source_frames,
        stage_directories,
        settings,
        monkeypatch,
    )

    local_dir = stage_directories["disparity_maps"]
    assert restored == [local_dir / f"{path.stem}.png" for path in source_frames]
    assert all(path.parent == local_dir for path in restored)
    assert json.loads((local_dir / "metadata.json").read_text()) == metadata
    assert [cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[0, 0] for path in restored] == [0, 1, 2, 3]


def test_legacy_array_api_rejects_more_than_512_mib_before_inference(
    tmp_path: Path,
) -> None:
    frame_path = tmp_path / "large.png"
    assert cv2.imwrite(str(frame_path), np.zeros((1024, 1024, 3), dtype=np.uint8))
    estimator = FakeRelativeDepthEstimator()
    estimator.fail_on_inference = True
    processor = DepthMapProcessor(estimator)

    with pytest.raises(MemoryError, match="file-backed"):
        processor.generate_depth_maps(
            [frame_path] * 129,
            {
                "depth_resolution": "1024",
                "super_sample": "none",
                "target_fps": 30,
            },
            {},
            progress_tracker=None,
        )
