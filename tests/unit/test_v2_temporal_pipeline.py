"""Shot-aware file-backed V2 temporal inference tests."""

from __future__ import annotations

import shutil
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.depth_surge_3d.core.depth_contract import canonical_json_hash
from src.depth_surge_3d.inference.depth.types import DepthBatch, DepthRepresentation
from src.depth_surge_3d.processing.frames import depth_processor as depth_processor_module
from src.depth_surge_3d.processing.frames.depth_processor import DepthMapProcessor
from src.depth_surge_3d.processing.frames.scene_analyzer import (
    SCENE_ALGORITHM_VERSION,
    SCENE_SCHEMA_VERSION,
)
from src.depth_surge_3d.utils.domain.depth_cache import compute_cache_key


class FakeSequenceEstimator:
    model_path = None

    def __init__(self, *, fail_shot_length: int | None = None) -> None:
        self.fail_shot_length = fail_shot_length
        self.shot_lengths: list[int] = []
        self.loader_requests: list[list[int]] = []
        self.loaded_markers: list[list[int]] = []
        self.input_sizes: list[int] = []
        self.fp32_values: list[bool] = []

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
        del target_fps
        self.shot_lengths.append(frame_count)
        self.input_sizes.append(input_size)
        self.fp32_values.append(fp32)
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


class RetryableCapacityEstimator(FakeSequenceEstimator):
    def __init__(
        self,
        *,
        oom_failures: dict[tuple[int, int], int] | None = None,
        runtime_failures: dict[tuple[int, int], int] | None = None,
    ) -> None:
        super().__init__()
        self.oom_failures = dict(oom_failures or {})
        self.runtime_failures = dict(runtime_failures or {})
        self.attempts: list[tuple[int, int]] = []

    def iter_sequence_depth(
        self,
        frame_count,
        load_frames,
        *,
        target_fps,
        input_size,
        fp32,
    ):
        del target_fps, fp32
        self.shot_lengths.append(frame_count)
        self.input_sizes.append(input_size)
        estimator = self

        class RetryableIterator:
            def __init__(self) -> None:
                self.position = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.position >= frame_count:
                    raise StopIteration
                frame = load_frames([self.position])[0]
                marker = int(frame[0, 0, 0])
                key = (input_size, marker)
                estimator.attempts.append(key)
                if estimator.oom_failures.get(key, 0):
                    estimator.oom_failures[key] -= 1
                    raise torch.cuda.OutOfMemoryError(f"OOM at {input_size} for frame {marker}")
                if estimator.runtime_failures.get(key, 0):
                    estimator.runtime_failures[key] -= 1
                    raise RuntimeError(f"failed at {input_size} for frame {marker}")
                value = np.asarray([[[marker + input_size / 1000.0]]], dtype=np.float32)
                result = self.position, DepthBatch(
                    value,
                    DepthRepresentation.INVERSE_DEPTH,
                )
                self.position += 1
                return result

        return RetryableIterator()


class EventSequenceEstimator(FakeSequenceEstimator):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def iter_sequence_depth(self, *args, **kwargs):
        self.events.append("inference")
        return super().iter_sequence_depth(*args, **kwargs)


class ClosingFailureEstimator(FakeSequenceEstimator):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    def iter_sequence_depth(self, frame_count, load_frames, **_kwargs):
        estimator = self

        class ClosingIterator:
            def __init__(self) -> None:
                self.position = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self.position:
                    raise RuntimeError("sequence failed")
                frame = load_frames([0])[0]
                self.position = 1
                value = np.asarray([[[frame[0, 0, 0]]]], dtype=np.float32)
                return 0, DepthBatch(value, DepthRepresentation.INVERSE_DEPTH)

            def close(self):
                estimator.closed = True

        return ClosingIterator()


class RecordingProgress:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def update_progress(self, stage: str, **_values) -> None:
        self.events.append(stage)


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
        "depth_resolution": "518",
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
    progress_tracker=None,
    fp32_plan: bool = False,
) -> list[Path]:
    processor = DepthMapProcessor(estimator)
    manifest = _candidate_manifest(frame_files)
    requested_size = int(str(settings["depth_resolution"]))
    if fp32_plan:
        monkeypatch.setattr(
            processor,
            "_requested_v2_execution_plan",
            lambda *_args: processor._v2_execution_plan(
                requested_size,
                requested_size,
                fp32=True,
            ),
        )
    monkeypatch.setattr(processor, "_clear_gpu_memory", lambda: None)
    monkeypatch.setattr(processor, "_load_or_analyze_scenes", lambda *_args: manifest)
    result = processor.generate_depth_map_files(
        frame_files,
        settings,
        directories,
        progress_tracker=progress_tracker,
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
    assert semantic["execution_plan"] == {
        "requested_input_size": 518,
        "effective_input_size": 518,
        "precision": "fp16",
        "fallback_policy": "v2-uniform-halving-v1",
    }
    assert estimator.fp32_values == [False, False, False]


def test_v2_execution_plan_precision_drives_the_estimator_argument(
    v2_frames: list[Path],
    v2_directories: dict[str, Path],
    v2_settings: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimator = FakeSequenceEstimator()

    _run_v2_pipeline(
        estimator,
        v2_frames,
        v2_settings,
        v2_directories,
        monkeypatch,
        fp32_plan=True,
    )

    assert estimator.fp32_values == [True, True, True]
    metadata = json.loads(
        (v2_directories["depth_raw"] / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["semantic_fingerprint"]["execution_plan"]["precision"] == "fp32"


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


def test_v2_processor_closes_sequence_state_after_an_exception(
    v2_frames: list[Path],
    v2_directories: dict[str, Path],
    v2_settings: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimator = ClosingFailureEstimator()

    with pytest.raises(RuntimeError, match="sequence failed"):
        _run_v2_pipeline(
            estimator,
            v2_frames,
            v2_settings,
            v2_directories,
            monkeypatch,
        )

    assert estimator.closed is True


def test_v2_retry_converts_unexpected_iterator_exhaustion_to_sequence_end() -> None:
    class OomThenStop:
        def __init__(self) -> None:
            self.calls = 0

        def __next__(self):
            self.calls += 1
            if self.calls == 1:
                raise torch.cuda.OutOfMemoryError("late OOM")
            raise StopIteration

    processor = DepthMapProcessor(FakeSequenceEstimator())

    item = processor._next_v2_sequence_item(OomThenStop(), 518, True)

    assert item is depth_processor_module._V2_SEQUENCE_END


def test_v2_first_window_oom_selects_one_lower_resolution_for_the_whole_stage(
    v2_frames: list[Path],
    v2_directories: dict[str, Path],
    v2_settings: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    estimator = RetryableCapacityEstimator(oom_failures={(518, 1): 1})

    _run_v2_pipeline(
        estimator,
        v2_frames,
        v2_settings,
        v2_directories,
        monkeypatch,
    )

    assert estimator.attempts[:2] == [(518, 1), (384, 1)]
    assert all(size == 384 for size, _marker in estimator.attempts[1:])
    metadata = json.loads(
        (v2_directories["depth_raw"] / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["semantic_fingerprint"]["execution_plan"]["effective_input_size"] == 384
    warning = capsys.readouterr().err
    assert "requested size 518" in warning
    assert "input size 384" in warning
    assert "whole V2 raw stage" in warning


def test_v2_late_oom_retries_the_same_window_once_without_changing_resolution(
    v2_frames: list[Path],
    v2_directories: dict[str, Path],
    v2_settings: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimator = RetryableCapacityEstimator(oom_failures={(518, 2): 1})

    _run_v2_pipeline(
        estimator,
        v2_frames,
        v2_settings,
        v2_directories,
        monkeypatch,
    )

    assert estimator.attempts.count((518, 2)) == 2
    assert {size for size, _marker in estimator.attempts} == {518}


def test_v2_repeated_late_oom_restarts_the_entire_stage_at_the_lower_resolution(
    v2_frames: list[Path],
    v2_directories: dict[str, Path],
    v2_settings: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimator = RetryableCapacityEstimator(oom_failures={(518, 2): 2})

    _run_v2_pipeline(
        estimator,
        v2_frames,
        v2_settings,
        v2_directories,
        monkeypatch,
    )

    assert estimator.attempts[:4] == [(518, 1), (518, 2), (518, 2), (384, 1)]
    with np.load(v2_directories["depth_raw"] / "frame_000000.npz") as payload:
        assert float(payload["values"][0, 0]) == pytest.approx(1.384, abs=1e-3)


def test_v2_final_resolution_candidate_oom_reports_requested_and_effective_sizes(
    v2_frames: list[Path],
    v2_directories: dict[str, Path],
    v2_settings: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v2_settings["depth_resolution"] = "384"
    estimator = RetryableCapacityEstimator(oom_failures={(384, 1): 1})

    with pytest.raises(
        RuntimeError,
        match=r"requested input size 384.*effective input size 384",
    ):
        _run_v2_pipeline(
            estimator,
            v2_frames,
            v2_settings,
            v2_directories,
            monkeypatch,
        )


def test_v2_resume_adopts_a_compatible_persisted_fallback_plan(
    v2_frames: list[Path],
    v2_directories: dict[str, Path],
    v2_settings: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = RetryableCapacityEstimator(
        oom_failures={(518, 1): 1},
        runtime_failures={(384, 2): 1},
    )
    with pytest.raises(RuntimeError, match="failed at 384 for frame 2"):
        _run_v2_pipeline(
            first,
            v2_frames,
            v2_settings,
            v2_directories,
            monkeypatch,
        )

    persisted = json.loads(
        (v2_directories["depth_raw"] / "metadata.json").read_text(encoding="utf-8")
    )
    assert persisted["semantic_fingerprint"]["execution_plan"]["effective_input_size"] == 384
    resumed = RetryableCapacityEstimator()

    _run_v2_pipeline(
        resumed,
        v2_frames,
        v2_settings,
        v2_directories,
        monkeypatch,
    )

    assert {size for size, _marker in resumed.attempts} == {384}
    assert resumed.attempts[:3] == [(384, 1), (384, 2), (384, 3)]


def test_v2_resume_rejects_a_fallback_plan_from_different_source_frames(
    v2_frames: list[Path],
    v2_directories: dict[str, Path],
    v2_settings: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = RetryableCapacityEstimator(
        oom_failures={(518, 1): 1},
        runtime_failures={(384, 2): 1},
    )
    with pytest.raises(RuntimeError, match="failed at 384 for frame 2"):
        _run_v2_pipeline(
            first,
            v2_frames,
            v2_settings,
            v2_directories,
            monkeypatch,
        )
    changed = np.full((2, 3, 3), 10, dtype=np.uint8)
    assert cv2.imwrite(str(v2_frames[0]), changed)
    resumed = RetryableCapacityEstimator()

    _run_v2_pipeline(
        resumed,
        v2_frames,
        v2_settings,
        v2_directories,
        monkeypatch,
    )

    assert resumed.attempts[0] == (518, 10)


def test_v2_nondefault_compatibility_windows_warn_once_before_cache_or_inference(
    v2_frames: list[Path],
    v2_directories: dict[str, Path],
    v2_settings: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    v2_settings["temporal_window_size"] = 64
    events: list[str] = []
    estimator = EventSequenceEstimator(events)
    restore_cache = DepthMapProcessor._try_restore_global_canonical_cache

    def record_cache_lookup(processor, *args, **kwargs):
        events.append("cache lookup")
        return restore_cache(processor, *args, **kwargs)

    monkeypatch.setattr(
        DepthMapProcessor,
        "_try_restore_global_canonical_cache",
        record_cache_lookup,
    )

    _run_v2_pipeline(
        estimator,
        v2_frames,
        v2_settings,
        v2_directories,
        monkeypatch,
        RecordingProgress(events),
    )

    warnings = [event for event in events if "fixed window 32" in event]
    assert len(warnings) == 1
    assert events.index(warnings[0]) < events.index("cache lookup")
    assert events.index(warnings[0]) < events.index("inference")
    assert capsys.readouterr().err.count("fixed window 32") == 1


def test_v2_cli_compatibility_warning_is_written_once_to_stderr(
    v2_frames: list[Path],
    v2_directories: dict[str, Path],
    v2_settings: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    v2_settings["temporal_window_overlap"] = 20

    _run_v2_pipeline(
        FakeSequenceEstimator(),
        v2_frames,
        v2_settings,
        v2_directories,
        monkeypatch,
    )

    stderr = capsys.readouterr().err
    assert stderr.count("temporal_window_size=32") == 1
    assert "temporal_window_overlap=20" in stderr
    assert "fixed window 32 and overlap 10" in stderr


def test_v2_default_compatibility_windows_do_not_warn(
    v2_settings: dict[str, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[str] = []

    DepthMapProcessor._warn_ignored_v2_temporal_settings(
        v2_settings,
        RecordingProgress(events),
    )

    assert events == []
    assert capsys.readouterr().err == ""


def test_v2_effective_resolution_changes_the_raw_and_global_model_identity(
    v2_frames: list[Path],
    v2_settings: dict[str, object],
) -> None:
    processor = DepthMapProcessor(FakeSequenceEstimator())
    requested = processor._raw_semantic_fingerprint(
        v2_frames,
        v2_settings,
        processor._v2_execution_plan(518, 518),
    )
    fallback = processor._raw_semantic_fingerprint(
        v2_frames,
        v2_settings,
        processor._v2_execution_plan(518, 384),
    )

    assert requested["execution_plan"]["effective_input_size"] == 518
    assert fallback["execution_plan"]["effective_input_size"] == 384
    requested_hash = canonical_json_hash(requested)
    fallback_hash = canonical_json_hash(fallback)
    assert requested_hash != fallback_hash
    video_path = v2_frames[0].parents[1] / "source.mp4"
    video_path.write_bytes(b"source video")
    assert compute_cache_key(
        str(video_path),
        dict(v2_settings, model_fingerprint=requested_hash),
    ) != compute_cache_key(
        str(video_path),
        dict(v2_settings, model_fingerprint=fallback_hash),
    )


def test_v2_persisted_execution_plan_requires_exact_compatible_structure() -> None:
    requested = DepthMapProcessor._v2_execution_plan(518, 518)
    fallback = DepthMapProcessor._v2_execution_plan(518, 384)

    assert DepthMapProcessor._is_compatible_v2_execution_plan(fallback, requested)
    assert not DepthMapProcessor._is_compatible_v2_execution_plan(
        {**fallback, "effective_input_size": 500},
        requested,
    )
    assert not DepthMapProcessor._is_compatible_v2_execution_plan(
        {**fallback, "fallback_policy": "unknown"},
        requested,
    )
    assert not DepthMapProcessor._is_compatible_v2_execution_plan(
        {key: value for key, value in fallback.items() if key != "precision"},
        requested,
    )


def test_v2_requested_plan_does_not_run_frame_batch_or_vram_sizing(
    v2_frames: list[Path],
    v2_settings: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    processor = DepthMapProcessor(FakeSequenceEstimator())
    monkeypatch.setattr(
        processor,
        "_determine_chunk_params",
        lambda *_args: pytest.fail("V2 requested resolution must not run batch sizing"),
    )

    plan = processor._requested_v2_execution_plan(3, 2, v2_settings)

    assert plan["requested_input_size"] == 518
    assert "frames/chunk" not in capsys.readouterr().out
