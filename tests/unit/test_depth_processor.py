"""Tests for DepthMapProcessor module."""

import cv2
import hashlib
import json
import numpy as np
import pytest
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.depth_surge_3d.core.constants import INTERMEDIATE_DIRS
from src.depth_surge_3d.inference.depth.types import (
    DepthBatch,
    DepthRepresentation,
    PinholeCameraBatch,
)
from src.depth_surge_3d.processing.frames import depth_processor
from src.depth_surge_3d.processing.frames.depth_processor import DepthMapProcessor
from src.depth_surge_3d.processing.frames.depth_storage import (
    estimate_raw_depth_only_bytes,
)
from src.depth_surge_3d.processing.frames.metric_geometry import (
    MetricGeometryStore,
    estimate_metric_geometry_disk_bytes,
    filesystem_allocation_unit,
)


class FakeMogeEstimator:
    camera_model = "pinhole_fx"
    model_path = None
    backend_id = "moge2"
    model_size = "vitb"
    max_batch_size = 2

    def __init__(self) -> None:
        self.calls = 0
        self.on_inference = None

    @staticmethod
    def get_model_info() -> dict[str, object]:
        return {
            "family": "moge",
            "model_name": "vitb",
            "source_revision": "processor-test",
            "camera_model": "pinhole_fx",
        }

    @staticmethod
    def estimate_output_shape(
        frame_width: int, frame_height: int, input_size: int
    ) -> tuple[int, int]:
        del input_size
        return frame_height, frame_width

    def estimate_depth_batch(self, frames: np.ndarray, **_kwargs) -> DepthBatch:
        self.calls += 1
        if self.on_inference is not None:
            self.on_inference()
        count, height, width = frames.shape[:3]
        depth = np.arange(1, height * width + 1, dtype=np.float32).reshape(height, width)
        values = np.repeat(depth[None, :, :], count, axis=0)
        focal = np.full(count, 0.8, dtype=np.float32)
        return DepthBatch(
            values,
            DepthRepresentation.METRIC_DEPTH,
            camera=PinholeCameraBatch(focal),
        )


@pytest.fixture
def frame_files(tmp_path: Path) -> list[Path]:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    files: list[Path] = []
    for index, value in enumerate((20, 80, 140)):
        path = frame_dir / f"frame_{index:04d}.png"
        assert cv2.imwrite(str(path), np.full((8, 6, 3), value, dtype=np.uint8))
        files.append(path)
    return files


@pytest.fixture
def fake_moge_estimator() -> FakeMogeEstimator:
    return FakeMogeEstimator()


@pytest.fixture
def default_settings() -> dict[str, object]:
    return {
        "depth_resolution": "8",
        "target_fps": 30,
        "keep_intermediates": True,
        "super_sample": "none",
        "scene_detection": True,
        "scene_cut_threshold": 0.55,
        "min_scene_frames": 1,
        "raw_storage_dtype": "auto",
    }


def make_depth_directories(tmp_path: Path) -> dict[str, Path]:
    directories = {"base": tmp_path}
    for name in ("scene_data", "depth_raw", "disparity_maps", "metric_geometry"):
        directories[name] = tmp_path / INTERMEDIATE_DIRS[name]
    return directories


def hash_directory(directory: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


@dataclass
class MetricJob:
    processor: DepthMapProcessor
    estimator: FakeMogeEstimator
    frame_files: list[Path]
    settings: dict[str, object]
    directories: dict[str, Path]
    calls_after_first_run: int = 0

    def run(self, *, keep_intermediates: bool = True) -> list[Path]:
        files = self.processor.generate_depth_map_files(
            self.frame_files,
            {**self.settings, "keep_intermediates": keep_intermediates},
            self.directories,
            None,
        )
        assert files is not None
        return files


@pytest.fixture
def fresh_metric_job(
    tmp_path: Path,
    frame_files: list[Path],
    fake_moge_estimator: FakeMogeEstimator,
    default_settings: dict[str, object],
) -> MetricJob:
    settings = {
        **default_settings,
        "depth_model_version": "moge2",
        "stereo_geometry_mode": "metric_camera",
    }
    return MetricJob(
        DepthMapProcessor(fake_moge_estimator),
        fake_moge_estimator,
        frame_files,
        settings,
        make_depth_directories(tmp_path),
    )


@pytest.fixture
def metric_job(fresh_metric_job: MetricJob) -> MetricJob:
    fresh_metric_job.run()
    fresh_metric_job.calls_after_first_run = fresh_metric_job.estimator.calls
    return fresh_metric_job


def test_metric_mode_builds_only_metric_geometry(
    tmp_path: Path,
    frame_files: list[Path],
    fake_moge_estimator: FakeMogeEstimator,
    default_settings: dict[str, object],
) -> None:
    directories = make_depth_directories(tmp_path)
    settings = {
        **default_settings,
        "depth_model_version": "moge2",
        "stereo_geometry_mode": "metric_camera",
        "keep_intermediates": True,
    }
    files = DepthMapProcessor(fake_moge_estimator).generate_depth_map_files(
        frame_files, settings, directories, progress_tracker=None
    )
    assert files is not None
    assert all(path.parent == directories["metric_geometry"] for path in files)
    assert all(path.suffix == ".npz" for path in files)
    assert not list(directories["disparity_maps"].glob("*.png"))


def test_switching_to_relative_preserves_valid_metric_stage(
    tmp_path: Path,
    frame_files: list[Path],
    fake_moge_estimator: FakeMogeEstimator,
    default_settings: dict[str, object],
) -> None:
    directories = make_depth_directories(tmp_path)
    processor = DepthMapProcessor(fake_moge_estimator)
    metric_settings = {
        **default_settings,
        "depth_model_version": "moge2",
        "stereo_geometry_mode": "metric_camera",
        "keep_intermediates": True,
    }
    metric_files = processor.generate_depth_map_files(
        frame_files, metric_settings, directories, None
    )
    assert metric_files is not None
    metric_bytes = [path.read_bytes() for path in metric_files]
    relative_files = processor.generate_depth_map_files(
        frame_files,
        {**metric_settings, "stereo_geometry_mode": "relative"},
        directories,
        None,
    )
    assert relative_files is not None
    assert all(path.parent == directories["disparity_maps"] for path in relative_files)
    assert [path.read_bytes() for path in metric_files] == metric_bytes


def test_metric_preflight_sums_raw_and_exact_selected_stage_before_inference(
    tmp_path: Path,
    frame_files: list[Path],
    fake_moge_estimator: FakeMogeEstimator,
    default_settings: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, int | None]] = []
    monkeypatch.setattr(
        depth_processor,
        "require_disk_space",
        lambda _path, required: events.append(("preflight", required)),
    )
    fake_moge_estimator.on_inference = lambda: events.append(("inference", None))
    settings = {
        **default_settings,
        "depth_model_version": "moge2",
        "stereo_geometry_mode": "metric_camera",
        "keep_intermediates": True,
    }
    DepthMapProcessor(fake_moge_estimator).generate_depth_map_files(
        frame_files, settings, make_depth_directories(tmp_path), None
    )
    native_shape = fake_moge_estimator.estimate_output_shape(6, 8, 8)
    expected = estimate_raw_depth_only_bytes(
        frame_count=len(frame_files),
        native_width=native_shape[1],
        native_height=native_shape[0],
        storage_bytes=4,
        camera_bytes_per_frame=4,
    ) + estimate_metric_geometry_disk_bytes(
        [native_shape] * len(frame_files),
        allocation_unit=filesystem_allocation_unit(tmp_path),
        include_visual_previews=True,
    )
    assert events[0] == ("preflight", expected)
    assert next(index for index, event in enumerate(events) if event[0] == "inference") > 0


@pytest.mark.parametrize(
    "change",
    [
        {"virtual_baseline_mm": 70.0},
        {"metric_convergence_distance": 3.0},
        {"max_disparity_percent": 1.0},
        {"crop_factor": 0.8},
        {"per_eye_width": 1920},
    ],
)
def test_metric_render_setting_changes_preserve_stage3_hashes(
    metric_job: MetricJob, change: dict[str, object]
) -> None:
    before = hash_directory(metric_job.directories["metric_geometry"])
    files = metric_job.processor.generate_depth_map_files(
        metric_job.frame_files,
        {**metric_job.settings, **change},
        metric_job.directories,
        None,
    )
    assert files
    assert hash_directory(metric_job.directories["metric_geometry"]) == before
    assert metric_job.estimator.calls == metric_job.calls_after_first_run


def test_complete_metric_stage_reuse_makes_zero_estimator_calls(metric_job: MetricJob) -> None:
    metric_job.estimator.estimate_depth_batch = Mock(
        side_effect=AssertionError("inference must not run")
    )
    assert metric_job.processor.generate_depth_map_files(
        metric_job.frame_files,
        metric_job.settings,
        metric_job.directories,
        None,
    )


def test_metric_preview_repair_preflights_before_writing(
    metric_job: MetricJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview = next(metric_job.directories["metric_geometry"].glob("*.png"))
    preview.unlink()
    events: list[tuple[str, int | None]] = []
    real_write = DepthMapProcessor._atomic_write_png

    monkeypatch.setattr(
        depth_processor,
        "require_metric_geometry_disk_space",
        lambda _path, required: events.append(("preflight", required)),
    )

    def recording_write(path: Path, values: np.ndarray) -> None:
        events.append(("write", None))
        real_write(path, values)

    monkeypatch.setattr(DepthMapProcessor, "_atomic_write_png", staticmethod(recording_write))

    assert metric_job.processor.generate_depth_map_files(
        metric_job.frame_files,
        metric_job.settings,
        metric_job.directories,
        None,
    )

    shape = (8, 6)
    allocation = filesystem_allocation_unit(metric_job.directories["metric_geometry"])
    expected = estimate_metric_geometry_disk_bytes(
        [shape], allocation_unit=allocation, include_visual_previews=True
    ) - estimate_metric_geometry_disk_bytes([shape], allocation_unit=allocation)
    assert events == [("preflight", expected), ("write", None)]
    assert preview.is_file()


def test_no_retention_removes_raw_only_after_metric_finalize(
    fresh_metric_job: MetricJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    real_finalize = MetricGeometryStore.finalize
    real_remove = fresh_metric_job.processor._remove_raw_payloads

    def recording_finalize(store, convergence):
        result = real_finalize(store, convergence)
        events.append("finalized")
        return result

    def recording_remove(store):
        events.append("remove_raw")
        return real_remove(store)

    monkeypatch.setattr(MetricGeometryStore, "finalize", recording_finalize)
    monkeypatch.setattr(fresh_metric_job.processor, "_remove_raw_payloads", recording_remove)
    fresh_metric_job.run(keep_intermediates=False)
    assert events == ["finalized", "remove_raw"]


def test_no_retention_reuses_complete_metric_stage_without_raw_payloads(
    fresh_metric_job: MetricJob,
) -> None:
    metric_files = fresh_metric_job.run(keep_intermediates=False)
    assert not list(fresh_metric_job.directories["depth_raw"].glob("*.npz"))
    assert (fresh_metric_job.directories["depth_raw"] / "metadata.json").is_file()
    calls = fresh_metric_job.estimator.calls
    fresh_metric_job.estimator.estimate_depth_batch = Mock(
        side_effect=AssertionError("validated metric stage must not require raw payloads")
    )

    assert fresh_metric_job.run(keep_intermediates=False) == metric_files
    assert fresh_metric_job.estimator.calls == calls


def test_candidate_scene_assignment_change_rebuilds_metric_without_inference(
    metric_job: MetricJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_analyze = depth_processor.analyze_scenes

    def changed_analyze(*args, **kwargs):
        manifest = original_analyze(*args, **kwargs)
        manifest["scene_ids"] = [0, 1, 1]
        manifest["candidate_cuts"] = [1]
        return manifest

    monkeypatch.setattr(depth_processor, "analyze_scenes", changed_analyze)
    calls = metric_job.estimator.calls
    before = hash_directory(metric_job.directories["metric_geometry"])
    files = metric_job.processor.generate_depth_map_files(
        metric_job.frame_files,
        {**metric_job.settings, "scene_cut_threshold": 0.1},
        metric_job.directories,
        None,
    )
    assert files
    assert metric_job.estimator.calls == calls
    assert hash_directory(metric_job.directories["metric_geometry"]) != before


def test_metric_derivation_persists_the_visual_depth_previews(
    fresh_metric_job: MetricJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = SimpleNamespace(
        update_progress=Mock(),
        send_preview_frame_from_array=Mock(),
    )
    monkeypatch.setattr(depth_processor, "PREVIEW_FRAME_SAMPLE_RATE", 1)

    files = fresh_metric_job.processor.generate_depth_map_files(
        fresh_metric_job.frame_files,
        fresh_metric_job.settings,
        fresh_metric_job.directories,
        tracker,
    )

    assert files
    preview_calls = tracker.send_preview_frame_from_array.call_args_list
    assert [call.args[1:] for call in preview_calls] == [
        ("depth_map", 1),
        ("depth_map", 2),
        ("depth_map", 3),
    ]
    for call in preview_calls:
        assert call.args[0].dtype == np.uint8

    preview_files = sorted(fresh_metric_job.directories["metric_geometry"].glob("*.png"))
    assert [path.stem for path in preview_files] == [path.stem for path in files]
    for metric_file, preview_file in zip(files, preview_files):
        with np.load(metric_file) as payload:
            expected = DepthMapProcessor._metric_depth_preview(
                payload["inverse_depth"], payload["valid"]
            )
        actual = cv2.imread(str(preview_file), cv2.IMREAD_UNCHANGED)
        assert actual is not None
        assert actual.dtype == np.uint8
        np.testing.assert_array_equal(actual, expected)


def test_metric_stage_resume_keeps_committed_payload_bytes(
    fresh_metric_job: MetricJob, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_write = MetricGeometryStore.write_frame
    writes = 0

    def interrupt_second_write(store, name, frame):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise RuntimeError("interrupted")
        return real_write(store, name, frame)

    monkeypatch.setattr(MetricGeometryStore, "write_frame", interrupt_second_write)
    with pytest.raises(RuntimeError, match="interrupted"):
        fresh_metric_job.run()
    committed = next(fresh_metric_job.directories["metric_geometry"].glob("*.npz"))
    committed_bytes = committed.read_bytes()
    monkeypatch.setattr(MetricGeometryStore, "write_frame", real_write)

    files = fresh_metric_job.run()

    assert len(files) == len(fresh_metric_job.frame_files)
    assert committed.read_bytes() == committed_bytes


def test_clean_chunked_and_resumed_metric_stages_are_byte_identical(
    tmp_path: Path,
    frame_files: list[Path],
    default_settings: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = {
        **default_settings,
        "depth_model_version": "moge2",
        "stereo_geometry_mode": "metric_camera",
    }

    def build(label: str, chunk_size: int) -> tuple[DepthMapProcessor, dict[str, Path]]:
        processor = DepthMapProcessor(FakeMogeEstimator())
        directories = make_depth_directories(tmp_path / label)
        with patch.object(processor, "_determine_chunk_params", return_value=(chunk_size, 8)):
            files = processor.generate_depth_map_files(frame_files, settings, directories, None)
        assert files
        return processor, directories

    _clean_processor, clean_directories = build("clean", len(frame_files))
    _chunked_processor, chunked_directories = build("chunked", 1)

    resumed_processor = DepthMapProcessor(FakeMogeEstimator())
    resumed_directories = make_depth_directories(tmp_path / "resumed")
    real_write = MetricGeometryStore.write_frame
    writes = 0

    def interrupt_second_write(store, name, frame):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise RuntimeError("interrupted")
        return real_write(store, name, frame)

    monkeypatch.setattr(MetricGeometryStore, "write_frame", interrupt_second_write)
    with (
        patch.object(resumed_processor, "_determine_chunk_params", return_value=(1, 8)),
        pytest.raises(RuntimeError, match="interrupted"),
    ):
        resumed_processor.generate_depth_map_files(frame_files, settings, resumed_directories, None)
    monkeypatch.setattr(MetricGeometryStore, "write_frame", real_write)
    with patch.object(resumed_processor, "_determine_chunk_params", return_value=(1, 8)):
        assert resumed_processor.generate_depth_map_files(
            frame_files, settings, resumed_directories, None
        )

    expected = hash_directory(clean_directories["metric_geometry"])
    assert hash_directory(chunked_directories["metric_geometry"]) == expected
    assert hash_directory(resumed_directories["metric_geometry"]) == expected


def test_relative_cache_hit_clears_incompatible_raw_and_both_derived_stages(
    fresh_metric_job: MetricJob,
    tmp_path: Path,
) -> None:
    fresh_metric_job.run()
    directories = fresh_metric_job.directories
    stale_canonical = directories["disparity_maps"] / "stale.png"
    stale_canonical.write_bytes(b"stale")
    restored = tmp_path / "restored.png"
    restored.write_bytes(b"restored")

    def lookup_after_classification(*_args):
        assert not list(directories["depth_raw"].glob("*.npz"))
        assert not list(directories["disparity_maps"].iterdir())
        assert not list(directories["metric_geometry"].iterdir())
        return [restored]

    with (
        patch.object(
            depth_processor,
            "get_cached_depth_map_files",
            side_effect=lookup_after_classification,
        ),
        patch.object(
            fresh_metric_job.processor,
            "_restore_cached_canonical_stage",
            return_value=[restored],
        ),
    ):
        files = fresh_metric_job.processor.generate_depth_map_files(
            fresh_metric_job.frame_files,
            {
                **fresh_metric_job.settings,
                "stereo_geometry_mode": "relative",
                "depth_resolution": "7",
                "video_path": "source.mp4",
            },
            directories,
            None,
        )

    assert files == [restored]


def test_relative_cache_hit_removes_compatible_raw_after_validation(
    fresh_metric_job: MetricJob,
    tmp_path: Path,
) -> None:
    settings = {
        **fresh_metric_job.settings,
        "stereo_geometry_mode": "relative",
        "keep_intermediates": True,
    }
    files = fresh_metric_job.processor.generate_depth_map_files(
        fresh_metric_job.frame_files,
        settings,
        fresh_metric_job.directories,
        None,
    )
    assert files
    raw_dir = fresh_metric_job.directories["depth_raw"]
    assert list(raw_dir.glob("*.npz"))
    restored = tmp_path / "restored.png"
    restored.write_bytes(b"restored")

    with (
        patch.object(depth_processor, "get_cached_depth_map_files", return_value=[restored]),
        patch.object(
            fresh_metric_job.processor,
            "_restore_cached_canonical_stage",
            return_value=[restored],
        ),
    ):
        result = fresh_metric_job.processor.generate_depth_map_files(
            fresh_metric_job.frame_files,
            {**settings, "keep_intermediates": False, "video_path": "source.mp4"},
            fresh_metric_job.directories,
            None,
        )

    assert result == [restored]
    assert not list(raw_dir.glob("*.npz"))
    metadata = json.loads((raw_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["storage_status"] == "ready"
    assert metadata["completed_count"] == 0


def test_relative_cache_validation_failure_preserves_compatible_raw_payloads(
    fresh_metric_job: MetricJob,
) -> None:
    settings = {
        **fresh_metric_job.settings,
        "stereo_geometry_mode": "relative",
        "keep_intermediates": True,
    }
    assert fresh_metric_job.processor.generate_depth_map_files(
        fresh_metric_job.frame_files,
        settings,
        fresh_metric_job.directories,
        None,
    )
    raw_dir = fresh_metric_job.directories["depth_raw"]
    before = hash_directory(raw_dir)

    with (
        patch.object(depth_processor, "get_cached_depth_map_files", return_value=[raw_dir]),
        patch.object(
            fresh_metric_job.processor,
            "_restore_cached_canonical_stage",
            side_effect=OSError("restored canonical validation failed"),
        ),
        pytest.raises(OSError, match="validation failed"),
    ):
        fresh_metric_job.processor.generate_depth_map_files(
            fresh_metric_job.frame_files,
            {**settings, "keep_intermediates": False, "video_path": "source.mp4"},
            fresh_metric_job.directories,
            None,
        )

    assert hash_directory(raw_dir) == before


class TestDepthMapProcessorInit:
    """Test DepthMapProcessor initialization."""

    def test_init_with_estimator(self):
        """Test initialization with depth estimator."""
        estimator = Mock()
        processor = DepthMapProcessor(estimator, verbose=False)

        assert processor.depth_estimator == estimator
        assert processor.verbose is False

    def test_init_with_verbose(self):
        """Test initialization with verbose enabled."""
        estimator = Mock()
        processor = DepthMapProcessor(estimator, verbose=True)

        assert processor.verbose is True

    def test_unbounded_array_api_is_removed(self):
        assert "generate_depth_maps" not in vars(DepthMapProcessor)


class TestGenerateDepthMaps:
    """Test generate_depth_maps main entry point."""

    @pytest.fixture
    def mock_estimator(self):
        """Create mock depth estimator."""
        estimator = Mock()
        estimator.estimate_depth_batch = Mock(return_value=np.random.rand(3, 100, 100))
        return estimator

    @pytest.fixture
    def mock_progress_tracker(self):
        """Create mock progress tracker."""
        tracker = Mock()
        tracker.update_progress = Mock()
        return tracker

    @pytest.fixture
    def temp_frames(self, tmp_path):
        """Create temporary frame files."""
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()

        frame_files = []
        for i in range(3):
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            frame_path = frame_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_files.append(frame_path)

        return frame_files

    def test_generate_depth_map_files_writes_each_chunk_without_stacking(
        self, mock_progress_tracker, temp_frames, tmp_path
    ):
        """Long videos retain only one inference chunk in memory."""
        estimator = Mock()
        estimator.camera_model = "none"
        estimator.model_path = None
        estimator.get_model_info.return_value = {
            "family": "processor-test",
            "revision": "1",
        }
        estimator.estimate_depth_batch.side_effect = lambda frames, **kwargs: DepthBatch(
            np.full((len(frames), 2, 3), 0.5, dtype=np.float32),
            DepthRepresentation.INVERSE_DEPTH,
        )
        processor = DepthMapProcessor(estimator)
        depth_dir = tmp_path / "03_disparity_maps"
        depth_dir.mkdir()
        settings = {
            "depth_resolution": "1080",
            "target_fps": 30,
            "keep_intermediates": False,
            "super_sample": "none",
            "per_eye_width": 100,
            "per_eye_height": 100,
        }

        with (
            patch.object(processor, "_determine_chunk_params", return_value=(2, 1080)),
            patch.object(processor, "_clear_gpu_memory") as clear_gpu_memory,
        ):
            result = processor.generate_depth_map_files(
                temp_frames,
                settings,
                {"base": tmp_path, "disparity_maps": depth_dir},
                mock_progress_tracker,
            )

        assert result == [depth_dir / f"frame_{i:04d}.png" for i in range(3)]
        assert all(path.exists() for path in result)
        assert all(
            cv2.imread(str(path), cv2.IMREAD_UNCHANGED).dtype == np.uint16 for path in result
        )
        assert estimator.estimate_depth_batch.call_count == 2
        assert clear_gpu_memory.call_count == 1

    def test_moge_chunk_persists_focal_with_depth_and_keeps_canonical_output(
        self, mock_progress_tracker, temp_frames, tmp_path
    ):
        estimator = Mock()
        estimator.camera_model = "pinhole_fx"
        estimator.model_path = None
        estimator.get_model_info.return_value = {
            "family": "moge",
            "repository": "Ruicheng/moge-2-vitl-normal",
            "source_revision": "immutable",
            "resolution_level": 9,
            "preprocessing_algorithm": "rgb-area-max-edge-v1",
            "camera_model": "pinhole_fx",
        }
        estimator.estimate_depth_batch.side_effect = lambda frames, **kwargs: DepthBatch(
            np.full((len(frames), 2, 3), 2.0, dtype=np.float32),
            DepthRepresentation.METRIC_DEPTH,
            camera=PinholeCameraBatch(
                np.full(len(frames), 0.8, dtype=np.float32),
            ),
        )
        processor = DepthMapProcessor(estimator)
        canonical_dir = tmp_path / "03_disparity_maps"
        settings = {
            "depth_model_version": "moge2",
            "depth_resolution": "1080",
            "target_fps": 30,
            "keep_intermediates": True,
            "super_sample": "none",
            "per_eye_width": 100,
            "per_eye_height": 100,
        }

        with (
            patch.object(processor, "_determine_chunk_params", return_value=(2, 1080)),
            patch.object(processor, "_clear_gpu_memory"),
        ):
            result = processor.generate_depth_map_files(
                temp_frames,
                settings,
                {"base": tmp_path, "disparity_maps": canonical_dir},
                mock_progress_tracker,
            )

        metadata = json.loads((tmp_path / "02_depth_raw" / "metadata.json").read_text())
        assert metadata["schema_version"] == 3
        assert metadata["camera_model"] == "pinhole_fx"
        assert metadata["semantic_fingerprint"]["camera_model"] == "pinhole_fx"
        assert metadata["semantic_fingerprint"]["preprocessing_algorithm"] == (
            "moge2-rgb-area-max-edge-v1"
        )
        with zipfile.ZipFile(tmp_path / "02_depth_raw" / "frame_0000.npz") as payload:
            assert payload.namelist() == ["values.npy", "focal_x_normalized.npy"]
        assert result is not None
        assert all(path.is_file() for path in result)


def test_native_shape_estimate_uses_estimator_output_contract():
    class SquareEstimator:
        @staticmethod
        def estimate_output_shape(
            frame_width: int, frame_height: int, input_size: int
        ) -> tuple[int, int]:
            del frame_width, frame_height
            return input_size, input_size

    processor = DepthMapProcessor(SquareEstimator())

    assert processor._estimate_native_shape(1920, 1080, 768) == (768, 768)


class TestDetermineChunkParams:
    """Test chunk parameter determination."""

    @pytest.fixture
    def mock_estimator(self):
        """Create mock depth estimator with model info."""
        estimator = Mock()
        estimator.model_type = "v3"
        estimator.get_model_size = Mock(return_value="large")
        return estimator

    def test_determine_chunk_params_auto_resolution(self, mock_estimator):
        """Test auto resolution detection."""
        processor = DepthMapProcessor(mock_estimator, verbose=False)

        with patch(
            "src.depth_surge_3d.processing.frames.depth_processor.get_vram_info",
            return_value={"total": 8.0, "available": 6.0},
        ):
            with patch(
                "src.depth_surge_3d.processing.frames.depth_processor.calculate_optimal_chunk_size",
                return_value=4,
            ):
                chunk_size, input_size = processor._determine_chunk_params(1920, 1080, "auto")

        assert chunk_size == 4
        assert input_size == 1080  # Auto selected based on resolution

    def test_determine_chunk_params_manual_resolution(self, mock_estimator):
        """Test manual resolution setting."""
        processor = DepthMapProcessor(mock_estimator, verbose=False)

        with patch(
            "src.depth_surge_3d.processing.frames.depth_processor.get_vram_info",
            return_value={"total": 8.0, "available": 6.0},
        ):
            with patch(
                "src.depth_surge_3d.processing.frames.depth_processor.calculate_optimal_chunk_size",
                return_value=2,
            ):
                chunk_size, input_size = processor._determine_chunk_params(1920, 1080, "720")

        assert chunk_size == 2
        assert input_size == 720

    def test_determine_chunk_params_invalid_manual(self, mock_estimator):
        """Test invalid manual resolution falls back to auto."""
        processor = DepthMapProcessor(mock_estimator, verbose=False)

        with patch(
            "src.depth_surge_3d.processing.frames.depth_processor.get_vram_info",
            return_value={"total": 8.0, "available": 6.0},
        ):
            with patch(
                "src.depth_surge_3d.processing.frames.depth_processor.calculate_optimal_chunk_size",
                return_value=4,
            ):
                chunk_size, input_size = processor._determine_chunk_params(1920, 1080, "invalid")

        assert input_size == 1080  # Fell back to auto

    def test_determine_chunk_params_cpu_mode(self, mock_estimator):
        """Test CPU mode without VRAM."""
        processor = DepthMapProcessor(mock_estimator, verbose=False)

        with patch(
            "src.depth_surge_3d.processing.frames.depth_processor.get_vram_info",
            return_value={"total": 0, "available": 0},
        ):
            chunk_size, input_size = processor._determine_chunk_params(1920, 1080, "auto")

        assert chunk_size in [4, 6, 8, 12, 16, 24, 32]  # Fixed size for CPU (actual constants)
        assert input_size == 1080

    def test_moge_uses_framewise_base_profile_and_single_frame_cap(self):
        """MoGe metadata selects its framewise profile before enforcing its batch cap."""

        class FakeMoGeEstimator:
            backend_id = "moge2"
            model_size = "vitb"
            max_batch_size = 1

            @staticmethod
            def get_model_info() -> dict[str, str]:
                return {"family": "moge", "model_name": "vitb"}

        processor = DepthMapProcessor(FakeMoGeEstimator(), verbose=False)
        with (
            patch(
                "src.depth_surge_3d.processing.frames.depth_processor.get_vram_info",
                return_value={"total": 8.0, "available": 6.0},
            ),
            patch(
                "src.depth_surge_3d.processing.frames.depth_processor.calculate_optimal_chunk_size",
                return_value=8,
            ) as calculate_chunk_size,
        ):
            chunk_size, input_size = processor._determine_chunk_params(1920, 1080, "auto")

        assert (chunk_size, input_size) == (1, 1080)
        calculate_chunk_size.assert_called_once_with(1920, 1080, 1080, "v3", "base")


class TestAutoDetermineInputSize:
    """Test automatic input size determination."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return DepthMapProcessor(Mock(), verbose=False)

    def test_4k_resolution(self, processor):
        """Test 4K resolution input sizing."""
        input_size = processor._auto_determine_input_size(3840, 2160, 8.3)
        assert input_size == 2160  # Should cap at source resolution

    def test_1080p_resolution(self, processor):
        """Test 1080p resolution input sizing."""
        input_size = processor._auto_determine_input_size(1920, 1080, 2.1)
        assert input_size == 1080

    def test_720p_resolution(self, processor):
        """Test 720p resolution input sizing."""
        input_size = processor._auto_determine_input_size(1280, 720, 0.9)
        assert input_size == 640  # Falls to SD since 0.9 is not > MEGAPIXELS_720P (1.0)

    def test_sd_resolution(self, processor):
        """Test SD resolution input sizing."""
        input_size = processor._auto_determine_input_size(640, 480, 0.3)
        assert input_size == 640


class TestGetChunkSizeForResolution:
    """Test chunk size selection based on resolution."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return DepthMapProcessor(Mock(), verbose=False)

    def test_4k_chunk_size(self, processor):
        """Test 4K chunk size."""
        chunk_size = processor._get_chunk_size_for_resolution(2160)
        assert chunk_size == 4  # CHUNK_SIZE_4K

    def test_1440p_chunk_size(self, processor):
        """Test 1440p chunk size."""
        chunk_size = processor._get_chunk_size_for_resolution(1440)
        assert chunk_size == 6  # CHUNK_SIZE_1440P

    def test_1080p_chunk_size(self, processor):
        """Test 1080p chunk size."""
        chunk_size = processor._get_chunk_size_for_resolution(1080)
        assert chunk_size == 12  # CHUNK_SIZE_1080P_MANUAL

    def test_720p_chunk_size(self, processor):
        """Test 720p chunk size."""
        chunk_size = processor._get_chunk_size_for_resolution(720)
        assert chunk_size == 16  # CHUNK_SIZE_720P

    def test_small_chunk_size(self, processor):
        """Test small resolution chunk size."""
        chunk_size = processor._get_chunk_size_for_resolution(480)
        assert chunk_size == 32  # CHUNK_SIZE_SMALL


class TestClearGPUMemory:
    """Test GPU memory clearing."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return DepthMapProcessor(Mock(), verbose=False)

    def test_clear_gpu_memory_cuda_available(self, processor):
        """Test clearing GPU memory when CUDA is available."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.empty_cache") as mock_empty:
                with patch("torch.cuda.synchronize") as mock_sync:
                    with patch("torch.cuda.mem_get_info", return_value=(4 * 1024**3, 8 * 1024**3)):
                        processor._clear_gpu_memory()

        mock_empty.assert_called_once()
        mock_sync.assert_called_once()

    def test_clear_gpu_memory_cpu_mode(self, processor):
        """Test clearing GPU memory in CPU mode (no-op)."""
        with patch("torch.cuda.is_available", return_value=False):
            processor._clear_gpu_memory()
        # Should not raise any errors


class TestLoadChunkFrames:
    """Test loading chunk of frames."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return DepthMapProcessor(Mock(), verbose=False)

    @pytest.fixture
    def temp_frames(self, tmp_path):
        """Create temporary frame files."""
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()

        frame_files = []
        for i in range(3):
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            frame_path = frame_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_files.append(frame_path)

        return frame_files

    def test_load_chunk_frames_success(self, processor, temp_frames):
        """Test successful frame loading."""
        settings = {"super_sample": "none", "per_eye_width": 100, "per_eye_height": 100}

        result = processor._load_chunk_frames(temp_frames, settings)

        assert result is not None
        assert len(result) == 3
        assert all(isinstance(frame, np.ndarray) for frame in result)

    def test_load_chunk_frames_never_applies_super_sampling(self, processor, temp_frames):
        """Depth inference always receives the persisted source-frame resolution."""
        settings = {"super_sample": "2x", "per_eye_width": 200, "per_eye_height": 200}

        result = processor._load_chunk_frames(temp_frames, settings)

        assert result is not None
        assert len(result) == 3
        assert all(frame.shape == (100, 100, 3) for frame in result)

    def test_load_chunk_frames_missing_file(self, processor, temp_frames):
        """Test loading with missing file."""
        settings = {"super_sample": "none", "per_eye_width": 100, "per_eye_height": 100}

        # Add non-existent file
        bad_path = Path("/nonexistent/frame.png")
        chunk_files = temp_frames + [bad_path]

        result = processor._load_chunk_frames(chunk_files, settings)

        # Should still load valid frames
        assert result is not None
        assert len(result) == 3

    def test_load_chunk_frames_all_missing(self, processor):
        """Test loading with all files missing."""
        settings = {"super_sample": "none", "per_eye_width": 100, "per_eye_height": 100}
        chunk_files = [Path("/nonexistent/frame1.png"), Path("/nonexistent/frame2.png")]

        result = processor._load_chunk_frames(chunk_files, settings)

        assert result is None
