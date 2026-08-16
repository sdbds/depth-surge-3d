"""Production-shaped MoGe-2 pipeline evidence without optional model downloads."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from depth_surge_3d.core.constants import DEFAULT_SETTINGS  # noqa: E402
from depth_surge_3d.inference.depth.backend_registry import (  # noqa: E402
    BackendAvailability,
)
from depth_surge_3d.inference.depth.types import (  # noqa: E402
    DepthBatch,
    DepthRepresentation,
    PinholeCameraBatch,
)
from depth_surge_3d.io.operations import generate_output_filename  # noqa: E402
from depth_surge_3d.io.resume import build_resume_report  # noqa: E402
from depth_surge_3d.processing.frames import depth_processor  # noqa: E402
from depth_surge_3d.processing.frames.metric_geometry import (  # noqa: E402
    MetricGeometryStore,
)
from depth_surge_3d.processing.frames.source_frame_manifest import (  # noqa: E402
    write_source_frame_manifest,
)
from depth_surge_3d.processing.orchestration.video_processor import (  # noqa: E402
    VideoProcessor,
)
from depth_surge_3d.processing.video.video_encoder import VideoEncoder  # noqa: E402
from depth_surge_3d.rendering.stereo_renderer import (  # noqa: E402
    SPLAT_BYTES_PER_PIXEL,
    StereoRenderSettings,
    StereoRenderer,
)
from depth_surge_3d.utils.imaging.png_header import read_png_header  # noqa: E402


pytestmark = pytest.mark.integration

_FRAME_COUNT = 3
_FRAME_SHAPE = (6, 8)
_RAW_MEMBERS = ["values.npy", "focal_x_normalized.npy"]
_METRIC_MEMBERS = ["inverse_depth.npy", "valid.npy", "focal_x_normalized.npy"]


class _FakeMoGeEstimator:
    max_batch_size = 1
    metric = True
    camera_model = "pinhole_fx"
    model_size = "vitb"
    repo_id = "Ruicheng/moge-2-vitb-normal"
    revision = "54ad3a693e61907ea4633d13dec6ee682fa09419"
    resolution_level = 9
    inference_precision = "float32"
    processing_resolution = 8
    device = "cpu"

    def __init__(self) -> None:
        self.calls = 0

    def estimate_output_shape(self, height: int, width: int, _input_size: int) -> tuple[int, int]:
        return height, width

    def estimate_depth_batch(self, frames, **_kwargs) -> DepthBatch:
        self.calls += len(frames)
        height, width = frames.shape[1:3]
        x = np.linspace(0.75, 4.0, width, dtype=np.float32)
        values = np.broadcast_to(x, (len(frames), height, width)).copy()
        values[:, 0, 0] = np.nan
        focal = np.full((len(frames),), 0.5, dtype=np.float32)
        return DepthBatch(
            values,
            DepthRepresentation.METRIC_DEPTH,
            PinholeCameraBatch(focal),
        )

    def get_model_size(self) -> str:
        return self.model_size

    def get_model_info(self) -> dict[str, object]:
        return {"model_size": self.model_size, "repo_id": self.repo_id}

    def load_model(self) -> bool:
        return True

    def unload_model(self) -> None:
        return None


@dataclass(frozen=True)
class _RecordedEvent:
    sequence: int
    kind: str
    path: Path
    png_bytes: bytes | None


class _RecordingProgressTracker:
    def __init__(self) -> None:
        self.events: list[_RecordedEvent] = []

    def _record(self, kind: str, path: Path, png_bytes: bytes | None) -> None:
        self.events.append(_RecordedEvent(len(self.events), kind, path, png_bytes))

    def record_metric_complete(self, metadata_path: Path) -> None:
        self._record("metric_complete", metadata_path, None)

    def update_progress(self, *_args: object, **_kwargs: object) -> None:
        return None

    def send_preview_frame(self, path: Path, kind: str, _frame_num: int) -> None:
        committed = Path(path)
        self._record(kind, committed, committed.read_bytes())

    def finish(self, _message: str) -> None:
        return None


@dataclass(frozen=True)
class _PipelineResult:
    success: bool
    output_dir: Path
    source_video: Path
    estimator: _FakeMoGeEstimator
    settings: dict[str, Any]
    settings_file: Path | None
    progress: _RecordingProgressTracker

    @property
    def final_video(self) -> Path:
        return self.output_dir / generate_output_filename(
            self.source_video.name,
            self.settings["vr_format"],
            self.settings["vr_resolution"],
        )


class _PipelineHarness:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.source_video = root / "clip.mp4"
        self.source_video.write_bytes(b"deterministic fake video boundary")
        self._active_tracker: _RecordingProgressTracker | None = None
        self._source_colors = ((20, 40, 220), (30, 210, 70), (220, 60, 30))

        monkeypatch.setattr(
            VideoEncoder,
            "extract_frames",
            lambda _encoder, video_path, directories, properties, settings: self._extract_frames(
                video_path,
                directories,
                properties,
                settings,
            ),
        )
        monkeypatch.setattr(
            depth_processor,
            "build_current_model_fingerprint",
            lambda _estimator, settings: {
                "backend": "fake-moge2-integration",
                "model_info": {
                    "revision": _FakeMoGeEstimator.revision,
                    "camera_model": _FakeMoGeEstimator.camera_model,
                },
                "depth_settings": {
                    "depth_model_version": settings.get("depth_model_version"),
                    "model_size": settings.get("model_size"),
                    "depth_resolution": settings.get("depth_resolution"),
                    "use_metric_depth": settings.get("use_metric_depth"),
                    "device": settings.get("device"),
                },
                "weight_sha256": None,
                "artifact_identity": _FakeMoGeEstimator.revision,
            },
        )
        monkeypatch.setattr(
            VideoEncoder,
            "create_video",
            lambda _encoder, vr_frames, output, video, settings: self._create_video(
                vr_frames,
                output,
                video,
                settings,
            ),
        )
        real_finalize = MetricGeometryStore.finalize

        def observed_finalize(store, convergence):
            metadata = real_finalize(store, convergence)
            if self._active_tracker is not None:
                self._active_tracker.record_metric_complete(store.metadata_path)
            return metadata

        monkeypatch.setattr(MetricGeometryStore, "finalize", observed_finalize)

    def patch_package_namespace(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        """Apply the same media/fingerprint boundaries to the product import namespace."""

        encoder_module = importlib.import_module("depth_surge_3d.processing.video.video_encoder")
        processor_module = importlib.import_module(
            "depth_surge_3d.processing.frames.depth_processor"
        )
        monkeypatch.setattr(
            encoder_module.VideoEncoder,
            "extract_frames",
            lambda _encoder, video_path, directories, properties, settings: self._extract_frames(
                video_path,
                directories,
                properties,
                settings,
            ),
        )
        monkeypatch.setattr(
            encoder_module.VideoEncoder,
            "create_video",
            lambda _encoder, vr_frames, output, video, settings: self._create_video(
                vr_frames,
                output,
                video,
                settings,
            ),
        )
        monkeypatch.setattr(
            processor_module,
            "build_current_model_fingerprint",
            depth_processor.build_current_model_fingerprint,
        )
        return importlib.import_module("depth_surge_3d.rendering.stereo_projector")

    def settings(self, geometry_mode: str, **overrides: object) -> dict[str, Any]:
        values = dict(DEFAULT_SETTINGS)
        values.update(
            {
                "depth_model_version": "moge2",
                "model_size": "vitb",
                "model_path": None,
                "depth_resolution": 8,
                "use_metric_depth": True,
                "device": "cpu",
                "stereo_geometry_mode": geometry_mode,
                "scene_detection": False,
                "min_scene_frames": 1,
                "raw_storage_dtype": "float32",
                "stereo_io_workers": 1,
                "apply_distortion": False,
                "crop_factor": 1.0,
                "preserve_audio": False,
                "target_fps": "original",
                "keep_intermediates": True,
                "vr_format": "side_by_side",
                "vr_resolution": "16x9-480p",
                "per_eye_width": 854,
                "per_eye_height": 480,
                "vr_output_width": 1708,
                "vr_output_height": 480,
                "source_width": 8,
                "source_height": 6,
                "source_fps": 24.0,
            }
        )
        values.update(overrides)
        return values

    def run(
        self,
        geometry_mode: str,
        *,
        output_dir: Path | None = None,
        estimator: _FakeMoGeEstimator | None = None,
        progress: _RecordingProgressTracker | None = None,
        fail_stereo: bool = False,
        **overrides: object,
    ) -> _PipelineResult:
        output_dir = output_dir or self.root / f"{geometry_mode}-job"
        estimator = estimator or _FakeMoGeEstimator()
        progress = progress or _RecordingProgressTracker()
        settings = self.settings(geometry_mode, **overrides)
        processor = VideoProcessor(estimator)
        if fail_stereo:
            processor.stereo_generator._run_file_pipeline = self._fail_stereo
        self._active_tracker = progress
        try:
            success = processor.process(
                video_path=str(self.source_video),
                output_dir=str(output_dir),
                video_properties=self.video_properties(),
                settings=settings,
                progress_callback=progress,
            )
        finally:
            self._active_tracker = None
        return _PipelineResult(
            success=success,
            output_dir=output_dir,
            source_video=self.source_video,
            estimator=estimator,
            settings=settings,
            settings_file=processor.orchestrator._settings_file,
            progress=progress,
        )

    @staticmethod
    def video_properties() -> dict[str, Any]:
        return {
            "width": 8,
            "height": 6,
            "fps": 24.0,
            "frame_count": _FRAME_COUNT,
            "duration": _FRAME_COUNT / 24.0,
            "sample_aspect_ratio": "1:1",
            "sample_aspect_ratio_numerator": 1,
            "sample_aspect_ratio_denominator": 1,
        }

    def _extract_frames(
        self,
        video_path: str,
        directories: dict[str, Path],
        _video_properties: dict[str, Any],
        _settings: dict[str, Any],
    ) -> list[Path]:
        frame_files = []
        for index, color in enumerate(self._source_colors, start=1):
            frame = np.full((*_FRAME_SHAPE, 3), color, dtype=np.uint8)
            frame[:, index : index + 2] = (255, 255, 255)
            path = directories["frames"] / f"frame_{index:06d}.png"
            if not path.is_file():
                assert cv2.imwrite(str(path), frame)
            frame_files.append(path)
        write_source_frame_manifest(frame_files, video_path)
        return frame_files

    def _create_video(
        self,
        vr_frames_dir: Path,
        output_dir: Path,
        video_path: str,
        settings: dict[str, Any],
    ) -> bool:
        packed = sorted(vr_frames_dir.glob("*.png"))
        assert len(packed) == _FRAME_COUNT
        for path in packed:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            assert image is not None
            assert image.shape == (480, 1708, 3)
        output = output_dir / generate_output_filename(
            Path(video_path).name,
            settings["vr_format"],
            settings["vr_resolution"],
        )
        output.write_bytes(b"mocked media boundary\n")
        return True

    @staticmethod
    def _fail_stereo(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected stereo failure after selected stage")


@pytest.fixture
def pipeline_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _PipelineHarness:
    return _PipelineHarness(tmp_path, monkeypatch)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _assert_frozen_relative_renderer_boundary() -> None:
    generator = np.random.default_rng(20260816)
    frame = generator.integers(0, 256, size=(7, 19, 3), dtype=np.uint8)
    canonical = generator.random((5, 13), dtype=np.float32)
    result = StereoRenderer(
        device="cpu",
        temporary_budget_bytes=19 * SPLAT_BYTES_PER_PIXEL * 2,
    ).render(
        frame,
        canonical,
        StereoRenderSettings(
            stereo_strength=3.75,
            convergence=0.42,
            occlusion_fill="background",
        ),
    )
    assert _array_sha256(result.left_image) == (
        "48f5e73497d9adea81c5a0dc1444dfa23776f71af291c4a8995470435052d0ce"
    )
    assert _array_sha256(result.right_image) == (
        "5fbd0cb67a9a31f7f18957d597979a0c88db49d5df7b29e2ce731bfd7e3aae05"
    )
    assert _array_sha256(result.left_valid_mask) == (
        "2be1e207cb3c363ebec163e3de22b4b60a5357f10f7e3afc1e10a7e47c4dc03a"
    )
    assert _array_sha256(result.right_valid_mask) == (
        "2be1e207cb3c363ebec163e3de22b4b60a5357f10f7e3afc1e10a7e47c4dc03a"
    )
    assert _array_sha256(result.left_hole_mask) == (
        "57ffc9ca3beb6ee6226c28248ab9c77b2076ef6acffba839cec21fac28a8fd1f"
    )
    assert _array_sha256(result.right_hole_mask) == (
        "57ffc9ca3beb6ee6226c28248ab9c77b2076ef6acffba839cec21fac28a8fd1f"
    )


def _assert_clean_stage_contract(result: _PipelineResult, geometry_mode: str) -> None:
    assert result.success
    assert result.estimator.calls == _FRAME_COUNT
    assert result.final_video.is_file()
    assert result.settings_file is not None
    job = json.loads(result.settings_file.read_text(encoding="utf-8"))
    assert job["metadata"]["processing_status"] == "completed"
    assert job["runtime_info"]["frames_processed"] == _FRAME_COUNT

    packed = sorted((result.output_dir / "99_vr_frames").glob("*.png"))
    assert len(packed) == _FRAME_COUNT
    for path in packed:
        header = read_png_header(path)
        assert header is not None
        assert (header.height, header.width, header.channels, header.bit_depth) == (
            480,
            1708,
            3,
            8,
        )

    raw_dir = result.output_dir / "02_depth_raw"
    raw_metadata = json.loads((raw_dir / "metadata.json").read_text(encoding="utf-8"))
    assert raw_metadata["schema_version"] == 3
    assert raw_metadata["camera_model"] == "pinhole_fx"
    assert raw_metadata["representation"] == "metric_depth"
    raw_files = sorted(raw_dir.glob("*.npz"))
    assert len(raw_files) == _FRAME_COUNT
    for path in raw_files:
        with zipfile.ZipFile(path) as payload:
            assert payload.namelist() == _RAW_MEMBERS

    relative_dir = result.output_dir / "03_disparity_maps"
    metric_dir = result.output_dir / "03_metric_geometry"
    left_dir = result.output_dir / "04_left_frames"
    if geometry_mode == "relative":
        assert len(list(relative_dir.glob("*.png"))) == _FRAME_COUNT
        assert not any(metric_dir.iterdir())
        metadata = json.loads((relative_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["representation"] == "relative_disparity"
        for path in relative_dir.glob("*.png"):
            header = read_png_header(path)
            assert header is not None
            assert (header.height, header.width, header.channels, header.bit_depth) == (6, 8, 1, 16)
        assert not (left_dir / "clamp_summary.json").exists()
        assert "metric_clamp_summary" not in job["runtime_info"]
        _assert_frozen_relative_renderer_boundary()
    else:
        assert not any(relative_dir.iterdir())
        metric_metadata = json.loads((metric_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metric_metadata["status"] == "complete"
        assert metric_metadata["convergence"]["resolved_auto_distance_m"] > 0.0
        metric_files = sorted(metric_dir.glob("*.npz"))
        assert len(metric_files) == _FRAME_COUNT
        for path in metric_files:
            with zipfile.ZipFile(path) as payload:
                assert payload.namelist() == _METRIC_MEMBERS
        summary = json.loads((left_dir / "clamp_summary.json").read_text(encoding="utf-8"))
        assert set(summary) == {
            "schema_version",
            "frame_names",
            "clamped_fractions",
            "affected_frame_count",
            "mean_clamped_fraction",
            "max_clamped_fraction",
        }
        runtime_summary = job["runtime_info"]["metric_clamp_summary"]
        for key in (
            "affected_frame_count",
            "mean_clamped_fraction",
            "max_clamped_fraction",
        ):
            assert runtime_summary[key] == summary[key]


@pytest.mark.parametrize("geometry_mode", ["relative", "metric_camera"])
def test_mocked_moge_runs_selected_geometry_through_sbs_assembly(
    pipeline_harness: _PipelineHarness,
    geometry_mode: str,
) -> None:
    result = pipeline_harness.run(geometry_mode)

    _assert_clean_stage_contract(result, geometry_mode)


def test_metric_stereo_preview_follows_completed_clip_convergence(
    pipeline_harness: _PipelineHarness,
) -> None:
    result = pipeline_harness.run("metric_camera")

    assert result.success
    stereo_event = next(event for event in result.progress.events if event.kind == "stereo_left")
    complete_event = next(
        event for event in result.progress.events if event.kind == "metric_complete"
    )
    assert complete_event.sequence < stereo_event.sequence
    metadata = json.loads(complete_event.path.read_text(encoding="utf-8"))
    assert metadata["status"] == "complete"
    assert metadata["convergence"]["resolved_auto_distance_m"] > 0.0
    assert stereo_event.png_bytes == stereo_event.path.read_bytes()


def test_explicit_metric_preview_and_final_use_the_same_convergence(
    pipeline_harness: _PipelineHarness,
) -> None:
    result = pipeline_harness.run(
        "metric_camera",
        metric_convergence_distance=2.25,
    )

    assert result.success
    stereo_metadata = json.loads(
        (result.output_dir / "04_left_frames" / "metadata.json").read_text(encoding="utf-8")
    )
    assert stereo_metadata["requested_convergence_distance"] == 2.25
    assert stereo_metadata["effective_convergence_distance_m"] == 2.25
    preview = next(event for event in result.progress.events if event.kind == "stereo_left")
    assert preview.png_bytes == preview.path.read_bytes()


def test_metric_projection_setting_change_reuses_raw_and_metric_stage(
    pipeline_harness: _PipelineHarness,
) -> None:
    output_dir = pipeline_harness.root / "metric-resume"
    first = pipeline_harness.run("metric_camera", output_dir=output_dir)
    metric_before = _tree_hashes(output_dir / "03_metric_geometry")
    raw_before = _tree_hashes(output_dir / "02_depth_raw")
    stereo_before = {
        **_tree_hashes(output_dir / "04_left_frames"),
        **{
            f"right/{name}": digest
            for name, digest in _tree_hashes(output_dir / "04_right_frames").items()
        },
    }

    second = pipeline_harness.run(
        "metric_camera",
        output_dir=output_dir,
        estimator=_FakeMoGeEstimator(),
        virtual_baseline_mm=100.0,
        metric_convergence_distance=0.1,
        max_disparity_percent=5.0,
    )
    stereo_after = {
        **_tree_hashes(output_dir / "04_left_frames"),
        **{
            f"right/{name}": digest
            for name, digest in _tree_hashes(output_dir / "04_right_frames").items()
        },
    }

    assert first.success and second.success
    assert first.estimator.calls == _FRAME_COUNT
    assert second.estimator.calls == 0
    assert _tree_hashes(output_dir / "02_depth_raw") == raw_before
    assert _tree_hashes(output_dir / "03_metric_geometry") == metric_before
    assert stereo_after != stereo_before


def test_retained_mode_switch_builds_only_missing_selected_stage(
    pipeline_harness: _PipelineHarness,
) -> None:
    output_dir = pipeline_harness.root / "mode-switch"
    metric = pipeline_harness.run(
        "metric_camera",
        output_dir=output_dir,
        keep_intermediates=True,
    )
    metric_before = _tree_hashes(output_dir / "03_metric_geometry")
    raw_before = _tree_hashes(output_dir / "02_depth_raw")

    relative = pipeline_harness.run(
        "relative",
        output_dir=output_dir,
        estimator=_FakeMoGeEstimator(),
        keep_intermediates=True,
    )

    assert metric.success and relative.success
    assert relative.estimator.calls == 0
    assert _tree_hashes(output_dir / "02_depth_raw") == raw_before
    assert _tree_hashes(output_dir / "03_metric_geometry") == metric_before
    assert len(list((output_dir / "03_disparity_maps").glob("*.png"))) == _FRAME_COUNT


def test_no_retention_failure_keeps_completed_stage_and_reports_required_inference(
    pipeline_harness: _PipelineHarness,
) -> None:
    output_dir = pipeline_harness.root / "no-retention-failure"
    failed = pipeline_harness.run(
        "metric_camera",
        output_dir=output_dir,
        keep_intermediates=False,
        fail_stereo=True,
    )

    assert not failed.success
    raw_dir = output_dir / "02_depth_raw"
    raw_metadata = json.loads((raw_dir / "metadata.json").read_text(encoding="utf-8"))
    assert raw_metadata["storage_status"] == "ready"
    assert list(raw_dir.glob("*.npz")) == []
    source_before = _tree_hashes(output_dir / "00_original_frames")
    metric_before = _tree_hashes(output_dir / "03_metric_geometry")
    assert len(list((output_dir / "00_original_frames").glob("*.png"))) == _FRAME_COUNT
    assert len(list((output_dir / "03_metric_geometry").glob("*.npz"))) == _FRAME_COUNT
    assert failed.settings_file is not None
    status = json.loads(failed.settings_file.read_text(encoding="utf-8"))
    assert status["metadata"]["processing_status"] == "failed"

    relative_settings = pipeline_harness.settings(
        "relative",
        keep_intermediates=False,
    )
    fingerprint = depth_processor.build_current_model_fingerprint(
        _FakeMoGeEstimator(),
        relative_settings,
    )
    report = build_resume_report(
        output_dir,
        relative_settings,
        source_video=failed.source_video,
        model_fingerprint=fingerprint,
        settings_file=failed.settings_file,
    )

    assert report.stage("disparity_maps").reason == (
        "MoGe inference is required to build the selected geometry stage"
    )
    assert _tree_hashes(output_dir / "00_original_frames") == source_before
    assert _tree_hashes(output_dir / "03_metric_geometry") == metric_before


def _load_cli_module():
    project_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "depth_surge_3d_cli_moge2_integration",
        project_root / "depth_surge_3d.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_dispatch_reaches_metric_pipeline(
    pipeline_harness: _PipelineHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli_module = _load_cli_module()
    projector_module = pipeline_harness.patch_package_namespace(monkeypatch)
    requests = []

    def estimator_factory(backend_id, request):
        requests.append((backend_id, request))
        return _FakeMoGeEstimator()

    monkeypatch.setattr(projector_module, "create_registered_depth_estimator", estimator_factory)
    monkeypatch.setattr(projector_module, "validate_video_file", lambda _path: True)
    monkeypatch.setattr(
        projector_module,
        "get_video_properties",
        lambda _path: pipeline_harness.video_properties(),
    )
    monkeypatch.setattr(cli_module, "validate_video_file", lambda _path: True)
    monkeypatch.setattr(
        cli_module,
        "backend_availability",
        lambda _backend_id: BackendAvailability(True),
    )
    output_root = pipeline_harness.root / "cli-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "depth_surge_3d.py",
            str(pipeline_harness.source_video),
            "--output-dir",
            str(output_root),
            "--depth-model-version",
            "moge2",
            "--model-size",
            "vitb",
            "--stereo-geometry-mode",
            "metric_camera",
            "--format",
            "side_by_side",
            "--no-distortion",
            "--vr-resolution",
            "16x9-480p",
            "--depth-resolution",
            "8",
            "--no-audio",
        ],
    )

    exit_code = cli_module.main()

    assert exit_code == 0
    assert len(requests) == 1
    assert requests[0][0] == "moge2"
    assert requests[0][1].model_size == "vitb"
    completed = list(output_root.glob("*/03_metric_geometry/metadata.json"))
    assert len(completed) == 1
    metadata = json.loads(completed[0].read_text(encoding="utf-8"))
    assert metadata["status"] == "complete"
    expected_output = completed[0].parents[1] / generate_output_filename(
        pipeline_harness.source_video.name,
        "side_by_side",
        "16x9-480p",
    )
    assert expected_output.is_file()


def test_web_dispatch_reaches_relative_pipeline(
    pipeline_harness: _PipelineHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app as web_app

    projector_module = pipeline_harness.patch_package_namespace(monkeypatch)

    requests = []
    acknowledgements = []
    emissions = []
    real_emit = web_app.socketio.emit

    def estimator_factory(backend_id, request):
        requests.append((backend_id, request))
        return _FakeMoGeEstimator()

    def run_synchronously(target, *args):
        target(*args)
        return object()

    def acknowledge(event, data, *, to, timeout):
        acknowledgements.append((event, data, to, timeout))
        web_app.socketio.emit(event, data, room=to)
        return {"accepted": True}

    def record_emit(event, *args, **kwargs):
        emissions.append((event, args, kwargs))
        return real_emit(event, *args, **kwargs)

    web_root = pipeline_harness.root / "web-output"
    output_dir = web_root / "relative-job"
    output_dir.mkdir(parents=True)
    source_video = output_dir / "clip.mp4"
    source_video.write_bytes(b"web fake video boundary")
    web_app.current_processing.update(
        {"active": False, "session_id": None, "thread": None, "stop_requested": False}
    )
    monkeypatch.setitem(web_app.app.config, "OUTPUT_FOLDER", str(web_root))
    monkeypatch.setattr(
        web_app,
        "backend_availability",
        lambda _backend_id: BackendAvailability(True),
    )
    monkeypatch.setattr(
        web_app,
        "get_video_properties",
        lambda _path: pipeline_harness.video_properties(),
    )
    monkeypatch.setattr(projector_module, "create_registered_depth_estimator", estimator_factory)
    monkeypatch.setattr(web_app.socketio, "start_background_task", run_synchronously)
    monkeypatch.setattr(web_app.socketio, "emit", record_emit)
    monkeypatch.setattr(web_app.socketio, "call", acknowledge)
    monkeypatch.setattr(web_app.socketio, "sleep", lambda *_args, **_kwargs: None)

    client = web_app.app.test_client()
    socket_client = web_app.socketio.test_client(web_app.app, flask_test_client=client)
    try:
        socket_id = web_app.socketio.server.manager.sid_from_eio_sid(
            socket_client.eio_sid,
            "/",
        )
        response = client.post(
            "/process",
            json={
                "output_dir": str(output_dir),
                "socket_id": socket_id,
                "settings": {
                    "depth_model_version": "moge2",
                    "model_size": "vits",
                    "stereo_geometry_mode": "relative",
                    "apply_distortion": False,
                    "vr_format": "side_by_side",
                    "vr_resolution": "16x9-480p",
                    "depth_resolution": 8,
                    "target_fps": None,
                    "preserve_audio": False,
                },
            },
        )
        socket_client.get_received()
    finally:
        socket_client.disconnect()
        web_app.current_processing.update(
            {"active": False, "session_id": None, "thread": None, "stop_requested": False}
        )

    assert response.status_code == 200
    assert len(requests) == 1
    assert requests[0][0] == "moge2"
    assert requests[0][1].model_size == "vits"
    assert len(acknowledgements) == 1
    assert acknowledgements[0][0] == "processing_configuration"
    assert acknowledgements[0][2] == socket_id
    assert acknowledgements[0][3] == web_app.PROCESSING_CONFIGURATION_ACK_TIMEOUT
    assert any(event == "processing_complete" for event, _args, _kwargs in emissions)
    assert len(list((output_dir / "03_disparity_maps").glob("*.png"))) == _FRAME_COUNT
    assert not any((output_dir / "03_metric_geometry").iterdir())
    expected_output = output_dir / generate_output_filename(
        source_video.name,
        "side_by_side",
        "16x9-480p",
    )
    assert expected_output.is_file()
