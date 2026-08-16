"""Tests for the bounded stereo rendering and I/O pipeline."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.processing.frames import stereo_generator
from src.depth_surge_3d.processing.frames.depth_storage import canonical_json_hash
from src.depth_surge_3d.processing.frames.metric_geometry import (
    ClipConvergence,
    MetricGeometryFrame,
    MetricGeometryStore,
)
from src.depth_surge_3d.processing.frames.stereo_generator import (
    HOST_SLOT_OVERHEAD,
    HOST_STEREO_BYTES_PER_PIXEL,
    STEREO_HOST_BUDGET,
    StereoPairGenerator,
    _atomic_write_png,
    calculate_stereo_pipeline_capacity,
    validate_stereo_io_workers,
)
from src.depth_surge_3d.rendering.stereo_renderer import (
    StereoRenderResult,
    StereoRenderer,
    StereoSplatSettings,
)


def test_stereo_algorithm_version_identifies_production_sixteen_sample_zbuffer() -> None:
    assert stereo_generator.STEREO_STAGE_ALGORITHM_VERSION == "torch-horizontal-16x-zbuffer-v3"


def _write_canonical_metadata(
    depth_dir: Path,
    frame_names: list[str],
    shape: tuple[int, int] = (8, 8),
) -> None:
    metadata = {
        "schema_version": 1,
        "algorithm_version": "scene-percentile-v1",
        "representation": "relative_disparity",
        "near_value": 1.0,
        "far_value": 0.0,
        "encoding": "uint16_png",
        "encoding_scale": 65535.0,
        "num_frames": len(frame_names),
        "frame_names": frame_names,
        "native_shape": list(shape),
        "source_raw_fingerprint": "raw",
        "source_model_fingerprint": "model",
        "scene_manifest_fingerprint": "scene",
        "depth_bounds_fingerprint": "bounds",
    }
    metadata["fingerprint"] = canonical_json_hash(metadata)
    (depth_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _make_file_inputs(
    root: Path,
    *,
    count: int = 3,
    frame_shape: tuple[int, int] = (8, 8),
    depth_shape: tuple[int, int] | None = None,
) -> tuple[list[Path], list[Path], dict[str, Path]]:
    depth_shape = depth_shape or frame_shape
    frame_dir = root / "frames"
    depth_dir = root / "canonical"
    left_dir = root / "left"
    right_dir = root / "right"
    for directory in (frame_dir, depth_dir, left_dir, right_dir):
        directory.mkdir(parents=True)

    frame_files: list[Path] = []
    depth_files: list[Path] = []
    for index in range(count):
        name = f"frame_{index:04d}.png"
        frame_path = frame_dir / name
        depth_path = depth_dir / name
        frame = np.full((*frame_shape, 3), 20 + index, dtype=np.uint8)
        depth = np.full(depth_shape, 32768 + index, dtype=np.uint16)
        assert cv2.imwrite(str(frame_path), frame)
        assert cv2.imwrite(str(depth_path), depth)
        frame_files.append(frame_path)
        depth_files.append(depth_path)
    _write_canonical_metadata(depth_dir, [path.name for path in frame_files], depth_shape)
    return frame_files, depth_files, {"left_frames": left_dir, "right_frames": right_dir}


def _settings(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "stereo_strength": 2.0,
        "convergence": 0.5,
        "occlusion_fill": "background",
        "stereo_io_workers": 2,
        "keep_intermediates": False,
    }
    values.update(overrides)
    return values


class _FakeRenderer:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[int, object, StereoSplatSettings]] = []

    def render_geometry(
        self,
        frame: np.ndarray,
        geometry: object,
        settings: StereoSplatSettings,
    ) -> StereoRenderResult:
        self.calls.append((threading.get_ident(), geometry, settings))
        if self.fail:
            raise RuntimeError("render failed")
        mask = np.ones(frame.shape[:2], dtype=np.bool_)
        holes = np.zeros(frame.shape[:2], dtype=np.bool_)
        return StereoRenderResult(
            left_image=np.clip(frame.astype(np.int16) + 1, 0, 255).astype(np.uint8),
            right_image=np.clip(frame.astype(np.int16) + 2, 0, 255).astype(np.uint8),
            left_valid_mask=mask.copy(),
            right_valid_mask=mask.copy(),
            left_hole_mask=holes.copy(),
            right_hole_mask=holes.copy(),
        )


class _RecordingCommonRenderer:
    def __init__(self) -> None:
        self.geometry_calls: list[tuple[int, object, StereoSplatSettings]] = []
        self.legacy_relative_calls: list[object] = []

    def render(self, *args: object, **kwargs: object) -> StereoRenderResult:
        self.legacy_relative_calls.append((args, kwargs))
        raise AssertionError("the file pipeline must not call the legacy wrapper")

    def render_geometry(
        self,
        frame: np.ndarray,
        geometry: object,
        settings: StereoSplatSettings,
    ) -> StereoRenderResult:
        self.geometry_calls.append((threading.get_ident(), geometry, settings))
        mask = np.ones(frame.shape[:2], dtype=np.bool_)
        holes = np.zeros(frame.shape[:2], dtype=np.bool_)
        return StereoRenderResult(
            left_image=frame.copy(),
            right_image=frame.copy(),
            left_valid_mask=mask.copy(),
            right_valid_mask=mask.copy(),
            left_hole_mask=holes.copy(),
            right_hole_mask=holes.copy(),
        )


def _make_metric_inputs(
    root: Path,
    *,
    inverse_values: list[float] | None = None,
) -> tuple[list[Path], list[Path], dict[str, Path], float]:
    inverse_values = inverse_values or [1.0, 0.5, 0.25]
    count = len(inverse_values)
    frame_dir = root / "frames"
    metric_dir = root / "metric"
    left_dir = root / "left"
    right_dir = root / "right"
    for directory in (frame_dir, metric_dir, left_dir, right_dir):
        directory.mkdir(parents=True)

    frame_files: list[Path] = []
    for index in range(count):
        frame_path = frame_dir / f"frame_{index:04d}.png"
        assert cv2.imwrite(str(frame_path), np.full((4, 5, 3), index + 1, dtype=np.uint8))
        frame_files.append(frame_path)
    store = MetricGeometryStore.open_or_create(
        metric_dir,
        frame_names=[path.name for path in frame_files],
        native_shape=(2, 3),
        source_raw_fingerprint="raw-metric",
        source_frame_fingerprint="frames",
        candidate_scene_fingerprint="scenes",
        preflight_required_bytes=0,
    )
    for frame_path, inverse_value in zip(frame_files, inverse_values):
        inverse = np.full((2, 3), inverse_value, dtype=np.float32)
        store.write_frame(
            frame_path.name,
            MetricGeometryFrame(
                inverse,
                np.ones((2, 3), dtype=np.bool_),
                np.float32(0.5),
            ),
        )
    resolved_auto_distance_m = 2.0
    store.finalize(ClipConvergence(np.float32(resolved_auto_distance_m), (0,), 6))
    return (
        frame_files,
        list(store.complete_files),
        {"left_frames": left_dir, "right_frames": right_dir},
        resolved_auto_distance_m,
    )


@pytest.mark.parametrize("workers", [0, -1, 17, 100])
def test_stereo_io_workers_reject_out_of_range_values(workers: int) -> None:
    with pytest.raises(ValueError, match=r"1\.\.16"):
        validate_stereo_io_workers(workers)


@pytest.mark.parametrize("workers", [1, 4, 16])
def test_stereo_io_workers_accept_supported_values(workers: int) -> None:
    assert validate_stereo_io_workers(workers) == workers


def test_4k_capacity_stays_within_host_budget_at_max_workers() -> None:
    capacity = calculate_stereo_pipeline_capacity(3840, 2160, 16)
    slot_bytes = 3840 * 2160 * HOST_STEREO_BYTES_PER_PIXEL + HOST_SLOT_OVERHEAD

    assert HOST_STEREO_BYTES_PER_PIXEL == 24
    assert capacity == min(32, STEREO_HOST_BUDGET // slot_bytes)
    assert capacity * slot_bytes <= STEREO_HOST_BUDGET


def test_frame_larger_than_one_host_slot_is_rejected() -> None:
    with pytest.raises(MemoryError, match="required"):
        calculate_stereo_pipeline_capacity(16384, 8192, 4)


def test_supported_8k_frame_is_rejected_before_decode() -> None:
    with pytest.raises(MemoryError, match=r"7680x4320.*required"):
        calculate_stereo_pipeline_capacity(7680, 4320, 4)


def test_atomic_png_write_replaces_from_same_directory(tmp_path: Path) -> None:
    output = tmp_path / "frame.png"
    image = np.full((4, 5, 3), 73, dtype=np.uint8)

    with patch.object(os, "replace", wraps=os.replace) as replace:
        _atomic_write_png(output, image)

    assert replace.call_count == 1
    temporary, destination = replace.call_args.args
    assert Path(temporary).parent == output.parent
    assert Path(destination) == output
    assert np.array_equal(cv2.imread(str(output), cv2.IMREAD_COLOR), image)
    assert list(tmp_path.glob("*.tmp")) == []


def test_file_pipeline_renders_only_on_calling_thread(tmp_path: Path) -> None:
    frame_files, depth_files, directories = _make_file_inputs(tmp_path, count=5)
    renderer = _FakeRenderer()
    calling_thread = threading.get_ident()

    result = StereoPairGenerator(renderer=renderer).create_stereo_pairs_from_files(
        frame_files,
        depth_files,
        directories,
        _settings(stereo_io_workers=4),
    )

    assert result is True
    assert len(renderer.calls) == 5
    assert {thread_id for thread_id, _, _ in renderer.calls} == {calling_thread}
    assert not any(thread.name.startswith("stereo-") for thread in threading.enumerate())


def test_file_pipeline_uses_corrected_stereo_sign_end_to_end(tmp_path: Path) -> None:
    frame_files, depth_files, directories = _make_file_inputs(
        tmp_path,
        count=1,
        frame_shape=(5, 100),
    )
    frame = np.zeros((5, 100, 3), dtype=np.uint8)
    frame[:, 50] = 255
    assert cv2.imwrite(str(frame_files[0]), frame)
    assert cv2.imwrite(
        str(depth_files[0]),
        np.full((5, 100), 65535, dtype=np.uint16),
    )

    result = StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_files(
        frame_files,
        depth_files,
        directories,
        _settings(
            stereo_strength=4.0,
            convergence=0.5,
            occlusion_fill="none",
        ),
    )

    left = cv2.imread(str(directories["left_frames"] / frame_files[0].name))
    right = cv2.imread(str(directories["right_frames"] / frame_files[0].name))
    assert result is True
    assert int(np.argmax(left[2, :, 0])) > int(np.argmax(right[2, :, 0]))


def test_file_pipeline_uses_bounded_lifecycle_permits(tmp_path: Path) -> None:
    frame_files, depth_files, directories = _make_file_inputs(tmp_path, count=10)
    generator = StereoPairGenerator(renderer=_FakeRenderer())

    assert generator.create_stereo_pairs_from_files(
        frame_files,
        depth_files,
        directories,
        _settings(stereo_io_workers=4),
    )

    stats = generator.last_pipeline_stats
    assert stats is not None
    assert stats.queue_capacity == stats.permit_count
    assert stats.permit_count == 8
    assert stats.max_active_permits <= stats.permit_count
    assert stats.permits_acquired == 10
    assert stats.permits_released == 10
    assert stats.active_permits == 0
    assert stats.decoded_frames == 10
    assert stats.rendered_frames == 10
    assert stats.written_frames == 10
    assert stats.pipeline_wall_seconds > 0.0
    assert stats.writer_busy_seconds >= 0.0


def test_file_pipeline_always_writes_downstream_frames(tmp_path: Path) -> None:
    frame_files, depth_files, directories = _make_file_inputs(tmp_path, count=2)

    result = StereoPairGenerator(renderer=_FakeRenderer()).create_stereo_pairs_from_files(
        frame_files,
        depth_files,
        directories,
        _settings(keep_intermediates=False),
    )

    assert result is True
    assert len(list(directories["left_frames"].glob("*.png"))) == 2
    assert len(list(directories["right_frames"].glob("*.png"))) == 2


def test_resume_skips_only_complete_stereo_pairs(tmp_path: Path) -> None:
    frame_files, depth_files, directories = _make_file_inputs(tmp_path, count=2)
    assert StereoPairGenerator(renderer=_FakeRenderer()).create_stereo_pairs_from_files(
        frame_files,
        depth_files,
        directories,
        _settings(),
    )
    complete_left = directories["left_frames"] / "frame_0000.png"
    complete_right = directories["right_frames"] / "frame_0000.png"
    singleton_left = directories["left_frames"] / "frame_0001.png"
    assert cv2.imwrite(str(complete_left), np.full((8, 8, 3), 7, dtype=np.uint8))
    assert cv2.imwrite(str(complete_right), np.full((8, 8, 3), 8, dtype=np.uint8))
    assert cv2.imwrite(str(singleton_left), np.full((8, 8, 3), 9, dtype=np.uint8))
    (directories["right_frames"] / "frame_0001.png").unlink()
    renderer = _FakeRenderer()

    result = StereoPairGenerator(renderer=renderer).create_stereo_pairs_from_files(
        frame_files,
        depth_files,
        directories,
        _settings(),
    )

    assert result is True
    assert len(renderer.calls) == 1
    assert np.all(cv2.imread(str(complete_left)) == 7)
    assert np.all(cv2.imread(str(complete_right)) == 8)
    assert np.all(cv2.imread(str(singleton_left)) == 22)
    assert (directories["right_frames"] / "frame_0001.png").is_file()


def test_resume_regenerates_a_corrupt_stereo_pair(tmp_path: Path) -> None:
    frame_files, depth_files, directories = _make_file_inputs(tmp_path, count=2)
    assert StereoPairGenerator(renderer=_FakeRenderer()).create_stereo_pairs_from_files(
        frame_files,
        depth_files,
        directories,
        _settings(),
    )
    corrupt_right = directories["right_frames"] / "frame_0000.png"
    corrupt_right.write_bytes(b"corrupt")
    renderer = _FakeRenderer()

    result = StereoPairGenerator(renderer=renderer).create_stereo_pairs_from_files(
        frame_files,
        depth_files,
        directories,
        _settings(),
    )

    assert result is True
    assert len(renderer.calls) == 1
    restored = cv2.imread(str(corrupt_right), cv2.IMREAD_UNCHANGED)
    assert restored is not None and restored.shape == (8, 8, 3)


def test_v3_clean_and_resumed_runs_write_byte_identical_eye_pngs(tmp_path: Path) -> None:
    clean_inputs = _make_file_inputs(tmp_path / "clean", count=3, frame_shape=(8, 16))
    resumed_inputs = _make_file_inputs(tmp_path / "resumed", count=3, frame_shape=(8, 16))
    settings = _settings(stereo_strength=3.0, stereo_io_workers=1)
    renderer = StereoRenderer(device="cpu")

    assert StereoPairGenerator(renderer=renderer).create_stereo_pairs_from_files(
        *clean_inputs,
        settings,
    )
    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_files(
        *resumed_inputs,
        settings,
    )
    resumed_frame = resumed_inputs[0][1].name
    (resumed_inputs[2]["right_frames"] / resumed_frame).unlink()

    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_files(
        *resumed_inputs,
        settings,
    )

    for eye in ("left_frames", "right_frames"):
        for frame_file in clean_inputs[0]:
            clean_png = clean_inputs[2][eye] / frame_file.name
            resumed_png = resumed_inputs[2][eye] / frame_file.name
            assert resumed_png.read_bytes() == clean_png.read_bytes()


def test_canonical_fingerprint_change_invalidates_existing_stereo_pairs(tmp_path: Path) -> None:
    frame_files, depth_files, directories = _make_file_inputs(tmp_path, count=2)
    directories["vr_frames"] = tmp_path / "vr"
    directories["vr_frames"].mkdir()
    assert StereoPairGenerator(renderer=_FakeRenderer()).create_stereo_pairs_from_files(
        frame_files,
        depth_files,
        directories,
        _settings(),
    )
    canonical_metadata_path = depth_files[0].parent / "metadata.json"
    canonical_metadata = json.loads(canonical_metadata_path.read_text(encoding="utf-8"))
    canonical_metadata["source_raw_fingerprint"] = "different-model-and-source"
    canonical_metadata.pop("fingerprint")
    canonical_metadata["fingerprint"] = canonical_json_hash(canonical_metadata)
    canonical_metadata_path.write_text(json.dumps(canonical_metadata), encoding="utf-8")
    stale_vr_frame = directories["vr_frames"] / "frame_0000.png"
    stale_vr_frame.write_bytes(b"stale downstream output")
    renderer = _FakeRenderer()

    result = StereoPairGenerator(renderer=renderer).create_stereo_pairs_from_files(
        frame_files,
        depth_files,
        directories,
        _settings(),
    )

    assert result is True
    assert len(renderer.calls) == 2
    stage_metadata = json.loads(
        (directories["left_frames"] / "metadata.json").read_text(encoding="utf-8")
    )
    assert stage_metadata["source_canonical_fingerprint"] == canonical_metadata["fingerprint"]
    assert not stale_vr_frame.exists()


def test_decode_failure_releases_every_lifecycle_permit(tmp_path: Path) -> None:
    frame_files, depth_files, directories = _make_file_inputs(tmp_path, count=3)
    frame_files[1].write_bytes(b"not an image")
    generator = StereoPairGenerator(renderer=_FakeRenderer())

    result = generator.create_stereo_pairs_from_files(
        frame_files,
        depth_files,
        directories,
        _settings(),
    )

    assert result is False
    stats = generator.last_pipeline_stats
    assert stats is not None
    assert stats.active_permits == 0
    assert stats.permits_acquired == stats.permits_released


def test_render_failure_releases_every_lifecycle_permit(tmp_path: Path) -> None:
    frame_files, depth_files, directories = _make_file_inputs(tmp_path, count=3)
    generator = StereoPairGenerator(renderer=_FakeRenderer(fail=True))

    result = generator.create_stereo_pairs_from_files(
        frame_files,
        depth_files,
        directories,
        _settings(),
    )

    assert result is False
    stats = generator.last_pipeline_stats
    assert stats is not None
    assert stats.active_permits == 0
    assert stats.permits_acquired == stats.permits_released


def test_write_failure_cleans_pair_and_releases_permit(tmp_path: Path) -> None:
    frame_files, depth_files, directories = _make_file_inputs(tmp_path, count=2)
    generator = StereoPairGenerator(renderer=_FakeRenderer())

    with patch.object(stereo_generator, "_atomic_write_png", side_effect=OSError("disk full")):
        result = generator.create_stereo_pairs_from_files(
            frame_files,
            depth_files,
            directories,
            _settings(),
        )

    assert result is False
    assert list(directories["left_frames"].glob("*.png")) == []
    assert list(directories["right_frames"].glob("*.png")) == []
    stats = generator.last_pipeline_stats
    assert stats is not None
    assert stats.active_permits == 0
    assert stats.permits_acquired == stats.permits_released
    assert not any(thread.name.startswith("stereo-") for thread in threading.enumerate())


def test_file_pipeline_rejects_missing_canonical_metadata(tmp_path: Path) -> None:
    frame_files, depth_files, directories = _make_file_inputs(tmp_path, count=1)
    (depth_files[0].parent / "metadata.json").unlink()

    result = StereoPairGenerator(renderer=_FakeRenderer()).create_stereo_pairs_from_files(
        frame_files,
        depth_files,
        directories,
        _settings(),
    )

    assert result is False


def test_in_memory_array_api_is_removed() -> None:
    assert "create_stereo_pairs" not in vars(StereoPairGenerator)


def test_canonical_metadata_requires_source_fingerprints(tmp_path: Path) -> None:
    frame_files, depth_files, _directories = _make_file_inputs(tmp_path, count=1)
    metadata_path = depth_files[0].parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("source_raw_fingerprint")
    metadata.pop("fingerprint")
    metadata["fingerprint"] = canonical_json_hash(metadata)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        StereoPairGenerator._get_canonical_metadata(depth_files, frame_files)


def test_canonical_metadata_rejects_depth_from_another_directory(tmp_path: Path) -> None:
    frame_files, depth_files, _directories = _make_file_inputs(tmp_path, count=2)
    foreign_dir = tmp_path / "foreign"
    foreign_dir.mkdir()
    foreign = foreign_dir / depth_files[1].name
    foreign.write_bytes(depth_files[1].read_bytes())

    with pytest.raises(ValueError, match="path manifest"):
        StereoPairGenerator._get_canonical_metadata(
            [depth_files[0], foreign],
            frame_files,
        )


def test_wrong_native_shape_is_rejected_before_rendering(tmp_path: Path) -> None:
    frame_files, depth_files, directories = _make_file_inputs(tmp_path, count=1)
    cv2.imwrite(str(depth_files[0]), np.zeros((4, 4), dtype=np.uint16))
    renderer = _FakeRenderer()

    result = StereoPairGenerator(renderer=renderer).create_stereo_pairs_from_files(
        frame_files,
        depth_files,
        directories,
        _settings(),
    )

    assert result is False
    assert renderer.calls == []


def test_relative_generator_decodes_geometry_before_the_common_renderer(tmp_path: Path) -> None:
    frame_files, depth_files, directories = _make_file_inputs(tmp_path, count=1)
    renderer = _RecordingCommonRenderer()

    assert StereoPairGenerator(renderer=renderer).create_stereo_pairs_from_files(
        frame_files,
        depth_files,
        directories,
        _settings(stereo_geometry_mode="relative"),
    )

    assert len(renderer.geometry_calls) == 1
    assert not renderer.legacy_relative_calls
    _, geometry, splat_settings = renderer.geometry_calls[0]
    assert geometry.near_score.shape == (8, 8)
    assert splat_settings.max_eye_shift_fraction == pytest.approx(0.01)


def test_metric_generator_uses_resolved_auto_convergence_and_common_renderer(
    tmp_path: Path,
) -> None:
    frame_files, metric_files, directories, resolved_auto_distance_m = _make_metric_inputs(tmp_path)
    renderer = _RecordingCommonRenderer()
    settings = _settings(
        stereo_geometry_mode="metric_camera",
        metric_convergence_distance="auto",
        virtual_baseline_mm=63.0,
        max_disparity_percent=2.0,
        crop_factor=0.75,
        apply_distortion=False,
        vr_format="side_by_side",
        per_eye_width=1920,
        vr_output_width=3840,
    )
    generator = StereoPairGenerator(renderer=renderer)

    assert generator.create_stereo_pairs_from_files(
        frame_files, metric_files, directories, settings
    )

    assert len(renderer.geometry_calls) == len(frame_files)
    assert not renderer.legacy_relative_calls
    metadata = json.loads((directories["left_frames"] / "metadata.json").read_text())
    assert metadata["geometry_mode"] == "metric_camera"
    assert metadata["effective_convergence_distance_m"] == pytest.approx(resolved_auto_distance_m)
    assert metadata["source_width"] == 5
    assert metadata["retained_crop_width"] == 3
    assert metadata["sample_aspect_ratio"] == "1:1"
    assert "stereo_strength" not in json.dumps(metadata)
    assert 'convergence"' not in json.dumps(metadata)
    assert "per_eye_width" not in json.dumps(metadata)
    assert "vr_output_width" not in json.dumps(metadata)


def test_mode_specific_fingerprints_ignore_inactive_and_final_width_settings(
    tmp_path: Path,
) -> None:
    frame_files, metric_files, directories, _ = _make_metric_inputs(tmp_path)
    renderer = _RecordingCommonRenderer()
    base = _settings(
        stereo_geometry_mode="metric_camera",
        metric_convergence_distance=3.0,
        virtual_baseline_mm=63.0,
        max_disparity_percent=2.0,
        crop_factor=0.75,
        apply_distortion=False,
        vr_format="side_by_side",
        stereo_strength=1.0,
        convergence=0.1,
        per_eye_width=7,
        vr_output_width=14,
    )
    generator = StereoPairGenerator(renderer=renderer)
    assert generator.create_stereo_pairs_from_files(frame_files, metric_files, directories, base)
    first = json.loads((directories["left_frames"] / "metadata.json").read_text())

    changed_inactive = {
        **base,
        "stereo_strength": 5.0,
        "convergence": 0.9,
        "per_eye_width": 3840,
        "vr_output_width": 7680,
    }
    assert generator.create_stereo_pairs_from_files(
        frame_files, metric_files, directories, changed_inactive
    )
    second = json.loads((directories["left_frames"] / "metadata.json").read_text())

    assert second["fingerprint"] == first["fingerprint"]
    assert len(renderer.geometry_calls) == len(frame_files)


def test_metric_pair_and_stats_sidecar_are_one_failure_unit(tmp_path: Path) -> None:
    frame_files, metric_files, directories, _ = _make_metric_inputs(tmp_path, inverse_values=[1.0])
    stat_dir = directories["left_frames"] / "clamp_stats"
    generator = StereoPairGenerator(renderer=_RecordingCommonRenderer())

    original_atomic_write_json = stereo_generator._atomic_write_json

    def fail_stat_write(path: Path, payload: dict[str, object]) -> None:
        if path.parent.name == "clamp_stats":
            raise OSError("disk full")
        original_atomic_write_json(path, payload)

    with patch.object(stereo_generator, "_atomic_write_json", side_effect=fail_stat_write):
        result = generator.create_stereo_pairs_from_files(
            frame_files,
            metric_files,
            directories,
            _settings(
                stereo_geometry_mode="metric_camera",
                metric_convergence_distance="auto",
                virtual_baseline_mm=63.0,
                max_disparity_percent=2.0,
                crop_factor=0.75,
            ),
        )

    assert result is False
    assert list(directories["left_frames"].glob("*.png")) == []
    assert list(directories["right_frames"].glob("*.png")) == []
    assert not stat_dir.exists() or list(stat_dir.glob("*.json")) == []


def test_metric_sidecar_repair_summary_and_source_order_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frame_files, metric_files, directories, _ = _make_metric_inputs(
        tmp_path,
        inverse_values=[0.5, 100.0, 200.0],
    )
    settings = _settings(
        stereo_geometry_mode="metric_camera",
        metric_convergence_distance="auto",
        virtual_baseline_mm=100.0,
        max_disparity_percent=1.0,
        crop_factor=0.75,
    )
    first_renderer = _RecordingCommonRenderer()
    first_generator = StereoPairGenerator(renderer=first_renderer)
    assert first_generator.create_stereo_pairs_from_files(
        frame_files, metric_files, directories, settings
    )
    captured = capsys.readouterr().out

    assert captured.count("clamped") == 1
    assert frame_files[1].stem in captured
    summary = first_generator.last_metric_clamp_summary
    assert summary is not None
    assert summary["frame_names"] == [path.stem for path in frame_files]
    assert summary["clamped_fractions"] == [0.0, 1.0, 1.0]
    assert summary["affected_frame_count"] == 2
    assert summary["mean_clamped_fraction"] == pytest.approx(2.0 / 3.0)
    assert summary["max_clamped_fraction"] == 1.0

    corrupt_sidecar = directories["left_frames"] / "clamp_stats" / "frame_0002.json"
    corrupt_sidecar.write_text("{}", encoding="utf-8")
    second_renderer = _RecordingCommonRenderer()
    second_generator = StereoPairGenerator(renderer=second_renderer)
    assert second_generator.create_stereo_pairs_from_files(
        frame_files, metric_files, directories, settings
    )
    assert len(second_renderer.geometry_calls) == 1
    repaired = json.loads(corrupt_sidecar.read_text(encoding="utf-8"))
    assert repaired["frame_name"] == "frame_0002"
