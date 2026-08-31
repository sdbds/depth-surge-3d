"""Tests for the bounded stereo rendering and I/O pipeline."""

from __future__ import annotations

from dataclasses import replace
import gc
import json
import hashlib
import threading
import time
import weakref
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.processing.frames import stereo_generator
from src.depth_surge_3d.io.durable_publish import (
    OwnedSiblingTemporary,
    OwnedTemporaryCollisionError,
    publication_temporary_path,
)
from src.depth_surge_3d.io.file_identity import capture_directory_identity
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
from src.depth_surge_3d.processing.frames.stereo_input_manifest import (
    publish_validated_stereo_input_manifest_v1,
)
from src.depth_surge_3d.rendering.stereo_geometry import StereoGeometryFrame
from src.depth_surge_3d.rendering.stereo_renderer import (
    StereoRenderResult,
    StereoRenderer,
    StereoRenderSettings,
    StereoSplatSettings,
)
from src.depth_surge_3d.rendering.quality.types import (
    MetricStereoPrimitiveInput,
    RelativeStereoPrimitiveInput,
)


def test_stereo_algorithm_version_identifies_production_sixteen_sample_zbuffer() -> None:
    assert stereo_generator.STEREO_STAGE_ALGORITHM_VERSION == "torch-horizontal-16x-zbuffer-v3"
    assert stereo_generator.QUALITY_STEREO_ALGORITHM_VERSION == (
        "torch-horizontal-16x-rgb-geodesic-repair-v10"
    )


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


def _make_manifest_inputs(
    root: Path,
    *,
    count: int = 1,
    repeat_after_first: bool = False,
):
    frame_dir = root / "00_original_frames"
    depth_dir = root / "03_disparity_maps"
    left_dir = root / "04_left_frames"
    right_dir = root / "04_right_frames"
    for directory in (frame_dir, depth_dir, left_dir, right_dir):
        directory.mkdir(parents=True)
    frames: list[Path] = []
    depths: list[Path] = []
    for index in range(1, count + 1):
        payload_index = 1 if repeat_after_first else index
        frame = frame_dir / f"frame_{index:06d}.png"
        depth = depth_dir / f"frame_{index:06d}.png"
        assert cv2.imwrite(
            str(frame),
            np.full((8, 16, 3), 19 + payload_index, dtype=np.uint8),
        )
        assert cv2.imwrite(
            str(depth),
            np.full((4, 8), 32767 + payload_index, dtype=np.uint16),
        )
        frames.append(frame)
        depths.append(depth)
    _write_canonical_metadata(depth_dir, [frame.name for frame in frames], (4, 8))
    manifest = publish_validated_stereo_input_manifest_v1(
        job_root=root,
        guide_files=frames,
        geometry_files=depths,
        geometry_mode="relative",
    )
    return manifest, {"base": root, "left_frames": left_dir, "right_frames": right_dir}


def _make_metric_manifest_inputs(
    root: Path,
    *,
    count: int = 2,
    repeat_after_first: bool = False,
):
    frame_dir = root / "00_original_frames"
    metric_dir = root / "03_metric_geometry"
    left_dir = root / "04_left_frames"
    right_dir = root / "04_right_frames"
    for directory in (frame_dir, metric_dir, left_dir, right_dir):
        directory.mkdir(parents=True)
    frames: list[Path] = []
    for index in range(count):
        payload_index = 0 if repeat_after_first else index
        frame = frame_dir / f"frame_{index:06d}.png"
        assert cv2.imwrite(
            str(frame),
            np.full((4, 8, 3), 20 + payload_index, dtype=np.uint8),
        )
        frames.append(frame)
    store = MetricGeometryStore.open_or_create(
        metric_dir,
        frame_names=[path.name for path in frames],
        native_shape=(2, 4),
        source_raw_fingerprint="raw-metric",
        source_frame_fingerprint="frames",
        candidate_scene_fingerprint="scenes",
        preflight_required_bytes=0,
    )
    for index, frame in enumerate(frames):
        payload_index = 0 if repeat_after_first else index
        store.write_frame(
            frame.name,
            MetricGeometryFrame(
                np.full((2, 4), 1.0 / (payload_index + 1), dtype=np.float32),
                np.ones((2, 4), dtype=np.bool_),
                np.float32(0.5),
            ),
        )
    store.finalize(ClipConvergence(np.float32(2.0), (0,), 8))
    manifest = publish_validated_stereo_input_manifest_v1(
        job_root=root,
        guide_files=frames,
        geometry_files=list(store.complete_files),
        geometry_mode="metric_camera",
    )
    return manifest, {"base": root, "left_frames": left_dir, "right_frames": right_dir}


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


def test_generator_accepts_durable_stereo_input_manifest(tmp_path: Path) -> None:
    manifest, directories = _make_manifest_inputs(tmp_path)
    generator = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))

    assert generator.create_stereo_pairs_from_manifest(
        manifest,
        directories,
        _settings(stereo_strength=0.0, stereo_render_mode="fast"),
    )

    assert (directories["left_frames"] / "frame_000001.png").is_file()
    assert (directories["right_frames"] / "frame_000001.png").is_file()


def test_manifest_pipeline_never_rebuilds_eager_input_lists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, directories = _make_manifest_inputs(tmp_path)
    generator = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("manifest pipeline delegated to the eager list API")

    monkeypatch.setattr(generator, "create_stereo_pairs_from_files", forbidden)

    assert generator.create_stereo_pairs_from_manifest(
        manifest,
        directories,
        _settings(stereo_strength=0.0, stereo_render_mode="fast"),
    )


def test_manifest_pipeline_transfers_the_provider_item_without_copying_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, directories = _make_manifest_inputs(tmp_path)
    generator = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))
    original_run = stereo_generator.run_streaming_stereo_pipeline

    def observing_run(**kwargs):
        original_decode = kwargs["decode"]

        def decode(item):
            payload = original_decode(item)
            assert payload.work is item
            return payload

        kwargs["decode"] = decode
        return original_run(**kwargs)

    monkeypatch.setattr(stereo_generator, "run_streaming_stereo_pipeline", observing_run)

    assert generator.create_stereo_pairs_from_manifest(
        manifest,
        directories,
        _settings(stereo_strength=0.0, stereo_render_mode="fast"),
    )


def test_manifest_pipeline_supports_quality_with_committed_diagnostics(tmp_path: Path) -> None:
    manifest, directories = _make_manifest_inputs(tmp_path)
    generator = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))

    assert generator.create_stereo_pairs_from_manifest(
        manifest,
        directories,
        _settings(
            stereo_strength=0.0,
            stereo_render_mode="quality",
            occlusion_fill_max_px=8,
            keep_intermediates=True,
        ),
    )

    diagnostics = tmp_path / "04_stereo_diagnostics"
    assert (diagnostics / "frames" / "frame_000001" / "manifest.json").is_file()
    assert (diagnostics / "stereo_coverage_summary.json").is_file()
    assert (
        json.loads((diagnostics / "metadata.json").read_text(encoding="ascii"))["status"]
        == "complete"
    )


@pytest.mark.parametrize("render_mode", ["fast", "quality"])
def test_relative_manifest_streams_ordered_pixels_without_rgb_frame_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    render_mode: str,
) -> None:
    baseline_root = tmp_path / "baseline"
    baseline_manifest, baseline_directories = _make_manifest_inputs(
        baseline_root,
        count=2,
    )
    settings = _settings(
        stereo_strength=2.0,
        stereo_render_mode=render_mode,
        occlusion_fill_max_px=8,
        keep_intermediates=True,
    )
    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_manifest(
        baseline_manifest,
        baseline_directories,
        settings,
    )
    expected = [
        (
            cv2.imread(str(baseline_directories["left_frames"] / f"frame_{index:06d}.png")),
            cv2.imread(str(baseline_directories["right_frames"] / f"frame_{index:06d}.png")),
        )
        for index in (1, 2)
    ]

    streamed_root = tmp_path / "streamed"
    manifest, directories = _make_manifest_inputs(streamed_root, count=2)
    consumed: list[tuple[str, np.ndarray, np.ndarray]] = []

    def consume(frame_name: str, left: np.ndarray, right: np.ndarray) -> None:
        consumed.append((frame_name, left.copy(), right.copy()))

    class Progress:
        @staticmethod
        def update_progress(*_args, **_kwargs) -> None:
            return None

        @staticmethod
        def send_preview_frame(*_args, **_kwargs) -> None:
            raise AssertionError("transient stereo path exposed a nonexistent RGB file")

    monkeypatch.setattr(
        stereo_generator,
        "publish_quality_frame",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("transient stereo path published a per-frame transaction")
        ),
    )
    generator = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))
    assert generator.create_stereo_pairs_from_manifest(
        manifest,
        directories,
        _settings(
            stereo_strength=2.0,
            stereo_render_mode=render_mode,
            occlusion_fill_max_px=8,
            keep_intermediates=False,
        ),
        Progress(),
        frame_consumer=consume,
    )

    assert [name for name, _left, _right in consumed] == [
        "frame_000001",
        "frame_000002",
    ]
    for (_name, left, right), (expected_left, expected_right) in zip(
        consumed,
        expected,
        strict=True,
    ):
        np.testing.assert_array_equal(left, expected_left)
        np.testing.assert_array_equal(right, expected_right)
    assert not list(directories["left_frames"].glob("*.png"))
    assert not list(directories["right_frames"].glob("*.png"))
    diagnostics = streamed_root / "04_stereo_diagnostics"
    assert not (diagnostics / "frames").exists()
    assert (diagnostics / "quality_input_manifest.json").is_file() == (render_mode == "quality")
    assert len((diagnostics / "stereo_coverage_frames.jsonl").read_bytes().splitlines()) == 2
    metadata = json.loads((diagnostics / "metadata.json").read_text(encoding="ascii"))
    assert metadata["status"] == "complete"
    assert metadata["render_mode"] == render_mode
    assert generator.last_pipeline_stats is not None
    assert generator.last_pipeline_stats.writer_thread_count == 1
    assert generator.last_pipeline_stats.png_encoder_thread_count == 0


def test_transient_quality_capacity_charges_declared_consumer_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, directories = _make_manifest_inputs(tmp_path, count=2)
    real_select = stereo_generator.select_quality_pipeline_capacity
    retained_charges: list[int] = []

    def record_capacity_selection(**kwargs):
        retained_charges.append(kwargs.get("retained_bytes", 0))
        return real_select(**kwargs)

    monkeypatch.setattr(
        stereo_generator,
        "select_quality_pipeline_capacity",
        record_capacity_selection,
    )
    consumer_bytes = 7 * 1024 * 1024

    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_manifest(
        manifest,
        directories,
        _settings(
            stereo_strength=2.0,
            stereo_render_mode="quality",
            occlusion_fill_max_px=8,
            keep_intermediates=False,
        ),
        frame_consumer=lambda _name, _left, _right: None,
        frame_consumer_memory_bytes=consumer_bytes,
    )

    assert retained_charges == [consumer_bytes]


def test_transient_fast_capacity_charges_declared_consumer_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, directories = _make_manifest_inputs(tmp_path, count=2)
    real_select = stereo_generator.select_fast_pipeline_capacity
    retained_charges: list[int] = []

    def record_capacity_selection(**kwargs):
        retained_charges.append(kwargs.get("retained_bytes", 0))
        return real_select(**kwargs)

    monkeypatch.setattr(
        stereo_generator,
        "select_fast_pipeline_capacity",
        record_capacity_selection,
    )
    consumer_bytes = 7 * 1024 * 1024

    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_manifest(
        manifest,
        directories,
        _settings(
            stereo_strength=2.0,
            stereo_render_mode="fast",
            keep_intermediates=False,
        ),
        frame_consumer=lambda _name, _left, _right: None,
        frame_consumer_memory_bytes=consumer_bytes,
    )

    assert retained_charges == [consumer_bytes]


def test_quality_manifest_prepares_relative_geometry_on_decode_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, directories = _make_manifest_inputs(tmp_path, count=2)
    main_thread = threading.get_ident()
    prepare_threads: list[int] = []
    prepare_lock = threading.Lock()
    real_prepare = stereo_generator.prepare_relative_quality_frame

    def observe_prepare(**kwargs):
        with prepare_lock:
            prepare_threads.append(threading.get_ident())
        return real_prepare(**kwargs)

    monkeypatch.setattr(
        stereo_generator,
        "prepare_relative_quality_frame",
        observe_prepare,
    )
    generator = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))

    assert generator.create_stereo_pairs_from_manifest(
        manifest,
        directories,
        _settings(
            stereo_strength=2.0,
            stereo_render_mode="quality",
            occlusion_fill_max_px=8,
        ),
    )

    assert len(prepare_threads) == manifest.frame_count
    assert main_thread not in prepare_threads
    assert generator.last_pipeline_stats is not None
    assert generator.last_pipeline_stats.configured_capacity == 2
    assert generator.last_pipeline_stats.decoder_thread_count == 2
    assert generator.last_pipeline_stats.writer_thread_count == 2


def test_quality_manifest_reuses_exact_consecutive_input_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, directories = _make_manifest_inputs(
        tmp_path,
        count=3,
        repeat_after_first=True,
    )
    real_render = stereo_generator.render_relative_quality_compact
    real_select = stereo_generator.select_quality_pipeline_capacity
    render_calls = 0
    retained_charges: list[int] = []

    def count_render(**kwargs):
        nonlocal render_calls
        render_calls += 1
        return real_render(**kwargs)

    def record_capacity_selection(**kwargs):
        retained_charges.append(kwargs.get("retained_bytes", 0))
        return real_select(**kwargs)

    monkeypatch.setattr(
        stereo_generator,
        "render_relative_quality_compact",
        count_render,
    )
    monkeypatch.setattr(
        stereo_generator,
        "select_quality_pipeline_capacity",
        record_capacity_selection,
    )

    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_manifest(
        manifest,
        directories,
        _settings(
            stereo_strength=2.0,
            stereo_render_mode="quality",
            occlusion_fill_max_px=8,
            keep_intermediates=True,
            stereo_io_workers=1,
        ),
    )

    assert render_calls == 1
    assert len(retained_charges) == 1
    assert retained_charges[0] > 0
    left_payloads = [
        (directories["left_frames"] / f"frame_{index:06d}.png").read_bytes()
        for index in range(1, 4)
    ]
    assert left_payloads[0] == left_payloads[1] == left_payloads[2]
    for index in range(1, 4):
        frame_name = f"frame_{index:06d}"
        frame_manifest = json.loads(
            (
                tmp_path / "04_stereo_diagnostics" / "frames" / frame_name / "manifest.json"
            ).read_text(encoding="ascii")
        )
        assert frame_manifest["frame_name"] == frame_name


def test_quality_manifest_reuses_exact_input_across_overlapped_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    manifest, directories = _make_manifest_inputs(
        tmp_path,
        count=6,
        repeat_after_first=True,
    )
    real_prepare = stereo_generator.prepare_relative_quality_frame
    real_render = stereo_generator.render_prepared_relative_quality_compact
    prepare_calls = 0
    render_calls = 0
    png_encode_calls = 0
    calls_lock = threading.Lock()
    real_png_encode = diagnostic_transaction._png_buffer

    def count_prepare(**kwargs):
        nonlocal prepare_calls
        with calls_lock:
            prepare_calls += 1
        return real_prepare(**kwargs)

    def count_render(**kwargs):
        nonlocal render_calls
        with calls_lock:
            render_calls += 1
        return real_render(**kwargs)

    def count_png_encode(values):
        nonlocal png_encode_calls
        with calls_lock:
            png_encode_calls += 1
        return real_png_encode(values)

    monkeypatch.setattr(
        stereo_generator,
        "prepare_relative_quality_frame",
        count_prepare,
    )
    monkeypatch.setattr(
        stereo_generator,
        "render_prepared_relative_quality_compact",
        count_render,
    )
    monkeypatch.setattr(diagnostic_transaction, "_png_buffer", count_png_encode)
    generator = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))

    assert generator.create_stereo_pairs_from_manifest(
        manifest,
        directories,
        _settings(
            stereo_strength=2.0,
            stereo_render_mode="quality",
            occlusion_fill_max_px=8,
            keep_intermediates=True,
            stereo_io_workers=8,
        ),
    )

    assert generator.last_pipeline_stats is not None
    assert generator.last_pipeline_stats.configured_capacity == 4
    assert generator.last_pipeline_stats.decoder_thread_count == 4
    assert generator.last_pipeline_stats.writer_thread_count == 3
    assert 1 <= prepare_calls <= generator.last_pipeline_stats.configured_capacity
    assert render_calls == 1
    assert png_encode_calls == 6
    assert generator.last_rendered_frame_count == 1
    for side in ("left_frames", "right_frames"):
        payloads = [
            (directories[side] / f"frame_{index:06d}.png").read_bytes() for index in range(1, 7)
        ]
        assert len(set(payloads)) == 1


def test_quality_file_pipeline_parallelizes_png_encoding_beyond_writers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    manifest, directories = _make_manifest_inputs(tmp_path, count=4)
    real_encode = diagnostic_transaction._png_buffer
    active = 0
    peak = 0
    lock = threading.Lock()

    def observe_encode(values):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.02)
            return real_encode(values)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(diagnostic_transaction, "_png_buffer", observe_encode)
    generator = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))

    assert generator.create_stereo_pairs_from_manifest(
        manifest,
        directories,
        _settings(
            stereo_render_mode="quality",
            occlusion_fill_max_px=8,
            keep_intermediates=True,
            stereo_io_workers=8,
        ),
    )

    stats = generator.last_pipeline_stats
    assert stats is not None
    assert stats.png_encoder_thread_count == 4
    assert stats.actual_thread_count == (
        stats.feeder_thread_count
        + stats.decoder_thread_count
        + stats.writer_thread_count
        + stats.png_encoder_thread_count
    )
    assert stats.calculated_thread_bytes == (
        (stats.actual_thread_count - stats.png_encoder_thread_count) * 2 * 1024 * 1024
        + stats.png_encoder_thread_count * 4 * 1024 * 1024
    )
    assert peak > stats.writer_thread_count
    assert not [
        thread.name for thread in threading.enumerate() if thread.name.startswith("quality-png")
    ]


def test_quality_png_encoder_failure_joins_pool_without_publishing_a_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    manifest, directories = _make_manifest_inputs(tmp_path, count=2)

    def fail_encode(_values):
        raise OSError("injected pipeline PNG encode failure")

    monkeypatch.setattr(diagnostic_transaction, "_png_buffer", fail_encode)
    generator = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))

    with pytest.raises(OSError, match="injected pipeline PNG encode failure"):
        generator.create_stereo_pairs_from_manifest(
            manifest,
            directories,
            _settings(
                stereo_render_mode="quality",
                occlusion_fill_max_px=8,
                keep_intermediates=True,
                stereo_io_workers=8,
            ),
        )

    assert not list(directories["left_frames"].glob("*.png"))
    assert not list(directories["right_frames"].glob("*.png"))
    assert not list((tmp_path / "04_stereo_diagnostics" / "frames").rglob("manifest.json"))
    assert not [
        thread.name for thread in threading.enumerate() if thread.name.startswith("quality-png")
    ]


def test_quality_reuse_revalidates_each_duplicate_input_after_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, directories = _make_manifest_inputs(
        tmp_path,
        count=3,
        repeat_after_first=True,
    )
    original_publish = stereo_generator.publish_quality_frame
    published = 0

    def mutate_next_input(publication, **kwargs):
        nonlocal published
        result = original_publish(publication, **kwargs)
        published += 1
        if published == 1:
            (tmp_path / "00_original_frames" / "frame_000002.png").write_bytes(b"changed")
        return result

    monkeypatch.setattr(stereo_generator, "publish_quality_frame", mutate_next_input)

    with pytest.raises(stereo_generator.StereoStageInputChangedError):
        StereoPairGenerator(
            renderer=StereoRenderer(device="cpu")
        ).create_stereo_pairs_from_manifest(
            manifest,
            directories,
            _settings(
                stereo_strength=2.0,
                stereo_render_mode="quality",
                occlusion_fill_max_px=8,
                keep_intermediates=False,
                stereo_io_workers=1,
            ),
        )

    assert published == 1


def test_quality_manifest_reuses_exact_metric_input_across_overlapped_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, directories = _make_metric_manifest_inputs(
        tmp_path,
        count=5,
        repeat_after_first=True,
    )
    real_render = stereo_generator.render_metric_primitives_compact
    render_calls = 0

    def count_render(**kwargs):
        nonlocal render_calls
        render_calls += 1
        return real_render(**kwargs)

    monkeypatch.setattr(
        stereo_generator,
        "render_metric_primitives_compact",
        count_render,
    )
    generator = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))

    assert generator.create_stereo_pairs_from_manifest(
        manifest,
        directories,
        _settings(
            stereo_geometry_mode="metric_camera",
            stereo_render_mode="quality",
            metric_convergence_distance="auto",
            max_disparity_percent=2.0,
            keep_intermediates=False,
        ),
    )

    assert generator.last_pipeline_stats is not None
    assert generator.last_pipeline_stats.configured_capacity == 2
    assert render_calls == 1
    assert generator.last_rendered_frame_count == 1
    for side in ("left_frames", "right_frames"):
        payloads = [
            (directories[side] / f"frame_{index:06d}.png").read_bytes() for index in range(5)
        ]
        assert len(set(payloads)) == 1


def test_mask_pruning_resumes_after_all_migration_manifests_were_committed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, directories = _make_manifest_inputs(tmp_path)
    settings = _settings(
        stereo_strength=0.0,
        stereo_render_mode="quality",
        occlusion_fill_max_px=8,
        keep_intermediates=True,
    )
    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_manifest(
        manifest,
        directories,
        settings,
    )
    original_prune = stereo_generator.prune_streaming_unreferenced_masks

    def fail_prune(_context):
        raise OSError("crash before mask pruning")

    monkeypatch.setattr(stereo_generator, "prune_streaming_unreferenced_masks", fail_prune)
    settings["keep_intermediates"] = False
    with pytest.raises(OSError, match="crash before mask pruning"):
        StereoPairGenerator(
            renderer=StereoRenderer(device="cpu")
        ).create_stereo_pairs_from_manifest(
            manifest,
            directories,
            settings,
        )
    monkeypatch.setattr(
        stereo_generator,
        "prune_streaming_unreferenced_masks",
        original_prune,
    )

    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_manifest(
        manifest,
        directories,
        settings,
    )

    frame_root = tmp_path / "04_stereo_diagnostics" / "frames"
    assert not list(frame_root.glob("*/*_coverage.png"))
    assert not list(frame_root.glob("*/*_repair.png"))


def test_mask_migration_retires_a_crash_residue_before_republishing(
    tmp_path: Path,
) -> None:
    manifest, directories = _make_manifest_inputs(tmp_path)
    settings = _settings(
        stereo_strength=0.0,
        stereo_render_mode="quality",
        occlusion_fill_max_px=8,
        keep_intermediates=True,
    )
    generator = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))
    assert generator.create_stereo_pairs_from_manifest(manifest, directories, settings)
    committed = tmp_path / "04_stereo_diagnostics" / "frames" / "frame_000001" / "manifest.json"
    residue = stereo_generator.publication_temporary_path(committed)
    residue.write_bytes(committed.read_bytes())

    assert generator.create_stereo_pairs_from_manifest(
        manifest,
        directories,
        {**settings, "keep_intermediates": False},
    )

    assert not residue.exists()
    migrated = json.loads(committed.read_text(encoding="ascii"))
    assert migrated["mask_payloads_enabled"] is False


def test_streaming_progress_advances_after_each_committed_render_then_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, directories = _make_manifest_inputs(tmp_path, count=2)
    base = _settings(
        stereo_strength=0.0,
        stereo_render_mode="quality",
        keep_intermediates=True,
    )
    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_manifest(
        manifest,
        directories,
        base,
    )
    (directories["left_frames"] / "frame_000001.png").write_bytes(b"damaged")
    events: list[tuple[int, int, int]] = []
    monkeypatch.setattr(
        StereoPairGenerator,
        "_report_progress",
        staticmethod(
            lambda _tracker, **values: events.append(
                (values["processed"], values["total"], values["work_index"])
            )
        ),
    )

    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_manifest(
        manifest,
        directories,
        {**base, "keep_intermediates": False},
    )

    assert events == [(1, 2, 0), (2, 2, 1)]


def test_streaming_progress_reports_manifest_only_migrations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, directories = _make_manifest_inputs(tmp_path, count=2)
    base = _settings(
        stereo_strength=0.0,
        stereo_render_mode="quality",
        keep_intermediates=True,
    )
    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_manifest(
        manifest,
        directories,
        base,
    )
    events: list[tuple[int, int, int]] = []
    monkeypatch.setattr(
        StereoPairGenerator,
        "_report_progress",
        staticmethod(
            lambda _tracker, **values: events.append(
                (values["processed"], values["total"], values["work_index"])
            )
        ),
    )

    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_manifest(
        manifest,
        directories,
        {**base, "keep_intermediates": False},
    )

    assert events == [(1, 2, 0), (2, 2, 1)]


def test_streaming_progress_does_not_advance_for_failed_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, directories = _make_manifest_inputs(tmp_path, count=2)
    base = _settings(
        stereo_strength=0.0,
        stereo_render_mode="quality",
        keep_intermediates=True,
    )
    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_manifest(
        manifest,
        directories,
        base,
    )
    original_migrate = stereo_generator.migrate_streaming_frame_manifest
    migrations = 0

    def fail_second(context, frame_name, **kwargs):
        nonlocal migrations
        migrations += 1
        if migrations == 2:
            raise OSError("injected second migration failure")
        return original_migrate(context, frame_name, **kwargs)

    events: list[int] = []
    monkeypatch.setattr(stereo_generator, "migrate_streaming_frame_manifest", fail_second)
    monkeypatch.setattr(
        StereoPairGenerator,
        "_report_progress",
        staticmethod(lambda _tracker, **values: events.append(values["processed"])),
    )

    error: OSError | None = None
    try:
        StereoPairGenerator(
            renderer=StereoRenderer(device="cpu")
        ).create_stereo_pairs_from_manifest(
            manifest,
            directories,
            {**base, "keep_intermediates": False},
        )
    except OSError as caught:
        error = caught

    assert migrations == 2
    assert error is not None and "second migration failure" in str(error)
    assert events == [1]


@pytest.mark.parametrize("root_damage", ["missing", "corrupt", "wrong_generation"])
def test_unclaimed_diagnostics_root_never_reuses_orphan_frame_transactions(
    tmp_path: Path,
    root_damage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, directories = _make_manifest_inputs(tmp_path)
    settings = _settings(
        stereo_strength=0.0,
        stereo_render_mode="quality",
        occlusion_fill_max_px=8,
        keep_intermediates=False,
    )
    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_manifest(
        manifest,
        directories,
        settings,
    )
    metadata_path = tmp_path / "04_stereo_diagnostics" / "metadata.json"
    if root_damage == "missing":
        metadata_path.unlink()
    elif root_damage == "corrupt":
        metadata_path.write_bytes(b"not canonical JSON")
    else:
        from src.depth_surge_3d.rendering.quality import diagnostic_transaction

        metadata = json.loads(metadata_path.read_text(encoding="ascii"))
        metadata["rgb_stage_fingerprint"] = "different-generation"
        metadata["fingerprint"] = diagnostic_transaction.self_fingerprint(metadata)
        metadata_path.write_bytes(diagnostic_transaction.canonical_json_bytes(metadata))
    diagnostics_root = tmp_path / "04_stereo_diagnostics"
    residue = diagnostics_root / "unknown-old-generation.bin"
    residue.write_bytes(b"must not cross generations")
    previous_root_identity = capture_directory_identity(diagnostics_root)
    original_publish = stereo_generator.StreamedQualityInputManifest.publish
    publication_observations: list[tuple[bool, bool]] = []

    def publish_after_root_retirement(self, path):
        publication_observations.append((not diagnostics_root.exists(), not residue.exists()))
        return original_publish(self, path)

    monkeypatch.setattr(
        stereo_generator.StreamedQualityInputManifest,
        "publish",
        publish_after_root_retirement,
    )

    generator = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))
    assert generator.create_stereo_pairs_from_manifest(manifest, directories, settings)

    assert publication_observations == [(True, True)]
    assert not residue.exists()
    assert capture_directory_identity(diagnostics_root) != previous_root_identity
    assert generator.last_pipeline_stats is not None
    assert generator.last_pipeline_stats.decoded == manifest.frame_count


@pytest.mark.parametrize("render_mode", ["fast", "quality"])
def test_manifest_pipeline_persists_stage_performance_report(
    tmp_path: Path,
    render_mode: str,
) -> None:
    manifest, directories = _make_manifest_inputs(tmp_path, count=2)
    settings = _settings(
        stereo_strength=0.0,
        stereo_render_mode=render_mode,
        keep_intermediates=False,
    )
    generator = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))

    assert generator.create_stereo_pairs_from_manifest(manifest, directories, settings)

    report = json.loads((tmp_path / "stereo-performance-v1.json").read_text())
    assert report["schema_version"] == 1
    assert report["render_mode"] == render_mode
    assert report["pipeline"]["pipeline_wall_seconds"] > 0.0
    assert report["pipeline"]["render_busy_seconds"] > 0.0
    assert report["rendered_frame_count"] == manifest.frame_count
    assert report["render_seconds"]["total"] > 0.0
    assert report["stage_seconds"]["preflight"] >= 0.0
    assert report["stage_seconds"]["pipeline"] > 0.0
    assert report["stage_seconds"]["consolidation"] >= 0.0
    assert report["stage_seconds"]["total"] >= sum(
        report["stage_seconds"][name] for name in ("preflight", "pipeline", "consolidation")
    )
    if render_mode == "quality":
        assert report["render_seconds"]["geometry_nearest_query"] >= 0.0
        assert report["pipeline"]["configured_capacity"] == min(2, manifest.frame_count)
    else:
        assert report["render_seconds"]["fast_geometry"] >= 0.0


def test_downstream_is_invalidated_before_the_first_stereo_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, directories = _make_manifest_inputs(tmp_path)
    settings = _settings(
        stereo_strength=0.0,
        stereo_render_mode="quality",
        keep_intermediates=False,
    )
    generator = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))
    assert generator.create_stereo_pairs_from_manifest(manifest, directories, settings)
    (directories["left_frames"] / "frame_000001.png").write_bytes(b"damaged")
    downstream = []
    for name in (
        "left_distorted",
        "right_distorted",
        "left_cropped",
        "right_cropped",
        "left_upscaled",
        "right_upscaled",
        "vr_frames",
    ):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "stale.bin").write_bytes(b"stale")
        directories[name] = directory
        downstream.append(directory)

    def fail_first_stereo_deletion(self, root, child, *, evidence):
        assert all(not any(directory.iterdir()) for directory in downstream)
        raise OSError("injected first stereo deletion failure")

    monkeypatch.setattr(
        stereo_generator.SafeTreeMutation,
        "unlink_child",
        fail_first_stereo_deletion,
    )

    with pytest.raises(OSError, match="first stereo deletion"):
        generator.create_stereo_pairs_from_manifest(manifest, directories, settings)

    assert all(not any(directory.iterdir()) for directory in downstream)


def test_downstream_invalidation_failure_leaves_stereo_generation_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, directories = _make_manifest_inputs(tmp_path)
    settings = _settings(
        stereo_strength=0.0,
        stereo_render_mode="quality",
        keep_intermediates=False,
    )
    generator = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))
    assert generator.create_stereo_pairs_from_manifest(manifest, directories, settings)
    (directories["left_frames"] / "frame_000001.png").write_bytes(b"damaged")
    diagnostics_root = tmp_path / "04_stereo_diagnostics"
    protected_roots = (
        directories["left_frames"],
        directories["right_frames"],
        diagnostics_root,
    )
    before = {
        (root.name, str(path.relative_to(root))): path.read_bytes()
        for root in protected_roots
        for path in root.rglob("*")
        if path.is_file()
    }
    downstream = tmp_path / "05_left_distorted"
    downstream.mkdir()
    (downstream / "stale.bin").write_bytes(b"stale")
    directories["left_distorted"] = downstream
    original_clear = stereo_generator.SafeTreeMutation.clear_root

    def fail_downstream_clear(self, root):
        if Path(root) == downstream:
            raise OSError("injected downstream invalidation failure")
        return original_clear(self, root)

    monkeypatch.setattr(
        stereo_generator.SafeTreeMutation,
        "clear_root",
        fail_downstream_clear,
    )

    with pytest.raises(OSError, match="downstream invalidation"):
        generator.create_stereo_pairs_from_manifest(manifest, directories, settings)

    after = {
        (root.name, str(path.relative_to(root))): path.read_bytes()
        for root in protected_roots
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize("damaged_payload", ["rgb", "stats", "manifest"])
def test_claimed_complete_root_rerenders_only_the_damaged_frame_transaction(
    tmp_path: Path,
    damaged_payload: str,
) -> None:
    manifest, directories = _make_manifest_inputs(tmp_path)
    settings = _settings(
        stereo_strength=0.0,
        stereo_render_mode="quality",
        occlusion_fill_max_px=8,
        keep_intermediates=False,
    )
    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_manifest(
        manifest,
        directories,
        settings,
    )
    frame_name = "frame_000001"
    frame_root = tmp_path / "04_stereo_diagnostics" / "frames" / frame_name
    damaged = {
        "rgb": directories["left_frames"] / f"{frame_name}.png",
        "stats": frame_root / "stats.json",
        "manifest": frame_root / "manifest.json",
    }[damaged_payload]
    damaged.write_bytes(b"damaged")

    generator = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))
    assert generator.create_stereo_pairs_from_manifest(manifest, directories, settings)

    assert generator.last_pipeline_stats is not None
    assert generator.last_pipeline_stats.decoded == 1
    assert generator.last_pipeline_stats.rendered == 1
    assert generator.last_pipeline_stats.written == 1


@pytest.mark.parametrize("boundary", ["content", "rgb_metadata", "building", "aggregate"])
def test_post_mutation_manifest_replay_change_is_always_typed_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    manifest, directories = _make_manifest_inputs(tmp_path)

    def changed(*_args, **_kwargs):
        raise stereo_generator.StereoInputIdentityMismatch(f"injected {boundary} replay change")

    render_mode = "quality" if boundary == "content" else "fast"
    if boundary == "content":
        monkeypatch.setattr(stereo_generator.StreamedQualityInputManifest, "publish", changed)
    elif boundary == "rgb_metadata":
        monkeypatch.setattr(stereo_generator.StreamedPrettyJsonValue, "publish", changed)
    elif boundary == "building":
        monkeypatch.setattr(stereo_generator, "begin_streaming_diagnostics_stage", changed)
    else:
        monkeypatch.setattr(stereo_generator, "complete_streaming_diagnostics_stage", changed)

    with pytest.raises(
        stereo_generator.StereoStageInputChangedError,
        match=f"injected {boundary} replay change",
    ):
        StereoPairGenerator(
            renderer=StereoRenderer(device="cpu")
        ).create_stereo_pairs_from_manifest(
            manifest,
            directories,
            _settings(
                stereo_strength=0.0,
                stereo_render_mode=render_mode,
                occlusion_fill_max_px=8,
                keep_intermediates=False,
            ),
        )


def test_real_manifest_replay_protocol_error_after_mutation_is_not_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, directories = _make_manifest_inputs(tmp_path)
    original_publish = stereo_generator.StreamedPrettyJsonValue.publish

    def change_manifest_then_publish(document, path, **kwargs):
        manifest.path.write_bytes(b"{}\n")
        return original_publish(document, path, **kwargs)

    monkeypatch.setattr(
        stereo_generator.StreamedPrettyJsonValue,
        "publish",
        change_manifest_then_publish,
    )

    class CountingGenerator(StereoPairGenerator):
        attempts = 0

        def _create_stereo_pairs_from_manifest_once(self, *args, **kwargs):
            self.attempts += 1
            return super()._create_stereo_pairs_from_manifest_once(*args, **kwargs)

    generator = CountingGenerator(renderer=StereoRenderer(device="cpu"))
    with pytest.raises(
        stereo_generator.StereoStageInputChangedError,
        match="stereo input manifest",
    ):
        generator.create_stereo_pairs_from_manifest(
            manifest,
            directories,
            _settings(
                stereo_strength=0.0,
                stereo_render_mode="fast",
                keep_intermediates=False,
            ),
        )

    assert generator.attempts == 1


def test_pre_mutation_provider_change_restarts_from_a_fresh_read_only_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, directories = _make_manifest_inputs(tmp_path)
    real_run = stereo_generator.run_streaming_stereo_pipeline
    real_open = stereo_generator.StereoInputManifestV1.open_provider
    run_calls = 0
    generations: list[str] = []

    def open_provider(self, **kwargs):
        provider = real_open(self, **kwargs)
        generations.append(provider.generation.audit_generation_id)
        return provider

    def run_pipeline(**kwargs):
        nonlocal run_calls
        run_calls += 1
        if run_calls == 1:
            assert not (directories["left_frames"] / "metadata.json").exists()
            assert not (tmp_path / "04_stereo_diagnostics" / "metadata.json").exists()
            raise stereo_generator.StereoProviderChangedError("injected read-only change")
        return real_run(**kwargs)

    monkeypatch.setattr(stereo_generator.StereoInputManifestV1, "open_provider", open_provider)
    monkeypatch.setattr(stereo_generator, "run_streaming_stereo_pipeline", run_pipeline)

    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_manifest(
        manifest,
        directories,
        _settings(
            stereo_strength=0.0,
            stereo_render_mode="fast",
            keep_intermediates=False,
        ),
    )
    assert len(generations) == 2
    assert len(set(generations)) == 2


def test_reused_output_damage_is_reclassified_before_noop_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, directories = _make_manifest_inputs(tmp_path, count=2)
    settings = _settings(
        stereo_strength=0.0,
        stereo_render_mode="quality",
        keep_intermediates=False,
    )
    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_manifest(
        manifest,
        directories,
        settings,
    )
    real_aggregates = stereo_generator.streaming_diagnostics_aggregates_match
    aggregate_calls = 0

    def damage_after_initial_aggregate(context):
        nonlocal aggregate_calls
        aggregate_calls += 1
        result = real_aggregates(context)
        if aggregate_calls == 1:
            (directories["left_frames"] / "frame_000002.png").unlink()
        return result

    monkeypatch.setattr(
        stereo_generator,
        "streaming_diagnostics_aggregates_match",
        damage_after_initial_aggregate,
    )
    generator = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))

    assert generator.create_stereo_pairs_from_manifest(manifest, directories, settings)

    assert generator.last_pipeline_stats is not None
    assert generator.last_pipeline_stats.decoded == 1


def test_mixed_action_damage_restarts_before_the_first_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, directories = _make_manifest_inputs(tmp_path, count=2)
    settings = _settings(
        stereo_strength=0.0,
        stereo_render_mode="quality",
        keep_intermediates=False,
    )
    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_manifest(
        manifest,
        directories,
        settings,
    )
    (directories["right_frames"] / "frame_000002.png").write_bytes(b"damaged")
    real_disk_plan = stereo_generator._build_stereo_stage_disk_plan
    disk_calls = 0

    def damage_reused_item(**kwargs):
        nonlocal disk_calls
        disk_calls += 1
        result = real_disk_plan(**kwargs)
        if disk_calls == 1:
            (directories["left_frames"] / "frame_000001.png").unlink()
        return result

    real_begin = stereo_generator.begin_streaming_diagnostics_stage
    mutation_calls = 0

    def observe_begin(*args, **kwargs):
        nonlocal mutation_calls
        mutation_calls += 1
        return real_begin(*args, **kwargs)

    monkeypatch.setattr(stereo_generator, "_build_stereo_stage_disk_plan", damage_reused_item)
    monkeypatch.setattr(stereo_generator, "begin_streaming_diagnostics_stage", observe_begin)
    generator = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))

    assert generator.create_stereo_pairs_from_manifest(manifest, directories, settings)

    assert disk_calls >= 2
    assert mutation_calls == 1
    assert generator.last_pipeline_stats is not None
    assert generator.last_pipeline_stats.decoded == 2


def test_metric_semantic_content_audit_cannot_run_before_quality_stage_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, directories = _make_metric_manifest_inputs(tmp_path)
    semantic_audit_called = False

    def audit_payloads(_self) -> None:
        nonlocal semantic_audit_called
        semantic_audit_called = True
        raise AssertionError("semantic metric audit ran before the host gate")

    def reject_stage(**_kwargs):
        raise stereo_generator.QualityHostBudgetError.for_limit(
            "quality-input-content-audit",
            stereo_generator.STEREO_HOST_BUDGET + 1,
            stereo_generator.STEREO_HOST_BUDGET,
        )

    monkeypatch.setattr(stereo_generator.StereoInputManifestV1, "audit_payloads", audit_payloads)
    monkeypatch.setattr(stereo_generator, "_quality_stage_memory_preflight", reject_stage)

    with pytest.raises(stereo_generator.QualityHostBudgetError):
        StereoPairGenerator(
            renderer=StereoRenderer(device="cpu")
        ).create_stereo_pairs_from_manifest(
            manifest,
            directories,
            _settings(
                stereo_geometry_mode="metric_camera",
                stereo_render_mode="quality",
                metric_convergence_distance="auto",
                keep_intermediates=False,
            ),
        )
    assert semantic_audit_called is False


def test_quality_file_stage_gate_charges_compressed_input_threads_and_control(
    tmp_path: Path,
) -> None:
    manifest, directories = _make_manifest_inputs(tmp_path)
    provider = manifest.open_provider(
        left_dir=directories["left_frames"],
        right_dir=directories["right_frames"],
    )
    control = stereo_generator.audit_stereo_provider(
        provider,
        lambda _item: stereo_generator.StereoAction.RENDER,
    )
    compressed_bytes = stereo_generator.STEREO_HOST_BUDGET
    inflated = replace(
        control,
        maximum_compressed_payload_bytes=compressed_bytes,
    )
    output_pixels = manifest.render_shape[0] * manifest.render_shape[1]
    native_pixels = inflated.maximum_native_pixels
    expected = (
        3 * output_pixels
        + 2 * native_pixels
        + compressed_bytes
        + stereo_generator.QUALITY_RUNTIME_OVERHEAD_BYTES
        + stereo_generator.QUALITY_IO_THREAD_BYTES
        + stereo_generator.STEREO_CONTROL_OVERHEAD
    )

    with pytest.raises(
        stereo_generator.QualityHostBudgetError,
        match="quality-file-pipeline",
    ) as captured:
        stereo_generator._quality_stage_memory_preflight(
            control=inflated,
            render_shape=manifest.render_shape,
            native_shape=manifest.native_shape,
            geometry_mode="relative",
            occlusion_fill="background",
            consolidation_planned=True,
        )
    assert captured.value.required_bytes == expected


@pytest.mark.parametrize("render_mode", ["fast", "quality"])
def test_manifest_pipeline_supports_metric_fast_and_quality(
    tmp_path: Path,
    render_mode: str,
) -> None:
    manifest, directories = _make_metric_manifest_inputs(tmp_path)
    generator = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))

    assert generator.create_stereo_pairs_from_manifest(
        manifest,
        directories,
        _settings(
            stereo_geometry_mode="metric_camera",
            stereo_render_mode=render_mode,
            metric_convergence_distance="auto",
            max_disparity_percent=2.0,
            keep_intermediates=False,
        ),
    )

    for index in range(2):
        frame_name = f"frame_{index:06d}"
        assert (directories["left_frames"] / f"{frame_name}.png").is_file()
        manifest_path = tmp_path / "04_stereo_diagnostics" / "frames" / frame_name / "manifest.json"
        assert json.loads(manifest_path.read_text(encoding="ascii"))["geometry_mode"] == "metric"
    assert (directories["left_frames"] / "clamp_summary.json").is_file()
    assert generator.last_metric_clamp_summary is not None
    assert set(generator.last_metric_clamp_summary) == {
        "schema_version",
        "affected_frame_count",
        "mean_clamped_fraction",
        "max_clamped_fraction",
        "compatibility_summary_sha256",
    }


def test_quality_file_pipeline_uses_one_slot_and_publishes_compact_diagnostics(
    tmp_path: Path,
) -> None:
    frame_files, depth_files, directories = _make_file_inputs(
        tmp_path,
        count=1,
        frame_shape=(8, 16),
        depth_shape=(4, 8),
    )
    generator = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))

    assert generator.create_stereo_pairs_from_files(
        frame_files,
        depth_files,
        directories,
        _settings(
            stereo_render_mode="quality",
            occlusion_fill_max_px=8,
            stereo_strength=0.0,
            keep_intermediates=True,
        ),
    )

    assert generator.last_pipeline_stats is not None
    assert generator.last_pipeline_stats.permit_count == 1
    frame_dir = tmp_path / "04_stereo_diagnostics" / "frames" / "frame_0000"
    manifest = json.loads((frame_dir / "manifest.json").read_text(encoding="ascii"))
    assert manifest["render_mode"] == "quality"
    assert manifest["mask_payloads_enabled"] is True
    assert (frame_dir / "left_coverage.png").exists()
    assert (frame_dir / "right_repair.png").exists()
    rgb_metadata = json.loads(
        (directories["left_frames"] / "metadata.json").read_text(encoding="ascii")
    )
    assert rgb_metadata["schema_version"] == 2
    assert rgb_metadata["algorithm_version"] == ("torch-horizontal-16x-rgb-geodesic-repair-v10")
    diagnostics_metadata = json.loads(
        (tmp_path / "04_stereo_diagnostics" / "metadata.json").read_text(encoding="ascii")
    )
    assert diagnostics_metadata["status"] == "complete"
    assert (tmp_path / "04_stereo_diagnostics" / "stereo_coverage_summary.json").exists()


def test_writer_queue_keeps_serialized_stats_not_dense_histograms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_files, depth_files, directories = _make_file_inputs(
        tmp_path,
        count=1,
        frame_shape=(8, 16),
        depth_shape=(4, 8),
    )
    calling_thread = threading.get_ident()
    original_prepare = stereo_generator.prepare_quality_frame
    original_publish = stereo_generator.publish_quality_frame
    prepared_threads: list[int] = []
    published_threads: list[int] = []

    def prepare(publication):
        prepared_threads.append(threading.get_ident())
        return original_prepare(publication)

    def publish(publication):
        published_threads.append(threading.get_ident())
        assert not hasattr(publication, "compact")
        assert isinstance(publication.stats_raw, bytes)
        assert not any(
            isinstance(value, np.ndarray) and value.dtype == np.uint64 and value.ndim == 1
            for value in vars(publication).values()
        )
        return original_publish(publication)

    monkeypatch.setattr(stereo_generator, "prepare_quality_frame", prepare)
    monkeypatch.setattr(stereo_generator, "publish_quality_frame", publish)

    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_files(
        frame_files,
        depth_files,
        directories,
        _settings(
            stereo_render_mode="quality",
            occlusion_fill_max_px=8,
            stereo_strength=0.0,
            keep_intermediates=True,
        ),
    )
    assert prepared_threads == [calling_thread]
    assert len(published_threads) == 1
    assert published_threads[0] != calling_thread


def test_quality_prepare_failure_releases_its_lifecycle_permit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_files, depth_files, directories = _make_file_inputs(
        tmp_path,
        count=1,
        frame_shape=(8, 16),
        depth_shape=(4, 8),
    )
    generator = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))
    monkeypatch.setattr(
        stereo_generator,
        "prepare_quality_frame",
        lambda _publication: (_ for _ in ()).throw(RuntimeError("prepare failed")),
    )

    assert not generator.create_stereo_pairs_from_files(
        frame_files,
        depth_files,
        directories,
        _settings(
            stereo_render_mode="quality",
            occlusion_fill_max_px=8,
            stereo_strength=0.0,
            keep_intermediates=True,
        ),
    )

    assert generator.last_pipeline_stats is not None
    assert generator.last_pipeline_stats.active_permits == 0
    assert (
        generator.last_pipeline_stats.permits_acquired
        == generator.last_pipeline_stats.permits_released
    )
    assert not any(thread.name.startswith("stereo-") for thread in threading.enumerate())


def test_quality_resume_rebuilds_damaged_aggregates_without_rerendering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_files, depth_files, directories = _make_file_inputs(
        tmp_path,
        count=2,
        frame_shape=(6, 12),
        depth_shape=(3, 6),
    )
    settings = _settings(
        stereo_render_mode="quality",
        stereo_strength=0.0,
        keep_intermediates=False,
    )
    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_files(frame_files, depth_files, directories, settings)
    left_hashes = [
        hashlib.sha256((directories["left_frames"] / path.name).read_bytes()).hexdigest()
        for path in frame_files
    ]
    summary_path = tmp_path / "04_stereo_diagnostics" / "stereo_coverage_summary.json"
    summary_path.unlink()

    def forbidden_render(**_kwargs):
        raise AssertionError("aggregate repair must not rerender RGB")

    monkeypatch.setattr(
        stereo_generator,
        "render_relative_quality_compact",
        forbidden_render,
    )
    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_files(frame_files, depth_files, directories, settings)

    assert summary_path.exists()
    assert left_hashes == [
        hashlib.sha256((directories["left_frames"] / path.name).read_bytes()).hexdigest()
        for path in frame_files
    ]


def test_quality_resume_rerenders_only_the_invalid_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_files, depth_files, directories = _make_file_inputs(
        tmp_path,
        count=2,
        frame_shape=(6, 12),
        depth_shape=(3, 6),
    )
    settings = _settings(
        stereo_render_mode="quality",
        stereo_strength=0.0,
        keep_intermediates=False,
    )
    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_files(frame_files, depth_files, directories, settings)
    target = directories["left_frames"] / frame_files[1].name
    target.write_bytes(b"damaged")
    original = stereo_generator.render_relative_quality_compact
    rendered: list[int] = []

    def record_render(**kwargs):
        rendered.append(int(kwargs["frame"][0, 0, 0]))
        return original(**kwargs)

    monkeypatch.setattr(
        stereo_generator,
        "render_relative_quality_compact",
        record_render,
    )
    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_files(frame_files, depth_files, directories, settings)

    assert rendered == [21]


def test_quality_mask_true_to_false_migrates_manifests_without_rerendering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_files, depth_files, directories = _make_file_inputs(
        tmp_path,
        count=2,
        frame_shape=(6, 12),
        depth_shape=(3, 6),
    )
    base = _settings(
        stereo_render_mode="quality",
        stereo_strength=0.0,
        keep_intermediates=True,
    )
    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_files(frame_files, depth_files, directories, base)

    def forbidden_render(**_kwargs):
        raise AssertionError("mask removal must not rerender RGB")

    monkeypatch.setattr(
        stereo_generator,
        "render_relative_quality_compact",
        forbidden_render,
    )
    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_files(
        frame_files,
        depth_files,
        directories,
        {**base, "keep_intermediates": False},
    )

    for frame_file in frame_files:
        frame_dir = tmp_path / "04_stereo_diagnostics" / "frames" / frame_file.stem
        manifest = json.loads((frame_dir / "manifest.json").read_text(encoding="ascii"))
        assert manifest["mask_payloads_enabled"] is False
        assert manifest["payloads"]["left_coverage"] is None
        assert not (frame_dir / "left_coverage.png").exists()


def test_new_mask_free_stage_skips_unreferenced_mask_prune(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, directories = _make_manifest_inputs(tmp_path, count=2)

    def forbidden_prune(_context) -> None:
        raise AssertionError("a fresh mask-free stage has no masks to prune")

    monkeypatch.setattr(
        stereo_generator,
        "prune_streaming_unreferenced_masks",
        forbidden_prune,
    )

    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_manifest(
        manifest,
        directories,
        _settings(
            stereo_render_mode="fast",
            stereo_strength=0.0,
            keep_intermediates=False,
        ),
    )


def test_quality_mask_false_to_true_rerenders_every_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_files, depth_files, directories = _make_file_inputs(
        tmp_path,
        count=2,
        frame_shape=(6, 12),
        depth_shape=(3, 6),
    )
    base = _settings(
        stereo_render_mode="quality",
        stereo_strength=0.0,
        keep_intermediates=False,
    )
    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_files(frame_files, depth_files, directories, base)
    original = stereo_generator.render_relative_quality_compact
    rendered: list[int] = []

    def record_render(**kwargs):
        rendered.append(int(kwargs["frame"][0, 0, 0]))
        return original(**kwargs)

    monkeypatch.setattr(
        stereo_generator,
        "render_relative_quality_compact",
        record_render,
    )
    assert StereoPairGenerator(
        renderer=StereoRenderer(device="cpu")
    ).create_stereo_pairs_from_files(
        frame_files,
        depth_files,
        directories,
        {**base, "keep_intermediates": True},
    )

    assert rendered == [20, 21]


def test_new_fast_render_keeps_rgb_bytes_and_publishes_bounded_diagnostics(
    tmp_path: Path,
) -> None:
    frame_files, depth_files, directories = _make_file_inputs(
        tmp_path,
        count=1,
        frame_shape=(6, 12),
        depth_shape=(3, 6),
    )
    frame = cv2.imread(str(frame_files[0]), cv2.IMREAD_COLOR)
    encoded = cv2.imread(str(depth_files[0]), cv2.IMREAD_UNCHANGED)
    canonical = encoded.astype(np.float32) / np.float32(65535.0)
    renderer = StereoRenderer(device="cpu")
    expected = renderer.render(
        frame,
        canonical,
        StereoRenderSettings(
            stereo_strength=2.0,
            convergence=0.5,
            occlusion_fill="background",
        ),
    )

    assert StereoPairGenerator(renderer=renderer).create_stereo_pairs_from_files(
        frame_files,
        depth_files,
        directories,
        _settings(stereo_render_mode="fast", keep_intermediates=True),
    )

    assert np.array_equal(
        cv2.imread(str(directories["left_frames"] / frame_files[0].name)),
        expected.left_image,
    )
    frame_dir = tmp_path / "04_stereo_diagnostics" / "frames" / "frame_0000"
    manifest = json.loads((frame_dir / "manifest.json").read_text(encoding="ascii"))
    stats = json.loads((frame_dir / "stats.json").read_text(encoding="ascii"))
    assert manifest["render_mode"] == "fast"
    assert stats["geometry_nearest"] is None
    assert stats["eyes"]["left"]["quality_budgets"] is None


def test_relative_decoder_retains_owned_uint16_primitive_until_render_thread(
    tmp_path: Path,
) -> None:
    frame_files, depth_files, directories = _make_file_inputs(
        tmp_path,
        count=1,
        frame_shape=(8, 16),
        depth_shape=(4, 8),
    )
    work = stereo_generator._FileWorkItem(
        index=0,
        frame_path=frame_files[0],
        depth_path=depth_files[0],
        frame_name=frame_files[0].stem,
        left_path=directories["left_frames"] / frame_files[0].name,
        right_path=directories["right_frames"] / frame_files[0].name,
    )

    frame, primitive, stats = stereo_generator._decode_relative_work_item(
        work,
        encoding_scale=65535.0,
        render_shape=(8, 16),
    )

    assert frame.dtype == np.uint8
    assert isinstance(primitive, RelativeStereoPrimitiveInput)
    assert primitive.encoded_canonical.dtype == np.uint16
    assert primitive.encoded_canonical.flags.owndata
    assert stats is None


def test_metric_decoder_returns_native_primitive_without_render_geometry(
    tmp_path: Path,
) -> None:
    frame_files, metric_files, directories, resolved = _make_metric_inputs(tmp_path)
    generator = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))
    store, _metadata = generator._get_metric_store(metric_files, frame_files)
    work = stereo_generator._FileWorkItem(
        index=0,
        frame_path=frame_files[0],
        depth_path=metric_files[0],
        frame_name=frame_files[0].stem,
        left_path=directories["left_frames"] / frame_files[0].name,
        right_path=directories["right_frames"] / frame_files[0].name,
    )

    frame, primitive, stats = stereo_generator._decode_metric_work_item(
        work,
        store=store,
        render_shape=(4, 5),
        virtual_baseline_mm=63.0,
        convergence_distance_m=resolved,
        max_disparity_percent=2.0,
        retained_crop_width=8,
    )

    assert frame.dtype == np.uint8
    assert isinstance(primitive, MetricStereoPrimitiveInput)
    assert isinstance(primitive.metric, MetricGeometryFrame)
    assert stats is None


def test_stage_plans_are_four_discriminated_variants(tmp_path: Path) -> None:
    relative_frames, relative_depths, _ = _make_file_inputs(tmp_path / "relative", count=1)
    metric_frames, metric_depths, _, _ = _make_metric_inputs(
        tmp_path / "metric",
        inverse_values=[1.0],
    )
    generator = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))

    fast_relative = generator._build_stage_plan(
        relative_depths,
        relative_frames,
        (8, 8),
        _settings(stereo_render_mode="fast"),
        "background",
    )
    quality_relative = generator._build_stage_plan(
        relative_depths,
        relative_frames,
        (8, 8),
        _settings(stereo_render_mode="quality"),
        "background",
    )
    fast_metric = generator._build_stage_plan(
        metric_depths,
        metric_frames,
        (4, 5),
        _settings(stereo_geometry_mode="metric_camera", stereo_render_mode="fast"),
        "background",
    )
    quality_metric = generator._build_stage_plan(
        metric_depths,
        metric_frames,
        (4, 5),
        _settings(stereo_geometry_mode="metric_camera", stereo_render_mode="quality"),
        "background",
    )

    assert [
        fast_relative.kind,
        quality_relative.kind,
        fast_metric.kind,
        quality_metric.kind,
    ] == ["fast_relative", "quality_relative", "fast_metric", "quality_metric"]
    assert isinstance(quality_metric.settings, stereo_generator.QualityStereoControls)


def test_impossible_discriminated_plan_rejects_mismatched_settings() -> None:
    with pytest.raises(TypeError, match="StereoRenderSettings"):
        stereo_generator._FastRelativePlan(  # type: ignore[arg-type]
            source_metadata={},
            decode=lambda work: None,
            settings=StereoSplatSettings(max_eye_shift_fraction=0.01),
        )


class _FakeRenderer:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[int, object, object]] = []

    def render(
        self,
        frame: np.ndarray,
        canonical: np.ndarray,
        settings: StereoRenderSettings,
    ) -> StereoRenderResult:
        return self._record(frame, canonical, settings)

    def render_geometry(
        self,
        frame: np.ndarray,
        geometry: object,
        settings: StereoSplatSettings,
    ) -> StereoRenderResult:
        return self._record(frame, geometry, settings)

    def _record(
        self,
        frame: np.ndarray,
        render_input: object,
        settings: object,
    ) -> StereoRenderResult:
        self.calls.append((threading.get_ident(), render_input, settings))
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
        self.geometry_calls: list[tuple[int, StereoGeometryFrame, StereoSplatSettings]] = []
        self.relative_calls: list[tuple[int, np.ndarray, StereoRenderSettings]] = []

    def render(
        self,
        frame: np.ndarray,
        canonical: np.ndarray,
        settings: StereoRenderSettings,
    ) -> StereoRenderResult:
        self.relative_calls.append((threading.get_ident(), canonical, settings))
        return self._result(frame)

    def render_geometry(
        self,
        frame: np.ndarray,
        geometry: StereoGeometryFrame,
        settings: StereoSplatSettings,
    ) -> StereoRenderResult:
        self.geometry_calls.append((threading.get_ident(), geometry, settings))
        return self._result(frame)

    @staticmethod
    def _result(frame: np.ndarray) -> StereoRenderResult:
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


@pytest.mark.parametrize(
    "metadata_mutation",
    [
        {"status": "writing"},
        {"fingerprint": "not-the-completed-stage-fingerprint"},
        {"frame_names": ["frame_0000.png", "frame_0002.png", "frame_0001.png"]},
        {"convergence": None},
    ],
    ids=["status", "fingerprint", "ordered-manifest", "convergence"],
)
def test_metric_stereo_rejects_incomplete_stage_before_constructing_renderer(
    tmp_path: Path,
    metadata_mutation: dict[str, object],
) -> None:
    frame_files, metric_files, directories, _ = _make_metric_inputs(tmp_path)
    metadata_path = metric_files[0].parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(metadata_mutation)
    if "fingerprint" not in metadata_mutation:
        metadata.pop("fingerprint")
        metadata["fingerprint"] = canonical_json_hash(metadata)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with patch.object(
        stereo_generator,
        "StereoRenderer",
        side_effect=AssertionError("renderer constructed before metric barrier"),
    ) as renderer_factory:
        generator = StereoPairGenerator()
        with pytest.raises(
            ValueError,
            match="^metric stereo requires completed clip-global convergence metadata$",
        ):
            generator.create_stereo_pairs_from_files(
                frame_files,
                metric_files,
                directories,
                _settings(stereo_geometry_mode="metric_camera"),
            )

    renderer_factory.assert_not_called()


@pytest.mark.parametrize(
    ("frame_count", "metric_count"),
    [(0, 0), (3, 0), (3, 2)],
    ids=["empty", "missing-metric-payloads", "mismatched-manifest"],
)
def test_metric_stereo_rejects_invalid_payload_count_before_constructing_renderer(
    tmp_path: Path,
    frame_count: int,
    metric_count: int,
) -> None:
    frame_files, metric_files, directories, _ = _make_metric_inputs(tmp_path)

    with patch.object(
        stereo_generator,
        "StereoRenderer",
        side_effect=AssertionError("renderer constructed before metric barrier"),
    ) as renderer_factory:
        generator = StereoPairGenerator()
        with pytest.raises(
            ValueError,
            match="^metric stereo requires completed clip-global convergence metadata$",
        ):
            generator.create_stereo_pairs_from_files(
                frame_files[:frame_count],
                metric_files[:metric_count],
                directories,
                _settings(stereo_geometry_mode="metric_camera"),
            )

    renderer_factory.assert_not_called()


def test_relative_stereo_preserves_legacy_count_mismatch_result(tmp_path: Path) -> None:
    frame_files, depth_files, directories = _make_file_inputs(tmp_path, count=2)
    renderer = _FakeRenderer()

    assert not StereoPairGenerator(renderer=renderer).create_stereo_pairs_from_files(
        frame_files,
        depth_files[:1],
        directories,
        _settings(stereo_geometry_mode="relative"),
    )
    assert renderer.calls == []


def test_relative_stereo_setup_does_not_read_metric_metadata(tmp_path: Path) -> None:
    frame_files, depth_files, directories = _make_file_inputs(tmp_path, count=1)
    renderer = _FakeRenderer()

    with patch.object(
        MetricGeometryStore,
        "read_metadata",
        side_effect=AssertionError("relative setup read metric metadata"),
    ):
        assert StereoPairGenerator(renderer=renderer).create_stereo_pairs_from_files(
            frame_files,
            depth_files,
            directories,
            _settings(stereo_geometry_mode="relative"),
        )

    assert len(renderer.calls) == 1


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
    replacements: list[tuple[Path, Path]] = []
    original_replace = OwnedSiblingTemporary.replace

    def record_replace(self: OwnedSiblingTemporary, **kwargs) -> None:
        replacements.append((self.path, self.target))
        original_replace(self, **kwargs)

    with patch.object(OwnedSiblingTemporary, "replace", record_replace):
        _atomic_write_png(output, image)

    assert len(replacements) == 1
    temporary, destination = replacements[0]
    assert Path(temporary).parent == output.parent
    assert Path(destination) == output
    assert np.array_equal(cv2.imread(str(output), cv2.IMREAD_COLOR), image)
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_png_write_preserves_an_unowned_temporary_collision(tmp_path: Path) -> None:
    output = tmp_path / "frame.png"
    temporary = publication_temporary_path(output)
    temporary.write_bytes(b"foreign")

    with pytest.raises(OwnedTemporaryCollisionError):
        _atomic_write_png(output, np.zeros((2, 3, 3), dtype=np.uint8))

    assert temporary.read_bytes() == b"foreign"
    assert not output.exists()


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


def test_relative_file_pipeline_matches_base_legacy_bytes_at_half_lane_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    width = 8950
    canonical_value = np.float32(0.23064221441745758)
    frame_files, depth_files, directories = _make_file_inputs(
        tmp_path,
        count=1,
        frame_shape=(1, width),
    )
    frame = np.random.default_rng(20260817).integers(
        0,
        256,
        size=(1, width, 3),
        dtype=np.uint8,
    )
    assert cv2.imwrite(str(frame_files[0]), frame)
    assert cv2.imwrite(str(depth_files[0]), np.ones((1, width), dtype=np.uint16))
    metadata = json.loads((depth_files[0].parent / "metadata.json").read_text())
    metadata["encoding_scale"] = 1.0 / float(canonical_value)
    decoded_canonical = np.ones((1, width), dtype=np.float32)
    decoded_canonical *= np.float32(1.0 / float(metadata["encoding_scale"]))
    assert decoded_canonical[0, 0] == canonical_value
    settings = _settings(
        stereo_strength=0.473053267491137,
        convergence=0.05202130106440961,
        occlusion_fill="none",
        stereo_io_workers=1,
    )
    expected = StereoRenderer(device="cpu").render(
        frame,
        decoded_canonical,
        StereoRenderSettings(
            stereo_strength=0.473053267491137,
            convergence=0.05202130106440961,
            occlusion_fill="none",
        ),
    )

    generator = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))
    monkeypatch.setattr(
        generator,
        "_get_canonical_metadata",
        lambda _depth_files, _frame_files: metadata,
    )
    result = generator.create_stereo_pairs_from_files(
        frame_files,
        depth_files,
        directories,
        settings,
    )

    actual_left = cv2.imread(str(directories["left_frames"] / frame_files[0].name))
    actual_right = cv2.imread(str(directories["right_frames"] / frame_files[0].name))
    assert result is True
    assert np.array_equal(actual_left, expected.left_image)
    assert np.array_equal(actual_right, expected.right_image)


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


def test_capacity_one_permit_brackets_provider_replay_and_writer_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work_items = [
        stereo_generator._FileWorkItem(
            index=index,
            frame_path=tmp_path / f"frame-{index}.png",
            depth_path=tmp_path / f"depth-{index}.png",
            frame_name=f"frame-{index}",
            left_path=tmp_path / f"left-{index}.png",
            right_path=tmp_path / f"right-{index}.png",
        )
        for index in range(2)
    ]
    payload_refs: list[weakref.ReferenceType[np.ndarray]] = []
    replay_violations: list[str] = []

    def work_factory():
        yield work_items[0]
        gc.collect()
        if not payload_refs:
            replay_violations.append("provider advanced before the first decode")
        elif any(reference() is not None for reference in payload_refs):
            replay_violations.append("provider advanced while the prior payload was live")
        yield work_items[1]

    def decode(_work):
        frame = np.full((2, 4, 3), 17, dtype=np.uint8)
        encoded = np.full((2, 4), 32768, dtype=np.uint16)
        payload_refs.extend((weakref.ref(frame), weakref.ref(encoded)))
        return (
            frame,
            RelativeStereoPrimitiveInput(
                encoded_canonical=encoded,
                encoding_scale=np.float32(65535.0),
            ),
            None,
        )

    class ReleasingRenderer:
        def render(self, frame, _canonical, _settings):
            left = frame.copy()
            right = frame.copy()
            payload_refs.extend((weakref.ref(left), weakref.ref(right)))
            valid = np.ones(frame.shape[:2], dtype=np.bool_)
            holes = np.zeros(frame.shape[:2], dtype=np.bool_)
            return StereoRenderResult(
                left_image=left,
                right_image=right,
                left_valid_mask=valid,
                right_valid_mask=valid.copy(),
                left_hole_mask=holes,
                right_hole_mask=holes.copy(),
            )

    monkeypatch.setattr(stereo_generator, "_write_pair", lambda _item: None)
    pipeline = stereo_generator._StereoFilePipeline(
        renderer=ReleasingRenderer(),
        work_factory=work_factory,
        work_count=2,
        decode=decode,
        render_settings=StereoRenderSettings(
            stereo_strength=0.0,
            convergence=0.5,
            occlusion_fill="none",
        ),
        render_mode="fast",
        geometry_mode="relative",
        occlusion_fill="none",
        rgb_stage_fingerprint="rgb",
        keep_masks=False,
        workers=1,
        capacity=1,
        completed=0,
        total_frames=2,
        report_progress=lambda **_kwargs: None,
    )

    stats = pipeline.run()

    gc.collect()
    assert replay_violations == []
    assert all(reference() is None for reference in payload_refs)
    assert stats.permits_acquired == stats.permits_released == 2
    assert stats.active_permits == 0


def test_file_pipeline_surfaces_provider_replay_failure_without_hanging(
    tmp_path: Path,
) -> None:
    def broken_work_factory():
        raise RuntimeError("provider replay failed")
        yield  # pragma: no cover - makes this a generator

    pipeline = stereo_generator._StereoFilePipeline(
        renderer=_FakeRenderer(),
        work_factory=broken_work_factory,
        work_count=1,
        decode=lambda _work: (_ for _ in ()).throw(AssertionError("decode ran")),
        render_settings=StereoRenderSettings(
            stereo_strength=0.0,
            convergence=0.5,
            occlusion_fill="none",
        ),
        render_mode="fast",
        geometry_mode="relative",
        occlusion_fill="none",
        rgb_stage_fingerprint="rgb",
        keep_masks=False,
        workers=1,
        capacity=1,
        completed=0,
        total_frames=1,
        report_progress=lambda **_kwargs: None,
    )
    result: list[BaseException] = []

    def run_pipeline() -> None:
        try:
            pipeline.run()
        except BaseException as error:
            result.append(error)

    runner = threading.Thread(target=run_pipeline, daemon=True)
    runner.start()
    runner.join(timeout=3.0)

    assert not runner.is_alive(), "provider replay failure left pipeline.run blocked"
    assert len(result) == 1
    assert isinstance(result[0], RuntimeError)
    assert "provider replay failed" in str(result[0])
    stats = pipeline.metrics.snapshot()
    assert stats.permits_acquired == stats.permits_released == 1
    assert stats.active_permits == 0
    assert not any(thread.name.startswith("stereo-") for thread in threading.enumerate())


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


def test_relative_generator_retains_canonical_for_the_legacy_renderer(
    tmp_path: Path,
) -> None:
    frame_files, depth_files, directories = _make_file_inputs(tmp_path, count=1)
    renderer = _RecordingCommonRenderer()

    assert StereoPairGenerator(renderer=renderer).create_stereo_pairs_from_files(
        frame_files,
        depth_files,
        directories,
        _settings(stereo_geometry_mode="relative"),
    )

    assert len(renderer.relative_calls) == 1
    assert not renderer.geometry_calls
    _, canonical, render_settings = renderer.relative_calls[0]
    assert canonical.shape == (8, 8)
    assert render_settings == StereoRenderSettings(
        stereo_strength=2.0,
        convergence=0.5,
        occlusion_fill="background",
    )


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
    assert not renderer.relative_calls
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


def test_metric_quality_pipeline_uses_native_renderer_and_persists_stats(
    tmp_path: Path,
) -> None:
    frame_files, metric_files, directories, _ = _make_metric_inputs(
        tmp_path,
        inverse_values=[1.0],
    )
    generator = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))

    assert generator.create_stereo_pairs_from_files(
        frame_files,
        metric_files,
        directories,
        _settings(
            stereo_geometry_mode="metric_camera",
            stereo_render_mode="quality",
            metric_convergence_distance="auto",
            virtual_baseline_mm=63.0,
            max_disparity_percent=2.0,
            crop_factor=0.75,
            occlusion_fill="none",
            occlusion_fill_max_px=8,
            keep_intermediates=True,
        ),
    )

    assert generator.last_pipeline_stats is not None
    assert generator.last_pipeline_stats.permit_count == 1
    frame_manifest = tmp_path / "04_stereo_diagnostics" / "frames" / "frame_0000" / "manifest.json"
    manifest = json.loads(frame_manifest.read_text(encoding="ascii"))
    assert manifest["geometry_mode"] == "metric"
    stats_path = frame_manifest.with_name("stats.json")
    assert (
        json.loads(stats_path.read_text(encoding="ascii"))["metric_projection"]["valid_pixel_count"]
        > 0
    )
    assert not (directories["left_frames"] / "clamp_stats").exists()
    assert (directories["left_frames"] / "clamp_summary.json").exists()


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


def test_metric_pair_without_stats_is_recovered_as_one_failure_unit(tmp_path: Path) -> None:
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
    assert len(list(directories["left_frames"].glob("*.png"))) == 1
    assert len(list(directories["right_frames"].glob("*.png"))) == 1
    assert not stat_dir.exists() or list(stat_dir.glob("*.json")) == []

    assert generator.create_stereo_pairs_from_files(
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
    assert len(list(directories["left_frames"].glob("*.png"))) == 1
    assert len(list(directories["right_frames"].glob("*.png"))) == 1
    assert len(list(stat_dir.glob("*.json"))) == 1


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
