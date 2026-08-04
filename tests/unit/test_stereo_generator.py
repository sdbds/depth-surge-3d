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
    StereoRenderSettings,
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
        self.calls: list[tuple[int, np.ndarray, StereoRenderSettings]] = []

    def render(
        self,
        frame: np.ndarray,
        canonical: np.ndarray,
        settings: StereoRenderSettings,
    ) -> StereoRenderResult:
        self.calls.append((threading.get_ident(), canonical.copy(), settings))
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

    assert HOST_STEREO_BYTES_PER_PIXEL == 16
    assert capacity == min(32, STEREO_HOST_BUDGET // slot_bytes)
    assert capacity * slot_bytes <= STEREO_HOST_BUDGET


def test_frame_larger_than_one_host_slot_is_rejected() -> None:
    with pytest.raises(MemoryError, match="required"):
        calculate_stereo_pipeline_capacity(16384, 8192, 4)


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
    complete_left = directories["left_frames"] / "frame_0000.png"
    complete_right = directories["right_frames"] / "frame_0000.png"
    singleton_left = directories["left_frames"] / "frame_0001.png"
    assert cv2.imwrite(str(complete_left), np.full((8, 8, 3), 7, dtype=np.uint8))
    assert cv2.imwrite(str(complete_right), np.full((8, 8, 3), 8, dtype=np.uint8))
    assert cv2.imwrite(str(singleton_left), np.full((8, 8, 3), 9, dtype=np.uint8))
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


def test_in_memory_path_uses_shared_renderer_without_process_pool(tmp_path: Path) -> None:
    renderer = _FakeRenderer()
    generator = StereoPairGenerator(renderer=renderer)
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    frames = np.full((2, 4, 5, 3), 10, dtype=np.uint8)
    canonical = np.full((2, 2, 3), 0.5, dtype=np.float32)
    frame_files = [tmp_path / f"frame_{index:04d}.png" for index in range(2)]

    result = generator.create_stereo_pairs(
        frames,
        canonical,
        frame_files,
        {"left_frames": left_dir, "right_frames": right_dir},
        _settings(keep_intermediates=True),
    )

    assert result is True
    assert len(renderer.calls) == 2
    assert len(list(left_dir.glob("*.png"))) == 2
    assert not hasattr(stereo_generator, "mp")


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
