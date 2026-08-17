"""File-backed VDPP stage coordinator tests."""

from __future__ import annotations

import json
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.core.depth_contract import canonical_json_hash
from src.depth_surge_3d.inference.depth.vdpp_contract import (
    build_vdpp_execution_plan,
    vdpp_model_identity,
)
from src.depth_surge_3d.processing.frames.temporal_stabilizer import (
    TemporalDepthStabilizer,
)


class _FakePostprocessor:
    def __init__(self, *, fail_shot: int | None = None) -> None:
        self.fail_shot = fail_shot
        self.shot_number = 0
        self.loaded_windows: list[list[float]] = []
        self.preflight_lengths: list[int] = []
        self.released = False

    def model_identity(self):
        return vdpp_model_identity()

    def execution_plan(self, native_shape):
        return build_vdpp_execution_plan(native_shape)

    def preflight(self, length, native_shape):
        self.preflight_lengths.append(length)
        return {
            "preflight_window_lengths": [min(32, length)],
            "preflight_max_memory_allocated": 123,
            "preflight_max_memory_reserved": 456,
        }

    def process_shot(self, frame_count, load_window):
        shot = self.shot_number
        self.shot_number += 1
        values = load_window(0, frame_count)
        self.loaded_windows.append(values[:, 0, 0].tolist())
        if self.fail_shot == shot:
            raise RuntimeError("fake VDPP failure")
        for index in range(frame_count):
            yield index, values[index] + np.float32(0.01)

    def release(self):
        self.released = True


def _write_base(tmp_path: Path, count: int, cuts: list[int]):
    base = tmp_path / "03_disparity_maps"
    scene = tmp_path / "01_scene_data"
    base.mkdir()
    scene.mkdir()
    frame_names = [f"frame_{index + 1:06d}.png" for index in range(count)]
    frame_files = [Path(name) for name in frame_names]
    depth_files = []
    for index, name in enumerate(frame_names):
        path = base / name
        assert cv2.imwrite(
            str(path),
            np.full((3, 5), index * 5000, dtype=np.uint16),
        )
        depth_files.append(path)
    manifest = {
        "schema_version": 2,
        "algorithm_version": "hsv-histogram-final-v2",
        "status": "final",
        "frame_names": frame_names,
        "scene_ids": [sum(index >= cut for cut in cuts) for index in range(count)],
        "candidate_cuts": cuts,
        "final_cuts": cuts,
        "settings": {"enabled": True, "threshold": 0.55, "min_frames": 1},
        "source_frame_fingerprint": "frames",
        "sample_fingerprint": "samples",
        "bounds_fingerprint": "bounds",
    }
    (scene / "scene_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "algorithm_version": "scene-percentile-v1",
        "representation": "relative_disparity",
        "near_value": 1.0,
        "far_value": 0.0,
        "encoding": "uint16_png",
        "encoding_scale": 65535.0,
        "num_frames": count,
        "frame_names": frame_names,
        "native_shape": [3, 5],
        "source_raw_fingerprint": "raw",
        "source_model_fingerprint": "model",
        "scene_manifest_fingerprint": canonical_json_hash(manifest),
        "depth_bounds_fingerprint": "bounds",
    }
    metadata["fingerprint"] = canonical_json_hash(metadata)
    (base / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    directories = {
        "scene_data": scene,
        "disparity_stabilized": tmp_path / "03_disparity_stabilized",
    }
    return depth_files, frame_files, directories


def _coordinator(
    tmp_path: Path,
    postprocessor: _FakePostprocessor,
    *,
    device: str = "cuda",
    cuda_available=lambda: True,
    disk_free: int = 10**12,
    checkpoint_resolver=None,
    factory=None,
    runtime_provider=None,
):
    usage = namedtuple("usage", "total used free")
    return TemporalDepthStabilizer(
        effective_device=device,
        models_dir=tmp_path / "models",
        cuda_available=cuda_available,
        disk_usage=lambda _path: usage(10**12, 0, disk_free),
        checkpoint_resolver=checkpoint_resolver
        or (lambda *_args, **_kwargs: tmp_path / "models/VDPP/vdpp.pth"),
        postprocessor_factory=factory or (lambda _path, _device: postprocessor),
        runtime_identity_provider=runtime_provider
        or (lambda _device: ({"runtime": "same"}, {"runtime": "same"})),
    )


def test_generation_resets_state_at_final_scene_cuts_and_maps_local_indexes(
    tmp_path: Path,
) -> None:
    depth_files, frame_files, directories = _write_base(tmp_path, 7, [2, 5])
    fake = _FakePostprocessor()
    coordinator = _coordinator(tmp_path, fake)

    result = coordinator.generate_files(
        depth_files,
        {"temporal_postprocessor": "vdpp"},
        directories,
    )

    scale = np.float32(65535.0)
    assert fake.loaded_windows == [
        [0.0, float(np.float32(5000) / scale)],
        [
            float(np.float32(10000) / scale),
            float(np.float32(15000) / scale),
            float(np.float32(20000) / scale),
        ],
        [float(np.float32(25000) / scale), float(np.float32(30000) / scale)],
    ]
    assert [path.name for path in result] == [path.name for path in frame_files]
    assert fake.preflight_lengths == [3]
    assert fake.released is True


def test_complete_artifact_is_selected_without_cuda_checkpoint_runtime_or_model(
    tmp_path: Path,
) -> None:
    depth_files, _frame_files, directories = _write_base(tmp_path, 2, [])
    first = _FakePostprocessor()
    _coordinator(tmp_path, first).generate_files(
        depth_files,
        {"temporal_postprocessor": "vdpp"},
        directories,
    )

    cache_only = _coordinator(
        tmp_path,
        _FakePostprocessor(),
        device="cpu",
        cuda_available=lambda: (_ for _ in ()).throw(AssertionError("CUDA probe")),
        checkpoint_resolver=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("checkpoint resolution")
        ),
        factory=lambda *_args: (_ for _ in ()).throw(AssertionError("model construction")),
        runtime_provider=lambda *_args: (_ for _ in ()).throw(AssertionError("runtime import")),
    )

    result = cache_only.generate_files(
        depth_files,
        {"temporal_postprocessor": "vdpp"},
        directories,
    )

    assert len(result) == 2


def test_all_committed_shots_finalize_without_reloading_checkpoint_or_model(
    tmp_path: Path,
) -> None:
    depth_files, _frame_files, directories = _write_base(tmp_path, 2, [])
    _coordinator(tmp_path, _FakePostprocessor()).generate_files(
        depth_files,
        {"temporal_postprocessor": "vdpp"},
        directories,
    )
    metadata_path = directories["disparity_stabilized"] / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["status"] = "building"
    metadata["payload_fingerprint"] = None
    metadata["artifact_fingerprint"] = None
    metadata["state_fingerprint"] = canonical_json_hash(
        {
            "status": metadata["status"],
            "completed_shots": metadata["completed_shots"],
        }
    )
    metadata.pop("metadata_fingerprint")
    metadata["metadata_fingerprint"] = canonical_json_hash(metadata)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    resumed = _coordinator(
        tmp_path,
        _FakePostprocessor(),
        checkpoint_resolver=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("checkpoint must not be resolved")
        ),
        factory=lambda *_args: (_ for _ in ()).throw(
            AssertionError("model must not be reconstructed")
        ),
    )
    result = resumed.generate_files(
        depth_files,
        {"temporal_postprocessor": "vdpp"},
        directories,
    )

    assert len(result) == 2
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["status"] == "complete"


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_generation_rejects_non_cuda_before_checkpoint_or_stage_mutation(
    tmp_path: Path,
    device: str,
) -> None:
    depth_files, _frame_files, directories = _write_base(tmp_path, 2, [])
    coordinator = _coordinator(
        tmp_path,
        _FakePostprocessor(),
        device=device,
        checkpoint_resolver=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("checkpoint must not be resolved")
        ),
    )

    with pytest.raises(RuntimeError, match="CUDA"):
        coordinator.generate_files(
            depth_files,
            {"temporal_postprocessor": "vdpp"},
            directories,
        )

    assert not directories["disparity_stabilized"].exists()


def test_disk_preflight_happens_before_checkpoint_download(tmp_path: Path) -> None:
    depth_files, _frame_files, directories = _write_base(tmp_path, 2, [])
    coordinator = _coordinator(
        tmp_path,
        _FakePostprocessor(),
        disk_free=0,
        checkpoint_resolver=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("checkpoint must not be resolved")
        ),
    )

    with pytest.raises(OSError, match="disk space"):
        coordinator.generate_files(
            depth_files,
            {"temporal_postprocessor": "vdpp"},
            directories,
        )

    assert not directories["disparity_stabilized"].exists()


def test_failed_later_shot_preserves_earlier_commit_and_releases_model(
    tmp_path: Path,
) -> None:
    depth_files, _frame_files, directories = _write_base(tmp_path, 4, [2])
    fake = _FakePostprocessor(fail_shot=1)
    coordinator = _coordinator(tmp_path, fake)

    with pytest.raises(RuntimeError, match="fake VDPP failure"):
        coordinator.generate_files(
            depth_files,
            {"temporal_postprocessor": "vdpp"},
            directories,
        )

    metadata = json.loads((directories["disparity_stabilized"] / "metadata.json").read_text())
    assert [record["shot_id"] for record in metadata["completed_shots"]] == [0]
    assert (directories["disparity_stabilized"] / "frame_000001.png").is_file()
    assert not (directories["disparity_stabilized"] / "frame_000003.png").exists()
    assert fake.released is True


def test_off_mode_is_not_a_silent_passthrough_api(tmp_path: Path) -> None:
    depth_files, _frame_files, directories = _write_base(tmp_path, 1, [])
    coordinator = _coordinator(tmp_path, _FakePostprocessor())

    with pytest.raises(ValueError, match="only be invoked"):
        coordinator.generate_files(
            depth_files,
            {"temporal_postprocessor": "off"},
            directories,
        )
