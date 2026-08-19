"""File-backed VDPP stage coordinator tests."""

from __future__ import annotations

import json
import platform
from collections import namedtuple
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.core.depth_contract import canonical_json_hash
from src.depth_surge_3d.core.vdpp_calibration import PairTileState
from src.depth_surge_3d.inference.depth.vdpp_contract import (
    VDPP_WINDOW_SIZE,
    build_vdpp_execution_plan,
    vdpp_model_identity,
)
import src.depth_surge_3d.processing.frames.temporal_stabilizer as temporal_stabilizer_module
from src.depth_surge_3d.processing.frames.temporal_stabilizer import (
    TemporalDepthStabilizer,
)


class _FakePostprocessor:
    def __init__(self, *, fail_shot: int | None = None, residual: float = 0.01) -> None:
        self.fail_shot = fail_shot
        self.residual = np.float32(residual)
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
            yield index, values[index] + self.residual

    def release(self):
        self.released = True


class _OverlappingWindowPostprocessor(_FakePostprocessor):
    def process_shot(self, frame_count, load_window):
        assert frame_count == 61
        first = load_window(0, 32)
        second = load_window(28, 60)
        tail = load_window(56, 61)
        self.loaded_windows = [
            first[:, 0, 0].tolist(),
            second[:, 0, 0].tolist(),
            tail[:, 0, 0].tolist(),
        ]
        final = np.concatenate([first, second[4:], tail[4:]], axis=0)
        for index in range(frame_count):
            yield index, final[index] + self.residual


class _PatternPostprocessor(_FakePostprocessor):
    def __init__(self, transform) -> None:
        super().__init__()
        self.transform = transform

    def process_shot(self, frame_count, load_window):
        values = load_window(0, frame_count)
        for index in range(frame_count):
            output = np.asarray(self.transform(values[index]), dtype=np.float32)
            yield index, output


class _InvalidOutputPostprocessor(_FakePostprocessor):
    def __init__(self, failure: str) -> None:
        super().__init__()
        self.failure = failure

    def process_shot(self, frame_count, load_window):
        values = load_window(0, frame_count)
        if self.failure == "ordering":
            yield 1, values[0]
            return
        output = values[0].copy()
        if self.failure == "nonfinite":
            output[0, 0] = np.float32(np.nan)
        elif self.failure == "shape":
            output = output[:-1]
        yield 0, output


class _ReleaseFailingPostprocessor(_FakePostprocessor):
    def process_shot(self, frame_count, load_window):
        del frame_count, load_window
        raise RuntimeError("primary inference failure")
        yield  # pragma: no cover - keeps this method an iterator

    def release(self):
        self.released = True
        raise RuntimeError("release failure")


def _write_base(
    tmp_path: Path,
    count: int,
    cuts: list[int],
    *,
    encoded_frames: list[np.ndarray] | None = None,
):
    base = tmp_path / "03_disparity_maps"
    scene = tmp_path / "01_scene_data"
    base.mkdir()
    scene.mkdir()
    frame_names = [f"frame_{index + 1:06d}.png" for index in range(count)]
    frame_files = [Path(name) for name in frame_names]
    native_shape = encoded_frames[0].shape if encoded_frames is not None else (3, 5)
    if len(native_shape) != 2 or any(
        frame.shape != native_shape for frame in (encoded_frames or [])
    ):
        raise ValueError("test base frames must share one two-dimensional shape")
    depth_files = []
    for index, name in enumerate(frame_names):
        path = base / name
        values = (
            encoded_frames[index]
            if encoded_frames is not None
            else (np.arange(15, dtype=np.uint16).reshape(3, 5) * np.uint16(1000))
        )
        assert cv2.imwrite(
            str(path),
            values,
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
        "native_shape": list(native_shape),
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

    assert fake.loaded_windows == [
        [0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0],
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


def test_disk_bound_charges_pending_pngs_largest_memmap_and_atomic_temporary() -> None:
    shape = (608, 1080)
    pending_frames = 61
    largest_shot = 35
    frame_u16 = shape[0] * shape[1] * 2
    png_bound = int(np.ceil(frame_u16 * 1.10))
    expected = int(
        np.ceil(
            1.10
            * (
                pending_frames * png_bound
                + largest_shot * shape[0] * shape[1] * 4
                + png_bound
                + 1024 * 1024
            )
        )
    )
    assert (
        TemporalDepthStabilizer._required_disk_bytes(
            pending_frames,
            largest_shot,
            shape,
        )
        == expected
    )


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
    assert not (directories["disparity_stabilized"] / ".vdpp-work").exists()
    assert fake.released is True


def test_memmap_creation_failure_removes_partial_private_work_and_releases_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    depth_files, _frame_files, directories = _write_base(tmp_path, 2, [])
    fake = _FakePostprocessor()

    def fail_memmap(path, *_args, **_kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("memmap allocation failed")

    monkeypatch.setattr(temporal_stabilizer_module.np, "memmap", fail_memmap)

    with pytest.raises(OSError, match="memmap allocation failed"):
        _coordinator(tmp_path, fake).generate_files(
            depth_files,
            {"temporal_postprocessor": "vdpp"},
            directories,
        )

    assert not (directories["disparity_stabilized"] / ".vdpp-work").exists()
    assert fake.released is True


def test_work_directory_creation_failure_removes_partial_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    depth_files, _frame_files, directories = _write_base(tmp_path, 2, [])
    original_mkdir = Path.mkdir

    def fail_after_creating_work(path, *args, **kwargs):
        result = original_mkdir(path, *args, **kwargs)
        if path.name == ".vdpp-work":
            raise OSError("work directory creation failed")
        return result

    monkeypatch.setattr(Path, "mkdir", fail_after_creating_work)

    with pytest.raises(OSError, match="work directory creation failed"):
        _coordinator(tmp_path, _FakePostprocessor()).generate_files(
            depth_files,
            {"temporal_postprocessor": "vdpp"},
            directories,
        )

    assert not (directories["disparity_stabilized"] / ".vdpp-work").exists()


def test_window_loader_rejects_requests_above_the_pinned_host_bound(tmp_path: Path) -> None:
    depth_files, _frame_files, _directories = _write_base(
        tmp_path,
        VDPP_WINDOW_SIZE + 1,
        [],
    )
    loader = TemporalDepthStabilizer._window_loader(
        depth_files,
        shot_start=0,
        shot_length=len(depth_files),
        native_shape=(3, 5),
    )

    with pytest.raises(ValueError, match="window size"):
        loader(0, VDPP_WINDOW_SIZE + 1)


def test_too_few_pairs_outranks_a_statistics_failure() -> None:
    scan = temporal_stabilizer_module._ShotPreScan(
        shot_id=0,
        start=0,
        end=1,
        midpoint_count=0,
        flat_frame_count=0,
        shot_pixels=15,
    )

    calibration, scale, shift = TemporalDepthStabilizer._calibrate_raw_shot(
        np.empty((1, 3, 5), dtype=np.float32),
        [],
        scan,
        (3, 5),
        PairTileState.empty(),
        1,
        statistics_failed=True,
    )

    assert calibration["fallback_reason"] == "too_few_pairs"
    assert scale is None
    assert shift is None


def test_commit_failure_closes_output_iterator_before_memmap_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    depth_files, _frame_files, directories = _write_base(tmp_path, 2, [])

    class ClosableOutputs:
        closed = False

        def __iter__(self):
            return self

        def __next__(self):
            raise AssertionError("commit must fail before consuming output")

        def close(self):
            self.closed = True

    outputs = ClosableOutputs()
    monkeypatch.setattr(
        TemporalDepthStabilizer,
        "_accepted_outputs",
        lambda *_args, **_kwargs: outputs,
    )
    monkeypatch.setattr(
        TemporalDepthStabilizer,
        "_copy_outputs",
        lambda *_args, **_kwargs: outputs,
    )
    monkeypatch.setattr(
        temporal_stabilizer_module.StabilizedDepthStore,
        "commit_shot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("commit failure")),
    )

    with pytest.raises(RuntimeError, match="commit failure"):
        _coordinator(tmp_path, _FakePostprocessor()).generate_files(
            depth_files,
            {"temporal_postprocessor": "vdpp"},
            directories,
        )

    assert outputs.closed is True
    assert not (directories["disparity_stabilized"] / ".vdpp-work").exists()


def test_release_failure_does_not_hide_primary_inference_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    depth_files, _frame_files, directories = _write_base(tmp_path, 2, [])
    fake = _ReleaseFailingPostprocessor()

    with pytest.raises(RuntimeError, match="primary inference failure"):
        _coordinator(tmp_path, fake).generate_files(
            depth_files,
            {"temporal_postprocessor": "vdpp"},
            directories,
        )

    assert fake.released is True
    assert "release also failed" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("nonfinite", "non-finite"),
        ("shape", "native-shape"),
        ("ordering", "ordered and contiguous"),
    ],
)
def test_invalid_model_output_is_a_hard_failure_and_never_committed(
    tmp_path: Path,
    failure: str,
    message: str,
) -> None:
    depth_files, _frame_files, directories = _write_base(tmp_path, 2, [])
    fake = _InvalidOutputPostprocessor(failure)

    with pytest.raises((FloatingPointError, TypeError, ValueError), match=message):
        _coordinator(tmp_path, fake).generate_files(
            depth_files,
            {"temporal_postprocessor": "vdpp"},
            directories,
        )

    metadata = json.loads((directories["disparity_stabilized"] / "metadata.json").read_text())
    assert metadata["completed_shots"] == []
    assert not (directories["disparity_stabilized"] / ".vdpp-work").exists()
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


@pytest.mark.parametrize(
    ("values", "mode", "reason"),
    [
        (
            np.full((3, 5), 32768, dtype=np.uint16),
            "all_midpoint",
            None,
        ),
        (
            np.full((3, 5), 1234, dtype=np.uint16),
            "base_fallback",
            "source_no_range",
        ),
    ],
)
def test_degenerate_shot_commits_exact_copy_without_checkpoint_or_model(
    tmp_path: Path,
    values: np.ndarray,
    mode: str,
    reason: str | None,
) -> None:
    depth_files, _frame_files, directories = _write_base(
        tmp_path,
        2,
        [],
        encoded_frames=[values, values],
    )
    coordinator = _coordinator(
        tmp_path,
        _FakePostprocessor(),
        checkpoint_resolver=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("checkpoint must not be resolved")
        ),
        factory=lambda *_args: (_ for _ in ()).throw(
            AssertionError("model must not be constructed")
        ),
    )

    result = coordinator.generate_files(
        depth_files,
        {"temporal_postprocessor": "vdpp"},
        directories,
    )

    assert all(
        np.array_equal(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), values) for path in result
    )
    manifest = json.loads(
        (directories["disparity_stabilized"] / "shot_manifests/shot_000000.json").read_text()
    )
    assert manifest["calibration"]["mode"] == mode
    assert manifest["calibration"]["fallback_reason"] == reason


def test_large_positive_residual_is_affine_calibrated_without_saturation(
    tmp_path: Path,
) -> None:
    depth_files, _frame_files, directories = _write_base(tmp_path, 3, [])
    coordinator = _coordinator(tmp_path, _FakePostprocessor(residual=2.0))

    result = coordinator.generate_files(
        depth_files,
        {"temporal_postprocessor": "vdpp"},
        directories,
    )

    output = cv2.imread(str(result[0]), cv2.IMREAD_UNCHANGED)
    assert int(output.max()) < 65535
    assert int(output.min()) == 0
    manifest = json.loads(
        (directories["disparity_stabilized"] / "shot_manifests/shot_000000.json").read_text()
    )
    assert manifest["calibration"]["mode"] == "ols"


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        ("source_variance", "source_variance"),
        ("raw_variance", "raw_variance"),
        ("negative_scale", "scale_below_minimum"),
        ("correlation", "correlation"),
        ("contrast", "contrast"),
        ("mean_drift", "mean_drift"),
        ("preclip", "preclip_out_of_range"),
    ],
)
def test_failed_calibration_gate_commits_a_bit_exact_whole_shot_fallback(
    tmp_path: Path,
    failure: str,
    reason: str,
) -> None:
    default_frame = np.arange(15, dtype=np.uint16).reshape(3, 5) * np.uint16(1000)
    if failure == "source_variance":
        source = np.full((1, 5000), 1000, dtype=np.uint16)
        source[0, -1] = np.uint16(1001)
    elif failure == "mean_drift":
        source = np.rint(np.linspace(0, 65535, 15)).astype(np.uint16).reshape(3, 5)
    else:
        source = default_frame
    encoded_frames = [source.copy(), source.copy()]

    patterns = {
        "correlation": [12, 1, 10, 0, 2, 7, 9, 8, 13, 6, 4, 3, 5, 11, 14],
        "contrast": [
            -8.5689535,
            -0.67274094,
            -0.49786147,
            -0.38589665,
            -0.43361571,
            0.58333975,
            -0.44064295,
            0.50335228,
            -0.18577257,
            -0.59654385,
            0.08887443,
            -0.15801181,
            -0.72753948,
            0.05718039,
            1.53304803,
        ],
        "mean_drift": [
            -1.03313446,
            -5.51687527,
            -0.84493542,
            -1.09125042,
            -0.61941183,
            -0.26001722,
            -0.79743642,
            -0.1725955,
            -0.52325255,
            -0.39608467,
            -0.39347661,
            -0.1065691,
            -0.26357427,
            0.69655734,
            1.27443695,
        ],
        "preclip": [
            -2.3800869,
            -0.88344377,
            -0.93835044,
            -0.98808235,
            -0.54724258,
            0.23526877,
            -0.12726012,
            -0.3896403,
            0.28386921,
            0.51349509,
            0.42926314,
            0.32201767,
            -0.55639195,
            0.54918045,
            1.56169784,
        ],
    }

    def transform(values: np.ndarray) -> np.ndarray:
        if failure == "source_variance":
            return values
        if failure == "raw_variance":
            return np.full(values.shape, 0.5, dtype=np.float32)
        if failure == "negative_scale":
            return np.float32(1.0) - values
        return np.asarray(patterns[failure], dtype=np.float32).reshape(values.shape)

    depth_files, _frame_files, directories = _write_base(
        tmp_path,
        2,
        [],
        encoded_frames=encoded_frames,
    )
    result = _coordinator(tmp_path, _PatternPostprocessor(transform)).generate_files(
        depth_files,
        {"temporal_postprocessor": "vdpp"},
        directories,
    )

    assert all(
        np.array_equal(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), source) for path in result
    )
    manifest = json.loads(
        (directories["disparity_stabilized"] / "shot_manifests/shot_000000.json").read_text()
    )
    assert manifest["calibration"]["mode"] == "base_fallback"
    assert manifest["calibration"]["fallback_reason"] == reason


def test_flat_frame_inside_ranged_shot_is_model_fed_midpoint_and_copied_exactly(
    tmp_path: Path,
) -> None:
    ranged = np.arange(15, dtype=np.uint16).reshape(3, 5) * np.uint16(1000)
    flat = np.full((3, 5), 1234, dtype=np.uint16)
    depth_files, _frame_files, directories = _write_base(
        tmp_path,
        3,
        [],
        encoded_frames=[ranged, flat, ranged],
    )
    fake = _FakePostprocessor(residual=2.0)

    result = _coordinator(tmp_path, fake).generate_files(
        depth_files,
        {"temporal_postprocessor": "vdpp"},
        directories,
    )

    assert fake.loaded_windows[0][1] == 0.5
    assert np.array_equal(cv2.imread(str(result[1]), cv2.IMREAD_UNCHANGED), flat)
    manifest = json.loads(
        (directories["disparity_stabilized"] / "shot_manifests/shot_000000.json").read_text()
    )
    assert manifest["calibration"]["flat_frame_count"] == 1


def test_runtime_identity_records_host_versions_and_numeric_behavior(
    tmp_path: Path,
) -> None:
    depth_files, _frame_files, directories = _write_base(tmp_path, 1, [])
    _coordinator(tmp_path, _FakePostprocessor()).generate_files(
        depth_files,
        {"temporal_postprocessor": "vdpp"},
        directories,
    )
    metadata = json.loads((directories["disparity_stabilized"] / "metadata.json").read_text())
    provenance = metadata["execution_provenance"]
    assert provenance["python_implementation"] == platform.python_implementation()
    assert provenance["python_version"] == platform.python_version()
    assert provenance["platform_system"] == platform.system()
    assert provenance["platform_machine"] == platform.machine()
    assert provenance["sys_byteorder"] == sys.byteorder
    assert provenance["numpy_version"] == np.__version__
    assert provenance["opencv_version"] == cv2.__version__
    assert provenance["numeric_reducer_probe"]["schema"] == ("vdpp-numeric-reducer-probe-v1")


@pytest.mark.parametrize(
    ("first_numpy", "first_probe", "second_numpy", "second_probe"),
    [
        ("same", "probe-a", "same", "probe-b"),
        ("numpy-a", "same-probe", "numpy-b", "same-probe"),
    ],
)
def test_partial_resume_resets_when_host_numeric_identity_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_numpy: str,
    first_probe: str,
    second_numpy: str,
    second_probe: str,
) -> None:
    depth_files, _frame_files, directories = _write_base(tmp_path, 4, [2])

    def host_identity(numpy_version: str, probe_value: str) -> dict[str, object]:
        return {
            "python_implementation": "CPython",
            "python_version": "3.test",
            "platform_system": "test-system",
            "platform_machine": "test-machine",
            "sys_byteorder": "little",
            "numpy_version": numpy_version,
            "opencv_version": "opencv-same",
            "numeric_reducer_probe": {
                "schema": "vdpp-numeric-reducer-probe-v1",
                "value": probe_value,
            },
        }

    monkeypatch.setattr(
        temporal_stabilizer_module,
        "collect_vdpp_host_runtime_identity",
        lambda: host_identity(first_numpy, first_probe),
        raising=False,
    )
    first = _FakePostprocessor(fail_shot=1)
    with pytest.raises(RuntimeError, match="fake VDPP failure"):
        _coordinator(tmp_path, first).generate_files(
            depth_files,
            {"temporal_postprocessor": "vdpp"},
            directories,
        )

    monkeypatch.setattr(
        temporal_stabilizer_module,
        "collect_vdpp_host_runtime_identity",
        lambda: host_identity(second_numpy, second_probe),
        raising=False,
    )
    resumed = _FakePostprocessor()
    _coordinator(tmp_path, resumed).generate_files(
        depth_files,
        {"temporal_postprocessor": "vdpp"},
        directories,
    )

    assert resumed.shot_number == 2


def test_61_frame_overlap_counts_each_pair_once_and_preserves_midpoints(
    tmp_path: Path,
) -> None:
    frame = np.arange(15, dtype=np.uint16).reshape(3, 5) * np.uint16(1000)
    frame[0, 0] = np.uint16(32768)
    depth_files, _frame_files, directories = _write_base(
        tmp_path,
        61,
        [],
        encoded_frames=[frame.copy() for _ in range(61)],
    )
    fake = _OverlappingWindowPostprocessor(residual=2.0)

    result = _coordinator(tmp_path, fake).generate_files(
        depth_files,
        {"temporal_postprocessor": "vdpp"},
        directories,
    )

    manifest = json.loads(
        (directories["disparity_stabilized"] / "shot_manifests/shot_000000.json").read_text()
    )
    assert [len(window) for window in fake.loaded_windows] == [32, 32, 5]
    assert manifest["calibration"]["pair_count"] == 61 * 14
    assert manifest["calibration"]["midpoint_count"] == 61
    assert all(
        cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[0, 0] == np.uint16(32768) for path in result
    )
