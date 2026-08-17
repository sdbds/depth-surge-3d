"""Restartable metric-geometry storage and convergence tests."""

import errno
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pytest

from src.depth_surge_3d.core.depth_contract import canonical_json_hash
from src.depth_surge_3d.inference.depth.types import (
    DepthBatch,
    DepthRepresentation,
    PinholeCameraBatch,
)
from src.depth_surge_3d.processing.frames import metric_geometry
from src.depth_surge_3d.processing.frames.depth_storage import RawDepthStore
from src.depth_surge_3d.processing.frames.metric_geometry import (
    METRIC_CONVERGENCE_ALGORITHM_VERSION,
    ClipConvergence,
    MetricGeometryDiskFullError,
    MetricGeometryFrame,
    MetricGeometryStore,
    estimate_metric_geometry_disk_bytes,
    filesystem_allocation_unit,
    metric_frame_from_depth,
    require_metric_geometry_disk_space,
    sample_clip_convergence,
    select_convergence_frame_indexes,
)
from src.depth_surge_3d.rendering.stereo_geometry import build_metric_geometry


FRAME_NAMES = ["frame_000001.png", "frame_000002.png"]
VALID_FRAME = MetricGeometryFrame(
    inverse_depth=np.array([[0.5]], dtype=np.float32),
    valid=np.array([[True]], dtype=np.bool_),
    focal_x_normalized=np.float32(0.8),
)


def _create_metric_store(
    directory: Path,
    *,
    frame_names: list[str] | None = None,
    preflight_required_bytes: int = 1024,
) -> MetricGeometryStore:
    return MetricGeometryStore.open_or_create(
        directory,
        frame_names=list(frame_names or FRAME_NAMES),
        native_shape=(1, 1),
        source_raw_fingerprint="raw-fingerprint",
        source_frame_fingerprint="frame-fingerprint",
        candidate_scene_fingerprint="scene-fingerprint",
        preflight_required_bytes=preflight_required_bytes,
    )


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _disk_full_failure(kind: str) -> OSError:
    if kind == "enospc":
        return OSError(errno.ENOSPC, "disk full")
    failure = OSError("windows disk full")
    failure.winerror = 112  # type: ignore[attr-defined]
    return failure


def _raw_store_for_depths(directory: Path, depths: np.ndarray) -> RawDepthStore:
    values = np.asarray(depths, dtype=np.float32).reshape(-1, 1, 1)
    camera = PinholeCameraBatch(np.full(len(values), 0.8, dtype=np.float32))
    batch = DepthBatch(values, DepthRepresentation.METRIC_DEPTH, camera=camera)
    names = [f"frame_{index:06d}.png" for index in range(len(values))]
    store = RawDepthStore.create(
        directory,
        frame_names=names,
        semantic_fingerprint={"camera_model": "pinhole_fx"},
        requested_dtype="float32",
        first_batch=batch,
    )
    store.write_batch(names, batch)
    return store


@pytest.fixture
def raw_metric_store(tmp_path: Path) -> RawDepthStore:
    values = np.array([[[2.0]], [[4.0]], [[6.0]], [[8.0]]], dtype=np.float32)
    camera = PinholeCameraBatch(np.full(4, 0.8, dtype=np.float32))
    batch = DepthBatch(values, DepthRepresentation.METRIC_DEPTH, camera=camera)
    names = [f"frame_{index:06d}.png" for index in range(4)]
    store = RawDepthStore.create(
        tmp_path / "raw",
        frame_names=names,
        semantic_fingerprint={"camera_model": "pinhole_fx"},
        requested_dtype="float32",
        first_batch=batch,
    )
    store.write_batch(names, batch)
    return store


@pytest.fixture
def raw_store_with_only_invalid_depth(tmp_path: Path) -> RawDepthStore:
    values = np.array([[[0.0, -1.0, np.inf, np.nan]]], dtype=np.float32)
    camera = PinholeCameraBatch(np.array([0.8], dtype=np.float32))
    batch = DepthBatch(values, DepthRepresentation.METRIC_DEPTH, camera=camera)
    store = RawDepthStore.create(
        tmp_path / "raw-invalid",
        frame_names=["frame_000000.png"],
        semantic_fingerprint={"camera_model": "pinhole_fx"},
        requested_dtype="float32",
        first_batch=batch,
    )
    store.write_batch(["frame_000000.png"], batch)
    return store


@pytest.fixture
def partial_metric_store(tmp_path: Path) -> MetricGeometryStore:
    store = _create_metric_store(tmp_path / "partial")
    store.write_frame(FRAME_NAMES[0], VALID_FRAME)
    return store


@pytest.fixture
def complete_metric_store(tmp_path: Path) -> MetricGeometryStore:
    store = _create_metric_store(tmp_path / "complete", frame_names=[FRAME_NAMES[0]])
    store.write_frame(FRAME_NAMES[0], VALID_FRAME)
    store.finalize(ClipConvergence(np.float32(2.0), (0,), 1))
    return store


def test_metric_frame_derives_finite_inverse_depth_and_explicit_validity() -> None:
    smallest_subnormal = np.nextafter(np.float32(0.0), np.float32(1.0), dtype=np.float32)
    depth = np.array([[2.0, 0.0, np.inf], [4.0, -1.0, smallest_subnormal]], dtype=np.float32)
    frame = metric_frame_from_depth(depth, np.float32(0.8))
    assert frame.valid.tolist() == [[True, False, False], [True, False, False]]
    np.testing.assert_array_equal(
        frame.inverse_depth,
        np.array([[0.5, 0.0, 0.0], [0.25, 0.0, 0.0]], dtype=np.float32),
    )


def test_smallest_normal_metric_depth_retains_its_finite_reciprocal() -> None:
    depth = np.array([[np.finfo(np.float32).tiny]], dtype=np.float32)
    frame = metric_frame_from_depth(depth, np.float32(0.8))
    assert frame.valid.tolist() == [[True]]
    assert np.isfinite(frame.inverse_depth).all()
    assert frame.inverse_depth.item() == np.float32(1.0) / depth.item()


def test_metric_source_valid_predicate_uses_float32_reciprocal_boundary() -> None:
    smallest_subnormal = np.nextafter(np.float32(0.0), np.float32(1.0), dtype=np.float32)
    depth = np.array(
        [[np.finfo(np.float32).tiny, smallest_subnormal], [0.0, np.inf]],
        dtype=np.float32,
    )

    valid = metric_geometry.metric_source_valid(depth)

    assert valid.dtype == np.bool_
    assert valid.tolist() == [[True, False], [False, False]]


def test_clip_convergence_rejects_reciprocal_overflow_distance() -> None:
    smallest_subnormal = np.nextafter(np.float32(0.0), np.float32(1.0), dtype=np.float32)

    with pytest.raises(ValueError, match="finite float32 reciprocal"):
        ClipConvergence(smallest_subnormal, (0,), 1)


def test_metric_geometry_frame_rejects_nonzero_invalid_scores() -> None:
    with pytest.raises(ValueError, match="invalid locations"):
        MetricGeometryFrame(
            inverse_depth=np.ones((1, 1), dtype=np.float32),
            valid=np.zeros((1, 1), dtype=np.bool_),
            focal_x_normalized=np.float32(0.8),
        )


def test_metric_geometry_frame_owns_read_only_copies() -> None:
    inverse = np.array([[0.5]], dtype=np.float32)
    valid = np.array([[True]], dtype=np.bool_)
    frame = MetricGeometryFrame(inverse, valid, np.float32(0.8))
    inverse[0, 0] = 0.25
    valid[0, 0] = False
    assert frame.inverse_depth.item() == np.float32(0.5)
    assert frame.valid.item() is True
    with pytest.raises(ValueError, match="read-only"):
        frame.inverse_depth[0, 0] = np.float32(0.25)


@pytest.mark.parametrize(
    ("inverse", "valid", "focal", "message"),
    [
        (np.ones((1, 1), np.float64), np.ones((1, 1), np.bool_), np.float32(0.8), "float32"),
        (np.ones((1, 1), np.float32), np.ones((1, 1), np.uint8), np.float32(0.8), "bool"),
        (np.ones((1,), np.float32), np.ones((1,), np.bool_), np.float32(0.8), "2D"),
        (np.ones((1, 1), np.float32), np.ones((1, 2), np.bool_), np.float32(0.8), "shape"),
        (
            np.array([[np.inf]], np.float32),
            np.ones((1, 1), np.bool_),
            np.float32(0.8),
            "finite",
        ),
        (np.ones((1, 1), np.float32), np.ones((1, 1), np.bool_), np.float64(0.8), "float32"),
        (np.ones((1, 1), np.float32), np.ones((1, 1), np.bool_), np.float32(0.0), "positive"),
    ],
)
def test_metric_geometry_frame_rejects_invalid_contract(
    inverse: np.ndarray,
    valid: np.ndarray,
    focal: np.floating,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        MetricGeometryFrame(inverse, valid, focal)  # type: ignore[arg-type]


def test_clip_convergence_is_one_source_ordered_float32_median(
    raw_metric_store: RawDepthStore,
) -> None:
    result = sample_clip_convergence(
        raw_metric_store,
        raw_metric_store.complete_files,
        candidate_scene_ids=[0, 0, 1, 1],
    )
    assert result.distance_m.dtype == np.float32
    assert result.distance_m == np.float32(5.0)
    assert result.selected_frame_indexes == (0, 1, 2, 3)
    assert result.sample_count == 4


def test_convergence_selects_at_most_32_frames_per_candidate_scene() -> None:
    scene_ids = [0] * 100 + [1] * 100
    indexes = select_convergence_frame_indexes(scene_ids)
    assert len(indexes) == 64
    assert indexes == tuple(sorted(indexes))
    assert sum(index < 100 for index in indexes) == 32


def test_no_valid_metric_sample_is_a_hard_error(
    raw_store_with_only_invalid_depth: RawDepthStore,
) -> None:
    with pytest.raises(ValueError, match="No valid positive metric depth"):
        sample_clip_convergence(
            raw_store_with_only_invalid_depth,
            raw_store_with_only_invalid_depth.complete_files,
            candidate_scene_ids=[0],
        )


def test_reciprocal_overflow_metric_sample_is_a_hard_error(tmp_path: Path) -> None:
    smallest_subnormal = np.nextafter(np.float32(0.0), np.float32(1.0), dtype=np.float32)
    store = _raw_store_for_depths(tmp_path / "overflow-raw", np.array([smallest_subnormal]))

    with pytest.raises(ValueError, match="No valid positive metric depth"):
        sample_clip_convergence(store, store.complete_files, candidate_scene_ids=[0])


@pytest.mark.parametrize(
    "distance_m",
    [np.float32(0.05), np.float32(1001.0), np.float32(np.finfo(np.float32).tiny)],
    ids=["below-explicit-minimum", "above-explicit-maximum", "smallest-normal"],
)
def test_source_valid_auto_convergence_flows_from_stage3_into_stage4(
    tmp_path: Path, distance_m: np.float32
) -> None:
    store = _raw_store_for_depths(tmp_path / "auto-raw", np.array([distance_m]))

    convergence = sample_clip_convergence(store, store.complete_files, candidate_scene_ids=[0])
    frame = metric_frame_from_depth(np.array([[distance_m]], np.float32), np.float32(0.8))
    geometry, stats = build_metric_geometry(
        frame.inverse_depth,
        frame.valid,
        frame.focal_x_normalized,
        (1, 1),
        virtual_baseline_mm=63.0,
        convergence_distance_m=float(convergence.distance_m),
        max_disparity_percent=2.0,
        retained_crop_width=1,
    )

    assert convergence.distance_m == distance_m
    assert geometry.source_valid.tolist() == [[True]]
    assert np.isfinite(geometry.total_disparity_fraction).all()
    assert stats.valid_pixel_count == 1


def test_convergence_rejects_manifest_length_mismatch(raw_metric_store: RawDepthStore) -> None:
    with pytest.raises(ValueError, match="same length"):
        sample_clip_convergence(
            raw_metric_store,
            raw_metric_store.complete_files,
            candidate_scene_ids=[0],
        )


def test_metric_disk_bound_uses_exact_uncompressed_formula() -> None:
    assert estimate_metric_geometry_disk_bytes([(2, 3), (4, 5)], allocation_unit=4096) == 16_781_600
    assert (
        estimate_metric_geometry_disk_bytes(
            [(2, 3), (4, 5)],
            allocation_unit=4096,
            include_visual_previews=True,
        )
        == 16_781_657
    )
    assert estimate_metric_geometry_disk_bytes([], allocation_unit=4096) == 0


def test_allocation_unit_falls_back_to_64_kib(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "os.statvfs",
        lambda _path: (_ for _ in ()).throw(OSError("no statvfs")),
        raising=False,
    )
    monkeypatch.setattr(metric_geometry, "_windows_allocation_unit", lambda _path: None)
    assert filesystem_allocation_unit(tmp_path) == 65_536


def test_windows_allocation_unit_queries_folder_mounted_volume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, str] = {}

    class Kernel32:
        def GetVolumePathNameW(self, source, volume_path, _length) -> int:
            calls["source"] = source.value
            volume_path.value = "C:\\mounted-volume\\"
            return 1

        def GetDiskFreeSpaceW(
            self, volume_path, sectors, bytes_per_sector, free_clusters, total_clusters
        ) -> int:
            calls["volume_path"] = volume_path.value
            sectors._obj.value = 8
            bytes_per_sector._obj.value = 4096
            free_clusters._obj.value = 10
            total_clusters._obj.value = 20
            return 1

    monkeypatch.setattr(metric_geometry.ctypes, "WinDLL", lambda *_args, **_kwargs: Kernel32())

    assert metric_geometry._windows_allocation_unit(tmp_path / "folder") == 32_768
    assert calls["source"] == str((tmp_path / "folder").resolve())
    assert calls["volume_path"] == "C:\\mounted-volume\\"


@pytest.mark.parametrize("failure_point", ["volume", "disk", "invalid", "exception"])
def test_windows_allocation_query_failure_or_invalid_value_uses_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure_point: str
) -> None:
    class Kernel32:
        def GetVolumePathNameW(self, _source, volume_path, _length) -> int:
            if failure_point == "exception":
                raise TypeError("unavailable Windows API")
            volume_path.value = "C:\\mounted-volume\\"
            return int(failure_point != "volume")

        def GetDiskFreeSpaceW(
            self, _volume_path, sectors, bytes_per_sector, _free_clusters, _total_clusters
        ) -> int:
            sectors._obj.value = 0 if failure_point == "invalid" else 8
            bytes_per_sector._obj.value = 4096
            return int(failure_point != "disk")

    monkeypatch.setattr(metric_geometry.ctypes, "WinDLL", lambda *_args, **_kwargs: Kernel32())
    assert metric_geometry._windows_allocation_unit(tmp_path) is None
    monkeypatch.setattr(metric_geometry, "_windows_allocation_unit", lambda _path: None)
    monkeypatch.setattr(
        "os.statvfs",
        lambda _path: (_ for _ in ()).throw(OSError("no statvfs")),
        raising=False,
    )
    assert filesystem_allocation_unit(tmp_path) == 65_536


def test_preflight_disk_error_carries_exact_space_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        metric_geometry.shutil,
        "disk_usage",
        lambda _path: type(usage)(usage.total, usage.used, 99),
    )
    with pytest.raises(MetricGeometryDiskFullError) as raised:
        require_metric_geometry_disk_space(tmp_path / "metric", 100)
    assert raised.value.required_bytes == 100
    assert raised.value.free_bytes == 99
    assert raised.value.failing_path == tmp_path / "metric"


def test_new_and_complete_metadata_have_distinct_transaction_fields(tmp_path: Path) -> None:
    store = _create_metric_store(tmp_path / "metric", frame_names=[FRAME_NAMES[0]])
    assert store.metadata["status"] == "writing"
    assert store.metadata["representation"] == "metric_inverse_depth"
    assert store.metadata["near_value"] == "larger"
    assert store.metadata["preflight_required_bytes"] == 1024
    assert "fingerprint" not in store.metadata
    store.write_frame(FRAME_NAMES[0], VALID_FRAME)

    completed = store.finalize(ClipConvergence(np.float32(2.0), (0,), 1))

    assert completed["status"] == "complete"
    assert "preflight_required_bytes" not in completed
    assert completed["convergence"]["algorithm_version"] == METRIC_CONVERGENCE_ALGORITHM_VERSION
    assert completed["fingerprint"] == canonical_json_hash(
        {key: value for key, value in completed.items() if key != "fingerprint"}
    )


def test_metadata_read_treats_only_absence_and_malformed_json_as_missing(
    tmp_path: Path,
) -> None:
    assert MetricGeometryStore.read_metadata(tmp_path) is None
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("{", encoding="utf-8")
    assert MetricGeometryStore.read_metadata(tmp_path) is None


def test_metadata_read_preserves_non_absence_oserror_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    failure = OSError(errno.EIO, "metadata I/O failure")

    def fail_read(_path: Path, *_args, **_kwargs) -> str:
        raise failure

    monkeypatch.setattr(Path, "read_text", fail_read)
    with pytest.raises(OSError) as raised:
        MetricGeometryStore.read_metadata(tmp_path)
    assert raised.value is failure
    assert raised.value.__cause__ is None


def test_metric_payload_has_exact_ordered_npz_members(tmp_path: Path) -> None:
    store = _create_metric_store(tmp_path, frame_names=[FRAME_NAMES[0]])
    path = store.write_frame(FRAME_NAMES[0], VALID_FRAME)
    with zipfile.ZipFile(path) as payload:
        assert payload.namelist() == [
            "inverse_depth.npy",
            "valid.npy",
            "focal_x_normalized.npy",
        ]


@pytest.mark.parametrize(
    ("arrays", "message"),
    [
        (
            {
                "inverse_depth": np.ones((1, 1), np.float32),
                "valid": np.ones((1, 1), np.bool_),
            },
            "exact members",
        ),
        (
            {
                "inverse_depth": np.ones((1, 1), np.float64),
                "valid": np.ones((1, 1), np.bool_),
                "focal_x_normalized": np.array(0.8, np.float32),
            },
            "inverse_depth.*float32",
        ),
        (
            {
                "inverse_depth": np.ones((1, 1), np.float32),
                "valid": np.ones((1, 1), np.uint8),
                "focal_x_normalized": np.array(0.8, np.float32),
            },
            "valid.*bool",
        ),
        (
            {
                "inverse_depth": np.ones((1, 1), np.float32),
                "valid": np.ones((1, 1), np.bool_),
                "focal_x_normalized": np.array([0.8], np.float32),
            },
            "scalar",
        ),
    ],
)
def test_metric_store_rejects_corrupt_payload_contract(
    partial_metric_store: MetricGeometryStore,
    arrays: dict[str, np.ndarray],
    message: str,
) -> None:
    _write_npz(partial_metric_store.path_for(FRAME_NAMES[0]), arrays)
    with pytest.raises(ValueError, match=message):
        partial_metric_store.validate_payloads()


def test_payload_header_validation_preserves_oserror_identity(
    monkeypatch: pytest.MonkeyPatch, partial_metric_store: MetricGeometryStore
) -> None:
    failure = OSError(errno.EIO, "payload header I/O failure")

    def fail_zip(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(metric_geometry.zipfile, "ZipFile", fail_zip)
    with pytest.raises(OSError) as raised:
        partial_metric_store.validate_payloads()
    assert raised.value is failure
    assert raised.value.__cause__ is None


def test_payload_body_read_preserves_oserror_identity(
    monkeypatch: pytest.MonkeyPatch, partial_metric_store: MetricGeometryStore
) -> None:
    path = partial_metric_store.path_for(FRAME_NAMES[0])
    failure = OSError(errno.EIO, "payload body I/O failure")

    def fail_load(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(metric_geometry.np, "load", fail_load)
    with pytest.raises(OSError) as raised:
        partial_metric_store.load(path)
    assert raised.value is failure
    assert raised.value.__cause__ is None


def test_payload_presence_check_preserves_oserror_identity(
    monkeypatch: pytest.MonkeyPatch, partial_metric_store: MetricGeometryStore
) -> None:
    path = partial_metric_store.path_for(FRAME_NAMES[0])
    failure = PermissionError(errno.EACCES, "payload stat denied")
    real_stat = Path.stat

    def fail_payload_stat(candidate: Path, *args, **kwargs):
        if candidate == path:
            raise failure
        return real_stat(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fail_payload_stat)
    with pytest.raises(PermissionError) as raised:
        partial_metric_store.validate_payloads()
    assert raised.value is failure
    assert raised.value.__cause__ is None


def test_corrupt_npz_archive_is_reported_as_value_error(
    partial_metric_store: MetricGeometryStore,
) -> None:
    partial_metric_store.path_for(FRAME_NAMES[0]).write_bytes(b"not a zip archive")
    with pytest.raises(ValueError, match="payload is unreadable"):
        partial_metric_store.validate_payloads()


def test_metric_store_rejects_changed_candidate_scene_identity(
    complete_metric_store: MetricGeometryStore,
) -> None:
    with pytest.raises(ValueError, match="candidate scene fingerprint"):
        MetricGeometryStore.open_existing(
            complete_metric_store.directory,
            frame_names=complete_metric_store.metadata["frame_names"],
            source_raw_fingerprint=complete_metric_store.metadata["source_raw_fingerprint"],
            source_frame_fingerprint=complete_metric_store.metadata["source_frame_fingerprint"],
            candidate_scene_fingerprint="changed",
        )


@pytest.mark.parametrize(
    ("frame_names", "raw_fingerprint", "frame_fingerprint", "message"),
    [
        (["other.png"], "raw-fingerprint", "frame-fingerprint", "ordered frame names"),
        ([FRAME_NAMES[0]], "other-raw", "frame-fingerprint", "source raw fingerprint"),
        ([FRAME_NAMES[0]], "raw-fingerprint", "other-frame", "source frame fingerprint"),
    ],
)
def test_metric_store_rejects_every_changed_source_identity(
    complete_metric_store: MetricGeometryStore,
    frame_names: list[str],
    raw_fingerprint: str,
    frame_fingerprint: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MetricGeometryStore.open_existing(
            complete_metric_store.directory,
            frame_names=frame_names,
            source_raw_fingerprint=raw_fingerprint,
            source_frame_fingerprint=frame_fingerprint,
            candidate_scene_fingerprint="scene-fingerprint",
        )


def test_complete_store_rejects_metadata_fingerprint_tampering(
    complete_metric_store: MetricGeometryStore,
) -> None:
    complete_metric_store.metadata["convergence"]["resolved_auto_distance_m"] = 3.0
    complete_metric_store.metadata_path.write_text(
        json.dumps(complete_metric_store.metadata), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="completion fingerprint"):
        MetricGeometryStore.open_existing(
            complete_metric_store.directory,
            frame_names=[FRAME_NAMES[0]],
            source_raw_fingerprint="raw-fingerprint",
            source_frame_fingerprint="frame-fingerprint",
            candidate_scene_fingerprint="scene-fingerprint",
        )


def test_open_existing_requires_complete_status(partial_metric_store: MetricGeometryStore) -> None:
    with pytest.raises(ValueError, match="complete"):
        MetricGeometryStore.open_existing(
            partial_metric_store.directory,
            frame_names=FRAME_NAMES,
            source_raw_fingerprint="raw-fingerprint",
            source_frame_fingerprint="frame-fingerprint",
            candidate_scene_fingerprint="scene-fingerprint",
        )


def test_resume_rejects_smaller_preflight_and_persists_larger_estimate(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "metric"
    _create_metric_store(directory, preflight_required_bytes=100)
    with pytest.raises(ValueError, match="preflight"):
        _create_metric_store(directory, preflight_required_bytes=99)
    resumed = _create_metric_store(directory, preflight_required_bytes=150)
    persisted = json.loads(resumed.metadata_path.read_text(encoding="utf-8"))
    assert resumed.metadata["preflight_required_bytes"] == 150
    assert persisted["preflight_required_bytes"] == 150
    assert "fingerprint" not in persisted


def test_complete_store_identity_excludes_later_preflight_estimate(
    complete_metric_store: MetricGeometryStore,
) -> None:
    before = complete_metric_store.metadata_path.read_bytes()
    reopened = MetricGeometryStore.open_or_create(
        complete_metric_store.directory,
        frame_names=[FRAME_NAMES[0]],
        native_shape=(1, 1),
        source_raw_fingerprint="raw-fingerprint",
        source_frame_fingerprint="frame-fingerprint",
        candidate_scene_fingerprint="scene-fingerprint",
        preflight_required_bytes=999_999,
    )
    assert reopened.metadata_path.read_bytes() == before


def test_interrupted_temporary_is_removed_without_touching_commits(
    partial_metric_store: MetricGeometryStore,
) -> None:
    temporary = partial_metric_store.directory / "frame_000002.npz.tmp"
    temporary.write_bytes(b"incomplete")
    assert partial_metric_store.validate_payloads() == 1
    assert partial_metric_store.path_for(FRAME_NAMES[0]).is_file()
    assert not temporary.exists()


def test_complete_files_are_always_returned_in_source_order(tmp_path: Path) -> None:
    store = _create_metric_store(tmp_path / "ordered")
    store.write_frame(FRAME_NAMES[1], VALID_FRAME)
    store.write_frame(FRAME_NAMES[0], VALID_FRAME)
    store.finalize(ClipConvergence(np.float32(2.0), (0, 1), 2))
    assert store.complete_files == (
        store.path_for(FRAME_NAMES[0]),
        store.path_for(FRAME_NAMES[1]),
    )


@pytest.mark.parametrize("error_kind", ["enospc", "windows"])
def test_initial_directory_creation_normalizes_disk_full(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error_kind: str
) -> None:
    directory = tmp_path / "missing" / "metric"
    failure = _disk_full_failure(error_kind)
    real_mkdir = Path.mkdir

    def fail_target(path: Path, *args, **kwargs) -> None:
        if path == directory:
            raise failure
        real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_target)
    with pytest.raises(MetricGeometryDiskFullError) as raised:
        _create_metric_store(directory, preflight_required_bytes=321)
    assert raised.value.required_bytes == 321
    assert raised.value.failing_path == directory
    assert raised.value.__cause__ is failure


@pytest.mark.parametrize("error_kind", ["enospc", "windows"])
def test_initial_metadata_creation_normalizes_disk_full(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error_kind: str
) -> None:
    directory = tmp_path / "initial-metadata"
    directory.mkdir()
    failure = _disk_full_failure(error_kind)

    def fail_replace(_path: Path, destination: Path) -> Path:
        if Path(destination) == directory / "metadata.json":
            raise failure
        raise AssertionError(f"Unexpected replace destination: {destination}")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(MetricGeometryDiskFullError) as raised:
        _create_metric_store(directory, preflight_required_bytes=654)
    assert raised.value.required_bytes == 654
    assert raised.value.failing_path == directory / "metadata.json"
    assert raised.value.__cause__ is failure
    assert not (directory / "metadata.json").exists()
    assert not list(directory.glob("*.tmp"))


@pytest.mark.parametrize("failure_point", ["directory", "metadata"])
def test_initial_creation_preserves_every_other_oserror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure_point: str
) -> None:
    directory = tmp_path / "other-oserror"
    failure = OSError(errno.EIO, f"{failure_point} I/O failure")
    if failure_point == "directory":
        real_mkdir = Path.mkdir

        def fail_target(path: Path, *args, **kwargs) -> None:
            if path == directory:
                raise failure
            real_mkdir(path, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", fail_target)
    else:
        directory.mkdir()

        def fail_replace(_path: Path, _destination: Path) -> Path:
            raise failure

        monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError) as raised:
        _create_metric_store(directory, preflight_required_bytes=777)
    assert raised.value is failure
    assert raised.value.__cause__ is None
    if directory.exists():
        assert not list(directory.glob("*.tmp"))


def test_enospc_removes_only_current_temp_and_keeps_committed_frames(
    monkeypatch: pytest.MonkeyPatch, partial_metric_store: MetricGeometryStore
) -> None:
    committed = partial_metric_store.path_for(FRAME_NAMES[0])
    failing = partial_metric_store.path_for(FRAME_NAMES[1])
    real_replace = Path.replace

    def fail_second_replace(path: Path, destination: Path) -> Path:
        if Path(destination) == failing:
            raise OSError(errno.ENOSPC, "disk full")
        return real_replace(path, destination)

    monkeypatch.setattr(Path, "replace", fail_second_replace)
    with pytest.raises(MetricGeometryDiskFullError) as raised:
        partial_metric_store.write_frame(FRAME_NAMES[1], VALID_FRAME)
    assert committed.is_file()
    assert not failing.is_file()
    assert not list(partial_metric_store.directory.glob("*.tmp"))
    assert partial_metric_store.metadata.get("status") == "writing"
    assert "fingerprint" not in partial_metric_store.metadata
    assert raised.value.required_bytes == 1024
    assert raised.value.failing_path == failing


def test_disk_full_samples_free_space_before_current_temp_cleanup(
    monkeypatch: pytest.MonkeyPatch, partial_metric_store: MetricGeometryStore
) -> None:
    failing = partial_metric_store.path_for(FRAME_NAMES[1])
    usage = shutil.disk_usage(partial_metric_store.directory)
    observed_temp_sizes: list[int] = []

    def fail_replace(_path: Path, destination: Path) -> Path:
        if Path(destination) == failing:
            raise OSError(errno.ENOSPC, "disk full")
        raise AssertionError(f"Unexpected replace destination: {destination}")

    def free_depends_on_temp(_path: Path):
        temp_size = sum(
            path.stat().st_size for path in partial_metric_store.directory.glob("*.tmp")
        )
        observed_temp_sizes.append(temp_size)
        free = 11 if temp_size > 0 else 99
        return type(usage)(usage.total, usage.used, free)

    monkeypatch.setattr(Path, "replace", fail_replace)
    monkeypatch.setattr(metric_geometry.shutil, "disk_usage", free_depends_on_temp)
    with pytest.raises(MetricGeometryDiskFullError) as raised:
        partial_metric_store.write_frame(FRAME_NAMES[1], VALID_FRAME)
    assert observed_temp_sizes and observed_temp_sizes[0] > 0
    assert raised.value.free_bytes == 11
    assert not list(partial_metric_store.directory.glob("*.tmp"))


@pytest.mark.parametrize("error_kind", ["enospc", "windows"])
def test_finalize_disk_full_keeps_writing_metadata_and_all_payloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error_kind: str
) -> None:
    store = _create_metric_store(tmp_path / "finalize", frame_names=[FRAME_NAMES[0]])
    payload = store.write_frame(FRAME_NAMES[0], VALID_FRAME)
    real_replace = Path.replace
    failure = _disk_full_failure(error_kind)

    def fail_metadata_replace(path: Path, destination: Path) -> Path:
        if Path(destination) == store.metadata_path:
            raise failure
        return real_replace(path, destination)

    monkeypatch.setattr(Path, "replace", fail_metadata_replace)
    with pytest.raises(MetricGeometryDiskFullError) as raised:
        store.finalize(ClipConvergence(np.float32(2.0), (0,), 1))
    persisted = json.loads(store.metadata_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "writing"
    assert "fingerprint" not in persisted
    assert payload.is_file()
    assert not list(store.directory.glob("*.tmp"))
    assert raised.value.required_bytes == persisted["preflight_required_bytes"]
    assert raised.value.__cause__ is failure


@pytest.mark.parametrize("error_kind", ["enospc", "windows"])
def test_preflight_update_disk_full_reports_the_persisted_current_estimate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error_kind: str
) -> None:
    directory = tmp_path / "preflight-update"
    store = _create_metric_store(directory, preflight_required_bytes=100)
    failure = _disk_full_failure(error_kind)

    def fail_replace(_path: Path, _destination: Path) -> Path:
        raise failure

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(MetricGeometryDiskFullError) as raised:
        _create_metric_store(directory, preflight_required_bytes=200)
    persisted = json.loads(store.metadata_path.read_text(encoding="utf-8"))
    assert persisted["preflight_required_bytes"] == 100
    assert raised.value.required_bytes == 100
    assert raised.value.__cause__ is failure
    assert not list(directory.glob("*.tmp"))


def test_windows_error_disk_full_112_uses_same_restartable_error(
    monkeypatch: pytest.MonkeyPatch, partial_metric_store: MetricGeometryStore
) -> None:
    failure = OSError("windows disk full")
    failure.winerror = 112  # type: ignore[attr-defined]

    def fail_replace(_path: Path, _destination: Path) -> Path:
        raise failure

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(MetricGeometryDiskFullError) as raised:
        partial_metric_store.write_frame(FRAME_NAMES[1], VALID_FRAME)
    assert raised.value.failing_path == partial_metric_store.path_for(FRAME_NAMES[1])
    assert raised.value.required_bytes == 1024
    assert partial_metric_store.path_for(FRAME_NAMES[0]).is_file()
    assert not list(partial_metric_store.directory.glob("*.tmp"))


def test_non_disk_full_oserror_is_reraised_unchanged(
    monkeypatch: pytest.MonkeyPatch, partial_metric_store: MetricGeometryStore
) -> None:
    failure = OSError(errno.EACCES, "denied")

    def fail_replace(_path: Path, _destination: Path) -> Path:
        raise failure

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError) as raised:
        partial_metric_store.write_frame(FRAME_NAMES[1], VALID_FRAME)
    assert raised.value is failure
    assert not list(partial_metric_store.directory.glob("*.tmp"))
