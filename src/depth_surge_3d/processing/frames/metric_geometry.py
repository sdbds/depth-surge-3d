"""Restartable native-resolution metric geometry and clip convergence."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Any, Sequence
import zipfile

import numpy as np

from ...core.depth_contract import canonical_json_hash
from ...inference.depth.types import DepthRepresentation
from .depth_storage import RawDepthStore


METRIC_GEOMETRY_SCHEMA_VERSION = 1
METRIC_GEOMETRY_ALGORITHM_VERSION = "metric-inverse-depth-v1"
METRIC_CONVERGENCE_ALGORITHM_VERSION = "clip-scene-grid-median-v1"

_FALLBACK_ALLOCATION_UNIT = 64 * 1024
_METRIC_PAYLOAD_MEMBERS = [
    "inverse_depth.npy",
    "valid.npy",
    "focal_x_normalized.npy",
]


def _validate_inverse_array(inverse: np.ndarray) -> None:
    if not isinstance(inverse, np.ndarray):
        raise TypeError("Metric inverse_depth must be a numpy array")
    if inverse.dtype != np.float32:
        raise TypeError("Metric inverse_depth must use float32")
    if inverse.ndim != 2:
        raise ValueError("Metric inverse_depth must be 2D")
    if not np.isfinite(inverse).all():
        raise ValueError("Metric inverse_depth values must be finite")


def _validate_valid_array(valid: np.ndarray) -> None:
    if not isinstance(valid, np.ndarray):
        raise TypeError("Metric valid must be a numpy array")
    if valid.dtype != np.bool_:
        raise TypeError("Metric valid must use bool")
    if valid.ndim != 2:
        raise ValueError("Metric valid must be 2D")


def _validate_focal(focal: np.float32) -> None:
    if not isinstance(focal, np.float32):
        raise TypeError("Metric focal_x_normalized must use float32")
    if not np.isfinite(focal):
        raise ValueError("Metric focal_x_normalized must be finite")
    if focal <= np.float32(0.0):
        raise ValueError("Metric focal_x_normalized must be positive")


def metric_source_valid(depth: np.ndarray) -> np.ndarray:
    """Return elementwise metric validity after a float32 reciprocal probe."""

    if not isinstance(depth, np.ndarray):
        raise TypeError("Metric depth must be a numpy array")
    if depth.dtype != np.float32:
        raise TypeError("Metric depth must use float32")
    valid = np.isfinite(depth) & (depth > np.float32(0.0))
    reciprocal = np.zeros(depth.shape, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(np.float32(1.0), depth, out=reciprocal, where=valid)
    valid &= np.isfinite(reciprocal)
    return valid


@dataclass(frozen=True)
class MetricGeometryFrame:
    """Owned immutable metric inverse depth, validity, and pinhole focal data."""

    inverse_depth: np.ndarray
    valid: np.ndarray
    focal_x_normalized: np.float32

    def __post_init__(self) -> None:
        inverse = self.inverse_depth
        valid = self.valid
        focal = self.focal_x_normalized
        _validate_inverse_array(inverse)
        _validate_valid_array(valid)
        if inverse.shape != valid.shape:
            raise ValueError("Metric inverse_depth and valid shape must match")
        if np.any(inverse[~valid] != np.float32(0.0)):
            raise ValueError("Metric inverse_depth must be zero at invalid locations")
        if np.any(inverse[valid] <= np.float32(0.0)):
            raise ValueError("Valid metric inverse_depth values must be positive")
        _validate_focal(focal)

        owned_inverse = np.array(inverse, dtype=np.float32, copy=True, order="C")
        owned_valid = np.array(valid, dtype=np.bool_, copy=True, order="C")
        owned_inverse.setflags(write=False)
        owned_valid.setflags(write=False)
        object.__setattr__(self, "inverse_depth", owned_inverse)
        object.__setattr__(self, "valid", owned_valid)
        object.__setattr__(self, "focal_x_normalized", np.float32(focal))


@dataclass(frozen=True)
class ClipConvergence:
    """One deterministic clip-global metric convergence result."""

    distance_m: np.float32
    selected_frame_indexes: tuple[int, ...]
    sample_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.distance_m, np.float32):
            raise TypeError("Clip convergence distance_m must use float32")
        if not bool(metric_source_valid(np.asarray(self.distance_m)).item()):
            raise ValueError(
                "Clip convergence distance_m must be finite and positive with a finite "
                "float32 reciprocal"
            )
        if not isinstance(self.selected_frame_indexes, tuple):
            raise TypeError("Clip convergence frame indexes must be a tuple")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0
            for value in self.selected_frame_indexes
        ):
            raise ValueError("Clip convergence frame indexes must be nonnegative integers")
        normalized_indexes = tuple(int(value) for value in self.selected_frame_indexes)
        if normalized_indexes != tuple(sorted(set(normalized_indexes))):
            raise ValueError("Clip convergence frame indexes must be unique and source ordered")
        if isinstance(self.sample_count, bool) or not isinstance(
            self.sample_count, (int, np.integer)
        ):
            raise TypeError("Clip convergence sample_count must be an integer")
        if self.sample_count <= 0:
            raise ValueError("Clip convergence sample_count must be positive")
        if not normalized_indexes:
            raise ValueError("Clip convergence must select at least one frame")
        object.__setattr__(self, "selected_frame_indexes", normalized_indexes)
        object.__setattr__(self, "sample_count", int(self.sample_count))


class MetricGeometryDiskFullError(OSError):
    """Disk exhaustion with the persisted preflight and current free space."""

    required_bytes: int
    free_bytes: int
    failing_path: Path

    def __init__(self, required_bytes: int, free_bytes: int, failing_path: Path) -> None:
        self.required_bytes = int(required_bytes)
        self.free_bytes = int(free_bytes)
        self.failing_path = Path(failing_path)
        message = (
            "Insufficient disk space for metric geometry: "
            f"required {self.required_bytes} bytes, free {self.free_bytes} bytes, "
            f"failing path {self.failing_path}"
        )
        super().__init__(errno.ENOSPC, message, str(self.failing_path))


def metric_frame_from_depth(
    depth: np.ndarray, focal_x_normalized: np.float32
) -> MetricGeometryFrame:
    """Derive finite float32 inverse depth without admitting reciprocal overflow."""

    if not isinstance(depth, np.ndarray):
        raise TypeError("Metric depth must be a numpy array")
    if depth.dtype != np.float32:
        raise TypeError("Metric depth must use float32")
    if depth.ndim != 2:
        raise ValueError("Metric depth must be 2D")
    valid = metric_source_valid(depth)
    inverse = np.zeros(depth.shape, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(np.float32(1.0), depth, out=inverse, where=valid)
    return MetricGeometryFrame(inverse, valid, focal_x_normalized)


def select_convergence_frame_indexes(scene_ids: Sequence[int]) -> tuple[int, ...]:
    """Select no more than 32 source-ordered frames from each candidate scene."""

    selected: list[int] = []
    scene_array = np.asarray(scene_ids)
    for scene_id in dict.fromkeys(int(value) for value in scene_ids):
        source_indexes = np.flatnonzero(scene_array == scene_id)
        if len(source_indexes) > 32:
            positions = np.unique(
                np.rint(np.linspace(0, len(source_indexes) - 1, 32)).astype(np.int64)
            )
            source_indexes = source_indexes[positions]
        selected.extend(int(value) for value in source_indexes)
    return tuple(sorted(selected))


def _grid_indexes(size: int) -> np.ndarray:
    return np.rint(np.linspace(0, size - 1, min(size, 64))).astype(np.int64)


def sample_clip_convergence(
    raw_store: RawDepthStore,
    raw_files: Sequence[Path],
    candidate_scene_ids: Sequence[int],
) -> ClipConvergence:
    """Resolve one float32 median from source-ordered scene/frame/grid samples."""

    if len(raw_files) != len(candidate_scene_ids):
        raise ValueError("Raw files and candidate scene IDs must have the same length")
    selected_indexes = select_convergence_frame_indexes(candidate_scene_ids)
    samples: list[np.ndarray] = []
    for frame_index in selected_indexes:
        batch = raw_store.load_batch([Path(raw_files[frame_index])])
        if batch.representation is not DepthRepresentation.METRIC_DEPTH:
            raise ValueError("Clip convergence requires metric raw depth")
        if len(batch.values) != 1:
            raise ValueError("Clip convergence raw loads must contain one frame")
        depth = batch.values[0]
        source_valid = metric_source_valid(depth)
        row_indexes = _grid_indexes(depth.shape[0])
        column_indexes = _grid_indexes(depth.shape[1])
        grid = depth[np.ix_(row_indexes, column_indexes)].reshape(-1)
        valid = source_valid[np.ix_(row_indexes, column_indexes)].reshape(-1)
        if np.any(valid):
            samples.append(grid[valid].astype(np.float32, copy=False))
    if not samples:
        raise ValueError("No valid positive metric depth samples for clip convergence")
    combined = np.concatenate(samples).astype(np.float32, copy=False)
    distance = np.float32(np.median(combined))
    if not bool(metric_source_valid(np.asarray(distance)).item()):
        raise ValueError("No valid positive metric depth median for clip convergence")
    return ClipConvergence(distance, selected_indexes, int(combined.size))


def _ceil_five_quarters(value: int) -> int:
    return (5 * value + 3) // 4


def estimate_metric_geometry_disk_bytes(
    frame_shapes: Sequence[tuple[int, int]], *, allocation_unit: int
) -> int:
    """Return the exact conservative uncompressed metric-stage disk bound."""

    if not frame_shapes:
        return 0
    allocation = max(4096, int(allocation_unit))
    normalized_shapes: list[tuple[int, int]] = []
    for shape in frame_shapes:
        if len(shape) != 2:
            raise ValueError("Metric frame shapes must contain height and width")
        height, width = shape
        if (
            isinstance(height, bool)
            or isinstance(width, bool)
            or not isinstance(height, (int, np.integer))
            or not isinstance(width, (int, np.integer))
            or height <= 0
            or width <= 0
        ):
            raise ValueError("Metric frame dimensions must be positive integers")
        normalized_shapes.append((int(height), int(width)))
    frame_payloads = [5 * height * width for height, width in normalized_shapes]
    payload_bound = sum(frame_payloads)
    metadata_bound = max(16 * 1024 * 1024, allocation * len(normalized_shapes))
    atomic_overlap = _ceil_five_quarters(max(frame_payloads)) + allocation
    return _ceil_five_quarters(payload_bound) + metadata_bound + atomic_overlap


def _windows_allocation_unit(directory: Path) -> int | None:
    """Query the Windows cluster size, or report that the query is unavailable."""

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_volume_path = kernel32.GetVolumePathNameW
        get_disk_free_space = kernel32.GetDiskFreeSpaceW
        resolved = str(Path(directory).resolve())
        volume_path = ctypes.create_unicode_buffer(32_768)
        found_volume = get_volume_path(
            ctypes.c_wchar_p(resolved),
            volume_path,
            ctypes.c_ulong(len(volume_path)),
        )
        if not found_volume or not volume_path.value:
            return None
        sectors_per_cluster = ctypes.c_ulong()
        bytes_per_sector = ctypes.c_ulong()
        free_clusters = ctypes.c_ulong()
        total_clusters = ctypes.c_ulong()
        succeeded = get_disk_free_space(
            ctypes.c_wchar_p(volume_path.value),
            ctypes.byref(sectors_per_cluster),
            ctypes.byref(bytes_per_sector),
            ctypes.byref(free_clusters),
            ctypes.byref(total_clusters),
        )
        if not succeeded:
            return None
        allocation = int(sectors_per_cluster.value) * int(bytes_per_sector.value)
        return allocation if allocation > 0 else None
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def filesystem_allocation_unit(directory: Path) -> int:
    """Return the platform filesystem allocation unit with a 64 KiB fallback."""

    allocation: int | None = None
    if os.name == "nt":
        allocation = _windows_allocation_unit(Path(directory))
    else:
        try:
            statvfs = getattr(os, "statvfs")
            queried = statvfs(Path(directory)).f_frsize
            allocation = int(queried) if int(queried) > 0 else None
        except (AttributeError, OSError, TypeError, ValueError):
            allocation = None
    return allocation if allocation is not None and allocation > 0 else _FALLBACK_ALLOCATION_UNIT


def _current_free_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def _nearest_existing_path(path: Path) -> Path:
    candidate = Path(path)
    while True:
        try:
            candidate.stat()
            return candidate
        except FileNotFoundError:
            parent = candidate.parent
            if parent == candidate:
                raise
            candidate = parent


def _is_regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.stat().st_mode)
    except FileNotFoundError:
        return False


def _mkdir_with_disk_full_context(
    directory: Path, *, required_bytes: int, failing_path: Path
) -> None:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        if not _is_disk_full(error):
            raise
        free_bytes = _current_free_bytes(_nearest_existing_path(directory.parent))
        raise MetricGeometryDiskFullError(required_bytes, free_bytes, failing_path) from error


def require_metric_geometry_disk_space(directory: Path, required_bytes: int) -> None:
    """Fail preflight with the exact estimate and current target-filesystem free bytes."""

    if isinstance(required_bytes, bool) or not isinstance(required_bytes, (int, np.integer)):
        raise TypeError("Metric geometry required bytes must be an integer")
    if required_bytes < 0:
        raise ValueError("Metric geometry required bytes must be nonnegative")
    directory = Path(directory)
    _mkdir_with_disk_full_context(
        directory,
        required_bytes=int(required_bytes),
        failing_path=directory,
    )
    free_bytes = _current_free_bytes(directory)
    if free_bytes < required_bytes:
        raise MetricGeometryDiskFullError(int(required_bytes), free_bytes, directory)


def _atomic_write_json(path: Path, payload: dict[str, Any], *, required_bytes: int) -> None:
    _mkdir_with_disk_full_context(
        path.parent,
        required_bytes=required_bytes,
        failing_path=path,
    )
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    except OSError as error:
        if not _is_disk_full(error):
            raise
        free_bytes = _current_free_bytes(path.parent)
        raise MetricGeometryDiskFullError(required_bytes, free_bytes, path) from error
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_save_npz(path: Path, frame: MetricGeometryFrame, *, required_bytes: int) -> None:
    _mkdir_with_disk_full_context(
        path.parent,
        required_bytes=required_bytes,
        failing_path=path,
    )
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                inverse_depth=frame.inverse_depth,
                valid=frame.valid,
                focal_x_normalized=np.asarray(frame.focal_x_normalized, dtype=np.float32),
            )
        temporary.replace(path)
    except OSError as error:
        if not _is_disk_full(error):
            raise
        free_bytes = _current_free_bytes(path.parent)
        raise MetricGeometryDiskFullError(required_bytes, free_bytes, path) from error
    finally:
        temporary.unlink(missing_ok=True)


def _is_disk_full(error: OSError) -> bool:
    return error.errno == errno.ENOSPC or getattr(error, "winerror", None) == 112


def _read_npy_header(member: Any) -> tuple[tuple[int, ...], np.dtype[Any]]:
    version = np.lib.format.read_magic(member)
    if version == (1, 0):
        shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(member)
    elif version in {(2, 0), (3, 0)}:
        shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(member)
    else:
        raise ValueError(f"Unsupported npy header version: {version}")
    return tuple(int(value) for value in shape), np.dtype(dtype)


def _validate_grid_header(
    *,
    label: str,
    header: tuple[tuple[int, ...], np.dtype[Any]],
    expected_dtype: np.dtype[Any],
    native_shape: tuple[int, ...],
    path: Path,
) -> None:
    shape, dtype = header
    if dtype != expected_dtype:
        raise ValueError(f"Metric geometry {label} must use {expected_dtype.name}: {path}")
    if len(shape) != 2 or shape != native_shape:
        raise ValueError(f"Metric geometry {label} shape mismatch: {path}")


def _validate_focal_header(header: tuple[tuple[int, ...], np.dtype[Any]], path: Path) -> None:
    shape, dtype = header
    if dtype != np.dtype(np.float32):
        raise ValueError(f"Metric geometry focal_x_normalized must use float32: {path}")
    if shape != ():
        raise ValueError(f"Metric geometry focal_x_normalized must be a scalar: {path}")


class MetricGeometryStore:
    """Atomic, restartable storage for one ordered metric-geometry stage."""

    def __init__(self, directory: Path, metadata: dict[str, Any]) -> None:
        self.directory = Path(directory)
        self.metadata_path = self.directory / "metadata.json"
        self.metadata = metadata

    @staticmethod
    def read_metadata(directory: Path) -> dict[str, Any] | None:
        path = Path(directory) / "metadata.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (UnicodeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _requested_identity(
        *,
        frame_names: list[str],
        native_shape: tuple[int, int],
        source_raw_fingerprint: str,
        source_frame_fingerprint: str,
        candidate_scene_fingerprint: str,
    ) -> dict[str, Any]:
        if not isinstance(frame_names, list) or not frame_names:
            raise ValueError("Metric geometry ordered frame names must be a nonempty list")
        if not all(isinstance(name, str) and Path(name).stem for name in frame_names):
            raise ValueError("Metric geometry frame names must be nonempty strings")
        payload_names = [f"{Path(name).stem}.npz" for name in frame_names]
        if len(set(payload_names)) != len(payload_names):
            raise ValueError("Metric geometry frame names must map to unique payloads")
        if (
            not isinstance(native_shape, tuple)
            or len(native_shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0
                for value in native_shape
            )
        ):
            raise ValueError("Metric geometry native shape must contain two positive integers")
        fingerprints = {
            "source_raw_fingerprint": source_raw_fingerprint,
            "source_frame_fingerprint": source_frame_fingerprint,
            "candidate_scene_fingerprint": candidate_scene_fingerprint,
        }
        if not all(isinstance(value, str) and value for value in fingerprints.values()):
            raise ValueError("Metric geometry source fingerprints must be nonempty strings")
        return {
            "frame_names": list(frame_names),
            "native_shape": [int(value) for value in native_shape],
            **fingerprints,
        }

    @classmethod
    def open_existing(
        cls,
        directory: Path,
        *,
        frame_names: list[str],
        source_raw_fingerprint: str,
        source_frame_fingerprint: str,
        candidate_scene_fingerprint: str,
        cleanup_temporaries: bool = True,
    ) -> "MetricGeometryStore":
        metadata = cls.read_metadata(directory)
        if metadata is None:
            raise ValueError("Metric geometry metadata is missing or malformed")
        native_shape = metadata.get("native_shape")
        if not isinstance(native_shape, list) or len(native_shape) != 2:
            raise ValueError("Metric geometry native shape metadata is invalid")
        requested = cls._requested_identity(
            frame_names=frame_names,
            native_shape=tuple(native_shape),  # type: ignore[arg-type]
            source_raw_fingerprint=source_raw_fingerprint,
            source_frame_fingerprint=source_frame_fingerprint,
            candidate_scene_fingerprint=candidate_scene_fingerprint,
        )
        store = cls(Path(directory), metadata)
        store._validate_metadata_base()
        if metadata.get("status") != "complete":
            raise ValueError("Metric geometry store is not complete")
        store._validate_complete_metadata()
        store._validate_expected_identity(requested)
        store.validate_payloads(cleanup_temporaries=cleanup_temporaries)
        return store

    @classmethod
    def open_or_create(
        cls,
        directory: Path,
        *,
        frame_names: list[str],
        native_shape: tuple[int, int],
        source_raw_fingerprint: str,
        source_frame_fingerprint: str,
        candidate_scene_fingerprint: str,
        preflight_required_bytes: int,
    ) -> "MetricGeometryStore":
        requested = cls._requested_identity(
            frame_names=frame_names,
            native_shape=native_shape,
            source_raw_fingerprint=source_raw_fingerprint,
            source_frame_fingerprint=source_frame_fingerprint,
            candidate_scene_fingerprint=candidate_scene_fingerprint,
        )
        if isinstance(preflight_required_bytes, bool) or not isinstance(
            preflight_required_bytes, (int, np.integer)
        ):
            raise TypeError("Metric geometry preflight estimate must be an integer")
        if preflight_required_bytes < 0:
            raise ValueError("Metric geometry preflight estimate must be nonnegative")
        directory = Path(directory)
        _mkdir_with_disk_full_context(
            directory,
            required_bytes=int(preflight_required_bytes),
            failing_path=directory,
        )
        metadata_path = directory / "metadata.json"
        metadata = cls.read_metadata(directory)
        if metadata is None:
            return cls._create_new(
                directory,
                metadata_path,
                requested,
                int(preflight_required_bytes),
            )
        return cls._resume(
            directory,
            metadata,
            requested,
            int(preflight_required_bytes),
        )

    @classmethod
    def _create_new(
        cls,
        directory: Path,
        metadata_path: Path,
        requested: dict[str, Any],
        preflight_required_bytes: int,
    ) -> "MetricGeometryStore":
        if _is_regular_file(metadata_path):
            raise ValueError("Metric geometry metadata is malformed")
        for temporary in directory.glob("*.tmp"):
            temporary.unlink(missing_ok=True)
        if any(directory.glob("*.npz")):
            raise ValueError("Metric geometry payloads exist without metadata")
        metadata = {
            "schema_version": METRIC_GEOMETRY_SCHEMA_VERSION,
            "algorithm_version": METRIC_GEOMETRY_ALGORITHM_VERSION,
            "status": "writing",
            "representation": "metric_inverse_depth",
            "near_value": "larger",
            **requested,
            "preflight_required_bytes": preflight_required_bytes,
            "storage_dtype": "float32",
            "compression": "npz_deflate",
        }
        _atomic_write_json(
            metadata_path,
            metadata,
            required_bytes=preflight_required_bytes,
        )
        return cls(directory, metadata)

    @classmethod
    def _resume(
        cls,
        directory: Path,
        metadata: dict[str, Any],
        requested: dict[str, Any],
        preflight_required_bytes: int,
    ) -> "MetricGeometryStore":
        store = cls(directory, metadata)
        store._validate_metadata_base()
        store._validate_expected_identity(requested)
        status = metadata.get("status")
        if status == "complete":
            store._validate_complete_metadata()
            store.validate_payloads()
            return store
        if status != "writing":
            raise ValueError("Metric geometry transaction status is invalid")
        store._validate_writing_metadata()
        persisted_required = metadata.get("preflight_required_bytes")
        assert isinstance(persisted_required, int)
        if preflight_required_bytes < persisted_required:
            raise ValueError("Metric geometry resumed preflight estimate cannot decrease")
        if preflight_required_bytes > persisted_required:
            updated = dict(metadata)
            updated["preflight_required_bytes"] = int(preflight_required_bytes)
            store._commit_metadata(updated)
        store.validate_payloads()
        return store

    def _validate_metadata_base(self) -> None:
        metadata = self.metadata
        static_fields = {
            "schema_version": METRIC_GEOMETRY_SCHEMA_VERSION,
            "algorithm_version": METRIC_GEOMETRY_ALGORITHM_VERSION,
            "representation": "metric_inverse_depth",
            "near_value": "larger",
            "storage_dtype": "float32",
            "compression": "npz_deflate",
        }
        labels = {
            "schema_version": "schema version",
            "algorithm_version": "algorithm version",
            "representation": "representation",
            "near_value": "near direction",
            "storage_dtype": "storage dtype",
            "compression": "compression",
        }
        for key, expected in static_fields.items():
            if metadata.get(key) != expected:
                raise ValueError(f"Metric geometry {labels[key]} mismatch")
        self._validate_manifest_metadata()
        self._validate_source_metadata()

    def _validate_manifest_metadata(self) -> None:
        frame_names = self.metadata.get("frame_names")
        native_shape = self.metadata.get("native_shape")
        if not isinstance(frame_names, list) or not frame_names:
            raise ValueError("Metric geometry ordered frame names are invalid")
        if not all(isinstance(name, str) and Path(name).stem for name in frame_names):
            raise ValueError("Metric geometry ordered frame names are invalid")
        payload_names = [f"{Path(name).stem}.npz" for name in frame_names]
        if len(payload_names) != len(set(payload_names)):
            raise ValueError("Metric geometry frame names do not map to unique payloads")
        if not isinstance(native_shape, list) or len(native_shape) != 2:
            raise ValueError("Metric geometry native shape metadata is invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in native_shape
        ):
            raise ValueError("Metric geometry native shape metadata is invalid")

    def _validate_source_metadata(self) -> None:
        for key in (
            "source_raw_fingerprint",
            "source_frame_fingerprint",
            "candidate_scene_fingerprint",
        ):
            if not isinstance(self.metadata.get(key), str) or not self.metadata[key]:
                raise ValueError(f"Metric geometry {key.replace('_', ' ')} is invalid")

    def _validate_expected_identity(self, requested: dict[str, Any]) -> None:
        comparisons = [
            ("frame_names", "ordered frame names"),
            ("native_shape", "native shape"),
            ("source_raw_fingerprint", "source raw fingerprint"),
            ("source_frame_fingerprint", "source frame fingerprint"),
            ("candidate_scene_fingerprint", "candidate scene fingerprint"),
        ]
        for key, label in comparisons:
            if self.metadata.get(key) != requested[key]:
                raise ValueError(f"Metric geometry {label} mismatch")

    def _validate_writing_metadata(self) -> None:
        if "fingerprint" in self.metadata:
            raise ValueError("Incomplete metric geometry must not have a completion fingerprint")
        if "convergence" in self.metadata:
            raise ValueError("Incomplete metric geometry must not have convergence metadata")
        required = self.metadata.get("preflight_required_bytes")
        if isinstance(required, bool) or not isinstance(required, int) or required < 0:
            raise ValueError("Metric geometry persisted preflight estimate is invalid")

    def _validate_complete_metadata(self) -> None:
        if "preflight_required_bytes" in self.metadata:
            raise ValueError("Complete metric geometry contains an operational preflight estimate")
        fingerprint = self.metadata.get("fingerprint")
        if not isinstance(fingerprint, str):
            raise ValueError("Metric geometry completion fingerprint is missing")
        without_fingerprint = {
            key: value for key, value in self.metadata.items() if key != "fingerprint"
        }
        if fingerprint != canonical_json_hash(without_fingerprint):
            raise ValueError("Metric geometry completion fingerprint mismatch")
        self._validate_convergence_metadata()

    def _validate_convergence_metadata(self) -> None:
        convergence = self.metadata.get("convergence")
        if not isinstance(convergence, dict) or set(convergence) != {
            "algorithm_version",
            "selected_frame_indexes",
            "sample_count",
            "resolved_auto_distance_m",
        }:
            raise ValueError("Metric geometry convergence metadata is invalid")
        if convergence.get("algorithm_version") != METRIC_CONVERGENCE_ALGORITHM_VERSION:
            raise ValueError("Metric geometry convergence algorithm version mismatch")
        self._validate_convergence_values(convergence)

    def _validate_convergence_values(self, convergence: dict[str, Any]) -> None:
        indexes = convergence.get("selected_frame_indexes")
        if not isinstance(indexes, list) or not indexes:
            raise ValueError("Metric geometry convergence frame indexes are invalid")
        invalid_indexes = (
            any(isinstance(value, bool) or not isinstance(value, int) for value in indexes)
            or indexes != sorted(set(indexes))
            or indexes[0] < 0
            or indexes[-1] >= len(self.metadata["frame_names"])
        )
        if invalid_indexes:
            raise ValueError("Metric geometry convergence frame indexes are invalid")
        sample_count = convergence.get("sample_count")
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
            raise ValueError("Metric geometry convergence sample count is invalid")
        distance = convergence.get("resolved_auto_distance_m")
        numeric_distance = not isinstance(distance, bool) and isinstance(distance, (int, float))
        if numeric_distance:
            with np.errstate(over="ignore", invalid="ignore"):
                distance_array = np.asarray(distance, dtype=np.float32)
            invalid_distance = not bool(metric_source_valid(distance_array).item())
        else:
            invalid_distance = True
        if invalid_distance:
            raise ValueError("Metric geometry resolved auto distance is invalid")

    def _persisted_required_bytes(self) -> int:
        required = self.metadata.get("preflight_required_bytes")
        if isinstance(required, bool) or not isinstance(required, int) or required < 0:
            raise ValueError("Metric geometry persisted preflight estimate is invalid")
        return required

    def _commit_metadata(self, updated: dict[str, Any]) -> None:
        _atomic_write_json(
            self.metadata_path,
            updated,
            required_bytes=self._persisted_required_bytes(),
        )
        self.metadata = updated

    def path_for(self, frame_name: str) -> Path:
        if frame_name not in self.metadata.get("frame_names", []):
            raise ValueError(f"Unknown metric geometry frame name: {frame_name}")
        return self.directory / f"{Path(frame_name).stem}.npz"

    def _validate_payload_header(self, path: Path) -> None:
        native_shape = tuple(self.metadata["native_shape"])
        try:
            with zipfile.ZipFile(path) as payload:
                if payload.namelist() != _METRIC_PAYLOAD_MEMBERS:
                    expected = ", ".join(_METRIC_PAYLOAD_MEMBERS)
                    raise ValueError(
                        f"Metric geometry payload must contain exact members {expected}: {path}"
                    )
                headers: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}
                for member_name in _METRIC_PAYLOAD_MEMBERS:
                    with payload.open(member_name) as member:
                        headers[member_name] = _read_npy_header(member)
        except FileNotFoundError as error:
            raise ValueError(f"Metric geometry payload is missing: {path}") from error
        except OSError:
            raise
        except ValueError:
            raise
        except Exception as error:
            raise ValueError(f"Metric geometry payload is unreadable: {path}") from error

        _validate_grid_header(
            label="inverse_depth",
            header=headers["inverse_depth.npy"],
            expected_dtype=np.dtype(np.float32),
            native_shape=native_shape,
            path=path,
        )
        _validate_grid_header(
            label="valid",
            header=headers["valid.npy"],
            expected_dtype=np.dtype(np.bool_),
            native_shape=native_shape,
            path=path,
        )
        _validate_focal_header(headers["focal_x_normalized.npy"], path)

    def _load_validated_payload(self, path: Path) -> MetricGeometryFrame:
        self._validate_payload_header(path)
        try:
            with np.load(path, allow_pickle=False) as payload:
                inverse = np.array(payload["inverse_depth"], copy=True)
                valid = np.array(payload["valid"], copy=True)
                focal = np.float32(payload["focal_x_normalized"].item())
        except FileNotFoundError as error:
            raise ValueError(f"Metric geometry payload is missing: {path}") from error
        except OSError:
            raise
        except Exception as error:
            raise ValueError(f"Metric geometry payload is unreadable: {path}") from error
        return MetricGeometryFrame(inverse, valid, focal)

    def validate_payloads(self, *, cleanup_temporaries: bool = True) -> int:
        """Validate exact payloads, optionally cleaning interrupted writes."""

        if cleanup_temporaries:
            for temporary in self.directory.glob("*.tmp"):
                temporary.unlink(missing_ok=True)
        frame_names = self.metadata["frame_names"]
        expected_names = {f"{Path(name).stem}.npz" for name in frame_names}
        actual_names = {path.name for path in self.directory.glob("*.npz")}
        unexpected = actual_names.difference(expected_names)
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise ValueError(f"Unexpected metric geometry payload files: {names}")
        completed = 0
        for frame_name in frame_names:
            path = self.path_for(frame_name)
            if not _is_regular_file(path):
                continue
            self._load_validated_payload(path)
            completed += 1
        if self.metadata.get("status") == "complete" and completed != len(frame_names):
            raise ValueError("Complete metric geometry store has missing payloads")
        return completed

    @property
    def complete_files(self) -> tuple[Path, ...]:
        self.validate_payloads()
        return tuple(
            path
            for frame_name in self.metadata["frame_names"]
            if _is_regular_file(path := self.path_for(frame_name))
        )

    def write_frame(self, frame_name: str, frame: MetricGeometryFrame) -> Path:
        if self.metadata.get("status") != "writing":
            raise ValueError("Cannot write to a complete metric geometry store")
        self._validate_writing_metadata()
        if not isinstance(frame, MetricGeometryFrame):
            raise TypeError("Metric geometry writes require a MetricGeometryFrame")
        path = self.path_for(frame_name)
        if tuple(frame.inverse_depth.shape) != tuple(self.metadata["native_shape"]):
            raise ValueError("Metric geometry frame shape does not match native shape")
        if _is_regular_file(path):
            self._load_validated_payload(path)
            return path
        _atomic_save_npz(
            path,
            frame,
            required_bytes=self._persisted_required_bytes(),
        )
        return path

    def load(self, path: Path) -> MetricGeometryFrame:
        path = Path(path)
        expected = {self.path_for(name) for name in self.metadata["frame_names"]}
        if path not in expected:
            raise ValueError(f"Unknown metric geometry payload path: {path}")
        return self._load_validated_payload(path)

    def finalize(self, convergence: ClipConvergence) -> dict[str, Any]:
        if not isinstance(convergence, ClipConvergence):
            raise TypeError("Metric geometry finalization requires ClipConvergence")
        self.validate_payloads()
        if self.metadata.get("status") == "complete":
            self._validate_complete_metadata()
            return self.metadata
        if self.metadata.get("status") != "writing":
            raise ValueError("Metric geometry transaction status is invalid")
        self._validate_writing_metadata()
        if self.validate_payloads() != len(self.metadata["frame_names"]):
            raise ValueError("Cannot finalize metric geometry with missing payloads")
        if convergence.selected_frame_indexes[-1] >= len(self.metadata["frame_names"]):
            raise ValueError("Clip convergence frame index is outside the frame manifest")
        completed = {
            key: value
            for key, value in self.metadata.items()
            if key not in {"preflight_required_bytes", "fingerprint", "convergence"}
        }
        completed["status"] = "complete"
        completed["convergence"] = {
            "algorithm_version": METRIC_CONVERGENCE_ALGORITHM_VERSION,
            "selected_frame_indexes": list(convergence.selected_frame_indexes),
            "sample_count": convergence.sample_count,
            "resolved_auto_distance_m": float(convergence.distance_m),
        }
        completed["fingerprint"] = canonical_json_hash(completed)
        self._commit_metadata(completed)
        return self.metadata
