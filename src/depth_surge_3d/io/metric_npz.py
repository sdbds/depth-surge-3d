"""Fast standard-zipfile metric geometry NPZ reader and atomic writer."""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import io
import os
from pathlib import Path
from typing import Callable, Literal
import zipfile

import numpy as np
from numpy.lib import format as npy_format

from .durable_publish import OwnedSiblingTemporary
from .nofollow_io import open_regular_beneath


METRIC_NPZ_WRITE_STREAM_BYTES = 1024 * 1024
_MEMBER_NAMES = (
    "inverse_depth.npy",
    "valid.npy",
    "focal_x_normalized.npy",
)

ArchiveForm = Literal[
    "owned-zip32-v1",
    "standard-zip-v1",
    "numpy-force-zip64-local-u32-v1",
    "numpy-force-zip64-local-sentinel-v1",
]


class MetricNpzError(ValueError):
    """Base error for the metric NPZ contract."""


class MetricNpzFormatError(MetricNpzError):
    """The ZIP or NPY structure is invalid."""


class MetricNpzContentError(MetricNpzError):
    """The arrays violate the metric semantic contract."""


class MetricNpzSizeError(MetricNpzError):
    """Array dimensions exceed supported bounds."""


class MetricNpzAllocationError(MemoryError):
    """Array allocation failed."""


class MetricNpzWriteError(OSError):
    """Writing a metric NPZ archive failed."""


class MetricNpzDiskFullError(MetricNpzWriteError):
    """Disk space was exhausted while writing a metric NPZ archive."""


@dataclass(frozen=True)
class MetricNpzLayout:
    """Minimal archive metadata."""

    archive_form: ArchiveForm
    archive_size: int


@dataclass
class MetricNpzAudit:
    """Loaded metric depth arrays and archive metadata."""

    layout: MetricNpzLayout
    inverse_depth: np.ndarray
    valid: np.ndarray
    focal_x_normalized: np.float32
    sha256: str = ""
    byte_count: int = 0


@dataclass
class MetricNpzWriteTrace:
    """Summary returned by ``write_metric_npz``."""

    archive_bytes: int = 0
    replaced: bool = False


def _is_disk_full(error: OSError) -> bool:
    return error.errno == errno.ENOSPC or getattr(error, "winerror", None) == 112


def _validate_metric_arrays(
    inverse: np.ndarray,
    valid: np.ndarray,
    focal_x_normalized: np.float32,
    *,
    expected_shape: tuple[int, int] | None = None,
) -> tuple[int, int]:
    if not isinstance(inverse, np.ndarray) or inverse.dtype != np.float32 or inverse.ndim != 2:
        raise MetricNpzContentError("inverse_depth dtype/shape must be 2D float32")
    if not isinstance(valid, np.ndarray) or valid.dtype != np.bool_ or valid.ndim != 2:
        raise MetricNpzContentError("valid dtype/shape must be 2D bool")
    if inverse.shape != valid.shape or min(inverse.shape) <= 0:
        raise MetricNpzContentError(
            "inverse_depth and valid dtype/shape must match and be positive"
        )
    if expected_shape is not None and tuple(inverse.shape) != tuple(expected_shape):
        raise MetricNpzContentError(
            f"inverse_depth dtype/shape {inverse.shape} does not match expected {expected_shape}"
        )
    if not np.isfinite(inverse).all():
        raise MetricNpzContentError("inverse_depth contains non-finite values")
    if np.any(inverse[~valid] != np.float32(0.0)):
        raise MetricNpzContentError("inverse_depth must be zero at invalid pixels")
    if np.any(inverse[valid] <= np.float32(0.0)):
        raise MetricNpzContentError("valid inverse_depth values must be positive")
    focal = np.float32(focal_x_normalized)
    if not np.isfinite(focal) or focal <= np.float32(0.0):
        raise MetricNpzContentError("focal_x_normalized must be finite and positive")
    return int(inverse.shape[0]), int(inverse.shape[1])


def metric_npz_temp_logical_bound(shape: tuple[int, int]) -> int:
    """Conservative upper bound in bytes for a single metric NPZ archive."""

    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("metric shape must be positive")
    return int(height * width * 5 * 1.15) + 16 * 1024


def write_metric_npz(
    path: Path | str,
    inverse: np.ndarray,
    valid: np.ndarray,
    focal_x_normalized: np.float32,
    *,
    reserve: Callable[[Path, int], None] | None = None,
    sync: bool = False,
) -> MetricNpzWriteTrace:
    """Write a compressed metric NPZ archive atomically using fast DEFLATE level 1."""

    target = Path(path)
    shape = _validate_metric_arrays(inverse, valid, focal_x_normalized)
    if reserve is not None:
        reserve(target.parent, metric_npz_temp_logical_bound(shape))
    target.parent.mkdir(parents=True, exist_ok=True)

    inv_c = np.ascontiguousarray(inverse, dtype=np.float32)
    val_c = np.ascontiguousarray(valid, dtype=np.bool_)
    focal_arr = np.asarray(np.float32(focal_x_normalized), dtype=np.float32)

    owned = OwnedSiblingTemporary.create(target)
    try:
        with zipfile.ZipFile(
            owned.handle,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=1,
            allowZip64=True,
        ) as archive:
            for member_name, array in (
                (_MEMBER_NAMES[0], inv_c),
                (_MEMBER_NAMES[1], val_c),
                (_MEMBER_NAMES[2], focal_arr),
            ):
                with archive.open(member_name, mode="w", force_zip64=False) as member:
                    npy_format.write_array(member, array, allow_pickle=False)
        owned.handle.flush()
        archive_bytes = owned.handle.tell()
        if sync:
            owned.sync()
        owned.replace()
        return MetricNpzWriteTrace(archive_bytes=archive_bytes, replaced=True)
    except OSError as error:
        if _is_disk_full(error):
            raise MetricNpzDiskFullError(
                error.errno or errno.ENOSPC, str(error), str(target)
            ) from error
        raise
    finally:
        owned.cleanup()


def _read_member_array(archive: zipfile.ZipFile, name: str) -> np.ndarray:
    try:
        with archive.open(name, mode="r") as member:
            array = npy_format.read_array(member, allow_pickle=False)
    except (KeyError, zipfile.BadZipFile, ValueError, EOFError) as error:
        raise MetricNpzFormatError(f"failed to read metric NPZ member {name}: {error}") from error
    return array if array.ndim == 0 else np.ascontiguousarray(array)


def _regular_open(target: Path, job_root: Path | None = None):
    if job_root is not None:
        return open_regular_beneath(job_root, target)
    return target.open("rb")


def audit_metric_npz(
    path: Path | str,
    *,
    expected_shape: tuple[int, int] | None = None,
    job_root: Path | None = None,
    compute_sha256: bool = False,
    **_ignored: object,
) -> MetricNpzAudit:
    """Load and validate a metric NPZ archive without pickle execution."""

    target = Path(path)
    with _regular_open(target, job_root=job_root) as handle:
        size = os.fstat(handle.fileno()).st_size
        digest_hex = ""
        if compute_sha256:
            digest = hashlib.sha256()
            while True:
                chunk = handle.read(METRIC_NPZ_WRITE_STREAM_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
            digest_hex = digest.hexdigest()
            handle.seek(0)
        try:
            with zipfile.ZipFile(handle, mode="r") as archive:
                names = set(archive.namelist())
                if names != set(_MEMBER_NAMES):
                    raise MetricNpzFormatError(f"unexpected metric NPZ members: {sorted(names)}")
                inverse = _read_member_array(archive, _MEMBER_NAMES[0])
                valid = _read_member_array(archive, _MEMBER_NAMES[1])
                focal_arr = _read_member_array(archive, _MEMBER_NAMES[2])
        except zipfile.BadZipFile as error:
            raise MetricNpzFormatError(f"corrupt metric NPZ archive: {target}") from error
        except MemoryError as error:
            raise MetricNpzAllocationError(str(error)) from error

    if focal_arr.dtype != np.float32 or focal_arr.shape != ():
        raise MetricNpzContentError("focal_x_normalized dtype/shape must be a 0D float32 scalar")
    focal = np.float32(focal_arr.item())
    _validate_metric_arrays(inverse, valid, focal, expected_shape=expected_shape)
    return MetricNpzAudit(
        layout=MetricNpzLayout(archive_form="standard-zip-v1", archive_size=size),
        inverse_depth=inverse,
        valid=valid,
        focal_x_normalized=focal,
        sha256=digest_hex,
        byte_count=size,
    )


def read_metric_npz(
    path: Path | str,
    *,
    expected_shape: tuple[int, int] | None = None,
    job_root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.float32]:
    """Convenience loader returning ``(inverse_depth, valid, focal_x_normalized)``."""

    audit = audit_metric_npz(path, expected_shape=expected_shape, job_root=job_root)
    return audit.inverse_depth, audit.valid, audit.focal_x_normalized


# Keep io import referenced cleanly if callers inspect BytesIO helpers
_ = io
