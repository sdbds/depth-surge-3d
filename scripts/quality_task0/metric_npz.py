"""Closed metric NPZ reader and bounded owned writer for the Task 0 gate."""

from __future__ import annotations

from dataclasses import dataclass, field
import binascii
import errno
import hashlib
import os
from pathlib import Path
import stat
import struct
from typing import Any, BinaryIO, Callable, Literal, Protocol, cast
import zlib

import numpy as np


METRIC_NPZ_WRITE_INPUT_CHUNK_BYTES = 512 * 1024
METRIC_NPZ_WRITE_STREAM_BYTES = 1024 * 1024
METRIC_NPZ_WRITE_DEFLATE_STATE_BYTES = 512 * 1024
METRIC_NPZ_WRITE_RECORD_BYTES = 4 * 1024
METRIC_NPZ_WRITE_ADDITIONAL_PEAK = (
    METRIC_NPZ_WRITE_STREAM_BYTES
    + METRIC_NPZ_WRITE_DEFLATE_STATE_BYTES
    + METRIC_NPZ_WRITE_RECORD_BYTES
)
METRIC_NPZ_VALUE_CHUNK_ELEMENTS = 64 * 1024
METRIC_NPZ_AUDIT_COMPRESSED_CHUNK_BYTES = 64 * 1024
_METRIC_NPZ_AUDIT_HEADER_ALLOWANCE = 512

_UINT32_ARCHIVE_MAX = 0xFFFFFFFE
_UINT64_MAX = 0xFFFFFFFFFFFFFFFF
_LOCAL_SIGNATURE = 0x04034B50
_CENTRAL_SIGNATURE = 0x02014B50
_EOCD_SIGNATURE = 0x06054B50
_ZIP64_EXTRA_ID = 0x0001
_DOS_TIME = 0
_DOS_DATE = 33
_MEMBER_NAMES = (
    "inverse_depth.npy",
    "valid.npy",
    "focal_x_normalized.npy",
)

ArchiveForm = Literal[
    "owned-zip32-v1",
    "numpy-force-zip64-local-u32-v1",
    "numpy-force-zip64-local-sentinel-v1",
]


class MetricNpzError(ValueError):
    """Base error for the closed metric payload contract."""


class MetricNpzFormatError(MetricNpzError):
    """The ZIP or NPY byte grammar is not one of the three admitted forms."""


class MetricNpzContentError(MetricNpzError):
    """The arrays violate the metric semantic contract."""


class MetricNpzSizeError(MetricNpzError):
    """A checked count cannot be represented by the owned ZIP32 form."""


class MetricNpzAllocationError(MemoryError):
    """The exact inverse/valid ownership allocation failed."""


class MetricNpzWriteError(OSError):
    """A bounded owned-writer operation failed."""


class MetricNpzDiskFullError(MetricNpzWriteError):
    """The owned writer observed an explicit disk-exhaustion failure."""


class _Allocator(Protocol):
    def __call__(self, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray: ...


@dataclass(frozen=True)
class MetricNpzMember:
    """One checked local/central record pair."""

    name: str
    local_offset: int
    data_offset: int
    compressed_size: int
    uncompressed_size: int
    crc32: int


@dataclass(frozen=True)
class MetricNpzLayout:
    """Checked scalar archive layout with no payload ownership."""

    archive_form: ArchiveForm
    members: tuple[MetricNpzMember, ...]
    central_offset: int
    central_size: int
    archive_size: int


@dataclass
class MetricNpzAudit:
    """Directly inflated metric values and their concrete archive identity."""

    layout: MetricNpzLayout
    inverse_depth: np.ndarray
    valid: np.ndarray
    focal_x_normalized: np.float32
    sha256: str
    byte_count: int
    maximum_stream_bytes: int
    stream_workspace_peak_bytes: int
    value_scratch_peak_bytes: int


@dataclass
class MetricNpzWriteTrace:
    """Measured bounded-writer facts used by Task 0 assertions."""

    events: list[str] = field(default_factory=list)
    maximum_input_chunk_bytes: int = 0
    maximum_output_block_bytes: int = 0
    complete_raw_member_copies: int = 0
    complete_compressed_member_copies: int = 0
    logical_bound_bytes: int = 0
    archive_bytes: int = 0
    replaced: bool = False


@dataclass(frozen=True)
class _NpyHeader:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    header_bytes: bytes
    data_bytes: int


@dataclass
class _InflateOwnership:
    inverse: np.ndarray | None = None
    valid: np.ndarray | None = None
    focal_bytes: bytearray = field(default_factory=lambda: bytearray(4))
    shape: tuple[int, int] | None = None
    maximum_stream_bytes: int = 0
    stream_workspace_peak_bytes: int = 0


@dataclass(frozen=True)
class _WriteRecord:
    name: str
    local_offset: int
    crc32: int
    compressed_size: int
    uncompressed_size: int


def _format_error(path: Path, detail: str) -> MetricNpzFormatError:
    return MetricNpzFormatError(f"metric NPZ {path}: {detail}")


def _checked_u64(value: int, label: str) -> int:
    if value < 0 or value > _UINT64_MAX:
        raise MetricNpzSizeError(f"{label} exceeds checked uint64")
    return value


def _checked_add(left: int, right: int, label: str) -> int:
    return _checked_u64(_checked_u64(left, label) + _checked_u64(right, label), label)


def _checked_product(left: int, right: int, label: str) -> int:
    return _checked_u64(_checked_u64(left, label) * _checked_u64(right, label), label)


def raw_deflate_bound(raw_bytes: int) -> int:
    """Return the frozen conservative raw-DEFLATE bound with checked arithmetic."""

    raw_bytes = _checked_u64(int(raw_bytes), "raw DEFLATE input")
    result = raw_bytes
    result = _checked_add(result, raw_bytes >> 12, "raw DEFLATE bound")
    result = _checked_add(result, raw_bytes >> 14, "raw DEFLATE bound")
    result = _checked_add(result, raw_bytes >> 25, "raw DEFLATE bound")
    return _checked_add(result, 13, "raw DEFLATE bound")


def _validate_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise TypeError("metric NPZ shape must be a two-item tuple")
    height, width = shape
    if (
        isinstance(height, bool)
        or isinstance(width, bool)
        or not isinstance(height, (int, np.integer))
        or not isinstance(width, (int, np.integer))
    ):
        raise TypeError("metric NPZ dimensions must be integers")
    height = int(height)
    width = int(width)
    if height <= 0 or width <= 0:
        raise ValueError("metric NPZ dimensions must be positive")
    _checked_product(height, width, "metric grid elements")
    return height, width


def _npy_header(name: str, shape: tuple[int, ...]) -> bytes:
    if name == "inverse_depth.npy":
        height, width = _validate_shape((int(shape[0]), int(shape[1])))
        body = (
            "{'descr': '<f4', 'fortran_order': False, " f"'shape': ({height}, {width}), }}"
        ).encode("ascii")
    elif name == "valid.npy":
        height, width = _validate_shape((int(shape[0]), int(shape[1])))
        body = (
            "{'descr': '|b1', 'fortran_order': False, " f"'shape': ({height}, {width}), }}"
        ).encode("ascii")
    elif name == "focal_x_normalized.npy":
        if shape != ():
            raise ValueError("metric focal NPY shape must be scalar")
        body = b"{'descr': '<f4', 'fortran_order': False, 'shape': (), }"
    else:
        raise ValueError(f"unknown metric member name: {name}")
    preamble_bytes = 10
    spaces = (-(preamble_bytes + len(body) + 1)) % 64
    if spaces == 0:
        spaces = 64
    header_body = body + (b" " * spaces) + b"\n"
    if len(header_body) > 256:
        raise MetricNpzSizeError("metric NPY v1 header exceeds 256 bytes")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header_body)) + header_body


def _member_raw_sizes(shape: tuple[int, int]) -> tuple[int, int, int]:
    height, width = _validate_shape(shape)
    grid = _checked_product(height, width, "metric grid elements")
    inverse = _checked_add(len(_npy_header(_MEMBER_NAMES[0], shape)), 4 * grid, "inverse member")
    valid = _checked_add(len(_npy_header(_MEMBER_NAMES[1], shape)), grid, "valid member")
    focal = _checked_add(len(_npy_header(_MEMBER_NAMES[2], ())), 4, "focal member")
    return inverse, valid, focal


def metric_npz_temp_logical_bound(shape: tuple[int, int]) -> int:
    """Return the checked frozen upper bound before any temporary is opened."""

    raw_sizes = _member_raw_sizes(_validate_shape(shape))
    total = 22
    for name, raw_size in zip(_MEMBER_NAMES, raw_sizes, strict=True):
        name_bytes = len(name.encode("ascii"))
        local = _checked_add(30 + name_bytes, raw_deflate_bound(raw_size), "archive bound")
        total = _checked_add(total, local, "archive bound")
        total = _checked_add(total, 46 + name_bytes, "archive bound")
    if total > _UINT32_ARCHIVE_MAX:
        raise MetricNpzSizeError("owned metric NPZ archive bound exceeds ZIP32")
    return total


def _read_exact(handle: BinaryIO, byte_count: int, path: Path, label: str) -> bytes:
    payload = handle.read(byte_count)
    if len(payload) != byte_count:
        raise _format_error(path, f"short {label}")
    return payload


def _regular_open(path: Path) -> BinaryIO:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise MetricNpzFormatError(f"metric NPZ {path}: path is not a regular file")
        attributes = getattr(metadata, "st_file_attributes", 0)
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if attributes & reparse:
            raise MetricNpzFormatError(f"metric NPZ {path}: reparse points are forbidden")
        return os.fdopen(descriptor, "rb", closefd=True)
    except BaseException:
        os.close(descriptor)
        raise


def _parse_local_extra(
    *,
    path: Path,
    version_needed: int,
    compressed_field: int,
    uncompressed_field: int,
    extra: bytes,
) -> tuple[ArchiveForm, int, int]:
    if version_needed == 20 and not extra:
        if compressed_field == 0xFFFFFFFF or uncompressed_field == 0xFFFFFFFF:
            raise _format_error(path, "owned size sentinel is forbidden")
        return "owned-zip32-v1", compressed_field, uncompressed_field
    if len(extra) != 20:
        raise _format_error(path, "local extra length is not the closed ZIP64 form")
    extra_id, payload_length = struct.unpack_from("<HH", extra)
    if extra_id != _ZIP64_EXTRA_ID or payload_length != 16:
        raise _format_error(path, "local extra field is not the admitted ZIP64 record")
    uncompressed_extra, compressed_extra = struct.unpack_from("<QQ", extra, 4)
    if uncompressed_extra > _UINT32_ARCHIVE_MAX or compressed_extra > _UINT32_ARCHIVE_MAX:
        raise _format_error(path, "historical ZIP64 resolved size exceeds ordinary range")
    if version_needed == 20:
        if compressed_field != compressed_extra or uncompressed_field != uncompressed_extra:
            raise _format_error(path, "historical u32 local size disagrees with ZIP64 extra")
        return (
            "numpy-force-zip64-local-u32-v1",
            int(compressed_extra),
            int(uncompressed_extra),
        )
    if version_needed == 45:
        if compressed_field != 0xFFFFFFFF or uncompressed_field != 0xFFFFFFFF:
            raise _format_error(path, "historical sentinel local size fields are required")
        return (
            "numpy-force-zip64-local-sentinel-v1",
            int(compressed_extra),
            int(uncompressed_extra),
        )
    raise _format_error(path, "version-needed is outside the closed archive grammar")


def parse_metric_npz_layout(handle: BinaryIO, path: Path) -> MetricNpzLayout:  # noqa: C901
    """Parse the six records plus EOCD without delegating to ``zipfile``."""

    path = Path(path)
    handle.seek(0, os.SEEK_END)
    archive_size = handle.tell()
    if archive_size > _UINT32_ARCHIVE_MAX:
        raise _format_error(path, "archive size exceeds the admitted ordinary range")
    handle.seek(0)
    members: list[MetricNpzMember] = []
    archive_form: ArchiveForm | None = None
    cursor = 0
    for expected_name in _MEMBER_NAMES:
        if handle.tell() != cursor:
            raise _format_error(path, "local offset is not contiguous")
        local = _read_exact(handle, 30, path, "local header")
        (
            signature,
            version_needed,
            flags,
            method,
            dos_time,
            dos_date,
            crc32,
            compressed_field,
            uncompressed_field,
            name_length,
            extra_length,
        ) = struct.unpack("<IHHHHHIIIHH", local)
        if signature != _LOCAL_SIGNATURE:
            raise _format_error(path, "local signature is invalid")
        if flags != 0:
            raise _format_error(path, "flag bits are forbidden")
        if method != 8:
            raise _format_error(path, "method must be raw DEFLATE")
        if dos_time != _DOS_TIME or dos_date != _DOS_DATE:
            raise _format_error(path, "DOS timestamp is noncanonical")
        name_bytes = _read_exact(handle, name_length, path, "local name")
        try:
            name = name_bytes.decode("ascii")
        except UnicodeDecodeError as error:
            raise _format_error(path, "local name is not ASCII") from error
        if name != expected_name:
            raise _format_error(path, "local name or member order is invalid")
        extra = _read_exact(handle, extra_length, path, "local extra")
        form, compressed_size, uncompressed_size = _parse_local_extra(
            path=path,
            version_needed=version_needed,
            compressed_field=compressed_field,
            uncompressed_field=uncompressed_field,
            extra=extra,
        )
        if archive_form is None:
            archive_form = form
        elif archive_form != form:
            raise _format_error(path, "crossed archive forms are forbidden")
        data_offset = handle.tell()
        data_end = data_offset + compressed_size
        if data_end > archive_size:
            raise _format_error(path, "compressed extent exceeds archive")
        members.append(
            MetricNpzMember(
                name=name,
                local_offset=cursor,
                data_offset=data_offset,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                crc32=crc32,
            )
        )
        handle.seek(data_end)
        cursor = data_end

    if archive_form is None:
        raise _format_error(path, "archive has no records")
    central_offset = cursor
    for index, (expected_name, member) in enumerate(zip(_MEMBER_NAMES, members, strict=True)):
        central = _read_exact(handle, 46, path, "central header")
        (
            signature,
            version_made_by,
            version_needed,
            flags,
            method,
            dos_time,
            dos_date,
            crc32,
            compressed_size,
            uncompressed_size,
            name_length,
            extra_length,
            comment_length,
            disk_start,
            internal_attributes,
            external_attributes,
            local_offset,
        ) = struct.unpack("<IHHHHHHIIIHHHHHII", central)
        if signature != _CENTRAL_SIGNATURE:
            raise _format_error(path, "central signature is invalid")
        platform = version_made_by >> 8
        made_version = version_made_by & 0xFF
        expected_version = 20 if archive_form != "numpy-force-zip64-local-sentinel-v1" else 45
        if platform not in {0, 3} or made_version != expected_version:
            raise _format_error(path, "central version-made-by is invalid")
        if version_needed != expected_version:
            raise _format_error(path, "central version-needed is invalid")
        if flags != 0:
            raise _format_error(path, "central flag bits are forbidden")
        if method != 8:
            raise _format_error(path, "central method must be raw DEFLATE")
        if dos_time != _DOS_TIME or dos_date != _DOS_DATE:
            raise _format_error(path, "central DOS timestamp is noncanonical")
        name_bytes = _read_exact(handle, name_length, path, "central name")
        try:
            name = name_bytes.decode("ascii")
        except UnicodeDecodeError as error:
            raise _format_error(path, "central name is not ASCII") from error
        if name != expected_name:
            raise _format_error(path, "central name or member order is invalid")
        if extra_length != 0 or comment_length != 0:
            raise _format_error(path, "central extra or comment is forbidden")
        if disk_start != 0 or internal_attributes != 0:
            raise _format_error(path, "central disk or internal attributes are invalid")
        if archive_form == "owned-zip32-v1":
            if external_attributes not in {0, 0x01800000}:
                raise _format_error(path, "owned external attributes are invalid")
        elif external_attributes != 0x01800000:
            raise _format_error(path, "historical external attributes are invalid")
        if compressed_size in {0xFFFFFFFF} or uncompressed_size in {0xFFFFFFFF}:
            raise _format_error(path, "central size sentinel is forbidden")
        if (
            crc32 != member.crc32
            or compressed_size != member.compressed_size
            or uncompressed_size != member.uncompressed_size
        ):
            raise _format_error(path, f"central/local record disagreement at member {index}")
        if local_offset != member.local_offset:
            raise _format_error(path, "central local offset disagrees with checked offset")
    central_end = handle.tell()
    central_size = central_end - central_offset
    eocd = _read_exact(handle, 22, path, "EOCD")
    (
        signature,
        disk_number,
        central_disk,
        entries_on_disk,
        entries_total,
        declared_central_size,
        declared_central_offset,
        comment_length,
    ) = struct.unpack("<IHHHHIIH", eocd)
    if signature != _EOCD_SIGNATURE:
        raise _format_error(path, "EOCD signature is invalid")
    if disk_number != 0 or central_disk != 0:
        raise _format_error(path, "EOCD disk fields are invalid")
    if entries_on_disk != 3 or entries_total != 3:
        raise _format_error(path, "EOCD entry count is invalid")
    if declared_central_size != central_size or declared_central_offset != central_offset:
        raise _format_error(path, "EOCD central offset or size is invalid")
    if comment_length != 0:
        raise _format_error(path, "EOCD comment is forbidden")
    if handle.tell() != archive_size:
        raise _format_error(path, "trailing archive data is forbidden")
    return MetricNpzLayout(
        archive_form=archive_form,
        members=tuple(members),
        central_offset=central_offset,
        central_size=central_size,
        archive_size=archive_size,
    )


def _parse_npy_header(path: Path, name: str, header: bytes) -> _NpyHeader:  # noqa: C901
    if len(header) < 10 or header[:8] != b"\x93NUMPY\x01\x00":
        raise _format_error(path, f"{name} NPY descriptor/version is invalid")
    header_length = struct.unpack_from("<H", header, 8)[0]
    if header_length > 256 or len(header) != 10 + header_length:
        raise _format_error(path, f"{name} NPY header length is invalid")
    if len(header) % 64 != 0:
        raise _format_error(path, f"{name} NPY header alignment is invalid")
    body = header[10:]
    if not body.endswith(b"\n") or body[:-1].endswith(b"\n"):
        raise _format_error(path, f"{name} NPY header newline is invalid")
    try:
        text = body.decode("ascii")
    except UnicodeDecodeError as error:
        raise _format_error(path, f"{name} NPY header is not ASCII") from error
    unpadded = text[:-1].rstrip(" ")
    padding = text[len(unpadded) : -1]
    if not padding:
        raise _format_error(path, f"{name} NPY header requires positive padding")
    if name == "focal_x_normalized.npy":
        expected_body = "{'descr': '<f4', 'fortran_order': False, 'shape': (), }"
        if unpadded != expected_body or header != _npy_header(name, ()):
            raise _format_error(path, f"{name} NPY dtype/shape grammar is invalid")
        return _NpyHeader(name, (), np.dtype("<f4"), header, 4)

    prefix = (
        "{'descr': '<f4', 'fortran_order': False, 'shape': ("
        if name == "inverse_depth.npy"
        else "{'descr': '|b1', 'fortran_order': False, 'shape': ("
    )
    suffix = "), }"
    if not unpadded.startswith(prefix) or not unpadded.endswith(suffix):
        raise _format_error(path, f"{name} NPY dtype/shape grammar is invalid")
    dimensions = unpadded[len(prefix) : -len(suffix)].split(", ")
    if len(dimensions) != 2 or any(not value.isdecimal() for value in dimensions):
        raise _format_error(path, f"{name} NPY shape tokens are invalid")
    if any(value.startswith("0") for value in dimensions):
        raise _format_error(path, f"{name} NPY shape is noncanonical")
    shape = _validate_shape((int(dimensions[0]), int(dimensions[1])))
    if header != _npy_header(name, shape):
        raise _format_error(path, f"{name} NPY padding or grammar is noncanonical")
    grid = _checked_product(shape[0], shape[1], "metric NPY grid")
    dtype = np.dtype("<f4") if name == "inverse_depth.npy" else np.dtype("|b1")
    data_bytes = 4 * grid if name == "inverse_depth.npy" else grid
    return _NpyHeader(name, shape, dtype, header, data_bytes)


def _allocate_arrays(
    path: Path,
    shape: tuple[int, int],
    allocator: _Allocator,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        inverse = allocator(shape, np.dtype(np.float32))
        valid = allocator(shape, np.dtype(np.bool_))
    except MemoryError as error:
        raise MetricNpzAllocationError(f"metric NPZ {path}: allocation failed") from error
    if (
        not isinstance(inverse, np.ndarray)
        or inverse.shape != shape
        or inverse.dtype != np.float32
        or not inverse.flags.c_contiguous
        or not inverse.flags.writeable
    ):
        raise MetricNpzAllocationError(f"metric NPZ {path}: inverse allocator contract failed")
    if (
        not isinstance(valid, np.ndarray)
        or valid.shape != shape
        or valid.dtype != np.bool_
        or not valid.flags.c_contiguous
        or not valid.flags.writeable
    ):
        raise MetricNpzAllocationError(f"metric NPZ {path}: valid allocator contract failed")
    return inverse, valid


def _inflate_member(  # noqa: C901
    handle: BinaryIO,
    path: Path,
    member: MetricNpzMember,
    ownership: _InflateOwnership,
    expected_shape: tuple[int, int] | None,
    allocator: _Allocator,
) -> None:
    handle.seek(member.data_offset)
    decoder = zlib.decompressobj(-zlib.MAX_WBITS)
    compressed_remaining = member.compressed_size
    raw_count = 0
    crc32 = 0
    header = bytearray()
    header_total: int | None = None
    parsed: _NpyHeader | None = None
    destination: memoryview | None = None
    destination_offset = 0

    def consume(output: bytes, compressed_live_bytes: int = 0) -> None:
        nonlocal raw_count, crc32, header_total, parsed, destination, destination_offset
        if len(output) > METRIC_NPZ_WRITE_STREAM_BYTES:
            raise _format_error(path, f"{member.name} inflated block exceeds stream cap")
        ownership.maximum_stream_bytes = max(ownership.maximum_stream_bytes, len(output))
        ownership.stream_workspace_peak_bytes = max(
            ownership.stream_workspace_peak_bytes,
            len(output) + compressed_live_bytes + len(header),
        )
        if ownership.stream_workspace_peak_bytes > METRIC_NPZ_WRITE_STREAM_BYTES:
            raise _format_error(path, f"{member.name} stream workspace exceeds fixed cap")
        raw_count += len(output)
        if raw_count > member.uncompressed_size:
            raise _format_error(path, f"{member.name} overlong inflated output")
        crc32 = binascii.crc32(output, crc32) & 0xFFFFFFFF
        output_view = memoryview(output)
        cursor = 0
        while cursor < len(output_view):
            if parsed is None:
                if header_total is None:
                    needed = 10 - len(header)
                else:
                    needed = header_total - len(header)
                take = min(needed, len(output_view) - cursor)
                header.extend(output_view[cursor : cursor + take])
                cursor += take
                if header_total is None and len(header) == 10:
                    if header[:8] != b"\x93NUMPY\x01\x00":
                        raise _format_error(
                            path, f"{member.name} NPY descriptor/version is invalid"
                        )
                    header_length = struct.unpack_from("<H", header, 8)[0]
                    if header_length > 256:
                        raise _format_error(path, f"{member.name} NPY header length is invalid")
                    header_total = 10 + header_length
                if header_total is not None and len(header) == header_total:
                    parsed = _parse_npy_header(path, member.name, bytes(header))
                    if member.uncompressed_size != len(parsed.header_bytes) + parsed.data_bytes:
                        raise _format_error(
                            path, f"{member.name} declared uncompressed size is invalid"
                        )
                    if member.name == "inverse_depth.npy":
                        shape = (int(parsed.shape[0]), int(parsed.shape[1]))
                        if expected_shape is not None and shape != expected_shape:
                            raise _format_error(path, "inverse_depth.npy NPY shape mismatch")
                        ownership.inverse, ownership.valid = _allocate_arrays(
                            path, shape, allocator
                        )
                        ownership.shape = shape
                        destination = memoryview(cast(Any, ownership.inverse)).cast("B")
                    elif member.name == "valid.npy":
                        if ownership.shape is None or parsed.shape != ownership.shape:
                            raise _format_error(path, "valid.npy NPY shape mismatch")
                        if ownership.valid is None:
                            raise _format_error(path, "valid.npy destination is unavailable")
                        destination = memoryview(cast(Any, ownership.valid)).cast("B")
                    else:
                        destination = memoryview(ownership.focal_bytes)
                continue
            if destination is None:
                raise _format_error(path, f"{member.name} destination is unavailable")
            remaining = len(destination) - destination_offset
            take = min(remaining, len(output_view) - cursor)
            if take == 0:
                raise _format_error(path, f"{member.name} overlong member data")
            chunk = output_view[cursor : cursor + take]
            if member.name == "valid.npy" and np.any(np.frombuffer(chunk, dtype=np.uint8) > 1):
                raise MetricNpzContentError(f"metric NPZ {path}: valid bytes must be 0 or 1")
            destination[destination_offset : destination_offset + take] = chunk
            destination_offset += take
            cursor += take

    try:
        pending = b""
        while compressed_remaining > 0:
            if not pending:
                read_size = min(METRIC_NPZ_AUDIT_COMPRESSED_CHUNK_BYTES, compressed_remaining)
                pending = _read_exact(handle, read_size, path, f"{member.name} compressed extent")
                compressed_remaining -= read_size
            output_limit = max(
                1,
                METRIC_NPZ_WRITE_STREAM_BYTES - len(pending) - _METRIC_NPZ_AUDIT_HEADER_ALLOWANCE,
            )
            previous_pending = len(pending)
            output = decoder.decompress(pending, output_limit)
            pending = decoder.unconsumed_tail
            consume(output, len(pending))
            if decoder.unused_data:
                raise _format_error(path, f"{member.name} has trailing compressed bytes")
            if decoder.eof and (pending or compressed_remaining):
                raise _format_error(path, f"{member.name} reaches DEFLATE end before its extent")
            if not output and pending and len(pending) == previous_pending:
                raise _format_error(path, f"{member.name} DEFLATE decoder made no progress")
        if pending:
            while pending:
                before = len(pending)
                output_limit = max(
                    1,
                    METRIC_NPZ_WRITE_STREAM_BYTES
                    - len(pending)
                    - _METRIC_NPZ_AUDIT_HEADER_ALLOWANCE,
                )
                output = decoder.decompress(pending, output_limit)
                pending = decoder.unconsumed_tail
                consume(output, len(pending))
                if decoder.unused_data:
                    raise _format_error(path, f"{member.name} has trailing compressed bytes")
                if not output and len(pending) == before:
                    raise _format_error(path, f"{member.name} DEFLATE decoder made no progress")
        consume(decoder.flush())
    except zlib.error as error:
        raise _format_error(path, f"{member.name} DEFLATE stream is invalid") from error
    if not decoder.eof:
        raise _format_error(path, f"{member.name} DEFLATE stream is incomplete")
    if decoder.unused_data or decoder.unconsumed_tail:
        raise _format_error(path, f"{member.name} has trailing compressed bytes")
    if parsed is None or destination is None:
        raise _format_error(path, f"{member.name} NPY header is incomplete")
    if destination_offset != len(destination):
        raise _format_error(path, f"{member.name} member data is short")
    if raw_count != member.uncompressed_size:
        raise _format_error(path, f"{member.name} inflated size is short")
    if crc32 != member.crc32:
        raise _format_error(path, f"{member.name} crc mismatch")


def _validate_metric_values(path: Path, ownership: _InflateOwnership) -> int:
    inverse = ownership.inverse
    valid = ownership.valid
    if inverse is None or valid is None:
        raise _format_error(path, "metric arrays were not inflated")
    flat_inverse = inverse.reshape(-1)
    flat_valid = valid.reshape(-1)
    bits = flat_inverse.view(np.uint32)
    scratch_peak = 0
    for start in range(0, flat_inverse.size, METRIC_NPZ_VALUE_CHUNK_ELEMENTS):
        stop = min(start + METRIC_NPZ_VALUE_CHUNK_ELEMENTS, flat_inverse.size)
        inverse_chunk = flat_inverse[start:stop]
        valid_chunk = flat_valid[start:stop]
        bits_chunk = bits[start:stop]
        scratch_peak = max(scratch_peak, (stop - start) * 3)
        if not bool(np.isfinite(inverse_chunk).all()):
            raise MetricNpzContentError(f"metric NPZ {path}: inverse values must be finite")
        if bool(np.logical_and(np.logical_not(valid_chunk), bits_chunk != 0).any()):
            raise MetricNpzContentError(
                f"metric NPZ {path}: invalid inverse values must be exact positive zero"
            )
        if bool(np.logical_and(valid_chunk, inverse_chunk <= np.float32(0.0)).any()):
            raise MetricNpzContentError(f"metric NPZ {path}: valid inverse values must be positive")
    if len(ownership.focal_bytes) != 4:
        raise _format_error(path, "focal member data is short")
    focal = np.float32(struct.unpack("<f", ownership.focal_bytes)[0])
    if not bool(np.isfinite(focal)) or focal <= np.float32(0.0):
        raise MetricNpzContentError(f"metric NPZ {path}: focal value must be finite and positive")
    return scratch_peak


def audit_metric_npz(
    path: Path,
    *,
    expected_shape: tuple[int, int] | None = None,
    allocator: _Allocator = np.empty,
) -> MetricNpzAudit:
    """Hash, parse, directly inflate, and semantically validate one payload."""

    path = Path(path)
    if expected_shape is not None:
        expected_shape = _validate_shape(expected_shape)
    digest = hashlib.sha256()
    byte_count = 0
    maximum_stream_bytes = 0
    with _regular_open(path) as handle:
        while True:
            chunk = handle.read(METRIC_NPZ_WRITE_STREAM_BYTES)
            if not chunk:
                break
            maximum_stream_bytes = max(maximum_stream_bytes, len(chunk))
            digest.update(chunk)
            byte_count += len(chunk)
        handle.seek(0)
        layout = parse_metric_npz_layout(handle, path)
        ownership = _InflateOwnership(maximum_stream_bytes=maximum_stream_bytes)
        ownership.stream_workspace_peak_bytes = maximum_stream_bytes
        for member in layout.members:
            _inflate_member(handle, path, member, ownership, expected_shape, allocator)
    scratch_peak = _validate_metric_values(path, ownership)
    ownership.stream_workspace_peak_bytes = max(
        ownership.stream_workspace_peak_bytes,
        scratch_peak,
    )
    if ownership.inverse is None or ownership.valid is None:
        raise _format_error(path, "metric arrays are unavailable")
    focal = np.float32(struct.unpack("<f", ownership.focal_bytes)[0])
    ownership.inverse.setflags(write=False)
    ownership.valid.setflags(write=False)
    return MetricNpzAudit(
        layout=layout,
        inverse_depth=ownership.inverse,
        valid=ownership.valid,
        focal_x_normalized=focal,
        sha256=digest.hexdigest(),
        byte_count=byte_count,
        maximum_stream_bytes=ownership.maximum_stream_bytes,
        stream_workspace_peak_bytes=ownership.stream_workspace_peak_bytes,
        value_scratch_peak_bytes=scratch_peak,
    )


def _write_checked(handle: BinaryIO, payload: bytes | memoryview) -> None:
    written = handle.write(payload)
    if written != len(payload):
        raise MetricNpzWriteError("metric NPZ temporary write was short")


def _call_fault(fault: Callable[[str], None] | None, stage: str) -> None:
    if fault is not None:
        fault(stage)


def _validate_writer_inputs(
    inverse: np.ndarray,
    valid: np.ndarray,
    focal: np.float32,
) -> tuple[int, int]:
    if not isinstance(inverse, np.ndarray) or inverse.dtype != np.float32 or inverse.ndim != 2:
        raise TypeError("metric NPZ inverse input must be a 2D float32 array")
    if not inverse.flags.c_contiguous:
        raise ValueError("metric NPZ inverse input must be C-contiguous")
    if not isinstance(valid, np.ndarray) or valid.dtype != np.bool_ or valid.ndim != 2:
        raise TypeError("metric NPZ valid input must be a 2D bool array")
    if not valid.flags.c_contiguous or valid.shape != inverse.shape:
        raise ValueError("metric NPZ valid input must be shape-matched and C-contiguous")
    if not isinstance(focal, np.float32):
        raise TypeError("metric NPZ focal input must be float32")
    return _validate_shape((int(inverse.shape[0]), int(inverse.shape[1])))


def _emit_compressed_input(
    *,
    handle: BinaryIO,
    compressor: Any,
    payload: bytes | memoryview,
    trace: MetricNpzWriteTrace,
    crc32: int,
    raw_count: int,
    compressed_count: int,
) -> tuple[int, int, int]:
    payload_length = len(payload)
    if payload_length > METRIC_NPZ_WRITE_INPUT_CHUNK_BYTES:
        raise MetricNpzWriteError("metric NPZ compressor input exceeds fixed chunk cap")
    trace.maximum_input_chunk_bytes = max(trace.maximum_input_chunk_bytes, payload_length)
    crc32 = binascii.crc32(payload, crc32) & 0xFFFFFFFF
    raw_count = _checked_add(raw_count, payload_length, "metric member raw count")
    output = compressor.compress(payload)
    if len(output) > METRIC_NPZ_WRITE_STREAM_BYTES:
        raise MetricNpzWriteError("metric NPZ compressor output exceeds stream cap")
    trace.maximum_output_block_bytes = max(trace.maximum_output_block_bytes, len(output))
    if output:
        _write_checked(handle, output)
        compressed_count = _checked_add(
            compressed_count,
            len(output),
            "metric member compressed count",
        )
    return crc32, raw_count, compressed_count


def _write_member(
    *,
    handle: BinaryIO,
    name: str,
    header: bytes,
    data: memoryview,
    trace: MetricNpzWriteTrace,
    fault: Callable[[str], None] | None,
) -> _WriteRecord:
    local_offset = handle.tell()
    if local_offset > _UINT32_ARCHIVE_MAX:
        raise MetricNpzSizeError("metric NPZ local offset exceeds ZIP32")
    name_bytes = name.encode("ascii")
    local_header = struct.pack(
        "<IHHHHHIIIHH",
        _LOCAL_SIGNATURE,
        20,
        0,
        8,
        _DOS_TIME,
        _DOS_DATE,
        0,
        0,
        0,
        len(name_bytes),
        0,
    )
    _write_checked(handle, local_header)
    _write_checked(handle, name_bytes)
    _call_fault(fault, "compression")
    compressor = zlib.compressobj(level=9, wbits=-zlib.MAX_WBITS)
    crc32 = 0
    raw_count = 0
    compressed_count = 0
    crc32, raw_count, compressed_count = _emit_compressed_input(
        handle=handle,
        compressor=compressor,
        payload=header,
        trace=trace,
        crc32=crc32,
        raw_count=raw_count,
        compressed_count=compressed_count,
    )
    for offset in range(0, len(data), METRIC_NPZ_WRITE_INPUT_CHUNK_BYTES):
        chunk = data[offset : offset + METRIC_NPZ_WRITE_INPUT_CHUNK_BYTES]
        crc32, raw_count, compressed_count = _emit_compressed_input(
            handle=handle,
            compressor=compressor,
            payload=chunk,
            trace=trace,
            crc32=crc32,
            raw_count=raw_count,
            compressed_count=compressed_count,
        )
    output = compressor.flush(zlib.Z_FINISH)
    if len(output) > METRIC_NPZ_WRITE_STREAM_BYTES:
        raise MetricNpzWriteError("metric NPZ final compressor output exceeds stream cap")
    trace.maximum_output_block_bytes = max(trace.maximum_output_block_bytes, len(output))
    if output:
        _write_checked(handle, output)
        compressed_count = _checked_add(
            compressed_count,
            len(output),
            "metric member compressed count",
        )
    if raw_count > _UINT32_ARCHIVE_MAX or compressed_count > _UINT32_ARCHIVE_MAX:
        raise MetricNpzSizeError("metric NPZ member size exceeds ZIP32")
    if compressed_count > raw_deflate_bound(raw_count):
        raise MetricNpzWriteError("metric NPZ compressor exceeded the frozen DEFLATE bound")
    end_offset = handle.tell()
    _call_fault(fault, "seek")
    handle.seek(local_offset + 14)
    _call_fault(fault, "patch")
    _write_checked(handle, struct.pack("<III", crc32, compressed_count, raw_count))
    _call_fault(fault, "seek")
    handle.seek(end_offset)
    trace.events.append(f"member:{name}")
    return _WriteRecord(name, local_offset, crc32, compressed_count, raw_count)


def _write_directory(handle: BinaryIO, records: list[_WriteRecord]) -> None:
    central_offset = handle.tell()
    if central_offset > _UINT32_ARCHIVE_MAX:
        raise MetricNpzSizeError("metric NPZ central offset exceeds ZIP32")
    for record in records:
        name_bytes = record.name.encode("ascii")
        central = struct.pack(
            "<IHHHHHHIIIHHHHHII",
            _CENTRAL_SIGNATURE,
            20,
            20,
            0,
            8,
            _DOS_TIME,
            _DOS_DATE,
            record.crc32,
            record.compressed_size,
            record.uncompressed_size,
            len(name_bytes),
            0,
            0,
            0,
            0,
            0,
            record.local_offset,
        )
        _write_checked(handle, central)
        _write_checked(handle, name_bytes)
    central_size = handle.tell() - central_offset
    if central_size > _UINT32_ARCHIVE_MAX:
        raise MetricNpzSizeError("metric NPZ central size exceeds ZIP32")
    eocd = struct.pack(
        "<IHHHHIIH",
        _EOCD_SIGNATURE,
        0,
        0,
        3,
        3,
        central_size,
        central_offset,
        0,
    )
    _write_checked(handle, eocd)
    if handle.tell() > _UINT32_ARCHIVE_MAX:
        raise MetricNpzSizeError("metric NPZ archive length exceeds ZIP32")


def _sync_parent(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_metric_npz(
    path: Path,
    inverse: np.ndarray,
    valid: np.ndarray,
    focal_x_normalized: np.float32,
    *,
    fault: Callable[[str], None] | None = None,
    reserve: Callable[[Path, int], None] | None = None,
) -> MetricNpzWriteTrace:
    """Write the exact owned ZIP32 form with one bounded sibling temporary."""

    path = Path(path)
    shape = _validate_writer_inputs(inverse, valid, focal_x_normalized)
    logical_bound = metric_npz_temp_logical_bound(shape)
    if reserve is not None:
        reserve(path.parent, logical_bound)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.metric-npz.tmp")
    trace = MetricNpzWriteTrace(logical_bound_bytes=logical_bound)
    records: list[_WriteRecord] = []
    replaced = False
    try:
        _call_fault(fault, "disk-full")
        with temporary.open("w+b") as handle:
            inverse_header = _npy_header(_MEMBER_NAMES[0], shape)
            valid_header = _npy_header(_MEMBER_NAMES[1], shape)
            focal_header = _npy_header(_MEMBER_NAMES[2], ())
            records.append(
                _write_member(
                    handle=handle,
                    name=_MEMBER_NAMES[0],
                    header=inverse_header,
                    data=memoryview(cast(Any, inverse)).cast("B"),
                    trace=trace,
                    fault=fault,
                )
            )
            records.append(
                _write_member(
                    handle=handle,
                    name=_MEMBER_NAMES[1],
                    header=valid_header,
                    data=memoryview(cast(Any, valid)).cast("B"),
                    trace=trace,
                    fault=fault,
                )
            )
            focal_bytes = struct.pack("<f", float(focal_x_normalized))
            records.append(
                _write_member(
                    handle=handle,
                    name=_MEMBER_NAMES[2],
                    header=focal_header,
                    data=memoryview(focal_bytes),
                    trace=trace,
                    fault=fault,
                )
            )
            _write_directory(handle, records)
            trace.archive_bytes = handle.tell()
            if trace.archive_bytes > trace.logical_bound_bytes:
                raise MetricNpzWriteError("metric NPZ archive exceeded its preflight bound")
            _call_fault(fault, "sync")
            handle.flush()
            os.fsync(handle.fileno())
        _call_fault(fault, "replace")
        os.replace(temporary, path)
        replaced = True
        trace.replaced = True
        _call_fault(fault, "parent-sync")
        _sync_parent(path.parent)
        return trace
    except OSError as error:
        if error.errno == errno.ENOSPC or getattr(error, "winerror", None) == 112:
            raise MetricNpzDiskFullError(
                error.errno or errno.ENOSPC, str(error), str(path)
            ) from error
        raise
    finally:
        if not replaced:
            temporary.unlink(missing_ok=True)
