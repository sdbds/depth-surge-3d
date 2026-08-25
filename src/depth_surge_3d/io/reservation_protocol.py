"""Indexed reservation and irreversible-publication production protocol."""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import time
from contextlib import contextmanager, nullcontext
from io import BufferedIOBase
from typing import TYPE_CHECKING, Callable, Iterable, Literal

from .canonical_json import CanonicalJsonError, strict_canonical_json_loads
from .file_identity import (
    DirectoryIdentityV1 as DirectoryIdentity,
    PersistedFileIdentityV1 as ReservationFileIdentity,
    PersistentIdentityCapabilityError,
    PersistentIdentitySchemaError,
    capture_directory_identity,
    capture_file_identity,
    directory_identity_from_posix_fd,
    file_identity_from_posix_fd,
    windows_identity_from_handle,
)
from .settings_artifact import SETTINGS_ARTIFACT_MAX_RAW_BYTES
from .nofollow_io import (
    AuthenticatedFixedFileV1,
    NoFollowPathError,
    posix_remove_verified_name,
    scan_fixed_regular_v1,
)

if TYPE_CHECKING:
    from .settings_artifact import ProcessingSettingsArtifactV5


RESERVATION_ZERO_FILL_CHUNK_BYTES = 64 * 1024
CONTROL_ARTIFACT_WRITE_CHUNK_BYTES = 4 * 1024
FINAL_RESERVATION_DESCRIPTOR_RAW_BYTES = 8 * 1024
JOB_CONTROL_RELATIVE_PATH = Path("job-control-v1.json")
SETTINGS_ARTIFACT_RELATIVE_PATH = Path("processing-settings-v5.json")
SETTINGS_TRANSITION_INDEX_RELATIVE_PATH = Path(
    ".depth-surge-reservations-v1/settings-transition-reservation-v1.json"
)

PublicationMethod = Literal["replace_placeholder", "replace_existing"]


class ReservationError(RuntimeError):
    """Base reservation protocol error."""


class ReservationIncompleteError(ReservationError):
    """An indexed materialization or release proof is retryably incomplete."""


class ReservationCapabilityError(ReservationError):
    """The platform cannot provide the frozen allocation/rename evidence."""


class ReservationConflictError(ReservationError):
    """Observed bytes, identity, or namespace contradict durable authority."""


class ReservationEvidenceConflictError(ReservationConflictError):
    """Full-source allocation evidence changed after descriptor publication."""


class WindowsRenameProtocolError(ReservationConflictError):
    """The Windows Boolean result contradicts its observed physical state."""


class InjectedReservationCrash(RuntimeError):
    """A deterministic crash fixture stopped at a durable protocol boundary."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_canonical(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _scan_reservation_fixed_file_v1(
    job_root: Path,
    path: Path,
    *,
    expected_identity: ReservationFileIdentity | None = None,
    expected_raw_sha256: str | None = None,
    expected_byte_count: int | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> AuthenticatedFixedFileV1:
    try:
        return scan_fixed_regular_v1(
            job_root,
            path,
            expected_identity=expected_identity,
            expected_raw_sha256=expected_raw_sha256,
            expected_byte_count=expected_byte_count,
            cancelled=cancelled,
        )
    except InterruptedError:
        raise
    except (NoFollowPathError, FileNotFoundError, OSError) as error:
        raise ReservationConflictError(
            "reservation fixed target failed authenticated scan"
        ) from error


def _stream_file_sha256_count(
    job_root: Path,
    path: Path,
    *,
    expected_identity: ReservationFileIdentity | None = None,
    expected_raw_sha256: str | None = None,
    expected_byte_count: int | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[str, int]:
    scanned = _scan_reservation_fixed_file_v1(
        job_root,
        path,
        expected_identity=expected_identity,
        expected_raw_sha256=expected_raw_sha256,
        expected_byte_count=expected_byte_count,
        cancelled=cancelled,
    )
    return scanned.raw_sha256, scanned.byte_count


def _allocate(raw: int, allocation_unit: int) -> int:
    if raw < 0 or allocation_unit <= 0:
        raise ValueError("allocation inputs must be nonnegative with a positive unit")
    return ((raw + allocation_unit - 1) // allocation_unit) * allocation_unit


def payload_logical_bytes(raw: int, allocation_unit: int) -> int:
    return _allocate(raw, allocation_unit) + allocation_unit


def payload_extent_create_charge(raw: int, allocation_unit: int) -> int:
    return payload_logical_bytes(raw, allocation_unit) + allocation_unit


def descriptor_create_new_charge(descriptor_raw: int, allocation_unit: int) -> int:
    return _allocate(descriptor_raw, allocation_unit) + 2 * allocation_unit


def placeholder_entry_charge(method: PublicationMethod, allocation_unit: int) -> int:
    if method == "replace_placeholder":
        return allocation_unit
    if method == "replace_existing":
        return 0
    raise ValueError("unknown reservation publication method")


def reservation_extent_charge(
    payload_raw: int,
    descriptor_raw: int,
    method: PublicationMethod,
    allocation_unit: int,
) -> int:
    return (
        payload_extent_create_charge(payload_raw, allocation_unit)
        + descriptor_create_new_charge(descriptor_raw, allocation_unit)
        + placeholder_entry_charge(method, allocation_unit)
    )


def reservation_directory_bootstrap_forecast(allocation_unit: int) -> int:
    if allocation_unit <= 0:
        raise ValueError("allocation unit must be positive")
    return 2 * allocation_unit


@dataclass(frozen=True)
class AdmissionForecast:
    byte_count: int
    is_strict_directory_upper_bound: bool
    namespace_enospc_error: type[ReservationIncompleteError]


def minimum_admission_forecast(
    *,
    payload_raw: int,
    descriptor_raw: int,
    method: PublicationMethod,
    allocation_unit: int,
) -> AdmissionForecast:
    return AdmissionForecast(
        reservation_extent_charge(payload_raw, descriptor_raw, method, allocation_unit),
        False,
        ReservationIncompleteError,
    )


def classify_payload_frame(raw: bytes, expected_payload: bytes) -> str:
    if not expected_payload or b"\x00" in expected_payload:
        raise ValueError("canonical reservation payload must be nonempty and NUL-free")
    if raw == expected_payload:
        return "complete-short"
    if raw and all(byte == 0 for byte in raw):
        return "zero"
    if len(raw) > len(expected_payload) and raw.startswith(expected_payload):
        tail = raw[len(expected_payload) :]
        if tail and all(byte == 0 for byte in tail):
            return "complete-full"
    return "partial"


def _sync_parent(
    path: Path,
    *,
    expected_identity: DirectoryIdentity | None = None,
) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        if (
            expected_identity is not None
            and directory_identity_from_posix_fd(descriptor) != expected_identity
        ):
            raise ReservationConflictError("control parent identity changed before sync")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _open_relative_regular_file(
    path: Path,
    *,
    writable: bool,
    create: bool,
    exclusive: bool = False,
    expected_parent_identity: DirectoryIdentity | None = None,
    sync_parent_on_exit: bool = False,
):
    """Open one child without following it while retaining its authenticated parent."""

    path = Path(path)
    if path.name in {"", ".", ".."}:
        raise ReservationConflictError("relative control filename is invalid")
    if os.name != "nt":
        parent_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        parent_fd = os.open(path.parent, parent_flags)
        child_fd = -1
        try:
            parent_identity = directory_identity_from_posix_fd(parent_fd)
            if (
                expected_parent_identity is not None
                and parent_identity != expected_parent_identity
            ):
                raise ReservationConflictError(
                    "control parent identity changed before child open"
                )
            child_flags = (
                (os.O_RDWR if writable else os.O_RDONLY)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            if create:
                child_flags |= os.O_CREAT
            if exclusive:
                child_flags |= os.O_EXCL
            child_fd = os.open(path.name, child_flags, 0o600, dir_fd=parent_fd)
            identity = file_identity_from_posix_fd(child_fd)
            if identity.link_count != 1:
                raise ReservationConflictError("control file is not single-link")
            mode = "r+b" if writable else "rb"
            with os.fdopen(child_fd, mode, buffering=0, closefd=True) as handle:
                child_fd = -1
                yield handle, identity, parent_identity
            if sync_parent_on_exit:
                os.fsync(parent_fd)
            if directory_identity_from_posix_fd(parent_fd) != parent_identity:
                raise ReservationConflictError("control parent identity changed")
        finally:
            if child_fd >= 0:
                os.close(child_fd)
            os.close(parent_fd)
        return

    import ctypes
    import msvcrt
    from ctypes import wintypes

    class IoStatusBlock(ctypes.Structure):
        _fields_ = [("status", ctypes.c_void_p), ("information", ctypes.c_void_p)]

    class UnicodeString(ctypes.Structure):
        _fields_ = [
            ("length", wintypes.USHORT),
            ("maximum_length", wintypes.USHORT),
            ("buffer", wintypes.LPWSTR),
        ]

    class ObjectAttributes(ctypes.Structure):
        _fields_ = [
            ("length", wintypes.ULONG),
            ("root_directory", wintypes.HANDLE),
            ("object_name", ctypes.POINTER(UnicodeString)),
            ("attributes", wintypes.ULONG),
            ("security_descriptor", wintypes.LPVOID),
            ("security_qos", wintypes.LPVOID),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll")
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    nt_create_file = ntdll.NtCreateFile
    nt_create_file.argtypes = (
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        ctypes.POINTER(ObjectAttributes),
        ctypes.POINTER(IoStatusBlock),
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    nt_create_file.restype = ctypes.c_long
    ntstatus_to_error = ntdll.RtlNtStatusToDosError
    ntstatus_to_error.argtypes = (ctypes.c_long,)
    ntstatus_to_error.restype = wintypes.DWORD
    invalid_handle = ctypes.c_void_p(-1).value
    parent_handle = create_file(
        str(path.parent),
        0x80000000 | 0x00100000,
        0x1 | 0x2 | 0x4,
        None,
        3,
        0x00200000 | 0x02000000,
        None,
    )
    if parent_handle == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    raw_parent = int(parent_handle)
    child_value = -1
    child_fd = -1
    try:
        parent_inner, _links, _attributes = windows_identity_from_handle(
            raw_parent,
            require_directory=True,
        )
        parent_identity = DirectoryIdentity("windows", parent_inner)
        if (
            expected_parent_identity is not None
            and parent_identity != expected_parent_identity
        ):
            raise ReservationConflictError(
                "control parent identity changed before child open"
            )
        name_buffer = ctypes.create_unicode_buffer(path.name)
        name_bytes = path.name.encode("utf-16-le")
        unicode_name = UnicodeString(
            len(name_bytes),
            len(name_bytes) + 2,
            ctypes.cast(name_buffer, wintypes.LPWSTR),
        )
        attributes = ObjectAttributes(
            ctypes.sizeof(ObjectAttributes),
            wintypes.HANDLE(raw_parent),
            ctypes.pointer(unicode_name),
            0x40,
            None,
            None,
        )
        child = wintypes.HANDLE()
        io_status = IoStatusBlock()
        access = 0x80000000 | 0x00100000
        if writable:
            access |= 0x40000000 | 0x00010000
        status = int(
            nt_create_file(
                ctypes.byref(child),
                access,
                ctypes.byref(attributes),
                ctypes.byref(io_status),
                None,
                0,
                0x1 | 0x2 | 0x4,
                2 if exclusive else 3 if create else 1,
                0x00200000 | 0x20 | 0x40,
                None,
                0,
            )
        )
        if status < 0:
            raise ctypes.WinError(int(ntstatus_to_error(status)))
        if child.value is None:
            raise ReservationCapabilityError("NtCreateFile returned a null control handle")
        child_value = int(child.value)
        inner, link_count, _attributes = windows_identity_from_handle(
            child_value,
            require_directory=False,
        )
        if link_count != 1:
            raise ReservationConflictError("control file is not single-link")
        identity = ReservationFileIdentity("windows", inner, link_count)
        flags = (os.O_RDWR if writable else os.O_RDONLY) | getattr(os, "O_BINARY", 0)
        child_fd = msvcrt.open_osfhandle(child_value, flags)
        child_value = -1
        mode = "r+b" if writable else "rb"
        with os.fdopen(child_fd, mode, buffering=0, closefd=True) as handle:
            child_fd = -1
            yield handle, identity, parent_identity
        current_parent, _links, _attributes = windows_identity_from_handle(
            raw_parent,
            require_directory=True,
        )
        if DirectoryIdentity("windows", current_parent) != parent_identity:
            raise ReservationConflictError("control parent identity changed")
    finally:
        if child_fd >= 0:
            os.close(child_fd)
        if child_value not in {-1, invalid_handle}:
            close_handle(wintypes.HANDLE(child_value))
        close_handle(wintypes.HANDLE(raw_parent))


def create_parent_bound_placeholder_v1(
    path: Path,
    *,
    expected_parent_identity: DirectoryIdentity,
) -> ReservationFileIdentity:
    """Create and durably bind one empty child to an authenticated parent."""

    with _open_relative_regular_file(
        Path(path),
        writable=True,
        create=True,
        exclusive=True,
        expected_parent_identity=expected_parent_identity,
        sync_parent_on_exit=True,
    ) as (handle, identity, _parent_identity):
        if os.fstat(handle.fileno()).st_size != 0:
            raise ReservationConflictError("new placeholder is not empty")
        handle.flush()
        os.fsync(handle.fileno())
    return identity


def observe_parent_bound_regular_v1(
    path: Path,
    *,
    expected_parent_identity: DirectoryIdentity,
) -> tuple[ReservationFileIdentity, int]:
    """Observe one regular child through the exact authenticated parent."""

    with _open_relative_regular_file(
        Path(path),
        writable=False,
        create=False,
        expected_parent_identity=expected_parent_identity,
    ) as (handle, identity, _parent_identity):
        return identity, int(os.fstat(handle.fileno()).st_size)


def _read_bounded_regular_file_with_identity(
    path: Path,
    *,
    max_bytes: int,
    expected_parent_identity: DirectoryIdentity | None = None,
) -> tuple[bytes, ReservationFileIdentity, DirectoryIdentity]:
    if max_bytes <= 0:
        raise ValueError("bounded read limit must be positive")
    try:
        with _open_relative_regular_file(
            path,
            writable=False,
            create=False,
            expected_parent_identity=expected_parent_identity,
        ) as (handle, identity, parent_identity):
            assert isinstance(handle, BufferedIOBase) or hasattr(handle, "read")
            size = os.fstat(handle.fileno()).st_size
            if size > max_bytes:
                raise ReservationConflictError(f"control artifact exceeds {max_bytes} bytes")
            raw = handle.read(size + 1)
            if len(raw) != size:
                raise ReservationConflictError("control artifact changed during bounded read")
            return raw, identity, parent_identity
    except FileNotFoundError:
        raise
    except OSError as error:
        if getattr(error, "winerror", None) in {2, 3} or error.errno == errno.ENOENT:
            raise FileNotFoundError(path) from error
        raise


def _read_bounded_regular_file(
    path: Path,
    *,
    max_bytes: int,
    expected_parent_identity: DirectoryIdentity | None = None,
) -> bytes:
    raw, _identity_value, _parent_identity = _read_bounded_regular_file_with_identity(
        path,
        max_bytes=max_bytes,
        expected_parent_identity=expected_parent_identity,
    )
    return raw


def _write_control_temporary(
    path: Path,
    raw: bytes,
    *,
    event: Callable[[str], None] | None,
    expected_parent_identity: DirectoryIdentity | None = None,
) -> tuple[ReservationFileIdentity, DirectoryIdentity]:
    with _open_relative_regular_file(
        path,
        writable=True,
        create=True,
        exclusive=True,
        expected_parent_identity=expected_parent_identity,
    ) as (handle, identity, parent_identity):
        handle.truncate(0)
        handle.seek(0)
        for offset in range(0, len(raw), CONTROL_ARTIFACT_WRITE_CHUNK_BYTES):
            chunk = raw[offset : offset + CONTROL_ARTIFACT_WRITE_CHUNK_BYTES]
            if handle.write(chunk) != len(chunk):
                raise OSError("short control-artifact write")
        if event is not None:
            event("index:temporary")
        handle.flush()
        os.fsync(handle.fileno())
        if os.fstat(handle.fileno()).st_size != len(raw):
            raise ReservationConflictError("control temporary EOF changed")
        return identity, parent_identity


def _sync_complete_control_temporary(
    path: Path,
    raw: bytes,
    *,
    expected_parent_identity: DirectoryIdentity | None = None,
) -> tuple[ReservationFileIdentity, DirectoryIdentity]:
    """Make an authenticated complete temporary durable before promotion."""

    with _open_relative_regular_file(
        path,
        writable=True,
        create=False,
        expected_parent_identity=expected_parent_identity,
    ) as (handle, identity, parent_identity):
        size = os.fstat(handle.fileno()).st_size
        if size != len(raw):
            raise ReservationConflictError("complete control temporary EOF changed")
        handle.seek(0)
        if handle.read(size + 1) != raw:
            raise ReservationConflictError("complete control temporary bytes changed")
        handle.flush()
        os.fsync(handle.fileno())
    if _read_bounded_regular_file(
        path,
        max_bytes=max(len(raw), 1),
        expected_parent_identity=expected_parent_identity,
    ) != raw:
        raise ReservationConflictError("complete control temporary failed reopen validation")
    return identity, parent_identity


def _unlink_safe_control_temporary(
    path: Path,
    *,
    expected_identity: ReservationFileIdentity | None = None,
    expected_parent_identity: DirectoryIdentity | None = None,
    sync_parent: bool = True,
) -> None:
    path = Path(path)
    if not (path.exists() or path.is_symlink()):
        return
    if os.name != "nt":
        parent_fd = os.open(
            path.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        child_fd = -1
        try:
            parent_identity = directory_identity_from_posix_fd(parent_fd)
            child_fd = os.open(
                path.name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_fd,
            )
            identity = file_identity_from_posix_fd(child_fd)
            if expected_parent_identity is not None and parent_identity != expected_parent_identity:
                raise ReservationConflictError("control temporary parent identity changed")
            if expected_identity is not None and identity != expected_identity:
                raise ReservationConflictError("control temporary identity changed")
            if identity.link_count != 1:
                raise ReservationConflictError("control temporary is not single-link")
            try:
                posix_remove_verified_name(
                    parent_fd,
                    path.name,
                    child_fd,
                    directory=False,
                )
            except NoFollowPathError as error:
                raise ReservationConflictError(
                    "control temporary changed during authenticated deletion"
                ) from error
            if directory_identity_from_posix_fd(parent_fd) != parent_identity:
                raise ReservationConflictError("control temporary parent changed")
        finally:
            if child_fd >= 0:
                os.close(child_fd)
            os.close(parent_fd)
        return
    with _open_relative_regular_file(
        path,
        writable=True,
        create=False,
    ) as (handle, opened_identity, parent_identity):
        import ctypes
        import msvcrt
        from ctypes import wintypes

        class FileDispositionInfo(ctypes.Structure):
            _fields_ = [("delete_file", wintypes.BOOL)]

        native = msvcrt.get_osfhandle(handle.fileno())
        current_inner, link_count, _attributes = windows_identity_from_handle(
            native,
            require_directory=False,
        )
        current_identity = ReservationFileIdentity("windows", current_inner, link_count)
        if current_identity != opened_identity:
            raise ReservationConflictError("control temporary identity changed")
        if expected_identity is not None and current_identity != expected_identity:
            raise ReservationConflictError("control temporary identity changed")
        if expected_parent_identity is not None and parent_identity != expected_parent_identity:
            raise ReservationConflictError("control temporary parent identity changed")
        set_information = ctypes.WinDLL(
            "kernel32", use_last_error=True
        ).SetFileInformationByHandle
        set_information.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        )
        set_information.restype = wintypes.BOOL
        disposition = FileDispositionInfo(True)
        if not set_information(
            wintypes.HANDLE(native),
            4,
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if parent_identity != _directory_identity(path.parent):
            raise ReservationConflictError("control temporary parent changed")


def _unlink_exact_control_artifact(
    path: Path,
    expected_raw: bytes,
    *,
    expected_parent_identity: DirectoryIdentity | None = None,
) -> None:
    if not (path.exists() or path.is_symlink()):
        return
    raw, identity, parent_identity = _read_bounded_regular_file_with_identity(
        path,
        max_bytes=max(len(expected_raw), 1),
        expected_parent_identity=expected_parent_identity,
    )
    if raw != expected_raw:
        raise ReservationConflictError(f"control artifact changed before retirement: {path}")
    _unlink_safe_control_temporary(
        path,
        expected_identity=identity,
        expected_parent_identity=parent_identity,
    )


def _write_json_file(
    path: Path,
    payload: dict,
    *,
    event: Callable[[str], None] | None,
    expected_parent_identity: DirectoryIdentity | None = None,
) -> None:
    raw = canonical_json_bytes(payload)
    temporary = path.with_name(f".{path.name}.create-new.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    observed_parent_identity = _directory_identity(path.parent)
    if (
        expected_parent_identity is not None
        and observed_parent_identity != expected_parent_identity
    ):
        raise ReservationConflictError("control parent identity changed before publication")
    parent_authority = expected_parent_identity or observed_parent_identity
    if path.exists() or path.is_symlink():
        if _read_bounded_regular_file(
            path,
            max_bytes=max(len(raw), 1),
            expected_parent_identity=parent_authority,
        ) != raw:
            raise ReservationConflictError(f"authoritative control artifact differs: {path}")
        if temporary.exists() or temporary.is_symlink():
            _temporary_raw, temporary_identity, parent_identity = (
                _read_bounded_regular_file_with_identity(
                    temporary,
                    max_bytes=max(len(raw), 1),
                    expected_parent_identity=parent_authority,
                )
            )
            _unlink_safe_control_temporary(
                temporary,
                expected_identity=temporary_identity,
                expected_parent_identity=parent_identity,
            )
        return
    if temporary.exists() or temporary.is_symlink():
        if _read_bounded_regular_file(
            temporary,
            max_bytes=max(len(raw), 1),
            expected_parent_identity=parent_authority,
        ) != raw:
            raise ReservationIncompleteError(
                f"control temporary is an incomplete durable prefix: {temporary}"
            )
        source_identity, parent_identity = _sync_complete_control_temporary(
            temporary,
            raw,
            expected_parent_identity=parent_authority,
        )
        same_parent_replace_v1(
            temporary,
            path,
            expected_source_identity=source_identity,
            expected_target_identity=None,
            expected_parent_identity=parent_identity,
            replace_existing=False,
        )
        if _read_bounded_regular_file(
            path,
            max_bytes=max(len(raw), 1),
            expected_parent_identity=parent_authority,
        ) != raw:
            raise ReservationConflictError(
                f"published control artifact failed reopen: {path}"
            )
        if event is not None:
            event("index:published")
        return
    source_identity, parent_identity = _write_control_temporary(
        temporary,
        raw,
        event=event,
        expected_parent_identity=parent_authority,
    )
    if _read_bounded_regular_file(
        temporary,
        max_bytes=max(len(raw), 1),
        expected_parent_identity=parent_authority,
    ) != raw:
        raise ReservationConflictError("control temporary failed reopen validation")
    same_parent_replace_v1(
        temporary,
        path,
        expected_source_identity=source_identity,
        expected_target_identity=None,
        expected_parent_identity=parent_identity,
        replace_existing=False,
    )
    if _read_bounded_regular_file(
        path,
        max_bytes=max(len(raw), 1),
        expected_parent_identity=parent_authority,
    ) != raw:
        raise ReservationConflictError(f"published control artifact failed reopen: {path}")
    if event is not None:
        event("index:published")


def publish_index_owned_descriptor_v1(
    path: Path,
    payload: dict,
    *,
    validate_authority: Callable[[], None],
    event: Callable[[str], None] | None = None,
    expected_parent_identity: DirectoryIdentity | None = None,
) -> None:
    """Publish one descriptor, repairing only an authenticated partial temp."""

    if not callable(validate_authority):
        raise TypeError("index-owned descriptor validation must be callable")
    path = Path(path)
    raw = canonical_json_bytes(payload)
    if len(raw) > FINAL_RESERVATION_DESCRIPTOR_RAW_BYTES:
        raise ReservationConflictError(
            "indexed reservation descriptor exceeds its frozen raw-byte bound"
        )
    temporary = path.with_name(f".{path.name}.create-new.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    observed_parent_identity = _directory_identity(path.parent)
    if (
        expected_parent_identity is not None
        and observed_parent_identity != expected_parent_identity
    ):
        raise ReservationConflictError("descriptor parent identity changed before publication")
    parent_authority = expected_parent_identity or observed_parent_identity
    if path.exists() or path.is_symlink():
        if _read_bounded_regular_file(
            path,
            max_bytes=max(len(raw), 1),
            expected_parent_identity=parent_authority,
        ) != raw:
            raise ReservationConflictError(f"authoritative descriptor differs: {path}")
        if temporary.exists() or temporary.is_symlink():
            temporary_raw, temporary_identity, parent_identity = (
                _read_bounded_regular_file_with_identity(
                    temporary,
                    max_bytes=max(len(raw), 1),
                    expected_parent_identity=parent_authority,
                )
            )
            if temporary_raw != raw:
                raise ReservationConflictError(
                    "indexed descriptor has a foreign deterministic temporary"
                )
            _unlink_safe_control_temporary(
                temporary,
                expected_identity=temporary_identity,
                expected_parent_identity=parent_identity,
            )
        return

    if temporary.exists() or temporary.is_symlink():
        with _open_relative_regular_file(
            temporary,
            writable=True,
            create=False,
            expected_parent_identity=parent_authority,
        ) as (handle, source_identity, parent_identity):
            size = os.fstat(handle.fileno()).st_size
            if size > len(raw):
                raise ReservationConflictError(
                    "indexed descriptor temporary exceeds its canonical payload"
                )
            existing = handle.read(size + 1)
            if len(existing) != size:
                raise ReservationConflictError(
                    "indexed descriptor temporary changed during validation"
                )
            if existing != raw:
                if len(existing) >= len(raw) or not raw.startswith(existing):
                    raise ReservationConflictError(
                        "indexed descriptor temporary is not the deterministic payload prefix"
                    )
            validate_authority()
            if existing != raw:
                handle.seek(0)
                handle.truncate(0)
                for offset in range(0, len(raw), CONTROL_ARTIFACT_WRITE_CHUNK_BYTES):
                    chunk = raw[offset : offset + CONTROL_ARTIFACT_WRITE_CHUNK_BYTES]
                    if handle.write(chunk) != len(chunk):
                        raise OSError("short indexed descriptor rewrite")
                if event is not None:
                    event("descriptor:temporary-rewritten")
            handle.flush()
            os.fsync(handle.fileno())
            if os.fstat(handle.fileno()).st_size != len(raw):
                raise ReservationConflictError("indexed descriptor temporary EOF changed")
        if _read_bounded_regular_file(
            temporary,
            max_bytes=max(len(raw), 1),
            expected_parent_identity=parent_authority,
        ) != raw:
            raise ReservationConflictError(
                "indexed descriptor temporary failed reopen validation"
            )
    else:
        validate_authority()
        source_identity, parent_identity = _write_control_temporary(
            temporary,
            raw,
            event=event,
            expected_parent_identity=parent_authority,
        )

    same_parent_replace_v1(
        temporary,
        path,
        expected_source_identity=source_identity,
        expected_target_identity=None,
        expected_parent_identity=parent_identity,
        replace_existing=False,
    )
    if _read_bounded_regular_file(
        path,
        max_bytes=max(len(raw), 1),
        expected_parent_identity=parent_authority,
    ) != raw:
        raise ReservationConflictError("published indexed descriptor failed reopen")


def publish_write_ahead_intent(
    path: Path,
    payload: dict,
    *,
    event: Callable[[str], None] | None = None,
    expected_parent_identity: DirectoryIdentity | None = None,
) -> None:
    try:
        _write_json_file(
            Path(path),
            payload,
            event=event,
            expected_parent_identity=expected_parent_identity,
        )
    except OSError as error:
        if error.errno == errno.ENOSPC or getattr(error, "winerror", None) == 112:
            raise ReservationIncompleteError(f"namespace reservation incomplete: {path}") from error
        raise


def recover_control_artifact_pair_v1(
    path: Path,
    *,
    max_bytes: int,
    parser: Callable[[bytes], object],
    prepublish_validate: Callable[[object], None] | None = None,
    preunlink_validate: Callable[[], None] | None = None,
    expected_parent_identity: DirectoryIdentity | None = None,
) -> object | None:
    """Recover one deterministic create-new final/temporary pair without caller state."""

    path = Path(path)
    temporary = path.with_name(f".{path.name}.create-new.tmp")
    observed_parent_identity = _directory_identity(path.parent)
    if (
        expected_parent_identity is not None
        and observed_parent_identity != expected_parent_identity
    ):
        raise ReservationConflictError("control parent identity changed before recovery")
    parent_authority = expected_parent_identity or observed_parent_identity
    final_present = path.exists() or path.is_symlink()
    temporary_present = temporary.exists() or temporary.is_symlink()
    if not final_present and not temporary_present:
        return None
    if final_present:
        raw = _read_bounded_regular_file(
            path,
            max_bytes=max_bytes,
            expected_parent_identity=parent_authority,
        )
        try:
            result = parser(raw)
        except Exception as error:
            raise ReservationConflictError(
                f"authoritative control artifact is malformed: {path}"
            ) from error
        if temporary_present:
            _temporary_raw, temporary_identity, parent_identity = (
                _read_bounded_regular_file_with_identity(
                    temporary,
                    max_bytes=max_bytes,
                    expected_parent_identity=parent_authority,
                )
            )
            _unlink_safe_control_temporary(
                temporary,
                expected_identity=temporary_identity,
                expected_parent_identity=parent_identity,
            )
            _sync_parent(temporary.parent, expected_identity=parent_authority)
        return result
    raw = _read_bounded_regular_file(
        temporary,
        max_bytes=max_bytes,
        expected_parent_identity=parent_authority,
    )
    try:
        result = parser(raw)
    except Exception:
        if preunlink_validate is not None:
            preunlink_validate()
        _unlink_safe_control_temporary(
            temporary,
            expected_parent_identity=parent_authority,
        )
        _sync_parent(temporary.parent, expected_identity=parent_authority)
        return None
    if prepublish_validate is not None:
        try:
            prepublish_validate(result)
        except ReservationError:
            raise
        except Exception as error:
            raise ReservationConflictError(
                f"control temporary prepublication validation failed: {temporary}"
            ) from error
    source_identity, parent_identity = _sync_complete_control_temporary(
        temporary,
        raw,
        expected_parent_identity=parent_authority,
    )
    same_parent_replace_v1(
        temporary,
        path,
        expected_source_identity=source_identity,
        expected_target_identity=None,
        expected_parent_identity=parent_identity,
        replace_existing=False,
    )
    reopened = _read_bounded_regular_file(
        path,
        max_bytes=max_bytes,
        expected_parent_identity=parent_authority,
    )
    try:
        reparsed = parser(reopened)
    except Exception as error:
        raise ReservationConflictError(
            f"recovered control artifact failed reopen: {path}"
        ) from error
    if reparsed != result:
        raise ReservationConflictError("control artifact changed across recovery rename")
    return reparsed


def materialize_after_index(
    index_path: Path,
    payload: dict,
    *,
    publisher: Callable[..., None] = publish_write_ahead_intent,
) -> None:
    try:
        publisher(Path(index_path), payload)
    except OSError as error:
        if error.errno == errno.ENOSPC or getattr(error, "winerror", None) == 112:
            raise ReservationIncompleteError(
                "indexed namespace materialization is incomplete"
            ) from error
        raise


@dataclass(frozen=True)
class SettingsWriteAheadIntent:
    reservation_generation: str
    target_settings_payload: dict
    target_settings_raw_sha256: str
    fingerprint: str

    @classmethod
    def create(
        cls,
        *,
        reservation_generation: str,
        target_settings_payload: dict,
    ) -> SettingsWriteAheadIntent:
        target_hash = sha256_canonical(target_settings_payload)
        base = {
            "algorithm_version": "settings-write-ahead-intent-v1",
            "reservation_generation": reservation_generation,
            "target_settings_payload": target_settings_payload,
            "target_settings_raw_sha256": target_hash,
        }
        return cls(
            reservation_generation,
            target_settings_payload,
            target_hash,
            sha256_canonical(base),
        )

    def to_object(self) -> dict:
        return {
            "algorithm_version": "settings-write-ahead-intent-v1",
            "reservation_generation": self.reservation_generation,
            "target_settings_payload": self.target_settings_payload,
            "target_settings_raw_sha256": self.target_settings_raw_sha256,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def load(cls, path: Path) -> SettingsWriteAheadIntent:
        raw = Path(path).read_bytes()
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ReservationConflictError("settings intent is malformed") from error
        if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
            raise ReservationConflictError("settings intent is noncanonical")
        expected_keys = {
            "algorithm_version",
            "reservation_generation",
            "target_settings_payload",
            "target_settings_raw_sha256",
            "fingerprint",
        }
        if (
            set(value) != expected_keys
            or value["algorithm_version"] != "settings-write-ahead-intent-v1"
        ):
            raise ReservationConflictError("settings intent schema is invalid")
        intent = cls.create(
            reservation_generation=value["reservation_generation"],
            target_settings_payload=value["target_settings_payload"],
        )
        if intent.to_object() != value:
            raise ReservationConflictError("settings intent hashes are invalid")
        return intent


@dataclass(frozen=True)
class ReservationAllocationEvidence:
    platform: str
    logical_coverage_byte_count: int
    allocated_byte_count: int
    token: str

    def to_object(self) -> dict:
        return {
            "platform": self.platform,
            "logical_coverage_byte_count": self.logical_coverage_byte_count,
            "allocated_byte_count": self.allocated_byte_count,
            "token": self.token,
        }

    @classmethod
    def from_object(cls, value: dict) -> ReservationAllocationEvidence:
        return cls(
            value["platform"],
            int(value["logical_coverage_byte_count"]),
            int(value["allocated_byte_count"]),
            value["token"],
        )


def _identity(path: Path) -> ReservationFileIdentity:
    try:
        identity = capture_file_identity(Path(path))
    except PersistentIdentityCapabilityError as error:
        raise ReservationCapabilityError(str(error)) from error
    except (OSError, PersistentIdentitySchemaError) as error:
        raise ReservationConflictError(f"reservation path identity is invalid: {path}") from error
    if identity.link_count != 1:
        raise ReservationConflictError(f"reservation path is not single-link: {path}")
    return identity


def _directory_identity(path: Path) -> DirectoryIdentity:
    try:
        return capture_directory_identity(Path(path))
    except PersistentIdentityCapabilityError as error:
        raise ReservationCapabilityError(str(error)) from error
    except (OSError, PersistentIdentitySchemaError) as error:
        raise ReservationConflictError(f"directory identity is invalid: {path}") from error


def _identity_token(value: ReservationFileIdentity | None) -> str | None:
    if value is None:
        return None
    return sha256_canonical(value.to_object())


def windows_same_parent_rename_v1(  # noqa: C901
    source_path: Path,
    target_path: Path,
    *,
    expected_source_identity: ReservationFileIdentity,
    expected_target_identity: ReservationFileIdentity | None,
    expected_parent_identity: DirectoryIdentity,
    replace_existing: bool,
    after_namespace_replace: Callable[[], None] | None = None,
) -> str:
    """Perform the frozen handle-relative Windows rename and classify disk state."""

    if os.name != "nt":
        raise ReservationCapabilityError("Windows rename adapter requires Windows")
    import ctypes
    from ctypes import wintypes

    source = Path(source_path)
    target = Path(target_path)
    if source.parent.resolve() != target.parent.resolve() or target.name in {"", ".", ".."}:
        raise ReservationConflictError("Windows rename requires one authenticated parent")
    if any(separator in target.name for separator in ("/", "\\")):
        raise ReservationConflictError("Windows rename target must be one component")

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    invalid_handle = wintypes.HANDLE(-1).value
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    flush = kernel32.FlushFileBuffers
    flush.argtypes = (wintypes.HANDLE,)
    flush.restype = wintypes.BOOL
    ntdll = ctypes.WinDLL("ntdll")

    class IoStatusBlock(ctypes.Structure):
        _fields_ = [("status", ctypes.c_void_p), ("information", ctypes.c_size_t)]

    class UnicodeString(ctypes.Structure):
        _fields_ = [
            ("length", wintypes.USHORT),
            ("maximum_length", wintypes.USHORT),
            ("buffer", wintypes.LPWSTR),
        ]

    class ObjectAttributes(ctypes.Structure):
        _fields_ = [
            ("length", wintypes.ULONG),
            ("root_directory", wintypes.HANDLE),
            ("object_name", ctypes.POINTER(UnicodeString)),
            ("attributes", wintypes.ULONG),
            ("security_descriptor", wintypes.LPVOID),
            ("security_qos", wintypes.LPVOID),
        ]

    rename = ntdll.NtSetInformationFile
    rename.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(IoStatusBlock),
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.c_int,
    )
    rename.restype = ctypes.c_long
    ntstatus_to_error = ntdll.RtlNtStatusToDosError
    ntstatus_to_error.argtypes = (ctypes.c_long,)
    ntstatus_to_error.restype = wintypes.DWORD
    nt_create_file = ntdll.NtCreateFile
    nt_create_file.argtypes = (
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        ctypes.POINTER(ObjectAttributes),
        ctypes.POINTER(IoStatusBlock),
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    nt_create_file.restype = ctypes.c_long

    class RenameUnion(ctypes.Union):
        _fields_ = [("replace_if_exists", ctypes.c_ubyte), ("flags", wintypes.DWORD)]

    class RenameInfo(ctypes.Structure):
        _anonymous_ = ("mode",)
        _fields_ = [
            ("mode", RenameUnion),
            ("root_directory", wintypes.HANDLE),
            ("file_name_length", wintypes.DWORD),
            ("file_name", wintypes.WCHAR * 1),
        ]

    def win_error(code: int | None = None) -> OSError:
        return ctypes.WinError(ctypes.get_last_error() if code is None else code)

    def open_handle(path: Path, access: int, *, directory: bool) -> int:
        flags = 0x00200000 | (0x02000000 if directory else 0)
        handle = create_file(
            str(path),
            access,
            0x00000001 | 0x00000002 | 0x00000004,
            None,
            3,
            flags,
            None,
        )
        if handle == invalid_handle:
            raise win_error()
        return int(handle)

    def open_relative_file(parent: int, name: str, access: int) -> int:
        name_buffer = ctypes.create_unicode_buffer(name)
        name_bytes = name.encode("utf-16-le")
        unicode_name = UnicodeString(
            len(name_bytes),
            len(name_bytes) + 2,
            ctypes.cast(name_buffer, wintypes.LPWSTR),
        )
        attributes = ObjectAttributes(
            ctypes.sizeof(ObjectAttributes),
            wintypes.HANDLE(parent),
            ctypes.pointer(unicode_name),
            0x40,
            None,
            None,
        )
        child = wintypes.HANDLE()
        io_status = IoStatusBlock()
        status = int(
            nt_create_file(
                ctypes.byref(child),
                access,
                ctypes.byref(attributes),
                ctypes.byref(io_status),
                None,
                0,
                0x1 | 0x2 | 0x4,
                1,
                0x00200000 | 0x20 | 0x40,
                None,
                0,
            )
        )
        if status < 0:
            raise win_error(int(ntstatus_to_error(status)))
        if child.value is None:
            raise ReservationCapabilityError("NtCreateFile returned a null child handle")
        return int(child.value)

    def handle_identity(
        handle: int,
        *,
        directory: bool,
    ) -> ReservationFileIdentity | DirectoryIdentity:
        try:
            inner, link_count, _attributes = windows_identity_from_handle(
                handle,
                require_directory=directory,
            )
        except PersistentIdentityCapabilityError as error:
            raise ReservationCapabilityError(str(error)) from error
        except PersistentIdentitySchemaError as error:
            raise ReservationConflictError("Windows rename handle type is unsafe") from error
        if directory:
            return DirectoryIdentity("windows", inner)
        if link_count != 1:
            raise ReservationConflictError("Windows rename source is not single-link")
        return ReservationFileIdentity("windows", inner, link_count)

    def observe_relative(
        parent: int,
        name: str,
        *,
        flush_present: bool,
    ) -> ReservationFileIdentity | None:
        try:
            handle = open_relative_file(
                parent,
                name,
                0x80000000 | 0x40000000 | 0x00100000,
            )
        except OSError as error:
            if getattr(error, "winerror", None) in {2, 3}:
                return None
            raise
        try:
            value = handle_identity(handle, directory=False)
            assert isinstance(value, ReservationFileIdentity)
            if flush_present and not flush(wintypes.HANDLE(handle)):
                raise win_error()
            return value
        finally:
            close_handle(wintypes.HANDLE(handle))

    parent_handle = open_handle(
        source.parent,
        0x00000002 | 0x00100000,
        directory=True,
    )
    source_handle = -1
    try:
        parent_identity = handle_identity(parent_handle, directory=True)
        if parent_identity != expected_parent_identity:
            raise ReservationConflictError("Windows rename parent identity changed")
        source_handle = open_relative_file(
            parent_handle,
            source.name,
            0x80000000 | 0x40000000 | 0x00010000 | 0x00100000,
        )
        if handle_identity(source_handle, directory=False) != expected_source_identity:
            raise ReservationConflictError("Windows rename source identity changed")
        if expected_source_identity.volume_key != expected_parent_identity.volume_key:
            raise ReservationConflictError("Windows rename source volume differs from parent")
        current_target = observe_relative(parent_handle, target.name, flush_present=False)
        if replace_existing:
            if expected_target_identity is None or current_target != expected_target_identity:
                raise ReservationConflictError("Windows rename old target identity changed")
        elif current_target is not None:
            raise ReservationConflictError("Windows create-new rename target is present")
        if not flush(wintypes.HANDLE(source_handle)):
            raise win_error()

        encoded_name = target.name.encode("utf-16-le")
        offset = RenameInfo.file_name.offset
        buffer_size = offset + len(encoded_name) + 2
        raw = ctypes.create_string_buffer(buffer_size)
        info = RenameInfo.from_buffer(raw)
        info.replace_if_exists = int(replace_existing)
        info.root_directory = wintypes.HANDLE(parent_handle)
        info.file_name_length = len(encoded_name)
        ctypes.memmove(ctypes.addressof(raw) + offset, encoded_name, len(encoded_name))

        def invoke(replace_target: bool) -> tuple[bool, int]:
            info.replace_if_exists = int(replace_target)
            if handle_identity(parent_handle, directory=True) != expected_parent_identity:
                raise ReservationConflictError("Windows rename parent changed before call")
            io_status = IoStatusBlock()
            status = int(
                rename(
                    wintypes.HANDLE(source_handle),
                    ctypes.byref(io_status),
                    ctypes.byref(raw),
                    buffer_size,
                    10,
                )
            )
            returned = status >= 0
            saved_error = 0 if returned else int(ntstatus_to_error(status))
            if not flush(wintypes.HANDLE(source_handle)):
                raise win_error()
            if handle_identity(parent_handle, directory=True) != expected_parent_identity:
                raise ReservationConflictError("Windows rename parent changed after call")
            return returned, saved_error

        returned, saved_error = invoke(replace_existing)
        source_token = _identity_token(
            observe_relative(parent_handle, source.name, flush_present=False)
        )
        target_token = _identity_token(
            observe_relative(parent_handle, target.name, flush_present=True)
        )
        expected_source_token = _identity_token(expected_source_identity)
        assert expected_source_token is not None
        if replace_existing:
            expected_old_token = _identity_token(expected_target_identity)
            assert expected_old_token is not None
            classification = classify_windows_rename_return(
                source_identity=source_token,
                target_identity=target_token,
                expected_source_identity=expected_source_token,
                expected_old_target_identity=expected_old_token,
                api_return=returned,
            )
        elif source_token == expected_source_token and target_token is None:
            if returned:
                raise WindowsRenameProtocolError(
                    "true create-new rename return left the source unchanged"
                )
            classification = "unchanged"
        elif source_token is None and target_token == expected_source_token:
            classification = "committed"
        else:
            raise ReservationConflictError(
                "Windows create-new rename returned an unauthorized identity combination"
            )
        if classification == "replacement-gap":
            returned, saved_error = invoke(False)
            source_token = _identity_token(
                observe_relative(parent_handle, source.name, flush_present=False)
            )
            target_token = _identity_token(
                observe_relative(parent_handle, target.name, flush_present=True)
            )
            if source_token is None and target_token == expected_source_token:
                classification = "committed"
            elif source_token == expected_source_token and target_token is None and not returned:
                if saved_error == 112:
                    raise ReservationIncompleteError("Windows rename gap recovery ran out of space")
                raise win_error(saved_error)
            else:
                raise ReservationConflictError(
                    "Windows rename gap recovery produced an unauthorized state"
                )
        if classification == "unchanged":
            if saved_error == 112:
                raise ReservationIncompleteError("Windows rename ran out of space")
            raise win_error(saved_error)
        if classification != "committed":
            raise WindowsRenameProtocolError("Windows rename did not reach committed state")
        if after_namespace_replace is not None:
            after_namespace_replace()
        return classification
    finally:
        if source_handle not in {-1, invalid_handle}:
            close_handle(wintypes.HANDLE(source_handle))
        close_handle(wintypes.HANDLE(parent_handle))


def same_parent_replace_v1(
    source_path: Path,
    target_path: Path,
    *,
    expected_source_identity: ReservationFileIdentity,
    expected_target_identity: ReservationFileIdentity | None,
    expected_parent_identity: DirectoryIdentity,
    replace_existing: bool,
    after_namespace_replace: Callable[[], None] | None = None,
) -> None:
    """Use the frozen platform replacement primitive for an authenticated pair."""

    if os.name == "nt":
        windows_same_parent_rename_v1(
            source_path,
            target_path,
            expected_source_identity=expected_source_identity,
            expected_target_identity=expected_target_identity,
            expected_parent_identity=expected_parent_identity,
            replace_existing=replace_existing,
            after_namespace_replace=after_namespace_replace,
        )
        return
    _same_parent_replace_posix_v1(
        source_path,
        target_path,
        expected_source_identity=expected_source_identity,
        expected_target_identity=expected_target_identity,
        expected_parent_identity=expected_parent_identity,
        replace_existing=replace_existing,
        after_namespace_replace=after_namespace_replace,
    )


def _same_parent_replace_posix_v1(
    source_path: Path,
    target_path: Path,
    *,
    expected_source_identity: ReservationFileIdentity,
    expected_target_identity: ReservationFileIdentity | None,
    expected_parent_identity: DirectoryIdentity,
    replace_existing: bool,
    after_namespace_replace: Callable[[], None] | None = None,
) -> None:
    source = Path(source_path)
    target = Path(target_path)
    if source.parent != target.parent:
        raise ReservationConflictError("rename paths do not share one parent")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    parent_fd = os.open(source.parent, flags)
    source_fd = -1
    try:
        if directory_identity_from_posix_fd(parent_fd) != expected_parent_identity:
            raise ReservationConflictError("rename parent identity changed")
        source_fd = os.open(
            source.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        if file_identity_from_posix_fd(source_fd) != expected_source_identity:
            raise ReservationConflictError("rename source identity changed")
        try:
            target_identity = _identity(target)
        except FileNotFoundError:
            target_identity = None
        if replace_existing:
            if expected_target_identity is None or target_identity != expected_target_identity:
                raise ReservationConflictError("rename old target identity changed")
            os.replace(
                source.name,
                target.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
        else:
            if target_identity is not None:
                raise ReservationConflictError("create-new rename target appeared")
            if not sys.platform.startswith("linux"):
                raise ReservationCapabilityError("atomic no-replace rename is Linux-only")
            import ctypes

            libc = ctypes.CDLL(None, use_errno=True)
            renameat2 = getattr(libc, "renameat2", None)
            if renameat2 is None:
                raise ReservationCapabilityError("renameat2 is unavailable")
            renameat2.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            renameat2.restype = ctypes.c_int
            if renameat2(
                parent_fd,
                os.fsencode(source.name),
                parent_fd,
                os.fsencode(target.name),
                1,
            ) != 0:
                code = ctypes.get_errno()
                if code in {errno.EEXIST, errno.ENOTEMPTY}:
                    raise ReservationConflictError("create-new rename target appeared")
                if code in {errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTTY}:
                    raise ReservationCapabilityError("renameat2 no-replace is unsupported")
                raise OSError(code, os.strerror(code))
        if after_namespace_replace is not None:
            after_namespace_replace()
        os.fsync(parent_fd)
        if directory_identity_from_posix_fd(parent_fd) != expected_parent_identity:
            raise ReservationConflictError("rename parent identity changed after sync")
        if _identity(target) != expected_source_identity:
            raise ReservationConflictError("renamed target identity changed after sync")
    finally:
        if source_fd >= 0:
            os.close(source_fd)
        os.close(parent_fd)


@dataclass(frozen=True)
class ReservationDescriptor:
    job_id: str
    settings_artifact_relative_path: str
    settings_identity_fingerprint: str
    settings_revision_at_creation: int | None
    settings_raw_sha256_at_creation: str | None
    publication_generation: str | None
    reservation_generation: str
    reservation_id: str
    reservation_kind: str
    transaction_index_relative_path: str
    transaction_index_fingerprint: str
    source_extent_relative_path: str
    temporary_relative_path: str
    target_relative_path: str
    target_parent_identity: DirectoryIdentity
    publication_method: PublicationMethod
    existing_target_identity_at_preflight: ReservationFileIdentity | None
    existing_target_byte_count_at_preflight: int | None
    existing_target_raw_sha256_at_preflight: str | None
    reserved_placeholder_identity: ReservationFileIdentity | None
    payload_role: str
    payload_raw_max: int
    logical_byte_count: int
    allocation_evidence_at_readiness: ReservationAllocationEvidence
    allocation_unit: int
    filesystem_identity: ReservationFileIdentity
    prune_root_relative_path: str | None
    marker_name: str | None
    marker_payload_sha256: str | None
    fingerprint: str

    def _base_object(self) -> dict:
        return {
            "schema_version": 1,
            "algorithm_version": "reservation-extent-v1",
            "job_id": self.job_id,
            "settings_artifact_relative_path": self.settings_artifact_relative_path,
            "settings_identity_fingerprint": self.settings_identity_fingerprint,
            "settings_revision_at_creation": self.settings_revision_at_creation,
            "settings_raw_sha256_at_creation": self.settings_raw_sha256_at_creation,
            "publication_generation": self.publication_generation,
            "reservation_generation": self.reservation_generation,
            "reservation_id": self.reservation_id,
            "reservation_kind": self.reservation_kind,
            "transaction_index_relative_path": self.transaction_index_relative_path,
            "transaction_index_fingerprint": self.transaction_index_fingerprint,
            "source_extent_relative_path": self.source_extent_relative_path,
            "temporary_relative_path": self.temporary_relative_path,
            "target_relative_path": self.target_relative_path,
            "target_parent_identity": self.target_parent_identity.to_object(),
            "publication_method": self.publication_method,
            "existing_target_identity_at_preflight": (
                self.existing_target_identity_at_preflight.to_object()
                if self.existing_target_identity_at_preflight is not None
                else None
            ),
            "existing_target_byte_count_at_preflight": (
                self.existing_target_byte_count_at_preflight
            ),
            "existing_target_raw_sha256_at_preflight": (
                self.existing_target_raw_sha256_at_preflight
            ),
            "reserved_placeholder_identity": (
                self.reserved_placeholder_identity.to_object()
                if self.reserved_placeholder_identity is not None
                else None
            ),
            "payload_role": self.payload_role,
            "payload_raw_max": self.payload_raw_max,
            "logical_byte_count": self.logical_byte_count,
            "allocation_evidence_at_readiness": self.allocation_evidence_at_readiness.to_object(),
            "allocation_unit": self.allocation_unit,
            "filesystem_identity": self.filesystem_identity.to_object(),
            "prune_root_relative_path": self.prune_root_relative_path,
            "marker_name": self.marker_name,
            "marker_payload_sha256": self.marker_payload_sha256,
        }

    @property
    def index_relative_path(self) -> str:
        return self.transaction_index_relative_path

    @property
    def generation(self) -> str:
        return self.reservation_generation

    @property
    def source_relative_path(self) -> str:
        return self.source_extent_relative_path

    @property
    def placeholder_identity(self) -> ReservationFileIdentity | None:
        return self.reserved_placeholder_identity

    @property
    def descriptor_relative_path(self) -> str:
        publication = self.publication_generation or "none"
        name = (
            f"d-p-{publication}-r-{self.reservation_generation}-"
            f"{self.reservation_kind}-{self.reservation_id}.json"
        )
        return (Path(".depth-surge-reservations-v1") / name).as_posix()

    def to_object(self) -> dict:
        return {**self._base_object(), "fingerprint": self.fingerprint}

    @classmethod
    def build(cls, **values) -> ReservationDescriptor:
        provisional = cls(**values, fingerprint="")
        return cls(**values, fingerprint=sha256_canonical(provisional._base_object()))

    @classmethod
    def from_object(cls, value: dict) -> ReservationDescriptor:
        expected = {
            "schema_version",
            "algorithm_version",
            "job_id",
            "settings_artifact_relative_path",
            "settings_identity_fingerprint",
            "settings_revision_at_creation",
            "settings_raw_sha256_at_creation",
            "publication_generation",
            "reservation_generation",
            "reservation_id",
            "reservation_kind",
            "transaction_index_relative_path",
            "transaction_index_fingerprint",
            "source_extent_relative_path",
            "temporary_relative_path",
            "target_relative_path",
            "target_parent_identity",
            "publication_method",
            "existing_target_identity_at_preflight",
            "existing_target_byte_count_at_preflight",
            "existing_target_raw_sha256_at_preflight",
            "reserved_placeholder_identity",
            "payload_role",
            "payload_raw_max",
            "logical_byte_count",
            "allocation_evidence_at_readiness",
            "allocation_unit",
            "filesystem_identity",
            "prune_root_relative_path",
            "marker_name",
            "marker_payload_sha256",
            "fingerprint",
        }
        if type(value) is not dict or set(value) != expected:
            raise ReservationConflictError("reservation descriptor schema is not closed")
        if (
            type(value["schema_version"]) is not int
            or value["schema_version"] != 1
            or value["algorithm_version"] != "reservation-extent-v1"
            or value["settings_artifact_relative_path"]
            != SETTINGS_ARTIFACT_RELATIVE_PATH.as_posix()
        ):
            raise ReservationConflictError("reservation descriptor version is invalid")
        nullable_revision = value["settings_revision_at_creation"]
        nullable_settings_hash = value["settings_raw_sha256_at_creation"]
        if (nullable_revision is None) != (nullable_settings_hash is None):
            raise ReservationConflictError("reservation settings preflight pair is incomplete")
        if nullable_revision is not None and (
            type(nullable_revision) is not int or not 0 <= nullable_revision <= 2**64 - 1
        ):
            raise ReservationConflictError("reservation settings revision is invalid")
        if nullable_settings_hash is not None:
            _require_lower_hex(nullable_settings_hash, 64, "reservation settings hash")
        publication = value["publication_generation"]
        if publication is not None:
            _require_lower_hex(publication, 32, "reservation publication generation")
        for key, digits, label in (
            ("job_id", 32, "reservation job id"),
            ("settings_identity_fingerprint", 64, "reservation settings identity"),
            ("reservation_generation", 32, "reservation generation"),
            ("reservation_id", 32, "reservation id"),
            ("transaction_index_fingerprint", 64, "reservation index fingerprint"),
            ("fingerprint", 64, "reservation descriptor fingerprint"),
        ):
            _require_lower_hex(value[key], digits, label)
        if value["reservation_kind"] not in _RESERVATION_KINDS:
            raise ReservationConflictError("reservation descriptor kind is invalid")
        for key in (
            "transaction_index_relative_path",
            "source_extent_relative_path",
            "temporary_relative_path",
            "target_relative_path",
        ):
            _canonical_job_relative_path(value[key], key.replace("_", " "))
        if value["source_extent_relative_path"] != value["temporary_relative_path"]:
            raise ReservationConflictError("reservation source/temporary paths differ")
        if value["publication_method"] not in {"replace_placeholder", "replace_existing"}:
            raise ReservationConflictError("reservation publication method is invalid")
        for key in ("payload_raw_max", "logical_byte_count", "allocation_unit"):
            if type(value[key]) is not int or value[key] <= 0:
                raise ReservationConflictError(f"{key} is invalid")
        if value["logical_byte_count"] != payload_logical_bytes(
            value["payload_raw_max"], value["allocation_unit"]
        ):
            raise ReservationConflictError("reservation logical length equation is invalid")
        descriptor = cls(
            job_id=value["job_id"],
            settings_artifact_relative_path=value["settings_artifact_relative_path"],
            settings_identity_fingerprint=value["settings_identity_fingerprint"],
            settings_revision_at_creation=value["settings_revision_at_creation"],
            settings_raw_sha256_at_creation=value["settings_raw_sha256_at_creation"],
            publication_generation=value["publication_generation"],
            reservation_generation=value["reservation_generation"],
            reservation_id=value["reservation_id"],
            reservation_kind=value["reservation_kind"],
            transaction_index_relative_path=value["transaction_index_relative_path"],
            transaction_index_fingerprint=value["transaction_index_fingerprint"],
            source_extent_relative_path=value["source_extent_relative_path"],
            temporary_relative_path=value["temporary_relative_path"],
            target_relative_path=value["target_relative_path"],
            target_parent_identity=DirectoryIdentity.from_object(value["target_parent_identity"]),
            publication_method=value["publication_method"],
            existing_target_identity_at_preflight=(
                ReservationFileIdentity.from_object(
                    value["existing_target_identity_at_preflight"]
                )
                if value["existing_target_identity_at_preflight"] is not None
                else None
            ),
            existing_target_byte_count_at_preflight=value[
                "existing_target_byte_count_at_preflight"
            ],
            existing_target_raw_sha256_at_preflight=value[
                "existing_target_raw_sha256_at_preflight"
            ],
            reserved_placeholder_identity=(
                ReservationFileIdentity.from_object(value["reserved_placeholder_identity"])
                if value["reserved_placeholder_identity"] is not None
                else None
            ),
            payload_role=value["payload_role"],
            payload_raw_max=value["payload_raw_max"],
            logical_byte_count=value["logical_byte_count"],
            allocation_evidence_at_readiness=ReservationAllocationEvidence.from_object(
                value["allocation_evidence_at_readiness"]
            ),
            allocation_unit=value["allocation_unit"],
            filesystem_identity=ReservationFileIdentity.from_object(value["filesystem_identity"]),
            prune_root_relative_path=value["prune_root_relative_path"],
            marker_name=value["marker_name"],
            marker_payload_sha256=value["marker_payload_sha256"],
            fingerprint=value["fingerprint"],
        )
        if descriptor.fingerprint != sha256_canonical(descriptor._base_object()):
            raise ReservationConflictError("reservation descriptor fingerprint is invalid")
        return descriptor


_RESERVATION_ENTRY_KEYS = {
    "reservation_generation",
    "reservation_id",
    "reservation_kind",
    "descriptor_publication_generation",
    "descriptor_relative_path",
    "source_extent_relative_path",
    "temporary_relative_path",
    "target_relative_path",
    "target_parent_identity",
    "publication_method",
    "existing_target_identity_at_preflight",
    "existing_target_byte_count_at_preflight",
    "existing_target_raw_sha256_at_preflight",
    "payload_role",
    "payload_raw_max",
    "logical_byte_count",
    "allocation_unit",
    "prune_root_relative_path",
    "marker_name",
    "marker_payload_sha256",
}
_RESERVATION_KINDS = {
    "initial_settings",
    "migration_settings",
    "job_terminal_settings",
    "nonterminal_settings",
    "producer_settings",
    "cleanup_pending_settings",
    "encoding_input_manifest",
    "final_video_manifest",
    "prune_marker",
    "diagnostics_payload_pruned",
}
_PAYLOAD_ROLES = {
    "initial_settings": "settings_initial",
    "migration_settings": "settings_migration",
    "job_terminal_settings": "settings_terminal",
    "nonterminal_settings": "settings_nonterminal",
    "producer_settings": "settings_producer",
    "cleanup_pending_settings": "settings_cleanup_pending",
    "encoding_input_manifest": "encoding_input_manifest",
    "final_video_manifest": "final_video_manifest",
    "prune_marker": "prune_marker",
    "diagnostics_payload_pruned": "diagnostics_payload_pruned",
}


def _canonical_job_relative_path(value: object, label: str) -> str:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise ReservationConflictError(f"{label} is not a canonical relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ReservationConflictError(f"{label} is not a canonical relative path")
    if path.as_posix() != value:
        raise ReservationConflictError(f"{label} is not a canonical relative path")
    return value


def validate_reservation_entry_v1(value: object) -> dict:
    """Parse the exact pre-materialization authority for one reserved extent."""

    if type(value) is not dict or set(value) != _RESERVATION_ENTRY_KEYS:
        raise ReservationConflictError("reservation entry schema is not closed")
    assert isinstance(value, dict)
    generation = _require_lower_hex(
        value["reservation_generation"], 32, "reservation generation"
    )
    reservation_id = _require_lower_hex(value["reservation_id"], 32, "reservation id")
    kind = value["reservation_kind"]
    if kind not in _RESERVATION_KINDS:
        raise ReservationConflictError("reservation entry kind is invalid")
    publication = value["descriptor_publication_generation"]
    if publication is not None:
        publication = _require_lower_hex(publication, 32, "descriptor publication generation")
    descriptor = _canonical_job_relative_path(
        value["descriptor_relative_path"], "descriptor relative path"
    )
    expected_descriptor = (
        Path(".depth-surge-reservations-v1")
        / (
            f"d-p-{publication or 'none'}-r-{generation}-{kind}-{reservation_id}.json"
        )
    ).as_posix()
    if descriptor != expected_descriptor:
        raise ReservationConflictError("reservation descriptor name is not canonical")
    source = _canonical_job_relative_path(
        value["source_extent_relative_path"], "source extent relative path"
    )
    temporary = _canonical_job_relative_path(
        value["temporary_relative_path"], "temporary relative path"
    )
    target = _canonical_job_relative_path(value["target_relative_path"], "target relative path")
    expected_source_name = f".depth-surge-r-{generation}-{kind}-{reservation_id}.reserved"
    if source != temporary or PurePosixPath(source).name != expected_source_name:
        raise ReservationConflictError("reservation source/temporary mapping is invalid")
    if PurePosixPath(source).parent != PurePosixPath(target).parent:
        raise ReservationConflictError("reservation source is not target-local")
    DirectoryIdentity.from_object(value["target_parent_identity"])
    method = value["publication_method"]
    if method not in {"replace_placeholder", "replace_existing"}:
        raise ReservationConflictError("reservation publication method is invalid")
    triple = (
        value["existing_target_identity_at_preflight"],
        value["existing_target_byte_count_at_preflight"],
        value["existing_target_raw_sha256_at_preflight"],
    )
    if any(item is not None for item in triple):
        if any(item is None for item in triple) or method != "replace_existing":
            raise ReservationConflictError("reservation existing-target triple is incomplete")
        ReservationFileIdentity.from_object(triple[0])
        if type(triple[1]) is not int or not 0 <= triple[1] <= 2**64 - 1:
            raise ReservationConflictError("reservation existing-target size is invalid")
        _require_lower_hex(triple[2], 64, "reservation existing-target hash")
    settings_kinds = {
        "initial_settings",
        "migration_settings",
        "job_terminal_settings",
        "nonterminal_settings",
        "producer_settings",
        "cleanup_pending_settings",
    }
    if kind in settings_kinds and target != SETTINGS_ARTIFACT_RELATIVE_PATH.as_posix():
        raise ReservationConflictError("settings reservation target is not fixed")
    fixed_targets = {
        "encoding_input_manifest": "encoding_input_manifest.json",
        "final_video_manifest": "final_video_manifest.json",
        "diagnostics_payload_pruned": "04_stereo_diagnostics/metadata.json",
    }
    if kind in fixed_targets and target != fixed_targets[kind]:
        raise ReservationConflictError(f"{kind} reservation target is not fixed")
    placeholder_only = {"initial_settings", "migration_settings", "prune_marker"}
    replace_existing_only = {
        "job_terminal_settings",
        "nonterminal_settings",
        "producer_settings",
        "cleanup_pending_settings",
        "diagnostics_payload_pruned",
    }
    if kind in placeholder_only and method != "replace_placeholder":
        raise ReservationConflictError(f"{kind} must reserve a placeholder")
    if kind in replace_existing_only and method != "replace_existing":
        raise ReservationConflictError(f"{kind} must replace an existing target")
    preflight_bound = kind in {
        "encoding_input_manifest",
        "final_video_manifest",
        "diagnostics_payload_pruned",
    } and method == "replace_existing"
    if preflight_bound != all(item is not None for item in triple):
        raise ReservationConflictError("reservation preflight target binding is invalid")
    if kind in settings_kinds and any(item is not None for item in triple):
        raise ReservationConflictError("typed settings reservation has a raw target triple")
    if value["payload_role"] != _PAYLOAD_ROLES[kind]:
        raise ReservationConflictError("reservation payload role differs from its kind")
    raw_max = value["payload_raw_max"]
    logical = value["logical_byte_count"]
    unit = value["allocation_unit"]
    if any(type(item) is not int or item <= 0 for item in (raw_max, logical, unit)):
        raise ReservationConflictError("reservation byte bounds are invalid")
    if logical != payload_logical_bytes(raw_max, unit):
        raise ReservationConflictError("reservation logical length equation is invalid")
    prune = (
        value["prune_root_relative_path"],
        value["marker_name"],
        value["marker_payload_sha256"],
    )
    if kind == "prune_marker":
        if any(type(item) is not str for item in prune):
            raise ReservationConflictError("prune reservation triple is incomplete")
        root = _canonical_job_relative_path(prune[0], "prune root relative path")
        if (
            re.fullmatch(r"\.depth-surge-prune-v2-[0-9a-f]{32}\.marker", prune[1])
            is None
            or target != f"{root}/{prune[1]}"
            or PurePosixPath(source).parent != PurePosixPath(root)
        ):
            raise ReservationConflictError("prune reservation target differs from its marker")
        _require_lower_hex(prune[2], 64, "prune marker payload hash")
    elif any(item is not None for item in prune):
        raise ReservationConflictError("non-prune reservation has prune fields")
    return dict(value)


def validate_descriptor_against_entry_v1(
    descriptor: ReservationDescriptor,
    entry: object,
    *,
    job_id: str,
    settings_identity_fingerprint: str,
    settings_revision_at_creation: int | None,
    settings_raw_sha256_at_creation: str | None,
    transaction_index_relative_path: Path,
    transaction_index_fingerprint: str,
) -> None:
    """Require a materialized descriptor to equal its complete indexed plan."""

    plan = validate_reservation_entry_v1(entry)
    comparisons = {
        "job_id": job_id,
        "settings_identity_fingerprint": settings_identity_fingerprint,
        "settings_revision_at_creation": settings_revision_at_creation,
        "settings_raw_sha256_at_creation": settings_raw_sha256_at_creation,
        "publication_generation": plan["descriptor_publication_generation"],
        "reservation_generation": plan["reservation_generation"],
        "reservation_id": plan["reservation_id"],
        "reservation_kind": plan["reservation_kind"],
        "transaction_index_relative_path": Path(
            transaction_index_relative_path
        ).as_posix(),
        "transaction_index_fingerprint": transaction_index_fingerprint,
        "source_extent_relative_path": plan["source_extent_relative_path"],
        "temporary_relative_path": plan["temporary_relative_path"],
        "target_relative_path": plan["target_relative_path"],
        "target_parent_identity": DirectoryIdentity.from_object(
            plan["target_parent_identity"]
        ),
        "publication_method": plan["publication_method"],
        "existing_target_identity_at_preflight": (
            ReservationFileIdentity.from_object(
                plan["existing_target_identity_at_preflight"]
            )
            if plan["existing_target_identity_at_preflight"] is not None
            else None
        ),
        "existing_target_byte_count_at_preflight": plan[
            "existing_target_byte_count_at_preflight"
        ],
        "existing_target_raw_sha256_at_preflight": plan[
            "existing_target_raw_sha256_at_preflight"
        ],
        "payload_role": plan["payload_role"],
        "payload_raw_max": plan["payload_raw_max"],
        "logical_byte_count": plan["logical_byte_count"],
        "allocation_unit": plan["allocation_unit"],
        "prune_root_relative_path": plan["prune_root_relative_path"],
        "marker_name": plan["marker_name"],
        "marker_payload_sha256": plan["marker_payload_sha256"],
    }
    for field, expected in comparisons.items():
        if getattr(descriptor, field) != expected:
            raise ReservationConflictError(
                f"reservation descriptor {field} differs from its index"
            )
    expected_descriptor_path = plan["descriptor_relative_path"]
    if descriptor.descriptor_relative_path != expected_descriptor_path:
        raise ReservationConflictError(
            "reservation descriptor filename differs from its index"
        )


@dataclass
class ReservationMetrics:
    reservation_write_bytes: int = 0
    reservation_fsync_count: int = 0
    reservation_wall_time_ms: int = 0


AllocationAdapter = Callable[[Path, int], ReservationAllocationEvidence]
AllocationCounter = Callable[[Path], int]


def _default_counter(path: Path, allocation_unit: int) -> int:
    return _allocate(path.stat().st_size, allocation_unit)


def _windows_standard_info(path_or_handle) -> tuple[int, int]:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class FileStandardInfo(ctypes.Structure):
        _fields_ = [
            ("AllocationSize", ctypes.c_longlong),
            ("EndOfFile", ctypes.c_longlong),
            ("NumberOfLinks", wintypes.DWORD),
            ("DeletePending", wintypes.BOOLEAN),
            ("Directory", wintypes.BOOLEAN),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_info = kernel32.GetFileInformationByHandleEx
    get_info.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    get_info.restype = wintypes.BOOL
    opened = (
        nullcontext(path_or_handle)
        if hasattr(path_or_handle, "fileno")
        else Path(path_or_handle).open("rb", buffering=0)
    )
    with opened as handle:
        native = wintypes.HANDLE(msvcrt.get_osfhandle(handle.fileno()))
        info = FileStandardInfo()
        if not get_info(native, 1, ctypes.byref(info), ctypes.sizeof(info)):
            error = ctypes.get_last_error()
            raise ReservationCapabilityError(
                f"GetFileInformationByHandleEx(FileStandardInfo) failed: {error}"
            )
    return int(info.EndOfFile), int(info.AllocationSize)


def _windows_allocated_ranges(path_or_handle, logical: int) -> tuple[tuple[int, int], ...]:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class FileAllocatedRangeBuffer(ctypes.Structure):
        _fields_ = [
            ("FileOffset", ctypes.c_longlong),
            ("Length", ctypes.c_longlong),
        ]

    fsctl_query_allocated_ranges = 0x000940CF
    error_more_data = 234
    extent_capacity = 64
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    device_io = kernel32.DeviceIoControl
    device_io.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    device_io.restype = wintypes.BOOL
    ranges: list[tuple[int, int]] = []
    start = 0
    opened = (
        nullcontext(path_or_handle)
        if hasattr(path_or_handle, "fileno")
        else Path(path_or_handle).open("rb", buffering=0)
    )
    with opened as handle:
        native = wintypes.HANDLE(msvcrt.get_osfhandle(handle.fileno()))
        while start < logical:
            query = FileAllocatedRangeBuffer(start, logical - start)
            output = (FileAllocatedRangeBuffer * extent_capacity)()
            returned = wintypes.DWORD()
            ctypes.set_last_error(0)
            ok = bool(
                device_io(
                    native,
                    fsctl_query_allocated_ranges,
                    ctypes.byref(query),
                    ctypes.sizeof(query),
                    ctypes.byref(output),
                    ctypes.sizeof(output),
                    ctypes.byref(returned),
                    None,
                )
            )
            error = ctypes.get_last_error()
            if not ok and error != error_more_data:
                raise ReservationCapabilityError(f"FSCTL_QUERY_ALLOCATED_RANGES failed: {error}")
            if returned.value % ctypes.sizeof(FileAllocatedRangeBuffer):
                raise ReservationCapabilityError(
                    "FSCTL_QUERY_ALLOCATED_RANGES returned a partial record"
                )
            count = returned.value // ctypes.sizeof(FileAllocatedRangeBuffer)
            if count == 0:
                raise ReservationIncompleteError("allocated-range coverage has a gap")
            for index in range(count):
                offset = int(output[index].FileOffset)
                length = int(output[index].Length)
                if length <= 0 or offset < start:
                    raise ReservationCapabilityError("allocated-range records are invalid")
                ranges.append((offset, length))
            next_start = ranges[-1][0] + ranges[-1][1]
            if next_start <= start:
                raise ReservationCapabilityError("allocated-range pagination made no progress")
            start = next_start
            if ok:
                break
    return tuple(ranges)


def _linux_fiemap_ranges(path_or_handle, logical: int) -> tuple[tuple[int, int], ...]:
    import fcntl

    fiemap_ioctl = 0xC020660B
    fiemap_flag_sync = 0x00000001
    extent_last = 0x00000001
    forbidden_flags = 0x00000002 | 0x00000004 | 0x00000008 | 0x00000080 | 0x00000100
    forbidden_flags |= 0x00000200 | 0x00000400 | 0x00000800 | 0x00001000
    extent_capacity = 64
    header_size = 32
    extent_size = 56
    ranges: list[tuple[int, int]] = []
    start = 0
    opened = (
        nullcontext(path_or_handle)
        if hasattr(path_or_handle, "fileno")
        else Path(path_or_handle).open("rb", buffering=0)
    )
    with opened as handle:
        while start < logical:
            buffer = bytearray(header_size + extent_capacity * extent_size)
            struct.pack_into(
                "=QQIIII",
                buffer,
                0,
                start,
                logical - start,
                fiemap_flag_sync,
                0,
                extent_capacity,
                0,
            )
            try:
                fcntl.ioctl(  # type: ignore[attr-defined]
                    handle.fileno(), fiemap_ioctl, buffer, True
                )
            except OSError as error:
                raise ReservationCapabilityError("FIEMAP ioctl failed") from error
            _, _, _, mapped, capacity, _ = struct.unpack_from("=QQIIII", buffer, 0)
            if capacity != extent_capacity:
                raise ReservationCapabilityError("FIEMAP changed extent capacity")
            last_flag = False
            next_start = start
            for index in range(mapped):
                record = struct.unpack_from(
                    "=QQQQQIIII",
                    buffer,
                    header_size + index * extent_size,
                )
                offset, length, flags = int(record[0]), int(record[2]), int(record[5])
                if length <= 0 or offset < start or flags & forbidden_flags:
                    raise ReservationCapabilityError("FIEMAP extent flags or order are unsupported")
                ranges.append((offset, length))
                next_start = offset + length
                last_flag = bool(flags & extent_last)
            disposition = classify_fiemap_batch(
                mapped_extents=mapped,
                extent_capacity=extent_capacity,
                last_flag=last_flag,
                previous_start=start,
                next_start=next_start,
                coverage_complete=next_start >= logical,
            )
            if disposition == "done":
                break
            assert isinstance(disposition, int)
            start = disposition
    return tuple(ranges)


def _normalize_allocation_ranges(
    ranges: tuple[tuple[int, int], ...], logical: int
) -> tuple[tuple[int, int], ...]:
    cursor = 0
    normalized: list[tuple[int, int]] = []
    for offset, length in ranges:
        start = max(0, offset)
        end = min(logical, offset + length)
        if end <= start:
            continue
        if start > cursor:
            raise ReservationIncompleteError("allocation evidence has a coverage gap")
        if end > cursor:
            normalized.append((start, end - start))
            cursor = end
        if cursor >= logical:
            break
    if cursor != logical:
        raise ReservationIncompleteError("allocation evidence does not cover logical EOF")
    return tuple(normalized)


def _platform_allocation_counter_source(path_or_handle) -> int:
    if os.name == "nt":
        _, allocated = _windows_standard_info(path_or_handle)
        return allocated
    if sys.platform.startswith("linux"):
        metadata = (
            os.fstat(path_or_handle.fileno())
            if hasattr(path_or_handle, "fileno")
            else Path(path_or_handle).stat(follow_symlinks=False)
        )
        return int(metadata.st_blocks) * 512
    raise ReservationCapabilityError("allocation evidence is unsupported on this platform")


def platform_allocation_counter(path: Path) -> int:
    """Return the platform's physical allocation counter for release proofs."""

    return _platform_allocation_counter_source(Path(path))


def platform_allocation_adapter(
    path: Path,
    logical: int,
) -> ReservationAllocationEvidence:
    """Prove complete physical allocation using the frozen platform adapter."""

    return _platform_allocation_adapter_source(Path(path), logical)


def _platform_allocation_adapter_source(
    path_or_handle,
    logical: int,
) -> ReservationAllocationEvidence:
    if type(logical) is not int or logical <= 0:
        raise ValueError("logical allocation length must be a positive integer")
    if os.name == "nt":
        eof, allocated = _windows_standard_info(path_or_handle)
        if eof != logical:
            raise ReservationIncompleteError("Windows EOF differs from logical reservation")
        ranges = _normalize_allocation_ranges(
            _windows_allocated_ranges(path_or_handle, logical),
            logical,
        )
        platform = "windows-allocated-ranges-v1"
    elif sys.platform.startswith("linux"):
        metadata = (
            os.fstat(path_or_handle.fileno())
            if hasattr(path_or_handle, "fileno")
            else Path(path_or_handle).stat(follow_symlinks=False)
        )
        if metadata.st_size != logical:
            raise ReservationIncompleteError("Linux EOF differs from logical reservation")
        ranges = _normalize_allocation_ranges(
            _linux_fiemap_ranges(path_or_handle, logical),
            logical,
        )
        allocated = _platform_allocation_counter_source(path_or_handle)
        platform = "linux-fiemap-v1"
    else:
        raise ReservationCapabilityError("allocation evidence is unsupported on this platform")
    if allocated < logical:
        raise ReservationIncompleteError("physical allocation is smaller than logical coverage")
    normalized = {
        "algorithm_version": platform,
        "logical_coverage_byte_count": logical,
        "allocated_byte_count": allocated,
        "ranges": [[offset, length] for offset, length in ranges],
    }
    return ReservationAllocationEvidence(
        platform=platform,
        logical_coverage_byte_count=logical,
        allocated_byte_count=allocated,
        token=hashlib.sha256(canonical_json_bytes(normalized)).hexdigest(),
    )


def allocation_unit_for_path(path: Path) -> int:
    """Return the volume allocation unit used by reservation accounting."""

    path = Path(path).resolve()
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        sectors = wintypes.DWORD()
        bytes_per_sector = wintypes.DWORD()
        free_clusters = wintypes.DWORD()
        total_clusters = wintypes.DWORD()
        root = path.anchor
        if not ctypes.windll.kernel32.GetDiskFreeSpaceW(
            root,
            ctypes.byref(sectors),
            ctypes.byref(bytes_per_sector),
            ctypes.byref(free_clusters),
            ctypes.byref(total_clusters),
        ):
            raise ReservationCapabilityError("GetDiskFreeSpaceW failed")
        return int(sectors.value * bytes_per_sector.value)
    if sys.platform.startswith("linux"):
        info = os.statvfs(path)
        return int(info.f_frsize or info.f_bsize)
    raise ReservationCapabilityError("allocation-unit query is unsupported")


class ReservedArtifactTransaction:
    """One target-local extent with phase-discriminated crash recovery."""

    def __init__(
        self,
        *,
        job_root: Path,
        descriptor: ReservationDescriptor,
        descriptor_path: Path,
        allocation_adapter: AllocationAdapter,
        allocation_counter: AllocationCounter | None,
        metrics: ReservationMetrics | None = None,
    ) -> None:
        self.job_root = Path(job_root)
        self.descriptor = descriptor
        self.descriptor_path = Path(descriptor_path)
        self.allocation_adapter = allocation_adapter
        self.allocation_counter = allocation_counter or platform_allocation_counter
        self.metrics = metrics or ReservationMetrics()

    @property
    def source_path(self) -> Path:
        return self.job_root / self.descriptor.source_relative_path

    @property
    def target_path(self) -> Path:
        return self.job_root / self.descriptor.target_relative_path

    @contextmanager
    def open_authenticated_source(
        self,
        *,
        writable: bool = False,
        missing_ok: bool = False,
    ):
        """Open the exact indexed source inode without following its name."""

        try:
            with _open_relative_regular_file(
                self.source_path,
                writable=writable,
                create=False,
                expected_parent_identity=self.descriptor.target_parent_identity,
            ) as (handle, identity, _parent_identity):
                if identity != self.descriptor.filesystem_identity:
                    raise ReservationConflictError("reservation source identity changed")
                yield handle
        except FileNotFoundError:
            if missing_ok:
                yield None
                return
            raise ReservationConflictError("reservation source is absent") from None
        except ReservationConflictError:
            raise
        except (
            NoFollowPathError,
            OSError,
            PersistentIdentityCapabilityError,
            PersistentIdentitySchemaError,
        ) as error:
            raise ReservationConflictError(
                "reservation source cannot be opened without following links"
            ) from error

    @contextmanager
    def open_authenticated_target(
        self,
        *,
        expected_identity: ReservationFileIdentity | None,
        missing_ok: bool = False,
    ):
        """Open one target through the indexed parent and optional exact identity."""

        try:
            with _open_relative_regular_file(
                self.target_path,
                writable=False,
                create=False,
                expected_parent_identity=self.descriptor.target_parent_identity,
            ) as (handle, identity, _parent_identity):
                if expected_identity is not None and identity != expected_identity:
                    label = (
                        "reservation preflight target"
                        if self.descriptor.publication_method == "replace_existing"
                        else "reservation target"
                    )
                    raise ReservationConflictError(f"{label} identity changed")
                yield handle, identity
        except FileNotFoundError:
            if missing_ok:
                yield None
                return
            raise ReservationConflictError("reservation target is absent") from None
        except ReservationConflictError:
            raise
        except (
            NoFollowPathError,
            OSError,
            PersistentIdentityCapabilityError,
            PersistentIdentitySchemaError,
        ) as error:
            raise ReservationConflictError(
                "reservation target cannot be opened without following links"
            ) from error

    @staticmethod
    def _read_open_payload(handle, *, byte_count: int) -> bytes:
        handle.seek(0)
        raw = handle.read(byte_count + 1)
        if len(raw) != byte_count:
            raise ReservationConflictError("reservation payload length changed")
        return raw

    def _allocation_evidence_for_open_source(
        self,
        handle,
        logical: int,
    ) -> ReservationAllocationEvidence:
        if self.allocation_adapter is platform_allocation_adapter:
            return _platform_allocation_adapter_source(handle, logical)
        return self.allocation_adapter(self.source_path, logical)

    def _allocation_counter_for_open_source(self, handle) -> int:
        if self.allocation_counter is platform_allocation_counter:
            return _platform_allocation_counter_source(handle)
        return self.allocation_counter(self.source_path)

    def reset_authenticated_source_to_zero_full(self) -> None:
        """Reset only the exact indexed source handle to its full zero state."""

        zero_chunk = b"\x00" * RESERVATION_ZERO_FILL_CHUNK_BYTES
        with self.open_authenticated_source(writable=True) as handle:
            handle.seek(0)
            remaining = self.descriptor.logical_byte_count
            while remaining:
                chunk = zero_chunk[: min(len(zero_chunk), remaining)]
                if handle.write(chunk) != len(chunk):
                    raise OSError("short authenticated reservation reset")
                remaining -= len(chunk)
            handle.truncate(self.descriptor.logical_byte_count)
            handle.flush()
            os.fsync(handle.fileno())

    @classmethod
    def create(  # noqa: C901
        cls,
        *,
        job_root: Path,
        index_relative_path: Path,
        target_relative_path: Path,
        reservation_kind: str,
        payload_raw_max: int,
        allocation_unit: int,
        allocation_adapter: AllocationAdapter,
        generation: str,
        reservation_id: str,
        allocation_counter: AllocationCounter | None = None,
        expected_publication_method: PublicationMethod | None = None,
        event: Callable[[str], None] | None = None,
        descriptor_publisher: Callable[..., None] = publish_index_owned_descriptor_v1,
        materialize_missing: bool = True,
        cancelled: Callable[[], bool] | None = None,
    ) -> ReservedArtifactTransaction:
        def check_cancelled() -> None:
            if cancelled is not None and cancelled():
                raise InterruptedError("reservation extent materialization was cancelled")

        started = time.monotonic_ns()
        check_cancelled()
        job_root = Path(job_root)
        index_path = job_root / index_relative_path
        if not index_path.is_file():
            raise ReservationConflictError("reservation extent has no durable write-ahead index")
        index_raw, _index_identity, index_parent_identity = (
            _read_bounded_regular_file_with_identity(
                index_path,
                max_bytes=1024 * 1024,
            )
        )
        try:
            index_value = strict_canonical_json_loads(index_raw, max_bytes=1024 * 1024)
        except CanonicalJsonError as error:
            raise ReservationConflictError("reservation write-ahead index is malformed") from error
        if type(index_value) is not dict:
            raise ReservationConflictError("reservation write-ahead index is not an object")
        index_fingerprint = index_value.get("fingerprint")
        if type(index_fingerprint) is not str or len(index_fingerprint) != 64:
            index_fingerprint = hashlib.sha256(index_raw).hexdigest()
        indexed_entries = index_value.get("extent_entries")
        if not isinstance(indexed_entries, list):
            indexed_entries = index_value.get("bootstrap_extent_entries")
        indexed_entry = None
        if isinstance(indexed_entries, list):
            matches = [
                entry
                for entry in indexed_entries
                if isinstance(entry, dict)
                and entry.get("reservation_kind") == reservation_kind
                and entry.get("reservation_id") == reservation_id
                and entry.get("reservation_generation", entry.get("generation")) == generation
            ]
            if len(matches) > 1:
                raise ReservationConflictError("reservation index contains duplicate extent entries")
            indexed_entry = matches[0] if matches else None
        strict_index = index_value.get("algorithm_version") in {
            "final-encoding-reservation-v1",
            "job-control-v1",
            "settings-transition-reservation-v1",
        }
        if strict_index:
            persisted_fingerprint = index_value.get("fingerprint")
            unsigned_index = dict(index_value)
            unsigned_index.pop("fingerprint", None)
            if (
                type(persisted_fingerprint) is not str
                or persisted_fingerprint != sha256_canonical(unsigned_index)
            ):
                raise ReservationConflictError("reservation index fingerprint is invalid")
            if indexed_entry is None:
                raise ReservationConflictError("reservation index has no unique complete extent entry")
            indexed_entry = validate_reservation_entry_v1(indexed_entry)
            preflight_comparisons = {
                "target_relative_path": Path(target_relative_path).as_posix(),
                "reservation_kind": reservation_kind,
                "reservation_generation": generation,
                "reservation_id": reservation_id,
                "payload_raw_max": payload_raw_max,
                "logical_byte_count": payload_logical_bytes(payload_raw_max, allocation_unit),
                "allocation_unit": allocation_unit,
            }
            for field, actual in preflight_comparisons.items():
                if indexed_entry[field] != actual:
                    raise ReservationConflictError(f"indexed {field} changed before readiness")
            if (
                expected_publication_method is not None
                and indexed_entry["publication_method"] != expected_publication_method
            ):
                raise ReservationConflictError("indexed publication method changed before readiness")
        target_path = job_root / target_relative_path
        if strict_index:
            if not target_path.parent.is_dir() or target_path.parent.is_symlink():
                raise ReservationConflictError("indexed reservation target parent is unavailable")
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
        target_parent_identity = _directory_identity(target_path.parent)
        if strict_index and DirectoryIdentity.from_object(
            indexed_entry["target_parent_identity"]
        ) != target_parent_identity:
            raise ReservationConflictError("indexed reservation target parent changed")
        if event is not None:
            event("parent:authenticated")
        target_exists = target_path.exists() or target_path.is_symlink()
        publication_method: PublicationMethod = (
            indexed_entry["publication_method"]
            if strict_index
            else expected_publication_method
        ) or (
            "replace_existing" if target_exists else "replace_placeholder"
        )
        old_target_identity: ReservationFileIdentity | None = None
        old_target_byte_count: int | None = None
        old_target_sha256: str | None = None
        if publication_method == "replace_existing":
            if not target_exists:
                raise ReservationConflictError("declared existing reservation target is absent")
            old_target_scan = _scan_reservation_fixed_file_v1(
                job_root,
                target_path,
                cancelled=cancelled,
            )
            if not isinstance(old_target_scan.identity, ReservationFileIdentity):
                raise ReservationConflictError(
                    "existing reservation target identity type changed"
                )
            old_target_identity = old_target_scan.identity
            old_target_sha256 = old_target_scan.raw_sha256
            old_target_byte_count = old_target_scan.byte_count
            del old_target_scan
            if strict_index:
                expected_identity = indexed_entry["existing_target_identity_at_preflight"]
                expected_count = indexed_entry["existing_target_byte_count_at_preflight"]
                expected_hash = indexed_entry["existing_target_raw_sha256_at_preflight"]
                if expected_identity is not None and (
                    ReservationFileIdentity.from_object(expected_identity)
                    != old_target_identity
                    or expected_count != old_target_byte_count
                    or expected_hash != old_target_sha256
                ):
                    raise ReservationConflictError("indexed existing target changed before readiness")
            placeholder_identity = None
        elif publication_method == "replace_placeholder" and target_exists:
            with _open_relative_regular_file(
                target_path,
                writable=False,
                create=False,
                expected_parent_identity=target_parent_identity,
            ) as (placeholder, placeholder_identity, _parent_identity):
                if os.fstat(placeholder.fileno()).st_size != 0:
                    raise ReservationConflictError(
                        "declared reservation placeholder is nonempty"
                    )
        elif publication_method == "replace_placeholder":
            if not materialize_missing:
                raise ReservationIncompleteError(
                    "reservation placeholder is absent during read-only readiness validation"
                )
            try:
                with _open_relative_regular_file(
                    target_path,
                    writable=True,
                    create=True,
                    exclusive=True,
                    expected_parent_identity=target_parent_identity,
                    sync_parent_on_exit=True,
                ) as (placeholder, placeholder_identity, _parent_identity):
                    placeholder.flush()
                    os.fsync(placeholder.fileno())
            except OSError as error:
                if error.errno == errno.ENOSPC or getattr(error, "winerror", None) == 112:
                    raise ReservationIncompleteError(
                        "placeholder creation is incomplete"
                    ) from error
                raise
            if event is not None:
                event("placeholder:created")
        else:
            raise ValueError("unknown expected reservation publication method")
        logical = payload_logical_bytes(payload_raw_max, allocation_unit)
        source_name = f".depth-surge-r-{generation}-{reservation_kind}-{reservation_id}.reserved"
        source_path = target_path.parent / source_name
        metrics = ReservationMetrics()

        def prepare_source(
            handle,
            *,
            current_size: int,
        ) -> None:
            if current_size > logical:
                raise ReservationConflictError(
                    "reservation source exceeds logical length"
                )
            handle.seek(0)
            while True:
                check_cancelled()
                chunk = handle.read(RESERVATION_ZERO_FILL_CHUNK_BYTES)
                if not chunk:
                    break
                if any(chunk):
                    raise ReservationConflictError(
                        "descriptorless reservation source is not all zero"
                    )
            if not materialize_missing:
                if current_size != logical:
                    raise ReservationIncompleteError(
                        "reservation source is not ready during read-only validation"
                    )
            else:
                handle.seek(current_size)
                if event is not None:
                    event("extent:created")
                remaining = logical - current_size
                zero_chunk = b"\x00" * RESERVATION_ZERO_FILL_CHUNK_BYTES
                while remaining:
                    check_cancelled()
                    chunk = zero_chunk[: min(remaining, len(zero_chunk))]
                    written = handle.write(chunk)
                    if written != len(chunk):
                        raise OSError("short reservation zero fill")
                    metrics.reservation_write_bytes += written
                    remaining -= written
                handle.flush()
                os.fsync(handle.fileno())
                metrics.reservation_fsync_count += 1
            if os.fstat(handle.fileno()).st_size != logical:
                raise ReservationIncompleteError(
                    "reservation extent is not a full zero frame"
                )
            handle.seek(0)
            while True:
                check_cancelled()
                chunk = handle.read(RESERVATION_ZERO_FILL_CHUNK_BYTES)
                if not chunk:
                    break
                if any(chunk):
                    raise ReservationIncompleteError(
                        "reservation extent is not a full zero frame"
                    )

        try:
            try:
                with _open_relative_regular_file(
                    source_path,
                    writable=True,
                    create=False,
                    expected_parent_identity=target_parent_identity,
                ) as (handle, source_identity, _parent_identity):
                    prepare_source(
                        handle,
                        current_size=os.fstat(handle.fileno()).st_size,
                    )
            except FileNotFoundError:
                if not materialize_missing:
                    raise ReservationIncompleteError(
                        "reservation source is absent during read-only validation"
                    ) from None
                with _open_relative_regular_file(
                    source_path,
                    writable=True,
                    create=True,
                    exclusive=True,
                    expected_parent_identity=target_parent_identity,
                    sync_parent_on_exit=True,
                ) as (handle, source_identity, _parent_identity):
                    prepare_source(handle, current_size=0)
        except OSError as error:
            if error.errno == errno.ENOSPC or getattr(error, "winerror", None) == 112:
                raise ReservationIncompleteError("reservation extent fill is incomplete") from error
            raise
        check_cancelled()
        evidence = allocation_adapter(source_path, logical)
        check_cancelled()
        if evidence.logical_coverage_byte_count != logical:
            raise ReservationIncompleteError("reservation allocation coverage is incomplete")
        if event is not None:
            event("extent:ready")
        with _open_relative_regular_file(
            source_path,
            writable=False,
            create=False,
            expected_parent_identity=target_parent_identity,
        ) as (source, current_source_identity, _parent_identity):
            if (
                current_source_identity != source_identity
                or os.fstat(source.fileno()).st_size != logical
            ):
                raise ReservationConflictError(
                    "reservation source changed during allocation proof"
                )
        publication_generation = (
            indexed_entry["descriptor_publication_generation"]
            if strict_index
            else index_value.get("publication_generation")
        )
        if publication_generation is not None and (
            type(publication_generation) is not str or len(publication_generation) != 32
        ):
            raise ReservationConflictError("reservation publication generation is invalid")
        publication_token = publication_generation or "none"
        descriptor_name = (
            f"d-p-{publication_token}-r-{generation}-{reservation_kind}-{reservation_id}.json"
        )
        descriptor_relative = Path(".depth-surge-reservations-v1") / descriptor_name
        settings_value = None
        settings_path = job_root / SETTINGS_ARTIFACT_RELATIVE_PATH
        if settings_path.is_file():
            try:
                settings_value = strict_canonical_json_loads(
                    _read_bounded_regular_file(settings_path, max_bytes=512 * 1024),
                    max_bytes=1024 * 1024,
                )
            except (OSError, CanonicalJsonError):
                settings_value = None
        if settings_value is None:
            bootstrap_intent = index_value.get("bootstrap_settings_intent")
            if isinstance(bootstrap_intent, dict):
                settings_value = bootstrap_intent.get("target_settings_payload")
        metadata = (
            settings_value.get("metadata")
            if isinstance(settings_value, dict)
            and isinstance(settings_value.get("metadata"), dict)
            else {}
        )
        fallback_identity = hashlib.sha256(index_raw).hexdigest()
        job_id = index_value.get("job_id", metadata.get("job_id", fallback_identity[:32]))
        settings_identity = index_value.get(
            "settings_identity_fingerprint",
            metadata.get("settings_identity_fingerprint", fallback_identity),
        )
        settings_revision = index_value.get(
            "settings_revision_at_preflight",
            index_value.get("source_settings_revision"),
        )
        settings_raw_sha256 = index_value.get(
            "settings_raw_sha256_at_preflight",
            index_value.get("source_settings_raw_sha256"),
        )
        payload_roles = {
            "initial_settings": "settings_initial",
            "migration_settings": "settings_migration",
            "job_terminal_settings": "settings_terminal",
            "nonterminal_settings": "settings_nonterminal",
            "settings_transition": "settings_nonterminal",
            "producer_settings": "settings_producer",
            "cleanup_pending_settings": "settings_cleanup_pending",
        }
        payload_role = payload_roles.get(reservation_kind, reservation_kind)
        prune_root = None
        marker_name = None
        marker_sha256 = None
        if indexed_entry is not None:
            indexed_source = indexed_entry.get("source_extent_relative_path")
            indexed_temporary = indexed_entry.get("temporary_relative_path")
            indexed_descriptor = indexed_entry.get("descriptor_relative_path")
            comparisons = {
                "target_relative_path": Path(target_relative_path).as_posix(),
                "publication_method": publication_method,
                "payload_raw_max": payload_raw_max,
                "allocation_unit": allocation_unit,
            }
            for field, actual in comparisons.items():
                if field in indexed_entry and indexed_entry[field] != actual:
                    raise ReservationConflictError(f"indexed {field} changed before readiness")
            if indexed_source is not None and indexed_source != source_path.relative_to(job_root).as_posix():
                raise ReservationConflictError("indexed reservation source path changed")
            if indexed_temporary is not None and indexed_temporary != indexed_source:
                raise ReservationConflictError("indexed reservation temporary path differs")
            if indexed_descriptor is not None and indexed_descriptor != descriptor_relative.as_posix():
                raise ReservationConflictError("indexed reservation descriptor path changed")
            indexed_parent = indexed_entry.get("target_parent_identity")
            if indexed_parent is not None and DirectoryIdentity.from_object(indexed_parent) != target_parent_identity:
                raise ReservationConflictError("indexed reservation target parent changed")
            payload_role = indexed_entry.get("payload_role", payload_role)
            prune_root = indexed_entry.get("prune_root_relative_path")
            marker_name = indexed_entry.get("marker_name")
            marker_sha256 = indexed_entry.get("marker_payload_sha256")
            indexed_existing = indexed_entry.get("existing_target_identity_at_preflight")
            indexed_count = indexed_entry.get("existing_target_byte_count_at_preflight")
            indexed_hash = indexed_entry.get("existing_target_raw_sha256_at_preflight")
            if publication_method == "replace_existing" and indexed_existing is not None:
                if (
                    ReservationFileIdentity.from_object(indexed_existing) != old_target_identity
                    or indexed_count != old_target_byte_count
                    or indexed_hash != old_target_sha256
                ):
                    raise ReservationConflictError("indexed existing target changed before readiness")
        descriptor_existing_identity = old_target_identity
        descriptor_existing_count = old_target_byte_count
        descriptor_existing_hash = old_target_sha256
        if strict_index:
            indexed_existing = indexed_entry["existing_target_identity_at_preflight"]
            descriptor_existing_identity = (
                ReservationFileIdentity.from_object(indexed_existing)
                if indexed_existing is not None
                else None
            )
            descriptor_existing_count = indexed_entry[
                "existing_target_byte_count_at_preflight"
            ]
            descriptor_existing_hash = indexed_entry[
                "existing_target_raw_sha256_at_preflight"
            ]
        descriptor = ReservationDescriptor.build(
            job_id=job_id,
            settings_artifact_relative_path=SETTINGS_ARTIFACT_RELATIVE_PATH.as_posix(),
            settings_identity_fingerprint=settings_identity,
            settings_revision_at_creation=settings_revision,
            settings_raw_sha256_at_creation=settings_raw_sha256,
            publication_generation=publication_generation,
            reservation_generation=generation,
            reservation_id=reservation_id,
            reservation_kind=reservation_kind,
            transaction_index_relative_path=Path(index_relative_path).as_posix(),
            transaction_index_fingerprint=index_fingerprint,
            source_extent_relative_path=source_path.relative_to(job_root).as_posix(),
            temporary_relative_path=source_path.relative_to(job_root).as_posix(),
            target_relative_path=Path(target_relative_path).as_posix(),
            target_parent_identity=target_parent_identity,
            publication_method=publication_method,
            existing_target_identity_at_preflight=descriptor_existing_identity,
            existing_target_byte_count_at_preflight=descriptor_existing_count,
            existing_target_raw_sha256_at_preflight=descriptor_existing_hash,
            reserved_placeholder_identity=placeholder_identity,
            payload_role=payload_role,
            payload_raw_max=payload_raw_max,
            logical_byte_count=logical,
            allocation_evidence_at_readiness=evidence,
            allocation_unit=allocation_unit,
            filesystem_identity=source_identity,
            prune_root_relative_path=prune_root,
            marker_name=marker_name,
            marker_payload_sha256=marker_sha256,
        )
        descriptor_path = job_root / descriptor_relative
        descriptor_parent_authority = (
            index_parent_identity
            if Path(index_relative_path).parent == descriptor_relative.parent
            else None
        )

        def validate_descriptor_authority() -> None:
            if _read_bounded_regular_file(
                index_path,
                max_bytes=1024 * 1024,
                expected_parent_identity=index_parent_identity,
            ) != index_raw:
                raise ReservationConflictError(
                    "reservation index changed before descriptor publication"
                )
            with _open_relative_regular_file(
                source_path,
                writable=False,
                create=False,
                expected_parent_identity=target_parent_identity,
            ) as (source, opened_source_identity, _parent_identity):
                if (
                    opened_source_identity != source_identity
                    or os.fstat(source.fileno()).st_size != logical
                ):
                    raise ReservationConflictError(
                        "reservation source changed before descriptor publication"
                    )
            if publication_method == "replace_placeholder":
                with _open_relative_regular_file(
                    target_path,
                    writable=False,
                    create=False,
                    expected_parent_identity=target_parent_identity,
                ) as (placeholder, opened_placeholder_identity, _parent_identity):
                    if (
                        placeholder_identity is None
                        or opened_placeholder_identity != placeholder_identity
                        or os.fstat(placeholder.fileno()).st_size != 0
                    ):
                        raise ReservationConflictError(
                            "reservation placeholder changed before descriptor publication"
                        )
            else:
                if old_target_identity is None:
                    raise ReservationConflictError(
                        "existing reservation target changed before descriptor publication"
                    )
                _stream_file_sha256_count(
                    job_root,
                    target_path,
                    expected_identity=old_target_identity,
                    expected_raw_sha256=old_target_sha256,
                    expected_byte_count=old_target_byte_count,
                    cancelled=cancelled,
                )

        descriptor_publisher(
            descriptor_path,
            descriptor.to_object(),
            validate_authority=validate_descriptor_authority,
            expected_parent_identity=descriptor_parent_authority,
        )
        if event is not None:
            event("descriptor:published")
        metrics.reservation_wall_time_ms = (time.monotonic_ns() - started) // 1_000_000
        return cls(
            job_root=job_root,
            descriptor=descriptor,
            descriptor_path=descriptor_path,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
            metrics=metrics,
        )

    @classmethod
    def load(
        cls,
        descriptor_path: Path,
        *,
        allocation_adapter: AllocationAdapter,
        allocation_counter: AllocationCounter | None = None,
    ) -> ReservedArtifactTransaction:
        descriptor_path = Path(descriptor_path)
        raw = _read_bounded_regular_file(descriptor_path, max_bytes=1024 * 1024)
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ReservationConflictError("reservation descriptor is malformed") from error
        if canonical_json_bytes(value) != raw:
            raise ReservationConflictError("reservation descriptor is noncanonical")
        descriptor = ReservationDescriptor.from_object(value)
        job_root = descriptor_path.parent.parent
        if descriptor_path != job_root / descriptor.descriptor_relative_path:
            raise ReservationConflictError("reservation descriptor path is incorrect")
        return cls(
            job_root=job_root,
            descriptor=descriptor,
            descriptor_path=descriptor_path,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
        )

    def _validate_placeholder(self) -> None:
        if self.descriptor.publication_method == "replace_placeholder":
            expected = self.descriptor.placeholder_identity
            if expected is None:
                raise ReservationConflictError("reserved target placeholder is unbound")
            with self.open_authenticated_target(expected_identity=expected) as opened:
                assert opened is not None
                handle, _identity_value = opened
                if os.fstat(handle.fileno()).st_size != 0:
                    raise ReservationConflictError(
                        "reserved target placeholder changed"
                    )

    def validate_physical_phase(self, expected_payload: bytes) -> str:
        with self.open_authenticated_source(missing_ok=True) as source:
            if source is not None:
                size = int(os.fstat(source.fileno()).st_size)
                expected_target = (
                    self.descriptor.placeholder_identity
                    if self.descriptor.publication_method == "replace_placeholder"
                    else self.descriptor.existing_target_identity_at_preflight
                )
                with self.open_authenticated_target(
                    expected_identity=expected_target,
                    missing_ok=True,
                ) as target:
                    target_exists = target is not None
                if not target_exists:
                    if (
                        os.name == "nt"
                        and size == len(expected_payload)
                        and self._read_open_payload(source, byte_count=size)
                        == expected_payload
                    ):
                        return "replacement-gap"
                    raise ReservationConflictError(
                        "missing replacement target has no complete-short intent"
                    )
                self._validate_placeholder()
                if (
                    size == self.descriptor.logical_byte_count
                ):
                    evidence = self._allocation_evidence_for_open_source(
                        source,
                        size,
                    )
                    if evidence != self.descriptor.allocation_evidence_at_readiness:
                        raise ReservationEvidenceConflictError(
                            "full-source readiness evidence changed"
                        )
                    return "full-source"
                if (
                    size == len(expected_payload)
                    and self._read_open_payload(source, byte_count=size)
                    == expected_payload
                ):
                    return "short-source"
                raise ReservationConflictError(
                    "reservation source length/content is invalid"
                )
        with self.open_authenticated_target(
            expected_identity=self.descriptor.filesystem_identity,
            missing_ok=True,
        ) as target:
            if target is not None:
                handle, _identity_value = target
                size = int(os.fstat(handle.fileno()).st_size)
                if (
                    size == len(expected_payload)
                    and self._read_open_payload(handle, byte_count=size)
                    == expected_payload
                ):
                    return "committed-target"
                raise ReservationConflictError(
                    "reservation target does not contain the committed extent"
                )
        raise ReservationConflictError("reserved extent exists at neither authorized location")

    def stage_complete_full(self, payload: bytes) -> None:
        if not payload or len(payload) > self.descriptor.payload_raw_max or b"\x00" in payload:
            raise ValueError("reservation payload is outside its canonical raw bound")
        if self.validate_physical_phase(payload) != "full-source":
            raise ReservationConflictError("reservation source is not full")
        with self.open_authenticated_source(writable=True) as handle:
            assert handle is not None
            if os.fstat(handle.fileno()).st_size != self.descriptor.logical_byte_count:
                raise ReservationConflictError("reservation source length changed")
            handle.seek(0)
            if handle.write(payload) != len(payload):
                raise OSError("short reservation payload write")
            handle.flush()
            os.fsync(handle.fileno())
            evidence = self._allocation_evidence_for_open_source(
                handle,
                self.descriptor.logical_byte_count,
            )
            if evidence != self.descriptor.allocation_evidence_at_readiness:
                raise ReservationEvidenceConflictError(
                    "complete-full readiness evidence changed"
                )
            raw = self._read_open_payload(
                handle,
                byte_count=self.descriptor.logical_byte_count,
            )
            if classify_payload_frame(raw, payload) != "complete-full":
                raise ReservationConflictError(
                    "reservation source is not exact complete-full"
                )

    def truncate_to_payload(self, payload: bytes) -> None:
        with self.open_authenticated_source(writable=True) as handle:
            assert handle is not None
            raw = self._read_open_payload(
                handle,
                byte_count=int(os.fstat(handle.fileno()).st_size),
            )
            if classify_payload_frame(raw, payload) != "complete-full":
                raise ReservationConflictError("only complete-full may enter truncate")
            handle.truncate(len(payload))

    def _short_release_recovery(self, payload: bytes) -> None:
        if self.validate_physical_phase(payload) != "short-source":
            raise ReservationConflictError("short release recovery requires complete-short")
        with self.open_authenticated_source(writable=True) as handle:
            assert handle is not None
            handle.flush()
            os.fsync(handle.fileno())
            if self._read_open_payload(handle, byte_count=len(payload)) != payload:
                raise ReservationConflictError("short source changed after file sync")
        threshold = (
            self.descriptor.allocation_evidence_at_readiness.allocated_byte_count
            - self.descriptor.allocation_unit
        )
        if self.allocation_counter(self.source_path) <= threshold:
            return
        with self.open_authenticated_source(writable=True) as handle:
            assert handle is not None
            handle.truncate(len(payload))
            handle.flush()
            os.fsync(handle.fileno())
            if self._read_open_payload(handle, byte_count=len(payload)) != payload:
                raise ReservationConflictError(
                    "short source changed after retry truncate"
                )
        if self.allocation_counter(self.source_path) > threshold:
            raise ReservationIncompleteError(
                "short-source allocation release remains unproved"
            )

    def _stream_open_prefix_and_zero_tail(
        self,
        handle,
        *,
        prefix_byte_count: int,
        expected_sha256: str,
        expected_total_byte_count: int,
    ) -> None:
        if (
            type(prefix_byte_count) is not int
            or prefix_byte_count <= 0
            or prefix_byte_count > self.descriptor.payload_raw_max
            or prefix_byte_count > expected_total_byte_count
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
        ):
            raise ValueError("streamed reservation identity is invalid")
        digest = hashlib.sha256()
        remaining = prefix_byte_count
        handle.seek(0)
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ReservationConflictError("streamed reservation prefix ended early")
            digest.update(chunk)
            remaining -= len(chunk)
        tail_remaining = expected_total_byte_count - prefix_byte_count
        while tail_remaining:
            chunk = handle.read(min(1024 * 1024, tail_remaining))
            if not chunk or any(chunk):
                raise ReservationConflictError("streamed reservation tail is not all zero")
            tail_remaining -= len(chunk)
        if handle.read(1):
            raise ReservationConflictError("streamed reservation has trailing bytes")
        if digest.hexdigest() != expected_sha256:
            raise ReservationConflictError("streamed reservation prefix hash differs")

    def stage_complete_full_streamed(
        self,
        *,
        chunks: Callable[[], Iterable[bytes]],
        prefix_byte_count: int,
        prefix_sha256: str,
    ) -> None:
        """Write one replayable canonical prefix while retaining the reserved zero tail."""

        if (
            not callable(chunks)
            or type(prefix_byte_count) is not int
            or not 0 < prefix_byte_count <= self.descriptor.payload_raw_max
            or not isinstance(prefix_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", prefix_sha256, flags=re.ASCII) is None
        ):
            raise ValueError("streamed reservation identity is invalid")
        with self.open_authenticated_source(writable=True) as handle:
            assert handle is not None
            if os.fstat(handle.fileno()).st_size != self.descriptor.logical_byte_count:
                raise ReservationConflictError("streamed reservation source is not full")
            self._validate_placeholder()
            evidence = self._allocation_evidence_for_open_source(
                handle,
                self.descriptor.logical_byte_count,
            )
            if evidence != self.descriptor.allocation_evidence_at_readiness:
                raise ReservationEvidenceConflictError(
                    "full-source readiness evidence changed"
                )

            digest = hashlib.sha256()
            count = 0
            reached_zero_tail = False
            handle.seek(0)
            for chunk in chunks():
                if not isinstance(chunk, bytes) or not chunk or b"\x00" in chunk:
                    raise ValueError("streamed reservation chunks must be nonempty NUL-free bytes")
                current = handle.read(len(chunk))
                if len(current) != len(chunk):
                    raise ReservationConflictError("streamed reservation source ended early")
                for current_byte, expected_byte in zip(current, chunk, strict=True):
                    if reached_zero_tail:
                        if current_byte != 0:
                            raise ReservationConflictError(
                                "streamed reservation source is not a canonical prefix"
                            )
                    elif current_byte == 0:
                        reached_zero_tail = True
                    elif current_byte != expected_byte:
                        raise ReservationConflictError(
                            "streamed reservation source has foreign semantic intent"
                        )
                digest.update(chunk)
                count += len(chunk)
            remaining = self.descriptor.logical_byte_count - count
            while remaining:
                current = handle.read(
                    min(RESERVATION_ZERO_FILL_CHUNK_BYTES, remaining)
                )
                if not current or any(current):
                    raise ReservationConflictError("streamed reservation tail is not all zero")
                remaining -= len(current)
            if handle.read(1):
                raise ReservationConflictError("streamed reservation source has trailing bytes")
            if count != prefix_byte_count or digest.hexdigest() != prefix_sha256:
                raise ReservationConflictError(
                    "streamed reservation replay identity differs"
                )

            digest = hashlib.sha256()
            count = 0
            handle.seek(0)
            for chunk in chunks():
                if not isinstance(chunk, bytes) or not chunk or b"\x00" in chunk:
                    raise ValueError("streamed reservation chunks must be nonempty NUL-free bytes")
                if handle.write(chunk) != len(chunk):
                    raise OSError("short streamed reservation payload write")
                digest.update(chunk)
                count += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
            if count != prefix_byte_count or digest.hexdigest() != prefix_sha256:
                raise ReservationConflictError(
                    "streamed reservation changed while writing"
                )
            if self._allocation_evidence_for_open_source(
                handle,
                self.descriptor.logical_byte_count,
            ) != self.descriptor.allocation_evidence_at_readiness:
                raise ReservationEvidenceConflictError(
                    "complete-full readiness evidence changed"
                )
            self._stream_open_prefix_and_zero_tail(
                handle,
                prefix_byte_count=prefix_byte_count,
                expected_sha256=prefix_sha256,
                expected_total_byte_count=self.descriptor.logical_byte_count,
            )

    def validate_streamed_physical_phase(
        self,
        *,
        prefix_byte_count: int,
        prefix_sha256: str,
    ) -> str:
        """Validate a large canonical prefix without materializing it in memory."""

        with self.open_authenticated_source(missing_ok=True) as source:
            if source is not None:
                size = int(os.fstat(source.fileno()).st_size)
                expected_target = (
                    self.descriptor.placeholder_identity
                    if self.descriptor.publication_method == "replace_placeholder"
                    else self.descriptor.existing_target_identity_at_preflight
                )
                with self.open_authenticated_target(
                    expected_identity=expected_target,
                    missing_ok=True,
                ) as target:
                    target_exists = target is not None
                if not target_exists:
                    if os.name != "nt" or size != prefix_byte_count:
                        raise ReservationConflictError(
                            "missing streamed replacement target has no complete-short intent"
                        )
                    self._stream_open_prefix_and_zero_tail(
                        source,
                        prefix_byte_count=prefix_byte_count,
                        expected_sha256=prefix_sha256,
                        expected_total_byte_count=size,
                    )
                    return "replacement-gap"
                self._validate_placeholder()
                if size == self.descriptor.logical_byte_count:
                    evidence = self._allocation_evidence_for_open_source(
                        source,
                        size,
                    )
                    if evidence != self.descriptor.allocation_evidence_at_readiness:
                        raise ReservationEvidenceConflictError(
                            "full-source readiness evidence changed"
                        )
                    self._stream_open_prefix_and_zero_tail(
                        source,
                        prefix_byte_count=prefix_byte_count,
                        expected_sha256=prefix_sha256,
                        expected_total_byte_count=size,
                    )
                    return "full-source"
                if size == prefix_byte_count:
                    self._stream_open_prefix_and_zero_tail(
                        source,
                        prefix_byte_count=prefix_byte_count,
                        expected_sha256=prefix_sha256,
                        expected_total_byte_count=size,
                    )
                    return "short-source"
                raise ReservationConflictError(
                    "streamed reservation source length is invalid"
                )
        with self.open_authenticated_target(
            expected_identity=self.descriptor.filesystem_identity,
            missing_ok=True,
        ) as target:
            if target is not None:
                handle, _identity_value = target
                size = int(os.fstat(handle.fileno()).st_size)
                if size != prefix_byte_count:
                    raise ReservationConflictError(
                        "streamed reservation target identity differs"
                    )
                self._stream_open_prefix_and_zero_tail(
                    handle,
                    prefix_byte_count=prefix_byte_count,
                    expected_sha256=prefix_sha256,
                    expected_total_byte_count=prefix_byte_count,
                )
                return "committed-target"
        raise ReservationConflictError("streamed extent exists at neither authorized location")

    def _short_release_recovery_streamed(
        self,
        *,
        prefix_byte_count: int,
        prefix_sha256: str,
    ) -> None:
        if (
            self.validate_streamed_physical_phase(
                prefix_byte_count=prefix_byte_count,
                prefix_sha256=prefix_sha256,
            )
            != "short-source"
        ):
            raise ReservationConflictError("streamed release recovery requires short source")
        with self.open_authenticated_source(writable=True) as handle:
            assert handle is not None
            handle.flush()
            os.fsync(handle.fileno())
            self._stream_open_prefix_and_zero_tail(
                handle,
                prefix_byte_count=prefix_byte_count,
                expected_sha256=prefix_sha256,
                expected_total_byte_count=prefix_byte_count,
            )
        threshold = (
            self.descriptor.allocation_evidence_at_readiness.allocated_byte_count
            - self.descriptor.allocation_unit
        )
        if self.allocation_counter(self.source_path) <= threshold:
            return
        with self.open_authenticated_source(writable=True) as handle:
            assert handle is not None
            handle.truncate(prefix_byte_count)
            handle.flush()
            os.fsync(handle.fileno())
            self._stream_open_prefix_and_zero_tail(
                handle,
                prefix_byte_count=prefix_byte_count,
                expected_sha256=prefix_sha256,
                expected_total_byte_count=prefix_byte_count,
            )
        if self.allocation_counter(self.source_path) > threshold:
            raise ReservationIncompleteError(
                "streamed short-source release remains unproved"
            )

    def _rename_and_sync(self, *, replacement_gap: bool = False) -> None:
        if replacement_gap:
            with self.open_authenticated_target(
                expected_identity=None,
                missing_ok=True,
            ) as target:
                if target is not None:
                    raise ReservationConflictError(
                        "replacement-gap target reappeared"
                    )
            expected_target = None
        else:
            self._validate_placeholder()
            expected_target = self.descriptor.placeholder_identity
            if self.descriptor.publication_method == "replace_existing":
                expected_target = self.descriptor.existing_target_identity_at_preflight
                if expected_target is not None:
                    _stream_file_sha256_count(
                        self.job_root,
                        self.target_path,
                        expected_identity=expected_target,
                        expected_raw_sha256=(
                            self.descriptor.existing_target_raw_sha256_at_preflight
                        ),
                        expected_byte_count=(
                            self.descriptor.existing_target_byte_count_at_preflight
                        ),
                    )
                else:
                    with self.open_authenticated_target(
                        expected_identity=None,
                    ) as target:
                        assert target is not None
                        _handle, expected_target = target
        same_parent_replace_v1(
            self.source_path,
            self.target_path,
            expected_source_identity=self.descriptor.filesystem_identity,
            expected_target_identity=expected_target,
            expected_parent_identity=self.descriptor.target_parent_identity,
            replace_existing=not replacement_gap,
        )

    def _retire_descriptor(self) -> None:
        _unlink_exact_control_artifact(
            self.descriptor_path,
            canonical_json_bytes(self.descriptor.to_object()),
        )

    def consume(self, payload: bytes, *, crash_at: str | None = None) -> str:
        self.stage_complete_full(payload)
        if crash_at == "after-complete-full-sync":
            raise InjectedReservationCrash(crash_at)
        self.truncate_to_payload(payload)
        if crash_at == "after-truncate":
            raise InjectedReservationCrash(crash_at)
        with self.open_authenticated_source(writable=True) as handle:
            assert handle is not None
            handle.flush()
            os.fsync(handle.fileno())
        if crash_at == "after-truncate-sync":
            raise InjectedReservationCrash(crash_at)
        self._short_release_recovery(payload)
        self._rename_and_sync()
        if crash_at == "after-rename-parent-sync":
            raise InjectedReservationCrash(crash_at)
        if self.validate_physical_phase(payload) != "committed-target":
            raise ReservationConflictError("target commit validation failed")
        self._retire_descriptor()
        return "committed-target"

    def recover(self, payload: bytes) -> str:
        phase = self.validate_physical_phase(payload)
        if phase == "full-source":
            with self.open_authenticated_source() as handle:
                assert handle is not None
                raw = self._read_open_payload(
                    handle,
                    byte_count=int(os.fstat(handle.fileno()).st_size),
                )
            state = classify_payload_frame(raw, payload)
            if state != "complete-full":
                raise ReservationConflictError("recovery full source has no semantic intent")
            self.truncate_to_payload(payload)
            phase = "short-source"
        if phase == "short-source":
            self._short_release_recovery(payload)
            self._rename_and_sync()
            phase = "committed-target"
        if phase == "replacement-gap":
            self._rename_and_sync(replacement_gap=True)
            phase = "committed-target"
        if phase == "committed-target":
            if self.validate_physical_phase(payload) != "committed-target":
                raise ReservationConflictError("committed target changed during recovery")
            self._retire_descriptor()
            return phase
        raise ReservationConflictError(f"unsupported reservation recovery phase: {phase}")

    def consume_streamed_prefix(
        self,
        *,
        prefix_byte_count: int,
        prefix_sha256: str,
        crash_at: str | None = None,
    ) -> str:
        phase = self.validate_streamed_physical_phase(
            prefix_byte_count=prefix_byte_count,
            prefix_sha256=prefix_sha256,
        )
        if phase != "full-source":
            raise ReservationConflictError("streamed consumption requires complete-full source")
        with self.open_authenticated_source(writable=True) as handle:
            assert handle is not None
            handle.truncate(prefix_byte_count)
        if crash_at == "after-truncate":
            raise InjectedReservationCrash(crash_at)
        with self.open_authenticated_source(writable=True) as handle:
            assert handle is not None
            handle.flush()
            os.fsync(handle.fileno())
        if crash_at == "after-truncate-sync":
            raise InjectedReservationCrash(crash_at)
        self._short_release_recovery_streamed(
            prefix_byte_count=prefix_byte_count,
            prefix_sha256=prefix_sha256,
        )
        self._rename_and_sync()
        if crash_at == "after-rename-parent-sync":
            raise InjectedReservationCrash(crash_at)
        if (
            self.validate_streamed_physical_phase(
                prefix_byte_count=prefix_byte_count,
                prefix_sha256=prefix_sha256,
            )
            != "committed-target"
        ):
            raise ReservationConflictError("streamed target commit validation failed")
        self._retire_descriptor()
        return "committed-target"

    def recover_streamed_prefix(
        self,
        *,
        prefix_byte_count: int,
        prefix_sha256: str,
    ) -> str:
        phase = self.validate_streamed_physical_phase(
            prefix_byte_count=prefix_byte_count,
            prefix_sha256=prefix_sha256,
        )
        if phase == "full-source":
            with self.open_authenticated_source(writable=True) as handle:
                assert handle is not None
                handle.truncate(prefix_byte_count)
            phase = "short-source"
        if phase == "short-source":
            self._short_release_recovery_streamed(
                prefix_byte_count=prefix_byte_count,
                prefix_sha256=prefix_sha256,
            )
            self._rename_and_sync()
            phase = "committed-target"
        if phase == "replacement-gap":
            self._rename_and_sync(replacement_gap=True)
            phase = "committed-target"
        if phase == "committed-target":
            self.validate_streamed_physical_phase(
                prefix_byte_count=prefix_byte_count,
                prefix_sha256=prefix_sha256,
            )
            self._retire_descriptor()
            return phase
        raise ReservationConflictError(f"unsupported streamed recovery phase: {phase}")


def classify_fiemap_batch(
    *,
    mapped_extents: int,
    extent_capacity: int,
    last_flag: bool,
    previous_start: int,
    next_start: int,
    coverage_complete: bool,
) -> int | str:
    if mapped_extents < 0 or mapped_extents > extent_capacity or extent_capacity <= 0:
        raise ReservationCapabilityError("FIEMAP returned an invalid mapped count")
    if mapped_extents == 0 and not coverage_complete:
        raise ReservationIncompleteError("FIEMAP coverage has an unready gap")
    if 0 < mapped_extents < extent_capacity and not last_flag:
        raise ReservationCapabilityError("FIEMAP returned a non-full batch without LAST")
    if last_flag:
        return "done"
    if next_start <= previous_start:
        raise ReservationCapabilityError("FIEMAP pagination made no forward progress")
    return next_start


def classify_windows_rename_return(
    *,
    source_identity: str | None,
    target_identity: str | None,
    expected_source_identity: str,
    expected_old_target_identity: str,
    api_return: bool,
) -> str:
    if (
        source_identity == expected_source_identity
        and target_identity == expected_old_target_identity
    ):
        if api_return:
            raise WindowsRenameProtocolError("true rename return left both identities unchanged")
        return "unchanged"
    if source_identity is None and target_identity == expected_source_identity:
        return "committed"
    if source_identity == expected_source_identity and target_identity is None:
        return "replacement-gap"
    raise ReservationConflictError("Windows rename returned an unauthorized identity combination")


def classify_final_publication_prefix(
    *,
    active_index: bool,
    video_generation: str,
    input_manifest_generation: str,
    final_manifest_generation: str,
    sibling_present: bool,
) -> str:
    if not active_index:
        if (
            video_generation == input_manifest_generation == final_manifest_generation
            and not sibling_present
        ):
            return "publication-authenticated"
        return "publication-validation-failed"
    if video_generation != "new":
        if sibling_present and video_generation == "old":
            return "pre-boundary-sibling-present"
        return "pre-boundary"
    if sibling_present:
        return "post-boundary-retry-sibling-present"
    if input_manifest_generation != "new":
        return "video-replaced-input-pending"
    if final_manifest_generation != "new":
        return "input-manifest-committed-no-final-intent"
    return "publication-authenticated"


def validate_final_encoding_entry_order(kinds: list[str]) -> None:
    if not kinds or kinds[-1] != "job_terminal_settings":
        raise ReservationConflictError("job terminal reservation must be last")
    cursor = 0
    if kinds[cursor : cursor + 1] == ["producer_settings"]:
        cursor += 1
    required = ["encoding_input_manifest", "final_video_manifest"]
    if kinds[cursor : cursor + 2] != required:
        raise ReservationConflictError("final manifest entries are out of order")
    cursor += 2
    while cursor < len(kinds) - 1 and kinds[cursor] == "prune_marker":
        cursor += 1
    if cursor < len(kinds) - 1 and kinds[cursor] == "diagnostics_payload_pruned":
        cursor += 1
    if cursor < len(kinds) - 1 and kinds[cursor] == "cleanup_pending_settings":
        cursor += 1
    if cursor != len(kinds) - 1:
        raise ReservationConflictError("final encoding reservation entries are out of order")


def _require_lower_hex(value: object, digits: int, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != digits
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReservationConflictError(f"{label} must be {digits} lowercase hexadecimal digits")
    return value


def _validate_legacy_settings_component(value: object) -> None:
    if value is None:
        return
    if type(value) is not str:
        raise ReservationConflictError("legacy settings component is invalid")
    if (
        not value.endswith("-settings.json")
        or value in {".", ".."}
        or any(character in value for character in ("\x00", "/", "\\"))
        or (
            len(value) >= 2
            and value[0].isascii()
            and value[0].isalpha()
            and value[1] == ":"
        )
    ):
        raise ReservationConflictError("legacy settings component is not contained")


def _validate_job_control_contract(control: JobControlV1) -> None:
    """Validate every semantic and physical cross-binding in JobControlV1."""

    _validate_legacy_settings_component(control.legacy_settings_artifact_relative_path)
    entries = tuple(validate_reservation_entry_v1(entry) for entry in control.bootstrap_extent_entries)
    expected_initial_kind = (
        "migration_settings"
        if control.legacy_settings_artifact_relative_path is not None
        else "initial_settings"
    )
    if len(entries) != 2 or [entry["reservation_kind"] for entry in entries] != [
        expected_initial_kind,
        "job_terminal_settings",
    ]:
        raise ReservationConflictError("job-control bootstrap entry order is invalid")
    initial, terminal = entries
    expected_common = {
        "reservation_generation": control.job_reservation_generation,
        "descriptor_publication_generation": None,
        "target_relative_path": SETTINGS_ARTIFACT_RELATIVE_PATH.as_posix(),
        "target_parent_identity": control.job_root_identity.to_object(),
        "payload_raw_max": 512 * 1024,
        "prune_root_relative_path": None,
        "marker_name": None,
        "marker_payload_sha256": None,
    }
    for entry in entries:
        for field, expected in expected_common.items():
            if entry[field] != expected:
                raise ReservationConflictError(
                    f"job-control {field} differs from its locator"
                )
        if any(
            entry[field] is not None
            for field in (
                "existing_target_identity_at_preflight",
                "existing_target_byte_count_at_preflight",
                "existing_target_raw_sha256_at_preflight",
            )
        ):
            raise ReservationConflictError("job-control bootstrap target triple must be null")
    if (
        initial["reservation_id"] != control.initial_settings_reservation_id
        or initial["publication_method"] != "replace_placeholder"
        or terminal["reservation_id"] != control.job_terminal_reservation_id
        or terminal["publication_method"] != "replace_existing"
        or initial["allocation_unit"] != terminal["allocation_unit"]
    ):
        raise ReservationConflictError("job-control reservation plan differs from locator IDs")
    artifact = _parse_bootstrap_target(control)
    metadata = artifact.to_object()["metadata"]
    if (
        metadata["job_reservation_generation"] != control.job_reservation_generation
        or metadata["job_terminal_reservation_id"] != control.job_terminal_reservation_id
    ):
        raise ReservationConflictError("bootstrap payload differs from job reservation identity")


@dataclass(frozen=True)
class JobControlV1:
    """Closed Task 0 form of the fixed schema-5 job locator."""

    job_id: str
    legacy_settings_artifact_relative_path: str | None
    settings_identity_fingerprint: str
    job_reservation_generation: str
    initial_settings_reservation_id: str
    job_terminal_reservation_id: str
    job_root_identity: DirectoryIdentity
    bootstrap_extent_entries: tuple[dict, ...]
    bootstrap_settings_intent: dict
    fingerprint: str

    def _base_object(self) -> dict:
        return {
            "schema_version": 1,
            "algorithm_version": "job-control-v1",
            "job_id": self.job_id,
            "settings_artifact_relative_path": SETTINGS_ARTIFACT_RELATIVE_PATH.as_posix(),
            "legacy_settings_artifact_relative_path": self.legacy_settings_artifact_relative_path,
            "settings_identity_fingerprint": self.settings_identity_fingerprint,
            "job_reservation_generation": self.job_reservation_generation,
            "initial_settings_reservation_id": self.initial_settings_reservation_id,
            "job_terminal_reservation_id": self.job_terminal_reservation_id,
            "job_root_identity": self.job_root_identity.to_object(),
            "bootstrap_extent_entries": list(self.bootstrap_extent_entries),
            "bootstrap_settings_intent": self.bootstrap_settings_intent,
        }

    def to_object(self) -> dict:
        return {**self._base_object(), "fingerprint": self.fingerprint}

    @classmethod
    def create(
        cls,
        *,
        job_root: Path,
        job_id: str,
        settings_identity_fingerprint: str,
        job_reservation_generation: str,
        initial_settings_reservation_id: str,
        job_terminal_reservation_id: str,
        legacy_settings_artifact_relative_path: str | None,
        bootstrap_extent_entries: list[dict],
        bootstrap_settings_intent: dict,
    ) -> JobControlV1:
        _require_lower_hex(job_id, 32, "job_id")
        _require_lower_hex(settings_identity_fingerprint, 64, "settings identity")
        _require_lower_hex(job_reservation_generation, 32, "job reservation generation")
        _require_lower_hex(initial_settings_reservation_id, 32, "initial reservation id")
        _require_lower_hex(job_terminal_reservation_id, 32, "terminal reservation id")
        _validate_legacy_settings_component(legacy_settings_artifact_relative_path)
        bootstrap_extent_entries = [
            validate_reservation_entry_v1(entry) for entry in bootstrap_extent_entries
        ]
        kinds = [entry["reservation_kind"] for entry in bootstrap_extent_entries]
        if kinds not in (
            ["initial_settings", "job_terminal_settings"],
            ["migration_settings", "job_terminal_settings"],
        ):
            raise ReservationConflictError("job-control bootstrap entry order is invalid")
        provisional = cls(
            job_id=job_id,
            legacy_settings_artifact_relative_path=legacy_settings_artifact_relative_path,
            settings_identity_fingerprint=settings_identity_fingerprint,
            job_reservation_generation=job_reservation_generation,
            initial_settings_reservation_id=initial_settings_reservation_id,
            job_terminal_reservation_id=job_terminal_reservation_id,
            job_root_identity=_directory_identity(Path(job_root)),
            bootstrap_extent_entries=tuple(bootstrap_extent_entries),
            bootstrap_settings_intent=bootstrap_settings_intent,
            fingerprint="",
        )
        result = cls(
            **{
                **provisional.__dict__,
                "fingerprint": sha256_canonical(provisional._base_object()),
            }
        )
        _validate_job_control_contract(result)
        return result

    @classmethod
    def from_object(cls, value: object) -> JobControlV1:
        if not isinstance(value, dict):
            raise ReservationConflictError("job control root is not an object")
        expected = {
            "schema_version",
            "algorithm_version",
            "job_id",
            "settings_artifact_relative_path",
            "legacy_settings_artifact_relative_path",
            "settings_identity_fingerprint",
            "job_reservation_generation",
            "initial_settings_reservation_id",
            "job_terminal_reservation_id",
            "job_root_identity",
            "bootstrap_extent_entries",
            "bootstrap_settings_intent",
            "fingerprint",
        }
        if set(value) != expected:
            raise ReservationConflictError("job control has missing or extra keys")
        if value["schema_version"] != 1 or value["algorithm_version"] != "job-control-v1":
            raise ReservationConflictError("job control version is invalid")
        if value["settings_artifact_relative_path"] != SETTINGS_ARTIFACT_RELATIVE_PATH.as_posix():
            raise ReservationConflictError("job control settings locator is not fixed")
        entries = value["bootstrap_extent_entries"]
        intent = value["bootstrap_settings_intent"]
        if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
            raise ReservationConflictError("job control bootstrap entries are invalid")
        entries = [validate_reservation_entry_v1(entry) for entry in entries]
        if not isinstance(intent, dict):
            raise ReservationConflictError("job control bootstrap intent is invalid")
        result = cls(
            job_id=_require_lower_hex(value["job_id"], 32, "job_id"),
            legacy_settings_artifact_relative_path=value["legacy_settings_artifact_relative_path"],
            settings_identity_fingerprint=_require_lower_hex(
                value["settings_identity_fingerprint"], 64, "settings identity"
            ),
            job_reservation_generation=_require_lower_hex(
                value["job_reservation_generation"], 32, "job reservation generation"
            ),
            initial_settings_reservation_id=_require_lower_hex(
                value["initial_settings_reservation_id"], 32, "initial reservation id"
            ),
            job_terminal_reservation_id=_require_lower_hex(
                value["job_terminal_reservation_id"], 32, "terminal reservation id"
            ),
            job_root_identity=DirectoryIdentity.from_object(value["job_root_identity"]),
            bootstrap_extent_entries=tuple(entries),
            bootstrap_settings_intent=intent,
            fingerprint=_require_lower_hex(value["fingerprint"], 64, "job control fingerprint"),
        )
        if result.fingerprint != sha256_canonical(result._base_object()):
            raise ReservationConflictError("job control fingerprint is invalid")
        _validate_job_control_contract(result)
        return result


def publish_job_control(job_root: Path, control: JobControlV1) -> None:
    job_root = Path(job_root)
    if _directory_identity(job_root) != control.job_root_identity:
        raise ReservationConflictError("job control belongs to a different job root")
    publish_write_ahead_intent(
        job_root / JOB_CONTROL_RELATIVE_PATH,
        control.to_object(),
        expected_parent_identity=control.job_root_identity,
    )


_FINAL_ENCODING_RESERVED_PAYLOAD_PARENTS_V1 = (
    "00_original_frames",
    "01_scene_data",
    "02_depth_raw",
    "03_disparity_maps",
    "03_metric_geometry",
    "03_disparity_stabilized",
    "04_left_frames",
    "04_right_frames",
    "05_left_distorted",
    "05_right_distorted",
    "06_left_cropped",
    "06_right_cropped",
    "07_left_upscaled",
    "07_right_upscaled",
    "99_vr_frames",
    "04_stereo_diagnostics",
    "04_stereo_diagnostics/frames",
)


def _iter_reserved_payload_parents_v1(root: Path):
    yield PurePosixPath("."), root
    for relative in _FINAL_ENCODING_RESERVED_PAYLOAD_PARENTS_V1:
        path = root / relative
        if not (path.exists() or path.is_symlink()):
            continue
        if path.is_symlink() or not path.is_dir():
            raise ReservationConflictError(
                f"reserved payload parent is unsafe: {relative}"
            )
        yield PurePosixPath(relative), path


def _assert_job_control_candidate_has_no_downstream(job_root: Path) -> None:
    root = Path(job_root)
    fixed = root / SETTINGS_ARTIFACT_RELATIVE_PATH
    if fixed.exists() or fixed.is_symlink():
        raise ReservationConflictError(
            "job control candidate has an unindexed fixed settings target"
        )
    reservation_dir = root / ".depth-surge-reservations-v1"
    if reservation_dir.exists() or reservation_dir.is_symlink():
        if reservation_dir.is_symlink() or not reservation_dir.is_dir():
            raise ReservationConflictError("reservation namespace is unsafe")
        with os.scandir(reservation_dir) as entries:
            if next(entries, None) is not None:
                raise ReservationConflictError(
                    "job control candidate has unindexed reservation artifacts"
                )
    for _relative_parent, parent in _iter_reserved_payload_parents_v1(root):
        with os.scandir(parent) as entries:
            for entry in entries:
                if (
                    entry.name.startswith(".depth-surge-r-")
                    or entry.name.startswith(".depth-surge-final-v4-")
                    or entry.name.startswith(".depth-surge-prune-v2-")
                ):
                    raise ReservationConflictError(
                        "job control candidate has an unindexed reserved payload"
                    )


def assert_only_job_control_owned_reservations_v1(
    job_root: Path,
    control: JobControlV1,
    *,
    allowed_relative_paths: tuple[str, ...] = (),
) -> None:
    """Reject reservation-named objects not authorized by JobControl or a candidate."""

    root = Path(job_root)
    allowed = {PurePosixPath(value).as_posix() for value in allowed_relative_paths}
    for entry in control.bootstrap_extent_entries:
        descriptor = PurePosixPath(entry["descriptor_relative_path"])
        allowed.add(descriptor.as_posix())
        allowed.add(
            (descriptor.parent / f".{descriptor.name}.create-new.tmp").as_posix()
        )
        allowed.add(PurePosixPath(entry["source_extent_relative_path"]).as_posix())
    reservation_dir = root / ".depth-surge-reservations-v1"
    if reservation_dir.exists() or reservation_dir.is_symlink():
        if reservation_dir.is_symlink() or not reservation_dir.is_dir():
            raise ReservationConflictError("reservation namespace is unsafe")
        with os.scandir(reservation_dir) as entries:
            for entry in entries:
                relative = PurePosixPath(".depth-surge-reservations-v1", entry.name).as_posix()
                if relative not in allowed:
                    raise ReservationConflictError(
                        f"unindexed reservation artifact is present: {relative}"
                    )
    for relative_parent, parent in _iter_reserved_payload_parents_v1(root):
        with os.scandir(parent) as entries:
            for entry in entries:
                if not (
                    entry.name.startswith(".depth-surge-r-")
                    or entry.name.startswith(".depth-surge-final-v4-")
                    or entry.name.startswith(".depth-surge-prune-v2-")
                ):
                    continue
                relative = (
                    PurePosixPath(entry.name)
                    if relative_parent == PurePosixPath(".")
                    else relative_parent / entry.name
                ).as_posix()
                if relative not in allowed:
                    raise ReservationConflictError(
                        f"unindexed reserved payload is present: {relative}"
                    )


def assert_index_owned_reservation_namespace_v1(
    job_root: Path,
    control: JobControlV1,
    *,
    index_relative_path: str,
    extent_entries: tuple[dict, ...],
    extra_relative_paths: tuple[str, ...] = (),
) -> None:
    root = Path(job_root)
    index = PurePosixPath(index_relative_path)
    allowed = {
        index.as_posix(),
        (index.parent / f".{index.name}.create-new.tmp").as_posix(),
        *extra_relative_paths,
    }
    local_names: dict[Path, set[str]] = {}
    for entry in extent_entries:
        descriptor = PurePosixPath(entry["descriptor_relative_path"])
        source = PurePosixPath(entry["source_extent_relative_path"])
        allowed.add(descriptor.as_posix())
        allowed.add(
            (descriptor.parent / f".{descriptor.name}.create-new.tmp").as_posix()
        )
        allowed.add(source.as_posix())
        source_path = root / source.as_posix()
        local_names.setdefault(source_path.parent, set()).add(source_path.name)
        if entry["reservation_kind"] == "prune_marker":
            target_relative = PurePosixPath(entry["target_relative_path"])
            allowed.add(target_relative.as_posix())
            target = root / target_relative.as_posix()
            local_names.setdefault(target.parent, set()).add(target.name)
    assert_only_job_control_owned_reservations_v1(
        root,
        control,
        allowed_relative_paths=tuple(sorted(allowed)),
    )
    for parent, names in local_names.items():
        if parent == root or not parent.exists():
            continue
        if parent.is_symlink() or not parent.is_dir():
            raise ReservationConflictError("indexed reservation parent is unsafe")
        with os.scandir(parent) as children:
            for child in children:
                if (
                    child.name.startswith(".depth-surge-r-")
                    or child.name.startswith(".depth-surge-prune-v2-")
                ) and child.name not in names:
                    raise ReservationConflictError(
                        f"unindexed target-local reservation artifact is present: {child.name}"
                    )


def load_job_control(job_root: Path) -> JobControlV1:
    job_root = Path(job_root)
    path = job_root / JOB_CONTROL_RELATIVE_PATH
    def parse(raw: bytes) -> JobControlV1:
        return JobControlV1.from_object(
            strict_canonical_json_loads(raw, max_bytes=1024 * 1024)
        )

    def validate_before_publish(value: object) -> None:
        if not isinstance(value, JobControlV1):
            raise ReservationConflictError("job control temporary has the wrong type")
        if _directory_identity(job_root) != value.job_root_identity:
            raise ReservationConflictError("job control belongs to a different job root")
        _assert_job_control_candidate_has_no_downstream(job_root)

    try:
        recovered = recover_control_artifact_pair_v1(
            path,
            max_bytes=1024 * 1024,
            parser=parse,
            prepublish_validate=validate_before_publish,
            preunlink_validate=lambda: _assert_job_control_candidate_has_no_downstream(
                job_root
            ),
        )
    except (OSError, CanonicalJsonError) as error:
        raise ReservationConflictError("job control is missing or malformed") from error
    if not isinstance(recovered, JobControlV1):
        raise ReservationConflictError("job control is missing or malformed")
    control = recovered
    if _directory_identity(job_root) != control.job_root_identity:
        raise ReservationConflictError("job root identity differs from job control")
    return control


def resolve_fixed_settings_path(job_root: Path, control: JobControlV1) -> Path:
    job_root = Path(job_root)
    if _directory_identity(job_root) != control.job_root_identity:
        raise ReservationConflictError("job root identity differs from job control")
    path = job_root / SETTINGS_ARTIFACT_RELATIVE_PATH
    if not path.is_file() or path.is_symlink():
        raise ReservationConflictError("fixed settings artifact is missing or unsafe")
    return path


@dataclass(frozen=True)
class JobTerminalObservation:
    current_status: str
    current_revision: int
    source_frame: str
    descriptor_present: bool
    descriptor_temporary_present: bool = False
    target_has_reserved_identity: bool = False
    intent_status: str | None = None
    intent_revision: int | None = None
    intent_is_legal: bool = False
    fixed_index_active: bool = False


def reconcile_job_control_terminal_state(  # noqa: C901
    observation: JobTerminalObservation,
) -> str:
    if observation.fixed_index_active:
        return "defer-to-fixed-index"
    if observation.current_revision < 0:
        raise ReservationConflictError("settings revision is invalid")
    if observation.current_status == "in_progress":
        if not observation.descriptor_present:
            raise ReservationConflictError("in-progress terminal reserve has no descriptor")
        if observation.source_frame == "zero":
            return "retain-zero-terminal-reserve"
        if observation.source_frame == "partial":
            return "reset-partial-terminal-reserve"
        if observation.source_frame in {"complete-full", "complete-short"}:
            if (
                observation.intent_is_legal
                and observation.intent_status in {"completed", "failed"}
                and observation.intent_revision == observation.current_revision + 1
            ):
                return "publish-exact-terminal-intent"
            raise ReservationConflictError("terminal intent is not the unique legal next revision")
        raise ReservationConflictError("in-progress terminal extent is in an illegal state")
    if observation.current_status in {"completed", "failed"}:
        if observation.source_frame != "absent":
            raise ReservationConflictError("terminal settings retain an unexpected source extent")
        if observation.descriptor_present:
            if not observation.target_has_reserved_identity:
                raise ReservationConflictError("terminal target identity is not reserved")
            return "retire-consumed-terminal-descriptor"
        if observation.descriptor_temporary_present:
            return "retire-residual-terminal-temporary"
        return "terminal-reserve-consumed"
    raise ReservationConflictError("current settings status is outside the closed lifecycle")


def recovery_audit_order() -> tuple[str, ...]:
    return (
        "job-control-publication",
        "job-control-bootstrap",
        "fixed-index-classification",
        "settings-transition-recovery",
        "final-encoding-recovery",
        "job-control-owned-extents",
        "ordinary-artifact-audit",
    )


@dataclass(frozen=True)
class FinalPublicationObservation:
    video_state: str
    input_manifest_state: str
    final_manifest_state: str
    sibling_present: bool


@dataclass(frozen=True)
class FinalPublicationRecovery:
    phase: str
    action: str


def classify_final_publication_observation(  # noqa: C901
    observation: FinalPublicationObservation,
) -> FinalPublicationRecovery:
    video = observation.video_state
    input_state = observation.input_manifest_state
    final_state = observation.final_manifest_state
    sibling = observation.sibling_present
    if video == "reserved-target":
        if input_state not in {"zero", "partial", "complete-full", "encoding-prepass-staged"}:
            raise ReservationConflictError("pre-boundary input manifest state is illegal")
        if final_state != "zero":
            raise ReservationConflictError("final-manifest intent exists before video publication")
        action = "remove-sibling-and-encode" if sibling else "encode"
        return FinalPublicationRecovery("ready_pre_video", action)
    if video == "windows-replacement-gap":
        if not sibling or final_state != "zero":
            raise ReservationConflictError("Windows replacement gap has an illegal namespace")
        if input_state == "complete-full":
            return FinalPublicationRecovery(
                "windows_video_replace_gap_before_input_commit", "forward-rename-video"
            )
        if input_state == "committed-new":
            return FinalPublicationRecovery(
                "windows_video_replace_gap_after_input_commit", "forward-rename-video"
            )
        raise ReservationConflictError("Windows replacement gap has no legal input prefix")
    if video != "committed-new":
        raise ReservationConflictError("final video is outside the indexed physical states")
    if input_state in {"complete-full", "complete-short", "replacement-gap"}:
        if sibling or final_state != "zero":
            raise ReservationConflictError("input intent is not the unique post-video prefix")
        return FinalPublicationRecovery("video_replaced_input_intent", "commit-input-manifest")
    if input_state != "committed-new":
        raise ReservationConflictError("post-boundary input manifest has no semantic intent")
    if final_state in {"zero", "partial"}:
        if sibling:
            return FinalPublicationRecovery(
                "post_boundary_retry_sibling_present",
                "reset-final-intent-remove-sibling-reencode",
            )
        return FinalPublicationRecovery(
            "input_manifest_committed_no_final_intent", "revalidate-input-and-reencode"
        )
    if sibling:
        raise ReservationConflictError("complete final intent cannot coexist with a sibling")
    if final_state in {"complete-full", "complete-short", "replacement-gap"}:
        return FinalPublicationRecovery(
            "final_manifest_intent", "validate-and-commit-final-manifest"
        )
    if final_state == "committed-new":
        return FinalPublicationRecovery("publication_authenticated", "continue-finalization")
    raise ReservationConflictError("final manifest is outside the indexed physical states")


def final_index_retirement_order(
    *,
    entry_names: tuple[str, ...],
    descriptor_names: tuple[str, ...],
) -> tuple[str, ...]:
    if len(entry_names) != len(descriptor_names):
        raise ValueError("every final-encoding entry requires one descriptor name")
    steps = [
        "validate-terminal-settings",
        "reconcile-sibling-and-control-temporary",
    ]
    steps.extend(f"reconcile:{name}" for name in entry_names)
    steps.append("reconcile:final-video-target")
    steps.extend(f"unlink:{name}" for name in descriptor_names)
    steps.extend(
        (
            "unlink:final-video-target-descriptor",
            "sync-and-prove-descriptors-absent",
            "unlink:final-encoding-reservation-v1.json",
            "sync-and-prove-index-absent",
        )
    )
    return tuple(steps)


def _validate_restart_terminal_entry(
    value: object,
    target: ProcessingSettingsArtifactV5,
) -> dict | None:
    if value is None:
        return None
    value = validate_reservation_entry_v1(value)
    metadata = target.to_object()["metadata"]
    if (
        value["reservation_kind"] != "job_terminal_settings"
        or value["reservation_generation"] != metadata["job_reservation_generation"]
        or value["reservation_id"] != metadata["job_terminal_reservation_id"]
        or value["target_relative_path"] != SETTINGS_ARTIFACT_RELATIVE_PATH.as_posix()
        or value["publication_method"] != "replace_existing"
        or value["payload_raw_max"] != 512 * 1024
        or type(value["allocation_unit"]) is not int
        or value["allocation_unit"] <= 0
    ):
        raise ReservationConflictError("restart terminal entry does not match job control")
    return dict(value)


@dataclass(frozen=True)
class SettingsTransitionIntentV1:
    """Durable semantic and physical plan for one schema-5 replacement."""

    job_id: str
    settings_identity_fingerprint: str
    settings_revision_at_preflight: int
    settings_raw_sha256_at_preflight: str
    target_settings_revision: int
    target_settings_content_fingerprint: str
    target_settings_raw_sha256: str
    target_settings_payload: dict
    transition_kind: Literal["nonterminal_rewrite", "attempt_restart"]
    publication_generation: str | None
    reservation_generation: str
    extent_entries: tuple[dict, ...]
    fingerprint: str

    def _base_object(self) -> dict:
        return {
            "schema_version": 1,
            "algorithm_version": "settings-transition-reservation-v1",
            "job_id": self.job_id,
            "settings_artifact_relative_path": SETTINGS_ARTIFACT_RELATIVE_PATH.as_posix(),
            "settings_identity_fingerprint": self.settings_identity_fingerprint,
            "settings_revision_at_preflight": self.settings_revision_at_preflight,
            "settings_raw_sha256_at_preflight": self.settings_raw_sha256_at_preflight,
            "target_settings_revision": self.target_settings_revision,
            "target_settings_content_fingerprint": self.target_settings_content_fingerprint,
            "target_settings_raw_sha256": self.target_settings_raw_sha256,
            "target_settings_payload": self.target_settings_payload,
            "transition_kind": self.transition_kind,
            "publication_generation": self.publication_generation,
            "reservation_generation": self.reservation_generation,
            "extent_entries": list(self.extent_entries),
        }

    @property
    def source_settings_revision(self) -> int:
        return self.settings_revision_at_preflight

    @property
    def source_settings_raw_sha256(self) -> str:
        return self.settings_raw_sha256_at_preflight

    @property
    def reservation_id(self) -> str:
        return self.extent_entries[0]["reservation_id"]

    @property
    def allocation_unit(self) -> int:
        return self.extent_entries[0]["allocation_unit"]

    @property
    def replacement_terminal_entry(self) -> dict | None:
        return dict(self.extent_entries[1]) if len(self.extent_entries) == 2 else None

    def to_object(self) -> dict:
        return {**self._base_object(), "fingerprint": self.fingerprint}

    @classmethod
    def create(
        cls,
        *,
        job_root: Path,
        current: ProcessingSettingsArtifactV5,
        target: ProcessingSettingsArtifactV5,
        reservation_generation: str,
        reservation_id: str,
        allocation_unit: int,
        replacement_terminal_entry: dict | None = None,
    ) -> SettingsTransitionIntentV1:
        _require_lower_hex(reservation_generation, 32, "settings reservation generation")
        _require_lower_hex(reservation_id, 32, "settings reservation id")
        if type(allocation_unit) is not int or allocation_unit <= 0:
            raise ReservationConflictError("settings allocation unit is invalid")
        current_value = current.to_object()
        target_value = target.to_object()
        current_metadata = current_value["metadata"]
        target_metadata = target_value["metadata"]
        if target.revision != current.revision + 1:
            raise ReservationConflictError("settings target is not the next revision")
        immutable_identity = (
            "job_id",
            "settings_identity_fingerprint",
            "settings_artifact_relative_path",
            "job_reservation_generation",
            "job_terminal_reservation_id",
            "created_at",
        )
        if any(current_metadata[key] != target_metadata[key] for key in immutable_identity):
            raise ReservationConflictError("settings transition changed immutable identity")
        if any(
            current_value[key] != target_value[key] for key in ("video_properties", "output_info")
        ):
            raise ReservationConflictError("settings transition changed immutable job inputs")
        attempt_restart = (
            current.status in {"completed", "failed"} and target.status == "in_progress"
        )
        if attempt_restart:
            if (
                replacement_terminal_entry is None
                or target_metadata["processing_attempt"]
                != current_metadata["processing_attempt"] + 1
            ):
                raise ReservationConflictError(
                    "attempt restart lacks its replacement terminal extent"
                )
            transition_kind: Literal["nonterminal_rewrite", "attempt_restart"] = "attempt_restart"
        else:
            if (
                current.status != "in_progress"
                or target.status != "in_progress"
                or replacement_terminal_entry is not None
                or target_metadata["processing_attempt"] != current_metadata["processing_attempt"]
            ):
                raise ReservationConflictError("settings transition status delta is invalid")
            transition_kind = "nonterminal_rewrite"
        terminal_entry = _validate_restart_terminal_entry(replacement_terminal_entry, target)
        current_raw = current.encode()
        target_raw = target.encode()
        publication_generation = target_metadata["final_media_publication_generation"]
        source_name = (
            f".depth-surge-r-{reservation_generation}-nonterminal_settings-"
            f"{reservation_id}.reserved"
        )
        descriptor_name = (
            f"d-p-{publication_generation or 'none'}-r-{reservation_generation}-"
            f"nonterminal_settings-{reservation_id}.json"
        )
        primary_entry = validate_reservation_entry_v1(
            {
                "reservation_generation": reservation_generation,
                "reservation_id": reservation_id,
                "reservation_kind": "nonterminal_settings",
                "descriptor_publication_generation": publication_generation,
                "descriptor_relative_path": (
                    Path(".depth-surge-reservations-v1") / descriptor_name
                ).as_posix(),
                "source_extent_relative_path": source_name,
                "temporary_relative_path": source_name,
                "target_relative_path": SETTINGS_ARTIFACT_RELATIVE_PATH.as_posix(),
                "target_parent_identity": _directory_identity(Path(job_root)).to_object(),
                "publication_method": "replace_existing",
                "existing_target_identity_at_preflight": None,
                "existing_target_byte_count_at_preflight": None,
                "existing_target_raw_sha256_at_preflight": None,
                "payload_role": "settings_nonterminal",
                "payload_raw_max": 512 * 1024,
                "logical_byte_count": payload_logical_bytes(512 * 1024, allocation_unit),
                "allocation_unit": allocation_unit,
                "prune_root_relative_path": None,
                "marker_name": None,
                "marker_payload_sha256": None,
            }
        )
        entries = (primary_entry,) + ((terminal_entry,) if terminal_entry is not None else ())
        provisional = cls(
            job_id=target_metadata["job_id"],
            settings_identity_fingerprint=target_metadata["settings_identity_fingerprint"],
            settings_revision_at_preflight=current.revision,
            settings_raw_sha256_at_preflight=hashlib.sha256(current_raw).hexdigest(),
            target_settings_revision=target.revision,
            target_settings_content_fingerprint=target_value["content_fingerprint"],
            target_settings_raw_sha256=hashlib.sha256(target_raw).hexdigest(),
            target_settings_payload=target_value,
            transition_kind=transition_kind,
            publication_generation=publication_generation,
            reservation_generation=reservation_generation,
            extent_entries=entries,
            fingerprint="",
        )
        return cls(
            **{
                **provisional.__dict__,
                "fingerprint": sha256_canonical(provisional._base_object()),
            }
        )

    @classmethod
    def from_object(cls, value: object) -> SettingsTransitionIntentV1:
        from .settings_artifact import ProcessingSettingsArtifactV5

        expected = {
            "schema_version",
            "algorithm_version",
            "job_id",
            "settings_artifact_relative_path",
            "settings_identity_fingerprint",
            "settings_revision_at_preflight",
            "settings_raw_sha256_at_preflight",
            "target_settings_revision",
            "target_settings_content_fingerprint",
            "target_settings_raw_sha256",
            "target_settings_payload",
            "transition_kind",
            "publication_generation",
            "reservation_generation",
            "extent_entries",
            "fingerprint",
        }
        if type(value) is not dict or set(value) != expected:
            raise ReservationConflictError("settings transition intent schema is invalid")
        assert isinstance(value, dict)
        if (
            value["schema_version"] != 1
            or type(value["schema_version"]) is not int
            or value["algorithm_version"] != "settings-transition-reservation-v1"
            or value["settings_artifact_relative_path"]
            != SETTINGS_ARTIFACT_RELATIVE_PATH.as_posix()
        ):
            raise ReservationConflictError("settings transition intent version is invalid")
        for key in ("settings_revision_at_preflight", "target_settings_revision"):
            if type(value[key]) is not int or not 0 <= value[key] <= 2**64 - 1:
                raise ReservationConflictError(f"{key} is invalid")
        if value["target_settings_revision"] != value["settings_revision_at_preflight"] + 1:
            raise ReservationConflictError("settings intent revision equation is invalid")
        target = ProcessingSettingsArtifactV5.from_object(value["target_settings_payload"])
        target_value = target.to_object()
        transition_kind = value["transition_kind"]
        if transition_kind not in {"nonterminal_rewrite", "attempt_restart"}:
            raise ReservationConflictError("settings transition kind is invalid")
        raw_entries = value["extent_entries"]
        if not isinstance(raw_entries, list):
            raise ReservationConflictError("settings transition entries are invalid")
        entries = tuple(validate_reservation_entry_v1(entry) for entry in raw_entries)
        if not entries or len(entries) > 2 or entries[0]["reservation_kind"] != "nonterminal_settings":
            raise ReservationConflictError("settings transition primary entry is invalid")
        terminal_entry = _validate_restart_terminal_entry(
            entries[1] if len(entries) == 2 else None,
            target,
        )
        if (transition_kind == "attempt_restart") != (terminal_entry is not None):
            raise ReservationConflictError("settings transition terminal entry is inconsistent")
        publication_generation = value["publication_generation"]
        if publication_generation is not None:
            publication_generation = _require_lower_hex(
                publication_generation, 32, "settings publication generation"
            )
        if entries[0]["descriptor_publication_generation"] != publication_generation:
            raise ReservationConflictError("settings entry publication generation differs")
        target_raw = target.encode()
        result = cls(
            job_id=_require_lower_hex(value["job_id"], 32, "job id"),
            settings_identity_fingerprint=_require_lower_hex(
                value["settings_identity_fingerprint"], 64, "settings identity"
            ),
            settings_revision_at_preflight=value["settings_revision_at_preflight"],
            settings_raw_sha256_at_preflight=_require_lower_hex(
                value["settings_raw_sha256_at_preflight"],
                64,
                "source settings raw hash",
            ),
            target_settings_revision=value["target_settings_revision"],
            target_settings_content_fingerprint=_require_lower_hex(
                value["target_settings_content_fingerprint"], 64, "target content fingerprint"
            ),
            target_settings_raw_sha256=_require_lower_hex(
                value["target_settings_raw_sha256"], 64, "target settings raw hash"
            ),
            target_settings_payload=target_value,
            transition_kind=transition_kind,
            publication_generation=publication_generation,
            reservation_generation=_require_lower_hex(
                value["reservation_generation"], 32, "settings reservation generation"
            ),
            extent_entries=entries,
            fingerprint=_require_lower_hex(value["fingerprint"], 64, "settings intent fingerprint"),
        )
        if (
            target.revision != result.target_settings_revision
            or target_value["metadata"]["job_id"] != result.job_id
            or target_value["metadata"]["settings_identity_fingerprint"]
            != result.settings_identity_fingerprint
            or target_value["content_fingerprint"] != result.target_settings_content_fingerprint
            or hashlib.sha256(target_raw).hexdigest() != result.target_settings_raw_sha256
            or sha256_canonical(result._base_object()) != result.fingerprint
        ):
            raise ReservationConflictError("settings transition intent hashes or identity differ")
        return result


class SettingsArtifactTransactionV1:
    """Indexed, reservation-backed replacement of processing-settings-v5.json."""

    def __init__(
        self,
        *,
        job_root: Path,
        intent: SettingsTransitionIntentV1,
        transaction: ReservedArtifactTransaction,
        replacement_terminal: ReservedArtifactTransaction | None = None,
    ) -> None:
        self.job_root = Path(job_root)
        self.intent = intent
        self.transaction = transaction
        self.replacement_terminal = replacement_terminal

    @property
    def index_path(self) -> Path:
        return self.job_root / SETTINGS_TRANSITION_INDEX_RELATIVE_PATH

    @staticmethod
    def _descriptor_path(
        job_root: Path,
        *,
        generation: str,
        reservation_kind: str,
        reservation_id: str,
    ) -> Path:
        return (
            Path(job_root)
            / ".depth-surge-reservations-v1"
            / f"d-p-none-r-{generation}-{reservation_kind}-{reservation_id}.json"
        )

    @staticmethod
    def _validate_transaction_plan(
        transaction: ReservedArtifactTransaction,
        *,
        entry: dict,
        job_id: str,
        settings_identity_fingerprint: str,
        settings_revision_at_creation: int | None,
        settings_raw_sha256_at_creation: str | None,
        index_relative_path: Path,
        index_fingerprint: str,
    ) -> None:
        validate_descriptor_against_entry_v1(
            transaction.descriptor,
            entry,
            job_id=job_id,
            settings_identity_fingerprint=settings_identity_fingerprint,
            settings_revision_at_creation=settings_revision_at_creation,
            settings_raw_sha256_at_creation=settings_raw_sha256_at_creation,
            transaction_index_relative_path=index_relative_path,
            transaction_index_fingerprint=index_fingerprint,
        )

    @classmethod
    def _load_or_create_primary(
        cls,
        *,
        job_root: Path,
        intent: SettingsTransitionIntentV1,
        allocation_adapter: AllocationAdapter,
        allocation_counter: AllocationCounter | None,
    ) -> ReservedArtifactTransaction:
        entry = intent.extent_entries[0]
        descriptor_path = Path(job_root) / entry["descriptor_relative_path"]
        if descriptor_path.is_symlink() or (
            descriptor_path.exists() and not descriptor_path.is_file()
        ):
            raise ReservationConflictError("settings descriptor path is unsafe")
        if descriptor_path.is_file():
            transaction = ReservedArtifactTransaction.load(
                descriptor_path,
                allocation_adapter=allocation_adapter,
                allocation_counter=allocation_counter,
            )
        else:
            transaction = ReservedArtifactTransaction.create(
                job_root=job_root,
                index_relative_path=SETTINGS_TRANSITION_INDEX_RELATIVE_PATH,
                target_relative_path=SETTINGS_ARTIFACT_RELATIVE_PATH,
                reservation_kind="nonterminal_settings",
                payload_raw_max=512 * 1024,
                allocation_unit=intent.allocation_unit,
                allocation_adapter=allocation_adapter,
                allocation_counter=allocation_counter,
                generation=intent.reservation_generation,
                reservation_id=intent.reservation_id,
                expected_publication_method="replace_existing",
            )
        cls._validate_transaction_plan(
            transaction,
            entry=entry,
            job_id=intent.job_id,
            settings_identity_fingerprint=intent.settings_identity_fingerprint,
            settings_revision_at_creation=intent.settings_revision_at_preflight,
            settings_raw_sha256_at_creation=intent.settings_raw_sha256_at_preflight,
            index_relative_path=SETTINGS_TRANSITION_INDEX_RELATIVE_PATH,
            index_fingerprint=intent.fingerprint,
        )
        return transaction

    @classmethod
    def _load_or_create_replacement_terminal(
        cls,
        *,
        job_root: Path,
        intent: SettingsTransitionIntentV1,
        target: ProcessingSettingsArtifactV5,
        allocation_adapter: AllocationAdapter,
        allocation_counter: AllocationCounter | None,
    ) -> ReservedArtifactTransaction | None:
        entry = intent.replacement_terminal_entry
        if entry is None:
            return None
        descriptor_path = Path(job_root) / entry["descriptor_relative_path"]
        if descriptor_path.is_symlink() or (
            descriptor_path.exists() and not descriptor_path.is_file()
        ):
            raise ReservationConflictError("terminal descriptor path is unsafe")
        if descriptor_path.is_file():
            terminal = ReservedArtifactTransaction.load(
                descriptor_path,
                allocation_adapter=allocation_adapter,
                allocation_counter=allocation_counter,
            )
        else:
            terminal = ReservedArtifactTransaction.create(
                job_root=job_root,
                index_relative_path=JOB_CONTROL_RELATIVE_PATH,
                target_relative_path=SETTINGS_ARTIFACT_RELATIVE_PATH,
                reservation_kind="job_terminal_settings",
                payload_raw_max=512 * 1024,
                allocation_unit=entry["allocation_unit"],
                allocation_adapter=allocation_adapter,
                allocation_counter=allocation_counter,
                generation=entry["reservation_generation"],
                reservation_id=entry["reservation_id"],
                expected_publication_method="replace_existing",
            )
        control = load_job_control(job_root)
        cls._validate_transaction_plan(
            terminal,
            entry=entry,
            job_id=control.job_id,
            settings_identity_fingerprint=control.settings_identity_fingerprint,
            settings_revision_at_creation=None,
            settings_raw_sha256_at_creation=None,
            index_relative_path=JOB_CONTROL_RELATIVE_PATH,
            index_fingerprint=control.fingerprint,
        )
        target_raw = target.encode()
        if terminal.validate_physical_phase(target_raw) != "full-source":
            raise ReservationConflictError("replacement terminal extent is not full-length")
        if classify_payload_frame(terminal.source_path.read_bytes(), target_raw) != "zero":
            raise ReservationConflictError("replacement terminal extent is not a zero frame")
        return terminal

    @staticmethod
    def _restart_terminal_entry(job_root: Path, target: ProcessingSettingsArtifactV5) -> dict:
        control = load_job_control(job_root)
        metadata = target.to_object()["metadata"]
        if (
            control.job_id != metadata["job_id"]
            or control.settings_identity_fingerprint != metadata["settings_identity_fingerprint"]
            or control.job_reservation_generation != metadata["job_reservation_generation"]
            or control.job_terminal_reservation_id != metadata["job_terminal_reservation_id"]
        ):
            raise ReservationConflictError("restart target differs from immutable job control")
        matches = [
            entry
            for entry in control.bootstrap_extent_entries
            if entry.get("reservation_kind") == "job_terminal_settings"
        ]
        if len(matches) != 1:
            raise ReservationConflictError("job control has no unique terminal extent plan")
        return dict(matches[0])

    @staticmethod
    def _validate_source_or_target(
        *,
        fixed_path: Path,
        intent: SettingsTransitionIntentV1,
        target: ProcessingSettingsArtifactV5,
    ) -> str:
        if fixed_path.is_symlink():
            raise ReservationConflictError("fixed settings target is unsafe")
        if not fixed_path.exists():
            return "gap"
        if not fixed_path.is_file():
            raise ReservationConflictError("fixed settings target is unsafe")
        raw = _read_bounded_regular_file(
            fixed_path,
            max_bytes=SETTINGS_ARTIFACT_MAX_RAW_BYTES,
        )
        target_raw = target.encode()
        if raw == target_raw:
            return "target"
        if hashlib.sha256(raw).hexdigest() != intent.source_settings_raw_sha256:
            raise ReservationConflictError("fixed settings bytes match neither transition endpoint")
        try:
            from .settings_artifact import ProcessingSettingsArtifactV5

            current = ProcessingSettingsArtifactV5.parse(raw)
        except Exception as error:
            raise ReservationConflictError("preflight settings artifact is invalid") from error
        recreated = SettingsTransitionIntentV1.create(
            job_root=fixed_path.parent,
            current=current,
            target=target,
            reservation_generation=intent.reservation_generation,
            reservation_id=intent.reservation_id,
            allocation_unit=intent.allocation_unit,
            replacement_terminal_entry=intent.replacement_terminal_entry,
        )
        if recreated != intent:
            raise ReservationConflictError("settings transition no longer matches its preflight")
        return "source"

    @staticmethod
    def _consume_or_recover(
        transaction: ReservedArtifactTransaction,
        target_raw: bytes,
        *,
        crash_at: str | None = None,
    ) -> None:
        phase = transaction.validate_physical_phase(target_raw)
        if phase == "full-source":
            raw = transaction.source_path.read_bytes()
            state = classify_payload_frame(raw, target_raw)
            if state == "complete-full":
                transaction.recover(target_raw)
                return
            nul = raw.find(b"\x00")
            candidate = raw[:nul] if nul > 0 and not any(raw[nul:]) else None
            if candidate is not None:
                try:
                    from .settings_artifact import ProcessingSettingsArtifactV5

                    parsed = ProcessingSettingsArtifactV5.parse(candidate)
                except Exception:
                    parsed = None
                if parsed is not None and parsed.encode() == candidate:
                    raise ReservationConflictError(
                        "settings reservation contains a different complete intent"
                    )
            if any(raw[len(target_raw) :]):
                raise ReservationConflictError(
                    "settings reservation has nonzero bytes beyond the intended payload"
                )
            transaction.consume(target_raw, crash_at=crash_at)
            return
        transaction.recover(target_raw)

    @classmethod
    def begin(
        cls,
        *,
        job_root: Path,
        current: ProcessingSettingsArtifactV5,
        target: ProcessingSettingsArtifactV5,
        allocation_adapter: AllocationAdapter,
        allocation_unit: int,
        reservation_generation: str,
        reservation_id: str,
        allocation_counter: AllocationCounter | None = None,
    ) -> SettingsArtifactTransactionV1:
        job_root = Path(job_root)
        fixed_path = job_root / SETTINGS_ARTIFACT_RELATIVE_PATH
        if not fixed_path.is_file() or fixed_path.is_symlink():
            raise ReservationConflictError("fixed settings target is missing or unsafe")
        if _read_bounded_regular_file(
            fixed_path,
            max_bytes=SETTINGS_ARTIFACT_MAX_RAW_BYTES,
        ) != current.encode():
            raise ReservationConflictError("current settings bytes changed before transaction")
        attempt_restart = (
            current.status in {"completed", "failed"} and target.status == "in_progress"
        )
        terminal_entry = cls._restart_terminal_entry(job_root, target) if attempt_restart else None
        intent = SettingsTransitionIntentV1.create(
            job_root=job_root,
            current=current,
            target=target,
            reservation_generation=reservation_generation,
            reservation_id=reservation_id,
            allocation_unit=allocation_unit,
            replacement_terminal_entry=terminal_entry,
        )
        index_path = job_root / SETTINGS_TRANSITION_INDEX_RELATIVE_PATH
        publish_write_ahead_intent(
            index_path,
            intent.to_object(),
            expected_parent_identity=_directory_identity(index_path.parent),
        )
        transaction = cls._load_or_create_primary(
            job_root=job_root,
            intent=intent,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
        )
        replacement_terminal = cls._load_or_create_replacement_terminal(
            job_root=job_root,
            intent=intent,
            target=target,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
        )
        return cls(
            job_root=job_root,
            intent=intent,
            transaction=transaction,
            replacement_terminal=replacement_terminal,
        )

    @staticmethod
    def _retire_index(
        index_path: Path,
        intent: SettingsTransitionIntentV1,
    ) -> None:
        raw = canonical_json_bytes(intent.to_object())
        _unlink_exact_control_artifact(index_path, raw)
        temporary = index_path.with_name(f".{index_path.name}.create-new.tmp")
        _unlink_exact_control_artifact(temporary, raw)

    def commit(self, *, crash_at: str | None = None) -> ProcessingSettingsArtifactV5:
        from .settings_artifact import ProcessingSettingsArtifactV5

        target = ProcessingSettingsArtifactV5.from_object(self.intent.target_settings_payload)
        fixed_path = self.job_root / SETTINGS_ARTIFACT_RELATIVE_PATH
        if (
            self._validate_source_or_target(
                fixed_path=fixed_path,
                intent=self.intent,
                target=target,
            )
            != "source"
        ):
            raise ReservationConflictError(
                "settings transaction target was committed before commit"
            )
        replacement_terminal = self._load_or_create_replacement_terminal(
            job_root=self.job_root,
            intent=self.intent,
            target=target,
            allocation_adapter=self.transaction.allocation_adapter,
            allocation_counter=self.transaction.allocation_counter,
        )
        nested_crash = (
            crash_at
            if crash_at
            in {
                "after-complete-full-sync",
                "after-truncate",
                "after-truncate-sync",
                "after-rename-parent-sync",
            }
            else None
        )
        self._consume_or_recover(self.transaction, target.encode(), crash_at=nested_crash)
        if crash_at == "after-target-commit":
            raise InjectedReservationCrash(crash_at)
        if _read_bounded_regular_file(
            fixed_path,
            max_bytes=SETTINGS_ARTIFACT_MAX_RAW_BYTES,
        ) != target.encode():
            raise ReservationConflictError("settings target changed after commit")
        if replacement_terminal is not None:
            self._load_or_create_replacement_terminal(
                job_root=self.job_root,
                intent=self.intent,
                target=target,
                allocation_adapter=self.transaction.allocation_adapter,
                allocation_counter=self.transaction.allocation_counter,
            )
        self._retire_index(self.index_path, self.intent)
        return target

    @classmethod
    def recover(
        cls,
        job_root: Path,
        *,
        allocation_adapter: AllocationAdapter,
        allocation_counter: AllocationCounter | None = None,
    ) -> ProcessingSettingsArtifactV5:
        from .settings_artifact import ProcessingSettingsArtifactV5

        job_root = Path(job_root)
        index_path = job_root / SETTINGS_TRANSITION_INDEX_RELATIVE_PATH
        def parse(raw: bytes) -> SettingsTransitionIntentV1:
            return SettingsTransitionIntentV1.from_object(
                strict_canonical_json_loads(raw, max_bytes=1024 * 1024)
            )

        def validate_before_publish(value: object) -> None:
            if not isinstance(value, SettingsTransitionIntentV1):
                raise ReservationConflictError("settings transition temporary has the wrong type")
            control = load_job_control(job_root)
            if (
                value.job_id != control.job_id
                or value.settings_identity_fingerprint
                != control.settings_identity_fingerprint
                or _directory_identity(job_root) != control.job_root_identity
            ):
                raise ReservationConflictError(
                    "settings transition temporary belongs to another job"
                )
            final_index = job_root / (
                ".depth-surge-reservations-v1/final-encoding-reservation-v1.json"
            )
            final_index_temporary = final_index.with_name(
                f".{final_index.name}.create-new.tmp"
            )
            if any(
                path.exists() or path.is_symlink()
                for path in (final_index, final_index_temporary)
            ):
                raise ReservationConflictError(
                    "settings candidate overlaps a final-encoding authority"
                )
            candidate_temporary = index_path.with_name(
                f".{index_path.name}.create-new.tmp"
            )
            assert_only_job_control_owned_reservations_v1(
                job_root,
                control,
                allowed_relative_paths=(
                    candidate_temporary.relative_to(job_root).as_posix(),
                ),
            )
            for entry in value.extent_entries:
                descriptor = job_root / entry["descriptor_relative_path"]
                descriptor_temporary = descriptor.with_name(
                    f".{descriptor.name}.create-new.tmp"
                )
                source = job_root / entry["source_extent_relative_path"]
                if any(
                    path.exists() or path.is_symlink()
                    for path in (descriptor, descriptor_temporary, source)
                ):
                    raise ReservationConflictError(
                        "settings candidate has downstream reservation artifacts"
                    )
            target_artifact = ProcessingSettingsArtifactV5.from_object(
                value.target_settings_payload
            )
            cls._validate_source_or_target(
                fixed_path=job_root / SETTINGS_ARTIFACT_RELATIVE_PATH,
                intent=value,
                target=target_artifact,
            )

        def validate_before_unlink() -> None:
            control = load_job_control(job_root)
            candidate_temporary = index_path.with_name(
                f".{index_path.name}.create-new.tmp"
            )
            assert_only_job_control_owned_reservations_v1(
                job_root,
                control,
                allowed_relative_paths=(
                    candidate_temporary.relative_to(job_root).as_posix(),
                ),
            )

        try:
            recovered = recover_control_artifact_pair_v1(
                index_path,
                max_bytes=1024 * 1024,
                parser=parse,
                prepublish_validate=validate_before_publish,
                preunlink_validate=validate_before_unlink,
            )
        except (OSError, CanonicalJsonError) as error:
            raise ReservationConflictError(
                "settings transition index is missing or malformed"
            ) from error
        if recovered is None:
            fixed = job_root / SETTINGS_ARTIFACT_RELATIVE_PATH
            current = ProcessingSettingsArtifactV5.parse(
                _read_bounded_regular_file(fixed, max_bytes=512 * 1024)
            )
            control = load_job_control(job_root)
            metadata = current.to_object()["metadata"]
            if (
                metadata["job_id"] != control.job_id
                or metadata["settings_identity_fingerprint"]
                != control.settings_identity_fingerprint
            ):
                raise ReservationConflictError(
                    "settings target differs after discarding an incomplete index"
                )
            return current
        if not isinstance(recovered, SettingsTransitionIntentV1):
            raise ReservationConflictError(
                "settings transition index is missing or malformed"
            )
        intent = recovered
        assert_index_owned_reservation_namespace_v1(
            job_root,
            load_job_control(job_root),
            index_relative_path=SETTINGS_TRANSITION_INDEX_RELATIVE_PATH.as_posix(),
            extent_entries=intent.extent_entries,
        )
        target = ProcessingSettingsArtifactV5.from_object(intent.target_settings_payload)
        fixed_path = job_root / SETTINGS_ARTIFACT_RELATIVE_PATH
        endpoint = cls._validate_source_or_target(
            fixed_path=fixed_path,
            intent=intent,
            target=target,
        )
        cls._load_or_create_replacement_terminal(
            job_root=job_root,
            intent=intent,
            target=target,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
        )
        descriptor_path = job_root / intent.extent_entries[0]["descriptor_relative_path"]
        if endpoint == "gap" and not descriptor_path.is_file():
            raise ReservationConflictError(
                "settings replacement gap has no authoritative descriptor"
            )
        if descriptor_path.is_file() or endpoint in {"source", "gap"}:
            transaction = cls._load_or_create_primary(
                job_root=job_root,
                intent=intent,
                allocation_adapter=allocation_adapter,
                allocation_counter=allocation_counter,
            )
            cls._consume_or_recover(transaction, target.encode())
        if _read_bounded_regular_file(
            fixed_path,
            max_bytes=SETTINGS_ARTIFACT_MAX_RAW_BYTES,
        ) != target.encode():
            raise ReservationConflictError("settings transition did not commit its exact target")
        cls._load_or_create_replacement_terminal(
            job_root=job_root,
            intent=intent,
            target=target,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
        )
        cls._retire_index(index_path, intent)
        return target


def _settings_descriptor_path(
    job_root: Path,
    *,
    generation: str,
    reservation_kind: str,
    reservation_id: str,
) -> Path:
    return (
        Path(job_root)
        / ".depth-surge-reservations-v1"
        / f"d-p-none-r-{generation}-{reservation_kind}-{reservation_id}.json"
    )


def _bootstrap_settings_intent(
    artifact: ProcessingSettingsArtifactV5,
    *,
    reservation_generation: str,
) -> dict:
    value = artifact.to_object()
    raw = artifact.encode()
    return {
        "algorithm_version": "settings-bootstrap-intent-v1",
        "settings_identity_fingerprint": value["metadata"]["settings_identity_fingerprint"],
        "reservation_generation": reservation_generation,
        "target_settings_revision": 0,
        "target_settings_content_fingerprint": value["content_fingerprint"],
        "target_settings_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "target_settings_payload": value,
    }


def _bootstrap_entry(
    *,
    job_root: Path,
    reservation_kind: str,
    generation: str,
    reservation_id: str,
    publication_method: PublicationMethod,
    allocation_unit: int,
) -> dict:
    target = SETTINGS_ARTIFACT_RELATIVE_PATH
    source = target.parent / (
        f".depth-surge-r-{generation}-{reservation_kind}-{reservation_id}.reserved"
    )
    descriptor = Path(".depth-surge-reservations-v1") / (
        f"d-p-none-r-{generation}-{reservation_kind}-{reservation_id}.json"
    )
    payload_role = (
        "settings_initial"
        if reservation_kind == "initial_settings"
        else "settings_migration"
        if reservation_kind == "migration_settings"
        else "settings_terminal"
    )
    return validate_reservation_entry_v1({
        "reservation_generation": generation,
        "reservation_id": reservation_id,
        "reservation_kind": reservation_kind,
        "descriptor_publication_generation": None,
        "descriptor_relative_path": descriptor.as_posix(),
        "source_extent_relative_path": source.as_posix(),
        "temporary_relative_path": source.as_posix(),
        "target_relative_path": target.as_posix(),
        "target_parent_identity": _directory_identity(Path(job_root)).to_object(),
        "publication_method": publication_method,
        "existing_target_identity_at_preflight": None,
        "existing_target_byte_count_at_preflight": None,
        "existing_target_raw_sha256_at_preflight": None,
        "payload_role": payload_role,
        "payload_raw_max": 512 * 1024,
        "logical_byte_count": payload_logical_bytes(512 * 1024, allocation_unit),
        "allocation_unit": allocation_unit,
        "prune_root_relative_path": None,
        "marker_name": None,
        "marker_payload_sha256": None,
    })


def bootstrap_settings_job_v1(  # noqa: C901
    *,
    job_root: Path,
    artifact: ProcessingSettingsArtifactV5,
    initial_settings_reservation_id: str,
    allocation_adapter: AllocationAdapter,
    allocation_unit: int,
    allocation_counter: AllocationCounter | None = None,
    legacy_settings_artifact_relative_path: str | None = None,
    event: Callable[[str], None] | None = None,
) -> JobControlV1:
    """Publish the immutable locator before reserving initial and terminal settings."""

    from .settings_artifact import ProcessingSettingsArtifactV5

    if not isinstance(artifact, ProcessingSettingsArtifactV5):
        raise TypeError("artifact must be ProcessingSettingsArtifactV5")
    job_root = Path(job_root)
    job_root.mkdir(parents=True, exist_ok=True)
    if (job_root / JOB_CONTROL_RELATIVE_PATH).exists():
        raise ReservationConflictError("job control already exists")
    if (job_root / SETTINGS_ARTIFACT_RELATIVE_PATH).exists():
        raise ReservationConflictError("fixed settings target already exists")
    value = artifact.to_object()
    metadata = value["metadata"]
    if artifact.revision != 0 or artifact.status not in {"in_progress", "completed", "failed"}:
        raise ReservationConflictError("bootstrap settings must be revision zero")
    generation = metadata["job_reservation_generation"]
    terminal_id = metadata["job_terminal_reservation_id"]
    initial_kind = (
        "migration_settings"
        if legacy_settings_artifact_relative_path is not None
        else "initial_settings"
    )
    initial_entry = _bootstrap_entry(
        job_root=job_root,
        reservation_kind=initial_kind,
        generation=generation,
        reservation_id=initial_settings_reservation_id,
        publication_method="replace_placeholder",
        allocation_unit=allocation_unit,
    )
    terminal_entry = _bootstrap_entry(
        job_root=job_root,
        reservation_kind="job_terminal_settings",
        generation=generation,
        reservation_id=terminal_id,
        publication_method="replace_existing",
        allocation_unit=allocation_unit,
    )
    control = JobControlV1.create(
        job_root=job_root,
        job_id=metadata["job_id"],
        settings_identity_fingerprint=metadata["settings_identity_fingerprint"],
        job_reservation_generation=generation,
        initial_settings_reservation_id=initial_settings_reservation_id,
        job_terminal_reservation_id=terminal_id,
        legacy_settings_artifact_relative_path=legacy_settings_artifact_relative_path,
        bootstrap_extent_entries=[initial_entry, terminal_entry],
        bootstrap_settings_intent=_bootstrap_settings_intent(
            artifact,
            reservation_generation=generation,
        ),
    )
    publish_job_control(job_root, control)
    if event is not None:
        event("job-control:published")
    initial = ReservedArtifactTransaction.create(
        job_root=job_root,
        index_relative_path=JOB_CONTROL_RELATIVE_PATH,
        target_relative_path=SETTINGS_ARTIFACT_RELATIVE_PATH,
        reservation_kind=initial_kind,
        payload_raw_max=512 * 1024,
        allocation_unit=allocation_unit,
        allocation_adapter=allocation_adapter,
        allocation_counter=allocation_counter,
        generation=generation,
        reservation_id=initial_settings_reservation_id,
    )
    if event is not None:
        event("initial-settings:ready")
    terminal = ReservedArtifactTransaction.create(
        job_root=job_root,
        index_relative_path=JOB_CONTROL_RELATIVE_PATH,
        target_relative_path=SETTINGS_ARTIFACT_RELATIVE_PATH,
        reservation_kind="job_terminal_settings",
        payload_raw_max=512 * 1024,
        allocation_unit=allocation_unit,
        allocation_adapter=allocation_adapter,
        allocation_counter=allocation_counter,
        generation=generation,
        reservation_id=terminal_id,
    )
    if event is not None:
        event("terminal-settings:ready")
    initial.consume(artifact.encode())
    if event is not None:
        event("initial-settings:committed")
    if artifact.status == "in_progress":
        if terminal.validate_physical_phase(artifact.encode()) != "full-source":
            raise ReservationConflictError("terminal settings extent is not retained full-length")
        if classify_payload_frame(terminal.source_path.read_bytes(), artifact.encode()) != "zero":
            raise ReservationConflictError("terminal settings extent is not a zero frame")
    else:
        _unlink_safe_control_temporary(
            terminal.source_path,
            expected_identity=terminal.descriptor.filesystem_identity,
        )
        terminal._retire_descriptor()
    return control


def _parse_bootstrap_target(control: JobControlV1) -> ProcessingSettingsArtifactV5:
    from .settings_artifact import ProcessingSettingsArtifactV5

    intent = control.bootstrap_settings_intent
    expected = {
        "algorithm_version",
        "settings_identity_fingerprint",
        "reservation_generation",
        "target_settings_revision",
        "target_settings_content_fingerprint",
        "target_settings_raw_sha256",
        "target_settings_payload",
    }
    if type(intent) is not dict or set(intent) != expected:
        raise ReservationConflictError("bootstrap settings intent schema is invalid")
    if (
        intent["algorithm_version"] != "settings-bootstrap-intent-v1"
        or intent["settings_identity_fingerprint"] != control.settings_identity_fingerprint
        or intent["reservation_generation"] != control.job_reservation_generation
        or intent["target_settings_revision"] != 0
        or type(intent["target_settings_revision"]) is not int
    ):
        raise ReservationConflictError("bootstrap settings intent identity is invalid")
    _require_lower_hex(
        intent["settings_identity_fingerprint"], 64, "bootstrap settings identity"
    )
    _require_lower_hex(intent["reservation_generation"], 32, "bootstrap generation")
    _require_lower_hex(
        intent["target_settings_content_fingerprint"],
        64,
        "bootstrap content fingerprint",
    )
    _require_lower_hex(
        intent["target_settings_raw_sha256"], 64, "bootstrap settings raw hash"
    )
    artifact = ProcessingSettingsArtifactV5.from_object(intent["target_settings_payload"])
    value = artifact.to_object()
    if (
        artifact.revision != 0
        or value["metadata"]["job_id"] != control.job_id
        or value["metadata"]["settings_identity_fingerprint"]
        != control.settings_identity_fingerprint
        or value["metadata"]["job_reservation_generation"]
        != control.job_reservation_generation
        or value["metadata"]["job_terminal_reservation_id"]
        != control.job_terminal_reservation_id
        or value["content_fingerprint"] != intent["target_settings_content_fingerprint"]
        or hashlib.sha256(artifact.encode()).hexdigest() != intent["target_settings_raw_sha256"]
    ):
        raise ReservationConflictError("bootstrap target hash or identity differs")
    return artifact


def _terminal_settings_delta_is_legal(
    current: ProcessingSettingsArtifactV5,
    target: ProcessingSettingsArtifactV5,
) -> bool:
    before = current.to_object()
    after = target.to_object()
    if target.revision != current.revision + 1 or target.status not in {"completed", "failed"}:
        return False
    if any(
        before[key] != after[key]
        for key in ("video_properties", "processing_settings", "output_info")
    ):
        return False
    mutable_metadata = {
        "settings_revision",
        "last_updated_at",
        "terminal_at",
        "processing_duration_ms",
        "processing_status",
        "cleanup_status",
    }
    return all(
        before["metadata"][key] == after["metadata"][key]
        for key in before["metadata"]
        if key not in mutable_metadata
    )


def commit_terminal_settings_v1(
    *,
    job_root: Path,
    current: ProcessingSettingsArtifactV5,
    target: ProcessingSettingsArtifactV5,
    allocation_adapter: AllocationAdapter,
    allocation_counter: AllocationCounter | None = None,
    crash_at: str | None = None,
) -> ProcessingSettingsArtifactV5:
    """Consume the lifecycle terminal extent for one completed/failed revision."""

    job_root = Path(job_root)
    control = load_job_control(job_root)
    fixed_path = resolve_fixed_settings_path(job_root, control)
    if _read_bounded_regular_file(
        fixed_path,
        max_bytes=SETTINGS_ARTIFACT_MAX_RAW_BYTES,
    ) != current.encode():
        raise ReservationConflictError("terminal transition current settings changed")
    if not _terminal_settings_delta_is_legal(current, target):
        raise ReservationConflictError("terminal settings delta is not the unique legal revision")
    descriptor_path = _settings_descriptor_path(
        job_root,
        generation=control.job_reservation_generation,
        reservation_kind="job_terminal_settings",
        reservation_id=control.job_terminal_reservation_id,
    )
    terminal = ReservedArtifactTransaction.load(
        descriptor_path,
        allocation_adapter=allocation_adapter,
        allocation_counter=allocation_counter,
    )
    terminal.consume(target.encode(), crash_at=crash_at)
    return target


def _reset_terminal_extent(transaction: ReservedArtifactTransaction) -> None:
    with transaction.source_path.open("r+b") as handle:
        handle.seek(0)
        remaining = transaction.descriptor.logical_byte_count
        zeroes = b"\x00" * RESERVATION_ZERO_FILL_CHUNK_BYTES
        while remaining:
            chunk = zeroes[: min(remaining, len(zeroes))]
            if handle.write(chunk) != len(chunk):
                raise OSError("short terminal reset write")
            remaining -= len(chunk)
        handle.flush()
        os.fsync(handle.fileno())


def _parse_terminal_source_payload(
    raw: bytes,
    *,
    full_length: bool,
) -> ProcessingSettingsArtifactV5 | None:
    from .settings_artifact import ProcessingSettingsArtifactV5, SettingsArtifactError

    if full_length:
        if raw and all(byte == 0 for byte in raw):
            return None
        nul = raw.find(b"\x00")
        if nul <= 0 or any(raw[nul:]):
            raise ReservationIncompleteError("terminal full source has a partial frame")
        raw = raw[:nul]
    try:
        return ProcessingSettingsArtifactV5.parse(raw)
    except SettingsArtifactError as error:
        if full_length:
            raise ReservationIncompleteError("terminal full source has a partial frame") from error
        raise ReservationConflictError(
            "terminal short source is not a canonical artifact"
        ) from error


def _load_or_create_job_control_extent(
    *,
    job_root: Path,
    control: JobControlV1,
    entry: dict,
    allocation_adapter: AllocationAdapter,
    allocation_counter: AllocationCounter | None,
) -> ReservedArtifactTransaction:
    descriptor_path = Path(job_root) / entry["descriptor_relative_path"]
    if descriptor_path.is_symlink() or (
        descriptor_path.exists() and not descriptor_path.is_file()
    ):
        raise ReservationConflictError("job-control descriptor path is unsafe")
    if descriptor_path.is_file():
        transaction = ReservedArtifactTransaction.load(
            descriptor_path,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
        )
    else:
        transaction = ReservedArtifactTransaction.create(
            job_root=job_root,
            index_relative_path=JOB_CONTROL_RELATIVE_PATH,
            target_relative_path=Path(entry["target_relative_path"]),
            reservation_kind=entry["reservation_kind"],
            payload_raw_max=entry["payload_raw_max"],
            allocation_unit=entry["allocation_unit"],
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
            generation=entry["reservation_generation"],
            reservation_id=entry["reservation_id"],
            expected_publication_method=entry["publication_method"],
        )
    validate_descriptor_against_entry_v1(
        transaction.descriptor,
        entry,
        job_id=control.job_id,
        settings_identity_fingerprint=control.settings_identity_fingerprint,
        settings_revision_at_creation=None,
        settings_raw_sha256_at_creation=None,
        transaction_index_relative_path=JOB_CONTROL_RELATIVE_PATH,
        transaction_index_fingerprint=control.fingerprint,
    )
    return transaction


def recover_settings_job_v1(  # noqa: C901
    job_root: Path,
    *,
    allocation_adapter: AllocationAdapter,
    allocation_counter: AllocationCounter | None = None,
) -> ProcessingSettingsArtifactV5:
    """Recover settings control work before any ordinary artifact audit."""

    from .settings_artifact import ProcessingSettingsArtifactV5

    job_root = Path(job_root)
    control = load_job_control(job_root)
    bootstrap_target = _parse_bootstrap_target(control)
    fixed_path = job_root / SETTINGS_ARTIFACT_RELATIVE_PATH
    if fixed_path.is_symlink() or (fixed_path.exists() and not fixed_path.is_file()):
        raise ReservationConflictError("fixed settings target is unsafe during bootstrap")
    current: ProcessingSettingsArtifactV5 | None = None
    if fixed_path.is_file() and fixed_path.stat().st_size:
        try:
            current = ProcessingSettingsArtifactV5.parse(
                _read_bounded_regular_file(
                    fixed_path,
                    max_bytes=SETTINGS_ARTIFACT_MAX_RAW_BYTES,
                )
            )
        except Exception as error:
            raise ReservationConflictError("fixed settings target is malformed") from error
    initial_entry, terminal_entry = control.bootstrap_extent_entries
    initial_descriptor = job_root / initial_entry["descriptor_relative_path"]
    if current is None:
        initial = _load_or_create_job_control_extent(
            job_root=job_root,
            control=control,
            entry=initial_entry,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
        )
        initial_phase = initial.validate_physical_phase(bootstrap_target.encode())
        if initial_phase == "replacement-gap":
            initial.recover(bootstrap_target.encode())
            current = bootstrap_target
        terminal = _load_or_create_job_control_extent(
            job_root=job_root,
            control=control,
            entry=terminal_entry,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
        )
        terminal_raw = terminal.source_path.read_bytes()
        if (
            terminal.validate_physical_phase(bootstrap_target.encode()) != "full-source"
            or classify_payload_frame(terminal_raw, bootstrap_target.encode()) != "zero"
        ):
            raise ReservationConflictError(
                "bootstrap terminal reserve is not a complete zero frame"
            )
        if current is None:
            SettingsArtifactTransactionV1._consume_or_recover(
                initial,
                bootstrap_target.encode(),
            )
            current = bootstrap_target
    elif initial_descriptor.is_file():
        initial = _load_or_create_job_control_extent(
            job_root=job_root,
            control=control,
            entry=initial_entry,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
        )
        if current == bootstrap_target:
            initial.recover(bootstrap_target.encode())
        elif initial.source_path.exists():
            raise ReservationConflictError(
                "later settings revision retains an initial source extent"
            )
        else:
            initial._retire_descriptor()
    if not fixed_path.is_file() or fixed_path.is_symlink():
        raise ReservationConflictError("fixed settings target is missing after bootstrap")
    settings_index_path = job_root / SETTINGS_TRANSITION_INDEX_RELATIVE_PATH
    settings_index_temporary = settings_index_path.with_name(
        f".{settings_index_path.name}.create-new.tmp"
    )
    if any(
        path.exists() or path.is_symlink()
        for path in (settings_index_path, settings_index_temporary)
    ):
        current = SettingsArtifactTransactionV1.recover(
            job_root,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
        )
    else:
        current = ProcessingSettingsArtifactV5.parse(
            _read_bounded_regular_file(
                fixed_path,
                max_bytes=SETTINGS_ARTIFACT_MAX_RAW_BYTES,
            )
        )
    current_value = current.to_object()
    if (
        current_value["metadata"]["job_id"] != control.job_id
        or current_value["metadata"]["settings_identity_fingerprint"]
        != control.settings_identity_fingerprint
    ):
        raise ReservationConflictError("current settings do not belong to job control")
    terminal_descriptor = job_root / terminal_entry["descriptor_relative_path"]
    if terminal_descriptor.is_file() or current.status == "in_progress":
        terminal = _load_or_create_job_control_extent(
            job_root=job_root,
            control=control,
            entry=terminal_entry,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
        )
        if terminal.source_path.exists():
            if _identity(terminal.source_path) != terminal.descriptor.filesystem_identity:
                raise ReservationConflictError("terminal extent identity changed")
            size = terminal.source_path.stat().st_size
            if size == terminal.descriptor.logical_byte_count:
                evidence = allocation_adapter(terminal.source_path, size)
                if evidence != terminal.descriptor.allocation_evidence_at_readiness:
                    raise ReservationEvidenceConflictError("terminal readiness evidence changed")
                try:
                    intended = _parse_terminal_source_payload(
                        terminal.source_path.read_bytes(),
                        full_length=True,
                    )
                except ReservationIncompleteError:
                    _reset_terminal_extent(terminal)
                    intended = None
                if intended is not None:
                    if not _terminal_settings_delta_is_legal(current, intended):
                        raise ReservationConflictError(
                            "terminal extent carries an illegal semantic intent"
                        )
                    terminal.recover(intended.encode())
                    current = intended
            else:
                intended = _parse_terminal_source_payload(
                    terminal.source_path.read_bytes(),
                    full_length=False,
                )
                assert intended is not None
                if not _terminal_settings_delta_is_legal(current, intended):
                    raise ReservationConflictError(
                        "terminal short source carries an illegal semantic intent"
                    )
                terminal.recover(intended.encode())
                current = intended
        elif current.status in {"completed", "failed"}:
            terminal.recover(current.encode())
        else:
            raise ReservationConflictError(
                "in-progress job lost the terminal source before target commit"
            )
    elif current.status == "in_progress":
        raise ReservationConflictError("in-progress job lost its terminal settings reserve")
    return current
