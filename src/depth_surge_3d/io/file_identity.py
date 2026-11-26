"""Restart-stable file identities used by reservations and destructive cleanup."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
from functools import lru_cache
import os
from pathlib import Path
import re
import stat
import sys
from typing import BinaryIO, Callable, Literal, TypeAlias


from . import _win32

LINUX_UUID_DIRECTORY = "/dev/disk/by-uuid"
LINUX_UUID_ENTRY_CAP = 4096
LINUX_UUID_SCAN_BUFFER_BYTES = 64 * 1024
_UUID_NAME = re.compile(
    rb"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
)
_DEVICE_COMPONENT = re.compile(rb"[A-Za-z0-9._+-]+")
_LOWER_HEX = re.compile(r"[0-9a-f]+", flags=re.ASCII)


class PersistentIdentityCapabilityError(RuntimeError):
    """The platform cannot produce the frozen restart-stable identity."""


class PersistentIdentitySchemaError(ValueError):
    """A persisted identity does not match its closed schema."""


def _closed_dict(value: object, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise PersistentIdentitySchemaError(f"{label} schema is invalid")
    return value


@dataclass(frozen=True)
class PosixFileIdentityV1:
    filesystem_uuid: str
    handle_type: int
    file_handle_hex: str

    def __post_init__(self) -> None:
        if (
            len(self.filesystem_uuid) != 32
            or _LOWER_HEX.fullmatch(self.filesystem_uuid) is None
            or isinstance(self.handle_type, bool)
            or not -(1 << 31) <= self.handle_type < (1 << 31)
            or not 2 <= len(self.file_handle_hex) <= 256
            or len(self.file_handle_hex) % 2
            or _LOWER_HEX.fullmatch(self.file_handle_hex) is None
        ):
            raise PersistentIdentitySchemaError("POSIX file identity is invalid")

    def to_object(self) -> dict[str, str | int]:
        return {
            "filesystem_uuid": self.filesystem_uuid,
            "handle_type": self.handle_type,
            "file_handle_hex": self.file_handle_hex,
        }

    @classmethod
    def from_object(cls, value: object) -> PosixFileIdentityV1:
        item = _closed_dict(
            value,
            {"filesystem_uuid", "handle_type", "file_handle_hex"},
            "POSIX file identity",
        )
        if type(item["handle_type"]) is not int:
            raise PersistentIdentitySchemaError("POSIX handle type is invalid")
        if type(item["filesystem_uuid"]) is not str or type(item["file_handle_hex"]) is not str:
            raise PersistentIdentitySchemaError("POSIX identity strings are invalid")
        return cls(
            item["filesystem_uuid"],
            item["handle_type"],
            item["file_handle_hex"],
        )


@dataclass(frozen=True)
class WindowsFileIdentityV1:
    volume_serial: str
    file_id: str

    def __post_init__(self) -> None:
        if (
            len(self.volume_serial) != 16
            or _LOWER_HEX.fullmatch(self.volume_serial) is None
            or len(self.file_id) != 32
            or _LOWER_HEX.fullmatch(self.file_id) is None
        ):
            raise PersistentIdentitySchemaError("Windows file identity is invalid")

    def to_object(self) -> dict[str, str]:
        return {"volume_serial": self.volume_serial, "file_id": self.file_id}

    @classmethod
    def from_object(cls, value: object) -> WindowsFileIdentityV1:
        item = _closed_dict(value, {"volume_serial", "file_id"}, "Windows file identity")
        if type(item["volume_serial"]) is not str or type(item["file_id"]) is not str:
            raise PersistentIdentitySchemaError("Windows identity strings are invalid")
        return cls(item["volume_serial"], item["file_id"])


BaseFileIdentity: TypeAlias = PosixFileIdentityV1 | WindowsFileIdentityV1


@dataclass(frozen=True)
class PersistedFileIdentityV1:
    platform: Literal["posix", "windows"]
    file_identity: BaseFileIdentity
    link_count: int

    def __post_init__(self) -> None:
        expected = PosixFileIdentityV1 if self.platform == "posix" else WindowsFileIdentityV1
        if self.platform not in {"posix", "windows"} or not isinstance(
            self.file_identity, expected
        ):
            raise PersistentIdentitySchemaError("file identity platform is invalid")
        if isinstance(self.link_count, bool) or not 0 < self.link_count < (1 << 64):
            raise PersistentIdentitySchemaError("file identity link count is invalid")

    @property
    def volume_key(self) -> tuple[str, str]:
        if isinstance(self.file_identity, PosixFileIdentityV1):
            return self.platform, self.file_identity.filesystem_uuid
        return self.platform, self.file_identity.volume_serial

    def to_object(self) -> dict:
        return {
            "platform": self.platform,
            "file_identity": self.file_identity.to_object(),
            "link_count": self.link_count,
        }

    @classmethod
    def from_object(cls, value: object) -> PersistedFileIdentityV1:
        item = _closed_dict(value, {"platform", "file_identity", "link_count"}, "file identity")
        platform = item["platform"]
        if platform == "posix":
            inner: BaseFileIdentity = PosixFileIdentityV1.from_object(item["file_identity"])
        elif platform == "windows":
            inner = WindowsFileIdentityV1.from_object(item["file_identity"])
        else:
            raise PersistentIdentitySchemaError("file identity platform is invalid")
        if type(item["link_count"]) is not int:
            raise PersistentIdentitySchemaError("file identity link count is invalid")
        return cls(platform, inner, item["link_count"])


@dataclass(frozen=True)
class DirectoryIdentityV1:
    kind: Literal["posix", "windows"]
    file_identity: BaseFileIdentity

    def __post_init__(self) -> None:
        expected = PosixFileIdentityV1 if self.kind == "posix" else WindowsFileIdentityV1
        if self.kind not in {"posix", "windows"} or not isinstance(self.file_identity, expected):
            raise PersistentIdentitySchemaError("directory identity kind is invalid")

    @property
    def volume_key(self) -> tuple[str, str]:
        if isinstance(self.file_identity, PosixFileIdentityV1):
            return self.kind, self.file_identity.filesystem_uuid
        return self.kind, self.file_identity.volume_serial

    def to_object(self) -> dict:
        return {"kind": self.kind, "file_identity": self.file_identity.to_object()}

    @classmethod
    def from_object(cls, value: object) -> DirectoryIdentityV1:
        item = _closed_dict(value, {"kind", "file_identity"}, "directory identity")
        kind = item["kind"]
        if kind == "posix":
            inner: BaseFileIdentity = PosixFileIdentityV1.from_object(item["file_identity"])
        elif kind == "windows":
            inner = WindowsFileIdentityV1.from_object(item["file_identity"])
        else:
            raise PersistentIdentitySchemaError("directory identity kind is invalid")
        return cls(kind, inner)


@lru_cache(maxsize=1)
def _windows_identity_bindings() -> tuple[
    type[ctypes.Structure],
    type[ctypes.Structure],
    type[ctypes.Structure],
    Callable[..., int],
]:
    """Cache ABI declarations only; each identity query keeps fresh result buffers."""
    from ctypes import wintypes

    class FileId128(ctypes.Structure):
        _fields_ = [("identifier", ctypes.c_ubyte * 16)]

    class FileIdInfo(ctypes.Structure):
        _fields_ = [("volume_serial", ctypes.c_ulonglong), ("file_id", FileId128)]

    class FileStandardInfo(ctypes.Structure):
        _fields_ = [
            ("allocation_size", ctypes.c_longlong),
            ("end_of_file", ctypes.c_longlong),
            ("number_of_links", wintypes.DWORD),
            ("delete_pending", wintypes.BOOLEAN),
            ("directory", wintypes.BOOLEAN),
        ]

    class FileAttributeTagInfo(ctypes.Structure):
        _fields_ = [
            ("file_attributes", wintypes.DWORD),
            ("reparse_tag", wintypes.DWORD),
        ]

    get_info = _win32.load_library("kernel32", use_last_error=True).GetFileInformationByHandleEx
    get_info.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    get_info.restype = wintypes.BOOL
    return FileIdInfo, FileStandardInfo, FileAttributeTagInfo, get_info


def _windows_handle_identity(
    handle: int,
    *,
    require_directory: bool,
    allow_reparse: bool = False,
) -> tuple[WindowsFileIdentityV1, int, int]:
    if os.name != "nt":
        raise PersistentIdentityCapabilityError("Windows FileIdInfo is unavailable")
    from ctypes import wintypes

    FileIdInfo, FileStandardInfo, FileAttributeTagInfo, get_info = _windows_identity_bindings()

    def query(info_class: int, target: ctypes.Structure, label: str) -> None:
        if not get_info(
            wintypes.HANDLE(handle),
            info_class,
            ctypes.byref(target),
            ctypes.sizeof(target),
        ):
            code = _win32.last_error()
            raise PersistentIdentityCapabilityError(f"{label} failed: Windows error {code}")

    file_id = FileIdInfo()
    standard = FileStandardInfo()
    attributes = FileAttributeTagInfo()
    query(18, file_id, "GetFileInformationByHandleEx(FileIdInfo)")
    query(1, standard, "GetFileInformationByHandleEx(FileStandardInfo)")
    query(9, attributes, "GetFileInformationByHandleEx(FileAttributeTagInfo)")
    if bool(standard.directory) != require_directory:
        raise PersistentIdentitySchemaError("identity handle type changed")
    if not allow_reparse and int(attributes.file_attributes) & 0x400:
        raise PersistentIdentitySchemaError("identity handle is a reparse point")
    link_count = int(standard.number_of_links)
    if link_count <= 0:
        raise PersistentIdentitySchemaError("identity handle has no links")
    identity = WindowsFileIdentityV1(
        f"{int(file_id.volume_serial):016x}",
        bytes(file_id.file_id.identifier).hex(),
    )
    return identity, link_count, int(attributes.file_attributes)


def windows_identity_from_handle(
    handle: int,
    *,
    require_directory: bool,
    allow_reparse: bool = False,
) -> tuple[WindowsFileIdentityV1, int, int]:
    return _windows_handle_identity(
        handle,
        require_directory=require_directory,
        allow_reparse=allow_reparse,
    )


def directory_identity_from_windows_handle(handle: int) -> DirectoryIdentityV1:
    identity, _links, _attributes = _windows_handle_identity(
        handle,
        require_directory=True,
    )
    return DirectoryIdentityV1("windows", identity)


def _open_windows_path(path: Path, *, directory: bool) -> int:
    from ctypes import wintypes

    kernel32 = _win32.load_library("kernel32", use_last_error=True)
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
    handle = create_file(
        str(path),
        0x00000080,
        0x1 | 0x2 | 0x4,
        None,
        3,
        0x00200000 | (0x02000000 if directory else 0),
        None,
    )
    invalid = wintypes.HANDLE(-1).value
    if handle == invalid:
        code = _win32.last_error()
        raise PersistentIdentityCapabilityError(f"CreateFileW(identity) failed: {code}")
    return int(handle)


class _StatxTimestamp(ctypes.Structure):
    _fields_ = [
        ("seconds", ctypes.c_int64),
        ("nanoseconds", ctypes.c_uint32),
        ("reserved", ctypes.c_int32),
    ]


class _Statx(ctypes.Structure):
    _fields_ = [
        ("mask", ctypes.c_uint32),
        ("block_size", ctypes.c_uint32),
        ("attributes", ctypes.c_uint64),
        ("link_count", ctypes.c_uint32),
        ("uid", ctypes.c_uint32),
        ("gid", ctypes.c_uint32),
        ("mode", ctypes.c_uint16),
        ("spare0", ctypes.c_uint16),
        ("inode", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
        ("blocks", ctypes.c_uint64),
        ("attributes_mask", ctypes.c_uint64),
        ("access_time", _StatxTimestamp),
        ("birth_time", _StatxTimestamp),
        ("change_time", _StatxTimestamp),
        ("modify_time", _StatxTimestamp),
        ("rdev_major", ctypes.c_uint32),
        ("rdev_minor", ctypes.c_uint32),
        ("dev_major", ctypes.c_uint32),
        ("dev_minor", ctypes.c_uint32),
        ("mount_id", ctypes.c_uint64),
        ("dio_mem_align", ctypes.c_uint32),
        ("dio_offset_align", ctypes.c_uint32),
        ("subvol", ctypes.c_uint64),
        ("atomic_write_unit_min", ctypes.c_uint32),
        ("atomic_write_unit_max", ctypes.c_uint32),
        ("atomic_write_segments_max", ctypes.c_uint32),
        ("spare1", ctypes.c_uint32),
        ("spare", ctypes.c_uint64 * 9),
    ]


def linux_statx_for_fd(descriptor: int, mask: int) -> _Statx:
    if not sys.platform.startswith("linux"):
        raise PersistentIdentityCapabilityError("Linux statx is unavailable")
    libc = ctypes.CDLL(None, use_errno=True)
    statx = getattr(libc, "statx", None)
    if statx is None:
        raise PersistentIdentityCapabilityError("libc does not expose statx")
    statx.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(_Statx),
    )
    statx.restype = ctypes.c_int
    result = _Statx()
    if statx(descriptor, b"", 0x1000 | 0x100, mask, ctypes.byref(result)) != 0:
        code = ctypes.get_errno()
        raise PersistentIdentityCapabilityError(f"statx identity failed: errno {code}")
    if result.mask & mask != mask:
        raise PersistentIdentityCapabilityError("statx omitted required identity fields")
    return result


def _uuid_matches_device(
    name: str, uuid_fd: int, dev_fd: int, nofollow: int, target_device: tuple[int, int]
) -> bool:
    link = os.readlink(name, dir_fd=uuid_fd).encode("ascii", "strict")
    if len(link) > 255 or not link.startswith(b"../../"):
        raise PersistentIdentityCapabilityError("Linux UUID link target is unsafe")
    component = link[6:]
    if b"/" in component or _DEVICE_COMPONENT.fullmatch(component) is None:
        raise PersistentIdentityCapabilityError("Linux UUID device component is unsafe")
    node_fd = os.open(
        component.decode("ascii"),
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow,
        dir_fd=dev_fd,
    )
    try:
        node = os.fstat(node_fd)
        if not stat.S_ISBLK(node.st_mode):
            raise PersistentIdentityCapabilityError("Linux UUID target is not block storage")
        if (
            os.major(node.st_rdev),
            os.minor(node.st_rdev),
        ) != target_device:
            return False
        if os.readlink(name, dir_fd=uuid_fd).encode("ascii", "strict") != link:
            raise PersistentIdentityCapabilityError("Linux UUID link changed during audit")
        node_after = os.fstat(node_fd)
        if (node.st_dev, node.st_ino, node.st_rdev) != (
            node_after.st_dev,
            node_after.st_ino,
            node_after.st_rdev,
        ):
            raise PersistentIdentityCapabilityError("Linux block identity changed")
        return True
    finally:
        os.close(node_fd)


def _matching_uuids(
    uuid_fd: int, dev_fd: int, nofollow: int, target_device: tuple[int, int]
) -> list[str]:
    matches: list[str] = []
    count = 0
    with os.scandir(uuid_fd) as entries:
        for entry in entries:
            if entry.name in {".", ".."}:
                continue
            count += 1
            if count > LINUX_UUID_ENTRY_CAP:
                raise PersistentIdentityCapabilityError("Linux UUID entry cap exceeded")
            try:
                name = entry.name.encode("ascii")
            except UnicodeEncodeError:
                continue
            if _UUID_NAME.fullmatch(name) is None:
                continue
            if _uuid_matches_device(entry.name, uuid_fd, dev_fd, nofollow, target_device):
                matches.append(name.decode("ascii").replace("-", "").lower())
    return matches


def _linux_filesystem_uuid(descriptor: int) -> str:
    statx = linux_statx_for_fd(descriptor, 0x00000001)
    target_device = (int(statx.dev_major), int(statx.dev_minor))
    before = os.fstat(descriptor)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    dev_fd = disk_fd = uuid_fd = -1
    try:
        dev_fd = os.open("/dev", flags | nofollow)
        disk_fd = os.open("disk", flags | nofollow, dir_fd=dev_fd)
        uuid_fd = os.open("by-uuid", flags | nofollow, dir_fd=disk_fd)
        matches = _matching_uuids(uuid_fd, dev_fd, nofollow, target_device)
    except (OSError, UnicodeError) as error:
        raise PersistentIdentityCapabilityError(
            "Linux filesystem UUID resolution failed"
        ) from error
    finally:
        for opened in (uuid_fd, disk_fd, dev_fd):
            if opened >= 0:
                os.close(opened)
    after = os.fstat(descriptor)
    if (before.st_dev, before.st_ino, before.st_mode) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
    ):
        raise PersistentIdentityCapabilityError("owned object changed during UUID audit")
    if len(matches) != 1:
        raise PersistentIdentityCapabilityError("filesystem UUID match is not unique")
    return matches[0]


def _linux_opaque_handle(descriptor: int) -> tuple[int, bytes]:
    libc = ctypes.CDLL(None, use_errno=True)
    name_to_handle_at = getattr(libc, "name_to_handle_at", None)
    if name_to_handle_at is None:
        raise PersistentIdentityCapabilityError("libc does not expose name_to_handle_at")

    class FileHandle(ctypes.Structure):
        _fields_ = [
            ("handle_bytes", ctypes.c_uint),
            ("handle_type", ctypes.c_int),
            ("value", ctypes.c_ubyte * 128),
        ]

    name_to_handle_at.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(FileHandle),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    )
    name_to_handle_at.restype = ctypes.c_int
    mount_id = ctypes.c_int()
    first = FileHandle()
    first.handle_bytes = 0
    result = name_to_handle_at(
        descriptor,
        b"",
        ctypes.byref(first),
        ctypes.byref(mount_id),
        0x1000,
    )
    code = ctypes.get_errno()
    if result == 0:
        needed = int(first.handle_bytes)
    elif code == errno.EOVERFLOW:
        needed = int(first.handle_bytes)
    else:
        raise PersistentIdentityCapabilityError(
            f"name_to_handle_at size query failed: errno {code}"
        )
    if not 1 <= needed <= 128:
        raise PersistentIdentityCapabilityError("Linux file handle size is unsupported")
    second = FileHandle()
    second.handle_bytes = needed
    if (
        name_to_handle_at(
            descriptor,
            b"",
            ctypes.byref(second),
            ctypes.byref(mount_id),
            0x1000,
        )
        != 0
    ):
        retry_code = ctypes.get_errno()
        raise PersistentIdentityCapabilityError(
            f"name_to_handle_at identity query failed: errno {retry_code}"
        )
    if int(second.handle_bytes) != needed:
        raise PersistentIdentityCapabilityError("Linux file handle size changed")
    return int(second.handle_type), bytes(second.value[:needed])


def _capture_posix_fd(
    descriptor: int, *, require_directory: bool
) -> tuple[PosixFileIdentityV1, int]:
    if not sys.platform.startswith("linux"):
        raise PersistentIdentityCapabilityError("persistent POSIX identity is Linux-only")
    metadata = os.fstat(descriptor)
    if stat.S_ISDIR(metadata.st_mode) != require_directory:
        raise PersistentIdentitySchemaError("identity object type changed")
    if not require_directory and not stat.S_ISREG(metadata.st_mode):
        raise PersistentIdentitySchemaError("identity object is not a regular file")
    filesystem_uuid = _linux_filesystem_uuid(descriptor)
    handle_type, handle = _linux_opaque_handle(descriptor)
    after = os.fstat(descriptor)
    if (metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_nlink) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
    ):
        raise PersistentIdentityCapabilityError("owned object changed during handle audit")
    return PosixFileIdentityV1(filesystem_uuid, handle_type, handle.hex()), int(after.st_nlink)


def directory_identity_from_posix_fd(descriptor: int) -> DirectoryIdentityV1:
    posix_identity, _links = _capture_posix_fd(descriptor, require_directory=True)
    return DirectoryIdentityV1("posix", posix_identity)


def file_identity_from_posix_fd(descriptor: int) -> PersistedFileIdentityV1:
    identity, links = _capture_posix_fd(descriptor, require_directory=False)
    return PersistedFileIdentityV1("posix", identity, links)


def file_identity_from_open_handle(handle: BinaryIO) -> PersistedFileIdentityV1:
    """Capture the restart-stable identity of an already opened regular file."""

    if os.name == "nt":
        native = _win32.get_osfhandle(handle.fileno())
        identity, links, _attributes = _windows_handle_identity(
            native,
            require_directory=False,
        )
        return PersistedFileIdentityV1("windows", identity, links)
    return file_identity_from_posix_fd(handle.fileno())


def capture_file_identity(path: Path) -> PersistedFileIdentityV1:
    path = Path(path)
    if os.name == "nt":
        from ctypes import wintypes

        handle = _open_windows_path(path, directory=False)
        try:
            identity, links, _attributes = _windows_handle_identity(
                handle,
                require_directory=False,
            )
        finally:
            _win32.load_library("kernel32").CloseHandle(wintypes.HANDLE(handle))
        return PersistedFileIdentityV1("windows", identity, links)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        posix_identity, links = _capture_posix_fd(descriptor, require_directory=False)
    finally:
        os.close(descriptor)
    return PersistedFileIdentityV1("posix", posix_identity, links)


def capture_directory_identity(path: Path) -> DirectoryIdentityV1:
    path = Path(path)
    if os.name == "nt":
        from ctypes import wintypes

        handle = _open_windows_path(path, directory=True)
        try:
            identity, _links, _attributes = _windows_handle_identity(
                handle,
                require_directory=True,
            )
        finally:
            _win32.load_library("kernel32").CloseHandle(wintypes.HANDLE(handle))
        return DirectoryIdentityV1("windows", identity)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    descriptor = os.open(path, flags)
    try:
        posix_identity, _links = _capture_posix_fd(descriptor, require_directory=True)
    finally:
        os.close(descriptor)
    return DirectoryIdentityV1("posix", posix_identity)
