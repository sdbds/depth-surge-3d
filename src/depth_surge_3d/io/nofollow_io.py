"""Handle-relative, no-follow access beneath one acquired directory root."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
import stat
import secrets
import sys
from typing import BinaryIO, Callable, Iterator, Protocol


class NoFollowPathError(ValueError):
    """A path escaped its root or crossed a link, mount, or reparse component."""


_POSIX_QUARANTINE_PREFIX = ".depth-surge-quarantine-v1-"
_RENAME_NOREPLACE = 1
FIXED_FILE_HASH_STREAM_BYTES = 1024 * 1024


def _posix_quarantine_noreplace(
    source_parent: int,
    source_name: str,
    target_parent: int,
    target_name: str,
) -> None:
    """Atomically move one name without replacing an existing quarantine entry."""

    if not sys.platform.startswith("linux"):
        raise NoFollowPathError("POSIX safe deletion requires Linux renameat2")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise NoFollowPathError("POSIX safe deletion requires renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_parent,
        os.fsencode(source_name),
        target_parent,
        os.fsencode(target_name),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), source_name)


def posix_quarantine_name_v1(identity: bytes) -> str:
    """Return a bounded deterministic quarantine name for a durable delete intent."""

    return _POSIX_QUARANTINE_PREFIX + hashlib.sha256(identity).hexdigest()[:32]


def posix_remove_verified_name(
    parent: int,
    name: str,
    opened_descriptor: int,
    *,
    directory: bool,
    quarantine_name: str | None = None,
    recovering: bool = False,
    fault: Callable[[str, str], None] | None = None,
) -> None:
    """Move a verified object away from its public name before deleting it.

    Linux cannot unlink by object descriptor. This adapter closes races on the
    authorized public name with renameat2(NO_REPLACE), then operates in the
    process-private quarantine namespace. Same-principal active monitoring of
    that internal random name is outside the application's concurrency model.
    """

    if "/" in name or name in {"", ".", ".."}:
        raise NoFollowPathError("verified deletion name is not one component")
    opened = os.fstat(opened_descriptor)
    expected_type = stat.S_IFMT(opened.st_mode)
    if directory != stat.S_ISDIR(opened.st_mode):
        raise NoFollowPathError(f"verified deletion object has the wrong type: {name}")
    if not isinstance(recovering, bool):
        raise TypeError("quarantine recovery state must be boolean")
    if recovering and not name.startswith(_POSIX_QUARANTINE_PREFIX):
        raise NoFollowPathError("quarantine recovery name is outside its namespace")
    quarantine = name if recovering else quarantine_name
    if not recovering:
        if quarantine is None:
            quarantine = _POSIX_QUARANTINE_PREFIX + secrets.token_hex(16)
        if (
            not quarantine.startswith(_POSIX_QUARANTINE_PREFIX)
            or "/" in quarantine
            or quarantine in {"", ".", ".."}
        ):
            raise NoFollowPathError("deletion quarantine name is outside its namespace")
        try:
            _posix_quarantine_noreplace(parent, name, parent, quarantine)
        except OSError as error:
            if error.errno == errno.EEXIST:
                raise NoFollowPathError(
                    f"deletion quarantine already exists: {quarantine}"
                ) from error
            raise
        os.fsync(parent)
        if fault is not None:
            fault("after_quarantine_rename", quarantine)
    assert quarantine is not None
    if directory:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
        )
    else:
        path_only = getattr(os, "O_PATH", 0)
        if not path_only:
            raise NoFollowPathError("POSIX safe deletion requires O_PATH")
        flags = path_only | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    quarantined = -1
    try:
        quarantined = os.open(quarantine, flags, dir_fd=parent)
        actual = os.fstat(quarantined)
        if stat.S_IFMT(actual.st_mode) != expected_type or (
            int(actual.st_dev),
            int(actual.st_ino),
        ) != (int(opened.st_dev), int(opened.st_ino)):
            if not recovering:
                try:
                    _posix_quarantine_noreplace(parent, quarantine, parent, name)
                except OSError:
                    pass
            raise NoFollowPathError(
                f"name changed before quarantine acquisition: {name}"
            )
        if fault is not None:
            fault("after_quarantine_identity", quarantine)
        if directory:
            with os.scandir(quarantined) as iterator:
                if next(iterator, None) is not None:
                    raise NoFollowPathError(
                        f"quarantined directory is no longer empty: {name}"
                    )
        latest = os.stat(quarantine, dir_fd=parent, follow_symlinks=False)
        if stat.S_IFMT(latest.st_mode) != expected_type or (
            int(latest.st_dev),
            int(latest.st_ino),
        ) != (int(opened.st_dev), int(opened.st_ino)):
            raise NoFollowPathError(
                f"quarantine identity changed before deletion: {name}"
            )
        if directory:
            os.rmdir(quarantine, dir_fd=parent)
        else:
            os.unlink(quarantine, dir_fd=parent)
        os.fsync(parent)
        if not recovering:
            try:
                os.stat(name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise NoFollowPathError(
                    f"verified deletion name reappeared during deletion: {name}"
                )
    finally:
        if quarantined >= 0:
            os.close(quarantined)


@dataclass(frozen=True)
class ContainedDirectoryEntry:
    """One entry observed and type-checked through its parent directory handle."""

    name: str
    is_directory: bool
    is_regular: bool


@dataclass(frozen=True, slots=True)
class WindowsEnumeratedChildV1:
    """One FILE_ID_EXTD_DIR_INFO entry with its full 128-bit object ID."""

    name: str
    attributes: int
    file_id: str


def windows_file_id_extd_entries_v1(
    buffer: ctypes.Array,
    buffer_size: int,
) -> Iterator[WindowsEnumeratedChildV1]:
    """Parse a bounded FileIdExtdDirectoryInfo page without losing identity."""

    offset = 0
    while True:
        if offset < 0 or offset + 88 > buffer_size:
            raise NoFollowPathError("extended directory entry exceeds its buffer")
        next_offset = ctypes.c_uint32.from_buffer(buffer, offset).value
        attributes = ctypes.c_uint32.from_buffer(buffer, offset + 56).value
        name_length = ctypes.c_uint32.from_buffer(buffer, offset + 60).value
        if name_length % 2 or name_length > buffer_size - offset - 88:
            raise NoFollowPathError("extended directory entry name is malformed")
        file_id = ctypes.string_at(
            ctypes.addressof(buffer) + offset + 72,
            16,
        ).hex()
        name = ctypes.wstring_at(
            ctypes.addressof(buffer) + offset + 88,
            name_length // 2,
        )
        if name not in {".", ".."}:
            yield WindowsEnumeratedChildV1(name, int(attributes), file_id)
        if next_offset == 0:
            return
        minimum_next = 88 + int(name_length)
        if next_offset < minimum_next or next_offset > buffer_size - offset:
            raise NoFollowPathError("extended directory entry offset is malformed")
        offset += int(next_offset)


@dataclass(frozen=True)
class RegularAllocationEvidence:
    """Invocation-local identity and bytes uniquely released by one unlink."""

    identity: tuple[str, ...]
    allocated_bytes: int


@dataclass(frozen=True)
class ChildMutationEvidence:
    """Identity or durable absence observed before one authorized child mutation."""

    path: Path
    kind: str
    parent_identity: object | None
    identity: object | None


@dataclass(frozen=True, slots=True)
class AuthenticatedFixedFileV1:
    """Scalar identity, digest, and length from one exact regular-file handle."""

    identity: object
    raw_sha256: str
    byte_count: int


@dataclass(frozen=True)
class _MountIdentity:
    algorithm: str
    value: int


def _relative_parts(
    root: Path,
    path: Path,
    *,
    allow_root: bool = False,
) -> tuple[str, ...]:
    lexical_root = Path(os.path.abspath(os.fspath(root)))
    lexical_path = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = lexical_path.relative_to(lexical_root)
    except ValueError as error:
        raise NoFollowPathError(
            f"path is not beneath the acquired root: {path}"
        ) from error
    if not relative.parts:
        if allow_root:
            return ()
        raise NoFollowPathError(f"path beneath the acquired root is invalid: {path}")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise NoFollowPathError(f"path beneath the acquired root is invalid: {path}")
    return tuple(relative.parts)


def _is_missing_path_error(error: NoFollowPathError) -> bool:
    if isinstance(error.__cause__, FileNotFoundError):
        return True
    return os.name == "nt" and any(
        marker in str(error) for marker in ("Windows error 2", "Windows error 3")
    )


def _linux_mount_identity(descriptor: int) -> _MountIdentity:
    if not sys.platform.startswith("linux"):
        raise NoFollowPathError("no-follow mount identity is Linux-only on POSIX")

    class StatxTimestamp(ctypes.Structure):
        _fields_ = [
            ("seconds", ctypes.c_int64),
            ("nanoseconds", ctypes.c_uint32),
            ("reserved", ctypes.c_int32),
        ]

    class Statx(ctypes.Structure):
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
            ("access_time", StatxTimestamp),
            ("birth_time", StatxTimestamp),
            ("change_time", StatxTimestamp),
            ("modify_time", StatxTimestamp),
            ("rdev_major", ctypes.c_uint32),
            ("rdev_minor", ctypes.c_uint32),
            ("dev_major", ctypes.c_uint32),
            ("dev_minor", ctypes.c_uint32),
            ("mount_id", ctypes.c_uint64),
            ("spare", ctypes.c_uint64 * 13),
        ]

    libc = ctypes.CDLL(None, use_errno=True)
    statx = getattr(libc, "statx", None)
    if statx is None:
        raise NoFollowPathError("libc does not expose statx mount identity")
    statx.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(Statx),
    )
    statx.restype = ctypes.c_int
    result = Statx()
    statx_mnt_id = 0x00001000
    statx_mnt_id_unique = 0x00004000
    if (
        statx(
            descriptor,
            b"",
            0x1000 | 0x100,
            statx_mnt_id | statx_mnt_id_unique,
            ctypes.byref(result),
        )
        != 0
    ):
        code = ctypes.get_errno()
        raise NoFollowPathError(f"statx mount identity failed: errno {code}")
    if result.mask & statx_mnt_id_unique:
        return _MountIdentity("statx-mnt-id-unique", int(result.mount_id))
    if result.mask & statx_mnt_id:
        return _MountIdentity("statx-mnt-id", int(result.mount_id))
    raise NoFollowPathError("statx returned no mount identity")


def _linux_filesystem_magic(descriptor: int) -> int:
    class StatFs(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.c_long),
            ("block_size", ctypes.c_long),
            ("blocks", ctypes.c_ulong),
            ("blocks_free", ctypes.c_ulong),
            ("blocks_available", ctypes.c_ulong),
            ("files", ctypes.c_ulong),
            ("files_free", ctypes.c_ulong),
            ("filesystem_id", ctypes.c_int * 2),
            ("name_length", ctypes.c_long),
            ("fragment_size", ctypes.c_long),
            ("flags", ctypes.c_long),
            ("spare", ctypes.c_long * 4),
        ]

    libc = ctypes.CDLL(None, use_errno=True)
    result = StatFs()
    if libc.fstatfs(descriptor, ctypes.byref(result)) != 0:
        code = ctypes.get_errno()
        raise NoFollowPathError(f"fstatfs failed: errno {code}")
    return int(result.type)


def regular_allocation_evidence(handle: BinaryIO) -> RegularAllocationEvidence | None:
    """Return conservative unique-release evidence for one acquired regular handle."""

    descriptor = handle.fileno()
    if os.name != "nt":
        metadata = os.fstat(descriptor)
        if metadata.st_nlink != 1 or _linux_filesystem_magic(descriptor) != 0xEF53:
            return None
        mount = _linux_mount_identity(descriptor)
        return RegularAllocationEvidence(
            (
                mount.algorithm,
                str(mount.value),
                str(int(metadata.st_dev)),
                str(int(metadata.st_ino)),
            ),
            int(metadata.st_blocks) * 512,
        )

    from ctypes import wintypes
    import msvcrt

    from .file_identity import windows_identity_from_handle

    windows_handle = int(msvcrt.get_osfhandle(descriptor))
    identity, links, attributes = windows_identity_from_handle(
        windows_handle,
        require_directory=False,
    )
    if links != 1 or attributes & (0x200 | 0x400 | 0x800 | 0x4000):
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_volume = getattr(kernel32, "GetVolumeInformationByHandleW", None)
    if get_volume is None:
        return None
    get_volume.argtypes = (
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPWSTR,
        wintypes.DWORD,
    )
    get_volume.restype = wintypes.BOOL
    filesystem_name = ctypes.create_unicode_buffer(32)
    if not get_volume(
        wintypes.HANDLE(windows_handle),
        None,
        0,
        None,
        None,
        None,
        filesystem_name,
        len(filesystem_name),
    ):
        return None
    if filesystem_name.value.upper() != "NTFS":
        return None

    class FileStandardInfo(ctypes.Structure):
        _fields_ = [
            ("allocation_size", ctypes.c_longlong),
            ("end_of_file", ctypes.c_longlong),
            ("number_of_links", wintypes.DWORD),
            ("delete_pending", wintypes.BOOLEAN),
            ("directory", wintypes.BOOLEAN),
        ]

    standard = FileStandardInfo()
    get_information = kernel32.GetFileInformationByHandleEx
    if not get_information(
        wintypes.HANDLE(windows_handle),
        1,
        ctypes.byref(standard),
        ctypes.sizeof(standard),
    ):
        return None
    return RegularAllocationEvidence(
        ("windows-file-id-v1", identity.volume_serial, identity.file_id),
        int(standard.allocation_size),
    )


class _AcquiredRootImpl(Protocol):
    def close(self) -> None: ...

    def persistent_identity(self) -> object: ...

    def open_regular(self, parts: tuple[str, ...]) -> BinaryIO: ...

    def open_regular_update(self, parts: tuple[str, ...]) -> BinaryIO: ...

    def open_regular_locked(self, parts: tuple[str, ...]) -> BinaryIO: ...

    def create_regular_exclusive(
        self,
        parts: tuple[str, ...],
    ) -> tuple[BinaryIO, object]: ...

    def regular_identity(
        self,
        parts: tuple[str, ...],
        *,
        missing_ok: bool,
    ) -> object | None: ...

    def iter_directory(
        self, parts: tuple[str, ...]
    ) -> Iterator[ContainedDirectoryEntry]: ...

    def attest_tree(
        self, parts: tuple[str, ...], *, missing_ok: bool
    ) -> object | None: ...

    def directory_identity(
        self,
        parts: tuple[str, ...],
        *,
        missing_ok: bool,
    ) -> object | None: ...

    def persistent_directory_identity(
        self,
        parts: tuple[str, ...],
        *,
        missing_ok: bool,
    ) -> object | None: ...

    def clear_tree(
        self,
        parts: tuple[str, ...],
        *,
        missing_ok: bool,
        expected_identity: object | None,
    ) -> None: ...

    def remove_tree(
        self,
        parts: tuple[str, ...],
        *,
        missing_ok: bool,
        expected_identity: object | None,
        expected_parent_identity: object | None,
    ) -> None: ...

    def unlink_regular(
        self,
        parts: tuple[str, ...],
        *,
        missing_ok: bool,
        expected_parent_identity: object | None,
        expected_identity: object | None = None,
    ) -> None: ...


class _PosixAcquiredRoot:
    def __init__(self, root: Path) -> None:
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        if not nofollow:
            raise NoFollowPathError("platform has no no-follow regular-file open")
        self._directory_flags = (
            os.O_RDONLY
            | nofollow
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            self._root_fd = os.open(root, self._directory_flags)
        except OSError as error:
            raise NoFollowPathError(f"could not acquire job root: {error}") from error
        try:
            metadata = os.fstat(self._root_fd)
            if not stat.S_ISDIR(metadata.st_mode):
                raise NoFollowPathError("acquired job root is not a directory")
            self._device = int(metadata.st_dev)
            self._mount = _linux_mount_identity(self._root_fd)
        except BaseException:
            os.close(self._root_fd)
            raise

    def close(self) -> None:
        if self._root_fd >= 0:
            os.close(self._root_fd)
            self._root_fd = -1

    def persistent_identity(self) -> object:
        from .file_identity import directory_identity_from_posix_fd

        return directory_identity_from_posix_fd(self._root_fd)

    def _verify(
        self, descriptor: int, *, directory: bool, component: object
    ) -> os.stat_result:
        metadata = os.fstat(descriptor)
        expected_type = stat.S_ISDIR if directory else stat.S_ISREG
        if not expected_type(metadata.st_mode):
            raise NoFollowPathError(f"path component has the wrong type: {component}")
        if (
            int(metadata.st_dev) != self._device
            or _linux_mount_identity(descriptor) != self._mount
        ):
            raise NoFollowPathError(f"path crossed a filesystem mount: {component}")
        return metadata

    def _open_directory(self, parts: tuple[str, ...]) -> int:
        current = os.dup(self._root_fd)
        try:
            for component in parts:
                child = os.open(component, self._directory_flags, dir_fd=current)
                try:
                    self._verify(child, directory=True, component=component)
                except BaseException:
                    os.close(child)
                    raise
                os.close(current)
                current = child
            return current
        except BaseException as error:
            os.close(current)
            if isinstance(error, OSError):
                raise NoFollowPathError(
                    f"path crossed a link or invalid component: {error}"
                ) from error
            raise

    def open_regular(self, parts: tuple[str, ...]) -> BinaryIO:
        parent = self._open_directory(parts[:-1])
        descriptor = -1
        try:
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent,
            )
            self._verify(descriptor, directory=False, component=parts[-1])
            handle = os.fdopen(descriptor, "rb", buffering=0, closefd=True)
            descriptor = -1
            return handle
        except OSError as error:
            raise NoFollowPathError(
                f"path crossed a link or invalid component: {error}"
            ) from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent)

    @staticmethod
    def _require_enumerated_child_identity(
        opened: os.stat_result,
        enumerated: os.stat_result,
        name: str,
    ) -> None:
        if (
            int(opened.st_dev) != int(enumerated.st_dev)
            or int(opened.st_ino) != int(enumerated.st_ino)
            or stat.S_IFMT(opened.st_mode) != stat.S_IFMT(enumerated.st_mode)
        ):
            raise NoFollowPathError(
                f"directory child changed before acquisition: {name}"
            )

    def open_regular_update(self, parts: tuple[str, ...]) -> BinaryIO:
        parent = self._open_directory(parts[:-1])
        descriptor = -1
        try:
            descriptor = os.open(
                parts[-1],
                os.O_RDWR
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_BINARY", 0),
                dir_fd=parent,
            )
            self._verify(descriptor, directory=False, component=parts[-1])
            handle = os.fdopen(descriptor, "r+b", buffering=0, closefd=True)
            descriptor = -1
            return handle
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent)

    def open_regular_locked(self, parts: tuple[str, ...]) -> BinaryIO:
        return self.open_regular_update(parts)

    def create_regular_exclusive(
        self,
        parts: tuple[str, ...],
    ) -> tuple[BinaryIO, object]:
        parent = self._open_directory(parts[:-1])
        descriptor = -1
        try:
            descriptor = os.open(
                parts[-1],
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_BINARY", 0),
                0o600,
                dir_fd=parent,
            )
            metadata = self._verify(
                descriptor,
                directory=False,
                component=parts[-1],
            )
            identity = (int(metadata.st_dev), int(metadata.st_ino))
            handle = os.fdopen(descriptor, "w+b", buffering=0, closefd=True)
            descriptor = -1
            return handle, identity
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent)

    def regular_identity(
        self,
        parts: tuple[str, ...],
        *,
        missing_ok: bool,
    ) -> object | None:
        parent = self._open_directory(parts[:-1])
        descriptor = -1
        try:
            try:
                descriptor = os.open(
                    parts[-1],
                    getattr(os, "O_PATH", os.O_RDONLY)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=parent,
                )
            except FileNotFoundError:
                if missing_ok:
                    return None
                raise
            metadata = self._verify(
                descriptor,
                directory=False,
                component=parts[-1],
            )
            return int(metadata.st_dev), int(metadata.st_ino)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent)

    def iter_directory(
        self, parts: tuple[str, ...]
    ) -> Iterator[ContainedDirectoryEntry]:
        directory = self._open_directory(parts)
        try:
            with os.scandir(directory) as iterator:
                for entry in iterator:
                    metadata = os.stat(
                        entry.name, dir_fd=directory, follow_symlinks=False
                    )
                    if stat.S_ISLNK(metadata.st_mode):
                        raise NoFollowPathError(
                            f"directory contains a link: {entry.name}"
                        )
                    is_directory = stat.S_ISDIR(metadata.st_mode)
                    is_regular = stat.S_ISREG(metadata.st_mode)
                    flags = (
                        self._directory_flags
                        if is_directory
                        else os.O_RDONLY
                        | getattr(os, "O_NOFOLLOW", 0)
                        | getattr(os, "O_NONBLOCK", 0)
                        | getattr(os, "O_CLOEXEC", 0)
                    )
                    child = os.open(entry.name, flags, dir_fd=directory)
                    try:
                        if is_directory or is_regular:
                            opened = self._verify(
                                child,
                                directory=is_directory,
                                component=entry.name,
                            )
                        else:
                            opened = os.fstat(child)
                            if int(opened.st_dev) != self._device:
                                raise NoFollowPathError(
                                    "directory entry crossed a filesystem: "
                                    f"{entry.name}"
                                )
                        self._require_enumerated_child_identity(
                            opened,
                            metadata,
                            entry.name,
                        )
                    finally:
                        os.close(child)
                    yield ContainedDirectoryEntry(entry.name, is_directory, is_regular)
        except OSError as error:
            raise NoFollowPathError(
                f"directory changed or contains an unsafe entry: {error}"
            ) from error
        finally:
            os.close(directory)

    def _open_directory_optional(
        self, parts: tuple[str, ...], *, missing_ok: bool
    ) -> int:
        try:
            return self._open_directory(parts)
        except NoFollowPathError as error:
            if missing_ok and isinstance(error.__cause__, FileNotFoundError):
                return -1
            raise

    def _attest_directory(self, directory: int) -> None:
        with os.scandir(directory) as iterator:
            for entry in iterator:
                metadata = os.stat(entry.name, dir_fd=directory, follow_symlinks=False)
                if stat.S_ISLNK(metadata.st_mode):
                    raise NoFollowPathError(f"tree contains a link: {entry.name}")
                if stat.S_ISDIR(metadata.st_mode):
                    child = os.open(entry.name, self._directory_flags, dir_fd=directory)
                    try:
                        opened = self._verify(
                            child,
                            directory=True,
                            component=entry.name,
                        )
                        self._require_enumerated_child_identity(
                            opened,
                            metadata,
                            entry.name,
                        )
                        self._attest_directory(child)
                    finally:
                        os.close(child)
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    raise NoFollowPathError(
                        f"tree contains a special file: {entry.name}"
                    )
                child = os.open(
                    entry.name,
                    os.O_RDONLY
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=directory,
                )
                try:
                    opened = self._verify(
                        child,
                        directory=False,
                        component=entry.name,
                    )
                    self._require_enumerated_child_identity(
                        opened,
                        metadata,
                        entry.name,
                    )
                finally:
                    os.close(child)

    def _directory_identity(self, directory: int) -> tuple[int, int, _MountIdentity]:
        metadata = self._verify(directory, directory=True, component="authorized root")
        return int(metadata.st_dev), int(metadata.st_ino), self._mount

    def _require_directory_identity(
        self,
        directory: int,
        expected: object | None,
    ) -> None:
        if expected is not None and self._directory_identity(directory) != expected:
            raise NoFollowPathError(
                "authorized directory identity changed before mutation"
            )

    def _remove_verified_name(
        self,
        parent: int,
        name: str,
        opened_descriptor: int,
        *,
        directory: bool,
    ) -> None:
        posix_remove_verified_name(
            parent,
            name,
            opened_descriptor,
            directory=directory,
        )

    def directory_identity(
        self,
        parts: tuple[str, ...],
        *,
        missing_ok: bool,
    ) -> object | None:
        directory = self._open_directory_optional(parts, missing_ok=missing_ok)
        if directory < 0:
            return None
        try:
            return self._directory_identity(directory)
        finally:
            os.close(directory)

    def persistent_directory_identity(
        self,
        parts: tuple[str, ...],
        *,
        missing_ok: bool,
    ) -> object | None:
        directory = self._open_directory_optional(parts, missing_ok=missing_ok)
        if directory < 0:
            return None
        try:
            from .file_identity import directory_identity_from_posix_fd

            return directory_identity_from_posix_fd(directory)
        finally:
            os.close(directory)

    def attest_tree(self, parts: tuple[str, ...], *, missing_ok: bool) -> object | None:
        directory = self._open_directory_optional(parts, missing_ok=missing_ok)
        if directory < 0:
            return None
        try:
            identity = self._directory_identity(directory)
            self._attest_directory(directory)
            return identity
        finally:
            os.close(directory)

    def _delete_directory_contents(self, directory: int) -> None:
        while True:
            with os.scandir(directory) as iterator:
                entry = next(iterator, None)
            if entry is None:
                os.fsync(directory)
                return
            metadata = os.stat(entry.name, dir_fd=directory, follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise NoFollowPathError(f"tree contains a link: {entry.name}")
            if stat.S_ISDIR(metadata.st_mode):
                child = os.open(entry.name, self._directory_flags, dir_fd=directory)
                try:
                    opened = self._verify(
                        child,
                        directory=True,
                        component=entry.name,
                    )
                    self._require_enumerated_child_identity(
                        opened,
                        metadata,
                        entry.name,
                    )
                    self._delete_directory_contents(child)
                    self._remove_verified_name(
                        directory,
                        entry.name,
                        child,
                        directory=True,
                    )
                finally:
                    os.close(child)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise NoFollowPathError(f"tree contains a special file: {entry.name}")
            child = os.open(
                entry.name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory,
            )
            try:
                opened = self._verify(
                    child,
                    directory=False,
                    component=entry.name,
                )
                self._require_enumerated_child_identity(
                    opened,
                    metadata,
                    entry.name,
                )
                self._remove_verified_name(
                    directory,
                    entry.name,
                    child,
                    directory=False,
                )
            finally:
                os.close(child)

    def clear_tree(
        self,
        parts: tuple[str, ...],
        *,
        missing_ok: bool,
        expected_identity: object | None,
    ) -> None:
        directory = self._open_directory_optional(parts, missing_ok=missing_ok)
        if directory < 0:
            if expected_identity is not None:
                raise NoFollowPathError(
                    "authorized directory disappeared before mutation"
                )
            return
        try:
            self._require_directory_identity(directory, expected_identity)
            self._delete_directory_contents(directory)
        finally:
            os.close(directory)

    def remove_tree(
        self,
        parts: tuple[str, ...],
        *,
        missing_ok: bool,
        expected_identity: object | None,
        expected_parent_identity: object | None,
    ) -> None:
        parent = self._open_directory(parts[:-1])
        directory = -1
        try:
            self._require_directory_identity(parent, expected_parent_identity)
            try:
                directory = os.open(parts[-1], self._directory_flags, dir_fd=parent)
            except FileNotFoundError:
                if expected_identity is not None:
                    raise NoFollowPathError(
                        "authorized directory disappeared before mutation"
                    )
                if missing_ok:
                    return
                raise
            self._verify(directory, directory=True, component=parts[-1])
            self._require_directory_identity(directory, expected_identity)
            self._delete_directory_contents(directory)
            self._remove_verified_name(
                parent,
                parts[-1],
                directory,
                directory=True,
            )
        except OSError as error:
            if isinstance(error, FileNotFoundError) and missing_ok:
                return
            raise NoFollowPathError(f"tree removal failed safely: {error}") from error
        finally:
            if directory >= 0:
                os.close(directory)
            os.close(parent)

    def unlink_regular(
        self,
        parts: tuple[str, ...],
        *,
        missing_ok: bool,
        expected_parent_identity: object | None,
        expected_identity: object | None = None,
    ) -> None:
        parent = self._open_directory(parts[:-1])
        descriptor = -1
        try:
            self._require_directory_identity(parent, expected_parent_identity)
            try:
                descriptor = os.open(
                    parts[-1],
                    os.O_RDONLY
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=parent,
                )
            except FileNotFoundError:
                if missing_ok:
                    return
                raise
            metadata = self._verify(descriptor, directory=False, component=parts[-1])
            actual_identity = (int(metadata.st_dev), int(metadata.st_ino))
            if expected_identity is not None and actual_identity != expected_identity:
                raise NoFollowPathError("regular-file identity changed before mutation")
            self._remove_verified_name(
                parent,
                parts[-1],
                descriptor,
                directory=False,
            )
        except OSError as error:
            if isinstance(error, FileNotFoundError) and missing_ok:
                return
            raise NoFollowPathError(f"file removal failed safely: {error}") from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent)


class _WindowsAcquiredRoot:  # noqa: C901
    def __init__(self, root: Path) -> None:
        from ctypes import wintypes

        from .file_identity import (
            PersistentIdentityCapabilityError,
            PersistentIdentitySchemaError,
            windows_identity_from_handle,
        )

        self._wintypes = wintypes
        self._identity_errors = (
            PersistentIdentityCapabilityError,
            PersistentIdentitySchemaError,
        )
        self._identity_from_handle = windows_identity_from_handle
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._ntdll = ctypes.WinDLL("ntdll")
        self._invalid_handle = wintypes.HANDLE(-1).value
        self._close_handle = self._kernel32.CloseHandle
        self._close_handle.argtypes = (wintypes.HANDLE,)
        self._close_handle.restype = wintypes.BOOL
        self._create_file = self._kernel32.CreateFileW
        self._create_file.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        self._create_file.restype = wintypes.HANDLE
        self._get_directory = self._kernel32.GetFileInformationByHandleEx
        self._get_directory.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        )
        self._get_directory.restype = wintypes.BOOL
        self._set_information = self._kernel32.SetFileInformationByHandle
        self._set_information.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        )
        self._set_information.restype = wintypes.BOOL

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

        class IoStatusBlock(ctypes.Structure):
            _fields_ = [("status", ctypes.c_void_p), ("information", ctypes.c_size_t)]

        self._unicode_string = UnicodeString
        self._object_attributes = ObjectAttributes
        self._io_status_block = IoStatusBlock
        self._nt_create_file = self._ntdll.NtCreateFile
        self._nt_create_file.argtypes = (
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
        self._nt_create_file.restype = ctypes.c_long
        self._status_to_error = self._ntdll.RtlNtStatusToDosError
        self._status_to_error.argtypes = (ctypes.c_long,)
        self._status_to_error.restype = wintypes.DWORD

        handle = self._create_file(
            str(root),
            0x00000001 | 0x00000080 | 0x00100000,
            0x1 | 0x2 | 0x4,
            None,
            3,
            0x02000000 | 0x00200000,
            None,
        )
        if handle == self._invalid_handle:
            raise NoFollowPathError(
                f"could not acquire job root: Windows error {ctypes.get_last_error()}"
            )
        self._root_handle = int(handle)
        try:
            identity, _links, attributes = self._verify(
                self._root_handle,
                directory=True,
                component=root,
            )
            if attributes & 0x400:
                raise NoFollowPathError("acquired job root is a reparse point")
            self._volume = identity.volume_serial
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._root_handle >= 0:
            self._close_handle(self._wintypes.HANDLE(self._root_handle))
            self._root_handle = -1

    def persistent_identity(self) -> object:
        from .file_identity import DirectoryIdentityV1

        return DirectoryIdentityV1(
            "windows",
            self._directory_identity(self._root_handle),
        )

    def _verify(self, handle: int, *, directory: bool, component: object):
        try:
            identity, links, attributes = self._identity_from_handle(
                handle,
                require_directory=directory,
                allow_reparse=True,
            )
        except self._identity_errors as error:
            raise NoFollowPathError(str(error)) from error
        if attributes & 0x400:
            raise NoFollowPathError(f"path crossed a reparse point: {component}")
        if hasattr(self, "_volume") and identity.volume_serial != self._volume:
            raise NoFollowPathError(f"path crossed a filesystem volume: {component}")
        return identity, links, attributes

    def _open_relative(
        self,
        parent: int,
        name: str,
        *,
        directory: bool,
        delete: bool = False,
        missing_ok: bool = False,
        create: bool = False,
        write: bool = False,
        share_delete: bool = True,
    ) -> int:
        if create and directory:
            raise NoFollowPathError(
                "exclusive regular creation cannot create a directory"
            )
        name_buffer = ctypes.create_unicode_buffer(name)
        encoded_length = len(name.encode("utf-16-le"))
        unicode_name = self._unicode_string(
            encoded_length,
            encoded_length + 2,
            ctypes.cast(name_buffer, self._wintypes.LPWSTR),
        )
        attributes = self._object_attributes(
            ctypes.sizeof(self._object_attributes),
            self._wintypes.HANDLE(parent),
            ctypes.pointer(unicode_name),
            0x40,
            None,
            None,
        )
        child = self._wintypes.HANDLE()
        io_status = self._io_status_block()
        desired_access = (
            0x00000001 | 0x00000080 | 0x00100000
            if directory
            else 0x80000000 | 0x00000080 | 0x00100000
        )
        if write:
            desired_access |= 0x40000000
        if delete:
            desired_access |= 0x00010000
        options = 0x00200000 | 0x20 | (0x1 if directory else 0x40)
        status = int(
            self._nt_create_file(
                ctypes.byref(child),
                desired_access,
                ctypes.byref(attributes),
                ctypes.byref(io_status),
                None,
                0x80 if create else 0,
                0x1 | 0x2 | (0x4 if share_delete else 0),
                2 if create else 1,
                options,
                None,
                0,
            )
        )
        if status < 0:
            code = int(self._status_to_error(status))
            if missing_ok and code in {2, 3}:
                return -1
            if create and code in {80, 183}:
                raise FileExistsError(
                    code, "exclusive regular-file target exists", name
                )
            raise NoFollowPathError(
                f"path crossed a link or invalid component: Windows error {code}"
            )
        if child.value is None:
            raise NoFollowPathError("NtCreateFile returned a null handle")
        return int(child.value)

    def _open_directory_optional(
        self,
        parts: tuple[str, ...],
        *,
        missing_ok: bool,
    ) -> tuple[int, bool]:
        if not parts:
            return self._root_handle, False
        opened: list[int] = []
        parent = self._root_handle
        try:
            for component in parts:
                child = self._open_relative(
                    parent,
                    component,
                    directory=True,
                    missing_ok=missing_ok,
                )
                if child < 0:
                    return -1, False
                opened.append(child)
                self._verify(child, directory=True, component=component)
                parent = child
            final = opened.pop()
            return final, True
        finally:
            for handle in reversed(opened):
                self._close_handle(self._wintypes.HANDLE(handle))

    def _open_directory(self, parts: tuple[str, ...]) -> tuple[int, bool]:
        directory, close_directory = self._open_directory_optional(
            parts,
            missing_ok=False,
        )
        if directory < 0:
            raise AssertionError("strict directory open returned a missing sentinel")
        return directory, close_directory

    def open_regular(self, parts: tuple[str, ...]) -> BinaryIO:
        import msvcrt

        parent, close_parent = self._open_directory(parts[:-1])
        final = -1
        try:
            final = self._open_relative(parent, parts[-1], directory=False)
            self._verify(final, directory=False, component=parts[-1])
            descriptor = msvcrt.open_osfhandle(
                final,
                os.O_RDONLY | getattr(os, "O_BINARY", 0),
            )
            final = -1
            return os.fdopen(descriptor, "rb", buffering=0, closefd=True)
        finally:
            if final >= 0:
                self._close_handle(self._wintypes.HANDLE(final))
            if close_parent:
                self._close_handle(self._wintypes.HANDLE(parent))

    def open_regular_update(self, parts: tuple[str, ...]) -> BinaryIO:
        import msvcrt

        parent, close_parent = self._open_directory(parts[:-1])
        final = -1
        try:
            final = self._open_relative(
                parent,
                parts[-1],
                directory=False,
                write=True,
            )
            self._verify(final, directory=False, component=parts[-1])
            descriptor = msvcrt.open_osfhandle(
                final,
                os.O_RDWR | getattr(os, "O_BINARY", 0),
            )
            final = -1
            return os.fdopen(descriptor, "r+b", buffering=0, closefd=True)
        finally:
            if final >= 0:
                self._close_handle(self._wintypes.HANDLE(final))
            if close_parent:
                self._close_handle(self._wintypes.HANDLE(parent))

    def open_regular_locked(self, parts: tuple[str, ...]) -> BinaryIO:
        import msvcrt

        parent, close_parent = self._open_directory(parts[:-1])
        final = -1
        try:
            final = self._open_relative(
                parent,
                parts[-1],
                directory=False,
                write=True,
                share_delete=False,
            )
            self._verify(final, directory=False, component=parts[-1])
            descriptor = msvcrt.open_osfhandle(
                final,
                os.O_RDWR | getattr(os, "O_BINARY", 0),
            )
            final = -1
            return os.fdopen(descriptor, "r+b", buffering=0, closefd=True)
        finally:
            if final >= 0:
                self._close_handle(self._wintypes.HANDLE(final))
            if close_parent:
                self._close_handle(self._wintypes.HANDLE(parent))

    def create_regular_exclusive(
        self,
        parts: tuple[str, ...],
    ) -> tuple[BinaryIO, object]:
        import msvcrt

        parent, close_parent = self._open_directory(parts[:-1])
        final = -1
        try:
            final = self._open_relative(
                parent,
                parts[-1],
                directory=False,
                delete=True,
                create=True,
                write=True,
            )
            identity, _links, _attributes = self._verify(
                final,
                directory=False,
                component=parts[-1],
            )
            descriptor = msvcrt.open_osfhandle(
                final,
                os.O_RDWR | getattr(os, "O_BINARY", 0),
            )
            final = -1
            return os.fdopen(descriptor, "w+b", buffering=0, closefd=True), identity
        finally:
            if final >= 0:
                self._close_handle(self._wintypes.HANDLE(final))
            if close_parent:
                self._close_handle(self._wintypes.HANDLE(parent))

    def regular_identity(
        self,
        parts: tuple[str, ...],
        *,
        missing_ok: bool,
    ) -> object | None:
        parent, close_parent = self._open_directory(parts[:-1])
        descriptor = -1
        try:
            descriptor = self._open_relative(
                parent,
                parts[-1],
                directory=False,
                missing_ok=missing_ok,
            )
            if descriptor < 0:
                return None
            identity, _links, _attributes = self._verify(
                descriptor,
                directory=False,
                component=parts[-1],
            )
            return identity
        finally:
            if descriptor >= 0:
                self._close_handle(self._wintypes.HANDLE(descriptor))
            if close_parent:
                self._close_handle(self._wintypes.HANDLE(parent))

    def _iter_names(self, handle: int) -> Iterator[WindowsEnumeratedChildV1]:
        restart = True
        while True:
            buffer = ctypes.create_string_buffer(64 * 1024)
            if not self._get_directory(
                self._wintypes.HANDLE(handle),
                20 if restart else 19,
                buffer,
                len(buffer),
            ):
                code = ctypes.get_last_error()
                if code == 18:
                    return
                raise NoFollowPathError(
                    f"directory enumeration failed: Windows error {code}"
                )
            restart = False
            yield from windows_file_id_extd_entries_v1(buffer, len(buffer))

    def _open_enumerated_child(
        self,
        parent: int,
        entry: WindowsEnumeratedChildV1,
        *,
        delete: bool = False,
    ) -> int:
        is_directory = bool(entry.attributes & 0x10)
        child = self._open_relative(
            parent,
            entry.name,
            directory=is_directory,
            delete=delete,
        )
        try:
            identity, _links, _attributes = self._verify(
                child,
                directory=is_directory,
                component=entry.name,
            )
            if identity.file_id != entry.file_id:
                raise NoFollowPathError(
                    f"directory child changed before acquisition: {entry.name}"
                )
            return child
        except BaseException:
            self._close_handle(self._wintypes.HANDLE(child))
            raise

    def iter_directory(
        self, parts: tuple[str, ...]
    ) -> Iterator[ContainedDirectoryEntry]:
        directory, close_directory = self._open_directory(parts)
        try:
            for entry in self._iter_names(directory):
                is_directory = bool(entry.attributes & 0x10)
                child = self._open_enumerated_child(directory, entry)
                self._close_handle(self._wintypes.HANDLE(child))
                yield ContainedDirectoryEntry(
                    entry.name,
                    is_directory,
                    not is_directory,
                )
        finally:
            if close_directory:
                self._close_handle(self._wintypes.HANDLE(directory))

    def _attest_directory(self, directory: int) -> None:
        for entry in self._iter_names(directory):
            is_directory = bool(entry.attributes & 0x10)
            child = self._open_enumerated_child(directory, entry)
            try:
                if is_directory:
                    self._attest_directory(child)
            finally:
                self._close_handle(self._wintypes.HANDLE(child))

    def _directory_identity(self, directory: int):
        identity, _links, _attributes = self._verify(
            directory,
            directory=True,
            component="authorized root",
        )
        return identity

    def _require_directory_identity(
        self,
        directory: int,
        expected: object | None,
    ) -> None:
        if expected is not None and self._directory_identity(directory) != expected:
            raise NoFollowPathError(
                "authorized directory identity changed before mutation"
            )

    def directory_identity(
        self,
        parts: tuple[str, ...],
        *,
        missing_ok: bool,
    ) -> object | None:
        directory, close_directory = self._open_directory_optional(
            parts,
            missing_ok=missing_ok,
        )
        if directory < 0:
            return None
        try:
            return self._directory_identity(directory)
        finally:
            if close_directory:
                self._close_handle(self._wintypes.HANDLE(directory))

    def persistent_directory_identity(
        self,
        parts: tuple[str, ...],
        *,
        missing_ok: bool,
    ) -> object | None:
        directory, close_directory = self._open_directory_optional(
            parts,
            missing_ok=missing_ok,
        )
        if directory < 0:
            return None
        try:
            from .file_identity import DirectoryIdentityV1

            return DirectoryIdentityV1(
                "windows",
                self._directory_identity(directory),
            )
        finally:
            if close_directory:
                self._close_handle(self._wintypes.HANDLE(directory))

    def attest_tree(self, parts: tuple[str, ...], *, missing_ok: bool) -> object | None:
        if not parts:
            directory, close_directory = self._root_handle, False
        else:
            parent, close_parent = self._open_directory_optional(
                parts[:-1],
                missing_ok=missing_ok,
            )
            if parent < 0:
                return None
            try:
                directory = self._open_relative(
                    parent,
                    parts[-1],
                    directory=True,
                    missing_ok=missing_ok,
                )
            finally:
                if close_parent:
                    self._close_handle(self._wintypes.HANDLE(parent))
            if directory < 0:
                return None
            close_directory = True
        try:
            identity = self._directory_identity(directory)
            self._attest_directory(directory)
            return identity
        finally:
            if close_directory:
                self._close_handle(self._wintypes.HANDLE(directory))

    def _first_name(self, directory: int) -> WindowsEnumeratedChildV1 | None:
        return next(self._iter_names(directory), None)

    def _name_present(self, directory: int, expected: str) -> bool:
        folded = expected.casefold()
        return any(
            entry.name.casefold() == folded for entry in self._iter_names(directory)
        )

    def _delete_on_close(self, handle: int) -> None:
        delete = self._wintypes.BOOL(True)
        if not self._set_information(
            self._wintypes.HANDLE(handle),
            4,
            ctypes.byref(delete),
            ctypes.sizeof(delete),
        ):
            code = ctypes.get_last_error()
            raise NoFollowPathError(
                f"safe handle deletion failed: Windows error {code}"
            )

    def _delete_directory_contents(self, directory: int) -> None:
        while True:
            entry = self._first_name(directory)
            if entry is None:
                return
            name = entry.name
            is_directory = bool(entry.attributes & 0x10)
            child = self._open_enumerated_child(
                directory,
                entry,
                delete=True,
            )
            try:
                if is_directory:
                    self._delete_directory_contents(child)
                self._delete_on_close(child)
            finally:
                self._close_handle(self._wintypes.HANDLE(child))
            if self._name_present(directory, name):
                raise NoFollowPathError(
                    f"deleted entry remains visible through its acquired parent: {name}"
                )

    def _open_mutation_parent(self, parts: tuple[str, ...]) -> tuple[int, bool]:
        return self._open_directory(parts)

    def clear_tree(
        self,
        parts: tuple[str, ...],
        *,
        missing_ok: bool,
        expected_identity: object | None,
    ) -> None:
        parent, close_parent = self._open_mutation_parent(parts[:-1])
        try:
            directory = self._open_relative(
                parent,
                parts[-1],
                directory=True,
                missing_ok=missing_ok,
            )
        finally:
            if close_parent:
                self._close_handle(self._wintypes.HANDLE(parent))
        if directory < 0:
            if expected_identity is not None:
                raise NoFollowPathError(
                    "authorized directory disappeared before mutation"
                )
            return
        try:
            self._require_directory_identity(directory, expected_identity)
            self._delete_directory_contents(directory)
        finally:
            self._close_handle(self._wintypes.HANDLE(directory))

    def remove_tree(
        self,
        parts: tuple[str, ...],
        *,
        missing_ok: bool,
        expected_identity: object | None,
        expected_parent_identity: object | None,
    ) -> None:
        parent, close_parent = self._open_mutation_parent(parts[:-1])
        directory = -1
        try:
            self._require_directory_identity(parent, expected_parent_identity)
            directory = self._open_relative(
                parent,
                parts[-1],
                directory=True,
                delete=True,
                missing_ok=missing_ok,
            )
            if directory < 0:
                if expected_identity is not None:
                    raise NoFollowPathError(
                        "authorized directory disappeared before mutation"
                    )
                return
            self._require_directory_identity(directory, expected_identity)
            self._delete_directory_contents(directory)
            self._delete_on_close(directory)
            self._close_handle(self._wintypes.HANDLE(directory))
            directory = -1
            if self._name_present(parent, parts[-1]):
                raise NoFollowPathError(
                    "deleted tree remains visible through its acquired parent"
                )
        finally:
            if directory >= 0:
                self._close_handle(self._wintypes.HANDLE(directory))
            if close_parent:
                self._close_handle(self._wintypes.HANDLE(parent))

    def unlink_regular(
        self,
        parts: tuple[str, ...],
        *,
        missing_ok: bool,
        expected_parent_identity: object | None,
        expected_identity: object | None = None,
    ) -> None:
        parent, close_parent = self._open_mutation_parent(parts[:-1])
        descriptor = -1
        try:
            self._require_directory_identity(parent, expected_parent_identity)
            descriptor = self._open_relative(
                parent,
                parts[-1],
                directory=False,
                delete=True,
                missing_ok=missing_ok,
            )
            if descriptor < 0:
                return
            actual_identity, _links, _attributes = self._verify(
                descriptor,
                directory=False,
                component=parts[-1],
            )
            if expected_identity is not None and actual_identity != expected_identity:
                raise NoFollowPathError("regular-file identity changed before mutation")
            self._delete_on_close(descriptor)
            self._close_handle(self._wintypes.HANDLE(descriptor))
            descriptor = -1
            if self._name_present(parent, parts[-1]):
                raise NoFollowPathError(
                    "deleted file remains visible through its acquired parent"
                )
        finally:
            if descriptor >= 0:
                self._close_handle(self._wintypes.HANDLE(descriptor))
            if close_parent:
                self._close_handle(self._wintypes.HANDLE(parent))


class AcquiredRoot:
    """Keep one verified job-root handle for a complete read-only audit phase."""

    def __init__(self, job_root: Path) -> None:
        self.root = Path(os.path.abspath(os.fspath(job_root)))
        self._impl: _AcquiredRootImpl = (
            _WindowsAcquiredRoot(self.root)
            if os.name == "nt"
            else _PosixAcquiredRoot(self.root)
        )

    def __enter__(self) -> AcquiredRoot:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self._impl.close()

    @property
    def persistent_identity(self) -> object:
        return self._impl.persistent_identity()

    def directory_absent(self, path: Path) -> bool:
        parts = _relative_parts(self.root, Path(path), allow_root=False)
        return self._impl.attest_tree(parts, missing_ok=True) is None

    def open_regular(self, path: Path) -> BinaryIO:
        parts = _relative_parts(self.root, Path(path))
        return self._impl.open_regular(parts)

    def open_regular_update(self, path: Path) -> BinaryIO:
        parts = _relative_parts(self.root, Path(path))
        return self._impl.open_regular_update(parts)

    def open_regular_locked(self, path: Path) -> BinaryIO:
        parts = _relative_parts(self.root, Path(path))
        return self._impl.open_regular_locked(parts)

    def create_regular_exclusive(self, path: Path) -> tuple[BinaryIO, object]:
        parts = _relative_parts(self.root, Path(path))
        return self._impl.create_regular_exclusive(parts)

    def regular_identity(
        self,
        path: Path,
        *,
        missing_ok: bool = False,
    ) -> object | None:
        parts = _relative_parts(self.root, Path(path))
        return self._impl.regular_identity(parts, missing_ok=missing_ok)

    def persistent_regular_identity(
        self,
        path: Path,
        *,
        missing_ok: bool = False,
    ) -> object | None:
        try:
            handle = self.open_regular(path)
        except (NoFollowPathError, FileNotFoundError) as error:
            if missing_ok and _is_missing_path_error(error):
                return None
            raise
        try:
            from .file_identity import file_identity_from_open_handle

            return file_identity_from_open_handle(handle)
        finally:
            handle.close()

    def directory_identity(
        self,
        path: Path,
        *,
        missing_ok: bool = False,
    ) -> object | None:
        parts = _relative_parts(self.root, Path(path), allow_root=True)
        return self._impl.persistent_directory_identity(
            parts,
            missing_ok=missing_ok,
        )

    def unlink_owned_regular(
        self,
        path: Path,
        *,
        expected_identity: object,
        missing_ok: bool = False,
    ) -> None:
        parts = _relative_parts(self.root, Path(path))
        parent_identity = self._impl.directory_identity(
            parts[:-1],
            missing_ok=False,
        )
        self._impl.unlink_regular(
            parts,
            missing_ok=missing_ok,
            expected_parent_identity=parent_identity,
            expected_identity=expected_identity,
        )

    def open_regular_optional(self, path: Path) -> BinaryIO | None:
        try:
            return self.open_regular(path)
        except NoFollowPathError as error:
            if _is_missing_path_error(error):
                return None
            raise

    def iter_directory(self, path: Path) -> Iterator[ContainedDirectoryEntry]:
        parts = _relative_parts(self.root, Path(path), allow_root=True)
        yield from self._impl.iter_directory(parts)


def scan_fixed_regular_v1(
    job_root: Path,
    path: Path,
    *,
    expected_identity: object | None = None,
    expected_raw_sha256: str | None = None,
    expected_byte_count: int | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> AuthenticatedFixedFileV1:
    """Scan one link-count-one file with fixed memory and no-follow authority."""

    from .file_identity import file_identity_from_open_handle

    def check_cancelled() -> None:
        if cancelled is not None and cancelled():
            raise InterruptedError("fixed-file scan cancelled")

    candidate = Path(path)
    check_cancelled()
    with AcquiredRoot(job_root) as acquired:
        root_identity = acquired.persistent_identity
        parent_identity = acquired.directory_identity(candidate.parent)
        with acquired.open_regular(candidate) as handle:
            frozen_identity = file_identity_from_open_handle(handle)
            if frozen_identity.link_count != 1:
                raise NoFollowPathError("fixed file is not link-count-one")
            if expected_identity is not None and frozen_identity != expected_identity:
                raise NoFollowPathError("fixed file identity differs")
            before = os.fstat(handle.fileno())
            byte_count = int(before.st_size)
            if not 0 <= byte_count <= (1 << 64) - 1:
                raise NoFollowPathError("fixed file byte count exceeds u64")
            if expected_byte_count is not None and byte_count != expected_byte_count:
                raise NoFollowPathError("fixed file byte count differs")
            digest = hashlib.sha256()
            buffer = bytearray(FIXED_FILE_HASH_STREAM_BYTES)
            view = memoryview(buffer)
            handle.seek(0)
            remaining = byte_count
            try:
                while remaining:
                    check_cancelled()
                    requested = min(len(buffer), remaining)
                    measured = handle.readinto(view[:requested])
                    if (
                        type(measured) is not int
                        or measured <= 0
                        or measured > requested
                    ):
                        raise NoFollowPathError(
                            "fixed file ended before its authenticated length"
                        )
                    digest.update(view[:measured])
                    remaining -= measured
                check_cancelled()
                extra = handle.readinto(view[:1])
                if extra != 0:
                    raise NoFollowPathError(
                        "fixed file exceeds its authenticated length"
                    )
            finally:
                view.release()
                del view, buffer
            check_cancelled()
            after = os.fstat(handle.fileno())
            if (
                int(after.st_size) != byte_count
                or int(after.st_mtime_ns) != int(before.st_mtime_ns)
                or int(after.st_ctime_ns) != int(before.st_ctime_ns)
                or file_identity_from_open_handle(handle) != frozen_identity
                or acquired.persistent_regular_identity(candidate) != frozen_identity
            ):
                raise NoFollowPathError(
                    "fixed file changed during authenticated scan"
                )
        if (
            acquired.directory_identity(candidate.parent) != parent_identity
            or acquired.persistent_identity != root_identity
        ):
            raise NoFollowPathError(
                "fixed file namespace changed during authenticated scan"
            )
        raw_sha256 = digest.hexdigest()
        if expected_raw_sha256 is not None and raw_sha256 != expected_raw_sha256:
            raise NoFollowPathError("fixed file digest differs")
        return AuthenticatedFixedFileV1(
            identity=frozen_identity,
            raw_sha256=raw_sha256,
            byte_count=byte_count,
        )


class SafeTreeMutation:
    """Preflight fixed deletion roots, then mutate only through verified handles."""

    def __init__(
        self,
        job_root: Path,
        authorized_roots: tuple[Path, ...],
        *,
        acquired: AcquiredRoot | None = None,
    ) -> None:
        self._owns_acquired = acquired is None
        self._acquired = AcquiredRoot(job_root) if acquired is None else acquired
        expected_root = Path(os.path.abspath(os.fspath(job_root)))
        if self._acquired.root != expected_root:
            raise NoFollowPathError("tree mutation acquired the wrong job root")
        self._roots: dict[Path, tuple[tuple[str, ...], object | None]] = {}
        try:
            for root in authorized_roots:
                lexical = Path(os.path.abspath(os.fspath(root)))
                parts = _relative_parts(self._acquired.root, lexical)
                if lexical in self._roots:
                    continue
                identity = self._acquired._impl.attest_tree(parts, missing_ok=True)
                self._roots[lexical] = (parts, identity)
        except BaseException:
            if self._owns_acquired:
                self._acquired.close()
            raise

    def __enter__(self) -> SafeTreeMutation:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_acquired:
            self._acquired.close()

    def _authorized_root(self, root: Path) -> tuple[tuple[str, ...], object | None]:
        lexical = Path(os.path.abspath(os.fspath(root)))
        try:
            return self._roots[lexical]
        except KeyError as error:
            raise NoFollowPathError(
                f"tree mutation root was not preflighted: {root}"
            ) from error

    def _still_absent(self, parts: tuple[str, ...], identity: object | None) -> bool:
        if identity is not None:
            return False
        if self._acquired._impl.attest_tree(parts, missing_ok=True) is not None:
            raise NoFollowPathError(
                "absent authorized directory appeared before mutation"
            )
        return True

    def clear_root(self, root: Path) -> None:
        parts, identity = self._authorized_root(root)
        if self._still_absent(parts, identity):
            return
        self._acquired._impl.clear_tree(
            parts,
            missing_ok=False,
            expected_identity=identity,
        )

    def remove_root(self, root: Path) -> None:
        parts, identity = self._authorized_root(root)
        if self._still_absent(parts, identity):
            return
        self._acquired._impl.remove_tree(
            parts,
            missing_ok=False,
            expected_identity=identity,
            expected_parent_identity=None,
        )

    def capture_regular_child(self, root: Path, child: Path) -> ChildMutationEvidence:
        root_parts, root_identity = self._authorized_root(root)
        lexical_child = Path(os.path.abspath(os.fspath(child)))
        lexical_root = Path(os.path.abspath(os.fspath(root)))
        try:
            relative = lexical_child.relative_to(lexical_root)
        except ValueError as error:
            raise NoFollowPathError(
                "regular-file deletion escaped its authorized root"
            ) from error
        if not relative.parts:
            raise NoFollowPathError("regular-file deletion requires a child")
        child_parts = (*root_parts, *relative.parts)
        if root_identity is None:
            parent_identity = None
            identity = None
        else:
            parent_identity = self._acquired._impl.directory_identity(
                child_parts[:-1],
                missing_ok=True,
            )
            identity = (
                None
                if parent_identity is None
                else self._acquired._impl.regular_identity(
                    child_parts,
                    missing_ok=True,
                )
            )
        return ChildMutationEvidence(
            path=lexical_child,
            kind="regular",
            parent_identity=parent_identity,
            identity=identity,
        )

    def capture_child_tree(self, root: Path, child: Path) -> ChildMutationEvidence:
        root_parts, root_identity = self._authorized_root(root)
        lexical_child = Path(os.path.abspath(os.fspath(child)))
        lexical_root = Path(os.path.abspath(os.fspath(root)))
        if lexical_child.parent != lexical_root:
            raise NoFollowPathError("tree deletion is not a direct authorized child")
        child_parts = (*root_parts, lexical_child.name)
        identity = (
            None
            if root_identity is None
            else self._acquired._impl.directory_identity(
                child_parts,
                missing_ok=True,
            )
        )
        return ChildMutationEvidence(
            path=lexical_child,
            kind="directory",
            parent_identity=root_identity,
            identity=identity,
        )

    @staticmethod
    def _require_child_evidence(
        child: Path,
        evidence: ChildMutationEvidence,
        *,
        kind: str,
    ) -> None:
        lexical_child = Path(os.path.abspath(os.fspath(child)))
        if (
            not isinstance(evidence, ChildMutationEvidence)
            or evidence.path != lexical_child
            or evidence.kind != kind
        ):
            raise NoFollowPathError("child mutation evidence does not match its target")

    def unlink_child(
        self,
        root: Path,
        child: Path,
        *,
        evidence: ChildMutationEvidence,
        missing_ok: bool = True,
    ) -> None:
        root_parts, identity = self._authorized_root(root)
        if self._still_absent(root_parts, identity):
            return
        lexical_child = Path(os.path.abspath(os.fspath(child)))
        lexical_root = Path(os.path.abspath(os.fspath(root)))
        if lexical_child.parent != lexical_root:
            raise NoFollowPathError(
                "regular-file deletion is not a direct authorized child"
            )
        self._require_child_evidence(lexical_child, evidence, kind="regular")
        if evidence.parent_identity != identity:
            raise NoFollowPathError(
                "regular-file deletion parent changed after preflight"
            )
        if evidence.identity is None:
            if (
                self._acquired._impl.regular_identity(
                    (*root_parts, lexical_child.name),
                    missing_ok=True,
                )
                is not None
            ):
                raise NoFollowPathError("absent regular child appeared before mutation")
            return
        self._acquired._impl.unlink_regular(
            (*root_parts, lexical_child.name),
            missing_ok=missing_ok,
            expected_parent_identity=identity,
            expected_identity=evidence.identity,
        )

    def unlink_descendant(
        self,
        root: Path,
        child: Path,
        *,
        evidence: ChildMutationEvidence,
        missing_ok: bool = True,
    ) -> None:
        """Delete one regular descendant after reattesting its fixed root and parent."""

        root_parts, expected_root_identity = self._authorized_root(root)
        if self._still_absent(root_parts, expected_root_identity):
            return
        actual_root_identity = self._acquired._impl.directory_identity(
            root_parts,
            missing_ok=False,
        )
        if actual_root_identity != expected_root_identity:
            raise NoFollowPathError(
                "authorized directory identity changed before mutation"
            )
        lexical_child = Path(os.path.abspath(os.fspath(child)))
        lexical_root = Path(os.path.abspath(os.fspath(root)))
        try:
            relative = lexical_child.relative_to(lexical_root)
        except ValueError as error:
            raise NoFollowPathError(
                "regular-file deletion escaped its authorized root"
            ) from error
        if len(relative.parts) < 2:
            raise NoFollowPathError("descendant deletion requires a nested child")
        self._require_child_evidence(lexical_child, evidence, kind="regular")
        child_parts = (*root_parts, *relative.parts)
        parent_identity = self._acquired._impl.directory_identity(
            child_parts[:-1],
            missing_ok=missing_ok,
        )
        if parent_identity is None:
            if evidence.parent_identity is not None or evidence.identity is not None:
                raise NoFollowPathError("regular descendant parent disappeared")
            return
        if parent_identity != evidence.parent_identity:
            raise NoFollowPathError("regular descendant parent changed after preflight")
        if evidence.identity is None:
            if (
                self._acquired._impl.regular_identity(
                    child_parts,
                    missing_ok=True,
                )
                is not None
            ):
                raise NoFollowPathError(
                    "absent regular descendant appeared before mutation"
                )
            return
        self._acquired._impl.unlink_regular(
            child_parts,
            missing_ok=missing_ok,
            expected_parent_identity=parent_identity,
            expected_identity=evidence.identity,
        )

    def remove_child_tree(
        self,
        root: Path,
        child: Path,
        *,
        evidence: ChildMutationEvidence,
        missing_ok: bool = True,
    ) -> None:
        root_parts, identity = self._authorized_root(root)
        if self._still_absent(root_parts, identity):
            return
        lexical_child = Path(os.path.abspath(os.fspath(child)))
        lexical_root = Path(os.path.abspath(os.fspath(root)))
        if lexical_child.parent != lexical_root:
            raise NoFollowPathError("tree deletion is not a direct authorized child")
        self._require_child_evidence(lexical_child, evidence, kind="directory")
        if evidence.parent_identity != identity:
            raise NoFollowPathError("child-tree parent changed after preflight")
        child_parts = (*root_parts, lexical_child.name)
        if evidence.identity is None:
            if (
                self._acquired._impl.directory_identity(
                    child_parts,
                    missing_ok=True,
                )
                is not None
            ):
                raise NoFollowPathError("absent child tree appeared before mutation")
            return
        self._acquired._impl.remove_tree(
            child_parts,
            missing_ok=missing_ok,
            expected_identity=evidence.identity,
            expected_parent_identity=identity,
        )


class _OwnedBinaryHandle:
    def __init__(self, handle: BinaryIO, acquired: AcquiredRoot) -> None:
        self._handle = handle
        self._acquired = acquired

    def __getattr__(self, name: str):
        return getattr(self._handle, name)

    def __enter__(self) -> _OwnedBinaryHandle:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._handle.close()
        finally:
            self._acquired.close()


def open_regular_beneath(job_root: Path, path: Path) -> BinaryIO:
    """Open one regular file without following any relative path component."""

    acquired = AcquiredRoot(job_root)
    try:
        handle = acquired.open_regular(path)
    except BaseException:
        acquired.close()
        raise
    return _OwnedBinaryHandle(handle, acquired)  # type: ignore[return-value]
