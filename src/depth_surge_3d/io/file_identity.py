"""Cross-platform portable file and directory identity helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat
from typing import Any, BinaryIO


class PersistentIdentityError(RuntimeError):
    """Base error for file identity inspection failures."""


class PersistentIdentityCapabilityError(PersistentIdentityError):
    """Raised when a path does not expose a usable regular/directory identity."""


class PersistentIdentitySchemaError(PersistentIdentityError):
    """Raised when serialized identity metadata is malformed."""


def _is_reparse_or_link(metadata: os.stat_result) -> bool:
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attrs = getattr(metadata, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attrs & reparse)


@dataclass(frozen=True, slots=True)
class PersistedFileIdentityV1:
    """Portable identity snapshot derived from ``os.lstat`` / ``os.fstat``."""

    device: int
    inode: int
    size: int
    mtime_ns: int

    def to_object(self) -> dict[str, Any]:
        return {
            "platform": "windows" if os.name == "nt" else "posix",
            "device": self.device,
            "inode": self.inode,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
        }

    @classmethod
    def from_object(cls, value: object) -> "PersistedFileIdentityV1":
        if not isinstance(value, dict):
            raise PersistentIdentitySchemaError("file identity must be a dict")
        try:
            return cls(
                device=int(value["device"]),
                inode=int(value["inode"]),
                size=int(value["size"]),
                mtime_ns=int(value["mtime_ns"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise PersistentIdentitySchemaError("invalid file identity payload") from error

    @classmethod
    def from_stat(cls, metadata: os.stat_result) -> "PersistedFileIdentityV1":
        return cls(
            device=int(metadata.st_dev),
            inode=int(metadata.st_ino),
            size=int(metadata.st_size),
            mtime_ns=int(metadata.st_mtime_ns),
        )


@dataclass(frozen=True, slots=True)
class DirectoryIdentityV1:
    """Portable directory identity snapshot derived from ``os.lstat``."""

    device: int
    inode: int

    @property
    def volume_key(self) -> int:
        return self.device

    def to_object(self) -> dict[str, Any]:
        return {
            "platform": "windows" if os.name == "nt" else "posix",
            "device": self.device,
            "inode": self.inode,
        }

    @classmethod
    def from_object(cls, value: object) -> "DirectoryIdentityV1":
        if not isinstance(value, dict):
            raise PersistentIdentitySchemaError("directory identity must be a dict")
        try:
            return cls(device=int(value["device"]), inode=int(value["inode"]))
        except (KeyError, TypeError, ValueError) as error:
            raise PersistentIdentitySchemaError("invalid directory identity payload") from error

    @classmethod
    def from_stat(cls, metadata: os.stat_result) -> "DirectoryIdentityV1":
        return cls(device=int(metadata.st_dev), inode=int(metadata.st_ino))


def capture_file_identity(path: Path | str) -> PersistedFileIdentityV1:
    """Capture a regular non-link file identity using standard ``os.lstat``."""

    candidate = Path(path)
    metadata = os.lstat(candidate)
    if _is_reparse_or_link(metadata) or not stat.S_ISREG(metadata.st_mode):
        raise PersistentIdentityCapabilityError(f"path is not a regular non-link file: {candidate}")
    return PersistedFileIdentityV1.from_stat(metadata)


def capture_directory_identity(path: Path | str) -> DirectoryIdentityV1:
    """Capture a non-link directory identity using standard ``os.lstat``."""

    candidate = Path(path)
    metadata = os.lstat(candidate)
    if _is_reparse_or_link(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise PersistentIdentityCapabilityError(f"path is not a non-link directory: {candidate}")
    return DirectoryIdentityV1.from_stat(metadata)


def file_identity_from_open_handle(handle: BinaryIO) -> PersistedFileIdentityV1:
    """Capture the identity of an already-opened regular file handle."""

    metadata = os.fstat(handle.fileno())
    if not stat.S_ISREG(metadata.st_mode):
        raise PersistentIdentityCapabilityError("open handle is not a regular file")
    return PersistedFileIdentityV1.from_stat(metadata)
