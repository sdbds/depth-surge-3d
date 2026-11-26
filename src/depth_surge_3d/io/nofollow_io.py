"""Portable containment-checked, no-symlink filesystem I/O."""

from __future__ import annotations

from dataclasses import dataclass
import io
import os
from pathlib import Path
import stat
from typing import Iterator

from .file_identity import (
    DirectoryIdentityV1,
    PersistedFileIdentityV1,
    _is_reparse_or_link,
    capture_directory_identity,
)


class NoFollowPathError(OSError):
    """Raised when a path escapes its root or traverses a symlink/reparse point."""


@dataclass(frozen=True, slots=True)
class ContainedDirectoryEntry:
    """Lightweight directory entry yielded without following symlinks."""

    name: str
    path: Path
    is_regular: bool
    is_directory: bool
    is_symlink: bool


def _abs_path(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _relative_parts(root: Path, target: Path) -> tuple[str, ...]:
    abs_root = _abs_path(root)
    abs_target = _abs_path(target)
    try:
        rel = abs_target.relative_to(abs_root)
    except ValueError as error:
        raise NoFollowPathError(f"path escapes root {abs_root}: {abs_target}") from error
    parts = rel.parts
    if any(part in {".", ".."} or "\x00" in part for part in parts):
        raise NoFollowPathError(f"path contains noncanonical component: {abs_target}")
    return parts


def _assert_no_link_chain(root: Path, target: Path, *, allow_missing_leaf: bool = False) -> Path:
    abs_root = _abs_path(root)
    parts = _relative_parts(abs_root, target)
    current = abs_root
    try:
        root_meta = os.lstat(current)
    except OSError as error:
        raise NoFollowPathError(f"root directory is unavailable: {current}") from error
    if _is_reparse_or_link(root_meta) or not stat.S_ISDIR(root_meta.st_mode):
        raise NoFollowPathError(f"root is not a non-link directory: {current}")

    for index, part in enumerate(parts):
        current = current / part
        is_leaf = index == len(parts) - 1
        try:
            meta = os.lstat(current)
        except FileNotFoundError:
            if is_leaf and allow_missing_leaf:
                return current
            raise
        if _is_reparse_or_link(meta):
            raise NoFollowPathError(f"refusing symlink or reparse point: {current}")
        if not is_leaf and not stat.S_ISDIR(meta.st_mode):
            raise NoFollowPathError(f"path component is not a directory: {current}")
    return current


class AcquiredRoot:
    """Scoped root directory handle enforcing containment and no-follow access."""

    def __init__(self, root: Path | str) -> None:
        self.root = _abs_path(root)
        self.persistent_identity = capture_directory_identity(self.root)

    def __enter__(self) -> "AcquiredRoot":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        return None

    def directory_identity(self, path: Path | str) -> DirectoryIdentityV1:
        checked = _assert_no_link_chain(self.root, Path(path))
        return capture_directory_identity(checked)

    def directory_absent(self, path: Path | str) -> bool:
        try:
            checked = _assert_no_link_chain(self.root, Path(path), allow_missing_leaf=True)
        except FileNotFoundError:
            return True
        try:
            meta = os.lstat(checked)
        except FileNotFoundError:
            return True
        if _is_reparse_or_link(meta):
            raise NoFollowPathError(f"expected absent directory is a symlink: {checked}")
        return False

    def regular_identity(
        self, path: Path | str, *, missing_ok: bool = False
    ) -> PersistedFileIdentityV1 | None:
        try:
            checked = _assert_no_link_chain(self.root, Path(path), allow_missing_leaf=missing_ok)
            meta = os.lstat(checked)
        except FileNotFoundError:
            if missing_ok:
                return None
            raise
        if _is_reparse_or_link(meta) or not stat.S_ISREG(meta.st_mode):
            raise NoFollowPathError(f"path is not a regular non-link file: {checked}")
        return PersistedFileIdentityV1.from_stat(meta)

    def persistent_regular_identity(
        self, path: Path | str, *, missing_ok: bool = False
    ) -> PersistedFileIdentityV1 | None:
        return self.regular_identity(path, missing_ok=missing_ok)

    def open_regular(self, path: Path | str, *, buffered: bool = True) -> io.BufferedReader:
        checked = _assert_no_link_chain(self.root, Path(path))
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(checked, flags)
        try:
            meta = os.fstat(fd)
            if _is_reparse_or_link(meta) or not stat.S_ISREG(meta.st_mode):
                raise NoFollowPathError(f"path is not a regular file: {checked}")
            buffering = -1 if buffered else 0
            return os.fdopen(fd, "rb", buffering=buffering)  # type: ignore[return-value]
        except BaseException:
            os.close(fd)
            raise

    def create_regular_exclusive(
        self, path: Path | str
    ) -> tuple[io.BufferedWriter, PersistedFileIdentityV1]:
        checked = _assert_no_link_chain(self.root, Path(path), allow_missing_leaf=True)
        flags = (
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        fd = os.open(checked, flags, 0o644)
        try:
            meta = os.fstat(fd)
            identity = PersistedFileIdentityV1.from_stat(meta)
            handle = os.fdopen(fd, "wb")
            return handle, identity  # type: ignore[return-value]
        except BaseException:
            os.close(fd)
            try:
                os.unlink(checked)
            except OSError:
                pass
            raise

    def unlink_owned_regular(
        self,
        path: Path | str,
        *,
        expected_identity: object | None = None,
        missing_ok: bool = False,
    ) -> bool:
        try:
            checked = _assert_no_link_chain(self.root, Path(path), allow_missing_leaf=missing_ok)
            meta = os.lstat(checked)
        except FileNotFoundError:
            if missing_ok:
                return False
            raise
        if _is_reparse_or_link(meta) or not stat.S_ISREG(meta.st_mode):
            raise NoFollowPathError(f"refusing to unlink non-regular file: {checked}")
        if expected_identity is not None and isinstance(expected_identity, PersistedFileIdentityV1):
            actual = PersistedFileIdentityV1.from_stat(meta)
            if (actual.device, actual.inode) != (
                expected_identity.device,
                expected_identity.inode,
            ):
                raise NoFollowPathError(f"file identity changed before unlink: {checked}")
        try:
            os.unlink(checked)
            return True
        except FileNotFoundError:
            if missing_ok:
                return False
            raise

    def iter_directory(self, directory: Path | str) -> Iterator[ContainedDirectoryEntry]:
        checked = _assert_no_link_chain(self.root, Path(directory))
        with os.scandir(checked) as entries:
            for entry in entries:
                try:
                    meta = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                is_link = _is_reparse_or_link(meta)
                yield ContainedDirectoryEntry(
                    name=entry.name,
                    path=Path(entry.path),
                    is_regular=not is_link and stat.S_ISREG(meta.st_mode),
                    is_directory=not is_link and stat.S_ISDIR(meta.st_mode),
                    is_symlink=is_link,
                )


def open_regular_beneath(root: Path | str, path: Path | str) -> io.BufferedReader:
    """Open a regular file guaranteed to reside under ``root`` without following symlinks."""

    with AcquiredRoot(root) as acquired:
        return acquired.open_regular(path)


class SafeTreeMutation:
    """Simple containment helper for creating directories and removing files under a root."""

    def __init__(self, root: Path | str) -> None:
        self.root = _abs_path(root)

    def ensure_dir(self, path: Path | str) -> Path:
        parts = _relative_parts(self.root, Path(path))
        current = self.root
        for part in parts:
            current = current / part
            current.mkdir(exist_ok=True)
        return _assert_no_link_chain(self.root, current)

    def unlink_regular(self, path: Path | str, *, missing_ok: bool = True) -> bool:
        with AcquiredRoot(self.root) as acquired:
            return acquired.unlink_owned_regular(path, missing_ok=missing_ok)
