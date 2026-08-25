"""Cross-platform durable same-directory publication primitives."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import BinaryIO, Callable

from .file_identity import (
    DirectoryIdentityV1,
    PersistedFileIdentityV1,
    capture_directory_identity,
    capture_file_identity,
)
from .nofollow_io import AcquiredRoot, NoFollowPathError
from .reservation_protocol import ReservationConflictError, same_parent_replace_v1


class OwnedTemporaryCollisionError(FileExistsError):
    """A deterministic transaction temporary already exists without ownership."""


class OwnedTemporaryIdentityError(RuntimeError):
    """A transaction temporary no longer names the file created by this invocation."""


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def publication_temporary_path(path: Path) -> Path:
    """Return the deterministic sibling name proposed by a target transaction."""

    target = Path(path)
    return target.with_name(f".{target.name}.depth-surge-publish-v1.tmp")


def capture_optional_publication_target_identity(
    path: Path,
) -> PersistedFileIdentityV1 | None:
    """Freeze one ordinary target identity, or durable absence, without following links."""

    with AcquiredRoot(path.parent) as acquired:
        if acquired.regular_identity(path, missing_ok=True) is None:
            return None
    return capture_file_identity(path)


@dataclass
class OwnedSiblingTemporary:
    """Exclusive no-follow temporary whose cleanup remains bound to create identity."""

    target: Path
    path: Path
    handle: BinaryIO
    identity: object
    _acquired: AcquiredRoot
    source_identity: PersistedFileIdentityV1
    target_identity: PersistedFileIdentityV1 | None
    parent_identity: DirectoryIdentityV1
    _published: bool = False
    _closed: bool = False

    @classmethod
    def create(
        cls,
        target: Path,
        *,
        temporary: Path | None = None,
    ) -> "OwnedSiblingTemporary":
        target = _absolute(Path(target))
        path = _absolute(
            publication_temporary_path(target) if temporary is None else Path(temporary)
        )
        if path.parent != target.parent or path == target:
            raise ValueError("owned temporary must be a distinct same-directory sibling")
        acquired = AcquiredRoot(target.parent)
        try:
            handle, identity = acquired.create_regular_exclusive(path)
        except FileExistsError as error:
            acquired.close()
            raise OwnedTemporaryCollisionError(
                f"transaction temporary already exists: {path}"
            ) from error
        except BaseException:
            acquired.close()
            raise
        try:
            source_identity = capture_file_identity(path)
            parent_identity = capture_directory_identity(target.parent)
            current_target = acquired.regular_identity(target, missing_ok=True)
            target_identity = (
                None if current_target is None else capture_file_identity(target)
            )
        except BaseException:
            handle.close()
            try:
                acquired.unlink_owned_regular(
                    path,
                    expected_identity=identity,
                    missing_ok=True,
                )
            finally:
                acquired.close()
            raise
        return cls(
            target,
            path,
            handle,
            identity,
            acquired,
            source_identity,
            target_identity,
            parent_identity,
        )

    def sync(self) -> None:
        if self._closed:
            raise ValueError("owned temporary is already closed")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        if not self._closed:
            self.handle.close()
            self._closed = True

    def assert_current_identity(self) -> None:
        self.close()
        try:
            actual = self._acquired.regular_identity(self.path, missing_ok=True)
        except NoFollowPathError as error:
            raise OwnedTemporaryIdentityError(
                f"transaction temporary became unsafe: {self.path}"
            ) from error
        if actual != self.identity:
            raise OwnedTemporaryIdentityError(
                f"transaction temporary identity changed: {self.path}"
            )

    def replace(self, *, after_replace: Callable[[], None] | None = None) -> None:
        self.assert_current_identity()

        def replaced() -> None:
            self._published = True
            if after_replace is not None:
                after_replace()

        durable_replace(
            self.path,
            self.target,
            expected_source_identity=self.source_identity,
            expected_target_identity=self.target_identity,
            expected_parent_identity=self.parent_identity,
            after_replace=replaced,
        )
        self._published = True

    def remove_published_target(self) -> None:
        self.close()
        if not self._published:
            return
        try:
            self._acquired.unlink_owned_regular(
                self.target,
                expected_identity=self.identity,
                missing_ok=True,
            )
        except NoFollowPathError as error:
            raise OwnedTemporaryIdentityError(
                f"published target changed before rollback: {self.target}"
            ) from error
        self._published = False

    def cleanup(self) -> None:
        self.close()
        try:
            if not self._published:
                try:
                    self._acquired.unlink_owned_regular(
                        self.path,
                        expected_identity=self.identity,
                        missing_ok=True,
                    )
                except NoFollowPathError as error:
                    raise OwnedTemporaryIdentityError(
                        f"transaction temporary changed before cleanup: {self.path}"
                    ) from error
        finally:
            self._acquired.close()


def _sync_posix_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_replace(
    source: Path,
    target: Path,
    *,
    expected_source_identity: PersistedFileIdentityV1 | None = None,
    expected_target_identity: PersistedFileIdentityV1 | None = None,
    expected_parent_identity: DirectoryIdentityV1 | None = None,
    after_replace: Callable[[], None] | None = None,
) -> None:
    """Atomically replace one sibling and durably publish its directory entry."""

    source = _absolute(source)
    target = _absolute(target)
    if source.parent != target.parent:
        raise ValueError("durable replacement requires same-directory siblings")
    source_identity = expected_source_identity or capture_file_identity(source)
    parent_identity = expected_parent_identity or capture_directory_identity(source.parent)
    if expected_source_identity is not None:
        try:
            if capture_file_identity(source) != expected_source_identity:
                raise OwnedTemporaryIdentityError(
                    f"transaction temporary identity changed: {source}"
                )
        except FileNotFoundError as error:
            raise OwnedTemporaryIdentityError(
                f"transaction temporary disappeared: {source}"
            ) from error
    current_target = capture_optional_publication_target_identity(target)
    if current_target != expected_target_identity:
        raise OwnedTemporaryIdentityError(
            f"publication target identity changed: {target}"
        )
    try:
        same_parent_replace_v1(
            source,
            target,
            expected_source_identity=source_identity,
            expected_target_identity=expected_target_identity,
            expected_parent_identity=parent_identity,
            replace_existing=expected_target_identity is not None,
            after_namespace_replace=after_replace,
        )
    except ReservationConflictError as error:
        raise OwnedTemporaryIdentityError(
            f"publication identity changed during replacement: {source} -> {target}"
        ) from error


def durable_sync_parent(path: Path) -> None:
    """Sync a parent after a non-replace namespace mutation on POSIX."""

    parent = _absolute(path)
    if os.name == "nt":
        raise OSError(
            "Windows namespace durability requires a write-through native operation"
        )
    _sync_posix_directory(parent)
