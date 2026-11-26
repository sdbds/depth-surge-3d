"""Portable atomic same-directory file publication primitives."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import secrets
from typing import Any, BinaryIO, Callable

from .file_identity import PersistedFileIdentityV1
from .nofollow_io import AcquiredRoot, NoFollowPathError


class OwnedTemporaryCollisionError(FileExistsError):
    """A transaction temporary already exists."""


class OwnedTemporaryIdentityError(RuntimeError):
    """A transaction temporary changed unexpectedly before publication."""


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def publication_temporary_path(path: Path | str) -> Path:
    """Return a unique sibling temporary path in the same directory."""

    target = Path(path)
    return target.with_name(f".{target.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")


@dataclass
class OwnedSiblingTemporary:
    """Same-directory temporary file published atomically via ``os.replace``."""

    target: Path
    path: Path
    handle: BinaryIO
    identity: PersistedFileIdentityV1
    _published: bool = False
    _closed: bool = False

    @classmethod
    def create(
        cls,
        target: Path | str,
        *,
        temporary: Path | str | None = None,
    ) -> "OwnedSiblingTemporary":
        abs_target = _absolute(target)
        abs_temp = (
            _absolute(publication_temporary_path(abs_target))
            if temporary is None
            else _absolute(temporary)
        )
        if abs_temp.parent != abs_target.parent or abs_temp == abs_target:
            raise ValueError("owned temporary must be a distinct same-directory sibling")
        abs_target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.lstat(abs_temp)
        except FileNotFoundError:
            pass
        else:
            raise OwnedTemporaryCollisionError(f"transaction temporary already exists: {abs_temp}")
        with AcquiredRoot(abs_target.parent) as acquired:
            try:
                handle, identity = acquired.create_regular_exclusive(abs_temp)
            except FileExistsError as error:
                raise OwnedTemporaryCollisionError(
                    f"transaction temporary already exists: {abs_temp}"
                ) from error
        return cls(
            target=abs_target,
            path=abs_temp,
            handle=handle,
            identity=identity,
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

    def replace(
        self,
        *,
        sync: bool = False,
        after_replace: Callable[[], None] | None = None,
    ) -> None:
        if sync and not self._closed:
            self.sync()
        self.close()
        try:
            meta = os.lstat(self.path)
        except FileNotFoundError as error:
            raise OwnedTemporaryIdentityError(
                f"transaction temporary disappeared: {self.path}"
            ) from error
        if (int(meta.st_dev), int(meta.st_ino)) != (self.identity.device, self.identity.inode):
            raise OwnedTemporaryIdentityError(
                f"transaction temporary identity changed: {self.path}"
            )
        durable_replace(self.path, self.target, after_replace=after_replace)
        self._published = True

    def cleanup(self) -> None:
        self.close()
        if self._published:
            return
        try:
            meta = os.lstat(self.path)
            if (int(meta.st_dev), int(meta.st_ino)) == (self.identity.device, self.identity.inode):
                self.path.unlink(missing_ok=True)
        except OSError:
            pass


def durable_replace(
    source: Path | str,
    target: Path | str,
    *,
    after_replace: Callable[[], None] | None = None,
    **_ignored: object,
) -> None:
    """Atomically replace ``target`` with same-directory sibling ``source``."""

    abs_source = _absolute(source)
    abs_target = _absolute(target)
    if abs_source.parent != abs_target.parent:
        raise ValueError("atomic replacement requires same-directory siblings")
    try:
        os.replace(abs_source, abs_target)
    except OSError as error:
        raise OwnedTemporaryIdentityError(
            f"failed to atomically replace {abs_target} from {abs_source}: {error}"
        ) from error
    if after_replace is not None:
        after_replace()


def atomic_write_bytes(
    path: Path | str,
    data: bytes | bytearray | memoryview,
    *,
    sync: bool = False,
) -> Path:
    """Write ``data`` to a sibling temporary and atomically replace ``path``."""

    target = _absolute(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    owned = OwnedSiblingTemporary.create(target)
    try:
        view = memoryview(data)
        written = owned.handle.write(view)
        if written != len(view):
            raise OSError(f"short write to {target}")
        if sync:
            owned.sync()
        owned.replace()
    except NoFollowPathError:
        raise
    finally:
        owned.cleanup()
    return target


def atomic_write_json(
    path: Path | str,
    payload: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = True,
    sync: bool = True,
) -> Path:
    """Serialize ``payload`` as UTF-8 JSON and publish atomically."""

    if indent is None:
        text = json.dumps(payload, sort_keys=sort_keys, separators=(",", ":"), ensure_ascii=False)
    else:
        text = json.dumps(payload, indent=indent, sort_keys=sort_keys, ensure_ascii=False)
    return atomic_write_bytes(path, text.encode("utf-8"), sync=sync)
