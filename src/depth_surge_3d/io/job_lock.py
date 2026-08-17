"""Fail-fast OS-backed lock for one authoritative job writer."""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO


LOCK_FILE_NAME = ".depth-surge.lock"
_PROCESS_START_IDENTITY = f"{os.getpid()}:{time.time_ns()}:{time.monotonic_ns()}"


class JobAlreadyLockedError(RuntimeError):
    """Raised when another process owns the output-directory writer lock."""

    def __init__(self, path: Path, owner: dict[str, Any] | None = None) -> None:
        self.path = path
        self.owner = owner or {}
        owner_text = ""
        if self.owner:
            owner_text = (
                f" (pid={self.owner.get('pid', 'unknown')}, "
                f"host={self.owner.get('hostname', 'unknown')})"
            )
        super().__init__(f"Output directory is already being processed{owner_text}: {path.parent}")


class JobWriterLock:
    """Exclusive non-blocking lock held for the lifetime of a job mutation."""

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.path = self.output_dir / LOCK_FILE_NAME
        self._handle: BinaryIO | None = None
        self.owner: dict[str, Any] = {}

    @property
    def is_acquired(self) -> bool:
        return self._handle is not None

    def _read_owner(self, handle: BinaryIO) -> dict[str, Any]:
        try:
            handle.seek(1)
            decoded = json.loads(handle.read().decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return decoded if isinstance(decoded, dict) else {}

    @staticmethod
    def _try_lock(handle: BinaryIO) -> None:
        handle.seek(0)
        if sys.platform == "win32":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise BlockingIOError from exc
            return

        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise BlockingIOError from exc

    @staticmethod
    def _unlock(handle: BinaryIO) -> None:
        handle.seek(0)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            return

        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def acquire(self) -> "JobWriterLock":
        if self.is_acquired:
            return self
        if not self.output_dir.is_dir():
            raise FileNotFoundError(f"Job output directory does not exist: {self.output_dir}")

        handle = self.path.open("a+b", buffering=0)
        if self.path.stat().st_size == 0:
            handle.write(b"\n")
            handle.flush()
        try:
            self._try_lock(handle)
        except BlockingIOError as exc:
            owner = self._read_owner(handle)
            handle.close()
            raise JobAlreadyLockedError(self.path, owner) from exc

        owner = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "process_start_identity": _PROCESS_START_IDENTITY,
            "acquired_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            payload = b"\n" + json.dumps(owner, sort_keys=True).encode("utf-8")
            handle.seek(0)
            handle.truncate()
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            try:
                self._unlock(handle)
            finally:
                handle.close()
            raise

        self.owner = owner
        self._handle = handle
        return self

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            self._unlock(handle)
        finally:
            handle.close()

    def __enter__(self) -> "JobWriterLock":
        return self.acquire()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()
