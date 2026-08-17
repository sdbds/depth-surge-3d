"""Pinned and integrity-checked VDPP checkpoint resolution."""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator, cast


VDPP_UPSTREAM_REPOSITORY = "https://github.com/injun-baek/VDPP"
VDPP_UPSTREAM_RELEASE = "v1.0"
VDPP_UPSTREAM_REVISION = "73cc2b4dc6b3b5cfb2e37f51e452461e03fe26f5"
VDPP_CHECKPOINT_URL = "https://github.com/injun-baek/VDPP/releases/download/v1.0/vdpp.pth"
VDPP_CHECKPOINT_SIZE = 116485370
VDPP_CHECKPOINT_SHA256 = "7368315b126093f0335147f42a1920f255d529613bfffc5c6cf4ef832deb73a7"
VDPP_DOWNLOAD_TIMEOUT_SECONDS = 120
VDPP_DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_is_valid(path: Path) -> bool:
    """Return true only for the exact pinned checkpoint bytes."""

    try:
        return (
            path.stat().st_size == VDPP_CHECKPOINT_SIZE and _sha256(path) == VDPP_CHECKPOINT_SHA256
        )
    except OSError:
        return False


@contextmanager
def _artifact_lock(path: Path) -> Iterator[None]:
    """Serialize one checkpoint download and revalidation across processes."""

    handle: BinaryIO = path.open("a+b", buffering=0)
    if path.stat().st_size == 0:
        handle.write(b"\n")
        handle.flush()
    handle.seek(0)
    try:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            handle.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _open_url(url: str, *, timeout: float):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "depth-surge-3d-vdpp/1"},
    )
    return urllib.request.urlopen(request, timeout=timeout)


def ensure_vdpp_checkpoint(  # noqa: C901
    models_dir: Path | str,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
    opener: Callable[..., object] | None = None,
    timeout: float = VDPP_DOWNLOAD_TIMEOUT_SECONDS,
    chunk_size: int = VDPP_DOWNLOAD_CHUNK_SIZE,
) -> Path:
    """Return the exact pinned checkpoint, downloading only when required."""

    if chunk_size < 1:
        raise ValueError("VDPP download chunk size must be positive")
    final_path = Path(models_dir) / "VDPP" / "vdpp.pth"
    if checkpoint_is_valid(final_path):
        return final_path

    final_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = final_path.with_suffix(final_path.suffix + ".lock")
    with _artifact_lock(lock_path):
        if checkpoint_is_valid(final_path):
            return final_path

        part_path = final_path.with_suffix(final_path.suffix + ".part")
        part_path.unlink(missing_ok=True)
        selected_opener = opener or _open_url
        try:
            response_context = cast(
                Any,
                selected_opener(VDPP_CHECKPOINT_URL, timeout=timeout),
            )
            with response_context as response:
                headers = getattr(response, "headers", {})
                content_length = headers.get("Content-Length") if headers is not None else None
                if content_length is not None and int(content_length) != VDPP_CHECKPOINT_SIZE:
                    raise ValueError(
                        "VDPP checkpoint Content-Length does not match the pinned size"
                    )
                downloaded = 0
                digest = hashlib.sha256()
                with part_path.open("wb") as handle:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        handle.write(chunk)
                        digest.update(chunk)
                        downloaded += len(chunk)
                        if downloaded > VDPP_CHECKPOINT_SIZE:
                            raise ValueError("VDPP checkpoint exceeds the pinned size")
                        if progress_callback is not None:
                            progress_callback(downloaded, VDPP_CHECKPOINT_SIZE)
                    handle.flush()
                    os.fsync(handle.fileno())
                if downloaded != VDPP_CHECKPOINT_SIZE:
                    raise ValueError(
                        f"VDPP checkpoint size is {downloaded}; expected {VDPP_CHECKPOINT_SIZE}"
                    )
                if digest.hexdigest() != VDPP_CHECKPOINT_SHA256:
                    raise ValueError("VDPP checkpoint SHA-256 does not match the pinned artifact")
            os.replace(part_path, final_path)
        except BaseException:
            part_path.unlink(missing_ok=True)
            raise

        if not checkpoint_is_valid(final_path):
            raise ValueError("Published VDPP checkpoint failed final integrity verification")
        return final_path
