"""Lightweight, lock-guarded cleanup of private VDPP work files."""

from __future__ import annotations

import os
from pathlib import Path
import re
import stat
from typing import TYPE_CHECKING, Any

from .core.constants import INTERMEDIATE_DIRS

if TYPE_CHECKING:
    from .io.job_lock import JobWriterLock


_OWNED_WORK_NAME = re.compile(r"^shot_[0-9]+[.]raw[.]f32[.]mmap$")
_REPARSE_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _is_reparse(info: os.stat_result) -> bool:
    attributes = int(getattr(info, "st_file_attributes", 0))
    return stat.S_ISLNK(info.st_mode) or bool(attributes & _REPARSE_ATTRIBUTE)


def _require_ordinary_directory(path: Path, description: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise OSError(f"Could not inspect {description}: {path}") from exc
    if _is_reparse(info):
        raise ValueError(f"{description} must not be a link or reparse point: {path}")
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{description} must be an ordinary directory: {path}")


def cleanup_vdpp_private_work(
    output_root: Path,
    job_lock: "JobWriterLock | Any",
) -> None:
    """Remove only recognized stale memmaps under the lock's exact job root."""

    root = Path(output_root).resolve()
    if not bool(getattr(job_lock, "is_acquired", False)):
        raise ValueError("VDPP private-work cleanup requires an acquired job writer lock")
    lock_root = getattr(job_lock, "output_dir", None)
    if lock_root is None or Path(lock_root).resolve() != root:
        raise ValueError("VDPP private-work cleanup lock belongs to another output directory")

    stable_root = root / INTERMEDIATE_DIRS["disparity_stabilized"]
    if not os.path.lexists(stable_root):
        return
    _require_ordinary_directory(stable_root, "VDPP stabilized directory")
    work_dir = stable_root / ".vdpp-work"
    if not os.path.lexists(work_dir):
        return
    _require_ordinary_directory(work_dir, "VDPP private-work directory")

    owned_files: list[Path] = []
    try:
        entries = list(os.scandir(work_dir))
    except OSError as exc:
        raise OSError(f"Could not inspect VDPP private-work directory: {work_dir}") from exc
    for entry in entries:
        path = Path(entry.path)
        try:
            info = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise OSError(f"Could not inspect VDPP private-work entry: {path}") from exc
        if (
            _is_reparse(info)
            or not stat.S_ISREG(info.st_mode)
            or _OWNED_WORK_NAME.fullmatch(entry.name) is None
        ):
            raise ValueError(
                f"VDPP private-work directory contains an unknown or unsafe entry: {path}"
            )
        owned_files.append(path)

    for path in owned_files:
        path.unlink()
    work_dir.rmdir()
