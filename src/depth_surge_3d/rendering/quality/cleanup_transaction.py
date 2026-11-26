"""Fast O(N) containment-checked intermediate cleanup after video publication."""

from __future__ import annotations

import os
from pathlib import Path
import stat
from typing import Any

from ...io.file_identity import _is_reparse_or_link
from ...io.nofollow_io import AcquiredRoot, NoFollowPathError


class CleanupIdentityMismatchError(RuntimeError):
    """A cleanup target escaped the job root or became a symlink/reparse point."""


class CleanupTransactionError(RuntimeError):
    """Cleanup of intermediate job directories failed."""


class CleanupCapabilityError(CleanupTransactionError):
    """Platform safety precondition for cleanup was violated."""


INTERMEDIATE_CLEANUP_DIRS = (
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
    "04_stereo_diagnostics/frames",
)


def _unlink_writable(path: Path) -> None:
    try:
        os.unlink(path)
    except PermissionError:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        os.unlink(path)


def _rmdir_writable(path: Path) -> None:
    try:
        os.rmdir(path)
    except PermissionError:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
        os.rmdir(path)


def delete_tree_contained(job_root: Path, target: Path) -> int:  # noqa: C901
    """Recursively delete ``target`` inside ``job_root`` in O(N) without following symlinks."""

    root = Path(os.path.abspath(os.fspath(job_root)))
    abs_target = Path(os.path.abspath(os.fspath(target)))
    try:
        abs_target.relative_to(root)
    except ValueError as error:
        raise CleanupIdentityMismatchError(
            f"cleanup target escapes job root {root}: {abs_target}"
        ) from error
    if abs_target == root:
        raise CleanupIdentityMismatchError("refusing to delete job root itself")

    try:
        meta = os.lstat(abs_target)
    except FileNotFoundError:
        return 0
    if _is_reparse_or_link(meta):
        raise CleanupIdentityMismatchError(
            f"refusing to delete symlink/reparse point: {abs_target}"
        )
    if stat.S_ISREG(meta.st_mode):
        _unlink_writable(abs_target)
        return 1
    if not stat.S_ISDIR(meta.st_mode):
        raise CleanupIdentityMismatchError(f"unsupported cleanup target type: {abs_target}")

    removed = 0
    stack: list[tuple[Path, bool]] = [(abs_target, False)]
    while stack:
        current, visited = stack.pop()
        if visited:
            _rmdir_writable(current)
            continue
        stack.append((current, True))
        with os.scandir(current) as entries:
            for entry in entries:
                entry_path = Path(entry.path)
                try:
                    entry_meta = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if _is_reparse_or_link(entry_meta):
                    raise CleanupIdentityMismatchError(
                        f"refusing symlink/reparse point during cleanup: {entry_path}"
                    )
                if stat.S_ISDIR(entry_meta.st_mode):
                    stack.append((entry_path, False))
                else:
                    _unlink_writable(entry_path)
                    removed += 1
    return removed


def commit_payload_pruned(job_root: Path, **_ignored: object) -> dict[str, Any]:
    """Mark diagnostics metadata as pruned when present."""

    from .diagnostic_transaction import mark_diagnostics_pruned

    return mark_diagnostics_pruned(Path(job_root))


def run_authorized_cleanup(
    job_root: Path,
    **_ignored: object,
) -> int:
    """Remove all standard intermediate stage directories inside ``job_root`` in O(N)."""

    root = Path(os.path.abspath(os.fspath(job_root)))
    try:
        with AcquiredRoot(root):
            removed = 0
            for rel in INTERMEDIATE_CLEANUP_DIRS:
                removed += delete_tree_contained(root, root / rel)
            return removed
    except NoFollowPathError as error:
        raise CleanupIdentityMismatchError(str(error)) from error


def cleanup_status_for_error(error: BaseException) -> str:
    if isinstance(error, CleanupIdentityMismatchError):
        return "incomplete_identity_mismatch"
    return "incomplete_error"
