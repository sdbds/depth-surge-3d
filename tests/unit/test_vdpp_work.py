"""Safe startup cleanup for private VDPP memmaps."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.depth_surge_3d.io.job_lock import JobWriterLock
from src.depth_surge_3d.vdpp_work import cleanup_vdpp_private_work


def _work_dir(output_root: Path) -> Path:
    return output_root / "03_disparity_stabilized/.vdpp-work"


def test_cleanup_removes_only_validated_private_memmaps_and_is_idempotent(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "job"
    work = _work_dir(output_root)
    work.mkdir(parents=True)
    (work / "shot_0.raw.f32.mmap").write_bytes(b"raw")
    (work / "shot_123.raw.f32.mmap").write_bytes(b"raw")
    lock = JobWriterLock(output_root).acquire()
    try:
        cleanup_vdpp_private_work(output_root, lock)
        cleanup_vdpp_private_work(output_root, lock)
    finally:
        lock.release()

    assert not work.exists()
    assert (output_root / "03_disparity_stabilized").is_dir()


def test_cleanup_validates_every_entry_before_deleting_anything(tmp_path: Path) -> None:
    output_root = tmp_path / "job"
    work = _work_dir(output_root)
    work.mkdir(parents=True)
    owned = work / "shot_0.raw.f32.mmap"
    owned.write_bytes(b"raw")
    (work / "notes.txt").write_text("unknown", encoding="utf-8")
    lock = JobWriterLock(output_root).acquire()
    try:
        with pytest.raises(ValueError, match="unknown or unsafe"):
            cleanup_vdpp_private_work(output_root, lock)
    finally:
        lock.release()

    assert owned.is_file()


def test_cleanup_requires_the_matching_acquired_job_lock(tmp_path: Path) -> None:
    output_root = tmp_path / "job"
    other_root = tmp_path / "other"
    output_root.mkdir()
    other_root.mkdir()

    with pytest.raises(ValueError, match="acquired"):
        cleanup_vdpp_private_work(output_root, JobWriterLock(output_root))

    other_lock = JobWriterLock(other_root).acquire()
    try:
        with pytest.raises(ValueError, match="another output"):
            cleanup_vdpp_private_work(output_root, other_lock)
    finally:
        other_lock.release()


def test_cleanup_rejects_symlinked_work_directory_without_following_it(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "job"
    stable = output_root / "03_disparity_stabilized"
    target = tmp_path / "outside"
    stable.mkdir(parents=True)
    target.mkdir()
    outside = target / "shot_0.raw.f32.mmap"
    outside.write_bytes(b"keep")
    try:
        os.symlink(target, stable / ".vdpp-work", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")
    lock = JobWriterLock(output_root).acquire()
    try:
        with pytest.raises(ValueError, match="reparse|link"):
            cleanup_vdpp_private_work(output_root, lock)
    finally:
        lock.release()

    assert outside.read_bytes() == b"keep"


def test_importing_cleanup_in_a_fresh_interpreter_does_not_import_torch(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import depth_surge_3d.vdpp_work; "
                "raise SystemExit(1 if 'torch' in sys.modules else 0)"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
    )
    assert completed.returncode == 0
