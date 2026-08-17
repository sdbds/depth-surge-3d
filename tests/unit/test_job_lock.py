"""OS-backed single-writer job lock tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.depth_surge_3d.io.job_lock import JobAlreadyLockedError, JobWriterLock


def test_second_writer_fails_fast_and_reports_owner(tmp_path: Path) -> None:
    first = JobWriterLock(tmp_path)
    first.acquire()
    try:
        with pytest.raises(JobAlreadyLockedError, match="already being processed") as caught:
            JobWriterLock(tmp_path).acquire()

        assert caught.value.owner["pid"] > 0
        assert caught.value.owner["hostname"]
        assert caught.value.owner["process_start_identity"]
        assert caught.value.owner["acquired_at"]
    finally:
        first.release()


def test_stale_diagnostics_do_not_block_an_os_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / ".depth-surge.lock"
    lock_path.write_text('{"pid": 999999, "hostname": "stale"}', encoding="utf-8")

    with JobWriterLock(tmp_path) as acquired:
        with lock_path.open("rb") as handle:
            handle.seek(1)
            diagnostics = json.loads(handle.read().decode("utf-8"))
        assert diagnostics["pid"] == acquired.owner["pid"]
        assert diagnostics["hostname"] == acquired.owner["hostname"]


def test_release_is_idempotent_and_lock_can_be_reacquired(tmp_path: Path) -> None:
    lock = JobWriterLock(tmp_path)
    lock.acquire()
    lock.release()
    lock.release()

    with JobWriterLock(tmp_path) as reacquired:
        assert reacquired.is_acquired is True


def test_lock_requires_an_existing_output_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="output directory"):
        JobWriterLock(missing).acquire()

    assert not missing.exists()
