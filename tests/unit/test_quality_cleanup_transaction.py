"""Fast O(N) containment-checked intermediate cleanup tests."""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from src.depth_surge_3d.rendering.quality.cleanup_transaction import (
    CleanupIdentityMismatchError,
    cleanup_status_for_error,
    commit_payload_pruned,
    delete_tree_contained,
    run_authorized_cleanup,
)


def test_run_authorized_cleanup_removes_intermediates_and_keeps_final_video(tmp_path: Path) -> None:
    final_mp4 = tmp_path / "output_sbs.mp4"
    final_mp4.write_bytes(b"mp4-bytes")
    settings_json = tmp_path / "processing-settings-v5.json"
    settings_json.write_text("{}", encoding="utf-8")

    for rel in (
        "00_original_frames",
        "04_left_frames",
        "04_right_frames",
        "04_stereo_diagnostics/frames/f1",
    ):
        d = tmp_path / rel
        d.mkdir(parents=True, exist_ok=True)
        (d / "frame_000001.png").write_bytes(b"png")
    summary = tmp_path / "04_stereo_diagnostics" / "summary.json"
    summary.write_text('{"status": "complete"}', encoding="utf-8")

    pruned = commit_payload_pruned(tmp_path)
    assert pruned["status"] == "payload_pruned"

    removed = run_authorized_cleanup(tmp_path)
    assert removed == 4
    assert final_mp4.is_file()
    assert settings_json.is_file()
    assert summary.is_file()
    assert not (tmp_path / "00_original_frames").exists()
    assert not (tmp_path / "04_left_frames").exists()
    assert not (tmp_path / "04_stereo_diagnostics" / "frames").exists()


def test_delete_tree_contained_rejects_escape_and_root(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(CleanupIdentityMismatchError):
        delete_tree_contained(job, outside)
    with pytest.raises(CleanupIdentityMismatchError):
        delete_tree_contained(job, job)
    assert (
        cleanup_status_for_error(CleanupIdentityMismatchError("x"))
        == "incomplete_identity_mismatch"
    )
    assert cleanup_status_for_error(OSError("x")) == "incomplete_error"


def test_delete_tree_contained_refuses_symlinks_when_supported(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_bytes(b"safe")
    link = job / "04_left_frames"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks unavailable")

    with pytest.raises(CleanupIdentityMismatchError):
        run_authorized_cleanup(job)
    assert (outside / "keep.txt").is_file()
