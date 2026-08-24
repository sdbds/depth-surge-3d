from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.depth_surge_3d.rendering.quality import cleanup_transaction
from src.depth_surge_3d.rendering.quality.cleanup_transaction import (
    CleanupIdentityMismatchError,
    CleanupTransactionError,
    commit_payload_pruned,
    run_authorized_cleanup,
)
from src.depth_surge_3d.rendering.quality.diagnostic_transaction import (
    canonical_json_bytes,
    self_fingerprint,
)


def _complete_job(root: Path) -> None:
    diagnostics = root / "04_stereo_diagnostics"
    diagnostics.mkdir(parents=True)
    frames_jsonl = canonical_json_bytes({"frame_name": "frame_000000"}) + b"\n"
    summary = canonical_json_bytes({"schema_version": 1})
    (diagnostics / "stereo_coverage_frames.jsonl").write_bytes(frames_jsonl)
    (diagnostics / "stereo_coverage_summary.json").write_bytes(summary)
    metadata = {
        "schema_version": 1,
        "algorithm_version": "stereo-coverage-sidecar-v1",
        "status": "complete",
        "rgb_stage_fingerprint": "rgb",
        "source_guide_fingerprint": "guide",
        "native_geometry_fingerprint": None,
        "quality_input_manifest_sha256": None,
        "frame_names": ["frame_000000"],
        "render_shape": [2, 3],
        "render_mode": "fast",
        "geometry_mode": "relative",
        "occlusion_fill": "background",
        "mask_payloads_enabled": False,
        "stats_schema_version": 1,
        "frame_manifest_schema_version": 1,
        "ordered_frame_manifest_fingerprint": "a" * 64,
        "frames_jsonl_sha256": hashlib.sha256(frames_jsonl).hexdigest(),
        "summary_sha256": hashlib.sha256(summary).hexdigest(),
        "metric_clamp_summary_sha256": None,
        "pruned_from": None,
        "final_video_manifest_sha256": None,
        "prune_entries": None,
    }
    metadata["fingerprint"] = self_fingerprint(metadata)
    (diagnostics / "metadata.json").write_bytes(canonical_json_bytes(metadata))
    (root / "final_video_manifest.json").write_bytes(canonical_json_bytes({"schema_version": 4}))
    for relative in (
        "00_original_frames",
        "04_left_frames",
        "04_right_frames",
        "04_stereo_diagnostics/frames",
    ):
        directory = root / relative
        directory.mkdir(parents=True)
        (directory / "payload.bin").write_bytes(relative.encode("ascii"))


def test_payload_pruned_authorizes_only_persisted_identity_bound_roots(tmp_path: Path) -> None:
    _complete_job(tmp_path)
    retained = tmp_path / "not-an-intermediate.txt"
    retained.write_text("keep", encoding="ascii")

    metadata = commit_payload_pruned(tmp_path)

    persisted = json.loads(
        (tmp_path / "04_stereo_diagnostics" / "metadata.json").read_text(encoding="ascii")
    )
    assert persisted == metadata
    assert persisted["status"] == "payload_pruned"
    assert persisted["ordered_frame_manifest_fingerprint"] is None
    assert persisted["pruned_from"]["ordered_frame_manifest_fingerprint"] == "a" * 64
    assert [entry["relative_path"] for entry in persisted["prune_entries"]] == [
        "00_original_frames",
        "04_left_frames",
        "04_right_frames",
        "04_stereo_diagnostics/frames",
    ]

    removed = run_authorized_cleanup(tmp_path)

    assert removed == 8
    assert retained.read_text(encoding="ascii") == "keep"
    assert (tmp_path / "04_stereo_diagnostics" / "metadata.json").is_file()
    assert (tmp_path / "04_stereo_diagnostics" / "stereo_coverage_frames.jsonl").is_file()
    assert (tmp_path / "04_stereo_diagnostics" / "stereo_coverage_summary.json").is_file()
    assert (tmp_path / "final_video_manifest.json").is_file()
    assert run_authorized_cleanup(tmp_path) == 0


def test_payload_pruned_consumes_the_active_final_reservation(tmp_path: Path, monkeypatch) -> None:
    _complete_job(tmp_path)
    original_raw = (tmp_path / "04_stereo_diagnostics" / "metadata.json").read_bytes()
    original = json.loads(original_raw)
    planned = cleanup_transaction.plan_prune_markers_v1(tmp_path)
    entries = tuple(marker.prune_entry() for marker in planned)
    committed_payloads: list[bytes] = []

    class Reservation:
        index = SimpleNamespace(
            diagnostics_metadata_raw_sha256_at_preflight=hashlib.sha256(original_raw).hexdigest(),
            diagnostics_metadata_fingerprint_at_preflight=original["fingerprint"],
        )

        def commit_prune_markers(self):
            for marker in planned:
                target = tmp_path / marker.relative_path / marker.marker_name
                target.write_bytes(marker.payload)
            return entries

        def commit_payload_pruned_metadata(self, payload: bytes) -> None:
            committed_payloads.append(payload)
            (tmp_path / "04_stereo_diagnostics" / "metadata.json").write_bytes(payload)

    monkeypatch.setattr(
        cleanup_transaction,
        "_publish_marker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("standalone marker publication was used")
        ),
    )
    monkeypatch.setattr(
        cleanup_transaction,
        "_replace_metadata",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("standalone metadata replacement was used")
        ),
    )

    metadata = commit_payload_pruned(tmp_path, reservation=Reservation())

    assert len(committed_payloads) == 1
    assert json.loads(committed_payloads[0]) == metadata
    assert metadata["status"] == "payload_pruned"
    assert tuple(metadata["prune_entries"]) == entries


def test_cleanup_rejects_a_replacement_directory_without_touching_it(tmp_path: Path) -> None:
    _complete_job(tmp_path)
    metadata = commit_payload_pruned(tmp_path)
    original = tmp_path / "00_original_frames"
    displaced = tmp_path / "displaced"
    original.rename(displaced)
    original.mkdir()
    replacement = original / "must-survive.bin"
    replacement.write_bytes(b"replacement")

    with pytest.raises(CleanupIdentityMismatchError, match="identity differs"):
        run_authorized_cleanup(tmp_path, metadata=metadata)

    assert replacement.read_bytes() == b"replacement"
    assert (displaced / "payload.bin").is_file()


def test_prune_planning_rejects_a_bind_mount_identity_before_marker_publication(
    tmp_path: Path, monkeypatch
) -> None:
    _complete_job(tmp_path)
    root_mount = cleanup_transaction._InvocationMountIdentity("statx-mnt-id", 7)
    crossed_mount = cleanup_transaction._InvocationMountIdentity("statx-mnt-id", 8)
    calls = 0

    def mount_identity(_path: Path):
        nonlocal calls
        calls += 1
        return root_mount if calls == 1 else crossed_mount

    monkeypatch.setattr(
        cleanup_transaction,
        "_requires_posix_mount_identity",
        lambda: True,
    )
    monkeypatch.setattr(cleanup_transaction, "_path_mount_identity", mount_identity)

    with pytest.raises(CleanupIdentityMismatchError, match="crosses a mount"):
        cleanup_transaction.plan_prune_markers_v1(tmp_path)

    assert not list(tmp_path.rglob(".depth-surge-prune-v2-*.marker"))


def test_cleanup_resumes_marker_unlink_before_root_removal(tmp_path: Path) -> None:
    _complete_job(tmp_path)
    metadata = commit_payload_pruned(tmp_path)

    def crash(stage: str) -> None:
        if stage == "after_prune_marker_unlink_0":
            raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError, match="simulated crash"):
        run_authorized_cleanup(tmp_path, metadata=metadata, fault=crash)

    assert (tmp_path / "00_original_frames").is_dir()
    assert list((tmp_path / "00_original_frames").iterdir()) == []

    run_authorized_cleanup(tmp_path)

    assert not (tmp_path / "00_original_frames").exists()


def test_payload_pruned_rejects_damaged_retained_aggregates(tmp_path: Path) -> None:
    _complete_job(tmp_path)
    (tmp_path / "04_stereo_diagnostics" / "stereo_coverage_summary.json").write_bytes(b"damaged")

    with pytest.raises(Exception, match="summary"):
        commit_payload_pruned(tmp_path)

    assert (
        json.loads(
            (tmp_path / "04_stereo_diagnostics" / "metadata.json").read_text(encoding="ascii")
        )["status"]
        == "complete"
    )


@pytest.mark.parametrize(
    ("stage_key", "relative_path"),
    [
        ("frames", "."),
        ("stereo_diagnostics_frames", "04_stereo_diagnostics"),
    ],
)
def test_cleanup_rejects_forged_parent_or_retained_root_before_mutation(
    tmp_path: Path,
    monkeypatch,
    stage_key: str,
    relative_path: str,
) -> None:
    _complete_job(tmp_path)
    metadata = commit_payload_pruned(tmp_path)
    forged_root = tmp_path if relative_path == "." else tmp_path / relative_path
    entry = dict(metadata["prune_entries"][0])
    entry.update(
        {
            "stage_key": stage_key,
            "relative_path": relative_path,
            "directory_identity": cleanup_transaction._directory_identity(forged_root).to_object(),
        }
    )
    forged = {**metadata, "prune_entries": [entry]}
    forged["fingerprint"] = self_fingerprint(forged)
    sentinel = tmp_path / "final_video_manifest.json"
    sentinel_raw = sentinel.read_bytes()
    monkeypatch.setattr(
        cleanup_transaction,
        "_delete_descendants_no_follow",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cleanup mutation was reached")
        ),
    )

    with pytest.raises(CleanupTransactionError, match="authorization"):
        run_authorized_cleanup(tmp_path, metadata=forged)

    assert sentinel.read_bytes() == sentinel_raw
