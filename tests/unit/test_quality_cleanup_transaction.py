from __future__ import annotations

import errno
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import weakref

import pytest

from src.depth_surge_3d.io.durable_publish import (
    OwnedTemporaryCollisionError,
    publication_temporary_path,
)
from src.depth_surge_3d.rendering.quality import cleanup_transaction
from src.depth_surge_3d.rendering.quality.cleanup_transaction import (
    CleanupIdentityMismatchError,
    CleanupTransactionError,
    commit_payload_pruned,
    run_authorized_cleanup,
)
from src.depth_surge_3d.rendering.quality.diagnostic_transaction import (
    DiagnosticsMetadataProjection,
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
        "rgb_stage_fingerprint": "1" * 64,
        "source_guide_fingerprint": "2" * 64,
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


def test_payload_pruned_authorizes_only_persisted_identity_bound_roots(
    tmp_path: Path,
) -> None:
    _complete_job(tmp_path)
    retained = tmp_path / "not-an-intermediate.txt"
    retained.write_text("keep", encoding="ascii")

    metadata = commit_payload_pruned(tmp_path)

    persisted = json.loads(
        (tmp_path / "04_stereo_diagnostics" / "metadata.json").read_text(encoding="ascii")
    )
    assert metadata.status == "payload_pruned"
    assert persisted["fingerprint"] == metadata.fingerprint
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


def test_payload_pruned_rejects_oversized_final_manifest_before_bulk_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _complete_job(tmp_path)
    final_manifest = tmp_path / "final_video_manifest.json"
    with final_manifest.open("r+b") as handle:
        handle.truncate(1024 * 1024 + 1)
    original_read_bytes = Path.read_bytes

    def reject_bulk_read(path: Path) -> bytes:
        if path == final_manifest:
            raise AssertionError("oversized final manifest was bulk-read")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_bulk_read)

    with pytest.raises(CleanupTransactionError, match="exceeds its bound or is unsafe"):
        commit_payload_pruned(tmp_path)


def test_historical_audit_requires_retained_evidence_not_pruned_payloads(tmp_path: Path):
    _complete_job(tmp_path)
    audit = cleanup_transaction.audit_payload_pruned_historical_diagnostics_v1
    assert not audit(tmp_path)
    authorization = commit_payload_pruned(tmp_path)
    assert audit(tmp_path)
    assert run_authorized_cleanup(tmp_path, metadata=authorization) > 0
    assert audit(tmp_path)

    manifest = tmp_path / "final_video_manifest.json"
    original = manifest.read_bytes()
    manifest.write_bytes(b"other export metadata")
    assert not audit(tmp_path)
    with pytest.raises(CleanupTransactionError, match="retained cleanup evidence hash differs"):
        commit_payload_pruned(tmp_path)
    assert manifest.read_bytes() == b"other export metadata"
    manifest.write_bytes(original)
    assert audit(tmp_path)
    assert run_authorized_cleanup(tmp_path) == 0


def test_cleanup_does_not_treat_another_jobs_cache_as_authority(tmp_path: Path):
    first, second = tmp_path / "first", tmp_path / "second"
    for job in (first, second):
        _complete_job(job)
    first_cache = commit_payload_pruned(first)
    second_cache = commit_payload_pruned(second)
    assert first_cache.fingerprint != second_cache.fingerprint

    with pytest.raises(CleanupTransactionError, match="cache differs from durable authorization"):
        run_authorized_cleanup(second, metadata=first_cache)
    for job in (first, second):
        assert (job / "00_original_frames" / "payload.bin").read_bytes() == b"00_original_frames"
    assert run_authorized_cleanup(second, metadata=second_cache) > 0
    assert not (second / "00_original_frames").exists()
    assert (first / "00_original_frames" / "payload.bin").is_file()


@pytest.mark.parametrize(
    ("evidence", "message"),
    [
        ("summary", "summary hash differs"),
        ("jsonl", "JSONL hash differs"),
        ("final-missing", "manifest exceeds its bound or is unsafe"),
        ("metadata", "metadata is invalid"),
    ],
)
def test_pruning_waits_for_intact_durable_evidence_before_creating_any_marker(
    tmp_path, evidence, message
):
    _complete_job(tmp_path)
    diagnostics = tmp_path / "04_stereo_diagnostics"
    selected = {
        "summary": diagnostics / "stereo_coverage_summary.json",
        "jsonl": diagnostics / "stereo_coverage_frames.jsonl",
        "final-missing": tmp_path / "final_video_manifest.json",
        "metadata": diagnostics / "metadata.json",
    }[evidence]
    saved = selected.read_bytes()
    if evidence == "final-missing":
        selected.unlink()
    else:
        selected.write_bytes(b"interrupted output")
    with pytest.raises(CleanupTransactionError, match=message):
        commit_payload_pruned(tmp_path)
    assert not list(tmp_path.rglob(".depth-surge-prune-*.marker"))
    assert (tmp_path / "00_original_frames" / "payload.bin").read_bytes() == b"00_original_frames"
    selected.write_bytes(saved)
    assert commit_payload_pruned(tmp_path).status == "payload_pruned"


def test_marker_fsync_failure_does_not_leave_deletion_authority(tmp_path, monkeypatch):
    _complete_job(tmp_path)
    metadata = tmp_path / "04_stereo_diagnostics" / "metadata.json"
    original = metadata.read_bytes()
    fsync = os.fsync

    def fail_regular_sync(descriptor):
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("storage unavailable while syncing marker")
        return fsync(descriptor)

    with monkeypatch.context() as interrupted:
        interrupted.setattr(os, "fsync", fail_regular_sync)
        with pytest.raises(OSError, match="syncing marker"):
            commit_payload_pruned(tmp_path)
    assert metadata.read_bytes() == original
    assert not list(tmp_path.rglob(".depth-surge-prune-*.marker"))
    assert (tmp_path / "00_original_frames" / "payload.bin").read_bytes() == b"00_original_frames"
    assert commit_payload_pruned(tmp_path).status == "payload_pruned"


@pytest.mark.skipif(os.name == "nt", reason="POSIX removal and directory sync")
@pytest.mark.parametrize("stage", ["unsupported-rename", "sync-after-rename"])
def test_retry_after_storage_refuses_cleanup_does_not_touch_unlisted_files(
    tmp_path, monkeypatch, stage
):
    from src.depth_surge_3d.io import nofollow_io

    _complete_job(tmp_path)
    projection = commit_payload_pruned(tmp_path)
    user_file = tmp_path / "my-receipt.txt"
    user_file.write_bytes(b"must remain")
    with monkeypatch.context() as fault:
        if stage == "unsupported-rename":

            def unsupported(*args):
                raise OSError(errno.EOPNOTSUPP, "volume does not support this operation")

            fault.setattr(nofollow_io, "_posix_quarantine_noreplace", unsupported)
            expected_error = cleanup_transaction.CleanupCapabilityError
        else:
            real_fsync = os.fsync

            def not_durable(fd):
                if stat.S_ISDIR(os.fstat(fd).st_mode):
                    raise OSError(errno.EIO, "device temporarily unavailable")
                return real_fsync(fd)

            fault.setattr(os, "fsync", not_durable)
            expected_error = CleanupTransactionError
        with pytest.raises(expected_error) as error:
            run_authorized_cleanup(tmp_path, metadata=projection)
    assert cleanup_transaction.cleanup_status_for_error(error.value) == "incomplete_error"
    assert user_file.read_bytes() == b"must remain"
    assert run_authorized_cleanup(tmp_path) > 0
    assert user_file.read_bytes() == b"must remain"
    assert run_authorized_cleanup(tmp_path) == 0


@pytest.mark.parametrize("marker_state", ["missing", "truncated", "oversized"])
def test_cleanup_preserves_nonempty_stage_until_its_own_marker_is_recovered(tmp_path, marker_state):
    _complete_job(tmp_path)
    auth = commit_payload_pruned(tmp_path)
    first = auth.static_fields["prune_entries"][0]
    stage = tmp_path / first["relative_path"]
    marker = stage / first["marker"]["name"]
    prior = marker.read_bytes()
    if marker_state == "missing":
        marker.unlink()
    else:
        marker.write_bytes(prior[:4] if marker_state == "truncated" else prior.ljust(65537, b" "))
    with pytest.raises(CleanupIdentityMismatchError) as failure:
        run_authorized_cleanup(tmp_path)
    assert (
        cleanup_transaction.cleanup_status_for_error(failure.value)
        == "incomplete_identity_mismatch"
    )
    assert (stage / "payload.bin").read_bytes() == b"00_original_frames"
    marker.write_bytes(prior)
    assert run_authorized_cleanup(tmp_path) > 0
    assert not stage.exists()


def test_payload_pruned_streams_all_variable_diagnostics_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _complete_job(tmp_path)
    diagnostics = tmp_path / "04_stereo_diagnostics"
    protected = {
        diagnostics / "metadata.json",
        diagnostics / "stereo_coverage_frames.jsonl",
        diagnostics / "stereo_coverage_summary.json",
    }
    original_read_bytes = Path.read_bytes

    def reject_variable_read_bytes(path: Path) -> bytes:
        if path in protected:
            raise AssertionError("variable diagnostics artifacts must remain streamed")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_variable_read_bytes)

    projection = commit_payload_pruned(tmp_path)

    assert isinstance(projection, DiagnosticsMetadataProjection)
    assert projection.status == "payload_pruned"
    assert projection.frame_count == 1


def test_payload_pruned_preserves_an_unowned_metadata_temporary(tmp_path: Path) -> None:
    _complete_job(tmp_path)
    metadata_path = tmp_path / "04_stereo_diagnostics" / "metadata.json"
    temporary = publication_temporary_path(metadata_path)
    temporary.write_bytes(b"unowned")

    with pytest.raises(OwnedTemporaryCollisionError, match="already exists"):
        commit_payload_pruned(tmp_path)

    assert temporary.read_bytes() == b"unowned"


def test_payload_pruned_consumes_the_active_final_reservation(tmp_path: Path, monkeypatch) -> None:
    _complete_job(tmp_path)
    original_raw = (tmp_path / "04_stereo_diagnostics" / "metadata.json").read_bytes()
    original = json.loads(original_raw)
    planned = cleanup_transaction.plan_prune_markers_v1(tmp_path)
    entries = tuple(marker.prune_entry() for marker in planned)
    committed_payloads: list[tuple[int, str]] = []

    class Reservation:
        index_fingerprint = "f" * 64

        def validate_diagnostics_preflight(
            self,
            *,
            raw_sha256: str,
            fingerprint: str,
        ) -> None:
            assert raw_sha256 == hashlib.sha256(original_raw).hexdigest()
            assert fingerprint == original["fingerprint"]

        def commit_prune_markers(self):
            for marker in planned:
                target = tmp_path / marker.relative_path / marker.marker_name
                target.write_bytes(marker.payload)
            return entries

        def commit_payload_pruned_metadata(self, payload) -> None:
            committed_payloads.append((payload.raw_byte_count, payload.raw_sha256))
            payload.publish(tmp_path / "04_stereo_diagnostics" / "metadata.json")

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
    assert committed_payloads[0] == (metadata.raw_byte_count, metadata.raw_sha256)
    persisted = json.loads((tmp_path / "04_stereo_diagnostics" / "metadata.json").read_bytes())
    assert persisted["fingerprint"] == metadata.fingerprint
    assert persisted["status"] == "payload_pruned"
    assert tuple(persisted["prune_entries"]) == entries


def test_schema5_cleanup_releases_settings_and_reservation_before_tree_walk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _complete_job(tmp_path)
    commit_payload_pruned(tmp_path)
    (tmp_path / "job-control-v1.json").write_bytes(b"control")
    active_index = tmp_path / ".depth-surge-reservations-v1" / "final-encoding-reservation-v1.json"
    active_index.parent.mkdir(exist_ok=True)
    active_index.write_bytes(b"index")
    references: dict[str, weakref.ReferenceType[object]] = {}

    class Artifact:
        @staticmethod
        def to_object():
            return {"metadata": {"cleanup_status": "pending"}}

    class Reservation:
        @staticmethod
        def recover_publication_prefix():
            return type("Recovery", (), {"phase": "publication_authenticated"})()

    def recover(*_args, **_kwargs):
        artifact = Artifact()
        references["settings"] = weakref.ref(artifact)
        return artifact

    def load(_root, settings, **_kwargs):
        assert references["settings"]() is settings
        reservation = Reservation()
        references["reservation"] = weakref.ref(reservation)
        return reservation

    def verify_release(stage: str) -> None:
        if stage != "before_cleanup_walk":
            return
        gc.collect()
        assert references["settings"]() is None
        assert references["reservation"]() is None

    monkeypatch.setattr(
        "src.depth_surge_3d.io.operations.recover_processing_control_prefix_v1",
        recover,
    )
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.video.final_publication."
        "load_final_encoding_reservation_v1",
        load,
    )

    assert run_authorized_cleanup(tmp_path, fault=verify_release) > 0


def test_cleanup_rejects_a_replacement_directory_without_touching_it(
    tmp_path: Path,
) -> None:
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


@pytest.mark.skipif(os.name != "nt", reason="Windows enumerated identity fixture")
@pytest.mark.parametrize("directory", [False, True])
def test_windows_cleanup_rejects_name_swap_before_child_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directory: bool,
) -> None:
    _complete_job(tmp_path)
    root = tmp_path / "00_original_frames"
    victim = root / "victim"
    parked = root / "parked"
    if directory:
        victim.mkdir()
        (victim / "authorized.bin").write_bytes(b"authorized")
    else:
        victim.write_bytes(b"authorized")
    metadata = commit_payload_pruned(tmp_path)
    original_entries = cleanup_transaction.windows_file_id_extd_entries_v1
    injected = False

    def swap_after_enumeration(buffer, buffer_size):
        nonlocal injected
        for entry in original_entries(buffer, buffer_size):
            in_destructive_selection = any(
                frame.function == "first_name" for frame in inspect.stack()
            )
            if not injected and entry.name == victim.name and in_destructive_selection:
                injected = True
                victim.rename(parked)
                if directory:
                    victim.mkdir()
                    (victim / "replacement.bin").write_bytes(b"replacement")
                else:
                    victim.write_bytes(b"replacement")
            yield entry

    monkeypatch.setattr(
        cleanup_transaction,
        "windows_file_id_extd_entries_v1",
        swap_after_enumeration,
    )

    with pytest.raises(CleanupIdentityMismatchError, match="changed before acquisition"):
        run_authorized_cleanup(tmp_path, metadata=metadata)

    assert injected
    assert parked.exists()
    assert victim.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX enumerated identity fixture")
@pytest.mark.parametrize("directory", [False, True])
def test_posix_cleanup_rejects_name_swap_before_child_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directory: bool,
) -> None:
    _complete_job(tmp_path)
    root = tmp_path / "00_original_frames"
    victim = root / "victim"
    parked = root / "parked"
    if directory:
        victim.mkdir()
        (victim / "authorized.bin").write_bytes(b"authorized")
    else:
        victim.write_bytes(b"authorized")
    metadata = commit_payload_pruned(tmp_path)
    original_open = cleanup_transaction.os.open
    injected = False

    def swap_before_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal injected
        in_destructive_visit = any(frame.function == "visit" for frame in inspect.stack())
        if not injected and path == victim.name and dir_fd is not None and in_destructive_visit:
            injected = True
            os.rename(
                victim.name,
                parked.name,
                src_dir_fd=dir_fd,
                dst_dir_fd=dir_fd,
            )
            if directory:
                os.mkdir(victim.name, dir_fd=dir_fd)
            else:
                replacement = original_open(
                    victim.name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=dir_fd,
                )
                os.write(replacement, b"replacement")
                os.close(replacement)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(cleanup_transaction.os, "open", swap_before_open)

    with pytest.raises(CleanupIdentityMismatchError, match="changed before acquisition"):
        run_authorized_cleanup(tmp_path, metadata=metadata)

    assert injected
    assert parked.exists()
    assert victim.exists()


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


@pytest.mark.skipif(os.name == "nt", reason="POSIX quarantine recovery fixture")
@pytest.mark.parametrize("removed_role", ["marker", "root"])
def test_cleanup_resumes_a_durable_quarantine_after_a_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    removed_role: str,
) -> None:
    _complete_job(tmp_path)
    metadata = commit_payload_pruned(tmp_path)
    first_entry = metadata.static_fields["prune_entries"][0]
    marker_name = first_entry["marker"]["name"]
    original_remove = cleanup_transaction.posix_remove_verified_name
    crashed = False

    def remove_with_crash(
        parent,
        name,
        descriptor,
        *,
        directory,
        quarantine_name=None,
        recovering=False,
    ):
        def fault(stage: str, _quarantine: str) -> None:
            nonlocal crashed
            is_target = name == ("00_original_frames" if removed_role == "root" else marker_name)
            if not crashed and is_target and stage == "after_quarantine_rename":
                crashed = True
                raise RuntimeError("simulated crash")

        return original_remove(
            parent,
            name,
            descriptor,
            directory=directory,
            quarantine_name=quarantine_name,
            recovering=recovering,
            fault=fault,
        )

    monkeypatch.setattr(
        cleanup_transaction,
        "posix_remove_verified_name",
        remove_with_crash,
    )

    with pytest.raises(RuntimeError, match="simulated crash"):
        run_authorized_cleanup(tmp_path, metadata=metadata)
    assert crashed

    run_authorized_cleanup(tmp_path)

    assert not (tmp_path / "00_original_frames").exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX quarantine race fixture")
@pytest.mark.parametrize("marker_already_removed", [False, True])
def test_cleanup_never_deletes_a_root_replacement_created_after_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker_already_removed: bool,
) -> None:
    _complete_job(tmp_path)
    metadata = commit_payload_pruned(tmp_path)
    prune_root = tmp_path / "00_original_frames"
    if marker_already_removed:

        def crash(stage: str) -> None:
            if stage == "after_prune_marker_unlink_0":
                raise RuntimeError("simulated crash")

        with pytest.raises(RuntimeError, match="simulated crash"):
            run_authorized_cleanup(tmp_path, metadata=metadata, fault=crash)
    original_remove = cleanup_transaction.posix_remove_verified_name
    replaced = False

    def remove_with_replacement(
        parent, name, descriptor, *, directory, quarantine_name=None, recovering=False
    ):
        def fault(stage: str, _quarantine: str) -> None:
            nonlocal replaced
            if (
                not replaced
                and directory
                and name == prune_root.name
                and stage == "after_quarantine_rename"
            ):
                os.mkdir(name, dir_fd=parent)
                replacement_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=parent,
                )
                try:
                    sentinel_fd = os.open(
                        "must-survive.bin",
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=replacement_fd,
                    )
                    os.write(sentinel_fd, b"replacement")
                    os.close(sentinel_fd)
                finally:
                    os.close(replacement_fd)
                replaced = True

        return original_remove(
            parent,
            name,
            descriptor,
            directory=directory,
            quarantine_name=quarantine_name,
            recovering=recovering,
            fault=fault,
        )

    monkeypatch.setattr(
        cleanup_transaction,
        "posix_remove_verified_name",
        remove_with_replacement,
    )

    with pytest.raises(CleanupIdentityMismatchError, match="changed|reappeared"):
        run_authorized_cleanup(tmp_path)

    assert replaced
    assert (prune_root / "must-survive.bin").read_bytes() == b"replacement"


def test_cleanup_final_sweep_rejects_a_reappeared_prune_root(tmp_path: Path) -> None:
    _complete_job(tmp_path)
    metadata = commit_payload_pruned(tmp_path)
    reappeared = tmp_path / "00_original_frames"

    def recreate_after_delete(stage: str) -> None:
        if stage == "after_prune_entry_0":
            reappeared.mkdir()
            (reappeared / "must-survive.bin").write_bytes(b"replacement")

    with pytest.raises(CleanupIdentityMismatchError, match="reappeared"):
        run_authorized_cleanup(
            tmp_path,
            metadata=metadata,
            fault=recreate_after_delete,
        )

    assert (reappeared / "must-survive.bin").read_bytes() == b"replacement"


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


def test_payload_pruned_rejects_linked_retained_aggregate(tmp_path: Path) -> None:
    _complete_job(tmp_path)
    summary = tmp_path / "04_stereo_diagnostics" / "stereo_coverage_summary.json"
    external = tmp_path.parent / f"{tmp_path.name}-external-summary.json"
    external.write_bytes(summary.read_bytes())
    summary.unlink()
    try:
        os.symlink(external, summary)
    except OSError as error:
        pytest.skip(f"file symlink creation is unavailable: {error}")

    with pytest.raises(CleanupTransactionError, match="retained cleanup evidence"):
        commit_payload_pruned(tmp_path)

    assert external.read_bytes() == canonical_json_bytes({"schema_version": 1})


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
    commit_payload_pruned(tmp_path)
    persisted = json.loads((tmp_path / "04_stereo_diagnostics" / "metadata.json").read_bytes())
    forged_root = tmp_path if relative_path == "." else tmp_path / relative_path
    entry = dict(persisted["prune_entries"][0])
    entry.update(
        {
            "stage_key": stage_key,
            "relative_path": relative_path,
            "directory_identity": cleanup_transaction._directory_identity(forged_root).to_object(),
        }
    )
    forged = {**persisted, "prune_entries": [entry]}
    forged["fingerprint"] = self_fingerprint(forged)
    (tmp_path / "04_stereo_diagnostics" / "metadata.json").write_bytes(canonical_json_bytes(forged))
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
        run_authorized_cleanup(tmp_path)

    assert sentinel.read_bytes() == sentinel_raw
