from __future__ import annotations

import hashlib
import json
import os

import pytest

from src.depth_surge_3d.core.settings import validate_settings
from src.depth_surge_3d.rendering.quality.diagnostic_transaction import (
    canonical_json_bytes,
    self_fingerprint,
)
from src.depth_surge_3d.rendering.quality.cleanup_transaction import (
    commit_payload_pruned,
    run_authorized_cleanup,
)
from src.depth_surge_3d.processing.video.final_publication import (
    EncodingInputPublicationSource,
    FinalMediaAuditDispositionV1,
    FinalPublicationTransactionV4,
    InjectedFinalPublicationCrash,
    build_final_video_manifest_v4,
    parse_final_video_manifest_v4,
    prepare_final_encoding_reservation_v1,
    load_final_encoding_reservation_v1,
    recover_final_encoding_terminal_retirement_v1,
    audit_final_media_v1,
)
from src.depth_surge_3d.io.reservation_protocol import ReservationAllocationEvidence
from tests.unit.test_settings_artifact_v5 import _artifact


def _input_manifest() -> tuple[bytes, str]:
    base = {
        "algorithm_version": "encoding-input-sequence-v2",
        "frame_count": 2,
        "schema_version": 1,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            base,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    value = {**base, "fingerprint": fingerprint}
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii"),
        fingerprint,
    )


def _producer_artifact():
    return _artifact().with_producer_marker(
        generation="44" * 16,
        updated_at="2026-08-24T12:34:57.123456Z",
    )


def _pruning_artifact():
    return _artifact(
        processing_settings=validate_settings(
            {"keep_intermediates": False},
            source="explicit",
        )
    )


def _complete_prune_diagnostics(root) -> None:
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
    for relative in (
        "00_original_frames",
        "04_left_frames",
        "04_right_frames",
        "04_stereo_diagnostics/frames",
    ):
        directory = root / relative
        directory.mkdir(parents=True)
        (directory / "payload.bin").write_bytes(relative.encode("ascii"))


def _test_allocation(path, logical):
    return ReservationAllocationEvidence("test", logical, logical, f"test:{logical}")


def _test_allocation_counter(path):
    size = path.stat().st_size
    return ((size + 4095) // 4096) * 4096


def test_final_video_manifest_has_closed_canonical_self_fingerprint() -> None:
    manifest = build_final_video_manifest_v4(
        relative_path="result.mp4",
        video_sha256="12" * 32,
        video_byte_count=4217,
        input_stage_fingerprint="34" * 32,
        resolved_encoding_arguments=(
            "ffmpeg",
            "-i",
            "@image-sequence:sha256=" + "34" * 32 + ":view=frames",
            "@output:path64=cmVzdWx0Lm1wNA",
        ),
    )
    raw = manifest.encode()

    assert not raw.endswith(b"\n")
    assert parse_final_video_manifest_v4(raw) == manifest
    value = manifest.to_object()
    assert set(value) == {
        "schema_version",
        "algorithm_version",
        "relative_path",
        "sha256",
        "byte_count",
        "input_stage_fingerprint",
        "resolved_encoding_arguments",
        "encoding_settings_fingerprint",
        "fingerprint",
    }


def test_successful_publication_orders_video_then_input_then_final_and_audits(tmp_path) -> None:
    artifact = _producer_artifact()
    expected = artifact.to_object()["output_info"]["expected_final_relative_path"]
    sibling = tmp_path / f".{expected}.encoding-v4.tmp"
    sibling.write_bytes(b"validated-mp4-fixture")
    input_raw, input_fingerprint = _input_manifest()
    events: list[str] = []
    transaction = FinalPublicationTransactionV4(
        job_root=tmp_path,
        settings=artifact,
        sibling_relative_path=sibling.name,
        encoding_input_manifest_raw=input_raw,
        input_stage_fingerprint=input_fingerprint,
        resolved_encoding_arguments=(
            "ffmpeg",
            "-i",
            "@image-sequence:sha256=" + input_fingerprint + ":view=frames",
            "@output:path64=" + FinalPublicationTransactionV4.path64(expected),
        ),
        container_validator=lambda path: path.read_bytes().startswith(b"validated"),
        event=events.append,
    )

    manifest = transaction.publish()

    assert events == [
        "video:published",
        "input-manifest:published",
        "final-manifest:published",
    ]
    assert manifest.sha256 == hashlib.sha256(b"validated-mp4-fixture").hexdigest()
    assert audit_final_media_v1(tmp_path, artifact) == FinalMediaAuditDispositionV1(
        final_video_present=True,
        final_video_authenticated=True,
        final_video_authentication_reason=None,
    )


def test_post_video_crash_is_classified_from_actual_mixed_generation(tmp_path) -> None:
    artifact = _producer_artifact()
    expected = artifact.to_object()["output_info"]["expected_final_relative_path"]
    (tmp_path / expected).write_bytes(b"old-video")
    old_input, old_input_fingerprint = _input_manifest()
    (tmp_path / "encoding_input_manifest.json").write_bytes(old_input)
    old_manifest = build_final_video_manifest_v4(
        relative_path=expected,
        video_sha256=hashlib.sha256(b"old-video").hexdigest(),
        video_byte_count=len(b"old-video"),
        input_stage_fingerprint=old_input_fingerprint,
        resolved_encoding_arguments=(
            "ffmpeg",
            "@output:path64=" + FinalPublicationTransactionV4.path64(expected),
        ),
    )
    (tmp_path / "final_video_manifest.json").write_bytes(old_manifest.encode())
    sibling = tmp_path / f".{expected}.encoding-v4.tmp"
    sibling.write_bytes(b"validated-new-video")
    input_raw, input_fingerprint = _input_manifest()
    transaction = FinalPublicationTransactionV4(
        job_root=tmp_path,
        settings=artifact,
        sibling_relative_path=sibling.name,
        encoding_input_manifest_raw=input_raw,
        input_stage_fingerprint=input_fingerprint,
        resolved_encoding_arguments=(
            "ffmpeg",
            "-i",
            "@image-sequence:sha256=" + input_fingerprint + ":view=frames",
            "@output:path64=" + FinalPublicationTransactionV4.path64(expected),
        ),
        container_validator=lambda _path: True,
    )

    with pytest.raises(InjectedFinalPublicationCrash):
        transaction.publish(crash_at="after-video")

    assert (tmp_path / expected).read_bytes() == b"validated-new-video"
    disposition = audit_final_media_v1(tmp_path, artifact)
    assert disposition.final_video_present is True
    assert disposition.final_video_authenticated is False
    assert disposition.final_video_authentication_reason == "publication_validation_failed"


def test_publication_streams_staged_input_manifest_without_read_bytes(
    tmp_path, monkeypatch
) -> None:
    artifact = _producer_artifact()
    expected = artifact.to_object()["output_info"]["expected_final_relative_path"]
    sibling = tmp_path / f".{expected}.encoding-v4.tmp"
    sibling.write_bytes(b"validated-streamed-video")
    raw, fingerprint = _input_manifest()
    staged_path = tmp_path / ".encoding-input-manifest.staged"
    staged_path.write_bytes(raw)
    source = EncodingInputPublicationSource(
        path=staged_path,
        fingerprint=fingerprint,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        raw_byte_count=len(raw),
    )
    original_read_bytes = type(staged_path).read_bytes
    protected = {
        staged_path.resolve(),
        (tmp_path / "encoding_input_manifest.json").resolve(),
    }

    def reject_manifest_materialization(path):
        if path.resolve() in protected:
            raise AssertionError("encoding manifest must be streamed")
        return original_read_bytes(path)

    monkeypatch.setattr(type(staged_path), "read_bytes", reject_manifest_materialization)
    transaction = FinalPublicationTransactionV4(
        job_root=tmp_path,
        settings=artifact,
        sibling_relative_path=sibling.name,
        encoding_input_source=source,
        input_stage_fingerprint=fingerprint,
        resolved_encoding_arguments=(
            "ffmpeg",
            "-i",
            "@image-sequence:sha256=" + fingerprint + ":view=frames",
            "@output:path64=" + FinalPublicationTransactionV4.path64(expected),
        ),
        container_validator=lambda _path: True,
    )

    transaction.publish()

    assert not staged_path.exists()
    assert audit_final_media_v1(tmp_path, artifact).final_video_authenticated is True


def test_final_encoding_reservation_publishes_index_before_owned_namespace(tmp_path) -> None:
    artifact = _producer_artifact()
    (tmp_path / "processing-settings-v5.json").write_bytes(artifact.encode())
    events: list[str] = []

    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
        event=events.append,
    )

    expected = artifact.to_object()["output_info"]["expected_final_relative_path"]
    assert events[0:2] == ["index:temporary", "index:published"]
    assert events.index("index:published") < events.index("final-target:reserved")
    assert (tmp_path / expected).stat().st_size == 0
    assert reservation.sibling_path.name == f".depth-surge-final-v4-{'44' * 16}.tmp.mp4"
    assert reservation.index_path.is_file()
    assert reservation.target_descriptor_path.is_file()
    assert reservation.input_manifest_transaction.descriptor_path.is_file()
    assert reservation.final_manifest_transaction.descriptor_path.is_file()

    loaded = load_final_encoding_reservation_v1(
        tmp_path,
        artifact,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    recovery = loaded.observe()
    assert recovery.phase == "ready_pre_video"
    assert recovery.action == "encode"


def test_pruning_final_reservation_preallocates_every_post_publication_extent(tmp_path) -> None:
    artifact = _pruning_artifact()
    (tmp_path / "processing-settings-v5.json").write_bytes(artifact.encode())
    _complete_prune_diagnostics(tmp_path)

    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )

    assert [entry["reservation_kind"] for entry in reservation.index.extent_entries] == [
        "producer_settings",
        "encoding_input_manifest",
        "final_video_manifest",
        "prune_marker",
        "prune_marker",
        "prune_marker",
        "prune_marker",
        "diagnostics_payload_pruned",
        "cleanup_pending_settings",
    ]
    assert reservation.producer_settings_transaction is not None
    assert len(reservation.prune_marker_transactions) == 4
    assert reservation.diagnostics_payload_pruned_transaction is not None
    assert reservation.cleanup_pending_settings_transaction is not None
    assert (
        reservation.index.diagnostics_metadata_raw_sha256_at_preflight
        == hashlib.sha256(
            (tmp_path / "04_stereo_diagnostics" / "metadata.json").read_bytes()
        ).hexdigest()
    )
    for _entry, transaction, _payload in reservation.prune_marker_transactions:
        assert transaction.target_path.stat().st_size == 0
        assert transaction.source_path.stat().st_size == transaction.descriptor.logical_byte_count


def test_pruning_final_reservation_reopens_every_indexed_extent(tmp_path) -> None:
    artifact = _pruning_artifact()
    (tmp_path / "processing-settings-v5.json").write_bytes(artifact.encode())
    _complete_prune_diagnostics(tmp_path)
    prepared = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )

    reopened = load_final_encoding_reservation_v1(
        tmp_path,
        artifact,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )

    assert reopened.producer_settings_transaction is not None
    assert reopened.producer_settings_payload is None
    assert len(reopened.prune_marker_transactions) == 4
    assert all(transaction is not None for _, transaction, _ in reopened.prune_marker_transactions)
    assert reopened.diagnostics_payload_pruned_transaction is not None
    assert reopened.cleanup_pending_settings_transaction is not None
    assert reopened.cleanup_pending_settings_payload is None

    producer = reopened.commit_producer_settings()
    assert producer.to_object()["metadata"]["final_media_publication_generation"] == "44" * 16
    pending = reopened.commit_cleanup_pending_settings()
    assert pending.to_object()["metadata"]["cleanup_status"] == "pending"


def test_partial_producer_settings_source_is_reset_before_any_dependent_mutation(
    tmp_path,
) -> None:
    artifact = _pruning_artifact()
    (tmp_path / "processing-settings-v5.json").write_bytes(artifact.encode())
    _complete_prune_diagnostics(tmp_path)
    prepared = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    assert prepared.producer_settings_transaction is not None
    with prepared.producer_settings_transaction.source_path.open("r+b", buffering=0) as handle:
        handle.write(b'{"content_fingerprint":"partial')
        handle.flush()
        os.fsync(handle.fileno())

    reopened = load_final_encoding_reservation_v1(
        tmp_path,
        artifact,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    committed = reopened.commit_producer_settings()

    assert committed.to_object()["metadata"]["final_media_publication_generation"] == "44" * 16


def test_pruning_failure_retirement_leaves_no_unindexed_reservation_objects(tmp_path) -> None:
    artifact = _pruning_artifact()
    (tmp_path / "processing-settings-v5.json").write_bytes(artifact.encode())
    _complete_prune_diagnostics(tmp_path)
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    owned_paths = []
    for entry in reservation.index.extent_entries:
        if entry["reservation_kind"] == "job_terminal_settings":
            continue
        descriptor = (
            tmp_path
            / ".depth-surge-reservations-v1"
            / (
                f"d-r-{entry['generation']}-{entry['reservation_kind']}-"
                f"{entry['reservation_id']}.json"
            )
        )
        target = tmp_path / entry["target_relative_path"]
        source = target.parent / (
            f".depth-surge-r-{entry['generation']}-{entry['reservation_kind']}-"
            f"{entry['reservation_id']}.reserved"
        )
        owned_paths.extend((descriptor, source))

    reservation.retire_terminal_failure()

    assert not reservation.index_path.exists()
    assert not reservation.target_descriptor_path.exists()
    assert all(not path.exists() for path in owned_paths)


def test_pruning_success_consumes_all_extents_before_index_last_retirement(tmp_path) -> None:
    artifact = _pruning_artifact()
    (tmp_path / "processing-settings-v5.json").write_bytes(artifact.encode())
    _complete_prune_diagnostics(tmp_path)
    raw, fingerprint = _input_manifest()
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=len(raw),
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    producer = reservation.commit_producer_settings()
    assert reservation.input_manifest_transaction is not None
    reservation.input_manifest_transaction.stage_complete_full(raw)
    reservation.sibling_path.write_bytes(b"validated-pruning-video")
    FinalPublicationTransactionV4(
        job_root=tmp_path,
        settings=producer,
        sibling_relative_path=reservation.sibling_path.name,
        encoding_input_manifest_raw=raw,
        input_stage_fingerprint=fingerprint,
        resolved_encoding_arguments=(
            "ffmpeg",
            "@output:path64="
            + FinalPublicationTransactionV4.path64(
                producer.to_object()["output_info"]["expected_final_relative_path"]
            ),
        ),
        container_validator=lambda _path: True,
        reservation=reservation,
        retire_reservation=False,
    ).publish()

    pruned = commit_payload_pruned(tmp_path, reservation=reservation)
    pending = reservation.commit_cleanup_pending_settings()
    removed = run_authorized_cleanup(tmp_path, metadata=pruned)
    reservation.retire()

    assert pending.to_object()["metadata"]["cleanup_status"] == "pending"
    assert removed > 0
    assert not reservation.index_path.exists()
    assert not reservation.target_descriptor_path.exists()
    reservation_dir = tmp_path / ".depth-surge-reservations-v1"
    assert not list(reservation_dir.glob("d-r-55*"))


def test_pruning_crash_after_markers_reopens_the_remaining_ordered_prefix(tmp_path) -> None:
    artifact = _pruning_artifact()
    (tmp_path / "processing-settings-v5.json").write_bytes(artifact.encode())
    _complete_prune_diagnostics(tmp_path)
    raw, fingerprint = _input_manifest()
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=len(raw),
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    producer = reservation.commit_producer_settings()
    assert reservation.input_manifest_transaction is not None
    reservation.input_manifest_transaction.stage_complete_full(raw)
    reservation.sibling_path.write_bytes(b"validated-pruning-recovery-video")
    FinalPublicationTransactionV4(
        job_root=tmp_path,
        settings=producer,
        sibling_relative_path=reservation.sibling_path.name,
        encoding_input_manifest_raw=raw,
        input_stage_fingerprint=fingerprint,
        resolved_encoding_arguments=(
            "ffmpeg",
            "@output:path64="
            + FinalPublicationTransactionV4.path64(
                producer.to_object()["output_info"]["expected_final_relative_path"]
            ),
        ),
        container_validator=lambda _path: True,
        reservation=reservation,
        retire_reservation=False,
    ).publish()

    def crash_after_markers(stage: str) -> None:
        if stage == "after_prune_markers":
            raise RuntimeError(stage)

    with pytest.raises(RuntimeError, match="after_prune_markers"):
        commit_payload_pruned(
            tmp_path,
            reservation=reservation,
            fault=crash_after_markers,
        )

    reopened = load_final_encoding_reservation_v1(
        tmp_path,
        producer,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    assert all(transaction is None for _, transaction, _ in reopened.prune_marker_transactions)
    assert reopened.diagnostics_payload_pruned_transaction is not None
    assert reopened.cleanup_pending_settings_transaction is not None

    pruned = commit_payload_pruned(tmp_path, reservation=reopened)
    reopened.commit_cleanup_pending_settings()
    run_authorized_cleanup(tmp_path, metadata=pruned)
    reopened.retire()

    assert not reopened.index_path.exists()


@pytest.mark.parametrize(
    "stop_event",
    [
        "index:published",
        "final-target:reserved",
        "input-manifest:descriptor:published",
    ],
)
def test_final_reservation_construction_recovers_every_indexed_prefix(tmp_path, stop_event) -> None:
    artifact = _producer_artifact()
    (tmp_path / "processing-settings-v5.json").write_bytes(artifact.encode())

    def stop(value: str) -> None:
        if value == stop_event:
            raise RuntimeError(stop_event)

    with pytest.raises(RuntimeError, match=stop_event):
        prepare_final_encoding_reservation_v1(
            job_root=tmp_path,
            settings=artifact,
            publication_generation="44" * 16,
            input_manifest_raw_max=4096,
            final_manifest_raw_max=4096,
            allocation_unit=4096,
            allocation_adapter=_test_allocation,
            allocation_counter=_test_allocation_counter,
            reservation_generation="55" * 16,
            input_reservation_id="66" * 16,
            final_reservation_id="77" * 16,
            event=stop,
        )

    recovered = load_final_encoding_reservation_v1(
        tmp_path,
        artifact,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )

    assert recovered.target_descriptor_path.is_file()
    assert recovered.input_manifest_transaction is not None
    assert recovered.final_manifest_transaction is not None
    assert recovered.observe().phase == "ready_pre_video"


def test_reserved_publication_retires_descriptors_then_index_last(tmp_path) -> None:
    artifact = _producer_artifact()
    (tmp_path / "processing-settings-v5.json").write_bytes(artifact.encode())
    raw, fingerprint = _input_manifest()
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=len(raw),
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    reservation.input_manifest_transaction.stage_complete_full(raw)
    input_descriptor = reservation.input_manifest_transaction.descriptor_path
    final_descriptor = reservation.final_manifest_transaction.descriptor_path
    reservation.sibling_path.write_bytes(b"validated-reserved-video")
    events: list[str] = []
    transaction = FinalPublicationTransactionV4(
        job_root=tmp_path,
        settings=artifact,
        sibling_relative_path=reservation.sibling_path.name,
        encoding_input_manifest_raw=raw,
        input_stage_fingerprint=fingerprint,
        resolved_encoding_arguments=(
            "ffmpeg",
            "-i",
            "@image-sequence:sha256=" + fingerprint + ":view=frames",
            "@output:path64="
            + FinalPublicationTransactionV4.path64(
                artifact.to_object()["output_info"]["expected_final_relative_path"]
            ),
        ),
        container_validator=lambda _path: True,
        reservation=reservation,
        event=events.append,
    )

    transaction.publish()

    assert not input_descriptor.exists()
    assert not final_descriptor.exists()
    assert not reservation.target_descriptor_path.exists()
    assert not reservation.index_path.exists()
    assert events[-2:] == ["reservation:target-descriptor-retired", "reservation:index-retired"]
    assert audit_final_media_v1(tmp_path, artifact).final_video_authenticated is True


def test_reserved_post_video_crash_keeps_index_as_recovery_authority(tmp_path) -> None:
    artifact = _producer_artifact()
    (tmp_path / "processing-settings-v5.json").write_bytes(artifact.encode())
    raw, fingerprint = _input_manifest()
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=len(raw),
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    reservation.input_manifest_transaction.stage_complete_full(raw)
    reservation.sibling_path.write_bytes(b"validated-reserved-video")
    transaction = FinalPublicationTransactionV4(
        job_root=tmp_path,
        settings=artifact,
        sibling_relative_path=reservation.sibling_path.name,
        encoding_input_manifest_raw=raw,
        input_stage_fingerprint=fingerprint,
        resolved_encoding_arguments=(
            "ffmpeg",
            "@output:path64="
            + FinalPublicationTransactionV4.path64(
                artifact.to_object()["output_info"]["expected_final_relative_path"]
            ),
        ),
        container_validator=lambda _path: True,
        reservation=reservation,
    )

    with pytest.raises(InjectedFinalPublicationCrash):
        transaction.publish(crash_at="after-video")

    assert reservation.index_path.is_file()
    assert reservation.target_descriptor_path.is_file()
    assert reservation.input_manifest_transaction.descriptor_path.is_file()
    assert reservation.final_manifest_transaction.descriptor_path.is_file()

    loaded = load_final_encoding_reservation_v1(
        tmp_path,
        artifact,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    recovery = loaded.observe()
    assert recovery.phase == "video_replaced_input_intent"
    assert recovery.action == "commit-input-manifest"


def test_reserved_input_manifest_refuses_a_replaced_preflight_target(tmp_path) -> None:
    artifact = _producer_artifact()
    (tmp_path / "processing-settings-v5.json").write_bytes(artifact.encode())
    old_raw, _old_fingerprint = _input_manifest()
    input_target = tmp_path / "encoding_input_manifest.json"
    input_target.write_bytes(old_raw)
    new_raw, new_fingerprint = _input_manifest()
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=len(new_raw),
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    assert reservation.input_manifest_transaction is not None
    reservation.input_manifest_transaction.stage_complete_full(new_raw)
    replacement = tmp_path / "replacement-input.json"
    replacement.write_bytes(old_raw)
    os.replace(replacement, input_target)
    reservation.sibling_path.write_bytes(b"validated-predecessor-video")

    with pytest.raises(Exception, match="preflight target"):
        FinalPublicationTransactionV4(
            job_root=tmp_path,
            settings=artifact,
            sibling_relative_path=reservation.sibling_path.name,
            encoding_input_manifest_raw=new_raw,
            input_stage_fingerprint=new_fingerprint,
            resolved_encoding_arguments=(
                "ffmpeg",
                "@output:path64="
                + FinalPublicationTransactionV4.path64(
                    artifact.to_object()["output_info"]["expected_final_relative_path"]
                ),
            ),
            container_validator=lambda _path: True,
            reservation=reservation,
            retire_reservation=False,
        ).publish()

    assert input_target.read_bytes() == old_raw


@pytest.mark.skipif(os.name != "nt", reason="Windows replacement gaps are platform-specific")
def test_reserved_windows_video_replacement_gap_moves_indexed_sibling_forward(tmp_path) -> None:
    artifact = _producer_artifact()
    (tmp_path / "processing-settings-v5.json").write_bytes(artifact.encode())
    raw, _fingerprint = _input_manifest()
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=len(raw),
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    assert reservation.input_manifest_transaction is not None
    reservation.input_manifest_transaction.stage_complete_full(raw)
    reservation.sibling_path.write_bytes(b"validated-replacement-gap-video")
    reservation.final_video_path.unlink()

    loaded = load_final_encoding_reservation_v1(
        tmp_path,
        artifact,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )

    assert loaded.observe().phase == "windows_video_replace_gap_before_input_commit"
    loaded.recover_video_replacement_gap()
    assert loaded.final_video_path.read_bytes() == b"validated-replacement-gap-video"
    assert not loaded.sibling_path.exists()
    assert loaded.observe().phase == "video_replaced_input_intent"


@pytest.mark.parametrize(
    ("crash_at", "phase", "action"),
    [
        (
            "after-input-manifest",
            "input_manifest_committed_no_final_intent",
            "revalidate-input-and-reencode",
        ),
        (
            "after-final-manifest",
            "publication_authenticated",
            "continue-finalization",
        ),
    ],
)
def test_reserved_manifest_crash_prefix_reopens_to_one_phase(
    tmp_path, crash_at, phase, action
) -> None:
    artifact = _producer_artifact()
    (tmp_path / "processing-settings-v5.json").write_bytes(artifact.encode())
    raw, fingerprint = _input_manifest()
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=len(raw),
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    reservation.input_manifest_transaction.stage_complete_full(raw)
    reservation.sibling_path.write_bytes(b"validated-reserved-video")
    transaction = FinalPublicationTransactionV4(
        job_root=tmp_path,
        settings=artifact,
        sibling_relative_path=reservation.sibling_path.name,
        encoding_input_manifest_raw=raw,
        input_stage_fingerprint=fingerprint,
        resolved_encoding_arguments=(
            "ffmpeg",
            "@output:path64="
            + FinalPublicationTransactionV4.path64(
                artifact.to_object()["output_info"]["expected_final_relative_path"]
            ),
        ),
        container_validator=lambda _path: True,
        reservation=reservation,
    )

    with pytest.raises(InjectedFinalPublicationCrash):
        transaction.publish(crash_at=crash_at)

    loaded = load_final_encoding_reservation_v1(
        tmp_path,
        artifact,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    recovery = loaded.observe()
    assert (recovery.phase, recovery.action) == (phase, action)
    if crash_at == "after-final-manifest":
        loaded.retire()
        assert not loaded.index_path.exists()


def test_terminal_retirement_recovers_after_target_descriptor_unlink(tmp_path) -> None:
    artifact = _producer_artifact()
    (tmp_path / "processing-settings-v5.json").write_bytes(artifact.encode())
    raw, fingerprint = _input_manifest()
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=len(raw),
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    reservation.input_manifest_transaction.stage_complete_full(raw)
    reservation.sibling_path.write_bytes(b"validated-reserved-video")
    FinalPublicationTransactionV4(
        job_root=tmp_path,
        settings=artifact,
        sibling_relative_path=reservation.sibling_path.name,
        encoding_input_manifest_raw=raw,
        input_stage_fingerprint=fingerprint,
        resolved_encoding_arguments=(
            "ffmpeg",
            "@output:path64="
            + FinalPublicationTransactionV4.path64(
                artifact.to_object()["output_info"]["expected_final_relative_path"]
            ),
        ),
        container_validator=lambda _path: True,
        reservation=reservation,
        retire_reservation=False,
    ).publish()
    completed = artifact.completed(
        terminal_at="2026-08-24T12:35:00.000000Z",
        frames_processed=12,
        cleanup_status="not_requested",
    )

    with pytest.raises(InjectedFinalPublicationCrash):
        reservation.retire(crash_at="after-target-descriptor-retirement")

    assert recover_final_encoding_terminal_retirement_v1(
        tmp_path,
        completed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )
    assert not reservation.index_path.exists()


def test_terminal_failure_retirement_removes_only_unused_placeholders(tmp_path) -> None:
    artifact = _producer_artifact()
    (tmp_path / "processing-settings-v5.json").write_bytes(artifact.encode())
    expected = artifact.to_object()["output_info"]["expected_final_relative_path"]
    reservation = prepare_final_encoding_reservation_v1(
        job_root=tmp_path,
        settings=artifact,
        publication_generation="44" * 16,
        input_manifest_raw_max=4096,
        final_manifest_raw_max=4096,
        allocation_unit=4096,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
        reservation_generation="55" * 16,
        input_reservation_id="66" * 16,
        final_reservation_id="77" * 16,
    )
    reservation.sibling_path.write_bytes(b"partial-video")
    failed = artifact.failed(
        terminal_at="2026-08-24T12:35:00.000000Z",
        diagnostic_code="FINAL_ENCODING_FAILED",
        diagnostic_message="fixture",
    )

    assert recover_final_encoding_terminal_retirement_v1(
        tmp_path,
        failed,
        allocation_adapter=_test_allocation,
        allocation_counter=_test_allocation_counter,
    )

    assert not (tmp_path / expected).exists()
    assert not (tmp_path / "encoding_input_manifest.json").exists()
    assert not (tmp_path / "final_video_manifest.json").exists()
    assert not reservation.index_path.exists()
