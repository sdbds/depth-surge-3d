from __future__ import annotations

import json
import math
import os

import pytest

from src.depth_surge_3d.core.settings import validate_settings
from src.depth_surge_3d.io.canonical_json import CanonicalJsonError, strict_canonical_json_loads
from src.depth_surge_3d.io.settings_artifact import (
    PROCESSING_SETTINGS_V5_KEYS,
    ProcessingSettingsArtifactV5,
    SettingsArtifactSchemaError,
    attempt_duration_ms_v1,
    create_processing_settings_artifact_v5,
    duration_seconds_from_ms_v1,
    job_output_name_v1,
    migrate_processing_settings_artifact_v5,
    utc_ordinal_usec_v1,
    validate_portable_final_component_v1,
)
from src.depth_surge_3d.io import reservation_protocol


def _artifact(**overrides: object) -> ProcessingSettingsArtifactV5:
    values = {
        "batch_name": "fixture",
        "source_video": "C:/input/CON.foo.mp4",
        "source_video_name": "CON.foo.mp4",
        "source_video_fingerprint": "ab" * 16,
        "video_properties": {
            "width": 1920,
            "height": 1080,
            "fps": 23.976,
            "frame_count": 12,
            "duration": 0.5005005005005005,
            "codec": 875967048,
            "sample_aspect_ratio_numerator": 1,
            "sample_aspect_ratio_denominator": 1,
            "sample_aspect_ratio": "1:1",
        },
        "processing_settings": validate_settings({}, source="explicit"),
        "output_directory": "C:/output",
        "project_version": "0.9.0",
        "creation_now_usec": "2026-08-24T12:34:56.123456Z",
        "job_id": "01" * 16,
        "job_reservation_generation": "02" * 16,
        "job_terminal_reservation_id": "03" * 16,
    }
    values.update(overrides)
    return create_processing_settings_artifact_v5(**values)


def test_creation_has_exact_closed_schema_and_canonical_fingerprint() -> None:
    artifact = _artifact()
    value = artifact.to_object()

    assert set(value) == {
        "content_fingerprint",
        "metadata",
        "video_properties",
        "processing_settings",
        "output_info",
        "runtime_info",
    }
    assert set(value["processing_settings"]) == PROCESSING_SETTINGS_V5_KEYS
    assert len(value["processing_settings"]) == 56
    assert value["metadata"]["settings_revision"] == 0
    assert value["metadata"]["processing_attempt"] == 1
    assert value["metadata"]["processing_status"] == "in_progress"
    assert value["output_info"]["expected_final_relative_path"] == (
        "CON~f5db0cac5622670434fbba171afe5a92_3D_side-by-side_auto.mp4"
    )

    raw = artifact.encode()
    assert raw == json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    assert not raw.endswith(b"\n")
    assert ProcessingSettingsArtifactV5.parse(raw).encode() == raw


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":1}',
        b'{"a": 1}',
        b'{"a":"\\ud800"}',
        b'{"a":NaN}',
        b"\xef\xbb\xbf{}",
        b"{}\n",
    ],
)
def test_strict_canonical_json_rejects_duplicates_noncanonical_and_nonscalar(raw: bytes) -> None:
    with pytest.raises(CanonicalJsonError):
        strict_canonical_json_loads(raw)


def test_schema_rejects_extra_missing_and_integer_for_binary64() -> None:
    value = _artifact().to_object()
    value["metadata"]["extra"] = 1
    with pytest.raises(SettingsArtifactSchemaError):
        ProcessingSettingsArtifactV5.from_object(value)

    value = _artifact().to_object()
    del value["runtime_info"]["frames_processed"]
    with pytest.raises(SettingsArtifactSchemaError):
        ProcessingSettingsArtifactV5.from_object(value)

    value = _artifact().to_object()
    value["video_properties"]["fps"] = 24
    with pytest.raises(SettingsArtifactSchemaError):
        ProcessingSettingsArtifactV5.from_object(value)


def test_utc_integer_duration_and_binary64_projection_are_exact() -> None:
    start = "2024-02-29T23:59:59.999999Z"
    end = "2024-03-01T00:00:00.001999Z"
    assert utc_ordinal_usec_v1(end) - utc_ordinal_usec_v1(start) == 2_000
    assert attempt_duration_ms_v1(end, start) == 2

    seconds = duration_seconds_from_ms_v1(9_007_199_254_740_993)
    assert isinstance(seconds, float)
    assert math.isfinite(seconds)


def test_typed_completed_transition_increments_revision_and_preserves_identity() -> None:
    initial = _artifact()
    completed = initial.completed(
        terminal_at="2026-08-24T12:35:00.123456Z",
        frames_processed=12,
        cleanup_status="not_requested",
    )

    before = initial.to_object()
    after = completed.to_object()
    assert after["metadata"]["settings_revision"] == 1
    assert after["metadata"]["processing_status"] == "completed"
    assert after["metadata"]["created_at"] == before["metadata"]["created_at"]
    assert after["metadata"]["attempt_started_at"] == before["metadata"]["attempt_started_at"]
    assert after["metadata"]["last_updated_at"] == after["metadata"]["terminal_at"]
    assert after["runtime_info"]["frames_processed"] == 12
    assert (
        after["runtime_info"]["final_output_relative_path"]
        == after["output_info"]["expected_final_relative_path"]
    )
    assert ProcessingSettingsArtifactV5.parse(completed.encode()) == completed


def test_completed_transition_accepts_selected_clip_frame_count() -> None:
    initial = _artifact(
        video_properties={
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "frame_count": 473,
            "duration": 15.766666666666667,
            "codec": 875967048,
            "sample_aspect_ratio_numerator": 1,
            "sample_aspect_ratio_denominator": 1,
            "sample_aspect_ratio": "1:1",
        },
        processing_settings=validate_settings(
            {"end_time": "00:15"}, source="explicit"
        ),
    )

    completed = initial.completed(
        terminal_at="2026-08-24T12:35:00.123456Z",
        frames_processed=450,
        cleanup_status="not_requested",
    )

    assert completed.to_object()["runtime_info"]["frames_processed"] == 450


def test_output_name_and_portable_component_freeze_device_and_unicode_rules() -> None:
    goldens = {
        "CON.foo.mp4": "CON~afc0ef4e07a53c5302a14d2d1fd02201_3D_side-by-side_16x9-1080p.mp4",
        "con.foo.mp4": "con~3f9776dda8dc90294815dd61a8c3ace9_3D_side-by-side_16x9-1080p.mp4",
        "AUX.bar.mp4": "AUX~e8fac5846b90380bd5f72d652d5736c4_3D_side-by-side_16x9-1080p.mp4",
        "NUL..mp4": "NUL~9143c618d678d7b537d35615d8540440_3D_side-by-side_16x9-1080p.mp4",
        "COM1.part.mp4": "COM1~fe6252979b85daead85a257e026e2f3e_3D_side-by-side_16x9-1080p.mp4",
        "LPT9.part.mp4": "LPT9~7fc1ea62997d746b58e6250491ee1b78_3D_side-by-side_16x9-1080p.mp4",
        "COM\u00b9.foo.mp4": "COM\u00b9~f98034cae9fd53d32245fa532dab5a16_3D_side-by-side_16x9-1080p.mp4",
        "COM\u00b2.part.mp4": "COM\u00b2~94e29ca23c82f11626cf8a52a80b5dda_3D_side-by-side_16x9-1080p.mp4",
        "COM\u00b3.part.mp4": "COM\u00b3~21b0d5fa00b2d2aa801dc222e836bd0d_3D_side-by-side_16x9-1080p.mp4",
        "LPT\u00b9.foo.mp4": "LPT\u00b9~95b3c94fd50b2c12b8af03aa3028b2a6_3D_side-by-side_16x9-1080p.mp4",
        "LPT\u00b2.part.mp4": "LPT\u00b2~fee1cd545f4c01fc702da4e08e0a1b40_3D_side-by-side_16x9-1080p.mp4",
        "LPT\u00b3.part.mp4": "LPT\u00b3~7a788f8c065d09115bb8cddbe4b0b354_3D_side-by-side_16x9-1080p.mp4",
    }
    for source, expected in goldens.items():
        assert job_output_name_v1(source, "side_by_side", "16x9-1080p") == expected
    assert job_output_name_v1("COM\u00b3.mp4", "side_by_side", "16x9-1080p") == (
        "COM\u00b3_3D_side-by-side_16x9-1080p.mp4"
    )
    validate_portable_final_component_v1("normal_3D_side-by-side_auto.mp4")
    with pytest.raises(SettingsArtifactSchemaError):
        validate_portable_final_component_v1("LPT\u00b2.foo.mp4")


def _allocation_adapter(path, logical: int) -> reservation_protocol.ReservationAllocationEvidence:
    assert path.stat().st_size == logical
    return reservation_protocol.ReservationAllocationEvidence(
        platform="test",
        logical_coverage_byte_count=logical,
        allocated_byte_count=logical,
        token="settings-ready-v1",
    )


def _bootstrap_settings(tmp_path, artifact: ProcessingSettingsArtifactV5) -> None:
    reservation_protocol.bootstrap_settings_job_v1(
        job_root=tmp_path,
        artifact=artifact,
        initial_settings_reservation_id="77" * 16,
        allocation_adapter=_allocation_adapter,
        allocation_unit=4096,
    )


def test_settings_transaction_recovers_exact_durable_semantic_payload(tmp_path) -> None:
    initial = _artifact()
    target_path = tmp_path / "processing-settings-v5.json"
    _bootstrap_settings(tmp_path, initial)
    intended = initial.with_producer_marker(
        generation="44" * 16,
        updated_at="2026-08-24T12:34:57.123456Z",
    )
    transaction = reservation_protocol.SettingsArtifactTransactionV1.begin(
        job_root=tmp_path,
        current=initial,
        target=intended,
        allocation_adapter=_allocation_adapter,
        allocation_unit=4096,
        reservation_generation="55" * 16,
        reservation_id="66" * 16,
    )

    with pytest.raises(reservation_protocol.InjectedReservationCrash):
        transaction.commit(crash_at="after-truncate-sync")

    recovered = reservation_protocol.SettingsArtifactTransactionV1.recover(
        tmp_path,
        allocation_adapter=lambda *_args: (_ for _ in ()).throw(
            AssertionError("complete-short recovery must not rerun readiness evidence")
        ),
    )
    assert recovered == intended
    assert target_path.read_bytes() == intended.encode()
    assert not (
        tmp_path / ".depth-surge-reservations-v1" / "settings-transition-reservation-v1.json"
    ).exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows replacement gaps are platform-specific")
def test_settings_transaction_recovers_missing_fixed_target_from_descriptor(tmp_path) -> None:
    initial = _artifact()
    target_path = tmp_path / "processing-settings-v5.json"
    _bootstrap_settings(tmp_path, initial)
    intended = initial.with_producer_marker(
        generation="44" * 16,
        updated_at="2026-08-24T12:34:57.123456Z",
    )
    transaction = reservation_protocol.SettingsArtifactTransactionV1.begin(
        job_root=tmp_path,
        current=initial,
        target=intended,
        allocation_adapter=_allocation_adapter,
        allocation_unit=4096,
        reservation_generation="55" * 16,
        reservation_id="66" * 16,
    )
    transaction.transaction.stage_complete_full(intended.encode())
    transaction.transaction.truncate_to_payload(intended.encode())
    transaction.transaction._short_release_recovery(intended.encode())
    target_path.unlink()

    recovered = reservation_protocol.SettingsArtifactTransactionV1.recover(
        tmp_path,
        allocation_adapter=lambda *_args: (_ for _ in ()).throw(
            AssertionError("replacement-gap recovery must not rerun readiness evidence")
        ),
    )

    assert recovered == intended
    assert target_path.read_bytes() == intended.encode()


def test_settings_transaction_recovers_ready_zero_extent_from_durable_intent(tmp_path) -> None:
    initial = _artifact()
    target_path = tmp_path / "processing-settings-v5.json"
    _bootstrap_settings(tmp_path, initial)
    intended = initial.with_processing_settings(
        validate_settings(
            {"stereo_render_mode": "quality", "occlusion_fill_max_px": 7},
            source="explicit",
        ),
        updated_at="2026-08-24T12:34:57.123456Z",
    )
    reservation_protocol.SettingsArtifactTransactionV1.begin(
        job_root=tmp_path,
        current=initial,
        target=intended,
        allocation_adapter=_allocation_adapter,
        allocation_unit=4096,
        reservation_generation="55" * 16,
        reservation_id="66" * 16,
    )

    recovered = reservation_protocol.SettingsArtifactTransactionV1.recover(
        tmp_path,
        allocation_adapter=_allocation_adapter,
    )

    assert recovered == intended
    assert target_path.read_bytes() == intended.encode()


def test_settings_transaction_rejects_a_different_complete_source_intent(tmp_path) -> None:
    initial = _artifact()
    target_path = tmp_path / "processing-settings-v5.json"
    _bootstrap_settings(tmp_path, initial)
    intended = initial.with_producer_marker(
        generation="44" * 16,
        updated_at="2026-08-24T12:34:57.123456Z",
    )
    different = initial.with_producer_marker(
        generation="45" * 16,
        updated_at="2026-08-24T12:34:58.123456Z",
    )
    transaction = reservation_protocol.SettingsArtifactTransactionV1.begin(
        job_root=tmp_path,
        current=initial,
        target=intended,
        allocation_adapter=_allocation_adapter,
        allocation_unit=4096,
        reservation_generation="55" * 16,
        reservation_id="66" * 16,
    )
    with transaction.transaction.source_path.open("r+b", buffering=0) as handle:
        handle.write(different.encode())
        handle.flush()
        os.fsync(handle.fileno())

    with pytest.raises(
        reservation_protocol.ReservationConflictError,
        match="different complete intent",
    ):
        reservation_protocol.SettingsArtifactTransactionV1.recover(
            tmp_path,
            allocation_adapter=_allocation_adapter,
        )

    assert target_path.read_bytes() == initial.encode()


def test_bootstrap_publishes_locator_then_settings_and_retains_terminal_extent(tmp_path) -> None:
    artifact = _artifact()
    control = reservation_protocol.bootstrap_settings_job_v1(
        job_root=tmp_path,
        artifact=artifact,
        initial_settings_reservation_id="77" * 16,
        allocation_adapter=_allocation_adapter,
        allocation_unit=4096,
    )

    assert (tmp_path / "job-control-v1.json").is_file()
    assert (tmp_path / "processing-settings-v5.json").read_bytes() == artifact.encode()
    assert control.job_id == artifact.to_object()["metadata"]["job_id"]
    terminal_descriptor = (
        tmp_path
        / ".depth-surge-reservations-v1"
        / (
            "d-p-none-r-"
            + artifact.to_object()["metadata"]["job_reservation_generation"]
            + "-job_terminal_settings-"
            + artifact.to_object()["metadata"]["job_terminal_reservation_id"]
            + ".json"
        )
    )
    assert terminal_descriptor.is_file()
    recovered = reservation_protocol.recover_settings_job_v1(
        tmp_path,
        allocation_adapter=_allocation_adapter,
    )
    assert recovered == artifact
    assert terminal_descriptor.is_file()


@pytest.mark.parametrize("stop_event", ["initial-settings:ready", "terminal-settings:ready"])
def test_bootstrap_recovery_materializes_both_extents_before_settings(
    tmp_path,
    stop_event: str,
) -> None:
    artifact = _artifact()

    def crash(event: str) -> None:
        if event == stop_event:
            raise RuntimeError(stop_event)

    with pytest.raises(RuntimeError, match=stop_event):
        reservation_protocol.bootstrap_settings_job_v1(
            job_root=tmp_path,
            artifact=artifact,
            initial_settings_reservation_id="77" * 16,
            allocation_adapter=_allocation_adapter,
            allocation_unit=4096,
            event=crash,
        )

    recovered = reservation_protocol.recover_settings_job_v1(
        tmp_path,
        allocation_adapter=_allocation_adapter,
    )

    assert recovered == artifact
    assert (tmp_path / "processing-settings-v5.json").read_bytes() == artifact.encode()
    metadata = artifact.to_object()["metadata"]
    terminal_descriptor = (
        tmp_path
        / ".depth-surge-reservations-v1"
        / (
            f"d-p-none-r-{metadata['job_reservation_generation']}-"
            f"job_terminal_settings-{metadata['job_terminal_reservation_id']}.json"
        )
    )
    assert terminal_descriptor.is_file()


@pytest.mark.skipif(os.name != "nt", reason="Windows replacement gaps are platform-specific")
def test_bootstrap_initial_replacement_gap_is_forward_recovered_before_terminal_check(
    tmp_path,
) -> None:
    artifact = _artifact()

    def crash(event: str) -> None:
        if event == "terminal-settings:ready":
            raise RuntimeError(event)

    with pytest.raises(RuntimeError, match="terminal-settings:ready"):
        reservation_protocol.bootstrap_settings_job_v1(
            job_root=tmp_path,
            artifact=artifact,
            initial_settings_reservation_id="77" * 16,
            allocation_adapter=_allocation_adapter,
            allocation_unit=4096,
            event=crash,
        )
    control = reservation_protocol.load_job_control(tmp_path)
    initial_entry = control.bootstrap_extent_entries[0]
    initial = reservation_protocol.ReservedArtifactTransaction.load(
        tmp_path / initial_entry["descriptor_relative_path"],
        allocation_adapter=_allocation_adapter,
    )
    initial.stage_complete_full(artifact.encode())
    initial.truncate_to_payload(artifact.encode())
    initial._short_release_recovery(artifact.encode())
    (tmp_path / "processing-settings-v5.json").unlink()

    recovered = reservation_protocol.recover_settings_job_v1(
        tmp_path,
        allocation_adapter=_allocation_adapter,
    )

    assert recovered == artifact
    assert (tmp_path / "processing-settings-v5.json").read_bytes() == artifact.encode()


def test_lifecycle_terminal_extent_recovers_short_failed_intent(tmp_path) -> None:
    initial = _artifact()
    reservation_protocol.bootstrap_settings_job_v1(
        job_root=tmp_path,
        artifact=initial,
        initial_settings_reservation_id="77" * 16,
        allocation_adapter=_allocation_adapter,
        allocation_unit=4096,
    )
    failed = initial.failed(
        terminal_at="2026-08-24T12:35:01.123456Z",
        diagnostic_code="VIDEO_ENCODING_FAILED",
        diagnostic_message="encoder returned a nonzero status",
    )

    with pytest.raises(reservation_protocol.InjectedReservationCrash):
        reservation_protocol.commit_terminal_settings_v1(
            job_root=tmp_path,
            current=initial,
            target=failed,
            allocation_adapter=_allocation_adapter,
            crash_at="after-truncate-sync",
        )

    recovered = reservation_protocol.recover_settings_job_v1(
        tmp_path,
        allocation_adapter=lambda *_args: (_ for _ in ()).throw(
            AssertionError("complete-short recovery must not rerun readiness evidence")
        ),
    )
    assert recovered == failed
    assert (tmp_path / "processing-settings-v5.json").read_bytes() == failed.encode()


def test_lifecycle_terminal_extent_recovers_complete_full_intent(tmp_path) -> None:
    initial = _artifact()
    reservation_protocol.bootstrap_settings_job_v1(
        job_root=tmp_path,
        artifact=initial,
        initial_settings_reservation_id="77" * 16,
        allocation_adapter=_allocation_adapter,
        allocation_unit=4096,
    )
    failed = initial.failed(
        terminal_at="2026-08-24T12:35:01.123456Z",
        diagnostic_code="IO_ERROR",
    )

    with pytest.raises(reservation_protocol.InjectedReservationCrash):
        reservation_protocol.commit_terminal_settings_v1(
            job_root=tmp_path,
            current=initial,
            target=failed,
            allocation_adapter=_allocation_adapter,
            crash_at="after-complete-full-sync",
        )

    assert (
        reservation_protocol.recover_settings_job_v1(
            tmp_path,
            allocation_adapter=_allocation_adapter,
        )
        == failed
    )


def test_attempt_restart_updates_settings_and_recreates_terminal_extent(tmp_path) -> None:
    initial = _artifact()
    reservation_protocol.bootstrap_settings_job_v1(
        job_root=tmp_path,
        artifact=initial,
        initial_settings_reservation_id="77" * 16,
        allocation_adapter=_allocation_adapter,
        allocation_unit=4096,
    )
    failed = initial.failed(
        terminal_at="2026-08-24T12:35:01.123456Z",
        diagnostic_code="IO_ERROR",
    )
    reservation_protocol.commit_terminal_settings_v1(
        job_root=tmp_path,
        current=initial,
        target=failed,
        allocation_adapter=_allocation_adapter,
    )
    assert (
        reservation_protocol.recover_settings_job_v1(
            tmp_path,
            allocation_adapter=_allocation_adapter,
        )
        == failed
    )

    changed = validate_settings(
        {"stereo_render_mode": "quality", "occlusion_fill_max_px": 7},
        source="explicit",
    )
    restarted = failed.restart(
        started_at="2026-08-24T12:36:00.123456Z",
        processing_settings=changed,
    )
    transaction = reservation_protocol.SettingsArtifactTransactionV1.begin(
        job_root=tmp_path,
        current=failed,
        target=restarted,
        allocation_adapter=_allocation_adapter,
        allocation_unit=4096,
        reservation_generation="88" * 16,
        reservation_id="99" * 16,
    )
    transaction.commit()

    recovered = reservation_protocol.recover_settings_job_v1(
        tmp_path,
        allocation_adapter=_allocation_adapter,
    )
    metadata = initial.to_object()["metadata"]
    terminal_descriptor = (
        tmp_path
        / ".depth-surge-reservations-v1"
        / (
            f"d-p-none-r-{metadata['job_reservation_generation']}-job_terminal_settings-"
            f"{metadata['job_terminal_reservation_id']}.json"
        )
    )
    assert recovered == restarted
    assert recovered.to_object()["processing_settings"]["stereo_render_mode"] == "quality"
    assert recovered.to_object()["metadata"]["processing_attempt"] == 2
    assert terminal_descriptor.is_file()


@pytest.mark.parametrize("crash_at", [None, "after-target-commit"])
def test_attempt_restart_recovers_before_and_after_settings_commit(
    tmp_path,
    crash_at: str | None,
) -> None:
    initial = _artifact()
    reservation_protocol.bootstrap_settings_job_v1(
        job_root=tmp_path,
        artifact=initial,
        initial_settings_reservation_id="77" * 16,
        allocation_adapter=_allocation_adapter,
        allocation_unit=4096,
    )
    failed = initial.failed(
        terminal_at="2026-08-24T12:35:01.123456Z",
        diagnostic_code="IO_ERROR",
    )
    reservation_protocol.commit_terminal_settings_v1(
        job_root=tmp_path,
        current=initial,
        target=failed,
        allocation_adapter=_allocation_adapter,
    )
    reservation_protocol.recover_settings_job_v1(
        tmp_path,
        allocation_adapter=_allocation_adapter,
    )
    restarted = failed.restart(started_at="2026-08-24T12:36:00.123456Z")
    transaction = reservation_protocol.SettingsArtifactTransactionV1.begin(
        job_root=tmp_path,
        current=failed,
        target=restarted,
        allocation_adapter=_allocation_adapter,
        allocation_unit=4096,
        reservation_generation="88" * 16,
        reservation_id="99" * 16,
    )

    if crash_at is not None:
        with pytest.raises(reservation_protocol.InjectedReservationCrash):
            transaction.commit(crash_at=crash_at)

    recovered = reservation_protocol.recover_settings_job_v1(
        tmp_path,
        allocation_adapter=_allocation_adapter,
    )

    assert recovered == restarted
    assert not (
        tmp_path / ".depth-surge-reservations-v1/settings-transition-reservation-v1.json"
    ).exists()


def test_in_progress_settings_rewrite_preserves_attempt_and_terminal_reserve() -> None:
    initial = _artifact()
    changed = validate_settings(
        {"stereo_render_mode": "quality", "occlusion_fill_max_px": 6},
        source="explicit",
    )

    rewritten = initial.with_processing_settings(
        changed,
        updated_at="2026-08-24T12:35:00.123456Z",
    )

    before = initial.to_object()
    after = rewritten.to_object()
    assert rewritten.status == "in_progress"
    assert rewritten.revision == initial.revision + 1
    assert after["metadata"]["processing_attempt"] == before["metadata"]["processing_attempt"]
    assert after["metadata"]["attempt_started_at"] == before["metadata"]["attempt_started_at"]
    assert after["processing_settings"]["stereo_render_mode"] == "quality"


def test_operations_new_job_path_uses_schema5_locator_and_terminal_extent(tmp_path) -> None:
    from src.depth_surge_3d.io.operations import (
        commit_final_media_producer_marker,
        load_processing_settings,
        prepare_processing_settings_artifact,
        save_processing_settings,
        update_processing_status,
    )

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source-fixture")
    output = tmp_path / "job"
    video_properties = {
        "width": 8,
        "height": 8,
        "fps": 24.0,
        "frame_count": 2,
        "duration": 2.0 / 24.0,
        "codec": 0,
        "sample_aspect_ratio_numerator": 1,
        "sample_aspect_ratio_denominator": 1,
        "sample_aspect_ratio": "1:1",
    }
    path = save_processing_settings(
        output,
        "job",
        validate_settings({}, source="explicit"),
        video_properties,
        str(source),
        transactional_v5=True,
    )

    assert path == output / "processing-settings-v5.json"
    assert (output / "job-control-v1.json").is_file()
    assert load_processing_settings(path)["metadata"]["processing_status"] == "in_progress"
    marked = commit_final_media_producer_marker(path, generation="99" * 16)
    assert marked.to_object()["metadata"]["final_media_producer_contract_version"] == 4
    assert marked.to_object()["metadata"]["final_media_publication_generation"] == "99" * 16
    assert update_processing_status(
        path,
        "failed",
        {"terminal_diagnostic_code": "TEST_FAILURE", "error": "fixture"},
    )
    assert ProcessingSettingsArtifactV5.parse(path.read_bytes()).status == "failed"

    restarted = prepare_processing_settings_artifact(
        path,
        settings=validate_settings(
            {"stereo_render_mode": "quality", "occlusion_fill_max_px": 7},
            source="explicit",
        ),
        video_properties=video_properties,
        source_video_path=str(source),
    )
    assert restarted.status == "in_progress"
    assert restarted.to_object()["metadata"]["processing_attempt"] == 2
    assert restarted.to_object()["processing_settings"]["stereo_render_mode"] == "quality"
    assert (
        reservation_protocol.recover_settings_job_v1(
            output,
            allocation_adapter=reservation_protocol.platform_allocation_adapter,
            allocation_counter=reservation_protocol.platform_allocation_counter,
        )
        == restarted
    )


def test_operations_completed_default_uses_selected_clip_frame_count(tmp_path) -> None:
    from src.depth_surge_3d.io.operations import (
        save_processing_settings,
        update_processing_status,
    )

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source-fixture")
    output = tmp_path / "job"
    settings = validate_settings(
        {"end_time": "00:15", "keep_intermediates": True}, source="explicit"
    )
    path = save_processing_settings(
        output,
        "job",
        settings,
        {
            "width": 1920,
            "height": 1080,
            "fps": 30.0,
            "frame_count": 473,
            "duration": 15.766666666666667,
            "codec": 875967048,
            "sample_aspect_ratio_numerator": 1,
            "sample_aspect_ratio_denominator": 1,
            "sample_aspect_ratio": "1:1",
        },
        str(source),
        transactional_v5=True,
    )

    assert update_processing_status(path, "completed")
    completed = ProcessingSettingsArtifactV5.parse(path.read_bytes())
    assert completed.to_object()["runtime_info"]["frames_processed"] == 450


def test_legacy_completed_migration_is_revision_zero_and_preserves_target() -> None:
    legacy = {
        "metadata": {
            "settings_schema_version": 4,
            "batch_name": "old-job",
            "source_video": "D:/archive/input.mp4",
            "source_video_name": "input.mp4",
            "source_video_fingerprint_algorithm": "file-sample-blake2b-v1",
            "source_video_fingerprint": "aa" * 16,
            "project_version": "0.8.0",
            "created_timestamp": 1_700_000_000.25,
            "completed_timestamp": 1_700_000_010.75,
            "processing_status": "completed",
        },
        "video_properties": {
            "width": 1920,
            "height": 1080,
            "fps": 24.0,
            "frame_count": 240,
            "codec": 0,
            "sample_aspect_ratio": "1:1",
        },
        "processing_settings": {},
        "output_info": {
            "output_directory": "D:/archive/output",
            "expected_output_filename": "historical name .mp4",
        },
        "runtime_info": {"frames_processed": 238},
    }
    settings = validate_settings(
        {"keep_intermediates": False},
        source="legacy_disk",
    )
    migrated = migrate_processing_settings_artifact_v5(
        legacy_settings=legacy,
        migrated_processing_settings=settings,
        migration_now_usec="2026-08-24T12:34:56.123456Z",
        job_id="10" * 16,
        job_reservation_generation="11" * 16,
        job_terminal_reservation_id="12" * 16,
    )
    value = migrated.to_object()

    assert migrated.revision == 0
    assert migrated.status == "completed"
    assert value["metadata"]["source_settings_schema_version"] == 4
    assert value["metadata"]["processing_attempt"] == 1
    assert value["metadata"]["cleanup_status"] == "incomplete_error"
    assert value["output_info"]["expected_final_relative_path"] == "historical name .mp4"
    assert value["output_info"]["output_name_algorithm_version"] == (
        "persisted-expected-output-filename-v1"
    )
    assert value["runtime_info"]["frames_processed"] == 238
    assert ProcessingSettingsArtifactV5.parse(migrated.encode()) == migrated
