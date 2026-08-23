"""Task 0 crash/recovery contracts for indexed reservation extents and publication."""

from __future__ import annotations

import errno
from pathlib import Path

import pytest

from scripts.quality_task0 import reservation_protocol


def _adapter(path: Path, logical: int) -> reservation_protocol.ReservationAllocationEvidence:
    assert path.stat().st_size == logical
    return reservation_protocol.ReservationAllocationEvidence(
        platform="test",
        logical_coverage_byte_count=logical,
        allocated_byte_count=logical,
        token="ready-v1",
    )


def test_reservation_forecast_keeps_payload_padding_and_directory_entries_distinct() -> None:
    allocation_unit = 4096

    def allocate(raw: int) -> int:
        return ((raw + allocation_unit - 1) // allocation_unit) * allocation_unit

    assert reservation_protocol.payload_logical_bytes(4217, allocation_unit) == 12_288
    assert reservation_protocol.payload_extent_create_charge(4217, allocation_unit) == 16_384
    assert reservation_protocol.descriptor_create_new_charge(3000, allocation_unit) == 12_288
    assert (
        reservation_protocol.placeholder_entry_charge("replace_placeholder", allocation_unit)
        == 4096
    )
    assert (
        reservation_protocol.reservation_extent_charge(
            4217,
            3000,
            "replace_placeholder",
            allocation_unit,
        )
        == allocate(4217) + allocate(3000) + 5 * allocation_unit
    )
    assert reservation_protocol.reservation_directory_bootstrap_forecast(allocation_unit) == 8192


def test_forecast_is_explicitly_not_a_strict_directory_growth_proof() -> None:
    forecast = reservation_protocol.minimum_admission_forecast(
        payload_raw=100,
        descriptor_raw=100,
        method="replace_placeholder",
        allocation_unit=4096,
    )

    assert forecast.is_strict_directory_upper_bound is False
    assert forecast.namespace_enospc_error is reservation_protocol.ReservationIncompleteError


def test_canonical_payload_frame_distinguishes_zero_full_partial_and_short() -> None:
    payload = b'{"schema_version":1}'
    logical = 128

    assert reservation_protocol.classify_payload_frame(b"\x00" * logical, payload) == "zero"
    assert (
        reservation_protocol.classify_payload_frame(
            payload + b"\x00" * (logical - len(payload)), payload
        )
        == "complete-full"
    )
    assert (
        reservation_protocol.classify_payload_frame(
            payload + b"\x00x" + b"\x00" * (logical - len(payload) - 2), payload
        )
        == "partial"
    )
    assert reservation_protocol.classify_payload_frame(payload, payload) == "complete-short"


def _create_transaction(
    tmp_path: Path,
    *,
    counter=None,
) -> reservation_protocol.ReservedArtifactTransaction:
    intent_path = tmp_path / ".depth-surge-reservations-v1/final-index.json"
    reservation_protocol.publish_write_ahead_intent(
        intent_path,
        {"algorithm_version": "test-final-index-v1", "generation": "ab" * 16},
    )
    return reservation_protocol.ReservedArtifactTransaction.create(
        job_root=tmp_path,
        index_relative_path=Path(".depth-surge-reservations-v1/final-index.json"),
        target_relative_path=Path("encoding_input_manifest.json"),
        reservation_kind="encoding_input_manifest",
        payload_raw_max=4217,
        allocation_unit=4096,
        allocation_adapter=_adapter,
        allocation_counter=counter,
        generation="cd" * 16,
        reservation_id="ef" * 16,
    )


def test_write_ahead_index_is_durable_before_placeholder_extent_and_descriptor(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    index = tmp_path / ".depth-surge-reservations-v1/final-index.json"
    reservation_protocol.publish_write_ahead_intent(
        index,
        {"algorithm_version": "test-final-index-v1", "generation": "ab" * 16},
        event=events.append,
    )
    transaction = reservation_protocol.ReservedArtifactTransaction.create(
        job_root=tmp_path,
        index_relative_path=index.relative_to(tmp_path),
        target_relative_path=Path("manifest.json"),
        reservation_kind="encoding_input_manifest",
        payload_raw_max=100,
        allocation_unit=4096,
        allocation_adapter=_adapter,
        generation="cd" * 16,
        reservation_id="ef" * 16,
        event=events.append,
    )

    assert events.index("index:published") < events.index("placeholder:created")
    assert events.index("placeholder:created") < events.index("extent:created")
    assert events.index("extent:ready") < events.index("descriptor:published")
    assert transaction.metrics.reservation_write_bytes == transaction.descriptor.logical_byte_count
    assert transaction.metrics.reservation_fsync_count == 1


def test_truncate_fsync_crash_recovers_short_source_without_readiness_adapter(
    tmp_path: Path,
) -> None:
    adapter_calls = 0

    def adapter(path: Path, logical: int) -> reservation_protocol.ReservationAllocationEvidence:
        nonlocal adapter_calls
        adapter_calls += 1
        return _adapter(path, logical)

    transaction = _create_transaction(tmp_path)
    transaction.allocation_adapter = adapter
    payload = b'{"schema_version":1,"value":"ready"}'

    with pytest.raises(reservation_protocol.InjectedReservationCrash):
        transaction.consume(payload, crash_at="after-truncate-sync")
    calls_at_crash = adapter_calls
    loaded = reservation_protocol.ReservedArtifactTransaction.load(
        transaction.descriptor_path,
        allocation_adapter=lambda *_args: (_ for _ in ()).throw(
            AssertionError("short recovery must not call readiness adapter")
        ),
    )
    phase = loaded.recover(payload)

    assert phase == "committed-target"
    assert adapter_calls == calls_at_crash
    assert loaded.target_path.read_bytes() == payload
    assert not loaded.source_path.exists()


def test_rename_parent_sync_crash_recovers_committed_target_without_full_adapter(
    tmp_path: Path,
) -> None:
    transaction = _create_transaction(tmp_path)
    payload = b'{"schema_version":1,"value":"renamed"}'

    with pytest.raises(reservation_protocol.InjectedReservationCrash):
        transaction.consume(payload, crash_at="after-rename-parent-sync")
    loaded = reservation_protocol.ReservedArtifactTransaction.load(
        transaction.descriptor_path,
        allocation_adapter=lambda *_args: (_ for _ in ()).throw(
            AssertionError("target recovery must not call readiness adapter")
        ),
    )

    assert loaded.recover(payload) == "committed-target"
    assert loaded.target_path.read_bytes() == payload
    assert not loaded.descriptor_path.exists()


def test_truncate_before_sync_uses_short_release_recovery_and_preserves_on_failure(
    tmp_path: Path,
) -> None:
    counters = iter([12_288, 12_288])
    transaction = _create_transaction(tmp_path, counter=lambda _path: next(counters))
    payload = b'{"schema_version":1,"value":"short"}'

    with pytest.raises(reservation_protocol.InjectedReservationCrash):
        transaction.consume(payload, crash_at="after-truncate")
    loaded = reservation_protocol.ReservedArtifactTransaction.load(
        transaction.descriptor_path,
        allocation_adapter=lambda *_args: (_ for _ in ()).throw(AssertionError("forbidden")),
        allocation_counter=lambda _path: next(counters),
    )

    with pytest.raises(reservation_protocol.ReservationIncompleteError, match="release"):
        loaded.recover(payload)

    assert loaded.source_path.read_bytes() == payload
    assert loaded.target_path.stat().st_size == 0
    assert loaded.descriptor_path.exists()


def test_readiness_evidence_is_strict_only_while_full_source_exists(tmp_path: Path) -> None:
    transaction = _create_transaction(tmp_path)
    payload = b'{"schema_version":1,"value":"phase"}'
    transaction.stage_complete_full(payload)
    transaction.allocation_adapter = (
        lambda path, logical: reservation_protocol.ReservationAllocationEvidence(
            platform="test",
            logical_coverage_byte_count=logical,
            allocated_byte_count=logical - 1,
            token="changed",
        )
    )

    with pytest.raises(reservation_protocol.ReservationEvidenceConflictError):
        transaction.validate_physical_phase(payload)

    transaction.allocation_adapter = _adapter
    transaction.truncate_to_payload(payload)
    transaction.allocation_adapter = lambda *_args: (_ for _ in ()).throw(
        AssertionError("short phase must not invoke readiness evidence")
    )
    assert transaction.validate_physical_phase(payload) == "short-source"


def test_settings_write_ahead_intent_recovers_exact_target_payload(tmp_path: Path) -> None:
    payload = {
        "settings_revision": 7,
        "processing_status": "failed",
        "terminal_diagnostic": "IO_ERROR",
    }
    intent = reservation_protocol.SettingsWriteAheadIntent.create(
        reservation_generation="ab" * 16,
        target_settings_payload=payload,
    )
    path = tmp_path / ".depth-surge-reservations-v1/settings-transition-reservation-v1.json"
    reservation_protocol.publish_write_ahead_intent(path, intent.to_object())

    recovered = reservation_protocol.SettingsWriteAheadIntent.load(path)

    assert recovered.target_settings_payload == payload
    assert recovered.target_settings_raw_sha256 == reservation_protocol.sha256_canonical(payload)


@pytest.mark.parametrize(
    ("mapped", "last", "next_start", "expected"),
    [
        (0, False, 0, reservation_protocol.ReservationIncompleteError),
        (3, False, 4096, reservation_protocol.ReservationCapabilityError),
        (64, False, 4096, 4096),
        (64, False, 0, reservation_protocol.ReservationCapabilityError),
        (3, True, 4096, "done"),
    ],
)
def test_fiemap_pagination_has_one_disposition_for_every_batch_shape(
    mapped: int,
    last: bool,
    next_start: int,
    expected,
) -> None:
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            reservation_protocol.classify_fiemap_batch(
                mapped_extents=mapped,
                extent_capacity=64,
                last_flag=last,
                previous_start=0,
                next_start=next_start,
                coverage_complete=False,
            )
    else:
        assert (
            reservation_protocol.classify_fiemap_batch(
                mapped_extents=mapped,
                extent_capacity=64,
                last_flag=last,
                previous_start=0,
                next_start=next_start,
                coverage_complete=last,
            )
            == expected
        )


@pytest.mark.parametrize(
    ("source", "target", "api_return", "expected"),
    [
        ("S", "OLD", False, "unchanged"),
        (None, "S", False, "committed"),
        (None, "S", True, "committed"),
        ("S", None, False, "replacement-gap"),
    ],
)
def test_windows_rename_classifies_physical_identity_before_boolean_return(
    source: str | None,
    target: str | None,
    api_return: bool,
    expected: str,
) -> None:
    assert (
        reservation_protocol.classify_windows_rename_return(
            source_identity=source,
            target_identity=target,
            expected_source_identity="S",
            expected_old_target_identity="OLD",
            api_return=api_return,
        )
        == expected
    )


def test_windows_true_unchanged_return_is_protocol_error() -> None:
    with pytest.raises(reservation_protocol.WindowsRenameProtocolError):
        reservation_protocol.classify_windows_rename_return(
            source_identity="S",
            target_identity="OLD",
            expected_source_identity="S",
            expected_old_target_identity="OLD",
            api_return=True,
        )


@pytest.mark.parametrize(
    ("video", "input_manifest", "final_manifest", "sibling", "expected"),
    [
        ("new", "old", "old", False, "video-replaced-input-pending"),
        ("new", "new", "old", False, "input-manifest-committed-no-final-intent"),
        ("new", "new", "new", False, "publication-authenticated"),
        ("new", "new", "old", True, "post-boundary-retry-sibling-present"),
    ],
)
def test_irreversible_publication_keeps_actual_ordered_prefix_instead_of_rollback(
    video: str,
    input_manifest: str,
    final_manifest: str,
    sibling: bool,
    expected: str,
) -> None:
    assert (
        reservation_protocol.classify_final_publication_prefix(
            active_index=True,
            video_generation=video,
            input_manifest_generation=input_manifest,
            final_manifest_generation=final_manifest,
            sibling_present=sibling,
        )
        == expected
    )


def test_retired_mixed_generation_prefix_uses_ordinary_audit_matrix() -> None:
    assert (
        reservation_protocol.classify_final_publication_prefix(
            active_index=False,
            video_generation="new",
            input_manifest_generation="old",
            final_manifest_generation="old",
            sibling_present=False,
        )
        == "publication-validation-failed"
    )


def test_final_encoding_entry_order_keeps_job_terminal_last() -> None:
    kinds = [
        "producer_settings",
        "encoding_input_manifest",
        "final_video_manifest",
        "prune_marker",
        "diagnostics_payload_pruned",
        "cleanup_pending_settings",
        "job_terminal_settings",
    ]

    reservation_protocol.validate_final_encoding_entry_order(kinds)
    with pytest.raises(reservation_protocol.ReservationConflictError):
        reservation_protocol.validate_final_encoding_entry_order(
            ["job_terminal_settings", "encoding_input_manifest"]
        )


def test_namespace_enospc_is_retryable_incomplete_not_conflict(tmp_path: Path) -> None:
    def fail(_path: Path, _payload: dict, **_kwargs) -> None:
        raise OSError(errno.ENOSPC, "injected namespace full")

    with pytest.raises(reservation_protocol.ReservationIncompleteError):
        reservation_protocol.materialize_after_index(
            tmp_path / "index.json",
            {"generation": "ab" * 16},
            publisher=fail,
        )


def _job_control(job_root: Path) -> reservation_protocol.JobControlV1:
    return reservation_protocol.JobControlV1.create(
        job_root=job_root,
        job_id="01" * 16,
        settings_identity_fingerprint="02" * 32,
        job_reservation_generation="03" * 16,
        initial_settings_reservation_id="04" * 16,
        job_terminal_reservation_id="05" * 16,
        legacy_settings_artifact_relative_path=None,
        bootstrap_extent_entries=[
            {"reservation_kind": "initial_settings"},
            {"reservation_kind": "job_terminal_settings"},
        ],
        bootstrap_settings_intent={"target_settings_revision": 0},
    )


def test_job_control_is_the_only_settings_locator_after_bootstrap(tmp_path: Path) -> None:
    control = _job_control(tmp_path)
    reservation_protocol.publish_job_control(tmp_path, control)
    fixed = tmp_path / "processing-settings-v5.json"
    fixed.write_bytes(b"fixed")
    (tmp_path / "newer-settings.json").write_bytes(b"decoy")

    loaded = reservation_protocol.load_job_control(tmp_path)

    assert reservation_protocol.resolve_fixed_settings_path(tmp_path, loaded) == fixed
    assert loaded == control


def test_fixed_settings_locator_never_adopts_a_legacy_glob_match(tmp_path: Path) -> None:
    control = _job_control(tmp_path)
    reservation_protocol.publish_job_control(tmp_path, control)
    (tmp_path / "only-settings.json").write_bytes(b"legacy")

    with pytest.raises(reservation_protocol.ReservationConflictError, match="fixed settings"):
        reservation_protocol.resolve_fixed_settings_path(tmp_path, control)


@pytest.mark.parametrize(
    ("observation", "expected"),
    [
        (
            {
                "current_status": "in_progress",
                "current_revision": 7,
                "source_frame": "zero",
                "descriptor_present": True,
            },
            "retain-zero-terminal-reserve",
        ),
        (
            {
                "current_status": "in_progress",
                "current_revision": 7,
                "source_frame": "partial",
                "descriptor_present": True,
            },
            "reset-partial-terminal-reserve",
        ),
        (
            {
                "current_status": "in_progress",
                "current_revision": 7,
                "source_frame": "complete-short",
                "descriptor_present": True,
                "intent_status": "failed",
                "intent_revision": 8,
                "intent_is_legal": True,
            },
            "publish-exact-terminal-intent",
        ),
        (
            {
                "current_status": "completed",
                "current_revision": 8,
                "source_frame": "absent",
                "descriptor_present": True,
                "target_has_reserved_identity": True,
            },
            "retire-consumed-terminal-descriptor",
        ),
        (
            {
                "current_status": "failed",
                "current_revision": 8,
                "source_frame": "absent",
                "descriptor_present": False,
                "descriptor_temporary_present": True,
            },
            "retire-residual-terminal-temporary",
        ),
    ],
)
def test_job_control_terminal_reconciler_has_one_action_for_each_legal_state(
    observation: dict,
    expected: str,
) -> None:
    state = reservation_protocol.JobTerminalObservation(**observation)
    assert reservation_protocol.reconcile_job_control_terminal_state(state) == expected


def test_fixed_index_owns_terminal_extent_before_job_control_reconciler() -> None:
    observation = reservation_protocol.JobTerminalObservation(
        current_status="in_progress",
        current_revision=7,
        source_frame="complete-full",
        descriptor_present=True,
        intent_status="completed",
        intent_revision=8,
        intent_is_legal=True,
        fixed_index_active=True,
    )

    assert (
        reservation_protocol.reconcile_job_control_terminal_state(observation)
        == "defer-to-fixed-index"
    )
    assert reservation_protocol.recovery_audit_order()[-1] == "ordinary-artifact-audit"
    assert reservation_protocol.recovery_audit_order().index("job-control-owned-extents") < (
        reservation_protocol.recovery_audit_order().index("ordinary-artifact-audit")
    )


def test_illegal_terminal_intent_never_becomes_recovery_authority() -> None:
    observation = reservation_protocol.JobTerminalObservation(
        current_status="in_progress",
        current_revision=7,
        source_frame="complete-full",
        descriptor_present=True,
        intent_status="failed",
        intent_revision=9,
        intent_is_legal=True,
    )

    with pytest.raises(reservation_protocol.ReservationConflictError, match="terminal intent"):
        reservation_protocol.reconcile_job_control_terminal_state(observation)


@pytest.mark.parametrize(
    ("observation", "phase", "action"),
    [
        (
            {
                "video_state": "reserved-target",
                "input_manifest_state": "encoding-prepass-staged",
                "final_manifest_state": "zero",
                "sibling_present": True,
            },
            "ready_pre_video",
            "remove-sibling-and-encode",
        ),
        (
            {
                "video_state": "windows-replacement-gap",
                "input_manifest_state": "complete-full",
                "final_manifest_state": "zero",
                "sibling_present": True,
            },
            "windows_video_replace_gap_before_input_commit",
            "forward-rename-video",
        ),
        (
            {
                "video_state": "committed-new",
                "input_manifest_state": "complete-full",
                "final_manifest_state": "zero",
                "sibling_present": False,
            },
            "video_replaced_input_intent",
            "commit-input-manifest",
        ),
        (
            {
                "video_state": "committed-new",
                "input_manifest_state": "committed-new",
                "final_manifest_state": "partial",
                "sibling_present": True,
            },
            "post_boundary_retry_sibling_present",
            "reset-final-intent-remove-sibling-reencode",
        ),
        (
            {
                "video_state": "committed-new",
                "input_manifest_state": "committed-new",
                "final_manifest_state": "complete-short",
                "sibling_present": False,
            },
            "final_manifest_intent",
            "validate-and-commit-final-manifest",
        ),
        (
            {
                "video_state": "committed-new",
                "input_manifest_state": "committed-new",
                "final_manifest_state": "committed-new",
                "sibling_present": False,
            },
            "publication_authenticated",
            "continue-finalization",
        ),
    ],
)
def test_final_publication_crash_prefixes_have_one_forward_only_recovery_action(
    observation: dict,
    phase: str,
    action: str,
) -> None:
    state = reservation_protocol.FinalPublicationObservation(**observation)
    result = reservation_protocol.classify_final_publication_observation(state)

    assert result.phase == phase
    assert result.action == action


def test_post_boundary_complete_input_with_zero_final_intent_reencodes_without_rewrite() -> None:
    observation = reservation_protocol.FinalPublicationObservation(
        video_state="committed-new",
        input_manifest_state="committed-new",
        final_manifest_state="zero",
        sibling_present=False,
    )

    result = reservation_protocol.classify_final_publication_observation(observation)

    assert result.phase == "input_manifest_committed_no_final_intent"
    assert result.action == "revalidate-input-and-reencode"


def test_final_publication_rejects_complete_input_before_video_boundary() -> None:
    observation = reservation_protocol.FinalPublicationObservation(
        video_state="reserved-target",
        input_manifest_state="complete-short",
        final_manifest_state="zero",
        sibling_present=False,
    )

    with pytest.raises(reservation_protocol.ReservationConflictError):
        reservation_protocol.classify_final_publication_observation(observation)


def test_final_index_retirement_is_index_last_and_never_recreates_absent_descriptors() -> None:
    order = reservation_protocol.final_index_retirement_order(
        entry_names=("input", "final-manifest"),
        descriptor_names=("input-desc", "final-desc"),
    )

    assert order[-2:] == (
        "unlink:final-encoding-reservation-v1.json",
        "sync-and-prove-index-absent",
    )
    assert order.index("unlink:input-desc") < order.index(
        "unlink:final-encoding-reservation-v1.json"
    )
    assert all(not step.startswith("create:") for step in order)
