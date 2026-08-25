"""Production crash/recovery contracts for reservation extents and publication."""

from __future__ import annotations

import errno
import hashlib
from pathlib import Path
import os

import pytest

from src.depth_surge_3d.io import reservation_protocol
from tests.unit.test_settings_artifact_v5 import _artifact


def _adapter(path: Path, logical: int) -> reservation_protocol.ReservationAllocationEvidence:
    assert path.stat().st_size == logical
    return reservation_protocol.ReservationAllocationEvidence(
        platform="test",
        logical_coverage_byte_count=logical,
        allocated_byte_count=logical,
        token="ready-v1",
    )


def test_persisted_file_and_directory_identities_use_the_frozen_nested_schema(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"identity")

    file_identity = reservation_protocol._identity(payload)
    directory_identity = reservation_protocol._directory_identity(tmp_path)

    file_object = file_identity.to_object()
    directory_object = directory_identity.to_object()
    assert set(file_object) == {"platform", "file_identity", "link_count"}
    assert set(directory_object) == {"kind", "file_identity"}
    assert file_object["platform"] == directory_object["kind"]
    if os.name == "nt":
        assert set(file_object["file_identity"]) == {"volume_serial", "file_id"}
        assert len(file_object["file_identity"]["volume_serial"]) == 16
        assert len(file_object["file_identity"]["file_id"]) == 32
    else:
        assert set(file_object["file_identity"]) == {
            "filesystem_uuid",
            "handle_type",
            "file_handle_hex",
        }
    assert reservation_protocol.ReservationFileIdentity.from_object(file_object) == file_identity
    assert reservation_protocol.DirectoryIdentity.from_object(directory_object) == directory_identity


def test_platform_allocation_adapter_proves_a_fsynced_full_extent(tmp_path: Path) -> None:
    path = tmp_path / "allocation-evidence.bin"
    logical = reservation_protocol.payload_logical_bytes(70_000, 4096)
    with path.open("wb") as handle:
        handle.write(b"\xa5" * logical)
        handle.flush()
        os.fsync(handle.fileno())

    evidence = reservation_protocol.platform_allocation_adapter(path, logical)

    assert evidence.logical_coverage_byte_count == logical
    assert evidence.allocated_byte_count >= logical
    assert evidence.platform in {"windows-allocated-ranges-v1", "linux-fiemap-v1"}
    assert len(evidence.token) == 64
    assert reservation_protocol.platform_allocation_counter(path) >= logical


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-relative rename contract")
@pytest.mark.parametrize("target_present", [False, True])
def test_windows_same_parent_rename_moves_the_authenticated_source(
    tmp_path: Path, target_present: bool
) -> None:
    source = tmp_path / "source.tmp"
    target = tmp_path / "target.json"
    source.write_bytes(b"new-generation")
    expected_source = reservation_protocol._identity(source)
    expected_target = None
    if target_present:
        target.write_bytes(b"old-generation")
        expected_target = reservation_protocol._identity(target)

    result = reservation_protocol.windows_same_parent_rename_v1(
        source,
        target,
        expected_source_identity=expected_source,
        expected_target_identity=expected_target,
        expected_parent_identity=reservation_protocol._directory_identity(tmp_path),
        replace_existing=target_present,
    )

    assert result == "committed"
    assert not source.exists()
    assert target.read_bytes() == b"new-generation"
    assert reservation_protocol._identity(target) == expected_source


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


def test_reservation_create_propagates_cancellation_from_fixed_target_scan(
    tmp_path: Path,
) -> None:
    intent_path = tmp_path / ".depth-surge-reservations-v1/final-index.json"
    reservation_protocol.publish_write_ahead_intent(
        intent_path,
        {"algorithm_version": "test-final-index-v1", "generation": "ab" * 16},
    )
    target = tmp_path / "encoding_input_manifest.json"
    target.write_bytes(b"old-generation")

    with pytest.raises(InterruptedError, match="cancelled"):
        reservation_protocol.ReservedArtifactTransaction.create(
            job_root=tmp_path,
            index_relative_path=Path(
                ".depth-surge-reservations-v1/final-index.json"
            ),
            target_relative_path=Path("encoding_input_manifest.json"),
            reservation_kind="encoding_input_manifest",
            payload_raw_max=4217,
            allocation_unit=4096,
            allocation_adapter=_adapter,
            generation="cd" * 16,
            reservation_id="ef" * 16,
            expected_publication_method="replace_existing",
            cancelled=lambda: True,
        )

    assert target.read_bytes() == b"old-generation"
    assert not any(tmp_path.glob("*.reserved"))


def test_control_publication_preserves_a_preplaced_hardlink_temporary(tmp_path: Path) -> None:
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"do-not-truncate")
    final = tmp_path / "job-control-v1.json"
    temporary = tmp_path / ".job-control-v1.json.create-new.tmp"
    os.link(outside, temporary)

    with pytest.raises(
        reservation_protocol.ReservationConflictError,
        match="single-link",
    ):
        reservation_protocol.publish_write_ahead_intent(
            final,
            {"algorithm_version": "test-control-v1"},
        )

    assert outside.read_bytes() == b"do-not-truncate"
    assert temporary.exists()
    assert not final.exists()


def test_control_publication_adopts_only_the_exact_durable_temporary(tmp_path: Path) -> None:
    final = tmp_path / "job-control-v1.json"
    temporary = tmp_path / ".job-control-v1.json.create-new.tmp"
    payload = {"algorithm_version": "test-control-v1"}
    raw = reservation_protocol.canonical_json_bytes(payload)
    reservation_protocol._write_control_temporary(temporary, raw, event=None)

    reservation_protocol.publish_write_ahead_intent(final, payload)

    assert final.read_bytes() == raw
    assert not temporary.exists()


def test_control_publication_syncs_a_complete_temporary_before_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = tmp_path / "job-control-v1.json"
    temporary = tmp_path / ".job-control-v1.json.create-new.tmp"
    payload = {"algorithm_version": "test-control-v1"}
    raw = reservation_protocol.canonical_json_bytes(payload)
    temporary.write_bytes(raw)
    original_fsync = reservation_protocol.os.fsync
    original_replace = reservation_protocol.same_parent_replace_v1
    events: list[str] = []

    def fsync(descriptor: int) -> None:
        events.append("file-sync")
        original_fsync(descriptor)

    def replace(*args, **kwargs) -> None:
        assert "file-sync" in events
        events.append("rename")
        original_replace(*args, **kwargs)

    monkeypatch.setattr(reservation_protocol.os, "fsync", fsync)
    monkeypatch.setattr(reservation_protocol, "same_parent_replace_v1", replace)

    reservation_protocol.publish_write_ahead_intent(final, payload)

    assert events.index("file-sync") < events.index("rename")
    assert final.read_bytes() == raw


def test_control_publication_rejects_parent_replacement_after_temporary_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = tmp_path / ".depth-surge-reservations-v1"
    parked = tmp_path / "reservation-namespace-original"
    namespace.mkdir()
    final = namespace / "final-index.json"
    temporary = namespace / ".final-index.json.create-new.tmp"
    victim_payload = b"do-not-overwrite"
    swapped = False

    original_write = reservation_protocol._write_control_temporary

    def write_then_swap(
        path: Path,
        raw: bytes,
        *,
        event,
        expected_parent_identity=None,
    ) -> tuple[object, object]:
        nonlocal swapped
        result = original_write(
            path,
            raw,
            event=event,
            expected_parent_identity=expected_parent_identity,
        )
        swapped = True
        namespace.rename(parked)
        namespace.mkdir()
        temporary.write_bytes(victim_payload)
        return result

    monkeypatch.setattr(
        reservation_protocol,
        "_write_control_temporary",
        write_then_swap,
    )

    with pytest.raises(
        reservation_protocol.ReservationConflictError,
        match="parent identity changed",
    ):
        reservation_protocol.publish_write_ahead_intent(
            final,
            {"algorithm_version": "test-control-v1"},
        )

    assert swapped
    assert temporary.read_bytes() == victim_payload
    assert not final.exists()
    assert (parked / temporary.name).is_file()


def test_posix_replace_fault_boundary_is_between_rename_and_parent_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path("parent/source.tmp")
    target = Path("parent/target.json")
    source_identity = object()
    target_identity = object()
    parent_identity = object()
    identities = iter((target_identity, source_identity))
    descriptors = iter((41, 42))
    events: list[str] = []

    monkeypatch.setattr(
        reservation_protocol.os,
        "open",
        lambda *_args, **_kwargs: next(descriptors),
    )
    monkeypatch.setattr(reservation_protocol.os, "close", lambda _fd: None)
    monkeypatch.setattr(
        reservation_protocol.os,
        "replace",
        lambda *_args, **_kwargs: events.append("rename"),
    )
    monkeypatch.setattr(
        reservation_protocol.os,
        "fsync",
        lambda _fd: events.append("parent-sync"),
    )
    monkeypatch.setattr(
        reservation_protocol,
        "directory_identity_from_posix_fd",
        lambda _fd: parent_identity,
    )
    monkeypatch.setattr(
        reservation_protocol,
        "file_identity_from_posix_fd",
        lambda _fd: source_identity,
    )
    monkeypatch.setattr(
        reservation_protocol,
        "_identity",
        lambda _path: next(identities),
    )

    reservation_protocol._same_parent_replace_posix_v1(
        source,
        target,
        expected_source_identity=source_identity,
        expected_target_identity=target_identity,
        expected_parent_identity=parent_identity,
        replace_existing=True,
        after_namespace_replace=lambda: events.append("fault-boundary"),
    )

    assert events == ["rename", "fault-boundary", "parent-sync"]


def test_control_recovery_syncs_a_complete_temporary_before_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = tmp_path / "job-control-v1.json"
    temporary = tmp_path / ".job-control-v1.json.create-new.tmp"
    raw = reservation_protocol.canonical_json_bytes(
        {"algorithm_version": "test-control-v1"}
    )
    temporary.write_bytes(raw)
    original_fsync = reservation_protocol.os.fsync
    original_replace = reservation_protocol.same_parent_replace_v1
    events: list[str] = []

    def fsync(descriptor: int) -> None:
        events.append("file-sync")
        original_fsync(descriptor)

    def replace(*args, **kwargs) -> None:
        assert "file-sync" in events
        events.append("rename")
        original_replace(*args, **kwargs)

    monkeypatch.setattr(reservation_protocol.os, "fsync", fsync)
    monkeypatch.setattr(reservation_protocol, "same_parent_replace_v1", replace)

    assert reservation_protocol.recover_control_artifact_pair_v1(
        final,
        max_bytes=4096,
        parser=lambda value: value,
    ) == raw

    assert events.index("file-sync") < events.index("rename")
    assert final.read_bytes() == raw


def test_control_publication_never_overwrites_a_different_durable_temporary(
    tmp_path: Path,
) -> None:
    final = tmp_path / "job-control-v1.json"
    temporary = tmp_path / ".job-control-v1.json.create-new.tmp"
    original = reservation_protocol.canonical_json_bytes(
        {"algorithm_version": "first-control-v1"}
    )
    reservation_protocol._write_control_temporary(temporary, original, event=None)

    with pytest.raises(reservation_protocol.ReservationIncompleteError):
        reservation_protocol.publish_write_ahead_intent(
            final,
            {"algorithm_version": "second-control-v1"},
        )

    assert temporary.read_bytes() == original
    assert not final.exists()


def test_control_recovery_checks_preconditions_before_promoting_temporary(
    tmp_path: Path,
) -> None:
    final = tmp_path / "job-control-v1.json"
    temporary = tmp_path / ".job-control-v1.json.create-new.tmp"
    raw = reservation_protocol.canonical_json_bytes(
        {"algorithm_version": "test-control-v1"}
    )
    reservation_protocol._write_control_temporary(temporary, raw, event=None)

    def reject(_value: object) -> None:
        raise reservation_protocol.ReservationConflictError("foreign root")

    with pytest.raises(reservation_protocol.ReservationConflictError, match="foreign root"):
        reservation_protocol.recover_control_artifact_pair_v1(
            final,
            max_bytes=4096,
            parser=lambda value: value,
            prepublish_validate=reject,
        )

    assert temporary.read_bytes() == raw
    assert not final.exists()


def test_index_owned_extent_descriptor_rewrites_a_partial_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = (
        tmp_path
        / ".depth-surge-reservations-v1"
        / f"d-p-none-r-{'cd' * 16}-encoding_input_manifest-{'ef' * 16}.json"
    )
    temporary = descriptor.with_name(f".{descriptor.name}.create-new.tmp")
    original = reservation_protocol._write_control_temporary

    def interrupt_descriptor(
        path: Path,
        raw: bytes,
        *,
        event,
        expected_parent_identity=None,
    ):
        if path == temporary:
            with path.open("xb", buffering=0) as handle:
                prefix = raw[: max(1, min(37, len(raw) - 1))]
                assert handle.write(prefix) == len(prefix)
                handle.flush()
                os.fsync(handle.fileno())
            raise OSError("injected descriptor short write")
        return original(
            path,
            raw,
            event=event,
            expected_parent_identity=expected_parent_identity,
        )

    monkeypatch.setattr(
        reservation_protocol,
        "_write_control_temporary",
        interrupt_descriptor,
    )
    with pytest.raises(OSError, match="short write"):
        _create_transaction(tmp_path)
    prefix = temporary.read_bytes()
    monkeypatch.setattr(reservation_protocol, "_write_control_temporary", original)

    transaction = _create_transaction(tmp_path)

    assert transaction.descriptor_path.read_bytes().startswith(prefix)
    assert transaction.descriptor_path == descriptor
    assert descriptor.is_file()
    assert not temporary.exists()


def test_index_owned_descriptor_syncs_a_complete_temporary_before_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = tmp_path / ".depth-surge-reservations-v1" / "descriptor.json"
    descriptor.parent.mkdir(parents=True)
    temporary = descriptor.with_name(f".{descriptor.name}.create-new.tmp")
    payload = {"algorithm_version": "test-descriptor-v1"}
    raw = reservation_protocol.canonical_json_bytes(payload)
    temporary.write_bytes(raw)
    original_fsync = reservation_protocol.os.fsync
    original_replace = reservation_protocol.same_parent_replace_v1
    events: list[str] = []

    def fsync(file_descriptor: int) -> None:
        events.append("file-sync")
        original_fsync(file_descriptor)

    def replace(*args, **kwargs) -> None:
        assert "file-sync" in events
        events.append("rename")
        original_replace(*args, **kwargs)

    monkeypatch.setattr(reservation_protocol.os, "fsync", fsync)
    monkeypatch.setattr(reservation_protocol, "same_parent_replace_v1", replace)

    reservation_protocol.publish_index_owned_descriptor_v1(
        descriptor,
        payload,
        validate_authority=lambda: None,
    )

    assert events.index("file-sync") < events.index("rename")
    assert descriptor.read_bytes() == raw


def test_index_owned_descriptor_rejects_temporary_swap_before_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = tmp_path / ".depth-surge-reservations-v1" / "descriptor.json"
    descriptor.parent.mkdir(parents=True)
    temporary = descriptor.with_name(f".{descriptor.name}.create-new.tmp")
    payload = {"algorithm_version": "test-descriptor-v1"}
    raw = reservation_protocol.canonical_json_bytes(payload)
    descriptor.write_bytes(raw)
    temporary.write_bytes(raw)
    original_read = reservation_protocol._read_bounded_regular_file_with_identity
    swapped = False

    def read_then_swap(path: Path, *, max_bytes: int, **kwargs):
        nonlocal swapped
        result = original_read(path, max_bytes=max_bytes, **kwargs)
        if Path(path) == temporary and not swapped:
            swapped = True
            temporary.unlink()
            temporary.write_bytes(raw)
        return result

    monkeypatch.setattr(
        reservation_protocol,
        "_read_bounded_regular_file_with_identity",
        read_then_swap,
    )

    with pytest.raises(
        reservation_protocol.ReservationConflictError,
        match="temporary identity changed",
    ):
        reservation_protocol.publish_index_owned_descriptor_v1(
            descriptor,
            payload,
            validate_authority=lambda: None,
        )

    assert swapped
    assert descriptor.read_bytes() == raw
    assert temporary.read_bytes() == raw


def test_control_recovery_rejects_temporary_swap_before_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = tmp_path / "job-control-v1.json"
    temporary = tmp_path / ".job-control-v1.json.create-new.tmp"
    raw = reservation_protocol.canonical_json_bytes(
        {"algorithm_version": "test-control-v1"}
    )
    final.write_bytes(raw)
    temporary.write_bytes(b"durable-prefix")
    original_read = reservation_protocol._read_bounded_regular_file_with_identity
    swapped = False

    def read_then_swap(path: Path, *, max_bytes: int, **kwargs):
        nonlocal swapped
        result = original_read(path, max_bytes=max_bytes, **kwargs)
        if Path(path) == temporary and not swapped:
            swapped = True
            temporary.unlink()
            temporary.write_bytes(b"replacement")
        return result

    monkeypatch.setattr(
        reservation_protocol,
        "_read_bounded_regular_file_with_identity",
        read_then_swap,
    )

    with pytest.raises(
        reservation_protocol.ReservationConflictError,
        match="temporary identity changed",
    ):
        reservation_protocol.recover_control_artifact_pair_v1(
            final,
            max_bytes=4096,
            parser=lambda value: value,
        )

    assert swapped
    assert final.read_bytes() == raw
    assert temporary.read_bytes() == b"replacement"


def test_index_owned_extent_descriptor_preserves_a_malformed_nonprefix_temporary(
    tmp_path: Path,
) -> None:
    descriptor = (
        tmp_path
        / ".depth-surge-reservations-v1"
        / f"d-p-none-r-{'cd' * 16}-encoding_input_manifest-{'ef' * 16}.json"
    )
    descriptor.parent.mkdir(parents=True)
    temporary = descriptor.with_name(f".{descriptor.name}.create-new.tmp")
    foreign = b"not-a-descriptor-prefix"
    temporary.write_bytes(foreign)

    with pytest.raises(reservation_protocol.ReservationConflictError, match="prefix"):
        _create_transaction(tmp_path)

    assert temporary.read_bytes() == foreign
    assert not descriptor.exists()


def test_index_owned_extent_descriptor_preserves_a_complete_foreign_intent(
    tmp_path: Path,
) -> None:
    descriptor = (
        tmp_path
        / ".depth-surge-reservations-v1"
        / f"d-p-none-r-{'cd' * 16}-encoding_input_manifest-{'ef' * 16}.json"
    )
    descriptor.parent.mkdir(parents=True)
    temporary = descriptor.with_name(f".{descriptor.name}.create-new.tmp")
    foreign = reservation_protocol.canonical_json_bytes(
        {"algorithm_version": "foreign-descriptor-v1"}
    )
    reservation_protocol._write_control_temporary(temporary, foreign, event=None)

    with pytest.raises(reservation_protocol.ReservationConflictError, match="prefix"):
        _create_transaction(tmp_path)

    assert temporary.read_bytes() == foreign
    assert not descriptor.exists()


def test_streamed_prefix_consumption_never_requires_a_payload_bytes_object(tmp_path: Path) -> None:
    payload = b'{"large":"' + b"x" * 3000 + b'"}'

    def allocated(path: Path) -> int:
        size = path.stat().st_size
        return ((size + 4095) // 4096) * 4096

    transaction = _create_transaction(tmp_path, counter=allocated)
    with transaction.source_path.open("r+b", buffering=0) as handle:
        assert handle.write(payload) == len(payload)
        handle.flush()
        os.fsync(handle.fileno())

    transaction.consume_streamed_prefix(
        prefix_byte_count=len(payload),
        prefix_sha256=hashlib.sha256(payload).hexdigest(),
    )

    assert transaction.target_path.read_bytes() == payload
    assert not transaction.descriptor_path.exists()


def test_streamed_prefix_can_be_staged_from_bounded_chunks(tmp_path: Path) -> None:
    payload_parts = (b'{"large":"', b"x" * 3000, b'"}')
    payload_size = sum(len(part) for part in payload_parts)
    payload_sha256 = hashlib.sha256(b"".join(payload_parts)).hexdigest()

    def allocated(path: Path) -> int:
        size = path.stat().st_size
        return ((size + 4095) // 4096) * 4096

    transaction = _create_transaction(tmp_path, counter=allocated)
    transaction.stage_complete_full_streamed(
        chunks=lambda: iter(payload_parts),
        prefix_byte_count=payload_size,
        prefix_sha256=payload_sha256,
    )
    transaction.consume_streamed_prefix(
        prefix_byte_count=payload_size,
        prefix_sha256=payload_sha256,
    )

    assert transaction.target_path.stat().st_size == payload_size
    assert hashlib.sha256(transaction.target_path.read_bytes()).hexdigest() == payload_sha256


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


def test_extent_materialization_never_writes_through_a_swapped_target_parent(
    tmp_path: Path,
) -> None:
    index = tmp_path / ".depth-surge-reservations-v1/final-index.json"
    reservation_protocol.publish_write_ahead_intent(
        index,
        {"algorithm_version": "test-final-index-v1", "generation": "ab" * 16},
    )
    target_parent = tmp_path / "target"
    original_parent = tmp_path / "target-original"
    target_parent.mkdir()
    victim_payload = b"foreign target bytes"
    swapped = False

    def swap_after_parent_authentication(stage: str) -> None:
        nonlocal swapped
        if stage != "parent:authenticated" or swapped:
            return
        swapped = True
        target_parent.rename(original_parent)
        target_parent.mkdir()
        (target_parent / "manifest.json").write_bytes(victim_payload)

    with pytest.raises(
        reservation_protocol.ReservationConflictError,
        match="parent identity changed",
    ):
        reservation_protocol.ReservedArtifactTransaction.create(
            job_root=tmp_path,
            index_relative_path=index.relative_to(tmp_path),
            target_relative_path=Path("target/manifest.json"),
            reservation_kind="encoding_input_manifest",
            payload_raw_max=100,
            allocation_unit=4096,
            allocation_adapter=_adapter,
            generation="cd" * 16,
            reservation_id="ef" * 16,
            expected_publication_method="replace_placeholder",
            event=swap_after_parent_authentication,
        )

    assert swapped is True
    assert (target_parent / "manifest.json").read_bytes() == victim_payload
    assert list(original_parent.iterdir()) == []
    assert not any(path.name.endswith(".reserved") for path in target_parent.iterdir())


def test_later_extent_write_never_uses_a_replaced_target_parent(
    tmp_path: Path,
) -> None:
    intent_path = tmp_path / ".depth-surge-reservations-v1/final-index.json"
    reservation_protocol.publish_write_ahead_intent(
        intent_path,
        {"algorithm_version": "test-final-index-v1", "generation": "ab" * 16},
    )
    target_parent = tmp_path / "manifests"
    parked_parent = tmp_path / "manifests-original"
    target_parent.mkdir()
    transaction = reservation_protocol.ReservedArtifactTransaction.create(
        job_root=tmp_path,
        index_relative_path=Path(".depth-surge-reservations-v1/final-index.json"),
        target_relative_path=Path("manifests/encoding_input_manifest.json"),
        reservation_kind="encoding_input_manifest",
        payload_raw_max=4217,
        allocation_unit=4096,
        allocation_adapter=_adapter,
        generation="cd" * 16,
        reservation_id="ef" * 16,
    )
    original_zero_bytes = transaction.source_path.read_bytes()
    source_name = transaction.source_path.name

    target_parent.rename(parked_parent)
    target_parent.mkdir()
    victim = target_parent / source_name
    victim.write_bytes(b"do-not-overwrite")

    with pytest.raises(
        reservation_protocol.ReservationConflictError,
        match="parent identity changed",
    ):
        transaction.stage_complete_full(b'{"schema_version":1}')

    assert victim.read_bytes() == b"do-not-overwrite"
    assert (parked_parent / source_name).read_bytes() == original_zero_bytes


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


@pytest.mark.skipif(os.name != "nt", reason="Windows replacement gaps are platform-specific")
def test_windows_replacement_gap_recovers_short_source_to_missing_target(
    tmp_path: Path,
) -> None:
    transaction = _create_transaction(tmp_path)
    payload = b'{"schema_version":1,"value":"replacement-gap"}'

    with pytest.raises(reservation_protocol.InjectedReservationCrash):
        transaction.consume(payload, crash_at="after-truncate-sync")
    transaction._short_release_recovery(payload)
    transaction.target_path.unlink()
    loaded = reservation_protocol.ReservedArtifactTransaction.load(
        transaction.descriptor_path,
        allocation_adapter=lambda *_args: (_ for _ in ()).throw(
            AssertionError("replacement-gap recovery must not call readiness adapter")
        ),
    )

    assert loaded.validate_physical_phase(payload) == "replacement-gap"
    assert loaded.recover(payload) == "committed-target"
    assert loaded.target_path.read_bytes() == payload
    assert not loaded.source_path.exists()
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
    artifact = _artifact()
    metadata = artifact.to_object()["metadata"]
    generation = metadata["job_reservation_generation"]
    initial_id = "04" * 16
    terminal_id = metadata["job_terminal_reservation_id"]
    return reservation_protocol.JobControlV1.create(
        job_root=job_root,
        job_id=metadata["job_id"],
        settings_identity_fingerprint=metadata["settings_identity_fingerprint"],
        job_reservation_generation=generation,
        initial_settings_reservation_id=initial_id,
        job_terminal_reservation_id=terminal_id,
        legacy_settings_artifact_relative_path=None,
        bootstrap_extent_entries=[
            reservation_protocol._bootstrap_entry(
                job_root=job_root,
                reservation_kind="initial_settings",
                generation=generation,
                reservation_id=initial_id,
                publication_method="replace_placeholder",
                allocation_unit=4096,
            ),
            reservation_protocol._bootstrap_entry(
                job_root=job_root,
                reservation_kind="job_terminal_settings",
                generation=generation,
                reservation_id=terminal_id,
                publication_method="replace_existing",
                allocation_unit=4096,
            ),
        ],
        bootstrap_settings_intent=reservation_protocol._bootstrap_settings_intent(
            artifact,
            reservation_generation=generation,
        ),
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


def test_job_control_temporary_from_another_root_is_never_promoted(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    control = _job_control(source_root)
    temporary = target_root / ".job-control-v1.json.create-new.tmp"
    reservation_protocol._write_control_temporary(
        temporary,
        reservation_protocol.canonical_json_bytes(control.to_object()),
        event=None,
    )

    with pytest.raises(
        reservation_protocol.ReservationConflictError,
        match="different job root",
    ):
        reservation_protocol.load_job_control(target_root)

    assert temporary.is_file()
    assert not (target_root / "job-control-v1.json").exists()


def test_job_control_candidate_preserves_an_unindexed_nested_prune_marker(
    tmp_path: Path,
) -> None:
    control = _job_control(tmp_path)
    temporary = tmp_path / ".job-control-v1.json.create-new.tmp"
    reservation_protocol._write_control_temporary(
        temporary,
        reservation_protocol.canonical_json_bytes(control.to_object()),
        event=None,
    )
    parent = tmp_path / "00_original_frames"
    parent.mkdir()
    marker = parent / ".depth-surge-prune-v2-foreign.json"
    marker.write_bytes(b"foreign-marker")

    with pytest.raises(
        reservation_protocol.ReservationConflictError,
        match="unindexed reserved payload",
    ):
        reservation_protocol.load_job_control(tmp_path)

    assert marker.read_bytes() == b"foreign-marker"
    assert temporary.is_file()
    assert not (tmp_path / "job-control-v1.json").exists()


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
