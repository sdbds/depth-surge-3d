"""Irreversible final-video publication and three-way media audit."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import os
import re
import stat
import secrets
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Literal

from ...rendering.quality.cleanup_transaction import (
    plan_prune_markers_v1,
    prune_entry_from_reservation_v1,
)
from ...rendering.quality.diagnostic_transaction import self_fingerprint

from ...io.canonical_json import (
    CanonicalJsonError,
    canonical_json_bytes,
    strict_canonical_json_loads,
)
from ...io.settings_artifact import (
    ProcessingSettingsArtifactV5,
    SettingsArtifactSchemaError,
    validate_contained_legacy_final_component_v1,
    validate_portable_final_component_v1,
)
from ...io.reservation_protocol import (
    AllocationAdapter,
    AllocationCounter,
    FinalPublicationRecovery,
    FinalPublicationObservation,
    InjectedReservationCrash,
    ReservationConflictError,
    DirectoryIdentity,
    ReservationFileIdentity,
    ReservationDescriptor,
    ReservedArtifactTransaction,
    allocation_unit_for_path,
    platform_allocation_adapter,
    platform_allocation_counter,
    payload_logical_bytes,
    publish_write_ahead_intent,
    recover_control_artifact_pair_v1,
    same_parent_replace_v1,
    sha256_canonical,
    validate_descriptor_against_entry_v1,
    validate_reservation_entry_v1,
    load_job_control,
    validate_final_encoding_entry_order,
    assert_only_job_control_owned_reservations_v1,
    assert_index_owned_reservation_namespace_v1,
    classify_final_publication_observation,
    _directory_identity,
    _identity,
    _read_bounded_regular_file,
    _unlink_safe_control_temporary,
)


FINAL_VIDEO_MANIFEST_RELATIVE_PATH = Path("final_video_manifest.json")
ENCODING_INPUT_MANIFEST_RELATIVE_PATH = Path("encoding_input_manifest.json")
FINAL_ENCODING_RESERVATION_RELATIVE_PATH = Path(
    ".depth-surge-reservations-v1/final-encoding-reservation-v1.json"
)
_HEX64 = re.compile(r"[0-9a-f]{64}", flags=re.ASCII)
_IMAGE_TOKEN = re.compile(
    r"@image-sequence:sha256=([0-9a-f]{64}):view=(left|right|frames)",
    flags=re.ASCII,
)
_AUDIO_TOKEN = re.compile(
    r"@audio:sha256=([0-9a-f]{64}):bytes=(0|[1-9][0-9]*):stream=a:0",
    flags=re.ASCII,
)
_OUTPUT_TOKEN = re.compile(r"@output:path64=([A-Za-z0-9_-]+)", flags=re.ASCII)


class FinalPublicationError(RuntimeError):
    """The final publication cannot safely advance."""


class FinalPublicationValidationError(FinalPublicationError):
    """A sibling, manifest, or three-way binding is invalid."""


class InjectedFinalPublicationCrash(RuntimeError):
    """A fault fixture stopped after a durable publication boundary."""


def _reservation_identity(path: Path) -> ReservationFileIdentity:
    return _identity(Path(path))


def _reservation_directory_identity(path: Path) -> DirectoryIdentity:
    return _directory_identity(Path(path))


@dataclass(frozen=True)
class FinalEncodingReservationIndexV1:
    job_id: str
    settings_identity_fingerprint: str
    settings_raw_sha256_at_preflight: str
    publication_generation: str
    reservation_generation: str
    settings_revision_at_preflight: int
    final_video_relative_path: str
    final_video_publication_method: Literal["replace_placeholder", "replace_existing"]
    final_video_existing_target_identity_at_preflight: ReservationFileIdentity | None
    final_video_target_descriptor_relative_path: str
    sibling_video_temporary_relative_path: str
    sibling_video_parent_identity: DirectoryIdentity
    diagnostics_metadata_raw_sha256_at_preflight: str | None
    diagnostics_metadata_fingerprint_at_preflight: str | None
    extent_entries: tuple[dict, ...]
    fingerprint: str

    def _base_object(self) -> dict:
        return {
            "schema_version": 1,
            "algorithm_version": "final-encoding-reservation-v1",
            "job_id": self.job_id,
            "settings_artifact_relative_path": "processing-settings-v5.json",
            "settings_identity_fingerprint": self.settings_identity_fingerprint,
            "settings_raw_sha256_at_preflight": self.settings_raw_sha256_at_preflight,
            "publication_generation": self.publication_generation,
            "reservation_generation": self.reservation_generation,
            "settings_revision_at_preflight": self.settings_revision_at_preflight,
            "final_video_relative_path": self.final_video_relative_path,
            "final_video_publication_method": self.final_video_publication_method,
            "final_video_existing_target_identity_at_preflight": (
                self.final_video_existing_target_identity_at_preflight.to_object()
                if self.final_video_existing_target_identity_at_preflight is not None
                else None
            ),
            "final_video_target_descriptor_relative_path": (
                self.final_video_target_descriptor_relative_path
            ),
            "sibling_video_temporary_relative_path": (self.sibling_video_temporary_relative_path),
            "sibling_video_parent_identity": self.sibling_video_parent_identity.to_object(),
            "diagnostics_metadata_raw_sha256_at_preflight": (
                self.diagnostics_metadata_raw_sha256_at_preflight
            ),
            "diagnostics_metadata_fingerprint_at_preflight": (
                self.diagnostics_metadata_fingerprint_at_preflight
            ),
            "extent_entries": list(self.extent_entries),
        }

    def to_object(self) -> dict:
        return {**self._base_object(), "fingerprint": self.fingerprint}

    def _validate_contract(self) -> None:
        for value, digits, label in (
            (self.job_id, 32, "final index job id"),
            (self.settings_identity_fingerprint, 64, "final index settings identity"),
            (self.settings_raw_sha256_at_preflight, 64, "final index settings hash"),
            (self.publication_generation, 32, "final index publication generation"),
            (self.reservation_generation, 32, "final index reservation generation"),
            (self.fingerprint, 64, "final index fingerprint"),
        ):
            if not isinstance(value, str) or re.fullmatch(
                rf"[0-9a-f]{{{digits}}}", value, flags=re.ASCII
            ) is None:
                raise ReservationConflictError(f"{label} is invalid")
        if (
            type(self.settings_revision_at_preflight) is not int
            or not 0 <= self.settings_revision_at_preflight <= 2**64 - 1
        ):
            raise ReservationConflictError("final index settings revision is invalid")
        validate_portable_final_component_v1(self.final_video_relative_path)
        if self.sibling_video_temporary_relative_path != (
            f".depth-surge-final-v4-{self.publication_generation}.tmp.mp4"
        ):
            raise ReservationConflictError("final index sibling name is not canonical")
        expected_target_descriptor = (
            Path(".depth-surge-reservations-v1")
            / (
                f"final-video-target-p-{self.publication_generation}-"
                f"r-{self.reservation_generation}.json"
            )
        ).as_posix()
        if self.final_video_target_descriptor_relative_path != expected_target_descriptor:
            raise ReservationConflictError("final target descriptor name is not canonical")
        if self.final_video_publication_method == "replace_placeholder":
            if self.final_video_existing_target_identity_at_preflight is not None:
                raise ReservationConflictError("placeholder publication has an old target")
        elif self.final_video_publication_method == "replace_existing":
            if self.final_video_existing_target_identity_at_preflight is None:
                raise ReservationConflictError("replacement publication lost its old target")
        else:
            raise ReservationConflictError("final video publication method is invalid")
        diagnostics = (
            self.diagnostics_metadata_raw_sha256_at_preflight,
            self.diagnostics_metadata_fingerprint_at_preflight,
        )
        if (diagnostics[0] is None) != (diagnostics[1] is None):
            raise ReservationConflictError("final index diagnostics binding is incomplete")
        for value in diagnostics:
            if value is not None and _HEX64.fullmatch(value) is None:
                raise ReservationConflictError("final index diagnostics hash is invalid")
        if not 1 <= len(self.extent_entries) <= 64:
            raise ReservationConflictError("final index extent count is invalid")
        entries = tuple(validate_reservation_entry_v1(entry) for entry in self.extent_entries)
        validate_final_encoding_entry_order(
            [entry["reservation_kind"] for entry in entries]
        )
        seen_reservations: set[tuple[str, str]] = set()
        seen_descriptors: set[str] = set()
        seen_sources: set[str] = set()
        for entry in entries[:-1]:
            if (
                entry["descriptor_publication_generation"] != self.publication_generation
                or entry["reservation_generation"] != self.reservation_generation
            ):
                raise ReservationConflictError(
                    "final index invocation entry differs from its generation"
                )
        terminal = entries[-1]
        if terminal["descriptor_publication_generation"] is not None:
            raise ReservationConflictError("final index terminal entry has a publication")
        for entry in entries:
            reservation_key = (
                entry["reservation_generation"],
                entry["reservation_id"],
            )
            if reservation_key in seen_reservations:
                raise ReservationConflictError("final index repeats a reservation identity")
            seen_reservations.add(reservation_key)
            if entry["descriptor_relative_path"] in seen_descriptors:
                raise ReservationConflictError("final index repeats a descriptor path")
            seen_descriptors.add(entry["descriptor_relative_path"])
            if entry["source_extent_relative_path"] in seen_sources:
                raise ReservationConflictError("final index repeats a source path")
            seen_sources.add(entry["source_extent_relative_path"])

    @classmethod
    def build(cls, **values) -> "FinalEncodingReservationIndexV1":
        values = {
            **values,
            "extent_entries": tuple(
                validate_reservation_entry_v1(entry) for entry in values["extent_entries"]
            ),
        }
        provisional = cls(**values, fingerprint="")
        result = cls(
            **{
                **provisional.__dict__,
                "fingerprint": sha256_canonical(provisional._base_object()),
            }
        )
        result._validate_contract()
        return result

    @classmethod
    def from_object(cls, value: object) -> "FinalEncodingReservationIndexV1":
        expected = set(cls.__dataclass_fields__) - {"fingerprint"}
        expected.update(
            {
                "schema_version",
                "algorithm_version",
                "settings_artifact_relative_path",
                "fingerprint",
            }
        )
        if not isinstance(value, dict) or set(value) != expected:
            raise ReservationConflictError("final encoding index schema is not closed")
        if (
            value["schema_version"] != 1
            or value["algorithm_version"] != "final-encoding-reservation-v1"
            or value["settings_artifact_relative_path"] != "processing-settings-v5.json"
            or not isinstance(value["extent_entries"], list)
        ):
            raise ReservationConflictError("final encoding index version is invalid")
        existing = value["final_video_existing_target_identity_at_preflight"]
        index = cls(
            job_id=value["job_id"],
            settings_identity_fingerprint=value["settings_identity_fingerprint"],
            settings_raw_sha256_at_preflight=value["settings_raw_sha256_at_preflight"],
            publication_generation=value["publication_generation"],
            reservation_generation=value["reservation_generation"],
            settings_revision_at_preflight=value["settings_revision_at_preflight"],
            final_video_relative_path=value["final_video_relative_path"],
            final_video_publication_method=value["final_video_publication_method"],
            final_video_existing_target_identity_at_preflight=(
                ReservationFileIdentity.from_object(existing) if existing is not None else None
            ),
            final_video_target_descriptor_relative_path=value[
                "final_video_target_descriptor_relative_path"
            ],
            sibling_video_temporary_relative_path=value["sibling_video_temporary_relative_path"],
            sibling_video_parent_identity=DirectoryIdentity.from_object(
                value["sibling_video_parent_identity"]
            ),
            diagnostics_metadata_raw_sha256_at_preflight=value[
                "diagnostics_metadata_raw_sha256_at_preflight"
            ],
            diagnostics_metadata_fingerprint_at_preflight=value[
                "diagnostics_metadata_fingerprint_at_preflight"
            ],
            extent_entries=tuple(
                validate_reservation_entry_v1(entry) for entry in value["extent_entries"]
            ),
            fingerprint=value["fingerprint"],
        )
        if index.fingerprint != sha256_canonical(index._base_object()):
            raise ReservationConflictError("final encoding index fingerprint differs")
        index._validate_contract()
        return index


@dataclass(frozen=True)
class FinalVideoTargetReservationV1:
    job_id: str
    settings_identity_fingerprint: str
    settings_raw_sha256_at_preflight: str
    settings_revision_at_preflight: int
    publication_generation: str
    reservation_generation: str
    transaction_index_fingerprint: str
    final_video_relative_path: str
    sibling_video_temporary_relative_path: str
    target_parent_identity: DirectoryIdentity
    publication_method: Literal["replace_placeholder", "replace_existing"]
    reserved_target_identity: ReservationFileIdentity
    fingerprint: str

    def _base_object(self) -> dict:
        return {
            "schema_version": 1,
            "algorithm_version": "final-video-target-reservation-v1",
            "job_id": self.job_id,
            "settings_artifact_relative_path": "processing-settings-v5.json",
            "settings_identity_fingerprint": self.settings_identity_fingerprint,
            "settings_raw_sha256_at_preflight": self.settings_raw_sha256_at_preflight,
            "settings_revision_at_preflight": self.settings_revision_at_preflight,
            "publication_generation": self.publication_generation,
            "reservation_generation": self.reservation_generation,
            "transaction_index_relative_path": FINAL_ENCODING_RESERVATION_RELATIVE_PATH.as_posix(),
            "transaction_index_fingerprint": self.transaction_index_fingerprint,
            "final_video_relative_path": self.final_video_relative_path,
            "sibling_video_temporary_relative_path": self.sibling_video_temporary_relative_path,
            "target_parent_identity": self.target_parent_identity.to_object(),
            "publication_method": self.publication_method,
            "reserved_target_identity": self.reserved_target_identity.to_object(),
        }

    def to_object(self) -> dict:
        return {**self._base_object(), "fingerprint": self.fingerprint}

    def _validate_contract(self) -> None:
        for value, digits, label in (
            (self.job_id, 32, "final target job id"),
            (self.settings_identity_fingerprint, 64, "final target settings identity"),
            (self.settings_raw_sha256_at_preflight, 64, "final target settings hash"),
            (self.publication_generation, 32, "final target publication generation"),
            (self.reservation_generation, 32, "final target reservation generation"),
            (self.transaction_index_fingerprint, 64, "final target index fingerprint"),
            (self.fingerprint, 64, "final target fingerprint"),
        ):
            if not isinstance(value, str) or re.fullmatch(
                rf"[0-9a-f]{{{digits}}}", value, flags=re.ASCII
            ) is None:
                raise ReservationConflictError(f"{label} is invalid")
        if (
            type(self.settings_revision_at_preflight) is not int
            or not 0 <= self.settings_revision_at_preflight <= 2**64 - 1
        ):
            raise ReservationConflictError("final target settings revision is invalid")
        validate_portable_final_component_v1(self.final_video_relative_path)
        if self.sibling_video_temporary_relative_path != (
            f".depth-surge-final-v4-{self.publication_generation}.tmp.mp4"
        ):
            raise ReservationConflictError("final target sibling name is not canonical")
        if self.publication_method not in {"replace_placeholder", "replace_existing"}:
            raise ReservationConflictError("final target publication method is invalid")
        if self.reserved_target_identity.link_count != 1:
            raise ReservationConflictError("final target identity is not single-link")
        if self.reserved_target_identity.volume_key != self.target_parent_identity.volume_key:
            raise ReservationConflictError("final target identity is on another volume")

    @classmethod
    def build(cls, **values) -> "FinalVideoTargetReservationV1":
        provisional = cls(**values, fingerprint="")
        result = cls(
            **{
                **provisional.__dict__,
                "fingerprint": sha256_canonical(provisional._base_object()),
            }
        )
        result._validate_contract()
        return result

    @classmethod
    def from_object(cls, value: object) -> "FinalVideoTargetReservationV1":
        expected = set(cls.__dataclass_fields__) - {"fingerprint"}
        expected.update(
            {
                "schema_version",
                "algorithm_version",
                "settings_artifact_relative_path",
                "transaction_index_relative_path",
                "fingerprint",
            }
        )
        if not isinstance(value, dict) or set(value) != expected:
            raise ReservationConflictError("final target descriptor schema is not closed")
        if (
            value["schema_version"] != 1
            or value["algorithm_version"] != "final-video-target-reservation-v1"
            or value["settings_artifact_relative_path"] != "processing-settings-v5.json"
            or value["transaction_index_relative_path"]
            != FINAL_ENCODING_RESERVATION_RELATIVE_PATH.as_posix()
        ):
            raise ReservationConflictError("final target descriptor version is invalid")
        descriptor = cls(
            job_id=value["job_id"],
            settings_identity_fingerprint=value["settings_identity_fingerprint"],
            settings_raw_sha256_at_preflight=value["settings_raw_sha256_at_preflight"],
            settings_revision_at_preflight=value["settings_revision_at_preflight"],
            publication_generation=value["publication_generation"],
            reservation_generation=value["reservation_generation"],
            transaction_index_fingerprint=value["transaction_index_fingerprint"],
            final_video_relative_path=value["final_video_relative_path"],
            sibling_video_temporary_relative_path=value["sibling_video_temporary_relative_path"],
            target_parent_identity=DirectoryIdentity.from_object(
                value["target_parent_identity"]
            ),
            publication_method=value["publication_method"],
            reserved_target_identity=ReservationFileIdentity.from_object(
                value["reserved_target_identity"]
            ),
            fingerprint=value["fingerprint"],
        )
        if descriptor.fingerprint != sha256_canonical(descriptor._base_object()):
            raise ReservationConflictError("final target descriptor fingerprint differs")
        descriptor._validate_contract()
        return descriptor


def _validate_committed_publication_triad_v1(
    root: Path,
    index: FinalEncodingReservationIndexV1,
) -> None:
    """Bind descriptorless committed manifests to each other and the current video."""

    input_path = root / ENCODING_INPUT_MANIFEST_RELATIVE_PATH
    final_manifest_path = root / FINAL_VIDEO_MANIFEST_RELATIVE_PATH
    final_video_path = root / index.final_video_relative_path
    if not all(
        _regular_nonlink(path)
        for path in (input_path, final_manifest_path, final_video_path)
    ):
        raise ReservationConflictError("committed publication artifact is missing or unsafe")
    try:
        input_fingerprint, _input_raw_sha256, _input_byte_count = (
            _stream_input_manifest_identity(input_path)
        )
        manifest = parse_final_video_manifest_v4(
            _read_bounded_regular_file(final_manifest_path, max_bytes=1024 * 1024)
        )
        video_sha256, video_byte_count = _hash_file(final_video_path)
    except (FinalPublicationValidationError, OSError) as error:
        raise ReservationConflictError(
            "committed publication evidence is malformed"
        ) from error
    if (
        manifest.relative_path != index.final_video_relative_path
        or manifest.input_stage_fingerprint != input_fingerprint
        or manifest.sha256 != video_sha256
        or manifest.byte_count != video_byte_count
    ):
        raise ReservationConflictError(
            "committed video and manifest evidence are not three-way bound"
        )


@dataclass
class PreparedFinalEncodingReservationV1:
    job_root: Path
    index: FinalEncodingReservationIndexV1
    target_descriptor: FinalVideoTargetReservationV1
    input_manifest_transaction: ReservedArtifactTransaction | None
    final_manifest_transaction: ReservedArtifactTransaction | None
    event: Callable[[str], None] | None = None
    producer_settings_transaction: ReservedArtifactTransaction | None = None
    producer_settings_payload: bytes | None = None
    prune_marker_transactions: tuple[
        tuple[dict, ReservedArtifactTransaction | None, bytes], ...
    ] = ()
    diagnostics_payload_pruned_transaction: ReservedArtifactTransaction | None = None
    cleanup_pending_settings_transaction: ReservedArtifactTransaction | None = None
    cleanup_pending_settings_payload: bytes | None = None

    @property
    def index_path(self) -> Path:
        return self.job_root / FINAL_ENCODING_RESERVATION_RELATIVE_PATH

    @property
    def target_descriptor_path(self) -> Path:
        return self.job_root / self.index.final_video_target_descriptor_relative_path

    @property
    def sibling_path(self) -> Path:
        return self.job_root / self.index.sibling_video_temporary_relative_path

    @property
    def final_video_path(self) -> Path:
        return self.job_root / self.index.final_video_relative_path

    @property
    def has_post_publication_work(self) -> bool:
        return bool(
            self.prune_marker_transactions
            or self.diagnostics_payload_pruned_transaction is not None
            or self.cleanup_pending_settings_transaction is not None
        )

    def validate_reserved_target(self) -> None:
        if (
            _reservation_identity(self.final_video_path)
            != self.target_descriptor.reserved_target_identity
        ):
            raise ReservationConflictError("final video target no longer has its reserved identity")

    def _consume_reserved_payload(
        self,
        transaction: ReservedArtifactTransaction,
        payload: bytes,
    ) -> None:
        phase = transaction.validate_physical_phase(payload)
        if phase in {"full-source", "short-source"}:
            self.validate_indexed_predecessor(transaction)
        if phase == "full-source":
            transaction.consume(payload)
        else:
            transaction.recover(payload)

    def validate_indexed_predecessor(
        self,
        transaction: ReservedArtifactTransaction,
    ) -> None:
        if not transaction.source_path.exists() or not transaction.target_path.exists():
            return
        matches = [
            entry
            for entry in self.index.extent_entries
            if entry.get("reservation_kind") == transaction.descriptor.reservation_kind
            and entry.get("reservation_id") == transaction.descriptor.reservation_id
            and entry.get("reservation_generation") == transaction.descriptor.generation
        ]
        if len(matches) != 1:
            raise ReservationConflictError("extent descriptor has no unique indexed plan")
        entry = matches[0]
        if transaction.descriptor.publication_method != "replace_existing":
            return
        if transaction.descriptor.reservation_kind in {
            "producer_settings",
            "cleanup_pending_settings",
            "job_terminal_settings",
        }:
            return
        expected = entry.get("existing_target_identity_at_preflight")
        expected_count = entry.get("existing_target_byte_count_at_preflight")
        expected_hash = entry.get("existing_target_raw_sha256_at_preflight")
        target = transaction.target_path
        if (
            expected is None
            or type(expected_count) is not int
            or not isinstance(expected_hash, str)
            or _reservation_identity(target) != ReservationFileIdentity.from_object(expected)
            or _hash_file(target) != (expected_hash, expected_count)
        ):
            raise ReservationConflictError(
                f"{transaction.descriptor.reservation_kind} preflight target changed"
            )

    @staticmethod
    def _reset_uncommitted_settings_source(
        transaction: ReservedArtifactTransaction,
        expected_payload: bytes,
    ) -> None:
        if transaction.validate_physical_phase(expected_payload) != "full-source":
            raise ReservationConflictError("only an unconsumed full settings source may be reset")
        zeroes = b"\x00" * (64 * 1024)
        with transaction.source_path.open("r+b", buffering=0) as handle:
            handle.seek(0)
            remaining = transaction.descriptor.logical_byte_count
            while remaining:
                chunk = zeroes[: min(len(zeroes), remaining)]
                if handle.write(chunk) != len(chunk):
                    raise OSError("short settings reservation reset")
                remaining -= len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if transaction.validate_physical_phase(expected_payload) != "full-source":
            raise ReservationConflictError("settings reservation reset changed physical phase")

    def commit_producer_settings(self) -> ProcessingSettingsArtifactV5:
        payload = self.producer_settings_payload
        transaction = self.producer_settings_transaction
        if transaction is not None:
            current = ProcessingSettingsArtifactV5.parse(
                (self.job_root / "processing-settings-v5.json").read_bytes()
            )
            if payload is None:
                payload = current.with_producer_marker(
                    generation=self.index.publication_generation,
                    updated_at=_utc_now_usec(),
                ).encode()
                self.producer_settings_payload = payload
                self._reset_uncommitted_settings_source(transaction, payload)
            intended = ProcessingSettingsArtifactV5.parse(payload)
            timestamp = intended.to_object()["metadata"]["last_updated_at"]
            if intended != current.with_producer_marker(
                generation=self.index.publication_generation,
                updated_at=timestamp,
            ):
                raise ReservationConflictError(
                    "producer settings intent is not its exact successor"
                )
            self._consume_reserved_payload(transaction, payload)
            self.producer_settings_transaction = None
        current = ProcessingSettingsArtifactV5.parse(
            (self.job_root / "processing-settings-v5.json").read_bytes()
        )
        metadata = current.to_object()["metadata"]
        if (
            metadata["final_media_producer_contract_version"] != 4
            or metadata["final_media_publication_generation"] != self.index.publication_generation
        ):
            raise ReservationConflictError("producer settings transition did not commit")
        return current

    def commit_prune_markers(self) -> tuple[dict, ...]:
        committed: list[tuple[dict, ReservedArtifactTransaction | None, bytes]] = []
        entries: list[dict] = []
        for entry, transaction, payload in self.prune_marker_transactions:
            if transaction is not None:
                self._consume_reserved_payload(transaction, payload)
            target = self.job_root / entry["relative_path"] / entry["marker"]["name"]
            if not _regular_nonlink(target) or _hash_file(target) != (
                entry["marker"]["sha256"],
                len(payload),
            ):
                raise ReservationConflictError("indexed prune marker did not commit")
            committed.append((entry, None, payload))
            entries.append(entry)
        self.prune_marker_transactions = tuple(committed)
        return tuple(entries)

    def commit_payload_pruned_metadata(self, payload: bytes) -> None:
        transaction = self.diagnostics_payload_pruned_transaction
        if transaction is None:
            target = self.job_root / "04_stereo_diagnostics" / "metadata.json"
            if target.read_bytes() != payload:
                raise ReservationConflictError(
                    "payload-pruned metadata differs from committed target"
                )
            return
        self._consume_reserved_payload(transaction, payload)
        self.diagnostics_payload_pruned_transaction = None

    def commit_cleanup_pending_settings(self) -> ProcessingSettingsArtifactV5:
        payload = self.cleanup_pending_settings_payload
        transaction = self.cleanup_pending_settings_transaction
        if transaction is not None:
            current = ProcessingSettingsArtifactV5.parse(
                (self.job_root / "processing-settings-v5.json").read_bytes()
            )
            if payload is None:
                payload = current.cleanup_pending(updated_at=_utc_now_usec()).encode()
                self.cleanup_pending_settings_payload = payload
                self._reset_uncommitted_settings_source(transaction, payload)
            intended = ProcessingSettingsArtifactV5.parse(payload)
            timestamp = intended.to_object()["metadata"]["last_updated_at"]
            if intended != current.cleanup_pending(updated_at=timestamp):
                raise ReservationConflictError(
                    "cleanup-pending settings intent is not its exact successor"
                )
            self._consume_reserved_payload(transaction, payload)
            self.cleanup_pending_settings_transaction = None
        current = ProcessingSettingsArtifactV5.parse(
            (self.job_root / "processing-settings-v5.json").read_bytes()
        )
        if current.to_object()["metadata"]["cleanup_status"] != "pending":
            raise ReservationConflictError("cleanup-pending settings transition did not commit")
        return current

    @staticmethod
    def _canonical_prefix_byte_count(path: Path) -> int | None:
        offset = 0
        first_zero: int | None = None
        with path.open("rb", buffering=0) as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                if first_zero is None:
                    position = chunk.find(b"\x00")
                    if position < 0:
                        offset += len(chunk)
                        continue
                    first_zero = offset + position
                    if any(chunk[position:]):
                        return None
                elif any(chunk):
                    return None
                offset += len(chunk)
        return offset if first_zero is None else first_zero

    def _video_state(self) -> str:
        final = self.final_video_path
        sibling_present = self.sibling_path.exists() or self.sibling_path.is_symlink()
        if final.exists() or final.is_symlink():
            identity = _reservation_identity(final)
            if identity == self.target_descriptor.reserved_target_identity:
                return "reserved-target"
            return "committed-new"
        if sibling_present and _regular_nonlink(self.sibling_path):
            return "windows-replacement-gap"
        raise ReservationConflictError("final video is outside its indexed namespace states")

    def _input_manifest_state(self, video_state: str) -> str:
        transaction = self.input_manifest_transaction
        target = self.job_root / ENCODING_INPUT_MANIFEST_RELATIVE_PATH
        if transaction is None:
            if not _regular_nonlink(target):
                raise ReservationConflictError("committed input manifest is missing")
            _stream_input_manifest_identity(target)
            return "committed-new"
        source = transaction.source_path
        if not source.exists():
            if _regular_nonlink(target):
                _stream_input_manifest_identity(target)
                return "committed-new"
            raise ReservationConflictError("input extent has no authorized location")
        prefix = self._canonical_prefix_byte_count(source)
        if prefix is None:
            return "partial"
        if prefix == 0:
            return "zero"
        try:
            fingerprint, digest, count = _stream_input_manifest_identity(
                source,
                raw_byte_count=prefix,
                logical_byte_count=(
                    transaction.descriptor.logical_byte_count
                    if source.stat().st_size == transaction.descriptor.logical_byte_count
                    else None
                ),
            )
            del fingerprint
            transaction.validate_streamed_physical_phase(
                prefix_byte_count=count,
                prefix_sha256=digest,
            )
        except (FinalPublicationValidationError, ReservationConflictError):
            return "partial"
        if source.stat().st_size == count:
            return "complete-short"
        return "complete-full" if video_state != "reserved-target" else "encoding-prepass-staged"

    def _final_manifest_state(self) -> str:  # noqa: C901
        transaction = self.final_manifest_transaction
        target = self.job_root / FINAL_VIDEO_MANIFEST_RELATIVE_PATH
        if transaction is None:
            if not _regular_nonlink(target):
                raise ReservationConflictError("committed final manifest is missing")
            parse_final_video_manifest_v4(target.read_bytes())
            return "committed-new"
        source = transaction.source_path
        if not source.exists():
            if _regular_nonlink(target):
                parse_final_video_manifest_v4(target.read_bytes())
                return "committed-new"
            raise ReservationConflictError("final manifest extent has no authorized location")
        prefix = self._canonical_prefix_byte_count(source)
        if prefix is None:
            return "partial"
        if prefix == 0:
            return "zero"
        if prefix > transaction.descriptor.payload_raw_max:
            return "partial"
        with source.open("rb", buffering=0) as handle:
            raw = handle.read(prefix)
        try:
            parse_final_video_manifest_v4(raw)
        except FinalPublicationValidationError:
            return "partial"
        if source.stat().st_size == prefix:
            return "complete-short"
        if source.stat().st_size == transaction.descriptor.logical_byte_count:
            return "complete-full"
        return "partial"

    def _validate_committed_publication_triad(self) -> None:
        _validate_committed_publication_triad_v1(self.job_root, self.index)

    def observe(self) -> FinalPublicationRecovery:
        video_state = self._video_state()
        input_state = self._input_manifest_state(video_state)
        final_state = self._final_manifest_state()
        if (video_state, input_state, final_state) == (
            "committed-new",
            "committed-new",
            "committed-new",
        ):
            self._validate_committed_publication_triad()
        return classify_final_publication_observation(
            FinalPublicationObservation(
                video_state=video_state,
                input_manifest_state=input_state,
                final_manifest_state=final_state,
                sibling_present=self.sibling_path.exists() or self.sibling_path.is_symlink(),
            )
        )

    def recover_video_replacement_gap(self) -> None:
        """Move the indexed sibling forward after Windows removed the old target."""

        recovery = self.observe()
        if recovery.phase not in {
            "windows_video_replace_gap_before_input_commit",
            "windows_video_replace_gap_after_input_commit",
        }:
            raise ReservationConflictError("final video is not in a replacement gap")
        if os.name != "nt":
            raise ReservationConflictError("final-video replacement gaps are Windows-only")
        if not _regular_nonlink(self.sibling_path):
            raise ReservationConflictError("indexed replacement-gap sibling is unsafe")
        if self.final_video_path.exists() or self.final_video_path.is_symlink():
            raise ReservationConflictError("replacement-gap final target reappeared")
        same_parent_replace_v1(
            self.sibling_path,
            self.final_video_path,
            expected_source_identity=_reservation_identity(self.sibling_path),
            expected_target_identity=None,
            expected_parent_identity=self.target_descriptor.target_parent_identity,
            replace_existing=False,
        )
        if self._video_state() != "committed-new":
            raise ReservationConflictError("replacement-gap video forward move did not commit")

    def reset_incomplete_manifest_sources(self) -> None:
        """Wholly zero only indexed zero/partial source frames before reconstruction."""

        video_state = self._video_state()
        states = (
            (
                self.input_manifest_transaction,
                self._input_manifest_state(video_state),
                {"zero", "partial"},
            ),
            (
                self.final_manifest_transaction,
                self._final_manifest_state(),
                {"zero", "partial"},
            ),
        )
        zero_chunk = b"\x00" * (64 * 1024)
        for transaction, state, resettable in states:
            if transaction is None or state not in resettable:
                continue
            with transaction.source_path.open("r+b", buffering=0) as handle:
                handle.seek(0)
                remaining = transaction.descriptor.logical_byte_count
                while remaining:
                    chunk = zero_chunk[: min(len(zero_chunk), remaining)]
                    if handle.write(chunk) != len(chunk):
                        raise OSError("short manifest reservation reset")
                    remaining -= len(chunk)
                handle.truncate(transaction.descriptor.logical_byte_count)
                handle.flush()
                os.fsync(handle.fileno())

    def input_publication_source(self) -> "EncodingInputPublicationSource | None":
        transaction = self.input_manifest_transaction
        if transaction is None:
            target = self.job_root / ENCODING_INPUT_MANIFEST_RELATIVE_PATH
            return EncodingInputPublicationSource.from_path(target)
        source = transaction.source_path
        prefix = self._canonical_prefix_byte_count(source)
        if prefix in {None, 0}:
            return None
        return EncodingInputPublicationSource.from_path(
            source,
            raw_byte_count=prefix,
            logical_byte_count=(
                transaction.descriptor.logical_byte_count
                if source.stat().st_size == transaction.descriptor.logical_byte_count
                else None
            ),
        )

    def recover_input_manifest_intent(self) -> None:
        transaction = self.input_manifest_transaction
        if transaction is None:
            return
        source = self.input_publication_source()
        if source is None:
            raise ReservationConflictError("input manifest has no complete recovery intent")
        self.validate_indexed_predecessor(transaction)
        transaction.recover_streamed_prefix(
            prefix_byte_count=source.raw_byte_count,
            prefix_sha256=source.raw_sha256,
        )
        self.input_manifest_transaction = None

    def recover_final_manifest_intent(self) -> FinalVideoManifestV4:
        transaction = self.final_manifest_transaction
        if transaction is None:
            target = self.job_root / FINAL_VIDEO_MANIFEST_RELATIVE_PATH
            return parse_final_video_manifest_v4(target.read_bytes())
        prefix = self._canonical_prefix_byte_count(transaction.source_path)
        if prefix in {None, 0} or prefix > transaction.descriptor.payload_raw_max:
            raise ReservationConflictError("final manifest has no complete recovery intent")
        with transaction.source_path.open("rb", buffering=0) as handle:
            raw = handle.read(prefix)
        manifest = parse_final_video_manifest_v4(raw)
        video_hash, video_count = _hash_file(self.final_video_path)
        if (video_hash, video_count) != (manifest.sha256, manifest.byte_count):
            raise ReservationConflictError("final manifest no longer binds the current video")
        input_source = EncodingInputPublicationSource.from_path(
            self.job_root / ENCODING_INPUT_MANIFEST_RELATIVE_PATH
        )
        if input_source.fingerprint != manifest.input_stage_fingerprint:
            raise ReservationConflictError("final manifest no longer binds the input manifest")
        self.validate_indexed_predecessor(transaction)
        transaction.recover(raw)
        self.final_manifest_transaction = None
        return manifest

    def _retire_committed_manifest_descriptor_prefix(self) -> bool:
        input_transaction = self.input_manifest_transaction
        if input_transaction is not None and not (
            input_transaction.source_path.exists()
            or input_transaction.source_path.is_symlink()
        ):
            source = EncodingInputPublicationSource.from_path(
                self.job_root / ENCODING_INPUT_MANIFEST_RELATIVE_PATH
            )
            input_transaction.recover_streamed_prefix(
                prefix_byte_count=source.raw_byte_count,
                prefix_sha256=source.raw_sha256,
            )
            self.input_manifest_transaction = None
            return True
        final_transaction = self.final_manifest_transaction
        if final_transaction is not None and not (
            final_transaction.source_path.exists()
            or final_transaction.source_path.is_symlink()
        ):
            target = self.job_root / FINAL_VIDEO_MANIFEST_RELATIVE_PATH
            raw = _read_bounded_regular_file(
                target,
                max_bytes=final_transaction.descriptor.payload_raw_max,
            )
            manifest = parse_final_video_manifest_v4(raw)
            video_hash, video_count = _hash_file(self.final_video_path)
            input_source = EncodingInputPublicationSource.from_path(
                self.job_root / ENCODING_INPUT_MANIFEST_RELATIVE_PATH
            )
            if (
                (manifest.sha256, manifest.byte_count) != (video_hash, video_count)
                or manifest.input_stage_fingerprint != input_source.fingerprint
            ):
                raise ReservationConflictError(
                    "committed final manifest does not bind its publication triad"
                )
            final_transaction.recover(raw)
            self.final_manifest_transaction = None
            return True
        return False

    def recover_publication_prefix(self) -> FinalPublicationRecovery:
        """Finish every deterministic, index-owned publication prefix step."""

        for _ in range(7):
            if self._retire_committed_manifest_descriptor_prefix():
                continue
            recovery = self.observe()
            if recovery.phase in {
                "windows_video_replace_gap_before_input_commit",
                "windows_video_replace_gap_after_input_commit",
            }:
                self.recover_video_replacement_gap()
                continue
            if recovery.phase == "video_replaced_input_intent":
                self.recover_input_manifest_intent()
                continue
            if recovery.phase == "final_manifest_intent":
                self.recover_final_manifest_intent()
                continue
            return recovery
        raise ReservationConflictError("final publication recovery did not converge")

    def retire(
        self,
        *,
        event: Callable[[str], None] | None = None,
        crash_at: str | None = None,
    ) -> None:
        if self.sibling_path.exists() or self.sibling_path.is_symlink():
            raise ReservationConflictError(
                "indexed sibling remains at final reservation retirement"
            )
        for entry in self.index.extent_entries:
            if entry["reservation_kind"] == "job_terminal_settings":
                continue
            descriptor, source = _indexed_extent_paths(self.job_root, entry)
            if descriptor.exists() or descriptor.is_symlink():
                raise ReservationConflictError(
                    f"{entry['reservation_kind']} descriptor remains active"
                )
            if source.exists() or source.is_symlink():
                raise ReservationConflictError(f"{entry['reservation_kind']} source remains active")
        self.target_descriptor_path.unlink(missing_ok=True)
        _sync_parent(self.target_descriptor_path.parent)
        emit = event or self.event
        if emit is not None:
            emit("reservation:target-descriptor-retired")
        if crash_at == "after-target-descriptor-retirement":
            raise InjectedFinalPublicationCrash(crash_at)
        self.index_path.unlink(missing_ok=True)
        _sync_parent(self.index_path.parent)
        if emit is not None:
            emit("reservation:index-retired")

    def retire_terminal_failure(  # noqa: C901
        self, *, event: Callable[[str], None] | None = None
    ) -> None:
        """Remove only authenticated unused reservation objects after durable failure."""

        transactions = [
            self.producer_settings_transaction,
            self.input_manifest_transaction,
            self.final_manifest_transaction,
            *(transaction for _entry, transaction, _payload in self.prune_marker_transactions),
            self.diagnostics_payload_pruned_transaction,
            self.cleanup_pending_settings_transaction,
        ]
        active = next((item for item in transactions if item is not None), None)
        allocation_adapter = (
            active.allocation_adapter if active is not None else platform_allocation_adapter
        )
        allocation_counter = (
            active.allocation_counter if active is not None else platform_allocation_counter
        )
        settings = ProcessingSettingsArtifactV5.parse(
            _read_bounded_regular_file(
                self.job_root / "processing-settings-v5.json",
                max_bytes=512 * 1024,
            )
        )
        if settings.status != "failed":
            raise ReservationConflictError(
                "failure retirement requires durable failed terminal settings"
            )
        while self._retire_committed_manifest_descriptor_prefix():
            pass
        _recover_failed_final_retirement_v1(
            self.job_root,
            self.index,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
            event=event or self.event,
            crash_at=None,
        )


def _extent_plan(
    *,
    kind: str,
    target: Path,
    raw_max: int,
    allocation_unit: int,
    generation: str,
    reservation_id: str,
    job_root: Path,
    publication_generation: str,
) -> dict:
    target_path = job_root / target
    if target_path.exists() or target_path.is_symlink():
        method = "replace_existing"
        if kind in {
            "encoding_input_manifest",
            "final_video_manifest",
            "diagnostics_payload_pruned",
        }:
            identity = _reservation_identity(target_path)
            target_sha256, target_byte_count = _hash_file(target_path)
        else:
            identity = None
            target_sha256 = None
            target_byte_count = None
    else:
        method = "replace_placeholder"
        identity = None
        target_sha256 = None
        target_byte_count = None
    source_relative = target.parent / (
        f".depth-surge-r-{generation}-{kind}-{reservation_id}.reserved"
    )
    descriptor_relative = Path(".depth-surge-reservations-v1") / (
        f"d-p-{publication_generation}-r-{generation}-{kind}-{reservation_id}.json"
    )
    payload_roles = {
        "producer_settings": "settings_producer",
        "cleanup_pending_settings": "settings_cleanup_pending",
    }
    return {
        "reservation_generation": generation,
        "reservation_id": reservation_id,
        "reservation_kind": kind,
        "descriptor_publication_generation": publication_generation,
        "descriptor_relative_path": descriptor_relative.as_posix(),
        "source_extent_relative_path": source_relative.as_posix(),
        "temporary_relative_path": source_relative.as_posix(),
        "target_relative_path": target.as_posix(),
        "target_parent_identity": _reservation_directory_identity(target_path.parent).to_object(),
        "publication_method": method,
        "existing_target_identity_at_preflight": (
            identity.to_object() if identity is not None else None
        ),
        "existing_target_byte_count_at_preflight": target_byte_count,
        "existing_target_raw_sha256_at_preflight": target_sha256,
        "payload_role": payload_roles.get(kind, kind),
        "payload_raw_max": raw_max,
        "logical_byte_count": payload_logical_bytes(raw_max, allocation_unit),
        "allocation_unit": allocation_unit,
        "prune_root_relative_path": None,
        "marker_name": None,
        "marker_payload_sha256": None,
    }


def _indexed_extent_paths(root: Path, entry: dict) -> tuple[Path, Path]:
    validated = validate_reservation_entry_v1(entry)
    return (
        root / validated["descriptor_relative_path"],
        root / validated["source_extent_relative_path"],
    )


def _utc_now_usec() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _diagnostics_prune_preflight(job_root: Path) -> tuple[bytes, str]:
    path = Path(job_root) / "04_stereo_diagnostics" / "metadata.json"
    raw = path.read_bytes()
    try:
        value = strict_canonical_json_loads(raw, max_bytes=16 * 1024 * 1024)
    except CanonicalJsonError as error:
        raise FinalPublicationValidationError(
            "diagnostics metadata is not canonical at final preflight"
        ) from error
    if (
        not isinstance(value, dict)
        or value.get("status") not in {"complete", "legacy_fast_unavailable"}
        or value.get("fingerprint") != self_fingerprint(value)
    ):
        raise FinalPublicationValidationError(
            "diagnostics metadata is not complete at final preflight"
        )
    return raw, value["fingerprint"]


def prepare_final_encoding_reservation_v1(  # noqa: C901
    *,
    job_root: Path,
    settings: ProcessingSettingsArtifactV5,
    publication_generation: str,
    input_manifest_raw_max: int,
    final_manifest_raw_max: int,
    allocation_unit: int | None = None,
    allocation_adapter: AllocationAdapter = platform_allocation_adapter,
    allocation_counter: AllocationCounter | None = platform_allocation_counter,
    reservation_generation: str | None = None,
    input_reservation_id: str | None = None,
    final_reservation_id: str | None = None,
    event: Callable[[str], None] | None = None,
) -> PreparedFinalEncodingReservationV1:
    """Publish the final-encoding index before materializing any owned path."""

    root = Path(job_root).resolve()
    values = settings.to_object()
    metadata = values["metadata"]
    if metadata["final_media_publication_generation"] not in {
        None,
        publication_generation,
    }:
        raise FinalPublicationValidationError("reservation generation differs from settings")
    if not re.fullmatch(r"[0-9a-f]{32}", publication_generation, flags=re.ASCII):
        raise FinalPublicationValidationError("publication generation is invalid")
    if input_manifest_raw_max <= 0 or final_manifest_raw_max <= 0:
        raise ValueError("final reservation raw bounds must be positive")
    final_relative = values["output_info"]["expected_final_relative_path"]
    validate_portable_final_component_v1(final_relative)
    final_path = root / final_relative
    sibling_relative = f".depth-surge-final-v4-{publication_generation}.tmp.mp4"
    sibling_path = root / sibling_relative
    if sibling_path.exists() or sibling_path.is_symlink():
        raise ReservationConflictError("fixed final-video sibling collides at preflight")
    if final_path.exists() or final_path.is_symlink():
        existing_identity = _reservation_identity(final_path)
        publication_method: Literal["replace_placeholder", "replace_existing"] = "replace_existing"
    else:
        existing_identity = None
        publication_method = "replace_placeholder"
    settings_path = root / "processing-settings-v5.json"
    settings_raw = settings_path.read_bytes()
    if ProcessingSettingsArtifactV5.parse(settings_raw) != settings:
        raise FinalPublicationValidationError("settings bytes differ at reservation preflight")
    raw_sha = hashlib.sha256(settings_raw).hexdigest()
    control = load_job_control(root)
    if (
        control.job_id != metadata["job_id"]
        or control.settings_identity_fingerprint
        != metadata["settings_identity_fingerprint"]
        or control.job_reservation_generation
        != metadata["job_reservation_generation"]
        or control.job_terminal_reservation_id
        != metadata["job_terminal_reservation_id"]
        or control.job_root_identity != _reservation_directory_identity(root)
    ):
        raise ReservationConflictError("job control differs from final settings")
    terminal_entries = [
        dict(entry)
        for entry in control.bootstrap_extent_entries
        if entry.get("reservation_kind") == "job_terminal_settings"
    ]
    if len(terminal_entries) != 1:
        raise ReservationConflictError("job control has no unique terminal extent")
    reserve_generation = reservation_generation or secrets.token_hex(16)
    input_id = input_reservation_id or secrets.token_hex(16)
    final_id = final_reservation_id or secrets.token_hex(16)
    for token, label in (
        (reserve_generation, "reservation generation"),
        (input_id, "input reservation id"),
        (final_id, "final reservation id"),
    ):
        if not re.fullmatch(r"[0-9a-f]{32}", token, flags=re.ASCII):
            raise FinalPublicationValidationError(f"{label} is invalid")
    unit = allocation_unit or allocation_unit_for_path(root)
    producer_target = settings.with_producer_marker(
        generation=publication_generation,
        updated_at=_utc_now_usec(),
    )
    producer_payload = producer_target.encode() if producer_target != settings else None
    producer_plan = None
    if producer_payload is not None:
        producer_plan = _extent_plan(
            kind="producer_settings",
            target=Path("processing-settings-v5.json"),
            raw_max=512 * 1024,
            allocation_unit=unit,
            generation=reserve_generation,
            reservation_id=secrets.token_hex(16),
            job_root=root,
            publication_generation=publication_generation,
        )
    input_plan = _extent_plan(
        kind="encoding_input_manifest",
        target=ENCODING_INPUT_MANIFEST_RELATIVE_PATH,
        raw_max=input_manifest_raw_max,
        allocation_unit=unit,
        generation=reserve_generation,
        reservation_id=input_id,
        job_root=root,
        publication_generation=publication_generation,
    )
    keep_intermediates = values["processing_settings"]["keep_intermediates"]
    diagnostics_raw: bytes | None = None
    diagnostics_fingerprint: str | None = None
    marker_specs = ()
    marker_plans: list[dict] = []
    diagnostics_plan = None
    cleanup_plan = None
    cleanup_payload: bytes | None = None
    if not keep_intermediates:
        diagnostics_raw, diagnostics_fingerprint = _diagnostics_prune_preflight(root)
        marker_specs = plan_prune_markers_v1(root)
        for marker in marker_specs:
            plan = _extent_plan(
                kind="prune_marker",
                target=Path(marker.relative_path) / marker.marker_name,
                raw_max=len(marker.payload),
                allocation_unit=unit,
                generation=reserve_generation,
                reservation_id=secrets.token_hex(16),
                job_root=root,
                publication_generation=publication_generation,
            )
            plan.update(
                {
                    "prune_root_relative_path": marker.relative_path,
                    "marker_name": marker.marker_name,
                    "marker_payload_sha256": marker.payload_sha256,
                }
            )
            marker_plans.append(plan)
        diagnostics_plan = _extent_plan(
            kind="diagnostics_payload_pruned",
            target=Path("04_stereo_diagnostics/metadata.json"),
            raw_max=16 * 1024 * 1024,
            allocation_unit=unit,
            generation=reserve_generation,
            reservation_id=secrets.token_hex(16),
            job_root=root,
            publication_generation=publication_generation,
        )
        cleanup_target = producer_target.cleanup_pending(updated_at=_utc_now_usec())
        cleanup_payload = cleanup_target.encode()
        cleanup_plan = _extent_plan(
            kind="cleanup_pending_settings",
            target=Path("processing-settings-v5.json"),
            raw_max=512 * 1024,
            allocation_unit=unit,
            generation=reserve_generation,
            reservation_id=secrets.token_hex(16),
            job_root=root,
            publication_generation=publication_generation,
        )
    final_plan = _extent_plan(
        kind="final_video_manifest",
        target=FINAL_VIDEO_MANIFEST_RELATIVE_PATH,
        raw_max=final_manifest_raw_max,
        allocation_unit=unit,
        generation=reserve_generation,
        reservation_id=final_id,
        job_root=root,
        publication_generation=publication_generation,
    )
    target_descriptor_relative = (
        Path(".depth-surge-reservations-v1")
        / f"final-video-target-p-{publication_generation}-r-{reserve_generation}.json"
    )
    extent_entries: list[dict] = []
    if producer_plan is not None:
        extent_entries.append(producer_plan)
    extent_entries.extend((input_plan, final_plan))
    extent_entries.extend(marker_plans)
    if diagnostics_plan is not None:
        extent_entries.append(diagnostics_plan)
    if cleanup_plan is not None:
        extent_entries.append(cleanup_plan)
    extent_entries.extend(terminal_entries)
    validate_final_encoding_entry_order(
        [entry["reservation_kind"] for entry in extent_entries]
    )
    index = FinalEncodingReservationIndexV1.build(
        job_id=metadata["job_id"],
        settings_identity_fingerprint=metadata["settings_identity_fingerprint"],
        settings_raw_sha256_at_preflight=raw_sha,
        publication_generation=publication_generation,
        reservation_generation=reserve_generation,
        settings_revision_at_preflight=metadata["settings_revision"],
        final_video_relative_path=final_relative,
        final_video_publication_method=publication_method,
        final_video_existing_target_identity_at_preflight=existing_identity,
        final_video_target_descriptor_relative_path=target_descriptor_relative.as_posix(),
        sibling_video_temporary_relative_path=sibling_relative,
        sibling_video_parent_identity=_reservation_directory_identity(root),
        diagnostics_metadata_raw_sha256_at_preflight=(
            hashlib.sha256(diagnostics_raw).hexdigest() if diagnostics_raw is not None else None
        ),
        diagnostics_metadata_fingerprint_at_preflight=diagnostics_fingerprint,
        extent_entries=tuple(extent_entries),
    )
    index_path = root / FINAL_ENCODING_RESERVATION_RELATIVE_PATH
    publish_write_ahead_intent(index_path, index.to_object(), event=event)

    if publication_method == "replace_placeholder":
        with final_path.open("xb") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        _sync_parent(final_path.parent)
    reserved_identity = _reservation_identity(final_path)
    if existing_identity is not None and reserved_identity != existing_identity:
        raise ReservationConflictError("existing final target changed after index publication")
    target_descriptor = FinalVideoTargetReservationV1.build(
        job_id=index.job_id,
        settings_identity_fingerprint=index.settings_identity_fingerprint,
        settings_raw_sha256_at_preflight=index.settings_raw_sha256_at_preflight,
        settings_revision_at_preflight=index.settings_revision_at_preflight,
        publication_generation=publication_generation,
        reservation_generation=reserve_generation,
        transaction_index_fingerprint=index.fingerprint,
        final_video_relative_path=final_relative,
        sibling_video_temporary_relative_path=sibling_relative,
        target_parent_identity=index.sibling_video_parent_identity,
        publication_method=publication_method,
        reserved_target_identity=reserved_identity,
    )
    publish_write_ahead_intent(
        root / target_descriptor_relative,
        target_descriptor.to_object(),
    )
    if event is not None:
        event("final-target:reserved")

    def extent_event(kind: str) -> Callable[[str], None] | None:
        if event is None:
            return None
        return lambda value: event(f"{kind}:{value}")

    def create_extent(plan: dict, label: str) -> ReservedArtifactTransaction:
        transaction = ReservedArtifactTransaction.create(
            job_root=root,
            index_relative_path=FINAL_ENCODING_RESERVATION_RELATIVE_PATH,
            target_relative_path=Path(plan["target_relative_path"]),
            reservation_kind=plan["reservation_kind"],
            payload_raw_max=plan["payload_raw_max"],
            allocation_unit=unit,
            allocation_adapter=allocation_adapter,
            generation=reserve_generation,
            reservation_id=plan["reservation_id"],
            allocation_counter=allocation_counter,
            expected_publication_method=plan["publication_method"],
            event=extent_event(label),
        )
        if transaction.descriptor.publication_method != plan["publication_method"]:
            raise ReservationConflictError("extent publication method changed after preflight")
        return transaction

    producer_transaction = (
        create_extent(producer_plan, "producer-settings") if producer_plan is not None else None
    )
    input_transaction = create_extent(input_plan, "input-manifest")
    final_transaction = create_extent(final_plan, "final-manifest")
    marker_transactions: list[tuple[dict, ReservedArtifactTransaction, bytes]] = []
    for marker, plan in zip(marker_specs, marker_plans, strict=True):
        marker_transactions.append(
            (
                marker.prune_entry(),
                create_extent(plan, f"prune-marker:{marker.stage_key}"),
                marker.payload,
            )
        )
    diagnostics_transaction = (
        create_extent(diagnostics_plan, "diagnostics-payload-pruned")
        if diagnostics_plan is not None
        else None
    )
    cleanup_transaction = (
        create_extent(cleanup_plan, "cleanup-pending-settings")
        if cleanup_plan is not None
        else None
    )
    return PreparedFinalEncodingReservationV1(
        job_root=root,
        index=index,
        target_descriptor=target_descriptor,
        input_manifest_transaction=input_transaction,
        final_manifest_transaction=final_transaction,
        event=event,
        producer_settings_transaction=producer_transaction,
        producer_settings_payload=producer_payload,
        prune_marker_transactions=tuple(marker_transactions),
        diagnostics_payload_pruned_transaction=diagnostics_transaction,
        cleanup_pending_settings_transaction=cleanup_transaction,
        cleanup_pending_settings_payload=cleanup_payload,
    )


def _load_reservation_json(path: Path, *, max_bytes: int) -> object:
    try:
        return strict_canonical_json_loads(
            _read_bounded_regular_file(path, max_bytes=max_bytes),
            max_bytes=max_bytes,
        )
    except (OSError, CanonicalJsonError) as error:
        raise ReservationConflictError(
            f"reservation control artifact is malformed: {path}"
        ) from error


def recover_final_encoding_index_pair_v1(
    job_root: Path,
) -> FinalEncodingReservationIndexV1 | None:
    root = Path(job_root).resolve()
    path = root / FINAL_ENCODING_RESERVATION_RELATIVE_PATH

    def parse(raw: bytes) -> FinalEncodingReservationIndexV1:
        return FinalEncodingReservationIndexV1.from_object(
            strict_canonical_json_loads(raw, max_bytes=256 * 1024)
        )

    def validate_before_publish(value: object) -> None:
        if not isinstance(value, FinalEncodingReservationIndexV1):
            raise ReservationConflictError("final encoding temporary has the wrong type")
        _validate_index_job_control(root, value)
        settings_path = root / "processing-settings-v5.json"
        if not _regular_nonlink(settings_path):
            raise ReservationConflictError(
                "final encoding temporary has no durable preflight settings"
            )
        settings_raw = _read_bounded_regular_file(settings_path, max_bytes=512 * 1024)
        settings = ProcessingSettingsArtifactV5.parse(settings_raw)
        metadata = settings.to_object()["metadata"]
        if (
            hashlib.sha256(settings_raw).hexdigest()
            != value.settings_raw_sha256_at_preflight
            or settings.revision != value.settings_revision_at_preflight
            or metadata["job_id"] != value.job_id
            or metadata["settings_identity_fingerprint"]
            != value.settings_identity_fingerprint
        ):
            raise ReservationConflictError(
                "final encoding temporary preflight settings changed"
            )
        settings_index = root / (
            ".depth-surge-reservations-v1/settings-transition-reservation-v1.json"
        )
        settings_index_temporary = settings_index.with_name(
            f".{settings_index.name}.create-new.tmp"
        )
        if any(
            candidate.exists() or candidate.is_symlink()
            for candidate in (settings_index, settings_index_temporary)
        ):
            raise ReservationConflictError(
                "final encoding candidate overlaps a settings authority"
            )
        control = load_job_control(root)
        candidate_temporary = path.with_name(f".{path.name}.create-new.tmp")
        assert_only_job_control_owned_reservations_v1(
            root,
            control,
            allowed_relative_paths=(
                candidate_temporary.relative_to(root).as_posix(),
            ),
        )
        target_descriptor = root / value.final_video_target_descriptor_relative_path
        target_descriptor_temporary = target_descriptor.with_name(
            f".{target_descriptor.name}.create-new.tmp"
        )
        sibling = root / value.sibling_video_temporary_relative_path
        if any(
            candidate.exists() or candidate.is_symlink()
            for candidate in (
                target_descriptor,
                target_descriptor_temporary,
                sibling,
            )
        ):
            raise ReservationConflictError(
                "final encoding candidate has downstream target artifacts"
            )
        final_target = root / value.final_video_relative_path
        if value.final_video_publication_method == "replace_placeholder":
            if final_target.exists() or final_target.is_symlink():
                raise ReservationConflictError(
                    "final encoding candidate placeholder already exists"
                )
        elif (
            not _regular_nonlink(final_target)
            or _reservation_identity(final_target)
            != value.final_video_existing_target_identity_at_preflight
        ):
            raise ReservationConflictError(
                "final encoding candidate existing target changed"
            )
        for entry in value.extent_entries:
            if entry["reservation_kind"] == "job_terminal_settings":
                continue
            descriptor, source = _indexed_extent_paths(root, entry)
            descriptor_temporary = descriptor.with_name(
                f".{descriptor.name}.create-new.tmp"
            )
            if any(
                candidate.exists() or candidate.is_symlink()
                for candidate in (descriptor, descriptor_temporary, source)
            ):
                raise ReservationConflictError(
                    "final encoding candidate has downstream extent artifacts"
                )
            target = root / entry["target_relative_path"]
            if entry["publication_method"] == "replace_placeholder":
                if target.exists() or target.is_symlink():
                    raise ReservationConflictError(
                        "final encoding candidate extent placeholder already exists"
                    )
                continue
            expected = entry["existing_target_identity_at_preflight"]
            if (
                expected is None
                or not _regular_nonlink(target)
                or _reservation_identity(target)
                != ReservationFileIdentity.from_object(expected)
                or _hash_file(target)
                != (
                    entry["existing_target_raw_sha256_at_preflight"],
                    entry["existing_target_byte_count_at_preflight"],
                )
            ):
                raise ReservationConflictError(
                    "final encoding candidate extent target changed"
                )

    def validate_before_unlink() -> None:
        control = load_job_control(root)
        candidate_temporary = path.with_name(f".{path.name}.create-new.tmp")
        assert_only_job_control_owned_reservations_v1(
            root,
            control,
            allowed_relative_paths=(
                candidate_temporary.relative_to(root).as_posix(),
            ),
        )

    recovered = recover_control_artifact_pair_v1(
        path,
        max_bytes=256 * 1024,
        parser=parse,
        prepublish_validate=validate_before_publish,
        preunlink_validate=validate_before_unlink,
    )
    if recovered is None:
        return None
    if not isinstance(recovered, FinalEncodingReservationIndexV1):
        raise ReservationConflictError("final encoding control type changed")
    _validate_index_job_control(root, recovered)
    target_descriptor = PurePosixPath(
        recovered.final_video_target_descriptor_relative_path
    )
    assert_index_owned_reservation_namespace_v1(
        root,
        load_job_control(root),
        index_relative_path=FINAL_ENCODING_RESERVATION_RELATIVE_PATH.as_posix(),
        extent_entries=recovered.extent_entries,
        extra_relative_paths=(
            target_descriptor.as_posix(),
            (
                target_descriptor.parent
                / f".{target_descriptor.name}.create-new.tmp"
            ).as_posix(),
            recovered.sibling_video_temporary_relative_path,
        ),
    )
    return recovered


def _load_indexed_extent(  # noqa: C901
    *,
    root: Path,
    index: FinalEncodingReservationIndexV1,
    entry: dict,
    allocation_adapter: AllocationAdapter,
    allocation_counter: AllocationCounter | None,
    committed: bool = False,
) -> ReservedArtifactTransaction | None:
    kind = entry.get("reservation_kind")
    if not isinstance(kind, str):
        raise ReservationConflictError("final index extent kind is invalid")
    descriptor_path, source_path = _indexed_extent_paths(root, entry)
    if not descriptor_path.exists():
        target_path = root / entry["target_relative_path"]
        target_is_preflight = False
        if target_path.exists() or target_path.is_symlink():
            if entry["existing_target_identity_at_preflight"] is None:
                target_is_preflight = (
                    _regular_nonlink(target_path) and target_path.stat().st_size == 0
                )
            else:
                expected_identity = ReservationFileIdentity.from_object(
                    entry["existing_target_identity_at_preflight"]
                )
                target_is_preflight = _reservation_identity(
                    target_path
                ) == expected_identity and _hash_file(target_path) == (
                    entry["existing_target_raw_sha256_at_preflight"],
                    entry["existing_target_byte_count_at_preflight"],
                )
        if not source_path.exists() and committed:
            return None
        if not source_path.exists() and not target_is_preflight and target_path.exists():
            try:
                if kind == "encoding_input_manifest":
                    _stream_input_manifest_identity(target_path)
                elif kind == "final_video_manifest":
                    parse_final_video_manifest_v4(target_path.read_bytes())
                else:
                    raise FinalPublicationValidationError(
                        f"descriptorless {kind} target has no consumption evidence"
                    )
            except FinalPublicationValidationError as error:
                raise ReservationConflictError(
                    f"descriptorless {kind} target is not committed evidence"
                ) from error
            return None
        transaction = ReservedArtifactTransaction.create(
            job_root=root,
            index_relative_path=FINAL_ENCODING_RESERVATION_RELATIVE_PATH,
            target_relative_path=Path(entry["target_relative_path"]),
            reservation_kind=kind,
            payload_raw_max=entry["payload_raw_max"],
            allocation_unit=entry["allocation_unit"],
            allocation_adapter=allocation_adapter,
            generation=entry["reservation_generation"],
            reservation_id=entry["reservation_id"],
            allocation_counter=allocation_counter,
            expected_publication_method=entry["publication_method"],
        )
        return transaction
    transaction = ReservedArtifactTransaction.load(
        descriptor_path,
        allocation_adapter=allocation_adapter,
        allocation_counter=allocation_counter,
    )
    if kind == "job_terminal_settings":
        control = load_job_control(root)
        owner_job_id = control.job_id
        owner_settings_identity = control.settings_identity_fingerprint
        owner_revision = None
        owner_raw_sha256 = None
        owner_index_path = Path("job-control-v1.json")
        owner_index_fingerprint = control.fingerprint
    else:
        owner_job_id = index.job_id
        owner_settings_identity = index.settings_identity_fingerprint
        owner_revision = index.settings_revision_at_preflight
        owner_raw_sha256 = index.settings_raw_sha256_at_preflight
        owner_index_path = FINAL_ENCODING_RESERVATION_RELATIVE_PATH
        owner_index_fingerprint = index.fingerprint
    validate_descriptor_against_entry_v1(
        transaction.descriptor,
        entry,
        job_id=owner_job_id,
        settings_identity_fingerprint=owner_settings_identity,
        settings_revision_at_creation=owner_revision,
        settings_raw_sha256_at_creation=owner_raw_sha256,
        transaction_index_relative_path=owner_index_path,
        transaction_index_fingerprint=owner_index_fingerprint,
    )
    return transaction


def _unique_extent_entry(index: FinalEncodingReservationIndexV1, kind: str) -> dict:
    entries = [entry for entry in index.extent_entries if entry.get("reservation_kind") == kind]
    if len(entries) != 1:
        raise ReservationConflictError(f"final index has no unique {kind} entry")
    return entries[0]


def _validate_index_job_control(
    root: Path,
    index: FinalEncodingReservationIndexV1,
) -> None:
    control = load_job_control(root)
    control_terminal = [
        dict(entry)
        for entry in control.bootstrap_extent_entries
        if entry.get("reservation_kind") == "job_terminal_settings"
    ]
    index_terminal = [
        dict(entry)
        for entry in index.extent_entries
        if entry.get("reservation_kind") == "job_terminal_settings"
    ]
    if (
        index.job_id != control.job_id
        or index.settings_identity_fingerprint != control.settings_identity_fingerprint
        or index.sibling_video_parent_identity != control.job_root_identity
        or len(control_terminal) != 1
        or len(index_terminal) != 1
        or index_terminal[0] != control_terminal[0]
    ):
        raise ReservationConflictError("final index terminal plan differs from job control")


def _settings_payload_intent(
    transaction: ReservedArtifactTransaction | None,
) -> bytes | None:
    if transaction is None:
        return None
    path = transaction.source_path if transaction.source_path.exists() else transaction.target_path
    prefix = PreparedFinalEncodingReservationV1._canonical_prefix_byte_count(path)
    if prefix in {None, 0}:
        return None
    if prefix > transaction.descriptor.payload_raw_max:
        raise ReservationConflictError("indexed settings payload exceeds its raw bound")
    with path.open("rb", buffering=0) as handle:
        raw = handle.read(prefix)
    try:
        artifact = ProcessingSettingsArtifactV5.parse(raw)
    except SettingsArtifactSchemaError:
        return None
    if artifact.encode() != raw:
        raise ReservationConflictError("indexed settings payload intent is noncanonical")
    return raw


def recover_settings_replacement_gap_from_final_index_v1(
    job_root: Path,
    *,
    allocation_adapter: AllocationAdapter = platform_allocation_adapter,
    allocation_counter: AllocationCounter | None = platform_allocation_counter,
) -> ProcessingSettingsArtifactV5 | None:
    """Forward-complete the sole settings replacement gap owned by an active final index."""

    root = Path(job_root).resolve()
    fixed = root / "processing-settings-v5.json"
    if fixed.is_symlink() or (fixed.exists() and not fixed.is_file()):
        raise ReservationConflictError("fixed settings target is unsafe")
    if fixed.is_file():
        return ProcessingSettingsArtifactV5.parse(
            _read_bounded_regular_file(fixed, max_bytes=512 * 1024)
        )
    index = recover_final_encoding_index_pair_v1(root)
    if index is None:
        return None
    control = load_job_control(root)
    candidates: list[
        tuple[ReservedArtifactTransaction, ProcessingSettingsArtifactV5]
    ] = []
    for entry in index.extent_entries:
        kind = entry["reservation_kind"]
        if kind not in {
            "producer_settings",
            "cleanup_pending_settings",
            "job_terminal_settings",
        }:
            continue
        descriptor_path = root / entry["descriptor_relative_path"]
        if not descriptor_path.is_file() or descriptor_path.is_symlink():
            continue
        transaction = ReservedArtifactTransaction.load(
            descriptor_path,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
        )
        if kind == "job_terminal_settings":
            owner_revision = None
            owner_hash = None
            owner_path = Path("job-control-v1.json")
            owner_fingerprint = control.fingerprint
        else:
            owner_revision = index.settings_revision_at_preflight
            owner_hash = index.settings_raw_sha256_at_preflight
            owner_path = FINAL_ENCODING_RESERVATION_RELATIVE_PATH
            owner_fingerprint = index.fingerprint
        validate_descriptor_against_entry_v1(
            transaction.descriptor,
            entry,
            job_id=index.job_id,
            settings_identity_fingerprint=index.settings_identity_fingerprint,
            settings_revision_at_creation=owner_revision,
            settings_raw_sha256_at_creation=owner_hash,
            transaction_index_relative_path=owner_path,
            transaction_index_fingerprint=owner_fingerprint,
        )
        if not transaction.source_path.is_file() or transaction.source_path.is_symlink():
            continue
        payload = _settings_payload_intent(transaction)
        if payload is None:
            continue
        artifact = ProcessingSettingsArtifactV5.parse(payload)
        metadata = artifact.to_object()["metadata"]
        if (
            metadata["job_id"] != index.job_id
            or metadata["settings_identity_fingerprint"]
            != index.settings_identity_fingerprint
        ):
            raise ReservationConflictError("indexed settings gap belongs to another job")
        if kind == "producer_settings" and (
            metadata["final_media_publication_generation"]
            != index.publication_generation
        ):
            raise ReservationConflictError("producer settings gap has another generation")
        if kind == "cleanup_pending_settings" and metadata["cleanup_status"] != "pending":
            raise ReservationConflictError("cleanup settings gap is not pending")
        if kind == "job_terminal_settings" and artifact.status not in {
            "completed",
            "failed",
        }:
            raise ReservationConflictError("terminal settings gap is nonterminal")
        if transaction.validate_physical_phase(payload) == "replacement-gap":
            candidates.append((transaction, artifact))
    if len(candidates) != 1:
        raise ReservationConflictError(
            "missing fixed settings has no unique active-index replacement gap"
        )
    transaction, artifact = candidates[0]
    transaction.recover(artifact.encode())
    reopened = ProcessingSettingsArtifactV5.parse(
        _read_bounded_regular_file(fixed, max_bytes=512 * 1024)
    )
    if reopened != artifact:
        raise ReservationConflictError("settings replacement-gap recovery changed payload")
    return reopened


def _marker_payload_from_entry(entry: dict) -> tuple[dict, bytes]:
    relative_path = entry.get("prune_root_relative_path")
    name = entry.get("marker_name")
    if not isinstance(name, str) or not re.fullmatch(
        r"\.depth-surge-prune-v2-[0-9a-f]{32}\.marker", name, flags=re.ASCII
    ) or not isinstance(relative_path, str):
        raise ReservationConflictError("prune marker name is invalid")
    token = name.removeprefix(".depth-surge-prune-v2-").removesuffix(".marker")
    payload = (
        canonical_json_bytes(
            {
                "cleanup_contract_version": 2,
                "relative_path": relative_path,
                "token": token,
            }
        )
        + b"\n"
    )
    if (
        entry.get("prune_root_relative_path") != relative_path
        or entry.get("marker_name") != name
        or entry.get("marker_payload_sha256") != hashlib.sha256(payload).hexdigest()
        or entry.get("target_relative_path") != f"{relative_path}/{name}"
    ):
        raise ReservationConflictError("prune marker plan differs from its authorization")
    try:
        prune_entry = prune_entry_from_reservation_v1(
            relative_path=relative_path,
            directory_identity=entry.get("target_parent_identity"),
            marker_name=name,
            marker_payload_sha256=entry["marker_payload_sha256"],
        )
    except Exception as error:
        raise ReservationConflictError("prune marker authorization is malformed") from error
    return prune_entry, payload


def _load_diagnostics_metadata(root: Path) -> tuple[dict, bytes]:
    path = root / "04_stereo_diagnostics" / "metadata.json"
    raw = path.read_bytes()
    try:
        value = strict_canonical_json_loads(raw, max_bytes=16 * 1024 * 1024)
    except CanonicalJsonError as error:
        raise ReservationConflictError("diagnostics metadata is noncanonical") from error
    if not isinstance(value, dict) or value.get("fingerprint") != self_fingerprint(value):
        raise ReservationConflictError("diagnostics metadata fingerprint is invalid")
    return value, raw


def _validate_completed_descriptorless_retirement_v1(
    root: Path,
    index: FinalEncodingReservationIndexV1,
    settings: ProcessingSettingsArtifactV5,
) -> None:
    _validate_committed_publication_triad_v1(root, index)
    metadata = settings.to_object()["metadata"]
    kinds = [entry["reservation_kind"] for entry in index.extent_entries]
    if "producer_settings" in kinds and (
        metadata["final_media_producer_contract_version"] != 4
        or metadata["final_media_publication_generation"]
        != index.publication_generation
    ):
        raise ReservationConflictError(
            "completed retirement lacks producer-settings successor evidence"
        )
    if "cleanup_pending_settings" in kinds and metadata["cleanup_status"] in {
        "not_started",
        "not_requested",
        "pending",
    }:
        raise ReservationConflictError(
            "completed retirement lacks cleanup-settings successor evidence"
        )
    marker_entries = [
        entry for entry in index.extent_entries if entry["reservation_kind"] == "prune_marker"
    ]
    diagnostics_entries = [
        entry
        for entry in index.extent_entries
        if entry["reservation_kind"] == "diagnostics_payload_pruned"
    ]
    if marker_entries or diagnostics_entries:
        diagnostics, _raw = _load_diagnostics_metadata(root)
        if diagnostics.get("status") != "payload_pruned":
            raise ReservationConflictError(
                "completed retirement lacks payload-pruned successor evidence"
            )
        expected_prune_entries = [
            _marker_payload_from_entry(entry)[0] for entry in marker_entries
        ]
        if diagnostics.get("prune_entries") != expected_prune_entries:
            raise ReservationConflictError(
                "completed retirement prune authorization differs from the index"
            )
        final_manifest = root / FINAL_VIDEO_MANIFEST_RELATIVE_PATH
        final_manifest_sha256, _count = _hash_file(final_manifest)
        if diagnostics.get("final_video_manifest_sha256") != final_manifest_sha256:
            raise ReservationConflictError(
                "completed retirement diagnostics bind another final manifest"
            )


def _descriptorless_failed_entry_is_committed_v1(
    root: Path,
    index: FinalEncodingReservationIndexV1,
    entry: dict,
) -> bool:
    kind = entry["reservation_kind"]
    target = root / entry["target_relative_path"]
    if not _regular_nonlink(target):
        return False
    if kind == "encoding_input_manifest":
        try:
            _stream_input_manifest_identity(target)
        except (FinalPublicationValidationError, OSError):
            return False
        return True
    if kind == "final_video_manifest":
        try:
            _validate_committed_publication_triad_v1(root, index)
        except ReservationConflictError:
            return False
        return True
    if kind in {"producer_settings", "cleanup_pending_settings"}:
        try:
            settings = ProcessingSettingsArtifactV5.parse(
                _read_bounded_regular_file(
                    root / "processing-settings-v5.json",
                    max_bytes=512 * 1024,
                )
            )
        except (OSError, SettingsArtifactSchemaError):
            return False
        metadata = settings.to_object()["metadata"]
        if kind == "producer_settings":
            return (
                metadata["final_media_producer_contract_version"] == 4
                and metadata["final_media_publication_generation"]
                == index.publication_generation
            )
        return metadata["cleanup_status"] not in {
            "not_started",
            "not_requested",
        }
    if kind in {"prune_marker", "diagnostics_payload_pruned"}:
        try:
            diagnostics, _raw = _load_diagnostics_metadata(root)
        except (OSError, ReservationConflictError):
            return False
        if diagnostics.get("status") != "payload_pruned":
            return False
        if kind == "diagnostics_payload_pruned":
            return True
        expected, _payload = _marker_payload_from_entry(entry)
        return expected in diagnostics.get("prune_entries", [])
    return False


def load_final_encoding_reservation_v1(
    job_root: Path,
    settings: ProcessingSettingsArtifactV5,
    *,
    allocation_adapter: AllocationAdapter = platform_allocation_adapter,
    allocation_counter: AllocationCounter | None = platform_allocation_counter,
    event: Callable[[str], None] | None = None,
) -> PreparedFinalEncodingReservationV1:
    """Reopen the durable index and derive its current physical publication phase."""

    root = Path(job_root).resolve()
    index = recover_final_encoding_index_pair_v1(root)
    if index is None:
        raise ReservationConflictError("final encoding reservation is missing")
    metadata = settings.to_object()["metadata"]
    if (
        index.job_id != metadata["job_id"]
        or index.settings_identity_fingerprint != metadata["settings_identity_fingerprint"]
        or metadata["final_media_publication_generation"]
        not in {None, index.publication_generation}
        or index.final_video_relative_path
        != settings.to_object()["output_info"]["expected_final_relative_path"]
        or index.sibling_video_parent_identity != _reservation_directory_identity(root)
    ):
        raise ReservationConflictError("active final index belongs to different job settings")
    producer_entries = [
        entry
        for entry in index.extent_entries
        if entry.get("reservation_kind") == "producer_settings"
    ]
    marker_entries = [
        entry for entry in index.extent_entries if entry.get("reservation_kind") == "prune_marker"
    ]
    diagnostics_entries = [
        entry
        for entry in index.extent_entries
        if entry.get("reservation_kind") == "diagnostics_payload_pruned"
    ]
    cleanup_entries = [
        entry
        for entry in index.extent_entries
        if entry.get("reservation_kind") == "cleanup_pending_settings"
    ]
    if len(producer_entries) > 1 or len(diagnostics_entries) > 1 or len(cleanup_entries) > 1:
        raise ReservationConflictError("final index has duplicate singleton extents")
    producer_consumed = (
        metadata["final_media_publication_generation"] == index.publication_generation
    )
    cleanup_consumed = metadata["cleanup_status"] not in {"not_started", "not_requested"}
    diagnostics_value: dict | None = None
    diagnostics_raw: bytes | None = None
    if marker_entries or diagnostics_entries:
        diagnostics_value, diagnostics_raw = _load_diagnostics_metadata(root)
    diagnostics_consumed = bool(
        diagnostics_value is not None and diagnostics_value.get("status") == "payload_pruned"
    )
    target_path = root / index.final_video_target_descriptor_relative_path
    if target_path.exists():
        target = FinalVideoTargetReservationV1.from_object(
            _load_reservation_json(target_path, max_bytes=1024 * 1024)
        )
    else:
        final_path = root / index.final_video_relative_path
        if index.final_video_publication_method == "replace_placeholder":
            if not (final_path.exists() or final_path.is_symlink()):
                with final_path.open("xb") as handle:
                    handle.flush()
                    os.fsync(handle.fileno())
                _sync_parent(final_path.parent)
            if not _regular_nonlink(final_path) or final_path.stat().st_size != 0:
                raise ReservationConflictError(
                    "descriptorless final-video placeholder is not adoptable"
                )
        else:
            expected = index.final_video_existing_target_identity_at_preflight
            if expected is None or _reservation_identity(final_path) != expected:
                raise ReservationConflictError("descriptorless existing final-video target changed")
        reserved_identity = _reservation_identity(final_path)
        target = FinalVideoTargetReservationV1.build(
            job_id=index.job_id,
            settings_identity_fingerprint=index.settings_identity_fingerprint,
            settings_raw_sha256_at_preflight=index.settings_raw_sha256_at_preflight,
            settings_revision_at_preflight=index.settings_revision_at_preflight,
            publication_generation=index.publication_generation,
            reservation_generation=index.reservation_generation,
            transaction_index_fingerprint=index.fingerprint,
            final_video_relative_path=index.final_video_relative_path,
            sibling_video_temporary_relative_path=(index.sibling_video_temporary_relative_path),
            target_parent_identity=index.sibling_video_parent_identity,
            publication_method=index.final_video_publication_method,
            reserved_target_identity=reserved_identity,
        )
        publish_write_ahead_intent(target_path, target.to_object())
    if (
        target.job_id != index.job_id
        or target.settings_identity_fingerprint != index.settings_identity_fingerprint
        or target.settings_raw_sha256_at_preflight
        != index.settings_raw_sha256_at_preflight
        or target.settings_revision_at_preflight != index.settings_revision_at_preflight
        or target.transaction_index_fingerprint != index.fingerprint
        or target.publication_generation != index.publication_generation
        or target.reservation_generation != index.reservation_generation
        or target.final_video_relative_path != index.final_video_relative_path
        or target.sibling_video_temporary_relative_path
        != index.sibling_video_temporary_relative_path
        or target.target_parent_identity != index.sibling_video_parent_identity
        or target.publication_method != index.final_video_publication_method
    ):
        raise ReservationConflictError("final target descriptor differs from its index")
    if target.publication_method == "replace_existing" and (
        target.reserved_target_identity
        != index.final_video_existing_target_identity_at_preflight
    ):
        raise ReservationConflictError("final target descriptor changed the old target identity")
    current_final = root / index.final_video_relative_path
    if (
        target.publication_method == "replace_placeholder"
        and _regular_nonlink(current_final)
        and current_final.stat().st_size == 0
        and _reservation_identity(current_final) != target.reserved_target_identity
    ):
        raise ReservationConflictError("final placeholder identity differs from descriptor")
    producer_transaction = (
        _load_indexed_extent(
            root=root,
            index=index,
            entry=producer_entries[0],
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
            committed=producer_consumed,
        )
        if producer_entries
        else None
    )
    prune_transactions: list[tuple[dict, ReservedArtifactTransaction | None, bytes]] = []
    committed_prune_entries = (
        diagnostics_value.get("prune_entries")
        if diagnostics_consumed and diagnostics_value is not None
        else None
    )
    for marker_index, entry in enumerate(marker_entries):
        prune_entry, payload = _marker_payload_from_entry(entry)
        target = root / entry["target_relative_path"]
        marker_committed = (
            _regular_nonlink(target)
            and _hash_file(target) == (hashlib.sha256(payload).hexdigest(), len(payload))
        ) or (
            isinstance(committed_prune_entries, list)
            and marker_index < len(committed_prune_entries)
            and committed_prune_entries[marker_index] == prune_entry
            and not (target.exists() or target.is_symlink())
        )
        prune_transactions.append(
            (
                prune_entry,
                _load_indexed_extent(
                    root=root,
                    index=index,
                    entry=entry,
                    allocation_adapter=allocation_adapter,
                    allocation_counter=allocation_counter,
                    committed=marker_committed,
                ),
                payload,
            )
        )
    diagnostics_transaction = (
        _load_indexed_extent(
            root=root,
            index=index,
            entry=diagnostics_entries[0],
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
            committed=diagnostics_consumed,
        )
        if diagnostics_entries
        else None
    )
    cleanup_transaction = (
        _load_indexed_extent(
            root=root,
            index=index,
            entry=cleanup_entries[0],
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
            committed=cleanup_consumed,
        )
        if cleanup_entries
        else None
    )
    return PreparedFinalEncodingReservationV1(
        job_root=root,
        index=index,
        target_descriptor=target,
        input_manifest_transaction=_load_indexed_extent(
            root=root,
            index=index,
            entry=_unique_extent_entry(index, "encoding_input_manifest"),
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
        ),
        final_manifest_transaction=_load_indexed_extent(
            root=root,
            index=index,
            entry=_unique_extent_entry(index, "final_video_manifest"),
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
        ),
        event=event,
        producer_settings_transaction=producer_transaction,
        producer_settings_payload=_settings_payload_intent(producer_transaction),
        prune_marker_transactions=tuple(prune_transactions),
        diagnostics_payload_pruned_transaction=diagnostics_transaction,
        cleanup_pending_settings_transaction=cleanup_transaction,
        cleanup_pending_settings_payload=_settings_payload_intent(cleanup_transaction),
    )


def recover_final_encoding_terminal_retirement_v1(  # noqa: C901
    job_root: Path,
    settings: ProcessingSettingsArtifactV5,
    *,
    allocation_adapter: AllocationAdapter = platform_allocation_adapter,
    allocation_counter: AllocationCounter | None = platform_allocation_counter,
    event: Callable[[str], None] | None = None,
    crash_at: str | None = None,
) -> bool:
    """Resume descriptor-first, index-last retirement after terminal settings."""

    root = Path(job_root).resolve()
    index_path = root / FINAL_ENCODING_RESERVATION_RELATIVE_PATH
    if not (index_path.exists() or index_path.is_symlink()):
        return False
    if settings.status not in {"completed", "failed"}:
        return False
    fixed_settings = root / "processing-settings-v5.json"
    if (
        ProcessingSettingsArtifactV5.parse(
            _read_bounded_regular_file(fixed_settings, max_bytes=512 * 1024)
        )
        != settings
    ):
        raise ReservationConflictError("terminal retirement lacks durable settings evidence")
    index = recover_final_encoding_index_pair_v1(root)
    if index is None:
        return False
    target_descriptor_path = root / index.final_video_target_descriptor_relative_path
    if settings.status == "failed":
        _retire_committed_manifest_descriptors_from_index_v1(
            root,
            index,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
        )
        _recover_failed_final_retirement_v1(
            root,
            index,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
            event=event,
            crash_at=crash_at,
        )
        return True
    if target_descriptor_path.exists() or target_descriptor_path.is_symlink():
        reservation = load_final_encoding_reservation_v1(
            root,
            settings,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
        )
        if settings.status == "completed":
            if reservation.observe().phase != "publication_authenticated":
                raise ReservationConflictError(
                    "completed settings do not have authenticated final publication"
                )
            reservation.retire()
        else:
            reservation.retire_terminal_failure()
        return True

    sibling = root / index.sibling_video_temporary_relative_path
    if sibling.exists() or sibling.is_symlink():
        raise ReservationConflictError("retirement prefix retains the indexed sibling")
    descriptor_temporary = target_descriptor_path.with_name(
        f".{target_descriptor_path.name}.create-new.tmp"
    )
    if descriptor_temporary.exists() or descriptor_temporary.is_symlink():
        raise ReservationConflictError("retirement prefix retains target descriptor temporary")
    for entry in index.extent_entries:
        if entry.get("reservation_kind") == "job_terminal_settings":
            continue
        descriptor, source = _indexed_extent_paths(root, entry)
        descriptor_temporary = descriptor.with_name(f".{descriptor.name}.create-new.tmp")
        if any(
            path.exists() or path.is_symlink()
            for path in (descriptor, descriptor_temporary, source)
        ):
            raise ReservationConflictError(
                "target-descriptor retirement preceded an extent cleanup"
            )
    _validate_completed_descriptorless_retirement_v1(root, index, settings)
    index_identity = _reservation_identity(index_path)
    _unlink_safe_control_temporary(
        index_path,
        expected_identity=index_identity,
        expected_parent_identity=_reservation_directory_identity(index_path.parent),
    )
    _sync_parent(index_path.parent)
    return True


def _retire_committed_manifest_descriptors_from_index_v1(
    root: Path,
    index: FinalEncodingReservationIndexV1,
    *,
    allocation_adapter: AllocationAdapter,
    allocation_counter: AllocationCounter | None,
) -> None:
    for kind in ("encoding_input_manifest", "final_video_manifest"):
        entry = _unique_extent_entry(index, kind)
        descriptor_path, source_path = _indexed_extent_paths(root, entry)
        if not descriptor_path.exists() or source_path.exists() or source_path.is_symlink():
            continue
        transaction = _load_indexed_extent(
            root=root,
            index=index,
            entry=entry,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
        )
        if transaction is None:
            raise ReservationConflictError("committed manifest descriptor disappeared")
        if kind == "encoding_input_manifest":
            source = EncodingInputPublicationSource.from_path(
                root / ENCODING_INPUT_MANIFEST_RELATIVE_PATH
            )
            transaction.recover_streamed_prefix(
                prefix_byte_count=source.raw_byte_count,
                prefix_sha256=source.raw_sha256,
            )
            continue
        target = root / FINAL_VIDEO_MANIFEST_RELATIVE_PATH
        raw = _read_bounded_regular_file(
            target,
            max_bytes=transaction.descriptor.payload_raw_max,
        )
        _validate_committed_publication_triad_v1(root, index)
        transaction.recover(raw)


def _recover_failed_final_retirement_v1(  # noqa: C901
    root: Path,
    index: FinalEncodingReservationIndexV1,
    *,
    allocation_adapter: AllocationAdapter,
    allocation_counter: AllocationCounter | None,
    event: Callable[[str], None] | None,
    crash_at: str | None,
) -> None:
    """Continue terminal-failure cleanup monotonically without materializing work."""

    control = load_job_control(root)

    def emit(name: str) -> None:
        if event is not None:
            event(name)
        if crash_at == name:
            raise InjectedFinalPublicationCrash(name)

    sibling = root / index.sibling_video_temporary_relative_path
    if sibling.exists() or sibling.is_symlink():
        _reservation_identity(sibling)
        _unlink_safe_control_temporary(sibling)
        _sync_parent(sibling.parent)
    emit("failure-retirement:sibling")

    for position, entry in enumerate(index.extent_entries):
        if entry["reservation_kind"] == "job_terminal_settings":
            continue
        descriptor_path, source_path = _indexed_extent_paths(root, entry)

        def parse_descriptor(raw: bytes) -> ReservationDescriptor:
            value = strict_canonical_json_loads(raw, max_bytes=1024 * 1024)
            if not isinstance(value, dict):
                raise ReservationConflictError("reservation descriptor is not an object")
            return ReservationDescriptor.from_object(value)

        descriptor_temporary = descriptor_path.with_name(
            f".{descriptor_path.name}.create-new.tmp"
        )
        if not (descriptor_path.exists() or descriptor_path.is_symlink()) and (
            descriptor_temporary.exists() or descriptor_temporary.is_symlink()
        ):
            if source_path.exists() or source_path.is_symlink():
                raise ReservationConflictError(
                    "incomplete descriptor temporary has a materialized source"
                )
            _unlink_safe_control_temporary(descriptor_temporary)
            _sync_parent(descriptor_temporary.parent)
            recovered = None
        else:
            recovered = recover_control_artifact_pair_v1(
                descriptor_path,
                max_bytes=1024 * 1024,
                parser=parse_descriptor,
            )
        descriptor = recovered if isinstance(recovered, ReservationDescriptor) else None
        if descriptor is not None:
            validate_descriptor_against_entry_v1(
                descriptor,
                entry,
                job_id=index.job_id,
                settings_identity_fingerprint=index.settings_identity_fingerprint,
                settings_revision_at_creation=index.settings_revision_at_preflight,
                settings_raw_sha256_at_creation=index.settings_raw_sha256_at_preflight,
                transaction_index_relative_path=FINAL_ENCODING_RESERVATION_RELATIVE_PATH,
                transaction_index_fingerprint=index.fingerprint,
            )
        if source_path.exists() or source_path.is_symlink():
            if descriptor is None:
                raise ReservationConflictError(
                    "failed retirement found a source without its descriptor"
                )
            if _reservation_identity(source_path) != descriptor.filesystem_identity:
                raise ReservationConflictError("unused reservation source identity changed")
            _unlink_safe_control_temporary(
                source_path,
                expected_identity=descriptor.filesystem_identity,
            )
            _sync_parent(source_path.parent)
        emit(f"failure-retirement:{position}:source")
        if descriptor is not None and descriptor.publication_method == "replace_placeholder":
            target = root / descriptor.target_relative_path
            if target.exists() or target.is_symlink():
                if (
                    descriptor.placeholder_identity is None
                    or _reservation_identity(target) != descriptor.placeholder_identity
                    or target.stat().st_size != 0
                ):
                    raise ReservationConflictError("unused reservation placeholder changed")
                _unlink_safe_control_temporary(
                    target,
                    expected_identity=descriptor.placeholder_identity,
                    expected_parent_identity=descriptor.target_parent_identity,
                )
                _sync_parent(target.parent)
        elif descriptor is None:
            target = root / entry["target_relative_path"]
            if target.exists() or target.is_symlink():
                target_is_preflight = False
                if entry["publication_method"] == "replace_existing":
                    expected = entry["existing_target_identity_at_preflight"]
                    target_is_preflight = bool(
                        expected is not None
                        and _regular_nonlink(target)
                        and _reservation_identity(target)
                        == ReservationFileIdentity.from_object(expected)
                        and _hash_file(target)
                        == (
                            entry["existing_target_raw_sha256_at_preflight"],
                            entry["existing_target_byte_count_at_preflight"],
                        )
                    )
                if not target_is_preflight and not _descriptorless_failed_entry_is_committed_v1(
                    root,
                    index,
                    entry,
                ):
                    raise ReservationConflictError(
                        "descriptorless extent target has no retirement evidence"
                    )
            elif entry["publication_method"] == "replace_existing":
                raise ReservationConflictError(
                    "descriptorless existing extent target is missing"
                )
        emit(f"failure-retirement:{position}:placeholder")
        if descriptor_path.exists() or descriptor_path.is_symlink():
            _unlink_safe_control_temporary(descriptor_path)
            _sync_parent(descriptor_path.parent)
        emit(f"failure-retirement:{position}:descriptor")

    target_descriptor_path = root / index.final_video_target_descriptor_relative_path

    def parse_target(raw: bytes) -> FinalVideoTargetReservationV1:
        return FinalVideoTargetReservationV1.from_object(
            strict_canonical_json_loads(raw, max_bytes=1024 * 1024)
        )

    target_descriptor_temporary = target_descriptor_path.with_name(
        f".{target_descriptor_path.name}.create-new.tmp"
    )
    if not (target_descriptor_path.exists() or target_descriptor_path.is_symlink()) and (
        target_descriptor_temporary.exists() or target_descriptor_temporary.is_symlink()
    ):
        _unlink_safe_control_temporary(target_descriptor_temporary)
        _sync_parent(target_descriptor_temporary.parent)
        recovered_target = None
    else:
        recovered_target = recover_control_artifact_pair_v1(
            target_descriptor_path,
            max_bytes=1024 * 1024,
            parser=parse_target,
        )
    if recovered_target is not None and not isinstance(
        recovered_target, FinalVideoTargetReservationV1
    ):
        raise ReservationConflictError("final target descriptor type changed")
    if isinstance(recovered_target, FinalVideoTargetReservationV1):
        if (
            recovered_target.transaction_index_fingerprint != index.fingerprint
            or recovered_target.job_id != control.job_id
            or recovered_target.settings_identity_fingerprint
            != control.settings_identity_fingerprint
            or recovered_target.settings_raw_sha256_at_preflight
            != index.settings_raw_sha256_at_preflight
            or recovered_target.settings_revision_at_preflight
            != index.settings_revision_at_preflight
            or recovered_target.publication_generation != index.publication_generation
            or recovered_target.reservation_generation != index.reservation_generation
            or recovered_target.final_video_relative_path != index.final_video_relative_path
            or recovered_target.sibling_video_temporary_relative_path
            != index.sibling_video_temporary_relative_path
            or recovered_target.target_parent_identity != index.sibling_video_parent_identity
            or recovered_target.publication_method != index.final_video_publication_method
        ):
            raise ReservationConflictError("final target descriptor differs from failed index")
        if recovered_target.publication_method == "replace_existing" and (
            recovered_target.reserved_target_identity
            != index.final_video_existing_target_identity_at_preflight
        ):
            raise ReservationConflictError("failed target descriptor changed old identity")
        final = root / index.final_video_relative_path
        if final.exists() or final.is_symlink():
            current = _reservation_identity(final)
            if (
                current == recovered_target.reserved_target_identity
                and recovered_target.publication_method == "replace_placeholder"
            ):
                if final.stat().st_size != 0:
                    raise ReservationConflictError("unused final placeholder is nonempty")
                _unlink_safe_control_temporary(
                    final,
                    expected_identity=recovered_target.reserved_target_identity,
                    expected_parent_identity=recovered_target.target_parent_identity,
                )
                _sync_parent(final.parent)
        if target_descriptor_path.exists() or target_descriptor_path.is_symlink():
            _unlink_safe_control_temporary(target_descriptor_path)
            _sync_parent(target_descriptor_path.parent)
    else:
        final = root / index.final_video_relative_path
        if final.exists() or final.is_symlink():
            if not _regular_nonlink(final):
                raise ReservationConflictError(
                    "descriptorless final target is unsafe"
                )
            if (
                index.final_video_publication_method == "replace_placeholder"
                and final.stat().st_size == 0
            ):
                raise ReservationConflictError(
                    "descriptorless zero final placeholder has no retirement evidence"
                )
        elif index.final_video_publication_method == "replace_existing":
            raise ReservationConflictError(
                "descriptorless existing final target is missing"
            )
    emit("failure-retirement:target-descriptor")
    index_path = root / FINAL_ENCODING_RESERVATION_RELATIVE_PATH
    if index_path.exists() or index_path.is_symlink():
        index_identity = _reservation_identity(index_path)
        _unlink_safe_control_temporary(
            index_path,
            expected_identity=index_identity,
            expected_parent_identity=_reservation_directory_identity(index_path.parent),
        )
        _sync_parent(index_path.parent)
    emit("failure-retirement:index")


def _sync_parent(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb", buffering=0) as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
    return digest.hexdigest(), byte_count


_INPUT_MANIFEST_PREFIX = b'{"algorithm_version":"encoding-input-sequence-v2","fingerprint":"'
_INPUT_MANIFEST_SELF_PREFIX = b'{"algorithm_version":"encoding-input-sequence-v2"'


def _stream_input_manifest_identity(  # noqa: C901
    path: Path,
    *,
    raw_byte_count: int | None = None,
    logical_byte_count: int | None = None,
) -> tuple[str, str, int]:
    """Validate the fixed self-fingerprint prefix while retaining no manifest tree."""

    digest = hashlib.sha256()
    self_digest = hashlib.sha256()
    byte_count = 0
    header_size = len(_INPUT_MANIFEST_PREFIX) + 64 + 1
    with path.open("rb", buffering=0) as handle:
        header = handle.read(header_size)
        if (
            len(header) != header_size
            or not header.startswith(_INPUT_MANIFEST_PREFIX)
            or header[-1:] != b'"'
        ):
            raise FinalPublicationValidationError("encoding input manifest prefix is malformed")
        fingerprint_raw = header[len(_INPUT_MANIFEST_PREFIX) : -1]
        try:
            fingerprint = fingerprint_raw.decode("ascii")
        except UnicodeDecodeError as error:
            raise FinalPublicationValidationError(
                "encoding input fingerprint is not ASCII"
            ) from error
        if _HEX64.fullmatch(fingerprint) is None:
            raise FinalPublicationValidationError("encoding input fingerprint is invalid")
        digest.update(header)
        byte_count += len(header)
        self_digest.update(_INPUT_MANIFEST_SELF_PREFIX)
        remaining = None if raw_byte_count is None else raw_byte_count - len(header)
        if remaining is not None and remaining < 0:
            raise FinalPublicationValidationError("encoding input raw byte count is too small")
        while remaining is None or remaining:
            chunk = handle.read(1024 * 1024 if remaining is None else min(1024 * 1024, remaining))
            if not chunk:
                if remaining:
                    raise FinalPublicationValidationError(
                        "encoding input manifest ended before its raw byte count"
                    )
                break
            digest.update(chunk)
            self_digest.update(chunk)
            byte_count += len(chunk)
            if remaining is not None:
                remaining -= len(chunk)
        if logical_byte_count is not None:
            if logical_byte_count < byte_count:
                raise FinalPublicationValidationError(
                    "encoding input logical byte count is too small"
                )
            tail = logical_byte_count - byte_count
            while tail:
                chunk = handle.read(min(1024 * 1024, tail))
                if not chunk or any(chunk):
                    raise FinalPublicationValidationError(
                        "encoding input reservation tail is not all zero"
                    )
                tail -= len(chunk)
            if handle.read(1):
                raise FinalPublicationValidationError(
                    "encoding input reservation exceeds its logical byte count"
                )
    if self_digest.hexdigest() != fingerprint:
        raise FinalPublicationValidationError("encoding input self-fingerprint differs")
    return fingerprint, digest.hexdigest(), byte_count


def _regular_nonlink(path: Path) -> bool:
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISREG(metadata.st_mode) and not path.is_symlink()


def _atomic_replace(path: Path, raw: bytes) -> None:
    temporary = path.with_name(f".{path.name}.publication-v4.tmp")
    with temporary.open("w+b") as handle:
        if handle.write(raw) != len(raw):
            raise OSError("short final publication manifest write")
        handle.flush()
        os.fsync(handle.fileno())
    if temporary.read_bytes() != raw:
        raise FinalPublicationValidationError("manifest temporary failed reopen validation")
    os.replace(temporary, path)
    _sync_parent(path.parent)
    if path.read_bytes() != raw:
        raise FinalPublicationValidationError("published manifest failed reopen validation")


def _path64(relative_path: str) -> str:
    return base64.urlsafe_b64encode(relative_path.encode("utf-8")).rstrip(b"=").decode("ascii")


def _decode_path64(token: str) -> str:
    if not token or re.fullmatch(r"[A-Za-z0-9_-]+", token, flags=re.ASCII) is None:
        raise FinalPublicationValidationError("output path64 token is malformed")
    padding = "=" * ((4 - len(token) % 4) % 4)
    try:
        decoded = base64.b64decode(
            token + padding,
            altchars=b"-_",
            validate=True,
        ).decode("utf-8", errors="strict")
    except (ValueError, UnicodeDecodeError) as error:
        raise FinalPublicationValidationError("output path64 token cannot be decoded") from error
    if _path64(decoded) != token:
        raise FinalPublicationValidationError("output path64 token is noncanonical")
    return decoded


def _validate_normalized_argv(  # noqa: C901
    arguments: tuple[str, ...],
    *,
    input_stage_fingerprint: str,
    relative_path: str,
) -> None:
    if not arguments or arguments[0] != "ffmpeg":
        raise FinalPublicationValidationError("normalized argv must start with ffmpeg")
    output_count = 0
    image_views: set[str] = set()
    for argument in arguments:
        if type(argument) is not str:
            raise FinalPublicationValidationError("normalized argv members must be strings")
        try:
            raw = argument.encode("ascii")
        except UnicodeEncodeError as error:
            raise FinalPublicationValidationError("normalized argv must be ASCII") from error
        if any(byte < 0x20 or byte > 0x7E for byte in raw):
            raise FinalPublicationValidationError("normalized argv has a control byte")
        if argument.startswith("@image-sequence:"):
            match = _IMAGE_TOKEN.fullmatch(argument)
            if match is None or match.group(1) != input_stage_fingerprint:
                raise FinalPublicationValidationError("normalized image token is invalid")
            if match.group(2) in image_views:
                raise FinalPublicationValidationError("normalized image view is duplicated")
            image_views.add(match.group(2))
        elif argument.startswith("@audio:"):
            if _AUDIO_TOKEN.fullmatch(argument) is None:
                raise FinalPublicationValidationError("normalized audio token is invalid")
        elif argument.startswith("@output:"):
            match = _OUTPUT_TOKEN.fullmatch(argument)
            if match is None or _decode_path64(match.group(1)) != relative_path:
                raise FinalPublicationValidationError("normalized output token is invalid")
            output_count += 1
    if output_count != 1:
        raise FinalPublicationValidationError("normalized argv requires one output token")


@dataclass(frozen=True)
class FinalVideoManifestV4:
    relative_path: str
    sha256: str
    byte_count: int
    input_stage_fingerprint: str
    resolved_encoding_arguments: tuple[str, ...]
    encoding_settings_fingerprint: str
    fingerprint: str

    def _base_object(self) -> dict:
        return {
            "schema_version": 1,
            "algorithm_version": "final-video-publication-v4",
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "input_stage_fingerprint": self.input_stage_fingerprint,
            "resolved_encoding_arguments": list(self.resolved_encoding_arguments),
            "encoding_settings_fingerprint": self.encoding_settings_fingerprint,
        }

    def to_object(self) -> dict:
        return {**self._base_object(), "fingerprint": self.fingerprint}

    def encode(self) -> bytes:
        return canonical_json_bytes(self.to_object())


def build_final_video_manifest_v4(
    *,
    relative_path: str,
    video_sha256: str,
    video_byte_count: int,
    input_stage_fingerprint: str,
    resolved_encoding_arguments: tuple[str, ...],
) -> FinalVideoManifestV4:
    try:
        validate_contained_legacy_final_component_v1(relative_path)
    except SettingsArtifactSchemaError as error:
        raise FinalPublicationValidationError("final relative path is not contained") from error
    if not relative_path.endswith(".mp4"):
        raise FinalPublicationValidationError("final relative path must end in lowercase .mp4")
    if _HEX64.fullmatch(video_sha256) is None or _HEX64.fullmatch(input_stage_fingerprint) is None:
        raise FinalPublicationValidationError("final publication hash is invalid")
    if type(video_byte_count) is not int or video_byte_count <= 0:
        raise FinalPublicationValidationError("final video byte count must be positive")
    arguments = tuple(resolved_encoding_arguments)
    _validate_normalized_argv(
        arguments,
        input_stage_fingerprint=input_stage_fingerprint,
        relative_path=relative_path,
    )
    settings_object = {
        "input_stage_fingerprint": input_stage_fingerprint,
        "resolved_encoding_arguments": list(arguments),
    }
    settings_fingerprint = hashlib.sha256(canonical_json_bytes(settings_object)).hexdigest()
    provisional = FinalVideoManifestV4(
        relative_path=relative_path,
        sha256=video_sha256,
        byte_count=video_byte_count,
        input_stage_fingerprint=input_stage_fingerprint,
        resolved_encoding_arguments=arguments,
        encoding_settings_fingerprint=settings_fingerprint,
        fingerprint="",
    )
    fingerprint = hashlib.sha256(canonical_json_bytes(provisional._base_object())).hexdigest()
    return FinalVideoManifestV4(**{**provisional.__dict__, "fingerprint": fingerprint})


def parse_final_video_manifest_v4(raw: bytes) -> FinalVideoManifestV4:
    try:
        value = strict_canonical_json_loads(raw, max_bytes=1024 * 1024)
    except CanonicalJsonError as error:
        raise FinalPublicationValidationError("final manifest is malformed") from error
    expected = {
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
    if type(value) is not dict or set(value) != expected:
        raise FinalPublicationValidationError("final manifest schema is not closed")
    assert isinstance(value, dict)
    if (
        value["schema_version"] != 1
        or type(value["schema_version"]) is not int
        or value["algorithm_version"] != "final-video-publication-v4"
        or type(value["resolved_encoding_arguments"]) is not list
    ):
        raise FinalPublicationValidationError("final manifest version is invalid")
    manifest = build_final_video_manifest_v4(
        relative_path=value["relative_path"],
        video_sha256=value["sha256"],
        video_byte_count=value["byte_count"],
        input_stage_fingerprint=value["input_stage_fingerprint"],
        resolved_encoding_arguments=tuple(value["resolved_encoding_arguments"]),
    )
    if (
        manifest.encoding_settings_fingerprint != value["encoding_settings_fingerprint"]
        or manifest.fingerprint != value["fingerprint"]
    ):
        raise FinalPublicationValidationError("final manifest fingerprints differ")
    return manifest


def _parse_input_manifest_fingerprint(raw: bytes) -> str:
    try:
        value = strict_canonical_json_loads(raw, max_bytes=512 * 1024 * 1024)
    except CanonicalJsonError as error:
        raise FinalPublicationValidationError("encoding input manifest is malformed") from error
    if type(value) is not dict:
        raise FinalPublicationValidationError("encoding input manifest root is invalid")
    algorithm = value.get("algorithm_version")
    fingerprint = value.get("fingerprint")
    if algorithm != "encoding-input-sequence-v2" or not isinstance(fingerprint, str):
        raise FinalPublicationValidationError("encoding input manifest identity is invalid")
    if _HEX64.fullmatch(fingerprint) is None:
        raise FinalPublicationValidationError("encoding input fingerprint is invalid")
    prefix_end = len(_INPUT_MANIFEST_PREFIX) + 64 + 1
    if (
        len(raw) < prefix_end
        or not raw.startswith(_INPUT_MANIFEST_PREFIX)
        or raw[prefix_end - 1 : prefix_end] != b'"'
    ):
        raise FinalPublicationValidationError("encoding input manifest prefix differs")
    self_digest = hashlib.sha256()
    self_digest.update(_INPUT_MANIFEST_SELF_PREFIX)
    self_digest.update(raw[prefix_end:])
    if self_digest.hexdigest() != fingerprint:
        raise FinalPublicationValidationError("encoding input self-fingerprint differs")
    return fingerprint


@dataclass(frozen=True)
class EncodingInputPublicationSource:
    """Bounded staged-manifest evidence used by the production publication path."""

    path: Path
    fingerprint: str
    raw_sha256: str
    raw_byte_count: int
    logical_byte_count: int | None = None

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        raw_byte_count: int | None = None,
        logical_byte_count: int | None = None,
    ) -> "EncodingInputPublicationSource":
        candidate = Path(path)
        fingerprint, raw_sha256, measured = _stream_input_manifest_identity(
            candidate,
            raw_byte_count=raw_byte_count,
            logical_byte_count=logical_byte_count,
        )
        return cls(
            path=candidate,
            fingerprint=fingerprint,
            raw_sha256=raw_sha256,
            raw_byte_count=measured,
            logical_byte_count=logical_byte_count,
        )

    def validated(self) -> "EncodingInputPublicationSource":
        path = Path(self.path)
        if not _regular_nonlink(path):
            raise FinalPublicationValidationError(
                "staged encoding input manifest is not an ordinary file"
            )
        if self.logical_byte_count is not None and path.stat().st_size != self.logical_byte_count:
            raise FinalPublicationValidationError(
                "staged encoding input manifest logical length differs"
            )
        fingerprint, raw_sha256, raw_byte_count = _stream_input_manifest_identity(
            path,
            raw_byte_count=self.raw_byte_count if self.logical_byte_count is not None else None,
            logical_byte_count=self.logical_byte_count,
        )
        if (
            fingerprint != self.fingerprint
            or raw_sha256 != self.raw_sha256
            or raw_byte_count != self.raw_byte_count
        ):
            raise FinalPublicationValidationError("staged encoding input manifest evidence differs")
        return self


class FinalPublicationTransactionV4:
    """Publish the validated video, input manifest, and final manifest in order."""

    def __init__(
        self,
        *,
        job_root: Path,
        settings: ProcessingSettingsArtifactV5,
        sibling_relative_path: str,
        encoding_input_manifest_raw: bytes | None = None,
        encoding_input_source: EncodingInputPublicationSource | None = None,
        input_stage_fingerprint: str,
        resolved_encoding_arguments: tuple[str, ...],
        container_validator: Callable[[Path], bool],
        reservation: PreparedFinalEncodingReservationV1 | None = None,
        retire_reservation: bool = True,
        event: Callable[[str], None] | None = None,
    ) -> None:
        self.job_root = Path(job_root)
        self.settings = settings
        self.sibling_relative_path = validate_contained_legacy_final_component_v1(
            sibling_relative_path
        )
        if (encoding_input_manifest_raw is None) == (encoding_input_source is None):
            raise ValueError("exactly one encoding input manifest source is required")
        self.encoding_input_manifest_raw = (
            bytes(encoding_input_manifest_raw) if encoding_input_manifest_raw is not None else None
        )
        self.encoding_input_source = encoding_input_source
        self.input_stage_fingerprint = input_stage_fingerprint
        self.resolved_encoding_arguments = tuple(resolved_encoding_arguments)
        self.container_validator = container_validator
        self.reservation = reservation
        self.retire_reservation = retire_reservation
        self.event = event
        if reservation is not None:
            if reservation.job_root != self.job_root.resolve():
                raise ValueError("final reservation belongs to a different job root")
            if reservation.sibling_path.name != self.sibling_relative_path:
                raise ValueError("runtime sibling differs from final reservation")

    @staticmethod
    def path64(relative_path: str) -> str:
        return _path64(relative_path)

    def _emit(self, value: str) -> None:
        if self.event is not None:
            self.event(value)

    def publish(self, *, crash_at: str | None = None) -> FinalVideoManifestV4:  # noqa: C901
        settings_value = self.settings.to_object()
        metadata = settings_value["metadata"]
        if (
            metadata["final_media_producer_contract_version"] != 4
            or metadata["final_media_publication_generation"] is None
        ):
            raise FinalPublicationValidationError("publication-v4 producer marker is absent")
        relative_path = settings_value["output_info"]["expected_final_relative_path"]
        try:
            validate_portable_final_component_v1(relative_path)
        except SettingsArtifactSchemaError as error:
            raise FinalPublicationValidationError("final target is not portable") from error
        if self.encoding_input_source is not None:
            source = self.encoding_input_source.validated()
            manifest_fingerprint = source.fingerprint
        else:
            assert self.encoding_input_manifest_raw is not None
            manifest_fingerprint = _parse_input_manifest_fingerprint(
                self.encoding_input_manifest_raw
            )
        if manifest_fingerprint != self.input_stage_fingerprint:
            raise FinalPublicationValidationError("input manifest fingerprint differs")
        _validate_normalized_argv(
            self.resolved_encoding_arguments,
            input_stage_fingerprint=self.input_stage_fingerprint,
            relative_path=relative_path,
        )
        sibling = self.job_root / self.sibling_relative_path
        final_path = self.job_root / relative_path
        reservation_phase = "standalone"
        if self.reservation is not None:
            recovery = self.reservation.observe()
            reservation_phase = recovery.phase
            if reservation_phase == "ready_pre_video":
                self.reservation.validate_reserved_target()
            elif reservation_phase not in {
                "input_manifest_committed_no_final_intent",
                "post_boundary_retry_sibling_present",
            }:
                raise FinalPublicationValidationError(
                    f"reservation cannot accept an encoded sibling in phase {reservation_phase}"
                )
        if not _regular_nonlink(sibling) or not self.container_validator(sibling):
            raise FinalPublicationValidationError("sibling video failed container validation")
        with sibling.open("r+b", buffering=0) as handle:
            handle.flush()
            os.fsync(handle.fileno())
        video_sha256, video_byte_count = _hash_file(sibling)
        if self.reservation is not None:
            expected_target_identity = (
                self.reservation.target_descriptor.reserved_target_identity
                if reservation_phase == "ready_pre_video"
                else _reservation_identity(final_path)
            )
            same_parent_replace_v1(
                sibling,
                final_path,
                expected_source_identity=_reservation_identity(sibling),
                expected_target_identity=expected_target_identity,
                expected_parent_identity=(
                    self.reservation.target_descriptor.target_parent_identity
                ),
                replace_existing=True,
            )
        else:
            os.replace(sibling, final_path)
            _sync_parent(final_path.parent)
        if not _regular_nonlink(final_path):
            raise FinalPublicationValidationError("final video target failed reopen")
        reopened_hash, reopened_count = _hash_file(final_path)
        if (reopened_hash, reopened_count) != (video_sha256, video_byte_count):
            raise FinalPublicationValidationError("final video changed during replacement")
        self._emit("video:published")
        if crash_at == "after-video":
            raise InjectedFinalPublicationCrash(crash_at)

        input_target = self.job_root / ENCODING_INPUT_MANIFEST_RELATIVE_PATH
        if self.reservation is not None:
            input_transaction = self.reservation.input_manifest_transaction
            if input_transaction is None:
                committed = EncodingInputPublicationSource.from_path(input_target)
                if committed.fingerprint != self.input_stage_fingerprint:
                    raise FinalPublicationValidationError(
                        "committed input manifest fingerprint differs"
                    )
            elif self.encoding_input_manifest_raw is not None:
                self.reservation.validate_indexed_predecessor(input_transaction)
                input_transaction.consume(self.encoding_input_manifest_raw)
                self.reservation.input_manifest_transaction = None
            else:
                assert self.encoding_input_source is not None
                source = self.encoding_input_source.validated()
                if source.path != input_transaction.source_path:
                    raise FinalPublicationValidationError(
                        "streamed input source is not the indexed extent"
                    )
                self.reservation.validate_indexed_predecessor(input_transaction)
                input_transaction.consume_streamed_prefix(
                    prefix_byte_count=source.raw_byte_count,
                    prefix_sha256=source.raw_sha256,
                )
                self.reservation.input_manifest_transaction = None
        elif self.encoding_input_source is not None:
            source = self.encoding_input_source.validated()
            os.replace(source.path, input_target)
            _sync_parent(input_target.parent)
            reopened = EncodingInputPublicationSource(
                path=input_target,
                fingerprint=source.fingerprint,
                raw_sha256=source.raw_sha256,
                raw_byte_count=source.raw_byte_count,
            )
            reopened.validated()
        else:
            assert self.encoding_input_manifest_raw is not None
            _atomic_replace(input_target, self.encoding_input_manifest_raw)
        self._emit("input-manifest:published")
        if crash_at == "after-input-manifest":
            raise InjectedFinalPublicationCrash(crash_at)

        final_manifest = build_final_video_manifest_v4(
            relative_path=relative_path,
            video_sha256=video_sha256,
            video_byte_count=video_byte_count,
            input_stage_fingerprint=self.input_stage_fingerprint,
            resolved_encoding_arguments=self.resolved_encoding_arguments,
        )
        if self.reservation is not None:
            final_transaction = self.reservation.final_manifest_transaction
            if final_transaction is None:
                raise FinalPublicationValidationError("final manifest reservation is consumed")
            try:
                self.reservation.validate_indexed_predecessor(final_transaction)
                final_transaction.consume(
                    final_manifest.encode(),
                    crash_at=(
                        "after-complete-full-sync"
                        if crash_at == "after-final-intent-sync"
                        else None
                    ),
                )
            except InjectedReservationCrash as error:
                raise InjectedFinalPublicationCrash("after-final-intent-sync") from error
            self.reservation.final_manifest_transaction = None
        else:
            _atomic_replace(
                self.job_root / FINAL_VIDEO_MANIFEST_RELATIVE_PATH,
                final_manifest.encode(),
            )
        self._emit("final-manifest:published")
        if crash_at == "after-final-manifest":
            raise InjectedFinalPublicationCrash(crash_at)
        if self.reservation is not None and self.retire_reservation:
            self.reservation.retire(event=self.event)
        return final_manifest


AuthenticationReason = Literal[
    "not_present",
    "legacy_no_publication_manifests",
    "incomplete_publication_evidence",
    "publication_validation_failed",
]


@dataclass(frozen=True)
class FinalMediaAuditDispositionV1:
    final_video_present: bool
    final_video_authenticated: bool
    final_video_authentication_reason: AuthenticationReason | None


def audit_final_media_v1(
    job_root: Path,
    settings: ProcessingSettingsArtifactV5,
    *,
    container_validator: Callable[[Path], bool] | None = None,
) -> FinalMediaAuditDispositionV1:
    """Classify the actual V/I/P matrix without inventing prior generations."""

    root = Path(job_root)
    settings_value = settings.to_object()
    expected = settings_value["output_info"]["expected_final_relative_path"]
    final_path = root / expected
    input_path = root / ENCODING_INPUT_MANIFEST_RELATIVE_PATH
    publication_path = root / FINAL_VIDEO_MANIFEST_RELATIVE_PATH
    video_present = final_path.exists() or final_path.is_symlink()
    input_present = input_path.exists() or input_path.is_symlink()
    publication_present = publication_path.exists() or publication_path.is_symlink()
    if not video_present and not input_present and not publication_present:
        return FinalMediaAuditDispositionV1(False, False, "not_present")
    if not video_present or not input_present or not publication_present:
        metadata = settings_value["metadata"]
        legacy = (
            video_present
            and not input_present
            and not publication_present
            and metadata["source_settings_schema_version"] <= 4
            and metadata["processing_status"] == "completed"
            and metadata["final_media_producer_contract_version"] is None
            and metadata["final_media_publication_generation"] is None
            and _regular_nonlink(final_path)
        )
        return FinalMediaAuditDispositionV1(
            video_present,
            False,
            "legacy_no_publication_manifests" if legacy else "incomplete_publication_evidence",
        )
    try:
        if not all(_regular_nonlink(path) for path in (final_path, input_path, publication_path)):
            raise FinalPublicationValidationError("publication entry is unsafe")
        metadata = settings_value["metadata"]
        if (
            metadata["final_media_producer_contract_version"] != 4
            or metadata["final_media_publication_generation"] is None
        ):
            raise FinalPublicationValidationError("producer marker is invalid")
        input_fingerprint, _, _ = _stream_input_manifest_identity(input_path)
        manifest = parse_final_video_manifest_v4(publication_path.read_bytes())
        if (
            manifest.relative_path != expected
            or manifest.input_stage_fingerprint != input_fingerprint
        ):
            raise FinalPublicationValidationError("three-way publication binding differs")
        video_hash, video_count = _hash_file(final_path)
        if manifest.sha256 != video_hash or manifest.byte_count != video_count:
            raise FinalPublicationValidationError("final video identity differs")
        if container_validator is not None and not container_validator(final_path):
            raise FinalPublicationValidationError("final video container validation failed")
    except (OSError, CanonicalJsonError, FinalPublicationError, ValueError):
        return FinalMediaAuditDispositionV1(
            True,
            False,
            "publication_validation_failed",
        )
    return FinalMediaAuditDispositionV1(True, True, None)
