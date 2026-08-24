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
from pathlib import Path
from typing import Callable, Literal

from ...rendering.quality.cleanup_transaction import plan_prune_markers_v1
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
    ReservationFileIdentity,
    ReservedArtifactTransaction,
    allocation_unit_for_path,
    platform_allocation_adapter,
    platform_allocation_counter,
    publish_write_ahead_intent,
    same_parent_replace_v1,
    sha256_canonical,
    load_job_control,
    validate_final_encoding_entry_order,
    classify_final_publication_observation,
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
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink() or metadata.st_nlink != 1:
        raise ReservationConflictError(
            f"reservation target is not a single-link regular file: {path}"
        )
    return ReservationFileIdentity(int(metadata.st_dev), int(metadata.st_ino))


def _reservation_directory_identity(path: Path) -> ReservationFileIdentity:
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise ReservationConflictError(f"reservation parent is not a safe directory: {path}")
    return ReservationFileIdentity(int(metadata.st_dev), int(metadata.st_ino))


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
    sibling_video_parent_identity: ReservationFileIdentity
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

    @classmethod
    def build(cls, **values) -> "FinalEncodingReservationIndexV1":
        provisional = cls(**values, fingerprint="")
        return cls(
            **{
                **provisional.__dict__,
                "fingerprint": sha256_canonical(provisional._base_object()),
            }
        )

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
            sibling_video_parent_identity=ReservationFileIdentity.from_object(
                value["sibling_video_parent_identity"]
            ),
            diagnostics_metadata_raw_sha256_at_preflight=value[
                "diagnostics_metadata_raw_sha256_at_preflight"
            ],
            diagnostics_metadata_fingerprint_at_preflight=value[
                "diagnostics_metadata_fingerprint_at_preflight"
            ],
            extent_entries=tuple(value["extent_entries"]),
            fingerprint=value["fingerprint"],
        )
        if index.fingerprint != sha256_canonical(index._base_object()):
            raise ReservationConflictError("final encoding index fingerprint differs")
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
    target_parent_identity: ReservationFileIdentity
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

    @classmethod
    def build(cls, **values) -> "FinalVideoTargetReservationV1":
        provisional = cls(**values, fingerprint="")
        return cls(
            **{
                **provisional.__dict__,
                "fingerprint": sha256_canonical(provisional._base_object()),
            }
        )

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
            target_parent_identity=ReservationFileIdentity.from_object(
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
        return descriptor


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
            and entry.get("generation") == transaction.descriptor.generation
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
        expected = entry.get("preflight_target_identity")
        expected_count = entry.get("preflight_target_byte_count")
        expected_hash = entry.get("preflight_target_raw_sha256")
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

    def observe(self) -> FinalPublicationRecovery:
        video_state = self._video_state()
        return classify_final_publication_observation(
            FinalPublicationObservation(
                video_state=video_state,
                input_manifest_state=self._input_manifest_state(video_state),
                final_manifest_state=self._final_manifest_state(),
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

        emit = event or self.event
        if self.sibling_path.exists() or self.sibling_path.is_symlink():
            _reservation_identity(self.sibling_path)
            self.sibling_path.unlink()
            _sync_parent(self.sibling_path.parent)
        transactions = [
            self.producer_settings_transaction,
            self.input_manifest_transaction,
            self.final_manifest_transaction,
            *(transaction for _entry, transaction, _payload in self.prune_marker_transactions),
            self.diagnostics_payload_pruned_transaction,
            self.cleanup_pending_settings_transaction,
        ]
        for transaction in transactions:
            if transaction is None or not transaction.descriptor_path.exists():
                continue
            if transaction.source_path.exists():
                if (
                    _reservation_identity(transaction.source_path)
                    != transaction.descriptor.filesystem_identity
                ):
                    raise ReservationConflictError("unused reservation source identity changed")
                transaction.source_path.unlink()
            if transaction.descriptor.publication_method == "replace_placeholder":
                target = transaction.target_path
                if target.exists() or target.is_symlink():
                    if (
                        transaction.descriptor.placeholder_identity is None
                        or _reservation_identity(target)
                        != transaction.descriptor.placeholder_identity
                        or target.stat().st_size != 0
                    ):
                        raise ReservationConflictError("unused reservation placeholder changed")
                    target.unlink()
            _sync_parent(transaction.target_path.parent)
            transaction.descriptor_path.unlink()
        final = self.final_video_path
        if final.exists() or final.is_symlink():
            current = _reservation_identity(final)
            if current == self.target_descriptor.reserved_target_identity:
                if self.target_descriptor.publication_method == "replace_placeholder":
                    if final.stat().st_size != 0:
                        raise ReservationConflictError(
                            "unused final-video placeholder is not empty"
                        )
                    final.unlink()
                    _sync_parent(final.parent)
            # A different identity is post-boundary media and is deliberately retained.
        self.target_descriptor_path.unlink(missing_ok=True)
        _sync_parent(self.target_descriptor_path.parent)
        if emit is not None:
            emit("reservation:failure-target-descriptor-retired")
        self.index_path.unlink(missing_ok=True)
        _sync_parent(self.index_path.parent)
        if emit is not None:
            emit("reservation:failure-index-retired")


def _extent_plan(
    *,
    kind: str,
    target: Path,
    raw_max: int,
    allocation_unit: int,
    generation: str,
    reservation_id: str,
    job_root: Path,
) -> dict:
    target_path = job_root / target
    if target_path.exists() or target_path.is_symlink():
        method = "replace_existing"
        identity = _reservation_identity(target_path)
        target_sha256, target_byte_count = _hash_file(target_path)
    else:
        method = "replace_placeholder"
        identity = None
        target_sha256 = None
        target_byte_count = None
    return {
        "index_relative_path": FINAL_ENCODING_RESERVATION_RELATIVE_PATH.as_posix(),
        "target_relative_path": target.as_posix(),
        "reservation_kind": kind,
        "publication_method": method,
        "payload_raw_max": raw_max,
        "allocation_unit": allocation_unit,
        "generation": generation,
        "reservation_id": reservation_id,
        "preflight_target_identity": identity.to_object() if identity is not None else None,
        "preflight_target_byte_count": target_byte_count,
        "preflight_target_raw_sha256": target_sha256,
    }


def _indexed_extent_paths(root: Path, entry: dict) -> tuple[Path, Path]:
    kind = entry["reservation_kind"]
    descriptor = (
        root
        / ".depth-surge-reservations-v1"
        / f"d-r-{entry['generation']}-{kind}-{entry['reservation_id']}.json"
    )
    target = root / entry["target_relative_path"]
    source = target.parent / (
        f".depth-surge-r-{entry['generation']}-{kind}-{entry['reservation_id']}.reserved"
    )
    return descriptor, source


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
        )
    input_plan = _extent_plan(
        kind="encoding_input_manifest",
        target=ENCODING_INPUT_MANIFEST_RELATIVE_PATH,
        raw_max=input_manifest_raw_max,
        allocation_unit=unit,
        generation=reserve_generation,
        reservation_id=input_id,
        job_root=root,
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
            )
            plan.update(
                {
                    "prune_root_relative_path": marker.relative_path,
                    "marker_name": marker.marker_name,
                    "marker_payload_sha256": marker.payload_sha256,
                    "prune_entry": marker.prune_entry(),
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
        )
    final_plan = _extent_plan(
        kind="final_video_manifest",
        target=FINAL_VIDEO_MANIFEST_RELATIVE_PATH,
        raw_max=final_manifest_raw_max,
        allocation_unit=unit,
        generation=reserve_generation,
        reservation_id=final_id,
        job_root=root,
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
    job_control_path = root / "job-control-v1.json"
    if job_control_path.is_file():
        control = load_job_control(root)
        terminal_entries = [
            dict(entry)
            for entry in control.bootstrap_extent_entries
            if entry.get("reservation_kind") == "job_terminal_settings"
        ]
        if len(terminal_entries) != 1:
            raise ReservationConflictError("job control has no unique terminal extent")
        extent_entries.extend(terminal_entries)
        validate_final_encoding_entry_order([entry["reservation_kind"] for entry in extent_entries])
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
        return strict_canonical_json_loads(path.read_bytes(), max_bytes=max_bytes)
    except (OSError, CanonicalJsonError) as error:
        raise ReservationConflictError(
            f"reservation control artifact is malformed: {path}"
        ) from error


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
            if entry["preflight_target_identity"] is None:
                target_is_preflight = (
                    _regular_nonlink(target_path) and target_path.stat().st_size == 0
                )
            else:
                expected_identity = ReservationFileIdentity.from_object(
                    entry["preflight_target_identity"]
                )
                target_is_preflight = _reservation_identity(
                    target_path
                ) == expected_identity and _hash_file(target_path) == (
                    entry["preflight_target_raw_sha256"],
                    entry["preflight_target_byte_count"],
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
            generation=entry["generation"],
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
    descriptor = transaction.descriptor
    comparisons = {
        "index_relative_path": FINAL_ENCODING_RESERVATION_RELATIVE_PATH.as_posix(),
        "target_relative_path": entry["target_relative_path"],
        "reservation_kind": kind,
        "publication_method": entry["publication_method"],
        "payload_raw_max": entry["payload_raw_max"],
        "allocation_unit": entry["allocation_unit"],
        "generation": entry["generation"],
        "reservation_id": entry["reservation_id"],
    }
    for field, expected in comparisons.items():
        if getattr(descriptor, field) != expected:
            raise ReservationConflictError(f"{kind} descriptor differs from final index")
    return transaction


def _unique_extent_entry(index: FinalEncodingReservationIndexV1, kind: str) -> dict:
    entries = [entry for entry in index.extent_entries if entry.get("reservation_kind") == kind]
    if len(entries) != 1:
        raise ReservationConflictError(f"final index has no unique {kind} entry")
    return entries[0]


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


def _marker_payload_from_entry(entry: dict) -> tuple[dict, bytes]:
    prune_entry = entry.get("prune_entry")
    if not isinstance(prune_entry, dict):
        raise ReservationConflictError("prune marker entry has no durable authorization")
    marker = prune_entry.get("marker")
    relative_path = prune_entry.get("relative_path")
    if not isinstance(marker, dict) or not isinstance(relative_path, str):
        raise ReservationConflictError("prune marker authorization is malformed")
    name = marker.get("name")
    if not isinstance(name, str) or not re.fullmatch(
        r"\.depth-surge-prune-v2-[0-9a-f]{32}\.marker", name, flags=re.ASCII
    ):
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
        or marker.get("sha256") != hashlib.sha256(payload).hexdigest()
        or entry.get("target_relative_path") != f"{relative_path}/{name}"
    ):
        raise ReservationConflictError("prune marker plan differs from its authorization")
    return dict(prune_entry), payload


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
    index_path = root / FINAL_ENCODING_RESERVATION_RELATIVE_PATH
    index = FinalEncodingReservationIndexV1.from_object(
        _load_reservation_json(index_path, max_bytes=256 * 1024)
    )
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
        target.transaction_index_fingerprint != index.fingerprint
        or target.publication_generation != index.publication_generation
        or target.reservation_generation != index.reservation_generation
        or target.final_video_relative_path != index.final_video_relative_path
        or target.sibling_video_temporary_relative_path
        != index.sibling_video_temporary_relative_path
        or target.target_parent_identity != index.sibling_video_parent_identity
        or target.publication_method != index.final_video_publication_method
    ):
        raise ReservationConflictError("final target descriptor differs from its index")
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
) -> bool:
    """Resume descriptor-first, index-last retirement after terminal settings."""

    root = Path(job_root).resolve()
    index_path = root / FINAL_ENCODING_RESERVATION_RELATIVE_PATH
    if not index_path.exists():
        return False
    if settings.status not in {"completed", "failed"}:
        return False
    index = FinalEncodingReservationIndexV1.from_object(
        _load_reservation_json(index_path, max_bytes=256 * 1024)
    )
    target_descriptor_path = root / index.final_video_target_descriptor_relative_path
    if target_descriptor_path.exists():
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
    index_path.unlink()
    _sync_parent(index_path.parent)
    return True


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
