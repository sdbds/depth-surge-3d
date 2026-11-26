"""Irreversible final-video publication and three-way media audit."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import os
import re
import shutil
import stat
import secrets
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable, Iterable, Literal, TypeVar, cast

from ...core.video_timing import EncodingTimeline, rate_text
from ...rendering.quality.cleanup_transaction import (
    _validate_payload_pruned_projection,
    PlannedPruneMarkerV1,
    build_payload_pruned_document,
    plan_prune_markers_v1,
    prune_entry_from_reservation_v1,
)
from ...rendering.quality.diagnostic_transaction import (
    DiagnosticsMetadataProjection,
    project_diagnostics_metadata,
    validate_quality_input_binding,
)

from ...io.canonical_json import (
    CanonicalJsonError,
    canonical_json_bytes,
    strict_canonical_json_loads,
)
from ...io.settings_artifact import (
    ProcessingSettingsArtifactV5,
    SETTINGS_ARTIFACT_MAX_RAW_BYTES,
    SettingsArtifactSchemaError,
    validate_contained_legacy_final_component_v1,
    validate_portable_final_component_v1,
)
from ...io.durable_publish import (
    OwnedSiblingTemporary,
    OwnedTemporaryIdentityError,
    capture_optional_publication_target_identity,
    durable_replace,
)
from ...io.file_identity import (
    DirectoryIdentityV1,
    PersistedFileIdentityV1,
    PosixFileIdentityV1,
    WindowsFileIdentityV1,
    capture_file_identity,
    file_identity_from_open_handle,
)
from ...io.nofollow_io import (
    AcquiredRoot,
    AuthenticatedFixedFileV1,
    FIXED_FILE_HASH_STREAM_BYTES,
    NoFollowPathError,
    scan_fixed_regular_v1,
)
from ...io.streamed_canonical_json import StreamedCanonicalObject
from .encoding_provider import FinalContainerExpectation
from .final_container_validation import (
    FinalContainerProcessError,
    validate_final_container_v4,
    validate_open_linux_container,
)
from .streamed_vr_evidence import (
    StreamedVrInvocationV1,
    StreamedVrOutputEvidenceV1,
    validate_raw_rgb24_input_v1,
)
from ...io.reservation_protocol import (
    SETTINGS_ARTIFACT_RELATIVE_PATH,
    AllocationAdapter,
    JobControlV1,
    AllocationCounter,
    FinalPublicationRecovery,
    FinalPublicationObservation,
    InjectedReservationCrash,
    ReservationConflictError,
    ReservationIncompleteError,
    DirectoryIdentity,
    ReservationFileIdentity,
    ReservationDescriptor,
    ReservedArtifactTransaction,
    FINAL_RESERVATION_DESCRIPTOR_RAW_BYTES,
    allocation_unit_for_path,
    descriptor_create_new_charge,
    platform_allocation_adapter,
    platform_allocation_counter,
    payload_logical_bytes,
    placeholder_entry_charge,
    publish_index_owned_descriptor_v1,
    publish_write_ahead_intent,
    recover_control_artifact_pair_v1,
    reservation_extent_charge,
    same_parent_replace_v1,
    sha256_canonical,
    validate_descriptor_against_entry_v1,
    validate_reservation_entry_v1,
    load_job_control,
    validate_final_encoding_entry_order,
    assert_only_job_control_owned_reservations_v1,
    assert_index_owned_reservation_namespace_v1,
    classify_payload_frame,
    classify_final_publication_observation,
    create_parent_bound_placeholder_v1,
    observe_parent_bound_regular_v1,
    _directory_identity,
    _identity,
    _read_bounded_regular_file,
    _unlink_safe_control_temporary,
)


FINAL_VIDEO_MANIFEST_RELATIVE_PATH = Path("final_video_manifest.json")
ENCODING_INPUT_MANIFEST_RELATIVE_PATH = Path("encoding_input_manifest.json")
FINAL_VIDEO_MANIFEST_MAX_RAW_BYTES = 1024 * 1024
FINAL_FIXED_TARGET_HASH_STREAM_BYTES = FIXED_FILE_HASH_STREAM_BYTES
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


class FinalEncodingDiskBudgetError(FinalPublicationError):
    """The read-only final-encoding reservation admission did not fit."""

    def __init__(self, path: Path, required_bytes: int, free_bytes: int) -> None:
        self.path = Path(path)
        self.required_bytes = int(required_bytes)
        self.free_bytes = int(free_bytes)
        super().__init__(
            f"final encoding requires {self.required_bytes} reservation bytes on "
            f"{self.path}; free space is {self.free_bytes} bytes"
        )


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
    schema_version: int = 1

    def _base_object(self) -> dict:
        return {
            "schema_version": self.schema_version,
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
        self._validate_index_identity()
        self._validate_video_paths()
        diagnostics = self._validated_diagnostics_binding()
        entries = self._validated_extents(diagnostics)
        self._validate_extent_ownership(entries)

    def _validate_index_identity(self) -> None:
        if type(self.schema_version) is not int or self.schema_version not in (1, 2):
            raise ReservationConflictError("final encoding index version is invalid")
        if self.schema_version == 2 and self.final_video_publication_method != "replace_existing":
            raise ReservationConflictError("explicit replacement index has no existing target")
        for value, digits, label in (
            (self.job_id, 32, "final index job id"),
            (self.settings_identity_fingerprint, 64, "final index settings identity"),
            (self.settings_raw_sha256_at_preflight, 64, "final index settings hash"),
            (self.publication_generation, 32, "final index publication generation"),
            (self.reservation_generation, 32, "final index reservation generation"),
            (self.fingerprint, 64, "final index fingerprint"),
        ):
            if (
                not isinstance(value, str)
                or re.fullmatch(rf"[0-9a-f]{{{digits}}}", value, flags=re.ASCII) is None
            ):
                raise ReservationConflictError(f"{label} is invalid")
        if (
            type(self.settings_revision_at_preflight) is not int
            or not 0 <= self.settings_revision_at_preflight <= 2**64 - 1
        ):
            raise ReservationConflictError("final index settings revision is invalid")

    def _validate_video_paths(self) -> None:
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

    def _validated_diagnostics_binding(self) -> tuple[str | None, str | None]:
        diagnostics = (
            self.diagnostics_metadata_raw_sha256_at_preflight,
            self.diagnostics_metadata_fingerprint_at_preflight,
        )
        if (diagnostics[0] is None) != (diagnostics[1] is None):
            raise ReservationConflictError("final index diagnostics binding is incomplete")
        for value in diagnostics:
            if value is not None and _HEX64.fullmatch(value) is None:
                raise ReservationConflictError("final index diagnostics hash is invalid")
        return diagnostics

    def _validated_extents(self, diagnostics: tuple[str | None, str | None]) -> tuple[dict, ...]:
        if not 1 <= len(self.extent_entries) <= 64:
            raise ReservationConflictError("final index extent count is invalid")
        entries = tuple(validate_reservation_entry_v1(entry) for entry in self.extent_entries)
        validate_final_encoding_entry_order([entry["reservation_kind"] for entry in entries])
        diagnostics_entries = tuple(
            entry for entry in entries if entry["reservation_kind"] == "diagnostics_payload_pruned"
        )
        diagnostics_bound = diagnostics[0] is not None
        if (len(diagnostics_entries) == 1) != diagnostics_bound:
            raise ReservationConflictError(
                "final index diagnostics binding differs from its extent entry"
            )
        if diagnostics_entries and (
            diagnostics_entries[0]["existing_target_raw_sha256_at_preflight"] != diagnostics[0]
        ):
            raise ReservationConflictError(
                "final index diagnostics extent hash crosses its metadata binding"
            )
        return entries

    def _validate_extent_ownership(self, entries: tuple[dict, ...]) -> None:
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

    def reject_implicit_video_replacement(self, root: Path) -> None:
        # V1 recorded a file found at preflight, not the caller's permission to replace it.
        if self.schema_version == 1 and self.final_video_publication_method == "replace_existing":
            current = capture_optional_publication_target_identity(
                root / self.final_video_relative_path
            )
            if current == self.final_video_existing_target_identity_at_preflight:
                raise ReservationConflictError(
                    "historical reservation has no explicit replacement authority; preserving the existing video"
                )

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
            type(value["schema_version"]) is not int
            or value["schema_version"] not in (1, 2)
            or value["algorithm_version"] != "final-encoding-reservation-v1"
            or value["settings_artifact_relative_path"] != "processing-settings-v5.json"
            or not isinstance(value["extent_entries"], list)
        ):
            raise ReservationConflictError("final encoding index version is invalid")
        existing = value["final_video_existing_target_identity_at_preflight"]
        index = cls(
            schema_version=value["schema_version"],
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


_UINT64_MAX = (1 << 64) - 1


def _checked_u64(value: int, label: str) -> int:
    if type(value) is not int or not 0 <= value <= _UINT64_MAX:
        raise FinalPublicationValidationError(f"{label} exceeds the unsigned 64-bit range")
    return value


def _checked_u64_add(left: int, right: int, label: str) -> int:
    return _checked_u64(_checked_u64(left, label) + _checked_u64(right, label), label)


def final_encoding_new_reserve_v1(
    index: FinalEncodingReservationIndexV1,
    *,
    allocation_unit: int,
) -> int:
    """Return the complete create-new reserve before the final index exists."""

    if not isinstance(index, FinalEncodingReservationIndexV1):
        raise TypeError("final encoding reserve requires its frozen index")
    if type(allocation_unit) is not int or allocation_unit <= 0:
        raise ValueError("final encoding allocation unit must be positive")
    total = descriptor_create_new_charge(
        len(canonical_json_bytes(index.to_object())),
        allocation_unit,
    )
    target_descriptor_raw = _final_video_target_descriptor_raw_bound_v1(index)
    if target_descriptor_raw > FINAL_RESERVATION_DESCRIPTOR_RAW_BYTES:
        raise ReservationConflictError(
            "final-video target descriptor exceeds its frozen raw-byte bound"
        )
    target_charge = descriptor_create_new_charge(
        target_descriptor_raw,
        allocation_unit,
    ) + placeholder_entry_charge(
        index.final_video_publication_method,
        allocation_unit,
    )
    total = _checked_u64_add(total, target_charge, "final-video target reservation charge")
    for entry in index.extent_entries:
        if entry["reservation_kind"] == "job_terminal_settings":
            continue
        total = _checked_u64_add(
            total,
            reservation_extent_charge(
                entry["payload_raw_max"],
                FINAL_RESERVATION_DESCRIPTOR_RAW_BYTES,
                entry["publication_method"],
                allocation_unit,
            ),
            "final-encoding reservation charge",
        )
    return total


def _admit_final_encoding_reserve_v1(
    *,
    job_root: Path,
    index: FinalEncodingReservationIndexV1,
    allocation_unit: int,
    free_bytes: Callable[[Path], int] | None,
) -> int:
    root_identity = _reservation_directory_identity(job_root)
    for entry in index.extent_entries:
        parent = DirectoryIdentity.from_object(entry["target_parent_identity"])
        if parent.volume_key != root_identity.volume_key:
            raise FinalPublicationValidationError(
                "final reservation targets must remain on the job-root volume"
            )
    required = final_encoding_new_reserve_v1(index, allocation_unit=allocation_unit)
    query = free_bytes or (lambda path: int(shutil.disk_usage(path).free))
    available = _checked_u64(int(query(job_root)), "final-encoding free bytes")
    if available < required:
        raise FinalEncodingDiskBudgetError(job_root, required, available)
    return required


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
            if (
                not isinstance(value, str)
                or re.fullmatch(rf"[0-9a-f]{{{digits}}}", value, flags=re.ASCII) is None
            ):
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
            target_parent_identity=DirectoryIdentity.from_object(value["target_parent_identity"]),
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


def _final_video_target_descriptor_raw_bound_v1(
    index: FinalEncodingReservationIndexV1,
) -> int:
    """Return the concrete/max target-descriptor bytes before any publication."""

    identity = index.final_video_existing_target_identity_at_preflight
    if identity is None:
        parent_identity = index.sibling_video_parent_identity.file_identity
        if index.sibling_video_parent_identity.kind == "windows":
            assert isinstance(parent_identity, WindowsFileIdentityV1)
            maximum_inner: WindowsFileIdentityV1 | PosixFileIdentityV1 = WindowsFileIdentityV1(
                parent_identity.volume_serial,
                "f" * 32,
            )
            identity = ReservationFileIdentity("windows", maximum_inner, 1)
        else:
            assert isinstance(parent_identity, PosixFileIdentityV1)
            maximum_inner = PosixFileIdentityV1(
                parent_identity.filesystem_uuid,
                -(1 << 31),
                "f" * 256,
            )
            identity = ReservationFileIdentity("posix", maximum_inner, 1)
    descriptor = FinalVideoTargetReservationV1.build(
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
        reserved_target_identity=identity,
    )
    return len(canonical_json_bytes(descriptor.to_object()))


def _assert_reserved_final_target_preboundary_v1(
    path: Path,
    descriptor: FinalVideoTargetReservationV1,
) -> None:
    """Validate the exact final target state that may still be replaced."""

    if not _regular_nonlink(path):
        raise ReservationConflictError("final reserved target is missing or unsafe")
    metadata = path.stat(follow_symlinks=False)
    if metadata.st_nlink != 1:
        raise ReservationConflictError("final reserved target is not single-link")
    if _reservation_identity(path) != descriptor.reserved_target_identity:
        raise ReservationConflictError("final reserved target identity changed")
    if descriptor.publication_method == "replace_placeholder" and metadata.st_size != 0:
        raise ReservationConflictError("final reserved placeholder is nonempty")


def _validate_committed_publication_triad_v1(
    root: Path,
    index: FinalEncodingReservationIndexV1,
) -> None:
    """Bind descriptorless committed manifests to each other and the current video."""

    input_path = root / ENCODING_INPUT_MANIFEST_RELATIVE_PATH
    final_manifest_path = root / FINAL_VIDEO_MANIFEST_RELATIVE_PATH
    final_video_path = root / index.final_video_relative_path
    if not all(
        _regular_nonlink(path) for path in (input_path, final_manifest_path, final_video_path)
    ):
        raise ReservationConflictError("committed publication artifact is missing or unsafe")
    try:
        input_projection = _stream_input_manifest_projection_v1(
            root,
            input_path,
            expected_frame_count=_load_settings_frame_count_v1(root),
        )
        manifest = parse_final_video_manifest_v4(
            _read_bounded_regular_file(final_manifest_path, max_bytes=1024 * 1024)
        )
        _validate_normalized_argv(
            manifest.resolved_encoding_arguments,
            input_stage_fingerprint=input_projection.fingerprint,
            relative_path=index.final_video_relative_path,
            expected_mode=input_projection.mode,
        )
        proof = manifest.encoded_output
        if proof is not None and (
            input_projection.streamed_invocation is not None
            or proof.reservation_index_fingerprint != index.fingerprint
            or proof.sibling_relative_path != index.sibling_video_temporary_relative_path
        ):
            raise FinalPublicationValidationError(
                "committed file-backed evidence differs from index"
            )
        scanned = _scan_fixed_file_v1(
            root, final_video_path, expected_identity=proof.identity if proof else None
        )
        video_sha256, video_byte_count = scanned.raw_sha256, scanned.byte_count
        streamed_invocation = input_projection.streamed_invocation
        streamed_output = input_projection.streamed_output
        if streamed_invocation is not None:
            assert streamed_output is not None
            if (
                streamed_invocation.materialized_arguments(input_projection.fingerprint)
                != manifest.resolved_encoding_arguments
                or streamed_output.reservation_index_fingerprint != index.fingerprint
                or streamed_output.sibling_relative_path
                != index.sibling_video_temporary_relative_path
                or streamed_output.identity != capture_file_identity(final_video_path)
                or streamed_output.raw_sha256 != video_sha256
                or streamed_output.byte_count != video_byte_count
            ):
                raise FinalPublicationValidationError(
                    "committed streamed output differs from its launch evidence"
                )
    except (FinalPublicationValidationError, OSError) as error:
        raise ReservationConflictError("committed publication evidence is malformed") from error
    if (
        manifest.relative_path != index.final_video_relative_path
        or manifest.input_stage_fingerprint != input_projection.fingerprint
        or manifest.sha256 != video_sha256
        or manifest.byte_count != video_byte_count
    ):
        raise ReservationConflictError(
            "committed video and manifest evidence are not three-way bound"
        )


@dataclass(frozen=True, slots=True)
class _FinalSettingsLoadAuthorityV1:
    job_id: str
    settings_identity_fingerprint: str
    final_media_publication_generation: str | None
    expected_final_relative_path: str
    cleanup_consumed: bool


def _scalarize_final_settings_for_load_v1(
    settings: ProcessingSettingsArtifactV5,
) -> _FinalSettingsLoadAuthorityV1:
    value = settings.to_object()
    metadata = value["metadata"]
    output_info = value["output_info"]
    authority = _FinalSettingsLoadAuthorityV1(
        job_id=metadata["job_id"],
        settings_identity_fingerprint=metadata["settings_identity_fingerprint"],
        final_media_publication_generation=metadata["final_media_publication_generation"],
        expected_final_relative_path=output_info["expected_final_relative_path"],
        cleanup_consumed=metadata["cleanup_status"] not in {"not_started", "not_requested"},
    )
    del output_info, metadata, value
    return authority


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
        _assert_reserved_final_target_preboundary_v1(
            self.final_video_path,
            self.target_descriptor,
        )

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

    @staticmethod
    def _bounded_transaction_has_durable_intent(
        transaction: ReservedArtifactTransaction,
        payload: bytes,
    ) -> bool:
        phase = transaction.validate_physical_phase(payload)
        if phase != "full-source":
            return phase in {"short-source", "replacement-gap", "committed-target"}
        raw = _read_bounded_regular_file(
            transaction.source_path,
            max_bytes=transaction.descriptor.logical_byte_count,
        )
        return classify_payload_frame(raw, payload) == "complete-full"

    @staticmethod
    def _bounded_payload_intent(
        transaction: ReservedArtifactTransaction,
    ) -> bytes | None:
        if transaction.source_path.exists():
            path = transaction.source_path
        elif transaction.target_path.exists() and (
            _reservation_identity(transaction.target_path)
            == transaction.descriptor.filesystem_identity
        ):
            path = transaction.target_path
        else:
            return None
        prefix = PreparedFinalEncodingReservationV1._canonical_prefix_byte_count(path)
        if prefix in {None, 0}:
            return None
        if prefix > transaction.descriptor.payload_raw_max:
            raise ReservationConflictError("indexed payload intent exceeds its raw bound")
        return _read_bounded_regular_file(path, max_bytes=prefix)

    def _payload_pruned_document(
        self,
        transaction: ReservedArtifactTransaction,
    ) -> StreamedCanonicalObject:
        """Rebuild the indexed O(N) metadata intent from durable scalar evidence."""
        return _rebuild_payload_pruned_document_v1(
            self.job_root,
            self.index,
            transaction,
        )

    @staticmethod
    def _streamed_intent_state(
        transaction: ReservedArtifactTransaction,
        document: StreamedCanonicalObject,
    ) -> str:
        """Classify zero/partial/full streamed intent without retaining its bytes."""

        if transaction.source_path.exists():
            path = transaction.source_path
            if _reservation_identity(path) != transaction.descriptor.filesystem_identity:
                raise ReservationConflictError("streamed intent source identity changed")
        elif (
            transaction.target_path.exists()
            and _reservation_identity(transaction.target_path)
            == transaction.descriptor.filesystem_identity
        ):
            path = transaction.target_path
        else:
            return "zero"
        prefix = PreparedFinalEncodingReservationV1._canonical_prefix_byte_count(path)
        if prefix is None or prefix > document.raw_byte_count:
            raise ReservationConflictError("streamed intent has a foreign prefix")
        if prefix == 0:
            return "zero"
        PreparedFinalEncodingReservationV1._verify_streamed_prefix(path, prefix, document)
        if prefix < document.raw_byte_count:
            return "partial"
        return transaction.validate_streamed_physical_phase(
            prefix_byte_count=document.raw_byte_count,
            prefix_sha256=document.raw_sha256,
        )

    @staticmethod
    def _verify_streamed_prefix(path: Path, prefix: int, document: StreamedCanonicalObject) -> None:
        remaining = prefix
        digest = hashlib.sha256()
        count = 0
        with path.open("rb", buffering=0) as handle:
            for chunk in document.chunks():
                digest.update(chunk)
                count += len(chunk)
                if remaining:
                    expected = chunk[:remaining]
                    actual = handle.read(len(expected))
                    if actual != expected:
                        raise ReservationConflictError(
                            "streamed intent has a foreign canonical prefix"
                        )
                    remaining -= len(expected)
            if remaining:
                raise ReservationConflictError("streamed intent prefix exceeds its document")
        if count != document.raw_byte_count or digest.hexdigest() != document.raw_sha256:
            raise ReservationConflictError("streamed intent replay identity changed")

    def _recover_monotonic_extent_intent_prefix(self) -> bool:
        """Publish one durable semantic intent before any terminal retirement."""

        transaction = self.producer_settings_transaction
        payload = self.producer_settings_payload
        if (
            transaction is not None
            and payload is not None
            and (self._bounded_transaction_has_durable_intent(transaction, payload))
        ):
            self.commit_producer_settings()
            return True

        video_state = self._video_state()
        input_state = self._input_manifest_state(video_state)
        if input_state in {"complete-full", "complete-short"}:
            self.recover_input_manifest_intent()
            return True

        final_state = self._final_manifest_state()
        if final_state in {"complete-full", "complete-short"}:
            self.recover_final_manifest_intent()
            return True

        for position, (entry, transaction, payload) in enumerate(self.prune_marker_transactions):
            if transaction is None or not self._bounded_transaction_has_durable_intent(
                transaction,
                payload,
            ):
                continue
            self._consume_reserved_payload(transaction, payload)
            updated = list(self.prune_marker_transactions)
            updated[position] = (entry, None, payload)
            self.prune_marker_transactions = tuple(updated)
            return True

        if self._recover_diagnostics_intent_prefix():
            return True

        transaction = self.cleanup_pending_settings_transaction
        payload = self.cleanup_pending_settings_payload
        if (
            transaction is not None
            and payload is not None
            and (self._bounded_transaction_has_durable_intent(transaction, payload))
        ):
            self.commit_cleanup_pending_settings()
            return True
        return False

    def _recover_diagnostics_intent_prefix(self) -> bool:
        transaction = self.diagnostics_payload_pruned_transaction
        if transaction is None:
            return False
        if transaction.source_path.exists():
            prefix = self._canonical_prefix_byte_count(transaction.source_path)
            if prefix is None:
                raise ReservationConflictError(
                    "diagnostics payload-pruned source has a foreign prefix"
                )
            if prefix == 0:
                return False
        document = self._payload_pruned_document(transaction)
        state = self._streamed_intent_state(transaction, document)
        if state not in {"full-source", "short-source", "replacement-gap", "committed-target"}:
            return False
        transaction.recover_streamed_prefix(
            prefix_byte_count=document.raw_byte_count,
            prefix_sha256=document.raw_sha256,
        )
        self.diagnostics_payload_pruned_transaction = None
        return True

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
            or _hash_file(self.job_root, target) != (expected_hash, expected_count)
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
        transaction.reset_authenticated_source_to_zero_full()
        if transaction.validate_physical_phase(expected_payload) != "full-source":
            raise ReservationConflictError("settings reservation reset changed physical phase")

    def commit_producer_settings(self) -> ProcessingSettingsArtifactV5:
        payload = self.producer_settings_payload
        transaction = self.producer_settings_transaction
        if transaction is not None:
            current = ProcessingSettingsArtifactV5.parse(
                _read_bounded_regular_file(
                    self.job_root / "processing-settings-v5.json",
                    max_bytes=SETTINGS_ARTIFACT_MAX_RAW_BYTES,
                )
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
            _read_bounded_regular_file(
                self.job_root / "processing-settings-v5.json",
                max_bytes=SETTINGS_ARTIFACT_MAX_RAW_BYTES,
            )
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
            if not _regular_nonlink(target) or _hash_file(self.job_root, target) != (
                entry["marker"]["sha256"],
                len(payload),
            ):
                raise ReservationConflictError("indexed prune marker did not commit")
            committed.append((entry, None, payload))
            entries.append(entry)
        self.prune_marker_transactions = tuple(committed)
        return tuple(entries)

    @property
    def index_fingerprint(self) -> str:
        return self.index.fingerprint

    def validate_diagnostics_preflight(
        self,
        *,
        raw_sha256: str,
        fingerprint: str,
    ) -> None:
        if (
            self.index.diagnostics_metadata_raw_sha256_at_preflight != raw_sha256
            or self.index.diagnostics_metadata_fingerprint_at_preflight != fingerprint
        ):
            raise ReservationConflictError(
                "diagnostics metadata differs from final reservation preflight"
            )

    def commit_payload_pruned_metadata(
        self,
        payload: StreamedCanonicalObject,
    ) -> None:
        transaction = self.diagnostics_payload_pruned_transaction
        if not isinstance(payload, StreamedCanonicalObject):
            raise TypeError("payload-pruned metadata must be replayable")
        if transaction is None:
            target = self.job_root / "04_stereo_diagnostics" / "metadata.json"
            digest, byte_count = _hash_file(self.job_root, target)
            if digest != payload.raw_sha256 or byte_count != payload.raw_byte_count:
                raise ReservationConflictError(
                    "payload-pruned metadata differs from committed target"
                )
            return
        if transaction.descriptor.payload_raw_max != payload.raw_byte_count:
            raise ReservationConflictError(
                "payload-pruned metadata differs from its indexed raw bound"
            )
        try:
            phase = transaction.validate_streamed_physical_phase(
                prefix_byte_count=payload.raw_byte_count,
                prefix_sha256=payload.raw_sha256,
            )
        except ReservationConflictError:
            transaction.stage_complete_full_streamed(
                chunks=payload.chunks,
                prefix_byte_count=payload.raw_byte_count,
                prefix_sha256=payload.raw_sha256,
            )
            phase = "full-source"
        if phase == "full-source":
            transaction.consume_streamed_prefix(
                prefix_byte_count=payload.raw_byte_count,
                prefix_sha256=payload.raw_sha256,
            )
        else:
            transaction.recover_streamed_prefix(
                prefix_byte_count=payload.raw_byte_count,
                prefix_sha256=payload.raw_sha256,
            )
        self.diagnostics_payload_pruned_transaction = None

    def commit_cleanup_pending_settings(self) -> ProcessingSettingsArtifactV5:
        payload = self.cleanup_pending_settings_payload
        transaction = self.cleanup_pending_settings_transaction
        if transaction is not None:
            current = ProcessingSettingsArtifactV5.parse(
                _read_bounded_regular_file(
                    self.job_root / "processing-settings-v5.json",
                    max_bytes=SETTINGS_ARTIFACT_MAX_RAW_BYTES,
                )
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
            _read_bounded_regular_file(
                self.job_root / "processing-settings-v5.json",
                max_bytes=SETTINGS_ARTIFACT_MAX_RAW_BYTES,
            )
        )
        if current.to_object()["metadata"]["cleanup_status"] != "pending":
            raise ReservationConflictError("cleanup-pending settings transition did not commit")
        return current

    @staticmethod
    def _canonical_prefix_byte_count_handle(handle: BinaryIO) -> int | None:
        offset = 0
        first_zero: int | None = None
        handle.seek(0)
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

    @staticmethod
    def _canonical_prefix_byte_count(path: Path) -> int | None:
        with path.open("rb", buffering=0) as handle:
            return PreparedFinalEncodingReservationV1._canonical_prefix_byte_count_handle(handle)

    def _video_state(self) -> str:
        self.index.reject_implicit_video_replacement(self.job_root)
        final = self.final_video_path
        sibling_present = self.sibling_path.exists() or self.sibling_path.is_symlink()
        if final.exists() or final.is_symlink():
            identity = _reservation_identity(final)
            if identity == self.target_descriptor.reserved_target_identity:
                _assert_reserved_final_target_preboundary_v1(
                    final,
                    self.target_descriptor,
                )
                return "reserved-target"
            return "committed-new"
        if os.name == "nt" and sibling_present and _regular_nonlink(self.sibling_path):
            return "windows-replacement-gap"
        raise ReservationConflictError("final video is outside its indexed namespace states")

    def _input_manifest_state(self, video_state: str) -> str:
        transaction = self.input_manifest_transaction
        target = self.job_root / ENCODING_INPUT_MANIFEST_RELATIVE_PATH
        if transaction is None or not transaction.source_path.exists():
            if not _regular_nonlink(target):
                if transaction is None:
                    raise ReservationConflictError("committed input manifest is missing")
                raise ReservationConflictError("input extent has no authorized location")
            _stream_input_manifest_identity(
                self.job_root,
                target,
                expected_frame_count=_load_settings_frame_count_v1(self.job_root),
            )
            return "committed-new"
        with transaction.open_authenticated_source() as handle:
            source_size = os.fstat(handle.fileno()).st_size
            prefix = self._canonical_prefix_byte_count_handle(handle)
            if prefix is None:
                return "partial"
            if prefix == 0:
                return "zero"
            handle.seek(0)
            try:
                projection = _parse_encoding_input_manifest_stream_v1(
                    handle,
                    raw_byte_count=prefix,
                    expected_frame_count=_load_settings_frame_count_v1(self.job_root),
                )
                self._validate_input_extent_tail(
                    handle,
                    source_size,
                    transaction.descriptor.logical_byte_count,
                    projection.raw_byte_count,
                )
            except FinalPublicationValidationError:
                return "partial"
        transaction.validate_streamed_physical_phase(
            prefix_byte_count=projection.raw_byte_count,
            prefix_sha256=projection.raw_sha256,
        )
        if source_size == projection.raw_byte_count:
            return "complete-short"
        return "complete-full" if video_state != "reserved-target" else "encoding-prepass-staged"

    @staticmethod
    def _validate_input_extent_tail(
        handle: BinaryIO, source_size: int, logical_size: int, raw_size: int
    ) -> None:
        if source_size == logical_size:
            tail = source_size - raw_size
            while tail:
                chunk = handle.read(min(1024 * 1024, tail))
                if not chunk or any(chunk):
                    raise FinalPublicationValidationError(
                        "encoding input reservation tail is not all zero"
                    )
                tail -= len(chunk)
        elif source_size != raw_size:
            raise FinalPublicationValidationError(
                "encoding input source has an intermediate length"
            )

    def _final_manifest_state(self) -> str:
        return self._final_manifest_observation()[0]

    def _final_manifest_observation(self) -> tuple[str, FinalVideoManifestV4 | None]:
        transaction = self.final_manifest_transaction
        target = self.job_root / FINAL_VIDEO_MANIFEST_RELATIVE_PATH
        if transaction is None or not transaction.source_path.exists():
            if not _regular_nonlink(target):
                if transaction is None:
                    raise ReservationConflictError("committed final manifest is missing")
                raise ReservationConflictError("final manifest extent has no authorized location")
            return "committed-new", parse_final_video_manifest_v4(
                _read_bounded_regular_file(target, max_bytes=FINAL_VIDEO_MANIFEST_MAX_RAW_BYTES)
            )
        return self._final_manifest_source_observation(transaction)

    def _final_manifest_source_observation(
        self, transaction: ReservedArtifactTransaction
    ) -> tuple[str, FinalVideoManifestV4 | None]:
        with transaction.open_authenticated_source() as handle:
            source_size = os.fstat(handle.fileno()).st_size
            prefix = self._canonical_prefix_byte_count_handle(handle)
            if prefix is None:
                return "partial", None
            if prefix == 0:
                return "zero", None
            if prefix > transaction.descriptor.payload_raw_max:
                return "partial", None
            handle.seek(0)
            raw = handle.read(prefix)
            if len(raw) != prefix:
                raise ReservationConflictError(
                    "final manifest source changed during authenticated read"
                )
        try:
            manifest = parse_final_video_manifest_v4(raw)
        except FinalPublicationValidationError:
            return "partial", None
        if source_size == prefix:
            return "complete-short", manifest
        if source_size == transaction.descriptor.logical_byte_count:
            return "complete-full", manifest
        return "partial", None

    def _validate_committed_publication_triad(self) -> None:
        _validate_committed_publication_triad_v1(self.job_root, self.index)

    def validate_authenticated_publication(
        self,
        *,
        container_validator: Callable[[Path], bool] | None = None,
    ) -> None:
        """Validate the complete active-index publication, including its container."""

        final_path, expectation = self.authenticated_container_expectation()
        validator = container_validator
        if validator is None:

            def validator(path: Path) -> bool:
                return validate_final_container_v4(path, expectation)

        if not validator(final_path):
            raise ReservationConflictError("active final publication container validation failed")

    def authenticated_container_expectation(
        self,
    ) -> tuple[Path, FinalContainerExpectation]:
        """Authenticate persisted intent and return only the final probe scalars."""

        recovery = self.observe()
        if recovery.phase != "publication_authenticated":
            raise ReservationConflictError(
                "active final publication has not reached its authenticated phase"
            )
        settings = ProcessingSettingsArtifactV5.parse(
            _read_bounded_regular_file(
                self.job_root / "processing-settings-v5.json",
                max_bytes=512 * 1024,
            )
        )
        input_projection = _stream_input_manifest_projection_v1(
            self.job_root,
            self.job_root / ENCODING_INPUT_MANIFEST_RELATIVE_PATH,
            expected_frame_count=_settings_frame_count_v1(settings),
        )
        manifest = parse_final_video_manifest_v4(
            _read_bounded_regular_file(
                self.job_root / FINAL_VIDEO_MANIFEST_RELATIVE_PATH,
                max_bytes=FINAL_VIDEO_MANIFEST_MAX_RAW_BYTES,
            )
        )
        expectation = _final_container_expectation_from_intent_v1(
            settings=settings,
            input_projection=input_projection,
            arguments=manifest.resolved_encoding_arguments,
        )
        final_path = self.job_root / self.index.final_video_relative_path
        return final_path, expectation

    def observe(self) -> FinalPublicationRecovery:
        video_state = self._video_state()
        input_state = self._input_manifest_state(video_state)
        final_state, manifest = self._final_manifest_observation()
        sibling_present = self.sibling_path.exists() or self.sibling_path.is_symlink()
        if (
            video_state == "reserved-target"
            and input_state == "encoding-prepass-staged"
            and sibling_present
            and final_state == "partial"
        ):
            raise ReservationIncompleteError(
                "incomplete encoded-output evidence; the candidate and existing target are preserved. "
                "Restore the complete job metadata from backup or use a new output directory."
            )
        if manifest is not None and manifest.encoded_output is not None:
            proof = manifest.encoded_output
            pre_rename = video_state in {"reserved-target", "windows-replacement-gap"}
            candidate = self.sibling_path if pre_rename else self.final_video_path
            if (
                proof.reservation_index_fingerprint != self.index.fingerprint
                or proof.sibling_relative_path != self.sibling_path.name
                or manifest.relative_path != self.index.final_video_relative_path
                or sibling_present != pre_rename
                or not _regular_nonlink(candidate)
                or _reservation_identity(candidate) != proof.identity
            ):
                raise ReservationConflictError(
                    "encoded video no longer belongs to this publication"
                )
            if pre_rename:
                if final_state != "complete-full" or input_state not in {
                    "encoding-prepass-staged",
                    "complete-full",
                }:
                    raise ReservationConflictError(
                        "encoded output has an invalid pre-rename prefix"
                    )
                return FinalPublicationRecovery("encoded_output_staged", "publish-validated-video")
            if final_state == "complete-full" and input_state in {
                "complete-full",
                "complete-short",
            }:
                return FinalPublicationRecovery(
                    "video_replaced_input_intent", "commit-input-manifest"
                )
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
                sibling_present=sibling_present,
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

    def discard_nonsemantic_sibling(self) -> bool:
        """Remove only a sibling classified as pre-boundary or retry scratch."""

        if not (self.sibling_path.exists() or self.sibling_path.is_symlink()):
            return False
        recovery = self.observe()
        if recovery.phase not in {
            "ready_pre_video",
            "post_boundary_retry_sibling_present",
        }:
            return False
        if not _regular_nonlink(self.sibling_path):
            raise ReservationConflictError("indexed nonsemantic sibling is unsafe")
        metadata = self.sibling_path.stat(follow_symlinks=False)
        if metadata.st_nlink != 1:
            raise ReservationConflictError("indexed nonsemantic sibling is not single-link")
        identity = _reservation_identity(self.sibling_path)
        _unlink_safe_control_temporary(
            self.sibling_path,
            expected_identity=identity,
            expected_parent_identity=self.target_descriptor.target_parent_identity,
        )
        _sync_parent(self.sibling_path.parent)
        if self.sibling_path.exists() or self.sibling_path.is_symlink():
            raise ReservationConflictError("indexed nonsemantic sibling removal did not persist")
        return True

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
        for transaction, state, resettable in states:
            if transaction is None or state not in resettable:
                continue
            transaction.reset_authenticated_source_to_zero_full()

    def discard_prelaunch_invocation(self) -> None:
        """Retire one authenticated pre-video reservation while retaining job authority."""

        recovery = self.observe()
        if recovery.phase != "ready_pre_video":
            raise ReservationConflictError("prelaunch discard requires the ready_pre_video phase")
        current = ProcessingSettingsArtifactV5.parse(
            _read_bounded_regular_file(
                self.job_root / "processing-settings-v5.json",
                max_bytes=512 * 1024,
            )
        )
        if current.status != "in_progress":
            raise ReservationConflictError("prelaunch discard requires in-progress settings")
        producer_transaction = self.producer_settings_transaction
        producer_payload = self.producer_settings_payload
        if producer_transaction is not None and producer_payload is not None:
            producer_phase = producer_transaction.validate_physical_phase(producer_payload)
            if producer_phase == "full-source":
                intended = ProcessingSettingsArtifactV5.parse(producer_payload)
                timestamp = intended.to_object()["metadata"]["last_updated_at"]
                if intended != current.with_producer_marker(
                    generation=self.index.publication_generation,
                    updated_at=timestamp,
                ):
                    raise ReservationConflictError(
                        "discarded producer settings intent is not its exact successor"
                    )
                self._reset_uncommitted_settings_source(
                    producer_transaction,
                    producer_payload,
                )
                self.producer_settings_payload = None
        transactions = (
            producer_transaction,
            self.input_manifest_transaction,
            self.final_manifest_transaction,
            *(transaction for _entry, transaction, _payload in self.prune_marker_transactions),
            self.diagnostics_payload_pruned_transaction,
            self.cleanup_pending_settings_transaction,
        )
        active = next((item for item in transactions if item is not None), None)
        allocation_adapter = (
            active.allocation_adapter if active is not None else platform_allocation_adapter
        )
        allocation_counter = (
            active.allocation_counter if active is not None else platform_allocation_counter
        )
        _recover_failed_final_retirement_v1(
            self.job_root,
            self.index,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
            event=self.event,
            crash_at=None,
            expected_status="in_progress",
        )

    def input_publication_source(self) -> "EncodingInputPublicationSource | None":
        transaction = self.input_manifest_transaction
        if transaction is None or not transaction.source_path.exists():
            target = self.job_root / ENCODING_INPUT_MANIFEST_RELATIVE_PATH
            if (
                transaction is not None
                and _reservation_identity(target) != transaction.descriptor.filesystem_identity
            ):
                raise ReservationConflictError("committed input differs from its reserved file")
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

    def requires_explicit_streamed_recovery(self) -> bool:
        """Keep v3 publication phases read-only until their evidence is revalidated."""

        source = self.input_publication_source()
        if source is None:
            return False
        projection = source.projection(
            expected_frame_count=_load_settings_frame_count_v1(self.job_root),
        )
        if projection.streamed_invocation is None:
            return False
        return self.observe().phase != "publication_authenticated"

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
            return parse_final_video_manifest_v4(
                _read_bounded_regular_file(
                    target,
                    max_bytes=FINAL_VIDEO_MANIFEST_MAX_RAW_BYTES,
                )
            )
        prefix = self._canonical_prefix_byte_count(transaction.source_path)
        if prefix in {None, 0} or prefix > transaction.descriptor.payload_raw_max:
            raise ReservationConflictError("final manifest has no complete recovery intent")
        with transaction.source_path.open("rb", buffering=0) as handle:
            raw = handle.read(prefix)
        manifest = parse_final_video_manifest_v4(raw)
        scan = _scan_fixed_file_v1(
            self.job_root,
            self.final_video_path,
            expected_identity=manifest.encoded_output.identity if manifest.encoded_output else None,
        )
        video_hash, video_count = scan.raw_sha256, scan.byte_count
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

    def complete_final_manifest_from_existing_video(
        self,
        *,
        input_stage_fingerprint: str,
        resolved_encoding_arguments: tuple[str, ...],
        expected_video_identity: PersistedFileIdentityV1,
        expected_video_sha256: str,
        expected_video_byte_count: int,
    ) -> FinalVideoManifestV4:
        """Finish a validated post-video v3 publication without re-encoding."""

        recovery = self._advance_existing_video_prefix()
        if recovery.phase == "final_manifest_intent":
            return self._recover_matching_final_manifest(
                input_stage_fingerprint,
                resolved_encoding_arguments,
                expected_video_sha256,
                expected_video_byte_count,
            )
        if recovery.phase == "publication_authenticated":
            return parse_final_video_manifest_v4(
                _read_bounded_regular_file(
                    self.job_root / FINAL_VIDEO_MANIFEST_RELATIVE_PATH,
                    max_bytes=FINAL_VIDEO_MANIFEST_MAX_RAW_BYTES,
                )
            )
        if recovery.phase != "input_manifest_committed_no_final_intent":
            raise ReservationConflictError(
                "existing-video completion requires committed video and input manifest"
            )

        settings = ProcessingSettingsArtifactV5.parse(
            _read_bounded_regular_file(
                self.job_root / "processing-settings-v5.json",
                max_bytes=SETTINGS_ARTIFACT_MAX_RAW_BYTES,
            )
        )
        metadata = settings.to_object()["metadata"]
        if (
            metadata["final_media_producer_contract_version"] != 4
            or metadata["final_media_publication_generation"] != self.index.publication_generation
        ):
            raise ReservationConflictError(
                "existing-video completion lacks producer settings authority"
            )
        input_projection = _stream_input_manifest_projection_v1(
            self.job_root,
            self.job_root / ENCODING_INPUT_MANIFEST_RELATIVE_PATH,
            expected_frame_count=_settings_frame_count_v1(settings),
        )
        invocation = input_projection.streamed_invocation
        output = input_projection.streamed_output
        if (
            input_projection.fingerprint != input_stage_fingerprint
            or invocation is None
            or output is None
            or invocation.materialized_arguments(input_stage_fingerprint)
            != resolved_encoding_arguments
            or output.reservation_index_fingerprint != self.index.fingerprint
            or output.sibling_relative_path != self.index.sibling_video_temporary_relative_path
            or output.identity != expected_video_identity
            or output.raw_sha256 != expected_video_sha256
            or output.byte_count != expected_video_byte_count
        ):
            raise ReservationConflictError(
                "existing streamed video differs from its v3 input evidence"
            )
        _validate_normalized_argv(
            resolved_encoding_arguments,
            input_stage_fingerprint=input_stage_fingerprint,
            relative_path=self.index.final_video_relative_path,
            expected_mode=input_projection.mode,
        )
        video_sha256, video_byte_count = self._verify_existing_video(
            expected_video_identity, expected_video_sha256, expected_video_byte_count
        )
        final_manifest = build_final_video_manifest_v4(
            relative_path=self.index.final_video_relative_path,
            video_sha256=video_sha256,
            video_byte_count=video_byte_count,
            input_stage_fingerprint=input_stage_fingerprint,
            resolved_encoding_arguments=resolved_encoding_arguments,
        )
        transaction = self.final_manifest_transaction
        if transaction is None:
            raise ReservationConflictError(
                "existing-video completion consumed its final manifest reservation"
            )
        self.validate_indexed_predecessor(transaction)
        transaction.consume(final_manifest.encode())
        self.final_manifest_transaction = None
        if self.observe().phase != "publication_authenticated":
            raise ReservationConflictError(
                "existing-video completion did not authenticate publication"
            )
        return final_manifest

    def _advance_existing_video_prefix(self) -> FinalPublicationRecovery:
        recovery = self.observe()
        if recovery.phase == "post_boundary_retry_sibling_present":
            self.discard_nonsemantic_sibling()
            recovery = self.observe()
        if recovery.phase in {
            "windows_video_replace_gap_before_input_commit",
            "windows_video_replace_gap_after_input_commit",
        }:
            self.recover_video_replacement_gap()
            recovery = self.observe()
        if recovery.phase == "video_replaced_input_intent":
            self.recover_input_manifest_intent()
            recovery = self.observe()
        return recovery

    def _recover_matching_final_manifest(
        self,
        input_stage_fingerprint: str,
        resolved_encoding_arguments: tuple[str, ...],
        expected_video_sha256: str,
        expected_video_byte_count: int,
    ) -> FinalVideoManifestV4:
        transaction = self.final_manifest_transaction
        if transaction is None:
            raise ReservationConflictError("streamed final-manifest intent lost its reservation")
        prefix = self._canonical_prefix_byte_count(transaction.source_path)
        if prefix in {None, 0} or prefix > transaction.descriptor.payload_raw_max:
            raise ReservationConflictError("streamed final-manifest intent is incomplete")
        with transaction.open_authenticated_source() as handle:
            raw = handle.read(prefix)
            if len(raw) != prefix:
                raise ReservationConflictError("streamed final-manifest intent changed during read")
        intended = parse_final_video_manifest_v4(raw)
        expected = build_final_video_manifest_v4(
            relative_path=self.index.final_video_relative_path,
            video_sha256=expected_video_sha256,
            video_byte_count=expected_video_byte_count,
            input_stage_fingerprint=input_stage_fingerprint,
            resolved_encoding_arguments=resolved_encoding_arguments,
        )
        if intended != expected:
            raise ReservationConflictError(
                "streamed final-manifest intent differs from v3 evidence"
            )
        manifest = self.recover_final_manifest_intent()
        if self.observe().phase != "publication_authenticated":
            raise ReservationConflictError("streamed final-manifest recovery did not authenticate")
        return manifest

    def _verify_existing_video(
        self,
        expected_video_identity: PersistedFileIdentityV1,
        expected_video_sha256: str,
        expected_video_byte_count: int,
    ) -> tuple[str, int]:
        final_path = self.final_video_path
        try:
            with AcquiredRoot(self.job_root) as acquired:
                with acquired.open_regular_locked(final_path) as handle:
                    identity = file_identity_from_open_handle(handle)
                    video_sha256, video_byte_count = _hash_open_file(handle)
                    if (
                        identity != expected_video_identity
                        or acquired.persistent_regular_identity(final_path) != identity
                    ):
                        raise ReservationConflictError("existing streamed video identity changed")
        except (NoFollowPathError, FileNotFoundError, OSError) as error:
            raise ReservationConflictError(
                "existing streamed video cannot be acquired without following links"
            ) from error
        if (video_sha256, video_byte_count) != (
            expected_video_sha256,
            expected_video_byte_count,
        ):
            raise ReservationConflictError("existing streamed video bytes changed")
        return video_sha256, video_byte_count

    def _retire_committed_manifest_descriptor_prefix(self) -> bool:
        input_transaction = self.input_manifest_transaction
        if input_transaction is not None and not (
            input_transaction.source_path.exists() or input_transaction.source_path.is_symlink()
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
            final_transaction.source_path.exists() or final_transaction.source_path.is_symlink()
        ):
            target = self.job_root / FINAL_VIDEO_MANIFEST_RELATIVE_PATH
            raw = _read_bounded_regular_file(
                target,
                max_bytes=final_transaction.descriptor.payload_raw_max,
            )
            manifest = parse_final_video_manifest_v4(raw)
            video_hash, video_count = _hash_file(self.job_root, self.final_video_path)
            input_source = EncodingInputPublicationSource.from_path(
                self.job_root / ENCODING_INPUT_MANIFEST_RELATIVE_PATH
            )
            if (manifest.sha256, manifest.byte_count) != (
                video_hash,
                video_count,
            ) or manifest.input_stage_fingerprint != input_source.fingerprint:
                raise ReservationConflictError(
                    "committed final manifest does not bind its publication triad"
                )
            final_transaction.recover(raw)
            self.final_manifest_transaction = None
            return True
        return False

    def has_file_backed_output(self) -> bool:
        _, manifest = self._final_manifest_observation()
        return manifest is not None and manifest.encoded_output is not None

    def _resume_file_backed_output(self) -> None:
        state, manifest = self._final_manifest_observation()
        if manifest is None or manifest.encoded_output is None or state == "committed-new":
            return
        phase = self.observe().phase
        proof = manifest.encoded_output
        source = self.input_publication_source()
        if source is None or source.fingerprint != manifest.input_stage_fingerprint:
            raise ReservationConflictError("encoded output does not bind the reserved inputs")
        projection = source.projection(
            expected_frame_count=_load_settings_frame_count_v1(self.job_root)
        )
        if projection.streamed_invocation is not None:
            raise ReservationConflictError("file-backed proof cannot publish streamed inputs")
        _validate_normalized_argv(
            manifest.resolved_encoding_arguments,
            input_stage_fingerprint=source.fingerprint,
            relative_path=manifest.relative_path,
            expected_mode=projection.mode,
        )
        pre_rename = phase == "encoded_output_staged"
        _scan_fixed_file_v1(
            self.job_root,
            self.sibling_path if pre_rename else self.final_video_path,
            expected_identity=proof.identity,
            expected_raw_sha256=proof.raw_sha256,
            expected_byte_count=proof.byte_count,
        )
        if pre_rename:
            transaction = self.final_manifest_transaction
            assert transaction is not None
            # A dead writer may have filled the extent but not yet returned from fsync.
            with transaction.open_authenticated_source(writable=True) as handle:
                raw = manifest.encode()
                if handle.read(len(raw)) != raw:
                    raise ReservationConflictError("encoded-output intent changed before sync")
                os.fsync(handle.fileno())
            gap = self._video_state() == "windows-replacement-gap"
            same_parent_replace_v1(
                self.sibling_path,
                self.final_video_path,
                expected_source_identity=proof.identity,
                expected_target_identity=(
                    None if gap else self.target_descriptor.reserved_target_identity
                ),
                expected_parent_identity=self.target_descriptor.target_parent_identity,
                replace_existing=not gap,
            )
        else:
            # Re-establish the video rename barrier before committing either manifest.
            _sync_parent(self.final_video_path.parent)

    def recover_publication_prefix(self) -> FinalPublicationRecovery:
        """Finish every deterministic, index-owned publication prefix step."""

        if self.requires_explicit_streamed_recovery():
            raise ReservationConflictError(
                "streamed v3 publication requires evidence-bound recovery"
            )
        if (
            self.input_manifest_transaction is not None
            or self.final_manifest_transaction is not None
        ):
            # Consuming a staged manifest can disguise an unrelated final as a legal retry.
            self.observe()
        self._resume_file_backed_output()
        for _ in range(len(self.index.extent_entries) + 8):
            if self._retire_committed_manifest_descriptor_prefix():
                continue
            if self._recover_monotonic_extent_intent_prefix():
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
        current = ProcessingSettingsArtifactV5.parse(
            _read_bounded_regular_file(
                self.job_root / "processing-settings-v5.json",
                max_bytes=512 * 1024,
            )
        )
        _recover_completed_final_retirement_v1(
            self.job_root,
            self.index,
            current,
            expected_target_descriptor=self.target_descriptor,
            event=event or self.event,
            crash_at=crash_at,
        )

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
        _recover_failed_final_retirement_v1(
            self.job_root,
            self.index,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
            event=event or self.event,
            crash_at=None,
        )


@dataclass(frozen=True, slots=True)
class FinalEncodingReservationCapabilityV1:
    """Scalar authority that reloads one reservation phase from durable state."""

    job_root: Path
    index_fingerprint: str
    sibling_relative_path: str
    final_video_relative_path: str

    @classmethod
    def from_prepared(
        cls,
        reservation: PreparedFinalEncodingReservationV1,
    ) -> "FinalEncodingReservationCapabilityV1":
        return cls(
            job_root=reservation.job_root,
            index_fingerprint=reservation.index.fingerprint,
            sibling_relative_path=reservation.index.sibling_video_temporary_relative_path,
            final_video_relative_path=reservation.index.final_video_relative_path,
        )

    @property
    def index_path(self) -> Path:
        return self.job_root / FINAL_ENCODING_RESERVATION_RELATIVE_PATH

    @property
    def sibling_path(self) -> Path:
        return self.job_root / self.sibling_relative_path

    @property
    def final_video_path(self) -> Path:
        return self.job_root / self.final_video_relative_path

    def materialize(self) -> PreparedFinalEncodingReservationV1:
        from ...io.operations import recover_processing_control_prefix_v1

        settings = recover_processing_control_prefix_v1(self.job_root)
        if settings is None:
            raise ReservationConflictError(
                "final reservation capability has no recovered settings authority"
            )
        return self._load_bound_reservation(settings)

    def materialize_for_publication(self) -> PreparedFinalEncodingReservationV1:
        # The live publisher owns the ordering of video, input and final manifest.
        # Running global restart recovery here would commit its next step prematurely.
        settings = ProcessingSettingsArtifactV5.parse(
            _read_bounded_regular_file(
                self.job_root / SETTINGS_ARTIFACT_RELATIVE_PATH,
                max_bytes=SETTINGS_ARTIFACT_MAX_RAW_BYTES,
            )
        )
        return self._load_bound_reservation(settings)

    def _load_bound_reservation(
        self, settings: ProcessingSettingsArtifactV5
    ) -> PreparedFinalEncodingReservationV1:
        reservation = load_final_encoding_reservation_v1(self.job_root, settings)
        if reservation.index.fingerprint != self.index_fingerprint:
            raise ReservationConflictError("final reservation capability changed generation")
        return reservation

    def recover_publication_prefix(self) -> FinalPublicationRecovery:
        return self.materialize().recover_publication_prefix()

    def validate_authenticated_publication(
        self,
        *,
        container_validator: Callable[[Path], bool] | None = None,
    ) -> None:
        reservation = self.materialize()
        final_path, expectation = reservation.authenticated_container_expectation()
        del reservation
        validator = container_validator
        if validator is None:

            def validator(path: Path) -> bool:
                return validate_final_container_v4(path, expectation)

        if not validator(final_path):
            raise ReservationConflictError("active final publication container validation failed")
        self.materialize().validate_authenticated_publication(
            container_validator=lambda _path: True
        )

    def discard_nonsemantic_sibling(self) -> bool:
        return self.materialize().discard_nonsemantic_sibling()

    def discard_prelaunch_invocation(self) -> None:
        self.materialize().discard_prelaunch_invocation()

    def commit_producer_settings(self) -> ProcessingSettingsArtifactV5:
        return self.materialize().commit_producer_settings()

    def complete_final_manifest_from_existing_video(
        self,
        *,
        input_stage_fingerprint: str,
        resolved_encoding_arguments: tuple[str, ...],
        expected_video_identity: PersistedFileIdentityV1,
        expected_video_sha256: str,
        expected_video_byte_count: int,
    ) -> FinalVideoManifestV4:
        reservation = self.materialize()
        try:
            return reservation.complete_final_manifest_from_existing_video(
                input_stage_fingerprint=input_stage_fingerprint,
                resolved_encoding_arguments=resolved_encoding_arguments,
                expected_video_identity=expected_video_identity,
                expected_video_sha256=expected_video_sha256,
                expected_video_byte_count=expected_video_byte_count,
            )
        finally:
            del reservation

    def validate_diagnostics_preflight(
        self,
        *,
        raw_sha256: str,
        fingerprint: str,
    ) -> None:
        reservation = self.materialize()
        try:
            reservation.validate_diagnostics_preflight(
                raw_sha256=raw_sha256,
                fingerprint=fingerprint,
            )
        finally:
            del reservation

    def commit_prune_markers(self) -> tuple[dict, ...]:
        reservation = self.materialize()
        try:
            return reservation.commit_prune_markers()
        finally:
            del reservation

    def commit_payload_pruned_metadata(
        self,
        payload: StreamedCanonicalObject,
    ) -> None:
        reservation = self.materialize()
        try:
            reservation.commit_payload_pruned_metadata(payload)
        finally:
            del reservation

    def retire(self) -> None:
        self.materialize().retire()

    def retire_terminal_failure(self) -> None:
        self.materialize().retire_terminal_failure()


@dataclass(frozen=True, slots=True)
class _ExtentTargetPreflightV1:
    target: Path
    parent_identity: DirectoryIdentity
    publication_method: Literal["replace_placeholder", "replace_existing"]
    existing_identity: PersistedFileIdentityV1 | None
    existing_byte_count: int | None
    existing_raw_sha256: str | None


def _preflight_extent_target_v1(
    *,
    job_root: Path,
    target: Path,
    bind_existing_content: bool,
    cancelled: Callable[[], bool] | None,
) -> _ExtentTargetPreflightV1:
    """Return only scalar target authority; release every scan buffer first."""

    if cancelled is not None and cancelled():
        raise InterruptedError("final reservation target audit was cancelled")
    target_path = job_root / target
    with AcquiredRoot(job_root) as acquired:
        parent_identity = acquired.directory_identity(target_path.parent)
        existing_identity = acquired.persistent_regular_identity(
            target_path,
            missing_ok=True,
        )
    if not isinstance(parent_identity, DirectoryIdentity):
        raise FinalPublicationValidationError(
            "final reservation target parent identity is unsupported"
        )
    if existing_identity is None:
        return _ExtentTargetPreflightV1(
            target=target,
            parent_identity=parent_identity,
            publication_method="replace_placeholder",
            existing_identity=None,
            existing_byte_count=None,
            existing_raw_sha256=None,
        )
    if not bind_existing_content:
        return _ExtentTargetPreflightV1(
            target=target,
            parent_identity=parent_identity,
            publication_method="replace_existing",
            existing_identity=None,
            existing_byte_count=None,
            existing_raw_sha256=None,
        )
    scanned = _scan_fixed_file_v1(
        job_root,
        target_path,
        expected_identity=existing_identity,
        cancelled=cancelled,
    )
    result = _ExtentTargetPreflightV1(
        target=target,
        parent_identity=parent_identity,
        publication_method="replace_existing",
        existing_identity=scanned.identity,
        existing_byte_count=scanned.byte_count,
        existing_raw_sha256=scanned.raw_sha256,
    )
    del scanned
    if cancelled is not None and cancelled():
        raise InterruptedError("final reservation target audit was cancelled")
    return result


def _extent_plan(
    *,
    kind: str,
    target_preflight: _ExtentTargetPreflightV1,
    raw_max: int,
    allocation_unit: int,
    generation: str,
    reservation_id: str,
    publication_generation: str,
) -> dict:
    target = target_preflight.target
    method = target_preflight.publication_method
    identity = target_preflight.existing_identity
    target_sha256 = target_preflight.existing_raw_sha256
    target_byte_count = target_preflight.existing_byte_count
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
        "target_parent_identity": target_preflight.parent_identity.to_object(),
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


def _diagnostics_prune_preflight(
    job_root: Path,
    *,
    expected_frame_count: int,
) -> DiagnosticsMetadataProjection:
    path = Path(job_root) / "04_stereo_diagnostics" / "metadata.json"
    try:
        projection = project_diagnostics_metadata(path).validate_state_contract()
        validate_quality_input_binding(projection)
    except (OSError, ValueError) as error:
        raise FinalPublicationValidationError(
            "diagnostics metadata is not canonical at final preflight"
        ) from error
    if projection.status not in {"complete", "legacy_fast_unavailable"}:
        raise FinalPublicationValidationError(
            "diagnostics metadata is not complete at final preflight"
        )
    if projection.frame_count != expected_frame_count:
        raise FinalPublicationValidationError(
            "diagnostics frame count differs from schema-5 settings"
        )
    return projection


def prepare_final_encoding_reservation_v1(  # noqa: C901
    *,
    job_root: Path,
    settings: ProcessingSettingsArtifactV5,
    publication_generation: str,
    input_manifest_raw_max: int,
    final_manifest_raw_max: int,
    expected_existing_video_identity: ReservationFileIdentity | None = None,
    allocation_unit: int | None = None,
    free_bytes: Callable[[Path], int] | None = None,
    allocation_adapter: AllocationAdapter = platform_allocation_adapter,
    allocation_counter: AllocationCounter | None = platform_allocation_counter,
    reservation_generation: str | None = None,
    input_reservation_id: str | None = None,
    final_reservation_id: str | None = None,
    event: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
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
        if expected_existing_video_identity is None:
            raise ReservationConflictError(
                "final video already exists without replacement authority; preserving it"
            )
        existing_identity = _reservation_identity(final_path)
        if existing_identity != expected_existing_video_identity:
            raise ReservationConflictError("authorized final-video predecessor identity changed")
        publication_method: Literal["replace_placeholder", "replace_existing"] = "replace_existing"
    else:
        if expected_existing_video_identity is not None:
            raise ReservationConflictError("authorized final-video predecessor is missing")
        existing_identity = None
        publication_method = "replace_placeholder"
    settings_path = root / "processing-settings-v5.json"
    settings_raw = _read_bounded_regular_file(
        settings_path,
        max_bytes=SETTINGS_ARTIFACT_MAX_RAW_BYTES,
    )
    if ProcessingSettingsArtifactV5.parse(settings_raw) != settings:
        raise FinalPublicationValidationError("settings bytes differ at reservation preflight")
    raw_sha = hashlib.sha256(settings_raw).hexdigest()
    control = load_job_control(root)
    if (
        control.job_id != metadata["job_id"]
        or control.settings_identity_fingerprint != metadata["settings_identity_fingerprint"]
        or control.job_reservation_generation != metadata["job_reservation_generation"]
        or control.job_terminal_reservation_id != metadata["job_terminal_reservation_id"]
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
    producer_updated_at = _utc_now_usec()
    cleanup_updated_at = _utc_now_usec()
    producer_preview = settings.with_producer_marker(
        generation=publication_generation,
        updated_at=producer_updated_at,
    )
    producer_required = producer_preview != settings
    del producer_preview
    keep_intermediates = values["processing_settings"]["keep_intermediates"]
    frame_count = _settings_frame_count_v1(settings)
    diagnostics_raw_sha256: str | None = None
    diagnostics_fingerprint: str | None = None
    diagnostics_raw_max: int | None = None
    marker_specs: tuple[PlannedPruneMarkerV1, ...] = ()
    if not keep_intermediates:
        diagnostics_projection = _diagnostics_prune_preflight(
            root,
            expected_frame_count=frame_count,
        )
        marker_specs = plan_prune_markers_v1(root)
        payload_pruned_preview = build_payload_pruned_document(
            diagnostics_projection,
            prune_entries=[marker.prune_entry() for marker in marker_specs],
            final_video_manifest_sha256="0" * 64,
        )
        diagnostics_raw_sha256 = diagnostics_projection.raw_sha256
        diagnostics_fingerprint = diagnostics_projection.fingerprint
        diagnostics_raw_max = payload_pruned_preview.raw_byte_count
        del diagnostics_projection, payload_pruned_preview

    job_id = metadata["job_id"]
    settings_identity_fingerprint = metadata["settings_identity_fingerprint"]
    settings_revision = metadata["settings_revision"]
    del values, metadata, settings_raw, control, settings

    # Fixed-file content audits are one serial host phase. Only scalar authority
    # and bounded marker intents remain live while the shared 1 MiB buffer runs.
    producer_preflight = (
        _preflight_extent_target_v1(
            job_root=root,
            target=Path("processing-settings-v5.json"),
            bind_existing_content=False,
            cancelled=cancelled,
        )
        if producer_required
        else None
    )
    input_preflight = _preflight_extent_target_v1(
        job_root=root,
        target=ENCODING_INPUT_MANIFEST_RELATIVE_PATH,
        bind_existing_content=True,
        cancelled=cancelled,
    )
    marker_preflights = tuple(
        _preflight_extent_target_v1(
            job_root=root,
            target=Path(marker.relative_path) / marker.marker_name,
            bind_existing_content=False,
            cancelled=cancelled,
        )
        for marker in marker_specs
    )
    diagnostics_preflight = (
        _preflight_extent_target_v1(
            job_root=root,
            target=Path("04_stereo_diagnostics/metadata.json"),
            bind_existing_content=True,
            cancelled=cancelled,
        )
        if diagnostics_raw_max is not None
        else None
    )
    cleanup_preflight = (
        _preflight_extent_target_v1(
            job_root=root,
            target=Path("processing-settings-v5.json"),
            bind_existing_content=False,
            cancelled=cancelled,
        )
        if not keep_intermediates
        else None
    )
    final_preflight = _preflight_extent_target_v1(
        job_root=root,
        target=FINAL_VIDEO_MANIFEST_RELATIVE_PATH,
        bind_existing_content=True,
        cancelled=cancelled,
    )

    # Reconstruct bounded settings payloads only after every fixed scan buffer
    # and parser projection has been released.
    current_settings_raw = _read_bounded_regular_file(
        settings_path,
        max_bytes=SETTINGS_ARTIFACT_MAX_RAW_BYTES,
    )
    if hashlib.sha256(current_settings_raw).hexdigest() != raw_sha:
        raise FinalPublicationValidationError(
            "settings bytes changed during reservation target audits"
        )
    current_settings = ProcessingSettingsArtifactV5.parse(current_settings_raw)
    del current_settings_raw
    producer_target = current_settings.with_producer_marker(
        generation=publication_generation,
        updated_at=producer_updated_at,
    )
    if (producer_target != current_settings) != producer_required:
        raise FinalPublicationValidationError(
            "settings producer state changed during reservation target audits"
        )
    producer_payload = producer_target.encode() if producer_required else None
    cleanup_payload = (
        producer_target.cleanup_pending(updated_at=cleanup_updated_at).encode()
        if not keep_intermediates
        else None
    )
    del current_settings, producer_target

    producer_plan = (
        _extent_plan(
            kind="producer_settings",
            target_preflight=producer_preflight,
            raw_max=512 * 1024,
            allocation_unit=unit,
            generation=reserve_generation,
            reservation_id=secrets.token_hex(16),
            publication_generation=publication_generation,
        )
        if producer_preflight is not None
        else None
    )
    input_plan = _extent_plan(
        kind="encoding_input_manifest",
        target_preflight=input_preflight,
        raw_max=input_manifest_raw_max,
        allocation_unit=unit,
        generation=reserve_generation,
        reservation_id=input_id,
        publication_generation=publication_generation,
    )
    marker_plans: list[dict] = []
    for marker, target_preflight in zip(
        marker_specs,
        marker_preflights,
        strict=True,
    ):
        plan = _extent_plan(
            kind="prune_marker",
            target_preflight=target_preflight,
            raw_max=len(marker.payload),
            allocation_unit=unit,
            generation=reserve_generation,
            reservation_id=secrets.token_hex(16),
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
    diagnostics_plan = (
        _extent_plan(
            kind="diagnostics_payload_pruned",
            target_preflight=diagnostics_preflight,
            raw_max=diagnostics_raw_max,
            allocation_unit=unit,
            generation=reserve_generation,
            reservation_id=secrets.token_hex(16),
            publication_generation=publication_generation,
        )
        if diagnostics_preflight is not None and diagnostics_raw_max is not None
        else None
    )
    cleanup_plan = (
        _extent_plan(
            kind="cleanup_pending_settings",
            target_preflight=cleanup_preflight,
            raw_max=512 * 1024,
            allocation_unit=unit,
            generation=reserve_generation,
            reservation_id=secrets.token_hex(16),
            publication_generation=publication_generation,
        )
        if cleanup_preflight is not None
        else None
    )
    final_plan = _extent_plan(
        kind="final_video_manifest",
        target_preflight=final_preflight,
        raw_max=final_manifest_raw_max,
        allocation_unit=unit,
        generation=reserve_generation,
        reservation_id=final_id,
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
    validate_final_encoding_entry_order([entry["reservation_kind"] for entry in extent_entries])
    index = FinalEncodingReservationIndexV1.build(
        schema_version=2 if expected_existing_video_identity is not None else 1,
        job_id=job_id,
        settings_identity_fingerprint=settings_identity_fingerprint,
        settings_raw_sha256_at_preflight=raw_sha,
        publication_generation=publication_generation,
        reservation_generation=reserve_generation,
        settings_revision_at_preflight=settings_revision,
        final_video_relative_path=final_relative,
        final_video_publication_method=publication_method,
        final_video_existing_target_identity_at_preflight=existing_identity,
        final_video_target_descriptor_relative_path=target_descriptor_relative.as_posix(),
        sibling_video_temporary_relative_path=sibling_relative,
        sibling_video_parent_identity=_reservation_directory_identity(root),
        diagnostics_metadata_raw_sha256_at_preflight=diagnostics_raw_sha256,
        diagnostics_metadata_fingerprint_at_preflight=diagnostics_fingerprint,
        extent_entries=tuple(extent_entries),
    )
    target_descriptor_raw_bound = _final_video_target_descriptor_raw_bound_v1(index)
    if target_descriptor_raw_bound > FINAL_RESERVATION_DESCRIPTOR_RAW_BYTES:
        raise ReservationConflictError(
            "final-video target descriptor exceeds its frozen raw-byte bound"
        )
    index_path = root / FINAL_ENCODING_RESERVATION_RELATIVE_PATH
    reservation_directory_identity = _reservation_directory_identity(index_path.parent)
    _admit_final_encoding_reserve_v1(
        job_root=root,
        index=index,
        allocation_unit=unit,
        free_bytes=free_bytes,
    )
    publish_write_ahead_intent(
        index_path,
        index.to_object(),
        event=event,
        expected_parent_identity=reservation_directory_identity,
    )

    if event is not None:
        event("final-target:parent-authenticated")
    if publication_method == "replace_placeholder":
        reserved_identity = create_parent_bound_placeholder_v1(
            final_path,
            expected_parent_identity=index.sibling_video_parent_identity,
        )
    else:
        reserved_identity, _reserved_size = observe_parent_bound_regular_v1(
            final_path,
            expected_parent_identity=index.sibling_video_parent_identity,
        )
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
    if len(canonical_json_bytes(target_descriptor.to_object())) > target_descriptor_raw_bound:
        raise ReservationConflictError(
            "final-video target descriptor exceeds its admitted raw-byte bound"
        )
    publish_write_ahead_intent(
        root / target_descriptor_relative,
        target_descriptor.to_object(),
        expected_parent_identity=reservation_directory_identity,
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
            cancelled=cancelled,
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


def _validate_descriptor_temporary_prefix_v1(
    path: Path,
    expected_raw: bytes,
    *,
    label: str,
) -> None:
    temporary_raw = _read_bounded_regular_file(path, max_bytes=max(len(expected_raw), 1))
    if len(temporary_raw) > len(expected_raw) or not expected_raw.startswith(temporary_raw):
        raise ReservationConflictError(f"{label} is not the authoritative descriptor prefix")


def _current_target_descriptor_from_index_v1(
    root: Path,
    index: FinalEncodingReservationIndexV1,
) -> FinalVideoTargetReservationV1:
    final_path = root / index.final_video_relative_path
    if not _regular_nonlink(final_path):
        raise ReservationConflictError(
            "partial final target descriptor has no reserved target authority"
        )
    reserved_identity = _reservation_identity(final_path)
    if index.final_video_publication_method == "replace_placeholder":
        if final_path.stat(follow_symlinks=False).st_size != 0:
            raise ReservationConflictError("partial final target descriptor lost its placeholder")
    else:
        expected = index.final_video_existing_target_identity_at_preflight
        if expected is None or reserved_identity != expected:
            raise ReservationConflictError("partial final target descriptor old target changed")
    return FinalVideoTargetReservationV1.build(
        job_id=index.job_id,
        settings_identity_fingerprint=index.settings_identity_fingerprint,
        settings_raw_sha256_at_preflight=index.settings_raw_sha256_at_preflight,
        settings_revision_at_preflight=index.settings_revision_at_preflight,
        publication_generation=index.publication_generation,
        reservation_generation=index.reservation_generation,
        transaction_index_fingerprint=index.fingerprint,
        final_video_relative_path=index.final_video_relative_path,
        sibling_video_temporary_relative_path=index.sibling_video_temporary_relative_path,
        target_parent_identity=index.sibling_video_parent_identity,
        publication_method=index.final_video_publication_method,
        reserved_target_identity=reserved_identity,
    )


def _publish_current_target_descriptor_v1(
    root: Path,
    index: FinalEncodingReservationIndexV1,
    descriptor: FinalVideoTargetReservationV1,
) -> None:
    descriptor_path = root / index.final_video_target_descriptor_relative_path
    final_path = root / index.final_video_relative_path

    def validate_authority() -> None:
        if _read_bounded_regular_file(
            root / FINAL_ENCODING_RESERVATION_RELATIVE_PATH,
            max_bytes=1024 * 1024,
        ) != canonical_json_bytes(index.to_object()):
            raise ReservationConflictError(
                "final index changed before target descriptor normalization"
            )
        _assert_reserved_final_target_preboundary_v1(final_path, descriptor)

    publish_index_owned_descriptor_v1(
        descriptor_path,
        descriptor.to_object(),
        validate_authority=validate_authority,
    )


def _check_final_candidate_namespace(root: Path, path: Path) -> None:
    control = load_job_control(root)
    candidate_temporary = path.with_name(f".{path.name}.create-new.tmp")
    assert_only_job_control_owned_reservations_v1(
        root,
        control,
        allowed_relative_paths=(candidate_temporary.relative_to(root).as_posix(),),
    )


def _check_final_candidate_target(root: Path, value: FinalEncodingReservationIndexV1) -> None:
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
        raise ReservationConflictError("final encoding candidate has downstream target artifacts")
    final_target = root / value.final_video_relative_path
    if value.final_video_publication_method == "replace_placeholder":
        if final_target.exists() or final_target.is_symlink():
            raise ReservationConflictError("final encoding candidate placeholder already exists")
    elif (
        not _regular_nonlink(final_target)
        or _reservation_identity(final_target)
        != value.final_video_existing_target_identity_at_preflight
    ):
        raise ReservationConflictError("final encoding candidate existing target changed")


def _check_final_candidate_extent(root: Path, entry: dict) -> None:
    if entry["reservation_kind"] == "job_terminal_settings":
        return
    descriptor, source = _indexed_extent_paths(root, entry)
    descriptor_temporary = descriptor.with_name(f".{descriptor.name}.create-new.tmp")
    if any(
        candidate.exists() or candidate.is_symlink()
        for candidate in (descriptor, descriptor_temporary, source)
    ):
        raise ReservationConflictError("final encoding candidate has downstream extent artifacts")
    target = root / entry["target_relative_path"]
    if entry["publication_method"] == "replace_placeholder":
        if target.exists() or target.is_symlink():
            raise ReservationConflictError(
                "final encoding candidate extent placeholder already exists"
            )
        return
    if entry["reservation_kind"] in {"producer_settings", "cleanup_pending_settings"}:
        # These entries bind the single fixed settings artifact via the index
        # hash and revision checked above, not a separate per-extent triple.
        return
    expected = entry["existing_target_identity_at_preflight"]
    if (
        expected is None
        or not _regular_nonlink(target)
        or _reservation_identity(target) != ReservationFileIdentity.from_object(expected)
        or _hash_file(root, target)
        != (
            entry["existing_target_raw_sha256_at_preflight"],
            entry["existing_target_byte_count_at_preflight"],
        )
    ):
        raise ReservationConflictError("final encoding candidate extent target changed")


def _validate_final_index_candidate(root: Path, path: Path, value: object) -> None:
    if not isinstance(value, FinalEncodingReservationIndexV1):
        raise ReservationConflictError("final encoding temporary has the wrong type")
    _validate_index_job_control(root, value)
    value.reject_implicit_video_replacement(root)
    settings_path = root / "processing-settings-v5.json"
    if not _regular_nonlink(settings_path):
        raise ReservationConflictError("final encoding temporary has no durable preflight settings")
    settings_raw = _read_bounded_regular_file(settings_path, max_bytes=512 * 1024)
    settings = ProcessingSettingsArtifactV5.parse(settings_raw)
    metadata = settings.to_object()["metadata"]
    if (
        hashlib.sha256(settings_raw).hexdigest() != value.settings_raw_sha256_at_preflight
        or settings.revision != value.settings_revision_at_preflight
        or metadata["job_id"] != value.job_id
        or metadata["settings_identity_fingerprint"] != value.settings_identity_fingerprint
    ):
        raise ReservationConflictError("final encoding temporary preflight settings changed")
    settings_index = root / (".depth-surge-reservations-v1/settings-transition-reservation-v1.json")
    settings_index_temporary = settings_index.with_name(f".{settings_index.name}.create-new.tmp")
    if any(
        candidate.exists() or candidate.is_symlink()
        for candidate in (settings_index, settings_index_temporary)
    ):
        raise ReservationConflictError("final encoding candidate overlaps a settings authority")
    _check_final_candidate_namespace(root, path)
    _check_final_candidate_target(root, value)
    for entry in value.extent_entries:
        _check_final_candidate_extent(root, entry)


def recover_final_encoding_index_pair_v1(
    job_root: Path,
) -> FinalEncodingReservationIndexV1 | None:
    root = Path(job_root).resolve()
    path = root / FINAL_ENCODING_RESERVATION_RELATIVE_PATH

    def parse(raw: bytes) -> FinalEncodingReservationIndexV1:
        return FinalEncodingReservationIndexV1.from_object(
            strict_canonical_json_loads(raw, max_bytes=256 * 1024)
        )

    # A committed index wins, but even a redundant temporary must survive a refusal.
    temporary = path.with_name(f".{path.name}.create-new.tmp")
    if path.is_file() and (temporary.exists() or temporary.is_symlink()):
        _load_final_encoding_index_readonly_v1(root).reject_implicit_video_replacement(root)
    recovered = recover_control_artifact_pair_v1(
        path,
        max_bytes=256 * 1024,
        parser=parse,
        prepublish_validate=lambda value: _validate_final_index_candidate(root, path, value),
        preunlink_validate=lambda: _check_final_candidate_namespace(root, path),
    )
    if recovered is None:
        return None
    if not isinstance(recovered, FinalEncodingReservationIndexV1):
        raise ReservationConflictError("final encoding control type changed")
    _validate_index_job_control(root, recovered)
    recovered.reject_implicit_video_replacement(root)
    target_descriptor = PurePosixPath(recovered.final_video_target_descriptor_relative_path)
    assert_index_owned_reservation_namespace_v1(
        root,
        load_job_control(root),
        index_relative_path=FINAL_ENCODING_RESERVATION_RELATIVE_PATH.as_posix(),
        extent_entries=recovered.extent_entries,
        extra_relative_paths=(
            target_descriptor.as_posix(),
            (target_descriptor.parent / f".{target_descriptor.name}.create-new.tmp").as_posix(),
            recovered.sibling_video_temporary_relative_path,
        ),
    )
    return recovered


def _load_final_encoding_index_readonly_v1(
    job_root: Path,
) -> FinalEncodingReservationIndexV1:
    """Load the committed final index without reconciling its temporary pair."""

    root = Path(job_root).resolve()
    path = root / FINAL_ENCODING_RESERVATION_RELATIVE_PATH
    if not _regular_nonlink(path):
        raise ReservationConflictError("terminal retirement index is missing or unsafe")
    if path.stat(follow_symlinks=False).st_nlink != 1:
        raise ReservationConflictError("terminal retirement index is not single-link")
    index = FinalEncodingReservationIndexV1.from_object(
        _load_reservation_json(path, max_bytes=256 * 1024)
    )
    _validate_index_job_control(root, index)
    target_descriptor = PurePosixPath(index.final_video_target_descriptor_relative_path)
    assert_index_owned_reservation_namespace_v1(
        root,
        load_job_control(root),
        index_relative_path=FINAL_ENCODING_RESERVATION_RELATIVE_PATH.as_posix(),
        extent_entries=index.extent_entries,
        extra_relative_paths=(
            target_descriptor.as_posix(),
            (target_descriptor.parent / f".{target_descriptor.name}.create-new.tmp").as_posix(),
            index.sibling_video_temporary_relative_path,
        ),
    )
    return index


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
    if not (descriptor_path.exists() or descriptor_path.is_symlink()):
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
                ) == expected_identity and _hash_file(root, target_path) == (
                    entry["existing_target_raw_sha256_at_preflight"],
                    entry["existing_target_byte_count_at_preflight"],
                )
        if not source_path.exists() and committed:
            return None
        if not source_path.exists() and not target_is_preflight and target_path.exists():
            try:
                if kind == "encoding_input_manifest":
                    _stream_input_manifest_identity(
                        root,
                        target_path,
                        expected_frame_count=_load_settings_frame_count_v1(root),
                    )
                elif kind == "final_video_manifest":
                    parse_final_video_manifest_v4(
                        _read_bounded_regular_file(
                            target_path,
                            max_bytes=FINAL_VIDEO_MANIFEST_MAX_RAW_BYTES,
                        )
                    )
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


def _validate_partial_indexed_extent_descriptor_v1(
    *,
    root: Path,
    index: FinalEncodingReservationIndexV1,
    entry: dict,
    allocation_adapter: AllocationAdapter,
    allocation_counter: AllocationCounter | None,
) -> ReservedArtifactTransaction:
    """Rebuild descriptor bytes without publishing and authenticate the temp prefix."""

    descriptor_path, _source_path = _indexed_extent_paths(root, entry)
    temporary = descriptor_path.with_name(f".{descriptor_path.name}.create-new.tmp")
    validated = False

    def validate_only(
        path: Path,
        payload: dict,
        *,
        validate_authority: Callable[[], None],
        event=None,
        expected_parent_identity=None,
    ) -> None:
        nonlocal validated
        if path != descriptor_path or event is not None:
            raise ReservationConflictError("descriptor normalization target changed")
        expected_raw = canonical_json_bytes(payload)
        _validate_descriptor_temporary_prefix_v1(
            temporary,
            expected_raw,
            label=f"{entry['reservation_kind']} descriptor temporary",
        )
        validate_authority()
        validated = True

    transaction = ReservedArtifactTransaction.create(
        job_root=root,
        index_relative_path=FINAL_ENCODING_RESERVATION_RELATIVE_PATH,
        target_relative_path=Path(entry["target_relative_path"]),
        reservation_kind=entry["reservation_kind"],
        payload_raw_max=entry["payload_raw_max"],
        allocation_unit=entry["allocation_unit"],
        allocation_adapter=allocation_adapter,
        generation=entry["reservation_generation"],
        reservation_id=entry["reservation_id"],
        allocation_counter=allocation_counter,
        expected_publication_method=entry["publication_method"],
        descriptor_publisher=validate_only,
        materialize_missing=False,
    )
    if not validated:
        raise ReservationConflictError("descriptor prefix validation did not run")
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


def _validate_settings_gap_descriptor(
    index: FinalEncodingReservationIndexV1,
    control: JobControlV1,
    entry: dict,
    transaction: ReservedArtifactTransaction,
) -> None:
    kind = entry["reservation_kind"]
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


def _validate_settings_gap_artifact(
    index: FinalEncodingReservationIndexV1, kind: str, artifact: ProcessingSettingsArtifactV5
) -> None:
    metadata = artifact.to_object()["metadata"]
    if (
        metadata["job_id"] != index.job_id
        or metadata["settings_identity_fingerprint"] != index.settings_identity_fingerprint
    ):
        raise ReservationConflictError("indexed settings gap belongs to another job")
    if kind == "producer_settings" and (
        metadata["final_media_publication_generation"] != index.publication_generation
    ):
        raise ReservationConflictError("producer settings gap has another generation")
    if kind == "cleanup_pending_settings" and metadata["cleanup_status"] != "pending":
        raise ReservationConflictError("cleanup settings gap is not pending")
    if kind == "job_terminal_settings" and artifact.status not in {
        "completed",
        "failed",
    }:
        raise ReservationConflictError("terminal settings gap is nonterminal")


def _settings_gap_candidate(
    root: Path,
    index: FinalEncodingReservationIndexV1,
    control: JobControlV1,
    entry: dict,
    allocation_adapter: AllocationAdapter,
    allocation_counter: AllocationCounter | None,
) -> tuple[ReservedArtifactTransaction, ProcessingSettingsArtifactV5] | None:
    kind = entry["reservation_kind"]
    if kind not in {
        "producer_settings",
        "cleanup_pending_settings",
        "job_terminal_settings",
    }:
        return None
    descriptor_path = root / entry["descriptor_relative_path"]
    if not descriptor_path.is_file() or descriptor_path.is_symlink():
        return None
    transaction = ReservedArtifactTransaction.load(
        descriptor_path,
        allocation_adapter=allocation_adapter,
        allocation_counter=allocation_counter,
    )
    _validate_settings_gap_descriptor(index, control, entry, transaction)
    if not transaction.source_path.is_file() or transaction.source_path.is_symlink():
        return None
    payload = _settings_payload_intent(transaction)
    if payload is None:
        return None
    artifact = ProcessingSettingsArtifactV5.parse(payload)
    _validate_settings_gap_artifact(index, kind, artifact)
    if transaction.validate_physical_phase(payload) == "replacement-gap":
        return transaction, artifact
    return None


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
    candidates: list[tuple[ReservedArtifactTransaction, ProcessingSettingsArtifactV5]] = []
    for entry in index.extent_entries:
        candidate = _settings_gap_candidate(
            root, index, control, entry, allocation_adapter, allocation_counter
        )
        if candidate is not None:
            candidates.append(candidate)
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
    if (
        not isinstance(name, str)
        or not re.fullmatch(r"\.depth-surge-prune-v2-[0-9a-f]{32}\.marker", name, flags=re.ASCII)
        or not isinstance(relative_path, str)
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


def _load_diagnostics_projection(root: Path) -> DiagnosticsMetadataProjection:
    path = root / "04_stereo_diagnostics" / "metadata.json"
    try:
        projection = project_diagnostics_metadata(path).validate_state_contract()
    except (OSError, ValueError) as error:
        raise ReservationConflictError("diagnostics metadata is noncanonical") from error
    return projection


def _check_prune_predecessor_binding(projection: DiagnosticsMetadataProjection) -> None:
    try:
        projection.validate_state_contract()
        validate_quality_input_binding(projection)
    except (OSError, ValueError) as error:
        raise ReservationConflictError(
            "payload-pruned diagnostics predecessor is invalid"
        ) from error


def _payload_prune_predecessor(
    transaction: ReservedArtifactTransaction,
) -> DiagnosticsMetadataProjection:
    projection: DiagnosticsMetadataProjection | None = None
    if _regular_nonlink(transaction.target_path):
        try:
            projection = project_diagnostics_metadata(transaction.target_path)
        except (OSError, ValueError) as error:
            raise ReservationConflictError(
                "diagnostics target cannot rebuild payload-pruned intent"
            ) from error
    elif _regular_nonlink(transaction.source_path):
        prefix = PreparedFinalEncodingReservationV1._canonical_prefix_byte_count(
            transaction.source_path
        )
        if prefix == transaction.descriptor.payload_raw_max:
            try:
                projection = project_diagnostics_metadata(
                    transaction.source_path,
                    raw_byte_count=prefix,
                )
            except (OSError, ValueError) as error:
                raise ReservationConflictError(
                    "diagnostics source intent is noncanonical"
                ) from error
    if projection is None:
        raise ReservationConflictError(
            "payload-pruned intent has no replayable diagnostics predecessor"
        )
    _check_prune_predecessor_binding(projection)
    return projection


def _payload_prune_successor(
    root: Path,
    index: FinalEncodingReservationIndexV1,
    projection: DiagnosticsMetadataProjection,
    expected_entries: list[dict],
    final_sha256: str,
) -> StreamedCanonicalObject:
    if projection.status == "payload_pruned":
        try:
            _validate_payload_pruned_projection(root, projection)
        except RuntimeError as error:
            raise ReservationConflictError(
                "payload-pruned successor evidence is invalid"
            ) from error
        fields = projection.static_fields
        pruned_from = fields.get("pruned_from")
        if (
            fields.get("prune_entries") != expected_entries
            or fields.get("final_video_manifest_sha256") != final_sha256
            or not isinstance(pruned_from, dict)
            or pruned_from.get("previous_diagnostics_fingerprint")
            != index.diagnostics_metadata_fingerprint_at_preflight
        ):
            raise ReservationConflictError(
                "payload-pruned successor belongs to another reservation"
            )
        document = projection.rebuild(dict(fields))
    else:
        if (
            projection.status not in {"complete", "legacy_fast_unavailable"}
            or projection.raw_sha256 != index.diagnostics_metadata_raw_sha256_at_preflight
            or projection.fingerprint != index.diagnostics_metadata_fingerprint_at_preflight
        ):
            raise ReservationConflictError("diagnostics predecessor differs from final preflight")
        document = build_payload_pruned_document(
            projection,
            prune_entries=expected_entries,
            final_video_manifest_sha256=final_sha256,
        )
    return document


def _rebuild_payload_pruned_document_v1(
    root: Path,
    index: FinalEncodingReservationIndexV1,
    transaction: ReservedArtifactTransaction,
) -> StreamedCanonicalObject:
    """Rebuild one indexed payload-pruned intent without materializing frame names."""

    marker_entries = [
        entry for entry in index.extent_entries if entry["reservation_kind"] == "prune_marker"
    ]
    expected_entries = [_marker_payload_from_entry(entry)[0] for entry in marker_entries]
    final_manifest = root / FINAL_VIDEO_MANIFEST_RELATIVE_PATH
    if not _regular_nonlink(final_manifest):
        raise ReservationConflictError("payload-pruned intent has no committed final manifest")
    final_sha256, _final_count = _hash_file(root, final_manifest)
    projection = _payload_prune_predecessor(transaction)
    document = _payload_prune_successor(root, index, projection, expected_entries, final_sha256)
    if document.raw_byte_count != transaction.descriptor.payload_raw_max:
        raise ReservationConflictError(
            "rebuilt payload-pruned intent differs from its indexed bound"
        )
    return document


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
        or metadata["final_media_publication_generation"] != index.publication_generation
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
        diagnostics = _load_diagnostics_projection(root)
        if diagnostics.status != "payload_pruned":
            raise ReservationConflictError(
                "completed retirement lacks payload-pruned successor evidence"
            )
        try:
            _validate_payload_pruned_projection(root, diagnostics)
        except RuntimeError as error:
            raise ReservationConflictError(
                "completed retirement payload-pruned evidence is invalid"
            ) from error
        expected_prune_entries = [_marker_payload_from_entry(entry)[0] for entry in marker_entries]
        if diagnostics.static_fields.get("prune_entries") != expected_prune_entries:
            raise ReservationConflictError(
                "completed retirement prune authorization differs from the index"
            )
        final_manifest = root / FINAL_VIDEO_MANIFEST_RELATIVE_PATH
        final_manifest_sha256, _count = _hash_file(root, final_manifest)
        if diagnostics.static_fields.get("final_video_manifest_sha256") != final_manifest_sha256:
            raise ReservationConflictError(
                "completed retirement diagnostics bind another final manifest"
            )


def _retired_settings_entry_is_committed(
    root: Path, index: FinalEncodingReservationIndexV1, kind: str
) -> bool:
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
            and metadata["final_media_publication_generation"] == index.publication_generation
        )
    return metadata["cleanup_status"] not in {
        "not_started",
        "not_requested",
    }


def _retired_prune_entry_is_committed(root: Path, entry: dict) -> bool:
    kind = entry["reservation_kind"]
    try:
        diagnostics = _load_diagnostics_projection(root)
    except (OSError, ReservationConflictError):
        return False
    if diagnostics.status != "payload_pruned":
        return False
    try:
        _validate_payload_pruned_projection(root, diagnostics)
    except RuntimeError:
        return False
    if kind == "diagnostics_payload_pruned":
        return True
    expected, _payload = _marker_payload_from_entry(entry)
    return expected in diagnostics.static_fields.get("prune_entries", [])


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
            _stream_input_manifest_identity(
                root,
                target,
                expected_frame_count=_load_settings_frame_count_v1(root),
            )
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
        return _retired_settings_entry_is_committed(root, index, kind)
    if kind in {"prune_marker", "diagnostics_payload_pruned"}:
        return _retired_prune_entry_is_committed(root, entry)
    return False


@dataclass(frozen=True)
class _RetirementPathV1:
    path: Path
    identity: ReservationFileIdentity
    parent_identity: DirectoryIdentity


@dataclass(frozen=True)
class _FailedExtentRetirementV1:
    position: int
    source: _RetirementPathV1 | None
    target_placeholder: _RetirementPathV1 | None
    descriptor_temporary: _RetirementPathV1 | None
    descriptor: _RetirementPathV1 | None


@dataclass(frozen=True)
class _FailedRetirementPlanV1:
    sibling: _RetirementPathV1 | None
    index_temporary: _RetirementPathV1 | None
    extents: tuple[_FailedExtentRetirementV1, ...]
    final_placeholder: _RetirementPathV1 | None
    target_descriptor_temporary: _RetirementPathV1 | None
    target_descriptor: _RetirementPathV1 | None
    index: _RetirementPathV1


@dataclass(frozen=True)
class _CompletedRetirementPlanV1:
    sibling: _RetirementPathV1 | None
    index_temporary: _RetirementPathV1 | None
    target_descriptor_temporary: _RetirementPathV1 | None
    target_descriptor: _RetirementPathV1 | None
    index: _RetirementPathV1


def _snapshot_retirement_path_v1(
    path: Path,
    *,
    label: str,
    expected_identity: ReservationFileIdentity | None = None,
    expected_parent_identity: DirectoryIdentity | None = None,
) -> _RetirementPathV1 | None:
    if not (path.exists() or path.is_symlink()):
        return None
    if not _regular_nonlink(path):
        raise ReservationConflictError(f"{label} is unsafe")
    identity = _reservation_identity(path)
    if identity.link_count != 1:
        raise ReservationConflictError(f"{label} is not single-link")
    if expected_identity is not None and identity != expected_identity:
        raise ReservationConflictError(f"{label} identity changed")
    parent_identity = _reservation_directory_identity(path.parent)
    if expected_parent_identity is not None and parent_identity != expected_parent_identity:
        raise ReservationConflictError(f"{label} parent identity changed")
    return _RetirementPathV1(path, identity, parent_identity)


def _unlink_retirement_path_v1(
    item: _RetirementPathV1 | None,
    *,
    sync_parent: bool = True,
) -> None:
    if item is None:
        return
    _unlink_safe_control_temporary(
        item.path,
        expected_identity=item.identity,
        expected_parent_identity=item.parent_identity,
        sync_parent=sync_parent,
    )
    if sync_parent:
        _sync_parent(item.path.parent)
    if item.path.exists() or item.path.is_symlink():
        raise ReservationConflictError(f"retirement deletion did not persist: {item.path}")


def _validate_terminal_revision_v1(
    index: FinalEncodingReservationIndexV1,
    settings: ProcessingSettingsArtifactV5,
    *,
    expected_status: Literal["in_progress", "completed", "failed"],
) -> dict[str, bool]:
    metadata = settings.to_object()["metadata"]
    if (
        settings.status != expected_status
        or metadata["job_id"] != index.job_id
        or metadata["settings_identity_fingerprint"] != index.settings_identity_fingerprint
    ):
        raise ReservationConflictError(
            f"final retirement requires exact {expected_status} settings"
        )
    extent_kinds = {entry["reservation_kind"] for entry in index.extent_entries}
    producer_consumed = bool(
        "producer_settings" in extent_kinds
        and metadata["final_media_producer_contract_version"] == 4
        and metadata["final_media_publication_generation"] == index.publication_generation
    )
    cleanup_consumed = bool(
        "cleanup_pending_settings" in extent_kinds
        and metadata["cleanup_status"] not in {"not_started", "not_requested"}
    )
    if expected_status == "in_progress" and cleanup_consumed:
        raise ReservationConflictError("prelaunch retirement cannot cross cleanup settings")
    expected_revision = (
        index.settings_revision_at_preflight
        + int(producer_consumed)
        + int(cleanup_consumed)
        + int(expected_status in {"completed", "failed"})
    )
    if metadata["settings_revision"] != expected_revision:
        raise ReservationConflictError(
            f"{expected_status} settings revision does not match the indexed prefix"
        )
    return {
        "producer_settings": producer_consumed,
        "cleanup_pending_settings": cleanup_consumed,
    }


def _validate_target_descriptor_against_index_v1(
    index: FinalEncodingReservationIndexV1,
    descriptor: FinalVideoTargetReservationV1,
) -> None:
    if (
        descriptor.transaction_index_fingerprint != index.fingerprint
        or descriptor.job_id != index.job_id
        or descriptor.settings_identity_fingerprint != index.settings_identity_fingerprint
        or descriptor.settings_raw_sha256_at_preflight != index.settings_raw_sha256_at_preflight
        or descriptor.settings_revision_at_preflight != index.settings_revision_at_preflight
        or descriptor.publication_generation != index.publication_generation
        or descriptor.reservation_generation != index.reservation_generation
        or descriptor.final_video_relative_path != index.final_video_relative_path
        or descriptor.sibling_video_temporary_relative_path
        != index.sibling_video_temporary_relative_path
        or descriptor.target_parent_identity != index.sibling_video_parent_identity
        or descriptor.publication_method != index.final_video_publication_method
    ):
        raise ReservationConflictError("final target descriptor differs from its index")
    if descriptor.publication_method == "replace_existing" and (
        descriptor.reserved_target_identity
        != index.final_video_existing_target_identity_at_preflight
    ):
        raise ReservationConflictError("final target descriptor changed the old target identity")


def _snapshot_completed_retirement_plan_v1(
    root: Path,
    index: FinalEncodingReservationIndexV1,
    settings: ProcessingSettingsArtifactV5,
    *,
    expected_target_descriptor: FinalVideoTargetReservationV1 | None,
) -> _CompletedRetirementPlanV1:
    persisted_index = _load_final_encoding_index_readonly_v1(root)
    if persisted_index != index:
        raise ReservationConflictError("final reservation index changed before retirement")
    _validate_terminal_revision_v1(index, settings, expected_status="completed")
    _validate_completed_descriptorless_retirement_v1(root, index, settings)

    for entry in index.extent_entries:
        if entry["reservation_kind"] == "job_terminal_settings":
            continue
        descriptor_path, source = _indexed_extent_paths(root, entry)
        descriptor_temporary = descriptor_path.with_name(f".{descriptor_path.name}.create-new.tmp")
        if any(
            path.exists() or path.is_symlink()
            for path in (descriptor_path, descriptor_temporary, source)
        ):
            raise ReservationConflictError(
                f"{entry['reservation_kind']} extent remains active at completed retirement"
            )

    sibling = _snapshot_retirement_path_v1(
        root / index.sibling_video_temporary_relative_path,
        label="completed retirement sibling",
        expected_parent_identity=index.sibling_video_parent_identity,
    )
    index_path = root / FINAL_ENCODING_RESERVATION_RELATIVE_PATH
    index_temporary = _snapshot_retirement_path_v1(
        index_path.with_name(f".{index_path.name}.create-new.tmp"),
        label="final index temporary",
    )
    target_path = root / index.final_video_target_descriptor_relative_path
    target_temporary_path = target_path.with_name(f".{target_path.name}.create-new.tmp")
    target_temporary = _snapshot_retirement_path_v1(
        target_temporary_path,
        label="final target descriptor temporary",
    )
    target = _snapshot_retirement_path_v1(
        target_path,
        label="final target descriptor",
    )
    if target is not None:
        descriptor = FinalVideoTargetReservationV1.from_object(
            _load_reservation_json(target_path, max_bytes=1024 * 1024)
        )
        _validate_target_descriptor_against_index_v1(index, descriptor)
        if expected_target_descriptor is not None and descriptor != expected_target_descriptor:
            raise ReservationConflictError(
                "final target descriptor changed before completed retirement"
            )
        if target is not None and target_temporary is not None:
            _validate_descriptor_temporary_prefix_v1(
                target_temporary_path,
                canonical_json_bytes(descriptor.to_object()),
                label="final target descriptor temporary",
            )
    elif target_temporary is not None:
        descriptor = FinalVideoTargetReservationV1.from_object(
            _load_reservation_json(target_temporary_path, max_bytes=1024 * 1024)
        )
        _validate_target_descriptor_against_index_v1(index, descriptor)
    index_snapshot = _snapshot_retirement_path_v1(
        index_path,
        label="final encoding index",
    )
    if index_snapshot is None:
        raise ReservationConflictError("final encoding index disappeared during preflight")
    return _CompletedRetirementPlanV1(
        sibling=sibling,
        index_temporary=index_temporary,
        target_descriptor_temporary=target_temporary,
        target_descriptor=target,
        index=index_snapshot,
    )


def _recover_completed_final_retirement_v1(
    root: Path,
    index: FinalEncodingReservationIndexV1,
    settings: ProcessingSettingsArtifactV5,
    *,
    expected_target_descriptor: FinalVideoTargetReservationV1 | None,
    event: Callable[[str], None] | None,
    crash_at: str | None,
) -> None:
    plan = _snapshot_completed_retirement_plan_v1(
        root,
        index,
        settings,
        expected_target_descriptor=expected_target_descriptor,
    )

    _unlink_retirement_path_v1(plan.sibling)
    _unlink_retirement_path_v1(plan.index_temporary)
    _unlink_retirement_path_v1(plan.target_descriptor_temporary, sync_parent=False)
    _unlink_retirement_path_v1(plan.target_descriptor, sync_parent=False)
    if crash_at == "after-target-descriptor-unlink-before-sync":
        raise InjectedFinalPublicationCrash(crash_at)
    _sync_parent(plan.index.path.parent)
    if event is not None:
        event("reservation:target-descriptor-retired")
    if crash_at == "after-target-descriptor-retirement":
        raise InjectedFinalPublicationCrash(crash_at)

    index_path = root / FINAL_ENCODING_RESERVATION_RELATIVE_PATH
    forbidden = [
        root / index.sibling_video_temporary_relative_path,
        index_path.with_name(f".{index_path.name}.create-new.tmp"),
        root / index.final_video_target_descriptor_relative_path,
    ]
    forbidden.append(forbidden[-1].with_name(f".{forbidden[-1].name}.create-new.tmp"))
    for entry in index.extent_entries:
        if entry["reservation_kind"] == "job_terminal_settings":
            continue
        descriptor, source = _indexed_extent_paths(root, entry)
        forbidden.extend(
            (
                descriptor.with_name(f".{descriptor.name}.create-new.tmp"),
                descriptor,
                source,
            )
        )
    if any(path.exists() or path.is_symlink() for path in forbidden):
        raise ReservationConflictError(
            "completed retirement suffix is not empty before index retirement"
        )

    _unlink_retirement_path_v1(plan.index, sync_parent=False)
    if crash_at == "after-index-unlink-before-sync":
        raise InjectedFinalPublicationCrash(crash_at)
    _sync_parent(plan.index.path.parent)
    if plan.index.path.exists() or plan.index.path.is_symlink():
        raise ReservationConflictError("final reservation index retirement did not persist")
    if event is not None:
        event("reservation:index-retired")


def _group_final_extents(
    index: FinalEncodingReservationIndexV1,
) -> tuple[dict | None, list[dict], dict | None, dict | None]:
    singletons: dict[str, dict] = {}
    markers: list[dict] = []
    for entry in index.extent_entries:
        kind = entry.get("reservation_kind")
        if kind == "prune_marker":
            markers.append(entry)
        elif kind in {
            "producer_settings",
            "diagnostics_payload_pruned",
            "cleanup_pending_settings",
        }:
            if kind in singletons:
                raise ReservationConflictError("final index has duplicate singleton extents")
            singletons[kind] = entry
    return (
        singletons.get("producer_settings"),
        markers,
        singletons.get("diagnostics_payload_pruned"),
        singletons.get("cleanup_pending_settings"),
    )


def _load_committed_prune_entries(
    root: Path,
    index: FinalEncodingReservationIndexV1,
    marker_entries: list[dict],
    diagnostics_entry: dict | None,
) -> tuple[dict, ...] | None:
    diagnostics_projection: DiagnosticsMetadataProjection | None = None
    diagnostics_path = root / "04_stereo_diagnostics" / "metadata.json"
    if (marker_entries or diagnostics_entry is not None) and _regular_nonlink(diagnostics_path):
        diagnostics_projection = _load_diagnostics_projection(root)
    committed_prune_entries: tuple[dict, ...] | None = None
    if diagnostics_projection is not None and diagnostics_projection.status == "payload_pruned":
        try:
            _validate_payload_pruned_projection(root, diagnostics_projection)
        except RuntimeError as error:
            raise ReservationConflictError(
                "payload-pruned diagnostics retained evidence is invalid"
            ) from error
        expected_entries = [_marker_payload_from_entry(entry)[0] for entry in marker_entries]
        fields = diagnostics_projection.static_fields
        pruned_from = fields.get("pruned_from")
        if (
            fields.get("prune_entries") != expected_entries
            or not isinstance(pruned_from, dict)
            or pruned_from.get("previous_diagnostics_fingerprint")
            != index.diagnostics_metadata_fingerprint_at_preflight
        ):
            raise ReservationConflictError(
                "payload-pruned diagnostics belong to another final reservation"
            )
        committed_prune_entries = tuple(dict(entry) for entry in expected_entries)
        del expected_entries, fields, pruned_from
    del diagnostics_projection
    return committed_prune_entries


def _load_or_create_final_target(
    root: Path, index: FinalEncodingReservationIndexV1
) -> FinalVideoTargetReservationV1:
    target_path = root / index.final_video_target_descriptor_relative_path
    if target_path.exists() or target_path.is_symlink():
        target = FinalVideoTargetReservationV1.from_object(
            _load_reservation_json(target_path, max_bytes=1024 * 1024)
        )
    else:
        final_path = root / index.final_video_relative_path
        if index.final_video_publication_method == "replace_placeholder":
            if not (final_path.exists() or final_path.is_symlink()):
                create_parent_bound_placeholder_v1(
                    final_path,
                    expected_parent_identity=index.sibling_video_parent_identity,
                )
            reserved_identity, reserved_size = observe_parent_bound_regular_v1(
                final_path,
                expected_parent_identity=index.sibling_video_parent_identity,
            )
            if reserved_size != 0:
                raise ReservationConflictError(
                    "descriptorless final-video placeholder is not adoptable"
                )
        else:
            expected = index.final_video_existing_target_identity_at_preflight
            reserved_identity, _reserved_size = observe_parent_bound_regular_v1(
                final_path,
                expected_parent_identity=index.sibling_video_parent_identity,
            )
            if expected is None or reserved_identity != expected:
                raise ReservationConflictError("descriptorless existing final-video target changed")
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

        def validate_target_descriptor_authority() -> None:
            index_path = root / FINAL_ENCODING_RESERVATION_RELATIVE_PATH
            expected_index_raw = canonical_json_bytes(index.to_object())
            if _read_bounded_regular_file(index_path, max_bytes=1024 * 1024) != expected_index_raw:
                raise ReservationConflictError(
                    "final index changed before target descriptor publication"
                )
            _assert_reserved_final_target_preboundary_v1(final_path, target)

        publish_index_owned_descriptor_v1(
            target_path,
            target.to_object(),
            validate_authority=validate_target_descriptor_authority,
        )
    _validate_target_descriptor_against_index_v1(index, target)
    return target


def _observe_final_target(
    root: Path, index: FinalEncodingReservationIndexV1, target: FinalVideoTargetReservationV1
) -> None:
    current_final = root / index.final_video_relative_path
    if current_final.exists() or current_final.is_symlink():
        current_identity = _reservation_identity(current_final)
        if current_identity == target.reserved_target_identity:
            _assert_reserved_final_target_preboundary_v1(current_final, target)
        elif (
            target.publication_method == "replace_placeholder"
            and _regular_nonlink(current_final)
            and current_final.stat().st_size == 0
        ):
            raise ReservationConflictError("final placeholder identity differs from descriptor")


def _load_final_prune_markers(
    root: Path,
    index: FinalEncodingReservationIndexV1,
    marker_entries: list[dict],
    committed_prune_entries: tuple[dict, ...] | None,
    allocation_adapter: AllocationAdapter,
    allocation_counter: AllocationCounter | None,
) -> tuple[tuple[dict, ReservedArtifactTransaction | None, bytes], ...]:
    prune_transactions: list[tuple[dict, ReservedArtifactTransaction | None, bytes]] = []
    for marker_index, entry in enumerate(marker_entries):
        prune_entry, payload = _marker_payload_from_entry(entry)
        marker_target = root / entry["target_relative_path"]
        marker_committed = (
            _regular_nonlink(marker_target)
            and _hash_file(root, marker_target)
            == (hashlib.sha256(payload).hexdigest(), len(payload))
        ) or (
            committed_prune_entries is not None
            and marker_index < len(committed_prune_entries)
            and committed_prune_entries[marker_index] == prune_entry
            and not (marker_target.exists() or marker_target.is_symlink())
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
    return tuple(prune_transactions)


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
    settings_authority = _scalarize_final_settings_for_load_v1(settings)
    del settings
    index = recover_final_encoding_index_pair_v1(root)
    if index is None:
        raise ReservationConflictError("final encoding reservation is missing")
    if (
        index.job_id != settings_authority.job_id
        or index.settings_identity_fingerprint != settings_authority.settings_identity_fingerprint
        or settings_authority.final_media_publication_generation
        not in {None, index.publication_generation}
        or index.final_video_relative_path != settings_authority.expected_final_relative_path
        or index.sibling_video_parent_identity != _reservation_directory_identity(root)
    ):
        raise ReservationConflictError("active final index belongs to different job settings")
    producer_entry, marker_entries, diagnostics_entry, cleanup_entry = _group_final_extents(index)
    producer_consumed = (
        settings_authority.final_media_publication_generation == index.publication_generation
    )
    cleanup_consumed = settings_authority.cleanup_consumed
    del settings_authority
    committed_prune_entries = _load_committed_prune_entries(
        root, index, marker_entries, diagnostics_entry
    )
    diagnostics_consumed = committed_prune_entries is not None
    target = _load_or_create_final_target(root, index)
    _observe_final_target(root, index, target)
    producer_transaction = (
        _load_indexed_extent(
            root=root,
            index=index,
            entry=producer_entry,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
            committed=producer_consumed,
        )
        if producer_entry is not None
        else None
    )
    prune_transactions = _load_final_prune_markers(
        root, index, marker_entries, committed_prune_entries, allocation_adapter, allocation_counter
    )
    diagnostics_transaction = (
        _load_indexed_extent(
            root=root,
            index=index,
            entry=diagnostics_entry,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
            committed=diagnostics_consumed,
        )
        if diagnostics_entry is not None
        else None
    )
    cleanup_transaction = (
        _load_indexed_extent(
            root=root,
            index=index,
            entry=cleanup_entry,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
            committed=cleanup_consumed,
        )
        if cleanup_entry is not None
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
        prune_marker_transactions=prune_transactions,
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
    index = _load_final_encoding_index_readonly_v1(root)
    if settings.status == "failed":
        _recover_failed_final_retirement_v1(
            root,
            index,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
            event=event,
            crash_at=crash_at,
        )
        return True
    _recover_completed_final_retirement_v1(
        root,
        index,
        settings,
        expected_target_descriptor=None,
        event=event,
        crash_at=crash_at,
    )
    return True


def _load_retirement_extent_descriptor(
    path: Path, index: FinalEncodingReservationIndexV1, entry: dict
) -> ReservationDescriptor:
    descriptor = ReservationDescriptor.from_object(
        _load_reservation_json(path, max_bytes=1024 * 1024)
    )
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
    return descriptor


def _snapshot_failed_extent(
    root: Path, index: FinalEncodingReservationIndexV1, position: int, entry: dict
) -> _FailedExtentRetirementV1:
    kind = entry["reservation_kind"]
    descriptor_path, source_path = _indexed_extent_paths(root, entry)
    descriptor_temporary_path = descriptor_path.with_name(f".{descriptor_path.name}.create-new.tmp")
    descriptor_temporary = _snapshot_retirement_path_v1(
        descriptor_temporary_path,
        label=f"{kind} descriptor temporary",
    )
    descriptor_snapshot = _snapshot_retirement_path_v1(
        descriptor_path,
        label=f"{kind} descriptor",
    )
    source_snapshot: _RetirementPathV1 | None = None
    target_snapshot: _RetirementPathV1 | None = None
    if descriptor_snapshot is not None:
        descriptor = _load_retirement_extent_descriptor(descriptor_path, index, entry)
        if descriptor_temporary is not None:
            _validate_descriptor_temporary_prefix_v1(
                descriptor_temporary_path,
                canonical_json_bytes(descriptor.to_object()),
                label=f"{kind} descriptor temporary",
            )
        source_snapshot = _snapshot_retirement_path_v1(
            source_path,
            label=f"{kind} source",
            expected_identity=descriptor.filesystem_identity,
        )
        target_path = root / descriptor.target_relative_path
        if (
            descriptor.publication_method == "replace_placeholder"
            and descriptor.placeholder_identity is not None
            and (target_path.exists() or target_path.is_symlink())
            and _reservation_identity(target_path) == descriptor.placeholder_identity
        ):
            if target_path.stat(follow_symlinks=False).st_size != 0:
                raise ReservationConflictError(f"unused {kind} placeholder is nonempty")
            target_snapshot = _snapshot_retirement_path_v1(
                target_path,
                label=f"{kind} placeholder",
                expected_identity=descriptor.placeholder_identity,
                expected_parent_identity=descriptor.target_parent_identity,
            )
    elif source_path.exists() or source_path.is_symlink():
        raise ReservationConflictError(f"{kind} source appeared without its descriptor")
    elif descriptor_temporary is not None:
        _load_retirement_extent_descriptor(descriptor_temporary_path, index, entry)
    return _FailedExtentRetirementV1(
        position=position,
        source=source_snapshot,
        target_placeholder=target_snapshot,
        descriptor_temporary=descriptor_temporary,
        descriptor=descriptor_snapshot,
    )


def _snapshot_failed_final_placeholder(
    root: Path,
    index: FinalEncodingReservationIndexV1,
    target_descriptor: FinalVideoTargetReservationV1,
) -> _RetirementPathV1 | None:
    final_path = root / index.final_video_relative_path
    if (
        target_descriptor.publication_method == "replace_placeholder"
        and (final_path.exists() or final_path.is_symlink())
        and _reservation_identity(final_path) == target_descriptor.reserved_target_identity
    ):
        if final_path.stat(follow_symlinks=False).st_size != 0:
            raise ReservationConflictError("unused final placeholder is nonempty")
        return _snapshot_retirement_path_v1(
            final_path,
            label="final video placeholder",
            expected_identity=target_descriptor.reserved_target_identity,
            expected_parent_identity=target_descriptor.target_parent_identity,
        )
    return None


def _snapshot_failed_final_target(
    root: Path,
    index: FinalEncodingReservationIndexV1,
    target_descriptor: FinalVideoTargetReservationV1 | None,
) -> tuple[_RetirementPathV1 | None, _RetirementPathV1 | None, _RetirementPathV1 | None]:
    target_descriptor_path = root / index.final_video_target_descriptor_relative_path
    target_descriptor_temporary_path = target_descriptor_path.with_name(
        f".{target_descriptor_path.name}.create-new.tmp"
    )
    target_descriptor_temporary = _snapshot_retirement_path_v1(
        target_descriptor_temporary_path,
        label="final target descriptor temporary",
    )
    target_descriptor_snapshot = _snapshot_retirement_path_v1(
        target_descriptor_path,
        label="final target descriptor",
    )
    final_placeholder: _RetirementPathV1 | None = None
    if target_descriptor_snapshot is not None:
        current_target_descriptor = FinalVideoTargetReservationV1.from_object(
            _load_reservation_json(target_descriptor_path, max_bytes=1024 * 1024)
        )
        if target_descriptor is None or current_target_descriptor != target_descriptor:
            raise ReservationConflictError("final target descriptor changed during preflight")
        if target_descriptor_temporary is not None:
            _validate_descriptor_temporary_prefix_v1(
                target_descriptor_temporary_path,
                canonical_json_bytes(current_target_descriptor.to_object()),
                label="final target descriptor temporary",
            )
        final_placeholder = _snapshot_failed_final_placeholder(root, index, target_descriptor)
    elif target_descriptor_temporary is not None:
        temporary_target_descriptor = FinalVideoTargetReservationV1.from_object(
            _load_reservation_json(
                target_descriptor_temporary_path,
                max_bytes=1024 * 1024,
            )
        )
        _validate_target_descriptor_against_index_v1(
            index,
            temporary_target_descriptor,
        )
        if target_descriptor is not None and temporary_target_descriptor != target_descriptor:
            raise ReservationConflictError(
                "final target descriptor temporary changed during preflight"
            )
    elif target_descriptor is not None:
        raise ReservationConflictError("final target descriptor disappeared during preflight")

    return final_placeholder, target_descriptor_temporary, target_descriptor_snapshot


def _snapshot_failed_retirement_plan_v1(
    root: Path,
    index: FinalEncodingReservationIndexV1,
    *,
    target_descriptor: FinalVideoTargetReservationV1 | None,
) -> _FailedRetirementPlanV1:
    """Freeze every delete identity after the read-only semantic preflight."""

    sibling = _snapshot_retirement_path_v1(
        root / index.sibling_video_temporary_relative_path,
        label="indexed sibling",
        expected_parent_identity=index.sibling_video_parent_identity,
    )
    index_path = root / FINAL_ENCODING_RESERVATION_RELATIVE_PATH
    index_temporary_path = index_path.with_name(f".{index_path.name}.create-new.tmp")
    index_temporary = _snapshot_retirement_path_v1(
        index_temporary_path,
        label="final index temporary",
    )

    extent_plans: list[_FailedExtentRetirementV1] = []
    for position, entry in enumerate(index.extent_entries):
        if entry["reservation_kind"] == "job_terminal_settings":
            continue
        extent_plans.append(_snapshot_failed_extent(root, index, position, entry))

    final_placeholder, target_descriptor_temporary, target_descriptor_snapshot = (
        _snapshot_failed_final_target(root, index, target_descriptor)
    )
    index_snapshot = _snapshot_retirement_path_v1(
        index_path,
        label="final encoding index",
    )
    if index_snapshot is None:
        raise ReservationConflictError("final encoding index disappeared during preflight")
    return _FailedRetirementPlanV1(
        sibling=sibling,
        index_temporary=index_temporary,
        extents=tuple(extent_plans),
        final_placeholder=final_placeholder,
        target_descriptor_temporary=target_descriptor_temporary,
        target_descriptor=target_descriptor_snapshot,
        index=index_snapshot,
    )


def _published_failed_retirement_target(
    final: Path, target_descriptor: FinalVideoTargetReservationV1
) -> bool:
    if final.exists() or final.is_symlink():
        if _reservation_identity(final) != target_descriptor.reserved_target_identity:
            if (
                target_descriptor.publication_method == "replace_placeholder"
                and _regular_nonlink(final)
                and final.stat().st_size == 0
            ):
                raise ReservationConflictError("final placeholder identity changed")
            if not _regular_nonlink(final) or final.stat().st_nlink != 1:
                raise ReservationConflictError("published final target is unsafe")
            return True
        else:
            _assert_reserved_final_target_preboundary_v1(final, target_descriptor)
    elif target_descriptor.publication_method == "replace_existing":
        raise ReservationConflictError("existing final target is missing")
    return False


def _observe_failed_target_temporary(
    root: Path, index: FinalEncodingReservationIndexV1, target_descriptor_temporary: Path
) -> tuple[FinalVideoTargetReservationV1, FinalVideoTargetReservationV1 | None]:
    temporary_raw = _read_bounded_regular_file(
        target_descriptor_temporary,
        max_bytes=1024 * 1024,
    )
    try:
        complete_temporary = FinalVideoTargetReservationV1.from_object(
            strict_canonical_json_loads(
                temporary_raw,
                max_bytes=1024 * 1024,
            )
        )
    except (CanonicalJsonError, ReservationConflictError):
        complete_temporary = None
    if complete_temporary is not None:
        _validate_target_descriptor_against_index_v1(index, complete_temporary)
        return complete_temporary, None
    else:
        target_descriptor = _current_target_descriptor_from_index_v1(root, index)
        _validate_descriptor_temporary_prefix_v1(
            target_descriptor_temporary,
            canonical_json_bytes(target_descriptor.to_object()),
            label="final target descriptor temporary",
        )
        return target_descriptor, target_descriptor


def _validate_failed_retirement_sibling(
    sibling: Path,
    final: Path,
    target_descriptor: FinalVideoTargetReservationV1 | None,
    final_published: bool,
) -> None:
    if not _regular_nonlink(sibling):
        raise ReservationConflictError("indexed sibling is unsafe at failed retirement")
    if target_descriptor is None:
        raise ReservationConflictError("final-video sibling has no authoritative target descriptor")
    if not (final.exists() or final.is_symlink()):
        raise ReservationConflictError("failed retirement crossed a final-video replacement gap")
    if not _regular_nonlink(final):
        raise ReservationConflictError("final target is unsafe at failed retirement")
    if (
        not final_published
        and _reservation_identity(final) != target_descriptor.reserved_target_identity
    ):
        raise ReservationConflictError(
            "failed retirement found a sibling beside a replaced final target"
        )


def _descriptorless_final_is_published(final: Path, index: FinalEncodingReservationIndexV1) -> bool:
    if not _regular_nonlink(final):
        raise ReservationConflictError("descriptorless final target is unsafe")
    if index.final_video_publication_method == "replace_placeholder":
        if final.stat().st_size == 0:
            raise ReservationConflictError("descriptorless final target has no retirement evidence")
        if final.stat().st_nlink != 1:
            raise ReservationConflictError("descriptorless published final is unsafe")
        return True
    else:
        expected = index.final_video_existing_target_identity_at_preflight
        if expected is None or _reservation_identity(final) != expected:
            if final.stat().st_nlink != 1:
                raise ReservationConflictError("descriptorless published final is unsafe")
            return True

    return False


def _observe_failed_retirement_target(
    root: Path, index: FinalEncodingReservationIndexV1
) -> tuple[FinalVideoTargetReservationV1 | None, bool, FinalVideoTargetReservationV1 | None]:
    final = root / index.final_video_relative_path
    target_descriptor_path = root / index.final_video_target_descriptor_relative_path
    target_descriptor_temporary = target_descriptor_path.with_name(
        f".{target_descriptor_path.name}.create-new.tmp"
    )
    if target_descriptor_temporary.exists() or target_descriptor_temporary.is_symlink():
        if not _regular_nonlink(target_descriptor_temporary):
            raise ReservationConflictError("final target descriptor temporary is unsafe")
    target_descriptor: FinalVideoTargetReservationV1 | None = None
    normalize_target = None
    final_published = False
    if target_descriptor_path.exists() or target_descriptor_path.is_symlink():
        if not _regular_nonlink(target_descriptor_path):
            raise ReservationConflictError("final target descriptor is unsafe")
        target_descriptor = FinalVideoTargetReservationV1.from_object(
            _load_reservation_json(target_descriptor_path, max_bytes=1024 * 1024)
        )
        _validate_target_descriptor_against_index_v1(index, target_descriptor)
        if target_descriptor_temporary.exists() or target_descriptor_temporary.is_symlink():
            _validate_descriptor_temporary_prefix_v1(
                target_descriptor_temporary,
                canonical_json_bytes(target_descriptor.to_object()),
                label="final target descriptor temporary",
            )
        final_published = _published_failed_retirement_target(final, target_descriptor)
    elif target_descriptor_temporary.exists() or target_descriptor_temporary.is_symlink():
        target_descriptor, normalize_target = _observe_failed_target_temporary(
            root, index, target_descriptor_temporary
        )

    sibling = root / index.sibling_video_temporary_relative_path
    if sibling.exists() or sibling.is_symlink():
        _validate_failed_retirement_sibling(sibling, final, target_descriptor, final_published)
    elif target_descriptor is None and (final.exists() or final.is_symlink()):
        final_published = _descriptorless_final_is_published(final, index)
    return target_descriptor, final_published, normalize_target


def _failed_input_intent_state(
    root: Path,
    source_path: Path,
    transaction: ReservedArtifactTransaction,
    final: Path,
    target_descriptor: FinalVideoTargetReservationV1 | None,
) -> str | None:
    prefix = PreparedFinalEncodingReservationV1._canonical_prefix_byte_count(source_path)
    if prefix is None:
        return "partial"
    if prefix == 0:
        return "zero"
    try:
        _fingerprint, digest, count = _stream_input_manifest_identity(
            root,
            source_path,
            raw_byte_count=prefix,
            logical_byte_count=(
                transaction.descriptor.logical_byte_count
                if source_path.stat().st_size == transaction.descriptor.logical_byte_count
                else None
            ),
            expected_frame_count=_load_settings_frame_count_v1(root),
        )
        phase = transaction.validate_streamed_physical_phase(
            prefix_byte_count=count,
            prefix_sha256=digest,
        )
    except FinalPublicationValidationError:
        return None
    final_is_reserved = bool(
        target_descriptor is not None
        and _regular_nonlink(final)
        and _reservation_identity(final) == target_descriptor.reserved_target_identity
    )
    if phase == "full-source" and final_is_reserved:
        return "staged"
    raise ReservationConflictError("failed retirement crossed a durable encoding-input intent")


def _failed_final_manifest_intent_state(
    source_path: Path, transaction: ReservedArtifactTransaction
) -> str | None:
    payload = PreparedFinalEncodingReservationV1._bounded_payload_intent(transaction)
    if payload is None:
        prefix = PreparedFinalEncodingReservationV1._canonical_prefix_byte_count(source_path)
        return "zero" if prefix == 0 else "partial"
    try:
        parse_final_video_manifest_v4(payload)
    except FinalPublicationValidationError:
        return None
    if PreparedFinalEncodingReservationV1._bounded_transaction_has_durable_intent(
        transaction,
        payload,
    ):
        raise ReservationConflictError("failed retirement crossed a durable final-manifest intent")
    return None


def _check_failed_prune_intent(
    root: Path, index: FinalEncodingReservationIndexV1, transaction: ReservedArtifactTransaction
) -> None:
    prefix = PreparedFinalEncodingReservationV1._canonical_prefix_byte_count(
        transaction.source_path
    )
    if prefix is None:
        raise ReservationConflictError("failed retirement found a foreign payload-pruned prefix")
    if prefix == 0:
        return
    document = _rebuild_payload_pruned_document_v1(root, index, transaction)
    state = PreparedFinalEncodingReservationV1._streamed_intent_state(
        transaction,
        document,
    )
    if state == "partial":
        return
    raise ReservationConflictError(
        "failed retirement crossed a durable diagnostics_payload_pruned intent"
    )


def _check_failed_metadata_intent(
    root: Path,
    index: FinalEncodingReservationIndexV1,
    entry: dict,
    transaction: ReservedArtifactTransaction,
) -> None:
    kind = entry["reservation_kind"]
    payload: bytes | None
    if kind == "prune_marker":
        _prune_entry, payload = _marker_payload_from_entry(entry)
    elif kind in {"producer_settings", "cleanup_pending_settings"}:
        payload = _settings_payload_intent(transaction)
    elif kind == "diagnostics_payload_pruned":
        _check_failed_prune_intent(root, index, transaction)
        return
    else:
        payload = None
    if payload is not None and (
        PreparedFinalEncodingReservationV1._bounded_transaction_has_durable_intent(
            transaction,
            payload,
        )
    ):
        raise ReservationConflictError(f"failed retirement crossed a durable {kind} intent")


@dataclass(frozen=True)
class _FailedExtentNames:
    descriptor: Path
    temporary: Path
    source: Path
    target: Path
    descriptor_present: bool
    temporary_present: bool
    source_present: bool
    target_present: bool

    @classmethod
    def observe(cls, root: Path, entry: dict) -> "_FailedExtentNames":
        descriptor, source = _indexed_extent_paths(root, entry)
        temporary = descriptor.with_name(f".{descriptor.name}.create-new.tmp")
        target = root / entry["target_relative_path"]
        return cls(
            descriptor,
            temporary,
            source,
            target,
            descriptor.exists() or descriptor.is_symlink(),
            temporary.exists() or temporary.is_symlink(),
            source.exists() or source.is_symlink(),
            target.exists() or target.is_symlink(),
        )


@dataclass
class _FailedRetirementExtents:
    root: Path
    index: FinalEncodingReservationIndexV1
    allocation_adapter: AllocationAdapter
    allocation_counter: AllocationCounter | None
    target_descriptor: FinalVideoTargetReservationV1 | None
    settings_entry_consumed: dict[str, bool]
    normalize_entries: list[dict]

    def observe(self, entry: dict) -> str | None:
        kind = entry["reservation_kind"]
        names = _FailedExtentNames.observe(self.root, entry)
        if names.temporary_present and not _regular_nonlink(names.temporary):
            raise ReservationConflictError(f"{kind} descriptor temporary is unsafe")
        if not names.descriptor_present and not (names.temporary_present and names.source_present):
            return self._descriptorless_state(entry, names)
        transaction = self._active_transaction(entry, names)
        descriptor = transaction.descriptor
        if names.source_present:
            if not _regular_nonlink(names.source):
                raise ReservationConflictError(f"{kind} source is unsafe")
            if _reservation_identity(names.source) != descriptor.filesystem_identity:
                raise ReservationConflictError(f"{kind} source identity changed")
        if descriptor.publication_method == "replace_placeholder":
            committed = self._placeholder_committed(entry, names, descriptor)
        else:
            committed = self._existing_committed(entry, names, descriptor)
        if committed:
            return "committed"
        if not names.source_present:
            return "unused"
        return self._source_intent_state(entry, names.source, transaction)

    def _source_intent_state(
        self, entry: dict, source: Path, transaction: ReservedArtifactTransaction
    ) -> str | None:
        kind = entry["reservation_kind"]
        if kind == "encoding_input_manifest":
            return _failed_input_intent_state(
                self.root,
                source,
                transaction,
                self.root / self.index.final_video_relative_path,
                self.target_descriptor,
            )
        if kind == "final_video_manifest":
            return _failed_final_manifest_intent_state(source, transaction)
        _check_failed_metadata_intent(self.root, self.index, entry, transaction)
        return None

    def _active_transaction(
        self, entry: dict, names: _FailedExtentNames
    ) -> ReservedArtifactTransaction:
        kind = entry["reservation_kind"]
        if not names.descriptor_present:
            transaction = _validate_partial_indexed_extent_descriptor_v1(
                root=self.root,
                index=self.index,
                entry=entry,
                allocation_adapter=self.allocation_adapter,
                allocation_counter=self.allocation_counter,
            )
            self.normalize_entries.append(entry)
            return transaction
        if not _regular_nonlink(names.descriptor):
            raise ReservationConflictError(f"{kind} descriptor is unsafe")
        loaded = _load_indexed_extent(
            root=self.root,
            index=self.index,
            entry=entry,
            allocation_adapter=self.allocation_adapter,
            allocation_counter=self.allocation_counter,
        )
        if loaded is None:
            raise ReservationConflictError(f"{kind} descriptor disappeared")
        return loaded

    def _descriptorless_state(self, entry: dict, names: _FailedExtentNames) -> str:
        kind = entry["reservation_kind"]
        if names.temporary_present:
            _load_retirement_extent_descriptor(names.temporary, self.index, entry)
        if names.source_present:
            raise ReservationConflictError(f"{kind} source has no descriptor")
        if not names.target_present:
            if entry["publication_method"] == "replace_existing":
                raise ReservationConflictError("descriptorless existing extent target is missing")
            return "unused"
        expected = (
            entry["existing_target_identity_at_preflight"]
            if entry["publication_method"] == "replace_existing"
            else None
        )
        if (
            expected is not None
            and _regular_nonlink(names.target)
            and _reservation_identity(names.target) == ReservationFileIdentity.from_object(expected)
            and _hash_file(self.root, names.target)
            == (
                entry["existing_target_raw_sha256_at_preflight"],
                entry["existing_target_byte_count_at_preflight"],
            )
        ):
            return "unused"
        if _descriptorless_failed_entry_is_committed_v1(self.root, self.index, entry):
            return "committed"
        raise ReservationConflictError("descriptorless extent target has no retirement evidence")

    def _require_committed(self, entry: dict, names: _FailedExtentNames) -> None:
        kind = entry["reservation_kind"]
        if names.source_present:
            raise ReservationConflictError(f"{kind} source and committed target are both present")
        if not _descriptorless_failed_entry_is_committed_v1(self.root, self.index, entry):
            raise ReservationConflictError(
                f"failed retirement crossed an unauthenticated {kind} target"
            )

    def _placeholder_committed(
        self, entry: dict, names: _FailedExtentNames, descriptor: ReservationDescriptor
    ) -> bool:
        kind = entry["reservation_kind"]
        if not names.target_present:
            if names.source_present:
                raise ReservationConflictError(f"failed retirement found a {kind} replacement gap")
            return False
        if not _regular_nonlink(names.target):
            raise ReservationConflictError(f"{kind} placeholder is unsafe")
        identity = _reservation_identity(names.target)
        if identity == descriptor.filesystem_identity:
            self._require_committed(entry, names)
            return True
        if (
            descriptor.placeholder_identity is None
            or identity != descriptor.placeholder_identity
            or names.target.stat().st_size != 0
        ):
            raise ReservationConflictError(f"unused {kind} placeholder changed")
        return False

    def _existing_committed(
        self, entry: dict, names: _FailedExtentNames, descriptor: ReservationDescriptor
    ) -> bool:
        kind = entry["reservation_kind"]
        if not names.target_present or not _regular_nonlink(names.target):
            raise ReservationConflictError(f"existing {kind} target is missing or unsafe")
        identity = _reservation_identity(names.target)
        committed = identity == descriptor.filesystem_identity
        if committed:
            self._require_committed(entry, names)
        settings_successor = (
            kind in {"producer_settings", "cleanup_pending_settings"}
            and entry["target_relative_path"] == "processing-settings-v5.json"
        )
        if settings_successor and self.settings_entry_consumed[kind]:
            if names.source_present:
                raise ReservationConflictError(f"consumed {kind} unexpectedly retains its source")
            return True
        if (
            not committed
            and not settings_successor
            and (
                descriptor.existing_target_identity_at_preflight is None
                or identity != descriptor.existing_target_identity_at_preflight
                or _hash_file(self.root, names.target)
                != (
                    descriptor.existing_target_raw_sha256_at_preflight,
                    descriptor.existing_target_byte_count_at_preflight,
                )
            )
        ):
            raise ReservationConflictError(f"existing {kind} target changed")
        return committed

    def normalize(self, normalize_target: FinalVideoTargetReservationV1 | None) -> None:
        if normalize_target is not None:
            _publish_current_target_descriptor_v1(self.root, self.index, normalize_target)
        for entry in self.normalize_entries:
            transaction = _load_indexed_extent(
                root=self.root,
                index=self.index,
                entry=entry,
                allocation_adapter=self.allocation_adapter,
                allocation_counter=self.allocation_counter,
            )
            if transaction is None:
                raise ReservationConflictError(
                    f"{entry['reservation_kind']} descriptor normalization disappeared"
                )


def _check_failed_publication_suffix(
    final_published: bool,
    sibling: Path,
    input_retirement_state: str | None,
    final_manifest_retirement_state: str | None,
) -> None:
    if final_published:
        if input_retirement_state != "committed":
            raise ReservationConflictError("published final video lacks a committed input manifest")
        if final_manifest_retirement_state not in {
            "zero",
            "partial",
            "unused",
            "committed",
        }:
            raise ReservationConflictError(
                "published final video has an invalid final-manifest suffix"
            )
        if sibling.exists() or sibling.is_symlink():
            if final_manifest_retirement_state not in {"zero", "partial"}:
                raise ReservationConflictError("post-boundary sibling has no retry-phase evidence")
    elif input_retirement_state == "committed" or final_manifest_retirement_state == "committed":
        raise ReservationConflictError(
            "committed publication manifest exists before the video boundary"
        )


def _assert_failed_retirement_has_no_forward_obligation_v1(
    root: Path,
    index: FinalEncodingReservationIndexV1,
    *,
    allocation_adapter: AllocationAdapter,
    allocation_counter: AllocationCounter | None,
    expected_status: Literal["in_progress", "failed"],
) -> _FailedRetirementPlanV1:
    """Fail closed if a terminal transition crossed a durable artifact intent."""

    terminal_settings = ProcessingSettingsArtifactV5.parse(
        _read_bounded_regular_file(
            root / "processing-settings-v5.json",
            max_bytes=512 * 1024,
        )
    )
    settings_entry_consumed = _validate_terminal_revision_v1(
        index,
        terminal_settings,
        expected_status=expected_status,
    )

    target_descriptor, final_published, normalize_target = _observe_failed_retirement_target(
        root, index
    )
    audit = _FailedRetirementExtents(
        root,
        index,
        allocation_adapter,
        allocation_counter,
        target_descriptor,
        settings_entry_consumed,
        [],
    )
    states: dict[str, str] = {}
    for entry in index.extent_entries:
        kind = entry["reservation_kind"]
        if kind == "job_terminal_settings":
            continue
        state = audit.observe(entry)
        if kind in {"encoding_input_manifest", "final_video_manifest"} and state is not None:
            states[kind] = state
    _check_failed_publication_suffix(
        final_published,
        root / index.sibling_video_temporary_relative_path,
        states.get("encoding_input_manifest"),
        states.get("final_video_manifest"),
    )
    if normalize_target is not None or audit.normalize_entries:
        audit.normalize(normalize_target)
        return _assert_failed_retirement_has_no_forward_obligation_v1(
            root,
            index,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
            expected_status=expected_status,
        )
    return _snapshot_failed_retirement_plan_v1(
        root,
        index,
        target_descriptor=target_descriptor,
    )


def _recover_failed_final_retirement_v1(  # noqa: C901
    root: Path,
    index: FinalEncodingReservationIndexV1,
    *,
    allocation_adapter: AllocationAdapter,
    allocation_counter: AllocationCounter | None,
    event: Callable[[str], None] | None,
    crash_at: str | None,
    expected_status: Literal["in_progress", "failed"] = "failed",
) -> None:
    """Continue terminal-failure cleanup monotonically without materializing work."""

    plan = _assert_failed_retirement_has_no_forward_obligation_v1(
        root,
        index,
        allocation_adapter=allocation_adapter,
        allocation_counter=allocation_counter,
        expected_status=expected_status,
    )

    def emit(name: str) -> None:
        if event is not None:
            event(name)
        if crash_at == name:
            raise InjectedFinalPublicationCrash(name)

    # Phase 2: reconcile scratch namespace only after every path passed phase 1.
    _unlink_retirement_path_v1(plan.sibling)
    _unlink_retirement_path_v1(plan.index_temporary)
    emit("failure-retirement:sibling")

    # Phase 3: retire every unused payload/source path before any descriptor.
    for extent in plan.extents:
        _unlink_retirement_path_v1(extent.source)
        emit(f"failure-retirement:{extent.position}:source")
        _unlink_retirement_path_v1(extent.target_placeholder)
        emit(f"failure-retirement:{extent.position}:placeholder")
    _unlink_retirement_path_v1(plan.final_placeholder)

    # Phase 4: descriptor temporaries and finals are retired in index order.
    for extent in plan.extents:
        _unlink_retirement_path_v1(
            extent.descriptor_temporary,
            sync_parent=False,
        )
        _unlink_retirement_path_v1(extent.descriptor, sync_parent=False)
        emit(f"failure-retirement:{extent.position}:descriptor")

    # Phase 5: the final-video descriptor pair follows every extent descriptor.
    _unlink_retirement_path_v1(
        plan.target_descriptor_temporary,
        sync_parent=False,
    )
    _unlink_retirement_path_v1(plan.target_descriptor, sync_parent=False)
    emit("failure-retirement:descriptors-unlinked-before-sync")
    _sync_parent(plan.index.path.parent)
    emit("failure-retirement:target-descriptor")

    retired_paths = [plan.sibling, plan.index_temporary, plan.final_placeholder]
    for extent in plan.extents:
        retired_paths.extend(
            (
                extent.source,
                extent.target_placeholder,
                extent.descriptor_temporary,
                extent.descriptor,
            )
        )
    retired_paths.extend((plan.target_descriptor_temporary, plan.target_descriptor))
    if any(
        item is not None and (item.path.exists() or item.path.is_symlink())
        for item in retired_paths
    ):
        raise ReservationConflictError(
            "failed retirement suffix is not empty before index retirement"
        )

    # Phase 7: index-last is the only authority-retirement boundary.
    _unlink_retirement_path_v1(plan.index, sync_parent=False)
    emit("failure-retirement:index-unlinked-before-sync")
    _sync_parent(plan.index.path.parent)
    if plan.index.path.exists() or plan.index.path.is_symlink():
        raise ReservationConflictError("failed retirement index deletion did not persist")
    emit("failure-retirement:index")


def _sync_parent(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _scan_fixed_file_v1(
    job_root: Path,
    path: Path,
    *,
    expected_identity: PersistedFileIdentityV1 | None = None,
    expected_raw_sha256: str | None = None,
    expected_byte_count: int | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> AuthenticatedFixedFileV1:
    try:
        return scan_fixed_regular_v1(
            job_root,
            path,
            expected_identity=expected_identity,
            expected_raw_sha256=expected_raw_sha256,
            expected_byte_count=expected_byte_count,
            cancelled=cancelled,
        )
    except InterruptedError:
        raise
    except (NoFollowPathError, FileNotFoundError, OSError) as error:
        raise FinalPublicationValidationError("fixed file failed no-follow acquisition") from error


def _hash_file(job_root: Path, path: Path) -> tuple[str, int]:
    scanned = _scan_fixed_file_v1(job_root, path)
    return scanned.raw_sha256, scanned.byte_count


def _hash_open_file(handle: io.FileIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    buffer = bytearray(FINAL_FIXED_TARGET_HASH_STREAM_BYTES)
    view = memoryview(buffer)
    handle.seek(0)
    try:
        while True:
            measured = handle.readinto(view)
            if not measured:
                break
            digest.update(view[:measured])
            byte_count += measured
    finally:
        view.release()
    return digest.hexdigest(), byte_count


def _validate_open_container_v1(
    path: Path, handle: BinaryIO, validator: Callable[[Path], bool]
) -> bool:
    if os.name == "nt":
        return validator(path)
    if sys.platform.startswith("linux") and Path("/proc").is_dir():
        return validate_open_linux_container(handle, validator)
    raise FinalPublicationValidationError(
        "container validation cannot bind an opened file on this platform"
    )


@dataclass(frozen=True, slots=True)
class ValidatedFinalContainerV1:
    identity: PersistedFileIdentityV1
    raw_sha256: str
    byte_count: int


def validate_final_container_evidence_v1(
    job_root: Path,
    path: Path,
    container_validator: Callable[[Path], bool],
) -> ValidatedFinalContainerV1:
    """Validate and hash the exact opened sibling generation."""

    candidate = Path(path)
    try:
        with AcquiredRoot(job_root) as acquired:
            root_identity = acquired.persistent_identity
            parent_identity = acquired.directory_identity(candidate.parent)
            path_identity = acquired.persistent_regular_identity(candidate)
            with acquired.open_regular_locked(candidate) as handle:
                frozen_identity = file_identity_from_open_handle(handle)
                if frozen_identity.link_count != 1:
                    raise FinalPublicationValidationError("sibling video is not single-link")
                if path_identity != frozen_identity:
                    raise FinalPublicationValidationError(
                        "sibling video identity changed before validation"
                    )
                before = os.fstat(handle.fileno())
                if not _validate_open_container_v1(candidate, handle, container_validator):
                    raise FinalPublicationValidationError(
                        "sibling video failed container validation"
                    )
                raw_sha256, byte_count = _hash_open_file(handle)
                after = os.fstat(handle.fileno())
                if (
                    int(after.st_size) != int(before.st_size)
                    or int(after.st_mtime_ns) != int(before.st_mtime_ns)
                    or int(after.st_ctime_ns) != int(before.st_ctime_ns)
                    or file_identity_from_open_handle(handle) != frozen_identity
                    or acquired.persistent_regular_identity(candidate) != frozen_identity
                ):
                    raise FinalPublicationValidationError(
                        "sibling video changed during container validation"
                    )
            if (
                acquired.directory_identity(candidate.parent) != parent_identity
                or acquired.persistent_identity != root_identity
            ):
                raise FinalPublicationValidationError(
                    "sibling namespace changed during container validation"
                )
            return ValidatedFinalContainerV1(
                identity=frozen_identity,
                raw_sha256=raw_sha256,
                byte_count=byte_count,
            )
    except InterruptedError:
        raise
    except (NoFollowPathError, FileNotFoundError, OSError) as error:
        raise FinalPublicationValidationError(
            "sibling video container validation failed no-follow acquisition"
        ) from error


_INPUT_MANIFEST_ALGORITHM_PREFIX = b'{"algorithm_version":"encoding-input-sequence-v'
_ENCODING_CONTROL_FRAME_CAP = 4_194_304
_MAX_SUPPORTED_IMAGE2_INDEX = 2_147_483_646
_ENCODING_MANIFEST_STREAM_BYTES = 64 * 1024
_ENCODING_MANIFEST_VALUE_MAX_BYTES = 1024 * 1024
_ENCODING_MANIFEST_TOKEN_DELIMITERS = re.compile(rb'["\\{}]')
_FRAME_STEM = re.compile(r"frame_([0-9]+)", flags=re.ASCII)


@dataclass(frozen=True, slots=True)
class _EncodingInputManifestProjection:
    fingerprint: str
    raw_sha256: str
    raw_byte_count: int
    mode: Literal["direct_stereo", "assembled_vr"]
    frame_count: int
    first_index: int
    left_parent: PurePosixPath | None
    right_parent: PurePosixPath | None
    frames_parent: PurePosixPath | None
    streamed_invocation: StreamedVrInvocationV1 | None
    streamed_output: StreamedVrOutputEvidenceV1 | None
    raw_pixel_format: Literal["rgb24"] | None = None


class _EncodingManifestReader:
    """Bounded-buffer reader that hashes only consumed manifest bytes."""

    def __init__(self, handle: BinaryIO, *, raw_byte_count: int | None) -> None:
        self.handle = handle
        self.raw_limit = raw_byte_count
        self.buffer = b""
        self.position = 0
        self.loaded = 0
        self.consumed = 0
        self.raw_digest = hashlib.sha256()
        self.self_digest = hashlib.sha256()
        self.include_in_self_digest = False

    def _fill(self) -> bool:
        if self.position < len(self.buffer):
            return True
        maximum = _ENCODING_MANIFEST_STREAM_BYTES
        if self.raw_limit is not None:
            remaining = self.raw_limit - self.loaded
            if remaining <= 0:
                self.buffer = b""
                self.position = 0
                return False
            maximum = min(maximum, remaining)
        chunk = self.handle.read(maximum)
        if not chunk:
            self.buffer = b""
            self.position = 0
            return False
        self.buffer = chunk
        self.position = 0
        self.loaded += len(chunk)
        return True

    def peek_byte(self) -> int | None:
        if not self._fill():
            return None
        return self.buffer[self.position]

    def take(self, count: int) -> bytes:
        if count < 0:
            raise ValueError("manifest read count must be nonnegative")
        output = bytearray()
        while count:
            if not self._fill():
                raise FinalPublicationValidationError("encoding input manifest ended unexpectedly")
            available = min(count, len(self.buffer) - self.position)
            chunk = self.buffer[self.position : self.position + available]
            self.position += available
            self.consumed += available
            self.raw_digest.update(chunk)
            if self.include_in_self_digest:
                self.self_digest.update(chunk)
            output.extend(chunk)
            count -= available
        return bytes(output)

    def take_byte(self) -> int:
        return self.take(1)[0]

    def expect(self, expected: bytes) -> None:
        if self.take(len(expected)) != expected:
            raise FinalPublicationValidationError(
                "encoding input manifest canonical grammar differs"
            )

    def json_string_token(self) -> bytes:
        return self._json_token(object_value=False)

    def json_object_token(self) -> bytes:
        return self._json_token(object_value=True)

    def _json_token(self, *, object_value: bool) -> bytes:
        description = "object" if object_value else "string"
        if self.peek_byte() != (ord("{") if object_value else ord('"')):
            raise FinalPublicationValidationError(
                f"encoding input manifest expected a JSON {description}"
            )
        token = bytearray()
        scan = _ManifestTokenScan(object_value)
        while True:
            if len(token) >= _ENCODING_MANIFEST_VALUE_MAX_BYTES:
                raise FinalPublicationValidationError(
                    f"encoding input manifest {description} exceeds its bound"
                )
            if not self._fill():
                raise FinalPublicationValidationError("encoding input manifest ended unexpectedly")
            start = self.position
            end = min(len(self.buffer), start + _ENCODING_MANIFEST_VALUE_MAX_BYTES - len(token))
            end, complete = scan.chunk(self.buffer, start, end)
            token.extend(self.take(end - start))
            if complete:
                return bytes(token)


@dataclass
class _ManifestTokenScan:
    object_value: bool
    depth: int = 0
    in_string: bool = False
    escaped: bool = False

    def chunk(self, buffer: bytes, start: int, end: int) -> tuple[int, bool]:
        # Scan delimiters in C, retaining only lexical state across bounded chunks.
        escaped_position = start if self.escaped else -1
        for match in _ENCODING_MANIFEST_TOKEN_DELIMITERS.finditer(buffer, start, end):
            pos = match.start()
            byte = buffer[pos]
            if self.in_string:
                if pos == escaped_position:
                    continue
                if byte == ord("\\"):
                    escaped_position = pos + 1
                elif byte == ord('"'):
                    self.in_string = False
                    if not self.object_value:
                        return pos + 1, True
            elif self.delimiter_closes_object(byte):
                return pos + 1, True
        self.escaped = escaped_position == end
        return end, False

    def delimiter_closes_object(self, byte: int) -> bool:
        if byte == ord('"'):
            self.in_string = True
        elif byte == ord("{"):
            self.depth += 1
        elif byte == ord("}"):
            self.depth -= 1
            if self.depth < 0:
                raise FinalPublicationValidationError(
                    "encoding input manifest object nesting is invalid"
                )
            return self.depth == 0
        return False


def _canonical_frame_stem_v1(index: int) -> str:
    decimal = str(index)
    return "frame_" + decimal.zfill(max(6, len(decimal)))


def _parse_bounded_canonical_value(raw: bytes, *, description: str) -> object:
    try:
        return strict_canonical_json_loads(
            raw,
            max_bytes=_ENCODING_MANIFEST_VALUE_MAX_BYTES,
        )
    except CanonicalJsonError as error:
        raise FinalPublicationValidationError(
            f"encoding input manifest {description} is malformed"
        ) from error


def _buffered_canonical_array_v1(
    reader: _EncodingManifestReader,
) -> tuple[list[object], int] | None:
    start = reader.position
    if start >= len(reader.buffer) or reader.buffer[start] != ord("["):
        return None
    maximum = min(_ENCODING_MANIFEST_STREAM_BYTES, _ENCODING_MANIFEST_VALUE_MAX_BYTES)
    end = reader.buffer.find(b"]", start + 1, start + maximum)
    if end < 0:
        return None
    # A bracket inside a string or nested array fails strict decoding and falls back.
    raw = reader.buffer[start : end + 1]
    try:
        value = strict_canonical_json_loads(raw, max_bytes=maximum)
    except (CanonicalJsonError, RecursionError):
        return None
    if type(value) is not list:
        return None
    return value, len(raw)


def _buffered_array_items_v1(
    reader: _EncodingManifestReader,
    *,
    object_values: bool,
) -> tuple[list[object], int] | None:
    start = reader.position
    opening = ord("{") if object_values else ord('"')
    if start >= len(reader.buffer) or reader.buffer[start] != opening:
        return None
    maximum = min(_ENCODING_MANIFEST_STREAM_BYTES, _ENCODING_MANIFEST_VALUE_MAX_BYTES)
    stop = start + max(0, maximum - 2)
    closing = reader.buffer.find(b"]", start, stop)
    boundary = reader.buffer.rfind(b"},{" if object_values else b'","', start, stop)
    for end in (closing, boundary + 1 if boundary >= 0 else -1):
        if end <= start:
            continue
        raw = b"[" + reader.buffer[start:end] + b"]"
        try:
            values = strict_canonical_json_loads(raw, max_bytes=maximum)
        except (CanonicalJsonError, RecursionError):
            continue
        if type(values) is list and values:
            return values, end - start
    return None


def _extend_frame_names_v1(
    names: Iterable[object],
    *,
    first_index: int | None,
    count: int,
) -> tuple[int | None, int]:
    for name in names:
        if type(name) is not str:
            raise FinalPublicationValidationError("encoding input frame name is not a string")
        match = _FRAME_STEM.fullmatch(name)
        if match is None:
            raise FinalPublicationValidationError("encoding input frame name is noncanonical")
        index = int(match.group(1))
        if name != _canonical_frame_stem_v1(index):
            raise FinalPublicationValidationError(
                "encoding input frame name padding is noncanonical"
            )
        if first_index is None:
            first_index = index
        elif index != first_index + count:
            raise FinalPublicationValidationError("encoding input frame names are not gap-free")
        count += 1
        if count > _ENCODING_CONTROL_FRAME_CAP:
            raise FinalPublicationValidationError(
                "encoding input frame count exceeds the control cap"
            )
    return first_index, count


def _frame_names_projection_v1(
    first_index: int | None,
    count: int,
    *,
    expected_frame_count: int | None,
) -> tuple[int, int]:
    if count < 1 or first_index is None:
        raise FinalPublicationValidationError("encoding input frame list is empty")
    if first_index + count - 1 > _MAX_SUPPORTED_IMAGE2_INDEX:
        raise FinalPublicationValidationError(
            "encoding input image2 endpoint exceeds the signed-int boundary"
        )
    if expected_frame_count is not None and count != expected_frame_count:
        raise FinalPublicationValidationError(
            "encoding input frame count differs from the settings frame count"
        )
    return first_index, count


_ArrayResult = TypeVar("_ArrayResult")


def _try_buffered_array(
    reader: _EncodingManifestReader,
    buffered: tuple[list[object], int] | None,
    project: Callable[[list[object]], _ArrayResult],
) -> _ArrayResult | None:
    if buffered is None:
        return None
    values, byte_count = buffered
    try:
        result = project(values)
    except (FinalPublicationValidationError, ValueError):
        return None
    reader.take(byte_count)
    return result


def _manifest_array_has_next(reader: _EncodingManifestReader, description: str) -> bool:
    separator = reader.take_byte()
    if separator == ord("]"):
        return False
    if separator != ord(","):
        raise FinalPublicationValidationError(f"encoding input {description} array is malformed")
    return True


def _stream_manifest_array(
    reader: _EncodingManifestReader,
    state: _ArrayResult,
    *,
    object_values: bool,
    element_description: str,
    array_description: str,
    extend: Callable[[Iterable[object], _ArrayResult], _ArrayResult],
    before_item: Callable[[_ArrayResult], None] | None = None,
) -> _ArrayResult:
    reader.expect(b"[")
    if reader.peek_byte() == ord("]"):
        reader.expect(b"]")
        return state
    attempted_loaded = -1
    while True:
        if before_item is not None:
            before_item(state)
        next_state = None
        if reader.loaded != attempted_loaded:
            attempted_loaded = reader.loaded
            buffered = _buffered_array_items_v1(reader, object_values=object_values)
            next_state = _try_buffered_array(reader, buffered, lambda values: extend(values, state))
        if next_state is None:
            token = reader.json_object_token() if object_values else reader.json_string_token()
            value = _parse_bounded_canonical_value(token, description=element_description)
            next_state = extend((value,), state)
        state = next_state
        if not _manifest_array_has_next(reader, array_description):
            return state


def _parse_frame_names_v1(
    reader: _EncodingManifestReader,
    *,
    expected_frame_count: int | None,
) -> tuple[int, int]:
    buffered = _buffered_canonical_array_v1(reader)
    result = _try_buffered_array(
        reader,
        buffered,
        lambda values: _frame_names_projection_v1(
            *_extend_frame_names_v1(values, first_index=None, count=0),
            expected_frame_count=expected_frame_count,
        ),
    )
    if result is not None:
        return result

    def extend(values: Iterable[object], state: tuple[int | None, int]) -> tuple[int | None, int]:
        return _extend_frame_names_v1(values, first_index=state[0], count=state[1])

    initial: tuple[int | None, int] = (None, 0)
    first_index, count = _stream_manifest_array(
        reader,
        initial,
        object_values=False,
        element_description="frame name",
        array_description="frame-name",
        extend=extend,
    )
    return _frame_names_projection_v1(first_index, count, expected_frame_count=expected_frame_count)


def _validate_identity_path_v1(
    relative_path: object,
    *,
    expected_stem: str,
    expected_parent: str | None,
    suffix: str = ".png",
) -> str:
    if type(relative_path) is not str or not relative_path:
        raise FinalPublicationValidationError("encoding input image path is invalid")
    parts = relative_path.split("/")
    if (
        relative_path.startswith("/")
        or len(parts) < 2
        or any(part in {"", ".", ".."} for part in parts)
        or any("\\" in part or "\x00" in part for part in parts)
    ):
        raise FinalPublicationValidationError(
            "encoding input image path is not contained canonical POSIX form"
        )
    if PurePosixPath(relative_path).as_posix() != relative_path:
        raise FinalPublicationValidationError("encoding input image path is noncanonical")
    if parts[-1] != f"{expected_stem}{suffix}":
        raise FinalPublicationValidationError(
            "encoding input image filename differs from its frame name"
        )
    parent = "/".join(parts[:-1])
    if expected_parent is not None and parent != expected_parent:
        raise FinalPublicationValidationError("encoding input image parent changes within an array")
    return parent


def _validate_image_identity_v1(
    value: object,
    *,
    expected_stem: str,
    expected_parent: str | None,
) -> tuple[str, tuple[int, int]]:
    if type(value) is not dict or set(value) != {
        "byte_count",
        "png_header",
        "relative_path",
        "sha256",
    }:
        raise FinalPublicationValidationError("encoding input image identity schema is not closed")
    byte_count = value["byte_count"]
    sha256 = value["sha256"]
    header = value["png_header"]
    if type(byte_count) is not int or not 1 <= byte_count <= 2**64 - 1:
        raise FinalPublicationValidationError("encoding input image byte count is invalid")
    if type(sha256) is not str or _HEX64.fullmatch(sha256) is None:
        raise FinalPublicationValidationError("encoding input image hash is invalid")
    if type(header) is not dict or set(header) != {
        "bit_depth",
        "color_type",
        "height",
        "width",
    }:
        raise FinalPublicationValidationError("encoding input PNG header schema is not closed")
    width = header["width"]
    height = header["height"]
    if (
        type(width) is not int
        or type(height) is not int
        or not 1 <= width <= 2**32 - 1
        or not 1 <= height <= 2**32 - 1
        or type(header["bit_depth"]) is not int
        or header["bit_depth"] != 8
        or type(header["color_type"]) is not int
        or header["color_type"] != 2
    ):
        raise FinalPublicationValidationError("encoding input PNG header is outside RGB8")
    parent = _validate_identity_path_v1(
        value["relative_path"],
        expected_stem=expected_stem,
        expected_parent=expected_parent,
    )
    return parent, (width, height)


def _validate_raw_rgb24_identity_v1(
    value: object,
    *,
    expected_stem: str,
    expected_parent: str | None,
) -> tuple[str, tuple[int, int]]:
    if type(value) is not dict or set(value) != {
        "byte_count",
        "raw_frame",
        "relative_path",
        "sha256",
    }:
        raise FinalPublicationValidationError("raw RGB24 identity schema is not closed")
    header = value["raw_frame"]
    if type(header) is not dict or set(header) != {"width", "height", "pixel_format"}:
        raise FinalPublicationValidationError("raw RGB24 frame schema is not closed")
    width, height = header["width"], header["height"]
    if (
        type(width) is not int
        or not 1 <= width <= 2**32 - 1
        or type(height) is not int
        or not 1 <= height <= 2**32 - 1
        or header["pixel_format"] != "rgb24"
    ):
        raise FinalPublicationValidationError("raw RGB24 frame format is invalid")
    byte_count = value["byte_count"]
    if (
        type(byte_count) is not int
        or not 1 <= byte_count <= 2**64 - 1
        or byte_count != width * height * 3
    ):
        raise FinalPublicationValidationError("raw RGB24 byte count differs from its dimensions")
    if type(value["sha256"]) is not str or _HEX64.fullmatch(value["sha256"]) is None:
        raise FinalPublicationValidationError("raw RGB24 hash is invalid")
    parent = _validate_identity_path_v1(
        value["relative_path"],
        expected_stem=expected_stem,
        expected_parent=expected_parent,
        suffix=".rgb",
    )
    return parent, (width, height)


_IdentityArrayState = tuple[int, str | None, tuple[int, int] | None]


def _parse_identity_array_v1(
    reader: _EncodingManifestReader,
    *,
    first_index: int,
    frame_count: int,
    raw_rgb24: bool = False,
) -> tuple[int, str, tuple[int, int]] | None:
    if reader.peek_byte() == ord("n"):
        reader.expect(b"null")
        return None
    initial: _IdentityArrayState = (0, None, None)

    def extend(values: Iterable[object], state: _IdentityArrayState) -> _IdentityArrayState:
        return _extend_identity_array_v1(
            values,
            first_index=first_index,
            frame_count=frame_count,
            raw_rgb24=raw_rgb24,
            count=state[0],
            parent=state[1],
            first_header=state[2],
        )

    def check_capacity(state: _IdentityArrayState) -> None:
        if state[0] >= frame_count:
            raise FinalPublicationValidationError(
                "encoding input identity array exceeds frame cardinality"
            )

    buffered = _buffered_canonical_array_v1(reader)
    if buffered is not None and type(frame_count) is int and 1 <= len(buffered[0]) <= frame_count:
        projection = _try_buffered_array(
            reader,
            buffered,
            lambda values: _identity_array_projection_v1(
                *extend(values, initial), frame_count=frame_count
            ),
        )
        if projection is not None:
            return projection
    state = _stream_manifest_array(
        reader,
        initial,
        object_values=True,
        element_description="raw RGB24 identity" if raw_rgb24 else "image identity",
        array_description="identity",
        extend=extend,
        before_item=check_capacity,
    )
    return _identity_array_projection_v1(*state, frame_count=frame_count)


def _extend_identity_array_v1(
    values: Iterable[object],
    *,
    first_index: int,
    frame_count: int,
    raw_rgb24: bool,
    count: int,
    parent: str | None,
    first_header: tuple[int, int] | None,
) -> tuple[int, str | None, tuple[int, int] | None]:
    validate_identity = (
        _validate_raw_rgb24_identity_v1 if raw_rgb24 else _validate_image_identity_v1
    )
    for value in values:
        if count >= frame_count:
            raise FinalPublicationValidationError(
                "encoding input identity array exceeds frame cardinality"
            )
        parent, header = validate_identity(
            value,
            expected_stem=_canonical_frame_stem_v1(first_index + count),
            expected_parent=parent,
        )
        if first_header is None:
            first_header = header
        elif header != first_header:
            raise FinalPublicationValidationError(
                "encoding input image shape changes within a sequence"
            )
        count += 1
    return count, parent, first_header


def _identity_array_projection_v1(
    count: int,
    parent: str | None,
    first_header: tuple[int, int] | None,
    *,
    frame_count: int,
) -> tuple[int, str, tuple[int, int]]:
    if count != frame_count:
        raise FinalPublicationValidationError(
            "encoding input identity cardinality differs from frame names"
        )
    assert parent is not None
    assert first_header is not None
    return count, parent, first_header


def _read_input_manifest_header(reader: _EncodingManifestReader) -> tuple[bytes, str]:
    reader.expect(_INPUT_MANIFEST_ALGORITHM_PREFIX)
    algorithm_digit = reader.take(1)
    if algorithm_digit not in {b"2", b"3", b"4"}:
        raise FinalPublicationValidationError("encoding input algorithm version is invalid")
    reader.expect(b'","fingerprint":"')
    fingerprint_raw = reader.take(64)
    reader.expect(b'"')
    try:
        fingerprint = fingerprint_raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise FinalPublicationValidationError("encoding input fingerprint is not ASCII") from error
    if _HEX64.fullmatch(fingerprint) is None:
        raise FinalPublicationValidationError("encoding input fingerprint is invalid")
    reader.self_digest.update(_INPUT_MANIFEST_ALGORITHM_PREFIX + algorithm_digit + b'"')
    reader.include_in_self_digest = True

    return algorithm_digit, fingerprint


def _read_manifest_invocation(reader: _EncodingManifestReader) -> StreamedVrInvocationV1 | None:
    reader.expect(b",")
    next_key = _parse_bounded_canonical_value(
        reader.json_string_token(),
        description="post-frame-names key",
    )
    reader.expect(b":")
    streamed_invocation: StreamedVrInvocationV1 | None = None
    if next_key == "frame_stream_invocation":
        try:
            streamed_invocation = StreamedVrInvocationV1.from_object(
                _parse_bounded_canonical_value(
                    reader.json_object_token(),
                    description="streamed invocation",
                )
            )
        except (TypeError, ValueError) as error:
            raise FinalPublicationValidationError(
                "streamed invocation evidence is invalid"
            ) from error
        reader.expect(b',"frames":')
    elif next_key != "frames":
        raise FinalPublicationValidationError("encoding input post-frame-names key is invalid")
    return streamed_invocation


def _read_manifest_output(
    reader: _EncodingManifestReader,
) -> tuple[bytes, StreamedVrOutputEvidenceV1 | None]:
    reader.expect(b',"schema_version":')
    schema_byte = reader.take(1)
    streamed_output: StreamedVrOutputEvidenceV1 | None = None
    if schema_byte == b"1":
        reader.expect(b"}")
    elif schema_byte in {b"2", b"3"}:
        reader.expect(b',"streamed_output":')
        try:
            streamed_output = StreamedVrOutputEvidenceV1.from_object(
                _parse_bounded_canonical_value(
                    reader.json_object_token(),
                    description="streamed output evidence",
                )
            )
        except (TypeError, ValueError) as error:
            raise FinalPublicationValidationError("streamed output evidence is invalid") from error
        reader.expect(b"}")
    else:
        raise FinalPublicationValidationError("encoding input schema version is invalid")
    return schema_byte, streamed_output


def _check_input_manifest_schema(
    algorithm_digit: bytes,
    schema_byte: bytes,
    streamed_invocation: StreamedVrInvocationV1 | None,
    streamed_output: StreamedVrOutputEvidenceV1 | None,
) -> None:
    if algorithm_digit == b"2" and (
        schema_byte != b"1" or streamed_invocation is not None or streamed_output is not None
    ):
        raise FinalPublicationValidationError("encoding input v2 schema is inconsistent")
    if algorithm_digit == b"3" and (
        schema_byte != b"2" or streamed_invocation is None or streamed_output is None
    ):
        raise FinalPublicationValidationError("encoding input v3 schema is incomplete")
    if algorithm_digit == b"4" and (
        schema_byte != b"3" or streamed_invocation is None or streamed_output is None
    ):
        raise FinalPublicationValidationError("encoding input v4 schema is incomplete")


_ManifestSequence = tuple[int, str, tuple[int, int]]


def _check_manifest_arrays(
    mode: Literal["direct_stereo", "assembled_vr"],
    frames_sequence: _ManifestSequence | None,
    left_sequence: _ManifestSequence | None,
    right_sequence: _ManifestSequence | None,
) -> None:
    if mode == "direct_stereo":
        if frames_sequence is not None or left_sequence is None or right_sequence is None:
            raise FinalPublicationValidationError(
                "direct encoding input manifest has the wrong array shape"
            )
    elif frames_sequence is None or left_sequence is not None or right_sequence is not None:
        raise FinalPublicationValidationError(
            "assembled encoding input manifest has the wrong array shape"
        )


def _check_manifest_invocation(
    algorithm_digit: bytes,
    mode: Literal["direct_stereo", "assembled_vr"],
    frame_count: int,
    first_index: int,
    frames_sequence: _ManifestSequence | None,
    streamed_invocation: StreamedVrInvocationV1,
    streamed_output: StreamedVrOutputEvidenceV1 | None,
) -> None:
    if (
        mode != "assembled_vr"
        or frames_sequence is None
        or streamed_invocation.frame_count != frame_count
        or streamed_invocation.first_index != first_index
        or streamed_invocation.expected_shape != frames_sequence[2]
    ):
        raise FinalPublicationValidationError("streamed invocation differs from its frame sequence")
    assert streamed_output is not None
    expected_sibling = (
        f".depth-surge-final-v4-" f"{streamed_invocation.publication_generation}.tmp.mp4"
    )
    if streamed_output.sibling_relative_path != expected_sibling:
        raise FinalPublicationValidationError(
            "streamed output sibling differs from its publication generation"
        )
    transport = _normalized_image_transport_v1(
        streamed_invocation.resolved_arguments_template, mode
    )
    if algorithm_digit == b"4":
        if (
            transport != "rawvideo"
            or validate_raw_rgb24_input_v1(streamed_invocation.resolved_arguments_template)
            != streamed_invocation.expected_shape
        ):
            raise FinalPublicationValidationError("raw RGB24 invocation shape or transport differs")
    elif transport != "image2pipe":
        raise FinalPublicationValidationError("PNG invocation requires image2pipe transport")


def _parse_encoding_input_manifest_stream_v1(
    handle: BinaryIO,
    *,
    raw_byte_count: int | None,
    expected_frame_count: int | None,
) -> _EncodingInputManifestProjection:
    if raw_byte_count is not None and (type(raw_byte_count) is not int or raw_byte_count < 1):
        raise FinalPublicationValidationError("encoding input raw byte count is invalid")
    if expected_frame_count is not None and (
        type(expected_frame_count) is not int
        or not 1 <= expected_frame_count <= _ENCODING_CONTROL_FRAME_CAP
    ):
        raise FinalPublicationValidationError("encoding input expected frame count is invalid")
    reader = _EncodingManifestReader(handle, raw_byte_count=raw_byte_count)
    algorithm_digit, fingerprint = _read_input_manifest_header(reader)

    reader.expect(b',"frame_names":')
    first_index, frame_count = _parse_frame_names_v1(
        reader,
        expected_frame_count=expected_frame_count,
    )
    streamed_invocation = _read_manifest_invocation(reader)
    frames_sequence = _parse_identity_array_v1(
        reader,
        first_index=first_index,
        frame_count=frame_count,
        raw_rgb24=algorithm_digit == b"4",
    )
    reader.expect(b',"left":')
    left_sequence = _parse_identity_array_v1(
        reader,
        first_index=first_index,
        frame_count=frame_count,
    )
    reader.expect(b',"mode":')
    mode_value = _parse_bounded_canonical_value(
        reader.json_string_token(),
        description="mode",
    )
    if mode_value not in {"direct_stereo", "assembled_vr"}:
        raise FinalPublicationValidationError("encoding input mode is invalid")
    mode = cast(Literal["direct_stereo", "assembled_vr"], mode_value)
    reader.expect(b',"right":')
    right_sequence = _parse_identity_array_v1(
        reader,
        first_index=first_index,
        frame_count=frame_count,
    )
    schema_byte, streamed_output = _read_manifest_output(reader)
    _check_input_manifest_schema(algorithm_digit, schema_byte, streamed_invocation, streamed_output)
    _check_manifest_arrays(mode, frames_sequence, left_sequence, right_sequence)
    if streamed_invocation is not None:
        _check_manifest_invocation(
            algorithm_digit,
            mode,
            frame_count,
            first_index,
            frames_sequence,
            streamed_invocation,
            streamed_output,
        )
    if raw_byte_count is not None and reader.consumed != raw_byte_count:
        raise FinalPublicationValidationError(
            "encoding input raw byte count includes trailing bytes"
        )
    if raw_byte_count is None and reader.peek_byte() is not None:
        raise FinalPublicationValidationError("encoding input manifest has trailing bytes")
    if reader.self_digest.hexdigest() != fingerprint:
        raise FinalPublicationValidationError("encoding input self-fingerprint differs")
    return _EncodingInputManifestProjection(
        fingerprint=fingerprint,
        raw_sha256=reader.raw_digest.hexdigest(),
        raw_byte_count=reader.consumed,
        mode=mode,
        frame_count=frame_count,
        first_index=first_index,
        left_parent=(PurePosixPath(left_sequence[1]) if left_sequence is not None else None),
        right_parent=(PurePosixPath(right_sequence[1]) if right_sequence is not None else None),
        frames_parent=(PurePosixPath(frames_sequence[1]) if frames_sequence is not None else None),
        streamed_invocation=streamed_invocation,
        streamed_output=streamed_output,
        raw_pixel_format="rgb24" if algorithm_digit == b"4" else None,
    )


def _stream_input_manifest_projection_handle_v1(  # noqa: C901
    handle: BinaryIO,
    *,
    raw_byte_count: int | None = None,
    logical_byte_count: int | None = None,
    expected_frame_count: int | None = None,
) -> _EncodingInputManifestProjection:
    if logical_byte_count is not None and raw_byte_count is None:
        raise FinalPublicationValidationError(
            "encoding input logical validation requires a raw byte count"
        )
    if logical_byte_count is not None and (
        type(logical_byte_count) is not int
        or raw_byte_count is None
        or logical_byte_count < raw_byte_count
    ):
        raise FinalPublicationValidationError("encoding input logical byte count is invalid")
    handle.seek(0)
    projection = _parse_encoding_input_manifest_stream_v1(
        handle,
        raw_byte_count=raw_byte_count,
        expected_frame_count=expected_frame_count,
    )
    if logical_byte_count is not None:
        tail = logical_byte_count - projection.raw_byte_count
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
    elif raw_byte_count is not None and handle.read(1):
        raise FinalPublicationValidationError("encoding input file exceeds its raw byte count")
    return projection


def _stream_input_manifest_projection_v1(
    job_root: Path,
    path: Path,
    *,
    raw_byte_count: int | None = None,
    logical_byte_count: int | None = None,
    expected_frame_count: int | None = None,
) -> _EncodingInputManifestProjection:
    """Validate one manifest through an identity-bound no-follow root handle."""

    candidate = Path(path)
    try:
        with AcquiredRoot(job_root) as acquired:
            root_identity = acquired.persistent_identity
            parent_identity = acquired.directory_identity(candidate.parent)
            with acquired.open_regular(candidate) as handle:
                frozen_identity = file_identity_from_open_handle(handle)
                projection = _stream_input_manifest_projection_handle_v1(
                    handle,
                    raw_byte_count=raw_byte_count,
                    logical_byte_count=logical_byte_count,
                    expected_frame_count=expected_frame_count,
                )
                if (
                    file_identity_from_open_handle(handle) != frozen_identity
                    or acquired.persistent_regular_identity(candidate) != frozen_identity
                ):
                    raise FinalPublicationValidationError(
                        "encoding input manifest name changed during read"
                    )
            if (
                acquired.directory_identity(candidate.parent) != parent_identity
                or acquired.persistent_identity != root_identity
            ):
                raise FinalPublicationValidationError(
                    "encoding input manifest namespace changed during read"
                )
            return projection
    except (NoFollowPathError, FileNotFoundError) as error:
        raise FinalPublicationValidationError(
            "encoding input manifest failed no-follow acquisition"
        ) from error


def _stream_input_manifest_identity(
    job_root: Path,
    path: Path,
    *,
    raw_byte_count: int | None = None,
    logical_byte_count: int | None = None,
    expected_frame_count: int | None = None,
) -> tuple[str, str, int]:
    projection = _stream_input_manifest_projection_v1(
        job_root,
        path,
        raw_byte_count=raw_byte_count,
        logical_byte_count=logical_byte_count,
        expected_frame_count=expected_frame_count,
    )
    return projection.fingerprint, projection.raw_sha256, projection.raw_byte_count


def _settings_frame_count_v1(settings: ProcessingSettingsArtifactV5) -> int:
    frame_count = settings.selected_frame_count
    if type(frame_count) is not int or not 1 <= frame_count <= _ENCODING_CONTROL_FRAME_CAP:
        raise FinalPublicationValidationError("settings frame count is invalid")
    return frame_count


def _load_settings_frame_count_v1(job_root: Path) -> int:
    try:
        settings = ProcessingSettingsArtifactV5.parse(
            _read_bounded_regular_file(
                Path(job_root) / "processing-settings-v5.json",
                max_bytes=SETTINGS_ARTIFACT_MAX_RAW_BYTES,
            )
        )
    except (OSError, SettingsArtifactSchemaError) as error:
        raise FinalPublicationValidationError(
            "current settings cannot bind the encoding input manifest"
        ) from error
    return _settings_frame_count_v1(settings)


def _regular_nonlink(path: Path) -> bool:
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISREG(metadata.st_mode) and not path.is_symlink()


def _atomic_replace(
    path: Path,
    raw: bytes,
    *,
    expected_target_identity: PersistedFileIdentityV1 | None,
) -> None:
    if capture_optional_publication_target_identity(path) != expected_target_identity:
        raise OwnedTemporaryIdentityError(f"publication target identity changed: {path}")
    temporary = path.with_name(f".{path.name}.publication-v4.tmp")
    owned = OwnedSiblingTemporary.create(path, temporary=temporary)
    try:
        if owned.target_identity != expected_target_identity:
            raise OwnedTemporaryIdentityError(f"publication target identity changed: {path}")
        if owned.handle.write(raw) != len(raw):
            raise OSError("short final publication manifest write")
        owned.sync()
        owned.assert_current_identity()
        if _read_bounded_regular_file(temporary, max_bytes=max(len(raw), 1)) != raw:
            raise FinalPublicationValidationError("manifest temporary failed reopen validation")
        owned.replace()
        if _read_bounded_regular_file(path, max_bytes=max(len(raw), 1)) != raw:
            raise FinalPublicationValidationError("published manifest failed reopen validation")
    finally:
        owned.cleanup()


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
    expected_mode: Literal["direct_stereo", "assembled_vr"] | None = None,
) -> Literal["direct_stereo", "assembled_vr"]:
    if not arguments or arguments[0] != "ffmpeg":
        raise FinalPublicationValidationError("normalized argv must start with ffmpeg")
    output_token = "@output:path64=" + _path64(relative_path)
    image_views: list[str] = []
    input_kinds: list[str] = []
    audio_count = 0
    forbidden_path_options = {
        "-attach",
        "-dump_attachment",
        "-filter_complex_script",
        "-filter_script",
        "-passlogfile",
        "-vstats_file",
    }
    for index, argument in enumerate(arguments):
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
            if (
                match is None
                or match.group(1) != input_stage_fingerprint
                or index == 0
                or arguments[index - 1] != "-i"
            ):
                raise FinalPublicationValidationError("normalized image token is invalid")
            if match.group(2) in image_views:
                raise FinalPublicationValidationError("normalized image view is duplicated")
            image_views.append(match.group(2))
            input_kinds.append(f"image:{match.group(2)}")
        elif argument.startswith("@audio:"):
            if (
                _AUDIO_TOKEN.fullmatch(argument) is None
                or index == 0
                or arguments[index - 1] != "-i"
            ):
                raise FinalPublicationValidationError("normalized audio token is invalid")
            audio_count += 1
            input_kinds.append("audio")
        elif argument.startswith("@output:"):
            match = _OUTPUT_TOKEN.fullmatch(argument)
            if (
                match is None
                or argument != output_token
                or _decode_path64(match.group(1)) != relative_path
                or index != len(arguments) - 1
            ):
                raise FinalPublicationValidationError("normalized output token is invalid")
        elif argument.startswith("@"):
            raise FinalPublicationValidationError("normalized argv token is unrecognized")
        elif argument == "-i":
            if index + 1 >= len(arguments) or not arguments[index + 1].startswith(
                ("@image-sequence:", "@audio:")
            ):
                raise FinalPublicationValidationError(
                    "normalized input option is not followed by an identity token"
                )
        elif argument in forbidden_path_options:
            raise FinalPublicationValidationError(
                "normalized argv contains a forbidden path-bearing option"
            )
        elif (
            argument.startswith(("/", "\\", "file:", "http:", "https:"))
            or re.match(r"^[A-Za-z]:[\\/]", argument, flags=re.ASCII) is not None
            or "\\" in argument
            or argument.casefold().endswith(
                (".png", ".mp4", ".mov", ".mkv", ".flac", ".wav", ".json")
            )
        ):
            raise FinalPublicationValidationError("normalized argv contains an unnormalized path")
    if arguments[-1] != output_token:
        raise FinalPublicationValidationError(
            "normalized argv requires its output token as the final element"
        )
    if audio_count > 1:
        raise FinalPublicationValidationError("normalized audio input is duplicated")
    if image_views == ["frames"]:
        inferred_mode: Literal["direct_stereo", "assembled_vr"] = "assembled_vr"
        expected_inputs = ["image:frames"] + (["audio"] if audio_count else [])
    elif image_views == ["left", "right"]:
        inferred_mode = "direct_stereo"
        expected_inputs = ["image:left", "image:right"] + (["audio"] if audio_count else [])
    else:
        raise FinalPublicationValidationError(
            "normalized argv image view set is incomplete or out of order"
        )
    if input_kinds != expected_inputs:
        raise FinalPublicationValidationError(
            "normalized argv input indexes do not match the encoding mode"
        )
    if expected_mode is not None and inferred_mode != expected_mode:
        raise FinalPublicationValidationError(
            "normalized argv mode differs from the encoding input manifest"
        )
    _normalized_image_transport_v1(arguments, inferred_mode)
    expected_audio_index = 1 if inferred_mode == "assembled_vr" else 2
    audio_references = [
        argument
        for argument in arguments
        if re.fullmatch(r"[0-9]+:a:0\??", argument, flags=re.ASCII) is not None
    ]
    if audio_count:
        if audio_references != [f"{expected_audio_index}:a:0"]:
            raise FinalPublicationValidationError(
                "normalized argv audio map uses the wrong input index"
            )
    elif audio_references:
        raise FinalPublicationValidationError(
            "normalized argv references audio without an audio identity"
        )
    return inferred_mode


def _normalized_option_values_v1(
    arguments: tuple[str, ...],
    option: str,
) -> tuple[str, ...]:
    values: list[str] = []
    for index, argument in enumerate(arguments):
        if argument != option:
            continue
        if index + 1 >= len(arguments) - 1:
            raise FinalPublicationValidationError(f"normalized argv option {option} has no value")
        values.append(arguments[index + 1])
    return tuple(values)


def _validate_image2pipe_transport(
    arguments: tuple[str, ...], mode: str, formats: tuple[str, ...], input_codecs: tuple[str, ...]
) -> None:
    if mode != "assembled_vr":
        raise FinalPublicationValidationError(
            "normalized image2pipe transport requires assembled VR"
        )
    if formats != ("image2pipe",):
        raise FinalPublicationValidationError(
            "normalized image2pipe input declaration is incomplete"
        )
    if input_codecs != ("png",):
        raise FinalPublicationValidationError(
            "normalized image2pipe PNG codec declaration is incomplete"
        )
    image_token_index = next(
        (index for index, value in enumerate(arguments) if value.startswith("@image-sequence:")),
        -1,
    )
    if (
        image_token_index < 0
        or arguments.index("-f") >= image_token_index
        or arguments.index("-vcodec") >= image_token_index
    ):
        raise FinalPublicationValidationError(
            "normalized image2pipe options do not bind the image input"
        )


def _normalized_image_transport_v1(
    arguments: tuple[str, ...],
    mode: Literal["direct_stereo", "assembled_vr"],
) -> Literal["image2", "image2pipe", "rawvideo"]:
    formats = _normalized_option_values_v1(arguments, "-f")
    input_codecs = _normalized_option_values_v1(arguments, "-vcodec")
    raw_options = _normalized_option_values_v1(arguments, "-pixel_format") + (
        _normalized_option_values_v1(arguments, "-video_size")
    )
    if formats == ("rawvideo",):
        if mode != "assembled_vr":
            raise FinalPublicationValidationError("raw RGB24 transport requires assembled VR")
        try:
            validate_raw_rgb24_input_v1(arguments)
        except ValueError as error:
            raise FinalPublicationValidationError(
                "raw RGB24 input declaration is invalid"
            ) from error
        return "rawvideo"
    if raw_options:
        raise FinalPublicationValidationError("raw input options cannot bind a PNG transport")
    if not formats and not input_codecs:
        return "image2"
    _validate_image2pipe_transport(arguments, mode, formats, input_codecs)
    return "image2pipe"


def _normalized_input_rate(
    arguments: tuple[str, ...],
    input_projection: _EncodingInputManifestProjection,
    width: int,
    height: int,
) -> Fraction:
    transport = _normalized_image_transport_v1(arguments, input_projection.mode)
    if (transport == "rawvideo") != (input_projection.raw_pixel_format == "rgb24"):
        raise FinalPublicationValidationError("encoding transport differs from raw frame evidence")
    if transport == "rawvideo" and validate_raw_rgb24_input_v1(arguments) != (width, height):
        raise FinalPublicationValidationError("raw RGB24 dimensions differ from durable settings")
    start_values = _normalized_option_values_v1(arguments, "-start_number")
    expected_start_count = (
        0 if transport != "image2" else (1 if input_projection.mode == "assembled_vr" else 2)
    )
    if start_values != (str(input_projection.first_index),) * expected_start_count:
        raise FinalPublicationValidationError(
            "normalized argv start number differs from the input manifest"
        )
    rate_values = _normalized_option_values_v1(arguments, "-framerate")
    expected_rate_count = 1 if transport != "image2" else expected_start_count
    if len(rate_values) != expected_rate_count or len(set(rate_values)) != 1:
        raise FinalPublicationValidationError(
            "normalized argv frame rate is incomplete or inconsistent"
        )
    try:
        rate = Fraction(rate_values[0])
    except (ValueError, ZeroDivisionError) as error:
        raise FinalPublicationValidationError("normalized argv frame rate is invalid") from error
    if rate <= 0:
        raise FinalPublicationValidationError("normalized argv frame rate must be positive")
    return rate


def _validate_timeline_filter(
    timeline: EncodingTimeline,
    input_projection: _EncodingInputManifestProjection,
    filters: tuple[str, ...],
    converted: bool,
    bounded_filter: bool,
) -> None:
    expected_filter, _ = timeline.ffmpeg_video_options(
        input_projection.frame_count, drain_audio=bounded_filter
    )
    if input_projection.mode == "assembled_vr":
        valid_filter = filters == (expected_filter,)
    else:
        valid_filter = (
            len(filters) == 1
            and filters[0].endswith(f";[stacked]{expected_filter}[vr]")
            and filters[0].count("fps=") == int(converted)
        )
    if not valid_filter:
        raise FinalPublicationValidationError("normalized frame-rate filter is inconsistent")


def _normalized_output_timing(
    values: dict,
    arguments: tuple[str, ...],
    input_projection: _EncodingInputManifestProjection,
    rate: Fraction,
) -> tuple[Fraction, int, bool]:
    filters = _normalized_option_values_v1(
        arguments, "-vf" if input_projection.mode == "assembled_vr" else "-filter_complex"
    )
    converted = any("fps=" in graph for graph in filters)
    frame_values = _normalized_option_values_v1(arguments, "-frames:v")
    bounded_filter = not frame_values
    audio_expected = any(_AUDIO_TOKEN.fullmatch(item) is not None for item in arguments)
    output_count = input_projection.frame_count
    if converted or bounded_filter or "frame_rate" in values["video_properties"]:
        # Old, already published jobs lacking an exact source clock keep their frozen argv.
        timeline = EncodingTimeline.from_artifact(values)
        if rate != timeline.source or converted != (timeline.source != timeline.output):
            raise FinalPublicationValidationError(
                "normalized timeline differs from durable settings"
            )
        if bounded_filter and not audio_expected:
            raise FinalPublicationValidationError("normalized argv has no video frame limit")
        if converted or bounded_filter:
            _validate_timeline_filter(
                timeline, input_projection, filters, converted, bounded_filter
            )
        output_count = timeline.output_frames(input_projection.frame_count)
        rate = timeline.output
    if not bounded_filter and frame_values != (str(output_count),):
        raise FinalPublicationValidationError(
            "normalized argv frame limit differs from the input manifest"
        )

    return rate, output_count, audio_expected


def _normalized_audio_packet_bound(arguments: tuple[str, ...], audio_expected: bool) -> int:
    audio_packet_upper_bound = 0
    audio_filters = _normalized_option_values_v1(arguments, "-filter:a")
    if audio_expected:
        if len(audio_filters) != 1:
            raise FinalPublicationValidationError("normalized argv audio filter is missing")
        match = re.fullmatch(
            r"atrim=end_sample=([1-9][0-9]*),asetpts=PTS-STARTPTS",
            audio_filters[0],
            flags=re.ASCII,
        )
        if match is None:
            raise FinalPublicationValidationError("normalized argv audio sample bound is invalid")
        audio_end_sample = int(match.group(1))
        if audio_end_sample > 2**64 - 1:
            raise FinalPublicationValidationError("normalized argv audio sample bound exceeds u64")
        audio_packet_upper_bound = (audio_end_sample + 1023) // 1024 + 2
    elif audio_filters:
        raise FinalPublicationValidationError(
            "normalized argv has an audio filter without audio input"
        )
    return audio_packet_upper_bound


def _final_container_expectation_from_intent_v1(
    *,
    settings: ProcessingSettingsArtifactV5,
    input_projection: _EncodingInputManifestProjection,
    arguments: tuple[str, ...],
) -> FinalContainerExpectation:
    values = settings.to_object()
    settings_value = values["processing_settings"]
    width = settings_value["vr_output_width"]
    height = settings_value["vr_output_height"]
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise FinalPublicationValidationError("durable settings lack final container dimensions")

    rate = _normalized_input_rate(arguments, input_projection, width, height)
    rate, output_count, audio_expected = _normalized_output_timing(
        values, arguments, input_projection, rate
    )
    encoder_values = _normalized_option_values_v1(arguments, "-c:v")
    if len(encoder_values) != 1 or encoder_values[0] not in {
        "libx264",
        "libx265",
        "hevc_nvenc",
    }:
        raise FinalPublicationValidationError("normalized argv final encoder is invalid")
    audio_packet_upper_bound = _normalized_audio_packet_bound(arguments, audio_expected)
    try:
        return FinalContainerExpectation(
            width=width,
            height=height,
            fps=rate_text(rate),
            frame_count=output_count,
            resolved_encoder=cast(Literal["libx264", "libx265", "hevc_nvenc"], encoder_values[0]),
            audio_expected=audio_expected,
            audio_packet_upper_bound=audio_packet_upper_bound,
        )
    except (TypeError, ValueError) as error:
        raise FinalPublicationValidationError(
            "final container expectation cannot be reconstructed"
        ) from error


@dataclass(frozen=True)
class FinalVideoManifestV4:
    relative_path: str
    sha256: str
    byte_count: int
    input_stage_fingerprint: str
    resolved_encoding_arguments: tuple[str, ...]
    encoding_settings_fingerprint: str
    fingerprint: str
    encoded_output: StreamedVrOutputEvidenceV1 | None = None

    def _base_object(self) -> dict:
        return {
            "schema_version": 2 if self.encoded_output is not None else 1,
            "algorithm_version": "final-video-publication-v4",
            **({"encoded_output": self.encoded_output.to_object()} if self.encoded_output else {}),
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
    encoded_output: StreamedVrOutputEvidenceV1 | None = None,
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
    if encoded_output is not None and (encoded_output.raw_sha256, encoded_output.byte_count) != (
        video_sha256,
        video_byte_count,
    ):
        raise FinalPublicationValidationError("encoded output differs from final manifest")
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
        encoded_output=encoded_output,
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
    if type(value) is not dict:
        raise FinalPublicationValidationError("final manifest must be an object")
    if value.get("schema_version") == 2:
        expected.add("encoded_output")
    if set(value) != expected:
        raise FinalPublicationValidationError("final manifest schema is not closed")
    assert isinstance(value, dict)
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] not in (1, 2)
        or value["algorithm_version"] != "final-video-publication-v4"
        or type(value["resolved_encoding_arguments"]) is not list
    ):
        raise FinalPublicationValidationError("final manifest version is invalid")
    try:
        output = (
            StreamedVrOutputEvidenceV1.from_object(value["encoded_output"])
            if value["schema_version"] == 2
            else None
        )
    except (TypeError, ValueError) as error:
        raise FinalPublicationValidationError("invalid file-backed output evidence") from error
    manifest = build_final_video_manifest_v4(
        relative_path=value["relative_path"],
        video_sha256=value["sha256"],
        video_byte_count=value["byte_count"],
        input_stage_fingerprint=value["input_stage_fingerprint"],
        resolved_encoding_arguments=tuple(value["resolved_encoding_arguments"]),
        encoded_output=output,
    )
    if (
        manifest.encoding_settings_fingerprint != value["encoding_settings_fingerprint"]
        or manifest.fingerprint != value["fingerprint"]
    ):
        raise FinalPublicationValidationError("final manifest fingerprints differ")
    return manifest


def _parse_input_manifest_fingerprint(
    raw: bytes,
    *,
    expected_frame_count: int | None = None,
) -> str:
    return _parse_input_manifest_projection_v1(
        raw,
        expected_frame_count=expected_frame_count,
    ).fingerprint


def _parse_input_manifest_projection_v1(
    raw: bytes,
    *,
    expected_frame_count: int | None = None,
) -> _EncodingInputManifestProjection:
    if not isinstance(raw, bytes):
        raise TypeError("encoding input manifest must be bytes")
    return _parse_encoding_input_manifest_stream_v1(
        io.BytesIO(raw),
        raw_byte_count=len(raw),
        expected_frame_count=expected_frame_count,
    )


@dataclass(frozen=True)
class EncodingInputPublicationSource:
    """Bounded staged-manifest evidence used by the production publication path."""

    path: Path
    fingerprint: str
    raw_sha256: str
    raw_byte_count: int
    logical_byte_count: int | None = None
    job_root: Path | None = None
    job_root_identity: DirectoryIdentityV1 | None = None
    source_parent_identity: DirectoryIdentityV1 | None = None
    source_identity: PersistedFileIdentityV1 | None = None

    @classmethod
    def from_authenticated_handle(
        cls,
        *,
        path: Path,
        handle: BinaryIO,
        job_root: Path,
        job_root_identity: DirectoryIdentityV1,
        source_parent_identity: DirectoryIdentityV1,
        source_identity: PersistedFileIdentityV1,
        raw_byte_count: int,
        logical_byte_count: int | None,
        expected_frame_count: int | None = None,
    ) -> "EncodingInputPublicationSource":
        projection = _stream_input_manifest_projection_handle_v1(
            handle,
            raw_byte_count=raw_byte_count if logical_byte_count is not None else None,
            logical_byte_count=logical_byte_count,
            expected_frame_count=expected_frame_count,
        )
        return cls(
            path=Path(path),
            fingerprint=projection.fingerprint,
            raw_sha256=projection.raw_sha256,
            raw_byte_count=projection.raw_byte_count,
            logical_byte_count=logical_byte_count,
            job_root=Path(job_root),
            job_root_identity=job_root_identity,
            source_parent_identity=source_parent_identity,
            source_identity=source_identity,
        )

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        raw_byte_count: int | None = None,
        logical_byte_count: int | None = None,
        expected_frame_count: int | None = None,
    ) -> "EncodingInputPublicationSource":
        candidate = Path(path)
        fingerprint, raw_sha256, measured = _stream_input_manifest_identity(
            candidate.parent,
            candidate,
            raw_byte_count=raw_byte_count,
            logical_byte_count=logical_byte_count,
            expected_frame_count=expected_frame_count,
        )
        return cls(
            path=candidate,
            fingerprint=fingerprint,
            raw_sha256=raw_sha256,
            raw_byte_count=measured,
            logical_byte_count=logical_byte_count,
        )

    def projection(
        self,
        *,
        expected_frame_count: int | None = None,
    ) -> _EncodingInputManifestProjection:
        path = Path(self.path)
        authority = (
            self.job_root,
            self.job_root_identity,
            self.source_parent_identity,
            self.source_identity,
        )
        if any(value is not None for value in authority):
            if any(value is None for value in authority):
                raise FinalPublicationValidationError(
                    "staged encoding input manifest authority is incomplete"
                )
            projection = self._authenticated_projection(path, expected_frame_count)
        else:
            if not _regular_nonlink(path):
                raise FinalPublicationValidationError(
                    "staged encoding input manifest is not an ordinary file"
                )
            if (
                self.logical_byte_count is not None
                and path.stat().st_size != self.logical_byte_count
            ):
                raise FinalPublicationValidationError(
                    "staged encoding input manifest logical length differs"
                )
            projection = _stream_input_manifest_projection_v1(
                path.parent,
                path,
                raw_byte_count=(
                    self.raw_byte_count if self.logical_byte_count is not None else None
                ),
                logical_byte_count=self.logical_byte_count,
                expected_frame_count=expected_frame_count,
            )
        if (
            projection.fingerprint != self.fingerprint
            or projection.raw_sha256 != self.raw_sha256
            or projection.raw_byte_count != self.raw_byte_count
        ):
            raise FinalPublicationValidationError("staged encoding input manifest evidence differs")
        return projection

    def _authenticated_projection(
        self, path: Path, expected_frame_count: int | None
    ) -> _EncodingInputManifestProjection:
        assert self.job_root is not None
        assert self.job_root_identity is not None
        assert self.source_parent_identity is not None
        assert self.source_identity is not None
        try:
            with AcquiredRoot(self.job_root) as acquired:
                if acquired.persistent_identity != self.job_root_identity:
                    raise FinalPublicationValidationError(
                        "staged encoding input job root identity differs"
                    )
                if acquired.directory_identity(path.parent) != self.source_parent_identity:
                    raise FinalPublicationValidationError(
                        "staged encoding input parent identity differs"
                    )
                with acquired.open_regular(path) as handle:
                    if (
                        file_identity_from_open_handle(handle) != self.source_identity
                        or acquired.persistent_regular_identity(path) != self.source_identity
                    ):
                        raise FinalPublicationValidationError(
                            "staged encoding input file identity differs"
                        )
                    if (
                        self.logical_byte_count is not None
                        and os.fstat(handle.fileno()).st_size != self.logical_byte_count
                    ):
                        raise FinalPublicationValidationError(
                            "staged encoding input manifest logical length differs"
                        )
                    projection = _stream_input_manifest_projection_handle_v1(
                        handle,
                        raw_byte_count=(
                            self.raw_byte_count if self.logical_byte_count is not None else None
                        ),
                        logical_byte_count=self.logical_byte_count,
                        expected_frame_count=expected_frame_count,
                    )
                    if (
                        file_identity_from_open_handle(handle) != self.source_identity
                        or acquired.persistent_regular_identity(path) != self.source_identity
                    ):
                        raise FinalPublicationValidationError(
                            "staged encoding input identity changed during read"
                        )
                if (
                    acquired.directory_identity(path.parent) != self.source_parent_identity
                    or acquired.persistent_identity != self.job_root_identity
                ):
                    raise FinalPublicationValidationError(
                        "staged encoding input namespace changed during read"
                    )
        except (NoFollowPathError, FileNotFoundError) as error:
            raise FinalPublicationValidationError(
                "staged encoding input manifest failed no-follow acquisition"
            ) from error
        return projection

    def validated(
        self,
        *,
        expected_frame_count: int | None = None,
    ) -> "EncodingInputPublicationSource":
        self.projection(expected_frame_count=expected_frame_count)
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
        container_validator: Callable[[Path], bool] | None,
        expected_sibling_identity: PersistedFileIdentityV1 | None = None,
        expected_sibling_sha256: str | None = None,
        expected_sibling_byte_count: int | None = None,
        expected_existing_video_identity: PersistedFileIdentityV1 | None = None,
        reservation: (
            PreparedFinalEncodingReservationV1 | FinalEncodingReservationCapabilityV1 | None
        ) = None,
        retire_reservation: bool = False,
        event: Callable[[str], None] | None = None,
    ) -> None:
        self.job_root = Path(job_root)
        settings_value = settings.to_object()
        metadata = settings_value["metadata"]
        self._producer_marker_valid = bool(
            metadata["final_media_producer_contract_version"] == 4
            and metadata["final_media_publication_generation"] is not None
        )
        self._expected_relative_path = settings_value["output_info"]["expected_final_relative_path"]
        self._expected_frame_count = _settings_frame_count_v1(settings)
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
        self.expected_sibling_identity = expected_sibling_identity
        if (expected_sibling_sha256 is None) != (expected_sibling_byte_count is None):
            raise ValueError("validated sibling digest/count evidence is incomplete")
        if expected_sibling_sha256 is not None and expected_sibling_identity is None:
            raise ValueError("validated sibling bytes lack frozen identity")
        if container_validator is None and expected_sibling_sha256 is None:
            raise ValueError("publication lacks container validation evidence")
        self.expected_sibling_sha256 = expected_sibling_sha256
        self.expected_sibling_byte_count = expected_sibling_byte_count
        self.expected_existing_video_identity = expected_existing_video_identity
        self.reservation = reservation
        self.retire_reservation = retire_reservation
        self.event = event
        if reservation is not None:
            if expected_existing_video_identity is not None:
                raise ValueError("an active reservation supplies its own final target identity")
            if retire_reservation:
                raise ValueError("final reservation retirement requires durable terminal settings")
            if reservation.job_root != self.job_root.resolve():
                raise ValueError("final reservation belongs to a different job root")
            if reservation.sibling_path.name != self.sibling_relative_path:
                raise ValueError("runtime sibling differs from final reservation")

    def _materialize_reservation(self) -> PreparedFinalEncodingReservationV1:
        reservation = self.reservation
        if isinstance(reservation, FinalEncodingReservationCapabilityV1):
            return reservation.materialize_for_publication()
        if reservation is None:
            raise ValueError("final publication has no reservation authority")
        return reservation

    @staticmethod
    def path64(relative_path: str) -> str:
        return _path64(relative_path)

    def _emit(self, value: str) -> None:
        if self.event is not None:
            self.event(value)

    def publish(self, *, crash_at: str | None = None) -> FinalVideoManifestV4:  # noqa: C901
        if not self._producer_marker_valid:
            raise FinalPublicationValidationError("publication-v4 producer marker is absent")
        relative_path = self._expected_relative_path
        expected_frame_count = self._expected_frame_count
        try:
            validate_portable_final_component_v1(relative_path)
        except SettingsArtifactSchemaError as error:
            raise FinalPublicationValidationError("final target is not portable") from error
        sibling = self.job_root / self.sibling_relative_path
        final_path = self.job_root / relative_path
        input_target = self.job_root / ENCODING_INPUT_MANIFEST_RELATIVE_PATH
        final_manifest_target = self.job_root / FINAL_VIDEO_MANIFEST_RELATIVE_PATH
        standalone_final_target_identity = None
        standalone_input_target_identity = None
        standalone_manifest_target_identity = None
        if self.reservation is None:
            standalone_final_target_identity = self.expected_existing_video_identity
            if (
                capture_optional_publication_target_identity(final_path)
                != standalone_final_target_identity
            ):
                raise FinalPublicationValidationError(
                    "final video differs from the authorized predecessor; preserving the target"
                )
            standalone_input_target_identity = capture_optional_publication_target_identity(
                input_target
            )
            standalone_manifest_target_identity = capture_optional_publication_target_identity(
                final_manifest_target
            )
        if self.encoding_input_source is not None:
            source = self.encoding_input_source
            input_projection = source.projection(
                expected_frame_count=expected_frame_count,
            )
            manifest_fingerprint = input_projection.fingerprint
        else:
            assert self.encoding_input_manifest_raw is not None
            input_projection = _parse_input_manifest_projection_v1(
                self.encoding_input_manifest_raw,
                expected_frame_count=expected_frame_count,
            )
            manifest_fingerprint = input_projection.fingerprint
        if manifest_fingerprint != self.input_stage_fingerprint:
            raise FinalPublicationValidationError("input manifest fingerprint differs")
        _validate_normalized_argv(
            self.resolved_encoding_arguments,
            input_stage_fingerprint=self.input_stage_fingerprint,
            relative_path=relative_path,
            expected_mode=input_projection.mode,
        )
        streamed_invocation = input_projection.streamed_invocation
        streamed_output = input_projection.streamed_output
        if streamed_invocation is not None:
            assert streamed_output is not None
            if (
                streamed_invocation.materialized_arguments(manifest_fingerprint)
                != self.resolved_encoding_arguments
                or streamed_output.sibling_relative_path != self.sibling_relative_path
                or self.reservation is None
            ):
                raise FinalPublicationValidationError(
                    "streamed publication differs from its launch evidence"
                )
        reservation_phase = "standalone"
        reserved_target_identity = None
        reserved_target_parent_identity = None
        if self.reservation is not None:
            reservation = self._materialize_reservation()
            recovery = reservation.observe()
            reservation_phase = recovery.phase
            if (
                streamed_output is not None
                and streamed_output.reservation_index_fingerprint != reservation.index.fingerprint
            ):
                raise FinalPublicationValidationError(
                    "streamed output differs from the final reservation"
                )
            if reservation_phase == "ready_pre_video":
                reservation.validate_reserved_target()
            else:
                raise FinalPublicationValidationError(
                    f"reservation cannot accept an encoded sibling in phase {reservation_phase}"
                )
            reserved_target_identity = reservation.target_descriptor.reserved_target_identity
            reserved_target_parent_identity = reservation.target_descriptor.target_parent_identity
            del reservation
        try:
            with AcquiredRoot(self.job_root) as acquired:
                path_identity = acquired.persistent_regular_identity(sibling)
                with acquired.open_regular_locked(sibling) as frozen_handle:
                    frozen_sibling_identity = file_identity_from_open_handle(frozen_handle)
                    if frozen_sibling_identity.link_count != 1:
                        raise FinalPublicationValidationError("sibling video is not single-link")
                    if path_identity != frozen_sibling_identity:
                        raise FinalPublicationValidationError(
                            "sibling video identity changed before acquisition"
                        )
                    if (
                        self.expected_sibling_identity is not None
                        and frozen_sibling_identity != self.expected_sibling_identity
                    ):
                        raise FinalPublicationValidationError(
                            "sibling video identity changed since container validation"
                        )
                    before = os.fstat(frozen_handle.fileno())
                    if self.container_validator is not None:
                        try:
                            container_valid = _validate_open_container_v1(
                                sibling, frozen_handle, self.container_validator
                            )
                        except InterruptedError:
                            raise
                        except OSError as error:
                            raise FinalPublicationValidationError(
                                "sibling video container validation failed"
                            ) from error
                        if not container_valid:
                            raise FinalPublicationValidationError(
                                "sibling video failed container validation"
                            )
                    video_sha256, video_byte_count = _hash_open_file(frozen_handle)
                    if streamed_output is not None and (
                        frozen_sibling_identity != streamed_output.identity
                        or video_sha256 != streamed_output.raw_sha256
                        or video_byte_count != streamed_output.byte_count
                    ):
                        raise FinalPublicationValidationError(
                            "streamed sibling differs from persisted output evidence"
                        )
                    if self.expected_sibling_sha256 is not None and (
                        video_sha256 != self.expected_sibling_sha256
                        or video_byte_count != self.expected_sibling_byte_count
                    ):
                        raise FinalPublicationValidationError(
                            "sibling video bytes changed since container validation"
                        )
                    frozen_handle.flush()
                    os.fsync(frozen_handle.fileno())
                    after = os.fstat(frozen_handle.fileno())
                    if (
                        int(after.st_size) != int(before.st_size)
                        or int(after.st_mtime_ns) != int(before.st_mtime_ns)
                        or int(after.st_ctime_ns) != int(before.st_ctime_ns)
                        or file_identity_from_open_handle(frozen_handle) != frozen_sibling_identity
                        or acquired.persistent_regular_identity(sibling) != frozen_sibling_identity
                    ):
                        raise FinalPublicationValidationError(
                            "sibling video identity changed after sync"
                        )
        except (NoFollowPathError, FileNotFoundError) as error:
            raise FinalPublicationValidationError(
                "sibling video failed no-follow acquisition"
            ) from error
        final_manifest = None
        if (
            self.reservation is not None
            and streamed_output is None
            and self.expected_sibling_sha256 is not None
        ):
            reservation = self._materialize_reservation()
            if (
                reservation_phase != "ready_pre_video"
                or reservation.final_manifest_transaction is None
            ):
                raise ReservationConflictError(
                    "file-backed encode cannot replace a published video"
                )
            evidence = StreamedVrOutputEvidenceV1(
                reservation.index.fingerprint,
                self.sibling_relative_path,
                frozen_sibling_identity,
                video_sha256,
                video_byte_count,
            )
            final_manifest = build_final_video_manifest_v4(
                relative_path=relative_path,
                video_sha256=video_sha256,
                video_byte_count=video_byte_count,
                input_stage_fingerprint=self.input_stage_fingerprint,
                resolved_encoding_arguments=self.resolved_encoding_arguments,
                encoded_output=evidence,
            )
            # Sync the exact manifest we will later publish, without consuming its extent yet.
            # The input extent and encoded video are already durable at this boundary.
            _sync_parent(sibling.parent)
            reservation.final_manifest_transaction.stage_complete_full(final_manifest.encode())
            del reservation
            self._emit("encoded-output:synced")
        if self.reservation is not None:
            if reserved_target_parent_identity is None:
                raise ReservationConflictError("final target parent authority is missing")
            same_parent_replace_v1(
                sibling,
                final_path,
                expected_source_identity=frozen_sibling_identity,
                expected_target_identity=reserved_target_identity,
                expected_parent_identity=reserved_target_parent_identity,
                replace_existing=True,
            )
        else:
            durable_replace(
                sibling,
                final_path,
                expected_source_identity=frozen_sibling_identity,
                expected_target_identity=standalone_final_target_identity,
            )
        try:
            with AcquiredRoot(self.job_root) as acquired:
                with acquired.open_regular(final_path) as reopened_handle:
                    if file_identity_from_open_handle(reopened_handle) != frozen_sibling_identity:
                        raise FinalPublicationValidationError(
                            "final video identity changed during replacement"
                        )
                    reopened_hash, reopened_count = _hash_open_file(reopened_handle)
        except (NoFollowPathError, FileNotFoundError) as error:
            raise FinalPublicationValidationError(
                "final video target failed no-follow reopen"
            ) from error
        if (reopened_hash, reopened_count) != (video_sha256, video_byte_count):
            raise FinalPublicationValidationError("final video changed during replacement")
        self._emit("video:published")
        if crash_at == "after-video":
            raise InjectedFinalPublicationCrash(crash_at)

        if self.reservation is not None:
            reservation = self._materialize_reservation()
            input_transaction = reservation.input_manifest_transaction
            if input_transaction is None:
                committed = EncodingInputPublicationSource.from_path(
                    input_target,
                    expected_frame_count=expected_frame_count,
                )
                if committed.fingerprint != self.input_stage_fingerprint:
                    raise FinalPublicationValidationError(
                        "committed input manifest fingerprint differs"
                    )
            elif self.encoding_input_manifest_raw is not None:
                reservation.validate_indexed_predecessor(input_transaction)
                input_transaction.consume(self.encoding_input_manifest_raw)
                reservation.input_manifest_transaction = None
            else:
                assert self.encoding_input_source is not None
                source = self.encoding_input_source.validated(
                    expected_frame_count=expected_frame_count,
                )
                if source.path != input_transaction.source_path:
                    raise FinalPublicationValidationError(
                        "streamed input source is not the indexed extent"
                    )
                reservation.validate_indexed_predecessor(input_transaction)
                input_transaction.consume_streamed_prefix(
                    prefix_byte_count=source.raw_byte_count,
                    prefix_sha256=source.raw_sha256,
                )
                reservation.input_manifest_transaction = None
            del reservation
        elif self.encoding_input_source is not None:
            source = self.encoding_input_source.validated(
                expected_frame_count=expected_frame_count,
            )
            durable_replace(
                source.path,
                input_target,
                expected_target_identity=standalone_input_target_identity,
            )
            reopened = EncodingInputPublicationSource(
                path=input_target,
                fingerprint=source.fingerprint,
                raw_sha256=source.raw_sha256,
                raw_byte_count=source.raw_byte_count,
            )
            reopened.validated(expected_frame_count=expected_frame_count)
        else:
            assert self.encoding_input_manifest_raw is not None
            _atomic_replace(
                input_target,
                self.encoding_input_manifest_raw,
                expected_target_identity=standalone_input_target_identity,
            )
        self._emit("input-manifest:published")
        if crash_at == "after-input-manifest":
            raise InjectedFinalPublicationCrash(crash_at)

        final_manifest = final_manifest or build_final_video_manifest_v4(
            relative_path=relative_path,
            video_sha256=video_sha256,
            video_byte_count=video_byte_count,
            input_stage_fingerprint=self.input_stage_fingerprint,
            resolved_encoding_arguments=self.resolved_encoding_arguments,
        )
        if self.reservation is not None:
            reservation = self._materialize_reservation()
            final_transaction = reservation.final_manifest_transaction
            if final_transaction is None:
                raise FinalPublicationValidationError("final manifest reservation is consumed")
            try:
                reservation.validate_indexed_predecessor(final_transaction)
                if final_manifest.encoded_output is not None:
                    if crash_at == "after-final-intent-sync":
                        raise InjectedFinalPublicationCrash(crash_at)
                    final_transaction.recover(final_manifest.encode())
                else:
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
            reservation.final_manifest_transaction = None
            del reservation
        else:
            _atomic_replace(
                final_manifest_target,
                final_manifest.encode(),
                expected_target_identity=standalone_manifest_target_identity,
            )
        self._emit("final-manifest:published")
        if crash_at == "after-final-manifest":
            raise InjectedFinalPublicationCrash(crash_at)
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


def _verify_audited_video(
    root: Path,
    final_path: Path,
    manifest: FinalVideoManifestV4,
    input_projection: _EncodingInputManifestProjection,
) -> None:
    proof = manifest.encoded_output
    if proof is not None and input_projection.streamed_invocation is not None:
        raise FinalPublicationValidationError(
            "file-backed proof cannot authenticate streamed inputs"
        )
    scanned = _scan_fixed_file_v1(
        root, final_path, expected_identity=proof.identity if proof else None
    )
    video_hash, video_count = scanned.raw_sha256, scanned.byte_count
    if manifest.sha256 != video_hash or manifest.byte_count != video_count:
        raise FinalPublicationValidationError("final video identity differs")
    streamed_invocation = input_projection.streamed_invocation
    streamed_output = input_projection.streamed_output
    if streamed_invocation is not None:
        assert streamed_output is not None
        if (
            streamed_invocation.materialized_arguments(input_projection.fingerprint)
            != manifest.resolved_encoding_arguments
            or streamed_output.identity != capture_file_identity(final_path)
            or streamed_output.raw_sha256 != video_hash
            or streamed_output.byte_count != video_count
        ):
            raise FinalPublicationValidationError(
                "final streamed video differs from its launch evidence"
            )


def _authenticate_media_audit(
    root: Path,
    settings: ProcessingSettingsArtifactV5,
    settings_value: dict,
    expected: str,
    final_path: Path,
    input_path: Path,
    publication_path: Path,
    container_validator: Callable[[Path], bool] | None,
) -> None:
    if not all(_regular_nonlink(path) for path in (final_path, input_path, publication_path)):
        raise FinalPublicationValidationError("publication entry is unsafe")
    metadata = settings_value["metadata"]
    if (
        metadata["final_media_producer_contract_version"] != 4
        or metadata["final_media_publication_generation"] is None
    ):
        raise FinalPublicationValidationError("producer marker is invalid")
    input_projection = _stream_input_manifest_projection_v1(
        root,
        input_path,
        expected_frame_count=_settings_frame_count_v1(settings),
    )
    manifest = parse_final_video_manifest_v4(
        _read_bounded_regular_file(
            publication_path,
            max_bytes=FINAL_VIDEO_MANIFEST_MAX_RAW_BYTES,
        )
    )
    if (
        manifest.relative_path != expected
        or manifest.input_stage_fingerprint != input_projection.fingerprint
    ):
        raise FinalPublicationValidationError("three-way publication binding differs")
    _validate_normalized_argv(
        manifest.resolved_encoding_arguments,
        input_stage_fingerprint=input_projection.fingerprint,
        relative_path=expected,
        expected_mode=input_projection.mode,
    )
    _verify_audited_video(root, final_path, manifest, input_projection)
    if container_validator is None:
        expectation = _final_container_expectation_from_intent_v1(
            settings=settings,
            input_projection=input_projection,
            arguments=manifest.resolved_encoding_arguments,
        )
        valid = validate_final_container_v4(final_path, expectation)
    else:
        valid = container_validator(final_path)
    if not valid:
        raise FinalPublicationValidationError("final video container validation failed")


def audit_final_media_v1(
    job_root: Path,
    settings: ProcessingSettingsArtifactV5,
    *,
    container_validator: Callable[[Path], bool] | None = None,
) -> FinalMediaAuditDispositionV1:
    """Classify the actual V/I/P matrix without inventing prior generations."""

    root = Path(job_root)
    active_index = root / FINAL_ENCODING_RESERVATION_RELATIVE_PATH
    active_temporary = active_index.with_name(f".{active_index.name}.create-new.tmp")
    if any(path.exists() or path.is_symlink() for path in (active_index, active_temporary)):
        raise FinalPublicationValidationError(
            "ordinary final-media audit is forbidden while the fixed index is active"
        )
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
        _authenticate_media_audit(
            root,
            settings,
            settings_value,
            expected,
            final_path,
            input_path,
            publication_path,
            container_validator,
        )
    except (
        OSError,
        CanonicalJsonError,
        FinalPublicationError,
        FinalContainerProcessError,
        ReservationConflictError,
        ValueError,
    ):
        return FinalMediaAuditDispositionV1(
            True,
            False,
            "publication_validation_failed",
        )
    return FinalMediaAuditDispositionV1(True, True, None)
