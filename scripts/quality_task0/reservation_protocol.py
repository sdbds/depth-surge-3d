"""Indexed reservation and irreversible-publication Task 0 prototype."""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import time
from typing import Callable, Literal


RESERVATION_ZERO_FILL_CHUNK_BYTES = 64 * 1024
CONTROL_ARTIFACT_WRITE_CHUNK_BYTES = 4 * 1024
JOB_CONTROL_RELATIVE_PATH = Path("job-control-v1.json")
SETTINGS_ARTIFACT_RELATIVE_PATH = Path("processing-settings-v5.json")

PublicationMethod = Literal["replace_placeholder", "replace_existing"]


class ReservationError(RuntimeError):
    """Base reservation protocol error."""


class ReservationIncompleteError(ReservationError):
    """An indexed materialization or release proof is retryably incomplete."""


class ReservationCapabilityError(ReservationError):
    """The platform cannot provide the frozen allocation/rename evidence."""


class ReservationConflictError(ReservationError):
    """Observed bytes, identity, or namespace contradict durable authority."""


class ReservationEvidenceConflictError(ReservationConflictError):
    """Full-source allocation evidence changed after descriptor publication."""


class WindowsRenameProtocolError(ReservationConflictError):
    """The Windows Boolean result contradicts its observed physical state."""


class InjectedReservationCrash(RuntimeError):
    """A deterministic crash fixture stopped at a durable protocol boundary."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_canonical(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _allocate(raw: int, allocation_unit: int) -> int:
    if raw < 0 or allocation_unit <= 0:
        raise ValueError("allocation inputs must be nonnegative with a positive unit")
    return ((raw + allocation_unit - 1) // allocation_unit) * allocation_unit


def payload_logical_bytes(raw: int, allocation_unit: int) -> int:
    return _allocate(raw, allocation_unit) + allocation_unit


def payload_extent_create_charge(raw: int, allocation_unit: int) -> int:
    return payload_logical_bytes(raw, allocation_unit) + allocation_unit


def descriptor_create_new_charge(descriptor_raw: int, allocation_unit: int) -> int:
    return _allocate(descriptor_raw, allocation_unit) + 2 * allocation_unit


def placeholder_entry_charge(method: PublicationMethod, allocation_unit: int) -> int:
    if method == "replace_placeholder":
        return allocation_unit
    if method == "replace_existing":
        return 0
    raise ValueError("unknown reservation publication method")


def reservation_extent_charge(
    payload_raw: int,
    descriptor_raw: int,
    method: PublicationMethod,
    allocation_unit: int,
) -> int:
    return (
        payload_extent_create_charge(payload_raw, allocation_unit)
        + descriptor_create_new_charge(descriptor_raw, allocation_unit)
        + placeholder_entry_charge(method, allocation_unit)
    )


def reservation_directory_bootstrap_forecast(allocation_unit: int) -> int:
    if allocation_unit <= 0:
        raise ValueError("allocation unit must be positive")
    return 2 * allocation_unit


@dataclass(frozen=True)
class AdmissionForecast:
    byte_count: int
    is_strict_directory_upper_bound: bool
    namespace_enospc_error: type[ReservationIncompleteError]


def minimum_admission_forecast(
    *,
    payload_raw: int,
    descriptor_raw: int,
    method: PublicationMethod,
    allocation_unit: int,
) -> AdmissionForecast:
    return AdmissionForecast(
        reservation_extent_charge(payload_raw, descriptor_raw, method, allocation_unit),
        False,
        ReservationIncompleteError,
    )


def classify_payload_frame(raw: bytes, expected_payload: bytes) -> str:
    if not expected_payload or b"\x00" in expected_payload:
        raise ValueError("canonical reservation payload must be nonempty and NUL-free")
    if raw == expected_payload:
        return "complete-short"
    if raw and all(byte == 0 for byte in raw):
        return "zero"
    if len(raw) > len(expected_payload) and raw.startswith(expected_payload):
        tail = raw[len(expected_payload) :]
        if tail and all(byte == 0 for byte in tail):
            return "complete-full"
    return "partial"


def _sync_parent(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_file(path: Path, payload: dict, *, event: Callable[[str], None] | None) -> None:
    raw = canonical_json_bytes(payload)
    temporary = path.with_name(f".{path.name}.create-new.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ReservationConflictError(f"authoritative control artifact differs: {path}")
        temporary.unlink(missing_ok=True)
        return
    try:
        with temporary.open("w+b") as handle:
            for offset in range(0, len(raw), CONTROL_ARTIFACT_WRITE_CHUNK_BYTES):
                chunk = raw[offset : offset + CONTROL_ARTIFACT_WRITE_CHUNK_BYTES]
                if handle.write(chunk) != len(chunk):
                    raise OSError("short control-artifact write")
            if event is not None:
                event("index:temporary")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise ReservationConflictError(f"control artifact appeared during publication: {path}")
        os.replace(temporary, path)
        _sync_parent(path.parent)
        if path.read_bytes() != raw:
            raise ReservationConflictError(f"published control artifact failed reopen: {path}")
        if event is not None:
            event("index:published")
    finally:
        if not path.exists():
            temporary.unlink(missing_ok=True)


def publish_write_ahead_intent(
    path: Path,
    payload: dict,
    *,
    event: Callable[[str], None] | None = None,
) -> None:
    try:
        _write_json_file(Path(path), payload, event=event)
    except OSError as error:
        if error.errno == errno.ENOSPC or getattr(error, "winerror", None) == 112:
            raise ReservationIncompleteError(f"namespace reservation incomplete: {path}") from error
        raise


def materialize_after_index(
    index_path: Path,
    payload: dict,
    *,
    publisher: Callable[..., None] = publish_write_ahead_intent,
) -> None:
    try:
        publisher(Path(index_path), payload)
    except OSError as error:
        if error.errno == errno.ENOSPC or getattr(error, "winerror", None) == 112:
            raise ReservationIncompleteError(
                "indexed namespace materialization is incomplete"
            ) from error
        raise


@dataclass(frozen=True)
class SettingsWriteAheadIntent:
    reservation_generation: str
    target_settings_payload: dict
    target_settings_raw_sha256: str
    fingerprint: str

    @classmethod
    def create(
        cls,
        *,
        reservation_generation: str,
        target_settings_payload: dict,
    ) -> SettingsWriteAheadIntent:
        target_hash = sha256_canonical(target_settings_payload)
        base = {
            "algorithm_version": "settings-write-ahead-intent-v1",
            "reservation_generation": reservation_generation,
            "target_settings_payload": target_settings_payload,
            "target_settings_raw_sha256": target_hash,
        }
        return cls(
            reservation_generation,
            target_settings_payload,
            target_hash,
            sha256_canonical(base),
        )

    def to_object(self) -> dict:
        return {
            "algorithm_version": "settings-write-ahead-intent-v1",
            "reservation_generation": self.reservation_generation,
            "target_settings_payload": self.target_settings_payload,
            "target_settings_raw_sha256": self.target_settings_raw_sha256,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def load(cls, path: Path) -> SettingsWriteAheadIntent:
        raw = Path(path).read_bytes()
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ReservationConflictError("settings intent is malformed") from error
        if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
            raise ReservationConflictError("settings intent is noncanonical")
        expected_keys = {
            "algorithm_version",
            "reservation_generation",
            "target_settings_payload",
            "target_settings_raw_sha256",
            "fingerprint",
        }
        if (
            set(value) != expected_keys
            or value["algorithm_version"] != "settings-write-ahead-intent-v1"
        ):
            raise ReservationConflictError("settings intent schema is invalid")
        intent = cls.create(
            reservation_generation=value["reservation_generation"],
            target_settings_payload=value["target_settings_payload"],
        )
        if intent.to_object() != value:
            raise ReservationConflictError("settings intent hashes are invalid")
        return intent


@dataclass(frozen=True)
class ReservationAllocationEvidence:
    platform: str
    logical_coverage_byte_count: int
    allocated_byte_count: int
    token: str

    def to_object(self) -> dict:
        return {
            "platform": self.platform,
            "logical_coverage_byte_count": self.logical_coverage_byte_count,
            "allocated_byte_count": self.allocated_byte_count,
            "token": self.token,
        }

    @classmethod
    def from_object(cls, value: dict) -> ReservationAllocationEvidence:
        return cls(
            value["platform"],
            int(value["logical_coverage_byte_count"]),
            int(value["allocated_byte_count"]),
            value["token"],
        )


@dataclass(frozen=True)
class ReservationFileIdentity:
    device: int
    inode: int

    def to_object(self) -> dict:
        return {"device": self.device, "inode": self.inode}

    @classmethod
    def from_object(cls, value: dict) -> ReservationFileIdentity:
        return cls(int(value["device"]), int(value["inode"]))


def _identity(path: Path) -> ReservationFileIdentity:
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise ReservationConflictError(f"reservation path is not a regular file: {path}")
    return ReservationFileIdentity(int(metadata.st_dev), int(metadata.st_ino))


def _directory_identity(path: Path) -> ReservationFileIdentity:
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ReservationConflictError(f"job root is not a directory: {path}")
    return ReservationFileIdentity(int(metadata.st_dev), int(metadata.st_ino))


@dataclass(frozen=True)
class ReservationDescriptor:
    index_relative_path: str
    descriptor_relative_path: str
    source_relative_path: str
    target_relative_path: str
    reservation_kind: str
    publication_method: PublicationMethod
    payload_raw_max: int
    logical_byte_count: int
    allocation_unit: int
    generation: str
    reservation_id: str
    filesystem_identity: ReservationFileIdentity
    placeholder_identity: ReservationFileIdentity | None
    allocation_evidence_at_readiness: ReservationAllocationEvidence
    fingerprint: str

    def _base_object(self) -> dict:
        return {
            "schema_version": 1,
            "algorithm_version": "reservation-extent-task0-v1",
            "index_relative_path": self.index_relative_path,
            "descriptor_relative_path": self.descriptor_relative_path,
            "source_relative_path": self.source_relative_path,
            "target_relative_path": self.target_relative_path,
            "reservation_kind": self.reservation_kind,
            "publication_method": self.publication_method,
            "payload_raw_max": self.payload_raw_max,
            "logical_byte_count": self.logical_byte_count,
            "allocation_unit": self.allocation_unit,
            "generation": self.generation,
            "reservation_id": self.reservation_id,
            "filesystem_identity": self.filesystem_identity.to_object(),
            "placeholder_identity": (
                self.placeholder_identity.to_object()
                if self.placeholder_identity is not None
                else None
            ),
            "allocation_evidence_at_readiness": self.allocation_evidence_at_readiness.to_object(),
        }

    def to_object(self) -> dict:
        return {**self._base_object(), "fingerprint": self.fingerprint}

    @classmethod
    def build(cls, **values) -> ReservationDescriptor:
        provisional = cls(**values, fingerprint="")
        return cls(**values, fingerprint=sha256_canonical(provisional._base_object()))

    @classmethod
    def from_object(cls, value: dict) -> ReservationDescriptor:
        descriptor = cls(
            index_relative_path=value["index_relative_path"],
            descriptor_relative_path=value["descriptor_relative_path"],
            source_relative_path=value["source_relative_path"],
            target_relative_path=value["target_relative_path"],
            reservation_kind=value["reservation_kind"],
            publication_method=value["publication_method"],
            payload_raw_max=int(value["payload_raw_max"]),
            logical_byte_count=int(value["logical_byte_count"]),
            allocation_unit=int(value["allocation_unit"]),
            generation=value["generation"],
            reservation_id=value["reservation_id"],
            filesystem_identity=ReservationFileIdentity.from_object(value["filesystem_identity"]),
            placeholder_identity=(
                ReservationFileIdentity.from_object(value["placeholder_identity"])
                if value["placeholder_identity"] is not None
                else None
            ),
            allocation_evidence_at_readiness=ReservationAllocationEvidence.from_object(
                value["allocation_evidence_at_readiness"]
            ),
            fingerprint=value["fingerprint"],
        )
        if descriptor.fingerprint != sha256_canonical(descriptor._base_object()):
            raise ReservationConflictError("reservation descriptor fingerprint is invalid")
        return descriptor


@dataclass
class ReservationMetrics:
    reservation_write_bytes: int = 0
    reservation_fsync_count: int = 0
    reservation_wall_time_ms: int = 0


AllocationAdapter = Callable[[Path, int], ReservationAllocationEvidence]
AllocationCounter = Callable[[Path], int]


def _default_counter(path: Path, allocation_unit: int) -> int:
    return _allocate(path.stat().st_size, allocation_unit)


class ReservedArtifactTransaction:
    """One target-local extent with phase-discriminated crash recovery."""

    def __init__(
        self,
        *,
        job_root: Path,
        descriptor: ReservationDescriptor,
        descriptor_path: Path,
        allocation_adapter: AllocationAdapter,
        allocation_counter: AllocationCounter | None,
        metrics: ReservationMetrics | None = None,
    ) -> None:
        self.job_root = Path(job_root)
        self.descriptor = descriptor
        self.descriptor_path = Path(descriptor_path)
        self.allocation_adapter = allocation_adapter
        self.allocation_counter = allocation_counter or (
            lambda path: _default_counter(path, descriptor.allocation_unit)
        )
        self.metrics = metrics or ReservationMetrics()

    @property
    def source_path(self) -> Path:
        return self.job_root / self.descriptor.source_relative_path

    @property
    def target_path(self) -> Path:
        return self.job_root / self.descriptor.target_relative_path

    @classmethod
    def create(  # noqa: C901
        cls,
        *,
        job_root: Path,
        index_relative_path: Path,
        target_relative_path: Path,
        reservation_kind: str,
        payload_raw_max: int,
        allocation_unit: int,
        allocation_adapter: AllocationAdapter,
        generation: str,
        reservation_id: str,
        allocation_counter: AllocationCounter | None = None,
        event: Callable[[str], None] | None = None,
    ) -> ReservedArtifactTransaction:
        started = time.monotonic_ns()
        job_root = Path(job_root)
        index_path = job_root / index_relative_path
        if not index_path.is_file():
            raise ReservationConflictError("reservation extent has no durable write-ahead index")
        target_path = job_root / target_relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            publication_method: PublicationMethod = "replace_existing"
            placeholder_identity = None
        else:
            publication_method = "replace_placeholder"
            try:
                with target_path.open("xb") as placeholder:
                    placeholder.flush()
                    os.fsync(placeholder.fileno())
            except OSError as error:
                if error.errno == errno.ENOSPC or getattr(error, "winerror", None) == 112:
                    raise ReservationIncompleteError(
                        "placeholder creation is incomplete"
                    ) from error
                raise
            _sync_parent(target_path.parent)
            placeholder_identity = _identity(target_path)
            if event is not None:
                event("placeholder:created")
        logical = payload_logical_bytes(payload_raw_max, allocation_unit)
        source_name = f".depth-surge-r-{generation}-{reservation_kind}-{reservation_id}.reserved"
        source_path = target_path.parent / source_name
        metrics = ReservationMetrics()
        try:
            with source_path.open("xb") as handle:
                if event is not None:
                    event("extent:created")
                remaining = logical
                zero_chunk = b"\x00" * RESERVATION_ZERO_FILL_CHUNK_BYTES
                while remaining:
                    chunk = zero_chunk[: min(remaining, len(zero_chunk))]
                    written = handle.write(chunk)
                    if written != len(chunk):
                        raise OSError("short reservation zero fill")
                    metrics.reservation_write_bytes += written
                    remaining -= written
                handle.flush()
                os.fsync(handle.fileno())
                metrics.reservation_fsync_count += 1
        except OSError as error:
            if error.errno == errno.ENOSPC or getattr(error, "winerror", None) == 112:
                raise ReservationIncompleteError("reservation extent fill is incomplete") from error
            raise
        if (
            source_path.stat().st_size != logical
            or classify_payload_frame(source_path.read_bytes(), b"{}") != "zero"
        ):
            raise ReservationIncompleteError("reservation extent is not a full zero frame")
        evidence = allocation_adapter(source_path, logical)
        if evidence.logical_coverage_byte_count != logical:
            raise ReservationIncompleteError("reservation allocation coverage is incomplete")
        if event is not None:
            event("extent:ready")
        source_identity = _identity(source_path)
        descriptor_name = f"d-r-{generation}-{reservation_kind}-{reservation_id}.json"
        descriptor_relative = Path(".depth-surge-reservations-v1") / descriptor_name
        descriptor = ReservationDescriptor.build(
            index_relative_path=Path(index_relative_path).as_posix(),
            descriptor_relative_path=descriptor_relative.as_posix(),
            source_relative_path=source_path.relative_to(job_root).as_posix(),
            target_relative_path=Path(target_relative_path).as_posix(),
            reservation_kind=reservation_kind,
            publication_method=publication_method,
            payload_raw_max=payload_raw_max,
            logical_byte_count=logical,
            allocation_unit=allocation_unit,
            generation=generation,
            reservation_id=reservation_id,
            filesystem_identity=source_identity,
            placeholder_identity=placeholder_identity,
            allocation_evidence_at_readiness=evidence,
        )
        descriptor_path = job_root / descriptor_relative
        _write_json_file(descriptor_path, descriptor.to_object(), event=None)
        if event is not None:
            event("descriptor:published")
        metrics.reservation_wall_time_ms = (time.monotonic_ns() - started) // 1_000_000
        return cls(
            job_root=job_root,
            descriptor=descriptor,
            descriptor_path=descriptor_path,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
            metrics=metrics,
        )

    @classmethod
    def load(
        cls,
        descriptor_path: Path,
        *,
        allocation_adapter: AllocationAdapter,
        allocation_counter: AllocationCounter | None = None,
    ) -> ReservedArtifactTransaction:
        descriptor_path = Path(descriptor_path)
        raw = descriptor_path.read_bytes()
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ReservationConflictError("reservation descriptor is malformed") from error
        if canonical_json_bytes(value) != raw:
            raise ReservationConflictError("reservation descriptor is noncanonical")
        descriptor = ReservationDescriptor.from_object(value)
        job_root = descriptor_path.parent.parent
        if descriptor_path != job_root / descriptor.descriptor_relative_path:
            raise ReservationConflictError("reservation descriptor path is incorrect")
        return cls(
            job_root=job_root,
            descriptor=descriptor,
            descriptor_path=descriptor_path,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
        )

    def _validate_placeholder(self) -> None:
        if self.descriptor.publication_method == "replace_placeholder":
            if (
                self.descriptor.placeholder_identity is None
                or not self.target_path.exists()
                or self.target_path.stat().st_size != 0
                or _identity(self.target_path) != self.descriptor.placeholder_identity
            ):
                raise ReservationConflictError("reserved target placeholder changed")

    def validate_physical_phase(self, expected_payload: bytes) -> str:
        source_exists = self.source_path.exists()
        target_exists = self.target_path.exists()
        if source_exists:
            if _identity(self.source_path) != self.descriptor.filesystem_identity:
                raise ReservationConflictError("reservation source identity changed")
            self._validate_placeholder()
            size = self.source_path.stat().st_size
            if size == self.descriptor.logical_byte_count:
                evidence = self.allocation_adapter(self.source_path, size)
                if evidence != self.descriptor.allocation_evidence_at_readiness:
                    raise ReservationEvidenceConflictError("full-source readiness evidence changed")
                return "full-source"
            if size == len(expected_payload) and self.source_path.read_bytes() == expected_payload:
                return "short-source"
            raise ReservationConflictError("reservation source length/content is invalid")
        if target_exists:
            if (
                _identity(self.target_path) == self.descriptor.filesystem_identity
                and self.target_path.stat().st_size == len(expected_payload)
                and self.target_path.read_bytes() == expected_payload
            ):
                return "committed-target"
            raise ReservationConflictError(
                "reservation target does not contain the committed extent"
            )
        raise ReservationConflictError("reserved extent exists at neither authorized location")

    def stage_complete_full(self, payload: bytes) -> None:
        if not payload or len(payload) > self.descriptor.payload_raw_max or b"\x00" in payload:
            raise ValueError("reservation payload is outside its canonical raw bound")
        if self.validate_physical_phase(payload) != "full-source":
            raise ReservationConflictError("reservation source is not full")
        with self.source_path.open("r+b") as handle:
            handle.seek(0)
            if handle.write(payload) != len(payload):
                raise OSError("short reservation payload write")
            handle.flush()
            os.fsync(handle.fileno())
        evidence = self.allocation_adapter(self.source_path, self.descriptor.logical_byte_count)
        if evidence != self.descriptor.allocation_evidence_at_readiness:
            raise ReservationEvidenceConflictError("complete-full readiness evidence changed")
        if classify_payload_frame(self.source_path.read_bytes(), payload) != "complete-full":
            raise ReservationConflictError("reservation source is not exact complete-full")

    def truncate_to_payload(self, payload: bytes) -> None:
        if classify_payload_frame(self.source_path.read_bytes(), payload) != "complete-full":
            raise ReservationConflictError("only complete-full may enter truncate")
        with self.source_path.open("r+b") as handle:
            handle.truncate(len(payload))

    def _short_release_recovery(self, payload: bytes) -> None:
        if self.validate_physical_phase(payload) != "short-source":
            raise ReservationConflictError("short release recovery requires complete-short")
        with self.source_path.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        if self.source_path.read_bytes() != payload:
            raise ReservationConflictError("short source changed after file sync")
        threshold = (
            self.descriptor.allocation_evidence_at_readiness.allocated_byte_count
            - self.descriptor.allocation_unit
        )
        if self.allocation_counter(self.source_path) <= threshold:
            return
        with self.source_path.open("r+b") as handle:
            handle.truncate(len(payload))
            handle.flush()
            os.fsync(handle.fileno())
        if self.source_path.read_bytes() != payload:
            raise ReservationConflictError("short source changed after retry truncate")
        if self.allocation_counter(self.source_path) > threshold:
            raise ReservationIncompleteError("short-source allocation release remains unproved")

    def _rename_and_sync(self) -> None:
        self._validate_placeholder()
        os.replace(self.source_path, self.target_path)
        _sync_parent(self.target_path.parent)

    def _retire_descriptor(self) -> None:
        self.descriptor_path.unlink(missing_ok=True)
        _sync_parent(self.descriptor_path.parent)

    def consume(self, payload: bytes, *, crash_at: str | None = None) -> str:
        self.stage_complete_full(payload)
        self.truncate_to_payload(payload)
        if crash_at == "after-truncate":
            raise InjectedReservationCrash(crash_at)
        with self.source_path.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        if crash_at == "after-truncate-sync":
            raise InjectedReservationCrash(crash_at)
        self._short_release_recovery(payload)
        self._rename_and_sync()
        if crash_at == "after-rename-parent-sync":
            raise InjectedReservationCrash(crash_at)
        if self.validate_physical_phase(payload) != "committed-target":
            raise ReservationConflictError("target commit validation failed")
        self._retire_descriptor()
        return "committed-target"

    def recover(self, payload: bytes) -> str:
        phase = self.validate_physical_phase(payload)
        if phase == "full-source":
            state = classify_payload_frame(self.source_path.read_bytes(), payload)
            if state != "complete-full":
                raise ReservationConflictError("recovery full source has no semantic intent")
            self.truncate_to_payload(payload)
            phase = "short-source"
        if phase == "short-source":
            self._short_release_recovery(payload)
            self._rename_and_sync()
            phase = "committed-target"
        if phase == "committed-target":
            if self.validate_physical_phase(payload) != "committed-target":
                raise ReservationConflictError("committed target changed during recovery")
            self._retire_descriptor()
            return phase
        raise ReservationConflictError(f"unsupported reservation recovery phase: {phase}")


def classify_fiemap_batch(
    *,
    mapped_extents: int,
    extent_capacity: int,
    last_flag: bool,
    previous_start: int,
    next_start: int,
    coverage_complete: bool,
) -> int | str:
    if mapped_extents < 0 or mapped_extents > extent_capacity or extent_capacity <= 0:
        raise ReservationCapabilityError("FIEMAP returned an invalid mapped count")
    if mapped_extents == 0 and not coverage_complete:
        raise ReservationIncompleteError("FIEMAP coverage has an unready gap")
    if 0 < mapped_extents < extent_capacity and not last_flag:
        raise ReservationCapabilityError("FIEMAP returned a non-full batch without LAST")
    if last_flag:
        return "done"
    if next_start <= previous_start:
        raise ReservationCapabilityError("FIEMAP pagination made no forward progress")
    return next_start


def classify_windows_rename_return(
    *,
    source_identity: str | None,
    target_identity: str | None,
    expected_source_identity: str,
    expected_old_target_identity: str,
    api_return: bool,
) -> str:
    if (
        source_identity == expected_source_identity
        and target_identity == expected_old_target_identity
    ):
        if api_return:
            raise WindowsRenameProtocolError("true rename return left both identities unchanged")
        return "unchanged"
    if source_identity is None and target_identity == expected_source_identity:
        return "committed"
    if source_identity == expected_source_identity and target_identity is None:
        return "replacement-gap"
    raise ReservationConflictError("Windows rename returned an unauthorized identity combination")


def classify_final_publication_prefix(
    *,
    active_index: bool,
    video_generation: str,
    input_manifest_generation: str,
    final_manifest_generation: str,
    sibling_present: bool,
) -> str:
    if not active_index:
        if (
            video_generation == input_manifest_generation == final_manifest_generation
            and not sibling_present
        ):
            return "publication-authenticated"
        return "publication-validation-failed"
    if video_generation != "new":
        if sibling_present and video_generation == "old":
            return "pre-boundary-sibling-present"
        return "pre-boundary"
    if sibling_present:
        return "post-boundary-retry-sibling-present"
    if input_manifest_generation != "new":
        return "video-replaced-input-pending"
    if final_manifest_generation != "new":
        return "input-manifest-committed-no-final-intent"
    return "publication-authenticated"


def validate_final_encoding_entry_order(kinds: list[str]) -> None:
    if not kinds or kinds[-1] != "job_terminal_settings":
        raise ReservationConflictError("job terminal reservation must be last")
    cursor = 0
    if kinds[cursor : cursor + 1] == ["producer_settings"]:
        cursor += 1
    required = ["encoding_input_manifest", "final_video_manifest"]
    if kinds[cursor : cursor + 2] != required:
        raise ReservationConflictError("final manifest entries are out of order")
    cursor += 2
    while cursor < len(kinds) - 1 and kinds[cursor] == "prune_marker":
        cursor += 1
    if cursor < len(kinds) - 1 and kinds[cursor] == "diagnostics_payload_pruned":
        cursor += 1
    if cursor < len(kinds) - 1 and kinds[cursor] == "cleanup_pending_settings":
        cursor += 1
    if cursor != len(kinds) - 1:
        raise ReservationConflictError("final encoding reservation entries are out of order")


def _require_lower_hex(value: object, digits: int, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != digits
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReservationConflictError(f"{label} must be {digits} lowercase hexadecimal digits")
    return value


@dataclass(frozen=True)
class JobControlV1:
    """Closed Task 0 form of the fixed schema-5 job locator."""

    job_id: str
    legacy_settings_artifact_relative_path: str | None
    settings_identity_fingerprint: str
    job_reservation_generation: str
    initial_settings_reservation_id: str
    job_terminal_reservation_id: str
    job_root_identity: ReservationFileIdentity
    bootstrap_extent_entries: tuple[dict, ...]
    bootstrap_settings_intent: dict
    fingerprint: str

    def _base_object(self) -> dict:
        return {
            "schema_version": 1,
            "algorithm_version": "job-control-v1",
            "job_id": self.job_id,
            "settings_artifact_relative_path": SETTINGS_ARTIFACT_RELATIVE_PATH.as_posix(),
            "legacy_settings_artifact_relative_path": self.legacy_settings_artifact_relative_path,
            "settings_identity_fingerprint": self.settings_identity_fingerprint,
            "job_reservation_generation": self.job_reservation_generation,
            "initial_settings_reservation_id": self.initial_settings_reservation_id,
            "job_terminal_reservation_id": self.job_terminal_reservation_id,
            "job_root_identity": self.job_root_identity.to_object(),
            "bootstrap_extent_entries": list(self.bootstrap_extent_entries),
            "bootstrap_settings_intent": self.bootstrap_settings_intent,
        }

    def to_object(self) -> dict:
        return {**self._base_object(), "fingerprint": self.fingerprint}

    @classmethod
    def create(
        cls,
        *,
        job_root: Path,
        job_id: str,
        settings_identity_fingerprint: str,
        job_reservation_generation: str,
        initial_settings_reservation_id: str,
        job_terminal_reservation_id: str,
        legacy_settings_artifact_relative_path: str | None,
        bootstrap_extent_entries: list[dict],
        bootstrap_settings_intent: dict,
    ) -> JobControlV1:
        _require_lower_hex(job_id, 32, "job_id")
        _require_lower_hex(settings_identity_fingerprint, 64, "settings identity")
        _require_lower_hex(job_reservation_generation, 32, "job reservation generation")
        _require_lower_hex(initial_settings_reservation_id, 32, "initial reservation id")
        _require_lower_hex(job_terminal_reservation_id, 32, "terminal reservation id")
        if legacy_settings_artifact_relative_path is not None:
            legacy = legacy_settings_artifact_relative_path
            if (
                not legacy.endswith("-settings.json")
                or legacy in {".", ".."}
                or any(character in legacy for character in ("\x00", "/", "\\"))
                or (
                    len(legacy) >= 2
                    and legacy[0].isascii()
                    and legacy[0].isalpha()
                    and legacy[1] == ":"
                )
            ):
                raise ReservationConflictError("legacy settings component is not contained")
        kinds = [entry.get("reservation_kind") for entry in bootstrap_extent_entries]
        if kinds not in (
            ["initial_settings", "job_terminal_settings"],
            ["migration_settings", "job_terminal_settings"],
        ):
            raise ReservationConflictError("job-control bootstrap entry order is invalid")
        provisional = cls(
            job_id=job_id,
            legacy_settings_artifact_relative_path=legacy_settings_artifact_relative_path,
            settings_identity_fingerprint=settings_identity_fingerprint,
            job_reservation_generation=job_reservation_generation,
            initial_settings_reservation_id=initial_settings_reservation_id,
            job_terminal_reservation_id=job_terminal_reservation_id,
            job_root_identity=_directory_identity(Path(job_root)),
            bootstrap_extent_entries=tuple(bootstrap_extent_entries),
            bootstrap_settings_intent=bootstrap_settings_intent,
            fingerprint="",
        )
        return cls(
            **{
                **provisional.__dict__,
                "fingerprint": sha256_canonical(provisional._base_object()),
            }
        )

    @classmethod
    def from_object(cls, value: object) -> JobControlV1:
        if not isinstance(value, dict):
            raise ReservationConflictError("job control root is not an object")
        expected = {
            "schema_version",
            "algorithm_version",
            "job_id",
            "settings_artifact_relative_path",
            "legacy_settings_artifact_relative_path",
            "settings_identity_fingerprint",
            "job_reservation_generation",
            "initial_settings_reservation_id",
            "job_terminal_reservation_id",
            "job_root_identity",
            "bootstrap_extent_entries",
            "bootstrap_settings_intent",
            "fingerprint",
        }
        if set(value) != expected:
            raise ReservationConflictError("job control has missing or extra keys")
        if value["schema_version"] != 1 or value["algorithm_version"] != "job-control-v1":
            raise ReservationConflictError("job control version is invalid")
        if value["settings_artifact_relative_path"] != SETTINGS_ARTIFACT_RELATIVE_PATH.as_posix():
            raise ReservationConflictError("job control settings locator is not fixed")
        entries = value["bootstrap_extent_entries"]
        intent = value["bootstrap_settings_intent"]
        if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
            raise ReservationConflictError("job control bootstrap entries are invalid")
        if not isinstance(intent, dict):
            raise ReservationConflictError("job control bootstrap intent is invalid")
        result = cls(
            job_id=_require_lower_hex(value["job_id"], 32, "job_id"),
            legacy_settings_artifact_relative_path=value["legacy_settings_artifact_relative_path"],
            settings_identity_fingerprint=_require_lower_hex(
                value["settings_identity_fingerprint"], 64, "settings identity"
            ),
            job_reservation_generation=_require_lower_hex(
                value["job_reservation_generation"], 32, "job reservation generation"
            ),
            initial_settings_reservation_id=_require_lower_hex(
                value["initial_settings_reservation_id"], 32, "initial reservation id"
            ),
            job_terminal_reservation_id=_require_lower_hex(
                value["job_terminal_reservation_id"], 32, "terminal reservation id"
            ),
            job_root_identity=ReservationFileIdentity.from_object(value["job_root_identity"]),
            bootstrap_extent_entries=tuple(entries),
            bootstrap_settings_intent=intent,
            fingerprint=_require_lower_hex(value["fingerprint"], 64, "job control fingerprint"),
        )
        if result.fingerprint != sha256_canonical(result._base_object()):
            raise ReservationConflictError("job control fingerprint is invalid")
        return result


def publish_job_control(job_root: Path, control: JobControlV1) -> None:
    job_root = Path(job_root)
    if _directory_identity(job_root) != control.job_root_identity:
        raise ReservationConflictError("job control belongs to a different job root")
    publish_write_ahead_intent(job_root / JOB_CONTROL_RELATIVE_PATH, control.to_object())


def load_job_control(job_root: Path) -> JobControlV1:
    job_root = Path(job_root)
    path = job_root / JOB_CONTROL_RELATIVE_PATH
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReservationConflictError("job control is missing or malformed") from error
    if canonical_json_bytes(value) != raw:
        raise ReservationConflictError("job control is noncanonical")
    control = JobControlV1.from_object(value)
    if _directory_identity(job_root) != control.job_root_identity:
        raise ReservationConflictError("job root identity differs from job control")
    return control


def resolve_fixed_settings_path(job_root: Path, control: JobControlV1) -> Path:
    job_root = Path(job_root)
    if _directory_identity(job_root) != control.job_root_identity:
        raise ReservationConflictError("job root identity differs from job control")
    path = job_root / SETTINGS_ARTIFACT_RELATIVE_PATH
    if not path.is_file() or path.is_symlink():
        raise ReservationConflictError("fixed settings artifact is missing or unsafe")
    return path


@dataclass(frozen=True)
class JobTerminalObservation:
    current_status: str
    current_revision: int
    source_frame: str
    descriptor_present: bool
    descriptor_temporary_present: bool = False
    target_has_reserved_identity: bool = False
    intent_status: str | None = None
    intent_revision: int | None = None
    intent_is_legal: bool = False
    fixed_index_active: bool = False


def reconcile_job_control_terminal_state(  # noqa: C901
    observation: JobTerminalObservation,
) -> str:
    if observation.fixed_index_active:
        return "defer-to-fixed-index"
    if observation.current_revision < 0:
        raise ReservationConflictError("settings revision is invalid")
    if observation.current_status == "in_progress":
        if not observation.descriptor_present:
            raise ReservationConflictError("in-progress terminal reserve has no descriptor")
        if observation.source_frame == "zero":
            return "retain-zero-terminal-reserve"
        if observation.source_frame == "partial":
            return "reset-partial-terminal-reserve"
        if observation.source_frame in {"complete-full", "complete-short"}:
            if (
                observation.intent_is_legal
                and observation.intent_status in {"completed", "failed"}
                and observation.intent_revision == observation.current_revision + 1
            ):
                return "publish-exact-terminal-intent"
            raise ReservationConflictError("terminal intent is not the unique legal next revision")
        raise ReservationConflictError("in-progress terminal extent is in an illegal state")
    if observation.current_status in {"completed", "failed"}:
        if observation.source_frame != "absent":
            raise ReservationConflictError("terminal settings retain an unexpected source extent")
        if observation.descriptor_present:
            if not observation.target_has_reserved_identity:
                raise ReservationConflictError("terminal target identity is not reserved")
            return "retire-consumed-terminal-descriptor"
        if observation.descriptor_temporary_present:
            return "retire-residual-terminal-temporary"
        return "terminal-reserve-consumed"
    raise ReservationConflictError("current settings status is outside the closed lifecycle")


def recovery_audit_order() -> tuple[str, ...]:
    return (
        "job-control-publication",
        "job-control-bootstrap",
        "fixed-index-classification",
        "settings-transition-recovery",
        "final-encoding-recovery",
        "job-control-owned-extents",
        "ordinary-artifact-audit",
    )


@dataclass(frozen=True)
class FinalPublicationObservation:
    video_state: str
    input_manifest_state: str
    final_manifest_state: str
    sibling_present: bool


@dataclass(frozen=True)
class FinalPublicationRecovery:
    phase: str
    action: str


def classify_final_publication_observation(  # noqa: C901
    observation: FinalPublicationObservation,
) -> FinalPublicationRecovery:
    video = observation.video_state
    input_state = observation.input_manifest_state
    final_state = observation.final_manifest_state
    sibling = observation.sibling_present
    if video == "reserved-target":
        if input_state not in {"zero", "partial", "complete-full", "encoding-prepass-staged"}:
            raise ReservationConflictError("pre-boundary input manifest state is illegal")
        if final_state != "zero":
            raise ReservationConflictError("final-manifest intent exists before video publication")
        action = "remove-sibling-and-encode" if sibling else "encode"
        return FinalPublicationRecovery("ready_pre_video", action)
    if video == "windows-replacement-gap":
        if not sibling or final_state != "zero":
            raise ReservationConflictError("Windows replacement gap has an illegal namespace")
        if input_state == "complete-full":
            return FinalPublicationRecovery(
                "windows_video_replace_gap_before_input_commit", "forward-rename-video"
            )
        if input_state == "committed-new":
            return FinalPublicationRecovery(
                "windows_video_replace_gap_after_input_commit", "forward-rename-video"
            )
        raise ReservationConflictError("Windows replacement gap has no legal input prefix")
    if video != "committed-new":
        raise ReservationConflictError("final video is outside the indexed physical states")
    if input_state in {"complete-full", "complete-short", "replacement-gap"}:
        if sibling or final_state != "zero":
            raise ReservationConflictError("input intent is not the unique post-video prefix")
        return FinalPublicationRecovery("video_replaced_input_intent", "commit-input-manifest")
    if input_state != "committed-new":
        raise ReservationConflictError("post-boundary input manifest has no semantic intent")
    if final_state in {"zero", "partial"}:
        if sibling:
            return FinalPublicationRecovery(
                "post_boundary_retry_sibling_present",
                "reset-final-intent-remove-sibling-reencode",
            )
        return FinalPublicationRecovery(
            "input_manifest_committed_no_final_intent", "revalidate-input-and-reencode"
        )
    if sibling:
        raise ReservationConflictError("complete final intent cannot coexist with a sibling")
    if final_state in {"complete-full", "complete-short", "replacement-gap"}:
        return FinalPublicationRecovery(
            "final_manifest_intent", "validate-and-commit-final-manifest"
        )
    if final_state == "committed-new":
        return FinalPublicationRecovery("publication_authenticated", "continue-finalization")
    raise ReservationConflictError("final manifest is outside the indexed physical states")


def final_index_retirement_order(
    *,
    entry_names: tuple[str, ...],
    descriptor_names: tuple[str, ...],
) -> tuple[str, ...]:
    if len(entry_names) != len(descriptor_names):
        raise ValueError("every final-encoding entry requires one descriptor name")
    steps = [
        "validate-terminal-settings",
        "reconcile-sibling-and-control-temporary",
    ]
    steps.extend(f"reconcile:{name}" for name in entry_names)
    steps.append("reconcile:final-video-target")
    steps.extend(f"unlink:{name}" for name in descriptor_names)
    steps.extend(
        (
            "unlink:final-video-target-descriptor",
            "sync-and-prove-descriptors-absent",
            "unlink:final-encoding-reservation-v1.json",
            "sync-and-prove-index-absent",
        )
    )
    return tuple(steps)
