"""Indexed reservation and irreversible-publication production protocol."""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import sys
import time
from typing import TYPE_CHECKING, Callable, Literal

from .canonical_json import CanonicalJsonError, strict_canonical_json_loads

if TYPE_CHECKING:
    from .settings_artifact import ProcessingSettingsArtifactV5


RESERVATION_ZERO_FILL_CHUNK_BYTES = 64 * 1024
CONTROL_ARTIFACT_WRITE_CHUNK_BYTES = 4 * 1024
JOB_CONTROL_RELATIVE_PATH = Path("job-control-v1.json")
SETTINGS_ARTIFACT_RELATIVE_PATH = Path("processing-settings-v5.json")
SETTINGS_TRANSITION_INDEX_RELATIVE_PATH = Path(
    ".depth-surge-reservations-v1/settings-transition-reservation-v1.json"
)

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


def _windows_standard_info(path: Path) -> tuple[int, int]:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class FileStandardInfo(ctypes.Structure):
        _fields_ = [
            ("AllocationSize", ctypes.c_longlong),
            ("EndOfFile", ctypes.c_longlong),
            ("NumberOfLinks", wintypes.DWORD),
            ("DeletePending", wintypes.BOOLEAN),
            ("Directory", wintypes.BOOLEAN),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_info = kernel32.GetFileInformationByHandleEx
    get_info.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    get_info.restype = wintypes.BOOL
    with Path(path).open("rb", buffering=0) as handle:
        native = wintypes.HANDLE(msvcrt.get_osfhandle(handle.fileno()))
        info = FileStandardInfo()
        if not get_info(native, 1, ctypes.byref(info), ctypes.sizeof(info)):
            error = ctypes.get_last_error()
            raise ReservationCapabilityError(
                f"GetFileInformationByHandleEx(FileStandardInfo) failed: {error}"
            )
    return int(info.EndOfFile), int(info.AllocationSize)


def _windows_allocated_ranges(path: Path, logical: int) -> tuple[tuple[int, int], ...]:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class FileAllocatedRangeBuffer(ctypes.Structure):
        _fields_ = [
            ("FileOffset", ctypes.c_longlong),
            ("Length", ctypes.c_longlong),
        ]

    fsctl_query_allocated_ranges = 0x000940CF
    error_more_data = 234
    extent_capacity = 64
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    device_io = kernel32.DeviceIoControl
    device_io.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    device_io.restype = wintypes.BOOL
    ranges: list[tuple[int, int]] = []
    start = 0
    with Path(path).open("rb", buffering=0) as handle:
        native = wintypes.HANDLE(msvcrt.get_osfhandle(handle.fileno()))
        while start < logical:
            query = FileAllocatedRangeBuffer(start, logical - start)
            output = (FileAllocatedRangeBuffer * extent_capacity)()
            returned = wintypes.DWORD()
            ctypes.set_last_error(0)
            ok = bool(
                device_io(
                    native,
                    fsctl_query_allocated_ranges,
                    ctypes.byref(query),
                    ctypes.sizeof(query),
                    ctypes.byref(output),
                    ctypes.sizeof(output),
                    ctypes.byref(returned),
                    None,
                )
            )
            error = ctypes.get_last_error()
            if not ok and error != error_more_data:
                raise ReservationCapabilityError(f"FSCTL_QUERY_ALLOCATED_RANGES failed: {error}")
            if returned.value % ctypes.sizeof(FileAllocatedRangeBuffer):
                raise ReservationCapabilityError(
                    "FSCTL_QUERY_ALLOCATED_RANGES returned a partial record"
                )
            count = returned.value // ctypes.sizeof(FileAllocatedRangeBuffer)
            if count == 0:
                raise ReservationIncompleteError("allocated-range coverage has a gap")
            for index in range(count):
                offset = int(output[index].FileOffset)
                length = int(output[index].Length)
                if length <= 0 or offset < start:
                    raise ReservationCapabilityError("allocated-range records are invalid")
                ranges.append((offset, length))
            next_start = ranges[-1][0] + ranges[-1][1]
            if next_start <= start:
                raise ReservationCapabilityError("allocated-range pagination made no progress")
            start = next_start
            if ok:
                break
    return tuple(ranges)


def _linux_fiemap_ranges(path: Path, logical: int) -> tuple[tuple[int, int], ...]:
    import fcntl

    fiemap_ioctl = 0xC020660B
    fiemap_flag_sync = 0x00000001
    extent_last = 0x00000001
    forbidden_flags = 0x00000002 | 0x00000004 | 0x00000008 | 0x00000080 | 0x00000100
    forbidden_flags |= 0x00000200 | 0x00000400 | 0x00000800 | 0x00001000
    extent_capacity = 64
    header_size = 32
    extent_size = 56
    ranges: list[tuple[int, int]] = []
    start = 0
    with Path(path).open("rb", buffering=0) as handle:
        while start < logical:
            buffer = bytearray(header_size + extent_capacity * extent_size)
            struct.pack_into(
                "=QQIIII",
                buffer,
                0,
                start,
                logical - start,
                fiemap_flag_sync,
                0,
                extent_capacity,
                0,
            )
            try:
                fcntl.ioctl(  # type: ignore[attr-defined]
                    handle.fileno(), fiemap_ioctl, buffer, True
                )
            except OSError as error:
                raise ReservationCapabilityError("FIEMAP ioctl failed") from error
            _, _, _, mapped, capacity, _ = struct.unpack_from("=QQIIII", buffer, 0)
            if capacity != extent_capacity:
                raise ReservationCapabilityError("FIEMAP changed extent capacity")
            last_flag = False
            next_start = start
            for index in range(mapped):
                record = struct.unpack_from(
                    "=QQQQQIIII",
                    buffer,
                    header_size + index * extent_size,
                )
                offset, length, flags = int(record[0]), int(record[2]), int(record[5])
                if length <= 0 or offset < start or flags & forbidden_flags:
                    raise ReservationCapabilityError("FIEMAP extent flags or order are unsupported")
                ranges.append((offset, length))
                next_start = offset + length
                last_flag = bool(flags & extent_last)
            disposition = classify_fiemap_batch(
                mapped_extents=mapped,
                extent_capacity=extent_capacity,
                last_flag=last_flag,
                previous_start=start,
                next_start=next_start,
                coverage_complete=next_start >= logical,
            )
            if disposition == "done":
                break
            assert isinstance(disposition, int)
            start = disposition
    return tuple(ranges)


def _normalize_allocation_ranges(
    ranges: tuple[tuple[int, int], ...], logical: int
) -> tuple[tuple[int, int], ...]:
    cursor = 0
    normalized: list[tuple[int, int]] = []
    for offset, length in ranges:
        start = max(0, offset)
        end = min(logical, offset + length)
        if end <= start:
            continue
        if start > cursor:
            raise ReservationIncompleteError("allocation evidence has a coverage gap")
        if end > cursor:
            normalized.append((start, end - start))
            cursor = end
        if cursor >= logical:
            break
    if cursor != logical:
        raise ReservationIncompleteError("allocation evidence does not cover logical EOF")
    return tuple(normalized)


def platform_allocation_counter(path: Path) -> int:
    """Return the platform's physical allocation counter for release proofs."""

    path = Path(path)
    if os.name == "nt":
        _, allocated = _windows_standard_info(path)
        return allocated
    if sys.platform.startswith("linux"):
        return int(path.stat(follow_symlinks=False).st_blocks) * 512
    raise ReservationCapabilityError("allocation evidence is unsupported on this platform")


def platform_allocation_adapter(
    path: Path,
    logical: int,
) -> ReservationAllocationEvidence:
    """Prove complete physical allocation using the frozen platform adapter."""

    path = Path(path)
    if type(logical) is not int or logical <= 0:
        raise ValueError("logical allocation length must be a positive integer")
    if os.name == "nt":
        eof, allocated = _windows_standard_info(path)
        if eof != logical:
            raise ReservationIncompleteError("Windows EOF differs from logical reservation")
        ranges = _normalize_allocation_ranges(_windows_allocated_ranges(path, logical), logical)
        platform = "windows-allocated-ranges-v1"
    elif sys.platform.startswith("linux"):
        if path.stat(follow_symlinks=False).st_size != logical:
            raise ReservationIncompleteError("Linux EOF differs from logical reservation")
        ranges = _normalize_allocation_ranges(_linux_fiemap_ranges(path, logical), logical)
        allocated = platform_allocation_counter(path)
        platform = "linux-fiemap-v1"
    else:
        raise ReservationCapabilityError("allocation evidence is unsupported on this platform")
    if allocated < logical:
        raise ReservationIncompleteError("physical allocation is smaller than logical coverage")
    normalized = {
        "algorithm_version": platform,
        "logical_coverage_byte_count": logical,
        "allocated_byte_count": allocated,
        "ranges": [[offset, length] for offset, length in ranges],
    }
    return ReservationAllocationEvidence(
        platform=platform,
        logical_coverage_byte_count=logical,
        allocated_byte_count=allocated,
        token=hashlib.sha256(canonical_json_bytes(normalized)).hexdigest(),
    )


def allocation_unit_for_path(path: Path) -> int:
    """Return the volume allocation unit used by reservation accounting."""

    path = Path(path).resolve()
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        sectors = wintypes.DWORD()
        bytes_per_sector = wintypes.DWORD()
        free_clusters = wintypes.DWORD()
        total_clusters = wintypes.DWORD()
        root = path.anchor
        if not ctypes.windll.kernel32.GetDiskFreeSpaceW(
            root,
            ctypes.byref(sectors),
            ctypes.byref(bytes_per_sector),
            ctypes.byref(free_clusters),
            ctypes.byref(total_clusters),
        ):
            raise ReservationCapabilityError("GetDiskFreeSpaceW failed")
        return int(sectors.value * bytes_per_sector.value)
    if sys.platform.startswith("linux"):
        info = os.statvfs(path)
        return int(info.f_frsize or info.f_bsize)
    raise ReservationCapabilityError("allocation-unit query is unsupported")


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
        self.allocation_counter = allocation_counter or platform_allocation_counter
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
        expected_publication_method: PublicationMethod | None = None,
        event: Callable[[str], None] | None = None,
    ) -> ReservedArtifactTransaction:
        started = time.monotonic_ns()
        job_root = Path(job_root)
        index_path = job_root / index_relative_path
        if not index_path.is_file():
            raise ReservationConflictError("reservation extent has no durable write-ahead index")
        target_path = job_root / target_relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_exists = target_path.exists() or target_path.is_symlink()
        publication_method: PublicationMethod = expected_publication_method or (
            "replace_existing" if target_exists else "replace_placeholder"
        )
        if publication_method == "replace_existing":
            if not target_exists:
                raise ReservationConflictError("declared existing reservation target is absent")
            _identity(target_path)
            placeholder_identity = None
        elif publication_method == "replace_placeholder" and target_exists:
            if target_path.stat(follow_symlinks=False).st_size != 0:
                raise ReservationConflictError("declared reservation placeholder is nonempty")
            placeholder_identity = _identity(target_path)
        elif publication_method == "replace_placeholder":
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
        else:
            raise ValueError("unknown expected reservation publication method")
        logical = payload_logical_bytes(payload_raw_max, allocation_unit)
        source_name = f".depth-surge-r-{generation}-{reservation_kind}-{reservation_id}.reserved"
        source_path = target_path.parent / source_name
        metrics = ReservationMetrics()
        try:
            if source_path.exists() or source_path.is_symlink():
                if source_path.is_symlink() or not source_path.is_file():
                    raise ReservationConflictError("reservation source is unsafe")
                current_size = source_path.stat().st_size
                if current_size > logical:
                    raise ReservationConflictError("reservation source exceeds logical length")
                with source_path.open("rb", buffering=0) as existing:
                    while True:
                        chunk = existing.read(RESERVATION_ZERO_FILL_CHUNK_BYTES)
                        if not chunk:
                            break
                        if any(chunk):
                            raise ReservationConflictError(
                                "descriptorless reservation source is not all zero"
                            )
                open_mode = "r+b"
            else:
                current_size = 0
                open_mode = "x+b"
            with source_path.open(open_mode) as handle:
                handle.seek(current_size)
                if event is not None:
                    event("extent:created")
                remaining = logical - current_size
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
        zero_ready = source_path.stat().st_size == logical
        if zero_ready:
            with source_path.open("rb", buffering=0) as handle:
                while True:
                    chunk = handle.read(RESERVATION_ZERO_FILL_CHUNK_BYTES)
                    if not chunk:
                        break
                    if any(chunk):
                        zero_ready = False
                        break
        if not zero_ready:
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

    def _stream_prefix_and_zero_tail(
        self,
        path: Path,
        *,
        prefix_byte_count: int,
        expected_sha256: str,
        expected_total_byte_count: int,
    ) -> None:
        if (
            type(prefix_byte_count) is not int
            or prefix_byte_count <= 0
            or prefix_byte_count > self.descriptor.payload_raw_max
            or prefix_byte_count > expected_total_byte_count
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
        ):
            raise ValueError("streamed reservation identity is invalid")
        digest = hashlib.sha256()
        remaining = prefix_byte_count
        with path.open("rb", buffering=0) as handle:
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ReservationConflictError("streamed reservation prefix ended early")
                digest.update(chunk)
                remaining -= len(chunk)
            tail_remaining = expected_total_byte_count - prefix_byte_count
            while tail_remaining:
                chunk = handle.read(min(1024 * 1024, tail_remaining))
                if not chunk or any(chunk):
                    raise ReservationConflictError("streamed reservation tail is not all zero")
                tail_remaining -= len(chunk)
            if handle.read(1):
                raise ReservationConflictError("streamed reservation has trailing bytes")
        if digest.hexdigest() != expected_sha256:
            raise ReservationConflictError("streamed reservation prefix hash differs")

    def validate_streamed_physical_phase(
        self,
        *,
        prefix_byte_count: int,
        prefix_sha256: str,
    ) -> str:
        """Validate a large canonical prefix without materializing it in memory."""

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
                self._stream_prefix_and_zero_tail(
                    self.source_path,
                    prefix_byte_count=prefix_byte_count,
                    expected_sha256=prefix_sha256,
                    expected_total_byte_count=size,
                )
                return "full-source"
            if size == prefix_byte_count:
                self._stream_prefix_and_zero_tail(
                    self.source_path,
                    prefix_byte_count=prefix_byte_count,
                    expected_sha256=prefix_sha256,
                    expected_total_byte_count=size,
                )
                return "short-source"
            raise ReservationConflictError("streamed reservation source length is invalid")
        if target_exists:
            if (
                _identity(self.target_path) != self.descriptor.filesystem_identity
                or self.target_path.stat().st_size != prefix_byte_count
            ):
                raise ReservationConflictError("streamed reservation target identity differs")
            self._stream_prefix_and_zero_tail(
                self.target_path,
                prefix_byte_count=prefix_byte_count,
                expected_sha256=prefix_sha256,
                expected_total_byte_count=prefix_byte_count,
            )
            return "committed-target"
        raise ReservationConflictError("streamed extent exists at neither authorized location")

    def _short_release_recovery_streamed(
        self,
        *,
        prefix_byte_count: int,
        prefix_sha256: str,
    ) -> None:
        if (
            self.validate_streamed_physical_phase(
                prefix_byte_count=prefix_byte_count,
                prefix_sha256=prefix_sha256,
            )
            != "short-source"
        ):
            raise ReservationConflictError("streamed release recovery requires short source")
        with self.source_path.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        threshold = (
            self.descriptor.allocation_evidence_at_readiness.allocated_byte_count
            - self.descriptor.allocation_unit
        )
        if self.allocation_counter(self.source_path) <= threshold:
            return
        with self.source_path.open("r+b") as handle:
            handle.truncate(prefix_byte_count)
            handle.flush()
            os.fsync(handle.fileno())
        self.validate_streamed_physical_phase(
            prefix_byte_count=prefix_byte_count,
            prefix_sha256=prefix_sha256,
        )
        if self.allocation_counter(self.source_path) > threshold:
            raise ReservationIncompleteError("streamed short-source release remains unproved")

    def _rename_and_sync(self) -> None:
        self._validate_placeholder()
        os.replace(self.source_path, self.target_path)
        _sync_parent(self.target_path.parent)

    def _retire_descriptor(self) -> None:
        self.descriptor_path.unlink(missing_ok=True)
        _sync_parent(self.descriptor_path.parent)

    def consume(self, payload: bytes, *, crash_at: str | None = None) -> str:
        self.stage_complete_full(payload)
        if crash_at == "after-complete-full-sync":
            raise InjectedReservationCrash(crash_at)
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

    def consume_streamed_prefix(
        self,
        *,
        prefix_byte_count: int,
        prefix_sha256: str,
        crash_at: str | None = None,
    ) -> str:
        phase = self.validate_streamed_physical_phase(
            prefix_byte_count=prefix_byte_count,
            prefix_sha256=prefix_sha256,
        )
        if phase != "full-source":
            raise ReservationConflictError("streamed consumption requires complete-full source")
        with self.source_path.open("r+b") as handle:
            handle.truncate(prefix_byte_count)
        if crash_at == "after-truncate":
            raise InjectedReservationCrash(crash_at)
        with self.source_path.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        if crash_at == "after-truncate-sync":
            raise InjectedReservationCrash(crash_at)
        self._short_release_recovery_streamed(
            prefix_byte_count=prefix_byte_count,
            prefix_sha256=prefix_sha256,
        )
        self._rename_and_sync()
        if crash_at == "after-rename-parent-sync":
            raise InjectedReservationCrash(crash_at)
        if (
            self.validate_streamed_physical_phase(
                prefix_byte_count=prefix_byte_count,
                prefix_sha256=prefix_sha256,
            )
            != "committed-target"
        ):
            raise ReservationConflictError("streamed target commit validation failed")
        self._retire_descriptor()
        return "committed-target"

    def recover_streamed_prefix(
        self,
        *,
        prefix_byte_count: int,
        prefix_sha256: str,
    ) -> str:
        phase = self.validate_streamed_physical_phase(
            prefix_byte_count=prefix_byte_count,
            prefix_sha256=prefix_sha256,
        )
        if phase == "full-source":
            with self.source_path.open("r+b") as handle:
                handle.truncate(prefix_byte_count)
            phase = "short-source"
        if phase == "short-source":
            self._short_release_recovery_streamed(
                prefix_byte_count=prefix_byte_count,
                prefix_sha256=prefix_sha256,
            )
            self._rename_and_sync()
            phase = "committed-target"
        if phase == "committed-target":
            self.validate_streamed_physical_phase(
                prefix_byte_count=prefix_byte_count,
                prefix_sha256=prefix_sha256,
            )
            self._retire_descriptor()
            return phase
        raise ReservationConflictError(f"unsupported streamed recovery phase: {phase}")


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
        value = strict_canonical_json_loads(raw, max_bytes=1024 * 1024)
    except (OSError, CanonicalJsonError) as error:
        raise ReservationConflictError("job control is missing or malformed") from error
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


@dataclass(frozen=True)
class SettingsTransitionIntentV1:
    """Durable semantic payload for one schema-5 settings replacement."""

    job_id: str
    settings_identity_fingerprint: str
    reservation_generation: str
    reservation_id: str
    source_settings_revision: int
    target_settings_revision: int
    target_settings_content_fingerprint: str
    target_settings_raw_sha256: str
    target_settings_payload: dict
    fingerprint: str

    def _base_object(self) -> dict:
        return {
            "schema_version": 1,
            "algorithm_version": "settings-transition-reservation-v1",
            "job_id": self.job_id,
            "settings_identity_fingerprint": self.settings_identity_fingerprint,
            "settings_artifact_relative_path": SETTINGS_ARTIFACT_RELATIVE_PATH.as_posix(),
            "reservation_generation": self.reservation_generation,
            "reservation_id": self.reservation_id,
            "reservation_kind": "settings_transition",
            "source_settings_revision": self.source_settings_revision,
            "target_settings_revision": self.target_settings_revision,
            "target_settings_content_fingerprint": self.target_settings_content_fingerprint,
            "target_settings_raw_sha256": self.target_settings_raw_sha256,
            "target_settings_payload": self.target_settings_payload,
        }

    def to_object(self) -> dict:
        return {**self._base_object(), "fingerprint": self.fingerprint}

    @classmethod
    def create(
        cls,
        *,
        current: ProcessingSettingsArtifactV5,
        target: ProcessingSettingsArtifactV5,
        reservation_generation: str,
        reservation_id: str,
    ) -> SettingsTransitionIntentV1:
        _require_lower_hex(reservation_generation, 32, "settings reservation generation")
        _require_lower_hex(reservation_id, 32, "settings reservation id")
        current_value = current.to_object()
        target_value = target.to_object()
        current_metadata = current_value["metadata"]
        target_metadata = target_value["metadata"]
        if target.revision != current.revision + 1:
            raise ReservationConflictError("settings target is not the next revision")
        immutable_identity = (
            "job_id",
            "settings_identity_fingerprint",
            "settings_artifact_relative_path",
            "job_reservation_generation",
            "job_terminal_reservation_id",
            "created_at",
        )
        if any(current_metadata[key] != target_metadata[key] for key in immutable_identity):
            raise ReservationConflictError("settings transition changed immutable identity")
        raw = target.encode()
        provisional = cls(
            job_id=target_metadata["job_id"],
            settings_identity_fingerprint=target_metadata["settings_identity_fingerprint"],
            reservation_generation=reservation_generation,
            reservation_id=reservation_id,
            source_settings_revision=current.revision,
            target_settings_revision=target.revision,
            target_settings_content_fingerprint=target_value["content_fingerprint"],
            target_settings_raw_sha256=hashlib.sha256(raw).hexdigest(),
            target_settings_payload=target_value,
            fingerprint="",
        )
        return cls(
            **{
                **provisional.__dict__,
                "fingerprint": sha256_canonical(provisional._base_object()),
            }
        )

    @classmethod
    def from_object(cls, value: object) -> SettingsTransitionIntentV1:
        from .settings_artifact import ProcessingSettingsArtifactV5

        expected = {
            "schema_version",
            "algorithm_version",
            "job_id",
            "settings_identity_fingerprint",
            "settings_artifact_relative_path",
            "reservation_generation",
            "reservation_id",
            "reservation_kind",
            "source_settings_revision",
            "target_settings_revision",
            "target_settings_content_fingerprint",
            "target_settings_raw_sha256",
            "target_settings_payload",
            "fingerprint",
        }
        if type(value) is not dict or set(value) != expected:
            raise ReservationConflictError("settings transition intent schema is invalid")
        assert isinstance(value, dict)
        if (
            value["schema_version"] != 1
            or type(value["schema_version"]) is not int
            or value["algorithm_version"] != "settings-transition-reservation-v1"
            or value["settings_artifact_relative_path"]
            != SETTINGS_ARTIFACT_RELATIVE_PATH.as_posix()
            or value["reservation_kind"] != "settings_transition"
        ):
            raise ReservationConflictError("settings transition intent version is invalid")
        for key in ("source_settings_revision", "target_settings_revision"):
            if type(value[key]) is not int or not 0 <= value[key] <= 2**64 - 1:
                raise ReservationConflictError(f"{key} is invalid")
        if value["target_settings_revision"] != value["source_settings_revision"] + 1:
            raise ReservationConflictError("settings intent revision equation is invalid")
        target = ProcessingSettingsArtifactV5.from_object(value["target_settings_payload"])
        target_value = target.to_object()
        raw = target.encode()
        result = cls(
            job_id=_require_lower_hex(value["job_id"], 32, "job id"),
            settings_identity_fingerprint=_require_lower_hex(
                value["settings_identity_fingerprint"], 64, "settings identity"
            ),
            reservation_generation=_require_lower_hex(
                value["reservation_generation"], 32, "settings reservation generation"
            ),
            reservation_id=_require_lower_hex(
                value["reservation_id"], 32, "settings reservation id"
            ),
            source_settings_revision=value["source_settings_revision"],
            target_settings_revision=value["target_settings_revision"],
            target_settings_content_fingerprint=_require_lower_hex(
                value["target_settings_content_fingerprint"], 64, "target content fingerprint"
            ),
            target_settings_raw_sha256=_require_lower_hex(
                value["target_settings_raw_sha256"], 64, "target settings raw hash"
            ),
            target_settings_payload=target_value,
            fingerprint=_require_lower_hex(value["fingerprint"], 64, "settings intent fingerprint"),
        )
        if (
            target.revision != result.target_settings_revision
            or target_value["metadata"]["job_id"] != result.job_id
            or target_value["metadata"]["settings_identity_fingerprint"]
            != result.settings_identity_fingerprint
            or target_value["content_fingerprint"] != result.target_settings_content_fingerprint
            or hashlib.sha256(raw).hexdigest() != result.target_settings_raw_sha256
            or sha256_canonical(result._base_object()) != result.fingerprint
        ):
            raise ReservationConflictError("settings transition intent hashes or identity differ")
        return result


class SettingsArtifactTransactionV1:
    """Indexed, reservation-backed replacement of processing-settings-v5.json."""

    def __init__(
        self,
        *,
        job_root: Path,
        intent: SettingsTransitionIntentV1,
        transaction: ReservedArtifactTransaction,
    ) -> None:
        self.job_root = Path(job_root)
        self.intent = intent
        self.transaction = transaction

    @property
    def index_path(self) -> Path:
        return self.job_root / SETTINGS_TRANSITION_INDEX_RELATIVE_PATH

    @classmethod
    def begin(
        cls,
        *,
        job_root: Path,
        current: ProcessingSettingsArtifactV5,
        target: ProcessingSettingsArtifactV5,
        allocation_adapter: AllocationAdapter,
        allocation_unit: int,
        reservation_generation: str,
        reservation_id: str,
        allocation_counter: AllocationCounter | None = None,
    ) -> SettingsArtifactTransactionV1:
        job_root = Path(job_root)
        fixed_path = job_root / SETTINGS_ARTIFACT_RELATIVE_PATH
        if not fixed_path.is_file() or fixed_path.is_symlink():
            raise ReservationConflictError("fixed settings target is missing or unsafe")
        if fixed_path.read_bytes() != current.encode():
            raise ReservationConflictError("current settings bytes changed before transaction")
        intent = SettingsTransitionIntentV1.create(
            current=current,
            target=target,
            reservation_generation=reservation_generation,
            reservation_id=reservation_id,
        )
        index_path = job_root / SETTINGS_TRANSITION_INDEX_RELATIVE_PATH
        publish_write_ahead_intent(index_path, intent.to_object())
        transaction = ReservedArtifactTransaction.create(
            job_root=job_root,
            index_relative_path=SETTINGS_TRANSITION_INDEX_RELATIVE_PATH,
            target_relative_path=SETTINGS_ARTIFACT_RELATIVE_PATH,
            reservation_kind="settings_transition",
            payload_raw_max=512 * 1024,
            allocation_unit=allocation_unit,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
            generation=reservation_generation,
            reservation_id=reservation_id,
        )
        return cls(job_root=job_root, intent=intent, transaction=transaction)

    @staticmethod
    def _retire_index(index_path: Path) -> None:
        index_path.unlink(missing_ok=True)
        temporary = index_path.with_name(f".{index_path.name}.create-new.tmp")
        temporary.unlink(missing_ok=True)
        _sync_parent(index_path.parent)

    def commit(self, *, crash_at: str | None = None) -> ProcessingSettingsArtifactV5:
        from .settings_artifact import ProcessingSettingsArtifactV5

        target = ProcessingSettingsArtifactV5.from_object(self.intent.target_settings_payload)
        nested_crash = (
            crash_at
            if crash_at in {"after-truncate", "after-truncate-sync", "after-rename-parent-sync"}
            else None
        )
        self.transaction.consume(target.encode(), crash_at=nested_crash)
        if crash_at == "after-target-commit":
            raise InjectedReservationCrash(crash_at)
        self._retire_index(self.index_path)
        return target

    @classmethod
    def recover(
        cls,
        job_root: Path,
        *,
        allocation_adapter: AllocationAdapter,
        allocation_counter: AllocationCounter | None = None,
    ) -> ProcessingSettingsArtifactV5:
        from .settings_artifact import ProcessingSettingsArtifactV5

        job_root = Path(job_root)
        index_path = job_root / SETTINGS_TRANSITION_INDEX_RELATIVE_PATH
        try:
            raw = index_path.read_bytes()
            value = strict_canonical_json_loads(raw, max_bytes=1024 * 1024)
        except (OSError, CanonicalJsonError) as error:
            raise ReservationConflictError(
                "settings transition index is missing or malformed"
            ) from error
        intent = SettingsTransitionIntentV1.from_object(value)
        target = ProcessingSettingsArtifactV5.from_object(intent.target_settings_payload)
        descriptor_name = (
            f"d-r-{intent.reservation_generation}-settings_transition-"
            f"{intent.reservation_id}.json"
        )
        descriptor_path = job_root / ".depth-surge-reservations-v1" / descriptor_name
        if descriptor_path.exists():
            transaction = ReservedArtifactTransaction.load(
                descriptor_path,
                allocation_adapter=allocation_adapter,
                allocation_counter=allocation_counter,
            )
            transaction.recover(target.encode())
        else:
            fixed_path = job_root / SETTINGS_ARTIFACT_RELATIVE_PATH
            if not fixed_path.is_file() or fixed_path.read_bytes() != target.encode():
                raise ReservationConflictError(
                    "settings transition has neither descriptor nor committed target"
                )
        cls._retire_index(index_path)
        return target


def _settings_descriptor_path(
    job_root: Path,
    *,
    generation: str,
    reservation_kind: str,
    reservation_id: str,
) -> Path:
    return (
        Path(job_root)
        / ".depth-surge-reservations-v1"
        / f"d-r-{generation}-{reservation_kind}-{reservation_id}.json"
    )


def _bootstrap_settings_intent(
    artifact: ProcessingSettingsArtifactV5,
    *,
    reservation_generation: str,
) -> dict:
    value = artifact.to_object()
    raw = artifact.encode()
    return {
        "algorithm_version": "settings-bootstrap-intent-v1",
        "settings_identity_fingerprint": value["metadata"]["settings_identity_fingerprint"],
        "reservation_generation": reservation_generation,
        "target_settings_revision": 0,
        "target_settings_content_fingerprint": value["content_fingerprint"],
        "target_settings_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "target_settings_payload": value,
    }


def _bootstrap_entry(
    *,
    reservation_kind: str,
    generation: str,
    reservation_id: str,
    publication_method: PublicationMethod,
    allocation_unit: int,
) -> dict:
    return {
        "reservation_kind": reservation_kind,
        "generation": generation,
        "reservation_id": reservation_id,
        "index_relative_path": JOB_CONTROL_RELATIVE_PATH.as_posix(),
        "target_relative_path": SETTINGS_ARTIFACT_RELATIVE_PATH.as_posix(),
        "publication_method": publication_method,
        "payload_raw_max": 512 * 1024,
        "allocation_unit": allocation_unit,
    }


def bootstrap_settings_job_v1(  # noqa: C901
    *,
    job_root: Path,
    artifact: ProcessingSettingsArtifactV5,
    initial_settings_reservation_id: str,
    allocation_adapter: AllocationAdapter,
    allocation_unit: int,
    allocation_counter: AllocationCounter | None = None,
    legacy_settings_artifact_relative_path: str | None = None,
    event: Callable[[str], None] | None = None,
) -> JobControlV1:
    """Publish the immutable locator before reserving initial and terminal settings."""

    from .settings_artifact import ProcessingSettingsArtifactV5

    if not isinstance(artifact, ProcessingSettingsArtifactV5):
        raise TypeError("artifact must be ProcessingSettingsArtifactV5")
    job_root = Path(job_root)
    job_root.mkdir(parents=True, exist_ok=True)
    if (job_root / JOB_CONTROL_RELATIVE_PATH).exists():
        raise ReservationConflictError("job control already exists")
    if (job_root / SETTINGS_ARTIFACT_RELATIVE_PATH).exists():
        raise ReservationConflictError("fixed settings target already exists")
    value = artifact.to_object()
    metadata = value["metadata"]
    if artifact.revision != 0 or artifact.status not in {"in_progress", "completed", "failed"}:
        raise ReservationConflictError("bootstrap settings must be revision zero")
    generation = metadata["job_reservation_generation"]
    terminal_id = metadata["job_terminal_reservation_id"]
    initial_kind = (
        "migration_settings"
        if legacy_settings_artifact_relative_path is not None
        else "initial_settings"
    )
    initial_entry = _bootstrap_entry(
        reservation_kind=initial_kind,
        generation=generation,
        reservation_id=initial_settings_reservation_id,
        publication_method="replace_placeholder",
        allocation_unit=allocation_unit,
    )
    terminal_entry = _bootstrap_entry(
        reservation_kind="job_terminal_settings",
        generation=generation,
        reservation_id=terminal_id,
        publication_method="replace_existing",
        allocation_unit=allocation_unit,
    )
    control = JobControlV1.create(
        job_root=job_root,
        job_id=metadata["job_id"],
        settings_identity_fingerprint=metadata["settings_identity_fingerprint"],
        job_reservation_generation=generation,
        initial_settings_reservation_id=initial_settings_reservation_id,
        job_terminal_reservation_id=terminal_id,
        legacy_settings_artifact_relative_path=legacy_settings_artifact_relative_path,
        bootstrap_extent_entries=[initial_entry, terminal_entry],
        bootstrap_settings_intent=_bootstrap_settings_intent(
            artifact,
            reservation_generation=generation,
        ),
    )
    publish_job_control(job_root, control)
    if event is not None:
        event("job-control:published")
    initial = ReservedArtifactTransaction.create(
        job_root=job_root,
        index_relative_path=JOB_CONTROL_RELATIVE_PATH,
        target_relative_path=SETTINGS_ARTIFACT_RELATIVE_PATH,
        reservation_kind=initial_kind,
        payload_raw_max=512 * 1024,
        allocation_unit=allocation_unit,
        allocation_adapter=allocation_adapter,
        allocation_counter=allocation_counter,
        generation=generation,
        reservation_id=initial_settings_reservation_id,
    )
    if event is not None:
        event("initial-settings:ready")
    terminal = ReservedArtifactTransaction.create(
        job_root=job_root,
        index_relative_path=JOB_CONTROL_RELATIVE_PATH,
        target_relative_path=SETTINGS_ARTIFACT_RELATIVE_PATH,
        reservation_kind="job_terminal_settings",
        payload_raw_max=512 * 1024,
        allocation_unit=allocation_unit,
        allocation_adapter=allocation_adapter,
        allocation_counter=allocation_counter,
        generation=generation,
        reservation_id=terminal_id,
    )
    if event is not None:
        event("terminal-settings:ready")
    initial.consume(artifact.encode())
    if event is not None:
        event("initial-settings:committed")
    if artifact.status == "in_progress":
        if terminal.validate_physical_phase(artifact.encode()) != "full-source":
            raise ReservationConflictError("terminal settings extent is not retained full-length")
        if classify_payload_frame(terminal.source_path.read_bytes(), artifact.encode()) != "zero":
            raise ReservationConflictError("terminal settings extent is not a zero frame")
    else:
        terminal.source_path.unlink()
        _sync_parent(terminal.source_path.parent)
        terminal._retire_descriptor()
    return control


def _parse_bootstrap_target(control: JobControlV1) -> ProcessingSettingsArtifactV5:
    from .settings_artifact import ProcessingSettingsArtifactV5

    intent = control.bootstrap_settings_intent
    expected = {
        "algorithm_version",
        "settings_identity_fingerprint",
        "reservation_generation",
        "target_settings_revision",
        "target_settings_content_fingerprint",
        "target_settings_raw_sha256",
        "target_settings_payload",
    }
    if type(intent) is not dict or set(intent) != expected:
        raise ReservationConflictError("bootstrap settings intent schema is invalid")
    if (
        intent["algorithm_version"] != "settings-bootstrap-intent-v1"
        or intent["settings_identity_fingerprint"] != control.settings_identity_fingerprint
        or intent["reservation_generation"] != control.job_reservation_generation
        or intent["target_settings_revision"] != 0
        or type(intent["target_settings_revision"]) is not int
    ):
        raise ReservationConflictError("bootstrap settings intent identity is invalid")
    artifact = ProcessingSettingsArtifactV5.from_object(intent["target_settings_payload"])
    value = artifact.to_object()
    if (
        artifact.revision != 0
        or value["metadata"]["job_id"] != control.job_id
        or value["metadata"]["settings_identity_fingerprint"]
        != control.settings_identity_fingerprint
        or value["content_fingerprint"] != intent["target_settings_content_fingerprint"]
        or hashlib.sha256(artifact.encode()).hexdigest() != intent["target_settings_raw_sha256"]
    ):
        raise ReservationConflictError("bootstrap target hash or identity differs")
    return artifact


def _terminal_settings_delta_is_legal(
    current: ProcessingSettingsArtifactV5,
    target: ProcessingSettingsArtifactV5,
) -> bool:
    before = current.to_object()
    after = target.to_object()
    if target.revision != current.revision + 1 or target.status not in {"completed", "failed"}:
        return False
    if any(
        before[key] != after[key]
        for key in ("video_properties", "processing_settings", "output_info")
    ):
        return False
    mutable_metadata = {
        "settings_revision",
        "last_updated_at",
        "terminal_at",
        "processing_duration_ms",
        "processing_status",
        "cleanup_status",
    }
    return all(
        before["metadata"][key] == after["metadata"][key]
        for key in before["metadata"]
        if key not in mutable_metadata
    )


def commit_terminal_settings_v1(
    *,
    job_root: Path,
    current: ProcessingSettingsArtifactV5,
    target: ProcessingSettingsArtifactV5,
    allocation_adapter: AllocationAdapter,
    allocation_counter: AllocationCounter | None = None,
    crash_at: str | None = None,
) -> ProcessingSettingsArtifactV5:
    """Consume the lifecycle terminal extent for one completed/failed revision."""

    job_root = Path(job_root)
    control = load_job_control(job_root)
    fixed_path = resolve_fixed_settings_path(job_root, control)
    if fixed_path.read_bytes() != current.encode():
        raise ReservationConflictError("terminal transition current settings changed")
    if not _terminal_settings_delta_is_legal(current, target):
        raise ReservationConflictError("terminal settings delta is not the unique legal revision")
    descriptor_path = _settings_descriptor_path(
        job_root,
        generation=control.job_reservation_generation,
        reservation_kind="job_terminal_settings",
        reservation_id=control.job_terminal_reservation_id,
    )
    terminal = ReservedArtifactTransaction.load(
        descriptor_path,
        allocation_adapter=allocation_adapter,
        allocation_counter=allocation_counter,
    )
    terminal.consume(target.encode(), crash_at=crash_at)
    return target


def _reset_terminal_extent(transaction: ReservedArtifactTransaction) -> None:
    with transaction.source_path.open("r+b") as handle:
        handle.seek(0)
        remaining = transaction.descriptor.logical_byte_count
        zeroes = b"\x00" * RESERVATION_ZERO_FILL_CHUNK_BYTES
        while remaining:
            chunk = zeroes[: min(remaining, len(zeroes))]
            if handle.write(chunk) != len(chunk):
                raise OSError("short terminal reset write")
            remaining -= len(chunk)
        handle.flush()
        os.fsync(handle.fileno())


def _parse_terminal_source_payload(
    raw: bytes,
    *,
    full_length: bool,
) -> ProcessingSettingsArtifactV5 | None:
    from .settings_artifact import ProcessingSettingsArtifactV5, SettingsArtifactError

    if full_length:
        if raw and all(byte == 0 for byte in raw):
            return None
        nul = raw.find(b"\x00")
        if nul <= 0 or any(raw[nul:]):
            raise ReservationIncompleteError("terminal full source has a partial frame")
        raw = raw[:nul]
    try:
        return ProcessingSettingsArtifactV5.parse(raw)
    except SettingsArtifactError as error:
        if full_length:
            raise ReservationIncompleteError("terminal full source has a partial frame") from error
        raise ReservationConflictError(
            "terminal short source is not a canonical artifact"
        ) from error


def recover_settings_job_v1(  # noqa: C901
    job_root: Path,
    *,
    allocation_adapter: AllocationAdapter,
    allocation_counter: AllocationCounter | None = None,
) -> ProcessingSettingsArtifactV5:
    """Recover settings control work before any ordinary artifact audit."""

    from .settings_artifact import ProcessingSettingsArtifactV5

    job_root = Path(job_root)
    control = load_job_control(job_root)
    if (job_root / SETTINGS_TRANSITION_INDEX_RELATIVE_PATH).exists():
        SettingsArtifactTransactionV1.recover(
            job_root,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
        )
    bootstrap_target = _parse_bootstrap_target(control)
    fixed_path = job_root / SETTINGS_ARTIFACT_RELATIVE_PATH
    initial_descriptor = _settings_descriptor_path(
        job_root,
        generation=control.job_reservation_generation,
        reservation_kind=control.bootstrap_extent_entries[0]["reservation_kind"],
        reservation_id=control.initial_settings_reservation_id,
    )
    if initial_descriptor.exists():
        initial = ReservedArtifactTransaction.load(
            initial_descriptor,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
        )
        initial.recover(bootstrap_target.encode())
    if not fixed_path.is_file() or fixed_path.is_symlink():
        raise ReservationConflictError("fixed settings target is missing after bootstrap")
    current = ProcessingSettingsArtifactV5.parse(fixed_path.read_bytes())
    current_value = current.to_object()
    if (
        current_value["metadata"]["job_id"] != control.job_id
        or current_value["metadata"]["settings_identity_fingerprint"]
        != control.settings_identity_fingerprint
    ):
        raise ReservationConflictError("current settings do not belong to job control")
    terminal_descriptor = _settings_descriptor_path(
        job_root,
        generation=control.job_reservation_generation,
        reservation_kind="job_terminal_settings",
        reservation_id=control.job_terminal_reservation_id,
    )
    if terminal_descriptor.is_file():
        terminal = ReservedArtifactTransaction.load(
            terminal_descriptor,
            allocation_adapter=allocation_adapter,
            allocation_counter=allocation_counter,
        )
        if terminal.source_path.exists():
            if _identity(terminal.source_path) != terminal.descriptor.filesystem_identity:
                raise ReservationConflictError("terminal extent identity changed")
            size = terminal.source_path.stat().st_size
            if size == terminal.descriptor.logical_byte_count:
                evidence = allocation_adapter(terminal.source_path, size)
                if evidence != terminal.descriptor.allocation_evidence_at_readiness:
                    raise ReservationEvidenceConflictError("terminal readiness evidence changed")
                try:
                    intended = _parse_terminal_source_payload(
                        terminal.source_path.read_bytes(),
                        full_length=True,
                    )
                except ReservationIncompleteError:
                    _reset_terminal_extent(terminal)
                    intended = None
                if intended is not None:
                    if not _terminal_settings_delta_is_legal(current, intended):
                        raise ReservationConflictError(
                            "terminal extent carries an illegal semantic intent"
                        )
                    terminal.recover(intended.encode())
                    current = intended
            else:
                intended = _parse_terminal_source_payload(
                    terminal.source_path.read_bytes(),
                    full_length=False,
                )
                assert intended is not None
                if not _terminal_settings_delta_is_legal(current, intended):
                    raise ReservationConflictError(
                        "terminal short source carries an illegal semantic intent"
                    )
                terminal.recover(intended.encode())
                current = intended
        elif current.status in {"completed", "failed"}:
            terminal.recover(current.encode())
        else:
            raise ReservationConflictError(
                "in-progress job lost the terminal source before target commit"
            )
    elif current.status == "in_progress":
        raise ReservationConflictError("in-progress job lost its terminal settings reserve")
    return current
