"""Canonical, fail-closed reporting primitives for the Quality Task 0 gate."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform as platform_module
import re
import subprocess
import sys
from typing import Any, Literal


REPORT_ALGORITHM_VERSION = "stereo-quality-task0-report-v1"
REPORT_MAX_RAW_BYTES = 8 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]+")
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}")
_DEVICE_NAME_CHARACTER = re.compile(r"[A-Za-z0-9_.-]")

RunStatus = Literal["passed", "failed"]
GateDecision = Literal["existing_dependencies", "prebuilt_extension"]


class ReportError(RuntimeError):
    """Base error for deterministic Task 0 report handling."""


class ReportSchemaError(ReportError):
    """A report scalar, object, or invariant is outside the closed schema."""


class IncompleteTask0ReportError(ReportSchemaError):
    """One or more required runs are missing, duplicated, or unexpected."""


class StaleTask0ReportError(ReportError):
    """The report was produced from a different source identity."""


class GateSelectionError(ValueError):
    """CLI gate selection is unknown or duplicated."""


def canonical_json_bytes(value: object) -> bytes:
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ReportSchemaError("value cannot be represented as canonical JSON") from error
    if len(raw) > REPORT_MAX_RAW_BYTES:
        raise ReportSchemaError("canonical report exceeds its raw-byte cap")
    return raw


def checked_u64(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 1 << 64:
        raise ReportSchemaError(f"{label} must be an unsigned 64-bit integer")
    return value


def _require_hex(value: object, digits: int, label: str) -> str:
    if not isinstance(value, str) or len(value) != digits or _HEX.fullmatch(value) is None:
        raise ReportSchemaError(f"{label} must be {digits} lowercase hexadecimal digits")
    return value


def _require_string(value: object, label: str, *, maximum: int = 1024) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > maximum:
        raise ReportSchemaError(f"{label} must be a bounded nonempty string")
    return value


def _require_run_id(value: object, label: str = "run_id") -> str:
    if not isinstance(value, str) or _RUN_ID.fullmatch(value) is None:
        raise ReportSchemaError(f"{label} is outside the portable run-id grammar")
    return value


@dataclass(frozen=True)
class SourceIdentity:
    commit: str
    dirty_diff_sha256: str

    def __post_init__(self) -> None:
        _require_hex(self.commit, 40, "source commit")
        _require_hex(self.dirty_diff_sha256, 64, "source diff hash")

    def to_object(self) -> dict:
        return {
            "commit": self.commit,
            "dirty_diff_sha256": self.dirty_diff_sha256,
        }

    @classmethod
    def from_object(cls, value: object) -> SourceIdentity:
        if not isinstance(value, dict) or set(value) != {"commit", "dirty_diff_sha256"}:
            raise ReportSchemaError("source identity schema is invalid")
        return cls(value["commit"], value["dirty_diff_sha256"])


@dataclass(frozen=True)
class EnvironmentIdentity:
    python_version: str
    numpy_version: str
    torch_version: str
    opencv_version: str
    platform: str

    def __post_init__(self) -> None:
        for label, value in (
            ("python_version", self.python_version),
            ("numpy_version", self.numpy_version),
            ("torch_version", self.torch_version),
            ("opencv_version", self.opencv_version),
            ("platform", self.platform),
        ):
            _require_string(value, label, maximum=512)

    def to_object(self) -> dict:
        return {
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "torch_version": self.torch_version,
            "opencv_version": self.opencv_version,
            "platform": self.platform,
        }

    @classmethod
    def from_object(cls, value: object) -> EnvironmentIdentity:
        expected = {
            "python_version",
            "numpy_version",
            "torch_version",
            "opencv_version",
            "platform",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ReportSchemaError("environment identity schema is invalid")
        return cls(**value)


@dataclass(frozen=True)
class GateRunDescriptor:
    run_id: str
    gate: str
    case: str
    device_context: str
    status: RunStatus
    wall_time_ns: int
    peak_rss_bytes: int
    rss_cap_bytes: int
    arena_bytes: int
    visit_total: int
    visit_max: int
    visit_p95: int
    equality_sha256: str
    facts: dict
    fingerprint: str

    def _base_object(self) -> dict:
        return {
            "run_id": self.run_id,
            "gate": self.gate,
            "case": self.case,
            "device_context": self.device_context,
            "status": self.status,
            "wall_time_ns": self.wall_time_ns,
            "peak_rss_bytes": self.peak_rss_bytes,
            "rss_cap_bytes": self.rss_cap_bytes,
            "arena_bytes": self.arena_bytes,
            "visit_total": self.visit_total,
            "visit_max": self.visit_max,
            "visit_p95": self.visit_p95,
            "equality_sha256": self.equality_sha256,
            "facts": self.facts,
        }

    def to_object(self) -> dict:
        return {**self._base_object(), "fingerprint": self.fingerprint}

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        gate: str,
        case: str,
        device_context: str,
        status: RunStatus,
        wall_time_ns: int,
        peak_rss_bytes: int,
        rss_cap_bytes: int,
        arena_bytes: int,
        visit_total: int,
        visit_max: int,
        visit_p95: int,
        equality_sha256: str,
        facts: dict,
    ) -> GateRunDescriptor:
        _require_run_id(run_id)
        _require_run_id(gate, "gate")
        _require_run_id(case, "case")
        _require_string(device_context, "device_context", maximum=256)
        if status not in {"passed", "failed"}:
            raise ReportSchemaError("run status is outside the closed set")
        counters = {
            "wall_time_ns": checked_u64(wall_time_ns, "wall time"),
            "peak_rss_bytes": checked_u64(peak_rss_bytes, "peak RSS"),
            "rss_cap_bytes": checked_u64(rss_cap_bytes, "RSS cap"),
            "arena_bytes": checked_u64(arena_bytes, "arena bytes"),
            "visit_total": checked_u64(visit_total, "visit total"),
            "visit_max": checked_u64(visit_max, "visit max"),
            "visit_p95": checked_u64(visit_p95, "visit p95"),
        }
        if counters["rss_cap_bytes"] == 0:
            raise ReportSchemaError("RSS cap must be positive")
        if status == "passed" and counters["peak_rss_bytes"] > counters["rss_cap_bytes"]:
            raise ReportSchemaError("a run above its RSS cap cannot be labeled passed")
        if counters["visit_max"] > counters["visit_total"]:
            raise ReportSchemaError("visit maximum exceeds the total")
        if counters["visit_p95"] > counters["visit_max"]:
            raise ReportSchemaError("visit p95 exceeds the maximum")
        _require_hex(equality_sha256, 64, "equality hash")
        if not isinstance(facts, dict):
            raise ReportSchemaError("run facts must be an object")
        canonical_json_bytes(facts)
        provisional = cls(
            run_id=run_id,
            gate=gate,
            case=case,
            device_context=device_context,
            status=status,
            equality_sha256=equality_sha256,
            facts=facts,
            fingerprint="",
            **counters,
        )
        return cls(
            **{
                **provisional.__dict__,
                "fingerprint": hashlib.sha256(
                    canonical_json_bytes(provisional._base_object())
                ).hexdigest(),
            }
        )

    @classmethod
    def from_object(cls, value: object) -> GateRunDescriptor:
        expected = {
            "run_id",
            "gate",
            "case",
            "device_context",
            "status",
            "wall_time_ns",
            "peak_rss_bytes",
            "rss_cap_bytes",
            "arena_bytes",
            "visit_total",
            "visit_max",
            "visit_p95",
            "equality_sha256",
            "facts",
            "fingerprint",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ReportSchemaError("gate run descriptor schema is invalid")
        fingerprint = _require_hex(value["fingerprint"], 64, "run fingerprint")
        run = cls.create(**{key: value[key] for key in expected - {"fingerprint"}})
        if run.fingerprint != fingerprint:
            raise ReportSchemaError("gate run descriptor fingerprint is invalid")
        return run


@dataclass(frozen=True)
class Task0Report:
    source: SourceIdentity
    environment: EnvironmentIdentity
    required_run_ids: tuple[str, ...]
    runs: tuple[GateRunDescriptor, ...]
    gate_decision: GateDecision
    fingerprint: str

    def _base_object(self) -> dict:
        return {
            "schema_version": 1,
            "algorithm_version": REPORT_ALGORITHM_VERSION,
            "source": self.source.to_object(),
            "environment": self.environment.to_object(),
            "required_run_ids": list(self.required_run_ids),
            "runs": [run.to_object() for run in self.runs],
            "gate_decision": self.gate_decision,
        }

    def to_object(self) -> dict:
        return {**self._base_object(), "fingerprint": self.fingerprint}

    @classmethod
    def create(
        cls,
        *,
        source: SourceIdentity,
        environment: EnvironmentIdentity,
        required_run_ids: tuple[str, ...],
        runs: tuple[GateRunDescriptor, ...],
        gate_decision: GateDecision,
    ) -> Task0Report:
        if not required_run_ids:
            raise IncompleteTask0ReportError("required run set is empty")
        required = tuple(_require_run_id(item, "required run id") for item in required_run_ids)
        if len(set(required)) != len(required):
            raise IncompleteTask0ReportError("required run ids contain a duplicate")
        observed = [run.run_id for run in runs]
        if len(set(observed)) != len(observed):
            raise IncompleteTask0ReportError("run descriptors contain a duplicate")
        if set(observed) != set(required):
            raise IncompleteTask0ReportError("run descriptors do not equal the required set")
        if gate_decision not in {"existing_dependencies", "prebuilt_extension"}:
            raise ReportSchemaError("gate decision is outside the closed set")
        ordered_runs = tuple(sorted(runs, key=lambda run: run.run_id))
        if gate_decision == "existing_dependencies" and any(
            run.status != "passed" for run in ordered_runs
        ):
            raise IncompleteTask0ReportError("existing-dependencies decision has failed runs")
        provisional = cls(source, environment, required, ordered_runs, gate_decision, "")
        return cls(
            source,
            environment,
            required,
            ordered_runs,
            gate_decision,
            hashlib.sha256(canonical_json_bytes(provisional._base_object())).hexdigest(),
        )

    @classmethod
    def from_object(cls, value: object) -> Task0Report:
        expected = {
            "schema_version",
            "algorithm_version",
            "source",
            "environment",
            "required_run_ids",
            "runs",
            "gate_decision",
            "fingerprint",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ReportSchemaError("Task 0 report schema is invalid")
        if value["schema_version"] != 1 or value["algorithm_version"] != REPORT_ALGORITHM_VERSION:
            raise ReportSchemaError("Task 0 report version is invalid")
        if not isinstance(value["required_run_ids"], list) or not isinstance(value["runs"], list):
            raise ReportSchemaError("Task 0 report run collections are invalid")
        fingerprint = _require_hex(value["fingerprint"], 64, "report fingerprint")
        report = cls.create(
            source=SourceIdentity.from_object(value["source"]),
            environment=EnvironmentIdentity.from_object(value["environment"]),
            required_run_ids=tuple(value["required_run_ids"]),
            runs=tuple(GateRunDescriptor.from_object(run) for run in value["runs"]),
            gate_decision=value["gate_decision"],
        )
        if report.fingerprint != fingerprint:
            raise ReportSchemaError("Task 0 report fingerprint is invalid")
        return report


def _strict_json(raw: bytes) -> object:
    if len(raw) > REPORT_MAX_RAW_BYTES:
        raise ReportSchemaError("report exceeds its raw-byte cap")

    def unique_object(pairs: list[tuple[str, object]]) -> dict:
        result: dict = {}
        for key, value in pairs:
            if key in result:
                raise ReportSchemaError(f"duplicate report key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("ascii"), object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReportSchemaError("report JSON is malformed") from error
    if canonical_json_bytes(value) != raw:
        raise ReportSchemaError("report JSON is not canonical")
    return value


def publish_report(path: Path, report: Task0Report) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    raw = canonical_json_bytes(report.to_object())
    try:
        with temporary.open("w+b") as handle:
            if handle.write(raw) != len(raw):
                raise OSError("short Task 0 report write")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        if path.read_bytes() != raw:
            raise ReportSchemaError("published Task 0 report failed reopen validation")
    finally:
        temporary.unlink(missing_ok=True)


def load_report(
    path: Path,
    *,
    expected_source: SourceIdentity | None = None,
) -> Task0Report:
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise ReportSchemaError("Task 0 report is missing") from error
    report = Task0Report.from_object(_strict_json(raw))
    if expected_source is not None and report.source != expected_source:
        raise StaleTask0ReportError("Task 0 report source identity differs from this checkout")
    return report


def select_gate_names(requested: list[str], available: tuple[str, ...]) -> tuple[str, ...]:
    if len(set(available)) != len(available) or not available:
        raise GateSelectionError("available gate registry is invalid")
    if not requested:
        return available
    selected: list[str] = []
    for name in requested:
        if name not in available:
            raise GateSelectionError(f"unknown gate name: {name}")
        if name in selected:
            raise GateSelectionError(f"duplicate gate name: {name}")
        selected.append(name)
    return tuple(selected)


def device_context_label(
    kind: Literal["cpu", "cuda"],
    *,
    index: int | None = None,
    name: str | None = None,
) -> str:
    if kind == "cpu":
        if index is not None or name is not None:
            raise ReportSchemaError("CPU context cannot carry CUDA fields")
        return "cpu"
    if kind != "cuda" or isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ReportSchemaError("CUDA context requires a nonnegative device index")
    if not isinstance(name, str) or not name:
        raise ReportSchemaError("CUDA context requires a device name")
    normalized = "".join(
        character if _DEVICE_NAME_CHARACTER.fullmatch(character) else "_" for character in name
    )
    if not normalized or len(normalized) > 128:
        raise ReportSchemaError("CUDA device name is outside its bound")
    return f"cuda:{index}:{normalized}"


def read_process_rss_bytes(process_id: int) -> int:
    if isinstance(process_id, bool) or not isinstance(process_id, int) or process_id <= 0:
        raise ReportSchemaError("process id must be positive")
    if os.name == "nt":
        return _read_windows_process_rss(process_id)
    status = Path(f"/proc/{process_id}/status")
    if status.is_file():
        for line in status.read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                fields = line.split()
                if len(fields) == 3 and fields[2] == "kB":
                    return checked_u64(int(fields[1]) * 1024, "process RSS")
        raise ReportSchemaError("process status does not expose VmRSS")
    if process_id != os.getpid():
        raise ReportSchemaError("this platform cannot sample another process RSS")
    import resource  # type: ignore[import-not-found]

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # type: ignore[attr-defined]
    multiplier = 1 if sys.platform == "darwin" else 1024
    return checked_u64(int(value) * multiplier, "process RSS")


def _read_windows_process_rss(process_id: int) -> int:
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    query_information = 0x0400
    query_limited_information = 0x1000
    virtual_memory_read = 0x0010
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    psapi.GetProcessMemoryInfo.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCountersEx),
        wintypes.DWORD,
    )
    handle = kernel32.OpenProcess(
        query_information | query_limited_information | virtual_memory_read,
        False,
        process_id,
    )
    if not handle:
        raise OSError(ctypes.get_last_error(), "OpenProcess failed")
    try:
        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return checked_u64(int(counters.WorkingSetSize), "process RSS")
    finally:
        kernel32.CloseHandle(handle)


def capture_environment_identity() -> EnvironmentIdentity:
    def version(distribution: str) -> str:
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            return "not-installed"

    return EnvironmentIdentity(
        python_version=platform_module.python_version(),
        numpy_version=version("numpy"),
        torch_version=version("torch"),
        opencv_version=version("opencv-python"),
        platform=platform_module.platform(),
    )


def _hash_git_stdout(command: list[str], *, cwd: Path, digest: Any) -> None:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    stream = process.stdout
    assert stream is not None
    for chunk in iter(lambda: stream.read(64 * 1024), b""):
        digest.update(chunk)
    if process.wait() != 0:
        raise ReportSchemaError(f"source identity command failed: {command[1]}")


def capture_source_identity(  # noqa: C901
    repository: Path,
    *,
    excluded_relative_paths: tuple[str, ...] = (),
) -> SourceIdentity:
    repository = Path(repository).resolve()
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReportSchemaError("cannot resolve the source commit") from error
    try:
        commit = completed.stdout.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise ReportSchemaError("source commit is not ASCII") from error
    _require_hex(commit, 40, "source commit")
    normalized_exclusions: set[str] = set()
    for value in excluded_relative_paths:
        normalized = Path(value).as_posix()
        if normalized.startswith("../") or normalized in {"", ".", ".."}:
            raise ReportSchemaError("source identity exclusion is not contained")
        normalized_exclusions.add(normalized)

    digest = hashlib.sha256()
    digest.update(b"task0-source-diff-v1\x00tracked\x00")
    diff_command = ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--", "."]
    diff_command.extend(f":(exclude){path}" for path in sorted(normalized_exclusions))
    _hash_git_stdout(diff_command, cwd=repository, digest=digest)
    digest.update(b"\x00untracked\x00")
    try:
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReportSchemaError("cannot enumerate untracked source files") from error
    if len(untracked) > REPORT_MAX_RAW_BYTES:
        raise ReportSchemaError("untracked source-name list exceeds its bound")
    for raw_name in untracked.split(b"\x00"):
        if not raw_name:
            continue
        name = raw_name.decode("utf-8", errors="surrogateescape")
        normalized = Path(name).as_posix()
        if normalized in normalized_exclusions:
            continue
        path = repository / Path(name)
        if path.is_symlink() or not path.is_file():
            raise ReportSchemaError(f"untracked source entry is not a regular file: {name}")
        digest.update(len(raw_name).to_bytes(8, "little"))
        digest.update(raw_name)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(64 * 1024), b""):
                digest.update(chunk)
    return SourceIdentity(commit=commit, dirty_diff_sha256=digest.hexdigest())
