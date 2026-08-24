"""Authenticated diagnostics pruning after final-media publication."""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Protocol

from ...io.file_identity import (
    DirectoryIdentityV1 as _DirectoryIdentity,
    PersistentIdentityCapabilityError,
    PersistentIdentitySchemaError,
    capture_directory_identity,
    directory_identity_from_posix_fd,
    file_identity_from_posix_fd,
    windows_identity_from_handle,
)
from .diagnostic_transaction import (
    _parse_canonical_object,
    canonical_json_bytes,
    self_fingerprint,
)


class CleanupIdentityMismatchError(RuntimeError):
    """A persisted prune root no longer names its authenticated directory."""


class CleanupTransactionError(RuntimeError):
    """A prune marker, metadata transition, or authorized deletion failed."""


class CleanupCapabilityError(CleanupTransactionError):
    """The platform cannot prove a destructive-cleanup safety invariant."""


FaultHook = Callable[[str], None]


class FinalCleanupReservationV1(Protocol):
    index: Any

    def commit_prune_markers(self) -> tuple[dict, ...]: ...

    def commit_payload_pruned_metadata(self, payload: bytes) -> None: ...


_PRUNE_ROOT_ORDER = (
    ("frames", "00_original_frames"),
    ("scene_data", "01_scene_data"),
    ("depth_raw", "02_depth_raw"),
    ("disparity_maps", "03_disparity_maps"),
    ("metric_geometry", "03_metric_geometry"),
    ("disparity_stabilized", "03_disparity_stabilized"),
    ("left_frames", "04_left_frames"),
    ("right_frames", "04_right_frames"),
    ("left_distorted", "05_left_distorted"),
    ("right_distorted", "05_right_distorted"),
    ("left_cropped", "06_left_cropped"),
    ("right_cropped", "06_right_cropped"),
    ("left_upscaled", "07_left_upscaled"),
    ("right_upscaled", "07_right_upscaled"),
    ("vr_frames", "99_vr_frames"),
    ("stereo_diagnostics_frames", "04_stereo_diagnostics/frames"),
)
_PRUNE_ROOT_INDEX = {entry: index for index, entry in enumerate(_PRUNE_ROOT_ORDER)}
_PAYLOAD_PRUNED_KEYS = {
    "schema_version",
    "algorithm_version",
    "status",
    "rgb_stage_fingerprint",
    "source_guide_fingerprint",
    "native_geometry_fingerprint",
    "quality_input_manifest_sha256",
    "frame_names",
    "render_shape",
    "render_mode",
    "geometry_mode",
    "occlusion_fill",
    "mask_payloads_enabled",
    "stats_schema_version",
    "frame_manifest_schema_version",
    "ordered_frame_manifest_fingerprint",
    "frames_jsonl_sha256",
    "summary_sha256",
    "metric_clamp_summary_sha256",
    "pruned_from",
    "final_video_manifest_sha256",
    "prune_entries",
    "fingerprint",
}
_PRUNED_FROM_KEYS = {
    "previous_status",
    "previous_diagnostics_fingerprint",
    "ordered_frame_manifest_fingerprint",
    "frames_jsonl_sha256",
    "summary_sha256",
    "metric_clamp_summary_sha256",
}
_PRUNE_ENTRY_KEYS = {
    "stage_key",
    "relative_path",
    "operation",
    "cleanup_contract_version",
    "directory_identity",
    "marker",
}
_HEX32 = re.compile(r"[0-9a-f]{32}", flags=re.ASCII)
_HEX64 = re.compile(r"[0-9a-f]{64}", flags=re.ASCII)


@dataclass(frozen=True)
class _InvocationMountIdentity:
    algorithm: str
    value: int


def _linux_mount_identity(descriptor: int) -> _InvocationMountIdentity:
    if not sys.platform.startswith("linux"):
        raise CleanupCapabilityError("statx mount identity is unavailable")
    import ctypes

    class StatxTimestamp(ctypes.Structure):
        _fields_ = [
            ("seconds", ctypes.c_int64),
            ("nanoseconds", ctypes.c_uint32),
            ("reserved", ctypes.c_int32),
        ]

    class Statx(ctypes.Structure):
        _fields_ = [
            ("mask", ctypes.c_uint32),
            ("block_size", ctypes.c_uint32),
            ("attributes", ctypes.c_uint64),
            ("link_count", ctypes.c_uint32),
            ("uid", ctypes.c_uint32),
            ("gid", ctypes.c_uint32),
            ("mode", ctypes.c_uint16),
            ("spare0", ctypes.c_uint16),
            ("inode", ctypes.c_uint64),
            ("size", ctypes.c_uint64),
            ("blocks", ctypes.c_uint64),
            ("attributes_mask", ctypes.c_uint64),
            ("access_time", StatxTimestamp),
            ("birth_time", StatxTimestamp),
            ("change_time", StatxTimestamp),
            ("modify_time", StatxTimestamp),
            ("rdev_major", ctypes.c_uint32),
            ("rdev_minor", ctypes.c_uint32),
            ("dev_major", ctypes.c_uint32),
            ("dev_minor", ctypes.c_uint32),
            ("mount_id", ctypes.c_uint64),
            ("spare", ctypes.c_uint64 * 13),
        ]

    libc = ctypes.CDLL(None, use_errno=True)
    statx = getattr(libc, "statx", None)
    if statx is None:
        raise CleanupCapabilityError("libc does not expose statx")
    statx.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(Statx),
    )
    statx.restype = ctypes.c_int
    result = Statx()
    statx_mnt_id = 0x00001000
    statx_mnt_id_unique = 0x00004000
    if (
        statx(
            descriptor,
            b"",
            0x1000 | 0x100,
            statx_mnt_id | statx_mnt_id_unique,
            ctypes.byref(result),
        )
        != 0
    ):
        code = ctypes.get_errno()
        raise CleanupCapabilityError(f"statx mount identity failed: errno {code}")
    if result.mask & statx_mnt_id_unique:
        algorithm = "statx-mnt-id-unique"
    elif result.mask & statx_mnt_id:
        algorithm = "statx-mnt-id"
    else:
        raise CleanupCapabilityError("statx returned no mount identity")
    return _InvocationMountIdentity(algorithm, int(result.mount_id))


def _require_same_invocation_mount(
    descriptor: int,
    expected: _InvocationMountIdentity,
    path: object,
) -> None:
    if _linux_mount_identity(descriptor) != expected:
        raise CleanupIdentityMismatchError(f"cleanup crossed a mount at {path}")


def _path_mount_identity(path: Path) -> _InvocationMountIdentity:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        return _linux_mount_identity(descriptor)
    finally:
        os.close(descriptor)


def _requires_posix_mount_identity() -> bool:
    return os.name != "nt"


@dataclass(frozen=True)
class PlannedPruneMarkerV1:
    stage_key: str
    relative_path: str
    directory_identity: dict[str, int]
    marker_name: str
    payload: bytes

    @property
    def payload_sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    def prune_entry(self) -> dict[str, Any]:
        return {
            "stage_key": self.stage_key,
            "relative_path": self.relative_path,
            "operation": "delete_tree",
            "cleanup_contract_version": 2,
            "directory_identity": dict(self.directory_identity),
            "marker": {
                "name": self.marker_name,
                "sha256": self.payload_sha256,
            },
        }


def _emit(fault: FaultHook | None, stage: str) -> None:
    if fault is not None:
        fault(stage)


def _sync_parent(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _directory_identity(path: Path) -> _DirectoryIdentity:
    try:
        return capture_directory_identity(Path(path))
    except PersistentIdentityCapabilityError as error:
        raise CleanupCapabilityError(str(error)) from error
    except (OSError, PersistentIdentitySchemaError) as error:
        raise CleanupIdentityMismatchError(
            f"prune root is not an ordinary directory: {path}"
        ) from error


def _safe_relative_path(job_root: Path, path: Path) -> str:
    relative = path.resolve(strict=True).relative_to(job_root.resolve(strict=True))
    token = PurePosixPath(relative.as_posix())
    if (
        token.is_absolute()
        or not token.parts
        or any(part in {"", ".", ".."} for part in token.parts)
    ):
        raise CleanupTransactionError("prune path is not canonical and contained")
    return token.as_posix()


def discover_prune_roots(job_root: Path) -> tuple[tuple[str, Path], ...]:
    """Freeze concrete normal-run roots; recovery uses only persisted paths."""

    root = Path(job_root).resolve()
    candidates: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for stage_key, relative in _PRUNE_ROOT_ORDER:
        if relative in seen:
            continue
        seen.add(relative)
        candidate = root / relative
        if candidate.is_dir() and not candidate.is_symlink():
            candidates.append((stage_key, candidate))
    return tuple(candidates)


def plan_prune_markers_v1(job_root: Path) -> tuple[PlannedPruneMarkerV1, ...]:
    """Freeze target-local marker names, payloads, and root identities read-only."""

    root = Path(job_root).resolve()
    root_identity = _directory_identity(root)
    root_mount = _path_mount_identity(root) if _requires_posix_mount_identity() else None
    planned: list[PlannedPruneMarkerV1] = []
    for stage_key, prune_root in discover_prune_roots(root):
        identity = _directory_identity(prune_root)
        if identity.volume_key != root_identity.volume_key:
            raise CleanupIdentityMismatchError(f"prune root crosses the job volume: {prune_root}")
        if root_mount is not None and _path_mount_identity(prune_root) != root_mount:
            raise CleanupIdentityMismatchError(f"prune root crosses a mount: {prune_root}")
        relative_path = _safe_relative_path(root, prune_root)
        token = secrets.token_hex(16)
        marker_name = f".depth-surge-prune-v2-{token}.marker"
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
        planned.append(
            PlannedPruneMarkerV1(
                stage_key=stage_key,
                relative_path=relative_path,
                directory_identity=identity.to_object(),
                marker_name=marker_name,
                payload=payload,
            )
        )
    return tuple(planned)


def prune_entry_from_reservation_v1(
    *,
    relative_path: str,
    directory_identity: object,
    marker_name: str,
    marker_payload_sha256: str,
) -> dict[str, Any]:
    """Rebuild the fixed cleanup authorization carried by a marker extent."""

    matches = [stage_key for stage_key, relative in _PRUNE_ROOT_ORDER if relative == relative_path]
    if len(matches) != 1:
        raise CleanupTransactionError("reserved prune root is outside cleanup contract v2")
    entry = {
        "stage_key": matches[0],
        "relative_path": relative_path,
        "operation": "delete_tree",
        "cleanup_contract_version": 2,
        "directory_identity": _DirectoryIdentity.from_object(directory_identity).to_object(),
        "marker": {
            "name": marker_name,
            "sha256": marker_payload_sha256,
        },
    }
    _validate_prune_entry(entry, previous_index=-1)
    return entry


def _is_hex64(value: object) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    workspace = bytearray(1024 * 1024)
    view = memoryview(workspace)
    with path.open("rb", buffering=0) as handle:
        while True:
            count = handle.readinto(workspace)
            if not count:
                break
            digest.update(view[:count])
    return digest.hexdigest()


def _validate_prune_entry(entry: object, *, previous_index: int) -> int:
    if not isinstance(entry, dict) or set(entry) != _PRUNE_ENTRY_KEYS:
        raise CleanupTransactionError("cleanup authorization entry schema is invalid")
    if entry["operation"] != "delete_tree" or entry["cleanup_contract_version"] != 2:
        raise CleanupTransactionError("cleanup authorization operation is invalid")
    stage_key = entry["stage_key"]
    relative_path = entry["relative_path"]
    if not isinstance(stage_key, str) or not isinstance(relative_path, str):
        raise CleanupTransactionError("cleanup authorization path is invalid")
    relative = PurePosixPath(relative_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or relative.as_posix() != relative_path
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise CleanupTransactionError("cleanup authorization path is invalid")
    index = _PRUNE_ROOT_INDEX.get((stage_key, relative_path))
    if index is None or index <= previous_index:
        raise CleanupTransactionError("cleanup authorization roots are not unique pipeline order")
    _DirectoryIdentity.from_object(entry["directory_identity"])
    marker = entry["marker"]
    if not isinstance(marker, dict) or set(marker) != {"name", "sha256"}:
        raise CleanupTransactionError("cleanup authorization marker schema is invalid")
    name = marker["name"]
    if (
        not isinstance(name, str)
        or not name.startswith(".depth-surge-prune-v2-")
        or not name.endswith(".marker")
        or _HEX32.fullmatch(name[len(".depth-surge-prune-v2-") : -len(".marker")]) is None
        or not _is_hex64(marker["sha256"])
    ):
        raise CleanupTransactionError("cleanup authorization marker identity is invalid")
    return index


def _validate_payload_pruned_authorization(job_root: Path, value: object) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != _PAYLOAD_PRUNED_KEYS
        or value.get("schema_version") != 1
        or value.get("algorithm_version") != "stereo-coverage-sidecar-v1"
        or value.get("status") != "payload_pruned"
        or value.get("mask_payloads_enabled") is not False
        or value.get("ordered_frame_manifest_fingerprint") is not None
        or value.get("metric_clamp_summary_sha256") is not None
        or value.get("fingerprint") != self_fingerprint(value)
        or not _is_hex64(value.get("frames_jsonl_sha256"))
        or not _is_hex64(value.get("summary_sha256"))
        or not _is_hex64(value.get("final_video_manifest_sha256"))
        or value.get("render_mode") not in {"fast", "quality"}
        or value.get("geometry_mode") not in {"relative", "metric"}
    ):
        raise CleanupTransactionError("payload-pruned cleanup authorization is invalid")
    frame_names = value.get("frame_names")
    render_shape = value.get("render_shape")
    if (
        not isinstance(frame_names, list)
        or not frame_names
        or len(frame_names) != len(set(frame_names))
        or any(
            not isinstance(name, str) or not name or PurePosixPath(name).name != name
            for name in frame_names
        )
        or not isinstance(render_shape, list)
        or len(render_shape) != 2
        or any(type(dimension) is not int or dimension <= 0 for dimension in render_shape)
    ):
        raise CleanupTransactionError("payload-pruned cleanup sequence identity is invalid")
    pruned_from = value.get("pruned_from")
    if (
        not isinstance(pruned_from, dict)
        or set(pruned_from) != _PRUNED_FROM_KEYS
        or pruned_from.get("previous_status") not in {"complete", "legacy_fast_unavailable"}
        or not _is_hex64(pruned_from.get("previous_diagnostics_fingerprint"))
        or not _is_hex64(pruned_from.get("ordered_frame_manifest_fingerprint"))
        or pruned_from.get("frames_jsonl_sha256") != value["frames_jsonl_sha256"]
        or pruned_from.get("summary_sha256") != value["summary_sha256"]
        or (
            pruned_from.get("metric_clamp_summary_sha256") is not None
            and not _is_hex64(pruned_from.get("metric_clamp_summary_sha256"))
        )
    ):
        raise CleanupTransactionError("payload-pruned cleanup history is invalid")
    entries = value.get("prune_entries")
    if not isinstance(entries, list):
        raise CleanupTransactionError("payload-pruned cleanup entry list is invalid")
    previous_index = -1
    for entry in entries:
        previous_index = _validate_prune_entry(entry, previous_index=previous_index)

    root = Path(job_root)
    diagnostics = root / "04_stereo_diagnostics"
    retained = (
        (diagnostics / "stereo_coverage_frames.jsonl", value["frames_jsonl_sha256"]),
        (diagnostics / "stereo_coverage_summary.json", value["summary_sha256"]),
        (root / "final_video_manifest.json", value["final_video_manifest_sha256"]),
    )
    content_hash = value.get("quality_input_manifest_sha256")
    if value["render_mode"] == "quality":
        if not _is_hex64(content_hash):
            raise CleanupTransactionError("Quality cleanup content identity is invalid")
        retained += ((diagnostics / "quality_input_manifest.json", content_hash),)
    elif content_hash is not None:
        raise CleanupTransactionError("Fast cleanup cannot bind a Quality content manifest")
    try:
        if any(_hash_file(path) != expected for path, expected in retained):
            raise CleanupTransactionError("retained cleanup evidence hash differs")
    except OSError as error:
        raise CleanupTransactionError("retained cleanup evidence is unavailable") from error
    return value


def _publish_marker(root: Path, relative_path: str) -> dict[str, str]:
    token = secrets.token_hex(16)
    name = f".depth-surge-prune-v2-{token}.marker"
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
    marker = root / name
    with marker.open("xb", buffering=0) as handle:
        if handle.write(payload) != len(payload):
            raise OSError("short prune-marker write")
        handle.flush()
        os.fsync(handle.fileno())
    _sync_parent(root)
    return {"name": name, "sha256": hashlib.sha256(payload).hexdigest()}


def _replace_metadata(path: Path, raw: bytes) -> None:
    from ...io.reservation_protocol import (
        _directory_identity as reservation_directory_identity,
        _identity as reservation_file_identity,
        same_parent_replace_v1,
    )

    temporary = path.with_name(f".{path.name}.payload-pruned.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("xb", buffering=0) as handle:
            if handle.write(raw) != len(raw):
                raise OSError("short payload-pruned metadata write")
            handle.flush()
            os.fsync(handle.fileno())
        source_identity = reservation_file_identity(temporary)
        target_identity = reservation_file_identity(path)
        parent_identity = reservation_directory_identity(path.parent)
        same_parent_replace_v1(
            temporary,
            path,
            expected_source_identity=source_identity,
            expected_target_identity=target_identity,
            expected_parent_identity=parent_identity,
            replace_existing=True,
        )
    finally:
        temporary.unlink(missing_ok=True)


def commit_payload_pruned(  # noqa: C901
    job_root: Path,
    *,
    reservation: FinalCleanupReservationV1 | None = None,
    fault: FaultHook | None = None,
) -> dict[str, Any]:
    """Publish markers and the sole payload-pruned diagnostics authorization."""

    root = Path(job_root).resolve()
    active_index = root / ".depth-surge-reservations-v1" / "final-encoding-reservation-v1.json"
    index_temporary = active_index.with_name(f".{active_index.name}.create-new.tmp")
    if reservation is None:
        from ...io.operations import recover_processing_control_prefix_v1
        from ...processing.video.final_publication import (
            load_final_encoding_reservation_v1,
        )

        settings = recover_processing_control_prefix_v1(root)
        active_pair = any(
            path.exists() or path.is_symlink() for path in (active_index, index_temporary)
        )
        if active_pair:
            if settings is None:
                raise CleanupTransactionError("schema-5 cleanup has no recovered job control")
            reservation = load_final_encoding_reservation_v1(root, settings)
    metadata_path = root / "04_stereo_diagnostics" / "metadata.json"
    final_manifest_path = root / "final_video_manifest.json"
    metadata_raw = metadata_path.read_bytes()
    if len(metadata_raw) > 16 * 1024 * 1024:
        raise CleanupTransactionError("diagnostics metadata exceeds its bound")
    metadata = _parse_canonical_object(metadata_raw)
    if isinstance(metadata, dict) and metadata.get("status") == "payload_pruned":
        validated = _validate_payload_pruned_authorization(root, metadata)
        if reservation is not None:
            reservation.commit_payload_pruned_metadata(metadata_raw)
        return validated
    if not isinstance(metadata, dict) or metadata.get("status") not in {
        "complete",
        "legacy_fast_unavailable",
    }:
        raise CleanupTransactionError("diagnostics are not complete before pruning")
    if metadata.get("fingerprint") != self_fingerprint(metadata):
        raise CleanupTransactionError("diagnostics metadata fingerprint is invalid")
    if reservation is not None:
        index = reservation.index
        if (
            index.diagnostics_metadata_raw_sha256_at_preflight
            != hashlib.sha256(metadata_raw).hexdigest()
            or index.diagnostics_metadata_fingerprint_at_preflight != metadata["fingerprint"]
        ):
            raise CleanupTransactionError(
                "diagnostics metadata differs from final reservation preflight"
            )
    diagnostics_root = metadata_path.parent
    frames_jsonl_raw = (diagnostics_root / "stereo_coverage_frames.jsonl").read_bytes()
    if hashlib.sha256(frames_jsonl_raw).hexdigest() != metadata.get("frames_jsonl_sha256"):
        raise CleanupTransactionError("retained diagnostics JSONL hash differs")
    summary_raw = (diagnostics_root / "stereo_coverage_summary.json").read_bytes()
    try:
        _parse_canonical_object(summary_raw)
    except ValueError as error:
        raise CleanupTransactionError("retained diagnostics summary is invalid") from error
    if hashlib.sha256(summary_raw).hexdigest() != metadata.get("summary_sha256"):
        raise CleanupTransactionError("retained diagnostics summary hash differs")
    clamp_sha = metadata.get("metric_clamp_summary_sha256")
    if clamp_sha is not None:
        clamp_raw = (root / "04_left_frames" / "clamp_summary.json").read_bytes()
        if hashlib.sha256(clamp_raw).hexdigest() != clamp_sha:
            raise CleanupTransactionError("metric clamp summary hash differs")
    content_sha = metadata.get("quality_input_manifest_sha256")
    if content_sha is not None:
        content_raw = (diagnostics_root / "quality_input_manifest.json").read_bytes()
        if hashlib.sha256(content_raw).hexdigest() != content_sha:
            raise CleanupTransactionError("retained Quality input manifest hash differs")
    final_manifest_raw = final_manifest_path.read_bytes()
    if len(final_manifest_raw) > 1024 * 1024:
        raise CleanupTransactionError("final-video manifest exceeds its bound")
    _parse_canonical_object(final_manifest_raw)

    if reservation is not None:
        entries = list(reservation.commit_prune_markers())
    else:
        root_identity = _directory_identity(root)
        entries = []
        for stage_key, prune_root in discover_prune_roots(root):
            identity = _directory_identity(prune_root)
            if identity.volume_key != root_identity.volume_key:
                raise CleanupIdentityMismatchError(
                    f"prune root crosses the job volume: {prune_root}"
                )
            relative_path = _safe_relative_path(root, prune_root)
            marker = _publish_marker(prune_root, relative_path)
            if _directory_identity(prune_root) != identity:
                raise CleanupIdentityMismatchError(
                    f"prune root changed while its marker was published: {prune_root}"
                )
            entries.append(
                {
                    "stage_key": stage_key,
                    "relative_path": relative_path,
                    "operation": "delete_tree",
                    "cleanup_contract_version": 2,
                    "directory_identity": identity.to_object(),
                    "marker": marker,
                }
            )
    _emit(fault, "after_prune_markers")

    pruned_from = {
        "previous_status": metadata["status"],
        "previous_diagnostics_fingerprint": metadata["fingerprint"],
        "ordered_frame_manifest_fingerprint": metadata["ordered_frame_manifest_fingerprint"],
        "frames_jsonl_sha256": metadata["frames_jsonl_sha256"],
        "summary_sha256": metadata["summary_sha256"],
        "metric_clamp_summary_sha256": metadata["metric_clamp_summary_sha256"],
    }
    pruned = dict(metadata)
    pruned.update(
        {
            "status": "payload_pruned",
            "mask_payloads_enabled": False,
            "ordered_frame_manifest_fingerprint": None,
            "metric_clamp_summary_sha256": None,
            "pruned_from": pruned_from,
            "final_video_manifest_sha256": hashlib.sha256(final_manifest_raw).hexdigest(),
            "prune_entries": entries,
        }
    )
    pruned["fingerprint"] = self_fingerprint(pruned)
    pruned_raw = canonical_json_bytes(pruned)
    if reservation is not None:
        reservation.commit_payload_pruned_metadata(pruned_raw)
    else:
        _replace_metadata(metadata_path, pruned_raw)
    _emit(fault, "after_payload_pruned_metadata")
    return pruned


def _validate_marker_raw(entry: dict[str, Any], raw: bytes, marker_name: str) -> None:
    marker_value = entry.get("marker")
    if not isinstance(marker_value, dict) or set(marker_value) != {"name", "sha256"}:
        raise CleanupTransactionError("prune marker schema is invalid")
    if marker_name != marker_value["name"]:
        raise CleanupIdentityMismatchError("prune marker name differs")
    if hashlib.sha256(raw).hexdigest() != marker_value["sha256"]:
        raise CleanupIdentityMismatchError(f"prune marker hash differs: {marker_name}")
    expected = (
        canonical_json_bytes(
            {
                "cleanup_contract_version": 2,
                "relative_path": entry["relative_path"],
                "token": marker_name.removeprefix(".depth-surge-prune-v2-").removesuffix(".marker"),
            }
        )
        + b"\n"
    )
    if raw != expected:
        raise CleanupIdentityMismatchError(f"prune marker content differs: {marker_name}")


def _entry_is_reparse(metadata: os.stat_result) -> bool:
    return bool(
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _delete_tree_posix(  # noqa: C901
    job_root: Path,
    root: Path,
    entry: dict[str, Any],
    expected: _DirectoryIdentity,
    expected_job_root: _DirectoryIdentity,
    volume: int,
    mount_identity: _InvocationMountIdentity,
    *,
    after_marker_unlink: Callable[[], None],
) -> int:
    if os.name == "nt":
        raise AssertionError("POSIX cleanup adapter selected on Windows")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    job_root_fd = os.open(job_root, os.O_RDONLY | directory_flag | nofollow)
    opened_parent_fds: list[int] = []
    parent_fd = job_root_fd
    root_fd = -1
    removed = 0

    def checked_identity(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode) or int(metadata.st_dev) != volume:
            raise CleanupIdentityMismatchError(f"cleanup crossed a mount: {root}")
        _require_same_invocation_mount(descriptor, mount_identity, root)

    def attest_tree(directory_fd: int) -> None:
        checked_identity(directory_fd)
        with os.scandir(directory_fd) as iterator:
            for directory_entry in iterator:
                name = directory_entry.name
                if name == entry["marker"]["name"]:
                    continue
                metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if int(metadata.st_dev) != volume:
                    raise CleanupIdentityMismatchError(f"cleanup crossed a mount at {name}")
                if stat.S_ISDIR(metadata.st_mode) and not _entry_is_reparse(metadata):
                    child_fd = os.open(
                        name,
                        os.O_RDONLY | directory_flag | nofollow,
                        dir_fd=directory_fd,
                    )
                    try:
                        _require_same_invocation_mount(child_fd, mount_identity, name)
                        attest_tree(child_fd)
                    finally:
                        os.close(child_fd)

    def visit(directory_fd: int) -> None:
        nonlocal removed
        checked_identity(directory_fd)
        with os.scandir(directory_fd) as iterator:
            for directory_entry in iterator:
                name = directory_entry.name
                if name == entry["marker"]["name"]:
                    continue
                metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if int(metadata.st_dev) != volume:
                    raise CleanupIdentityMismatchError(f"cleanup crossed a mount at {name}")
                if stat.S_ISDIR(metadata.st_mode) and not _entry_is_reparse(metadata):
                    child_fd = os.open(
                        name,
                        os.O_RDONLY | directory_flag | nofollow,
                        dir_fd=directory_fd,
                    )
                    try:
                        visit(child_fd)
                    finally:
                        os.close(child_fd)
                    os.rmdir(name, dir_fd=directory_fd)
                else:
                    os.unlink(name, dir_fd=directory_fd)
                removed += 1

    try:
        try:
            opened_job_root_identity = directory_identity_from_posix_fd(job_root_fd)
        except PersistentIdentityCapabilityError as error:
            raise CleanupCapabilityError(str(error)) from error
        if opened_job_root_identity != expected_job_root:
            raise CleanupIdentityMismatchError("job root identity changed during cleanup")
        _require_same_invocation_mount(job_root_fd, mount_identity, job_root)
        relative_parts = PurePosixPath(entry["relative_path"]).parts
        for part in relative_parts[:-1]:
            child = os.open(
                part,
                os.O_RDONLY | directory_flag | nofollow,
                dir_fd=parent_fd,
            )
            try:
                _require_same_invocation_mount(child, mount_identity, part)
                if int(os.fstat(child).st_dev) != volume:
                    raise CleanupIdentityMismatchError(
                        f"cleanup crossed a mount at {part}"
                    )
            except BaseException:
                os.close(child)
                raise
            opened_parent_fds.append(child)
            parent_fd = child
        root_fd = os.open(
            relative_parts[-1],
            os.O_RDONLY | directory_flag | nofollow,
            dir_fd=parent_fd,
        )
        checked_identity(root_fd)
        try:
            opened_identity = directory_identity_from_posix_fd(root_fd)
        except PersistentIdentityCapabilityError as error:
            raise CleanupCapabilityError(str(error)) from error
        except PersistentIdentitySchemaError as error:
            raise CleanupIdentityMismatchError(f"prune root is unsafe: {root}") from error
        if opened_identity != expected:
            raise CleanupIdentityMismatchError(f"prune root identity differs: {root}")
        marker_name = str(entry["marker"]["name"])
        marker_present = False
        root_nonempty = False
        with os.scandir(root_fd) as iterator:
            for directory_entry in iterator:
                root_nonempty = True
                if directory_entry.name == marker_name:
                    marker_present = True
        if not marker_present:
            if root_nonempty:
                raise CleanupIdentityMismatchError(
                    f"prune marker is missing from a nonempty root: {root}"
                )
            reopened_root = os.open(
                relative_parts[-1],
                os.O_RDONLY | directory_flag | nofollow,
                dir_fd=parent_fd,
            )
            try:
                if directory_identity_from_posix_fd(reopened_root) != opened_identity:
                    raise CleanupIdentityMismatchError(
                        f"empty prune root identity changed: {root}"
                    )
                os.rmdir(relative_parts[-1], dir_fd=parent_fd)
            finally:
                os.close(reopened_root)
            return 0
        marker_fd = os.open(
            marker_name,
            os.O_RDONLY | nofollow | getattr(os, "O_NONBLOCK", 0),
            dir_fd=root_fd,
        )
        try:
            marker_metadata = os.fstat(marker_fd)
            if not stat.S_ISREG(marker_metadata.st_mode) or int(marker_metadata.st_dev) != volume:
                raise CleanupIdentityMismatchError(f"prune marker is unsafe: {root / marker_name}")
            _require_same_invocation_mount(marker_fd, mount_identity, root / marker_name)
            with os.fdopen(os.dup(marker_fd), "rb", buffering=0) as marker_handle:
                marker_raw = marker_handle.read(64 * 1024)
                if marker_handle.read(1):
                    raise CleanupIdentityMismatchError("prune marker exceeds its bound")
            marker_identity = file_identity_from_posix_fd(marker_fd)
            _validate_marker_raw(entry, marker_raw, marker_name)
            attest_tree(root_fd)
            visit(root_fd)
            remaining_count = 0
            remaining_marker = False
            with os.scandir(root_fd) as iterator:
                for directory_entry in iterator:
                    remaining_count += 1
                    remaining_marker = directory_entry.name == marker_name
            if remaining_count != 1 or not remaining_marker:
                raise CleanupIdentityMismatchError(
                    f"prune marker is not the sole final entry: {root}"
                )
            reopened_marker = os.open(
                marker_name,
                os.O_RDONLY | nofollow | getattr(os, "O_NONBLOCK", 0),
                dir_fd=root_fd,
            )
            try:
                if file_identity_from_posix_fd(reopened_marker) != marker_identity:
                    raise CleanupIdentityMismatchError(
                        f"prune marker identity changed during cleanup: {root / marker_name}"
                    )
                os.unlink(marker_name, dir_fd=root_fd)
            finally:
                os.close(reopened_marker)
        finally:
            os.close(marker_fd)
        removed += 1
        after_marker_unlink()
        with os.scandir(root_fd) as iterator:
            repopulated = next(iterator, None) is not None
        if repopulated:
            raise CleanupIdentityMismatchError(f"prune root repopulated during cleanup: {root}")
        reopened_root = os.open(
            relative_parts[-1],
            os.O_RDONLY | directory_flag | nofollow,
            dir_fd=parent_fd,
        )
        try:
            if directory_identity_from_posix_fd(reopened_root) != opened_identity:
                raise CleanupIdentityMismatchError(
                    f"prune root identity changed before removal: {root}"
                )
            os.rmdir(relative_parts[-1], dir_fd=parent_fd)
        finally:
            os.close(reopened_root)
        return removed
    finally:
        if root_fd >= 0:
            os.close(root_fd)
        for descriptor in reversed(opened_parent_fds):
            os.close(descriptor)
        os.close(job_root_fd)


def _delete_tree_windows(  # noqa: C901
    job_root: Path,
    root: Path,
    entry: dict[str, Any],
    expected: _DirectoryIdentity,
    expected_job_root: _DirectoryIdentity,
    volume: str,
    *,
    after_marker_unlink: Callable[[], None],
) -> int:  # noqa: C901
    if _requires_posix_mount_identity():
        raise AssertionError("Windows cleanup adapter selected off Windows")
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll")
    invalid_handle = wintypes.HANDLE(-1).value
    file_attribute_directory = 0x10
    file_attribute_reparse = 0x400

    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    get_directory = kernel32.GetFileInformationByHandleEx
    get_directory.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    get_directory.restype = wintypes.BOOL
    set_information = kernel32.SetFileInformationByHandle
    set_information.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    set_information.restype = wintypes.BOOL
    read_file = kernel32.ReadFile
    read_file.argtypes = (
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    )
    read_file.restype = wintypes.BOOL

    class UnicodeString(ctypes.Structure):
        _fields_ = [
            ("length", wintypes.USHORT),
            ("maximum_length", wintypes.USHORT),
            ("buffer", wintypes.LPWSTR),
        ]

    class ObjectAttributes(ctypes.Structure):
        _fields_ = [
            ("length", wintypes.ULONG),
            ("root_directory", wintypes.HANDLE),
            ("object_name", ctypes.POINTER(UnicodeString)),
            ("attributes", wintypes.ULONG),
            ("security_descriptor", wintypes.LPVOID),
            ("security_qos", wintypes.LPVOID),
        ]

    class IoStatusBlock(ctypes.Structure):
        _fields_ = [("status", ctypes.c_void_p), ("information", ctypes.c_size_t)]

    nt_create_file = ntdll.NtCreateFile
    nt_create_file.argtypes = (
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        ctypes.POINTER(ObjectAttributes),
        ctypes.POINTER(IoStatusBlock),
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    nt_create_file.restype = ctypes.c_long
    status_to_error = ntdll.RtlNtStatusToDosError
    status_to_error.argtypes = (ctypes.c_long,)
    status_to_error.restype = wintypes.DWORD

    def win_error(code: int | None = None) -> OSError:
        return ctypes.WinError(ctypes.get_last_error() if code is None else code)

    def identity(handle: int, *, require_directory: bool, allow_reparse: bool = False):
        try:
            value, _links, attributes = windows_identity_from_handle(
                handle,
                require_directory=require_directory,
                allow_reparse=allow_reparse,
            )
        except PersistentIdentityCapabilityError as error:
            raise CleanupCapabilityError(str(error)) from error
        except PersistentIdentitySchemaError as error:
            raise CleanupIdentityMismatchError("cleanup handle type changed") from error
        return value, attributes

    def open_job_root() -> int:
        handle = create_file(
            str(job_root),
            0x00010000 | 0x00000001 | 0x00000080 | 0x00100000,
            0x1 | 0x2 | 0x4,
            None,
            3,
            0x02000000 | 0x00200000,
            None,
        )
        if handle == invalid_handle:
            raise win_error()
        return int(handle)

    def open_relative(parent: int, name: str, *, directory: bool) -> int:
        name_buffer = ctypes.create_unicode_buffer(name)
        encoded_length = len(name.encode("utf-16-le"))
        unicode_name = UnicodeString(
            encoded_length,
            encoded_length + 2,
            ctypes.cast(name_buffer, wintypes.LPWSTR),
        )
        attributes = ObjectAttributes(
            ctypes.sizeof(ObjectAttributes),
            wintypes.HANDLE(parent),
            ctypes.pointer(unicode_name),
            0x40,
            None,
            None,
        )
        child = wintypes.HANDLE()
        io_status = IoStatusBlock()
        options = 0x00200000 | 0x20 | (0x1 if directory else 0x40)
        status = int(
            nt_create_file(
                ctypes.byref(child),
                0x00010000 | 0x00000001 | 0x00000080 | 0x00100000,
                ctypes.byref(attributes),
                ctypes.byref(io_status),
                None,
                0,
                0x1 | 0x2 | 0x4,
                1,
                options,
                None,
                0,
            )
        )
        if status < 0:
            raise win_error(int(status_to_error(status)))
        if child.value is None:
            raise CleanupTransactionError("NtCreateFile returned a null child handle")
        return int(child.value)

    def iter_names(handle: int):
        """Yield one fixed-buffer directory page at a time."""

        restart = True
        while True:
            buffer = ctypes.create_string_buffer(64 * 1024)
            if not get_directory(
                wintypes.HANDLE(handle),
                11 if restart else 10,
                buffer,
                len(buffer),
            ):
                code = ctypes.get_last_error()
                if code == 18:
                    return
                raise win_error(code)
            restart = False
            offset = 0
            while True:
                next_offset = ctypes.c_uint32.from_buffer(buffer, offset).value
                attributes = ctypes.c_uint32.from_buffer(buffer, offset + 56).value
                name_length = ctypes.c_uint32.from_buffer(buffer, offset + 60).value
                name = ctypes.wstring_at(ctypes.addressof(buffer) + offset + 104, name_length // 2)
                if name not in {".", ".."}:
                    yield name, int(attributes)
                if next_offset == 0:
                    break
                offset += int(next_offset)

    def first_name(handle: int, *, excluded: str | None = None) -> tuple[str, int] | None:
        # Restart after each deletion. Continuing a Windows directory cursor across
        # namespace mutation may skip the entry which moved into the current slot.
        for name, attributes in iter_names(handle):
            if name != excluded:
                return name, attributes
        return None

    def delete_on_close(handle: int) -> None:
        delete = wintypes.BOOL(True)
        if not set_information(
            wintypes.HANDLE(handle),
            4,
            ctypes.byref(delete),
            ctypes.sizeof(delete),
        ):
            raise win_error()

    def read_marker(handle: int) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while True:
            buffer = ctypes.create_string_buffer(4096)
            count = wintypes.DWORD()
            if not read_file(
                wintypes.HANDLE(handle),
                buffer,
                len(buffer),
                ctypes.byref(count),
                None,
            ):
                raise win_error()
            if count.value == 0:
                break
            total += int(count.value)
            if total > 64 * 1024:
                raise CleanupIdentityMismatchError("prune marker exceeds its bound")
            chunks.append(buffer.raw[: count.value])
        return b"".join(chunks)

    removed = 0

    def visit(directory_handle: int, marker_name: str | None = None) -> None:
        nonlocal removed
        directory_identity, attributes = identity(directory_handle, require_directory=True)
        if directory_identity.volume_serial != volume or attributes & file_attribute_reparse:
            raise CleanupIdentityMismatchError("cleanup crossed a volume or reparse point")
        while True:
            next_entry = first_name(directory_handle, excluded=marker_name)
            if next_entry is None:
                return
            name, child_attributes = next_entry
            is_directory = bool(child_attributes & file_attribute_directory)
            child = open_relative(directory_handle, name, directory=is_directory)
            try:
                child_identity, current_attributes = identity(
                    child,
                    require_directory=is_directory,
                    allow_reparse=True,
                )
                if child_identity.volume_serial != volume:
                    raise CleanupIdentityMismatchError(f"cleanup crossed a volume at {name}")
                if is_directory and not current_attributes & file_attribute_reparse:
                    visit(child)
                delete_on_close(child)
            finally:
                close_handle(wintypes.HANDLE(child))
            removed += 1

    job_root_handle = open_job_root()
    opened_parent_handles: list[int] = []
    root_handle = -1
    try:
        job_root_file_identity, job_root_attributes = identity(
            job_root_handle,
            require_directory=True,
        )
        if (
            _DirectoryIdentity("windows", job_root_file_identity) != expected_job_root
            or job_root_file_identity.volume_serial != volume
            or job_root_attributes & file_attribute_reparse
        ):
            raise CleanupIdentityMismatchError("job root identity changed during cleanup")
        parent_handle = job_root_handle
        relative_parts = PurePosixPath(entry["relative_path"]).parts
        for part_index, part in enumerate(relative_parts):
            child_handle = open_relative(parent_handle, part, directory=True)
            child_identity, child_attributes = identity(
                child_handle,
                require_directory=True,
            )
            if (
                child_identity.volume_serial != volume
                or child_attributes & file_attribute_reparse
            ):
                close_handle(wintypes.HANDLE(child_handle))
                raise CleanupIdentityMismatchError(f"cleanup crossed a volume at {part}")
            if part_index == len(relative_parts) - 1:
                root_handle = child_handle
            else:
                opened_parent_handles.append(child_handle)
                parent_handle = child_handle
        if root_handle < 0:
            raise CleanupTransactionError("prune root path is empty")
        actual_file_identity, attributes = identity(root_handle, require_directory=True)
        actual = _DirectoryIdentity("windows", actual_file_identity)
        if (
            actual != expected
            or actual_file_identity.volume_serial != volume
            or attributes & file_attribute_reparse
        ):
            raise CleanupIdentityMismatchError(f"prune root identity differs: {root}")
        marker_name = str(entry["marker"]["name"])
        marker_present = False
        other_present = False
        for name, _attributes in iter_names(root_handle):
            marker_present = marker_present or name == marker_name
            other_present = other_present or name != marker_name
        if not marker_present:
            if other_present:
                raise CleanupIdentityMismatchError(
                    f"prune marker is missing from a nonempty root: {root}"
                )
            delete_on_close(root_handle)
            return 0
        marker_handle = open_relative(root_handle, marker_name, directory=False)
        try:
            marker_identity, marker_attributes = identity(
                marker_handle,
                require_directory=False,
            )
            if marker_identity.volume_serial != volume or marker_attributes & file_attribute_reparse:
                raise CleanupIdentityMismatchError(f"prune marker is unsafe: {root / marker_name}")
            marker_raw = read_marker(marker_handle)
            _validate_marker_raw(entry, marker_raw, marker_name)
            visit(root_handle, marker_name)
            remaining_count = 0
            marker_still_present = False
            for name, _attributes in iter_names(root_handle):
                remaining_count += 1
                marker_still_present = marker_still_present or name == marker_name
                if remaining_count > 1:
                    break
            if remaining_count != 1 or not marker_still_present:
                raise CleanupIdentityMismatchError(
                    f"prune marker is not the sole final entry: {root}"
                )
            delete_on_close(marker_handle)
        finally:
            close_handle(wintypes.HANDLE(marker_handle))
        removed += 1
        after_marker_unlink()
        if first_name(root_handle) is not None:
            raise CleanupIdentityMismatchError(f"prune root repopulated during cleanup: {root}")
        delete_on_close(root_handle)
        return removed
    finally:
        if root_handle >= 0:
            close_handle(wintypes.HANDLE(root_handle))
        for handle in reversed(opened_parent_handles):
            close_handle(wintypes.HANDLE(handle))
        close_handle(wintypes.HANDLE(job_root_handle))


def _delete_descendants_no_follow(
    job_root: Path,
    root: Path,
    entry: dict[str, Any],
    expected: _DirectoryIdentity,
    expected_job_root: _DirectoryIdentity,
    volume: int | str,
    mount_identity: _InvocationMountIdentity | None = None,
    *,
    after_marker_unlink: Callable[[], None],
) -> int:
    if os.name == "nt":
        if not isinstance(volume, str):
            raise CleanupCapabilityError("Windows cleanup has no stable volume identity")
        return _delete_tree_windows(
            job_root,
            root,
            entry,
            expected,
            expected_job_root,
            volume,
            after_marker_unlink=after_marker_unlink,
        )
    if mount_identity is None:
        raise CleanupCapabilityError("POSIX cleanup has no job-root mount identity")
    if not isinstance(volume, int):
        raise CleanupCapabilityError("POSIX cleanup has no device identity")
    return _delete_tree_posix(
        job_root,
        root,
        entry,
        expected,
        expected_job_root,
        volume,
        mount_identity,
        after_marker_unlink=after_marker_unlink,
    )


def _parse_payload_pruned(job_root: Path) -> dict[str, Any]:
    path = Path(job_root) / "04_stereo_diagnostics" / "metadata.json"
    raw = path.read_bytes()
    if len(raw) > 16 * 1024 * 1024:
        raise CleanupTransactionError("payload-pruned metadata exceeds its bound")
    value = _parse_canonical_object(raw)
    return _validate_payload_pruned_authorization(Path(job_root), value)


def run_authorized_cleanup(
    job_root: Path,
    *,
    metadata: dict[str, Any] | None = None,
    fault: FaultHook | None = None,
) -> int:
    """Delete only persisted, identity-bound roots and keep each marker until last."""

    root = Path(job_root).resolve()
    value = (
        _validate_payload_pruned_authorization(root, metadata)
        if metadata is not None
        else _parse_payload_pruned(root)
    )
    root_identity = _directory_identity(root)
    mount_identity: _InvocationMountIdentity | None = None
    if os.name != "nt":
        directory_flag = getattr(os, "O_DIRECTORY", 0)
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        root_fd = os.open(root, os.O_RDONLY | directory_flag | nofollow)
        try:
            mount_identity = _linux_mount_identity(root_fd)
        finally:
            os.close(root_fd)
    removed = 0
    for index, entry_value in enumerate(value["prune_entries"]):
        assert isinstance(entry_value, dict)
        entry = entry_value
        relative = PurePosixPath(entry["relative_path"])
        prune_root = root.joinpath(*relative.parts)
        if not (prune_root.exists() or prune_root.is_symlink()):
            _emit(fault, f"after_prune_entry_{index}")
            continue
        expected = _DirectoryIdentity.from_object(entry["directory_identity"])
        actual = _directory_identity(prune_root)
        if actual != expected or actual.volume_key != root_identity.volume_key:
            raise CleanupIdentityMismatchError(f"prune root identity differs: {prune_root}")

        def emit_marker_unlink(index: int = index) -> None:
            _emit(fault, f"after_prune_marker_unlink_{index}")

        removed += _delete_descendants_no_follow(
            root,
            prune_root,
            entry,
            expected,
            root_identity,
            (
                root_identity.file_identity.volume_serial
                if root_identity.kind == "windows"
                else int(root.stat(follow_symlinks=False).st_dev)
            ),
            mount_identity,
            after_marker_unlink=emit_marker_unlink,
        )
        _emit(fault, f"after_prune_entry_{index}")
    return removed


def cleanup_status_for_error(error: BaseException) -> str:
    if isinstance(error, CleanupIdentityMismatchError):
        return "incomplete_identity_mismatch"
    return "incomplete_error"
