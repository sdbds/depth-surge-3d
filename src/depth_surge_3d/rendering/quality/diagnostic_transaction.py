"""Atomic RGB and compact-diagnostics publication for one Quality frame."""

from __future__ import annotations

import hashlib
import io
import json
import os
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import cv2
import numpy as np

from .diagnostics import build_diagnostics_summary, build_quality_frame_stats
from .types import CompactStereoRenderResult, DenseHistogramPairs


FaultHook = Callable[[str], None]
_HEX_64 = frozenset("0123456789abcdef")


def canonical_json_bytes(value: object) -> bytes:
    """Encode canonical JSON while streaming dense histogram pairs."""

    output = io.BytesIO()

    def write(current: object) -> None:  # noqa: C901
        if isinstance(current, DenseHistogramPairs):
            output.write(b"[")
            first = True
            for bucket in range(int(current.bins.size)):
                count = int(current.bins[bucket])
                if count == 0:
                    continue
                if not first:
                    output.write(b",")
                output.write(f"[{bucket},{count}]".encode("ascii"))
                first = False
            output.write(b"]")
            return
        if current is None:
            output.write(b"null")
            return
        if isinstance(current, (bool, np.bool_)):
            output.write(b"true" if bool(current) else b"false")
            return
        if isinstance(current, (int, np.integer)):
            output.write(str(int(current)).encode("ascii"))
            return
        if isinstance(current, (float, np.floating)):
            output.write(
                json.dumps(float(current), allow_nan=False, separators=(",", ":")).encode(
                    "ascii"
                )
            )
            return
        if isinstance(current, str):
            output.write(json.dumps(current, ensure_ascii=True).encode("ascii"))
            return
        if isinstance(current, dict):
            if not all(isinstance(key, str) for key in current):
                raise TypeError("canonical JSON object keys must be strings")
            output.write(b"{")
            for index, key in enumerate(sorted(current)):
                if index:
                    output.write(b",")
                write(key)
                output.write(b":")
                write(current[key])
            output.write(b"}")
            return
        if isinstance(current, (list, tuple)):
            output.write(b"[")
            for index, item in enumerate(current):
                if index:
                    output.write(b",")
                write(item)
            output.write(b"]")
            return
        raise TypeError(f"unsupported canonical JSON value: {type(current).__name__}")

    write(value)
    return output.getvalue()


def self_fingerprint(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "fingerprint"}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class QualityFramePublication:
    """All immutable inputs for one frame transaction."""

    job_root: Path
    frame_name: str
    left_rgb_path: Path
    right_rgb_path: Path
    compact: CompactStereoRenderResult
    geometry_mode: str
    occlusion_fill: str
    rgb_stage_fingerprint: str
    keep_masks: bool
    render_mode: Literal["fast", "quality"] = "quality"

    def __post_init__(self) -> None:
        root = Path(self.job_root).resolve()
        if not self.frame_name or Path(self.frame_name).name != self.frame_name:
            raise ValueError("frame_name must be one nonempty path component")
        if self.geometry_mode not in {"relative", "metric"}:
            raise ValueError("geometry_mode must be relative or metric")
        if self.occlusion_fill not in {"none", "background"}:
            raise ValueError("occlusion_fill must be none or background")
        if not isinstance(self.keep_masks, bool):
            raise TypeError("keep_masks must be boolean")
        if self.render_mode not in {"fast", "quality"}:
            raise ValueError("render_mode must be fast or quality")
        for name in ("left_rgb_path", "right_rgb_path"):
            path = Path(getattr(self, name)).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError(f"{name} must be contained by job_root") from error
            object.__setattr__(self, name, path)
        object.__setattr__(self, "job_root", root)


@dataclass(frozen=True)
class PreparedQualityFramePublication:
    """Writer payload with frame statistics serialized and planning state released."""

    job_root: Path
    frame_name: str
    left_rgb_path: Path
    right_rgb_path: Path
    left_image: np.ndarray
    right_image: np.ndarray
    left_coverage_count: np.ndarray
    right_coverage_count: np.ndarray
    left_repair_bits: np.ndarray
    right_repair_bits: np.ndarray
    stats_raw: bytes
    render_shape: tuple[int, int]
    geometry_mode: str
    occlusion_fill: str
    rgb_stage_fingerprint: str
    keep_masks: bool
    render_mode: Literal["fast", "quality"]
    quality_limits: dict[str, dict[str, int | float]] | None


def prepare_quality_frame(
    publication: QualityFramePublication,
) -> PreparedQualityFramePublication:
    """Serialize variable diagnostics before crossing the writer queue boundary."""

    if not isinstance(publication, QualityFramePublication):
        raise TypeError("publication must be QualityFramePublication")
    compact = publication.compact
    stats = build_quality_frame_stats(
        compact,
        frame_name=publication.frame_name,
        geometry_mode=publication.geometry_mode,
        occlusion_fill=publication.occlusion_fill,
        render_mode=publication.render_mode,
    )
    return PreparedQualityFramePublication(
        job_root=publication.job_root,
        frame_name=publication.frame_name,
        left_rgb_path=publication.left_rgb_path,
        right_rgb_path=publication.right_rgb_path,
        left_image=compact.left_image,
        right_image=compact.right_image,
        left_coverage_count=compact.left_coverage_count,
        right_coverage_count=compact.right_coverage_count,
        left_repair_bits=compact.left_repair_bits,
        right_repair_bits=compact.right_repair_bits,
        stats_raw=canonical_json_bytes(stats),
        render_shape=(int(compact.left_image.shape[0]), int(compact.left_image.shape[1])),
        geometry_mode=publication.geometry_mode,
        occlusion_fill=publication.occlusion_fill,
        rgb_stage_fingerprint=publication.rgb_stage_fingerprint,
        keep_masks=publication.keep_masks,
        render_mode=publication.render_mode,
        quality_limits=compact.quality_limits,
    )


@dataclass(frozen=True)
class DiagnosticsStageContext:
    """Closed identity required to build or audit one diagnostics generation."""

    job_root: Path
    frame_names: tuple[str, ...]
    render_shape: tuple[int, int]
    render_mode: Literal["fast", "quality"]
    geometry_mode: Literal["relative", "metric"]
    occlusion_fill: Literal["none", "background"]
    mask_payloads_enabled: bool
    rgb_stage_fingerprint: str
    source_guide_fingerprint: str
    native_geometry_fingerprint: str | None
    quality_input_manifest_sha256: str | None
    left_rgb_parent: Path | None = None
    right_rgb_parent: Path | None = None

    def __post_init__(self) -> None:  # noqa: C901
        root = Path(self.job_root).resolve()
        if not self.frame_names or len(set(self.frame_names)) != len(self.frame_names):
            raise ValueError("diagnostics frame_names must be nonempty and unique")
        for name in self.frame_names:
            if not name or Path(name).name != name or Path(name).suffix:
                raise ValueError("diagnostics frame names must be plain stems")
        if len(self.render_shape) != 2 or any(
            isinstance(value, bool) or int(value) <= 0 for value in self.render_shape
        ):
            raise ValueError("diagnostics render_shape must be positive")
        if self.render_mode not in {"fast", "quality"}:
            raise ValueError("diagnostics render_mode is invalid")
        if self.geometry_mode not in {"relative", "metric"}:
            raise ValueError("diagnostics geometry_mode is invalid")
        if self.occlusion_fill not in {"none", "background"}:
            raise ValueError("diagnostics occlusion_fill is invalid")
        if not isinstance(self.mask_payloads_enabled, bool):
            raise TypeError("mask_payloads_enabled must be boolean")
        if self.render_mode == "quality":
            if not _is_hex64(self.quality_input_manifest_sha256):
                raise ValueError("Quality diagnostics require an input-manifest SHA-256")
            if not self.native_geometry_fingerprint:
                raise ValueError("Quality diagnostics require native geometry identity")
        elif self.quality_input_manifest_sha256 is not None:
            raise ValueError("Fast diagnostics cannot bind a Quality input manifest")
        for field_name, fallback in (
            ("left_rgb_parent", root / "04_left_frames"),
            ("right_rgb_parent", root / "04_right_frames"),
        ):
            path = Path(getattr(self, field_name) or fallback).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError("diagnostics RGB parent escapes the job root") from error
            object.__setattr__(self, field_name, path)
        object.__setattr__(self, "job_root", root)
        object.__setattr__(
            self,
            "render_shape",
            (int(self.render_shape[0]), int(self.render_shape[1])),
        )


@dataclass(frozen=True)
class DiagnosticsAudit:
    disposition: Literal["absent", "building", "complete", "payload_pruned"]
    reusable_frame_names: tuple[str, ...]
    invalid_frame_names: tuple[str, ...]
    aggregates_valid: bool
    migrate_frame_names: tuple[str, ...] = ()


def _is_hex64(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_64 for character in value)
    )


def _png_bytes(values: np.ndarray) -> bytes:
    encoded, payload = cv2.imencode(".png", values)
    if not encoded:
        raise OSError("could not encode diagnostics PNG")
    return payload.tobytes()


def _identity(root: Path, path: Path, raw: bytes) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }


def _write_temporary(
    path: Path,
    raw: bytes,
    *,
    fault: FaultHook | None = None,
    label: str | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            if label is not None:
                _call_fault(fault, f"after_{label}_temporary_write")
            handle.flush()
            os.fsync(handle.fileno())
            if label is not None:
                _call_fault(fault, f"after_{label}_temporary_sync")
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _sync_parent(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _call_fault(fault: FaultHook | None, stage: str) -> None:
    if fault is not None:
        fault(stage)


def publish_quality_frame(
    publication: QualityFramePublication | PreparedQualityFramePublication,
    *,
    fault: FaultHook | None = None,
) -> dict[str, Any]:
    """Publish every payload first and the authenticated manifest last."""

    if isinstance(publication, QualityFramePublication):
        publication = prepare_quality_frame(publication)
    elif not isinstance(publication, PreparedQualityFramePublication):
        raise TypeError("publication must be a Quality frame publication")
    root = publication.job_root
    frame_dir = root / "04_stereo_diagnostics" / "frames" / publication.frame_name
    stats_path = frame_dir / "stats.json"
    mask_paths = {
        "left_coverage": frame_dir / "left_coverage.png",
        "left_repair": frame_dir / "left_repair.png",
        "right_coverage": frame_dir / "right_coverage.png",
        "right_repair": frame_dir / "right_repair.png",
    }
    raw_payloads: dict[str, tuple[Path, bytes] | None] = {
        "left_rgb": (publication.left_rgb_path, _png_bytes(publication.left_image)),
        "right_rgb": (
            publication.right_rgb_path,
            _png_bytes(publication.right_image),
        ),
        "left_coverage": None,
        "left_repair": None,
        "right_coverage": None,
        "right_repair": None,
        "stats": (stats_path, publication.stats_raw),
    }
    if publication.keep_masks:
        raw_payloads.update(
            {
                "left_coverage": (
                    mask_paths["left_coverage"],
                    _png_bytes(publication.left_coverage_count),
                ),
                "left_repair": (
                    mask_paths["left_repair"],
                    _png_bytes(publication.left_repair_bits),
                ),
                "right_coverage": (
                    mask_paths["right_coverage"],
                    _png_bytes(publication.right_coverage_count),
                ),
                "right_repair": (
                    mask_paths["right_repair"],
                    _png_bytes(publication.right_repair_bits),
                ),
            }
        )
    payload_identities = {
        name: None if item is None else _identity(root, item[0], item[1])
        for name, item in raw_payloads.items()
    }
    limits = publication.quality_limits
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "algorithm_version": "stereo-coverage-sidecar-v1",
        "frame_name": publication.frame_name,
        "render_mode": publication.render_mode,
        "geometry_mode": publication.geometry_mode,
        "occlusion_fill": publication.occlusion_fill,
        "render_shape": [
            int(publication.render_shape[0]),
            int(publication.render_shape[1]),
        ],
        "rgb_stage_fingerprint": publication.rgb_stage_fingerprint,
        "mask_payloads_enabled": publication.keep_masks,
        "quality_limits": limits if publication.occlusion_fill == "background" else None,
        "payloads": payload_identities,
    }
    manifest["fingerprint"] = self_fingerprint(manifest)
    manifest_path = frame_dir / "manifest.json"
    raw_manifest = canonical_json_bytes(manifest)
    temporaries: dict[Path, Path] = {}
    final_paths = [item[0] for item in raw_payloads.values() if item is not None]
    try:
        for name, item in raw_payloads.items():
            if item is not None:
                path, raw = item
                temporaries[path] = _write_temporary(
                    path,
                    raw,
                    fault=fault,
                    label=name,
                )
        temporaries[manifest_path] = _write_temporary(
            manifest_path,
            raw_manifest,
            fault=fault,
            label="manifest",
        )
        for name in (
            "left_rgb",
            "right_rgb",
            "left_coverage",
            "left_repair",
            "right_coverage",
            "right_repair",
            "stats",
        ):
            item = raw_payloads[name]
            if item is None:
                continue
            path = item[0]
            os.replace(temporaries.pop(path), path)
            _call_fault(fault, f"after_{name}_replace")
            _sync_parent(path.parent)
            _call_fault(fault, f"after_{name}_directory_sync")
        _call_fault(fault, "before_manifest_replace")
        os.replace(temporaries.pop(manifest_path), manifest_path)
        _call_fault(fault, "after_manifest_replace")
        _sync_parent(manifest_path.parent)
        _call_fault(fault, "after_manifest_directory_sync")
        return manifest
    except Exception:
        manifest_path.unlink(missing_ok=True)
        for path in final_paths:
            path.unlink(missing_ok=True)
        raise
    finally:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)


def _atomic_replace_raw(
    path: Path,
    raw: bytes,
    *,
    fault: FaultHook | None = None,
    label: str,
) -> None:
    temporary = _write_temporary(path, raw, fault=fault, label=label)
    try:
        _call_fault(fault, f"before_{label}_replace")
        os.replace(temporary, path)
        _call_fault(fault, f"after_{label}_replace")
        _sync_parent(path.parent)
        _call_fault(fault, f"after_{label}_directory_sync")
    finally:
        temporary.unlink(missing_ok=True)


def _diagnostics_metadata(
    context: DiagnosticsStageContext,
    *,
    status: Literal["building", "complete"],
    ordered_frame_manifest_fingerprint: str | None = None,
    frames_jsonl_sha256: str | None = None,
    summary_sha256: str | None = None,
    metric_clamp_summary_sha256: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "algorithm_version": "stereo-coverage-sidecar-v1",
        "status": status,
        "rgb_stage_fingerprint": context.rgb_stage_fingerprint,
        "source_guide_fingerprint": context.source_guide_fingerprint,
        "native_geometry_fingerprint": context.native_geometry_fingerprint,
        "quality_input_manifest_sha256": context.quality_input_manifest_sha256,
        "frame_names": list(context.frame_names),
        "render_shape": [context.render_shape[0], context.render_shape[1]],
        "render_mode": context.render_mode,
        "geometry_mode": context.geometry_mode,
        "occlusion_fill": context.occlusion_fill,
        "mask_payloads_enabled": context.mask_payloads_enabled,
        "stats_schema_version": 1,
        "frame_manifest_schema_version": 1,
        "ordered_frame_manifest_fingerprint": ordered_frame_manifest_fingerprint,
        "frames_jsonl_sha256": frames_jsonl_sha256,
        "summary_sha256": summary_sha256,
        "metric_clamp_summary_sha256": metric_clamp_summary_sha256,
        "pruned_from": None,
        "final_video_manifest_sha256": None,
        "prune_entries": None,
    }
    metadata["fingerprint"] = self_fingerprint(metadata)
    return metadata


def begin_diagnostics_stage(
    context: DiagnosticsStageContext,
    *,
    fault: FaultHook | None = None,
) -> dict[str, Any]:
    """Publish the building identity before the first frame mutation."""

    if not isinstance(context, DiagnosticsStageContext):
        raise TypeError("context must be DiagnosticsStageContext")
    metadata = _diagnostics_metadata(context, status="building")
    path = context.job_root / "04_stereo_diagnostics" / "metadata.json"
    _atomic_replace_raw(
        path,
        canonical_json_bytes(metadata),
        fault=fault,
        label="building_metadata",
    )
    return metadata


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_canonical_object(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("diagnostics JSON is not canonical ASCII JSON") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError("diagnostics JSON bytes are not canonical")
    return value


def _read_canonical_object(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    return _parse_canonical_object(raw), raw


def _contained_payload_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise ValueError("diagnostic payload path must be a string")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("diagnostic payload path escapes the job root")
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("diagnostic payload path escapes the job root") from error
    return path


def _audit_payload_identity(
    root: Path,
    identity: object,
    *,
    expected_path: Path,
) -> bytes:
    if not isinstance(identity, dict) or set(identity) != {
        "relative_path",
        "sha256",
        "byte_count",
    }:
        raise ValueError("diagnostic payload identity has the wrong keys")
    path = _contained_payload_path(root, identity["relative_path"])
    if path != expected_path.resolve():
        raise ValueError("diagnostic payload path does not match its declared slot")
    raw = path.read_bytes()
    if (
        isinstance(identity["byte_count"], bool)
        or not isinstance(identity["byte_count"], int)
        or identity["byte_count"] != len(raw)
        or not _is_hex64(identity["sha256"])
        or identity["sha256"] != hashlib.sha256(raw).hexdigest()
    ):
        raise ValueError("diagnostic payload identity does not match its bytes")
    return raw


def _audit_png(raw: bytes, shape: tuple[int, int], *, colour: bool) -> None:
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.shape[:2] != shape or decoded.dtype != np.uint8:
        raise ValueError("diagnostic PNG header or payload is invalid")
    expected_ndim = 3 if colour else 2
    if decoded.ndim != expected_ndim or (colour and decoded.shape[2] != 3):
        raise ValueError("diagnostic PNG channel layout is invalid")


_FRAME_MANIFEST_KEYS = {
    "schema_version",
    "algorithm_version",
    "frame_name",
    "render_mode",
    "geometry_mode",
    "occlusion_fill",
    "render_shape",
    "rgb_stage_fingerprint",
    "mask_payloads_enabled",
    "quality_limits",
    "payloads",
    "fingerprint",
}
_PAYLOAD_KEYS = {
    "left_rgb",
    "right_rgb",
    "left_coverage",
    "left_repair",
    "right_coverage",
    "right_repair",
    "stats",
}


def _audit_frame_transaction(  # noqa: C901
    context: DiagnosticsStageContext,
    frame_name: str,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    root = context.job_root
    frame_dir = root / "04_stereo_diagnostics" / "frames" / frame_name
    manifest, raw_manifest = _read_canonical_object(frame_dir / "manifest.json")
    if set(manifest) != _FRAME_MANIFEST_KEYS:
        raise ValueError("diagnostic frame manifest has the wrong keys")
    expected_scalars = {
        "schema_version": 1,
        "algorithm_version": "stereo-coverage-sidecar-v1",
        "frame_name": frame_name,
        "render_mode": context.render_mode,
        "geometry_mode": context.geometry_mode,
        "occlusion_fill": context.occlusion_fill,
        "render_shape": list(context.render_shape),
        "rgb_stage_fingerprint": context.rgb_stage_fingerprint,
        "mask_payloads_enabled": context.mask_payloads_enabled,
    }
    if any(manifest[key] != value for key, value in expected_scalars.items()):
        raise ValueError("diagnostic frame manifest identity changed")
    if not _is_hex64(manifest["fingerprint"]) or manifest["fingerprint"] != self_fingerprint(
        manifest
    ):
        raise ValueError("diagnostic frame manifest fingerprint is invalid")
    payloads = manifest["payloads"]
    if not isinstance(payloads, dict) or set(payloads) != _PAYLOAD_KEYS:
        raise ValueError("diagnostic frame manifest payload map is invalid")
    assert context.left_rgb_parent is not None
    assert context.right_rgb_parent is not None
    left_raw = _audit_payload_identity(
        root,
        payloads["left_rgb"],
        expected_path=context.left_rgb_parent / f"{frame_name}.png",
    )
    right_raw = _audit_payload_identity(
        root,
        payloads["right_rgb"],
        expected_path=context.right_rgb_parent / f"{frame_name}.png",
    )
    _audit_png(left_raw, context.render_shape, colour=True)
    _audit_png(right_raw, context.render_shape, colour=True)
    mask_names = ("left_coverage", "left_repair", "right_coverage", "right_repair")
    for name in mask_names:
        identity = payloads[name]
        if context.mask_payloads_enabled:
            mask_raw = _audit_payload_identity(
                root,
                identity,
                expected_path=frame_dir / f"{name}.png",
            )
            _audit_png(mask_raw, context.render_shape, colour=False)
        elif identity is not None:
            raise ValueError("disabled diagnostic mask has a payload identity")
    stats_raw = _audit_payload_identity(
        root,
        payloads["stats"],
        expected_path=frame_dir / "stats.json",
    )
    stats = _parse_canonical_object(stats_raw)
    expected_stats = {
        "schema_version": 1,
        "frame_name": frame_name,
        "render_mode": context.render_mode,
        "geometry_mode": context.geometry_mode,
        "occlusion_fill": context.occlusion_fill,
        "render_shape": list(context.render_shape),
    }
    if any(stats.get(key) != value for key, value in expected_stats.items()):
        raise ValueError("diagnostic frame statistics identity changed")
    if set(stats) != {
        "schema_version",
        "frame_name",
        "render_mode",
        "geometry_mode",
        "occlusion_fill",
        "render_shape",
        "eyes",
        "metric_projection",
        "geometry_nearest",
    }:
        raise ValueError("diagnostic frame statistics have the wrong keys")
    if (
        manifest["quality_limits"]
        != {eye: stats["eyes"][eye]["quality_limits"] for eye in ("left", "right")}
        and manifest["quality_limits"] is not None
    ):
        raise ValueError("manifest and statistics Quality limits disagree")
    return manifest, raw_manifest, stats


def _compatibility_clamp_summary(summary: dict[str, Any]) -> dict[str, Any]:
    metric = summary["metric_projection"]
    if not isinstance(metric, dict):
        raise ValueError("metric compatibility summary requires metric aggregate")
    return {
        "schema_version": 1,
        "frame_names": summary["frame_names"],
        "clamped_fractions": metric["clamped_fractions"],
        "affected_frame_count": metric["affected_frame_count"],
        "mean_clamped_fraction": metric["mean_clamped_fraction"],
        "max_clamped_fraction": metric["max_clamped_fraction"],
    }


def complete_diagnostics_stage(
    context: DiagnosticsStageContext,
    *,
    fault: FaultHook | None = None,
) -> dict[str, Any]:
    """Consolidate committed frames and publish complete metadata last."""

    root = context.job_root / "04_stereo_diagnostics"
    jsonl_path = root / "stereo_coverage_frames.jsonl"
    jsonl_temporary = jsonl_path.with_name(f".{jsonl_path.name}.{uuid.uuid4().hex}.tmp")
    ordered_digest = hashlib.sha256()
    ordered_digest.update(b"[")
    jsonl_digest = hashlib.sha256()
    try:
        with jsonl_temporary.open("xb", buffering=0) as handle:
            for index, frame_name in enumerate(context.frame_names):
                _manifest, raw_manifest, stats = _audit_frame_transaction(context, frame_name)
                if index:
                    ordered_digest.update(b",")
                ordered_digest.update(
                    canonical_json_bytes(
                        {
                            "frame_name": frame_name,
                            "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
                        }
                    )
                )
                line = canonical_json_bytes(stats) + b"\n"
                if handle.write(line) != len(line):
                    raise OSError("short diagnostics JSONL write")
                jsonl_digest.update(line)
            _call_fault(fault, "after_frames_jsonl_temporary_write")
            handle.flush()
            os.fsync(handle.fileno())
            _call_fault(fault, "after_frames_jsonl_temporary_sync")
        ordered_digest.update(b"]")
        ordered_fingerprint = ordered_digest.hexdigest()
        jsonl_sha = jsonl_digest.hexdigest()
        _call_fault(fault, "before_frames_jsonl_replace")
        os.replace(jsonl_temporary, jsonl_path)
        _call_fault(fault, "after_frames_jsonl_replace")
        _sync_parent(jsonl_path.parent)
        _call_fault(fault, "after_frames_jsonl_directory_sync")
    finally:
        jsonl_temporary.unlink(missing_ok=True)

    def replay_frame_stats():
        count = 0
        with jsonl_path.open("rb", buffering=0) as handle:
            for line in handle:
                if not line.endswith(b"\n"):
                    raise ValueError("diagnostics JSONL row is not LF-terminated")
                count += 1
                yield _parse_canonical_object(line[:-1])
        if count != len(context.frame_names):
            raise ValueError("diagnostics JSONL row count differs from the frame sequence")

    summary = build_diagnostics_summary(
        replay_frame_stats(),
        frame_names=context.frame_names,
        render_shape=context.render_shape,
        render_mode=context.render_mode,
        geometry_mode=context.geometry_mode,
        occlusion_fill=context.occlusion_fill,
        ordered_frame_manifest_fingerprint=ordered_fingerprint,
        frames_jsonl_sha256=jsonl_sha,
    )
    summary_path = root / "stereo_coverage_summary.json"
    summary_raw = canonical_json_bytes(summary)
    _atomic_replace_raw(summary_path, summary_raw, fault=fault, label="root_summary")
    clamp_sha = None
    if context.geometry_mode == "metric":
        assert context.left_rgb_parent is not None
        clamp_path = context.left_rgb_parent / "clamp_summary.json"
        clamp_raw = canonical_json_bytes(_compatibility_clamp_summary(summary))
        _atomic_replace_raw(
            clamp_path,
            clamp_raw,
            fault=fault,
            label="metric_clamp_summary",
        )
        clamp_sha = hashlib.sha256(clamp_raw).hexdigest()
    metadata = _diagnostics_metadata(
        context,
        status="complete",
        ordered_frame_manifest_fingerprint=ordered_fingerprint,
        frames_jsonl_sha256=jsonl_sha,
        summary_sha256=hashlib.sha256(summary_raw).hexdigest(),
        metric_clamp_summary_sha256=clamp_sha,
    )
    _atomic_replace_raw(
        root / "metadata.json",
        canonical_json_bytes(metadata),
        fault=fault,
        label="complete_metadata",
    )
    return metadata


def migrate_frame_manifests_sync(
    context: DiagnosticsStageContext,
    frame_names: Iterable[str] | Callable[[], Iterable[str]],
    *,
    fault: FaultHook | None = None,
) -> None:
    """Commit mask-disabled manifests before deleting unreferenced masks."""

    if context.mask_payloads_enabled:
        raise ValueError("mask manifest migration requires a disabled target policy")
    replay = frame_names if callable(frame_names) else lambda: iter(frame_names)
    previous = replace(context, mask_payloads_enabled=True)
    for source_index, frame_name in enumerate(replay()):
        manifest, _raw, _stats = _audit_frame_transaction(previous, frame_name)
        payloads = dict(manifest["payloads"])
        for name in (
            "left_coverage",
            "left_repair",
            "right_coverage",
            "right_repair",
        ):
            payloads[name] = None
        migrated = dict(manifest)
        migrated["mask_payloads_enabled"] = False
        migrated["payloads"] = payloads
        migrated["fingerprint"] = self_fingerprint(migrated)
        manifest_path = (
            context.job_root / "04_stereo_diagnostics" / "frames" / frame_name / "manifest.json"
        )
        _call_fault(fault, f"before_mask_migration_{source_index}_temporary_write")
        _atomic_replace_raw(
            manifest_path,
            canonical_json_bytes(migrated),
            fault=fault,
            label=f"mask_migration_{source_index}",
        )
        _audit_frame_transaction(context, frame_name)
    for frame_name in replay():
        frame_dir = context.job_root / "04_stereo_diagnostics" / "frames" / frame_name
        for name in (
            "left_coverage.png",
            "left_repair.png",
            "right_coverage.png",
            "right_repair.png",
        ):
            (frame_dir / name).unlink(missing_ok=True)


def _metadata_matches_context(
    metadata: dict[str, Any],
    context: DiagnosticsStageContext,
) -> bool:
    expected = _diagnostics_metadata(
        context,
        status="building",
    )
    identity_keys = {
        "schema_version",
        "algorithm_version",
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
    }
    return (
        set(metadata) == set(expected)
        and all(metadata.get(key) == expected[key] for key in identity_keys)
        and metadata.get("fingerprint") == self_fingerprint(metadata)
    )


def _aggregates_match(
    metadata: dict[str, Any],
    context: DiagnosticsStageContext,
) -> bool:
    root = context.job_root / "04_stereo_diagnostics"
    try:
        jsonl_raw = (root / "stereo_coverage_frames.jsonl").read_bytes()
        summary, summary_raw = _read_canonical_object(root / "stereo_coverage_summary.json")
        if hashlib.sha256(jsonl_raw).hexdigest() != metadata["frames_jsonl_sha256"]:
            return False
        if hashlib.sha256(summary_raw).hexdigest() != metadata["summary_sha256"]:
            return False
        if summary.get("frames_jsonl_sha256") != metadata["frames_jsonl_sha256"]:
            return False
        if (
            summary.get("ordered_frame_manifest_fingerprint")
            != metadata["ordered_frame_manifest_fingerprint"]
        ):
            return False
        if context.geometry_mode == "metric":
            assert context.left_rgb_parent is not None
            clamp_raw = (context.left_rgb_parent / "clamp_summary.json").read_bytes()
            if hashlib.sha256(clamp_raw).hexdigest() != metadata["metric_clamp_summary_sha256"]:
                return False
        elif metadata["metric_clamp_summary_sha256"] is not None:
            return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def _payload_pruned_matches(  # noqa: C901
    metadata: dict[str, Any],
    context: DiagnosticsStageContext,
) -> bool:
    if (
        metadata.get("status") != "payload_pruned"
        or metadata.get("mask_payloads_enabled") is not False
        or metadata.get("ordered_frame_manifest_fingerprint") is not None
        or metadata.get("metric_clamp_summary_sha256") is not None
        or not _metadata_matches_context(metadata, context)
        or not _is_hex64(metadata.get("frames_jsonl_sha256"))
        or not _is_hex64(metadata.get("summary_sha256"))
        or not _is_hex64(metadata.get("final_video_manifest_sha256"))
        or not isinstance(metadata.get("prune_entries"), list)
    ):
        return False
    pruned_from = metadata.get("pruned_from")
    if not isinstance(pruned_from, dict) or set(pruned_from) != {
        "previous_status",
        "previous_diagnostics_fingerprint",
        "ordered_frame_manifest_fingerprint",
        "frames_jsonl_sha256",
        "summary_sha256",
        "metric_clamp_summary_sha256",
    }:
        return False
    if (
        pruned_from["previous_status"] not in {"complete", "legacy_fast_unavailable"}
        or not _is_hex64(pruned_from["previous_diagnostics_fingerprint"])
        or not _is_hex64(pruned_from["ordered_frame_manifest_fingerprint"])
        or pruned_from["frames_jsonl_sha256"] != metadata["frames_jsonl_sha256"]
        or pruned_from["summary_sha256"] != metadata["summary_sha256"]
    ):
        return False
    root = context.job_root / "04_stereo_diagnostics"
    try:
        jsonl_raw = (root / "stereo_coverage_frames.jsonl").read_bytes()
        summary, summary_raw = _read_canonical_object(root / "stereo_coverage_summary.json")
        final_manifest_raw = (context.job_root / "final_video_manifest.json").read_bytes()
        if hashlib.sha256(jsonl_raw).hexdigest() != metadata["frames_jsonl_sha256"]:
            return False
        if hashlib.sha256(summary_raw).hexdigest() != metadata["summary_sha256"]:
            return False
        if summary.get("frames_jsonl_sha256") != metadata["frames_jsonl_sha256"]:
            return False
        if (
            summary.get("ordered_frame_manifest_fingerprint")
            != pruned_from["ordered_frame_manifest_fingerprint"]
        ):
            return False
        if (
            hashlib.sha256(final_manifest_raw).hexdigest()
            != metadata["final_video_manifest_sha256"]
        ):
            return False
        _parse_canonical_object(final_manifest_raw)
        if context.render_mode == "quality":
            content_raw = (root / "quality_input_manifest.json").read_bytes()
            if hashlib.sha256(content_raw).hexdigest() != context.quality_input_manifest_sha256:
                return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def audit_diagnostics_stage(  # noqa: C901
    context: DiagnosticsStageContext,
) -> DiagnosticsAudit:
    """Read-only disposition for frame reuse and aggregate rebuilding."""

    root = context.job_root / "04_stereo_diagnostics"
    metadata_path = root / "metadata.json"
    if not root.exists():
        return DiagnosticsAudit("absent", (), context.frame_names, False)
    try:
        metadata, _raw = _read_canonical_object(metadata_path)
    except (OSError, ValueError):
        metadata = {}
    if metadata.get("status") == "payload_pruned":
        historical_valid = _payload_pruned_matches(metadata, context)
        return DiagnosticsAudit(
            "payload_pruned",
            (),
            context.frame_names,
            historical_valid,
        )
    reusable: list[str] = []
    invalid: list[str] = []
    migrate: list[str] = []
    previous_mask_context = (
        replace(context, mask_payloads_enabled=True) if not context.mask_payloads_enabled else None
    )
    for frame_name in context.frame_names:
        try:
            _audit_frame_transaction(context, frame_name)
        except (OSError, ValueError, KeyError, TypeError):
            if previous_mask_context is not None:
                try:
                    _audit_frame_transaction(previous_mask_context, frame_name)
                except (OSError, ValueError, KeyError, TypeError):
                    invalid.append(frame_name)
                else:
                    migrate.append(frame_name)
            else:
                invalid.append(frame_name)
        else:
            reusable.append(frame_name)
    if invalid or migrate or not _metadata_matches_context(metadata, context):
        return DiagnosticsAudit(
            "building",
            tuple(reusable),
            tuple(invalid),
            False,
            tuple(migrate),
        )
    aggregates_valid = metadata.get("status") == "complete" and _aggregates_match(metadata, context)
    return DiagnosticsAudit(
        "complete" if aggregates_valid else "building",
        tuple(reusable),
        (),
        aggregates_valid,
    )
