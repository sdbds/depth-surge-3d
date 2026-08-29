"""Atomic RGB and compact-diagnostics publication for one Quality frame."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import threading
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator, Literal

import cv2
import numpy as np

from ...io.canonical_json import CanonicalJsonError, strict_canonical_json_loads
from ...io.durable_publish import OwnedSiblingTemporary, publication_temporary_path
from ...io.nofollow_io import (
    AcquiredRoot,
    NoFollowPathError,
    SafeTreeMutation,
    open_regular_beneath,
)
from ...io.streamed_canonical_json import (
    ReplayableJsonArray,
    StreamedCanonicalObject,
    StreamedCanonicalValue,
    build_streamed_canonical_object,
    build_streamed_canonical_value,
)
from ...io.stereo_contract import STEREO_CONTROL_FRAME_CAP, StereoControlBudgetError
from ...io.reservation_protocol import _sync_parent
from ...utils.imaging.png_header import png_header_bytes_match
from .diagnostics import (
    build_diagnostics_summary,
    build_quality_frame_stats,
    checked_u64_add,
    checked_u64_mul,
    validate_diagnostics_cardinality_dimensions,
)
from .geometry import QUALITY_GEOMETRY_NEAREST_VISIT_CAP
from .local_strip import (
    QUALITY_EXEMPLAR_EYE_EVALUATION_CAP,
    QUALITY_LOCAL_NEIGHBOR_SAMPLE_CAP,
    QUALITY_LOCAL_SLOT_CAP,
    QUALITY_REPAIR_FALLBACK_VISIT_CAP,
)
from .types import CompactStereoRenderResult, DenseHistogramPairs


FaultHook = Callable[[str], None]
_HEX_64 = frozenset("0123456789abcdef")
_U64_MAX = 2**64 - 1
_QUALITY_RECORD_ARENA_BYTES = 64 * 1024 * 1024


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
        if isinstance(current, _SparseHistogramPairs):
            output.write(b"[")
            for index, (bucket, count) in enumerate(current):
                if index:
                    output.write(b",")
                output.write(f"[{bucket},{count}]".encode("ascii"))
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
                json.dumps(float(current), allow_nan=False, separators=(",", ":")).encode("ascii")
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


class SharedQualityPngPayloads:
    """Lazily encode one immutable PNG payload per compact raster."""

    __slots__ = ("_condition", "_errors", "_payloads", "_sources", "_states")

    _PENDING = 0
    _ENCODING = 1
    _READY = 2
    _FAILED = 3

    def __init__(
        self,
        *,
        compact: CompactStereoRenderResult,
        keep_masks: bool,
    ) -> None:
        if not isinstance(compact, CompactStereoRenderResult):
            raise TypeError("shared PNG payloads require a compact render result")
        if not isinstance(keep_masks, bool):
            raise TypeError("shared PNG mask selection must be boolean")
        self._sources: dict[str, np.ndarray | None] = {
            "left_rgb": compact.left_image,
            "right_rgb": compact.right_image,
            "left_coverage": compact.left_coverage_count if keep_masks else None,
            "left_repair": compact.left_repair_bits if keep_masks else None,
            "right_coverage": compact.right_coverage_count if keep_masks else None,
            "right_repair": compact.right_repair_bits if keep_masks else None,
        }
        self._states = {name: self._PENDING for name in self._sources}
        self._payloads: dict[str, np.ndarray] = {}
        self._errors: dict[str, BaseException] = {}
        self._condition = threading.Condition()

    def matches(self, compact: CompactStereoRenderResult, *, keep_masks: bool) -> bool:
        expected = (
            compact.left_image,
            compact.right_image,
            compact.left_coverage_count if keep_masks else None,
            compact.left_repair_bits if keep_masks else None,
            compact.right_coverage_count if keep_masks else None,
            compact.right_repair_bits if keep_masks else None,
        )
        return all(
            source is candidate
            for source, candidate in zip(self._sources.values(), expected, strict=True)
        )

    def payload(self, name: str) -> np.ndarray | None:
        try:
            source = self._sources[name]
        except KeyError as error:
            raise ValueError(f"unknown shared PNG payload: {name}") from error
        if source is None:
            return None

        with self._condition:
            while True:
                state = self._states[name]
                if state == self._READY:
                    return self._payloads[name]
                if state == self._FAILED:
                    raise OSError(f"shared PNG encoding failed for {name}") from self._errors[name]
                if state == self._PENDING:
                    self._states[name] = self._ENCODING
                    break
                self._condition.wait()

        try:
            payload = _png_buffer(source)
            payload.setflags(write=False)
        except BaseException as error:
            with self._condition:
                self._errors[name] = error
                self._states[name] = self._FAILED
                self._condition.notify_all()
            raise

        with self._condition:
            self._payloads[name] = payload
            self._states[name] = self._READY
            self._condition.notify_all()
        return payload


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
    png_payloads: SharedQualityPngPayloads | None = None


def prepare_quality_frame(
    publication: QualityFramePublication,
    *,
    png_payloads: SharedQualityPngPayloads | None = None,
) -> PreparedQualityFramePublication:
    """Serialize variable diagnostics before crossing the writer queue boundary."""

    if not isinstance(publication, QualityFramePublication):
        raise TypeError("publication must be QualityFramePublication")
    compact = publication.compact
    if png_payloads is not None:
        if not isinstance(png_payloads, SharedQualityPngPayloads):
            raise TypeError("png_payloads must be SharedQualityPngPayloads")
        if not png_payloads.matches(compact, keep_masks=publication.keep_masks):
            raise ValueError("shared PNG payloads do not match the compact render result")
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
        png_payloads=png_payloads,
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
    native_geometry_sample_count: int | None = None
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
        validate_diagnostics_cardinality_dimensions(
            frame_count=len(self.frame_names),
            height=self.render_shape[0],
            width=self.render_shape[1],
        )
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
            if (
                type(self.native_geometry_sample_count) is not int
                or self.native_geometry_sample_count <= 0
            ):
                raise ValueError("Quality diagnostics require native geometry cardinality")
            checked_u64_add(
                "native_geometry_sample_count",
                self.native_geometry_sample_count,
            )
        elif self.quality_input_manifest_sha256 is not None:
            raise ValueError("Fast diagnostics cannot bind a Quality input manifest")
        elif self.native_geometry_sample_count is not None:
            raise ValueError("Fast diagnostics cannot bind Quality geometry cardinality")
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


@dataclass(frozen=True, slots=True)
class StreamingDiagnosticsStageContext:
    """Scalar diagnostics identity with a replayable source-order name stream."""

    job_root: Path
    frame_count: int
    frame_names: Callable[[], Iterable[str]]
    render_shape: tuple[int, int]
    render_mode: Literal["fast", "quality"]
    geometry_mode: Literal["relative", "metric"]
    occlusion_fill: Literal["none", "background"]
    mask_payloads_enabled: bool
    rgb_stage_fingerprint: str
    source_guide_fingerprint: str
    native_geometry_fingerprint: str | None
    quality_input_manifest_sha256: str | None
    native_geometry_sample_count: int | None = None
    left_rgb_parent: Path | None = None
    right_rgb_parent: Path | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.frame_count, bool)
            or not isinstance(self.frame_count, int)
            or self.frame_count <= 0
        ):
            raise ValueError("streaming diagnostics frame count must be positive")
        if not callable(self.frame_names):
            raise TypeError("streaming diagnostics names must be replayable")
        if len(self.render_shape) != 2:
            raise ValueError("diagnostics render_shape must contain height and width")
        validate_diagnostics_cardinality_dimensions(
            frame_count=self.frame_count,
            height=self.render_shape[0],
            width=self.render_shape[1],
        )
        probe = DiagnosticsStageContext(
            job_root=self.job_root,
            frame_names=("frame_000000",),
            render_shape=self.render_shape,
            render_mode=self.render_mode,
            geometry_mode=self.geometry_mode,
            occlusion_fill=self.occlusion_fill,
            mask_payloads_enabled=self.mask_payloads_enabled,
            rgb_stage_fingerprint=self.rgb_stage_fingerprint,
            source_guide_fingerprint=self.source_guide_fingerprint,
            native_geometry_fingerprint=self.native_geometry_fingerprint,
            quality_input_manifest_sha256=self.quality_input_manifest_sha256,
            native_geometry_sample_count=self.native_geometry_sample_count,
            left_rgb_parent=self.left_rgb_parent,
            right_rgb_parent=self.right_rgb_parent,
        )
        object.__setattr__(self, "job_root", probe.job_root)
        object.__setattr__(self, "render_shape", probe.render_shape)
        object.__setattr__(self, "left_rgb_parent", probe.left_rgb_parent)
        object.__setattr__(self, "right_rgb_parent", probe.right_rgb_parent)

    def one_frame(self, frame_name: str, *, masks: bool | None = None) -> DiagnosticsStageContext:
        return DiagnosticsStageContext(
            job_root=self.job_root,
            frame_names=(frame_name,),
            render_shape=self.render_shape,
            render_mode=self.render_mode,
            geometry_mode=self.geometry_mode,
            occlusion_fill=self.occlusion_fill,
            mask_payloads_enabled=(self.mask_payloads_enabled if masks is None else masks),
            rgb_stage_fingerprint=self.rgb_stage_fingerprint,
            source_guide_fingerprint=self.source_guide_fingerprint,
            native_geometry_fingerprint=self.native_geometry_fingerprint,
            quality_input_manifest_sha256=self.quality_input_manifest_sha256,
            native_geometry_sample_count=self.native_geometry_sample_count,
            left_rgb_parent=self.left_rgb_parent,
            right_rgb_parent=self.right_rgb_parent,
        )


@dataclass(frozen=True, slots=True)
class StreamingDiagnosticsCompletion:
    metadata: StreamedCanonicalObject
    summary: StreamedCanonicalValue
    metric_clamp_summary: StreamedCanonicalValue | None
    ordered_frame_manifest_fingerprint: str
    frames_jsonl_sha256: str


def _is_hex64(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_64 for character in value)
    )


def _png_buffer(values: np.ndarray) -> np.ndarray:
    encoded, payload = cv2.imencode(".png", values)
    if not encoded:
        raise OSError("could not encode diagnostics PNG")
    return payload


def _identity(root: Path, path: Path, raw: bytes | np.ndarray) -> dict[str, Any]:
    view = memoryview(raw).cast("B")
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(view).hexdigest(),
        "byte_count": view.nbytes,
    }


def _write_temporary(
    path: Path,
    raw: bytes | np.ndarray,
    *,
    fault: FaultHook | None = None,
    label: str | None = None,
) -> OwnedSiblingTemporary:
    path.parent.mkdir(parents=True, exist_ok=True)
    owned = OwnedSiblingTemporary.create(path)
    try:
        view = memoryview(raw).cast("B")
        if label is not None:
            _call_fault(fault, f"before_{label}_temporary_write")
        offset = 0
        while offset < view.nbytes:
            count = owned.handle.write(view[offset:])
            if not count:
                raise OSError("short diagnostics payload write")
            offset += count
        if label is not None:
            _call_fault(fault, f"after_{label}_temporary_write")
        owned.sync()
        if label is not None:
            _call_fault(fault, f"after_{label}_temporary_sync")
        return owned
    except Exception:
        owned.cleanup()
        raise


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
    payload_sources: tuple[tuple[str, Path, np.ndarray | bytes | None, bool], ...] = (
        ("left_rgb", publication.left_rgb_path, publication.left_image, True),
        ("right_rgb", publication.right_rgb_path, publication.right_image, True),
        (
            "left_coverage",
            mask_paths["left_coverage"],
            publication.left_coverage_count if publication.keep_masks else None,
            True,
        ),
        (
            "left_repair",
            mask_paths["left_repair"],
            publication.left_repair_bits if publication.keep_masks else None,
            True,
        ),
        (
            "right_coverage",
            mask_paths["right_coverage"],
            publication.right_coverage_count if publication.keep_masks else None,
            True,
        ),
        (
            "right_repair",
            mask_paths["right_repair"],
            publication.right_repair_bits if publication.keep_masks else None,
            True,
        ),
        ("stats", stats_path, publication.stats_raw, False),
    )
    payload_identities: dict[str, dict[str, Any] | None] = {}
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
    temporaries: dict[Path, OwnedSiblingTemporary] = {}
    try:
        for name, path, source, encode_png in payload_sources:
            if source is None:
                payload_identities[name] = None
                continue
            if encode_png and publication.png_payloads is not None:
                raw = publication.png_payloads.payload(name)
                if raw is None:
                    raise RuntimeError("enabled PNG payload is absent from its shared cache")
            else:
                raw = _png_buffer(source) if encode_png else source
            payload_identities[name] = _identity(root, path, raw)
            temporaries[path] = _write_temporary(
                path,
                raw,
                fault=fault,
                label=name,
            )
            del raw
        manifest["payloads"] = payload_identities
        manifest["fingerprint"] = self_fingerprint(manifest)
        raw_manifest = canonical_json_bytes(manifest)
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
            identity = payload_identities[name]
            if identity is None:
                continue
            path = next(
                path for source_name, path, _source, _png in payload_sources if source_name == name
            )
            temporaries[path].replace(
                after_replace=lambda name=name: _call_fault(fault, f"after_{name}_replace"),
            )
            _call_fault(fault, f"after_{name}_directory_sync")
        _call_fault(fault, "before_manifest_replace")
        temporaries[manifest_path].replace(
            after_replace=lambda: _call_fault(fault, "after_manifest_replace"),
        )
        _call_fault(fault, "after_manifest_directory_sync")
        return manifest
    except Exception:
        for owned in reversed(tuple(temporaries.values())):
            owned.remove_published_target()
        raise
    finally:
        for owned in temporaries.values():
            owned.cleanup()


def _atomic_replace_raw(
    path: Path,
    raw: bytes,
    *,
    fault: FaultHook | None = None,
    label: str,
) -> None:
    owned = _write_temporary(path, raw, fault=fault, label=label)
    try:
        _call_fault(fault, f"before_{label}_replace")
        owned.replace(
            after_replace=lambda: _call_fault(fault, f"after_{label}_replace"),
        )
        _call_fault(fault, f"after_{label}_directory_sync")
    finally:
        owned.cleanup()


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


def _parse_canonical_small_object(raw: bytes) -> dict[str, Any]:
    """Decode canonical JSON after variable histogram arrays were removed."""

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


def _parse_canonical_uint(raw: bytearray | bytes, cursor: int) -> tuple[int, int]:
    start = cursor
    length = len(raw)
    while cursor < length and 48 <= raw[cursor] <= 57:
        cursor += 1
    if cursor == start:
        raise ValueError("diagnostic histogram integer is missing")
    token = raw[start:cursor]
    if len(token) > 1 and token[0] == 48:
        raise ValueError("diagnostic histogram integer is noncanonical")
    value = int(token)
    if value > np.iinfo(np.uint64).max:
        raise ValueError("diagnostic histogram integer exceeds uint64")
    return value, cursor


@dataclass(frozen=True, slots=True)
class _SparseHistogramPairs:
    """Replay validated pairs directly from the one resident frame JSON buffer."""

    raw: bytearray | bytes
    start: int
    end: int

    def __iter__(self):
        cursor = self.start + 1
        if cursor < self.end and self.raw[cursor] == 93:  # ]
            return
        while cursor < self.end:
            bucket, cursor = _parse_canonical_uint(self.raw, cursor + 1)
            cursor += 1  # comma
            count, cursor = _parse_canonical_uint(self.raw, cursor)
            yield bucket, count
            cursor += 1  # pair close
            if cursor < self.end and self.raw[cursor] == 93:  # outer close
                return
            if cursor < self.end and self.raw[cursor] == 44:  # comma
                cursor += 1


def _parse_histogram_span(
    raw: bytearray | bytes,
    start: int,
    *,
    bin_count: int,
) -> tuple[_SparseHistogramPairs, int]:
    if raw[start] != 91:  # [
        raise ValueError("diagnostic histogram array is malformed")
    cursor = start + 1
    previous_bucket = -1
    if cursor < len(raw) and raw[cursor] == 93:  # ]
        return _SparseHistogramPairs(raw, start, cursor + 1), cursor + 1
    while True:
        if cursor >= len(raw) or raw[cursor] != 91:
            raise ValueError("diagnostic histogram pair is malformed")
        bucket, cursor = _parse_canonical_uint(raw, cursor + 1)
        if cursor >= len(raw) or raw[cursor] != 44:  # ,
            raise ValueError("diagnostic histogram pair is malformed")
        count, cursor = _parse_canonical_uint(raw, cursor + 1)
        if cursor >= len(raw) or raw[cursor] != 93:  # ]
            raise ValueError("diagnostic histogram pair is malformed")
        if bucket <= previous_bucket or bucket >= bin_count or count == 0:
            raise ValueError("diagnostic histogram pairs are not canonical sparse bins")
        previous_bucket = bucket
        cursor += 1
        if cursor >= len(raw):
            raise ValueError("diagnostic histogram array is unterminated")
        if raw[cursor] == 93:  # ]
            return _SparseHistogramPairs(raw, start, cursor + 1), cursor + 1
        if raw[cursor] != 44:  # ,
            raise ValueError("diagnostic histogram array is malformed")
        cursor += 1


def _parse_streaming_frame_stats(
    raw: bytearray | bytes,
    *,
    render_width: int,
    raw_end: int | None = None,
) -> dict[str, Any]:
    """Parse frame stats without materializing variable `[bucket,count]` lists."""

    if raw_end is None:
        raw_end = len(raw)
    if raw_end <= 0 or raw_end > len(raw):
        raise ValueError("diagnostic frame statistics byte range is invalid")

    markers = (
        (b'"lane_count_histogram":', 16 * (render_width + 1)),
        (b'"touched_pixel_span_histogram":', render_width + 1),
        (b'"lane_count_histogram":', 16 * (render_width + 1)),
        (b'"touched_pixel_span_histogram":', render_width + 1),
    )
    spans: list[tuple[int, int, _SparseHistogramPairs]] = []
    search_from = 0
    for marker, bin_count in markers:
        marker_at = raw.find(marker, search_from)
        if marker_at < 0:
            raise ValueError("diagnostic frame statistics omit a histogram")
        array_start = marker_at + len(marker)
        histogram, array_end = _parse_histogram_span(
            raw,
            array_start,
            bin_count=bin_count,
        )
        spans.append((array_start, array_end, histogram))
        search_from = array_end
    if (
        raw.find(b'"lane_count_histogram":', search_from) >= 0
        or raw.find(
            b'"touched_pixel_span_histogram":',
            search_from,
        )
        >= 0
    ):
        raise ValueError("diagnostic frame statistics contain extra histograms")

    compact_parts: list[bytes] = []
    cursor = 0
    for start, end, _histogram in spans:
        compact_parts.append(bytes(raw[cursor:start]))
        compact_parts.append(b"[]")
        cursor = end
    compact_parts.append(bytes(raw[cursor:raw_end]))
    compact_raw = b"".join(compact_parts)
    value = _parse_canonical_small_object(compact_raw)
    try:
        holes = (
            value["eyes"]["left"]["hole_runs"],
            value["eyes"]["right"]["hole_runs"],
        )
        fields = (
            (holes[0], "lane_count_histogram"),
            (holes[0], "touched_pixel_span_histogram"),
            (holes[1], "lane_count_histogram"),
            (holes[1], "touched_pixel_span_histogram"),
        )
        for (owner, field), (_start, _end, histogram) in zip(
            fields,
            spans,
            strict=True,
        ):
            if owner[field] != []:
                raise ValueError("diagnostic histogram placeholder is not empty")
            owner[field] = histogram
    except (KeyError, TypeError) as error:
        raise ValueError("diagnostic frame histogram structure is malformed") from error
    return value


_STATE_FIELDS = (
    "prefill_partial",
    "prefill_full",
    "local_filled",
    "post_local_residual",
    "exemplar_filled",
    "fallback_filled",
    "final_unresolved",
)
_BACKEND_FIELDS = ("local", "exemplar", "fallback")
_HOLE_RUN_FIELDS = {
    "count",
    "lane_count_histogram",
    "touched_pixel_span_histogram",
    "lane_count_max",
    "lane_count_p95",
    "touched_pixel_span_max",
    "touched_pixel_span_p95",
    "physical_width_px_max",
    "physical_width_px_p95",
}
_QUALITY_LIMIT_FIELDS = {
    "max_neighbour_abs_q_jump_px",
    "predicted_gap_px",
    "local_limit_px",
}
_QUALITY_BUDGET_FIELDS = {
    "availability",
    "segment_record_count",
    "segment_table_bytes",
    "local_slot_charge",
    "local_neighborhood_sample_charge",
    "local_budget_skipped_run_count",
    "exemplar_evaluations",
    "fallback_indexed_donor_count",
    "fallback_query_count",
    "fallback_visited_nodes_total",
    "fallback_visited_nodes_max",
    "fallback_visited_nodes_p95",
}


def _exact_object(value: object, keys: set[str], field: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{field} has the wrong object schema")
    return value


def _json_u64(value: object, field: str, *, maximum: int = _U64_MAX) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be a JSON integer")
    normalized = checked_u64_add(field, value)
    if normalized > maximum:
        raise ValueError(f"{field} exceeds its cardinality bound")
    return normalized


def _json_binary64(
    value: object,
    field: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if (
        type(value) is not float
        or not math.isfinite(value)
        or value == 0.0
        and math.copysign(1.0, value) < 0.0
    ):
        raise ValueError(f"{field} must be a finite JSON binary64")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{field} exceeds its numeric bound")
    return value


def _sparse_histogram_metrics(
    pairs: object,
    *,
    field: str,
    maximum_bucket: int,
    maximum_count: int,
) -> tuple[int, int, float]:
    if not isinstance(pairs, _SparseHistogramPairs):
        raise ValueError(f"{field} must be a streamed sparse histogram")
    total = 0
    maximum = 0
    for bucket, count in pairs:
        bucket_value = _json_u64(bucket, f"{field}.bucket", maximum=maximum_bucket)
        count_value = _json_u64(count, f"{field}.count", maximum=maximum_count)
        if bucket_value == 0 or count_value == 0:
            raise ValueError(f"{field} requires positive sparse pairs")
        total = checked_u64_add(field, total, count_value)
        if total > maximum_count:
            raise ValueError(f"{field} exceeds its total-count bound")
        maximum = bucket_value
    if total == 0:
        return 0, 0, 0.0
    position = 0.95 * float(total - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    lower: int | None = None
    upper: int | None = None
    seen = 0
    for bucket, count in pairs:
        next_seen = checked_u64_add(field, seen, count)
        if lower is None and lower_index < next_seen:
            lower = bucket
        if upper_index < next_seen:
            upper = bucket
            break
        seen = next_seen
    if lower is None or upper is None:
        raise ValueError(f"{field} percentile lookup exceeded its total")
    percentile = float(lower) + (float(upper) - float(lower)) * (position - float(lower_index))
    return total, maximum, percentile


def _validate_quality_limits(value: object, field: str) -> dict[str, Any]:
    limits = _exact_object(value, _QUALITY_LIMIT_FIELDS, field)
    _json_binary64(limits["max_neighbour_abs_q_jump_px"], f"{field}.max_jump")
    _json_u64(limits["predicted_gap_px"], f"{field}.predicted_gap_px")
    _json_u64(limits["local_limit_px"], f"{field}.local_limit_px")
    return limits


def _validate_eye_statistics(
    eye: object,
    *,
    field: str,
    context: DiagnosticsStageContext,
    pixel_count: int,
    lane_count: int,
    record_cap: int,
) -> dict[str, Any]:
    value = _exact_object(
        eye,
        {
            "pixel_count",
            "state_pixel_counts",
            "state_pixel_ratios",
            "hole_runs",
            "backend_lane_counts",
            "backend_pixel_counts",
            "backend_component_counts",
            "final_unresolved_lane_count",
            "quality_limits",
            "quality_budgets",
        },
        field,
    )
    if _json_u64(value["pixel_count"], f"{field}.pixel_count", maximum=pixel_count) != pixel_count:
        raise ValueError(f"{field}.pixel_count does not equal render pixels")
    counts = _exact_object(
        value["state_pixel_counts"], set(_STATE_FIELDS), f"{field}.state_pixel_counts"
    )
    ratios = _exact_object(
        value["state_pixel_ratios"], set(_STATE_FIELDS), f"{field}.state_pixel_ratios"
    )
    for name in _STATE_FIELDS:
        count = _json_u64(counts[name], f"{field}.state_pixel_counts.{name}", maximum=pixel_count)
        ratio = _json_binary64(
            ratios[name],
            f"{field}.state_pixel_ratios.{name}",
            maximum=1.0,
        )
        if ratio != float(count) / float(pixel_count):
            raise ValueError(f"{field}.state_pixel_ratios.{name} is not derived from its count")

    holes = _exact_object(value["hole_runs"], _HOLE_RUN_FIELDS, f"{field}.hole_runs")
    lane_total, lane_max, lane_p95 = _sparse_histogram_metrics(
        holes["lane_count_histogram"],
        field=f"{field}.hole_runs.lane_count_histogram",
        maximum_bucket=16 * context.render_shape[1],
        maximum_count=lane_count,
    )
    span_total, span_max, span_p95 = _sparse_histogram_metrics(
        holes["touched_pixel_span_histogram"],
        field=f"{field}.hole_runs.touched_pixel_span_histogram",
        maximum_bucket=context.render_shape[1],
        maximum_count=lane_count,
    )
    if (
        lane_total != span_total
        or _json_u64(holes["count"], f"{field}.hole_runs.count", maximum=lane_count) != lane_total
    ):
        raise ValueError(f"{field}.hole_runs histogram totals disagree")
    integer_derived = {
        "lane_count_max": lane_max,
        "touched_pixel_span_max": span_max,
    }
    float_derived = {
        "lane_count_p95": lane_p95,
        "touched_pixel_span_p95": span_p95,
        "physical_width_px_max": float(lane_max) / 16.0,
        "physical_width_px_p95": lane_p95 / 16.0,
    }
    for name, expected in integer_derived.items():
        if _json_u64(holes[name], f"{field}.hole_runs.{name}") != expected:
            raise ValueError(f"{field}.hole_runs.{name} is not histogram-derived")
    for name, expected in float_derived.items():
        if _json_binary64(holes[name], f"{field}.hole_runs.{name}") != expected:
            raise ValueError(f"{field}.hole_runs.{name} is not histogram-derived")

    for collection, maximum in (
        ("backend_lane_counts", lane_count),
        ("backend_pixel_counts", pixel_count),
    ):
        backend_counts = _exact_object(
            value[collection], set(_BACKEND_FIELDS), f"{field}.{collection}"
        )
        for name in _BACKEND_FIELDS:
            _json_u64(backend_counts[name], f"{field}.{collection}.{name}", maximum=maximum)

    components = _exact_object(
        value["backend_component_counts"],
        {"availability", *_BACKEND_FIELDS},
        f"{field}.backend_component_counts",
    )
    expected_component_availability = (
        "quality_segment_graph"
        if context.render_mode == "quality" and context.occlusion_fill == "background"
        else (
            "unavailable_fast_no_segment_graph"
            if context.render_mode == "fast"
            else "unavailable_quality_none_no_repair_graph"
        )
    )
    if components["availability"] != expected_component_availability:
        raise ValueError(f"{field}.backend_component_counts availability is invalid")
    for name in _BACKEND_FIELDS:
        component_count = components[name]
        if expected_component_availability == "quality_segment_graph":
            _json_u64(
                component_count,
                f"{field}.backend_component_counts.{name}",
                maximum=record_cap,
            )
        elif component_count is not None:
            raise ValueError(f"{field}.backend_component_counts.{name} must be null")

    _json_u64(
        value["final_unresolved_lane_count"],
        f"{field}.final_unresolved_lane_count",
        maximum=lane_count,
    )
    requires_quality_plan = (
        context.render_mode == "quality" and context.occlusion_fill == "background"
    )
    if requires_quality_plan:
        _validate_quality_limits(value["quality_limits"], f"{field}.quality_limits")
    elif value["quality_limits"] is not None:
        raise ValueError(f"{field}.quality_limits must be null")

    budgets = value["quality_budgets"]
    if context.render_mode == "fast":
        if budgets is not None:
            raise ValueError(f"{field}.quality_budgets must be null")
        return value
    budget_values = _exact_object(budgets, _QUALITY_BUDGET_FIELDS, f"{field}.quality_budgets")
    expected_budget_availability = (
        "quality_repair_plan"
        if context.occlusion_fill == "background"
        else "unavailable_quality_none_no_repair_plan"
    )
    if budget_values["availability"] != expected_budget_availability:
        raise ValueError(f"{field}.quality_budgets availability is invalid")
    numeric_bounds = {
        "segment_record_count": record_cap,
        "segment_table_bytes": 16 * record_cap,
        "local_slot_charge": QUALITY_LOCAL_SLOT_CAP,
        "local_neighborhood_sample_charge": QUALITY_LOCAL_NEIGHBOR_SAMPLE_CAP,
        "local_budget_skipped_run_count": record_cap,
        "exemplar_evaluations": QUALITY_EXEMPLAR_EYE_EVALUATION_CAP,
        "fallback_indexed_donor_count": pixel_count,
        "fallback_query_count": record_cap,
        "fallback_visited_nodes_total": QUALITY_REPAIR_FALLBACK_VISIT_CAP,
        "fallback_visited_nodes_max": pixel_count,
    }
    for name, maximum in numeric_bounds.items():
        current = budget_values[name]
        if expected_budget_availability == "quality_repair_plan":
            _json_u64(current, f"{field}.quality_budgets.{name}", maximum=maximum)
        elif current is not None:
            raise ValueError(f"{field}.quality_budgets.{name} must be null")
    p95 = budget_values["fallback_visited_nodes_p95"]
    if expected_budget_availability == "quality_repair_plan":
        _json_binary64(
            p95,
            f"{field}.quality_budgets.fallback_visited_nodes_p95",
            maximum=float(pixel_count),
        )
    elif p95 is not None:
        raise ValueError(f"{field}.quality_budgets.fallback_visited_nodes_p95 must be null")
    return value


def _validate_frame_statistics_schema(
    context: DiagnosticsStageContext,
    stats: object,
) -> None:
    """Validate the complete recursive FrameStats schema and numeric bounds."""

    value = _exact_object(
        stats,
        {
            "schema_version",
            "frame_name",
            "render_mode",
            "geometry_mode",
            "occlusion_fill",
            "render_shape",
            "eyes",
            "metric_projection",
            "geometry_nearest",
        },
        "frame statistics",
    )
    if _json_u64(value["schema_version"], "schema_version") != 1:
        raise ValueError("frame statistics schema version is invalid")
    pixel_count = validate_diagnostics_cardinality_dimensions(
        frame_count=1,
        height=context.render_shape[0],
        width=context.render_shape[1],
    )
    lane_count = checked_u64_mul("frame_lane_count", 16, pixel_count)
    record_cap = min(lane_count, _QUALITY_RECORD_ARENA_BYTES // 16)
    eyes = _exact_object(value["eyes"], {"left", "right"}, "frame statistics eyes")
    audited_eyes = {
        name: _validate_eye_statistics(
            eyes[name],
            field=f"eyes.{name}",
            context=context,
            pixel_count=pixel_count,
            lane_count=lane_count,
            record_cap=record_cap,
        )
        for name in ("left", "right")
    }

    if context.render_mode == "quality" and context.occlusion_fill == "background":
        for field, maximum in (
            ("local_slot_charge", QUALITY_LOCAL_SLOT_CAP),
            ("local_neighborhood_sample_charge", QUALITY_LOCAL_NEIGHBOR_SAMPLE_CAP),
            ("local_budget_skipped_run_count", record_cap),
            ("fallback_visited_nodes_total", QUALITY_REPAIR_FALLBACK_VISIT_CAP),
        ):
            total = checked_u64_add(
                f"shared_quality_budget.{field}",
                audited_eyes["left"]["quality_budgets"][field],
                audited_eyes["right"]["quality_budgets"][field],
            )
            if total > maximum:
                raise ValueError(f"shared Quality budget {field} exceeds its frame cap")

    metric = value["metric_projection"]
    if context.geometry_mode == "relative":
        if metric is not None:
            raise ValueError("relative frame statistics require null metric projection")
    else:
        metric_value = _exact_object(
            metric,
            {"valid_pixel_count", "clamped_pixel_count", "clamped_fraction"},
            "metric_projection",
        )
        valid = _json_u64(
            metric_value["valid_pixel_count"],
            "metric_projection.valid_pixel_count",
            maximum=pixel_count,
        )
        clamped = _json_u64(
            metric_value["clamped_pixel_count"],
            "metric_projection.clamped_pixel_count",
            maximum=valid,
        )
        fraction = _json_binary64(
            metric_value["clamped_fraction"],
            "metric_projection.clamped_fraction",
            maximum=1.0,
        )
        expected_fraction = float(clamped) / float(valid) if valid else 0.0
        if fraction != expected_fraction:
            raise ValueError("metric projection fraction is not count-derived")

    nearest = value["geometry_nearest"]
    if context.render_mode == "fast":
        if nearest is not None:
            raise ValueError("Fast frame statistics require null geometry-nearest statistics")
    else:
        nearest_value = _exact_object(
            nearest,
            {"indexed_sample_count", "query_count", "visited_nodes_total", "visited_nodes_max"},
            "geometry_nearest",
        )
        indexed = _json_u64(nearest_value["indexed_sample_count"], "geometry_nearest.indexed")
        if indexed != context.native_geometry_sample_count:
            raise ValueError("geometry-nearest indexed sample count differs from native geometry")
        query_count = _json_u64(
            nearest_value["query_count"],
            "geometry_nearest.query_count",
            maximum=pixel_count,
        )
        visited_total = _json_u64(
            nearest_value["visited_nodes_total"],
            "geometry_nearest.visited_nodes_total",
            maximum=QUALITY_GEOMETRY_NEAREST_VISIT_CAP,
        )
        visited_max = _json_u64(
            nearest_value["visited_nodes_max"],
            "geometry_nearest.visited_nodes_max",
            maximum=indexed,
        )
        if visited_max > visited_total or (query_count == 0 and (visited_total or visited_max)):
            raise ValueError("geometry-nearest query and visit counters disagree")


def _read_canonical_object(
    path: Path,
    *,
    max_bytes: int | None = None,
    acquired: AcquiredRoot | None = None,
) -> tuple[dict[str, Any], bytes]:
    if max_bytes is not None and max_bytes <= 0:
        raise ValueError("canonical JSON byte bound must be positive")
    try:
        handle = (
            acquired.open_regular(path)
            if acquired is not None
            else open_regular_beneath(path.parent, path)
        )
        with handle:
            raw = handle.read() if max_bytes is None else handle.read(max_bytes + 1)
    except NoFollowPathError as error:
        raise ValueError(str(error)) from error
    if max_bytes is not None and len(raw) > max_bytes:
        raise ValueError("canonical JSON payload exceeds its byte bound")
    return _parse_canonical_object(raw), raw


def _contained_payload_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise ValueError("diagnostic payload path must be a string")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("diagnostic payload path escapes the job root")
    lexical_root = Path(os.path.abspath(os.fspath(root)))
    path = Path(os.path.abspath(os.fspath(lexical_root / relative_path)))
    try:
        path.relative_to(lexical_root)
    except ValueError as error:
        raise ValueError("diagnostic payload path escapes the job root") from error
    return path


@dataclass(frozen=True)
class _PayloadAudit:
    captured: bytearray | None
    prefix: bytes


def _audit_payload_identity(
    root: Path,
    identity: object,
    *,
    expected_path: Path,
    capture_limit: int | None = None,
    prefix_bytes: int = 0,
    acquired: AcquiredRoot,
) -> _PayloadAudit:
    if not isinstance(identity, dict) or set(identity) != {
        "relative_path",
        "sha256",
        "byte_count",
    }:
        raise ValueError("diagnostic payload identity has the wrong keys")
    path = _contained_payload_path(root, identity["relative_path"])
    expected_lexical = Path(os.path.abspath(os.fspath(expected_path)))
    if path != expected_lexical:
        raise ValueError("diagnostic payload path does not match its declared slot")
    byte_count = identity["byte_count"]
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
        or not _is_hex64(identity["sha256"])
    ):
        raise ValueError("diagnostic payload identity is invalid")
    if capture_limit is not None and byte_count > capture_limit:
        raise ValueError("diagnostic payload exceeds its capture bound")
    digest = hashlib.sha256()
    total = 0
    captured = bytearray(byte_count) if capture_limit is not None else None
    prefix = bytearray(min(byte_count, prefix_bytes))
    workspace = bytearray(1024 * 1024)
    try:
        with acquired.open_regular(path) as handle:
            metadata = os.fstat(handle.fileno())
            if metadata.st_size != byte_count:
                raise ValueError("diagnostic payload byte count changed")
            while True:
                count = handle.readinto(workspace)
                if not count:
                    break
                digest.update(memoryview(workspace)[:count])
                if captured is not None:
                    captured[total : total + count] = memoryview(workspace)[:count]
                if total < len(prefix):
                    copied = min(count, len(prefix) - total)
                    prefix[total : total + copied] = memoryview(workspace)[:copied]
                total += count
    except NoFollowPathError as error:
        raise ValueError(str(error)) from error
    if total != byte_count or identity["sha256"] != digest.hexdigest():
        raise ValueError("diagnostic payload identity does not match its bytes")
    return _PayloadAudit(captured, bytes(prefix))


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
    *,
    acquired: AcquiredRoot | None = None,
) -> tuple[dict[str, Any], bytes, dict[str, Any], bytearray]:
    root = context.job_root
    if acquired is None:
        with AcquiredRoot(root) as owned:
            return _audit_frame_transaction(
                context,
                frame_name,
                acquired=owned,
            )
    frame_dir = root / "04_stereo_diagnostics" / "frames" / frame_name
    frame_json_bound = 4 * 272 * (context.render_shape[1] + 1) + 1024 * 1024
    manifest, raw_manifest = _read_canonical_object(
        frame_dir / "manifest.json",
        max_bytes=frame_json_bound,
        acquired=acquired,
    )
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
    left_rgb = _audit_payload_identity(
        root,
        payloads["left_rgb"],
        expected_path=context.left_rgb_parent / f"{frame_name}.png",
        prefix_bytes=33,
        acquired=acquired,
    )
    right_rgb = _audit_payload_identity(
        root,
        payloads["right_rgb"],
        expected_path=context.right_rgb_parent / f"{frame_name}.png",
        prefix_bytes=33,
        acquired=acquired,
    )
    if not png_header_bytes_match(
        left_rgb.prefix,
        shape=(*context.render_shape, 3),
        bit_depth=8,
    ) or not png_header_bytes_match(
        right_rgb.prefix,
        shape=(*context.render_shape, 3),
        bit_depth=8,
    ):
        raise ValueError("diagnostic RGB PNG header is invalid")
    mask_names = ("left_coverage", "left_repair", "right_coverage", "right_repair")
    for name in mask_names:
        identity = payloads[name]
        if context.mask_payloads_enabled:
            audited_mask = _audit_payload_identity(
                root,
                identity,
                expected_path=frame_dir / f"{name}.png",
                prefix_bytes=33,
                acquired=acquired,
            )
            if not png_header_bytes_match(
                audited_mask.prefix,
                shape=context.render_shape,
                bit_depth=8,
            ):
                raise ValueError("diagnostic mask PNG header is invalid")
        elif identity is not None:
            raise ValueError("disabled diagnostic mask has a payload identity")
    stats_audit = _audit_payload_identity(
        root,
        payloads["stats"],
        expected_path=frame_dir / "stats.json",
        capture_limit=frame_json_bound,
        acquired=acquired,
    )
    stats_raw = stats_audit.captured
    assert stats_raw is not None
    stats = _parse_streaming_frame_stats(
        stats_raw,
        render_width=context.render_shape[1],
    )
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
    _validate_frame_statistics_schema(context, stats)
    expected_quality_limits = (
        {eye: stats["eyes"][eye]["quality_limits"] for eye in ("left", "right")}
        if context.render_mode == "quality" and context.occlusion_fill == "background"
        else None
    )
    if manifest["quality_limits"] != expected_quality_limits:
        raise ValueError("manifest and statistics Quality limits disagree")
    return manifest, raw_manifest, stats, stats_raw


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


def _streaming_diagnostics_metadata(
    context: StreamingDiagnosticsStageContext,
    *,
    status: Literal["building", "complete", "legacy_fast_unavailable"],
    ordered_frame_manifest_fingerprint: str | None = None,
    frames_jsonl_sha256: str | None = None,
    summary_sha256: str | None = None,
    metric_clamp_summary_sha256: str | None = None,
) -> StreamedCanonicalObject:
    return build_streamed_canonical_object(
        static_fields={
            "algorithm_version": "stereo-coverage-sidecar-v1",
            "final_video_manifest_sha256": None,
            "frame_manifest_schema_version": 1,
            "frames_jsonl_sha256": frames_jsonl_sha256,
            "geometry_mode": context.geometry_mode,
            "mask_payloads_enabled": context.mask_payloads_enabled,
            "metric_clamp_summary_sha256": metric_clamp_summary_sha256,
            "native_geometry_fingerprint": context.native_geometry_fingerprint,
            "occlusion_fill": context.occlusion_fill,
            "ordered_frame_manifest_fingerprint": ordered_frame_manifest_fingerprint,
            "prune_entries": None,
            "pruned_from": None,
            "quality_input_manifest_sha256": context.quality_input_manifest_sha256,
            "render_mode": context.render_mode,
            "render_shape": [context.render_shape[0], context.render_shape[1]],
            "rgb_stage_fingerprint": context.rgb_stage_fingerprint,
            "schema_version": 1,
            "source_guide_fingerprint": context.source_guide_fingerprint,
            "stats_schema_version": 1,
            "status": status,
            "summary_sha256": summary_sha256,
        },
        array_key="frame_names",
        items=context.frame_names,
        fingerprint_key="fingerprint",
    )


def _matches_streamed_file(
    acquired: AcquiredRoot,
    path: Path,
    *,
    raw_sha256: str,
    raw_byte_count: int,
) -> bool:
    digest = hashlib.sha256()
    byte_count = 0
    with acquired.open_regular(path) as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
    return byte_count == raw_byte_count and digest.hexdigest() == raw_sha256


def _is_stream_prefix(
    acquired: AcquiredRoot,
    path: Path,
    chunks: Iterable[bytes],
) -> bool:
    with acquired.open_regular(path) as handle:
        buffered = b""
        for expected in chunks:
            if not expected:
                continue
            while len(buffered) < len(expected):
                piece = handle.read(len(expected) - len(buffered))
                if not piece:
                    return expected.startswith(buffered)
                buffered += piece
            if buffered[: len(expected)] != expected:
                return False
            buffered = buffered[len(expected) :]
        return not buffered and not handle.read(1)


def _prepare_legacy_output(
    acquired: AcquiredRoot,
    target: Path,
    *,
    chunks: Callable[[], Iterable[bytes]],
    raw_sha256: str,
    raw_byte_count: int,
) -> bool:
    """Validate a final artifact and retire only a canonical temporary prefix."""

    temporary = publication_temporary_path(target)
    temporary_identity = acquired.regular_identity(temporary, missing_ok=True)
    if temporary_identity is not None:
        if not _is_stream_prefix(acquired, temporary, chunks()):
            raise ValueError(f"legacy Fast diagnostics temporary is foreign: {temporary.name}")
        acquired.unlink_owned_regular(
            temporary,
            expected_identity=temporary_identity,
        )
    target_identity = acquired.regular_identity(target, missing_ok=True)
    if target_identity is None:
        return False
    if not _matches_streamed_file(
        acquired,
        target,
        raw_sha256=raw_sha256,
        raw_byte_count=raw_byte_count,
    ):
        raise ValueError(f"legacy Fast diagnostics artifact differs: {target.name}")
    return True


def _directory_contains_only_regular_names(
    acquired: AcquiredRoot,
    root: Path,
    *,
    allowed_names: set[str],
    require_exact: bool,
) -> bool:
    """Audit a tiny closed namespace without materializing a foreign directory."""

    seen: set[str] = set()
    for entry in acquired.iter_directory(root):
        if (
            entry.name not in allowed_names
            or entry.name in seen
            or entry.is_directory
            or not entry.is_regular
        ):
            return False
        seen.add(entry.name)
    return not require_exact or seen == allowed_names


def _legacy_fast_documents(
    context: StreamingDiagnosticsStageContext,
    *,
    metric_projection: dict[str, Any] | None = None,
    metric_clamp_summary_sha256: str | None = None,
) -> tuple[str, str, StreamedCanonicalValue, StreamedCanonicalObject]:
    if context.render_mode != "fast" or context.mask_payloads_enabled:
        raise ValueError("legacy Fast adoption requires Fast with masks disabled")
    if (context.geometry_mode == "metric") != (metric_projection is not None):
        raise ValueError("legacy Fast metric projection availability is inconsistent")
    if (context.geometry_mode == "metric") != _is_hex64(metric_clamp_summary_sha256):
        raise ValueError("legacy Fast clamp-summary identity is inconsistent")

    ordered_fingerprint = hashlib.sha256(b"[]").hexdigest()
    jsonl_sha256 = hashlib.sha256(b"").hexdigest()
    summary = build_streamed_canonical_value(
        {
            "algorithm_version": "stereo-coverage-sidecar-v1",
            "availability": "legacy_fast_unavailable",
            "eyes": None,
            "frame_count": context.frame_count,
            "frame_names": ReplayableJsonArray(lambda: _iter_streaming_frame_names(context)),
            "frames_jsonl_sha256": jsonl_sha256,
            "geometry_mode": context.geometry_mode,
            "geometry_nearest": None,
            "metric_projection": metric_projection,
            "occlusion_fill": context.occlusion_fill,
            "ordered_frame_manifest_fingerprint": ordered_fingerprint,
            "render_mode": "fast",
            "render_shape": [context.render_shape[0], context.render_shape[1]],
            "schema_version": 1,
        }
    )
    metadata = _streaming_diagnostics_metadata(
        context,
        status="legacy_fast_unavailable",
        ordered_frame_manifest_fingerprint=ordered_fingerprint,
        frames_jsonl_sha256=jsonl_sha256,
        summary_sha256=summary.raw_sha256,
        metric_clamp_summary_sha256=metric_clamp_summary_sha256,
    )
    return ordered_fingerprint, jsonl_sha256, summary, metadata


def legacy_fast_unavailable_aggregates_match(
    context: StreamingDiagnosticsStageContext,
    *,
    metric_projection: dict[str, Any] | None = None,
    metric_clamp_summary_sha256: str | None = None,
) -> bool:
    """Authenticate an already completed aggregate-only legacy marker."""

    try:
        _ordered, jsonl_sha256, summary, metadata = _legacy_fast_documents(
            context,
            metric_projection=metric_projection,
            metric_clamp_summary_sha256=metric_clamp_summary_sha256,
        )
        root = context.job_root / "04_stereo_diagnostics"
        jsonl_path = root / "stereo_coverage_frames.jsonl"
        summary_path = root / "stereo_coverage_summary.json"
        metadata_path = root / "metadata.json"
        with AcquiredRoot(context.job_root) as acquired:
            expected_names = {
                jsonl_path.name,
                summary_path.name,
                metadata_path.name,
            }
            if not _directory_contains_only_regular_names(
                acquired,
                root,
                allowed_names=expected_names,
                require_exact=True,
            ):
                return False
            return all(
                (
                    _matches_streamed_file(
                        acquired,
                        jsonl_path,
                        raw_sha256=jsonl_sha256,
                        raw_byte_count=0,
                    ),
                    _matches_streamed_file(
                        acquired,
                        summary_path,
                        raw_sha256=summary.raw_sha256,
                        raw_byte_count=summary.raw_byte_count,
                    ),
                    _matches_streamed_file(
                        acquired,
                        metadata_path,
                        raw_sha256=metadata.raw_sha256,
                        raw_byte_count=metadata.raw_byte_count,
                    ),
                )
            )
    except (OSError, TypeError, ValueError, NoFollowPathError):
        return False


def publish_legacy_fast_unavailable(
    context: StreamingDiagnosticsStageContext,
    adoption: object,
    *,
    metric_projection: dict[str, Any] | None = None,
    metric_clamp_summary_sha256: str | None = None,
    fault: FaultHook | None = None,
) -> StreamingDiagnosticsCompletion:
    """Publish the aggregate-only marker for one intact legacy Fast stage."""

    ordered_fingerprint, jsonl_sha256, summary, metadata = _legacy_fast_documents(
        context,
        metric_projection=metric_projection,
        metric_clamp_summary_sha256=metric_clamp_summary_sha256,
    )

    from ...io.resume import (
        LegacyFastAdoptionV1,
        load_legacy_fast_adoption_intent_v1,
        retire_legacy_fast_adoption_intent_v1,
    )

    if not isinstance(adoption, LegacyFastAdoptionV1):
        raise TypeError("legacy Fast publication requires its durable adoption proof")
    intent = load_legacy_fast_adoption_intent_v1(context.job_root)
    if intent != adoption:
        raise ValueError("legacy Fast diagnostics lack the matching durable intent")
    if (
        adoption.job_root != context.job_root
        or adoption.frame_count != context.frame_count
        or adoption.render_shape != context.render_shape
        or adoption.geometry_mode != context.geometry_mode
        or adoption.occlusion_fill != context.occlusion_fill
        or adoption.rgb_stage_fingerprint != context.rgb_stage_fingerprint
        or adoption.source_guide_fingerprint != context.source_guide_fingerprint
        or adoption.metric_clamp_summary_sha256 != metric_clamp_summary_sha256
    ):
        raise ValueError("legacy Fast diagnostics context differs from its intent")

    root = context.job_root / "04_stereo_diagnostics"
    if not root.exists():
        root.mkdir()
        _sync_parent(root.parent)
        _call_fault(fault, "after_legacy_root_directory_sync")

    jsonl_path = root / "stereo_coverage_frames.jsonl"
    summary_path = root / "stereo_coverage_summary.json"
    metadata_path = root / "metadata.json"
    allowed_names = {
        path.name
        for target in (jsonl_path, summary_path, metadata_path)
        for path in (target, publication_temporary_path(target))
    }
    with AcquiredRoot(context.job_root) as acquired:
        if not _directory_contains_only_regular_names(
            acquired,
            root,
            allowed_names=allowed_names,
            require_exact=False,
        ):
            raise ValueError("legacy Fast diagnostics prefix contains a foreign entry")

        jsonl_present = _prepare_legacy_output(
            acquired,
            jsonl_path,
            chunks=lambda: iter((b"",)),
            raw_sha256=jsonl_sha256,
            raw_byte_count=0,
        )
        if not jsonl_present:
            owned = OwnedSiblingTemporary.create(jsonl_path)
            try:
                _call_fault(fault, "before_frames_jsonl_temporary_write")
                owned.sync()
                _call_fault(fault, "after_frames_jsonl_temporary_sync")
                _call_fault(fault, "before_frames_jsonl_replace")
                owned.replace(
                    after_replace=lambda: _call_fault(
                        fault,
                        "after_frames_jsonl_replace",
                    ),
                )
                _call_fault(fault, "after_frames_jsonl_directory_sync")
            finally:
                owned.cleanup()

        summary_present = _prepare_legacy_output(
            acquired,
            summary_path,
            chunks=summary.chunks,
            raw_sha256=summary.raw_sha256,
            raw_byte_count=summary.raw_byte_count,
        )
        if not summary_present:
            summary.publish(
                summary_path,
                fault=fault,
                label="root_summary",
            )
        metadata_present = _prepare_legacy_output(
            acquired,
            metadata_path,
            chunks=metadata.chunks,
            raw_sha256=metadata.raw_sha256,
            raw_byte_count=metadata.raw_byte_count,
        )
        if not metadata_present:
            metadata.publish(
                metadata_path,
                fault=fault,
                label="root_metadata",
            )
        if not all(
            (
                _matches_streamed_file(
                    acquired,
                    jsonl_path,
                    raw_sha256=jsonl_sha256,
                    raw_byte_count=0,
                ),
                _matches_streamed_file(
                    acquired,
                    summary_path,
                    raw_sha256=summary.raw_sha256,
                    raw_byte_count=summary.raw_byte_count,
                ),
                _matches_streamed_file(
                    acquired,
                    metadata_path,
                    raw_sha256=metadata.raw_sha256,
                    raw_byte_count=metadata.raw_byte_count,
                ),
            )
        ):
            raise ValueError("legacy Fast diagnostics failed final reopen validation")
    retire_legacy_fast_adoption_intent_v1(adoption)
    return StreamingDiagnosticsCompletion(
        metadata=metadata,
        summary=summary,
        metric_clamp_summary=None,
        ordered_frame_manifest_fingerprint=ordered_fingerprint,
        frames_jsonl_sha256=jsonl_sha256,
    )


def begin_streaming_diagnostics_stage(
    context: StreamingDiagnosticsStageContext,
    *,
    fault: FaultHook | None = None,
) -> StreamedCanonicalObject:
    """Publish a building root without retaining the frame-name sequence."""

    metadata = _streaming_diagnostics_metadata(context, status="building")
    metadata.publish(
        context.job_root / "04_stereo_diagnostics" / "metadata.json",
        fault=fault,
        label="building_metadata",
    )
    return metadata


def classify_streaming_diagnostics_frame(
    context: StreamingDiagnosticsStageContext,
    frame_name: str,
) -> Literal["reuse", "migrate", "render"]:
    """Audit one committed frame transaction using only one resident name."""

    try:
        with AcquiredRoot(context.job_root) as acquired:
            try:
                _audit_frame_transaction(
                    context.one_frame(frame_name),
                    frame_name,
                    acquired=acquired,
                )
            except (OSError, ValueError, KeyError, TypeError):
                if not context.mask_payloads_enabled:
                    _audit_frame_transaction(
                        context.one_frame(frame_name, masks=True),
                        frame_name,
                        acquired=acquired,
                    )
                    return "migrate"
                return "render"
    except (OSError, ValueError, KeyError, TypeError):
        return "render"
    return "reuse"


def _streamable_diagnostics_value(value: object) -> object:
    if isinstance(value, DenseHistogramPairs):
        bins = value.bins

        def replay():
            for bucket in range(int(bins.size)):
                count = int(bins[bucket])
                if count:
                    yield [bucket, count]

        return ReplayableJsonArray(replay)
    if isinstance(value, dict):
        return {key: _streamable_diagnostics_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_streamable_diagnostics_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_streamable_diagnostics_value(item) for item in value)
    return value


def _iter_streaming_frame_names(
    context: StreamingDiagnosticsStageContext,
) -> Iterable[str]:
    count = 0
    for frame_name in context.frame_names():
        if count >= context.frame_count:
            raise ValueError("diagnostics name replay exceeds its frame count")
        if not isinstance(frame_name, str) or not frame_name or Path(frame_name).suffix:
            raise ValueError("diagnostics name replay yielded an invalid stem")
        count += 1
        yield frame_name
    if count != context.frame_count:
        raise ValueError("diagnostics name replay ended before its frame count")


def _streaming_jsonl_identity(
    context: StreamingDiagnosticsStageContext,
    *,
    output: Any | None,
    acquired: AcquiredRoot | None = None,
) -> tuple[str, str]:
    if acquired is None:
        with AcquiredRoot(context.job_root) as owned:
            return _streaming_jsonl_identity(context, output=output, acquired=owned)
    ordered_digest = hashlib.sha256()
    ordered_digest.update(b"[")
    jsonl_digest = hashlib.sha256()
    for index, frame_name in enumerate(_iter_streaming_frame_names(context)):
        _manifest, raw_manifest, _stats, stats_raw = _audit_frame_transaction(
            context.one_frame(frame_name),
            frame_name,
            acquired=acquired,
        )
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
        if output is not None:
            if output.write(stats_raw) != len(stats_raw) or output.write(b"\n") != 1:
                raise OSError("short diagnostics JSONL write")
        jsonl_digest.update(stats_raw)
        jsonl_digest.update(b"\n")
        del _manifest, raw_manifest, _stats, stats_raw
    ordered_digest.update(b"]")
    return ordered_digest.hexdigest(), jsonl_digest.hexdigest()


def _iter_streaming_jsonl(
    path: Path,
    context: StreamingDiagnosticsStageContext,
    *,
    acquired: AcquiredRoot | None = None,
) -> Iterable[dict[str, Any]]:
    if acquired is None:
        with AcquiredRoot(context.job_root) as owned:
            yield from _iter_streaming_jsonl(path, context, acquired=owned)
        return
    maximum_line_bytes = 4 * 272 * (context.render_shape[1] + 1) + 1024 * 1024
    count = 0
    with acquired.open_regular(path) as raw_handle:
        with io.BufferedReader(raw_handle, buffer_size=1024 * 1024) as handle:
            while True:
                line = handle.readline(maximum_line_bytes + 1)
                if not line:
                    break
                if len(line) > maximum_line_bytes or not line.endswith(b"\n"):
                    raise ValueError("diagnostics JSONL row is unbounded or unterminated")
                if count >= context.frame_count:
                    raise ValueError("diagnostics JSONL exceeds its frame count")
                count += 1
                value = _parse_streaming_frame_stats(
                    line,
                    render_width=context.render_shape[1],
                    raw_end=len(line) - 1,
                )
                line = None
                yield value
                del value
    if count != context.frame_count:
        raise ValueError("diagnostics JSONL row count differs from the frame sequence")


def _stream_file_sha256(path: Path, *, acquired: AcquiredRoot | None = None) -> str:
    digest = hashlib.sha256()
    workspace = bytearray(1024 * 1024)
    view = memoryview(workspace)
    handle = (
        acquired.open_regular(path)
        if acquired is not None
        else open_regular_beneath(path.parent, path)
    )
    with handle:
        while True:
            count = handle.readinto(workspace)
            if not count:
                break
            digest.update(view[:count])
    return digest.hexdigest()


class _ExactCanonicalReader:
    """Bounded exact-byte reader used for the variable-length root metadata."""

    def __init__(self, handle: Any) -> None:
        self.handle = handle
        self.self_digest = hashlib.sha256()
        self.raw_digest = hashlib.sha256()
        self.raw_byte_count = 0
        self._pending = b""
        self._buffer = b""
        self._buffer_offset = 0

    def _source_read(self, byte_count: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < byte_count:
            available = len(self._buffer) - self._buffer_offset
            if available:
                take = min(byte_count - len(chunks), available)
                chunks.extend(self._buffer[self._buffer_offset : self._buffer_offset + take])
                self._buffer_offset += take
                continue
            self._buffer = self.handle.read(max(64 * 1024, byte_count - len(chunks)))
            self._buffer_offset = 0
            if not self._buffer:
                break
        return bytes(chunks)

    def _source_chunk(self) -> bytes:
        if self._buffer_offset == len(self._buffer):
            self._buffer = self.handle.read(64 * 1024)
            self._buffer_offset = 0
        return self._buffer[self._buffer_offset :]

    def _read_exact(self, byte_count: int) -> bytes:
        chunks = bytearray()
        if self._pending and byte_count:
            chunks.extend(self._pending)
            self._pending = b""
        while len(chunks) < byte_count:
            chunk = self._source_read(byte_count - len(chunks))
            if not chunk:
                raise ValueError("diagnostics metadata ended early")
            chunks.extend(chunk)
        raw = bytes(chunks)
        self.raw_digest.update(raw)
        self.raw_byte_count += len(raw)
        return raw

    def peek(self) -> bytes:
        if not self._pending:
            self._pending = self._source_read(1)
        return self._pending

    def expect(self, expected: bytes, *, fingerprinted: bool = True) -> None:
        actual = self._read_exact(len(expected))
        if actual != expected:
            raise ValueError("diagnostics metadata is noncanonical or changed")
        if fingerprinted:
            self.self_digest.update(actual)

    def canonical_string(self, *, maximum_bytes: int, fingerprinted: bool = True) -> str:
        raw = bytearray(self._read_exact(1))
        if raw != b'"':
            raise ValueError("diagnostics metadata string is malformed")
        content_bytes = 0
        while True:
            chunk = self._source_chunk()[: maximum_bytes - content_bytes + 1]
            if not chunk:
                raise ValueError("diagnostics metadata string ended early")
            quote = chunk.find(b'"')
            content = chunk if quote < 0 else chunk[:quote]
            slash = content.find(b"\\")
            control = next(
                (index for index, value in enumerate(content) if value < 0x20),
                -1,
            )
            invalid = min(
                (index for index in (slash, control) if index >= 0),
                default=-1,
            )
            if invalid >= 0 and (quote < 0 or invalid < quote):
                raise ValueError("diagnostics metadata string is noncanonical")
            if quote >= 0:
                if content_bytes + quote > maximum_bytes:
                    raise ValueError("diagnostics metadata string exceeds its bound")
                raw.extend(self._read_exact(quote + 1))
                break
            raw.extend(self._read_exact(len(chunk)))
            content_bytes += len(chunk)
            if content_bytes > maximum_bytes:
                raise ValueError("diagnostics metadata string exceeds its bound")
        if fingerprinted:
            self.self_digest.update(raw)
        return raw[1:-1].decode("ascii")

    def optional_hex64(self) -> str | None:
        first = self._read_exact(1)
        self.self_digest.update(first)
        if first == b"n":
            self.expect(b"ull")
            return None
        if first != b'"':
            raise ValueError("diagnostics metadata hash is malformed")
        raw = self._read_exact(64)
        self.self_digest.update(raw)
        if not _is_hex64(raw.decode("ascii")):
            raise ValueError("diagnostics metadata hash is malformed")
        self.expect(b'"')
        return raw.decode("ascii")

    def canonical_value(
        self,
        *,
        maximum_bytes: int,
        fingerprinted: bool = True,
        maximum_depth: int = 8,
        maximum_containers: int = 32,
        maximum_scalars: int = 256,
        maximum_string_bytes: int = 4096,
    ) -> object:
        if maximum_bytes <= 0:
            raise ValueError("diagnostics canonical value bound must be positive")
        if (
            min(
                maximum_depth,
                maximum_containers,
                maximum_scalars,
                maximum_string_bytes,
            )
            <= 0
        ):
            raise ValueError("diagnostics canonical parser bounds must be positive")
        raw = bytearray(self._read_exact(1))
        first = raw[0]
        container_count = 0
        scalar_count = 0

        def check_raw_bound() -> None:
            if len(raw) > maximum_bytes:
                raise StereoControlBudgetError(
                    "diagnostics canonical value exceeds its byte budget"
                )

        def count_scalar() -> None:
            nonlocal scalar_count
            scalar_count += 1
            if scalar_count > maximum_scalars:
                raise StereoControlBudgetError(
                    "diagnostics canonical value exceeds its scalar budget"
                )

        def count_container(depth: int) -> None:
            nonlocal container_count
            if depth > maximum_depth:
                raise StereoControlBudgetError(
                    "diagnostics canonical value exceeds its depth budget"
                )
            container_count += 1
            if container_count > maximum_containers:
                raise StereoControlBudgetError(
                    "diagnostics canonical value exceeds its container budget"
                )

        if first == ord('"'):
            count_scalar()
            escaped = False
            string_bytes = 0
            while True:
                character = self._read_exact(1)
                raw.extend(character)
                if escaped:
                    escaped = False
                elif character == b"\\":
                    escaped = True
                elif character == b'"':
                    break
                string_bytes += 1
                if string_bytes > maximum_string_bytes:
                    raise StereoControlBudgetError(
                        "diagnostics canonical value exceeds its string budget"
                    )
                check_raw_bound()
        elif first in {ord("["), ord("{")}:
            count_container(1)
            stack = [ord("]") if first == ord("[") else ord("}")]
            in_string = False
            escaped = False
            scalar_active = False
            scalar_bytes = 0
            string_bytes = 0
            while stack:
                character = self._read_exact(1)
                raw.extend(character)
                byte = character[0]
                if in_string:
                    if escaped:
                        escaped = False
                        string_bytes += 1
                    elif byte == ord("\\"):
                        escaped = True
                        string_bytes += 1
                    elif byte == ord('"'):
                        in_string = False
                    else:
                        string_bytes += 1
                    if string_bytes > maximum_string_bytes:
                        raise StereoControlBudgetError(
                            "diagnostics canonical value exceeds its string budget"
                        )
                elif byte == ord('"'):
                    count_scalar()
                    in_string = True
                    scalar_active = False
                    string_bytes = 0
                elif byte == ord("["):
                    count_container(len(stack) + 1)
                    stack.append(ord("]"))
                    scalar_active = False
                elif byte == ord("{"):
                    count_container(len(stack) + 1)
                    stack.append(ord("}"))
                elif byte in {ord("]"), ord("}")}:
                    if byte != stack.pop():
                        raise ValueError("diagnostics canonical value nesting is malformed")
                    scalar_active = False
                elif byte in {ord(","), ord(":")}:
                    scalar_active = False
                elif byte in b" \t\r\n":
                    raise ValueError("diagnostics canonical value is noncanonical")
                else:
                    if not scalar_active:
                        count_scalar()
                        scalar_active = True
                        scalar_bytes = 0
                    scalar_bytes += 1
                    if scalar_bytes > maximum_string_bytes:
                        raise StereoControlBudgetError(
                            "diagnostics canonical value exceeds its scalar-token budget"
                        )
                check_raw_bound()
        else:
            while self.peek() not in {b",", b"}"}:
                if not self.peek():
                    raise ValueError("diagnostics canonical value ended early")
                raw.extend(self._read_exact(1))
                if raw[-1] in b" \t\r\n":
                    raise ValueError("diagnostics canonical value is noncanonical")
                if len(raw) > maximum_string_bytes:
                    raise StereoControlBudgetError(
                        "diagnostics canonical value exceeds its scalar-token budget"
                    )
                check_raw_bound()
        try:
            value = strict_canonical_json_loads(bytes(raw), max_bytes=maximum_bytes)
        except (MemoryError, RecursionError) as error:
            raise StereoControlBudgetError(
                "diagnostics canonical value exceeds its decoder budget"
            ) from error
        except CanonicalJsonError as error:
            raise ValueError("diagnostics metadata value is noncanonical") from error
        if fingerprinted:
            self.self_digest.update(raw)
        return value


class _BoundedReadHandle:
    """Expose exactly one declared prefix of an already-open binary handle."""

    def __init__(self, handle: Any, byte_count: int) -> None:
        if type(byte_count) is not int or byte_count <= 0:
            raise ValueError("canonical prefix byte count must be positive")
        self.handle = handle
        self.remaining = byte_count

    def read(self, byte_count: int = -1) -> bytes:
        if self.remaining == 0:
            return b""
        requested = self.remaining if byte_count < 0 else min(byte_count, self.remaining)
        chunk = self.handle.read(requested)
        self.remaining -= len(chunk)
        return chunk


_DIAGNOSTICS_METADATA_KEYS = {
    "algorithm_version",
    "final_video_manifest_sha256",
    "fingerprint",
    "frame_manifest_schema_version",
    "frame_names",
    "frames_jsonl_sha256",
    "geometry_mode",
    "mask_payloads_enabled",
    "metric_clamp_summary_sha256",
    "native_geometry_fingerprint",
    "occlusion_fill",
    "ordered_frame_manifest_fingerprint",
    "prune_entries",
    "pruned_from",
    "quality_input_manifest_sha256",
    "render_mode",
    "render_shape",
    "rgb_stage_fingerprint",
    "schema_version",
    "source_guide_fingerprint",
    "stats_schema_version",
    "status",
    "summary_sha256",
}
_DIAGNOSTICS_SMALL_FIELD_MAX_BYTES = 64 * 1024
_DIAGNOSTICS_PRUNE_ENTRIES_MAX_CONTAINERS = 1 + 16 * 4
_DIAGNOSTICS_PRUNE_ENTRIES_MAX_SCALARS = 16 * 21


@dataclass(frozen=True, slots=True)
class DiagnosticsMetadataProjection:
    path: Path
    static_fields: dict[str, Any]
    fingerprint: str
    frame_count: int
    raw_sha256: str
    raw_byte_count: int
    prefix_only: bool = False

    @property
    def status(self) -> str:
        return str(self.static_fields["status"])

    def validate_state_contract(self) -> DiagnosticsMetadataProjection:
        """Validate every scalar and status-dependent metadata invariant."""

        fields = self.static_fields
        status = fields.get("status")
        render_mode = fields.get("render_mode")
        geometry_mode = fields.get("geometry_mode")
        render_shape = fields.get("render_shape")
        if (
            fields.get("schema_version") != 1
            or type(fields.get("schema_version")) is not int
            or fields.get("algorithm_version") != "stereo-coverage-sidecar-v1"
            or fields.get("stats_schema_version") != 1
            or type(fields.get("stats_schema_version")) is not int
            or fields.get("frame_manifest_schema_version") != 1
            or type(fields.get("frame_manifest_schema_version")) is not int
            or status not in {"building", "complete", "legacy_fast_unavailable", "payload_pruned"}
            or render_mode not in {"fast", "quality"}
            or geometry_mode not in {"relative", "metric"}
            or fields.get("occlusion_fill") not in {"none", "background"}
            or type(fields.get("mask_payloads_enabled")) is not bool
            or not _is_hex64(fields.get("rgb_stage_fingerprint"))
            or not _is_hex64(fields.get("source_guide_fingerprint"))
            or not isinstance(render_shape, list)
            or len(render_shape) != 2
            or any(type(dimension) is not int or dimension <= 0 for dimension in render_shape)
        ):
            raise ValueError("diagnostics metadata scalar contract is invalid")

        native_fingerprint = fields.get("native_geometry_fingerprint")
        content_sha256 = fields.get("quality_input_manifest_sha256")
        if render_mode == "quality":
            if not _is_hex64(native_fingerprint):
                raise ValueError("Quality diagnostics native geometry identity is invalid")
            if not _is_hex64(content_sha256):
                raise ValueError("Quality diagnostics content identity is invalid")
        elif native_fingerprint is not None or content_sha256 is not None:
            raise ValueError("Fast diagnostics carry Quality-only provenance")

        ordered = fields.get("ordered_frame_manifest_fingerprint")
        frames_hash = fields.get("frames_jsonl_sha256")
        summary_hash = fields.get("summary_sha256")
        current_clamp = fields.get("metric_clamp_summary_sha256")
        pruned_from = fields.get("pruned_from")
        final_hash = fields.get("final_video_manifest_sha256")
        prune_entries = fields.get("prune_entries")
        if status == "building":
            if any(
                value is not None
                for value in (
                    ordered,
                    frames_hash,
                    summary_hash,
                    current_clamp,
                    pruned_from,
                    final_hash,
                    prune_entries,
                )
            ):
                raise ValueError("building diagnostics carry completed state")
            return self

        if status in {"complete", "legacy_fast_unavailable"}:
            if (
                not _is_hex64(ordered)
                or not _is_hex64(frames_hash)
                or not _is_hex64(summary_hash)
                or (geometry_mode == "metric") != _is_hex64(current_clamp)
                or pruned_from is not None
                or final_hash is not None
                or prune_entries is not None
            ):
                raise ValueError("complete diagnostics aggregate state is invalid")
            if status == "legacy_fast_unavailable":
                if (
                    render_mode != "fast"
                    or fields["mask_payloads_enabled"] is not False
                    or ordered != hashlib.sha256(b"[]").hexdigest()
                ):
                    raise ValueError("legacy Fast diagnostics state is invalid")
            return self

        if (
            fields["mask_payloads_enabled"] is not False
            or ordered is not None
            or not _is_hex64(frames_hash)
            or not _is_hex64(summary_hash)
            or current_clamp is not None
            or not _is_hex64(final_hash)
            or not isinstance(prune_entries, list)
            or not isinstance(pruned_from, dict)
            or set(pruned_from)
            != {
                "previous_status",
                "previous_diagnostics_fingerprint",
                "ordered_frame_manifest_fingerprint",
                "frames_jsonl_sha256",
                "summary_sha256",
                "metric_clamp_summary_sha256",
            }
            or pruned_from.get("previous_status") not in {"complete", "legacy_fast_unavailable"}
            or not _is_hex64(pruned_from.get("previous_diagnostics_fingerprint"))
            or not _is_hex64(pruned_from.get("ordered_frame_manifest_fingerprint"))
            or pruned_from.get("frames_jsonl_sha256") != frames_hash
            or pruned_from.get("summary_sha256") != summary_hash
            or (geometry_mode == "metric")
            != _is_hex64(pruned_from.get("metric_clamp_summary_sha256"))
            or (
                pruned_from.get("previous_status") == "legacy_fast_unavailable"
                and render_mode != "fast"
            )
        ):
            raise ValueError("payload-pruned diagnostics state is invalid")
        return self

    def iter_frame_names(self, *, acquired: AcquiredRoot | None = None) -> Iterator[str]:
        if acquired is None:
            with AcquiredRoot(self.path.parent.parent) as owned:
                yield from self.iter_frame_names(acquired=owned)
            return
        final_projection: DiagnosticsMetadataProjection | None = None
        for kind, value in _diagnostics_metadata_events(
            self.path,
            raw_byte_count=self.raw_byte_count if self.prefix_only else None,
            acquired=acquired,
        ):
            if kind == "frame":
                yield str(value)
            else:
                final_projection = value
        if final_projection != self:
            raise ValueError("diagnostics metadata changed during frame-name replay")

    def rebuild(self, static_fields: dict[str, Any]) -> StreamedCanonicalObject:
        if set(static_fields) != _DIAGNOSTICS_METADATA_KEYS - {
            "fingerprint",
            "frame_names",
        }:
            raise ValueError("diagnostics metadata rebuild fields are not closed")
        return build_streamed_canonical_object(
            static_fields=static_fields,
            array_key="frame_names",
            items=self.iter_frame_names,
            fingerprint_key="fingerprint",
        )


def _diagnostics_metadata_events(
    path: Path,
    *,
    raw_byte_count: int | None = None,
    acquired: AcquiredRoot | None = None,
) -> Iterator[tuple[Literal["frame", "projection"], Any]]:
    path = Path(path)
    if acquired is None:
        with AcquiredRoot(path.parent.parent) as owned:
            yield from _diagnostics_metadata_events(
                path,
                raw_byte_count=raw_byte_count,
                acquired=owned,
            )
        return
    static_fields: dict[str, Any] = {}
    fingerprint = ""
    frame_count = 0
    previous_index: int | None = None
    with acquired.open_regular(path) as handle:
        bounded = (
            _BoundedReadHandle(handle, raw_byte_count) if raw_byte_count is not None else handle
        )
        reader = _ExactCanonicalReader(bounded)
        ordered_keys = sorted(_DIAGNOSTICS_METADATA_KEYS)
        reader.expect(b"{")
        for position, key in enumerate(ordered_keys):
            prefix = (b"," if position else b"") + canonical_json_bytes(key) + b":"
            reader.expect(prefix, fingerprinted=(key != "fingerprint"))
            if key == "fingerprint":
                fingerprint = reader.canonical_string(
                    maximum_bytes=64,
                    fingerprinted=False,
                )
                if not _is_hex64(fingerprint):
                    raise ValueError("diagnostics metadata fingerprint is malformed")
                continue
            if key != "frame_names":
                static_fields[key] = reader.canonical_value(
                    maximum_bytes=_DIAGNOSTICS_SMALL_FIELD_MAX_BYTES,
                    maximum_containers=(
                        _DIAGNOSTICS_PRUNE_ENTRIES_MAX_CONTAINERS if key == "prune_entries" else 32
                    ),
                    maximum_scalars=(
                        _DIAGNOSTICS_PRUNE_ENTRIES_MAX_SCALARS if key == "prune_entries" else 256
                    ),
                )
                continue
            reader.expect(b"[")
            while reader.peek() != b"]":
                if frame_count >= STEREO_CONTROL_FRAME_CAP:
                    raise StereoControlBudgetError(
                        "diagnostics frame-name sequence exceeds the control cap"
                    )
                if frame_count:
                    reader.expect(b",")
                frame_name = reader.canonical_string(maximum_bytes=1024)
                match = re.fullmatch(r"frame_([0-9]+)", frame_name, flags=re.ASCII)
                frame_index = int(match.group(1)) if match is not None else -1
                if (
                    match is None
                    or frame_index > (1 << 64) - 1
                    or frame_name != f"frame_{frame_index:06d}"
                    or previous_index is not None
                    and frame_index != previous_index + 1
                ):
                    raise ValueError("diagnostics frame-name sequence is noncanonical")
                previous_index = frame_index
                frame_count += 1
                yield "frame", frame_name
            reader.expect(b"]")
        reader.expect(b"}")
        if reader.peek():
            raise ValueError("diagnostics metadata has trailing bytes")
        if isinstance(bounded, _BoundedReadHandle) and bounded.remaining != 0:
            raise ValueError("diagnostics metadata ended before its declared prefix")
    if frame_count <= 0:
        raise ValueError("diagnostics metadata scalar contract is invalid")
    if fingerprint != reader.self_digest.hexdigest():
        raise ValueError("diagnostics metadata self fingerprint changed")
    yield (
        "projection",
        DiagnosticsMetadataProjection(
            path=path,
            static_fields=static_fields,
            fingerprint=fingerprint,
            frame_count=frame_count,
            raw_sha256=reader.raw_digest.hexdigest(),
            raw_byte_count=reader.raw_byte_count,
            prefix_only=raw_byte_count is not None,
        ),
    )


def project_diagnostics_metadata(
    path: Path,
    *,
    raw_byte_count: int | None = None,
    acquired: AcquiredRoot | None = None,
) -> DiagnosticsMetadataProjection:
    projection: DiagnosticsMetadataProjection | None = None
    for kind, value in _diagnostics_metadata_events(
        path,
        raw_byte_count=raw_byte_count,
        acquired=acquired,
    ):
        if kind == "projection":
            projection = value
    if projection is None:
        raise ValueError("diagnostics metadata projection is incomplete")
    return projection


_QUALITY_CONTENT_KEYS = {
    "algorithm_version",
    "fingerprint",
    "frame_names",
    "geometry",
    "geometry_mode",
    "guides",
    "native_geometry_fingerprint",
    "relative_encoding_scale_float32_bits",
    "schema_version",
    "source_guide_fingerprint",
}


@dataclass(frozen=True, slots=True)
class QualityInputMetadataProjection:
    """Bounded scalar projection of one canonical Quality content manifest."""

    path: Path
    geometry_mode: Literal["relative", "metric"]
    source_guide_fingerprint: str
    native_geometry_fingerprint: str
    fingerprint: str
    frame_count: int
    first_frame_index: int
    raw_sha256: str
    raw_byte_count: int

    def iter_frame_names(self) -> Iterator[str]:
        for offset in range(self.frame_count):
            yield f"frame_{self.first_frame_index + offset:06d}"


def _quality_frame_index(value: object) -> int:
    if not isinstance(value, str):
        raise ValueError("Quality content frame name is not a string")
    match = re.fullmatch(r"frame_([0-9]+)", value, flags=re.ASCII)
    index = int(match.group(1)) if match is not None else -1
    if match is None or index > (1 << 64) - 1 or value != f"frame_{index:06d}":
        raise ValueError("Quality content frame name is noncanonical")
    return index


def _quality_identity_common(
    value: object,
    *,
    expected_index: int,
    suffix: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Quality content identity is malformed")
    frame_name = f"frame_{expected_index:06d}"
    relative = value.get("relative_path")
    if value.get("frame_name") != frame_name or not isinstance(relative, str):
        raise ValueError("Quality content identity frame is invalid")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or pure.as_posix() != relative
        or any(part in {"", ".", ".."} or "\\" in part or "\x00" in part for part in pure.parts)
        or pure.name != f"{frame_name}{suffix}"
        or not _is_hex64(value.get("sha256"))
        or type(value.get("byte_count")) is not int
        or value["byte_count"] <= 0
    ):
        raise ValueError("Quality content identity is invalid")
    return value


def _validate_quality_guide(value: object, *, expected_index: int) -> None:
    guide = _quality_identity_common(value, expected_index=expected_index, suffix=".png")
    if set(guide) != {
        "byte_count",
        "frame_name",
        "png_header",
        "relative_path",
        "sha256",
    }:
        raise ValueError("Quality guide identity schema is invalid")
    header = guide.get("png_header")
    if (
        not isinstance(header, dict)
        or set(header) != {"bit_depth", "color_type", "height", "width"}
        or header.get("bit_depth") != 8
        or type(header.get("bit_depth")) is not int
        or header.get("color_type") != 2
        or type(header.get("color_type")) is not int
        or type(header.get("height")) is not int
        or header["height"] <= 0
        or type(header.get("width")) is not int
        or header["width"] <= 0
    ):
        raise ValueError("Quality guide PNG header is invalid")


def _validate_quality_geometry(
    value: object,
    *,
    expected_index: int,
) -> Literal["relative", "metric"]:
    if not isinstance(value, dict):
        raise ValueError("Quality geometry identity is malformed")
    payload_kind = value.get("payload_kind")
    if payload_kind == "canonical_png":
        geometry = _quality_identity_common(
            value,
            expected_index=expected_index,
            suffix=".png",
        )
        if set(geometry) != {
            "byte_count",
            "frame_name",
            "payload_kind",
            "png_header",
            "relative_path",
            "sha256",
        }:
            raise ValueError("relative Quality geometry schema is invalid")
        header = geometry.get("png_header")
        if (
            not isinstance(header, dict)
            or set(header) != {"bit_depth", "color_type", "height", "width"}
            or header.get("bit_depth") != 16
            or type(header.get("bit_depth")) is not int
            or header.get("color_type") != 0
            or type(header.get("color_type")) is not int
            or type(header.get("height")) is not int
            or header["height"] <= 0
            or type(header.get("width")) is not int
            or header["width"] <= 0
        ):
            raise ValueError("relative Quality geometry header is invalid")
        return "relative"
    geometry = _quality_identity_common(
        value,
        expected_index=expected_index,
        suffix=".npz",
    )
    if (
        set(geometry)
        != {
            "archive_form",
            "byte_count",
            "focal_x_normalized_float32_bits",
            "frame_name",
            "inverse_depth_dtype",
            "members",
            "native_shape",
            "payload_contract",
            "payload_kind",
            "relative_path",
            "sha256",
            "valid_dtype",
        }
        or payload_kind != "metric_npz"
        or geometry.get("payload_contract") != "metric-npz-v1"
        or geometry.get("archive_form")
        not in {
            "owned-zip32-v1",
            "numpy-force-zip64-local-u32-v1",
            "numpy-force-zip64-local-sentinel-v1",
        }
        or geometry.get("inverse_depth_dtype") != "float32"
        or geometry.get("valid_dtype") != "bool"
        or geometry.get("members") != ["inverse_depth.npy", "valid.npy", "focal_x_normalized.npy"]
        or not isinstance(geometry.get("native_shape"), list)
        or len(geometry["native_shape"]) != 2
        or any(type(value) is not int or value <= 0 for value in geometry["native_shape"])
        or not isinstance(geometry.get("focal_x_normalized_float32_bits"), str)
        or re.fullmatch(
            r"[0-9a-f]{8}",
            geometry["focal_x_normalized_float32_bits"],
            flags=re.ASCII,
        )
        is None
    ):
        raise ValueError("metric Quality geometry schema is invalid")
    return "metric"


def project_quality_input_manifest(  # noqa: C901
    path: Path,
    *,
    acquired: AcquiredRoot | None = None,
) -> QualityInputMetadataProjection:
    """Strictly audit a Quality content manifest without retaining its arrays."""

    path = Path(path)
    if acquired is None:
        with AcquiredRoot(path.parent.parent) as owned:
            return project_quality_input_manifest(path, acquired=owned)
    fields: dict[str, Any] = {}
    fingerprint = ""
    frame_count = 0
    first_frame_index: int | None = None
    geometry_count = 0
    geometry_kind: Literal["relative", "metric"] | None = None
    guide_count = 0
    geometry_digest = hashlib.sha256(b'{"geometry":[')
    guide_digest = hashlib.sha256(b"[")
    with acquired.open_regular(path) as handle:
        reader = _ExactCanonicalReader(handle)
        reader.expect(b"{")
        for position, key in enumerate(sorted(_QUALITY_CONTENT_KEYS)):
            prefix = (b"," if position else b"") + canonical_json_bytes(key) + b":"
            reader.expect(prefix, fingerprinted=(key != "fingerprint"))
            if key == "fingerprint":
                fingerprint = reader.canonical_string(
                    maximum_bytes=64,
                    fingerprinted=False,
                )
                if not _is_hex64(fingerprint):
                    raise ValueError("Quality content fingerprint is malformed")
            elif key == "frame_names":
                reader.expect(b"[")
                while reader.peek() != b"]":
                    if frame_count >= STEREO_CONTROL_FRAME_CAP:
                        raise StereoControlBudgetError(
                            "Quality frame-name sequence exceeds the control cap"
                        )
                    if frame_count:
                        reader.expect(b",")
                    frame_index = _quality_frame_index(reader.canonical_string(maximum_bytes=1024))
                    if first_frame_index is None:
                        first_frame_index = frame_index
                    elif frame_index != first_frame_index + frame_count:
                        raise ValueError("Quality content frame sequence has a gap")
                    frame_count += 1
                reader.expect(b"]")
            elif key == "geometry":
                reader.expect(b"[")
                while reader.peek() != b"]":
                    if geometry_count >= STEREO_CONTROL_FRAME_CAP:
                        raise StereoControlBudgetError(
                            "Quality geometry sequence exceeds the control cap"
                        )
                    if geometry_count:
                        reader.expect(b",")
                        geometry_digest.update(b",")
                    value = reader.canonical_value(maximum_bytes=_DIAGNOSTICS_SMALL_FIELD_MAX_BYTES)
                    if first_frame_index is None:
                        raise ValueError("Quality geometry precedes its frame sequence")
                    kind = _validate_quality_geometry(
                        value,
                        expected_index=first_frame_index + geometry_count,
                    )
                    if geometry_kind is not None and kind != geometry_kind:
                        raise ValueError("Quality geometry modes are mixed")
                    geometry_kind = kind
                    geometry_digest.update(canonical_json_bytes(value))
                    geometry_count += 1
                reader.expect(b"]")
            elif key == "guides":
                reader.expect(b"[")
                while reader.peek() != b"]":
                    if guide_count >= STEREO_CONTROL_FRAME_CAP:
                        raise StereoControlBudgetError(
                            "Quality guide sequence exceeds the control cap"
                        )
                    if guide_count:
                        reader.expect(b",")
                        guide_digest.update(b",")
                    value = reader.canonical_value(maximum_bytes=_DIAGNOSTICS_SMALL_FIELD_MAX_BYTES)
                    if first_frame_index is None:
                        raise ValueError("Quality guides precede their frame sequence")
                    _validate_quality_guide(
                        value,
                        expected_index=first_frame_index + guide_count,
                    )
                    guide_digest.update(canonical_json_bytes(value))
                    guide_count += 1
                reader.expect(b"]")
            else:
                fields[key] = reader.canonical_value(
                    maximum_bytes=_DIAGNOSTICS_SMALL_FIELD_MAX_BYTES
                )
        reader.expect(b"}")
        if reader.peek():
            raise ValueError("Quality content manifest has trailing bytes")
    geometry_mode = fields.get("geometry_mode")
    scale_bits = fields.get("relative_encoding_scale_float32_bits")
    if (
        frame_count <= 0
        or first_frame_index is None
        or geometry_count != frame_count
        or guide_count != frame_count
        or fields.get("schema_version") != 1
        or type(fields.get("schema_version")) is not int
        or fields.get("algorithm_version") != "quality-input-content-v4"
        or geometry_mode not in {"relative", "metric"}
        or geometry_kind != geometry_mode
        or (
            geometry_mode == "relative"
            and (
                not isinstance(scale_bits, str)
                or re.fullmatch(r"[0-9a-f]{8}", scale_bits, flags=re.ASCII) is None
            )
        )
        or geometry_mode == "metric"
        and scale_bits is not None
    ):
        raise ValueError("Quality content scalar contract is invalid")
    guide_digest.update(b"]")
    geometry_digest.update(
        b'],"geometry_mode":'
        + canonical_json_bytes(geometry_mode)
        + b',"relative_encoding_scale_float32_bits":'
        + canonical_json_bytes(scale_bits)
        + b"}"
    )
    source_fingerprint = fields.get("source_guide_fingerprint")
    native_fingerprint = fields.get("native_geometry_fingerprint")
    if (
        source_fingerprint != guide_digest.hexdigest()
        or native_fingerprint != geometry_digest.hexdigest()
        or fingerprint != reader.self_digest.hexdigest()
    ):
        raise ValueError("Quality content derived fingerprint is invalid")
    return QualityInputMetadataProjection(
        path=path,
        geometry_mode=geometry_mode,
        source_guide_fingerprint=source_fingerprint,
        native_geometry_fingerprint=native_fingerprint,
        fingerprint=fingerprint,
        frame_count=frame_count,
        first_frame_index=first_frame_index,
        raw_sha256=reader.raw_digest.hexdigest(),
        raw_byte_count=reader.raw_byte_count,
    )


def validate_quality_input_binding(
    projection: DiagnosticsMetadataProjection,
    *,
    acquired: AcquiredRoot | None = None,
) -> QualityInputMetadataProjection | None:
    """Validate the retained Quality content identity bound by diagnostics."""

    projection.validate_state_contract()
    if projection.static_fields["render_mode"] == "fast":
        return None
    content = project_quality_input_manifest(
        projection.path.parent / "quality_input_manifest.json",
        acquired=acquired,
    )
    fields = projection.static_fields
    if (
        content.raw_sha256 != fields["quality_input_manifest_sha256"]
        or content.source_guide_fingerprint != fields["source_guide_fingerprint"]
        or content.native_geometry_fingerprint != fields["native_geometry_fingerprint"]
        or content.geometry_mode != fields["geometry_mode"]
        or content.frame_count != projection.frame_count
        or any(
            diagnostics_name != content_name
            for diagnostics_name, content_name in zip(
                projection.iter_frame_names(acquired=acquired),
                content.iter_frame_names(),
                strict=True,
            )
        )
    ):
        raise ValueError("Quality diagnostics content binding is invalid")
    return content


def _streaming_root_metadata_status(
    context: StreamingDiagnosticsStageContext,
    *,
    acquired: AcquiredRoot | None = None,
) -> Literal["building", "complete"]:
    if acquired is None:
        with AcquiredRoot(context.job_root) as owned:
            return _streaming_root_metadata_status(context, acquired=owned)
    path = context.job_root / "04_stereo_diagnostics" / "metadata.json"
    with acquired.open_regular(path) as handle:
        reader = _ExactCanonicalReader(handle)
        fields = {
            "algorithm_version": "stereo-coverage-sidecar-v1",
            "final_video_manifest_sha256": None,
            "frame_manifest_schema_version": 1,
            "geometry_mode": context.geometry_mode,
            "mask_payloads_enabled": context.mask_payloads_enabled,
            "native_geometry_fingerprint": context.native_geometry_fingerprint,
            "occlusion_fill": context.occlusion_fill,
            "prune_entries": None,
            "pruned_from": None,
            "quality_input_manifest_sha256": context.quality_input_manifest_sha256,
            "render_mode": context.render_mode,
            "render_shape": [context.render_shape[0], context.render_shape[1]],
            "rgb_stage_fingerprint": context.rgb_stage_fingerprint,
            "schema_version": 1,
            "source_guide_fingerprint": context.source_guide_fingerprint,
            "stats_schema_version": 1,
        }
        ordered_keys = sorted(
            set(fields)
            | {
                "fingerprint",
                "frame_names",
                "frames_jsonl_sha256",
                "metric_clamp_summary_sha256",
                "ordered_frame_manifest_fingerprint",
                "status",
                "summary_sha256",
            }
        )
        hashes: dict[str, str | None] = {}
        fingerprint = ""
        status = ""
        reader.expect(b"{")
        for position, key in enumerate(ordered_keys):
            prefix = (b"," if position else b"") + canonical_json_bytes(key) + b":"
            reader.expect(prefix, fingerprinted=(key != "fingerprint"))
            if key == "fingerprint":
                fingerprint = reader.canonical_string(
                    maximum_bytes=64,
                    fingerprinted=False,
                )
                if not _is_hex64(fingerprint):
                    raise ValueError("diagnostics metadata fingerprint is malformed")
            elif key == "frame_names":
                reader.expect(b"[")
                for index, frame_name in enumerate(_iter_streaming_frame_names(context)):
                    if index:
                        reader.expect(b",")
                    reader.expect(canonical_json_bytes(frame_name))
                reader.expect(b"]")
            elif key in {
                "frames_jsonl_sha256",
                "metric_clamp_summary_sha256",
                "ordered_frame_manifest_fingerprint",
                "summary_sha256",
            }:
                hashes[key] = reader.optional_hex64()
            elif key == "status":
                status = reader.canonical_string(maximum_bytes=16)
            else:
                reader.expect(canonical_json_bytes(fields[key]))
        reader.expect(b"}")
        if handle.read(1):
            raise ValueError("diagnostics metadata has trailing bytes")
    if fingerprint != reader.self_digest.hexdigest():
        raise ValueError("diagnostics metadata self fingerprint changed")
    if status == "building":
        if any(value is not None for value in hashes.values()):
            raise ValueError("building diagnostics metadata declares aggregates")
        return "building"
    if status != "complete":
        raise ValueError("diagnostics metadata status is not reusable")
    required = {
        "frames_jsonl_sha256",
        "ordered_frame_manifest_fingerprint",
        "summary_sha256",
    }
    if any(hashes[key] is None for key in required):
        raise ValueError("complete diagnostics metadata omits aggregate identity")
    metric_hash = hashes["metric_clamp_summary_sha256"]
    if (context.geometry_mode == "metric") != (metric_hash is not None):
        raise ValueError("diagnostics metric aggregate identity is inconsistent")
    return "complete"


def _streaming_summary_document(
    context: StreamingDiagnosticsStageContext,
    *,
    jsonl_path: Path,
    ordered_fingerprint: str,
    jsonl_sha256: str,
    acquired: AcquiredRoot,
) -> StreamedCanonicalValue:
    def replay_fractions():
        for stats in _iter_streaming_jsonl(jsonl_path, context, acquired=acquired):
            projection = stats["metric_projection"]
            if not isinstance(projection, dict):
                raise ValueError("metric diagnostics require projection statistics")
            fraction = float(projection["clamped_fraction"])
            del projection, stats
            yield fraction

    summary = build_diagnostics_summary(
        _iter_streaming_jsonl(jsonl_path, context, acquired=acquired),
        frame_name_replay=lambda: _iter_streaming_frame_names(context),
        frame_count=context.frame_count,
        clamped_fraction_replay=(replay_fractions if context.geometry_mode == "metric" else None),
        render_shape=context.render_shape,
        render_mode=context.render_mode,
        geometry_mode=context.geometry_mode,
        occlusion_fill=context.occlusion_fill,
        ordered_frame_manifest_fingerprint=ordered_fingerprint,
        frames_jsonl_sha256=jsonl_sha256,
    )
    return build_streamed_canonical_value(_streamable_diagnostics_value(summary))


def streaming_diagnostics_aggregates_match(
    context: StreamingDiagnosticsStageContext,
) -> bool:
    """Reconstruct exact aggregate identities without retaining variable arrays."""

    root = context.job_root / "04_stereo_diagnostics"
    jsonl_path = root / "stereo_coverage_frames.jsonl"
    try:
        with AcquiredRoot(context.job_root) as acquired:
            ordered_fingerprint, jsonl_sha256 = _streaming_jsonl_identity(
                context,
                output=None,
                acquired=acquired,
            )
            if _stream_file_sha256(jsonl_path, acquired=acquired) != jsonl_sha256:
                return False
            summary = _streaming_summary_document(
                context,
                jsonl_path=jsonl_path,
                ordered_fingerprint=ordered_fingerprint,
                jsonl_sha256=jsonl_sha256,
                acquired=acquired,
            )
            if not summary.matches(
                root / "stereo_coverage_summary.json",
                acquired=acquired,
            ):
                return False
            clamp: StreamedCanonicalValue | None = None
            if context.geometry_mode == "metric":
                assert context.left_rgb_parent is not None
                value = summary.value
                assert isinstance(value, dict)
                clamp = build_streamed_canonical_value(_compatibility_clamp_summary(value))
                if not clamp.matches(
                    context.left_rgb_parent / "clamp_summary.json",
                    acquired=acquired,
                ):
                    return False
            metadata = _streaming_diagnostics_metadata(
                context,
                status="complete",
                ordered_frame_manifest_fingerprint=ordered_fingerprint,
                frames_jsonl_sha256=jsonl_sha256,
                summary_sha256=summary.raw_sha256,
                metric_clamp_summary_sha256=(None if clamp is None else clamp.raw_sha256),
            )
            return metadata.matches(root / "metadata.json", acquired=acquired)
    except (OSError, ValueError, KeyError, TypeError, NoFollowPathError):
        return False


def classify_streaming_diagnostics_root(
    context: StreamingDiagnosticsStageContext,
) -> Literal["unclaimed", "building", "complete"]:
    """Accept frame reuse only under a durable root for the same generation."""

    try:
        with AcquiredRoot(context.job_root) as acquired:
            try:
                return _streaming_root_metadata_status(context, acquired=acquired)
            except (OSError, UnicodeDecodeError, ValueError, NoFollowPathError):
                if context.mask_payloads_enabled:
                    raise
                return _streaming_root_metadata_status(
                    replace(context, mask_payloads_enabled=True),
                    acquired=acquired,
                )
    except (OSError, UnicodeDecodeError, ValueError, NoFollowPathError):
        return "unclaimed"


def complete_streaming_diagnostics_stage(
    context: StreamingDiagnosticsStageContext,
    *,
    before_complete_publish: Callable[[], None] | None = None,
    fault: FaultHook | None = None,
) -> StreamingDiagnosticsCompletion:
    """Publish source-ordered aggregates with no resident per-frame graph."""

    root = context.job_root / "04_stereo_diagnostics"
    root.mkdir(parents=True, exist_ok=True)
    jsonl_path = root / "stereo_coverage_frames.jsonl"
    owned = OwnedSiblingTemporary.create(jsonl_path)
    try:
        _call_fault(fault, "before_frames_jsonl_temporary_write")
        ordered_fingerprint, jsonl_sha256 = _streaming_jsonl_identity(
            context,
            output=owned.handle,
        )
        _call_fault(fault, "after_frames_jsonl_temporary_write")
        owned.sync()
        _call_fault(fault, "after_frames_jsonl_temporary_sync")
        _call_fault(fault, "before_frames_jsonl_replace")
        owned.replace(
            after_replace=lambda: _call_fault(fault, "after_frames_jsonl_replace"),
        )
        _call_fault(fault, "after_frames_jsonl_directory_sync")
    finally:
        owned.cleanup()

    with AcquiredRoot(context.job_root) as acquired:
        summary = _streaming_summary_document(
            context,
            jsonl_path=jsonl_path,
            ordered_fingerprint=ordered_fingerprint,
            jsonl_sha256=jsonl_sha256,
            acquired=acquired,
        )
        summary.publish(
            root / "stereo_coverage_summary.json",
            fault=fault,
            label="root_summary",
        )
        clamp: StreamedCanonicalValue | None = None
        if context.geometry_mode == "metric":
            assert context.left_rgb_parent is not None
            value = summary.value
            assert isinstance(value, dict)
            clamp = build_streamed_canonical_value(_compatibility_clamp_summary(value))
            clamp.publish(
                context.left_rgb_parent / "clamp_summary.json",
                fault=fault,
                label="metric_clamp_summary",
            )
    if before_complete_publish is not None:
        before_complete_publish()
    metadata = _streaming_diagnostics_metadata(
        context,
        status="complete",
        ordered_frame_manifest_fingerprint=ordered_fingerprint,
        frames_jsonl_sha256=jsonl_sha256,
        summary_sha256=summary.raw_sha256,
        metric_clamp_summary_sha256=(None if clamp is None else clamp.raw_sha256),
    )
    metadata.publish(
        root / "metadata.json",
        fault=fault,
        label="root_metadata",
    )
    return StreamingDiagnosticsCompletion(
        metadata=metadata,
        summary=summary,
        metric_clamp_summary=clamp,
        ordered_frame_manifest_fingerprint=ordered_fingerprint,
        frames_jsonl_sha256=jsonl_sha256,
    )


def migrate_streaming_frame_manifest(
    context: StreamingDiagnosticsStageContext,
    frame_name: str,
    *,
    source_index: int | None = None,
    fault: FaultHook | None = None,
) -> None:
    """Migrate one mask-enabled frame and prune its now-unreferenced masks."""

    if context.mask_payloads_enabled:
        raise ValueError("mask manifest migration requires a disabled target policy")
    current = context.one_frame(frame_name)
    previous = context.one_frame(frame_name, masks=True)
    manifest, _raw, _stats, _stats_raw = _audit_frame_transaction(previous, frame_name)
    payloads = dict(manifest["payloads"])
    for name in ("left_coverage", "left_repair", "right_coverage", "right_repair"):
        payloads[name] = None
    migrated = dict(manifest)
    migrated["mask_payloads_enabled"] = False
    migrated["payloads"] = payloads
    migrated["fingerprint"] = self_fingerprint(migrated)
    frame_dir = context.job_root / "04_stereo_diagnostics" / "frames" / frame_name
    _atomic_replace_raw(
        frame_dir / "manifest.json",
        canonical_json_bytes(migrated),
        fault=fault,
        label=("mask_migration" if source_index is None else f"mask_migration_{source_index}"),
    )
    _audit_frame_transaction(current, frame_name)


def prune_streaming_unreferenced_masks(
    context: StreamingDiagnosticsStageContext,
) -> None:
    """Delete old mask payloads only after every migration manifest is durable."""

    if context.mask_payloads_enabled:
        raise ValueError("mask pruning requires a disabled target policy")
    with AcquiredRoot(context.job_root) as acquired:
        for frame_name in _iter_streaming_frame_names(context):
            frame_dir = context.job_root / "04_stereo_diagnostics" / "frames" / frame_name
            with SafeTreeMutation(
                context.job_root,
                (frame_dir,),
                acquired=acquired,
            ) as mutation:
                _audit_frame_transaction(
                    context.one_frame(frame_name),
                    frame_name,
                    acquired=acquired,
                )
                for name in (
                    "left_coverage.png",
                    "left_repair.png",
                    "right_coverage.png",
                    "right_repair.png",
                ):
                    child = frame_dir / name
                    evidence = mutation.capture_regular_child(frame_dir, child)
                    mutation.unlink_child(
                        frame_dir,
                        child,
                        evidence=evidence,
                    )


def complete_diagnostics_stage(
    context: DiagnosticsStageContext,
    *,
    fault: FaultHook | None = None,
) -> dict[str, Any]:
    """Consolidate committed frames and publish complete metadata last."""

    root = context.job_root / "04_stereo_diagnostics"
    jsonl_path = root / "stereo_coverage_frames.jsonl"
    owned = OwnedSiblingTemporary.create(jsonl_path)
    ordered_digest = hashlib.sha256()
    ordered_digest.update(b"[")
    jsonl_digest = hashlib.sha256()
    try:
        for index, frame_name in enumerate(context.frame_names):
            _manifest, raw_manifest, stats, _stats_raw = _audit_frame_transaction(
                context,
                frame_name,
            )
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
            if owned.handle.write(line) != len(line):
                raise OSError("short diagnostics JSONL write")
            jsonl_digest.update(line)
        _call_fault(fault, "after_frames_jsonl_temporary_write")
        owned.sync()
        _call_fault(fault, "after_frames_jsonl_temporary_sync")
        ordered_digest.update(b"]")
        ordered_fingerprint = ordered_digest.hexdigest()
        jsonl_sha = jsonl_digest.hexdigest()
        _call_fault(fault, "before_frames_jsonl_replace")
        owned.replace(
            after_replace=lambda: _call_fault(fault, "after_frames_jsonl_replace"),
        )
        _call_fault(fault, "after_frames_jsonl_directory_sync")
    finally:
        owned.cleanup()

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
        manifest, _raw, _stats, _stats_raw = _audit_frame_transaction(previous, frame_name)
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
    *,
    acquired: AcquiredRoot,
) -> bool:
    root = context.job_root / "04_stereo_diagnostics"
    try:
        jsonl_sha256 = _stream_file_sha256(
            root / "stereo_coverage_frames.jsonl",
            acquired=acquired,
        )
        summary, summary_raw = _read_canonical_object(
            root / "stereo_coverage_summary.json",
            acquired=acquired,
        )
        if jsonl_sha256 != metadata["frames_jsonl_sha256"]:
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
            clamp_sha256 = _stream_file_sha256(
                context.left_rgb_parent / "clamp_summary.json",
                acquired=acquired,
            )
            if clamp_sha256 != metadata["metric_clamp_summary_sha256"]:
                return False
        elif metadata["metric_clamp_summary_sha256"] is not None:
            return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def _payload_pruned_matches(  # noqa: C901
    metadata: dict[str, Any],
    context: DiagnosticsStageContext,
    *,
    acquired: AcquiredRoot,
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
        jsonl_sha256 = _stream_file_sha256(
            root / "stereo_coverage_frames.jsonl",
            acquired=acquired,
        )
        summary, summary_raw = _read_canonical_object(
            root / "stereo_coverage_summary.json",
            acquired=acquired,
        )
        with acquired.open_regular(context.job_root / "final_video_manifest.json") as handle:
            final_manifest_raw = handle.read()
        if jsonl_sha256 != metadata["frames_jsonl_sha256"]:
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
            content_sha256 = _stream_file_sha256(
                root / "quality_input_manifest.json",
                acquired=acquired,
            )
            if content_sha256 != context.quality_input_manifest_sha256:
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
    with AcquiredRoot(context.job_root) as acquired:
        try:
            metadata, _raw = _read_canonical_object(metadata_path, acquired=acquired)
        except (OSError, ValueError):
            metadata = {}
        if metadata.get("status") == "payload_pruned":
            historical_valid = _payload_pruned_matches(
                metadata,
                context,
                acquired=acquired,
            )
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
            replace(context, mask_payloads_enabled=True)
            if not context.mask_payloads_enabled
            else None
        )
        for frame_name in context.frame_names:
            try:
                _audit_frame_transaction(
                    context,
                    frame_name,
                    acquired=acquired,
                )
            except (OSError, ValueError, KeyError, TypeError):
                if previous_mask_context is not None:
                    try:
                        _audit_frame_transaction(
                            previous_mask_context,
                            frame_name,
                            acquired=acquired,
                        )
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
        aggregates_valid = metadata.get("status") == "complete" and _aggregates_match(
            metadata,
            context,
            acquired=acquired,
        )
        return DiagnosticsAudit(
            "complete" if aggregates_valid else "building",
            tuple(reusable),
            (),
            aggregates_valid,
        )
