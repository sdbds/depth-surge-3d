"""Bounded one-frame metric content ownership for Quality manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import struct
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal, Sequence

import numpy as np

from ..core.resource_errors import QualityHostBudgetError
from . import metric_npz
from .durable_publish import OwnedSiblingTemporary
from .nofollow_io import AcquiredRoot, open_regular_beneath
from .streamed_canonical_json import (
    StreamedCanonicalObject,
    build_streamed_canonical_object_arrays,
)


if TYPE_CHECKING:
    from ..processing.frames.stereo_input_manifest import StereoInputManifestV1


QUALITY_CONTENT_STREAM_BYTES = metric_npz.METRIC_NPZ_WRITE_STREAM_BYTES
QUALITY_METRIC_AUDIT_PHASE = "metric-content-audit"
_CONTENT_READ_BYTES = 1024 * 1024
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class MetricPayloadIdentity:
    """Scalar identity retained after an owned audit frame is released."""

    relative_path: str
    sha256: str
    byte_count: int
    archive_form: metric_npz.ArchiveForm
    native_shape: tuple[int, int]
    focal_x_normalized_float32_bits: str


@dataclass
class OwnedMetricAuditFrame:
    """Exactly one inverse array plus one validity array for one audited file."""

    path: Path
    inverse_depth: np.ndarray | None
    valid: np.ndarray | None
    focal_x_normalized: np.float32
    archive_form: metric_npz.ArchiveForm
    sha256: str
    byte_count: int
    maximum_stream_bytes: int
    stream_workspace_peak_bytes: int
    value_scratch_peak_bytes: int
    _released: bool = False

    @property
    def released(self) -> bool:
        return self._released

    @property
    def owned_payload_bytes(self) -> int:
        if self.inverse_depth is None or self.valid is None:
            return 0
        return int(self.inverse_depth.nbytes + self.valid.nbytes)

    @property
    def native_shape(self) -> tuple[int, int]:
        if self.inverse_depth is None:
            raise RuntimeError("metric audit frame is released")
        return int(self.inverse_depth.shape[0]), int(self.inverse_depth.shape[1])

    def identity(self) -> MetricPayloadIdentity:
        if self._released:
            raise RuntimeError("metric audit frame is released")
        focal_bits = struct.unpack(
            "<I",
            struct.pack("<f", float(self.focal_x_normalized)),
        )[0]
        return MetricPayloadIdentity(
            relative_path=self.path.as_posix(),
            sha256=self.sha256,
            byte_count=self.byte_count,
            archive_form=self.archive_form,
            native_shape=self.native_shape,
            focal_x_normalized_float32_bits=f"{focal_bits:08x}",
        )

    def release(self) -> None:
        if self._released:
            return
        self.inverse_depth = None
        self.valid = None
        self._released = True

    def __enter__(self) -> OwnedMetricAuditFrame:
        if self._released:
            raise RuntimeError("metric audit frame is released")
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


@dataclass
class MetricSequenceAuditTrace:
    """Lifetime trace proving a capacity-one metric audit phase."""

    events: list[str] = field(default_factory=list)
    active_frames_peak: int = 0
    active_payload_bytes_peak: int = 0
    maximum_stream_bytes: int = 0
    stream_workspace_peak_bytes: int = 0
    maximum_value_scratch_bytes: int = 0


@dataclass(frozen=True)
class MetricSequenceAuditResult:
    identities: tuple[MetricPayloadIdentity, ...]
    trace: MetricSequenceAuditTrace


def _default_allocator(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    return np.empty(shape, dtype=dtype, order="C")


def audit_metric_payload(
    path: Path,
    *,
    expected_shape: tuple[int, int],
    allocator=_default_allocator,
    job_root: Path | None = None,
) -> OwnedMetricAuditFrame:
    """Audit one regular payload without ``zipfile``, ``np.load``, or a copy constructor."""

    path = Path(path)
    requested_bytes = 5 * int(expected_shape[0]) * int(expected_shape[1])
    try:
        audit = metric_npz.audit_metric_npz(
            path,
            expected_shape=expected_shape,
            allocator=allocator,
            job_root=job_root,
        )
    except metric_npz.MetricNpzAllocationError as error:
        raise QualityHostBudgetError(
            QUALITY_METRIC_AUDIT_PHASE,
            path,
            requested_bytes,
        ) from error
    return OwnedMetricAuditFrame(
        path=path,
        inverse_depth=audit.inverse_depth,
        valid=audit.valid,
        focal_x_normalized=audit.focal_x_normalized,
        archive_form=audit.layout.archive_form,
        sha256=audit.sha256,
        byte_count=audit.byte_count,
        maximum_stream_bytes=audit.maximum_stream_bytes,
        stream_workspace_peak_bytes=audit.stream_workspace_peak_bytes,
        value_scratch_peak_bytes=audit.value_scratch_peak_bytes,
    )


def _iter_paths(paths: Iterable[Path]) -> Iterator[Path]:
    for path in paths:
        yield Path(path)


def audit_metric_sequence(
    paths: Iterable[Path],
    *,
    expected_shape: tuple[int, int],
) -> MetricSequenceAuditResult:
    """Audit in source order and drop every 5*G ownership before the next open."""

    identities: list[MetricPayloadIdentity] = []
    trace = MetricSequenceAuditTrace()
    active_frames = 0
    for index, path in enumerate(_iter_paths(paths)):
        trace.events.append(f"open:{index}")
        frame = audit_metric_payload(path, expected_shape=expected_shape)
        active_frames += 1
        trace.active_frames_peak = max(trace.active_frames_peak, active_frames)
        trace.active_payload_bytes_peak = max(
            trace.active_payload_bytes_peak,
            frame.owned_payload_bytes,
        )
        trace.maximum_stream_bytes = max(
            trace.maximum_stream_bytes,
            frame.maximum_stream_bytes,
        )
        trace.stream_workspace_peak_bytes = max(
            trace.stream_workspace_peak_bytes,
            frame.stream_workspace_peak_bytes,
        )
        trace.maximum_value_scratch_bytes = max(
            trace.maximum_value_scratch_bytes,
            frame.value_scratch_peak_bytes,
        )
        try:
            identities.append(frame.identity())
        finally:
            frame.release()
            active_frames -= 1
            trace.events.append(f"release:{index}")
    return MetricSequenceAuditResult(tuple(identities), trace)


@dataclass(frozen=True)
class QualityInputManifestResult:
    payload: dict[str, Any]
    raw: bytes
    raw_sha256: str


@dataclass(frozen=True, slots=True)
class StreamedQualityInputManifest:
    """Scalar identity plus a replayable canonical Quality content manifest."""

    document: StreamedCanonicalObject
    source_guide_fingerprint: str
    native_geometry_fingerprint: str

    @property
    def raw_sha256(self) -> str:
        return self.document.raw_sha256

    @property
    def raw_byte_count(self) -> int:
        return self.document.raw_byte_count

    @property
    def fingerprint(self) -> str:
        assert self.document.fingerprint is not None
        return self.document.fingerprint

    def publish(self, path: Path) -> None:
        self.document.publish(path)

    def matches(self, path: Path, *, acquired: AcquiredRoot | None = None) -> bool:
        return self.document.matches(path, acquired=acquired)


def _array_fingerprint(identities: Iterable[object]) -> str:
    digest = hashlib.sha256(b"[")
    for index, identity in enumerate(identities):
        if index:
            digest.update(b",")
        digest.update(_canonical_json_bytes(identity))
    digest.update(b"]")
    return digest.hexdigest()


def _audit_source_order(manifest: "StereoInputManifestV1") -> None:
    audited_count = 0
    for record in manifest.replay_records():
        if record.source_index != audited_count:
            raise ValueError("Quality input manifest source order changed")
        audited_count += 1
    if audited_count != manifest.frame_count:
        raise ValueError("Quality input manifest ended before its declared count")


def build_streamed_quality_input_manifest(
    manifest: Any,
    *,
    validate_payloads: bool = True,
) -> StreamedQualityInputManifest:
    """Freeze declared identities, optionally after the bounded semantic audit."""

    from ..processing.frames.stereo_input_manifest import StereoInputManifestV1

    if not isinstance(manifest, StereoInputManifestV1):
        raise TypeError("Quality content requires a stereo input manifest descriptor")

    if not isinstance(validate_payloads, bool):
        raise TypeError("validate_payloads must be boolean")
    if validate_payloads:
        _audit_source_order(manifest)

    def frame_names():
        for record in manifest.replay_declared_records():
            yield record.frame_name

    def guides():
        for record in manifest.replay_declared_records():
            yield record.guide_identity

    def geometry():
        for record in manifest.replay_declared_records():
            yield record.geometry_identity

    source_guide_fingerprint = _array_fingerprint(guides())

    scale_bits = manifest.relative_encoding_scale_float32_bits
    native_geometry_fingerprint = build_streamed_canonical_object_arrays(
        static_fields={
            "geometry_mode": manifest.geometry_mode,
            "relative_encoding_scale_float32_bits": scale_bits,
        },
        array_items={"geometry": geometry},
    ).raw_sha256
    document = build_streamed_canonical_object_arrays(
        static_fields={
            "algorithm_version": "quality-input-content-v4",
            "geometry_mode": manifest.geometry_mode,
            "native_geometry_fingerprint": native_geometry_fingerprint,
            "relative_encoding_scale_float32_bits": scale_bits,
            "schema_version": 1,
            "source_guide_fingerprint": source_guide_fingerprint,
        },
        array_items={
            "frame_names": frame_names,
            "geometry": geometry,
            "guides": guides,
        },
        fingerprint_key="fingerprint",
    )
    return StreamedQualityInputManifest(
        document=document,
        source_guide_fingerprint=source_guide_fingerprint,
        native_geometry_fingerprint=native_geometry_fingerprint,
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _contained_relative_path(root: Path, path: Path) -> str:
    root = root.resolve()
    if path.is_symlink():
        raise ValueError(f"Quality input cannot be a symlink: {path}")
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Quality input is outside the job root: {path}") from error
    return relative.as_posix()


def _hash_regular_file(root: Path, path: Path) -> tuple[str, int, bytes]:
    digest = hashlib.sha256()
    header = bytearray()
    total = 0
    workspace = bytearray(_CONTENT_READ_BYTES)
    with open_regular_beneath(root, path) as handle:
        view = memoryview(workspace)
        while True:
            count = handle.readinto(workspace)
            if not count:
                break
            digest.update(view[:count])
            if len(header) < 33:
                needed = 33 - len(header)
                header.extend(view[: min(count, needed)])
            total += count
    return digest.hexdigest(), total, bytes(header)


def _png_header(raw: bytes) -> dict[str, int]:
    if (
        len(raw) < 33
        or raw[:8] != _PNG_SIGNATURE
        or raw[8:12] != b"\x00\x00\x00\r"
        or raw[12:16] != b"IHDR"
    ):
        raise ValueError("Quality input PNG has an invalid IHDR")
    width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", raw[16:29]
    )
    if width <= 0 or height <= 0 or compression != 0 or filtering != 0 or interlace not in {0, 1}:
        raise ValueError("Quality input PNG has an unsupported IHDR")
    return {
        "width": int(width),
        "height": int(height),
        "bit_depth": int(bit_depth),
        "color_type": int(color_type),
    }


def _png_identity(
    root: Path,
    path: Path,
    frame_name: str,
    *,
    expected_bit_depth: int,
    expected_color_type: int,
) -> dict[str, Any]:
    sha256, byte_count, raw_header = _hash_regular_file(root, path)
    header = _png_header(raw_header)
    if header["bit_depth"] != expected_bit_depth or header["color_type"] != expected_color_type:
        raise ValueError(f"Quality input PNG format is invalid: {path}")
    if path.name != f"{frame_name}.png":
        raise ValueError("Quality input filename does not match its frame name")
    return {
        "frame_name": frame_name,
        "relative_path": _contained_relative_path(root, path),
        "sha256": sha256,
        "byte_count": byte_count,
        "png_header": header,
    }


def _float32_bits(value: np.float32) -> str:
    scalar = np.float32(value)
    if not np.isfinite(scalar) or scalar <= np.float32(0.0):
        raise ValueError("Quality input float32 scalar must be finite and positive")
    return f"{struct.unpack('<I', struct.pack('<f', float(scalar)))[0]:08x}"


def build_quality_input_manifest(  # noqa: C901
    *,
    job_root: Path,
    frame_files: Sequence[Path],
    geometry_files: Sequence[Path],
    geometry_mode: Literal["relative", "metric"],
    relative_encoding_scale: np.float32 | None,
    metric_native_shape: tuple[int, int] | None,
) -> QualityInputManifestResult:
    """Build the complete content-only Quality identity in source order."""

    root = Path(job_root).resolve()
    frames = [Path(path) for path in frame_files]
    geometry_paths = [Path(path) for path in geometry_files]
    if not frames or len(frames) != len(geometry_paths):
        raise ValueError("Quality guide and geometry sequences must have equal positive length")
    frame_names = [path.stem for path in frames]
    if len(set(frame_names)) != len(frame_names):
        raise ValueError("Quality frame names must be unique")
    guides: list[dict[str, Any]] = []
    geometry: list[dict[str, Any]] = []
    for frame_name, guide_path, geometry_path in zip(
        frame_names,
        frames,
        geometry_paths,
    ):
        guides.append(
            _png_identity(
                root,
                guide_path,
                frame_name,
                expected_bit_depth=8,
                expected_color_type=2,
            )
        )
        if geometry_mode == "relative":
            png_identity = _png_identity(
                root,
                geometry_path,
                frame_name,
                expected_bit_depth=16,
                expected_color_type=0,
            )
            png_identity["payload_kind"] = "canonical_png"
            geometry.append(png_identity)
        elif geometry_mode == "metric":
            if metric_native_shape is None:
                raise ValueError("Metric Quality inputs require native shape")
            with audit_metric_payload(
                geometry_path,
                expected_shape=metric_native_shape,
                job_root=root,
            ) as audited:
                metric_identity = audited.identity()
                if geometry_path.name != f"{frame_name}.npz":
                    raise ValueError("Metric filename does not match its frame name")
                geometry.append(
                    {
                        "frame_name": frame_name,
                        "relative_path": _contained_relative_path(root, geometry_path),
                        "sha256": metric_identity.sha256,
                        "byte_count": metric_identity.byte_count,
                        "payload_kind": "metric_npz",
                        "native_shape": list(metric_identity.native_shape),
                        "payload_contract": "metric-npz-v1",
                        "archive_form": metric_identity.archive_form,
                        "inverse_depth_dtype": "float32",
                        "valid_dtype": "bool",
                        "focal_x_normalized_float32_bits": (
                            metric_identity.focal_x_normalized_float32_bits
                        ),
                        "members": [
                            "inverse_depth.npy",
                            "valid.npy",
                            "focal_x_normalized.npy",
                        ],
                    }
                )
        else:
            raise ValueError("Quality geometry_mode must be relative or metric")
    if geometry_mode == "relative":
        if relative_encoding_scale is None:
            raise ValueError("Relative Quality inputs require encoding scale")
        scale_bits: str | None = _float32_bits(relative_encoding_scale)
    else:
        if relative_encoding_scale is not None:
            raise ValueError("Metric Quality inputs cannot carry relative encoding scale")
        scale_bits = None
    source_guide_fingerprint = hashlib.sha256(_canonical_json_bytes(guides)).hexdigest()
    native_geometry_fingerprint = hashlib.sha256(
        _canonical_json_bytes(
            {
                "geometry_mode": geometry_mode,
                "relative_encoding_scale_float32_bits": scale_bits,
                "geometry": geometry,
            }
        )
    ).hexdigest()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "algorithm_version": "quality-input-content-v4",
        "geometry_mode": geometry_mode,
        "frame_names": frame_names,
        "guides": guides,
        "geometry": geometry,
        "relative_encoding_scale_float32_bits": scale_bits,
        "source_guide_fingerprint": source_guide_fingerprint,
        "native_geometry_fingerprint": native_geometry_fingerprint,
    }
    payload["fingerprint"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    raw = _canonical_json_bytes(payload)
    return QualityInputManifestResult(
        payload=payload,
        raw=raw,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
    )


def publish_quality_input_manifest(
    path: Path,
    result: QualityInputManifestResult,
) -> None:
    """Atomically publish one already-audited canonical content manifest."""

    if not isinstance(result, QualityInputManifestResult):
        raise TypeError("result must be QualityInputManifestResult")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    owned = OwnedSiblingTemporary.create(path)
    try:
        if owned.handle.write(result.raw) != len(result.raw):
            raise OSError("short Quality input manifest write")
        owned.sync()
        owned.replace()
    finally:
        owned.cleanup()
