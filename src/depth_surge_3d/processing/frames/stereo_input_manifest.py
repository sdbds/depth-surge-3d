"""Durable canonical input manifest for the replayable stereo stage."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
import struct
from typing import Any, BinaryIO, cast, Iterable, Iterator, Literal, Mapping, Sequence
import zlib

import numpy as np

from ...core.render_disparity import validate_render_disparity_input
from ...io import metric_npz
from ...io.durable_publish import OwnedSiblingTemporary
from ...io.nofollow_io import AcquiredRoot, open_regular_beneath
from .stereo_control import (
    ReplayableStereoProvider,
    StereoControlBudgetError,
    StereoControlStateError,
    StereoInputIdentityMismatch,
    StereoProviderGeneration,
    StereoProviderItem,
    STEREO_CONTROL_FRAME_CAP,
)


STEREO_INPUT_MANIFEST_SCHEMA_VERSION = 1
STEREO_INPUT_MANIFEST_ALGORITHM_VERSION = "stereo-input-sequence-v1"
STEREO_INPUT_MANIFEST_LINE_BYTES = 64 * 1024
STEREO_INPUT_PAYLOAD_STREAM_BYTES = 1024 * 1024

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_HEX_64 = frozenset("0123456789abcdef")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_line(value: object) -> bytes:
    raw = _canonical_json_bytes(value) + b"\n"
    if len(raw) > STEREO_INPUT_MANIFEST_LINE_BYTES:
        raise StereoControlBudgetError("stereo input manifest line exceeds 64 KiB")
    return raw


def _is_hex64(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_64 for character in value)
    )


def _canonical_frame_index(stem: str) -> int:
    if not isinstance(stem, str) or not stem.startswith("frame_"):
        raise StereoControlStateError("stereo input frame stem is noncanonical")
    decimal = stem[6:]
    if not decimal or not decimal.isascii() or not decimal.isdecimal():
        raise StereoControlStateError("stereo input frame stem is noncanonical")
    index = int(decimal)
    expected = str(index).zfill(max(6, len(str(index))))
    if decimal != expected:
        raise StereoControlStateError("stereo input frame stem is noncanonical")
    return index


def _regular_file(
    path: Path,
    *,
    job_root: Path | None = None,
    acquired: AcquiredRoot | None = None,
) -> tuple[BinaryIO, os.stat_result]:
    if acquired is not None:
        if job_root is not None and acquired.root != Path(os.path.abspath(os.fspath(job_root))):
            raise StereoControlStateError("stereo input acquired the wrong job root")
        handle = acquired.open_regular(path)
        return handle, os.fstat(handle.fileno())
    if job_root is not None:
        handle = open_regular_beneath(job_root, path)
        return handle, os.fstat(handle.fileno())
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        attributes = getattr(metadata, "st_file_attributes", 0)
        if not stat.S_ISREG(metadata.st_mode) or attributes & getattr(
            stat,
            "FILE_ATTRIBUTE_REPARSE_POINT",
            0x400,
        ):
            raise StereoControlStateError(f"stereo input is not a regular file: {path}")
        return os.fdopen(descriptor, "rb", buffering=0, closefd=True), metadata
    except BaseException:
        os.close(descriptor)
        raise


def _file_version(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _stream_identity(
    path: Path,
    *,
    job_root: Path | None = None,
    acquired: AcquiredRoot | None = None,
) -> tuple[str, int, bytes]:
    digest = hashlib.sha256()
    byte_count = 0
    header = bytearray()
    workspace = bytearray(STEREO_INPUT_PAYLOAD_STREAM_BYTES)
    view = memoryview(workspace)
    with _regular_file(path, job_root=job_root, acquired=acquired)[0] as handle:
        while True:
            count = handle.readinto(workspace)
            if not count:
                break
            digest.update(view[:count])
            if len(header) < 33:
                needed = 33 - len(header)
                header.extend(view[: min(needed, count)])
            byte_count += count
    return digest.hexdigest(), byte_count, bytes(header)


def _parse_png_header(raw: bytes, path: Path) -> dict[str, int]:
    if len(raw) != 33 or raw[:8] != _PNG_SIGNATURE:
        raise StereoControlStateError(f"stereo input PNG signature is invalid: {path}")
    if raw[8:12] != b"\x00\x00\x00\r" or raw[12:16] != b"IHDR":
        raise StereoControlStateError(f"stereo input PNG IHDR is invalid: {path}")
    if zlib.crc32(raw[12:29]) & 0xFFFFFFFF != struct.unpack_from(">I", raw, 29)[0]:
        raise StereoControlStateError(f"stereo input PNG IHDR CRC is invalid: {path}")
    width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack_from(
        ">IIBBBBB",
        raw,
        16,
    )
    if width <= 0 or height <= 0 or compression != 0 or filtering != 0 or interlace not in {0, 1}:
        raise StereoControlStateError(f"stereo input PNG header is unsupported: {path}")
    return {
        "bit_depth": int(bit_depth),
        "color_type": int(color_type),
        "height": int(height),
        "width": int(width),
    }


def _contained_relative_path(root: Path, path: Path) -> str:
    path = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise StereoControlStateError(f"stereo input escapes the job root: {path}") from error
    if any(part in {"", ".", ".."} or "\\" in part or "\x00" in part for part in relative.parts):
        raise StereoControlStateError("stereo input relative path is noncanonical")
    return relative.as_posix()


def _png_identity(
    root: Path,
    path: Path,
    frame_name: str,
    *,
    expected_bit_depth: int,
    expected_color_type: int,
    acquired: AcquiredRoot | None = None,
) -> dict[str, Any]:
    if path.name != f"{frame_name}.png":
        raise StereoControlStateError("stereo input PNG name differs from its frame stem")
    sha256, byte_count, raw_header = _stream_identity(
        path,
        job_root=root,
        acquired=acquired,
    )
    header = _parse_png_header(raw_header, path)
    if header["bit_depth"] != expected_bit_depth or header["color_type"] != expected_color_type:
        raise StereoControlStateError(f"stereo input PNG format is invalid: {path}")
    return {
        "byte_count": byte_count,
        "frame_name": frame_name,
        "png_header": header,
        "relative_path": _contained_relative_path(root, path),
        "sha256": sha256,
    }


def _metric_identity(
    root: Path,
    path: Path,
    frame_name: str,
    native_shape: tuple[int, int],
    *,
    acquired: AcquiredRoot | None = None,
) -> dict[str, Any]:
    if path.name != f"{frame_name}.npz":
        raise StereoControlStateError("metric input name differs from its frame stem")
    audit = metric_npz.audit_metric_npz(
        path,
        expected_shape=native_shape,
        job_root=root,
        acquired=acquired,
    )
    try:
        focal_bits = struct.unpack(
            "<I",
            struct.pack("<f", float(audit.focal_x_normalized)),
        )[0]
        return {
            "archive_form": audit.layout.archive_form,
            "byte_count": int(audit.byte_count),
            "focal_x_normalized_float32_bits": f"{focal_bits:08x}",
            "frame_name": frame_name,
            "inverse_depth_dtype": "float32",
            "members": [
                "inverse_depth.npy",
                "valid.npy",
                "focal_x_normalized.npy",
            ],
            "native_shape": [int(native_shape[0]), int(native_shape[1])],
            "payload_contract": "metric-npz-v1",
            "payload_kind": "metric_npz",
            "relative_path": _contained_relative_path(root, path),
            "sha256": audit.sha256,
            "valid_dtype": "bool",
        }
    finally:
        del audit


def _float32_bits(value: object) -> str:
    scalar = np.float32(value)
    if not np.isfinite(scalar) or scalar <= np.float32(0.0):
        raise StereoControlStateError("relative encoding scale must be finite and positive")
    return f"{struct.unpack('<I', struct.pack('<f', float(scalar)))[0]:08x}"


def _compact_source_metadata(
    source_metadata: Mapping[str, Any],
    *,
    geometry_mode: Literal["relative", "metric"],
    render_shape: tuple[int, int],
    native_shape: tuple[int, int],
) -> dict[str, Any]:
    fingerprint = source_metadata.get("fingerprint")
    if not _is_hex64(fingerprint):
        raise StereoControlStateError("source geometry fingerprint is invalid")
    if geometry_mode == "relative":
        scale_bits: str | None = _float32_bits(source_metadata.get("encoding_scale"))
        resolved_auto: float | None = None
    else:
        scale_bits = None
        convergence = source_metadata.get("convergence")
        if not isinstance(convergence, Mapping):
            raise StereoControlStateError("metric convergence metadata is invalid")
        resolved_auto = float(convergence.get("resolved_auto_distance_m"))
        if not math.isfinite(resolved_auto) or resolved_auto <= 0.0:
            raise StereoControlStateError("metric auto convergence is invalid")
    return {
        "native_shape": [native_shape[0], native_shape[1]],
        "relative_encoding_scale_float32_bits": scale_bits,
        "render_shape": [render_shape[0], render_shape[1]],
        "resolved_auto_distance_m": resolved_auto,
        "source_geometry_fingerprint": fingerprint,
    }


def _write_line(handle: BinaryIO, digest: Any, value: object) -> int:
    raw = _canonical_line(value)
    if handle.write(raw) != len(raw):
        raise OSError("short stereo input manifest write")
    digest.update(raw)
    return len(raw)


def _next_pair(
    guide_iterator: Iterator[Path],
    geometry_iterator: Iterator[Path],
) -> tuple[Path, Path]:
    try:
        guide = Path(next(guide_iterator))
    except StopIteration as error:
        raise StereoControlStateError("guide sequence ended before its declared count") from error
    try:
        geometry = Path(next(geometry_iterator))
    except StopIteration as error:
        raise StereoControlStateError(
            "geometry sequence ended before its declared count"
        ) from error
    return guide, geometry


def _assert_exhausted(iterator: Iterator[Path], label: str) -> None:
    try:
        next(iterator)
    except StopIteration:
        return
    raise StereoControlStateError(f"{label} sequence exceeds its declared count")


def _build_item(
    *,
    root: Path,
    source_index: int,
    expected_frame_index: int | None,
    guide_path: Path,
    geometry_path: Path,
    geometry_mode: Literal["relative", "metric"],
    native_shape: tuple[int, int] | None,
) -> tuple[dict[str, Any], tuple[int, int], tuple[int, int], int]:
    if guide_path.stem != geometry_path.stem:
        raise StereoControlStateError("guide and geometry stems differ")
    frame_name = guide_path.stem
    frame_index = _canonical_frame_index(frame_name)
    if expected_frame_index is not None and frame_index != expected_frame_index:
        raise StereoControlStateError("stereo input frame indexes are noncontiguous")
    guide = _png_identity(
        root,
        guide_path,
        frame_name,
        expected_bit_depth=8,
        expected_color_type=2,
    )
    render_shape = (int(guide["png_header"]["height"]), int(guide["png_header"]["width"]))
    if geometry_mode == "relative":
        geometry = _png_identity(
            root,
            geometry_path,
            frame_name,
            expected_bit_depth=16,
            expected_color_type=0,
        )
        geometry["payload_kind"] = "canonical_png"
        actual_native_shape = (
            int(geometry["png_header"]["height"]),
            int(geometry["png_header"]["width"]),
        )
    else:
        if native_shape is None:
            raise StereoControlStateError("metric source metadata has no native shape")
        actual_native_shape = native_shape
        geometry = _metric_identity(root, geometry_path, frame_name, native_shape)
    return (
        {
            "frame_index": frame_index,
            "frame_name": frame_name,
            "geometry": geometry,
            "guide": guide,
            "source_index": source_index,
        },
        render_shape,
        actual_native_shape,
        frame_index,
    )


@dataclass(frozen=True, slots=True)
class StereoInputManifestV1:
    path: Path
    job_root: Path
    raw_sha256: str
    raw_byte_count: int
    frame_count: int
    geometry_mode: Literal["relative", "metric"]
    render_shape: tuple[int, int]
    native_shape: tuple[int, int]
    source_geometry_fingerprint: str
    relative_encoding_scale_float32_bits: str | None
    resolved_auto_distance_m: float | None

    @property
    def relative_encoding_scale(self) -> np.float32 | None:
        if self.relative_encoding_scale_float32_bits is None:
            return None
        bits = int(self.relative_encoding_scale_float32_bits, 16)
        return np.float32(struct.unpack("<f", struct.pack("<I", bits))[0])

    def open_provider(self, *, left_dir: Path, right_dir: Path) -> ReplayableStereoProvider:
        generation = StereoProviderGeneration(
            audit_generation_id=secrets.token_hex(16),
            input_manifest_fingerprint=self.raw_sha256,
            frame_count=self.frame_count,
        )
        replay = _ManifestReplay(self, generation, Path(left_dir), Path(right_dir))
        return ReplayableStereoProvider(generation, replay)

    def audit_payloads(self) -> None:
        """Validate every declared payload once without retaining row identities."""

        count = 0
        for record in self.replay_records():
            if record.source_index != count:
                raise StereoControlStateError("stereo input payload order changed")
            count += 1
        if count != self.frame_count:
            raise StereoControlStateError("stereo input payload audit ended early")

    def replay_records(self) -> Iterator["StereoInputManifestRecord"]:
        """Reopen and validate source content while yielding one manifest row."""

        return self._replay_records(payload_validation="content")

    def replay_declared_records(self) -> Iterator["StereoInputManifestRecord"]:
        """Replay identities after a content-audit phase without reopening payloads."""

        return self._replay_records(payload_validation="none")

    def _replay_records(
        self,
        *,
        payload_validation: Literal["none", "identity", "content"],
    ) -> Iterator["StereoInputManifestRecord"]:
        try:
            with AcquiredRoot(self.job_root) as acquired:
                with _regular_file(
                    self.path,
                    job_root=self.job_root,
                    acquired=acquired,
                )[0] as handle:
                    for value, guide_path, geometry_path in _scan_manifest(
                        handle,
                        self,
                        payload_validation=payload_validation,
                        acquired=acquired,
                    ):
                        yield StereoInputManifestRecord(
                            source_index=value["source_index"],
                            frame_name=value["frame_name"],
                            guide_path=guide_path,
                            geometry_path=geometry_path,
                            guide_identity=dict(value["guide"]),
                            geometry_identity=dict(value["geometry"]),
                        )
        except StereoInputIdentityMismatch:
            raise
        except (OSError, TypeError, ValueError) as error:
            raise StereoInputIdentityMismatch(str(error)) from error


@dataclass(frozen=True, slots=True)
class StereoInputManifestRecord:
    source_index: int
    frame_name: str
    guide_path: Path
    geometry_path: Path
    guide_identity: dict[str, Any]
    geometry_identity: dict[str, Any]


def publish_validated_stereo_input_manifest_v1(
    *,
    job_root: Path,
    guide_files: Sequence[Path],
    geometry_files: Sequence[Path],
    geometry_mode: Literal["relative", "metric_camera"],
) -> StereoInputManifestV1:
    """Publish the stereo handoff at the validated upstream completion point."""

    if not guide_files or len(guide_files) != len(geometry_files):
        raise StereoControlStateError("stereo upstream sequences are incomplete")
    if geometry_mode == "relative":
        artifact = validate_render_disparity_input(list(geometry_files), list(guide_files))
        source_metadata = dict(artifact.metadata)
        source_metadata["fingerprint"] = artifact.fingerprint
        manifest_geometry_mode: Literal["relative", "metric"] = "relative"
    elif geometry_mode == "metric_camera":
        from .metric_geometry import MetricGeometryStore

        geometry_parent = Path(geometry_files[0]).parent
        metadata = MetricGeometryStore.read_metadata(geometry_parent)
        if not isinstance(metadata, dict):
            raise StereoControlStateError("metric geometry metadata is missing")
        try:
            store = MetricGeometryStore.open_existing(
                geometry_parent,
                frame_names=[Path(path).name for path in guide_files],
                source_raw_fingerprint=metadata["source_raw_fingerprint"],
                source_frame_fingerprint=metadata["source_frame_fingerprint"],
                candidate_scene_fingerprint=metadata["candidate_scene_fingerprint"],
                cleanup_temporaries=False,
            )
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise StereoControlStateError("metric geometry stage is incomplete") from error
        if [Path(path).resolve() for path in geometry_files] != [
            path.resolve() for path in store.complete_files
        ]:
            raise StereoControlStateError("metric geometry paths differ from their manifest")
        source_metadata = store.metadata
        manifest_geometry_mode = "metric"
    else:
        raise StereoControlStateError("stereo geometry mode is unsupported")
    return publish_stereo_input_manifest_v1(
        manifest_path=Path(geometry_files[0]).parent / "stereo_input_manifest.jsonl",
        job_root=job_root,
        guide_files=iter(guide_files),
        geometry_files=iter(geometry_files),
        frame_count=len(guide_files),
        geometry_mode=manifest_geometry_mode,
        source_metadata=source_metadata,
    )


def publish_stereo_input_manifest_v1(  # noqa: C901
    *,
    manifest_path: Path,
    job_root: Path,
    guide_files: Iterable[Path],
    geometry_files: Iterable[Path],
    frame_count: int,
    geometry_mode: Literal["relative", "metric"],
    source_metadata: Mapping[str, Any],
) -> StereoInputManifestV1:
    """Publish one complete upstream manifest before the stereo transaction begins."""

    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count <= 0
        or frame_count > STEREO_CONTROL_FRAME_CAP
    ):
        raise StereoControlBudgetError("stereo input frame count is outside the control cap")
    if geometry_mode not in {"relative", "metric"}:
        raise ValueError("stereo input geometry mode must be relative or metric")
    root = Path(os.path.abspath(os.fspath(job_root)))
    manifest_path = Path(os.path.abspath(os.fspath(manifest_path)))
    try:
        manifest_path.relative_to(root)
    except ValueError as error:
        raise ValueError("stereo input manifest must be contained by the job root") from error
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    guide_iterator = iter(guide_files)
    geometry_iterator = iter(geometry_files)
    first_guide, first_geometry = _next_pair(guide_iterator, geometry_iterator)
    declared_native_shape: tuple[int, int] | None = None
    if geometry_mode == "metric":
        raw_shape = source_metadata.get("native_shape")
        if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) != 2:
            raise StereoControlStateError("metric source native shape is invalid")
        declared_native_shape = (int(raw_shape[0]), int(raw_shape[1]))
    first_item, render_shape, native_shape, first_frame_index = _build_item(
        root=root,
        source_index=0,
        expected_frame_index=None,
        guide_path=first_guide,
        geometry_path=first_geometry,
        geometry_mode=geometry_mode,
        native_shape=declared_native_shape,
    )
    compact_metadata = _compact_source_metadata(
        source_metadata,
        geometry_mode=geometry_mode,
        render_shape=render_shape,
        native_shape=native_shape,
    )
    header = {
        "algorithm_version": STEREO_INPUT_MANIFEST_ALGORITHM_VERSION,
        "frame_count": frame_count,
        "geometry_mode": geometry_mode,
        "schema_version": STEREO_INPUT_MANIFEST_SCHEMA_VERSION,
        "source_metadata": compact_metadata,
    }
    owned = OwnedSiblingTemporary.create(manifest_path)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        handle = owned.handle
        byte_count += _write_line(handle, digest, header)
        byte_count += _write_line(handle, digest, first_item)
        for source_index in range(1, frame_count):
            guide_path, geometry_path = _next_pair(guide_iterator, geometry_iterator)
            item, item_render_shape, item_native_shape, _frame_index = _build_item(
                root=root,
                source_index=source_index,
                expected_frame_index=first_frame_index + source_index,
                guide_path=guide_path,
                geometry_path=geometry_path,
                geometry_mode=geometry_mode,
                native_shape=native_shape,
            )
            if item_render_shape != render_shape or item_native_shape != native_shape:
                raise StereoControlStateError("stereo input dimensions changed within the sequence")
            byte_count += _write_line(handle, digest, item)
        _assert_exhausted(guide_iterator, "guide")
        _assert_exhausted(geometry_iterator, "geometry")
        owned.sync()
        owned.replace()
    finally:
        owned.cleanup()
    result = load_stereo_input_manifest_v1(manifest_path, job_root=root)
    if result.raw_sha256 != digest.hexdigest() or result.raw_byte_count != byte_count:
        raise StereoControlStateError("published stereo input manifest identity changed")
    return result


def _read_line(handle: BinaryIO) -> bytes | None:
    raw = handle.readline(STEREO_INPUT_MANIFEST_LINE_BYTES + 1)
    if not raw:
        return None
    if len(raw) > STEREO_INPUT_MANIFEST_LINE_BYTES or not raw.endswith(b"\n"):
        raise StereoControlStateError("stereo input manifest line is unbounded or unterminated")
    return raw


def _parse_line(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw[:-1].decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StereoControlStateError("stereo input manifest JSON is malformed") from error
    if not isinstance(value, dict) or _canonical_line(value) != raw:
        raise StereoControlStateError("stereo input manifest JSON is noncanonical")
    return value


def _positive_shape(value: object, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        raise StereoControlStateError(f"stereo input {label} is invalid")
    return int(value[0]), int(value[1])


def load_stereo_input_manifest_v1(
    manifest_path: Path,
    *,
    job_root: Path,
    validate_payloads: bool = True,
) -> StereoInputManifestV1:
    """Boundedly adopt a durable upstream manifest without directory inference."""

    root = Path(os.path.abspath(os.fspath(job_root)))
    path = Path(os.path.abspath(os.fspath(manifest_path)))
    try:
        path.relative_to(root)
    except ValueError as error:
        raise StereoControlStateError("stereo input manifest escapes the job root") from error
    raw_sha256, raw_byte_count, _header_prefix = _stream_identity(path, job_root=root)
    with _regular_file(path, job_root=root)[0] as handle:
        raw = _read_line(handle)
    if raw is None:
        raise StereoControlStateError("stereo input manifest is empty")
    header = _parse_line(raw)
    if set(header) != {
        "algorithm_version",
        "frame_count",
        "geometry_mode",
        "schema_version",
        "source_metadata",
    }:
        raise StereoControlStateError("stereo input manifest header schema changed")
    frame_count = header.get("frame_count")
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count <= 0
        or frame_count > STEREO_CONTROL_FRAME_CAP
    ):
        raise StereoControlBudgetError("stereo input manifest frame count is outside the cap")
    geometry_mode = header.get("geometry_mode")
    if geometry_mode not in {"relative", "metric"}:
        raise StereoControlStateError("stereo input manifest geometry mode is invalid")
    if (
        header.get("schema_version") != STEREO_INPUT_MANIFEST_SCHEMA_VERSION
        or header.get("algorithm_version") != STEREO_INPUT_MANIFEST_ALGORITHM_VERSION
    ):
        raise StereoControlStateError("stereo input manifest version is unsupported")
    metadata = header.get("source_metadata")
    if not isinstance(metadata, dict) or set(metadata) != {
        "native_shape",
        "relative_encoding_scale_float32_bits",
        "render_shape",
        "resolved_auto_distance_m",
        "source_geometry_fingerprint",
    }:
        raise StereoControlStateError("stereo input source metadata schema changed")
    render_shape = _positive_shape(metadata["render_shape"], "render shape")
    native_shape = _positive_shape(metadata["native_shape"], "native shape")
    source_fingerprint = metadata["source_geometry_fingerprint"]
    if not _is_hex64(source_fingerprint):
        raise StereoControlStateError("stereo input source fingerprint is invalid")
    scale_bits = metadata["relative_encoding_scale_float32_bits"]
    resolved_auto = metadata["resolved_auto_distance_m"]
    if geometry_mode == "relative":
        if (
            not isinstance(scale_bits, str)
            or len(scale_bits) != 8
            or any(character not in _HEX_64 for character in scale_bits)
            or resolved_auto is not None
        ):
            raise StereoControlStateError("relative stereo input scalar metadata is invalid")
    else:
        if (
            scale_bits is not None
            or isinstance(resolved_auto, bool)
            or not isinstance(
                resolved_auto,
                (int, float),
            )
        ):
            raise StereoControlStateError("metric stereo input scalar metadata is invalid")
        resolved_auto = float(resolved_auto)
        if not math.isfinite(resolved_auto) or resolved_auto <= 0.0:
            raise StereoControlStateError("metric stereo input convergence is invalid")
    descriptor = StereoInputManifestV1(
        path=path,
        job_root=root,
        raw_sha256=raw_sha256,
        raw_byte_count=raw_byte_count,
        frame_count=frame_count,
        geometry_mode=geometry_mode,
        render_shape=render_shape,
        native_shape=native_shape,
        source_geometry_fingerprint=source_fingerprint,
        relative_encoding_scale_float32_bits=scale_bits,
        resolved_auto_distance_m=resolved_auto,
    )
    if validate_payloads:
        _validate_descriptor_manifest(descriptor)
    return descriptor


def _validate_header(value: dict[str, Any], descriptor: StereoInputManifestV1) -> None:
    expected = {
        "algorithm_version": STEREO_INPUT_MANIFEST_ALGORITHM_VERSION,
        "frame_count": descriptor.frame_count,
        "geometry_mode": descriptor.geometry_mode,
        "schema_version": STEREO_INPUT_MANIFEST_SCHEMA_VERSION,
        "source_metadata": {
            "native_shape": list(descriptor.native_shape),
            "relative_encoding_scale_float32_bits": (
                descriptor.relative_encoding_scale_float32_bits
            ),
            "render_shape": list(descriptor.render_shape),
            "resolved_auto_distance_m": descriptor.resolved_auto_distance_m,
            "source_geometry_fingerprint": descriptor.source_geometry_fingerprint,
        },
    }
    if value != expected:
        raise StereoControlStateError("stereo input manifest header changed")


def _validate_identity_common(
    value: object,
    *,
    frame_name: str,
    suffix: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StereoControlStateError("stereo input identity is malformed")
    if value.get("frame_name") != frame_name:
        raise StereoControlStateError("stereo input identity frame name changed")
    relative = value.get("relative_path")
    if not isinstance(relative, str):
        raise StereoControlStateError("stereo input relative path is malformed")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} or "\\" in part or "\x00" in part for part in pure.parts)
        or pure.name != f"{frame_name}{suffix}"
    ):
        raise StereoControlStateError("stereo input relative path is noncanonical")
    if not _is_hex64(value.get("sha256")):
        raise StereoControlStateError("stereo input content hash is invalid")
    byte_count = value.get("byte_count")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
        raise StereoControlStateError("stereo input byte count is invalid")
    return value


def _validate_item(
    value: dict[str, Any],
    *,
    descriptor: StereoInputManifestV1,
    expected_source_index: int,
    expected_frame_index: int | None,
) -> tuple[dict[str, Any], int]:
    if set(value) != {"frame_index", "frame_name", "geometry", "guide", "source_index"}:
        raise StereoControlStateError("stereo input manifest item schema changed")
    if value["source_index"] != expected_source_index:
        raise StereoControlStateError("stereo input source indexes are noncontiguous")
    frame_name = value["frame_name"]
    frame_index = _canonical_frame_index(frame_name)
    if value["frame_index"] != frame_index or (
        expected_frame_index is not None and frame_index != expected_frame_index
    ):
        raise StereoControlStateError("stereo input frame indexes are noncontiguous")
    guide = _validate_identity_common(value["guide"], frame_name=frame_name, suffix=".png")
    if set(guide) != {"byte_count", "frame_name", "png_header", "relative_path", "sha256"}:
        raise StereoControlStateError("stereo guide identity schema changed")
    header = guide.get("png_header")
    if header != {
        "bit_depth": 8,
        "color_type": 2,
        "height": descriptor.render_shape[0],
        "width": descriptor.render_shape[1],
    }:
        raise StereoControlStateError("stereo guide header changed")
    geometry_suffix = ".png" if descriptor.geometry_mode == "relative" else ".npz"
    geometry = _validate_identity_common(
        value["geometry"],
        frame_name=frame_name,
        suffix=geometry_suffix,
    )
    if descriptor.geometry_mode == "relative":
        if (
            set(geometry)
            != {
                "byte_count",
                "frame_name",
                "payload_kind",
                "png_header",
                "relative_path",
                "sha256",
            }
            or geometry.get("payload_kind") != "canonical_png"
        ):
            raise StereoControlStateError("relative geometry identity schema changed")
        if geometry.get("png_header") != {
            "bit_depth": 16,
            "color_type": 0,
            "height": descriptor.native_shape[0],
            "width": descriptor.native_shape[1],
        }:
            raise StereoControlStateError("relative geometry header changed")
    else:
        if set(geometry) != {
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
        }:
            raise StereoControlStateError("metric geometry identity schema changed")
        if (
            geometry.get("payload_kind") != "metric_npz"
            or geometry.get("payload_contract") != "metric-npz-v1"
            or geometry.get("native_shape") != list(descriptor.native_shape)
            or geometry.get("inverse_depth_dtype") != "float32"
            or geometry.get("valid_dtype") != "bool"
            or geometry.get("members")
            != ["inverse_depth.npy", "valid.npy", "focal_x_normalized.npy"]
        ):
            raise StereoControlStateError("metric geometry contract changed")
    return value, frame_index


def _actual_identity_matches(
    descriptor: StereoInputManifestV1,
    identity: dict[str, Any],
    *,
    guide: bool,
    payload_validation: Literal["identity", "content"],
    acquired: AcquiredRoot | None = None,
) -> Path:
    relative = PurePosixPath(identity["relative_path"])
    path = descriptor.job_root.joinpath(*relative.parts)
    frame_name = identity["frame_name"]
    if guide:
        actual = _png_identity(
            descriptor.job_root,
            path,
            frame_name,
            expected_bit_depth=8,
            expected_color_type=2,
            acquired=acquired,
        )
    elif descriptor.geometry_mode == "relative":
        actual = _png_identity(
            descriptor.job_root,
            path,
            frame_name,
            expected_bit_depth=16,
            expected_color_type=0,
            acquired=acquired,
        )
        actual["payload_kind"] = "canonical_png"
    else:
        if payload_validation == "content":
            actual = _metric_identity(
                descriptor.job_root,
                path,
                frame_name,
                descriptor.native_shape,
                acquired=acquired,
            )
        else:
            audit = metric_npz.audit_metric_npz_identity(
                path,
                expected_shape=descriptor.native_shape,
                job_root=descriptor.job_root,
                acquired=acquired,
            )
            if (
                audit.sha256 != identity["sha256"]
                or audit.byte_count != identity["byte_count"]
                or audit.layout.archive_form != identity["archive_form"]
            ):
                raise StereoControlStateError(f"stereo input payload changed: {path}")
            return path
    if actual != identity:
        raise StereoControlStateError(f"stereo input payload changed: {path}")
    return path


def _scan_manifest(
    handle: BinaryIO,
    descriptor: StereoInputManifestV1,
    *,
    payload_validation: Literal["none", "identity", "content"],
    acquired: AcquiredRoot | None = None,
) -> Iterator[tuple[dict[str, Any], Path, Path]]:
    reader = (
        handle
        if isinstance(handle, io.BufferedReader)
        else io.BufferedReader(cast(Any, handle), buffer_size=STEREO_INPUT_PAYLOAD_STREAM_BYTES)
    )
    owns_reader = reader is not handle
    digest = hashlib.sha256()
    raw_byte_count = 0
    try:
        header_raw = _read_line(reader)
        if header_raw is None:
            raise StereoControlStateError("stereo input manifest is empty")
        digest.update(header_raw)
        raw_byte_count += len(header_raw)
        _validate_header(_parse_line(header_raw), descriptor)
        first_frame_index: int | None = None
        for source_index in range(descriptor.frame_count):
            raw = _read_line(reader)
            if raw is None:
                raise StereoControlStateError("stereo input manifest ended before its frame count")
            digest.update(raw)
            raw_byte_count += len(raw)
            value, frame_index = _validate_item(
                _parse_line(raw),
                descriptor=descriptor,
                expected_source_index=source_index,
                expected_frame_index=(
                    None if first_frame_index is None else first_frame_index + source_index
                ),
            )
            if first_frame_index is None:
                first_frame_index = frame_index
            if payload_validation != "none":
                guide_path = _actual_identity_matches(
                    descriptor,
                    value["guide"],
                    guide=True,
                    payload_validation=payload_validation,
                    acquired=acquired,
                )
                geometry_path = _actual_identity_matches(
                    descriptor,
                    value["geometry"],
                    guide=False,
                    payload_validation=payload_validation,
                    acquired=acquired,
                )
            else:
                guide_path = descriptor.job_root.joinpath(
                    *PurePosixPath(value["guide"]["relative_path"]).parts
                )
                geometry_path = descriptor.job_root.joinpath(
                    *PurePosixPath(value["geometry"]["relative_path"]).parts
                )
            yield value, guide_path, geometry_path
        if _read_line(reader) is not None:
            raise StereoControlStateError("stereo input manifest exceeds its frame count")
        if (
            raw_byte_count != descriptor.raw_byte_count
            or digest.hexdigest() != descriptor.raw_sha256
        ):
            raise StereoControlStateError("stereo input manifest raw identity changed")
    finally:
        if owns_reader:
            reader.detach()


def _validate_descriptor_manifest(descriptor: StereoInputManifestV1) -> None:
    with AcquiredRoot(descriptor.job_root) as acquired:
        with _regular_file(
            descriptor.path,
            job_root=descriptor.job_root,
            acquired=acquired,
        )[0] as handle:
            for _item in _scan_manifest(
                handle,
                descriptor,
                payload_validation="content",
                acquired=acquired,
            ):
                pass


@dataclass(frozen=True, slots=True)
class _ManifestReplay:
    descriptor: StereoInputManifestV1
    generation: StereoProviderGeneration
    left_dir: Path
    right_dir: Path

    def __post_init__(self) -> None:
        for path in (self.left_dir, self.right_dir):
            try:
                Path(os.path.abspath(os.fspath(path))).relative_to(self.descriptor.job_root)
            except ValueError as error:
                raise ValueError("stereo output parent escapes the job root") from error

    def __call__(self) -> Iterator[StereoProviderItem]:
        try:
            yield from self._iter_items()
        except StereoControlStateError as error:
            raise StereoInputIdentityMismatch(str(error)) from error

    def _iter_items(self) -> Iterator[StereoProviderItem]:
        with AcquiredRoot(self.descriptor.job_root) as acquired:
            with _regular_file(
                self.descriptor.path,
                job_root=self.descriptor.job_root,
                acquired=acquired,
            )[0] as handle:
                original_version = _file_version(os.fstat(handle.fileno()))
                for value, guide_path, geometry_path in _scan_manifest(
                    handle,
                    self.descriptor,
                    payload_validation="identity",
                    acquired=acquired,
                ):
                    if _file_version(os.fstat(handle.fileno())) != original_version:
                        raise StereoControlStateError("stereo input manifest changed during replay")
                    frame_name = value["frame_name"]
                    yield StereoProviderItem(
                        audit_generation_id=self.generation.audit_generation_id,
                        input_manifest_fingerprint=self.generation.input_manifest_fingerprint,
                        source_index=value["source_index"],
                        frame_name=frame_name,
                        guide_path=guide_path,
                        geometry_path=geometry_path,
                        left_path=self.left_dir / f"{frame_name}.png",
                        right_path=self.right_dir / f"{frame_name}.png",
                        guide_sha256=value["guide"]["sha256"],
                        geometry_sha256=value["geometry"]["sha256"],
                        guide_byte_count=value["guide"]["byte_count"],
                        geometry_byte_count=value["geometry"]["byte_count"],
                        guide_shape=self.descriptor.render_shape,
                        geometry_shape=self.descriptor.native_shape,
                        frame_json_bound=(
                            4 * 272 * (self.descriptor.render_shape[1] + 1) + 1024 * 1024
                        ),
                    )
