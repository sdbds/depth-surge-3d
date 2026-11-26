"""Lightweight stereo stage input validation without redundant per-frame SHA-256 hashing."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterator, Literal, Sequence

from ...utils.imaging.png_header import read_png_header
from .stereo_control import StereoControlStateError
from .video_frame_source import VideoFrameSource


_FRAME_STEM_RE = re.compile(r"^frame_([0-9]{6,})$")


@dataclass(frozen=True, slots=True)
class StereoInputRecord:
    """One validated guide/geometry input pair for the stereo stage."""

    frame_name: str
    frame_index: int
    guide_path: Path
    geometry_path: Path


@dataclass(frozen=True)
class StereoInputManifestV1:
    """Validated in-memory stereo stage input plan (zero per-frame disk hashing)."""

    job_root: Path
    geometry_mode: Literal["relative", "metric"]
    render_shape: tuple[int, int]
    native_shape: tuple[int, int]
    first_index: int
    records: tuple[StereoInputRecord, ...]
    source_metadata: dict[str, Any]
    frame_source: VideoFrameSource | None = None

    @property
    def frame_count(self) -> int:
        return len(self.records)

    @property
    def source_geometry_fingerprint(self) -> str:
        return str(self.source_metadata.get("fingerprint", ""))

    @property
    def relative_encoding_scale(self) -> float | None:
        if self.geometry_mode != "relative":
            return None
        val = self.source_metadata.get("encoding_scale")
        return float(val) if val is not None else None

    @property
    def resolved_auto_distance_m(self) -> float | None:
        if self.geometry_mode != "metric":
            return None
        conv = self.source_metadata.get("convergence")
        if isinstance(conv, dict) and conv.get("resolved_auto_distance_m") is not None:
            return float(conv["resolved_auto_distance_m"])
        return None

    def replay_declared_records(self) -> Iterator[StereoInputRecord]:
        return iter(self.records)

    def frame_names(self) -> tuple[str, ...]:
        return tuple(record.frame_name for record in self.records)


def _canonical_frame_index(stem: str) -> int:
    match = _FRAME_STEM_RE.fullmatch(stem)
    if match is None:
        raise StereoControlStateError(f"noncanonical stereo frame stem: {stem!r}")
    index = int(match.group(1))
    expected = str(index).zfill(max(6, len(str(index))))
    if match.group(1) != expected:
        raise StereoControlStateError(f"noncanonical zero padding in frame stem: {stem!r}")
    return index


def _load_stage_metadata(directory: Path) -> dict[str, Any]:
    meta_path = directory / "metadata.json"
    if not meta_path.is_file():
        raise StereoControlStateError(f"missing upstream stage metadata: {meta_path}")
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StereoControlStateError(f"unreadable upstream metadata: {meta_path}") from error
    if not isinstance(payload, dict):
        raise StereoControlStateError(f"upstream metadata is not a JSON object: {meta_path}")
    return payload


def publish_validated_stereo_input_manifest_v1(  # noqa: C901
    *,
    job_root: Path,
    guide_files: Sequence[Path],
    geometry_files: Sequence[Path],
    geometry_mode: Literal["relative", "metric_camera", "metric"],
    frame_source: VideoFrameSource | None = None,
) -> StereoInputManifestV1:
    """Validate guide/geometry alignment using headers and stage metadata only."""

    root = Path(os.path.abspath(os.fspath(job_root)))
    if not guide_files or not geometry_files:
        raise StereoControlStateError(
            "stereo stage requires non-empty guide and geometry sequences"
        )
    if len(guide_files) != len(geometry_files):
        raise StereoControlStateError(
            f"guide count ({len(guide_files)}) differs from geometry count ({len(geometry_files)})"
        )

    normalized_mode: Literal["relative", "metric"] = (
        "metric" if geometry_mode in {"metric", "metric_camera"} else "relative"
    )
    expected_geom_suffix = ".npz" if normalized_mode == "metric" else ".png"

    records: list[StereoInputRecord] = []
    for raw_guide, raw_geom in zip(guide_files, geometry_files):
        guide = Path(raw_guide)
        geom = Path(raw_geom)
        if guide.suffix.lower() != ".png" or geom.suffix.lower() != expected_geom_suffix:
            raise StereoControlStateError(
                f"unexpected stereo input extensions: {guide.name}, {geom.name}"
            )
        if guide.stem != geom.stem:
            raise StereoControlStateError(
                f"mismatched stereo input frame stems: {guide.name} vs {geom.name}"
            )
        idx = _canonical_frame_index(guide.stem)
        records.append(
            StereoInputRecord(
                frame_name=guide.stem,
                frame_index=idx,
                guide_path=guide,
                geometry_path=geom,
            )
        )

    if frame_source is not None:
        if frame_source.frame_count != len(records):
            raise StereoControlStateError(
                f"streaming frame count ({frame_source.frame_count}) differs from "
                f"geometry count ({len(records)})"
            )
        render_shape = (int(frame_source.height), int(frame_source.width))
    else:
        first_guide_header = read_png_header(records[0].guide_path)
        if first_guide_header is None or first_guide_header.bit_depth != 8:
            raise StereoControlStateError(
                f"first stereo guide PNG is invalid: {records[0].guide_path}"
            )
        render_shape = (int(first_guide_header.height), int(first_guide_header.width))

    geom_dir = records[0].geometry_path.parent
    source_metadata = _load_stage_metadata(geom_dir)

    if normalized_mode == "relative":
        first_geom_header = read_png_header(records[0].geometry_path)
        if first_geom_header is None or first_geom_header.bit_depth != 16:
            raise StereoControlStateError(
                f"first relative disparity PNG is invalid: {records[0].geometry_path}"
            )
        native_shape = (int(first_geom_header.height), int(first_geom_header.width))
        scale = source_metadata.get("encoding_scale")
        if scale is None or not math.isfinite(float(scale)) or float(scale) <= 0.0:
            raise StereoControlStateError(
                "relative disparity metadata lacks a valid encoding_scale"
            )
    else:
        raw_native = source_metadata.get("native_shape")
        if (
            not isinstance(raw_native, list)
            or len(raw_native) != 2
            or any(int(v) <= 0 for v in raw_native)
        ):
            raise StereoControlStateError("metric geometry metadata lacks a valid native_shape")
        native_shape = (int(raw_native[0]), int(raw_native[1]))
        conv = source_metadata.get("convergence")
        if not isinstance(conv, dict) or conv.get("resolved_auto_distance_m") is None:
            raise StereoControlStateError(
                "metric stereo requires completed clip-global convergence metadata"
            )
        dist = float(conv["resolved_auto_distance_m"])
        if not math.isfinite(dist) or dist <= 0.0:
            raise StereoControlStateError("metric convergence distance is invalid")

    return StereoInputManifestV1(
        job_root=root,
        geometry_mode=normalized_mode,
        render_shape=render_shape,
        native_shape=native_shape,
        first_index=records[0].frame_index,
        records=tuple(records),
        source_metadata=source_metadata,
        frame_source=frame_source,
    )


def load_stereo_input_manifest_v1(manifest: StereoInputManifestV1) -> StereoInputManifestV1:
    """Identity helper preserved for callers passing an already-built manifest."""

    return manifest
