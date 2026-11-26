"""Lightweight in-memory stereo diagnostics accumulator and optional mask writer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import cv2
import numpy as np

from ...io.canonical_json import canonical_json_bytes
from ...io.durable_publish import atomic_write_bytes, atomic_write_json
from ...rendering.stereo_geometry import MetricProjectionStats
from .diagnostics import build_quality_frame_stats
from .types import CompactStereoRenderResult, DenseHistogramPairs


def _json_compatible(value: Any) -> Any:
    if isinstance(value, DenseHistogramPairs):
        return [[int(idx), int(count)] for idx, count in enumerate(value.bins) if int(count) > 0]
    if isinstance(value, dict):
        return {k: _json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(v) for v in value]
    return value


DIAGNOSTICS_DIR_NAME = "04_stereo_diagnostics"
DIAGNOSTICS_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class QualityFramePublication:
    """Inputs for publishing one Quality/Fast frame's optional masks and RGB pair."""

    job_root: Path
    frame_name: str
    left_rgb_path: Path
    right_rgb_path: Path
    compact: CompactStereoRenderResult
    geometry_mode: Literal["relative", "metric"]
    occlusion_fill: Literal["none", "background"]
    rgb_stage_fingerprint: str
    keep_masks: bool = False
    render_mode: Literal["fast", "quality"] = "quality"
    metric_stats: MetricProjectionStats | None = None


@dataclass(frozen=True)
class PreparedQualityFramePublication:
    """Encoded PNG payloads and frame stats prepared for atomic disk write."""

    publication: QualityFramePublication
    left_rgb_png: bytes
    right_rgb_png: bytes
    mask_pngs: dict[str, bytes]
    stats: dict[str, Any]


def _encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("failed to encode PNG payload")
    return encoded.tobytes()


def prepare_quality_frame(publication: QualityFramePublication) -> PreparedQualityFramePublication:
    """Encode RGB pair (and optional diagnostic masks when ``keep_masks=True``)."""

    compact = publication.compact
    stats = build_quality_frame_stats(
        compact,
        frame_name=publication.frame_name,
        geometry_mode=publication.geometry_mode,
        occlusion_fill=publication.occlusion_fill,
        render_mode=publication.render_mode,
    )

    left_rgb_png = _encode_png(compact.left_image)
    right_rgb_png = _encode_png(compact.right_image)
    mask_pngs: dict[str, bytes] = {}
    if publication.keep_masks:
        mask_pngs = {
            "left_coverage": _encode_png(compact.left_coverage_count),
            "left_repair": _encode_png(compact.left_repair_bits),
            "right_coverage": _encode_png(compact.right_coverage_count),
            "right_repair": _encode_png(compact.right_repair_bits),
        }
    return PreparedQualityFramePublication(
        publication=publication,
        left_rgb_png=left_rgb_png,
        right_rgb_png=right_rgb_png,
        mask_pngs=mask_pngs,
        stats=stats,
    )


def publish_quality_frame(prepared: PreparedQualityFramePublication) -> dict[str, Any]:
    """Write prepared RGB pair (and optional masks) atomically without per-frame fsync."""

    pub = prepared.publication
    atomic_write_bytes(pub.left_rgb_path, prepared.left_rgb_png, sync=False)
    atomic_write_bytes(pub.right_rgb_path, prepared.right_rgb_png, sync=False)

    payloads: dict[str, dict[str, Any] | None] = {
        "left_rgb": {
            "path": str(pub.left_rgb_path),
            "byte_count": len(prepared.left_rgb_png),
            "sha256": hashlib.sha256(prepared.left_rgb_png).hexdigest(),
        },
        "right_rgb": {
            "path": str(pub.right_rgb_path),
            "byte_count": len(prepared.right_rgb_png),
            "sha256": hashlib.sha256(prepared.right_rgb_png).hexdigest(),
        },
    }

    if pub.keep_masks and prepared.mask_pngs:
        frame_dir = pub.job_root / DIAGNOSTICS_DIR_NAME / "frames" / pub.frame_name
        frame_dir.mkdir(parents=True, exist_ok=True)
        for key, png_bytes in prepared.mask_pngs.items():
            mask_path = frame_dir / f"{key}.png"
            atomic_write_bytes(mask_path, png_bytes, sync=False)
            payloads[key] = {
                "path": str(mask_path),
                "byte_count": len(png_bytes),
                "sha256": hashlib.sha256(png_bytes).hexdigest(),
            }
        stats_bytes = canonical_json_bytes(_json_compatible(prepared.stats))
        atomic_write_bytes(frame_dir / "stats.json", stats_bytes, sync=False)
        payloads["stats"] = {
            "path": str(frame_dir / "stats.json"),
            "byte_count": len(stats_bytes),
            "sha256": hashlib.sha256(stats_bytes).hexdigest(),
        }

    return {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "frame_name": pub.frame_name,
        "render_mode": pub.render_mode,
        "geometry_mode": pub.geometry_mode,
        "occlusion_fill": pub.occlusion_fill,
        "payloads": payloads,
        "stats": prepared.stats,
    }


@dataclass(frozen=True)
class StreamingDiagnosticsStageContext:
    """Lightweight stage descriptor for in-memory stereo diagnostics aggregation."""

    job_root: Path
    frame_count: int
    frame_names: Callable[[], Iterable[str]]
    render_shape: tuple[int, int]
    render_mode: Literal["fast", "quality"]
    geometry_mode: Literal["relative", "metric"]
    occlusion_fill: Literal["none", "background"]
    mask_payloads_enabled: bool = False
    rgb_stage_fingerprint: str = ""


class StreamingDiagnosticsAccumulator:
    """O(1)-memory accumulator writing one summary JSON at stage completion."""

    def __init__(self, context: StreamingDiagnosticsStageContext) -> None:
        self.context = context
        self.frame_count = 0
        self.eye_totals: dict[str, dict[str, int]] = {
            "left": {"pixels": 0, "prefill_full": 0, "prefill_partial": 0, "final_unresolved": 0},
            "right": {"pixels": 0, "prefill_full": 0, "prefill_partial": 0, "final_unresolved": 0},
        }
        self.metric_valid_pixels = 0
        self.metric_clamped_pixels = 0
        self.metric_affected_frames = 0
        self.metric_clamped_fraction_sum = 0.0
        self.metric_max_clamped_fraction = 0.0

    def record_compact(
        self,
        compact: CompactStereoRenderResult,
        *,
        metric_stats: MetricProjectionStats | None = None,
    ) -> None:
        self.frame_count += 1
        height, width = self.context.render_shape
        pixels = height * width
        for eye_name, repair_bits in (
            ("left", compact.left_repair_bits),
            ("right", compact.right_repair_bits),
        ):
            totals = self.eye_totals[eye_name]
            totals["pixels"] += pixels
            if repair_bits is not None and repair_bits.size > 0:
                totals["prefill_partial"] += int(np.count_nonzero(repair_bits & np.uint8(0x01)))
                totals["prefill_full"] += int(np.count_nonzero(repair_bits & np.uint8(0x02)))
                totals["final_unresolved"] += int(np.count_nonzero(repair_bits & np.uint8(0x40)))
        if metric_stats is not None:
            self.record_metric_stats(metric_stats)

    def record_metric_stats(self, stats: MetricProjectionStats) -> None:
        self.metric_valid_pixels += int(stats.valid_pixel_count)
        self.metric_clamped_pixels += int(stats.clamped_pixel_count)
        if stats.clamped_pixel_count > 0:
            self.metric_affected_frames += 1
        frac = float(stats.clamped_fraction)
        self.metric_clamped_fraction_sum += frac
        if frac > self.metric_max_clamped_fraction:
            self.metric_max_clamped_fraction = frac

    def metric_clamp_summary(self) -> dict[str, Any] | None:
        if self.context.geometry_mode != "metric":
            return None
        denom = max(1, self.frame_count)
        return {
            "schema_version": 1,
            "valid_pixel_count": self.metric_valid_pixels,
            "clamped_pixel_count": self.metric_clamped_pixels,
            "affected_frame_count": self.metric_affected_frames,
            "mean_clamped_fraction": self.metric_clamped_fraction_sum / denom,
            "max_clamped_fraction": self.metric_max_clamped_fraction,
        }

    def write_summary(self) -> dict[str, Any]:
        diag_dir = self.context.job_root / DIAGNOSTICS_DIR_NAME
        diag_dir.mkdir(parents=True, exist_ok=True)
        summary: dict[str, Any] = {
            "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
            "status": "complete",
            "frame_count": self.frame_count,
            "render_shape": list(self.context.render_shape),
            "render_mode": self.context.render_mode,
            "geometry_mode": self.context.geometry_mode,
            "occlusion_fill": self.context.occlusion_fill,
            "mask_payloads_enabled": self.context.mask_payloads_enabled,
            "rgb_stage_fingerprint": self.context.rgb_stage_fingerprint,
            "eyes": self.eye_totals,
            "metric_clamp_summary": self.metric_clamp_summary(),
        }
        atomic_write_json(diag_dir / "summary.json", summary, indent=2, sort_keys=True, sync=True)
        return summary


def mark_diagnostics_pruned(job_root: Path) -> dict[str, Any]:
    """Update diagnostics summary status to ``payload_pruned`` if present."""

    diag_dir = Path(job_root) / DIAGNOSTICS_DIR_NAME
    summary_path = diag_dir / "summary.json"
    if not summary_path.is_file():
        return {"status": "payload_pruned"}
    import json

    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except (OSError, json.JSONDecodeError):
        data = {}
    data["status"] = "payload_pruned"
    atomic_write_json(summary_path, data, indent=2, sort_keys=True, sync=True)
    return data
