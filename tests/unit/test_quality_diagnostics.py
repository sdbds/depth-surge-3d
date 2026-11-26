"""Lightweight in-memory stereo diagnostics and optional mask publication tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.depth_surge_3d.rendering.quality.diagnostic_transaction import (
    QualityFramePublication,
    StreamingDiagnosticsAccumulator,
    StreamingDiagnosticsStageContext,
    mark_diagnostics_pruned,
    prepare_quality_frame,
    publish_quality_frame,
)
from src.depth_surge_3d.rendering.quality.diagnostics import build_quality_frame_stats
from src.depth_surge_3d.rendering.quality.types import (
    CompactStereoRenderResult,
    QualityRepairPlanSummary,
)
from src.depth_surge_3d.rendering.stereo_geometry import MetricProjectionStats


def _sample_compact() -> CompactStereoRenderResult:
    left = np.full((4, 6, 3), 10, dtype=np.uint8)
    right = np.full((4, 6, 3), 20, dtype=np.uint8)
    coverage = np.full((4, 6), 16, dtype=np.uint8)
    repair = np.zeros((4, 6), dtype=np.uint8)
    repair[0, 0] = np.uint8(0x02 | 0x40)
    plan = QualityRepairPlanSummary(
        segment_record_count=0,
        segment_table_bytes=0,
        backend_lane_counts=(0, 0, 0),
        backend_pixel_counts=(0, 0, 0),
        backend_component_counts=(0, 0, 0),
        local_evaluated_slot_count=0,
        local_physical_sample_read_count=0,
        local_budget_skipped_run_count=0,
        fallback_indexed_donor_count=0,
        fallback_query_count=0,
        fallback_anchor_count=0,
        fallback_visited_nodes_total=0,
        fallback_visited_nodes_max=0,
        fallback_visited_nodes_p95=0.0,
        exemplar_donor_evaluation_count=0,
    )
    limits = {
        "left": {"max_neighbour_abs_q_jump_px": 0.0, "predicted_gap_px": 0, "local_limit_px": 0},
        "right": {"max_neighbour_abs_q_jump_px": 0.0, "predicted_gap_px": 0, "local_limit_px": 0},
    }
    return CompactStereoRenderResult(
        left_image=left,
        right_image=right,
        left_coverage_count=coverage.copy(),
        right_coverage_count=coverage.copy(),
        left_repair_bits=repair.copy(),
        right_repair_bits=repair.copy(),
        left_plan=plan,
        right_plan=plan,
        quality_limits=limits,
        geometry_nearest={
            "indexed_sample_count": 24,
            "query_count": 0,
            "visited_nodes_total": 0,
            "visited_nodes_max": 0,
        },
    )


def test_prepare_and_publish_quality_frame_with_optional_masks(tmp_path: Path) -> None:
    compact = _sample_compact()
    pub = QualityFramePublication(
        job_root=tmp_path,
        frame_name="frame_000001",
        left_rgb_path=tmp_path / "04_left_frames" / "frame_000001.png",
        right_rgb_path=tmp_path / "04_right_frames" / "frame_000001.png",
        compact=compact,
        geometry_mode="relative",
        occlusion_fill="background",
        rgb_stage_fingerprint="0" * 64,
        keep_masks=True,
    )
    prepared = prepare_quality_frame(pub)
    manifest = publish_quality_frame(prepared)
    assert pub.left_rgb_path.is_file()
    assert pub.right_rgb_path.is_file()
    assert (
        tmp_path / "04_stereo_diagnostics" / "frames" / "frame_000001" / "left_coverage.png"
    ).is_file()
    assert (tmp_path / "04_stereo_diagnostics" / "frames" / "frame_000001" / "stats.json").is_file()
    assert manifest["frame_name"] == "frame_000001"

    stats = build_quality_frame_stats(
        compact,
        frame_name="frame_000001",
        geometry_mode="relative",
        occlusion_fill="background",
    )
    assert stats["eyes"]["left"]["pixel_count"] == 24


def test_streaming_diagnostics_accumulator_writes_single_summary(tmp_path: Path) -> None:
    ctx = StreamingDiagnosticsStageContext(
        job_root=tmp_path,
        frame_count=2,
        frame_names=lambda: ("frame_000001", "frame_000002"),
        render_shape=(4, 6),
        render_mode="quality",
        geometry_mode="metric",
        occlusion_fill="background",
        mask_payloads_enabled=False,
        rgb_stage_fingerprint="a" * 64,
    )
    acc = StreamingDiagnosticsAccumulator(ctx)
    compact = _sample_compact()
    acc.record_compact(
        compact,
        metric_stats=MetricProjectionStats(
            valid_pixel_count=24, clamped_pixel_count=3, clamped_fraction=3.0 / 24.0
        ),
    )
    acc.record_compact(
        compact,
        metric_stats=MetricProjectionStats(
            valid_pixel_count=24, clamped_pixel_count=0, clamped_fraction=0.0
        ),
    )
    summary = acc.write_summary()
    assert summary["frame_count"] == 2
    assert summary["eyes"]["left"]["prefill_full"] == 2
    assert summary["metric_clamp_summary"]["affected_frame_count"] == 1
    assert not (tmp_path / "04_stereo_diagnostics" / "frames").exists()

    summary_path = tmp_path / "04_stereo_diagnostics" / "summary.json"
    assert json.loads(summary_path.read_text(encoding="utf-8"))["status"] == "complete"
    pruned = mark_diagnostics_pruned(tmp_path)
    assert pruned["status"] == "payload_pruned"
    assert json.loads(summary_path.read_text(encoding="utf-8"))["status"] == "payload_pruned"
