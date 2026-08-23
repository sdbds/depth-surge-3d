from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.depth_surge_3d.rendering.quality.types import CompactStereoRenderResult
from src.depth_surge_3d.rendering.stereo_renderer import (
    StereoRenderer,
    StereoRenderSettings,
)


def _compact() -> CompactStereoRenderResult:
    left = np.full((2, 3, 3), 10, dtype=np.uint8)
    right = np.full((2, 3, 3), 20, dtype=np.uint8)
    coverage = np.full((2, 3), 16, dtype=np.uint8)
    repair = np.zeros((2, 3), dtype=np.uint8)
    return CompactStereoRenderResult(
        left_image=left,
        right_image=right,
        left_coverage_count=coverage.copy(),
        right_coverage_count=coverage.copy(),
        left_repair_bits=repair.copy(),
        right_repair_bits=repair.copy(),
    )


def _publication(
    module,
    root: Path,
    *,
    keep_masks: bool = True,
    frame_name: str = "frame_0001",
):
    return module.QualityFramePublication(
        job_root=root,
        frame_name=frame_name,
        left_rgb_path=root / "04_left_frames" / f"{frame_name}.png",
        right_rgb_path=root / "04_right_frames" / f"{frame_name}.png",
        compact=_compact(),
        geometry_mode="relative",
        occlusion_fill="background",
        rgb_stage_fingerprint="rgb-stage",
        keep_masks=keep_masks,
    )


def test_quality_frame_manifest_is_published_last_with_exact_payload_hashes(
    tmp_path: Path,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    publication = _publication(diagnostic_transaction, tmp_path)

    manifest = diagnostic_transaction.publish_quality_frame(publication)

    frame_dir = tmp_path / "04_stereo_diagnostics" / "frames" / "frame_0001"
    raw_manifest = (frame_dir / "manifest.json").read_bytes()
    assert json.loads(raw_manifest) == manifest
    assert manifest["fingerprint"] == diagnostic_transaction.self_fingerprint(manifest)
    for key, identity in manifest["payloads"].items():
        assert identity is not None, key
        payload_path = tmp_path / Path(identity["relative_path"])
        raw = payload_path.read_bytes()
        assert identity["byte_count"] == len(raw)
        assert identity["sha256"] == hashlib.sha256(raw).hexdigest()


def test_quality_frame_can_omit_mask_payloads_without_omitting_stats(
    tmp_path: Path,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    manifest = diagnostic_transaction.publish_quality_frame(
        _publication(diagnostic_transaction, tmp_path, keep_masks=False)
    )

    assert manifest["mask_payloads_enabled"] is False
    assert manifest["payloads"]["left_coverage"] is None
    assert manifest["payloads"]["right_repair"] is None
    assert manifest["payloads"]["stats"] is not None


def test_failed_quality_frame_transaction_removes_every_final_payload(
    tmp_path: Path,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    def fault(stage: str) -> None:
        if stage == "before_manifest_replace":
            raise OSError("injected")

    publication = _publication(diagnostic_transaction, tmp_path)

    with pytest.raises(OSError, match="injected"):
        diagnostic_transaction.publish_quality_frame(publication, fault=fault)

    assert not publication.left_rgb_path.exists()
    assert not publication.right_rgb_path.exists()
    frame_dir = tmp_path / "04_stereo_diagnostics" / "frames" / "frame_0001"
    assert not any(frame_dir.glob("*.png"))
    assert not (frame_dir / "stats.json").exists()
    assert not (frame_dir / "manifest.json").exists()


@pytest.mark.parametrize(
    "failure_stage",
    [
        "after_left_rgb_temporary_write",
        "after_left_rgb_temporary_sync",
        "after_left_rgb_replace",
        "after_left_rgb_directory_sync",
        "after_stats_directory_sync",
        "before_manifest_replace",
        "after_manifest_replace",
        "after_manifest_directory_sync",
    ],
)
def test_frame_transaction_crash_boundaries_never_leave_a_committed_prefix(
    tmp_path: Path,
    failure_stage: str,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    observed: list[str] = []

    def fault(stage: str) -> None:
        observed.append(stage)
        if stage == failure_stage:
            raise OSError("injected")

    publication = _publication(diagnostic_transaction, tmp_path)
    with pytest.raises(OSError, match="injected"):
        diagnostic_transaction.publish_quality_frame(publication, fault=fault)

    assert failure_stage in observed
    frame_dir = tmp_path / "04_stereo_diagnostics" / "frames" / "frame_0001"
    assert not (frame_dir / "manifest.json").exists()
    assert not publication.left_rgb_path.exists()
    assert not publication.right_rgb_path.exists()


def test_quality_none_stats_stream_exact_hole_runs_without_a_repair_plan() -> None:
    from src.depth_surge_3d.rendering.quality.diagnostics import (
        build_quality_frame_stats,
    )
    from src.depth_surge_3d.rendering.quality.renderer import (
        render_relative_quality_compact,
    )

    compact = render_relative_quality_compact(
        renderer=StereoRenderer(device="cpu"),
        frame=np.full((2, 32, 3), 100, dtype=np.uint8),
        canonical=np.ones((1, 8), dtype=np.float32),
        settings=StereoRenderSettings(
            stereo_strength=5.0,
            convergence=0.0,
            occlusion_fill="none",
            stereo_render_mode="quality",
        ),
    )

    stats = build_quality_frame_stats(
        compact,
        frame_name="frame_0000",
        geometry_mode="relative",
        occlusion_fill="none",
    )

    for eye in ("left", "right"):
        eye_stats = stats["eyes"][eye]
        assert eye_stats["hole_runs"]["count"] > 0
        assert eye_stats["final_unresolved_lane_count"] == int(
            np.sum(16 - getattr(compact, f"{eye}_coverage_count"), dtype=np.uint64)
        )
        assert eye_stats["quality_budgets"]["segment_record_count"] is None


def test_diagnostics_consolidation_is_source_ordered_and_metadata_last(
    tmp_path: Path,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    context = diagnostic_transaction.DiagnosticsStageContext(
        job_root=tmp_path,
        frame_names=("frame_0000", "frame_0001"),
        render_shape=(2, 3),
        render_mode="quality",
        geometry_mode="relative",
        occlusion_fill="background",
        mask_payloads_enabled=False,
        rgb_stage_fingerprint="rgb-stage",
        source_guide_fingerprint="guide-stage",
        native_geometry_fingerprint="geometry-stage",
        quality_input_manifest_sha256="a" * 64,
    )
    diagnostic_transaction.begin_diagnostics_stage(context)
    diagnostic_transaction.publish_quality_frame(
        _publication(
            diagnostic_transaction,
            tmp_path,
            keep_masks=False,
            frame_name="frame_0001",
        )
    )
    diagnostic_transaction.publish_quality_frame(
        _publication(
            diagnostic_transaction,
            tmp_path,
            keep_masks=False,
            frame_name="frame_0000",
        )
    )

    metadata = diagnostic_transaction.complete_diagnostics_stage(context)

    diagnostics_root = tmp_path / "04_stereo_diagnostics"
    jsonl = (diagnostics_root / "stereo_coverage_frames.jsonl").read_bytes()
    rows = [json.loads(line) for line in jsonl.splitlines()]
    assert [row["frame_name"] for row in rows] == ["frame_0000", "frame_0001"]
    assert metadata["status"] == "complete"
    assert metadata["frames_jsonl_sha256"] == hashlib.sha256(jsonl).hexdigest()
    audit = diagnostic_transaction.audit_diagnostics_stage(context)
    assert audit.disposition == "complete"
    assert audit.invalid_frame_names == ()


def test_diagnostics_audit_identifies_only_the_corrupt_frame(tmp_path: Path) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    context = diagnostic_transaction.DiagnosticsStageContext(
        job_root=tmp_path,
        frame_names=("frame_0000", "frame_0001"),
        render_shape=(2, 3),
        render_mode="quality",
        geometry_mode="relative",
        occlusion_fill="background",
        mask_payloads_enabled=True,
        rgb_stage_fingerprint="rgb-stage",
        source_guide_fingerprint="guide-stage",
        native_geometry_fingerprint="geometry-stage",
        quality_input_manifest_sha256="b" * 64,
    )
    diagnostic_transaction.begin_diagnostics_stage(context)
    for frame_name in context.frame_names:
        diagnostic_transaction.publish_quality_frame(
            _publication(
                diagnostic_transaction,
                tmp_path,
                frame_name=frame_name,
            )
        )
    diagnostic_transaction.complete_diagnostics_stage(context)
    (tmp_path / "04_left_frames" / "frame_0001.png").write_bytes(b"damaged")

    audit = diagnostic_transaction.audit_diagnostics_stage(context)

    assert audit.disposition == "building"
    assert audit.invalid_frame_names == ("frame_0001",)
    assert audit.reusable_frame_names == ("frame_0000",)
