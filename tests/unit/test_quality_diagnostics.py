from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from src.depth_surge_3d.rendering.quality.types import CompactStereoRenderResult
from src.depth_surge_3d.rendering.stereo_renderer import (
    StereoRenderer,
    StereoRenderSettings,
)


def _compact() -> CompactStereoRenderResult:
    from src.depth_surge_3d.rendering.quality.planner import QualityRepairPlanSummary

    left = np.full((2, 3, 3), 10, dtype=np.uint8)
    right = np.full((2, 3, 3), 20, dtype=np.uint8)
    coverage = np.full((2, 3), 16, dtype=np.uint8)
    repair = np.zeros((2, 3), dtype=np.uint8)
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
        "left": {
            "max_neighbour_abs_q_jump_px": 0.0,
            "predicted_gap_px": 0,
            "local_limit_px": 0,
        },
        "right": {
            "max_neighbour_abs_q_jump_px": 0.0,
            "predicted_gap_px": 0,
            "local_limit_px": 0,
        },
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
            "indexed_sample_count": 6,
            "query_count": 0,
            "visited_nodes_total": 0,
            "visited_nodes_max": 0,
        },
    )


def _publication(
    module,
    root: Path,
    *,
    compact: CompactStereoRenderResult | None = None,
    keep_masks: bool = True,
    frame_name: str = "frame_0001",
    render_mode: str = "quality",
):
    return module.QualityFramePublication(
        job_root=root,
        frame_name=frame_name,
        left_rgb_path=root / "04_left_frames" / f"{frame_name}.png",
        right_rgb_path=root / "04_right_frames" / f"{frame_name}.png",
        compact=compact or _compact(),
        geometry_mode="relative",
        occlusion_fill="background",
        rgb_stage_fingerprint="rgb-stage",
        keep_masks=keep_masks,
        render_mode=render_mode,
    )


def _streaming_context(module, root: Path, *, masks: bool):
    names = ("frame_0000",)
    return module.StreamingDiagnosticsStageContext(
        job_root=root,
        frame_count=1,
        frame_names=lambda: iter(names),
        render_shape=(2, 3),
        render_mode="quality",
        geometry_mode="relative",
        occlusion_fill="background",
        mask_payloads_enabled=masks,
        rgb_stage_fingerprint="rgb-stage",
        source_guide_fingerprint="guide-stage",
        native_geometry_fingerprint="geometry-stage",
        quality_input_manifest_sha256="f" * 64,
        native_geometry_sample_count=6,
    )


def _legacy_fast_case(module, root: Path):
    from src.depth_surge_3d.io.resume import LegacyFastAdoptionV1

    names = ("frame_000001", "frame_000002")
    context = module.StreamingDiagnosticsStageContext(
        job_root=root,
        frame_count=len(names),
        frame_names=lambda: iter(names),
        render_shape=(4, 6),
        render_mode="fast",
        geometry_mode="relative",
        occlusion_fill="background",
        mask_payloads_enabled=False,
        rgb_stage_fingerprint="a" * 64,
        source_guide_fingerprint="b" * 64,
        native_geometry_fingerprint=None,
        quality_input_manifest_sha256=None,
    )
    adoption = LegacyFastAdoptionV1(
        job_root=root.resolve(),
        source_settings_schema_version=4,
        settings_fingerprint="c" * 64,
        frame_count=len(names),
        render_shape=(4, 6),
        geometry_mode="relative",
        occlusion_fill="background",
        rgb_stage_fingerprint="a" * 64,
        source_guide_fingerprint="b" * 64,
        stage_metadata_sha256="d" * 64,
        payload_sequence_sha256="e" * 64,
        metric_clamp_summary_sha256=None,
        metric_clamp_evidence_sha256=None,
        metric_valid_pixel_count=None,
        metric_clamped_pixel_count=None,
        metric_affected_frame_count=None,
        metric_mean_clamped_fraction=None,
        metric_max_clamped_fraction=None,
    )
    return context, adoption


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


def test_shared_quality_png_payloads_encode_once_across_concurrent_publications(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    compact = _compact()
    shared = diagnostic_transaction.SharedQualityPngPayloads(
        compact=compact,
        keep_masks=False,
    )
    real_encode = diagnostic_transaction._png_buffer
    encode_calls = 0
    encode_lock = threading.Lock()

    def count_encode(values):
        nonlocal encode_calls
        with encode_lock:
            encode_calls += 1
        time.sleep(0.02)
        return real_encode(values)

    monkeypatch.setattr(diagnostic_transaction, "_png_buffer", count_encode)

    def publish(index: int):
        publication = _publication(
            diagnostic_transaction,
            tmp_path,
            compact=compact,
            keep_masks=False,
            frame_name=f"frame_{index:06d}",
        )
        prepared = diagnostic_transaction.prepare_quality_frame(
            publication,
            png_payloads=shared,
        )
        return diagnostic_transaction.publish_quality_frame(prepared)

    with ThreadPoolExecutor(max_workers=3) as pool:
        manifests = list(pool.map(publish, range(1, 4)))

    assert encode_calls == 2
    assert [manifest["frame_name"] for manifest in manifests] == [
        "frame_000001",
        "frame_000002",
        "frame_000003",
    ]
    for side in ("04_left_frames", "04_right_frames"):
        payloads = [
            (tmp_path / side / f"frame_{index:06d}.png").read_bytes() for index in range(1, 4)
        ]
        assert len(set(payloads)) == 1


def test_shared_quality_png_payloads_reject_a_different_compact_result(tmp_path: Path) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    shared = diagnostic_transaction.SharedQualityPngPayloads(
        compact=_compact(),
        keep_masks=False,
    )
    publication = _publication(
        diagnostic_transaction,
        tmp_path,
        compact=_compact(),
        keep_masks=False,
    )

    with pytest.raises(ValueError, match="do not match"):
        diagnostic_transaction.prepare_quality_frame(
            publication,
            png_payloads=shared,
        )


def test_parallel_quality_png_encoding_is_byte_identical_and_concurrent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    serial_root = tmp_path / "serial"
    parallel_root = tmp_path / "parallel"
    serial_manifest = diagnostic_transaction.publish_quality_frame(
        _publication(diagnostic_transaction, serial_root)
    )
    real_encode = diagnostic_transaction._png_buffer
    active = 0
    peak = 0
    lock = threading.Lock()

    def observe_encode(values):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.02)
            return real_encode(values)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(diagnostic_transaction, "_png_buffer", observe_encode)
    with ThreadPoolExecutor(max_workers=4) as pool:
        parallel_manifest = diagnostic_transaction.publish_quality_frame(
            _publication(diagnostic_transaction, parallel_root),
            png_executor=pool,
        )

    assert peak >= 2
    assert parallel_manifest == serial_manifest
    serial_files = {
        path.relative_to(serial_root): path.read_bytes()
        for path in serial_root.rglob("*")
        if path.is_file()
    }
    parallel_files = {
        path.relative_to(parallel_root): path.read_bytes()
        for path in parallel_root.rglob("*")
        if path.is_file()
    }
    assert parallel_files == serial_files


def test_parallel_quality_png_failure_waits_and_publishes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    root = tmp_path / "failed"
    real_encode = diagnostic_transaction._png_buffer
    active = 0
    calls = 0
    lock = threading.Lock()
    concurrent_start = threading.Event()

    def fail_one_encode(values):
        nonlocal active, calls
        with lock:
            calls += 1
            call = calls
            active += 1
            if active >= 2:
                concurrent_start.set()
        try:
            if call == 1:
                assert concurrent_start.wait(timeout=2.0)
                raise OSError("injected PNG encode failure")
            time.sleep(0.08)
            return real_encode(values)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(diagnostic_transaction, "_png_buffer", fail_one_encode)
    with ThreadPoolExecutor(max_workers=4) as pool:
        with pytest.raises(OSError, match="injected PNG encode failure"):
            diagnostic_transaction.publish_quality_frame(
                _publication(diagnostic_transaction, root),
                png_executor=pool,
            )
        with lock:
            assert active == 0
        assert not [path for path in root.rglob("*") if path.is_file()]


def test_parallel_quality_png_submit_failure_drains_started_encodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    real_encode = diagnostic_transaction._png_buffer
    started = threading.Event()
    active = 0
    lock = threading.Lock()

    def slow_encode(values):
        nonlocal active
        with lock:
            active += 1
        started.set()
        try:
            time.sleep(0.08)
            return real_encode(values)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(diagnostic_transaction, "_png_buffer", slow_encode)
    with ThreadPoolExecutor(max_workers=2) as pool:
        real_submit = pool.submit
        submissions = 0

        def fail_third_submit(fn, *args, **kwargs):
            nonlocal submissions
            submissions += 1
            if submissions == 3:
                assert started.wait(timeout=2.0)
                raise RuntimeError("injected PNG submit failure")
            return real_submit(fn, *args, **kwargs)

        monkeypatch.setattr(pool, "submit", fail_third_submit)
        with pytest.raises(RuntimeError, match="injected PNG submit failure"):
            diagnostic_transaction.publish_quality_frame(
                _publication(diagnostic_transaction, tmp_path),
                png_executor=pool,
            )

        with lock:
            assert active == 0
        assert not [path for path in tmp_path.rglob("*") if path.is_file()]


def test_parallel_shared_png_failure_drains_every_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    compact = _compact()
    shared = diagnostic_transaction.SharedQualityPngPayloads(compact=compact, keep_masks=False)
    real_encode = diagnostic_transaction._png_buffer
    active = 0
    left_calls = 0
    lock = threading.Lock()
    publication_start = threading.Barrier(3)

    def fail_shared_left(values):
        nonlocal active, left_calls
        with lock:
            active += 1
            if values is compact.left_image:
                left_calls += 1
        try:
            time.sleep(0.04)
            if values is compact.left_image:
                raise OSError("injected shared PNG failure")
            return real_encode(values)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(diagnostic_transaction, "_png_buffer", fail_shared_left)
    with ThreadPoolExecutor(max_workers=4) as encoders:

        def publish(index: int):
            prepared = diagnostic_transaction.prepare_quality_frame(
                _publication(
                    diagnostic_transaction,
                    tmp_path,
                    compact=compact,
                    keep_masks=False,
                    frame_name=f"frame_{index:06d}",
                ),
                png_payloads=shared,
            )
            publication_start.wait(timeout=2.0)
            return diagnostic_transaction.publish_quality_frame(prepared, png_executor=encoders)

        with ThreadPoolExecutor(max_workers=3) as writers:
            publications = [writers.submit(publish, index) for index in range(1, 4)]
            for publication in publications:
                with pytest.raises(OSError, match="shared PNG"):
                    publication.result(timeout=5.0)
            with lock:
                assert active == 0
                assert left_calls == 1
            assert not [path for path in tmp_path.rglob("*") if path.is_file()]


def test_quality_frame_preserves_an_unowned_payload_temporary(tmp_path: Path) -> None:
    from src.depth_surge_3d.io.durable_publish import (
        OwnedTemporaryCollisionError,
        publication_temporary_path,
    )
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    publication = _publication(diagnostic_transaction, tmp_path)
    temporary = publication_temporary_path(publication.left_rgb_path)
    temporary.parent.mkdir(parents=True)
    temporary.write_bytes(b"foreign-frame-temporary")

    with pytest.raises(OwnedTemporaryCollisionError):
        diagnostic_transaction.publish_quality_frame(publication)

    assert temporary.read_bytes() == b"foreign-frame-temporary"
    assert not publication.left_rgb_path.exists()


def test_streaming_jsonl_preserves_an_unowned_temporary(tmp_path: Path) -> None:
    from src.depth_surge_3d.io.durable_publish import (
        OwnedTemporaryCollisionError,
        publication_temporary_path,
    )
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    context = _streaming_context(diagnostic_transaction, tmp_path, masks=False)
    jsonl = tmp_path / "04_stereo_diagnostics" / "stereo_coverage_frames.jsonl"
    temporary = publication_temporary_path(jsonl)
    temporary.parent.mkdir(parents=True)
    temporary.write_bytes(b"foreign-jsonl-temporary")

    with pytest.raises(OwnedTemporaryCollisionError):
        diagnostic_transaction.complete_streaming_diagnostics_stage(context)

    assert temporary.read_bytes() == b"foreign-jsonl-temporary"
    assert not jsonl.exists()


def test_legacy_fast_unavailable_publishes_empty_aggregate_without_frame_manifests(
    tmp_path: Path,
) -> None:
    from src.depth_surge_3d.io.resume import (
        LEGACY_FAST_ADOPTION_INTENT_RELATIVE_PATH,
        publish_legacy_fast_adoption_intent_v1,
    )
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    context, adoption = _legacy_fast_case(diagnostic_transaction, tmp_path)
    names = tuple(context.frame_names())
    publish_legacy_fast_adoption_intent_v1(adoption)

    completion = diagnostic_transaction.publish_legacy_fast_unavailable(
        context,
        adoption,
    )

    root = tmp_path / "04_stereo_diagnostics"
    assert (root / "stereo_coverage_frames.jsonl").read_bytes() == b""
    assert not (root / "frames").exists()
    summary = json.loads((root / "stereo_coverage_summary.json").read_bytes())
    assert summary["availability"] == "legacy_fast_unavailable"
    assert summary["frame_names"] == list(names)
    assert summary["eyes"] is None
    assert summary["metric_projection"] is None
    assert summary["geometry_nearest"] is None
    projection = diagnostic_transaction.project_diagnostics_metadata(
        root / "metadata.json"
    ).validate_state_contract()
    assert projection.status == "legacy_fast_unavailable"
    assert projection.fingerprint == completion.metadata.fingerprint
    assert not (tmp_path / LEGACY_FAST_ADOPTION_INTENT_RELATIVE_PATH).exists()
    assert projection.static_fields["ordered_frame_manifest_fingerprint"] == (
        hashlib.sha256(b"[]").hexdigest()
    )


@pytest.mark.parametrize(
    "crash_event",
    [
        "after_legacy_root_directory_sync",
        "after_frames_jsonl_replace",
        "after_root_summary_replace",
        "before_root_metadata_replace",
        "after_root_metadata_replace",
    ],
)
def test_legacy_fast_unavailable_recovers_every_durable_prefix(
    tmp_path: Path,
    crash_event: str,
) -> None:
    from src.depth_surge_3d.io.resume import (
        LEGACY_FAST_ADOPTION_INTENT_RELATIVE_PATH,
        publish_legacy_fast_adoption_intent_v1,
    )
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    context, adoption = _legacy_fast_case(diagnostic_transaction, tmp_path)
    publish_legacy_fast_adoption_intent_v1(adoption)

    def crash(stage: str) -> None:
        if stage == crash_event:
            raise RuntimeError("simulated process loss")

    with pytest.raises(RuntimeError, match="simulated process loss"):
        diagnostic_transaction.publish_legacy_fast_unavailable(
            context,
            adoption,
            fault=crash,
        )

    assert (tmp_path / LEGACY_FAST_ADOPTION_INTENT_RELATIVE_PATH).is_file()
    diagnostic_transaction.publish_legacy_fast_unavailable(context, adoption)

    root = tmp_path / "04_stereo_diagnostics"
    assert {path.name for path in root.iterdir()} == {
        "metadata.json",
        "stereo_coverage_frames.jsonl",
        "stereo_coverage_summary.json",
    }
    assert not (tmp_path / LEGACY_FAST_ADOPTION_INTENT_RELATIVE_PATH).exists()


def test_legacy_fast_unavailable_preserves_foreign_prefix_entry(tmp_path: Path) -> None:
    from src.depth_surge_3d.io.resume import publish_legacy_fast_adoption_intent_v1
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    context, adoption = _legacy_fast_case(diagnostic_transaction, tmp_path)
    publish_legacy_fast_adoption_intent_v1(adoption)
    root = tmp_path / "04_stereo_diagnostics"
    root.mkdir()
    foreign = root / "foreign.bin"
    foreign.write_bytes(b"keep me")

    with pytest.raises(ValueError, match="foreign entry"):
        diagnostic_transaction.publish_legacy_fast_unavailable(context, adoption)

    assert foreign.read_bytes() == b"keep me"


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


def test_quality_frame_encodes_and_writes_each_png_before_encoding_the_next(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    original_encode = diagnostic_transaction.cv2.imencode
    original_write = diagnostic_transaction._write_temporary
    pending_png_buffers = 0

    def encode(*args, **kwargs):
        nonlocal pending_png_buffers
        pending_png_buffers += 1
        return original_encode(*args, **kwargs)

    def write(path, raw, **kwargs):
        nonlocal pending_png_buffers
        if path.suffix == ".png":
            assert pending_png_buffers == 1
            pending_png_buffers = 0
        return original_write(path, raw, **kwargs)

    monkeypatch.setattr(diagnostic_transaction.cv2, "imencode", encode)
    monkeypatch.setattr(diagnostic_transaction, "_write_temporary", write)

    diagnostic_transaction.publish_quality_frame(_publication(diagnostic_transaction, tmp_path))

    assert pending_png_buffers == 0


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


def test_frame_stats_keep_dense_histograms_until_canonical_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction
    from src.depth_surge_3d.rendering.quality.diagnostics import (
        build_quality_frame_stats,
    )
    from src.depth_surge_3d.rendering.quality.renderer import (
        render_relative_quality_compact,
    )
    from src.depth_surge_3d.rendering.quality.types import DenseHistogramPairs

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
    with monkeypatch.context() as stats_patch:
        stats_patch.setattr(
            "src.depth_surge_3d.rendering.quality.diagnostics.np.flatnonzero",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("dense histogram expansion is forbidden")
            ),
        )

        stats = build_quality_frame_stats(
            compact,
            frame_name="frame_000000",
            geometry_mode="relative",
            occlusion_fill="none",
        )

    dense = stats["eyes"]["left"]["hole_runs"]["lane_count_histogram"]
    assert isinstance(dense, DenseHistogramPairs)
    parsed = json.loads(diagnostic_transaction.canonical_json_bytes(stats))
    assert isinstance(parsed["eyes"]["left"]["hole_runs"]["lane_count_histogram"], list)


def test_streaming_frame_stats_replays_sparse_histograms_without_dense_frame_bins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    diagnostic_transaction.publish_quality_frame(
        _publication(diagnostic_transaction, tmp_path, keep_masks=False)
    )
    raw = (tmp_path / "04_stereo_diagnostics" / "frames" / "frame_0001" / "stats.json").read_bytes()
    expected = json.loads(raw)
    monkeypatch.setattr(
        diagnostic_transaction.np,
        "zeros",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("frame parser must not allocate dense histogram bins")
        ),
    )

    parsed = diagnostic_transaction._parse_streaming_frame_stats(raw, render_width=3)

    for eye in ("left", "right"):
        holes = parsed["eyes"][eye]["hole_runs"]
        assert list(holes["lane_count_histogram"]) == [
            tuple(pair) for pair in expected["eyes"][eye]["hole_runs"]["lane_count_histogram"]
        ]
        assert list(holes["touched_pixel_span_histogram"]) == [
            tuple(pair)
            for pair in expected["eyes"][eye]["hole_runs"]["touched_pixel_span_histogram"]
        ]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda stats: stats.__setitem__("schema_version", True),
        lambda stats: stats["eyes"]["left"]["hole_runs"].__setitem__("lane_count_p95", 0),
        lambda stats: stats["eyes"]["left"]["hole_runs"].__setitem__("lane_count_p95", -0.0),
        lambda stats: stats["eyes"]["left"]["state_pixel_counts"].__setitem__("unexpected", 0),
        lambda stats: stats["eyes"]["left"]["backend_lane_counts"].__setitem__("local", -1),
        lambda stats: stats["eyes"]["left"]["backend_lane_counts"].__setitem__("local", 97),
        lambda stats: stats["geometry_nearest"].__setitem__("indexed_sample_count", 5),
    ],
    ids=(
        "bool-as-int",
        "int-as-float",
        "negative-zero",
        "nested-extra",
        "negative",
        "oversized",
        "geometry-cardinality",
    ),
)
def test_frame_statistics_schema_rejects_noncanonical_nested_values(
    tmp_path: Path,
    mutate,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    diagnostic_transaction.publish_quality_frame(
        _publication(diagnostic_transaction, tmp_path, keep_masks=False)
    )
    raw = (tmp_path / "04_stereo_diagnostics" / "frames" / "frame_0001" / "stats.json").read_bytes()
    stats = diagnostic_transaction._parse_streaming_frame_stats(raw, render_width=3)
    mutate(stats)
    context = _streaming_context(diagnostic_transaction, tmp_path, masks=False).one_frame(
        "frame_0001"
    )

    with pytest.raises(ValueError):
        diagnostic_transaction._validate_frame_statistics_schema(context, stats)


def test_diagnostics_checked_u64_add_rejects_aggregate_overflow() -> None:
    from src.depth_surge_3d.rendering.quality.diagnostics import (
        DiagnosticsCardinalityError,
        checked_u64_add,
    )

    with pytest.raises(DiagnosticsCardinalityError):
        checked_u64_add("test_total", 2**64 - 1, 1)


def test_histogram_metrics_are_exact_and_reject_uint64_overflow() -> None:
    from src.depth_surge_3d.rendering.quality.diagnostics import (
        DiagnosticsCardinalityError,
        _histogram_metrics,
    )

    histogram = np.array([0, 1, 2, 1], dtype=np.uint64)

    assert _histogram_metrics(histogram, 0.95) == (4, 3, 2.8499999999999996)
    with pytest.raises(DiagnosticsCardinalityError):
        _histogram_metrics(
            np.array([np.iinfo(np.uint64).max, 1], dtype=np.uint64),
            0.95,
        )


def test_diagnostics_directory_audit_rejects_foreign_entry_without_draining() -> None:
    from src.depth_surge_3d.io.nofollow_io import ContainedDirectoryEntry
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    class LazyDirectory:
        def iter_directory(self, _root):
            yield ContainedDirectoryEntry("metadata.json", False, True)
            yield ContainedDirectoryEntry("foreign", False, True)
            raise AssertionError("closed namespace audit drained past the first foreign entry")

    assert not diagnostic_transaction._directory_contains_only_regular_names(
        LazyDirectory(),
        Path("unused"),
        allowed_names={"metadata.json"},
        require_exact=True,
    )


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
        native_geometry_sample_count=6,
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


def test_streaming_diagnostics_consolidation_replays_names_without_a_sequence_graph(
    tmp_path: Path,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    names = ("frame_0000", "frame_0001")
    context = diagnostic_transaction.StreamingDiagnosticsStageContext(
        job_root=tmp_path,
        frame_count=2,
        frame_names=lambda: iter(names),
        render_shape=(2, 3),
        render_mode="quality",
        geometry_mode="relative",
        occlusion_fill="background",
        mask_payloads_enabled=False,
        rgb_stage_fingerprint="rgb-stage",
        source_guide_fingerprint="guide-stage",
        native_geometry_fingerprint="geometry-stage",
        quality_input_manifest_sha256="c" * 64,
        native_geometry_sample_count=6,
    )
    diagnostic_transaction.begin_streaming_diagnostics_stage(context)
    for frame_name in reversed(names):
        diagnostic_transaction.publish_quality_frame(
            _publication(
                diagnostic_transaction,
                tmp_path,
                keep_masks=False,
                frame_name=frame_name,
            )
        )

    completion = diagnostic_transaction.complete_streaming_diagnostics_stage(context)

    root = tmp_path / "04_stereo_diagnostics"
    rows = [
        json.loads(line)
        for line in (root / "stereo_coverage_frames.jsonl").read_bytes().splitlines()
    ]
    assert [row["frame_name"] for row in rows] == list(names)
    assert completion.metadata.raw_byte_count == (root / "metadata.json").stat().st_size
    assert diagnostic_transaction.streaming_diagnostics_aggregates_match(context)
    assert (
        diagnostic_transaction.classify_streaming_diagnostics_frame(
            context,
            names[0],
        )
        == "reuse"
    )

    (tmp_path / "04_left_frames" / f"{names[1]}.png").write_bytes(b"damaged")
    assert (
        diagnostic_transaction.classify_streaming_diagnostics_frame(
            context,
            names[1],
        )
        == "render"
    )


def test_transient_diagnostics_spool_completes_without_per_frame_files(
    tmp_path: Path,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    names = ("frame_0000", "frame_0001")
    context = diagnostic_transaction.StreamingDiagnosticsStageContext(
        job_root=tmp_path,
        frame_count=len(names),
        frame_names=lambda: iter(names),
        render_shape=(2, 3),
        render_mode="quality",
        geometry_mode="relative",
        occlusion_fill="background",
        mask_payloads_enabled=False,
        rgb_stage_fingerprint="rgb-stage",
        source_guide_fingerprint="guide-stage",
        native_geometry_fingerprint="geometry-stage",
        quality_input_manifest_sha256="d" * 64,
        native_geometry_sample_count=6,
    )
    diagnostic_transaction.begin_streaming_diagnostics_stage(context)
    spool = diagnostic_transaction.TransientDiagnosticsSpool(context)
    for frame_name in names:
        prepared = diagnostic_transaction.prepare_quality_frame(
            _publication(
                diagnostic_transaction,
                tmp_path,
                keep_masks=False,
                frame_name=frame_name,
            )
        )
        spool.append(frame_name, prepared.stats_raw)

    completion = spool.complete()

    root = tmp_path / "04_stereo_diagnostics"
    rows = [
        json.loads(line)
        for line in (root / "stereo_coverage_frames.jsonl").read_bytes().splitlines()
    ]
    assert [row["frame_name"] for row in rows] == list(names)
    assert not (root / "frames").exists()
    assert completion.metadata.static_fields["status"] == "complete"
    assert diagnostic_transaction.classify_streaming_diagnostics_root(context) == "complete"
    assert diagnostic_transaction.streaming_diagnostics_aggregates_match(context) is False
    assert (
        diagnostic_transaction.classify_streaming_diagnostics_frame(context, names[0]) == "render"
    )


def test_transient_diagnostics_spool_rejects_wrong_order_and_cleans_temporary(
    tmp_path: Path,
) -> None:
    from src.depth_surge_3d.io.durable_publish import publication_temporary_path
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    names = ("frame_0000", "frame_0001")
    context = diagnostic_transaction.StreamingDiagnosticsStageContext(
        job_root=tmp_path,
        frame_count=len(names),
        frame_names=lambda: iter(names),
        render_shape=(2, 3),
        render_mode="quality",
        geometry_mode="relative",
        occlusion_fill="background",
        mask_payloads_enabled=False,
        rgb_stage_fingerprint="rgb-stage",
        source_guide_fingerprint="guide-stage",
        native_geometry_fingerprint="geometry-stage",
        quality_input_manifest_sha256="e" * 64,
        native_geometry_sample_count=6,
    )
    prepared = diagnostic_transaction.prepare_quality_frame(
        _publication(
            diagnostic_transaction,
            tmp_path,
            keep_masks=False,
            frame_name=names[1],
        )
    )
    spool = diagnostic_transaction.TransientDiagnosticsSpool(context)

    with pytest.raises(ValueError, match="source order"):
        spool.append(names[1], prepared.stats_raw)
    spool.abort()

    jsonl_path = tmp_path / "04_stereo_diagnostics" / "stereo_coverage_frames.jsonl"
    assert not publication_temporary_path(jsonl_path).exists()


def test_transient_diagnostics_spool_rejects_malformed_statistics(tmp_path: Path) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    context = _streaming_context(diagnostic_transaction, tmp_path, masks=False)
    spool = diagnostic_transaction.TransientDiagnosticsSpool(context)
    try:
        with pytest.raises(ValueError):
            spool.append("frame_0000", b"{}")
    finally:
        spool.abort()


def test_streaming_summary_buffers_jsonl_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    context = _streaming_context(diagnostic_transaction, tmp_path, masks=False)
    diagnostic_transaction.begin_streaming_diagnostics_stage(context)
    diagnostic_transaction.publish_quality_frame(
        _publication(
            diagnostic_transaction,
            tmp_path,
            keep_masks=False,
            frame_name="frame_0000",
        )
    )
    diagnostic_transaction.complete_streaming_diagnostics_stage(context)
    original_open_regular = diagnostic_transaction.AcquiredRoot.open_regular
    line_reads = 0

    class CountingRaw(io.RawIOBase):
        def __init__(self, handle) -> None:
            self._handle = handle

        def readable(self) -> bool:
            return True

        def fileno(self) -> int:
            return self._handle.fileno()

        def readinto(self, target) -> int:
            return self._handle.readinto(target)

        def readline(self, size: int = -1) -> bytes:
            nonlocal line_reads
            line_reads += 1
            return self._handle.readline(size)

        def close(self) -> None:
            self._handle.close()
            super().close()

    def counted_open_regular(acquired, path: Path):
        handle = original_open_regular(acquired, path)
        if Path(path).name == "stereo_coverage_frames.jsonl":
            return CountingRaw(handle)
        return handle

    monkeypatch.setattr(
        diagnostic_transaction.AcquiredRoot,
        "open_regular",
        counted_open_regular,
    )

    assert diagnostic_transaction.streaming_diagnostics_aggregates_match(context)
    assert line_reads == 0


def test_diagnostics_metadata_projection_streams_frame_names_without_read_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    names = ("frame_000000", "frame_000001", "frame_000002")
    context = diagnostic_transaction.StreamingDiagnosticsStageContext(
        job_root=tmp_path,
        frame_count=len(names),
        frame_names=lambda: iter(names),
        render_shape=(2, 3),
        render_mode="quality",
        geometry_mode="relative",
        occlusion_fill="background",
        mask_payloads_enabled=False,
        rgb_stage_fingerprint="a" * 64,
        source_guide_fingerprint="b" * 64,
        native_geometry_fingerprint="c" * 64,
        quality_input_manifest_sha256="d" * 64,
        native_geometry_sample_count=6,
    )
    document = diagnostic_transaction._streaming_diagnostics_metadata(
        context,
        status="complete",
        ordered_frame_manifest_fingerprint="e" * 64,
        frames_jsonl_sha256="f" * 64,
        summary_sha256="1" * 64,
    )
    metadata_path = tmp_path / "04_stereo_diagnostics" / "metadata.json"
    document.publish(metadata_path)
    original_read_bytes = Path.read_bytes

    def reject_metadata_read_bytes(path: Path) -> bytes:
        if path == metadata_path:
            raise AssertionError("diagnostics metadata must remain streamed")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_metadata_read_bytes)

    projection = diagnostic_transaction.project_diagnostics_metadata(metadata_path)

    assert projection.status == "complete"
    assert projection.frame_count == len(names)
    assert projection.raw_sha256 == document.raw_sha256
    assert projection.raw_byte_count == document.raw_byte_count
    assert tuple(projection.iter_frame_names()) == names


def test_diagnostics_metadata_projection_accepts_full_cleanup_authorization(
    tmp_path: Path,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    names = ("frame_000000",)
    context = diagnostic_transaction.StreamingDiagnosticsStageContext(
        job_root=tmp_path,
        frame_count=1,
        frame_names=lambda: iter(names),
        render_shape=(2, 3),
        render_mode="fast",
        geometry_mode="relative",
        occlusion_fill="background",
        mask_payloads_enabled=False,
        rgb_stage_fingerprint="a" * 64,
        source_guide_fingerprint="b" * 64,
        native_geometry_fingerprint=None,
        quality_input_manifest_sha256=None,
    )
    document = diagnostic_transaction._streaming_diagnostics_metadata(
        context,
        status="complete",
        ordered_frame_manifest_fingerprint="c" * 64,
        frames_jsonl_sha256="d" * 64,
        summary_sha256="e" * 64,
    )
    path = tmp_path / "04_stereo_diagnostics" / "metadata.json"
    document.publish(path)
    complete = diagnostic_transaction.project_diagnostics_metadata(path)
    fields = dict(complete.static_fields)
    fields.update(
        {
            "final_video_manifest_sha256": "f" * 64,
            "frame_names": list(names),
            "metric_clamp_summary_sha256": None,
            "ordered_frame_manifest_fingerprint": None,
            "prune_entries": [
                {
                    "cleanup_contract_version": 2,
                    "directory_identity": {
                        "file_identity": {
                            "file_id": f"{index:032x}",
                            "volume_serial": "1" * 16,
                        },
                        "kind": "windows",
                    },
                    "marker": {
                        "name": f".depth-surge-prune-v2-{index:032x}.marker",
                        "sha256": f"{index:064x}",
                    },
                    "operation": "delete_tree",
                    "relative_path": f"stage-{index}",
                    "stage_key": f"stage_{index}",
                }
                for index in range(16)
            ],
            "pruned_from": {
                "frames_jsonl_sha256": "d" * 64,
                "metric_clamp_summary_sha256": None,
                "ordered_frame_manifest_fingerprint": "c" * 64,
                "previous_diagnostics_fingerprint": complete.fingerprint,
                "previous_status": "complete",
                "summary_sha256": "e" * 64,
            },
            "status": "payload_pruned",
        }
    )
    fields["fingerprint"] = diagnostic_transaction.self_fingerprint(fields)
    path.write_bytes(diagnostic_transaction.canonical_json_bytes(fields))

    projection = diagnostic_transaction.project_diagnostics_metadata(path)

    assert projection.validate_state_contract() is projection
    assert len(projection.static_fields["prune_entries"]) == 16


def test_diagnostics_projection_rejects_the_first_frame_over_control_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    names = ("frame_000000", "frame_000001")
    context = diagnostic_transaction.StreamingDiagnosticsStageContext(
        job_root=tmp_path,
        frame_count=2,
        frame_names=lambda: iter(names),
        render_shape=(2, 3),
        render_mode="fast",
        geometry_mode="relative",
        occlusion_fill="background",
        mask_payloads_enabled=False,
        rgb_stage_fingerprint="a" * 64,
        source_guide_fingerprint="b" * 64,
        native_geometry_fingerprint=None,
        quality_input_manifest_sha256=None,
    )
    document = diagnostic_transaction._streaming_diagnostics_metadata(
        context,
        status="complete",
        ordered_frame_manifest_fingerprint="c" * 64,
        frames_jsonl_sha256="d" * 64,
        summary_sha256="e" * 64,
    )
    path = tmp_path / "04_stereo_diagnostics" / "metadata.json"
    document.publish(path)
    monkeypatch.setattr(diagnostic_transaction, "STEREO_CONTROL_FRAME_CAP", 1)

    with pytest.raises(diagnostic_transaction.StereoControlBudgetError, match="control cap"):
        diagnostic_transaction.project_diagnostics_metadata(path)


@pytest.mark.parametrize(
    "raw,match",
    [
        (b"[" * 9 + b"0" + b"]" * 9, "depth budget"),
        (b"[" + b",".join([b"0"] * 257) + b"]", "scalar budget"),
        (b'"' + b"x" * 4097 + b'"', "string budget"),
    ],
)
def test_exact_canonical_reader_rejects_state_growth_before_general_decode(
    raw: bytes,
    match: str,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    reader = diagnostic_transaction._ExactCanonicalReader(io.BytesIO(raw))

    with pytest.raises(diagnostic_transaction.StereoControlBudgetError, match=match):
        reader.canonical_value(maximum_bytes=64 * 1024)


def test_streaming_consolidation_never_general_decodes_histogram_pair_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    context = diagnostic_transaction.StreamingDiagnosticsStageContext(
        job_root=tmp_path,
        frame_count=1,
        frame_names=lambda: iter(("frame_0000",)),
        render_shape=(2, 3),
        render_mode="quality",
        geometry_mode="relative",
        occlusion_fill="background",
        mask_payloads_enabled=False,
        rgb_stage_fingerprint="rgb-stage",
        source_guide_fingerprint="guide-stage",
        native_geometry_fingerprint="geometry-stage",
        quality_input_manifest_sha256="d" * 64,
        native_geometry_sample_count=6,
    )
    diagnostic_transaction.begin_streaming_diagnostics_stage(context)
    diagnostic_transaction.publish_quality_frame(
        _publication(
            diagnostic_transaction,
            tmp_path,
            keep_masks=False,
            frame_name="frame_0000",
        )
    )
    general_decode = diagnostic_transaction._parse_canonical_object

    def reject_histograms(raw):
        if b'"lane_count_histogram"' in raw:
            raise AssertionError("variable histogram arrays reached the general JSON decoder")
        return general_decode(raw)

    monkeypatch.setattr(diagnostic_transaction, "_parse_canonical_object", reject_histograms)

    diagnostic_transaction.complete_streaming_diagnostics_stage(context)
    assert diagnostic_transaction.streaming_diagnostics_aggregates_match(context)


def test_streaming_completion_runs_final_source_replay_before_complete_metadata(
    tmp_path: Path,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    context = diagnostic_transaction.StreamingDiagnosticsStageContext(
        job_root=tmp_path,
        frame_count=1,
        frame_names=lambda: iter(("frame_0000",)),
        render_shape=(2, 3),
        render_mode="quality",
        geometry_mode="relative",
        occlusion_fill="background",
        mask_payloads_enabled=False,
        rgb_stage_fingerprint="rgb-stage",
        source_guide_fingerprint="guide-stage",
        native_geometry_fingerprint="geometry-stage",
        quality_input_manifest_sha256="e" * 64,
        native_geometry_sample_count=6,
    )
    diagnostic_transaction.begin_streaming_diagnostics_stage(context)
    diagnostic_transaction.publish_quality_frame(
        _publication(
            diagnostic_transaction,
            tmp_path,
            keep_masks=False,
            frame_name="frame_0000",
        )
    )
    metadata_path = tmp_path / "04_stereo_diagnostics" / "metadata.json"

    def fail_final_replay() -> None:
        assert (tmp_path / "04_stereo_diagnostics" / "stereo_coverage_summary.json").is_file()
        raise ValueError("source changed during consolidation")

    with pytest.raises(ValueError, match="source changed during consolidation"):
        diagnostic_transaction.complete_streaming_diagnostics_stage(
            context,
            before_complete_publish=fail_final_replay,
        )

    assert json.loads(metadata_path.read_text(encoding="ascii"))["status"] == "building"


@pytest.mark.parametrize(
    "phase",
    [
        "before_frames_jsonl_temporary_write",
        "after_frames_jsonl_temporary_sync",
        "after_frames_jsonl_replace",
        "after_frames_jsonl_directory_sync",
        "before_root_summary_temporary_write",
        "after_root_summary_temporary_sync",
        "after_root_summary_replace",
        "after_root_summary_directory_sync",
        "before_root_metadata_temporary_write",
        "after_root_metadata_temporary_sync",
        "after_root_metadata_replace",
        "after_root_metadata_directory_sync",
    ],
)
def test_streaming_completion_fault_boundaries_resume_to_identical_complete_root(
    tmp_path: Path,
    phase: str,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    context = _streaming_context(diagnostic_transaction, tmp_path, masks=False)
    diagnostic_transaction.begin_streaming_diagnostics_stage(context)
    diagnostic_transaction.publish_quality_frame(
        _publication(
            diagnostic_transaction,
            tmp_path,
            keep_masks=False,
            frame_name="frame_0000",
        )
    )

    def crash(current: str) -> None:
        if current == phase:
            raise OSError(f"crash at {phase}")

    with pytest.raises(OSError, match="crash at"):
        diagnostic_transaction.complete_streaming_diagnostics_stage(
            context,
            fault=crash,
        )

    metadata_path = tmp_path / "04_stereo_diagnostics" / "metadata.json"
    observed = json.loads(metadata_path.read_text(encoding="ascii"))["status"]
    expected = (
        "complete"
        if phase.startswith("after_root_metadata_replace")
        or phase == ("after_root_metadata_directory_sync")
        else "building"
    )
    assert observed == expected

    diagnostic_transaction.complete_streaming_diagnostics_stage(context)
    first = metadata_path.read_bytes()
    diagnostic_transaction.complete_streaming_diagnostics_stage(context)
    assert metadata_path.read_bytes() == first
    assert diagnostic_transaction.streaming_diagnostics_aggregates_match(context)


@pytest.mark.parametrize(
    "phase_suffix, expected_action",
    [
        ("before_mask_migration_7_temporary_write", "migrate"),
        ("after_mask_migration_7_temporary_sync", "migrate"),
        ("before_mask_migration_7_replace", "migrate"),
        ("after_mask_migration_7_replace", "reuse"),
        ("after_mask_migration_7_directory_sync", "reuse"),
    ],
)
def test_streaming_migration_source_index_faults_preserve_recoverable_manifest_prefix(
    tmp_path: Path,
    phase_suffix: str,
    expected_action: str,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    previous = _streaming_context(diagnostic_transaction, tmp_path, masks=True)
    target = _streaming_context(diagnostic_transaction, tmp_path, masks=False)
    diagnostic_transaction.begin_streaming_diagnostics_stage(previous)
    diagnostic_transaction.publish_quality_frame(
        _publication(
            diagnostic_transaction,
            tmp_path,
            keep_masks=True,
            frame_name="frame_0000",
        )
    )

    def crash(current: str) -> None:
        if current == phase_suffix:
            raise OSError(f"crash at {phase_suffix}")

    with pytest.raises(OSError, match="crash at"):
        diagnostic_transaction.migrate_streaming_frame_manifest(
            target,
            "frame_0000",
            source_index=7,
            fault=crash,
        )

    assert (
        diagnostic_transaction.classify_streaming_diagnostics_frame(
            target,
            "frame_0000",
        )
        == expected_action
    )
    frame_root = tmp_path / "04_stereo_diagnostics" / "frames" / "frame_0000"
    assert (frame_root / "left_coverage.png").is_file()
    if expected_action == "migrate":
        diagnostic_transaction.migrate_streaming_frame_manifest(
            target,
            "frame_0000",
            source_index=7,
        )
    diagnostic_transaction.prune_streaming_unreferenced_masks(target)
    assert not (frame_root / "left_coverage.png").exists()


def test_streaming_mask_prune_rejects_frame_directory_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    previous = _streaming_context(diagnostic_transaction, tmp_path, masks=True)
    target = _streaming_context(diagnostic_transaction, tmp_path, masks=False)
    diagnostic_transaction.begin_streaming_diagnostics_stage(previous)
    diagnostic_transaction.publish_quality_frame(
        _publication(
            diagnostic_transaction,
            tmp_path,
            keep_masks=True,
            frame_name="frame_0000",
        )
    )
    diagnostic_transaction.migrate_streaming_frame_manifest(target, "frame_0000")
    frame_root = tmp_path / "04_stereo_diagnostics" / "frames" / "frame_0000"
    displaced = frame_root.with_name("frame_0000-displaced")
    original_audit = diagnostic_transaction._audit_frame_transaction
    replaced = False

    def replace_after_audit(*args, **kwargs):
        nonlocal replaced
        result = original_audit(*args, **kwargs)
        if not replaced:
            replaced = True
            frame_root.rename(displaced)
            frame_root.mkdir()
            (frame_root / "must-survive.png").write_bytes(b"replacement")
        return result

    monkeypatch.setattr(
        diagnostic_transaction,
        "_audit_frame_transaction",
        replace_after_audit,
    )

    with pytest.raises(diagnostic_transaction.NoFollowPathError):
        diagnostic_transaction.prune_streaming_unreferenced_masks(target)

    assert (frame_root / "must-survive.png").read_bytes() == b"replacement"
    assert (displaced / "left_coverage.png").is_file()


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
        native_geometry_sample_count=6,
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


def test_payload_pruned_audit_does_not_require_deleted_frame_payloads(tmp_path: Path) -> None:
    import shutil

    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    content_raw = diagnostic_transaction.canonical_json_bytes({"schema_version": 1})
    final_manifest_raw = diagnostic_transaction.canonical_json_bytes({"schema_version": 4})

    context = diagnostic_transaction.DiagnosticsStageContext(
        job_root=tmp_path,
        frame_names=("frame_0000",),
        render_shape=(2, 3),
        render_mode="quality",
        geometry_mode="relative",
        occlusion_fill="background",
        mask_payloads_enabled=False,
        rgb_stage_fingerprint="rgb-stage",
        source_guide_fingerprint="guide-stage",
        native_geometry_fingerprint="geometry-stage",
        quality_input_manifest_sha256=hashlib.sha256(content_raw).hexdigest(),
        native_geometry_sample_count=6,
    )
    diagnostic_transaction.begin_diagnostics_stage(context)
    diagnostic_transaction.publish_quality_frame(
        _publication(
            diagnostic_transaction,
            tmp_path,
            keep_masks=False,
            frame_name="frame_0000",
        )
    )
    complete = diagnostic_transaction.complete_diagnostics_stage(context)
    (tmp_path / "04_stereo_diagnostics" / "quality_input_manifest.json").write_bytes(content_raw)
    (tmp_path / "final_video_manifest.json").write_bytes(final_manifest_raw)
    pruned_from = {
        "previous_status": "complete",
        "previous_diagnostics_fingerprint": complete["fingerprint"],
        "ordered_frame_manifest_fingerprint": complete["ordered_frame_manifest_fingerprint"],
        "frames_jsonl_sha256": complete["frames_jsonl_sha256"],
        "summary_sha256": complete["summary_sha256"],
        "metric_clamp_summary_sha256": None,
    }
    pruned = dict(complete)
    pruned.update(
        {
            "status": "payload_pruned",
            "ordered_frame_manifest_fingerprint": None,
            "metric_clamp_summary_sha256": None,
            "pruned_from": pruned_from,
            "final_video_manifest_sha256": hashlib.sha256(final_manifest_raw).hexdigest(),
            "prune_entries": [],
        }
    )
    pruned["fingerprint"] = diagnostic_transaction.self_fingerprint(pruned)
    (tmp_path / "04_stereo_diagnostics" / "metadata.json").write_bytes(
        diagnostic_transaction.canonical_json_bytes(pruned)
    )
    shutil.rmtree(tmp_path / "04_stereo_diagnostics" / "frames")
    (tmp_path / "04_left_frames" / "frame_0000.png").unlink()
    (tmp_path / "04_right_frames" / "frame_0000.png").unlink()

    audit = diagnostic_transaction.audit_diagnostics_stage(context)

    assert audit.disposition == "payload_pruned"
    assert audit.reusable_frame_names == ()
    assert audit.invalid_frame_names == context.frame_names
    assert audit.aggregates_valid is True
