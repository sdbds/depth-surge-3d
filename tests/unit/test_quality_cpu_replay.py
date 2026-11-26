"""CPU record replay, its band boundaries, and the diagnostics it promises."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from src.depth_surge_3d.rendering import stereo_geometry, stereo_renderer
from src.depth_surge_3d.rendering.quality import diagnostics, geometry, planner, renderer
from src.depth_surge_3d.rendering.quality.arena import QualityFixedArenas
from src.depth_surge_3d.rendering.quality.local_strip import QualityRepairBudgets
from src.depth_surge_3d.rendering.quality.repair_records import QualityRepairRecordCoverageError
from src.depth_surge_3d.rendering.stereo_renderer import StereoRenderer, StereoRenderSettings


def test_fast_compact_common_geometry_resolves_subpixel_shifts_for_float_images():
    shape = (2, 4)
    frame = np.arange(24, dtype=np.float32).reshape(*shape, 3) + 0.125
    total = np.full(shape, 1 / (8 * shape[1]), np.float64)
    surfaces = stereo_geometry.StereoGeometryFrame(
        np.ones(shape, np.float32), total, np.ones(shape, np.bool_)
    )
    stats = stereo_geometry.MetricProjectionStats(8, 0, 0)
    compact = StereoRenderer(device="cpu").render_geometry_fast_compact(
        frame,
        surfaces,
        stereo_renderer.StereoSplatSettings(max_eye_shift_fraction=0.1, occlusion_fill="none"),
        metric_stats=stats,
    )
    expanded = frame.repeat(16, axis=1)
    black = np.zeros_like(expanded[:, :1])
    for eye, fine in (
        ("left", np.concatenate((black, expanded[:, :-1]), axis=1)),
        ("right", np.concatenate((expanded[:, 1:], black), axis=1)),
    ):
        expected = fine.reshape(2, 4, 16, 3).mean(axis=2)
        np.testing.assert_array_equal(getattr(compact, f"{eye}_image"), expected)
        coverage = getattr(compact, f"{eye}_coverage_count")
        assert int((16 - coverage).sum()) == shape[0]
        assert getattr(compact, f"{eye}_fast_diagnostics").final_unresolved_lane_count == shape[0]
    assert compact.metric_stats == stats


def test_dense_native_and_packed_lane_analysis_preserve_bit_order_and_row_boundaries():
    valid = torch.ones((2, 3, 16), dtype=torch.bool)
    valid[0].reshape(-1)[13:19] = False
    valid[1, 0, :2] = False
    valid[1, 2, -2:] = False
    extension = stereo_renderer._native_lane_analyzer()
    expected_coverage = [[13, 13, 16], [14, 16, 14]]

    def buffers():
        return np.zeros(64, np.uint64), np.zeros(4, np.uint64), np.zeros((2, 3), np.uint8)

    for lane_api in (renderer, stereo_renderer):
        packed = (
            lane_api._pack_missing_lane_masks(valid, lane_api._lane_mask_weights(valid.device))
            .numpy()
            .view(np.uint16)
        )
        assert packed.tolist() == [[0xE000, 7, 0], [3, 0, 0xC000]]
        lane, span, coverage = buffers()
        stereo_renderer._analyze_packed_lane_validity(
            packed, lane, span, coverage, native_extension=extension, analysis_queue=None
        )
        assert coverage.tolist() == expected_coverage
        assert {k: int(lane[k]) for k in np.flatnonzero(lane)} == {2: 2, 6: 1}
        assert span.tolist() == [0, 2, 1, 0]
    other = buffers()
    stereo_renderer._analyze_lane_validity(
        valid, *other, native_extension=extension, lane_weights=None
    )
    for native, dense in zip((lane, span, coverage), other, strict=True):
        np.testing.assert_array_equal(native, dense)


@pytest.fixture
def projected_rows():
    height, width = 3, 8
    source = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    depth = np.full((height, width), 0.5, np.float32)
    surfaces = geometry.build_relative_quality_geometry(
        source, depth, render_shape=depth.shape, stereo_strength=0.0, convergence=0.5
    )
    # A stationary row, a one-and-a-quarter-pixel gap at left, and four lanes at right.
    shifts = np.array([0, 20, -4], dtype=np.int32)
    offsets = np.ascontiguousarray(np.broadcast_to(shifts[:, None], depth.shape))
    args = dict(
        renderer=StereoRenderer(device="cpu"), source=source, geometry=surfaces, offsets=offsets
    )
    arenas = QualityFixedArenas.allocate()
    projected = renderer._analyze_eye_once(**args, arenas=arenas, band_height=1)
    return args, shifts, arenas, projected


def _shift_oracle(source, shifts, fill):
    height, width = source.shape[:2]
    resolved = np.empty((height, width * 16, 3), np.uint8)
    coverage = np.zeros((height, width), np.uint8)
    for row, shift in enumerate(shifts):
        for target in range(width * 16):
            source_lane = target - int(shift)
            if 0 <= source_lane < width * 16:
                resolved[row, target] = source[row, source_lane // 16]
                coverage[row, target // 16] += 1
            else:
                resolved[row, target] = fill
    return np.rint(resolved.reshape(height, width, 16, 3).mean(axis=2)).astype(np.uint8), coverage


def test_cpu_pass_a_and_replay_preserve_every_visible_lane_across_band_sizes(projected_rows):
    args, shifts, _, projected = projected_rows
    fill = (211, 17, 89)
    expected_image, expected_coverage = _shift_oracle(args["source"], shifts, fill)
    np.testing.assert_array_equal(projected.analysis.coverage_count, expected_coverage)
    records = projected.records.copy()
    assert records["pixel_index"].tolist() == [8, 9, 23]
    assert records["lane_mask"].tolist() == [0xFFFF, 0x000F, 0xF000]
    records["fill_bgr"] = fill
    records["backend"] = [1, 2, 3]
    original = records.tobytes()
    for height in (1, 2, 3):
        actual = renderer._replay_background_eye_once(
            **args, records=records, band_height=height, planned_lane_count=24
        )
        np.testing.assert_array_equal(actual, expected_image)
        assert records.tobytes() == original

    with pytest.raises(RuntimeError, match="scatter total"):
        renderer._replay_background_eye_once(
            **args, records=records, band_height=2, planned_lane_count=23
        )
    # Fail *inside* a later band, not before rebasing. Retrying must still use global indexes.
    records["lane_mask"][-1] = 0x7000
    with pytest.raises(QualityRepairRecordCoverageError):
        renderer._replay_background_eye_once(
            **args, records=records, band_height=1, planned_lane_count=24
        )
    assert records["pixel_index"].tolist() == [8, 9, 23]
    records["lane_mask"][-1] = 0xF000
    actual = renderer._replay_background_eye_once(
        **args, records=records, band_height=1, planned_lane_count=24
    )
    np.testing.assert_array_equal(actual, expected_image)


def test_detailed_and_compact_repair_diagnostics_agree_on_lanes_pixels_and_runs(projected_rows):
    args, _, arenas, projected = projected_rows
    shape = args["source"].shape[:2]
    plan = planner.plan_quality_repairs(
        projected.analysis,
        records=projected.records,
        render_shape=shape,
        local_limit_px=1,
        budgets=QualityRepairBudgets(),
        arenas=arenas,
    )
    # A run can span two differently filled segments. Counting records as runs is wrong.
    records = plan.records.copy()
    records["backend"] = [1, 2, 3]
    plan = replace(plan, records=records)
    coverage = projected.analysis.coverage_count
    bits = np.empty_like(coverage)
    counts = np.empty(3, np.uint64)
    assert (
        renderer._background_repair_bits(
            coverage, records, out=bits, backend_lane_counts_out=counts
        )
        is bits
    )
    assert counts.tolist() == [16, 4, 4]
    summary = planner.summarize_quality_repair_plan(plan, bits)
    assert summary == planner.summarize_quality_repair_plan(
        plan, bits, backend_lane_counts=(16, 4, 4)
    )
    assert summary.backend_pixel_counts == (1, 1, 1)
    assert summary.backend_component_counts == (1, 1, 1)
    holes = renderer._plan_hole_runs(records, shape[1])
    assert np.flatnonzero(holes.lane_count_histogram).tolist() == [4, 20]
    assert holes.touched_pixel_span_histogram.tolist() == [0, 1, 1] + [0] * 6
    options = dict(quality_limits=None, render_mode="quality", fast_diagnostics=None)
    detailed = diagnostics._eye_stats(coverage, bits, plan, streamed_hole_runs=None, **options)
    compact = diagnostics._eye_stats(coverage, bits, summary, streamed_hole_runs=holes, **options)
    for field in ("lane_count_histogram", "touched_pixel_span_histogram"):
        np.testing.assert_array_equal(
            detailed["hole_runs"][field].bins, compact["hole_runs"][field].bins
        )
    for field in (
        "backend_lane_counts",
        "backend_pixel_counts",
        "backend_component_counts",
        "quality_budgets",
    ):
        assert detailed[field] == compact[field]
    assert detailed["hole_runs"]["lane_count_p95"] == pytest.approx(np.percentile([4, 20], 95))
    assert detailed["hole_runs"]["count"] == 2
    assert detailed["final_unresolved_lane_count"] == 0


def _sparse_histograms(eye):
    runs = dict(eye["hole_runs"])
    for name in ("lane_count_histogram", "touched_pixel_span_histogram"):
        bins = runs[name].bins
        runs[name] = [[int(i), int(bins[i])] for i in np.flatnonzero(bins)]
    return {**eye, "hole_runs": runs}


@pytest.mark.parametrize(
    ("mode", "fill"), [("fast", "none"), ("quality", "none"), ("quality", "background")]
)
def test_streamed_diagnostics_match_batch_reduction_on_real_eye_planes(mode, fill):
    cpu = StereoRenderer(device="cpu")
    shape = (4, 48)
    frame = np.arange(4 * 48 * 3, dtype=np.uint8).reshape(*shape, 3)
    depth = np.zeros(shape, np.float32)
    depth[:, 24:] = 1
    names = ("first.png", "second.png")
    stats = []
    for name, strength in zip(names, (0, 5), strict=True):
        settings = StereoRenderSettings(
            stereo_render_mode=mode, occlusion_fill=fill, stereo_strength=strength, convergence=0.5
        )
        if mode == "quality":
            result = renderer.render_relative_quality_compact(
                renderer=cpu, frame=frame, canonical=depth, settings=settings
            )
        else:
            result = cpu.render_relative_fast_compact(frame, depth, settings)
        stats.append(
            diagnostics.build_quality_frame_stats(
                result,
                frame_name=name,
                geometry_mode="relative",
                occlusion_fill=fill,
                render_mode=mode,
            )
        )
    # JSON consumers supply sparse pairs; freshly rendered frame stats use dense arrays.
    stats[1]["eyes"] = {eye: _sparse_histograms(values) for eye, values in stats[1]["eyes"].items()}
    summary = diagnostics.build_diagnostics_summary(
        iter(stats),
        frame_names=names,
        render_shape=shape,
        render_mode=mode,
        geometry_mode="relative",
        occlusion_fill=fill,
        ordered_frame_manifest_fingerprint="a" * 64,
        frames_jsonl_sha256="b" * 64,
    )
    assert summary["frame_names"] == list(names)
    for eye in ("left", "right"):
        frames = [item["eyes"][eye] for item in stats]
        expected = diagnostics._aggregate_eye(frames, mode)
        actual = summary["eyes"][eye]
        assert _sparse_histograms(actual) == _sparse_histograms(expected)
        assert actual["pixel_count"] == 2 * 4 * 48
        assert actual["hole_runs"]["count"] == frames[1]["hole_runs"]["count"] > 0
        missing = int(np.sum(16 - getattr(result, f"{eye}_coverage_count"), dtype=np.uint64))
        if fill == "none":
            assert actual["final_unresolved_lane_count"] == missing
        else:
            assert actual["backend_lane_counts"]["fallback"] == missing
            assert actual["final_unresolved_lane_count"] == 0
