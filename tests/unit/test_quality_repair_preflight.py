"""Prove that rejected repair buffers cannot damage the caller's next eye or retry."""

from dataclasses import replace

import numpy as np
import pytest

from src.depth_surge_3d.rendering.quality import (
    arena,
    exemplar,
    fallback,
    geometry,
    local_strip,
    planner,
    repair_records,
)
from src.depth_surge_3d.rendering.quality import region_solver, native_region_solver


def test_fixed_arena_rewind_reuses_only_released_bytes_and_failed_requests_leave_cursor():
    scratch = arena.QualityByteArena(np.zeros(128, np.uint8), "left eye")
    permanent = scratch.allocate(4, np.uint32)
    permanent[:] = [17, 22, 31, 48]
    mark = scratch.mark()
    transient = scratch.allocate(4, np.uint32)
    transient.fill(0xFFFFFFFF)
    used = scratch.mark()
    with pytest.raises(arena.QualityArenaAllocationError, match="left eye"):
        scratch.allocate(1000, np.uint32)
    with pytest.raises(ValueError, match="rewind"):
        scratch.rewind(used + 1)
    with pytest.raises(ValueError, match="shape"):
        scratch.allocate((-1, 3), np.uint8)
    assert scratch.mark() == used
    scratch.rewind(mark)
    recycled = scratch.allocate(4, np.uint32, zero=True)
    assert np.shares_memory(recycled, transient)
    assert recycled.tolist() == [0, 0, 0, 0]
    assert permanent.tolist() == [17, 22, 31, 48]
    with pytest.raises(TypeError, match="contiguous"):
        arena.QualityByteArena(np.zeros(16, np.uint8)[::2], "not contiguous")


def test_releasing_one_eyes_planning_scratch_preserves_the_record_arena_for_replay():
    storage = arena.QualityFixedArenas.allocate()
    records = storage.records()
    records[0]["pixel_index"] = 291
    storage.graph.allocate(17, np.uint8)
    storage.repair.allocate(23, np.uint8)
    storage.release_planning_scratch()
    assert storage.graph is None and storage.repair is None
    assert storage.records()[0]["pixel_index"] == 291
    with pytest.raises(AssertionError, match="not both live"):
        storage.release_planning_scratch()
    storage.reset_eye()
    assert storage.graph.cursor == storage.repair.cursor == 0
    assert np.shares_memory(storage.records(), records)


@pytest.fixture
def repair_case():
    shape = (5, 7)
    coverage = np.full(shape, 16, np.uint8)
    coverage[2, 4] = 0
    analysis = local_strip.QualityVisibilityAnalysis(
        coverage, np.ones(shape, np.uint32), tuple(np.full(shape, k, np.uint8) for k in (8, 18, 28))
    )
    records = np.zeros(1, repair_records.REPAIR_RECORD_DTYPE)
    records["pixel_index"] = 18
    records["lane_mask"] = 0xFFFF
    records["region_id"] = 1
    records["fill_bgr"] = [3, 5, 7]
    return analysis, records


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("local_limit_px", True, "integer"),
        ("local_limit_px", 0, "positive"),
        ("render_shape", (False, 7), "integer"),
        ("render_shape", (0, 7), "positive"),
        ("budgets", None, "budgets"),
        ("analysis", None, "analysis"),
        ("records", np.zeros(1, np.int16), "DTYPE"),
        ("coverage_count", np.ones((1, 2), np.uint8), "coverage_count"),
        ("pure_region_id", np.zeros((5, 7), np.int32), "pure_region_id"),
        ("pure_bgr", (), "planes"),
        ("pure_bgr", "shared", "independently owned"),
        ("pure_bgr", "strided", "contiguous"),
    ],
)
def test_strip_preflight_rejects_bad_context_without_charging_or_recoloring(
    repair_case, field, value, reason
):
    analysis, records = repair_case
    budgets = local_strip.QualityRepairBudgets()
    if field == "pure_bgr" and value == "shared":
        value = (analysis.pure_bgr[0],) * 3
    elif field == "pure_bgr" and value == "strided":
        value = tuple(np.empty((5, 14), np.uint8)[:, ::2] for _ in range(3))
    args = dict(
        records=records, analysis=analysis, render_shape=(5, 7), local_limit_px=1, budgets=budgets
    )
    if field in args:
        args[field] = value
    else:
        args["analysis"] = replace(analysis, **{field: value})
    before = records.tobytes()
    with pytest.raises((ValueError, TypeError), match=reason):
        local_strip.plan_local_strips(**args)
    assert records.tobytes() == before
    assert budgets.local_slots_consumed == budgets.local_neighbor_samples_consumed == 0


@pytest.mark.parametrize(
    ("entry", "value", "reason"),
    [
        ("analysis", None, "analysis"),
        ("records", np.zeros((1,), np.int32), "record vector"),
        ("safe_donor_mask", np.ones((5, 7), np.uint8), "bool"),
        ("budgets", None, "budgets"),
        ("region_id", 0, "region"),
        ("record_indexes", np.ones(1, np.int64), "uint32"),
        ("record_indexes", np.zeros(1, np.uint32), "fixed-arena"),
        ("outside", 35, "outside"),
        ("cross-region", 2, "region_id"),
    ],
)
def test_exemplar_rejects_an_unusable_component_before_modifying_inputs(
    repair_case, entry, value, reason
):
    analysis, records = repair_case
    safe = np.ones((5, 7), np.bool_)
    args = dict(
        records=records,
        analysis=analysis,
        safe_donor_mask=safe,
        budgets=local_strip.QualityRepairBudgets(),
        region_id=1,
        render_shape=(5, 7),
    )
    if entry == "outside":
        records["pixel_index"] = value
    elif entry == "cross-region":
        records["region_id"] = value
    else:
        args[entry] = value
    before = records.tobytes()
    with pytest.raises((TypeError, ValueError), match=reason):
        exemplar.repair_component_exemplar(**args)
    assert records.tobytes() == before and safe.all()


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("valid", np.ones((1, 16), np.uint8), "bool raster"),
        ("valid", np.ones((1, 15), np.bool_), "multiple of 16"),
        ("winner_region_id", np.zeros((1, 16), np.uint32), "canonical"),
        ("winner_near_score", np.full((1, 16), np.nan, np.float32), "finite"),
        ("winner_region_id", np.ones((1, 16), np.int32), "uint32"),
        ("winner_near_score", np.ones((1, 16), np.float64), "float32"),
    ],
)
def test_builder_rejects_invalid_lane_buffers_but_accepts_sentinels_on_missing_lanes(
    field, bad, message
):
    valid = np.ones((1, 16), np.bool_)
    scores = np.ones((1, 16), np.float32)
    regions = np.ones((1, 16), np.uint32)
    args = dict(valid=valid, winner_near_score=scores, winner_region_id=regions)
    with pytest.raises((TypeError, ValueError), match=message):
        repair_records.build_repair_records(**{**args, field: bad})
    valid[0, 1] = False
    regions[0, 1] = 0
    scores[0, 1] = -np.inf
    result = repair_records.build_repair_records(**args)
    assert result.invalid_lane_count == 1 and result.records["lane_mask"].tolist() == [2]
    assert result.record_bytes == repair_records.REPAIR_RECORD_DTYPE.itemsize


def test_empty_component_graph_has_no_live_id_or_backend(repair_case):
    analysis, records = repair_case
    empty = records[:0]
    ids, count = planner.build_record_component_ids(empty, render_shape=(5, 7))
    assert count == 0 and not ids.size and not ids.flags.writeable
    plan = planner.plan_quality_repairs(
        analysis,
        records=empty,
        render_shape=(5, 7),
        local_limit_px=1,
        budgets=local_strip.QualityRepairBudgets(),
    )
    summary = planner.summarize_quality_repair_plan(plan, np.zeros((5, 7), np.uint8))
    assert summary.segment_table_bytes == summary.fallback_query_count == 0
    assert summary.backend_component_counts == (0, 0, 0)
    with pytest.raises(ValueError, match="nonnegative integers"):
        planner.summarize_quality_repair_plan(
            plan, np.zeros((5, 7), np.uint8), backend_lane_counts=(0, -1, 0)
        )


@pytest.mark.parametrize(
    ("entry", "value", "reason"),
    [
        ("render_shape", (False, 7), "integer"),
        ("render_shape", (5, -1), "positive"),
        ("records", np.zeros(1, np.uint32), "record vector"),
        ("region_count", True, "integer"),
        ("region_count", 2**32 - 1, "sentinel"),
        ("analysis", None, "analysis"),
        ("budgets", None, "budgets"),
    ],
)
def test_fallback_preflight_does_not_rewind_scratch_or_write_records_on_invalid_input(
    repair_case, entry, value, reason
):
    analysis, records = repair_case
    scratch = arena.QualityByteArena(np.full(256, 29, np.uint8), "fallback")
    scratch.allocate(3, np.uint32)
    mark = scratch.mark()
    args = dict(
        records=records,
        analysis=analysis,
        render_shape=(5, 7),
        region_count=1,
        budgets=local_strip.QualityRepairBudgets(),
        arena=scratch,
        in_place=True,
    )
    before = records.tobytes()
    with pytest.raises((ValueError, TypeError), match=reason):
        fallback.precompute_native_fallbacks(**{**args, entry: value})
    assert records.tobytes() == before and scratch.mark() == mark
    assert np.all(scratch.storage == 29)


@pytest.mark.parametrize(
    ("field", "fault"),
    [
        ("near_score", "readonly"),
        ("near_score", "dtype"),
        ("final_region_map", "readonly"),
        ("final_region_map", "shape"),
        ("source_valid", "readonly"),
        ("total_disparity_fraction", "dtype"),
        ("metric_valid", "shape"),
        ("candidate_mask", "shape"),
    ],
)
def test_matte_propagation_refuses_incompatible_parallel_rasters_before_writing_any_plane(
    field, fault
):
    shape = (2, 5)
    fields = dict(
        near_score=np.ones(shape, np.float32),
        total_disparity_fraction=np.ones(shape, np.float64),
        source_valid=np.ones(shape, np.bool_),
        final_region_map=np.ones(shape, np.uint32),
        one_eye_displacement_px=np.ones(shape, np.float64),
        metric_valid=np.ones(shape, np.bool_),
        candidate_mask=np.ones(shape, np.bool_),
    )
    if fault == "readonly":
        fields[field].setflags(write=False)
    elif fault == "dtype":
        fields[field] = fields[field].astype(np.uint16)
    else:
        fields[field] = fields[field][:1]
    originals = {name: raster.tobytes() for name, raster in fields.items()}
    with pytest.raises((ValueError, TypeError), match=field):
        geometry._expand_foreground_matte_one_pixel(**fields)
    assert {name: raster.tobytes() for name, raster in fields.items()} == originals


@pytest.mark.parametrize(
    "solve",
    [region_solver.solve_geodesic_regions, native_region_solver.solve_native_geodesic_regions],
    ids=["scalar", "native"],
)
def test_empty_geodesic_band_keeps_all_seed_labels_and_has_no_active_queue(solve):
    guide = np.zeros((2, 3, 3), np.uint8)
    mask = np.zeros((2, 3), np.bool_)
    seeds = np.array([[1, 1, 2], [1, 2, 2]], np.uint32)
    ranks = np.array([0, 10, 20], np.uint32)
    result = solve(guide, mask, seeds, ranks)
    np.testing.assert_array_equal(result.region_map, seeds)
    assert np.all(result.distance == np.iinfo(np.uint64).max)
    assert np.all(result.heap_position == -1)
    assert (
        result.settled_count
        == result.queue_insertions
        == result.max_live_entries
        == result.queue_pops
        == 0
    )
    seeds[:] = 0
    assert result.region_map.tolist() == [[1, 1, 2], [1, 2, 2]]


@pytest.mark.parametrize(
    ("field", "bad", "reason"),
    [
        ("band_mask", np.zeros((2, 3), np.uint8), "band_mask"),
        ("seed_region_map", np.zeros((3, 2), np.uint32), "seed_region_map"),
        ("seed_region_map", np.full((2, 3), 2, np.uint32), "exceeds"),
        ("region_rank_bits", np.array([0], np.uint32), "region_rank_bits"),
        ("movement_base_cost", 0, "positive"),
        ("movement_edge_scale", False, "integer"),
        ("mutable_region_map", 1, "bool"),
    ],
)
def test_native_geodesic_boundary_refuses_indices_or_weights_incompatible_with_raster(
    field, bad, reason
):
    args = dict(
        guide=np.zeros((2, 3, 3), np.uint8),
        band_mask=np.ones((2, 3), np.bool_),
        seed_region_map=np.ones((2, 3), np.uint32),
        region_rank_bits=np.array([0, 1], np.uint32),
    )
    before = args["seed_region_map"].copy()
    with pytest.raises((TypeError, ValueError), match=reason):
        native_region_solver.solve_native_geodesic_regions(**{**args, field: bad})
    np.testing.assert_array_equal(args["seed_region_map"], before)
