"""Differential tests for the shipped scalar and prebuilt CPU Quality kernels."""

from dataclasses import replace
import hashlib
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

from src.depth_surge_3d.rendering.quality import (
    fallback,
    local_strip,
    native_kd,
    planner,
    region_solver,
    repair_records,
)
from src.depth_surge_3d.rendering.quality.arena import QualityByteArena
from src.depth_surge_3d.rendering.quality import exemplar, indexed_heap


@pytest.mark.parametrize("shape", [(1, 5), (5, 1), (5, 7)])
def test_reduced_exemplar_cells_match_their_actual_children_including_odd_borders(shape):
    rng = np.random.default_rng(32)
    image = rng.integers(0, 200, (*shape, 3), dtype=np.uint8)
    donor, target, processed, barrier = (
        rng.random(shape) > probability for probability in (0.2, 0.6, 0.5, 0.95)
    )
    child = exemplar.ExemplarLevel(image, donor, target, processed, barrier)
    parent = exemplar.downsample_exemplar_level(child)
    for y, x in np.ndindex(parent.donor_mask.shape):
        coords = exemplar._child_coordinates(y, x, *shape)
        blocked = any(barrier[p] for p in coords)
        pending = not blocked and any(target[p] for p in coords)
        assert parent.barrier_mask[y, x] == blocked
        assert parent.target_mask[y, x] == pending
        assert parent.donor_mask[y, x] == (not blocked and all(donor[p] for p in coords))
        assert parent.processed_mask[y, x] == (
            not blocked and not pending and all(processed[p] for p in coords)
        )
        colours = [
            exemplar._divide_ties_to_even(sum(int(image[p][c]) for p in coords), len(coords))
            for c in range(3)
        ]
        assert parent.working_bgr[y, x].tolist() == ([0, 0, 0] if blocked else colours)


def test_empty_or_already_filled_exemplar_component_consumes_no_budget_with_or_without_arena():
    shape = (2, 2)
    coverage = np.full(shape, 16, np.uint8)
    analysis = local_strip.QualityVisibilityAnalysis(
        coverage, np.ones(shape, np.uint32), tuple(coverage.copy() for _ in range(3))
    )
    records = np.zeros(1, repair_records.REPAIR_RECORD_DTYPE)
    records["backend"] = 1
    records["region_id"] = 1
    records["fill_bgr"] = [11, 22, 33]
    for arena in (None, QualityByteArena(np.empty(1024, np.uint8), "component")):
        budgets = local_strip.QualityRepairBudgets()
        result = exemplar.repair_component_exemplar(
            records,
            analysis,
            np.ones(shape, np.bool_),
            region_id=1,
            render_shape=shape,
            budgets=budgets,
            arena=arena,
        )
        assert result.records.tobytes() == records.tobytes()
        assert not result.records.flags.writeable
        assert (
            result.processed_target_count
            == result.unfinished_target_count
            == result.patch_iteration_count
            == 0
        )
        assert budgets.exemplar_eye_evaluations_consumed == 0


@pytest.mark.parametrize(
    ("target", "donor", "reason"),
    [
        ((4, 4), (3, 4), "9x9"),
        ((2, 4), (4, 4), "7x7"),
        ((3, True), (4, 4), "integer"),
    ],
)
def test_exemplar_refuses_partially_supported_target_and_donor_patches(target, donor, reason):
    shape = (10, 10)
    level = exemplar.ExemplarLevel(
        np.zeros((*shape, 3), np.uint8), *(np.zeros(shape, np.bool_) for _ in range(4))
    )
    with pytest.raises((TypeError, ValueError), match=reason):
        exemplar.score_exemplar_patch(level, target_center=target, donor_center=donor)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("capacity", True),
        ("capacity", 3),
        ("distance", np.ones((1, 2), np.int64)),
        ("distance", np.ones(2, np.uint64)),
        ("owner", np.ones((1, 3), np.uint32)),
        ("region_rank_bits", np.empty(0, np.uint32)),
    ],
)
def test_geodesic_heap_rejects_invalid_storage_before_queue_creation(field, invalid):
    args = dict(
        distance=np.ones((1, 2), np.uint64),
        owner=np.ones((1, 2), np.uint32),
        region_rank_bits=np.array([0, 1], np.uint32),
        capacity=2,
    )
    args[field] = invalid
    with pytest.raises((TypeError, ValueError), match=field):
        indexed_heap.IndexedRegionHeap(**args)


def test_rejected_heap_inserts_do_not_consume_positions_or_change_priority():
    heap = indexed_heap.IndexedRegionHeap(
        np.zeros((1, 2), np.uint64),
        np.array([[1, 0]], np.uint32),
        np.array([0, 2], np.uint32),
        capacity=2,
    )
    with pytest.raises(IndexError, match="empty"):
        heap.pop()
    for pixel, error in [(False, TypeError), (-1, ValueError), (1, ValueError)]:
        with pytest.raises(error, match="pixel"):
            heap.push_or_decrease(pixel)
    assert len(heap) == heap.insertions == heap.pop_count == 0
    assert heap.position.tolist() == [[-1, -1]]
    heap.push_or_decrease(0)
    assert heap.pop() == 0


@pytest.mark.parametrize("shape", [(1, 17), (11, 1), (11, 13)])
def test_native_regions_match_canonical_scalar_components(shape):
    rng = np.random.default_rng(99001)
    # Adjacent disparities equal to and just above the cutoff exercise opposite graph edges.
    disparity = rng.choice([0, 0.5, 1.0, np.nextafter(1.0, 2.0), 2], size=shape)
    valid = rng.random(shape) > 0.2
    near = rng.choice(np.array([-0.0, 0.125, 0.5, 1.0], dtype=np.float32), size=shape)
    scalar = region_solver._build_low_resolution_regions_scalar(disparity, valid, near)
    native = region_solver.build_low_resolution_regions(disparity, valid, near)
    np.testing.assert_array_equal(native.region_map, scalar.region_map)
    np.testing.assert_array_equal(native.canonical_keys, scalar.canonical_keys)
    np.testing.assert_array_equal(native.region_rank_bits, scalar.region_rank_bits)
    for region in np.unique(native.region_map):
        members = np.flatnonzero(native.region_map == region)
        assert native.canonical_keys[region] == members[0]
        assert native.region_rank_bits[region] == np.float32(
            max(0.0, near.flat[members].max())
        ).view(np.uint32)


@pytest.mark.parametrize("far_side", [0, 1])
@pytest.mark.parametrize("condition", ["texture", "flat", "border", "unsafe", "lost-context"])
def test_local_strip_native_matches_scalar_patch_selection(far_side, condition):
    height, width = 34, 12
    rng = np.random.default_rng(14)
    bgr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    if condition == "flat":
        bgr[:] = (12, 81, 4)
    regions = np.ones((height, width), np.uint32)
    coverage = np.full(regions.shape, 16, np.uint8)
    row = 0 if condition == "border" else height // 2
    boundary = 5 if far_side == 0 else 6
    if condition == "unsafe":
        coverage[:, boundary] = 15
    elif condition == "lost-context":
        regions[row, boundary] = 2
    analysis = local_strip.QualityVisibilityAnalysis(
        coverage, regions, tuple(np.ascontiguousarray(bgr[..., c]) for c in range(3))
    )
    records = np.zeros(2, dtype=repair_records.REPAIR_RECORD_DTYPE)
    first_col = boundary + 1 if far_side == 0 else boundary - 2
    records["pixel_index"] = [row * width + first_col, row * width + first_col + 1]
    records["far_side"] = far_side
    records["region_id"] = 1
    records["lane_mask"] = 0xFFFF
    scalar_output = records.copy()
    (run,) = repair_records.reconstruct_repair_runs(records, render_shape=regions.shape)
    counters = local_strip._LocalDiagnostics()
    selected = local_strip._plan_one_run(
        scalar_output,
        run,
        analysis,
        height=height,
        width=width,
        search_limit=2,
        safe_radius=1,
        diagnostics=counters,
    )
    native = local_strip.plan_local_strips(
        records,
        analysis,
        render_shape=regions.shape,
        local_limit_px=2,
        budgets=local_strip.QualityRepairBudgets(),
    )
    assert selected == (condition not in {"unsafe", "lost-context"})
    assert native.local_filled_run_count == int(selected)
    assert native.eligible_slot_count == counters.eligible_slots
    np.testing.assert_array_equal(scalar_output, native.records)


@pytest.fixture
def missing_lanes_and_records():
    holes = np.zeros((2, 32), dtype=np.bool_)
    holes[0, 13:19] = True
    holes[1, :3] = True
    result = repair_records.build_repair_records(
        ~holes, np.full(holes.shape, 0.5, np.float32), np.ones(holes.shape, np.uint32)
    )
    return holes, result.records.copy()


def test_native_record_validation_matches_dense_scatter_oracle(missing_lanes_and_records):
    holes, records = missing_lanes_and_records
    words, count = repair_records._accumulate_record_masks(records, 4)
    expected = np.packbits(holes.reshape(-1, 16), axis=1, bitorder="little")
    np.testing.assert_array_equal(words, expected.view("<u2").ravel())
    assert count == int(holes.sum()) == 9
    assert repair_records._compare_invalid_masks(words, holes, width=2) == count
    assert repair_records.validate_record_coverage(records, holes) == count


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lane_mask", 0),
        ("lane_mask", 5),
        ("pixel_index", 4),
        ("region_id", 0),
        ("region_id", 0xFFFFFFFF),
        ("far_side", 2),
        ("reserved", 1),
    ],
)
def test_scalar_and_native_validators_reject_invalid_record_before_scatter(
    missing_lanes_and_records, field, value
):
    holes, records = missing_lanes_and_records
    records[0][field] = value
    before = records.tobytes()
    error = repair_records.QualityRepairRecordCoverageError
    with pytest.raises(error):
        repair_records._accumulate_record_masks(records, 4)
    with pytest.raises(error):
        repair_records.validate_record_coverage(records, holes)
    assert records.tobytes() == before


def test_binary_evidence_hashes_the_actually_loaded_native_module():
    path = Path(find_spec("_quality_kd_native").origin).resolve()
    evidence = native_kd.native_binary_evidence()
    assert Path(evidence["path"]).samefile(path)
    assert evidence["byte_count"] == path.stat().st_size > 0
    with path.open("rb") as binary:
        assert evidence["sha256"] == hashlib.sha256(binary.read()).hexdigest()
    assert evidence["build_info"] == native_kd.native_build_info()


def test_component_ids_match_scalar_connectivity_without_wrapping_rows_or_mixing_regions():
    # The first group continues horizontally over a pixel boundary, then into row two.
    records = np.zeros(9, dtype=repair_records.REPAIR_RECORD_DTYPE)
    records["pixel_index"] = [0, 0, 1, 1, 2, 3, 4, 4, 5]
    records["lane_mask"] = [0x6000, 0x8000, 1, 12, 0x8000, 0x8000, 1, 12, 1]
    records["region_id"] = [1, 1, 1, 2, 1, 1, 1, 3, 1]
    shape = (2, 3)
    parent = np.arange(len(records), dtype=np.uint32)
    rank = np.zeros(len(records), np.uint8)
    pixels = records["pixel_index"]
    for pixel in range(np.prod(shape)):
        left = np.searchsorted(pixels, [pixel, pixel + 1])
        planner._union_group_pairs(records, parent, rank, *left, *left, "same")
        if pixel % shape[1] + 1 < shape[1]:
            right = np.searchsorted(pixels, [pixel + 1, pixel + 2])
            planner._union_group_pairs(records, parent, rank, *left, *right, "horizontal")
        below = pixel + shape[1]
        if below < np.prod(shape):
            vertical = np.searchsorted(pixels, [below, below + 1])
            planner._union_group_pairs(records, parent, rank, *left, *vertical, "vertical")

    scalar_roots = [planner._find(parent, i) for i in range(len(records))]
    ordered = list(dict.fromkeys(scalar_roots))
    ids, count = planner.build_record_component_ids(records, render_shape=shape)
    assert count == 5
    assert ids.tolist() == [0, 0, 0, 1, 2, 0, 0, 3, 4]
    assert ids.tolist() == [ordered.index(root) for root in scalar_roots]
    assert not ids.flags.writeable


def test_exemplar_donor_mask_is_identical_with_retained_cache_or_rebuilt_from_records():
    shape = (9, 16)
    coverage = np.full(shape, 16, np.uint8)
    regions = np.ones(shape, np.uint32)
    regions[:, 10:] = 2
    coverage[4, 5] = 0
    analysis = local_strip.QualityVisibilityAnalysis(
        coverage, regions, tuple(coverage.copy() for _ in range(3))
    )
    records = np.zeros(1, repair_records.REPAIR_RECORD_DTYPE)
    records["pixel_index"] = 4 * shape[1] + 5
    records["lane_mask"] = 0xFFFF
    records["region_id"] = 1
    precomputed = fallback.precompute_native_fallbacks(
        records,
        analysis,
        render_shape=shape,
        region_count=2,
        budgets=local_strip.QualityRepairBudgets(),
    )
    assert precomputed._safe_cache_words is not None
    expected = np.zeros(shape, np.bool_)
    for y, x in np.ndindex(shape):
        expected[y, x] = fallback.is_safe_donor(analysis, row=y, column=x, region_id=1)
    for data in (precomputed, replace(precomputed, _safe_cache_words=None)):
        mask, has_donor = fallback.materialize_safe_donor_mask(
            data,
            analysis,
            region_count=2,
            arena=QualityByteArena(np.empty(1024, np.uint8), "donor mask"),
        )
        np.testing.assert_array_equal(mask, expected)
        assert has_donor.tolist() == [False, True, False]
        assert not mask.flags.writeable and not has_donor.flags.writeable
