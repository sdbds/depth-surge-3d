"""Queue optimizations must preserve the complete geodesic solve contract."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np
import pytest

from scripts.quality_task0 import region_solver as oracle
from src.depth_surge_3d.rendering.quality import native_region_solver


def _assert_same_solve(actual, expected) -> None:
    for name in ("region_map", "distance", "heap_position"):
        observed = getattr(actual, name)
        reference = getattr(expected, name)
        assert observed.dtype == reference.dtype
        assert observed.shape == reference.shape
        assert observed.tobytes() == reference.tobytes(), name
        assert not observed.flags.writeable
    for name in (
        "max_live_entries",
        "settled_count",
        "queue_insertions",
        "queue_decrease_keys",
        "queue_pops",
        "heap_position_bytes",
        "distance_bytes",
    ):
        assert getattr(actual, name) == getattr(expected, name), name
    # Each band pixel reserves its uint32 index and a 16-byte cached priority.
    assert actual.heap_pixel_bytes == expected.settled_count * 20


def _fixture(shape, seed, *, ties=False):
    rng = np.random.default_rng(seed)
    guide = rng.integers(0, 3, size=(*shape, 3), dtype=np.uint8) * np.uint8(127)
    if ties:
        guide.fill(0)
    band = np.ones(shape, dtype=np.bool_)
    seeds = np.zeros(shape, dtype=np.uint32)
    indices = rng.choice(seeds.size, size=min(17, seeds.size), replace=False)
    seeds.flat[indices] = rng.integers(1, 7, size=indices.size, dtype=np.uint32)
    ranks = np.array([0, 10, 10, 2**32 - 1, 10, 1, 2**32 - 1], dtype=np.uint32)
    return guide, band, seeds, ranks


@pytest.mark.parametrize("shape", [(1, 1), (1, 2), (1, 4), (1, 5), (2, 3), (3, 3), (7, 11)])
@pytest.mark.parametrize("ties", [False, True])
def test_native_queue_handles_partial_heap_levels_and_priority_ties(shape, ties) -> None:
    inputs = _fixture(shape, 20260831, ties=ties)
    expected = oracle.solve_geodesic_regions(*inputs)

    actual = native_region_solver.solve_native_geodesic_regions(*inputs)

    _assert_same_solve(actual, expected)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("ties", [False, True])
def test_native_queue_preserves_decrease_keys_and_exact_owner_order(seed, ties) -> None:
    inputs = _fixture((23, 37), seed, ties=ties)
    expected = oracle.solve_geodesic_regions(*inputs)

    actual = native_region_solver.solve_native_geodesic_regions(*inputs)

    _assert_same_solve(actual, expected)


def test_native_queue_compacts_interleaved_boundary_candidates_without_losing_pixels() -> None:
    guide, band, seeds, ranks = _fixture((17, 29), 917)
    band[:, :2] = False
    band[:, -2:] = False
    band[3:7, 11:17] = False
    seeds[:, :2] = 1
    seeds[:, -2:] = 2
    seeds[3:7, 11:17] = 3
    inputs = (guide, band, seeds, ranks)
    before = tuple(value.tobytes() for value in inputs)
    expected = oracle.solve_geodesic_regions(*inputs)
    for value in inputs:
        value.setflags(write=False)

    actual = native_region_solver.solve_native_geodesic_regions(*inputs)

    _assert_same_solve(actual, expected)
    assert tuple(value.tobytes() for value in inputs) == before


@pytest.mark.parametrize("empty", [False, True])
def test_native_queue_preserves_disconnected_and_empty_band_state(empty) -> None:
    guide = np.zeros((13, 21, 3), dtype=np.uint8)
    band = np.zeros(guide.shape[:2], dtype=np.bool_)
    seeds = np.zeros(band.shape, dtype=np.uint32)
    if not empty:
        band[1:5, 1:6] = True
        band[6:10, 8:15] = True
    seeds[2, 2] = 1
    seeds[8, 11] = 2
    ranks = np.array([0, 31, 31], dtype=np.uint32)
    inputs = (guide, band, seeds, ranks)
    expected = oracle.solve_geodesic_regions(*inputs)

    actual = native_region_solver.solve_native_geodesic_regions(*inputs)

    _assert_same_solve(actual, expected)


def test_native_queue_keeps_uint64_distances_with_maximum_movement_costs() -> None:
    guide = np.full((1, 5, 3), 255, dtype=np.uint8)
    guide[0, 0] = 0
    band = np.ones((1, 5), dtype=np.bool_)
    seeds = np.array([[1, 0, 0, 0, 0]], dtype=np.uint32)
    ranks = np.array([0, 1], dtype=np.uint32)

    actual = native_region_solver.solve_native_geodesic_regions(
        guide,
        band,
        seeds,
        ranks,
        movement_base_cost=2**32 - 1,
        movement_edge_scale=2**32 - 1,
    )

    assert actual.region_map.tolist() == [[1, 1, 1, 1, 1]]
    assert actual.distance.tolist() == [
        [0, 1099511627520, 1103806594815, 1108101562110, 1112396529405]
    ]
    assert actual.heap_position.tolist() == [[-2, -2, -2, -2, -2]]
    assert actual.queue_insertions == actual.queue_pops == 5
    assert actual.queue_decrease_keys == 0
    assert actual.heap_pixel_bytes == 100


@pytest.mark.parametrize(
    "base_cost, edge_scale, distances",
    [
        (4, 8, [0, 12, 8, 4, 0]),
        (256, 512, [0, 768, 512, 256, 0]),
        (2147483647, 4294967294, [0, 6442450941, 4294967294, 2147483647, 0]),
    ],
)
@pytest.mark.parametrize(
    "left_owner, right_owner, ranks",
    [(1, 2, [0, 1, 4294967295]), (2, 1, [0, 4294967295, 4294967295])],
)
def test_native_queue_keeps_later_owner_improvements_at_the_distance_lower_bound(
    base_cost, edge_scale, distances, left_owner, right_owner, ranks
) -> None:
    # The left seed reaches pixel 1 first; the right seed later ties its cost.
    guide = np.ones((1, 5, 3), dtype=np.uint8)
    guide[0, 0] = 0
    band = np.ones((1, 5), dtype=np.bool_)
    seeds = np.array([[left_owner, 0, 0, 0, right_owner]], dtype=np.uint32)

    actual = native_region_solver.solve_native_geodesic_regions(
        guide,
        band,
        seeds,
        np.array(ranks, dtype=np.uint32),
        movement_base_cost=base_cost,
        movement_edge_scale=edge_scale,
    )

    assert actual.distance.tolist() == [distances]
    assert actual.region_map.tolist() == [
        [left_owner, right_owner, right_owner, right_owner, right_owner]
    ]
    assert actual.queue_decrease_keys == 1
    assert actual.queue_insertions == actual.queue_pops == 5


@pytest.mark.parametrize(
    "base_cost, edge_scale, distances",
    [
        (1, 4294967295, [0, 1095216660226, 2190433320452, 3285649980678, 4380866640904]),
        (4294967295, 1, [0, 4294967550, 8589935100, 12884902650, 17179870200]),
    ],
)
def test_native_queue_preserves_distances_with_asymmetric_uint32_movement_limits(
    base_cost, edge_scale, distances
) -> None:
    guide = np.zeros((1, 5, 3), dtype=np.uint8)
    guide[:, 1::2] = 255
    band = np.ones((1, 5), dtype=np.bool_)
    seeds = np.array([[1, 0, 0, 0, 0]], dtype=np.uint32)

    actual = native_region_solver.solve_native_geodesic_regions(
        guide,
        band,
        seeds,
        np.array([0, 1], dtype=np.uint32),
        movement_base_cost=base_cost,
        movement_edge_scale=edge_scale,
    )

    assert actual.distance.tolist() == [distances]
    assert actual.region_map.tolist() == [[1, 1, 1, 1, 1]]
    assert actual.queue_decrease_keys == 0
    assert actual.queue_insertions == actual.queue_pops == 5


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("DEPTH_SURGE_RUN_LARGE_NATIVE_TESTS") != "1",
    reason="set DEPTH_SURGE_RUN_LARGE_NATIVE_TESTS=1 for the 704 MiB overflow fixture",
)
def test_native_queue_does_not_prune_overflow_on_the_final_settled_back_edge() -> None:
    # With maximum costs, 2**24 steps fit uint64, but one more edge does not.
    # This raw-solver fixture needs about 704 MiB, outside the public render budget.
    width = 16_777_217
    guide = np.zeros((1, width, 3), dtype=np.uint8)
    guide[:, 1::2] = 255
    band = np.ones((1, width), dtype=np.bool_)
    seeds = np.zeros((1, width), dtype=np.uint32)
    seeds[0, 0] = 1

    with pytest.raises(
        native_region_solver.QualityRegionQueueBudgetError, match="distance overflow"
    ):
        native_region_solver.solve_native_geodesic_regions(
            guide,
            band,
            seeds,
            np.array([0, 1], dtype=np.uint32),
            movement_base_cost=2**32 - 1,
            movement_edge_scale=2**32 - 1,
        )


def test_native_queue_rejects_an_unseeded_component_after_other_components_finish() -> None:
    guide = np.zeros((1, 7, 3), dtype=np.uint8)
    band = np.array([[True, True, False, False, False, True, True]], dtype=np.bool_)
    seeds = np.array([[1, 0, 0, 0, 0, 0, 0]], dtype=np.uint32)
    ranks = np.array([0, 1], dtype=np.uint32)

    with pytest.raises(ValueError, match="no seed"):
        native_region_solver.solve_native_geodesic_regions(guide, band, seeds, ranks)


def test_native_queue_workspaces_remain_independent_under_concurrent_solves() -> None:
    fixtures = [_fixture((17, 23), seed) for seed in range(6)]
    expected = [oracle.solve_geodesic_regions(*inputs) for inputs in fixtures]

    def solve(inputs):
        return native_region_solver.solve_native_geodesic_regions(*inputs)

    with ThreadPoolExecutor(max_workers=3) as executor:
        actual = list(executor.map(solve, fixtures))

    for observed, reference in zip(actual, expected, strict=True):
        _assert_same_solve(observed, reference)
