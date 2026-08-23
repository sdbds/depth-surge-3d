"""Task 0 contracts for the indexed heap and integer geodesic solver."""

from __future__ import annotations

import heapq
import importlib.util

import numpy as np
import pytest

from scripts.quality_task0 import indexed_heap, region_solver


def test_region_solver_prototype_module_is_importable() -> None:
    assert importlib.util.find_spec("scripts.quality_task0.region_solver") is not None


def _api(module, name: str):
    assert hasattr(module, name), f"missing Task 0 API: {module.__name__}.{name}"
    return getattr(module, name)


def _priority(
    pixel: int,
    distance: np.ndarray,
    owner: np.ndarray,
    ranks: np.ndarray,
) -> tuple[int, int, int, int, int]:
    width = distance.shape[1]
    y, x = divmod(pixel, width)
    region = int(owner.flat[pixel])
    return (
        int(distance.flat[pixel]),
        -int(ranks[region]),
        region,
        y,
        x,
    )


def _movement_cost(left: np.ndarray, right: np.ndarray) -> int:
    maximum = max(abs(int(left[channel]) - int(right[channel])) for channel in range(3))
    return 256 + 8 * maximum


def _scalar_geodesic_oracle(  # noqa: C901 - intentionally independent oracle
    guide: np.ndarray,
    band: np.ndarray,
    seeds: np.ndarray,
    ranks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = band.shape
    infinity = np.iinfo(np.uint64).max
    distance = np.full((height, width), infinity, dtype=np.uint64)
    owner = seeds.copy()
    queue: list[tuple[int, int, int, int, int]] = []

    def relax(pixel: int, candidate_distance: int, candidate_owner: int) -> None:
        y, x = divmod(pixel, width)
        current_distance = int(distance[y, x])
        current_owner = int(owner[y, x])
        candidate_key = (candidate_distance, -int(ranks[candidate_owner]), candidate_owner)
        current_key = (
            current_distance,
            -int(ranks[current_owner]) if current_owner else 0,
            current_owner,
        )
        if current_owner == 0 or candidate_key < current_key:
            distance[y, x] = np.uint64(candidate_distance)
            owner[y, x] = np.uint32(candidate_owner)
            heapq.heappush(
                queue,
                (candidate_distance, -int(ranks[candidate_owner]), candidate_owner, y, x),
            )

    for y in range(height):
        for x in range(width):
            if not band[y, x] or seeds[y, x] == 0:
                continue
            distance[y, x] = np.uint64(0)
            region = int(seeds[y, x])
            heapq.heappush(queue, (0, -int(ranks[region]), region, y, x))

    neighbours = ((-1, 0), (0, -1), (0, 1), (1, 0))
    for y in range(height):
        for x in range(width):
            if not band[y, x] or seeds[y, x] != 0:
                continue
            for dy, dx in neighbours:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and not band[ny, nx]:
                    region = int(seeds[ny, nx])
                    if region:
                        relax(
                            y * width + x,
                            _movement_cost(guide[ny, nx], guide[y, x]),
                            region,
                        )

    settled: set[int] = set()
    while queue:
        queued_distance, queued_rank, queued_owner, y, x = heapq.heappop(queue)
        pixel = y * width + x
        if pixel in settled:
            continue
        if (queued_distance, queued_rank, queued_owner, y, x) != _priority(
            pixel, distance, owner, ranks
        ):
            continue
        settled.add(pixel)
        for dy, dx in neighbours:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < height and 0 <= nx < width and band[ny, nx]):
                continue
            relax(
                ny * width + nx,
                queued_distance + _movement_cost(guide[y, x], guide[ny, nx]),
                queued_owner,
            )
    if np.any(band & (owner == 0)):
        raise ValueError("oracle has no seed for a band component")
    return owner, distance


def test_indexed_heap_uses_full_priority_and_one_live_entry_per_pixel() -> None:
    heap_type = _api(indexed_heap, "IndexedRegionHeap")
    distance = np.full((2, 3), np.iinfo(np.uint64).max, dtype=np.uint64)
    owner = np.zeros((2, 3), dtype=np.uint32)
    ranks = np.asarray([0, 10, 20], dtype=np.uint32)
    heap = heap_type(distance, owner, ranks, capacity=6)

    distance.flat[0], owner.flat[0] = 9, 1
    heap.push_or_decrease(0)
    distance.flat[1], owner.flat[1] = 5, 1
    heap.push_or_decrease(1)
    distance.flat[2], owner.flat[2] = 5, 2
    heap.push_or_decrease(2)
    distance.flat[0] = 4
    heap.push_or_decrease(0)

    assert len(heap) == 3
    assert heap.max_live_entries == 3
    assert [heap.pop(), heap.pop(), heap.pop()] == [0, 2, 1]
    assert np.all(heap.position.flat[[0, 1, 2]] == -2)


def test_indexed_heap_fails_on_capacity_and_settled_decrease() -> None:
    heap_type = _api(indexed_heap, "IndexedRegionHeap")
    queue_error = _api(indexed_heap, "QualityRegionQueueBudgetError")
    distance = np.asarray([[1, 2]], dtype=np.uint64)
    owner = np.asarray([[1, 1]], dtype=np.uint32)
    ranks = np.asarray([0, 1], dtype=np.uint32)
    heap = heap_type(distance, owner, ranks, capacity=1)

    heap.push_or_decrease(0)
    with pytest.raises(queue_error, match="capacity"):
        heap.push_or_decrease(1)
    assert heap.pop() == 0
    with pytest.raises(queue_error, match="settled"):
        heap.push_or_decrease(0)


def test_indexed_heap_rejects_an_inconsistent_dense_position() -> None:
    heap_type = _api(indexed_heap, "IndexedRegionHeap")
    queue_error = _api(indexed_heap, "QualityRegionQueueBudgetError")
    distance = np.asarray([[1, 2]], dtype=np.uint64)
    owner = np.asarray([[1, 1]], dtype=np.uint32)
    ranks = np.asarray([0, 1], dtype=np.uint32)
    heap = heap_type(distance, owner, ranks, capacity=2)
    heap.push_or_decrease(0)
    heap.position.flat[0] = np.int32(1)

    with pytest.raises(queue_error, match="inconsistent"):
        heap.push_or_decrease(0)


def test_integer_movement_cost_uses_maximum_bgr_channel_delta() -> None:
    movement_cost = _api(region_solver, "integer_bgr_movement_cost")

    assert (
        movement_cost(
            np.asarray([10, 20, 30], dtype=np.uint8),
            np.asarray([12, 90, 35], dtype=np.uint8),
        )
        == 256 + 8 * 70
    )


@pytest.mark.parametrize("step", [3, 4, 5, 6])
def test_clean_bgr_step_moves_the_region_boundary_to_the_exact_step(step: int) -> None:
    solve = _api(region_solver, "solve_geodesic_regions")
    width = 10
    guide = np.zeros((1, width, 3), dtype=np.uint8)
    guide[:, step:] = np.uint8(255)
    band = np.ones((1, width), dtype=np.bool_)
    seeds = np.zeros((1, width), dtype=np.uint32)
    seeds[0, 0] = 1
    seeds[0, -1] = 2
    ranks = np.asarray([0, 20, 10], dtype=np.uint32)

    result = solve(guide, band, seeds, ranks)

    assert result.region_map.tolist() == [[1] * step + [2] * (width - step)]
    assert result.max_live_entries <= int(np.count_nonzero(band))
    assert np.all(result.heap_position[band] == -2)


def test_indexed_solver_matches_independent_duplicate_heap_oracle() -> None:
    solve = _api(region_solver, "solve_geodesic_regions")
    rng = np.random.default_rng(20260823)
    guide = rng.integers(0, 256, size=(5, 7, 3), dtype=np.uint8)
    band = np.zeros((5, 7), dtype=np.bool_)
    band[:, 1:6] = True
    seeds = np.zeros((5, 7), dtype=np.uint32)
    seeds[:, 0] = 1
    seeds[:, 6] = 2
    seeds[2, 3] = 3
    ranks = np.asarray([0, 10, 20, 30], dtype=np.uint32)
    expected_regions, expected_distance = _scalar_geodesic_oracle(
        guide,
        band,
        seeds,
        ranks,
    )

    actual = solve(guide, band, seeds, ranks)

    assert np.array_equal(actual.region_map, expected_regions)
    assert np.array_equal(actual.distance, expected_distance)


def test_solver_rejects_non_uint8_guide_and_seedless_band() -> None:
    solve = _api(region_solver, "solve_geodesic_regions")
    band = np.ones((1, 3), dtype=np.bool_)
    seeds = np.zeros((1, 3), dtype=np.uint32)
    ranks = np.asarray([0, 1], dtype=np.uint32)

    with pytest.raises(TypeError, match="uint8 BGR"):
        solve(np.zeros((1, 3, 3), dtype=np.float32), band, seeds, ranks)
    with pytest.raises(ValueError, match="no seed"):
        solve(np.zeros((1, 3, 3), dtype=np.uint8), band, seeds, ranks)


def test_low_resolution_regions_use_validity_jump_and_canonical_keys() -> None:
    build = _api(region_solver, "build_low_resolution_regions")
    displacement = np.asarray(
        [[0.0, 0.5, 1.5], [0.2, 0.7, 1.6]],
        dtype=np.float64,
    )
    valid = np.asarray(
        [[True, True, True], [True, False, True]],
        dtype=np.bool_,
    )
    near_score = np.asarray(
        [[0.1, 0.3, 0.2], [0.4, -0.0, 0.9]],
        dtype=np.float32,
    )

    result = build(displacement, valid, near_score)

    assert result.region_map.tolist() == [[1, 1, 2], [1, 3, 2]]
    assert result.canonical_keys.tolist() == [0, 0, 2, 4]
    expected_ranks = np.asarray([0, 0.4, 0.9, 0.0], dtype=np.float32).view(np.uint32)
    assert np.array_equal(result.region_rank_bits, expected_ranks)


def test_half_pixel_nearest_labels_and_sample_centres_are_explicit() -> None:
    upsample = _api(region_solver, "nearest_label_upsample")
    centres = _api(region_solver, "map_low_resolution_centres")
    labels = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.uint32)

    actual = upsample(labels, render_height=3, render_width=5)
    rows, columns = centres(
        geometry_height=2,
        geometry_width=3,
        render_height=3,
        render_width=5,
    )

    assert actual.tolist() == [
        [1, 1, 2, 3, 3],
        [4, 4, 5, 6, 6],
        [4, 4, 5, 6, 6],
    ]
    assert rows.tolist() == [0, 2]
    assert columns.tolist() == [0, 2, 4]


def test_boundary_band_marks_both_sides_and_uses_cross_dilation() -> None:
    make_band = _api(region_solver, "build_four_connected_edge_band")
    labels = np.asarray(
        [
            [1, 1, 1, 2, 2],
            [1, 1, 1, 2, 2],
            [1, 1, 1, 2, 2],
            [1, 1, 1, 2, 2],
            [1, 1, 1, 2, 2],
        ],
        dtype=np.uint32,
    )

    band = make_band(labels, radius=1)

    assert band.tolist() == [
        [False, True, True, True, True],
        [False, True, True, True, True],
        [False, True, True, True, True],
        [False, True, True, True, True],
        [False, True, True, True, True],
    ]


def test_boundary_band_ignores_unknown_and_no_region_sentinels() -> None:
    make_band = _api(region_solver, "build_four_connected_edge_band")
    labels = np.asarray([[1, 1, 0, 2, 2, 0xFFFFFFFF, 3]], dtype=np.uint32)

    assert not np.any(make_band(labels, radius=2))


def test_cross_skeleton_matches_repeated_opening_and_erosion() -> None:
    skeleton = _api(region_solver, "cross_skeleton")
    fragment = np.ones((3, 3), dtype=np.bool_)

    assert skeleton(fragment).tolist() == [
        [True, False, True],
        [False, True, False],
        [True, False, True],
    ]


def test_sparse_region_seeds_erode_or_fall_back_to_fragment_skeleton() -> None:
    build_seeds = _api(region_solver, "build_sparse_region_seeds")
    labels = np.ones((11, 16), dtype=np.uint32)
    labels[:, 8:] = np.uint32(2)
    band = np.zeros((11, 16), dtype=np.bool_)
    band[:, 3:13] = True

    seeds = build_seeds(labels, band, radius=2)

    assert np.all(seeds[:, :3] == 1)
    assert np.all(seeds[:, 13:] == 2)
    assert np.all(seeds[2:9, 5] == 1)
    assert np.all(seeds[2:9, 10] == 2)
    assert np.count_nonzero(seeds[:, 3:13]) == 14

    tiny_labels = np.asarray([[1, 2, 2]], dtype=np.uint32)
    tiny_band = np.ones((1, 3), dtype=np.bool_)
    tiny_seeds = build_seeds(tiny_labels, tiny_band, radius=2)
    assert tiny_seeds.tolist() == [[1, 2, 2]]


def test_each_disconnected_region_fragment_gets_its_own_fallback_seed() -> None:
    build_seeds = _api(region_solver, "build_sparse_region_seeds")
    labels = np.full((7, 10), 2, dtype=np.uint32)
    labels[:, :5] = np.uint32(1)
    labels[3, 8] = np.uint32(1)
    band = np.ones((7, 10), dtype=np.bool_)

    seeds = build_seeds(labels, band, radius=1)

    assert seeds[3, 8] == 1
    assert np.count_nonzero(seeds == 1) > 1


def test_competing_highlight_transition_and_fringe_choose_true_edge() -> None:
    solve = _api(region_solver, "solve_geodesic_regions")
    guide = np.zeros((1, 12, 3), dtype=np.uint8)
    guide[:, 5:] = np.uint8(32)
    guide[:, 6:] = np.uint8(224)
    guide[:, 7:] = np.uint8(240)
    band = np.ones((1, 12), dtype=np.bool_)
    seeds = np.zeros((1, 12), dtype=np.uint32)
    seeds[0, 0] = 1
    seeds[0, -1] = 2
    ranks = np.asarray([0, 20, 10], dtype=np.uint32)

    result = solve(guide, band, seeds, ranks)

    assert result.region_map.tolist() == [[1] * 6 + [2] * 6]


def test_flat_colour_tie_uses_rank_then_region_id() -> None:
    solve = _api(region_solver, "solve_geodesic_regions")
    guide = np.zeros((1, 9, 3), dtype=np.uint8)
    band = np.ones((1, 9), dtype=np.bool_)
    seeds = np.zeros((1, 9), dtype=np.uint32)
    seeds[0, 0] = 1
    seeds[0, -1] = 2
    ranks = np.asarray([0, 10, 20], dtype=np.uint32)

    result = solve(guide, band, seeds, ranks)

    assert result.region_map.tolist() == [[1, 1, 1, 1, 2, 2, 2, 2, 2]]
    assert result.queue_pops == 9
    assert result.queue_insertions == 9
    assert result.queue_decrease_keys == 0
    assert result.heap_pixel_bytes == 9 * 4
    assert result.heap_position_bytes == 9 * 4
    assert result.distance_bytes == 9 * 8


def test_global_solver_propagates_across_an_imaginary_row_chunk_boundary() -> None:
    solve = _api(region_solver, "solve_geodesic_regions")
    guide = np.zeros((8, 5, 3), dtype=np.uint8)
    guide[:, 3:] = np.uint8(255)
    band = np.ones((8, 5), dtype=np.bool_)
    seeds = np.zeros((8, 5), dtype=np.uint32)
    seeds[0, 0] = 1
    seeds[-1, -1] = 2
    ranks = np.asarray([0, 20, 10], dtype=np.uint32)

    result = solve(guide, band, seeds, ranks)

    assert np.all(result.region_map[:, :3] == 1)
    assert np.all(result.region_map[:, 3:] == 2)
    assert result.settled_count == 40
