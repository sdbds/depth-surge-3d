"""Exactness contracts for the prebuilt native geodesic solver."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from scripts.quality_task0 import region_solver


def test_native_region_solver_module_is_importable() -> None:
    assert importlib.util.find_spec("scripts.quality_task0.native_region_solver") is not None


def _native_api(name: str):
    from scripts.quality_task0 import native_region_solver

    assert hasattr(native_region_solver, name), f"missing native region API: {name}"
    return getattr(native_region_solver, name)


def test_native_geodesic_solver_matches_python_oracle_exactly() -> None:
    solve_native = _native_api("solve_native_geodesic_regions")
    rng = np.random.default_rng(4711)
    guide = rng.integers(0, 256, size=(17, 23, 3), dtype=np.uint8)
    band = np.zeros((17, 23), dtype=np.bool_)
    band[:, 2:21] = True
    seeds = np.zeros((17, 23), dtype=np.uint32)
    seeds[:, :2] = 1
    seeds[:, 21:] = 2
    seeds[8, 11] = 3
    ranks = np.asarray([0, 10, 20, 30], dtype=np.uint32)
    expected = region_solver.solve_geodesic_regions(guide, band, seeds, ranks)

    actual = solve_native(guide, band, seeds, ranks)

    assert np.array_equal(actual.region_map, expected.region_map)
    assert np.array_equal(actual.distance, expected.distance)
    assert np.array_equal(actual.heap_position, expected.heap_position)
    assert actual.max_live_entries == expected.max_live_entries
    assert actual.settled_count == expected.settled_count
    assert actual.queue_insertions == expected.queue_insertions
    assert actual.queue_decrease_keys == expected.queue_decrease_keys
    assert actual.queue_pops == expected.queue_pops
    assert actual.heap_pixel_bytes == expected.heap_pixel_bytes
    assert actual.heap_position_bytes == expected.heap_position_bytes
    assert actual.distance_bytes == expected.distance_bytes
    assert not actual.region_map.flags.writeable
    assert not actual.distance.flags.writeable
    assert not actual.heap_position.flags.writeable


@pytest.mark.parametrize("step", [3, 4, 5, 6])
def test_native_clean_step_selects_the_exact_rgb_edge(step: int) -> None:
    solve_native = _native_api("solve_native_geodesic_regions")
    guide = np.zeros((1, 10, 3), dtype=np.uint8)
    guide[:, step:] = np.uint8(255)
    band = np.ones((1, 10), dtype=np.bool_)
    seeds = np.zeros((1, 10), dtype=np.uint32)
    seeds[0, 0] = 1
    seeds[0, -1] = 2
    ranks = np.asarray([0, 20, 10], dtype=np.uint32)

    actual = solve_native(guide, band, seeds, ranks)

    assert actual.region_map.tolist() == [[1] * step + [2] * (10 - step)]


def test_native_solver_rejects_invalid_inputs_before_native_mutation() -> None:
    solve_native = _native_api("solve_native_geodesic_regions")
    band = np.ones((1, 3), dtype=np.bool_)
    seeds = np.zeros((1, 3), dtype=np.uint32)
    ranks = np.asarray([0, 1], dtype=np.uint32)

    with pytest.raises(TypeError, match="uint8 BGR"):
        solve_native(np.zeros((1, 3, 3), dtype=np.float32), band, seeds, ranks)
    with pytest.raises(ValueError, match="no seed"):
        solve_native(np.zeros((1, 3, 3), dtype=np.uint8), band, seeds, ranks)
