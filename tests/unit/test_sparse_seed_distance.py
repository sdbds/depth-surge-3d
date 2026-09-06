"""Sparse seed acceleration must preserve repeated cross-erosion semantics."""

from collections import deque

import numpy as np
import pytest

from src.depth_surge_3d.rendering.quality import native_kd, region_solver


def _erode(mask):
    padded = np.pad(mask, 1, constant_values=False)
    return mask & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]


def _dilate(mask):
    padded = np.pad(mask, 1, constant_values=False)
    return mask | padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:]


def _fragment(regions, band, visited, row, column):
    height, width = regions.shape
    owner = regions[row, column]
    fragment = np.zeros(regions.shape, dtype=bool)
    queue = deque([(row, column)])
    visited[row, column] = True
    while queue:
        y, x = queue.popleft()
        fragment[y, x] = True
        for ny, nx in ((y - 1, x), (y, x - 1), (y, x + 1), (y + 1, x)):
            if (
                0 <= ny < height
                and 0 <= nx < width
                and band[ny, nx]
                and regions[ny, nx] == owner
                and not visited[ny, nx]
            ):
                visited[ny, nx] = True
                queue.append((ny, nx))
    return fragment


def _erosion_reference(regions, band, radius):
    height, width = regions.shape
    seeds = np.where(band, np.uint32(0), regions)
    visited = np.zeros(regions.shape, dtype=bool)
    for row in range(height):
        for column in range(width):
            owner = regions[row, column]
            if not band[row, column] or visited[row, column] or owner in (0, 0xFFFFFFFF):
                continue
            fragment = _fragment(regions, band, visited, row, column)
            current = fragment.copy()
            for _ in range(radius):
                current = _erode(current)
                if not current.any():
                    break
            if not current.any():
                current = fragment.copy()
                selected = np.zeros_like(fragment)
                while current.any():
                    eroded = _erode(current)
                    selected |= current & ~_dilate(eroded)
                    current = eroded
            else:
                selected = current
            seeds[selected] = owner
    return seeds


@pytest.mark.parametrize("case", range(60))
def test_native_seeds_match_erosion_for_fragment_topology_and_radius(case):
    rng = np.random.default_rng(case + 20260905)
    shape = (int(rng.integers(1, 35)), int(rng.integers(1, 35)))
    coarse = rng.integers(0, 5, (7, 7), dtype=np.uint32)
    regions = np.ascontiguousarray(np.repeat(np.repeat(coarse, 5, 0), 5, 1)[: shape[0], : shape[1]])
    if case % 7 == 0:
        regions[regions == 4] = np.uint32(0xFFFFFFFF)
    band = np.ascontiguousarray(rng.random(shape) >= (0.0 if case % 3 == 0 else 0.2))
    radius = (0, 1, 3, 7, 0xFFFFFFFF)[case % 5]
    expected = _erosion_reference(regions, band, radius)
    before = regions.tobytes(), band.tobytes()
    regions.setflags(write=False)
    band.setflags(write=False)
    actual = native_kd._require_native().build_sparse_region_seeds(regions, band, radius)
    np.testing.assert_array_equal(actual, expected)
    assert (regions.tobytes(), band.tobytes()) == before


@pytest.mark.parametrize("shape", [(1, 1), (1, 25), (25, 1), (21, 25)])
@pytest.mark.parametrize("full_band", [False, True])
def test_native_seeds_preserve_outside_markers_and_image_boundaries(shape, full_band):
    regions = np.full(shape, 7, dtype=np.uint32)
    band = np.full(shape, full_band, dtype=bool)
    expected = _erosion_reference(regions, band, 32)
    actual = region_solver.build_sparse_region_seeds(regions, band, radius=32)
    np.testing.assert_array_equal(actual, expected)
    assert not actual.flags.writeable


@pytest.mark.parametrize("radius", [256, 257, 0xFFFFFFFF])
def test_seed_distance_beyond_uint8_preserves_core_or_diagonal_skeleton(radius):
    regions = np.ones((513, 513), dtype=np.uint32)
    band = np.ones(regions.shape, dtype=bool)
    expected = np.zeros(regions.shape, dtype=np.uint32)
    if radius == 256:
        expected[256, 256] = 1
    else:
        expected[:] = np.eye(513, dtype=np.uint32) | np.fliplr(np.eye(513, dtype=np.uint32))
    np.testing.assert_array_equal(
        native_kd._require_native().build_sparse_region_seeds(regions, band, radius), expected
    )


def test_stable_filtering_still_removes_unstable_sources_after_seed_selection():
    regions = np.ones((9, 17), dtype=np.uint32)
    regions[:, 8:] = 2
    band = np.ones(regions.shape, dtype=bool)
    stable_initial = np.ones_like(band)
    stable_initial[::2] = False
    surfaces = np.array([False, True, False])
    expected = _erosion_reference(regions, band, 8)
    expected[~stable_initial | ~surfaces[expected]] = 0
    actual = region_solver.build_sparse_stable_region_seeds(
        regions, band, stable_initial, surfaces, radius=8
    )
    np.testing.assert_array_equal(actual, expected)
    assert not actual.flags.writeable
