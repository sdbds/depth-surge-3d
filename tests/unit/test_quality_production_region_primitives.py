"""Native Quality region primitive concurrency and consistency tests."""

from __future__ import annotations

import concurrent.futures
import importlib

import numpy as np


def test_native_nearest_label_upsample_is_exact_under_concurrent_calls() -> None:
    region_solver = importlib.import_module("src.depth_surge_3d.rendering.quality.region_solver")
    rng = np.random.default_rng(20260831)
    labels = np.ascontiguousarray(rng.integers(1, 5000, size=(37, 53), dtype=np.uint32))
    expected = region_solver.nearest_label_upsample(
        labels,
        render_height=131,
        render_width=197,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        actual = list(
            executor.map(
                lambda _index: region_solver.nearest_label_upsample(
                    labels,
                    render_height=131,
                    render_width=197,
                ),
                range(12),
            )
        )

    assert all(values.tobytes() == expected.tobytes() for values in actual)
    assert all(not values.flags.writeable for values in actual)


def test_native_stable_sparse_region_seeds_match_composed_filter() -> None:
    region_solver = importlib.import_module("src.depth_surge_3d.rendering.quality.region_solver")
    rng = np.random.default_rng(2026083101)
    regions = np.ascontiguousarray(rng.integers(1, 8, size=(37, 53), dtype=np.uint32))
    band = np.ascontiguousarray(rng.random(regions.shape) < 0.37, dtype=np.bool_)
    stable_initial = np.ascontiguousarray(rng.random(regions.shape) < 0.81, dtype=np.bool_)
    stable_surfaces = np.array(
        [False, True, False, True, True, False, True, True],
        dtype=np.bool_,
    )

    for radius in (0, 1, 4):
        expected = np.array(
            region_solver.build_sparse_region_seeds(regions, band, radius=radius),
            copy=True,
            order="C",
        )
        expected[~stable_initial] = np.uint32(0)
        expected[~stable_surfaces[expected]] = np.uint32(0)

        actual = region_solver.build_sparse_stable_region_seeds(
            regions,
            band,
            stable_initial,
            stable_surfaces,
            radius=radius,
        )

        assert actual.tobytes() == expected.tobytes()
        assert not actual.flags.writeable


def test_native_one_sided_collection_matches_streamed_edge_query_collection() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    geometry = importlib.import_module("src.depth_surge_3d.rendering.quality.geometry")
    source_regions = np.array(
        [
            [1, 1, 2, 2],
            [1, 3, 3, 2],
            [1, 1, 2, 2],
        ],
        dtype=np.uint32,
    )
    render_shape = (7, 9)
    final_regions = np.ascontiguousarray(
        source_regions[
            np.array([0, 0, 1, 1, 1, 2, 2])[:, None],
            np.array([0, 0, 1, 1, 2, 2, 2, 3, 3])[None, :],
        ]
    )
    final_regions[2:5, 3:6] = np.uint32(2)
    radius = 2
    edge_count = sum(
        int(np.count_nonzero(row))
        for _, row in geometry._iter_streamed_edge_rows(
            source_regions,
            render_shape,
            radius,
        )
    )
    source_primitives = (
        np.arange(source_regions.size, dtype=np.float32).reshape(source_regions.shape),
        np.ascontiguousarray(
            np.arange(source_regions.size, dtype=np.float32).reshape(source_regions.shape)
            / np.float32(7.0)
        ),
    )

    def output_primitives():
        return tuple(
            np.full(render_shape, np.float32(-77.0), dtype=np.float32) for _ in source_primitives
        )

    def query_arrays():
        return (
            np.empty(edge_count, dtype=np.uint32),
            np.empty(edge_count, dtype=np.uint32),
            np.empty(edge_count, dtype=np.float64),
            np.empty(edge_count, dtype=np.float64),
        )

    expected_outputs = output_primitives()
    expected_queries = query_arrays()
    expected_state = geometry._RegionState(
        source_region_map=source_regions,
        final_region_map=final_regions,
        edge_band=np.zeros(render_shape, dtype=np.bool_),
        stable_sample_include=np.ones(source_regions.shape, dtype=np.bool_),
        region_count=3,
        radius=radius,
        edge_band_pixel_count=edge_count,
        final_region_hash="unused",
    )
    expected_counts = geometry._collect_one_sided_queries(
        source_primitives,
        expected_outputs,
        expected_state,
        *expected_queries,
    )

    actual_outputs = output_primitives()
    actual_queries = query_arrays()
    actual_counts = native_kd._require_native().collect_one_sided_queries(
        source_primitives,
        actual_outputs,
        source_regions,
        final_regions,
        radius,
        *actual_queries,
    )

    assert tuple(int(value) for value in actual_counts) == expected_counts
    for actual, expected in zip(actual_outputs, expected_outputs, strict=True):
        assert actual.tobytes() == expected.tobytes()
    for actual, expected in zip(actual_queries, expected_queries, strict=True):
        assert actual[: expected_counts[0]].tobytes() == expected[: expected_counts[0]].tobytes()
