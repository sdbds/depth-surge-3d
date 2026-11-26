"""Production region primitives must stay exact without scalar raster loops."""

from __future__ import annotations

import concurrent.futures
import importlib

import numpy as np


def test_native_low_resolution_regions_match_the_scalar_oracle(monkeypatch) -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.region_solver")
    prototype = importlib.import_module("src.depth_surge_3d.rendering.quality.region_solver")
    rng = np.random.default_rng(9281)
    displacement = np.ascontiguousarray(
        rng.integers(-12, 13, size=(31, 37)).astype(np.float64) / 3.0
    )
    validity = np.ascontiguousarray(
        rng.integers(0, 2, size=(31, 37), dtype=np.uint8).astype(np.bool_)
    )
    near_score = np.ascontiguousarray(rng.random((31, 37), dtype=np.float32))
    expected = prototype.build_low_resolution_regions(
        displacement,
        validity,
        near_score,
    )
    monkeypatch.setattr(
        production,
        "_union_connected_low_resolution_samples",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("production entered scalar low-resolution union")
        ),
    )

    actual = production.build_low_resolution_regions(
        displacement,
        validity,
        near_score,
    )

    assert np.array_equal(actual.region_map, expected.region_map)
    assert np.array_equal(actual.canonical_keys, expected.canonical_keys)
    assert np.array_equal(actual.region_rank_bits, expected.region_rank_bits)


def test_native_mapping_and_edge_band_match_the_scalar_oracle(monkeypatch) -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.region_solver")
    prototype = importlib.import_module("src.depth_surge_3d.rendering.quality.region_solver")
    labels = np.arange(1, 8 * 11 + 1, dtype=np.uint32).reshape(8, 11)
    labels[:, 3:7] = np.uint32(7)
    expected_map = prototype.nearest_label_upsample(
        labels,
        render_height=37,
        render_width=53,
    )
    expected_band = prototype.build_four_connected_edge_band(expected_map, radius=4)
    monkeypatch.setattr(
        production,
        "_nearest_source_index",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("production entered scalar nearest-label mapping")
        ),
    )
    monkeypatch.setattr(
        production,
        "_cross_dilate",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("production entered scalar edge-band dilation")
        ),
    )

    actual_map = production.nearest_label_upsample(
        labels,
        render_height=37,
        render_width=53,
    )
    actual_band = production.build_four_connected_edge_band(actual_map, radius=4)

    assert np.array_equal(actual_map, expected_map)
    assert np.array_equal(actual_band, expected_band)


def test_native_nearest_label_upsample_is_exact_under_concurrent_calls() -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.region_solver")
    prototype = importlib.import_module("src.depth_surge_3d.rendering.quality.region_solver")
    rng = np.random.default_rng(20260831)
    labels = np.ascontiguousarray(rng.integers(1, 5000, size=(37, 53), dtype=np.uint32))
    expected = prototype.nearest_label_upsample(
        labels,
        render_height=131,
        render_width=197,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        actual = list(
            executor.map(
                lambda _index: production.nearest_label_upsample(
                    labels,
                    render_height=131,
                    render_width=197,
                ),
                range(12),
            )
        )

    assert all(values.tobytes() == expected.tobytes() for values in actual)
    assert all(not values.flags.writeable for values in actual)


def test_native_region_fragment_labels_match_the_scalar_oracle() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    prototype = importlib.import_module("src.depth_surge_3d.rendering.quality.region_solver")
    regions = np.array(
        [
            [1, 1, 2, 2, 2, 3],
            [1, 2, 2, 4, 2, 3],
            [1, 1, 4, 4, 3, 3],
            [5, 5, 4, 0, 3, 3],
            [5, 6, 6, 6, 0xFFFFFFFF, 3],
        ],
        dtype=np.uint32,
    )
    band = np.array(
        [
            [1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=np.bool_,
    )
    expected = prototype._label_region_fragments(regions, band)

    actual = native_kd._require_native().label_region_fragments(regions, band)

    assert int(actual[3]) == expected[3]
    for actual_values, expected_values in zip(actual[:3], expected[:3], strict=True):
        assert np.array_equal(actual_values, expected_values)


def test_native_sparse_region_seeds_match_the_scalar_oracle() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    prototype = importlib.import_module("src.depth_surge_3d.rendering.quality.region_solver")
    rng = np.random.default_rng(20260828)
    regions = np.ascontiguousarray(rng.integers(1, 8, size=(37, 53), dtype=np.uint32))
    regions[2:8, 3:11] = np.uint32(0)
    regions[25:31, 41:49] = np.uint32(0xFFFFFFFF)
    band = np.ascontiguousarray(
        rng.integers(0, 2, size=regions.shape, dtype=np.uint8).astype(np.bool_)
    )

    for radius in (0, 1, 4):
        expected = prototype.build_sparse_region_seeds(regions, band, radius=radius)
        actual = native_kd._require_native().build_sparse_region_seeds(
            regions,
            band,
            radius,
        )

        assert actual.dtype == np.uint32
        assert actual.flags.c_contiguous
        assert actual.tobytes() == expected.tobytes()


def test_production_sparse_region_seeds_bypass_scalar_morphology(monkeypatch) -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.region_solver")
    prototype = importlib.import_module("src.depth_surge_3d.rendering.quality.region_solver")
    regions = np.ones((11, 16), dtype=np.uint32)
    regions[:, 8:] = np.uint32(2)
    band = np.zeros(regions.shape, dtype=np.bool_)
    band[:, 3:13] = True
    expected = prototype.build_sparse_region_seeds(regions, band, radius=2)

    def scalar_path_error(*_args):
        raise AssertionError("production entered scalar sparse-seed morphology")

    monkeypatch.setattr(production, "_cross_erode", scalar_path_error)
    monkeypatch.setattr(production, "cross_skeleton", scalar_path_error)
    monkeypatch.setattr(production, "_label_region_fragments", scalar_path_error)

    actual = production.build_sparse_region_seeds(regions, band, radius=2)

    assert actual.tobytes() == expected.tobytes()
    assert not actual.flags.writeable


def test_native_stable_sparse_region_seeds_match_the_composed_scalar_oracle() -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.region_solver")
    prototype = importlib.import_module("src.depth_surge_3d.rendering.quality.region_solver")
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
            prototype.build_sparse_region_seeds(regions, band, radius=radius),
            copy=True,
            order="C",
        )
        expected[~stable_initial] = np.uint32(0)
        expected[~stable_surfaces[expected]] = np.uint32(0)

        actual = production.build_sparse_stable_region_seeds(
            regions,
            band,
            stable_initial,
            stable_surfaces,
            radius=radius,
        )

        assert actual.tobytes() == expected.tobytes()
        assert not actual.flags.writeable


def test_native_one_sided_collection_matches_the_scalar_oracle() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    prototype = importlib.import_module("src.depth_surge_3d.rendering.quality.geometry")
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
        for _, row in prototype._iter_streamed_edge_rows(
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
    expected_state = prototype._RegionState(
        source_region_map=source_regions,
        final_region_map=final_regions,
        edge_band=np.zeros(render_shape, dtype=np.bool_),
        stable_sample_include=np.ones(source_regions.shape, dtype=np.bool_),
        region_count=3,
        radius=radius,
        edge_band_pixel_count=edge_count,
        final_region_hash="unused",
    )
    expected_counts = prototype._collect_one_sided_queries(
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
