"""Production region primitives must stay exact without scalar raster loops."""

from __future__ import annotations

import importlib

import numpy as np


def test_native_low_resolution_regions_match_the_scalar_oracle(monkeypatch) -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.region_solver")
    prototype = importlib.import_module("scripts.quality_task0.region_solver")
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
    prototype = importlib.import_module("scripts.quality_task0.region_solver")
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
