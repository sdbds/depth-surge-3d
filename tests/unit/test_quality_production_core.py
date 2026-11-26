from __future__ import annotations

import concurrent.futures
import importlib

import numpy as np
import pytest


def _result_signature(result) -> tuple[object, ...]:
    return (
        result.near_score.tobytes(),
        result.total_disparity_fraction.tobytes(),
        result.source_valid.tobytes(),
        result.source_region_map.tobytes(),
        result.final_region_map.tobytes(),
        result.edge_band_pixel_count,
        result.selected_region_mismatch_count,
        result.geometry_query_count,
        result.max_neighbour_abs_q_jump_px,
        result.predicted_gap_px,
        result.final_region_hash_before_resample,
        result.final_region_hash_after_resample,
    )


def test_relative_geometry_avoids_kd_when_local_queries_are_proven() -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.geometry")
    rng = np.random.default_rng(83026)
    guide = rng.integers(0, 256, size=(18, 32, 3), dtype=np.uint8)
    canonical = rng.random((9, 16), dtype=np.float32)

    result = production.build_relative_quality_geometry(
        guide,
        canonical,
        render_shape=guide.shape[:2],
        stereo_strength=4.0,
        convergence=0.5,
    )

    assert result.geometry_query_count > 0
    assert result.geometry_index_bytes == 0


def test_geometry_adopts_mutable_geodesic_region_map_without_copy(monkeypatch) -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.geometry")
    native_region_solver = importlib.import_module(
        "src.depth_surge_3d.rendering.quality.native_region_solver"
    )
    original = native_region_solver.solve_native_geodesic_regions
    captured: dict[str, object] = {}

    def capture_region_map(*args, **kwargs):
        captured["mutable_region_map"] = kwargs.get("mutable_region_map")
        result = original(*args, **kwargs)
        captured["region_map"] = result.region_map
        return result

    monkeypatch.setattr(
        native_region_solver,
        "solve_native_geodesic_regions",
        capture_region_map,
    )
    rng = np.random.default_rng(2026083102)
    guide = rng.integers(0, 256, size=(18, 32, 3), dtype=np.uint8)
    canonical = rng.random((9, 16), dtype=np.float32)

    result = production.build_relative_quality_geometry(
        guide,
        canonical,
        render_shape=guide.shape[:2],
        stereo_strength=4.0,
        convergence=0.5,
    )

    assert captured["mutable_region_map"] is True
    assert np.shares_memory(result.final_region_map, captured["region_map"])
    assert not result.final_region_map.flags.writeable


def test_geometry_resampling_builds_filtered_kd_for_unproven_local_query() -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.geometry")
    implicit_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.implicit_kd")
    source_regions = np.ones((1, 64), dtype=np.uint32)
    source_regions[0, -1] = np.uint32(2)
    regions = production._RegionState(
        source_region_map=source_regions,
        final_region_map=np.array([[2]], dtype=np.uint32),
        edge_band=np.ones((1, 1), dtype=np.bool_),
        stable_sample_include=np.ones(source_regions.shape, dtype=np.bool_),
        region_count=2,
        radius=1,
        edge_band_pixel_count=1,
        final_region_hash="fixture",
    )
    source = np.arange(64, dtype=np.float32).reshape(1, 64)
    output = np.zeros((1, 1), dtype=np.float32)
    recorder = production._LifetimeRecorder()

    diagnostics = production._resample_primitives_in_place(
        (source,),
        (output,),
        regions,
        implicit_kd.VisitBudget(10_000),
        recorder,
    )
    recorder.release(9, "retained_zero_index")

    assert output.tolist() == [[63.0]]
    assert diagnostics.query_count == 1
    assert diagnostics.index_bytes == 16
    assert diagnostics.visited_nodes_total == 34
    assert diagnostics.visited_nodes_max == 34


def test_native_kd_build_filters_members_by_allowed_region_lookup() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    region_map = np.array([[1, 1, 2, 2]], dtype=np.uint32)
    index = native_kd.build_native_implicit_region_kd(
        region_map,
        region_count=2,
        include=np.ones(region_map.shape, dtype=np.bool_),
        allowed_regions=np.array([False, False, True], dtype=np.bool_),
    )

    assert index.member_index.tolist() == [2, 3]
    assert index.region_offsets.tolist() == [0, 0, 2]


def test_production_native_module_uses_frozen_prebuilt_algorithm() -> None:
    native = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")

    assert native.native_available()
    build_info = native.native_build_info()
    assert build_info["algorithm"] == "quality-implicit-region-kd-hybrid-nth-element-v2"
    assert build_info["index_build_parallelism"] == "deterministic-region-std-thread-v1"
    assert 1 <= build_info["index_build_worker_cap"] <= 32
    assert build_info["geometry_query_parallelism"] == "deterministic-std-thread-v1"
    assert 1 <= build_info["geometry_query_worker_cap"] <= 32
    assert build_info["foreground_matte_copy_strategy"] == "sparse-winner-records-dense-fallback-v1"
    assert build_info["foreground_matte_sparse_record_limit_divisor"] == 4
    assert build_info["foreground_matte_winner_scan"] == "fused-destination-order-v1"
    assert build_info["foreground_matte_candidate_domain"] == "edge-band-v1"
    assert build_info["background_proxy_preparation"] == "fused-lut-sliding-window-mask-bounds-v2"
    assert build_info["background_proxy_inpaint_domain"] == "exact-padded-mask-roi-v1"
    assert build_info["stable_region_selection"] == "known-cardinality-native-v1"
    assert build_info["stable_surface_support"] == "min-root-guarded-scan-fused-v2"
    assert build_info["local_geometry_query"] == "exact-bounded-grid-kd-fallback-v1"
    assert build_info["local_geometry_query_radius"] == 16
    assert build_info["local_geometry_query_parallelism"] == "deterministic-block-std-thread-v1"
    assert build_info["four_neighbour_jump_parallelism"] == "deterministic-row-std-thread-v1"
    assert 1 <= build_info["four_neighbour_jump_worker_cap"] <= 32
    assert build_info["native_spawned_worker_cap"] == 7
    assert build_info["kd_stack_capacity"] == 64
    assert build_info["runtime_jit"] is False
    assert len(build_info["source_sha256"]) == 64
    int(build_info["source_sha256"], 16)


def test_native_nearest_label_upsample_reports_precomputed_parallel_contract() -> None:
    native = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")

    build_info = native.native_build_info()

    assert (
        build_info["nearest_label_upsample_parallelism"]
        == "precomputed-axis-deterministic-row-std-thread-v1"
    )
    assert 1 <= build_info["nearest_label_upsample_worker_cap"] <= 32


def test_native_sparse_seed_filtering_reports_fused_contract() -> None:
    native = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")

    assert native.native_build_info()["sparse_seed_filtering"] == "fused-stability-v1"


def test_native_relative_band_update_matches_hand_checked_sparse_pixels() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    total = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float64)
    near_score = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    edge_band = np.array([[False, True, False], [True, False, True]], dtype=np.bool_)

    updated = native_kd.update_native_relative_total_inside_band(
        total,
        near_score,
        edge_band,
        strength=np.float64(2.0),
        convergence=np.float64(0.25),
    )

    assert updated == 3
    np.testing.assert_array_equal(
        total,
        np.array(
            [
                [10.0, float.fromhex("-0x1.0624dc28f5c29p-10"), 30.0],
                [
                    float.fromhex("0x1.89374cccccccdp-9"),
                    50.0,
                    float.fromhex("0x1.cac0851eb851fp-8"),
                ],
            ],
            dtype=np.float64,
        ),
    )


def test_native_relative_band_update_matches_row_oracle_on_random_masks() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    rng = np.random.default_rng(310830)
    near_score = rng.random((37, 43), dtype=np.float32)
    edge_band = np.ascontiguousarray(rng.random(near_score.shape) < 0.17, dtype=np.bool_)
    actual = np.ascontiguousarray(rng.normal(size=near_score.shape), dtype=np.float64)
    expected = actual.copy()
    strength = np.float64(4.3)
    convergence = np.float64(0.37)
    for row, edge_row in enumerate(edge_band):
        values = near_score[row, edge_row].astype(np.float64)
        np.subtract(values, convergence, out=values)
        np.multiply(values, strength, out=values)
        np.divide(values, np.float64(100.0), out=values)
        expected[row, edge_row] = values

    updated = native_kd.update_native_relative_total_inside_band(
        actual,
        near_score,
        edge_band,
        strength=strength,
        convergence=convergence,
    )

    assert updated == int(np.count_nonzero(edge_band))
    np.testing.assert_array_equal(actual, expected)


def test_native_four_neighbour_jump_matches_numpy_oracle_on_random_rasters() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    rng = np.random.default_rng(300830)

    for shape in ((1, 1), (1, 17), (19, 1), (31, 37), (129, 131)):
        displacement = np.ascontiguousarray(rng.normal(size=shape), dtype=np.float64)
        expected = np.float64(0.0)
        if shape[1] > 1:
            expected = max(
                expected,
                np.max(np.abs(displacement[:, 1:] - displacement[:, :-1])),
            )
        if shape[0] > 1:
            expected = max(
                expected,
                np.max(np.abs(displacement[1:, :] - displacement[:-1, :])),
            )
        actual = native_kd.max_native_four_neighbour_jump(displacement)

        assert actual.hex() == float(expected).hex()


def test_native_four_neighbour_jump_is_deterministic_across_concurrent_calls() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    rng = np.random.default_rng(320830)
    displacement = np.ascontiguousarray(rng.normal(size=(257, 263)), dtype=np.float64)
    expected = native_kd.max_native_four_neighbour_jump(displacement).hex()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        actual = list(
            executor.map(
                lambda _index: native_kd.max_native_four_neighbour_jump(displacement).hex(),
                range(12),
            )
        )

    assert actual == [expected] * 12


def test_stable_region_support_rejects_low_area_transition_slivers() -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.geometry")
    regions = np.ones((8, 10), dtype=np.uint32)
    regions[1:3, 1:3] = np.uint32(2)  # area 4, span 2: transient
    regions[4, 1:5] = np.uint32(3)  # area 4, span 4: transition sliver
    regions[5:7, 6:9] = np.uint32(4)  # area 6: stable surface

    stable = production._select_stable_region_support(
        regions,
        area_threshold=5,
    )
    included = production._stable_interior_support(regions, stable[regions])

    assert stable.tolist() == [False, True, False, False, True]
    assert not included[np.isin(regions, (2, 3))].any()
    for region in (1, 4):
        assert included[regions == np.uint32(region)].any()


@pytest.mark.parametrize("area_threshold", [1, 57, 10_000])
def test_native_stable_region_selection_matches_numpy_oracle(area_threshold: int) -> None:
    region_solver = importlib.import_module("src.depth_surge_3d.rendering.quality.region_solver")
    rng = np.random.default_rng(2026083103 + area_threshold)
    region_count = 8
    region_map = np.ascontiguousarray(
        rng.integers(1, region_count + 1, size=(37, 53), dtype=np.uint32)
    )
    counts = np.bincount(
        region_map.reshape(-1).astype(np.intp, copy=False),
        minlength=region_count + 1,
    )
    expected = np.ascontiguousarray(counts >= area_threshold, dtype=np.bool_)
    expected[0] = False
    if not bool(np.any(expected[1:])):
        expected[1:] = counts[1:] > 0

    actual = region_solver.select_stable_region_support(
        region_map,
        region_count=region_count,
        area_threshold=area_threshold,
    )

    assert actual.tobytes() == expected.tobytes()
    assert actual.dtype == np.bool_
    assert actual.flags.c_contiguous
    assert not actual.flags.writeable


def test_native_stable_region_selection_rejects_impossible_cardinality() -> None:
    region_solver = importlib.import_module("src.depth_surge_3d.rendering.quality.region_solver")

    with pytest.raises(ValueError, match="region_count cannot exceed the sample count"):
        region_solver.select_stable_region_support(
            np.ones((1, 1), dtype=np.uint32),
            region_count=2,
            area_threshold=1,
        )


def test_native_stable_surface_support_matches_python_oracle() -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.geometry")
    region_solver = importlib.import_module("src.depth_surge_3d.rendering.quality.region_solver")
    region_map = np.array(
        [
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 3, 2],
            [1, 1, 1, 3, 3, 2],
            [4, 4, 4, 3, 3, 3],
        ],
        dtype=np.uint32,
    )
    stable_regions = np.array([False, True, True, False, True], dtype=np.bool_)
    displacement = np.array(
        [
            [0.0, 0.0, 0.3, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.3, 2.0, 4.0, 2.4],
            [0.0, 0.0, 0.0, 4.0, 4.0, 2.0],
            [6.0, 6.0, 6.0, 4.0, 4.0, 4.0],
        ],
        dtype=np.float64,
    )
    near_score = np.linspace(0.0, 1.0, region_map.size, dtype=np.float32).reshape(region_map.shape)
    _, stable_samples = production._select_stable_sample_support(
        region_map,
        stable_regions,
        displacement,
    )
    expected_regions, expected_surfaces = production._build_stable_surface_regions(
        region_map,
        stable_samples,
        near_score,
    )
    expected_include = production._stable_interior_support(
        expected_regions.region_map,
        stable_samples,
    )

    actual = region_solver.build_stable_surface_support(
        region_map,
        stable_regions,
        displacement,
        near_score,
    )

    assert np.array_equal(actual.surface_regions.region_map, expected_regions.region_map)
    assert np.array_equal(actual.surface_regions.canonical_keys, expected_regions.canonical_keys)
    assert np.array_equal(
        actual.surface_regions.region_rank_bits, expected_regions.region_rank_bits
    )
    assert np.array_equal(actual.stable_samples, stable_samples)
    assert np.array_equal(actual.stable_surfaces, expected_surfaces)
    assert np.array_equal(actual.stable_sample_include, expected_include)
    for values in (
        actual.surface_regions.region_map,
        actual.surface_regions.canonical_keys,
        actual.surface_regions.region_rank_bits,
        actual.stable_samples,
        actual.stable_surfaces,
        actual.stable_sample_include,
    ):
        assert not values.flags.writeable


@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (7, 1), (8, 13), (17, 19)])
def test_native_stable_surface_support_matches_randomized_python_oracle(
    shape: tuple[int, int],
) -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.geometry")
    region_solver = importlib.import_module("src.depth_surge_3d.rendering.quality.region_solver")
    rng = np.random.default_rng(20260830 + shape[0] * 100 + shape[1])
    region_map = rng.integers(1, 8, size=shape, dtype=np.uint32)
    region_count = int(region_map.max())
    stable_regions = np.ascontiguousarray(
        rng.random(region_count + 1) >= 0.45,
        dtype=np.bool_,
    )
    stable_regions[0] = False
    stable_regions[int(region_map.reshape(-1)[0])] = True
    displacement = np.ascontiguousarray(rng.normal(0.0, 0.35, size=shape), dtype=np.float64)
    near_score = np.ascontiguousarray(rng.random(shape, dtype=np.float32))
    _, expected_samples = production._select_stable_sample_support(
        region_map,
        stable_regions,
        displacement,
    )
    expected_regions, expected_surfaces = production._build_stable_surface_regions(
        region_map,
        expected_samples,
        near_score,
    )
    expected_include = production._stable_interior_support(
        expected_regions.region_map,
        expected_samples,
    )

    actual = region_solver.build_stable_surface_support(
        region_map,
        stable_regions,
        displacement,
        near_score,
    )

    assert np.array_equal(actual.surface_regions.region_map, expected_regions.region_map)
    assert np.array_equal(actual.surface_regions.canonical_keys, expected_regions.canonical_keys)
    assert np.array_equal(
        actual.surface_regions.region_rank_bits, expected_regions.region_rank_bits
    )
    assert np.array_equal(actual.stable_samples, expected_samples)
    assert np.array_equal(actual.stable_surfaces, expected_surfaces)
    assert np.array_equal(actual.stable_sample_include, expected_include)


def test_native_stable_surface_support_preserves_all_filtered_fallback() -> None:
    region_solver = importlib.import_module("src.depth_surge_3d.rendering.quality.region_solver")
    actual = region_solver.build_stable_surface_support(
        np.ones((1, 3), dtype=np.uint32),
        np.array([False, True], dtype=np.bool_),
        np.array([[0.0, 0.3, 0.0]], dtype=np.float64),
        np.zeros((1, 3), dtype=np.float32),
    )

    assert actual.stable_samples.tolist() == [[True, True, True]]
    assert actual.stable_surfaces.tolist() == [False, True]
    assert actual.stable_sample_include.tolist() == [[True, True, True]]


def test_native_stable_surface_checkerboard_has_one_surface_per_sample() -> None:
    region_solver = importlib.import_module("src.depth_surge_3d.rendering.quality.region_solver")
    height, width = 96, 128
    count = height * width
    near_score = (np.indices((height, width)).sum(axis=0) % 2).astype(np.float32)
    displacement = near_score.astype(np.float64) * np.float64(1.6)
    low = region_solver.build_low_resolution_regions(
        displacement, np.ones((height, width), dtype=np.bool_), near_score
    )
    assert low.region_rank_bits.size == count + 1
    stable = region_solver.select_stable_region_support(
        low.region_map, region_count=count, area_threshold=2
    )
    actual = region_solver.build_stable_surface_support(
        low.region_map, stable, displacement, near_score
    )

    np.testing.assert_array_equal(actual.surface_regions.region_map, low.region_map)
    np.testing.assert_array_equal(
        actual.surface_regions.canonical_keys[1:], np.arange(count, dtype=np.uint32)
    )
    np.testing.assert_array_equal(
        actual.surface_regions.region_rank_bits[1:], near_score.reshape(-1).view(np.uint32)
    )
    assert not actual.stable_surfaces[0]
    assert actual.stable_surfaces[1:].all()
    assert actual.stable_samples.all()
    assert actual.stable_sample_include.all()


def test_native_stable_surface_support_is_concurrently_deterministic() -> None:
    region_solver = importlib.import_module("src.depth_surge_3d.rendering.quality.region_solver")
    rng = np.random.default_rng(240830)
    region_map = rng.integers(1, 24, size=(64, 96), dtype=np.uint32)
    stable_regions = np.ones(int(region_map.max()) + 1, dtype=np.bool_)
    stable_regions[0] = False
    displacement = np.ascontiguousarray(rng.normal(0.0, 0.2, size=region_map.shape))
    near_score = np.ascontiguousarray(rng.random(region_map.shape, dtype=np.float32))

    def signature() -> tuple[bytes, ...]:
        result = region_solver.build_stable_surface_support(
            region_map,
            stable_regions,
            displacement,
            near_score,
        )
        return (
            result.surface_regions.region_map.tobytes(),
            result.surface_regions.canonical_keys.tobytes(),
            result.surface_regions.region_rank_bits.tobytes(),
            result.stable_samples.tobytes(),
            result.stable_surfaces.tobytes(),
            result.stable_sample_include.tobytes(),
        )

    expected = signature()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        actual = list(executor.map(lambda _index: signature(), range(8)))

    assert actual == [expected] * 8


def test_native_local_geometry_query_proves_exact_ties_and_marks_distant_fallback() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    implicit_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.implicit_kd")
    region_map = np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 2],
        ],
        dtype=np.uint32,
    )
    include = np.ones(region_map.shape, dtype=np.bool_)
    result = native_kd.query_native_local_geometry_batch(
        region_map,
        include,
        np.array([1, 1, 2], dtype=np.uint32),
        np.array([0.5, 1.1, 0.0], dtype=np.float64),
        np.array([0.5, 1.8, 0.0], dtype=np.float64),
        radius_px=1,
        budget=implicit_kd.VisitBudget(100),
    )

    assert result.found.tolist() == [True, True, False]
    assert result.sample_index.tolist() == [0, 5, implicit_kd.UINT32_MAX]
    assert result.distance2[:2].tolist() == pytest.approx([0.5, 0.05])
    assert np.isinf(result.distance2[2])
    assert result.visited_nodes.tolist() == [4, 9, 4]


def test_native_local_geometry_query_clamps_radius_to_a_small_source_raster() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    implicit_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.implicit_kd")
    region_map = np.ones((2, 3), dtype=np.uint32)
    result = native_kd.query_native_local_geometry_batch(
        region_map,
        np.ones(region_map.shape, dtype=np.bool_),
        np.array([1], dtype=np.uint32),
        np.array([0.0], dtype=np.float64),
        np.array([1.8], dtype=np.float64),
        radius_px=16,
        budget=implicit_kd.VisitBudget(100),
    )

    assert result.found.tolist() == [True]
    assert result.sample_index.tolist() == [2]
    assert result.distance2.tolist() == pytest.approx([0.04])


def test_native_local_geometry_query_matches_kd_oracle_on_random_regions() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    implicit_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.implicit_kd")
    rng = np.random.default_rng(260830)
    shape = (17, 23)
    region_map = rng.integers(1, 7, size=shape, dtype=np.uint32)
    include = np.ascontiguousarray(rng.random(shape) > 0.2, dtype=np.bool_)
    for region in range(1, 7):
        include.reshape(-1)[np.flatnonzero(region_map.reshape(-1) == region)[0]] = True
    query_regions = rng.integers(1, 7, size=256, dtype=np.uint32)
    target_y = np.ascontiguousarray(rng.uniform(-0.5, shape[0] - 0.5, size=256))
    target_x = np.ascontiguousarray(rng.uniform(-0.5, shape[1] - 0.5, size=256))
    index = native_kd.build_native_implicit_region_kd(
        region_map,
        region_count=6,
        include=include,
    )
    expected = native_kd.query_native_geometry_batch(
        index,
        query_regions,
        target_y,
        target_x,
        budget=implicit_kd.VisitBudget(1_000_000),
    )

    actual = native_kd.query_native_local_geometry_batch(
        region_map,
        include,
        query_regions,
        target_y,
        target_x,
        radius_px=max(shape),
        budget=implicit_kd.VisitBudget(1_000_000),
    )

    assert actual.found.all()
    assert np.array_equal(actual.sample_index, expected.sample_index)
    assert np.array_equal(actual.distance2, expected.distance2)


def test_native_parallel_local_geometry_query_preserves_ordered_budget_failure() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    implicit_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.implicit_kd")
    query_count = 5_000
    region_map = np.ones((17, 19), dtype=np.uint32)
    budget = implicit_kd.VisitBudget(limit=100)

    with pytest.raises(implicit_kd.QualityGeometryQueryBudgetError, match="visit cap"):
        native_kd.query_native_local_geometry_batch(
            region_map,
            np.ones(region_map.shape, dtype=np.bool_),
            np.ones(query_count, dtype=np.uint32),
            np.zeros(query_count, dtype=np.float64),
            np.zeros(query_count, dtype=np.float64),
            radius_px=16,
            budget=budget,
        )

    assert budget.consumed == 100


def test_native_local_geometry_query_is_deterministic_across_concurrent_calls() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    implicit_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.implicit_kd")
    rng = np.random.default_rng(270830)
    region_map = rng.integers(1, 5, size=(29, 31), dtype=np.uint32)
    include = np.ascontiguousarray(rng.random(region_map.shape) > 0.15, dtype=np.bool_)
    query_regions = rng.integers(1, 5, size=4_096, dtype=np.uint32)
    target_y = np.ascontiguousarray(rng.uniform(-1.0, 30.0, size=query_regions.size))
    target_x = np.ascontiguousarray(rng.uniform(-1.0, 32.0, size=query_regions.size))

    def signature() -> tuple[bytes, bytes, bytes, int]:
        budget = implicit_kd.VisitBudget(limit=10_000_000)
        result = native_kd.query_native_local_geometry_batch(
            region_map,
            include,
            query_regions,
            target_y,
            target_x,
            radius_px=16,
            budget=budget,
        )
        return (
            result.sample_index.tobytes(),
            result.distance2.tobytes(),
            result.visited_nodes.tobytes(),
            budget.consumed,
        )

    expected = signature()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        actual = list(executor.map(lambda _index: signature(), range(8)))

    assert actual == [expected] * 8


def test_final_region_solve_routes_stable_support_through_fused_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.geometry")
    region_solver = importlib.import_module("src.depth_surge_3d.rendering.quality.region_solver")
    region_map = np.ones((1, 1), dtype=np.uint32)
    low_regions = region_solver.LowResolutionRegionResult(
        region_map=region_map,
        canonical_keys=np.array([0, 0], dtype=np.uint32),
        region_rank_bits=np.array([0, 0], dtype=np.uint32),
    )

    class FusedPathReached(Exception):
        pass

    monkeypatch.setattr(
        production,
        "_select_stable_sample_support",
        lambda *_args: (_ for _ in ()).throw(AssertionError("legacy stable-sample path used")),
    )
    monkeypatch.setattr(
        region_solver,
        "build_stable_surface_support",
        lambda *_args: (_ for _ in ()).throw(FusedPathReached),
    )

    with pytest.raises(FusedPathReached):
        production._solve_final_regions(
            np.zeros((1, 1, 3), dtype=np.uint8),
            low_regions,
            np.zeros((1, 1), dtype=np.float64),
            np.zeros((1, 1), dtype=np.float32),
            (1, 1),
            production._LifetimeRecorder(),
        )


def test_native_foreground_matte_matches_hand_checked_cross_copy() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    near_score = np.array([[0.1, 0.2, 0.8, 0.9, 0.91]], dtype=np.float32)
    total = np.array([[-0.4, -0.3, 0.3, 0.4, 0.41]], dtype=np.float64)
    source_valid = np.array([[True, False, True, True, True]], dtype=np.bool_)
    regions = np.array([[1, 1, 2, 2, 3]], dtype=np.uint32)
    displacement = np.array([[0.0, 0.0, 4.0, 4.0, 4.1]], dtype=np.float64)
    metric_valid = np.array([[True, False, True, True, True]], dtype=np.bool_)

    expanded = native_kd._require_native().expand_foreground_matte_one_pixel(
        near_score,
        total,
        source_valid,
        regions,
        displacement,
        metric_valid,
    )

    assert expanded.tolist() == [[False, True, False, False, False]]
    np.testing.assert_array_equal(
        near_score,
        np.array([[0.1, 0.8, 0.8, 0.9, 0.91]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        total,
        np.array([[-0.4, 0.3, 0.3, 0.4, 0.41]], dtype=np.float64),
    )
    assert source_valid.tolist() == [[True, True, True, True, True]]
    assert regions.tolist() == [[1, 2, 2, 2, 3]]
    assert metric_valid.tolist() == [[True, True, True, True, True]]


def test_native_foreground_matte_captures_donors_before_any_copy() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    near_score = np.array([[0.8, 0.2, 0.1]], dtype=np.float32)
    total = np.array([[8.0, 2.0, 1.0]], dtype=np.float64)
    source_valid = np.ones((1, 3), dtype=np.bool_)
    regions = np.array([[1, 2, 3]], dtype=np.uint32)
    displacement = np.array([[8.0, 4.0, 0.0]], dtype=np.float64)
    metric_valid = np.ones((1, 3), dtype=np.bool_)

    expanded = native_kd._require_native().expand_foreground_matte_one_pixel(
        near_score,
        total,
        source_valid,
        regions,
        displacement,
        metric_valid,
    )

    assert expanded.tolist() == [[False, True, True]]
    np.testing.assert_array_equal(
        near_score,
        np.array([[0.8, 0.8, 0.2]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        total,
        np.array([[8.0, 8.0, 2.0]], dtype=np.float64),
    )
    np.testing.assert_array_equal(regions, np.array([[1, 1, 2]], dtype=np.uint32))


@pytest.mark.parametrize("with_metric_valid", [False, True])
def test_native_foreground_matte_edge_band_matches_full_scan(with_metric_valid: bool) -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    native = native_kd._require_native()
    rng = np.random.default_rng(2026083108 + int(with_metric_valid))
    shape = (37, 53)
    near = rng.choice(
        np.asarray([0.0, 0.1, 0.5, 0.9, 1.0], dtype=np.float32),
        size=shape,
    )
    total = rng.normal(size=shape).astype(np.float64)
    source_valid = np.ascontiguousarray(rng.random(shape) > 0.2)
    regions = rng.integers(1, 7, size=shape, dtype=np.uint32)
    displacement = rng.choice(
        np.asarray([-1.0, -0.25, 0.0, 0.24, 0.25, 1.0, 4.0], dtype=np.float64),
        size=shape,
    )
    metric_valid = np.ascontiguousarray(rng.random(shape) > 0.25) if with_metric_valid else None
    edge_band = np.zeros(shape, dtype=np.bool_)
    vertical = regions[1:] != regions[:-1]
    edge_band[1:] |= vertical
    edge_band[:-1] |= vertical
    horizontal = regions[:, 1:] != regions[:, :-1]
    edge_band[:, 1:] |= horizontal
    edge_band[:, :-1] |= horizontal

    full = tuple(
        value.copy() if value is not None else None
        for value in (near, total, source_valid, regions, displacement, metric_valid)
    )
    masked = tuple(
        value.copy() if value is not None else None
        for value in (near, total, source_valid, regions, displacement, metric_valid)
    )
    full_expanded = native.expand_foreground_matte_one_pixel(*full)
    masked_expanded = native.expand_foreground_matte_one_pixel(*masked, edge_band)

    assert masked_expanded.tobytes() == full_expanded.tobytes()
    for actual, expected in zip(masked, full, strict=True):
        if actual is None:
            assert expected is None
        else:
            assert actual.tobytes() == expected.tobytes()


def test_foreground_matte_avoids_duplicate_numpy_content_scans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.geometry")
    near = np.asarray([[0.1, 0.9]], dtype=np.float32)
    total = np.asarray([[0.0, 1.0]], dtype=np.float64)
    source_valid = np.ones((1, 2), dtype=np.bool_)
    regions = np.asarray([[1, 2]], dtype=np.uint32)
    displacement = np.asarray([[0.0, 1.0]], dtype=np.float64)
    monkeypatch.setattr(
        production.np,
        "isfinite",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("foreground matte repeated a native finite-value scan in NumPy")
        ),
    )
    monkeypatch.setattr(
        production.np,
        "any",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("foreground matte repeated a native region-ID scan in NumPy")
        ),
    )

    expanded = production._expand_foreground_matte_one_pixel(
        near,
        total,
        source_valid,
        regions,
        displacement,
        metric_valid=None,
        candidate_mask=np.ones((1, 2), dtype=np.bool_),
    )

    assert expanded.tolist() == [[True, False]]


def test_relative_geometry_foreground_matte_uses_a_boundary_covering_edge_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.geometry")
    original = production._expand_foreground_matte_one_pixel
    observed: list[int] = []

    def capture(
        near_score,
        total_disparity_fraction,
        source_valid,
        final_region_map,
        one_eye_displacement_px,
        *,
        metric_valid,
        candidate_mask=None,
    ):
        assert candidate_mask is not None
        boundary = np.zeros(final_region_map.shape, dtype=np.bool_)
        vertical = final_region_map[1:] != final_region_map[:-1]
        boundary[1:] |= vertical
        boundary[:-1] |= vertical
        horizontal = final_region_map[:, 1:] != final_region_map[:, :-1]
        boundary[:, 1:] |= horizontal
        boundary[:, :-1] |= horizontal
        assert not np.any(boundary & ~candidate_mask)
        observed.append(int(np.count_nonzero(candidate_mask)))
        return original(
            near_score,
            total_disparity_fraction,
            source_valid,
            final_region_map,
            one_eye_displacement_px,
            metric_valid=metric_valid,
            candidate_mask=candidate_mask,
        )

    monkeypatch.setattr(production, "_expand_foreground_matte_one_pixel", capture)
    rng = np.random.default_rng(2026083109)
    guide = rng.integers(0, 256, size=(36, 64, 3), dtype=np.uint8)
    canonical = rng.random((18, 32), dtype=np.float32)

    production.build_relative_quality_geometry(
        guide,
        canonical,
        render_shape=guide.shape[:2],
        stereo_strength=4.0,
        convergence=0.5,
    )

    assert observed and observed[0] > 0


def test_native_foreground_matte_dense_fallback_preserves_results() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    row, column = np.indices((64, 64))
    high = (row + column) % 2 == 0
    near_score = np.where(high, 0.9, 0.1).astype(np.float32)
    total = np.where(high, 9.0, 1.0).astype(np.float64)
    source_valid = np.ones(high.shape, dtype=np.bool_)
    regions = np.where(high, 1, 2).astype(np.uint32)
    displacement = np.where(high, 8.0, 0.0).astype(np.float64)

    expanded = native_kd._require_native().expand_foreground_matte_one_pixel(
        near_score,
        total,
        source_valid,
        regions,
        displacement,
        None,
    )

    np.testing.assert_array_equal(expanded, ~high)
    np.testing.assert_array_equal(near_score, np.full(high.shape, 0.9, dtype=np.float32))
    np.testing.assert_array_equal(total, np.full(high.shape, 9.0, dtype=np.float64))
    np.testing.assert_array_equal(regions, np.ones(high.shape, dtype=np.uint32))


def test_native_foreground_matte_is_deterministic_across_parallel_calls() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    native = native_kd._require_native()
    row_count = 512
    near_base = np.tile(
        np.array([[0.1, 0.2, 0.8, 0.9, 0.91]], dtype=np.float32),
        (row_count, 1),
    )
    total_base = np.tile(
        np.array([[-0.4, -0.3, 0.3, 0.4, 0.41]], dtype=np.float64),
        (row_count, 1),
    )
    source_valid_base = np.tile(
        np.array([[True, False, True, True, True]], dtype=np.bool_),
        (row_count, 1),
    )
    regions_base = np.tile(
        np.array([[1, 1, 2, 2, 3]], dtype=np.uint32),
        (row_count, 1),
    )
    displacement_base = np.tile(
        np.array([[0.0, 0.0, 4.0, 4.0, 4.1]], dtype=np.float64),
        (row_count, 1),
    )
    metric_valid_base = source_valid_base.copy()
    expected = tuple(
        np.tile(row, (row_count, 1)).tobytes()
        for row in (
            np.array([[False, True, False, False, False]], dtype=np.bool_),
            np.array([[0.1, 0.8, 0.8, 0.9, 0.91]], dtype=np.float32),
            np.array([[-0.4, 0.3, 0.3, 0.4, 0.41]], dtype=np.float64),
            np.array([[True, True, True, True, True]], dtype=np.bool_),
            np.array([[1, 2, 2, 2, 3]], dtype=np.uint32),
            np.array([[True, True, True, True, True]], dtype=np.bool_),
        )
    )

    def render() -> tuple[bytes, ...]:
        near_score = near_base.copy()
        total = total_base.copy()
        source_valid = source_valid_base.copy()
        regions = regions_base.copy()
        displacement = displacement_base.copy()
        metric_valid = metric_valid_base.copy()
        expanded = native.expand_foreground_matte_one_pixel(
            near_score,
            total,
            source_valid,
            regions,
            displacement,
            metric_valid,
        )
        return tuple(
            value.tobytes()
            for value in (
                expanded,
                near_score,
                total,
                source_valid,
                regions,
                metric_valid,
            )
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda _index: render(), range(8)))

    assert actual == [expected] * 8


def test_foreground_matte_expands_only_a_large_nearer_surface_jump(monkeypatch) -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.geometry")
    near_score = np.array([[0.1, 0.2, 0.8, 0.9, 0.91]], dtype=np.float32)
    total = np.array([[-0.4, -0.3, 0.3, 0.4, 0.41]], dtype=np.float64)
    source_valid = np.ones((1, 5), dtype=np.bool_)
    regions = np.array([[1, 1, 2, 2, 3]], dtype=np.uint32)
    displacement = np.array([[0.0, 0.0, 4.0, 4.0, 4.1]], dtype=np.float64)

    with monkeypatch.context() as patch:
        patch.setattr(
            production.np,
            "copyto",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("production entered NumPy foreground-matte copying")
            ),
        )
        expanded = production._expand_foreground_matte_one_pixel(
            near_score,
            total,
            source_valid,
            regions,
            displacement,
            metric_valid=None,
        )

    assert expanded.tolist() == [[False, True, False, False, False]]
    np.testing.assert_array_equal(
        near_score,
        np.array([[0.1, 0.8, 0.8, 0.9, 0.91]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        total,
        np.array([[-0.4, 0.3, 0.3, 0.4, 0.41]], dtype=np.float64),
    )
    np.testing.assert_array_equal(regions, np.array([[1, 2, 2, 2, 3]], dtype=np.uint32))


def test_stable_surface_snaps_jagged_transition_to_persistent_depth_labels() -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.geometry")
    native_height, native_width = 48, 64
    render_shape = (96, 128)
    canonical = np.zeros((native_height, native_width), dtype=np.float32)
    canonical[:, native_width // 2 :] = np.float32(1.0)
    for row in range(native_height):
        canonical[row, native_width // 2 - 1 + row % 2] = np.float32(0.5)
    guide = np.zeros((*render_shape, 3), dtype=np.uint8)
    guide[:, render_shape[1] // 2 :] = np.uint8(255)

    result = production.build_relative_quality_geometry(
        guide,
        canonical,
        render_shape=render_shape,
        stereo_strength=5.0,
        convergence=0.5,
    )

    assert set(np.unique(result.near_score).tolist()) <= {0.0, 1.0}
    assert result.geometry_query_count == result.edge_band_pixel_count


def test_stable_surface_breaks_a_transitively_connected_depth_ramp() -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.geometry")
    region_solver = importlib.import_module("src.depth_surge_3d.rendering.quality.region_solver")
    native_height, native_width = 8, 64
    render_shape = (16, 1920)
    canonical = np.zeros((native_height, native_width), dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, 17, dtype=np.float32)
    canonical[:, 24:41] = ramp
    canonical[:, 41:] = np.float32(1.0)
    guide = np.zeros((*render_shape, 3), dtype=np.uint8)
    guide_edge = 1104
    guide[:, guide_edge:] = np.uint8(255)

    raw_regions = region_solver.build_low_resolution_regions(
        production._relative_displacement(
            canonical,
            render_shape[1],
            np.float64(1.2),
            np.float64(0.5),
        ),
        np.ones(canonical.shape, dtype=np.bool_),
        canonical,
    )
    assert set(np.unique(raw_regions.region_map).tolist()) == {1}

    result = production.build_relative_quality_geometry(
        guide,
        canonical,
        render_shape=render_shape,
        stereo_strength=1.2,
        convergence=0.5,
    )

    assert set(np.unique(result.near_score).tolist()) <= {0.0, 1.0}
    assert result.geometry_query_count > 0
    transitions = np.flatnonzero(np.diff(result.near_score[render_shape[0] // 2]))
    assert transitions.size == 1
    assert abs(int(transitions[0]) + 1 - guide_edge) <= 2
