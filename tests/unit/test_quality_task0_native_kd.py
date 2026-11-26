"""Shared tests for the prebuilt native implicit k-d implementation."""

from __future__ import annotations

import importlib.util
import inspect

import numpy as np
import pytest


def test_native_kd_wrapper_module_is_importable() -> None:
    assert importlib.util.find_spec("scripts.quality_task0.native_kd") is not None


def test_native_extension_is_prebuilt_and_reports_frozen_contract(native_kd) -> None:
    native_available = native_kd.native_available
    native_load_error = native_kd.native_load_error
    native_build_info = native_kd.native_build_info

    assert native_available(), native_load_error()
    build_info = native_build_info()
    assert build_info["algorithm"] == "quality-implicit-region-kd-hybrid-nth-element-v2"
    assert build_info["index_build_parallelism"] == "deterministic-region-std-thread-v1"
    assert 1 <= build_info["index_build_worker_cap"] <= 32
    assert build_info["geometry_query_parallelism"] == "deterministic-std-thread-v1"
    assert 1 <= build_info["geometry_query_worker_cap"] <= 32
    assert build_info["edge_band_parallelism"] == "deterministic-row-std-thread-v1"
    assert 1 <= build_info["edge_band_worker_cap"] <= 32
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
    assert build_info["sparse_seed_morphology"] == "active-fragment-pixels-v1"
    assert build_info["geodesic_workset"] == "band-pixel-cached-4ary-heap-storage-v2"
    assert build_info["kd_stack_capacity"] == 64
    assert build_info["runtime_jit"] is False
    assert len(build_info["source_sha256"]) == 64
    int(build_info["source_sha256"], 16)
    source = inspect.getsource(native_kd)
    assert "cpp_extension.load(" not in source
    assert "load_inline(" not in source


@pytest.mark.parametrize(
    ("region_ids", "region_count", "include"),
    [
        (np.ones((3, 4), dtype=np.uint32), 1, None),
        (
            np.asarray([[1, 1, 2], [1, 2, 2]], dtype=np.uint32),
            2,
            np.asarray([[True, False, True], [True, False, True]], dtype=np.bool_),
        ),
        (
            np.asarray([[1, 1, 2, 2, 3, 3, 3]], dtype=np.uint32),
            3,
            np.asarray([[False, True, False, False, True, False, True]], dtype=np.bool_),
        ),
    ],
)
def test_native_build_is_byte_exact_with_python_oracle(
    native_kd,
    kd,
    region_ids: np.ndarray,
    region_count: int,
    include: np.ndarray | None,
) -> None:
    build_native = native_kd.build_native_implicit_region_kd
    expected = kd.build_implicit_region_kd(
        region_ids,
        region_count=region_count,
        include=include,
    )

    actual = build_native(region_ids, region_count=region_count, include=include)

    assert np.array_equal(actual.member_index, expected.member_index)
    assert np.array_equal(actual.region_offsets, expected.region_offsets)
    assert actual.index_bytes == expected.index_bytes
    assert not actual.member_index.flags.writeable
    assert not actual.region_offsets.flags.writeable


def test_native_optimized_large_build_is_byte_exact_with_python_oracle(native_kd, kd) -> None:
    rows, columns = np.indices((19, 23), dtype=np.uint32)
    region_ids = ((rows + 2 * columns) % 3 + 1).astype(np.uint32)

    expected = kd.build_implicit_region_kd(region_ids, region_count=3)
    actual = native_kd.build_native_implicit_region_kd(region_ids, region_count=3)

    assert np.array_equal(actual.member_index, expected.member_index)
    assert np.array_equal(actual.region_offsets, expected.region_offsets)


def test_native_large_single_region_build_is_byte_exact_with_python_oracle(native_kd, kd) -> None:
    region_ids = np.ones((31, 37), dtype=np.uint32)

    expected = kd.build_implicit_region_kd(region_ids, region_count=1)
    actual = native_kd.build_native_implicit_region_kd(region_ids, region_count=1)

    assert np.array_equal(actual.member_index, expected.member_index)
    assert np.array_equal(actual.region_offsets, expected.region_offsets)


def test_native_build_above_packed_coordinate_limit_remains_byte_exact(native_kd, kd) -> None:
    region_ids = np.ones((1, 65_536), dtype=np.uint32)
    include = np.zeros_like(region_ids, dtype=np.bool_)
    include[0, [0, 1, 32_768, 65_534, 65_535]] = True

    expected = kd.build_implicit_region_kd(
        region_ids,
        region_count=1,
        include=include,
    )
    actual = native_kd.build_native_implicit_region_kd(
        region_ids,
        region_count=1,
        include=include,
    )

    assert np.array_equal(actual.member_index, expected.member_index)
    assert np.array_equal(actual.region_offsets, expected.region_offsets)


def test_native_build_rejects_noncanonical_inputs_before_native_mutation(native_kd) -> None:
    build_native = native_kd.build_native_implicit_region_kd

    with pytest.raises(TypeError, match="uint32"):
        build_native(np.ones((2, 2), dtype=np.int32), region_count=1)
    with pytest.raises(ValueError, match="contiguous"):
        build_native(np.asarray([[1, 3]], dtype=np.uint32), region_count=3)


def test_native_repair_batch_matches_scalar_queries_and_preserves_no_donor(native_kd, kd) -> None:
    build_native = native_kd.build_native_implicit_region_kd
    query_native = native_kd.query_native_repair_batch
    region_ids = np.asarray(
        [[1, 1, 1, 2, 2], [1, 1, 1, 2, 2], [1, 1, 1, 2, 2]],
        dtype=np.uint32,
    )
    include = np.asarray(
        [
            [True, False, True, False, False],
            [False, True, False, False, False],
            [True, False, True, False, False],
        ],
        dtype=np.bool_,
    )
    native_index = build_native(region_ids, region_count=2, include=include)
    oracle_index = kd.build_implicit_region_kd(
        region_ids,
        region_count=2,
        include=include,
    )
    query_regions = np.asarray([1, 1, 1, 2], dtype=np.uint32)
    target_y = np.asarray([0, 1, 2, 1], dtype=np.uint32)
    target_x = np.asarray([0, 1, 4, 4], dtype=np.uint32)
    native_budget = kd.VisitBudget(limit=10_000)
    oracle_budget = kd.VisitBudget(limit=10_000)

    actual = query_native(
        native_index,
        query_regions,
        target_y,
        target_x,
        radius_px=4,
        budget=native_budget,
    )
    expected = [
        oracle_index.query_repair(
            int(region),
            target_y=int(y),
            target_x=int(x),
            radius_px=4,
            budget=oracle_budget,
        )
        for region, y, x in zip(query_regions, target_y, target_x, strict=True)
    ]

    assert actual.found.tolist() == [value is not None for value in expected]
    assert actual.sample_index[:3].tolist() == [
        value.sample_index for value in expected[:3] if value is not None
    ]
    assert actual.distance2[:3].tolist() == [
        value.distance2 for value in expected[:3] if value is not None
    ]
    assert int(actual.visited_nodes.sum(dtype=np.uint64)) == native_budget.consumed
    assert not actual.sample_index.flags.writeable
    assert not actual.distance2.flags.writeable
    assert not actual.visited_nodes.flags.writeable


def test_native_repair_batch_debits_before_fatal_next_node(native_kd, kd) -> None:
    build_native = native_kd.build_native_implicit_region_kd
    query_native = native_kd.query_native_repair_batch
    index = build_native(np.ones((1, 7), dtype=np.uint32), region_count=1)
    budget = kd.VisitBudget(limit=1)

    with pytest.raises(kd.QualityRepairFallbackQueryBudgetError, match="visit cap"):
        query_native(
            index,
            np.asarray([1], dtype=np.uint32),
            np.asarray([0], dtype=np.uint32),
            np.asarray([3], dtype=np.uint32),
            radius_px=7,
            budget=budget,
        )

    assert budget.consumed == 1


def test_native_geometry_batch_matches_scalar_queries_bit_exactly(native_kd, kd) -> None:
    build_native = native_kd.build_native_implicit_region_kd
    query_native = native_kd.query_native_geometry_batch
    region_ids = np.asarray(
        [[1, 1, 1, 2], [1, 1, 2, 2], [1, 2, 2, 2]],
        dtype=np.uint32,
    )
    native_index = build_native(region_ids, region_count=2)
    oracle_index = kd.build_implicit_region_kd(region_ids, region_count=2)
    query_regions = np.asarray([1, 1, 2, 2, 2], dtype=np.uint32)
    target_y = np.asarray([-0.5, 0.5, 0.25, 1.5, 4.0], dtype=np.float64)
    target_x = np.asarray([-0.25, 1.5, 2.5, 3.5, 5.0], dtype=np.float64)
    native_budget = kd.VisitBudget(limit=10_000)
    oracle_budget = kd.VisitBudget(limit=10_000)

    actual = query_native(
        native_index,
        query_regions,
        target_y,
        target_x,
        budget=native_budget,
    )
    expected = [
        oracle_index.query_geometry(
            int(region),
            sy=float(y),
            sx=float(x),
            budget=oracle_budget,
        )
        for region, y, x in zip(query_regions, target_y, target_x, strict=True)
    ]

    assert actual.found.all()
    assert actual.sample_index.tolist() == [value.sample_index for value in expected]
    assert actual.distance2.tolist() == [value.distance2 for value in expected]
    assert int(actual.visited_nodes.sum(dtype=np.uint64)) == native_budget.consumed


def test_native_geometry_batch_uses_distinct_fatal_budget_error(native_kd, kd) -> None:
    build_native = native_kd.build_native_implicit_region_kd
    query_native = native_kd.query_native_geometry_batch
    index = build_native(np.ones((1, 7), dtype=np.uint32), region_count=1)
    budget = kd.VisitBudget(limit=0)

    with pytest.raises(kd.QualityGeometryQueryBudgetError, match="visit cap"):
        query_native(
            index,
            np.asarray([1], dtype=np.uint32),
            np.asarray([0.0], dtype=np.float64),
            np.asarray([3.0], dtype=np.float64),
            budget=budget,
        )

    assert budget.consumed == 0


def test_native_parallel_geometry_batch_preserves_ordered_budget_failure(native_kd, kd) -> None:
    build_native = native_kd.build_native_implicit_region_kd
    query_native = native_kd.query_native_geometry_batch
    index = build_native(np.ones((17, 19), dtype=np.uint32), region_count=1)
    query_count = 4096
    query_regions = np.ones(query_count, dtype=np.uint32)
    target_y = np.linspace(-2.0, 19.0, query_count, dtype=np.float64)
    target_x = np.linspace(21.0, -3.0, query_count, dtype=np.float64)
    budget = kd.VisitBudget(limit=1234)

    with pytest.raises(kd.QualityGeometryQueryBudgetError, match="visit cap"):
        query_native(
            index,
            query_regions,
            target_y,
            target_x,
            budget=budget,
        )

    assert budget.consumed == 1234
