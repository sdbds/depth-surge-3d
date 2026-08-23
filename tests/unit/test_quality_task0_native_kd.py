"""Task 0 tests for the prebuilt native implicit k-d implementation."""

from __future__ import annotations

import importlib.util
import inspect

import numpy as np
import pytest

from scripts.quality_task0 import implicit_kd, native_kd


def test_native_kd_wrapper_module_is_importable() -> None:
    assert importlib.util.find_spec("scripts.quality_task0.native_kd") is not None


def _api(name: str):
    assert hasattr(native_kd, name), f"missing native Task 0 API: {name}"
    return getattr(native_kd, name)


def test_native_extension_is_prebuilt_and_reports_frozen_contract() -> None:
    native_available = _api("native_available")
    native_load_error = _api("native_load_error")
    native_build_info = _api("native_build_info")

    assert native_available(), native_load_error()
    assert native_build_info() == {
        "algorithm": "quality-implicit-region-kd-median-of-medians-v1",
        "kd_stack_capacity": 64,
        "runtime_jit": False,
    }
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
    region_ids: np.ndarray,
    region_count: int,
    include: np.ndarray | None,
) -> None:
    build_native = _api("build_native_implicit_region_kd")
    expected = implicit_kd.build_implicit_region_kd(
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


def test_native_build_rejects_noncanonical_inputs_before_native_mutation() -> None:
    build_native = _api("build_native_implicit_region_kd")

    with pytest.raises(TypeError, match="uint32"):
        build_native(np.ones((2, 2), dtype=np.int32), region_count=1)
    with pytest.raises(ValueError, match="contiguous"):
        build_native(np.asarray([[1, 3]], dtype=np.uint32), region_count=3)


def test_native_repair_batch_matches_scalar_queries_and_preserves_no_donor() -> None:
    build_native = _api("build_native_implicit_region_kd")
    query_native = _api("query_native_repair_batch")
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
    oracle_index = implicit_kd.build_implicit_region_kd(
        region_ids,
        region_count=2,
        include=include,
    )
    query_regions = np.asarray([1, 1, 1, 2], dtype=np.uint32)
    target_y = np.asarray([0, 1, 2, 1], dtype=np.uint32)
    target_x = np.asarray([0, 1, 4, 4], dtype=np.uint32)
    native_budget = implicit_kd.VisitBudget(limit=10_000)
    oracle_budget = implicit_kd.VisitBudget(limit=10_000)

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


def test_native_repair_batch_debits_before_fatal_next_node() -> None:
    build_native = _api("build_native_implicit_region_kd")
    query_native = _api("query_native_repair_batch")
    index = build_native(np.ones((1, 7), dtype=np.uint32), region_count=1)
    budget = implicit_kd.VisitBudget(limit=1)

    with pytest.raises(implicit_kd.QualityRepairFallbackQueryBudgetError, match="visit cap"):
        query_native(
            index,
            np.asarray([1], dtype=np.uint32),
            np.asarray([0], dtype=np.uint32),
            np.asarray([3], dtype=np.uint32),
            radius_px=7,
            budget=budget,
        )

    assert budget.consumed == 1


def test_native_geometry_batch_matches_scalar_queries_bit_exactly() -> None:
    build_native = _api("build_native_implicit_region_kd")
    query_native = _api("query_native_geometry_batch")
    region_ids = np.asarray(
        [[1, 1, 1, 2], [1, 1, 2, 2], [1, 2, 2, 2]],
        dtype=np.uint32,
    )
    native_index = build_native(region_ids, region_count=2)
    oracle_index = implicit_kd.build_implicit_region_kd(region_ids, region_count=2)
    query_regions = np.asarray([1, 1, 2, 2, 2], dtype=np.uint32)
    target_y = np.asarray([-0.5, 0.5, 0.25, 1.5, 4.0], dtype=np.float64)
    target_x = np.asarray([-0.25, 1.5, 2.5, 3.5, 5.0], dtype=np.float64)
    native_budget = implicit_kd.VisitBudget(limit=10_000)
    oracle_budget = implicit_kd.VisitBudget(limit=10_000)

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


def test_native_geometry_batch_uses_distinct_fatal_budget_error() -> None:
    build_native = _api("build_native_implicit_region_kd")
    query_native = _api("query_native_geometry_batch")
    index = build_native(np.ones((1, 7), dtype=np.uint32), region_count=1)
    budget = implicit_kd.VisitBudget(limit=0)

    with pytest.raises(implicit_kd.QualityGeometryQueryBudgetError, match="visit cap"):
        query_native(
            index,
            np.asarray([1], dtype=np.uint32),
            np.asarray([0.0], dtype=np.float64),
            np.asarray([3.0], dtype=np.float64),
            budget=budget,
        )

    assert budget.consumed == 0
