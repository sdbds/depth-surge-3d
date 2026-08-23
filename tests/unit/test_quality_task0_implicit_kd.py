"""Task 0 contract tests for the deterministic implicit per-region k-d index."""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable

import numpy as np
import pytest

from scripts.quality_task0 import implicit_kd as kd


def test_implicit_kd_prototype_module_is_importable() -> None:
    assert importlib.util.find_spec("scripts.quality_task0.implicit_kd") is not None


def _api(name: str):
    assert hasattr(kd, name), f"missing Task 0 API: {name}"
    return getattr(kd, name)


def _included_members(
    region_ids: np.ndarray,
    region_id: int,
    include: np.ndarray | None = None,
) -> Iterable[tuple[int, int, int]]:
    height, width = region_ids.shape
    for sample_index in range(height * width):
        y, x = divmod(sample_index, width)
        if int(region_ids[y, x]) == region_id and (include is None or bool(include[y, x])):
            yield sample_index, y, x


def _repair_oracle(
    region_ids: np.ndarray,
    include: np.ndarray,
    region_id: int,
    target_y: int,
    target_x: int,
    radius_px: int,
) -> tuple[int, int] | None:
    best: tuple[int, int, int, int] | None = None
    radius2 = radius_px * radius_px
    for sample_index, y, x in _included_members(region_ids, region_id, include):
        dy = target_y - y
        dx = target_x - x
        distance2 = dy * dy + dx * dx
        if distance2 <= radius2:
            candidate = (distance2, y, x, sample_index)
            if best is None or candidate < best:
                best = candidate
    return None if best is None else (best[3], best[0])


def _geometry_oracle(
    region_ids: np.ndarray,
    region_id: int,
    sy: float,
    sx: float,
) -> tuple[int, float]:
    best: tuple[float, int, int, int] | None = None
    for sample_index, y, x in _included_members(region_ids, region_id):
        dx = sx - float(x)
        dy = sy - float(y)
        distance2 = (dx * dx) + (dy * dy)
        candidate = (distance2, y, x, sample_index)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best[3], best[0]


def test_grouping_uses_uint32_reverse_fill_and_row_major_members() -> None:
    group_region_members = _api("group_region_members")
    region_ids = np.asarray([[2, 1, 2], [1, 2, 1]], dtype=np.uint32)

    members, offsets = group_region_members(region_ids, region_count=2)

    assert members.dtype == np.uint32
    assert offsets.dtype == np.uint32
    assert members.tolist() == [1, 3, 5, 0, 2, 4]
    assert offsets.tolist() == [0, 3, 6]


def test_grouping_retains_empty_region_ranges_for_filtered_donors() -> None:
    group_region_members = _api("group_region_members")
    region_ids = np.asarray([[1, 1, 2], [2, 3, 3]], dtype=np.uint32)
    include = np.asarray([[True, False, False], [False, True, True]], dtype=np.bool_)

    members, offsets = group_region_members(region_ids, region_count=3, include=include)

    assert members.tolist() == [0, 4, 5]
    assert offsets.tolist() == [0, 1, 1, 3]


@pytest.mark.parametrize(
    ("region_ids", "region_count", "message"),
    [
        (np.asarray([[1, 3]], dtype=np.uint32), 3, "contiguous"),
        (np.asarray([[0, 1]], dtype=np.uint32), 1, "positive"),
    ],
)
def test_geometry_grouping_rejects_noncanonical_region_labels(
    region_ids: np.ndarray,
    region_count: int,
    message: str,
) -> None:
    group_region_members = _api("group_region_members")

    with pytest.raises(ValueError, match=message):
        group_region_members(region_ids, region_count=region_count)


def test_build_uses_lower_median_and_exact_array_byte_accounting() -> None:
    build_index = _api("build_implicit_region_kd")
    region_ids = np.ones((2, 2), dtype=np.uint32)

    index = build_index(region_ids, region_count=1)

    assert index.member_index[index.region_offsets[0] + 1] == np.uint32(1)
    assert index.member_index.dtype == np.uint32
    assert index.region_offsets.dtype == np.uint32
    assert index.indexed_sample_count == 4
    assert index.index_bytes == index.member_index.nbytes + index.region_offsets.nbytes == 24
    assert not index.member_index.flags.writeable
    assert not index.region_offsets.flags.writeable


def _expected_implicit_layout(samples: list[int], width: int, depth: int = 0) -> list[int]:
    if not samples:
        return []
    axis = depth & 1

    def key(sample_index: int) -> tuple[int, int, int]:
        y, x = divmod(sample_index, width)
        return (y, x, sample_index) if axis == 0 else (x, y, sample_index)

    ordered = sorted(samples, key=key)
    middle = (len(ordered) - 1) // 2
    return (
        _expected_implicit_layout(ordered[:middle], width, depth + 1)
        + [ordered[middle]]
        + _expected_implicit_layout(ordered[middle + 1 :], width, depth + 1)
    )


def test_build_matches_independent_alternating_axis_lower_median_layout() -> None:
    build_index = _api("build_implicit_region_kd")
    region_ids = np.ones((3, 4), dtype=np.uint32)

    index = build_index(region_ids, region_count=1)

    assert index.member_index.tolist() == _expected_implicit_layout(list(range(12)), width=4)


def test_build_checks_its_fixed_stack_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    build_index = _api("build_implicit_region_kd")
    stack_error = _api("QualityKdStackError")
    monkeypatch.setattr(kd, "KD_STACK_CAPACITY", 1)

    with pytest.raises(stack_error, match="construction stack"):
        build_index(np.ones((2, 2), dtype=np.uint32), region_count=1)


def test_repair_query_uses_exact_radius_and_row_column_ties() -> None:
    build_index = _api("build_implicit_region_kd")
    visit_budget = _api("VisitBudget")
    region_ids = np.ones((3, 3), dtype=np.uint32)
    donors = np.asarray(
        [[False, True, False], [True, False, True], [False, True, False]],
        dtype=np.bool_,
    )
    index = build_index(region_ids, region_count=1, include=donors)
    budget = visit_budget(limit=100)

    result = index.query_repair(1, target_y=1, target_x=1, radius_px=1, budget=budget)

    assert result is not None
    assert result.sample_index == 1
    assert result.distance2 == 1
    assert result.visited_nodes >= 1
    assert budget.consumed == result.visited_nodes

    exact = index.query_repair(1, target_y=0, target_x=0, radius_px=1, budget=budget)
    assert exact is not None and exact.sample_index == 1 and exact.distance2 == 1
    assert index.query_repair(1, target_y=0, target_x=0, radius_px=0, budget=budget) is None


def test_repair_query_matches_independent_scalar_oracle() -> None:
    build_index = _api("build_implicit_region_kd")
    visit_budget = _api("VisitBudget")
    rng = np.random.default_rng(8675309)
    region_ids = np.asarray(
        [[1, 1, 1, 2, 2, 2, 2], [1, 1, 1, 2, 2, 2, 2], [1, 1, 3, 3, 3, 2, 2]],
        dtype=np.uint32,
    )
    include = rng.random(region_ids.shape) > 0.35
    include[0, 0] = True
    include[0, 3] = True
    include[2, 2] = True
    index = build_index(region_ids, region_count=3, include=include)
    budget = visit_budget(limit=10_000)

    for region_id in (1, 2, 3):
        for target_y in range(region_ids.shape[0]):
            for target_x in range(region_ids.shape[1]):
                for radius_px in (0, 1, 2, 8):
                    expected = _repair_oracle(
                        region_ids,
                        include,
                        region_id,
                        target_y,
                        target_x,
                        radius_px,
                    )
                    actual = index.query_repair(
                        region_id,
                        target_y=target_y,
                        target_x=target_x,
                        radius_px=radius_px,
                        budget=budget,
                    )
                    assert (
                        None if actual is None else (actual.sample_index, actual.distance2)
                    ) == (expected)


def test_filtered_region_without_a_donor_has_no_repair_result() -> None:
    build_index = _api("build_implicit_region_kd")
    visit_budget = _api("VisitBudget")
    region_ids = np.asarray([[1, 1, 2, 2]], dtype=np.uint32)
    include = np.asarray([[True, False, False, False]], dtype=np.bool_)
    index = build_index(region_ids, region_count=2, include=include)
    budget = visit_budget(limit=100)

    assert index.query_repair(2, target_y=0, target_x=3, radius_px=4, budget=budget) is None
    assert budget.consumed == 0


def test_repair_query_fails_before_using_the_node_past_shared_budget() -> None:
    build_index = _api("build_implicit_region_kd")
    visit_budget = _api("VisitBudget")
    budget_error = _api("QualityRepairFallbackQueryBudgetError")
    index = build_index(np.ones((1, 7), dtype=np.uint32), region_count=1)
    budget = visit_budget(limit=1)

    with pytest.raises(budget_error, match="visit cap"):
        index.query_repair(1, target_y=0, target_x=3, radius_px=7, budget=budget)

    assert budget.consumed == 1


def test_geometry_query_uses_unclipped_binary64_coordinates_and_ties() -> None:
    build_index = _api("build_implicit_region_kd")
    visit_budget = _api("VisitBudget")
    region_ids = np.ones((2, 2), dtype=np.uint32)
    index = build_index(region_ids, region_count=1)
    budget = visit_budget(limit=100)

    tied = index.query_geometry(1, sy=0.5, sx=0.5, budget=budget)
    unclipped = index.query_geometry(1, sy=0.5, sx=-0.25, budget=budget)

    assert tied.sample_index == 0
    assert tied.distance2 == np.float64(0.5)
    assert unclipped.sample_index == 0
    assert unclipped.distance2 == np.float64(0.3125)


def test_geometry_query_matches_independent_scalar_oracle_on_degenerate_regions() -> None:
    build_index = _api("build_implicit_region_kd")
    visit_budget = _api("VisitBudget")
    region_ids = np.asarray([[1, 1, 1, 2, 2, 2, 2]], dtype=np.uint32)
    index = build_index(region_ids, region_count=2)
    budget = visit_budget(limit=10_000)

    for region_id in (1, 2):
        for sy, sx in ((-0.75, -0.25), (0.0, 2.5), (0.25, 3.5), (1.0, 8.0)):
            expected = _geometry_oracle(region_ids, region_id, sy, sx)
            actual = index.query_geometry(region_id, sy=sy, sx=sx, budget=budget)
            assert actual.sample_index == expected[0]
            assert actual.distance2 == expected[1]


def test_geometry_query_matches_independent_scalar_oracle_on_seeded_2d_queries() -> None:
    build_index = _api("build_implicit_region_kd")
    visit_budget = _api("VisitBudget")
    region_ids = np.asarray(
        [
            [1, 1, 1, 2, 2],
            [1, 1, 3, 2, 2],
            [1, 3, 3, 3, 2],
            [3, 3, 3, 3, 2],
        ],
        dtype=np.uint32,
    )
    rng = np.random.default_rng(20260823)
    index = build_index(region_ids, region_count=3)
    budget = visit_budget(limit=100_000)

    for region_id in (1, 2, 3):
        for sy, sx in rng.uniform((-2.0, -2.0), (6.0, 7.0), size=(40, 2)):
            expected = _geometry_oracle(region_ids, region_id, float(sy), float(sx))
            actual = index.query_geometry(
                region_id,
                sy=float(sy),
                sx=float(sx),
                budget=budget,
            )
            assert actual.sample_index == expected[0]
            assert actual.distance2 == expected[1]


def test_geometry_query_uses_its_distinct_fatal_budget_error() -> None:
    build_index = _api("build_implicit_region_kd")
    visit_budget = _api("VisitBudget")
    budget_error = _api("QualityGeometryQueryBudgetError")
    index = build_index(np.ones((1, 7), dtype=np.uint32), region_count=1)
    budget = visit_budget(limit=0)

    with pytest.raises(budget_error, match="visit cap"):
        index.query_geometry(1, sy=0.0, sx=3.0, budget=budget)

    assert budget.consumed == 0
