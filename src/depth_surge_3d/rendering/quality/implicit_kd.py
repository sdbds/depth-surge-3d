"""Deterministic implicit per-region k-d index prototype.

This module is deliberately outside the production package.  Task 0 uses it to
prove the exact array layout, tie order, and fatal visit-budget behavior before
the implementation is allowed to enter the renderer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np


KD_STACK_CAPACITY = 64
UINT32_MAX = int(np.iinfo(np.uint32).max)
UINT64_MAX = int(np.iinfo(np.uint64).max)

Distance: TypeAlias = int | float


class QualityKdStackError(RuntimeError):
    """A fixed construction or query stack could not represent the tree."""


class QualityRepairFallbackQueryBudgetError(RuntimeError):
    """Repair fallback lookup crossed its shared frame visit cap."""


class QualityGeometryQueryBudgetError(RuntimeError):
    """Retained-zero geometry lookup crossed its frame visit cap."""


@dataclass
class VisitBudget:
    """Checked uint64 visit counter shared by every applicable query."""

    limit: int
    consumed: int = 0

    def __post_init__(self) -> None:
        self.limit = _checked_uint64("limit", self.limit)
        self.consumed = _checked_uint64("consumed", self.consumed)
        if self.consumed > self.limit:
            raise ValueError("consumed visits cannot exceed the visit limit")

    def debit(self, error_type: type[RuntimeError]) -> None:
        """Debit before a node may affect the candidate or child ranges."""

        if self.consumed >= self.limit:
            raise error_type(f"implicit k-d visit cap {self.limit} exceeded")
        self.consumed += 1


@dataclass(frozen=True)
class KdQueryResult:
    """Exact selected sample and per-query traversal diagnostics."""

    sample_index: int
    distance2: Distance
    visited_nodes: int


@dataclass(frozen=True)
class ImplicitRegionKdIndex:
    """Median-partitioned region ranges with implicit child relationships."""

    height: int
    width: int
    region_count: int
    member_index: np.ndarray
    region_offsets: np.ndarray

    @property
    def indexed_sample_count(self) -> int:
        return int(self.member_index.size)

    @property
    def index_bytes(self) -> int:
        return int(self.member_index.nbytes + self.region_offsets.nbytes)

    def query_repair(
        self,
        region_id: int,
        *,
        target_y: int,
        target_x: int,
        radius_px: int,
        budget: VisitBudget,
    ) -> KdQueryResult | None:
        """Find the exact nearest donor inside an integer pixel radius."""

        _validate_pixel_coordinate("target_y", target_y, self.height)
        _validate_pixel_coordinate("target_x", target_x, self.width)
        radius = _validate_nonnegative_integer("radius_px", radius_px)
        return self._query(
            region_id,
            target_y=target_y,
            target_x=target_x,
            radius2=radius * radius,
            geometry=False,
            budget=budget,
            budget_error=QualityRepairFallbackQueryBudgetError,
        )

    def query_geometry(
        self,
        region_id: int,
        *,
        sy: float,
        sx: float,
        budget: VisitBudget,
    ) -> KdQueryResult:
        """Find the exact nearest native sample to an unclipped source point."""

        target_y = _validate_finite_float("sy", sy)
        target_x = _validate_finite_float("sx", sx)
        result = self._query(
            region_id,
            target_y=target_y,
            target_x=target_x,
            radius2=None,
            geometry=True,
            budget=budget,
            budget_error=QualityGeometryQueryBudgetError,
        )
        if result is None:
            raise ValueError("selected geometry region has no indexed member")
        return result

    def _query(
        self,
        region_id: int,
        *,
        target_y: int | float,
        target_x: int | float,
        radius2: int | None,
        geometry: bool,
        budget: VisitBudget,
        budget_error: type[RuntimeError],
    ) -> KdQueryResult | None:
        if not isinstance(budget, VisitBudget):
            raise TypeError("budget must be VisitBudget")
        start, end = self._region_range(region_id)
        if start == end:
            return None
        return _query_range(
            self.member_index,
            width=self.width,
            start=start,
            end=end,
            target_y=target_y,
            target_x=target_x,
            radius2=radius2,
            geometry=geometry,
            budget=budget,
            budget_error=budget_error,
        )

    def _region_range(self, region_id: int) -> tuple[int, int]:
        region = _validate_positive_integer("region_id", region_id)
        if region > self.region_count:
            raise ValueError("region_id exceeds the canonical region count")
        return int(self.region_offsets[region - 1]), int(self.region_offsets[region])


def _checked_uint64(name: str, value: object) -> int:
    result = _validate_nonnegative_integer(name, value)
    if result > UINT64_MAX:
        raise ValueError(f"{name} exceeds uint64 range")
    return result


def _validate_nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _validate_positive_integer(name: str, value: object) -> int:
    result = _validate_nonnegative_integer(name, value)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_pixel_coordinate(name: str, value: object, limit: int) -> int:
    result = _validate_nonnegative_integer(name, value)
    if result >= limit:
        raise ValueError(f"{name} must lie inside the indexed raster")
    return result


def _validate_finite_float(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_inputs(
    region_ids: np.ndarray,
    region_count: object,
    include: np.ndarray | None,
) -> tuple[int, int, int]:
    if not isinstance(region_ids, np.ndarray) or region_ids.dtype != np.uint32:
        raise TypeError("region_ids must be a uint32 NumPy array")
    if region_ids.ndim != 2 or region_ids.shape[0] <= 0 or region_ids.shape[1] <= 0:
        raise ValueError("region_ids must be a non-empty 2D raster")
    count = _validate_positive_integer("region_count", region_count)
    if count + 1 > UINT32_MAX:
        raise ValueError("region_count plus its sentinel must fit uint32")
    sample_count = int(region_ids.size)
    if sample_count + 1 > UINT32_MAX:
        raise ValueError("sample count plus its sentinel must fit uint32")
    if include is not None and (
        not isinstance(include, np.ndarray)
        or include.dtype != np.bool_
        or include.shape != region_ids.shape
    ):
        raise TypeError("include must be a bool NumPy array matching region_ids")
    return int(region_ids.shape[0]), int(region_ids.shape[1]), count


def group_region_members(
    region_ids: np.ndarray,
    *,
    region_count: int,
    include: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Group included sample indexes using the frozen reverse-fill algorithm."""

    _height, _width, count = _validate_inputs(region_ids, region_count, include)
    offsets = np.zeros(count + 1, dtype=np.uint32)
    seen = np.zeros(count, dtype=np.bool_)
    included_count = _count_regions(region_ids, include, offsets, seen)
    if not bool(seen.all()):
        raise ValueError("positive region IDs must be contiguous through region_count")
    _prefix_region_ends(offsets, included_count)
    members = np.empty(included_count, dtype=np.uint32)
    _reverse_fill_members(region_ids, include, members, offsets)
    return members, offsets


def _count_regions(
    region_ids: np.ndarray,
    include: np.ndarray | None,
    offsets: np.ndarray,
    seen: np.ndarray,
) -> int:
    included_count = 0
    for sample_index in range(int(region_ids.size)):
        region = int(region_ids.flat[sample_index])
        if region == 0:
            raise ValueError("every native sample must have a positive region ID")
        if region > seen.size:
            raise ValueError("region ID exceeds region_count")
        seen[region - 1] = True
        if include is None or bool(include.flat[sample_index]):
            offsets[region - 1] += np.uint32(1)
            included_count += 1
    return included_count


def _prefix_region_ends(offsets: np.ndarray, included_count: int) -> None:
    running = 0
    for region_index in range(offsets.size - 1):
        running += int(offsets[region_index])
        offsets[region_index] = np.uint32(running)
    offsets[-1] = np.uint32(included_count)


def _reverse_fill_members(
    region_ids: np.ndarray,
    include: np.ndarray | None,
    members: np.ndarray,
    offsets: np.ndarray,
) -> None:
    for sample_index in range(int(region_ids.size) - 1, -1, -1):
        if include is not None and not bool(include.flat[sample_index]):
            continue
        region_slot = int(region_ids.flat[sample_index]) - 1
        destination = int(offsets[region_slot]) - 1
        offsets[region_slot] = np.uint32(destination)
        members[destination] = np.uint32(sample_index)


def build_implicit_region_kd(
    region_ids: np.ndarray,
    *,
    region_count: int,
    include: np.ndarray | None = None,
) -> ImplicitRegionKdIndex:
    """Build deterministic lower-median trees inside contiguous region ranges."""

    height, width, count = _validate_inputs(region_ids, region_count, include)
    members, offsets = group_region_members(
        region_ids,
        region_count=count,
        include=include,
    )
    for region_index in range(count):
        _partition_region_tree(
            members,
            int(offsets[region_index]),
            int(offsets[region_index + 1]),
            width,
        )
    members.setflags(write=False)
    offsets.setflags(write=False)
    return ImplicitRegionKdIndex(height, width, count, members, offsets)


def _partition_region_tree(
    members: np.ndarray,
    start: int,
    end: int,
    width: int,
) -> None:
    if start == end:
        return
    stack_start = np.empty(KD_STACK_CAPACITY, dtype=np.uint32)
    stack_end = np.empty(KD_STACK_CAPACITY, dtype=np.uint32)
    stack_depth = np.empty(KD_STACK_CAPACITY, dtype=np.uint8)
    top = _push_build_frame(stack_start, stack_end, stack_depth, 0, start, end, 0)
    while top:
        top -= 1
        lower = int(stack_start[top])
        upper = int(stack_end[top])
        depth = int(stack_depth[top])
        middle = lower + ((upper - lower - 1) // 2)
        _select_kth(members, lower, upper - 1, middle, width, depth & 1)
        top = _push_children(
            stack_start,
            stack_end,
            stack_depth,
            top,
            lower,
            middle,
            upper,
            depth + 1,
        )


def _push_children(
    starts: np.ndarray,
    ends: np.ndarray,
    depths: np.ndarray,
    top: int,
    lower: int,
    middle: int,
    upper: int,
    depth: int,
) -> int:
    if middle + 1 < upper:
        top = _push_build_frame(starts, ends, depths, top, middle + 1, upper, depth)
    if lower < middle:
        top = _push_build_frame(starts, ends, depths, top, lower, middle, depth)
    return top


def _push_build_frame(
    starts: np.ndarray,
    ends: np.ndarray,
    depths: np.ndarray,
    top: int,
    start: int,
    end: int,
    depth: int,
) -> int:
    if top >= KD_STACK_CAPACITY:
        raise QualityKdStackError("implicit k-d construction stack capacity exceeded")
    starts[top] = np.uint32(start)
    ends[top] = np.uint32(end)
    depths[top] = np.uint8(depth)
    return top + 1


def _member_key(sample_index: int, width: int, axis: int) -> tuple[int, int, int]:
    y, x = divmod(sample_index, width)
    return (y, x, sample_index) if axis == 0 else (x, y, sample_index)


def _select_kth(
    members: np.ndarray,
    left: int,
    right: int,
    kth: int,
    width: int,
    axis: int,
) -> None:
    while left < right:
        pivot_index = _median_of_medians(members, left, right, width, axis)
        pivot_index = _partition_about(members, left, right, pivot_index, width, axis)
        if kth == pivot_index:
            return
        if kth < pivot_index:
            right = pivot_index - 1
        else:
            left = pivot_index + 1


def _median_of_medians(
    members: np.ndarray,
    left: int,
    right: int,
    width: int,
    axis: int,
) -> int:
    size = right - left + 1
    if size <= 5:
        _insertion_sort(members, left, right, width, axis)
        return left + ((size - 1) // 2)
    median_end = left
    group_start = left
    while group_start <= right:
        group_end = min(group_start + 4, right)
        _insertion_sort(members, group_start, group_end, width, axis)
        group_median = group_start + ((group_end - group_start) // 2)
        members[median_end], members[group_median] = (
            members[group_median],
            members[median_end],
        )
        median_end += 1
        group_start += 5
    median_kth = left + ((median_end - left - 1) // 2)
    _select_kth(members, left, median_end - 1, median_kth, width, axis)
    return median_kth


def _insertion_sort(
    members: np.ndarray,
    left: int,
    right: int,
    width: int,
    axis: int,
) -> None:
    for cursor in range(left + 1, right + 1):
        value = np.uint32(members[cursor])
        value_key = _member_key(int(value), width, axis)
        destination = cursor
        while (
            destination > left
            and _member_key(int(members[destination - 1]), width, axis) > value_key
        ):
            members[destination] = members[destination - 1]
            destination -= 1
        members[destination] = value


def _partition_about(
    members: np.ndarray,
    left: int,
    right: int,
    pivot_index: int,
    width: int,
    axis: int,
) -> int:
    pivot = np.uint32(members[pivot_index])
    pivot_key = _member_key(int(pivot), width, axis)
    members[pivot_index], members[right] = members[right], members[pivot_index]
    destination = left
    for cursor in range(left, right):
        if _member_key(int(members[cursor]), width, axis) < pivot_key:
            members[destination], members[cursor] = members[cursor], members[destination]
            destination += 1
    members[right], members[destination] = members[destination], members[right]
    return destination


def _query_range(
    members: np.ndarray,
    *,
    width: int,
    start: int,
    end: int,
    target_y: int | float,
    target_x: int | float,
    radius2: int | None,
    geometry: bool,
    budget: VisitBudget,
    budget_error: type[RuntimeError],
) -> KdQueryResult | None:
    stack = _QueryStack(geometry)
    stack.push_range(start, end, 0)
    best: tuple[Distance, int, int, int] | None = None
    visited = 0
    while stack.has_items:
        frame = stack.pop()
        if frame.check_far:
            if _far_range_is_needed(frame.plane2, radius2, best):
                stack.push_range(frame.start, frame.end, frame.depth)
            continue
        budget.debit(budget_error)
        visited += 1
        best = _visit_query_node(
            members,
            frame,
            width,
            target_y,
            target_x,
            radius2,
            geometry,
            best,
        )
        _push_query_children(stack, members, frame, width, target_y, target_x)
    if best is None:
        return None
    return KdQueryResult(best[3], best[0], visited)


@dataclass(frozen=True)
class _QueryFrame:
    start: int
    end: int
    depth: int
    check_far: bool
    plane2: Distance


class _QueryStack:
    def __init__(self, geometry: bool) -> None:
        self._starts = np.empty(KD_STACK_CAPACITY, dtype=np.uint32)
        self._ends = np.empty(KD_STACK_CAPACITY, dtype=np.uint32)
        self._depths = np.empty(KD_STACK_CAPACITY, dtype=np.uint8)
        self._checks = np.empty(KD_STACK_CAPACITY, dtype=np.bool_)
        plane_dtype = np.float64 if geometry else np.int64
        self._planes = np.empty(KD_STACK_CAPACITY, dtype=plane_dtype)
        self._top = 0

    @property
    def has_items(self) -> bool:
        return self._top > 0

    def push_range(self, start: int, end: int, depth: int) -> None:
        self._push(start, end, depth, False, 0)

    def push_far(self, start: int, end: int, depth: int, plane2: Distance) -> None:
        self._push(start, end, depth, True, plane2)

    def _push(
        self,
        start: int,
        end: int,
        depth: int,
        check_far: bool,
        plane2: Distance,
    ) -> None:
        if self._top >= KD_STACK_CAPACITY:
            raise QualityKdStackError("implicit k-d query stack capacity exceeded")
        self._starts[self._top] = np.uint32(start)
        self._ends[self._top] = np.uint32(end)
        self._depths[self._top] = np.uint8(depth)
        self._checks[self._top] = np.bool_(check_far)
        self._planes[self._top] = plane2
        self._top += 1

    def pop(self) -> _QueryFrame:
        self._top -= 1
        return _QueryFrame(
            int(self._starts[self._top]),
            int(self._ends[self._top]),
            int(self._depths[self._top]),
            bool(self._checks[self._top]),
            self._planes[self._top].item(),
        )


def _far_range_is_needed(
    plane2: Distance,
    radius2: int | None,
    best: tuple[Distance, int, int, int] | None,
) -> bool:
    if radius2 is None:
        return best is None or plane2 <= best[0]
    bound: Distance = radius2 if best is None else min(radius2, best[0])
    return plane2 <= bound


def _visit_query_node(
    members: np.ndarray,
    frame: _QueryFrame,
    width: int,
    target_y: int | float,
    target_x: int | float,
    radius2: int | None,
    geometry: bool,
    best: tuple[Distance, int, int, int] | None,
) -> tuple[Distance, int, int, int] | None:
    middle = frame.start + ((frame.end - frame.start - 1) // 2)
    sample_index = int(members[middle])
    y, x = divmod(sample_index, width)
    distance2 = _query_distance2(target_y, target_x, y, x, geometry)
    if radius2 is not None and distance2 > radius2:
        return best
    candidate = (distance2, y, x, sample_index)
    return candidate if best is None or candidate < best else best


def _query_distance2(
    target_y: int | float,
    target_x: int | float,
    y: int,
    x: int,
    geometry: bool,
) -> Distance:
    if geometry:
        dx = float(target_x) - float(x)
        dy = float(target_y) - float(y)
        return (dx * dx) + (dy * dy)
    dx_int = int(target_x) - x
    dy_int = int(target_y) - y
    return (dx_int * dx_int) + (dy_int * dy_int)


def _push_query_children(
    stack: _QueryStack,
    members: np.ndarray,
    frame: _QueryFrame,
    width: int,
    target_y: int | float,
    target_x: int | float,
) -> None:
    middle = frame.start + ((frame.end - frame.start - 1) // 2)
    sample_index = int(members[middle])
    y, x = divmod(sample_index, width)
    target_axis = target_y if (frame.depth & 1) == 0 else target_x
    node_axis = y if (frame.depth & 1) == 0 else x
    delta = target_axis - node_axis
    lower = (frame.start, middle)
    upper = (middle + 1, frame.end)
    near, far = (lower, upper) if delta <= 0 else (upper, lower)
    next_depth = frame.depth + 1
    if far[0] < far[1]:
        stack.push_far(far[0], far[1], next_depth, delta * delta)
    if near[0] < near[1]:
        stack.push_range(near[0], near[1], next_depth)
