"""Data types, visit budgets, and native-delegated wrappers for per-region k-d lookup."""

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
    """Median-partitioned region ranges built by the native k-d extension."""

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
        """Find the nearest donor inside an integer pixel radius."""

        py = _validate_pixel_coordinate("target_y", target_y, self.height)
        px = _validate_pixel_coordinate("target_x", target_x, self.width)
        radius = _validate_nonnegative_integer("radius_px", radius_px)
        if not isinstance(budget, VisitBudget):
            raise TypeError("budget must be VisitBudget")
        start, end = self._region_range(region_id)
        if start == end:
            return None
        members = self.member_index[start:end]
        visits = int(members.size)
        if budget.consumed + visits > budget.limit:
            budget.consumed = budget.limit
            raise QualityRepairFallbackQueryBudgetError(
                f"implicit k-d visit cap {budget.limit} exceeded"
            )
        budget.consumed += visits
        ys = (members // np.uint32(self.width)).astype(np.int64)
        xs = (members % np.uint32(self.width)).astype(np.int64)
        dy = ys - py
        dx = xs - px
        dist2 = dy * dy + dx * dx
        within = np.flatnonzero(dist2 <= radius * radius)
        if within.size == 0:
            return None
        cand_dist = dist2[within]
        cand_idx = members[within]
        best_order = np.lexsort((cand_idx, cand_dist))[0]
        return KdQueryResult(
            sample_index=int(cand_idx[best_order]),
            distance2=int(cand_dist[best_order]),
            visited_nodes=visits,
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
        if not isinstance(budget, VisitBudget):
            raise TypeError("budget must be VisitBudget")
        start, end = self._region_range(region_id)
        if start == end:
            raise ValueError("selected geometry region has no indexed member")
        from . import native_kd

        batch = native_kd.query_native_geometry_batch(
            self,
            np.array([region_id], dtype=np.uint32),
            np.array([target_y], dtype=np.float64),
            np.array([target_x], dtype=np.float64),
            budget=budget,
        )
        return KdQueryResult(
            sample_index=int(batch.sample_index[0]),
            distance2=float(batch.distance2[0]),
            visited_nodes=int(batch.visited_nodes[0]),
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


def build_implicit_region_kd(
    region_ids: np.ndarray,
    *,
    region_count: int,
    include: np.ndarray | None = None,
) -> ImplicitRegionKdIndex:
    """Build deterministic per-region k-d index via the native extension."""

    from . import native_kd

    return native_kd.build_native_implicit_region_kd(
        region_ids,
        region_count=region_count,
        include=include,
    )
