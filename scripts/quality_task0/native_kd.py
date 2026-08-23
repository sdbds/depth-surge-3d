"""Python boundary for the prebuilt Task 0 native k-d extension."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from types import ModuleType

import numpy as np

from .implicit_kd import (
    UINT32_MAX,
    ImplicitRegionKdIndex,
    QualityGeometryQueryBudgetError,
    QualityRepairFallbackQueryBudgetError,
    VisitBudget,
    _validate_inputs,
    _validate_nonnegative_integer,
)


_NATIVE: ModuleType | None
_LOAD_ERROR: BaseException | None

try:
    _NATIVE = importlib.import_module("_quality_kd_native")
except (ImportError, OSError) as error:
    _NATIVE = None
    _LOAD_ERROR = error
else:
    _LOAD_ERROR = None


def native_available() -> bool:
    """Return whether a build-time-produced extension is importable."""

    return _NATIVE is not None


def native_load_error() -> str | None:
    """Return an actionable import failure without compiling at runtime."""

    return None if _LOAD_ERROR is None else f"{type(_LOAD_ERROR).__name__}: {_LOAD_ERROR}"


def native_build_info() -> dict[str, object]:
    """Return the semantic contract embedded in the native binary."""

    native = _require_native()
    return dict(native.build_info())


def build_native_implicit_region_kd(
    region_ids: np.ndarray,
    *,
    region_count: int,
    include: np.ndarray | None = None,
) -> ImplicitRegionKdIndex:
    """Build one native index after the shared strict Python validation."""

    height, width, count = _validate_inputs(region_ids, region_count, include)
    native = _require_native()
    members, offsets = native.build_index(region_ids, count, include)
    members.setflags(write=False)
    offsets.setflags(write=False)
    return ImplicitRegionKdIndex(height, width, count, members, offsets)


@dataclass(frozen=True)
class NativeBatchQueryResult:
    """Immutable vector results plus exact per-query visit counts."""

    found: np.ndarray
    sample_index: np.ndarray
    distance2: np.ndarray
    visited_nodes: np.ndarray


def query_native_repair_batch(
    index: ImplicitRegionKdIndex,
    query_regions: np.ndarray,
    target_y: np.ndarray,
    target_x: np.ndarray,
    *,
    radius_px: int,
    budget: VisitBudget,
) -> NativeBatchQueryResult:
    """Execute exact integer-radius repair lookups in one native call."""

    _validate_index_and_budget(index, budget)
    query_count = _validate_query_vector("query_regions", query_regions, np.uint32)
    _validate_query_vector("target_y", target_y, np.uint32, query_count)
    _validate_query_vector("target_x", target_x, np.uint32, query_count)
    radius = _validate_nonnegative_integer("radius_px", radius_px)
    budget_state = np.asarray([budget.consumed, budget.limit], dtype=np.uint64)
    native = _require_native()
    try:
        samples, distances, visits = native.query_repair_batch(
            index.member_index,
            index.region_offsets,
            index.height,
            index.width,
            query_regions,
            target_y,
            target_x,
            radius,
            budget_state,
        )
    except RuntimeError as error:
        if "quality repair fallback visit cap exceeded" in str(error):
            raise QualityRepairFallbackQueryBudgetError(
                f"implicit k-d visit cap {budget.limit} exceeded"
            ) from error
        raise
    finally:
        budget.consumed = int(budget_state[0])
    return _freeze_batch(samples, distances, visits)


def query_native_geometry_batch(
    index: ImplicitRegionKdIndex,
    query_regions: np.ndarray,
    target_y: np.ndarray,
    target_x: np.ndarray,
    *,
    budget: VisitBudget,
) -> NativeBatchQueryResult:
    """Execute exact binary64 retained-zero lookups in one native call."""

    _validate_index_and_budget(index, budget)
    query_count = _validate_query_vector("query_regions", query_regions, np.uint32)
    _validate_query_vector("target_y", target_y, np.float64, query_count)
    _validate_query_vector("target_x", target_x, np.float64, query_count)
    budget_state = np.asarray([budget.consumed, budget.limit], dtype=np.uint64)
    native = _require_native()
    try:
        samples, distances, visits = native.query_geometry_batch(
            index.member_index,
            index.region_offsets,
            index.height,
            index.width,
            query_regions,
            target_y,
            target_x,
            budget_state,
        )
    except RuntimeError as error:
        if "quality geometry visit cap exceeded" in str(error):
            raise QualityGeometryQueryBudgetError(
                f"implicit k-d visit cap {budget.limit} exceeded"
            ) from error
        raise
    finally:
        budget.consumed = int(budget_state[0])
    return _freeze_batch(samples, distances, visits)


def _validate_index_and_budget(index: object, budget: object) -> None:
    if not isinstance(index, ImplicitRegionKdIndex):
        raise TypeError("index must be ImplicitRegionKdIndex")
    if not isinstance(budget, VisitBudget):
        raise TypeError("budget must be VisitBudget")


def _validate_query_vector(
    name: str,
    values: object,
    dtype: type[np.generic],
    expected_length: int | None = None,
) -> int:
    if not isinstance(values, np.ndarray) or values.dtype != np.dtype(dtype):
        raise TypeError(f"{name} must be a NumPy array with dtype {np.dtype(dtype)}")
    if values.ndim != 1 or not values.flags.c_contiguous:
        raise ValueError(f"{name} must be a C-contiguous one-dimensional array")
    if expected_length is not None and values.size != expected_length:
        raise ValueError(f"{name} must match the query count")
    return int(values.size)


def _freeze_batch(
    samples: np.ndarray,
    distances: np.ndarray,
    visits: np.ndarray,
) -> NativeBatchQueryResult:
    found = np.ascontiguousarray(samples != np.uint32(UINT32_MAX), dtype=np.bool_)
    for values in (found, samples, distances, visits):
        values.setflags(write=False)
    return NativeBatchQueryResult(found, samples, distances, visits)


def _require_native() -> ModuleType:
    if _NATIVE is None:
        detail = native_load_error() or "unknown native import failure"
        raise RuntimeError(
            "The prebuilt Quality k-d extension is unavailable; build or install "
            f"a matching wheel ({detail})"
        )
    return _NATIVE
