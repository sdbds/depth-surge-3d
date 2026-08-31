"""Python boundary for the prebuilt native RGB geodesic solver."""

from __future__ import annotations

import numpy as np

from . import native_kd, region_solver
from .indexed_heap import QualityRegionQueueBudgetError


def solve_native_geodesic_regions(
    guide: np.ndarray,
    band_mask: np.ndarray,
    seed_region_map: np.ndarray,
    region_rank_bits: np.ndarray,
    *,
    movement_base_cost: int = 256,
    movement_edge_scale: int = 8,
    mutable_region_map: bool = False,
) -> region_solver.GeodesicSolveResult:
    """Assign a full render band using the build-time native indexed heap."""

    region_solver._validate_solver_inputs(
        guide,
        band_mask,
        seed_region_map,
        region_rank_bits,
    )
    base_cost = region_solver._validate_positive_dimension(
        "movement_base_cost",
        movement_base_cost,
    )
    edge_scale = region_solver._validate_positive_dimension(
        "movement_edge_scale",
        movement_edge_scale,
    )
    if not isinstance(mutable_region_map, bool):
        raise TypeError("mutable_region_map must be a bool")
    native = native_kd._require_native()
    try:
        owner, distance, position, statistics = native.solve_geodesic_regions(
            guide,
            band_mask,
            seed_region_map,
            region_rank_bits,
            base_cost,
            edge_scale,
        )
    except RuntimeError as error:
        if "quality region queue" in str(error):
            raise QualityRegionQueueBudgetError(str(error)) from error
        raise
    frozen_values = (distance, position) if mutable_region_map else (owner, distance, position)
    for values in frozen_values:
        values.setflags(write=False)
    return region_solver.GeodesicSolveResult(
        region_map=owner,
        distance=distance,
        heap_position=position,
        max_live_entries=int(statistics["max_live_entries"]),
        settled_count=int(statistics["settled_count"]),
        queue_insertions=int(statistics["queue_insertions"]),
        queue_decrease_keys=int(statistics["queue_decrease_keys"]),
        queue_pops=int(statistics["queue_pops"]),
        heap_pixel_bytes=int(statistics["heap_pixel_bytes"]),
        heap_position_bytes=int(position.nbytes),
        distance_bytes=int(distance.nbytes),
    )
