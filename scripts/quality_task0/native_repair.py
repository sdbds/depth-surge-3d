"""Python boundary for prebuilt native repair-record construction."""

from __future__ import annotations

import numpy as np

from . import native_kd, repair_records


def build_native_repair_records(
    valid: np.ndarray,
    winner_near_score: np.ndarray,
    winner_region_id: np.ndarray,
    *,
    winner_bgr: np.ndarray | None = None,
) -> repair_records.RepairRecordBuildResult:
    """Construct the fixed record table without Python objects per run."""

    repair_records._validate_winner_rasters(
        valid,
        winner_near_score,
        winner_region_id,
    )
    if winner_bgr is not None and (
        not isinstance(winner_bgr, np.ndarray)
        or winner_bgr.dtype != np.float32
        or winner_bgr.shape != (*valid.shape, 3)
        or not winner_bgr.flags.c_contiguous
    ):
        raise TypeError("winner_bgr must be a matching C-contiguous float32 [H,W,3] raster")
    native = native_kd._require_native()
    raw_records, statistics = native.build_repair_records(
        valid,
        winner_near_score,
        winner_region_id,
        winner_bgr,
    )
    records = raw_records.reshape(-1).view(repair_records.REPAIR_RECORD_DTYPE)
    records.setflags(write=False)
    return repair_records.RepairRecordBuildResult(
        records=records,
        prefill_run_count=int(statistics["prefill_run_count"]),
        repair_run_count=int(statistics["repair_run_count"]),
        invalid_lane_count=int(statistics["invalid_lane_count"]),
    )
