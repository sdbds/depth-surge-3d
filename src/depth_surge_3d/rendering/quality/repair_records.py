"""Deterministic sparse repair-record prototype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


HORIZONTAL_SUBPIXELS = 16
QUALITY_REPAIR_RECORD_ARENA_BYTES = 64 * 1024 * 1024
QUALITY_REPAIR_RECORD_CAP = QUALITY_REPAIR_RECORD_ARENA_BYTES // 16
NO_REGION = np.uint32(0xFFFFFFFF)

REPAIR_RECORD_DTYPE = np.dtype(
    {
        "names": (
            "pixel_index",
            "lane_mask",
            "region_id",
            "fill_bgr",
            "backend",
            "far_side",
            "reserved",
        ),
        "formats": (
            "<u4",
            "<u2",
            "<u4",
            ("u1", (3,)),
            "u1",
            "u1",
            "u1",
        ),
        "offsets": (0, 4, 6, 10, 13, 14, 15),
        "itemsize": 16,
    },
    align=False,
)
if REPAIR_RECORD_DTYPE.itemsize != 16:
    raise RuntimeError("repair record dtype must remain exactly 16 bytes")

REPAIR_RUN_DTYPE = np.dtype(
    [
        ("record_start", "<u4"),
        ("record_end", "<u4"),
        ("row", "<u4"),
        ("start_fine", "<u4"),
        ("end_fine", "<u4"),
        ("region_id", "<u4"),
        ("far_side", "u1"),
    ],
    align=False,
)


class QualityRepairBudgetError(RuntimeError):
    """A fixed repair arena or checked counter would be exceeded."""


class QualityRepairNoDonorError(RuntimeError):
    """A pre-fill run has no legal boundary anchor."""


class QualityRepairRecordCoverageError(RuntimeError):
    """Sparse record masks do not exactly partition the invalid lanes."""


@dataclass(frozen=True)
class RepairRecordBuildResult:
    """Immutable sparse records and pre-fill/run counters."""

    records: np.ndarray
    prefill_run_count: int
    repair_run_count: int
    invalid_lane_count: int

    @property
    def record_bytes(self) -> int:
        """Return exact fixed-dtype payload bytes."""

        return int(self.records.nbytes)


@dataclass(frozen=True)
class _RepairRun:
    row: int
    start_fine: int
    end_fine: int
    region_id: int
    far_side: int


def build_repair_records(
    valid: np.ndarray,
    winner_near_score: np.ndarray,
    winner_region_id: np.ndarray,
) -> RepairRecordBuildResult:
    """Split maximal invalid sequences into canonical 16-byte records."""

    height, fine_width = _validate_winner_rasters(
        valid,
        winner_near_score,
        winner_region_id,
    )
    record_count = 0
    repair_run_count = 0
    for run in _iter_repair_runs(valid, winner_near_score, winner_region_id):
        repair_run_count += 1
        record_count += run.end_fine // 16 - run.start_fine // 16 + 1
        if record_count > QUALITY_REPAIR_RECORD_CAP:
            raise QualityRepairBudgetError("64 MiB repair record arena exceeded")

    records = np.zeros(record_count, dtype=REPAIR_RECORD_DTYPE)
    cursor = 0
    prefill_run_count = 0
    for y in range(height):
        prefill_run_count += _count_prefill_runs(valid[y])
    for run in _iter_repair_runs(valid, winner_near_score, winner_region_id):
        cursor = _write_run_records(records, cursor, run, fine_width // 16)
    if cursor != record_count:
        raise QualityRepairBudgetError("repair record preflight count changed")
    invalid_lane_count = int(np.count_nonzero(~valid))
    validate_record_coverage(records, ~valid)
    records.setflags(write=False)
    return RepairRecordBuildResult(
        records=records,
        prefill_run_count=prefill_run_count,
        repair_run_count=repair_run_count,
        invalid_lane_count=invalid_lane_count,
    )


def validate_record_coverage(records: np.ndarray, invalid_mask: np.ndarray) -> int:
    """Verify sorted, disjoint sparse masks against the exact invalid mask."""

    _validate_record_array(records)
    _validate_bool_invalid_mask(invalid_mask)
    height, fine_width = invalid_mask.shape
    width = fine_width // 16
    pixel_count = height * width
    accumulated, record_popcount = _accumulate_record_masks(records, pixel_count)
    expected_popcount = _compare_invalid_masks(accumulated, invalid_mask, width)
    if record_popcount != expected_popcount:
        raise QualityRepairRecordCoverageError(
            "record popcount does not equal the invalid-lane count"
        )
    return record_popcount


def _accumulate_record_masks(
    records: np.ndarray,
    pixel_count: int,
) -> tuple[np.ndarray, int]:
    accumulated = np.zeros(pixel_count, dtype=np.uint16)
    record_popcount = 0
    previous_key: tuple[int, int, int, int] | None = None
    for record in records:
        pixel = int(record["pixel_index"])
        mask = int(record["lane_mask"])
        region = int(record["region_id"])
        far_side = int(record["far_side"])
        if not 0 <= pixel < pixel_count:
            raise QualityRepairRecordCoverageError("record pixel lies outside the frame")
        if not _is_contiguous_nonzero_mask(mask):
            raise QualityRepairRecordCoverageError(
                "record lane mask must be nonzero and contiguous"
            )
        if region == 0 or region == int(NO_REGION):
            raise QualityRepairRecordCoverageError("record region must be positive")
        if far_side not in (0, 1) or int(record["reserved"]) != 0:
            raise QualityRepairRecordCoverageError("record side/reserved byte is invalid")
        key = (pixel, _least_set_bit(mask), region, far_side)
        if previous_key is not None and key < previous_key:
            raise QualityRepairRecordCoverageError("records are not canonically ordered")
        previous_key = key
        if int(accumulated[pixel]) & mask:
            raise QualityRepairRecordCoverageError("record lane masks overlap")
        accumulated[pixel] = np.uint16(int(accumulated[pixel]) | mask)
        record_popcount += mask.bit_count()
    return accumulated, record_popcount


def _compare_invalid_masks(
    accumulated: np.ndarray,
    invalid_mask: np.ndarray,
    width: int,
) -> int:
    expected_popcount = 0
    for pixel in range(accumulated.size):
        expected_mask = 0
        row, column = divmod(pixel, width)
        for lane in range(16):
            if bool(invalid_mask[row, column * 16 + lane]):
                expected_mask |= 1 << lane
        expected_popcount += expected_mask.bit_count()
        if int(accumulated[pixel]) != expected_mask:
            raise QualityRepairRecordCoverageError("record mask OR does not equal the invalid mask")
    return expected_popcount


def reconstruct_repair_runs(
    records: np.ndarray,
    *,
    render_shape: tuple[int, int],
) -> np.ndarray:
    """Reconstruct canonical repair runs from sorted adjacent record pieces."""

    _validate_record_array(records)
    height, width = _validate_render_shape(render_shape)
    runs = np.zeros(records.size, dtype=REPAIR_RUN_DTYPE)
    run_count = 0
    previous_key: tuple[int, int, int, int] | None = None
    for record_index, record in enumerate(records):
        pixel = int(record["pixel_index"])
        if not 0 <= pixel < height * width:
            raise ValueError("record pixel lies outside render_shape")
        mask = int(record["lane_mask"])
        if not _is_contiguous_nonzero_mask(mask):
            raise ValueError("record lane mask must be nonzero and contiguous")
        row, column = divmod(pixel, width)
        start = column * 16 + _least_set_bit(mask)
        end = column * 16 + mask.bit_length() - 1
        region = int(record["region_id"])
        far_side = int(record["far_side"])
        key = (pixel, _least_set_bit(mask), region, far_side)
        if previous_key is not None and key < previous_key:
            raise ValueError("records are not canonically ordered")
        previous_key = key
        if run_count and _record_continues_run(
            runs[run_count - 1],
            row=row,
            start_fine=start,
            region_id=region,
            far_side=far_side,
        ):
            runs[run_count - 1]["record_end"] = np.uint32(record_index + 1)
            runs[run_count - 1]["end_fine"] = np.uint32(end)
            continue
        runs[run_count] = (
            record_index,
            record_index + 1,
            row,
            start,
            end,
            region,
            far_side,
        )
        run_count += 1
    result = np.ascontiguousarray(runs[:run_count])
    result.setflags(write=False)
    return result


def _validate_winner_rasters(
    valid: np.ndarray,
    winner_near_score: np.ndarray,
    winner_region_id: np.ndarray,
) -> tuple[int, int]:
    if (
        not isinstance(valid, np.ndarray)
        or valid.dtype != np.bool_
        or valid.ndim != 2
        or not valid.flags.c_contiguous
    ):
        raise TypeError("valid must be a C-contiguous bool raster")
    if valid.shape[0] <= 0 or valid.shape[1] <= 0 or valid.shape[1] % 16:
        raise ValueError("valid width must be a nonzero multiple of 16")
    if (
        not isinstance(winner_near_score, np.ndarray)
        or winner_near_score.dtype != np.float32
        or winner_near_score.shape != valid.shape
        or not winner_near_score.flags.c_contiguous
    ):
        raise TypeError("winner_near_score must be a matching float32 raster")
    if (
        not isinstance(winner_region_id, np.ndarray)
        or winner_region_id.dtype != np.uint32
        or winner_region_id.shape != valid.shape
        or not winner_region_id.flags.c_contiguous
    ):
        raise TypeError("winner_region_id must be a matching uint32 raster")
    valid_scores = winner_near_score[valid]
    if not np.isfinite(valid_scores).all() or np.any(valid_scores < 0):
        raise ValueError("valid winner near scores must be finite and nonnegative")
    valid_regions = winner_region_id[valid]
    if np.any(valid_regions == 0) or np.any(valid_regions == NO_REGION):
        raise ValueError("valid winners must have positive canonical regions")
    return int(valid.shape[0]), int(valid.shape[1])


def _validate_record_array(records: np.ndarray) -> None:
    if (
        not isinstance(records, np.ndarray)
        or records.dtype != REPAIR_RECORD_DTYPE
        or records.ndim != 1
        or not records.flags.c_contiguous
    ):
        raise TypeError("records must be a C-contiguous repair record vector")


def _validate_bool_invalid_mask(invalid_mask: np.ndarray) -> None:
    if (
        not isinstance(invalid_mask, np.ndarray)
        or invalid_mask.dtype != np.bool_
        or invalid_mask.ndim != 2
        or invalid_mask.shape[0] <= 0
        or invalid_mask.shape[1] <= 0
        or invalid_mask.shape[1] % 16
        or not invalid_mask.flags.c_contiguous
    ):
        raise TypeError("invalid_mask must be a C-contiguous bool fine-grid raster")


def _validate_render_shape(render_shape: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(render_shape, tuple)
        or len(render_shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in render_shape)
    ):
        raise TypeError("render_shape must be an integer (height, width) tuple")
    height, width = render_shape
    if height <= 0 or width <= 0 or height * width > int(np.iinfo(np.uint32).max):
        raise ValueError("render_shape must be positive and fit uint32 pixels")
    return height, width


def _iter_repair_runs(
    valid: np.ndarray,
    near_score: np.ndarray,
    region_id: np.ndarray,
) -> Iterator[_RepairRun]:
    height, fine_width = valid.shape
    for y in range(height):
        x = 0
        while x < fine_width:
            if bool(valid[y, x]):
                x += 1
                continue
            start = x
            while x + 1 < fine_width and not bool(valid[y, x + 1]):
                x += 1
            end = x
            yield from _split_prefill_run(
                y,
                start,
                end,
                fine_width,
                near_score,
                region_id,
            )
            x += 1


def _split_prefill_run(
    row: int,
    start: int,
    end: int,
    fine_width: int,
    near_score: np.ndarray,
    region_id: np.ndarray,
) -> Iterator[_RepairRun]:
    has_left = start > 0
    has_right = end + 1 < fine_width
    if not has_left and not has_right:
        raise QualityRepairNoDonorError("row-wide pre-fill run has no donor")
    if not has_left:
        yield _RepairRun(row, start, end, int(region_id[row, end + 1]), 1)
        return
    if not has_right:
        yield _RepairRun(row, start, end, int(region_id[row, start - 1]), 0)
        return

    left_score = float(near_score[row, start - 1])
    right_score = float(near_score[row, end + 1])
    if left_score < right_score:
        yield _RepairRun(row, start, end, int(region_id[row, start - 1]), 0)
        return
    if right_score < left_score:
        yield _RepairRun(row, start, end, int(region_id[row, end + 1]), 1)
        return

    left_length = (end - start + 2) // 2
    left_end = start + left_length - 1
    yield _RepairRun(row, start, left_end, int(region_id[row, start - 1]), 0)
    if left_end < end:
        yield _RepairRun(row, left_end + 1, end, int(region_id[row, end + 1]), 1)


def _write_run_records(
    records: np.ndarray,
    cursor: int,
    run: _RepairRun,
    width: int,
) -> int:
    segment_start = run.start_fine
    while segment_start <= run.end_fine:
        column = segment_start // 16
        segment_end = min(run.end_fine, column * 16 + 15)
        first_lane = segment_start % 16
        lane_count = segment_end - segment_start + 1
        mask = ((1 << lane_count) - 1) << first_lane
        records[cursor]["pixel_index"] = np.uint32(run.row * width + column)
        records[cursor]["lane_mask"] = np.uint16(mask)
        records[cursor]["region_id"] = np.uint32(run.region_id)
        records[cursor]["far_side"] = np.uint8(run.far_side)
        cursor += 1
        segment_start = segment_end + 1
    return cursor


def _count_prefill_runs(row_valid: np.ndarray) -> int:
    count = 0
    previous_valid = True
    for value in row_valid:
        current_valid = bool(value)
        if previous_valid and not current_valid:
            count += 1
        previous_valid = current_valid
    return count


def _least_set_bit(mask: int) -> int:
    return (mask & -mask).bit_length() - 1


def _is_contiguous_nonzero_mask(mask: int) -> bool:
    if not 0 < mask <= 0xFFFF:
        return False
    shifted = mask >> _least_set_bit(mask)
    return (shifted & (shifted + 1)) == 0


def _record_continues_run(
    run: np.void,
    *,
    row: int,
    start_fine: int,
    region_id: int,
    far_side: int,
) -> bool:
    return (
        int(run["row"]) == row
        and int(run["end_fine"]) + 1 == start_fine
        and int(run["region_id"]) == region_id
        and int(run["far_side"]) == far_side
    )
