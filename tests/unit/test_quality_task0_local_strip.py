"""Shared contracts for repair records and bounded local-strip planning."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest


def test_repair_record_and_local_strip_modules_are_importable() -> None:
    assert importlib.util.find_spec("scripts.quality_task0.repair_records") is not None
    assert importlib.util.find_spec("scripts.quality_task0.local_strip") is not None
    assert importlib.util.find_spec("scripts.quality_task0.native_repair") is not None


def test_repair_record_dtype_is_frozen_unaligned_and_exactly_sixteen_bytes(repair_records) -> None:
    record_dtype = repair_records.REPAIR_RECORD_DTYPE

    assert record_dtype.itemsize == 16
    assert not record_dtype.isalignedstruct
    assert record_dtype.names == (
        "pixel_index",
        "lane_mask",
        "region_id",
        "fill_bgr",
        "backend",
        "far_side",
        "reserved",
    )
    assert [record_dtype.fields[name][1] for name in record_dtype.names] == [
        0,
        4,
        6,
        10,
        13,
        14,
        15,
    ]


def _winner_rasters(fine_width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.ones((1, fine_width), dtype=np.bool_)
    near_score = np.full((1, fine_width), 0.5, dtype=np.float32)
    region_id = np.ones((1, fine_width), dtype=np.uint32)
    return valid, near_score, region_id


def test_record_builder_splits_runs_and_preserves_far_side_identity(repair_records) -> None:
    build = repair_records.build_repair_records
    valid, near_score, region_id = _winner_rasters(48)

    valid[0, 2:6] = False
    near_score[0, 1] = np.float32(0.2)
    near_score[0, 6] = np.float32(0.8)
    region_id[0, 1] = 7
    region_id[0, 6] = 8

    valid[0, 10:18] = False
    near_score[0, 9] = np.float32(0.4)
    near_score[0, 18] = np.float32(0.4)
    region_id[0, 9] = 3
    region_id[0, 18] = 4

    result = build(valid, near_score, region_id)

    records = result.records
    assert records["pixel_index"].tolist() == [0, 0, 0, 1]
    assert records["lane_mask"].tolist() == [0x003C, 0x3C00, 0xC000, 0x0003]
    assert records["region_id"].tolist() == [7, 3, 4, 4]
    assert records["far_side"].tolist() == [0, 0, 1, 1]
    assert np.all(records["fill_bgr"] == 0)
    assert records["backend"].tolist() == [0, 0, 0, 0]
    assert records["reserved"].tolist() == [0, 0, 0, 0]
    assert result.prefill_run_count == 2
    assert result.repair_run_count == 3
    assert result.invalid_lane_count == 12


def test_record_builder_uses_the_only_boundary_at_each_frame_edge(repair_records) -> None:
    build = repair_records.build_repair_records
    valid, near_score, region_id = _winner_rasters(32)
    valid[0, :3] = False
    valid[0, -5:] = False
    region_id[0, 3] = 9
    region_id[0, -6] = 6

    result = build(valid, near_score, region_id)

    assert result.records["lane_mask"].tolist() == [0x0007, 0xF800]
    assert result.records["region_id"].tolist() == [9, 6]
    assert result.records["far_side"].tolist() == [1, 0]


def test_record_builder_rejects_a_row_wide_no_donor_run(repair_records) -> None:
    build = repair_records.build_repair_records
    no_donor_error = repair_records.QualityRepairNoDonorError
    valid = np.zeros((1, 16), dtype=np.bool_)
    near_score = np.zeros((1, 16), dtype=np.float32)
    region_id = np.zeros((1, 16), dtype=np.uint32)

    with pytest.raises(no_donor_error, match="row-wide"):
        build(valid, near_score, region_id)


def test_million_one_lane_runs_fit_the_fixed_record_arena_without_a_list() -> None:
    from scripts.quality_task0 import native_repair

    build = native_repair.build_native_repair_records
    fine_width = 2_000_000
    valid = np.ones((1, fine_width), dtype=np.bool_)
    valid[:, ::2] = False
    near_score = np.full((1, fine_width), 0.5, dtype=np.float32)
    region_id = np.ones((1, fine_width), dtype=np.uint32)

    result = build(valid, near_score, region_id)

    assert result.prefill_run_count == 1_000_000
    assert result.repair_run_count == 1_000_000
    assert result.records.size == 1_000_000
    assert result.record_bytes == 16_000_000
    assert result.records[0]["far_side"] == 1
    assert np.all(result.records[1:]["far_side"] == 0)


def test_native_record_builder_is_byte_exact_with_the_scalar_builder(
    repair_records, native_repair
) -> None:
    build_native = native_repair.build_native_repair_records
    build_scalar = repair_records.build_repair_records
    valid, near_score, region_id = _winner_rasters(64)
    valid[0, :3] = False
    valid[0, 10:35] = False
    valid[0, 61:] = False
    near_score[0, 9] = np.float32(0.25)
    near_score[0, 35] = np.float32(0.75)
    region_id[0, 3] = 2
    region_id[0, 9] = 3
    region_id[0, 35] = 4
    region_id[0, 60] = 5

    expected = build_scalar(valid, near_score, region_id)
    actual = build_native(valid, near_score, region_id)

    assert actual.records.tobytes() == expected.records.tobytes()
    assert actual.prefill_run_count == expected.prefill_run_count
    assert actual.repair_run_count == expected.repair_run_count
    assert actual.invalid_lane_count == expected.invalid_lane_count


def test_native_record_builder_seeds_each_run_from_its_selected_boundary_colour(
    native_repair,
) -> None:
    build_native = native_repair.build_native_repair_records
    valid, near_score, region_id = _winner_rasters(32)
    valid[0, :16] = False
    winner_bgr = np.zeros((1, 32, 3), dtype=np.float32)
    winner_bgr[0, 16] = [11, 22, 33]

    result = build_native(
        valid,
        near_score,
        region_id,
        winner_bgr=winner_bgr,
    )

    assert result.records["fill_bgr"].tolist() == [[11, 22, 33]]
    assert result.records["far_side"].tolist() == [1]


def test_record_coverage_detects_overlap_missing_bits_and_wrong_popcount(repair_records) -> None:
    validate = repair_records.validate_record_coverage
    coverage_error = repair_records.QualityRepairRecordCoverageError
    dtype = repair_records.REPAIR_RECORD_DTYPE
    invalid = np.zeros((1, 16), dtype=np.bool_)
    invalid[0, 2:6] = True
    records = np.zeros(2, dtype=dtype)
    records["pixel_index"] = 0
    records["region_id"] = 1
    records["lane_mask"] = [0x000C, 0x0030]

    assert validate(records, invalid) == 4

    records[1]["lane_mask"] = np.uint16(0x0018)
    with pytest.raises(coverage_error, match="overlap"):
        validate(records, invalid)

    records[1]["lane_mask"] = np.uint16(0x0010)
    with pytest.raises(coverage_error, match="OR|popcount"):
        validate(records, invalid)


def test_reconstruction_joins_only_consecutive_matching_record_pieces(repair_records) -> None:
    reconstruct = repair_records.reconstruct_repair_runs
    dtype = repair_records.REPAIR_RECORD_DTYPE
    records = np.zeros(4, dtype=dtype)
    records["pixel_index"] = [0, 1, 2, 2]
    records["lane_mask"] = [0xF000, 0xFFFF, 0x0003, 0x0030]
    records["region_id"] = [7, 7, 7, 7]
    records["far_side"] = [0, 0, 0, 0]

    runs = reconstruct(records, render_shape=(1, 3))

    assert runs["record_start"].tolist() == [0, 3]
    assert runs["record_end"].tolist() == [3, 4]
    assert runs["start_fine"].tolist() == [12, 36]
    assert runs["end_fine"].tolist() == [33, 37]


def _analysis(local_strip, height: int, width: int):
    analysis_type = local_strip.QualityVisibilityAnalysis
    coverage = np.full((height, width), 16, dtype=np.uint8)
    regions = np.ones((height, width), dtype=np.uint32)
    blue = np.tile(np.arange(width, dtype=np.uint8), (height, 1))
    green = np.ascontiguousarray(blue + np.uint8(40))
    red = np.ascontiguousarray(blue + np.uint8(80))
    return analysis_type(coverage, regions, (blue, green, red))


def _two_pixel_run_records(repair_records, *, height: int, width: int, far_side: int) -> np.ndarray:
    dtype = repair_records.REPAIR_RECORD_DTYPE
    records = np.zeros(2, dtype=dtype)
    row = height // 2
    if far_side == 0:
        columns = [6, 7]
    else:
        columns = [4, 5]
    records["pixel_index"] = [row * width + column for column in columns]
    records["lane_mask"] = 0xFFFF
    records["region_id"] = 1
    records["far_side"] = far_side
    return records


def _scalar_two_pixel_oracle(
    guide: np.ndarray,
    *,
    far_side: int,
) -> tuple[list[list[int]], int]:
    height, width = guide.shape[:2]
    row = height // 2
    direction = -1 if far_side == 0 else 1
    boundary = 5 if far_side == 0 else 6
    actual_columns = [boundary + (2 - index) * direction for index in range(3)]

    def score(candidate_row: int, candidate_columns: list[int]) -> int:
        bgr_l1 = 0
        actual_luma: list[int] = []
        candidate_luma: list[int] = []
        for actual_column, candidate_column in zip(
            actual_columns,
            candidate_columns,
            strict=True,
        ):
            actual = [int(value) for value in guide[row, actual_column]]
            candidate = [int(value) for value in guide[candidate_row, candidate_column]]
            bgr_l1 += sum(abs(left - right) for left, right in zip(actual, candidate))
            actual_luma.append((29 * actual[0] + 150 * actual[1] + 77 * actual[2] + 128) >> 8)
            candidate_luma.append(
                (29 * candidate[0] + 150 * candidate[1] + 77 * candidate[2] + 128) >> 8
            )
        derivative = sum(
            abs(
                (actual_luma[index + 1] - actual_luma[index])
                - (candidate_luma[index + 1] - candidate_luma[index])
            )
            for index in range(2)
        )
        return 2 * bgr_l1 + derivative

    best: tuple[int, int, list[int]] | None = None
    eligible = 0
    for offset in range(1, 3):
        for row_index, row_delta in enumerate((0, -1, 1, -2, 2)):
            candidate_row = row + row_delta
            start = boundary + direction * offset
            context = [start + (3 - index) * direction for index in range(3)]
            donors = [start - index * direction for index in range(2)]
            if not 0 <= candidate_row < height:
                continue
            if any(not 0 <= column < width for column in context + donors):
                continue
            if any((column - boundary) * direction <= 0 for column in donors):
                continue
            eligible += 1
            candidate = (score(candidate_row, context), (offset - 1) * 5 + row_index, donors)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
    assert best is not None
    donor_columns = best[2]
    best_row_delta = (0, -1, 1, -2, 2)[best[1] % 5]
    target_order_colours = [
        [int(value) for value in guide[row + best_row_delta, column]] for column in donor_columns
    ]
    if far_side == 1:
        target_order_colours.reverse()
    return target_order_colours, eligible


@pytest.mark.parametrize(
    ("far_side", "expected_blue"),
    [(0, [3, 4]), (1, [7, 8])],
)
def test_local_strip_preserves_target_and_donor_direction(
    local_strip,
    repair_records,
    far_side: int,
    expected_blue: list[int],
) -> None:
    plan = local_strip.plan_local_strips
    budgets_type = local_strip.QualityRepairBudgets
    height, width = 34, 12
    analysis = _analysis(local_strip, height, width)
    records = _two_pixel_run_records(
        repair_records,
        height=height,
        width=width,
        far_side=far_side,
    )
    budgets = budgets_type(local_slot_cap=100, local_neighbor_sample_cap=10_000)

    result = plan(
        records,
        analysis,
        render_shape=(height, width),
        local_limit_px=2,
        budgets=budgets,
    )

    assert result.records["backend"].tolist() == [1, 1]
    assert result.records["fill_bgr"][:, 0].tolist() == expected_blue
    assert result.records["fill_bgr"][:, 1].tolist() == [value + 40 for value in expected_blue]
    assert result.records["fill_bgr"][:, 2].tolist() == [value + 80 for value in expected_blue]
    assert budgets.local_slots_consumed == 10
    assert budgets.local_neighbor_samples_consumed == 213
    assert result.evaluated_slot_count == 10
    assert result.local_filled_run_count == 1


@pytest.mark.parametrize("far_side", [0, 1])
def test_local_strip_matches_an_independent_random_scalar_oracle(
    local_strip,
    repair_records,
    far_side: int,
) -> None:
    plan = local_strip.plan_local_strips
    budgets_type = local_strip.QualityRepairBudgets
    analysis_type = local_strip.QualityVisibilityAnalysis
    height, width = 34, 12
    rng = np.random.default_rng(8000 + far_side)
    guide = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    analysis = analysis_type(
        np.full((height, width), 16, dtype=np.uint8),
        np.ones((height, width), dtype=np.uint32),
        tuple(np.ascontiguousarray(guide[:, :, channel]) for channel in range(3)),
    )
    records = _two_pixel_run_records(
        repair_records,
        height=height,
        width=width,
        far_side=far_side,
    )
    expected_colours, expected_eligible = _scalar_two_pixel_oracle(
        guide,
        far_side=far_side,
    )

    result = plan(
        records,
        analysis,
        render_shape=(height, width),
        local_limit_px=2,
        budgets=budgets_type(local_slot_cap=100, local_neighbor_sample_cap=10_000),
    )

    assert result.records["fill_bgr"].tolist() == expected_colours
    assert result.eligible_slot_count == expected_eligible


def test_local_budget_skip_is_whole_run_and_performs_no_sample_reads(
    local_strip, repair_records
) -> None:
    plan = local_strip.plan_local_strips
    budgets_type = local_strip.QualityRepairBudgets
    height, width = 34, 12
    analysis = _analysis(local_strip, height, width)
    records = _two_pixel_run_records(repair_records, height=height, width=width, far_side=0)
    budgets = budgets_type(local_slot_cap=9, local_neighbor_sample_cap=10_000)

    result = plan(
        records,
        analysis,
        render_shape=(height, width),
        local_limit_px=2,
        budgets=budgets,
    )

    assert result.records.tobytes() == records.tobytes()
    assert budgets.local_slots_consumed == 0
    assert budgets.local_neighbor_samples_consumed == 0
    assert budgets.local_budget_skipped_run_count == 1
    assert result.evaluated_slot_count == 0
    assert result.physical_sample_read_count == 0


def test_local_strip_rejects_every_slot_whose_donor_crosses_a_region_edge(
    local_strip, repair_records
) -> None:
    plan = local_strip.plan_local_strips
    budgets_type = local_strip.QualityRepairBudgets
    height, width = 34, 12
    analysis = _analysis(local_strip, height, width)
    analysis.coverage_count[:, 5] = np.uint8(15)
    records = _two_pixel_run_records(repair_records, height=height, width=width, far_side=0)

    result = plan(
        records,
        analysis,
        render_shape=(height, width),
        local_limit_px=2,
        budgets=budgets_type(local_slot_cap=100, local_neighbor_sample_cap=10_000),
    )

    assert result.records["backend"].tolist() == [0, 0]
    assert result.eligible_slot_count == 0
    assert result.unsafe_donor_slot_count == 5
    assert result.evaluated_slot_count == 10


def test_shared_budgets_are_not_reset_between_left_and_right_eye_plans(
    local_strip, repair_records
) -> None:
    plan = local_strip.plan_local_strips
    budgets_type = local_strip.QualityRepairBudgets
    height, width = 34, 12
    analysis = _analysis(local_strip, height, width)
    left = _two_pixel_run_records(repair_records, height=height, width=width, far_side=0)
    right = _two_pixel_run_records(repair_records, height=height, width=width, far_side=1)
    budgets = budgets_type(local_slot_cap=10, local_neighbor_sample_cap=213)

    left_result = plan(
        left,
        analysis,
        render_shape=(height, width),
        local_limit_px=2,
        budgets=budgets,
    )
    right_result = plan(
        right,
        analysis,
        render_shape=(height, width),
        local_limit_px=2,
        budgets=budgets,
    )

    assert left_result.records["backend"].tolist() == [1, 1]
    assert right_result.records["backend"].tolist() == [0, 0]
    assert budgets.local_slots_consumed == 10
    assert budgets.local_neighbor_samples_consumed == 213
    assert budgets.local_budget_skipped_run_count == 1
