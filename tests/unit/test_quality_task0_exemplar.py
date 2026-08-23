"""Task 0 contracts for the bounded deterministic exemplar prototype."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from scripts.quality_task0 import exemplar, local_strip, repair_records


def _api(name: str):
    assert hasattr(exemplar, name), f"missing Task 0 exemplar API: {name}"
    return getattr(exemplar, name)


def test_exemplar_module_is_importable() -> None:
    assert importlib.util.find_spec("scripts.quality_task0.exemplar") is not None


def test_component_core_interiors_partition_bbox_and_halos_never_own_targets() -> None:
    partition = _api("partition_exemplar_cores")
    component_bbox = (10, 410, 20, 800)

    cores = partition(component_bbox, frame_shape=(600, 1000))

    assert len(cores) == 6
    assert [core.origin for core in cores] == [
        (10, 20),
        (10, 404),
        (10, 788),
        (394, 20),
        (394, 404),
        (394, 788),
    ]
    ownership = np.zeros((600, 1000), dtype=np.uint8)
    for core in cores:
        y0, y1, x0, x1 = core.interior_bbox
        ownership[y0:y1, x0:x1] += np.uint8(1)
        wy0, wy1, wx0, wx1 = core.working_bbox
        assert wy1 - wy0 <= 512
        assert wx1 - wx0 <= 512
    y0, y1, x0, x1 = component_bbox
    assert np.all(ownership[y0:y1, x0:x1] == 1)
    ownership[y0:y1, x0:x1] = 0
    assert not np.any(ownership)


def _level(
    working: np.ndarray,
    *,
    donor: np.ndarray,
    target: np.ndarray,
    processed: np.ndarray,
    barrier: np.ndarray,
):
    level_type = _api("ExemplarLevel")
    return level_type(working, donor, target, processed, barrier)


def test_pyramid_parent_flags_and_child_means_use_ties_to_even() -> None:
    downsample = _api("downsample_exemplar_level")
    working = np.asarray(
        [
            [[0, 1, 3], [1, 2, 3]],
            [[1, 2, 3], [0, 1, 3]],
        ],
        dtype=np.uint8,
    )
    all_true = np.ones((2, 2), dtype=np.bool_)
    all_false = np.zeros((2, 2), dtype=np.bool_)

    parent = downsample(
        _level(
            working,
            donor=all_true,
            target=all_false,
            processed=all_true,
            barrier=all_false,
        )
    )

    assert parent.working_bgr.tolist() == [[[0, 2, 3]]]
    assert parent.donor_mask.tolist() == [[True]]
    assert parent.target_mask.tolist() == [[False]]
    assert parent.processed_mask.tolist() == [[True]]
    assert parent.barrier_mask.tolist() == [[False]]

    barrier = all_false.copy()
    barrier[0, 0] = True
    target = all_false.copy()
    target[1, 1] = True
    blocked = downsample(
        _level(
            working,
            donor=all_true,
            target=target,
            processed=all_true,
            barrier=barrier,
        )
    )
    assert blocked.working_bgr.tolist() == [[[0, 0, 0]]]
    assert blocked.barrier_mask.tolist() == [[True]]
    assert blocked.target_mask.tolist() == [[False]]
    assert blocked.donor_mask.tolist() == [[False]]
    assert blocked.processed_mask.tolist() == [[False]]


def test_pyramid_omits_a_coarser_level_below_thirty_two_pixels() -> None:
    build = _api("build_exemplar_pyramid")
    shape = (64, 64)
    full = _level(
        np.zeros((*shape, 3), dtype=np.uint8),
        donor=np.ones(shape, dtype=np.bool_),
        target=np.zeros(shape, dtype=np.bool_),
        processed=np.ones(shape, dtype=np.bool_),
        barrier=np.zeros(shape, dtype=np.bool_),
    )

    levels = build(full)

    assert [level.working_bgr.shape[:2] for level in levels] == [(64, 64), (32, 32)]


def test_frontier_selection_uses_processed_count_then_row_and_column() -> None:
    select = _api("select_frontier_target")
    shape = (9, 10)
    target = np.zeros(shape, dtype=np.bool_)
    target[4, 1] = True
    target[4, 4] = True
    target[4, 5] = True
    processed = np.ones(shape, dtype=np.bool_)
    processed[target] = False
    level = _level(
        np.zeros((*shape, 3), dtype=np.uint8),
        donor=np.zeros(shape, dtype=np.bool_),
        target=target,
        processed=processed,
        barrier=np.zeros(shape, dtype=np.bool_),
    )

    assert select(level) == (4, 5, 47)


def test_frontier_rejects_clipped_or_barrier_crossing_7x7_patches() -> None:
    select = _api("select_frontier_target")
    shape = (9, 9)
    target = np.zeros(shape, dtype=np.bool_)
    target[2, 4] = True
    target[3, 3] = True
    processed = np.ones(shape, dtype=np.bool_)
    processed[target] = False
    barrier = np.zeros(shape, dtype=np.bool_)
    level = _level(
        np.zeros((*shape, 3), dtype=np.uint8),
        donor=np.zeros(shape, dtype=np.bool_),
        target=target,
        processed=processed,
        barrier=barrier,
    )

    assert select(level) == (3, 3, 47)

    barrier[0, 0] = True
    assert select(level) is None


def test_candidate_centres_require_a_complete_original_safe_donor_9x9() -> None:
    enumerate_centres = _api("enumerate_donor_centres")
    donor = np.ones((9, 10), dtype=np.bool_)

    assert enumerate_centres(donor).tolist() == [[4, 4], [4, 5]]

    donor[0, 0] = False
    assert enumerate_centres(donor).tolist() == [[4, 5]]


def test_reflect_101_and_candidate_subsampling_are_frozen() -> None:
    reflect = _api("reflect_101_index")
    subsample = _api("subsample_candidate_indexes")

    assert [reflect(index, 5) for index in range(-3, 8)] == [
        3,
        2,
        1,
        0,
        1,
        2,
        3,
        4,
        3,
        2,
        1,
    ]
    assert subsample(candidate_count=10, evaluation_limit=4).tolist() == [0, 2, 5, 7]
    assert subsample(candidate_count=3, evaluation_limit=5).tolist() == [0, 1, 2]
    assert subsample(candidate_count=3, evaluation_limit=0).size == 0


def _scalar_luma(bgr: np.ndarray) -> int:
    return (29 * int(bgr[0]) + 150 * int(bgr[1]) + 77 * int(bgr[2]) + 128) >> 8


def _scalar_reflect_101(index: int, length: int) -> int:
    if length == 1:
        return 0
    period = 2 * length - 2
    mapped = index % period
    return mapped if mapped < length else period - mapped


def _scalar_sobel(
    working: np.ndarray,
    row: int,
    column: int,
    *,
    reflect: bool,
) -> tuple[int, int]:
    height, width = working.shape[:2]
    luma = np.empty((3, 3), dtype=np.int32)
    for kernel_row, delta_row in enumerate((-1, 0, 1)):
        for kernel_column, delta_column in enumerate((-1, 0, 1)):
            sample_row = row + delta_row
            sample_column = column + delta_column
            if reflect:
                sample_row = _scalar_reflect_101(sample_row, height)
                sample_column = _scalar_reflect_101(sample_column, width)
            luma[kernel_row, kernel_column] = _scalar_luma(working[sample_row, sample_column])
    gx = (
        -int(luma[0, 0])
        + int(luma[0, 2])
        - 2 * int(luma[1, 0])
        + 2 * int(luma[1, 2])
        - int(luma[2, 0])
        + int(luma[2, 2])
    )
    gy = (
        -int(luma[0, 0])
        - 2 * int(luma[0, 1])
        - int(luma[0, 2])
        + int(luma[2, 0])
        + 2 * int(luma[2, 1])
        + int(luma[2, 2])
    )
    return gx, gy


def _scalar_patch_score(
    level,
    *,
    target: tuple[int, int],
    donor: tuple[int, int],
) -> int:
    target_row, target_column = target
    donor_row, donor_column = donor
    height, width = level.target_mask.shape
    gradient_offsets = []
    score = 0
    for delta_row in range(-3, 4):
        for delta_column in range(-3, 4):
            row = target_row + delta_row
            column = target_column + delta_column
            if not level.processed_mask[row, column]:
                continue
            for channel in range(3):
                score += 2 * abs(
                    int(level.working_bgr[row, column, channel])
                    - int(
                        level.working_bgr[
                            donor_row + delta_row,
                            donor_column + delta_column,
                            channel,
                        ]
                    )
                )
            gradient_known = True
            for support_row in (-1, 0, 1):
                mapped_row = _scalar_reflect_101(row + support_row, height)
                for support_column in (-1, 0, 1):
                    mapped_column = _scalar_reflect_101(
                        column + support_column,
                        width,
                    )
                    if (
                        level.barrier_mask[mapped_row, mapped_column]
                        or not level.processed_mask[mapped_row, mapped_column]
                    ):
                        gradient_known = False
            if gradient_known:
                gradient_offsets.append((delta_row, delta_column))
    for delta_row, delta_column in gradient_offsets:
        target_gradient = _scalar_sobel(
            level.working_bgr,
            target_row + delta_row,
            target_column + delta_column,
            reflect=True,
        )
        donor_gradient = _scalar_sobel(
            level.working_bgr,
            donor_row + delta_row,
            donor_column + delta_column,
            reflect=False,
        )
        score += abs(target_gradient[0] - donor_gradient[0])
        score += abs(target_gradient[1] - donor_gradient[1])
    return score


def test_patch_score_uses_reflect_101_only_for_target_gradient_support() -> None:
    score_patch = _api("score_exemplar_patch")
    height, width = 12, 13
    rows, columns = np.indices((height, width), dtype=np.uint16)
    working = np.empty((height, width, 3), dtype=np.uint8)
    working[:, :, 0] = (7 * rows + 3 * columns).astype(np.uint8)
    working[:, :, 1] = (11 * rows + 5 * columns + 17).astype(np.uint8)
    working[:, :, 2] = (13 * rows + 19 * columns + 23).astype(np.uint8)
    target = np.zeros((height, width), dtype=np.bool_)
    target[3, 4] = True
    processed = np.ones((height, width), dtype=np.bool_)
    processed[target] = False
    level = _level(
        working,
        donor=np.ones((height, width), dtype=np.bool_),
        target=target,
        processed=processed,
        barrier=np.zeros((height, width), dtype=np.bool_),
    )

    expected = _scalar_patch_score(level, target=(3, 4), donor=(7, 8))

    assert score_patch(level, target_center=(3, 4), donor_center=(7, 8)) == expected


def _component_fixture():
    height = width = 25
    target_row = target_column = 12
    coverage = np.full((height, width), 16, dtype=np.uint8)
    regions = np.ones((height, width), dtype=np.uint32)
    coverage[target_row, target_column] = np.uint8(0)
    regions[target_row, target_column] = np.uint32(0xFFFFFFFF)
    analysis = local_strip.QualityVisibilityAnalysis(
        coverage,
        regions,
        (
            np.full((height, width), 10, dtype=np.uint8),
            np.full((height, width), 20, dtype=np.uint8),
            np.full((height, width), 30, dtype=np.uint8),
        ),
    )
    safe_donor = np.ones((height, width), dtype=np.bool_)
    safe_donor[11:14, 11:14] = False
    records = np.zeros(1, dtype=repair_records.REPAIR_RECORD_DTYPE)
    records["pixel_index"] = target_row * width + target_column
    records["lane_mask"] = 0xFFFF
    records["region_id"] = 1
    records["fill_bgr"] = [99, 98, 97]
    return analysis, safe_donor, records


def test_full_level_completion_overwrites_provisional_but_not_donor_identity() -> None:
    repair = _api("repair_component_exemplar")
    analysis, safe_donor, records = _component_fixture()
    budgets = local_strip.QualityRepairBudgets()

    result = repair(
        records,
        analysis,
        safe_donor,
        region_id=1,
        render_shape=(25, 25),
        budgets=budgets,
    )

    assert result.records["backend"].tolist() == [2]
    assert result.records["fill_bgr"].tolist() == [[10, 20, 30]]
    assert result.processed_target_count == 1
    assert result.unfinished_target_count == 0
    assert result.patch_iteration_count == 1
    assert result.donor_mask_write_count == 0
    assert budgets.exemplar_eye_evaluations_consumed > 0


def test_unfinished_target_retains_provisional_backend_zero() -> None:
    repair = _api("repair_component_exemplar")
    analysis, safe_donor, records = _component_fixture()
    safe_donor[:] = False

    result = repair(
        records,
        analysis,
        safe_donor,
        region_id=1,
        render_shape=(25, 25),
        budgets=local_strip.QualityRepairBudgets(),
    )

    assert result.records["backend"].tolist() == [0]
    assert result.records["fill_bgr"].tolist() == [[99, 98, 97]]
    assert result.processed_target_count == 0
    assert result.unfinished_target_count == 1


@pytest.mark.parametrize(
    "budget_name",
    [
        "exemplar_core_evaluation_cap",
        "exemplar_component_evaluation_cap",
        "exemplar_eye_evaluation_cap",
    ],
)
def test_zero_budget_terminates_before_candidate_enumeration(
    monkeypatch: pytest.MonkeyPatch,
    budget_name: str,
) -> None:
    repair = _api("repair_component_exemplar")
    analysis, safe_donor, records = _component_fixture()
    budgets = local_strip.QualityRepairBudgets(**{budget_name: 0})

    def fail_if_enumerated(_mask: np.ndarray) -> np.ndarray:
        raise AssertionError("zero budget must stop before candidate enumeration")

    monkeypatch.setattr(exemplar, "enumerate_donor_centres", fail_if_enumerated)

    result = repair(
        records,
        analysis,
        safe_donor,
        region_id=1,
        render_shape=(25, 25),
        budgets=budgets,
    )

    assert result.records["backend"].tolist() == [0]
    assert result.patch_iteration_count == 0
    assert budgets.exemplar_eye_evaluations_consumed == 0


def test_exact_single_evaluation_budget_scores_once_and_completes() -> None:
    repair = _api("repair_component_exemplar")
    analysis, safe_donor, records = _component_fixture()
    budgets = local_strip.QualityRepairBudgets(
        exemplar_core_evaluation_cap=1,
        exemplar_component_evaluation_cap=1,
        exemplar_eye_evaluation_cap=1,
    )

    result = repair(
        records,
        analysis,
        safe_donor,
        region_id=1,
        render_shape=(25, 25),
        budgets=budgets,
    )

    assert result.records["backend"].tolist() == [2]
    assert result.donor_evaluation_count == 1
    assert budgets.exemplar_eye_evaluations_consumed == 1
