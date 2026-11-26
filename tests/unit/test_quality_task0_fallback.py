"""Shared contracts for full-frame provisional safe-donor fallback."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest


def test_fallback_module_is_importable() -> None:
    assert importlib.util.find_spec("scripts.quality_task0.fallback") is not None


def _fallback_fixture(local_strip, repair_records):
    height = width = 9
    coverage = np.full((height, width), 16, dtype=np.uint8)
    regions = np.ones((height, width), dtype=np.uint32)
    target_row = target_column = 4
    coverage[target_row, target_column] = np.uint8(0)
    regions[target_row, target_column] = np.uint32(0xFFFFFFFF)
    values = np.arange(height * width, dtype=np.uint8).reshape(height, width)
    analysis = local_strip.QualityVisibilityAnalysis(
        coverage,
        regions,
        (
            values.copy(),
            np.ascontiguousarray(values + np.uint8(80)),
            np.ascontiguousarray(values + np.uint8(160)),
        ),
    )
    records = np.zeros(1, dtype=repair_records.REPAIR_RECORD_DTYPE)
    records["pixel_index"] = target_row * width + target_column
    records["lane_mask"] = 0xFFFF
    records["region_id"] = 1
    return analysis, records


def _scalar_nearest_safe_donor(
    analysis,
    *,
    target_row: int,
    target_column: int,
    region: int,
    limit: int,
) -> tuple[int, int]:
    height, width = analysis.coverage_count.shape
    safe_radius = max(1, int(np.floor(height / 1080.0 + 0.5)))
    best: tuple[int, int, int] | None = None
    for row in range(max(0, target_row - limit), min(height, target_row + limit + 1)):
        for column in range(
            max(0, target_column - limit),
            min(width, target_column + limit + 1),
        ):
            safe = True
            for sample_row in range(
                max(0, row - safe_radius),
                min(height, row + safe_radius + 1),
            ):
                for sample_column in range(
                    max(0, column - safe_radius),
                    min(width, column + safe_radius + 1),
                ):
                    if (
                        int(analysis.coverage_count[sample_row, sample_column]) != 16
                        or int(analysis.pure_region_id[sample_row, sample_column]) != region
                    ):
                        safe = False
                        break
                if not safe:
                    break
            if not safe:
                continue
            distance2 = (row - target_row) ** 2 + (column - target_column) ** 2
            if distance2 > limit * limit:
                continue
            key = (distance2, row, column)
            if best is None or key < best:
                best = key
    assert best is not None
    return best[1], best[2]


def test_provisional_fallback_matches_full_frame_scalar_oracle_then_finalizes(
    fallback, local_strip, repair_records
) -> None:
    precompute = fallback.precompute_native_fallbacks
    finalize = fallback.finalize_safe_fallbacks
    analysis, records = _fallback_fixture(local_strip, repair_records)
    budgets = local_strip.QualityRepairBudgets(repair_fallback_visit_cap=100_000)
    limit = max(1, int(np.floor(256 * 9 / 1080.0 + 0.5)))
    donor_row, donor_column = _scalar_nearest_safe_donor(
        analysis,
        target_row=4,
        target_column=4,
        region=1,
        limit=limit,
    )

    provisional = precompute(
        records,
        analysis,
        render_shape=(9, 9),
        region_count=1,
        budgets=budgets,
    )

    expected = [int(analysis.pure_bgr[channel][donor_row, donor_column]) for channel in range(3)]
    assert provisional.records["backend"].tolist() == [0]
    assert provisional.records["fill_bgr"].tolist() == [expected]
    assert provisional.query_count == 1
    assert provisional.indexed_donor_count > 0
    assert provisional.index_bytes <= 48 * 1024 * 1024
    assert provisional.visited_nodes.nbytes <= 16 * 1024 * 1024
    assert provisional.retained_safe_donor_mask_bytes == 0
    assert int(provisional.visited_nodes.sum(dtype=np.uint64)) == (
        budgets.repair_fallback_visits_consumed
    )

    final_records = finalize(provisional.records)
    assert final_records["backend"].tolist() == [3]
    assert final_records["fill_bgr"].tolist() == [expected]


def test_fallback_rejects_partial_proxy_and_cross_region_donors(
    fallback, local_strip, repair_records
) -> None:
    is_safe = fallback.is_safe_donor
    analysis, _ = _fallback_fixture(local_strip, repair_records)
    analysis.coverage_count[0, 0] = np.uint8(15)
    analysis.pure_region_id[8, 8] = np.uint32(2)

    assert not is_safe(analysis, row=0, column=0, region_id=1)
    assert not is_safe(analysis, row=8, column=8, region_id=1)
    assert is_safe(analysis, row=0, column=4, region_id=1)


def test_fallback_fails_when_no_safe_donor_exists_within_the_limit(
    fallback, local_strip, repair_records
) -> None:
    precompute = fallback.precompute_native_fallbacks
    no_donor_error = fallback.QualityRepairNoSafeFallbackError
    analysis, records = _fallback_fixture(local_strip, repair_records)
    analysis.coverage_count[:] = np.uint8(15)

    with pytest.raises(no_donor_error, match="no safe donor"):
        precompute(
            records,
            analysis,
            render_shape=(9, 9),
            region_count=1,
            budgets=local_strip.QualityRepairBudgets(),
        )


def test_fallback_uses_full_pure_core_when_a_thin_region_has_no_safe_interior(
    fallback, local_strip, repair_records
) -> None:
    precompute = fallback.precompute_native_fallbacks
    height = width = 9
    coverage = np.full((height, width), 16, dtype=np.uint8)
    regions = np.full((height, width), 2, dtype=np.uint32)
    regions[:, 4] = np.uint32(1)
    coverage[4, 4] = np.uint8(0)
    regions[4, 4] = np.uint32(0xFFFFFFFF)
    values = np.arange(height * width, dtype=np.uint8).reshape(height, width)
    analysis = local_strip.QualityVisibilityAnalysis(
        coverage,
        regions,
        (
            values.copy(),
            np.ascontiguousarray(values + np.uint8(40)),
            np.ascontiguousarray(values + np.uint8(80)),
        ),
    )
    records = np.zeros(1, dtype=repair_records.REPAIR_RECORD_DTYPE)
    records["pixel_index"] = np.uint32(4 * width + 4)
    records["lane_mask"] = np.uint16(0xFFFF)
    records["region_id"] = np.uint32(1)
    records["far_side"] = np.uint8(0)

    result = precompute(
        records,
        analysis,
        render_shape=(height, width),
        region_count=2,
        budgets=local_strip.QualityRepairBudgets(),
    )

    assert result.records["fill_bgr"].tolist() == [
        [int(analysis.pure_bgr[channel][3, 4]) for channel in range(3)]
    ]
    assert result.indexed_donor_count == 8


def test_seeded_boundary_colour_is_retained_only_when_no_proxy_donor_exists(
    fallback, local_strip, repair_records
) -> None:
    precompute = fallback.precompute_native_fallbacks
    analysis, records = _fallback_fixture(local_strip, repair_records)
    analysis = local_strip.QualityVisibilityAnalysis(
        np.zeros_like(analysis.coverage_count),
        np.full_like(analysis.pure_region_id, np.uint32(0xFFFFFFFF)),
        analysis.pure_bgr,
        anchor_fallbacks_seeded=True,
    )
    records["fill_bgr"] = [7, 8, 9]

    result = precompute(
        records,
        analysis,
        render_shape=(9, 9),
        region_count=1,
        budgets=local_strip.QualityRepairBudgets(),
    )

    assert result.records["fill_bgr"].tolist() == [[7, 8, 9]]
    assert result.anchor_fallback_count == 1


def test_fallback_visit_cap_is_fatal_before_returning_a_current_best(
    fallback, local_strip, repair_records, kd
) -> None:
    precompute = fallback.precompute_native_fallbacks
    analysis, records = _fallback_fixture(local_strip, repair_records)
    original = records.tobytes()
    budgets = local_strip.QualityRepairBudgets(repair_fallback_visit_cap=0)

    with pytest.raises(kd.QualityRepairFallbackQueryBudgetError, match="visit cap"):
        precompute(
            records,
            analysis,
            render_shape=(9, 9),
            region_count=1,
            budgets=budgets,
        )

    assert records.tobytes() == original
    assert budgets.repair_fallback_visits_consumed == 0


def test_fallback_next_node_overflow_discards_an_existing_current_best(
    fallback, local_strip, repair_records, kd
) -> None:
    precompute = fallback.precompute_native_fallbacks
    analysis, records = _fallback_fixture(local_strip, repair_records)
    original = records.tobytes()
    budgets = local_strip.QualityRepairBudgets(repair_fallback_visit_cap=1)

    with pytest.raises(kd.QualityRepairFallbackQueryBudgetError, match="visit cap"):
        precompute(
            records,
            analysis,
            render_shape=(9, 9),
            region_count=1,
            budgets=budgets,
        )

    assert records.tobytes() == original
    assert budgets.repair_fallback_visits_consumed == 1


def test_degenerate_single_column_donors_match_the_scalar_oracle(
    fallback, local_strip, repair_records
) -> None:
    precompute = fallback.precompute_native_fallbacks
    height = width = 9
    coverage = np.zeros((height, width), dtype=np.uint8)
    regions = np.full((height, width), np.uint32(0xFFFFFFFF), dtype=np.uint32)
    coverage[:, 2:5] = np.uint8(16)
    regions[:, 2:5] = np.uint32(1)
    values = np.arange(height * width, dtype=np.uint8).reshape(height, width)
    analysis = local_strip.QualityVisibilityAnalysis(
        coverage,
        regions,
        (
            values.copy(),
            np.ascontiguousarray(values + np.uint8(80)),
            np.ascontiguousarray(values + np.uint8(160)),
        ),
    )
    records = np.zeros(1, dtype=repair_records.REPAIR_RECORD_DTYPE)
    records["pixel_index"] = np.uint32(4 * width + 5)
    records["lane_mask"] = np.uint16(0xFFFF)
    records["region_id"] = np.uint32(1)
    donor = _scalar_nearest_safe_donor(
        analysis,
        target_row=4,
        target_column=5,
        region=1,
        limit=2,
    )

    result = precompute(
        records,
        analysis,
        render_shape=(height, width),
        region_count=1,
        budgets=local_strip.QualityRepairBudgets(),
    )

    assert result.records["fill_bgr"].tolist() == [
        [int(plane[donor]) for plane in analysis.pure_bgr]
    ]


@pytest.mark.slow
def test_4k_twenty_five_percent_fallback_stays_inside_fixed_arenas_and_visit_cap() -> None:
    from scripts.quality_task0 import fallback, local_strip, repair_records

    precompute = fallback.precompute_native_fallbacks
    height, width = 2160, 3840
    hole_columns = np.arange(width) % 16 < 4
    coverage = np.full((height, width), 16, dtype=np.uint8)
    regions = np.ones((height, width), dtype=np.uint32)
    coverage[:, hole_columns] = np.uint8(0)
    regions[:, hole_columns] = np.uint32(0xFFFFFFFF)
    analysis = local_strip.QualityVisibilityAnalysis(
        coverage,
        regions,
        tuple(np.zeros((height, width), dtype=np.uint8) for _ in range(3)),
    )
    columns = np.flatnonzero(hole_columns).astype(np.uint32)
    pixels = (
        np.arange(height, dtype=np.uint32)[:, None] * np.uint32(width) + columns[None, :]
    ).reshape(-1)
    records = np.zeros(pixels.size, dtype=repair_records.REPAIR_RECORD_DTYPE)
    records["pixel_index"] = pixels
    records["lane_mask"] = np.uint16(0xFFFF)
    records["region_id"] = np.uint32(1)
    budgets = local_strip.QualityRepairBudgets()

    result = precompute(
        records,
        analysis,
        render_shape=(height, width),
        region_count=1,
        budgets=budgets,
    )

    assert result.query_count == height * width // 4
    assert result.index_bytes <= fallback.QUALITY_REPAIR_FALLBACK_INDEX_BYTES
    assert result.visited_nodes.nbytes <= fallback.QUALITY_REPAIR_FALLBACK_VISITS_BYTES
    assert float(np.percentile(result.visited_nodes, 95)) <= 4_096
    assert int(result.visited_nodes.sum(dtype=np.uint64)) == (
        budgets.repair_fallback_visits_consumed
    )
    assert budgets.repair_fallback_visits_consumed <= budgets.repair_fallback_visit_cap
