from __future__ import annotations

import importlib

import numpy as np

from scripts.quality_task0 import repair_records as prototype_records


def _fallback_fixture():
    local = importlib.import_module("src.depth_surge_3d.rendering.quality.local_strip")
    records_module = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
    height = width = 9
    valid = np.ones((height, width * 16), dtype=np.bool_)
    valid[4, 4 * 16 : 5 * 16] = False
    near = np.ones(valid.shape, dtype=np.float32)
    near[4, 4 * 16 - 1] = np.float32(0.5)
    regions = np.ones(valid.shape, dtype=np.uint32)
    built = records_module.build_repair_records(valid, near, regions)

    coverage = np.full((height, width), 16, dtype=np.uint8)
    coverage[4, 4] = 0
    pure_regions = np.ones((height, width), dtype=np.uint32)
    pure_regions[4, 4] = np.uint32(0xFFFFFFFF)
    blue = np.arange(height * width, dtype=np.uint8).reshape(height, width).copy()
    green = np.full((height, width), 70, dtype=np.uint8)
    red = np.full((height, width), 130, dtype=np.uint8)
    analysis = local.QualityVisibilityAnalysis(
        coverage_count=coverage,
        pure_region_id=pure_regions,
        pure_bgr=(blue, green, red),
    )
    budgets = local.QualityRepairBudgets(
        local_slot_cap=0,
        local_neighbor_sample_cap=0,
    )
    return built, analysis, budgets, valid


def test_production_record_builder_matches_gate_prototype() -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
    valid = np.array([[True, False, False, True] + [True] * 12], dtype=np.bool_)
    near = np.where(valid, np.float32(0.5), np.float32(0.0))
    regions = np.where(valid, np.uint32(1), np.uint32(0xFFFFFFFF))

    expected = prototype_records.build_repair_records(valid, near, regions)
    actual = production.build_repair_records(valid, near, regions)

    assert actual.records.dtype.itemsize == 16
    assert actual.records.tobytes() == expected.records.tobytes()
    assert actual.prefill_run_count == expected.prefill_run_count
    assert actual.repair_run_count == expected.repair_run_count
    assert actual.invalid_lane_count == expected.invalid_lane_count


def test_quality_repair_plan_covers_every_invalid_lane_once() -> None:
    planner = importlib.import_module("src.depth_surge_3d.rendering.quality.planner")
    records_module = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
    built, analysis, budgets, valid = _fallback_fixture()

    plan = planner.plan_quality_repairs(
        analysis,
        records=built.records,
        render_shape=(9, 9),
        local_limit_px=1,
        budgets=budgets,
    )

    assert records_module.validate_record_coverage(plan.records, ~valid) == 16
    assert plan.records["backend"].tolist() == [3]
    assert plan.component_count == 1
    assert plan.record_component_ids.tolist() == [0]
    assert budgets.local_budget_skipped_run_count == 1
    assert plan.fallback_query_count == 1


def test_repair_components_join_vertical_record_adjacency() -> None:
    planner = importlib.import_module("src.depth_surge_3d.rendering.quality.planner")
    records_module = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
    records = np.zeros(2, dtype=records_module.REPAIR_RECORD_DTYPE)
    records[0]["pixel_index"] = 4
    records[0]["lane_mask"] = np.uint16(0x0003)
    records[0]["region_id"] = np.uint32(1)
    records[1]["pixel_index"] = 7
    records[1]["lane_mask"] = np.uint16(0x0001)
    records[1]["region_id"] = np.uint32(1)

    component_ids, count = planner.build_record_component_ids(
        records,
        render_shape=(3, 3),
    )

    assert count == 1
    assert component_ids.tolist() == [0, 0]


def test_quality_repair_plan_accepts_a_fully_covered_eye() -> None:
    planner = importlib.import_module("src.depth_surge_3d.rendering.quality.planner")
    local = importlib.import_module("src.depth_surge_3d.rendering.quality.local_strip")
    records_module = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
    shape = (2, 3)
    analysis = local.QualityVisibilityAnalysis(
        coverage_count=np.full(shape, 16, dtype=np.uint8),
        pure_region_id=np.ones(shape, dtype=np.uint32),
        pure_bgr=tuple(np.zeros(shape, dtype=np.uint8) for _ in range(3)),
    )

    plan = planner.plan_quality_repairs(
        analysis,
        records=np.empty(0, dtype=records_module.REPAIR_RECORD_DTYPE),
        render_shape=shape,
        local_limit_px=1,
        budgets=local.QualityRepairBudgets(),
    )

    assert plan.records.size == 0
    assert plan.component_count == 0
    assert plan.fallback_query_count == 0
