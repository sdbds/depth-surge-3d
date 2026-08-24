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
    assert plan.local_budget_skipped_run_count == 1
    assert plan.fallback_query_count == 1


def test_production_record_coverage_uses_native_exact_validator(monkeypatch) -> None:
    records_module = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
    invalid = np.zeros((1, 32), dtype=np.bool_)
    invalid[0, 3:21] = True
    records = np.zeros(2, dtype=records_module.REPAIR_RECORD_DTYPE)
    records["pixel_index"] = [0, 1]
    records["lane_mask"] = [0xFFF8, 0x001F]
    records["region_id"] = 1
    monkeypatch.setattr(
        records_module,
        "_accumulate_record_masks",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("production entered scalar record coverage validation")
        ),
    )

    assert records_module.validate_record_coverage(records, invalid) == 18


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


def test_production_component_builder_uses_native_exact_graph(monkeypatch) -> None:
    planner = importlib.import_module("src.depth_surge_3d.rendering.quality.planner")
    records_module = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
    records = np.zeros(3, dtype=records_module.REPAIR_RECORD_DTYPE)
    records["pixel_index"] = [0, 1, 3]
    records["lane_mask"] = [0x8000, 0x0001, 0x0001]
    records["region_id"] = 1
    monkeypatch.setattr(
        planner,
        "_union_group_pairs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("production entered scalar component graph")
        ),
    )

    component_ids, component_count = planner.build_record_component_ids(
        records,
        render_shape=(2, 2),
    )

    assert component_ids.tolist() == [0, 0, 0]
    assert component_count == 1


def test_native_component_member_order_matches_canonical_counting_sort() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    component_ids = np.array([2, 0, 2, 1, 0, 2], dtype=np.uint32)

    members, offsets = native_kd._require_native().build_component_member_order(
        component_ids,
        3,
    )

    assert offsets.tolist() == [0, 2, 3, 6]
    assert members.tolist() == [1, 4, 3, 0, 2, 5]


def test_native_repair_scatter_expands_lanes_in_canonical_order() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    records_module = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
    records = np.zeros(2, dtype=records_module.REPAIR_RECORD_DTYPE)
    records["pixel_index"] = [0, 1]
    records["lane_mask"] = [0x0003, 0x8000]
    records["region_id"] = 1
    records["fill_bgr"] = [[10, 20, 30], [40, 50, 60]]
    records["backend"] = [1, 3]
    raw = records.view(np.uint8).reshape(records.size, records.dtype.itemsize)

    indexes, colours = native_kd._require_native().expand_repair_scatter(raw, 1, 2)

    assert indexes.tolist() == [0, 1, 31]
    assert colours.tolist() == [
        [10.0, 20.0, 30.0],
        [10.0, 20.0, 30.0],
        [40.0, 50.0, 60.0],
    ]


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


def test_native_exemplar_donor_selection_matches_scalar_score_oracle() -> None:
    exemplar = importlib.import_module("src.depth_surge_3d.rendering.quality.exemplar")
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    rows, columns = np.indices((15, 15), dtype=np.uint16)
    working = np.empty((15, 15, 3), dtype=np.uint8)
    working[..., 0] = ((rows * 17 + columns * 3) % 251).astype(np.uint8)
    working[..., 1] = ((rows * 5 + columns * 19) % 253).astype(np.uint8)
    working[..., 2] = ((rows * 11 + columns * 7) % 255).astype(np.uint8)
    processed = np.ones((15, 15), dtype=np.bool_)
    processed[7, 7] = False
    level = exemplar.ExemplarLevel(
        np.ascontiguousarray(working),
        np.ones((15, 15), dtype=np.bool_),
        np.zeros((15, 15), dtype=np.bool_),
        processed,
        np.zeros((15, 15), dtype=np.bool_),
    )
    candidates = np.array([[4, 4], [4, 10], [10, 4], [10, 10]], dtype=np.uint32)
    selected = np.array([0, 1, 2, 3], dtype=np.uint32)
    expected = min(
        (
            exemplar.score_exemplar_patch(
                level,
                target_center=(7, 7),
                donor_center=(int(row), int(column)),
            ),
            int(row),
            int(column),
        )
        for row, column in candidates
    )

    actual = native_kd._require_native().select_exemplar_donor(
        level.working_bgr,
        level.processed_mask,
        level.barrier_mask,
        7,
        7,
        candidates,
        selected,
    )
    gradient_x, gradient_y = native_kd._require_native().build_exemplar_sobel_planes(
        level.working_bgr
    )
    actual_cached = native_kd._require_native().select_exemplar_donor(
        level.working_bgr,
        level.processed_mask,
        level.barrier_mask,
        7,
        7,
        candidates,
        selected,
        gradient_x,
        gradient_y,
    )

    assert tuple(actual) == expected
    assert tuple(actual_cached) == expected


def test_native_exemplar_sobel_planes_match_scalar_oracle() -> None:
    exemplar = importlib.import_module("src.depth_surge_3d.rendering.quality.exemplar")
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    rows, columns = np.indices((15, 17), dtype=np.uint16)
    working = np.empty((15, 17, 3), dtype=np.uint8)
    working[..., 0] = ((rows * 31 + columns * 7) % 251).astype(np.uint8)
    working[..., 1] = ((rows * 13 + columns * 23) % 253).astype(np.uint8)
    working[..., 2] = ((rows * 19 + columns * 11) % 255).astype(np.uint8)

    gradient_x, gradient_y = native_kd._require_native().build_exemplar_sobel_planes(
        np.ascontiguousarray(working)
    )

    assert gradient_x.dtype == np.int16
    assert gradient_y.dtype == np.int16
    assert gradient_x.shape == working.shape[:2]
    assert gradient_y.shape == working.shape[:2]
    for row in range(1, working.shape[0] - 1):
        for column in range(1, working.shape[1] - 1):
            expected_x, expected_y = exemplar._direct_sobel(working, row, column)
            assert int(gradient_x[row, column]) == expected_x
            assert int(gradient_y[row, column]) == expected_y


def test_production_exemplar_does_not_enter_scalar_candidate_scoring(monkeypatch) -> None:
    exemplar = importlib.import_module("src.depth_surge_3d.rendering.quality.exemplar")
    local = importlib.import_module("src.depth_surge_3d.rendering.quality.local_strip")
    records_module = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
    height = width = 25
    coverage = np.full((height, width), 16, dtype=np.uint8)
    regions = np.ones((height, width), dtype=np.uint32)
    coverage[12, 12] = np.uint8(0)
    regions[12, 12] = np.uint32(0xFFFFFFFF)
    analysis = local.QualityVisibilityAnalysis(
        coverage,
        regions,
        tuple(np.full((height, width), value, dtype=np.uint8) for value in (10, 20, 30)),
    )
    safe_donor = np.ones((height, width), dtype=np.bool_)
    safe_donor[11:14, 11:14] = False
    records = np.zeros(1, dtype=records_module.REPAIR_RECORD_DTYPE)
    records["pixel_index"] = 12 * width + 12
    records["lane_mask"] = np.uint16(0xFFFF)
    records["region_id"] = np.uint32(1)
    records["fill_bgr"] = [99, 98, 97]
    monkeypatch.setattr(
        exemplar,
        "_score_patch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("production entered scalar exemplar scoring")
        ),
    )

    result = exemplar.repair_component_exemplar(
        records,
        analysis,
        safe_donor,
        region_id=1,
        render_shape=(height, width),
        budgets=local.QualityRepairBudgets(),
    )

    assert result.records["backend"].tolist() == [2]


def test_production_exemplar_reuses_one_native_sobel_pair_per_level(monkeypatch) -> None:
    exemplar = importlib.import_module("src.depth_surge_3d.rendering.quality.exemplar")
    local = importlib.import_module("src.depth_surge_3d.rendering.quality.local_strip")
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    records_module = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
    native = native_kd._require_native()
    built_planes: list[tuple[np.ndarray, np.ndarray]] = []
    selection_count = 0

    class RecordingNative:
        @staticmethod
        def build_exemplar_sobel_planes(working_bgr):
            planes = native.build_exemplar_sobel_planes(working_bgr)
            built_planes.append(planes)
            return planes

        @staticmethod
        def select_exemplar_frontier(target_mask, processed_mask, barrier_mask):
            return native.select_exemplar_frontier(
                target_mask,
                processed_mask,
                barrier_mask,
            )

        @staticmethod
        def select_exemplar_donor(
            working_bgr,
            processed_mask,
            barrier_mask,
            target_row,
            target_column,
            candidates,
            selected_indexes,
            donor_gradient_x,
            donor_gradient_y,
        ):
            nonlocal selection_count
            selection_count += 1
            assert any(
                donor_gradient_x is planes[0] and donor_gradient_y is planes[1]
                for planes in built_planes
            )
            return native.select_exemplar_donor(
                working_bgr,
                processed_mask,
                barrier_mask,
                target_row,
                target_column,
                candidates,
                selected_indexes,
                donor_gradient_x,
                donor_gradient_y,
            )

    monkeypatch.setattr(native_kd, "_require_native", lambda: RecordingNative())
    height = width = 33
    coverage = np.full((height, width), 16, dtype=np.uint8)
    regions = np.ones((height, width), dtype=np.uint32)
    coverage[16, 16] = np.uint8(0)
    regions[16, 16] = np.uint32(0xFFFFFFFF)
    analysis = local.QualityVisibilityAnalysis(
        coverage,
        regions,
        tuple(np.full((height, width), value, dtype=np.uint8) for value in (10, 20, 30)),
    )
    safe_donor = np.ones((height, width), dtype=np.bool_)
    safe_donor[15:18, 15:18] = False
    records = np.zeros(1, dtype=records_module.REPAIR_RECORD_DTYPE)
    records["pixel_index"] = 16 * width + 16
    records["lane_mask"] = np.uint16(0xFFFF)
    records["region_id"] = np.uint32(1)
    records["fill_bgr"] = [99, 98, 97]

    result = exemplar.repair_component_exemplar(
        records,
        analysis,
        safe_donor,
        region_id=1,
        render_shape=(height, width),
        budgets=local.QualityRepairBudgets(),
    )

    assert result.records["backend"].tolist() == [2]
    assert selection_count > 0
    assert len(built_planes) <= 3


def test_production_exemplar_downsample_is_vectorized_and_oracle_exact(monkeypatch) -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.exemplar")
    prototype = importlib.import_module("scripts.quality_task0.exemplar")
    rng = np.random.default_rng(7301)
    working = rng.integers(0, 256, size=(17, 19, 3), dtype=np.uint8)
    donor = rng.integers(0, 2, size=(17, 19), dtype=np.uint8).astype(np.bool_)
    target = rng.integers(0, 2, size=(17, 19), dtype=np.uint8).astype(np.bool_)
    processed = rng.integers(0, 2, size=(17, 19), dtype=np.uint8).astype(np.bool_)
    barrier = rng.integers(0, 5, size=(17, 19), dtype=np.uint8) == 0
    expected = prototype.downsample_exemplar_level(
        prototype.ExemplarLevel(working, donor, target, processed, barrier)
    )
    monkeypatch.setattr(
        production,
        "_child_coordinates",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("production entered scalar pyramid reduction")
        ),
    )

    actual = production.downsample_exemplar_level(
        production.ExemplarLevel(working, donor, target, processed, barrier)
    )

    for field in (
        "working_bgr",
        "donor_mask",
        "target_mask",
        "processed_mask",
        "barrier_mask",
    ):
        assert np.array_equal(getattr(actual, field), getattr(expected, field))


def test_production_local_strip_uses_native_exact_planner(monkeypatch) -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.local_strip")
    prototype = importlib.import_module("scripts.quality_task0.local_strip")
    built, analysis, _, _ = _fallback_fixture()
    expected_budgets = prototype.QualityRepairBudgets()
    expected_analysis = prototype.QualityVisibilityAnalysis(
        analysis.coverage_count,
        analysis.pure_region_id,
        analysis.pure_bgr,
    )
    expected = prototype.plan_local_strips(
        built.records,
        expected_analysis,
        render_shape=(9, 9),
        local_limit_px=1,
        budgets=expected_budgets,
    )
    monkeypatch.setattr(
        production,
        "_plan_one_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("production entered scalar local-strip planning")
        ),
    )
    actual_budgets = production.QualityRepairBudgets()

    actual = production.plan_local_strips(
        built.records,
        analysis,
        render_shape=(9, 9),
        local_limit_px=1,
        budgets=actual_budgets,
    )

    assert actual.records.tobytes() == expected.records.tobytes()
    for field in (
        "evaluated_slot_count",
        "physical_sample_read_count",
        "local_filled_run_count",
        "eligible_slot_count",
        "unsafe_donor_slot_count",
    ):
        assert getattr(actual, field) == getattr(expected, field)
    assert actual_budgets.local_slots_consumed == expected_budgets.local_slots_consumed
    assert (
        actual_budgets.local_neighbor_samples_consumed
        == expected_budgets.local_neighbor_samples_consumed
    )


def test_production_local_strip_uses_native_run_reconstruction(monkeypatch) -> None:
    production = importlib.import_module("src.depth_surge_3d.rendering.quality.local_strip")
    built, analysis, budgets, _ = _fallback_fixture()
    monkeypatch.setattr(
        production,
        "reconstruct_repair_runs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("production entered scalar repair-run reconstruction")
        ),
        raising=False,
    )

    result = production.plan_local_strips(
        built.records,
        analysis,
        render_shape=(9, 9),
        local_limit_px=1,
        budgets=budgets,
    )

    assert result.records.size == built.records.size


def test_native_repair_run_reconstruction_matches_scalar_oracle() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    records_module = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
    records = np.zeros(5, dtype=records_module.REPAIR_RECORD_DTYPE)
    records["pixel_index"] = [0, 1, 2, 2, 5]
    records["lane_mask"] = [0xF000, 0xFFFF, 0x0003, 0x0030, 0x0007]
    records["region_id"] = [7, 7, 7, 7, 9]
    records["far_side"] = [0, 0, 0, 0, 1]
    expected = records_module.reconstruct_repair_runs(records, render_shape=(2, 3))
    raw = records.view(np.uint8).reshape(records.size, records.dtype.itemsize)

    actual_raw = native_kd._require_native().reconstruct_repair_runs(raw, 2, 3)
    actual = actual_raw.reshape(-1).view(records_module.REPAIR_RUN_DTYPE)

    assert actual.tobytes() == expected.tobytes()


def test_planner_skips_exemplar_when_the_region_has_no_strict_safe_donor(
    monkeypatch,
) -> None:
    planner = importlib.import_module("src.depth_surge_3d.rendering.quality.planner")
    local = importlib.import_module("src.depth_surge_3d.rendering.quality.local_strip")
    records_module = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
    height = width = 9
    coverage = np.full((height, width), 16, dtype=np.uint8)
    regions = np.full((height, width), 2, dtype=np.uint32)
    regions[:, 4] = np.uint32(1)
    coverage[4, 4] = np.uint8(0)
    regions[4, 4] = np.uint32(0xFFFFFFFF)
    analysis = local.QualityVisibilityAnalysis(
        coverage,
        regions,
        tuple(np.full((height, width), value, dtype=np.uint8) for value in (9, 19, 29)),
    )
    records = np.zeros(1, dtype=records_module.REPAIR_RECORD_DTYPE)
    records["pixel_index"] = np.uint32(4 * width + 4)
    records["lane_mask"] = np.uint16(0xFFFF)
    records["region_id"] = np.uint32(1)
    monkeypatch.setattr(
        planner.exemplar,
        "repair_component_exemplar",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("donorless region entered exemplar construction")
        ),
    )

    result = planner.plan_quality_repairs(
        analysis,
        records=records,
        render_shape=(height, width),
        local_limit_px=1,
        budgets=local.QualityRepairBudgets(),
    )

    assert result.records["backend"].tolist() == [3]
    assert result.exemplar_donor_evaluation_count == 0


def test_native_safe_donor_mask_matches_the_strict_index_members() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    native = native_kd._require_native()
    coverage = np.full((11, 13), 16, dtype=np.uint8)
    regions = np.ones((11, 13), dtype=np.uint32)
    regions[:, 7:] = np.uint32(2)
    coverage[5, 3] = np.uint8(15)
    coverage[7, 9] = np.uint8(0)
    referenced = np.array([False, True, True], dtype=np.bool_)
    members, offsets = native.build_safe_donor_index(
        coverage,
        regions,
        referenced,
    )
    expected_mask = np.zeros(coverage.shape, dtype=np.bool_)
    expected_mask.reshape(-1)[members] = True
    expected_regions = np.zeros(referenced.shape, dtype=np.bool_)
    expected_regions[1:] = offsets[1:] != offsets[:-1]

    actual_mask, actual_regions = native.build_safe_donor_mask(
        coverage,
        regions,
        referenced,
    )

    assert np.array_equal(actual_mask, expected_mask)
    assert np.array_equal(actual_regions, expected_regions)


def test_native_exemplar_candidate_mask_matches_the_complete_support_oracle() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    generator = np.random.default_rng(2026082402)
    shape = (21, 27)
    safe = np.ascontiguousarray(generator.random(shape) >= 0.08)
    regions = np.ones(shape, dtype=np.uint32)
    regions[:, 9:18] = np.uint32(2)
    regions[:, 18:] = np.uint32(3)
    regions[10, 13] = np.uint32(0xFFFFFFFF)
    expected = np.zeros(shape, dtype=np.bool_)
    for row in range(4, shape[0] - 4):
        for column in range(4, shape[1] - 4):
            region = regions[row, column]
            support = np.s_[row - 4 : row + 5, column - 4 : column + 5]
            expected[row, column] = bool(
                region not in (0, 0xFFFFFFFF)
                and np.all(safe[support])
                and np.all(regions[support] == region)
            )

    actual = native_kd._require_native().build_exemplar_candidate_mask(safe, regions)

    assert np.array_equal(actual, expected)


def test_native_exemplar_frontier_matches_the_scalar_oracle() -> None:
    native_kd = importlib.import_module("src.depth_surge_3d.rendering.quality.native_kd")
    prototype = importlib.import_module("scripts.quality_task0.exemplar")
    generator = np.random.default_rng(2026082401)
    shape = (19, 23)
    target = np.ascontiguousarray(generator.random(shape) < 0.35)
    processed = np.ascontiguousarray(generator.random(shape) < 0.55)
    barrier = np.ascontiguousarray(generator.random(shape) < 0.04)
    processed[target] = False
    level = prototype.ExemplarLevel(
        np.zeros((*shape, 3), dtype=np.uint8),
        np.zeros(shape, dtype=np.bool_),
        target,
        processed,
        barrier,
    )
    expected = prototype.select_frontier_target(level)

    actual = native_kd._require_native().select_exemplar_frontier(
        target,
        processed,
        barrier,
    )

    assert (None if actual is None else tuple(int(value) for value in actual)) == expected


def test_planner_skips_exemplar_when_no_complete_donor_patch_exists(
    monkeypatch,
) -> None:
    planner = importlib.import_module("src.depth_surge_3d.rendering.quality.planner")
    local = importlib.import_module("src.depth_surge_3d.rendering.quality.local_strip")
    records_module = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
    height = width = 25
    coverage = np.full((height, width), 16, dtype=np.uint8)
    regions = np.full((height, width), 2, dtype=np.uint32)
    regions[:, 10:15] = np.uint32(1)
    coverage[12, 12] = np.uint8(0)
    regions[12, 12] = np.uint32(0xFFFFFFFF)
    analysis = local.QualityVisibilityAnalysis(
        coverage,
        regions,
        tuple(np.full((height, width), value, dtype=np.uint8) for value in (9, 19, 29)),
    )
    records = np.zeros(1, dtype=records_module.REPAIR_RECORD_DTYPE)
    records["pixel_index"] = np.uint32(12 * width + 12)
    records["lane_mask"] = np.uint16(0xFFFF)
    records["region_id"] = np.uint32(1)
    monkeypatch.setattr(
        planner.exemplar,
        "repair_component_exemplar",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("candidate-free region entered exemplar construction")
        ),
    )

    result = planner.plan_quality_repairs(
        analysis,
        records=records,
        render_shape=(height, width),
        local_limit_px=1,
        budgets=local.QualityRepairBudgets(),
    )

    assert result.records["backend"].tolist() == [3]
    assert result.exemplar_donor_evaluation_count == 0


def test_planner_skips_exemplar_when_no_pyramid_level_can_score(monkeypatch) -> None:
    planner = importlib.import_module("src.depth_surge_3d.rendering.quality.planner")
    local = importlib.import_module("src.depth_surge_3d.rendering.quality.local_strip")
    records_module = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
    height = width = 25
    coverage = np.full((height, width), 16, dtype=np.uint8)
    regions = np.full((height, width), 2, dtype=np.uint32)
    regions[:, :12] = np.uint32(1)
    coverage[12, 11] = np.uint8(0)
    regions[12, 11] = np.uint32(0xFFFFFFFF)
    analysis = local.QualityVisibilityAnalysis(
        coverage,
        regions,
        tuple(np.full((height, width), value, dtype=np.uint8) for value in (9, 19, 29)),
    )
    records = np.zeros(1, dtype=records_module.REPAIR_RECORD_DTYPE)
    records["pixel_index"] = np.uint32(12 * width + 11)
    records["lane_mask"] = np.uint16(0xFFFF)
    records["region_id"] = np.uint32(1)
    records["far_side"] = np.uint8(1)
    monkeypatch.setattr(
        planner.exemplar,
        "repair_component_exemplar",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("non-scoring pyramid entered exemplar construction")
        ),
    )

    result = planner.plan_quality_repairs(
        analysis,
        records=records,
        render_shape=(height, width),
        local_limit_px=1,
        budgets=local.QualityRepairBudgets(),
    )

    assert result.records["backend"].tolist() == [3]
    assert result.exemplar_donor_evaluation_count == 0


def test_planner_uses_stable_fallback_for_a_micro_target_component(monkeypatch) -> None:
    planner = importlib.import_module("src.depth_surge_3d.rendering.quality.planner")
    local = importlib.import_module("src.depth_surge_3d.rendering.quality.local_strip")
    records_module = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
    height = width = 25
    coverage = np.full((height, width), 16, dtype=np.uint8)
    regions = np.ones((height, width), dtype=np.uint32)
    coverage[9:15, 8:16] = np.uint8(0)
    regions[9:15, 8:16] = np.uint32(0xFFFFFFFF)
    analysis = local.QualityVisibilityAnalysis(
        coverage,
        regions,
        tuple(np.full((height, width), value, dtype=np.uint8) for value in (9, 19, 29)),
    )
    target_pixels = [row * width + column for row in range(9, 15) for column in range(8, 16)]
    records = np.zeros(len(target_pixels), dtype=records_module.REPAIR_RECORD_DTYPE)
    records["pixel_index"] = np.asarray(target_pixels, dtype=np.uint32)
    records["lane_mask"] = np.uint16(0xFFFF)
    records["region_id"] = np.uint32(1)
    monkeypatch.setattr(
        planner.exemplar,
        "repair_component_exemplar",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("micro target component entered exemplar synthesis")
        ),
    )

    result = planner.plan_quality_repairs(
        analysis,
        records=records,
        render_shape=(height, width),
        local_limit_px=1,
        budgets=local.QualityRepairBudgets(local_slot_cap=0),
    )

    assert result.records["backend"].tolist() == [3] * 48
    assert result.exemplar_donor_evaluation_count == 0
