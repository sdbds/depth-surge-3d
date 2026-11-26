from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def fixed_arenas():
    arena = importlib.import_module("src.depth_surge_3d.rendering.quality.arena")
    return arena.QualityFixedArenas.allocate()


def _single_invalid_lane_band():
    forward_splat = importlib.import_module("src.depth_surge_3d.rendering.forward_splat")
    visibility = importlib.import_module("src.depth_surge_3d.rendering.quality.visibility")
    valid = np.ones((1, 16), dtype=np.bool_)
    valid[0, 5] = False
    winner_sources = np.zeros((1, 16), dtype=np.int64)
    winner_sources[~valid] = np.int64(0xFFFFFFFF)
    return visibility.QualitySplatBand(
        splat=forward_splat.SubpixelSplatResult(
            colour=torch.zeros((1, 16, 3), dtype=torch.float32),
            disparity=torch.from_numpy(valid.astype(np.float32)),
            valid=torch.from_numpy(valid),
        ),
        winner_source_index=torch.from_numpy(winner_sources),
    )


def _single_pixel_analysis():
    local_strip = importlib.import_module("src.depth_surge_3d.rendering.quality.local_strip")
    return local_strip.QualityVisibilityAnalysis(
        coverage_count=np.empty((1, 1), dtype=np.uint8),
        pure_region_id=np.empty((1, 1), dtype=np.uint32),
        pure_bgr=tuple(np.empty((1, 1), dtype=np.uint8) for _ in range(3)),
        anchor_fallbacks_seeded=True,
    )


def _raw_records(records: np.ndarray) -> np.ndarray:
    return records.view(np.uint8).reshape(records.size, records.dtype.itemsize)


def _random_final_records(
    generator: np.random.Generator,
    *,
    pixel_count: int,
    record_count: int,
) -> np.ndarray:
    repair_records = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
    records = np.zeros(record_count, dtype=repair_records.REPAIR_RECORD_DTYPE)
    if record_count == 0:
        return records
    records["pixel_index"] = np.sort(
        generator.choice(pixel_count, size=record_count, replace=False).astype(np.uint32)
    )
    for record in records:
        lane_start = int(generator.integers(0, 16))
        lane_end = int(generator.integers(lane_start + 1, 17))
        record["lane_mask"] = np.uint16(((1 << (lane_end - lane_start)) - 1) << lane_start)
    records["fill_bgr"] = generator.integers(0, 256, size=(record_count, 3), dtype=np.uint8)
    records["backend"] = generator.integers(1, 4, size=record_count, dtype=np.uint8)
    return records


def _downsample_exemplar_oracle(
    donor: np.ndarray,
    target: np.ndarray,
    processed: np.ndarray,
    barrier: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parent_shape = ((donor.shape[0] + 1) // 2, (donor.shape[1] + 1) // 2)
    parent_donor = np.ones(parent_shape, dtype=np.bool_)
    parent_target = np.zeros(parent_shape, dtype=np.bool_)
    parent_processed = np.ones(parent_shape, dtype=np.bool_)
    parent_barrier = np.zeros(parent_shape, dtype=np.bool_)
    for row in range(parent_shape[0]):
        for column in range(parent_shape[1]):
            block = np.s_[
                2 * row : min(2 * row + 2, donor.shape[0]),
                2 * column : min(2 * column + 2, donor.shape[1]),
            ]
            block_target = bool(np.any(target[block]))
            block_barrier = bool(np.any(barrier[block]))
            parent_barrier[row, column] = block_barrier
            parent_target[row, column] = not block_barrier and block_target
            parent_donor[row, column] = not block_barrier and bool(np.all(donor[block]))
            parent_processed[row, column] = (
                not block_target and not block_barrier and bool(np.all(processed[block]))
            )
    return parent_donor, parent_target, parent_processed, parent_barrier


def _exemplar_level_may_score_oracle(
    donor: np.ndarray,
    target: np.ndarray,
    processed: np.ndarray,
    barrier: np.ndarray,
) -> bool:
    if not any(
        np.all(donor[row - 4 : row + 5, column - 4 : column + 5])
        for row in range(4, donor.shape[0] - 4)
        for column in range(4, donor.shape[1] - 4)
    ):
        return False
    for row in range(3, target.shape[0] - 3):
        for column in range(3, target.shape[1] - 3):
            if not target[row, column] or processed[row, column]:
                continue
            neighbours = (
                (row - 1, column),
                (row, column - 1),
                (row, column + 1),
                (row + 1, column),
            )
            if not any(processed[y, x] and not barrier[y, x] for y, x in neighbours):
                continue
            patch_barrier = barrier[row - 3 : row + 4, column - 3 : column + 4]
            patch_processed = processed[row - 3 : row + 4, column - 3 : column + 4]
            if not np.any(patch_barrier) and np.count_nonzero(patch_processed) >= 8:
                return True
    return False


def _component_may_score_exemplar_oracle(
    coverage: np.ndarray,
    regions: np.ndarray,
    safe: np.ndarray,
    targets: np.ndarray,
    region: int,
) -> bool:
    height, width = coverage.shape
    rows = targets // np.uint32(width)
    columns = targets % np.uint32(width)
    if int(rows.max()) - int(rows.min()) + 1 > 384:
        return True
    if int(columns.max()) - int(columns.min()) + 1 > 384:
        return True
    y0 = max(0, int(rows.min()) - 64)
    y1 = min(height, int(rows.max()) + 65)
    x0 = max(0, int(columns.min()) - 64)
    x1 = min(width, int(columns.max()) + 65)
    same_proxy = (coverage[y0:y1, x0:x1] > 0) & (regions[y0:y1, x0:x1] == np.uint32(region))
    donor = same_proxy & safe[y0:y1, x0:x1]
    target = np.zeros_like(donor)
    processed = same_proxy.copy()
    barrier = ~same_proxy
    local_rows = rows.astype(np.int64) - y0
    local_columns = columns.astype(np.int64) - x0
    donor[local_rows, local_columns] = False
    target[local_rows, local_columns] = True
    processed[local_rows, local_columns] = False
    barrier[local_rows, local_columns] = False
    levels = [(donor, target, processed, barrier)]
    while len(levels) < 3:
        next_level = _downsample_exemplar_oracle(*levels[-1])
        if min(next_level[0].shape) < 32:
            break
        levels.append(next_level)
    return any(_exemplar_level_may_score_oracle(*level) for level in levels)


@pytest.mark.parametrize("failed_allocation", [1, 2, 3])
def test_fixed_arena_preallocation_normalizes_each_host_oom(
    monkeypatch: pytest.MonkeyPatch,
    failed_allocation: int,
) -> None:
    arena = importlib.import_module("src.depth_surge_3d.rendering.quality.arena")
    real_empty = arena.np.empty
    calls = 0

    def fail_selected_allocation(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failed_allocation:
            raise MemoryError("injected fixed-arena allocation failure")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(arena.np, "empty", fail_selected_allocation)

    with pytest.raises(
        arena.QualityArenaAllocationError,
        match="preallocation failed before Pass A",
    ):
        arena.QualityFixedArenas.allocate()

    assert calls == failed_allocation


def test_fixed_graph_matches_legacy_component_and_member_order_oracle(
    fixed_arenas,
) -> None:
    planner = importlib.import_module("src.depth_surge_3d.rendering.quality.planner")
    repair_records = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
    generator = np.random.default_rng(2026082501)

    for _case in range(32):
        height = int(generator.integers(1, 10))
        width = int(generator.integers(1, 10))
        valid = np.ascontiguousarray(generator.random((height, width * 16)) > 0.22)
        near = np.ascontiguousarray(generator.random(valid.shape), dtype=np.float32)
        regions = np.ascontiguousarray(generator.integers(1, 5, size=valid.shape, dtype=np.uint32))
        built = repair_records.build_repair_records(valid, near, regions)

        expected_ids, expected_count = planner.build_record_component_ids(
            built.records,
            render_shape=(height, width),
        )
        expected_members, _expected_offsets = planner._component_member_order(
            expected_ids,
            expected_count,
        )
        actual_ids, actual_members, actual_count = planner._fixed_graph_components(
            built.records,
            render_shape=(height, width),
            arenas=fixed_arenas,
        )

        assert actual_count == expected_count
        assert actual_ids.tobytes() == expected_ids.tobytes()
        assert actual_members.tobytes() == expected_members.tobytes()


def test_fixed_graph_workspace_boundary_uses_budget_error(fixed_arenas) -> None:
    planner = importlib.import_module("src.depth_surge_3d.rendering.quality.planner")
    repair_records = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
    records = np.empty(0, dtype=repair_records.REPAIR_RECORD_DTYPE)
    workspace_width = (12 * 1024 * 1024) // (4 * np.dtype(np.int32).itemsize)

    component_ids, members, component_count = planner._fixed_graph_components(
        records,
        render_shape=(1, workspace_width),
        arenas=fixed_arenas,
    )

    assert component_ids.size == 0
    assert members.size == 0
    assert component_count == 0
    with pytest.raises(repair_records.QualityRepairBudgetError):
        planner._fixed_graph_components(
            records,
            render_shape=(1, workspace_width + 1),
            arenas=fixed_arenas,
        )


def test_record_arena_exposes_the_exact_canonical_cap(fixed_arenas) -> None:
    repair_records = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")

    records = fixed_arenas.records()

    assert records.size == repair_records.QUALITY_REPAIR_RECORD_CAP
    assert records.nbytes == 64 * 1024 * 1024


def test_first_band_record_capacity_overflow_uses_budget_error() -> None:
    renderer = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    repair_records = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
    output = np.empty(0, dtype=repair_records.REPAIR_RECORD_DTYPE)

    with pytest.raises(repair_records.QualityRepairBudgetError):
        renderer._write_analysis_band_native(
            _single_pixel_analysis(),
            _single_invalid_lane_band(),
            np.array([[[10, 20, 30]]], dtype=np.uint8),
            np.array([[0.5]], dtype=np.float32),
            np.array([[1]], dtype=np.uint32),
            0,
            1,
            output,
        )


def test_later_band_record_capacity_overflow_uses_budget_error() -> None:
    renderer = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    repair_records = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
    output = np.empty(1, dtype=repair_records.REPAIR_RECORD_DTYPE)
    arguments = (
        _single_invalid_lane_band(),
        np.array([[[10, 20, 30]]], dtype=np.uint8),
        np.array([[0.5]], dtype=np.float32),
        np.array([[1]], dtype=np.uint32),
        0,
        1,
    )

    first = renderer._write_analysis_band_native(
        _single_pixel_analysis(),
        *arguments,
        output,
    )

    assert first.records.size == 1
    with pytest.raises(repair_records.QualityRepairBudgetError):
        renderer._write_analysis_band_native(
            _single_pixel_analysis(),
            *arguments,
            output[1:],
        )


def test_fallback_phase_owns_full_repair_arena_before_reset() -> None:
    arena_module = importlib.import_module("src.depth_surge_3d.rendering.quality.arena")
    fallback = importlib.import_module("src.depth_surge_3d.rendering.quality.fallback")
    local_strip = importlib.import_module("src.depth_surge_3d.rendering.quality.local_strip")
    repair_records = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
    height = width = 9
    coverage = np.full((height, width), 16, dtype=np.uint8)
    coverage[4, 4] = 0
    regions = np.ones((height, width), dtype=np.uint32)
    regions[4, 4] = repair_records.NO_REGION
    analysis = local_strip.QualityVisibilityAnalysis(
        coverage,
        regions,
        tuple(np.full((height, width), value, dtype=np.uint8) for value in (3, 7, 11)),
    )
    records = np.zeros(1, dtype=repair_records.REPAIR_RECORD_DTYPE)
    records["pixel_index"] = np.uint32(4 * width + 4)
    records["lane_mask"] = np.uint16(0xFFFF)
    records["region_id"] = np.uint32(1)
    repair_arena = arena_module.QualityByteArena(
        np.empty(arena_module.QUALITY_REPAIR_ARENA_BYTES, dtype=np.uint8),
        "Quality repair",
    )

    result = fallback.precompute_native_fallbacks(
        records,
        analysis,
        render_shape=(height, width),
        region_count=1,
        budgets=local_strip.QualityRepairBudgets(local_slot_cap=0),
        in_place=True,
        arena=repair_arena,
    )

    assert result.query_count == 1
    assert repair_arena.cursor == repair_arena.capacity
    repair_arena.reset()
    first_exemplar_allocation = repair_arena.allocate((height, width), np.bool_)
    assert first_exemplar_allocation.ctypes.data == repair_arena.storage.ctypes.data


def test_injected_repair_arena_avoids_bulk_numpy_allocators(
    fixed_arenas,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = importlib.import_module("src.depth_surge_3d.rendering.quality.planner")
    local_strip = importlib.import_module("src.depth_surge_3d.rendering.quality.local_strip")
    repair_records = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
    height = width = 9
    coverage = np.full((height, width), 16, dtype=np.uint8)
    coverage[4, 4] = 0
    regions = np.ones((height, width), dtype=np.uint32)
    regions[4, 4] = repair_records.NO_REGION
    analysis = local_strip.QualityVisibilityAnalysis(
        coverage,
        regions,
        tuple(np.full((height, width), value, dtype=np.uint8) for value in (3, 7, 11)),
    )
    records = fixed_arenas.records()[:1]
    records.setflags(write=True)
    records.fill(0)
    records["pixel_index"] = np.uint32(4 * width + 4)
    records["lane_mask"] = np.uint16(0xFFFF)
    records["region_id"] = np.uint32(1)
    records.setflags(write=False)
    fixed_arenas.reset_eye()

    def reject_bulk_allocation(*_args, **_kwargs):
        raise AssertionError("repair planning used a bulk NumPy allocator")

    monkeypatch.setattr(np, "empty", reject_bulk_allocation)
    monkeypatch.setattr(np, "zeros", reject_bulk_allocation)
    monkeypatch.setattr(np, "array", reject_bulk_allocation)
    monkeypatch.setattr(np, "flatnonzero", reject_bulk_allocation)

    result = planner.plan_quality_repairs(
        analysis,
        records=records,
        render_shape=(height, width),
        local_limit_px=1,
        budgets=local_strip.QualityRepairBudgets(local_slot_cap=0),
        arenas=fixed_arenas,
    )

    assert result.records["backend"].tolist() == [3]


def test_native_float_scatter_matches_expand_oracle_randomized() -> None:
    native_kd = importlib.import_module(
        "src.depth_surge_3d.rendering.quality.native_kd"
    )._require_native()
    generator = np.random.default_rng(2026082502)

    for _case in range(200):
        height = int(generator.integers(1, 10))
        width = int(generator.integers(1, 10))
        pixel_count = height * width
        record_count = int(generator.integers(0, min(pixel_count, 24) + 1))
        records = _random_final_records(
            generator,
            pixel_count=pixel_count,
            record_count=record_count,
        )
        raw = _raw_records(records)
        expected_indexes, expected_colours = native_kd.expand_repair_scatter(raw, height, width)
        actual = np.full((pixel_count * 16, 3), -7.0, dtype=np.float32)
        expected = actual.copy()
        expected[expected_indexes] = expected_colours

        scattered = native_kd.scatter_repair_records_float_into(raw, 0, actual)

        assert scattered == expected_indexes.size
        np.testing.assert_array_equal(actual, expected)


def test_native_u8_scatter_clears_stale_workspace_and_matches_oracle() -> None:
    native_kd = importlib.import_module(
        "src.depth_surge_3d.rendering.quality.native_kd"
    )._require_native()
    generator = np.random.default_rng(2026082503)
    records = _random_final_records(generator, pixel_count=5, record_count=3)
    records[0]["lane_mask"] = np.uint16(0xFFFF)
    raw = _raw_records(records)
    expected_indexes, expected_colours = native_kd.expand_repair_scatter(raw, 1, 5)
    colours = np.full((5 * 16, 3), 255, dtype=np.uint8)
    mask = np.ones(5 * 16, dtype=np.bool_)

    scattered = native_kd.materialize_repair_scatter_u8_into(raw, 0, colours, mask)

    expected_mask = np.zeros(5 * 16, dtype=np.bool_)
    expected_mask[expected_indexes] = True
    expected_u8 = np.zeros((5 * 16, 3), dtype=np.uint8)
    expected_u8[expected_indexes] = expected_colours.astype(np.uint8)
    assert scattered == expected_indexes.size
    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_equal(colours, expected_u8)

    empty = raw[:0]
    native_kd.materialize_repair_scatter_u8_into(empty, 0, colours, mask)
    assert not np.any(mask)
    assert not np.any(colours)


def test_native_u8_scatter_rejects_overlapping_records() -> None:
    repair_records = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
    native_kd = importlib.import_module(
        "src.depth_surge_3d.rendering.quality.native_kd"
    )._require_native()
    records = np.zeros(2, dtype=repair_records.REPAIR_RECORD_DTYPE)
    records["pixel_index"] = np.uint32(0)
    records["lane_mask"] = np.array([0x000F, 0x000C], dtype=np.uint16)
    records["backend"] = np.uint8(1)
    colours = np.empty((16, 3), dtype=np.uint8)
    mask = np.empty(16, dtype=np.bool_)

    with pytest.raises(RuntimeError, match="overlap"):
        native_kd.materialize_repair_scatter_u8_into(_raw_records(records), 0, colours, mask)


def test_native_u8_scatter_preserves_chunk_boundary_offsets() -> None:
    renderer = importlib.import_module("src.depth_surge_3d.rendering.quality.renderer")
    repair_records = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
    native_kd = importlib.import_module(
        "src.depth_surge_3d.rendering.quality.native_kd"
    )._require_native()
    boundary = renderer.QUALITY_SCATTER_CHUNK_PIXELS
    records = np.zeros(2, dtype=repair_records.REPAIR_RECORD_DTYPE)
    records["pixel_index"] = np.array([boundary - 1, boundary], dtype=np.uint32)
    records["lane_mask"] = np.array([0x8000, 0x0001], dtype=np.uint16)
    records["fill_bgr"] = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    records["backend"] = np.uint8(1)

    first_colours = np.full((16, 3), 99, dtype=np.uint8)
    first_mask = np.ones(16, dtype=np.bool_)
    second_colours = np.full((16, 3), 99, dtype=np.uint8)
    second_mask = np.ones(16, dtype=np.bool_)
    first_count = native_kd.materialize_repair_scatter_u8_into(
        _raw_records(records[:1]), boundary - 1, first_colours, first_mask
    )
    second_count = native_kd.materialize_repair_scatter_u8_into(
        _raw_records(records[1:]), boundary, second_colours, second_mask
    )

    assert first_count == second_count == 1
    assert np.flatnonzero(first_mask).tolist() == [15]
    assert np.flatnonzero(second_mask).tolist() == [0]
    assert first_colours[15].tolist() == [1, 2, 3]
    assert second_colours[0].tolist() == [4, 5, 6]


def test_native_exemplar_preflight_matches_independent_mask_oracle() -> None:
    native_kd = importlib.import_module(
        "src.depth_surge_3d.rendering.quality.native_kd"
    )._require_native()
    generator = np.random.default_rng(2026082504)
    workspace = np.empty(4 * (512 * 512 + 256 * 256 + 128 * 128), dtype=np.uint8)

    for _case in range(48):
        height = int(generator.integers(24, 80))
        width = int(generator.integers(24, 80))
        coverage = generator.integers(0, 17, size=(height, width), dtype=np.uint8)
        regions = generator.integers(1, 4, size=(height, width), dtype=np.uint32)
        safe = np.ascontiguousarray(generator.random((height, width)) > 0.2)
        region = int(generator.integers(1, 4))
        target_count = int(generator.integers(1, min(20, height * width) + 1))
        targets = np.sort(
            generator.choice(height * width, size=target_count, replace=False).astype(np.uint32)
        )

        expected = _component_may_score_exemplar_oracle(coverage, regions, safe, targets, region)
        actual = native_kd.component_may_score_exemplar(
            coverage, regions, safe, targets, region, workspace
        )

        assert actual is expected


def test_native_exemplar_preflight_accepts_known_scoreable_component() -> None:
    native_kd = importlib.import_module(
        "src.depth_surge_3d.rendering.quality.native_kd"
    )._require_native()
    coverage = np.full((64, 64), 16, dtype=np.uint8)
    regions = np.ones((64, 64), dtype=np.uint32)
    safe = np.ones((64, 64), dtype=np.bool_)
    targets = np.array([32 * 64 + 32], dtype=np.uint32)
    workspace = np.empty(4 * (512 * 512 + 256 * 256 + 128 * 128), dtype=np.uint8)

    expected = _component_may_score_exemplar_oracle(coverage, regions, safe, targets, 1)
    actual = native_kd.component_may_score_exemplar(coverage, regions, safe, targets, 1, workspace)

    assert expected is True
    assert actual is True


def test_fixed_exemplar_pyramid_and_donor_enumeration_use_native_exact_oracles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arena_module = importlib.import_module("src.depth_surge_3d.rendering.quality.arena")
    exemplar = importlib.import_module("src.depth_surge_3d.rendering.quality.exemplar")
    generator = np.random.default_rng(2026082505)
    shape = (67, 71)
    working = generator.integers(0, 256, size=(*shape, 3), dtype=np.uint8)
    donor = np.ascontiguousarray(generator.random(shape) > 0.15)
    target = np.ascontiguousarray(generator.random(shape) < 0.12)
    processed = np.ascontiguousarray(generator.random(shape) > 0.25)
    barrier = np.ascontiguousarray(generator.random(shape) < 0.08)
    source = exemplar.ExemplarLevel(
        working,
        donor,
        target,
        processed,
        barrier,
    )
    expected_parent = exemplar.downsample_exemplar_level(source)
    expected_centres = exemplar.enumerate_donor_centres(donor)
    arena = arena_module.QualityByteArena(
        np.empty(8 * 1024 * 1024, dtype=np.uint8),
        "native exemplar oracle",
    )
    monkeypatch.setattr(
        exemplar,
        "_child_coordinates",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("fixed pyramid entered the scalar reducer")
        ),
    )
    monkeypatch.setattr(
        np,
        "cumsum",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fixed donor enumeration used a NumPy integral scan")
        ),
    )

    actual_parent = exemplar._downsample_exemplar_level_fixed(source, arena)
    actual_centres = exemplar._enumerate_donor_centres_fixed(donor, arena)

    for field in (
        "working_bgr",
        "donor_mask",
        "target_mask",
        "processed_mask",
        "barrier_mask",
    ):
        np.testing.assert_array_equal(
            getattr(actual_parent, field),
            getattr(expected_parent, field),
        )
    np.testing.assert_array_equal(actual_centres, expected_centres)


def test_indexed_fixed_exemplar_repairs_the_full_record_arena_without_copying() -> None:
    arena_module = importlib.import_module("src.depth_surge_3d.rendering.quality.arena")
    exemplar = importlib.import_module("src.depth_surge_3d.rendering.quality.exemplar")
    local_strip = importlib.import_module("src.depth_surge_3d.rendering.quality.local_strip")
    repair_records = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
    height = width = 65
    target_pixel = 32 * width + 32
    coverage = np.full((height, width), 16, dtype=np.uint8)
    regions = np.ones((height, width), dtype=np.uint32)
    coverage[32, 32] = np.uint8(0)
    regions[32, 32] = repair_records.NO_REGION
    analysis = local_strip.QualityVisibilityAnalysis(
        coverage,
        regions,
        tuple(np.full((height, width), value, dtype=np.uint8) for value in (10, 20, 30)),
    )
    safe = np.ones((height, width), dtype=np.bool_)
    safe[31:34, 31:34] = False
    records = np.zeros(2, dtype=repair_records.REPAIR_RECORD_DTYPE)
    records[0]["pixel_index"] = np.uint32(0)
    records[0]["lane_mask"] = np.uint16(0xFFFF)
    records[0]["region_id"] = np.uint32(1)
    records[0]["fill_bgr"] = np.array([1, 2, 3], dtype=np.uint8)
    records[0]["backend"] = np.uint8(3)
    records[1]["pixel_index"] = np.uint32(target_pixel)
    records[1]["lane_mask"] = np.uint16(0xFFFF)
    records[1]["region_id"] = np.uint32(1)
    records[1]["fill_bgr"] = np.array([99, 98, 97], dtype=np.uint8)
    distractor_before = records[:1].copy()
    expected = exemplar.repair_component_exemplar(
        records[1:].copy(),
        analysis,
        safe,
        region_id=1,
        render_shape=(height, width),
        budgets=local_strip.QualityRepairBudgets(),
    )
    arena = arena_module.QualityByteArena(
        np.empty(arena_module.QUALITY_REPAIR_ARENA_BYTES, dtype=np.uint8),
        "indexed exemplar",
    )

    actual = exemplar.repair_component_exemplar(
        records,
        analysis,
        safe,
        region_id=1,
        render_shape=(height, width),
        budgets=local_strip.QualityRepairBudgets(),
        in_place=True,
        arena=arena,
        record_indexes=np.array([1], dtype=np.uint32),
    )

    assert actual.records is records
    np.testing.assert_array_equal(actual.records[:1], distractor_before)
    np.testing.assert_array_equal(actual.records[1:], expected.records)
    assert actual.processed_target_count == expected.processed_target_count
    assert actual.unfinished_target_count == expected.unfinished_target_count
    assert actual.patch_iteration_count == expected.patch_iteration_count
    assert actual.donor_evaluation_count == expected.donor_evaluation_count


def test_planner_passes_component_indexes_without_allocating_record_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arena_module = importlib.import_module("src.depth_surge_3d.rendering.quality.arena")
    exemplar = importlib.import_module("src.depth_surge_3d.rendering.quality.exemplar")
    local_strip = importlib.import_module("src.depth_surge_3d.rendering.quality.local_strip")
    planner = importlib.import_module("src.depth_surge_3d.rendering.quality.planner")
    repair_records = importlib.import_module("src.depth_surge_3d.rendering.quality.repair_records")
    height = width = 64
    coverage = np.full((height, width), 16, dtype=np.uint8)
    regions = np.ones((height, width), dtype=np.uint32)
    target_pixels = np.array(
        [row * width + column for row in range(28, 35) for column in range(28, 35)],
        dtype=np.uint32,
    )
    target_rows = target_pixels // np.uint32(width)
    target_columns = target_pixels % np.uint32(width)
    coverage[target_rows, target_columns] = np.uint8(0)
    regions[target_rows, target_columns] = repair_records.NO_REGION
    analysis = local_strip.QualityVisibilityAnalysis(
        coverage,
        regions,
        tuple(np.full((height, width), value, dtype=np.uint8) for value in (10, 20, 30)),
    )
    records = np.zeros(50, dtype=repair_records.REPAIR_RECORD_DTYPE)
    records[0]["pixel_index"] = np.uint32(0)
    records[1:]["pixel_index"] = target_pixels
    records["lane_mask"] = np.uint16(0xFFFF)
    records["region_id"] = np.uint32(1)
    records["fill_bgr"] = np.array([99, 98, 97], dtype=np.uint8)
    arenas = arena_module.QualityFixedArenas.allocate()
    original_allocate = arena_module.QualityByteArena.allocate
    observed_component_indexes: list[np.ndarray] = []

    def reject_record_copy(self, shape, dtype, **kwargs):
        assert np.dtype(dtype) != repair_records.REPAIR_RECORD_DTYPE
        return original_allocate(self, shape, dtype, **kwargs)

    def observe_indexed_repair(full_records, *_args, record_indexes=None, **_kwargs):
        assert full_records.size == records.size
        assert record_indexes is not None
        observed_component_indexes.append(record_indexes.copy())
        return exemplar.ExemplarRepairResult(
            full_records,
            0,
            int(record_indexes.size),
            0,
            0,
        )

    monkeypatch.setattr(arena_module.QualityByteArena, "allocate", reject_record_copy)
    monkeypatch.setattr(exemplar, "repair_component_exemplar", observe_indexed_repair)

    planner.plan_quality_repairs(
        analysis,
        records=records,
        render_shape=(height, width),
        local_limit_px=1,
        budgets=local_strip.QualityRepairBudgets(local_slot_cap=0),
        arenas=arenas,
    )

    assert len(observed_component_indexes) == 1
    assert observed_component_indexes[0].size == 49
