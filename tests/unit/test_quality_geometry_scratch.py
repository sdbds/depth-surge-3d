"""Dense stable-surface queries must fit the declared temporary array budget."""

from __future__ import annotations

import sys
import weakref

import numpy as np
import pytest

from src.depth_surge_3d.rendering.quality import geometry, native_kd
from src.depth_surge_3d.rendering.quality.implicit_kd import (
    QualityGeometryQueryBudgetError,
    VisitBudget,
)
from src.depth_surge_3d.rendering.quality import renderer as quality_renderer
from src.depth_surge_3d.core.resource_errors import QualityHostBudgetError


def _root_array(value):
    while isinstance(value.base, np.ndarray):
        value = value.base
    return value


def _arrays(value):
    if isinstance(value, np.ndarray):
        yield _root_array(value)
    elif isinstance(value, native_kd.NativeBatchQueryResult):
        for name in ("found", "sample_index", "distance2", "visited_nodes"):
            yield _root_array(getattr(value, name))
    elif isinstance(value, tuple):
        for item in value:
            yield from _arrays(item)


@pytest.mark.skipif(not native_kd.native_available(), reason="prebuilt native extension required")
@pytest.mark.parametrize("fallback", [False, True])
def test_dense_stable_queries_keep_array_lifetimes_within_scratch_budget(fallback):
    height = width = 64
    count = height * width
    source_regions = np.arange(1, count + 1, dtype=np.uint32).reshape(height, width)
    final_regions = np.roll(source_regions, 32, axis=1) if fallback else source_regions.copy()
    included = np.ones((height, width), dtype=np.bool_)
    edge_band = included.copy()
    regions = geometry._RegionState(
        source_region_map=source_regions,
        final_region_map=final_regions,
        edge_band=edge_band,
        stable_sample_include=included,
        region_count=count,
        radius=1,
        edge_band_pixel_count=count,
        final_region_hash="test-only",
    )
    source = source_regions.astype(np.float32)
    output = np.full_like(source, -1)
    excluded = {
        id(_root_array(value))
        for value in (source_regions, final_regions, included, edge_band, source, output)
    }
    watched = {
        geometry._resample_primitives_in_place.__code__,
        geometry._resolve_stable_surface_queries.__code__,
        geometry._resolve_stable_query_fallback.__code__,
        native_kd.query_native_local_geometry_batch.__code__,
        native_kd.query_native_geometry_batch.__code__,
        native_kd._freeze_batch.__code__,
    }
    peak_bytes = 0

    def observe(frame, event, _argument):
        nonlocal peak_bytes
        if event in {"line", "return"} and frame.f_code in watched:
            live = {}
            current = frame
            while current is not None and current.f_code in watched:
                for value in current.f_locals.values():
                    for array in _arrays(value):
                        if id(array) not in excluded:
                            live[id(array)] = array.nbytes
                current = current.f_back
            peak_bytes = max(peak_bytes, sum(live.values()))
        return observe if frame.f_code in watched else None

    previous = sys.gettrace()
    budget = VisitBudget(limit=10_000_000)
    recorder = geometry._LifetimeRecorder()
    try:
        sys.settrace(observe)
        result = geometry._resample_primitives_in_place(
            (source,), (output,), regions, budget, recorder
        )
    finally:
        sys.settrace(previous)
    np.testing.assert_array_equal(output, final_regions.astype(np.float32))
    assert result.query_count == count
    if fallback:
        spans = [min(63, index + 16) - max(0, index - 16) + 1 for index in range(64)]
        expected_visits = sum(spans) ** 2 + count
        assert result.visited_nodes_max == 1090
        assert result.index_bytes > 0
    else:
        expected_visits = count
        assert result.visited_nodes_max == 1
        assert result.index_bytes == 0
    assert result.visited_nodes_total == budget.consumed == expected_visits
    assert (
        peak_bytes <= 40 * count + 64
    ), f"query arrays retain {peak_bytes} bytes for {count} queries"


@pytest.mark.parametrize("geometry_mode", ["relative", "metric"])
@pytest.mark.parametrize("native_shape", [(8, 16), (200, 200)])
def test_nearest_preflight_includes_mode_payload_and_index_build_scratch(
    geometry_mode, native_shape
):
    render_shape = (64, 64)
    pixels = 64 * 64
    native_pixels = native_shape[0] * native_shape[1]
    phases = quality_renderer._quality_public_memory_preflight(
        render_shape=render_shape,
        native_shape=native_shape,
        occlusion_fill="none",
        materialize_public_masks=False,
        geometry_mode=geometry_mode,
    )
    payload = (
        21 * pixels + 9 * native_pixels
        if geometry_mode == "relative"
        else 29 * pixels + 18 * native_pixels
    )
    expected = (
        payload
        + 40 * pixels
        + 10 * native_pixels
        + 5
        + quality_renderer.QUALITY_RUNTIME_OVERHEAD_BYTES
        + quality_renderer.QUALITY_AUXILIARY_THREAD_BYTES
    )
    assert phases["quality-geometry-nearest"] == expected


def test_nearest_preflight_rejects_an_undercharged_index_build(monkeypatch):
    pixels = 64 * 64
    native_pixels = 200 * 200
    previous_peak = (
        68 * pixels
        + 18 * native_pixels
        + quality_renderer.QUALITY_RUNTIME_OVERHEAD_BYTES
        + quality_renderer.QUALITY_AUXILIARY_THREAD_BYTES
    )
    monkeypatch.setattr(quality_renderer, "QUALITY_HOST_BUDGET_BYTES", previous_peak)
    with pytest.raises(QualityHostBudgetError, match="quality-geometry-nearest"):
        quality_renderer._quality_public_memory_preflight(
            render_shape=(64, 64),
            native_shape=(200, 200),
            occlusion_fill="none",
            materialize_public_masks=False,
        )


@pytest.mark.skipif(not native_kd.native_available(), reason="prebuilt native extension required")
@pytest.mark.parametrize("geometry_mode", ["relative", "metric"])
def test_native_construction_scratch_is_released_before_queries(monkeypatch, geometry_mode):
    low_refs = []
    solve = geometry._solve_final_regions
    resample = geometry._resample_primitives_in_place

    def capture_low_fields(*args, **kwargs):
        low_refs.append(weakref.ref(args[2]))
        if geometry_mode == "metric":
            low_refs.append(weakref.ref(args[3]))
        return solve(*args, **kwargs)

    def check_released(*args, **kwargs):
        assert low_refs
        assert all(reference() is None for reference in low_refs)
        return resample(*args, **kwargs)

    monkeypatch.setattr(geometry, "_solve_final_regions", capture_low_fields)
    monkeypatch.setattr(geometry, "_resample_primitives_in_place", check_released)
    values = np.linspace(0.01, 1.0, 32, dtype=np.float32).reshape(4, 8)
    guide = np.zeros((8, 16, 3), dtype=np.uint8)
    if geometry_mode == "relative":
        geometry.build_relative_quality_geometry(
            guide, values, render_shape=(8, 16), stereo_strength=2.0, convergence=0.5
        )
    else:
        geometry.build_metric_quality_geometry(
            guide,
            values,
            np.ones_like(values, dtype=np.bool_),
            np.float32(1.0),
            render_shape=(8, 16),
            virtual_baseline_mm=64.0,
            convergence_distance_m=2.0,
            max_disparity_percent=5.0,
            retained_crop_width=16,
        )


@pytest.mark.skipif(not native_kd.native_available(), reason="prebuilt native extension required")
def test_kd_builder_scratch_is_charged_then_released_before_fallback(monkeypatch):
    source_regions = np.ones((1, 64), dtype=np.uint32)
    source_regions[0, -1] = 2
    regions = geometry._RegionState(
        source_region_map=source_regions,
        final_region_map=np.array([[2]], dtype=np.uint32),
        edge_band=np.ones((1, 1), dtype=np.bool_),
        stable_sample_include=np.ones(source_regions.shape, dtype=np.bool_),
        region_count=2,
        radius=1,
        edge_band_pixel_count=1,
        final_region_hash="test-only",
    )
    recorder = geometry._LifetimeRecorder()
    build = native_kd.build_native_implicit_region_kd
    query = native_kd.query_native_geometry_batch
    mask_refs = []

    def checked_build(*args, **kwargs):
        assert recorder._live["retained_zero_index_build_scratch"] == 5
        mask_refs.append(weakref.ref(kwargs["allowed_regions"]))
        return build(*args, **kwargs)

    def checked_query(index, *args, **kwargs):
        assert "retained_zero_index_build_scratch" not in recorder._live
        assert recorder._live["retained_zero_index"] == index.index_bytes
        assert mask_refs[0]() is None
        return query(index, *args, **kwargs)

    monkeypatch.setattr(native_kd, "build_native_implicit_region_kd", checked_build)
    monkeypatch.setattr(native_kd, "query_native_geometry_batch", checked_query)
    output = np.full((1, 1), -1, dtype=np.float32)
    result = geometry._resample_primitives_in_place(
        (np.arange(64, dtype=np.float32).reshape(1, 64),),
        (output,),
        regions,
        VisitBudget(10_000),
        recorder,
    )
    assert mask_refs and output.item() == 63.0
    assert result.index_bytes == 16
    assert result.visited_nodes_total == result.visited_nodes_max == 34


def _mixed_query_fixture(count):
    source_regions = np.arange(1, 4097, dtype=np.uint32).reshape(64, 64)
    width = max(1, count)
    final_regions = np.ones((1, width), dtype=np.uint32)
    for column in range(0, count, 3):
        source_x = min(63, max(0, int((column + 0.5) * 64 / width - 0.5)))
        final_regions[0, column] = source_regions[31, source_x]
    final_regions[0, 1::3] = source_regions[-1, -1]
    band = np.ones((1, width), dtype=np.bool_)
    if count == 0:
        band[:] = False
    return geometry._RegionState(
        source_region_map=source_regions,
        final_region_map=final_regions,
        edge_band=band,
        stable_sample_include=np.ones((64, 64), dtype=np.bool_),
        region_count=4096,
        radius=1,
        edge_band_pixel_count=count,
        final_region_hash="test-only",
    )


def _whole_batch_query_oracle(regions):
    count = regions.edge_band_pixel_count
    pixels = np.empty(count, dtype=np.uint32)
    selected = np.empty(count, dtype=np.uint32)
    query_y = np.empty(count, dtype=np.float64)
    query_x = np.empty(count, dtype=np.float64)
    geometry._collect_stable_surface_queries(regions, pixels, selected, query_y, query_x)
    if not count:
        return pixels, pixels.copy(), np.empty(0, dtype=np.uint64), 0, 0
    budget = VisitBudget(1_000_000)
    local = native_kd.query_native_local_geometry_batch(
        regions.source_region_map,
        regions.stable_sample_include,
        selected,
        query_y,
        query_x,
        radius_px=16,
        budget=budget,
    )
    samples = local.sample_index.copy()
    visits = local.visited_nodes.astype(np.uint64)
    local_debits = budget.consumed
    missing = ~local.found
    index_bytes = 0
    if missing.any():
        allowed = np.zeros(regions.region_count + 1, dtype=np.bool_)
        allowed[selected[missing]] = True
        index = native_kd.build_native_implicit_region_kd(
            regions.source_region_map,
            region_count=regions.region_count,
            include=regions.stable_sample_include,
            allowed_regions=allowed,
        )
        fallback = native_kd.query_native_geometry_batch(
            index, selected[missing], query_y[missing], query_x[missing], budget=budget
        )
        assert fallback.found.all()
        samples[missing] = fallback.sample_index
        visits[missing] += fallback.visited_nodes
        index_bytes = index.index_bytes
    return pixels, samples, visits, index_bytes, local_debits


@pytest.mark.skipif(not native_kd.native_available(), reason="prebuilt native extension required")
@pytest.mark.parametrize("count", [0, 1, 2, 4, 9, 19, 130])
def test_query_chunks_match_whole_batch_samples_and_visit_accounting(monkeypatch, count):
    regions = _mixed_query_fixture(count)
    pixels, samples, visits, index_bytes, _ = _whole_batch_query_oracle(regions)
    sources = (regions.source_region_map.astype(np.float32),)
    sources += (np.square(sources[0]),)
    outputs = tuple(np.full(regions.final_region_map.shape, -1, dtype=np.float32) for _ in sources)
    expected = tuple(output.copy() for output in outputs)
    for source, output in zip(sources, expected, strict=True):
        output.reshape(-1)[pixels] = source.reshape(-1)[samples]
    calls = []
    local_query = native_kd.query_native_local_geometry_batch
    fallback_query = native_kd.query_native_geometry_batch

    def local(*args, **kwargs):
        calls.append(("local", args[2].size))
        return local_query(*args, **kwargs)

    def fallback(*args, **kwargs):
        calls.append(("fallback", args[1].size))
        return fallback_query(*args, **kwargs)

    monkeypatch.setattr(native_kd, "query_native_local_geometry_batch", local)
    monkeypatch.setattr(native_kd, "query_native_geometry_batch", fallback)
    monkeypatch.setattr(geometry, "_GEOMETRY_QUERY_CHUNK_CAP", 3)
    budget = VisitBudget(1_000_000)
    actual = geometry._resample_primitives_in_place(
        sources, outputs, regions, budget, geometry._LifetimeRecorder()
    )

    for actual_output, expected_output in zip(outputs, expected, strict=True):
        np.testing.assert_array_equal(actual_output, expected_output)
    assert actual.query_count == count
    assert actual.index_bytes == index_bytes
    assert actual.visited_nodes_total == budget.consumed == int(visits.sum(dtype=np.uint64))
    assert actual.visited_nodes_max == int(visits.max(initial=np.uint64(0)))
    phases = [phase for phase, _size in calls]
    assert phases == sorted(phases, key={"local": 0, "fallback": 1}.get)
    assert all(0 < size <= 3 for _phase, size in calls)
    if count == 0:
        assert not calls


@pytest.mark.skipif(not native_kd.native_available(), reason="prebuilt native extension required")
@pytest.mark.parametrize("failure_phase", ["local", "fallback"])
def test_query_budget_failure_does_not_partially_write_primitives(monkeypatch, failure_phase):
    regions = _mixed_query_fixture(130)
    _, _, _, _, local_debits = _whole_batch_query_oracle(regions)
    limit = local_debits - 1 if failure_phase == "local" else local_debits + 1
    budget = VisitBudget(limit)
    sources = (
        regions.source_region_map.astype(np.float32),
        np.ones((64, 64), dtype=np.float32),
    )
    outputs = tuple(np.full((1, 130), -1, dtype=np.float32) for _ in sources)
    monkeypatch.setattr(geometry, "_GEOMETRY_QUERY_CHUNK_CAP", 3)
    with pytest.raises(QualityGeometryQueryBudgetError):
        geometry._resample_primitives_in_place(
            sources, outputs, regions, budget, geometry._LifetimeRecorder()
        )
    assert budget.consumed == limit
    assert all(np.all(output == -1) for output in outputs)


def test_combined_local_and_fallback_visits_do_not_wrap_uint32(monkeypatch):
    maximum = np.iinfo(np.uint32).max
    samples = np.array([maximum, 7, maximum], dtype=np.uint32)
    local_visits = np.array([maximum, 1, maximum], dtype=np.uint32)
    fallback = native_kd._freeze_batch(
        np.array([5, 6], dtype=np.uint32),
        np.zeros(2, dtype=np.float64),
        np.array([5, 6], dtype=np.uint32),
    )
    monkeypatch.setattr(native_kd, "query_native_geometry_batch", lambda *_args, **_kw: fallback)
    extra_visits, combined_max = geometry._resolve_stable_query_fallback(
        None,
        np.ones(3, dtype=np.uint32),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        samples,
        local_visits,
        VisitBudget(2**40),
    )
    assert samples.tolist() == [5, 7, 6]
    assert extra_visits == 11
    assert combined_max == int(maximum) + 6
