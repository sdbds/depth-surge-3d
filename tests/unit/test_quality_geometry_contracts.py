"""Validate small real inputs at the geometry, native query and memory boundaries."""

import numpy as np
import pytest

from src.depth_surge_3d.rendering import stereo_geometry
from src.depth_surge_3d.rendering.quality import geometry, implicit_kd, local_strip, native_kd
from src.depth_surge_3d.rendering.quality.repair_records import QualityRepairBudgetError
from src.depth_surge_3d.rendering.quality import renderer, types
from src.depth_surge_3d.rendering.stereo_renderer import StereoRenderer, StereoRenderSettings


@pytest.mark.parametrize("mode", ["fast", "quality"])
@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("inverse_depth", np.ones((1, 1), np.float64), "float32"),
        ("inverse_depth", np.empty((0, 2), np.float32), "non-?empty"),
        ("inverse_depth", np.array([[-1]], np.float32), "nonnegative"),
        ("inverse_depth", np.array([[np.inf]], np.float32), "finite"),
        ("valid", np.ones((2, 3), np.uint8), "bool"),
        ("valid", np.ones((2, 1), np.bool_), "match"),
        ("focal_x_normalized", 1.0, "float32"),
        ("focal_x_normalized", np.float32(0), "positive|within"),
        ("virtual_baseline_mm", False, "numeric"),
        ("convergence_distance_m", True, "numeric"),
        ("convergence_distance_m", 0.0, "positive"),
        ("retained_crop_width", 4, "crop"),
        ("retained_crop_width", False, "crop"),
    ],
)
def test_physical_camera_projection_rejects_invalid_data_in_both_modes(mode, field, bad, message):
    shape = (2, 3)
    args = dict(
        inverse_depth=np.ones(shape, np.float32),
        valid=np.ones(shape, np.bool_),
        focal_x_normalized=np.float32(1),
        render_shape=shape,
        virtual_baseline_mm=63,
        convergence_distance_m=2.0,
        max_disparity_percent=2,
        retained_crop_width=3,
    )
    if field == "inverse_depth" and bad.size:
        bad = np.full(shape, bad.flat[0], dtype=bad.dtype)
    args[field] = bad
    build = stereo_geometry.build_metric_geometry
    if mode == "quality":
        build = geometry.build_metric_quality_geometry
        args["guide_bgr"] = np.ones((*shape, 3), np.uint8)
    with pytest.raises((TypeError, ValueError), match=message):
        build(**args)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_regions", np.ones((2, 2), np.int64), "uint32"),
        ("source_regions", np.ones((2, 1), np.uint32), "match"),
        ("source_regions", np.zeros((2, 2), np.uint32), "canonical"),
        ("source_regions", np.full((2, 2), 0xFFFFFFFF, np.uint32), "canonical"),
        ("selected_region", True, "integer"),
        ("selected_region", 2, "present"),
        ("render_shape", (2, False), "integer"),
        ("render_shape", (0, 2), "positive"),
        ("output_coordinate", (False, 0), "integer"),
        ("output_coordinate", (0, 2), "outside"),
    ],
)
def test_one_sided_sampler_rejects_invalid_coordinates_or_region_labels(field, value, message):
    args = dict(
        values=np.ones((2, 2), np.float32),
        source_regions=np.ones((2, 2), np.uint32),
        selected_region=1,
        render_shape=(2, 2),
        output_coordinate=(1, 1),
    )
    args[field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        geometry.one_sided_resample_scalar(**args)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("alias", "share storage"),
        ("read-only", "writable"),
        ("short-row", "width-sized"),
        ("wrong-eye", "eye"),
        ("nonfinite", "finite"),
        ("overflow", "int32 range"),
    ],
)
def test_offset_failure_does_not_overwrite_the_callers_first_row(case, message):
    total = np.ones((1, 3), np.float64)
    output = np.full((1, 3), 41, dtype=np.int32)
    scratch = np.zeros(3, np.float64)
    eye = "left"
    if case == "alias":
        output = scratch.view(np.int32)[:3].reshape(1, 3)
    elif case == "read-only":
        output.setflags(write=False)
    elif case == "short-row":
        scratch = scratch[:-1]
    elif case == "wrong-eye":
        eye = "both"
    elif case == "nonfinite":
        total[0, 1] = np.nan
    else:
        total[0, 1] = 2.0**32
    initial = output.copy()
    with pytest.raises((TypeError, ValueError), match=message):
        geometry.build_geometry_eye_offsets_into(
            total, eye=eye, output_int32=output, row_float64_scratch=scratch
        )
    np.testing.assert_array_equal(output, initial)


@pytest.mark.parametrize(("limit", "consumed"), [(True, 0), (-1, 0), (2**64, 0), (1, 2)])
def test_visit_budget_rejects_unrepresentable_accounting(limit, consumed):
    with pytest.raises((TypeError, ValueError), match="integer|nonnegative|uint64|exceed"):
        implicit_kd.VisitBudget(limit=limit, consumed=consumed)


def test_invalid_queries_do_not_debit_the_shared_visit_budget():
    ids = np.array([[1, 1, 2, 2]], np.uint32)
    index = implicit_kd.build_implicit_region_kd(ids, region_count=2, include=ids == 1)
    budget = implicit_kd.VisitBudget(100)
    for region, x, error in [
        (1, np.inf, "finite"),
        (1, "1", "numeric"),
        (3, 0.5, "exceeds"),
        (2, 0.5, "no indexed member"),
    ]:
        with pytest.raises((TypeError, ValueError), match=error):
            index.query_geometry(region, sy=0.0, sx=x, budget=budget)
    with pytest.raises(ValueError, match="inside"):
        index.query_repair(1, target_y=0, target_x=4, radius_px=0, budget=budget)
    assert budget.consumed == 0
    good = index.query_geometry(1, sy=0.0, sx=0.9, budget=budget)
    assert good.sample_index == 1 and good.visited_nodes == budget.consumed


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("index", object(), "index"),
        ("budget", None, "budget"),
        ("target_x", np.array([0], np.float32), "float64"),
        ("query_regions", np.ones((1, 2), np.uint32), "one-dimensional"),
        ("target_y", np.empty(0, np.float64), "query count"),
    ],
)
def test_native_geometry_batch_refuses_mismatched_vectors_without_consuming_budget(
    field, bad, message
):
    budget = implicit_kd.VisitBudget(100)
    args = dict(
        index=native_kd.build_native_implicit_region_kd(np.ones((1, 3), np.uint32), region_count=1),
        query_regions=np.ones(1, np.uint32),
        target_y=np.zeros(1, np.float64),
        target_x=np.ones(1, np.float64),
        budget=budget,
    )
    args[field] = bad
    with pytest.raises((TypeError, ValueError), match=message):
        native_kd.query_native_geometry_batch(**args)
    assert budget.consumed == 0


def test_local_reservation_is_atomic_even_at_the_uint64_ceiling():
    maximum = 2**64 - 1
    budgets = local_strip.QualityRepairBudgets(local_slot_cap=maximum, local_neighbor_sample_cap=20)
    assert budgets.reserve_local(maximum - 1, 10)
    assert not budgets.reserve_local(1, 11)
    assert budgets.local_budget_skipped_run_count == 1
    assert (budgets.local_slots_consumed, budgets.local_neighbor_samples_consumed) == (
        maximum - 1,
        10,
    )
    with pytest.raises(QualityRepairBudgetError, match="uint64"):
        budgets.reserve_local(2, 0)
    assert budgets.reserve_local(1, 10)
    assert (budgets.local_slots_consumed, budgets.local_neighbor_samples_consumed) == (maximum, 20)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("canonical", np.full((1, 2), 1.1, np.float32), "canonical"),
        ("stereo_strength", np.nan, "stereo_strength"),
        ("geometry_visit_cap", True, "integer"),
        ("geometry_visit_cap", -1, "nonnegative"),
        ("guide_bgr", np.zeros((2, 3, 3), np.float32), "guide_bgr"),
    ],
)
def test_relative_surface_build_rejects_invalid_primitives_before_region_work(field, value, error):
    args = dict(
        guide_bgr=np.zeros((2, 3, 3), np.uint8),
        canonical=np.ones((1, 2), np.float32),
        render_shape=(2, 3),
        stereo_strength=2.0,
        convergence=0.5,
    )
    args[field] = value
    with pytest.raises((ValueError, TypeError), match=error):
        geometry.build_relative_quality_geometry(**args)


@pytest.mark.parametrize(
    ("name", "bad", "message"),
    [
        ("scene", np.zeros((2, 3), np.uint8), "BGR"),
        ("scene", np.zeros((2, 3, 3), np.float32), "uint8"),
        ("depth", np.zeros(3, np.float32), "2D"),
        ("settings", StereoRenderSettings(), "quality mode"),
    ],
)
def test_relative_frame_preparation_validates_its_public_boundary(name, bad, message):
    inputs = dict(
        scene=np.zeros((2, 3, 3), np.uint8),
        depth=np.full((2, 3), 0.5, np.float32),
        settings=StereoRenderSettings(stereo_render_mode="quality"),
    )
    inputs[name] = bad
    with pytest.raises((TypeError, ValueError), match=message):
        renderer.prepare_relative_quality_frame(
            frame=inputs["scene"], canonical=inputs["depth"], settings=inputs["settings"]
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("renderer", None, "StereoRenderer"),
        ("primitives", object(), "MetricStereoPrimitiveInput"),
        ("controls", object(), "QualityStereoControls"),
        ("frame", np.zeros((2, 3), np.uint8), "BGR"),
        ("frame", np.zeros((2, 3, 3), np.float32), "uint8"),
        ("metric", {}, "missing"),
        ("inverse_depth", np.ones((2, 3, 1), np.float32), "two-dimensional"),
    ],
)
def test_metric_renderer_refuses_unusable_frames_and_sensor_packets(field, value, message):
    from types import SimpleNamespace

    metric = SimpleNamespace(
        inverse_depth=np.ones((2, 3), np.float32),
        valid=np.ones((2, 3), np.bool_),
        focal_x_normalized=np.float32(1),
    )
    if field == "inverse_depth":
        metric.inverse_depth = value
    if field == "metric":
        metric = value
    args = dict(
        renderer=StereoRenderer(device="cpu"),
        frame=np.zeros((2, 3, 3), np.uint8),
        primitives=types.MetricStereoPrimitiveInput(metric, 63.0, 1.0, 5.0, 3),
        controls=types.QualityStereoControls("none", 1),
    )
    if field in args:
        args[field] = value
    with pytest.raises(TypeError, match=message):
        renderer.render_metric_primitives_compact(**args)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("holes", 8), "occlusion_fill"),
        (("none", True), "non-boolean"),
        (("background", 33), "1..32"),
    ],
)
def test_quality_fill_controls_reject_unrepresentable_modes_and_radius(arguments, message):
    with pytest.raises(ValueError, match=message):
        types.QualityStereoControls(*arguments)


def test_hole_histograms_are_width_compatible_and_immutable_after_construction():
    with pytest.raises(TypeError, match="uint64"):
        types.DenseHistogramPairs(np.array([1], np.int32))
    with pytest.raises(TypeError, match="uint64"):
        types.HoleRunDiagnostics(np.zeros(32, np.uint64), np.zeros(2, np.uint8))
    with pytest.raises(ValueError, match="width"):
        types.HoleRunDiagnostics(np.zeros(31, np.uint64), np.zeros(2, np.uint64))
    ok = types.HoleRunDiagnostics(np.zeros(32, np.uint64), np.zeros(2, np.uint64))
    assert (
        not ok.lane_count_histogram.flags.writeable
        and not ok.touched_pixel_span_histogram.flags.writeable
    )
