"""Validation and relative-conversion tests for backend-neutral stereo geometry."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.depth_surge_3d.rendering.stereo_geometry import (
    MetricProjectionStats,
    StereoGeometryFrame,
    build_metric_geometry,
    build_relative_geometry,
    resize_metric_geometry,
)
from src.depth_surge_3d.rendering.stereo_renderer import (
    calculate_geometry_eye_sample_offsets,
)


def test_geometry_frame_requires_exact_dtypes_and_shapes() -> None:
    with pytest.raises(TypeError, match="near_score.*float32"):
        StereoGeometryFrame(
            np.ones((2, 3), dtype=np.float64),
            np.zeros((2, 3), dtype=np.float64),
            np.ones((2, 3), dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="same shape"):
        StereoGeometryFrame(
            np.ones((2, 3), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float64),
            np.ones((2, 3), dtype=np.bool_),
        )


@pytest.mark.parametrize(
    ("field", "values", "error", "message"),
    [
        (
            "total_disparity_fraction",
            np.zeros((2, 3), dtype=np.float32),
            TypeError,
            "total_disparity_fraction.*float64",
        ),
        (
            "source_valid",
            np.ones((2, 3), dtype=np.uint8),
            TypeError,
            "source_valid.*bool",
        ),
        (
            "near_score",
            np.array([[0.0, np.inf, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
            ValueError,
            "near_score.*finite",
        ),
        (
            "near_score",
            np.array([[0.0, -0.1, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
            ValueError,
            "near_score.*nonnegative",
        ),
        (
            "total_disparity_fraction",
            np.array([[0.0, np.nan, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64),
            ValueError,
            "total_disparity_fraction.*finite",
        ),
    ],
)
def test_geometry_frame_rejects_invalid_field_values(
    field: str,
    values: np.ndarray,
    error: type[Exception],
    message: str,
) -> None:
    arguments = {
        "near_score": np.ones((2, 3), dtype=np.float32),
        "total_disparity_fraction": np.zeros((2, 3), dtype=np.float64),
        "source_valid": np.ones((2, 3), dtype=np.bool_),
    }
    arguments[field] = values

    with pytest.raises(error, match=message):
        StereoGeometryFrame(**arguments)


def test_geometry_frame_requires_two_dimensions() -> None:
    with pytest.raises(ValueError, match="2D"):
        StereoGeometryFrame(
            np.ones((1, 2, 3), dtype=np.float32),
            np.zeros((1, 2, 3), dtype=np.float64),
            np.ones((1, 2, 3), dtype=np.bool_),
        )


def test_relative_builder_uses_existing_total_disparity_formula() -> None:
    geometry = build_relative_geometry(
        np.array([[0.0, 0.5, 1.0]], dtype=np.float32),
        (1, 3),
        stereo_strength=2.0,
        convergence=0.5,
    )
    np.testing.assert_allclose(
        geometry.total_disparity_fraction,
        np.array([[-0.01, 0.0, 0.01]], dtype=np.float64),
        rtol=0.0,
        atol=2e-18,
    )
    assert geometry.source_valid.all()


def test_geometry_frame_owns_immutable_array_storage() -> None:
    canonical = np.array([[0.25, 0.75]], dtype=np.float32)

    geometry = build_relative_geometry(
        canonical,
        canonical.shape,
        stereo_strength=2.0,
        convergence=0.5,
    )
    canonical[:] = 0.5

    assert geometry.near_score.tolist() == [[0.25, 0.75]]
    for values in (
        geometry.near_score,
        geometry.total_disparity_fraction,
        geometry.source_valid,
    ):
        assert not values.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            values[0, 0] = values[0, 0]


def test_metric_resize_does_not_bleed_invalid_zero_into_near_score() -> None:
    inverse = np.array([[4.0, 0.0], [4.0, 0.0]], dtype=np.float32)
    valid = np.array([[True, False], [True, False]], dtype=np.bool_)

    resized_inverse, resized_valid = resize_metric_geometry(inverse, valid, (2, 4))

    assert resized_valid[:, :2].all()
    assert not resized_valid[:, 2:].any()
    np.testing.assert_allclose(resized_inverse[:, :2], 4.0, rtol=0.0, atol=1e-6)
    assert np.count_nonzero(resized_inverse[:, 2:]) == 0


def test_metric_projection_zero_near_far_sign_and_foreground_convention() -> None:
    geometry, stats = build_metric_geometry(
        inverse_depth=np.array([[1.0, 0.5, 0.25]], dtype=np.float32),
        valid=np.ones((1, 3), dtype=np.bool_),
        focal_x_normalized=np.float32(0.5),
        render_shape=(1, 3),
        virtual_baseline_mm=63.0,
        convergence_distance_m=2.0,
        max_disparity_percent=5.0,
        retained_crop_width=3,
    )

    assert geometry.total_disparity_fraction[0, 0] > 0.0
    assert geometry.total_disparity_fraction[0, 1] == 0.0
    assert geometry.total_disparity_fraction[0, 2] < 0.0
    # A three-pixel raster is below the renderer's fixed 1/16-pixel quantization.
    # Repeat it to exercise the eye sign without altering the approved formula.
    expanded = np.tile(geometry.total_disparity_fraction, (1, 100))
    left, right = calculate_geometry_eye_sample_offsets(expanded)
    assert left[0, 0] > right[0, 0]
    assert stats.clamped_fraction == 0.0


def test_metric_projection_clamps_both_signs_in_final_output_coordinates() -> None:
    geometry, stats = build_metric_geometry(
        inverse_depth=np.array([[100.0, 0.001]], dtype=np.float32),
        valid=np.ones((1, 2), dtype=np.bool_),
        focal_x_normalized=np.float32(2.0),
        render_shape=(1, 2),
        virtual_baseline_mm=100.0,
        convergence_distance_m=2.0,
        max_disparity_percent=2.0,
        retained_crop_width=1,
    )

    np.testing.assert_allclose(
        geometry.total_disparity_fraction,
        np.array([[0.01, -0.01]], dtype=np.float64),
        rtol=0.0,
        atol=1e-15,
    )
    assert stats.clamped_pixel_count == 2
    assert stats.clamped_fraction == 1.0


def test_clamped_offsets_do_not_replace_unclamped_inverse_depth_z_order() -> None:
    geometry, _stats = build_metric_geometry(
        inverse_depth=np.array([[10.0, 9.0]], dtype=np.float32),
        valid=np.ones((1, 2), dtype=np.bool_),
        focal_x_normalized=np.float32(10.0),
        render_shape=(1, 2),
        virtual_baseline_mm=100.0,
        convergence_distance_m=1000.0,
        max_disparity_percent=1.0,
        retained_crop_width=2,
    )

    assert geometry.near_score.tolist() == [[10.0, 9.0]]
    assert geometry.total_disparity_fraction[0, 0] == geometry.total_disparity_fraction[0, 1]


@pytest.mark.parametrize("convergence_distance_m", [0.05, 1001.0])
def test_metric_projection_accepts_finite_positive_resolved_auto_convergence(
    convergence_distance_m: float,
) -> None:
    geometry, stats = build_metric_geometry(
        inverse_depth=np.array([[1.0]], dtype=np.float32),
        valid=np.ones((1, 1), dtype=np.bool_),
        focal_x_normalized=np.float32(0.8),
        render_shape=(1, 1),
        virtual_baseline_mm=63.0,
        convergence_distance_m=convergence_distance_m,
        max_disparity_percent=2.0,
        retained_crop_width=1,
    )

    assert np.isfinite(geometry.total_disparity_fraction).all()
    assert stats.valid_pixel_count == 1


def _project_one_valid(*, focal: float, baseline_mm: float):
    return build_metric_geometry(
        inverse_depth=np.array([[1.0]], dtype=np.float32),
        valid=np.ones((1, 1), dtype=np.bool_),
        focal_x_normalized=np.float32(focal),
        render_shape=(1, 1),
        virtual_baseline_mm=baseline_mm,
        convergence_distance_m=2.0,
        max_disparity_percent=100.0,
        retained_crop_width=1,
    )


def test_metric_disparity_scales_linearly_with_focal_and_baseline() -> None:
    base, _ = _project_one_valid(focal=0.5, baseline_mm=50.0)
    focal2, _ = _project_one_valid(focal=1.0, baseline_mm=50.0)
    baseline2, _ = _project_one_valid(focal=0.5, baseline_mm=100.0)

    assert focal2.total_disparity_fraction[0, 0] == pytest.approx(
        2.0 * base.total_disparity_fraction[0, 0]
    )
    assert baseline2.total_disparity_fraction[0, 0] == pytest.approx(
        2.0 * base.total_disparity_fraction[0, 0]
    )


def test_metric_invalid_depth_uses_infinite_background_without_counting() -> None:
    geometry, stats = build_metric_geometry(
        inverse_depth=np.array([[10.0, 0.0]], dtype=np.float32),
        valid=np.array([[False, False]], dtype=np.bool_),
        focal_x_normalized=np.float32(1.0),
        render_shape=(1, 2),
        virtual_baseline_mm=100.0,
        convergence_distance_m=2.0,
        max_disparity_percent=1.0,
        retained_crop_width=2,
    )

    assert geometry.source_valid.all()
    np.testing.assert_array_equal(geometry.near_score, np.zeros((1, 2), dtype=np.float32))
    np.testing.assert_allclose(
        geometry.total_disparity_fraction,
        np.full((1, 2), -0.01, dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert stats == MetricProjectionStats(0, 0, 0.0)


def test_odd_crop_width_converts_final_cap_back_to_source_fraction() -> None:
    geometry, _ = build_metric_geometry(
        inverse_depth=np.full((1, 5), 100.0, dtype=np.float32),
        valid=np.ones((1, 5), dtype=np.bool_),
        focal_x_normalized=np.float32(2.0),
        render_shape=(1, 5),
        virtual_baseline_mm=100.0,
        convergence_distance_m=2.0,
        max_disparity_percent=2.0,
        retained_crop_width=3,
    )

    np.testing.assert_allclose(geometry.total_disparity_fraction, 0.012)


def test_metric_builder_is_independent_of_final_output_width() -> None:
    assert "per_eye_width" not in inspect.signature(build_metric_geometry).parameters
    assert "vr_output_width" not in inspect.signature(build_metric_geometry).parameters
    source_fraction = 0.012
    retained_width = 3
    for final_width in (7, 1920, 3840):
        final_disparity = source_fraction * 5 / retained_width * final_width
        assert final_disparity == pytest.approx(0.02 * final_width)


def test_metric_projection_stats_reject_boolean_fraction() -> None:
    with pytest.raises(TypeError, match="clamped_fraction"):
        MetricProjectionStats(1, 1, True)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"focal_x_normalized": np.float32(0.0)}, "focal"),
        ({"virtual_baseline_mm": -0.1}, "baseline"),
        ({"convergence_distance_m": float("nan")}, "convergence"),
        ({"convergence_distance_m": 0.0}, "convergence"),
        ({"max_disparity_percent": -1.0}, "disparity"),
        ({"retained_crop_width": 0}, "crop width"),
        ({"retained_crop_width": 3}, "crop width"),
    ],
)
def test_metric_projection_rejects_invalid_projection_inputs(
    overrides: dict[str, object], message: str
) -> None:
    arguments: dict[str, object] = {
        "inverse_depth": np.array([[1.0, 2.0]], dtype=np.float32),
        "valid": np.ones((1, 2), dtype=np.bool_),
        "focal_x_normalized": np.float32(1.0),
        "render_shape": (1, 2),
        "virtual_baseline_mm": 63.0,
        "convergence_distance_m": 2.0,
        "max_disparity_percent": 5.0,
        "retained_crop_width": 2,
    }
    arguments.update(overrides)

    with pytest.raises((TypeError, ValueError), match=message):
        build_metric_geometry(**arguments)  # type: ignore[arg-type]
