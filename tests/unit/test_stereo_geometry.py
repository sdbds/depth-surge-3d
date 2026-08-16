"""Validation and relative-conversion tests for backend-neutral stereo geometry."""

from __future__ import annotations

import numpy as np
import pytest

from src.depth_surge_3d.rendering.stereo_geometry import (
    StereoGeometryFrame,
    build_relative_geometry,
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
