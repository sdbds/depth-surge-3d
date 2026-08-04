"""Pure canonical depth conversion tests."""

import numpy as np
import pytest

from src.depth_surge_3d.inference.depth.types import DepthBatch, DepthRepresentation
from src.depth_surge_3d.processing.frames.depth_normalizer import (
    SceneDepthBounds,
    canonicalize_depth,
    canonicalize_single_scene,
    decode_canonical_png,
    depth_to_score,
    encode_canonical_png,
)


def test_metric_depth_uses_safe_reciprocal_and_rejects_nonpositive_values() -> None:
    values = np.array([[1.0, 2.0, 0.0, -1.0, np.nan]], dtype=np.float32)

    score, valid = depth_to_score(values, DepthRepresentation.METRIC_DEPTH)

    np.testing.assert_array_equal(valid, [[True, True, False, False, False]])
    np.testing.assert_allclose(score[0, :2], [1.0, 0.5])
    assert np.all(score[~valid] == 0.0)


def test_inverse_depth_passes_every_finite_value_through() -> None:
    values = np.array([[-1.0, 0.0, 2.0, np.inf]], dtype=np.float32)

    score, valid = depth_to_score(values, DepthRepresentation.INVERSE_DEPTH)

    np.testing.assert_array_equal(valid, [[True, True, True, False]])
    np.testing.assert_array_equal(score[valid], [-1.0, 0.0, 2.0])


def test_relative_depth_negates_values_and_keeps_near_zero_valid() -> None:
    values = np.array([[0.0, 0.8]], dtype=np.float32)
    bounds = SceneDepthBounds(low=-0.8, high=0.0)

    canonical = canonicalize_depth(values, DepthRepresentation.RELATIVE_DEPTH, bounds)

    np.testing.assert_allclose(canonical, [[1.0, 0.0]])


def test_canonicalization_clips_outliers_and_neutralizes_invalid_values() -> None:
    values = np.array([[-2.0, 0.0, 0.5, 1.0, 2.0, np.nan]], dtype=np.float32)
    bounds = SceneDepthBounds(low=0.0, high=1.0)

    canonical = canonicalize_depth(values, DepthRepresentation.INVERSE_DEPTH, bounds)

    np.testing.assert_allclose(canonical, [[0.0, 0.0, 0.5, 1.0, 1.0, 0.5]])
    assert canonical.dtype == np.float32


def test_flat_bounds_produce_exact_neutral_midpoint() -> None:
    values = np.array([[3.0, 4.0, np.nan]], dtype=np.float32)

    canonical = canonicalize_depth(
        values,
        DepthRepresentation.INVERSE_DEPTH,
        SceneDepthBounds(low=3.0, high=3.0),
    )

    np.testing.assert_array_equal(canonical, np.full_like(values, 0.5))


def test_single_scene_canonicalization_uses_explicit_representation() -> None:
    batch = DepthBatch(
        values=np.array([[[0.0, 0.25, 0.75, 1.0]]], dtype=np.float32),
        representation=DepthRepresentation.RELATIVE_DEPTH,
    )

    canonical = canonicalize_single_scene(batch)

    assert canonical.shape == batch.values.shape
    assert canonical.dtype == np.float32
    assert canonical[0, 0, 0] == 1.0
    assert canonical[0, 0, -1] == 0.0


def test_single_scene_flat_depth_is_neutral() -> None:
    batch = DepthBatch(
        values=np.full((1, 3, 4), 7.0, dtype=np.float32),
        representation=DepthRepresentation.INVERSE_DEPTH,
    )

    canonical = canonicalize_single_scene(batch)

    np.testing.assert_array_equal(canonical, np.full(batch.values.shape, 0.5, np.float32))


def test_scene_bounds_reject_reversed_or_nonfinite_ranges() -> None:
    with pytest.raises(ValueError, match="low must not exceed high"):
        SceneDepthBounds(low=2.0, high=1.0)
    with pytest.raises(ValueError, match="finite"):
        SceneDepthBounds(low=np.nan, high=1.0)


def test_uint16_midpoint_encoding_has_documented_quantization() -> None:
    values = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)

    encoded = encode_canonical_png(values)
    decoded = decode_canonical_png(encoded)

    np.testing.assert_array_equal(encoded, [[0, 32768, 65535]])
    assert encoded.dtype == np.uint16
    assert decoded.dtype == np.float32
    assert decoded[0, 1] == np.float32(32768 / 65535)
    np.testing.assert_allclose(decoded[[0], [0, 2]], [0.0, 1.0])


def test_decoder_requires_uint16_canonical_storage() -> None:
    with pytest.raises(TypeError, match="uint16"):
        decode_canonical_png(np.zeros((2, 2), dtype=np.uint8))
