"""Bit-exact frozen-reference and allocation contracts for depth conversion."""

import tracemalloc

import numpy as np
import pytest

from src.depth_surge_3d.inference.depth.types import DepthRepresentation
from src.depth_surge_3d.processing.frames.depth_normalizer import (
    SceneDepthBounds,
    canonicalize_depth,
    depth_to_score,
    encode_canonical_png,
)


def _old_score(values, representation):
    source = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(source)
    score = np.zeros(source.shape, dtype=np.float32)
    if representation is DepthRepresentation.METRIC_DEPTH:
        valid = finite & (source > 0.0)
        np.divide(np.float32(1.0), source, out=score, where=valid)
    elif representation is DepthRepresentation.INVERSE_DEPTH:
        valid = finite
        score[valid] = source[valid]
    elif representation is DepthRepresentation.RELATIVE_DEPTH:
        valid = finite
        score[valid] = -source[valid]
    else:
        raise TypeError(f"Unsupported depth representation: {representation!r}")
    return score, valid


def _old_canonical(values, representation, bounds):
    score, valid = _old_score(values, representation)
    canonical = np.full(score.shape, np.float32(0.5), dtype=np.float32)
    if bounds.low == bounds.high:
        return canonical
    scaled = (score[valid] - np.float32(bounds.low)) / np.float32(bounds.high - bounds.low)
    canonical[valid] = np.clip(scaled, 0.0, 1.0).astype(np.float32, copy=False)
    return canonical


def _old_encode(values):
    canonical = np.asarray(values, dtype=np.float32)
    if not np.isfinite(canonical).all():
        raise ValueError("Canonical disparity must be finite before encoding")
    return np.rint(np.clip(canonical, 0.0, 1.0) * np.float32(65535.0)).astype(np.uint16)


def _inputs():
    rng = np.random.default_rng(48291)
    values = rng.standard_normal((31, 37)).astype(np.float32)
    values.flat[:9] = [0.0, -0.0, np.nan, np.inf, -np.inf, 1e-40, -1e-40, 1e30, -1e30]
    readonly = values.copy()
    readonly.flags.writeable = False
    return [
        values,
        values.astype(np.float64),
        values.T,
        values[::-2, ::3],
        readonly,
        np.empty((0, 3), np.float32),
        np.array(0.327811, np.float32),
        np.array(-0.0, np.float32),
        np.array(np.nan, np.float32),
        np.array([1, 2, -3], np.int16),
    ]


@pytest.mark.parametrize("values", _inputs())
@pytest.mark.parametrize("representation", list(DepthRepresentation))
def test_conversion_preserves_frozen_bits_and_input(values, representation):
    before = values.tobytes()
    with np.errstate(all="ignore"):
        actual, valid = depth_to_score(values, representation)
        expected, expected_valid = _old_score(values, representation)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()
        assert not np.shares_memory(actual, values)
        np.testing.assert_array_equal(valid, expected_valid)
        for low, high in [(0.0, 1.0), (-1.7311, 2.9987), (3.0, 3.0), (0.0, 1e-40)]:
            bounds = SceneDepthBounds(low, high)
            actual = canonicalize_depth(values, representation, bounds)
            expected = _old_canonical(values, representation, bounds)
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            assert actual.tobytes() == expected.tobytes()
            assert not np.shares_memory(actual, values)
    assert values.tobytes() == before


@pytest.mark.parametrize("values", _inputs())
def test_encoding_preserves_frozen_pixels_errors_and_input(values):
    before = values.tobytes()
    if not np.isfinite(values).all():
        with pytest.raises(ValueError, match="must be finite"):
            encode_canonical_png(values)
    else:
        actual = encode_canonical_png(values)
        expected = _old_encode(values)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()
    assert values.tobytes() == before


def test_encoding_rounding_thresholds_match_frozen_pixels():
    midpoint = (np.arange(65535, dtype=np.float32) + np.float32(0.5)) / np.float32(65535)
    values = np.stack([np.nextafter(midpoint, -np.inf), midpoint, np.nextafter(midpoint, np.inf)])
    np.testing.assert_array_equal(encode_canonical_png(values), _old_encode(values))


def test_encoding_finite_layouts_readonly_and_scalar_thresholds():
    base = np.linspace(-0.1, 1.1, 1023, dtype=np.float32).reshape(31, 33)
    readonly = base.copy()
    readonly.flags.writeable = False
    for values in [base, base.T, base[::-2, ::3], readonly, base.astype(np.float64)]:
        before = values.tobytes()
        actual = encode_canonical_png(values)
        assert actual.tobytes() == _old_encode(values).tobytes()
        assert not np.shares_memory(actual, values)
        assert values.tobytes() == before
    for value in base.flat:
        scalar = np.array(value)
        actual = encode_canonical_png(scalar)
        expected = _old_encode(scalar)
        assert type(actual) is type(expected)
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("representation", list(DepthRepresentation))
def test_random_float_bits_match_frozen_canonicalization(representation):
    rng = np.random.default_rng(8081)
    values = rng.integers(0, 2**32, size=65537, dtype=np.uint32).view(np.float32)
    before = values.tobytes()
    with np.errstate(all="ignore"):
        for low, high in [(-12.871, 33.479), (-1e38, 1e38), (1e-45, 1e-39)]:
            bounds = SceneDepthBounds(low, high)
            actual = canonicalize_depth(values, representation, bounds)
            expected = _old_canonical(values, representation, bounds)
            assert actual.tobytes() == expected.tobytes()
    assert values.tobytes() == before


def test_unsupported_representation_still_rejected():
    values = np.array([1.0], np.float32)
    with pytest.raises(TypeError, match="Unsupported depth representation"):
        depth_to_score(values, "metric_depth")
    with pytest.raises(TypeError, match="Unsupported depth representation"):
        canonicalize_depth(values, "metric_depth", SceneDepthBounds(1.0, 1.0))


def _peak_bytes(function, values):
    function(values[:8])
    tracemalloc.start()
    try:
        probe = np.empty_like(values)
        assert tracemalloc.get_traced_memory()[1] >= probe.nbytes
        del probe
        tracemalloc.reset_peak()
        function(values)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("representation", list(DepthRepresentation))
def test_canonicalization_uses_at_most_three_float_buffers(representation):
    values = np.linspace(0.125, 0.875, 1024 * 1024, dtype=np.float32)
    bounds = SceneDepthBounds(-0.371, 8.712)
    peak = _peak_bytes(lambda data: canonicalize_depth(data, representation, bounds), values)
    # Three float buffers plus two masks and generous fixed interpreter overhead.
    assert peak <= values.size * 14 + 256 * 1024


@pytest.mark.parametrize(
    "representation", [DepthRepresentation.INVERSE_DEPTH, DepthRepresentation.RELATIVE_DEPTH]
)
def test_score_uses_one_float_buffer_and_mask(representation):
    values = np.ones(1024 * 1024, dtype=np.float32)
    peak = _peak_bytes(lambda data: depth_to_score(data, representation), values)
    assert peak <= values.size * 5 + 256 * 1024


def test_encoding_reuses_one_float_scratch_buffer():
    values = np.linspace(0.0, 1.0, 1024 * 1024, dtype=np.float32)
    peak = _peak_bytes(encode_canonical_png, values)
    assert peak <= values.size * 6 + 256 * 1024
