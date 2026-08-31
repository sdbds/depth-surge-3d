"""Exact counters and bounded histogram serialization for diagnostic output."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.depth_surge_3d.io.streamed_canonical_json import build_streamed_canonical_value
from src.depth_surge_3d.rendering.quality import diagnostic_transaction as transaction
from src.depth_surge_3d.rendering.quality import diagnostics, native_kd
from src.depth_surge_3d.rendering.quality.types import (
    CompactStereoRenderResult,
    DenseHistogramPairs,
    FastEyeDiagnostics,
    HoleRunDiagnostics,
)


STATE_NAMES = (
    "prefill_partial",
    "prefill_full",
    "local_filled",
    "post_local_residual",
    "exemplar_filled",
    "fallback_filled",
    "final_unresolved",
)


@pytest.fixture
def native_counter():
    if not native_kd.native_available():
        pytest.skip("Native diagnostic primitives are unavailable")

    def count(values):
        counter = getattr(native_kd._require_native(), "count_repair_state_bits", None)
        assert callable(counter), "Native single-pass repair-state counting is unavailable"
        return counter(values)

    return count


def _weighted_states():
    values = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 255], dtype=np.uint8)
    return np.repeat(values, [3, 1, 2, 3, 4, 5, 6, 7, 8, 1]).reshape(1, -1)


def test_native_state_counts_preserve_bit_order_and_read_only_input(native_counter):
    values = _weighted_states()
    before = values.copy()
    values.setflags(write=False)
    assert native_counter(values) == (2, 3, 4, 5, 6, 7, 8)
    np.testing.assert_array_equal(values, before)
    assert not values.flags.writeable


def test_native_state_counts_cover_every_byte_without_counting_reserved_bit(native_counter):
    values = np.tile(np.arange(256, dtype=np.uint8), (3, 1))
    assert native_counter(values) == (384,) * 7
    assert native_counter(np.full((3, 7), 128, dtype=np.uint8)) == (0,) * 7


@pytest.mark.parametrize("shape", [(0, 7), (4, 0), (0, 0)])
def test_native_state_counts_accept_empty_planes(native_counter, shape):
    assert native_counter(np.empty(shape, dtype=np.uint8)) == (0,) * 7


@pytest.mark.parametrize("size", [1, 7, 8, 9, 15, 16, 17, 31, 32, 33])
def test_native_state_counts_include_short_and_odd_tails(native_counter, size):
    values = np.zeros((1, size), dtype=np.uint8)
    values[0, -1] = 0x49
    assert native_counter(values) == (1, 0, 0, 1, 0, 0, 1)


def test_native_state_counts_are_exact_above_float32_integer_range(native_counter):
    count = 2**24 + 1
    values = np.full((1, count), 0x05, dtype=np.uint8)
    assert native_counter(values) == (count, 0, count, 0, 0, 0, 0)


@pytest.mark.parametrize("shape", [(3, 17), (31, 33), (127, 129)])
@pytest.mark.parametrize("distribution", ["uniform", "sparse", "all_bits", "zeros"])
def test_native_state_counts_match_numpy_for_random_planes(native_counter, shape, distribution):
    rng = np.random.default_rng(20260831)
    values = rng.integers(0, 256, size=shape, dtype=np.uint8)
    if distribution == "sparse":
        values[rng.random(shape) < 0.98] = 0
    elif distribution == "all_bits":
        values.fill(255)
    elif distribution == "zeros":
        values.fill(0)
    expected = tuple(int(np.count_nonzero(values & np.uint8(1 << bit))) for bit in range(7))
    assert native_counter(values) == expected


@pytest.mark.parametrize("dtype", [np.bool_, np.uint16, np.int8, np.float32])
def test_native_state_counts_reject_implicit_dtype_conversion(native_counter, dtype):
    with pytest.raises(TypeError, match="C-contiguous uint8"):
        native_counter(np.ones((2, 3), dtype=dtype))


@pytest.mark.parametrize("layout", ["one_dimension", "three_dimensions", "reversed", "fortran"])
def test_native_state_counts_reject_non_raster_or_noncontiguous_input(native_counter, layout):
    values = np.arange(12, dtype=np.uint8).reshape(3, 4)
    if layout == "one_dimension":
        values = values.reshape(-1)
    elif layout == "three_dimensions":
        values = values.reshape(1, 3, 4)
    elif layout == "reversed":
        values = values[:, ::-1]
    else:
        values = np.asfortranarray(values)
    with pytest.raises(TypeError, match="C-contiguous uint8"):
        native_counter(values)


@pytest.mark.parametrize("native", [None, SimpleNamespace()])
def test_state_counts_keep_numpy_fallback_for_unavailable_or_older_native(monkeypatch, native):
    monkeypatch.setattr(native_kd, "_NATIVE", native)
    assert diagnostics._repair_state_counts(_weighted_states()) == dict(
        zip(STATE_NAMES, (2, 3, 4, 5, 6, 7, 8), strict=True)
    )


@pytest.mark.parametrize(
    "dtype",
    [np.dtype(np.uint8, metadata={"source": "test"}), np.dtype(np.uint8).newbyteorder("|")],
    ids=["metadata", "equivalent_non_singleton"],
)
def test_state_counts_preserve_equivalent_noncanonical_dtypes(native_counter, dtype):
    assert dtype == np.dtype(np.uint8)
    assert dtype is not np.dtype(np.uint8)
    values = np.empty(_weighted_states().shape, dtype=dtype)
    values[:] = _weighted_states()
    assert values.dtype is dtype
    assert native_counter(_weighted_states()) == (2, 3, 4, 5, 6, 7, 8)
    assert diagnostics._repair_state_counts(values) == dict(
        zip(STATE_NAMES, (2, 3, 4, 5, 6, 7, 8), strict=True)
    )


@pytest.mark.parametrize("layout", ["fortran", "reversed", "strided"])
def test_state_counts_preserve_noncontiguous_numpy_fallback(monkeypatch, layout):
    values = np.arange(48, dtype=np.uint8).reshape(6, 8)
    if layout == "fortran":
        values = np.asfortranarray(values)
    elif layout == "reversed":
        values = values[::-1, ::-1]
    else:
        values = values[::2, ::2]
    expected = {
        name: int(np.count_nonzero(values & np.uint8(1 << bit)))
        for bit, name in enumerate(STATE_NAMES)
    }

    def unexpected_native_call(*_args):
        raise AssertionError("Noncontiguous planes must retain the NumPy path")

    monkeypatch.setattr(
        native_kd, "_NATIVE", SimpleNamespace(count_repair_state_bits=unexpected_native_call)
    )
    assert diagnostics._repair_state_counts(values) == expected


def test_state_counts_preserve_array_subclass_bitwise_behavior():
    class LogicalStates(np.ndarray):
        def __and__(self, bit):
            return np.full(self.shape, int(bit) == 1, dtype=np.uint8)

    values = np.full((2, 3), 255, dtype=np.uint8).view(LogicalStates)
    assert diagnostics._repair_state_counts(values) == dict(
        zip(STATE_NAMES, (6, 0, 0, 0, 0, 0, 0), strict=True)
    )


def test_fast_frame_statistics_do_not_repeat_full_plane_bit_scans(monkeypatch, native_counter):
    repair = np.array([[0, 1, 2, 4, 5, 8, 64, 72]], dtype=np.uint8)
    image = np.zeros((1, 8, 3), dtype=np.uint8)
    coverage = np.full(repair.shape, 16, dtype=np.uint8)
    holes = HoleRunDiagnostics(np.zeros(144, dtype=np.uint64), np.zeros(9, dtype=np.uint64))
    fast = FastEyeDiagnostics(holes, 17, 32)
    compact = CompactStereoRenderResult(
        left_image=image,
        right_image=image.copy(),
        left_coverage_count=coverage,
        right_coverage_count=coverage.copy(),
        left_repair_bits=repair,
        right_repair_bits=repair.copy(),
        left_hole_runs=holes,
        right_hole_runs=holes,
        left_fast_diagnostics=fast,
        right_fast_diagnostics=fast,
    )

    def redundant_scan(*_args, **_kwargs):
        raise AssertionError("Supported native diagnostics must not rescan entire bit planes")

    monkeypatch.setattr(diagnostics.np, "count_nonzero", redundant_scan)
    stats = diagnostics.build_quality_frame_stats(
        compact,
        frame_name="frame_000001",
        geometry_mode="relative",
        occlusion_fill="background",
        render_mode="fast",
    )
    for eye in stats["eyes"].values():
        assert eye["state_pixel_counts"] == dict(
            zip(STATE_NAMES, (2, 1, 2, 2, 0, 0, 2), strict=True)
        )
        assert eye["backend_pixel_counts"]["local"] == 2


def test_dense_histogram_encoding_uses_bounded_native_nonzero_scans(monkeypatch):
    bins = np.zeros(8200, dtype=np.uint64)
    bins[[0, 4095, 4096, 8199]] = [1, 2**53 + 1, 2**64 - 1, 9]
    before = bins.copy()
    observed = []
    original = np.flatnonzero

    def observe(values):
        observed.append(values.size)
        return original(values)

    monkeypatch.setattr(transaction.np, "flatnonzero", observe)
    raw = transaction.canonical_json_bytes(DenseHistogramPairs(bins))
    assert raw == b"[[0,1],[4095,9007199254740993],[4096,18446744073709551615],[8199,9]]"
    assert observed and max(observed) <= 4096
    assert sum(observed) == bins.size
    np.testing.assert_array_equal(bins, before)


@pytest.mark.parametrize("count", [0, 1, 4095, 4096, 4097, 12345])
def test_empty_dense_histograms_encode_without_zero_pairs(count):
    value = DenseHistogramPairs(np.zeros(count, dtype=np.uint64))
    assert transaction.canonical_json_bytes(value) == b"[]"
    streamed = build_streamed_canonical_value(transaction._streamable_diagnostics_value(value))
    assert b"".join(streamed.chunks()) == b"[]"


def test_dense_and_streamed_histograms_keep_exact_nested_canonical_bytes():
    bins = np.array([0, 2**64 - 1, 0, 2**53 + 1, 1], dtype=np.uint64)
    bins.setflags(write=False)
    value = {"z": [DenseHistogramPairs(bins), None], "a": 2**64 - 1}
    expected = (
        b'{"a":18446744073709551615,"z":[[[1,18446744073709551615],'
        b"[3,9007199254740993],[4,1]],null]}"
    )
    assert transaction.canonical_json_bytes(value) == expected
    streamed = build_streamed_canonical_value(transaction._streamable_diagnostics_value(value))
    assert b"".join(streamed.chunks()) == expected
    assert streamed.raw_byte_count == len(expected)


@pytest.mark.parametrize("size", [0, 17, 4097, 8200])
@pytest.mark.parametrize("nonzero_fraction", [0.01, 0.5, 1.0])
def test_dense_histogram_paths_match_explicit_random_pairs(size, nonzero_fraction):
    rng = np.random.default_rng(20260831)
    bins = rng.integers(1, np.iinfo(np.uint64).max, size=size, dtype=np.uint64, endpoint=True)
    bins[rng.random(size) > nonzero_fraction] = 0
    explicit = [[index, int(count)] for index, count in enumerate(bins) if count]
    expected = transaction.canonical_json_bytes(explicit)
    value = DenseHistogramPairs(bins)
    assert transaction.canonical_json_bytes(value) == expected
    streamed = build_streamed_canonical_value(transaction._streamable_diagnostics_value(value))
    assert b"".join(streamed.chunks()) == expected


def test_dense_histogram_array_subclasses_keep_scalar_access_semantics():
    class VirtualCounts(np.ndarray):
        def __getitem__(self, index):
            if isinstance(index, (int, np.integer)) and index == 1:
                return np.uint64(7)
            return super().__getitem__(index)

    value = DenseHistogramPairs(np.zeros(4, dtype=np.uint64).view(VirtualCounts))
    assert transaction.canonical_json_bytes(value) == b"[[1,7]]"
    streamed = build_streamed_canonical_value(transaction._streamable_diagnostics_value(value))
    assert b"".join(streamed.chunks()) == b"[[1,7]]"


def test_streamed_histogram_replays_current_bins_instead_of_caching_pairs():
    bins = np.zeros(4097, dtype=np.uint64)
    value = transaction._streamable_diagnostics_value(DenseHistogramPairs(bins))
    assert b"".join(build_streamed_canonical_value(value).chunks()) == b"[]"
    bins[-1] = np.uint64(2**64 - 1)
    assert (
        b"".join(build_streamed_canonical_value(value).chunks()) == b"[[4096,18446744073709551615]]"
    )
