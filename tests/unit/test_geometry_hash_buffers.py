"""Region integrity hashes retain byte semantics without a full bytes copy."""

import hashlib
import tracemalloc

import numpy as np
import pytest

from src.depth_surge_3d.rendering.quality import geometry


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(np.arange(24, dtype=np.uint32).reshape(4, 6), id="contiguous"),
        pytest.param(np.arange(24, dtype=np.uint32).reshape(4, 6)[:, ::2], id="strided"),
        pytest.param(np.arange(24, dtype=np.uint32).reshape(4, 6)[::-1], id="reversed"),
        pytest.param(np.arange(24, dtype=np.uint32).reshape(4, 6).T, id="transposed"),
        pytest.param(
            np.asfortranarray(np.arange(24, dtype=np.uint32).reshape(4, 6)),
            id="fortran",
        ),
        pytest.param(np.array(42, dtype=np.uint32), id="scalar"),
        pytest.param(np.empty((0,), dtype=np.uint32), id="empty"),
        pytest.param(np.empty((3, 0, 2), dtype=np.uint32), id="empty-multidimensional"),
        pytest.param(np.arange(24, dtype=">u4").reshape(4, 6), id="big-endian"),
    ],
)
@pytest.mark.parametrize("readonly", [False, True], ids=["writable", "readonly"])
def test_array_hash_matches_c_order_bytes_for_supported_layouts(values, readonly):
    values = values.view()
    if readonly:
        values.flags.writeable = False
    expected = hashlib.sha256(values.tobytes(order="C")).hexdigest()

    assert geometry._array_sha256(values) == expected


def test_array_hash_reads_current_bytes_after_mutation():
    values = np.arange(24, dtype=np.uint32).reshape(4, 6)
    readonly = values.view()
    readonly.flags.writeable = False
    before = geometry._array_sha256(readonly)

    values[2, 3] = 999

    after = geometry._array_sha256(readonly)
    assert after == hashlib.sha256(values.tobytes(order="C")).hexdigest()
    assert after != before


def test_contiguous_array_hash_does_not_allocate_an_array_sized_bytes_copy():
    values = np.arange(1080 * 1920, dtype=np.uint32).reshape(1080, 1920)
    expected = hashlib.sha256(values.tobytes(order="C")).hexdigest()
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        baseline, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        actual = geometry._array_sha256(values)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not was_tracing:
            tracemalloc.stop()

    assert actual == expected
    # Allow interpreter bookkeeping, but not the 8,294,400-byte input copy.
    assert peak - baseline < 256 * 1024
