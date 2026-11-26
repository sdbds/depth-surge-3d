"""Differential tests for the shipped scalar and prebuilt CPU Quality kernels."""

import hashlib
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

from src.depth_surge_3d.rendering.quality import (
    indexed_heap,
    native_kd,
    region_solver,
)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("capacity", True),
        ("capacity", 3),
        ("distance", np.ones((1, 2), np.int64)),
        ("distance", np.ones(2, np.uint64)),
        ("owner", np.ones((1, 3), np.uint32)),
        ("region_rank_bits", np.empty(0, np.uint32)),
    ],
)
def test_geodesic_heap_rejects_invalid_storage_before_queue_creation(field, invalid):
    args = dict(
        distance=np.ones((1, 2), np.uint64),
        owner=np.ones((1, 2), np.uint32),
        region_rank_bits=np.array([0, 1], np.uint32),
        capacity=2,
    )
    args[field] = invalid
    with pytest.raises((TypeError, ValueError), match=field):
        indexed_heap.IndexedRegionHeap(**args)


def test_rejected_heap_inserts_do_not_consume_positions_or_change_priority():
    heap = indexed_heap.IndexedRegionHeap(
        np.zeros((1, 2), np.uint64),
        np.array([[1, 0]], np.uint32),
        np.array([0, 2], np.uint32),
        capacity=2,
    )
    with pytest.raises(IndexError, match="empty"):
        heap.pop()
    for pixel, error in [(False, TypeError), (-1, ValueError), (1, ValueError)]:
        with pytest.raises(error, match="pixel"):
            heap.push_or_decrease(pixel)
    assert len(heap) == heap.insertions == heap.pop_count == 0
    assert heap.position.tolist() == [[-1, -1]]
    heap.push_or_decrease(0)
    assert heap.pop() == 0


@pytest.mark.parametrize("shape", [(1, 17), (11, 1), (11, 13)])
def test_native_regions_match_canonical_scalar_components(shape):
    rng = np.random.default_rng(99001)
    # Adjacent disparities equal to and just above the cutoff exercise opposite graph edges.
    disparity = rng.choice([0, 0.5, 1.0, np.nextafter(1.0, 2.0), 2], size=shape)
    valid = rng.random(shape) > 0.2
    near = rng.choice(np.array([-0.0, 0.125, 0.5, 1.0], dtype=np.float32), size=shape)
    scalar = region_solver._build_low_resolution_regions_scalar(disparity, valid, near)
    native = region_solver.build_low_resolution_regions(disparity, valid, near)
    np.testing.assert_array_equal(native.region_map, scalar.region_map)
    np.testing.assert_array_equal(native.canonical_keys, scalar.canonical_keys)
    np.testing.assert_array_equal(native.region_rank_bits, scalar.region_rank_bits)
    for region in np.unique(native.region_map):
        members = np.flatnonzero(native.region_map == region)
        assert native.canonical_keys[region] == members[0]
        assert native.region_rank_bits[region] == np.float32(
            max(0.0, near.flat[members].max())
        ).view(np.uint32)


def test_binary_evidence_hashes_the_actually_loaded_native_module():
    path = Path(find_spec("_quality_kd_native").origin).resolve()
    evidence = native_kd.native_binary_evidence()
    assert Path(evidence["path"]).samefile(path)
    assert evidence["byte_count"] == path.stat().st_size > 0
    with path.open("rb") as binary:
        assert evidence["sha256"] == hashlib.sha256(binary.read()).hexdigest()
    assert evidence["build_info"] == native_kd.native_build_info()
