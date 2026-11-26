"""Native CPU Quality kernel invariants and binary evidence tests."""

import hashlib
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

from src.depth_surge_3d.rendering.quality import native_kd, region_solver


@pytest.mark.parametrize("shape", [(1, 17), (11, 1), (11, 13)])
def test_native_regions_build_canonical_components_and_ranks(shape):
    rng = np.random.default_rng(99001)
    disparity = rng.choice([0, 0.5, 1.0, np.nextafter(1.0, 2.0), 2], size=shape)
    valid = rng.random(shape) > 0.2
    near = rng.choice(np.array([-0.0, 0.125, 0.5, 1.0], dtype=np.float32), size=shape)
    native = region_solver.build_low_resolution_regions(disparity, valid, near)
    repeat = region_solver.build_low_resolution_regions(disparity, valid, near)
    np.testing.assert_array_equal(native.region_map, repeat.region_map)
    np.testing.assert_array_equal(native.canonical_keys, repeat.canonical_keys)
    np.testing.assert_array_equal(native.region_rank_bits, repeat.region_rank_bits)
    assert not native.region_map.flags.writeable
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
