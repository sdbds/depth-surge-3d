"""Fast standard-zipfile metric geometry NPZ tests."""

from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pytest

from src.depth_surge_3d.io import metric_npz


def test_write_and_read_metric_npz_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "frame_000001.npz"
    valid = np.array([[True, False, True], [True, True, False]], dtype=np.bool_)
    inverse = np.where(valid, np.array([[0.5, 0.0, 1.25], [0.2, 2.0, 0.0]], dtype=np.float32), 0.0)
    focal = np.float32(0.85)

    trace = metric_npz.write_metric_npz(target, inverse, valid, focal)
    assert trace.replaced is True
    assert trace.archive_bytes > 0

    audit = metric_npz.audit_metric_npz(target, expected_shape=(2, 3), compute_sha256=True)
    assert np.array_equal(audit.inverse_depth, inverse)
    assert np.array_equal(audit.valid, valid)
    assert audit.focal_x_normalized == focal
    assert len(audit.sha256) == 64

    inv2, val2, focal2 = metric_npz.read_metric_npz(
        target, expected_shape=(2, 3), job_root=tmp_path
    )
    assert np.array_equal(inv2, inverse)
    assert np.array_equal(val2, valid)
    assert focal2 == focal


def test_metric_npz_rejects_invalid_content_and_bad_members(tmp_path: Path) -> None:
    target = tmp_path / "bad.npz"
    valid = np.ones((2, 2), dtype=np.bool_)
    bad_inverse = np.array([[1.0, np.nan], [0.5, 0.25]], dtype=np.float32)
    with pytest.raises(metric_npz.MetricNpzContentError):
        metric_npz.write_metric_npz(target, bad_inverse, valid, np.float32(1.0))

    with zipfile.ZipFile(target, "w") as zf:
        zf.writestr("unexpected.npy", b"123")
    with pytest.raises(metric_npz.MetricNpzFormatError):
        metric_npz.audit_metric_npz(target)


def test_metric_npz_rejects_pickled_object_arrays(tmp_path: Path) -> None:
    target = tmp_path / "pickle.npz"
    np.savez(
        target,
        inverse_depth=np.array([object()], dtype=object),
        valid=np.ones((1, 1), dtype=np.bool_),
        focal_x_normalized=np.float32(1.0),
    )
    with pytest.raises(metric_npz.MetricNpzFormatError):
        metric_npz.audit_metric_npz(target)
