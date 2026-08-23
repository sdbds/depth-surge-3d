"""Task 0 contracts for bounded Quality metric content auditing."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.quality_task0 import content_reader, metric_npz


def _payload(path: Path, scale: float = 1.0) -> None:
    inverse = np.array([[0.0, scale], [scale * 2.0, scale * 4.0]], dtype=np.float32)
    valid = np.array([[False, True], [True, True]], dtype=np.bool_)
    metric_npz.write_metric_npz(path, inverse, valid, np.float32(1.5))


def test_metric_audit_owns_exactly_one_inverse_and_valid_array(tmp_path: Path) -> None:
    path = tmp_path / "frame_000000.npz"
    _payload(path)

    frame = content_reader.audit_metric_payload(path, expected_shape=(2, 2))

    assert frame.owned_payload_bytes == 5 * 4
    assert frame.inverse_depth.flags.c_contiguous
    assert frame.valid.flags.c_contiguous
    assert frame.inverse_depth.dtype == np.float32
    assert frame.valid.dtype == np.bool_
    assert frame.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert frame.maximum_stream_bytes <= content_reader.QUALITY_CONTENT_STREAM_BYTES
    assert frame.stream_workspace_peak_bytes <= content_reader.QUALITY_CONTENT_STREAM_BYTES
    frame.release()
    assert frame.released


def test_sequence_releases_each_frame_before_opening_the_next(tmp_path: Path) -> None:
    paths = [tmp_path / f"frame_{index:06d}.npz" for index in range(3)]
    for index, path in enumerate(paths, start=1):
        _payload(path, float(index))

    result = content_reader.audit_metric_sequence(paths, expected_shape=(2, 2))

    assert result.trace.active_frames_peak == 1
    assert result.trace.active_payload_bytes_peak == 5 * 4
    assert result.trace.stream_workspace_peak_bytes <= content_reader.QUALITY_CONTENT_STREAM_BYTES
    for index in range(3):
        release = result.trace.events.index(f"release:{index}")
        if index + 1 < len(paths):
            next_open = result.trace.events.index(f"open:{index + 1}")
            assert release < next_open
    assert [identity.byte_count for identity in result.identities] == [
        path.stat().st_size for path in paths
    ]


def test_audit_does_not_call_numpy_load_or_public_metric_constructor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "frame_000000.npz"
    _payload(path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("forbidden generic loader")

    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(
        "src.depth_surge_3d.processing.frames.metric_geometry.MetricGeometryFrame",
        forbidden,
    )

    frame = content_reader.audit_metric_payload(path, expected_shape=(2, 2))

    assert frame.focal_x_normalized == np.float32(1.5)
    frame.release()


@pytest.mark.parametrize("failing_call", [1, 2])
def test_allocation_failure_is_typed_as_host_budget_error(
    tmp_path: Path,
    failing_call: int,
) -> None:
    path = tmp_path / "frame_000000.npz"
    _payload(path)
    calls = 0

    def fail_selected(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == failing_call:
            raise MemoryError("injected allocation failure")
        return np.empty(shape, dtype=dtype)

    with pytest.raises(content_reader.QualityHostBudgetError, match="metric-content-audit"):
        content_reader.audit_metric_payload(
            path,
            expected_shape=(2, 2),
            allocator=fail_selected,
        )


@pytest.mark.parametrize("focal", [np.float32(0.0), np.float32(np.nan)])
def test_streamed_value_validation_rejects_invalid_focal(
    tmp_path: Path,
    focal: np.float32,
) -> None:
    path = tmp_path / "invalid-focal.npz"
    inverse = np.array([[0.0, 1.0]], dtype=np.float32)
    valid = np.array([[False, True]], dtype=np.bool_)
    metric_npz.write_metric_npz(path, inverse, valid, focal)

    with pytest.raises(metric_npz.MetricNpzContentError, match="focal"):
        content_reader.audit_metric_payload(path, expected_shape=(1, 2))


@pytest.mark.parametrize(
    ("inverse", "valid", "message"),
    [
        (np.array([[0.0, np.nan]], dtype=np.float32), np.array([[False, True]]), "finite"),
        (np.array([[-0.0, 1.0]], dtype=np.float32), np.array([[False, True]]), "positive zero"),
        (np.array([[0.0, 0.0]], dtype=np.float32), np.array([[False, True]]), "positive"),
    ],
)
def test_streamed_value_validation_rejects_invalid_metric_semantics(
    tmp_path: Path,
    inverse: np.ndarray,
    valid: np.ndarray,
    message: str,
) -> None:
    path = tmp_path / "invalid.npz"
    metric_npz.write_metric_npz(path, inverse, valid.astype(np.bool_), np.float32(1.0))

    with pytest.raises(metric_npz.MetricNpzContentError, match=message):
        content_reader.audit_metric_payload(path, expected_shape=inverse.shape)
