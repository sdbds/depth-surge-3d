"""Production contracts for bounded Quality metric content auditing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.io import content_manifest as content_reader
from src.depth_surge_3d.io import metric_npz
from src.depth_surge_3d.processing.frames.stereo_input_manifest import (
    publish_stereo_input_manifest_v1,
)
from src.depth_surge_3d.rendering.quality.diagnostic_transaction import (
    project_quality_input_manifest,
)


def _payload(path: Path, scale: float = 1.0) -> None:
    inverse = np.array([[0.0, scale], [scale * 2.0, scale * 4.0]], dtype=np.float32)
    valid = np.array([[False, True], [True, True]], dtype=np.bool_)
    metric_npz.write_metric_npz(path, inverse, valid, np.float32(1.5))


def test_relative_quality_input_manifest_hashes_complete_png_content(
    tmp_path: Path,
) -> None:
    guide_dir = tmp_path / "00_original_frames"
    geometry_dir = tmp_path / "03_depth_canonical"
    guide_dir.mkdir()
    geometry_dir.mkdir()
    guide = guide_dir / "frame_000000.png"
    geometry = geometry_dir / "frame_000000.png"
    assert cv2.imwrite(str(guide), np.full((3, 5, 3), 17, dtype=np.uint8))
    assert cv2.imwrite(str(geometry), np.full((2, 4), 32768, dtype=np.uint16))

    result = content_reader.build_quality_input_manifest(
        job_root=tmp_path,
        frame_files=[guide],
        geometry_files=[geometry],
        geometry_mode="relative",
        relative_encoding_scale=np.float32(65535.0),
        metric_native_shape=None,
    )

    assert result.payload["frame_names"] == ["frame_000000"]
    assert result.payload["guides"][0]["png_header"] == {
        "width": 5,
        "height": 3,
        "bit_depth": 8,
        "color_type": 2,
    }
    assert result.payload["geometry"][0]["png_header"]["bit_depth"] == 16
    assert result.raw == json.dumps(
        result.payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    assert result.raw_sha256 == hashlib.sha256(result.raw).hexdigest()

    geometry_identity = {
        "geometry_mode": "relative",
        "relative_encoding_scale_float32_bits": result.payload[
            "relative_encoding_scale_float32_bits"
        ],
        "geometry": result.payload["geometry"],
    }
    def canonical(value):
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    assert result.payload["native_geometry_fingerprint"] == hashlib.sha256(
        canonical(geometry_identity)
    ).hexdigest()
    assert result.payload["native_geometry_fingerprint"] != hashlib.sha256(
        canonical({**geometry_identity, "geometry_mode": "metric"})
    ).hexdigest()
    assert result.payload["native_geometry_fingerprint"] != hashlib.sha256(
        canonical(
            {
                **geometry_identity,
                "relative_encoding_scale_float32_bits": "3f800000",
            }
        )
    ).hexdigest()


def test_quality_manifest_streams_directly_from_durable_upstream_records(tmp_path: Path) -> None:
    guide_dir = tmp_path / "00_original_frames"
    geometry_dir = tmp_path / "03_depth_canonical"
    guide_dir.mkdir()
    geometry_dir.mkdir()
    guide = guide_dir / "frame_000001.png"
    geometry = geometry_dir / "frame_000001.png"
    assert cv2.imwrite(str(guide), np.full((3, 5, 3), 17, dtype=np.uint8))
    assert cv2.imwrite(str(geometry), np.full((2, 4), 32768, dtype=np.uint16))
    upstream = publish_stereo_input_manifest_v1(
        manifest_path=geometry_dir / "stereo_input_manifest.jsonl",
        job_root=tmp_path,
        guide_files=[guide],
        geometry_files=[geometry],
        frame_count=1,
        geometry_mode="relative",
        source_metadata={"fingerprint": "ab" * 32, "encoding_scale": 65535.0},
    )

    result = content_reader.build_streamed_quality_input_manifest(upstream)
    output = tmp_path / "04_stereo_diagnostics" / "quality_input_manifest.json"
    result.publish(output)
    projection = project_quality_input_manifest(output)
    payload = json.loads(output.read_text(encoding="ascii"))
    legacy = content_reader.build_quality_input_manifest(
        job_root=tmp_path,
        frame_files=[guide],
        geometry_files=[geometry],
        geometry_mode="relative",
        relative_encoding_scale=np.float32(65535.0),
        metric_native_shape=None,
    )

    assert payload["frame_names"] == ["frame_000001"]
    assert payload["guides"][0]["sha256"] == hashlib.sha256(guide.read_bytes()).hexdigest()
    assert payload["geometry"][0]["sha256"] == hashlib.sha256(
        geometry.read_bytes()
    ).hexdigest()
    assert result.raw_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert output.read_bytes() == legacy.raw
    assert not hasattr(result, "payload")
    assert projection.raw_sha256 == result.raw_sha256
    assert tuple(projection.iter_frame_names()) == ("frame_000001",)


def test_quality_projection_rejects_the_first_frame_over_control_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.depth_surge_3d.rendering.quality import diagnostic_transaction

    guide_dir = tmp_path / "00_original_frames"
    geometry_dir = tmp_path / "03_depth_canonical"
    guide_dir.mkdir()
    geometry_dir.mkdir()
    guides = []
    geometry = []
    for index in range(2):
        guide = guide_dir / f"frame_{index:06d}.png"
        depth = geometry_dir / f"frame_{index:06d}.png"
        assert cv2.imwrite(str(guide), np.full((3, 5, 3), 17, dtype=np.uint8))
        assert cv2.imwrite(str(depth), np.full((2, 4), 32768, dtype=np.uint16))
        guides.append(guide)
        geometry.append(depth)
    result = content_reader.build_quality_input_manifest(
        job_root=tmp_path,
        frame_files=guides,
        geometry_files=geometry,
        geometry_mode="relative",
        relative_encoding_scale=np.float32(65535.0),
        metric_native_shape=None,
    )
    path = tmp_path / "quality_input_manifest.json"
    path.write_bytes(result.raw)
    monkeypatch.setattr(diagnostic_transaction, "STEREO_CONTROL_FRAME_CAP", 1)

    with pytest.raises(diagnostic_transaction.StereoControlBudgetError, match="control cap"):
        diagnostic_transaction.project_quality_input_manifest(path)


def test_metric_quality_input_manifest_uses_closed_npz_identity(tmp_path: Path) -> None:
    guide_dir = tmp_path / "00_original_frames"
    geometry_dir = tmp_path / "03_metric_geometry"
    guide_dir.mkdir()
    geometry_dir.mkdir()
    guide = guide_dir / "frame_000000.png"
    geometry = geometry_dir / "frame_000000.npz"
    assert cv2.imwrite(str(guide), np.full((3, 5, 3), 17, dtype=np.uint8))
    _payload(geometry)

    result = content_reader.build_quality_input_manifest(
        job_root=tmp_path,
        frame_files=[guide],
        geometry_files=[geometry],
        geometry_mode="metric",
        relative_encoding_scale=None,
        metric_native_shape=(2, 2),
    )

    identity = result.payload["geometry"][0]
    assert identity["archive_form"] == "owned-zip32-v1"
    assert identity["native_shape"] == [2, 2]
    assert identity["members"] == [
        "inverse_depth.npy",
        "valid.npy",
        "focal_x_normalized.npy",
    ]
    assert result.payload["relative_encoding_scale_float32_bits"] is None


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
