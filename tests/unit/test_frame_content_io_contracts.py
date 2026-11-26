"""Image/metric manifests and streamed JSON must reject bad inputs without replacing good output."""

import hashlib
import json

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.io import content_manifest as content
from src.depth_surge_3d.io import metric_npz, streamed_canonical_json as streamed
from src.depth_surge_3d.io.durable_publish import OwnedSiblingTemporary


@pytest.fixture
def frame_evidence(tmp_path):
    guides = tmp_path / "guides"
    depths = tmp_path / "depth"
    guides.mkdir()
    depths.mkdir()
    guide = guides / "frame_000005.png"
    depth = depths / guide.name
    assert cv2.imwrite(str(guide), np.full((3, 5, 3), (21, 33, 44), np.uint8))
    assert cv2.imwrite(str(depth), np.arange(4, dtype=np.uint16).reshape(2, 2))
    metric = depth.with_suffix(".npz")
    metric_npz.write_metric_npz(
        metric, np.ones((2, 2), np.float32), np.ones((2, 2), np.bool_), np.float32(1.5)
    )
    return (
        dict(
            job_root=tmp_path,
            frame_files=[guide],
            geometry_files=[depth],
            geometry_mode="relative",
            relative_encoding_scale=np.float32(65535),
            metric_native_shape=None,
        ),
        metric,
    )


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("incomplete", "equal positive"),
        ("duplicate", "unique"),
        ("invalid-mode", "geometry_mode"),
        ("no-scale", "encoding scale"),
        ("bad-scale", "finite and positive"),
        ("signature", "invalid IHDR"),
        ("dimensions", "unsupported IHDR"),
        ("gray-guide", "format"),
        ("wrong-depth-name", "filename"),
        ("no-metric-shape", "native shape"),
        ("metric-with-scale", "relative encoding scale"),
        ("wrong-metric-name", "filename"),
    ],
)
def test_unsuitable_frame_or_sensor_evidence_cannot_produce_a_manifest(
    frame_evidence, fault, reason
):
    args, metric = frame_evidence
    guide, depth = args["frame_files"][0], args["geometry_files"][0]
    if fault == "incomplete":
        args["geometry_files"] = []
    elif fault == "duplicate":
        args["frame_files"] *= 2
        args["geometry_files"] *= 2
    elif fault == "invalid-mode":
        args["geometry_mode"] = "meters"
    elif fault in {"no-scale", "bad-scale"}:
        args["relative_encoding_scale"] = {"no-scale": None, "bad-scale": np.float32(np.inf)}[fault]
    elif fault == "signature":
        guide.write_bytes(b"part of header")
    elif fault == "dimensions":
        raw = bytearray(guide.read_bytes())
        raw[16:20] = bytes(4)
        guide.write_bytes(raw)
    elif fault == "gray-guide":
        assert cv2.imwrite(str(guide), np.full((3, 5), 17, np.uint8))
    elif fault == "wrong-depth-name":
        args["geometry_files"] = [depth.rename(depth.with_name("frame_000006.png"))]
    else:
        args["geometry_mode"] = "metric"
        args["geometry_files"] = [metric]
        args["metric_native_shape"] = None if fault == "no-metric-shape" else (2, 2)
        if fault == "wrong-metric-name":
            args["relative_encoding_scale"] = None
            args["geometry_files"] = [metric.rename(metric.with_name("frame_000006.npz"))]
    with pytest.raises(ValueError, match=reason):
        content.build_quality_input_manifest(**args)


def test_audited_metric_identity_outlives_released_array_ownership(frame_evidence):
    args, metric = frame_evidence
    with content.audit_metric_payload(
        metric, expected_shape=(2, 2), job_root=args["job_root"]
    ) as audit:
        identity = audit.identity()
        assert audit.owned_payload_bytes == 20
    audit.release()
    assert audit.released and audit.owned_payload_bytes == 0
    assert (
        identity.native_shape == (2, 2)
        and identity.sha256 == hashlib.sha256(metric.read_bytes()).hexdigest()
    )
    for access in (lambda: audit.native_shape, audit.identity, audit.__enter__):
        with pytest.raises(RuntimeError, match="released"):
            access()


def test_content_publication_requires_an_explicit_typed_audit_contract(frame_evidence):
    from src.depth_surge_3d.processing.frames.stereo_input_manifest import (
        publish_stereo_input_manifest_v1,
    )

    args, _ = frame_evidence
    root = args["job_root"]
    manifest = publish_stereo_input_manifest_v1(
        manifest_path=root / "declared.jsonl",
        job_root=root,
        guide_files=args["frame_files"],
        geometry_files=args["geometry_files"],
        frame_count=1,
        geometry_mode="relative",
        source_metadata={"fingerprint": "ab" * 32, "encoding_scale": 65535.0},
    )
    with pytest.raises(TypeError, match="descriptor"):
        content.build_streamed_quality_input_manifest(manifest.path)
    with pytest.raises(TypeError, match="boolean"):
        content.build_streamed_quality_input_manifest(manifest, validate_payloads="false")
    target = root / "result.json"
    audited = content.build_quality_input_manifest(**args)
    with pytest.raises(TypeError, match="QualityInputManifestResult"):
        content.publish_quality_input_manifest(target, audited.payload)
    assert not target.exists()
    eligible = content.build_streamed_quality_input_manifest(manifest)
    eligible.publish(target)
    assert target.read_bytes() == audited.raw


@pytest.mark.parametrize("kind", ["content", "streamed"])
def test_partial_manifest_write_preserves_previous_document_and_retry_is_byte_exact(
    frame_evidence, monkeypatch, kind
):
    args, _ = frame_evidence
    target = args["job_root"] / "attestation.json"
    target.write_bytes(b"previous data")
    if kind == "content":
        plan = content.build_quality_input_manifest(**args)

        def publish():
            content.publish_quality_input_manifest(target, plan)

        expected = plan.raw
    else:
        value = streamed.build_streamed_canonical_value({"records": [{}, [1, 2]]})

        def publish():
            value.publish(target)

        expected = b'{"records":[{},[1,2]]}'
    original = OwnedSiblingTemporary.create

    class ShortWrite:
        def __init__(self, handle):
            self.base = handle

        def write(self, data):
            return self.base.write(data[:-1])

        def __getattr__(self, name):
            return getattr(self.base, name)

    def short_file(path):
        owned = original(path)
        owned.handle = ShortWrite(owned.handle)
        return owned

    prior_names = set(target.parent.iterdir())
    with monkeypatch.context() as patch:
        patch.setattr(OwnedSiblingTemporary, "create", short_file)
        with pytest.raises(OSError, match="short"):
            publish()
    assert target.read_bytes() == b"previous data"
    assert set(target.parent.iterdir()) == prior_names
    publish()
    assert target.read_bytes() == expected


def test_streamed_json_keeps_empty_collections_and_rejects_nonstring_object_keys():
    value = {"B": [{}], "A": streamed.ReplayableJsonArray(lambda: iter(()))}
    for builder, indent in (
        (streamed.build_streamed_canonical_value, None),
        (streamed.build_streamed_pretty_json_value, 2),
    ):
        document = builder(value)
        text = json.dumps(
            {"A": [], "B": [{}]},
            sort_keys=True,
            indent=indent,
            separators=(",", ":") if indent is None else None,
        )
        assert b"".join(document.chunks()) == text.encode("ascii")
        assert document.raw_sha256 == hashlib.sha256(text.encode("ascii")).hexdigest()
        with pytest.raises(streamed.CanonicalJsonError, match="keys"):
            builder({3: "not a string"})
    with pytest.raises(TypeError, match="callable"):
        streamed.ReplayableJsonArray(iter(()))


@pytest.mark.parametrize(
    ("fields", "arrays", "fingerprint", "error"),
    [
        ({}, {}, None, "array keys"),
        ({}, {"": list}, None, "array keys"),
        ({}, {"items": []}, None, "replayable"),
        ({"items": 1}, {"items": list}, None, "stream-owned"),
        ({}, {"items": list}, "items", "fingerprint key"),
        ({"identity": 0}, {"items": list}, "identity", "stream-owned"),
    ],
)
def test_a_canonical_object_cannot_assign_two_streams_ownership_of_the_same_key(
    fields, arrays, fingerprint, error
):
    with pytest.raises((TypeError, streamed.CanonicalJsonError), match=error):
        streamed.build_streamed_canonical_object_arrays(
            static_fields=fields, array_items=arrays, fingerprint_key=fingerprint
        )
