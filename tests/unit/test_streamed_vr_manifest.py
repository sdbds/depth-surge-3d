from __future__ import annotations

import hashlib
import io
import json
from pathlib import PurePosixPath

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.processing.video.encoding_provider import (
    EncodingSequenceProvider,
    stage_encoding_input_manifest,
)
from src.depth_surge_3d.processing.video.final_publication import (
    EncodingInputPublicationSource,
    _stream_input_manifest_identity,
)
from src.depth_surge_3d.io.file_identity import capture_file_identity
from src.depth_surge_3d.processing.video.streamed_vr_evidence import (
    StreamedVrInvocationV1,
    StreamedVrOutputEvidenceV1,
)
from src.depth_surge_3d.processing.video.streamed_vr_manifest import (
    StreamedVrInputManifestV1,
)
from src.depth_surge_3d.rendering.quality.diagnostic_transaction import (
    self_fingerprint,
)


def _png(width: int, height: int, value: int) -> bytes:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(
        ".png",
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, 1],
    )
    assert ok
    return encoded.tobytes()


def _provider(frame_count: int = 2) -> EncodingSequenceProvider:
    return EncodingSequenceProvider.freeze(
        mode="assembled_vr",
        frame_count=frame_count,
        first_index=4,
        frames_parent=PurePosixPath("streamed_vr_frames"),
    )


def test_streamed_manifest_binds_the_exact_piped_png_payloads(tmp_path) -> None:
    first = _png(6, 2, 17)
    second = _png(6, 2, 29)
    spool = StreamedVrInputManifestV1(
        job_root=tmp_path,
        provider=_provider(),
        expected_shape=(6, 2),
    )

    spool.append("frame_000004", first)
    spool.append("frame_000005", second)
    completed = spool.complete()

    raw = completed.path.read_bytes()
    value = json.loads(raw)
    assert value["fingerprint"] == self_fingerprint(value)
    assert value["frame_names"] == ["frame_000004", "frame_000005"]
    assert value["left"] is None
    assert value["right"] is None
    assert value["mode"] == "assembled_vr"
    assert value["frames"] == [
        {
            "byte_count": len(first),
            "png_header": {
                "bit_depth": 8,
                "color_type": 2,
                "height": 2,
                "width": 6,
            },
            "relative_path": "streamed_vr_frames/frame_000004.png",
            "sha256": hashlib.sha256(first).hexdigest(),
        },
        {
            "byte_count": len(second),
            "png_header": {
                "bit_depth": 8,
                "color_type": 2,
                "height": 2,
                "width": 6,
            },
            "relative_path": "streamed_vr_frames/frame_000005.png",
            "sha256": hashlib.sha256(second).hexdigest(),
        },
    ]
    assert completed.staged.raw_sha256 == hashlib.sha256(raw).hexdigest()
    assert completed.staged.raw_byte_count == len(raw)
    assert completed.staged.first_headers == ((6, 2),)
    assert _stream_input_manifest_identity(
        tmp_path,
        completed.path,
        expected_frame_count=2,
    ) == (
        value["fingerprint"],
        hashlib.sha256(raw).hexdigest(),
        len(raw),
    )

    spool.cleanup()
    assert not completed.path.exists()


def test_streamed_manifest_rejects_wrong_order_and_shape_then_cleans(tmp_path) -> None:
    spool = StreamedVrInputManifestV1(
        job_root=tmp_path,
        provider=_provider(),
        expected_shape=(6, 2),
    )

    with pytest.raises(ValueError, match="source order"):
        spool.append("frame_000005", _png(6, 2, 17))
    with pytest.raises(ValueError, match="shape"):
        spool.append("frame_000004", _png(8, 2, 17))

    spool.abort()
    assert not any(tmp_path.iterdir())


def test_streamed_manifest_is_byte_exact_with_file_backed_manifest(tmp_path) -> None:
    provider = _provider()
    first = _png(6, 2, 17)
    second = _png(6, 2, 29)
    file_root = tmp_path / "file"
    frames = file_root / "streamed_vr_frames"
    frames.mkdir(parents=True)
    (frames / "frame_000004.png").write_bytes(first)
    (frames / "frame_000005.png").write_bytes(second)
    sink = io.BytesIO()
    staged = stage_encoding_input_manifest(
        provider,
        job_root=file_root,
        sink=sink,
        per_eye_size=(3, 2),
        vr_format="side_by_side",
    )

    spool = StreamedVrInputManifestV1(
        job_root=tmp_path / "streamed",
        provider=provider,
        expected_shape=(6, 2),
    )
    spool.append("frame_000004", first)
    spool.append("frame_000005", second)
    completed = spool.complete()

    assert completed.path.read_bytes() == sink.getvalue()
    assert completed.staged.fingerprint == staged.fingerprint
    assert completed.staged.raw_sha256 == staged.raw_sha256


def test_streamed_manifest_v3_binds_invocation_and_output_evidence(tmp_path) -> None:
    publication_generation = "33" * 16
    output_token = "@output:path64=b3V0cHV0Lm1wNA"
    invocation = StreamedVrInvocationV1(
        job_id="11" * 16,
        settings_identity_fingerprint="22" * 32,
        publication_generation=publication_generation,
        frame_count=2,
        first_index=4,
        expected_shape=(6, 2),
        resolved_encoder="libx264",
        resolved_fps="24/1",
        audio_end_sample=None,
        resolved_arguments_template=(
            "ffmpeg",
            "-framerate",
            "24/1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-i",
            "@image-sequence:sha256=self:view=frames",
            "-frames:v",
            "2",
            "-c:v",
            "libx264",
            output_token,
        ),
    )
    output = tmp_path / "work.mp4"
    output.write_bytes(b"encoded")
    evidence = StreamedVrOutputEvidenceV1(
        reservation_index_fingerprint="44" * 32,
        sibling_relative_path=(f".depth-surge-final-v4-{publication_generation}.tmp.mp4"),
        identity=capture_file_identity(output),
        raw_sha256=hashlib.sha256(b"encoded").hexdigest(),
        byte_count=len(b"encoded"),
    )
    spool = StreamedVrInputManifestV1(
        job_root=tmp_path,
        provider=_provider(),
        expected_shape=(6, 2),
        invocation=invocation,
    )
    spool.append("frame_000004", _png(6, 2, 17))
    spool.append("frame_000005", _png(6, 2, 29))

    completed = spool.complete(output_evidence=evidence)

    value = json.loads(completed.path.read_bytes())
    projection = EncodingInputPublicationSource.from_path(
        completed.path,
        expected_frame_count=2,
    ).projection(expected_frame_count=2)
    assert value["algorithm_version"] == "encoding-input-sequence-v3"
    assert value["schema_version"] == 2
    assert projection.streamed_invocation == invocation
    assert projection.streamed_output == evidence
    assert invocation.materialized_arguments(completed.staged.fingerprint)[-1] == (output_token)
