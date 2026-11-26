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


def test_reserved_manifest_abort_restores_zero_extent_and_retry_preserves_its_identity(tmp_path):
    from tests.unit.test_reservation_protocol import _create_transaction

    transaction = _create_transaction(tmp_path)
    authority = capture_file_identity(transaction.source_path)
    original_size = transaction.source_path.stat().st_size

    def new_manifest():
        return StreamedVrInputManifestV1(
            job_root=tmp_path,
            provider=_provider(1),
            expected_shape=(6, 2),
            reservation_transaction=transaction,
        )

    spool = new_manifest()
    spool.append("frame_000004", _png(6, 2, 31))
    assert spool.path == transaction.source_path
    with pytest.raises(ValueError, match="not an owned work file"):
        _ = spool.owned_source_identity
    spool.abort()
    spool.abort()
    assert transaction.source_path.read_bytes() == bytes(original_size)
    assert capture_file_identity(transaction.source_path) == authority

    retry = new_manifest()
    retry.append("frame_000004", _png(6, 2, 73))
    completed = retry.complete()
    assert retry.complete() is completed
    actual = completed.path.read_bytes()
    extent = completed.staged.raw_byte_count
    assert hashlib.sha256(actual[:extent]).hexdigest() == completed.staged.raw_sha256
    assert actual[extent:] == bytes(original_size - extent)
    retry.cleanup()
    retry.cleanup()
    assert capture_file_identity(transaction.source_path) == authority
    assert transaction.source_path.read_bytes() == actual


@pytest.mark.parametrize("failure", ["fingerprint-write", "hash-read"])
def test_short_manifest_io_discards_owned_temporary_and_allows_a_fresh_attempt(tmp_path, failure):
    options = dict(job_root=tmp_path, provider=_provider(1), expected_shape=(6, 2))
    png = _png(6, 2, 65)
    spool = StreamedVrInputManifestV1(**options)
    spool.append("frame_000004", png)
    original = spool._handle

    class ShortFile:
        def __getattr__(self, name):
            return getattr(original, name)

        def write(self, payload):
            if failure == "fingerprint-write" and original.tell() == spool._fingerprint_offset:
                return original.write(payload[:-1])
            return original.write(payload)

        def read(self, size):
            return b"" if failure == "hash-read" else original.read(size)

    spool._handle = ShortFile()
    with pytest.raises(OSError, match="fingerprint write|written extent"):
        spool.complete()
    assert original.closed
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="closed"):
        spool.complete()
    retry = StreamedVrInputManifestV1(**options)
    retry.append("frame_000004", png)
    try:
        completed = retry.complete()
        assert (
            json.loads(completed.path.read_bytes())["frames"][0]["sha256"]
            == hashlib.sha256(png).hexdigest()
        )
    finally:
        retry.cleanup()


def test_manifest_rejects_foreign_storage_paths_before_allocating_any_file(tmp_path):
    options = dict(job_root=tmp_path, provider=_provider(1), expected_shape=(6, 2))
    outside = tmp_path.parent / "not-owned.json"
    with pytest.raises(ValueError, match="together"):
        StreamedVrInputManifestV1(**options, owned_target_path=outside)
    with pytest.raises(ValueError, match="escaped"):
        StreamedVrInputManifestV1(
            **options, owned_target_path=outside, owned_temporary_path=tmp_path / "scratch"
        )
    assert list(tmp_path.iterdir()) == []


def test_raw_manifest_rejects_premature_completion_and_subsequent_evidence_changes(tmp_path):
    from dataclasses import replace
    from tests.unit.test_streamed_vr_raw import _spool, _output_evidence

    spool = _spool(tmp_path)
    evidence = _output_evidence(tmp_path)
    spool.append("frame_000004", bytes(36))
    with pytest.raises(ValueError, match="incomplete"):
        spool.complete(output_evidence=evidence)
    spool.append("frame_000005", bytes([7]) * 36)
    try:
        completed = spool.complete(output_evidence=evidence)
        prior = completed.path.read_bytes()
        with pytest.raises(ValueError, match="evidence changed"):
            spool.complete(output_evidence=replace(evidence, byte_count=evidence.byte_count + 1))
        assert spool.complete(output_evidence=evidence) is completed
        assert completed.path.read_bytes() == prior
        with pytest.raises(ValueError, match="closed"):
            spool.append("frame_000006", bytes(36))
    finally:
        spool.abort()
    assert not completed.path.exists()
    with pytest.raises(ValueError, match="cleaned"):
        _ = spool.path
    assert (tmp_path / "work.mp4").read_bytes() == b"encoded"


def test_legacy_and_recoverable_manifests_cannot_swap_completion_contracts(tmp_path):
    from tests.unit.test_streamed_vr_raw import _spool, _output_evidence

    old = StreamedVrInputManifestV1(job_root=tmp_path, provider=_provider(1), expected_shape=(6, 2))
    old.append("frame_000004", _png(6, 2, 8))
    with pytest.raises(ValueError, match="legacy"):
        old.complete(output_evidence=_output_evidence(tmp_path))
    raw = _spool(tmp_path)
    raw.append("frame_000004", bytes(36))
    raw.append("frame_000005", bytes(36))
    with pytest.raises(ValueError, match="lacks output evidence"):
        raw.complete()
    assert list(tmp_path.iterdir()) == [tmp_path / "work.mp4"]


def test_failed_header_creation_releases_both_owned_and_reserved_sources(tmp_path, monkeypatch):
    from tests.unit.test_reservation_protocol import _create_transaction

    root = tmp_path / "occupied"
    root.mkdir()
    transaction = _create_transaction(root)
    source = transaction.source_path
    original = source.read_bytes()
    saved_write = StreamedVrInputManifestV1._write

    def out_of_space(self, payload, **kwargs):
        if payload.startswith(b',"frame_names"'):
            raise OSError("header write exhausted storage")
        return saved_write(self, payload, **kwargs)

    monkeypatch.setattr(StreamedVrInputManifestV1, "_write", out_of_space)
    for reserved, target in ((None, tmp_path / "private"), (transaction, root)):
        with pytest.raises(OSError, match="exhausted storage"):
            StreamedVrInputManifestV1(
                job_root=target,
                provider=_provider(1),
                expected_shape=(6, 2),
                reservation_transaction=reserved,
            )
    assert not list((tmp_path / "private").iterdir())
    assert source.read_bytes() == original


def test_manifest_rejects_an_extent_for_another_root_or_an_additional_storage_owner(tmp_path):
    from tests.unit.test_reservation_protocol import _create_transaction

    root = tmp_path / "reserved"
    root.mkdir()
    transaction = _create_transaction(root)
    zeros = transaction.source_path.read_bytes()
    args = dict(provider=_provider(1), expected_shape=(6, 2), reservation_transaction=transaction)
    with pytest.raises(ValueError, match="different job root"):
        StreamedVrInputManifestV1(job_root=tmp_path, **args)
    with pytest.raises(ValueError, match="two storage authorities"):
        StreamedVrInputManifestV1(
            job_root=root,
            owned_target_path=root / "target",
            owned_temporary_path=root / "tmp",
            **args,
        )
    assert transaction.source_path.read_bytes() == zeros


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
