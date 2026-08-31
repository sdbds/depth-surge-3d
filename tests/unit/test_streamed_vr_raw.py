"""Raw transport must bind exact pixels without weakening publication evidence."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import PurePosixPath

import pytest

from src.depth_surge_3d.core.settings import validate_settings
from src.depth_surge_3d.io.file_identity import capture_file_identity
from src.depth_surge_3d.processing.video.encoding_provider import (
    FINAL_ENCODING_JSON_STREAM_BYTES,
    EncodingSequenceProvider,
)
from src.depth_surge_3d.processing.video.final_publication import (
    EncodingInputPublicationSource,
    FinalPublicationValidationError,
    _parse_encoding_input_manifest_stream_v1,
    _parse_input_manifest_projection_v1,
    _final_container_expectation_from_intent_v1,
)
from src.depth_surge_3d.processing.video.streamed_vr_evidence import (
    StreamedVrInvocationV1,
    StreamedVrOutputEvidenceV1,
)
from src.depth_surge_3d.processing.video.streamed_vr_manifest import StreamedVrInputManifestV1
from tests.unit.test_settings_artifact_v5 import _artifact


def _provider():
    return EncodingSequenceProvider.freeze(
        mode="assembled_vr",
        frame_count=2,
        first_index=4,
        frames_parent=PurePosixPath("streamed_vr_frames"),
    )


def _invocation():
    return StreamedVrInvocationV1(
        job_id="11" * 16,
        settings_identity_fingerprint="22" * 32,
        publication_generation="33" * 16,
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
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            "6x2",
            "-i",
            "@image-sequence:sha256=self:view=frames",
            "-vf",
            "setsar=1",
            "-map",
            "0:v:0",
            "-frames:v",
            "2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-preset",
            "medium",
            "@output:path64=b3V0cHV0Lm1wNA",
        ),
    )


def _output_evidence(tmp_path):
    path = tmp_path / "work.mp4"
    path.write_bytes(b"encoded")
    return StreamedVrOutputEvidenceV1(
        reservation_index_fingerprint="44" * 32,
        sibling_relative_path=f".depth-surge-final-v4-{'33' * 16}.tmp.mp4",
        identity=capture_file_identity(path),
        raw_sha256=hashlib.sha256(b"encoded").hexdigest(),
        byte_count=7,
    )


def _spool(tmp_path, **overrides):
    arguments = {
        "job_root": tmp_path,
        "provider": _provider(),
        "expected_shape": (6, 2),
        "invocation": _invocation(),
        "input_format": "rgb24",
    }
    arguments.update(overrides)
    return StreamedVrInputManifestV1(**arguments)


def _complete(tmp_path):
    spool = _spool(tmp_path)
    spool.append("frame_000004", bytes(range(36)))
    spool.append("frame_000005", bytes(reversed(range(36))))
    completed = spool.complete(output_evidence=_output_evidence(tmp_path))
    return spool, completed


def _canonical_fingerprinted(value):
    base = {key: item for key, item in value.items() if key != "fingerprint"}

    def encode(item):
        return json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "ascii"
        )

    return encode({**base, "fingerprint": hashlib.sha256(encode(base)).hexdigest()})


def test_raw_manifest_binds_exact_rgb24_bytes_and_output_evidence(tmp_path):
    spool, completed = _complete(tmp_path)
    raw = completed.path.read_bytes()
    value = json.loads(raw)
    projection = EncodingInputPublicationSource.from_path(completed.path).projection()

    assert value["algorithm_version"] == "encoding-input-sequence-v4"
    assert value["schema_version"] == 3
    assert value["frame_names"] == ["frame_000004", "frame_000005"]
    assert value["frames"] == [
        {
            "byte_count": 36,
            "raw_frame": {"width": 6, "height": 2, "pixel_format": "rgb24"},
            "relative_path": "streamed_vr_frames/frame_000004.rgb",
            "sha256": hashlib.sha256(bytes(range(36))).hexdigest(),
        },
        {
            "byte_count": 36,
            "raw_frame": {"width": 6, "height": 2, "pixel_format": "rgb24"},
            "relative_path": "streamed_vr_frames/frame_000005.rgb",
            "sha256": hashlib.sha256(bytes(reversed(range(36)))).hexdigest(),
        },
    ]
    assert projection.raw_pixel_format == "rgb24"
    assert projection.streamed_invocation == _invocation()
    assert projection.streamed_output == spool._completed_output_evidence
    assert projection.frames_parent == PurePosixPath("streamed_vr_frames")
    assert raw == _canonical_fingerprinted(value)
    assert completed.staged.first_headers == ((6, 2),)
    assert completed.staged.trace.maximum_live_identities == 1
    assert completed.staged.trace.maximum_stream_bytes == 36
    assert not (tmp_path / "streamed_vr_frames").exists()


def test_raw_manifest_replays_across_bounded_reader_chunks(tmp_path):
    count = 8192
    provider = EncodingSequenceProvider.freeze(
        mode="assembled_vr",
        frame_count=count,
        first_index=4,
        frames_parent=PurePosixPath("streamed_vr_frames"),
    )
    invocation = _invocation()
    arguments = list(invocation.resolved_arguments_template)
    arguments[arguments.index("-frames:v") + 1] = str(count)
    spool = _spool(
        tmp_path,
        provider=provider,
        invocation=replace(
            invocation, frame_count=count, resolved_arguments_template=tuple(arguments)
        ),
    )
    for index in range(4, 4 + count):
        spool.append(f"frame_{index:06d}", bytes(36))
    completed = spool.complete(output_evidence=_output_evidence(tmp_path))
    assert completed.staged.raw_byte_count > FINAL_ENCODING_JSON_STREAM_BYTES

    class BoundedReader:
        def __init__(self, handle):
            self.handle = handle

        def read(self, size):
            assert 0 <= size <= FINAL_ENCODING_JSON_STREAM_BYTES
            return self.handle.read(size)

    with completed.path.open("rb") as handle:
        projection = _parse_encoding_input_manifest_stream_v1(
            BoundedReader(handle),
            raw_byte_count=completed.staged.raw_byte_count,
            expected_frame_count=count,
        )
    assert projection.raw_pixel_format == "rgb24"
    assert projection.frame_count == count
    assert projection.first_index == 4
    assert projection.raw_sha256 == completed.staged.raw_sha256
    assert completed.staged.trace.maximum_live_identities == 1
    assert completed.staged.trace.maximum_stream_bytes == 36


@pytest.mark.parametrize("length", [0, 35, 37, 72])
def test_raw_manifest_rejects_non_frame_payload_lengths_without_advancing(tmp_path, length):
    spool = _spool(tmp_path)
    try:
        with pytest.raises(ValueError, match="byte count"):
            spool.append("frame_000004", b"a" * length)
        spool.append("frame_000004", bytes(range(36)))
        spool.append("frame_000005", bytes(range(36)))
        spool.complete(output_evidence=_output_evidence(tmp_path))
    finally:
        spool.abort()


@pytest.mark.parametrize("payload", [bytearray(36), memoryview(bytes(36)), "a" * 36])
def test_raw_manifest_requires_immutable_bytes(tmp_path, payload):
    spool = _spool(tmp_path)
    try:
        with pytest.raises(TypeError, match="immutable bytes"):
            spool.append("frame_000004", payload)
    finally:
        spool.abort()


def test_raw_manifest_rejects_out_of_order_frames_before_consuming_sequence(tmp_path):
    spool = _spool(tmp_path)
    try:
        with pytest.raises(ValueError, match="source order"):
            spool.append("frame_000005", bytes(36))
        spool.append("frame_000004", bytes(36))
        spool.append("frame_000005", bytes(36))
        spool.complete(output_evidence=_output_evidence(tmp_path))
    finally:
        spool.abort()


@pytest.mark.parametrize("input_format", ["bgr24", "rawvideo", None])
def test_raw_manifest_rejects_unknown_format_before_creating_files(tmp_path, input_format):
    with pytest.raises(ValueError):
        _spool(tmp_path, input_format=input_format)
    assert not list(tmp_path.iterdir())


def test_raw_manifest_requires_invocation_before_creating_files(tmp_path):
    with pytest.raises(ValueError, match="invocation"):
        _spool(tmp_path, invocation=None)
    assert not list(tmp_path.iterdir())


def test_raw_manifest_rejects_additional_image_inputs_before_creating_files(tmp_path):
    invocation = _invocation()
    arguments = list(invocation.resolved_arguments_template)
    arguments[-1:-1] = ["-i", f"@image-sequence:sha256={'a' * 64}:view=frames"]
    with pytest.raises(ValueError, match="image input"):
        _spool(
            tmp_path, invocation=replace(invocation, resolved_arguments_template=tuple(arguments))
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "option, value",
    [
        ("-f", "image2pipe"),
        ("-pixel_format", "bgr24"),
        ("-video_size", "3x4"),
        ("-video_size", "06x2"),
        ("-video_size", "6x0"),
        ("-video_size", "4294967296x2"),
    ],
)
def test_raw_manifest_rejects_invalid_input_declarations_before_files(tmp_path, option, value):
    invocation = _invocation()
    arguments = list(invocation.resolved_arguments_template)
    arguments[arguments.index(option) + 1] = value
    changed = replace(invocation, resolved_arguments_template=tuple(arguments))
    with pytest.raises(ValueError):
        _spool(tmp_path, invocation=changed)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("option", ["-f", "-pixel_format", "-video_size"])
@pytest.mark.parametrize("mutation", ["missing", "duplicate", "after_input"])
def test_raw_manifest_rejects_ambiguous_input_option_binding(tmp_path, option, mutation):
    invocation = _invocation()
    arguments = list(invocation.resolved_arguments_template)
    start = arguments.index(option)
    pair = arguments[start : start + 2]
    if mutation in {"missing", "after_input"}:
        del arguments[start : start + 2]
    if mutation == "duplicate":
        arguments[start:start] = pair
    elif mutation == "after_input":
        arguments[-1:-1] = pair
    with pytest.raises(ValueError):
        _spool(
            tmp_path, invocation=replace(invocation, resolved_arguments_template=tuple(arguments))
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("option, value", [("-s", "3x4"), ("-pix_fmt", "bgr24"), ("-c:v", "png")])
def test_raw_manifest_rejects_input_alias_overrides(tmp_path, option, value):
    invocation = _invocation()
    arguments = list(invocation.resolved_arguments_template)
    index = arguments.index("-i")
    arguments[index:index] = [option, value]
    with pytest.raises(ValueError):
        _spool(
            tmp_path, invocation=replace(invocation, resolved_arguments_template=tuple(arguments))
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "field, value",
    [
        ("byte_count", 35),
        ("byte_count", True),
        ("byte_count", 2**64),
        ("sha256", "A" * 64),
        ("relative_path", "streamed_vr_frames/frame_000004.png"),
        ("relative_path", "streamed_vr_frames/frame_000005.rgb"),
        ("raw_frame", {"width": 6, "height": 2, "pixel_format": "bgr24"}),
        ("raw_frame", {"width": 6, "height": True, "pixel_format": "rgb24"}),
        ("raw_frame", {"width": 3, "height": 4, "pixel_format": "rgb24"}),
        ("raw_frame", {"width": 6, "height": 2, "pixel_format": "rgb24", "extra": 0}),
    ],
)
def test_raw_identity_parser_rejects_self_consistently_fingerprinted_invalid_records(
    tmp_path, field, value
):
    _spool_owner, completed = _complete(tmp_path)
    manifest = json.loads(completed.path.read_bytes())
    manifest["frames"][0][field] = value
    with pytest.raises(FinalPublicationValidationError):
        _parse_input_manifest_projection_v1(_canonical_fingerprinted(manifest))


@pytest.mark.parametrize("algorithm, schema", [("v3", 2), ("v4", 2), ("v2", 1)])
def test_raw_identities_cannot_be_relabelled_as_legacy_png_evidence(tmp_path, algorithm, schema):
    _spool_owner, completed = _complete(tmp_path)
    manifest = json.loads(completed.path.read_bytes())
    manifest["algorithm_version"] = f"encoding-input-sequence-{algorithm}"
    manifest["schema_version"] = schema
    with pytest.raises(FinalPublicationValidationError):
        _parse_input_manifest_projection_v1(_canonical_fingerprinted(manifest))


@pytest.mark.parametrize("missing", ["frame_stream_invocation", "streamed_output"])
def test_raw_manifest_cannot_drop_its_recovery_evidence(tmp_path, missing):
    _spool_owner, completed = _complete(tmp_path)
    manifest = json.loads(completed.path.read_bytes())
    del manifest[missing]
    with pytest.raises(FinalPublicationValidationError):
        _parse_input_manifest_projection_v1(_canonical_fingerprinted(manifest))


def test_raw_parser_rejects_direct_stereo_mode_with_a_domain_error(tmp_path):
    _spool_owner, completed = _complete(tmp_path)
    manifest = json.loads(completed.path.read_bytes())
    png_records = []
    for record in manifest["frames"]:
        png_records.append(
            {
                "byte_count": record["byte_count"],
                "sha256": record["sha256"],
                "relative_path": record["relative_path"].replace(".rgb", ".png"),
                "png_header": {"width": 6, "height": 2, "bit_depth": 8, "color_type": 2},
            }
        )
    manifest.update(mode="direct_stereo", frames=None, left=png_records, right=png_records)
    with pytest.raises(FinalPublicationValidationError, match="streamed invocation"):
        _parse_input_manifest_projection_v1(_canonical_fingerprinted(manifest))


def test_raw_manifest_parser_rejects_truncation_and_trailing_data(tmp_path):
    _spool_owner, completed = _complete(tmp_path)
    payload = completed.path.read_bytes()
    for corrupt in (payload[:-1], payload[: len(payload) // 2], payload + b" "):
        with pytest.raises(FinalPublicationValidationError):
            _parse_input_manifest_projection_v1(corrupt)


@pytest.mark.parametrize(
    "option, value", [("-pixel_format", "bgr24"), ("-video_size", "3x4"), ("-f", "image2pipe")]
)
def test_raw_parser_independently_revalidates_input_declarations(tmp_path, option, value):
    _spool_owner, completed = _complete(tmp_path)
    manifest = json.loads(completed.path.read_bytes())
    arguments = manifest["frame_stream_invocation"]["resolved_arguments_template"]
    arguments[arguments.index(option) + 1] = value
    with pytest.raises(FinalPublicationValidationError):
        _parse_input_manifest_projection_v1(_canonical_fingerprinted(manifest))


def _png_input_arguments(arguments):
    result = list(arguments)
    index = result.index("-f")
    result[index : index + 6] = ["-f", "image2pipe", "-vcodec", "png"]
    return tuple(result)


def test_raw_parser_rejects_a_complete_png_input_declaration(tmp_path):
    _spool_owner, completed = _complete(tmp_path)
    manifest = json.loads(completed.path.read_bytes())
    invocation = manifest["frame_stream_invocation"]
    invocation["resolved_arguments_template"] = _png_input_arguments(
        invocation["resolved_arguments_template"]
    )
    with pytest.raises(FinalPublicationValidationError, match="transport"):
        _parse_input_manifest_projection_v1(_canonical_fingerprinted(manifest))


def test_legacy_png_evidence_cannot_claim_a_raw_invocation(tmp_path):
    _spool_owner, completed = _complete(tmp_path)
    manifest = json.loads(completed.path.read_bytes())
    manifest["algorithm_version"] = "encoding-input-sequence-v3"
    manifest["schema_version"] = 2
    for record in manifest["frames"]:
        del record["raw_frame"]
        record["png_header"] = {"width": 6, "height": 2, "bit_depth": 8, "color_type": 2}
        record["relative_path"] = record["relative_path"].replace(".rgb", ".png")
    with pytest.raises(FinalPublicationValidationError, match="PNG invocation"):
        _parse_input_manifest_projection_v1(_canonical_fingerprinted(manifest))


def _container_settings(width=6, height=2):
    properties = dict(_artifact().to_object()["video_properties"])
    properties["frame_count"] = 2
    return _artifact(
        video_properties=properties,
        processing_settings=validate_settings(
            {
                "keep_intermediates": True,
                "vr_resolution": f"custom:{width}x{height}",
                "per_eye_width": width // 2,
                "per_eye_height": height,
                "vr_output_width": width,
                "vr_output_height": height,
            },
            source="explicit",
        ),
    )


def test_raw_transport_reconstructs_container_dimensions_rate_and_count(tmp_path):
    _spool_owner, completed = _complete(tmp_path)
    projection = EncodingInputPublicationSource.from_path(completed.path).projection()
    expectation = _final_container_expectation_from_intent_v1(
        settings=_container_settings(),
        input_projection=projection,
        arguments=_invocation().materialized_arguments(completed.staged.fingerprint),
    )
    assert expectation.width == 6
    assert expectation.height == 2
    assert expectation.frame_count == 2
    assert expectation.fps == "24/1"


@pytest.mark.parametrize("change", ["png_command", "png_marker", "settings_shape"])
def test_raw_container_contract_rejects_transport_or_dimension_confusion(tmp_path, change):
    _spool_owner, completed = _complete(tmp_path)
    projection = EncodingInputPublicationSource.from_path(completed.path).projection()
    arguments = _invocation().materialized_arguments(completed.staged.fingerprint)
    settings = _container_settings()
    if change == "png_command":
        arguments = _png_input_arguments(arguments)
    elif change == "png_marker":
        projection = replace(projection, raw_pixel_format=None)
    else:
        settings = _container_settings(8, 2)
    with pytest.raises(FinalPublicationValidationError):
        _final_container_expectation_from_intent_v1(
            settings=settings,
            input_projection=projection,
            arguments=arguments,
        )
