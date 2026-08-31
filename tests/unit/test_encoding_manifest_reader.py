"""Token batching must preserve bounded reads and exact manifest hash boundaries."""

from __future__ import annotations

import hashlib
import io
import json
import math

import pytest

from src.depth_surge_3d.processing.video import final_publication as publication
from tests.unit.test_final_publication import _encode_input_manifest, _input_manifest_value


class _CountedDigest:
    def __init__(self):
        self.digest = hashlib.sha256()
        self.updates = 0

    def update(self, chunk):
        self.updates += 1
        self.digest.update(chunk)

    def hexdigest(self):
        return self.digest.hexdigest()


class _FragmentedInput(io.BytesIO):
    def __init__(self, raw, fragment_bytes):
        super().__init__(raw)
        self.fragment_bytes = fragment_bytes
        self.largest_request = 0

    def read(self, size):
        assert 0 <= size <= publication._ENCODING_MANIFEST_STREAM_BYTES
        self.largest_request = max(self.largest_request, size)
        return super().read(min(size, self.fragment_bytes))


@pytest.mark.parametrize("object_value", [False, True])
def test_tokens_batch_hash_updates_without_hashing_unconsumed_bytes(object_value):
    payload = "a" * (2 * publication._ENCODING_MANIFEST_STREAM_BYTES + 13)
    value = {"value": payload} if object_value else payload
    token = json.dumps(value, separators=(",", ":")).encode("ascii")
    prefix, suffix = b"prefix:", b',"unconsumed":true'
    reader = publication._EncodingManifestReader(
        io.BytesIO(prefix + token + suffix), raw_byte_count=None
    )
    reader.raw_digest = _CountedDigest()
    reader.self_digest = _CountedDigest()
    reader.expect(prefix)
    reader.include_in_self_digest = True

    result = reader.json_object_token() if object_value else reader.json_string_token()

    assert result == token
    assert reader.consumed == len(prefix + token)
    assert reader.peek_byte() == ord(",")
    assert reader.raw_digest.hexdigest() == hashlib.sha256(prefix + token).hexdigest()
    assert reader.self_digest.hexdigest() == hashlib.sha256(token).hexdigest()
    maximum_updates = math.ceil(len(prefix + token) / publication._ENCODING_MANIFEST_STREAM_BYTES)
    assert reader.raw_digest.updates <= maximum_updates + 1
    assert reader.self_digest.updates <= maximum_updates


@pytest.mark.parametrize("fragment_bytes", [1, 2, 3, 7, 64 * 1024])
@pytest.mark.parametrize(
    "value",
    [
        "",
        "braces }{ stay in the string",
        'quote " then slash \\ then quote "',
        "\\" * 7 + '"' + "\\" * 8,
        "line\nreturn\rtab\t\u4e2d\U0001f642",
        {},
        {"a": {"b": [{"c": '}\\"{'}, {}]}, "d": "\\"},
        {"a": "before\\", "b": "after", "c": "\\" * 8 + '"'},
    ],
)
def test_token_state_survives_short_reads_and_escaped_delimiters(fragment_bytes, value):
    token = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    raw = b"abc" + token + b",next"
    handle = _FragmentedInput(raw, fragment_bytes)
    reader = publication._EncodingManifestReader(handle, raw_byte_count=len(raw))
    reader.expect(b"abc")
    reader.include_in_self_digest = True

    result = reader.json_object_token() if isinstance(value, dict) else reader.json_string_token()

    assert result == token
    assert json.loads(result) == value
    assert reader.consumed == len(token) + 3
    assert reader.peek_byte() == ord(",")
    assert reader.raw_digest.hexdigest() == hashlib.sha256(b"abc" + token).hexdigest()
    assert reader.self_digest.hexdigest() == hashlib.sha256(token).hexdigest()
    reader.include_in_self_digest = False
    reader.expect(b",next")
    assert reader.peek_byte() is None
    assert reader.raw_digest.hexdigest() == hashlib.sha256(raw).hexdigest()
    assert reader.self_digest.hexdigest() == hashlib.sha256(token).hexdigest()


@pytest.mark.parametrize("object_value", [False, True])
@pytest.mark.parametrize("tail", ['"braces }{', '\\"braces }{', "\\", "\\\\"])
def test_escape_at_default_chunk_boundary_preserves_batched_hashing(object_value, tail):
    opening, closing = (b'{"a":"', b'"}') if object_value else (b'"', b'"')
    padding = "a" * (publication._ENCODING_MANIFEST_STREAM_BYTES - len(opening) - 1)
    encoded_tail = json.dumps(tail).encode("ascii")[1:-1]
    token = opening + padding.encode("ascii") + encoded_tail + closing
    assert token[publication._ENCODING_MANIFEST_STREAM_BYTES - 1] == ord("\\")
    reader = publication._EncodingManifestReader(
        io.BytesIO(token + b',"next":1'), raw_byte_count=None
    )
    reader.raw_digest = _CountedDigest()
    reader.self_digest = _CountedDigest()
    reader.include_in_self_digest = True

    result = reader.json_object_token() if object_value else reader.json_string_token()

    assert result == token
    assert json.loads(result) == ({"a": padding + tail} if object_value else padding + tail)
    assert reader.consumed == len(token)
    assert reader.peek_byte() == ord(",")
    assert reader.raw_digest.hexdigest() == hashlib.sha256(token).hexdigest()
    assert reader.self_digest.hexdigest() == hashlib.sha256(token).hexdigest()
    assert reader.raw_digest.updates <= 2
    assert reader.self_digest.updates <= 2


@pytest.mark.parametrize("object_value", [False, True])
@pytest.mark.parametrize("fragment_bytes", [1, 7, 64 * 1024])
@pytest.mark.parametrize("extra_bytes", [0, 1])
def test_token_byte_cap_includes_closing_delimiter(
    monkeypatch, object_value, fragment_bytes, extra_bytes
):
    monkeypatch.setattr(publication, "_ENCODING_MANIFEST_VALUE_MAX_BYTES", 32)
    opening, closing = (b'{"a":"', b'"}') if object_value else (b'"', b'"')
    token = opening + b"a" * (32 - len(opening + closing) + extra_bytes) + closing
    reader = publication._EncodingManifestReader(
        _FragmentedInput(token + b",", fragment_bytes), raw_byte_count=None
    )
    read_token = reader.json_object_token if object_value else reader.json_string_token

    if extra_bytes:
        with pytest.raises(publication.FinalPublicationValidationError, match="exceeds its bound"):
            read_token()
        assert reader.consumed == 32
    else:
        assert read_token() == token
        assert reader.peek_byte() == ord(",")
    assert reader.raw_digest.hexdigest() == hashlib.sha256(token[:32]).hexdigest()


@pytest.mark.parametrize("token", [b'"', b'"abc\\', b'{"a":', b'{"a":"abc\\'])
@pytest.mark.parametrize("fragment_bytes", [1, 7, 64 * 1024])
def test_unclosed_token_stops_at_eof(token, fragment_bytes):
    reader = publication._EncodingManifestReader(
        _FragmentedInput(token, fragment_bytes), raw_byte_count=None
    )
    read_token = reader.json_object_token if token.startswith(b"{") else reader.json_string_token

    with pytest.raises(publication.FinalPublicationValidationError, match="ended unexpectedly"):
        read_token()
    assert reader.consumed == len(token)
    assert reader.raw_digest.hexdigest() == hashlib.sha256(token).hexdigest()


@pytest.mark.parametrize("token", [b'"value"', b'{"a":{"b":"value"}}'])
def test_token_does_not_read_closing_delimiter_beyond_raw_limit(token):
    handle = _FragmentedInput(token + b"ignored", 64 * 1024)
    reader = publication._EncodingManifestReader(handle, raw_byte_count=len(token) - 1)
    read_token = reader.json_object_token if token.startswith(b"{") else reader.json_string_token

    with pytest.raises(publication.FinalPublicationValidationError, match="ended unexpectedly"):
        read_token()
    assert handle.tell() == len(token) - 1
    assert reader.consumed == len(token) - 1
    assert reader.raw_digest.hexdigest() == hashlib.sha256(token[:-1]).hexdigest()


@pytest.mark.parametrize("mode", ["assembled_vr", "direct_stereo"])
@pytest.mark.parametrize("fragment_bytes", [1, 2, 3, 7, 64 * 1024])
def test_fragmented_manifest_preserves_raw_and_self_fingerprints(mode, fragment_bytes):
    raw, fingerprint = _encode_input_manifest(_input_manifest_value(3, mode=mode))
    projection = publication._parse_encoding_input_manifest_stream_v1(
        _FragmentedInput(raw, fragment_bytes), raw_byte_count=len(raw), expected_frame_count=3
    )

    assert projection.fingerprint == fingerprint
    assert projection.raw_sha256 == hashlib.sha256(raw).hexdigest()
    assert projection.raw_byte_count == len(raw)
    assert projection.mode == mode
    assert projection.frame_count == 3
    assert projection.first_index == 0


@pytest.mark.parametrize(
    "token",
    [
        b'{"a":1,"a":2}',
        b'{"a":NaN}',
        b'{"a":1,}',
        b'{ "a":1}',
        b'{"a":"\\q"}',
        b'{"a":"\xff"}',
        b'{"a":"\x00"}',
        b'{"a":"\\ud800"}',
        b'"\\q"',
        b'"\xff"',
        b'"\x00"',
        b'"\\ud800"',
    ],
)
def test_token_scanning_still_requires_strict_canonical_value_validation(token):
    reader = publication._EncodingManifestReader(io.BytesIO(token), raw_byte_count=None)
    scanned = reader.json_object_token() if token.startswith(b"{") else reader.json_string_token()

    with pytest.raises(publication.FinalPublicationValidationError, match="malformed"):
        publication._parse_bounded_canonical_value(scanned, description="test token")
