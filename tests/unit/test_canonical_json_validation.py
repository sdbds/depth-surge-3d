"""Strict JSON keeps its byte contract without repeating scalar validation."""

from __future__ import annotations

import json

import pytest

from src.depth_surge_3d.io import canonical_json as codec


def test_strict_canonical_parse_visits_each_string_once(monkeypatch):
    visited = []
    original = codec.validate_unicode_scalars

    def observe(value):
        if isinstance(value, str):
            visited.append(value)
        return original(value)

    monkeypatch.setattr(codec, "validate_unicode_scalars", observe)
    raw = b'{"a":["first",{"b":"\\u4e2d\\u6587"}],"c":"last"}'

    assert codec.strict_canonical_json_loads(raw) == {
        "a": ["first", {"b": "\u4e2d\u6587"}],
        "c": "last",
    }
    assert visited == ["a", "first", "b", "\u4e2d\u6587", "c", "last"]


def test_plain_ascii_encoding_avoids_per_character_scalar_checks(monkeypatch):
    checked = []

    def observe(character):
        checked.append(character)
        return ord(character)

    monkeypatch.setattr(codec, "ord", observe, raising=False)
    value = "f" * 4096

    assert codec.canonical_json_bytes({"sha256": value}) == b'{"sha256":"' + b"f" * 4096 + b'"}'
    assert checked == []


def test_ascii_json_without_unicode_escapes_avoids_rewalking_the_tree(monkeypatch):
    visited = []
    original = codec.validate_unicode_scalars

    def observe(value):
        visited.append(value)
        return original(value)

    monkeypatch.setattr(codec, "validate_unicode_scalars", observe)
    raw = b'{"a":["frame_000001","line\\nnext\\tend",123],"b":true}'

    assert codec.strict_canonical_json_loads(raw) == {
        "a": ["frame_000001", "line\nnext\tend", 123],
        "b": True,
    }
    assert visited == []


def test_bytes_subclass_cannot_bypass_scalar_validation_with_a_custom_decoder():
    class CustomDecode(bytes):
        def decode(self, *args, **kwargs):
            return '"\ud800"'

    with pytest.raises(codec.CanonicalJsonError, match="isolated surrogate"):
        codec.strict_json_loads(CustomDecode(b'"safe"'))


def test_bytes_subclass_cannot_hide_unicode_escape_bytes():
    class HidesEscapes(bytes):
        def __contains__(self, item):
            return False

    with pytest.raises(codec.CanonicalJsonError, match="isolated surrogate"):
        codec.strict_json_loads(HidesEscapes(b'"\\ud800"'))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, b"null"),
        (True, b"true"),
        (False, b"false"),
        (0, b"0"),
        (2**64 - 1, b"18446744073709551615"),
        (-0.0, b"-0.0"),
        (1e-5, b"1e-05"),
        ("", b'""'),
        ("\u4e2d\u6587", b'"\\u4e2d\\u6587"'),
        ("\U0001f642", b'"\\ud83d\\ude42"'),
        ('\x00\x7f\n\t"\\', b'"\\u0000\\u007f\\n\\t\\"\\\\"'),
        ({"z": [None, True, 1], "a": {"b": "c"}}, b'{"a":{"b":"c"},"z":[null,true,1]}'),
    ],
)
def test_canonical_bytes_and_strict_round_trip_are_unchanged(value, expected):
    assert codec.canonical_json_bytes(value) == expected
    assert codec.strict_canonical_json_loads(expected) == value


@pytest.mark.parametrize("codepoint", [0, 127, 128, 0xD7FF, 0xE000, 0xFFFF, 0x10000, 0x10FFFF])
def test_unicode_scalar_boundaries_remain_valid(codepoint):
    character = chr(codepoint)
    value = {character: [character]}
    expected = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )

    assert codec.canonical_json_bytes(value) == expected
    assert codec.strict_canonical_json_loads(expected) == value


def test_every_isolated_surrogate_is_rejected_in_keys_and_nested_values():
    for codepoint in range(0xD800, 0xE000):
        character = chr(codepoint)
        escaped = f"\\u{codepoint:04x}".encode("ascii")
        for value, raw in (
            ({character: "value"}, b'{"' + escaped + b'":"value"}'),
            ({"key": [character]}, b'{"key":["' + escaped + b'"]}'),
        ):
            with pytest.raises(codec.CanonicalJsonError, match="isolated surrogate"):
                codec.canonical_json_bytes(value)
            with pytest.raises(codec.CanonicalJsonError, match="isolated surrogate"):
                codec.strict_canonical_json_loads(raw)


def test_string_subclass_cannot_bypass_scalar_validation_by_overriding_isascii():
    class ClaimsAscii(str):
        def isascii(self):
            return True

    value = ClaimsAscii("\ud800")
    with pytest.raises(codec.CanonicalJsonError, match="isolated surrogate"):
        codec.canonical_json_bytes({"value": value})


def test_string_subclass_keeps_its_existing_iteration_semantics():
    class OverridesIteration(str):
        def __iter__(self):
            return iter("\ud800")

    with pytest.raises(codec.CanonicalJsonError, match="isolated surrogate"):
        codec.canonical_json_bytes(OverridesIteration("ascii"))


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"a":1,"\\u0061":2}',
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'{"a":-Infinity}',
        b'{"a":1e999}',
        b'{ "a":1}',
        b'{"z":1,"a":2}',
        b'{"a":-0}',
        b'{"a":1e-5}',
        b'{"a":1}\n',
        b'{"a":1,}',
        b'{"a":"\\q"}',
        b'{"a":"\xff"}',
        b'{"a":"\x00"}',
        b"\xef\xbb\xbf{}",
        b'"\\ud800x\\udc00"',
        b'"\\udc00\\ud800"',
    ],
)
def test_strict_canonical_parser_rejects_the_same_invalid_spellings(raw):
    with pytest.raises(codec.CanonicalJsonError):
        codec.strict_canonical_json_loads(raw)


@pytest.mark.parametrize("value", [object(), {1, 2}, float("nan"), float("inf"), [float("-inf")]])
def test_encoder_errors_remain_canonical_json_errors(value):
    with pytest.raises(codec.CanonicalJsonError, match="cannot be encoded"):
        codec.canonical_json_bytes(value)


def test_strict_decode_enforces_byte_limit_before_parsing():
    raw = b'{"a":1}'
    assert codec.strict_canonical_json_loads(raw, max_bytes=len(raw)) == {"a": 1}
    with pytest.raises(codec.CanonicalJsonError, match="raw-byte limit"):
        codec.strict_canonical_json_loads(raw, max_bytes=len(raw) - 1)


@pytest.mark.parametrize("raw", ["{}", bytearray(b"{}"), memoryview(b"{}"), None, {}])
def test_public_json_decoder_rejects_non_bytes(raw):
    with pytest.raises(TypeError, match="must be bytes"):
        codec.strict_json_loads(raw)


def test_public_json_decoder_rejects_negative_byte_limit():
    with pytest.raises(codec.CanonicalJsonError, match="raw-byte limit"):
        codec.strict_json_loads(b"0", max_bytes=-1)


@pytest.mark.parametrize("raw", [b'"\\ud800"', b'{"\\udfff":0}', b'["ok","\\udc00"]'])
def test_public_json_decoder_still_rejects_isolated_surrogates(raw):
    with pytest.raises(codec.CanonicalJsonError, match="isolated surrogate"):
        codec.strict_json_loads(raw)


def test_public_json_decoder_does_not_require_canonical_spelling():
    assert codec.strict_json_loads(b' { "z": 1, "a": "\\u4e2d" }\n') == {"a": "\u4e2d", "z": 1}


def test_public_encoder_revalidates_mutated_objects_on_each_call():
    value = {"key": ["safe"]}
    assert codec.canonical_json_bytes(value) == b'{"key":["safe"]}'
    value["key"][0] = "\udfff"
    with pytest.raises(codec.CanonicalJsonError, match="isolated surrogate"):
        codec.canonical_json_bytes(value)


def test_strict_decoder_returns_a_fresh_validated_tree_each_time():
    raw = b'{"key":["safe"]}'
    first = codec.strict_canonical_json_loads(raw)
    first["key"][0] = "\udfff"

    assert codec.strict_canonical_json_loads(raw) == {"key": ["safe"]}
