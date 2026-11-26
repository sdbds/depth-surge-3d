"""Buffered canonical reads retain byte identities and all parser bounds."""

import hashlib
import io
import json
import tracemalloc

import pytest

from src.depth_surge_3d.io.stereo_contract import StereoControlBudgetError
from src.depth_surge_3d.rendering.quality.diagnostic_transaction import (
    _BoundedReadHandle,
    _ExactCanonicalReader,
)


class _ShortReads(io.BytesIO):
    def __init__(self, raw, maximum):
        super().__init__(raw)
        self.maximum = maximum

    def read(self, size=-1):
        return super().read(min(size, self.maximum) if size >= 0 else self.maximum)


def test_buffered_exact_read_avoids_multiple_payload_sized_copies():
    reader = _ExactCanonicalReader(io.BytesIO(b"x" + b"a" * 65536))
    reader.expect(b"x")
    tracemalloc.start()
    try:
        value = reader._read_exact(32768)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert value == b"a" * 32768
    assert peak < 2 * len(value)
    assert reader.raw_digest.hexdigest() == hashlib.sha256(b"x" + value).hexdigest()


@pytest.mark.parametrize("nested", (False, True))
def test_literal_strings_do_not_require_one_read_per_character(nested):
    class ObservedReader(_ExactCanonicalReader):
        def _read_exact(self, count):
            reads.append(count)
            return super()._read_exact(count)

    reads = []
    value = b'"' + b"a" * 256 + b'"'
    payload = b'{"value":' + value + b"}" if nested else value
    reader = ObservedReader(io.BytesIO(payload + b","))
    assert reader.canonical_value(maximum_bytes=4096) == json.loads(payload)
    assert reader.peek() == b"," and reader.raw_byte_count == len(payload)
    assert max(reads) >= 256
    assert len(reads) < 24


@pytest.mark.parametrize("fragment", (1, 7, 65536))
@pytest.mark.parametrize("fingerprinted", (False, True))
@pytest.mark.parametrize(
    "payload",
    (
        b'""',
        b'"long literal with spaces"',
        b'"quote\\" and slash\\\\ and newline\\n"',
        b'"\\u6df1\\u5ea6"',
        b'{"a":"abcdef","b":[true,false,null,-2,1.5]}',
        b'["abc",{"nested":"def"}]',
    ),
)
def test_value_reads_preserve_raw_and_self_hashes(payload, fragment, fingerprinted):
    prefix = b"prefix:"
    reader = _ExactCanonicalReader(_ShortReads(prefix + payload + b",tail", fragment))
    reader.expect(prefix)
    result = reader.canonical_value(maximum_bytes=4096, fingerprinted=fingerprinted)
    assert result == json.loads(payload)
    assert reader.peek() == reader.peek() == b"," and reader.raw_byte_count == len(prefix + payload)
    assert reader.raw_digest.hexdigest() == hashlib.sha256(prefix + payload).hexdigest()
    expected_self = prefix + payload if fingerprinted else prefix
    assert reader.self_digest.hexdigest() == hashlib.sha256(expected_self).hexdigest()
    assert reader._read_exact(0) == b"" and reader.peek() == b","
    reader.expect(b",")
    assert reader.raw_digest.hexdigest() == hashlib.sha256(prefix + payload + b",").hexdigest()


@pytest.mark.parametrize("fragment", (1, 7, 65536))
@pytest.mark.parametrize(
    "payload,bounds,message,consumed",
    (
        (b'"abcd"', {"maximum_bytes": 3}, "byte budget", b'"abc'),
        (b'"abcd"', {"maximum_string_bytes": 2}, "string budget", b'"abc'),
        (b'"abcd"', {"maximum_bytes": 3, "maximum_string_bytes": 2}, "string budget", b'"abc'),
        (b'["ab"]', {"maximum_bytes": 4}, "byte budget", b'["ab"'),
        (b"[[[0]]]", {"maximum_depth": 2}, "depth budget", b"[[["),
        (b'{"a":[],"b":[]}', {"maximum_containers": 2}, "container budget", b'{"a":[],"b":['),
        (b'["a","b"]', {"maximum_scalars": 1}, "scalar budget", b'["a","'),
        (b"[12345]", {"maximum_string_bytes": 3}, "scalar-token budget", b"[1234"),
    ),
)
def test_budgets_fail_at_the_original_first_byte(payload, bounds, message, consumed, fragment):
    reader = _ExactCanonicalReader(_ShortReads(payload + b",", fragment))
    arguments = {"maximum_bytes": 4096, **bounds}
    with pytest.raises(StereoControlBudgetError, match=message):
        reader.canonical_value(**arguments)
    assert reader.raw_byte_count == len(consumed)
    assert reader.raw_digest.hexdigest() == hashlib.sha256(consumed).hexdigest()
    assert reader.self_digest.hexdigest() == hashlib.sha256().hexdigest()


@pytest.mark.parametrize("fragment", (1, 7, 65536))
@pytest.mark.parametrize(
    "payload",
    (
        b'"unterminated',
        b'"bad\\',
        b'"\\uD800"',
        b'"\xc0\x80"',
        b'"\xe6\xb7\xb1"',
        b'"\x00"',
        b'{"a":1,"a":2}',
        b'{"z":1,"a":2}',
        b'{"a": true}',
        b"[1,]",
        b"[{]}",
    ),
)
def test_malformed_values_are_not_accepted_by_literal_scanning(payload, fragment):
    reader = _ExactCanonicalReader(_ShortReads(payload, fragment))
    with pytest.raises(ValueError):
        reader.canonical_value(maximum_bytes=4096)
    assert reader.self_digest.hexdigest() == hashlib.sha256().hexdigest()


def test_top_level_closing_quote_retains_decoder_byte_budget_failure():
    reader = _ExactCanonicalReader(io.BytesIO(b'"ab",'))
    with pytest.raises(ValueError, match="noncanonical"):
        reader.canonical_value(maximum_bytes=3)
    assert reader.raw_byte_count == 4
    assert reader.raw_digest.hexdigest() == hashlib.sha256(b'"ab"').hexdigest()


@pytest.mark.parametrize("padding", (65530, 65535, 65536))
def test_literal_runs_cross_buffer_boundaries_without_consuming_a_pending_delimiter(padding):
    prefix = b"x" * padding
    value = b'{"a":"' + b"b" * 1024 + b'","c":"escaped\\"tail"}'
    reader = _ExactCanonicalReader(io.BytesIO(prefix + value + b","))
    reader.expect(prefix)
    assert reader.peek() == b"{"
    assert reader.canonical_value(maximum_bytes=4096) == json.loads(value)
    assert reader.peek() == b"," and reader.raw_byte_count == padding + len(value)
    assert reader.raw_digest.hexdigest() == hashlib.sha256(prefix + value).hexdigest()


def test_declared_prefix_does_not_read_or_hash_appended_control_bytes():
    payload = b'{"a":"bounded"}'
    handle = io.BytesIO(payload + b"unrelated trailing bytes")
    reader = _ExactCanonicalReader(_BoundedReadHandle(handle, len(payload)))
    assert reader.canonical_value(maximum_bytes=4096) == {"a": "bounded"}
    assert not reader.peek()
    assert handle.tell() == len(payload)
    assert reader.raw_byte_count == len(payload)
    assert reader.raw_digest.hexdigest() == hashlib.sha256(payload).hexdigest()


def test_truncated_exact_read_does_not_hash_a_partially_consumed_payload():
    reader = _ExactCanonicalReader(_ShortReads(b"abc", 1))
    assert reader.peek() == b"a"
    with pytest.raises(ValueError, match="ended early"):
        reader._read_exact(5)
    assert reader.raw_byte_count == 0
    assert reader.raw_digest.hexdigest() == hashlib.sha256().hexdigest()
    assert not reader.peek()
