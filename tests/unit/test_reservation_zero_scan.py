"""Reservation padding is fully checked without Python loops or suffix copies."""

from __future__ import annotations

import tracemalloc

import pytest

from src.depth_surge_3d.io import reservation_protocol as protocol
from tests.unit.test_settings_artifact_v5 import _artifact


@pytest.mark.parametrize("operation", ["classify", "terminal"])
def test_zero_extents_do_not_check_each_byte_in_python(operation, monkeypatch):
    checked = 0

    def observed_all(values):
        def observe():
            nonlocal checked
            for value in values:
                checked += 1
                yield value

        return all(observe())

    monkeypatch.setattr(protocol, "all", observed_all, raising=False)
    raw = b"\x00" * 65536
    if operation == "classify":
        assert protocol.classify_payload_frame(raw, b"{}") == "zero"
    else:
        assert protocol._parse_terminal_source_payload(raw, full_length=True) is None
    assert checked == 0


def test_complete_frame_classification_does_not_copy_the_padding():
    payload = b'{"schema_version":1}'
    raw = payload + b"\x00" * (2 * 1024 * 1024)
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        assert protocol.classify_payload_frame(raw, payload) == "complete-full"
        _, peak = tracemalloc.get_traced_memory()
        assert peak - before < 64 * 1024
    finally:
        if not was_tracing:
            tracemalloc.stop()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b"", "partial"),
        (b"\x00", "zero"),
        (b"\x00\x00", "zero"),
        (b"{}", "complete-short"),
        (b"{}\x00", "complete-full"),
        (b"{}\x00\x00", "complete-full"),
        (b"{", "partial"),
        (b"{\x00", "partial"),
        (b"\x00{}", "partial"),
        (b"{}x", "partial"),
        (b"{}\x00x", "partial"),
        (b"{}x\x00", "partial"),
        (b"{}\x00{}\x00", "partial"),
    ],
)
def test_frame_classification_keeps_all_four_states(raw, expected):
    assert protocol.classify_payload_frame(raw, b"{}") == expected


@pytest.mark.parametrize("payload", [b"", b"\x00", b"{}\x00", b"{\x00}"])
def test_invalid_expected_payload_is_rejected_before_classification(payload):
    with pytest.raises(ValueError, match="nonempty and NUL-free"):
        protocol.classify_payload_frame(b"\x00" * 32, payload)


@pytest.mark.parametrize("padding_bytes", [1, 4095, 4096, 65535, 65536, 528384])
def test_zero_and_complete_frames_remain_valid_across_extent_sizes(padding_bytes):
    zeroes = b"\x00" * padding_bytes
    assert protocol.classify_payload_frame(zeroes, b"{}") == "zero"
    assert protocol.classify_payload_frame(b"{}" + zeroes, b"{}") == "complete-full"


@pytest.mark.parametrize("position", [0, 1, 7, 15])
def test_every_nonzero_byte_is_rejected_anywhere_in_zero_or_complete_padding(position):
    for value in range(1, 256):
        padding = b"\x00" * position + bytes([value]) + b"\x00" * (15 - position)
        assert protocol.classify_payload_frame(padding, b"{}") == "partial"
        assert protocol.classify_payload_frame(b"{}" + padding, b"{}") == "partial"


@pytest.mark.parametrize("position", [0, 65535, 65536, 528383])
def test_large_partial_padding_is_not_mistaken_for_a_complete_frame(position):
    padding = b"\x00" * position + b"\xff" + b"\x00" * (528383 - position)
    assert protocol.classify_payload_frame(padding, b"{}") == "partial"
    assert protocol.classify_payload_frame(b"{}" + padding, b"{}") == "partial"


@pytest.mark.parametrize("raw_type", [bytes, bytearray])
def test_builtin_byte_sequences_keep_classification_behavior(raw_type):
    assert protocol.classify_payload_frame(raw_type(b"\x00\x00"), b"{}") == "zero"
    assert protocol.classify_payload_frame(raw_type(b"{}\x00"), b"{}") == "complete-full"
    assert protocol.classify_payload_frame(raw_type(b"{}\x00x"), b"{}") == "partial"


def test_byte_subclass_cannot_make_nonzero_bytes_look_zero_by_overriding_count():
    class ClaimsZero(bytes):
        def count(self, *args):
            return len(self)

        def __getitem__(self, item):
            if not isinstance(item, slice):
                return 0
            return super().__getitem__(item)

    raw = ClaimsZero(b"\x00x")
    assert protocol.classify_payload_frame(raw, b"{}") == "partial"
    with pytest.raises(protocol.ReservationIncompleteError, match="partial frame"):
        protocol._parse_terminal_source_payload(raw, full_length=True)


def test_byte_subclass_retains_its_existing_iteration_semantics():
    class IteratesNonzero(bytes):
        def __iter__(self):
            return iter([1])

    class IteratesZero(bytes):
        def __iter__(self):
            return iter([0])

    assert protocol.classify_payload_frame(IteratesNonzero(b"\x00"), b"{}") == "partial"
    assert protocol.classify_payload_frame(IteratesZero(b"x"), b"{}") == "zero"


def test_byte_subclass_with_an_empty_suffix_is_not_a_complete_frame():
    class EmptySuffix(bytes):
        def __getitem__(self, item):
            if isinstance(item, slice):
                return b""
            return super().__getitem__(item)

    assert protocol.classify_payload_frame(EmptySuffix(b"{}\x00"), b"{}") == "partial"


@pytest.mark.parametrize("padding_bytes", [1, 4096, 528384])
def test_terminal_source_still_parses_full_and_short_settings(padding_bytes):
    raw = _artifact().encode()
    full = protocol._parse_terminal_source_payload(raw + b"\x00" * padding_bytes, full_length=True)
    short = protocol._parse_terminal_source_payload(raw, full_length=False)
    assert full.encode() == short.encode() == raw


@pytest.mark.parametrize("raw", [b"", b"\x00x", b"x\x00", b"{}", b"{}\x00", b"{}\x00x"])
def test_partial_terminal_sources_keep_the_full_length_error(raw):
    with pytest.raises(protocol.ReservationIncompleteError, match="partial frame"):
        protocol._parse_terminal_source_payload(raw, full_length=True)


@pytest.mark.parametrize("raw", [b"", b"\x00", b"{}", b"{}\x00", b"not-json"])
def test_invalid_short_terminal_sources_keep_the_conflict_error(raw):
    with pytest.raises(protocol.ReservationConflictError, match="not a canonical artifact"):
        protocol._parse_terminal_source_payload(raw, full_length=False)
