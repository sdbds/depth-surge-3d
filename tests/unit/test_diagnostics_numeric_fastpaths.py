"""Exact numeric and canonical byte contracts for diagnostics hot paths."""

from __future__ import annotations

import numpy as np
import pytest

from src.depth_surge_3d.rendering.quality import diagnostic_transaction, diagnostics


@pytest.mark.parametrize(
    "parser_name", ["_parse_canonical_object", "_parse_canonical_small_object"]
)
def test_json_native_diagnostics_do_not_use_the_histogram_encoder(
    monkeypatch: pytest.MonkeyPatch, parser_name: str
) -> None:
    def reject_histogram_encoder(_value):
        raise AssertionError("JSON-native values must not traverse the histogram encoder")

    monkeypatch.setattr(diagnostic_transaction, "canonical_json_bytes", reject_histogram_encoder)
    raw = b'{"a":[true,null,18446744073709551615],"b":1.25}'

    assert getattr(diagnostic_transaction, parser_name)(raw) == {
        "a": [True, None, 18446744073709551615],
        "b": 1.25,
    }


@pytest.mark.parametrize(
    "parser_name", ["_parse_canonical_object", "_parse_canonical_small_object"]
)
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b"{}", {}),
        (b'{"a":{"b":[false,true,null]}}', {"a": {"b": [False, True, None]}}),
        (b'{"n":18446744073709551616}', {"n": 18446744073709551616}),
        (b'{"n":-0.0}', {"n": -0.0}),
        (b'{"n":1e-06}', {"n": 1e-6}),
        (b'{"n":5e-324}', {"n": 5e-324}),
        (b'{"n":1.7976931348623157e+308}', {"n": 1.7976931348623157e308}),
        (b'{"s":"\\u4e2d\\u6587"}', {"s": "\u4e2d\u6587"}),
        (b'{"s":"\\ud800"}', {"s": "\ud800"}),
    ],
)
def test_json_native_diagnostics_preserve_existing_canonical_spellings(
    parser_name: str, raw: bytes, expected: dict
) -> None:
    parsed = getattr(diagnostic_transaction, parser_name)(raw)
    assert parsed == expected
    assert diagnostic_transaction.canonical_json_bytes(parsed) == raw


@pytest.mark.parametrize(
    "parser_name", ["_parse_canonical_object", "_parse_canonical_small_object"]
)
@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"a":{"b":1,"b":2}}',
        b'{"b":1,"a":2}',
        b'{"a": 1}',
        b'{"a":1}\n',
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'{"a":1e400}',
        b'{"a":1e0}',
        b'{"a":-0}',
        b'{"a":01}',
        b'{"a":"\\u00E9"}',
        b'{"a":"\xc3\xa9"}',
        b'\xef\xbb\xbf{"a":1}',
        b"[]",
        b"null",
    ],
)
def test_json_native_diagnostics_still_reject_noncanonical_bytes(
    parser_name: str, raw: bytes
) -> None:
    with pytest.raises(ValueError):
        getattr(diagnostic_transaction, parser_name)(raw)


def test_histogram_integer_scan_does_not_rebuild_numpy_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_limit_lookup(_dtype):
        raise AssertionError("histogram tokens must reuse the fixed uint64 limit")

    monkeypatch.setattr(diagnostic_transaction.np, "iinfo", reject_limit_lookup)

    assert diagnostic_transaction._parse_canonical_uint(b"[18446744073709551615]", 1) == (
        18446744073709551615,
        21,
    )


@pytest.mark.parametrize("buffer_type", [bytes, bytearray])
@pytest.mark.parametrize(
    ("raw", "offset", "expected"),
    [
        (b"0]", 0, (0, 1)),
        (b"[17,", 1, (17, 3)),
        (b"[4294967296]", 1, (4294967296, 11)),
        (b"[18446744073709551615]", 1, (18446744073709551615, 21)),
    ],
)
def test_histogram_integer_scan_preserves_values_and_token_boundaries(
    buffer_type, raw: bytes, offset: int, expected: tuple[int, int]
) -> None:
    assert diagnostic_transaction._parse_canonical_uint(buffer_type(raw), offset) == expected


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b",",
        b"01]",
        b"00]",
        b"+1]",
        b"-1]",
        b" 1]",
        b"\xff1]",
        b"18446744073709551616]",
        b"100000000000000000000]",
        b"9" * 5000,
    ],
)
def test_histogram_integer_scan_rejects_noncanonical_or_overflowing_tokens(raw: bytes) -> None:
    with pytest.raises(ValueError):
        diagnostic_transaction._parse_canonical_uint(raw, 0)


def test_builtin_diagnostics_integers_do_not_dispatch_to_the_integral_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RejectProtocol(type):
        def __instancecheck__(cls, value):
            raise AssertionError("exact Python ints do not need Integral protocol dispatch")

    class GenericIntegral(metaclass=RejectProtocol):
        pass

    monkeypatch.setattr(diagnostics, "Integral", GenericIntegral)

    assert diagnostics.checked_u64_add("total", 0, 17, 25) == 42
    assert diagnostics.checked_u64_mul("pixels", 16, 1920, 1080) == 33177600


class _NormalizedInt(int):
    def __int__(self) -> int:
        return 7


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (2**64 - 1, 2**64 - 1),
        (np.uint64(23), 23),
        (np.int64(19), 19),
        (_NormalizedInt(1), 7),
    ],
)
def test_checked_diagnostics_arithmetic_keeps_nonbuiltin_integer_normalization(
    value: object, expected: int
) -> None:
    assert diagnostics.checked_u64_add("total", value) == expected
    assert diagnostics.checked_u64_mul("product", value) == expected


@pytest.mark.parametrize("operation", [diagnostics.checked_u64_add, diagnostics.checked_u64_mul])
@pytest.mark.parametrize("value", [True, np.bool_(False), -1, 2**64, 1.0, "1", None])
def test_checked_diagnostics_arithmetic_preserves_operand_error_details(operation, value) -> None:
    with pytest.raises(diagnostics.DiagnosticsCardinalityError) as captured:
        operation("rejected", value)

    assert captured.value.field == "rejected"
    assert captured.value.operands == (value,)


@pytest.mark.parametrize(
    ("operation", "operands"),
    [
        (diagnostics.checked_u64_add, (2**64 - 1, 1)),
        (diagnostics.checked_u64_mul, (2**63, 2)),
    ],
)
def test_checked_diagnostics_arithmetic_preserves_overflow_operand_order(
    operation, operands
) -> None:
    with pytest.raises(diagnostics.DiagnosticsCardinalityError) as captured:
        operation("overflow", *operands)

    assert captured.value.field == "overflow"
    assert captured.value.operands == operands
