"""Canonical JSON codec shared by durable control artifacts.

The codec is intentionally small and strict.  It rejects duplicate members,
non-finite numbers, non-scalar Unicode strings, and any raw spelling that does
not round-trip to the frozen ASCII encoding.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any


class CanonicalJsonError(ValueError):
    """A JSON payload violates the durable canonical byte contract."""


def _reject_constant(value: str) -> None:
    raise CanonicalJsonError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise CanonicalJsonError(f"duplicate JSON member: {key}")
        value[key] = item
    return value


def validate_unicode_scalars(value: object) -> None:
    """Reject isolated UTF-16 surrogate code points recursively."""

    if isinstance(value, str):
        if type(value) is str and value.isascii():
            return
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise CanonicalJsonError("JSON string contains an isolated surrogate")
        return
    if isinstance(value, list):
        for item in value:
            validate_unicode_scalars(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            validate_unicode_scalars(key)
            validate_unicode_scalars(item)


def canonical_json_bytes(value: object) -> bytes:
    """Encode one object using the project's exact canonical JSON spelling."""

    validate_unicode_scalars(value)
    return _encode_validated_json(value)


def _encode_validated_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise CanonicalJsonError("value cannot be encoded as canonical JSON") from error


def strict_json_loads(raw: bytes, *, max_bytes: int | None = None) -> object:
    """Parse bounded ASCII JSON while rejecting duplicates and non-scalars."""

    if not isinstance(raw, bytes):
        raise TypeError("raw JSON must be bytes")
    if max_bytes is not None and (max_bytes < 0 or len(raw) > max_bytes):
        raise CanonicalJsonError("JSON payload exceeds its raw-byte limit")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise CanonicalJsonError("canonical JSON must be ASCII without a BOM") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except CanonicalJsonError:
        raise
    except (json.JSONDecodeError, UnicodeError, ValueError) as error:
        raise CanonicalJsonError("malformed JSON payload") from error
    # Only Unicode escapes can introduce non-ASCII characters from ASCII JSON.
    if type(raw) is not bytes or b"\\u" in raw:
        validate_unicode_scalars(value)
    return value


def strict_canonical_json_loads(raw: bytes, *, max_bytes: int | None = None) -> object:
    """Parse JSON and require its bytes to equal the canonical re-encoding."""

    value = strict_json_loads(raw, max_bytes=max_bytes)
    # The fresh parsed tree has already passed scalar validation.
    if _encode_validated_json(value) != raw:
        raise CanonicalJsonError("JSON payload is not in canonical byte form")
    return value
