"""Canonical JSON objects with one replayable, variable-cardinality array."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping

from .canonical_json import CanonicalJsonError, canonical_json_bytes
from .durable_publish import OwnedSiblingTemporary
from .nofollow_io import AcquiredRoot, NoFollowPathError


ItemReplay = Callable[[], Iterable[object]]
PublishFault = Callable[[str], None]


def _matches_identity(
    path: Path,
    *,
    raw_sha256: str,
    raw_byte_count: int,
    acquired: AcquiredRoot | None,
) -> bool:
    digest = hashlib.sha256()
    byte_count = 0
    workspace = bytearray(1024 * 1024)
    try:
        if acquired is None:
            with AcquiredRoot(Path(path).parent) as local_root:
                return _matches_identity(
                    path,
                    raw_sha256=raw_sha256,
                    raw_byte_count=raw_byte_count,
                    acquired=local_root,
                )
        with acquired.open_regular(Path(path)) as handle:
            while True:
                count = handle.readinto(workspace)
                if not count:
                    break
                digest.update(memoryview(workspace)[:count])
                byte_count += count
    except (OSError, NoFollowPathError, ValueError):
        return False
    return byte_count == raw_byte_count and digest.hexdigest() == raw_sha256


@dataclass(frozen=True, slots=True)
class ReplayableJsonArray:
    items: ItemReplay

    def __post_init__(self) -> None:
        if not callable(self.items):
            raise TypeError("replayable JSON array requires a callable factory")


def _value_chunks(value: object) -> Iterator[bytes]:
    if isinstance(value, ReplayableJsonArray):
        yield b"["
        for index, item in enumerate(value.items()):
            if index:
                yield b","
            yield from _value_chunks(item)
        yield b"]"
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise CanonicalJsonError("canonical JSON object keys must be strings")
        yield b"{"
        for index, key in enumerate(sorted(value)):
            if index:
                yield b","
            yield canonical_json_bytes(key)
            yield b":"
            yield from _value_chunks(value[key])
        yield b"}"
        return
    if isinstance(value, (list, tuple)):
        yield b"["
        for index, item in enumerate(value):
            if index:
                yield b","
            yield from _value_chunks(item)
        yield b"]"
        return
    yield canonical_json_bytes(value)


def _pretty_array_chunks(items: Iterable[object], level: int) -> Iterator[bytes]:
    iterator = iter(items)
    try:
        first = next(iterator)
    except StopIteration:
        yield b"[]"
        return
    child_prefix = b" " * (2 * (level + 1))
    closing_prefix = b" " * (2 * level)
    yield b"[\n"
    yield child_prefix
    yield from _pretty_value_chunks(first, level + 1)
    for item in iterator:
        yield b",\n"
        yield child_prefix
        yield from _pretty_value_chunks(item, level + 1)
    yield b"\n"
    yield closing_prefix
    yield b"]"


def _pretty_value_chunks(value: object, level: int = 0) -> Iterator[bytes]:
    if isinstance(value, ReplayableJsonArray):
        yield from _pretty_array_chunks(value.items(), level)
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise CanonicalJsonError("JSON object keys must be strings")
        keys = sorted(value)
        if not keys:
            yield b"{}"
            return
        child_prefix = b" " * (2 * (level + 1))
        closing_prefix = b" " * (2 * level)
        yield b"{\n"
        for index, key in enumerate(keys):
            if index:
                yield b",\n"
            yield child_prefix
            yield canonical_json_bytes(key)
            yield b": "
            yield from _pretty_value_chunks(value[key], level + 1)
        yield b"\n"
        yield closing_prefix
        yield b"}"
        return
    if isinstance(value, (list, tuple)):
        yield from _pretty_array_chunks(value, level)
        return
    yield canonical_json_bytes(value)


def _object_chunks(
    static_fields: Mapping[str, object],
    *,
    array_items: Mapping[str, ItemReplay],
    fingerprint_key: str | None,
    fingerprint: str | None,
) -> Iterator[bytes]:
    fields = dict(static_fields)
    if set(array_items) & set(fields) or (
        fingerprint_key is not None and fingerprint_key in fields
    ):
        raise CanonicalJsonError("stream-owned JSON keys must not appear in static fields")
    keys = set(fields)
    keys.update(array_items)
    if fingerprint_key is not None and fingerprint is not None:
        keys.add(fingerprint_key)
    yield b"{"
    for position, key in enumerate(sorted(keys)):
        if position:
            yield b","
        yield canonical_json_bytes(key)
        yield b":"
        if key in array_items:
            yield b"["
            for index, item in enumerate(array_items[key]()):
                if index:
                    yield b","
                yield canonical_json_bytes(item)
            yield b"]"
        elif fingerprint_key is not None and key == fingerprint_key:
            yield canonical_json_bytes(fingerprint)
        else:
            yield canonical_json_bytes(fields[key])
    yield b"}"


def _identity(chunks: Iterable[bytes]) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    for chunk in chunks:
        digest.update(chunk)
        byte_count += len(chunk)
    return digest.hexdigest(), byte_count


def _publish_chunks(
    path: Path,
    chunks: Iterable[bytes],
    *,
    raw_sha256: str,
    raw_byte_count: int,
    fault: PublishFault | None,
    label: str,
    description: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    owned = OwnedSiblingTemporary.create(path)
    digest = hashlib.sha256()
    byte_count = 0

    def emit(stage: str) -> None:
        if fault is not None:
            fault(stage)

    try:
        emit(f"before_{label}_temporary_write")
        for chunk in chunks:
            if owned.handle.write(chunk) != len(chunk):
                raise OSError(f"short {description} write")
            digest.update(chunk)
            byte_count += len(chunk)
        emit(f"after_{label}_temporary_write")
        owned.sync()
        emit(f"after_{label}_temporary_sync")
        if digest.hexdigest() != raw_sha256 or byte_count != raw_byte_count:
            raise CanonicalJsonError(f"{description} replay changed")
        emit(f"before_{label}_replace")
        owned.replace(
            after_replace=lambda: emit(f"after_{label}_replace"),
        )
        emit(f"after_{label}_directory_sync")
    finally:
        owned.cleanup()


@dataclass(frozen=True, slots=True)
class StreamedCanonicalObject:
    static_fields: Mapping[str, object]
    array_items: Mapping[str, ItemReplay]
    fingerprint_key: str | None
    fingerprint: str | None
    raw_sha256: str
    raw_byte_count: int

    def chunks(self) -> Iterator[bytes]:
        return _object_chunks(
            self.static_fields,
            array_items=self.array_items,
            fingerprint_key=self.fingerprint_key,
            fingerprint=self.fingerprint,
        )

    def publish(
        self,
        path: Path,
        *,
        fault: PublishFault | None = None,
        label: str = "canonical_json",
    ) -> None:
        _publish_chunks(
            path,
            self.chunks(),
            raw_sha256=self.raw_sha256,
            raw_byte_count=self.raw_byte_count,
            fault=fault,
            label=label,
            description="streamed canonical JSON",
        )

    def matches(self, path: Path, *, acquired: AcquiredRoot | None = None) -> bool:
        return _matches_identity(
            path,
            raw_sha256=self.raw_sha256,
            raw_byte_count=self.raw_byte_count,
            acquired=acquired,
        )


@dataclass(frozen=True, slots=True)
class StreamedCanonicalValue:
    value: object
    raw_sha256: str
    raw_byte_count: int

    def chunks(self) -> Iterator[bytes]:
        return _value_chunks(self.value)

    def publish(
        self,
        path: Path,
        *,
        fault: PublishFault | None = None,
        label: str = "canonical_json",
    ) -> None:
        _publish_chunks(
            path,
            self.chunks(),
            raw_sha256=self.raw_sha256,
            raw_byte_count=self.raw_byte_count,
            fault=fault,
            label=label,
            description="streamed canonical JSON",
        )

    def matches(self, path: Path, *, acquired: AcquiredRoot | None = None) -> bool:
        return _matches_identity(
            path,
            raw_sha256=self.raw_sha256,
            raw_byte_count=self.raw_byte_count,
            acquired=acquired,
        )


@dataclass(frozen=True, slots=True)
class StreamedPrettyJsonValue:
    value: object
    raw_sha256: str
    raw_byte_count: int

    def chunks(self) -> Iterator[bytes]:
        return _pretty_value_chunks(self.value)

    def publish(
        self,
        path: Path,
        *,
        fault: PublishFault | None = None,
        label: str = "pretty_json",
    ) -> None:
        _publish_chunks(
            path,
            self.chunks(),
            raw_sha256=self.raw_sha256,
            raw_byte_count=self.raw_byte_count,
            fault=fault,
            label=label,
            description="streamed pretty JSON",
        )

    def matches(self, path: Path, *, acquired: AcquiredRoot | None = None) -> bool:
        return _matches_identity(
            path,
            raw_sha256=self.raw_sha256,
            raw_byte_count=self.raw_byte_count,
            acquired=acquired,
        )


def build_streamed_canonical_value(value: object) -> StreamedCanonicalValue:
    raw_sha256, raw_byte_count = _identity(_value_chunks(value))
    return StreamedCanonicalValue(value, raw_sha256, raw_byte_count)


def build_streamed_pretty_json_value(value: object) -> StreamedPrettyJsonValue:
    raw_sha256, raw_byte_count = _identity(_pretty_value_chunks(value))
    return StreamedPrettyJsonValue(value, raw_sha256, raw_byte_count)


def build_streamed_canonical_object(
    *,
    static_fields: Mapping[str, object],
    array_key: str,
    items: ItemReplay,
    fingerprint_key: str | None = None,
) -> StreamedCanonicalObject:
    """Freeze exact canonical bytes while retaining only scalar stream identity."""

    return build_streamed_canonical_object_arrays(
        static_fields=static_fields,
        array_items={array_key: items},
        fingerprint_key=fingerprint_key,
    )


def build_streamed_canonical_object_arrays(
    *,
    static_fields: Mapping[str, object],
    array_items: Mapping[str, ItemReplay],
    fingerprint_key: str | None = None,
) -> StreamedCanonicalObject:
    """Freeze canonical bytes with multiple independently replayable arrays."""

    if not array_items or any(not isinstance(key, str) or not key for key in array_items):
        raise CanonicalJsonError("streamed array keys must be nonempty strings")
    if any(not callable(replay) for replay in array_items.values()):
        raise TypeError("streamed JSON items must be replayable")
    if fingerprint_key is not None and (
        not isinstance(fingerprint_key, str)
        or not fingerprint_key
        or fingerprint_key in array_items
    ):
        raise CanonicalJsonError("streamed fingerprint key is invalid")
    frozen_static = dict(static_fields)
    frozen_arrays = dict(array_items)
    fingerprint = None
    if fingerprint_key is not None:
        fingerprint, _count = _identity(
            _object_chunks(
                frozen_static,
                array_items=frozen_arrays,
                fingerprint_key=None,
                fingerprint=None,
            )
        )
    raw_sha256, raw_byte_count = _identity(
        _object_chunks(
            frozen_static,
            array_items=frozen_arrays,
            fingerprint_key=fingerprint_key,
            fingerprint=fingerprint,
        )
    )
    return StreamedCanonicalObject(
        static_fields=frozen_static,
        array_items=frozen_arrays,
        fingerprint_key=fingerprint_key,
        fingerprint=fingerprint,
        raw_sha256=raw_sha256,
        raw_byte_count=raw_byte_count,
    )
