"""Replayable encoding source, scalar argv, and bounded production parsers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from pathlib import PurePosixPath
import re
import secrets
import struct
from typing import BinaryIO, Callable, Iterator, Literal, Protocol
import unicodedata
import zlib

from ...io.nofollow_io import AcquiredRoot, NoFollowPathError


ENCODING_CONTROL_FRAME_CAP = 4_194_304
FFMPEG_IMAGE2_START_NUMBER_MAX = 2_147_483_647
MAX_SUPPORTED_IMAGE2_INDEX = 2_147_483_646
FINAL_ENCODING_CONTROL_BUDGET = 8 * 1024 * 1024
FINAL_ENCODING_CONTROL_OVERHEAD = 4 * 1024 * 1024
FINAL_ENCODING_IMAGE_STREAM_BYTES = 1 * 1024 * 1024
FINAL_ENCODING_JSON_STREAM_BYTES = 1 * 1024 * 1024
FINAL_FIXED_TARGET_HASH_STREAM_BYTES = 1 * 1024 * 1024
FINAL_ENCODE_DIAGNOSTIC_BYTES = 256 * 1024
FINAL_PROBE_STDOUT_BYTES = 64 * 1024
FINAL_PROBE_STDERR_BYTES = 64 * 1024
FINAL_PROBE_JSON_STATE_BYTES = 512 * 1024
FINAL_DECODE_DIAGNOSTIC_BYTES = 256 * 1024
FINAL_DECODE_PROGRESS_STATE_BYTES = 64 * 1024
FINAL_AUDIO_PROBE_STDOUT_BYTES = 32 * 1024
FINAL_AUDIO_PROBE_STDERR_BYTES = 32 * 1024
FINAL_AUDIO_PROBE_JSON_STATE_BYTES = 256 * 1024
FINAL_AUDIO_DECODE_DIAGNOSTIC_BYTES = 256 * 1024
FINAL_RESERVATION_INDEX_STREAM_BYTES = 256 * 1024
FINAL_RESERVATION_INDEX_JSON_STATE_BYTES = 256 * 1024
FINAL_ENCODING_LINE_BYTES = 8 * 1024

JOB_OUTPUT_NAME_V1_RESOLUTION_PRESETS = frozenset(
    {
        "square-480",
        "square-720",
        "square-1k",
        "square-2k",
        "square-3k",
        "square-4k",
        "square-5k",
        "16x9-480p",
        "16x9-720p",
        "16x9-1080p",
        "16x9-1440p",
        "16x9-4k",
        "16x9-5k",
        "16x9-8k",
        "ultrawide",
        "wide-2k",
        "wide-4k",
        "cinema-2k",
        "cinema-4k",
    }
)

_HEX_32 = re.compile(r"^[0-9a-f]{32}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_CUSTOM_RESOLUTION = re.compile(r"^custom:([1-9][0-9]{0,4})x([1-9][0-9]{0,4})$")
_PROGRESS_KEY = re.compile(rb"^[a-z0-9_]{1,32}$")
_CANONICAL_U64 = re.compile(rb"^(?:0|[1-9][0-9]*)$")
_FORBIDDEN_FILENAME = frozenset('<>:"/\\|?*')
_DEVICE_NAMES = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
        "com¹",
        "com²",
        "com³",
        "lpt¹",
        "lpt²",
        "lpt³",
    }
)

EncodingMode = Literal["direct_stereo", "assembled_vr"]
CancellationPredicate = Callable[[], bool]


def _raise_if_cancelled(cancelled: CancellationPredicate | None) -> None:
    if cancelled is not None and cancelled():
        raise InterruptedError("final encoding input audit was cancelled")


class EncodingProviderError(RuntimeError):
    """Base error for a replayable encoding generation."""


class EncodingImage2RangeError(EncodingProviderError):
    """The signed-int image2 endpoint cannot be represented safely."""


class EncodingProviderChangedError(EncodingProviderError):
    """A pre-launch generation changed and may be audited again."""


class FinalEncodingInputChangedError(EncodingProviderError):
    """An input changed after the FFmpeg launch boundary."""


class FinalMediaTargetMetadataError(ValueError):
    """Output naming metadata cannot produce a portable component."""


class FfprobeJsonError(ValueError):
    """A bounded FFprobe JSON payload is outside its closed grammar."""


class FfprobeJsonBudgetError(FfprobeJsonError):
    """A fixed FFprobe JSON parser-state or token budget was exceeded."""


class EncodingProgressError(ValueError):
    """The FFmpeg progress stream is malformed or incomplete."""


class FinalContainerValidationError(ValueError):
    """The generated container does not match its exact encoding contract."""


class FinalContainerAudioPacketBoundError(FinalContainerValidationError):
    """The generated AAC stream exceeds its preflight packet-count bound."""


class EncodingDirectoryError(EncodingProviderError):
    """A selected image parent is incomplete, noncanonical, or unsafe."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_frame_stem(index: int) -> str:
    decimal = str(index)
    return "frame_" + decimal.zfill(max(6, len(decimal)))


def _validate_parent(path: PurePosixPath | None) -> None:
    if path is None:
        return
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise EncodingProviderError("encoding parent must be a contained canonical POSIX path")
    if any("\\" in part or "\x00" in part for part in path.parts):
        raise EncodingProviderError("encoding parent contains a forbidden path scalar")


@dataclass(frozen=True, slots=True)
class EncodingSequenceItem:
    source_index: int
    frame_index: int
    frame_name: str
    left_relative_path: PurePosixPath | None
    right_relative_path: PurePosixPath | None
    frame_relative_path: PurePosixPath | None
    audit_generation_id: str
    generation_fingerprint: str


@dataclass(frozen=True, slots=True)
class EncodingSequenceProvider:
    mode: EncodingMode
    frame_count: int
    first_index: int
    last_index: int
    left_parent: PurePosixPath | None
    right_parent: PurePosixPath | None
    frames_parent: PurePosixPath | None
    audit_generation_id: str
    generation_fingerprint: str

    @classmethod
    def freeze(
        cls,
        *,
        mode: EncodingMode,
        frame_count: int,
        first_index: int,
        left_parent: PurePosixPath | None = None,
        right_parent: PurePosixPath | None = None,
        frames_parent: PurePosixPath | None = None,
    ) -> EncodingSequenceProvider:
        last_index = _checked_image2_range(frame_count, first_index)
        generation_id = secrets.token_hex(16)
        generation = _generation_object(
            audit_generation_id=generation_id,
            mode=mode,
            frame_count=frame_count,
            first_index=first_index,
            last_index=last_index,
            left_parent=left_parent,
            right_parent=right_parent,
            frames_parent=frames_parent,
        )
        return cls(
            mode=mode,
            frame_count=frame_count,
            first_index=first_index,
            last_index=last_index,
            left_parent=left_parent,
            right_parent=right_parent,
            frames_parent=frames_parent,
            audit_generation_id=generation_id,
            generation_fingerprint=hashlib.sha256(canonical_json_bytes(generation)).hexdigest(),
        ).validated()

    def validated(self) -> EncodingSequenceProvider:
        _checked_image2_range(self.frame_count, self.first_index)
        if self.last_index != self.first_index + self.frame_count - 1:
            raise EncodingProviderError("encoding provider last index is invalid")
        if self.mode == "direct_stereo":
            if (
                self.left_parent is None
                or self.right_parent is None
                or self.frames_parent is not None
            ):
                raise EncodingProviderError("direct provider parent shape is invalid")
        elif self.mode == "assembled_vr":
            if (
                self.frames_parent is None
                or self.left_parent is not None
                or self.right_parent is not None
            ):
                raise EncodingProviderError("assembled provider parent shape is invalid")
        else:
            raise EncodingProviderError("encoding provider mode is invalid")
        for parent in (self.left_parent, self.right_parent, self.frames_parent):
            _validate_parent(parent)
        if not _HEX_32.fullmatch(self.audit_generation_id):
            raise EncodingProviderError("encoding audit generation ID is invalid")
        if not _HEX_64.fullmatch(self.generation_fingerprint):
            raise EncodingProviderError("encoding generation fingerprint is invalid")
        expected = hashlib.sha256(
            canonical_json_bytes(
                _generation_object(
                    audit_generation_id=self.audit_generation_id,
                    mode=self.mode,
                    frame_count=self.frame_count,
                    first_index=self.first_index,
                    last_index=self.last_index,
                    left_parent=self.left_parent,
                    right_parent=self.right_parent,
                    frames_parent=self.frames_parent,
                )
            )
        ).hexdigest()
        if self.generation_fingerprint != expected:
            raise EncodingProviderError(
                "encoding generation fingerprint does not match its scalars"
            )
        return self

    def replay(self) -> Iterator[EncodingSequenceItem]:
        self.validated()
        for source_index in range(self.frame_count):
            frame_index = self.first_index + source_index
            frame_name = _canonical_frame_stem(frame_index)
            filename = f"{frame_name}.png"
            yield EncodingSequenceItem(
                source_index=source_index,
                frame_index=frame_index,
                frame_name=frame_name,
                left_relative_path=(
                    self.left_parent / filename if self.left_parent is not None else None
                ),
                right_relative_path=(
                    self.right_parent / filename if self.right_parent is not None else None
                ),
                frame_relative_path=(
                    self.frames_parent / filename if self.frames_parent is not None else None
                ),
                audit_generation_id=self.audit_generation_id,
                generation_fingerprint=self.generation_fingerprint,
            )


def _checked_image2_range(frame_count: int, first_index: int) -> int:
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count < 1
        or frame_count > ENCODING_CONTROL_FRAME_CAP
    ):
        raise EncodingImage2RangeError("encoding frame count is outside the control range")
    if isinstance(first_index, bool) or not isinstance(first_index, int) or first_index < 0:
        raise EncodingImage2RangeError("encoding first index is invalid")
    last_index = first_index + frame_count - 1
    if first_index > MAX_SUPPORTED_IMAGE2_INDEX or last_index > MAX_SUPPORTED_IMAGE2_INDEX:
        raise EncodingImage2RangeError("encoding image2 endpoint exceeds the signed-int boundary")
    return last_index


def _generation_object(
    *,
    audit_generation_id: str,
    mode: str,
    frame_count: int,
    first_index: int,
    last_index: int,
    left_parent: PurePosixPath | None,
    right_parent: PurePosixPath | None,
    frames_parent: PurePosixPath | None,
) -> dict[str, object]:
    return {
        "audit_generation_id": audit_generation_id,
        "mode": mode,
        "frame_count": frame_count,
        "first_index": first_index,
        "last_index": last_index,
        "left_parent": left_parent.as_posix() if left_parent is not None else None,
        "right_parent": right_parent.as_posix() if right_parent is not None else None,
        "frames_parent": frames_parent.as_posix() if frames_parent is not None else None,
    }


@dataclass(frozen=True)
class EncodingProviderAudit:
    audit_generation_id: str
    generation_fingerprint: str
    item_stream_sha256: str
    frame_count: int


def _validated_items(
    provider: EncodingSequenceProvider,
    cancelled: CancellationPredicate | None = None,
) -> Iterator[EncodingSequenceItem]:
    expected_source = 0
    for item in provider.replay():
        _raise_if_cancelled(cancelled)
        if item.source_index != expected_source:
            raise EncodingProviderError("encoding source index is noncontiguous")
        expected_frame = provider.first_index + expected_source
        if item.frame_index != expected_frame or item.frame_name != _canonical_frame_stem(
            expected_frame
        ):
            raise EncodingProviderError("encoding frame index or stem is noncanonical")
        if (
            item.audit_generation_id != provider.audit_generation_id
            or item.generation_fingerprint != provider.generation_fingerprint
        ):
            raise EncodingProviderError("encoding item generation changed")
        expected_source += 1
        yield item
    if expected_source != provider.frame_count:
        raise EncodingProviderError("encoding replay count changed")


def _item_stream_hash(
    provider: EncodingSequenceProvider,
    cancelled: CancellationPredicate | None = None,
) -> str:
    digest = hashlib.sha256()
    for item in _validated_items(provider, cancelled):
        digest.update(
            canonical_json_bytes(
                {
                    "source_index": item.source_index,
                    "frame_index": item.frame_index,
                    "frame_name": item.frame_name,
                    "left": (
                        item.left_relative_path.as_posix()
                        if item.left_relative_path is not None
                        else None
                    ),
                    "right": (
                        item.right_relative_path.as_posix()
                        if item.right_relative_path is not None
                        else None
                    ),
                    "frame": (
                        item.frame_relative_path.as_posix()
                        if item.frame_relative_path is not None
                        else None
                    ),
                }
            )
        )
    return digest.hexdigest()


def audit_encoding_provider(
    provider: EncodingSequenceProvider,
    *,
    cancelled: CancellationPredicate | None = None,
) -> EncodingProviderAudit:
    _raise_if_cancelled(cancelled)
    provider.validated()
    return EncodingProviderAudit(
        provider.audit_generation_id,
        provider.generation_fingerprint,
        _item_stream_hash(provider, cancelled),
        provider.frame_count,
    )


def validate_encoding_provider(
    provider: EncodingSequenceProvider,
    evidence: EncodingProviderAudit,
    *,
    launch_started: bool,
    cancelled: CancellationPredicate | None = None,
) -> None:
    error_type = FinalEncodingInputChangedError if launch_started else EncodingProviderChangedError
    try:
        current = audit_encoding_provider(provider, cancelled=cancelled)
    except InterruptedError:
        raise
    except (EncodingProviderError, TypeError, ValueError) as error:
        raise error_type(str(error)) from error
    if current != evidence:
        raise error_type("encoding provider replay identity changed")


@dataclass(frozen=True)
class EncodingPngHeader:
    width: int
    height: int
    bit_depth: int
    color_type: int


@dataclass
class EncodingManifestTrace:
    maximum_live_identities: int = 0
    retained_item_lists: int = 0
    maximum_stream_bytes: int = 0


@dataclass(frozen=True)
class StagedEncodingInputManifest:
    provider_audit: EncodingProviderAudit
    fingerprint: str
    raw_sha256: str
    raw_byte_count: int
    first_headers: tuple[tuple[int, int], ...]
    scale_required: bool
    trace: EncodingManifestTrace


class _ByteSink(Protocol):
    def write(self, payload: bytes) -> int: ...


class _DigestSink:
    def __init__(self) -> None:
        self.digest = hashlib.sha256()
        self.byte_count = 0

    def write(self, payload: bytes) -> int:
        self.digest.update(payload)
        self.byte_count += len(payload)
        return len(payload)


@dataclass
class _ManifestShapeState:
    left: tuple[int, int] | None = None
    right: tuple[int, int] | None = None
    frame: tuple[int, int] | None = None
    left_consistent: bool = True
    right_consistent: bool = True


def _write_exact(sink: _ByteSink, payload: bytes) -> None:
    if sink.write(payload) != len(payload):
        raise OSError("encoding manifest sink returned a short write")


def _parse_png_header(raw: bytes, path: Path) -> EncodingPngHeader:
    if len(raw) != 33 or raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise EncodingDirectoryError(f"encoding image has an invalid PNG signature: {path}")
    length = struct.unpack_from(">I", raw, 8)[0]
    if length != 13 or raw[12:16] != b"IHDR":
        raise EncodingDirectoryError(f"encoding image has no canonical IHDR: {path}")
    width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack_from(
        ">IIBBBBB", raw, 16
    )
    expected_crc = struct.unpack_from(">I", raw, 29)[0]
    if zlib.crc32(raw[12:29]) & 0xFFFFFFFF != expected_crc:
        raise EncodingDirectoryError(f"encoding image IHDR CRC is invalid: {path}")
    if (
        width <= 0
        or height <= 0
        or bit_depth != 8
        or color_type != 2
        or compression != 0
        or filtering != 0
        or interlace not in {0, 1}
    ):
        raise EncodingDirectoryError(f"encoding image PNG header is outside RGB8: {path}")
    return EncodingPngHeader(width, height, bit_depth, color_type)


def _image_identity(
    acquired: AcquiredRoot,
    relative_path: PurePosixPath,
    trace: EncodingManifestTrace,
    cancelled: CancellationPredicate | None = None,
) -> tuple[dict[str, object], EncodingPngHeader]:
    path = acquired.root.joinpath(*relative_path.parts)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with acquired.open_regular(path) as handle:
            header_raw = handle.read(33)
            header = _parse_png_header(header_raw, path)
            handle.seek(0)
            while True:
                _raise_if_cancelled(cancelled)
                chunk = handle.read(FINAL_ENCODING_IMAGE_STREAM_BYTES)
                if not chunk:
                    break
                trace.maximum_stream_bytes = max(trace.maximum_stream_bytes, len(chunk))
                digest.update(chunk)
                byte_count += len(chunk)
    except NoFollowPathError as error:
        raise EncodingDirectoryError(str(error)) from error
    trace.maximum_live_identities = max(trace.maximum_live_identities, 1)
    return (
        {
            "relative_path": relative_path.as_posix(),
            "sha256": digest.hexdigest(),
            "byte_count": byte_count,
            "png_header": {
                "width": header.width,
                "height": header.height,
                "bit_depth": header.bit_depth,
                "color_type": header.color_type,
            },
        },
        header,
    )


_FRAME_FILE = re.compile(r"^frame_([0-9]+)\.png$")


def _validate_selected_directory(
    acquired: AcquiredRoot,
    parent: PurePosixPath,
    provider: EncodingSequenceProvider,
    cancelled: CancellationPredicate | None = None,
) -> None:
    directory = acquired.root.joinpath(*parent.parts)
    try:
        count = 0
        for entry in acquired.iter_directory(directory):
            _raise_if_cancelled(cancelled)
            match = _FRAME_FILE.fullmatch(entry.name)
            if match is None:
                folded = entry.name.casefold()
                if folded.startswith("frame_") and folded.endswith(".png"):
                    raise EncodingDirectoryError(
                        "encoding directory contains a noncanonical frame name"
                    )
                continue
            index = int(match.group(1))
            if entry.name != f"{_canonical_frame_stem(index)}.png":
                raise EncodingDirectoryError(
                    "encoding directory contains a noncanonical frame name"
                )
            if index < provider.first_index or index > provider.last_index:
                raise EncodingDirectoryError(
                    "encoding directory frame is outside the declared range"
                )
            if not entry.is_regular:
                raise EncodingDirectoryError("encoding directory frame is not an ordinary file")
            count += 1
    except NoFollowPathError as error:
        raise EncodingDirectoryError(str(error)) from error
    if count != provider.frame_count:
        raise EncodingDirectoryError(
            f"encoding directory frame count {count} does not equal {provider.frame_count}"
        )


def _selected_parents(provider: EncodingSequenceProvider) -> tuple[PurePosixPath, ...]:
    if provider.mode == "direct_stereo":
        assert provider.left_parent is not None and provider.right_parent is not None
        return provider.left_parent, provider.right_parent
    assert provider.frames_parent is not None
    return (provider.frames_parent,)


def _write_identity_array(
    *,
    sink: _ByteSink,
    self_digest: hashlib._Hash,
    provider: EncodingSequenceProvider,
    acquired: AcquiredRoot,
    field: str,
    trace: EncodingManifestTrace,
    shape_state: _ManifestShapeState,
    cancelled: CancellationPredicate | None = None,
) -> None:
    _write_exact(sink, b"[")
    self_digest.update(b"[")
    for source_index, item in enumerate(_validated_items(provider, cancelled)):
        relative = {
            "left": item.left_relative_path,
            "right": item.right_relative_path,
            "frames": item.frame_relative_path,
        }[field]
        if relative is None:
            raise EncodingProviderError(f"encoding manifest {field} path is unexpectedly null")
        identity, header = _image_identity(
            acquired,
            relative,
            trace,
            cancelled,
        )
        shape = (header.width, header.height)
        previous = getattr(shape_state, "frame" if field == "frames" else field)
        if previous is None:
            setattr(shape_state, "frame" if field == "frames" else field, shape)
        elif previous != shape and field in {"left", "right"}:
            setattr(shape_state, f"{field}_consistent", False)
        elif previous != shape:
            raise EncodingDirectoryError(
                "assembled encoding image shape changes within the sequence"
            )
        if source_index:
            _write_exact(sink, b",")
            self_digest.update(b",")
        encoded = canonical_json_bytes(identity)
        _write_exact(sink, encoded)
        self_digest.update(encoded)
        del identity
    _write_exact(sink, b"]")
    self_digest.update(b"]")


def _write_manifest_stream(  # noqa: C901
    *,
    provider: EncodingSequenceProvider,
    acquired: AcquiredRoot,
    sink: _ByteSink,
    fingerprint_value: str,
    per_eye_size: tuple[int, int],
    vr_format: Literal["side_by_side", "over_under"],
    trace: EncodingManifestTrace,
    cancelled: CancellationPredicate | None = None,
) -> tuple[str, tuple[tuple[int, int], ...], bool]:
    _raise_if_cancelled(cancelled)
    if not _HEX_64.fullmatch(fingerprint_value):
        raise ValueError("encoding manifest fingerprint placeholder is invalid")
    self_digest = hashlib.sha256()
    shape_state = _ManifestShapeState()
    complete_prefix = (
        b'{"algorithm_version":"encoding-input-sequence-v2","fingerprint":"'
        + fingerprint_value.encode("ascii")
        + b'"'
    )
    _write_exact(sink, complete_prefix)
    self_digest.update(b'{"algorithm_version":"encoding-input-sequence-v2"')

    segment = b',"frame_names":['
    _write_exact(sink, segment)
    self_digest.update(segment)
    for source_index, item in enumerate(_validated_items(provider, cancelled)):
        if source_index:
            _write_exact(sink, b",")
            self_digest.update(b",")
        encoded = canonical_json_bytes(item.frame_name)
        _write_exact(sink, encoded)
        self_digest.update(encoded)
    _write_exact(sink, b"]")
    self_digest.update(b"]")

    for field in ("frames", "left"):
        segment = f',"{field}":'.encode("ascii")
        _write_exact(sink, segment)
        self_digest.update(segment)
        present = (field == "frames") == (provider.mode == "assembled_vr")
        if present:
            _write_identity_array(
                sink=sink,
                self_digest=self_digest,
                provider=provider,
                acquired=acquired,
                field=field,
                trace=trace,
                shape_state=shape_state,
                cancelled=cancelled,
            )
        else:
            _write_exact(sink, b"null")
            self_digest.update(b"null")

    mode_segment = b',"mode":' + canonical_json_bytes(provider.mode)
    _write_exact(sink, mode_segment)
    self_digest.update(mode_segment)

    segment = b',"right":'
    _write_exact(sink, segment)
    self_digest.update(segment)
    if provider.mode == "direct_stereo":
        _write_identity_array(
            sink=sink,
            self_digest=self_digest,
            provider=provider,
            acquired=acquired,
            field="right",
            trace=trace,
            shape_state=shape_state,
            cancelled=cancelled,
        )
    else:
        _write_exact(sink, b"null")
        self_digest.update(b"null")
    suffix = b',"schema_version":1}'
    _write_exact(sink, suffix)
    self_digest.update(suffix)

    width, height = per_eye_size
    if width <= 0 or height <= 0:
        raise ValueError("encoding per-eye dimensions must be positive")
    headers: tuple[tuple[int, int], ...]
    if provider.mode == "assembled_vr":
        expected = (width * 2, height) if vr_format == "side_by_side" else (width, height * 2)
        if shape_state.frame != expected:
            raise EncodingDirectoryError("assembled encoding shape does not equal final VR size")
        headers = (shape_state.frame,)
        scale_required = False
    else:
        if not shape_state.left_consistent or not shape_state.right_consistent:
            raise EncodingDirectoryError("direct encoding shape changes within an eye sequence")
        assert shape_state.left is not None and shape_state.right is not None
        headers = (shape_state.left, shape_state.right)
        scale_required = headers != ((width, height), (width, height))
    return self_digest.hexdigest(), headers, scale_required


def stage_encoding_input_manifest(
    provider: EncodingSequenceProvider,
    *,
    job_root: Path,
    sink: BinaryIO,
    per_eye_size: tuple[int, int],
    vr_format: Literal["side_by_side", "over_under"],
    cancelled: CancellationPredicate | None = None,
) -> StagedEncodingInputManifest:
    provider_audit = audit_encoding_provider(provider, cancelled=cancelled)
    with AcquiredRoot(Path(job_root)) as acquired:
        for parent in _selected_parents(provider):
            _validate_selected_directory(acquired, parent, provider, cancelled)
        trace = EncodingManifestTrace()
        start = sink.tell()
        placeholder = "0" * 64
        fingerprint, headers, scale_required = _write_manifest_stream(
            provider=provider,
            acquired=acquired,
            sink=sink,
            fingerprint_value=placeholder,
            per_eye_size=per_eye_size,
            vr_format=vr_format,
            trace=trace,
            cancelled=cancelled,
        )
    end = sink.tell()
    fingerprint_offset = start + len(
        b'{"algorithm_version":"encoding-input-sequence-v2","fingerprint":"'
    )
    sink.seek(fingerprint_offset)
    _write_exact(sink, fingerprint.encode("ascii"))
    sink.seek(start)
    raw_digest = hashlib.sha256()
    remaining = end - start
    while remaining:
        _raise_if_cancelled(cancelled)
        chunk = sink.read(min(FINAL_ENCODING_JSON_STREAM_BYTES, remaining))
        if not chunk:
            raise OSError("encoding manifest sink ended before its written extent")
        trace.maximum_stream_bytes = max(trace.maximum_stream_bytes, len(chunk))
        raw_digest.update(chunk)
        remaining -= len(chunk)
    sink.seek(end)
    return StagedEncodingInputManifest(
        provider_audit=provider_audit,
        fingerprint=fingerprint,
        raw_sha256=raw_digest.hexdigest(),
        raw_byte_count=end - start,
        first_headers=headers,
        scale_required=scale_required,
        trace=trace,
    )


def validate_encoding_postpass(
    provider: EncodingSequenceProvider,
    *,
    job_root: Path,
    staged: StagedEncodingInputManifest,
    per_eye_size: tuple[int, int],
    vr_format: Literal["side_by_side", "over_under"],
    cancelled: CancellationPredicate | None = None,
) -> None:
    try:
        validate_encoding_provider(
            provider,
            staged.provider_audit,
            launch_started=True,
            cancelled=cancelled,
        )
        with AcquiredRoot(Path(job_root)) as acquired:
            for parent in _selected_parents(provider):
                _validate_selected_directory(acquired, parent, provider, cancelled)
            sink = _DigestSink()
            fingerprint, headers, scale_required = _write_manifest_stream(
                provider=provider,
                acquired=acquired,
                sink=sink,
                fingerprint_value=staged.fingerprint,
                per_eye_size=per_eye_size,
                vr_format=vr_format,
                trace=EncodingManifestTrace(),
                cancelled=cancelled,
            )
    except InterruptedError:
        raise
    except (EncodingProviderError, EncodingDirectoryError, OSError, ValueError) as error:
        if isinstance(error, FinalEncodingInputChangedError):
            raise
        raise FinalEncodingInputChangedError(str(error)) from error
    if (
        fingerprint != staged.fingerprint
        or sink.digest.hexdigest() != staged.raw_sha256
        or sink.byte_count != staged.raw_byte_count
        or headers != staged.first_headers
        or scale_required != staged.scale_required
    ):
        raise FinalEncodingInputChangedError("encoding manifest postpass changed")


def audit_encoding_input_manifest(
    provider: EncodingSequenceProvider,
    *,
    job_root: Path,
    expected_fingerprint: str,
    expected_raw_sha256: str,
    expected_raw_byte_count: int,
    per_eye_size: tuple[int, int],
    vr_format: Literal["side_by_side", "over_under"],
    cancelled: CancellationPredicate | None = None,
) -> StagedEncodingInputManifest:
    """Rebuild only streaming identity needed to adopt a durable manifest on resume."""

    provider_audit = audit_encoding_provider(provider, cancelled=cancelled)
    with AcquiredRoot(Path(job_root)) as acquired:
        for parent in _selected_parents(provider):
            _validate_selected_directory(acquired, parent, provider, cancelled)
        trace = EncodingManifestTrace()
        sink = _DigestSink()
        fingerprint, headers, scale_required = _write_manifest_stream(
            provider=provider,
            acquired=acquired,
            sink=sink,
            fingerprint_value=expected_fingerprint,
            per_eye_size=per_eye_size,
            vr_format=vr_format,
            trace=trace,
            cancelled=cancelled,
        )
    if (
        fingerprint != expected_fingerprint
        or sink.digest.hexdigest() != expected_raw_sha256
        or sink.byte_count != expected_raw_byte_count
    ):
        raise FinalEncodingInputChangedError("committed encoding manifest differs on resume")
    return StagedEncodingInputManifest(
        provider_audit=provider_audit,
        fingerprint=fingerprint,
        raw_sha256=expected_raw_sha256,
        raw_byte_count=expected_raw_byte_count,
        first_headers=headers,
        scale_required=scale_required,
        trace=trace,
    )


def _image_pattern(parent: PurePosixPath) -> str:
    return (parent / "frame_%06d.png").as_posix()


def build_image_encoding_argv(
    provider: EncodingSequenceProvider,
    *,
    ffmpeg: str,
    fps: str,
    vr_format: Literal["side_by_side", "over_under"],
    per_eye_size: tuple[int, int],
    first_headers: tuple[tuple[int, int], ...],
    output_path: str,
) -> tuple[str, ...]:
    """Build image2 argv from provider scalars, never from an assembled frame list."""

    provider.validated()
    if not re.fullmatch(r"[1-9][0-9]*/[1-9][0-9]*", fps):
        raise ValueError("encoding FPS must be a positive canonical rational")
    width, height = per_eye_size
    if width <= 0 or height <= 0:
        raise ValueError("encoding per-eye dimensions must be positive")
    if vr_format not in {"side_by_side", "over_under"}:
        raise ValueError("encoding VR format is invalid")
    common = ["-framerate", fps, "-start_number", str(provider.first_index)]
    argv: list[str] = [ffmpeg, "-v", "error", "-nostats", "-progress", "pipe:1"]
    if provider.mode == "direct_stereo":
        if provider.left_parent is None or provider.right_parent is None or len(first_headers) != 2:
            raise EncodingProviderError("direct encoding provider/header shape is invalid")
        argv.extend([*common, "-i", _image_pattern(provider.left_parent)])
        argv.extend([*common, "-i", _image_pattern(provider.right_parent)])
        stack = "hstack" if vr_format == "side_by_side" else "vstack"
        if all(header == (width, height) for header in first_headers):
            graph = (
                "[0:v]setsar=1[left];[1:v]setsar=1[right];"
                f"[left][right]{stack}=inputs=2:shortest=1[stacked];"
                "[stacked]setsar=1[vr]"
            )
        else:
            scale = f"scale={width}:{height}:flags=bicubic+accurate_rnd,setsar=1"
            graph = (
                f"[0:v]{scale}[left];[1:v]{scale}[right];"
                f"[left][right]{stack}=inputs=2:shortest=1[stacked];"
                "[stacked]setsar=1[vr]"
            )
        argv.extend(["-filter_complex", graph, "-map", "[vr]"])
    else:
        if provider.frames_parent is None or len(first_headers) != 1:
            raise EncodingProviderError("assembled encoding provider/header shape is invalid")
        final_size = (width * 2, height) if vr_format == "side_by_side" else (width, height * 2)
        if first_headers[0] != final_size:
            raise EncodingProviderError("assembled image shape does not equal final VR shape")
        argv.extend([*common, "-i", _image_pattern(provider.frames_parent)])
        argv.extend(["-vf", "setsar=1", "-map", "0:v:0"])
    argv.extend(["-frames:v", str(provider.frame_count), output_path])
    return tuple(argv)


def _ascii_fold(value: str) -> str:
    return "".join(chr(ord(char) + 32) if "A" <= char <= "Z" else char for char in value)


def is_portable_final_component_v1(component: str) -> bool:
    if not isinstance(component, str) or not component or component in {".", ".."}:
        return False
    for char in component:
        category = unicodedata.category(char)
        if category in {"Cc", "Cs"} or char in _FORBIDDEN_FILENAME:
            return False
    if component.endswith((" ", ".")):
        return False
    try:
        if len(component.encode("utf-8")) > 240:
            return False
        if len(component.encode("utf-16-le")) // 2 > 240:
            return False
    except UnicodeEncodeError:
        return False
    device_stem = component.split(".", 1)[0]
    if _ascii_fold(device_stem) in _DEVICE_NAMES:
        return False
    return True


def _resolution_token(resolution: str) -> str:
    if resolution == "auto":
        return "auto"
    if resolution in JOB_OUTPUT_NAME_V1_RESOLUTION_PRESETS:
        return resolution
    match = _CUSTOM_RESOLUTION.fullmatch(resolution)
    if match is None:
        raise FinalMediaTargetMetadataError("resolution is outside job-output-name-v1")
    width, height = (int(match.group(1)), int(match.group(2)))
    if width > 10_000 or height > 10_000:
        raise FinalMediaTargetMetadataError("custom resolution exceeds 10,000")
    return f"custom-{width}x{height}"


def job_output_name_v1(
    *,
    source_video_name: str,
    vr_format: Literal["side_by_side", "over_under"],
    resolution: str,
) -> str:
    if not isinstance(source_video_name, str):
        raise FinalMediaTargetMetadataError("source video name must be a string")
    if any(char == "/" or unicodedata.category(char) in {"Cc", "Cs"} for char in source_video_name):
        raise FinalMediaTargetMetadataError("source video name contains a forbidden scalar")
    if source_video_name:
        final_dot = source_video_name.rfind(".")
        if 0 < final_dot < len(source_video_name) - 1:
            stem = source_video_name[:final_dot]
        else:
            stem = source_video_name
    else:
        stem = "output"
    sanitized_chars = ["_" if char in _FORBIDDEN_FILENAME else char for char in stem]
    sanitized = re.sub(r"_+", "_", "".join(sanitized_chars)).strip("_")
    format_token = {
        "side_by_side": "side-by-side",
        "over_under": "over-under",
    }.get(vr_format)
    if format_token is None:
        raise FinalMediaTargetMetadataError("VR format is outside job-output-name-v1")
    resolution_token = _resolution_token(resolution)
    tail = f"_3D_{format_token}_{resolution_token}.mp4"
    candidate = sanitized + tail
    if is_portable_final_component_v1(candidate):
        return candidate
    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:32]
    fallback_tail = f"~{digest}{tail}"
    for prefix_length in range(len(sanitized), -1, -1):
        fallback = sanitized[:prefix_length] + fallback_tail
        if is_portable_final_component_v1(fallback):
            return fallback
    raise FinalMediaTargetMetadataError("zero-prefix fallback is not portable")


_JSON_NUMBER = re.compile(rb"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")


class _BoundedFfprobeJsonCursor:
    """Byte cursor for the two fixed FFprobe schemas; never builds a JSON tree."""

    def __init__(self, raw: bytes, *, raw_cap: int, max_depth: int) -> None:
        if len(raw) > raw_cap:
            raise FfprobeJsonBudgetError("FFprobe JSON exceeds its raw byte cap")
        if b"\x00" in raw:
            raise FfprobeJsonError("FFprobe JSON contains NUL")
        self.raw = raw
        self.offset = 0
        self.max_depth = max_depth
        self.ignored_container_starts = 0

    def _space(self) -> None:
        raw = self.raw
        offset = self.offset
        while offset < len(raw) and raw[offset] in b" \t\r\n":
            offset += 1
        self.offset = offset

    def peek(self) -> int | None:
        self._space()
        if self.offset == len(self.raw):
            return None
        return self.raw[self.offset]

    def take(self, byte: int, label: str = "FFprobe JSON") -> None:
        self._space()
        if self.offset == len(self.raw) or self.raw[self.offset] != byte:
            raise FfprobeJsonError(f"{label} is malformed")
        self.offset += 1

    def finish(self) -> None:
        self._space()
        if self.offset != len(self.raw):
            raise FfprobeJsonError("FFprobe JSON has trailing data")

    def enter_known_container(self, byte: int, *, depth: int, label: str) -> None:
        if depth > self.max_depth:
            raise FfprobeJsonBudgetError("FFprobe JSON nesting depth exceeds the cap")
        self.take(byte, label)

    def string(self, *, maximum_bytes: int, label: str) -> str:
        self._space()
        start = self.offset
        if start == len(self.raw) or self.raw[start] != ord('"'):
            raise FfprobeJsonError(f"{label} must be a JSON string")
        offset = start + 1
        escaped = False
        while offset < len(self.raw):
            byte = self.raw[offset]
            if byte < 0x20:
                raise FfprobeJsonError(f"{label} contains an invalid control byte")
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                token = self.raw[start : offset + 1]
                if len(token) - 2 > maximum_bytes:
                    raise FfprobeJsonBudgetError(
                        f"{label} scalar token exceeds its byte cap"
                    )
                try:
                    value = json.loads(token)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise FfprobeJsonError(f"{label} is malformed") from error
                if not isinstance(value, str):
                    raise FfprobeJsonError(f"{label} must be a JSON string")
                try:
                    encoded_length = len(value.encode("utf-8", errors="strict"))
                except UnicodeEncodeError as error:
                    raise FfprobeJsonError(f"{label} contains invalid Unicode") from error
                if encoded_length > maximum_bytes:
                    raise FfprobeJsonBudgetError(
                        f"{label} scalar token exceeds its byte cap"
                    )
                self.offset = offset + 1
                return value
            offset += 1
        raise FfprobeJsonError(f"{label} is unterminated")

    def scalar(self, *, maximum_bytes: int, label: str) -> object:
        self._space()
        if self.offset == len(self.raw):
            raise FfprobeJsonError(f"{label} is missing")
        if self.raw[self.offset] == ord('"'):
            return self.string(maximum_bytes=maximum_bytes, label=label)
        start = self.offset
        while self.offset < len(self.raw) and self.raw[self.offset] not in b" \t\r\n,]}":
            self.offset += 1
        token = self.raw[start : self.offset]
        if not token or len(token) > maximum_bytes:
            raise FfprobeJsonBudgetError(f"{label} scalar token exceeds its byte cap")
        if token == b"true":
            return True
        if token == b"false":
            return False
        if token == b"null":
            return None
        if _JSON_NUMBER.fullmatch(token) is None:
            raise FfprobeJsonError(f"{label} is malformed")
        try:
            return json.loads(token)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise FfprobeJsonError(f"{label} is malformed") from error

    def skip_ignored_value(self, *, depth: int) -> None:  # noqa: C901
        """Consume one ignored JSON value with fixed depth/container/scalar caps."""

        # A frame is [kind, state, depth]. Parent state advances before a child
        # starts, so closing a child never needs recursion or a Python value.
        frames: list[list[object]] = []

        def start_value(value_depth: int) -> None:
            self._space()
            byte = self.peek()
            if byte not in {ord("{"), ord("[")}:
                self.scalar(maximum_bytes=128, label="ignored FFprobe JSON")
                return
            if value_depth > self.max_depth:
                raise FfprobeJsonBudgetError(
                    "FFprobe JSON nesting depth exceeds the cap"
                )
            self.ignored_container_starts += 1
            if self.ignored_container_starts > 256:
                raise FfprobeJsonBudgetError(
                    "ignored FFprobe JSON container count exceeds the cap"
                )
            self.offset += 1
            frames.append(
                ["object" if byte == ord("{") else "array", "first", value_depth]
            )

        start_value(depth)
        while frames:
            kind, state, container_depth = frames[-1]
            self._space()
            byte = self.peek()
            if kind == "object":
                if state == "first":
                    if byte == ord("}"):
                        self.offset += 1
                        frames.pop()
                        continue
                    self.string(maximum_bytes=128, label="ignored FFprobe JSON key")
                    self.take(ord(":"), "ignored FFprobe JSON object")
                    frames[-1][1] = "after_value"
                    start_value(int(container_depth) + 1)
                    continue
                if byte == ord("}"):
                    self.offset += 1
                    frames.pop()
                    continue
                self.take(ord(","), "ignored FFprobe JSON object")
                self.string(maximum_bytes=128, label="ignored FFprobe JSON key")
                self.take(ord(":"), "ignored FFprobe JSON object")
                start_value(int(container_depth) + 1)
                continue
            if state == "first":
                if byte == ord("]"):
                    self.offset += 1
                    frames.pop()
                    continue
                frames[-1][1] = "after_value"
                start_value(int(container_depth) + 1)
                continue
            if byte == ord("]"):
                self.offset += 1
                frames.pop()
                continue
            self.take(ord(","), "ignored FFprobe JSON array")
            start_value(int(container_depth) + 1)


def _next_member(cursor: _BoundedFfprobeJsonCursor, *, first: bool, label: str) -> bool:
    """Position an object/array parser at its next member; return false at close."""

    closing = ord("}") if label.endswith("object") else ord("]")
    if cursor.peek() == closing:
        cursor.offset += 1
        return False
    if not first:
        cursor.take(ord(","), label)
    return True


@dataclass(frozen=True)
class AudioProbeStream:
    codec_type: str
    sample_rate: int
    channels: int


def parse_audio_ffprobe_json(raw: bytes) -> AudioProbeStream | None:  # noqa: C901
    cursor = _BoundedFfprobeJsonCursor(
        raw,
        raw_cap=FINAL_AUDIO_PROBE_STDOUT_BYTES,
        max_depth=8,
    )
    cursor.enter_known_container(ord("{"), depth=1, label="audio FFprobe root")
    root_keys: set[str] = set()
    stream: dict[str, object] | None = None
    first_root = True
    while cursor.peek() != ord("}"):
        if not first_root:
            cursor.take(ord(","), "audio FFprobe root object")
        first_root = False
        key = cursor.string(maximum_bytes=128, label="audio FFprobe root key")
        if key not in {"streams", "programs", "stream_groups"}:
            raise FfprobeJsonError("audio FFprobe root keys are invalid")
        if key in root_keys:
            raise FfprobeJsonError(f"duplicate authoritative root key: {key}")
        root_keys.add(key)
        cursor.take(ord(":"), "audio FFprobe root object")
        if key == "streams":
            cursor.enter_known_container(
                ord("["), depth=2, label="audio FFprobe streams array"
            )
            if cursor.peek() != ord("]"):
                cursor.enter_known_container(
                    ord("{"), depth=3, label="audio FFprobe stream object"
                )
                fields: dict[str, object] = {}
                first_field = True
                while cursor.peek() != ord("}"):
                    if not first_field:
                        cursor.take(ord(","), "audio FFprobe stream object")
                    first_field = False
                    field = cursor.string(
                        maximum_bytes=128,
                        label="audio FFprobe stream key",
                    )
                    if field not in {"codec_type", "sample_rate", "channels"}:
                        raise FfprobeJsonError("audio FFprobe stream keys are invalid")
                    if field in fields:
                        raise FfprobeJsonError(
                            f"duplicate authoritative stream key: {field}"
                        )
                    cursor.take(ord(":"), "audio FFprobe stream object")
                    fields[field] = cursor.scalar(
                        maximum_bytes=32,
                        label=f"audio FFprobe {field}",
                    )
                cursor.take(ord("}"), "audio FFprobe stream object")
                stream = fields
                if cursor.peek() != ord("]"):
                    raise FfprobeJsonError(
                        "audio FFprobe streams must contain zero or one object"
                    )
            cursor.take(ord("]"), "audio FFprobe streams array")
        else:
            cursor.enter_known_container(
                ord("["), depth=2, label=f"audio FFprobe {key} wrapper"
            )
            first_item = True
            while cursor.peek() != ord("]"):
                if not first_item:
                    cursor.take(ord(","), f"audio FFprobe {key} wrapper")
                first_item = False
                if cursor.peek() != ord("{"):
                    raise FfprobeJsonError(f"audio FFprobe {key} wrapper is invalid")
                cursor.skip_ignored_value(depth=3)
            cursor.take(ord("]"), f"audio FFprobe {key} wrapper")
    cursor.take(ord("}"), "audio FFprobe root object")
    cursor.finish()
    if "streams" not in root_keys:
        raise FfprobeJsonError("audio FFprobe root keys are invalid")
    if stream is None:
        return None
    if set(stream) != {"codec_type", "sample_rate", "channels"}:
        raise FfprobeJsonError("audio FFprobe stream keys are invalid")
    codec_type = stream["codec_type"]
    sample_rate = stream["sample_rate"]
    channels = stream["channels"]
    if codec_type != "audio":
        raise FfprobeJsonError("audio FFprobe codec_type is invalid")
    if not isinstance(sample_rate, str) or not re.fullmatch(r"[1-9][0-9]*", sample_rate):
        raise FfprobeJsonError("audio FFprobe sample_rate is noncanonical")
    normalized_rate = int(sample_rate)
    if normalized_rate > 0xFFFFFFFF:
        raise FfprobeJsonError("audio FFprobe sample_rate exceeds u32")
    if (
        isinstance(channels, bool)
        or not isinstance(channels, int)
        or not 1 <= channels <= 0xFFFFFFFF
    ):
        raise FfprobeJsonError("audio FFprobe channels are invalid")
    return AudioProbeStream("audio", normalized_rate, channels)


@dataclass(frozen=True)
class FinalProbeStream:
    fields: dict[str, object]


@dataclass(frozen=True)
class FinalContainerExpectation:
    width: int
    height: int
    fps: str
    frame_count: int
    resolved_encoder: Literal["libx264", "libx265", "hevc_nvenc"]
    audio_expected: bool
    audio_packet_upper_bound: int

    def __post_init__(self) -> None:
        if (
            type(self.width) is not int
            or type(self.height) is not int
            or self.width <= 0
            or self.height <= 0
        ):
            raise FinalContainerValidationError("final dimensions must be positive integers")
        _canonical_slash_ratio(self.fps, "final frame rate")
        if type(self.frame_count) is not int or self.frame_count <= 0:
            raise FinalContainerValidationError("final frame count must be positive")
        if self.resolved_encoder not in {"libx264", "libx265", "hevc_nvenc"}:
            raise FinalContainerValidationError("final encoder is unsupported")
        if type(self.audio_expected) is not bool:
            raise FinalContainerValidationError("audio expectation must be boolean")
        if type(self.audio_packet_upper_bound) is not int:
            raise FinalContainerValidationError("audio packet bound must be an integer")
        if self.audio_expected:
            if self.audio_packet_upper_bound <= 0:
                raise FinalContainerValidationError("audio packet bound must be positive")
        elif self.audio_packet_upper_bound != 0:
            raise FinalContainerValidationError("no-audio validation requires a zero packet bound")


_FINAL_STREAM_KEYS = frozenset(
    {
        "codec_type",
        "codec_name",
        "width",
        "height",
        "pix_fmt",
        "sample_aspect_ratio",
        "display_aspect_ratio",
        "avg_frame_rate",
        "nb_read_frames",
    }
)


def parse_final_ffprobe_json(raw: bytes) -> tuple[FinalProbeStream, ...]:
    cursor = _BoundedFfprobeJsonCursor(
        raw,
        raw_cap=FINAL_PROBE_STDOUT_BYTES,
        max_depth=4,
    )
    cursor.enter_known_container(ord("{"), depth=1, label="final FFprobe root")
    root_keys: set[str] = set()
    result: list[FinalProbeStream] = []
    first_root = True
    while cursor.peek() != ord("}"):
        if not first_root:
            cursor.take(ord(","), "final FFprobe root object")
        first_root = False
        key = cursor.string(maximum_bytes=128, label="final FFprobe root key")
        if key not in {"streams", "programs", "stream_groups"}:
            raise FfprobeJsonError("final FFprobe root keys are invalid")
        if key in root_keys:
            raise FfprobeJsonError(f"duplicate authoritative root key: {key}")
        root_keys.add(key)
        cursor.take(ord(":"), "final FFprobe root object")
        if key != "streams":
            cursor.enter_known_container(
                ord("["), depth=2, label=f"final FFprobe {key} wrapper"
            )
            if cursor.peek() != ord("]"):
                raise FfprobeJsonError(f"final FFprobe {key} wrapper must be empty")
            cursor.take(ord("]"), f"final FFprobe {key} wrapper")
            continue
        cursor.enter_known_container(ord("["), depth=2, label="final FFprobe streams")
        first_stream = True
        while cursor.peek() != ord("]"):
            if not first_stream:
                cursor.take(ord(","), "final FFprobe streams array")
            first_stream = False
            if len(result) == 2:
                raise FfprobeJsonError("final FFprobe must contain one or two streams")
            cursor.enter_known_container(
                ord("{"), depth=3, label="final FFprobe stream object"
            )
            fields: dict[str, object] = {}
            first_field = True
            while cursor.peek() != ord("}"):
                if not first_field:
                    cursor.take(ord(","), "final FFprobe stream object")
                first_field = False
                field = cursor.string(maximum_bytes=128, label="final FFprobe stream key")
                if field not in _FINAL_STREAM_KEYS:
                    raise FfprobeJsonError(
                        "final FFprobe stream key is outside the allowlist"
                    )
                if field in fields:
                    raise FfprobeJsonError(
                        f"duplicate authoritative stream key: {field}"
                    )
                cursor.take(ord(":"), "final FFprobe stream object")
                fields[field] = cursor.scalar(
                    maximum_bytes=128,
                    label=f"final FFprobe {field}",
                )
            cursor.take(ord("}"), "final FFprobe stream object")
            result.append(FinalProbeStream(fields))
        cursor.take(ord("]"), "final FFprobe streams array")
    cursor.take(ord("}"), "final FFprobe root object")
    cursor.finish()
    if "streams" not in root_keys:
        raise FfprobeJsonError("final FFprobe root keys are invalid")
    if not 1 <= len(result) <= 2:
        raise FfprobeJsonError("final FFprobe must contain one or two streams")
    return tuple(result)


def _canonical_decimal(value: object, label: str, *, positive: bool) -> int:
    pattern = r"[1-9][0-9]*" if positive else r"(?:0|[1-9][0-9]*)"
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
        raise FinalContainerValidationError(f"{label} is not canonical decimal")
    number = int(value)
    if number > 0xFFFFFFFFFFFFFFFF:
        raise FinalContainerValidationError(f"{label} exceeds u64")
    return number


def _canonical_ratio(
    value: object,
    label: str,
    *,
    separator: Literal[":", "/"],
) -> tuple[int, int]:
    if not isinstance(value, str):
        raise FinalContainerValidationError(f"{label} is not a string ratio")
    escaped_separator = re.escape(separator)
    match = re.fullmatch(
        rf"([1-9][0-9]*){escaped_separator}([1-9][0-9]*)",
        value,
    )
    if match is None:
        raise FinalContainerValidationError(f"{label} is not a positive ratio")
    numerator, denominator = int(match.group(1)), int(match.group(2))
    if numerator > 0xFFFFFFFFFFFFFFFF or denominator > 0xFFFFFFFFFFFFFFFF:
        raise FinalContainerValidationError(f"{label} exceeds u64")
    if math.gcd(numerator, denominator) != 1:
        raise FinalContainerValidationError(f"{label} is not reduced")
    return numerator, denominator


def _canonical_colon_ratio(value: object, label: str) -> tuple[int, int]:
    return _canonical_ratio(value, label, separator=":")


def _canonical_slash_ratio(value: object, label: str) -> tuple[int, int]:
    return _canonical_ratio(value, label, separator="/")


def validate_final_probe_streams(  # noqa: C901
    streams: tuple[FinalProbeStream, ...],
    expectation: FinalContainerExpectation,
) -> None:
    """Validate the complete authoritative FFprobe scalar result."""

    video_streams = [
        stream.fields for stream in streams if stream.fields.get("codec_type") == "video"
    ]
    audio_streams = [
        stream.fields for stream in streams if stream.fields.get("codec_type") == "audio"
    ]
    if len(video_streams) != 1 or len(video_streams) + len(audio_streams) != len(streams):
        raise FinalContainerValidationError("final container stream topology is invalid")
    expected_video_keys = {
        "codec_type",
        "codec_name",
        "width",
        "height",
        "pix_fmt",
        "sample_aspect_ratio",
        "display_aspect_ratio",
        "avg_frame_rate",
        "nb_read_frames",
    }
    video = video_streams[0]
    if set(video) != expected_video_keys:
        raise FinalContainerValidationError("final video stream fields are incomplete")
    codec = {
        "libx264": "h264",
        "libx265": "hevc",
        "hevc_nvenc": "hevc",
    }[expectation.resolved_encoder]
    if video["codec_name"] != codec:
        raise FinalContainerValidationError("final video codec differs")
    if (
        type(video["width"]) is not int
        or type(video["height"]) is not int
        or video["width"] != expectation.width
        or video["height"] != expectation.height
    ):
        raise FinalContainerValidationError("final video dimensions differ")
    if video["pix_fmt"] != "yuv420p":
        raise FinalContainerValidationError("final video pixel format differs")
    if _canonical_colon_ratio(video["sample_aspect_ratio"], "sample aspect ratio") != (
        1,
        1,
    ):
        raise FinalContainerValidationError("final video sample aspect ratio differs")
    divisor = math.gcd(expectation.width, expectation.height)
    expected_display = (expectation.width // divisor, expectation.height // divisor)
    if (
        _canonical_colon_ratio(video["display_aspect_ratio"], "display aspect ratio")
        != expected_display
    ):
        raise FinalContainerValidationError("final video display aspect ratio differs")
    expected_fps = _canonical_slash_ratio(expectation.fps, "expected frame rate")
    if _canonical_slash_ratio(video["avg_frame_rate"], "average frame rate") != expected_fps:
        raise FinalContainerValidationError("final video frame rate differs")
    if (
        _canonical_decimal(video["nb_read_frames"], "video frame count", positive=True)
        != expectation.frame_count
    ):
        raise FinalContainerValidationError("final video frame count differs")
    if expectation.audio_expected:
        if len(audio_streams) != 1:
            raise FinalContainerValidationError("final AAC stream is missing")
        audio = audio_streams[0]
        if set(audio) != {"codec_type", "codec_name", "nb_read_frames"}:
            raise FinalContainerValidationError("final audio stream fields are incomplete")
        if audio["codec_name"] != "aac":
            raise FinalContainerValidationError("final audio codec differs")
        packet_count = _canonical_decimal(
            audio["nb_read_frames"], "audio packet count", positive=True
        )
        if packet_count > expectation.audio_packet_upper_bound:
            raise FinalContainerAudioPacketBoundError(
                "final audio packet bound was exceeded"
            )
    elif audio_streams:
        raise FinalContainerValidationError("final container has unexpected audio")


class FfmpegProgressParser:
    """Incremental legal-line parser with semantic stall-reset reporting."""

    def __init__(self) -> None:
        self.maximums = {key: -1 for key in ("frame", "out_time_us", "out_time_ms", "total_size")}
        self.ended = False

    def feed_line(self, raw_line: bytes) -> bool:  # noqa: C901
        if self.ended:
            raise EncodingProgressError("progress bytes follow progress=end")
        if len(raw_line) > FINAL_ENCODING_LINE_BYTES:
            raise EncodingProgressError("progress line exceeds its cap")
        if raw_line.endswith(b"\r\n"):
            line = raw_line[:-2]
        elif raw_line.endswith(b"\n"):
            line = raw_line[:-1]
        else:
            raise EncodingProgressError("progress line has no terminator")
        if b"=" not in line:
            raise EncodingProgressError("progress line has no key separator")
        key, value = line.split(b"=", 1)
        if not _PROGRESS_KEY.fullmatch(key) or any(byte < 0x20 or byte > 0x7E for byte in value):
            raise EncodingProgressError("progress line syntax is invalid")
        decoded_key = key.decode("ascii")
        if decoded_key in self.maximums:
            if decoded_key == "total_size" and value == b"N/A":
                return False
            if not _CANONICAL_U64.fullmatch(value):
                raise EncodingProgressError("progress numeric value is noncanonical")
            number = int(value)
            if number > 0xFFFFFFFFFFFFFFFF:
                raise EncodingProgressError("progress numeric value exceeds u64")
            increased = number > self.maximums[decoded_key]
            self.maximums[decoded_key] = max(self.maximums[decoded_key], number)
            return increased
        if decoded_key == "progress":
            if value == b"continue":
                return False
            if value == b"end":
                self.ended = True
                return False
            raise EncodingProgressError("progress control value is invalid")
        return False

    def finish(self) -> None:
        if not self.ended:
            raise EncodingProgressError("progress stream has no terminal progress=end")


class FinalDecodeProgressParser(FfmpegProgressParser):
    """Final-decode progress whose stall clock advances only on decoded media."""

    _VALUE_CAP = 256

    def __init__(self) -> None:
        self.maximums = {key: -1 for key in ("frame", "out_time_us", "out_time_ms")}
        self.ended = False

    def feed_line(self, raw_line: bytes) -> bool:
        if self.ended:
            raise EncodingProgressError("progress bytes follow progress=end")
        if len(raw_line) > FINAL_ENCODING_LINE_BYTES:
            raise EncodingProgressError("progress line exceeds its cap")
        if raw_line.endswith(b"\r\n"):
            line = raw_line[:-2]
        elif raw_line.endswith(b"\n"):
            line = raw_line[:-1]
        else:
            raise EncodingProgressError("progress line has no terminator")
        if b"=" not in line:
            raise EncodingProgressError("progress line has no key separator")
        key, value = line.split(b"=", 1)
        if len(value) > self._VALUE_CAP:
            raise EncodingProgressError("progress value exceeds its cap")
        if not _PROGRESS_KEY.fullmatch(key) or any(
            byte < 0x20 or byte > 0x7E for byte in value
        ):
            raise EncodingProgressError("progress line syntax is invalid")
        decoded_key = key.decode("ascii")
        if decoded_key in self.maximums:
            if not _CANONICAL_U64.fullmatch(value):
                raise EncodingProgressError("progress numeric value is noncanonical")
            number = int(value)
            if number > 0xFFFFFFFFFFFFFFFF:
                raise EncodingProgressError("progress numeric value exceeds u64")
            increased = number > self.maximums[decoded_key]
            self.maximums[decoded_key] = max(self.maximums[decoded_key], number)
            return increased
        if decoded_key == "progress":
            if value == b"continue":
                return False
            if value == b"end":
                self.ended = True
                return False
            raise EncodingProgressError("progress control value is invalid")
        return False


def encoding_control_phase_peaks() -> dict[str, int]:
    mib = 1024 * 1024
    kib = 1024
    return {
        "encoding_manifest": 6 * mib,
        "audio_probe": 4 * mib + 320 * kib,
        "audio_decode": 5 * mib + 256 * kib,
        "encoding_process": 4 * mib + 256 * kib,
        "final_probe": 4 * mib + 640 * kib,
        "final_decode": 4 * mib + 320 * kib,
        "reservation_index": 4 * mib + 512 * kib,
        "fixed_target_preflight": 5 * mib,
        "fixed_target_descriptor": 5 * mib,
        "fixed_target_consumption": 5 * mib,
        "job_control_validation": 5 * mib + 64 * kib,
        "settings_intent_commit": 6 * mib,
        "producer_settings_commit": 6 * mib,
        "pending_settings_commit": 6 * mib,
        "completion_settings_commit": 6 * mib,
    }
