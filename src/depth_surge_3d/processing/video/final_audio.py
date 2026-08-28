"""Invocation-local audio identity and bounded sample preflight for publication v4."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import os
from pathlib import Path
import stat
from typing import Any, Literal

from ...io.nofollow_io import AcquiredRoot, NoFollowPathError
from ...utils.path_utils import format_ffmpeg_time, parse_time_string
from .bounded_process import (
    BoundedProcessCaptureV1,
    ProcessOutputBudgetError,
    ProcessTimeoutError,
    run_streaming_byte_count_v1,
)
from .encoding_provider import (
    FINAL_AUDIO_DECODE_DIAGNOSTIC_BYTES,
    FINAL_AUDIO_PROBE_STDERR_BYTES,
    FINAL_AUDIO_PROBE_STDOUT_BYTES,
    AudioProbeStream,
    FfprobeJsonBudgetError,
    parse_audio_ffprobe_json,
)


FINAL_AUDIO_HASH_STREAM_BYTES = 1024 * 1024
FINAL_AUDIO_STREAM_BYTES = 1024 * 1024
FINAL_AUDIO_PROBE_TIMEOUT_SECONDS = 30.0
FINAL_AUDIO_DECODE_STALL_SECONDS = 120.0
_U64_MAX = 0xFFFFFFFFFFFFFFFF


class FinalAudioError(RuntimeError):
    """Base failure for final-publication audio preflight."""


class FinalAudioProbeError(FinalAudioError):
    """The selected source could not produce a valid first-audio-stream probe."""


class FinalAudioProbeTimeoutError(FinalAudioProbeError):
    """The bounded audio metadata probe exceeded its deadline."""


class FinalAudioProbeBudgetError(FinalAudioProbeError):
    """The bounded audio metadata probe exceeded a byte cap."""


class FinalAudioDecodeError(FinalAudioError):
    """The selected audio stream could not be counted exactly."""


class FinalAudioDecodeTimeoutError(FinalAudioDecodeError):
    """The streaming audio decode stopped making progress."""


class FinalAudioDecodeBudgetError(FinalAudioDecodeError):
    """The streaming audio decode exceeded its diagnostic cap."""


class FinalAudioPcmOverflowError(FinalAudioDecodeError):
    """A checked PCM or sample counter exceeded u64."""


class FinalAudioInputChangedError(FinalAudioError):
    """The selected source no longer matches its invocation-local identity."""


@dataclass(frozen=True)
class OpenFileIdentityV1:
    platform: Literal["posix", "windows"]
    first: int | str
    second: int | str


@dataclass(frozen=True)
class _HashedOpenFile:
    path: Path
    identity: OpenFileIdentityV1
    byte_count: int
    raw_sha256: str


@dataclass(frozen=True)
class SelectedAudioIdentityV1:
    selection: Literal["disabled", "original_audio_flac", "original_video"]
    selected_source_path: Path | None
    source_open_identity: OpenFileIdentityV1 | None
    source_byte_count: int | None
    source_raw_sha256: str | None
    resolved_clip_arguments: tuple[str, ...]
    disposition: Literal["disabled", "no_a0", "zero_decoded_samples", "usable"]
    sample_rate: int | None
    channels: int | None
    decoded_pcm_bytes: int | None
    audio_end_sample: int | None

    @property
    def usable(self) -> bool:
        return self.disposition == "usable"

    def input_arguments(self) -> tuple[str, ...]:
        if not self.usable:
            return ()
        assert self.selected_source_path is not None
        return (*self.resolved_clip_arguments, "-i", str(self.selected_source_path))

    def output_arguments(self, input_index: int) -> tuple[str, ...]:
        if not self.usable:
            return ()
        if type(input_index) is not int or input_index < 0:
            raise ValueError("audio input index must be a nonnegative integer")
        assert self.audio_end_sample is not None
        return (
            "-map",
            f"{input_index}:a:0",
            "-filter:a",
            f"atrim=end_sample={self.audio_end_sample},asetpts=PTS-STARTPTS",
            "-c:a",
            "aac",
        )

    def normalized_input_token(self) -> str | None:
        if not self.usable:
            return None
        assert self.source_raw_sha256 is not None and self.source_byte_count is not None
        return (
            f"@audio:sha256={self.source_raw_sha256}:" f"bytes={self.source_byte_count}:stream=a:0"
        )

    def revalidate(self, *, cancelled: Callable[[], bool] | None = None) -> None:
        if self.selection == "disabled":
            return
        assert self.selected_source_path is not None
        actual = _hash_open_regular_file(
            self.selected_source_path,
            cancelled=cancelled,
        )
        if (
            actual.path != self.selected_source_path
            or actual.identity != self.source_open_identity
            or actual.byte_count != self.source_byte_count
            or actual.raw_sha256 != self.source_raw_sha256
        ):
            raise FinalAudioInputChangedError("selected audio source changed during encoding")


def _checked_u64(value: int, label: str) -> int:
    if type(value) is not int or not 0 <= value <= _U64_MAX:
        raise FinalAudioPcmOverflowError(f"{label} exceeds u64")
    return value


def _checked_mul(left: int, right: int, label: str) -> int:
    _checked_u64(left, label)
    _checked_u64(right, label)
    if left and right > _U64_MAX // left:
        raise FinalAudioPcmOverflowError(f"{label} exceeds u64")
    return left * right


def _identity_from_stat(metadata: os.stat_result) -> OpenFileIdentityV1:
    device = int(metadata.st_dev)
    inode = int(metadata.st_ino)
    if device < 0 or inode < 0:
        raise FinalAudioInputChangedError("POSIX audio file identity is out of range")
    return OpenFileIdentityV1("posix", device, inode)


def _identity_from_open_handle(handle) -> OpenFileIdentityV1:
    if os.name != "nt":
        return _identity_from_stat(os.fstat(handle.fileno()))
    import msvcrt

    from ...io.file_identity import windows_identity_from_handle

    identity, _links, _attributes = windows_identity_from_handle(
        int(msvcrt.get_osfhandle(handle.fileno())),
        require_directory=False,
    )
    return OpenFileIdentityV1(
        "windows",
        identity.volume_serial,
        identity.file_id,
    )


def _matches_acquired_identity(identity: OpenFileIdentityV1, acquired: object) -> bool:
    if identity.platform == "posix":
        return acquired == (identity.first, identity.second)
    return (
        getattr(acquired, "volume_serial", None) == identity.first
        and getattr(acquired, "file_id", None) == identity.second
    )


def _hash_open_regular_file(
    path: Path,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> _HashedOpenFile:
    candidate = Path(os.path.abspath(os.fspath(path)))
    digest = hashlib.sha256()
    count = 0
    try:
        with AcquiredRoot(candidate.parent) as acquired:
            path_identity = acquired.regular_identity(candidate)
            with acquired.open_regular(candidate) as handle:
                opened_before = os.fstat(handle.fileno())
                identity = _identity_from_open_handle(handle)
                if not _matches_acquired_identity(identity, path_identity):
                    raise FinalAudioInputChangedError("selected audio path changed before hashing")
                while True:
                    if cancelled is not None and cancelled():
                        raise InterruptedError("final audio hashing was cancelled")
                    chunk = handle.read(FINAL_AUDIO_HASH_STREAM_BYTES)
                    if not chunk:
                        break
                    if count > _U64_MAX - len(chunk):
                        raise FinalAudioPcmOverflowError("selected audio byte count exceeds u64")
                    count += len(chunk)
                    digest.update(chunk)
                opened_after = os.fstat(handle.fileno())
                if (
                    _identity_from_open_handle(handle) != identity
                    or int(opened_after.st_size) != count
                    or opened_before.st_size != opened_after.st_size
                ):
                    raise FinalAudioInputChangedError("selected audio source changed while hashing")
            after_identity = acquired.regular_identity(candidate)
            if not _matches_acquired_identity(identity, after_identity):
                raise FinalAudioInputChangedError("selected audio path changed while hashing")
    except NoFollowPathError as error:
        raise FinalAudioInputChangedError(
            "selected audio path is not a non-link regular file"
        ) from error
    return _HashedOpenFile(candidate, identity, count, digest.hexdigest())


def _regular_nonlink(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode)


def _resolve_clip_arguments(
    settings: dict[str, Any],
    *,
    frame_count: int,
    output_rate: Fraction,
) -> tuple[str, ...]:
    start_value = settings.get("start_time")
    end_value = settings.get("end_time")
    start_seconds = parse_time_string(start_value) if start_value else None
    end_seconds = parse_time_string(end_value) if end_value else None
    source_rate_value = settings.get("source_fps")
    try:
        source_rate = Fraction(
            str(source_rate_value) if source_rate_value is not None else output_rate
        )
    except (ValueError, ZeroDivisionError) as error:
        raise FinalAudioPcmOverflowError("source frame rate must be positive") from error
    if source_rate <= 0:
        raise FinalAudioPcmOverflowError("source frame rate must be positive")
    start_fraction = Fraction(str(max(start_seconds or 0.0, 0.0)))
    start_frame = int(start_fraction * source_rate)
    clip_start = Fraction(start_frame, 1) / source_rate
    result: list[str] = []
    if clip_start > 0:
        result.extend(("-ss", format_ffmpeg_time(float(clip_start))))
    if end_seconds is not None and end_seconds > float(clip_start):
        clip_duration = Fraction(frame_count, 1) / output_rate
        result.extend(("-t", format_ffmpeg_time(float(clip_duration))))
    for argument in result:
        try:
            encoded = argument.encode("ascii")
        except UnicodeEncodeError as error:
            raise FinalAudioError("resolved clip argument is not ASCII") from error
        if any(byte < 0x20 or byte > 0x7E for byte in encoded):
            raise FinalAudioError("resolved clip argument has a control byte")
    return tuple(result)


def audio_probe_command_v1(source: Path) -> tuple[str, ...]:
    return (
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type,sample_rate,channels",
        "-of",
        "json",
        str(source),
    )


def audio_decode_command_v1(source: Path, clip_arguments: tuple[str, ...]) -> tuple[str, ...]:
    return (
        "ffmpeg",
        "-v",
        "error",
        "-xerror",
        *clip_arguments,
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-c:a",
        "pcm_s32le",
        "-f",
        "s32le",
        "pipe:1",
    )


def _probe_first_audio_stream(
    source: Path,
    clip_arguments: tuple[str, ...],
    *,
    cancelled: Callable[[], bool] | None = None,
) -> AudioProbeStream | None:
    del clip_arguments
    try:
        result = BoundedProcessCaptureV1(
            stdout_cap=FINAL_AUDIO_PROBE_STDOUT_BYTES,
            stderr_cap=FINAL_AUDIO_PROBE_STDERR_BYTES,
            timeout_seconds=FINAL_AUDIO_PROBE_TIMEOUT_SECONDS,
            cancelled=cancelled,
        ).run(audio_probe_command_v1(source))
    except ProcessTimeoutError as error:
        raise FinalAudioProbeTimeoutError("audio probe exceeded 30 seconds") from error
    except ProcessOutputBudgetError as error:
        raise FinalAudioProbeBudgetError("audio probe exceeded its byte budget") from error
    if result.returncode != 0 or result.stderr:
        raise FinalAudioProbeError("audio probe produced an error diagnostic")
    try:
        return parse_audio_ffprobe_json(result.stdout)
    except FfprobeJsonBudgetError as error:
        raise FinalAudioProbeBudgetError(
            "audio probe JSON exceeded its parser-state budget"
        ) from error
    except Exception as error:
        raise FinalAudioProbeError("audio probe JSON is invalid") from error


def _decode_pcm_byte_count(
    source: Path,
    clip_arguments: tuple[str, ...],
    *,
    cancelled: Callable[[], bool] | None = None,
) -> int:
    try:
        result = run_streaming_byte_count_v1(
            audio_decode_command_v1(source, clip_arguments),
            diagnostic_cap=FINAL_AUDIO_DECODE_DIAGNOSTIC_BYTES,
            stall_seconds=FINAL_AUDIO_DECODE_STALL_SECONDS,
            stream_buffer_bytes=FINAL_AUDIO_STREAM_BYTES,
            cancelled=cancelled,
        )
    except ProcessTimeoutError as error:
        raise FinalAudioDecodeTimeoutError("audio decode stopped making progress") from error
    except ProcessOutputBudgetError as error:
        raise FinalAudioDecodeBudgetError("audio decode exceeded its diagnostic budget") from error
    if result.returncode != 0 or result.stderr:
        raise FinalAudioDecodeError("audio decode produced an error diagnostic")
    return result.byte_count


def prepare_selected_audio_v1(
    *,
    output_dir: Path,
    original_video: str | Path,
    settings: dict[str, Any],
    frame_count: int,
    fps: str,
    cancelled: Callable[[], bool] | None = None,
) -> SelectedAudioIdentityV1:
    """Freeze the one selected audio source and its exact usable sample boundary."""

    if not settings.get("preserve_audio", True):
        return SelectedAudioIdentityV1(
            "disabled", None, None, None, None, (), "disabled", None, None, None, None
        )
    if type(frame_count) is not int or frame_count <= 0:
        raise FinalAudioPcmOverflowError("video frame count must be positive")
    try:
        rate = Fraction(fps)
    except (ValueError, ZeroDivisionError) as error:
        raise FinalAudioPcmOverflowError("video frame rate must be positive") from error
    if rate <= 0:
        raise FinalAudioPcmOverflowError("video frame rate must be positive")
    flac = Path(output_dir) / "original_audio.flac"
    if _regular_nonlink(flac):
        selection: Literal["original_audio_flac", "original_video"] = "original_audio_flac"
        selected_path = flac
    else:
        selection = "original_video"
        selected_path = Path(original_video)
    hashed = _hash_open_regular_file(selected_path, cancelled=cancelled)
    clip_arguments = _resolve_clip_arguments(
        settings,
        frame_count=frame_count,
        output_rate=rate,
    )
    if cancelled is None:
        stream = _probe_first_audio_stream(hashed.path, clip_arguments)
    else:
        stream = _probe_first_audio_stream(
            hashed.path,
            clip_arguments,
            cancelled=cancelled,
        )
    if stream is None:
        return SelectedAudioIdentityV1(
            selection,
            hashed.path,
            hashed.identity,
            hashed.byte_count,
            hashed.raw_sha256,
            clip_arguments,
            "no_a0",
            None,
            None,
            None,
            None,
        )
    pcm_frame_bytes = _checked_mul(4, stream.channels, "PCM frame byte count")
    if cancelled is None:
        decoded_count = _decode_pcm_byte_count(hashed.path, clip_arguments)
    else:
        decoded_count = _decode_pcm_byte_count(
            hashed.path,
            clip_arguments,
            cancelled=cancelled,
        )
    decoded = _checked_u64(decoded_count, "decoded PCM byte count")
    if decoded % pcm_frame_bytes:
        raise FinalAudioDecodeError("decoded byte count does not end on a PCM frame")
    if decoded == 0:
        return SelectedAudioIdentityV1(
            selection,
            hashed.path,
            hashed.identity,
            hashed.byte_count,
            hashed.raw_sha256,
            clip_arguments,
            "zero_decoded_samples",
            stream.sample_rate,
            stream.channels,
            0,
            None,
        )
    numerator = _checked_mul(frame_count, stream.sample_rate, "audio sample numerator")
    numerator = _checked_mul(numerator, rate.denominator, "audio sample numerator")
    if numerator > _U64_MAX - (rate.numerator - 1):
        raise FinalAudioPcmOverflowError("audio end-sample numerator exceeds u64")
    audio_end_sample = (numerator + rate.numerator - 1) // rate.numerator
    _checked_u64(audio_end_sample, "audio end sample")
    if audio_end_sample <= 0:
        raise FinalAudioPcmOverflowError("audio end sample must be positive")
    return SelectedAudioIdentityV1(
        selection,
        hashed.path,
        hashed.identity,
        hashed.byte_count,
        hashed.raw_sha256,
        clip_arguments,
        "usable",
        stream.sample_rate,
        stream.channels,
        decoded,
        audio_end_sample,
    )
