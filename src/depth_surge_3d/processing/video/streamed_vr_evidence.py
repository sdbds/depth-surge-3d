"""Closed recovery evidence for streamed assembled-VR encoding."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import PurePosixPath
import re
from typing import Any, Literal, cast

from ...io.file_identity import PersistedFileIdentityV1


_HEX32 = re.compile(r"^[0-9a-f]{32}$", flags=re.ASCII)
_HEX64 = re.compile(r"^[0-9a-f]{64}$", flags=re.ASCII)
_SELF_IMAGE_TOKEN = "@image-sequence:sha256=self:view=frames"
_ARGUMENT_COUNT_CAP = 256
_ARGUMENT_BYTES_CAP = 64 * 1024
_ARGUMENT_SCALAR_BYTES_CAP = 4096


def _closed_dict(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} schema is not closed")
    return cast(dict[str, Any], value)


def _hex(value: object, pattern: re.Pattern[str], label: str) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")
    return value


def _argument_value(arguments: tuple[str, ...], option: str) -> str:
    positions = [index for index, value in enumerate(arguments[:-1]) if value == option]
    if len(positions) != 1:
        raise ValueError(f"streamed invocation requires exactly one {option} option")
    return arguments[positions[0] + 1]


@dataclass(frozen=True, slots=True)
class StreamedVrInvocationV1:
    """Exact non-recursive FFmpeg invocation template bound before launch."""

    job_id: str
    settings_identity_fingerprint: str
    publication_generation: str
    frame_count: int
    first_index: int
    expected_shape: tuple[int, int]
    resolved_encoder: Literal["libx264", "libx265", "hevc_nvenc"]
    resolved_fps: str
    audio_end_sample: int | None
    resolved_arguments_template: tuple[str, ...]

    def __post_init__(self) -> None:
        _hex(self.job_id, _HEX32, "streamed invocation job id")
        _hex(
            self.settings_identity_fingerprint,
            _HEX64,
            "streamed invocation settings identity",
        )
        _hex(
            self.publication_generation,
            _HEX32,
            "streamed invocation publication generation",
        )
        if type(self.frame_count) is not int or not 1 <= self.frame_count <= 4_194_304:
            raise ValueError("streamed invocation frame count is invalid")
        if type(self.first_index) is not int or not 0 <= self.first_index <= 2_147_483_646:
            raise ValueError("streamed invocation first index is invalid")
        if self.first_index + self.frame_count - 1 > 2_147_483_646:
            raise ValueError("streamed invocation final index is invalid")
        if (
            not isinstance(self.expected_shape, tuple)
            or len(self.expected_shape) != 2
            or any(type(value) is not int or value <= 0 for value in self.expected_shape)
        ):
            raise ValueError("streamed invocation output shape is invalid")
        if self.resolved_encoder not in {"libx264", "libx265", "hevc_nvenc"}:
            raise ValueError("streamed invocation encoder is invalid")
        try:
            rate = Fraction(self.resolved_fps)
        except (TypeError, ValueError, ZeroDivisionError) as error:
            raise ValueError("streamed invocation frame rate is invalid") from error
        if rate <= 0 or self.resolved_fps != f"{rate.numerator}/{rate.denominator}":
            raise ValueError("streamed invocation frame rate is not canonical")
        if self.audio_end_sample is not None and (
            type(self.audio_end_sample) is not int or self.audio_end_sample <= 0
        ):
            raise ValueError("streamed invocation audio boundary is invalid")

        arguments = self.resolved_arguments_template
        if (
            not isinstance(arguments, tuple)
            or not 1 <= len(arguments) <= _ARGUMENT_COUNT_CAP
            or any(type(value) is not str for value in arguments)
        ):
            raise ValueError("streamed invocation argument vector is invalid")
        encoded_bytes = 0
        for argument in arguments:
            try:
                raw = argument.encode("ascii")
            except UnicodeEncodeError as error:
                raise ValueError("streamed invocation argument is not ASCII") from error
            if not raw or len(raw) > _ARGUMENT_SCALAR_BYTES_CAP or b"\x00" in raw:
                raise ValueError("streamed invocation argument scalar is invalid")
            encoded_bytes += len(raw)
        if encoded_bytes > _ARGUMENT_BYTES_CAP:
            raise ValueError("streamed invocation arguments exceed their byte budget")
        if arguments.count(_SELF_IMAGE_TOKEN) != 1:
            raise ValueError("streamed invocation lacks its unique self image token")
        if not arguments[-1].startswith("@output:path64="):
            raise ValueError("streamed invocation lacks its normalized output token")
        if _argument_value(arguments, "-framerate") != self.resolved_fps:
            raise ValueError("streamed invocation frame rate differs from its arguments")
        if _argument_value(arguments, "-c:v") != self.resolved_encoder:
            raise ValueError("streamed invocation encoder differs from its arguments")
        audio_filters = [value for value in arguments if value.startswith("atrim=end_sample=")]
        if self.audio_end_sample is None:
            if audio_filters:
                raise ValueError("streamed invocation has an unexpected audio boundary")
        elif audio_filters != [f"atrim=end_sample={self.audio_end_sample},asetpts=PTS-STARTPTS"]:
            raise ValueError("streamed invocation audio boundary differs from its arguments")

    def to_object(self) -> dict[str, Any]:
        return {
            "audio_end_sample": self.audio_end_sample,
            "expected_shape": list(self.expected_shape),
            "first_index": self.first_index,
            "frame_count": self.frame_count,
            "job_id": self.job_id,
            "publication_generation": self.publication_generation,
            "resolved_arguments_template": list(self.resolved_arguments_template),
            "resolved_encoder": self.resolved_encoder,
            "resolved_fps": self.resolved_fps,
            "settings_identity_fingerprint": self.settings_identity_fingerprint,
        }

    @classmethod
    def from_object(cls, value: object) -> "StreamedVrInvocationV1":
        item = _closed_dict(
            value,
            {
                "audio_end_sample",
                "expected_shape",
                "first_index",
                "frame_count",
                "job_id",
                "publication_generation",
                "resolved_arguments_template",
                "resolved_encoder",
                "resolved_fps",
                "settings_identity_fingerprint",
            },
            "streamed invocation",
        )
        shape = item["expected_shape"]
        arguments = item["resolved_arguments_template"]
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(type(member) is not int for member in shape)
            or not isinstance(arguments, list)
            or any(type(member) is not str for member in arguments)
        ):
            raise ValueError("streamed invocation arrays are invalid")
        return cls(
            job_id=item["job_id"],
            settings_identity_fingerprint=item["settings_identity_fingerprint"],
            publication_generation=item["publication_generation"],
            frame_count=item["frame_count"],
            first_index=item["first_index"],
            expected_shape=(shape[0], shape[1]),
            resolved_encoder=item["resolved_encoder"],
            resolved_fps=item["resolved_fps"],
            audio_end_sample=item["audio_end_sample"],
            resolved_arguments_template=tuple(arguments),
        )

    def materialized_arguments(self, manifest_fingerprint: str) -> tuple[str, ...]:
        fingerprint = _hex(
            manifest_fingerprint,
            _HEX64,
            "streamed input manifest fingerprint",
        )
        token = f"@image-sequence:sha256={fingerprint}:view=frames"
        return tuple(
            token if argument == _SELF_IMAGE_TOKEN else argument
            for argument in self.resolved_arguments_template
        )


@dataclass(frozen=True, slots=True)
class StreamedVrOutputEvidenceV1:
    """Frozen bytes and inode of the completed FFmpeg work output."""

    reservation_index_fingerprint: str
    sibling_relative_path: str
    identity: PersistedFileIdentityV1
    raw_sha256: str
    byte_count: int

    def __post_init__(self) -> None:
        _hex(
            self.reservation_index_fingerprint,
            _HEX64,
            "streamed output reservation fingerprint",
        )
        relative = PurePosixPath(self.sibling_relative_path)
        if (
            relative.is_absolute()
            or len(relative.parts) != 1
            or relative.parts[0] in {"", ".", ".."}
            or "\\" in self.sibling_relative_path
        ):
            raise ValueError("streamed output sibling path is invalid")
        if not isinstance(self.identity, PersistedFileIdentityV1):
            raise ValueError("streamed output identity is invalid")
        if self.identity.link_count != 1:
            raise ValueError("streamed output identity is not single-link")
        _hex(self.raw_sha256, _HEX64, "streamed output hash")
        if type(self.byte_count) is not int or self.byte_count <= 0:
            raise ValueError("streamed output byte count is invalid")

    def to_object(self) -> dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "identity": self.identity.to_object(),
            "raw_sha256": self.raw_sha256,
            "reservation_index_fingerprint": self.reservation_index_fingerprint,
            "sibling_relative_path": self.sibling_relative_path,
        }

    @classmethod
    def from_object(cls, value: object) -> "StreamedVrOutputEvidenceV1":
        item = _closed_dict(
            value,
            {
                "byte_count",
                "identity",
                "raw_sha256",
                "reservation_index_fingerprint",
                "sibling_relative_path",
            },
            "streamed output evidence",
        )
        return cls(
            reservation_index_fingerprint=item["reservation_index_fingerprint"],
            sibling_relative_path=item["sibling_relative_path"],
            identity=PersistedFileIdentityV1.from_object(item["identity"]),
            raw_sha256=item["raw_sha256"],
            byte_count=item["byte_count"],
        )
