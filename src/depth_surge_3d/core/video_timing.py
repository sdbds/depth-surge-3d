"""The source clock, output clock, and duration of a finite frame sequence."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import ceil
import re
from typing import Any

from ..utils.path_utils import calculate_frame_range


MOTION_INTERPOLATION_UNAVAILABLE = (
    "Motion interpolation is not implemented. Set a target FPS to duplicate or drop frames "
    "without changing playback speed."
)


def rate_text(rate: Fraction) -> str:
    return f"{rate.numerator}/{rate.denominator}"


def canonical_frame_rate(value: object) -> str | None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{1,10}/[0-9]{1,10}", value):
        return None
    numerator, denominator = map(int, value.split("/"))
    if not (0 < numerator < 2**31 and 0 < denominator < 2**31):
        return None
    return rate_text(Fraction(numerator, denominator))


def probe_frame_rate(stream: dict[str, Any]) -> str | None:
    return canonical_frame_rate(stream.get("avg_frame_rate")) or canonical_frame_rate(
        stream.get("r_frame_rate")
    )


def selected_frame_range(video: dict[str, Any], settings: dict[str, Any]) -> tuple[int, int]:
    return calculate_frame_range(
        int(video["frame_count"]),
        video.get("frame_rate") or float(video["fps"]),
        settings.get("start_time"),
        settings.get("end_time"),
    )


@dataclass(frozen=True)
class EncodingTimeline:
    source: Fraction
    output: Fraction
    clip_start_frame: int | None = None

    @classmethod
    def resolve(cls, source: object, target: object) -> EncodingTimeline:
        # Old jobs have only OpenCV's binary64 approximation of the source rate.
        rate = Fraction(str(source))
        if isinstance(source, float):
            rate = rate.limit_denominator(1_000_000)
        out = rate if target in (None, "original") else Fraction(str(target))
        if any(not 0 < r.numerator < 2**31 or r.denominator >= 2**31 for r in (rate, out)):
            raise ValueError("frame rates must be positive FFmpeg rationals")
        return cls(rate, out)

    @classmethod
    def from_artifact(cls, values: dict[str, Any]) -> EncodingTimeline:
        video, settings = values["video_properties"], values["processing_settings"]
        source = video.get("frame_rate") or settings.get("source_fps") or video["fps"]
        timeline = cls.resolve(source, settings["target_fps"])
        if "frame_rate" not in video:
            return timeline
        start, _ = selected_frame_range(video, settings)
        return cls(timeline.source, timeline.output, start)

    @property
    def video_filter(self) -> str:
        if self.source == self.output:
            return "setsar=1"
        # Round EOF up: emit each output tick whose timestamp precedes the source end.
        return f"setsar=1,fps={rate_text(self.output)}:round=up"

    def output_frames(self, input_frames: int) -> int:
        return ceil(input_frames * self.output / self.source)

    def ffmpeg_video_options(
        self, input_frames: int, *, drain_audio: bool
    ) -> tuple[str, tuple[str, ...]]:
        count = self.output_frames(input_frames)
        # -frames:v can stop the whole mux before AAC drains its final packet.
        if drain_audio:
            return f"{self.video_filter},trim=end_frame={count}", ()
        return self.video_filter, ("-frames:v", str(count))
