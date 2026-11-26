"""Fast FFprobe-based audio selection and clip-duration alignment for final encoding."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Literal

from ...utils.path_utils import ffmpeg_frame_interval, format_ffmpeg_time, parse_time_string


FINAL_AUDIO_PROBE_TIMEOUT_SECONDS = 30.0


class FinalAudioError(RuntimeError):
    """Base failure for audio selection."""


class FinalAudioProbeError(FinalAudioError):
    """Failed to probe the selected audio source."""


@dataclass(frozen=True)
class SelectedAudioIdentityV1:
    """Audio source metadata and FFmpeg arguments aligned to the encoded clip."""

    selection: Literal["disabled", "original_audio_flac", "original_video"]
    selected_source_path: Path | None
    resolved_clip_arguments: tuple[str, ...]
    disposition: Literal["disabled", "no_a0", "usable"]
    sample_rate: int | None
    channels: int | None
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
        if self.audio_end_sample is not None and self.audio_end_sample > 0:
            return (
                "-map",
                f"{input_index}:a:0",
                "-filter:a",
                f"atrim=end_sample={self.audio_end_sample},asetpts=PTS-STARTPTS",
                "-c:a",
                "aac",
            )
        return ("-map", f"{input_index}:a:0?", "-c:a", "aac", "-shortest")


def _is_regular_nonlink(path: Path) -> bool:
    try:
        meta = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(meta.st_mode) and not stat.S_ISLNK(meta.st_mode)


def _probe_first_audio_stream(  # noqa: C901
    source_path: Path,
    *,
    ffprobe: str = "ffprobe",
) -> tuple[int, int] | None:
    """Probe the first audio stream's ``(sample_rate, channels)`` via FFprobe."""

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type,sample_rate,channels",
        "-of",
        "json",
        str(source_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=FINAL_AUDIO_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as error:
        raise FinalAudioProbeError(f"audio ffprobe failed: {error}") from error
    if result.returncode != 0:
        raise FinalAudioProbeError(
            f"audio ffprobe returned {result.returncode}: {result.stderr.strip()}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise FinalAudioProbeError("audio ffprobe returned invalid JSON") from error

    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        return None
    stream = streams[0]
    if not isinstance(stream, dict) or stream.get("codec_type") != "audio":
        return None
    try:
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
    except (KeyError, TypeError, ValueError):
        return None
    if sample_rate <= 0 or channels <= 0:
        return None
    return sample_rate, channels


def _resolve_clip_arguments(
    settings: dict[str, Any],
    *,
    frame_count: int,
    source_rate: Fraction,
    clip_start_frame: int | None = None,
) -> tuple[str, ...]:
    if clip_start_frame is not None and clip_start_frame > 0:
        start_text, duration_text = ffmpeg_frame_interval(
            clip_start_frame,
            clip_start_frame + frame_count,
            source_rate,
        )
        return ("-ss", start_text, "-t", duration_text)

    start_value = settings.get("start_time")
    end_value = settings.get("end_time")
    start_seconds = parse_time_string(start_value) if start_value else None
    end_seconds = parse_time_string(end_value) if end_value else None
    clip_start = max(start_seconds or 0.0, 0.0)
    args: list[str] = []
    if clip_start > 0:
        args.extend(["-ss", format_ffmpeg_time(clip_start)])
    if end_seconds is not None and end_seconds > clip_start:
        args.extend(["-t", format_ffmpeg_time(end_seconds - clip_start)])
    return tuple(args)


def prepare_selected_audio_v1(
    *,
    output_dir: Path,
    original_video: str,
    settings: dict[str, Any],
    frame_count: int,
    fps: str,
    clip_start_frame: int | None = None,
    ffprobe: str = "ffprobe",
    **_ignored: object,
) -> SelectedAudioIdentityV1:
    """Inspect audio availability using one lightweight FFprobe call (no PCM decode)."""

    if not settings.get("preserve_audio", True):
        return SelectedAudioIdentityV1(
            selection="disabled",
            selected_source_path=None,
            resolved_clip_arguments=(),
            disposition="disabled",
            sample_rate=None,
            channels=None,
            audio_end_sample=None,
        )

    flac_candidate = Path(output_dir) / "original_audio.flac"
    if _is_regular_nonlink(flac_candidate):
        selection: Literal["original_audio_flac", "original_video"] = "original_audio_flac"
        source_path = flac_candidate
    else:
        selection = "original_video"
        source_path = Path(os.path.abspath(original_video))
        if not _is_regular_nonlink(source_path):
            return SelectedAudioIdentityV1(
                selection="original_video",
                selected_source_path=None,
                resolved_clip_arguments=(),
                disposition="no_a0",
                sample_rate=None,
                channels=None,
                audio_end_sample=None,
            )

    probed = _probe_first_audio_stream(source_path, ffprobe=ffprobe)
    if probed is None:
        return SelectedAudioIdentityV1(
            selection=selection,
            selected_source_path=source_path,
            resolved_clip_arguments=(),
            disposition="no_a0",
            sample_rate=None,
            channels=None,
            audio_end_sample=None,
        )

    sample_rate, channels = probed
    source_rate = Fraction(fps)
    clip_args = _resolve_clip_arguments(
        settings,
        frame_count=frame_count,
        source_rate=source_rate,
        clip_start_frame=clip_start_frame,
    )
    # Exact sample count matching the encoded clip duration (ceil to avoid truncating final sample)
    exact_samples = Fraction(frame_count * sample_rate, 1) / source_rate
    audio_end_sample = max(
        1, (exact_samples.numerator + exact_samples.denominator - 1) // exact_samples.denominator
    )
    return SelectedAudioIdentityV1(
        selection=selection,
        selected_source_path=source_path,
        resolved_clip_arguments=clip_args,
        disposition="usable",
        sample_rate=sample_rate,
        channels=channels,
        audio_end_sample=audio_end_sample,
    )
