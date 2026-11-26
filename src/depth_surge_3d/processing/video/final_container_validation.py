"""Lightweight FFprobe container validation for published final MP4 outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Literal


FINAL_PROBE_TIMEOUT_SECONDS = 30.0


class FinalContainerValidationError(ValueError):
    """The final MP4 container does not match expected video/audio properties."""


class FinalContainerProcessError(RuntimeError):
    """FFprobe failed or timed out while checking the final MP4."""


@dataclass(frozen=True)
class FinalContainerExpectation:
    """Expected scalar container properties checked after FFmpeg completes."""

    width: int
    height: int
    fps: str
    frame_count: int
    resolved_encoder: Literal["libx264", "libx265", "hevc_nvenc"]
    audio_expected: bool = False


_EXPECTED_CODEC = {
    "libx264": "h264",
    "libx265": "hevc",
    "hevc_nvenc": "hevc",
}


def probe_final_container_streams(  # noqa: C901
    path: Path | str,
    *,
    ffprobe: str = "ffprobe",
    timeout_seconds: float = FINAL_PROBE_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Run one lightweight metadata-only FFprobe (no frame decoding)."""

    candidate = Path(path)
    try:
        meta = candidate.lstat()
    except OSError as error:
        raise FinalContainerProcessError(
            f"final video candidate is unavailable: {candidate}"
        ) from error
    if not stat.S_ISREG(meta.st_mode) or meta.st_size <= 0:
        raise FinalContainerProcessError(
            f"final video candidate is empty or non-regular: {candidate}"
        )

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames",
        "-of",
        "json",
        str(candidate),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise FinalContainerProcessError("final FFprobe timed out") from error
    except OSError as error:
        raise FinalContainerProcessError(f"failed to execute ffprobe: {error}") from error

    if result.returncode != 0:
        raise FinalContainerProcessError(
            f"final FFprobe failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise FinalContainerValidationError("final FFprobe returned invalid JSON") from error

    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise FinalContainerValidationError("final MP4 contains no streams")
    return [stream for stream in streams if isinstance(stream, dict)]


def validate_final_container_v4(
    path: Path | str,
    expectation: FinalContainerExpectation,
    *,
    ffprobe: str = "ffprobe",
    timeout_seconds: float = FINAL_PROBE_TIMEOUT_SECONDS,
    **_ignored: object,
) -> bool:
    """Validate final MP4 streams with one fast metadata-only FFprobe call."""

    streams = probe_final_container_streams(
        path,
        ffprobe=ffprobe,
        timeout_seconds=timeout_seconds,
    )
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if len(video_streams) != 1:
        raise FinalContainerValidationError(f"expected 1 video stream, found {len(video_streams)}")
    video = video_streams[0]
    expected_codec = _EXPECTED_CODEC.get(expectation.resolved_encoder)
    actual_codec = video.get("codec_name")
    if expected_codec and actual_codec != expected_codec:
        raise FinalContainerValidationError(
            f"unexpected video codec {actual_codec!r}, expected {expected_codec!r}"
        )
    if (
        int(video.get("width") or 0) != expectation.width
        or int(video.get("height") or 0) != expectation.height
    ):
        raise FinalContainerValidationError(
            f"unexpected video dimensions {video.get('width')}x{video.get('height')}, "
            f"expected {expectation.width}x{expectation.height}"
        )
    if expectation.audio_expected and not audio_streams:
        raise FinalContainerValidationError("expected audio stream is missing from final MP4")
    return True


def verify_existing_final_video(
    path: Path | str,
    *,
    expected_width: int | None = None,
    expected_height: int | None = None,
    ffprobe: str = "ffprobe",
) -> bool:
    """Quickly verify that an already-published final MP4 is readable and non-empty."""

    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        return False
    try:
        if os.stat(candidate).st_size <= 0:
            return False
        streams = probe_final_container_streams(candidate, ffprobe=ffprobe)
    except (OSError, FinalContainerProcessError, FinalContainerValidationError):
        return False
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        return False
    if expected_width is not None and int(video.get("width") or 0) != expected_width:
        return False
    if expected_height is not None and int(video.get("height") or 0) != expected_height:
        return False
    return True
