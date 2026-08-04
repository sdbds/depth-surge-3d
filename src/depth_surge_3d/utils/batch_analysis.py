#!/usr/bin/env python3
"""Batch directory analysis and final VR video creation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from datetime import datetime

from ..core.constants import INTERMEDIATE_DIRS


CURRENT_FRAME_STAGES = {
    INTERMEDIATE_DIRS["vr_frames"]: "Final VR frames",
    INTERMEDIATE_DIRS["left_upscaled"]: "Upscaled left frames",
    INTERMEDIATE_DIRS["right_upscaled"]: "Upscaled right frames",
    INTERMEDIATE_DIRS["left_cropped"]: "Cropped left frames",
    INTERMEDIATE_DIRS["right_cropped"]: "Cropped right frames",
    INTERMEDIATE_DIRS["left_distorted"]: "Distorted left frames",
    INTERMEDIATE_DIRS["right_distorted"]: "Distorted right frames",
    INTERMEDIATE_DIRS["left_frames"]: "Stereo left frames",
    INTERMEDIATE_DIRS["right_frames"]: "Stereo right frames",
    INTERMEDIATE_DIRS["disparity_maps"]: "Canonical disparity maps",
    INTERMEDIATE_DIRS["frames"]: "Original frames",
}


# Lazy import cv2 to avoid blocking module loading when cv2 is not available
def _get_cv2():
    """Lazy import cv2 only when needed."""
    try:
        import cv2

        return cv2
    except ImportError:
        raise ImportError(
            "opencv-python is required for image processing. Install with: pip install opencv-python"
        )


def analyze_batch_directory(batch_path: Path) -> dict[str, Any]:
    """
    Analyze batch directory to determine available processing stages and settings.

    Args:
        batch_path: Path to the batch directory

    Returns:
        Dictionary with analysis results including stages, frame count, etc.
    """
    batch_path = Path(batch_path)

    # Detect highest processing stage and frame count
    highest_stage_num, highest_stage_name, frame_count = _detect_highest_stage(
        batch_path, CURRENT_FRAME_STAGES
    )

    # Detect VR format and resolution
    vr_format, resolution = _detect_vr_format_and_resolution(batch_path, highest_stage_num)

    # Load settings summary
    settings_summary = _load_settings_summary(batch_path)

    # Check audio availability
    has_audio = _detect_audio_availability(batch_path)

    return {
        "frame_count": frame_count,
        "vr_format": vr_format,
        "resolution": resolution,
        "highest_stage": highest_stage_name,
        "has_audio": has_audio,
        "settings_summary": settings_summary,
    }


def create_video_from_batch(batch_path: Path, settings: dict[str, Any]) -> Path | None:
    """
    Create video from batch frames using FFmpeg.

    Args:
        batch_path: Path to batch directory containing frames
        settings: Settings dictionary with frame_source, quality, etc.

    Returns:
        Path to created video file or None if failed
    """
    batch_path = Path(batch_path)
    frame_source = settings.get("frame_source", "auto")
    quality = settings.get("quality", "medium")
    fps = settings.get("fps", "original")
    output_filename = settings.get("output_filename")

    # Determine frame directory to use
    if frame_source not in {"auto", "vr_frames"}:
        raise ValueError(f"Unsupported frame source: {frame_source}")
    frame_dir = batch_path / INTERMEDIATE_DIRS["vr_frames"]

    if not frame_dir or not frame_dir.exists():
        raise ValueError(f"No frames found in selected stage: {frame_source}")

    # Generate output filename
    if not output_filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_name = batch_path.name
        quality_suffix = f"_{quality}" if quality != "medium" else ""
        output_filename = f"{batch_name}_stitched_{timestamp}{quality_suffix}.mp4"

    output_path = batch_path / output_filename

    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        "30" if fps == "original" else str(fps),
        "-i",
        str(frame_dir / "frame_%06d.png"),
    ]

    # Quality settings
    quality_settings = {
        "low": ["-crf", "28", "-preset", "fast"],
        "medium": ["-crf", "23", "-preset", "medium"],
        "high": ["-crf", "18", "-preset", "slow"],
        "lossless": ["-crf", "0", "-preset", "medium"],
    }

    cmd.extend(quality_settings.get(quality, quality_settings["medium"]))
    cmd.extend(["-pix_fmt", "yuv420p", str(output_path)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)

        return output_path
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr}")
        return None


def _get_stage_number(stage_dir: str) -> int:
    """Extract numeric stage number from directory name."""
    try:
        # Extract number from directory name (e.g., "99_vr_frames" -> 99)
        return int(stage_dir.split("_")[0])
    except (ValueError, IndexError):
        return 0


def _detect_highest_stage(batch_path: Path, stages: dict[str, str]) -> tuple[int, str, int]:
    """
    Detect the highest processing stage and frame count.

    Returns:
        Tuple of (highest_stage_num, highest_stage_name, frame_count)
    """
    highest_stage_num = 0
    highest_stage_name = "none"
    frame_count = 0

    for stage_dir, stage_name in stages.items():
        stage_path = batch_path / stage_dir
        if stage_path.exists():
            png_files = list(stage_path.glob("*.png"))
            if png_files:
                frame_count = max(frame_count, len(png_files))
                current_stage_num = _get_stage_number(stage_dir)
                if current_stage_num > highest_stage_num:
                    highest_stage_num = current_stage_num
                    highest_stage_name = stage_name

    return highest_stage_num, highest_stage_name, frame_count


def _detect_vr_format_and_resolution(batch_path: Path, highest_stage_num: int) -> tuple[str, str]:
    """
    Detect VR format and resolution from sample frames.

    Returns:
        Tuple of (vr_format, resolution)
    """
    vr_format = "unknown"
    resolution = "unknown"

    sample_frame_dirs = [
        d
        for d in [
            INTERMEDIATE_DIRS["vr_frames"],
            INTERMEDIATE_DIRS["left_upscaled"],
            INTERMEDIATE_DIRS["right_upscaled"],
            INTERMEDIATE_DIRS["left_cropped"],
            INTERMEDIATE_DIRS["right_cropped"],
            INTERMEDIATE_DIRS["left_distorted"],
            INTERMEDIATE_DIRS["right_distorted"],
            INTERMEDIATE_DIRS["left_frames"],
            INTERMEDIATE_DIRS["right_frames"],
            INTERMEDIATE_DIRS["frames"],
        ]
        if (batch_path / d).exists()
    ]

    for frame_dir in sample_frame_dirs:
        frame_path = batch_path / frame_dir
        sample_frames = list(frame_path.glob("*.png"))
        if sample_frames:
            try:
                cv2 = _get_cv2()
                sample_img = cv2.imread(str(sample_frames[0]))
                if sample_img is not None:
                    h, w = sample_img.shape[:2]
                    resolution = f"{w}x{h}"

                    # Detect format based on aspect ratio
                    if frame_dir == INTERMEDIATE_DIRS["vr_frames"]:
                        if w > h * 1.5:
                            vr_format = "side_by_side"
                        else:
                            vr_format = "over_under"
                    break
            except Exception:
                continue

    return vr_format, resolution


def _load_settings_summary(batch_path: Path) -> str:
    """Load and summarize settings from settings file."""
    settings_files = list(batch_path.glob("*-settings.json"))
    if not settings_files:
        return "unknown"

    try:
        with open(settings_files[0], "r") as f:
            settings = json.load(f)
            return _summarize_settings(settings)
    except Exception:
        return "unknown"


def _detect_audio_availability(batch_path: Path) -> bool:
    """Check if audio is available in uploads directory."""
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    for ext in video_extensions:
        if list(batch_path.parent.parent.glob(f"uploads/{ext}")):
            return True
    return False


def _summarize_settings(settings: dict[str, Any]) -> str:
    """Create a human-readable summary of settings."""
    summary_parts = []

    if "vr_format" in settings:
        summary_parts.append(f"Format: {settings['vr_format']}")

    if "vr_resolution" in settings:
        summary_parts.append(f"Resolution: {settings['vr_resolution']}")

    if "processing_mode" in settings:
        summary_parts.append(f"Mode: {settings['processing_mode']}")

    if "super_sample" in settings and settings["super_sample"] != "none":
        summary_parts.append(f"Super-sample: {settings['super_sample']}")

    if "fisheye_enabled" in settings and settings["fisheye_enabled"]:
        summary_parts.append("Fisheye: enabled")

    return ", ".join(summary_parts) if summary_parts else "Standard processing"
