"""
I/O operations with side effects for video processing.

This module contains functions that perform I/O operations and have side effects:
- Filesystem I/O (reading, writing, creating, deleting)
- Subprocess execution
- External state queries

All functions here are expected to interact with external systems.
For pure functions without side effects, see utils/path_utils.py.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import time
from collections.abc import Collection
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast
import cv2

from ..core.constants import (
    SUPPORTED_VIDEO_FORMATS,
    SUPPORTED_IMAGE_FORMATS,
    INTERMEDIATE_DIRS,
)
from ..core.settings import PROCESSING_SETTINGS_SCHEMA_VERSION
from ..core.file_identity import (
    FILE_IDENTITY_ALGORITHM_VERSION,
    file_sample_fingerprint,
)
from ..utils.path_utils import (
    generate_output_filename,
    format_time_duration,
)
from .canonical_json import canonical_json_bytes
from .reservation_protocol import (
    SETTINGS_ARTIFACT_RELATIVE_PATH,
    SettingsArtifactTransactionV1,
    allocation_unit_for_path,
    bootstrap_settings_job_v1,
    commit_terminal_settings_v1,
    platform_allocation_adapter,
    platform_allocation_counter,
    recover_settings_job_v1,
)
from .settings_artifact import (
    ProcessingSettingsArtifactV5,
    create_processing_settings_artifact_v5,
)


_SAR_PATTERN = re.compile(r"([0-9]+):([0-9]+)", flags=re.ASCII)
_SAR_COMPONENT_MAX = 2_147_483_647


def parse_sample_aspect_ratio(value: object) -> tuple[int, int]:
    if value is None or value == "N/A":
        return 1, 1
    if not isinstance(value, str):
        raise ValueError("sample_aspect_ratio must be an unsigned numerator:denominator")
    match = _SAR_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError("sample_aspect_ratio must be an unsigned numerator:denominator")
    numerator, denominator = (int(part) for part in match.groups())
    if not 1 <= numerator <= _SAR_COMPONENT_MAX or not 1 <= denominator <= _SAR_COMPONENT_MAX:
        raise ValueError("sample_aspect_ratio components must be in 1..2147483647")
    divisor = math.gcd(numerator, denominator)
    return numerator // divisor, denominator // divisor


def validate_video_file(video_path: str) -> bool:
    """
    Validate if file exists and is a supported video format.

    Args:
        video_path: Path to video file

    Returns:
        True if valid video file exists

    Side Effects:
        Reads filesystem to check file existence
    """
    if not os.path.exists(video_path):
        return False

    file_ext = Path(video_path).suffix.lower()
    return file_ext in SUPPORTED_VIDEO_FORMATS


def validate_image_file(image_path: str) -> bool:
    """
    Validate if file exists and is a supported image format.

    Args:
        image_path: Path to image file

    Returns:
        True if valid image file exists

    Side Effects:
        Reads filesystem to check file existence
    """
    if not os.path.exists(image_path):
        return False

    file_ext = Path(image_path).suffix.lower()
    return file_ext in SUPPORTED_IMAGE_FORMATS


def get_video_properties(video_path: str) -> dict[str, Any]:
    """
    Get video properties using OpenCV.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video properties (width, height, fps, frame_count, duration, codec)

    Side Effects:
        Opens and reads video file with cv2.VideoCapture
    """
    properties: dict[str, Any] = {}
    cap = None

    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return properties

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Guard against zero FPS to prevent division by zero
        if fps > 0:
            duration = frame_count / fps
        else:
            duration = 0.0

        properties.update(
            {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
            }
        )

    except Exception:
        # Return empty dict if unable to read properties
        pass
    finally:
        if cap is not None:
            cap.release()

    if properties:
        video_info = get_video_info_ffprobe(video_path)
        video_stream = next(
            (
                stream
                for stream in video_info.get("streams", [])
                if isinstance(stream, dict) and stream.get("codec_type") == "video"
            ),
            None,
        )
        sar_value = video_stream.get("sample_aspect_ratio") if video_stream is not None else None
        numerator, denominator = parse_sample_aspect_ratio(sar_value)
        properties.update(
            {
                "sample_aspect_ratio_numerator": numerator,
                "sample_aspect_ratio_denominator": denominator,
                "sample_aspect_ratio": f"{numerator}:{denominator}",
            }
        )

    return properties


def get_video_info_ffprobe(video_path: str) -> dict[str, Any]:
    """
    Get detailed video information using ffprobe subprocess.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video information from ffprobe

    Side Effects:
        Executes ffprobe subprocess
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            return json.loads(result.stdout)

    except Exception:
        pass

    return {}


def create_output_directories(
    base_path: Path,
    keep_intermediates: bool = True,
    omitted_intermediates: Collection[str] | None = None,
) -> dict[str, Path]:
    """
    Create output directory structure on filesystem.

    Args:
        base_path: Base output directory path
        keep_intermediates: Retention preference used after successful processing. All
            stage directories are created because they are required pipeline storage.
        omitted_intermediates: Directory keys to omit from creation and the returned map.

    Returns:
        Dictionary mapping directory names to paths

    Side Effects:
        Creates directories on filesystem
    """
    omitted = frozenset(omitted_intermediates or ())
    directories = {"base": base_path}

    # Always create base directory
    base_path.mkdir(parents=True, exist_ok=True)

    # Intermediates are working state, not optional output. The retention flag is
    # applied only after the final video has been encoded successfully.
    for dir_name, dir_path in INTERMEDIATE_DIRS.items():
        if dir_name in omitted:
            continue
        full_path = base_path / dir_path
        full_path.mkdir(exist_ok=True)
        directories[dir_name] = full_path

    return directories


def get_frame_files(frames_dir: Path) -> list[Path]:
    """
    Get sorted list of frame files from directory.

    Args:
        frames_dir: Directory containing frame files

    Returns:
        Sorted list of frame file paths

    Side Effects:
        Reads directory contents from filesystem
    """
    if not frames_dir.exists():
        return []

    frame_files: list[Path] = []
    for ext in SUPPORTED_IMAGE_FORMATS:
        frame_files.extend(frames_dir.glob(f"*{ext}"))

    # Sort numerically by filename
    def sort_key(path: Path) -> int:
        try:
            # Extract number from filename
            stem = path.stem
            if stem.startswith("frame_"):
                return int(stem.split("_")[1])
            else:
                return int("".join(filter(str.isdigit, stem)))
        except (ValueError, IndexError):
            return 0

    return sorted(frame_files, key=sort_key)


def calculate_directory_size(directory: Path) -> int:
    """
    Calculate total size of directory in bytes.

    Args:
        directory: Directory path

    Returns:
        Total size in bytes

    Side Effects:
        Walks directory tree and stats files
    """
    total_size = 0

    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except (OSError, PermissionError):
        pass

    return total_size


def _should_keep_file(file_path: Path, keep_patterns: list[str]) -> bool:
    """
    Check if file should be kept based on pattern matching.

    Args:
        file_path: Path to file to check
        keep_patterns: List of glob patterns to keep

    Returns:
        True if file should be kept

    Side Effects:
        None (pure pattern matching)
    """
    for pattern in keep_patterns:
        if file_path.match(pattern):
            return True
    return False


def _remove_file_safe(file_path: Path) -> bool:
    """
    Safely remove a file, handling errors.

    Args:
        file_path: Path to file to remove

    Returns:
        True if file was removed successfully

    Side Effects:
        Deletes file from filesystem
    """
    try:
        file_path.unlink()
        return True
    except OSError:
        return False


def _cleanup_directory(directory: Path, keep_patterns: list[str]) -> int:
    """
    Clean up files in a single directory.

    Args:
        directory: Directory to clean
        keep_patterns: Patterns to keep

    Returns:
        Number of files removed

    Side Effects:
        Deletes files from filesystem
    """
    removed_count = 0

    for file_path in directory.rglob("*"):
        if file_path.is_file():
            if not _should_keep_file(file_path, keep_patterns):
                if _remove_file_safe(file_path):
                    removed_count += 1

    return removed_count


def cleanup_intermediate_files(base_path: Path, keep_patterns: list[str] | None = None) -> int:
    """
    Clean up intermediate files, optionally keeping certain patterns.

    Args:
        base_path: Base directory containing intermediate files
        keep_patterns: List of glob patterns to keep

    Returns:
        Number of files removed

    Side Effects:
        Deletes files from filesystem
    """
    removed_count = 0
    keep_patterns = keep_patterns or []

    try:
        for intermediate_dir in INTERMEDIATE_DIRS.values():
            full_dir = base_path / intermediate_dir
            if full_dir.exists():
                removed_count += _cleanup_directory(full_dir, keep_patterns)

    except (OSError, PermissionError):
        pass

    return removed_count


def verify_ffmpeg_installation() -> bool:
    """
    Verify that FFmpeg is installed and accessible.

    Returns:
        True if FFmpeg is available

    Side Effects:
        Executes ffmpeg subprocess
    """
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return False


def get_available_space(directory: Path) -> int:
    """
    Get available disk space for directory in bytes.

    Args:
        directory: Directory path to check

    Returns:
        Available space in bytes

    Side Effects:
        Queries filesystem for disk space information
    """
    statvfs = getattr(os, "statvfs", None)
    if statvfs is not None:
        try:
            stat = statvfs(directory)
            return int(stat.f_bavail * stat.f_frsize)
        except (OSError, AttributeError):
            pass
    try:
        _, _, free = shutil.disk_usage(directory)
        return int(free)
    except OSError:
        return 0


def save_processing_settings(
    output_dir: Path,
    batch_name: str,
    settings: dict[str, Any],
    video_properties: dict[str, Any],
    source_video_path: str,
    *,
    transactional_v5: bool = False,
) -> Path:
    """
    Save processing settings to a JSON file in the output directory.

    Args:
        output_dir: Output directory path
        batch_name: Name of the processing batch/job
        settings: Complete processing settings dictionary
        video_properties: Video metadata
        source_video_path: Path to source video file

    Returns:
        Path to the saved settings file

    Side Effects:
        Writes JSON file to disk
    """
    if transactional_v5:
        return _save_processing_settings_v5(
            output_dir=output_dir,
            batch_name=batch_name,
            settings=settings,
            video_properties=video_properties,
            source_video_path=source_video_path,
        )

    source_path = Path(source_video_path)
    source_video_fingerprint = (
        file_sample_fingerprint(source_path) if source_path.is_file() else None
    )
    settings_data = {
        "metadata": {
            "batch_name": batch_name,
            "source_video": str(source_video_path),
            "source_video_name": Path(source_video_path).name,
            "source_video_fingerprint_algorithm": FILE_IDENTITY_ALGORITHM_VERSION,
            "source_video_fingerprint": source_video_fingerprint,
            "settings_schema_version": PROCESSING_SETTINGS_SCHEMA_VERSION,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "created_timestamp": time.time(),
            "project_version": "0.9.0",
            "processing_status": "in_progress",
        },
        "video_properties": video_properties,
        "processing_settings": settings,
        "output_info": {
            "output_directory": str(output_dir),
            "expected_output_filename": generate_output_filename(
                Path(source_video_path).name,
                settings["vr_format"],
                settings["vr_resolution"],
            ),
        },
    }

    # Save settings file
    settings_file = output_dir / f"{batch_name}-settings.json"

    try:
        with open(settings_file, "w", encoding="utf-8") as f:
            json.dump(settings_data, f, indent=2, ensure_ascii=False)

        print(f"Settings saved to: {settings_file}")
        return settings_file

    except Exception as e:
        print(f"Warning: Could not save settings file: {e}")
        # Create a minimal fallback file
        fallback_data = {
            "metadata": {
                "batch_name": batch_name,
                "error": f"Failed to save full settings: {e}",
            },
            "processing_settings": settings,
        }
        try:
            with open(settings_file, "w", encoding="utf-8") as f:
                json.dump(fallback_data, f, indent=2)
        except Exception:
            pass  # If we can't even save the fallback, continue without it

        return settings_file


def _utc_now_usec() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _save_processing_settings_v5(
    *,
    output_dir: Path,
    batch_name: str,
    settings: dict[str, Any],
    video_properties: dict[str, Any],
    source_video_path: str,
) -> Path:
    """Create a new schema-5 job through locator-first reserved publication."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(source_video_path)
    source_fingerprint = file_sample_fingerprint(source_path) if source_path.is_file() else None
    artifact = create_processing_settings_artifact_v5(
        batch_name=batch_name,
        source_video=str(source_video_path),
        source_video_name=source_path.name,
        source_video_fingerprint=source_fingerprint,
        video_properties=video_properties,
        processing_settings=settings,
        output_directory=str(output_dir),
        project_version="0.9.0",
        creation_now_usec=_utc_now_usec(),
    )
    allocation_unit = allocation_unit_for_path(output_dir)
    bootstrap_settings_job_v1(
        job_root=output_dir,
        artifact=artifact,
        initial_settings_reservation_id=secrets.token_hex(16),
        allocation_adapter=platform_allocation_adapter,
        allocation_counter=platform_allocation_counter,
        allocation_unit=allocation_unit,
    )
    settings_file = output_dir / SETTINGS_ARTIFACT_RELATIVE_PATH
    print(f"Settings saved to: {settings_file}")
    return settings_file


def load_processing_settings(settings_file: Path) -> dict[str, Any] | None:
    """
    Load processing settings from a JSON file.

    Args:
        settings_file: Path to settings file

    Returns:
        Dictionary with loaded settings data or None if failed

    Side Effects:
        Reads JSON file from disk
    """
    try:
        if not settings_file.exists():
            print(f"Settings file not found: {settings_file}")
            return None

        if settings_file.name == SETTINGS_ARTIFACT_RELATIVE_PATH.name:
            settings_data = ProcessingSettingsArtifactV5.parse(
                settings_file.read_bytes()
            ).to_object()
        else:
            with open(settings_file, "r", encoding="utf-8") as f:
                settings_data = json.load(f)

        print(f"Settings loaded from: {settings_file}")
        return settings_data

    except Exception as e:
        print(f"Error loading settings file: {e}")
        return None


def update_processing_status(
    settings_file: Path, status: str, additional_info: dict[str, Any] | None = None
) -> bool:
    """
    Update the processing status in the settings file.

    Args:
        settings_file: Path to settings file
        status: New status ('in_progress', 'completed', 'failed', 'paused')
        additional_info: Additional metadata to add

    Returns:
        True if update was successful

    Side Effects:
        Reads and writes JSON file
    """
    if settings_file.name == SETTINGS_ARTIFACT_RELATIVE_PATH.name:
        return _update_processing_status_v5(
            settings_file,
            status=status,
            additional_info=additional_info,
        )
    try:
        settings_data = load_processing_settings(settings_file)
        if not settings_data:
            return False

        # Update status and timestamp
        settings_data["metadata"]["processing_status"] = status
        settings_data["metadata"]["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        settings_data["metadata"]["last_updated_timestamp"] = time.time()

        if status == "completed":
            settings_data["metadata"]["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            settings_data["metadata"]["completed_timestamp"] = time.time()

            # Calculate processing duration
            if "created_timestamp" in settings_data["metadata"]:
                duration = time.time() - settings_data["metadata"]["created_timestamp"]
                settings_data["metadata"]["processing_duration_seconds"] = duration
                settings_data["metadata"]["processing_duration_formatted"] = format_time_duration(
                    duration
                )

        # Add any additional info
        if additional_info:
            if "runtime_info" not in settings_data:
                settings_data["runtime_info"] = {}
            settings_data["runtime_info"].update(additional_info)

        # Save updated settings
        with open(settings_file, "w", encoding="utf-8") as f:
            json.dump(settings_data, f, indent=2, ensure_ascii=False)

        return True

    except Exception as e:
        print(f"Warning: Could not update settings file status: {e}")
        return False


def _bounded_metric_summary(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("metric_clamp_summary must be an object")
    if set(value) == {
        "schema_version",
        "affected_frame_count",
        "mean_clamped_fraction",
        "max_clamped_fraction",
        "compatibility_summary_sha256",
    }:
        return dict(value)
    return {
        "schema_version": 1,
        "affected_frame_count": int(value["affected_frame_count"]),
        "mean_clamped_fraction": float(value["mean_clamped_fraction"]),
        "max_clamped_fraction": float(value["max_clamped_fraction"]),
        "compatibility_summary_sha256": hashlib.sha256(canonical_json_bytes(value)).hexdigest(),
    }


def commit_final_media_producer_marker(
    settings_file: Path,
    *,
    generation: str,
) -> ProcessingSettingsArtifactV5:
    """Durably enter publication-v4 before the first FFmpeg launch."""

    settings_file = Path(settings_file)
    if settings_file.name != SETTINGS_ARTIFACT_RELATIVE_PATH.name:
        raise ValueError("producer marker requires processing-settings-v5.json")
    job_root = settings_file.parent
    current = recover_settings_job_v1(
        job_root,
        allocation_adapter=platform_allocation_adapter,
        allocation_counter=platform_allocation_counter,
    )
    target = current.with_producer_marker(
        generation=generation,
        updated_at=_utc_now_usec(),
    )
    if target is current:
        return current
    transaction = SettingsArtifactTransactionV1.begin(
        job_root=job_root,
        current=current,
        target=target,
        allocation_adapter=platform_allocation_adapter,
        allocation_counter=platform_allocation_counter,
        allocation_unit=allocation_unit_for_path(job_root),
        reservation_generation=secrets.token_hex(16),
        reservation_id=secrets.token_hex(16),
    )
    return transaction.commit()


def recover_processing_settings_artifact(
    settings_file: Path,
) -> ProcessingSettingsArtifactV5:
    settings_file = Path(settings_file)
    if settings_file.name != SETTINGS_ARTIFACT_RELATIVE_PATH.name:
        raise ValueError("schema-5 recovery requires processing-settings-v5.json")
    return recover_settings_job_v1(
        settings_file.parent,
        allocation_adapter=platform_allocation_adapter,
        allocation_counter=platform_allocation_counter,
    )


def _update_processing_status_v5(
    settings_file: Path,
    *,
    status: str,
    additional_info: dict[str, Any] | None,
) -> bool:
    """Commit a closed terminal transition through the retained lifecycle extent."""

    try:
        job_root = settings_file.parent
        current = recover_settings_job_v1(
            job_root,
            allocation_adapter=platform_allocation_adapter,
            allocation_counter=platform_allocation_counter,
        )
        now = _utc_now_usec()
        info = dict(additional_info or {})
        if status == "completed":
            settings = current.to_object()["processing_settings"]
            cleanup_status = info.get("cleanup_status")
            if cleanup_status is None:
                cleanup_status = "not_requested" if settings["keep_intermediates"] else "complete"
            if cleanup_status not in {
                "not_requested",
                "complete",
                "incomplete_identity_mismatch",
                "incomplete_error",
            }:
                raise ValueError(f"invalid cleanup_status: {cleanup_status!r}")
            completed_cleanup_status = cast(
                Literal[
                    "not_requested",
                    "complete",
                    "incomplete_identity_mismatch",
                    "incomplete_error",
                ],
                cleanup_status,
            )
            target = current.completed(
                terminal_at=now,
                frames_processed=info.get(
                    "frames_processed",
                    current.to_object()["video_properties"]["frame_count"],
                ),
                cleanup_status=completed_cleanup_status,
                metric_clamp_summary=_bounded_metric_summary(info.get("metric_clamp_summary")),
            )
        elif status == "failed":
            message = info.get("error") or info.get("terminal_diagnostic_message")
            if message is not None:
                message = str(message).encode("utf-8")[:1024].decode("utf-8", errors="ignore")
            target = current.failed(
                terminal_at=now,
                diagnostic_code=str(info.get("terminal_diagnostic_code", "PROCESSING_FAILED")),
                diagnostic_message=message,
            )
        elif status == "in_progress" and current.status == "in_progress":
            return True
        else:
            raise ValueError(f"unsupported schema-5 processing status transition: {status}")
        commit_terminal_settings_v1(
            job_root=job_root,
            current=current,
            target=target,
            allocation_adapter=platform_allocation_adapter,
            allocation_counter=platform_allocation_counter,
        )
        return True
    except Exception as error:
        print(f"Warning: Could not commit schema-5 processing status: {error}")
        return False


def find_settings_file(output_dir: Path, batch_name: str | None = None) -> Path | None:
    """
    Find a settings file in the output directory.

    Args:
        output_dir: Output directory to search
        batch_name: Specific batch name to look for (if None, finds any)

    Returns:
        Path to settings file or None if not found

    Side Effects:
        Reads directory contents from filesystem
    """
    try:
        fixed = output_dir / SETTINGS_ARTIFACT_RELATIVE_PATH
        if fixed.is_file() and not fixed.is_symlink():
            return fixed
        if batch_name:
            # Look for specific batch
            settings_file = output_dir / f"{batch_name}-settings.json"
            if settings_file.exists():
                return settings_file
        else:
            candidates = sorted(
                output_dir.glob("*-settings.json"),
                key=lambda path: (path.stat().st_mtime_ns, path.name),
                reverse=True,
            )
            if candidates:
                return candidates[0]

        return None

    except Exception:
        return None


def can_resume_processing(output_dir: Path) -> dict[str, Any]:
    """
    Check if processing can be resumed from the given directory.

    Args:
        output_dir: Output directory to check

    Returns:
        Dictionary with resume information including:
        - can_resume: bool
        - settings_file: Path | None
        - batch_name: str | None
        - status: str | None
        - progress_info: dict | None
        - recommendations: list[str]

    Side Effects:
        Reads files and directory structure from filesystem
    """
    result: dict[str, Any] = {
        "can_resume": False,
        "settings_file": None,
        "batch_name": None,
        "status": None,
        "progress_info": None,
        "recommendations": [],
    }

    try:
        # Find settings file
        settings_file = find_settings_file(output_dir)
        if not settings_file:
            result["recommendations"].append("No settings file found - cannot resume")
            return result

        # Load settings
        settings_data = load_processing_settings(settings_file)
        if not settings_data:
            result["recommendations"].append("Could not load settings file")
            return result

        result["settings_file"] = settings_file
        result["batch_name"] = settings_data.get("metadata", {}).get("batch_name")
        result["status"] = settings_data.get("metadata", {}).get("processing_status")

        # Check if can resume based on status
        if result["status"] in ["completed"]:
            result["recommendations"].append("Processing already completed")
            return result
        elif result["status"] in ["in_progress", "paused", "failed"]:
            result["can_resume"] = True

            # Analyze progress
            progress_info = analyze_processing_progress(output_dir, settings_data)
            result["progress_info"] = progress_info

            if progress_info["frames_processed"] > 0:
                result["recommendations"].append(
                    f"Found {progress_info['frames_processed']} processed frames - can resume from where left off"
                )
            else:
                result["recommendations"].append(
                    "No processed frames found - will restart from beginning"
                )

        return result

    except Exception as e:
        result["recommendations"].append(f"Error checking resume capability: {e}")
        return result


def analyze_processing_progress(  # noqa: C901
    output_dir: Path, settings_data: dict[str, Any]
) -> dict[str, Any]:
    """
    Analyze how much processing has been completed.

    Args:
        output_dir: Output directory path
        settings_data: Loaded settings data

    Returns:
        Dictionary with progress analysis including:
        - frames_processed: int
        - vr_frames_created: int
        - intermediate_stages: dict
        - can_resume_from_intermediates: bool

    Side Effects:
        Reads directory contents and counts files
    """
    progress: dict[str, Any] = {
        "frames_processed": 0,
        "vr_frames_created": 0,
        "intermediate_stages": {},
        "can_resume_from_intermediates": False,
    }

    try:
        # Check each intermediate directory
        for stage_name, dir_name in INTERMEDIATE_DIRS.items():
            stage_dir = output_dir / dir_name
            if stage_dir.exists():
                frame_count = len(list(stage_dir.glob("*.png")))
                progress["intermediate_stages"][stage_name] = {
                    "directory": str(stage_dir),
                    "frames_found": frame_count,
                }

                if stage_name == "vr_frames":
                    progress["vr_frames_created"] = frame_count

        # Determine overall progress
        if progress["vr_frames_created"] > 0:
            progress["frames_processed"] = progress["vr_frames_created"]
            progress["can_resume_from_intermediates"] = True
        elif "disparity_maps" in progress["intermediate_stages"]:
            depth_frames = progress["intermediate_stages"]["disparity_maps"]["frames_found"]
            if depth_frames > 0:
                progress["frames_processed"] = depth_frames
                progress["can_resume_from_intermediates"] = True
        return progress

    except Exception as e:
        print(f"Warning: Could not analyze processing progress: {e}")
        return progress
