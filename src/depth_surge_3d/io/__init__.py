"""I/O package with lazy public exports to keep low-level adapters acyclic."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_OPERATION_EXPORTS = {
    "validate_video_file",
    "validate_image_file",
    "get_video_properties",
    "get_video_info_ffprobe",
    "create_output_directories",
    "get_frame_files",
    "calculate_directory_size",
    "cleanup_intermediate_files",
    "get_available_space",
    "save_processing_settings",
    "load_processing_settings",
    "update_processing_status",
    "find_settings_file",
    "can_resume_processing",
    "analyze_processing_progress",
    "verify_ffmpeg_installation",
}
_RESUME_EXPORTS = {
    "ResumeReport",
    "ResumeStage",
    "build_resume_report",
    "apply_legacy_migration",
}

__all__ = sorted(_OPERATION_EXPORTS | _RESUME_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _OPERATION_EXPORTS:
        value = getattr(import_module(".operations", __name__), name)
    elif name in _RESUME_EXPORTS:
        value = getattr(import_module(".resume", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
