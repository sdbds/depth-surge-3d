"""Typed validation for the final user-facing processing settings."""

from __future__ import annotations

import math
from numbers import Integral, Real
from os import PathLike
from typing import Any, Literal

from .constants import DEFAULT_SETTINGS, VALIDATION_RANGES


SettingsSource = Literal["explicit", "legacy_disk"]
REMOVED_SETTING_NAMES = {
    "baseline",
    "focal_length",
    "hole_fill_quality",
}

_OPTIONAL_SETTING_NAMES = {
    "start_time",
    "end_time",
    "depth_model_version",
    "model_path",
    "model_size",
    "depth_resolution",
    "use_metric_depth",
    "device",
    "denoising_steps",
    "seed",
    "processing_mode",
    "video_encoder",
    "per_eye_width",
    "per_eye_height",
    "vr_output_width",
    "vr_output_height",
    "source_width",
    "source_height",
    "source_fps",
    "enable_live_preview",
    "preview_update_interval",
    "verbose",
}
_KNOWN_SETTING_NAMES = set(DEFAULT_SETTINGS) | _OPTIONAL_SETTING_NAMES

_EXISTING_BOOLEAN_SETTINGS = {
    "preserve_audio",
    "keep_intermediates",
    "apply_distortion",
    "experimental_frame_interpolation",
}
_EXISTING_CHOICE_SETTINGS = {
    "vr_format": {"side_by_side", "over_under"},
    "fisheye_projection": {"equidistant", "equisolid", "orthogonal", "stereographic"},
    "super_sample": {"auto", "none", "1080p", "4k"},
    "upscale_model": {"none", "x2", "x4", "x4-conservative"},
}
_OPTIONAL_INTEGER_SETTINGS = {
    "denoising_steps",
    "seed",
    "per_eye_width",
    "per_eye_height",
    "vr_output_width",
    "vr_output_height",
    "source_width",
    "source_height",
}


def _number(name: str, value: object, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number in [{low}, {high}]")
    normalized = float(value)
    if not math.isfinite(normalized) or not low <= normalized <= high:
        raise ValueError(f"{name} must be a finite number in [{low}, {high}]")
    return normalized


def _integer(name: str, value: object, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    normalized = int(value)
    if not low <= normalized <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return normalized


def _boolean(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _choice(name: str, value: object, choices: set[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        options = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of: {options}")
    return value


def _validate_dibr_setting(name: str, value: object) -> Any:
    if name in {"stereo_strength", "convergence", "scene_cut_threshold"}:
        low, high = VALIDATION_RANGES[name]
        return _number(name, value, low, high)
    if name in {"min_scene_frames", "stereo_io_workers"}:
        low, high = VALIDATION_RANGES[name]
        return _integer(name, value, int(low), int(high))
    if name == "occlusion_fill":
        return _choice(name, value, {"none", "background"})
    if name == "raw_storage_dtype":
        return _choice(name, value, {"auto", "float16", "float32"})
    if name == "migrate_legacy":
        return _choice(name, value, {"archive", "delete"})
    if name == "scene_detection":
        return _boolean(name, value)
    raise ValueError(f"unknown setting: {name}")


def _validate_target_fps(value: object) -> int | str:
    if isinstance(value, str):
        if value == "original" or value.isdigit() and 1 <= int(value) <= 120:
            return value
        raise ValueError("target_fps must be original or a value in [1, 120]")
    return _integer("target_fps", value, 1, 120)


def _validate_resolution_setting(name: str, value: object) -> str | None:
    if isinstance(value, str):
        return value
    if value is None and name == "min_resolution":
        return None
    raise ValueError(f"{name} must be a string")


def _validate_output_dir(value: object) -> str:
    if isinstance(value, (str, PathLike)):
        return str(value)
    raise ValueError("output_dir must be a filesystem path")


def _validate_existing_setting(name: str, value: object) -> Any:
    if name in _EXISTING_BOOLEAN_SETTINGS:
        return _boolean(name, value)
    if name in _EXISTING_CHOICE_SETTINGS:
        return _choice(name, value, _EXISTING_CHOICE_SETTINGS[name])
    if name in {"fisheye_fov", "crop_factor", "fisheye_crop_factor"}:
        low, high = VALIDATION_RANGES[name]
        return _number(name, value, low, high)
    if name == "target_fps":
        return _validate_target_fps(value)
    if name in {"temporal_window_size", "temporal_window_overlap"}:
        return _integer(name, value, 0, 1_000_000)
    if name in {"vr_resolution", "min_resolution"}:
        return _validate_resolution_setting(name, value)
    if name == "output_dir":
        return _validate_output_dir(value)
    raise ValueError(f"unknown setting: {name}")


def _validate_optional_integer(name: str, value: object) -> int | None:
    if value is None and name in {"denoising_steps", "seed"}:
        return None
    return _integer(name, value, 0, 1_000_000)


def _validate_optional_text(name: str, value: object) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"{name} must be a string or null")


def _validate_depth_resolution(value: object) -> int | str:
    if isinstance(value, str) or (isinstance(value, Integral) and not isinstance(value, bool)):
        return value
    raise ValueError("depth_resolution must be auto or an integer resolution")


def _validate_optional_mode(name: str, value: object) -> str | None:
    if isinstance(value, str):
        return value
    if value is None and name == "processing_mode":
        return None
    raise ValueError(f"{name} must be a string")


def _validate_optional_setting(name: str, value: object) -> Any:
    if name in {"use_metric_depth", "enable_live_preview", "verbose"}:
        return _boolean(name, value)
    if name in _OPTIONAL_INTEGER_SETTINGS:
        return _validate_optional_integer(name, value)
    if name in {"source_fps", "preview_update_interval"}:
        return _number(name, value, 0.0, 1_000_000.0)
    if name in {"start_time", "end_time", "model_path", "model_size"}:
        return _validate_optional_text(name, value)
    if name == "depth_model_version":
        return _choice(name, value, {"v2", "v3", "see_through"})
    if name == "depth_resolution":
        return _validate_depth_resolution(value)
    if name in {"device", "processing_mode"}:
        return _validate_optional_mode(name, value)
    if name == "video_encoder":
        return _choice(name, value, {"auto", "libx264", "nvenc"})
    raise ValueError(f"unknown setting: {name}")


def _validate_value(name: str, value: object) -> Any:
    if name in {
        "stereo_strength",
        "convergence",
        "occlusion_fill",
        "scene_detection",
        "scene_cut_threshold",
        "min_scene_frames",
        "raw_storage_dtype",
        "stereo_io_workers",
        "migrate_legacy",
    }:
        return _validate_dibr_setting(name, value)
    if name in DEFAULT_SETTINGS:
        return _validate_existing_setting(name, value)
    return _validate_optional_setting(name, value)


def validate_settings(
    values: dict[str, Any],
    *,
    source: SettingsSource,
) -> dict[str, Any]:
    """Validate explicit settings or strip obsolete fields from legacy disk data."""

    if source not in {"explicit", "legacy_disk"}:
        raise ValueError("settings source must be explicit or legacy_disk")
    if not isinstance(values, dict):
        raise TypeError("settings values must be a dictionary")

    provided = dict(values)
    removed = sorted(REMOVED_SETTING_NAMES.intersection(provided))
    unknown = sorted(set(provided) - _KNOWN_SETTING_NAMES - REMOVED_SETTING_NAMES)
    if source == "explicit":
        if removed:
            raise ValueError(f"removed setting is not supported: {removed[0]}")
        if unknown:
            raise ValueError(f"unknown setting: {unknown[0]}")
    else:
        for name in removed + unknown:
            provided.pop(name, None)

    validated = dict(DEFAULT_SETTINGS)
    for name, value in provided.items():
        validated[name] = _validate_value(name, value)
    for name, value in tuple(validated.items()):
        if name not in provided:
            validated[name] = _validate_value(name, value)
    return validated
