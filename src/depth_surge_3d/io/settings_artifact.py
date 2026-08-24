"""Strict schema-5 processing settings and immutable output identity.

This module owns the semantic byte contract.  Filesystem reservation and
publication live in :mod:`depth_surge_3d.io.reservation_protocol` so recovery
can validate the intended bytes without rebuilding caller dictionaries.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
import secrets
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal

import numpy as np

from ..core.file_identity import FILE_IDENTITY_ALGORITHM_VERSION
from ..core.settings import JOB_OUTPUT_NAME_V1_RESOLUTION_PRESETS, validate_settings
from .canonical_json import (
    CanonicalJsonError,
    canonical_json_bytes,
    strict_canonical_json_loads,
    validate_unicode_scalars,
)


SETTINGS_ARTIFACT_MAX_RAW_BYTES = 512 * 1024
STEREO_CONTROL_FRAME_CAP = 4_194_304

TOP_LEVEL_KEYS = frozenset(
    {
        "content_fingerprint",
        "metadata",
        "video_properties",
        "processing_settings",
        "output_info",
        "runtime_info",
    }
)
METADATA_V5_KEYS = frozenset(
    {
        "batch_name",
        "job_id",
        "settings_identity_fingerprint",
        "settings_artifact_relative_path",
        "job_reservation_generation",
        "job_terminal_reservation_id",
        "settings_schema_version",
        "source_settings_schema_version",
        "settings_revision",
        "processing_attempt",
        "source_video",
        "source_video_name",
        "source_video_fingerprint_algorithm",
        "source_video_fingerprint",
        "project_version",
        "created_at",
        "attempt_started_at",
        "last_updated_at",
        "terminal_at",
        "processing_duration_ms",
        "processing_status",
        "cleanup_status",
        "final_media_producer_contract_version",
        "final_media_publication_generation",
    }
)
VIDEO_PROPERTIES_V5_KEYS = frozenset(
    {
        "width",
        "height",
        "fps",
        "frame_count",
        "duration",
        "codec",
        "sample_aspect_ratio_numerator",
        "sample_aspect_ratio_denominator",
        "sample_aspect_ratio",
    }
)
PROCESSING_SETTINGS_V5_KEYS = frozenset(
    {
        "apply_distortion",
        "convergence",
        "crop_factor",
        "denoising_steps",
        "depth_model_version",
        "depth_resolution",
        "device",
        "direct_vr_encode",
        "enable_live_preview",
        "end_time",
        "experimental_frame_interpolation",
        "fisheye_crop_factor",
        "fisheye_fov",
        "fisheye_projection",
        "keep_intermediates",
        "max_disparity_percent",
        "metric_convergence_distance",
        "migrate_legacy",
        "min_resolution",
        "min_scene_frames",
        "model_path",
        "model_size",
        "occlusion_fill",
        "occlusion_fill_max_px",
        "output_dir",
        "per_eye_height",
        "per_eye_width",
        "preserve_audio",
        "preview_update_interval",
        "raw_storage_dtype",
        "scene_cut_threshold",
        "scene_detection",
        "seed",
        "source_fps",
        "source_height",
        "source_width",
        "start_time",
        "stereo_geometry_mode",
        "stereo_io_workers",
        "stereo_render_mode",
        "stereo_strength",
        "super_sample",
        "target_fps",
        "temporal_postprocessor",
        "temporal_window_overlap",
        "temporal_window_size",
        "upscale_model",
        "use_metric_depth",
        "verbose",
        "video_encoder",
        "video_path",
        "virtual_baseline_mm",
        "vr_format",
        "vr_output_height",
        "vr_output_width",
        "vr_resolution",
    }
)
OUTPUT_INFO_V5_KEYS = frozenset(
    {
        "output_directory",
        "expected_output_filename",
        "expected_final_relative_path",
        "output_name_algorithm_version",
    }
)
RUNTIME_INFO_V5_KEYS = frozenset(
    {
        "final_output_relative_path",
        "frames_processed",
        "processing_time_seconds",
        "metric_clamp_summary",
        "terminal_diagnostic_code",
        "terminal_diagnostic_message",
    }
)
METRIC_SUMMARY_KEYS = frozenset(
    {
        "schema_version",
        "affected_frame_count",
        "mean_clamped_fraction",
        "max_clamped_fraction",
        "compatibility_summary_sha256",
    }
)

_UTC_USEC = re.compile(
    r"(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})T"
    r"(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2}):(?P<second>[0-9]{2})\."
    r"(?P<microsecond>[0-9]{6})Z",
    flags=re.ASCII,
)
_HEX = re.compile(r"[0-9a-f]+", flags=re.ASCII)
_DIAGNOSTIC_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,63}", flags=re.ASCII)
_CUSTOM_RESOLUTION = re.compile(r"custom:([1-9][0-9]{0,4})x([1-9][0-9]{0,4})", flags=re.ASCII)
_INVALID_FILENAME_SCALARS = frozenset('<>:"/\\|?*')
_WINDOWS_DEVICE_BASES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{suffix}" for suffix in (*range(1, 10), "\u00b9", "\u00b2", "\u00b3")}
    | {f"LPT{suffix}" for suffix in (*range(1, 10), "\u00b9", "\u00b2", "\u00b3")}
)


class SettingsArtifactError(ValueError):
    """Base class for a settings artifact contract violation."""


class SettingsArtifactSchemaError(SettingsArtifactError):
    """The object is not a complete, closed ProcessingSettingsArtifactV5."""


class SettingsArtifactStateError(SettingsArtifactError):
    """The object has an impossible processing/runtime state combination."""


class SettingsArtifactTransitionError(SettingsArtifactError):
    """A requested typed state transition is not legal from the current state."""


def _schema(condition: bool, message: str) -> None:
    if not condition:
        raise SettingsArtifactSchemaError(message)


def _state(condition: bool, message: str) -> None:
    if not condition:
        raise SettingsArtifactStateError(message)


def _closed_object(value: object, keys: frozenset[str], label: str) -> dict[str, Any]:
    _schema(type(value) is dict, f"{label} must be an object")
    assert isinstance(value, dict)
    _schema(set(value) == keys, f"{label} has missing or extra members")
    return value


def _scalar_string(value: object, label: str, *, nonempty: bool = False) -> str:
    _schema(type(value) is str, f"{label} must be a Unicode scalar string")
    assert isinstance(value, str)
    try:
        validate_unicode_scalars(value)
    except CanonicalJsonError as error:
        raise SettingsArtifactSchemaError(f"{label} contains an isolated surrogate") from error
    _schema(not nonempty or bool(value), f"{label} must be nonempty")
    return value


def _integer(value: object, label: str, low: int, high: int) -> int:
    _schema(
        type(value) is int and low <= value <= high, f"{label} must be an integer in {low}..{high}"
    )
    assert isinstance(value, int)
    return value


def _binary64(
    value: object, label: str, low: float | None = None, high: float | None = None
) -> float:
    _schema(type(value) is float and math.isfinite(value), f"{label} must be a finite binary64")
    assert isinstance(value, float)
    _schema(low is None or value >= low, f"{label} is below its minimum")
    _schema(high is None or value <= high, f"{label} is above its maximum")
    return value


def _hex(value: object, digits: int, label: str) -> str:
    text = _scalar_string(value, label)
    _schema(
        len(text) == digits and _HEX.fullmatch(text) is not None,
        f"{label} must be lowercase hex{digits}",
    )
    return text


def utc_ordinal_usec_v1(value: object) -> int:
    """Parse exact Gregorian UTC text and return integer microseconds since year 1."""

    text = _scalar_string(value, "utc-usec")
    match = _UTC_USEC.fullmatch(text)
    _schema(
        match is not None and len(text.encode("ascii")) == 27, "timestamp must be exact utc-usec"
    )
    assert match is not None
    parts = {key: int(item) for key, item in match.groupdict().items()}
    _schema(
        parts["hour"] <= 23 and parts["minute"] <= 59 and parts["second"] <= 59,
        "timestamp time is invalid",
    )
    try:
        day_ordinal = date(parts["year"], parts["month"], parts["day"]).toordinal() - 1
    except ValueError as error:
        raise SettingsArtifactSchemaError("timestamp Gregorian date is invalid") from error
    seconds = day_ordinal * 86_400 + parts["hour"] * 3_600 + parts["minute"] * 60 + parts["second"]
    return seconds * 1_000_000 + parts["microsecond"]


def attempt_duration_ms_v1(terminal_at: object, attempt_started_at: object) -> int:
    return (
        max(0, utc_ordinal_usec_v1(terminal_at) - utc_ordinal_usec_v1(attempt_started_at)) // 1_000
    )


def duration_seconds_from_ms_v1(milliseconds: object) -> float:
    value = _integer(milliseconds, "processing_duration_ms", 0, 2**64 - 1)
    return float(np.float64(np.float64(value) / np.float64(1000.0)))


def _ascii_fold(value: str) -> str:
    return "".join(
        chr(ord(character) + 32) if "A" <= character <= "Z" else character for character in value
    )


def validate_contained_legacy_final_component_v1(value: object) -> str:
    text = _scalar_string(value, "legacy final component", nonempty=True)
    _schema(text not in {".", ".."}, "legacy final component is not contained")
    _schema(
        "\x00" not in text and "/" not in text and "\\" not in text,
        "legacy final component is not contained",
    )
    _schema(
        re.match(r"[A-Za-z]:", text, flags=re.ASCII) is None,
        "legacy final component begins with a drive",
    )
    return text


def validate_portable_final_component_v1(value: object) -> str:
    text = validate_contained_legacy_final_component_v1(value)
    _schema(not text.endswith((" ", ".")), "portable final component has a forbidden suffix")
    _schema(
        all(unicodedata.category(character) != "Cc" for character in text),
        "portable final component contains a control scalar",
    )
    _schema(
        all(character not in _INVALID_FILENAME_SCALARS for character in text),
        "portable final component contains a forbidden scalar",
    )
    _schema(len(text.encode("utf-8")) <= 240, "portable final component exceeds 240 UTF-8 bytes")
    _schema(
        len(text.encode("utf-16-le")) // 2 <= 240,
        "portable final component exceeds 240 UTF-16 code units",
    )
    first_dot_base = text.split(".", 1)[0]
    _schema(
        _ascii_fold(first_dot_base)
        not in frozenset(_ascii_fold(name) for name in _WINDOWS_DEVICE_BASES),
        "portable final component uses a reserved Windows device base",
    )
    return text


def _resolution_filename_token(value: object) -> str:
    text = _scalar_string(value, "vr_resolution", nonempty=True)
    if text == "auto" or text in JOB_OUTPUT_NAME_V1_RESOLUTION_PRESETS:
        return text
    match = _CUSTOM_RESOLUTION.fullmatch(text)
    _schema(match is not None, "vr_resolution is not supported by job-output-name-v1")
    assert match is not None
    width, height = (int(part) for part in match.groups())
    _schema(width <= 10_000 and height <= 10_000, "custom resolution exceeds 10000")
    return f"custom-{width}x{height}"


def job_output_name_v1(source_video_name: object, vr_format: object, vr_resolution: object) -> str:
    """Return the frozen portable final component for a new job."""

    source_name = _scalar_string(source_video_name, "source_video_name")
    _schema("/" not in source_name, "source_video_name may not contain a slash")
    _schema(
        all(unicodedata.category(character) != "Cc" for character in source_name),
        "source_video_name may not contain control scalars",
    )
    if source_name:
        last_dot = source_name.rfind(".")
        stem = source_name[:last_dot] if 0 < last_dot < len(source_name) - 1 else source_name
    else:
        stem = "output"
    sanitized = "".join(
        "_" if character in _INVALID_FILENAME_SCALARS else character for character in stem
    )
    sanitized = re.sub(r"_+", "_", sanitized, flags=re.ASCII).strip("_")
    format_value = _scalar_string(vr_format, "vr_format")
    _schema(format_value in {"side_by_side", "over_under"}, "vr_format is invalid")
    format_token = "side-by-side" if format_value == "side_by_side" else "over-under"
    resolution_token = _resolution_filename_token(vr_resolution)
    tail = f"_3D_{format_token}_{resolution_token}.mp4"
    candidate = f"{sanitized}{tail}"
    try:
        return validate_portable_final_component_v1(candidate)
    except SettingsArtifactSchemaError:
        pass
    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:32]
    fallback_tail = f"~{digest}{tail}"
    for prefix_length in range(len(sanitized), -1, -1):
        fallback = f"{sanitized[:prefix_length]}{fallback_tail}"
        try:
            return validate_portable_final_component_v1(fallback)
        except SettingsArtifactSchemaError:
            continue
    raise SettingsArtifactSchemaError("job-output-name-v1 has no portable fallback")


def _materialize_processing_settings(values: dict[str, Any]) -> dict[str, Any]:
    _schema(type(values) is dict, "processing_settings must be an object")
    optional_defaults: dict[str, Any] = {
        "start_time": None,
        "end_time": None,
        "depth_model_version": "v2",
        "model_path": None,
        "model_size": None,
        "depth_resolution": "auto",
        "use_metric_depth": False,
        "device": "auto",
        "denoising_steps": None,
        "seed": None,
        "video_encoder": "auto",
        "video_path": None,
        "per_eye_width": None,
        "per_eye_height": None,
        "vr_output_width": None,
        "vr_output_height": None,
        "source_width": None,
        "source_height": None,
        "source_fps": None,
        "enable_live_preview": False,
        "preview_update_interval": None,
        "verbose": False,
    }
    nullable_extension_keys = frozenset(optional_defaults)
    validation_input = {
        key: item
        for key, item in values.items()
        if item is not None or key not in nullable_extension_keys
    }
    validated = {**optional_defaults, **validate_settings(validation_input, source="explicit")}
    for key in nullable_extension_keys:
        if key in values and values[key] is None:
            validated[key] = None
    if isinstance(validated["target_fps"], str) and validated["target_fps"].isdigit():
        validated["target_fps"] = int(validated["target_fps"])
    if isinstance(validated["depth_resolution"], str) and validated["depth_resolution"].isdigit():
        validated["depth_resolution"] = int(validated["depth_resolution"])
    return validated


def _validate_processing_settings(value: object) -> dict[str, Any]:  # noqa: C901
    settings = _closed_object(value, PROCESSING_SETTINGS_V5_KEYS, "processing_settings")
    numeric_ranges = {
        "stereo_strength": (0.0, 5.0),
        "convergence": (0.0, 1.0),
        "virtual_baseline_mm": (0.0, 100.0),
        "max_disparity_percent": (0.0, 5.0),
        "scene_cut_threshold": (0.0, 1.0),
        "fisheye_fov": (75.0, 180.0),
        "crop_factor": (0.5, 1.0),
        "fisheye_crop_factor": (0.5, 2.0),
    }
    for key, bounds in numeric_ranges.items():
        _binary64(settings[key], f"processing_settings.{key}", *bounds)
    for key in ("source_fps", "preview_update_interval"):
        if settings[key] is not None:
            _binary64(settings[key], f"processing_settings.{key}", 0.0, 1_000_000.0)
    convergence = settings["metric_convergence_distance"]
    if convergence != "auto":
        _binary64(convergence, "processing_settings.metric_convergence_distance", 0.1, 1000.0)
    _integer(settings["min_scene_frames"], "processing_settings.min_scene_frames", 1, 1_000_000)
    _integer(settings["stereo_io_workers"], "processing_settings.stereo_io_workers", 1, 16)
    for key in ("temporal_window_size", "temporal_window_overlap"):
        _integer(settings[key], f"processing_settings.{key}", 0, 1_000_000)
    _integer(settings["occlusion_fill_max_px"], "processing_settings.occlusion_fill_max_px", 1, 32)
    for key in ("denoising_steps", "seed"):
        if settings[key] is not None:
            _integer(settings[key], f"processing_settings.{key}", 0, 1_000_000)
    for key in (
        "per_eye_width",
        "per_eye_height",
        "vr_output_width",
        "vr_output_height",
        "source_width",
        "source_height",
    ):
        if settings[key] is not None:
            _integer(settings[key], f"processing_settings.{key}", 1, 10_000)
    depth_resolution = settings["depth_resolution"]
    _schema(
        depth_resolution == "auto" or type(depth_resolution) is int,
        "depth_resolution must be auto or integer",
    )
    if type(depth_resolution) is int:
        _integer(depth_resolution, "processing_settings.depth_resolution", 1, 10_000)
    target_fps = settings["target_fps"]
    _schema(
        target_fps is None or target_fps == "original" or type(target_fps) is int,
        "target_fps has an invalid type",
    )
    if type(target_fps) is int:
        _integer(target_fps, "processing_settings.target_fps", 1, 120)
    boolean_keys = {
        "scene_detection",
        "preserve_audio",
        "keep_intermediates",
        "direct_vr_encode",
        "apply_distortion",
        "experimental_frame_interpolation",
        "use_metric_depth",
        "enable_live_preview",
        "verbose",
    }
    for key in boolean_keys:
        _schema(type(settings[key]) is bool, f"processing_settings.{key} must be boolean")
    choices = {
        "stereo_geometry_mode": {"relative", "metric_camera"},
        "occlusion_fill": {"none", "background"},
        "stereo_render_mode": {"fast", "quality"},
        "raw_storage_dtype": {"auto", "float16", "float32"},
        "temporal_postprocessor": {"off", "vdpp"},
        "migrate_legacy": {"archive", "delete"},
        "vr_format": {"side_by_side", "over_under"},
        "fisheye_projection": {"equidistant", "equisolid", "orthogonal", "stereographic"},
        "super_sample": {"auto", "none", "1080p", "4k"},
        "upscale_model": {"none", "x2", "x4", "x4-conservative"},
        "depth_model_version": {"v2", "v3", "see_through", "moge2"},
        "device": {"auto", "cuda", "cpu"},
        "video_encoder": {"auto", "libx264", "nvenc"},
    }
    for key, allowed in choices.items():
        _schema(
            type(settings[key]) is str and settings[key] in allowed,
            f"processing_settings.{key} is invalid",
        )
    _resolution_filename_token(settings["vr_resolution"])
    for key in (
        "start_time",
        "end_time",
        "model_path",
        "model_size",
        "video_path",
        "min_resolution",
    ):
        if settings[key] is not None:
            _scalar_string(settings[key], f"processing_settings.{key}")
    _scalar_string(settings["output_dir"], "processing_settings.output_dir")
    _state(
        settings["stereo_geometry_mode"] != "metric_camera"
        or settings["temporal_postprocessor"] == "off",
        "metric geometry cannot use VDPP",
    )
    return settings


def _validate_video_properties(value: object) -> dict[str, Any]:
    video = _closed_object(value, VIDEO_PROPERTIES_V5_KEYS, "video_properties")
    _integer(video["width"], "video_properties.width", 1, 10_000)
    _integer(video["height"], "video_properties.height", 1, 10_000)
    _binary64(video["fps"], "video_properties.fps", np.nextafter(0.0, 1.0), None)
    _integer(video["frame_count"], "video_properties.frame_count", 1, STEREO_CONTROL_FRAME_CAP)
    _binary64(video["duration"], "video_properties.duration", 0.0, None)
    _integer(video["codec"], "video_properties.codec", 0, 4_294_967_295)
    numerator = _integer(
        video["sample_aspect_ratio_numerator"],
        "video_properties.sample_aspect_ratio_numerator",
        1,
        2**64 - 1,
    )
    denominator = _integer(
        video["sample_aspect_ratio_denominator"],
        "video_properties.sample_aspect_ratio_denominator",
        1,
        2**64 - 1,
    )
    _schema(math.gcd(numerator, denominator) == 1, "sample aspect ratio must be coprime")
    _schema(
        video["sample_aspect_ratio"] == f"{numerator}:{denominator}",
        "sample aspect ratio spelling is invalid",
    )
    return video


def _validate_metric_summary(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    summary = _closed_object(value, METRIC_SUMMARY_KEYS, "runtime_info.metric_clamp_summary")
    _schema(
        summary["schema_version"] == 1 and type(summary["schema_version"]) is int,
        "metric summary schema is invalid",
    )
    _integer(summary["affected_frame_count"], "metric summary affected_frame_count", 0, 2**64 - 1)
    mean = _binary64(summary["mean_clamped_fraction"], "metric summary mean", 0.0, 1.0)
    maximum = _binary64(summary["max_clamped_fraction"], "metric summary max", 0.0, 1.0)
    _schema(
        mean <= maximum or summary["affected_frame_count"] == 0, "metric summary mean exceeds max"
    )
    _hex(summary["compatibility_summary_sha256"], 64, "metric summary hash")
    return summary


def _settings_identity_object(metadata: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
    return {
        "algorithm_version": "settings-identity-v1",
        "job_id": metadata["job_id"],
        "settings_artifact_relative_path": metadata["settings_artifact_relative_path"],
        "source_video_fingerprint_algorithm": metadata["source_video_fingerprint_algorithm"],
        "source_video_fingerprint": metadata["source_video_fingerprint"],
        "expected_final_relative_path": output["expected_final_relative_path"],
        "output_name_algorithm_version": output["output_name_algorithm_version"],
        "job_reservation_generation": metadata["job_reservation_generation"],
        "job_terminal_reservation_id": metadata["job_terminal_reservation_id"],
    }


def _fingerprint_without(value: dict[str, Any], member: str) -> str:
    return hashlib.sha256(
        canonical_json_bytes({key: item for key, item in value.items() if key != member})
    ).hexdigest()


def _validate_complete_object(value: object) -> dict[str, Any]:  # noqa: C901
    root = _closed_object(value, TOP_LEVEL_KEYS, "ProcessingSettingsArtifactV5")
    _hex(root["content_fingerprint"], 64, "content_fingerprint")
    _schema(
        root["content_fingerprint"] == _fingerprint_without(root, "content_fingerprint"),
        "content fingerprint mismatch",
    )

    metadata = _closed_object(root["metadata"], METADATA_V5_KEYS, "metadata")
    for key in ("batch_name", "source_video", "source_video_name"):
        _scalar_string(metadata[key], f"metadata.{key}")
    _scalar_string(metadata["project_version"], "metadata.project_version", nonempty=True)
    for key in ("job_id", "job_reservation_generation", "job_terminal_reservation_id"):
        _hex(metadata[key], 32, f"metadata.{key}")
    _hex(metadata["settings_identity_fingerprint"], 64, "metadata.settings_identity_fingerprint")
    _schema(
        metadata["settings_artifact_relative_path"] == "processing-settings-v5.json",
        "settings path is invalid",
    )
    _schema(
        metadata["settings_schema_version"] == 5
        and type(metadata["settings_schema_version"]) is int,
        "settings schema must be 5",
    )
    _integer(
        metadata["source_settings_schema_version"], "metadata.source_settings_schema_version", 1, 5
    )
    _integer(metadata["settings_revision"], "metadata.settings_revision", 0, 2**64 - 1)
    _integer(metadata["processing_attempt"], "metadata.processing_attempt", 1, 2**64 - 1)
    _schema(
        metadata["source_video_fingerprint_algorithm"] == FILE_IDENTITY_ALGORITHM_VERSION,
        "source fingerprint algorithm is invalid",
    )
    if metadata["source_video_fingerprint"] is not None:
        _hex(metadata["source_video_fingerprint"], 32, "metadata.source_video_fingerprint")
    created = utc_ordinal_usec_v1(metadata["created_at"])
    started = utc_ordinal_usec_v1(metadata["attempt_started_at"])
    _state(started >= created, "attempt_started_at precedes created_at")
    for key in ("last_updated_at", "terminal_at"):
        if metadata[key] is not None:
            utc_ordinal_usec_v1(metadata[key])
    if metadata["processing_duration_ms"] is not None:
        _integer(
            metadata["processing_duration_ms"], "metadata.processing_duration_ms", 0, 2**64 - 1
        )
    _schema(
        metadata["processing_status"] in {"in_progress", "completed", "failed"},
        "processing status is invalid",
    )
    _schema(
        metadata["cleanup_status"]
        in {
            "not_started",
            "not_requested",
            "pending",
            "complete",
            "incomplete_identity_mismatch",
            "incomplete_error",
        },
        "cleanup status is invalid",
    )
    producer_version = metadata["final_media_producer_contract_version"]
    producer_generation = metadata["final_media_publication_generation"]
    _state(
        (producer_version is None and producer_generation is None)
        or (
            producer_version == 4
            and type(producer_version) is int
            and isinstance(producer_generation, str)
        ),
        "producer marker must be the null pair or version 4 plus generation",
    )
    if producer_generation is not None:
        _hex(producer_generation, 32, "metadata.final_media_publication_generation")

    video = _validate_video_properties(root["video_properties"])
    settings = _validate_processing_settings(root["processing_settings"])
    output = _closed_object(root["output_info"], OUTPUT_INFO_V5_KEYS, "output_info")
    _scalar_string(output["output_directory"], "output_info.output_directory")
    expected = _scalar_string(
        output["expected_final_relative_path"],
        "output_info.expected_final_relative_path",
        nonempty=True,
    )
    _schema(output["expected_output_filename"] == expected, "expected output path spellings differ")
    algorithm = output["output_name_algorithm_version"]
    _schema(
        algorithm in {"job-output-name-v1", "persisted-expected-output-filename-v1"},
        "output name algorithm is invalid",
    )
    _schema(expected.endswith(".mp4"), "expected output path must end in lowercase .mp4")
    if algorithm == "job-output-name-v1":
        validate_portable_final_component_v1(expected)
    else:
        validate_contained_legacy_final_component_v1(expected)
    identity = hashlib.sha256(
        canonical_json_bytes(_settings_identity_object(metadata, output))
    ).hexdigest()
    _schema(
        metadata["settings_identity_fingerprint"] == identity,
        "settings identity fingerprint mismatch",
    )

    runtime = _closed_object(root["runtime_info"], RUNTIME_INFO_V5_KEYS, "runtime_info")
    if runtime["final_output_relative_path"] is not None:
        _scalar_string(
            runtime["final_output_relative_path"], "runtime_info.final_output_relative_path"
        )
    if runtime["frames_processed"] is not None:
        _integer(runtime["frames_processed"], "runtime_info.frames_processed", 0, 2**64 - 1)
    if runtime["processing_time_seconds"] is not None:
        _binary64(
            runtime["processing_time_seconds"], "runtime_info.processing_time_seconds", 0.0, None
        )
    metric_summary = _validate_metric_summary(runtime["metric_clamp_summary"])
    code = runtime["terminal_diagnostic_code"]
    if code is not None:
        _schema(
            type(code) is str and _DIAGNOSTIC_CODE.fullmatch(code) is not None,
            "terminal diagnostic code is invalid",
        )
    message = runtime["terminal_diagnostic_message"]
    if message is not None:
        text = _scalar_string(message, "runtime_info.terminal_diagnostic_message")
        _schema(len(text.encode("utf-8")) <= 1024, "terminal diagnostic message exceeds 1024 bytes")

    status = metadata["processing_status"]
    if status == "in_progress":
        _state(
            metadata["terminal_at"] is None and metadata["processing_duration_ms"] is None,
            "in-progress time fields must be null",
        )
        _state(
            metadata["cleanup_status"] in {"not_started", "pending"},
            "in-progress cleanup state is invalid",
        )
        _state(code is None and message is None, "in-progress diagnostics must be null")
        _state(
            all(
                runtime[key] is None
                for key in (
                    "final_output_relative_path",
                    "frames_processed",
                    "processing_time_seconds",
                    "metric_clamp_summary",
                )
            ),
            "in-progress runtime success fields must be null",
        )
        if metadata["cleanup_status"] == "pending":
            _state(
                settings["keep_intermediates"] is False and producer_generation is not None,
                "cleanup pending requires current publication and disabled retention",
            )
    else:
        terminal = metadata["terminal_at"]
        _state(
            metadata["last_updated_at"] == terminal and terminal is not None,
            "terminal timestamps must match",
        )
        duration = attempt_duration_ms_v1(terminal, metadata["attempt_started_at"])
        _state(metadata["processing_duration_ms"] == duration, "terminal duration is invalid")
        if status == "completed":
            _state(code is None and message is None, "completed diagnostics must be null")
            _state(
                runtime["final_output_relative_path"] == expected, "completed final path is invalid"
            )
            untouched_legacy = (
                metadata["source_settings_schema_version"] <= 4
                and metadata["processing_attempt"] == 1
                and metadata["settings_revision"] == 0
            )
            if untouched_legacy:
                _state(
                    type(runtime["frames_processed"]) is int
                    and 1 <= runtime["frames_processed"] <= video["frame_count"],
                    "legacy completed frame count is invalid",
                )
            else:
                _state(
                    runtime["frames_processed"] == video["frame_count"],
                    "completed frame count is invalid",
                )
            projected = duration_seconds_from_ms_v1(duration)
            actual_seconds = runtime["processing_time_seconds"]
            _state(
                type(actual_seconds) is float
                and np.float64(actual_seconds).view(np.uint64)
                == np.float64(projected).view(np.uint64),
                "processing time projection is invalid",
            )
            allowed_cleanup = (
                {"not_requested"}
                if settings["keep_intermediates"]
                else {
                    "complete",
                    "incomplete_identity_mismatch",
                    "incomplete_error",
                }
            )
            _state(
                metadata["cleanup_status"] in allowed_cleanup, "completed cleanup state is invalid"
            )
            if settings["stereo_geometry_mode"] == "metric_camera" and not untouched_legacy:
                _state(metric_summary is not None, "completed metric jobs require a clamp summary")
            if settings["stereo_geometry_mode"] != "metric_camera":
                _state(metric_summary is None, "relative jobs cannot carry a metric clamp summary")
        else:
            _state(code is not None, "failed settings require a diagnostic code")
            _state(metadata["cleanup_status"] != "pending", "failed cleanup cannot be pending")
            _state(
                all(
                    runtime[key] is None
                    for key in (
                        "final_output_relative_path",
                        "frames_processed",
                        "processing_time_seconds",
                        "metric_clamp_summary",
                    )
                ),
                "failed success runtime fields must be null",
            )
    return root


def _with_fingerprints(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    metadata = result["metadata"]
    output = result["output_info"]
    metadata["settings_identity_fingerprint"] = hashlib.sha256(
        canonical_json_bytes(_settings_identity_object(metadata, output))
    ).hexdigest()
    result["content_fingerprint"] = _fingerprint_without(result, "content_fingerprint")
    return result


@dataclass(frozen=True)
class ProcessingSettingsArtifactV5:
    """Immutable validated schema-5 settings value."""

    _value: dict[str, Any]

    @classmethod
    def from_object(cls, value: object) -> "ProcessingSettingsArtifactV5":
        owned = copy.deepcopy(value)
        validated = _validate_complete_object(owned)
        return cls(validated)

    @classmethod
    def parse(cls, raw: bytes) -> "ProcessingSettingsArtifactV5":
        try:
            value = strict_canonical_json_loads(raw, max_bytes=SETTINGS_ARTIFACT_MAX_RAW_BYTES)
        except CanonicalJsonError as error:
            raise SettingsArtifactSchemaError(str(error)) from error
        return cls.from_object(value)

    def to_object(self) -> dict[str, Any]:
        return copy.deepcopy(self._value)

    def encode(self) -> bytes:
        raw = canonical_json_bytes(self._value)
        if len(raw) > SETTINGS_ARTIFACT_MAX_RAW_BYTES:
            raise SettingsArtifactSchemaError("settings artifact exceeds 512 KiB")
        return raw

    @property
    def revision(self) -> int:
        return int(self._value["metadata"]["settings_revision"])

    @property
    def status(self) -> str:
        return str(self._value["metadata"]["processing_status"])

    def _replace(
        self,
        *,
        metadata: dict[str, Any],
        runtime: dict[str, Any],
        processing_settings: dict[str, Any] | None = None,
    ) -> "ProcessingSettingsArtifactV5":
        value = self.to_object()
        value["metadata"] = metadata
        value["runtime_info"] = runtime
        if processing_settings is not None:
            value["processing_settings"] = processing_settings
        value = _with_fingerprints(value)
        return ProcessingSettingsArtifactV5.from_object(value)

    def with_processing_settings(
        self,
        processing_settings: dict[str, Any],
        *,
        updated_at: str,
    ) -> "ProcessingSettingsArtifactV5":
        if self.status != "in_progress":
            raise SettingsArtifactTransitionError(
                "processing settings rewrite requires an in-progress job"
            )
        utc_ordinal_usec_v1(updated_at)
        materialized = _materialize_processing_settings(processing_settings)
        metadata = copy.deepcopy(self._value["metadata"])
        metadata["settings_revision"] += 1
        metadata["last_updated_at"] = updated_at
        return self._replace(
            metadata=metadata,
            runtime=copy.deepcopy(self._value["runtime_info"]),
            processing_settings=materialized,
        )

    def with_producer_marker(
        self, *, generation: str, updated_at: str
    ) -> "ProcessingSettingsArtifactV5":
        if self.status != "in_progress":
            raise SettingsArtifactTransitionError("producer marker requires an in-progress job")
        _hex(generation, 32, "producer generation")
        utc_ordinal_usec_v1(updated_at)
        metadata = copy.deepcopy(self._value["metadata"])
        current = metadata["final_media_publication_generation"]
        if current is not None:
            if current != generation:
                raise SettingsArtifactTransitionError("producer marker is immutable")
            return self
        metadata["settings_revision"] += 1
        metadata["last_updated_at"] = updated_at
        metadata["final_media_producer_contract_version"] = 4
        metadata["final_media_publication_generation"] = generation
        return self._replace(metadata=metadata, runtime=copy.deepcopy(self._value["runtime_info"]))

    def cleanup_pending(self, *, updated_at: str) -> "ProcessingSettingsArtifactV5":
        if self.status != "in_progress" or self._value["processing_settings"]["keep_intermediates"]:
            raise SettingsArtifactTransitionError(
                "cleanup pending requires an in-progress pruning job"
            )
        if self._value["metadata"]["final_media_publication_generation"] is None:
            raise SettingsArtifactTransitionError("cleanup pending requires a producer marker")
        utc_ordinal_usec_v1(updated_at)
        metadata = copy.deepcopy(self._value["metadata"])
        metadata["settings_revision"] += 1
        metadata["last_updated_at"] = updated_at
        metadata["cleanup_status"] = "pending"
        return self._replace(metadata=metadata, runtime=copy.deepcopy(self._value["runtime_info"]))

    def completed(
        self,
        *,
        terminal_at: str,
        frames_processed: int,
        cleanup_status: Literal[
            "not_requested", "complete", "incomplete_identity_mismatch", "incomplete_error"
        ],
        metric_clamp_summary: dict[str, Any] | None = None,
    ) -> "ProcessingSettingsArtifactV5":
        if self.status != "in_progress":
            raise SettingsArtifactTransitionError(
                "completed transition requires in-progress settings"
            )
        utc_ordinal_usec_v1(terminal_at)
        metadata = copy.deepcopy(self._value["metadata"])
        metadata["settings_revision"] += 1
        metadata["processing_status"] = "completed"
        metadata["cleanup_status"] = cleanup_status
        metadata["last_updated_at"] = terminal_at
        metadata["terminal_at"] = terminal_at
        duration = attempt_duration_ms_v1(terminal_at, metadata["attempt_started_at"])
        metadata["processing_duration_ms"] = duration
        expected = self._value["output_info"]["expected_final_relative_path"]
        runtime = {
            "final_output_relative_path": expected,
            "frames_processed": frames_processed,
            "processing_time_seconds": duration_seconds_from_ms_v1(duration),
            "metric_clamp_summary": copy.deepcopy(metric_clamp_summary),
            "terminal_diagnostic_code": None,
            "terminal_diagnostic_message": None,
        }
        return self._replace(metadata=metadata, runtime=runtime)

    def failed(
        self,
        *,
        terminal_at: str,
        diagnostic_code: str,
        diagnostic_message: str | None = None,
    ) -> "ProcessingSettingsArtifactV5":
        if self.status != "in_progress":
            raise SettingsArtifactTransitionError("failed transition requires in-progress settings")
        utc_ordinal_usec_v1(terminal_at)
        metadata = copy.deepcopy(self._value["metadata"])
        metadata["settings_revision"] += 1
        metadata["processing_status"] = "failed"
        if metadata["cleanup_status"] == "pending":
            metadata["cleanup_status"] = "incomplete_error"
        metadata["last_updated_at"] = terminal_at
        metadata["terminal_at"] = terminal_at
        metadata["processing_duration_ms"] = attempt_duration_ms_v1(
            terminal_at, metadata["attempt_started_at"]
        )
        runtime = {
            "final_output_relative_path": None,
            "frames_processed": None,
            "processing_time_seconds": None,
            "metric_clamp_summary": None,
            "terminal_diagnostic_code": diagnostic_code,
            "terminal_diagnostic_message": diagnostic_message,
        }
        return self._replace(metadata=metadata, runtime=runtime)

    def restart(
        self,
        *,
        started_at: str,
        processing_settings: dict[str, Any] | None = None,
    ) -> "ProcessingSettingsArtifactV5":
        if self.status not in {"completed", "failed"}:
            raise SettingsArtifactTransitionError("restart requires terminal settings")
        utc_ordinal_usec_v1(started_at)
        materialized = (
            copy.deepcopy(self._value["processing_settings"])
            if processing_settings is None
            else _materialize_processing_settings(processing_settings)
        )
        metadata = copy.deepcopy(self._value["metadata"])
        metadata["settings_revision"] += 1
        metadata["processing_attempt"] += 1
        metadata["attempt_started_at"] = started_at
        metadata["last_updated_at"] = started_at
        metadata["terminal_at"] = None
        metadata["processing_duration_ms"] = None
        metadata["processing_status"] = "in_progress"
        metadata["cleanup_status"] = "not_started"
        runtime = {key: None for key in RUNTIME_INFO_V5_KEYS}
        return self._replace(
            metadata=metadata,
            runtime=runtime,
            processing_settings=materialized,
        )


def create_processing_settings_artifact_v5(
    *,
    batch_name: str,
    source_video: str,
    source_video_name: str,
    source_video_fingerprint: str | None,
    video_properties: dict[str, Any],
    processing_settings: dict[str, Any],
    output_directory: str,
    project_version: str,
    creation_now_usec: str,
    job_id: str | None = None,
    job_reservation_generation: str | None = None,
    job_terminal_reservation_id: str | None = None,
) -> ProcessingSettingsArtifactV5:
    """Build a deterministic initial object before any job mutation."""

    utc_ordinal_usec_v1(creation_now_usec)
    settings = _materialize_processing_settings(processing_settings)
    output_name = job_output_name_v1(
        source_video_name, settings["vr_format"], settings["vr_resolution"]
    )
    metadata: dict[str, Any] = {
        "batch_name": batch_name,
        "job_id": job_id or secrets.token_hex(16),
        "settings_identity_fingerprint": "0" * 64,
        "settings_artifact_relative_path": "processing-settings-v5.json",
        "job_reservation_generation": job_reservation_generation or secrets.token_hex(16),
        "job_terminal_reservation_id": job_terminal_reservation_id or secrets.token_hex(16),
        "settings_schema_version": 5,
        "source_settings_schema_version": 5,
        "settings_revision": 0,
        "processing_attempt": 1,
        "source_video": source_video,
        "source_video_name": source_video_name,
        "source_video_fingerprint_algorithm": FILE_IDENTITY_ALGORITHM_VERSION,
        "source_video_fingerprint": source_video_fingerprint,
        "project_version": project_version,
        "created_at": creation_now_usec,
        "attempt_started_at": creation_now_usec,
        "last_updated_at": None,
        "terminal_at": None,
        "processing_duration_ms": None,
        "processing_status": "in_progress",
        "cleanup_status": "not_started",
        "final_media_producer_contract_version": None,
        "final_media_publication_generation": None,
    }
    value: dict[str, Any] = {
        "content_fingerprint": "0" * 64,
        "metadata": metadata,
        "video_properties": copy.deepcopy(video_properties),
        "processing_settings": settings,
        "output_info": {
            "output_directory": output_directory,
            "expected_output_filename": output_name,
            "expected_final_relative_path": output_name,
            "output_name_algorithm_version": "job-output-name-v1",
        },
        "runtime_info": {key: None for key in RUNTIME_INFO_V5_KEYS},
    }
    return ProcessingSettingsArtifactV5.from_object(_with_fingerprints(value))


def _legacy_epoch_to_utc(value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 253_402_300_799.999999:
        return None
    microseconds = math.floor(normalized * 1_000_000.0)
    try:
        instant = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=microseconds)
    except (OverflowError, ValueError):
        return None
    return instant.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _legacy_video_properties(value: object) -> dict[str, Any]:
    _schema(type(value) is dict, "legacy video_properties must be an object")
    assert isinstance(value, dict)
    width = _integer(value.get("width"), "legacy video width", 1, 10_000)
    height = _integer(value.get("height"), "legacy video height", 1, 10_000)
    fps_value = value.get("fps")
    _schema(
        not isinstance(fps_value, bool)
        and isinstance(fps_value, (int, float))
        and math.isfinite(float(fps_value))
        and float(fps_value) > 0.0,
        "legacy video fps is invalid",
    )
    assert isinstance(fps_value, (int, float)) and not isinstance(fps_value, bool)
    fps = float(fps_value)
    frame_count = _integer(
        value.get("frame_count"),
        "legacy video frame_count",
        1,
        STEREO_CONTROL_FRAME_CAP,
    )
    duration_value = value.get("duration")
    if (
        isinstance(duration_value, bool)
        or not isinstance(duration_value, (int, float))
        or not math.isfinite(float(duration_value))
        or float(duration_value) < 0.0
    ):
        duration = float(np.float64(np.float64(frame_count) / np.float64(fps)))
    else:
        duration = float(duration_value)
    codec_value = value.get("codec")
    codec = codec_value if type(codec_value) is int and 0 <= codec_value <= 4_294_967_295 else 0
    numerator = value.get("sample_aspect_ratio_numerator")
    denominator = value.get("sample_aspect_ratio_denominator")
    if not (
        type(numerator) is int and type(denominator) is int and numerator > 0 and denominator > 0
    ):
        ratio = value.get("sample_aspect_ratio")
        match = (
            re.fullmatch(r"([0-9]+):([0-9]+)", ratio, flags=re.ASCII)
            if isinstance(ratio, str)
            else None
        )
        if match is None or int(match.group(1)) == 0 or int(match.group(2)) == 0:
            numerator, denominator = 1, 1
        else:
            numerator, denominator = (int(part) for part in match.groups())
    divisor = math.gcd(numerator, denominator)
    numerator //= divisor
    denominator //= divisor
    _integer(numerator, "legacy SAR numerator", 1, 2**64 - 1)
    _integer(denominator, "legacy SAR denominator", 1, 2**64 - 1)
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "codec": codec,
        "sample_aspect_ratio_numerator": numerator,
        "sample_aspect_ratio_denominator": denominator,
        "sample_aspect_ratio": f"{numerator}:{denominator}",
    }


def _legacy_metric_summary(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    try:
        affected = value["affected_frame_count"]
        mean = value["mean_clamped_fraction"]
        maximum = value["max_clamped_fraction"]
        if type(affected) is not int or affected < 0:
            return None
        if isinstance(mean, bool) or isinstance(maximum, bool):
            return None
        mean_float = float(mean)
        max_float = float(maximum)
        if not (0.0 <= mean_float <= max_float <= 1.0):
            return None
        raw_hash = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    except (KeyError, TypeError, ValueError, CanonicalJsonError):
        return None
    return {
        "schema_version": 1,
        "affected_frame_count": affected,
        "mean_clamped_fraction": mean_float,
        "max_clamped_fraction": max_float,
        "compatibility_summary_sha256": raw_hash,
    }


def migrate_processing_settings_artifact_v5(
    *,
    legacy_settings: dict[str, Any],
    migrated_processing_settings: dict[str, Any],
    migration_now_usec: str,
    job_id: str,
    job_reservation_generation: str,
    job_terminal_reservation_id: str,
) -> ProcessingSettingsArtifactV5:
    """Construct the sole revision-zero adapter from a schema-1-through-4 object."""

    utc_ordinal_usec_v1(migration_now_usec)
    _schema(type(legacy_settings) is dict, "legacy settings root must be an object")
    legacy_metadata = legacy_settings.get("metadata")
    _schema(type(legacy_metadata) is dict, "legacy metadata must be an object")
    assert isinstance(legacy_metadata, dict)
    source_version = legacy_metadata.get("settings_schema_version", 1)
    _integer(source_version, "legacy settings schema version", 1, 4)
    settings = _materialize_processing_settings(migrated_processing_settings)
    video = _legacy_video_properties(legacy_settings.get("video_properties"))
    legacy_output = legacy_settings.get("output_info")
    _schema(type(legacy_output) is dict, "legacy output_info must be an object")
    assert isinstance(legacy_output, dict)
    expected = validate_contained_legacy_final_component_v1(
        legacy_output.get("expected_output_filename")
    )
    _schema(expected.endswith(".mp4"), "legacy expected output must end in lowercase .mp4")
    output_directory = legacy_output.get("output_directory", settings["output_dir"])
    _scalar_string(output_directory, "legacy output directory")

    status_value = legacy_metadata.get("processing_status", "in_progress")
    _schema(
        status_value in {"in_progress", "paused", "completed", "failed"},
        "legacy processing status is invalid",
    )
    status = "in_progress" if status_value in {"in_progress", "paused"} else status_value
    created_at = _legacy_epoch_to_utc(legacy_metadata.get("created_timestamp"))
    if created_at is None:
        created_at = migration_now_usec
    if status == "in_progress":
        attempt_started_at = migration_now_usec
        last_updated_at = migration_now_usec
        terminal_at = None
        duration_ms = None
        cleanup_status = "not_started"
    else:
        attempt_started_at = created_at
        completed_at = _legacy_epoch_to_utc(legacy_metadata.get("completed_timestamp"))
        terminal_at = completed_at if status == "completed" and completed_at else migration_now_usec
        last_updated_at = terminal_at
        duration_ms = attempt_duration_ms_v1(terminal_at, attempt_started_at)
        if status == "failed":
            cleanup_status = "not_started"
        else:
            cleanup_status = (
                "not_requested" if settings["keep_intermediates"] else "incomplete_error"
            )

    source_video = legacy_metadata.get("source_video", "")
    source_video = _scalar_string(source_video, "legacy source_video")
    source_video_name = legacy_metadata.get("source_video_name")
    if not isinstance(source_video_name, str):
        source_video_name = source_video.replace("\\", "/").rsplit("/", 1)[-1]
    source_video_name = _scalar_string(source_video_name, "legacy source_video_name")
    source_fingerprint = legacy_metadata.get("source_video_fingerprint")
    if source_fingerprint is not None:
        _hex(source_fingerprint, 32, "legacy source fingerprint")
    project_version = legacy_metadata.get("project_version", "0.9.0")
    if not isinstance(project_version, str) or not project_version:
        project_version = "0.9.0"
    metadata: dict[str, Any] = {
        "batch_name": _scalar_string(legacy_metadata.get("batch_name", ""), "legacy batch_name"),
        "job_id": job_id,
        "settings_identity_fingerprint": "0" * 64,
        "settings_artifact_relative_path": "processing-settings-v5.json",
        "job_reservation_generation": job_reservation_generation,
        "job_terminal_reservation_id": job_terminal_reservation_id,
        "settings_schema_version": 5,
        "source_settings_schema_version": source_version,
        "settings_revision": 0,
        "processing_attempt": 1,
        "source_video": source_video,
        "source_video_name": source_video_name,
        "source_video_fingerprint_algorithm": FILE_IDENTITY_ALGORITHM_VERSION,
        "source_video_fingerprint": source_fingerprint,
        "project_version": project_version,
        "created_at": created_at,
        "attempt_started_at": attempt_started_at,
        "last_updated_at": last_updated_at,
        "terminal_at": terminal_at,
        "processing_duration_ms": duration_ms,
        "processing_status": status,
        "cleanup_status": cleanup_status,
        "final_media_producer_contract_version": None,
        "final_media_publication_generation": None,
    }
    runtime: dict[str, Any] = {key: None for key in RUNTIME_INFO_V5_KEYS}
    legacy_runtime = legacy_settings.get("runtime_info")
    if not isinstance(legacy_runtime, dict):
        legacy_runtime = {}
    if status == "completed":
        legacy_frames = legacy_runtime.get("frames_processed")
        frames = (
            legacy_frames
            if type(legacy_frames) is int and 1 <= legacy_frames <= video["frame_count"]
            else video["frame_count"]
        )
        runtime.update(
            {
                "final_output_relative_path": expected,
                "frames_processed": frames,
                "processing_time_seconds": duration_seconds_from_ms_v1(duration_ms),
                "metric_clamp_summary": (
                    _legacy_metric_summary(legacy_runtime.get("metric_clamp_summary"))
                    if settings["stereo_geometry_mode"] == "metric_camera"
                    else None
                ),
            }
        )
    elif status == "failed":
        runtime["terminal_diagnostic_code"] = "LEGACY_FAILED_STATUS"
    value: dict[str, Any] = {
        "content_fingerprint": "0" * 64,
        "metadata": metadata,
        "video_properties": video,
        "processing_settings": settings,
        "output_info": {
            "output_directory": output_directory,
            "expected_output_filename": expected,
            "expected_final_relative_path": expected,
            "output_name_algorithm_version": "persisted-expected-output-filename-v1",
        },
        "runtime_info": runtime,
    }
    return ProcessingSettingsArtifactV5.from_object(_with_fingerprints(value))
