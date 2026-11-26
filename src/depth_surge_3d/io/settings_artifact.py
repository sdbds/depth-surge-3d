"""Schema-5 processing settings artifact, portable output naming, and legacy migration."""

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
from ..core.video_timing import canonical_frame_rate, selected_frame_range
from .canonical_json import (
    CanonicalJsonError,
    canonical_json_bytes,
    strict_canonical_json_loads,
    validate_unicode_scalars,
)


SETTINGS_ARTIFACT_MAX_RAW_BYTES = 512 * 1024
STEREO_CONTROL_FRAME_CAP = 4_194_304

TOP_LEVEL_KEYS = frozenset(
    "content_fingerprint metadata video_properties processing_settings output_info runtime_info".split()
)
METADATA_V5_KEYS = frozenset(
    (
        "batch_name job_id settings_identity_fingerprint settings_artifact_relative_path "
        "job_reservation_generation job_terminal_reservation_id settings_schema_version "
        "source_settings_schema_version settings_revision processing_attempt source_video "
        "source_video_name source_video_fingerprint_algorithm source_video_fingerprint "
        "project_version created_at attempt_started_at last_updated_at terminal_at "
        "processing_duration_ms processing_status cleanup_status "
        "final_media_producer_contract_version final_media_publication_generation"
    ).split()
)
VIDEO_PROPERTIES_V5_KEYS = frozenset(
    (
        "width height fps frame_count duration codec "
        "sample_aspect_ratio_numerator sample_aspect_ratio_denominator sample_aspect_ratio"
    ).split()
)
PROCESSING_SETTINGS_V5_KEYS = frozenset(
    (
        "apply_distortion convergence crop_factor denoising_steps depth_model_version "
        "depth_resolution device direct_vr_encode enable_live_preview end_time "
        "experimental_frame_interpolation fisheye_crop_factor fisheye_fov fisheye_projection "
        "keep_intermediates max_disparity_percent metric_convergence_distance migrate_legacy "
        "min_resolution min_scene_frames model_path model_size occlusion_fill "
        "occlusion_fill_max_px output_dir per_eye_height per_eye_width preserve_audio "
        "preview_update_interval raw_storage_dtype scene_cut_threshold scene_detection seed "
        "source_fps source_height source_width start_time stereo_geometry_mode stereo_io_workers "
        "stereo_render_mode stereo_strength super_sample target_fps temporal_postprocessor "
        "temporal_window_overlap temporal_window_size upscale_model use_metric_depth verbose "
        "video_encoder video_path virtual_baseline_mm vr_format vr_output_height vr_output_width "
        "vr_resolution"
    ).split()
)
OUTPUT_INFO_V5_KEYS = frozenset(
    "output_directory expected_output_filename expected_final_relative_path output_name_algorithm_version".split()
)
RUNTIME_INFO_V5_KEYS = frozenset(
    (
        "final_output_relative_path frames_processed processing_time_seconds "
        "metric_clamp_summary terminal_diagnostic_code terminal_diagnostic_message"
    ).split()
)
METRIC_SUMMARY_KEYS = frozenset(
    (
        "schema_version affected_frame_count mean_clamped_fraction "
        "max_clamped_fraction compatibility_summary_sha256"
    ).split()
)
_OPTIONAL_SETTING_DEFAULTS: dict[str, Any] = {
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
_VALID_CLEANUP_STATUSES = frozenset(
    "not_started not_requested pending complete incomplete_identity_mismatch incomplete_error".split()
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
    {"con", "prn", "aux", "nul"}
    | {f"com{s}" for s in (*range(1, 10), "\u00b9", "\u00b2", "\u00b3")}
    | {f"lpt{s}" for s in (*range(1, 10), "\u00b9", "\u00b2", "\u00b3")}
)


class SettingsArtifactError(ValueError):
    """Base class for a settings artifact contract violation."""


class SettingsArtifactSchemaError(SettingsArtifactError):
    """The object is not a valid schema-5 processing settings artifact."""


class SettingsArtifactStateError(SettingsArtifactError):
    """The object has an inconsistent processing/runtime state combination."""


class SettingsArtifactTransitionError(SettingsArtifactError):
    """A requested state transition is not legal from the current state."""


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
    _schema(type(value) is str, f"{label} must be a string")
    assert isinstance(value, str)
    try:
        validate_unicode_scalars(value)
    except CanonicalJsonError as error:
        raise SettingsArtifactSchemaError(f"{label} contains an isolated surrogate") from error
    _schema(not nonempty or bool(value), f"{label} must be nonempty")
    return value


def _integer(value: object, label: str, low: int, high: int) -> int:
    _schema(type(value) is int and low <= value <= high, f"{label} must be in {low}..{high}")
    assert isinstance(value, int)
    return value


def _binary64(
    value: object, label: str, low: float | None = None, high: float | None = None
) -> float:
    _schema(type(value) is float and math.isfinite(value), f"{label} must be a finite float")
    assert isinstance(value, float)
    _schema(low is None or value >= low, f"{label} is below its minimum")
    _schema(high is None or value <= high, f"{label} is above its maximum")
    return value


def _hex(value: object, digits: int, label: str) -> str:
    text = _scalar_string(value, label)
    _schema(
        len(text) == digits and _HEX.fullmatch(text) is not None, f"{label} must be hex{digits}"
    )
    return text


def utc_ordinal_usec_v1(value: object) -> int:
    """Parse exact Gregorian UTC text and return microseconds since year 1."""

    text = _scalar_string(value, "utc-usec")
    match = _UTC_USEC.fullmatch(text)
    _schema(match is not None and len(text) == 27, "timestamp must be exact utc-usec")
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


def validate_contained_legacy_final_component_v1(value: object) -> str:
    text = _scalar_string(value, "legacy final component", nonempty=True)
    _schema(
        text not in {".", ".."} and "\x00" not in text and "/" not in text and "\\" not in text,
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
        all(
            unicodedata.category(ch) != "Cc" and ch not in _INVALID_FILENAME_SCALARS for ch in text
        ),
        "portable final component contains a forbidden character",
    )
    _schema(
        len(text.encode("utf-8")) <= 240 and len(text.encode("utf-16-le")) // 2 <= 240,
        "portable final component exceeds 240 units",
    )
    _schema(
        text.split(".", 1)[0].lower() not in _WINDOWS_DEVICE_BASES,
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
    """Return the deterministic portable final output filename for a job."""

    source_name = _scalar_string(source_video_name, "source_video_name")
    _schema(
        "/" not in source_name and all(unicodedata.category(ch) != "Cc" for ch in source_name),
        "source_video_name contains forbidden characters",
    )
    if source_name:
        last_dot = source_name.rfind(".")
        stem = source_name[:last_dot] if 0 < last_dot < len(source_name) - 1 else source_name
    else:
        stem = "output"
    sanitized = "".join("_" if ch in _INVALID_FILENAME_SCALARS else ch for ch in stem)
    sanitized = re.sub(r"_+", "_", sanitized, flags=re.ASCII).strip("_")
    format_value = _scalar_string(vr_format, "vr_format")
    _schema(format_value in {"side_by_side", "over_under"}, "vr_format is invalid")
    format_token = "side-by-side" if format_value == "side_by_side" else "over-under"
    tail = f"_3D_{format_token}_{_resolution_filename_token(vr_resolution)}.mp4"
    candidate = f"{sanitized}{tail}"
    try:
        return validate_portable_final_component_v1(candidate)
    except SettingsArtifactSchemaError:
        pass
    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:32]
    fallback_tail = f"~{digest}{tail}"
    for prefix_length in range(len(sanitized), -1, -1):
        try:
            return validate_portable_final_component_v1(
                f"{sanitized[:prefix_length]}{fallback_tail}"
            )
        except SettingsArtifactSchemaError:
            continue
    raise SettingsArtifactSchemaError("job-output-name-v1 has no portable fallback")


def materialize_processing_settings_v5(values: dict[str, Any]) -> dict[str, Any]:
    """Return the normalized settings mapping persisted in a schema-5 artifact."""

    _schema(type(values) is dict, "processing_settings must be an object")
    nullable_keys = frozenset(_OPTIONAL_SETTING_DEFAULTS)
    validation_input = {
        key: item for key, item in values.items() if item is not None or key not in nullable_keys
    }
    validated = {
        **_OPTIONAL_SETTING_DEFAULTS,
        **validate_settings(validation_input, source="explicit"),
    }
    for key in nullable_keys:
        if key in values and values[key] is None:
            validated[key] = None
    for int_or_str_key in ("target_fps", "depth_resolution"):
        raw_val = validated[int_or_str_key]
        if isinstance(raw_val, str) and raw_val.isdigit():
            validated[int_or_str_key] = int(raw_val)
    return validated


def _selected_frame_count_v1(
    video_properties: dict[str, Any], processing_settings: dict[str, Any]
) -> int:
    start_frame, end_frame = selected_frame_range(video_properties, processing_settings)
    return end_frame - start_frame


def _validate_video_properties(value: object) -> dict[str, Any]:
    keys = VIDEO_PROPERTIES_V5_KEYS
    if isinstance(value, dict) and "frame_rate" in value:
        keys = keys | {"frame_rate"}
        rate = value["frame_rate"]
        _schema(
            isinstance(rate, str) and canonical_frame_rate(rate) == rate,
            "video_properties.frame_rate must be canonical",
        )
    video = _closed_object(value, keys, "video_properties")
    _integer(video["width"], "video_properties.width", 1, 10_000)
    _integer(video["height"], "video_properties.height", 1, 10_000)
    _binary64(video["fps"], "video_properties.fps", float(np.nextafter(0.0, 1.0)))
    _integer(video["frame_count"], "video_properties.frame_count", 1, STEREO_CONTROL_FRAME_CAP)
    _binary64(video["duration"], "video_properties.duration", 0.0)
    _integer(video["codec"], "video_properties.codec", 0, 4_294_967_295)
    num = _integer(video["sample_aspect_ratio_numerator"], "SAR numerator", 1, 2**64 - 1)
    den = _integer(video["sample_aspect_ratio_denominator"], "SAR denominator", 1, 2**64 - 1)
    _schema(
        math.gcd(num, den) == 1 and video["sample_aspect_ratio"] == f"{num}:{den}", "invalid SAR"
    )
    return video


def _validate_metric_summary(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    summary = _closed_object(value, METRIC_SUMMARY_KEYS, "runtime_info.metric_clamp_summary")
    _schema(
        summary["schema_version"] == 1 and type(summary["schema_version"]) is int,
        "bad metric schema",
    )
    _integer(summary["affected_frame_count"], "affected_frame_count", 0, 2**64 - 1)
    mean = _binary64(summary["mean_clamped_fraction"], "mean_clamped_fraction", 0.0, 1.0)
    maximum = _binary64(summary["max_clamped_fraction"], "max_clamped_fraction", 0.0, 1.0)
    _schema(mean <= maximum or summary["affected_frame_count"] == 0, "metric mean exceeds max")
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


def _validate_runtime_state(
    metadata: dict[str, Any],
    video: dict[str, Any],
    settings: dict[str, Any],
    runtime: dict[str, Any],
    expected: str,
    p_gen: str | None,
) -> None:
    metric_summary = _validate_metric_summary(runtime["metric_clamp_summary"])
    code = runtime["terminal_diagnostic_code"]
    if code is not None:
        _schema(
            type(code) is str and _DIAGNOSTIC_CODE.fullmatch(code) is not None,
            "invalid diagnostic code",
        )
    msg = runtime["terminal_diagnostic_message"]
    if msg is not None:
        _schema(
            len(_scalar_string(msg, "terminal_diagnostic_message").encode("utf-8")) <= 1024,
            "diagnostic too long",
        )
    success_fields = (
        "final_output_relative_path",
        "frames_processed",
        "processing_time_seconds",
        "metric_clamp_summary",
    )
    status = metadata["processing_status"]
    if status == "in_progress":
        _state(
            metadata["terminal_at"] is None and metadata["processing_duration_ms"] is None,
            "in-progress terminal fields must be null",
        )
        _state(
            all(runtime[k] is None for k in success_fields) and code is None and msg is None,
            "in-progress runtime must be null",
        )
        if metadata["cleanup_status"] == "pending":
            _state(
                not settings["keep_intermediates"] and p_gen is not None,
                "cleanup pending requires publication",
            )
        return
    terminal = metadata["terminal_at"]
    _state(
        terminal is not None and metadata["last_updated_at"] == terminal,
        "terminal timestamps must match",
    )
    duration = attempt_duration_ms_v1(terminal, metadata["attempt_started_at"])
    _state(metadata["processing_duration_ms"] == duration, "terminal duration mismatch")
    if status != "completed":
        _state(code is not None and metadata["cleanup_status"] != "pending", "failed state invalid")
        _state(
            all(runtime[k] is None for k in success_fields),
            "failed runtime success fields must be null",
        )
        return
    _state(
        code is None and msg is None and runtime["final_output_relative_path"] == expected,
        "completed runtime invalid",
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
            "legacy completed frames invalid",
        )
    else:
        _state(
            runtime["frames_processed"] == _selected_frame_count_v1(video, settings),
            "completed frames mismatch",
        )
    if settings["stereo_geometry_mode"] == "metric_camera" and not untouched_legacy:
        _state(metric_summary is not None, "completed metric jobs require a clamp summary")
    if settings["stereo_geometry_mode"] != "metric_camera":
        _state(metric_summary is None, "relative jobs cannot carry a metric clamp summary")


def _validate_complete_object(value: object) -> dict[str, Any]:
    root = _closed_object(value, TOP_LEVEL_KEYS, "ProcessingSettingsArtifactV5")
    _hex(root["content_fingerprint"], 64, "content_fingerprint")
    _schema(
        root["content_fingerprint"] == _fingerprint_without(root, "content_fingerprint"),
        "content fingerprint mismatch",
    )
    metadata = _closed_object(root["metadata"], METADATA_V5_KEYS, "metadata")
    for key in ("job_id", "job_reservation_generation", "job_terminal_reservation_id"):
        _hex(metadata[key], 32, f"metadata.{key}")
    _hex(metadata["settings_identity_fingerprint"], 64, "metadata.settings_identity_fingerprint")
    _schema(
        metadata["settings_artifact_relative_path"] == "processing-settings-v5.json"
        and metadata["settings_schema_version"] == 5
        and type(metadata["settings_schema_version"]) is int,
        "settings schema metadata is invalid",
    )
    _integer(metadata["source_settings_schema_version"], "source_settings_schema_version", 1, 5)
    _integer(metadata["settings_revision"], "settings_revision", 0, 2**64 - 1)
    _integer(metadata["processing_attempt"], "processing_attempt", 1, 2**64 - 1)
    _schema(
        metadata["source_video_fingerprint_algorithm"] == FILE_IDENTITY_ALGORITHM_VERSION,
        "source fingerprint algorithm is invalid",
    )
    if metadata["source_video_fingerprint"] is not None:
        _hex(metadata["source_video_fingerprint"], 32, "metadata.source_video_fingerprint")
    _state(
        utc_ordinal_usec_v1(metadata["attempt_started_at"])
        >= utc_ordinal_usec_v1(metadata["created_at"]),
        "attempt_started_at precedes created_at",
    )
    for ts_key in ("last_updated_at", "terminal_at"):
        if metadata[ts_key] is not None:
            utc_ordinal_usec_v1(metadata[ts_key])
    _schema(
        metadata["processing_status"] in {"in_progress", "completed", "failed"},
        "processing status is invalid",
    )
    _schema(metadata["cleanup_status"] in _VALID_CLEANUP_STATUSES, "cleanup status is invalid")
    p_ver = metadata["final_media_producer_contract_version"]
    p_gen = metadata["final_media_publication_generation"]
    _state(
        (p_ver is None and p_gen is None)
        or (p_ver == 4 and type(p_ver) is int and isinstance(p_gen, str)),
        "producer marker is invalid",
    )
    if p_gen is not None:
        _hex(p_gen, 32, "metadata.final_media_publication_generation")

    video = _validate_video_properties(root["video_properties"])
    settings = _closed_object(
        root["processing_settings"], PROCESSING_SETTINGS_V5_KEYS, "processing_settings"
    )
    _resolution_filename_token(settings["vr_resolution"])
    _state(
        settings["stereo_geometry_mode"] != "metric_camera"
        or settings["temporal_postprocessor"] == "off",
        "metric geometry cannot use VDPP",
    )
    output = _closed_object(root["output_info"], OUTPUT_INFO_V5_KEYS, "output_info")
    expected = _scalar_string(
        output["expected_final_relative_path"], "expected_final_relative_path", nonempty=True
    )
    _schema(
        output["expected_output_filename"] == expected and expected.endswith(".mp4"),
        "invalid output filename",
    )
    algorithm = output["output_name_algorithm_version"]
    _schema(
        algorithm in {"job-output-name-v1", "persisted-expected-output-filename-v1"},
        "output name algorithm is invalid",
    )
    if algorithm == "job-output-name-v1":
        validate_portable_final_component_v1(expected)
    else:
        validate_contained_legacy_final_component_v1(expected)
    identity = hashlib.sha256(
        canonical_json_bytes(_settings_identity_object(metadata, output))
    ).hexdigest()
    _schema(metadata["settings_identity_fingerprint"] == identity, "settings identity mismatch")
    runtime = _closed_object(root["runtime_info"], RUNTIME_INFO_V5_KEYS, "runtime_info")
    _validate_runtime_state(metadata, video, settings, runtime, expected, p_gen)
    return root


def _with_fingerprints(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    metadata = result["metadata"]
    metadata["settings_identity_fingerprint"] = hashlib.sha256(
        canonical_json_bytes(_settings_identity_object(metadata, result["output_info"]))
    ).hexdigest()
    result["content_fingerprint"] = _fingerprint_without(result, "content_fingerprint")
    return result


def _build_initial_metadata(
    *,
    batch_name: str,
    job_id: str,
    reservation_generation: str,
    terminal_reservation_id: str,
    source_schema_version: int,
    source_video: str,
    source_video_name: str,
    source_video_fingerprint: str | None,
    project_version: str,
    created_at: str,
    attempt_started_at: str,
    last_updated_at: str | None,
    terminal_at: str | None,
    duration_ms: int | None,
    status: str,
    cleanup_status: str,
) -> dict[str, Any]:
    return {
        "batch_name": batch_name,
        "job_id": job_id,
        "settings_identity_fingerprint": "0" * 64,
        "settings_artifact_relative_path": "processing-settings-v5.json",
        "job_reservation_generation": reservation_generation,
        "job_terminal_reservation_id": terminal_reservation_id,
        "settings_schema_version": 5,
        "source_settings_schema_version": source_schema_version,
        "settings_revision": 0,
        "processing_attempt": 1,
        "source_video": source_video,
        "source_video_name": source_video_name,
        "source_video_fingerprint_algorithm": FILE_IDENTITY_ALGORITHM_VERSION,
        "source_video_fingerprint": source_video_fingerprint,
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


@dataclass(frozen=True)
class ProcessingSettingsArtifactV5:
    """Immutable validated schema-5 settings artifact."""

    _value: dict[str, Any]

    @classmethod
    def from_object(cls, value: object) -> "ProcessingSettingsArtifactV5":
        return cls(_validate_complete_object(copy.deepcopy(value)))

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

    @property
    def selected_frame_count(self) -> int:
        return _selected_frame_count_v1(
            self._value["video_properties"], self._value["processing_settings"]
        )

    def _transition(
        self,
        *,
        updated_at: str,
        meta_updates: dict[str, Any] | None = None,
        runtime: dict[str, Any] | None = None,
        processing_settings: dict[str, Any] | None = None,
    ) -> "ProcessingSettingsArtifactV5":
        utc_ordinal_usec_v1(updated_at)
        value = self.to_object()
        metadata = value["metadata"]
        metadata["settings_revision"] += 1
        metadata["last_updated_at"] = updated_at
        if meta_updates:
            metadata.update(meta_updates)
        if runtime is not None:
            value["runtime_info"] = runtime
        if processing_settings is not None:
            value["processing_settings"] = processing_settings
        return ProcessingSettingsArtifactV5.from_object(_with_fingerprints(value))

    def with_processing_settings(
        self, processing_settings: dict[str, Any], *, updated_at: str
    ) -> "ProcessingSettingsArtifactV5":
        if self.status != "in_progress":
            raise SettingsArtifactTransitionError("settings rewrite requires an in-progress job")
        return self._transition(
            updated_at=updated_at,
            processing_settings=materialize_processing_settings_v5(processing_settings),
        )

    def with_producer_marker(
        self, *, generation: str, updated_at: str
    ) -> "ProcessingSettingsArtifactV5":
        if self.status != "in_progress":
            raise SettingsArtifactTransitionError("producer marker requires an in-progress job")
        _hex(generation, 32, "producer generation")
        current = self._value["metadata"]["final_media_publication_generation"]
        if current is not None:
            if current != generation:
                raise SettingsArtifactTransitionError("producer marker is immutable")
            return self
        return self._transition(
            updated_at=updated_at,
            meta_updates={
                "final_media_producer_contract_version": 4,
                "final_media_publication_generation": generation,
            },
        )

    def cleanup_pending(self, *, updated_at: str) -> "ProcessingSettingsArtifactV5":
        if self.status != "in_progress" or self._value["processing_settings"]["keep_intermediates"]:
            raise SettingsArtifactTransitionError(
                "cleanup pending requires an in-progress pruning job"
            )
        if self._value["metadata"]["final_media_publication_generation"] is None:
            raise SettingsArtifactTransitionError("cleanup pending requires a producer marker")
        return self._transition(updated_at=updated_at, meta_updates={"cleanup_status": "pending"})

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
        duration = attempt_duration_ms_v1(
            terminal_at, self._value["metadata"]["attempt_started_at"]
        )
        return self._transition(
            updated_at=terminal_at,
            meta_updates={
                "processing_status": "completed",
                "cleanup_status": cleanup_status,
                "terminal_at": terminal_at,
                "processing_duration_ms": duration,
            },
            runtime={
                "final_output_relative_path": self._value["output_info"][
                    "expected_final_relative_path"
                ],
                "frames_processed": frames_processed,
                "processing_time_seconds": duration_seconds_from_ms_v1(duration),
                "metric_clamp_summary": copy.deepcopy(metric_clamp_summary),
                "terminal_diagnostic_code": None,
                "terminal_diagnostic_message": None,
            },
        )

    def failed(
        self,
        *,
        terminal_at: str,
        diagnostic_code: str,
        diagnostic_message: str | None = None,
    ) -> "ProcessingSettingsArtifactV5":
        if self.status != "in_progress":
            raise SettingsArtifactTransitionError("failed transition requires in-progress settings")
        cleanup = self._value["metadata"]["cleanup_status"]
        return self._transition(
            updated_at=terminal_at,
            meta_updates={
                "processing_status": "failed",
                "cleanup_status": "incomplete_error" if cleanup == "pending" else cleanup,
                "terminal_at": terminal_at,
                "processing_duration_ms": attempt_duration_ms_v1(
                    terminal_at, self._value["metadata"]["attempt_started_at"]
                ),
            },
            runtime={
                "final_output_relative_path": None,
                "frames_processed": None,
                "processing_time_seconds": None,
                "metric_clamp_summary": None,
                "terminal_diagnostic_code": diagnostic_code,
                "terminal_diagnostic_message": diagnostic_message,
            },
        )

    def restart(
        self, *, started_at: str, processing_settings: dict[str, Any] | None = None
    ) -> "ProcessingSettingsArtifactV5":
        if self.status not in {"completed", "failed"}:
            raise SettingsArtifactTransitionError("restart requires terminal settings")
        materialized = (
            copy.deepcopy(self._value["processing_settings"])
            if processing_settings is None
            else materialize_processing_settings_v5(processing_settings)
        )
        return self._transition(
            updated_at=started_at,
            meta_updates={
                "processing_attempt": self._value["metadata"]["processing_attempt"] + 1,
                "attempt_started_at": started_at,
                "terminal_at": None,
                "processing_duration_ms": None,
                "processing_status": "in_progress",
                "cleanup_status": "not_started",
            },
            runtime={key: None for key in RUNTIME_INFO_V5_KEYS},
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
    """Build the initial schema-5 artifact for a new job."""

    utc_ordinal_usec_v1(creation_now_usec)
    settings = materialize_processing_settings_v5(processing_settings)
    output_name = job_output_name_v1(
        source_video_name, settings["vr_format"], settings["vr_resolution"]
    )
    value: dict[str, Any] = {
        "content_fingerprint": "0" * 64,
        "metadata": _build_initial_metadata(
            batch_name=batch_name,
            job_id=job_id or secrets.token_hex(16),
            reservation_generation=job_reservation_generation or secrets.token_hex(16),
            terminal_reservation_id=job_terminal_reservation_id or secrets.token_hex(16),
            source_schema_version=5,
            source_video=source_video,
            source_video_name=source_video_name,
            source_video_fingerprint=source_video_fingerprint,
            project_version=project_version,
            created_at=creation_now_usec,
            attempt_started_at=creation_now_usec,
            last_updated_at=None,
            terminal_at=None,
            duration_ms=None,
            status="in_progress",
            cleanup_status="not_started",
        ),
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
    try:
        instant = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(
            microseconds=math.floor(normalized * 1_000_000.0)
        )
    except (OverflowError, ValueError):
        return None
    return instant.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _legacy_video_properties(value: object) -> dict[str, Any]:
    _schema(type(value) is dict, "legacy video_properties must be an object")
    assert isinstance(value, dict)
    width = _integer(value.get("width"), "legacy video width", 1, 10_000)
    height = _integer(value.get("height"), "legacy video height", 1, 10_000)
    fps_raw = value.get("fps")
    _schema(
        not isinstance(fps_raw, bool)
        and isinstance(fps_raw, (int, float))
        and math.isfinite(float(fps_raw))
        and float(fps_raw) > 0.0,
        "legacy video fps is invalid",
    )
    assert isinstance(fps_raw, (int, float))
    fps = float(fps_raw)
    frame_count = _integer(
        value.get("frame_count"), "legacy frame_count", 1, STEREO_CONTROL_FRAME_CAP
    )
    dur_raw = value.get("duration")
    if (
        isinstance(dur_raw, bool)
        or not isinstance(dur_raw, (int, float))
        or not math.isfinite(float(dur_raw))
        or float(dur_raw) < 0.0
    ):
        duration = float(np.float64(np.float64(frame_count) / np.float64(fps)))
    else:
        duration = float(dur_raw)
    codec_raw = value.get("codec")
    codec = codec_raw if type(codec_raw) is int and 0 <= codec_raw <= 4_294_967_295 else 0
    num = value.get("sample_aspect_ratio_numerator")
    den = value.get("sample_aspect_ratio_denominator")
    if not (type(num) is int and type(den) is int and num > 0 and den > 0):
        ratio = value.get("sample_aspect_ratio")
        match = (
            re.fullmatch(r"([0-9]+):([0-9]+)", ratio, flags=re.ASCII)
            if isinstance(ratio, str)
            else None
        )
        num, den = (
            (int(match.group(1)), int(match.group(2)))
            if match and int(match.group(1)) > 0 and int(match.group(2)) > 0
            else (1, 1)
        )
    divisor = math.gcd(num, den)
    num, den = num // divisor, den // divisor
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "codec": codec,
        "sample_aspect_ratio_numerator": num,
        "sample_aspect_ratio_denominator": den,
        "sample_aspect_ratio": f"{num}:{den}",
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
    """Adapt a schema-1-through-4 settings file into schema-5 revision 0."""

    utc_ordinal_usec_v1(migration_now_usec)
    _schema(type(legacy_settings) is dict, "legacy settings root must be an object")
    legacy_meta = legacy_settings.get("metadata")
    legacy_out = legacy_settings.get("output_info")
    _schema(
        type(legacy_meta) is dict and type(legacy_out) is dict,
        "legacy metadata/output_info invalid",
    )
    assert isinstance(legacy_meta, dict) and isinstance(legacy_out, dict)
    source_version = _integer(legacy_meta.get("settings_schema_version", 1), "legacy schema", 1, 4)
    settings = materialize_processing_settings_v5(migrated_processing_settings)
    video = _legacy_video_properties(legacy_settings.get("video_properties"))
    expected = validate_contained_legacy_final_component_v1(
        legacy_out.get("expected_output_filename")
    )
    _schema(expected.endswith(".mp4"), "legacy expected output must end in lowercase .mp4")
    output_dir = _scalar_string(
        legacy_out.get("output_directory", settings["output_dir"]), "output_dir"
    )
    status_raw = legacy_meta.get("processing_status", "in_progress")
    _schema(status_raw in {"in_progress", "paused", "completed", "failed"}, "legacy status invalid")
    status = "in_progress" if status_raw in {"in_progress", "paused"} else status_raw
    created_at = _legacy_epoch_to_utc(legacy_meta.get("created_timestamp")) or migration_now_usec
    if status == "in_progress":
        attempt_started_at = last_updated_at = migration_now_usec
        terminal_at = duration_ms = None
        cleanup_status = "not_started"
    else:
        attempt_started_at = created_at
        completed_at = _legacy_epoch_to_utc(legacy_meta.get("completed_timestamp"))
        terminal_at = completed_at if status == "completed" and completed_at else migration_now_usec
        last_updated_at = terminal_at
        duration_ms = attempt_duration_ms_v1(terminal_at, attempt_started_at)
        cleanup_status = (
            "not_started"
            if status == "failed"
            else ("not_requested" if settings["keep_intermediates"] else "incomplete_error")
        )
    source_video = _scalar_string(legacy_meta.get("source_video", ""), "legacy source_video")
    src_name = legacy_meta.get("source_video_name")
    if not isinstance(src_name, str):
        src_name = source_video.replace("\\", "/").rsplit("/", 1)[-1]
    src_fp = legacy_meta.get("source_video_fingerprint")
    if src_fp is not None:
        _hex(src_fp, 32, "legacy source fingerprint")
    proj_ver = legacy_meta.get("project_version")
    project_version = proj_ver if isinstance(proj_ver, str) and proj_ver else "0.9.0"

    runtime: dict[str, Any] = {key: None for key in RUNTIME_INFO_V5_KEYS}
    legacy_rt_raw = legacy_settings.get("runtime_info")
    legacy_rt: dict[str, Any] = legacy_rt_raw if isinstance(legacy_rt_raw, dict) else {}
    if status == "completed":
        lf = legacy_rt.get("frames_processed")
        frames = lf if type(lf) is int and 1 <= lf <= video["frame_count"] else video["frame_count"]
        runtime.update(
            {
                "final_output_relative_path": expected,
                "frames_processed": frames,
                "processing_time_seconds": duration_seconds_from_ms_v1(duration_ms),
            }
        )
    elif status == "failed":
        runtime["terminal_diagnostic_code"] = "LEGACY_FAILED_STATUS"

    value: dict[str, Any] = {
        "content_fingerprint": "0" * 64,
        "metadata": _build_initial_metadata(
            batch_name=_scalar_string(legacy_meta.get("batch_name", ""), "legacy batch_name"),
            job_id=job_id,
            reservation_generation=job_reservation_generation,
            terminal_reservation_id=job_terminal_reservation_id,
            source_schema_version=source_version,
            source_video=source_video,
            source_video_name=_scalar_string(src_name, "legacy source_video_name"),
            source_video_fingerprint=src_fp,
            project_version=project_version,
            created_at=created_at,
            attempt_started_at=attempt_started_at,
            last_updated_at=last_updated_at,
            terminal_at=terminal_at,
            duration_ms=duration_ms,
            status=status,
            cleanup_status=cleanup_status,
        ),
        "video_properties": video,
        "processing_settings": settings,
        "output_info": {
            "output_directory": output_dir,
            "expected_output_filename": expected,
            "expected_final_relative_path": expected,
            "output_name_algorithm_version": "persisted-expected-output-filename-v1",
        },
        "runtime_info": runtime,
    }
    return ProcessingSettingsArtifactV5.from_object(_with_fingerprints(value))
