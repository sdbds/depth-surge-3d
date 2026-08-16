"""Deterministic stage validation and one-way legacy resume migration."""

from __future__ import annotations

import json
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


from ..core.settings import (
    PROCESSING_SETTINGS_SCHEMA_VERSION,
    REMOVED_SETTING_NAMES,
    validate_settings,
)
from ..core.file_identity import (
    FILE_IDENTITY_ALGORITHM_VERSION,
    file_sample_fingerprint,
)
from ..core.depth_contract import (
    CANONICAL_DEPTH_ALGORITHM_VERSION,
    CANONICAL_DEPTH_SCHEMA_VERSION,
    canonical_json_hash,
)
from ..inference.depth.v2_temporal_contract import (
    VDA_INFERENCE_ALGORITHM,
    build_v2_execution_plan,
    is_compatible_v2_execution_plan,
)
from ..processing.frames.depth_processor import (
    DEPTH_BOUNDS_SCHEMA_VERSION,
)
from ..processing.frames.depth_resolution import resolve_depth_input_size
from ..processing.frames.depth_storage import (
    RAW_DEPTH_SCHEMA_VERSION,
    RawDepthFingerprintError,
    RawDepthStore,
    depth_preprocessing_algorithm,
    select_depth_model_settings,
)
from ..processing.frames.scene_analyzer import (
    SCENE_ALGORITHM_VERSION,
    SCENE_SCHEMA_VERSION,
)
from ..processing.frames.source_frame_manifest import (
    read_source_frame_manifest,
    source_frame_manifest_mismatch_reason,
)
from ..processing.frames.stage_manifest import FRAME_STAGE_SCHEMA_VERSION
from ..utils.imaging.png_header import png_header_matches, read_png_header
from ..processing.frames.stereo_generator import (
    STEREO_STAGE_ALGORITHM_VERSION,
    STEREO_STAGE_SCHEMA_VERSION,
)
from ..utils.path_utils import calculate_frame_range


Disposition = Literal["preserve", "resume", "invalidate", "missing"]
MigrationMode = Literal["archive", "delete"]

_STEREO_SETTING_KEYS = ("stereo_strength", "convergence", "occlusion_fill")
_DISTORTION_SETTING_KEYS = (
    "apply_distortion",
    "fisheye_projection",
    "fisheye_fov",
)
_CROP_SETTING_KEYS = ("crop_factor", "fisheye_crop_factor")
_UPSCALE_SETTING_KEYS = ("upscale_model",)
_VR_SETTING_KEYS = ("vr_format", "vr_resolution", "target_fps")
_DEPTH_MODEL_VERSIONS = frozenset({"v2", "v3", "see_through"})
_SEE_THROUGH_MARKERS = ("see_through", "seethrough")


@dataclass(frozen=True)
class ResumeStage:
    """One stage's validation result."""

    name: str
    paths: tuple[Path, ...]
    disposition: Disposition
    reason: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "paths": [str(path) for path in self.paths],
            "disposition": self.disposition,
            "reason": self.reason,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class ResumeReport:
    """Read-only resume decision plus the data required for migration."""

    output_dir: Path
    stages: tuple[ResumeStage, ...]
    removed_settings: tuple[str, ...]
    settings_file: Path | None
    original_settings_data: dict[str, Any] | None
    migrated_settings: dict[str, Any]
    settings_backup_required: bool

    def stage(self, name: str) -> ResumeStage:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(name)

    @property
    def preserved_stage_names(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages if stage.disposition == "preserve")

    @property
    def invalidated_stage_names(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages if stage.disposition == "invalidate")

    @property
    def migration_paths(self) -> tuple[Path, ...]:
        paths: list[Path] = []
        seen: set[Path] = set()
        protected = {(self.output_dir / "00_original_frames").resolve()}
        for stage in self.stages:
            if stage.disposition != "invalidate":
                continue
            for path in stage.paths:
                resolved = path.resolve()
                if resolved not in protected and resolved not in seen and _has_payload(path):
                    paths.append(path)
                    seen.add(resolved)
        return tuple(paths)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "stages": [stage.to_dict() for stage in self.stages],
            "removed_settings": list(self.removed_settings),
            "preserved_stages": list(self.preserved_stage_names),
            "invalidated_stages": list(self.invalidated_stage_names),
        }


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _has_payload(path: Path) -> bool:
    if path.is_file():
        return True
    return path.is_dir() and next(path.iterdir(), None) is not None


def _stage(
    name: str,
    paths: tuple[Path, ...],
    disposition: Disposition,
    reason: str,
) -> ResumeStage:
    return ResumeStage(
        name=name,
        paths=paths,
        disposition=disposition,
        reason=reason,
        size_bytes=sum(_directory_size(path) for path in paths),
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _contains_see_through_marker(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.casefold().replace("-", "_").replace(" ", "_")
    return any(marker in normalized for marker in _SEE_THROUGH_MARKERS)


def _raw_depth_uses_see_through(output_dir: Path) -> bool:
    metadata = _read_json(output_dir / "02_depth_raw" / "metadata.json")
    semantic = metadata.get("semantic_fingerprint") if metadata else None
    if not isinstance(semantic, dict):
        return False

    values: list[object] = [semantic.get("backend"), semantic.get("source")]
    for section_name in ("model_info", "depth_settings"):
        section = semantic.get(section_name)
        if isinstance(section, dict):
            values.extend(
                section.get(key)
                for key in (
                    "model_name",
                    "model_version",
                    "model_path",
                    "source",
                    "depth_model_version",
                )
            )
    return any(_contains_see_through_marker(value) for value in values)


def resolve_resume_depth_model_version(
    settings: dict[str, Any],
    output_dir: Path | str | None = None,
    *,
    default: str,
) -> str:
    """Recover the depth backend for a job whose legacy settings lack it."""
    if default not in _DEPTH_MODEL_VERSIONS:
        raise ValueError(f"Unsupported resume depth model default: {default}")

    explicit = settings.get("depth_model_version")
    if explicit in _DEPTH_MODEL_VERSIONS:
        return explicit

    if output_dir is not None and _raw_depth_uses_see_through(Path(output_dir)):
        return "see_through"
    if _contains_see_through_marker(settings.get("model_path")):
        return "see_through"
    return default


def _find_settings_file(output_dir: Path) -> Path | None:
    candidates = sorted(
        output_dir.glob("*-settings.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    return candidates[0] if candidates else None


def _resolve_settings_file(output_dir: Path, settings_file: Path | str | None) -> Path | None:
    if settings_file is None:
        return _find_settings_file(output_dir)
    candidate = Path(settings_file).resolve()
    if (
        candidate.parent != output_dir
        or not candidate.name.endswith("-settings.json")
        or not candidate.is_file()
    ):
        raise ValueError("Resume settings file must be a job settings file in the output directory")
    return candidate


def _expected_frame_count(
    settings_data: dict[str, Any] | None,
    saved_settings: dict[str, Any],
) -> int | None:
    properties = (settings_data or {}).get("video_properties", {})
    if not isinstance(properties, dict):
        return None
    try:
        total = int(properties["frame_count"])
        fps = float(properties["fps"])
    except (KeyError, TypeError, ValueError):
        return None
    start, end = calculate_frame_range(
        total,
        fps,
        saved_settings.get("start_time"),
        saved_settings.get("end_time"),
    )
    return end - start


def _source_video_mismatch_reason(
    settings_data: dict[str, Any] | None,
    source_video: Path | str | None,
) -> str | None:
    metadata = (settings_data or {}).get("metadata", {})
    saved_source = metadata.get("source_video") if isinstance(metadata, dict) else None
    source_value = source_video if source_video is not None else saved_source
    if not isinstance(source_value, (str, Path)):
        return "source video path is missing"
    source_path = Path(source_value)
    if not source_path.is_file():
        return "source video path is missing"
    saved_fingerprint = _current_source_fingerprint(metadata)
    if saved_fingerprint is not None and file_sample_fingerprint(source_path) != saved_fingerprint:
        return "source video fingerprint mismatch"
    return None


def _expected_frame_shape(settings_data: dict[str, Any] | None) -> tuple[int, int] | None:
    properties = (settings_data or {}).get("video_properties", {})
    if not isinstance(properties, dict):
        return None
    width = properties.get("width")
    height = properties.get("height")
    if isinstance(width, int) and isinstance(height, int):
        return height, width
    return None


def _frame_payload_mismatch_reason(
    frame_files: list[Path], expected_shape: tuple[int, int] | None
) -> str | None:
    for frame_file in frame_files:
        header = read_png_header(frame_file)
        if header is None:
            return f"invalid PNG header: {frame_file.name}"
        if expected_shape is not None and (header.height, header.width) != expected_shape:
            return (
                f"frame dimensions mismatch: {frame_file.name} is "
                f"{header.width}x{header.height}"
            )
    return None


def _validate_frame_stage(
    output_dir: Path,
    settings_data: dict[str, Any] | None,
    saved_settings: dict[str, Any],
) -> tuple[ResumeStage, list[Path], str | None]:
    frames_dir = output_dir / "00_original_frames"
    paths = (frames_dir,)
    frame_files = sorted(frames_dir.glob("frame_*.png")) if frames_dir.is_dir() else []
    if not frame_files:
        return _stage("frames", paths, "missing", "no extracted source frames"), [], None

    expected_count = _expected_frame_count(settings_data, saved_settings)
    if expected_count is not None and len(frame_files) != expected_count:
        reason = f"frame count mismatch: found {len(frame_files)}, expected {expected_count}"
        return _stage("frames", paths, "invalidate", reason), frame_files, None

    manifest_count = expected_count if expected_count is not None else len(frame_files)
    expected_names = [f"frame_{index:06d}.png" for index in range(1, manifest_count + 1)]
    if [path.name for path in frame_files] != expected_names:
        reason = "source frame manifest is not contiguous"
        return _stage("frames", paths, "invalidate", reason), frame_files, None

    payload_reason = _frame_payload_mismatch_reason(
        frame_files, _expected_frame_shape(settings_data)
    )
    if payload_reason is not None:
        return _stage("frames", paths, "invalidate", payload_reason), frame_files, None

    settings_metadata = (settings_data or {}).get("metadata", {})
    source_video_fingerprint = (
        settings_metadata.get("source_video_fingerprint")
        if isinstance(settings_metadata, dict)
        else None
    )
    if not isinstance(source_video_fingerprint, str):
        return (
            _stage("frames", paths, "invalidate", "source video fingerprint is missing"),
            frame_files,
            None,
        )
    source_frame_metadata = read_source_frame_manifest(frames_dir)
    manifest_reason = source_frame_manifest_mismatch_reason(
        source_frame_metadata,
        frame_files,
        source_video_fingerprint,
    )
    if manifest_reason is not None:
        return _stage("frames", paths, "invalidate", manifest_reason), frame_files, None

    if not isinstance(source_frame_metadata, dict):
        return (
            _stage("frames", paths, "invalidate", "source frame manifest is malformed"),
            frame_files,
            None,
        )
    fingerprint = source_frame_metadata.get("source_frame_fingerprint")
    if not isinstance(fingerprint, str):
        return (
            _stage(
                "frames",
                paths,
                "invalidate",
                "source frame manifest fingerprint is malformed",
            ),
            frame_files,
            None,
        )
    return (
        _stage("frames", paths, "preserve", "source frame manifest is reusable"),
        frame_files,
        fingerprint,
    )


def _resume_source_candidates(root: Path, metadata: dict[str, Any]) -> list[Path]:
    saved_path = metadata.get("source_video")
    saved_name = metadata.get("source_video_name")
    candidates: list[Path] = []
    if isinstance(saved_path, str) and saved_path:
        candidate = Path(saved_path).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        candidates.append(candidate.resolve())
    if isinstance(saved_name, str) and Path(saved_name).name == saved_name:
        candidates.append((root / saved_name).resolve())
    return candidates


def _current_source_fingerprint(metadata: Any) -> str | None:
    if not isinstance(metadata, dict):
        return None
    fingerprint = metadata.get("source_video_fingerprint")
    if (
        metadata.get("source_video_fingerprint_algorithm") == FILE_IDENTITY_ALGORITHM_VERSION
        and isinstance(fingerprint, str)
        and fingerprint
    ):
        return fingerprint
    return None


def _resume_source_matches(candidate: Path, root: Path, saved_fingerprint: str | None) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    if not candidate.is_file():
        return False
    return saved_fingerprint is None or file_sample_fingerprint(candidate) == saved_fingerprint


def resolve_resume_source_video(
    output_dir: Path | str, *, settings_file: Path | str | None = None
) -> Path:
    """Resolve the source recorded by one job, checking its fingerprint when available."""

    root = Path(output_dir).resolve()
    selected_settings = _resolve_settings_file(root, settings_file)
    if selected_settings is None:
        raise ValueError("Resume source video metadata is missing")
    settings_data = _read_json(selected_settings)
    metadata = (settings_data or {}).get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Resume source video metadata is invalid")
    saved_fingerprint = _current_source_fingerprint(metadata)

    checked: set[Path] = set()
    for candidate in _resume_source_candidates(root, metadata):
        if candidate in checked:
            continue
        checked.add(candidate)
        if _resume_source_matches(candidate, root, saved_fingerprint):
            return candidate
    if saved_fingerprint is None:
        raise ValueError("Recorded source video is missing")
    raise ValueError("Recorded source video is missing or its fingerprint does not match")


def _validate_final_bounds(scene_dir: Path, manifest: dict[str, Any]) -> dict[str, Any] | None:
    bounds = _read_json(scene_dir / "depth_bounds.json")
    if bounds is None:
        return None
    fingerprint = bounds.get("fingerprint")
    unhashed = {key: value for key, value in bounds.items() if key != "fingerprint"}
    scenes = bounds.get("scenes")
    scene_ids = manifest.get("scene_ids")
    if not isinstance(scene_ids, list):
        return None
    try:
        bounds_scene_ids = {int(value) for value in scenes} if isinstance(scenes, dict) else set()
        manifest_scene_ids = {int(value) for value in scene_ids}
    except (TypeError, ValueError):
        return None
    valid = (
        bounds.get("schema_version") == DEPTH_BOUNDS_SCHEMA_VERSION
        and bounds.get("algorithm_version") == CANONICAL_DEPTH_ALGORITHM_VERSION
        and isinstance(bounds.get("source_raw_fingerprint"), str)
        and bool(bounds["source_raw_fingerprint"])
        and isinstance(fingerprint, str)
        and fingerprint == canonical_json_hash(unhashed)
        and manifest.get("bounds_fingerprint") == fingerprint
        and bounds_scene_ids == manifest_scene_ids
    )
    return bounds if valid else None


def _bind_final_scene_to_raw_depth(
    scene: ResumeStage,
    manifest: dict[str, Any] | None,
    bounds: dict[str, Any] | None,
    raw: ResumeStage,
    raw_metadata: dict[str, Any] | None,
) -> tuple[ResumeStage, dict[str, Any] | None]:
    if scene.disposition != "preserve" or manifest is None or bounds is None:
        return scene, bounds
    raw_fingerprint = raw_metadata.get("fingerprint") if raw_metadata is not None else None
    if (
        raw.disposition not in {"preserve", "resume"}
        or not isinstance(raw_fingerprint, str)
        or bounds.get("source_raw_fingerprint") != raw_fingerprint
    ):
        reason = "depth-derived scene bounds will be recomputed for current raw depth"
        return _stage("scene_data", scene.paths, "resume", reason), None
    return scene, bounds


def _validate_scene_stage(
    output_dir: Path,
    frame_files: list[Path],
    source_fingerprint: str | None,
    frames_reusable: bool,
    current_settings: dict[str, Any],
) -> tuple[ResumeStage, dict[str, Any] | None, dict[str, Any] | None]:
    scene_dir = output_dir / "01_scene_data"
    paths = (scene_dir,)
    if not _has_payload(scene_dir):
        return _stage("scene_data", paths, "missing", "scene data is absent"), None, None
    if not frames_reusable or source_fingerprint is None:
        return _stage("scene_data", paths, "invalidate", "source frames are invalid"), None, None

    manifest = _read_json(scene_dir / "scene_manifest.json")
    frame_names = [path.name for path in frame_files]
    expected_settings = {
        "enabled": bool(current_settings.get("scene_detection", True)),
        "threshold": float(current_settings.get("scene_cut_threshold", 0.55)),
        "min_frames": int(current_settings.get("min_scene_frames", 8)),
    }
    if manifest is None:
        return (
            _stage("scene_data", paths, "invalidate", "scene manifest schema or source mismatch"),
            None,
            None,
        )
    valid = (
        manifest.get("schema_version") == SCENE_SCHEMA_VERSION
        and manifest.get("algorithm_version") == SCENE_ALGORITHM_VERSION
        and manifest.get("frame_names") == frame_names
        and manifest.get("source_frame_fingerprint") == source_fingerprint
    )
    if not valid:
        return (
            _stage("scene_data", paths, "invalidate", "scene manifest schema or source mismatch"),
            None,
            None,
        )
    if manifest.get("settings") != expected_settings:
        return (
            _stage("scene_data", paths, "invalidate", "scene settings mismatch"),
            None,
            None,
        )
    scene_ids = manifest.get("scene_ids")
    valid_scene_ids = (
        isinstance(scene_ids, list)
        and len(scene_ids) == len(frame_files)
        and all(
            isinstance(scene_id, int) and not isinstance(scene_id, bool) and scene_id >= 0
            for scene_id in scene_ids
        )
    )
    if not valid_scene_ids:
        return (
            _stage("scene_data", paths, "invalidate", "scene IDs do not match source frames"),
            None,
            None,
        )
    if manifest.get("status") == "candidate":
        reason = "candidate manifest will resume deterministic finalization"
        return _stage("scene_data", paths, "resume", reason), manifest, None
    if manifest.get("status") != "final":
        return (
            _stage("scene_data", paths, "invalidate", "scene manifest status is invalid"),
            None,
            None,
        )

    bounds = _validate_final_bounds(scene_dir, manifest)
    if bounds is None:
        return (
            _stage("scene_data", paths, "invalidate", "final scene bounds fingerprint mismatch"),
            None,
            None,
        )
    return (
        _stage("scene_data", paths, "preserve", "final scene manifest and bounds match"),
        manifest,
        bounds,
    )


def _v2_execution_contract_mismatch_reason(
    semantic: dict[str, Any],
    current_settings: dict[str, Any],
    frame_files: list[Path],
    current_model_fingerprint: dict[str, Any] | None,
) -> str | None:
    persisted_model_info = semantic.get("model_info")
    current_model_info = (
        current_model_fingerprint.get("model_info")
        if current_model_fingerprint is not None
        else None
    )
    persisted_algorithm = (
        persisted_model_info.get("inference_algorithm")
        if isinstance(persisted_model_info, dict)
        else None
    )
    current_algorithm = (
        current_model_info.get("inference_algorithm")
        if isinstance(current_model_info, dict)
        else None
    )
    is_v2 = (
        persisted_algorithm == VDA_INFERENCE_ALGORITHM
        or current_algorithm == VDA_INFERENCE_ALGORITHM
    )
    if not is_v2:
        return None
    if semantic.get("scene_algorithm_version") != SCENE_ALGORITHM_VERSION:
        return "raw-depth V2 scene algorithm fingerprint mismatch"
    header = read_png_header(frame_files[0]) if frame_files else None
    if header is None:
        return "raw-depth V2 execution plan source geometry is unavailable"
    requested_size = resolve_depth_input_size(
        header.width,
        header.height,
        current_settings.get("depth_resolution", "auto"),
    )
    try:
        requested_plan = build_v2_execution_plan(requested_size, requested_size)
    except ValueError:
        return "raw-depth V2 execution plan request is invalid"
    if not is_compatible_v2_execution_plan(
        semantic.get("execution_plan"),
        requested_plan,
    ):
        return "raw-depth V2 execution plan is missing or incompatible"
    return None


def _raw_semantic_mismatch_reason(
    semantic: object,
    current_settings: dict[str, Any],
    source_fingerprint: str,
    frame_files: list[Path],
    current_model_fingerprint: dict[str, Any] | None,
) -> str | None:
    if not isinstance(semantic, dict):
        return "raw-depth semantic fingerprint is missing"
    if semantic.get("source_frame_fingerprint") != source_fingerprint:
        return "raw-depth source-frame fingerprint mismatch"
    depth_settings = semantic.get("depth_settings")
    if not isinstance(depth_settings, dict):
        return "raw-depth model settings fingerprint is missing"
    persisted_model = {
        key: semantic.get(key)
        for key in (
            "backend",
            "model_info",
            "depth_settings",
            "weight_sha256",
            "artifact_identity",
        )
    }
    if current_model_fingerprint is not None and persisted_model != current_model_fingerprint:
        return "raw-depth model fingerprint mismatch"
    if current_model_fingerprint is None:
        expected_depth_settings = select_depth_model_settings(current_settings)
    else:
        model_depth_settings = current_model_fingerprint.get("depth_settings")
        if not isinstance(model_depth_settings, dict):
            return "current model settings fingerprint is missing"
        expected_depth_settings = model_depth_settings
    if depth_settings != expected_depth_settings:
        changed = sorted(set(depth_settings) | set(expected_depth_settings))
        key = next(
            (
                name
                for name in changed
                if depth_settings.get(name) != expected_depth_settings.get(name)
            ),
            "unknown",
        )
        return f"raw-depth model setting mismatch: {key}"
    if semantic.get("preprocessing_algorithm") != depth_preprocessing_algorithm(current_settings):
        return "raw-depth preprocessing fingerprint mismatch"
    return _v2_execution_contract_mismatch_reason(
        semantic,
        current_settings,
        frame_files,
        current_model_fingerprint,
    )


def _raw_storage_mismatch_reason(
    metadata: dict[str, Any], current_settings: dict[str, Any]
) -> str | None:
    requested_dtype = current_settings.get("raw_storage_dtype", "auto")
    persisted_request = metadata.get("requested_dtype")
    promotes = metadata.get("selected_dtype") == "float16" and requested_dtype == "float32"
    if persisted_request != requested_dtype and not promotes:
        return "raw-depth storage fingerprint mismatch"
    if metadata.get("storage_status") not in {"ready", "promoting"}:
        return "raw-depth storage transaction status is invalid"
    if metadata.get("storage_status") == "ready":
        if metadata.get("fingerprint") != RawDepthStore._fingerprint(metadata):
            return "raw-depth metadata fingerprint mismatch"
    return None


def _raw_promotion_pending(metadata: dict[str, Any], current_settings: dict[str, Any]) -> bool:
    return metadata.get("storage_status") == "promoting" or (
        metadata.get("selected_dtype") == "float16"
        and current_settings.get("raw_storage_dtype", "auto") == "float32"
    )


def _raw_mismatch_reason(
    metadata: dict[str, Any],
    current_settings: dict[str, Any],
    frame_files: list[Path],
    source_fingerprint: str,
    current_model_fingerprint: dict[str, Any] | None,
) -> str | None:
    if metadata.get("schema_version") != RAW_DEPTH_SCHEMA_VERSION:
        return "raw-depth schema mismatch"
    if metadata.get("frame_names") != [path.name for path in frame_files]:
        return "raw-depth frame manifest mismatch"
    semantic_reason = _raw_semantic_mismatch_reason(
        metadata.get("semantic_fingerprint"),
        current_settings,
        source_fingerprint,
        frame_files,
        current_model_fingerprint,
    )
    if semantic_reason is not None:
        return semantic_reason
    return _raw_storage_mismatch_reason(metadata, current_settings)


def _validate_raw_stage(
    output_dir: Path,
    current_settings: dict[str, Any],
    frame_files: list[Path],
    source_fingerprint: str | None,
    upstream_reusable: bool,
    current_model_fingerprint: dict[str, Any] | None,
) -> tuple[ResumeStage, dict[str, Any] | None]:
    raw_dir = output_dir / "02_depth_raw"
    paths = (raw_dir,)
    if not _has_payload(raw_dir):
        return _stage("depth_raw", paths, "missing", "raw depth is absent"), None
    if not upstream_reusable or source_fingerprint is None:
        return _stage("depth_raw", paths, "invalidate", "scene or frame stage is invalid"), None
    metadata = _read_json(raw_dir / "metadata.json")
    if metadata is None:
        return _stage("depth_raw", paths, "invalidate", "raw-depth metadata is missing"), None
    reason = _raw_mismatch_reason(
        metadata,
        current_settings,
        frame_files,
        source_fingerprint,
        current_model_fingerprint,
    )
    if reason is not None:
        return _stage("depth_raw", paths, "invalidate", reason), None
    try:
        completed_count = RawDepthStore(raw_dir, metadata).validate_payloads()
    except RawDepthFingerprintError as error:
        reason = f"raw-depth payload validation failed: {error}"
        return _stage("depth_raw", paths, "invalidate", reason), None
    if _raw_promotion_pending(metadata, current_settings):
        reason = "raw-depth float16-to-float32 promotion will resume"
        return _stage("depth_raw", paths, "resume", reason), None
    disposition: Disposition = "resume"
    if completed_count == len(frame_files):
        disposition = "preserve"
    reason = "raw-depth metadata and partial frame names are reusable"
    return _stage("depth_raw", paths, disposition, reason), metadata


def _canonical_mismatch_reason(
    metadata: dict[str, Any],
    frame_files: list[Path],
    raw_metadata: dict[str, Any],
    manifest: dict[str, Any],
    bounds: dict[str, Any],
) -> str | None:
    fingerprint = metadata.get("fingerprint")
    unhashed = {key: value for key, value in metadata.items() if key != "fingerprint"}
    checks = (
        metadata.get("schema_version") == CANONICAL_DEPTH_SCHEMA_VERSION,
        metadata.get("algorithm_version") == CANONICAL_DEPTH_ALGORITHM_VERSION,
        metadata.get("representation") == "relative_disparity",
        metadata.get("near_value") == 1.0,
        metadata.get("far_value") == 0.0,
        metadata.get("encoding") == "uint16_png",
        metadata.get("encoding_scale") == 65535.0,
        metadata.get("frame_names") == [path.name for path in frame_files],
        metadata.get("num_frames") == len(frame_files),
        metadata.get("source_raw_fingerprint") == raw_metadata.get("fingerprint"),
        metadata.get("source_model_fingerprint")
        == canonical_json_hash(raw_metadata.get("semantic_fingerprint")),
        metadata.get("scene_manifest_fingerprint") == canonical_json_hash(manifest),
        metadata.get("depth_bounds_fingerprint") == bounds.get("fingerprint"),
        isinstance(fingerprint, str) and fingerprint == canonical_json_hash(unhashed),
    )
    return None if all(checks) else "canonical disparity metadata fingerprint mismatch"


def _positive_shape(metadata: dict[str, Any], key: str) -> tuple[int, int] | None:
    try:
        shape = tuple(int(value) for value in metadata[key])
    except (KeyError, TypeError, ValueError):
        return None
    if len(shape) != 2 or any(value < 1 for value in shape):
        return None
    return shape


def _canonical_payload_state(
    expected_files: list[Path], native_shape: tuple[int, int]
) -> tuple[Disposition | None, str | None]:
    missing = False
    for path in expected_files:
        if not path.is_file():
            missing = True
            continue
        if not png_header_matches(path, shape=native_shape, bit_depth=16):
            return "resume", f"canonical payload will be regenerated: {path.name}"
    if missing:
        return "resume", "canonical stage is partially complete"
    return None, None


def _validate_canonical_stage(
    output_dir: Path,
    frame_files: list[Path],
    manifest: dict[str, Any] | None,
    bounds: dict[str, Any] | None,
    raw_metadata: dict[str, Any] | None,
    scene_is_final: bool,
) -> tuple[ResumeStage, dict[str, Any] | None]:
    canonical_dir = output_dir / "03_disparity_maps"
    paths = (canonical_dir,)
    if not _has_payload(canonical_dir):
        stage = _stage("disparity_maps", paths, "missing", "canonical disparity is absent")
        return stage, None
    if not scene_is_final:
        stage = _stage(
            "disparity_maps",
            paths,
            "invalidate",
            "candidate or invalid scene manifest cannot canonicalize",
        )
        return stage, None
    if raw_metadata is None or manifest is None or bounds is None:
        stage = _stage("disparity_maps", paths, "invalidate", "raw or scene metadata is invalid")
        return stage, None
    metadata = _read_json(canonical_dir / "metadata.json")
    if metadata is None:
        stage = _stage("disparity_maps", paths, "invalidate", "canonical metadata is missing")
        return stage, None
    reason = _canonical_mismatch_reason(metadata, frame_files, raw_metadata, manifest, bounds)
    if reason is not None:
        return _stage("disparity_maps", paths, "invalidate", reason), None
    expected_files = [canonical_dir / f"{path.stem}.png" for path in frame_files]
    native_shape = _positive_shape(metadata, "native_shape")
    if native_shape is None:
        return (
            _stage(
                "disparity_maps",
                paths,
                "invalidate",
                "canonical native shape metadata is invalid",
            ),
            None,
        )
    disposition, payload_reason = _canonical_payload_state(expected_files, native_shape)
    if disposition is not None and payload_reason is not None:
        return _stage("disparity_maps", paths, disposition, payload_reason), metadata
    stage = _stage("disparity_maps", paths, "preserve", "canonical metadata and files match")
    return stage, metadata


def _settings_changed(
    saved: dict[str, Any],
    current: dict[str, Any],
    keys: tuple[str, ...],
) -> str | None:
    for key in keys:
        if saved.get(key) != current.get(key):
            return key
    return None


def _validate_generated_stage(
    name: str,
    paths: tuple[Path, ...],
    *,
    upstream_reusable: bool,
    changed_setting: str | None,
    reusable_reason: str,
) -> ResumeStage:
    if not any(_has_payload(path) for path in paths):
        return _stage(name, paths, "missing", f"{name} output is absent")
    if not upstream_reusable:
        return _stage(name, paths, "invalidate", "an upstream stage is invalid")
    if changed_setting is not None:
        return _stage(name, paths, "invalidate", f"setting changed: {changed_setting}")
    metadata = _read_json(paths[0] / "metadata.json")
    if metadata is None:
        return _stage(name, paths, "invalidate", "completion manifest is missing")
    fingerprint = metadata.get("fingerprint")
    unhashed = {key: value for key, value in metadata.items() if key != "fingerprint"}
    if (
        metadata.get("schema_version") != FRAME_STAGE_SCHEMA_VERSION
        or not isinstance(fingerprint, str)
        or fingerprint != canonical_json_hash(unhashed)
    ):
        return _stage(name, paths, "invalidate", "completion manifest is invalid")
    return _stage(name, paths, "preserve", reusable_reason)


def _stereo_metadata_matches(
    metadata: dict[str, Any],
    canonical_metadata: dict[str, Any],
    frame_files: list[Path],
    current_settings: dict[str, Any],
) -> bool:
    fingerprint = metadata.get("fingerprint")
    unhashed = {key: value for key, value in metadata.items() if key != "fingerprint"}
    expected_render_settings = {
        "stereo_strength": current_settings.get("stereo_strength"),
        "convergence": current_settings.get("convergence"),
        "occlusion_fill": current_settings.get("occlusion_fill"),
    }
    checks = (
        metadata.get("schema_version") == STEREO_STAGE_SCHEMA_VERSION,
        metadata.get("algorithm_version") == STEREO_STAGE_ALGORITHM_VERSION,
        metadata.get("source_canonical_fingerprint") == canonical_metadata.get("fingerprint"),
        metadata.get("frame_names") == [path.name for path in frame_files],
        metadata.get("render_settings") == expected_render_settings,
        metadata.get("encoding") == "uint8_png",
        isinstance(fingerprint, str) and fingerprint == canonical_json_hash(unhashed),
    )
    return all(checks)


def _stereo_payload_state(
    paths: tuple[Path, ...],
    frame_files: list[Path],
    render_shape: tuple[int, int],
) -> tuple[Disposition | None, str | None]:
    missing = False
    for frame in frame_files:
        for directory in paths:
            path = directory / f"{frame.stem}.png"
            if not path.is_file():
                missing = True
                continue
            if not png_header_matches(path, shape=(*render_shape, 3), bit_depth=8):
                return "resume", f"stereo payload will be regenerated: {path.name}"
    if missing:
        return "resume", "stereo stage is partially complete"
    return None, None


def _validate_stereo_stage(
    output_dir: Path,
    frame_files: list[Path],
    current_settings: dict[str, Any],
    canonical_metadata: dict[str, Any] | None,
    *,
    current_settings_schema: bool,
    canonical_reusable: bool,
) -> ResumeStage:
    paths = (output_dir / "04_left_frames", output_dir / "04_right_frames")
    if not any(_has_payload(path) for path in paths):
        return _stage("stereo", paths, "missing", "stereo output is absent")
    if not canonical_reusable or canonical_metadata is None:
        return _stage("stereo", paths, "invalidate", "canonical disparity is invalid")
    if not current_settings_schema:
        return _stage("stereo", paths, "invalidate", "legacy renderer schema")

    metadata = _read_json(paths[0] / "metadata.json")
    if metadata is None:
        return _stage("stereo", paths, "invalidate", "stereo metadata is missing")
    if not _stereo_metadata_matches(
        metadata,
        canonical_metadata,
        frame_files,
        current_settings,
    ):
        return _stage("stereo", paths, "invalidate", "stereo stage fingerprint mismatch")
    render_shape = _positive_shape(metadata, "render_shape")
    if render_shape is None:
        return _stage("stereo", paths, "invalidate", "stereo render shape is invalid")
    disposition, reason = _stereo_payload_state(paths, frame_files, render_shape)
    if disposition is not None and reason is not None:
        return _stage("stereo", paths, disposition, reason)
    return _stage("stereo", paths, "preserve", "stereo stage fingerprint matches")


def _build_generated_stages(
    output_dir: Path,
    frame_files: list[Path],
    saved_settings: dict[str, Any],
    current_settings: dict[str, Any],
    canonical_metadata: dict[str, Any] | None,
    *,
    current_settings_schema: bool,
    canonical_reusable: bool,
) -> list[ResumeStage]:
    stereo = _validate_stereo_stage(
        output_dir,
        frame_files,
        current_settings,
        canonical_metadata,
        current_settings_schema=current_settings_schema,
        canonical_reusable=canonical_reusable,
    )
    distortion = _validate_generated_stage(
        "distortion",
        (output_dir / "05_left_distorted", output_dir / "05_right_distorted"),
        upstream_reusable=stereo.disposition in {"preserve", "resume"},
        changed_setting=_settings_changed(
            saved_settings, current_settings, _DISTORTION_SETTING_KEYS
        ),
        reusable_reason="distortion settings match",
    )
    crop = _validate_generated_stage(
        "crop",
        (output_dir / "06_left_cropped", output_dir / "06_right_cropped"),
        upstream_reusable=stereo.disposition in {"preserve", "resume"}
        and distortion.disposition in {"preserve", "missing"},
        changed_setting=_settings_changed(saved_settings, current_settings, _CROP_SETTING_KEYS),
        reusable_reason="crop settings match",
    )
    upscale = _validate_generated_stage(
        "upscale",
        (output_dir / "07_left_upscaled", output_dir / "07_right_upscaled"),
        upstream_reusable=crop.disposition in {"preserve", "missing"},
        changed_setting=_settings_changed(saved_settings, current_settings, _UPSCALE_SETTING_KEYS),
        reusable_reason="upscale settings match",
    )
    vr = _validate_generated_stage(
        "vr_frames",
        (output_dir / "99_vr_frames",),
        upstream_reusable=stereo.disposition in {"preserve", "resume"}
        and crop.disposition in {"preserve", "missing"}
        and upscale.disposition in {"preserve", "missing"},
        changed_setting=_settings_changed(saved_settings, current_settings, _VR_SETTING_KEYS),
        reusable_reason="VR assembly settings match",
    )
    return [stereo, distortion, crop, upscale, vr]


def build_resume_report(
    output_dir: Path | str,
    current_settings: dict[str, Any],
    *,
    source_video: Path | str | None = None,
    model_fingerprint: dict[str, Any] | None = None,
    settings_file: Path | str | None = None,
) -> ResumeReport:
    """Inspect an output directory without mutating it."""

    root = Path(output_dir).resolve()
    selected_settings_file = _resolve_settings_file(root, settings_file)
    settings_data = (
        _read_json(selected_settings_file) if selected_settings_file is not None else None
    )
    source_reason = _source_video_mismatch_reason(settings_data, source_video)
    if source_reason is not None:
        raise ValueError(f"Cannot resume: {source_reason}")
    raw_saved = (settings_data or {}).get("processing_settings", {})
    if not isinstance(raw_saved, dict):
        raw_saved = {}
    removed = tuple(sorted(REMOVED_SETTING_NAMES.intersection(raw_saved)))
    saved_settings = validate_settings(raw_saved, source="legacy_disk")
    migrated_settings = validate_settings(current_settings, source="explicit")
    settings_metadata = (settings_data or {}).get("metadata", {})
    settings_schema_current = (
        isinstance(settings_metadata, dict)
        and settings_metadata.get("settings_schema_version") == PROCESSING_SETTINGS_SCHEMA_VERSION
    )

    frames, frame_files, source_fingerprint = _validate_frame_stage(
        root, settings_data, saved_settings
    )
    frames_reusable = frames.disposition == "preserve"

    scene, manifest, bounds = _validate_scene_stage(
        root,
        frame_files,
        source_fingerprint,
        frames_reusable,
        migrated_settings,
    )
    scene_reusable = scene.disposition in {"preserve", "resume"}
    raw, raw_metadata = _validate_raw_stage(
        root,
        migrated_settings,
        frame_files,
        source_fingerprint,
        scene_reusable,
        model_fingerprint,
    )
    scene, bounds = _bind_final_scene_to_raw_depth(
        scene,
        manifest,
        bounds,
        raw,
        raw_metadata,
    )

    stages: list[ResumeStage] = [frames, scene, raw]

    canonical, canonical_metadata = _validate_canonical_stage(
        root,
        frame_files,
        manifest,
        bounds,
        raw_metadata,
        scene.disposition == "preserve",
    )
    stages.append(canonical)

    legacy_supersampled = root / "01_supersampled_frames"
    if _has_payload(legacy_supersampled):
        stages.append(
            _stage(
                "legacy_supersampled",
                (legacy_supersampled,),
                "invalidate",
                "legacy supersampled frames are outside the current pipeline",
            )
        )
    legacy_depth = root / "02_depth_maps"
    if _has_payload(legacy_depth):
        stages.append(
            _stage(
                "legacy_depth_maps",
                (legacy_depth,),
                "invalidate",
                "legacy depth PNG semantics are unknown and cannot be migrated",
            )
        )
    legacy_final_paths = (root / "08_left_final", root / "08_right_final")
    if any(_has_payload(path) for path in legacy_final_paths):
        stages.append(
            _stage(
                "legacy_final",
                legacy_final_paths,
                "invalidate",
                "legacy final-eye directories are outside the current pipeline",
            )
        )

    stages.extend(
        _build_generated_stages(
            root,
            frame_files,
            saved_settings,
            migrated_settings,
            canonical_metadata,
            current_settings_schema=settings_schema_current,
            canonical_reusable=canonical.disposition in {"preserve", "resume"},
        )
    )
    backup_required = selected_settings_file is not None and (
        bool(removed) or not settings_schema_current
    )
    return ResumeReport(
        output_dir=root,
        stages=tuple(stages),
        removed_settings=removed,
        settings_file=selected_settings_file,
        original_settings_data=settings_data,
        migrated_settings=migrated_settings,
        settings_backup_required=backup_required,
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
    _atomic_write_bytes(path, encoded)


def _preflight_migration(report: ResumeReport, mode: MigrationMode) -> dict[Path, Path]:
    destinations: dict[Path, Path] = {}
    if mode == "archive":
        archive_root = report.output_dir / "legacy_v1"
        for source in report.migration_paths:
            destination = archive_root / source.name
            if destination.exists():
                raise FileExistsError(f"Legacy archive destination already exists: {destination}")
            destinations[source] = destination
    else:
        staging_root = report.output_dir / ".resume_delete_staging"
        if staging_root.exists():
            raise FileExistsError(f"Legacy delete staging directory already exists: {staging_root}")
        for index, source in enumerate(report.migration_paths):
            destinations[source] = staging_root / f"{index:03d}_{source.name}"
    if report.settings_backup_required:
        backup = report.output_dir / "settings.legacy.json"
        settings_file = report.settings_file
        backup_matches = (
            backup.is_file()
            and settings_file is not None
            and backup.read_bytes() == settings_file.read_bytes()
        )
        if backup.exists() and not backup_matches:
            raise FileExistsError(f"Legacy settings backup already exists: {backup}")
    return destinations


def _migrate_settings(report: ResumeReport) -> None:
    if report.settings_file is None or report.original_settings_data is None:
        return
    if report.settings_backup_required:
        backup = report.output_dir / "settings.legacy.json"
        if not backup.exists():
            _atomic_write_bytes(backup, report.settings_file.read_bytes())

    migrated = dict(report.original_settings_data)
    original_metadata = migrated.get("metadata", {})
    metadata = dict(original_metadata) if isinstance(original_metadata, dict) else {}
    metadata["settings_schema_version"] = PROCESSING_SETTINGS_SCHEMA_VERSION
    migrated["metadata"] = metadata
    migrated["processing_settings"] = report.migrated_settings
    _atomic_write_json(report.settings_file, migrated)


def _rollback_migration(
    report: ResumeReport,
    moved: list[tuple[Path, Path]],
    original_settings: bytes | None,
) -> None:
    for source, destination in reversed(moved):
        if destination.exists() and not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(destination), str(source))
    if original_settings is not None and report.settings_file is not None:
        _atomic_write_bytes(report.settings_file, original_settings)


def _remove_empty_migration_directories(output_dir: Path) -> None:
    for directory in (
        output_dir / ".resume_delete_staging",
        output_dir / "legacy_v1",
    ):
        if directory.is_dir() and next(directory.iterdir(), None) is None:
            directory.rmdir()


def apply_legacy_migration(report: ResumeReport, mode: MigrationMode) -> None:
    """Apply a precomputed report; archive by default and delete only explicitly."""

    if mode not in {"archive", "delete"}:
        raise ValueError("legacy migration mode must be archive or delete")
    destinations = _preflight_migration(report, mode)
    original_settings = (
        report.settings_file.read_bytes()
        if report.settings_file is not None and report.settings_file.is_file()
        else None
    )
    moved: list[tuple[Path, Path]] = []
    try:
        for source in report.migration_paths:
            destination = destinations[source]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            moved.append((source, destination))
        _migrate_settings(report)
    except Exception:
        _rollback_migration(report, moved, original_settings)
        _remove_empty_migration_directories(report.output_dir)
        raise

    if mode == "delete" and moved:
        try:
            shutil.rmtree(report.output_dir / ".resume_delete_staging")
        except OSError as error:
            warnings.warn(
                "Legacy migration committed; delete-staging cleanup remains pending: " f"{error}",
                RuntimeWarning,
                stacklevel=2,
            )
