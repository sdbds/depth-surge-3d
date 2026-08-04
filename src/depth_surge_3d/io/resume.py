"""Deterministic stage validation and one-way legacy resume migration."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2

from ..core.settings import (
    PROCESSING_SETTINGS_SCHEMA_VERSION,
    REMOVED_SETTING_NAMES,
    validate_settings,
)
from ..processing.frames.depth_processor import (
    CANONICAL_DEPTH_ALGORITHM_VERSION,
    CANONICAL_DEPTH_SCHEMA_VERSION,
    DEPTH_BOUNDS_SCHEMA_VERSION,
)
from ..processing.frames.depth_storage import (
    RAW_DEPTH_SCHEMA_VERSION,
    RawDepthStore,
    canonical_json_hash,
)
from ..processing.frames.scene_analyzer import (
    SCENE_ALGORITHM_VERSION,
    SCENE_SCHEMA_VERSION,
)
from ..processing.frames.stereo_generator import (
    STEREO_STAGE_ALGORITHM_VERSION,
    STEREO_STAGE_SCHEMA_VERSION,
)
from ..utils.path_utils import calculate_frame_range


Disposition = Literal["preserve", "resume", "invalidate", "missing"]
MigrationMode = Literal["archive", "delete"]

_DEPTH_SETTING_KEYS = (
    "depth_model_version",
    "model_path",
    "model_size",
    "depth_resolution",
    "use_metric_depth",
    "device",
    "super_sample",
    "temporal_window_size",
    "temporal_window_overlap",
    "denoising_steps",
    "seed",
)
_STEREO_SETTING_KEYS = ("stereo_strength", "convergence", "occlusion_fill")
_DISTORTION_SETTING_KEYS = (
    "apply_distortion",
    "fisheye_projection",
    "fisheye_fov",
)
_CROP_SETTING_KEYS = ("crop_factor", "fisheye_crop_factor")
_UPSCALE_SETTING_KEYS = ("upscale_model",)
_VR_SETTING_KEYS = ("vr_format", "vr_resolution", "target_fps")


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
        for stage in self.stages:
            if stage.disposition != "invalidate":
                continue
            for path in stage.paths:
                resolved = path.resolve()
                if resolved not in seen and _has_payload(path):
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


def _find_settings_file(output_dir: Path) -> Path | None:
    candidates = sorted(
        output_dir.glob("*-settings.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    return candidates[0] if candidates else None


def _frame_fingerprint(frame_files: list[Path]) -> str:
    hasher = hashlib.sha256()
    for path in frame_files:
        hasher.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                hasher.update(chunk)
    return hasher.hexdigest()


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

    metadata = (settings_data or {}).get("metadata", {})
    saved_source_hash = metadata.get("source_video_sha256") if isinstance(metadata, dict) else None
    source_video = metadata.get("source_video") if isinstance(metadata, dict) else None
    if isinstance(saved_source_hash, str) and isinstance(source_video, str):
        source_path = Path(source_video)
        if not source_path.is_file() or _hash_file(source_path) != saved_source_hash:
            reason = "source video fingerprint mismatch"
            return _stage("frames", paths, "invalidate", reason), frame_files, None

    expected_count = _expected_frame_count(settings_data, saved_settings)
    if expected_count is not None and len(frame_files) != expected_count:
        reason = f"frame count mismatch: found {len(frame_files)}, expected {expected_count}"
        return _stage("frames", paths, "invalidate", reason), frame_files, None

    expected_properties = (settings_data or {}).get("video_properties", {})
    expected_shape = None
    if isinstance(expected_properties, dict):
        width = expected_properties.get("width")
        height = expected_properties.get("height")
        if isinstance(width, int) and isinstance(height, int):
            expected_shape = (height, width)
    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file), cv2.IMREAD_UNCHANGED)
        if frame is None:
            return (
                _stage("frames", paths, "invalidate", f"unreadable frame: {frame_file.name}"),
                frame_files,
                None,
            )
        if expected_shape is not None and frame.shape[:2] != expected_shape:
            reason = (
                f"frame dimensions mismatch: {frame_file.name} is {frame.shape[1]}x{frame.shape[0]}"
            )
            return _stage("frames", paths, "invalidate", reason), frame_files, None

    fingerprint = _frame_fingerprint(frame_files)
    return (
        _stage("frames", paths, "preserve", "source frame manifest is reusable"),
        frame_files,
        fingerprint,
    )


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _validate_final_bounds(scene_dir: Path, manifest: dict[str, Any]) -> dict[str, Any] | None:
    bounds = _read_json(scene_dir / "depth_bounds.json")
    if bounds is None:
        return None
    fingerprint = bounds.get("fingerprint")
    unhashed = {key: value for key, value in bounds.items() if key != "fingerprint"}
    valid = (
        bounds.get("schema_version") == DEPTH_BOUNDS_SCHEMA_VERSION
        and bounds.get("algorithm_version") == CANONICAL_DEPTH_ALGORITHM_VERSION
        and isinstance(fingerprint, str)
        and fingerprint == canonical_json_hash(unhashed)
        and manifest.get("bounds_fingerprint") == fingerprint
    )
    return bounds if valid else None


def _validate_scene_stage(
    output_dir: Path,
    frame_files: list[Path],
    source_fingerprint: str | None,
    frames_reusable: bool,
) -> tuple[ResumeStage, dict[str, Any] | None, dict[str, Any] | None]:
    scene_dir = output_dir / "01_scene_data"
    paths = (scene_dir,)
    if not _has_payload(scene_dir):
        return _stage("scene_data", paths, "missing", "scene data is absent"), None, None
    if not frames_reusable or source_fingerprint is None:
        return _stage("scene_data", paths, "invalidate", "source frames are invalid"), None, None

    manifest = _read_json(scene_dir / "scene_manifest.json")
    frame_names = [path.name for path in frame_files]
    valid = manifest is not None and (
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


def _raw_semantic_mismatch_reason(
    semantic: object,
    current_settings: dict[str, Any],
    source_fingerprint: str,
) -> str | None:
    if not isinstance(semantic, dict):
        return "raw-depth semantic fingerprint is missing"
    if semantic.get("source_frame_fingerprint") != source_fingerprint:
        return "raw-depth source-frame fingerprint mismatch"
    depth_settings = semantic.get("depth_settings")
    if not isinstance(depth_settings, dict):
        return "raw-depth model settings fingerprint is missing"
    for key in _DEPTH_SETTING_KEYS:
        if key in current_settings and depth_settings.get(key) != current_settings.get(key):
            return f"raw-depth model setting mismatch: {key}"
    expected_model = current_settings.get("model_fingerprint")
    if expected_model is not None and canonical_json_hash(semantic) != expected_model:
        return "raw-depth model fingerprint mismatch"
    return None


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


def _raw_mismatch_reason(
    metadata: dict[str, Any],
    current_settings: dict[str, Any],
    frame_files: list[Path],
    source_fingerprint: str,
) -> str | None:
    if metadata.get("schema_version") != RAW_DEPTH_SCHEMA_VERSION:
        return "raw-depth schema mismatch"
    if metadata.get("frame_names") != [path.name for path in frame_files]:
        return "raw-depth frame manifest mismatch"
    semantic_reason = _raw_semantic_mismatch_reason(
        metadata.get("semantic_fingerprint"),
        current_settings,
        source_fingerprint,
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
    reason = _raw_mismatch_reason(metadata, current_settings, frame_files, source_fingerprint)
    if reason is not None:
        return _stage("depth_raw", paths, "invalidate", reason), None
    disposition: Disposition = "resume"
    if metadata.get("completed_count") == len(frame_files):
        disposition = "preserve"
    reason = "raw-depth metadata and partial frame names are reusable"
    if metadata.get("storage_status") == "promoting":
        reason = "raw-depth float16-to-float32 promotion will resume"
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
        metadata.get("frame_names") == [path.name for path in frame_files],
        metadata.get("num_frames") == len(frame_files),
        metadata.get("source_raw_fingerprint") == raw_metadata.get("fingerprint"),
        metadata.get("scene_manifest_fingerprint") == canonical_json_hash(manifest),
        metadata.get("depth_bounds_fingerprint") == bounds.get("fingerprint"),
        isinstance(fingerprint, str) and fingerprint == canonical_json_hash(unhashed),
    )
    return None if all(checks) else "canonical disparity metadata fingerprint mismatch"


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
    if not all(path.is_file() for path in expected_files):
        stage = _stage("disparity_maps", paths, "resume", "canonical stage is partially complete")
        return stage, metadata
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
    return _stage(name, paths, "preserve", reusable_reason)


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
    fingerprint = metadata.get("fingerprint")
    unhashed = {key: value for key, value in metadata.items() if key != "fingerprint"}
    expected_render_settings = {
        "stereo_strength": current_settings.get("stereo_strength"),
        "convergence": current_settings.get("convergence"),
        "occlusion_fill": current_settings.get("occlusion_fill"),
    }
    valid = (
        metadata.get("schema_version") == STEREO_STAGE_SCHEMA_VERSION
        and metadata.get("algorithm_version") == STEREO_STAGE_ALGORITHM_VERSION
        and metadata.get("source_canonical_fingerprint") == canonical_metadata.get("fingerprint")
        and metadata.get("frame_names") == [path.name for path in frame_files]
        and metadata.get("render_settings") == expected_render_settings
        and isinstance(fingerprint, str)
        and fingerprint == canonical_json_hash(unhashed)
    )
    if not valid:
        return _stage("stereo", paths, "invalidate", "stereo stage fingerprint mismatch")
    complete = all(
        (paths[0] / f"{frame.stem}.png").is_file() and (paths[1] / f"{frame.stem}.png").is_file()
        for frame in frame_files
    )
    if not complete:
        return _stage("stereo", paths, "resume", "stereo stage is partially complete")
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


def build_resume_report(output_dir: Path | str, current_settings: dict[str, Any]) -> ResumeReport:
    """Inspect an output directory without mutating it."""

    root = Path(output_dir).resolve()
    settings_file = _find_settings_file(root)
    settings_data = _read_json(settings_file) if settings_file is not None else None
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

    stages: list[ResumeStage] = []
    frames, frame_files, source_fingerprint = _validate_frame_stage(
        root, settings_data, saved_settings
    )
    stages.append(frames)
    frames_reusable = frames.disposition == "preserve"

    scene, manifest, bounds = _validate_scene_stage(
        root,
        frame_files,
        source_fingerprint,
        frames_reusable,
    )
    stages.append(scene)
    scene_reusable = scene.disposition in {"preserve", "resume"}
    raw, raw_metadata = _validate_raw_stage(
        root,
        migrated_settings,
        frame_files,
        source_fingerprint,
        scene_reusable,
    )
    stages.append(raw)

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
    backup_required = settings_file is not None and (bool(removed) or not settings_schema_current)
    return ResumeReport(
        output_dir=root,
        stages=tuple(stages),
        removed_settings=removed,
        settings_file=settings_file,
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


def apply_legacy_migration(report: ResumeReport, mode: MigrationMode) -> None:
    """Apply a precomputed report; archive by default and delete only explicitly."""

    if mode not in {"archive", "delete"}:
        raise ValueError("legacy migration mode must be archive or delete")
    destinations = _preflight_migration(report, mode)

    for source in report.migration_paths:
        if mode == "archive":
            destination = destinations[source]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
        elif source.is_dir():
            shutil.rmtree(source)
        else:
            source.unlink()
    _migrate_settings(report)
