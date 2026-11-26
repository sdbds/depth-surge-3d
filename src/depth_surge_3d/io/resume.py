"""Deterministic stage validation and one-way legacy resume migration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
import shutil
from typing import Any, Literal
import warnings

from ..core.depth_contract import (
    CANONICAL_DEPTH_ALGORITHM_VERSION,
    CANONICAL_DEPTH_SCHEMA_VERSION,
)
from ..core.file_identity import (
    FILE_IDENTITY_ALGORITHM_VERSION,
    file_sample_fingerprint,
)
from ..core.render_disparity import (
    STABILIZED_DEPTH_ALGORITHM_VERSION,
    STABILIZED_DEPTH_SCHEMA_VERSION,
)
from ..core.settings import (
    parse_saved_processing_settings,
    validate_settings,
)
from ..core.video_timing import selected_frame_range
from ..processing.frames.depth_processor import DEPTH_BOUNDS_SCHEMA_VERSION
from ..processing.frames.depth_storage import RAW_DEPTH_READABLE_SCHEMA_VERSIONS, RawDepthStore
from ..processing.frames.metric_geometry import MetricGeometryStore
from ..processing.frames.scene_analyzer import (
    SCENE_ALGORITHM_VERSION,
    SCENE_SCHEMA_VERSION,
)
from ..processing.frames.source_frame_manifest import (
    read_source_frame_manifest,
    source_frame_manifest_mismatch_reason,
)
from ..processing.frames.stereo_generator import (
    QUALITY_STEREO_ALGORITHM_VERSION,
    STEREO_STAGE_ALGORITHM_VERSION,
)
from ..processing.video.final_container_validation import verify_existing_final_video
from ..utils.imaging.png_header import read_png_header
from .durable_publish import atomic_write_bytes
from .operations import SETTINGS_ARTIFACT_RELATIVE_PATH
from .settings_artifact import (
    ProcessingSettingsArtifactV5,
    SETTINGS_ARTIFACT_MAX_RAW_BYTES,
    migrate_processing_settings_artifact_v5,
)


Disposition = Literal["preserve", "resume", "invalidate", "missing"]
MigrationMode = Literal["archive", "delete"]

_DEPTH_MODEL_VERSIONS = frozenset({"v2", "v3", "see_through", "moge2"})
_SEE_THROUGH_MARKERS = ("see_through", "seethrough")
_DISTORTION_KEYS = ("apply_distortion", "fisheye_projection", "fisheye_fov")
_CROP_KEYS = ("crop_factor", "fisheye_crop_factor", "per_eye_width", "per_eye_height")
_UPSCALE_KEYS = ("upscale_model",)
_VR_KEYS = ("vr_format", "vr_resolution", "target_fps", "vr_output_width", "vr_output_height")


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
    final_video_valid: bool | None = None

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
            "final_video_valid": self.final_video_valid,
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
        return str(explicit)
    if output_dir is not None and _raw_depth_uses_see_through(Path(output_dir)):
        return "see_through"
    if _contains_see_through_marker(settings.get("model_path")):
        return "see_through"
    return default


def _find_settings_file(output_dir: Path) -> Path | None:
    fixed = output_dir / SETTINGS_ARTIFACT_RELATIVE_PATH
    if fixed.is_file() and not fixed.is_symlink():
        return fixed
    candidates = sorted(
        output_dir.glob("*-settings.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    return candidates[0] if candidates else None


def _resolve_settings_file(output_dir: Path, settings_file: Path | str | None) -> Path | None:
    if settings_file is None:
        return _find_settings_file(output_dir)
    candidate = Path(settings_file)
    return candidate if candidate.is_file() else None


def _current_source_fingerprint(metadata: dict[str, Any]) -> str | None:
    if metadata.get("source_video_fingerprint_algorithm") != FILE_IDENTITY_ALGORITHM_VERSION:
        return None
    fp = metadata.get("source_video_fingerprint")
    return fp if isinstance(fp, str) and fp else None


def _resume_source_candidates(root: Path, metadata: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    source_video = metadata.get("source_video")
    if isinstance(source_video, str) and source_video:
        candidates.append(Path(source_video))
    source_name = metadata.get("source_video_name")
    if isinstance(source_name, str) and source_name:
        candidates.append(root / source_name)
    return candidates


def resolve_resume_source_video(
    output_dir: Path | str, *, settings_file: Path | str | None = None
) -> Path:
    """Resolve the source video recorded by one job, checking its sample fingerprint."""

    root = Path(output_dir).resolve()
    selected = _resolve_settings_file(root, settings_file)
    if selected is None:
        raise ValueError("Resume source video metadata is missing")
    settings_data = _read_json(selected)
    metadata = (settings_data or {}).get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Resume source video metadata is invalid")
    saved_fp = _current_source_fingerprint(metadata)

    checked: set[Path] = set()
    for candidate in _resume_source_candidates(root, metadata):
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in checked or not resolved.is_file():
            continue
        checked.add(resolved)
        if saved_fp is None or file_sample_fingerprint(resolved) == saved_fp:
            return resolved
    if saved_fp is None:
        raise ValueError("Recorded source video is missing")
    raise ValueError("Recorded source video is missing or its fingerprint does not match")


def _source_video_mismatch_reason(
    settings_data: dict[str, Any] | None,
    source_video: Path | str | None,
) -> str | None:
    if settings_data is None or source_video is None:
        return None
    metadata = settings_data.get("metadata", {})
    if not isinstance(metadata, dict):
        return None
    saved_fp = _current_source_fingerprint(metadata)
    if saved_fp is None:
        return None
    path = Path(source_video)
    if not path.is_file():
        return "source video file does not exist"
    if file_sample_fingerprint(path) != saved_fp:
        return "source video content does not match the job fingerprint"
    return None


def _validate_frame_stage(
    root: Path,
    settings_data: dict[str, Any] | None,
    migrated_settings: dict[str, Any],
) -> tuple[ResumeStage, list[Path], str | None]:
    frames_dir = root / "00_original_frames"
    frame_files = sorted(frames_dir.glob("frame_*.png")) if frames_dir.is_dir() else []
    if not frame_files:
        return _stage("frames", (frames_dir,), "missing", "no extracted frames present"), [], None

    manifest = read_source_frame_manifest(frames_dir)
    if manifest is None:
        return (
            _stage("frames", (frames_dir,), "invalidate", "missing source frame manifest"),
            [],
            None,
        )
    source_fp = manifest.get("source_video_fingerprint")
    if not isinstance(source_fp, str) or not source_fp:
        return (
            _stage("frames", (frames_dir,), "invalidate", "invalid source frame fingerprint"),
            [],
            None,
        )
    video_props = (settings_data or {}).get("video_properties", {})
    if isinstance(video_props, dict) and video_props.get("frame_count"):
        frame_range = selected_frame_range(video_props, migrated_settings)
        mismatch = source_frame_manifest_mismatch_reason(
            manifest, frame_files, source_fp, frame_range
        )
        if mismatch is not None:
            return _stage("frames", (frames_dir,), "invalidate", mismatch), [], None

    return (
        _stage("frames", (frames_dir,), "preserve", f"{len(frame_files)} extracted frames valid"),
        frame_files,
        source_fp,
    )


def _validate_scene_stage(
    root: Path,
    frame_files: list[Path],
    frames_reusable: bool,
) -> tuple[ResumeStage, dict[str, Any] | None, dict[str, Any] | None]:
    scene_dir = root / "01_scene_data"
    if not _has_payload(scene_dir):
        return _stage("scene_data", (scene_dir,), "missing", "scene data not generated"), None, None
    if not frames_reusable:
        return (
            _stage("scene_data", (scene_dir,), "invalidate", "upstream frames invalidated"),
            None,
            None,
        )
    manifest = _read_json(scene_dir / "scene_manifest.json")
    if (
        manifest is None
        or manifest.get("schema_version") != SCENE_SCHEMA_VERSION
        or manifest.get("algorithm_version") != SCENE_ALGORITHM_VERSION
        or manifest.get("frame_count") != len(frame_files)
    ):
        return (
            _stage("scene_data", (scene_dir,), "invalidate", "scene manifest invalid or stale"),
            None,
            None,
        )
    bounds = _read_json(scene_dir / "depth_bounds.json")
    if bounds is not None and (
        bounds.get("schema_version") != DEPTH_BOUNDS_SCHEMA_VERSION
        or bounds.get("algorithm_version") != CANONICAL_DEPTH_ALGORITHM_VERSION
    ):
        bounds = None
    return (
        _stage("scene_data", (scene_dir,), "preserve", "scene manifest valid"),
        manifest,
        bounds,
    )


def _validate_raw_stage(
    root: Path,
    frame_files: list[Path],
    scene_reusable: bool,
    model_fingerprint: dict[str, Any] | None,
) -> tuple[ResumeStage, dict[str, Any] | None, int]:
    raw_dir = root / "02_depth_raw"
    if not _has_payload(raw_dir):
        return _stage("depth_raw", (raw_dir,), "missing", "raw depth cache not present"), None, 0
    if not scene_reusable:
        return (
            _stage("depth_raw", (raw_dir,), "invalidate", "upstream scene stage invalidated"),
            None,
            0,
        )
    metadata = _read_json(raw_dir / "metadata.json")
    if (
        metadata is None
        or metadata.get("schema_version") not in RAW_DEPTH_READABLE_SCHEMA_VERSIONS
        or metadata.get("frame_names") != [p.name for p in frame_files]
    ):
        return (
            _stage("depth_raw", (raw_dir,), "invalidate", "raw depth metadata mismatch"),
            None,
            0,
        )
    if model_fingerprint is not None:
        saved_fp = metadata.get("semantic_fingerprint")
        if saved_fp != model_fingerprint:
            return (
                _stage("depth_raw", (raw_dir,), "invalidate", "depth model or settings changed"),
                None,
                0,
            )
    try:
        store = RawDepthStore(raw_dir, metadata)
        completed = len(store.complete_files)
    except Exception:
        completed = 0
    if completed == len(frame_files) and metadata.get("status") == "complete":
        return (
            _stage("depth_raw", (raw_dir,), "preserve", f"all {completed} raw depth frames valid"),
            metadata,
            completed,
        )
    if completed > 0:
        return (
            _stage(
                "depth_raw",
                (raw_dir,),
                "resume",
                f"{completed}/{len(frame_files)} raw depth frames present",
            ),
            metadata,
            completed,
        )
    return _stage("depth_raw", (raw_dir,), "invalidate", "no valid raw depth payloads"), None, 0


def _validate_canonical_stage(
    root: Path,
    frame_files: list[Path],
    raw_metadata: dict[str, Any] | None,
) -> tuple[ResumeStage, dict[str, Any] | None]:
    disp_dir = root / "03_disparity_maps"
    if not _has_payload(disp_dir):
        return _stage("disparity_maps", (disp_dir,), "missing", "disparity maps not present"), None
    metadata = _read_json(disp_dir / "metadata.json")
    expected_names = [p.name for p in frame_files]
    if (
        metadata is None
        or metadata.get("schema_version") != CANONICAL_DEPTH_SCHEMA_VERSION
        or metadata.get("algorithm_version") != CANONICAL_DEPTH_ALGORITHM_VERSION
        or metadata.get("frame_names") != expected_names
    ):
        return (
            _stage("disparity_maps", (disp_dir,), "invalidate", "disparity metadata mismatch"),
            None,
        )
    if raw_metadata is not None and metadata.get("source_raw_fingerprint") != raw_metadata.get(
        "fingerprint"
    ):
        return (
            _stage("disparity_maps", (disp_dir,), "invalidate", "raw depth fingerprint changed"),
            None,
        )
    pngs = sorted(p.name for p in disp_dir.glob("*.png"))
    if pngs != expected_names:
        return (
            _stage("disparity_maps", (disp_dir,), "invalidate", "disparity frame set incomplete"),
            None,
        )
    first_hdr = read_png_header(disp_dir / expected_names[0])
    if first_hdr is None or first_hdr.bit_depth != 16:
        return (
            _stage("disparity_maps", (disp_dir,), "invalidate", "disparity PNG header invalid"),
            None,
        )
    return (
        _stage(
            "disparity_maps",
            (disp_dir,),
            "preserve",
            f"{len(expected_names)} canonical disparity maps valid",
        ),
        metadata,
    )


def _validate_metric_stage(
    root: Path,
    frame_files: list[Path],
    frames_reusable: bool,
) -> tuple[ResumeStage, dict[str, Any] | None]:
    metric_dir = root / "03_metric_geometry"
    if not _has_payload(metric_dir):
        return (
            _stage("metric_geometry", (metric_dir,), "missing", "metric geometry not present"),
            None,
        )
    if not frames_reusable:
        return (
            _stage("metric_geometry", (metric_dir,), "invalidate", "upstream frames invalidated"),
            None,
        )
    metadata = _read_json(metric_dir / "metadata.json")
    expected_names = [p.name for p in frame_files]
    if metadata is None or metadata.get("frame_names") != expected_names:
        return (
            _stage("metric_geometry", (metric_dir,), "invalidate", "metric metadata mismatch"),
            None,
        )
    try:
        store = MetricGeometryStore(metric_dir, metadata)
        completed = sum(1 for name in expected_names if store.path_for(name).is_file())
    except Exception:
        completed = 0
    if completed == len(expected_names) and metadata.get("status") == "complete":
        return (
            _stage(
                "metric_geometry",
                (metric_dir,),
                "preserve",
                f"{completed} metric geometry frames valid",
            ),
            metadata,
        )
    if completed > 0:
        return (
            _stage(
                "metric_geometry",
                (metric_dir,),
                "resume",
                f"{completed}/{len(expected_names)} metric geometry frames present",
            ),
            metadata,
        )
    return (
        _stage("metric_geometry", (metric_dir,), "invalidate", "no valid metric payloads"),
        None,
    )


def _validate_stabilized_stage(
    root: Path,
    frame_files: list[Path],
    canonical_metadata: dict[str, Any] | None,
) -> tuple[ResumeStage, dict[str, Any] | None]:
    stab_dir = root / "03_disparity_stabilized"
    if not _has_payload(stab_dir):
        return (
            _stage(
                "disparity_stabilized",
                (stab_dir,),
                "missing",
                "stabilized disparity maps not present",
            ),
            None,
        )
    metadata = _read_json(stab_dir / "metadata.json")
    expected_names = [p.name for p in frame_files]
    semantic = metadata.get("semantic_identity") if metadata else None
    if (
        metadata is None
        or not isinstance(semantic, dict)
        or metadata.get("schema_version") != STABILIZED_DEPTH_SCHEMA_VERSION
        or metadata.get("algorithm_version") != STABILIZED_DEPTH_ALGORITHM_VERSION
        or semantic.get("frame_names") != expected_names
    ):
        return (
            _stage(
                "disparity_stabilized",
                (stab_dir,),
                "invalidate",
                "stabilized disparity metadata mismatch",
            ),
            None,
        )
    if canonical_metadata is not None and semantic.get(
        "source_disparity_fingerprint"
    ) != canonical_metadata.get("fingerprint"):
        return (
            _stage(
                "disparity_stabilized",
                (stab_dir,),
                "invalidate",
                "base disparity fingerprint changed",
            ),
            None,
        )
    pngs = sorted(p.name for p in stab_dir.glob("*.png"))
    if pngs != expected_names:
        return (
            _stage(
                "disparity_stabilized",
                (stab_dir,),
                "invalidate",
                "stabilized disparity frame set incomplete",
            ),
            None,
        )
    return (
        _stage(
            "disparity_stabilized",
            (stab_dir,),
            "preserve",
            f"{len(expected_names)} stabilized disparity maps valid",
        ),
        metadata,
    )


def _validate_downstream_stage(
    name: str,
    paths: tuple[Path, ...],
    expected_names: list[str],
    upstream_ok: bool,
    saved_settings: dict[str, Any],
    migrated_settings: dict[str, Any],
    keys: tuple[str, ...],
) -> tuple[ResumeStage, bool]:
    if not any(_has_payload(p) for p in paths):
        return _stage(name, paths, "missing", f"{name} not generated"), False
    if not upstream_ok:
        return _stage(name, paths, "invalidate", "upstream stage invalidated"), False
    if any(saved_settings.get(k) != migrated_settings.get(k) for k in keys):
        return _stage(name, paths, "invalidate", f"{name} settings changed"), False
    if all(p.is_dir() and sorted(x.name for x in p.glob("*.png")) == expected_names for p in paths):
        return _stage(name, paths, "preserve", f"{name} outputs valid"), True
    return _stage(name, paths, "invalidate", f"{name} outputs incomplete"), False


def build_resume_report(  # noqa: C901
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
    settings_artifact: ProcessingSettingsArtifactV5 | None = None
    if selected_settings_file is None:
        settings_data = None
    elif selected_settings_file.name == SETTINGS_ARTIFACT_RELATIVE_PATH.name:
        raw_bytes = selected_settings_file.read_bytes()
        if len(raw_bytes) > SETTINGS_ARTIFACT_MAX_RAW_BYTES:
            raise ValueError("settings artifact exceeds maximum size")
        settings_artifact = ProcessingSettingsArtifactV5.parse(raw_bytes)
        settings_data = settings_artifact.to_object()
    else:
        settings_data = _read_json(selected_settings_file)

    raw_saved = (settings_data or {}).get("processing_settings", {})
    if not isinstance(raw_saved, dict):
        raw_saved = {}
    settings_metadata = (settings_data or {}).get("metadata", {})
    saved_version = (
        settings_metadata.get("settings_schema_version")
        if isinstance(settings_metadata, dict)
        else None
    )
    saved_result = parse_saved_processing_settings(raw_saved, saved_version=saved_version)
    removed = saved_result.removed_settings
    saved_settings = saved_result.settings
    migrated_settings = validate_settings(current_settings, source="explicit")
    settings_schema_current = not saved_result.migrated
    geometry_mode = str(migrated_settings.get("stereo_geometry_mode", "relative"))
    if (
        geometry_mode == "metric_camera"
        and migrated_settings.get("temporal_postprocessor") != "off"
    ):
        raise ValueError("VDPP is available only with relative stereo geometry")

    source_reason = _source_video_mismatch_reason(settings_data, source_video)
    if source_reason is not None:
        raise ValueError(f"Cannot resume: {source_reason}")

    frames, frame_files, _source_fp = _validate_frame_stage(root, settings_data, migrated_settings)
    frames_reusable = frames.disposition == "preserve"
    scene, _manifest, _bounds = _validate_scene_stage(root, frame_files, frames_reusable)
    scene_reusable = scene.disposition in {"preserve", "resume"}
    raw, raw_metadata, _raw_completed = _validate_raw_stage(
        root, frame_files, scene_reusable, model_fingerprint
    )
    canonical, canonical_metadata = _validate_canonical_stage(root, frame_files, raw_metadata)
    metric, metric_metadata = _validate_metric_stage(root, frame_files, frames_reusable)

    stages: list[ResumeStage] = [frames, scene, raw, canonical, metric]
    selected_stage = canonical if geometry_mode == "relative" else metric
    render_metadata = canonical_metadata if geometry_mode == "relative" else metric_metadata
    render_reusable = selected_stage.disposition in {"preserve", "resume"}

    if geometry_mode == "relative" and migrated_settings.get("temporal_postprocessor") == "vdpp":
        stabilized, stabilized_metadata = _validate_stabilized_stage(
            root, frame_files, canonical_metadata
        )
        stages.append(stabilized)
        render_metadata = stabilized_metadata
        render_reusable = stabilized.disposition == "preserve"

    # Flag legacy directories if present
    for legacy_name, legacy_paths, legacy_reason in (
        ("legacy_supersampled", (root / "01_supersampled_frames",), "legacy supersampled frames"),
        ("legacy_depth_maps", (root / "02_depth_maps",), "legacy depth PNGs cannot be migrated"),
        (
            "legacy_final",
            (root / "08_left_final", root / "08_right_final"),
            "legacy final-eye directories",
        ),
    ):
        if any(_has_payload(p) for p in legacy_paths):
            stages.append(_stage(legacy_name, legacy_paths, "invalidate", legacy_reason))

    # Validate stereo & downstream stages quickly via metadata + filenames
    expected_names = [p.name for p in frame_files]
    stereo_paths = (root / "04_left_frames", root / "04_right_frames")
    stereo_meta = _read_json(stereo_paths[0] / "metadata.json")
    stereo_mode = str(migrated_settings.get("stereo_render_mode", "fast"))
    expected_algo = (
        QUALITY_STEREO_ALGORITHM_VERSION
        if stereo_mode == "quality"
        else STEREO_STAGE_ALGORITHM_VERSION
    )
    stereo_ok = False
    if not any(_has_payload(p) for p in stereo_paths):
        stereo_stage = _stage("stereo", stereo_paths, "missing", "stereo pairs not generated")
    elif not render_reusable or stereo_meta is None:
        stereo_stage = _stage(
            "stereo", stereo_paths, "invalidate", "stereo stage stale or missing metadata"
        )
    elif (
        stereo_meta.get("algorithm_version") != expected_algo
        or stereo_meta.get("source_fingerprint") != (render_metadata or {}).get("fingerprint")
        or stereo_meta.get("occlusion_fill") != migrated_settings.get("occlusion_fill")
    ):
        stereo_stage = _stage(
            "stereo", stereo_paths, "invalidate", "stereo parameters or upstream changed"
        )
    else:
        left_count = sum(1 for _ in stereo_paths[0].glob("*.png"))
        right_count = sum(1 for _ in stereo_paths[1].glob("*.png"))
        if left_count == len(expected_names) and right_count == len(expected_names):
            stereo_stage = _stage(
                "stereo", stereo_paths, "preserve", f"{left_count} stereo pairs valid"
            )
            stereo_ok = True
        elif left_count > 0 and right_count > 0:
            stereo_stage = _stage(
                "stereo",
                stereo_paths,
                "resume",
                f"{min(left_count, right_count)}/{len(expected_names)} stereo pairs present",
            )
        else:
            stereo_stage = _stage("stereo", stereo_paths, "invalidate", "stereo outputs incomplete")

    dist_stage, dist_ok = _validate_downstream_stage(
        "distortion",
        (root / "05_left_distorted", root / "05_right_distorted"),
        expected_names,
        stereo_ok,
        saved_settings,
        migrated_settings,
        _DISTORTION_KEYS,
    )
    crop_stage, crop_ok = _validate_downstream_stage(
        "crop",
        (root / "06_left_cropped", root / "06_right_cropped"),
        expected_names,
        dist_ok or stereo_ok,
        saved_settings,
        migrated_settings,
        _CROP_KEYS,
    )
    upscale_stage, upscale_ok = _validate_downstream_stage(
        "upscale",
        (root / "07_left_upscaled", root / "07_right_upscaled"),
        expected_names,
        crop_ok,
        saved_settings,
        migrated_settings,
        _UPSCALE_KEYS,
    )
    vr_stage, _vr_ok = _validate_downstream_stage(
        "vr_frames",
        (root / "99_vr_frames",),
        expected_names,
        upscale_ok or crop_ok,
        saved_settings,
        migrated_settings,
        _VR_KEYS,
    )
    stages.extend([stereo_stage, dist_stage, crop_stage, upscale_stage, vr_stage])

    backup_required = selected_settings_file is not None and (
        bool(removed) or not settings_schema_current
    )
    final_video_valid: bool | None = None
    if (
        settings_artifact is not None
        and settings_data is not None
        and settings_data["metadata"]["processing_status"] == "completed"
    ):
        rel = settings_data["output_info"].get("expected_final_relative_path")
        if isinstance(rel, str) and rel:
            final_video_valid = verify_existing_final_video(root / rel)

    return ResumeReport(
        output_dir=root,
        stages=tuple(stages),
        removed_settings=removed,
        settings_file=selected_settings_file,
        original_settings_data=settings_data,
        migrated_settings=migrated_settings,
        settings_backup_required=backup_required,
        final_video_valid=final_video_valid,
    )


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
    if (
        report.settings_file is None
        or report.original_settings_data is None
        or not report.settings_backup_required
    ):
        return
    backup = report.output_dir / "settings.legacy.json"
    if not backup.exists():
        atomic_write_bytes(backup, report.settings_file.read_bytes(), sync=True)
    fixed = report.output_dir / SETTINGS_ARTIFACT_RELATIVE_PATH
    if fixed.is_file():
        return
    migration_now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    artifact = migrate_processing_settings_artifact_v5(
        legacy_settings=report.original_settings_data,
        migrated_processing_settings=report.migrated_settings,
        migration_now_usec=migration_now,
        job_id=secrets.token_hex(16),
        job_reservation_generation=secrets.token_hex(16),
        job_terminal_reservation_id=secrets.token_hex(16),
    )
    atomic_write_bytes(fixed, artifact.encode(), sync=True)


def apply_legacy_migration(report: ResumeReport, mode: MigrationMode) -> None:  # noqa: C901
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
        _migrate_settings(report)
        for source in report.migration_paths:
            destination = destinations[source]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            moved.append((source, destination))
    except Exception:
        for source, destination in reversed(moved):
            if destination.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(destination), str(source))
        if original_settings is not None and report.settings_file is not None:
            atomic_write_bytes(report.settings_file, original_settings, sync=True)
        raise

    if mode == "delete" and moved:
        try:
            shutil.rmtree(report.output_dir / ".resume_delete_staging")
        except OSError as error:
            warnings.warn(
                f"Legacy migration committed; delete-staging cleanup remains pending: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
