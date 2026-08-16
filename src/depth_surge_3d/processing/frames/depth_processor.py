"""
Depth map processing module.

Handles depth map generation with canonical caching, memory management, and chunking.
"""

from __future__ import annotations

import cv2
import hashlib
import json
import numpy as np
import shutil
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from ...core.constants import (
    DEFAULT_FALLBACK_FPS,
    RESOLUTION_4K,
    RESOLUTION_1440P,
    RESOLUTION_1080P,
    RESOLUTION_720P,
    CHUNK_SIZE_4K,
    CHUNK_SIZE_1440P,
    CHUNK_SIZE_1080P_MANUAL,
    CHUNK_SIZE_720P,
    CHUNK_SIZE_SMALL,
    PREVIEW_FRAME_SAMPLE_RATE,
)
from ...utils.domain.resolution import resolve_depth_input_size
from ...core.depth_contract import (
    CANONICAL_DEPTH_ALGORITHM_VERSION,
    CANONICAL_DEPTH_SCHEMA_VERSION,
    canonical_json_hash,
)
from ...utils import (
    get_cached_depth_map_files,
    save_depth_map_files_to_cache,
)
from ...utils import calculate_optimal_chunk_size, get_vram_info
from ...inference.depth.types import DepthBatch, DepthRepresentation
from .depth_normalizer import (
    SceneDepthBounds,
    canonicalize_depth,
    encode_canonical_png,
)
from .depth_storage import (
    RawDepthFingerprintError,
    RawDepthStore,
    build_current_model_fingerprint,
    depth_preprocessing_algorithm,
    estimate_depth_disk_bytes,
    estimate_raw_depth_only_bytes,
    require_disk_space,
)
from .frame_stage_parallelism import (
    calculate_frame_stage_workers,
    png_headers_match,
    run_ordered_frame_tasks,
)
from .scene_analyzer import (
    SCENE_ALGORITHM_VERSION,
    SCENE_SCHEMA_VERSION,
    analyze_scenes,
    finalize_scenes,
    sample_scene_depths,
)
from .source_frame_manifest import frame_sequence_fingerprint
from .metric_geometry import (
    MetricGeometryStore,
    estimate_metric_geometry_disk_bytes,
    filesystem_allocation_unit,
    metric_frame_from_depth,
    require_metric_geometry_disk_space,
    sample_clip_convergence,
)


DEPTH_BOUNDS_SCHEMA_VERSION = 2
DEPTH_SAMPLES_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class _RawStageContext:
    scene_dir: Path
    raw_dir: Path
    canonical_dir: Path
    metric_dir: Path
    manifest: dict[str, Any]
    raw_store: RawDepthStore
    raw_files: tuple[Path, ...]
    frame_names: tuple[str, ...]
    semantic_fingerprint: dict[str, Any]
    candidate_scene_fingerprint: str


class DepthMapProcessor:
    """
    Depth map generation with canonical caching and memory management.

    Responsibilities:
    - Depth map generation with model inference
    - VRAM-based chunk sizing
    - Canonical file cache management
    - GPU memory management
    - Batch and chunked processing strategies
    """

    def __init__(self, depth_estimator, verbose: bool = False):
        """
        Initialize depth map processor.

        Args:
            depth_estimator: Depth estimation model instance
            verbose: Enable verbose output
        """
        self.depth_estimator = depth_estimator
        self.verbose = verbose

    def generate_depth_map_files(  # noqa: C901
        self,
        frame_files: list[Path],
        settings: dict[str, Any],
        directories: dict[str, Path],
        progress_tracker,
    ) -> list[Path] | None:
        """Build native raw depth and exactly the selected stage-3 geometry."""
        if not frame_files:
            return None

        scene_dir, raw_dir, canonical_dir, metric_dir = self._resolve_depth_directories(
            frame_files, directories
        )
        for directory in (scene_dir, raw_dir, canonical_dir, metric_dir):
            directory.mkdir(parents=True, exist_ok=True)

        semantic_fingerprint = self._raw_semantic_fingerprint(frame_files, settings)
        source_frame_fingerprint = str(semantic_fingerprint["source_frame_fingerprint"])
        geometry_mode = str(settings.get("stereo_geometry_mode", "relative"))
        model_fingerprint = canonical_json_hash(semantic_fingerprint)
        cache_settings = dict(settings, model_fingerprint=model_fingerprint)
        video_path = settings.get("video_path")

        manifest = self._load_or_analyze_scenes(
            frame_files,
            scene_dir,
            raw_dir,
            canonical_dir,
            metric_dir,
            settings,
            source_frame_fingerprint,
        )
        frame_names = [path.name for path in frame_files]
        candidate_scene_fingerprint = self._candidate_scene_fingerprint(manifest)
        selected = self._reusable_selected_geometry_files(
            settings=settings,
            directories={
                **directories,
                "depth_raw": raw_dir,
                "metric_geometry": metric_dir,
            },
            frame_names=frame_names,
            source_frame_fingerprint=source_frame_fingerprint,
            semantic_fingerprint=semantic_fingerprint,
            candidate_scene_fingerprint=candidate_scene_fingerprint,
        )
        if selected is not None:
            if not settings.get("keep_intermediates", False):
                raw_metadata = RawDepthStore.read_metadata(raw_dir)
                if raw_metadata is not None:
                    self._remove_raw_payloads(RawDepthStore(raw_dir, raw_metadata))
            self._report_file_cache_hit(
                selected,
                progress_tracker,
                "validated metric geometry stage",
            )
            return selected

        requested_dtype = str(settings.get("raw_storage_dtype", "auto"))

        try:
            raw_store = self._open_raw_store_if_present(
                raw_dir,
                frame_names=frame_names,
                semantic_fingerprint=semantic_fingerprint,
                requested_dtype=requested_dtype,
            )
        except (RawDepthFingerprintError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._reset_stage_directory(raw_dir)
            self._reset_stage_directory(canonical_dir)
            self._reset_stage_directory(metric_dir)
            raw_store = None

        if geometry_mode == "relative":
            restored = self._try_restore_global_canonical_cache(
                video_path,
                frame_files,
                cache_settings,
                canonical_dir,
                progress_tracker,
            )
            if restored is not None:
                if raw_store is not None and not settings.get("keep_intermediates", False):
                    self._remove_raw_payloads(raw_store)
                return restored

            final_state = self._load_final_scene_state(
                scene_dir,
                manifest,
                raw_store.metadata["fingerprint"] if raw_store is not None else None,
            )
            existing = self._try_reuse_local_canonical_stage(
                raw_store,
                final_state,
                frame_names,
                canonical_dir,
                bool(settings.get("keep_intermediates", False)),
                progress_tracker,
            )
            if existing is not None:
                return existing

        self._report_geometry_generation_start(len(frame_files), geometry_mode, progress_tracker)
        context = self._prepare_raw_stage(
            frame_files,
            settings,
            scene_dir=scene_dir,
            raw_dir=raw_dir,
            canonical_dir=canonical_dir,
            metric_dir=metric_dir,
            manifest=manifest,
            semantic_fingerprint=semantic_fingerprint,
            candidate_scene_fingerprint=candidate_scene_fingerprint,
            raw_store=raw_store,
            progress_tracker=progress_tracker,
        )

        if geometry_mode == "metric_camera":
            geometry_files = self._write_metric_geometry_stage(context, settings, progress_tracker)
        else:
            final_manifest, bounds, bounds_payload = self._finalize_scene_stage(
                scene_dir,
                manifest,
                context.raw_store,
                list(context.raw_files),
            )

            expected_metadata = self._canonical_metadata(
                frame_names,
                context.raw_store,
                final_manifest,
                bounds_payload,
            )
            geometry_files = self._write_canonical_stage(
                context.raw_store,
                list(context.raw_files),
                frame_files,
                final_manifest,
                bounds,
                canonical_dir,
                expected_metadata,
                progress_tracker,
            )

        if not settings.get("keep_intermediates", False):
            self._remove_raw_payloads(context.raw_store)

        if (
            geometry_mode == "relative"
            and video_path
            and save_depth_map_files_to_cache(str(video_path), cache_settings, geometry_files)
        ):
            print("  Canonical disparity maps saved to global cache")
        return geometry_files

    def _try_restore_global_canonical_cache(
        self,
        video_path: Any,
        frame_files: list[Path],
        cache_settings: dict[str, Any],
        canonical_dir: Path,
        progress_tracker,
    ) -> list[Path] | None:
        if not video_path:
            return None
        cached_files = get_cached_depth_map_files(
            str(video_path),
            cache_settings,
            len(frame_files),
        )
        if cached_files is None:
            return None
        restored = self._restore_cached_canonical_stage(cached_files, frame_files, canonical_dir)
        self._report_file_cache_hit(
            restored,
            progress_tracker,
            "global canonical disparity cache",
        )
        return restored

    def _try_reuse_local_canonical_stage(
        self,
        raw_store: RawDepthStore | None,
        final_state: tuple[dict[str, Any], dict[int, SceneDepthBounds], dict[str, Any]] | None,
        frame_names: list[str],
        canonical_dir: Path,
        keep_intermediates: bool,
        progress_tracker,
    ) -> list[Path] | None:
        if raw_store is None or final_state is None:
            return None
        final_manifest, _bounds, bounds_payload = final_state
        expected_metadata = self._canonical_metadata(
            frame_names,
            raw_store,
            final_manifest,
            bounds_payload,
        )
        existing = self._validated_canonical_files(canonical_dir, expected_metadata)
        if existing is None:
            return None
        if not keep_intermediates:
            self._remove_raw_payloads(raw_store)
        self._report_file_cache_hit(
            existing,
            progress_tracker,
            "validated canonical disparity stage",
        )
        return existing

    def _reusable_selected_geometry_files(
        self,
        *,
        settings: Mapping[str, Any],
        directories: Mapping[str, Path],
        frame_names: Sequence[str],
        source_frame_fingerprint: str,
        semantic_fingerprint: Mapping[str, Any],
        candidate_scene_fingerprint: str,
    ) -> list[Path] | None:
        if settings.get("stereo_geometry_mode", "relative") != "metric_camera":
            return None
        raw_metadata = RawDepthStore.read_metadata(directories["depth_raw"])
        if raw_metadata is None or raw_metadata.get("storage_status") != "ready":
            return None
        persisted_semantic = raw_metadata.get("semantic_fingerprint")
        if not isinstance(persisted_semantic, dict):
            return None
        if canonical_json_hash(persisted_semantic) != canonical_json_hash(semantic_fingerprint):
            return None
        try:
            store = MetricGeometryStore.open_existing(
                directories["metric_geometry"],
                frame_names=list(frame_names),
                source_frame_fingerprint=source_frame_fingerprint,
                source_raw_fingerprint=str(raw_metadata["fingerprint"]),
                candidate_scene_fingerprint=candidate_scene_fingerprint,
            )
        except (OSError, KeyError, TypeError, ValueError):
            return None
        return list(store.complete_files)

    @staticmethod
    def _report_geometry_generation_start(
        frame_count: int, geometry_mode: str, progress_tracker
    ) -> None:
        label = (
            "metric geometry frames"
            if geometry_mode == "metric_camera"
            else "canonical disparity maps"
        )
        print(f"Step 2/7: Generating {label}...")
        print("  Using restartable scene, raw-depth, and selected geometry stages...")
        if progress_tracker:
            progress_tracker.update_progress(
                "Generating depth maps",
                phase="depth_estimation",
                frame_num=0,
                step_name="Depth Map Generation",
                step_progress=0,
                step_total=frame_count,
            )

    def _finalize_scene_stage(
        self,
        scene_dir: Path,
        manifest: dict[str, Any],
        raw_store: RawDepthStore,
        raw_files: list[Path],
    ) -> tuple[dict[str, Any], dict[int, SceneDepthBounds], dict[str, Any]]:
        source_raw_fingerprint = str(raw_store.metadata["fingerprint"])
        final_state = self._load_final_scene_state(
            scene_dir,
            manifest,
            source_raw_fingerprint,
        )
        if final_state is not None:
            return final_state
        candidate_manifest = self._candidate_manifest(manifest)
        samples, sample_fingerprint = self._load_or_create_scene_samples(
            scene_dir,
            raw_files,
            candidate_manifest,
            DepthRepresentation(raw_store.metadata["representation"]),
            source_raw_fingerprint,
        )
        final_manifest, bounds = finalize_scenes(candidate_manifest, samples)
        bounds_payload = self._write_depth_bounds(
            scene_dir,
            bounds,
            sample_fingerprint,
            source_raw_fingerprint,
        )
        final_manifest["sample_fingerprint"] = sample_fingerprint
        final_manifest["bounds_fingerprint"] = bounds_payload["fingerprint"]
        self._atomic_write_json(scene_dir / "scene_manifest.json", final_manifest)
        return final_manifest, bounds, bounds_payload

    @staticmethod
    def _resolve_depth_directories(
        frame_files: list[Path], directories: dict[str, Path]
    ) -> tuple[Path, Path, Path, Path]:
        canonical_dir = directories.get("disparity_maps")
        base_dir = directories.get("base")
        if base_dir is None:
            base_dir = (
                canonical_dir.parent if canonical_dir is not None else frame_files[0].parent.parent
            )
        return (
            directories.get("scene_data", base_dir / "01_scene_data"),
            directories.get("depth_raw", base_dir / "02_depth_raw"),
            canonical_dir or base_dir / "03_disparity_maps",
            directories.get("metric_geometry", base_dir / "03_metric_geometry"),
        )

    @staticmethod
    def _scene_settings(settings: dict[str, Any]) -> dict[str, Any]:
        return {
            "enabled": bool(settings.get("scene_detection", True)),
            "threshold": float(settings.get("scene_cut_threshold", 0.55)),
            "min_frames": int(settings.get("min_scene_frames", 8)),
        }

    def _load_or_analyze_scenes(
        self,
        frame_files: list[Path],
        scene_dir: Path,
        raw_dir: Path,
        canonical_dir: Path,
        metric_dir: Path,
        settings: dict[str, Any],
        source_frame_fingerprint: str,
    ) -> dict[str, Any]:
        manifest_path = scene_dir / "scene_manifest.json"
        expected_names = [path.name for path in frame_files]
        scene_settings = self._scene_settings(settings)
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            identity_valid = (
                manifest.get("schema_version") == SCENE_SCHEMA_VERSION
                and manifest.get("algorithm_version") == SCENE_ALGORITHM_VERSION
                and manifest.get("status") in {"candidate", "final"}
                and manifest.get("frame_names") == expected_names
                and manifest.get("source_frame_fingerprint") == source_frame_fingerprint
                and len(manifest.get("scene_ids", [])) == len(frame_files)
            )
            if identity_valid and manifest.get("settings") == scene_settings:
                return manifest
            self._reset_stage_directory(scene_dir)
            if not identity_valid:
                self._reset_stage_directory(raw_dir)
                self._reset_stage_directory(canonical_dir)
                self._reset_stage_directory(metric_dir)
        manifest = analyze_scenes(frame_files, scene_dir, **scene_settings)
        manifest["source_frame_fingerprint"] = source_frame_fingerprint
        self._atomic_write_json(scene_dir / "scene_manifest.json", manifest)
        return manifest

    def _candidate_scene_fingerprint(self, manifest: dict[str, Any]) -> str:
        candidate_manifest = self._candidate_manifest(manifest)
        return canonical_json_hash(
            {
                "schema_version": candidate_manifest["schema_version"],
                "algorithm_version": candidate_manifest["algorithm_version"],
                "frame_names": candidate_manifest["frame_names"],
                "scene_ids": candidate_manifest["scene_ids"],
            }
        )

    @staticmethod
    def _reset_stage_directory(directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    @staticmethod
    def _source_frame_fingerprint(frame_files: list[Path]) -> str:
        return frame_sequence_fingerprint(frame_files)

    def _raw_semantic_fingerprint(
        self, frame_files: list[Path], settings: dict[str, Any]
    ) -> dict[str, Any]:
        fingerprint = build_current_model_fingerprint(self.depth_estimator, settings)
        fingerprint["source_frame_fingerprint"] = self._source_frame_fingerprint(frame_files)
        fingerprint["preprocessing_algorithm"] = depth_preprocessing_algorithm(settings)
        camera_model = str(getattr(self.depth_estimator, "camera_model", "none"))
        fingerprint["camera_model"] = camera_model
        return fingerprint

    @staticmethod
    def _open_raw_store_if_present(
        raw_dir: Path,
        *,
        frame_names: list[str],
        semantic_fingerprint: dict[str, Any],
        requested_dtype: str,
    ) -> RawDepthStore | None:
        metadata = RawDepthStore.read_metadata(raw_dir)
        if metadata is None:
            if (raw_dir / "metadata.json").exists():
                raise RawDepthFingerprintError("Raw-depth metadata is malformed")
            return None
        return RawDepthStore.open_existing(
            raw_dir,
            frame_names=frame_names,
            semantic_fingerprint=semantic_fingerprint,
            requested_dtype=requested_dtype,
        )

    def _estimate_native_shape(
        self, frame_width: int, frame_height: int, input_size: int
    ) -> tuple[int, int]:
        estimator_method = getattr(type(self.depth_estimator), "estimate_output_shape", None)
        if callable(estimator_method):
            shape = self.depth_estimator.estimate_output_shape(
                frame_width,
                frame_height,
                input_size,
            )
            if (
                not isinstance(shape, tuple)
                or len(shape) != 2
                or any(not isinstance(value, int) or value < 1 for value in shape)
            ):
                raise ValueError("Estimator output shape must be two positive integers")
            return shape
        longest = max(frame_width, frame_height)
        scale = min(1.0, input_size / longest)
        return (
            max(1, int(round(frame_height * scale))),
            max(1, int(round(frame_width * scale))),
        )

    def _prepare_raw_stage(
        self,
        frame_files: list[Path],
        settings: dict[str, Any],
        *,
        scene_dir: Path,
        raw_dir: Path,
        canonical_dir: Path,
        metric_dir: Path,
        manifest: dict[str, Any],
        semantic_fingerprint: dict[str, Any],
        candidate_scene_fingerprint: str,
        raw_store: RawDepthStore | None,
        progress_tracker,
    ) -> _RawStageContext:
        completed_store = self._complete_raw_depth_stage(
            frame_files,
            settings,
            raw_dir,
            metric_dir,
            semantic_fingerprint,
            raw_store,
            progress_tracker,
        )
        frame_names = tuple(path.name for path in frame_files)
        raw_files = tuple(completed_store.path_for(name) for name in frame_names)
        if not all(path.is_file() for path in raw_files):
            raise RuntimeError("Raw-depth global barrier was reached with missing frames")
        return _RawStageContext(
            scene_dir=scene_dir,
            raw_dir=raw_dir,
            canonical_dir=canonical_dir,
            metric_dir=metric_dir,
            manifest=manifest,
            raw_store=completed_store,
            raw_files=raw_files,
            frame_names=frame_names,
            semantic_fingerprint=semantic_fingerprint,
            candidate_scene_fingerprint=candidate_scene_fingerprint,
        )

    def _complete_raw_depth_stage(
        self,
        frame_files: list[Path],
        settings: dict[str, Any],
        raw_dir: Path,
        metric_dir: Path,
        semantic_fingerprint: dict[str, Any],
        raw_store: RawDepthStore | None,
        progress_tracker,
    ) -> RawDepthStore:
        (
            chunk_size,
            input_size,
            target_fps,
            requested_dtype,
            estimated_shape,
            estimated_storage_bytes,
        ) = self._prepare_raw_depth_stage(frame_files, settings, raw_dir, metric_dir, raw_store)
        frame_names = [path.name for path in frame_files]
        total_chunks = (len(frame_files) + chunk_size - 1) // chunk_size
        for chunk_start in range(0, len(frame_files), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(frame_files))
            chunk_files = frame_files[chunk_start:chunk_end]
            chunk_names = frame_names[chunk_start:chunk_end]
            if raw_store is not None and all(
                raw_store.path_for(name).is_file() for name in chunk_names
            ):
                continue
            raw_store = self._infer_raw_chunk(
                chunk_files,
                chunk_names,
                frame_names,
                settings,
                raw_dir,
                metric_dir,
                semantic_fingerprint,
                raw_store,
                requested_dtype,
                input_size,
                target_fps,
                estimated_shape,
                estimated_storage_bytes,
            )
            self._report_raw_chunk_progress(
                progress_tracker,
                chunk_start // chunk_size + 1,
                total_chunks,
                chunk_end,
                len(frame_files),
            )

        if raw_store is None:
            raise RuntimeError("Depth estimator produced no raw depth")
        raw_store.flush_metadata()
        return raw_store

    def _prepare_raw_depth_stage(
        self,
        frame_files: list[Path],
        settings: dict[str, Any],
        raw_dir: Path,
        metric_dir: Path,
        raw_store: RawDepthStore | None,
    ) -> tuple[int, int, Any, str, tuple[int, int], int]:
        sample_frame = cv2.imread(str(frame_files[0]))
        if sample_frame is None:
            raise OSError(f"Could not load source frame: {frame_files[0]}")
        frame_height, frame_width = sample_frame.shape[:2]
        chunk_size, input_size = self._determine_chunk_params(
            frame_width,
            frame_height,
            settings.get("depth_resolution", "auto"),
        )
        requested_dtype = str(settings.get("raw_storage_dtype", "auto"))
        storage_bytes = 2 if requested_dtype == "float16" else 4
        estimated_height, estimated_width = self._estimate_native_shape(
            frame_width, frame_height, input_size
        )
        estimated_shape = (estimated_height, estimated_width)
        if settings.get("stereo_geometry_mode", "relative") == "metric_camera":
            if raw_store is not None:
                native_height, native_width = raw_store.metadata["native_shape"]
                estimated_shape = (int(native_height), int(native_width))
                storage_bytes = 2 if raw_store.metadata["selected_dtype"] == "float16" else 4
            missing_raw_count = len(frame_files) - (
                len(raw_store.complete_files) if raw_store is not None else 0
            )
            self._require_metric_pipeline_disk_space(
                raw_dir,
                metric_dir,
                frame_count=len(frame_files),
                missing_raw_count=missing_raw_count,
                native_shape=estimated_shape,
                storage_bytes=storage_bytes,
            )
        else:
            self._require_depth_disk_space(
                raw_dir,
                len(frame_files),
                estimated_height,
                estimated_width,
                storage_bytes,
                bool(settings.get("keep_intermediates", False)),
            )

        if (
            raw_store is not None
            and settings.get("stereo_geometry_mode", "relative") != "metric_camera"
        ):
            native_height, native_width = raw_store.metadata["native_shape"]
            selected_bytes = 2 if raw_store.metadata["selected_dtype"] == "float16" else 4
            self._require_depth_disk_space(
                raw_dir,
                len(frame_files),
                int(native_height),
                int(native_width),
                selected_bytes,
                bool(settings.get("keep_intermediates", False)),
            )

        self._clear_gpu_memory()
        target_fps = settings.get("target_fps", DEFAULT_FALLBACK_FPS)
        if target_fps is None or str(target_fps) in {"None", "original"}:
            target_fps = 30
        return (
            chunk_size,
            input_size,
            target_fps,
            requested_dtype,
            estimated_shape,
            storage_bytes,
        )

    @staticmethod
    def _require_metric_pipeline_disk_space(
        raw_dir: Path,
        metric_dir: Path,
        *,
        frame_count: int,
        missing_raw_count: int,
        native_shape: tuple[int, int],
        storage_bytes: int,
    ) -> None:
        native_height, native_width = native_shape
        raw_required = estimate_raw_depth_only_bytes(
            frame_count=missing_raw_count,
            native_width=native_width,
            native_height=native_height,
            storage_bytes=storage_bytes,
            camera_bytes_per_frame=4,
        )
        metric_required = estimate_metric_geometry_disk_bytes(
            [native_shape] * frame_count,
            allocation_unit=filesystem_allocation_unit(metric_dir),
        )
        require_disk_space(raw_dir, raw_required + metric_required)

    @staticmethod
    def _require_depth_disk_space(
        raw_dir: Path,
        frame_count: int,
        native_height: int,
        native_width: int,
        storage_bytes: int,
        keep_intermediates: bool,
    ) -> None:
        required = estimate_depth_disk_bytes(
            frame_count=frame_count,
            native_width=native_width,
            native_height=native_height,
            storage_bytes=storage_bytes,
            keep_intermediates=keep_intermediates,
        )
        require_disk_space(raw_dir, required)

    def _infer_raw_chunk(
        self,
        chunk_files: list[Path],
        chunk_names: list[str],
        frame_names: list[str],
        settings: dict[str, Any],
        raw_dir: Path,
        metric_dir: Path,
        semantic_fingerprint: dict[str, Any],
        raw_store: RawDepthStore | None,
        requested_dtype: str,
        input_size: int,
        target_fps: Any,
        estimated_shape: tuple[int, int],
        estimated_storage_bytes: int,
    ) -> RawDepthStore:
        chunk_frames = self._load_chunk_frames(chunk_files, settings)
        if chunk_frames is None or len(chunk_frames) != len(chunk_files):
            raise OSError("Not all source frames could be loaded for depth inference")
        result = self.depth_estimator.estimate_depth_batch(
            np.asarray(chunk_frames),
            target_fps=target_fps,
            input_size=input_size,
            fp32=False,
        )
        if not isinstance(result, DepthBatch):
            raise TypeError("File-backed depth inference requires a DepthBatch result")
        if len(result.values) != len(chunk_files):
            raise ValueError("Depth estimator returned an unexpected frame count")

        if raw_store is None:
            created_store = True
            raw_store = RawDepthStore.create(
                raw_dir,
                frame_names=frame_names,
                semantic_fingerprint=semantic_fingerprint,
                requested_dtype=requested_dtype,
                first_batch=result,
            )
            native_height, native_width = result.values.shape[1:]
            selected_bytes = 2 if raw_store.metadata["selected_dtype"] == "float16" else 4
            if settings.get("stereo_geometry_mode", "relative") != "metric_camera":
                self._require_depth_disk_space(
                    raw_dir,
                    len(frame_names),
                    int(native_height),
                    int(native_width),
                    selected_bytes,
                    bool(settings.get("keep_intermediates", False)),
                )
        else:
            created_store = False
            raw_store.validate_batch_contract(result)

        raw_store.write_batch(chunk_names, result)
        if created_store and settings.get("stereo_geometry_mode", "relative") == "metric_camera":
            actual_shape = tuple(int(value) for value in raw_store.metadata["native_shape"])
            selected_bytes = 2 if raw_store.metadata["selected_dtype"] == "float16" else 4
            if actual_shape != estimated_shape or selected_bytes != estimated_storage_bytes:
                self._require_metric_pipeline_disk_space(
                    raw_dir,
                    metric_dir,
                    frame_count=len(frame_names),
                    missing_raw_count=len(frame_names) - len(raw_store.complete_files),
                    native_shape=cast(tuple[int, int], actual_shape),
                    storage_bytes=selected_bytes,
                )
        return raw_store

    def _write_metric_geometry_stage(
        self,
        context: _RawStageContext,
        settings: dict[str, Any],
        progress_tracker,
    ) -> list[Path]:
        del settings
        native_shape = tuple(int(value) for value in context.raw_store.metadata["native_shape"])
        required = estimate_metric_geometry_disk_bytes(
            [cast(tuple[int, int], native_shape)] * len(context.frame_names),
            allocation_unit=filesystem_allocation_unit(context.metric_dir),
        )
        require_metric_geometry_disk_space(context.metric_dir, required)
        store = self._open_or_reset_metric_store(context, native_shape, required)
        for index, (name, raw_path) in enumerate(zip(context.frame_names, context.raw_files)):
            output = store.path_for(name)
            if output.is_file():
                frame = store.load(output)
            else:
                batch = context.raw_store.load_batch([raw_path])
                if (
                    batch.representation is not DepthRepresentation.METRIC_DEPTH
                    or batch.camera is None
                ):
                    raise ValueError(
                        "metric_camera requires metric raw depth with pinhole focal data"
                    )
                frame = metric_frame_from_depth(batch.values[0], batch.camera.focal_x_normalized[0])
                store.write_frame(name, frame)
            self._report_metric_progress(
                progress_tracker,
                index,
                len(context.frame_names),
                frame.inverse_depth,
                frame.valid,
            )
        convergence = sample_clip_convergence(
            context.raw_store,
            context.raw_files,
            self._candidate_manifest(context.manifest)["scene_ids"],
        )
        store.finalize(convergence)
        validated = MetricGeometryStore.open_existing(
            context.metric_dir,
            frame_names=list(context.frame_names),
            source_raw_fingerprint=str(context.raw_store.metadata["fingerprint"]),
            source_frame_fingerprint=str(context.semantic_fingerprint["source_frame_fingerprint"]),
            candidate_scene_fingerprint=context.candidate_scene_fingerprint,
        )
        return list(validated.complete_files)

    def _open_or_reset_metric_store(
        self,
        context: _RawStageContext,
        native_shape: tuple[int, ...],
        required: int,
    ) -> MetricGeometryStore:
        def open_store() -> MetricGeometryStore:
            return MetricGeometryStore.open_or_create(
                context.metric_dir,
                frame_names=list(context.frame_names),
                native_shape=cast(tuple[int, int], native_shape),
                source_raw_fingerprint=str(context.raw_store.metadata["fingerprint"]),
                source_frame_fingerprint=str(
                    context.semantic_fingerprint["source_frame_fingerprint"]
                ),
                candidate_scene_fingerprint=context.candidate_scene_fingerprint,
                preflight_required_bytes=required,
            )

        try:
            return open_store()
        except (KeyError, TypeError, ValueError):
            self._reset_stage_directory(context.metric_dir)
            return open_store()

    @staticmethod
    def _metric_depth_preview(inverse_depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
        preview = np.zeros(inverse_depth.shape, dtype=np.uint8)
        finite_valid = valid & np.isfinite(inverse_depth)
        if not np.any(finite_valid):
            return preview
        values = inverse_depth[finite_valid]
        low = np.float32(values.min())
        high = np.float32(values.max())
        if high > low:
            normalized = (values - low) / (high - low)
            preview[finite_valid] = np.rint(normalized * np.float32(255.0)).astype(np.uint8)
        else:
            preview[finite_valid] = np.uint8(255)
        return preview

    @classmethod
    def _report_metric_progress(
        cls,
        progress_tracker,
        index: int,
        frame_count: int,
        inverse_depth: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        if progress_tracker:
            progress_tracker.update_progress(
                f"Deriving metric geometry {index + 1}/{frame_count}",
                phase="depth_estimation",
                frame_num=index + 1,
                step_name="Depth Map Generation",
                step_progress=index + 1,
                step_total=frame_count,
            )
        if progress_tracker and hasattr(progress_tracker, "send_preview_frame_from_array"):
            if index % PREVIEW_FRAME_SAMPLE_RATE == 0 or index == frame_count - 1:
                progress_tracker.send_preview_frame_from_array(
                    cls._metric_depth_preview(inverse_depth, valid),
                    "depth_map",
                    index + 1,
                )

    @staticmethod
    def _report_raw_chunk_progress(
        progress_tracker,
        chunk_number: int,
        total_chunks: int,
        chunk_end: int,
        frame_count: int,
    ) -> None:
        if progress_tracker:
            progress_tracker.update_progress(
                f"Chunk {chunk_number}/{total_chunks}: raw depth {chunk_end}/{frame_count}",
                phase="depth_estimation",
                frame_num=chunk_end,
                step_name="Depth Map Generation",
                step_progress=chunk_end,
                step_total=frame_count,
            )

    @staticmethod
    def _candidate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
        if manifest.get("status") == "candidate":
            return dict(manifest)
        cuts = {int(value) for value in manifest.get("candidate_cuts", [])}
        scene_ids: list[int] = []
        scene_id = 0
        for index in range(len(manifest.get("frame_names", []))):
            if index in cuts:
                scene_id += 1
            scene_ids.append(scene_id)
        candidate = dict(manifest)
        candidate.update({"status": "candidate", "scene_ids": scene_ids})
        for key in ("final_cuts", "bounds_fingerprint", "sample_fingerprint"):
            candidate.pop(key, None)
        return candidate

    @staticmethod
    def _array_payload_fingerprint(arrays: dict[str, np.ndarray]) -> str:
        hasher = hashlib.sha256()
        for key in sorted(arrays):
            values = np.ascontiguousarray(arrays[key])
            hasher.update(key.encode("ascii"))
            hasher.update(values.dtype.str.encode("ascii"))
            hasher.update(json.dumps(values.shape).encode("ascii"))
            hasher.update(values.tobytes())
        return hasher.hexdigest()

    @staticmethod
    def _sample_frame_indexes(scene_ids: list[int], scene_id: int) -> np.ndarray:
        indexes = np.asarray(
            [index for index, value in enumerate(scene_ids) if value == scene_id],
            dtype=np.int64,
        )
        if len(indexes) <= 32:
            return indexes
        selected = np.unique(np.rint(np.linspace(0, len(indexes) - 1, 32)).astype(np.int64))
        return indexes[selected]

    def _load_or_create_scene_samples(
        self,
        scene_dir: Path,
        raw_files: list[Path],
        candidate_manifest: dict[str, Any],
        representation: DepthRepresentation,
        source_raw_fingerprint: str,
    ) -> tuple[dict[int, np.ndarray], str]:
        sample_path = scene_dir / "depth_samples.npz"
        manifest_fingerprint = canonical_json_hash(candidate_manifest)
        if sample_path.is_file():
            try:
                with np.load(sample_path, allow_pickle=False) as payload:
                    cached_arrays = {key: np.asarray(payload[key]) for key in payload.files}
                stored_fingerprint = str(cached_arrays.pop("content_fingerprint").item())
                if (
                    int(cached_arrays["schema_version"].item()) == DEPTH_SAMPLES_SCHEMA_VERSION
                    and str(cached_arrays["algorithm_version"].item()) == SCENE_ALGORITHM_VERSION
                    and str(cached_arrays["manifest_fingerprint"].item()) == manifest_fingerprint
                    and str(cached_arrays["source_raw_fingerprint"].item())
                    == source_raw_fingerprint
                    and self._array_payload_fingerprint(cached_arrays) == stored_fingerprint
                ):
                    scene_ids = list(
                        dict.fromkeys(int(value) for value in candidate_manifest["scene_ids"])
                    )
                    samples = {
                        scene_id: cached_arrays[f"scene_{scene_id}_samples"].astype(np.float32)
                        for scene_id in scene_ids
                    }
                    return samples, stored_fingerprint
            except (KeyError, OSError, TypeError, ValueError):
                pass

        samples = sample_scene_depths(raw_files, candidate_manifest, representation)
        scene_ids = [int(value) for value in candidate_manifest["scene_ids"]]
        arrays: dict[str, np.ndarray] = {
            "schema_version": np.asarray(DEPTH_SAMPLES_SCHEMA_VERSION, dtype=np.int64),
            "algorithm_version": np.asarray(SCENE_ALGORITHM_VERSION, dtype=np.str_),
            "manifest_fingerprint": np.asarray(manifest_fingerprint, dtype=np.str_),
            "source_raw_fingerprint": np.asarray(source_raw_fingerprint, dtype=np.str_),
            "frame_names": np.asarray(candidate_manifest["frame_names"], dtype=np.str_),
            "scene_ids": np.asarray(scene_ids, dtype=np.int64),
        }
        for scene_id, values in samples.items():
            arrays[f"scene_{scene_id}_samples"] = np.asarray(values, dtype=np.float32)
            arrays[f"scene_{scene_id}_frame_indexes"] = self._sample_frame_indexes(
                scene_ids, scene_id
            )
        sample_fingerprint = self._array_payload_fingerprint(arrays)
        arrays["content_fingerprint"] = np.asarray(sample_fingerprint, dtype=np.str_)
        self._atomic_save_npz(sample_path, arrays)
        return samples, sample_fingerprint

    def _write_depth_bounds(
        self,
        scene_dir: Path,
        bounds: dict[int, SceneDepthBounds],
        sample_fingerprint: str,
        source_raw_fingerprint: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": DEPTH_BOUNDS_SCHEMA_VERSION,
            "algorithm_version": CANONICAL_DEPTH_ALGORITHM_VERSION,
            "source_raw_fingerprint": source_raw_fingerprint,
            "sample_fingerprint": sample_fingerprint,
            "scenes": {
                str(scene_id): {"low": value.low, "high": value.high}
                for scene_id, value in sorted(bounds.items())
            },
        }
        payload["fingerprint"] = canonical_json_hash(payload)
        self._atomic_write_json(scene_dir / "depth_bounds.json", payload)
        return payload

    def _load_final_scene_state(
        self,
        scene_dir: Path,
        manifest: dict[str, Any],
        source_raw_fingerprint: str | None,
    ) -> tuple[dict[str, Any], dict[int, SceneDepthBounds], dict[str, Any]] | None:
        if manifest.get("status") != "final" or source_raw_fingerprint is None:
            return None
        bounds_path = scene_dir / "depth_bounds.json"
        if not bounds_path.is_file():
            return None
        try:
            payload = json.loads(bounds_path.read_text(encoding="utf-8"))
            fingerprint = payload.pop("fingerprint")
            payload["fingerprint"] = fingerprint
            unhashed = {key: value for key, value in payload.items() if key != "fingerprint"}
            if (
                payload.get("schema_version") != DEPTH_BOUNDS_SCHEMA_VERSION
                or payload.get("algorithm_version") != CANONICAL_DEPTH_ALGORITHM_VERSION
                or payload.get("source_raw_fingerprint") != source_raw_fingerprint
                or fingerprint != canonical_json_hash(unhashed)
                or manifest.get("bounds_fingerprint") != fingerprint
            ):
                return None
            bounds = {
                int(scene_id): SceneDepthBounds(float(value["low"]), float(value["high"]))
                for scene_id, value in payload["scenes"].items()
            }
            if set(bounds) != set(int(value) for value in manifest["scene_ids"]):
                return None
            return manifest, bounds, payload
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _canonical_metadata(
        frame_names: list[str],
        raw_store: RawDepthStore,
        manifest: dict[str, Any],
        bounds_payload: dict[str, Any],
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "schema_version": CANONICAL_DEPTH_SCHEMA_VERSION,
            "algorithm_version": CANONICAL_DEPTH_ALGORITHM_VERSION,
            "representation": "relative_disparity",
            "near_value": 1.0,
            "far_value": 0.0,
            "encoding": "uint16_png",
            "encoding_scale": 65535.0,
            "num_frames": len(frame_names),
            "frame_names": list(frame_names),
            "native_shape": list(raw_store.metadata["native_shape"]),
            "source_raw_fingerprint": raw_store.metadata["fingerprint"],
            "source_model_fingerprint": canonical_json_hash(
                raw_store.metadata["semantic_fingerprint"]
            ),
            "scene_manifest_fingerprint": canonical_json_hash(manifest),
            "depth_bounds_fingerprint": bounds_payload["fingerprint"],
        }
        metadata["fingerprint"] = canonical_json_hash(metadata)
        return metadata

    @staticmethod
    def _canonical_paths(canonical_dir: Path, frame_names: list[str]) -> list[Path]:
        return [canonical_dir / f"{Path(name).stem}.png" for name in frame_names]

    def _validated_canonical_files(
        self, canonical_dir: Path, expected_metadata: dict[str, Any]
    ) -> list[Path] | None:
        metadata_path = canonical_dir / "metadata.json"
        if not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata != expected_metadata:
                return None
            files = self._canonical_paths(canonical_dir, metadata["frame_names"])
            expected_shape = tuple(int(value) for value in metadata["native_shape"])
            if not png_headers_match(files, shape=expected_shape, bit_depth=16):
                return None
            return files
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _restore_cached_canonical_stage(
        self,
        cached_files: list[Path],
        frame_files: list[Path],
        canonical_dir: Path,
    ) -> list[Path]:
        metadata_path = cached_files[0].parent / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fingerprint = metadata.get("fingerprint")
        unhashed = {key: value for key, value in metadata.items() if key != "fingerprint"}
        if (
            metadata.get("frame_names") != [path.name for path in frame_files]
            or not isinstance(fingerprint, str)
            or fingerprint != canonical_json_hash(unhashed)
        ):
            raise ValueError("Cached canonical metadata does not match local source frames")

        destinations = self._canonical_paths(canonical_dir, metadata["frame_names"])
        for source, destination in zip(cached_files, destinations):
            temporary = destination.with_name(f"{destination.stem}.tmp{destination.suffix}")
            shutil.copy2(source, temporary)
            temporary.replace(destination)
        self._atomic_write_json(canonical_dir / "metadata.json", metadata)
        validated = self._validated_canonical_files(canonical_dir, metadata)
        if validated is None:
            raise OSError("Restored canonical cache failed local validation")
        return validated

    def _write_canonical_stage(
        self,
        raw_store: RawDepthStore,
        raw_files: list[Path],
        frame_files: list[Path],
        manifest: dict[str, Any],
        bounds: dict[int, SceneDepthBounds],
        canonical_dir: Path,
        metadata: dict[str, Any],
        progress_tracker,
    ) -> list[Path]:
        existing = self._validated_canonical_files(canonical_dir, metadata)
        if existing is not None:
            return existing

        representation = DepthRepresentation(raw_store.metadata["representation"])
        canonical_files = self._canonical_paths(canonical_dir, metadata["frame_names"])
        native_shape = raw_store.metadata.get("native_shape")
        if not isinstance(native_shape, list) or len(native_shape) != 2:
            raise ValueError("Raw-depth metadata is missing native_shape")
        estimated_item_bytes = int(native_shape[0]) * int(native_shape[1]) * 32
        worker_count = calculate_frame_stage_workers(len(raw_files), estimated_item_bytes)

        self._run_canonical_workers(
            raw_store,
            raw_files,
            canonical_files,
            manifest["scene_ids"],
            representation,
            bounds,
            worker_count,
            progress_tracker,
        )

        self._atomic_write_json(canonical_dir / "metadata.json", metadata)
        validated = self._validated_canonical_files(canonical_dir, metadata)
        if validated is None:
            raise OSError("Canonical disparity stage failed final metadata validation")
        return validated

    def _run_canonical_workers(
        self,
        raw_store: RawDepthStore,
        raw_files: list[Path],
        canonical_files: list[Path],
        scene_ids: list[int],
        representation: DepthRepresentation,
        bounds: dict[int, SceneDepthBounds],
        worker_count: int,
        progress_tracker,
    ) -> None:
        """Write canonical maps concurrently while reporting ordered progress and previews."""

        def write_one(item):
            raw_file, output_file, scene_id = item
            values = raw_store.load(raw_file)
            canonical = canonicalize_depth(values, representation, bounds[int(scene_id)])
            encoded = encode_canonical_png(canonical)
            self._atomic_write_png(output_file, encoded)
            return output_file

        def report_result(index: int, output_file: Path) -> None:
            if progress_tracker:
                progress_tracker.update_progress(
                    f"Canonicalizing depth map {index + 1}/{len(raw_files)}",
                    phase="depth_estimation",
                    frame_num=index + 1,
                    step_name="Depth Map Generation",
                )
            if progress_tracker and hasattr(progress_tracker, "send_preview_frame"):
                if index % PREVIEW_FRAME_SAMPLE_RATE == 0 or index == len(raw_files) - 1:
                    progress_tracker.send_preview_frame(output_file, "depth_map", index + 1)

        run_ordered_frame_tasks(
            zip(raw_files, canonical_files, scene_ids),
            write_one,
            worker_count=worker_count,
            on_ordered_result=report_result,
        )

    @staticmethod
    def _remove_raw_payloads(raw_store: RawDepthStore) -> None:
        for path in raw_store.complete_files:
            path.unlink(missing_ok=True)
        raw_store.metadata["completed_count"] = 0
        DepthMapProcessor._atomic_write_json(raw_store.metadata_path, raw_store.metadata)

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)

    @staticmethod
    def _atomic_write_png(path: Path, values: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
        if not cv2.imwrite(str(temporary), values):
            raise OSError(f"Could not write canonical depth map: {path}")
        temporary.replace(path)

    @staticmethod
    def _report_file_cache_hit(depth_files: list[Path], progress_tracker, source: str) -> None:
        print(f"Step 2/7: Reusing {len(depth_files)} depth maps from {source}")
        if progress_tracker:
            progress_tracker.update_progress(
                f"Reused {len(depth_files)} depth maps",
                phase="depth_estimation",
                frame_num=len(depth_files),
                step_name="Depth Map Generation",
                step_progress=len(depth_files),
                step_total=len(depth_files),
            )

    def _determine_chunk_params(
        self, frame_w: int, frame_h: int, depth_resolution: str = "auto"
    ) -> tuple[int, int]:
        """
        Determine chunk size and input size based on frame resolution, VRAM, and model.

        Uses smart VRAM-based sizing to maximize throughput without OOM errors.

        Args:
            frame_w: Frame width in pixels
            frame_h: Frame height in pixels
            depth_resolution: Either "auto" or specific resolution like "1080", "720", etc.

        Returns:
            Tuple of (chunk_size, input_size)
        """
        megapixels = (frame_h * frame_w) / 1_000_000
        print(f"  Frame resolution: {frame_w}x{frame_h} ({megapixels:.1f}MP)")

        # Get VRAM info for smart sizing
        vram_info = get_vram_info()
        if vram_info["total"] > 0:
            print(
                f"  GPU VRAM: {vram_info['available']:.1f}GB available / {vram_info['total']:.1f}GB total"
            )

        try:
            input_size = resolve_depth_input_size(frame_w, frame_h, depth_resolution)
        except (TypeError, ValueError):
            print(f"  Warning: Invalid depth_resolution '{depth_resolution}', using auto")
            input_size = resolve_depth_input_size(frame_w, frame_h, "auto")
        if depth_resolution == "auto":
            print(f"  Auto depth resolution: {input_size}px")
        else:
            print(f"  Using manual depth resolution: {input_size}px")

        model_version, model_size = self._resource_profile()

        # Calculate optimal chunk size based on VRAM
        if vram_info["total"] > 0:
            # Use smart VRAM-based sizing
            chunk_size = calculate_optimal_chunk_size(
                frame_w, frame_h, input_size, model_version, model_size
            )
            print(
                f"  Smart VRAM sizing: {chunk_size} frames/chunk (model: {model_version}/{model_size})"
            )
        else:
            # Fallback to fixed sizing (CPU or no CUDA)
            chunk_size = self._get_chunk_size_for_resolution(input_size)
            print(f"  CPU mode: {chunk_size} frames/chunk")

        max_batch_size = getattr(self.depth_estimator, "max_batch_size", None)
        if isinstance(max_batch_size, int) and max_batch_size > 0:
            chunk_size = min(chunk_size, max_batch_size)

        return chunk_size, input_size

    def _resource_profile(self) -> tuple[str, str]:
        """Map stable estimator metadata to the existing VRAM sizing profiles."""
        metadata: dict[str, Any] = {}
        get_model_info = getattr(self.depth_estimator, "get_model_info", None)
        if callable(get_model_info):
            reported = get_model_info()
            if isinstance(reported, dict):
                metadata = reported

        backend_id = metadata.get("backend_id", getattr(self.depth_estimator, "backend_id", None))
        family = metadata.get("family")
        if backend_id == "moge2" or family == "moge":
            model_size = metadata.get(
                "model_size",
                getattr(self.depth_estimator, "model_size", metadata.get("model_name", "vitb")),
            )
            return "v3", {"vits": "small", "vitb": "base", "vitl": "large"}.get(
                str(model_size), "base"
            )

        model_version = "v3" if hasattr(self.depth_estimator, "model_type") else "v2"
        model_size = (
            self.depth_estimator.get_model_size()
            if hasattr(self.depth_estimator, "get_model_size")
            else "base"
        )
        return model_version, model_size

    def _auto_determine_input_size(self, frame_w: int, frame_h: int, megapixels: float) -> int:
        """
        Determine input size automatically based on frame resolution.

        Args:
            frame_w: Frame width
            frame_h: Frame height
            megapixels: Frame megapixels

        Returns:
            Optimal input size for depth estimation
        """
        del megapixels
        input_size = resolve_depth_input_size(frame_w, frame_h, "auto")

        print(f"  Auto depth resolution: {input_size}px")
        return input_size

    def _get_chunk_size_for_resolution(self, input_size: int) -> int:
        """
        Get appropriate chunk size based on depth map resolution.

        Args:
            input_size: Depth map resolution in pixels

        Returns:
            Chunk size for processing
        """
        if input_size >= RESOLUTION_4K:
            return CHUNK_SIZE_4K
        elif input_size >= RESOLUTION_1440P:
            return CHUNK_SIZE_1440P
        elif input_size >= RESOLUTION_1080P:
            return CHUNK_SIZE_1080P_MANUAL
        elif input_size >= RESOLUTION_720P:
            return CHUNK_SIZE_720P
        else:
            return CHUNK_SIZE_SMALL

    def _clear_gpu_memory(self) -> None:
        """
        Clear GPU memory and cache.

        Side effects:
            - Clears CUDA cache
            - Frees GPU memory
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            mem_free = torch.cuda.mem_get_info()[0] / (1024**3)  # Convert to GB
            print(f"  GPU memory freed: {mem_free:.2f} GB available")

    def _load_chunk_frames(self, chunk_files: list[Path], settings: dict[str, Any]) -> list | None:
        """
        Load chunk of frames into memory.

        Args:
            chunk_files: List of frame file paths for this chunk
            settings: Processing settings

        Returns:
            List of loaded frame images

        Side effects:
            - Reads images from disk
        """
        chunk_frames = []
        for frame_file in chunk_files:
            frame = cv2.imread(str(frame_file))
            if frame is None:
                print(f"Warning: Could not load {frame_file}")
                continue

            chunk_frames.append(frame)

        return chunk_frames if chunk_frames else None
