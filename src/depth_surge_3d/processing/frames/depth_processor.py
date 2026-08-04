"""
Depth map processing module.

Handles depth map generation with caching, VRAM management, and chunking strategies.
"""

from __future__ import annotations

import cv2
import hashlib
import json
import numpy as np
import shutil
import torch
from pathlib import Path
from typing import Any

from ...core.constants import (
    DEPTH_MAP_SCALE_FLOAT,
    DEPTH_MAP_STORAGE_SCALE,
    DEFAULT_FALLBACK_FPS,
    RESOLUTION_4K,
    RESOLUTION_1440P,
    RESOLUTION_1080P,
    RESOLUTION_720P,
    RESOLUTION_SD,
    MEGAPIXELS_4K,
    MEGAPIXELS_1080P,
    MEGAPIXELS_720P,
    CHUNK_SIZE_4K,
    CHUNK_SIZE_1440P,
    CHUNK_SIZE_1080P_MANUAL,
    CHUNK_SIZE_720P,
    CHUNK_SIZE_SMALL,
    PREVIEW_FRAME_SAMPLE_RATE,
)
from ...utils import (
    get_cached_depth_maps,
    get_cached_depth_map_files,
    save_depth_maps_to_cache,
    save_depth_map_files_to_cache,
    get_cache_size,
)
from ...utils import calculate_optimal_chunk_size, get_vram_info
from ...utils import resize_image
from ...inference.depth.types import DepthBatch, DepthRepresentation
from .depth_normalizer import (
    SceneDepthBounds,
    canonicalize_depth,
    encode_canonical_png,
)
from .depth_storage import (
    RawDepthStore,
    build_model_fingerprint,
    canonical_json_hash,
    estimate_depth_disk_bytes,
    require_disk_space,
)
from .scene_analyzer import (
    SCENE_ALGORITHM_VERSION,
    SCENE_SCHEMA_VERSION,
    analyze_scenes,
    finalize_scenes,
    sample_scene_depths,
)


LEGACY_DEPTH_ARRAY_LIMIT_BYTES = 512 * 1024 * 1024
CANONICAL_DEPTH_SCHEMA_VERSION = 1
CANONICAL_DEPTH_ALGORITHM_VERSION = "scene-percentile-v1"
DEPTH_BOUNDS_SCHEMA_VERSION = 1
DEPTH_SAMPLES_SCHEMA_VERSION = 1

_RAW_SEMANTIC_SETTING_KEYS = (
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


class DepthMapProcessor:
    """
    Depth map generation with caching and memory management.

    Responsibilities:
    - Depth map generation with model inference
    - VRAM-based chunk sizing
    - Global and local depth cache management
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

    def generate_depth_maps(
        self,
        frame_files: list[Path],
        settings: dict[str, Any],
        directories: dict[str, Path],
        progress_tracker,
    ) -> np.ndarray | None:
        """
        Main entry point - generates depth maps with caching.

        Tries cache first, then generates if needed.

        Args:
            frame_files: List of frame file paths
            settings: Processing settings with depth parameters
            directories: Dictionary of processing directories
            progress_tracker: Optional progress tracker

        Returns:
            Numpy array of depth maps, or None if failed

        Side effects:
            - GPU memory operations
            - Filesystem I/O (cache reads/writes)
            - Depth map image writes
        """
        if frame_files:
            self._reject_large_legacy_depth_array(frame_files, settings)

        # Check if depth maps already exist (only if keep_intermediates is enabled)
        if settings.get("keep_intermediates") and "depth_maps" in directories:
            existing = self._try_load_existing_depth_maps(
                frame_files, directories, progress_tracker
            )
            if existing is not None:
                return existing

        # Check global depth cache (works across different output batches)
        video_path = settings.get("video_path")
        if video_path:
            cached = self._try_load_cached_depth_maps(
                video_path, settings, len(frame_files), progress_tracker
            )
            if cached is not None:
                return cached

        print("Step 2/7: Generating depth maps (temporal consistency enabled)...")
        print("  Using memory-efficient chunked processing...")
        if progress_tracker:
            progress_tracker.update_progress(
                "Generating depth maps",
                phase="depth_estimation",
                frame_num=0,
                step_name="Depth Map Generation",
                step_progress=0,
                step_total=len(frame_files),
            )

        depth_maps = self._generate_depth_maps_chunked(
            frame_files, settings, directories, progress_tracker
        )
        if depth_maps is None:
            return None

        # Save to global cache for future runs
        if video_path and depth_maps is not None:
            self._save_to_depth_cache(video_path, settings, depth_maps)

        return depth_maps

    def generate_depth_map_files(
        self,
        frame_files: list[Path],
        settings: dict[str, Any],
        directories: dict[str, Path],
        progress_tracker,
    ) -> list[Path] | None:
        """Build native raw depth, final scene bounds, and canonical disparity files."""
        if not frame_files:
            return None

        scene_dir, raw_dir, canonical_dir = self._resolve_depth_directories(
            frame_files, directories
        )
        for directory in (scene_dir, raw_dir, canonical_dir):
            directory.mkdir(parents=True, exist_ok=True)

        semantic_fingerprint = self._raw_semantic_fingerprint(frame_files, settings)
        model_fingerprint = canonical_json_hash(semantic_fingerprint)
        cache_settings = dict(settings, model_fingerprint=model_fingerprint)
        video_path = settings.get("video_path")
        restored = self._try_restore_global_canonical_cache(
            video_path,
            frame_files,
            cache_settings,
            model_fingerprint,
            canonical_dir,
            progress_tracker,
        )
        if restored is not None:
            return restored

        manifest = self._load_or_analyze_scenes(frame_files, scene_dir, settings)
        frame_names = [path.name for path in frame_files]
        requested_dtype = str(settings.get("raw_storage_dtype", "auto"))

        raw_store = self._open_raw_store_if_present(
            raw_dir,
            frame_names=frame_names,
            semantic_fingerprint=semantic_fingerprint,
            requested_dtype=requested_dtype,
        )
        final_state = self._load_final_scene_state(scene_dir, manifest)
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

        self._report_canonical_generation_start(len(frame_files), progress_tracker)

        raw_store = self._complete_raw_depth_stage(
            frame_files,
            settings,
            raw_dir,
            semantic_fingerprint,
            raw_store,
            progress_tracker,
        )
        raw_files = [raw_store.path_for(name) for name in frame_names]
        if not all(path.is_file() for path in raw_files):
            raise RuntimeError("Raw-depth global barrier was reached with missing frames")

        final_manifest, bounds, bounds_payload = self._finalize_scene_stage(
            scene_dir,
            manifest,
            raw_store,
            raw_files,
        )

        expected_metadata = self._canonical_metadata(
            frame_names,
            raw_store,
            final_manifest,
            bounds_payload,
        )
        canonical_files = self._write_canonical_stage(
            raw_store,
            raw_files,
            frame_files,
            final_manifest,
            bounds,
            canonical_dir,
            expected_metadata,
            progress_tracker,
        )

        if not settings.get("keep_intermediates", False):
            self._remove_raw_payloads(raw_store)

        if video_path and save_depth_map_files_to_cache(
            str(video_path), cache_settings, canonical_files
        ):
            print("  Canonical disparity maps saved to global cache")
        return canonical_files

    def _try_restore_global_canonical_cache(
        self,
        video_path: Any,
        frame_files: list[Path],
        cache_settings: dict[str, Any],
        model_fingerprint: str,
        canonical_dir: Path,
        progress_tracker,
    ) -> list[Path] | None:
        if not video_path:
            return None
        cached_files = get_cached_depth_map_files(
            str(video_path),
            cache_settings,
            len(frame_files),
            expected_model_fingerprint=model_fingerprint,
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

    @staticmethod
    def _report_canonical_generation_start(frame_count: int, progress_tracker) -> None:
        print("Step 2/7: Generating canonical disparity maps...")
        print("  Using restartable scene, raw-depth, and canonical stages...")
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
        final_state = self._load_final_scene_state(scene_dir, manifest)
        if final_state is not None:
            return final_state
        candidate_manifest = self._candidate_manifest(manifest)
        samples, sample_fingerprint = self._load_or_create_scene_samples(
            scene_dir,
            raw_files,
            candidate_manifest,
            DepthRepresentation(raw_store.metadata["representation"]),
        )
        final_manifest, bounds = finalize_scenes(candidate_manifest, samples)
        bounds_payload = self._write_depth_bounds(
            scene_dir,
            bounds,
            sample_fingerprint,
        )
        final_manifest["sample_fingerprint"] = sample_fingerprint
        final_manifest["bounds_fingerprint"] = bounds_payload["fingerprint"]
        self._atomic_write_json(scene_dir / "scene_manifest.json", final_manifest)
        return final_manifest, bounds, bounds_payload

    @staticmethod
    def _resolve_depth_directories(
        frame_files: list[Path], directories: dict[str, Path]
    ) -> tuple[Path, Path, Path]:
        canonical_dir = directories.get("disparity_maps") or directories.get("depth_maps")
        base_dir = directories.get("base")
        if base_dir is None:
            base_dir = (
                canonical_dir.parent if canonical_dir is not None else frame_files[0].parent.parent
            )
        return (
            directories.get("scene_data", base_dir / "01_scene_data"),
            directories.get("depth_raw", base_dir / "02_depth_raw"),
            canonical_dir or base_dir / "03_disparity_maps",
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
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        manifest_path = scene_dir / "scene_manifest.json"
        expected_names = [path.name for path in frame_files]
        scene_settings = self._scene_settings(settings)
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            valid = (
                manifest.get("schema_version") == SCENE_SCHEMA_VERSION
                and manifest.get("algorithm_version") == SCENE_ALGORITHM_VERSION
                and manifest.get("status") in {"candidate", "final"}
                and manifest.get("frame_names") == expected_names
                and manifest.get("settings") == scene_settings
                and len(manifest.get("scene_ids", [])) == len(frame_files)
            )
            if not valid:
                raise ValueError("Existing scene manifest does not match this source or settings")
            return manifest
        return analyze_scenes(frame_files, scene_dir, **scene_settings)

    @staticmethod
    def _source_frame_fingerprint(frame_files: list[Path]) -> str:
        hasher = hashlib.sha256()
        for path in frame_files:
            hasher.update(path.name.encode("utf-8"))
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    hasher.update(chunk)
        return hasher.hexdigest()

    def _raw_semantic_fingerprint(
        self, frame_files: list[Path], settings: dict[str, Any]
    ) -> dict[str, Any]:
        depth_settings = {
            key: settings.get(key) for key in _RAW_SEMANTIC_SETTING_KEYS if key in settings
        }
        fingerprint = build_model_fingerprint(self.depth_estimator, depth_settings)
        fingerprint["source_frame_fingerprint"] = self._source_frame_fingerprint(frame_files)
        fingerprint["preprocessing_algorithm"] = "native-depth-adapter-v1"
        return fingerprint

    @staticmethod
    def _open_raw_store_if_present(
        raw_dir: Path,
        *,
        frame_names: list[str],
        semantic_fingerprint: dict[str, Any],
        requested_dtype: str,
    ) -> RawDepthStore | None:
        metadata_path = raw_dir / "metadata.json"
        if not metadata_path.is_file():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        representation = DepthRepresentation(metadata["representation"])
        return RawDepthStore.open_or_create(
            raw_dir,
            frame_names=frame_names,
            representation=representation,
            semantic_fingerprint=semantic_fingerprint,
            requested_dtype=requested_dtype,
        )

    @staticmethod
    def _estimate_native_shape(
        frame_width: int, frame_height: int, input_size: int
    ) -> tuple[int, int]:
        longest = max(frame_width, frame_height)
        scale = min(1.0, input_size / longest)
        return (
            max(1, int(round(frame_height * scale))),
            max(1, int(round(frame_width * scale))),
        )

    def _complete_raw_depth_stage(
        self,
        frame_files: list[Path],
        settings: dict[str, Any],
        raw_dir: Path,
        semantic_fingerprint: dict[str, Any],
        raw_store: RawDepthStore | None,
        progress_tracker,
    ) -> RawDepthStore:
        chunk_size, input_size, target_fps, requested_dtype = self._prepare_raw_depth_stage(
            frame_files,
            settings,
            raw_dir,
            raw_store,
        )
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
                semantic_fingerprint,
                raw_store,
                requested_dtype,
                input_size,
                target_fps,
            )
            self._clear_gpu_memory()
            self._report_raw_chunk_progress(
                progress_tracker,
                chunk_start // chunk_size + 1,
                total_chunks,
                chunk_end,
                len(frame_files),
            )

        if raw_store is None:
            raise RuntimeError("Depth estimator produced no raw depth")
        return raw_store

    def _prepare_raw_depth_stage(
        self,
        frame_files: list[Path],
        settings: dict[str, Any],
        raw_dir: Path,
        raw_store: RawDepthStore | None,
    ) -> tuple[int, int, Any, str]:
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
        self._require_depth_disk_space(
            raw_dir,
            len(frame_files),
            estimated_height,
            estimated_width,
            storage_bytes,
            bool(settings.get("keep_intermediates", False)),
        )

        if raw_store is not None:
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
        return chunk_size, input_size, target_fps, requested_dtype

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
        semantic_fingerprint: dict[str, Any],
        raw_store: RawDepthStore | None,
        requested_dtype: str,
        input_size: int,
        target_fps: Any,
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
            raw_store = RawDepthStore.open_or_create(
                raw_dir,
                frame_names=frame_names,
                representation=result.representation,
                semantic_fingerprint=semantic_fingerprint,
                requested_dtype=requested_dtype,
                first_values=result.values,
            )
            native_height, native_width = result.values.shape[1:]
            selected_bytes = 2 if raw_store.metadata["selected_dtype"] == "float16" else 4
            self._require_depth_disk_space(
                raw_dir,
                len(frame_names),
                int(native_height),
                int(native_width),
                selected_bytes,
                bool(settings.get("keep_intermediates", False)),
            )
        elif result.representation.value != raw_store.metadata["representation"]:
            raise ValueError("Depth representation changed during inference")

        raw_store.write_batch(chunk_names, result.values)
        return raw_store

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
    ) -> tuple[dict[int, np.ndarray], str]:
        sample_path = scene_dir / "depth_samples.npz"
        manifest_fingerprint = canonical_json_hash(candidate_manifest)
        if sample_path.is_file():
            try:
                with np.load(sample_path, allow_pickle=False) as payload:
                    arrays = {key: np.asarray(payload[key]) for key in payload.files}
                stored_fingerprint = str(arrays.pop("content_fingerprint").item())
                if (
                    int(arrays["schema_version"].item()) == DEPTH_SAMPLES_SCHEMA_VERSION
                    and str(arrays["algorithm_version"].item()) == SCENE_ALGORITHM_VERSION
                    and str(arrays["manifest_fingerprint"].item()) == manifest_fingerprint
                    and self._array_payload_fingerprint(arrays) == stored_fingerprint
                ):
                    scene_ids = list(
                        dict.fromkeys(int(value) for value in candidate_manifest["scene_ids"])
                    )
                    samples = {
                        scene_id: arrays[f"scene_{scene_id}_samples"].astype(np.float32)
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
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": DEPTH_BOUNDS_SCHEMA_VERSION,
            "algorithm_version": CANONICAL_DEPTH_ALGORITHM_VERSION,
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
    ) -> tuple[dict[str, Any], dict[int, SceneDepthBounds], dict[str, Any]] | None:
        if manifest.get("status") != "final":
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
            for path in files:
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if image is None or image.dtype != np.uint16 or image.shape != expected_shape:
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
        for index, (raw_file, source_file, output_file, scene_id) in enumerate(
            zip(raw_files, frame_files, canonical_files, manifest["scene_ids"])
        ):
            values = raw_store.load(raw_file)
            canonical = canonicalize_depth(values, representation, bounds[int(scene_id)])
            encoded = encode_canonical_png(canonical)
            self._atomic_write_png(output_file, encoded)
            restored = cv2.imread(str(output_file), cv2.IMREAD_UNCHANGED)
            if restored is None or not np.array_equal(restored, encoded):
                raise OSError(f"Canonical depth verification failed: {output_file}")
            if progress_tracker and hasattr(progress_tracker, "send_preview_frame"):
                if index % PREVIEW_FRAME_SAMPLE_RATE == 0 or index == len(raw_files) - 1:
                    progress_tracker.send_preview_frame(output_file, "depth_map", index + 1)

        self._atomic_write_json(canonical_dir / "metadata.json", metadata)
        validated = self._validated_canonical_files(canonical_dir, metadata)
        if validated is None:
            raise OSError("Canonical disparity stage failed final metadata validation")
        return validated

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

    def _reject_large_legacy_depth_array(
        self, frame_files: list[Path], settings: dict[str, Any]
    ) -> None:
        sample = cv2.imread(str(frame_files[0]))
        if sample is None:
            return
        height, width = sample.shape[:2]
        depth_resolution = settings.get("depth_resolution", "auto")
        if depth_resolution == "auto":
            input_size = self._auto_determine_input_size(
                width, height, (width * height) / 1_000_000
            )
        else:
            try:
                input_size = int(depth_resolution)
            except (TypeError, ValueError):
                input_size = max(width, height)
        native_height, native_width = self._estimate_native_shape(width, height, input_size)
        required = len(frame_files) * native_height * native_width * 4
        if required > LEGACY_DEPTH_ARRAY_LIMIT_BYTES:
            raise MemoryError(
                "Legacy in-memory depth output exceeds 512 MiB; use generate_depth_map_files() "
                "for file-backed processing"
            )

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

    def _generate_depth_map_files_chunked(
        self,
        frame_files: list[Path],
        settings: dict[str, Any],
        directories: dict[str, Path],
        progress_tracker,
    ) -> list[Path] | None:
        """Run inference per chunk and persist each result before loading the next chunk."""
        sample_frame = cv2.imread(str(frame_files[0]))
        if sample_frame is None:
            return None

        frame_h, frame_w = sample_frame.shape[:2]
        chunk_size, input_size = self._determine_chunk_params(
            frame_w,
            frame_h,
            settings.get("depth_resolution", "auto"),
        )
        print(f"  Processing in chunks of {chunk_size} frames (input_size={input_size})...")
        self._clear_gpu_memory()

        depth_dir = directories["depth_maps"]
        depth_files = [depth_dir / f"{frame_file.stem}.png" for frame_file in frame_files]
        num_frames = len(frame_files)
        total_chunks = (num_frames + chunk_size - 1) // chunk_size

        for chunk_start in range(0, num_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_frames)
            chunk_files = frame_files[chunk_start:chunk_end]
            chunk_outputs = depth_files[chunk_start:chunk_end]
            chunk_num = chunk_start // chunk_size + 1

            if not all(path.is_file() for path in chunk_outputs):
                chunk_frames = self._load_chunk_frames(chunk_files, settings)
                if chunk_frames is None or len(chunk_frames) != len(chunk_files):
                    print("Error: Not all frames could be loaded in chunk")
                    return None

                try:
                    chunk_depth_maps = self._process_chunk_depth(
                        chunk_frames,
                        chunk_files,
                        settings,
                        directories,
                        input_size,
                        progress_tracker,
                    )
                    if chunk_depth_maps is None or len(chunk_depth_maps) != len(chunk_files):
                        print("Error: Depth estimator returned an unexpected frame count")
                        return None
                    if not all(path.is_file() for path in chunk_outputs):
                        print("Error: Could not persist all depth maps in chunk")
                        return None
                except Exception as e:
                    print(f"Error processing chunk {chunk_start}-{chunk_end}: {e}")
                    return None
                finally:
                    if "chunk_frames" in locals():
                        del chunk_frames
                    if "chunk_depth_maps" in locals():
                        del chunk_depth_maps
                    self._clear_gpu_memory()

            if progress_tracker:
                progress_tracker.update_progress(
                    f"Chunk {chunk_num}/{total_chunks}: Depth maps {chunk_end}/{num_frames}",
                    phase="depth_estimation",
                    frame_num=chunk_end,
                    step_name="Depth Map Generation",
                    step_progress=chunk_end,
                    step_total=num_frames,
                )

        return depth_files

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

        # Determine input size (depth resolution)
        if depth_resolution != "auto":
            try:
                input_size = int(depth_resolution)
                print(f"  Using manual depth resolution: {input_size}px")
            except (ValueError, TypeError):
                print(f"  Warning: Invalid depth_resolution '{depth_resolution}', using auto")
                input_size = self._auto_determine_input_size(frame_w, frame_h, megapixels)
        else:
            input_size = self._auto_determine_input_size(frame_w, frame_h, megapixels)

        # Get model information
        model_version = "v3" if hasattr(self.depth_estimator, "model_type") else "v2"
        model_size = (
            self.depth_estimator.get_model_size()
            if hasattr(self.depth_estimator, "get_model_size")
            else "base"
        )

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
        # Auto mode: Match depth resolution to actual frame size
        # Never exceed source frame resolution - upscaling depth is pointless
        if megapixels > MEGAPIXELS_4K:  # >8MP (4K is ~8.3MP)
            input_size = min(max(frame_w, frame_h), RESOLUTION_4K)
        elif megapixels > MEGAPIXELS_1080P:  # >2MP (1080p is 2.1MP)
            input_size = min(max(frame_w, frame_h), RESOLUTION_1080P)
        elif megapixels > MEGAPIXELS_720P:  # >1MP (720p is 0.9MP)
            input_size = min(max(frame_w, frame_h), RESOLUTION_720P)
        else:
            input_size = min(max(frame_w, frame_h), RESOLUTION_SD)

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

            # Apply super sampling if needed
            if settings["super_sample"] != "none":
                target_width = max(frame.shape[1], settings["per_eye_width"] * 2)
                target_height = max(frame.shape[0], settings["per_eye_height"] * 2)
                frame = resize_image(frame, target_width, target_height)

            chunk_frames.append(frame)

        return chunk_frames if chunk_frames else None

    def _process_chunk_depth(
        self,
        chunk_frames: list,
        chunk_files: list[Path],
        settings: dict[str, Any],
        directories: dict[str, Path],
        input_size: int,
        progress_tracker=None,
    ) -> np.ndarray | None:
        """
        Process chunk with depth estimation.

        Args:
            chunk_frames: List of frame images
            chunk_files: List of frame file paths
            settings: Processing settings
            directories: Dictionary of processing directories
            input_size: Depth map resolution
            progress_tracker: Optional progress tracker

        Returns:
            Numpy array of depth maps

        Side effects:
            - GPU inference
            - Progress updates
        """
        # Normalize target_fps
        target_fps = settings.get("target_fps", DEFAULT_FALLBACK_FPS)
        if target_fps is None or str(target_fps) == "None" or target_fps == "original":
            target_fps = 30

        # Estimate depth
        chunk_frames_array = np.array(chunk_frames)
        depth_result = self.depth_estimator.estimate_depth_batch(
            chunk_frames_array, target_fps=target_fps, input_size=input_size, fp32=False
        )
        chunk_depth_maps = (
            depth_result.values if isinstance(depth_result, DepthBatch) else depth_result
        )

        # Depth files are required working state. Retention is handled after encoding.
        if "depth_maps" in directories:
            self._save_depth_maps(
                chunk_depth_maps, chunk_files, directories["depth_maps"], progress_tracker
            )

        return chunk_depth_maps

    def _generate_depth_maps_chunked(
        self,
        frame_files: list[Path],
        settings: dict[str, Any],
        directories: dict[str, Path],
        progress_tracker,
    ) -> np.ndarray | None:
        """
        Memory-efficient chunked depth generation.

        Processes frames in small batches to avoid CUDA OOM errors.

        Args:
            frame_files: List of frame file paths
            settings: Processing settings
            directories: Dictionary of processing directories
            progress_tracker: Optional progress tracker

        Returns:
            Numpy array of depth maps, or None if failed

        Side effects:
            - GPU memory operations
            - Filesystem I/O
        """
        # Determine chunk parameters based on resolution
        sample_frame = cv2.imread(str(frame_files[0]))
        if sample_frame is None:
            return None

        frame_h, frame_w = sample_frame.shape[:2]
        depth_resolution = settings.get("depth_resolution", "auto")
        chunk_size, input_size = self._determine_chunk_params(frame_w, frame_h, depth_resolution)

        print(f"  Processing in chunks of {chunk_size} frames (input_size={input_size})...")

        # Clear GPU cache before processing
        self._clear_gpu_memory()

        # Process all chunks
        all_depth_maps: list[Any] = []
        num_frames = len(frame_files)
        total_chunks = (num_frames + chunk_size - 1) // chunk_size

        for chunk_start in range(0, num_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_frames)
            chunk_files = frame_files[chunk_start:chunk_end]
            chunk_num = chunk_start // chunk_size + 1

            # Load chunk frames
            chunk_frames = self._load_chunk_frames(chunk_files, settings)
            if not chunk_frames:
                print("Error: No frames loaded in chunk")
                return None

            # Process chunk for depth
            try:
                chunk_depth_maps = self._process_chunk_depth(
                    chunk_frames, chunk_files, settings, directories, input_size, progress_tracker
                )
                all_depth_maps.extend(chunk_depth_maps)  # type: ignore[arg-type]

                # Clear references and GPU cache
                del chunk_frames
                del chunk_depth_maps
                self._clear_gpu_memory()

                # Update progress
                if progress_tracker:
                    progress_tracker.update_progress(
                        f"Chunk {chunk_num}/{total_chunks}: Depth maps {chunk_end}/{num_frames}",
                        phase="depth_estimation",
                        frame_num=chunk_end,
                        step_name="Depth Map Generation",
                        step_progress=chunk_end,
                        step_total=num_frames,
                    )

            except Exception as e:
                print(f"Error processing chunk {chunk_start}-{chunk_end}: {e}")
                return None

        return np.array(all_depth_maps)

    def _generate_depth_maps_batch(
        self, frames: np.ndarray, settings: dict[str, Any], progress_tracker
    ) -> np.ndarray | None:
        """
        Full batch depth generation (no chunking).

        Generate depth maps for all frames with temporal consistency.

        Args:
            frames: Numpy array of frame images
            settings: Processing settings
            progress_tracker: Optional progress tracker

        Returns:
            Numpy array of depth maps, or None if failed

        Side effects:
            - GPU memory operations
            - Filesystem I/O
        """
        try:
            # Use Video-Depth-Anything for temporal consistency
            target_fps = settings.get("target_fps", DEFAULT_FALLBACK_FPS)
            if target_fps is None or str(target_fps) == "None" or target_fps == "original":
                target_fps = 30

            # Use depth resolution from settings (default: auto/1080px)
            depth_resolution = settings.get("depth_resolution", "auto")
            if depth_resolution == "auto":
                input_size = 1080  # Match typical 1080p video resolution
            else:
                try:
                    input_size = int(depth_resolution)
                except (ValueError, TypeError):
                    input_size = 1080

            depth_result = self.depth_estimator.estimate_depth_batch(
                frames, target_fps=target_fps, input_size=input_size, fp32=False
            )
            return depth_result.values if isinstance(depth_result, DepthBatch) else depth_result

        except Exception as e:
            print(f"Error generating depth maps: {e}")
            return None

    def _save_depth_maps(
        self,
        depth_maps: np.ndarray,
        frame_files: list[Path],
        depth_dir: Path,
        progress_tracker=None,
    ) -> None:
        """
        Save depth maps to disk.

        Args:
            depth_maps: Numpy array of depth maps
            frame_files: List of frame files (for naming)
            depth_dir: Output directory
            progress_tracker: Optional progress tracker

        Side effects:
            - Writes depth map images to disk
        """
        for i, (depth_map, frame_file) in enumerate(zip(depth_maps, frame_files)):
            depth_vis = np.clip(
                depth_map * DEPTH_MAP_STORAGE_SCALE,
                0,
                DEPTH_MAP_STORAGE_SCALE,
            ).astype(np.uint16)
            frame_name = frame_file.stem
            depth_path = depth_dir / f"{frame_name}.png"
            if not cv2.imwrite(str(depth_path), depth_vis):
                raise OSError(f"Could not write depth map: {depth_path}")

            # Send preview frame
            if progress_tracker and hasattr(progress_tracker, "send_preview_frame"):
                if i % PREVIEW_FRAME_SAMPLE_RATE == 0 or i == len(depth_maps) - 1:
                    progress_tracker.send_preview_frame(depth_path, "depth_map", i + 1)

    def _try_load_existing_depth_maps(
        self, frame_files: list[Path], directories: dict[str, Path], progress_tracker
    ) -> np.ndarray | None:
        """
        Try to load existing depth maps from output directory.

        Args:
            frame_files: List of frame file paths
            directories: Dictionary of processing directories
            progress_tracker: Optional progress tracker

        Returns:
            Numpy array of depth maps, or None if not found

        Side effects:
            - Filesystem I/O
        """
        depth_maps_dir = directories.get("depth_maps")
        if not depth_maps_dir or not depth_maps_dir.exists():
            return None

        existing_depth_maps = sorted(list(depth_maps_dir.glob("*.png")))
        if not existing_depth_maps or len(existing_depth_maps) < len(frame_files):
            return None

        print("Step 2/7: Skipping depth map generation (depth maps already exist)")
        print(f"  Found {len(existing_depth_maps):04d} existing depth maps")
        print(f"  Location: {depth_maps_dir}\n")

        # Load existing depth maps
        depth_maps = []
        for depth_file in existing_depth_maps[: len(frame_files)]:
            depth_img = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
            if depth_img is not None:
                depth_scale = (
                    DEPTH_MAP_STORAGE_SCALE
                    if depth_img.dtype == np.uint16
                    else DEPTH_MAP_SCALE_FLOAT
                )
                depth_maps.append(depth_img.astype(float) / depth_scale)

        if len(depth_maps) == len(frame_files):
            if progress_tracker:
                progress_tracker.update_progress(
                    "Skipped depth map generation (already exists)",
                    phase="depth_estimation",
                    frame_num=len(depth_maps),
                    step_name="Depth Map Generation",
                    step_progress=len(depth_maps),
                    step_total=len(depth_maps),
                )
            return np.array(depth_maps)
        return None

    def _try_load_cached_depth_maps(
        self, video_path: str, settings: dict[str, Any], num_frames: int, progress_tracker
    ) -> np.ndarray | None:
        """
        Try to load from global depth cache.

        Args:
            video_path: Path to video file (for cache key)
            settings: Processing settings for cache key
            num_frames: Expected number of frames
            progress_tracker: Optional progress tracker

        Returns:
            Numpy array of depth maps, or None if cache miss

        Side effects:
            - Filesystem I/O (cache reads)
        """
        cached_depths = get_cached_depth_maps(video_path, settings, num_frames)
        if cached_depths is None:
            return None

        print("Step 2/7: Loading depth maps from global cache")
        print(f"  Loaded {len(cached_depths):04d} cached depth maps")
        cache_entries, cache_size_bytes = get_cache_size()
        cache_size_mb = cache_size_bytes / (1024 * 1024)
        print(f"  Cache: {cache_entries} entries, {cache_size_mb:.1f} MB total\n")

        if progress_tracker:
            progress_tracker.update_progress(
                "Loaded depth maps from cache",
                phase="depth_estimation",
                frame_num=len(cached_depths),
                step_name="Depth Map Generation",
                step_progress=len(cached_depths),
                step_total=len(cached_depths),
            )
        return cached_depths

    def _save_to_depth_cache(
        self, video_path: str, settings: dict[str, Any], depth_maps: np.ndarray
    ):
        """
        Save depth maps to global cache.

        Args:
            video_path: Path to video file (for cache key)
            settings: Processing settings for cache key
            depth_maps: Numpy array of depth maps

        Side effects:
            - Filesystem I/O (cache writes)
        """
        if save_depth_maps_to_cache(video_path, settings, depth_maps):
            cache_entries, cache_size_bytes = get_cache_size()
            cache_size_mb = cache_size_bytes / (1024 * 1024)
            print("  Cached depth maps for future use")
            print(f"  Cache: {cache_entries} entries, {cache_size_mb:.1f} MB total\n")
