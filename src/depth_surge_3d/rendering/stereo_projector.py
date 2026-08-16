"""
Main StereoProjector class for 2D to 3D VR conversion.

This module provides the main orchestration class that coordinates all
processing steps using the modular utility functions.
"""

from __future__ import annotations

import cv2
import numpy as np
import traceback
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, cast

from ..inference.depth.backend_registry import (
    EstimatorRequest,
    TEMPORAL_STABILITY_WARNING,
    build_effective_depth_run_report,
    create_registered_depth_estimator,
    get_backend_spec,
    normalize_model_size,
    validate_backend_geometry_request,
)
from ..inference.depth.types import DepthBatch
from ..utils import (
    get_resolution_dimensions,
    calculate_vr_output_dimensions,
    validate_resolution_settings,
    auto_detect_resolution,
)
from ..utils.domain.resolution import resolve_depth_input_size
from ..io.operations import (
    validate_video_file,
    get_video_properties,
)
from ..processing import VideoProcessor
from ..processing.frames.depth_normalizer import canonicalize_single_scene
from ..core.settings import validate_settings
from .stereo_geometry import build_relative_geometry
from .stereo_renderer import StereoRenderer, StereoSplatSettings


def _freeze_snapshot(value: Any) -> Any:
    """Recursively freeze JSON-like run state for trusted later execution."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_snapshot(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_snapshot(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_snapshot(item) for item in value)
    return deepcopy(value)


def _thaw_snapshot(value: Any) -> Any:
    """Return a detached mutable copy of recursively frozen run state."""
    if isinstance(value, Mapping):
        return {key: _thaw_snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_snapshot(item) for item in value]
    if isinstance(value, frozenset):
        return {_thaw_snapshot(item) for item in value}
    return deepcopy(value)


@dataclass(frozen=True)
class VideoRunPreflight:
    """Validated, resolved video inputs that are safe to execute."""

    _owner: object
    _video_path: str
    _output_dir: str
    _settings: Mapping[str, Any]
    _video_properties: Mapping[str, Any]
    _report: Mapping[str, Any]

    @property
    def settings(self) -> dict[str, Any]:
        """Return a copy so callers cannot alter the validated execution state."""
        return cast(dict[str, Any], _thaw_snapshot(self._settings))

    @property
    def video_properties(self) -> dict[str, Any]:
        """Return a copy of the canonical source properties."""
        return cast(dict[str, Any], _thaw_snapshot(self._video_properties))

    @property
    def report(self) -> dict[str, Any]:
        """Return a copy of the report emitted for this execution state."""
        return cast(dict[str, Any], _thaw_snapshot(self._report))

    def _snapshot_for(
        self, owner: object
    ) -> tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any]]:
        if self._owner is not owner:
            raise ValueError("Video preflight belongs to a different projector")
        return (
            self._video_path,
            self._output_dir,
            cast(dict[str, Any], _thaw_snapshot(self._settings)),
            cast(dict[str, Any], _thaw_snapshot(self._video_properties)),
            cast(dict[str, Any], _thaw_snapshot(self._report)),
        )


class StereoProjector:
    """
    Main class for converting 2D videos to 3D VR format.

    Uses Video-Depth-Anything for temporal consistency across video frames.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = "auto",
        metric: bool = False,
        depth_model_version: str = "v2",
        temporal_window_overlap: int = 10,
        stereo_renderer: StereoRenderer | None = None,
        *,
        model_size: str | None = None,
    ) -> None:
        """
        Initialize StereoProjector.

        Args:
            model_path: Path to video depth estimation model (DA2) or model name (DA3)
            device: Processing device ('auto', 'cuda', 'cpu')
            metric: Use metric depth model (true depth values)
            depth_model_version: Registered depth backend ID
            temporal_window_overlap: Frame overlap for V2 temporal windows (default: 10)
            model_size: Optional registered model variant (vits, vitb, or vitl)
        """
        self.depth_model_version = depth_model_version
        self.model_path = model_path
        self.model_size = model_size
        self.device = device
        self.metric = metric
        self.stereo_renderer = stereo_renderer
        self.backend_spec = get_backend_spec(depth_model_version)
        self.depth_estimator: Any = create_registered_depth_estimator(
            depth_model_version,
            EstimatorRequest(
                model_path=model_path,
                model_size=model_size,
                device=device,
                metric=metric,
                temporal_window_overlap=temporal_window_overlap,
            ),
        )

        self._model_loaded = False
        self._preflight_owner = object()

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        settings: dict[str, Any],
    ) -> bool:
        """
        Process video to create 3D VR version.

        Args:
            video_path: Path to input video
            output_dir: Output directory path
            settings: User-facing processing settings

        Returns:
            True if processing completed successfully
        """
        preflight = self.preflight_video(video_path, output_dir, settings)
        return False if preflight is None else self.execute_video(preflight)

    def execute_video(self, preflight: VideoRunPreflight) -> bool:
        """Execute one owned preflight without accepting competing settings."""
        try:
            video_path, output_dir, settings, video_properties, _report = preflight._snapshot_for(
                self._preflight_owner
            )
            if not self._ensure_model_loaded():
                return False
            if not self._prepare_output_directory(output_dir):
                return False

            # Create video processor (always uses temporal consistency)
            processor = VideoProcessor(self.depth_estimator, verbose=settings.get("verbose", False))

            # Process the video
            return processor.process(
                video_path=video_path,
                output_dir=output_dir,
                video_properties=video_properties,
                settings=settings,
            )

        except Exception as e:
            print(f"Error during video processing: {e}")
            return False

    def preflight_video(
        self,
        video_path: str,
        output_dir: str,
        settings: dict[str, Any],
    ) -> VideoRunPreflight | None:
        """Validate and resolve a video run without loading models or mutating files."""
        return self._build_video_preflight(
            video_path,
            output_dir,
            settings,
            video_properties=None,
            emit_report=True,
            expected_report=None,
        )

    def revalidate_video_preflight(
        self,
        preflight: VideoRunPreflight,
        settings: dict[str, Any],
    ) -> VideoRunPreflight | None:
        """Rebuild an owned context with new settings without probing or reporting twice."""
        try:
            video_path, output_dir, _old_settings, video_properties, report = (
                preflight._snapshot_for(self._preflight_owner)
            )
        except ValueError as error:
            print(f"Error during video preflight: {error}")
            return None
        return self._build_video_preflight(
            video_path,
            output_dir,
            settings,
            video_properties=video_properties,
            emit_report=False,
            expected_report=report,
        )

    def _build_video_preflight(
        self,
        video_path: str,
        output_dir: str,
        settings: dict[str, Any],
        *,
        video_properties: dict[str, Any] | None,
        emit_report: bool,
        expected_report: dict[str, Any] | None,
    ) -> VideoRunPreflight | None:
        try:
            requested_settings = validate_settings(dict(settings), source="explicit")
            requested_settings["video_path"] = video_path
            depth_settings = self._get_depth_settings()
            if "depth_resolution" in requested_settings:
                depth_settings["depth_resolution"] = requested_settings["depth_resolution"]
            requested_settings.update(depth_settings)

            if not self._validate_inputs(video_path, output_dir, requested_settings):
                return None

            if video_properties is None:
                video_properties = get_video_properties(video_path)
            else:
                video_properties = deepcopy(video_properties)
            if not video_properties:
                print(f"Error: Cannot read video properties from {video_path}")
                return None

            resolved_settings = self._resolve_settings(requested_settings, video_properties)
            validate_backend_geometry_request(resolved_settings, video_properties)
            report = build_effective_depth_run_report(
                resolved_settings,
                self.depth_estimator,
            )
            if expected_report is not None and report != expected_report:
                raise ValueError("Migrated settings changed the reported processing configuration")
            if emit_report:
                self._print_effective_run_report(report)
                if resolved_settings["stereo_geometry_mode"] == "metric_camera":
                    print(TEMPORAL_STABILITY_WARNING)
            return VideoRunPreflight(
                _owner=self._preflight_owner,
                _video_path=video_path,
                _output_dir=output_dir,
                _settings=cast(Mapping[str, Any], _freeze_snapshot(resolved_settings)),
                _video_properties=cast(Mapping[str, Any], _freeze_snapshot(video_properties)),
                _report=cast(Mapping[str, Any], _freeze_snapshot(report)),
            )
        except Exception as error:
            print(f"Error during video preflight: {error}")
            return None

    def process_image(self, image_path: str, output_dir: str, **kwargs) -> bool:
        """
        Process single image to create 3D stereo pair.

        NOTE: Video-Depth-Anything is optimized for videos. For best results,
        convert your image to a short video clip first.

        Args:
            image_path: Path to input image
            output_dir: Output directory path
            **kwargs: Processing parameters

        Returns:
            True if processing completed successfully
        """
        print("WARNING: Single image processing is not optimized with Video-Depth-Anything.")
        print("For best results, convert your image to a video first.")
        print("This feature will process the image as a single-frame video.")

        settings = self._apply_default_settings(kwargs)
        for key in ("stereo_strength", "convergence", "occlusion_fill"):
            if kwargs.get(key) is not None:
                settings[key] = kwargs[key]
        if settings.get("stereo_geometry_mode", "relative") == "metric_camera":
            print("Error: metric_camera is supported for video processing only")
            return False

        try:
            # Ensure model is loaded
            if not self._ensure_model_loaded():
                return False

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Cannot load image from {image_path}")
                return False

            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Process as single-frame video
            frames = np.array([image])  # Shape: [1, H, W, 3]

            # Get depth map using video model
            # Higher input_size = better quality but more VRAM
            # Auto-detect or use user-specified depth resolution
            input_size = self._resolve_image_depth_input_size(
                image,
                settings.get("depth_resolution", "auto"),
            )

            depth_batch = self.depth_estimator.estimate_depth_batch(
                frames, target_fps=30, input_size=input_size, fp32=False
            )

            if not isinstance(depth_batch, DepthBatch) or depth_batch.values.shape[0] != 1:
                print("Error: Depth estimator did not return one explicit DepthBatch")
                return False
            canonical = canonicalize_single_scene(depth_batch)[0]

            # Process using simplified pipeline
            from ..utils import (
                resize_image,
                apply_center_crop,
                create_vr_frame,
            )

            per_eye_width = settings.get("per_eye_width", 1920)
            per_eye_height = settings.get("per_eye_height", 1080)

            stereo_strength = float(settings.get("stereo_strength", 2.0))
            geometry = build_relative_geometry(
                canonical,
                (int(image.shape[0]), int(image.shape[1])),
                stereo_strength=stereo_strength,
                convergence=float(settings.get("convergence", 0.5)),
            )
            render_settings = StereoSplatSettings(
                max_eye_shift_fraction=stereo_strength / 200.0,
                occlusion_fill=cast(
                    Literal["none", "background"],
                    str(settings.get("occlusion_fill", "background")),
                ),
            )
            stereo = self._get_stereo_renderer().render_geometry(
                image,
                geometry,
                render_settings,
            )
            left_img = stereo.left_image
            right_img = stereo.right_image

            # Apply center cropping
            left_cropped = apply_center_crop(left_img, settings["crop_factor"])
            right_cropped = apply_center_crop(right_img, settings["crop_factor"])

            # Resize to target dimensions
            left_final = resize_image(left_cropped, per_eye_width, per_eye_height)
            right_final = resize_image(right_cropped, per_eye_width, per_eye_height)

            # Create VR frame
            vr_frame = create_vr_frame(left_final, right_final, settings["vr_format"])

            # Save results
            base_name = Path(image_path).stem
            cv2.imwrite(str(output_path / f"{base_name}_left.png"), left_final)
            cv2.imwrite(str(output_path / f"{base_name}_right.png"), right_final)
            cv2.imwrite(str(output_path / f"{base_name}_vr.png"), vr_frame)
            cv2.imwrite(
                str(output_path / f"{base_name}_depth.png"),
                np.rint(canonical * np.float32(255.0)).astype(np.uint8),
            )

            print(f"Image processing complete. Output saved to: {output_path}")
            return True

        except Exception as e:
            print(f"Error during image processing: {e}")
            traceback.print_exc()
            return False

    def _apply_default_settings(self, params: dict[str, Any]) -> dict[str, Any]:
        """Apply and validate defaults for explicit processing parameters."""
        special_params = {
            "self",
            "video_path",
        }
        provided = {
            key: value
            for key, value in params.items()
            if key not in special_params and value is not None
        }
        settings = validate_settings(provided, source="explicit")

        # Handle special cases
        passthrough_params = [
            "video_path",
            "start_time",
            "end_time",
            "target_fps",
            "min_resolution",
        ]
        for param in passthrough_params:
            if param in params:
                settings[param] = params[param]

        return settings

    def _get_depth_settings(self) -> dict[str, Any]:
        """Describe the effective estimator inputs used to isolate depth caches."""
        processing_resolution = getattr(self.depth_estimator, "processing_resolution", None)
        reported_model_size = normalize_model_size(
            self.depth_model_version,
            model_path=self.model_path,
            model_size=self.model_size,
        )
        return {
            "depth_model_version": self.depth_model_version,
            "model_path": self.model_path,
            "model_size": reported_model_size,
            "depth_resolution": processing_resolution or "auto",
            "use_metric_depth": bool(getattr(self.depth_estimator, "metric", self.metric)),
            "device": str(getattr(self.depth_estimator, "device", self.device)),
            "denoising_steps": getattr(self.depth_estimator, "denoising_steps", None),
            "seed": getattr(self.depth_estimator, "seed", None),
        }

    def _validate_inputs(self, video_path: str, output_dir: str, settings: dict[str, Any]) -> bool:
        """Validate input and output paths without filesystem mutation."""
        del settings
        # Validate video file
        if not validate_video_file(video_path):
            print(f"Error: Invalid or unsupported video file: {video_path}")
            return False

        output_path = Path(output_dir)
        if output_path.exists() and not output_path.is_dir():
            print(f"Error: Output path is not a directory: {output_dir}")
            return False
        return True

    @staticmethod
    def _prepare_output_directory(output_dir: str) -> bool:
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        except Exception as error:
            print(f"Error: Cannot create output directory {output_dir}: {error}")
            return False
        return True

    @staticmethod
    def _print_effective_run_report(report: dict[str, Any]) -> None:
        print("Effective depth run:")
        for name, value in report.items():
            print(f"  {name}: {value}")

    def _ensure_model_loaded(self) -> bool:
        """Ensure the depth estimation model is loaded."""
        if not self._model_loaded:
            if self.depth_estimator.load_model():
                self._model_loaded = True
            else:
                return False
        return True

    def load_model(self) -> bool:
        """Load the configured estimator once for validation or processing."""

        return self._ensure_model_loaded()

    def _get_stereo_renderer(self) -> StereoRenderer:
        if self.stereo_renderer is None:
            estimator_device = str(getattr(self.depth_estimator, "device", self.device))
            render_device = None if estimator_device == "auto" else estimator_device
            self.stereo_renderer = StereoRenderer(device=render_device)
        return self.stereo_renderer

    @staticmethod
    def _resolve_image_depth_input_size(
        image: np.ndarray,
        depth_resolution: object,
    ) -> int:
        if depth_resolution == "auto":
            return max(image.shape[0], image.shape[1])
        if not isinstance(depth_resolution, (str, int)) or isinstance(depth_resolution, bool):
            return 1080
        try:
            return int(depth_resolution)
        except (ValueError, TypeError):
            return 1080

    def _resolve_settings(
        self, settings: dict[str, Any], video_props: dict[str, Any]
    ) -> dict[str, Any]:
        """Resolve and validate settings based on video properties."""
        resolved = settings.copy()

        resolved["depth_resolution"] = resolve_depth_input_size(
            int(video_props["width"]),
            int(video_props["height"]),
            resolved.get("depth_resolution", "auto"),
        )

        # Resolve VR resolution
        if resolved["vr_resolution"] == "auto":
            resolved["vr_resolution"] = auto_detect_resolution(
                video_props["width"], video_props["height"], resolved["vr_format"]
            )

        # Validate resolution settings
        validation = validate_resolution_settings(
            resolved["vr_resolution"],
            resolved["vr_format"],
            video_props["width"],
            video_props["height"],
        )

        if not validation["valid"]:
            print("Warning: Invalid resolution settings")
            for warning in validation["warnings"]:
                print(f"  - {warning}")

        for recommendation in validation["recommendations"]:
            print(f"Recommendation: {recommendation}")

        # Get final resolution dimensions
        per_eye_width, per_eye_height = get_resolution_dimensions(resolved["vr_resolution"])
        vr_output_width, vr_output_height = calculate_vr_output_dimensions(
            per_eye_width, per_eye_height, resolved["vr_format"]
        )

        resolved.update(
            {
                "per_eye_width": per_eye_width,
                "per_eye_height": per_eye_height,
                "vr_output_width": vr_output_width,
                "vr_output_height": vr_output_height,
                "source_width": video_props["width"],
                "source_height": video_props["height"],
                "source_fps": video_props["fps"],
            }
        )

        return resolved

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        return self.depth_estimator.get_model_info()

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        self.depth_estimator.unload_model()
        self._model_loaded = False


def create_stereo_projector(
    model_path: str | None = None,
    device: str = "auto",
    metric: bool = False,
    depth_model_version: str = "v2",
    *,
    model_size: str | None = None,
) -> StereoProjector:
    """
    Factory function to create a StereoProjector instance.

    Args:
        model_path: Path to model file (V2) or model name (V3)
        device: Processing device
        metric: Use metric depth model (true depth values)
        depth_model_version: Registered depth backend ID
        model_size: Optional registered model variant (vits, vitb, or vitl)

    Returns:
        Configured StereoProjector instance
    """
    return StereoProjector(
        model_path,
        device,
        metric,
        depth_model_version,
        model_size=model_size,
    )
