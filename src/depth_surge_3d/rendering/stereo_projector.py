"""
Main StereoProjector class for 2D to 3D VR conversion.

This module provides the main orchestration class that coordinates all
processing steps using the modular utility functions.
"""

from __future__ import annotations

import cv2
import numpy as np
import traceback
from pathlib import Path
from typing import Any, Literal, cast

from ..inference.depth.backend_registry import (
    EstimatorRequest,
    create_registered_depth_estimator,
    get_backend_spec,
)
from ..inference.depth.types import DepthBatch
from ..utils import (
    get_resolution_dimensions,
    calculate_vr_output_dimensions,
    validate_resolution_settings,
    auto_detect_resolution,
)
from ..io.operations import (
    validate_video_file,
    get_video_properties,
)
from ..processing import VideoProcessor
from ..processing.frames.depth_normalizer import canonicalize_single_scene
from ..core.settings import validate_settings
from .stereo_geometry import build_relative_geometry
from .stereo_renderer import StereoRenderer, StereoSplatSettings


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
        try:
            requested_settings = validate_settings(dict(settings), source="explicit")
            requested_settings["video_path"] = video_path
            depth_settings = self._get_depth_settings()
            if "depth_resolution" in requested_settings:
                depth_settings["depth_resolution"] = requested_settings["depth_resolution"]
            requested_settings.update(depth_settings)

            # Validate inputs
            if not self._validate_inputs(video_path, output_dir, requested_settings):
                return False

            # Ensure model is loaded
            if not self._ensure_model_loaded():
                return False

            # Get video properties
            video_props = get_video_properties(video_path)
            if not video_props:
                print(f"Error: Cannot read video properties from {video_path}")
                return False

            # Validate and resolve settings
            resolved_settings = self._resolve_settings(requested_settings, video_props)

            # Create video processor (always uses temporal consistency)
            processor = VideoProcessor(
                self.depth_estimator, verbose=resolved_settings.get("verbose", False)
            )

            # Process the video
            return processor.process(
                video_path=video_path,
                output_dir=output_dir,
                video_properties=video_props,
                settings=resolved_settings,
            )

        except Exception as e:
            print(f"Error during video processing: {e}")
            return False

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
        model_path = self.model_path
        if self.depth_model_version == "see_through":
            model_path = getattr(self.depth_estimator, "repo_id", model_path)
        elif self.depth_model_version == "v3":
            model_path = getattr(self.depth_estimator, "model_name", model_path)

        processing_resolution = getattr(self.depth_estimator, "processing_resolution", None)
        return {
            "depth_model_version": self.depth_model_version,
            "model_path": model_path,
            "model_size": self.depth_estimator.get_model_size(),
            "depth_resolution": processing_resolution or "auto",
            "use_metric_depth": bool(getattr(self.depth_estimator, "metric", self.metric)),
            "device": str(getattr(self.depth_estimator, "device", self.device)),
            "denoising_steps": getattr(self.depth_estimator, "denoising_steps", None),
            "seed": getattr(self.depth_estimator, "seed", None),
        }

    def _validate_inputs(self, video_path: str, output_dir: str, settings: dict[str, Any]) -> bool:
        """Validate input parameters."""
        # Validate video file
        if not validate_video_file(video_path):
            print(f"Error: Invalid or unsupported video file: {video_path}")
            return False

        # Validate output directory
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error: Cannot create output directory {output_dir}: {e}")
            return False

        return True

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
