"""
AI upscaling module.

Handles AI-based frame upscaling using Real-ESRGAN or similar models.
"""

from __future__ import annotations

import cv2
import traceback
from pathlib import Path
from typing import Any

from ...core.constants import INTERMEDIATE_DIRS, PREVIEW_FRAME_SAMPLE_RATE
from ...utils.imaging.png_header import read_png_header
from .stage_manifest import (
    build_stage_identity,
    clear_stage_outputs,
    complete_stage,
    stage_is_reusable,
)


UPSCALE_STAGE_ALGORITHM_VERSION = "realesrgan-upscale-v1"


class FrameUpscalerProcessor:
    """
    AI upscaling orchestrator for stereo frames.

    Responsibilities:
    - Model loading and management
    - Batch frame processing
    - GPU memory management during upscaling
    - Directory resolution for source frames
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize frame upscaler processor.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.upscaler = None

    def apply_upscaling(
        self,
        directories: dict[str, Path],
        settings: dict[str, Any],
        progress_tracker=None,
    ) -> bool:
        """
        Apply AI upscaling to stereo frames.

        Args:
            directories: Dictionary of processing directories
            settings: Processing settings with upscale_model parameter
            progress_tracker: Optional progress tracker

        Returns:
            True if successful, False otherwise

        Side effects:
            - Loads upscaling model into GPU memory
            - Reads and writes frame images
            - Modifies GPU memory state
        """
        try:
            if settings.get("upscale_model", "none") == "none":
                return True
            from ...inference import create_upscaler

            # Get source directories (cropped frames)
            source_left, source_right = self._get_upscaling_source_dirs(directories, settings)  # type: ignore[misc]
            if source_left is None or source_right is None:
                return False

            left_files = sorted(source_left.glob("*.png"))
            right_files = sorted(source_right.glob("*.png"))
            if (
                not left_files
                or len(left_files) != len(right_files)
                or [path.stem for path in left_files] != [path.stem for path in right_files]
            ):
                return False
            left_upscaled, right_upscaled = self._get_output_directories(directories)
            identity = self._build_stage_identity(left_files, right_files, settings)
            output_directories = (left_upscaled, right_upscaled)
            if stage_is_reusable(identity, output_directories):
                return True
            clear_stage_outputs(output_directories)

            # Create upscaler
            upscaler = create_upscaler(
                model_name=settings["upscale_model"],
                device=settings.get("device", "auto"),
            )

            if upscaler is None:
                return True  # No upscaling

            if not upscaler.load_model():
                print("Failed to load upscaling model")
                return False

            try:
                return self._process_upscaling_frames(
                    upscaler,
                    source_left,
                    source_right,
                    directories,
                    settings,
                    progress_tracker,
                    stage_identity=identity,
                )
            finally:
                upscaler.unload_model()

        except Exception as e:
            print(f"Error applying upscaling: {e}")

            traceback.print_exc()
            return False

    def _process_upscaling_frames(
        self,
        upscaler,
        left_dir,
        right_dir,
        directories,
        settings,
        progress_tracker,
        *,
        stage_identity: dict[str, Any],
    ) -> bool:
        """
        Process frames in batches with model management.

        Args:
            upscaler: Upscaler instance
            left_dir: Input directory for left frames
            right_dir: Input directory for right frames
            directories: Dictionary of processing directories
            settings: Processing settings
            progress_tracker: Optional progress tracker

        Returns:
            True if successful, False otherwise

        Side effects:
            - Loads/unloads upscaling model
            - GPU memory operations
            - Filesystem I/O
        """
        left_files = sorted(list(left_dir.glob("*.png")))
        right_files = sorted(list(right_dir.glob("*.png")))

        if (
            not left_files
            or len(left_files) != len(right_files)
            or [path.stem for path in left_files] != [path.stem for path in right_files]
        ):
            print(f"Frame count mismatch: {len(left_files)} left, {len(right_files)} right")
            return False

        # Upscaled files feed VR assembly, so they are always working state.
        left_upscaled, right_upscaled = self._get_output_directories(directories)
        identity = stage_identity
        output_directories = (left_upscaled, right_upscaled)

        # Process frames
        for i, (left_file, right_file) in enumerate(zip(left_files, right_files)):
            if not self._upscale_frame_pair(
                upscaler,
                left_file,
                right_file,
                left_upscaled,
                right_upscaled,
                settings,
                i,
                len(left_files),
                progress_tracker,
            ):
                return False

        header = read_png_header(left_upscaled / left_files[0].name)
        return bool(
            header
            and complete_stage(
                identity,
                output_directories,
                shape=(header.height, header.width, header.channels),
                bit_depth=header.bit_depth,
            )
        )

    @staticmethod
    def _build_stage_identity(left_files, right_files, settings):
        return build_stage_identity(
            stage="upscale",
            algorithm_version=UPSCALE_STAGE_ALGORITHM_VERSION,
            frame_names=[path.name for path in left_files],
            source_files=[*left_files, *right_files],
            settings={"upscale_model": settings.get("upscale_model", "unspecified")},
        )

    @staticmethod
    def _get_output_directories(directories):
        left_upscaled = directories.get("left_upscaled") or (
            directories["base"] / INTERMEDIATE_DIRS["left_upscaled"]
        )
        right_upscaled = directories.get("right_upscaled") or (
            directories["base"] / INTERMEDIATE_DIRS["right_upscaled"]
        )
        left_upscaled.mkdir(parents=True, exist_ok=True)
        right_upscaled.mkdir(parents=True, exist_ok=True)
        return left_upscaled, right_upscaled

    def _upscale_frame_pair(
        self,
        upscaler,
        left_file,
        right_file,
        left_upscaled,
        right_upscaled,
        settings,
        frame_idx,
        total_frames,
        progress_tracker,
    ):
        """
        Upscale single frame pair.

        Args:
            upscaler: Upscaler instance
            left_file: Input left frame path
            right_file: Input right frame path
            left_upscaled: Output left directory
            right_upscaled: Output right directory
            settings: Processing settings
            frame_idx: Current frame index
            total_frames: Total number of frames
            progress_tracker: Optional progress tracker

        Side effects:
            - GPU inference
            - Filesystem I/O
        """
        left_img = cv2.imread(str(left_file))
        right_img = cv2.imread(str(right_file))

        if left_img is None or right_img is None:
            print(f"Warning: Could not load {left_file} or {right_file}")
            return False

        # Upscale
        left_upscaled_img = upscaler.upscale_image(left_img)
        right_upscaled_img = upscaler.upscale_image(right_img)

        frame_name = left_file.stem

        # Determine if we should send preview for this frame
        should_send_preview = (
            progress_tracker
            and hasattr(progress_tracker, "send_preview_frame_from_array")
            and (frame_idx % PREVIEW_FRAME_SAMPLE_RATE == 0 or frame_idx == total_frames - 1)
        )

        # Send preview from memory if not keeping intermediates
        if should_send_preview and not settings["keep_intermediates"]:
            progress_tracker.send_preview_frame_from_array(
                left_upscaled_img, "upscaled_left", frame_idx + 1
            )

        writes_succeeded = True
        if left_upscaled:
            left_upscaled_path = left_upscaled / f"{frame_name}.png"
            writes_succeeded = cv2.imwrite(str(left_upscaled_path), left_upscaled_img)

            if (
                settings["keep_intermediates"]
                and should_send_preview
                and hasattr(progress_tracker, "send_preview_frame")
            ):
                progress_tracker.send_preview_frame(
                    left_upscaled_path, "upscaled_left", frame_idx + 1
                )

        if right_upscaled:
            writes_succeeded = bool(
                cv2.imwrite(str(right_upscaled / f"{frame_name}.png"), right_upscaled_img)
                and writes_succeeded
            )

        # Upscaling is slow enough that every completed frame is useful progress.
        if progress_tracker:
            progress_tracker.update_progress(
                f"Upscaling frame {frame_idx+1}/{total_frames}",
                phase="upscaling",
                frame_num=frame_idx + 1,
                step_name="AI Upscaling",
                step_progress=frame_idx + 1,
                step_total=total_frames,
            )
        return writes_succeeded

    @staticmethod
    def _get_upscaling_source_dirs(
        directories: dict[str, Path], settings: dict[str, Any]
    ) -> tuple[Path, Path] | None:
        """
        PURE: Determine source directories for upscaling.

        Args:
            directories: Dictionary of processing directories
            settings: Processing settings with apply_distortion flag

        Returns:
            Tuple of (left_source_dir, right_source_dir)
        """
        # Upscaling happens at Step 6.5, after VR assembly/cropping (Step 6)
        if "left_cropped" in directories and "right_cropped" in directories:
            return directories["left_cropped"], directories["right_cropped"]
        else:
            return None, None  # type: ignore[return-value]
