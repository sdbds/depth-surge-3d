"""
VR frame assembly module.

Assembles stereo pairs into final VR format (side-by-side or over-under).
"""

from __future__ import annotations

import cv2
import traceback
from pathlib import Path
from typing import Any

from ...utils import (
    resize_image,
    create_vr_frame,
)
from .frame_stage_parallelism import (
    calculate_frame_stage_workers,
    max_png_frame_pair_pixels,
    run_ordered_frame_tasks,
)
from .stage_manifest import (
    build_stage_identity,
    clear_stage_outputs,
    complete_stage,
    stage_is_reusable,
)


VR_STAGE_ALGORITHM_VERSION = "vr-layout-v1"


class VRFrameAssembler:
    """
    Assembles stereo frames into final VR format.

    Responsibilities:
    - Combine left/right frames into VR format
    - Support side-by-side and over-under layouts
    - Directory resolution for source frames
    - Batch frame processing
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize VR frame assembler.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose

    def assemble_vr_frames(
        self,
        directories: dict[str, Path],
        settings: dict[str, Any],
        progress_tracker=None,
        total_frames: int = 0,
    ) -> bool:
        """
        Assemble stereo frames into final VR format.

        Args:
            directories: Dictionary of processing directories
            settings: Processing settings with vr_format parameter
            progress_tracker: Optional progress tracker
            total_frames: Total number of frames (for progress tracking)

        Returns:
            True if successful, False otherwise

        Side effects:
            - Reads stereo frame pairs from disk
            - Writes assembled VR frames to disk
        """
        try:
            source_files = self._vr_source_files(directories, settings, total_frames)
            if source_files is None:
                return False
            left_files, right_files = source_files
            output_dir = directories.get("vr_frames")
            if output_dir is None:
                return False
            identity = build_stage_identity(
                stage="vr_assembly",
                algorithm_version=VR_STAGE_ALGORITHM_VERSION,
                frame_names=[path.name for path in left_files],
                source_files=[*left_files, *right_files],
                settings={
                    "vr_format": settings["vr_format"],
                    "per_eye_width": settings["per_eye_width"],
                    "per_eye_height": settings["per_eye_height"],
                },
            )
            output_directories = (output_dir,)
            if stage_is_reusable(identity, output_directories):
                return True
            clear_stage_outputs(output_directories)

            max_source_eye_pixels = max_png_frame_pair_pixels(left_files, right_files)
            if max_source_eye_pixels is None:
                return False
            target_eye_pixels = int(settings["per_eye_width"]) * int(settings["per_eye_height"])
            worker_count = calculate_frame_stage_workers(
                len(left_files), max(max_source_eye_pixels, target_eye_pixels) * 48
            )
            self._run_assembly_workers(
                left_files,
                right_files,
                directories,
                settings,
                worker_count,
                progress_tracker,
            )

            return complete_stage(
                identity,
                output_directories,
                shape=self._vr_output_shape(settings),
            )

        except Exception as e:
            print(f"Error assembling VR frames: {e}")

            traceback.print_exc()
            return False

    def _vr_source_files(
        self,
        directories: dict[str, Path],
        settings: dict[str, Any],
        total_frames: int,
    ) -> tuple[list[Path], list[Path]] | None:
        source_dirs = self._get_vr_assembly_source_dirs(directories, settings)
        if source_dirs is None:
            return None
        left_files = sorted(source_dirs[0].glob("*.png"))
        right_files = sorted(source_dirs[1].glob("*.png"))
        if not self._source_frame_manifest_is_complete(left_files, right_files, total_frames):
            print("Error: VR source frame manifest is incomplete")
            return None
        return left_files, right_files

    def _run_assembly_workers(
        self,
        left_files: list[Path],
        right_files: list[Path],
        directories: dict[str, Path],
        settings: dict[str, Any],
        worker_count: int,
        progress_tracker,
    ) -> None:
        def assemble_one(item) -> Path:
            left_file, right_file = item
            vr_path = self._assemble_single_vr_frame(left_file, right_file, directories, settings)
            if vr_path is None:
                raise OSError(f"Could not assemble VR frame: {left_file}")
            return vr_path

        run_ordered_frame_tasks(
            zip(left_files, right_files),
            assemble_one,
            worker_count=worker_count,
            on_ordered_result=lambda index, vr_path: self._report_assembly_progress(
                progress_tracker, vr_path, index, len(left_files)
            ),
        )

    @staticmethod
    def _report_assembly_progress(progress_tracker, vr_path: Path, index: int, total: int) -> None:
        if progress_tracker and hasattr(progress_tracker, "send_preview_frame"):
            progress_tracker.send_preview_frame(vr_path, "vr_frame", index + 1)
        if progress_tracker:
            progress_tracker.update_progress(
                f"Assembling VR frame {index + 1}/{total}",
                phase="vr_assembly",
                frame_num=index + 1,
                step_name="VR Assembly",
                step_progress=index + 1,
                step_total=total,
            )

    @staticmethod
    def _source_frame_manifest_is_complete(
        left_files: list[Path], right_files: list[Path], total_frames: int
    ) -> bool:
        return (
            bool(left_files)
            and len(left_files) == len(right_files)
            and (total_frames <= 0 or len(left_files) == total_frames)
            and [path.stem for path in left_files] == [path.stem for path in right_files]
        )

    @staticmethod
    def _vr_output_shape(settings: dict[str, Any]) -> tuple[int, int, int]:
        per_eye_height = int(settings["per_eye_height"])
        per_eye_width = int(settings["per_eye_width"])
        if settings["vr_format"] == "side_by_side":
            return per_eye_height, per_eye_width * 2, 3
        return per_eye_height * 2, per_eye_width, 3

    def _assemble_single_vr_frame(
        self,
        left_file: Path,
        right_file: Path,
        directories: dict[str, Path],
        settings: dict[str, Any],
    ) -> Path | None:
        """
        Assemble a single VR frame from cropped/upscaled frames.

        Args:
            left_file: Left frame file path
            right_file: Right frame file path
            directories: Dictionary of processing directories
            settings: Processing settings with vr_format, per_eye_width, per_eye_height

        Returns:
            Written VR frame path, or None on failure

        Side effects:
            - Reads images from disk
            - Writes VR frame to disk
        """
        # Load images
        left_img = cv2.imread(str(left_file))
        right_img = cv2.imread(str(right_file))

        if left_img is None or right_img is None:
            print(f"Warning: Could not load {left_file} or {right_file}")
            return None

        target_shape = (int(settings["per_eye_height"]), int(settings["per_eye_width"]))
        left_final = (
            left_img
            if left_img.shape[:2] == target_shape
            else resize_image(left_img, settings["per_eye_width"], settings["per_eye_height"])
        )
        right_final = (
            right_img
            if right_img.shape[:2] == target_shape
            else resize_image(right_img, settings["per_eye_width"], settings["per_eye_height"])
        )

        # Create and save final VR frame
        vr_frame = create_vr_frame(left_final, right_final, settings["vr_format"])
        frame_name = left_file.stem
        output_dir = directories.get("vr_frames")
        if output_dir is None:
            return None
        vr_path = output_dir / f"{frame_name}.png"
        if not cv2.imwrite(str(vr_path), vr_frame):
            return None

        return vr_path

    @staticmethod
    def _get_vr_assembly_source_dirs(
        directories: dict[str, Path], settings: dict[str, Any]
    ) -> tuple[Path, Path] | None:
        """
        PURE: Determine source directories for VR assembly.

        Args:
            directories: Dictionary of processing directories
            settings: Processing settings with upscale_model parameter

        Returns:
            Tuple of (left_source_dir, right_source_dir) or None if not found
        """
        if settings.get("upscale_model", "none") != "none":
            left_dir = directories.get("left_upscaled")
            right_dir = directories.get("right_upscaled")
            if left_dir is not None and right_dir is not None:
                if left_dir.is_dir() and right_dir.is_dir() and any(left_dir.glob("*.png")):
                    return left_dir, right_dir
            print("Error: Upscaling is enabled but upscaled frames are unavailable")
            return None

        # Fallback to cropped frames
        if "left_cropped" in directories and "right_cropped" in directories:
            left_dir = directories["left_cropped"]
            right_dir = directories["right_cropped"]
            if left_dir.exists() and right_dir.exists():
                left_files = list(left_dir.glob("*.png"))
                if left_files:  # Verify directory has frames
                    return left_dir, right_dir

        print("Error: No cropped or upscaled frames found for VR assembly")
        return None
