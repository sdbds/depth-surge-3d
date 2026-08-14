"""
VR distortion processing module.

Applies fisheye distortion and cropping to stereo frames for VR viewing.
"""

from __future__ import annotations

import cv2
import os
import shutil
import traceback
from pathlib import Path
from typing import Any

from ...utils import (
    apply_fisheye_square_crop,
    apply_center_crop,
    calculate_fisheye_coordinates,
    remap_fisheye,
)
from ...utils.imaging.png_header import read_png_header
from .frame_stage_parallelism import (
    calculate_frame_stage_workers,
    max_png_frame_pair_pixels,
    run_ordered_frame_tasks,
    uniform_png_frame_pair_header,
)
from .stage_manifest import (
    build_stage_identity,
    clear_stage_outputs,
    complete_stage,
    stage_is_reusable,
)


DISTORTION_STAGE_ALGORITHM_VERSION = "fisheye-distortion-v1"
CROP_STAGE_ALGORITHM_VERSION = "vr-crop-v1"


class DistortionProcessor:
    """
    Applies VR fisheye distortion and cropping to stereo frames.

    Responsibilities:
    - Apply fisheye distortion for VR headsets
    - Crop frames to VR specifications
    - Batch process frame pairs
    - Source directory resolution
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize distortion processor.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose

    def apply_distortion(
        self,
        left_files: list[Path],
        right_files: list[Path],
        directories: dict[str, Path],
        settings: dict[str, Any],
        progress_tracker=None,
    ) -> bool:
        """
        Apply fisheye distortion to stereo frames.

        Args:
            left_files: List of left frame file paths
            right_files: List of right frame file paths
            directories: Dictionary of processing directories
            settings: Processing settings with distortion parameters
            progress_tracker: Optional progress tracker

        Returns:
            True if successful, False otherwise

        Side effects:
            - Reads frames from source directories
            - Writes distorted frames to output directories
        """
        try:
            if not self._frame_pair_manifest_is_complete(left_files, right_files):
                return False
            outputs = self._distortion_outputs(directories)
            if outputs is None:
                return False
            left_output, right_output = outputs

            identity = build_stage_identity(
                stage="distortion",
                algorithm_version=DISTORTION_STAGE_ALGORITHM_VERSION,
                frame_names=[path.name for path in left_files],
                source_files=[*left_files, *right_files],
                settings={
                    "fisheye_fov": settings["fisheye_fov"],
                    "fisheye_projection": settings["fisheye_projection"],
                },
            )
            output_directories = (left_output, right_output)
            if stage_is_reusable(identity, output_directories):
                return True
            clear_stage_outputs(output_directories)

            source_header = uniform_png_frame_pair_header(left_files, right_files)
            if source_header is None:
                return False
            x_map, y_map = calculate_fisheye_coordinates(
                source_header.width,
                source_header.height,
                settings["fisheye_fov"],
                settings["fisheye_projection"],
            )
            x_map.setflags(write=False)
            y_map.setflags(write=False)
            worker_count = calculate_frame_stage_workers(
                len(left_files), source_header.width * source_header.height * 48
            )
            self._run_distortion_frame_pairs(
                left_files,
                right_files,
                left_output,
                right_output,
                x_map,
                y_map,
                worker_count,
                progress_tracker,
            )
            return self._complete_pair_stage(
                identity, output_directories, left_output / left_files[0].name
            )

        except Exception as e:
            print(f"Error applying distortion: {e}")
            return False

    @staticmethod
    def _frame_pair_manifest_is_complete(
        left_files: list[Path], right_files: list[Path], total_frames: int = 0
    ) -> bool:
        return bool(
            left_files
            and len(left_files) == len(right_files)
            and (total_frames <= 0 or len(left_files) == total_frames)
            and [path.stem for path in left_files] == [path.stem for path in right_files]
        )

    @staticmethod
    def _distortion_outputs(directories: dict[str, Path]) -> tuple[Path, Path] | None:
        left_output = directories.get("left_distorted")
        right_output = directories.get("right_distorted")
        if left_output is None or right_output is None:
            return None
        if not left_output.is_dir() or not right_output.is_dir():
            return None
        return left_output, right_output

    def _run_distortion_frame_pairs(
        self,
        left_files: list[Path],
        right_files: list[Path],
        left_output: Path,
        right_output: Path,
        x_map,
        y_map,
        worker_count: int,
        progress_tracker,
    ) -> None:
        def distort_one(item) -> None:
            left_file, right_file = item
            if not self._distort_single_frame_pair(
                left_file,
                right_file,
                left_output,
                right_output,
                x_map,
                y_map,
            ):
                raise OSError(f"Could not distort frame pair: {left_file}")

        run_ordered_frame_tasks(
            zip(left_files, right_files),
            distort_one,
            worker_count=worker_count,
            on_ordered_result=lambda index, _result: self._report_distortion_progress(
                progress_tracker, index, len(left_files)
            ),
        )

    @staticmethod
    def _report_distortion_progress(progress_tracker, index: int, total: int) -> None:
        if progress_tracker:
            progress_tracker.update_progress(
                "Applying distortion",
                phase="distortion",
                frame_num=index + 1,
                step_name="Fisheye Distortion",
                step_progress=index + 1,
                step_total=total,
            )

    @staticmethod
    def _complete_pair_stage(identity, output_directories, sample_path: Path) -> bool:
        header = read_png_header(sample_path)
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
    def _distort_single_frame_pair(
        left_file: Path,
        right_file: Path,
        left_output: Path,
        right_output: Path,
        x_map,
        y_map,
    ) -> bool:
        left_img = cv2.imread(str(left_file))
        right_img = cv2.imread(str(right_file))
        if left_img is None or right_img is None:
            print(f"Error: Could not load {left_file} or {right_file}")
            return False

        left_distorted = remap_fisheye(left_img, x_map, y_map)
        right_distorted = remap_fisheye(right_img, x_map, y_map)
        frame_name = left_file.stem
        return bool(
            cv2.imwrite(str(left_output / f"{frame_name}.png"), left_distorted)
            and cv2.imwrite(str(right_output / f"{frame_name}.png"), right_distorted)
        )

    def crop_frames(
        self,
        directories: dict[str, Path],
        settings: dict[str, Any],
        progress_tracker=None,
        total_frames: int = 0,
    ) -> bool:
        """
        Crop frames to VR specifications.

        Args:
            directories: Dictionary of processing directories
            settings: Processing settings with crop parameters
            progress_tracker: Optional progress tracker
            total_frames: Total number of frames (for progress tracking)

        Returns:
            True if successful, False otherwise

        Side effects:
            - Reads frames from source directories
            - Writes cropped frames to output directories
        """
        try:
            source_files = self._crop_source_files(directories, settings, total_frames)
            if source_files is None:
                return False
            left_files, right_files = source_files
            outputs = self._crop_outputs(directories)
            if outputs is None:
                return False
            left_output, right_output = outputs
            identity = build_stage_identity(
                stage="crop",
                algorithm_version=CROP_STAGE_ALGORITHM_VERSION,
                frame_names=[path.name for path in left_files],
                source_files=[*left_files, *right_files],
                settings={
                    "apply_distortion": bool(settings["apply_distortion"]),
                    "crop_factor": settings.get("crop_factor"),
                    "fisheye_crop_factor": settings.get("fisheye_crop_factor"),
                    "per_eye_width": settings.get("per_eye_width"),
                    "per_eye_height": settings.get("per_eye_height"),
                },
            )
            output_directories = (left_output, right_output)
            if stage_is_reusable(identity, output_directories):
                return True
            clear_stage_outputs(output_directories)

            if self._crop_is_no_op(settings):
                succeeded = self._materialize_crop_frame_pairs(
                    left_files,
                    right_files,
                    left_output,
                    right_output,
                    progress_tracker,
                )
            else:
                succeeded = self._run_transformed_crop(
                    left_files,
                    right_files,
                    directories,
                    settings,
                    progress_tracker,
                )
            if not succeeded:
                return False
            return self._complete_pair_stage(
                identity, output_directories, left_output / left_files[0].name
            )

        except Exception as e:
            print(f"Error cropping frames: {e}")

            traceback.print_exc()
            return False

    def _crop_source_files(
        self,
        directories: dict[str, Path],
        settings: dict[str, Any],
        total_frames: int,
    ) -> tuple[list[Path], list[Path]] | None:
        stereo_dirs = self._get_stereo_source_dirs(directories, settings)
        if stereo_dirs is None:
            return None
        left_files = sorted(stereo_dirs[0].glob("*.png"))
        right_files = sorted(stereo_dirs[1].glob("*.png"))
        if not self._frame_pair_manifest_is_complete(left_files, right_files, total_frames):
            print("Error: Crop source frame manifest is incomplete")
            return None
        return left_files, right_files

    @staticmethod
    def _crop_outputs(directories: dict[str, Path]) -> tuple[Path, Path] | None:
        left_output = directories.get("left_cropped")
        right_output = directories.get("right_cropped")
        if left_output is None or right_output is None:
            return None
        return left_output, right_output

    @staticmethod
    def _crop_is_no_op(settings: dict[str, Any]) -> bool:
        crop_factor = max(0.5, min(1.0, float(settings.get("crop_factor", 1.0))))
        return not settings["apply_distortion"] and crop_factor == 1.0

    def _materialize_crop_frame_pairs(
        self,
        left_files: list[Path],
        right_files: list[Path],
        left_output: Path,
        right_output: Path,
        progress_tracker,
    ) -> bool:
        for index, (left_file, right_file) in enumerate(zip(left_files, right_files)):
            left_succeeded = self._materialize_unchanged_frame(
                left_file, left_output / left_file.name
            )
            right_succeeded = self._materialize_unchanged_frame(
                right_file, right_output / right_file.name
            )
            if not left_succeeded or not right_succeeded:
                return False
            self._report_crop_progress(progress_tracker, index, len(left_files))
        return True

    def _run_transformed_crop(
        self,
        left_files: list[Path],
        right_files: list[Path],
        directories: dict[str, Path],
        settings: dict[str, Any],
        progress_tracker,
    ) -> bool:
        max_source_eye_pixels = max_png_frame_pair_pixels(left_files, right_files)
        if max_source_eye_pixels is None:
            return False
        target_eye_pixels = int(settings.get("per_eye_width") or 0) * int(
            settings.get("per_eye_height") or 0
        )
        estimated_item_bytes = max(max_source_eye_pixels, target_eye_pixels) * 48
        worker_count = calculate_frame_stage_workers(len(left_files), estimated_item_bytes)

        def crop_one(item) -> None:
            left_file, right_file = item
            if not self._crop_single_frame_pair(left_file, right_file, directories, settings):
                raise OSError(f"Could not crop frame pair: {left_file}")

        run_ordered_frame_tasks(
            zip(left_files, right_files),
            crop_one,
            worker_count=worker_count,
            on_ordered_result=lambda index, _result: self._report_crop_progress(
                progress_tracker, index, len(left_files)
            ),
        )
        return True

    @staticmethod
    def _materialize_unchanged_frame(source: Path, destination: Path) -> bool:
        try:
            os.link(source, destination)
        except OSError:
            try:
                shutil.copy2(source, destination)
            except OSError:
                return False
        return True

    @staticmethod
    def _report_crop_progress(progress_tracker, index: int, total: int) -> None:
        if progress_tracker:
            progress_tracker.update_progress(
                f"Cropping frame {index + 1}/{total}",
                phase="cropping",
                frame_num=index + 1,
                step_name="Frame Cropping",
                step_progress=index + 1,
                step_total=total,
            )

    def _crop_single_frame_pair(
        self,
        left_file: Path,
        right_file: Path,
        directories: dict[str, Path],
        settings: dict[str, Any],
    ) -> bool:
        """
        Crop a single stereo frame pair.

        Args:
            left_file: Left frame file path
            right_file: Right frame file path
            directories: Dictionary of processing directories
            settings: Processing settings with crop parameters

        Returns:
            True if successful, False otherwise

        Side effects:
            - Reads images from disk
            - Writes cropped images to disk
        """
        # Load images
        left_img = cv2.imread(str(left_file))
        right_img = cv2.imread(str(right_file))

        if left_img is None or right_img is None:
            print(f"Warning: Could not load {left_file} or {right_file}")
            return False

        # Crop based on distortion setting
        crop_factor = (
            max(0.5, min(2.0, float(settings.get("fisheye_crop_factor", 0.7))))
            if settings["apply_distortion"]
            else max(0.5, min(1.0, float(settings.get("crop_factor", 1.0))))
        )

        if settings["apply_distortion"]:
            left_cropped = apply_fisheye_square_crop(
                left_img,
                settings["per_eye_width"],
                settings["per_eye_height"],
                crop_factor,
            )
            right_cropped = apply_fisheye_square_crop(
                right_img,
                settings["per_eye_width"],
                settings["per_eye_height"],
                crop_factor,
            )
        else:
            left_cropped = apply_center_crop(left_img, crop_factor)
            right_cropped = apply_center_crop(right_img, crop_factor)

        # Save cropped frames (always - needed for upscaling or VR assembly)
        frame_name = left_file.stem
        left_output = directories.get("left_cropped")
        right_output = directories.get("right_cropped")
        if left_output is None or right_output is None:
            return False
        return bool(
            cv2.imwrite(str(left_output / f"{frame_name}.png"), left_cropped)
            and cv2.imwrite(str(right_output / f"{frame_name}.png"), right_cropped)
        )

    @staticmethod
    def _get_stereo_source_dirs(
        directories: dict[str, Path], settings: dict[str, Any]
    ) -> tuple[Path, Path] | None:
        """
        PURE: Determine source directories for cropping.

        Args:
            directories: Dictionary of processing directories
            settings: Processing settings with apply_distortion flag

        Returns:
            Tuple of (left_source_dir, right_source_dir) or None if not found
        """
        if settings.get("apply_distortion"):
            left_dir = directories.get("left_distorted")
            right_dir = directories.get("right_distorted")
            if left_dir is not None and right_dir is not None:
                if left_dir.is_dir() and right_dir.is_dir() and any(left_dir.glob("*.png")):
                    return left_dir, right_dir
            print("Error: Distortion is enabled but distorted frames are unavailable")
            return None

        if "left_frames" in directories and "right_frames" in directories:
            left_dir = directories["left_frames"]
            right_dir = directories["right_frames"]
            if left_dir.exists() and right_dir.exists():
                left_files = list(left_dir.glob("*.png"))
                if left_files:  # Verify directory has frames
                    return left_dir, right_dir

        print("Error: No stereo frames found")
        return None
