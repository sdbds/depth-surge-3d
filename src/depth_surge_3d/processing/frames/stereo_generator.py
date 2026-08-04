"""
Stereo pair generation module.

Converts frames and depth maps into stereo (left/right) pairs using disparity mapping.
"""

from __future__ import annotations

import cv2
import json
import multiprocessing as mp
import numpy as np
import traceback
from pathlib import Path
from typing import Any

from ...utils import (
    depth_to_disparity,
    create_shifted_image,
    hole_fill_image,
)
from ...core.constants import PREVIEW_FRAME_SAMPLE_RATE


def _process_single_stereo_pair(
    args: tuple[np.ndarray, np.ndarray, str, str | None, str | None, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    PURE worker function to process a single stereo pair in parallel.

    Args:
        args: Tuple of (frame, depth_map, frame_name, left_path, right_path, settings)

    Returns:
        Tuple of (left_img, right_img, frame_name)

    Side effects:
        - Writes to left_path and right_path if provided
    """
    frame, depth_map, frame_name, left_path, right_path, settings = args

    frame_height, frame_width = frame.shape[:2]
    if depth_map.shape[:2] != (frame_height, frame_width):
        depth_map = cv2.resize(
            depth_map,
            (frame_width, frame_height),
            interpolation=cv2.INTER_LINEAR,
        )

    # Create stereo pair
    disparity_map = depth_to_disparity(depth_map, settings["baseline"], settings["focal_length"])

    left_img = create_shifted_image(frame, disparity_map, "left")
    right_img = create_shifted_image(frame, disparity_map, "right")

    # Apply hole filling
    if settings["hole_fill_quality"] in ["fast", "advanced"]:
        left_img = hole_fill_image(left_img, method=settings["hole_fill_quality"])
        right_img = hole_fill_image(right_img, method=settings["hole_fill_quality"])

    # Save if paths provided
    if left_path:
        cv2.imwrite(left_path, left_img)
    if right_path:
        cv2.imwrite(right_path, right_img)

    return left_img, right_img, frame_name


def _process_single_stereo_pair_from_files(
    args: tuple[str, str, str, str, str, dict[str, Any], float | None],
) -> str:
    """Load one frame/depth pair, write stereo outputs, and return only its name."""
    frame_path, depth_path, frame_name, left_path, right_path, settings, depth_scale = args
    frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise OSError(f"Could not load frame: {frame_path}")
    if depth_map is None:
        raise OSError(f"Could not load depth map: {depth_path}")

    if depth_scale is None:
        depth_scale = 65535.0 if depth_map.dtype == np.uint16 else 255.0
    depth_float = depth_map.astype(np.float32) / depth_scale
    _process_single_stereo_pair((frame, depth_float, frame_name, left_path, right_path, settings))
    return frame_name


class StereoPairGenerator:
    """
    Generates stereo pairs from frames and depth maps.

    Responsibilities:
    - Convert depth maps to disparity maps
    - Create left/right shifted images
    - Apply hole filling
    - Parallel processing orchestration
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize stereo pair generator.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose

    def create_stereo_pairs(
        self,
        frames: np.ndarray,
        depth_maps: np.ndarray,
        frame_files: list[Path],
        directories: dict[str, Path],
        settings: dict[str, Any],
        progress_tracker=None,
    ) -> bool:
        """
        Generate stereo pairs using multiprocessing.

        Args:
            frames: Array of frame images
            depth_maps: Array of depth map images
            frame_files: List of frame file paths
            directories: Dictionary of processing directories
            settings: Processing settings with baseline, focal_length, etc.
            progress_tracker: Optional progress tracker

        Returns:
            True if successful, False otherwise

        Side effects:
            - Parallel processing via multiprocessing
            - Writes stereo pair images to disk
        """
        try:
            # Each worker holds full-resolution frame and depth arrays; keep memory bounded.
            num_workers = min(4, max(1, mp.cpu_count() - 2))
            print(f"  Using {num_workers} parallel workers for stereo generation...")

            # Prepare arguments for parallel processing
            args_list = []
            for frame, depth_map, frame_file in zip(frames, depth_maps, frame_files):
                frame_name = frame_file.stem

                # Determine save paths
                left_path = (
                    str(directories["left_frames"] / f"{frame_name}.png")
                    if settings["keep_intermediates"] and "left_frames" in directories
                    else None
                )
                right_path = (
                    str(directories["right_frames"] / f"{frame_name}.png")
                    if settings["keep_intermediates"] and "right_frames" in directories
                    else None
                )

                args_list.append((frame, depth_map, frame_name, left_path, right_path, settings))

            # Process stereo pairs in parallel
            with mp.Pool(processes=num_workers) as pool:
                # Use imap for progress tracking (processes in order, yields results as ready)
                results = []
                for i, result in enumerate(pool.imap(_process_single_stereo_pair, args_list)):
                    results.append(result)

                    # Update progress
                    if progress_tracker and (i % 5 == 0 or i == len(args_list) - 1):
                        progress_tracker.update_progress(
                            "Creating stereo pairs",
                            phase="stereo_generation",
                            frame_num=i + 1,
                            step_name="Stereo Pair Creation",
                            step_progress=i + 1,
                            step_total=len(frames),
                        )

                    # Send preview frame for left eye
                    if progress_tracker and hasattr(progress_tracker, "send_preview_frame"):
                        if i % PREVIEW_FRAME_SAMPLE_RATE == 0 or i == len(args_list) - 1:
                            left_path = args_list[i][3]  # left_path from args
                            if left_path:
                                progress_tracker.send_preview_frame(
                                    Path(left_path), "stereo_left", i + 1
                                )

            return True

        except Exception as e:
            print(f"Error creating stereo pairs: {e}")
            traceback.print_exc()
            return False

    def create_stereo_pairs_from_files(
        self,
        frame_files: list[Path],
        depth_files: list[Path],
        directories: dict[str, Path],
        settings: dict[str, Any],
        progress_tracker=None,
    ) -> bool:
        """Generate stereo pairs without loading the complete video into memory."""
        if len(frame_files) != len(depth_files):
            print(
                f"Error: Frame/depth count mismatch: {len(frame_files)} frames, "
                f"{len(depth_files)} depth maps"
            )
            return False

        try:
            left_dir = directories["left_frames"]
            right_dir = directories["right_frames"]
            left_dir.mkdir(parents=True, exist_ok=True)
            right_dir.mkdir(parents=True, exist_ok=True)
            depth_scale = self._get_depth_file_scale(depth_files)

            args_list, completed = self._build_file_work_items(
                frame_files,
                depth_files,
                left_dir,
                right_dir,
                settings,
                depth_scale,
            )

            if not args_list:
                print(f"  Reusing {completed} existing stereo pairs")
                return True

            num_workers = min(4, max(1, mp.cpu_count() - 2))
            print(f"  Using {num_workers} parallel workers for stereo generation...")
            total_frames = len(frame_files)
            with mp.Pool(processes=num_workers) as pool:
                for i, _frame_name in enumerate(
                    pool.imap(_process_single_stereo_pair_from_files, args_list)
                ):
                    processed = completed + i + 1
                    if progress_tracker and (i % 5 == 0 or i == len(args_list) - 1):
                        progress_tracker.update_progress(
                            "Creating stereo pairs",
                            phase="stereo_generation",
                            frame_num=processed,
                            step_name="Stereo Pair Creation",
                            step_progress=processed,
                            step_total=total_frames,
                        )

                    if progress_tracker and hasattr(progress_tracker, "send_preview_frame"):
                        if i % PREVIEW_FRAME_SAMPLE_RATE == 0 or i == len(args_list) - 1:
                            progress_tracker.send_preview_frame(
                                Path(args_list[i][3]), "stereo_left", processed
                            )
            return True
        except Exception as e:
            print(f"Error creating stereo pairs: {e}")
            traceback.print_exc()
            return False

    @staticmethod
    def _build_file_work_items(
        frame_files: list[Path],
        depth_files: list[Path],
        left_dir: Path,
        right_dir: Path,
        settings: dict[str, Any],
        depth_scale: float | None,
    ) -> tuple[list[tuple[str, str, str, str, str, dict[str, Any], float | None]], int]:
        args_list = []
        completed = 0
        for frame_file, depth_file in zip(frame_files, depth_files):
            frame_name = frame_file.stem
            left_path = left_dir / f"{frame_name}.png"
            right_path = right_dir / f"{frame_name}.png"
            if left_path.is_file() and right_path.is_file():
                completed += 1
                continue
            args_list.append(
                (
                    str(frame_file),
                    str(depth_file),
                    frame_name,
                    str(left_path),
                    str(right_path),
                    settings,
                    depth_scale,
                )
            )
        return args_list, completed

    @staticmethod
    def _get_depth_file_scale(depth_files: list[Path]) -> float | None:
        """Read cache encoding metadata once; local depth PNGs use dtype-based scaling."""
        if not depth_files:
            return None
        metadata_file = depth_files[0].parent / "metadata.json"
        if not metadata_file.is_file():
            return None
        try:
            metadata = json.loads(metadata_file.read_text())
            return float(metadata.get("depth_scale", 1000.0))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
