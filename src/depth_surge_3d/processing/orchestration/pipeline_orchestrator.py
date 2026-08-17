"""
Processing pipeline orchestrator.

Coordinates the complete video processing pipeline across all specialized processors.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from ...io.operations import (
    cleanup_intermediate_files,
    create_output_directories,
    save_processing_settings,
    update_processing_status,
)
from ...io.job_lock import JobWriterLock
from ...core.render_disparity import validate_render_disparity_input
from ...utils.path_utils import generate_output_filename
from ...utils import (
    step_complete,
    saved_to,
    title_bar,
    completion_banner,
)


class ProcessingOrchestrator:
    """
    High-level pipeline control and step sequencing.

    Responsibilities:
    - Pipeline execution flow
    - Step sequencing
    - Progress tracking coordination
    - Error handling
    - Settings management
    - Console output formatting
    """

    def __init__(
        self,
        depth_processor,
        stereo_generator,
        distortion_processor,
        upscaler,
        vr_assembler,
        video_encoder,
        verbose: bool = False,
        release_depth_model: Callable[[], None] | None = None,
        temporal_stabilizer=None,
        preselected_depth_files: list[Path] | None = None,
        preselected_render_source: str | None = None,
    ):
        """
        Initialize processing orchestrator.

        Args:
            depth_processor: DepthMapProcessor instance
            stereo_generator: StereoPairGenerator instance
            distortion_processor: DistortionProcessor instance
            upscaler: FrameUpscalerProcessor instance
            vr_assembler: VRFrameAssembler instance
            video_encoder: VideoEncoder instance
            verbose: Enable verbose output
            release_depth_model: Callback that unloads the depth model
        """
        self.depth_processor = depth_processor
        self.stereo_generator = stereo_generator
        self.distortion_processor = distortion_processor
        self.upscaler = upscaler
        self.vr_assembler = vr_assembler
        self.video_encoder = video_encoder
        self.verbose = verbose
        self._release_depth_model_callback = release_depth_model
        self.temporal_stabilizer = temporal_stabilizer
        self.preselected_depth_files = (
            list(preselected_depth_files) if preselected_depth_files is not None else None
        )
        self.preselected_render_source = preselected_render_source
        if (self.preselected_depth_files is None) != (preselected_render_source is None):
            raise ValueError("preselected files and render source must be supplied together")
        if preselected_render_source is not None and preselected_render_source not in {
            "base",
            "stabilized",
        }:
            raise ValueError("preselected render source must be base or stabilized")
        self._depth_model_released = False
        self._settings_file: Path | None = None  # Track settings file for error handling
        self._start_time: float = 0.0  # Track processing start time

    def process(
        self,
        video_path: Path,
        output_dir: Path,
        video_properties: dict[str, Any],
        settings: dict[str, Any],
        progress_tracker=None,
        job_lock: JobWriterLock | None = None,
    ) -> bool:
        """
        Main processing pipeline entry point.

        Args:
            video_path: Input video path
            output_dir: Output directory
            video_properties: Video metadata (frame_count, fps, etc.)
            settings: Processing settings
            progress_tracker: Optional progress tracker

        Returns:
            True if successful, False otherwise

        Side effects:
            - Console output
            - Filesystem state management
            - Delegates to all processor modules
        """
        self._depth_model_released = False
        owns_job_lock = job_lock is None
        try:
            # Start timer
            self._start_time = time.time()

            if job_lock is None:
                output_dir.mkdir(parents=True, exist_ok=True)
                job_lock = JobWriterLock(output_dir).acquire()
            elif not job_lock.is_acquired:
                raise ValueError("The supplied job writer lock is not acquired")
            elif job_lock.output_dir.resolve() != output_dir.resolve():
                raise ValueError("The supplied job writer lock belongs to another output directory")

            # Setup processing environment
            output_path, directories, self._settings_file = self._setup_processing(
                str(video_path), str(output_dir), settings, video_properties
            )

            # Execute processing pipeline
            success = self._execute_pipeline(
                str(video_path),
                output_path,
                directories,
                video_properties,
                settings,
                progress_tracker,
            )

            return success

        except InterruptedError:
            raise
        except Exception as e:
            print(f"Error in video processing: {e}")
            if self._settings_file:
                update_processing_status(self._settings_file, "failed", {"error": str(e)})
            return False
        finally:
            try:
                self._release_depth_model()
            finally:
                if owns_job_lock and job_lock is not None:
                    job_lock.release()

    def _execute_pipeline(  # noqa: C901
        self,
        video_path: str,
        output_path: Path,
        directories: dict[str, Path],
        video_properties: dict[str, Any],
        settings: dict[str, Any],
        progress_tracker=None,
    ) -> bool:
        """
        Execute complete 8-step pipeline.

        Args:
            video_path: Input video path
            output_path: Output directory path
            directories: Dictionary of processing directories
            video_properties: Video metadata (frame_count, fps, etc.)
            settings: Processing settings
            progress_tracker: Optional progress tracker

        Returns:
            True if all steps successful, False otherwise

        Side effects:
            - Executes all pipeline steps
            - Progress updates
            - Console output
        """
        # Step 1: Extract frames (delegated to video_encoder)
        frame_files = self.video_encoder.extract_frames(
            video_path, directories, video_properties, settings
        )
        if not frame_files:
            return False
        print(step_complete(f"Step 1: Extracted {len(frame_files)} frames"))
        self._print_saved_to(directories.get("frames"), "Extracted frames")
        print()  # Blank line after step

        fps = video_properties.get("fps", 30.0)

        selected_is_stabilized = self.preselected_render_source == "stabilized"
        if self.preselected_depth_files is not None:
            artifact = validate_render_disparity_input(
                self.preselected_depth_files,
                frame_files,
            )
            if artifact.producer != self.preselected_render_source:
                raise ValueError("Preselected render-disparity producer changed")
            depth_files = list(self.preselected_depth_files)
            self._release_depth_model()
            print(step_complete(f"Step 2: Reused {len(depth_files)} render disparity maps"))
        else:
            if self.depth_processor is None:
                raise RuntimeError("Depth generation requires a configured estimator")
            # Step 2: Generate canonical disparity maps (delegated to depth_processor)
            try:
                depth_files = self.depth_processor.generate_depth_map_files(
                    frame_files, settings, directories, progress_tracker
                )
            finally:
                self._release_depth_model()
            if depth_files is None:
                return False
            print(step_complete(f"Step 2: Prepared {len(depth_files)} canonical disparity maps"))
            self._print_saved_to(directories.get("disparity_maps"), "Canonical disparity maps")
        print()  # Blank line after step

        effective_depth_files = depth_files
        if settings.get("temporal_postprocessor", "off") == "vdpp" and not selected_is_stabilized:
            if self.temporal_stabilizer is None:
                raise RuntimeError("VDPP was requested but no temporal stabilizer is configured")
            effective_depth_files = self.temporal_stabilizer.generate_files(
                depth_files,
                settings,
                directories,
                progress_tracker,
            )
            if not effective_depth_files:
                return False
            print(
                step_complete(f"Temporal stabilization: Prepared {len(effective_depth_files)} maps")
            )
            self._print_saved_to(
                directories.get("disparity_stabilized"),
                "Stabilized disparity maps",
            )
            print()
        elif selected_is_stabilized and settings.get("temporal_postprocessor") != "vdpp":
            raise ValueError("Stabilized render input cannot satisfy temporal_postprocessor=off")

        # Execute steps 3-8
        return self._execute_remaining_steps(
            directories,
            settings,
            frame_files,
            effective_depth_files,
            fps,
            video_path,
            output_path,
            progress_tracker,
        )

    def _release_depth_model(self) -> None:
        """Release the depth model once per pipeline run."""
        if self._depth_model_released or self._release_depth_model_callback is None:
            return
        self._release_depth_model_callback()
        self._depth_model_released = True

    def _execute_remaining_steps(  # noqa: C901
        self,
        directories: dict[str, Path],
        settings: dict[str, Any],
        frame_files: list[Path],
        depth_files: list[Path],
        fps: float,
        video_path: str,
        output_path: Path,
        progress_tracker=None,
    ) -> bool:
        """
        Execute remaining pipeline steps after depth map generation.

        Args:
            directories: Dictionary of processing directories
            settings: Processing settings
            frame_files: List of extracted frame files
            depth_files: List of disk-backed depth map files
            fps: Video frames per second
            video_path: Input video path
            output_path: Output directory path
            progress_tracker: Optional progress tracker

        Returns:
            True if successful, False otherwise

        Side effects:
            - Executes steps 3-8 of pipeline
            - Progress updates
        """
        num_frames = len(frame_files)

        # Step 3: Create stereo pairs (delegated to stereo_generator)
        if not self.stereo_generator.create_stereo_pairs_from_files(
            frame_files,
            depth_files,
            directories,
            settings,
            progress_tracker,
        ):
            return self._handle_step_error("Stereo pair creation failed")
        print(step_complete(f"Step 3: Created {num_frames} stereo pairs"))
        self._print_saved_to(directories.get("left_frames"), "Left frames")
        self._print_saved_to(directories.get("right_frames"), "Right frames")
        print()  # Blank line after left/right pair

        # Step 4: Apply fisheye distortion (optional - delegated to distortion_processor)
        if settings.get("apply_distortion", True):
            left_dir = directories.get("left_frames")
            right_dir = directories.get("right_frames")
            if left_dir is None or right_dir is None:
                return self._handle_step_error("Distortion source directories are missing")
            left_files = sorted(left_dir.glob("*.png"))
            right_files = sorted(right_dir.glob("*.png"))
            if (
                len(left_files) != num_frames
                or len(right_files) != num_frames
                or [path.stem for path in left_files] != [path.stem for path in right_files]
            ):
                return self._handle_step_error("Distortion source frame manifest is incomplete")
            if not self.distortion_processor.apply_distortion(
                left_files, right_files, directories, settings, progress_tracker
            ):
                return self._handle_step_error("Distortion failed")
            print(
                step_complete(
                    f"Step 4: Applied {settings['fisheye_projection']} fisheye distortion"
                )
            )
            self._print_saved_to(directories.get("left_distorted"), "Distorted left frames")
            self._print_saved_to(directories.get("right_distorted"), "Distorted right frames")
            print()  # Blank line after left/right pair

        # Step 5: Crop frames (delegated to distortion_processor)
        if not self.distortion_processor.crop_frames(
            directories, settings, progress_tracker, num_frames
        ):
            return self._handle_step_error("Frame cropping failed")
        print(
            step_complete(
                f"Step 5: Cropped {num_frames} frames to {settings['per_eye_width']}x{settings['per_eye_height']}"
            )
        )
        self._print_saved_to(directories.get("left_cropped"), "Cropped left frames")
        self._print_saved_to(directories.get("right_cropped"), "Cropped right frames")
        print()  # Blank line after left/right pair

        # Step 6: Apply AI upscaling (optional - delegated to upscaler)
        if settings.get("upscale_model", "none") != "none":
            if not self.upscaler.apply_upscaling(directories, settings, progress_tracker):
                return self._handle_step_error("Upscaling failed")
            print(
                step_complete(
                    f"Step 6: Upscaled {num_frames} frames using {settings['upscale_model']}"
                )
            )
            self._print_saved_to(directories.get("left_upscaled"), "Upscaled left frames")
            self._print_saved_to(directories.get("right_upscaled"), "Upscaled right frames")
            print()  # Blank line after left/right pair

        if settings.get("direct_vr_encode", False):
            source_files = self.vr_assembler.resolve_vr_source_files(
                directories, settings, num_frames
            )
            if source_files is None:
                return self._handle_step_error("Direct VR source validation failed")
            left_files, right_files = source_files
            print(step_complete("Step 7: Deferred VR assembly to direct FFmpeg encoding"))
            success = self.video_encoder.create_video_from_stereo_sequences(
                left_files,
                right_files,
                directories["base"],
                video_path,
                settings,
                total_frames=num_frames,
                progress_tracker=progress_tracker,
            )
        else:
            # Step 7: Assemble VR frames (delegated to vr_assembler)
            if not self.vr_assembler.assemble_vr_frames(
                directories, settings, progress_tracker, num_frames
            ):
                return self._handle_step_error("VR frame assembly failed")
            print(
                step_complete(
                    f"Step 7: Assembled {num_frames} {settings['vr_format']} VR frames at {settings['vr_output_width']}x{settings['vr_output_height']}"
                )
            )
            self._print_saved_to(directories.get("vr_frames"), "VR frames")
            print()  # Blank line after step

            # Step 8: Create final video (delegated to video_encoder)
            vr_frames_dir = directories.get("vr_frames")
            if not vr_frames_dir:
                return self._handle_step_error("VR frames directory not found")

            success = self.video_encoder.create_video(
                vr_frames_dir,
                directories["base"],
                video_path,
                settings,
            )

        if success:
            output_filename = generate_output_filename(
                Path(video_path).name,
                settings["vr_format"],
                settings["vr_resolution"],
            )
            print(step_complete("Step 8: Created final video"))
            self._print_saved_to(directories["base"], f"Final output: {output_filename}")

        # Finalize and cleanup
        if progress_tracker and hasattr(progress_tracker, "finish"):
            progress_tracker.finish("Video processing complete")
        self._finalize_processing(success, output_path, video_path, settings, num_frames)
        return success

    def _setup_processing(
        self,
        video_path: str,
        output_dir: str,
        settings: dict[str, Any],
        video_properties: dict[str, Any],
    ) -> tuple[Path, dict[str, Path], Path | None]:
        """
        Setup processing directories and settings file.

        Args:
            video_path: Input video path
            output_dir: Output directory
            settings: Processing settings
            video_properties: Video metadata

        Returns:
            Tuple of (output_path, directories, settings_file)

        Side effects:
            - Creates directories
            - Writes settings file
            - Console output
        """
        output_path = Path(output_dir)
        omitted_intermediates: set[str] = set()
        if settings.get("temporal_postprocessor", "off") != "vdpp":
            omitted_intermediates.add("disparity_stabilized")
        if settings.get("direct_vr_encode", False):
            omitted_intermediates.add("vr_frames")
        if omitted_intermediates:
            directories = create_output_directories(
                output_path,
                settings["keep_intermediates"],
                omitted_intermediates=omitted_intermediates,
            )
        else:
            directories = create_output_directories(output_path, settings["keep_intermediates"])
        batch_name = f"{Path(video_path).stem}_{int(time.time())}"
        settings_file = save_processing_settings(
            output_path, batch_name, settings, video_properties, video_path
        )

        print(f"\n{title_bar('=== Depth Surge 3D Video Processing ===')}")
        print(f"Input: {video_path}")
        print(f"Output: {output_path}")
        if settings.get("depth_model_version") == "v2":
            print(
                "Using Video-Depth-Anything V2 for temporal consistency " "within detected shots\n"
            )
        else:
            print(f"Using depth backend: {settings.get('depth_model_version', 'v3')}\n")

        return output_path, directories, settings_file

    def _finalize_processing(
        self,
        success: bool,
        output_path: Path,
        video_path: str,
        settings: dict[str, Any],
        num_frames: int,
    ) -> None:
        """
        Finalize processing and update settings file.

        Args:
            success: Whether processing succeeded
            output_path: Output directory path
            video_path: Input video path
            settings: Processing settings
            num_frames: Number of frames processed

        Side effects:
            - Updates settings file with completion status
            - Console output
        """
        if success:
            # Calculate processing time
            elapsed_time = time.time() - self._start_time
            formatted_time = self._format_processing_time(elapsed_time)

            # Generate output filename
            output_filename = generate_output_filename(
                Path(video_path).name,
                settings["vr_format"],
                settings["vr_resolution"],
            )
            output_file_path = str(output_path / output_filename)

            # Display colored completion banner
            completion_banner(
                output_file=output_file_path,
                processing_time=formatted_time,
                num_frames=num_frames,
                vr_format=settings["vr_format"],
            )

            # Update settings file
            if self._settings_file:
                update_processing_status(
                    self._settings_file,
                    "completed",
                    {
                        "final_output": output_file_path,
                        "frames_processed": num_frames,
                        "processing_time_seconds": elapsed_time,
                    },
                )

            if not settings.get("keep_intermediates", True):
                removed_count = cleanup_intermediate_files(output_path)
                print(f"Removed {removed_count} temporary processing files")
        elif self._settings_file:
            update_processing_status(
                self._settings_file, "failed", {"error": "Video creation failed"}
            )

    @staticmethod
    def _format_processing_time(seconds: float) -> str:
        """
        PURE: Format processing time as human-readable string.

        Args:
            seconds: Processing time in seconds

        Returns:
            Formatted time string (e.g., "1h 23m 45s", "5m 30s", "45s")
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:  # Always show seconds if no other parts
            parts.append(f"{secs}s")

        return " ".join(parts)

    def _update_step_progress(
        self,
        progress_tracker,
        message: str,
        step_name: str,
        progress: int,
        total: int,
    ) -> None:
        """
        Update progress for a processing step.

        Args:
            progress_tracker: Progress tracker instance
            message: Progress message
            step_name: Name of current step
            progress: Current progress value
            total: Total progress value

        Side effects:
            - Updates progress tracker state
        """
        if progress_tracker:
            progress_tracker.update_progress(
                message,
                phase="processing",
                frame_num=progress,
                step_name=step_name,
                step_progress=progress,
                step_total=total,
            )

    def _print_saved_to(self, directory: Path | None, message_prefix: str = "Saved to") -> None:
        """
        Print save location message.

        Args:
            directory: Directory path or None
            message_prefix: Message prefix text

        Side effects:
            - Console output
        """
        if directory:
            print(saved_to(f"{message_prefix}: {directory}"))

    def _handle_step_error(self, error_msg: str) -> bool:
        """
        Handle step failure and update settings file.

        Args:
            error_msg: Error message

        Returns:
            False (always returns False to indicate failure)

        Side effects:
            - Console output
            - Updates settings file with error status
        """
        print(f"Error: {error_msg}")
        if self._settings_file:
            update_processing_status(self._settings_file, "failed", {"error": error_msg})
        return False
