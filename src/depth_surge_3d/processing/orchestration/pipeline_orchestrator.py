"""
Processing pipeline orchestrator.

Coordinates the complete video processing pipeline across all specialized processors.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, cast

from ...core.render_disparity import validate_render_disparity_input
from ...io.job_lock import JobWriterLock
from ...io.operations import (
    cleanup_intermediate_files,
    commit_cleanup_pending_status,
    create_output_directories,
    load_processing_settings,
    prepare_processing_settings_artifact,
    recover_processing_control_prefix_v1,
    recover_processing_settings_artifact,
    save_processing_settings,
    update_processing_status,
)
from ...rendering.quality.cleanup_transaction import (
    cleanup_status_for_error,
    commit_payload_pruned,
    run_authorized_cleanup,
)
from ...utils import (
    completion_banner,
    create_vr_frame,
    saved_to,
    step_complete,
    title_bar,
)
from ...utils.path_utils import generate_output_filename
from ...vdpp_work import cleanup_vdpp_private_work
from ..frames.stereo_control import StereoPipelineCapacityError
from ..frames.stereo_input_manifest import (
    StereoInputManifestV1,
    publish_validated_stereo_input_manifest_v1,
)
from ..video.final_container_validation import verify_existing_final_video


class ProcessingOrchestrator:
    """High-level pipeline control and step sequencing."""

    def __init__(
        self,
        depth_processor: Any,
        stereo_generator: Any,
        distortion_processor: Any,
        upscaler: Any,
        vr_assembler: Any,
        video_encoder: Any,
        verbose: bool = False,
        release_depth_model: Callable[[], None] | None = None,
        temporal_stabilizer: Any = None,
        preselected_depth_files: list[Path] | None = None,
        preselected_render_source: str | None = None,
        **_ignored: object,
    ) -> None:
        self.depth_processor = depth_processor
        self.stereo_generator = stereo_generator
        self.distortion_processor = distortion_processor
        self.upscaler = upscaler
        self.vr_assembler = vr_assembler
        self.video_encoder = video_encoder
        self.verbose = verbose
        self._release_depth_model_callback = release_depth_model
        self.temporal_stabilizer = temporal_stabilizer
        self.preselected_depth_files = preselected_depth_files
        self.preselected_render_source = preselected_render_source
        if (self.preselected_depth_files is None) != (preselected_render_source is None):
            raise ValueError("preselected files and render source must be supplied together")
        if preselected_render_source is not None and preselected_render_source not in {
            "base",
            "stabilized",
        }:
            raise ValueError("preselected render source must be base or stabilized")
        self._depth_model_released = False
        self._settings_file: Path | None = None
        self._start_time: float = 0.0
        self.process_elapsed_seconds: float | None = None
        self._metric_clamp_summary: dict[str, Any] | None = None

    @staticmethod
    def finish_publication_only(  # noqa: C901
        output_dir: Path, video_encoder: Any = None
    ) -> bool | None:
        """Finish cleanup if a previous run already published a valid final video."""

        _ = video_encoder
        root = Path(output_dir)
        settings_file = root / "processing-settings-v5.json"
        if not settings_file.is_file():
            return None
        try:
            artifact = recover_processing_settings_artifact(settings_file)
        except Exception as error:
            print(f"Error: Could not recover processing settings: {error}")
            return False
        if artifact.status != "in_progress":
            return None
        value = artifact.to_object()
        if value["metadata"].get("cleanup_status") != "pending":
            return None
        rel_final = value["output_info"].get("expected_final_relative_path")
        if not isinstance(rel_final, str) or not verify_existing_final_video(root / rel_final):
            return None

        keep_intermediates = bool(value["processing_settings"]["keep_intermediates"])
        cleanup_status = "not_requested"
        if not keep_intermediates:
            try:
                commit_payload_pruned(root)
                removed_count = run_authorized_cleanup(root)
                cleanup_status = "complete"
                print(f"Removed {removed_count} temporary processing files")
            except Exception as error:
                cleanup_status = cleanup_status_for_error(error)
                print(f"Warning: Intermediate cleanup remains pending ({cleanup_status}): {error}")
                return False

        info: dict[str, Any] = {
            "frames_processed": artifact.selected_frame_count,
            "cleanup_status": cleanup_status,
        }
        metric_summary = value.get("runtime_info", {}).get("metric_clamp_summary")
        if isinstance(metric_summary, dict):
            info["metric_clamp_summary"] = dict(metric_summary)
        return update_processing_status(settings_file, "completed", info)

    def process(
        self,
        video_path: Path,
        output_dir: Path,
        video_properties: dict[str, Any],
        settings: dict[str, Any],
        progress_tracker: Any = None,
        job_lock: JobWriterLock | None = None,
    ) -> bool:
        """Execute the full processing pipeline under the job writer lock."""

        self._depth_model_released = False
        self.process_elapsed_seconds = None
        owns_job_lock = job_lock is None
        try:
            self._start_time = time.perf_counter()
            job_lock = self._acquire_job_lock(output_dir, job_lock)
            recover_processing_control_prefix_v1(output_dir)
            cleanup_vdpp_private_work(output_dir, job_lock)

            cleanup_resume = self.finish_publication_only(output_dir, self.video_encoder)
            if cleanup_resume is not None:
                self._settings_file = Path(output_dir) / "processing-settings-v5.json"
                return cleanup_resume

            video_properties = dict(video_properties)
            output_path, directories, self._settings_file = self._setup_processing(
                str(video_path), str(output_dir), settings, video_properties
            )
            return self._execute_pipeline(
                str(video_path),
                output_path,
                directories,
                video_properties,
                settings,
                progress_tracker,
            )
        except InterruptedError:
            raise
        except Exception as error:
            print(f"Error in video processing: {error}")
            if self._settings_file:
                update_processing_status(self._settings_file, "failed", {"error": str(error)})
            return False
        finally:
            try:
                self._release_depth_model()
            finally:
                try:
                    if owns_job_lock and job_lock is not None:
                        job_lock.release()
                finally:
                    self.process_elapsed_seconds = time.perf_counter() - self._start_time

    @staticmethod
    def _acquire_job_lock(output_dir: Path, job_lock: JobWriterLock | None) -> JobWriterLock:
        if job_lock is None:
            output_dir.mkdir(parents=True, exist_ok=True)
            job_lock = JobWriterLock(output_dir).acquire()
        elif not job_lock.is_acquired:
            raise ValueError("The supplied job writer lock is not acquired")
        elif job_lock.output_dir.resolve() != output_dir.resolve():
            raise ValueError("The supplied job writer lock belongs to another output directory")
        return job_lock

    def _execute_pipeline(  # noqa: C901
        self,
        video_path: str,
        output_path: Path,
        directories: dict[str, Path],
        video_properties: dict[str, Any],
        settings: dict[str, Any],
        progress_tracker: Any = None,
    ) -> bool:
        frame_source = None
        can_stream_source = (
            settings.get("keep_intermediates", True) is False
            and self.preselected_depth_files is None
            and settings.get("temporal_postprocessor", "off") == "off"
            and callable(getattr(self.video_encoder, "prepare_streaming_frame_source", None))
            and bool(getattr(self.depth_processor, "supports_streaming_frame_source", False))
        )
        if can_stream_source:
            frame_source = self.video_encoder.prepare_streaming_frame_source(
                video_path, directories, video_properties, settings
            )

        if frame_source is not None:
            frame_files = frame_source.virtual_frame_files()
            print(step_complete(f"Step 1: Indexed {len(frame_files)} streaming source frames"))
        else:
            frame_files = self.video_encoder.extract_frames(
                video_path, directories, video_properties, settings
            )
            if not frame_files:
                return False
            print(step_complete(f"Step 1: Extracted {len(frame_files)} frames"))
            self._print_saved_to(directories.get("frames"), "Extracted frames")
        print()

        metric_mode = settings.get("stereo_geometry_mode", "relative") == "metric_camera"
        selected_is_stabilized = self.preselected_render_source == "stabilized"
        if metric_mode and self.preselected_depth_files is not None:
            raise ValueError("Preselected disparity cannot satisfy metric_camera geometry")
        if metric_mode and settings.get("temporal_postprocessor", "off") != "off":
            raise ValueError("VDPP is available only with relative stereo geometry")

        if self.preselected_depth_files is not None:
            artifact = validate_render_disparity_input(self.preselected_depth_files, frame_files)
            if artifact.producer != self.preselected_render_source:
                raise ValueError("Preselected render-disparity producer changed")
            geometry_files = list(self.preselected_depth_files)
            self._release_depth_model()
            print(step_complete(f"Step 2: Reused {len(geometry_files)} render disparity maps"))
        else:
            if self.depth_processor is None:
                raise RuntimeError("Depth generation requires a configured estimator")
            try:
                if frame_source is not None:
                    geometry_files = self.depth_processor.generate_depth_map_files(
                        frame_files,
                        settings,
                        directories,
                        progress_tracker,
                        frame_source=frame_source,
                    )
                else:
                    geometry_files = self.depth_processor.generate_depth_map_files(
                        frame_files, settings, directories, progress_tracker
                    )
            finally:
                self._release_depth_model()
            if geometry_files is None:
                return False
            geometry_label = "metric geometry frames" if metric_mode else "canonical disparity maps"
            saved_label = "Metric geometry frames" if metric_mode else "Canonical disparity maps"
            geometry_dir = directories.get("metric_geometry" if metric_mode else "disparity_maps")
            print(step_complete(f"Step 2: Prepared {len(geometry_files)} {geometry_label}"))
            self._print_saved_to(geometry_dir, saved_label)
        print()

        effective_geometry_files = geometry_files
        if settings.get("temporal_postprocessor", "off") == "vdpp" and not selected_is_stabilized:
            if self.temporal_stabilizer is None:
                raise RuntimeError("VDPP was requested but no temporal stabilizer is configured")
            effective_geometry_files = self.temporal_stabilizer.generate_files(
                geometry_files,
                settings,
                directories,
                progress_tracker,
            )
            if not effective_geometry_files:
                return False
            print(
                step_complete(
                    f"Temporal stabilization: Prepared {len(effective_geometry_files)} maps"
                )
            )
            self._print_saved_to(
                directories.get("disparity_stabilized"),
                "Stabilized disparity maps",
            )
            print()
        elif selected_is_stabilized and settings.get("temporal_postprocessor") != "vdpp":
            raise ValueError("Stabilized render input cannot satisfy temporal_postprocessor=off")

        stereo_input = publish_validated_stereo_input_manifest_v1(
            job_root=directories["base"],
            guide_files=frame_files,
            geometry_files=effective_geometry_files,
            geometry_mode=("metric_camera" if metric_mode else "relative"),
            frame_source=frame_source,
        )
        self.preselected_depth_files = None

        return self._execute_remaining_steps(
            directories,
            settings,
            stereo_input,
            video_path,
            output_path,
            progress_tracker,
        )

    def _release_depth_model(self) -> None:
        if self._depth_model_released or self._release_depth_model_callback is None:
            return
        self._release_depth_model_callback()
        self._depth_model_released = True

    @staticmethod
    def _streamed_stereo_settings_eligible(
        settings: dict[str, Any],
        stereo_input: StereoInputManifestV1,
    ) -> bool:
        if (
            settings.get("stereo_render_mode", "fast") not in ("fast", "quality")
            or settings.get("stereo_geometry_mode", "relative") != "relative"
            or getattr(stereo_input, "geometry_mode", None) != "relative"
            or settings.get("direct_vr_encode") is not True
            or settings.get("keep_intermediates", True) is not False
            or settings.get("upscale_model", "none") != "none"
            or settings.get("vr_format") not in {"side_by_side", "over_under"}
        ):
            return False

        frame_count = getattr(stereo_input, "frame_count", None)
        render_shape = getattr(stereo_input, "render_shape", None)
        dimensions = (
            settings.get("per_eye_width"),
            settings.get("per_eye_height"),
            settings.get("vr_output_width"),
            settings.get("vr_output_height"),
        )
        if (
            isinstance(frame_count, bool)
            or not isinstance(frame_count, int)
            or frame_count <= 0
            or not isinstance(render_shape, tuple)
            or len(render_shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in (*render_shape, *dimensions)
            )
        ):
            return False

        eye_width, eye_height, output_width, output_height = cast(
            tuple[int, int, int, int], dimensions
        )
        expected_output = (
            (eye_width * 2, eye_height)
            if settings["vr_format"] == "side_by_side"
            else (eye_width, eye_height * 2)
        )
        if (output_width, output_height) != expected_output:
            return False

        apply_distortion = settings.get("apply_distortion")
        if apply_distortion is True:
            return True
        if apply_distortion is not False:
            return False
        try:
            crop_factor = max(0.5, min(1.0, float(settings.get("crop_factor", 1.0))))
        except (TypeError, ValueError):
            return False
        return crop_factor == 1.0 and render_shape == (eye_height, eye_width)

    def _streamed_stereo_path_enabled(
        self,
        settings: dict[str, Any],
        stereo_input: StereoInputManifestV1,
    ) -> bool:
        return self._streamed_stereo_settings_eligible(settings, stereo_input) and callable(
            getattr(self.video_encoder, "open_streamed_vr_session", None)
        )

    def _execute_streamed_stereo_steps(  # noqa: C901
        self,
        directories: dict[str, Path],
        settings: dict[str, Any],
        stereo_input: StereoInputManifestV1,
        video_path: str,
        output_path: Path,
        progress_tracker: Any = None,
    ) -> bool | None:
        try:
            transform_pair = self.distortion_processor.prepare_stream_transform(
                source_shape=stereo_input.render_shape,
                settings=settings,
            )
            if not callable(transform_pair):
                raise TypeError("stream transform is not callable")
            session = self.video_encoder.open_streamed_vr_session(
                output_dir=directories["base"],
                original_video=video_path,
                settings=settings,
                frame_count=stereo_input.frame_count,
                first_index=stereo_input.first_index,
                progress_tracker=progress_tracker,
            )
        except InterruptedError:
            raise
        except Exception as error:
            if self.verbose:
                print(f"Stereo direct stream unavailable; using file-backed stages: {error}")
            return None

        session_needs_abort = True
        try:

            def consume_frame(frame_name: str, left_frame: Any, right_frame: Any) -> None:
                left_final, right_final = transform_pair(left_frame, right_frame)
                session.write(
                    frame_name,
                    create_vr_frame(left_final, right_final, settings["vr_format"]),
                )

            if not self.stereo_generator.create_stereo_pairs_from_manifest(
                stereo_input,
                directories,
                settings,
                progress_tracker,
                frame_consumer=consume_frame,
            ):
                session_needs_abort = False
                session.abort()
                return self._handle_step_error("Stereo pair creation failed")

            num_frames = stereo_input.frame_count
            print(step_complete(f"Step 3: Created and streamed {num_frames} stereo pairs"))
            if settings.get("apply_distortion") is True:
                print(
                    step_complete(
                        f"Step 4: Applied {settings['fisheye_projection']} fisheye distortion"
                    )
                )
            print(
                step_complete(
                    f"Step 5: Cropped {num_frames} frames to "
                    f"{settings['per_eye_width']}x{settings['per_eye_height']}"
                )
            )
            print(
                step_complete(
                    f"Step 7: Assembled {num_frames} {settings['vr_format']} VR frames in memory"
                )
            )

            success = bool(session.finish_and_publish())
            session_needs_abort = False
            if success:
                output_filename = self._expected_output_filename(video_path, settings)
                print(step_complete("Step 8: Created final video"))
                self._print_saved_to(directories["base"], f"Final output: {output_filename}")

            finalized = self._finalize_processing(
                success, output_path, video_path, settings, num_frames
            )
            completed = success and finalized is not False
            if completed and progress_tracker and hasattr(progress_tracker, "finish"):
                progress_tracker.finish("Video processing complete")
            return completed
        except StereoPipelineCapacityError as error:
            session.discard_for_file_backed_fallback()
            session_needs_abort = False
            if self.verbose:
                print(
                    "Stereo direct stream exceeds the shared memory budget; "
                    f"using file-backed stages: {error}"
                )
            return None
        finally:
            if session_needs_abort:
                session.abort()

    def _execute_remaining_steps(
        self,
        directories: dict[str, Path],
        settings: dict[str, Any],
        stereo_input: StereoInputManifestV1,
        video_path: str,
        output_path: Path,
        progress_tracker: Any = None,
    ) -> bool:
        num_frames = stereo_input.frame_count
        self._metric_clamp_summary = None

        if self._streamed_stereo_path_enabled(settings, stereo_input):
            streamed = self._execute_streamed_stereo_steps(
                directories,
                settings,
                stereo_input,
                video_path,
                output_path,
                progress_tracker,
            )
            if streamed is not None:
                return streamed

        if not self.stereo_generator.create_stereo_pairs_from_manifest(
            stereo_input,
            directories,
            settings,
            progress_tracker,
        ):
            return self._handle_step_error("Stereo pair creation failed")
        print(step_complete(f"Step 3: Created {num_frames} stereo pairs"))
        self._print_saved_to(directories.get("left_frames"), "Left frames")
        self._print_saved_to(directories.get("right_frames"), "Right frames")
        print()
        if settings.get("stereo_geometry_mode", "relative") == "metric_camera":
            summary = self.stereo_generator.last_metric_clamp_summary
            if isinstance(summary, dict):
                self._metric_clamp_summary = dict(summary)
                print(
                    "Metric disparity clamp summary: "
                    f"affected_frames={int(summary['affected_frame_count'])}, "
                    f"mean={float(summary['mean_clamped_fraction']):.4%}, "
                    f"max={float(summary['max_clamped_fraction']):.4%}"
                )

        return self._execute_post_stereo_steps(
            directories,
            settings,
            num_frames,
            video_path,
            output_path,
            progress_tracker,
        )

    def _execute_post_stereo_steps(  # noqa: C901
        self,
        directories: dict[str, Path],
        settings: dict[str, Any],
        num_frames: int,
        video_path: str,
        output_path: Path,
        progress_tracker: Any = None,
    ) -> bool:
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
            if not self.distortion_processor.apply_distortion_and_crop(
                left_files, right_files, directories, settings, progress_tracker
            ):
                return self._handle_step_error("Distortion and cropping failed")
            print(
                step_complete(
                    f"Step 4: Applied {settings['fisheye_projection']} fisheye distortion"
                )
            )
            print()
        else:
            if not self.distortion_processor.crop_frames(
                directories, settings, progress_tracker, num_frames
            ):
                return self._handle_step_error("Frame cropping failed")
        print(
            step_complete(
                f"Step 5: Cropped {num_frames} frames to "
                f"{settings['per_eye_width']}x{settings['per_eye_height']}"
            )
        )
        self._print_saved_to(directories.get("left_cropped"), "Cropped left frames")
        self._print_saved_to(directories.get("right_cropped"), "Cropped right frames")
        print()

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
            print()

        if settings.get("direct_vr_encode", False):
            direct_source = self.vr_assembler.resolve_direct_vr_source(
                directories, settings, num_frames
            )
            if direct_source is None:
                return self._handle_step_error("Direct VR source validation failed")
            print(step_complete("Step 7: Deferred VR assembly to direct FFmpeg encoding"))
            success = self.video_encoder.create_video_from_stereo_source(
                direct_source,
                directories["base"],
                video_path,
                settings,
                total_frames=num_frames,
                progress_tracker=progress_tracker,
            )
        else:
            if not self.vr_assembler.assemble_vr_frames(
                directories, settings, progress_tracker, num_frames
            ):
                return self._handle_step_error("VR frame assembly failed")
            print(
                step_complete(
                    f"Step 7: Assembled {num_frames} {settings['vr_format']} VR frames at "
                    f"{settings['vr_output_width']}x{settings['vr_output_height']}"
                )
            )
            self._print_saved_to(directories.get("vr_frames"), "VR frames")
            print()
            vr_frames_dir = directories.get("vr_frames")
            if not vr_frames_dir:
                return self._handle_step_error("VR frames directory not found")
            success = self.video_encoder.create_video(
                vr_frames_dir,
                directories["base"],
                video_path,
                settings,
                progress_tracker=progress_tracker,
            )

        if success:
            output_filename = self._expected_output_filename(video_path, settings)
            print(step_complete("Step 8: Created final video"))
            self._print_saved_to(directories["base"], f"Final output: {output_filename}")

        finalized = self._finalize_processing(
            success, output_path, video_path, settings, num_frames
        )
        completed = success and finalized is not False
        if completed and progress_tracker and hasattr(progress_tracker, "finish"):
            progress_tracker.finish("Video processing complete")
        return completed

    def _setup_processing(
        self,
        video_path: str,
        output_dir: str,
        settings: dict[str, Any],
        video_properties: dict[str, Any],
    ) -> tuple[Path, dict[str, Path], Path | None]:
        output_path = Path(output_dir)
        omitted_intermediates: set[str] = set()
        if settings.get("temporal_postprocessor", "off") != "vdpp":
            omitted_intermediates.add("disparity_stabilized")
        if settings.get("direct_vr_encode", False):
            omitted_intermediates.add("vr_frames")
        if settings.get("apply_distortion") is True:
            omitted_intermediates.update({"left_distorted", "right_distorted"})
        batch_name = f"{Path(video_path).stem}_{int(time.time())}"
        fixed_settings = output_path / "processing-settings-v5.json"
        if fixed_settings.is_file():
            prepared = prepare_processing_settings_artifact(
                fixed_settings,
                settings=settings,
                video_properties=video_properties,
                source_video_path=video_path,
            )
            if "frame_rate" not in prepared.to_object()["video_properties"]:
                video_properties.pop("frame_rate", None)
            settings_file = fixed_settings
        else:
            settings_file = save_processing_settings(
                output_path,
                batch_name,
                settings,
                video_properties,
                video_path,
                transactional_v5=True,
            )
        directories = create_output_directories(
            output_path,
            settings["keep_intermediates"],
            omitted_intermediates=omitted_intermediates or None,
        )

        print(f"\n{title_bar('=== Depth Surge 3D Video Processing ===')}")
        print(f"Input: {video_path}")
        print(f"Output: {output_path}")
        if settings.get("depth_model_version") == "v2":
            print("Using Video-Depth-Anything V2 for temporal consistency within detected shots\n")
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
    ) -> bool:
        if not success:
            if self._settings_file:
                return update_processing_status(
                    self._settings_file, "failed", {"error": "Video creation failed"}
                )
            return False

        elapsed_time = time.perf_counter() - self._start_time
        output_filename = self._expected_output_filename(video_path, settings)
        output_file_path = str(output_path / output_filename)
        schema5_settings = bool(
            self._settings_file is not None
            and self._settings_file.name == "processing-settings-v5.json"
        )

        cleanup_status = "not_requested"
        if not settings.get("keep_intermediates", True):
            if schema5_settings and self._settings_file is not None:
                try:
                    commit_payload_pruned(output_path)
                    commit_cleanup_pending_status(self._settings_file)
                    removed_count = run_authorized_cleanup(output_path)
                    cleanup_status = "complete"
                    print(f"Removed {removed_count} temporary processing files")
                except Exception as error:
                    cleanup_status = cleanup_status_for_error(error)
                    print(
                        f"Warning: Intermediate cleanup remains pending ({cleanup_status}): {error}"
                    )
                    return False
            else:
                removed_count = cleanup_intermediate_files(output_path)
                cleanup_status = "complete"
                print(f"Removed {removed_count} temporary processing files")

        if self._settings_file:
            runtime_info: dict[str, Any] = {
                "final_output": output_file_path,
                "frames_processed": num_frames,
                "processing_time_seconds": elapsed_time,
                "cleanup_status": cleanup_status,
            }
            if self._metric_clamp_summary is not None:
                runtime_info["metric_clamp_summary"] = dict(self._metric_clamp_summary)
            if not update_processing_status(self._settings_file, "completed", runtime_info):
                print("Error: Could not commit terminal processing settings")
                return False

        completion_banner(
            output_file=output_file_path,
            processing_time=self._format_processing_time(time.perf_counter() - self._start_time),
            num_frames=num_frames,
            vr_format=settings["vr_format"],
        )
        return True

    def _expected_output_filename(
        self,
        video_path: str,
        settings: dict[str, Any],
    ) -> str:
        if (
            self._settings_file is not None
            and self._settings_file.name == "processing-settings-v5.json"
        ):
            saved = load_processing_settings(self._settings_file)
            if saved is not None:
                output_info = saved.get("output_info")
                if isinstance(output_info, dict):
                    expected = output_info.get("expected_final_relative_path")
                    if isinstance(expected, str) and expected:
                        return expected
        return generate_output_filename(
            Path(video_path).name,
            settings["vr_format"],
            settings["vr_resolution"],
        )

    @staticmethod
    def _format_processing_time(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")
        return " ".join(parts)

    def _update_step_progress(
        self,
        progress_tracker: Any,
        message: str,
        step_name: str,
        progress: int,
        total: int,
    ) -> None:
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
        if directory:
            print(saved_to(f"{message_prefix}: {directory}"))

    def _handle_step_error(self, error_msg: str) -> bool:
        print(f"Error: {error_msg}")
        if self._settings_file:
            update_processing_status(self._settings_file, "failed", {"error": error_msg})
        return False
