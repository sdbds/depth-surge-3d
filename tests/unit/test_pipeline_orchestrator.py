"""Unit tests for pipeline orchestrator."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.depth_surge_3d.processing.orchestration.pipeline_orchestrator import (
    ProcessingOrchestrator,
)
from src.depth_surge_3d.io.job_lock import JobWriterLock


class TestProcessingOrchestratorInit:
    """Test ProcessingOrchestrator initialization."""

    def test_init_with_all_processors(self):
        """Test initialization with all required processors."""
        depth_proc = Mock()
        stereo_gen = Mock()
        distortion_proc = Mock()
        upscaler = Mock()
        vr_assembler = Mock()
        video_encoder = Mock()

        orchestrator = ProcessingOrchestrator(
            depth_proc, stereo_gen, distortion_proc, upscaler, vr_assembler, video_encoder
        )

        assert orchestrator.depth_processor == depth_proc
        assert orchestrator.stereo_generator == stereo_gen
        assert orchestrator.distortion_processor == distortion_proc
        assert orchestrator.upscaler == upscaler
        assert orchestrator.vr_assembler == vr_assembler
        assert orchestrator.video_encoder == video_encoder
        assert orchestrator.verbose is False
        assert orchestrator._start_time == 0.0
        assert not hasattr(orchestrator, "_total_steps")
        assert "_get_total_steps" not in vars(ProcessingOrchestrator)
        assert "_print_step_complete" not in vars(ProcessingOrchestrator)

    def test_init_with_verbose(self):
        """Test initialization with verbose flag."""
        orchestrator = ProcessingOrchestrator(
            Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), verbose=True
        )
        assert orchestrator.verbose is True


class TestFormatProcessingTime:
    """Test _format_processing_time pure function."""

    def test_seconds_only(self):
        """Test formatting seconds only."""
        assert ProcessingOrchestrator._format_processing_time(45) == "45s"

    def test_minutes_and_seconds(self):
        """Test formatting minutes and seconds."""
        assert ProcessingOrchestrator._format_processing_time(125) == "2m 5s"

    def test_hours_minutes_seconds(self):
        """Test formatting hours, minutes, and seconds."""
        assert ProcessingOrchestrator._format_processing_time(5025) == "1h 23m 45s"

    def test_exact_hour(self):
        """Test formatting exact hour."""
        assert ProcessingOrchestrator._format_processing_time(3600) == "1h"

    def test_exact_minute(self):
        """Test formatting exact minute."""
        assert ProcessingOrchestrator._format_processing_time(120) == "2m"

    def test_zero_seconds(self):
        """Test formatting zero seconds."""
        assert ProcessingOrchestrator._format_processing_time(0) == "0s"

    def test_large_duration(self):
        """Test formatting large duration."""
        # 10 hours, 30 minutes, 15 seconds
        assert ProcessingOrchestrator._format_processing_time(37815) == "10h 30m 15s"


class TestSetupProcessing:
    """Test _setup_processing method."""

    @patch(
        "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.create_output_directories"
    )
    @patch(
        "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.save_processing_settings"
    )
    @patch("src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.time.time")
    def test_setup_processing(self, mock_time, mock_save_settings, mock_create_dirs):
        """Test setup processing creates directories and settings file."""
        mock_time.return_value = 1234567890
        mock_create_dirs.return_value = {"base": Path("/output/dir")}
        mock_save_settings.return_value = Path("/output/dir/settings.json")

        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())

        video_props = {"fps": 30, "frame_count": 100}
        settings = {"keep_intermediates": True}

        output_path, directories, settings_file = orchestrator._setup_processing(
            "/input/video.mp4", "/output/dir", settings, video_props
        )

        assert output_path == Path("/output/dir")
        assert directories == {"base": Path("/output/dir")}
        assert settings_file == Path("/output/dir/settings.json")
        mock_create_dirs.assert_called_once_with(
            Path("/output/dir"),
            True,
            omitted_intermediates={"disparity_stabilized"},
        )

    def test_setup_processing_omits_vr_directory_in_direct_mode(self):
        """Direct encoding does not create the unused assembled-frame directory."""
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())
        with (
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.create_output_directories",
                return_value={"base": Path("/output/dir")},
            ) as create_dirs,
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.save_processing_settings"
            ),
        ):
            orchestrator._setup_processing(
                "/input/video.mp4",
                "/output/dir",
                {"keep_intermediates": True, "direct_vr_encode": True},
                {"fps": 30, "frame_count": 1},
            )

        create_dirs.assert_called_once_with(
            Path("/output/dir"),
            True,
            omitted_intermediates={"disparity_stabilized", "vr_frames"},
        )

    def test_setup_processing_keeps_legacy_directory_call_when_direct_mode_is_false(self):
        """Explicitly disabling direct mode preserves the legacy directory setup."""
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())
        with (
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.create_output_directories",
                return_value={"base": Path("/output/dir")},
            ) as create_dirs,
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.save_processing_settings"
            ),
        ):
            orchestrator._setup_processing(
                "/input/video.mp4",
                "/output/dir",
                {"keep_intermediates": True, "direct_vr_encode": False},
                {"fps": 30, "frame_count": 1},
            )

        create_dirs.assert_called_once_with(
            Path("/output/dir"),
            True,
            omitted_intermediates={"disparity_stabilized"},
        )

    def test_setup_processing_creates_stabilized_directory_only_when_requested(self):
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())
        with (
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.create_output_directories",
                return_value={"base": Path("/output/dir")},
            ) as create_dirs,
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.save_processing_settings"
            ),
        ):
            orchestrator._setup_processing(
                "/input/video.mp4",
                "/output/dir",
                {
                    "keep_intermediates": True,
                    "direct_vr_encode": False,
                    "temporal_postprocessor": "vdpp",
                },
                {"fps": 30, "frame_count": 1},
            )

        create_dirs.assert_called_once_with(Path("/output/dir"), True)


class TestFinalizeProcessing:
    """Test _finalize_processing method."""

    @patch("src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.completion_banner")
    @patch(
        "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.update_processing_status"
    )
    @patch(
        "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.generate_output_filename"
    )
    @patch("src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.time.time")
    def test_finalize_processing_success(
        self, mock_time, mock_gen_filename, mock_update_status, mock_banner
    ):
        """Test finalize processing on success."""
        mock_time.return_value = 1500.0  # Current time when finalize is called
        mock_gen_filename.return_value = "video_3D_side_by_side.mp4"

        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())
        orchestrator._start_time = 1000.0  # Started 500 seconds ago
        orchestrator._settings_file = Path("/output/settings.json")

        settings = {"vr_format": "side_by_side", "vr_resolution": "16x9-1080p"}

        orchestrator._finalize_processing(
            success=True,
            output_path=Path("/output"),
            video_path="/input/video.mp4",
            settings=settings,
            num_frames=300,
        )

        # Check completion banner was called
        mock_banner.assert_called_once()
        args = mock_banner.call_args[1]
        assert args["output_file"] == str(Path("/output/video_3D_side_by_side.mp4"))
        assert args["processing_time"] == "8m 20s"  # 500 seconds = 8m 20s
        assert args["num_frames"] == 300
        assert args["vr_format"] == "side_by_side"

        # Check status was updated
        mock_update_status.assert_called_once()

    @patch(
        "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.update_processing_status"
    )
    def test_finalize_processing_failure(self, mock_update_status):
        """Test finalize processing on failure."""
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())
        orchestrator._settings_file = Path("/output/settings.json")

        settings = {"vr_format": "side_by_side", "vr_resolution": "16x9-1080p"}

        orchestrator._finalize_processing(
            success=False,
            output_path=Path("/output"),
            video_path="/input/video.mp4",
            settings=settings,
            num_frames=300,
        )

        mock_update_status.assert_called_once_with(
            Path("/output/settings.json"), "failed", {"error": "Video creation failed"}
        )

    @patch(
        "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.cleanup_intermediate_files"
    )
    @patch("src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.completion_banner")
    @patch(
        "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.update_processing_status"
    )
    def test_finalize_success_cleans_temporary_files_when_not_retained(
        self, _update_status, _banner, cleanup
    ):
        """Intermediates are deleted only after final video creation succeeds."""
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())
        orchestrator._start_time = 0.0
        orchestrator._settings_file = Path("/output/settings.json")

        with patch(
            "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.time.time",
            return_value=1.0,
        ):
            orchestrator._finalize_processing(
                True,
                Path("/output"),
                "/input/video.mp4",
                {
                    "vr_format": "side_by_side",
                    "vr_resolution": "16x9-1080p",
                    "keep_intermediates": False,
                },
                3,
            )

        cleanup.assert_called_once_with(Path("/output"))


class TestExecutePipeline:
    def test_releases_depth_model_before_starting_step_3(self, tmp_path):
        """The depth model leaves memory as soon as its pipeline stage is complete."""

        class LoadedDepthModel:
            def __init__(self):
                self.loaded = True

            def unload_model(self):
                self.loaded = False

        class DepthProcessor:
            @staticmethod
            def generate_depth_map_files(*_args):
                return [tmp_path / "depth_000001.png"]

        class StereoGenerator:
            def __init__(self, model):
                self.model = model
                self.model_loaded_at_step_3 = None

            def create_stereo_pairs_from_files(self, *_args):
                self.model_loaded_at_step_3 = self.model.loaded
                return False

        model = LoadedDepthModel()
        stereo_generator = StereoGenerator(model)
        video_encoder = Mock()
        video_encoder.extract_frames.return_value = [tmp_path / "frame_000001.png"]
        orchestrator = ProcessingOrchestrator(
            DepthProcessor(),
            stereo_generator,
            Mock(),
            Mock(),
            Mock(),
            video_encoder,
            release_depth_model=model.unload_model,
        )

        result = orchestrator._execute_pipeline(
            "source.mp4",
            tmp_path,
            {"base": tmp_path, "frames": tmp_path / "frames"},
            {"fps": 30.0, "frame_count": 1},
            {},
        )

        assert result is False
        assert model.loaded is False
        assert stereo_generator.model_loaded_at_step_3 is False

    def test_vdpp_runs_after_owner_release_and_before_stereo(self, tmp_path):
        events = []
        frame = tmp_path / "frame_000001.png"
        base = tmp_path / "base_000001.png"
        stable = tmp_path / "stable_000001.png"

        depth_processor = Mock()
        depth_processor.generate_depth_map_files.return_value = base_files = [base]
        temporal = Mock()

        def stabilize(*args):
            events.append("vdpp")
            assert events == ["release", "vdpp"]
            assert args[0] == base_files
            return [stable]

        temporal.generate_files.side_effect = stabilize
        stereo = Mock()

        def render(_frames, depth_files, *_args):
            events.append("stereo")
            assert depth_files == [stable]
            return False

        stereo.create_stereo_pairs_from_files.side_effect = render
        encoder = Mock()
        encoder.extract_frames.return_value = [frame]
        orchestrator = ProcessingOrchestrator(
            depth_processor,
            stereo,
            Mock(),
            Mock(),
            Mock(),
            encoder,
            release_depth_model=lambda: events.append("release"),
            temporal_stabilizer=temporal,
        )

        assert not orchestrator._execute_pipeline(
            "source.mp4",
            tmp_path,
            {"base": tmp_path, "frames": tmp_path / "frames"},
            {"fps": 30.0, "frame_count": 1},
            {"temporal_postprocessor": "vdpp"},
        )

        assert events == ["release", "vdpp", "stereo"]

    def test_vdpp_failure_never_starts_stereo(self, tmp_path):
        depth_processor = Mock()
        depth_processor.generate_depth_map_files.return_value = [tmp_path / "base.png"]
        temporal = Mock()
        temporal.generate_files.side_effect = RuntimeError("VDPP failed")
        stereo = Mock()
        encoder = Mock()
        encoder.extract_frames.return_value = [tmp_path / "frame.png"]
        orchestrator = ProcessingOrchestrator(
            depth_processor,
            stereo,
            Mock(),
            Mock(),
            Mock(),
            encoder,
            release_depth_model=Mock(),
            temporal_stabilizer=temporal,
        )

        with pytest.raises(RuntimeError, match="VDPP failed"):
            orchestrator._execute_pipeline(
                "source.mp4",
                tmp_path,
                {"base": tmp_path, "frames": tmp_path / "frames"},
                {"fps": 30.0, "frame_count": 1},
                {"temporal_postprocessor": "vdpp"},
            )

        stereo.create_stereo_pairs_from_files.assert_not_called()

    @staticmethod
    def _direct_encoding_dependencies(tmp_path):
        frame = tmp_path / "frame_000001.png"
        depth = tmp_path / "depth_000001.png"
        left = tmp_path / "left_cropped" / "frame_000001.png"
        right = tmp_path / "right_cropped" / "frame_000001.png"
        directories = {
            "base": tmp_path,
            "left_frames": tmp_path / "left",
            "right_frames": tmp_path / "right",
            "left_cropped": tmp_path / "left_cropped",
            "right_cropped": tmp_path / "right_cropped",
        }
        settings = {
            "apply_distortion": False,
            "upscale_model": "none",
            "per_eye_width": 8,
            "per_eye_height": 8,
            "vr_format": "side_by_side",
            "vr_output_width": 16,
            "vr_output_height": 8,
            "vr_resolution": "custom",
            "keep_intermediates": True,
            "direct_vr_encode": True,
        }
        stereo_generator = Mock()
        stereo_generator.create_stereo_pairs_from_files.return_value = True
        distortion_processor = Mock()
        distortion_processor.crop_frames.return_value = True
        vr_assembler = Mock()
        video_encoder = Mock()
        orchestrator = ProcessingOrchestrator(
            Mock(),
            stereo_generator,
            distortion_processor,
            Mock(),
            vr_assembler,
            video_encoder,
        )
        return (
            orchestrator,
            directories,
            settings,
            frame,
            depth,
            left,
            right,
            vr_assembler,
            video_encoder,
        )

    def test_direct_encoding_uses_resolved_stereo_sources(self, tmp_path, capsys):
        """Direct mode sends resolved pairs to FFmpeg without assembling VR frames."""
        (
            orchestrator,
            directories,
            settings,
            frame,
            depth,
            left,
            right,
            vr_assembler,
            video_encoder,
        ) = self._direct_encoding_dependencies(tmp_path)
        progress_tracker = Mock()
        vr_assembler.resolve_vr_source_files.return_value = ([left], [right])
        video_encoder.create_video_from_stereo_sequences.return_value = True

        with patch.object(orchestrator, "_finalize_processing"):
            assert orchestrator._execute_remaining_steps(
                directories,
                settings,
                [frame],
                [depth],
                24.0,
                "source.mp4",
                directories["base"],
                progress_tracker,
            )

        vr_assembler.assemble_vr_frames.assert_not_called()
        vr_assembler.resolve_vr_source_files.assert_called_once_with(directories, settings, 1)
        video_encoder.create_video.assert_not_called()
        video_encoder.create_video_from_stereo_sequences.assert_called_once_with(
            [left],
            [right],
            directories["base"],
            "source.mp4",
            settings,
            total_frames=1,
            progress_tracker=progress_tracker,
        )
        assert "VR frames" not in capsys.readouterr().out

    def test_direct_encoding_stops_when_stereo_sources_are_invalid(self, tmp_path):
        """Direct source validation failure never starts either video encoder path."""
        (
            orchestrator,
            directories,
            settings,
            frame,
            depth,
            _left,
            _right,
            vr_assembler,
            video_encoder,
        ) = self._direct_encoding_dependencies(tmp_path)
        vr_assembler.resolve_vr_source_files.return_value = None

        assert not orchestrator._execute_remaining_steps(
            directories,
            settings,
            [frame],
            [depth],
            24.0,
            "source.mp4",
            directories["base"],
        )

        vr_assembler.assemble_vr_frames.assert_not_called()
        video_encoder.create_video.assert_not_called()
        video_encoder.create_video_from_stereo_sequences.assert_not_called()

    def test_direct_encoding_failure_finalizes_without_cleanup(self, tmp_path):
        """A failed direct FFmpeg encode uses failure finalization and keeps intermediates."""
        (
            orchestrator,
            directories,
            settings,
            frame,
            depth,
            left,
            right,
            vr_assembler,
            video_encoder,
        ) = self._direct_encoding_dependencies(tmp_path)
        settings["keep_intermediates"] = False
        progress_tracker = Mock()
        vr_assembler.resolve_vr_source_files.return_value = ([left], [right])
        video_encoder.create_video_from_stereo_sequences.return_value = False

        with (
            patch.object(
                orchestrator,
                "_finalize_processing",
                wraps=orchestrator._finalize_processing,
            ) as finalize,
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.cleanup_intermediate_files"
            ) as cleanup,
        ):
            assert not orchestrator._execute_remaining_steps(
                directories,
                settings,
                [frame],
                [depth],
                24.0,
                "source.mp4",
                directories["base"],
                progress_tracker,
            )

        finalize.assert_called_once_with(
            False,
            directories["base"],
            "source.mp4",
            settings,
            1,
        )
        progress_tracker.finish.assert_called_once_with("Video processing complete")
        cleanup.assert_not_called()

    def test_required_distortion_stage_fails_when_stereo_outputs_are_missing(self, tmp_path):
        stereo_generator = Mock()
        stereo_generator.create_stereo_pairs_from_files.return_value = True
        distortion_processor = Mock()
        orchestrator = ProcessingOrchestrator(
            Mock(), stereo_generator, distortion_processor, Mock(), Mock(), Mock()
        )
        left_dir = tmp_path / "left"
        right_dir = tmp_path / "right"
        left_dir.mkdir()
        right_dir.mkdir()

        result = orchestrator._execute_remaining_steps(
            {
                "base": tmp_path,
                "left_frames": left_dir,
                "right_frames": right_dir,
            },
            {"apply_distortion": True},
            [tmp_path / "frame_000001.png"],
            [tmp_path / "depth_000001.png"],
            30.0,
            "source.mp4",
            tmp_path,
        )

        assert result is False
        distortion_processor.apply_distortion.assert_not_called()

    def test_pipeline_uses_disk_backed_depth_and_stereo_apis(self, tmp_path):
        """The video path never builds whole-video frame or depth arrays."""
        frame_files = [tmp_path / "frame_000001.png"]
        depth_files = [tmp_path / "depth_000001.png"]
        depth_processor = Mock()
        depth_processor.generate_depth_map_files.return_value = depth_files
        depth_processor.generate_depth_maps.side_effect = AssertionError("array depth API used")
        stereo_generator = Mock()
        stereo_generator.create_stereo_pairs_from_files.return_value = True
        stereo_generator.create_stereo_pairs.side_effect = AssertionError("array stereo API used")
        distortion_processor = Mock()
        distortion_processor.crop_frames.return_value = True
        vr_assembler = Mock()
        vr_assembler.assemble_vr_frames.return_value = True
        video_encoder = Mock()
        video_encoder.extract_frames.return_value = frame_files
        video_encoder.create_video.return_value = True
        directories = {
            "base": tmp_path,
            "frames": tmp_path / "frames",
            "disparity_maps": tmp_path / "disparity",
            "left_frames": tmp_path / "left",
            "right_frames": tmp_path / "right",
            "left_cropped": tmp_path / "left_cropped",
            "right_cropped": tmp_path / "right_cropped",
            "vr_frames": tmp_path / "vr",
        }
        settings = {
            "apply_distortion": False,
            "upscale_model": "none",
            "per_eye_width": 8,
            "per_eye_height": 8,
            "vr_format": "side_by_side",
            "vr_output_width": 16,
            "vr_output_height": 8,
            "vr_resolution": "custom",
            "keep_intermediates": True,
        }
        orchestrator = ProcessingOrchestrator(
            depth_processor,
            stereo_generator,
            distortion_processor,
            Mock(),
            vr_assembler,
            video_encoder,
        )

        with (
            patch.object(orchestrator, "_finalize_processing"),
            patch.object(orchestrator, "_print_saved_to") as saved_to,
        ):
            result = orchestrator._execute_pipeline(
                "source.mp4",
                tmp_path,
                directories,
                {"fps": 30.0, "frame_count": 1},
                settings,
            )

        assert result is True
        depth_processor.generate_depth_map_files.assert_called_once()
        stereo_generator.create_stereo_pairs_from_files.assert_called_once()
        video_encoder.create_video_from_stereo_sequences.assert_not_called()
        saved_to.assert_any_call(directories["disparity_maps"], "Canonical disparity maps")


class TestHandleStepError:
    """Test _handle_step_error method."""

    @patch(
        "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.update_processing_status"
    )
    def test_handle_step_error(self, mock_update_status):
        """Test handle step error updates settings file."""
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())
        orchestrator._settings_file = Path("/output/settings.json")

        result = orchestrator._handle_step_error("Test error message")

        assert result is False
        mock_update_status.assert_called_once_with(
            Path("/output/settings.json"), "failed", {"error": "Test error message"}
        )

    def test_handle_step_error_no_settings_file(self):
        """Test handle step error without settings file."""
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())
        orchestrator._settings_file = None

        result = orchestrator._handle_step_error("Test error message")

        assert result is False


class TestPrintMethods:
    """Test console output helper methods."""

    def test_print_saved_to_with_directory(self, capsys):
        """Test _print_saved_to with valid directory."""
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())

        orchestrator._print_saved_to(Path("/output/dir"), "Test prefix")

        captured = capsys.readouterr()
        assert f"Test prefix: {Path('/output/dir')}" in captured.out

    def test_print_saved_to_with_none(self, capsys):
        """Test _print_saved_to with None directory."""
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())

        orchestrator._print_saved_to(None, "Test prefix")

        captured = capsys.readouterr()
        assert captured.out == ""


class TestProcessMethod:
    """Test main process method."""

    def test_process_releases_depth_model_when_step_1_raises(self, tmp_path):
        """A failure before depth inference must not strand a preloaded model."""

        class LoadedDepthModel:
            def __init__(self):
                self.loaded = True
                self.unload_count = 0

            def unload_model(self):
                self.loaded = False
                self.unload_count += 1

        model = LoadedDepthModel()
        video_encoder = Mock()
        video_encoder.extract_frames.side_effect = OSError("frame extraction failed")
        orchestrator = ProcessingOrchestrator(
            Mock(),
            Mock(),
            Mock(),
            Mock(),
            Mock(),
            video_encoder,
            release_depth_model=model.unload_model,
        )

        with patch.object(
            orchestrator,
            "_setup_processing",
            return_value=(tmp_path, {"base": tmp_path}, tmp_path / "settings.json"),
        ):
            result = orchestrator.process(
                video_path=tmp_path / "source.mp4",
                output_dir=tmp_path,
                video_properties={"fps": 30, "frame_count": 1},
                settings={"keep_intermediates": True},
            )

        assert result is False
        assert model.loaded is False
        assert model.unload_count == 1

    def test_process_reuses_preacquired_writer_lock_without_releasing_it(self, tmp_path):
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())
        lock = JobWriterLock(tmp_path).acquire()
        try:
            with (
                patch.object(
                    orchestrator,
                    "_setup_processing",
                    return_value=(tmp_path, {"base": tmp_path}, None),
                ),
                patch.object(orchestrator, "_execute_pipeline", return_value=True),
                patch(
                    "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.JobWriterLock",
                    side_effect=AssertionError("must reuse the authoritative lock"),
                ),
            ):
                assert orchestrator.process(
                    video_path=tmp_path / "source.mp4",
                    output_dir=tmp_path,
                    video_properties={"fps": 30, "frame_count": 1},
                    settings={"keep_intermediates": True},
                    job_lock=lock,
                )

            assert lock.is_acquired is True
        finally:
            lock.release()

    @patch("src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.time.time")
    def test_process_exception_handling(self, mock_time):
        """Test process method handles exceptions."""
        mock_time.return_value = 1000.0

        depth_proc = Mock()
        video_encoder = Mock()
        video_encoder.extract_frames.side_effect = Exception("Test error")

        orchestrator = ProcessingOrchestrator(
            depth_proc, Mock(), Mock(), Mock(), Mock(), video_encoder
        )

        with patch.object(orchestrator, "_setup_processing") as mock_setup:
            mock_setup.return_value = (
                Path("/output"),
                {"base": Path("/output")},
                Path("/output/settings.json"),
            )

            result = orchestrator.process(
                video_path=Path("/input/video.mp4"),
                output_dir=Path("/output"),
                video_properties={"fps": 30, "frame_count": 100},
                settings={"keep_intermediates": True},
            )

            assert result is False

    def test_process_propagates_interrupted_error(self, tmp_path):
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())

        with (
            patch.object(
                orchestrator,
                "_setup_processing",
                return_value=(tmp_path, {"base": tmp_path}, tmp_path / "settings.json"),
            ),
            patch.object(
                orchestrator,
                "_execute_pipeline",
                side_effect=InterruptedError("Processing stopped by user request"),
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.update_processing_status"
            ) as update_status,
        ):
            with pytest.raises(InterruptedError, match="stopped by user request"):
                orchestrator.process(
                    video_path=tmp_path / "source.mp4",
                    output_dir=tmp_path,
                    video_properties={"fps": 30, "frame_count": 1},
                    settings={"keep_intermediates": True, "direct_vr_encode": True},
                )

        update_status.assert_not_called()
