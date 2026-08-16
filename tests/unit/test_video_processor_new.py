"""Tests for VideoProcessor thin orchestrator."""

from unittest.mock import Mock

from src.depth_surge_3d.processing.orchestration.video_processor import VideoProcessor


class TestVideoProcessorInit:
    """Test VideoProcessor initialization."""

    def test_init_creates_all_processors(self):
        """Test that init creates all specialized processor modules."""
        mock_estimator = Mock()

        processor = VideoProcessor(mock_estimator, verbose=False)

        assert processor.depth_processor is not None
        assert processor.stereo_generator is not None
        assert processor.distortion_processor is not None
        assert processor.upscaler is not None
        assert processor.vr_assembler is not None
        assert processor.video_encoder is not None
        assert processor.orchestrator is not None
        removed_delegates = {
            "_get_total_steps",
            "_update_step_progress",
            "_handle_step_error",
            "_setup_processing",
            "_finalize_processing",
            "_crop_frames",
            "_apply_upscaling",
            "_process_upscaling_frames",
            "_upscale_frame_pair",
        }
        assert removed_delegates.isdisjoint(vars(VideoProcessor))
        assert not hasattr(processor, "_settings_file")

    def test_init_with_verbose(self):
        """Test initialization with verbose enabled."""
        mock_estimator = Mock()

        processor = VideoProcessor(mock_estimator, verbose=True)

        assert processor.depth_processor.verbose is True
        assert processor.stereo_generator.verbose is True


class TestVideoProcessorProcess:
    """Test VideoProcessor.process delegation."""

    def test_process_unloads_depth_model_before_step_3(self, tmp_path):
        """VideoProcessor wires its estimator into the stage-boundary cleanup."""

        class LoadedDepthEstimator:
            def __init__(self):
                self.loaded = True

            def unload_model(self):
                self.loaded = False

        estimator = LoadedDepthEstimator()
        processor = VideoProcessor(estimator)
        frame = tmp_path / "frame_000001.png"
        depth = tmp_path / "depth_000001.png"
        processor.orchestrator._setup_processing = Mock(
            return_value=(
                tmp_path,
                {"base": tmp_path, "frames": tmp_path / "frames"},
                None,
            )
        )
        processor.video_encoder.extract_frames = Mock(return_value=[frame])
        processor.depth_processor.generate_depth_map_files = Mock(return_value=[depth])
        model_loaded_at_step_3 = None

        def observe_model_state(*_args):
            nonlocal model_loaded_at_step_3
            model_loaded_at_step_3 = estimator.loaded
            return False

        processor.stereo_generator.create_stereo_pairs_from_files = observe_model_state

        result = processor.process(
            tmp_path / "source.mp4",
            tmp_path,
            {"fps": 30.0, "frame_count": 1},
            {},
        )

        assert result is False
        assert estimator.loaded is False
        assert model_loaded_at_step_3 is False

    def test_process_delegates_to_orchestrator(self, tmp_path):
        """Test that process method delegates to orchestrator."""
        mock_estimator = Mock()
        processor = VideoProcessor(mock_estimator)

        # Mock the orchestrator's process method
        processor.orchestrator.process = Mock(return_value=True)

        video_path = tmp_path / "test.mp4"
        video_path.touch()
        output_dir = tmp_path / "output"
        video_properties = {"fps": 30}
        settings = {"vr_format": "side_by_side"}
        progress_callback = None

        result = processor.process(
            video_path, output_dir, video_properties, settings, progress_callback
        )

        assert result is True
        processor.orchestrator.process.assert_called_once()
