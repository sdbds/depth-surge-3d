"""Tests for DepthMapProcessor module."""

import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock, patch

from src.depth_surge_3d.inference.depth.types import DepthBatch, DepthRepresentation
from src.depth_surge_3d.processing.frames.depth_processor import DepthMapProcessor


class TestDepthMapProcessorInit:
    """Test DepthMapProcessor initialization."""

    def test_init_with_estimator(self):
        """Test initialization with depth estimator."""
        estimator = Mock()
        processor = DepthMapProcessor(estimator, verbose=False)

        assert processor.depth_estimator == estimator
        assert processor.verbose is False

    def test_init_with_verbose(self):
        """Test initialization with verbose enabled."""
        estimator = Mock()
        processor = DepthMapProcessor(estimator, verbose=True)

        assert processor.verbose is True

    def test_unbounded_array_api_is_removed(self):
        assert "generate_depth_maps" not in vars(DepthMapProcessor)


class TestGenerateDepthMaps:
    """Test generate_depth_maps main entry point."""

    @pytest.fixture
    def mock_estimator(self):
        """Create mock depth estimator."""
        estimator = Mock()
        estimator.estimate_depth_batch = Mock(return_value=np.random.rand(3, 100, 100))
        return estimator

    @pytest.fixture
    def mock_progress_tracker(self):
        """Create mock progress tracker."""
        tracker = Mock()
        tracker.update_progress = Mock()
        return tracker

    @pytest.fixture
    def temp_frames(self, tmp_path):
        """Create temporary frame files."""
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()

        frame_files = []
        for i in range(3):
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            frame_path = frame_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_files.append(frame_path)

        return frame_files

    def test_generate_depth_map_files_writes_each_chunk_without_stacking(
        self, mock_progress_tracker, temp_frames, tmp_path
    ):
        """Long videos retain only one inference chunk in memory."""
        estimator = Mock()
        estimator.model_path = None
        estimator.get_model_info.return_value = {
            "family": "processor-test",
            "revision": "1",
        }
        estimator.estimate_depth_batch.side_effect = lambda frames, **kwargs: DepthBatch(
            np.full((len(frames), 2, 3), 0.5, dtype=np.float32),
            DepthRepresentation.INVERSE_DEPTH,
        )
        processor = DepthMapProcessor(estimator)
        depth_dir = tmp_path / "03_disparity_maps"
        depth_dir.mkdir()
        settings = {
            "depth_resolution": "1080",
            "target_fps": 30,
            "keep_intermediates": False,
            "super_sample": "none",
            "per_eye_width": 100,
            "per_eye_height": 100,
        }

        with (
            patch.object(processor, "_determine_chunk_params", return_value=(2, 1080)),
            patch.object(processor, "_clear_gpu_memory") as clear_gpu_memory,
        ):
            result = processor.generate_depth_map_files(
                temp_frames,
                settings,
                {"base": tmp_path, "disparity_maps": depth_dir},
                mock_progress_tracker,
            )

        assert result == [depth_dir / f"frame_{i:04d}.png" for i in range(3)]
        assert all(path.exists() for path in result)
        assert all(
            cv2.imread(str(path), cv2.IMREAD_UNCHANGED).dtype == np.uint16 for path in result
        )
        assert estimator.estimate_depth_batch.call_count == 2
        assert clear_gpu_memory.call_count == 1


def test_native_shape_estimate_uses_estimator_output_contract():
    class SquareEstimator:
        @staticmethod
        def estimate_output_shape(
            frame_width: int, frame_height: int, input_size: int
        ) -> tuple[int, int]:
            del frame_width, frame_height
            return input_size, input_size

    processor = DepthMapProcessor(SquareEstimator())

    assert processor._estimate_native_shape(1920, 1080, 768) == (768, 768)


class TestDetermineChunkParams:
    """Test chunk parameter determination."""

    @pytest.fixture
    def mock_estimator(self):
        """Create mock depth estimator with model info."""
        estimator = Mock()
        estimator.model_type = "v3"
        estimator.get_model_size = Mock(return_value="large")
        return estimator

    def test_determine_chunk_params_auto_resolution(self, mock_estimator):
        """Test auto resolution detection."""
        processor = DepthMapProcessor(mock_estimator, verbose=False)

        with patch(
            "src.depth_surge_3d.processing.frames.depth_processor.get_vram_info",
            return_value={"total": 8.0, "available": 6.0},
        ):
            with patch(
                "src.depth_surge_3d.processing.frames.depth_processor.calculate_optimal_chunk_size",
                return_value=4,
            ):
                chunk_size, input_size = processor._determine_chunk_params(1920, 1080, "auto")

        assert chunk_size == 4
        assert input_size == 1080  # Auto selected based on resolution

    def test_determine_chunk_params_manual_resolution(self, mock_estimator):
        """Test manual resolution setting."""
        processor = DepthMapProcessor(mock_estimator, verbose=False)

        with patch(
            "src.depth_surge_3d.processing.frames.depth_processor.get_vram_info",
            return_value={"total": 8.0, "available": 6.0},
        ):
            with patch(
                "src.depth_surge_3d.processing.frames.depth_processor.calculate_optimal_chunk_size",
                return_value=2,
            ):
                chunk_size, input_size = processor._determine_chunk_params(1920, 1080, "720")

        assert chunk_size == 2
        assert input_size == 720

    def test_determine_chunk_params_invalid_manual(self, mock_estimator):
        """Test invalid manual resolution falls back to auto."""
        processor = DepthMapProcessor(mock_estimator, verbose=False)

        with patch(
            "src.depth_surge_3d.processing.frames.depth_processor.get_vram_info",
            return_value={"total": 8.0, "available": 6.0},
        ):
            with patch(
                "src.depth_surge_3d.processing.frames.depth_processor.calculate_optimal_chunk_size",
                return_value=4,
            ):
                chunk_size, input_size = processor._determine_chunk_params(1920, 1080, "invalid")

        assert input_size == 1080  # Fell back to auto

    def test_determine_chunk_params_cpu_mode(self, mock_estimator):
        """Test CPU mode without VRAM."""
        processor = DepthMapProcessor(mock_estimator, verbose=False)

        with patch(
            "src.depth_surge_3d.processing.frames.depth_processor.get_vram_info",
            return_value={"total": 0, "available": 0},
        ):
            chunk_size, input_size = processor._determine_chunk_params(1920, 1080, "auto")

        assert chunk_size in [4, 6, 8, 12, 16, 24, 32]  # Fixed size for CPU (actual constants)
        assert input_size == 1080


class TestAutoDetermineInputSize:
    """Test automatic input size determination."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return DepthMapProcessor(Mock(), verbose=False)

    def test_4k_resolution(self, processor):
        """Test 4K resolution input sizing."""
        input_size = processor._auto_determine_input_size(3840, 2160, 8.3)
        assert input_size == 2160  # Should cap at source resolution

    def test_1080p_resolution(self, processor):
        """Test 1080p resolution input sizing."""
        input_size = processor._auto_determine_input_size(1920, 1080, 2.1)
        assert input_size == 1080

    def test_720p_resolution(self, processor):
        """Test 720p resolution input sizing."""
        input_size = processor._auto_determine_input_size(1280, 720, 0.9)
        assert input_size == 640  # Falls to SD since 0.9 is not > MEGAPIXELS_720P (1.0)

    def test_sd_resolution(self, processor):
        """Test SD resolution input sizing."""
        input_size = processor._auto_determine_input_size(640, 480, 0.3)
        assert input_size == 640


class TestGetChunkSizeForResolution:
    """Test chunk size selection based on resolution."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return DepthMapProcessor(Mock(), verbose=False)

    def test_4k_chunk_size(self, processor):
        """Test 4K chunk size."""
        chunk_size = processor._get_chunk_size_for_resolution(2160)
        assert chunk_size == 4  # CHUNK_SIZE_4K

    def test_1440p_chunk_size(self, processor):
        """Test 1440p chunk size."""
        chunk_size = processor._get_chunk_size_for_resolution(1440)
        assert chunk_size == 6  # CHUNK_SIZE_1440P

    def test_1080p_chunk_size(self, processor):
        """Test 1080p chunk size."""
        chunk_size = processor._get_chunk_size_for_resolution(1080)
        assert chunk_size == 12  # CHUNK_SIZE_1080P_MANUAL

    def test_720p_chunk_size(self, processor):
        """Test 720p chunk size."""
        chunk_size = processor._get_chunk_size_for_resolution(720)
        assert chunk_size == 16  # CHUNK_SIZE_720P

    def test_small_chunk_size(self, processor):
        """Test small resolution chunk size."""
        chunk_size = processor._get_chunk_size_for_resolution(480)
        assert chunk_size == 32  # CHUNK_SIZE_SMALL


class TestClearGPUMemory:
    """Test GPU memory clearing."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return DepthMapProcessor(Mock(), verbose=False)

    def test_clear_gpu_memory_cuda_available(self, processor):
        """Test clearing GPU memory when CUDA is available."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.empty_cache") as mock_empty:
                with patch("torch.cuda.synchronize") as mock_sync:
                    with patch("torch.cuda.mem_get_info", return_value=(4 * 1024**3, 8 * 1024**3)):
                        processor._clear_gpu_memory()

        mock_empty.assert_called_once()
        mock_sync.assert_called_once()

    def test_clear_gpu_memory_cpu_mode(self, processor):
        """Test clearing GPU memory in CPU mode (no-op)."""
        with patch("torch.cuda.is_available", return_value=False):
            processor._clear_gpu_memory()
        # Should not raise any errors


class TestLoadChunkFrames:
    """Test loading chunk of frames."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return DepthMapProcessor(Mock(), verbose=False)

    @pytest.fixture
    def temp_frames(self, tmp_path):
        """Create temporary frame files."""
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()

        frame_files = []
        for i in range(3):
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            frame_path = frame_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_files.append(frame_path)

        return frame_files

    def test_load_chunk_frames_success(self, processor, temp_frames):
        """Test successful frame loading."""
        settings = {"super_sample": "none", "per_eye_width": 100, "per_eye_height": 100}

        result = processor._load_chunk_frames(temp_frames, settings)

        assert result is not None
        assert len(result) == 3
        assert all(isinstance(frame, np.ndarray) for frame in result)

    def test_load_chunk_frames_never_applies_super_sampling(self, processor, temp_frames):
        """Depth inference always receives the persisted source-frame resolution."""
        settings = {"super_sample": "2x", "per_eye_width": 200, "per_eye_height": 200}

        result = processor._load_chunk_frames(temp_frames, settings)

        assert result is not None
        assert len(result) == 3
        assert all(frame.shape == (100, 100, 3) for frame in result)

    def test_load_chunk_frames_missing_file(self, processor, temp_frames):
        """Test loading with missing file."""
        settings = {"super_sample": "none", "per_eye_width": 100, "per_eye_height": 100}

        # Add non-existent file
        bad_path = Path("/nonexistent/frame.png")
        chunk_files = temp_frames + [bad_path]

        result = processor._load_chunk_frames(chunk_files, settings)

        # Should still load valid frames
        assert result is not None
        assert len(result) == 3

    def test_load_chunk_frames_all_missing(self, processor):
        """Test loading with all files missing."""
        settings = {"super_sample": "none", "per_eye_width": 100, "per_eye_height": 100}
        chunk_files = [Path("/nonexistent/frame1.png"), Path("/nonexistent/frame2.png")]

        result = processor._load_chunk_frames(chunk_files, settings)

        assert result is None
