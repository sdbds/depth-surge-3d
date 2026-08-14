"""Tests for DistortionProcessor module."""

import os
import threading
import time

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch

from src.depth_surge_3d.processing.frames.distortion_processor import DistortionProcessor
from src.depth_surge_3d.utils.imaging.image_processing import (
    apply_fisheye_distortion,
    calculate_fisheye_coordinates,
    remap_fisheye,
)


def test_remap_fisheye_matches_the_existing_opencv_contract_exactly():
    image = np.random.default_rng(7).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    x_map, y_map = calculate_fisheye_coordinates(64, 48, 100.0, "equisolid")
    expected = cv2.remap(
        image,
        x_map,
        y_map,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    actual = remap_fisheye(image, x_map, y_map)

    np.testing.assert_array_equal(actual, expected)


class TestDistortionProcessorInit:
    """Test DistortionProcessor initialization."""

    def test_init_default(self):
        """Test default initialization."""
        processor = DistortionProcessor()
        assert processor.verbose is False

    def test_init_verbose(self):
        """Test initialization with verbose enabled."""
        processor = DistortionProcessor(verbose=True)
        assert processor.verbose is True


class TestApplyDistortion:
    """Test apply_distortion method."""

    @pytest.fixture
    def mock_progress_tracker(self):
        """Create mock progress tracker."""
        tracker = Mock()
        tracker.update_progress = Mock()
        return tracker

    @pytest.fixture
    def temp_frames(self, tmp_path):
        """Create temporary frame files."""
        left_dir = tmp_path / "left"
        right_dir = tmp_path / "right"
        left_dir.mkdir()
        right_dir.mkdir()

        left_files = []
        right_files = []

        for i in range(3):
            left_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            right_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

            left_path = left_dir / f"frame_{i:04d}.png"
            right_path = right_dir / f"frame_{i:04d}.png"

            cv2.imwrite(str(left_path), left_frame)
            cv2.imwrite(str(right_path), right_frame)

            left_files.append(left_path)
            right_files.append(right_path)

        return {
            "left_files": left_files,
            "right_files": right_files,
            "base": tmp_path,
        }

    def test_apply_distortion_success(self, temp_frames, mock_progress_tracker, tmp_path):
        """Test successful distortion application."""
        processor = DistortionProcessor()

        left_distorted = tmp_path / "left_distorted"
        right_distorted = tmp_path / "right_distorted"
        left_distorted.mkdir()
        right_distorted.mkdir()

        directories = {
            "left_distorted": left_distorted,
            "right_distorted": right_distorted,
        }

        settings = {
            "keep_intermediates": True,
            "fisheye_fov": 90,
            "fisheye_projection": "equidistant",
        }

        with patch(
            "src.depth_surge_3d.processing.frames.distortion_processor.remap_fisheye",
            side_effect=lambda img, x_map, y_map: img,
        ) as distort:
            result = processor.apply_distortion(
                temp_frames["left_files"],
                temp_frames["right_files"],
                directories,
                settings,
                mock_progress_tracker,
            )
            resumed = processor.apply_distortion(
                temp_frames["left_files"],
                temp_frames["right_files"],
                directories,
                settings,
                mock_progress_tracker,
            )

        assert result is True
        assert resumed is True
        assert distort.call_count == 6

        # Check output files were created
        assert len(list(left_distorted.glob("*.png"))) == 3
        assert len(list(right_distorted.glob("*.png"))) == 3

    def test_distortion_progress_reports_every_ordered_pair(self):
        tracker = Mock()

        for index in range(3):
            DistortionProcessor._report_distortion_progress(tracker, index, 3)

        assert [call.kwargs["frame_num"] for call in tracker.update_progress.call_args_list] == [
            1,
            2,
            3,
        ]

    def test_apply_distortion_without_intermediates(
        self, temp_frames, mock_progress_tracker, tmp_path
    ):
        """Distorted working frames are written even when final retention is disabled."""
        processor = DistortionProcessor()

        left_distorted = tmp_path / "left_distorted"
        right_distorted = tmp_path / "right_distorted"
        left_distorted.mkdir()
        right_distorted.mkdir()
        directories = {
            "left_distorted": left_distorted,
            "right_distorted": right_distorted,
        }

        settings = {
            "keep_intermediates": False,
            "fisheye_fov": 90,
            "fisheye_projection": "equidistant",
        }

        with patch(
            "src.depth_surge_3d.processing.frames.distortion_processor.remap_fisheye",
            side_effect=lambda img, x_map, y_map: img,
        ):
            result = processor.apply_distortion(
                temp_frames["left_files"],
                temp_frames["right_files"],
                directories,
                settings,
                mock_progress_tracker,
            )

        assert result is True
        assert len(list(left_distorted.glob("*.png"))) == 3
        assert len(list(right_distorted.glob("*.png"))) == 3

    def test_apply_distortion_reuses_one_map_without_changing_pixels(
        self, temp_frames, mock_progress_tracker, tmp_path
    ):
        processor = DistortionProcessor()
        left_distorted = tmp_path / "left_distorted"
        right_distorted = tmp_path / "right_distorted"
        left_distorted.mkdir()
        right_distorted.mkdir()
        settings = {
            "fisheye_fov": 100.0,
            "fisheye_projection": "equisolid",
        }
        source = cv2.imread(str(temp_frames["left_files"][0]))
        expected = apply_fisheye_distortion(source, 100.0, "equisolid")

        with patch(
            "src.depth_surge_3d.processing.frames.distortion_processor."
            "calculate_fisheye_coordinates",
            wraps=calculate_fisheye_coordinates,
        ) as calculate:
            result = processor.apply_distortion(
                temp_frames["left_files"],
                temp_frames["right_files"],
                {
                    "left_distorted": left_distorted,
                    "right_distorted": right_distorted,
                },
                settings,
                mock_progress_tracker,
            )

        assert result is True
        assert calculate.call_count == 1
        actual = cv2.imread(str(left_distorted / temp_frames["left_files"][0].name))
        np.testing.assert_array_equal(actual, expected)

    def test_apply_distortion_runs_frame_pairs_concurrently(
        self, temp_frames, mock_progress_tracker, tmp_path
    ):
        processor = DistortionProcessor()
        left_distorted = tmp_path / "left_distorted"
        right_distorted = tmp_path / "right_distorted"
        left_distorted.mkdir()
        right_distorted.mkdir()
        lock = threading.Lock()
        active = 0
        max_active = 0

        def observed_remap(image, x_map, y_map):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return image

        with (
            patch(
                "src.depth_surge_3d.processing.frames.distortion_processor."
                "calculate_frame_stage_workers",
                return_value=2,
            ),
            patch(
                "src.depth_surge_3d.processing.frames.distortion_processor.remap_fisheye",
                side_effect=observed_remap,
            ),
        ):
            result = processor.apply_distortion(
                temp_frames["left_files"],
                temp_frames["right_files"],
                {
                    "left_distorted": left_distorted,
                    "right_distorted": right_distorted,
                },
                {"fisheye_fov": 90.0, "fisheye_projection": "equidistant"},
                mock_progress_tracker,
            )

        assert result is True
        assert max_active >= 2

    def test_apply_distortion_rejects_mixed_shapes_before_any_output(
        self, temp_frames, mock_progress_tracker, tmp_path
    ):
        mismatched = np.zeros((80, 100, 3), dtype=np.uint8)
        assert cv2.imwrite(str(temp_frames["right_files"][1]), mismatched)
        left_distorted = tmp_path / "left_distorted"
        right_distorted = tmp_path / "right_distorted"
        left_distorted.mkdir()
        right_distorted.mkdir()

        result = DistortionProcessor().apply_distortion(
            temp_frames["left_files"],
            temp_frames["right_files"],
            {
                "left_distorted": left_distorted,
                "right_distorted": right_distorted,
            },
            {"fisheye_fov": 90.0, "fisheye_projection": "equidistant"},
            mock_progress_tracker,
        )

        assert result is False
        assert not list(left_distorted.glob("*.png"))
        assert not list(right_distorted.glob("*.png"))
        assert not (left_distorted / "metadata.json").exists()

    def test_apply_distortion_cancels_pending_work_and_waits_on_failure(self, tmp_path):
        left_dir = tmp_path / "left"
        right_dir = tmp_path / "right"
        left_output = tmp_path / "left_distorted"
        right_output = tmp_path / "right_distorted"
        for directory in (left_dir, right_dir, left_output, right_output):
            directory.mkdir()
        left_files = []
        right_files = []
        for index in range(12):
            image = np.full((8, 8, 3), index, dtype=np.uint8)
            left_file = left_dir / f"frame_{index:04d}.png"
            right_file = right_dir / f"frame_{index:04d}.png"
            assert cv2.imwrite(str(left_file), image)
            assert cv2.imwrite(str(right_file), image)
            left_files.append(left_file)
            right_files.append(right_file)

        first_started = threading.Event()
        failure_raised = threading.Event()
        release_workers = threading.Event()
        executed = []
        executed_lock = threading.Lock()

        def controlled_worker(left_file, *_args):
            index = int(left_file.stem.rsplit("_", 1)[1])
            with executed_lock:
                executed.append(index)
            if index == 0:
                first_started.set()
                assert release_workers.wait(5)
                return True
            if index == 1:
                assert first_started.wait(2)
                failure_raised.set()
                raise RuntimeError("planned write failure")
            return True

        result = {}
        processor = DistortionProcessor()

        def run_stage():
            result["value"] = processor.apply_distortion(
                left_files,
                right_files,
                {
                    "left_distorted": left_output,
                    "right_distorted": right_output,
                },
                {"fisheye_fov": 90.0, "fisheye_projection": "equidistant"},
            )

        with (
            patch(
                "src.depth_surge_3d.processing.frames.distortion_processor."
                "calculate_frame_stage_workers",
                return_value=2,
            ),
            patch.object(
                processor,
                "_distort_single_frame_pair",
                side_effect=controlled_worker,
            ),
        ):
            stage_thread = threading.Thread(target=run_stage)
            stage_thread.start()
            try:
                assert failure_raised.wait(2)
                time.sleep(0.05)
                assert stage_thread.is_alive()
            finally:
                release_workers.set()
                stage_thread.join(timeout=5)

        assert not stage_thread.is_alive()
        assert result["value"] is False
        assert len(executed) == 2
        assert set(executed) == {0, 1}
        assert not (left_output / "metadata.json").exists()

    def test_apply_distortion_without_progress_tracker(self, temp_frames, tmp_path):
        """CLI distortion succeeds without a web progress tracker."""
        processor = DistortionProcessor()
        left_distorted = tmp_path / "left_distorted"
        right_distorted = tmp_path / "right_distorted"
        left_distorted.mkdir()
        right_distorted.mkdir()
        settings = {
            "keep_intermediates": False,
            "fisheye_fov": 90,
            "fisheye_projection": "equidistant",
        }

        with patch(
            "src.depth_surge_3d.processing.frames.distortion_processor.remap_fisheye",
            side_effect=lambda img, x_map, y_map: img,
        ):
            result = processor.apply_distortion(
                temp_frames["left_files"],
                temp_frames["right_files"],
                {
                    "left_distorted": left_distorted,
                    "right_distorted": right_distorted,
                },
                settings,
                progress_tracker=None,
            )

        assert result is True

    def test_apply_distortion_missing_file(self, mock_progress_tracker, tmp_path):
        """Test distortion with missing frame file."""
        processor = DistortionProcessor()

        fake_left = [tmp_path / "nonexistent_left.png"]
        fake_right = [tmp_path / "nonexistent_right.png"]

        directories = {}
        settings = {
            "keep_intermediates": False,
            "fisheye_fov": 90,
            "fisheye_projection": "equidistant",
        }

        result = processor.apply_distortion(
            fake_left, fake_right, directories, settings, mock_progress_tracker
        )

        assert result is False

    def test_apply_distortion_exception_handling(self, temp_frames, mock_progress_tracker):
        """Test exception handling during distortion."""
        processor = DistortionProcessor()

        directories = {}
        settings = {
            "keep_intermediates": False,
            "fisheye_fov": 90,
            "fisheye_projection": "equidistant",
        }

        with patch(
            "src.depth_surge_3d.processing.frames.distortion_processor.remap_fisheye",
            side_effect=RuntimeError("GPU error"),
        ):
            result = processor.apply_distortion(
                temp_frames["left_files"],
                temp_frames["right_files"],
                directories,
                settings,
                mock_progress_tracker,
            )

        assert result is False


class TestCropFrames:
    """Test crop_frames method."""

    @pytest.fixture
    def mock_progress_tracker(self):
        """Create mock progress tracker."""
        tracker = Mock()
        tracker.update_progress = Mock()
        return tracker

    @pytest.fixture
    def temp_frames(self, tmp_path):
        """Create temporary frame directories with test frames."""
        left_dir = tmp_path / "left_frames"
        right_dir = tmp_path / "right_frames"
        left_dir.mkdir()
        right_dir.mkdir()

        # Create test frames (large enough for cropping)
        for i in range(3):
            left_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            right_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            cv2.imwrite(str(left_dir / f"frame_{i:04d}.png"), left_frame)
            cv2.imwrite(str(right_dir / f"frame_{i:04d}.png"), right_frame)

        left_cropped = tmp_path / "left_cropped"
        right_cropped = tmp_path / "right_cropped"
        left_cropped.mkdir()
        right_cropped.mkdir()

        return {
            "base": tmp_path,
            "left_frames": left_dir,
            "right_frames": right_dir,
            "left_cropped": left_cropped,
            "right_cropped": right_cropped,
        }

    def test_crop_frames_success(self, temp_frames, mock_progress_tracker):
        """A factor-one crop materializes byte-identical files without decoding."""
        processor = DistortionProcessor()

        settings = {
            "apply_distortion": False,
            "per_eye_width": 1920,
            "per_eye_height": 1080,
            "crop_factor": 1.0,
        }

        with (
            patch(
                "src.depth_surge_3d.processing.frames.distortion_processor.cv2.imread",
                side_effect=AssertionError("factor-one crop must not decode"),
            ),
            patch(
                "src.depth_surge_3d.processing.frames.distortion_processor.cv2.imwrite",
                side_effect=AssertionError("factor-one crop must not encode"),
            ),
        ):
            result = processor.crop_frames(
                temp_frames, settings, mock_progress_tracker, total_frames=3
            )
            resumed = processor.crop_frames(
                temp_frames, settings, mock_progress_tracker, total_frames=3
            )

        assert result is True
        assert resumed is True

        for source_key, output_key in (
            ("left_frames", "left_cropped"),
            ("right_frames", "right_cropped"),
        ):
            sources = sorted(temp_frames[source_key].glob("*.png"))
            outputs = sorted(temp_frames[output_key].glob("*.png"))
            assert len(outputs) == 3
            for source, output in zip(sources, outputs):
                assert source.read_bytes() == output.read_bytes()
                assert os.path.samefile(source, output)

    def test_factor_one_crop_falls_back_to_byte_exact_copy(
        self, temp_frames, mock_progress_tracker
    ):
        processor = DistortionProcessor()
        settings = {
            "apply_distortion": False,
            "per_eye_width": 1920,
            "per_eye_height": 1080,
            "crop_factor": 1.0,
        }

        with patch(
            "src.depth_surge_3d.processing.frames.distortion_processor.os.link",
            side_effect=OSError("hard links unavailable"),
        ):
            result = processor.crop_frames(
                temp_frames, settings, mock_progress_tracker, total_frames=3
            )

        assert result is True
        source = sorted(temp_frames["left_frames"].glob("*.png"))[0]
        output = sorted(temp_frames["left_cropped"].glob("*.png"))[0]
        assert output.read_bytes() == source.read_bytes()
        assert not os.path.samefile(source, output)

    def test_transformed_crop_runs_frame_pairs_concurrently(
        self, temp_frames, mock_progress_tracker
    ):
        processor = DistortionProcessor()
        real_crop = processor._crop_single_frame_pair
        lock = threading.Lock()
        active = 0
        max_active = 0

        def observed_crop(*args):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(0.03)
                return real_crop(*args)
            finally:
                with lock:
                    active -= 1

        with (
            patch(
                "src.depth_surge_3d.processing.frames.distortion_processor."
                "calculate_frame_stage_workers",
                return_value=2,
            ),
            patch.object(
                processor,
                "_crop_single_frame_pair",
                side_effect=observed_crop,
            ),
        ):
            result = processor.crop_frames(
                temp_frames,
                {
                    "apply_distortion": False,
                    "per_eye_width": 1920,
                    "per_eye_height": 1080,
                    "crop_factor": 0.8,
                },
                mock_progress_tracker,
                total_frames=3,
            )

        assert result is True
        assert max_active >= 2

    def test_transformed_crop_uses_source_size_for_worker_memory(
        self, temp_frames, mock_progress_tracker
    ):
        with patch(
            "src.depth_surge_3d.processing.frames.distortion_processor."
            "calculate_frame_stage_workers",
            return_value=1,
        ) as calculate:
            result = DistortionProcessor().crop_frames(
                temp_frames,
                {
                    "apply_distortion": False,
                    "per_eye_width": 320,
                    "per_eye_height": 180,
                    "crop_factor": 0.8,
                },
                mock_progress_tracker,
                total_frames=3,
            )

        assert result is True
        calculate.assert_called_once_with(3, 1920 * 1080 * 48)

    def test_parallel_center_crop_preserves_exact_pixels(self, tmp_path):
        left_dir = tmp_path / "left_frames"
        right_dir = tmp_path / "right_frames"
        left_output = tmp_path / "left_cropped"
        right_output = tmp_path / "right_cropped"
        for directory in (left_dir, right_dir, left_output, right_output):
            directory.mkdir()
        source = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
        assert cv2.imwrite(str(left_dir / "frame_0000.png"), source)
        assert cv2.imwrite(str(right_dir / "frame_0000.png"), 255 - source)

        result = DistortionProcessor().crop_frames(
            {
                "left_frames": left_dir,
                "right_frames": right_dir,
                "left_cropped": left_output,
                "right_cropped": right_output,
            },
            {
                "apply_distortion": False,
                "per_eye_width": 10,
                "per_eye_height": 8,
                "crop_factor": 0.5,
            },
            total_frames=1,
        )

        assert result is True
        actual_left = cv2.imread(str(left_output / "frame_0000.png"))
        actual_right = cv2.imread(str(right_output / "frame_0000.png"))
        np.testing.assert_array_equal(actual_left, source[2:6, 2:7])
        np.testing.assert_array_equal(actual_right, (255 - source)[2:6, 2:7])

    def test_transformed_crop_detects_later_failure_before_submitting_more(self, tmp_path):
        left_dir = tmp_path / "left_frames"
        right_dir = tmp_path / "right_frames"
        left_output = tmp_path / "left_cropped"
        right_output = tmp_path / "right_cropped"
        for directory in (left_dir, right_dir, left_output, right_output):
            directory.mkdir()
        for index in range(12):
            image = np.full((8, 8, 3), index, dtype=np.uint8)
            assert cv2.imwrite(str(left_dir / f"frame_{index:04d}.png"), image)
            assert cv2.imwrite(str(right_dir / f"frame_{index:04d}.png"), image)

        first_started = threading.Event()
        failure_raised = threading.Event()
        release_first = threading.Event()
        executed = []
        result = {}
        processor = DistortionProcessor()

        def controlled_crop(left_file, *_args):
            index = int(left_file.stem.rsplit("_", 1)[1])
            executed.append(index)
            if index == 0:
                first_started.set()
                assert release_first.wait(5)
                return True
            if index == 1:
                assert first_started.wait(2)
                failure_raised.set()
                raise RuntimeError("planned crop failure")
            return True

        def run_stage() -> None:
            result["value"] = processor.crop_frames(
                {
                    "left_frames": left_dir,
                    "right_frames": right_dir,
                    "left_cropped": left_output,
                    "right_cropped": right_output,
                },
                {
                    "apply_distortion": False,
                    "per_eye_width": 8,
                    "per_eye_height": 8,
                    "crop_factor": 0.8,
                },
                total_frames=12,
            )

        with (
            patch(
                "src.depth_surge_3d.processing.frames.distortion_processor."
                "calculate_frame_stage_workers",
                return_value=2,
            ),
            patch.object(
                processor,
                "_crop_single_frame_pair",
                side_effect=controlled_crop,
            ),
        ):
            stage_thread = threading.Thread(target=run_stage)
            stage_thread.start()
            try:
                assert failure_raised.wait(2)
                time.sleep(0.05)
                assert stage_thread.is_alive()
                assert len(executed) == 2
                assert set(executed) == {0, 1}
            finally:
                release_first.set()
                stage_thread.join(timeout=5)

        assert not stage_thread.is_alive()
        assert result["value"] is False
        assert not (left_output / "metadata.json").exists()

    def test_crop_frames_without_progress_tracker(self, temp_frames):
        """CLI cropping succeeds without a web progress tracker."""
        processor = DistortionProcessor()
        settings = {
            "apply_distortion": False,
            "per_eye_width": 1920,
            "per_eye_height": 1080,
            "crop_factor": 1.0,
        }

        with patch(
            "src.depth_surge_3d.processing.frames.distortion_processor.apply_center_crop",
            side_effect=lambda img, factor: img,
        ):
            result = processor.crop_frames(
                temp_frames,
                settings,
                progress_tracker=None,
                total_frames=3,
            )

        assert result is True

    def test_crop_frames_with_distortion(self, temp_frames, mock_progress_tracker, tmp_path):
        """Test cropping with fisheye distortion enabled."""
        processor = DistortionProcessor()

        # Create distorted directories
        left_distorted = tmp_path / "left_distorted"
        right_distorted = tmp_path / "right_distorted"
        left_distorted.mkdir()
        right_distorted.mkdir()

        # Add distorted frames
        for i in range(3):
            frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            cv2.imwrite(str(left_distorted / f"frame_{i:04d}.png"), frame)
            cv2.imwrite(str(right_distorted / f"frame_{i:04d}.png"), frame)

        directories = {
            **temp_frames,
            "left_distorted": left_distorted,
            "right_distorted": right_distorted,
        }

        settings = {
            "apply_distortion": True,
            "per_eye_width": 1920,
            "per_eye_height": 1080,
            "fisheye_crop_factor": 0.8,
        }

        with patch(
            "src.depth_surge_3d.processing.frames.distortion_processor.apply_fisheye_square_crop",
            side_effect=lambda img, w, h, factor: img,
        ):
            result = processor.crop_frames(
                directories, settings, mock_progress_tracker, total_frames=3
            )

        assert result is True

    def test_crop_frames_mismatched_count(self, temp_frames, mock_progress_tracker):
        """Test cropping with mismatched frame counts."""
        processor = DistortionProcessor()

        # Remove one right frame
        right_frames = list(temp_frames["right_frames"].glob("*.png"))
        right_frames[0].unlink()

        settings = {
            "apply_distortion": False,
            "per_eye_width": 1920,
            "per_eye_height": 1080,
            "crop_factor": 1.0,
        }

        with patch(
            "src.depth_surge_3d.processing.frames.distortion_processor.apply_center_crop",
            side_effect=lambda img, factor: img,
        ):
            result = processor.crop_frames(
                temp_frames, settings, mock_progress_tracker, total_frames=3
            )

        assert result is False

    def test_crop_frames_no_source(self, mock_progress_tracker, tmp_path):
        """Test cropping with no source directories."""
        processor = DistortionProcessor()

        directories = {"base": tmp_path}
        settings = {"apply_distortion": False}

        result = processor.crop_frames(directories, settings, mock_progress_tracker, total_frames=0)

        assert result is False

    def test_crop_frames_exception_handling(self, temp_frames, mock_progress_tracker):
        """Test exception handling during cropping."""
        processor = DistortionProcessor()

        settings = {
            "apply_distortion": False,
            "per_eye_width": 1920,
            "per_eye_height": 1080,
            "crop_factor": 0.8,
        }

        with patch(
            "src.depth_surge_3d.processing.frames.distortion_processor.apply_center_crop",
            side_effect=RuntimeError("Crop error"),
        ):
            result = processor.crop_frames(
                temp_frames, settings, mock_progress_tracker, total_frames=3
            )

        assert result is False


class TestCropSingleFramePair:
    """Test _crop_single_frame_pair method."""

    @pytest.fixture
    def temp_frames(self, tmp_path):
        """Create temporary frame files."""
        left_dir = tmp_path / "left"
        right_dir = tmp_path / "right"
        left_dir.mkdir()
        right_dir.mkdir()

        left_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        right_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        left_path = left_dir / "frame_0000.png"
        right_path = right_dir / "frame_0000.png"

        cv2.imwrite(str(left_path), left_frame)
        cv2.imwrite(str(right_path), right_frame)

        left_cropped = tmp_path / "left_cropped"
        right_cropped = tmp_path / "right_cropped"
        left_cropped.mkdir()
        right_cropped.mkdir()

        return {
            "left_path": left_path,
            "right_path": right_path,
            "left_cropped": left_cropped,
            "right_cropped": right_cropped,
        }

    def test_crop_single_pair_without_distortion(self, temp_frames):
        """Test cropping single pair without distortion."""
        processor = DistortionProcessor()

        directories = {
            "left_cropped": temp_frames["left_cropped"],
            "right_cropped": temp_frames["right_cropped"],
        }

        settings = {
            "apply_distortion": False,
            "per_eye_width": 1920,
            "per_eye_height": 1080,
            "crop_factor": 0.9,
        }

        with patch(
            "src.depth_surge_3d.processing.frames.distortion_processor.apply_center_crop",
            side_effect=lambda img, factor: img,
        ):
            result = processor._crop_single_frame_pair(
                temp_frames["left_path"],
                temp_frames["right_path"],
                directories,
                settings,
            )

        assert result is True
        assert (temp_frames["left_cropped"] / "frame_0000.png").exists()
        assert (temp_frames["right_cropped"] / "frame_0000.png").exists()

    def test_crop_single_pair_with_distortion(self, temp_frames):
        """Test cropping single pair with distortion."""
        processor = DistortionProcessor()

        directories = {
            "left_cropped": temp_frames["left_cropped"],
            "right_cropped": temp_frames["right_cropped"],
        }

        settings = {
            "apply_distortion": True,
            "per_eye_width": 1920,
            "per_eye_height": 1080,
            "fisheye_crop_factor": 0.8,
        }

        with patch(
            "src.depth_surge_3d.processing.frames.distortion_processor.apply_fisheye_square_crop",
            side_effect=lambda img, w, h, factor: img,
        ):
            result = processor._crop_single_frame_pair(
                temp_frames["left_path"],
                temp_frames["right_path"],
                directories,
                settings,
            )

        assert result is True

    def test_crop_single_pair_missing_file(self, tmp_path):
        """Test cropping with missing file."""
        processor = DistortionProcessor()

        fake_left = tmp_path / "nonexistent_left.png"
        fake_right = tmp_path / "nonexistent_right.png"

        directories = {}
        settings = {"apply_distortion": False}

        result = processor._crop_single_frame_pair(fake_left, fake_right, directories, settings)

        assert result is False

    def test_crop_single_pair_crop_factor_clamping(self, temp_frames):
        """Test that crop factor is clamped to valid range."""
        processor = DistortionProcessor()

        directories = {
            "left_cropped": temp_frames["left_cropped"],
            "right_cropped": temp_frames["right_cropped"],
        }

        # Test with out-of-range crop factor
        settings = {
            "apply_distortion": False,
            "per_eye_width": 1920,
            "per_eye_height": 1080,
            "crop_factor": 5.0,  # Will be clamped to 1.0
        }

        with patch(
            "src.depth_surge_3d.processing.frames.distortion_processor.apply_center_crop",
            side_effect=lambda img, factor: img,
        ) as mock_crop:
            processor._crop_single_frame_pair(
                temp_frames["left_path"],
                temp_frames["right_path"],
                directories,
                settings,
            )

            # Check that clamped value was used (1.0 max for non-distortion)
            call_args = mock_crop.call_args_list[0][0]
            assert call_args[1] == 1.0  # Clamped from 5.0 to 1.0

    def test_crop_single_pair_fisheye_factor_clamping(self, temp_frames):
        """Test that fisheye crop factor is clamped to valid range."""
        processor = DistortionProcessor()

        directories = {
            "left_cropped": temp_frames["left_cropped"],
            "right_cropped": temp_frames["right_cropped"],
        }

        # Test with out-of-range fisheye crop factor
        settings = {
            "apply_distortion": True,
            "per_eye_width": 1920,
            "per_eye_height": 1080,
            "fisheye_crop_factor": 3.0,  # Will be clamped to 2.0
        }

        with patch(
            "src.depth_surge_3d.processing.frames.distortion_processor.apply_fisheye_square_crop",
            side_effect=lambda img, w, h, factor: img,
        ) as mock_crop:
            processor._crop_single_frame_pair(
                temp_frames["left_path"],
                temp_frames["right_path"],
                directories,
                settings,
            )

            # Check that clamped value was used (2.0 max for fisheye)
            call_args = mock_crop.call_args_list[0][0]
            assert call_args[3] == 2.0  # Clamped from 3.0 to 2.0


class TestGetStereoSourceDirs:
    """Test _get_stereo_source_dirs static method."""

    def test_get_source_with_distortion(self, tmp_path):
        """Test getting source directories when distortion is enabled."""
        left_distorted = tmp_path / "left_distorted"
        right_distorted = tmp_path / "right_distorted"
        left_distorted.mkdir()
        right_distorted.mkdir()

        # Create frame in distorted directory
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(left_distorted / "frame_0000.png"), frame)

        directories = {
            "left_distorted": left_distorted,
            "right_distorted": right_distorted,
        }

        settings = {"apply_distortion": True}

        result = DistortionProcessor._get_stereo_source_dirs(directories, settings)

        assert result == (left_distorted, right_distorted)

    def test_get_source_without_distortion(self, tmp_path):
        """Test getting source directories when distortion is disabled."""
        left_frames = tmp_path / "left_frames"
        right_frames = tmp_path / "right_frames"
        left_frames.mkdir()
        right_frames.mkdir()

        # Create frame in frames directory
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(left_frames / "frame_0000.png"), frame)

        directories = {
            "left_frames": left_frames,
            "right_frames": right_frames,
        }

        settings = {"apply_distortion": False}

        result = DistortionProcessor._get_stereo_source_dirs(directories, settings)

        assert result == (left_frames, right_frames)

    def test_get_source_does_not_fallback_when_distortion_is_required(self, tmp_path):
        """Undistorted frames cannot masquerade as a completed distortion stage."""
        left_frames = tmp_path / "left_frames"
        right_frames = tmp_path / "right_frames"
        left_frames.mkdir()
        right_frames.mkdir()

        # Create frame in frames directory
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(left_frames / "frame_0000.png"), frame)

        directories = {
            "left_frames": left_frames,
            "right_frames": right_frames,
        }

        # Request distortion but directories don't exist
        settings = {"apply_distortion": True}

        result = DistortionProcessor._get_stereo_source_dirs(directories, settings)

        assert result is None

    def test_get_source_empty_distorted_dirs(self, tmp_path):
        """Test handling empty distorted directories."""
        left_distorted = tmp_path / "left_distorted"
        right_distorted = tmp_path / "right_distorted"
        left_frames = tmp_path / "left_frames"
        right_frames = tmp_path / "right_frames"

        left_distorted.mkdir()
        right_distorted.mkdir()
        left_frames.mkdir()
        right_frames.mkdir()

        # Create frame only in frames directory (distorted is empty)
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(left_frames / "frame_0000.png"), frame)

        directories = {
            "left_distorted": left_distorted,
            "right_distorted": right_distorted,
            "left_frames": left_frames,
            "right_frames": right_frames,
        }

        settings = {"apply_distortion": True}

        result = DistortionProcessor._get_stereo_source_dirs(directories, settings)

        assert result is None

    def test_get_source_no_directories(self):
        """Test handling when no source directories exist."""
        directories = {}
        settings = {"apply_distortion": False}

        result = DistortionProcessor._get_stereo_source_dirs(directories, settings)

        assert result is None
