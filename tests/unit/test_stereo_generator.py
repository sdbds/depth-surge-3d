"""Tests for StereoPairGenerator module."""

import pytest
import cv2
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from src.depth_surge_3d.processing.frames.stereo_generator import (
    StereoPairGenerator,
    _canonical_to_signed_pixel_disparity,
    _process_single_stereo_pair,
)
from src.depth_surge_3d.processing.frames import stereo_generator
from src.depth_surge_3d.processing.frames.depth_storage import canonical_json_hash


def _write_canonical_metadata(depth_dir: Path, frame_names: list[str], shape=(8, 8)) -> None:
    metadata = {
        "schema_version": 1,
        "algorithm_version": "scene-percentile-v1",
        "representation": "relative_disparity",
        "near_value": 1.0,
        "far_value": 0.0,
        "encoding": "uint16_png",
        "encoding_scale": 65535.0,
        "num_frames": len(frame_names),
        "frame_names": frame_names,
        "native_shape": list(shape),
        "source_raw_fingerprint": "raw",
        "source_model_fingerprint": "model",
        "scene_manifest_fingerprint": "scene",
        "depth_bounds_fingerprint": "bounds",
    }
    metadata["fingerprint"] = canonical_json_hash(metadata)
    (depth_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


class TestProcessSingleStereoPair:
    """Test _process_single_stereo_pair worker function."""

    def test_canonical_disparity_uses_target_width_strength_and_convergence(self):
        canonical = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)

        disparity = _canonical_to_signed_pixel_disparity(
            canonical,
            render_width=200,
            stereo_strength=2.0,
            convergence=0.5,
        )

        np.testing.assert_allclose(disparity, [[-2.0, 0.0, 2.0]])

    def test_near_marker_is_rightward_in_left_eye(self):
        frame = np.zeros((5, 100, 3), dtype=np.uint8)
        frame[:, 50] = 255
        canonical = np.ones((5, 100), dtype=np.float32)

        left, right, _ = _process_single_stereo_pair(
            (
                frame,
                canonical,
                "frame_0000",
                None,
                None,
                {
                    "stereo_strength": 4.0,
                    "convergence": 0.5,
                    "hole_fill_quality": "none",
                },
            )
        )

        left_x = int(np.argmax(left[2, :, 0]))
        right_x = int(np.argmax(right[2, :, 0]))
        assert left_x > right_x

    def test_process_pair_basic(self, tmp_path):
        """Test basic stereo pair processing."""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        depth_map = np.random.rand(100, 100)
        frame_name = "frame_0000"

        left_path = str(tmp_path / "left.png")
        right_path = str(tmp_path / "right.png")

        settings = {
            "baseline": 0.065,
            "focal_length": 1000,
            "hole_fill_quality": "fast",
        }

        with patch(
            "src.depth_surge_3d.processing.frames.stereo_generator._canonical_to_signed_pixel_disparity"
        ) as mock_disp:
            with patch(
                "src.depth_surge_3d.processing.frames.stereo_generator.create_shifted_image"
            ) as mock_shift:
                with patch(
                    "src.depth_surge_3d.processing.frames.stereo_generator.hole_fill_image"
                ) as mock_fill:
                    mock_disp.return_value = np.zeros((100, 100))
                    mock_shift.return_value = frame
                    mock_fill.return_value = frame

                    left_img, right_img, name = _process_single_stereo_pair(
                        (frame, depth_map, frame_name, left_path, right_path, settings)
                    )

        assert name == frame_name
        assert Path(left_path).exists()
        assert Path(right_path).exists()

    def test_process_pair_no_hole_fill(self, tmp_path):
        """Test stereo pair processing without hole filling."""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        depth_map = np.random.rand(100, 100)

        settings = {
            "baseline": 0.065,
            "focal_length": 1000,
            "hole_fill_quality": "none",
        }

        with patch(
            "src.depth_surge_3d.processing.frames.stereo_generator._canonical_to_signed_pixel_disparity"
        ) as mock_disp:
            with patch(
                "src.depth_surge_3d.processing.frames.stereo_generator.create_shifted_image"
            ) as mock_shift:
                with patch(
                    "src.depth_surge_3d.processing.frames.stereo_generator.hole_fill_image"
                ) as mock_fill:
                    mock_disp.return_value = np.zeros((100, 100))
                    mock_shift.return_value = frame

                    _process_single_stereo_pair(
                        (frame, depth_map, "frame_0000", None, None, settings)
                    )

                    # Hole fill should not be called
                    mock_fill.assert_not_called()

    def test_process_pair_resizes_depth_map_to_frame_dimensions(self):
        """Depth inferred at a supersampled resolution still aligns to the source frame."""
        frame = np.random.randint(0, 255, (12, 16, 3), dtype=np.uint8)
        depth_map = np.ones((24, 32), dtype=np.float32)
        settings = {
            "baseline": 0.01,
            "focal_length": 10,
            "hole_fill_quality": "none",
        }

        left_img, right_img, name = _process_single_stereo_pair(
            (frame, depth_map, "frame_0000", None, None, settings)
        )

        assert left_img.shape == frame.shape
        assert right_img.shape == frame.shape
        assert name == "frame_0000"

    def test_process_pair_from_files_returns_only_frame_name(self, tmp_path):
        """A worker loads one pair, writes it, and does not return image arrays."""
        frame_path = tmp_path / "frame_0000.png"
        depth_path = tmp_path / "depth_0000.png"
        left_path = tmp_path / "left.png"
        right_path = tmp_path / "right.png"
        cv2.imwrite(
            str(frame_path),
            np.full((8, 8, 3), 127, dtype=np.uint8),
        )
        cv2.imwrite(
            str(depth_path),
            np.full((8, 8), 32768, dtype=np.uint16),
        )
        settings = {
            "baseline": 0.01,
            "focal_length": 10,
            "hole_fill_quality": "none",
        }

        result = stereo_generator._process_single_stereo_pair_from_files(
            (
                str(frame_path),
                str(depth_path),
                "frame_0000",
                str(left_path),
                str(right_path),
                settings,
                65535.0,
            )
        )

        assert result == "frame_0000"
        assert left_path.exists()
        assert right_path.exists()


class TestStereoPairGeneratorInit:
    """Test StereoPairGenerator initialization."""

    def test_init_default(self):
        """Test default initialization."""
        generator = StereoPairGenerator()
        assert generator.verbose is False

    def test_init_verbose(self):
        """Test initialization with verbose."""
        generator = StereoPairGenerator(verbose=True)
        assert generator.verbose is True


class TestCreateStereoPairs:
    """Test create_stereo_pairs method."""

    @pytest.fixture
    def mock_progress_tracker(self):
        """Create mock progress tracker."""
        tracker = Mock()
        tracker.update_progress = Mock()
        tracker.send_preview_frame = Mock()
        return tracker

    @pytest.fixture
    def temp_frames(self, tmp_path):
        """Create temporary frame data."""
        left_dir = tmp_path / "left_frames"
        right_dir = tmp_path / "right_frames"
        left_dir.mkdir()
        right_dir.mkdir()

        frames = np.random.randint(0, 255, (3, 100, 100, 3), dtype=np.uint8)
        depth_maps = np.random.rand(3, 100, 100)
        frame_files = [tmp_path / f"frame_{i:04d}.png" for i in range(3)]

        directories = {
            "left_frames": left_dir,
            "right_frames": right_dir,
        }

        return {
            "frames": frames,
            "depth_maps": depth_maps,
            "frame_files": frame_files,
            "directories": directories,
        }

    def test_create_stereo_pairs_success(self, temp_frames, mock_progress_tracker):
        """Test successful stereo pair creation."""
        generator = StereoPairGenerator()

        settings = {
            "baseline": 0.065,
            "focal_length": 1000,
            "hole_fill_quality": "fast",
            "keep_intermediates": True,
        }

        # Mock the multiprocessing Pool
        mock_pool = Mock()
        mock_imap_result = [
            (temp_frames["frames"][i], temp_frames["frames"][i], f"frame_{i:04d}") for i in range(3)
        ]
        mock_pool.__enter__ = Mock(return_value=mock_pool)
        mock_pool.__exit__ = Mock(return_value=False)
        mock_pool.imap = Mock(return_value=iter(mock_imap_result))

        with patch("multiprocessing.Pool", return_value=mock_pool):
            result = generator.create_stereo_pairs(
                temp_frames["frames"],
                temp_frames["depth_maps"],
                temp_frames["frame_files"],
                temp_frames["directories"],
                settings,
                mock_progress_tracker,
            )

        assert result is True
        mock_pool.imap.assert_called_once()

    def test_create_stereo_pairs_without_intermediates(self, temp_frames, mock_progress_tracker):
        """Test stereo pair creation without saving intermediates."""
        generator = StereoPairGenerator()

        settings = {
            "baseline": 0.065,
            "focal_length": 1000,
            "hole_fill_quality": "fast",
            "keep_intermediates": False,
        }

        # Mock the multiprocessing Pool
        mock_pool = Mock()
        mock_imap_result = [
            (temp_frames["frames"][i], temp_frames["frames"][i], f"frame_{i:04d}") for i in range(3)
        ]
        mock_pool.__enter__ = Mock(return_value=mock_pool)
        mock_pool.__exit__ = Mock(return_value=False)
        mock_pool.imap = Mock(return_value=iter(mock_imap_result))

        with patch("multiprocessing.Pool", return_value=mock_pool):
            result = generator.create_stereo_pairs(
                temp_frames["frames"],
                temp_frames["depth_maps"],
                temp_frames["frame_files"],
                temp_frames["directories"],
                settings,
                mock_progress_tracker,
            )

        assert result is True

    def test_create_stereo_pairs_without_progress_tracker(self, temp_frames):
        """CLI processing does not require a web progress tracker."""
        generator = StereoPairGenerator()
        settings = {
            "baseline": 0.065,
            "focal_length": 1000,
            "hole_fill_quality": "fast",
            "keep_intermediates": False,
        }
        mock_pool = Mock()
        mock_pool.__enter__ = Mock(return_value=mock_pool)
        mock_pool.__exit__ = Mock(return_value=False)
        mock_pool.imap = Mock(
            return_value=iter(
                [
                    (temp_frames["frames"][i], temp_frames["frames"][i], f"frame_{i:04d}")
                    for i in range(3)
                ]
            )
        )

        with patch("multiprocessing.Pool", return_value=mock_pool):
            result = generator.create_stereo_pairs(
                temp_frames["frames"],
                temp_frames["depth_maps"],
                temp_frames["frame_files"],
                temp_frames["directories"],
                settings,
                progress_tracker=None,
            )

        assert result is True

    def test_create_stereo_pairs_caps_worker_count(self, temp_frames, mock_progress_tracker):
        """High-core Windows hosts do not spawn enough workers to exhaust virtual memory."""
        generator = StereoPairGenerator()
        settings = {
            "baseline": 0.065,
            "focal_length": 1000,
            "hole_fill_quality": "fast",
            "keep_intermediates": False,
        }
        mock_pool = Mock()
        mock_pool.__enter__ = Mock(return_value=mock_pool)
        mock_pool.__exit__ = Mock(return_value=False)
        mock_pool.imap = Mock(return_value=iter([]))

        with (
            patch("multiprocessing.cpu_count", return_value=32),
            patch("multiprocessing.Pool", return_value=mock_pool) as pool_factory,
        ):
            result = generator.create_stereo_pairs(
                temp_frames["frames"],
                temp_frames["depth_maps"],
                temp_frames["frame_files"],
                temp_frames["directories"],
                settings,
                mock_progress_tracker,
            )

        assert result is True
        pool_factory.assert_called_once_with(processes=4)

    def test_create_stereo_pairs_exception_handling(self, temp_frames, mock_progress_tracker):
        """Test exception handling during stereo pair creation."""
        generator = StereoPairGenerator()

        settings = {
            "baseline": 0.065,
            "focal_length": 1000,
            "hole_fill_quality": "fast",
            "keep_intermediates": True,
        }

        with patch(
            "src.depth_surge_3d.processing.frames.stereo_generator._process_single_stereo_pair",
            side_effect=RuntimeError("Test error"),
        ):
            result = generator.create_stereo_pairs(
                temp_frames["frames"],
                temp_frames["depth_maps"],
                temp_frames["frame_files"],
                temp_frames["directories"],
                settings,
                mock_progress_tracker,
            )

        assert result is False

    def test_create_stereo_pairs_from_files_saves_when_retention_is_disabled(
        self, tmp_path, mock_progress_tracker
    ):
        """Temporary stereo files are always written for downstream stages."""
        frame_dir = tmp_path / "frames"
        depth_dir = tmp_path / "depth"
        left_dir = tmp_path / "left"
        right_dir = tmp_path / "right"
        for directory in (frame_dir, depth_dir, left_dir, right_dir):
            directory.mkdir()
        frame_files = []
        depth_files = []
        for i in range(2):
            frame_path = frame_dir / f"frame_{i:04d}.png"
            depth_path = depth_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(frame_path), np.full((8, 8, 3), 127, dtype=np.uint8))
            cv2.imwrite(str(depth_path), np.full((8, 8), 32768, dtype=np.uint16))
            frame_files.append(frame_path)
            depth_files.append(depth_path)
        _write_canonical_metadata(depth_dir, [path.name for path in frame_files])
        settings = {
            "baseline": 0.01,
            "focal_length": 10,
            "hole_fill_quality": "none",
            "keep_intermediates": False,
        }
        pool = Mock()
        pool.__enter__ = Mock(return_value=pool)
        pool.__exit__ = Mock(return_value=False)
        pool.imap.side_effect = lambda function, args: map(function, args)

        with patch("multiprocessing.Pool", return_value=pool):
            result = StereoPairGenerator().create_stereo_pairs_from_files(
                frame_files,
                depth_files,
                {"left_frames": left_dir, "right_frames": right_dir},
                settings,
                mock_progress_tracker,
            )

        assert result is True
        assert len(list(left_dir.glob("*.png"))) == 2
        assert len(list(right_dir.glob("*.png"))) == 2

    def test_create_stereo_pairs_from_files_rejects_missing_canonical_metadata(self, tmp_path):
        frame_dir = tmp_path / "frames"
        depth_dir = tmp_path / "depth"
        left_dir = tmp_path / "left"
        right_dir = tmp_path / "right"
        for directory in (frame_dir, depth_dir, left_dir, right_dir):
            directory.mkdir()
        frame_path = frame_dir / "frame_0000.png"
        depth_path = depth_dir / "frame_0000.png"
        cv2.imwrite(str(frame_path), np.zeros((8, 8, 3), dtype=np.uint8))
        cv2.imwrite(str(depth_path), np.full((8, 8), 32768, dtype=np.uint16))

        result = StereoPairGenerator().create_stereo_pairs_from_files(
            [frame_path],
            [depth_path],
            {"left_frames": left_dir, "right_frames": right_dir},
            {"stereo_strength": 2.0, "convergence": 0.5},
        )

        assert result is False

    def test_canonical_metadata_requires_source_fingerprints(self, tmp_path):
        frame_path = tmp_path / "frame_0000.png"
        depth_path = tmp_path / "depth_0000.png"
        frame_path.write_bytes(b"frame")
        depth_path.write_bytes(b"depth")
        _write_canonical_metadata(tmp_path, [frame_path.name])
        metadata_path = tmp_path / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.pop("source_raw_fingerprint")
        metadata.pop("fingerprint")
        metadata["fingerprint"] = canonical_json_hash(metadata)
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        with pytest.raises(ValueError, match="does not match"):
            StereoPairGenerator._get_canonical_metadata([depth_path], [frame_path])

    def test_canonical_metadata_rejects_depth_file_from_another_directory(self, tmp_path):
        canonical_dir = tmp_path / "canonical"
        foreign_dir = tmp_path / "foreign"
        canonical_dir.mkdir()
        foreign_dir.mkdir()
        frame_files = [tmp_path / "frame_0000.png", tmp_path / "frame_0001.png"]
        depth_files = [
            canonical_dir / "frame_0000.png",
            foreign_dir / "frame_0001.png",
        ]
        for path in depth_files:
            cv2.imwrite(str(path), np.zeros((8, 8), dtype=np.uint16))
        _write_canonical_metadata(canonical_dir, [path.name for path in frame_files])

        with pytest.raises(ValueError, match="path manifest"):
            StereoPairGenerator._get_canonical_metadata(depth_files, frame_files)

    def test_wrong_native_shape_is_rejected_before_workers_start(self, tmp_path):
        frame_dir = tmp_path / "frames"
        depth_dir = tmp_path / "depth"
        left_dir = tmp_path / "left"
        right_dir = tmp_path / "right"
        for directory in (frame_dir, depth_dir, left_dir, right_dir):
            directory.mkdir()
        frame_path = frame_dir / "frame_0000.png"
        depth_path = depth_dir / "frame_0000.png"
        cv2.imwrite(str(frame_path), np.zeros((8, 8, 3), dtype=np.uint8))
        cv2.imwrite(str(depth_path), np.zeros((4, 4), dtype=np.uint16))
        _write_canonical_metadata(depth_dir, [frame_path.name], shape=(8, 8))

        with patch("multiprocessing.Pool") as pool:
            result = StereoPairGenerator().create_stereo_pairs_from_files(
                [frame_path],
                [depth_path],
                {"left_frames": left_dir, "right_frames": right_dir},
                {"stereo_strength": 2.0, "convergence": 0.5},
            )

        assert result is False
        pool.assert_not_called()
