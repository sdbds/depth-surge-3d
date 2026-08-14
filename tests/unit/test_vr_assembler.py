"""Tests for VRFrameAssembler module."""

import threading
import time

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch

from src.depth_surge_3d.processing.frames.vr_assembler import VRFrameAssembler


def _write_source_pair(left_dir, right_dir, index):
    image = np.full((4, 6, 3), index, dtype=np.uint8)
    assert cv2.imwrite(str(left_dir / f"frame_{index:06d}.png"), image)
    assert cv2.imwrite(str(right_dir / f"frame_{index:06d}.png"), image)


class TestResolveVRSourceFiles:
    def test_resolve_vr_source_files_uses_cropped_sources(self, tmp_path):
        left = tmp_path / "left_cropped"
        right = tmp_path / "right_cropped"
        left.mkdir()
        right.mkdir()
        _write_source_pair(left, right, 1)

        result = VRFrameAssembler().resolve_vr_source_files(
            {"left_cropped": left, "right_cropped": right},
            {"upscale_model": "none"},
            total_frames=1,
        )

        assert result == ([left / "frame_000001.png"], [right / "frame_000001.png"])

    def test_resolve_vr_source_files_requires_upscaled_sources(self, tmp_path):
        cropped_left = tmp_path / "left_cropped"
        cropped_right = tmp_path / "right_cropped"
        cropped_left.mkdir()
        cropped_right.mkdir()
        _write_source_pair(cropped_left, cropped_right, 1)

        assert VRFrameAssembler().resolve_vr_source_files(
            {"left_cropped": cropped_left, "right_cropped": cropped_right},
            {"upscale_model": "x2"},
            total_frames=1,
        ) is None

    def test_resolve_vr_source_files_rejects_unequal_counts(self, tmp_path):
        left = tmp_path / "left_cropped"
        right = tmp_path / "right_cropped"
        left.mkdir()
        right.mkdir()
        _write_source_pair(left, right, 1)
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        assert cv2.imwrite(str(left / "frame_000002.png"), image)

        assert VRFrameAssembler().resolve_vr_source_files(
            {"left_cropped": left, "right_cropped": right},
            {},
            total_frames=0,
        ) is None

    def test_resolve_vr_source_files_requires_total_frame_count(self, tmp_path):
        left = tmp_path / "left_cropped"
        right = tmp_path / "right_cropped"
        left.mkdir()
        right.mkdir()
        _write_source_pair(left, right, 1)

        assert VRFrameAssembler().resolve_vr_source_files(
            {"left_cropped": left, "right_cropped": right},
            {},
            total_frames=2,
        ) is None

    def test_resolve_vr_source_files_requires_matching_stems(self, tmp_path):
        left = tmp_path / "left_cropped"
        right = tmp_path / "right_cropped"
        left.mkdir()
        right.mkdir()
        _write_source_pair(left, right, 1)
        (right / "frame_000001.png").rename(right / "other_000001.png")

        assert VRFrameAssembler().resolve_vr_source_files(
            {"left_cropped": left, "right_cropped": right},
            {},
            total_frames=1,
        ) is None


class TestVRFrameAssemblerInit:
    """Test VRFrameAssembler initialization."""

    def test_init_default(self):
        """Test default initialization."""
        assembler = VRFrameAssembler()
        assert assembler.verbose is False

    def test_init_verbose(self):
        """Test initialization with verbose."""
        assembler = VRFrameAssembler(verbose=True)
        assert assembler.verbose is True


class TestAssembleVRFrames:
    """Test assemble_vr_frames method."""

    @pytest.fixture
    def mock_progress_tracker(self):
        """Create mock progress tracker."""
        tracker = Mock()
        tracker.update_progress = Mock()
        tracker.send_preview_frame = Mock()
        return tracker

    @pytest.fixture
    def temp_frames(self, tmp_path):
        """Create temporary frame directories."""
        left_dir = tmp_path / "left_cropped"
        right_dir = tmp_path / "right_cropped"
        vr_dir = tmp_path / "vr_frames"

        left_dir.mkdir()
        right_dir.mkdir()
        vr_dir.mkdir()

        # Create test frames
        for i in range(3):
            frame = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
            cv2.imwrite(str(left_dir / f"frame_{i:04d}.png"), frame)
            cv2.imwrite(str(right_dir / f"frame_{i:04d}.png"), frame)

        return {
            "left_cropped": left_dir,
            "right_cropped": right_dir,
            "vr_frames": vr_dir,
            "base": tmp_path,
        }

    def test_assemble_vr_frames_side_by_side(self, temp_frames, mock_progress_tracker):
        """Test side-by-side VR frame assembly."""
        assembler = VRFrameAssembler()

        settings = {
            "vr_format": "side_by_side",
            "keep_intermediates": True,
            "per_eye_width": 200,
            "per_eye_height": 100,
        }

        with patch(
            "src.depth_surge_3d.processing.frames.vr_assembler.create_vr_frame"
        ) as mock_create:
            mock_create.return_value = np.zeros((100, 400, 3), dtype=np.uint8)

            result = assembler.assemble_vr_frames(temp_frames, settings, mock_progress_tracker)
            resumed = assembler.assemble_vr_frames(temp_frames, settings, mock_progress_tracker)

        assert result is True
        assert resumed is True
        assert mock_create.call_count == 3

        # Check output files
        vr_files = list(temp_frames["vr_frames"].glob("*.png"))
        assert len(vr_files) == 3

    def test_assemble_vr_frames_over_under(self, temp_frames, mock_progress_tracker):
        """Test over-under VR frame assembly."""
        assembler = VRFrameAssembler()

        settings = {
            "vr_format": "over_under",
            "keep_intermediates": True,
            "per_eye_width": 200,
            "per_eye_height": 100,
        }

        with patch(
            "src.depth_surge_3d.processing.frames.vr_assembler.create_vr_frame"
        ) as mock_create:
            mock_create.return_value = np.zeros((200, 200, 3), dtype=np.uint8)

            result = assembler.assemble_vr_frames(temp_frames, settings, mock_progress_tracker)

        assert result is True

    def test_assemble_vr_frames_without_intermediates(self, temp_frames, mock_progress_tracker):
        """Test VR assembly without saving intermediates."""
        assembler = VRFrameAssembler()

        settings = {
            "vr_format": "side_by_side",
            "keep_intermediates": False,
            "per_eye_width": 200,
            "per_eye_height": 100,
        }

        with patch(
            "src.depth_surge_3d.processing.frames.vr_assembler.create_vr_frame"
        ) as mock_create:
            mock_create.return_value = np.zeros((100, 400, 3), dtype=np.uint8)

            result = assembler.assemble_vr_frames(temp_frames, settings, mock_progress_tracker)

        assert result is True

        # VR frames are still saved even with keep_intermediates=False
        # because the VR frames directory is in temp_frames dict
        vr_files = list(temp_frames["vr_frames"].glob("*.png"))
        assert len(vr_files) == 3

    def test_assemble_vr_frames_without_progress_tracker(self, temp_frames):
        """CLI assembly succeeds without a web progress tracker."""
        assembler = VRFrameAssembler()
        settings = {
            "vr_format": "side_by_side",
            "keep_intermediates": True,
            "per_eye_width": 200,
            "per_eye_height": 100,
        }

        result = assembler.assemble_vr_frames(
            temp_frames,
            settings,
            progress_tracker=None,
        )

        assert result is True

    def test_matching_source_dimensions_skip_resize(self, temp_frames):
        assembler = VRFrameAssembler()
        settings = {
            "vr_format": "side_by_side",
            "keep_intermediates": True,
            "per_eye_width": 200,
            "per_eye_height": 100,
        }

        with patch(
            "src.depth_surge_3d.processing.frames.vr_assembler.resize_image",
            side_effect=AssertionError("matching dimensions must not resize"),
        ):
            result = assembler.assemble_vr_frames(
                temp_frames,
                settings,
                progress_tracker=None,
            )

        assert result is True

    def test_assemble_vr_frames_processes_pairs_concurrently(self, temp_frames):
        assembler = VRFrameAssembler()
        real_assemble = assembler._assemble_single_vr_frame
        lock = threading.Lock()
        active = 0
        max_active = 0

        def observed_assemble(*args, **kwargs):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(0.03)
                return real_assemble(*args, **kwargs)
            finally:
                with lock:
                    active -= 1

        with (
            patch(
                "src.depth_surge_3d.processing.frames.vr_assembler."
                "calculate_frame_stage_workers",
                return_value=2,
            ),
            patch.object(
                assembler,
                "_assemble_single_vr_frame",
                side_effect=observed_assemble,
            ),
        ):
            result = assembler.assemble_vr_frames(
                temp_frames,
                {
                    "vr_format": "side_by_side",
                    "per_eye_width": 200,
                    "per_eye_height": 100,
                },
                progress_tracker=None,
                total_frames=3,
            )

        assert result is True
        assert max_active >= 2

    def test_vr_worker_memory_uses_larger_source_dimensions(self, temp_frames):
        with patch(
            "src.depth_surge_3d.processing.frames.vr_assembler." "calculate_frame_stage_workers",
            return_value=1,
        ) as calculate:
            result = VRFrameAssembler().assemble_vr_frames(
                temp_frames,
                {
                    "vr_format": "side_by_side",
                    "per_eye_width": 20,
                    "per_eye_height": 10,
                },
                progress_tracker=None,
                total_frames=3,
            )

        assert result is True
        calculate.assert_called_once_with(3, 200 * 100 * 48)

    @pytest.mark.parametrize(
        ("vr_format", "expected_shape"),
        [("side_by_side", (100, 400, 3)), ("over_under", (200, 200, 3))],
    )
    def test_parallel_vr_layout_preserves_exact_pixels(
        self, temp_frames, vr_format, expected_shape
    ):
        result = VRFrameAssembler().assemble_vr_frames(
            temp_frames,
            {
                "vr_format": vr_format,
                "per_eye_width": 200,
                "per_eye_height": 100,
            },
            progress_tracker=None,
            total_frames=3,
        )

        assert result is True
        left = cv2.imread(str(temp_frames["left_cropped"] / "frame_0000.png"))
        right = cv2.imread(str(temp_frames["right_cropped"] / "frame_0000.png"))
        expected = (
            np.hstack((left, right)) if vr_format == "side_by_side" else np.vstack((left, right))
        )
        actual = cv2.imread(str(temp_frames["vr_frames"] / "frame_0000.png"))
        assert actual.shape == expected_shape
        np.testing.assert_array_equal(actual, expected)

    def test_vr_callbacks_remain_ordered_on_the_caller_thread(self, temp_frames):
        caller_thread = threading.get_ident()
        tracker = Mock()
        tracker.previews = []
        tracker.progress = []
        tracker.send_preview_frame.side_effect = (
            lambda path, phase, frame_num: tracker.previews.append(
                (path.name, phase, frame_num, threading.get_ident())
            )
        )
        tracker.update_progress.side_effect = lambda *_args, **kwargs: tracker.progress.append(
            (kwargs["frame_num"], threading.get_ident())
        )
        assembler = VRFrameAssembler()
        real_assemble = assembler._assemble_single_vr_frame

        def delayed_assemble(left_file, *args, **kwargs):
            index = int(left_file.stem.rsplit("_", 1)[1])
            time.sleep((3 - index) * 0.02)
            return real_assemble(left_file, *args, **kwargs)

        with (
            patch(
                "src.depth_surge_3d.processing.frames.vr_assembler."
                "calculate_frame_stage_workers",
                return_value=3,
            ),
            patch.object(
                assembler,
                "_assemble_single_vr_frame",
                side_effect=delayed_assemble,
            ),
        ):
            result = assembler.assemble_vr_frames(
                temp_frames,
                {
                    "vr_format": "side_by_side",
                    "per_eye_width": 200,
                    "per_eye_height": 100,
                },
                tracker,
                total_frames=3,
            )

        assert result is True
        assert [item[2] for item in tracker.previews] == [1, 2, 3]
        assert [item[0] for item in tracker.progress] == [1, 2, 3]
        assert {item[3] for item in tracker.previews} == {caller_thread}
        assert {item[1] for item in tracker.progress} == {caller_thread}

    def test_vr_failure_waits_for_running_workers_and_cancels_pending(self, tmp_path):
        left_dir = tmp_path / "left_cropped"
        right_dir = tmp_path / "right_cropped"
        vr_dir = tmp_path / "vr_frames"
        for directory in (left_dir, right_dir, vr_dir):
            directory.mkdir()
        for index in range(12):
            image = np.full((8, 8, 3), index, dtype=np.uint8)
            assert cv2.imwrite(str(left_dir / f"frame_{index:04d}.png"), image)
            assert cv2.imwrite(str(right_dir / f"frame_{index:04d}.png"), image)

        first_started = threading.Event()
        failure_raised = threading.Event()
        release_worker = threading.Event()
        executed = []
        lock = threading.Lock()
        result = {}
        assembler = VRFrameAssembler()

        def controlled_assemble(left_file, *_args, **_kwargs):
            index = int(left_file.stem.rsplit("_", 1)[1])
            with lock:
                executed.append(index)
            if index == 0:
                first_started.set()
                assert release_worker.wait(5)
                return vr_dir / left_file.name
            if index == 1:
                assert first_started.wait(2)
                failure_raised.set()
                raise RuntimeError("planned VR write failure")
            return vr_dir / left_file.name

        def run_stage():
            result["value"] = assembler.assemble_vr_frames(
                {
                    "left_cropped": left_dir,
                    "right_cropped": right_dir,
                    "vr_frames": vr_dir,
                },
                {
                    "vr_format": "side_by_side",
                    "per_eye_width": 8,
                    "per_eye_height": 8,
                },
                total_frames=12,
            )

        with (
            patch(
                "src.depth_surge_3d.processing.frames.vr_assembler."
                "calculate_frame_stage_workers",
                return_value=2,
            ),
            patch.object(
                assembler,
                "_assemble_single_vr_frame",
                side_effect=controlled_assemble,
            ),
        ):
            stage_thread = threading.Thread(target=run_stage)
            stage_thread.start()
            try:
                assert failure_raised.wait(2)
                time.sleep(0.05)
                assert stage_thread.is_alive()
            finally:
                release_worker.set()
                stage_thread.join(timeout=5)

        assert not stage_thread.is_alive()
        assert result["value"] is False
        assert len(executed) == 2
        assert set(executed) == {0, 1}
        assert not (vr_dir / "metadata.json").exists()

    def test_assemble_vr_frames_mismatched_count(self, temp_frames, mock_progress_tracker):
        """Test VR assembly with mismatched frame counts."""
        assembler = VRFrameAssembler()

        # Remove one right frame
        right_frames = list(temp_frames["right_cropped"].glob("*.png"))
        right_frames[0].unlink()

        settings = {
            "vr_format": "side_by_side",
            "keep_intermediates": True,
            "per_eye_width": 200,
            "per_eye_height": 100,
        }

        with patch(
            "src.depth_surge_3d.processing.frames.vr_assembler.create_vr_frame"
        ) as mock_create:
            mock_create.return_value = np.zeros((100, 400, 3), dtype=np.uint8)

            result = assembler.assemble_vr_frames(temp_frames, settings, mock_progress_tracker)

        assert result is False
        assert mock_create.call_count == 0

    def test_assemble_vr_frames_no_source(self, mock_progress_tracker, tmp_path):
        """Test VR assembly with no source frames."""
        assembler = VRFrameAssembler()

        directories = {"base": tmp_path}
        settings = {"vr_format": "side_by_side", "keep_intermediates": True}

        result = assembler.assemble_vr_frames(directories, settings, mock_progress_tracker)

        assert result is False

    def test_assemble_vr_frames_exception_handling(self, temp_frames, mock_progress_tracker):
        """Test exception handling during VR assembly."""
        assembler = VRFrameAssembler()

        settings = {
            "vr_format": "side_by_side",
            "keep_intermediates": True,
        }

        with patch(
            "src.depth_surge_3d.processing.frames.vr_assembler.create_vr_frame",
            side_effect=RuntimeError("Test error"),
        ):
            result = assembler.assemble_vr_frames(temp_frames, settings, mock_progress_tracker)

        assert result is False


class TestGetVRAssemblySourceDirs:
    """Test _get_vr_assembly_source_dirs static method."""

    def test_get_source_with_upscaled(self, tmp_path):
        """Test getting source with upscaled frames."""
        left_upscaled = tmp_path / "left_upscaled"
        right_upscaled = tmp_path / "right_upscaled"
        left_upscaled.mkdir()
        right_upscaled.mkdir()

        # Create frame
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(left_upscaled / "frame_0000.png"), frame)

        directories = {
            "left_upscaled": left_upscaled,
            "right_upscaled": right_upscaled,
        }

        # Must set upscale_model to something other than "none" for upscaled frames to be used
        settings = {"upscale_model": "x4"}

        result = VRFrameAssembler._get_vr_assembly_source_dirs(directories, settings)

        assert result == (left_upscaled, right_upscaled)

    def test_get_source_with_cropped(self, tmp_path):
        """Test getting source with cropped frames."""
        left_cropped = tmp_path / "left_cropped"
        right_cropped = tmp_path / "right_cropped"
        left_cropped.mkdir()
        right_cropped.mkdir()

        # Create frame
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(left_cropped / "frame_0000.png"), frame)

        directories = {
            "left_cropped": left_cropped,
            "right_cropped": right_cropped,
        }

        settings = {}

        result = VRFrameAssembler._get_vr_assembly_source_dirs(directories, settings)

        assert result == (left_cropped, right_cropped)

    def test_get_source_does_not_fallback_when_upscaling_is_required(self, tmp_path):
        left_cropped = tmp_path / "left_cropped"
        right_cropped = tmp_path / "right_cropped"
        left_cropped.mkdir()
        right_cropped.mkdir()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(left_cropped / "frame_0000.png"), frame)
        cv2.imwrite(str(right_cropped / "frame_0000.png"), frame)

        result = VRFrameAssembler._get_vr_assembly_source_dirs(
            {"left_cropped": left_cropped, "right_cropped": right_cropped},
            {"upscale_model": "x4"},
        )

        assert result is None

    def test_get_source_no_directories(self):
        """Test handling when no source directories exist."""
        directories = {}
        settings = {}

        result = VRFrameAssembler._get_vr_assembly_source_dirs(directories, settings)

        assert result is None
