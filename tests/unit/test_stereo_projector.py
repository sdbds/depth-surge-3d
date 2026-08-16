"""Unit tests for StereoProjector."""

import inspect
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.depth_surge_3d.inference.depth.types import DepthBatch, DepthRepresentation
from src.depth_surge_3d.inference.depth.backend_registry import EstimatorRequest
from src.depth_surge_3d.rendering import stereo_projector
from src.depth_surge_3d.rendering import (
    StereoRenderResult,
    StereoRenderSettings,
    StereoProjector,
    create_stereo_projector,
)


def _stereo_result(image: np.ndarray) -> StereoRenderResult:
    valid = np.ones(image.shape[:2], dtype=np.bool_)
    holes = np.zeros(image.shape[:2], dtype=np.bool_)
    return StereoRenderResult(
        left_image=image.copy(),
        right_image=image.copy(),
        left_valid_mask=valid.copy(),
        right_valid_mask=valid.copy(),
        left_hole_mask=holes.copy(),
        right_hole_mask=holes.copy(),
    )


def test_projector_rejects_unknown_backend_without_constructing_an_estimator(monkeypatch) -> None:
    """Invalid backend IDs fail before the registry factory can create anything."""
    factory = Mock()
    monkeypatch.setattr(stereo_projector, "create_registered_depth_estimator", factory)
    with pytest.raises(ValueError, match="Unknown depth backend: typo"):
        create_stereo_projector(depth_model_version="typo")
    factory.assert_not_called()


def test_projector_preserves_four_positional_arguments_and_forwards_model_size(monkeypatch) -> None:
    factory = Mock(return_value=MagicMock())
    monkeypatch.setattr(stereo_projector, "create_registered_depth_estimator", factory)

    projector = create_stereo_projector(
        None,
        "cpu",
        True,
        "moge2",
        model_size="vitb",
    )

    assert projector.model_size == "vitb"
    factory.assert_called_once_with("moge2", EstimatorRequest(None, "vitb", "cpu", True, 10))


class TestStereoProjector:
    """Test StereoProjector class."""

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    def test_init_with_v2_default(self, mock_create_v2):
        """Test initialization with V2 (default)."""
        mock_estimator = MagicMock()
        mock_create_v2.return_value = mock_estimator

        projector = StereoProjector(
            model_path="models/test.pth",
            device="cpu",
            metric=False,
            depth_model_version="v2",
        )

        assert projector.depth_model_version == "v2"
        assert projector.depth_estimator == mock_estimator
        assert projector._model_loaded is False
        mock_create_v2.assert_called_once_with(
            "v2", EstimatorRequest("models/test.pth", None, "cpu", False, 10)
        )

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    def test_init_with_v3(self, mock_create_v3):
        """Test initialization with V3."""
        mock_estimator = MagicMock()
        mock_create_v3.return_value = mock_estimator

        projector = StereoProjector(
            model_path="large",
            device="cpu",
            metric=False,
            depth_model_version="v3",
        )

        assert projector.depth_model_version == "v3"
        assert projector.depth_estimator == mock_estimator
        mock_create_v3.assert_called_once_with(
            "v3", EstimatorRequest("large", None, "cpu", False, 10)
        )

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    def test_init_with_see_through(self, mock_create_see_through):
        """Test initialization with the anime-focused See-Through model."""
        mock_estimator = MagicMock()
        mock_create_see_through.return_value = mock_estimator

        projector = StereoProjector(
            model_path="24yearsold/custom-marigold",
            device="cuda",
            metric=True,
            depth_model_version="see_through",
        )

        assert projector.depth_model_version == "see_through"
        assert projector.depth_estimator == mock_estimator
        mock_create_see_through.assert_called_once_with(
            "see_through",
            EstimatorRequest("24yearsold/custom-marigold", None, "cuda", True, 10),
        )

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    def test_init_with_none_model_path_v2(self, mock_create_v2):
        """Test initialization with None model path for V2."""
        mock_estimator = MagicMock()
        mock_create_v2.return_value = mock_estimator

        projector = StereoProjector(
            model_path=None,
            device="cpu",
            depth_model_version="v2",
        )

        assert projector.depth_model_version == "v2"
        mock_create_v2.assert_called_once_with("v2", EstimatorRequest(None, None, "cpu", False, 10))

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    def test_init_with_none_model_path_v3(self, mock_create_v3):
        """Test initialization with None model path for V3."""
        mock_estimator = MagicMock()
        mock_create_v3.return_value = mock_estimator

        projector = StereoProjector(
            model_path=None,
            device="cpu",
            depth_model_version="v3",
        )

        assert projector.depth_model_version == "v3"
        mock_create_v3.assert_called_once_with("v3", EstimatorRequest(None, None, "cpu", False, 10))

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    def test_apply_default_settings(self, mock_create_v2):
        """Test default settings application."""
        mock_estimator = MagicMock()
        mock_create_v2.return_value = mock_estimator

        projector = StereoProjector(device="cpu")

        # Create a locals dict similar to what process_video receives
        test_locals = {
            "self": projector,
            "video_path": "test.mp4",
            "output_dir": "output",
            "vr_format": None,
            "stereo_strength": 3.0,
            "scene_detection": False,
        }

        settings = projector._apply_default_settings(test_locals)

        # Should have defaults applied
        assert "vr_format" in settings
        assert "baseline" not in settings
        assert settings["stereo_strength"] == 3.0
        assert settings["scene_detection"] is False
        assert settings["video_path"] == "test.mp4"
        assert settings["output_dir"] == "output"

    def test_process_video_signature_uses_final_controls(self):
        parameter_names = list(inspect.signature(StereoProjector.process_video).parameters)

        assert parameter_names == ["self", "video_path", "output_dir", "settings"]

    def test_duplicate_pipeline_apis_are_removed(self):
        assert {
            "extract_frames",
            "determine_super_sample_resolution",
            "determine_vr_output_resolution",
            "create_output_video",
            "_check_nvenc_available",
            "_add_video_encoder_options",
        }.isdisjoint(vars(StereoProjector))


class TestCreateStereoProjector:
    """Test factory function for StereoProjector."""

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    def test_create_with_defaults(self, mock_create_v2):
        """Test factory function with defaults."""
        mock_estimator = MagicMock()
        mock_create_v2.return_value = mock_estimator

        projector = create_stereo_projector()

        assert isinstance(projector, StereoProjector)
        mock_create_v2.assert_called_once_with(
            "v2", EstimatorRequest(None, None, "auto", False, 10)
        )

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    def test_create_with_v2(self, mock_create_v2):
        """Test factory function with V2."""
        mock_estimator = MagicMock()
        mock_create_v2.return_value = mock_estimator

        projector = create_stereo_projector(
            model_path="models/test.pth",
            device="cpu",
            metric=True,
            depth_model_version="v2",
        )

        assert isinstance(projector, StereoProjector)
        assert projector.depth_model_version == "v2"
        mock_create_v2.assert_called_once_with(
            "v2", EstimatorRequest("models/test.pth", None, "cpu", True, 10)
        )

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    def test_create_with_v3(self, mock_create_v3):
        """Test factory function with V3."""
        mock_estimator = MagicMock()
        mock_create_v3.return_value = mock_estimator

        projector = create_stereo_projector(
            model_path="large",
            device="cuda",
            metric=False,
            depth_model_version="v3",
        )

        assert isinstance(projector, StereoProjector)
        assert projector.depth_model_version == "v3"
        mock_create_v3.assert_called_once_with(
            "v3", EstimatorRequest("large", None, "cuda", False, 10)
        )


class TestStereoProjectorHelpers:
    """Test helper methods of StereoProjector."""

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    def test_ensure_model_loaded_success(self, mock_create):
        """Test model loading success."""
        mock_estimator = MagicMock()
        mock_estimator.load_model.return_value = True
        mock_create.return_value = mock_estimator

        projector = StereoProjector(device="cpu")
        assert projector._model_loaded is False

        result = projector._ensure_model_loaded()

        assert result is True
        assert projector._model_loaded is True
        mock_estimator.load_model.assert_called_once()

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    def test_ensure_model_loaded_already_loaded(self, mock_create):
        """Test model loading when already loaded."""
        mock_estimator = MagicMock()
        mock_create.return_value = mock_estimator

        projector = StereoProjector(device="cpu")
        projector._model_loaded = True

        result = projector._ensure_model_loaded()

        assert result is True
        # Should not call load_model again
        mock_estimator.load_model.assert_not_called()

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    def test_ensure_model_loaded_failure(self, mock_create):
        """Test model loading failure."""
        mock_estimator = MagicMock()
        mock_estimator.load_model.return_value = False
        mock_create.return_value = mock_estimator

        projector = StereoProjector(device="cpu")

        result = projector._ensure_model_loaded()

        assert result is False
        assert projector._model_loaded is False

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    @patch("src.depth_surge_3d.rendering.stereo_projector.validate_video_file")
    def test_validate_inputs_valid(self, mock_validate, mock_create):
        """Test input validation with valid inputs."""
        mock_create.return_value = MagicMock()
        mock_validate.return_value = True

        projector = StereoProjector(device="cpu")

        with patch("pathlib.Path.mkdir"):
            result = projector._validate_inputs("video.mp4", "/tmp/output", {})

        assert result is True
        mock_validate.assert_called_once_with("video.mp4")

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    @patch("src.depth_surge_3d.rendering.stereo_projector.validate_video_file")
    def test_validate_inputs_invalid_video(self, mock_validate, mock_create):
        """Test input validation with invalid video."""
        mock_create.return_value = MagicMock()
        mock_validate.return_value = False

        projector = StereoProjector(device="cpu")

        result = projector._validate_inputs("invalid.txt", "/tmp/output", {})

        assert result is False


class TestResolveSettings:
    """Test _resolve_settings method."""

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    @patch("src.depth_surge_3d.rendering.stereo_projector.auto_detect_resolution")
    @patch("src.depth_surge_3d.rendering.stereo_projector.validate_resolution_settings")
    @patch("src.depth_surge_3d.rendering.stereo_projector.get_resolution_dimensions")
    @patch("src.depth_surge_3d.rendering.stereo_projector.calculate_vr_output_dimensions")
    def test_resolve_settings_with_auto_resolution(
        self,
        mock_calc_vr,
        mock_get_dims,
        mock_validate,
        mock_auto_detect,
        mock_create,
    ):
        """Test settings resolution with auto resolution detection."""
        mock_create.return_value = MagicMock()

        # Mock auto-detection
        mock_auto_detect.return_value = "16x9-1080p"

        # Mock validation
        mock_validate.return_value = {
            "valid": True,
            "warnings": [],
            "recommendations": ["Use this resolution"],
        }

        # Mock dimensions
        mock_get_dims.return_value = (1920, 1080)
        mock_calc_vr.return_value = (3840, 1080)

        projector = StereoProjector(device="cpu")

        settings = {
            "vr_resolution": "auto",
            "vr_format": "side_by_side",
        }
        video_props = {"width": 1920, "height": 1080, "fps": 30}

        resolved = projector._resolve_settings(settings, video_props)

        # Should have auto-detected resolution
        assert resolved["vr_resolution"] == "16x9-1080p"
        assert resolved["per_eye_width"] == 1920
        assert resolved["per_eye_height"] == 1080
        assert resolved["vr_output_width"] == 3840
        assert resolved["vr_output_height"] == 1080
        assert resolved["source_width"] == 1920
        assert resolved["source_height"] == 1080
        assert resolved["source_fps"] == 30

        mock_auto_detect.assert_called_once_with(1920, 1080, "side_by_side")

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    @patch("src.depth_surge_3d.rendering.stereo_projector.validate_resolution_settings")
    @patch("src.depth_surge_3d.rendering.stereo_projector.get_resolution_dimensions")
    @patch("src.depth_surge_3d.rendering.stereo_projector.calculate_vr_output_dimensions")
    def test_resolve_settings_with_manual_resolution(
        self,
        mock_calc_vr,
        mock_get_dims,
        mock_validate,
        mock_create,
    ):
        """Test settings resolution with manual resolution."""
        mock_create.return_value = MagicMock()

        # Mock validation
        mock_validate.return_value = {
            "valid": True,
            "warnings": [],
            "recommendations": [],
        }

        # Mock dimensions
        mock_get_dims.return_value = (2048, 2048)
        mock_calc_vr.return_value = (4096, 2048)

        projector = StereoProjector(device="cpu")

        settings = {
            "vr_resolution": "square-2k",
            "vr_format": "side_by_side",
        }
        video_props = {"width": 1920, "height": 1080, "fps": 60}

        resolved = projector._resolve_settings(settings, video_props)

        # Should keep manual resolution
        assert resolved["vr_resolution"] == "square-2k"
        assert resolved["per_eye_width"] == 2048
        assert resolved["per_eye_height"] == 2048
        assert resolved["vr_output_width"] == 4096
        assert resolved["vr_output_height"] == 2048

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    @patch("src.depth_surge_3d.rendering.stereo_projector.auto_detect_resolution")
    @patch("src.depth_surge_3d.rendering.stereo_projector.validate_resolution_settings")
    @patch("src.depth_surge_3d.rendering.stereo_projector.get_resolution_dimensions")
    @patch("src.depth_surge_3d.rendering.stereo_projector.calculate_vr_output_dimensions")
    def test_resolve_settings_with_validation_warnings(
        self,
        mock_calc_vr,
        mock_get_dims,
        mock_validate,
        mock_auto_detect,
        mock_create,
    ):
        """Test settings resolution with validation warnings."""
        mock_create.return_value = MagicMock()

        mock_auto_detect.return_value = "cinema-4k"

        # Mock validation with warnings
        mock_validate.return_value = {
            "valid": False,
            "warnings": ["Resolution too high for source", "Consider downscaling"],
            "recommendations": ["Use 16x9-1080p instead"],
        }

        mock_get_dims.return_value = (4096, 2160)
        mock_calc_vr.return_value = (8192, 2160)

        projector = StereoProjector(device="cpu")

        settings = {
            "vr_resolution": "auto",
            "vr_format": "side_by_side",
        }
        video_props = {"width": 1280, "height": 720, "fps": 30}

        # Should not raise error even with invalid validation
        resolved = projector._resolve_settings(settings, video_props)

        assert resolved is not None
        assert "per_eye_width" in resolved


class TestModelDelegation:
    """Test model delegation methods."""

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    def test_get_model_info(self, mock_create):
        """Test get_model_info delegation."""
        mock_estimator = MagicMock()
        mock_estimator.get_model_info.return_value = {
            "loaded": True,
            "encoder": "vitl",
        }
        mock_create.return_value = mock_estimator

        projector = StereoProjector(device="cpu")
        info = projector.get_model_info()

        assert info["loaded"] is True
        assert info["encoder"] == "vitl"
        mock_estimator.get_model_info.assert_called_once()

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    def test_unload_model(self, mock_create):
        """Test unload_model delegation."""
        mock_estimator = MagicMock()
        mock_create.return_value = mock_estimator

        projector = StereoProjector(device="cpu")
        projector._model_loaded = True

        projector.unload_model()

        assert projector._model_loaded is False
        mock_estimator.unload_model.assert_called_once()


class TestProcessVideoErrorPaths:
    """Test error handling in process_video method."""

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    @patch("src.depth_surge_3d.rendering.stereo_projector.validate_video_file")
    def test_process_video_invalid_input(self, mock_validate, mock_create):
        """Test process_video with invalid video input."""
        mock_create.return_value = MagicMock()
        mock_validate.return_value = False

        projector = StereoProjector(device="cpu")
        result = projector.process_video("invalid.txt", "/tmp/output", {})

        assert result is False
        mock_validate.assert_called_once_with("invalid.txt")

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    @patch("src.depth_surge_3d.rendering.stereo_projector.validate_video_file")
    def test_process_video_model_load_failure(self, mock_validate, mock_create):
        """Test process_video when model fails to load."""
        mock_estimator = MagicMock()
        mock_estimator.load_model.return_value = False
        mock_create.return_value = mock_estimator
        mock_validate.return_value = True

        projector = StereoProjector(device="cpu")

        with patch("pathlib.Path.mkdir"):
            result = projector.process_video("test.mp4", "/tmp/output", {})

        assert result is False
        mock_estimator.load_model.assert_called_once()

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    @patch("src.depth_surge_3d.rendering.stereo_projector.validate_video_file")
    @patch("src.depth_surge_3d.rendering.stereo_projector.get_video_properties")
    def test_process_video_invalid_video_properties(
        self, mock_get_props, mock_validate, mock_create
    ):
        """Test process_video when video properties cannot be read."""
        mock_estimator = MagicMock()
        mock_estimator.load_model.return_value = True
        mock_create.return_value = mock_estimator
        mock_validate.return_value = True
        mock_get_props.return_value = None

        projector = StereoProjector(device="cpu")

        with patch("pathlib.Path.mkdir"):
            result = projector.process_video("test.mp4", "/tmp/output", {})

        assert result is False
        mock_get_props.assert_called_once_with("test.mp4")

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    @patch("src.depth_surge_3d.rendering.stereo_projector.validate_video_file")
    @patch("src.depth_surge_3d.io.operations.get_video_properties")
    def test_process_video_exception_handling(self, mock_get_props, mock_validate, mock_create):
        """Test process_video handles exceptions gracefully."""
        mock_estimator = MagicMock()
        mock_estimator.load_model.side_effect = RuntimeError("Model error")
        mock_create.return_value = mock_estimator
        mock_validate.return_value = True

        projector = StereoProjector(device="cpu")

        with patch("pathlib.Path.mkdir"):
            result = projector.process_video("test.mp4", "/tmp/output", {})

        assert result is False


class TestProcessImage:
    """Test process_image method."""

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    @patch("cv2.imread")
    @patch("cv2.imwrite")
    @patch("pathlib.Path.mkdir")
    def test_process_image_success(self, mock_mkdir, mock_imwrite, mock_imread, mock_create):
        """Test successful image processing."""
        mock_estimator = MagicMock()
        mock_estimator.load_model.return_value = True
        mock_estimator.estimate_depth_batch.return_value = DepthBatch(
            np.linspace(0.0, 1.0, 480 * 640, dtype=np.float32).reshape(1, 480, 640),
            DepthRepresentation.INVERSE_DEPTH,
        )
        mock_create.return_value = mock_estimator

        # Mock imread to return a valid image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_imread.return_value = image
        mock_imwrite.return_value = True
        renderer = MagicMock()
        renderer.render.return_value = _stereo_result(image)

        projector = StereoProjector(device="cpu", stereo_renderer=renderer)
        result = projector.process_image("test.jpg", "/tmp/output")

        assert result is True
        mock_estimator.load_model.assert_called_once()
        mock_estimator.estimate_depth_batch.assert_called_once()
        render_image, canonical, render_settings = renderer.render.call_args.args
        assert render_image is image
        assert canonical.dtype == np.float32
        assert canonical[0, 0] == 0.0
        assert canonical[-1, -1] == 1.0
        assert isinstance(render_settings, StereoRenderSettings)
        # Should save 4 images: left, right, vr, depth
        assert mock_imwrite.call_count == 4

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    @patch("cv2.imread")
    def test_process_image_model_load_failure(self, mock_imread, mock_create):
        """Test process_image when model fails to load."""
        mock_estimator = MagicMock()
        mock_estimator.load_model.return_value = False
        mock_create.return_value = mock_estimator

        projector = StereoProjector(device="cpu")
        result = projector.process_image("test.jpg", "/tmp/output")

        assert result is False

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    @patch("cv2.imread")
    def test_process_image_invalid_image(self, mock_imread, mock_create):
        """Test process_image with invalid image file."""
        mock_estimator = MagicMock()
        mock_estimator.load_model.return_value = True
        mock_create.return_value = mock_estimator

        mock_imread.return_value = None

        projector = StereoProjector(device="cpu")
        result = projector.process_image("invalid.jpg", "/tmp/output")

        assert result is False

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    @patch("cv2.imread")
    def test_process_image_depth_estimation_failure(self, mock_imread, mock_create):
        """Test process_image when depth estimation fails."""
        import numpy as np

        mock_estimator = MagicMock()
        mock_estimator.load_model.return_value = True
        mock_estimator.estimate_depth_batch.return_value = None
        mock_create.return_value = mock_estimator

        mock_imread.return_value = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        projector = StereoProjector(device="cpu")
        result = projector.process_image("test.jpg", "/tmp/output")

        assert result is False

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    @patch("cv2.imread")
    def test_process_image_with_custom_settings(self, mock_imread, mock_create):
        """Test process_image with custom settings."""
        mock_estimator = MagicMock()
        mock_estimator.load_model.return_value = True
        mock_estimator.estimate_depth_batch.return_value = DepthBatch(
            np.linspace(0.0, 1.0, 480 * 640, dtype=np.float32).reshape(1, 480, 640),
            DepthRepresentation.RELATIVE_DEPTH,
        )
        mock_create.return_value = mock_estimator

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_imread.return_value = image
        renderer = MagicMock()
        renderer.render.return_value = _stereo_result(image)

        projector = StereoProjector(device="cpu", stereo_renderer=renderer)

        with patch("cv2.imwrite", return_value=True):
            with patch("pathlib.Path.mkdir"):
                result = projector.process_image(
                    "test.jpg",
                    "/tmp/output",
                    stereo_strength=4.0,
                    convergence=0.25,
                    occlusion_fill="none",
                    depth_resolution="720",
                )

        assert result is True
        # Verify custom depth resolution was used (batch estimation called)
        mock_estimator.estimate_depth_batch.assert_called_once()
        canonical = renderer.render.call_args.args[1]
        render_settings = renderer.render.call_args.args[2]
        assert canonical[0, 0] == 1.0
        assert canonical[-1, -1] == 0.0
        assert render_settings == StereoRenderSettings(
            stereo_strength=4.0,
            convergence=0.25,
            occlusion_fill="none",
        )

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    @patch("cv2.imread")
    def test_process_image_exception_handling(self, mock_imread, mock_create):
        """Test process_image handles exceptions gracefully."""
        import numpy as np

        mock_estimator = MagicMock()
        mock_estimator.load_model.return_value = True
        mock_estimator.estimate_depth_batch.side_effect = RuntimeError("Processing error")
        mock_create.return_value = mock_estimator

        mock_imread.return_value = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        projector = StereoProjector(device="cpu")
        result = projector.process_image("test.jpg", "/tmp/output")

        assert result is False


class TestProcessVideoSuccessPath:
    """Test process_video successful path."""

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    @patch("src.depth_surge_3d.rendering.stereo_projector.validate_video_file")
    @patch("src.depth_surge_3d.rendering.stereo_projector.get_video_properties")
    @patch("src.depth_surge_3d.rendering.stereo_projector.VideoProcessor")
    @patch("pathlib.Path.mkdir")
    def test_process_video_success(
        self, mock_mkdir, mock_processor_class, mock_get_props, mock_validate, mock_create
    ):
        """Test successful video processing path."""
        mock_estimator = MagicMock()
        mock_estimator.load_model.return_value = True
        mock_estimator.get_model_size.return_value = "vitl"
        mock_estimator.device = "cpu"
        mock_estimator.metric = False
        mock_estimator.processing_resolution = 768
        mock_create.return_value = mock_estimator

        mock_validate.return_value = True
        mock_get_props.return_value = {"width": 1920, "height": 1080, "fps": 30}

        mock_processor = MagicMock()
        mock_processor.process.return_value = True
        mock_processor_class.return_value = mock_processor

        projector = StereoProjector(device="cpu")
        result = projector.process_video(
            "test.mp4",
            "/tmp/output",
            {"upscale_model": "x4", "verbose": True, "depth_resolution": "1080"},
        )

        assert result is True
        mock_processor.process.assert_called_once()
        passed_settings = mock_processor.process.call_args.kwargs["settings"]
        assert passed_settings["depth_model_version"] == "v2"
        assert passed_settings["model_size"] == "vitl"
        assert passed_settings["use_metric_depth"] is False
        assert passed_settings["device"] == "cpu"
        assert passed_settings["model_path"] is None
        assert passed_settings["upscale_model"] == "x4"
        assert passed_settings["verbose"] is True
        assert passed_settings["depth_resolution"] == "1080"
        mock_processor_class.assert_called_once_with(mock_estimator, verbose=True)


class TestValidateInputsDirectoryError:
    """Test validate_inputs with directory creation failure."""

    @patch("src.depth_surge_3d.rendering.stereo_projector.create_registered_depth_estimator")
    @patch("src.depth_surge_3d.rendering.stereo_projector.validate_video_file")
    @patch("pathlib.Path.mkdir")
    def test_validate_inputs_directory_creation_fails(self, mock_mkdir, mock_validate, mock_create):
        """Test validate_inputs when directory creation fails."""
        mock_create.return_value = MagicMock()
        mock_validate.return_value = True
        mock_mkdir.side_effect = PermissionError("Permission denied")

        projector = StereoProjector(device="cpu")
        result = projector._validate_inputs("test.mp4", "/root/forbidden", {})

        assert result is False
