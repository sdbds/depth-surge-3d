"""Unit tests for constants module."""

from src.depth_surge_3d.core.constants import (
    DA3_MODEL_NAMES,
    DEFAULT_DA3_MODEL,
    MODEL_CONFIGS,
    VR_HEADSET_PRESETS,
    VR_RESOLUTIONS,
    VALIDATION_RANGES,
    DEFAULT_SETTINGS,
)


class TestConstants:
    """Test constants module."""

    def test_da3_model_names_structure(self):
        """Test DA3 model names dictionary structure."""
        assert isinstance(DA3_MODEL_NAMES, dict)
        assert "small" in DA3_MODEL_NAMES
        assert "base" in DA3_MODEL_NAMES
        assert "large" in DA3_MODEL_NAMES
        assert "large-metric" in DA3_MODEL_NAMES
        assert "giant" in DA3_MODEL_NAMES

    def test_da3_model_names_format(self):
        """Test DA3 model names are valid HuggingFace IDs."""
        for model_name, hf_id in DA3_MODEL_NAMES.items():
            assert isinstance(hf_id, str)
            assert "/" in hf_id  # HF format: org/model
            assert hf_id.startswith("depth-anything/")

    def test_default_da3_model_exists(self):
        """Test default DA3 model is valid."""
        assert DEFAULT_DA3_MODEL in DA3_MODEL_NAMES

    def test_model_configs_structure(self):
        """Test V2 model configurations."""
        assert isinstance(MODEL_CONFIGS, dict)
        for model_type in ["vits", "vitb", "vitl"]:
            assert model_type in MODEL_CONFIGS
            config = MODEL_CONFIGS[model_type]
            assert "encoder" in config
            assert "features" in config
            assert "out_channels" in config
            assert "num_frames" in config

    def test_vr_resolutions_structure(self):
        """Test VR resolution configurations."""
        assert isinstance(VR_RESOLUTIONS, dict)
        for res_name, (width, height) in VR_RESOLUTIONS.items():
            assert isinstance(width, int)
            assert isinstance(height, int)
            assert width > 0
            assert height > 0

    def test_pimax_crystal_light_headset_preset(self):
        """Pimax Crystal Light uses its official native display dimensions and FOV."""
        assert VR_HEADSET_PRESETS["pimax-crystal-light"] == {
            "name": "Pimax Crystal Light",
            "per_eye_width": 2880,
            "per_eye_height": 2880,
            "fisheye_fov": 110,
            "description": (
                "Optimized for Pimax Crystal Light (2880x2880 per eye, 110 degree FOV)"
            ),
        }

    def test_validation_ranges_valid(self):
        """Test validation ranges are sensible."""
        assert VALIDATION_RANGES["stereo_strength"][0] < VALIDATION_RANGES["stereo_strength"][1]
        assert VALIDATION_RANGES["convergence"][0] < VALIDATION_RANGES["convergence"][1]
        assert VALIDATION_RANGES["fisheye_fov"][0] < VALIDATION_RANGES["fisheye_fov"][1]
        assert VALIDATION_RANGES["target_fps"][0] < VALIDATION_RANGES["target_fps"][1]

    def test_default_settings_complete(self):
        """Test default settings contain all required keys."""
        required_keys = [
            "stereo_strength",
            "convergence",
            "occlusion_fill",
            "vr_format",
            "vr_resolution",
            "preserve_audio",
            "keep_intermediates",
        ]
        for key in required_keys:
            assert key in DEFAULT_SETTINGS
