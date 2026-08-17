"""
Unit tests for resolution utilities.

Basic tests to verify the modular architecture is working correctly.
"""

import unittest

import pytest

from src.depth_surge_3d.utils.domain.resolution import (
    parse_custom_resolution,
    get_resolution_dimensions,
    calculate_vr_output_dimensions,
    calculate_aspect_ratio,
    classify_aspect_ratio,
    auto_detect_resolution,
    resolve_depth_input_size,
)


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (640, 360, 640),
        (1280, 720, 640),
        (1920, 1080, 1080),
        (3840, 2160, 2160),
        (7680, 4320, 2160),
    ],
)
def test_shared_auto_depth_size_preserves_existing_thresholds(
    width: int, height: int, expected: int
) -> None:
    assert resolve_depth_input_size(width, height, "auto") == expected


@pytest.mark.parametrize(
    ("width", "height", "value", "message"),
    [
        (0, 360, "auto", "Source dimensions must be positive"),
        (640, 360, 0, "Depth resolution must be positive"),
    ],
)
def test_shared_depth_size_rejects_nonpositive_inputs(
    width: int, height: int, value: int | str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_depth_input_size(width, height, value)


class TestResolutionUtils(unittest.TestCase):
    """Test resolution utility functions."""

    def test_parse_custom_resolution(self):
        """Test custom resolution parsing."""
        # Valid formats
        self.assertEqual(parse_custom_resolution("custom:1920x1080"), (1920, 1080))
        self.assertEqual(parse_custom_resolution("custom:3840x2160"), (3840, 2160))

        # Invalid formats
        self.assertIsNone(parse_custom_resolution("1920x1080"))  # Missing custom:
        self.assertIsNone(parse_custom_resolution("custom:invalid"))
        self.assertIsNone(parse_custom_resolution("custom:0x0"))  # Zero dimensions

    def test_get_resolution_dimensions(self):
        """Test resolution dimension retrieval."""
        # Preset resolutions
        self.assertEqual(get_resolution_dimensions("square-1k"), (1080, 1080))
        self.assertEqual(get_resolution_dimensions("16x9-1080p"), (1920, 1080))

        # Custom resolutions
        self.assertEqual(get_resolution_dimensions("custom:1920x1080"), (1920, 1080))

        # Invalid resolution should raise error
        with self.assertRaises(ValueError):
            get_resolution_dimensions("invalid-resolution")

    def test_calculate_vr_output_dimensions(self):
        """Test VR output dimension calculation."""
        # Side by side
        self.assertEqual(calculate_vr_output_dimensions(1920, 1080, "side_by_side"), (3840, 1080))

        # Over under
        self.assertEqual(calculate_vr_output_dimensions(1920, 1080, "over_under"), (1920, 2160))

    def test_calculate_aspect_ratio(self):
        """Test aspect ratio calculation."""
        self.assertAlmostEqual(calculate_aspect_ratio(1920, 1080), 16 / 9, places=2)
        self.assertEqual(calculate_aspect_ratio(1080, 1080), 1.0)
        self.assertEqual(calculate_aspect_ratio(100, 0), 1.0)  # Zero height handling

    def test_classify_aspect_ratio(self):
        """Test aspect ratio classification."""
        self.assertEqual(classify_aspect_ratio(16 / 9), "wide")  # 1.78
        self.assertEqual(classify_aspect_ratio(1.0), "standard")  # Square
        self.assertEqual(classify_aspect_ratio(2.4), "ultra_wide")  # Cinema

    def test_auto_detect_resolution(self):
        """Test automatic resolution detection."""
        # 1080p source should get appropriate resolution
        resolution = auto_detect_resolution(1920, 1080, "side_by_side")
        self.assertIn("16x9", resolution)  # Should recommend 16:9 format

        # 4K source should get higher quality
        resolution = auto_detect_resolution(3840, 2160, "side_by_side")
        self.assertIn("4k", resolution.lower())


if __name__ == "__main__":
    unittest.main()
