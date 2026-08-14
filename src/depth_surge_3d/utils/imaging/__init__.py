"""Image and video processing utilities.

Core image transformations and video processing functions.
"""

from .image_processing import (
    # Image operations
    resize_image,
    validate_image_array,
    calculate_image_statistics,
    # Depth operations
    normalize_depth_map,
    # VR operations
    apply_center_crop,
    calculate_fisheye_coordinates,
    apply_fisheye_distortion,
    remap_fisheye,
    apply_fisheye_square_crop,
    create_vr_frame,
)
from .png_header import PngHeader, png_header_matches, read_png_header

__all__ = [
    # Image operations
    "resize_image",
    "validate_image_array",
    "calculate_image_statistics",
    # Depth operations
    "normalize_depth_map",
    # VR operations
    "apply_center_crop",
    "calculate_fisheye_coordinates",
    "apply_fisheye_distortion",
    "remap_fisheye",
    "apply_fisheye_square_crop",
    "create_vr_frame",
    "PngHeader",
    "png_header_matches",
    "read_png_header",
]
