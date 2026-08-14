"""Domain-specific depth caching and resolution utilities."""

from .depth_cache import (
    get_cache_dir,
    compute_cache_key,
    get_cached_depth_map_files,
    save_depth_map_files_to_cache,
    clear_cache,
    get_cache_size,
)
from .resolution import (
    parse_custom_resolution,
    get_resolution_dimensions,
    calculate_vr_output_dimensions,
    calculate_aspect_ratio,
    classify_aspect_ratio,
    auto_detect_resolution,
    get_format_recommendation,
    validate_resolution_settings,
    get_available_resolutions,
)

__all__ = [
    # Depth cache
    "get_cache_dir",
    "compute_cache_key",
    "get_cached_depth_map_files",
    "save_depth_map_files_to_cache",
    "clear_cache",
    "get_cache_size",
    # Resolution
    "parse_custom_resolution",
    "get_resolution_dimensions",
    "calculate_vr_output_dimensions",
    "calculate_aspect_ratio",
    "classify_aspect_ratio",
    "auto_detect_resolution",
    "get_format_recommendation",
    "validate_resolution_settings",
    "get_available_resolutions",
]
