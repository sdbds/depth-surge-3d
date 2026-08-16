"""Pure depth-inference resolution selection shared by processing and resume."""

from __future__ import annotations

from typing import Any

from ...core.constants import (
    MEGAPIXELS_1080P,
    MEGAPIXELS_4K,
    MEGAPIXELS_720P,
    RESOLUTION_1080P,
    RESOLUTION_4K,
    RESOLUTION_720P,
    RESOLUTION_SD,
)


def auto_depth_input_size(frame_width: int, frame_height: int) -> int:
    """Resolve the shared automatic inference size from source geometry."""
    megapixels = (frame_height * frame_width) / 1_000_000
    source_max = max(frame_width, frame_height)
    if megapixels > MEGAPIXELS_4K:
        return min(source_max, RESOLUTION_4K)
    if megapixels > MEGAPIXELS_1080P:
        return min(source_max, RESOLUTION_1080P)
    if megapixels > MEGAPIXELS_720P:
        return min(source_max, RESOLUTION_720P)
    return min(source_max, RESOLUTION_SD)


def resolve_depth_input_size(
    frame_width: int,
    frame_height: int,
    depth_resolution: Any,
) -> int:
    """Resolve manual or automatic inference size without runtime side effects."""
    if depth_resolution != "auto":
        try:
            return int(depth_resolution)
        except (ValueError, TypeError):
            pass
    return auto_depth_input_size(frame_width, frame_height)
