"""Rendering modules for Depth Surge 3D.

High-level stereo rendering and projection.
"""

from .stereo_projector import StereoProjector, create_stereo_projector
from .stereo_renderer import (
    StereoRenderResult,
    StereoRenderer,
    StereoRenderSettings,
)

__all__ = [
    "StereoProjector",
    "StereoRenderResult",
    "StereoRenderer",
    "StereoRenderSettings",
    "create_stereo_projector",
]
