"""Rendering modules for Depth Surge 3D."""

from __future__ import annotations

from typing import Any

__all__ = [
    "StereoProjector",
    "StereoRenderResult",
    "StereoRenderer",
    "StereoRenderSettings",
    "create_stereo_projector",
]


def __getattr__(name: str) -> Any:
    """Load renderers lazily so processing can depend on the low-level backend."""

    if name in {"StereoRenderResult", "StereoRenderer", "StereoRenderSettings"}:
        from .stereo_renderer import (
            StereoRenderResult,
            StereoRenderer,
            StereoRenderSettings,
        )

        return {
            "StereoRenderResult": StereoRenderResult,
            "StereoRenderer": StereoRenderer,
            "StereoRenderSettings": StereoRenderSettings,
        }[name]
    if name in {"StereoProjector", "create_stereo_projector"}:
        from .stereo_projector import StereoProjector, create_stereo_projector

        return {
            "StereoProjector": StereoProjector,
            "create_stereo_projector": create_stereo_projector,
        }[name]
    raise AttributeError(name)
