"""Typed depth-estimator output contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class DepthRepresentation(str, Enum):
    """Semantic representation of an estimator's unscaled output."""

    RELATIVE_DEPTH = "relative_depth"
    METRIC_DEPTH = "metric_depth"
    INVERSE_DEPTH = "inverse_depth"


@dataclass(frozen=True)
class PinholeCameraBatch:
    focal_x_normalized: np.ndarray

    def __post_init__(self) -> None:
        values = self.focal_x_normalized
        if not isinstance(values, np.ndarray):
            raise TypeError("PinholeCameraBatch.focal_x_normalized must be a numpy array")
        if values.dtype != np.float32:
            raise TypeError("PinholeCameraBatch.focal_x_normalized must use float32")
        if values.ndim != 1:
            raise ValueError("PinholeCameraBatch.focal_x_normalized must have shape [N]")
        if not np.isfinite(values).all():
            raise ValueError("Pinhole focal values must be finite")
        if np.any(values <= 0.0):
            raise ValueError("Pinhole focal values must be positive")


@dataclass(frozen=True)
class DepthBatch:
    """Native-resolution depth values and their required representation."""

    values: np.ndarray
    representation: DepthRepresentation
    camera: PinholeCameraBatch | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.values, np.ndarray):
            raise TypeError("DepthBatch.values must be a numpy array")
        if self.values.dtype != np.float32:
            raise TypeError("DepthBatch.values must use float32")
        if self.values.ndim != 3:
            raise ValueError("DepthBatch.values must have shape [N,H,W]")
        if not isinstance(self.representation, DepthRepresentation):
            raise TypeError("DepthBatch.representation must be a DepthRepresentation")
        if self.camera is not None:
            if not isinstance(self.camera, PinholeCameraBatch):
                raise TypeError("DepthBatch.camera must be a PinholeCameraBatch or None")
            if len(self.camera.focal_x_normalized) != len(self.values):
                raise ValueError("Depth and camera batch lengths must match")
