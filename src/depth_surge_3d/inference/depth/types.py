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
class DepthBatch:
    """Native-resolution depth values and their required representation."""

    values: np.ndarray
    representation: DepthRepresentation

    def __post_init__(self) -> None:
        if not isinstance(self.values, np.ndarray):
            raise TypeError("DepthBatch.values must be a numpy array")
        if self.values.dtype != np.float32:
            raise TypeError("DepthBatch.values must use float32")
        if self.values.ndim != 3:
            raise ValueError("DepthBatch.values must have shape [N,H,W]")
        if not isinstance(self.representation, DepthRepresentation):
            raise TypeError("DepthBatch.representation must be a DepthRepresentation")

