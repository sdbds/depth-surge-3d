"""Backend-neutral stereo geometry shared by relative and metric renderers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class StereoGeometryFrame:
    """Per-source visibility rank, eye displacement, and source validity."""

    near_score: np.ndarray
    total_disparity_fraction: np.ndarray
    source_valid: np.ndarray

    def __post_init__(self) -> None:
        fields = (
            ("near_score", self.near_score, np.dtype(np.float32)),
            (
                "total_disparity_fraction",
                self.total_disparity_fraction,
                np.dtype(np.float64),
            ),
            ("source_valid", self.source_valid, np.dtype(np.bool_)),
        )
        for name, values, expected_dtype in fields:
            if not isinstance(values, np.ndarray) or values.dtype != expected_dtype:
                raise TypeError(f"{name} must be a NumPy array with dtype {expected_dtype}")
        if any(values.ndim != 2 for _name, values, _dtype in fields):
            raise ValueError("Stereo geometry fields must be 2D")
        if not (
            self.near_score.shape == self.total_disparity_fraction.shape == self.source_valid.shape
        ):
            raise ValueError("Stereo geometry fields must have the same shape")
        if not np.isfinite(self.near_score).all():
            raise ValueError("near_score must contain only finite values")
        if np.any(self.near_score < 0.0):
            raise ValueError("near_score must be nonnegative")
        if not np.isfinite(self.total_disparity_fraction).all():
            raise ValueError("total_disparity_fraction must contain only finite values")
        for name, values, _expected_dtype in fields:
            owned = np.array(values, copy=True, order="C")
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)


def _resize_float32_bilinear(
    canonical: np.ndarray,
    render_shape: tuple[int, int],
) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(canonical, dtype=np.float32)).view(
        1,
        1,
        canonical.shape[0],
        canonical.shape[1],
    )
    if canonical.shape != render_shape:
        tensor = functional.interpolate(
            tensor,
            size=render_shape,
            mode="bilinear",
            align_corners=False,
        )
    return tensor[0, 0].contiguous().numpy()


def build_relative_geometry(
    canonical: np.ndarray,
    render_shape: tuple[int, int],
    *,
    stereo_strength: float,
    convergence: float,
) -> StereoGeometryFrame:
    """Convert canonical relative disparity into renderer-neutral geometry."""

    canonical = np.asarray(canonical, dtype=np.float32)
    if canonical.ndim != 2 or canonical.shape[0] <= 0 or canonical.shape[1] <= 0:
        raise ValueError("Canonical disparity must be a non-empty 2D array")
    if not np.isfinite(canonical).all():
        raise ValueError("Canonical disparity must be finite")
    if np.any(canonical < 0.0) or np.any(canonical > 1.0):
        raise ValueError("Canonical disparity must lie within [0, 1]")
    if len(render_shape) != 2 or render_shape[0] <= 0 or render_shape[1] <= 0:
        raise ValueError("Render shape must contain positive height and width")
    if not math.isfinite(stereo_strength) or not 0.0 <= stereo_strength <= 5.0:
        raise ValueError("stereo_strength must be finite and within [0, 5]")
    if not math.isfinite(convergence) or not 0.0 <= convergence <= 1.0:
        raise ValueError("convergence must be finite and within [0, 1]")

    resized = _resize_float32_bilinear(canonical, render_shape)
    near64 = np.asarray(resized, dtype=np.float64)
    total = near64 - np.float64(convergence)
    total *= np.float64(stereo_strength)
    total /= np.float64(100.0)
    return StereoGeometryFrame(
        near_score=np.ascontiguousarray(resized, dtype=np.float32),
        total_disparity_fraction=np.ascontiguousarray(total, dtype=np.float64),
        source_valid=np.ones(render_shape, dtype=np.bool_),
    )
