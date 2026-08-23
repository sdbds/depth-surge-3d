"""Typed boundaries for compact Quality rendering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class QualityStereoControls:
    """Geometry-independent Quality reconstruction controls."""

    occlusion_fill: Literal["none", "background"]
    occlusion_fill_max_px: int

    def __post_init__(self) -> None:
        if self.occlusion_fill not in {"none", "background"}:
            raise ValueError("occlusion_fill must be 'none' or 'background'")
        value = self.occlusion_fill_max_px
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError("occlusion_fill_max_px must be a non-boolean integer")
        normalized = int(value)
        if not 1 <= normalized <= 32:
            raise ValueError("occlusion_fill_max_px must be within 1..32")
        object.__setattr__(self, "occlusion_fill_max_px", normalized)


@dataclass(frozen=True)
class RelativeStereoPrimitiveInput:
    """One owned file-pipeline relative primitive."""

    encoded_canonical: np.ndarray
    encoding_scale: np.float32

    def __post_init__(self) -> None:
        encoded = self.encoded_canonical
        if (
            not isinstance(encoded, np.ndarray)
            or encoded.dtype != np.uint16
            or encoded.ndim != 2
            or not encoded.flags.c_contiguous
        ):
            raise TypeError("encoded_canonical must be a C-contiguous uint16 raster")
        scale = np.float32(self.encoding_scale)
        if not np.isfinite(scale) or scale <= np.float32(0.0):
            raise ValueError("encoding_scale must be finite and positive")
        object.__setattr__(self, "encoding_scale", scale)


@dataclass(frozen=True)
class MetricStereoPrimitiveInput:
    """One owned file-pipeline metric primitive and its projection controls."""

    metric: Any
    virtual_baseline_mm: np.float64
    convergence_distance_m: np.float64
    max_disparity_percent: np.float64
    retained_crop_width: int

    def __post_init__(self) -> None:
        for name in (
            "virtual_baseline_mm",
            "convergence_distance_m",
            "max_disparity_percent",
        ):
            value = np.float64(getattr(self, name))
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if (
            isinstance(self.retained_crop_width, bool)
            or not isinstance(self.retained_crop_width, Integral)
            or int(self.retained_crop_width) <= 0
        ):
            raise ValueError("retained_crop_width must be a positive integer")
        object.__setattr__(self, "retained_crop_width", int(self.retained_crop_width))


@dataclass(frozen=True)
class CompactStereoRenderResult:
    """RGB plus two one-byte diagnostic planes per eye."""

    left_image: np.ndarray
    right_image: np.ndarray
    left_coverage_count: np.ndarray
    right_coverage_count: np.ndarray
    left_repair_bits: np.ndarray
    right_repair_bits: np.ndarray
    left_plan: Any | None = None
    right_plan: Any | None = None
    metric_stats: Any | None = None

    def __post_init__(self) -> None:
        shape = self.left_image.shape
        if self.left_image.ndim != 3 or shape[-1] != 3 or self.right_image.shape != shape:
            raise ValueError("compact RGB images must have matching [H,W,3] shapes")
        plane_shape = shape[:2]
        for name in (
            "left_coverage_count",
            "right_coverage_count",
            "left_repair_bits",
            "right_repair_bits",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.uint8
                or value.shape != plane_shape
                or not value.flags.c_contiguous
            ):
                raise TypeError(f"{name} must be a matching C-contiguous uint8 raster")
