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


@dataclass(frozen=True)
class MetricProjectionStats:
    """Counts projection-cap use across valid pixels in one metric frame."""

    valid_pixel_count: int
    clamped_pixel_count: int
    clamped_fraction: float

    def __post_init__(self) -> None:
        for name, value in (
            ("valid_pixel_count", self.valid_pixel_count),
            ("clamped_pixel_count", self.clamped_pixel_count),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        if self.valid_pixel_count < 0:
            raise ValueError("valid_pixel_count must be nonnegative")
        if not 0 <= self.clamped_pixel_count <= self.valid_pixel_count:
            raise ValueError("clamped_pixel_count must lie within the valid pixel count")
        if isinstance(self.clamped_fraction, (bool, np.bool_)) or not isinstance(
            self.clamped_fraction, (int, float, np.number)
        ):
            raise TypeError("clamped_fraction must be numeric")
        clamped_fraction = float(self.clamped_fraction)
        object.__setattr__(self, "clamped_fraction", clamped_fraction)
        if not math.isfinite(clamped_fraction) or not 0.0 <= clamped_fraction <= 1.0:
            raise ValueError("clamped_fraction must be finite and within [0, 1]")
        expected_fraction = (
            self.clamped_pixel_count / self.valid_pixel_count if self.valid_pixel_count else 0.0
        )
        if clamped_fraction != expected_fraction:
            raise ValueError("clamped_fraction must match the pixel counts")


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


def resize_metric_geometry(
    inverse_depth: np.ndarray,
    valid: np.ndarray,
    render_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Resize inverse depth with matching bilinear validity weights."""

    if not isinstance(inverse_depth, np.ndarray) or inverse_depth.dtype != np.float32:
        raise TypeError("Metric inverse_depth must be a NumPy array with dtype float32")
    if not isinstance(valid, np.ndarray) or valid.dtype != np.bool_:
        raise TypeError("Metric valid must be a NumPy array with dtype bool")
    if inverse_depth.ndim != 2 or inverse_depth.shape[0] <= 0 or inverse_depth.shape[1] <= 0:
        raise ValueError("Metric inverse_depth must be a non-empty 2D array")
    if valid.ndim != 2 or valid.shape != inverse_depth.shape:
        raise ValueError("Metric valid must match inverse_depth shape")
    if not np.isfinite(inverse_depth).all() or np.any(inverse_depth < np.float32(0.0)):
        raise ValueError("Metric inverse_depth must be finite and nonnegative")
    if len(render_shape) != 2 or any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0
        for value in render_shape
    ):
        raise ValueError("Render shape must contain positive height and width")

    valid_float = np.ascontiguousarray(valid, dtype=np.float32)
    weighted_inverse = np.ascontiguousarray(inverse_depth * valid_float, dtype=np.float32)
    resized_weight = _resize_float32_bilinear(valid_float, render_shape)
    resized_weighted_inverse = _resize_float32_bilinear(weighted_inverse, render_shape)
    resized_valid = np.ascontiguousarray(resized_weight >= np.float32(0.5), dtype=np.bool_)
    resized_inverse = np.zeros(render_shape, dtype=np.float32)
    np.divide(
        resized_weighted_inverse,
        resized_weight,
        out=resized_inverse,
        where=resized_weight > np.float32(0.0),
    )
    resized_inverse[~resized_valid] = np.float32(0.0)
    return resized_inverse, resized_valid


def _validate_metric_scalar(
    name: str,
    value: object,
    *,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool = True,
) -> np.float64:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be numeric")
    result = np.float64(value)
    in_range = result >= minimum if minimum_inclusive else result > minimum
    if not np.isfinite(result) or not in_range or result > maximum:
        lower = "[" if minimum_inclusive else "("
        raise ValueError(f"{name} must be finite and within {lower}{minimum}, {maximum}]")
    return result


def _validate_finite_positive_metric_scalar(name: str, value: object) -> np.float64:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be numeric")
    result = np.float64(value)
    if not np.isfinite(result) or result <= np.float64(0.0):
        raise ValueError(f"{name} must be finite and positive")
    return result


def build_metric_geometry(
    inverse_depth: np.ndarray,
    valid: np.ndarray,
    focal_x_normalized: np.float32,
    render_shape: tuple[int, int],
    *,
    virtual_baseline_mm: float,
    convergence_distance_m: float,
    max_disparity_percent: float,
    retained_crop_width: int,
) -> tuple[StereoGeometryFrame, MetricProjectionStats]:
    """Project metric inverse depth into crop-aware renderer coordinates."""

    if not isinstance(focal_x_normalized, np.float32):
        raise TypeError("focal_x_normalized must use float32")
    focal = _validate_metric_scalar(
        "focal_x_normalized",
        focal_x_normalized,
        minimum=0.0,
        maximum=float(np.finfo(np.float32).max),
        minimum_inclusive=False,
    )
    baseline_mm = _validate_metric_scalar(
        "virtual_baseline_mm",
        virtual_baseline_mm,
        minimum=0.0,
        maximum=100.0,
    )
    convergence_m = _validate_finite_positive_metric_scalar(
        "convergence_distance_m", convergence_distance_m
    )
    disparity_percent = _validate_metric_scalar(
        "max_disparity_percent",
        max_disparity_percent,
        minimum=0.0,
        maximum=100.0,
    )
    resized_inverse, resized_valid = resize_metric_geometry(inverse_depth, valid, render_shape)
    _source_height, source_width = render_shape
    if (
        isinstance(retained_crop_width, (bool, np.bool_))
        or not isinstance(retained_crop_width, (int, np.integer))
        or not 1 <= int(retained_crop_width) <= source_width
    ):
        raise ValueError("retained crop width must be an integer in 1..source width")

    retained_fraction = np.float64(retained_crop_width) / np.float64(source_width)
    baseline_m = baseline_mm / np.float64(1000.0)
    limit = disparity_percent / np.float64(100.0)
    raw_output_fraction = (
        focal
        / retained_fraction
        * baseline_m
        * (resized_inverse.astype(np.float64) - np.float64(1.0 / convergence_m))
    )
    clamped_output_fraction = np.clip(raw_output_fraction, -limit, limit)
    render_fraction = clamped_output_fraction * retained_fraction
    render_fraction[~resized_valid] = np.float64(0.0)

    valid_pixel_count = int(np.count_nonzero(resized_valid))
    clamped_pixel_count = int(
        np.count_nonzero(
            resized_valid & ((raw_output_fraction < -limit) | (raw_output_fraction > limit))
        )
    )
    clamped_fraction = clamped_pixel_count / valid_pixel_count if valid_pixel_count else 0.0
    return (
        StereoGeometryFrame(
            near_score=np.ascontiguousarray(resized_inverse, dtype=np.float32),
            total_disparity_fraction=np.ascontiguousarray(render_fraction, dtype=np.float64),
            source_valid=np.ascontiguousarray(resized_valid, dtype=np.bool_),
        ),
        MetricProjectionStats(valid_pixel_count, clamped_pixel_count, clamped_fraction),
    )


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
