"""Bounded Torch renderer for canonical-disparity stereo projection."""

from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from .forward_splat import (
    HORIZONTAL_SUBPIXELS,
    SubpixelSplatResult,
    forward_splat_band,
)
from .stereo_geometry import (
    StereoGeometryFrame,
    _resize_float32_bilinear,
    build_relative_geometry,
)


GPU_TEMP_BUDGET = 256 * 1024 * 1024
SPLAT_BYTES_PER_PIXEL = 1280


@dataclass(frozen=True)
class StereoRenderSettings:
    """Geometry and deterministic occlusion-fill controls."""

    stereo_strength: float = 2.0
    convergence: float = 0.5
    occlusion_fill: Literal["none", "background"] = "background"

    def __post_init__(self) -> None:
        if not math.isfinite(self.stereo_strength) or not 0.0 <= self.stereo_strength <= 5.0:
            raise ValueError("stereo_strength must be finite and within [0, 5]")
        if not math.isfinite(self.convergence) or not 0.0 <= self.convergence <= 1.0:
            raise ValueError("convergence must be finite and within [0, 1]")
        if self.occlusion_fill not in {"none", "background"}:
            raise ValueError("occlusion_fill must be 'none' or 'background'")


@dataclass(frozen=True)
class StereoSplatSettings:
    """Geometry-independent controls for splatting and occlusion fill."""

    max_eye_shift_fraction: float
    occlusion_fill: Literal["none", "background"] = "background"

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_eye_shift_fraction) or self.max_eye_shift_fraction < 0.0:
            raise ValueError("max_eye_shift_fraction must be finite and nonnegative")
        if self.occlusion_fill not in {"none", "background"}:
            raise ValueError("occlusion_fill must be 'none' or 'background'")


@dataclass(frozen=True)
class StereoRenderResult:
    """Host-resident stereo images and explicit geometric coverage masks."""

    left_image: np.ndarray
    right_image: np.ndarray
    left_valid_mask: np.ndarray
    right_valid_mask: np.ndarray
    left_hole_mask: np.ndarray
    right_hole_mask: np.ndarray


@dataclass(frozen=True)
class StereoRenderProfile:
    """Optional benchmark-only timing and transfer counters for one render."""

    host_geometry_seconds: float
    host_geometry_bytes: int
    offset_transfer_seconds: float
    offset_transfer_bytes: int


def calculate_band_height(
    render_width: int,
    render_height: int,
    temporary_budget_bytes: int = GPU_TEMP_BUDGET,
) -> int:
    """Calculate deterministic complete-row band height from a fixed byte budget."""

    if render_width <= 0 or render_height <= 0:
        raise ValueError("Render width and height must be positive")
    if temporary_budget_bytes <= 0:
        raise ValueError("Temporary GPU budget must be positive")
    rows = temporary_budget_bytes // (render_width * SPLAT_BYTES_PER_PIXEL)
    return min(render_height, max(1, rows))


def _narrow_sample_offsets(values: np.ndarray) -> np.ndarray:
    limits = np.iinfo(np.int32)
    if values.size and (values.min() < limits.min or values.max() > limits.max):
        raise ValueError("Projected fine-sample offsets exceed int32 range")
    return np.ascontiguousarray(values.astype(np.int32))


def calculate_eye_sample_offsets(
    canonical: torch.Tensor,
    settings: StereoRenderSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Build byte-compatible legacy relative-disparity offsets."""

    if canonical.device.type != "cpu" or canonical.ndim != 2:
        raise ValueError("Canonical disparity must be a host-resident 2D tensor")
    if not isinstance(settings, StereoRenderSettings):
        raise TypeError("settings must be StereoRenderSettings")
    canonical64 = np.asarray(canonical.detach().numpy(), dtype=np.float64)
    width = canonical.shape[1]
    scale64 = np.float64(width) * np.float64(settings.stereo_strength)
    scale64 = scale64 / np.float64(200.0)
    fine_shift64 = canonical64 - np.float64(settings.convergence)
    fine_shift64 = fine_shift64 * scale64
    fine_shift64 = fine_shift64 * np.float64(HORIZONTAL_SUBPIXELS)

    left64 = np.ceil(fine_shift64 - np.float64(0.5))
    left = _narrow_sample_offsets(left64)
    del left64
    right64 = np.ceil(-fine_shift64 - np.float64(0.5))
    right = _narrow_sample_offsets(right64)
    return left, right


def calculate_geometry_eye_sample_offsets(
    total_disparity_fraction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic full-frame fine-lane offsets from common geometry."""

    width = total_disparity_fraction.shape[1]
    fine_shift = np.asarray(total_disparity_fraction, dtype=np.float64)
    fine_shift = fine_shift * np.float64(width) * np.float64(0.5)
    fine_shift = fine_shift * np.float64(HORIZONTAL_SUBPIXELS)
    return (
        _narrow_sample_offsets(np.ceil(fine_shift - np.float64(0.5))),
        _narrow_sample_offsets(np.ceil(-fine_shift - np.float64(0.5))),
    )


def _nearest_valid_indexes(
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = valid.shape
    columns = torch.arange(width, dtype=torch.int32, device=valid.device).view(1, width)
    columns = columns.expand(height, width)
    left_seed = torch.where(valid, columns, -1)
    left_index = torch.cummax(left_seed, dim=1).values
    del left_seed
    right_seed = torch.where(valid, columns, width)
    reversed_seed = torch.flip(right_seed, dims=(1,))
    del right_seed
    reversed_index = torch.cummin(reversed_seed, dim=1).values
    del reversed_seed
    right_index = torch.flip(reversed_index, dims=(1,))
    del reversed_index
    return columns, left_index, right_index


def _gather_columns(values: torch.Tensor, indexes: torch.Tensor) -> torch.Tensor:
    safe = indexes.to(dtype=torch.int64)
    safe.clamp_(0, values.shape[1] - 1)
    if values.ndim == 2:
        return torch.gather(values, 1, safe)
    expanded = safe.unsqueeze(-1).expand(-1, -1, values.shape[-1])
    return torch.gather(values, 1, expanded)


def _select_fill_indexes(
    projected: torch.Tensor,
    columns: torch.Tensor,
    left_index: torch.Tensor,
    right_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = projected.shape[1]
    left_exists = left_index >= 0
    right_exists = right_index < width
    has_candidate = left_exists | right_exists
    both_exist = left_exists & right_exists
    left_disparity = _gather_columns(projected, left_index)
    right_disparity = _gather_columns(projected, right_index)
    left_distance = columns - left_index
    right_distance = right_index - columns
    nearer_left = left_disparity < right_disparity
    equal_prefers_left = (left_disparity == right_disparity) & (left_distance <= right_distance)
    choose_left = (left_exists & ~right_exists) | (both_exist & (nearer_left | equal_prefers_left))
    return torch.where(choose_left, left_index, right_index), has_candidate


def _fill_background_band(
    splat: SubpixelSplatResult,
    *,
    max_gap_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill bounded horizontal holes from one selected discrete boundary."""

    image = splat.colour
    valid = splat.valid
    projected = splat.disparity
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Band image must have shape [H,W,3]")
    if valid.shape != image.shape[:2] or projected.shape != valid.shape:
        raise ValueError("Band masks and projected disparity must match the image")
    if max_gap_samples < 0:
        raise ValueError("Maximum gap width must be non-negative")

    columns, left_index, right_index = _nearest_valid_indexes(valid)
    selected, has_candidate = _select_fill_indexes(
        projected,
        columns,
        left_index,
        right_index,
    )
    del columns
    run_width = right_index - left_index
    run_width.sub_(1)
    del left_index, right_index
    fillable = (~valid) & has_candidate & (run_width <= max_gap_samples)
    del has_candidate, run_width
    hole_mask = (~valid) & ~fillable
    candidate_colour = _gather_columns(image, selected)
    del selected
    filled = torch.where(fillable.unsqueeze(-1), candidate_colour, image)
    del candidate_colour, fillable
    filled.masked_fill_(hole_mask.unsqueeze(-1), 0.0)
    return filled, hole_mask


def _downsample_subpixel_band(colour: torch.Tensor) -> torch.Tensor:
    if colour.ndim != 3 or colour.shape[-1] != 3:
        raise ValueError("Fine-grid colour must have shape [H,W*S,3]")
    if colour.shape[1] % HORIZONTAL_SUBPIXELS != 0:
        raise ValueError("Fine-grid width must be divisible by the sample count")
    height, fine_width, channels = colour.shape
    lanes = colour.reshape(
        height,
        fine_width // HORIZONTAL_SUBPIXELS,
        HORIZONTAL_SUBPIXELS,
        channels,
    )
    pairs = lanes[:, :, 0::2] + lanes[:, :, 1::2]
    quads = pairs[:, :, 0::2] + pairs[:, :, 1::2]
    octets = quads[:, :, 0::2] + quads[:, :, 1::2]
    total = octets[:, :, 0] + octets[:, :, 1]
    return total * 0.0625


def _downsample_masks(
    prefill_valid: torch.Tensor,
    postfill_hole: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, fine_width = prefill_valid.shape
    width = fine_width // HORIZONTAL_SUBPIXELS
    valid = prefill_valid.reshape(height, width, HORIZONTAL_SUBPIXELS).any(dim=2)
    hole = postfill_hole.reshape(height, width, HORIZONTAL_SUBPIXELS).all(dim=2)
    return valid, hole


def _convert_image_band(image: torch.Tensor, dtype: np.dtype) -> np.ndarray:
    values = image.detach().cpu().numpy()
    if np.issubdtype(dtype, np.integer):
        integer_limits = np.iinfo(dtype)
        values = np.clip(np.rint(values), integer_limits.min, integer_limits.max)
    elif np.issubdtype(dtype, np.floating):
        floating_limits = np.finfo(dtype)
        values = np.clip(values, floating_limits.min, floating_limits.max)
    else:
        raise TypeError(f"Unsupported frame dtype: {dtype}")
    return values.astype(dtype, copy=False)


class StereoRenderer:
    """Render stereo pairs without full-frame device-resident intermediates."""

    def __init__(
        self,
        device: str | torch.device | None = None,
        *,
        temporary_budget_bytes: int = GPU_TEMP_BUDGET,
        profile: bool = False,
    ) -> None:
        if temporary_budget_bytes <= 0:
            raise ValueError("Temporary GPU budget must be positive")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.temporary_budget_bytes = temporary_budget_bytes
        self.profile = profile
        self.last_profile: StereoRenderProfile | None = None
        self._offset_transfer_seconds = 0.0
        self._offset_transfer_bytes = 0

    def render(
        self,
        frame: np.ndarray,
        canonical: np.ndarray,
        settings: StereoRenderSettings | None = None,
    ) -> StereoRenderResult:
        """Render relative disparity with legacy-compatible offset arithmetic."""

        source = np.asarray(frame)
        canonical_values = np.asarray(canonical, dtype=np.float32)
        settings = settings or StereoRenderSettings()
        self._validate_inputs(source, canonical_values, settings)
        source = np.ascontiguousarray(source)
        height, width = int(source.shape[0]), int(source.shape[1])
        geometry = build_relative_geometry(
            canonical_values,
            (height, width),
            stereo_strength=settings.stereo_strength,
            convergence=settings.convergence,
        )
        geometry_started = time.perf_counter()
        legacy_near_score = torch.from_numpy(np.array(geometry.near_score, copy=True, order="C"))
        eye_offsets = calculate_eye_sample_offsets(legacy_near_score, settings)
        geometry_seconds = time.perf_counter() - geometry_started
        return self._render_splat_core(
            source,
            geometry,
            eye_offsets,
            StereoSplatSettings(
                max_eye_shift_fraction=settings.stereo_strength / 200.0,
                occlusion_fill=settings.occlusion_fill,
            ),
            geometry_seconds,
        )

    def render_geometry(
        self,
        frame: np.ndarray,
        geometry: StereoGeometryFrame,
        settings: StereoSplatSettings,
    ) -> StereoRenderResult:
        """Render backend-neutral geometry that already matches the source raster."""

        source = np.asarray(frame)
        self._validate_geometry_inputs(source, geometry, settings)
        source = np.ascontiguousarray(source)
        geometry_started = time.perf_counter()
        eye_offsets = calculate_geometry_eye_sample_offsets(geometry.total_disparity_fraction)
        geometry_seconds = time.perf_counter() - geometry_started
        return self._render_splat_core(
            source,
            geometry,
            eye_offsets,
            settings,
            geometry_seconds,
        )

    def _render_splat_core(
        self,
        source: np.ndarray,
        geometry: StereoGeometryFrame,
        eye_offsets: tuple[np.ndarray, np.ndarray],
        settings: StereoSplatSettings,
        geometry_seconds: float,
    ) -> StereoRenderResult:
        """Render precomputed eye offsets through the shared banded splat path."""

        height, width = int(source.shape[0]), int(source.shape[1])
        geometry_bytes = sum(values.nbytes for values in eye_offsets)
        self._offset_transfer_seconds = 0.0
        self._offset_transfer_bytes = 0
        first_band_height = calculate_band_height(
            width,
            height,
            self.temporary_budget_bytes,
        )

        try:
            result = self._render_with_band_height(
                source,
                geometry,
                eye_offsets,
                settings,
                first_band_height,
            )
        except torch.cuda.OutOfMemoryError:
            self._release_after_oom()
            retry_band_height = max(1, first_band_height // 2)
            try:
                result = self._render_with_band_height(
                    source,
                    geometry,
                    eye_offsets,
                    settings,
                    retry_band_height,
                )
            except torch.cuda.OutOfMemoryError as error:
                self._release_after_oom()
                raise RuntimeError(
                    "CUDA stereo rendering ran out of memory for frame "
                    f"{width}x{height}; attempted band heights "
                    f"{first_band_height} and {retry_band_height}"
                ) from error

        self.last_profile = (
            StereoRenderProfile(
                host_geometry_seconds=geometry_seconds,
                host_geometry_bytes=geometry_bytes,
                offset_transfer_seconds=self._offset_transfer_seconds,
                offset_transfer_bytes=self._offset_transfer_bytes,
            )
            if self.profile
            else None
        )
        return result

    @staticmethod
    def _validate_inputs(
        source: np.ndarray,
        canonical: np.ndarray,
        settings: StereoRenderSettings,
    ) -> None:
        if not isinstance(settings, StereoRenderSettings):
            raise TypeError("settings must be StereoRenderSettings")
        StereoRenderer._validate_source(source)
        if canonical.ndim != 2 or canonical.shape[0] <= 0 or canonical.shape[1] <= 0:
            raise ValueError("Canonical disparity must be a non-empty 2D array")
        if not np.isfinite(canonical).all():
            raise ValueError("Canonical disparity must be finite")
        if np.any(canonical < 0.0) or np.any(canonical > 1.0):
            raise ValueError("Canonical disparity must lie within [0, 1]")

    @staticmethod
    def _validate_geometry_inputs(
        source: np.ndarray,
        geometry: StereoGeometryFrame,
        settings: StereoSplatSettings,
    ) -> None:
        if not isinstance(geometry, StereoGeometryFrame):
            raise TypeError("geometry must be StereoGeometryFrame")
        if not isinstance(settings, StereoSplatSettings):
            raise TypeError("settings must be StereoSplatSettings")
        StereoRenderer._validate_source(source)
        if geometry.near_score.shape != source.shape[:2]:
            raise ValueError("Stereo geometry must match the source raster")

    @staticmethod
    def _validate_source(source: np.ndarray) -> None:
        if source.ndim != 3 or source.shape[-1] != 3:
            raise ValueError("Frame must have shape [H,W,3]")
        if source.shape[0] <= 0 or source.shape[1] <= 0:
            raise ValueError("Frame dimensions must be positive")
        if int(source.shape[0]) * int(source.shape[1]) >= 2**32:
            raise ValueError("Full-frame source indexes must fit in 32 bits")
        if not (
            np.issubdtype(source.dtype, np.integer) or np.issubdtype(source.dtype, np.floating)
        ):
            raise TypeError(f"Unsupported frame dtype: {source.dtype}")
        if np.issubdtype(source.dtype, np.floating) and not np.isfinite(source).all():
            raise ValueError("Frame values must be finite")

    @staticmethod
    def _resize_canonical(
        canonical: np.ndarray,
        render_shape: tuple[int, int],
    ) -> torch.Tensor:
        return torch.from_numpy(_resize_float32_bilinear(canonical, render_shape))

    def _render_with_band_height(
        self,
        source: np.ndarray,
        geometry: StereoGeometryFrame,
        eye_offsets: tuple[np.ndarray, np.ndarray],
        settings: StereoSplatSettings,
        band_height: int,
    ) -> StereoRenderResult:
        left_image, left_valid, left_hole = self._render_eye(
            source,
            geometry,
            eye_offsets[0],
            settings,
            band_height,
        )
        right_image, right_valid, right_hole = self._render_eye(
            source,
            geometry,
            eye_offsets[1],
            settings,
            band_height,
        )
        return StereoRenderResult(
            left_image=left_image,
            right_image=right_image,
            left_valid_mask=left_valid,
            right_valid_mask=right_valid,
            left_hole_mask=left_hole,
            right_hole_mask=right_hole,
        )

    def _render_eye(
        self,
        source: np.ndarray,
        geometry: StereoGeometryFrame,
        sample_offsets: np.ndarray,
        settings: StereoSplatSettings,
        band_height: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = source.shape[:2]
        output = np.empty_like(source)
        valid_output = np.empty((height, width), dtype=np.bool_)
        hole_output = np.empty((height, width), dtype=np.bool_)
        max_gap_width = math.ceil(width * settings.max_eye_shift_fraction) + 2
        max_gap_samples = HORIZONTAL_SUBPIXELS * max_gap_width

        with torch.inference_mode():
            for start_row in range(0, height, band_height):
                end_row = min(height, start_row + band_height)
                source_band = torch.from_numpy(np.ascontiguousarray(source[start_row:end_row])).to(
                    self.device
                )
                near_score_band = torch.from_numpy(
                    np.array(geometry.near_score[start_row:end_row], copy=True, order="C")
                ).to(self.device)
                source_valid_band = torch.from_numpy(
                    np.array(geometry.source_valid[start_row:end_row], copy=True, order="C")
                ).to(self.device)
                offset_band = self._transfer_offset_band(sample_offsets[start_row:end_row])
                splat = forward_splat_band(
                    source_band,
                    near_score_band,
                    offset_band,
                    source_index_offset=start_row * width,
                    source_valid=source_valid_band,
                )
                if settings.occlusion_fill == "background":
                    rendered, hole_mask = _fill_background_band(
                        splat,
                        max_gap_samples=max_gap_samples,
                    )
                else:
                    rendered = splat.colour
                    hole_mask = ~splat.valid
                if not torch.isfinite(rendered).all():
                    raise ValueError("Rendered image contains non-finite values")

                downsampled = _downsample_subpixel_band(rendered)
                pixel_valid, pixel_hole = _downsample_masks(splat.valid, hole_mask)
                output[start_row:end_row] = _convert_image_band(downsampled, source.dtype)
                valid_output[start_row:end_row] = pixel_valid.detach().cpu().numpy()
                hole_output[start_row:end_row] = pixel_hole.detach().cpu().numpy()

                del (
                    source_band,
                    near_score_band,
                    source_valid_band,
                    offset_band,
                    splat,
                    rendered,
                    hole_mask,
                    downsampled,
                    pixel_valid,
                    pixel_hole,
                )

        return output, valid_output, hole_output

    def _transfer_offset_band(self, values: np.ndarray) -> torch.Tensor:
        if self.profile and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        result = torch.from_numpy(values).to(self.device)
        if self.profile and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if self.profile:
            self._offset_transfer_seconds += time.perf_counter() - started
            self._offset_transfer_bytes += values.nbytes
        return result

    def _release_after_oom(self) -> None:
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
