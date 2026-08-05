"""Bounded Torch renderer for canonical-disparity stereo projection."""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as functional

from .forward_splat import SplatBandResult, forward_splat_band


GPU_TEMP_BUDGET = 256 * 1024 * 1024
SPLAT_BYTES_PER_PIXEL = 192


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
class StereoRenderResult:
    """Host-resident stereo images and explicit geometric coverage masks."""

    left_image: np.ndarray
    right_image: np.ndarray
    left_valid_mask: np.ndarray
    right_valid_mask: np.ndarray
    left_hole_mask: np.ndarray
    right_hole_mask: np.ndarray


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


def _fill_background_band(
    splat: SplatBandResult,
    *,
    max_gap_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill bounded horizontal holes from the farther valid boundary."""

    image = splat.image
    valid = splat.valid_mask
    projected = splat.projected_disparity
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Band image must have shape [H,W,3]")
    if valid.shape != image.shape[:2] or projected.shape != valid.shape:
        raise ValueError("Band masks and projected disparity must match the image")
    if max_gap_width < 0:
        raise ValueError("Maximum gap width must be non-negative")

    height, width = valid.shape
    columns = torch.arange(width, dtype=torch.int64, device=image.device).view(1, width)
    columns = columns.expand(height, width)

    left_seed = torch.where(valid, columns, -1)
    left_index = torch.cummax(left_seed, dim=1).values
    right_seed = torch.where(valid, columns, width)
    right_index = torch.flip(
        torch.cummin(torch.flip(right_seed, dims=(1,)), dim=1).values,
        dims=(1,),
    )

    left_exists = left_index >= 0
    right_exists = right_index < width
    has_candidate = left_exists | right_exists
    left_safe = left_index.clamp(0, width - 1)
    right_safe = right_index.clamp(0, width - 1)

    left_disparity = torch.gather(projected, 1, left_safe)
    right_disparity = torch.gather(projected, 1, right_safe)
    left_distance = columns - left_index
    right_distance = right_index - columns
    both_exist = left_exists & right_exists
    equal_depth = left_disparity == right_disparity
    choose_left = (left_exists & ~right_exists) | (
        both_exist
        & ((left_disparity < right_disparity) | (equal_depth & (left_distance <= right_distance)))
    )

    left_colour = torch.gather(
        image,
        1,
        left_safe.unsqueeze(-1).expand(-1, -1, image.shape[-1]),
    )
    right_colour = torch.gather(
        image,
        1,
        right_safe.unsqueeze(-1).expand(-1, -1, image.shape[-1]),
    )
    candidate_colour = torch.where(choose_left.unsqueeze(-1), left_colour, right_colour)

    run_width = right_index - left_index - 1
    fillable = (~valid) & has_candidate & (run_width <= max_gap_width)
    hole_mask = (~valid) & ~fillable
    filled = torch.where(fillable.unsqueeze(-1), candidate_colour, image)
    filled.masked_fill_(hole_mask.unsqueeze(-1), 0.0)
    return filled, hole_mask


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
    ) -> None:
        if temporary_budget_bytes <= 0:
            raise ValueError("Temporary GPU budget must be positive")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.temporary_budget_bytes = temporary_budget_bytes

    def render(
        self,
        frame: np.ndarray,
        canonical: np.ndarray,
        settings: StereoRenderSettings | None = None,
    ) -> StereoRenderResult:
        """Resize canonical disparity once, then render both eyes in row bands."""

        source = np.asarray(frame)
        canonical_values = np.asarray(canonical, dtype=np.float32)
        settings = settings or StereoRenderSettings()
        self._validate_inputs(source, canonical_values, settings)
        source = np.ascontiguousarray(source)
        height, width = int(source.shape[0]), int(source.shape[1])
        resized_canonical = self._resize_canonical(canonical_values, (height, width))
        first_band_height = calculate_band_height(
            width,
            height,
            self.temporary_budget_bytes,
        )

        try:
            return self._render_with_band_height(
                source,
                resized_canonical,
                settings,
                first_band_height,
            )
        except torch.OutOfMemoryError:
            self._release_after_oom()

        retry_band_height = max(1, first_band_height // 2)
        try:
            return self._render_with_band_height(
                source,
                resized_canonical,
                settings,
                retry_band_height,
            )
        except torch.OutOfMemoryError as error:
            self._release_after_oom()
            raise RuntimeError(
                "CUDA stereo rendering ran out of memory for frame "
                f"{width}x{height}; attempted band heights "
                f"{first_band_height} and {retry_band_height}"
            ) from error

    @staticmethod
    def _validate_inputs(
        source: np.ndarray,
        canonical: np.ndarray,
        settings: StereoRenderSettings,
    ) -> None:
        if not isinstance(settings, StereoRenderSettings):
            raise TypeError("settings must be StereoRenderSettings")
        if source.ndim != 3 or source.shape[-1] != 3:
            raise ValueError("Frame must have shape [H,W,3]")
        if source.shape[0] <= 0 or source.shape[1] <= 0:
            raise ValueError("Frame dimensions must be positive")
        if not (
            np.issubdtype(source.dtype, np.integer) or np.issubdtype(source.dtype, np.floating)
        ):
            raise TypeError(f"Unsupported frame dtype: {source.dtype}")
        if np.issubdtype(source.dtype, np.floating) and not np.isfinite(source).all():
            raise ValueError("Frame values must be finite")
        if canonical.ndim != 2 or canonical.shape[0] <= 0 or canonical.shape[1] <= 0:
            raise ValueError("Canonical disparity must be a non-empty 2D array")
        if not np.isfinite(canonical).all():
            raise ValueError("Canonical disparity must be finite")
        if np.any(canonical < 0.0) or np.any(canonical > 1.0):
            raise ValueError("Canonical disparity must lie within [0, 1]")

    @staticmethod
    def _resize_canonical(
        canonical: np.ndarray,
        render_shape: tuple[int, int],
    ) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(canonical)).view(
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
        return tensor[0, 0].contiguous()

    def _render_with_band_height(
        self,
        source: np.ndarray,
        resized_canonical: torch.Tensor,
        settings: StereoRenderSettings,
        band_height: int,
    ) -> StereoRenderResult:
        left_image, left_valid, left_hole = self._render_eye(
            source,
            resized_canonical,
            settings,
            band_height,
            eye_sign=1,
        )
        right_image, right_valid, right_hole = self._render_eye(
            source,
            resized_canonical,
            settings,
            band_height,
            eye_sign=-1,
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
        resized_canonical: torch.Tensor,
        settings: StereoRenderSettings,
        band_height: int,
        *,
        eye_sign: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = source.shape[:2]
        output = np.empty_like(source)
        valid_output = np.empty((height, width), dtype=np.bool_)
        hole_output = np.empty((height, width), dtype=np.bool_)
        disparity_scale = float(np.float32(width * (settings.stereo_strength / 100.0)))
        max_gap_width = math.ceil(width * settings.stereo_strength / 200.0) + 2

        with torch.inference_mode():
            for start_row in range(0, height, band_height):
                end_row = min(height, start_row + band_height)
                source_band = torch.from_numpy(np.ascontiguousarray(source[start_row:end_row])).to(
                    self.device
                )
                canonical_band = resized_canonical[start_row:end_row].to(self.device)
                disparity_band = (canonical_band - float(settings.convergence)) * disparity_scale
                splat = forward_splat_band(source_band, disparity_band, eye_sign)
                if settings.occlusion_fill == "background":
                    rendered, hole_mask = _fill_background_band(
                        splat,
                        max_gap_width=max_gap_width,
                    )
                else:
                    rendered = splat.image
                    hole_mask = ~splat.valid_mask
                if not torch.isfinite(rendered).all():
                    raise ValueError("Rendered image contains non-finite values")

                output[start_row:end_row] = _convert_image_band(rendered, source.dtype)
                valid_output[start_row:end_row] = splat.valid_mask.detach().cpu().numpy()
                hole_output[start_row:end_row] = hole_mask.detach().cpu().numpy()

                del source_band, canonical_band, disparity_band, splat, rendered, hole_mask

        return output, valid_output, hole_output

    def _release_after_oom(self) -> None:
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
