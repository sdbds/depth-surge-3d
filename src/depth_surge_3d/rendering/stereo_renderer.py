"""Bounded Torch renderer for canonical-disparity stereo projection."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
import gc
import math
import time
from dataclasses import dataclass, field, replace
from numbers import Integral
from typing import Literal

import numpy as np
import torch

from .forward_splat import (
    HORIZONTAL_SUBPIXELS,
    SubpixelSplatResult,
    _forward_splat_band_prevalidated,
    forward_splat_band,
)
from .stereo_geometry import (
    StereoGeometryFrame,
    _resize_float32_bilinear,
    build_metric_geometry,
    build_relative_geometry,
)


GPU_TEMP_BUDGET = 256 * 1024 * 1024
SPLAT_BYTES_PER_PIXEL = 1280
TRITON_SPLAT_BYTES_PER_PIXEL = 1088
_FAST_CUDA_BUDGET_CAP_BYTES = 1536 * 1024 * 1024
_FAST_CUDA_FREE_BUDGET_DIVISOR = 4
_FAST_CUDA_TARGET_MAX_BANDS_PER_EYE = 4


@dataclass(frozen=True)
class StereoRenderSettings:
    """Geometry and deterministic occlusion-fill controls."""

    stereo_strength: float = 2.0
    convergence: float = 0.5
    occlusion_fill: Literal["none", "background"] = "background"
    stereo_render_mode: Literal["fast", "quality"] = field(default="fast", kw_only=True)
    occlusion_fill_max_px: int = field(default=8, kw_only=True)

    def __post_init__(self) -> None:
        if not math.isfinite(self.stereo_strength) or not 0.0 <= self.stereo_strength <= 5.0:
            raise ValueError("stereo_strength must be finite and within [0, 5]")
        if not math.isfinite(self.convergence) or not 0.0 <= self.convergence <= 1.0:
            raise ValueError("convergence must be finite and within [0, 1]")
        if self.occlusion_fill not in {"none", "background"}:
            raise ValueError("occlusion_fill must be 'none' or 'background'")
        if self.stereo_render_mode not in {"fast", "quality"}:
            raise ValueError("stereo_render_mode must be 'fast' or 'quality'")
        value = self.occlusion_fill_max_px
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError("occlusion_fill_max_px must be a non-boolean integer")
        normalized = int(value)
        if not 1 <= normalized <= 32:
            raise ValueError("occlusion_fill_max_px must be within 1..32")
        object.__setattr__(self, "occlusion_fill_max_px", normalized)


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


@dataclass(frozen=True)
class _FastRelativeGeometry:
    """Owned raster fields for the validated relative-offset path."""

    near_score: np.ndarray
    source_valid: np.ndarray


@dataclass(frozen=True)
class _FastDeviceFrame:
    source: torch.Tensor
    near_score: torch.Tensor
    source_valid: torch.Tensor


@dataclass
class _FastRelativeOffsets:
    """Keep host inputs for fallback, never resident device allocations."""

    near_score: torch.Tensor
    settings: StereoRenderSettings
    build_seconds: float = field(default=0.0, init=False)
    _host_offsets: tuple[np.ndarray, np.ndarray] | None = field(default=None, init=False)

    def host_offsets(self) -> tuple[np.ndarray, np.ndarray]:
        if self._host_offsets is None:
            started = time.perf_counter()
            self._host_offsets = calculate_eye_sample_offsets(self.near_score, self.settings)
            self.build_seconds += time.perf_counter() - started
        return self._host_offsets

    def device_offsets(
        self, near_score: torch.Tensor, *, profile: bool
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        from .fast_offsets_triton import triton_relative_offsets_prevalidated

        if profile:
            torch.cuda.synchronize(near_score.device)
        started = time.perf_counter()
        try:
            result = triton_relative_offsets_prevalidated(
                near_score,
                stereo_strength=self.settings.stereo_strength,
                convergence=self.settings.convergence,
            )
            if profile:
                torch.cuda.synchronize(near_score.device)
            return result
        finally:
            self.build_seconds += time.perf_counter() - started


def calculate_band_height(
    render_width: int,
    render_height: int,
    temporary_budget_bytes: int = GPU_TEMP_BUDGET,
    *,
    bytes_per_pixel: int = SPLAT_BYTES_PER_PIXEL,
) -> int:
    """Calculate deterministic complete-row band height from a fixed byte budget."""

    if render_width <= 0 or render_height <= 0:
        raise ValueError("Render width and height must be positive")
    if temporary_budget_bytes <= 0 or bytes_per_pixel <= 0:
        raise ValueError("Temporary GPU budget and bytes per pixel must be positive")
    rows = temporary_budget_bytes // (render_width * bytes_per_pixel)
    return min(render_height, max(1, rows))


def _fast_temporary_budget_bytes(
    renderer: StereoRenderer,
    *,
    render_width: int,
    render_height: int,
    bytes_per_pixel: int,
) -> int:
    """Grow the default CUDA budget only enough to bound band dispatches."""

    configured = int(renderer.temporary_budget_bytes)
    if renderer.device.type != "cuda" or not getattr(
        renderer, "_temporary_budget_is_default", False
    ):
        return configured
    target_rows = max(
        1,
        math.ceil(render_height / _FAST_CUDA_TARGET_MAX_BANDS_PER_EYE),
    )
    target_budget = min(
        _FAST_CUDA_BUDGET_CAP_BYTES,
        render_width * target_rows * bytes_per_pixel,
    )
    if target_budget <= configured:
        return configured
    try:
        free_bytes, _ = torch.cuda.mem_get_info(renderer.device)
    except (AssertionError, RuntimeError):
        return configured
    adaptive = min(
        target_budget,
        int(free_bytes) // _FAST_CUDA_FREE_BUDGET_DIVISOR,
    )
    return max(configured, adaptive)


def _triton_splat_available(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    from .fast_splat_triton import triton_splat_available

    return triton_splat_available(device)


def _narrow_torch_sample_offsets(values: torch.Tensor) -> np.ndarray:
    limits = torch.iinfo(torch.int32)
    if values.numel():
        minimum, maximum = torch.aminmax(values)
        if minimum < limits.min or maximum > limits.max:
            raise ValueError("Projected fine-sample offsets exceed int32 range")
    return values.to(dtype=torch.int32).contiguous().numpy()


def calculate_eye_sample_offsets(
    canonical: torch.Tensor,
    settings: StereoRenderSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Build byte-compatible legacy relative-disparity offsets."""

    if canonical.device.type != "cpu" or canonical.ndim != 2:
        raise ValueError("Canonical disparity must be a host-resident 2D tensor")
    if not isinstance(settings, StereoRenderSettings):
        raise TypeError("settings must be StereoRenderSettings")
    width = canonical.shape[1]
    scale64 = np.float64(width) * np.float64(settings.stereo_strength)
    scale64 = scale64 / np.float64(200.0)
    fine_shift64 = canonical.detach().to(dtype=torch.float64, copy=True).contiguous()
    fine_shift64.sub_(float(np.float64(settings.convergence)))
    fine_shift64.mul_(float(scale64))
    fine_shift64.mul_(float(np.float64(HORIZONTAL_SUBPIXELS)))

    left64 = fine_shift64.clone()
    left64.sub_(float(np.float64(0.5))).ceil_()
    left = _narrow_torch_sample_offsets(left64)
    del left64
    right64 = fine_shift64.neg()
    right64.sub_(float(np.float64(0.5))).ceil_()
    right = _narrow_torch_sample_offsets(right64)
    return left, right


def calculate_geometry_eye_sample_offsets(
    total_disparity_fraction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic full-frame fine-lane offsets from common geometry."""

    width = total_disparity_fraction.shape[1]
    fine_shift64 = torch.from_numpy(
        np.array(
            total_disparity_fraction,
            dtype=np.float64,
            copy=True,
            order="C",
        )
    )
    fine_shift64.mul_(float(np.float64(width)))
    fine_shift64.mul_(float(np.float64(0.5)))
    fine_shift64.mul_(float(np.float64(HORIZONTAL_SUBPIXELS)))
    left64 = fine_shift64.clone()
    left64.sub_(float(np.float64(0.5))).ceil_()
    left = _narrow_torch_sample_offsets(left64)
    del left64
    right64 = fine_shift64.neg()
    right64.sub_(float(np.float64(0.5))).ceil_()
    right = _narrow_torch_sample_offsets(right64)
    return left, right


def _nearest_valid_indexes(
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = valid.shape
    columns = torch.arange(width, dtype=torch.int32, device=valid.device).view(1, width)
    columns = columns.expand(height, width)
    if valid.device.type == "cuda":
        from .fast_scan_triton import triton_nearest_valid_indexes

        boundaries = triton_nearest_valid_indexes(valid)
        if boundaries is not None:
            return columns, boundaries[0], boundaries[1]
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


def _fill_background_band_state(
    splat: SubpixelSplatResult,
    *,
    max_gap_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    if image.device.type == "cuda":
        from .fast_fill_triton import triton_background_fill

        fused = triton_background_fill(
            image,
            valid,
            projected,
            left_index,
            right_index,
            max_gap_samples=max_gap_samples,
        )
        if fused is not None:
            del columns, left_index, right_index
            return fused
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
    del candidate_colour
    filled.masked_fill_(hole_mask.unsqueeze(-1), 0.0)
    return filled, hole_mask, fillable


def _fill_background_band(
    splat: SubpixelSplatResult,
    *,
    max_gap_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill bounded horizontal holes with byte-compatible legacy behavior."""

    filled, hole_mask, fillable = _fill_background_band_state(
        splat,
        max_gap_samples=max_gap_samples,
    )
    del fillable
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
    if image.device.type == "cuda" and np.dtype(dtype) == np.dtype(np.uint8):
        values = torch.round(image.detach())
        values.clamp_(0, 255)
        return values.to(dtype=torch.uint8).cpu().numpy()
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


def _download_fast_cuda_band(
    image: torch.Tensor,
    lane_valid: torch.Tensor,
    lane_weights: torch.Tensor,
    repair: torch.Tensor,
    lane_counts: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download uint8 RGB and native diagnostics with one bounded CUDA copy."""

    values = torch.round(image.detach())
    values.clamp_(0, 255)
    rgb = values.to(dtype=torch.uint8)
    missing = _pack_missing_lane_masks(lane_valid, lane_weights)
    payload = torch.cat(
        (
            lane_counts.view(torch.uint8),
            missing.view(torch.uint8).reshape(-1),
            rgb.reshape(-1),
            repair.reshape(-1),
        )
    )
    return _download_fast_cuda_payload(
        payload,
        tuple(image.shape),
        tuple(repair.shape),
        count_bytes=int(lane_counts.numel() * lane_counts.element_size()),
    )


def _download_fast_cuda_payload(
    payload: torch.Tensor,
    image_shape: tuple[int, ...],
    repair_shape: tuple[int, ...],
    *,
    count_bytes: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expose aligned views of either producer's identical bounded packet."""

    host = payload.cpu().numpy()
    pixel_count = math.prod(repair_shape)
    mask_end = count_bytes + 2 * pixel_count
    rgb_end = mask_end + 3 * pixel_count
    return (
        host[mask_end:rgb_end].reshape(image_shape),
        host[count_bytes:mask_end].view(np.uint16).reshape(repair_shape),
        host[rgb_end:].reshape(repair_shape),
        host[:count_bytes].view(np.int64),
    )


def _update_hole_histograms(
    lane_valid: np.ndarray,
    lane_histogram: np.ndarray,
    span_histogram: np.ndarray,
) -> None:
    for row in lane_valid:
        missing = np.flatnonzero(~row.reshape(-1))
        cursor = 0
        while cursor < missing.size:
            start = int(missing[cursor])
            end_cursor = cursor + 1
            while (
                end_cursor < missing.size
                and int(missing[end_cursor]) == int(missing[end_cursor - 1]) + 1
            ):
                end_cursor += 1
            end = int(missing[end_cursor - 1])
            lane_histogram[end - start + 1] += np.uint64(1)
            span_histogram[
                end // HORIZONTAL_SUBPIXELS - start // HORIZONTAL_SUBPIXELS + 1
            ] += np.uint64(1)
            cursor = end_cursor


def _lane_mask_weights(device: torch.device) -> torch.Tensor:
    return torch.bitwise_left_shift(
        torch.ones(HORIZONTAL_SUBPIXELS, dtype=torch.int32, device=device),
        torch.arange(HORIZONTAL_SUBPIXELS, dtype=torch.int32, device=device),
    )


def _pack_missing_lane_masks(
    lane_valid: torch.Tensor,
    lane_weights: torch.Tensor,
) -> torch.Tensor:
    valid_masks = torch.sum(
        lane_valid * lane_weights,
        dim=2,
        dtype=torch.int32,
    )
    valid_masks.bitwise_xor_(0xFFFF)
    return valid_masks.to(dtype=torch.int16)


def _native_lane_analyzer():
    from .quality import native_kd

    if not native_kd.native_available():
        return None
    return native_kd._require_native()


def _packed_lane_coverage_lut() -> np.ndarray:
    values = np.arange(1 << HORIZONTAL_SUBPIXELS, dtype=np.uint16)
    bits = np.unpackbits(values.view(np.uint8).reshape(-1, 2), axis=1)
    return np.ascontiguousarray(
        HORIZONTAL_SUBPIXELS - np.sum(bits, axis=1, dtype=np.uint8),
        dtype=np.uint8,
    )


_PACKED_LANE_COVERAGE = _packed_lane_coverage_lut()


class _FastLaneAnalysisQueue:
    """Overlap exact native hole statistics with subsequent CUDA bands."""

    _MAX_PENDING = 2
    _DEFAULT_ASYNC_THRESHOLD_SECONDS = 0.002
    _DEFAULT_DEFERRED_ASYNC_THRESHOLD_SECONDS = 0.0005

    def __init__(
        self,
        native_extension,
        *,
        async_threshold_seconds: float | None = None,
    ) -> None:
        self._native_extension = native_extension
        self._async_threshold_seconds = (
            None if async_threshold_seconds is None else float(async_threshold_seconds)
        )
        self._executor: ThreadPoolExecutor | None = None
        self._pending: deque[Future[object]] = deque()
        self._closed = False

    def __enter__(self) -> _FastLaneAnalysisQueue:
        return self

    def __exit__(self, exception_type, _exception, _traceback) -> Literal[False]:
        failure: BaseException | None = None
        while self._pending:
            try:
                self._pending.popleft().result()
            except BaseException as error:
                if failure is None:
                    failure = error
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=exception_type is not None)
        if exception_type is None and failure is not None:
            raise failure
        return False

    def submit(
        self,
        packed_missing: np.ndarray,
        lane_histogram: np.ndarray,
        span_histogram: np.ndarray,
        coverage_output: np.ndarray,
        *,
        defer_coverage: bool = False,
    ) -> None:
        """Deferred coverage is readable only after joining the submitted work."""

        if self._closed:
            raise RuntimeError("Fast lane-analysis queue is closed")
        if self._executor is None:
            threshold = self._async_threshold_seconds
            if threshold is None:
                threshold = (
                    self._DEFAULT_DEFERRED_ASYNC_THRESHOLD_SECONDS
                    if defer_coverage
                    else self._DEFAULT_ASYNC_THRESHOLD_SECONDS
                )
            started = time.perf_counter()
            self._native_extension.accumulate_packed_lane_hole_run_histograms_into(
                packed_missing,
                lane_histogram,
                span_histogram,
                coverage_output,
            )
            if time.perf_counter() - started >= threshold:
                self._executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="fast-lane-analysis",
                )
            return
        if not defer_coverage:
            coverage_output[:] = _PACKED_LANE_COVERAGE[packed_missing]
        if len(self._pending) >= self._MAX_PENDING:
            self._pending.popleft().result()
        assert self._executor is not None
        self._pending.append(
            self._executor.submit(
                self._native_extension.accumulate_packed_lane_hole_run_histograms_into,
                packed_missing,
                lane_histogram,
                span_histogram,
                coverage_output if defer_coverage else None,
            )
        )

    def finish_pending(self) -> None:
        """Finish queued updates before their histogram buffers become immutable."""

        while self._pending:
            self._pending.popleft().result()


def _analyze_lane_validity(
    lane_valid: torch.Tensor,
    lane_histogram: np.ndarray,
    span_histogram: np.ndarray,
    coverage_output: np.ndarray,
    *,
    native_extension,
    lane_weights: torch.Tensor | None,
    analysis_queue: _FastLaneAnalysisQueue | None = None,
) -> None:
    if native_extension is None:
        host_valid = np.ascontiguousarray(lane_valid.detach().cpu().numpy())
        coverage_output[:] = np.sum(host_valid, axis=2, dtype=np.uint16).astype(np.uint8)
        _update_hole_histograms(host_valid, lane_histogram, span_histogram)
        return
    if lane_valid.device.type == "cuda":
        if lane_weights is None:
            raise AssertionError("CUDA lane-mask weights are unavailable")
        packed_missing = np.ascontiguousarray(
            _pack_missing_lane_masks(lane_valid, lane_weights).cpu().numpy().view(np.uint16)
        )
        _analyze_packed_lane_validity(
            packed_missing,
            lane_histogram,
            span_histogram,
            coverage_output,
            native_extension=native_extension,
            analysis_queue=analysis_queue,
        )
        return
    host_valid = np.ascontiguousarray(lane_valid.detach().numpy())
    native_extension.accumulate_lane_hole_run_histograms_into(
        host_valid,
        lane_histogram,
        span_histogram,
        coverage_output,
    )


def _analyze_packed_lane_validity(
    packed_missing: np.ndarray,
    lane_histogram: np.ndarray,
    span_histogram: np.ndarray,
    coverage_output: np.ndarray,
    *,
    native_extension,
    analysis_queue: _FastLaneAnalysisQueue | None,
    defer_coverage: bool = False,
) -> None:
    if analysis_queue is not None:
        analysis_queue.submit(
            packed_missing,
            lane_histogram,
            span_histogram,
            coverage_output,
            defer_coverage=defer_coverage,
        )
        return
    native_extension.accumulate_packed_lane_hole_run_histograms_into(
        packed_missing,
        lane_histogram,
        span_histogram,
        coverage_output,
    )


def _unfused_fast_band(
    rendered: torch.Tensor, filled: torch.Tensor, unresolved: torch.Tensor, background: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fine_repair = _fast_repair_lane_flags(filled, unresolved, background=background)
    downsampled = _downsample_subpixel_band(rendered)
    lane_counts = torch.stack(
        (
            torch.count_nonzero(filled),
            torch.count_nonzero(unresolved),
            torch.count_nonzero(~torch.isfinite(downsampled)),
        )
    )
    return fine_repair, downsampled, lane_counts


def _check_finite_fast_band(counts: np.ndarray) -> None:
    if counts[2] != 0:
        raise ValueError("Rendered image contains non-finite values")


def _store_prefill_flags(repair: np.ndarray, coverage: np.ndarray) -> None:
    repair[(coverage > 0) & (coverage < HORIZONTAL_SUBPIXELS)] |= np.uint8(0x01)
    repair[coverage == 0] |= np.uint8(0x02)


def _store_packed_fast_band(
    rendered: torch.Tensor,
    prefill: torch.Tensor,
    filled: torch.Tensor,
    unresolved: torch.Tensor,
    lane_histogram: np.ndarray,
    span_histogram: np.ndarray,
    coverage: np.ndarray,
    repair: np.ndarray,
    output: np.ndarray,
    *,
    background: bool,
    native_extension,
    lane_weights: torch.Tensor,
    analysis_queue: _FastLaneAnalysisQueue | None,
) -> tuple[int, int]:
    from .fast_output_triton import triton_fast_band_payload

    payload = triton_fast_band_payload(
        rendered,
        prefill,
        filled,
        unresolved,
        background=background,
        include_coverage_flags=True,
    )
    if payload is not None:
        host_image, packed_missing, host_repair, counts = _download_fast_cuda_payload(
            payload, output.shape, coverage.shape
        )
    else:
        fine_repair, downsampled, lane_counts = _unfused_fast_band(
            rendered, filled, unresolved, background
        )
        host_image, packed_missing, host_repair, counts = _download_fast_cuda_band(
            downsampled, prefill, lane_weights, fine_repair, lane_counts
        )
    _analyze_packed_lane_validity(
        packed_missing,
        lane_histogram,
        span_histogram,
        coverage,
        native_extension=native_extension,
        analysis_queue=analysis_queue,
        defer_coverage=payload is not None,
    )
    _check_finite_fast_band(counts)
    if payload is not None:
        repair[:] = host_repair
    else:
        _store_prefill_flags(repair, coverage)
    output[:] = host_image
    if payload is None:
        repair |= host_repair
    return int(counts[0]), int(counts[1])


def _store_unpacked_fast_band(
    rendered: torch.Tensor,
    prefill: torch.Tensor,
    filled: torch.Tensor,
    unresolved: torch.Tensor,
    lane_histogram: np.ndarray,
    span_histogram: np.ndarray,
    coverage: np.ndarray,
    repair: np.ndarray,
    output: np.ndarray,
    *,
    background: bool,
    native_extension,
    lane_weights: torch.Tensor | None,
    analysis_queue: _FastLaneAnalysisQueue | None,
    cuda: bool,
) -> tuple[int, int]:
    _analyze_lane_validity(
        prefill,
        lane_histogram,
        span_histogram,
        coverage,
        native_extension=native_extension,
        lane_weights=lane_weights,
        analysis_queue=analysis_queue,
    )
    fine_repair, downsampled, lane_counts = _unfused_fast_band(
        rendered, filled, unresolved, background
    )
    counts = None
    if not cuda:
        counts = lane_counts.detach().numpy()
        _check_finite_fast_band(counts)
    _store_prefill_flags(repair, coverage)
    output[:] = _convert_image_band(downsampled, output.dtype)
    repair |= fine_repair.detach().cpu().numpy()
    if counts is None:
        counts = lane_counts.detach().cpu().numpy()
        _check_finite_fast_band(counts)
    return int(counts[0]), int(counts[1])


def _fast_repair_lane_flags(
    filled: torch.Tensor,
    unresolved: torch.Tensor,
    *,
    background: bool,
) -> torch.Tensor:
    """Collapse fine-lane outcomes without losing overlapping state bits."""

    filled_flags = torch.any(filled, dim=2).to(dtype=torch.uint8)
    filled_flags.mul_(0x04)
    unresolved_flags = torch.any(unresolved, dim=2).to(dtype=torch.uint8)
    unresolved_flags.mul_(0x48 if background else 0x40)
    filled_flags.bitwise_or_(unresolved_flags)
    return filled_flags


def _attach_fast_performance(
    result,
    *,
    geometry_seconds: float,
    offset_seconds: float,
    total_started: float,
):
    return replace(
        result,
        performance_seconds={
            "fast_geometry": float(geometry_seconds),
            "fast_offset_build": float(offset_seconds),
            **result.performance_seconds,
            "total": time.perf_counter() - total_started,
        },
    )


class StereoRenderer:
    """Render stereo pairs without full-frame device-resident splat intermediates."""

    def __init__(
        self,
        device: str | torch.device | None = None,
        *,
        temporary_budget_bytes: int | None = None,
        profile: bool = False,
    ) -> None:
        temporary_budget_is_default = temporary_budget_bytes is None
        if temporary_budget_bytes is None:
            temporary_budget_bytes = GPU_TEMP_BUDGET
        if temporary_budget_bytes <= 0:
            raise ValueError("Temporary GPU budget must be positive")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.temporary_budget_bytes = temporary_budget_bytes
        self._temporary_budget_is_default = temporary_budget_is_default
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
        if settings.stereo_render_mode == "quality":
            return self._render_relative_quality(source, canonical_values, settings)
        return self._render_relative_fast(source, canonical_values, settings)

    def _render_relative_fast(
        self,
        source: np.ndarray,
        canonical_values: np.ndarray,
        settings: StereoRenderSettings,
    ) -> StereoRenderResult:
        """Run the unchanged Revision 5 relative renderer."""

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

    def _render_relative_quality(
        self,
        source: np.ndarray,
        canonical_values: np.ndarray,
        settings: StereoRenderSettings,
    ) -> StereoRenderResult:
        """Load the optional production Quality path only when selected."""

        from .quality.renderer import render_relative_quality_public

        return render_relative_quality_public(
            renderer=self,
            frame=source,
            canonical=canonical_values,
            settings=settings,
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
        bytes_per_pixel = (
            TRITON_SPLAT_BYTES_PER_PIXEL
            if _triton_splat_available(self.device)
            else SPLAT_BYTES_PER_PIXEL
        )
        temporary_budget_bytes = _fast_temporary_budget_bytes(
            self,
            render_width=width,
            render_height=height,
            bytes_per_pixel=bytes_per_pixel,
        )
        first_band_height = calculate_band_height(
            width,
            height,
            temporary_budget_bytes,
            bytes_per_pixel=bytes_per_pixel,
        )

        render_failed = False
        try:
            result = self._render_with_band_height(
                source,
                geometry,
                eye_offsets,
                settings,
                first_band_height,
            )
        except torch.cuda.OutOfMemoryError:
            render_failed = True
        # Drop the failed attempt's traceback before releasing and retrying.
        if render_failed:
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

    def render_relative_fast_compact(
        self,
        frame: np.ndarray,
        canonical: np.ndarray,
        settings: StereoRenderSettings,
    ):
        """Render legacy RGB while retaining only compact diagnostic planes."""

        total_started = time.perf_counter()
        source = np.asarray(frame)
        canonical_values = np.asarray(canonical, dtype=np.float32)
        self._validate_inputs(source, canonical_values, settings)
        if settings.stereo_render_mode != "fast":
            raise ValueError("Fast compact rendering requires fast settings")
        source = np.ascontiguousarray(source)
        height, width = source.shape[:2]
        geometry_started = time.perf_counter()
        if (
            not math.isfinite(settings.stereo_strength)
            or not 0.0 <= settings.stereo_strength <= 5.0
        ):
            raise ValueError("stereo_strength must be finite and within [0, 5]")
        if not math.isfinite(settings.convergence) or not 0.0 <= settings.convergence <= 1.0:
            raise ValueError("convergence must be finite and within [0, 1]")
        # Relative offsets do not consume the common float64 disparity plane.
        legacy_near_score = self._resize_canonical(canonical_values, (height, width)).clone()
        geometry = _FastRelativeGeometry(
            near_score=legacy_near_score.numpy(),
            source_valid=np.ones((height, width), dtype=np.bool_),
        )
        geometry_seconds = time.perf_counter() - geometry_started
        del frame, canonical, canonical_values
        offset_started = time.perf_counter()
        eye_offsets = (
            _FastRelativeOffsets(legacy_near_score, settings)
            if self.device.type == "cuda"
            else calculate_eye_sample_offsets(legacy_near_score, settings)
        )
        offset_seconds = time.perf_counter() - offset_started
        result = self._render_fast_compact_core(
            source,
            geometry,
            eye_offsets,
            StereoSplatSettings(
                max_eye_shift_fraction=settings.stereo_strength / 200.0,
                occlusion_fill=settings.occlusion_fill,
            ),
        )
        if isinstance(eye_offsets, _FastRelativeOffsets):
            offset_seconds += eye_offsets.build_seconds
        del source, geometry, legacy_near_score, eye_offsets
        return _attach_fast_performance(
            result,
            geometry_seconds=geometry_seconds,
            offset_seconds=offset_seconds,
            total_started=total_started,
        )

    def render_metric_primitives_fast_compact(
        self,
        frame: np.ndarray,
        primitives,
        settings: StereoSplatSettings,
    ):
        """Consume one owned metric primitive without retaining native arrays."""

        total_started = time.perf_counter()
        metric = primitives.metric
        source = np.asarray(frame)
        geometry_started = time.perf_counter()
        geometry, metric_stats = build_metric_geometry(
            metric.inverse_depth,
            metric.valid,
            metric.focal_x_normalized,
            (int(source.shape[0]), int(source.shape[1])),
            virtual_baseline_mm=float(primitives.virtual_baseline_mm),
            convergence_distance_m=float(primitives.convergence_distance_m),
            max_disparity_percent=float(primitives.max_disparity_percent),
            retained_crop_width=primitives.retained_crop_width,
        )
        geometry_seconds = time.perf_counter() - geometry_started
        del primitives, metric
        self._validate_geometry_inputs(source, geometry, settings)
        source = np.ascontiguousarray(source)
        del frame
        offset_started = time.perf_counter()
        eye_offsets = calculate_geometry_eye_sample_offsets(geometry.total_disparity_fraction)
        offset_seconds = time.perf_counter() - offset_started
        result = self._render_fast_compact_core(
            source,
            geometry,
            eye_offsets,
            settings,
            metric_stats=metric_stats,
        )
        del source, geometry, eye_offsets
        return _attach_fast_performance(
            result,
            geometry_seconds=geometry_seconds,
            offset_seconds=offset_seconds,
            total_started=total_started,
        )

    def render_geometry_fast_compact(
        self,
        frame: np.ndarray,
        geometry: StereoGeometryFrame,
        settings: StereoSplatSettings,
        *,
        metric_stats=None,
    ):
        """Render common Fast geometry without public boolean-mask allocations."""

        total_started = time.perf_counter()
        source = np.asarray(frame)
        self._validate_geometry_inputs(source, geometry, settings)
        source = np.ascontiguousarray(source)
        offset_started = time.perf_counter()
        eye_offsets = calculate_geometry_eye_sample_offsets(geometry.total_disparity_fraction)
        offset_seconds = time.perf_counter() - offset_started
        result = self._render_fast_compact_core(
            source,
            geometry,
            eye_offsets,
            settings,
            metric_stats=metric_stats,
        )
        return _attach_fast_performance(
            result,
            geometry_seconds=0.0,
            offset_seconds=offset_seconds,
            total_started=total_started,
        )

    def _render_fast_compact_core(
        self,
        source: np.ndarray,
        geometry: StereoGeometryFrame | _FastRelativeGeometry,
        eye_offsets: tuple[np.ndarray, np.ndarray] | _FastRelativeOffsets,
        settings: StereoSplatSettings,
        *,
        metric_stats=None,
    ):
        height, width = source.shape[:2]
        bytes_per_pixel = (
            TRITON_SPLAT_BYTES_PER_PIXEL
            if _triton_splat_available(self.device)
            else SPLAT_BYTES_PER_PIXEL
        )
        temporary_budget_bytes = _fast_temporary_budget_bytes(
            self,
            render_width=width,
            render_height=height,
            bytes_per_pixel=bytes_per_pixel,
        )
        first_band_height = calculate_band_height(
            width,
            height,
            temporary_budget_bytes,
            bytes_per_pixel=bytes_per_pixel,
        )
        try:
            return self._render_fast_compact_with_band_height(
                source,
                geometry,
                eye_offsets,
                settings,
                first_band_height,
                metric_stats=metric_stats,
            )
        except torch.cuda.OutOfMemoryError:
            pass
        # Drop the failed attempt's traceback before releasing and retrying.
        self._release_after_oom()
        fallback_band_height = calculate_band_height(
            width,
            height,
            self.temporary_budget_bytes,
            bytes_per_pixel=bytes_per_pixel,
        )
        retry_band_height = max(
            1,
            min(first_band_height // 2, fallback_band_height),
        )
        try:
            return self._render_fast_compact_with_band_height(
                source,
                geometry,
                eye_offsets,
                settings,
                retry_band_height,
                metric_stats=metric_stats,
                preload_device_frame=False,
            )
        except torch.cuda.OutOfMemoryError as error:
            self._release_after_oom()
            raise RuntimeError(
                "CUDA stereo rendering ran out of memory for frame "
                f"{width}x{height}; attempted band heights "
                f"{first_band_height} and {retry_band_height}"
            ) from error

    def _render_fast_compact_with_band_height(
        self,
        source: np.ndarray,
        geometry: StereoGeometryFrame | _FastRelativeGeometry,
        eye_offsets: tuple[np.ndarray, np.ndarray] | _FastRelativeOffsets,
        settings: StereoSplatSettings,
        band_height: int,
        *,
        metric_stats=None,
        preload_device_frame: bool = True,
    ):
        from .quality.types import CompactStereoRenderResult

        preloaded = (
            self._preload_fast_device_frame(source, geometry, eye_offsets)
            if preload_device_frame
            else None
        )
        if preloaded is None:
            device_frame = None
            device_offsets: tuple[torch.Tensor | None, torch.Tensor | None] = (
                None,
                None,
            )
            host_offsets = (
                eye_offsets.host_offsets()
                if isinstance(eye_offsets, _FastRelativeOffsets)
                else eye_offsets
            )
        else:
            device_frame, loaded_offsets = preloaded
            device_offsets = loaded_offsets
            host_offsets = (None, None)

        native_extension = _native_lane_analyzer() if self.device.type == "cuda" else None
        analysis_context = (
            _FastLaneAnalysisQueue(native_extension)
            if native_extension is not None
            else nullcontext(None)
        )
        with analysis_context as analysis_queue:
            left_started = time.perf_counter()
            left = self._render_fast_eye_compact(
                source,
                geometry,
                host_offsets[0],
                settings,
                band_height,
                device_frame=device_frame,
                device_offsets=device_offsets[0],
                analysis_queue=analysis_queue,
            )
            left_seconds = time.perf_counter() - left_started
            right_started = time.perf_counter()
            right = self._render_fast_eye_compact(
                source,
                geometry,
                host_offsets[1],
                settings,
                band_height,
                device_frame=device_frame,
                device_offsets=device_offsets[1],
                analysis_queue=analysis_queue,
            )
        right_seconds = time.perf_counter() - right_started
        return CompactStereoRenderResult(
            left_image=left[0],
            right_image=right[0],
            left_coverage_count=left[1],
            right_coverage_count=right[1],
            left_repair_bits=left[2],
            right_repair_bits=right[2],
            metric_stats=metric_stats,
            left_hole_runs=left[3].hole_runs,
            right_hole_runs=right[3].hole_runs,
            left_fast_diagnostics=left[3],
            right_fast_diagnostics=right[3],
            performance_seconds={
                "left_eye": left_seconds,
                "right_eye": right_seconds,
            },
        )

    def _render_fast_eye_compact(
        self,
        source: np.ndarray,
        geometry: StereoGeometryFrame | _FastRelativeGeometry,
        sample_offsets: np.ndarray | None,
        settings: StereoSplatSettings,
        band_height: int,
        *,
        device_frame: _FastDeviceFrame | None = None,
        device_offsets: torch.Tensor | None = None,
        analysis_queue: _FastLaneAnalysisQueue | None = None,
    ):
        from .quality.types import FastEyeDiagnostics, HoleRunDiagnostics

        assert sample_offsets is not None or device_offsets is not None
        height, width = source.shape[:2]
        output = np.empty_like(source)
        coverage_output = np.empty((height, width), dtype=np.uint8)
        repair_output = np.zeros((height, width), dtype=np.uint8)
        lane_histogram = np.zeros(16 * (width + 1), dtype=np.uint64)
        span_histogram = np.zeros(width + 1, dtype=np.uint64)
        local_filled_lane_count = 0
        final_unresolved_lane_count = 0
        max_gap_width = math.ceil(width * settings.max_eye_shift_fraction) + 2
        max_gap_samples = HORIZONTAL_SUBPIXELS * max_gap_width
        native_extension = _native_lane_analyzer()
        lane_weights = (
            _lane_mask_weights(self.device)
            if native_extension is not None and self.device.type == "cuda"
            else None
        )
        batched_download = lane_weights is not None and source.dtype == np.dtype(np.uint8)

        with torch.inference_mode():
            for start_row in range(0, height, band_height):
                end_row = min(height, start_row + band_height)
                if device_frame is None:
                    source_band = torch.from_numpy(
                        np.ascontiguousarray(source[start_row:end_row])
                    ).to(self.device)
                    near_score_band = torch.from_numpy(
                        np.array(
                            geometry.near_score[start_row:end_row],
                            copy=True,
                            order="C",
                        )
                    ).to(self.device)
                    source_valid_band = torch.from_numpy(
                        np.array(
                            geometry.source_valid[start_row:end_row],
                            copy=True,
                            order="C",
                        )
                    ).to(self.device)
                else:
                    source_band = device_frame.source[start_row:end_row]
                    near_score_band = device_frame.near_score[start_row:end_row]
                    source_valid_band = device_frame.source_valid[start_row:end_row]
                if device_offsets is None:
                    assert sample_offsets is not None
                    offset_band = self._transfer_offset_band(sample_offsets[start_row:end_row])
                else:
                    offset_band = device_offsets[start_row:end_row]
                splat = _forward_splat_band_prevalidated(
                    source_band,
                    near_score_band,
                    offset_band,
                    source_index_offset=start_row * width,
                    source_valid=source_valid_band,
                )
                if settings.occlusion_fill == "background":
                    rendered, hole_mask, fillable = _fill_background_band_state(
                        splat,
                        max_gap_samples=max_gap_samples,
                    )
                else:
                    rendered = splat.colour
                    hole_mask = ~splat.valid
                    fillable = torch.zeros_like(splat.valid)
                row_count = end_row - start_row
                prefill = splat.valid.reshape(row_count, width, HORIZONTAL_SUBPIXELS)
                filled = fillable.reshape(row_count, width, HORIZONTAL_SUBPIXELS)
                unresolved = hole_mask.reshape(row_count, width, HORIZONTAL_SUBPIXELS)
                if batched_download:
                    assert lane_weights is not None
                    counts = _store_packed_fast_band(
                        rendered,
                        prefill,
                        filled,
                        unresolved,
                        lane_histogram,
                        span_histogram,
                        coverage_output[start_row:end_row],
                        repair_output[start_row:end_row],
                        output[start_row:end_row],
                        background=settings.occlusion_fill == "background",
                        native_extension=native_extension,
                        lane_weights=lane_weights,
                        analysis_queue=analysis_queue,
                    )
                else:
                    counts = _store_unpacked_fast_band(
                        rendered,
                        prefill,
                        filled,
                        unresolved,
                        lane_histogram,
                        span_histogram,
                        coverage_output[start_row:end_row],
                        repair_output[start_row:end_row],
                        output[start_row:end_row],
                        background=settings.occlusion_fill == "background",
                        native_extension=native_extension,
                        lane_weights=lane_weights,
                        analysis_queue=analysis_queue,
                        cuda=self.device.type == "cuda",
                    )
                local_filled_lane_count += counts[0]
                final_unresolved_lane_count += counts[1]
                del source_band, near_score_band, source_valid_band, offset_band, splat
                del rendered, hole_mask, fillable, prefill, filled, unresolved
        if analysis_queue is not None:
            analysis_queue.finish_pending()
        hole_runs = HoleRunDiagnostics(
            lane_count_histogram=lane_histogram,
            touched_pixel_span_histogram=span_histogram,
        )
        diagnostics = FastEyeDiagnostics(
            hole_runs=hole_runs,
            local_filled_lane_count=local_filled_lane_count,
            final_unresolved_lane_count=final_unresolved_lane_count,
        )
        return output, coverage_output, repair_output, diagnostics

    def _preload_fast_device_frame(
        self,
        source: np.ndarray,
        geometry: StereoGeometryFrame | _FastRelativeGeometry,
        eye_offsets: tuple[np.ndarray, np.ndarray] | _FastRelativeOffsets,
    ) -> tuple[_FastDeviceFrame, tuple[torch.Tensor, torch.Tensor]] | None:
        if self.device.type != "cuda":
            return None
        device_source = None
        device_near_score = None
        device_source_valid = None
        left_offsets = None
        right_offsets = None
        loaded_offsets = None
        preload_failed = False
        try:
            device_source = torch.from_numpy(source).to(self.device)
            device_near_score = torch.from_numpy(
                np.array(geometry.near_score, copy=True, order="C")
            ).to(self.device)
            device_source_valid = torch.from_numpy(
                np.array(geometry.source_valid, copy=True, order="C")
            ).to(self.device)
            if isinstance(eye_offsets, _FastRelativeOffsets):
                loaded_offsets = eye_offsets.device_offsets(device_near_score, profile=self.profile)
            if loaded_offsets is None:
                host_offsets = (
                    eye_offsets.host_offsets()
                    if isinstance(eye_offsets, _FastRelativeOffsets)
                    else eye_offsets
                )
                left_offsets = self._transfer_offset_band(host_offsets[0])
                right_offsets = self._transfer_offset_band(host_offsets[1])
            else:
                left_offsets, right_offsets = loaded_offsets
        except torch.cuda.OutOfMemoryError:
            device_source = None
            device_near_score = None
            device_source_valid = None
            left_offsets = None
            right_offsets = None
            loaded_offsets = None
            preload_failed = True
        if preload_failed:
            # Helper tracebacks may retain CUDA inputs even after locals are cleared.
            self._release_after_oom()
            return None
        assert device_source is not None
        assert device_near_score is not None
        assert device_source_valid is not None
        assert left_offsets is not None
        assert right_offsets is not None
        return (
            _FastDeviceFrame(
                source=device_source,
                near_score=device_near_score,
                source_valid=device_source_valid,
            ),
            (left_offsets, right_offsets),
        )

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
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
