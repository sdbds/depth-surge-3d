"""Compact one-pass offline Quality stereo renderer."""

from __future__ import annotations

import atexit
import gc
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait as wait_futures
from dataclasses import dataclass
from typing import Callable, Literal

import cv2
import numpy as np
import torch

from ...core.host_memory import dynamic_host_budget_bytes
from ...core.resource_errors import QualityHostBudgetError
from ..forward_splat import HORIZONTAL_SUBPIXELS
from ..stereo_renderer import (
    StereoRenderResult,
    StereoRenderer,
    StereoRenderSettings,
    _convert_image_band,
    _downsample_subpixel_band,
)
from . import native_kd
from .geometry import (
    QualityGeometryResult,
    build_geometry_eye_offsets_into,
    build_metric_quality_geometry,
    build_relative_eye_offsets_into,
    build_relative_quality_geometry,
)
from .types import (
    REPAIR_RECORD_ITEMSIZE,
    CompactStereoRenderResult,
    HoleRunDiagnostics,
    MetricStereoPrimitiveInput,
    QualityRepairPlanSummary,
    QualityStereoControls,
)
from .visibility import QualitySplatBand, _quality_splat_band_prevalidated


QUALITY_HOST_BUDGET_BYTES = dynamic_host_budget_bytes(
    fraction=0.45,
    minimum=2 * 1024 * 1024 * 1024,
    maximum=4 * 1024 * 1024 * 1024,
)
QUALITY_RUNTIME_OVERHEAD_BYTES = 16 * 1024 * 1024
QUALITY_NATIVE_SPAWNED_WORKER_CAP = 7
QUALITY_PYTHON_REPAIR_WORKER_COUNT = 1
QUALITY_AUXILIARY_THREAD_STACK_BYTES = 8 * 1024 * 1024
QUALITY_AUXILIARY_THREAD_COUNT = (
    QUALITY_NATIVE_SPAWNED_WORKER_CAP + QUALITY_PYTHON_REPAIR_WORKER_COUNT
)
QUALITY_AUXILIARY_THREAD_BYTES = (
    QUALITY_AUXILIARY_THREAD_COUNT * QUALITY_AUXILIARY_THREAD_STACK_BYTES
)
QUALITY_SPLAT_BYTES_PER_PIXEL = 1536
QUALITY_CUDA_BUDGET_CAP_BYTES = 1536 * 1024 * 1024
QUALITY_CUDA_FREE_BUDGET_DIVISOR = 4
QUALITY_INPAINT_ROI_AREA_NUMERATOR = 9
QUALITY_INPAINT_ROI_AREA_DENOMINATOR = 10

_LEFT_REPAIR_EXECUTOR_LOCK = threading.Lock()
_LEFT_REPAIR_EXECUTOR: ThreadPoolExecutor | None = None


def _get_left_repair_executor() -> ThreadPoolExecutor:
    global _LEFT_REPAIR_EXECUTOR
    with _LEFT_REPAIR_EXECUTOR_LOCK:
        if _LEFT_REPAIR_EXECUTOR is None:
            _LEFT_REPAIR_EXECUTOR = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="quality-left-repair",
            )
            atexit.register(_shutdown_left_repair_executor)
        return _LEFT_REPAIR_EXECUTOR


def _shutdown_left_repair_executor() -> None:
    global _LEFT_REPAIR_EXECUTOR
    with _LEFT_REPAIR_EXECUTOR_LOCK:
        executor = _LEFT_REPAIR_EXECUTOR
        _LEFT_REPAIR_EXECUTOR = None
    if executor is not None:
        executor.shutdown(wait=False)


def _calculate_quality_band_height(
    render_width: int,
    render_height: int,
    temporary_budget_bytes: int,
) -> int:
    if render_width <= 0 or render_height <= 0:
        raise ValueError("Quality render width and height must be positive")
    if temporary_budget_bytes <= 0:
        raise ValueError("Quality temporary GPU budget must be positive")
    rows = temporary_budget_bytes // (render_width * QUALITY_SPLAT_BYTES_PER_PIXEL)
    return min(render_height, max(1, rows))


def _quality_temporary_budget_bytes(renderer: StereoRenderer) -> int:
    """Grow only the default budget while retaining most live free VRAM."""

    configured = int(renderer.temporary_budget_bytes)
    if (
        renderer.device.type != "cuda"
        or not getattr(renderer, "_temporary_budget_is_default", False)
        or configured >= QUALITY_CUDA_BUDGET_CAP_BYTES
    ):
        return configured
    try:
        free_bytes, _ = torch.cuda.mem_get_info(renderer.device)
    except (AssertionError, RuntimeError):
        return configured
    adaptive = min(
        QUALITY_CUDA_BUDGET_CAP_BYTES,
        int(free_bytes) // QUALITY_CUDA_FREE_BUDGET_DIVISOR,
    )
    return max(configured, adaptive)


def _quality_public_memory_preflight(
    *,
    render_shape: tuple[int, int],
    native_shape: tuple[int, int],
    occlusion_fill: Literal["none", "background"],
    materialize_public_masks: bool,
    geometry_mode: Literal["relative", "metric"] = "relative",
) -> dict[str, int]:
    height, width = render_shape
    native_height, native_width = native_shape
    if min(height, width, native_height, native_width) <= 0:
        raise ValueError("Quality render and native shapes must be positive")
    output_pixels = height * width
    native_pixels = native_height * native_width
    if 16 * output_pixels > np.iinfo(np.uint32).max:
        raise QualityHostBudgetError.for_limit(
            "quality-index-range",
            16 * output_pixels,
            int(np.iinfo(np.uint32).max),
        )
    if native_pixels + 1 > np.iinfo(np.uint32).max:
        raise QualityHostBudgetError.for_limit(
            "quality-native-index-range",
            native_pixels + 1,
            int(np.iinfo(np.uint32).max),
        )
    histogram_bytes = 272 * (width + 1)
    runtime = QUALITY_RUNTIME_OVERHEAD_BYTES
    auxiliary_threads = QUALITY_AUXILIARY_THREAD_BYTES
    geometry_payload = (
        21 * output_pixels + 9 * native_pixels
        if geometry_mode == "relative"
        else 29 * output_pixels + 18 * native_pixels
    )
    phases = {
        "quality-lowres-region": (
            3 * output_pixels + 22 * native_pixels + runtime + auxiliary_threads
        ),
        "quality-geometry-nearest": (
            geometry_payload
            + 40 * output_pixels
            + 10 * native_pixels  # KD members/offsets, allowed regions, and native seen flags.
            + 5
            + runtime
            + auxiliary_threads
        ),
        "quality-region": (68 * output_pixels + 18 * native_pixels + runtime + auxiliary_threads),
        "quality-geometry-matte": (
            68 * output_pixels + 18 * native_pixels + runtime + auxiliary_threads
        ),
    }
    if occlusion_fill == "background":
        visibility = (
            40 * output_pixels + 9 * native_pixels + histogram_bytes + runtime + auxiliary_threads
        )
        proxy = (
            100 * output_pixels + 9 * native_pixels + histogram_bytes + runtime + auxiliary_threads
        )
        phases["quality-background-visibility"] = visibility
        phases["quality-background-proxy"] = proxy
        if materialize_public_masks:
            phases["quality-public"] = max(
                proxy,
                14 * output_pixels + histogram_bytes + runtime + auxiliary_threads,
            )
    else:
        visibility = (
            40 * output_pixels + 9 * native_pixels + histogram_bytes + runtime + auxiliary_threads
        )
        phases["quality-none-visibility"] = visibility
        if materialize_public_masks:
            phases["quality-none-public"] = max(
                visibility,
                14 * output_pixels + histogram_bytes + runtime + auxiliary_threads,
            )
    phase, required = max(phases.items(), key=lambda item: item[1])
    if required > QUALITY_HOST_BUDGET_BYTES:
        label = "quality-public" if materialize_public_masks else phase
        raise QualityHostBudgetError.for_limit(
            label,
            required,
            QUALITY_HOST_BUDGET_BYTES,
        )
    return phases


@dataclass(frozen=True)
class _CompactEyeResult:
    image: np.ndarray
    coverage_count: np.ndarray
    repair_bits: np.ndarray
    plan: QualityRepairPlanSummary | None
    hole_runs: HoleRunDiagnostics | None = None


@dataclass(frozen=True, slots=True)
class PreparedRelativeQualityFrame:
    source: np.ndarray
    geometry: QualityGeometryResult
    settings: StereoRenderSettings
    indexed_sample_count: int
    prepare_seconds: float


@dataclass(frozen=True, slots=True)
class PreparedMetricQualityFrame:
    source: np.ndarray
    geometry: QualityGeometryResult
    controls: QualityStereoControls
    indexed_sample_count: int
    prepare_seconds: float


@dataclass(frozen=True)
class _UnrepairedEyeResult:
    image: np.ndarray
    coverage_count: np.ndarray
    hole_runs: HoleRunDiagnostics


@dataclass(frozen=True)
class _DeferredUnrepairedEyeResult:
    """Single-copy CUDA splat payload whose CPU coverage/hole-run unpack runs in Stage 3."""

    image: np.ndarray
    packed_missing: np.ndarray


@dataclass(frozen=True, slots=True)
class SplattedQualityFrame:
    """GPU-splatted Quality stereo frame awaiting CPU background proxy repair."""

    source: np.ndarray
    occlusion_fill: Literal["none", "background"]
    controls_limit_px: int
    predicted_gap_px: int
    max_neighbour_abs_q_jump_px: float
    indexed_sample_count: int
    geometry_query_count: int
    geometry_visited_nodes_total: int
    geometry_visited_nodes_max: int
    metric_stats: object | None
    geometry_performance_seconds: dict[str, float]
    prepare_seconds: float
    splat_seconds: float
    left_prefix_seconds: float
    right_prefix_seconds: float
    left_pre_repair: _UnrepairedEyeResult | _DeferredUnrepairedEyeResult | None = None
    right_pre_repair: _UnrepairedEyeResult | _DeferredUnrepairedEyeResult | None = None
    left_none: _CompactEyeResult | None = None
    right_none: _CompactEyeResult | None = None


@dataclass(frozen=True)
class _QualityDeviceFrame:
    source: torch.Tensor
    near_score: torch.Tensor
    source_valid: torch.Tensor


class _PinnedTransferCache:
    """Reusable pinned host buffers for fast CUDA H2D/D2H without per-frame VirtualAlloc."""

    def __init__(self) -> None:
        self.shape: tuple[int, int] | None = None
        self.h2d_source: torch.Tensor | None = None
        self.h2d_near: torch.Tensor | None = None
        self.h2d_valid: torch.Tensor | None = None
        self.d2h_rgb: torch.Tensor | None = None
        self.d2h_packed: torch.Tensor | None = None

    def ensure_h2d(
        self, height: int, width: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if self.shape != (height, width) or self.h2d_source is None:
            self.shape = (height, width)
            self.h2d_source = None
            self.h2d_near = None
            self.h2d_valid = None
            self.d2h_rgb = None
            self.d2h_packed = None
            try:
                self.h2d_source = torch.empty(
                    (height, width, 3), dtype=torch.uint8, pin_memory=True
                )
                self.h2d_near = torch.empty((height, width), dtype=torch.float32, pin_memory=True)
                self.h2d_valid = torch.empty((height, width), dtype=torch.bool, pin_memory=True)
            except RuntimeError:
                self.h2d_source = None
                self.h2d_near = None
                self.h2d_valid = None
                return None
        assert (
            self.h2d_source is not None and self.h2d_near is not None and self.h2d_valid is not None
        )
        return self.h2d_source, self.h2d_near, self.h2d_valid

    def ensure_d2h(self, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.shape != (height, width):
            self.shape = (height, width)
            self.h2d_source = None
            self.h2d_near = None
            self.h2d_valid = None
            self.d2h_rgb = None
            self.d2h_packed = None
        if self.d2h_rgb is None or self.d2h_packed is None:
            try:
                self.d2h_rgb = torch.empty((height, width, 3), dtype=torch.uint8, pin_memory=True)
                self.d2h_packed = torch.empty((height, width), dtype=torch.int16, pin_memory=True)
            except RuntimeError:
                self.d2h_rgb = None
                self.d2h_packed = None
                return None
        return self.d2h_rgb, self.d2h_packed


_PINNED_TRANSFER_TLS = threading.local()


def _get_pinned_transfer_cache() -> _PinnedTransferCache:
    cache = getattr(_PINNED_TRANSFER_TLS, "cache", None)
    if cache is None:
        cache = _PinnedTransferCache()
        _PINNED_TRANSFER_TLS.cache = cache
    return cache


@dataclass
class _FrameBandState:
    initial_height: int
    retry_height: int
    current_height: int
    retry_used: bool = False


def _release_after_oom(renderer: StereoRenderer) -> None:
    renderer._release_after_oom()
    gc.collect()


def _retry_after_oom(
    owner: StereoRenderer,
    state: _FrameBandState,
    operation,
    **kwargs,
):
    while True:
        try:
            return operation(band_height=state.current_height, **kwargs)
        except torch.cuda.OutOfMemoryError as error:
            _release_after_oom(owner)
            if state.retry_used:
                width = int(kwargs["source"].shape[1])
                height = int(kwargs["source"].shape[0])
                raise RuntimeError(
                    "CUDA Quality stereo rendering ran out of memory for frame "
                    f"{width}x{height}; attempted band heights "
                    f"{state.initial_height} and {state.retry_height}"
                ) from error
            state.retry_used = True
            state.current_height = state.retry_height


def _preload_quality_device_frame(
    renderer: StereoRenderer,
    source: np.ndarray,
    geometry: QualityGeometryResult,
) -> _QualityDeviceFrame | None:
    if renderer.device.type != "cuda":
        return None
    device_source = None
    device_near_score = None
    device_source_valid = None
    try:
        height, width = int(source.shape[0]), int(source.shape[1])
        pinned = (
            _get_pinned_transfer_cache().ensure_h2d(height, width)
            if source.dtype == np.uint8 and geometry.near_score.dtype == np.float32
            else None
        )
        if pinned is not None:
            pin_src, pin_near, pin_valid = pinned
            pin_src.numpy()[...] = source
            pin_near.numpy()[...] = geometry.near_score
            pin_valid.numpy()[...] = geometry.source_valid
            device_source = pin_src.to(renderer.device, non_blocking=True)
            device_near_score = pin_near.to(renderer.device, non_blocking=True)
            device_source_valid = pin_valid.to(renderer.device, non_blocking=True)
        else:
            device_source = torch.from_numpy(source).to(renderer.device)
            device_near_score = torch.from_numpy(
                np.array(geometry.near_score, copy=True, order="C")
            ).to(renderer.device)
            device_source_valid = torch.from_numpy(
                np.array(geometry.source_valid, copy=True, order="C")
            ).to(renderer.device)
    except torch.cuda.OutOfMemoryError:
        device_source = None
        device_near_score = None
        device_source_valid = None
        _release_after_oom(renderer)
        return None
    assert device_source is not None
    assert device_near_score is not None
    assert device_source_valid is not None
    return _QualityDeviceFrame(
        source=device_source,
        near_score=device_near_score,
        source_valid=device_source_valid,
    )


def _quality_device_band(
    renderer: StereoRenderer,
    source: np.ndarray,
    geometry: QualityGeometryResult,
    offsets: np.ndarray | torch.Tensor,
    start_row: int,
    end_row: int,
    *,
    device_frame: _QualityDeviceFrame | None = None,
    device_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(offsets, torch.Tensor):
        device_offsets = offsets
    if device_frame is None:
        source_band = torch.from_numpy(np.ascontiguousarray(source[start_row:end_row])).to(
            renderer.device
        )
        near_score_band = torch.from_numpy(
            np.array(geometry.near_score[start_row:end_row], copy=True, order="C")
        ).to(renderer.device)
        source_valid_band = torch.from_numpy(
            np.array(geometry.source_valid[start_row:end_row], copy=True, order="C")
        ).to(renderer.device)
    else:
        source_band = device_frame.source[start_row:end_row]
        near_score_band = device_frame.near_score[start_row:end_row]
        source_valid_band = device_frame.source_valid[start_row:end_row]
    if device_offsets is None:
        assert isinstance(offsets, np.ndarray)
        offset_band = renderer._transfer_offset_band(offsets[start_row:end_row])
    else:
        offset_band = device_offsets[start_row:end_row]
    return source_band, near_score_band, source_valid_band, offset_band


def _quality_splat(
    renderer: StereoRenderer,
    source: np.ndarray,
    geometry: QualityGeometryResult,
    offsets: np.ndarray | torch.Tensor,
    start_row: int,
    end_row: int,
    *,
    device_frame: _QualityDeviceFrame | None = None,
    device_offsets: torch.Tensor | None = None,
) -> QualitySplatBand:
    width = source.shape[1]
    source_band, near_score_band, source_valid_band, offset_band = _quality_device_band(
        renderer,
        source,
        geometry,
        offsets,
        start_row,
        end_row,
        device_frame=device_frame,
        device_offsets=device_offsets,
    )
    return _quality_splat_band_prevalidated(
        source_band,
        near_score_band,
        offset_band,
        source_index_offset=start_row * width,
        source_valid=source_valid_band,
    )


def _quality_compact_cuda_splat(
    renderer: StereoRenderer,
    source: np.ndarray,
    geometry: QualityGeometryResult,
    offsets: np.ndarray | torch.Tensor,
    start_row: int,
    end_row: int,
    *,
    device_frame: _QualityDeviceFrame | None = None,
    device_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if renderer.device.type != "cuda":
        return None
    from ..fast_splat_triton import (
        triton_compact_quality_winners,
        triton_splat_available,
    )

    if not triton_splat_available(renderer.device):
        return None
    from ..forward_splat import _winner_keys

    source_band, near_score_band, source_valid_band, offset_band = _quality_device_band(
        renderer,
        source,
        geometry,
        offsets,
        start_row,
        end_row,
        device_frame=device_frame,
        device_offsets=device_offsets,
    )
    winners = _winner_keys(
        near_score_band,
        offset_band,
        start_row * source.shape[1],
        source_valid_band,
    )
    compact = triton_compact_quality_winners(
        source_band,
        winners,
        start_row * source.shape[1],
    )
    del winners
    return compact


def _none_repair_bits(coverage: np.ndarray) -> np.ndarray:
    result = np.zeros(coverage.shape, dtype=np.uint8)
    partial = (coverage > 0) & (coverage < HORIZONTAL_SUBPIXELS)
    full = coverage == 0
    result[partial] |= np.uint8(0x01)
    result[full] |= np.uint8(0x02)
    result[coverage < HORIZONTAL_SUBPIXELS] |= np.uint8(0x40)
    return result


def _accumulate_hole_run_histograms(
    lane_valid: np.ndarray,
    lane_histogram: np.ndarray,
    span_histogram: np.ndarray,
    coverage_output: np.ndarray | None = None,
) -> None:
    native_kd._require_native().accumulate_lane_hole_run_histograms_into(
        lane_valid,
        lane_histogram,
        span_histogram,
        coverage_output,
    )


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


def _render_unrepaired_eye_once(
    *,
    renderer: StereoRenderer,
    source: np.ndarray,
    geometry: QualityGeometryResult,
    offsets: np.ndarray | torch.Tensor,
    band_height: int,
    device_frame: _QualityDeviceFrame | None = None,
) -> _UnrepairedEyeResult:
    height, width = source.shape[:2]
    output = np.empty_like(source)
    coverage = np.empty((height, width), dtype=np.uint8)
    lane_histogram = np.zeros(16 * (width + 1), dtype=np.uint64)
    span_histogram = np.zeros(width + 1, dtype=np.uint64)
    native_extension = None
    lane_weights = None
    if renderer.device.type == "cuda":
        native_extension = native_kd._require_native()
        lane_weights = _lane_mask_weights(renderer.device)
    device_offsets = offsets if isinstance(offsets, torch.Tensor) else None
    if device_frame is not None and isinstance(offsets, np.ndarray):
        try:
            device_offsets = renderer._transfer_offset_band(offsets)
        except torch.cuda.OutOfMemoryError:
            _release_after_oom(renderer)
    for start_row in range(0, height, band_height):
        end_row = min(height, start_row + band_height)
        compact_cuda = _quality_compact_cuda_splat(
            renderer,
            source,
            geometry,
            offsets,
            start_row,
            end_row,
            device_frame=device_frame,
            device_offsets=device_offsets,
        )
        if compact_cuda is not None:
            downsampled, packed_missing_device = compact_cuda
            packed_missing = np.ascontiguousarray(
                packed_missing_device.cpu().numpy().view(np.uint16)
            )
            assert native_extension is not None
            native_extension.accumulate_packed_lane_hole_run_histograms_into(
                packed_missing,
                lane_histogram,
                span_histogram,
                coverage[start_row:end_row],
            )
            output[start_row:end_row] = _convert_image_band(downsampled, source.dtype)
            del compact_cuda, downsampled, packed_missing_device, packed_missing
            continue
        band = _quality_splat(
            renderer,
            source,
            geometry,
            offsets,
            start_row,
            end_row,
            device_frame=device_frame,
            device_offsets=device_offsets,
        )
        lane_valid = band.splat.valid.reshape(
            end_row - start_row,
            width,
            HORIZONTAL_SUBPIXELS,
        )
        if native_extension is None:
            lane_valid_host = np.ascontiguousarray(lane_valid.detach().numpy())
            _accumulate_hole_run_histograms(
                lane_valid_host,
                lane_histogram,
                span_histogram,
                coverage[start_row:end_row],
            )
            del lane_valid_host
        else:
            if lane_weights is None:
                raise AssertionError("CUDA lane-mask weights are unavailable")
            packed_missing = np.ascontiguousarray(
                _pack_missing_lane_masks(lane_valid, lane_weights).cpu().numpy().view(np.uint16)
            )
            native_extension.accumulate_packed_lane_hole_run_histograms_into(
                packed_missing,
                lane_histogram,
                span_histogram,
                coverage[start_row:end_row],
            )
            del packed_missing
        downsampled = _downsample_subpixel_band(band.splat.colour)
        output[start_row:end_row] = _convert_image_band(downsampled, source.dtype)
        del band, lane_valid, downsampled
    hole_runs = HoleRunDiagnostics(
        lane_count_histogram=lane_histogram,
        touched_pixel_span_histogram=span_histogram,
    )
    return _UnrepairedEyeResult(
        output,
        coverage,
        hole_runs,
    )


def _render_none_eye_once(
    *,
    renderer: StereoRenderer,
    source: np.ndarray,
    geometry: QualityGeometryResult,
    offsets: np.ndarray | torch.Tensor,
    band_height: int,
    device_frame: _QualityDeviceFrame | None = None,
) -> _CompactEyeResult:
    unrepaired = _render_unrepaired_eye_once(
        renderer=renderer,
        source=source,
        geometry=geometry,
        offsets=offsets,
        band_height=band_height,
        device_frame=device_frame,
    )
    return _CompactEyeResult(
        unrepaired.image,
        unrepaired.coverage_count,
        _none_repair_bits(unrepaired.coverage_count),
        None,
        unrepaired.hole_runs,
    )


def _build_background_proxy_mask(
    coverage: np.ndarray,
    *,
    far_side: Literal["left", "right"],
    foreground_exclusion_px: int,
) -> np.ndarray:
    repair_core = np.ascontiguousarray(coverage <= np.uint8(HORIZONTAL_SUBPIXELS // 2))
    exclusion_zone = np.array(repair_core, copy=True, order="C")
    far_margin = 2
    if far_side == "right":
        for distance in range(1, foreground_exclusion_px + 1):
            exclusion_zone[:, :-distance] |= repair_core[:, distance:]
        for distance in range(1, far_margin + 1):
            exclusion_zone[:, distance:] |= repair_core[:, :-distance]
    else:
        for distance in range(1, foreground_exclusion_px + 1):
            exclusion_zone[:, distance:] |= repair_core[:, :-distance]
        for distance in range(1, far_margin + 1):
            exclusion_zone[:, :-distance] |= repair_core[:, distance:]
    vertical = np.array(exclusion_zone, copy=True, order="C")
    exclusion_zone[1:] |= vertical[:-1]
    exclusion_zone[:-1] |= vertical[1:]
    return np.ascontiguousarray(exclusion_zone)


def _inpaint_background_proxy(
    proxy_source: np.ndarray,
    mask: np.ndarray,
    *,
    radius_px: int,
    mask_bounds: tuple[int, int, int, int],
) -> np.ndarray:
    start_row, end_row, start_column, end_column = mask_bounds
    if end_row <= start_row or end_column <= start_column:
        return proxy_source

    height, width = mask.shape
    halo = radius_px + 1
    roi_start_row = max(0, start_row - halo)
    roi_end_row = min(height, end_row + halo)
    roi_start_column = max(0, start_column - halo)
    roi_end_column = min(width, end_column + halo)
    roi_pixels = (roi_end_row - roi_start_row) * (roi_end_column - roi_start_column)
    if (
        roi_pixels * QUALITY_INPAINT_ROI_AREA_DENOMINATOR
        >= height * width * QUALITY_INPAINT_ROI_AREA_NUMERATOR
    ):
        return cv2.inpaint(
            proxy_source,
            mask,
            float(radius_px),
            cv2.INPAINT_TELEA,
        )

    row_slice = slice(roi_start_row, roi_end_row)
    column_slice = slice(roi_start_column, roi_end_column)
    repaired_roi = cv2.inpaint(
        proxy_source[row_slice, column_slice],
        mask[row_slice, column_slice],
        float(radius_px),
        cv2.INPAINT_TELEA,
    )
    proxy_source[row_slice, column_slice] = repaired_roi
    return proxy_source


def _composite_background_proxy(  # noqa: C901 - validation stays at this boundary.
    pre_repair: np.ndarray,
    coverage: np.ndarray,
    source: np.ndarray,
    *,
    radius_px: int,
    far_side: Literal["left", "right"],
    foreground_exclusion_px: int,
) -> np.ndarray:
    if (
        not isinstance(pre_repair, np.ndarray)
        or pre_repair.dtype != np.uint8
        or pre_repair.ndim != 3
        or pre_repair.shape[2] != 3
        or not pre_repair.flags.c_contiguous
    ):
        raise TypeError("pre_repair must be a contiguous uint8 BGR raster")
    shape = (int(pre_repair.shape[0]), int(pre_repair.shape[1]))
    if (
        not isinstance(coverage, np.ndarray)
        or coverage.dtype != np.uint8
        or coverage.shape != shape
        or not coverage.flags.c_contiguous
    ):
        raise TypeError("coverage must be a matching contiguous uint8 raster")
    if (
        not isinstance(source, np.ndarray)
        or source.dtype != np.uint8
        or source.shape != pre_repair.shape
        or not source.flags.c_contiguous
    ):
        raise TypeError("source must match the pre-repair BGR raster")
    if np.any(coverage > np.uint8(HORIZONTAL_SUBPIXELS)):
        raise ValueError("coverage exceeds the horizontal subpixel count")
    if isinstance(radius_px, (bool, np.bool_)) or not isinstance(radius_px, (int, np.integer)):
        raise TypeError("radius_px must be an integer")
    radius = int(radius_px)
    if radius < 1 or radius > 3:
        raise ValueError("radius_px must be within 1..3")
    if far_side not in {"left", "right"}:
        raise ValueError("far_side must be 'left' or 'right'")
    if isinstance(foreground_exclusion_px, (bool, np.bool_)) or not isinstance(
        foreground_exclusion_px,
        (int, np.integer),
    ):
        raise TypeError("foreground_exclusion_px must be an integer")
    exclusion = int(foreground_exclusion_px)
    if exclusion < 1 or exclusion > 32:
        raise ValueError("foreground_exclusion_px must be within 1..32")

    missing = np.ascontiguousarray(coverage < np.uint8(HORIZONTAL_SUBPIXELS))
    if not bool(np.any(missing)):
        return np.array(pre_repair, copy=True, order="C")

    native = native_kd._require_native()
    proxy_source = native.normalize_background_proxy_source(pre_repair, coverage)

    mask = _build_background_proxy_mask(
        coverage,
        far_side=far_side,
        foreground_exclusion_px=exclusion,
    )
    if bool(np.all(mask)):
        proxy = np.array(source, copy=True, order="C")
    else:
        mask_u8 = np.ascontiguousarray(mask, dtype=np.uint8) * np.uint8(255)
        x, y, width, height = cv2.boundingRect(mask_u8)
        proxy = native.inpaint_background_proxy_wavefront(
            proxy_source,
            mask_u8,
            int(y),
            int(y + height),
            int(x),
            int(x + width),
        )

    return native.composite_background_proxy(pre_repair, coverage, proxy)


def _background_proxy_repair_state(
    coverage: np.ndarray,
) -> tuple[np.ndarray, QualityRepairPlanSummary]:
    repair_bits = np.empty_like(coverage)
    missing = np.empty_like(coverage)
    missing_lanes, missing_pixels = native_kd._require_native().build_background_proxy_state_into(
        coverage,
        repair_bits,
        missing,
    )
    missing_lanes = int(missing_lanes)
    missing_pixels = int(missing_pixels)
    component_count = 0
    if missing_pixels:
        component_count = int(cv2.connectedComponents(missing, connectivity=4)[0]) - 1
    return repair_bits, _background_proxy_summary(
        missing_lanes=missing_lanes,
        missing_pixels=missing_pixels,
        component_count=component_count,
    )


def _background_proxy_summary(
    *,
    missing_lanes: int,
    missing_pixels: int,
    component_count: int,
) -> QualityRepairPlanSummary:
    return QualityRepairPlanSummary(
        segment_record_count=component_count,
        segment_table_bytes=component_count * REPAIR_RECORD_ITEMSIZE,
        backend_lane_counts=(0, 0, missing_lanes),
        backend_pixel_counts=(0, 0, missing_pixels),
        backend_component_counts=(0, 0, component_count),
        local_evaluated_slot_count=0,
        local_physical_sample_read_count=0,
        local_budget_skipped_run_count=0,
        fallback_indexed_donor_count=0,
        fallback_query_count=0,
        fallback_anchor_count=0,
        fallback_visited_nodes_total=0,
        fallback_visited_nodes_max=0,
        fallback_visited_nodes_p95=0.0,
        exemplar_donor_evaluation_count=0,
    )


def _render_background_eye(
    *,
    renderer: StereoRenderer,
    source: np.ndarray,
    geometry: QualityGeometryResult,
    offsets: np.ndarray,
    eye: Literal["left", "right"],
    controls_limit_px: int,
    band_state: _FrameBandState,
) -> _CompactEyeResult:
    pre_repair = _retry_after_oom(
        renderer,
        band_state,
        _render_unrepaired_eye_once,
        renderer=renderer,
        source=source,
        geometry=geometry,
        offsets=offsets,
    )
    return _finish_background_eye(
        pre_repair=pre_repair,
        source=source,
        eye=eye,
        controls_limit_px=controls_limit_px,
        predicted_gap_px=geometry.predicted_gap_px,
    )


def _finish_background_eye(
    *,
    pre_repair: _UnrepairedEyeResult,
    source: np.ndarray,
    eye: Literal["left", "right"],
    controls_limit_px: int,
    predicted_gap_px: int,
    compute_component_count: bool = True,
) -> _CompactEyeResult:
    safe_limit = max(1, math.floor(controls_limit_px * source.shape[0] / 1080.0 + 0.5))
    local_limit = min(safe_limit, predicted_gap_px)
    native = native_kd._require_native()
    (
        proxy_source,
        mask,
        repair_bits,
        missing,
        missing_lanes,
        missing_pixels,
        masked_pixels,
        mask_bounds,
    ) = native.prepare_background_proxy(
        pre_repair.image,
        pre_repair.coverage_count,
        1 if eye == "right" else -1,
        local_limit,
    )
    missing_pixels = int(missing_pixels)
    if not missing_pixels:
        image = np.array(pre_repair.image, copy=True, order="C")
        component_count = 0
    else:
        if int(masked_pixels) == int(mask.size):
            proxy = np.array(source, copy=True, order="C")
        else:
            first_row, end_row, first_col, end_col = mask_bounds
            proxy = native.inpaint_background_proxy_wavefront(
                proxy_source,
                mask,
                int(first_row),
                int(end_row),
                int(first_col),
                int(end_col),
            )
        image = native.composite_background_proxy(
            pre_repair.image,
            pre_repair.coverage_count,
            proxy,
        )
        component_count = (
            int(cv2.connectedComponents(missing, connectivity=4)[0]) - 1
            if compute_component_count
            else 0
        )
    summary = _background_proxy_summary(
        missing_lanes=int(missing_lanes),
        missing_pixels=missing_pixels,
        component_count=component_count,
    )
    return _CompactEyeResult(
        image,
        pre_repair.coverage_count,
        repair_bits,
        summary,
        pre_repair.hole_runs,
    )


def _splat_unrepaired_eye_deferred_once(
    *,
    renderer: StereoRenderer,
    source: np.ndarray,
    geometry: QualityGeometryResult,
    offsets: np.ndarray | torch.Tensor,
    band_height: int,
    device_frame: _QualityDeviceFrame | None = None,
) -> _UnrepairedEyeResult | _DeferredUnrepairedEyeResult:
    """Banded CUDA splat with GPU uint8 conversion, pinned D2H, and deferred CPU histogram."""

    height, width = source.shape[:2]
    if renderer.device.type == "cuda" and source.dtype == np.uint8:
        device_offsets = offsets if isinstance(offsets, torch.Tensor) else None
        if device_frame is not None and isinstance(offsets, np.ndarray):
            try:
                device_offsets = renderer._transfer_offset_band(offsets)
            except torch.cuda.OutOfMemoryError:
                _release_after_oom(renderer)
        pinned = _get_pinned_transfer_cache().ensure_d2h(height, width)
        if pinned is not None:
            pin_rgb, pin_packed = pinned
            used_compact = True
            for start_row in range(0, height, band_height):
                end_row = min(height, start_row + band_height)
                compact_cuda = _quality_compact_cuda_splat(
                    renderer,
                    source,
                    geometry,
                    offsets,
                    start_row,
                    end_row,
                    device_frame=device_frame,
                    device_offsets=device_offsets,
                )
                if compact_cuda is None:
                    used_compact = False
                    break
                downsampled, packed_missing_device = compact_cuda
                rgb_gpu = torch.round(downsampled.detach())
                rgb_gpu.clamp_(0, 255)
                pin_rgb[start_row:end_row].copy_(rgb_gpu.to(dtype=torch.uint8), non_blocking=True)
                pin_packed[start_row:end_row].copy_(
                    packed_missing_device.contiguous(), non_blocking=True
                )
                del compact_cuda, downsampled, packed_missing_device, rgb_gpu
            if used_compact:
                torch.cuda.current_stream(renderer.device).synchronize()
                image = np.array(pin_rgb.numpy(), copy=True, order="C")
                packed_missing = np.array(pin_packed.numpy().view(np.uint16), copy=True, order="C")
                return _DeferredUnrepairedEyeResult(image=image, packed_missing=packed_missing)
    return _render_unrepaired_eye_once(
        renderer=renderer,
        source=source,
        geometry=geometry,
        offsets=offsets,
        band_height=band_height,
        device_frame=device_frame,
    )


def _unpack_unrepaired_eye(
    pre_repair: _UnrepairedEyeResult | _DeferredUnrepairedEyeResult,
) -> _UnrepairedEyeResult:
    if isinstance(pre_repair, _UnrepairedEyeResult):
        return pre_repair
    height, width = pre_repair.image.shape[:2]
    coverage = np.empty((height, width), dtype=np.uint8)
    lane_histogram = np.zeros(16 * (width + 1), dtype=np.uint64)
    span_histogram = np.zeros(width + 1, dtype=np.uint64)
    native_kd._require_native().accumulate_packed_lane_hole_run_histograms_into(
        pre_repair.packed_missing,
        lane_histogram,
        span_histogram,
        coverage,
    )
    return _UnrepairedEyeResult(
        pre_repair.image,
        coverage,
        HoleRunDiagnostics(lane_histogram, span_histogram),
    )


def _splat_quality_eyes_only(
    *,
    renderer: StereoRenderer,
    source: np.ndarray,
    geometry: QualityGeometryResult,
    offset_builder: Callable[
        [Literal["left", "right"], _QualityDeviceFrame | None], np.ndarray | torch.Tensor
    ],
    occlusion_fill: Literal["none", "background"],
) -> tuple[
    _UnrepairedEyeResult | _DeferredUnrepairedEyeResult | None,
    _UnrepairedEyeResult | _DeferredUnrepairedEyeResult | None,
    _CompactEyeResult | None,
    _CompactEyeResult | None,
    float,
    float,
]:
    """Run only GPU/CPU forward splat for both eyes without CPU background repair."""

    height, width = source.shape[:2]
    initial_height = _calculate_quality_band_height(
        width,
        height,
        _quality_temporary_budget_bytes(renderer),
    )
    device_frame = _preload_quality_device_frame(renderer, source, geometry)
    band_state = _FrameBandState(
        initial_height=initial_height,
        retry_height=max(1, initial_height // 2),
        current_height=initial_height,
    )

    def build_offsets(eye: Literal["left", "right"]):
        try:
            return offset_builder(eye, device_frame)
        except torch.cuda.OutOfMemoryError:
            pass
        _release_after_oom(renderer)
        return offset_builder(eye, None)

    if occlusion_fill == "none":
        left_started = time.perf_counter()
        left_offsets = build_offsets("left")
        left_none = _retry_after_oom(
            renderer,
            band_state,
            _render_none_eye_once,
            renderer=renderer,
            source=source,
            geometry=geometry,
            offsets=left_offsets,
            device_frame=device_frame,
        )
        del left_offsets
        left_seconds = time.perf_counter() - left_started

        right_started = time.perf_counter()
        right_offsets = build_offsets("right")
        right_none = _retry_after_oom(
            renderer,
            band_state,
            _render_none_eye_once,
            renderer=renderer,
            source=source,
            geometry=geometry,
            offsets=right_offsets,
            device_frame=device_frame,
        )
        del right_offsets
        right_seconds = time.perf_counter() - right_started
        return None, None, left_none, right_none, left_seconds, right_seconds

    left_started = time.perf_counter()
    left_offsets = build_offsets("left")
    left_pre = _retry_after_oom(
        renderer,
        band_state,
        _splat_unrepaired_eye_deferred_once,
        renderer=renderer,
        source=source,
        geometry=geometry,
        offsets=left_offsets,
        device_frame=device_frame,
    )
    del left_offsets
    left_seconds = time.perf_counter() - left_started

    right_started = time.perf_counter()
    right_offsets = build_offsets("right")
    right_pre = _retry_after_oom(
        renderer,
        band_state,
        _splat_unrepaired_eye_deferred_once,
        renderer=renderer,
        source=source,
        geometry=geometry,
        offsets=right_offsets,
        device_frame=device_frame,
    )
    del right_offsets
    right_seconds = time.perf_counter() - right_started
    return left_pre, right_pre, None, None, left_seconds, right_seconds


def _render_quality_eyes(
    *,
    renderer: StereoRenderer,
    source: np.ndarray,
    geometry: QualityGeometryResult,
    offset_builder: Callable[
        [Literal["left", "right"], _QualityDeviceFrame | None], np.ndarray | torch.Tensor
    ],
    occlusion_fill: Literal["none", "background"],
    controls_limit_px: int,
) -> tuple[_CompactEyeResult, _CompactEyeResult, dict[str, float]]:
    height, width = source.shape[:2]
    initial_height = _calculate_quality_band_height(
        width,
        height,
        _quality_temporary_budget_bytes(renderer),
    )
    device_frame = _preload_quality_device_frame(renderer, source, geometry)
    band_state = _FrameBandState(
        initial_height=initial_height,
        retry_height=max(1, initial_height // 2),
        current_height=initial_height,
    )

    def build_offsets(eye: Literal["left", "right"]):
        try:
            return offset_builder(eye, device_frame)
        except torch.cuda.OutOfMemoryError:
            pass
        # End the exception scope before releasing partial kernel allocations.
        _release_after_oom(renderer)
        return offset_builder(eye, None)

    if occlusion_fill == "none":
        eyes: list[_CompactEyeResult] = []
        eye_seconds: dict[str, float] = {}
        for eye in ("left", "right"):
            eye_started = time.perf_counter()
            offsets = build_offsets(eye)
            result = _retry_after_oom(
                renderer,
                band_state,
                _render_none_eye_once,
                renderer=renderer,
                source=source,
                geometry=geometry,
                offsets=offsets,
                device_frame=device_frame,
            )
            eyes.append(result)
            del offsets
            eye_seconds[f"{eye}_eye"] = time.perf_counter() - eye_started
        return eyes[0], eyes[1], eye_seconds

    def render_unrepaired(eye: Literal["left", "right"]):
        started = time.perf_counter()
        offsets = build_offsets(eye)
        result = _retry_after_oom(
            renderer,
            band_state,
            _render_unrepaired_eye_once,
            renderer=renderer,
            source=source,
            geometry=geometry,
            offsets=offsets,
            device_frame=device_frame,
        )
        del offsets
        return result, time.perf_counter() - started

    def finish(pre_repair: _UnrepairedEyeResult, eye: Literal["left", "right"]):
        started = time.perf_counter()
        result = _finish_background_eye(
            pre_repair=pre_repair,
            source=source,
            eye=eye,
            controls_limit_px=controls_limit_px,
            predicted_gap_px=geometry.predicted_gap_px,
        )
        return result, time.perf_counter() - started

    left_pre_repair, left_prefix_seconds = render_unrepaired("left")
    executor = _get_left_repair_executor()
    left_future = executor.submit(finish, left_pre_repair, "left")
    try:
        right_pre_repair, right_prefix_seconds = render_unrepaired("right")
        right, right_finish_seconds = finish(right_pre_repair, "right")
        left, left_finish_seconds = left_future.result()
    except BaseException:
        wait_futures([left_future])
        raise
    return (
        left,
        right,
        {
            "left_eye": left_prefix_seconds + left_finish_seconds,
            "right_eye": right_prefix_seconds + right_finish_seconds,
        },
    )


def _build_eye_offsets(
    geometry: QualityGeometryResult,
    settings: StereoRenderSettings,
    eye: Literal["left", "right"],
    device_frame: _QualityDeviceFrame | None = None,
) -> np.ndarray | torch.Tensor:
    if (
        device_frame is not None
        and np.min(geometry.near_score) >= -1.0
        and np.max(geometry.near_score) <= 2.0
    ):
        from ..fast_offsets_triton import triton_relative_eye_offsets_prevalidated

        # Keep the shared kernel's two-unit int32 proof, including resize roundoff.
        resident = triton_relative_eye_offsets_prevalidated(
            device_frame.near_score,
            stereo_strength=settings.stereo_strength,
            convergence=settings.convergence,
            eye=eye,
        )
        if resident is not None:
            return resident
    output = np.empty(geometry.near_score.shape, dtype=np.int32)
    build_relative_eye_offsets_into(
        geometry.near_score,
        stereo_strength=settings.stereo_strength,
        convergence=settings.convergence,
        eye=eye,
        output_int32=output,
        row_float64_scratch=np.empty(geometry.near_score.shape[1], dtype=np.float64),
    )
    return output


def _build_metric_eye_offsets(
    geometry: QualityGeometryResult,
    eye: Literal["left", "right"],
) -> np.ndarray:
    output = np.empty(geometry.near_score.shape, dtype=np.int32)
    build_geometry_eye_offsets_into(
        geometry.total_disparity_fraction,
        eye=eye,
        output_int32=output,
        row_float64_scratch=np.empty(geometry.near_score.shape[1], dtype=np.float64),
    )
    return output


def _quality_limits(
    geometry: QualityGeometryResult,
    *,
    height: int,
    occlusion_fill: Literal["none", "background"],
    occlusion_fill_max_px: int,
) -> dict[str, dict[str, int | float]] | None:
    if occlusion_fill != "background":
        return None
    safe_limit = max(
        1,
        math.floor(occlusion_fill_max_px * height / 1080.0 + 0.5),
    )
    local_limit = min(safe_limit, geometry.predicted_gap_px)
    return {
        eye: {
            "max_neighbour_abs_q_jump_px": float(geometry.max_neighbour_abs_q_jump_px),
            "predicted_gap_px": int(geometry.predicted_gap_px),
            "local_limit_px": int(local_limit),
        }
        for eye in ("left", "right")
    }


def _render_metric_quality_eyes(
    *,
    renderer: StereoRenderer,
    source: np.ndarray,
    geometry: QualityGeometryResult,
    controls: QualityStereoControls,
) -> tuple[_CompactEyeResult, _CompactEyeResult, dict[str, float]]:
    return _render_quality_eyes(
        renderer=renderer,
        source=source,
        geometry=geometry,
        offset_builder=lambda eye, _device_frame: _build_metric_eye_offsets(geometry, eye),
        occlusion_fill=controls.occlusion_fill,
        controls_limit_px=controls.occlusion_fill_max_px,
    )


def prepare_relative_quality_frame(
    *,
    frame: np.ndarray,
    canonical: np.ndarray,
    settings: StereoRenderSettings,
) -> PreparedRelativeQualityFrame:
    """Build deterministic CPU geometry before the GPU eye stage."""

    prepare_started = time.perf_counter()
    if not isinstance(settings, StereoRenderSettings) or settings.stereo_render_mode != "quality":
        raise ValueError("Quality rendering requires StereoRenderSettings in quality mode")
    frame_values = np.asarray(frame)
    canonical_input = np.asarray(canonical)
    if frame_values.ndim != 3 or frame_values.shape[2] != 3 or canonical_input.ndim != 2:
        raise TypeError("Quality rendering requires BGR frames and a 2D canonical map")
    _quality_public_memory_preflight(
        render_shape=(int(frame_values.shape[0]), int(frame_values.shape[1])),
        native_shape=(int(canonical_input.shape[0]), int(canonical_input.shape[1])),
        occlusion_fill=settings.occlusion_fill,
        materialize_public_masks=False,
    )
    source = np.ascontiguousarray(frame_values)
    if source.dtype != np.uint8:
        raise TypeError("Quality rendering requires uint8 BGR input")
    canonical_values = np.ascontiguousarray(np.asarray(canonical, dtype=np.float32))
    StereoRenderer._validate_inputs(source, canonical_values, settings)
    height, width = source.shape[:2]
    geometry = build_relative_quality_geometry(
        source,
        canonical_values,
        render_shape=(height, width),
        stereo_strength=settings.stereo_strength,
        convergence=settings.convergence,
    )
    indexed_sample_count = int(canonical_values.size)
    return PreparedRelativeQualityFrame(
        source=source,
        geometry=geometry,
        settings=settings,
        indexed_sample_count=indexed_sample_count,
        prepare_seconds=time.perf_counter() - prepare_started,
    )


def render_prepared_relative_quality_compact(
    *,
    renderer: StereoRenderer,
    prepared: PreparedRelativeQualityFrame,
) -> CompactStereoRenderResult:
    """Render GPU eyes from one owned, precomputed relative geometry frame."""

    render_started = time.perf_counter()
    if not isinstance(renderer, StereoRenderer):
        raise TypeError("renderer must be StereoRenderer")
    if not isinstance(prepared, PreparedRelativeQualityFrame):
        raise TypeError("prepared must be PreparedRelativeQualityFrame")
    source = prepared.source
    geometry = prepared.geometry
    settings = prepared.settings
    height = source.shape[0]
    left, right, eye_seconds = _render_quality_eyes(
        renderer=renderer,
        source=source,
        geometry=geometry,
        offset_builder=lambda eye, device_frame: _build_eye_offsets(
            geometry, settings, eye, device_frame
        ),
        occlusion_fill=settings.occlusion_fill,
        controls_limit_px=settings.occlusion_fill_max_px,
    )
    safe_limit = max(
        1,
        math.floor(settings.occlusion_fill_max_px * height / 1080.0 + 0.5),
    )
    local_limit = min(safe_limit, geometry.predicted_gap_px)
    quality_limits = (
        {
            eye: {
                "max_neighbour_abs_q_jump_px": float(geometry.max_neighbour_abs_q_jump_px),
                "predicted_gap_px": int(geometry.predicted_gap_px),
                "local_limit_px": int(local_limit),
            }
            for eye in ("left", "right")
        }
        if settings.occlusion_fill == "background"
        else None
    )
    result = CompactStereoRenderResult(
        left_image=left.image,
        right_image=right.image,
        left_coverage_count=left.coverage_count,
        right_coverage_count=right.coverage_count,
        left_repair_bits=left.repair_bits,
        right_repair_bits=right.repair_bits,
        left_plan=left.plan,
        right_plan=right.plan,
        geometry_nearest={
            "indexed_sample_count": prepared.indexed_sample_count,
            "query_count": int(geometry.geometry_query_count),
            "visited_nodes_total": int(geometry.geometry_visited_nodes_total),
            "visited_nodes_max": int(geometry.geometry_visited_nodes_max),
        },
        quality_limits=quality_limits,
        left_hole_runs=left.hole_runs,
        right_hole_runs=right.hole_runs,
        performance_seconds={
            **geometry.performance_seconds,
            **eye_seconds,
            "total": prepared.prepare_seconds + time.perf_counter() - render_started,
        },
    )
    return result


def splat_prepared_relative_quality_frame(
    *,
    renderer: StereoRenderer,
    prepared: PreparedRelativeQualityFrame,
) -> SplattedQualityFrame:
    """Execute only the GPU/splat stage for a prepared relative Quality frame."""

    splat_started = time.perf_counter()
    if not isinstance(renderer, StereoRenderer):
        raise TypeError("renderer must be StereoRenderer")
    if not isinstance(prepared, PreparedRelativeQualityFrame):
        raise TypeError("prepared must be PreparedRelativeQualityFrame")
    geometry = prepared.geometry
    settings = prepared.settings
    (
        left_pre,
        right_pre,
        left_none,
        right_none,
        left_sec,
        right_sec,
    ) = _splat_quality_eyes_only(
        renderer=renderer,
        source=prepared.source,
        geometry=geometry,
        offset_builder=lambda eye, device_frame: _build_eye_offsets(
            geometry, settings, eye, device_frame
        ),
        occlusion_fill=settings.occlusion_fill,
    )
    return SplattedQualityFrame(
        source=prepared.source,
        occlusion_fill=settings.occlusion_fill,
        controls_limit_px=settings.occlusion_fill_max_px,
        predicted_gap_px=int(geometry.predicted_gap_px),
        max_neighbour_abs_q_jump_px=float(geometry.max_neighbour_abs_q_jump_px),
        indexed_sample_count=prepared.indexed_sample_count,
        geometry_query_count=int(geometry.geometry_query_count),
        geometry_visited_nodes_total=int(geometry.geometry_visited_nodes_total),
        geometry_visited_nodes_max=int(geometry.geometry_visited_nodes_max),
        metric_stats=None,
        geometry_performance_seconds=dict(geometry.performance_seconds),
        prepare_seconds=prepared.prepare_seconds,
        splat_seconds=time.perf_counter() - splat_started,
        left_prefix_seconds=left_sec,
        right_prefix_seconds=right_sec,
        left_pre_repair=left_pre,
        right_pre_repair=right_pre,
        left_none=left_none,
        right_none=right_none,
    )


def prepare_metric_quality_frame(
    *,
    frame: np.ndarray,
    primitives: MetricStereoPrimitiveInput,
    controls: QualityStereoControls,
) -> PreparedMetricQualityFrame:
    """Build deterministic CPU metric geometry before the GPU eye stage."""

    prepare_started = time.perf_counter()
    if not isinstance(primitives, MetricStereoPrimitiveInput):
        raise TypeError("primitives must be MetricStereoPrimitiveInput")
    if not isinstance(controls, QualityStereoControls):
        raise TypeError("controls must be QualityStereoControls")
    frame_values = np.asarray(frame)
    if frame_values.ndim != 3 or frame_values.shape[2] != 3:
        raise TypeError("Quality rendering requires uint8 BGR input")
    metric = primitives.metric
    for name in ("inverse_depth", "valid", "focal_x_normalized"):
        if not hasattr(metric, name):
            raise TypeError("metric primitive is missing native geometry fields")
    metric_shape = np.asarray(metric.inverse_depth).shape
    if len(metric_shape) != 2:
        raise TypeError("metric inverse depth must be two-dimensional")
    _quality_public_memory_preflight(
        render_shape=(int(frame_values.shape[0]), int(frame_values.shape[1])),
        native_shape=(int(metric_shape[0]), int(metric_shape[1])),
        occlusion_fill=controls.occlusion_fill,
        materialize_public_masks=False,
        geometry_mode="metric",
    )
    source = np.ascontiguousarray(frame_values)
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
        raise TypeError("Quality rendering requires uint8 BGR input")
    height, width = source.shape[:2]
    geometry = build_metric_quality_geometry(
        source,
        metric.inverse_depth,
        metric.valid,
        metric.focal_x_normalized,
        render_shape=(height, width),
        virtual_baseline_mm=float(primitives.virtual_baseline_mm),
        convergence_distance_m=float(primitives.convergence_distance_m),
        max_disparity_percent=float(primitives.max_disparity_percent),
        retained_crop_width=primitives.retained_crop_width,
    )
    if geometry.metric_stats is None:
        raise RuntimeError("Metric Quality geometry did not produce projection statistics")
    return PreparedMetricQualityFrame(
        source=source,
        geometry=geometry,
        controls=controls,
        indexed_sample_count=int(metric.inverse_depth.size),
        prepare_seconds=time.perf_counter() - prepare_started,
    )


def splat_prepared_metric_quality_frame(
    *,
    renderer: StereoRenderer,
    prepared: PreparedMetricQualityFrame,
) -> SplattedQualityFrame:
    """Execute only the GPU/splat stage for a prepared metric Quality frame."""

    splat_started = time.perf_counter()
    if not isinstance(renderer, StereoRenderer):
        raise TypeError("renderer must be StereoRenderer")
    if not isinstance(prepared, PreparedMetricQualityFrame):
        raise TypeError("prepared must be PreparedMetricQualityFrame")
    geometry = prepared.geometry
    controls = prepared.controls
    (
        left_pre,
        right_pre,
        left_none,
        right_none,
        left_sec,
        right_sec,
    ) = _splat_quality_eyes_only(
        renderer=renderer,
        source=prepared.source,
        geometry=geometry,
        offset_builder=lambda eye, _device_frame: _build_metric_eye_offsets(geometry, eye),
        occlusion_fill=controls.occlusion_fill,
    )
    return SplattedQualityFrame(
        source=prepared.source,
        occlusion_fill=controls.occlusion_fill,
        controls_limit_px=controls.occlusion_fill_max_px,
        predicted_gap_px=int(geometry.predicted_gap_px),
        max_neighbour_abs_q_jump_px=float(geometry.max_neighbour_abs_q_jump_px),
        indexed_sample_count=prepared.indexed_sample_count,
        geometry_query_count=int(geometry.geometry_query_count),
        geometry_visited_nodes_total=int(geometry.geometry_visited_nodes_total),
        geometry_visited_nodes_max=int(geometry.geometry_visited_nodes_max),
        metric_stats=geometry.metric_stats,
        geometry_performance_seconds=dict(geometry.performance_seconds),
        prepare_seconds=prepared.prepare_seconds,
        splat_seconds=time.perf_counter() - splat_started,
        left_prefix_seconds=left_sec,
        right_prefix_seconds=right_sec,
        left_pre_repair=left_pre,
        right_pre_repair=right_pre,
        left_none=left_none,
        right_none=right_none,
    )


def finish_splatted_quality_frame(
    splatted: SplattedQualityFrame,
    *,
    compute_component_count: bool = True,
) -> CompactStereoRenderResult:
    """Complete pure-CPU background proxy repair and assemble CompactStereoRenderResult."""

    finish_started = time.perf_counter()
    if splatted.occlusion_fill == "none":
        assert splatted.left_none is not None and splatted.right_none is not None
        left = splatted.left_none
        right = splatted.right_none
        left_finish_sec = 0.0
        right_finish_sec = 0.0
        quality_limits = None
    else:
        assert splatted.left_pre_repair is not None and splatted.right_pre_repair is not None
        l_start = time.perf_counter()
        left_unpacked = _unpack_unrepaired_eye(splatted.left_pre_repair)
        left = _finish_background_eye(
            pre_repair=left_unpacked,
            source=splatted.source,
            eye="left",
            controls_limit_px=splatted.controls_limit_px,
            predicted_gap_px=splatted.predicted_gap_px,
            compute_component_count=compute_component_count,
        )
        left_finish_sec = time.perf_counter() - l_start

        r_start = time.perf_counter()
        right_unpacked = _unpack_unrepaired_eye(splatted.right_pre_repair)
        right = _finish_background_eye(
            pre_repair=right_unpacked,
            source=splatted.source,
            eye="right",
            controls_limit_px=splatted.controls_limit_px,
            predicted_gap_px=splatted.predicted_gap_px,
            compute_component_count=compute_component_count,
        )
        right_finish_sec = time.perf_counter() - r_start

        height = int(splatted.source.shape[0])
        safe_limit = max(1, math.floor(splatted.controls_limit_px * height / 1080.0 + 0.5))
        local_limit = min(safe_limit, splatted.predicted_gap_px)
        quality_limits = {
            eye: {
                "max_neighbour_abs_q_jump_px": splatted.max_neighbour_abs_q_jump_px,
                "predicted_gap_px": splatted.predicted_gap_px,
                "local_limit_px": int(local_limit),
            }
            for eye in ("left", "right")
        }
    return CompactStereoRenderResult(
        left_image=left.image,
        right_image=right.image,
        left_coverage_count=left.coverage_count,
        right_coverage_count=right.coverage_count,
        left_repair_bits=left.repair_bits,
        right_repair_bits=right.repair_bits,
        left_plan=left.plan,
        right_plan=right.plan,
        metric_stats=splatted.metric_stats,  # type: ignore[arg-type]
        geometry_nearest={
            "indexed_sample_count": splatted.indexed_sample_count,
            "query_count": splatted.geometry_query_count,
            "visited_nodes_total": splatted.geometry_visited_nodes_total,
            "visited_nodes_max": splatted.geometry_visited_nodes_max,
        },
        quality_limits=quality_limits,
        left_hole_runs=left.hole_runs,
        right_hole_runs=right.hole_runs,
        performance_seconds={
            **splatted.geometry_performance_seconds,
            "left_eye": splatted.left_prefix_seconds + left_finish_sec,
            "right_eye": splatted.right_prefix_seconds + right_finish_sec,
            "total": (
                splatted.prepare_seconds
                + splatted.splat_seconds
                + (time.perf_counter() - finish_started)
            ),
        },
    )


def render_relative_quality_compact(
    *,
    renderer: StereoRenderer,
    frame: np.ndarray,
    canonical: np.ndarray,
    settings: StereoRenderSettings,
) -> CompactStereoRenderResult:
    """Render relative primitives through the split Quality work stages."""

    prepared = prepare_relative_quality_frame(
        frame=frame,
        canonical=canonical,
        settings=settings,
    )
    return render_prepared_relative_quality_compact(
        renderer=renderer,
        prepared=prepared,
    )


def render_metric_primitives_compact(
    *,
    renderer: StereoRenderer,
    frame: np.ndarray,
    primitives: MetricStereoPrimitiveInput,
    controls: QualityStereoControls,
) -> CompactStereoRenderResult:
    """Render native metric primitives after edge-aware Quality resampling."""

    total_started = time.perf_counter()
    if not isinstance(renderer, StereoRenderer):
        raise TypeError("renderer must be StereoRenderer")
    if not isinstance(primitives, MetricStereoPrimitiveInput):
        raise TypeError("primitives must be MetricStereoPrimitiveInput")
    if not isinstance(controls, QualityStereoControls):
        raise TypeError("controls must be QualityStereoControls")
    frame_values = np.asarray(frame)
    if frame_values.ndim != 3 or frame_values.shape[2] != 3:
        raise TypeError("Quality rendering requires uint8 BGR input")
    metric = primitives.metric
    for name in ("inverse_depth", "valid", "focal_x_normalized"):
        if not hasattr(metric, name):
            raise TypeError("metric primitive is missing native geometry fields")
    metric_shape = np.asarray(metric.inverse_depth).shape
    if len(metric_shape) != 2:
        raise TypeError("metric inverse depth must be two-dimensional")
    _quality_public_memory_preflight(
        render_shape=(int(frame_values.shape[0]), int(frame_values.shape[1])),
        native_shape=(int(metric_shape[0]), int(metric_shape[1])),
        occlusion_fill=controls.occlusion_fill,
        materialize_public_masks=False,
        geometry_mode="metric",
    )
    source = np.ascontiguousarray(frame_values)
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
        raise TypeError("Quality rendering requires uint8 BGR input")
    height, width = source.shape[:2]
    geometry = build_metric_quality_geometry(
        source,
        metric.inverse_depth,
        metric.valid,
        metric.focal_x_normalized,
        render_shape=(height, width),
        virtual_baseline_mm=float(primitives.virtual_baseline_mm),
        convergence_distance_m=float(primitives.convergence_distance_m),
        max_disparity_percent=float(primitives.max_disparity_percent),
        retained_crop_width=primitives.retained_crop_width,
    )
    indexed_sample_count = int(metric.inverse_depth.size)
    del frame, frame_values, primitives, metric
    left, right, eye_seconds = _render_metric_quality_eyes(
        renderer=renderer,
        source=source,
        geometry=geometry,
        controls=controls,
    )
    if geometry.metric_stats is None:
        raise RuntimeError("Metric Quality geometry did not produce projection statistics")
    metric_stats = geometry.metric_stats
    geometry_nearest = {
        "indexed_sample_count": indexed_sample_count,
        "query_count": int(geometry.geometry_query_count),
        "visited_nodes_total": int(geometry.geometry_visited_nodes_total),
        "visited_nodes_max": int(geometry.geometry_visited_nodes_max),
    }
    quality_limits = _quality_limits(
        geometry,
        height=height,
        occlusion_fill=controls.occlusion_fill,
        occlusion_fill_max_px=controls.occlusion_fill_max_px,
    )
    result = CompactStereoRenderResult(
        left_image=left.image,
        right_image=right.image,
        left_coverage_count=left.coverage_count,
        right_coverage_count=right.coverage_count,
        left_repair_bits=left.repair_bits,
        right_repair_bits=right.repair_bits,
        left_plan=left.plan,
        right_plan=right.plan,
        metric_stats=metric_stats,
        geometry_nearest=geometry_nearest,
        quality_limits=quality_limits,
        left_hole_runs=left.hole_runs,
        right_hole_runs=right.hole_runs,
        performance_seconds={
            **geometry.performance_seconds,
            **eye_seconds,
            "total": time.perf_counter() - total_started,
        },
    )
    del source, geometry, left, right
    return result


def render_relative_quality_public(
    *,
    renderer: StereoRenderer,
    frame: np.ndarray,
    canonical: np.ndarray,
    settings: StereoRenderSettings,
) -> StereoRenderResult:
    """Materialize the four legacy boolean masks for the public API."""

    frame_values = np.asarray(frame)
    canonical_values = np.asarray(canonical)
    if frame_values.ndim != 3 or canonical_values.ndim != 2:
        raise TypeError("Quality rendering requires BGR frames and a 2D canonical map")
    _quality_public_memory_preflight(
        render_shape=(int(frame_values.shape[0]), int(frame_values.shape[1])),
        native_shape=(int(canonical_values.shape[0]), int(canonical_values.shape[1])),
        occlusion_fill=settings.occlusion_fill,
        materialize_public_masks=True,
    )
    compact = render_relative_quality_compact(
        renderer=renderer,
        frame=frame,
        canonical=canonical,
        settings=settings,
    )
    left_valid = np.array(compact.left_coverage_count > 0, copy=True, order="C")
    right_valid = np.array(compact.right_coverage_count > 0, copy=True, order="C")
    if settings.occlusion_fill == "background":
        left_hole = np.zeros(left_valid.shape, dtype=np.bool_)
        right_hole = np.zeros(right_valid.shape, dtype=np.bool_)
    else:
        left_hole = np.array(compact.left_coverage_count == 0, copy=True, order="C")
        right_hole = np.array(compact.right_coverage_count == 0, copy=True, order="C")
    return StereoRenderResult(
        left_image=compact.left_image,
        right_image=compact.right_image,
        left_valid_mask=left_valid,
        right_valid_mask=right_valid,
        left_hole_mask=left_hole,
        right_hole_mask=right_hole,
    )
