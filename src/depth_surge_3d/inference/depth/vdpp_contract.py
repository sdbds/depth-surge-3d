"""Lightweight immutable VDPP identity; importing this never imports PyTorch."""

from __future__ import annotations

from typing import Any

from ...core.vdpp_calibration import (
    FALLBACK_REASON_ORDER,
    MAX_POSTCLIP_MEAN_DRIFT,
    MAX_PRECLIP_OUT_OF_RANGE_FRACTION,
    MIDPOINT_CODE,
    MIN_CORRELATION,
    MIN_PAIR_COUNT,
    MIN_POSITIVE_SCALE,
    MIN_POSTCLIP_CONTRAST_RATIO,
    MODEL_MIDPOINT_VALUE,
    PHYSICAL_BOUND_BASE_ULPS,
    PHYSICAL_BOUND_REFERENCE_FLOOR,
    PHYSICAL_BOUND_ULPS_PER_PLANNED_TILE,
    STATS_TILE_PIXELS,
    VARIANCE_EPSILON,
)
from .vdpp_artifact import (
    VDPP_CHECKPOINT_SHA256,
    VDPP_CHECKPOINT_SIZE,
    VDPP_UPSTREAM_RELEASE,
    VDPP_UPSTREAM_REVISION,
)


VDPP_WINDOW_SIZE = 32
VDPP_OVERLAP = 4
VDPP_STRIDE = VDPP_WINDOW_SIZE - VDPP_OVERLAP
VDPP_DOWNSIZE = True
VDPP_PRECISION = "fp32"
VDPP_VENDOR_PORT_VERSION = 1
VDPP_CHECKPOINT_COMPATIBILITY = "released-zero-shift-head-v1"

VDPP_MODEL_CONFIG: dict[str, Any] = {
    "encoder": "vits",
    "features": 64,
    "out_channels": [48, 96, 192, 384],
    "use_bn": False,
    "use_clstoken": False,
    "num_frames": VDPP_WINDOW_SIZE,
    "max_depth": 1.0,
    "pe": "ape",
}


def ceil_multiple(value: float | int, multiple: int = 14) -> int:
    return int((value + multiple - 1) // multiple) * multiple


def vdpp_model_identity() -> dict[str, Any]:
    return {
        "name": "vdpp",
        "upstream_release": VDPP_UPSTREAM_RELEASE,
        "upstream_revision": VDPP_UPSTREAM_REVISION,
        "checkpoint_sha256": VDPP_CHECKPOINT_SHA256,
        "checkpoint_size": VDPP_CHECKPOINT_SIZE,
        "architecture": "vits-temporal-dpt",
        "model_config": dict(VDPP_MODEL_CONFIG),
        "vendor_port_version": VDPP_VENDOR_PORT_VERSION,
        "checkpoint_compatibility": VDPP_CHECKPOINT_COMPATIBILITY,
    }


def build_vdpp_execution_plan(native_shape: tuple[int, int]) -> dict[str, Any]:
    if (
        not isinstance(native_shape, tuple)
        or len(native_shape) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in native_shape
        )
    ):
        raise ValueError("VDPP native shape must contain two positive integers")
    padded_height = ceil_multiple(native_shape[0], 14)
    padded_width = ceil_multiple(native_shape[1], 14)
    working_height = max(ceil_multiple(padded_height / 2, 14), 224)
    working_width = max(ceil_multiple(padded_width / 2, 14), 224)
    return {
        "batch_size": 1,
        "window_size": VDPP_WINDOW_SIZE,
        "overlap": VDPP_OVERLAP,
        "stride": VDPP_STRIDE,
        "tail_padding": False,
        "downsize": VDPP_DOWNSIZE,
        "precision": VDPP_PRECISION,
        "input_normalization": "per-frame-minmax-excluding-midpoint-code-v2",
        "input_arithmetic": "u16-extrema-float32-subtract-divide-v1",
        "tile_iteration": "c-row-major-fixed-262144-v1",
        "tile_reducer": "float64-two-pass-numpy-sum-centered-c-order-skip-empty-v1",
        "chan_merge": "python-f64-sequential-nonempty-canonical-zero-v2",
        "midpoint_code_policy": "preserve-u16-32768-heuristic-v2",
        "calibration_fit": "positive-ols-chan-v1",
        "degenerate_policy": "prescan-flat-frame-copy-u16-v1",
        "fit_quality_policy": "corr-contrast-mean-drift-v1",
        "clip_policy": "preclip-fraction-gate-then-clip-v1",
        "base_fallback_policy": "copy-source-u16-v1",
        "fit_pair_conversion": "raw-f32-source-decode-f32-then-f64-v1",
        "candidate_arithmetic": "raw-f32-to-f64-affine-clip-cast-f32-v1",
        "candidate_statistics": "float64-chan-over-candidate-f32-v1",
        "encoding_policy": "canonical-encoder-then-restore-midpoint-v1",
        "calibration_precision": "float32-input-float64-fit-float32-candidate-v2",
        "physical_bound_policy": "snap-outward-planned-tile-ulp-budget-v2",
        "physical_bound_tile_count": ("shot-length-times-ceil-frame-pixels-over-tile-v1"),
        "signed_zero_policy": "canonical-positive-zero-all-diagnostic-floats-v1",
        "diagnostic_float_equality": "finite-binary64-float-hex-v1",
        "derived_diagnostics_policy": "recompute-from-canonical-persisted-moments-v1",
        "variance_diagnostics": "persist-compared-population-variance-v1",
        "calibration_diagnostics_schema": "strict-exact-keys-derived-tile-budget-v5",
        "partial_resume_numeric_runtime_policy": ("interpreter-platform-versions-reducer-probe-v1"),
        "opencv_runtime_policy": "version-bound-decoded-u16-semantics-v1",
        "fallback_reason_order": list(FALLBACK_REASON_ORDER),
        "midpoint_code": MIDPOINT_CODE,
        "model_midpoint_value": MODEL_MIDPOINT_VALUE,
        "stats_tile_pixels": STATS_TILE_PIXELS,
        "min_pair_count": MIN_PAIR_COUNT,
        "variance_epsilon": VARIANCE_EPSILON,
        "physical_bound_base_ulps": PHYSICAL_BOUND_BASE_ULPS,
        "physical_bound_ulps_per_planned_tile": PHYSICAL_BOUND_ULPS_PER_PLANNED_TILE,
        "physical_bound_reference_floor": PHYSICAL_BOUND_REFERENCE_FLOOR,
        "min_positive_scale": MIN_POSITIVE_SCALE,
        "min_correlation": MIN_CORRELATION,
        "min_postclip_contrast_ratio": MIN_POSTCLIP_CONTRAST_RATIO,
        "max_postclip_mean_drift": MAX_POSTCLIP_MEAN_DRIFT,
        "max_preclip_out_of_range_fraction": MAX_PRECLIP_OUT_OF_RANGE_FRACTION,
        "spatial_transform": "ceil14-half-ceil14-min224-bilinear-align-corners-v1",
        "padded_input_shape": [padded_height, padded_width],
        "working_shape": [working_height, working_width],
    }
