"""Lightweight immutable VDPP identity; importing this never imports PyTorch."""

from __future__ import annotations

from typing import Any

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
        "input_normalization": "canonical-scene-percentile-v1-no-frame-renorm",
        "output_range_policy": "native-resize-then-clip-0-1-v1",
        "alignment": "vdpp-global-affine-v1",
        "spatial_transform": "ceil14-half-ceil14-min224-bilinear-align-corners-v1",
        "padded_input_shape": [padded_height, padded_width],
        "working_shape": [working_height, working_width],
    }
