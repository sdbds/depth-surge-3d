"""Shared identity and execution-plan contract for VDA offline inference."""

from __future__ import annotations

from typing import Any

VDA_INFER_LEN = 32
VDA_OVERLAP = 10
VDA_KEYFRAMES = (0, 12, 24, 25, 26, 27, 28, 29, 30, 31)
VDA_INTERP_LEN = 8
VDA_FRAME_STEP = VDA_INFER_LEN - VDA_OVERLAP
VDA_INFERENCE_ALGORITHM = "vda-offline-shot-v1"
V2_FALLBACK_POLICY = "v2-uniform-halving-v1"
V2_MINIMUM_INPUT_SIZE = 384


def build_v2_execution_plan(
    requested_input_size: int,
    effective_input_size: int,
    *,
    fp32: bool = False,
) -> dict[str, Any]:
    """Build the persisted identity for one uniform-resolution V2 raw stage."""
    if (
        isinstance(requested_input_size, bool)
        or not isinstance(requested_input_size, int)
        or requested_input_size < 1
        or isinstance(effective_input_size, bool)
        or not isinstance(effective_input_size, int)
        or effective_input_size < 1
    ):
        raise ValueError("V2 execution-plan input sizes must be positive")
    return {
        "requested_input_size": int(requested_input_size),
        "effective_input_size": int(effective_input_size),
        "precision": "fp32" if fp32 else "fp16",
        "fallback_policy": V2_FALLBACK_POLICY,
    }


def next_v2_input_size(current_size: int) -> int | None:
    """Return the next whole-stage fallback size, or None at the floor."""
    if current_size <= V2_MINIMUM_INPUT_SIZE:
        return None
    return max(V2_MINIMUM_INPUT_SIZE, current_size // 2)


def valid_v2_input_sizes(requested_size: int) -> set[int]:
    """Return every effective size reachable under the persisted fallback policy."""
    sizes = {requested_size}
    current = requested_size
    while (next_size := next_v2_input_size(current)) is not None:
        sizes.add(next_size)
        current = next_size
    return sizes


def is_compatible_v2_execution_plan(
    persisted_plan: Any,
    requested_plan: dict[str, Any],
) -> bool:
    """Validate a persisted plan against the current request and policy."""
    expected_keys = {
        "requested_input_size",
        "effective_input_size",
        "precision",
        "fallback_policy",
    }
    if not isinstance(persisted_plan, dict) or set(persisted_plan) != expected_keys:
        return False
    requested_size = persisted_plan.get("requested_input_size")
    effective_size = persisted_plan.get("effective_input_size")
    if (
        isinstance(requested_size, bool)
        or not isinstance(requested_size, int)
        or requested_size < 1
        or isinstance(effective_size, bool)
        or not isinstance(effective_size, int)
        or effective_size < 1
    ):
        return False
    if requested_size != requested_plan.get("requested_input_size"):
        return False
    if persisted_plan.get("precision") != requested_plan.get("precision"):
        return False
    if persisted_plan.get("fallback_policy") != requested_plan.get("fallback_policy"):
        return False
    return effective_size in valid_v2_input_sizes(requested_size)
