"""Torch-free numerical contract for VDPP shot-global calibration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


MIDPOINT_CODE = 32768
MODEL_MIDPOINT_VALUE = 0.5
STATS_TILE_PIXELS = 262144
MIN_PAIR_COUNT = 2
VARIANCE_EPSILON = 1e-12
PHYSICAL_BOUND_BASE_ULPS = 4
PHYSICAL_BOUND_ULPS_PER_PLANNED_TILE = 64
PHYSICAL_BOUND_REFERENCE_FLOOR = 1.0
MIN_POSITIVE_SCALE = 1e-8
MIN_CORRELATION = 0.50
MIN_POSTCLIP_CONTRAST_RATIO = 0.50
MAX_POSTCLIP_MEAN_DRIFT = 0.01
MAX_PRECLIP_OUT_OF_RANGE_FRACTION = 0.01

FALLBACK_REASON_ORDER = (
    "source_no_range",
    "too_few_pairs",
    "nonfinite_statistics",
    "source_variance",
    "raw_variance",
    "nonfinite_fit",
    "scale_below_minimum",
    "correlation",
    "contrast",
    "mean_drift",
    "preclip_out_of_range",
)

CALIBRATION_KEYS = (
    "mode",
    "pair_count",
    "midpoint_count",
    "midpoint_fraction",
    "flat_frame_count",
    "source_mean",
    "source_variance",
    "source_std",
    "raw_mean",
    "raw_variance",
    "raw_std",
    "covariance",
    "correlation",
    "scale",
    "shift",
    "candidate_mean",
    "candidate_std",
    "postclip_contrast_ratio",
    "postclip_mean_drift",
    "preclip_low_fraction",
    "preclip_high_fraction",
    "fallback_reason",
)

_MOMENT_FIELDS = (
    "source_mean",
    "source_variance",
    "source_std",
    "raw_mean",
    "raw_variance",
    "raw_std",
    "covariance",
)
_FIT_FIELDS = ("correlation", "scale", "shift")
_QUALITY_FIELDS = (
    "candidate_mean",
    "candidate_std",
    "postclip_contrast_ratio",
    "postclip_mean_drift",
    "preclip_low_fraction",
    "preclip_high_fraction",
)


class NumericalContractError(ValueError):
    """A finite statistic exceeded its permitted numerical boundary budget."""


@dataclass(frozen=True)
class ScalarTileState:
    """Population-moment state for one scalar tile or merged tile prefix."""

    count: int
    mean: float
    m2: float

    @classmethod
    def empty(cls) -> "ScalarTileState":
        return cls(0, 0.0, 0.0)


@dataclass(frozen=True)
class PairTileState:
    """Population-moment and co-moment state for paired fit values."""

    count: int
    mean_x: float
    mean_y: float
    m2_x: float
    m2_y: float
    c_xy: float

    @classmethod
    def empty(cls) -> "PairTileState":
        return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0)


def canonicalize_float(value: float) -> float:
    """Require finite built-in binary64 and collapse signed zero."""

    if type(value) is not float:
        raise TypeError("VDPP diagnostic values must be built-in floats")
    if not math.isfinite(value):
        raise ValueError("VDPP diagnostic values must be finite")
    return 0.0 if value == 0.0 else value


def _positive_int(value: object, description: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{description} must be a positive integer")
    return value


def planned_tile_count(shot_length: int, native_shape: tuple[int, int]) -> int:
    """Return the persisted-structure upper bound used for ULP allowances."""

    length = _positive_int(shot_length, "VDPP shot length")
    if (
        type(native_shape) is not tuple
        or len(native_shape) != 2
        or any(type(value) is not int or value < 1 for value in native_shape)
    ):
        raise ValueError("VDPP native shape must contain two positive integers")
    frame_pixels = native_shape[0] * native_shape[1]
    tiles_per_frame = (frame_pixels + STATS_TILE_PIXELS - 1) // STATS_TILE_PIXELS
    return length * tiles_per_frame


def normalize_bounded_stat(
    value: float,
    low: float,
    high: float | None = None,
    *,
    planned_tile_count: int,
) -> float:
    """Snap finite outward rounding within the planned-tile ULP budget."""

    normalized = canonicalize_float(value)
    if type(low) is not float or not math.isfinite(low):
        raise TypeError("VDPP statistic lower bound must be a finite float")
    if high is not None and (type(high) is not float or not math.isfinite(high)):
        raise TypeError("VDPP statistic upper bound must be a finite float")
    if high is not None and high < low:
        raise ValueError("VDPP statistic upper bound must not be below its lower bound")
    tile_count = _positive_int(planned_tile_count, "VDPP planned tile count")
    reference = max(
        PHYSICAL_BOUND_REFERENCE_FLOOR,
        abs(low),
        abs(high) if high is not None else 0.0,
    )
    budget_ulps = PHYSICAL_BOUND_BASE_ULPS + PHYSICAL_BOUND_ULPS_PER_PLANNED_TILE * tile_count
    try:
        slack = budget_ulps * math.ulp(reference)
    except OverflowError as exc:
        raise ValueError("VDPP planned tile count is too large") from exc

    if normalized < low:
        if low - normalized <= slack:
            return canonicalize_float(float(low))
        raise NumericalContractError(
            f"VDPP statistic {normalized!r} is below physical bound {low!r}"
        )
    if high is not None and normalized > high:
        if normalized - high <= slack:
            return canonicalize_float(float(high))
        raise NumericalContractError(
            f"VDPP statistic {normalized!r} is above physical bound {high!r}"
        )
    return normalized


def _validate_tile_inputs(
    values: np.ndarray,
    eligible_mask: np.ndarray,
    *,
    description: str,
) -> None:
    if not isinstance(values, np.ndarray) or values.dtype != np.float32 or values.ndim != 1:
        raise TypeError(f"{description} values must be a flattened float32 NumPy array")
    if values.size > STATS_TILE_PIXELS:
        raise ValueError(f"{description} tile exceeds STATS_TILE_PIXELS")
    if (
        not isinstance(eligible_mask, np.ndarray)
        or eligible_mask.dtype != np.bool_
        or eligible_mask.ndim != 1
        or eligible_mask.shape != values.shape
    ):
        raise TypeError(f"{description} eligibility mask must be a matching boolean array")


def reduce_scalar_tile(
    values_f32: np.ndarray,
    eligible_mask: np.ndarray,
) -> ScalarTileState:
    """Reduce one fixed tile with the canonical two-pass NumPy sequence."""

    _validate_tile_inputs(values_f32, eligible_mask, description="VDPP scalar")
    count = int(np.count_nonzero(eligible_mask))
    if count == 0:
        return ScalarTileState.empty()
    values64 = np.ascontiguousarray(values_f32[eligible_mask], dtype=np.float64)
    mean = canonicalize_float(float(np.sum(values64, dtype=np.float64)) / count)
    centered = values64 - mean
    m2 = canonicalize_float(float(np.sum(centered * centered, dtype=np.float64)))
    return ScalarTileState(count, mean, m2)


def reduce_pair_tile(
    x_f32: np.ndarray,
    y_f32: np.ndarray,
    eligible_mask: np.ndarray,
) -> PairTileState:
    """Reduce one fixed paired tile with the canonical two-pass sequence."""

    _validate_tile_inputs(x_f32, eligible_mask, description="VDPP pair x")
    if not isinstance(y_f32, np.ndarray) or y_f32.dtype != np.float32:
        raise TypeError("VDPP pair y values must be a float32 NumPy array")
    if y_f32.ndim != 1 or y_f32.shape != x_f32.shape:
        raise ValueError("VDPP pair tiles must have matching flattened shapes")
    count = int(np.count_nonzero(eligible_mask))
    if count == 0:
        return PairTileState.empty()

    x64 = np.ascontiguousarray(x_f32[eligible_mask], dtype=np.float64)
    y64 = np.ascontiguousarray(y_f32[eligible_mask], dtype=np.float64)
    mean_x = canonicalize_float(float(np.sum(x64, dtype=np.float64)) / count)
    mean_y = canonicalize_float(float(np.sum(y64, dtype=np.float64)) / count)
    centered_x = x64 - mean_x
    centered_y = y64 - mean_y
    m2_x = canonicalize_float(float(np.sum(centered_x * centered_x, dtype=np.float64)))
    m2_y = canonicalize_float(float(np.sum(centered_y * centered_y, dtype=np.float64)))
    c_xy = canonicalize_float(float(np.sum(centered_x * centered_y, dtype=np.float64)))
    return PairTileState(count, mean_x, mean_y, m2_x, m2_y, c_xy)


def merge_scalar_states(a: ScalarTileState, b: ScalarTileState) -> ScalarTileState:
    """Merge non-empty scalar tiles sequentially with fixed scalar operations."""

    if not isinstance(a, ScalarTileState) or not isinstance(b, ScalarTileState):
        raise TypeError("VDPP scalar Chan merge requires ScalarTileState values")
    if a.count == 0:
        return b
    if b.count == 0:
        return a
    n_a = a.count
    n_b = b.count
    n = n_a + n_b
    delta = canonicalize_float(b.mean - a.mean)
    mean = canonicalize_float(a.mean + ((delta * n_b) / n))
    correction = (((delta * delta) * n_a) * n_b) / n
    m2 = canonicalize_float((a.m2 + b.m2) + correction)
    return ScalarTileState(n, mean, m2)


def merge_pair_states(a: PairTileState, b: PairTileState) -> PairTileState:
    """Merge non-empty pair tiles sequentially with fixed scalar operations."""

    if not isinstance(a, PairTileState) or not isinstance(b, PairTileState):
        raise TypeError("VDPP pair Chan merge requires PairTileState values")
    if a.count == 0:
        return b
    if b.count == 0:
        return a
    n_a = a.count
    n_b = b.count
    n = n_a + n_b
    delta_x = canonicalize_float(b.mean_x - a.mean_x)
    delta_y = canonicalize_float(b.mean_y - a.mean_y)
    mean_x = canonicalize_float(a.mean_x + ((delta_x * n_b) / n))
    mean_y = canonicalize_float(a.mean_y + ((delta_y * n_b) / n))
    correction_x = (((delta_x * delta_x) * n_a) * n_b) / n
    correction_y = (((delta_y * delta_y) * n_a) * n_b) / n
    correction_xy = (((delta_x * delta_y) * n_a) * n_b) / n
    m2_x = canonicalize_float((a.m2_x + b.m2_x) + correction_x)
    m2_y = canonicalize_float((a.m2_y + b.m2_y) + correction_y)
    c_xy = canonicalize_float((a.c_xy + b.c_xy) + correction_xy)
    return PairTileState(n, mean_x, mean_y, m2_x, m2_y, c_xy)


def collect_vdpp_numeric_runtime_probe() -> dict[str, str]:
    """Fingerprint the exact NumPy reductions and Python Chan operations in use."""

    scalar_values = np.array(
        [
            0.2712900638580322,
            0.7885109782218933,
            0.9661115407943726,
            0.8313783407211304,
            0.8321383595466614,
            0.8936092853546143,
            0.4458301067352295,
            0.4170106053352356,
            0.8177664875984192,
            0.1028597354888916,
            0.7648928761482239,
            0.16799843311309814,
            0.09131741523742676,
            0.8898534178733826,
            0.634506344795227,
            0.13384193181991577,
            0.39620745182037354,
        ],
        dtype=np.float32,
    )
    scalar = reduce_scalar_tile(
        scalar_values,
        np.ones(scalar_values.shape, dtype=np.bool_),
    )

    pair_x = np.array(
        [0.6486990451812744, 0.6523162722587585, 0.3402478098869324],
        dtype=np.float32,
    )
    pair_y = np.array(
        [0.3692907691001892, 0.03194546699523926, 0.4765521287918091],
        dtype=np.float32,
    )
    pair = reduce_pair_tile(pair_x, pair_y, np.ones(3, dtype=np.bool_))

    chan_states = (
        PairTileState(2, 0.25, 0.5, 0.125, 0.5, 0.25),
        PairTileState(3, 0.75, 0.25, 0.375, 0.125, -0.125),
        PairTileState(4, 0.125, 0.875, 0.25, 0.75, -0.2),
    )
    chan_pair = merge_pair_states(
        merge_pair_states(chan_states[0], chan_states[1]),
        chan_states[2],
    )
    chan_scalar = merge_scalar_states(
        merge_scalar_states(
            ScalarTileState(
                chan_states[0].count,
                chan_states[0].mean_x,
                chan_states[0].m2_x,
            ),
            ScalarTileState(
                chan_states[1].count,
                chan_states[1].mean_x,
                chan_states[1].m2_x,
            ),
        ),
        ScalarTileState(
            chan_states[2].count,
            chan_states[2].mean_x,
            chan_states[2].m2_x,
        ),
    )

    return {
        "schema": "vdpp-numeric-reducer-probe-v1",
        "scalar_mean_hex": scalar.mean.hex(),
        "scalar_m2_hex": scalar.m2.hex(),
        "pair_mean_x_hex": pair.mean_x.hex(),
        "pair_mean_y_hex": pair.mean_y.hex(),
        "pair_m2_x_hex": pair.m2_x.hex(),
        "pair_m2_y_hex": pair.m2_y.hex(),
        "pair_c_xy_hex": pair.c_xy.hex(),
        "chan_mean_hex": chan_scalar.mean.hex(),
        "chan_variance_hex": (chan_scalar.m2 / chan_scalar.count).hex(),
        "chan_pair_mean_x_hex": chan_pair.mean_x.hex(),
        "chan_pair_mean_y_hex": chan_pair.mean_y.hex(),
        "chan_pair_m2_x_hex": chan_pair.m2_x.hex(),
        "chan_pair_m2_y_hex": chan_pair.m2_y.hex(),
        "chan_pair_c_xy_hex": chan_pair.c_xy.hex(),
    }


def finalize_pair_moments(
    state: PairTileState,
    *,
    planned_tile_count: int,
) -> dict[str, float]:
    """Finalize only persisted primitive moments, never derived fit fields."""

    if not isinstance(state, PairTileState) or state.count < 1:
        raise ValueError("VDPP fit statistics require at least one pair")
    raw_mean = canonicalize_float(state.mean_x)
    source_mean = normalize_bounded_stat(
        state.mean_y,
        0.0,
        1.0,
        planned_tile_count=planned_tile_count,
    )
    raw_variance = normalize_bounded_stat(
        canonicalize_float(state.m2_x / state.count),
        0.0,
        planned_tile_count=planned_tile_count,
    )
    source_variance = normalize_bounded_stat(
        canonicalize_float(state.m2_y / state.count),
        0.0,
        0.25,
        planned_tile_count=planned_tile_count,
    )
    covariance = canonicalize_float(state.c_xy / state.count)
    return {
        "source_mean": source_mean,
        "source_variance": source_variance,
        "raw_mean": raw_mean,
        "raw_variance": raw_variance,
        "covariance": covariance,
    }


def normalize_vdpp_input_frame(source_u16: np.ndarray, target_f32: np.ndarray) -> bool:
    """Fill one model input frame using exact uint16 extrema and float32 math."""

    if not isinstance(source_u16, np.ndarray) or source_u16.dtype != np.uint16:
        raise TypeError("VDPP source frame must be a uint16 NumPy array")
    if source_u16.ndim != 2:
        raise ValueError("VDPP source frame must be two-dimensional")
    if (
        not isinstance(target_f32, np.ndarray)
        or target_f32.dtype != np.float32
        or target_f32.shape != source_u16.shape
    ):
        raise TypeError("VDPP target frame must be a matching float32 NumPy array")

    frame_mask = source_u16 != np.uint16(MIDPOINT_CODE)
    if not bool(np.any(frame_mask)):
        target_f32.fill(np.float32(MODEL_MIDPOINT_VALUE))
        return False
    lo_code = int(np.min(source_u16, where=frame_mask, initial=np.iinfo(np.uint16).max))
    hi_code = int(np.max(source_u16, where=frame_mask, initial=np.iinfo(np.uint16).min))
    if lo_code == hi_code:
        target_f32.fill(np.float32(MODEL_MIDPOINT_VALUE))
        return False

    target_f32[...] = source_u16
    target_f32 -= np.float32(lo_code)
    target_f32 /= np.float32(hi_code - lo_code)
    np.logical_not(frame_mask, out=frame_mask)
    target_f32[frame_mask] = np.float32(MODEL_MIDPOINT_VALUE)
    return True


def candidate_tile(
    raw_tile_f32: np.ndarray,
    scale_f64: float,
    shift_f64: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the canonical float64 affine and clip into float32 candidate values."""

    if (
        not isinstance(raw_tile_f32, np.ndarray)
        or raw_tile_f32.dtype != np.float32
        or raw_tile_f32.ndim != 1
    ):
        raise TypeError("VDPP raw candidate tile must be a flattened float32 array")
    if raw_tile_f32.size > STATS_TILE_PIXELS:
        raise ValueError("VDPP candidate tile exceeds STATS_TILE_PIXELS")
    scale = canonicalize_float(scale_f64)
    shift = canonicalize_float(shift_f64)
    preclip64 = raw_tile_f32.astype(np.float64)
    preclip64 *= scale
    preclip64 += shift
    candidate32 = np.empty(raw_tile_f32.shape, dtype=np.float32)
    np.clip(preclip64, np.float64(0.0), np.float64(1.0), out=candidate32)
    return preclip64, candidate32


def canonical_derived_diagnostics(
    *,
    midpoint_count: int,
    shot_pixels: int,
    source_mean: float | None,
    source_variance: float | None,
    raw_mean: float | None,
    raw_variance: float | None,
    covariance: float | None,
    candidate_mean: float | None,
    candidate_std: float | None,
    preclip_low_fraction: float | None,
    preclip_high_fraction: float | None,
    planned_tile_count: int,
) -> dict[str, float | None]:
    """Derive diagnostics through the one authoritative persisted-moment graph."""

    if type(midpoint_count) is not int or midpoint_count < 0:
        raise ValueError("VDPP midpoint count must be a non-negative integer")
    if type(shot_pixels) is not int or shot_pixels < 1 or midpoint_count > shot_pixels:
        raise ValueError("VDPP shot pixel count is invalid")
    tile_count = _positive_int(planned_tile_count, "VDPP planned tile count")
    result: dict[str, float | None] = {
        "midpoint_fraction": canonicalize_float(midpoint_count / shot_pixels),
        "source_std": None,
        "raw_std": None,
        "correlation": None,
        "scale": None,
        "shift": None,
        "postclip_contrast_ratio": None,
        "postclip_mean_drift": None,
        "preclip_out_of_range_fraction": None,
    }

    moments = (source_mean, source_variance, raw_mean, raw_variance, covariance)
    if all(value is not None for value in moments):
        source_mean_value = normalize_bounded_stat(
            canonicalize_float(source_mean),  # type: ignore[arg-type]
            0.0,
            1.0,
            planned_tile_count=tile_count,
        )
        source_variance_value = normalize_bounded_stat(
            canonicalize_float(source_variance),  # type: ignore[arg-type]
            0.0,
            0.25,
            planned_tile_count=tile_count,
        )
        raw_mean_value = canonicalize_float(raw_mean)  # type: ignore[arg-type]
        raw_variance_value = normalize_bounded_stat(
            canonicalize_float(raw_variance),  # type: ignore[arg-type]
            0.0,
            planned_tile_count=tile_count,
        )
        covariance_value = canonicalize_float(covariance)  # type: ignore[arg-type]
        source_std = normalize_bounded_stat(
            canonicalize_float(math.sqrt(source_variance_value)),
            0.0,
            0.5,
            planned_tile_count=tile_count,
        )
        raw_std = normalize_bounded_stat(
            canonicalize_float(math.sqrt(raw_variance_value)),
            0.0,
            planned_tile_count=tile_count,
        )
        result["source_std"] = source_std
        result["raw_std"] = raw_std
        if source_variance_value > 0.0 and raw_variance_value > 0.0:
            correlation = normalize_bounded_stat(
                canonicalize_float(
                    covariance_value / math.sqrt(raw_variance_value * source_variance_value)
                ),
                -1.0,
                1.0,
                planned_tile_count=tile_count,
            )
            scale = canonicalize_float(covariance_value / raw_variance_value)
            shift = canonicalize_float(source_mean_value - scale * raw_mean_value)
            result["correlation"] = correlation
            result["scale"] = scale
            result["shift"] = shift

        quality = (
            candidate_mean,
            candidate_std,
            preclip_low_fraction,
            preclip_high_fraction,
        )
        if all(value is not None for value in quality):
            candidate_mean_value = normalize_bounded_stat(
                canonicalize_float(candidate_mean),  # type: ignore[arg-type]
                0.0,
                1.0,
                planned_tile_count=tile_count,
            )
            candidate_std_value = normalize_bounded_stat(
                canonicalize_float(candidate_std),  # type: ignore[arg-type]
                0.0,
                0.5,
                planned_tile_count=tile_count,
            )
            low = normalize_bounded_stat(
                canonicalize_float(preclip_low_fraction),  # type: ignore[arg-type]
                0.0,
                1.0,
                planned_tile_count=tile_count,
            )
            high = normalize_bounded_stat(
                canonicalize_float(preclip_high_fraction),  # type: ignore[arg-type]
                0.0,
                1.0,
                planned_tile_count=tile_count,
            )
            if source_std <= 0.0:
                raise ValueError("VDPP quality diagnostics require positive source variance")
            result["postclip_contrast_ratio"] = normalize_bounded_stat(
                canonicalize_float(candidate_std_value / source_std),
                0.0,
                planned_tile_count=tile_count,
            )
            result["postclip_mean_drift"] = normalize_bounded_stat(
                canonicalize_float(abs(candidate_mean_value - source_mean_value)),
                0.0,
                1.0,
                planned_tile_count=tile_count,
            )
            result["preclip_out_of_range_fraction"] = normalize_bounded_stat(
                canonicalize_float(low + high),
                0.0,
                1.0,
                planned_tile_count=tile_count,
            )
    elif any(value is not None for value in moments):
        raise ValueError("VDPP primitive moment diagnostics must be supplied as one group")
    return result


def _require_count(value: object, name: str, maximum: int) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        raise ValueError(f"VDPP calibration {name} is out of range")
    return value


def _require_float_or_null(value: object, name: str) -> float | None:
    if value is None:
        return None
    try:
        return canonicalize_float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"VDPP calibration {name} must be a finite built-in float") from exc


def _same_float(actual: float, expected: float, name: str) -> None:
    if actual.hex() != expected.hex():
        raise ValueError(f"VDPP calibration {name} does not match its canonical derivation")


def _require_nullability(
    values: dict[str, Any],
    fields: tuple[str, ...],
    required: bool,
) -> None:
    for field in fields:
        if required and values[field] is None:
            raise ValueError(f"VDPP calibration {field} is required")
        if not required and values[field] is not None:
            raise ValueError(f"VDPP calibration {field} must be null")


def _canonicalize_present_fields(
    values: dict[str, Any],
    *,
    tile_count: int,
) -> None:
    bounded: dict[str, tuple[float, float | None]] = {
        "source_mean": (0.0, 1.0),
        "source_variance": (0.0, 0.25),
        "source_std": (0.0, 0.5),
        "raw_variance": (0.0, None),
        "raw_std": (0.0, None),
        "correlation": (-1.0, 1.0),
        "candidate_mean": (0.0, 1.0),
        "candidate_std": (0.0, 0.5),
        "postclip_contrast_ratio": (0.0, None),
        "postclip_mean_drift": (0.0, 1.0),
        "preclip_low_fraction": (0.0, 1.0),
        "preclip_high_fraction": (0.0, 1.0),
    }
    numeric_fields = (*_MOMENT_FIELDS, *_FIT_FIELDS, *_QUALITY_FIELDS)
    for field in numeric_fields:
        value = _require_float_or_null(values[field], field)
        if value is not None and field in bounded:
            low, high = bounded[field]
            value = normalize_bounded_stat(
                value,
                low,
                high,
                planned_tile_count=tile_count,
            )
        values[field] = value


def canonicalize_vdpp_calibration_diagnostics(
    calibration: object,
    *,
    shot_length: int,
    native_shape: tuple[int, int],
) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
    """Canonicalize and strictly validate a writer-supplied calibration record."""

    tile_count = planned_tile_count(shot_length, native_shape)
    if type(calibration) is not dict:
        raise TypeError("VDPP calibration diagnostics must be a dictionary")
    if set(calibration) != set(CALIBRATION_KEYS):
        raise ValueError("VDPP calibration diagnostics have missing or unknown keys")
    values = {key: calibration[key] for key in CALIBRATION_KEYS}
    mode = values["mode"]
    if type(mode) is not str or mode not in {"ols", "base_fallback", "all_midpoint"}:
        raise ValueError("VDPP calibration mode is invalid")
    reason = values["fallback_reason"]
    if reason is not None and (type(reason) is not str or reason not in FALLBACK_REASON_ORDER):
        raise ValueError("VDPP calibration fallback reason is invalid")

    height, width = native_shape
    shot_pixels = shot_length * height * width
    pair_count = _require_count(values["pair_count"], "pair_count", shot_pixels)
    midpoint_count = _require_count(values["midpoint_count"], "midpoint_count", shot_pixels)
    flat_frame_count = _require_count(values["flat_frame_count"], "flat_frame_count", shot_length)
    if pair_count > shot_pixels - midpoint_count:
        raise ValueError("VDPP calibration pair count includes midpoint-coded pixels")

    midpoint_fraction = _require_float_or_null(values["midpoint_fraction"], "midpoint_fraction")
    if midpoint_fraction is None:
        raise ValueError("VDPP calibration midpoint_fraction is required")
    midpoint_fraction = normalize_bounded_stat(
        midpoint_fraction,
        0.0,
        1.0,
        planned_tile_count=tile_count,
    )
    expected_midpoint_fraction = canonicalize_float(midpoint_count / shot_pixels)
    _same_float(midpoint_fraction, expected_midpoint_fraction, "midpoint_fraction")
    values["midpoint_fraction"] = expected_midpoint_fraction
    _canonicalize_present_fields(values, tile_count=tile_count)

    if mode == "all_midpoint":
        _require_nullability(values, _MOMENT_FIELDS, False)
        _require_nullability(values, _FIT_FIELDS, False)
        _require_nullability(values, _QUALITY_FIELDS, False)
        if (
            reason is not None
            or pair_count != 0
            or midpoint_count != shot_pixels
            or flat_frame_count != shot_length
        ):
            raise ValueError("VDPP all_midpoint diagnostics are inconsistent")
        return values

    if mode == "ols" and reason is not None:
        raise ValueError("VDPP OLS calibration cannot have a fallback reason")
    if mode == "base_fallback" and reason is None:
        raise ValueError("VDPP base fallback requires a reason")
    if mode != "base_fallback" and reason is not None:
        raise ValueError("VDPP fallback reason requires base_fallback mode")

    if reason in {"source_no_range", "too_few_pairs", "nonfinite_statistics"}:
        moment_required = False
        fit_kind = "none"
        quality_required = False
    elif reason in {"source_variance", "raw_variance"}:
        moment_required = True
        fit_kind = "none"
        quality_required = False
    elif reason == "nonfinite_fit":
        moment_required = True
        fit_kind = "correlation"
        quality_required = False
    elif reason in {"scale_below_minimum", "correlation"}:
        moment_required = True
        fit_kind = "all"
        quality_required = False
    elif reason in {"contrast", "mean_drift", "preclip_out_of_range"} or mode == "ols":
        moment_required = True
        fit_kind = "all"
        quality_required = True
    else:
        raise ValueError("VDPP calibration fallback reason is inconsistent")

    _require_nullability(values, _MOMENT_FIELDS, moment_required)
    if fit_kind == "all":
        _require_nullability(values, _FIT_FIELDS, True)
    elif fit_kind == "correlation":
        if (
            values["correlation"] is None
            or values["scale"] is not None
            or values["shift"] is not None
        ):
            raise ValueError("VDPP nonfinite_fit diagnostics require correlation only")
    else:
        _require_nullability(values, _FIT_FIELDS, False)
    _require_nullability(values, _QUALITY_FIELDS, quality_required)

    derived: dict[str, float | None] | None = None
    if moment_required:
        expected_source_std = normalize_bounded_stat(
            canonicalize_float(math.sqrt(values["source_variance"])),
            0.0,
            0.5,
            planned_tile_count=tile_count,
        )
        expected_raw_std = normalize_bounded_stat(
            canonicalize_float(math.sqrt(values["raw_variance"])),
            0.0,
            planned_tile_count=tile_count,
        )
        _same_float(values["source_std"], expected_source_std, "source_std")
        _same_float(values["raw_std"], expected_raw_std, "raw_std")
        values["source_std"] = expected_source_std
        values["raw_std"] = expected_raw_std
        if fit_kind == "all":
            derived = canonical_derived_diagnostics(
                midpoint_count=midpoint_count,
                shot_pixels=shot_pixels,
                source_mean=values["source_mean"],
                source_variance=values["source_variance"],
                raw_mean=values["raw_mean"],
                raw_variance=values["raw_variance"],
                covariance=values["covariance"],
                candidate_mean=values["candidate_mean"] if quality_required else None,
                candidate_std=values["candidate_std"] if quality_required else None,
                preclip_low_fraction=(values["preclip_low_fraction"] if quality_required else None),
                preclip_high_fraction=(
                    values["preclip_high_fraction"] if quality_required else None
                ),
                planned_tile_count=tile_count,
            )

    source_variance = values["source_variance"]
    raw_variance = values["raw_variance"]
    if fit_kind in {"correlation", "all"}:
        if source_variance <= 0.0 or raw_variance <= 0.0:
            raise ValueError("VDPP fit diagnostics require positive variances")
        correlation = normalize_bounded_stat(
            canonicalize_float(values["covariance"] / math.sqrt(raw_variance * source_variance)),
            -1.0,
            1.0,
            planned_tile_count=tile_count,
        )
        _same_float(values["correlation"], correlation, "correlation")
        values["correlation"] = correlation
        raw_scale = values["covariance"] / raw_variance
        raw_shift = values["source_mean"] - raw_scale * values["raw_mean"]
        finite_fit = math.isfinite(raw_scale) and math.isfinite(raw_shift)
        if fit_kind == "correlation":
            if finite_fit:
                raise ValueError("VDPP nonfinite_fit reason has finite fit parameters")
        else:
            if not finite_fit:
                raise ValueError("VDPP finite fit diagnostics contain non-finite parameters")
            scale = canonicalize_float(raw_scale)
            shift = canonicalize_float(raw_shift)
            _same_float(values["scale"], scale, "scale")
            _same_float(values["shift"], shift, "shift")
            values["scale"] = scale
            values["shift"] = shift

    preclip_total: float | None = None
    if quality_required:
        if derived is None:
            raise AssertionError("VDPP quality derivation requires moments")
        for field in ("postclip_contrast_ratio", "postclip_mean_drift"):
            expected = derived[field]
            if expected is None:
                raise ValueError(f"VDPP calibration {field} could not be derived")
            _same_float(values[field], expected, field)
            values[field] = expected
        preclip_total = derived["preclip_out_of_range_fraction"]
        if preclip_total is None:
            raise ValueError("VDPP preclip fraction could not be derived")

    identifiable_variances = (
        source_variance is not None
        and raw_variance is not None
        and source_variance > VARIANCE_EPSILON
        and raw_variance > VARIANCE_EPSILON
    )
    fit_prerequisites = (
        flat_frame_count < shot_length and pair_count >= MIN_PAIR_COUNT and identifiable_variances
    )
    if reason == "source_no_range":
        causal = pair_count == 0 and flat_frame_count == shot_length
    elif reason == "too_few_pairs":
        causal = flat_frame_count < shot_length and pair_count < MIN_PAIR_COUNT
    elif reason == "nonfinite_statistics":
        causal = flat_frame_count < shot_length and pair_count >= MIN_PAIR_COUNT
    elif reason == "source_variance":
        causal = (
            flat_frame_count < shot_length
            and pair_count >= MIN_PAIR_COUNT
            and source_variance <= VARIANCE_EPSILON
        )
    elif reason == "raw_variance":
        causal = (
            flat_frame_count < shot_length
            and pair_count >= MIN_PAIR_COUNT
            and source_variance > VARIANCE_EPSILON
            and raw_variance <= VARIANCE_EPSILON
        )
    elif reason == "nonfinite_fit":
        causal = fit_prerequisites
    elif reason == "scale_below_minimum":
        causal = fit_prerequisites and values["scale"] < MIN_POSITIVE_SCALE
    elif reason == "correlation":
        causal = (
            fit_prerequisites
            and values["scale"] >= MIN_POSITIVE_SCALE
            and values["correlation"] < MIN_CORRELATION
        )
    elif reason == "contrast":
        causal = (
            fit_prerequisites
            and values["scale"] >= MIN_POSITIVE_SCALE
            and values["correlation"] >= MIN_CORRELATION
            and values["postclip_contrast_ratio"] < MIN_POSTCLIP_CONTRAST_RATIO
        )
    elif reason == "mean_drift":
        causal = (
            fit_prerequisites
            and values["scale"] >= MIN_POSITIVE_SCALE
            and values["correlation"] >= MIN_CORRELATION
            and values["postclip_contrast_ratio"] >= MIN_POSTCLIP_CONTRAST_RATIO
            and values["postclip_mean_drift"] > MAX_POSTCLIP_MEAN_DRIFT
        )
    elif reason == "preclip_out_of_range":
        causal = (
            fit_prerequisites
            and values["scale"] >= MIN_POSITIVE_SCALE
            and values["correlation"] >= MIN_CORRELATION
            and values["postclip_contrast_ratio"] >= MIN_POSTCLIP_CONTRAST_RATIO
            and values["postclip_mean_drift"] <= MAX_POSTCLIP_MEAN_DRIFT
            and preclip_total > MAX_PRECLIP_OUT_OF_RANGE_FRACTION
        )
    else:
        causal = (
            mode == "ols"
            and fit_prerequisites
            and values["scale"] >= MIN_POSITIVE_SCALE
            and values["correlation"] >= MIN_CORRELATION
            and values["postclip_contrast_ratio"] >= MIN_POSTCLIP_CONTRAST_RATIO
            and values["postclip_mean_drift"] <= MAX_POSTCLIP_MEAN_DRIFT
            and preclip_total <= MAX_PRECLIP_OUT_OF_RANGE_FRACTION
        )
    if not causal:
        raise ValueError(
            "VDPP calibration fallback mode/reason does not match the first failed gate"
        )
    return values


def _same_scalar_representation(actual: object, canonical: object) -> bool:
    if type(actual) is not type(canonical):
        return False
    if type(actual) is float:
        return actual.hex() == canonical.hex()  # type: ignore[union-attr]
    return actual == canonical


def validate_vdpp_calibration_diagnostics(
    calibration: object,
    *,
    shot_length: int,
    native_shape: tuple[int, int],
) -> dict[str, Any]:
    """Validate persisted diagnostics and reject any non-canonical representation."""

    canonical = canonicalize_vdpp_calibration_diagnostics(
        calibration,
        shot_length=shot_length,
        native_shape=native_shape,
    )
    if type(calibration) is not dict:
        raise TypeError("VDPP calibration diagnostics must be a dictionary")
    for key in CALIBRATION_KEYS:
        if not _same_scalar_representation(calibration[key], canonical[key]):
            raise ValueError(f"VDPP calibration {key} is not in canonical representation")
    return canonical
