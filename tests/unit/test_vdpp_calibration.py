"""Core numerical contract tests for VDPP shot calibration."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.depth_surge_3d.core.vdpp_calibration import (
    MAX_POSTCLIP_MEAN_DRIFT,
    MAX_PRECLIP_OUT_OF_RANGE_FRACTION,
    MIDPOINT_CODE,
    MIN_CORRELATION,
    MIN_PAIR_COUNT,
    MIN_POSITIVE_SCALE,
    MODEL_MIDPOINT_VALUE,
    PHYSICAL_BOUND_BASE_ULPS,
    PHYSICAL_BOUND_ULPS_PER_PLANNED_TILE,
    STATS_TILE_PIXELS,
    VARIANCE_EPSILON,
    NumericalContractError,
    PairTileState,
    ScalarTileState,
    candidate_tile,
    canonical_derived_diagnostics,
    canonicalize_float,
    canonicalize_vdpp_calibration_diagnostics,
    collect_vdpp_numeric_runtime_probe,
    finalize_pair_moments,
    merge_pair_states,
    merge_scalar_states,
    normalize_bounded_stat,
    normalize_vdpp_input_frame,
    planned_tile_count,
    reduce_pair_tile,
    reduce_scalar_tile,
    validate_vdpp_calibration_diagnostics,
)
from src.depth_surge_3d.inference.depth.vdpp_contract import build_vdpp_execution_plan


def _ols_calibration() -> dict[str, object]:
    planned = planned_tile_count(1, (3, 5))
    derived = canonical_derived_diagnostics(
        midpoint_count=0,
        shot_pixels=15,
        source_mean=0.5,
        source_variance=0.04,
        raw_mean=0.5,
        raw_variance=0.04,
        covariance=0.04,
        candidate_mean=0.5,
        candidate_std=0.2,
        preclip_low_fraction=0.0,
        preclip_high_fraction=0.0,
        planned_tile_count=planned,
    )
    return {
        "mode": "ols",
        "pair_count": 15,
        "midpoint_count": 0,
        "midpoint_fraction": derived["midpoint_fraction"],
        "flat_frame_count": 0,
        "source_mean": 0.5,
        "source_variance": 0.04,
        "source_std": derived["source_std"],
        "raw_mean": 0.5,
        "raw_variance": 0.04,
        "raw_std": derived["raw_std"],
        "covariance": 0.04,
        "correlation": derived["correlation"],
        "scale": derived["scale"],
        "shift": derived["shift"],
        "candidate_mean": 0.5,
        "candidate_std": 0.2,
        "postclip_contrast_ratio": derived["postclip_contrast_ratio"],
        "postclip_mean_drift": derived["postclip_mean_drift"],
        "preclip_low_fraction": 0.0,
        "preclip_high_fraction": 0.0,
        "fallback_reason": None,
    }


def test_fixed_constants_and_execution_plan() -> None:
    assert MIDPOINT_CODE == 32768
    assert MODEL_MIDPOINT_VALUE == 0.5
    assert STATS_TILE_PIXELS == 262144
    assert MIN_PAIR_COUNT == 2
    assert VARIANCE_EPSILON == 1e-12
    assert PHYSICAL_BOUND_BASE_ULPS == 4
    assert PHYSICAL_BOUND_ULPS_PER_PLANNED_TILE == 64
    assert MIN_POSITIVE_SCALE == 1e-8
    assert MIN_CORRELATION == 0.5
    assert MAX_POSTCLIP_MEAN_DRIFT == 0.01
    assert MAX_PRECLIP_OUT_OF_RANGE_FRACTION == 0.01
    plan = build_vdpp_execution_plan((608, 1080))
    assert plan["stats_tile_pixels"] == STATS_TILE_PIXELS
    assert plan["physical_bound_ulps_per_planned_tile"] == 64


def test_numeric_runtime_probe_is_deterministic() -> None:
    probe = collect_vdpp_numeric_runtime_probe()
    assert probe["schema"] == "vdpp-numeric-reducer-probe-v1"
    assert probe == collect_vdpp_numeric_runtime_probe()


def test_canonicalize_and_bounded_stat_snapping() -> None:
    assert canonicalize_float(-0.0).hex() == (0.0).hex()
    within = math.nextafter(0.0, -math.inf)
    assert normalize_bounded_stat(within, 0.0, 1.0, planned_tile_count=2) == 0.0
    with pytest.raises(NumericalContractError):
        normalize_bounded_stat(-0.01, 0.0, 1.0, planned_tile_count=2)
    with pytest.raises(ValueError):
        normalize_bounded_stat(0.5, 2.0, 1.0, planned_tile_count=2)
    with pytest.raises(TypeError):
        normalize_bounded_stat(0.5, float("nan"), 1.0, planned_tile_count=2)


def test_frame_normalization_excludes_midpoint_and_handles_flat_frames() -> None:
    source = np.array([[1000, MIDPOINT_CODE, 1001], [1003, 1002, MIDPOINT_CODE]], dtype=np.uint16)
    target = np.empty_like(source, dtype=np.float32)
    assert normalize_vdpp_input_frame(source, target) is True
    expected = np.array([[0.0, 0.5, 1.0 / 3.0], [1.0, 2.0 / 3.0, 0.5]], dtype=np.float32)
    np.testing.assert_allclose(target, expected, atol=1e-6)

    flat = np.full((2, 3), 2000, dtype=np.uint16)
    assert normalize_vdpp_input_frame(flat, target) is False
    np.testing.assert_array_equal(target, np.full((2, 3), 0.5, dtype=np.float32))


def test_tile_reductions_chan_merges_and_moment_finalization() -> None:
    x = np.array([0.1, 0.2, 0.5, 0.8, 0.9, 1.0], dtype=np.float32)
    y = np.array([0.2, 0.3, 0.4, 0.7, 0.8, 0.9], dtype=np.float32)
    mask = np.array([True, True, True, True, True, True], dtype=np.bool_)

    s_a = reduce_scalar_tile(x[:3], mask[:3])
    s_b = reduce_scalar_tile(x[3:], mask[3:])
    s_all = merge_scalar_states(s_a, s_b)
    assert s_all.count == 6
    assert math.isclose(s_all.mean, float(np.mean(x.astype(np.float64))), rel_tol=1e-12)
    assert merge_scalar_states(ScalarTileState.empty(), s_a) == s_a

    p_a = reduce_pair_tile(x[:3], y[:3], mask[:3])
    p_b = reduce_pair_tile(x[3:], y[3:], mask[3:])
    p_all = merge_pair_states(p_a, p_b)
    assert p_all.count == 6
    assert merge_pair_states(PairTileState.empty(), p_a) == p_a

    moments = finalize_pair_moments(p_all, planned_tile_count=2)
    assert math.isclose(moments["raw_mean"], float(np.mean(x.astype(np.float64))), rel_tol=1e-12)
    assert math.isclose(moments["source_mean"], float(np.mean(y.astype(np.float64))), rel_tol=1e-12)
    assert moments["covariance"] > 0.0


def test_candidate_tile_applies_float64_affine_and_clips_to_float32() -> None:
    raw = np.array([-0.5, 0.0, 0.5, 1.5], dtype=np.float32)
    preclip, cand = candidate_tile(raw, 2.0, -0.25)
    assert preclip.dtype == np.float64
    assert cand.dtype == np.float32
    np.testing.assert_allclose(preclip, [-1.25, -0.25, 0.75, 2.75])
    np.testing.assert_allclose(cand, [0.0, 0.0, 0.75, 1.0])


@pytest.mark.parametrize(
    "changes",
    [
        {"mode": "guessed"},
        {"fallback_reason": "unknown"},
        {"fallback_reason": "source_variance"},
        {"mode": "base_fallback"},
        {"midpoint_fraction": None},
        {"midpoint_count": 1},
        {"mode": "all_midpoint"},
        {"correlation": 0.5},
        {"postclip_contrast_ratio": 0.0},
        {"source_variance": 0.0},
        {"pair_count": 1},
        {"preclip_high_fraction": 1.1},
    ],
)
def test_stored_ols_record_rejects_inconsistent_mutations(changes: dict[str, object]) -> None:
    record = {**_ols_calibration(), **changes}
    with pytest.raises((TypeError, ValueError)):
        canonicalize_vdpp_calibration_diagnostics(record, shot_length=1, native_shape=(3, 5))


def test_validate_vdpp_calibration_accepts_valid_ols_and_rejects_negative_zero() -> None:
    valid = _ols_calibration()
    validated = validate_vdpp_calibration_diagnostics(valid, shot_length=1, native_shape=(3, 5))
    assert validated["mode"] == "ols"

    non_canonical = {**valid, "preclip_low_fraction": -0.0}
    with pytest.raises(ValueError, match="canonical representation"):
        validate_vdpp_calibration_diagnostics(non_canonical, shot_length=1, native_shape=(3, 5))
