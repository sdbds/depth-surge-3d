"""Deterministic numerical contract for VDPP shot calibration."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

import src.depth_surge_3d.core.vdpp_calibration as vdpp_calibration_module
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


def _empty_calibration(
    *,
    mode: str = "all_midpoint",
    reason: str | None = None,
    pair_count: int = 0,
    midpoint_count: int = 15,
    flat_frame_count: int = 1,
) -> dict[str, object]:
    return {
        "mode": mode,
        "pair_count": pair_count,
        "midpoint_count": midpoint_count,
        "midpoint_fraction": float(midpoint_count / 15),
        "flat_frame_count": flat_frame_count,
        "source_mean": None,
        "source_variance": None,
        "source_std": None,
        "raw_mean": None,
        "raw_variance": None,
        "raw_std": None,
        "covariance": None,
        "correlation": None,
        "scale": None,
        "shift": None,
        "candidate_mean": None,
        "candidate_std": None,
        "postclip_contrast_ratio": None,
        "postclip_mean_drift": None,
        "preclip_low_fraction": None,
        "preclip_high_fraction": None,
        "fallback_reason": reason,
    }


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


def _moment_calibration(
    *,
    source_mean: float = 0.5,
    source_variance: float = 0.25,
    raw_mean: float = 0.5,
    raw_variance: float = 0.25,
    covariance: float = 0.25,
    candidate_mean: float = 0.5,
    candidate_std: float = 0.5,
    preclip_low_fraction: float = 0.0,
    preclip_high_fraction: float = 0.0,
    pair_count: int = 15,
    mode: str = "ols",
    reason: str | None = None,
) -> dict[str, object]:
    quality_required = mode == "ols" or reason in {
        "contrast",
        "mean_drift",
        "preclip_out_of_range",
    }
    fit_required = mode == "ols" or reason in {
        "scale_below_minimum",
        "correlation",
        "contrast",
        "mean_drift",
        "preclip_out_of_range",
    }
    derived = canonical_derived_diagnostics(
        midpoint_count=0,
        shot_pixels=15,
        source_mean=source_mean,
        source_variance=source_variance,
        raw_mean=raw_mean,
        raw_variance=raw_variance,
        covariance=covariance,
        candidate_mean=candidate_mean if quality_required else None,
        candidate_std=candidate_std if quality_required else None,
        preclip_low_fraction=preclip_low_fraction if quality_required else None,
        preclip_high_fraction=preclip_high_fraction if quality_required else None,
        planned_tile_count=1,
    )
    return {
        "mode": mode,
        "pair_count": pair_count,
        "midpoint_count": 0,
        "midpoint_fraction": derived["midpoint_fraction"],
        "flat_frame_count": 0,
        "source_mean": source_mean,
        "source_variance": source_variance,
        "source_std": derived["source_std"],
        "raw_mean": raw_mean,
        "raw_variance": raw_variance,
        "raw_std": derived["raw_std"],
        "covariance": covariance,
        "correlation": derived["correlation"] if fit_required else None,
        "scale": derived["scale"] if fit_required else None,
        "shift": derived["shift"] if fit_required else None,
        "candidate_mean": candidate_mean if quality_required else None,
        "candidate_std": candidate_std if quality_required else None,
        "postclip_contrast_ratio": (
            derived["postclip_contrast_ratio"] if quality_required else None
        ),
        "postclip_mean_drift": derived["postclip_mean_drift"] if quality_required else None,
        "preclip_low_fraction": preclip_low_fraction if quality_required else None,
        "preclip_high_fraction": preclip_high_fraction if quality_required else None,
        "fallback_reason": reason,
    }


def test_fixed_constants_match_the_v2_execution_contract() -> None:
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


def test_numeric_runtime_probe_fingerprints_the_pinned_reducer_and_chan_paths() -> None:
    collect_probe = getattr(
        vdpp_calibration_module,
        "collect_vdpp_numeric_runtime_probe",
        None,
    )
    assert callable(collect_probe), "VDPP numeric runtime probe is missing"

    assert collect_probe() == {
        "schema": "vdpp-numeric-reducer-probe-v1",
        "scalar_mean_hex": "0x1.1c77032d2d2d3p-1",
        "scalar_m2_hex": "0x1.9b2e1ef3e8a83p+0",
        "pair_mean_x_hex": "0x1.181be15555555p-1",
        "pair_mean_y_hex": "0x1.2b9e515555555p-2",
        "pair_m2_x_hex": "0x1.06e24c34790abp-4",
        "pair_m2_y_hex": "0x1.b8fa5e68e8caap-4",
        "pair_c_xy_hex": "-0x1.d88cb98467556p-5",
        "chan_mean_hex": "0x1.71c71c71c71c8p-2",
        "chan_variance_hex": "0x1.4a4587e6b74f1p-3",
        "chan_pair_mean_x_hex": "0x1.71c71c71c71c8p-2",
        "chan_pair_mean_y_hex": "0x1.2aaaaaaaaaaaap-1",
        "chan_pair_m2_x_hex": "0x1.738e38e38e38fp+0",
        "chan_pair_m2_y_hex": "0x1.0800000000000p+1",
        "chan_pair_c_xy_hex": "-0x1.7111111111112p-1",
    }


def test_v2_execution_plan_persists_every_calibration_policy() -> None:
    plan = build_vdpp_execution_plan((608, 1080))

    assert plan["input_normalization"] == "per-frame-minmax-excluding-midpoint-code-v2"
    assert plan["input_arithmetic"] == "u16-extrema-float32-subtract-divide-v1"
    assert plan["tile_iteration"] == "c-row-major-fixed-262144-v1"
    assert plan["tile_reducer"] == ("float64-two-pass-numpy-sum-centered-c-order-skip-empty-v1")
    assert plan["chan_merge"] == "python-f64-sequential-nonempty-canonical-zero-v2"
    assert plan["midpoint_code_policy"] == "preserve-u16-32768-heuristic-v2"
    assert plan["derived_diagnostics_policy"] == ("recompute-from-canonical-persisted-moments-v1")
    assert plan["calibration_diagnostics_schema"] == ("strict-exact-keys-derived-tile-budget-v5")
    assert plan["partial_resume_numeric_runtime_policy"] == (
        "interpreter-platform-versions-reducer-probe-v1"
    )
    assert plan["opencv_runtime_policy"] == ("version-bound-decoded-u16-semantics-v1")
    assert plan["fallback_reason_order"] == [
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
    ]
    assert plan["stats_tile_pixels"] == STATS_TILE_PIXELS
    assert plan["physical_bound_ulps_per_planned_tile"] == 64


def test_frame_normalization_is_float32_and_preserves_midpoint_model_value() -> None:
    source = np.array(
        [[1000, MIDPOINT_CODE, 1001], [1003, 1002, MIDPOINT_CODE]],
        dtype=np.uint16,
    )
    target = np.empty(source.shape, dtype=np.float32)

    ranged = normalize_vdpp_input_frame(source, target)

    expected = source.astype(np.float32)
    expected -= np.float32(1000)
    expected /= np.float32(3)
    expected[source == MIDPOINT_CODE] = np.float32(MODEL_MIDPOINT_VALUE)
    assert ranged is True
    assert target.dtype == np.float32
    assert np.array_equal(target, expected)


def test_frame_normalization_reuses_its_only_full_frame_mask_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = np.array(
        [[1000, MIDPOINT_CODE, 1001], [1003, 1002, MIDPOINT_CODE]],
        dtype=np.uint16,
    )
    target = np.empty(source.shape, dtype=np.float32)
    original_logical_not = np.logical_not
    inverted_mask_ids: list[int] = []

    def track_logical_not(values, *args, **kwargs):
        assert kwargs.get("out") is values
        inverted_mask_ids.append(id(values))
        return original_logical_not(values, *args, **kwargs)

    monkeypatch.setattr(vdpp_calibration_module.np, "logical_not", track_logical_not)

    assert normalize_vdpp_input_frame(source, target) is True
    assert len(inverted_mask_ids) == 1


@pytest.mark.parametrize(
    "source",
    [
        np.full((2, 3), MIDPOINT_CODE, dtype=np.uint16),
        np.array([[7, 7, MIDPOINT_CODE]], dtype=np.uint16),
    ],
)
def test_flat_frame_normalization_fills_model_midpoint(source: np.ndarray) -> None:
    target = np.empty(source.shape, dtype=np.float32)
    assert normalize_vdpp_input_frame(source, target) is False
    assert np.array_equal(target, np.full(source.shape, 0.5, dtype=np.float32))


def test_candidate_tile_uses_float64_affine_then_float32_clip() -> None:
    raw = np.array([-1.0, 0.25, 1.0, 3.0], dtype=np.float32)
    preclip, candidate = candidate_tile(raw, 0.5, 0.125)
    assert preclip.dtype == np.float64
    assert candidate.dtype == np.float32
    assert np.array_equal(preclip, np.array([-0.375, 0.25, 0.625, 1.625]))
    assert np.array_equal(candidate, np.array([0.0, 0.25, 0.625, 1.0], dtype=np.float32))


def test_tile_reducers_skip_empty_and_handle_one_value_without_nan() -> None:
    values = np.array([0.25, 0.75], dtype=np.float32)
    empty = np.array([False, False])
    one = np.array([False, True])

    assert reduce_scalar_tile(values, empty) == ScalarTileState.empty()
    assert reduce_pair_tile(values, values, empty) == PairTileState.empty()
    assert reduce_scalar_tile(values, one) == ScalarTileState(1, 0.75, 0.0)
    assert reduce_pair_tile(values, values, one) == PairTileState(1, 0.75, 0.75, 0.0, 0.0, 0.0)


def test_pair_and_scalar_reducers_use_the_pinned_two_pass_numpy_sequence() -> None:
    values = np.array(
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
    eligible = np.ones(values.shape, dtype=bool)

    scalar = reduce_scalar_tile(values, eligible)
    pair = reduce_pair_tile(values, values, eligible)

    assert scalar.m2.hex() == "0x1.9b2e1ef3e8a83p+0"
    assert pair.m2_x.hex() == "0x1.9b2e1ef3e8a83p+0"
    assert pair.m2_y.hex() == "0x1.9b2e1ef3e8a83p+0"
    assert pair.c_xy.hex() == "0x1.9b2e1ef3e8a83p+0"


def test_sequential_chan_merge_matches_the_pinned_parenthesization() -> None:
    left = PairTileState(2, 0.25, 0.5, 0.125, 0.5, 0.25)
    right = PairTileState(3, 0.75, 0.25, 0.375, 0.125, -0.125)
    merged = merge_pair_states(left, right)
    scalar = merge_scalar_states(
        ScalarTileState(left.count, left.mean_x, left.m2_x),
        ScalarTileState(right.count, right.mean_x, right.m2_x),
    )

    assert merged.count == 5
    assert merged.mean_x.hex() == float(0.25 + (((0.75 - 0.25) * 3) / 5)).hex()
    assert merged.m2_x.hex() == scalar.m2.hex()


def test_canonicalize_float_forces_positive_zero_and_rejects_nonfinite_or_nonfloat() -> None:
    assert canonicalize_float(-0.0).hex() == "0x0.0p+0"
    for value in [0, True, np.float64(0.0), math.inf, math.nan]:
        with pytest.raises((TypeError, ValueError)):
            canonicalize_float(value)  # type: ignore[arg-type]


def test_planned_tile_bound_snaps_at_budget_and_rejects_first_value_beyond() -> None:
    tiles = 7
    budget = PHYSICAL_BOUND_BASE_ULPS + PHYSICAL_BOUND_ULPS_PER_PLANNED_TILE * tiles
    boundary = 1.0 + budget * math.ulp(1.0)
    beyond = math.nextafter(boundary, math.inf)

    inside = math.nextafter(1.0, 0.0)
    assert normalize_bounded_stat(inside, 0.0, 1.0, planned_tile_count=tiles) == inside
    assert normalize_bounded_stat(1.0, 0.0, 1.0, planned_tile_count=tiles) == 1.0
    assert (
        normalize_bounded_stat(
            math.nextafter(1.0, math.inf),
            0.0,
            1.0,
            planned_tile_count=tiles,
        )
        == 1.0
    )
    assert normalize_bounded_stat(boundary, 0.0, 1.0, planned_tile_count=tiles) == 1.0
    with pytest.raises(NumericalContractError):
        normalize_bounded_stat(beyond, 0.0, 1.0, planned_tile_count=tiles)


def test_nineteen_zero_one_std_rounding_snaps_to_the_physical_boundary() -> None:
    values = np.array([0.0] * 19 + [1.0] * 19, dtype=np.float32)
    state = ScalarTileState.empty()
    for start in range(0, values.size, 3):
        tile = values[start : start + 3]
        state = merge_scalar_states(
            state,
            reduce_scalar_tile(tile, np.ones(tile.shape, dtype=bool)),
        )

    raw_std = math.sqrt(state.m2 / state.count)
    assert raw_std.hex() == "0x1.0000000000001p-1"
    assert (
        normalize_bounded_stat(
            raw_std,
            0.0,
            0.5,
            planned_tile_count=13,
        ).hex()
        == float(0.5).hex()
    )


def test_long_sequential_boundary_variance_uses_the_dynamic_tile_budget() -> None:
    rng = random.Random(70)
    tiles: list[ScalarTileState] = []
    for _ in range(2048):
        count = rng.randrange(2, STATS_TILE_PIXELS + 1)
        ones = rng.randrange(1, count)
        for selected_ones in (ones, count - ones):
            mean = selected_ones / count
            m2 = (count - selected_ones) * (mean * mean) + selected_ones * (
                (1.0 - mean) * (1.0 - mean)
            )
            tiles.append(ScalarTileState(count, mean, m2))
    rng.shuffle(tiles)
    state = ScalarTileState.empty()
    for tile in tiles:
        state = merge_scalar_states(state, tile)

    raw_variance = state.m2 / state.count
    assert raw_variance.hex() == "0x1.0000000000011p-2"
    assert (
        normalize_bounded_stat(
            raw_variance,
            0.0,
            0.25,
            planned_tile_count=4096,
        ).hex()
        == float(0.25).hex()
    )


def test_correlation_is_derived_only_from_canonical_persisted_moments() -> None:
    x = np.array(
        [0.6486990451812744, 0.6523162722587585, 0.3402478098869324],
        dtype=np.float32,
    )
    y = np.array(
        [0.3692907691001892, 0.03194546699523926, 0.4765521287918091],
        dtype=np.float32,
    )
    state = reduce_pair_tile(x, y, np.ones(3, dtype=bool))
    moments = finalize_pair_moments(state, planned_tile_count=1)
    derived = canonical_derived_diagnostics(
        midpoint_count=0,
        shot_pixels=3,
        **moments,
        candidate_mean=None,
        candidate_std=None,
        preclip_low_fraction=None,
        preclip_high_fraction=None,
        planned_tile_count=1,
    )

    assert derived["correlation"].hex() == "-0x1.634d29d1cfb6bp-1"
    assert derived["correlation"].hex() != "-0x1.634d29d1cfb6ap-1"

    calibration = {
        "mode": "base_fallback",
        "pair_count": 3,
        "midpoint_count": 0,
        "midpoint_fraction": 0.0,
        "flat_frame_count": 0,
        "source_mean": moments["source_mean"],
        "source_variance": moments["source_variance"],
        "source_std": derived["source_std"],
        "raw_mean": moments["raw_mean"],
        "raw_variance": moments["raw_variance"],
        "raw_std": derived["raw_std"],
        "covariance": moments["covariance"],
        "correlation": derived["correlation"],
        "scale": derived["scale"],
        "shift": derived["shift"],
        "candidate_mean": None,
        "candidate_std": None,
        "postclip_contrast_ratio": None,
        "postclip_mean_drift": None,
        "preclip_low_fraction": None,
        "preclip_high_fraction": None,
        "fallback_reason": "scale_below_minimum",
    }
    validate_vdpp_calibration_diagnostics(
        calibration,
        shot_length=1,
        native_shape=(1, 3),
    )
    calibration["correlation"] = float.fromhex("-0x1.634d29d1cfb6ap-1")
    with pytest.raises(ValueError, match="correlation"):
        validate_vdpp_calibration_diagnostics(
            calibration,
            shot_length=1,
            native_shape=(1, 3),
        )


def test_strict_diagnostics_accepts_a_canonical_ols_record() -> None:
    calibration = _ols_calibration()
    canonical = canonicalize_vdpp_calibration_diagnostics(
        calibration,
        shot_length=1,
        native_shape=(3, 5),
    )
    assert (
        validate_vdpp_calibration_diagnostics(
            canonical,
            shot_length=1,
            native_shape=(3, 5),
        )
        == canonical
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("correlation", 0.9),
        ("scale", 2.0),
        ("shift", -0.5),
        ("postclip_contrast_ratio", 0.9),
        ("postclip_mean_drift", 0.001),
    ],
)
def test_strict_diagnostics_rejects_inconsistent_derived_fields(
    field: str,
    value: float,
) -> None:
    calibration = _ols_calibration()
    calibration[field] = value
    with pytest.raises(ValueError, match=field):
        canonicalize_vdpp_calibration_diagnostics(
            calibration,
            shot_length=1,
            native_shape=(3, 5),
        )


def test_reader_rejects_negative_zero_even_when_numeric_equality_would_pass() -> None:
    calibration = _ols_calibration()
    calibration["shift"] = -0.0
    with pytest.raises(ValueError, match="canonical"):
        validate_vdpp_calibration_diagnostics(
            calibration,
            shot_length=1,
            native_shape=(3, 5),
        )


def test_writer_canonicalizes_negative_zero_in_bounded_unbounded_and_derived_fields() -> None:
    calibration = _moment_calibration(raw_mean=0.0)
    calibration["midpoint_fraction"] = -0.0
    calibration["raw_mean"] = -0.0
    calibration["preclip_low_fraction"] = -0.0
    calibration["postclip_mean_drift"] = -0.0

    canonical = canonicalize_vdpp_calibration_diagnostics(
        calibration,
        shot_length=1,
        native_shape=(3, 5),
    )

    for field in (
        "midpoint_fraction",
        "raw_mean",
        "preclip_low_fraction",
        "postclip_mean_drift",
    ):
        assert canonical[field].hex() == "0x0.0p+0"


@pytest.mark.parametrize(
    ("variance_field", "value", "reason"),
    [
        ("source_variance", math.nextafter(VARIANCE_EPSILON, 0.0), "source_variance"),
        ("source_variance", VARIANCE_EPSILON, "source_variance"),
        ("source_variance", math.nextafter(VARIANCE_EPSILON, math.inf), None),
        ("raw_variance", math.nextafter(VARIANCE_EPSILON, 0.0), "raw_variance"),
        ("raw_variance", VARIANCE_EPSILON, "raw_variance"),
        ("raw_variance", math.nextafter(VARIANCE_EPSILON, math.inf), None),
    ],
)
def test_variance_gates_use_the_exact_persisted_variance(
    variance_field: str,
    value: float,
    reason: str | None,
) -> None:
    source_variance = value if variance_field == "source_variance" else 0.25
    raw_variance = value if variance_field == "raw_variance" else 0.25
    covariance = math.sqrt(source_variance * raw_variance)
    mode = "ols" if reason is None else "base_fallback"
    calibration = _moment_calibration(
        source_variance=source_variance,
        raw_variance=raw_variance,
        covariance=covariance,
        candidate_std=math.sqrt(source_variance),
        mode=mode,
        reason=reason,
    )

    canonical = canonicalize_vdpp_calibration_diagnostics(
        calibration,
        shot_length=1,
        native_shape=(3, 5),
    )

    assert canonical[variance_field].hex() == value.hex()
    assert canonical["fallback_reason"] == reason


@pytest.mark.parametrize(
    ("gate", "value", "reason"),
    [
        ("scale", math.nextafter(MIN_POSITIVE_SCALE, 0.0), "scale_below_minimum"),
        ("scale", MIN_POSITIVE_SCALE, None),
        ("scale", math.nextafter(MIN_POSITIVE_SCALE, math.inf), None),
        ("scale", -1.0, "scale_below_minimum"),
        ("correlation", math.nextafter(MIN_CORRELATION, 0.0), "correlation"),
        ("correlation", MIN_CORRELATION, None),
        ("correlation", math.nextafter(MIN_CORRELATION, math.inf), None),
        ("contrast", math.nextafter(0.5, 0.0), "contrast"),
        ("contrast", 0.5, None),
        ("contrast", math.nextafter(0.5, math.inf), None),
        ("mean_drift", math.nextafter(MAX_POSTCLIP_MEAN_DRIFT, 0.0), None),
        ("mean_drift", MAX_POSTCLIP_MEAN_DRIFT, None),
        (
            "mean_drift",
            math.nextafter(MAX_POSTCLIP_MEAN_DRIFT, math.inf),
            "mean_drift",
        ),
        (
            "preclip",
            math.nextafter(MAX_PRECLIP_OUT_OF_RANGE_FRACTION, 0.0),
            None,
        ),
        ("preclip", MAX_PRECLIP_OUT_OF_RANGE_FRACTION, None),
        (
            "preclip",
            math.nextafter(MAX_PRECLIP_OUT_OF_RANGE_FRACTION, math.inf),
            "preclip_out_of_range",
        ),
    ],
)
def test_every_finite_gate_accepts_its_documented_boundary(
    gate: str,
    value: float,
    reason: str | None,
) -> None:
    kwargs: dict[str, float] = {}
    if gate == "scale":
        raw_variance = 2.5e15 if value >= 0.0 else 0.25
        kwargs.update(raw_variance=raw_variance, covariance=value * raw_variance)
    elif gate == "correlation":
        kwargs.update(covariance=value * 0.25)
    elif gate == "contrast":
        kwargs.update(candidate_std=value * 0.5)
    elif gate == "mean_drift":
        kwargs.update(source_mean=0.0, raw_mean=0.0, candidate_mean=value)
    else:
        kwargs.update(preclip_low_fraction=value)
    mode = "ols" if reason is None else "base_fallback"
    calibration = _moment_calibration(mode=mode, reason=reason, **kwargs)

    canonical = canonicalize_vdpp_calibration_diagnostics(
        calibration,
        shot_length=1,
        native_shape=(3, 5),
    )

    assert canonical["fallback_reason"] == reason


def test_pair_count_two_is_accepted_while_one_uses_too_few_pairs() -> None:
    too_few = _empty_calibration(
        mode="base_fallback",
        reason="too_few_pairs",
        pair_count=1,
        midpoint_count=0,
        flat_frame_count=0,
    )
    assert (
        canonicalize_vdpp_calibration_diagnostics(
            too_few,
            shot_length=1,
            native_shape=(3, 5),
        )["fallback_reason"]
        == "too_few_pairs"
    )

    accepted = _moment_calibration(pair_count=2)
    assert (
        canonicalize_vdpp_calibration_diagnostics(
            accepted,
            shot_length=1,
            native_shape=(3, 5),
        )["mode"]
        == "ols"
    )


def test_diagnostics_reject_missing_unknown_and_non_builtin_numeric_fields() -> None:
    for mutation in ("missing", "unknown", "numpy"):
        calibration = _ols_calibration()
        if mutation == "missing":
            calibration.pop("raw_variance")
        elif mutation == "unknown":
            calibration["extra"] = 1.0
        else:
            calibration["raw_mean"] = np.float64(0.5)
        with pytest.raises((TypeError, ValueError)):
            validate_vdpp_calibration_diagnostics(
                calibration,
                shot_length=1,
                native_shape=(3, 5),
            )


def test_all_midpoint_and_source_no_range_require_exact_copy_semantics() -> None:
    all_midpoint = _empty_calibration()
    assert (
        canonicalize_vdpp_calibration_diagnostics(
            all_midpoint,
            shot_length=1,
            native_shape=(3, 5),
        )["mode"]
        == "all_midpoint"
    )

    source_no_range = _empty_calibration(
        mode="base_fallback",
        reason="source_no_range",
        midpoint_count=5,
    )
    assert (
        canonicalize_vdpp_calibration_diagnostics(
            source_no_range,
            shot_length=1,
            native_shape=(3, 5),
        )["fallback_reason"]
        == "source_no_range"
    )


def test_nonfinite_fit_persists_only_the_finite_correlation() -> None:
    calibration = _ols_calibration()
    calibration.update(
        {
            "mode": "base_fallback",
            "fallback_reason": "nonfinite_fit",
            "raw_mean": 1.0e308,
            "raw_variance": 0.01,
            "raw_std": 0.1,
            "covariance": 0.02,
            "correlation": 1.0,
            "scale": None,
            "shift": None,
            "candidate_mean": None,
            "candidate_std": None,
            "postclip_contrast_ratio": None,
            "postclip_mean_drift": None,
            "preclip_low_fraction": None,
            "preclip_high_fraction": None,
        }
    )

    canonical = canonicalize_vdpp_calibration_diagnostics(
        calibration,
        shot_length=1,
        native_shape=(3, 5),
    )

    assert canonical["correlation"] == 1.0
    assert canonical["scale"] is None
    assert canonical["shift"] is None


def test_later_fallback_reason_cannot_bypass_variance_priority() -> None:
    calibration = _ols_calibration()
    source_variance = VARIANCE_EPSILON
    raw_variance = 0.04
    covariance = math.sqrt(source_variance * raw_variance)
    derived = canonical_derived_diagnostics(
        midpoint_count=0,
        shot_pixels=15,
        source_mean=0.5,
        source_variance=source_variance,
        raw_mean=0.5,
        raw_variance=raw_variance,
        covariance=covariance,
        candidate_mean=None,
        candidate_std=None,
        preclip_low_fraction=None,
        preclip_high_fraction=None,
        planned_tile_count=1,
    )
    calibration.update(
        {
            "mode": "base_fallback",
            "fallback_reason": "correlation",
            "source_variance": source_variance,
            "source_std": derived["source_std"],
            "raw_variance": raw_variance,
            "raw_std": derived["raw_std"],
            "covariance": covariance,
            "correlation": derived["correlation"],
            "scale": derived["scale"],
            "shift": derived["shift"],
            "candidate_mean": None,
            "candidate_std": None,
            "postclip_contrast_ratio": None,
            "postclip_mean_drift": None,
            "preclip_low_fraction": None,
            "preclip_high_fraction": None,
        }
    )

    with pytest.raises(ValueError, match="first failed gate"):
        canonicalize_vdpp_calibration_diagnostics(
            calibration,
            shot_length=1,
            native_shape=(3, 5),
        )


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("scale", math.nextafter(MIN_POSITIVE_SCALE, 0.0), "scale_below_minimum"),
        ("correlation", math.nextafter(MIN_CORRELATION, 0.0), "correlation"),
        (
            "postclip_mean_drift",
            math.nextafter(MAX_POSTCLIP_MEAN_DRIFT, math.inf),
            "mean_drift",
        ),
    ],
)
def test_fallback_reason_must_match_the_first_failed_gate(
    field: str,
    value: float,
    reason: str,
) -> None:
    calibration = _ols_calibration()
    calibration["mode"] = "base_fallback"
    calibration["fallback_reason"] = reason
    if field == "scale":
        calibration["raw_variance"] = 4.0e14
        calibration["covariance"] = value * 4.0e14
    elif field == "correlation":
        calibration["covariance"] = value * math.sqrt(0.04 * 0.04)
    elif field == "postclip_mean_drift":
        calibration["candidate_mean"] = 0.5 + value
    quality_required = reason == "mean_drift"
    if not quality_required:
        for name in (
            "candidate_mean",
            "candidate_std",
            "postclip_contrast_ratio",
            "postclip_mean_drift",
            "preclip_low_fraction",
            "preclip_high_fraction",
        ):
            calibration[name] = None
    derived = canonical_derived_diagnostics(
        midpoint_count=0,
        shot_pixels=15,
        source_mean=calibration["source_mean"],
        source_variance=calibration["source_variance"],
        raw_mean=calibration["raw_mean"],
        raw_variance=calibration["raw_variance"],
        covariance=calibration["covariance"],
        candidate_mean=calibration["candidate_mean"] if quality_required else None,
        candidate_std=calibration["candidate_std"] if quality_required else None,
        preclip_low_fraction=(calibration["preclip_low_fraction"] if quality_required else None),
        preclip_high_fraction=(calibration["preclip_high_fraction"] if quality_required else None),
        planned_tile_count=1,
    )
    for name in (
        "source_std",
        "raw_std",
        "correlation",
        "scale",
        "shift",
    ):
        calibration[name] = derived[name]
    if quality_required:
        calibration["postclip_contrast_ratio"] = derived["postclip_contrast_ratio"]
        calibration["postclip_mean_drift"] = derived["postclip_mean_drift"]
    canonical = canonicalize_vdpp_calibration_diagnostics(
        calibration,
        shot_length=1,
        native_shape=(3, 5),
    )
    assert canonical["fallback_reason"] == reason


def test_planned_tile_count_uses_integer_ceil_and_rejects_bools() -> None:
    assert planned_tile_count(3, (1, STATS_TILE_PIXELS + 1)) == 6
    with pytest.raises(ValueError):
        planned_tile_count(True, (3, 5))
