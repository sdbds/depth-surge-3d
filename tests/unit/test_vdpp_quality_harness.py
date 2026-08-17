"""Deterministic quality-gate tests for experimental VDPP rollout."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_vdpp_quality import (
    METRIC_ALGORITHM,
    aggregate_runs,
    compute_metrics,
    evaluate_manifest,
)


def test_metrics_pin_paper_temporal_gradient_and_standard_spatial_errors() -> None:
    ground_truth = np.array([[[1.0, 2.0]], [[2.0, 4.0]]], dtype=np.float64)
    prediction = np.array([[[1.0, 2.0]], [[2.0, 3.0]]], dtype=np.float64)

    metrics = compute_metrics(prediction, ground_truth)

    assert metrics["tgse"] == pytest.approx(0.5)
    assert metrics["abs_rel"] == pytest.approx(0.0625)
    assert metrics["delta1"] == pytest.approx(0.75)


def test_aggregate_uses_repeat_then_sequence_medians_and_unrounded_gate() -> None:
    runs = []
    for sequence, base_tgse, vdpp_tgse in (
        ("a", 10.0, 9.8),
        ("b", 20.0, 19.7),
    ):
        for seed, adjustment in zip((0, 1, 2), (-0.1, 0.0, 0.1), strict=True):
            runs.append(
                {
                    "backend": "da3mono",
                    "sequence": sequence,
                    "seed": seed,
                    "baseline": {
                        "tgse": base_tgse + adjustment,
                        "abs_rel": 0.10,
                        "delta1": 0.90,
                    },
                    "vdpp": {
                        "tgse": vdpp_tgse + adjustment,
                        "abs_rel": 0.102,
                        "delta1": 0.88,
                    },
                }
            )

    aggregate = aggregate_runs(runs, expected_seeds=(0, 1, 2))

    assert aggregate["da3mono"]["baseline"]["tgse"] == 15.0
    assert aggregate["da3mono"]["vdpp"]["tgse"] == 14.75
    assert aggregate["da3mono"]["gate"] == {
        "tgse": True,
        "abs_rel": True,
        "delta1": True,
        "passed": True,
    }


def test_manifest_evaluation_preserves_identities_and_resolved_digest(tmp_path: Path) -> None:
    ground_truth = np.array([[[1.0]], [[2.0]]], dtype=np.float64)
    baseline = np.array([[[1.0]], [[1.0]]], dtype=np.float64)
    candidate = np.array([[[1.0]], [[1.02]]], dtype=np.float64)
    runs = []
    for seed in (0, 1, 2):
        payload = tmp_path / f"run-{seed}.npz"
        np.savez(
            payload,
            baseline=baseline,
            vdpp=candidate,
            ground_truth=ground_truth,
        )
        runs.append(
            {
                "backend": "v2-control",
                "sequence": "clip-a",
                "seed": seed,
                "payload": payload.name,
                "payload_sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "metric_algorithm": METRIC_ALGORITHM,
        "seeds": [0, 1, 2],
        "source_identity": {"dataset": "synthetic-test", "sha256": "a" * 64},
        "checkpoint_identity": {"sha256": "b" * 64},
        "runtime_identity": {"torch": "test", "hardware": "cpu"},
        "base_settings": {"temporal_postprocessor": "off"},
        "candidate_settings": {"temporal_postprocessor": "vdpp"},
        "runs": runs,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = evaluate_manifest(manifest_path)

    assert report["metric_algorithm"] == METRIC_ALGORITHM
    assert report["manifest_sha256"]
    assert report["identities"]["checkpoint_identity"] == manifest["checkpoint_identity"]
    assert [run["seed"] for run in report["runs"]] == [0, 1, 2]


def test_manifest_rejects_missing_fixed_seed_repeat(tmp_path: Path) -> None:
    manifest = {
        "schema_version": 1,
        "metric_algorithm": METRIC_ALGORITHM,
        "seeds": [0, 1, 2],
        "source_identity": {},
        "checkpoint_identity": {},
        "runtime_identity": {},
        "base_settings": {},
        "candidate_settings": {},
        "runs": [],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="run matrix"):
        evaluate_manifest(path)
