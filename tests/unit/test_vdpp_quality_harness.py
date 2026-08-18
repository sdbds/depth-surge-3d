"""Deterministic quality-gate tests for experimental VDPP rollout."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.evaluate_vdpp_quality import (
    METRIC_ALGORITHM,
    aggregate_runs,
    compute_metrics,
    evaluate_manifest,
)
from scripts.verify_vdpp_calibration import (
    REPORTED_FIXTURE,
    compute_final_png_metrics,
    verify_final_png_quality,
    verify_reported_fixture,
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


def test_manifest_evaluation_preserves_identities_and_resolved_digest(
    tmp_path: Path,
) -> None:
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


def _write_final_png_fixture(tmp_path: Path) -> tuple[list[Path], list[Path]]:
    base_root = tmp_path / "base"
    stable_root = tmp_path / "stable"
    base_root.mkdir()
    stable_root.mkdir()
    sources = [
        np.array(
            [[10000, 20000, 30000, 32768], [40000, 50000, 60000, 15000]],
            dtype=np.uint16,
        ),
        np.array(
            [[12000, 22000, 32000, 32768], [42000, 52000, 62000, 17000]],
            dtype=np.uint16,
        ),
    ]
    eligible_values = np.concatenate(
        [frame[frame != np.uint16(32768)].astype(np.float64) for frame in sources]
    )
    center = float(np.mean(eligible_values))
    base_files: list[Path] = []
    stable_files: list[Path] = []
    for index, source in enumerate(sources):
        output = np.rint(center + 0.9 * (source.astype(np.float64) - center)).astype(np.uint16)
        output[source == np.uint16(32768)] = np.uint16(32768)
        base_path = base_root / f"frame_{index + 1:06d}.png"
        stable_path = stable_root / f"frame_{index + 1:06d}.png"
        assert cv2.imwrite(str(base_path), source)
        assert cv2.imwrite(str(stable_path), output)
        base_files.append(base_path)
        stable_files.append(stable_path)
    return base_files, stable_files


def test_final_png_verifier_recomputes_quality_and_source_diagnostics(
    tmp_path: Path,
) -> None:
    base_files, stable_files = _write_final_png_fixture(tmp_path)
    metrics = compute_final_png_metrics(
        base_files,
        stable_files,
        native_shape=(2, 4),
    )
    calibration = {
        "pair_count": metrics["actual_pair_count"],
        "midpoint_count": metrics["actual_midpoint_count"],
        "midpoint_fraction": metrics["actual_midpoint_fraction"],
        "flat_frame_count": metrics["actual_flat_frame_count"],
        "source_mean": metrics["actual_source_mean"],
        "source_variance": metrics["actual_source_variance"],
        "source_std": metrics["actual_source_std"],
        "candidate_mean": metrics["actual_output_mean"],
        "candidate_std": metrics["actual_output_std"],
    }

    verified = verify_final_png_quality(
        base_files,
        stable_files,
        calibration,
        native_shape=(2, 4),
    )

    assert verified["actual_output_to_source_correlation"] > 0.99
    assert 0.89 < verified["actual_output_contrast_ratio"] < 0.91
    assert verified["pixel_identical_to_base"] is False


def test_final_png_verifier_detects_midpoint_change_and_saturated_output(
    tmp_path: Path,
) -> None:
    base_files, stable_files = _write_final_png_fixture(tmp_path)
    changed = cv2.imread(str(stable_files[0]), cv2.IMREAD_UNCHANGED)
    changed[0, 3] = np.uint16(65535)
    assert cv2.imwrite(str(stable_files[0]), changed)
    with pytest.raises(ValueError, match="midpoint"):
        compute_final_png_metrics(base_files, stable_files, native_shape=(2, 4))

    for source_path, output_path in zip(base_files, stable_files, strict=True):
        source = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
        saturated = np.full(source.shape, 65535, dtype=np.uint16)
        saturated[source == np.uint16(32768)] = np.uint16(32768)
        assert cv2.imwrite(str(output_path), saturated)
    with pytest.raises(ValueError, match="variance is zero"):
        compute_final_png_metrics(base_files, stable_files, native_shape=(2, 4))


def test_reported_fixture_refuses_pending_identity_before_touching_job(
    tmp_path: Path,
) -> None:
    assert REPORTED_FIXTURE["ordered_source_payload_fingerprint"] == "PENDING_RECAPTURE"
    with pytest.raises(RuntimeError, match="PENDING_RECAPTURE"):
        verify_reported_fixture(tmp_path / "does-not-exist")
