"""Evaluate the versioned VDPP quality gate from pinned NumPy payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any, Iterable

import numpy as np


METRIC_ALGORITHM = "vdpp-paper-eq9-10-absrel-delta1-v1"
MANIFEST_SCHEMA_VERSION = 1
FIXED_SEEDS = (0, 1, 2)
METRIC_NAMES = ("tgse", "abs_rel", "delta1")


def _volume(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] < 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite [frames, height, width] volume")
    return array


def compute_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute paper TGSE plus standard AbsRel and delta-1 without rounding.

    Predictions and ground truth must already share the benchmark's metric-depth
    scale. TGSE implements equations 9-10 as a mean over valid temporal pairs;
    this normalization is pinned here because it preserves within-clip model
    comparisons while avoiding resolution-dependent totals.
    """

    prediction = _volume("prediction", prediction)
    ground_truth = _volume("ground_truth", ground_truth)
    if prediction.shape != ground_truth.shape:
        raise ValueError("prediction and ground_truth shapes must match")

    if valid_mask is None:
        mask = ground_truth > 0.0
    else:
        mask = np.asarray(valid_mask)
        if mask.shape != ground_truth.shape:
            raise ValueError("valid_mask shape must match ground_truth")
        mask = mask.astype(bool, copy=False) & (ground_truth > 0.0)
    if not mask.any():
        raise ValueError("quality payload contains no valid spatial samples")

    temporal_mask = mask[1:] & mask[:-1]
    if not temporal_mask.any():
        raise ValueError("quality payload contains no valid temporal pairs")
    predicted_gradient = prediction[1:] - prediction[:-1]
    truth_gradient = ground_truth[1:] - ground_truth[:-1]
    temporal_error = np.square(predicted_gradient - truth_gradient)

    spatial_prediction = prediction[mask]
    spatial_truth = ground_truth[mask]
    ratio = np.full(spatial_truth.shape, np.inf, dtype=np.float64)
    positive_prediction = spatial_prediction > 0.0
    ratio[positive_prediction] = np.maximum(
        spatial_prediction[positive_prediction] / spatial_truth[positive_prediction],
        spatial_truth[positive_prediction] / spatial_prediction[positive_prediction],
    )
    return {
        "tgse": float(np.mean(temporal_error[temporal_mask], dtype=np.float64)),
        "abs_rel": float(
            np.mean(np.abs(spatial_prediction - spatial_truth) / spatial_truth, dtype=np.float64)
        ),
        "delta1": float(np.mean(ratio < 1.25, dtype=np.float64)),
    }


def _median_metrics(records: Iterable[dict[str, float]]) -> dict[str, float]:
    materialized = list(records)
    if not materialized:
        raise ValueError("metric records are required")
    return {
        metric: float(statistics.median(record[metric] for record in materialized))
        for metric in METRIC_NAMES
    }


def aggregate_runs(
    runs: list[dict[str, Any]],
    *,
    expected_seeds: tuple[int, ...] = FIXED_SEEDS,
) -> dict[str, Any]:
    """Aggregate repeat medians per sequence, then unweighted sequence medians."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in runs:
        backend = run.get("backend")
        sequence = run.get("sequence")
        seed = run.get("seed")
        if (
            not isinstance(backend, str)
            or not backend
            or not isinstance(sequence, str)
            or not sequence
        ):
            raise ValueError("every run requires non-empty backend and sequence names")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("every run requires an integer seed")
        for variant in ("baseline", "vdpp"):
            metrics = run.get(variant)
            if not isinstance(metrics, dict) or any(
                metric not in metrics for metric in METRIC_NAMES
            ):
                raise ValueError(f"every run requires complete {variant} metrics")
        grouped.setdefault((backend, sequence), []).append(run)

    if not grouped:
        raise ValueError("quality run matrix is empty")

    by_backend: dict[str, list[dict[str, Any]]] = {}
    expected = list(expected_seeds)
    for (backend, sequence), repeats in sorted(grouped.items()):
        seeds = sorted(run["seed"] for run in repeats)
        if seeds != expected:
            raise ValueError(
                f"quality run matrix for {backend}/{sequence} requires seeds {expected}; got {seeds}"
            )
        sequence_result = {
            "sequence": sequence,
            "baseline": _median_metrics(run["baseline"] for run in repeats),
            "vdpp": _median_metrics(run["vdpp"] for run in repeats),
        }
        by_backend.setdefault(backend, []).append(sequence_result)

    result: dict[str, Any] = {}
    for backend, sequences in sorted(by_backend.items()):
        baseline = _median_metrics(sequence["baseline"] for sequence in sequences)
        vdpp = _median_metrics(sequence["vdpp"] for sequence in sequences)
        gate = {
            "tgse": vdpp["tgse"] <= 0.99 * baseline["tgse"],
            "abs_rel": vdpp["abs_rel"] <= 1.02 * baseline["abs_rel"],
            "delta1": vdpp["delta1"] >= baseline["delta1"] - 0.02,
        }
        gate["passed"] = all(gate.values())
        result[backend] = {
            "sequences": sequences,
            "baseline": baseline,
            "vdpp": vdpp,
            "gate": gate,
        }
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_payload(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("every quality run requires a payload path")
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("quality payload must stay inside the manifest directory") from exc
    if not path.is_file():
        raise ValueError(f"quality payload is missing: {path}")
    return path


def evaluate_manifest(manifest_path: Path | str) -> dict[str, Any]:  # noqa: C901
    """Resolve, verify, evaluate, and aggregate one pinned quality manifest."""

    path = Path(manifest_path).resolve()
    raw = path.read_bytes()
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("quality manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("quality manifest must be a JSON object")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported quality manifest schema")
    if manifest.get("metric_algorithm") != METRIC_ALGORITHM:
        raise ValueError("quality metric algorithm does not match this evaluator")
    seeds = manifest.get("seeds")
    if seeds != list(FIXED_SEEDS):
        raise ValueError(f"quality manifest seeds must be {list(FIXED_SEEDS)}")

    identity_names = (
        "source_identity",
        "checkpoint_identity",
        "runtime_identity",
        "base_settings",
        "candidate_settings",
    )
    identities = {}
    for name in identity_names:
        value = manifest.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"quality manifest requires object field {name}")
        identities[name] = value

    raw_runs = manifest.get("runs")
    if not isinstance(raw_runs, list):
        raise ValueError("quality manifest run matrix must be a list")
    evaluated_runs = []
    for raw_run in raw_runs:
        if not isinstance(raw_run, dict):
            raise ValueError("quality run matrix entries must be objects")
        payload_path = _resolved_payload(path.parent, raw_run.get("payload"))
        expected_digest = raw_run.get("payload_sha256")
        if not isinstance(expected_digest, str) or _sha256(payload_path) != expected_digest:
            raise ValueError(f"quality payload digest mismatch: {payload_path}")
        with np.load(payload_path, allow_pickle=False) as payload:
            required = {"baseline", "vdpp", "ground_truth"}
            if not required.issubset(payload.files):
                raise ValueError(f"quality payload is missing arrays: {payload_path}")
            mask = payload["valid_mask"] if "valid_mask" in payload.files else None
            evaluated_runs.append(
                {
                    "backend": raw_run.get("backend"),
                    "sequence": raw_run.get("sequence"),
                    "seed": raw_run.get("seed"),
                    "payload": str(payload_path),
                    "payload_sha256": expected_digest,
                    "baseline": compute_metrics(payload["baseline"], payload["ground_truth"], mask),
                    "vdpp": compute_metrics(payload["vdpp"], payload["ground_truth"], mask),
                }
            )
    evaluated_runs.sort(key=lambda run: (str(run["backend"]), str(run["sequence"]), run["seed"]))
    aggregate = aggregate_runs(evaluated_runs, expected_seeds=FIXED_SEEDS)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "metric_algorithm": METRIC_ALGORITHM,
        "manifest": str(path),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "identities": identities,
        "runs": evaluated_runs,
        "aggregate": aggregate,
        "passed": all(result["gate"]["passed"] for result in aggregate.values()),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = evaluate_manifest(args.manifest)
    _write_json_atomic(args.output, report)
    print(args.output.resolve())
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
