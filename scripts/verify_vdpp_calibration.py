"""Verify VDPP calibration from final uint16 PNGs, independently of producer stats."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.depth_surge_3d.core.depth_contract import canonical_json_hash  # noqa: E402
from src.depth_surge_3d.core.render_disparity import (  # noqa: E402
    validate_render_disparity_input,
)


MIDPOINT_CODE = 32768
STATS_TILE_PIXELS = 262144
PHYSICAL_BOUND_BASE_ULPS = 4
PHYSICAL_BOUND_ULPS_PER_PLANNED_TILE = 64
PNG_QUANTIZATION_ALLOWANCE = 1.0 / 65535.0

REPORTED_FIXTURE = {
    "job_basename": (
        "1786983915_BanG Dream Its MyGO S01E03-[1080p][BDRIP][AV1.OPUS]_20260818_002515"
    ),
    "base_canonical_fingerprint": (
        "8292b1291fe6c552fc3843deadbbf5657efe5a1d734f48f1ef3389dc4253cf75"
    ),
    "source_raw_fingerprint": ("b7529c0a7e348ab599ef1446015d210022603e89d3ff960ae419f7aa50f8cc65"),
    "scene_manifest_fingerprint": (
        "08defe1eee63a921191df2db93e1c71637548233df30e051c194e5e911f3967a"
    ),
    "frame_500_sha256": ("cc19d33a8dbceee4056ac0e141980228e335e6d3fb6c930a42c7f185791cecb8"),
    "shot_range": (207, 562),
    "native_shape": (608, 1080),
    "ordered_source_payload_fingerprint": "PENDING_RECAPTURE",
    "frame_500_source_mean": 0.05490560770408996,
    "frame_500_source_std": 0.027351981535192124,
    "frame_500_midpoint_count": 522561,
    "frame_500_midpoint_fraction": 0.7958104897660818,
}


@dataclass(frozen=True)
class _ScalarState:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0


@dataclass(frozen=True)
class _PairState:
    count: int = 0
    mean_x: float = 0.0
    mean_y: float = 0.0
    m2_x: float = 0.0
    m2_y: float = 0.0
    c_xy: float = 0.0


def _canonical(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("final-PNG statistics are non-finite")
    return 0.0 if value == 0.0 else value


def _planned_tiles(frame_count: int, shape: tuple[int, int]) -> int:
    pixels = shape[0] * shape[1]
    return frame_count * ((pixels + STATS_TILE_PIXELS - 1) // STATS_TILE_PIXELS)


def _normalize_bound(
    value: float,
    low: float,
    high: float | None,
    *,
    planned_tiles: int,
) -> float:
    value = _canonical(value)
    reference = max(1.0, abs(low), abs(high) if high is not None else 0.0)
    slack = (
        PHYSICAL_BOUND_BASE_ULPS + PHYSICAL_BOUND_ULPS_PER_PLANNED_TILE * planned_tiles
    ) * math.ulp(reference)
    if value < low:
        if low - value <= slack:
            return _canonical(low)
        raise ValueError("final-PNG statistic is below its physical bound")
    if high is not None and value > high:
        if value - high <= slack:
            return _canonical(high)
        raise ValueError("final-PNG statistic is above its physical bound")
    return value


def _reduce_scalar(values_f32: np.ndarray, eligible: np.ndarray) -> _ScalarState:
    count = int(np.count_nonzero(eligible))
    if count == 0:
        return _ScalarState()
    values64 = np.ascontiguousarray(values_f32[eligible], dtype=np.float64)
    mean = _canonical(float(np.sum(values64, dtype=np.float64)) / count)
    centered = values64 - mean
    m2 = _canonical(float(np.sum(centered * centered, dtype=np.float64)))
    return _ScalarState(count, mean, m2)


def _reduce_pair(
    x_f32: np.ndarray,
    y_f32: np.ndarray,
    eligible: np.ndarray,
) -> _PairState:
    count = int(np.count_nonzero(eligible))
    if count == 0:
        return _PairState()
    x64 = np.ascontiguousarray(x_f32[eligible], dtype=np.float64)
    y64 = np.ascontiguousarray(y_f32[eligible], dtype=np.float64)
    mean_x = _canonical(float(np.sum(x64, dtype=np.float64)) / count)
    mean_y = _canonical(float(np.sum(y64, dtype=np.float64)) / count)
    centered_x = x64 - mean_x
    centered_y = y64 - mean_y
    m2_x = _canonical(float(np.sum(centered_x * centered_x, dtype=np.float64)))
    m2_y = _canonical(float(np.sum(centered_y * centered_y, dtype=np.float64)))
    c_xy = _canonical(float(np.sum(centered_x * centered_y, dtype=np.float64)))
    return _PairState(count, mean_x, mean_y, m2_x, m2_y, c_xy)


def _merge_scalar(a: _ScalarState, b: _ScalarState) -> _ScalarState:
    if a.count == 0:
        return b
    if b.count == 0:
        return a
    n = a.count + b.count
    delta = _canonical(b.mean - a.mean)
    mean = _canonical(a.mean + ((delta * b.count) / n))
    correction = (((delta * delta) * a.count) * b.count) / n
    return _ScalarState(n, mean, _canonical((a.m2 + b.m2) + correction))


def _merge_pair(a: _PairState, b: _PairState) -> _PairState:
    if a.count == 0:
        return b
    if b.count == 0:
        return a
    n = a.count + b.count
    delta_x = _canonical(b.mean_x - a.mean_x)
    delta_y = _canonical(b.mean_y - a.mean_y)
    mean_x = _canonical(a.mean_x + ((delta_x * b.count) / n))
    mean_y = _canonical(a.mean_y + ((delta_y * b.count) / n))
    correction_x = (((delta_x * delta_x) * a.count) * b.count) / n
    correction_y = (((delta_y * delta_y) * a.count) * b.count) / n
    correction_xy = (((delta_x * delta_y) * a.count) * b.count) / n
    return _PairState(
        n,
        mean_x,
        mean_y,
        _canonical((a.m2_x + b.m2_x) + correction_x),
        _canonical((a.m2_y + b.m2_y) + correction_y),
        _canonical((a.c_xy + b.c_xy) + correction_xy),
    )


def _decode_png(path: Path, shape: tuple[int, int]) -> np.ndarray:
    values = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if values is None:
        raise OSError(f"could not decode PNG: {path}")
    if values.dtype != np.uint16 or values.ndim != 2 or values.shape != shape:
        raise ValueError(f"PNG is not native-shape uint16 grayscale: {path}")
    return values


def _has_range(source: np.ndarray) -> bool:
    eligible = source != np.uint16(MIDPOINT_CODE)
    if not bool(np.any(eligible)):
        return False
    low = int(np.min(source, where=eligible, initial=np.iinfo(np.uint16).max))
    high = int(np.max(source, where=eligible, initial=np.iinfo(np.uint16).min))
    return low != high


def compute_final_png_metrics(
    base_files: Iterable[Path],
    stabilized_files: Iterable[Path],
    *,
    native_shape: tuple[int, int],
) -> dict[str, Any]:
    """Independently recompute durable source/output metrics from final PNGs."""

    sources = [Path(path) for path in base_files]
    outputs = [Path(path) for path in stabilized_files]
    if not sources or len(sources) != len(outputs):
        raise ValueError("base and stabilized PNG manifests must have equal non-zero length")
    pair_state = _PairState()
    source_state = _ScalarState()
    output_state = _ScalarState()
    midpoint_count = 0
    flat_frame_count = 0
    output_zero_count = 0
    output_one_count = 0
    pixel_identical = True
    frame_pixels = native_shape[0] * native_shape[1]
    for source_path, output_path in zip(sources, outputs, strict=True):
        source = _decode_png(source_path, native_shape)
        output = _decode_png(output_path, native_shape)
        midpoint = source == np.uint16(MIDPOINT_CODE)
        midpoint_count += int(np.count_nonzero(midpoint))
        if not np.all(output[midpoint] == np.uint16(MIDPOINT_CODE)):
            raise ValueError(f"stabilized PNG changed a source midpoint code: {output_path}")
        output_zero_count += int(np.count_nonzero(output == np.uint16(0)))
        output_one_count += int(np.count_nonzero(output == np.uint16(65535)))
        pixel_identical = pixel_identical and bool(np.array_equal(source, output))
        if not _has_range(source):
            flat_frame_count += 1
            continue
        source_flat = source.reshape(-1)
        output_flat = output.reshape(-1)
        for start in range(0, frame_pixels, STATS_TILE_PIXELS):
            end = min(start + STATS_TILE_PIXELS, frame_pixels)
            source_tile_u16 = source_flat[start:end]
            output_tile_u16 = output_flat[start:end]
            eligible = source_tile_u16 != np.uint16(MIDPOINT_CODE)
            source_f32 = source_tile_u16.astype(np.float32) / np.float32(65535.0)
            output_f32 = output_tile_u16.astype(np.float32) / np.float32(65535.0)
            source_state = _merge_scalar(
                source_state,
                _reduce_scalar(source_f32, eligible),
            )
            output_state = _merge_scalar(
                output_state,
                _reduce_scalar(output_f32, eligible),
            )
            pair_state = _merge_pair(
                pair_state,
                _reduce_pair(output_f32, source_f32, eligible),
            )
    if pair_state.count < 2 or source_state.count != pair_state.count:
        raise ValueError("final PNGs do not contain enough fit-eligible pairs")
    if output_state.count != pair_state.count:
        raise ValueError("final PNG output population changed")

    planned_tiles = _planned_tiles(len(sources), native_shape)
    source_mean = _normalize_bound(source_state.mean, 0.0, 1.0, planned_tiles=planned_tiles)
    source_variance = _normalize_bound(
        _canonical(source_state.m2 / source_state.count),
        0.0,
        0.25,
        planned_tiles=planned_tiles,
    )
    source_std = _normalize_bound(
        _canonical(math.sqrt(source_variance)),
        0.0,
        0.5,
        planned_tiles=planned_tiles,
    )
    output_mean = _normalize_bound(output_state.mean, 0.0, 1.0, planned_tiles=planned_tiles)
    output_variance = _normalize_bound(
        _canonical(output_state.m2 / output_state.count),
        0.0,
        0.25,
        planned_tiles=planned_tiles,
    )
    output_std = _normalize_bound(
        _canonical(math.sqrt(output_variance)),
        0.0,
        0.5,
        planned_tiles=planned_tiles,
    )
    covariance = _canonical(pair_state.c_xy / pair_state.count)
    if source_variance <= 0.0 or output_variance <= 0.0:
        raise ValueError("final PNG source or output variance is zero")
    correlation = _normalize_bound(
        _canonical(covariance / math.sqrt(output_variance * source_variance)),
        -1.0,
        1.0,
        planned_tiles=planned_tiles,
    )
    contrast = _normalize_bound(
        _canonical(output_std / source_std),
        0.0,
        None,
        planned_tiles=planned_tiles,
    )
    drift = _normalize_bound(
        _canonical(abs(output_mean - source_mean)),
        0.0,
        1.0,
        planned_tiles=planned_tiles,
    )
    shot_pixels = len(sources) * frame_pixels
    return {
        "actual_pair_count": pair_state.count,
        "actual_midpoint_count": midpoint_count,
        "actual_midpoint_fraction": _canonical(midpoint_count / shot_pixels),
        "actual_flat_frame_count": flat_frame_count,
        "actual_source_mean": source_mean,
        "actual_source_variance": source_variance,
        "actual_source_std": source_std,
        "actual_output_mean": output_mean,
        "actual_output_std": output_std,
        "actual_output_to_source_correlation": correlation,
        "actual_output_contrast_ratio": contrast,
        "actual_output_mean_drift": drift,
        "actual_endpoint_counts": {
            "zero_count": output_zero_count,
            "zero_fraction": _canonical(output_zero_count / shot_pixels),
            "one_count": output_one_count,
            "one_fraction": _canonical(output_one_count / shot_pixels),
        },
        "pixel_identical_to_base": pixel_identical,
    }


def verify_final_png_quality(
    base_files: Iterable[Path],
    stabilized_files: Iterable[Path],
    calibration: dict[str, Any],
    *,
    native_shape: tuple[int, int],
) -> dict[str, Any]:
    """Apply the real quality verdict to independently recomputed PNG metrics."""

    metrics = compute_final_png_metrics(
        base_files,
        stabilized_files,
        native_shape=native_shape,
    )
    exact_pairs = {
        "actual_pair_count": "pair_count",
        "actual_midpoint_count": "midpoint_count",
        "actual_flat_frame_count": "flat_frame_count",
    }
    for actual_name, diagnostic_name in exact_pairs.items():
        if metrics[actual_name] != calibration.get(diagnostic_name):
            raise ValueError(f"final PNG {actual_name} does not match diagnostics")
    exact_floats = {
        "actual_midpoint_fraction": "midpoint_fraction",
        "actual_source_mean": "source_mean",
        "actual_source_variance": "source_variance",
        "actual_source_std": "source_std",
    }
    for actual_name, diagnostic_name in exact_floats.items():
        diagnostic = calibration.get(diagnostic_name)
        if type(diagnostic) is not float or metrics[actual_name].hex() != diagnostic.hex():
            raise ValueError(f"final PNG {actual_name} does not match diagnostics")

    tolerance = PNG_QUANTIZATION_ALLOWANCE + 4 * math.ulp(1.0)
    if abs(metrics["actual_output_mean"] - calibration.get("candidate_mean", math.inf)) > tolerance:
        raise ValueError("final PNG output mean does not match diagnostics")
    if abs(metrics["actual_output_std"] - calibration.get("candidate_std", math.inf)) > tolerance:
        raise ValueError("final PNG output standard deviation does not match diagnostics")
    if metrics["actual_output_to_source_correlation"] < 0.50:
        raise ValueError("final PNG correlation is below the quality gate")
    if not 0.50 <= metrics["actual_output_contrast_ratio"] <= 1.05:
        raise ValueError("final PNG contrast is outside the quality gate")
    if metrics["actual_output_mean_drift"] > 0.01 + PNG_QUANTIZATION_ALLOWANCE:
        raise ValueError("final PNG mean drift exceeds the quality gate")
    if metrics["pixel_identical_to_base"]:
        raise ValueError("accepted VDPP output is pixel-identical to base")
    return metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_frame_metrics(path: Path, shape: tuple[int, int]) -> dict[str, float | int]:
    source = _decode_png(path, shape)
    flat = source.reshape(-1)
    state = _ScalarState()
    midpoint_count = int(np.count_nonzero(source == np.uint16(MIDPOINT_CODE)))
    for start in range(0, flat.size, STATS_TILE_PIXELS):
        end = min(start + STATS_TILE_PIXELS, flat.size)
        tile = flat[start:end]
        eligible = tile != np.uint16(MIDPOINT_CODE)
        decoded = tile.astype(np.float32) / np.float32(65535.0)
        state = _merge_scalar(state, _reduce_scalar(decoded, eligible))
    planned_tiles = _planned_tiles(1, shape)
    variance = _normalize_bound(
        _canonical(state.m2 / state.count),
        0.0,
        0.25,
        planned_tiles=planned_tiles,
    )
    return {
        "mean": _normalize_bound(state.mean, 0.0, 1.0, planned_tiles=planned_tiles),
        "std": _normalize_bound(
            _canonical(math.sqrt(variance)),
            0.0,
            0.5,
            planned_tiles=planned_tiles,
        ),
        "midpoint_count": midpoint_count,
        "midpoint_fraction": _canonical(midpoint_count / flat.size),
    }


def verify_reported_fixture(job_root: Path | str) -> dict[str, Any]:
    """Verify the reported 355-frame shot once its indivisible fixture is recaptured."""

    expected_payload = REPORTED_FIXTURE["ordered_source_payload_fingerprint"]
    if expected_payload == "PENDING_RECAPTURE":
        raise RuntimeError(
            "VDPP fixture identity is PENDING_RECAPTURE; sign-off is blocked before CUDA/model loading"
        )
    root = Path(job_root).resolve()
    if root.name != REPORTED_FIXTURE["job_basename"]:
        raise ValueError("VDPP fixture job basename does not match")
    start, end = REPORTED_FIXTURE["shot_range"]
    frame_names = [f"frame_{index + 1:06d}.png" for index in range(start, end)]
    base_root = root / "03_disparity_maps"
    stable_root = root / "03_disparity_stabilized"
    base_metadata = json.loads((base_root / "metadata.json").read_text(encoding="utf-8"))
    all_frame_names = base_metadata.get("frame_names")
    if not isinstance(all_frame_names, list) or not all(
        type(name) is str for name in all_frame_names
    ):
        raise ValueError("VDPP fixture base frame manifest is invalid")
    all_frame_files = [Path(name) for name in all_frame_names]
    all_base_files = [base_root / f"{Path(name).stem}.png" for name in all_frame_names]
    base_artifact = validate_render_disparity_input(all_base_files, all_frame_files)
    if (
        base_artifact.producer != "base"
        or base_artifact.fingerprint != REPORTED_FIXTURE["base_canonical_fingerprint"]
    ):
        raise ValueError("VDPP fixture base canonical fingerprint changed")
    if base_metadata.get("source_raw_fingerprint") != REPORTED_FIXTURE["source_raw_fingerprint"]:
        raise ValueError("VDPP fixture source raw fingerprint changed")
    if (
        base_metadata.get("scene_manifest_fingerprint")
        != REPORTED_FIXTURE["scene_manifest_fingerprint"]
    ):
        raise ValueError("VDPP fixture scene manifest fingerprint changed")
    if all_frame_names[start:end] != frame_names:
        raise ValueError("VDPP fixture ordered source frame-name range changed")
    base_files = [base_root / name for name in frame_names]
    stable_files = [stable_root / name for name in frame_names]
    source_records = [{"name": path.name, "sha256": _sha256(path)} for path in base_files]
    if canonical_json_hash(source_records) != expected_payload:
        raise ValueError("VDPP fixture ordered source payload fingerprint changed")
    if _sha256(base_root / "frame_000500.png") != REPORTED_FIXTURE["frame_500_sha256"]:
        raise ValueError("VDPP fixture frame 500 hash changed")
    frame_500 = _source_frame_metrics(
        base_root / "frame_000500.png",
        tuple(REPORTED_FIXTURE["native_shape"]),
    )
    for actual_name, fixture_name in (
        ("mean", "frame_500_source_mean"),
        ("std", "frame_500_source_std"),
        ("midpoint_fraction", "frame_500_midpoint_fraction"),
    ):
        if frame_500[actual_name].hex() != REPORTED_FIXTURE[fixture_name].hex():
            raise ValueError(f"VDPP fixture frame 500 {actual_name} changed")
    if frame_500["midpoint_count"] != REPORTED_FIXTURE["frame_500_midpoint_count"]:
        raise ValueError("VDPP fixture frame 500 midpoint count changed")

    all_stable_files = [stable_root / f"{Path(name).stem}.png" for name in all_frame_names]
    artifact = validate_render_disparity_input(all_stable_files, all_frame_files)
    semantic = artifact.metadata["semantic_identity"]
    matching_shots = [
        shot for shot in semantic["shot_plan"] if (shot["start"], shot["end"]) == (start, end)
    ]
    if len(matching_shots) != 1:
        raise ValueError("VDPP fixture shot range changed")
    shot_id = matching_shots[0]["shot_id"]
    manifest = json.loads(
        (stable_root / f"shot_manifests/shot_{shot_id:06d}.json").read_text(encoding="utf-8")
    )
    calibration = manifest["calibration"]
    if calibration["mode"] != "ols":
        raise ValueError("VDPP fixture did not pass OLS calibration")
    if (
        calibration["correlation"] < 0.50
        or calibration["postclip_contrast_ratio"] < 0.50
        or calibration["postclip_mean_drift"] > 0.01
        or calibration["preclip_low_fraction"] + calibration["preclip_high_fraction"] > 0.01
    ):
        raise ValueError("VDPP fixture diagnostics fail their quality gates")
    return verify_final_png_quality(
        base_files,
        stable_files,
        calibration,
        native_shape=tuple(REPORTED_FIXTURE["native_shape"]),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = verify_reported_fixture(args.job_root)
    except Exception as exc:
        print(f"VDPP calibration verification failed: {exc}")
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
