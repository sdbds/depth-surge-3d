"""Run the opt-in numeric and visual gate for the reported stereo edge sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as functional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.depth_surge_3d.processing.frames.stereo_generator import (  # noqa: E402
    STEREO_STAGE_ALGORITHM_VERSION,
)
from src.depth_surge_3d.rendering.forward_splat import HORIZONTAL_SUBPIXELS  # noqa: E402
from src.depth_surge_3d.rendering.stereo_renderer import (  # noqa: E402
    StereoRenderer,
    StereoRenderSettings,
)


@dataclass(frozen=True)
class FixtureFile:
    """One hash-pinned local input."""

    key: str
    relative_path: str
    sha256: str


@dataclass(frozen=True)
class Region:
    """Half-open output region evaluated by the manual gate."""

    name: str
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class VerificationConfig:
    """Injectable fixture and oracle settings; production defaults stay pinned."""

    files: tuple[FixtureFile, ...]
    regions: tuple[Region, ...]
    render_width: int
    render_height: int
    strength: float
    convergence: float
    oracle_samples: int = 64
    source_halo: int = 64
    occlusion_fill: str = "background"
    oracle_band_rows: int = 16


DEFAULT_CONFIG = VerificationConfig(
    files=(
        FixtureFile(
            "source",
            "00_original_frames/frame_001213.png",
            "744938e0944ab76cd2074161e90f821776edbe0e0397889416e152760b757b3c",
        ),
        FixtureFile(
            "canonical",
            "03_disparity_maps/frame_001213.png",
            "b14ec2ac557d41e3020ed5df9faef038f22f19580eb35765f2b86550417f4e77",
        ),
        FixtureFile(
            "v1_left",
            "04_left_frames/frame_001213.png",
            "2db507b403329a9c3c4a8212c24f912bd63fb5d0b73355aa620cf3c6393dfd45",
        ),
        FixtureFile(
            "v1_right",
            "04_right_frames/frame_001213.png",
            "e82c495b53c0615617650e1e4a3fdb3851bde98cddc0e563ddc31fb84b3dd599",
        ),
    ),
    regions=(
        Region("left_sleeve_hand", 280, 250, 720, 650),
        Region("right_sleeve", 1080, 140, 1540, 540),
        Region("guitar_dress_boundary", 640, 300, 1160, 920),
    ),
    render_width=1920,
    render_height=1080,
    strength=1.25,
    convergence=0.5,
)


@dataclass(frozen=True)
class LoadedFixture:
    source: np.ndarray
    canonical: np.ndarray
    oracle_canonical: np.ndarray
    v1_left: np.ndarray
    v1_right: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inspect_inputs(
    fixture_root: Path,
    config: VerificationConfig,
) -> tuple[dict[str, dict[str, Any]], bool]:
    inputs: dict[str, dict[str, Any]] = {}
    available = True
    for expected in config.files:
        path = fixture_root / expected.relative_path
        actual_hash = _sha256(path) if path.is_file() else None
        if actual_hash is None:
            status = "missing"
        elif actual_hash != expected.sha256:
            status = "hash_mismatch"
        else:
            status = "verified"
        available = available and status == "verified"
        inputs[expected.key] = {
            "path": str(path.resolve()),
            "expected_sha256": expected.sha256,
            "actual_sha256": actual_hash,
            "status": status,
        }
    return inputs, available


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_state() -> tuple[str, bool, str | None]:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if not status:
        return _git_commit(), False, None

    digest = hashlib.sha256(status)
    tracked_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    digest.update(tracked_diff)
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    for encoded_path in filter(None, untracked.split(b"\0")):
        digest.update(encoded_path)
        path = PROJECT_ROOT / encoded_path.decode("utf-8", errors="surrogateescape")
        if path.is_file():
            digest.update(path.read_bytes())
    return _git_commit(), True, digest.hexdigest()


def _runtime_metadata(device: str) -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "device": device,
        "gpu_model": torch.cuda.get_device_name() if device == "cuda" else None,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_colour(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode colour PNG: {path}")
    return image


def _load_fixture(
    inputs: dict[str, dict[str, Any]],
    config: VerificationConfig,
) -> LoadedFixture:
    source = _read_colour(inputs["source"]["path"])
    canonical_encoded = cv2.imread(
        inputs["canonical"]["path"],
        cv2.IMREAD_UNCHANGED,
    )
    if canonical_encoded is None or canonical_encoded.dtype != np.uint16:
        raise ValueError("Canonical fixture must be a uint16 PNG")
    canonical = canonical_encoded.astype(np.float32) / np.float32(65535.0)
    canonical_tensor = torch.from_numpy(np.ascontiguousarray(canonical)).view(
        1,
        1,
        canonical.shape[0],
        canonical.shape[1],
    )
    oracle_canonical = functional.interpolate(
        canonical_tensor,
        size=(config.render_height, config.render_width),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    return LoadedFixture(
        source=source,
        canonical=canonical,
        oracle_canonical=oracle_canonical,
        v1_left=_read_colour(inputs["v1_left"]["path"]),
        v1_right=_read_colour(inputs["v1_right"]["path"]),
    )


def _validate_loaded_fixture(
    loaded: LoadedFixture,
    config: VerificationConfig,
) -> None:
    expected_colour = (config.render_height, config.render_width, 3)
    for name, image in (
        ("source", loaded.source),
        ("v1_left", loaded.v1_left),
        ("v1_right", loaded.v1_right),
    ):
        if image.shape != expected_colour:
            raise ValueError(f"{name} shape {image.shape} is not {expected_colour}")
    if loaded.canonical.ndim != 2 or not loaded.canonical.size:
        raise ValueError("canonical fixture must be a non-empty 2D image")
    if loaded.oracle_canonical.shape != expected_colour[:2]:
        raise ValueError("resized oracle canonical shape does not match output")


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA verification requested but CUDA is unavailable")
    return device


def _oracle_offsets(
    depths: np.ndarray,
    config: VerificationConfig,
    eye_sign: int,
) -> np.ndarray:
    scale = np.float64(config.render_width) * np.float64(config.strength)
    scale = scale / np.float64(200.0)
    shifts = depths.astype(np.float64) - np.float64(config.convergence)
    shifts = shifts * scale
    shifts = shifts * np.float64(eye_sign)
    shifts = shifts * np.float64(config.oracle_samples)
    return np.ceil(shifts - np.float64(0.5)).astype(np.int64)


def _oracle_winners(
    loaded: LoadedFixture,
    config: VerificationConfig,
    region: Region,
    eye_sign: int,
    start_row: int,
    end_row: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    samples = config.oracle_samples
    target_x0 = max(0, region.x0 - config.source_halo)
    target_x1 = min(config.render_width, region.x1 + config.source_halo)
    source_x0 = target_x0
    source_x1 = target_x1
    fine_width = (target_x1 - target_x0) * samples
    source_depth = np.ascontiguousarray(
        loaded.oracle_canonical[start_row:end_row, source_x0:source_x1]
    )
    offsets = _oracle_offsets(source_depth, config, eye_sign)
    source_columns = np.arange(source_x0, source_x1, dtype=np.int64)
    first_target = source_columns[None, :] * samples + offsets
    lanes = np.arange(samples, dtype=np.int64)
    target_columns = first_target[:, :, None] + lanes
    target_columns -= target_x0 * samples
    inside = (target_columns >= 0) & (target_columns < fine_width)

    band_height = end_row - start_row
    target_rows = np.arange(band_height, dtype=np.int64)[:, None, None]
    linear_targets = target_rows * fine_width + target_columns
    safe_targets = linear_targets.clip(0, band_height * fine_width - 1)
    contribution_depth = np.broadcast_to(source_depth[:, :, None], target_columns.shape)
    maximum_depth = np.full(band_height * fine_width, -np.inf, dtype=np.float32)
    np.maximum.at(
        maximum_depth,
        linear_targets[inside],
        contribution_depth[inside],
    )

    source_rows = np.arange(start_row, end_row, dtype=np.int64)[:, None]
    source_indexes = source_rows * config.render_width + source_columns[None, :]
    source_indexes = np.broadcast_to(source_indexes[:, :, None], target_columns.shape)
    wins_depth = contribution_depth == maximum_depth[safe_targets]
    winning_contribution = inside & wins_depth
    no_source = np.iinfo(np.int64).max
    winner_indexes = np.full(band_height * fine_width, no_source, dtype=np.int64)
    np.minimum.at(
        winner_indexes,
        linear_targets[winning_contribution],
        source_indexes[winning_contribution],
    )
    winner_indexes = winner_indexes.reshape(band_height, fine_width)
    valid = winner_indexes != no_source
    safe_indexes = np.where(valid, winner_indexes, 0)
    colour = loaded.source.reshape(-1, 3)[safe_indexes].copy()
    depth = loaded.oracle_canonical.reshape(-1)[safe_indexes].copy()
    colour[~valid] = 0
    depth[~valid] = -np.inf
    return colour, depth, valid, target_x0


def _invalid_runs(valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.pad(valid, (1, 1), constant_values=True)
    changes = np.diff(padded.astype(np.int8))
    return np.flatnonzero(changes == -1), np.flatnonzero(changes == 1)


def _fill_equal_depth_run(
    colour: np.ndarray,
    depth: np.ndarray,
    start: int,
    end: int,
) -> None:
    left = start - 1
    right = end
    columns = np.arange(start, end)
    use_left = columns - left <= right - columns
    colour[start:end] = np.where(
        use_left[:, None],
        colour[left],
        colour[right],
    )
    depth[start:end] = depth[left]


def _fill_run(
    colour: np.ndarray,
    depth: np.ndarray,
    start: int,
    end: int,
) -> None:
    left = start - 1
    right = end
    if left < 0:
        colour[start:end] = colour[right]
        depth[start:end] = depth[right]
    elif right >= depth.shape[0]:
        colour[start:end] = colour[left]
        depth[start:end] = depth[left]
    elif depth[left] < depth[right]:
        colour[start:end] = colour[left]
        depth[start:end] = depth[left]
    elif depth[right] < depth[left]:
        colour[start:end] = colour[right]
        depth[start:end] = depth[right]
    else:
        _fill_equal_depth_run(colour, depth, start, end)


def _fill_oracle_band(
    colour: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    max_gap_samples: int,
) -> None:
    for row in range(valid.shape[0]):
        starts, ends = _invalid_runs(valid[row])
        for start, end in zip(starts, ends):
            if end - start <= max_gap_samples and not (start == 0 and end == valid.shape[1]):
                _fill_run(colour[row], depth[row], int(start), int(end))
                valid[row, start:end] = True


def _downsample_oracle(colour: np.ndarray, samples: int) -> np.ndarray:
    lanes = colour.reshape(
        colour.shape[0],
        colour.shape[1] // samples,
        samples,
        3,
    )
    averaged = lanes.sum(axis=2, dtype=np.uint32) / np.float64(samples)
    return np.rint(averaged).clip(0, 255).astype(np.uint8)


def _render_oracle_region(
    loaded: LoadedFixture,
    config: VerificationConfig,
    region: Region,
    eye_sign: int,
) -> np.ndarray:
    target_x0 = max(0, region.x0 - config.source_halo)
    target_x1 = min(config.render_width, region.x1 + config.source_halo)
    target_width = target_x1 - target_x0
    rows: list[np.ndarray] = []
    max_gap = config.oracle_samples * (math.ceil(config.render_width * config.strength / 200.0) + 2)
    for start_row in range(region.y0, region.y1, config.oracle_band_rows):
        end_row = min(region.y1, start_row + config.oracle_band_rows)
        colour, depth, valid, observed_x0 = _oracle_winners(
            loaded,
            config,
            region,
            eye_sign,
            start_row,
            end_row,
        )
        if observed_x0 != target_x0:
            raise AssertionError("Oracle target origin changed within one region")
        if config.occlusion_fill == "background":
            _fill_oracle_band(colour, depth, valid, max_gap)
        colour[~valid] = 0
        rows.append(_downsample_oracle(colour, config.oracle_samples))
    rendered = np.concatenate(rows, axis=0)
    if rendered.shape[1] != target_width:
        raise AssertionError("Oracle target width does not match its halo")
    crop_x0 = region.x0 - target_x0
    crop_x1 = crop_x0 + region.x1 - region.x0
    return rendered[:, crop_x0:crop_x1]


def _edge_mask(canonical: np.ndarray, region: Region) -> np.ndarray:
    kernel3 = np.ones((3, 3), dtype=np.uint8)
    local_max = cv2.dilate(canonical, kernel3)
    local_min = cv2.erode(canonical, kernel3)
    edge = (local_max - local_min) >= np.float32(0.02)
    dilated = cv2.dilate(edge.astype(np.uint8), np.ones((9, 9), dtype=np.uint8))
    return dilated[region.y0 : region.y1, region.x0 : region.x1].astype(bool)


def _error_metrics(
    candidate: np.ndarray,
    baseline: np.ndarray,
    oracle: np.ndarray,
    edge_mask: np.ndarray,
) -> dict[str, float]:
    candidate_error = np.abs(candidate.astype(np.float64) - oracle.astype(np.float64))
    baseline_error = np.abs(baseline.astype(np.float64) - oracle.astype(np.float64))
    outside = ~edge_mask
    if not edge_mask.any() or not outside.any():
        raise ValueError("Each review ROI must contain edge and non-edge pixels")
    candidate_edge = candidate_error[edge_mask].reshape(-1)
    baseline_edge = baseline_error[edge_mask].reshape(-1)
    candidate_outside = candidate_error[outside].reshape(-1)
    baseline_outside = baseline_error[outside].reshape(-1)
    return {
        "candidate_edge_mae": float(candidate_edge.mean() / 255.0),
        "v1_edge_mae": float(baseline_edge.mean() / 255.0),
        "candidate_edge_p95": float(np.percentile(candidate_edge, 95, method="linear") / 255.0),
        "v1_edge_p95": float(np.percentile(baseline_edge, 95, method="linear") / 255.0),
        "candidate_outside_mae": float(candidate_outside.mean() / 255.0),
        "v1_outside_mae": float(baseline_outside.mean() / 255.0),
    }


def _thresholds(metrics: dict[str, float]) -> dict[str, bool]:
    mae_pass = metrics["candidate_edge_mae"] <= 0.70 * metrics["v1_edge_mae"]
    p95_pass = metrics["candidate_edge_p95"] <= 0.85 * metrics["v1_edge_p95"]
    outside_pass = metrics["candidate_outside_mae"] <= metrics["v1_outside_mae"] + 1.0 / 255.0
    return {
        "edge_mae_at_most_70_percent_of_v1": mae_pass,
        "edge_p95_at_most_85_percent_of_v1": p95_pass,
        "outside_mae_regression_at_most_one_level": outside_pass,
        "passed": mae_pass and p95_pass and outside_pass,
    }


def _write_comparison_crop(
    path: Path,
    baseline: np.ndarray,
    candidate: np.ndarray,
    oracle: np.ndarray,
) -> None:
    comparison = np.concatenate((baseline, candidate, oracle), axis=1)
    enlarged = cv2.resize(
        comparison,
        None,
        fx=4.0,
        fy=4.0,
        interpolation=cv2.INTER_NEAREST,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), enlarged):
        raise OSError(f"Could not write comparison crop: {path}")


def _marker_centroid(image: np.ndarray) -> float:
    weights = image.astype(np.float64).sum(axis=(0, 2))
    total = float(weights.sum())
    if total <= 0.0:
        return float("nan")
    columns = np.arange(image.shape[1], dtype=np.float64)
    return float(np.dot(columns, weights) / total)


def _candidate_geometry_checks(
    renderer: Any,
    config: VerificationConfig,
) -> dict[str, bool]:
    width = 256
    marker_column = width // 2
    frame = np.zeros((1, width, 3), dtype=np.uint8)
    frame[0, marker_column] = 255
    marker_depth = 1.0 if config.convergence < 1.0 else 0.0
    direction = np.full((1, width), config.convergence, dtype=np.float32)
    direction[0, marker_column] = np.float32(marker_depth)
    settings = StereoRenderSettings(
        stereo_strength=config.strength,
        convergence=config.convergence,
        occlusion_fill="none",
    )
    shifted = renderer.render(frame, direction, settings)
    depth_sign = 1.0 if marker_depth > config.convergence else -1.0
    left_shift = _marker_centroid(shifted.left_image) - marker_column
    right_shift = _marker_centroid(shifted.right_image) - marker_column
    displacement = depth_sign * left_shift > 0.0 and depth_sign * right_shift < 0.0

    zero = np.full((1, width), config.convergence, dtype=np.float32)
    stationary = renderer.render(frame, zero, settings)
    zero_parallax = np.array_equal(stationary.left_image, frame) and np.array_equal(
        stationary.right_image,
        frame,
    )
    return {
        "stereo_displacement_direction_unchanged": displacement,
        "zero_parallax_plane_unchanged": zero_parallax,
    }


def _structural_checks(
    loaded: LoadedFixture,
    left: np.ndarray,
    right: np.ndarray,
    config: VerificationConfig,
    renderer: Any,
) -> dict[str, bool]:
    expected = (config.render_height, config.render_width, 3)
    dimensions = all(
        image.shape == expected for image in (loaded.v1_left, loaded.v1_right, left, right)
    )
    geometry = _candidate_geometry_checks(renderer, config)
    return {
        "output_dimensions_unchanged": dimensions,
        **geometry,
        "passed": dimensions and all(geometry.values()),
    }


def _evaluate_eye(
    *,
    name: str,
    eye_sign: int,
    candidate: np.ndarray,
    baseline: np.ndarray,
    loaded: LoadedFixture,
    config: VerificationConfig,
    region: Region,
    edge_mask: np.ndarray,
    crops_dir: Path,
) -> dict[str, Any]:
    candidate_crop = candidate[region.y0 : region.y1, region.x0 : region.x1]
    baseline_crop = baseline[region.y0 : region.y1, region.x0 : region.x1]
    oracle = _render_oracle_region(loaded, config, region, eye_sign)
    metrics = _error_metrics(candidate_crop, baseline_crop, oracle, edge_mask)
    thresholds = _thresholds(metrics)
    crop_path = (crops_dir / f"{region.name}_{name}_v1_candidate_oracle_400pct.png").resolve()
    _write_comparison_crop(crop_path, baseline_crop, candidate_crop, oracle)
    return {
        "eye": name,
        "metrics": metrics,
        "thresholds": thresholds,
        "crop_path": str(crop_path),
        "crop_order": ["v1", "candidate", "oracle_s64"],
    }


def _evaluate_regions(
    loaded: LoadedFixture,
    left: np.ndarray,
    right: np.ndarray,
    config: VerificationConfig,
    crops_dir: Path,
) -> list[dict[str, Any]]:
    regions = []
    for region in config.regions:
        edge_mask = _edge_mask(loaded.oracle_canonical, region)
        eyes = [
            _evaluate_eye(
                name="left",
                eye_sign=1,
                candidate=left,
                baseline=loaded.v1_left,
                loaded=loaded,
                config=config,
                region=region,
                edge_mask=edge_mask,
                crops_dir=crops_dir,
            ),
            _evaluate_eye(
                name="right",
                eye_sign=-1,
                candidate=right,
                baseline=loaded.v1_right,
                loaded=loaded,
                config=config,
                region=region,
                edge_mask=edge_mask,
                crops_dir=crops_dir,
            ),
        ]
        regions.append(
            {
                "name": region.name,
                "bounds": [region.x0, region.y0, region.x1, region.y1],
                "edge_pixels": int(edge_mask.sum()),
                "eyes": eyes,
                "thresholds": {"passed": all(eye["thresholds"]["passed"] for eye in eyes)},
            }
        )
    return regions


def _settings_report(config: VerificationConfig) -> dict[str, Any]:
    return {
        "render_width": config.render_width,
        "render_height": config.render_height,
        "stereo_strength": config.strength,
        "convergence": config.convergence,
        "occlusion_fill": config.occlusion_fill,
        "oracle_samples": config.oracle_samples,
        "source_halo_pixels": config.source_halo,
        "edge_local_range_threshold": 0.02,
        "edge_dilation_chebyshev_radius": 4,
    }


def _base_report(
    inputs: dict[str, dict[str, Any]],
    config: VerificationConfig,
    device: str,
) -> dict[str, Any]:
    git_commit, git_dirty, git_diff_sha256 = _git_state()
    return {
        "schema_version": 1,
        "candidate_commit": git_commit,
        "git_dirty": git_dirty,
        "git_diff_sha256": git_diff_sha256,
        "algorithm_version": STEREO_STAGE_ALGORITHM_VERSION,
        "horizontal_samples": HORIZONTAL_SUBPIXELS,
        "runtime": _runtime_metadata(device),
        "settings": _settings_report(config),
        "inputs": inputs,
    }


def _run_numeric_gate(
    fixture_root: Path,
    crops_dir: Path,
    config: VerificationConfig,
    device: str,
    inputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    del fixture_root
    loaded = _load_fixture(inputs, config)
    _validate_loaded_fixture(loaded, config)
    settings = StereoRenderSettings(
        stereo_strength=config.strength,
        convergence=config.convergence,
        occlusion_fill=config.occlusion_fill,
    )
    renderer = StereoRenderer(device=device)
    rendered = renderer.render(
        loaded.source,
        loaded.canonical,
        settings,
    )
    structural = _structural_checks(
        loaded,
        rendered.left_image,
        rendered.right_image,
        config,
        renderer,
    )
    regions = _evaluate_regions(
        loaded,
        rendered.left_image,
        rendered.right_image,
        config,
        crops_dir,
    )
    passed = structural["passed"] and all(region["thresholds"]["passed"] for region in regions)
    return {
        "structural_checks": structural,
        "regions": regions,
        "numeric_passed": passed,
    }


def run_verification(
    fixture_root: Path,
    report_json: Path,
    crops_dir: Path,
    *,
    device: str = "auto",
    config: VerificationConfig = DEFAULT_CONFIG,
) -> int:
    """Run the manual gate and always leave a machine-readable status report."""

    resolved_device = _resolve_device(device)
    inputs, available = _inspect_inputs(fixture_root, config)
    report = _base_report(inputs, config, resolved_device)
    if not available:
        report["status"] = "fixture_unavailable"
        _write_report(report_json, report)
        return 1

    try:
        numeric = _run_numeric_gate(
            fixture_root,
            crops_dir,
            config,
            resolved_device,
            inputs,
        )
    except Exception as error:  # The standalone gate must preserve failure evidence.
        report["status"] = "evaluation_error"
        report["error"] = f"{type(error).__name__}: {error}"
        _write_report(report_json, report)
        return 3

    report.update(numeric)
    if numeric["numeric_passed"]:
        report["status"] = "numeric_pass_human_review_required"
        exit_code = 0
    else:
        report["status"] = "numeric_failure"
        exit_code = 2
    _write_report(report_json, report)
    return exit_code


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--crops-dir", required=True, type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    exit_code = run_verification(
        args.fixture_root,
        args.report_json,
        args.crops_dir,
        device=args.device,
    )
    print(args.report_json.resolve())
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
