"""Tests for the opt-in reported-sample verification command."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

import scripts.verify_stereo_edge_fixture as verifier
from scripts.verify_stereo_edge_fixture import (
    FixtureFile,
    Region,
    VerificationConfig,
    _error_metrics,
    run_verification,
)
from src.depth_surge_3d.rendering.stereo_renderer import StereoRenderResult, StereoRenderer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(root: Path) -> tuple[FixtureFile, ...]:
    height, width = 16, 32
    frame = np.full((height, width, 3), (80, 150, 220), dtype=np.uint8)
    canonical = np.full((height, width), 6554, dtype=np.uint16)
    canonical[:, width // 3 : 2 * width // 3] = 32768
    canonical[:, 2 * width // 3 :] = 58982
    baseline = np.zeros_like(frame)
    values = {
        "source": ("source.png", frame),
        "canonical": ("canonical.png", canonical),
        "v1_left": ("left.png", baseline),
        "v1_right": ("right.png", baseline),
    }
    files = []
    for key, (relative_path, image) in values.items():
        path = root / relative_path
        assert cv2.imwrite(str(path), image)
        files.append(FixtureFile(key, relative_path, _sha256(path)))
    return tuple(files)


def _config(files: tuple[FixtureFile, ...]) -> VerificationConfig:
    return VerificationConfig(
        files=files,
        regions=(Region("boundary", 4, 2, 28, 14),),
        render_width=32,
        render_height=16,
        strength=1.25,
        convergence=0.5,
        oracle_samples=8,
        source_halo=4,
    )


def test_missing_fixture_is_a_nonzero_unavailable_report(tmp_path) -> None:
    config = _config(
        (
            FixtureFile("source", "missing.png", "0" * 64),
            FixtureFile("canonical", "canonical.png", "0" * 64),
            FixtureFile("v1_left", "left.png", "0" * 64),
            FixtureFile("v1_right", "right.png", "0" * 64),
        )
    )
    report_path = tmp_path / "report.json"

    exit_code = run_verification(
        tmp_path,
        report_path,
        tmp_path / "crops",
        device="cpu",
        config=config,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code != 0
    assert report["status"] == "fixture_unavailable"
    assert report["inputs"]["source"]["status"] == "missing"


def test_hash_mismatch_is_not_treated_as_a_skip(tmp_path) -> None:
    files = list(_write_fixture(tmp_path))
    source = files[0]
    files[0] = FixtureFile(source.key, source.relative_path, "0" * 64)
    report_path = tmp_path / "report.json"

    exit_code = run_verification(
        tmp_path,
        report_path,
        tmp_path / "crops",
        device="cpu",
        config=_config(tuple(files)),
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code != 0
    assert report["status"] == "fixture_unavailable"
    assert report["inputs"]["source"]["status"] == "hash_mismatch"


def test_p95_aggregates_masked_rgb_channels_without_pixel_averaging() -> None:
    candidate = np.zeros((2, 20, 3), dtype=np.uint8)
    candidate[0, 0, 0] = 255
    oracle = np.zeros_like(candidate)
    baseline = np.zeros_like(candidate)
    edge_mask = np.zeros((2, 20), dtype=np.bool_)
    edge_mask[0] = True

    metrics = _error_metrics(candidate, baseline, oracle, edge_mask)

    assert metrics["candidate_edge_mae"] == 1.0 / 60.0
    assert metrics["candidate_edge_p95"] == 0.0


def test_candidate_geometry_checks_observe_actual_renderer_outputs() -> None:
    config = _config(())
    renderer = StereoRenderer(device="cpu")

    checks = verifier._candidate_geometry_checks(renderer, config)

    assert checks == {
        "stereo_displacement_direction_unchanged": True,
        "zero_parallax_plane_unchanged": True,
    }

    class SwappedEyeRenderer:
        def render(self, *args, **kwargs) -> StereoRenderResult:
            result = renderer.render(*args, **kwargs)
            return StereoRenderResult(
                left_image=result.right_image,
                right_image=result.left_image,
                left_valid_mask=result.right_valid_mask,
                right_valid_mask=result.left_valid_mask,
                left_hole_mask=result.right_hole_mask,
                right_hole_mask=result.left_hole_mask,
            )

    swapped = verifier._candidate_geometry_checks(SwappedEyeRenderer(), config)

    assert swapped["stereo_displacement_direction_unchanged"] is False


def test_tiny_fixture_writes_numeric_report_and_comparison_crop(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    crops_dir = tmp_path / "crops"

    exit_code = run_verification(
        tmp_path,
        report_path,
        crops_dir,
        device="cpu",
        config=_config(_write_fixture(tmp_path)),
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["status"] == "numeric_pass_human_review_required"
    assert report["algorithm_version"] == "torch-horizontal-16x-zbuffer-v3"
    assert report["horizontal_samples"] == 16
    assert isinstance(report["git_dirty"], bool)
    assert report["git_diff_sha256"] is None or len(report["git_diff_sha256"]) == 64
    assert report["settings"]["oracle_samples"] == 8
    assert report["regions"][0]["thresholds"]["passed"] is True
    crop_path = Path(report["regions"][0]["eyes"][0]["crop_path"])
    assert crop_path.name.endswith("_400pct.png")
    assert crop_path.is_file()
    crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    assert crop.shape[:2] == (12 * 4, 24 * 3 * 4)
