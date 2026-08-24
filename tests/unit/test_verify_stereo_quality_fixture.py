"""Tests for the hash-bound stereo Quality production verifier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import scripts.verify_stereo_quality_fixture as verifier
from scripts.verify_stereo_quality_fixture import (
    FixtureFrame,
    ReviewRegion,
    VerificationConfig,
    run_verification,
)
from src.depth_surge_3d.rendering.stereo_renderer import StereoRenderer, StereoRenderSettings


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def _write_fixture(root: Path, *, compatible_fast: bool = True) -> VerificationConfig:
    height, width = 16, 32
    rows, columns = np.indices((height, width))
    source = np.stack(
        (
            (columns * 7 + rows * 3) % 256,
            (columns * 5 + 40) % 256,
            (rows * 11 + 80) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    canonical = np.full((8, 16), 6554, dtype=np.uint16)
    canonical[:, 5:11] = np.uint16(32768)
    canonical[:, 11:] = np.uint16(58982)
    frame_name = "frame_000001.png"
    source_path = root / "00_original_frames" / frame_name
    canonical_path = root / "03_disparity_maps" / frame_name
    left_path = root / "04_left_frames" / frame_name
    right_path = root / "04_right_frames" / frame_name
    _write_png(source_path, source)
    _write_png(canonical_path, canonical)

    renderer = StereoRenderer(device="cpu")
    settings = StereoRenderSettings(
        stereo_strength=1.5,
        convergence=0.5,
        occlusion_fill="background",
        stereo_render_mode="fast",
        occlusion_fill_max_px=8,
    )
    rendered = renderer.render(source, canonical.astype(np.float32) / np.float32(65535.0), settings)
    if compatible_fast:
        _write_png(left_path, rendered.left_image)
        _write_png(right_path, rendered.right_image)
    else:
        _write_png(left_path, np.zeros_like(source))
        _write_png(right_path, np.zeros_like(source))

    processing = {
        "stereo_strength": 1.5,
        "convergence": 0.5,
        "stereo_geometry_mode": "relative",
        "occlusion_fill": "background",
    }
    settings_path = root / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "metadata": {"settings_schema_version": 4},
                "video_properties": {"width": width, "height": height},
                "processing_settings": processing,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    source_metadata = root / "00_original_frames" / "metadata.json"
    canonical_metadata = root / "03_disparity_maps" / "metadata.json"
    source_metadata.write_text(json.dumps({"fingerprint": "a" * 64}), encoding="utf-8")
    canonical_metadata.write_text(json.dumps({"fingerprint": "b" * 64}), encoding="utf-8")

    frame = FixtureFrame(
        number=1,
        source_sha256=_sha256(source_path),
        canonical_sha256=_sha256(canonical_path),
        fast_left_sha256=_sha256(left_path),
        fast_right_sha256=_sha256(right_path),
        regions=(ReviewRegion("edge", 4, 2, 28, 14),),
    )
    return VerificationConfig(
        settings_relative_path="settings.json",
        settings_raw_sha256=_sha256(settings_path),
        settings_payload_sha256=_canonical_hash(processing),
        source_stage_fingerprint="a" * 64,
        source_frame_fingerprint=None,
        canonical_stage_fingerprint="b" * 64,
        ordered_manifest_sha256=None,
        frames=(frame,),
        render_width=width,
        render_height=height,
        stereo_strength=1.5,
        convergence=0.5,
        occlusion_fill_max_px=8,
        performance_gate_enabled=False,
    )


def test_half_pixel_nearest_diagnostic_has_frozen_mapping() -> None:
    source = np.arange(6, dtype=np.float32).reshape(2, 3)

    actual = verifier._nearest_half_pixel_upsample(source, (4, 6))

    assert np.array_equal(
        actual,
        np.asarray(
            [
                [0, 0, 1, 1, 2, 2],
                [0, 0, 1, 1, 2, 2],
                [3, 3, 4, 4, 5, 5],
                [3, 3, 4, 4, 5, 5],
            ],
            dtype=np.float32,
        ),
    )


def test_cpu_affinity_parser_expands_a_canonical_processor_set() -> None:
    assert verifier._parse_cpu_affinity("0-3,6,8-9") == (0, 1, 2, 3, 6, 8, 9)

    with pytest.raises(ValueError, match="duplicate"):
        verifier._parse_cpu_affinity("0-2,2")
    with pytest.raises(ValueError, match="ascending"):
        verifier._parse_cpu_affinity("3-1")
    with pytest.raises(ValueError, match="nonnegative"):
        verifier._parse_cpu_affinity("-1")


def test_missing_hash_bound_fixture_writes_nonzero_report(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    missing = config.frames[0]
    (tmp_path / "00_original_frames" / missing.file_name).unlink()
    report_path = tmp_path / "report.json"

    exit_code = run_verification(
        tmp_path,
        report_path,
        tmp_path / "artifacts",
        device="cpu",
        config=config,
        benchmark_warmups=0,
        benchmark_measured=0,
        temporal_radius=0,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report["status"] == "fixture_unavailable"
    assert report["inputs"][missing.stem]["source"]["status"] == "missing"


def test_tiny_fixture_writes_four_modes_diagnostics_and_review_crops(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    report_path = tmp_path / "report.json"
    artifacts = tmp_path / "artifacts"

    exit_code = run_verification(
        tmp_path,
        report_path,
        artifacts,
        device="cpu",
        config=config,
        benchmark_warmups=0,
        benchmark_measured=0,
        temporal_radius=0,
        benchmark_command=("python", "verify.py", "--device", "cpu"),
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    frame = report["frames"][0]
    assert exit_code == 0
    assert report["status"] == "machine_pass_human_review_required"
    assert report["gates"]["fast_byte_compatibility"] is True
    assert report["gates"]["quality_final_unresolved_lanes_zero"] is True
    assert report["benchmark_commands"] == [
        {
            "argv": ["python", "verify.py", "--device", "cpu"],
            "cwd": str(Path.cwd().resolve()),
        }
    ]
    assert set(frame["modes"]) == {"fast", "quality", "quality_without_fill", "nearest"}
    assert Path(frame["diagnostics"]["coverage_path"]).is_file()
    assert Path(frame["diagnostics"]["repair_path"]).is_file()
    crop = frame["regions"][0]["artifacts"]["quality_left"]
    crop_image = cv2.imread(crop, cv2.IMREAD_COLOR)
    assert crop_image.shape[:2] == (12 * 4, 24 * 4)


def test_fast_pixel_mismatch_is_a_numeric_failure(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path, compatible_fast=False)
    report_path = tmp_path / "report.json"

    exit_code = run_verification(
        tmp_path,
        report_path,
        tmp_path / "artifacts",
        device="cpu",
        config=config,
        benchmark_warmups=0,
        benchmark_measured=0,
        temporal_radius=0,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert report["status"] == "machine_gate_failure"
    assert report["gates"]["fast_byte_compatibility"] is False
