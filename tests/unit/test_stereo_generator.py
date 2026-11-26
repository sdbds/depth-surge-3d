"""Unified StereoPairGenerator tests across Fast/Quality, Relative/Metric, and streaming."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.io import metric_npz
from src.depth_surge_3d.processing.frames.stereo_generator import StereoPairGenerator
from src.depth_surge_3d.processing.frames.stereo_input_manifest import (
    publish_validated_stereo_input_manifest_v1,
)
from src.depth_surge_3d.rendering.stereo_renderer import StereoRenderer


def _write_relative_fixture(
    root: Path, count: int = 3
) -> tuple[list[Path], list[Path], dict[str, Path]]:
    frames_dir = root / "00_original_frames"
    disp_dir = root / "03_disparity_maps"
    left_dir = root / "04_left_frames"
    right_dir = root / "04_right_frames"
    for d in (frames_dir, disp_dir, left_dir, right_dir):
        d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    frames: list[Path] = []
    disps: list[Path] = []
    for i in range(1, count + 1):
        fp = frames_dir / f"frame_{i:06d}.png"
        dp = disp_dir / f"frame_{i:06d}.png"
        cv2.imwrite(str(fp), rng.integers(0, 255, size=(24, 32, 3), dtype=np.uint8))
        grad = np.tile(np.linspace(10000, 50000, 16, dtype=np.uint16), (12, 1))
        cv2.imwrite(str(dp), grad)
        frames.append(fp)
        disps.append(dp)
    (disp_dir / "metadata.json").write_text(
        json.dumps({"encoding_scale": 1.0, "fingerprint": "a" * 64}),
        encoding="utf-8",
    )
    return (
        frames,
        disps,
        {
            "base": root,
            "frames": frames_dir,
            "disparity_maps": disp_dir,
            "left_frames": left_dir,
            "right_frames": right_dir,
        },
    )


@pytest.mark.parametrize("render_mode", ["fast", "quality"])
def test_relative_stereo_generation_and_resume_skip(tmp_path: Path, render_mode: str) -> None:
    frames, disps, dirs = _write_relative_fixture(tmp_path, count=3)
    gen = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))
    settings = {
        "stereo_geometry_mode": "relative",
        "stereo_render_mode": render_mode,
        "stereo_strength": 2.0,
        "convergence": 0.5,
        "occlusion_fill": "background",
        "occlusion_fill_max_px": 8,
    }
    assert gen.create_stereo_pairs_from_files(frames, disps, dirs, settings) is True
    assert len(list(dirs["left_frames"].glob("*.png"))) == 3
    assert len(list(dirs["right_frames"].glob("*.png"))) == 3
    assert (dirs["left_frames"] / "metadata.json").is_file()
    assert (tmp_path / "04_stereo_diagnostics" / "summary.json").is_file()
    assert gen.last_pipeline_stats is not None
    assert gen.last_pipeline_stats.render_count == 3

    # Re-running with identical settings reuses all frames cleanly
    assert gen.create_stereo_pairs_from_files(frames, disps, dirs, settings) is True


def test_streamed_frame_consumer_skips_disk_rgb_writes(tmp_path: Path) -> None:
    frames, disps, dirs = _write_relative_fixture(tmp_path, count=2)
    manifest = publish_validated_stereo_input_manifest_v1(
        job_root=tmp_path, guide_files=frames, geometry_files=disps, geometry_mode="relative"
    )
    seen: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    gen = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))
    ok = gen.create_stereo_pairs_from_manifest(
        manifest,
        dirs,
        {
            "stereo_geometry_mode": "relative",
            "stereo_render_mode": "fast",
            "stereo_strength": 1.5,
            "convergence": 0.5,
            "occlusion_fill": "background",
        },
        frame_consumer=lambda name, l_img, r_img: seen.append((name, l_img.shape, r_img.shape)),
    )
    assert ok is True
    assert [s[0] for s in seen] == ["frame_000001", "frame_000002"]
    assert list(dirs["left_frames"].glob("*.png")) == []


@pytest.mark.parametrize("render_mode", ["fast", "quality"])
def test_metric_stereo_generation_populates_clamp_summary(tmp_path: Path, render_mode: str) -> None:
    frames_dir = tmp_path / "00_original_frames"
    metric_dir = tmp_path / "03_metric_geometry"
    left_dir = tmp_path / "04_left_frames"
    right_dir = tmp_path / "04_right_frames"
    for d in (frames_dir, metric_dir, left_dir, right_dir):
        d.mkdir(parents=True)

    frames: list[Path] = []
    geoms: list[Path] = []
    for idx in (1, 2):
        fp = frames_dir / f"frame_{idx:06d}.png"
        gp = metric_dir / f"frame_{idx:06d}.npz"
        cv2.imwrite(str(fp), np.full((24, 32, 3), 100, dtype=np.uint8))
        inv = np.full((12, 16), 0.5, dtype=np.float32)
        val = np.ones((12, 16), dtype=np.bool_)
        metric_npz.write_metric_npz(gp, inv, val, np.float32(0.8))
        frames.append(fp)
        geoms.append(gp)

    (metric_dir / "metadata.json").write_text(
        json.dumps(
            {
                "native_shape": [12, 16],
                "fingerprint": "b" * 64,
                "convergence": {"resolved_auto_distance_m": 2.0},
            }
        ),
        encoding="utf-8",
    )
    gen = StereoPairGenerator(renderer=StereoRenderer(device="cpu"))
    dirs = {"base": tmp_path, "left_frames": left_dir, "right_frames": right_dir}
    settings = {
        "stereo_geometry_mode": "metric_camera",
        "stereo_render_mode": render_mode,
        "virtual_baseline_mm": 63.0,
        "metric_convergence_distance": "auto",
        "max_disparity_percent": 2.0,
        "crop_factor": 1.0,
        "occlusion_fill": "background",
        "occlusion_fill_max_px": 8,
    }
    assert gen.create_stereo_pairs_from_files(frames, geoms, dirs, settings) is True
    assert gen.last_metric_clamp_summary is not None
    assert gen.last_metric_clamp_summary["valid_pixel_count"] > 0
