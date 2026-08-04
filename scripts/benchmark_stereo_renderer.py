"""Non-gating 1080p/4K benchmark for rendering and PNG pipeline throughput."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.depth_surge_3d.processing.frames.depth_storage import (  # noqa: E402
    canonical_json_hash,
)
from src.depth_surge_3d.processing.frames.stereo_generator import (  # noqa: E402
    StereoPairGenerator,
)
from src.depth_surge_3d.rendering.stereo_renderer import (  # noqa: E402
    StereoRenderer,
    StereoRenderSettings,
)


RESOLUTIONS = {
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


def _synthetic_inputs(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(20260804)
    frame = generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    horizontal = np.linspace(0.0, 1.0, width, dtype=np.float32)
    vertical = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    canonical = (horizontal[None, :] + vertical) * np.float32(0.5)
    return frame, canonical.astype(np.float32, copy=False)


def _write_pipeline_inputs(
    root: Path,
    frame: np.ndarray,
    canonical: np.ndarray,
    frame_count: int,
) -> tuple[list[Path], list[Path], dict[str, Path]]:
    frame_dir = root / "frames"
    depth_dir = root / "canonical"
    left_dir = root / "left"
    right_dir = root / "right"
    for directory in (frame_dir, depth_dir, left_dir, right_dir):
        directory.mkdir(parents=True, exist_ok=True)

    encoded_depth = np.rint(canonical * np.float32(65535.0)).astype(np.uint16)
    frame_files: list[Path] = []
    depth_files: list[Path] = []
    for index in range(frame_count):
        name = f"frame_{index:06d}.png"
        frame_path = frame_dir / name
        depth_path = depth_dir / name
        if not cv2.imwrite(str(frame_path), frame):
            raise OSError(f"Could not write benchmark frame: {frame_path}")
        if not cv2.imwrite(str(depth_path), encoded_depth):
            raise OSError(f"Could not write benchmark canonical map: {depth_path}")
        frame_files.append(frame_path)
        depth_files.append(depth_path)

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "algorithm_version": "scene-percentile-v1",
        "representation": "relative_disparity",
        "near_value": 1.0,
        "far_value": 0.0,
        "encoding": "uint16_png",
        "encoding_scale": 65535.0,
        "num_frames": frame_count,
        "frame_names": [path.name for path in frame_files],
        "native_shape": [canonical.shape[0], canonical.shape[1]],
        "source_raw_fingerprint": "benchmark-raw",
        "source_model_fingerprint": "benchmark-model",
        "scene_manifest_fingerprint": "benchmark-scene",
        "depth_bounds_fingerprint": "benchmark-bounds",
    }
    metadata["fingerprint"] = canonical_json_hash(metadata)
    (depth_dir / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True),
        encoding="utf-8",
    )
    return (
        frame_files,
        depth_files,
        {
            "left_frames": left_dir,
            "right_frames": right_dir,
        },
    )


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    return device


def benchmark_resolution(
    *,
    width: int,
    height: int,
    frame_count: int,
    device: str,
    workers: int,
    workspace: Path,
) -> dict[str, Any]:
    """Benchmark renderer latency and the complete bounded file pipeline."""

    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    resolved_device = _resolve_device(device)
    run_dir = workspace / f"stereo_{width}x{height}_{uuid.uuid4().hex}"
    frame, canonical = _synthetic_inputs(width, height)
    renderer = StereoRenderer(device=resolved_device)
    render_settings = StereoRenderSettings(
        stereo_strength=2.0,
        convergence=0.5,
        occlusion_fill="background",
    )

    warmup = renderer.render(frame, canonical, render_settings)
    del warmup
    gpu_milliseconds = 0.0
    peak_cuda_bytes = 0
    if resolved_device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cuda_baseline = torch.cuda.memory_allocated()

    render_started = time.perf_counter()
    for _ in range(frame_count):
        if resolved_device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        result = renderer.render(frame, canonical, render_settings)
        if resolved_device == "cuda":
            end_event.record()
            end_event.synchronize()
            gpu_milliseconds += start_event.elapsed_time(end_event)
        del result
    renderer_wall_seconds = time.perf_counter() - render_started
    if resolved_device == "cuda":
        torch.cuda.synchronize()
        peak_cuda_bytes = int(torch.cuda.max_memory_allocated() - cuda_baseline)

    frame_files, depth_files, directories = _write_pipeline_inputs(
        run_dir,
        frame,
        canonical,
        frame_count,
    )
    generator = StereoPairGenerator(renderer=renderer)
    pipeline_started = time.perf_counter()
    succeeded = generator.create_stereo_pairs_from_files(
        frame_files,
        depth_files,
        directories,
        {
            "stereo_strength": render_settings.stereo_strength,
            "convergence": render_settings.convergence,
            "occlusion_fill": render_settings.occlusion_fill,
            "stereo_io_workers": workers,
            "keep_intermediates": False,
        },
    )
    observed_pipeline_wall = time.perf_counter() - pipeline_started
    if not succeeded or generator.last_pipeline_stats is None:
        raise RuntimeError("Stereo pipeline benchmark failed")
    stats = generator.last_pipeline_stats
    pipeline_wall_seconds = stats.pipeline_wall_seconds or observed_pipeline_wall
    writer_capacity_seconds = pipeline_wall_seconds * workers
    writer_utilization = (
        0.0
        if writer_capacity_seconds <= 0.0
        else min(100.0, 100.0 * stats.writer_busy_seconds / writer_capacity_seconds)
    )

    return {
        "resolution": f"{width}x{height}",
        "frames": frame_count,
        "device": resolved_device,
        "renderer_wall_seconds": renderer_wall_seconds,
        "renderer_fps": frame_count / renderer_wall_seconds,
        "gpu_render_ms_per_frame": (
            gpu_milliseconds / frame_count if resolved_device == "cuda" else None
        ),
        "peak_cuda_bytes": peak_cuda_bytes,
        "pipeline_wall_seconds": pipeline_wall_seconds,
        "pipeline_fps": frame_count / pipeline_wall_seconds,
        "writer_utilization_percent": writer_utilization,
        "queue_wait_seconds": stats.queue_wait_seconds,
        "permit_wait_seconds": stats.permit_wait_seconds,
        "pipeline_slots": stats.permit_count,
        "written_frames": stats.written_frames,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", choices=["1080p", "4k", "all"], default="all")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.workspace is None:
        temporary = tempfile.TemporaryDirectory(prefix="depth-surge-benchmark-")
        workspace = Path(temporary.name)
    else:
        workspace = args.workspace
        workspace.mkdir(parents=True, exist_ok=True)

    try:
        if args.smoke:
            targets = [(64, 36)]
            frame_count = 1
        else:
            names = RESOLUTIONS if args.resolution == "all" else [args.resolution]
            targets = [RESOLUTIONS[name] for name in names]
            frame_count = args.frames
        results = [
            benchmark_resolution(
                width=width,
                height=height,
                frame_count=frame_count,
                device=args.device,
                workers=args.workers,
                workspace=workspace,
            )
            for width, height in targets
        ]
        payload = json.dumps(results, indent=2, sort_keys=True)
        print(payload)
        if args.json_output is not None:
            args.json_output.write_text(payload + "\n", encoding="utf-8")
        return 0
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
