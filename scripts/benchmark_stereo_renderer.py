"""Reproducible 1080p/4K stereo renderer and PNG pipeline benchmark."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

BENCHMARK_HARNESS = Path(__file__).resolve()
PROJECT_ROOT = Path(
    os.environ.get("DEPTH_SURGE_BENCHMARK_PROJECT_ROOT", BENCHMARK_HARNESS.parents[1])
).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.depth_surge_3d.rendering.forward_splat as forward_splat_module  # noqa: E402
from src.depth_surge_3d.processing.frames.depth_storage import (  # noqa: E402
    canonical_json_hash,
)
from src.depth_surge_3d.processing.frames.stereo_generator import (  # noqa: E402
    STEREO_STAGE_ALGORITHM_VERSION,
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


def _synthetic_inputs(
    width: int,
    height: int,
    fixture: str = "smooth",
) -> tuple[np.ndarray, np.ndarray]:
    if fixture not in {"smooth", "collision"}:
        raise ValueError(f"Unsupported fixture: {fixture}")

    generator = np.random.default_rng(20260804)
    frame = generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    if fixture == "collision":
        rows, columns = np.indices((height, width), dtype=np.int32)
        first_boundary = width // 3 + (rows % 7) - 3
        second_boundary = (2 * width) // 3 - (rows % 5) + 2
        canonical = np.where(
            columns < first_boundary,
            np.float32(0.12),
            np.where(
                columns < second_boundary,
                np.float32(0.56),
                np.float32(0.94),
            ),
        )
        return frame, canonical.astype(np.float32, copy=False)

    horizontal = np.linspace(0.0, 1.0, width, dtype=np.float32)
    vertical = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    canonical = (horizontal[None, :] + vertical) * np.float32(0.5)
    return frame, canonical.astype(np.float32, copy=False)


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


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
    git_commit, git_dirty, git_diff_sha256 = _git_state()
    metadata: dict[str, Any] = {
        "process_id": os.getpid(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "git_diff_sha256": git_diff_sha256,
        "benchmark_harness_sha256": hashlib.sha256(BENCHMARK_HARNESS.read_bytes()).hexdigest(),
        "algorithm_version": STEREO_STAGE_ALGORITHM_VERSION,
        "horizontal_samples": getattr(forward_splat_module, "HORIZONTAL_SUBPIXELS", None),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "gpu_model": None,
        "gpu_driver_version": None,
    }
    if device != "cuda":
        return metadata

    metadata["gpu_model"] = torch.cuda.get_device_name(torch.cuda.current_device())
    driver = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if driver.returncode == 0 and driver.stdout.strip():
        metadata["gpu_driver_version"] = driver.stdout.splitlines()[0].strip()
    return metadata


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


def _create_renderer(device: str) -> StereoRenderer:
    parameters = inspect.signature(StereoRenderer).parameters
    if "profile" in parameters:
        return StereoRenderer(device=device, profile=True)
    return StereoRenderer(device=device)


def _measure_renderer(
    renderer: StereoRenderer,
    frame: np.ndarray,
    canonical: np.ndarray,
    settings: StereoRenderSettings,
    *,
    frame_count: int,
    warmup_count: int,
) -> dict[str, Any]:
    is_cuda = renderer.device.type == "cuda"
    for _ in range(warmup_count):
        warmup = renderer.render(frame, canonical, settings)
        del warmup

    if is_cuda:
        torch.cuda.synchronize(renderer.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(renderer.device)

    latencies: list[float] = []
    geometry: list[float] = []
    geometry_bytes: list[int] = []
    transfer: list[float] = []
    transfer_bytes: list[int] = []
    for _ in range(frame_count):
        if is_cuda:
            torch.cuda.synchronize(renderer.device)
        started = time.perf_counter()
        rendered = renderer.render(frame, canonical, settings)
        if is_cuda:
            torch.cuda.synchronize(renderer.device)
        latencies.append(time.perf_counter() - started)
        profile = getattr(renderer, "last_profile", None)
        geometry.append(profile.host_geometry_seconds if profile is not None else 0.0)
        geometry_bytes.append(profile.host_geometry_bytes if profile is not None else 0)
        transfer.append(profile.offset_transfer_seconds if profile is not None else 0.0)
        transfer_bytes.append(profile.offset_transfer_bytes if profile is not None else 0)
        del rendered

    wall_seconds = sum(latencies)
    allocated = int(torch.cuda.max_memory_allocated(renderer.device)) if is_cuda else 0
    reserved = int(torch.cuda.max_memory_reserved(renderer.device)) if is_cuda else 0
    return {
        "renderer_wall_seconds": wall_seconds,
        "renderer_fps": frame_count / wall_seconds,
        "renderer_latency_median_seconds": float(np.median(latencies)),
        "renderer_latency_p95_seconds": float(np.percentile(latencies, 95, method="linear")),
        "host_geometry_seconds_median": float(np.median(geometry)),
        "host_geometry_bytes_per_frame": int(np.median(geometry_bytes)),
        "offset_transfer_seconds_median": float(np.median(transfer)),
        "offset_transfer_bytes_per_frame": int(np.median(transfer_bytes)),
        "renderer_mean_ms_per_frame": 1000.0 * wall_seconds / frame_count,
        "peak_cuda_allocated_bytes": allocated,
        "peak_cuda_reserved_bytes": reserved,
        "peak_cuda_bytes": allocated,
    }


def _measure_pipeline(
    renderer: StereoRenderer,
    frame: np.ndarray,
    canonical: np.ndarray,
    settings: StereoRenderSettings,
    *,
    frame_count: int,
    workers: int,
    run_dir: Path,
) -> dict[str, Any]:
    frame_files, depth_files, directories = _write_pipeline_inputs(
        run_dir,
        frame,
        canonical,
        frame_count,
    )
    generator = StereoPairGenerator(renderer=renderer)
    started = time.perf_counter()
    succeeded = generator.create_stereo_pairs_from_files(
        frame_files,
        depth_files,
        directories,
        {
            "stereo_strength": settings.stereo_strength,
            "convergence": settings.convergence,
            "occlusion_fill": settings.occlusion_fill,
            "stereo_io_workers": workers,
            "keep_intermediates": False,
        },
    )
    observed_wall = time.perf_counter() - started
    if not succeeded or generator.last_pipeline_stats is None:
        raise RuntimeError("Stereo pipeline benchmark failed")
    stats = generator.last_pipeline_stats
    wall_seconds = stats.pipeline_wall_seconds or observed_wall
    capacity_seconds = wall_seconds * workers
    utilization = (
        0.0
        if capacity_seconds <= 0.0
        else min(100.0, 100.0 * stats.writer_busy_seconds / capacity_seconds)
    )
    return {
        "pipeline_wall_seconds": wall_seconds,
        "pipeline_fps": frame_count / wall_seconds,
        "writer_utilization_percent": utilization,
        "queue_wait_seconds": stats.queue_wait_seconds,
        "permit_wait_seconds": stats.permit_wait_seconds,
        "pipeline_slots": stats.permit_count,
        "written_frames": stats.written_frames,
    }


def benchmark_resolution(
    *,
    width: int,
    height: int,
    frame_count: int,
    warmup_count: int = 5,
    fixture: str = "smooth",
    device: str,
    workers: int,
    workspace: Path,
) -> dict[str, Any]:
    """Benchmark renderer latency and the complete bounded file pipeline."""

    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if warmup_count < 0:
        raise ValueError("warmup_count must not be negative")
    resolved_device = _resolve_device(device)
    run_dir = workspace / f"stereo_{width}x{height}_{uuid.uuid4().hex}"
    frame, canonical = _synthetic_inputs(width, height, fixture)
    renderer = _create_renderer(resolved_device)
    render_settings = StereoRenderSettings(
        stereo_strength=2.0,
        convergence=0.5,
        occlusion_fill="background",
    )

    render_metrics = _measure_renderer(
        renderer,
        frame,
        canonical,
        render_settings,
        frame_count=frame_count,
        warmup_count=warmup_count,
    )
    pipeline_metrics = _measure_pipeline(
        renderer,
        frame,
        canonical,
        render_settings,
        frame_count=frame_count,
        workers=workers,
        run_dir=run_dir,
    )

    result: dict[str, Any] = {
        "resolution": f"{width}x{height}",
        "fixture": fixture,
        "warmup_frames": warmup_count,
        "measured_frames": frame_count,
        "frames": frame_count,
        "device": resolved_device,
        "settings": {
            "stereo_strength": render_settings.stereo_strength,
            "convergence": render_settings.convergence,
            "occlusion_fill": render_settings.occlusion_fill,
        },
    }
    result.update(render_metrics)
    result.update(pipeline_metrics)
    result.update(_runtime_metadata(resolved_device))
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", choices=["1080p", "4k", "all"], default="all")
    parser.add_argument("--fixture", choices=["smooth", "collision", "all"], default="all")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=5)
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
            fixtures = ["collision"]
            frame_count = 1
            warmup_count = 1
        else:
            names = RESOLUTIONS if args.resolution == "all" else [args.resolution]
            targets = [RESOLUTIONS[name] for name in names]
            fixtures = ["smooth", "collision"] if args.fixture == "all" else [args.fixture]
            frame_count = args.frames
            warmup_count = args.warmups
        results = [
            benchmark_resolution(
                width=width,
                height=height,
                frame_count=frame_count,
                warmup_count=warmup_count,
                fixture=fixture,
                device=args.device,
                workers=args.workers,
                workspace=workspace,
            )
            for width, height in targets
            for fixture in fixtures
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
