"""Verify an installed wheel from outside the checkout (run with its python -I)."""

from __future__ import annotations

import argparse
from email.parser import Parser
import hashlib
from importlib import machinery, metadata, util
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 160, 96


def check(ok, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def verify_install(wheel: Path) -> Path:
    dist = metadata.distribution("depth-surge-3d")
    site = Path(dist.locate_file("")).resolve()
    check(
        site.is_relative_to(Path(sys.prefix).resolve()), "distribution is outside the environment"
    )
    spec = util.find_spec("depth_surge_3d")
    check(spec is not None and spec.origin is not None, "installed Python package missing")
    package = Path(spec.origin).resolve()
    native_spec = util.find_spec("_quality_kd_native")
    check(
        native_spec is not None and isinstance(native_spec.loader, machinery.ExtensionFileLoader),
        "installed Quality module is not a compiled extension",
    )
    import _quality_kd_native as native

    binary = Path(native.__file__).resolve()
    check(package == site / "depth_surge_3d" / "__init__.py", f"package shadowed: {package}")
    check(binary.parent == site, f"native extension shadowed: {binary}")
    # The matrix publishes CPython-specific builds, not portable or limited-ABI wheels.
    _, py_tag, abi_tag, platforms = wheel.stem.rsplit("-", 3)
    cpython = f"cp{sys.version_info.major}{sys.version_info.minor}"
    wheel_metadata = Parser().parsestr(dist.read_text("WHEEL") or "")
    check(
        py_tag == abi_tag == cpython
        and "any" not in platforms.split(".")
        and wheel_metadata.get("Root-Is-Purelib", "").lower() == "false"
        and set(wheel_metadata.get_all("Tag", []))
        == {f"{py_tag}-{abi_tag}-{platform}" for platform in platforms.split(".")},
        f"not a platform wheel for this CPython ABI: {wheel.name}",
    )
    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()
        check(binary.name in members, "wheel does not contain the loaded native extension")
        check("depth_surge_3d/cli.py" in members, "wheel lacks its CLI module")
        for name in members:
            if not name.endswith("/") and (
                name.startswith("depth_surge_3d/")
                or name == binary.name
                or name.endswith(
                    (".dist-info/METADATA", ".dist-info/WHEEL", ".dist-info/entry_points.txt")
                )
            ):
                check(archive.read(name) == (site / name).read_bytes(), f"wheel mismatch: {name}")
    entries = dist.entry_points.select(group="console_scripts", name="depth-surge-3d")
    check(
        [entry.value for entry in entries] == ["depth_surge_3d.cli:main"],
        "wrong console entry point",
    )
    cli = Path(sysconfig.get_path("scripts")) / (
        "depth-surge-3d.exe" if os.name == "nt" else "depth-surge-3d"
    )
    check(cli.is_file(), "installed console entry point missing")
    check(
        any(f.name == cli.name and f.locate().resolve() == cli.resolve() for f in dist.files or ()),
        "console launcher is not owned by this distribution",
    )
    info = native.build_info()
    digest = hashlib.sha256((ROOT / "native/quality_kd_native.cpp").read_bytes()).hexdigest()
    check(info["source_sha256"] == digest, "native module was built from another source")
    check(info["runtime_jit"] is False and info["kd_stack_capacity"] == 64, "wrong native ABI")
    print(f"Wheel: {wheel.name}\nPackage: {package}\nNative: {binary}\nCLI: {cli}", flush=True)
    return cli


def prepare_known_depth(work: Path, source: Path) -> Path:
    """Seed only the upstream stages; the installed CLI must render and encode the stereo pair."""
    import numpy as np
    from depth_surge_3d.core.settings import validate_settings
    from depth_surge_3d.inference.depth.types import DepthBatch, DepthRepresentation
    from depth_surge_3d.io.operations import (
        create_output_directories,
        get_video_properties,
        save_processing_settings,
    )
    from depth_surge_3d.processing.frames.depth_processor import DepthMapProcessor
    from depth_surge_3d.processing.video.video_encoder import VideoEncoder

    class GradientDepth:
        def estimate_depth_batch(self, frames, **_kwargs):
            n, h, w = frames.shape[:3]
            plane = np.linspace(0, 1, w, dtype=np.float32)
            return DepthBatch(
                np.broadcast_to(plane, (n, h, w)).copy(), DepthRepresentation.INVERSE_DEPTH
            )

    job = work / "job"
    props = get_video_properties(str(source))
    settings = validate_settings(
        dict(
            device="cpu",
            depth_model_version="v3",
            model_size="vits",
            use_metric_depth=False,
            stereo_render_mode="quality",
            depth_resolution=140,
            scene_detection=False,
            keep_intermediates=True,
            preserve_audio=False,
            apply_distortion=False,
            vr_format="side_by_side",
            vr_resolution=f"custom:{WIDTH}x{HEIGHT}",
            per_eye_width=WIDTH,
            per_eye_height=HEIGHT,
            vr_output_width=2 * WIDTH,
            vr_output_height=HEIGHT,
            source_width=WIDTH,
            source_height=HEIGHT,
            source_fps=props["fps"],
            target_fps=None,
            direct_vr_encode=False,
            video_encoder="libx264",
        ),
        source="explicit",
    )
    saved = save_processing_settings(
        job, "quality-smoke", settings, props, str(source), transactional_v5=True
    )
    settings = json.loads(saved.read_bytes())["processing_settings"]
    dirs = create_output_directories(job)
    frames = VideoEncoder().extract_frames(str(source), dirs, props, settings)
    check(len(frames) == 1, "fixture did not extract one frame")
    check(
        DepthMapProcessor(GradientDepth()).generate_depth_map_files(frames, settings, dirs, None),
        "failed to prepare the known-depth fixture",
    )
    return job


def render(cli: Path, work: Path, model: Path | None) -> None:
    import cv2
    import numpy as np

    source = work / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={WIDTH}x{HEIGHT}:rate=24",
            "-frames:v",
            "1",
            "-c:v",
            "libx264",
            str(source),
        ],
        check=True,
        cwd=work,
        timeout=60,
    )
    if model is None:
        job = prepare_known_depth(work, source)
        args = ["--resume", str(job), "--stereo-render-mode", "quality"]
    else:
        args = [
            str(source),
            "--depth-model-version",
            "v3",
            "--model",
            str(model),
            "--device",
            "cpu",
            "--stereo-render-mode",
            "quality",
            "--depth-resolution",
            "140",
            "--format",
            "side_by_side",
            "--vr-resolution",
            f"custom:{WIDTH}x{HEIGHT}",
            "--no-distortion",
            "--no-audio",
            "--keep-intermediates",
            "--no-direct-vr-encode",
            "--output-dir",
            str(work / "output"),
        ]
    subprocess.run([str(cli), *args], cwd=work, timeout=300, check=True)
    (settings_file,) = work.glob("**/processing-settings-v5.json")
    job = settings_file.parent
    saved = json.loads(settings_file.read_bytes())
    check(saved["metadata"]["processing_status"] == "completed", "job did not complete")
    diagnostics = json.loads((job / "04_stereo_diagnostics/metadata.json").read_bytes())
    check(diagnostics["render_mode"] == "quality", "missing Quality rendering evidence")
    movie = job / saved["output_info"]["expected_final_relative_path"]
    capture = cv2.VideoCapture(str(movie))
    try:
        ok, frame = capture.read()
        check(ok and frame.shape == (HEIGHT, WIDTH * 2, 3), "wrong decoded stereo dimensions")
        check(not capture.read()[0], "expected exactly one decoded frame")
        check(
            frame.max() != frame.min() and not np.array_equal(frame[:, :WIDTH], frame[:, WIDTH:]),
            "blank output or identical eye views",
        )
    finally:
        capture.release()
    print(f"PASS: one {WIDTH * 2}x{HEIGHT} Quality stereo frame: {movie}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument(
        "--native-only", action="store_true", help="verify installation and ABI only"
    )
    parser.add_argument(
        "--model", type=Path, help="local DA3-Small snapshot for full depth inference"
    )
    parser.add_argument(
        "--work-dir", type=Path, help="retain artifacts in this new, external directory"
    )
    args = parser.parse_args()
    check(
        sys.flags.isolated and sys.prefix != sys.base_prefix, "use the test environment's python -I"
    )
    check(not Path.cwd().resolve().is_relative_to(ROOT), "run from outside the source checkout")
    cli = verify_install(args.wheel.resolve(strict=True))
    if args.native_only:
        return
    os.environ.pop("PYTHONPATH", None)
    os.environ.pop("PYTHONHOME", None)
    os.environ.update(PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="-1")
    model = args.model.resolve(strict=True) if args.model else None
    if args.work_dir:
        work = args.work_dir.resolve()
        check(not work.is_relative_to(ROOT), "work directory is in the checkout")
        work.mkdir(parents=True, exist_ok=False)
        os.environ["XDG_CACHE_HOME"] = str(work / "cache")
        render(cli, work, model)
    else:
        with tempfile.TemporaryDirectory(prefix="quality-smoke-", dir=Path.cwd()) as tmp:
            os.environ["XDG_CACHE_HOME"] = str(Path(tmp) / "cache")
            render(cli, Path(tmp), model)


if __name__ == "__main__":
    main()
