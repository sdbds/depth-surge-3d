"""Verify the hash-bound seven-frame stereo Quality production fixture."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.depth_surge_3d.processing.frames.stereo_generator import (  # noqa: E402
    STEREO_STAGE_ALGORITHM_VERSION,
)
from src.depth_surge_3d.rendering.forward_splat import HORIZONTAL_SUBPIXELS  # noqa: E402
from src.depth_surge_3d.rendering.quality.diagnostics import (  # noqa: E402
    build_quality_frame_stats,
)
from src.depth_surge_3d.rendering.quality.diagnostic_transaction import (  # noqa: E402
    canonical_json_bytes as diagnostics_canonical_json_bytes,
)
from src.depth_surge_3d.rendering.quality.renderer import (  # noqa: E402
    render_relative_quality_compact,
)
from src.depth_surge_3d.rendering.quality.types import (  # noqa: E402
    CompactStereoRenderResult,
)
from src.depth_surge_3d.rendering.stereo_renderer import (  # noqa: E402
    StereoRenderer,
    StereoRenderSettings,
)


REPORT_SCHEMA_VERSION = 1
REPORT_ALGORITHM_VERSION = "stereo-quality-production-gate-v1"
DEFAULT_FIXTURE_ROOT = Path(
    r"H:\3dtest\1787051840_f908a5f038277cf447b8a6a9b5072311_20260818_191720"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "docs/superpowers/reports/2026-08-24-stereo-quality-production.json"
)
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "output/stereo-quality-production-gate"


@dataclass(frozen=True)
class ReviewRegion:
    """One half-open full-resolution crop used for contour review."""

    name: str
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class FixtureFrame:
    """One hash-pinned source/canonical/Fast tuple."""

    number: int
    source_sha256: str
    canonical_sha256: str
    fast_left_sha256: str
    fast_right_sha256: str
    regions: tuple[ReviewRegion, ...]

    @property
    def stem(self) -> str:
        return f"frame_{self.number:06d}"

    @property
    def file_name(self) -> str:
        return f"{self.stem}.png"


@dataclass(frozen=True)
class VerificationConfig:
    """Injectable immutable fixture and release-gate settings."""

    settings_relative_path: str
    settings_raw_sha256: str
    settings_payload_sha256: str
    source_stage_fingerprint: str
    source_frame_fingerprint: str | None
    canonical_stage_fingerprint: str
    ordered_manifest_sha256: str | None
    frames: tuple[FixtureFrame, ...]
    render_width: int
    render_height: int
    stereo_strength: float
    convergence: float
    occlusion_fill_max_px: int
    performance_gate_enabled: bool = True
    quality_renderer_p95_limit_seconds: float = 5.0
    quality_to_fast_p95_limit: float = 12.0


DEFAULT_CONFIG = VerificationConfig(
    settings_relative_path=("f908a5f038277cf447b8a6a9b5072311_1787051942-settings.json"),
    settings_raw_sha256="13b66e657877c8227b100f10265114f1a88162faff4f6a950d8b2d7234d41fbe",
    settings_payload_sha256=("861ba59c027f57c62b94460b23906d6ddcb7c0fde50c96e002f72ce9544180da"),
    source_stage_fingerprint="29ccafdddc83b6c408708b29c41e5d2fabc392028d7d4149a11bdb9b9fbc60c3",
    source_frame_fingerprint="af573cc60b900ab279466c11e53c9e64daca173b1818e30509d562e8485d78e4",
    canonical_stage_fingerprint=(
        "bfc9dca3f7a4ee61816df04e97b1fc56b89d54a536cb5b434862883d0a7ec7fa"
    ),
    ordered_manifest_sha256="ee9b8b6126667baa995e6c8055340e8494b38e4d6f9d65c64554d4389127c979",
    frames=(
        FixtureFrame(
            89,
            "cca5b9dd367ab23d4931c73ec98b1d091c431c69385e5d1077608f4ec3fd060b",
            "66d524dd394d82ad9b96c1fda89b725b1b9ff954a25b26fcd0ea0eed8138f77c",
            "9f2b60cbc1aee589ed00afa6d854eaaa009ae708acc62534a47dddcb68473bfd",
            "be77f8fc1f8552c4d8240350d1e1a5c6f1b7d89efca059b260eed5490426534a",
            (
                ReviewRegion("boot_edge", 1370, 210, 1650, 570),
                ReviewRegion("thin_ribbon", 930, 430, 1320, 720),
            ),
        ),
        FixtureFrame(
            111,
            "6e5213f560e651495c8df6f98663b8d01190403ca55fa51d811415ce8131a11c",
            "cf63208d7f948274ecdbd14445b19b557efe13cf329bfb35879a01bfc22d5281",
            "306c73362fe4e6df290a7beff0330078f53ca152c2a736b1a4aa42192efb724b",
            "35f773da7ad214c918963144c9acf529ad0641141119007bf8c1c4a7299ab48f",
            (
                ReviewRegion("hat_feather", 0, 320, 340, 700),
                ReviewRegion("hair_curl", 0, 650, 480, 930),
            ),
        ),
        FixtureFrame(
            171,
            "6a3f46b3d952ab44bb6dcff85364e97b40e320168ed9c52272fd087f32b310cb",
            "deb8512a101beee29fe08b53db2ac779ea864b47a1e2d21da355fafd22aa6fff",
            "4b57b0ae58ad6acf6484f7a66319d3a3e6fcbc7db82110e46041600a08095a0f",
            "b9c20330df16b3a47ddef7992a76b81ae431701598db1c2c37dd105d09b8fcf0",
            (
                ReviewRegion("hair_and_feather", 720, 140, 1050, 560),
                ReviewRegion("gun_boundary", 980, 280, 1450, 800),
            ),
        ),
        FixtureFrame(
            176,
            "154969bd5ff228c94bfd5f1b0b0f7c66531d55758abcdbd03de53d363192d4bb",
            "48e12f34b40345b81dcca7104e216cf5ae8a71614d8faed6a6a1f8061d55c1eb",
            "890cd5db85b17dfde53e2bcc466a179a25b6b2f9b4b786413a6f4fcd8f52979c",
            "c741c8bc7356ee8decd8c3178df46a3673dcee0813e2ef7a0697a2151e6f9c60",
            (
                ReviewRegion("hair_curl", 580, 250, 900, 540),
                ReviewRegion("gun_muzzle", 890, 280, 1370, 760),
                ReviewRegion("chest_ribbon", 450, 330, 900, 620),
            ),
        ),
        FixtureFrame(
            231,
            "be5f4a523a88386d0a9f781332a02cd6d9b716dc49e0504a052f91949f99a73b",
            "1fcf5555f1679e564dd73aaa324ee0c873468e2f0ba6165ccbf5a7dfb3bb8d35",
            "e48e5043538abb16f01d8eab2686bf62dfa3ffbaecee3a297f12403bd605f9c4",
            "e7efaa5096bb28d5763d398f690ec9d1493a299ea16e72d171574e9365b23db5",
            (
                ReviewRegion("long_gun_barrel", 500, 480, 1060, 800),
                ReviewRegion("twin_curls", 1120, 380, 1580, 730),
                ReviewRegion("chest_ribbon", 1240, 650, 1600, 900),
            ),
        ),
        FixtureFrame(
            301,
            "0d9182b0aba89afdfafd5adce9a7d2ca0db5742e8bc5b2db04d07f4f1884c932",
            "296511484ffddee5b6dd74ad8bc7b51e7da2757a7dc4444ac08fee9ae9f2745b",
            "23031b568c3868e3f22487cce73a3784ddc67b41c9457dcf627cff5f84ae5896",
            "eecf711d3a603d1e82c523df455e178a59ec47c10f9f6d2bde0f50c53fc275ea",
            (
                ReviewRegion("crossed_gun_barrels", 430, 160, 1300, 780),
                ReviewRegion("waist_ribbon", 940, 500, 1300, 850),
            ),
        ),
        FixtureFrame(
            401,
            "fdf65df83cfe7d647f66b651eeb9cabe0a53c83578a66aad8f7b40662293ba3a",
            "4cecaffe00821a84ff81f1422ebc2ef6b50269a0541eb4f23593a1fc80d20347",
            "2a554442e95657ca9b94c0524c4a65a732ea64f26610860c779d47eac6f2b065",
            "93b39254b0e685ac44ce91272ab723e076c3afe4dab4f0346acafd1a577ed1ca",
            (ReviewRegion("ring_guns_and_silhouette", 730, 170, 1190, 600),),
        ),
    ),
    render_width=1920,
    render_height=1080,
    stereo_strength=1.5,
    convergence=0.5,
    occlusion_fill_max_px=8,
)


@dataclass(frozen=True)
class _LoadedFrame:
    source: np.ndarray
    canonical: np.ndarray
    fast_left: np.ndarray
    fast_right: np.ndarray


@dataclass(frozen=True)
class _RenderedMode:
    compact: CompactStereoRenderResult
    seconds: float


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_report(path: Path, report: dict[str, Any]) -> None:
    report.pop("fingerprint", None)
    report["fingerprint"] = _canonical_hash(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_png(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write PNG: {path}")
    return _sha256(path)


def _read_colour(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode RGB PNG: {path}")
    return np.ascontiguousarray(image)


def _read_canonical(path: Path) -> np.ndarray:
    encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if encoded is None or encoded.dtype != np.uint16 or encoded.ndim != 2:
        raise ValueError(f"Canonical PNG must decode as uint16 [H,W]: {path}")
    result = np.ascontiguousarray(encoded.astype(np.float32))
    result *= np.float32(1.0 / 65535.0)
    return result


def _file_status(path: Path, expected_hash: str) -> dict[str, Any]:
    actual = _sha256(path) if path.is_file() else None
    status = (
        "missing" if actual is None else "verified" if actual == expected_hash else "hash_mismatch"
    )
    return {
        "path": str(path.resolve()),
        "expected_sha256": expected_hash,
        "actual_sha256": actual,
        "status": status,
    }


def _metadata_status(path: Path, field: str, expected: str | None) -> dict[str, Any]:
    try:
        value = _read_json(path)
        actual = value.get(field)
    except (OSError, json.JSONDecodeError, ValueError):
        actual = None
    verified = expected is None or actual == expected
    return {
        "path": str(path.resolve()),
        "field": field,
        "expected": expected,
        "actual": actual,
        "status": "verified" if verified else "metadata_mismatch",
    }


def _settings_status(fixture_root: Path, config: VerificationConfig) -> dict[str, Any]:
    path = fixture_root / config.settings_relative_path
    raw = _file_status(path, config.settings_raw_sha256)
    try:
        settings = _read_json(path)
        processing = settings["processing_settings"]
        properties = settings["video_properties"]
        payload_hash = _canonical_hash(processing)
        semantics_match = (
            isinstance(processing, dict)
            and isinstance(properties, dict)
            and payload_hash == config.settings_payload_sha256
            and properties.get("width") == config.render_width
            and properties.get("height") == config.render_height
            and processing.get("stereo_strength") == config.stereo_strength
            and processing.get("convergence") == config.convergence
            and processing.get("stereo_geometry_mode") == "relative"
            and processing.get("occlusion_fill") == "background"
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        payload_hash = None
        semantics_match = False
    raw["expected_payload_sha256"] = config.settings_payload_sha256
    raw["actual_payload_sha256"] = payload_hash
    raw["semantics_match"] = semantics_match
    if raw["status"] == "verified" and not semantics_match:
        raw["status"] = "settings_semantics_mismatch"
    return raw


def _frame_paths(fixture_root: Path, frame: FixtureFrame) -> dict[str, Path]:
    return {
        "source": fixture_root / "00_original_frames" / frame.file_name,
        "canonical": fixture_root / "03_disparity_maps" / frame.file_name,
        "fast_left": fixture_root / "04_left_frames" / frame.file_name,
        "fast_right": fixture_root / "04_right_frames" / frame.file_name,
    }


def _inspect_inputs(
    fixture_root: Path,
    config: VerificationConfig,
) -> tuple[dict[str, Any], bool]:
    settings = _settings_status(fixture_root, config)
    stages = {
        "source_stage": _metadata_status(
            fixture_root / "00_original_frames/metadata.json",
            "fingerprint",
            config.source_stage_fingerprint,
        ),
        "source_frames": _metadata_status(
            fixture_root / "00_original_frames/metadata.json",
            "source_frame_fingerprint",
            config.source_frame_fingerprint,
        ),
        "canonical_stage": _metadata_status(
            fixture_root / "03_disparity_maps/metadata.json",
            "fingerprint",
            config.canonical_stage_fingerprint,
        ),
    }
    frames: dict[str, Any] = {}
    for frame in config.frames:
        paths = _frame_paths(fixture_root, frame)
        frames[frame.stem] = {
            "source": _file_status(paths["source"], frame.source_sha256),
            "canonical": _file_status(paths["canonical"], frame.canonical_sha256),
            "fast_left": _file_status(paths["fast_left"], frame.fast_left_sha256),
            "fast_right": _file_status(paths["fast_right"], frame.fast_right_sha256),
        }
    statuses = [settings["status"]]
    statuses.extend(value["status"] for value in stages.values())
    statuses.extend(
        value["status"] for frame_values in frames.values() for value in frame_values.values()
    )
    inputs = {
        "settings": settings,
        "stages": stages,
        **frames,
        "spec_declared_ordered_manifest_sha256": config.ordered_manifest_sha256,
        "embedded_manifest_sha256": _canonical_hash(
            [
                {
                    "frame_name": frame.stem,
                    "source_sha256": frame.source_sha256,
                    "canonical_sha256": frame.canonical_sha256,
                    "fast_left_sha256": frame.fast_left_sha256,
                    "fast_right_sha256": frame.fast_right_sha256,
                }
                for frame in config.frames
            ]
        ),
    }
    return inputs, all(value == "verified" for value in statuses)


def _load_frame(
    fixture_root: Path, frame: FixtureFrame, config: VerificationConfig
) -> _LoadedFrame:
    paths = _frame_paths(fixture_root, frame)
    loaded = _LoadedFrame(
        source=_read_colour(paths["source"]),
        canonical=_read_canonical(paths["canonical"]),
        fast_left=_read_colour(paths["fast_left"]),
        fast_right=_read_colour(paths["fast_right"]),
    )
    expected_rgb = (config.render_height, config.render_width, 3)
    for name, value in (
        ("source", loaded.source),
        ("fast_left", loaded.fast_left),
        ("fast_right", loaded.fast_right),
    ):
        if value.dtype != np.uint8 or value.shape != expected_rgb:
            raise ValueError(
                f"{frame.stem} {name} shape/dtype changed: {value.shape}/{value.dtype}"
            )
    if loaded.canonical.dtype != np.float32 or loaded.canonical.ndim != 2:
        raise ValueError(f"{frame.stem} canonical shape/dtype changed")
    return loaded


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA verification requested but CUDA is unavailable")
    return device


def _synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _time_render(
    operation: Callable[[], CompactStereoRenderResult],
    device: str,
) -> _RenderedMode:
    _synchronize(device)
    started = time.perf_counter()
    compact = operation()
    _synchronize(device)
    return _RenderedMode(compact=compact, seconds=time.perf_counter() - started)


def _nearest_axis(source_size: int, destination_size: int) -> np.ndarray:
    coordinates = np.arange(destination_size, dtype=np.float64)
    coordinates += np.float64(0.5)
    coordinates *= np.float64(source_size)
    coordinates /= np.float64(destination_size)
    coordinates -= np.float64(0.5)
    indexes = np.floor(coordinates + np.float64(0.5)).astype(np.int64)
    return np.clip(indexes, 0, source_size - 1)


def _nearest_half_pixel_upsample(
    source: np.ndarray,
    render_shape: tuple[int, int],
) -> np.ndarray:
    """Apply the explicit half-pixel nearest diagnostic without a library resize."""

    if source.ndim != 2 or source.shape[0] <= 0 or source.shape[1] <= 0:
        raise ValueError("nearest diagnostic source must be a non-empty 2D raster")
    rows = _nearest_axis(source.shape[0], render_shape[0])
    columns = _nearest_axis(source.shape[1], render_shape[1])
    return np.ascontiguousarray(source[rows[:, None], columns[None, :]])


def _settings(
    config: VerificationConfig,
    *,
    mode: Literal["fast", "quality"],
    fill: Literal["none", "background"],
) -> StereoRenderSettings:
    return StereoRenderSettings(
        stereo_strength=config.stereo_strength,
        convergence=config.convergence,
        occlusion_fill=fill,
        stereo_render_mode=mode,
        occlusion_fill_max_px=config.occlusion_fill_max_px,
    )


def _render_modes(
    renderer: StereoRenderer,
    loaded: _LoadedFrame,
    config: VerificationConfig,
    device: str,
) -> dict[str, _RenderedMode]:
    fast_settings = _settings(config, mode="fast", fill="background")
    quality_settings = _settings(config, mode="quality", fill="background")
    no_fill_settings = _settings(config, mode="quality", fill="none")
    nearest = _nearest_half_pixel_upsample(loaded.canonical, loaded.source.shape[:2])
    with torch.inference_mode():
        modes = {
            "fast": _time_render(
                lambda: renderer.render_relative_fast_compact(
                    loaded.source, loaded.canonical, fast_settings
                ),
                device,
            ),
            "quality": _time_render(
                lambda: render_relative_quality_compact(
                    renderer=renderer,
                    frame=loaded.source,
                    canonical=loaded.canonical,
                    settings=quality_settings,
                ),
                device,
            ),
            "quality_without_fill": _time_render(
                lambda: render_relative_quality_compact(
                    renderer=renderer,
                    frame=loaded.source,
                    canonical=loaded.canonical,
                    settings=no_fill_settings,
                ),
                device,
            ),
            "nearest": _time_render(
                lambda: renderer.render_relative_fast_compact(
                    loaded.source, nearest, fast_settings
                ),
                device,
            ),
        }
    return modes


def _coverage_visual(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    def colourize(values: np.ndarray) -> np.ndarray:
        output = np.full((*values.shape, 3), (32, 32, 32), dtype=np.uint8)
        partial = (values > 0) & (values < HORIZONTAL_SUBPIXELS)
        output[partial] = (0, 180, 255)
        output[values == 0] = (255, 0, 255)
        return output

    return np.concatenate((colourize(left), colourize(right)), axis=1)


def _repair_visual(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    def colourize(bits: np.ndarray) -> np.ndarray:
        output = np.full((*bits.shape, 3), (32, 32, 32), dtype=np.uint8)
        output[(bits & np.uint8(0x01)) != 0] = (0, 180, 255)
        output[(bits & np.uint8(0x02)) != 0] = (0, 96, 255)
        output[(bits & np.uint8(0x04)) != 0] = (0, 220, 0)
        output[(bits & np.uint8(0x10)) != 0] = (220, 220, 0)
        output[(bits & np.uint8(0x20)) != 0] = (220, 0, 220)
        output[(bits & np.uint8(0x40)) != 0] = (0, 0, 255)
        return output

    return np.concatenate((colourize(left), colourize(right)), axis=1)


def _crop_400(image: np.ndarray, region: ReviewRegion) -> np.ndarray:
    crop = image[region.y0 : region.y1, region.x0 : region.x1]
    if crop.size == 0:
        raise ValueError(f"Review region is empty: {region.name}")
    return cv2.resize(crop, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST)


def _mode_stats(
    name: str,
    rendered: _RenderedMode,
    frame_name: str,
) -> dict[str, Any]:
    fill = "none" if name == "quality_without_fill" else "background"
    render_mode = "quality" if name.startswith("quality") else "fast"
    compact = rendered.compact
    stats = build_quality_frame_stats(
        compact,
        frame_name=frame_name,
        geometry_mode="relative",
        occlusion_fill=fill,
        render_mode=render_mode,
    )
    serialized_stats = json.loads(diagnostics_canonical_json_bytes(stats))
    return {
        "render_seconds": rendered.seconds,
        "left_pixel_sha256": _array_sha256(compact.left_image),
        "right_pixel_sha256": _array_sha256(compact.right_image),
        "diagnostics": serialized_stats,
    }


def _write_mode_artifacts(
    artifacts_dir: Path,
    frame: FixtureFrame,
    modes: dict[str, _RenderedMode],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for name, rendered in modes.items():
        compact = rendered.compact
        eyes: dict[str, Any] = {}
        for eye, image in (("left", compact.left_image), ("right", compact.right_image)):
            path = (artifacts_dir / "full" / f"{frame.stem}_{name}_{eye}.png").resolve()
            eyes[eye] = {"path": str(path), "png_sha256": _write_png(path, image)}
        output[name] = eyes
    return output


def _region_report(
    artifacts_dir: Path,
    frame: FixtureFrame,
    modes: dict[str, _RenderedMode],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for region in frame.regions:
        paths: dict[str, str] = {}
        quality = modes["quality"].compact
        fast = modes["fast"].compact
        for mode_name, rendered in modes.items():
            for eye, image in (
                ("left", rendered.compact.left_image),
                ("right", rendered.compact.right_image),
            ):
                key = f"{mode_name}_{eye}"
                path = (
                    artifacts_dir / "crops" / f"{frame.stem}_{region.name}_{key}_400pct.png"
                ).resolve()
                _write_png(path, _crop_400(image, region))
                paths[key] = str(path)
        ys = slice(region.y0, region.y1)
        xs = slice(region.x0, region.x1)
        left_difference = np.abs(
            quality.left_image[ys, xs].astype(np.int16) - fast.left_image[ys, xs].astype(np.int16)
        )
        right_difference = np.abs(
            quality.right_image[ys, xs].astype(np.int16) - fast.right_image[ys, xs].astype(np.int16)
        )
        reports.append(
            {
                "name": region.name,
                "bounds": [region.x0, region.y0, region.x1, region.y1],
                "artifacts": paths,
                "quality_vs_fast_mae": {
                    "left": float(left_difference.mean()),
                    "right": float(right_difference.mean()),
                },
                "quality_repaired_pixel_count": {
                    "left": int(np.count_nonzero(quality.left_repair_bits[ys, xs] & 0x34)),
                    "right": int(np.count_nonzero(quality.right_repair_bits[ys, xs] & 0x34)),
                },
            }
        )
    return reports


def _write_diagnostics(
    artifacts_dir: Path,
    frame: FixtureFrame,
    quality: CompactStereoRenderResult,
) -> dict[str, str]:
    coverage_path = (artifacts_dir / "diagnostics" / f"{frame.stem}_coverage.png").resolve()
    repair_path = (artifacts_dir / "diagnostics" / f"{frame.stem}_repair.png").resolve()
    _write_png(
        coverage_path,
        _coverage_visual(quality.left_coverage_count, quality.right_coverage_count),
    )
    _write_png(
        repair_path,
        _repair_visual(quality.left_repair_bits, quality.right_repair_bits),
    )
    return {"coverage_path": str(coverage_path), "repair_path": str(repair_path)}


def _frame_machine_checks(
    loaded: _LoadedFrame,
    modes: dict[str, _RenderedMode],
    artifacts: dict[str, dict[str, Any]],
    frame: FixtureFrame,
) -> dict[str, bool]:
    fast = modes["fast"].compact
    quality = modes["quality"].compact
    no_fill = modes["quality_without_fill"].compact
    expected_shape = loaded.source.shape
    all_images = [
        image
        for rendered in modes.values()
        for image in (rendered.compact.left_image, rendered.compact.right_image)
    ]
    return {
        "fast_left_pixels_equal": np.array_equal(fast.left_image, loaded.fast_left),
        "fast_right_pixels_equal": np.array_equal(fast.right_image, loaded.fast_right),
        "fast_left_png_equal": artifacts["fast"]["left"]["png_sha256"] == frame.fast_left_sha256,
        "fast_right_png_equal": artifacts["fast"]["right"]["png_sha256"] == frame.fast_right_sha256,
        "all_output_shapes_and_dtypes_valid": all(
            value.shape == expected_shape and value.dtype == np.uint8 for value in all_images
        ),
        "quality_final_unresolved_pixels_zero": not bool(
            np.any(quality.left_repair_bits & np.uint8(0x40))
            or np.any(quality.right_repair_bits & np.uint8(0x40))
        ),
        "quality_without_fill_exposes_geometry_holes": bool(
            np.any(no_fill.left_repair_bits & np.uint8(0x40))
            or np.any(no_fill.right_repair_bits & np.uint8(0x40))
        ),
        "nearest_diagnostic_is_distinct": bool(
            not np.array_equal(modes["nearest"].compact.left_image, fast.left_image)
            or not np.array_equal(modes["nearest"].compact.right_image, fast.right_image)
        ),
    }


def _evaluate_frame(
    fixture_root: Path,
    artifacts_dir: Path,
    renderer: StereoRenderer,
    frame: FixtureFrame,
    config: VerificationConfig,
    device: str,
) -> tuple[dict[str, Any], dict[str, bool]]:
    loaded = _load_frame(fixture_root, frame, config)
    modes = _render_modes(renderer, loaded, config, device)
    artifacts = _write_mode_artifacts(artifacts_dir, frame, modes)
    checks = _frame_machine_checks(loaded, modes, artifacts, frame)
    report = {
        "frame_name": frame.stem,
        "modes": {
            name: {**_mode_stats(name, rendered, frame.stem), "artifacts": artifacts[name]}
            for name, rendered in modes.items()
        },
        "diagnostics": _write_diagnostics(artifacts_dir, frame, modes["quality"].compact),
        "regions": _region_report(artifacts_dir, frame, modes),
        "machine_checks": checks,
    }
    del loaded, modes
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return report, checks


def _linear_p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _parse_cpu_affinity(value: str) -> tuple[int, ...]:  # noqa: C901
    """Parse a comma-separated processor list with inclusive ranges."""

    if not value.strip():
        raise ValueError("CPU affinity must not be empty")
    processors: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError("CPU affinity contains an empty processor")
        if part.startswith("-"):
            raise ValueError("CPU affinity processors must be nonnegative")
        if "-" in part:
            if part.count("-") != 1:
                raise ValueError(f"invalid CPU affinity range: {part!r}")
            start_text, end_text = part.split("-", maxsplit=1)
            if not start_text.isdecimal() or not end_text.isdecimal():
                raise ValueError(f"invalid CPU affinity range: {part!r}")
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError("CPU affinity ranges must be ascending")
            expanded = tuple(range(start, end + 1))
        else:
            if not part.isdecimal():
                raise ValueError(f"invalid CPU affinity processor: {part!r}")
            expanded = (int(part),)
        for processor in expanded:
            if processor in processors:
                raise ValueError(f"duplicate CPU affinity processor: {processor}")
            processors.add(processor)
    return tuple(sorted(processors))


def _set_process_cpu_affinity(processors: tuple[int, ...]) -> None:
    if not processors:
        raise ValueError("CPU affinity must not be empty")
    if os.name == "nt":
        pointer_bits = ctypes.sizeof(ctypes.c_size_t) * 8
        if processors[-1] >= pointer_bits:
            raise ValueError(
                f"processor {processors[-1]} exceeds the {pointer_bits}-bit affinity mask"
            )
        mask = sum(1 << processor for processor in processors)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), ctypes.c_size_t(mask)):
            raise ctypes.WinError(ctypes.get_last_error())
        return
    setter = getattr(os, "sched_setaffinity", None)
    if setter is None:
        raise RuntimeError("CPU affinity control is unsupported on this platform")
    setter(0, set(processors))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("value must be positive")
    return parsed


def _benchmark_mode(
    operation: Callable[[], CompactStereoRenderResult],
    *,
    device: str,
    warmups: int,
    measured: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        _time_render(operation, device)
    timings: list[float] = []
    identities: set[tuple[str, str]] = set()
    for _ in range(measured):
        result = _time_render(operation, device)
        timings.append(result.seconds)
        identities.add(
            (
                _array_sha256(result.compact.left_image),
                _array_sha256(result.compact.right_image),
            )
        )
    return {
        "warmup_count": warmups,
        "measured_count": measured,
        "seconds": timings,
        "p95_seconds": _linear_p95(timings),
        "deterministic_output": len(identities) <= 1,
        "output_identities": [list(value) for value in sorted(identities)],
    }


def _run_benchmark(
    fixture_root: Path,
    renderer: StereoRenderer,
    config: VerificationConfig,
    device: str,
    warmups: int,
    measured: int,
) -> dict[str, Any]:
    if measured == 0:
        return {
            "status": "disabled",
            "performance_gate_enabled": config.performance_gate_enabled,
            "passed": not config.performance_gate_enabled,
        }
    loaded = _load_frame(fixture_root, config.frames[0], config)
    fast_settings = _settings(config, mode="fast", fill="background")
    quality_settings = _settings(config, mode="quality", fill="background")
    with torch.inference_mode():
        fast = _benchmark_mode(
            lambda: renderer.render_relative_fast_compact(
                loaded.source, loaded.canonical, fast_settings
            ),
            device=device,
            warmups=warmups,
            measured=measured,
        )
        quality = _benchmark_mode(
            lambda: render_relative_quality_compact(
                renderer=renderer,
                frame=loaded.source,
                canonical=loaded.canonical,
                settings=quality_settings,
            ),
            device=device,
            warmups=warmups,
            measured=measured,
        )
    ratio = quality["p95_seconds"] / fast["p95_seconds"] if fast["p95_seconds"] else None
    count_valid = warmups >= 5 and measured >= 30
    limits_pass = (
        quality["p95_seconds"] <= config.quality_renderer_p95_limit_seconds
        and ratio is not None
        and ratio <= config.quality_to_fast_p95_limit
    )
    passed = (
        count_valid
        and limits_pass
        and fast["deterministic_output"]
        and quality["deterministic_output"]
    )
    return {
        "status": "passed" if passed else "failed",
        "performance_gate_enabled": config.performance_gate_enabled,
        "fast": fast,
        "quality": quality,
        "quality_to_fast_p95_ratio": ratio,
        "limits": {
            "minimum_warmups": 5,
            "minimum_measured": 30,
            "quality_renderer_p95_seconds": config.quality_renderer_p95_limit_seconds,
            "quality_to_fast_p95_ratio": config.quality_to_fast_p95_limit,
        },
        "passed": passed or not config.performance_gate_enabled,
    }


def _load_unpinned_pair(
    fixture_root: Path, number: int
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    name = f"frame_{number:06d}.png"
    source_path = fixture_root / "00_original_frames" / name
    canonical_path = fixture_root / "03_disparity_maps" / name
    return (
        _read_colour(source_path),
        _read_canonical(canonical_path),
        {"source_sha256": _sha256(source_path), "canonical_sha256": _sha256(canonical_path)},
    )


def _temporal_window(
    fixture_root: Path,
    artifacts_dir: Path,
    renderer: StereoRenderer,
    frame: FixtureFrame,
    config: VerificationConfig,
    device: str,
    radius: int,
) -> dict[str, Any]:
    settings = _settings(config, mode="quality", fill="background")
    crops: dict[tuple[str, str], list[np.ndarray]] = {
        (region.name, eye): [] for region in frame.regions for eye in ("left", "right")
    }
    entries: list[dict[str, Any]] = []
    for number in range(frame.number - radius, frame.number + radius + 1):
        source, canonical, hashes = _load_unpinned_pair(fixture_root, number)
        if number == frame.number:
            left = _read_colour(artifacts_dir / "full" / f"{frame.stem}_quality_left.png")
            right = _read_colour(artifacts_dir / "full" / f"{frame.stem}_quality_right.png")
            seconds = None
        else:
            rendered = _time_render(
                lambda: render_relative_quality_compact(
                    renderer=renderer,
                    frame=source,
                    canonical=canonical,
                    settings=settings,
                ),
                device,
            )
            left = rendered.compact.left_image
            right = rendered.compact.right_image
            seconds = rendered.seconds
        for region in frame.regions:
            crops[(region.name, "left")].append(_crop_400(left, region))
            crops[(region.name, "right")].append(_crop_400(right, region))
        entries.append({"frame_name": f"frame_{number:06d}", **hashes, "render_seconds": seconds})
        del left, right
    artifacts: dict[str, str] = {}
    for (region_name, eye), sequence in crops.items():
        path = (
            artifacts_dir
            / "temporal"
            / f"{frame.stem}_{region_name}_quality_{eye}_window_400pct.png"
        ).resolve()
        _write_png(path, np.concatenate(sequence, axis=1))
        artifacts[f"{region_name}_{eye}"] = str(path)
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"radius": radius, "frames": entries, "artifacts": artifacts}


def _run_temporal_windows(
    fixture_root: Path,
    artifacts_dir: Path,
    renderer: StereoRenderer,
    config: VerificationConfig,
    device: str,
    radius: int,
) -> list[dict[str, Any]]:
    if radius == 0:
        return []
    return [
        {
            "target_frame": frame.stem,
            **_temporal_window(
                fixture_root,
                artifacts_dir,
                renderer,
                frame,
                config,
                device,
                radius,
            ),
        }
        for frame in config.frames
    ]


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if not status:
        return {"commit": commit, "dirty": False, "dirty_fingerprint": None}
    digest = hashlib.sha256(status)
    digest.update(
        subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    )
    return {"commit": commit, "dirty": True, "dirty_fingerprint": digest.hexdigest()}


def _physical_memory_bytes() -> int | None:
    if os.name != "nt":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.total_physical)


def _optional_psutil() -> Any | None:
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return None
    return psutil


def _physical_core_count() -> int | None:
    psutil = _optional_psutil()
    if psutil is None:
        return None
    count = psutil.cpu_count(logical=False)
    return int(count) if count is not None else None


def _process_cpu_affinity() -> list[int] | None:
    psutil = _optional_psutil()
    if psutil is not None:
        affinity = getattr(psutil.Process(), "cpu_affinity", None)
        if affinity is not None:
            return sorted(int(processor) for processor in affinity())
    getter = getattr(os, "sched_getaffinity", None)
    if getter is not None:
        return sorted(int(processor) for processor in getter(0))
    if os.name != "nt":
        return None
    process_mask = ctypes.c_size_t()
    system_mask = ctypes.c_size_t()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetProcessAffinityMask.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.GetProcessAffinityMask.restype = ctypes.c_int
    if not kernel32.GetProcessAffinityMask(
        kernel32.GetCurrentProcess(),
        ctypes.byref(process_mask),
        ctypes.byref(system_mask),
    ):
        return None
    return [
        processor
        for processor in range(ctypes.sizeof(ctypes.c_size_t) * 8)
        if process_mask.value & (1 << processor)
    ]


def _command_first_line(argv: list[str]) -> str | None:
    try:
        result = subprocess.run(argv, check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    output = result.stdout.strip() or result.stderr.strip()
    return output.splitlines()[0] if output else None


def _runtime_metadata(device: str) -> dict[str, Any]:
    gpu = torch.cuda.get_device_name() if device == "cuda" else None
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_model": os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor(),
        "physical_core_count": _physical_core_count(),
        "logical_core_count": os.cpu_count(),
        "process_cpu_affinity": _process_cpu_affinity(),
        "physical_memory_bytes": _physical_memory_bytes(),
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "torch_version": torch.__version__,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime_version": torch.version.cuda,
        "device": device,
        "gpu_model": gpu,
        "gpu_driver": _command_first_line(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ),
        "ffmpeg": _command_first_line(["ffmpeg", "-version"]),
    }


def _base_report(
    fixture_root: Path,
    artifacts_dir: Path,
    inputs: dict[str, Any],
    config: VerificationConfig,
    device: str,
    benchmark_command: tuple[str, ...] | None,
) -> dict[str, Any]:
    task0 = PROJECT_ROOT / "docs/superpowers/reports/2026-08-24-stereo-quality-task0.json"
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "algorithm_version": REPORT_ALGORITHM_VERSION,
        "stereo_stage_algorithm_version": STEREO_STAGE_ALGORITHM_VERSION,
        "source": _git_state(),
        "runtime": _runtime_metadata(device),
        "benchmark_commands": (
            [{"argv": list(benchmark_command), "cwd": str(Path.cwd().resolve())}]
            if benchmark_command is not None
            else []
        ),
        "fixture_root": str(fixture_root.resolve()),
        "artifacts_dir": str(artifacts_dir.resolve()),
        "inputs": inputs,
        "settings": {
            "render_shape": [config.render_height, config.render_width],
            "stereo_strength": config.stereo_strength,
            "convergence": config.convergence,
            "occlusion_fill": "background",
            "occlusion_fill_max_px": config.occlusion_fill_max_px,
            "stereo_render_modes": ["fast", "quality"],
            "diagnostic_modes": ["quality_without_fill", "nearest"],
        },
        "task0_report": {
            "path": str(task0.resolve()),
            "sha256": _sha256(task0) if task0.is_file() else None,
        },
    }


def _aggregate_gates(
    frame_checks: list[dict[str, bool]],
    benchmark: dict[str, Any],
    temporal_radius: int,
) -> dict[str, bool]:
    return {
        "fast_byte_compatibility": all(
            check["fast_left_pixels_equal"]
            and check["fast_right_pixels_equal"]
            and check["fast_left_png_equal"]
            and check["fast_right_png_equal"]
            for check in frame_checks
        ),
        "quality_final_unresolved_lanes_zero": all(
            check["quality_final_unresolved_pixels_zero"] for check in frame_checks
        ),
        "quality_without_fill_diagnostic_observed": all(
            check["quality_without_fill_exposes_geometry_holes"] for check in frame_checks
        ),
        "nearest_diagnostic_distinct": all(
            check["nearest_diagnostic_is_distinct"] for check in frame_checks
        ),
        "output_contract": all(
            check["all_output_shapes_and_dtypes_valid"] for check in frame_checks
        ),
        "performance_and_determinism": bool(benchmark["passed"]),
        "temporal_windows_exported": temporal_radius >= 0,
    }


def run_verification(  # noqa: C901
    fixture_root: Path,
    report_json: Path,
    artifacts_dir: Path,
    *,
    device: str = "auto",
    config: VerificationConfig = DEFAULT_CONFIG,
    benchmark_warmups: int = 5,
    benchmark_measured: int = 30,
    temporal_radius: int = 1,
    benchmark_command: tuple[str, ...] | None = None,
) -> int:
    """Run the release gate and always leave a machine-readable report."""

    if benchmark_warmups < 0 or benchmark_measured < 0 or temporal_radius < 0:
        raise ValueError("benchmark counts and temporal radius must be nonnegative")
    resolved_device = _resolve_device(device)
    inputs, available = _inspect_inputs(fixture_root, config)
    report = _base_report(
        fixture_root,
        artifacts_dir,
        inputs,
        config,
        resolved_device,
        benchmark_command,
    )
    if not available:
        report["status"] = "fixture_unavailable"
        _write_report(report_json, report)
        return 1

    try:
        renderer = StereoRenderer(device=resolved_device)
        frame_reports: list[dict[str, Any]] = []
        frame_checks: list[dict[str, bool]] = []
        for frame in config.frames:
            frame_report, checks = _evaluate_frame(
                fixture_root,
                artifacts_dir,
                renderer,
                frame,
                config,
                resolved_device,
            )
            frame_reports.append(frame_report)
            frame_checks.append(checks)
        benchmark = _run_benchmark(
            fixture_root,
            renderer,
            config,
            resolved_device,
            benchmark_warmups,
            benchmark_measured,
        )
        temporal = _run_temporal_windows(
            fixture_root,
            artifacts_dir,
            renderer,
            config,
            resolved_device,
            temporal_radius,
        )
        gates = _aggregate_gates(frame_checks, benchmark, temporal_radius)
    except Exception as error:  # The standalone gate must preserve failure evidence.
        report["status"] = "evaluation_error"
        report["error"] = f"{type(error).__name__}: {error}"
        _write_report(report_json, report)
        return 3

    report.update(
        {
            "frames": frame_reports,
            "benchmark": benchmark,
            "temporal_windows": temporal,
            "gates": gates,
            "human_review": {
                "agent_visual_review": "pending",
                "user_approval": "pending",
                "rejection_criteria": [
                    "background_stretch",
                    "soft_halo_wider_than_one_output_pixel",
                    "nearest_style_stair_step",
                    "missing_thin_structure",
                    "copied_foreground_streak",
                    "temporal_flicker",
                    "black_hole",
                    "cross_depth_donor_leakage",
                ],
            },
        }
    )
    if all(gates.values()):
        report["status"] = "machine_pass_human_review_required"
        exit_code = 0
    else:
        report["status"] = "machine_gate_failure"
        exit_code = 2
    _write_report(report_json, report)
    return exit_code


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", type=Path, default=DEFAULT_FIXTURE_ROOT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--benchmark-warmups", type=int, default=5)
    parser.add_argument("--benchmark-measured", type=int, default=30)
    parser.add_argument("--temporal-radius", type=int, default=1)
    parser.add_argument(
        "--cpu-affinity",
        type=_parse_cpu_affinity,
        help="Comma-separated logical processors or inclusive ranges (for example 0-15)",
    )
    parser.add_argument(
        "--torch-intraop-threads",
        type=_positive_int,
        help="Explicit Torch intra-op thread count for a reproducible reference run",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.cpu_affinity is not None:
        _set_process_cpu_affinity(args.cpu_affinity)
    if args.torch_intraop_threads is not None:
        torch.set_num_threads(args.torch_intraop_threads)
    exit_code = run_verification(
        args.fixture_root,
        args.report_json,
        args.artifacts_dir,
        device=args.device,
        benchmark_warmups=args.benchmark_warmups,
        benchmark_measured=args.benchmark_measured,
        temporal_radius=args.temporal_radius,
        benchmark_command=(
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ),
    )
    print(args.report_json.resolve())
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
