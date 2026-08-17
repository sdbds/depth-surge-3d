"""Build auditable fixed-corpus evidence for every pinned MoGe-2 variant.

This is an explicit, non-CI release command. Importing the module and requesting
``--help`` intentionally avoid optional model, media, CUDA, and snapshot work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar, cast

import numpy as np


TOOL_SCHEMA_VERSION = 1
MOGE_SOURCE_COMMIT = "925b8ed835a7a9cdb7578ba15c658a0afc969030"
ADAPTER_RESOLUTION_LEVEL = 9
CANONICAL_CLIP_IDS = ("indoor-near", "outdoor-far", "scene-cut")
RELEASE_VARIANT_PINS = (
    (
        "vits",
        "Ruicheng/moge-2-vits-normal",
        "679230677b4d282c6f304189a93e98e14f085902",
    ),
    (
        "vitb",
        "Ruicheng/moge-2-vitb-normal",
        "54ad3a693e61907ea4633d13dec6ee682fa09419",
    ),
    (
        "vitl",
        "Ruicheng/moge-2-vitl",
        "39c4d5e957afe587e04eec59dc2bcc3be5ecd968",
    ),
)
PROJECTION_SETTINGS: dict[str, Any] = {
    "virtual_baseline_mm": 63.0,
    "metric_convergence_distance": "auto",
    "max_disparity_percent": 2.0,
    "vr_format": "side_by_side",
    "apply_distortion": False,
    "crop_factor": 1.0,
    "occlusion_fill": "background",
    "stereo_io_workers": 1,
}
REPORT_SETTINGS: dict[str, Any] = {
    "geometry_modes": ["relative", "metric_camera"],
    "relative_stereo_strength": 2.0,
    "relative_convergence": 0.5,
    **PROJECTION_SETTINGS,
    "vr_resolution": "auto-per-source",
    "upscale_model": "none",
    "preserve_audio": False,
}
EXPECTED_FIXED_MEMBERS = (
    "depth.npy",
    "valid.npy",
    "focal_x_normalized.npy",
)
_SHA256_PATTERN = __import__("re").compile(r"[0-9a-f]{64}", flags=__import__("re").ASCII)
_SAR_PATTERN = __import__("re").compile(r"([0-9]+):([0-9]+)", flags=__import__("re").ASCII)
_SAR_COMPONENT_MAX = 2_147_483_647


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--depth-resolution", type=_positive_int, default=1080)
    return parser


@dataclass(frozen=True)
class FixedImageInput:
    path: Path
    sha256: str
    width: int
    height: int


@dataclass(frozen=True)
class ClipInput:
    clip_id: str
    path: Path
    sha256: str
    static_roi_xywh: tuple[int, int, int, int]
    width: int
    height: int
    fps: float
    frame_count: int
    sar_numerator: int
    sar_denominator: int


@dataclass(frozen=True)
class CorpusConfig:
    config_path: Path
    fixed_image: FixedImageInput
    clips: tuple[ClipInput, ...]


@dataclass(frozen=True)
class FixedDepth:
    depth: np.ndarray
    valid: np.ndarray
    focal_x_normalized: np.float32


@dataclass(frozen=True)
class RawClip:
    directory: Path
    depth: np.ndarray
    valid: np.ndarray
    focal_x_normalized: np.ndarray
    frame_names: tuple[str, ...]
    inference_calls: int
    inferred_frame_count: int
    inference_seconds: float | None = None


@dataclass(frozen=True)
class ClipRender:
    mode: Literal["relative", "metric_camera"]
    output_path: Path
    hole_mask: np.ndarray
    total_disparity_pixels: np.ndarray | None = None
    retained_source_xyxy: tuple[int, int, int, int] | None = None
    clamp_sidecars: tuple[Path, ...] = ()
    disparity_valid_mask: np.ndarray | None = None
    output_shape: tuple[int, int] | None = None


class ReleaseSession(Protocol):
    def load(self) -> None: ...

    def infer_fixed(self, path: Path, depth_resolution: int) -> FixedDepth: ...

    def infer_clip(self, clip: ClipInput, depth_resolution: int, workspace: Path) -> RawClip: ...

    def render_clip(
        self,
        clip: ClipInput,
        raw: RawClip,
        mode: str,
        output_path: Path,
        settings: dict[str, Any],
    ) -> ClipRender: ...

    def unload(self) -> None: ...


class CudaProbe(Protocol):
    def synchronize(self) -> None: ...

    def reset_peak_memory_stats(self) -> None: ...

    def max_memory_allocated(self) -> int: ...


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_tree(directory: Path) -> str:
    if not directory.is_dir():
        raise ValueError(f"raw-stage directory is missing: {directory}")
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"raw-stage directory is empty: {directory}")
    hasher = hashlib.sha256()
    for path in files:
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        hasher.update(len(relative).to_bytes(8, "big"))
        hasher.update(relative)
        hasher.update(bytes.fromhex(_hash_file(path)))
    return hasher.hexdigest()


def _require_exact_keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} must have exact {label} keys: {sorted(expected)}")
    return cast(dict[str, Any], value)


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a 64-character lowercase SHA-256")
    return value


def _resolve_verified_file(config_dir: Path, value: object, expected_hash: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("corpus path must be a nonempty string")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = config_dir / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise ValueError(f"corpus input must be an existing regular file: {candidate}")
    actual = _hash_file(candidate)
    if actual != expected_hash:
        raise ValueError(
            f"corpus checksum mismatch for {candidate}: expected {expected_hash}, got {actual}"
        )
    return candidate


def _default_image_probe(path: Path) -> tuple[int, int]:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim < 2:
        raise ValueError(f"fixed image is unreadable: {path}")
    return int(image.shape[1]), int(image.shape[0])


def _default_video_probe(path: Path) -> dict[str, Any]:
    _ensure_project_import_path()
    from depth_surge_3d.io.operations import get_video_properties

    return get_video_properties(str(path))


def _parse_square_sar(properties: Mapping[str, Any]) -> tuple[int, int]:
    raw = properties.get("sample_aspect_ratio")
    numeric_numerator = properties.get("sample_aspect_ratio_numerator")
    numeric_denominator = properties.get("sample_aspect_ratio_denominator")
    if raw is None and numeric_numerator is not None and numeric_denominator is not None:
        numerator, denominator = numeric_numerator, numeric_denominator
        if not isinstance(numerator, int) or not isinstance(denominator, int):
            raise ValueError("sample_aspect_ratio must be an unsigned numerator:denominator")
    elif raw is None or raw == "N/A":
        numerator, denominator = 1, 1
    elif isinstance(raw, str):
        match = _SAR_PATTERN.fullmatch(raw)
        if match is None:
            raise ValueError("sample_aspect_ratio must be an unsigned numerator:denominator")
        numerator, denominator = (int(part) for part in match.groups())
    else:
        numerator = properties.get("sample_aspect_ratio_numerator")
        denominator = properties.get("sample_aspect_ratio_denominator")
        if not isinstance(numerator, int) or not isinstance(denominator, int):
            raise ValueError("sample_aspect_ratio must be an unsigned numerator:denominator")
    if not 1 <= numerator <= _SAR_COMPONENT_MAX or not 1 <= denominator <= _SAR_COMPONENT_MAX:
        raise ValueError("sample_aspect_ratio components must be in 1..2147483647")
    divisor = math.gcd(numerator, denominator)
    reduced = numerator // divisor, denominator // divisor
    if reduced != (1, 1):
        raise ValueError("release clips require explicit square-pixel sample_aspect_ratio=1:1")
    return reduced


def _probe_dimension(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def load_corpus_config(  # noqa: C901
    config_path: Path,
    *,
    image_probe: Callable[[Path], tuple[int, int]] | None = None,
    video_probe: Callable[[Path], dict[str, Any]] | None = None,
) -> CorpusConfig:
    """Load, authenticate, probe, and freeze the exact three-clip corpus."""

    config_path = Path(config_path).expanduser().resolve()
    try:
        root = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"corpus JSON is missing, unreadable, or malformed: {config_path}"
        ) from error
    if not isinstance(root, dict):
        raise ValueError("corpus JSON root must be an object")
    root = _require_exact_keys(root, {"fixed_image", "clips"}, "top-level")
    fixed_payload = _require_exact_keys(root["fixed_image"], {"path", "sha256"}, "fixed_image")
    fixed_hash = _require_hash(fixed_payload["sha256"], "fixed_image.sha256")
    fixed_path = _resolve_verified_file(config_path.parent, fixed_payload["path"], fixed_hash)
    width, height = (image_probe or _default_image_probe)(fixed_path)
    width = _probe_dimension(width, "fixed image width")
    height = _probe_dimension(height, "fixed image height")

    clips_payload = root["clips"]
    if not isinstance(clips_payload, list) or len(clips_payload) != 3:
        raise ValueError("corpus must contain exactly three clips")
    ids = [item.get("id") if isinstance(item, dict) else None for item in clips_payload]
    if ids != list(CANONICAL_CLIP_IDS):
        raise ValueError(f"clips must use canonical order and IDs: {list(CANONICAL_CLIP_IDS)}")

    probe = video_probe or _default_video_probe
    clips: list[ClipInput] = []
    for item in clips_payload:
        clip_payload = _require_exact_keys(
            item,
            {"id", "path", "sha256", "static_roi_xywh"},
            "clip",
        )
        clip_id = cast(str, clip_payload["id"])
        expected_hash = _require_hash(clip_payload["sha256"], f"{clip_id}.sha256")
        clip_path = _resolve_verified_file(config_path.parent, clip_payload["path"], expected_hash)
        properties = probe(clip_path)
        if not isinstance(properties, dict) or not properties:
            raise ValueError(f"video properties are missing for {clip_path}")
        clip_width = _probe_dimension(properties.get("width"), f"{clip_id} width")
        clip_height = _probe_dimension(properties.get("height"), f"{clip_id} height")
        frame_count = _probe_dimension(properties.get("frame_count"), f"{clip_id} frame_count")
        fps = properties.get("fps")
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps):
            raise ValueError(f"{clip_id} fps must be finite and positive")
        if fps <= 0:
            raise ValueError(f"{clip_id} fps must be finite and positive")
        roi_value = clip_payload["static_roi_xywh"]
        if (
            not isinstance(roi_value, list)
            or len(roi_value) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in roi_value)
        ):
            raise ValueError(f"{clip_id} ROI values must be four non-boolean integers")
        x, y, roi_width, roi_height = cast(list[int], roi_value)
        if x < 0 or y < 0:
            raise ValueError(f"{clip_id} ROI x and y must be nonnegative")
        if roi_width <= 0 or roi_height <= 0:
            raise ValueError(f"{clip_id} ROI width and height must be positive")
        if x + roi_width > clip_width or y + roi_height > clip_height:
            raise ValueError(f"{clip_id} ROI exceeds video bounds")
        sar_numerator, sar_denominator = _parse_square_sar(properties)
        clips.append(
            ClipInput(
                clip_id=clip_id,
                path=clip_path,
                sha256=expected_hash,
                static_roi_xywh=(x, y, roi_width, roi_height),
                width=clip_width,
                height=clip_height,
                fps=float(fps),
                frame_count=frame_count,
                sar_numerator=sar_numerator,
                sar_denominator=sar_denominator,
            )
        )
    return CorpusConfig(
        config_path=config_path,
        fixed_image=FixedImageInput(fixed_path, fixed_hash, width, height),
        clips=tuple(clips),
    )


def release_variants(
    registry_getter: Callable[[str], Any] | None = None,
) -> tuple[tuple[str, str, str], ...]:
    """Return the exact release order only when the production registry agrees."""

    if registry_getter is None:
        _ensure_project_import_path()
        from depth_surge_3d.inference.depth.backend_registry import get_backend_spec

        registry_getter = get_backend_spec
    spec = registry_getter("moge2")
    observed = tuple(
        (str(name), str(variant.repo_id), str(variant.revision))
        for name, variant in spec.variants.items()
    )
    if observed != RELEASE_VARIANT_PINS:
        raise RuntimeError(
            f"MoGe release registry drift: expected {RELEASE_VARIANT_PINS!r}, got {observed!r}"
        )
    return observed


def _validate_fixed_depth(result: FixedDepth) -> tuple[np.ndarray, np.ndarray, np.float32]:
    if not isinstance(result, FixedDepth):
        raise TypeError("fixed inference must return FixedDepth")
    depth = result.depth
    valid = result.valid
    focal = result.focal_x_normalized
    if not isinstance(depth, np.ndarray) or depth.dtype != np.float32 or depth.ndim != 2:
        raise TypeError("fixed depth must be a float32 [H,W] array")
    if not isinstance(valid, np.ndarray) or valid.dtype != np.bool_ or valid.shape != depth.shape:
        raise TypeError("fixed valid mask must be Boolean and match depth")
    if not isinstance(focal, np.float32) or not np.isfinite(focal) or focal <= 0:
        raise ValueError("fixed focal must be a positive finite float32 scalar")
    metric_valid = valid & np.isfinite(depth) & (depth > 0)
    if not np.any(metric_valid):
        raise ValueError("fixed image must contain at least one valid metric-depth pixel")
    if np.any(valid & ~metric_valid):
        raise ValueError("fixed valid metric-depth pixels must be finite and positive")
    sanitized = np.zeros(depth.shape, dtype=np.float32)
    sanitized[valid] = depth[valid]
    return sanitized, np.array(valid, copy=True), focal


def write_fixed_image_artifact(path: Path, result: FixedDepth) -> None:
    depth, valid, focal = _validate_fixed_depth(result)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("wb") as handle:
        np.savez_compressed(
            handle,
            depth=depth,
            valid=valid,
            focal_x_normalized=np.asarray(focal, dtype=np.float32),
        )


def validate_fixed_image_artifact(path: Path) -> dict[str, Any]:  # noqa: C901
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as archive:
            if tuple(archive.namelist()) != EXPECTED_FIXED_MEMBERS:
                raise ValueError(
                    f"fixed image NPZ must contain exact members {EXPECTED_FIXED_MEMBERS}"
                )
        with np.load(path, allow_pickle=False) as payload:
            depth = np.asarray(payload["depth"])
            valid = np.asarray(payload["valid"])
            focal_array = np.asarray(payload["focal_x_normalized"])
    except ValueError as error:
        if "Object arrays cannot be loaded" in str(error):
            raise ValueError("fixed image NPZ arrays must use non-object dtypes") from error
        raise
    except Exception as error:
        raise ValueError(f"fixed image NPZ is unreadable: {path}") from error
    if depth.dtype != np.float32:
        raise ValueError("fixed image depth dtype must be float32")
    if valid.dtype != np.bool_:
        raise ValueError("fixed image valid dtype must be Boolean")
    if focal_array.dtype != np.float32:
        raise ValueError("fixed image focal dtype must be float32")
    if depth.ndim != 2 or depth.shape != valid.shape or not depth.size:
        raise ValueError("fixed image depth/valid shapes must be matching nonempty [H,W]")
    if focal_array.shape != ():
        raise ValueError("fixed image focal must be zero-dimensional")
    focal = float(focal_array.item())
    if not math.isfinite(focal) or focal <= 0:
        raise ValueError("fixed image focal must be positive and finite")
    if not np.isfinite(depth).all() or np.any(depth < 0) or np.any(depth[~valid] != 0):
        raise ValueError("fixed image depth must be finite, nonnegative, and zero where invalid")
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0 or np.any(depth[valid] <= 0):
        raise ValueError("fixed image must contain positive valid metric depth")
    return {
        "native_shape": [int(depth.shape[0]), int(depth.shape[1])],
        "focal_x_normalized": focal,
        "valid_metric_pixels": valid_count,
    }


def map_source_roi(
    roi_xywh: tuple[int, int, int, int],
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    retained_source_xyxy: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Map source half-open ROI bounds to target integer bounds deterministically.

    Mapping first intersects the ROI with the retained source rectangle. Starts
    use floor and ends use ceil, then clamp to the target raster.
    """

    source_height, source_width = source_shape
    target_height, target_width = target_shape
    if min(source_height, source_width, target_height, target_width) <= 0:
        raise ValueError("source and target shapes must be positive")
    x, y, width, height = roi_xywh
    retained_x0, retained_y0, retained_x1, retained_y1 = retained_source_xyxy
    if not (0 <= retained_x0 < retained_x1 <= source_width):
        raise ValueError("retained source horizontal bounds are invalid")
    if not (0 <= retained_y0 < retained_y1 <= source_height):
        raise ValueError("retained source vertical bounds are invalid")
    intersect_x0 = max(x, retained_x0)
    intersect_y0 = max(y, retained_y0)
    intersect_x1 = min(x + width, retained_x1)
    intersect_y1 = min(y + height, retained_y1)
    if intersect_x0 >= intersect_x1 or intersect_y0 >= intersect_y1:
        raise ValueError("ROI has no retained samples after mapping/crop")
    retained_width = retained_x1 - retained_x0
    retained_height = retained_y1 - retained_y0
    x0 = math.floor((intersect_x0 - retained_x0) * target_width / retained_width)
    y0 = math.floor((intersect_y0 - retained_y0) * target_height / retained_height)
    x1 = math.ceil((intersect_x1 - retained_x0) * target_width / retained_width)
    y1 = math.ceil((intersect_y1 - retained_y0) * target_height / retained_height)
    mapped = (
        max(0, min(x0, target_width)),
        max(0, min(y0, target_height)),
        max(0, min(x1, target_width)),
        max(0, min(y1, target_height)),
    )
    if mapped[0] >= mapped[2] or mapped[1] >= mapped[3]:
        raise ValueError("ROI has no retained samples after integer mapping")
    return mapped


def _require_finite_number(value: object, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{label} must be finite" + (f" and >= {minimum}" if minimum else ""))
    return result


def _validate_raw_clip(raw: RawClip) -> None:
    if not isinstance(raw, RawClip):
        raise TypeError("clip inference must return RawClip")
    if raw.inference_calls != raw.inferred_frame_count:
        raise ValueError("MoGe inference call count must equal the source frame count")
    if raw.inferred_frame_count != len(raw.frame_names):
        raise ValueError("inferred frame count must match the raw frame manifest")
    if raw.depth.dtype != np.float32 or raw.depth.ndim != 3:
        raise TypeError("raw metric depth must be float32 [N,H,W]")
    if raw.valid.dtype != np.bool_ or raw.valid.shape != raw.depth.shape:
        raise TypeError("raw valid mask must be Boolean and match metric depth")
    if (
        raw.focal_x_normalized.dtype != np.float32
        or raw.focal_x_normalized.shape != (len(raw.depth),)
        or not np.isfinite(raw.focal_x_normalized).all()
        or np.any(raw.focal_x_normalized <= 0)
    ):
        raise ValueError("raw focal values must be positive finite float32 [N]")
    if len(raw.frame_names) != len(raw.depth) or len(set(raw.frame_names)) != len(raw.frame_names):
        raise ValueError("raw frame names must be unique and match depth frames")
    if np.any(raw.valid & (~np.isfinite(raw.depth) | (raw.depth <= 0))):
        raise ValueError("valid raw metric depth must be finite and positive")


def _validate_raw_stage(raw: RawClip) -> None:
    metadata_path = raw.directory / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("production raw stage metadata is missing or malformed") from error
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 3:
        raise ValueError("production raw stage must use schema v3")
    required = {
        "storage_status": "ready",
        "representation": "metric_depth",
        "camera_model": "pinhole_fx",
    }
    for field, expected in required.items():
        if metadata.get(field) != expected:
            raise ValueError(f"production raw stage {field} must be {expected}")
    names = metadata.get("frame_names")
    if (
        not isinstance(names, list)
        or tuple(Path(name).stem for name in names if isinstance(name, str)) != raw.frame_names
        or len(names) != len(raw.frame_names)
    ):
        raise ValueError("production raw stage frame manifest does not match inference")
    if metadata.get("completed_count") != len(raw.frame_names):
        raise ValueError("production raw stage is incomplete")
    expected_payloads = {f"{name}.npz" for name in raw.frame_names}
    actual_payloads = {path.name for path in raw.directory.glob("*.npz")}
    if actual_payloads != expected_payloads:
        raise ValueError("production raw stage payload manifest is incomplete or unexpected")


def _hole_fraction(render: ClipRender, expected_frames: int) -> float:
    holes = render.hole_mask
    if not isinstance(holes, np.ndarray) or holes.dtype != np.bool_ or holes.ndim != 3:
        raise TypeError(f"{render.mode} hole mask must be a Boolean [N,H,W] array")
    if holes.shape[0] != expected_frames or holes.size == 0:
        raise ValueError(f"{render.mode} hole mask frame count/shape is invalid")
    return float(np.count_nonzero(holes) / holes.size)


def _validate_output_shape(render: ClipRender) -> tuple[int, int]:
    shape = render.output_shape
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape
        )
    ):
        raise ValueError(f"{render.mode} final output shape must contain positive height/width")
    return shape


def _read_clamp_fractions(paths: Sequence[Path], frame_names: Sequence[str]) -> list[float]:
    if len(paths) != len(frame_names):
        raise ValueError("metric clamp sidecar count must match frames")
    fractions: list[float] = []
    expected_keys = {
        "schema_version",
        "frame_name",
        "valid_pixel_count",
        "clamped_pixel_count",
        "clamped_fraction",
    }
    for path, frame_name in zip(paths, frame_names):
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"metric clamp sidecar is unreadable: {path}") from error
        if not isinstance(payload, dict) or set(payload) != expected_keys:
            raise ValueError(f"metric clamp sidecar schema is invalid: {path}")
        if payload["schema_version"] != 1 or payload["frame_name"] != frame_name:
            raise ValueError(f"metric clamp sidecar identity is invalid: {path}")
        valid_count = payload["valid_pixel_count"]
        clamped_count = payload["clamped_pixel_count"]
        if (
            isinstance(valid_count, bool)
            or not isinstance(valid_count, int)
            or valid_count < 0
            or isinstance(clamped_count, bool)
            or not isinstance(clamped_count, int)
            or not 0 <= clamped_count <= valid_count
        ):
            raise ValueError(f"metric clamp sidecar counts are invalid: {path}")
        fraction = _require_finite_number(payload["clamped_fraction"], "clamped fraction")
        expected = clamped_count / valid_count if valid_count else 0.0
        if not 0.0 <= fraction <= 1.0 or fraction != expected:
            raise ValueError(f"metric clamp sidecar fraction is inconsistent: {path}")
        fractions.append(fraction)
    return fractions


def _population_stddev(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("structural measurement sample set must be nonempty")
    result = float(statistics.pstdev(values))
    if not math.isfinite(result):
        raise ValueError("structural measurement standard deviation must be finite")
    return result


def compute_clip_measurements(
    raw: RawClip,
    relative: ClipRender,
    metric: ClipRender,
    *,
    source_shape: tuple[int, int],
    source_roi_xywh: tuple[int, int, int, int],
    inference_seconds: float,
) -> dict[str, Any]:
    _validate_raw_clip(raw)
    seconds = _require_finite_number(inference_seconds, "inference seconds", minimum=0.0)
    if relative.mode != "relative" or metric.mode != "metric_camera":
        raise ValueError("clip renders must be in relative, metric_camera order")
    focal_values = [float(value) for value in raw.focal_x_normalized]
    full_source = (0, 0, source_shape[1], source_shape[0])
    raw_roi = map_source_roi(
        source_roi_xywh,
        source_shape,
        (raw.depth.shape[1], raw.depth.shape[2]),
        full_source,
    )
    x0, y0, x1, y1 = raw_roi
    depth_means: list[float] = []
    for frame_depth, frame_valid in zip(raw.depth, raw.valid):
        depth_roi = frame_depth[y0:y1, x0:x1]
        valid_roi = frame_valid[y0:y1, x0:x1] & np.isfinite(depth_roi) & (depth_roi > 0)
        samples = depth_roi[valid_roi]
        if samples.size == 0:
            raise ValueError("ROI has no valid metric-depth samples after mapping")
        mean = float(np.mean(samples, dtype=np.float64))
        depth_means.append(_require_finite_number(mean, "ROI metric depth mean"))

    disparity = metric.total_disparity_pixels
    retained = metric.retained_source_xyxy
    if (
        not isinstance(disparity, np.ndarray)
        or disparity.dtype != np.float64
        or disparity.ndim != 3
        or disparity.shape[0] != len(raw.frame_names)
        or not np.isfinite(disparity).all()
    ):
        raise ValueError("metric final-coordinate disparity must be finite float64 [N,H,W]")
    if retained is None:
        raise ValueError("metric render must record retained source bounds")
    disparity_valid = metric.disparity_valid_mask
    if (
        not isinstance(disparity_valid, np.ndarray)
        or disparity_valid.dtype != np.bool_
        or disparity_valid.shape != disparity.shape
    ):
        raise TypeError("metric disparity valid mask must be Boolean and match disparity")
    disparity_roi = map_source_roi(
        source_roi_xywh,
        source_shape,
        (disparity.shape[1], disparity.shape[2]),
        retained,
    )
    dx0, dy0, dx1, dy1 = disparity_roi
    disparity_means: list[float] = []
    for frame, frame_valid in zip(disparity, disparity_valid):
        frame_roi = frame[dy0:dy1, dx0:dx1]
        valid_roi = frame_valid[dy0:dy1, dx0:dx1]
        samples = frame_roi[valid_roi]
        if samples.size == 0:
            raise ValueError("ROI has no valid output-disparity samples after mapping/crop")
        disparity_means.append(
            _require_finite_number(
                float(np.mean(samples, dtype=np.float64)),
                "ROI output disparity mean",
            )
        )
    if not disparity_means:
        raise ValueError("ROI has no output-disparity samples after mapping")
    relative_holes = _hole_fraction(relative, len(raw.frame_names))
    metric_holes = _hole_fraction(metric, len(raw.frame_names))
    clamp_fractions = _read_clamp_fractions(metric.clamp_sidecars, raw.frame_names)
    return {
        "inference_seconds_per_frame": seconds / len(raw.frame_names),
        "focal_min": min(focal_values),
        "focal_max": max(focal_values),
        "focal_stddev": _population_stddev(focal_values),
        "roi_metric_depth_mean_per_frame": depth_means,
        "roi_metric_depth_stddev": _population_stddev(depth_means),
        "roi_output_disparity_mean_per_frame": disparity_means,
        "roi_output_disparity_stddev": _population_stddev(disparity_means),
        "relative_hole_fraction": relative_holes,
        "metric_hole_fraction": metric_holes,
        "metric_clamped_fraction_per_frame": clamp_fractions,
    }


class AtomicPublisher:
    """Publish one complete file through a unique same-directory temporary."""

    def __init__(
        self,
        *,
        replace_fn: Callable[[Path, Path], None] = os.replace,
        token_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        self._replace = replace_fn
        self._token_factory = token_factory

    def temporary_path(self, destination: Path) -> Path:
        destination = Path(destination)
        token = self._token_factory()
        return destination.with_name(f".{destination.stem}.{token}.tmp{destination.suffix}")

    def commit_temporary(self, temporary: Path, destination: Path) -> None:
        temporary = Path(temporary)
        destination = Path(destination)
        if temporary.parent != destination.parent or temporary == destination:
            raise ValueError("atomic temporary must be a unique sibling of its destination")
        if not temporary.is_file():
            raise ValueError(f"atomic temporary is missing: {temporary}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def write_bytes(self, destination: Path, payload: bytes) -> None:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.temporary_path(destination)
        try:
            temporary.write_bytes(payload)
            self.commit_temporary(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def write_text(self, destination: Path, payload: str) -> None:
        self.write_bytes(Path(destination), payload.encode("utf-8"))

    def write_json(self, destination: Path, payload: Mapping[str, Any]) -> None:
        _validate_json_numbers(payload)
        self.write_text(
            Path(destination),
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )


def _validate_json_numbers(value: object) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _validate_json_numbers(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_json_numbers(item)
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError("reports must not contain non-finite JSON numbers")


class _TorchCudaProbe:
    def __init__(self, device: str) -> None:
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA release gate requested but CUDA is unavailable")
        self._torch = torch

    def synchronize(self) -> None:
        self._torch.cuda.synchronize()

    def reset_peak_memory_stats(self) -> None:
        self._torch.cuda.reset_peak_memory_stats()

    def max_memory_allocated(self) -> int:
        return int(self._torch.cuda.max_memory_allocated())


def _default_system_probe(device: str) -> dict[str, Any]:
    import torch

    gpu: str | None = None
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA release gate requested but CUDA is unavailable")
        gpu = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": gpu,
    }


def _default_git_probe() -> tuple[str, bool]:
    project_root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, bool(status)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class ReleaseDependencies:
    session_factory: Callable[[str, str, str, str], ReleaseSession]
    perf_counter: Callable[[], float] = time.perf_counter
    utc_now: Callable[[], str] = _utc_now
    system_probe: Callable[[str], dict[str, Any]] = _default_system_probe
    git_probe: Callable[[], tuple[str, bool]] = _default_git_probe
    cuda: CudaProbe | None = None
    publisher: AtomicPublisher = AtomicPublisher()


class ReleaseRunFailed(RuntimeError):
    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


def _input_report(corpus: CorpusConfig) -> dict[str, Any]:
    return {
        "corpus_config": str(corpus.config_path),
        "fixed_image": {
            "path": str(corpus.fixed_image.path),
            "sha256": corpus.fixed_image.sha256,
            "width": corpus.fixed_image.width,
            "height": corpus.fixed_image.height,
        },
        "clips": [
            {
                "id": clip.clip_id,
                "path": str(clip.path),
                "sha256": clip.sha256,
                "static_roi_xywh": list(clip.static_roi_xywh),
                "width": clip.width,
                "height": clip.height,
                "fps": clip.fps,
                "frame_count": clip.frame_count,
                "sample_aspect_ratio": f"{clip.sar_numerator}:{clip.sar_denominator}",
            }
            for clip in corpus.clips
        ],
    }


def _expected_evidence_paths() -> set[Path]:
    expected: set[Path] = set()
    for model_size, _repository, _revision in RELEASE_VARIANT_PINS:
        expected.add(Path(model_size) / "fixed-image-depth.npz")
        for clip_id in CANONICAL_CLIP_IDS:
            expected.add(Path(model_size) / f"{clip_id}-relative.mp4")
            expected.add(Path(model_size) / f"{clip_id}-metric-camera.mp4")
    return expected


def _output_record(output_dir: Path, path: Path) -> dict[str, str]:
    resolved_output = output_dir.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_output)
    except ValueError as error:
        raise ValueError(f"evidence output escaped output directory: {path}") from error
    return {"path": relative.as_posix(), "sha256": _hash_file(path)}


def _recorded_outputs(report: Mapping[str, Any]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for variant in cast(list[dict[str, Any]], report.get("variants", [])):
        fixed = variant.get("fixed_image_output")
        if isinstance(fixed, dict):
            records.append(cast(dict[str, str], fixed))
        for clip in cast(list[dict[str, Any]], variant.get("clips", [])):
            for field in ("relative_output", "metric_output"):
                record = clip.get(field)
                if isinstance(record, dict):
                    records.append(cast(dict[str, str], record))
    return records


def _validate_complete_evidence(report: Mapping[str, Any], output_dir: Path) -> None:
    variants = report.get("variants")
    if not isinstance(variants, list) or [item.get("model_size") for item in variants] != [
        pin[0] for pin in RELEASE_VARIANT_PINS
    ]:
        raise ValueError("complete evidence is missing the canonical variant order")
    if any(
        not isinstance(item.get("clips"), list)
        or [clip.get("id") for clip in item["clips"]] != list(CANONICAL_CLIP_IDS)
        for item in variants
    ):
        raise ValueError("complete evidence is missing the canonical clip order")
    records = _recorded_outputs(report)
    expected = _expected_evidence_paths()
    observed = {Path(record["path"]) for record in records}
    if len(records) != 21 or observed != expected:
        raise ValueError("complete evidence must record exactly all 21 media/NPZ outputs")
    actual_candidates = {
        path.relative_to(output_dir)
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".mp4", ".npz"}
    }
    if actual_candidates != expected:
        raise ValueError("evidence directory contains missing or unrecorded media/NPZ output")
    for record in records:
        path = output_dir / record["path"]
        if not path.is_file() or _hash_file(path) != record["sha256"]:
            raise ValueError(f"recorded hash mismatch: {record['path']}")
        if path.suffix == ".npz":
            validate_fixed_image_artifact(path)
    if report.get("failures") != []:
        raise ValueError("complete evidence cannot contain failures")


def _format_output(record: Mapping[str, str]) -> str:
    return f"`{record['path']}` (`{record['sha256']}`)"


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render the JSON evidence identities and observations without thresholds."""

    lines = [
        "# MoGe-2 Three-Variant Release Evidence",
        "",
        f"Status: `{report.get('status', 'incomplete')}`",
        "",
        "## Identity",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Tool schema | `{report['tool_schema_version']}` |",
        f"| UTC timestamp | `{report['timestamp_utc']}` |",
        f"| Git commit | `{report['project_git']['commit']}` |",
        f"| Git dirty | `{str(report['project_git']['dirty']).lower()}` |",
        f"| OS | `{report['system']['os']}` |",
        f"| Python | `{report['system']['python']}` |",
        f"| PyTorch | `{report['system']['pytorch']}` |",
        f"| CUDA | `{report['system']['cuda']}` |",
        f"| GPU | `{report['system']['gpu']}` |",
        f"| MoGe source commit | `{report['moge_source_commit']}` |",
        f"| Adapter resolution level | `{report['adapter_resolution_level']}` |",
        f"| Requested depth resolution | `{report['requested_depth_resolution']}` |",
        "",
        "## Inputs",
        "",
        "| Kind/ID | Path | SHA-256 | Dimensions | SAR | ROI |",
        "|---|---|---|---|---|---|",
    ]
    fixed = report["inputs"]["fixed_image"]
    lines.append(
        f"| fixed image | `{fixed['path']}` | `{fixed['sha256']}` | "
        f"{fixed['width']}x{fixed['height']} | n/a | n/a |"
    )
    for clip in report["inputs"]["clips"]:
        lines.append(
            f"| {clip['id']} | `{clip['path']}` | `{clip['sha256']}` | "
            f"{clip['width']}x{clip['height']} | {clip['sample_aspect_ratio']} | "
            f"`{clip['static_roi_xywh']}` |"
        )
    lines.extend(
        [
            "",
            "## Active Settings",
            "",
            "| Setting | Value |",
            "|---|---|",
        ]
    )
    for key, value in report["settings"].items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(["", "## Measurements and Outputs", ""])
    for variant in report["variants"]:
        lines.extend(
            [
                f"### {variant['model_size']}",
                "",
                f"Repository/revision: `{variant['repository']}@{variant['revision']}`  ",
                f"Load seconds: `{variant['load_seconds']}`  ",
                f"Peak VRAM bytes: `{variant['peak_vram_bytes']}`  ",
                f"Fixed native shape: `{variant['fixed_image_native_shape']}`  ",
                f"Fixed focal: `{variant['fixed_image_focal_x_normalized']}`  ",
                f"Fixed valid metric pixels: `{variant['fixed_image_valid_metric_pixels']}`  ",
                f"Fixed output: {_format_output(variant['fixed_image_output'])}",
                "",
                "| Clip | Output HxW | sec/frame | focal min/max/stddev | ROI depth means/stddev | "
                "ROI disparity means/stddev | holes relative/metric | clamp fractions | Outputs |",
                "|---|---|---:|---|---|---|---|---|---|",
            ]
        )
        for clip in variant["clips"]:
            outputs = (
                f"relative {_format_output(clip['relative_output'])}; "
                f"metric {_format_output(clip['metric_output'])}"
            )
            lines.append(
                f"| {clip['id']} | {clip['output_shape']} | "
                f"{clip['inference_seconds_per_frame']} | "
                f"{clip['focal_min']} / {clip['focal_max']} / {clip['focal_stddev']} | "
                f"{clip['roi_metric_depth_mean_per_frame']} / "
                f"{clip['roi_metric_depth_stddev']} | "
                f"{clip['roi_output_disparity_mean_per_frame']} / "
                f"{clip['roi_output_disparity_stddev']} | "
                f"{clip['relative_hole_fraction']} / {clip['metric_hole_fraction']} | "
                f"{clip['metric_clamped_fraction_per_frame']} | {outputs} |"
            )
    lines.extend(["", "## Failures", ""])
    if report["failures"]:
        lines.extend(["| Variant | Clip | Stage | Type | Message |", "|---|---|---|---|---|"])
        for failure in report["failures"]:
            lines.append(
                f"| {failure['variant'] or ''} | {failure['clip'] or ''} | "
                f"{failure['stage']} | {failure['error_type']} | {failure['message']} |"
            )
    else:
        lines.append("None recorded.")
    lines.extend(
        [
            "",
            "## Human Inspection",
            "",
            "| Sign-off | Observation | Notes |",
            "|---|---|---|",
            "| [ ] | edge tearing | |",
            "| [ ] | foreground sign | |",
            "| [ ] | scale pumping | |",
            "| [ ] | focal breathing | |",
            "| [ ] | convergence placement | |",
            "| [ ] | viewing discomfort | |",
            "",
            "These unit metrics are structural observations, not portable thresholds. They do not "
            "establish physical calibration, better quality, comfort, or temporal stability.",
            "",
        ]
    )
    return "\n".join(lines)


T = TypeVar("T")


class ReleaseRunner:
    def __init__(self, dependencies: ReleaseDependencies) -> None:
        self.dependencies = dependencies

    def _measure(self, device: str, operation: Callable[[], T]) -> tuple[T, float, int]:
        cuda = self.dependencies.cuda
        if device == "cuda":
            if cuda is None:
                cuda = _TorchCudaProbe(device)
            cuda.synchronize()
            cuda.reset_peak_memory_stats()
            cuda.synchronize()
        started = self.dependencies.perf_counter()
        result = operation()
        if device == "cuda":
            assert cuda is not None
            cuda.synchronize()
        elapsed = self.dependencies.perf_counter() - started
        elapsed = _require_finite_number(elapsed, "measured seconds", minimum=0.0)
        peak = cuda.max_memory_allocated() if device == "cuda" and cuda is not None else 0
        if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
            raise ValueError("peak VRAM bytes must be a nonnegative integer")
        return result, elapsed, peak

    @staticmethod
    def _reject_stale_outputs(output_dir: Path) -> None:
        if not output_dir.exists():
            return
        candidates = [
            path
            for path in output_dir.rglob("*")
            if path.is_file()
            and path.relative_to(output_dir).as_posix() not in {"report.json", "report.md"}
        ]
        if candidates:
            relative = candidates[0].relative_to(output_dir).as_posix()
            raise ValueError(f"stale or unrecorded evidence output exists: {relative}")

    def _write_reports(self, report: dict[str, Any], output_dir: Path) -> None:
        self.dependencies.publisher.write_text(
            output_dir / "report.md", render_markdown_report(report)
        )
        self.dependencies.publisher.write_json(output_dir / "report.json", report)

    def run(  # noqa: C901
        self,
        corpus: CorpusConfig,
        output_dir: Path,
        device: str,
        depth_resolution: int,
    ) -> dict[str, Any]:
        if device not in {"cpu", "cuda"}:
            raise ValueError("release device must be cpu or cuda")
        if isinstance(depth_resolution, bool) or not isinstance(depth_resolution, int):
            raise TypeError("depth resolution must be an integer")
        if depth_resolution <= 0:
            raise ValueError("depth resolution must be positive")
        output_dir = Path(output_dir).expanduser().resolve()
        commit, dirty = self.dependencies.git_probe()
        report: dict[str, Any] = {
            "tool_schema_version": TOOL_SCHEMA_VERSION,
            "timestamp_utc": self.dependencies.utc_now(),
            "status": "incomplete",
            "project_git": {"commit": commit, "dirty": bool(dirty)},
            "system": self.dependencies.system_probe(device),
            "moge_source_commit": MOGE_SOURCE_COMMIT,
            "adapter_resolution_level": ADAPTER_RESOLUTION_LEVEL,
            "requested_depth_resolution": depth_resolution,
            "inputs": _input_report(corpus),
            "settings": {"device": device, **REPORT_SETTINGS},
            "variants": [],
            "failures": [],
        }
        current_variant: str | None = None
        current_clip: str | None = None
        current_stage = "preflight"
        session: ReleaseSession | None = None
        try:
            self._reject_stale_outputs(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for model_size, repository, revision in release_variants():
                current_variant = model_size
                current_clip = None
                current_stage = "construct_model"
                session = self.dependencies.session_factory(
                    model_size, repository, revision, device
                )
                current_stage = "load_model"
                _unused, load_seconds, peak_vram_bytes = self._measure(device, session.load)
                variant_report: dict[str, Any] = {
                    "model_size": model_size,
                    "repository": repository,
                    "revision": revision,
                    "load_seconds": load_seconds,
                    "peak_vram_bytes": peak_vram_bytes,
                    "clips": [],
                }
                report["variants"].append(variant_report)

                current_stage = "fixed_image_inference"
                fixed = session.infer_fixed(corpus.fixed_image.path, depth_resolution)
                fixed_destination = output_dir / model_size / "fixed-image-depth.npz"
                fixed_destination.parent.mkdir(parents=True, exist_ok=True)
                fixed_temporary = self.dependencies.publisher.temporary_path(fixed_destination)
                try:
                    write_fixed_image_artifact(fixed_temporary, fixed)
                    fixed_values = validate_fixed_image_artifact(fixed_temporary)
                    self.dependencies.publisher.commit_temporary(fixed_temporary, fixed_destination)
                finally:
                    fixed_temporary.unlink(missing_ok=True)
                variant_report.update(
                    {
                        "fixed_image_native_shape": fixed_values["native_shape"],
                        "fixed_image_focal_x_normalized": fixed_values["focal_x_normalized"],
                        "fixed_image_valid_metric_pixels": fixed_values["valid_metric_pixels"],
                        "fixed_image_output": _output_record(output_dir, fixed_destination),
                    }
                )

                for clip in corpus.clips:
                    current_clip = clip.clip_id
                    with tempfile.TemporaryDirectory(
                        prefix=f"moge2-release-{model_size}-{clip.clip_id}-"
                    ) as temporary_directory:
                        workspace = Path(temporary_directory)
                        current_stage = "clip_inference"
                        active_session = session
                        assert active_session is not None

                        def infer_current_clip() -> RawClip:
                            return active_session.infer_clip(clip, depth_resolution, workspace)

                        raw, measured_seconds, _inference_peak = self._measure(
                            device,
                            infer_current_clip,
                        )
                        _validate_raw_clip(raw)
                        _validate_raw_stage(raw)
                        inference_seconds = (
                            measured_seconds
                            if raw.inference_seconds is None
                            else _require_finite_number(
                                raw.inference_seconds,
                                "session inference seconds",
                                minimum=0.0,
                            )
                        )
                        raw_hash = _hash_tree(raw.directory)
                        renders: dict[str, ClipRender] = {}
                        output_records: dict[str, dict[str, str]] = {}
                        for mode, suffix in (
                            ("relative", "relative"),
                            ("metric_camera", "metric-camera"),
                        ):
                            current_stage = f"render_{mode}"
                            destination = output_dir / model_size / f"{clip.clip_id}-{suffix}.mp4"
                            temporary = self.dependencies.publisher.temporary_path(destination)
                            try:
                                rendered = session.render_clip(
                                    clip,
                                    raw,
                                    mode,
                                    temporary,
                                    dict(PROJECTION_SETTINGS),
                                )
                                if Path(rendered.output_path).resolve() != temporary.resolve():
                                    raise ValueError(
                                        "session rendered output does not match requested sibling temporary"
                                    )
                                if not temporary.is_file() or temporary.stat().st_size <= 0:
                                    raise ValueError("rendered evidence video is missing or empty")
                                _validate_output_shape(rendered)
                                if _hash_tree(raw.directory) != raw_hash:
                                    raise ValueError(
                                        "raw stage changed while reusing it across modes"
                                    )
                                self.dependencies.publisher.commit_temporary(temporary, destination)
                            finally:
                                temporary.unlink(missing_ok=True)
                            renders[mode] = rendered
                            output_records[mode] = _output_record(output_dir, destination)
                        current_stage = "measure_clip"
                        if (
                            renders["relative"].output_shape
                            != renders["metric_camera"].output_shape
                        ):
                            raise ValueError(
                                "relative and metric modes must use identical final output shapes"
                            )
                        measurements = compute_clip_measurements(
                            raw,
                            renders["relative"],
                            renders["metric_camera"],
                            source_shape=(clip.height, clip.width),
                            source_roi_xywh=clip.static_roi_xywh,
                            inference_seconds=inference_seconds,
                        )
                        variant_report["clips"].append(
                            {
                                "id": clip.clip_id,
                                "input_sha256": clip.sha256,
                                "raw_stage_sha256": raw_hash,
                                "output_shape": list(
                                    cast(tuple[int, int], renders["relative"].output_shape)
                                ),
                                **measurements,
                                "relative_output": output_records["relative"],
                                "metric_output": output_records["metric_camera"],
                            }
                        )
                current_stage = "unload_model"
                session.unload()
                session = None

            current_variant = None
            current_clip = None
            current_stage = "complete_revalidation"
            _validate_complete_evidence(report, output_dir)
            self._write_reports(report, output_dir)
            _validate_complete_evidence(report, output_dir)
            completed = dict(report)
            completed["status"] = "complete"
            self.dependencies.publisher.write_text(
                output_dir / "report.md", render_markdown_report(completed)
            )
            self.dependencies.publisher.write_json(output_dir / "report.json", completed)
            return completed
        except Exception as error:
            if session is not None:
                try:
                    session.unload()
                except Exception as unload_error:
                    report["failures"].append(
                        {
                            "variant": current_variant,
                            "clip": current_clip,
                            "stage": "unload_model",
                            "error_type": type(unload_error).__name__,
                            "message": str(unload_error),
                        }
                    )
            report["status"] = "incomplete"
            report["failures"].insert(
                0,
                {
                    "variant": current_variant,
                    "clip": current_clip,
                    "stage": current_stage,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                self._write_reports(report, output_dir)
            except Exception:
                pass
            raise ReleaseRunFailed(str(error), report) from error


class _RecordingStereoRenderer:
    def __init__(self, device: str) -> None:
        _ensure_project_import_path()
        from depth_surge_3d.rendering.stereo_renderer import StereoRenderer

        self._renderer = StereoRenderer(device=device)
        self.device = self._renderer.device
        self.results: list[Any] = []
        self.geometries: list[Any] = []

    def render_geometry(self, frame: np.ndarray, geometry: Any, settings: Any) -> Any:
        result = self._renderer.render_geometry(frame, geometry, settings)
        self.results.append(result)
        self.geometries.append(geometry)
        return result


@dataclass
class _ProductionClipState:
    workspace: Path
    relative_output: Path
    relative_renderer: _RecordingStereoRenderer
    relative_settings: dict[str, Any]


class ProductionVariantSession:
    """Thin adapter over the shipped registry, raw store, renderer, and encoder."""

    def __init__(self, model_size: str, repository: str, revision: str, device: str) -> None:
        _ensure_project_import_path()
        from depth_surge_3d.inference.depth.backend_registry import (
            EstimatorRequest,
            create_registered_depth_estimator,
        )

        self.model_size = model_size
        self.repository = repository
        self.revision = revision
        self.device = device
        self.estimator = create_registered_depth_estimator(
            "moge2",
            EstimatorRequest(
                model_path=None,
                model_size=model_size,
                device=device,
                metric=True,
                temporal_window_overlap=10,
            ),
        )
        self._states: dict[str, _ProductionClipState] = {}

    def load(self) -> None:
        if self.estimator.repo_id != self.repository or self.estimator.revision != self.revision:
            raise RuntimeError("constructed MoGe estimator identity differs from the registry pin")
        if not self.estimator.load_model():
            raise RuntimeError(f"failed to load MoGe release variant {self.model_size}")

    def infer_fixed(self, path: Path, depth_resolution: int) -> FixedDepth:
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"fixed image is unreadable: {path}")
        batch = self.estimator.estimate_depth_batch(
            np.asarray([image]), target_fps=30, input_size=depth_resolution, fp32=False
        )
        if batch.camera is None or len(batch.values) != 1:
            raise ValueError("fixed MoGe inference did not return one pinhole camera frame")
        depth = np.asarray(batch.values[0], dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        return FixedDepth(
            depth,
            np.asarray(valid, dtype=np.bool_),
            np.float32(batch.camera.focal_x_normalized[0]),
        )

    def _resolved_settings(
        self, clip: ClipInput, depth_resolution: int, mode: str
    ) -> dict[str, Any]:
        from depth_surge_3d.core.settings import validate_settings
        from depth_surge_3d.inference.depth.backend_registry import (
            validate_backend_geometry_request,
        )
        from depth_surge_3d.utils import (
            auto_detect_resolution,
            calculate_vr_output_dimensions,
            get_resolution_dimensions,
        )

        requested = validate_settings(
            {
                **PROJECTION_SETTINGS,
                "stereo_geometry_mode": mode,
                "depth_resolution": depth_resolution,
                "vr_resolution": "auto",
                "keep_intermediates": True,
                "preserve_audio": False,
                "direct_vr_encode": False,
                "target_fps": clip.fps,
                "raw_storage_dtype": "float32",
            },
            source="explicit",
        )
        requested.update(
            {
                "depth_model_version": "moge2",
                "model_path": None,
                "model_size": self.model_size,
                "use_metric_depth": True,
                "device": self.device,
                "source_width": clip.width,
                "source_height": clip.height,
                "source_fps": clip.fps,
            }
        )
        requested["vr_resolution"] = auto_detect_resolution(clip.width, clip.height, "side_by_side")
        per_eye_width, per_eye_height = get_resolution_dimensions(requested["vr_resolution"])
        vr_width, vr_height = calculate_vr_output_dimensions(
            per_eye_width, per_eye_height, "side_by_side"
        )
        requested.update(
            {
                "per_eye_width": per_eye_width,
                "per_eye_height": per_eye_height,
                "vr_output_width": vr_width,
                "vr_output_height": vr_height,
            }
        )
        validate_backend_geometry_request(
            requested,
            {
                "sample_aspect_ratio_numerator": clip.sar_numerator,
                "sample_aspect_ratio_denominator": clip.sar_denominator,
            },
        )
        return requested

    def _run_mode(
        self, clip: ClipInput, workspace: Path, settings: dict[str, Any]
    ) -> tuple[Path, _RecordingStereoRenderer]:
        from depth_surge_3d.processing.frames.depth_processor import DepthMapProcessor
        from depth_surge_3d.processing.frames.distortion_processor import DistortionProcessor
        from depth_surge_3d.processing.frames.frame_upscaler import FrameUpscalerProcessor
        from depth_surge_3d.processing.frames.stereo_generator import StereoPairGenerator
        from depth_surge_3d.processing.frames.vr_assembler import VRFrameAssembler
        from depth_surge_3d.processing.orchestration.pipeline_orchestrator import (
            ProcessingOrchestrator,
        )
        from depth_surge_3d.processing.video.video_encoder import VideoEncoder
        from depth_surge_3d.utils.path_utils import generate_output_filename

        renderer = _RecordingStereoRenderer(self.device)
        orchestrator = ProcessingOrchestrator(
            depth_processor=DepthMapProcessor(self.estimator),
            stereo_generator=StereoPairGenerator(renderer=cast(Any, renderer)),
            distortion_processor=DistortionProcessor(),
            upscaler=FrameUpscalerProcessor(),
            vr_assembler=VRFrameAssembler(),
            video_encoder=VideoEncoder(),
        )
        properties = {
            "width": clip.width,
            "height": clip.height,
            "fps": clip.fps,
            "frame_count": clip.frame_count,
            "duration": clip.frame_count / clip.fps,
            "sample_aspect_ratio_numerator": clip.sar_numerator,
            "sample_aspect_ratio_denominator": clip.sar_denominator,
            "sample_aspect_ratio": f"{clip.sar_numerator}:{clip.sar_denominator}",
        }
        if not orchestrator.process(clip.path, workspace, properties, settings):
            raise RuntimeError(f"production {settings['stereo_geometry_mode']} pipeline failed")
        output = workspace / generate_output_filename(
            clip.path.name, settings["vr_format"], settings["vr_resolution"]
        )
        if not output.is_file() or output.stat().st_size <= 0:
            raise ValueError(f"production pipeline output is missing: {output}")
        from depth_surge_3d.io.operations import get_video_properties

        if not get_video_properties(str(output)):
            raise ValueError(f"production pipeline output is not a readable video: {output}")
        return output, renderer

    def _instrument_inference(self) -> tuple[Callable[..., Any], dict[str, Any]]:
        original = self.estimator.estimate_depth_batch
        state: dict[str, Any] = {"calls": 0, "frames": 0, "seconds": 0.0}

        def measured(*args: Any, **kwargs: Any) -> Any:
            frames = args[0] if args else kwargs.get("frames")
            if not isinstance(frames, np.ndarray):
                raise TypeError("production inference frames are not a NumPy array")
            if self.device == "cuda":
                import torch

                torch.cuda.synchronize()
            started = time.perf_counter()
            result = original(*args, **kwargs)
            if self.device == "cuda":
                import torch

                torch.cuda.synchronize()
            state["seconds"] += time.perf_counter() - started
            state["calls"] += 1
            state["frames"] += len(frames)
            return result

        return measured, state

    def infer_clip(self, clip: ClipInput, depth_resolution: int, workspace: Path) -> RawClip:
        from depth_surge_3d.processing.frames.depth_storage import RawDepthStore

        settings = self._resolved_settings(clip, depth_resolution, "relative")
        measured, state = self._instrument_inference()
        original = self.estimator.estimate_depth_batch
        self.estimator.estimate_depth_batch = measured
        try:
            relative_output, renderer = self._run_mode(clip, workspace, settings)
        finally:
            self.estimator.estimate_depth_batch = original
        raw_dir = workspace / "02_depth_raw"
        metadata = RawDepthStore.read_metadata(raw_dir)
        if metadata is None or metadata.get("schema_version") != 3:
            raise ValueError("production raw stage is not completed schema v3")
        store = RawDepthStore(raw_dir, metadata)
        if store.validate_payloads(cleanup_temporaries=False) != len(metadata["frame_names"]):
            raise ValueError("production raw stage is incomplete")
        batch = store.load_batch(store.complete_files)
        if batch.camera is None:
            raise ValueError("production raw stage is missing pinhole camera values")
        depth = np.asarray(batch.values, dtype=np.float32)
        valid = np.asarray(np.isfinite(depth) & (depth > 0), dtype=np.bool_)
        self._states[clip.clip_id] = _ProductionClipState(
            workspace, relative_output, renderer, settings
        )
        return RawClip(
            directory=raw_dir,
            depth=depth,
            valid=valid,
            focal_x_normalized=np.asarray(batch.camera.focal_x_normalized, dtype=np.float32),
            frame_names=tuple(Path(name).stem for name in metadata["frame_names"]),
            inference_calls=int(state["calls"]),
            inferred_frame_count=int(state["frames"]),
            inference_seconds=float(state["seconds"]),
        )

    @staticmethod
    def _holes(renderer: _RecordingStereoRenderer) -> np.ndarray:
        if not renderer.results:
            raise ValueError("production renderer did not expose output masks")
        return np.stack(
            [
                np.concatenate((result.left_hole_mask, result.right_hole_mask), axis=1)
                for result in renderer.results
            ]
        ).astype(np.bool_, copy=False)

    def render_clip(
        self,
        clip: ClipInput,
        raw: RawClip,
        mode: str,
        output_path: Path,
        settings: dict[str, Any],
    ) -> ClipRender:
        del settings
        state = self._states.get(clip.clip_id)
        if state is None or state.workspace != raw.directory.parent:
            raise ValueError("production clip state does not match the raw stage")
        if mode == "relative":
            shutil.copyfile(state.relative_output, output_path)
            return ClipRender(
                "relative",
                output_path,
                self._holes(state.relative_renderer),
                output_shape=(
                    int(state.relative_settings["vr_output_height"]),
                    int(state.relative_settings["vr_output_width"]),
                ),
            )
        if mode != "metric_camera":
            raise ValueError(f"unsupported release render mode: {mode}")
        metric_settings = self._resolved_settings(
            clip, int(state.relative_settings["depth_resolution"]), "metric_camera"
        )
        measured, inference_state = self._instrument_inference()
        original = self.estimator.estimate_depth_batch
        self.estimator.estimate_depth_batch = measured
        try:
            metric_output, renderer = self._run_mode(clip, state.workspace, metric_settings)
        finally:
            self.estimator.estimate_depth_batch = original
        if inference_state["calls"] != 0 or inference_state["frames"] != 0:
            raise ValueError("metric render reinferred frames instead of reusing raw schema v3")
        shutil.copyfile(metric_output, output_path)
        source_width = clip.width
        crop_factor = float(metric_settings["crop_factor"])
        retained_width = max(1, min(source_width, int(round(source_width * crop_factor))))
        retained_height = max(1, min(clip.height, int(round(clip.height * crop_factor))))
        retained_x0 = (source_width - retained_width) // 2
        retained_y0 = (clip.height - retained_height) // 2
        scale = metric_settings["per_eye_width"] / retained_width
        disparities = np.stack(
            [
                geometry.total_disparity_fraction * source_width * scale
                for geometry in renderer.geometries
            ]
        ).astype(np.float64, copy=False)
        sidecar_dir = state.workspace / "04_left_frames" / "clamp_stats"
        sidecars = tuple(sidecar_dir / f"{name}.json" for name in raw.frame_names)
        return ClipRender(
            "metric_camera",
            output_path,
            self._holes(renderer),
            disparities,
            (
                retained_x0,
                retained_y0,
                retained_x0 + retained_width,
                retained_y0 + retained_height,
            ),
            sidecars,
            np.stack([geometry.source_valid for geometry in renderer.geometries]).astype(
                np.bool_, copy=False
            ),
            (
                int(metric_settings["vr_output_height"]),
                int(metric_settings["vr_output_width"]),
            ),
        )

    def unload(self) -> None:
        self.estimator.unload_model()
        self._states.clear()


def _default_session_factory(
    model_size: str, repository: str, revision: str, device: str
) -> ReleaseSession:
    return ProductionVariantSession(model_size, repository, revision, device)


def _ensure_project_import_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "src"
    source_text = str(source_root)
    while source_text in sys.path:
        sys.path.remove(source_text)
    sys.path.insert(0, source_text)


def main(argv: Sequence[str] | None = None) -> int:
    args = create_argument_parser().parse_args(argv)
    try:
        corpus = load_corpus_config(args.corpus_config)
        dependencies = ReleaseDependencies(session_factory=_default_session_factory)
        report = ReleaseRunner(dependencies).run(
            corpus,
            args.output_dir,
            args.device,
            args.depth_resolution,
        )
    except (ReleaseRunFailed, ValueError, RuntimeError, OSError) as error:
        print(f"MoGe release verification failed: {error}", file=sys.stderr)
        return 1
    print(
        f"MoGe release evidence complete: {args.output_dir} "
        f"({len(report['variants'])} variants)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
