"""File-backed coordinator for optional VDPP temporal stabilization."""

from __future__ import annotations

import importlib.metadata
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from ...core.depth_contract import canonical_json_hash
from ...core.render_disparity import validate_render_disparity_input
from ...core.vdpp_calibration import (
    MAX_POSTCLIP_MEAN_DRIFT,
    MAX_PRECLIP_OUT_OF_RANGE_FRACTION,
    MIDPOINT_CODE,
    MIN_CORRELATION,
    MIN_PAIR_COUNT,
    MIN_POSITIVE_SCALE,
    MIN_POSTCLIP_CONTRAST_RATIO,
    STATS_TILE_PIXELS,
    VARIANCE_EPSILON,
    NumericalContractError,
    PairTileState,
    ScalarTileState,
    candidate_tile,
    canonical_derived_diagnostics,
    canonicalize_float,
    finalize_pair_moments,
    merge_pair_states,
    merge_scalar_states,
    normalize_bounded_stat,
    normalize_vdpp_input_frame,
    planned_tile_count,
    reduce_pair_tile,
    reduce_scalar_tile,
)
from ...inference.depth.vdpp_artifact import ensure_vdpp_checkpoint
from ...inference.depth.vdpp_contract import (
    VDPP_WINDOW_SIZE,
    build_vdpp_execution_plan,
    vdpp_model_identity,
)
from .depth_normalizer import decode_canonical_png, encode_canonical_png
from .temporal_storage import StabilizedDepthStore, build_final_shot_plan


def _default_cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def _default_postprocessor_factory(checkpoint_path: Path, device: str):
    from ...inference.depth.vdpp_temporal_postprocessor import (
        VDPPTemporalPostprocessor,
    )

    return VDPPTemporalPostprocessor(
        checkpoint_path=checkpoint_path,
        device=device,
    )


def collect_vdpp_runtime_identity(
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collect only after artifact-first planning proves generation is needed."""

    import torch

    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("VDPP generation requires an available CUDA device")
    index = selected.index if selected.index is not None else torch.cuda.current_device()
    selected = torch.device("cuda", index)
    try:
        xformers_version = importlib.metadata.version("xformers")
        attention_backend = "xformers"
    except importlib.metadata.PackageNotFoundError:
        xformers_version = None
        attention_backend = "torch"
    driver_version = None
    get_driver_version = getattr(torch._C, "_cuda_getDriverVersion", None)
    if callable(get_driver_version):
        try:
            driver_version = str(get_driver_version())
        except RuntimeError:
            pass
    identity = {
        "numpy_version": str(np.__version__),
        "opencv_version": str(cv2.__version__),
        "torch_version": str(torch.__version__),
        "xformers_version": xformers_version,
        "attention_backend": attention_backend,
        "attention_operator": "auto",
        "cuda_runtime": str(torch.version.cuda),
        "cuda_driver": driver_version,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_name": torch.cuda.get_device_name(selected),
        "compute_capability": list(torch.cuda.get_device_capability(selected)),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    return identity, dict(identity)


@dataclass(frozen=True)
class _ShotPreScan:
    shot_id: int
    start: int
    end: int
    midpoint_count: int
    flat_frame_count: int
    shot_pixels: int

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def all_midpoint(self) -> bool:
        return self.midpoint_count == self.shot_pixels

    @property
    def needs_model(self) -> bool:
        return not self.all_midpoint and self.flat_frame_count != self.length


class TemporalDepthStabilizer:
    """Own validation, artifacts, shot resume, model laziness, and progress."""

    def __init__(
        self,
        *,
        effective_device: str,
        models_dir: Path | str,
        cuda_available: Callable[[], bool] = _default_cuda_available,
        disk_usage: Callable[[Path], Any] = shutil.disk_usage,
        checkpoint_resolver: Callable[..., Path] = ensure_vdpp_checkpoint,
        postprocessor_factory: Callable[[Path, str], Any] = _default_postprocessor_factory,
        runtime_identity_provider: Callable[
            [str], tuple[dict[str, Any], dict[str, Any]]
        ] = collect_vdpp_runtime_identity,
    ) -> None:
        self.effective_device = str(effective_device)
        self.models_dir = Path(models_dir)
        self._cuda_available = cuda_available
        self._disk_usage = disk_usage
        self._checkpoint_resolver = checkpoint_resolver
        self._postprocessor_factory = postprocessor_factory
        self._runtime_identity_provider = runtime_identity_provider

    @staticmethod
    def _read_final_scene_manifest(
        scene_dir: Path,
        *,
        frame_names: list[str],
        canonical_scene_fingerprint: str,
    ) -> dict[str, Any]:
        path = scene_dir / "scene_manifest.json"
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Final scene manifest is invalid: {path}") from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("status") != "final"
            or manifest.get("frame_names") != frame_names
            or canonical_json_hash(manifest) != canonical_scene_fingerprint
        ):
            raise ValueError("Final scene manifest does not match canonical disparity")
        return manifest

    @staticmethod
    def _semantic_identity(
        *,
        frame_names: list[str],
        native_shape: tuple[int, int],
        source_fingerprint: str,
        scene_fingerprint: str,
        shot_plan: list[dict[str, int]],
    ) -> dict[str, Any]:
        return {
            "frame_names": frame_names,
            "native_shape": list(native_shape),
            "source_canonical_fingerprint": source_fingerprint,
            "scene_manifest_fingerprint": scene_fingerprint,
            "postprocessor_settings": {"temporal_postprocessor": "vdpp"},
            "model_identity": vdpp_model_identity(),
            "execution_plan": build_vdpp_execution_plan(native_shape),
            "shot_plan": shot_plan,
        }

    @staticmethod
    def _report_progress(
        progress_tracker: Any,
        message: str,
        *,
        completed: int,
        total: int,
    ) -> None:
        if progress_tracker is None:
            return
        progress_tracker.update_progress(
            message,
            phase="temporal_stabilization",
            frame_num=completed,
            step_name="Temporal Depth Stabilization",
            step_progress=completed,
            step_total=total,
        )

    @staticmethod
    def _required_disk_bytes(
        pending_frames: int,
        largest_pending_shot: int,
        native_shape: tuple[int, int],
    ) -> int:
        frame_u16_bytes = native_shape[0] * native_shape[1] * 2
        conservative_png_bound = math.ceil(frame_u16_bytes * 1.10)
        raw_memmap_bytes = largest_pending_shot * native_shape[0] * native_shape[1] * 4
        inner = (
            pending_frames * conservative_png_bound
            + raw_memmap_bytes
            + conservative_png_bound
            + 1024 * 1024
        )
        return math.ceil(1.10 * inner)

    def _preflight_disk(
        self,
        stable_root: Path,
        *,
        pending_frames: int,
        largest_pending_shot: int,
        native_shape: tuple[int, int],
    ) -> None:
        required = self._required_disk_bytes(
            pending_frames,
            largest_pending_shot,
            native_shape,
        )
        try:
            free = int(self._disk_usage(stable_root.parent).free)
        except OSError as exc:
            raise OSError(f"Could not inspect disk space for VDPP output: {stable_root}") from exc
        if free < required:
            raise OSError(
                f"Insufficient disk space for VDPP output: need {required} bytes, "
                f"only {free} bytes are free"
            )

    @staticmethod
    def _read_source(path: Path, native_shape: tuple[int, int]) -> np.ndarray:
        decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise OSError(f"Could not decode canonical disparity: {path}")
        if decoded.dtype != np.uint16 or decoded.ndim != 2:
            raise TypeError(f"Canonical disparity must be uint16 grayscale: {path}")
        if decoded.shape != native_shape:
            raise ValueError(
                f"Canonical disparity shape {decoded.shape} does not match {native_shape}: {path}"
            )
        return decoded

    @staticmethod
    def _has_non_midpoint_range(source: np.ndarray) -> bool:
        non_midpoint = source != np.uint16(MIDPOINT_CODE)
        if not bool(np.any(non_midpoint)):
            return False
        lo_code = int(np.min(source, where=non_midpoint, initial=np.iinfo(np.uint16).max))
        hi_code = int(np.max(source, where=non_midpoint, initial=np.iinfo(np.uint16).min))
        return lo_code != hi_code

    @classmethod
    def _pre_scan_shot(
        cls,
        base_files: list[Path],
        shot: dict[str, int],
        native_shape: tuple[int, int],
    ) -> _ShotPreScan:
        midpoint_count = 0
        flat_frame_count = 0
        for frame_index in range(shot["start"], shot["end"]):
            source = cls._read_source(base_files[frame_index], native_shape)
            midpoint_count += int(np.count_nonzero(source == np.uint16(MIDPOINT_CODE)))
            if not cls._has_non_midpoint_range(source):
                flat_frame_count += 1
        length = shot["end"] - shot["start"]
        return _ShotPreScan(
            shot_id=shot["shot_id"],
            start=shot["start"],
            end=shot["end"],
            midpoint_count=midpoint_count,
            flat_frame_count=flat_frame_count,
            shot_pixels=length * native_shape[0] * native_shape[1],
        )

    @classmethod
    def _window_loader(
        cls,
        base_files: list[Path],
        *,
        shot_start: int,
        shot_length: int,
        native_shape: tuple[int, int],
    ) -> Callable[[int, int], np.ndarray]:
        def load_window(start: int, end: int) -> np.ndarray:
            if (
                type(start) is not int
                or type(end) is not int
                or start < 0
                or end <= start
                or end > shot_length
            ):
                raise ValueError(f"Invalid VDPP shot-local range [{start}, {end})")
            if end - start > VDPP_WINDOW_SIZE:
                raise ValueError(
                    f"VDPP loader window size exceeds the pinned bound of {VDPP_WINDOW_SIZE}"
                )
            output = np.empty((end - start, *native_shape), dtype=np.float32)
            for output_index, local_index in enumerate(range(start, end)):
                source = cls._read_source(
                    base_files[shot_start + local_index],
                    native_shape,
                )
                normalize_vdpp_input_frame(source, output[output_index])
            return output

        return load_window

    @staticmethod
    def _empty_calibration(
        scan: _ShotPreScan,
        *,
        mode: str,
        fallback_reason: str | None,
        pair_count: int = 0,
    ) -> dict[str, Any]:
        return {
            "mode": mode,
            "pair_count": pair_count,
            "midpoint_count": scan.midpoint_count,
            "midpoint_fraction": canonicalize_float(scan.midpoint_count / scan.shot_pixels),
            "flat_frame_count": scan.flat_frame_count,
            "source_mean": None,
            "source_variance": None,
            "source_std": None,
            "raw_mean": None,
            "raw_variance": None,
            "raw_std": None,
            "covariance": None,
            "correlation": None,
            "scale": None,
            "shift": None,
            "candidate_mean": None,
            "candidate_std": None,
            "postclip_contrast_ratio": None,
            "postclip_mean_drift": None,
            "preclip_low_fraction": None,
            "preclip_high_fraction": None,
            "fallback_reason": fallback_reason,
        }

    @staticmethod
    def _copy_outputs(
        base_files: list[Path],
        scan: _ShotPreScan,
        native_shape: tuple[int, int],
    ) -> Iterable[tuple[int, np.ndarray]]:
        for local_index, frame_index in enumerate(range(scan.start, scan.end)):
            yield (
                local_index,
                TemporalDepthStabilizer._read_source(base_files[frame_index], native_shape),
            )

    @staticmethod
    def _tile_ranges(pixel_count: int) -> Iterable[tuple[int, int]]:
        for start in range(0, pixel_count, STATS_TILE_PIXELS):
            yield start, min(start + STATS_TILE_PIXELS, pixel_count)

    @classmethod
    def _quality_pass(
        cls,
        raw_memmap: np.memmap,
        base_files: list[Path],
        scan: _ShotPreScan,
        native_shape: tuple[int, int],
        *,
        scale: float,
        shift: float,
        pair_count: int,
        tile_count: int,
    ) -> dict[str, float]:
        state = ScalarTileState.empty()
        preclip_low_count = 0
        preclip_high_count = 0
        pixel_count = native_shape[0] * native_shape[1]
        for local_index, frame_index in enumerate(range(scan.start, scan.end)):
            source = cls._read_source(base_files[frame_index], native_shape)
            if not cls._has_non_midpoint_range(source):
                continue
            source_flat = source.reshape(-1)
            raw_flat = raw_memmap[local_index].reshape(-1)
            for start, end in cls._tile_ranges(pixel_count):
                source_tile = source_flat[start:end]
                eligible = source_tile != np.uint16(MIDPOINT_CODE)
                preclip64, candidate32 = candidate_tile(
                    raw_flat[start:end],
                    scale,
                    shift,
                )
                preclip_low_count += int(np.count_nonzero((preclip64 < 0.0) & eligible))
                preclip_high_count += int(np.count_nonzero((preclip64 > 1.0) & eligible))
                state = merge_scalar_states(
                    state,
                    reduce_scalar_tile(candidate32, eligible),
                )
        if state.count != pair_count:
            raise RuntimeError("VDPP quality-pass pair count changed")
        candidate_mean = normalize_bounded_stat(
            state.mean,
            0.0,
            1.0,
            planned_tile_count=tile_count,
        )
        candidate_variance = normalize_bounded_stat(
            canonicalize_float(state.m2 / state.count),
            0.0,
            0.25,
            planned_tile_count=tile_count,
        )
        candidate_std = normalize_bounded_stat(
            canonicalize_float(math.sqrt(candidate_variance)),
            0.0,
            0.5,
            planned_tile_count=tile_count,
        )
        return {
            "candidate_mean": candidate_mean,
            "candidate_std": candidate_std,
            "preclip_low_fraction": normalize_bounded_stat(
                canonicalize_float(preclip_low_count / pair_count),
                0.0,
                1.0,
                planned_tile_count=tile_count,
            ),
            "preclip_high_fraction": normalize_bounded_stat(
                canonicalize_float(preclip_high_count / pair_count),
                0.0,
                1.0,
                planned_tile_count=tile_count,
            ),
        }

    @classmethod
    def _accepted_outputs(
        cls,
        raw_memmap: np.memmap,
        base_files: list[Path],
        scan: _ShotPreScan,
        native_shape: tuple[int, int],
        *,
        scale: float,
        shift: float,
    ) -> Iterable[tuple[int, np.ndarray]]:
        pixel_count = native_shape[0] * native_shape[1]
        for local_index, frame_index in enumerate(range(scan.start, scan.end)):
            source = cls._read_source(base_files[frame_index], native_shape)
            if not cls._has_non_midpoint_range(source):
                yield local_index, source
                continue
            candidate_frame32 = np.empty(native_shape, dtype=np.float32)
            raw_flat = raw_memmap[local_index].reshape(-1)
            candidate_flat = candidate_frame32.reshape(-1)
            for start, end in cls._tile_ranges(pixel_count):
                _, candidate_tile32 = candidate_tile(
                    raw_flat[start:end],
                    scale,
                    shift,
                )
                candidate_flat[start:end] = candidate_tile32
            encoded = encode_canonical_png(candidate_frame32)
            encoded[source == np.uint16(MIDPOINT_CODE)] = np.uint16(MIDPOINT_CODE)
            yield local_index, encoded

    @staticmethod
    def _populate_moments(
        calibration: dict[str, Any],
        moments: dict[str, float],
        *,
        tile_count: int,
    ) -> None:
        calibration.update(moments)
        calibration["source_std"] = normalize_bounded_stat(
            canonicalize_float(math.sqrt(moments["source_variance"])),
            0.0,
            0.5,
            planned_tile_count=tile_count,
        )
        calibration["raw_std"] = normalize_bounded_stat(
            canonicalize_float(math.sqrt(moments["raw_variance"])),
            0.0,
            planned_tile_count=tile_count,
        )

    @classmethod
    def _calibrate_raw_shot(  # noqa: C901, PLR0912
        cls,
        raw_memmap: np.memmap,
        base_files: list[Path],
        scan: _ShotPreScan,
        native_shape: tuple[int, int],
        pair_state: PairTileState,
        pair_count: int,
        *,
        statistics_failed: bool,
    ) -> tuple[dict[str, Any], float | None, float | None]:
        tile_count = planned_tile_count(scan.length, native_shape)
        if pair_count < MIN_PAIR_COUNT:
            return (
                cls._empty_calibration(
                    scan,
                    mode="base_fallback",
                    fallback_reason="too_few_pairs",
                    pair_count=pair_count,
                ),
                None,
                None,
            )
        if statistics_failed:
            return (
                cls._empty_calibration(
                    scan,
                    mode="base_fallback",
                    fallback_reason="nonfinite_statistics",
                    pair_count=pair_count,
                ),
                None,
                None,
            )
        if pair_state.count != pair_count:
            raise RuntimeError("VDPP fit-pass pair count changed")
        try:
            moments = finalize_pair_moments(
                pair_state,
                planned_tile_count=tile_count,
            )
        except NumericalContractError:
            raise
        except (FloatingPointError, ValueError):
            return (
                cls._empty_calibration(
                    scan,
                    mode="base_fallback",
                    fallback_reason="nonfinite_statistics",
                    pair_count=pair_count,
                ),
                None,
                None,
            )

        calibration = cls._empty_calibration(
            scan,
            mode="base_fallback",
            fallback_reason=None,
            pair_count=pair_count,
        )
        cls._populate_moments(calibration, moments, tile_count=tile_count)
        if moments["source_variance"] <= VARIANCE_EPSILON:
            calibration["fallback_reason"] = "source_variance"
            return calibration, None, None
        if moments["raw_variance"] <= VARIANCE_EPSILON:
            calibration["fallback_reason"] = "raw_variance"
            return calibration, None, None

        try:
            correlation = normalize_bounded_stat(
                canonicalize_float(
                    moments["covariance"]
                    / math.sqrt(moments["raw_variance"] * moments["source_variance"])
                ),
                -1.0,
                1.0,
                planned_tile_count=tile_count,
            )
            raw_scale = moments["covariance"] / moments["raw_variance"]
            raw_shift = moments["source_mean"] - raw_scale * moments["raw_mean"]
        except NumericalContractError:
            raise
        except (OverflowError, ValueError, ZeroDivisionError):
            return (
                cls._empty_calibration(
                    scan,
                    mode="base_fallback",
                    fallback_reason="nonfinite_statistics",
                    pair_count=pair_count,
                ),
                None,
                None,
            )
        calibration["correlation"] = correlation
        if not math.isfinite(raw_scale) or not math.isfinite(raw_shift):
            calibration["fallback_reason"] = "nonfinite_fit"
            return calibration, None, None
        scale = canonicalize_float(raw_scale)
        shift = canonicalize_float(raw_shift)
        calibration["scale"] = scale
        calibration["shift"] = shift
        if scale < MIN_POSITIVE_SCALE:
            calibration["fallback_reason"] = "scale_below_minimum"
            return calibration, None, None
        if correlation < MIN_CORRELATION:
            calibration["fallback_reason"] = "correlation"
            return calibration, None, None

        try:
            quality = cls._quality_pass(
                raw_memmap,
                base_files,
                scan,
                native_shape,
                scale=scale,
                shift=shift,
                pair_count=pair_count,
                tile_count=tile_count,
            )
        except NumericalContractError:
            raise
        except (FloatingPointError, ValueError):
            return (
                cls._empty_calibration(
                    scan,
                    mode="base_fallback",
                    fallback_reason="nonfinite_statistics",
                    pair_count=pair_count,
                ),
                None,
                None,
            )
        derived = canonical_derived_diagnostics(
            midpoint_count=scan.midpoint_count,
            shot_pixels=scan.shot_pixels,
            source_mean=moments["source_mean"],
            source_variance=moments["source_variance"],
            raw_mean=moments["raw_mean"],
            raw_variance=moments["raw_variance"],
            covariance=moments["covariance"],
            candidate_mean=quality["candidate_mean"],
            candidate_std=quality["candidate_std"],
            preclip_low_fraction=quality["preclip_low_fraction"],
            preclip_high_fraction=quality["preclip_high_fraction"],
            planned_tile_count=tile_count,
        )
        calibration.update(quality)
        calibration["postclip_contrast_ratio"] = derived["postclip_contrast_ratio"]
        calibration["postclip_mean_drift"] = derived["postclip_mean_drift"]
        preclip_total = derived["preclip_out_of_range_fraction"]
        if calibration["postclip_contrast_ratio"] < MIN_POSTCLIP_CONTRAST_RATIO:
            calibration["fallback_reason"] = "contrast"
        elif calibration["postclip_mean_drift"] > MAX_POSTCLIP_MEAN_DRIFT:
            calibration["fallback_reason"] = "mean_drift"
        elif preclip_total > MAX_PRECLIP_OUT_OF_RANGE_FRACTION:
            calibration["fallback_reason"] = "preclip_out_of_range"
        else:
            calibration["mode"] = "ols"
            calibration["fallback_reason"] = None
            return calibration, scale, shift
        return calibration, None, None

    @classmethod
    def _run_model_shot(  # noqa: C901
        cls,
        *,
        postprocessor: Any,
        store: StabilizedDepthStore,
        stable_root: Path,
        base_files: list[Path],
        scan: _ShotPreScan,
        native_shape: tuple[int, int],
    ) -> None:
        work_dir = stable_root / ".vdpp-work"
        memmap_path = work_dir / f"shot_{scan.shot_id}.raw.f32.mmap"
        raw_memmap: np.memmap | None = None
        iterator = None
        output_iterator = None
        try:
            work_dir.mkdir(parents=True, exist_ok=True)
            raw_memmap = np.memmap(
                memmap_path,
                mode="w+",
                dtype=np.float32,
                shape=(scan.length, *native_shape),
            )
            loader = cls._window_loader(
                base_files,
                shot_start=scan.start,
                shot_length=scan.length,
                native_shape=native_shape,
            )
            iterator = postprocessor.process_shot(scan.length, loader)
            pair_state = PairTileState.empty()
            pair_count = 0
            statistics_failed = False
            consumed = 0
            pixel_count = native_shape[0] * native_shape[1]
            for expected_local, item in enumerate(iterator):
                if not isinstance(item, tuple) or len(item) != 2:
                    raise ValueError("VDPP output must be an index/value pair")
                local_index, raw_output = item
                if local_index != expected_local or expected_local >= scan.length:
                    raise ValueError("VDPP output indexes are not ordered and contiguous")
                if (
                    not isinstance(raw_output, np.ndarray)
                    or raw_output.dtype != np.float32
                    or raw_output.shape != native_shape
                ):
                    raise TypeError("VDPP output must be native-shape float32")
                if not np.isfinite(raw_output).all():
                    raise FloatingPointError("VDPP model output contains non-finite values")
                raw_memmap[local_index] = raw_output
                source = cls._read_source(
                    base_files[scan.start + local_index],
                    native_shape,
                )
                if cls._has_non_midpoint_range(source):
                    raw_flat = raw_output.reshape(-1)
                    source_flat = source.reshape(-1)
                    for start, end in cls._tile_ranges(pixel_count):
                        source_tile = source_flat[start:end]
                        eligible = source_tile != np.uint16(MIDPOINT_CODE)
                        pair_count += int(np.count_nonzero(eligible))
                        if statistics_failed:
                            continue
                        source_f32 = decode_canonical_png(source_tile)
                        try:
                            pair_state = merge_pair_states(
                                pair_state,
                                reduce_pair_tile(
                                    raw_flat[start:end],
                                    source_f32,
                                    eligible,
                                ),
                            )
                        except NumericalContractError:
                            raise
                        except (FloatingPointError, ValueError):
                            statistics_failed = True
                consumed += 1
            if consumed != scan.length:
                raise ValueError(f"VDPP emitted {consumed} frames; expected {scan.length}")
            raw_memmap.flush()
            calibration, scale, shift = cls._calibrate_raw_shot(
                raw_memmap,
                base_files,
                scan,
                native_shape,
                pair_state,
                pair_count,
                statistics_failed=statistics_failed,
            )
            output_iterator = (
                cls._accepted_outputs(
                    raw_memmap,
                    base_files,
                    scan,
                    native_shape,
                    scale=scale,
                    shift=shift,
                )
                if calibration["mode"] == "ols"
                else cls._copy_outputs(base_files, scan, native_shape)
            )
            store.commit_shot(
                scan.shot_id,
                output_iterator,
                calibration=calibration,
            )
        finally:
            active_exception = sys.exc_info()[0] is not None
            cleanup_errors: list[BaseException] = []
            if output_iterator is not None:
                close = getattr(output_iterator, "close", None)
                if callable(close):
                    try:
                        close()
                    except BaseException as exc:
                        cleanup_errors.append(exc)
            if iterator is not None:
                close = getattr(iterator, "close", None)
                if callable(close):
                    try:
                        close()
                    except BaseException as exc:
                        cleanup_errors.append(exc)
            if raw_memmap is not None:
                try:
                    raw_memmap.flush()
                except BaseException as exc:
                    cleanup_errors.append(exc)
                try:
                    mapping = getattr(raw_memmap, "_mmap", None)
                    if mapping is not None:
                        mapping.close()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            try:
                memmap_path.unlink(missing_ok=True)
            except BaseException as exc:
                cleanup_errors.append(exc)
            try:
                work_dir.rmdir()
            except FileNotFoundError:
                pass
            except BaseException as exc:
                cleanup_errors.append(exc)
            if cleanup_errors:
                if active_exception:
                    for cleanup_error in cleanup_errors:
                        print(
                            f"VDPP memmap cleanup also failed: {cleanup_error}",
                            file=sys.stderr,
                        )
                else:
                    raise cleanup_errors[0]

    def generate_files(  # noqa: C901, PLR0912, PLR0915
        self,
        base_files: list[Path],
        settings: dict[str, Any],
        directories: dict[str, Path],
        progress_tracker: Any = None,
    ) -> list[Path]:
        """Return a complete stabilized artifact or fail without passthrough."""

        if settings.get("temporal_postprocessor") != "vdpp":
            raise ValueError(
                "TemporalDepthStabilizer may only be invoked when temporal_postprocessor=vdpp"
            )
        if not base_files:
            raise ValueError("VDPP requires canonical disparity files")

        metadata_path = base_files[0].parent / "metadata.json"
        try:
            base_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Canonical disparity metadata is invalid: {metadata_path}") from exc
        if not isinstance(base_metadata, dict):
            raise ValueError(f"Canonical disparity metadata is invalid: {metadata_path}")
        frame_names = base_metadata.get("frame_names")
        if not isinstance(frame_names, list) or not all(
            isinstance(name, str) for name in frame_names
        ):
            raise ValueError("Canonical disparity frame manifest is invalid")
        frame_files = [Path(name) for name in frame_names]
        base_artifact = validate_render_disparity_input(base_files, frame_files)
        if base_artifact.producer != "base":
            raise ValueError("VDPP input must be base canonical disparity")
        scene_fingerprint = base_artifact.metadata.get("scene_manifest_fingerprint")
        if not isinstance(scene_fingerprint, str):
            raise ValueError("Canonical disparity is missing its scene fingerprint")
        manifest = self._read_final_scene_manifest(
            Path(directories["scene_data"]),
            frame_names=frame_names,
            canonical_scene_fingerprint=scene_fingerprint,
        )
        cuts = manifest.get("final_cuts")
        if not isinstance(cuts, list) or not all(
            not isinstance(cut, bool) and isinstance(cut, int) for cut in cuts
        ):
            raise ValueError("Final scene manifest has invalid cut indexes")
        shot_plan = build_final_shot_plan(len(base_files), cuts)
        semantic_identity = self._semantic_identity(
            frame_names=frame_names,
            native_shape=base_artifact.native_shape,
            source_fingerprint=base_artifact.fingerprint,
            scene_fingerprint=scene_fingerprint,
            shot_plan=shot_plan,
        )
        stable_root = Path(directories["disparity_stabilized"])

        lightweight_store = StabilizedDepthStore(
            stable_root,
            frame_files=frame_files,
            semantic_identity=semantic_identity,
            runtime_identity={},
            execution_provenance={},
        )
        lightweight_audit = lightweight_store.audit()
        if lightweight_audit.complete:
            self._report_progress(
                progress_tracker,
                f"Reused {len(base_files)} stabilized depth maps",
                completed=len(base_files),
                total=len(base_files),
            )
            return lightweight_store.depth_files

        if not self.effective_device.startswith("cuda") or not self._cuda_available():
            raise RuntimeError(
                "VDPP generation requires CUDA; a complete cached stabilized artifact "
                "can still be rendered without CUDA"
            )
        runtime_identity, execution_provenance = self._runtime_identity_provider(
            self.effective_device
        )
        runtime_identity = {
            **runtime_identity,
            "numpy_version": str(np.__version__),
            "opencv_version": str(cv2.__version__),
        }
        execution_provenance = {
            **execution_provenance,
            "numpy_version": str(np.__version__),
            "opencv_version": str(cv2.__version__),
        }
        store = StabilizedDepthStore(
            stable_root,
            frame_files=frame_files,
            semantic_identity=semantic_identity,
            runtime_identity=runtime_identity,
            execution_provenance=execution_provenance,
        )
        audit = store.audit()
        if audit.complete:
            return store.depth_files

        required_ids = set(audit.invalid_shot_ids) | set(audit.pending_shot_ids)
        if audit.reset_required:
            required_ids = {shot["shot_id"] for shot in shot_plan}
        if not required_ids:
            store.prepare(audit)
            completed = store.finalize()
            self._report_progress(
                progress_tracker,
                f"Reused {len(completed)} stabilized depth maps",
                completed=len(completed),
                total=len(completed),
            )
            return completed

        scans = {
            shot["shot_id"]: self._pre_scan_shot(
                base_files,
                shot,
                base_artifact.native_shape,
            )
            for shot in shot_plan
            if shot["shot_id"] in required_ids
        }
        model_scans = [scan for scan in scans.values() if scan.needs_model]
        pending_frames = sum(scan.length for scan in scans.values())
        largest_pending_shot = max(
            (scan.length for scan in model_scans),
            default=0,
        )
        self._preflight_disk(
            stable_root,
            pending_frames=pending_frames,
            largest_pending_shot=largest_pending_shot,
            native_shape=base_artifact.native_shape,
        )

        postprocessor = None
        try:
            if model_scans:
                checkpoint = self._checkpoint_resolver(
                    self.models_dir,
                    progress_callback=lambda current, total: self._report_progress(
                        progress_tracker,
                        f"Downloading VDPP checkpoint {current}/{total} bytes",
                        completed=0,
                        total=len(base_files),
                    ),
                )
                postprocessor = self._postprocessor_factory(
                    Path(checkpoint),
                    self.effective_device,
                )
                if postprocessor.model_identity() != semantic_identity["model_identity"]:
                    raise RuntimeError("VDPP model identity does not match the persisted plan")
                if (
                    postprocessor.execution_plan(base_artifact.native_shape)
                    != semantic_identity["execution_plan"]
                ):
                    raise RuntimeError("VDPP execution arguments do not match the persisted plan")
                preflight = postprocessor.preflight(
                    largest_pending_shot,
                    base_artifact.native_shape,
                )
                store.execution_provenance.update(preflight)
            store.prepare(audit)

            completed_frames = sum(
                shot["end"] - shot["start"]
                for shot in shot_plan
                if shot["shot_id"] in audit.reusable_shot_ids
            )
            self._report_progress(
                progress_tracker,
                "Stabilizing temporal depth",
                completed=completed_frames,
                total=len(base_files),
            )
            for shot in shot_plan:
                shot_id = shot["shot_id"]
                if shot_id in audit.reusable_shot_ids and not audit.reset_required:
                    continue
                scan = scans[shot_id]
                if scan.all_midpoint:
                    calibration = self._empty_calibration(
                        scan,
                        mode="all_midpoint",
                        fallback_reason=None,
                    )
                    store.commit_shot(
                        shot_id,
                        self._copy_outputs(
                            base_files,
                            scan,
                            base_artifact.native_shape,
                        ),
                        calibration=calibration,
                    )
                elif not scan.needs_model:
                    calibration = self._empty_calibration(
                        scan,
                        mode="base_fallback",
                        fallback_reason="source_no_range",
                    )
                    store.commit_shot(
                        shot_id,
                        self._copy_outputs(
                            base_files,
                            scan,
                            base_artifact.native_shape,
                        ),
                        calibration=calibration,
                    )
                else:
                    if postprocessor is None:
                        raise AssertionError("VDPP model is missing for a non-degenerate shot")
                    self._run_model_shot(
                        postprocessor=postprocessor,
                        store=store,
                        stable_root=stable_root,
                        base_files=base_files,
                        scan=scan,
                        native_shape=base_artifact.native_shape,
                    )
                completed_frames += scan.length
                self._report_progress(
                    progress_tracker,
                    f"Stabilized {completed_frames}/{len(base_files)} depth maps",
                    completed=completed_frames,
                    total=len(base_files),
                )
            return store.finalize()
        except BaseException as exc:
            print(f"VDPP temporal stabilization failed: {exc}", file=sys.stderr)
            raise
        finally:
            if postprocessor is not None:
                active_exception = sys.exc_info()[0] is not None
                try:
                    postprocessor.release()
                except BaseException as release_error:
                    if active_exception:
                        print(
                            f"VDPP postprocessor release also failed: {release_error}",
                            file=sys.stderr,
                        )
                    else:
                        raise
