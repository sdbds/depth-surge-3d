"""Unified streaming stereo generator over relative disparity and metric geometry."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
import traceback
from typing import Any, Callable, Literal, cast

import cv2
import numpy as np

from ...core.constants import PREVIEW_FRAME_SAMPLE_RATE
from ...core.depth_contract import canonical_json_hash
from ...io import metric_npz
from ...io.durable_publish import atomic_write_bytes, atomic_write_json
from ...rendering.quality.diagnostic_transaction import (
    StreamingDiagnosticsAccumulator,
    StreamingDiagnosticsStageContext,
)
from ...rendering.quality.renderer import (
    finish_splatted_quality_frame,
    prepare_metric_quality_frame,
    prepare_relative_quality_frame,
    splat_prepared_metric_quality_frame,
    splat_prepared_relative_quality_frame,
)
from ...rendering.quality.types import (
    CompactStereoRenderResult,
    MetricStereoPrimitiveInput,
    QualityStereoControls,
)
from ...rendering.stereo_renderer import (
    StereoRenderer,
    StereoRenderSettings,
    StereoSplatSettings,
)
from ...utils.imaging.image_processing import (
    CENTER_CROP_ALGORITHM_VERSION,
    calculate_center_crop_dimensions,
)
from ...utils.imaging.png_header import read_png_header
from .metric_geometry import MetricGeometryFrame
from .stereo_control import (
    StereoPipelineCapacityError,
    StreamingStereoPipelineStats,
    run_streaming_stereo_pipeline,
    select_fast_pipeline_capacity,
    select_quality_pipeline_capacity,
)
from .stereo_input_manifest import (
    StereoInputManifestV1,
    StereoInputRecord,
    publish_validated_stereo_input_manifest_v1,
)


STEREO_STAGE_SCHEMA_VERSION = 1
STEREO_STAGE_ALGORITHM_VERSION = "torch-horizontal-16x-zbuffer-v3"
QUALITY_STEREO_ALGORITHM_VERSION = "torch-horizontal-16x-rgb-geodesic-repair-v11"
METRIC_PROJECTION_ALGORITHM_VERSION = "crop-aware-metric-pinhole-v2"
METRIC_CLAMP_STATS_SCHEMA_VERSION = 1
METRIC_STEREO_COMPLETION_ERROR = "metric stereo requires completed clip-global convergence metadata"
MIN_STEREO_IO_WORKERS = 1
MAX_STEREO_IO_WORKERS = 16


def validate_stereo_io_workers(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("stereo_io_workers must be an integer or None")
    if not MIN_STEREO_IO_WORKERS <= value <= MAX_STEREO_IO_WORKERS:
        raise ValueError(
            f"stereo_io_workers must be between {MIN_STEREO_IO_WORKERS} and {MAX_STEREO_IO_WORKERS}"
        )
    return value


def _default_stereo_io_workers() -> int:
    return max(2, min(8, (os.cpu_count() or 4) // 2))


def _encode_png_bytes(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("cv2.imencode('.png') failed")
    return encoded.tobytes()


def _atomic_write_png(path: Path, image: np.ndarray) -> None:
    atomic_write_bytes(path, _encode_png_bytes(image), sync=False)


def _frame_png_valid(path: Path, expected_shape: tuple[int, int]) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    header = read_png_header(path)
    if header is None:
        return False
    return (header.height, header.width) == expected_shape and header.bit_depth == 8


class StereoPairGenerator:
    """Generates stereo left/right frames in Fast or Quality mode (file-backed or streamed)."""

    def __init__(self, verbose: bool = False, renderer: StereoRenderer | None = None) -> None:
        self.verbose = verbose
        self._renderer = renderer
        self.last_pipeline_stats: StreamingStereoPipelineStats | None = None
        self.last_metric_clamp_summary: dict[str, Any] | None = None

    @property
    def renderer(self) -> StereoRenderer:
        if self._renderer is None:
            self._renderer = StereoRenderer()
        return self._renderer

    @staticmethod
    def _occlusion_fill(settings: dict[str, Any]) -> Literal["none", "background"]:
        value = str(settings.get("occlusion_fill", "background"))
        if value not in {"none", "background"}:
            raise ValueError(f"Unsupported occlusion_fill: {value}")
        return cast(Literal["none", "background"], value)

    @staticmethod
    def _cancellation_predicate(progress_tracker: Any) -> Callable[[], bool] | None:
        if progress_tracker is None:
            return None
        checker = getattr(type(progress_tracker), "is_cancelled", None)
        if callable(checker):
            return lambda: bool(progress_tracker.is_cancelled())
        return None

    @staticmethod
    def _stage_metadata_payload(
        *,
        manifest: StereoInputManifestV1,
        settings: dict[str, Any],
        render_mode: Literal["fast", "quality"],
        occlusion_fill: Literal["none", "background"],
    ) -> dict[str, Any]:
        algorithm = (
            QUALITY_STEREO_ALGORITHM_VERSION
            if render_mode == "quality"
            else STEREO_STAGE_ALGORITHM_VERSION
        )
        payload: dict[str, Any] = {
            "schema_version": STEREO_STAGE_SCHEMA_VERSION,
            "stage": "stereo",
            "algorithm_version": algorithm,
            "render_mode": render_mode,
            "geometry_mode": manifest.geometry_mode,
            "frame_names": [f"{name}.png" for name in manifest.frame_names()],
            "render_shape": list(manifest.render_shape),
            "source_fingerprint": manifest.source_geometry_fingerprint,
            "occlusion_fill": occlusion_fill,
        }
        if render_mode == "quality":
            payload["occlusion_fill_max_px"] = int(settings.get("occlusion_fill_max_px", 8))
        if manifest.geometry_mode == "relative":
            payload["stereo_strength"] = float(settings.get("stereo_strength", 2.0))
            payload["convergence"] = float(settings.get("convergence", 0.5))
            payload["encoding_scale"] = manifest.relative_encoding_scale
        else:
            requested = settings.get("metric_convergence_distance", "auto")
            effective_conv = (
                manifest.resolved_auto_distance_m if requested == "auto" else float(requested)
            )
            payload["projection_algorithm"] = METRIC_PROJECTION_ALGORITHM_VERSION
            payload["crop_algorithm"] = CENTER_CROP_ALGORITHM_VERSION
            payload["virtual_baseline_mm"] = float(settings.get("virtual_baseline_mm", 63.0))
            payload["convergence_distance_m"] = effective_conv
            payload["max_disparity_percent"] = float(settings.get("max_disparity_percent", 2.0))
            payload["crop_factor"] = float(settings.get("crop_factor", 1.0))
        payload["fingerprint"] = canonical_json_hash(payload)
        return payload

    @staticmethod
    def _read_existing_metadata(left_dir: Path) -> dict[str, Any] | None:
        meta_path = left_dir / "metadata.json"
        if not meta_path.is_file():
            return None
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _reset_downstream_stages(directories: dict[str, Path]) -> None:
        for key in (
            "left_distorted",
            "right_distorted",
            "left_cropped",
            "right_cropped",
            "left_upscaled",
            "right_upscaled",
            "vr_frames",
        ):
            stage_dir = directories.get(key)
            if stage_dir is None or not stage_dir.is_dir():
                continue
            for file_path in stage_dir.glob("*.png"):
                file_path.unlink(missing_ok=True)
            (stage_dir / "metadata.json").unlink(missing_ok=True)

    @staticmethod
    def _clear_stereo_outputs(left_dir: Path, right_dir: Path) -> None:
        for directory in (left_dir, right_dir):
            directory.mkdir(parents=True, exist_ok=True)
            for file_path in directory.glob("*.png"):
                file_path.unlink(missing_ok=True)
            for temp_path in directory.glob("*.tmp"):
                temp_path.unlink(missing_ok=True)
        (left_dir / "metadata.json").unlink(missing_ok=True)

    def create_stereo_pairs_from_files(
        self,
        frame_files: list[Path],
        depth_files: list[Path],
        directories: dict[str, Path],
        settings: dict[str, Any],
        progress_tracker: Any = None,
    ) -> bool:
        """Convenience wrapper building a lightweight manifest and running the unified pipeline."""

        geometry_mode = cast(
            Literal["relative", "metric_camera"],
            str(settings.get("stereo_geometry_mode", "relative")),
        )
        if geometry_mode == "metric_camera" and (
            not frame_files or len(frame_files) != len(depth_files)
        ):
            raise ValueError(METRIC_STEREO_COMPLETION_ERROR)
        if len(frame_files) != len(depth_files):
            print(
                f"Error: Frame/depth count mismatch: {len(frame_files)} frames, "
                f"{len(depth_files)} depth maps"
            )
            return False
        job_root = directories.get("base") or directories["left_frames"].parent
        manifest = publish_validated_stereo_input_manifest_v1(
            job_root=job_root,
            guide_files=frame_files,
            geometry_files=depth_files,
            geometry_mode=geometry_mode,
        )
        return self.create_stereo_pairs_from_manifest(
            manifest,
            directories,
            settings,
            progress_tracker=progress_tracker,
        )

    def create_stereo_pairs_from_manifest(  # noqa: C901
        self,
        stereo_input: StereoInputManifestV1,
        directories: dict[str, Path],
        settings: dict[str, Any],
        progress_tracker: Any = None,
        *,
        frame_consumer: Callable[[str, np.ndarray, np.ndarray], None] | None = None,
        frame_consumer_memory_bytes: int = 0,
    ) -> bool:
        """Decode, render, and write (or stream) all stereo pairs cleanly."""

        self.last_pipeline_stats = None
        self.last_metric_clamp_summary = None
        try:
            render_mode = cast(
                Literal["fast", "quality"],
                str(settings.get("stereo_render_mode", "fast")),
            )
            occlusion_fill = self._occlusion_fill(settings)
            render_shape = stereo_input.render_shape
            height, width = render_shape
            is_metric = stereo_input.geometry_mode == "metric"
            is_streamed = frame_consumer is not None
            save_masks = bool(settings.get("save_quality_masks", False)) and not is_streamed

            left_dir = directories.get("left_frames")
            right_dir = directories.get("right_frames")
            if not is_streamed:
                if left_dir is None or right_dir is None:
                    raise ValueError(
                        "File-backed stereo generation requires left/right directories"
                    )
                left_dir.mkdir(parents=True, exist_ok=True)
                right_dir.mkdir(parents=True, exist_ok=True)

            expected_meta = self._stage_metadata_payload(
                manifest=stereo_input,
                settings=settings,
                render_mode=render_mode,
                occlusion_fill=occlusion_fill,
            )

            records_to_render: list[StereoInputRecord] = list(stereo_input.records)
            skipped_count = 0
            if not is_streamed:
                assert left_dir is not None and right_dir is not None
                existing_meta = self._read_existing_metadata(left_dir)
                meta_matches = (
                    existing_meta is not None
                    and existing_meta.get("fingerprint") == expected_meta["fingerprint"]
                )
                if meta_matches:
                    pending: list[StereoInputRecord] = []
                    for rec in stereo_input.records:
                        left_path = left_dir / f"{rec.frame_name}.png"
                        right_path = right_dir / f"{rec.frame_name}.png"
                        if _frame_png_valid(left_path, render_shape) and _frame_png_valid(
                            right_path, render_shape
                        ):
                            skipped_count += 1
                        else:
                            pending.append(rec)
                    records_to_render = pending
                    if not records_to_render:
                        if self.verbose:
                            print(
                                f"  Reusing {skipped_count} existing stereo pairs "
                                f"({render_mode}/{stereo_input.geometry_mode})"
                            )
                        return True
                else:
                    self._clear_stereo_outputs(left_dir, right_dir)
                    self._reset_downstream_stages(directories)

            # Prepare rendering settings
            if not is_metric:
                encoding_scale = np.float32(stereo_input.relative_encoding_scale or 65535.0)
                relative_settings = StereoRenderSettings(
                    stereo_strength=float(settings.get("stereo_strength", 2.0)),
                    convergence=float(settings.get("convergence", 0.5)),
                    occlusion_fill=occlusion_fill,
                    stereo_render_mode=render_mode,
                    occlusion_fill_max_px=int(settings.get("occlusion_fill_max_px", 8)),
                )
            else:
                requested_conv = settings.get("metric_convergence_distance", "auto")
                effective_conv = (
                    float(stereo_input.resolved_auto_distance_m or 2.0)
                    if requested_conv == "auto"
                    else float(requested_conv)
                )
                retained_crop_width, _ = calculate_center_crop_dimensions(
                    width,
                    height,
                    float(settings.get("crop_factor", 1.0)),
                )
                virtual_baseline_mm = float(settings.get("virtual_baseline_mm", 63.0))
                max_disparity_percent = float(settings.get("max_disparity_percent", 2.0))
                metric_controls = QualityStereoControls(
                    occlusion_fill=occlusion_fill,
                    occlusion_fill_max_px=int(settings.get("occlusion_fill_max_px", 8)),
                )
                metric_splat_settings = StereoSplatSettings(
                    max_eye_shift_fraction=(
                        max_disparity_percent / 100.0 * (retained_crop_width / width) / 2.0
                    ),
                    occlusion_fill=occlusion_fill,
                )

            # Configure workers and dynamic RAM-scaled queue capacity
            io_workers_opt = validate_stereo_io_workers(settings.get("stereo_io_workers"))
            io_workers = (
                io_workers_opt if io_workers_opt is not None else _default_stereo_io_workers()
            )
            if render_mode == "quality":
                capacity = select_quality_pipeline_capacity(
                    render_shape,
                    extra_per_frame_bytes=frame_consumer_memory_bytes,
                )
            else:
                capacity = select_fast_pipeline_capacity(
                    render_shape,
                    extra_per_frame_bytes=frame_consumer_memory_bytes,
                )
            is_quality = render_mode == "quality"
            if is_quality and io_workers_opt is None:
                cpu_count = os.cpu_count() or 4
                decode_workers = min(capacity, max(4, min(10, cpu_count // 3)))
                postprocess_workers: int | None = min(capacity, max(2, min(8, cpu_count // 4)))
            else:
                decode_workers = min(io_workers, capacity)
                postprocess_workers = min(io_workers, capacity) if is_quality else None
            write_workers = 1 if is_streamed else min(io_workers, capacity)

            diag_context = StreamingDiagnosticsStageContext(
                job_root=stereo_input.job_root,
                frame_count=stereo_input.frame_count,
                frame_names=stereo_input.frame_names,
                render_shape=render_shape,
                render_mode=render_mode,
                geometry_mode=stereo_input.geometry_mode,
                occlusion_fill=occlusion_fill,
                mask_payloads_enabled=save_masks,
                rgb_stage_fingerprint=expected_meta["fingerprint"],
            )
            diagnostics = StreamingDiagnosticsAccumulator(diag_context)
            diag_lock = threading.Lock()
            renderer = self.renderer
            cancelled = self._cancellation_predicate(progress_tracker)
            total_frames = stereo_input.frame_count
            stream_reader = (
                stereo_input.frame_source.open_reader(
                    window_size=capacity + 8,
                    consume_once=True,
                )
                if stereo_input.frame_source is not None
                else None
            )

            def decode_record(record: StereoInputRecord) -> Any:
                if stream_reader is not None:
                    frame = stream_reader.get_frame(record.frame_index - stereo_input.first_index)
                else:
                    frame = cv2.imread(str(record.guide_path), cv2.IMREAD_COLOR)
                if frame is None or frame.shape[:2] != render_shape:
                    raise ValueError(f"Failed to decode guide frame: {record.guide_path}")
                if not is_metric:
                    raw_disp = cv2.imread(str(record.geometry_path), cv2.IMREAD_UNCHANGED)
                    if raw_disp is None or raw_disp.dtype != np.uint16 or raw_disp.ndim != 2:
                        raise ValueError(
                            f"Failed to decode 16-bit disparity PNG: {record.geometry_path}"
                        )
                    canonical = np.ascontiguousarray(
                        raw_disp.astype(np.float32) / encoding_scale,
                        dtype=np.float32,
                    )
                    if is_quality:
                        return prepare_relative_quality_frame(
                            frame=frame,
                            canonical=canonical,
                            settings=relative_settings,
                        )
                    return frame, canonical
                inv, valid, focal = metric_npz.read_metric_npz(
                    record.geometry_path,
                    expected_shape=stereo_input.native_shape,
                )
                metric_frame = MetricGeometryFrame._from_owned(inv, valid, focal)
                primitive = MetricStereoPrimitiveInput(
                    metric=metric_frame,
                    virtual_baseline_mm=np.float64(virtual_baseline_mm),
                    convergence_distance_m=np.float64(effective_conv),
                    max_disparity_percent=np.float64(max_disparity_percent),
                    retained_crop_width=retained_crop_width,
                )
                if is_quality:
                    return prepare_metric_quality_frame(
                        frame=frame,
                        primitives=primitive,
                        controls=metric_controls,
                    )
                return frame, primitive

            def render_record(record: StereoInputRecord, decoded: Any) -> Any:
                _ = record
                if is_quality:
                    if not is_metric:
                        return splat_prepared_relative_quality_frame(
                            renderer=renderer,
                            prepared=decoded,
                        )
                    return splat_prepared_metric_quality_frame(
                        renderer=renderer,
                        prepared=decoded,
                    )
                frame, geom_payload = decoded
                if not is_metric:
                    compact = renderer.render_relative_fast_compact(
                        frame,
                        geom_payload,
                        relative_settings,
                    )
                else:
                    compact = renderer.render_metric_primitives_fast_compact(
                        frame,
                        geom_payload,
                        metric_splat_settings,
                    )
                with diag_lock:
                    diagnostics.record_compact(compact, metric_stats=compact.metric_stats)
                return compact

            def postprocess_quality_record(
                record: StereoInputRecord, splatted: Any
            ) -> CompactStereoRenderResult:
                _ = record
                compact = finish_splatted_quality_frame(
                    splatted,
                    compute_component_count=save_masks,
                )
                with diag_lock:
                    diagnostics.record_compact(compact, metric_stats=compact.metric_stats)
                return compact

            def write_record(record: StereoInputRecord, compact: CompactStereoRenderResult) -> None:
                if frame_consumer is not None:
                    frame_consumer(record.frame_name, compact.left_image, compact.right_image)
                    return
                assert left_dir is not None and right_dir is not None
                _atomic_write_png(left_dir / f"{record.frame_name}.png", compact.left_image)
                _atomic_write_png(right_dir / f"{record.frame_name}.png", compact.right_image)
                if save_masks:
                    mask_dir = (
                        stereo_input.job_root
                        / "04_stereo_diagnostics"
                        / "frames"
                        / record.frame_name
                    )
                    mask_dir.mkdir(parents=True, exist_ok=True)
                    _atomic_write_png(mask_dir / "left_coverage.png", compact.left_coverage_count)
                    _atomic_write_png(mask_dir / "left_repair.png", compact.left_repair_bits)
                    _atomic_write_png(mask_dir / "right_coverage.png", compact.right_coverage_count)
                    _atomic_write_png(mask_dir / "right_repair.png", compact.right_repair_bits)

            def on_completed(record: StereoInputRecord, rendered_so_far: int) -> None:
                done = skipped_count + rendered_so_far
                if progress_tracker is None:
                    return
                try:
                    progress_tracker.update_progress(
                        f"Generated stereo pair {done}/{total_frames}",
                        phase="stereo_generation",
                        frame_num=done,
                        step_name="Stereo Pair Creation",
                        step_progress=done,
                        step_total=total_frames,
                    )
                    if (
                        not is_streamed
                        and left_dir is not None
                        and right_dir is not None
                        and (
                            done == 1
                            or done == total_frames
                            or done % PREVIEW_FRAME_SAMPLE_RATE == 0
                        )
                        and hasattr(progress_tracker, "send_preview_frame")
                    ):
                        left_img = cv2.imread(str(left_dir / f"{record.frame_name}.png"))
                        right_img = cv2.imread(str(right_dir / f"{record.frame_name}.png"))
                        if left_img is not None and right_img is not None:
                            progress_tracker.send_preview_frame(
                                left_img,
                                "stereo_left",
                                frame_num=done,
                            )
                except InterruptedError:
                    raise
                except Exception:
                    pass

            started = time.perf_counter()
            prev_cv_threads = cv2.getNumThreads() if (is_quality and capacity > 1) else None
            if prev_cv_threads is not None and prev_cv_threads != 1:
                cv2.setNumThreads(1)
            try:
                stats = run_streaming_stereo_pipeline(
                    records_to_render,
                    decode_fn=decode_record,
                    render_fn=render_record,
                    postprocess_fn=postprocess_quality_record if is_quality else None,
                    postprocess_workers=postprocess_workers,
                    write_fn=write_record,
                    capacity=capacity,
                    decode_workers=decode_workers,
                    write_workers=write_workers,
                    ordered_writes=is_streamed,
                    on_completed=on_completed,
                    is_cancelled=cancelled,
                )
            finally:
                if stream_reader is not None:
                    stream_reader.close()
                if prev_cv_threads is not None and prev_cv_threads != 1:
                    cv2.setNumThreads(prev_cv_threads)
            self.last_pipeline_stats = StreamingStereoPipelineStats(
                frame_count=total_frames,
                render_count=stats.render_count,
                skip_count=skipped_count,
                capacity=stats.capacity,
                decode_workers=stats.decode_workers,
                write_workers=stats.write_workers,
                peak_in_flight=stats.peak_in_flight,
                elapsed_seconds=time.perf_counter() - started,
            )

            self.last_metric_clamp_summary = diagnostics.metric_clamp_summary()
            if not is_streamed:
                assert left_dir is not None
                atomic_write_json(
                    left_dir / "metadata.json",
                    expected_meta,
                    indent=2,
                    sort_keys=True,
                    sync=True,
                )
            diagnostics.write_summary()
            return True
        except (InterruptedError, StereoPipelineCapacityError):
            raise
        except Exception as error:
            print(f"Error creating stereo pairs: {error}")
            if self.verbose:
                traceback.print_exc()
            return False
