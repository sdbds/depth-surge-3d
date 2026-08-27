"""Bounded stereo generation over canonical disparity maps."""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import shutil
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, cast

import cv2
import numpy as np

from ...core.constants import PREVIEW_FRAME_SAMPLE_RATE
from ...core.depth_contract import canonical_json_hash
from ...core.resource_errors import QualityHostBudgetError
from ...core.render_disparity import validate_render_disparity_input
from ...io import metric_npz
from ...io.durable_publish import OwnedSiblingTemporary, publication_temporary_path
from ...io.nofollow_io import (
    AcquiredRoot,
    SafeTreeMutation,
    open_regular_beneath,
)
from ...io.content_manifest import (
    StreamedQualityInputManifest,
    build_streamed_quality_input_manifest,
)
from ...io.streamed_canonical_json import (
    ReplayableJsonArray,
    StreamedCanonicalObject,
    StreamedPrettyJsonValue,
    build_streamed_canonical_object,
    build_streamed_pretty_json_value,
)
from ...utils.imaging.png_header import png_header_matches, read_png_header
from ...utils.imaging.image_processing import (
    CENTER_CROP_ALGORITHM_VERSION,
    calculate_center_crop_dimensions,
)
from ...rendering.stereo_geometry import (
    MetricProjectionStats,
    build_metric_geometry,
)
from ...rendering.stereo_renderer import (
    StereoRenderer,
    StereoRenderSettings,
    StereoSplatSettings,
)
from ...rendering.quality.diagnostic_transaction import (
    DiagnosticsStageContext,
    PreparedQualityFramePublication,
    QualityFramePublication,
    audit_diagnostics_stage,
    begin_diagnostics_stage,
    canonical_json_bytes,
    complete_diagnostics_stage,
    migrate_frame_manifests_sync,
    prepare_quality_frame,
    publish_quality_frame,
    StreamingDiagnosticsStageContext,
    begin_streaming_diagnostics_stage,
    classify_streaming_diagnostics_root,
    classify_streaming_diagnostics_frame,
    complete_streaming_diagnostics_stage,
    migrate_streaming_frame_manifest,
    prune_streaming_unreferenced_masks,
    streaming_diagnostics_aggregates_match,
)
from ...rendering.quality.renderer import (
    QUALITY_HOST_BUDGET_BYTES,
    QUALITY_RUNTIME_OVERHEAD_BYTES,
    _quality_public_memory_preflight,
    render_metric_primitives_compact,
    render_relative_quality_compact,
)
from ...rendering.quality.types import (
    MetricStereoPrimitiveInput,
    QualityStereoControls,
    RelativeStereoPrimitiveInput,
)
from .metric_geometry import MetricGeometryFrame, MetricGeometryStore
from .stereo_control import (
    STEREO_CONTROL_OVERHEAD,
    StereoAction,
    StereoControlError,
    StereoControlStateError,
    StereoInputIdentityMismatch,
    StereoProviderChangedError,
    StereoProviderItem,
    StereoControlAuditPlan,
    StereoStageInputChangedError,
    StreamingStereoPipelineStats,
    audit_stereo_provider,
    iter_stereo_action,
    quality_pipeline_capacity,
    run_streaming_stereo_pipeline,
    select_fast_pipeline_capacity,
    validate_stereo_action_replay,
    validate_stereo_provider_replay,
    PackedStereoActions,
)
from .stereo_input_manifest import (
    StereoInputManifestV1,
    load_stereo_input_manifest_v1,
)
from .stereo_disk import (
    StereoStageDiskPlan,
    StereoStageDiskPlanBuilder,
    admit_stereo_stage_disk,
)

STEREO_STAGE_SCHEMA_VERSION = 1
STEREO_STAGE_ALGORITHM_VERSION = "torch-horizontal-16x-zbuffer-v3"
QUALITY_STEREO_ALGORITHM_VERSION = "torch-horizontal-16x-rgb-geodesic-repair-v10"
METRIC_PROJECTION_ALGORITHM_VERSION = "crop-aware-metric-pinhole-v2"
METRIC_CLAMP_STATS_SCHEMA_VERSION = 1
METRIC_STEREO_COMPLETION_ERROR = "metric stereo requires completed clip-global convergence metadata"
STEREO_HOST_BUDGET = 512 * 1024 * 1024
HOST_STEREO_BYTES_PER_PIXEL = 24
HOST_SLOT_OVERHEAD = 1024 * 1024
MIN_STEREO_IO_WORKERS = 1
MAX_STEREO_IO_WORKERS = 16
STEREO_READ_ONLY_RESTART_LIMIT = 8
QUALITY_CONTENT_STREAM_BYTES = 1024 * 1024
QUALITY_JSON_STREAM_BYTES = 1024 * 1024
QUALITY_IO_THREAD_BYTES = 6 * 1024 * 1024


def _quality_stage_memory_phases(
    *,
    control: StereoControlAuditPlan,
    render_shape: tuple[int, int],
    native_shape: tuple[int, int],
    geometry_mode: Literal["relative", "metric"],
    occlusion_fill: Literal["none", "background"],
    consolidation_planned: bool,
) -> dict[str, int]:
    """Calculate every mutually exclusive Quality file-stage host phase."""

    output_pixels = int(render_shape[0]) * int(render_shape[1])
    histogram_bytes = 272 * (int(render_shape[1]) + 1)
    runtime = QUALITY_RUNTIME_OVERHEAD_BYTES
    phases: dict[str, int] = {}
    content_payload = (
        5 * control.maximum_source_native_pixels
        if geometry_mode == "metric"
        else 0
    )
    content_peak = (
        content_payload
        + QUALITY_CONTENT_STREAM_BYTES
        + runtime
        + STEREO_CONTROL_OVERHEAD
    )
    phases["quality-input-content-audit"] = content_peak
    if consolidation_planned:
        phases["quality-input-content-revalidation"] = content_peak

    if control.render_count:
        render_phases = _quality_public_memory_preflight(
            render_shape=render_shape,
            native_shape=native_shape,
            occlusion_fill=occlusion_fill,
            materialize_public_masks=False,
        )
        if geometry_mode == "metric":
            decode_peak = (
                3 * output_pixels
                + 5 * control.maximum_native_pixels
                + control.maximum_compressed_payload_bytes
                + runtime
            )
        else:
            decode_peak = (
                3 * output_pixels
                + 2 * control.maximum_native_pixels
                + control.maximum_compressed_payload_bytes
                + runtime
            )
        writer_peak = (
            23 * output_pixels
            + histogram_bytes
            + control.maximum_frame_json_bound
            + runtime
        )
        lifecycle_peak = max(decode_peak, writer_peak, *render_phases.values())
        phases["quality-file-pipeline"] = (
            lifecycle_peak + QUALITY_IO_THREAD_BYTES + STEREO_CONTROL_OVERHEAD
        )

    if consolidation_planned:
        metric_fraction_bound = (
            33 * control.generation.frame_count + 2
            if geometry_mode == "metric"
            else 0
        )
        root_json_bound = (
            control.maximum_frame_json_bound
            + control.frame_name_json_array_bytes
            + metric_fraction_bound
        )
        phases["quality-consolidation"] = (
            histogram_bytes
            + control.maximum_frame_json_bound
            + root_json_bound
            + QUALITY_JSON_STREAM_BYTES
            + runtime
            + STEREO_CONTROL_OVERHEAD
        )

    if control.migrate_count:
        phases["quality-manifest-migration"] = (
            STEREO_CONTROL_OVERHEAD
            + 2 * control.maximum_frame_json_bound
            + QUALITY_JSON_STREAM_BYTES
        )
    return phases


def _quality_stage_memory_preflight(**kwargs: Any) -> dict[str, int]:
    phases = _quality_stage_memory_phases(**kwargs)
    phase, required = max(phases.items(), key=lambda item: item[1])
    if required > QUALITY_HOST_BUDGET_BYTES:
        raise QualityHostBudgetError.for_limit(
            phase,
            required,
            QUALITY_HOST_BUDGET_BYTES,
        )
    return phases


def _build_stereo_stage_disk_plan(
    *,
    provider: Any,
    control: StereoControlAuditPlan,
    job_root: Path,
    render_shape: tuple[int, int],
    render_mode: Literal["fast", "quality"],
    geometry_mode: Literal["relative", "metric"],
    keep_masks: bool,
    left_dir: Path,
    stage_document: StreamedCanonicalObject | StreamedPrettyJsonValue,
    quality_content: StreamedQualityInputManifest | None,
) -> StereoStageDiskPlan:
    """Derive all pending frame/root/atomic charges without retaining item paths."""

    builder = StereoStageDiskPlanBuilder()
    height, width = render_shape
    rgb_bound = math.ceil(1.25 * height * width * 3)
    mask_bound = math.ceil(1.25 * height * width)
    diagnostics_root = job_root / "04_stereo_diagnostics"

    for item in iter_stereo_action(
        provider,
        control,
        StereoAction.RENDER,
        mutation_started=False,
    ):
        frame_root = diagnostics_root / "frames" / item.frame_name
        transaction_entries = [
            (item.left_path, rgb_bound),
            (item.right_path, rgb_bound),
            (frame_root / "stats.json", item.frame_json_bound),
            (frame_root / "manifest.json", item.frame_json_bound),
        ]
        if keep_masks:
            transaction_entries.extend(
                (frame_root / name, mask_bound)
                for name in (
                    "left_coverage.png",
                    "left_repair.png",
                    "right_coverage.png",
                    "right_repair.png",
                )
            )
        for path, logical_bytes in transaction_entries:
            builder.add_payload(path, logical_bytes)
            builder.add_directory_entries(path, 1)
        builder.add_transaction(transaction_entries)

    for item in iter_stereo_action(
        provider,
        control,
        StereoAction.MIGRATE,
        mutation_started=False,
    ):
        path = diagnostics_root / "frames" / item.frame_name / "manifest.json"
        builder.add_payload(path, item.frame_json_bound)
        builder.add_directory_entries(path, 1)
        builder.add_transaction(((path, item.frame_json_bound),))

    frame_bound = control.maximum_frame_json_bound
    metric_fraction_bound = (
        33 * control.generation.frame_count + 2 if geometry_mode == "metric" else 0
    )
    root_json_bound = (
        frame_bound + control.frame_name_json_array_bytes + metric_fraction_bound
    )
    jsonl_bound = control.generation.frame_count * (frame_bound + 1)
    clamp_bound = control.frame_name_json_array_bytes + metric_fraction_bound + 1024 * 1024
    root_artifacts: list[tuple[Path, int]] = [
        (left_dir / "metadata.json", stage_document.raw_byte_count),
        (diagnostics_root / "metadata.json", root_json_bound),
        (diagnostics_root / "stereo_coverage_frames.jsonl", jsonl_bound),
        (diagnostics_root / "stereo_coverage_summary.json", root_json_bound),
    ]
    if geometry_mode == "metric":
        root_artifacts.append((left_dir / "clamp_summary.json", clamp_bound))
    if render_mode == "quality":
        if quality_content is None:
            raise ValueError("Quality disk planning requires its content identity")
        root_artifacts.append(
            (diagnostics_root / "quality_input_manifest.json", quality_content.raw_byte_count)
        )
    for path, logical_bytes in root_artifacts:
        builder.add_payload(path, int(logical_bytes), count=2)
        builder.add_directory_entries(path, 1)

    builder.add_first_mutation_transaction(
        diagnostics_root / "metadata.json",
        root_json_bound,
    )
    builder.add_first_mutation_transaction(
        left_dir / "metadata.json",
        stage_document.raw_byte_count,
    )
    if quality_content is not None:
        builder.add_first_mutation_transaction(
            diagnostics_root / "quality_input_manifest.json",
            quality_content.raw_byte_count,
        )

    return builder.build()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    owned = OwnedSiblingTemporary.create(path)
    try:
        if owned.handle.write(raw) != len(raw):
            raise OSError("short stereo JSON write")
        owned.sync()
        owned.replace()
    finally:
        owned.cleanup()


def _atomic_write_canonical_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(payload)
    owned = OwnedSiblingTemporary.create(path)
    try:
        if owned.handle.write(raw) != len(raw):
            raise OSError("short canonical stereo JSON write")
        owned.sync()
        owned.replace()
    finally:
        owned.cleanup()


def validate_stereo_io_workers(workers: int) -> int:
    """Validate the bounded decode/write worker count."""

    if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)):
        raise ValueError("stereo_io_workers must be an integer in 1..16")
    value = int(workers)
    if not MIN_STEREO_IO_WORKERS <= value <= MAX_STEREO_IO_WORKERS:
        raise ValueError("stereo_io_workers must be an integer in 1..16")
    return value


def _default_stereo_io_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return min(4, max(1, cpu_count - 2))


def calculate_stereo_pipeline_capacity(
    render_width: int,
    render_height: int,
    stereo_io_workers: int,
) -> int:
    """Derive queue and lifecycle-permit capacity from the fixed host budget."""

    workers = validate_stereo_io_workers(stereo_io_workers)
    if render_width <= 0 or render_height <= 0:
        raise ValueError("Render width and height must be positive")
    slot_bytes = render_width * render_height * HOST_STEREO_BYTES_PER_PIXEL + HOST_SLOT_OVERHEAD
    memory_slots = STEREO_HOST_BUDGET // slot_bytes
    capacity = min(2 * workers, memory_slots)
    if capacity < 1:
        raise MemoryError(
            f"Stereo frame {render_width}x{render_height} cannot fit one lifecycle "
            f"slot: required {slot_bytes} host bytes; budget is "
            f"{STEREO_HOST_BUDGET} bytes"
        )
    return capacity


@dataclass(frozen=True)
class StereoPipelineStats:
    """Observable host-memory and I/O backpressure accounting."""

    permit_count: int
    queue_capacity: int
    permits_acquired: int
    permits_released: int
    active_permits: int
    max_active_permits: int
    permit_wait_seconds: float
    queue_wait_seconds: float
    pipeline_wall_seconds: float
    writer_busy_seconds: float
    decoded_frames: int
    rendered_frames: int
    written_frames: int


class _PipelineMetrics:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.lock = threading.Lock()
        self.permits_acquired = 0
        self.permits_released = 0
        self.active_permits = 0
        self.max_active_permits = 0
        self.permit_wait_seconds = 0.0
        self.queue_wait_seconds = 0.0
        self.pipeline_wall_seconds = 0.0
        self.writer_busy_seconds = 0.0
        self.decoded_frames = 0
        self.rendered_frames = 0
        self.written_frames = 0

    def permit_acquired(self, waited: float) -> None:
        with self.lock:
            self.permits_acquired += 1
            self.active_permits += 1
            self.max_active_permits = max(
                self.max_active_permits,
                self.active_permits,
            )
            self.permit_wait_seconds += waited

    def permit_released(self) -> None:
        with self.lock:
            self.permits_released += 1
            self.active_permits -= 1

    def queue_waited(self, waited: float) -> None:
        with self.lock:
            self.queue_wait_seconds += waited

    def writer_busy(self, elapsed: float) -> None:
        with self.lock:
            self.writer_busy_seconds += elapsed

    def pipeline_finished(self, elapsed: float) -> None:
        with self.lock:
            self.pipeline_wall_seconds = elapsed

    def increment(self, field: str) -> None:
        with self.lock:
            setattr(self, field, getattr(self, field) + 1)

    def snapshot(self) -> StereoPipelineStats:
        with self.lock:
            return StereoPipelineStats(
                permit_count=self.capacity,
                queue_capacity=self.capacity,
                permits_acquired=self.permits_acquired,
                permits_released=self.permits_released,
                active_permits=self.active_permits,
                max_active_permits=self.max_active_permits,
                permit_wait_seconds=self.permit_wait_seconds,
                queue_wait_seconds=self.queue_wait_seconds,
                pipeline_wall_seconds=self.pipeline_wall_seconds,
                writer_busy_seconds=self.writer_busy_seconds,
                decoded_frames=self.decoded_frames,
                rendered_frames=self.rendered_frames,
                written_frames=self.written_frames,
            )


class _LifecyclePermits:
    def __init__(self, capacity: int, metrics: _PipelineMetrics) -> None:
        self._semaphore = threading.BoundedSemaphore(capacity)
        self._metrics = metrics

    def acquire(self) -> _LifecyclePermitLease:
        started = time.perf_counter()
        self._semaphore.acquire()
        self._metrics.permit_acquired(time.perf_counter() - started)
        return _LifecyclePermitLease(self)

    def release(self) -> None:
        self._semaphore.release()
        self._metrics.permit_released()


class _LifecyclePermitLease:
    """Linear ownership token for one provider-to-publication lifecycle."""

    __slots__ = ("_owner",)

    def __init__(self, owner: _LifecyclePermits) -> None:
        self._owner: _LifecyclePermits | None = owner

    def release(self) -> None:
        owner = self._owner
        if owner is None:
            raise RuntimeError("lifecycle permit was already released")
        self._owner = None
        owner.release()


@dataclass(frozen=True)
class _FileWorkItem:
    index: int
    frame_path: Path
    depth_path: Path
    frame_name: str
    left_path: Path
    right_path: Path
    stat_path: Path | None = None


@dataclass(slots=True)
class _PermittedWork:
    work: _FileWorkItem
    permit: _LifecyclePermitLease | None

    def take_permit(self) -> _LifecyclePermitLease:
        permit = self.permit
        if permit is None:
            raise RuntimeError("work permit ownership was already transferred")
        self.permit = None
        return permit


@dataclass(frozen=True)
class _FileWorkPlan:
    """Packed source-order actions plus a replayable O(1)-item work stream."""

    frame_files: list[Path]
    depth_files: list[Path]
    left_dir: Path
    right_dir: Path
    actions: PackedStereoActions
    render_count: int
    reuse_count: int
    migrate_count: int
    repaired_outputs: bool
    metric_mode: bool

    def iter_render(self):
        stat_dir = self.left_dir / "clamp_stats"
        for index, (frame_file, depth_file) in enumerate(zip(self.frame_files, self.depth_files)):
            if self.actions.get(index) is not StereoAction.RENDER:
                continue
            frame_name = frame_file.stem
            yield _FileWorkItem(
                index=index,
                frame_path=frame_file,
                depth_path=depth_file,
                frame_name=frame_name,
                left_path=self.left_dir / f"{frame_name}.png",
                right_path=self.right_dir / f"{frame_name}.png",
                stat_path=(stat_dir / f"{frame_name}.json" if self.metric_mode else None),
            )

    def iter_frame_names(self, action: StereoAction):
        for index, frame_file in enumerate(self.frame_files):
            if self.actions.get(index) is action:
                yield frame_file.stem


_DecodeFrame = Callable[
    [_FileWorkItem],
    tuple[
        np.ndarray,
        RelativeStereoPrimitiveInput | MetricStereoPrimitiveInput,
        None,
    ],
]


@dataclass(frozen=True)
class _FastRelativePlan:
    source_metadata: dict[str, Any]
    decode: _DecodeFrame
    settings: StereoRenderSettings
    kind: Literal["fast_relative"] = field(init=False, default="fast_relative")

    def __post_init__(self) -> None:
        if not isinstance(self.settings, StereoRenderSettings):
            raise TypeError("Fast relative plan requires StereoRenderSettings")
        if self.settings.stereo_render_mode != "fast":
            raise ValueError("Fast relative plan requires fast settings")

    geometry_mode = "relative"
    render_mode: Literal["fast"] = "fast"
    retained_crop_width = None
    effective_convergence = None

    @property
    def render_settings(self) -> StereoRenderSettings:
        return self.settings


@dataclass(frozen=True)
class _QualityRelativePlan:
    source_metadata: dict[str, Any]
    decode: _DecodeFrame
    settings: StereoRenderSettings
    kind: Literal["quality_relative"] = field(init=False, default="quality_relative")

    def __post_init__(self) -> None:
        if not isinstance(self.settings, StereoRenderSettings):
            raise TypeError("Quality relative plan requires StereoRenderSettings")
        if self.settings.stereo_render_mode != "quality":
            raise ValueError("Quality relative plan requires quality settings")

    geometry_mode = "relative"
    render_mode: Literal["quality"] = "quality"
    retained_crop_width = None
    effective_convergence = None

    @property
    def render_settings(self) -> StereoRenderSettings:
        return self.settings


@dataclass(frozen=True)
class _FastMetricPlan:
    source_metadata: dict[str, Any]
    decode: _DecodeFrame
    settings: StereoSplatSettings
    retained_crop_width: int
    effective_convergence: float
    kind: Literal["fast_metric"] = field(init=False, default="fast_metric")

    def __post_init__(self) -> None:
        if not isinstance(self.settings, StereoSplatSettings):
            raise TypeError("Fast metric plan requires StereoSplatSettings")

    geometry_mode = "metric_camera"
    render_mode: Literal["fast"] = "fast"

    @property
    def render_settings(self) -> StereoSplatSettings:
        return self.settings


@dataclass(frozen=True)
class _QualityMetricPlan:
    source_metadata: dict[str, Any]
    decode: _DecodeFrame
    settings: QualityStereoControls
    retained_crop_width: int
    effective_convergence: float
    kind: Literal["quality_metric"] = field(init=False, default="quality_metric")

    def __post_init__(self) -> None:
        if not isinstance(self.settings, QualityStereoControls):
            raise TypeError("Quality metric plan requires QualityStereoControls")

    geometry_mode = "metric_camera"
    render_mode: Literal["quality"] = "quality"

    @property
    def render_settings(self) -> QualityStereoControls:
        return self.settings


_StereoStagePlan = _FastRelativePlan | _QualityRelativePlan | _FastMetricPlan | _QualityMetricPlan


@dataclass(slots=True)
class _DecodedMessage:
    work: _FileWorkItem
    permit: _LifecyclePermitLease | None = None
    frame: np.ndarray | None = None
    render_input: RelativeStereoPrimitiveInput | MetricStereoPrimitiveInput | None = None
    projection_stats: MetricProjectionStats | None = None
    error: Exception | None = None

    def take_payload(
        self,
    ) -> tuple[
        _LifecyclePermitLease,
        np.ndarray,
        RelativeStereoPrimitiveInput | MetricStereoPrimitiveInput,
        MetricProjectionStats | None,
    ]:
        permit = self.permit
        frame = self.frame
        render_input = self.render_input
        if permit is None or frame is None or render_input is None:
            raise RuntimeError("decoded payload ownership was already transferred")
        self.permit = None
        self.frame = None
        self.render_input = None
        projection_stats = self.projection_stats
        self.projection_stats = None
        return permit, frame, render_input, projection_stats


@dataclass(slots=True)
class _ManifestDecodedFrame:
    work: StereoProviderItem
    frame: np.ndarray | None
    primitive: RelativeStereoPrimitiveInput | MetricStereoPrimitiveInput | None

    def take_frame(self) -> np.ndarray:
        frame = self.frame
        if frame is None:
            raise RuntimeError("manifest decoded frame ownership was already transferred")
        self.frame = None
        return frame

    def take_primitive(self) -> RelativeStereoPrimitiveInput | MetricStereoPrimitiveInput:
        primitive = self.primitive
        if primitive is None:
            raise RuntimeError("manifest primitive ownership was already transferred")
        self.primitive = None
        return primitive


def _take_relative_canonical(payload: _ManifestDecodedFrame) -> np.ndarray:
    primitive = payload.take_primitive()
    if not isinstance(primitive, RelativeStereoPrimitiveInput):
        raise TypeError("relative manifest payload contains metric primitives")
    encoded = primitive.encoded_canonical
    reciprocal_scale = np.float32(1.0 / primitive.encoding_scale)
    del primitive
    canonical = encoded.astype(np.float32)
    canonical *= reciprocal_scale
    del encoded
    return canonical


def _take_metric_primitive(payload: _ManifestDecodedFrame) -> MetricStereoPrimitiveInput:
    primitive = payload.take_primitive()
    if not isinstance(primitive, MetricStereoPrimitiveInput):
        raise TypeError("metric manifest payload contains relative primitives")
    return primitive


@dataclass(slots=True)
class _WriteItem:
    work: _FileWorkItem | StereoProviderItem
    left_image: np.ndarray | None
    right_image: np.ndarray | None
    permit: _LifecyclePermitLease | None = None
    projection_stats: MetricProjectionStats | None = None
    quality_publication: PreparedQualityFramePublication | None = None
    geometry_mode: str = "relative"
    occlusion_fill: Literal["none", "background"] = "background"
    rgb_stage_fingerprint: str = ""
    keep_masks: bool = False
    render_mode: Literal["fast", "quality"] = "fast"

    def release_payload_and_permit(self) -> None:
        self.left_image = None
        self.right_image = None
        self.projection_stats = None
        self.quality_publication = None
        permit = self.permit
        if permit is None:
            raise RuntimeError("write permit ownership was already transferred")
        self.permit = None
        permit.release()


@dataclass(frozen=True)
class _WriteCompletion:
    work: _FileWorkItem | StereoProviderItem
    error: Exception | None = None


_QUEUE_SENTINEL = object()


def _timed_put(
    target: queue.Queue[object],
    item: object,
    metrics: _PipelineMetrics,
) -> None:
    started = time.perf_counter()
    target.put(item)
    metrics.queue_waited(time.perf_counter() - started)


def _atomic_write_png(path: Path, image: np.ndarray) -> None:
    """Encode a PNG and atomically replace its destination from the same directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded, payload = cv2.imencode(".png", image)
    if not encoded:
        raise OSError(f"Could not encode PNG: {path}")
    raw = payload.tobytes()
    owned = OwnedSiblingTemporary.create(path)
    try:
        if owned.handle.write(raw) != len(raw):
            raise OSError(f"Short PNG write: {path}")
        owned.sync()
        owned.replace()
    finally:
        owned.cleanup()


def _decode_verified_png(
    path: Path,
    *,
    job_root: Path,
    expected_sha256: str,
    expected_byte_count: int,
    flags: int,
) -> np.ndarray:
    """Decode only the exact regular-file bytes bound by the provider item."""

    if expected_byte_count <= 0:
        raise ValueError("verified PNG byte count must be positive")
    try:
        handle = open_regular_beneath(job_root, path)
    except (OSError, ValueError) as error:
        raise StereoInputIdentityMismatch(
            f"could not reopen frozen stereo input: {path}"
        ) from error
    with handle:
        raw = bytearray(expected_byte_count)
        view = memoryview(raw)
        offset = 0
        while offset < expected_byte_count:
            count = handle.readinto(view[offset:])
            if not count:
                raise StereoInputIdentityMismatch(f"stereo input ended early: {path}")
            offset += count
        tail = bytearray(1)
        if handle.readinto(tail):
            raise StereoInputIdentityMismatch(f"stereo input grew during decode: {path}")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise StereoInputIdentityMismatch(
            f"stereo input content changed before decode: {path}"
        )
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), flags)
    if decoded is None:
        raise StereoInputIdentityMismatch(f"Could not decode frozen PNG: {path}")
    return decoded


def _stereo_creation_failed(error: Exception) -> bool:
    if isinstance(error, ValueError) and str(error) == METRIC_STEREO_COMPLETION_ERROR:
        raise error
    print(f"Error creating stereo pairs: {error}")
    traceback.print_exc()
    return False


def _write_pair(item: _WriteItem) -> None:
    if item.quality_publication is not None:
        publish_quality_frame(item.quality_publication)
    else:
        _atomic_write_png(item.work.left_path, item.left_image)
        _atomic_write_png(item.work.right_path, item.right_image)
    if item.work.stat_path is not None:
        if item.projection_stats is None:
            raise ValueError("Metric stereo writes require projection statistics")
        stats = item.projection_stats
        _atomic_write_json(
            item.work.stat_path,
            {
                "schema_version": METRIC_CLAMP_STATS_SCHEMA_VERSION,
                "frame_name": item.work.frame_name,
                "valid_pixel_count": stats.valid_pixel_count,
                "clamped_pixel_count": stats.clamped_pixel_count,
                "clamped_fraction": stats.clamped_fraction,
            },
        )


def _decode_relative_work_item(
    work: _FileWorkItem,
    *,
    encoding_scale: float,
    render_shape: tuple[int, int],
) -> tuple[np.ndarray, RelativeStereoPrimitiveInput, None]:
    frame = cv2.imread(str(work.frame_path), cv2.IMREAD_COLOR)
    encoded = cv2.imread(str(work.depth_path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise OSError(f"Could not load frame: {work.frame_path}")
    if encoded is None:
        raise OSError(f"Could not load canonical disparity: {work.depth_path}")
    if frame.shape[:2] != render_shape:
        raise ValueError(
            f"Frame shape changed at {work.frame_path}: "
            f"expected {render_shape}, got {frame.shape[:2]}"
        )
    if encoded.dtype != np.uint16 or encoded.ndim != 2:
        raise TypeError(f"Canonical disparity must be uint16: {work.depth_path}")
    primitive = RelativeStereoPrimitiveInput(
        encoded_canonical=np.ascontiguousarray(encoded),
        encoding_scale=np.float32(encoding_scale),
    )
    return frame, primitive, None


def _decode_metric_work_item(
    work: _FileWorkItem,
    *,
    store: MetricGeometryStore,
    render_shape: tuple[int, int],
    virtual_baseline_mm: float,
    convergence_distance_m: float,
    max_disparity_percent: float,
    retained_crop_width: int,
) -> tuple[np.ndarray, MetricStereoPrimitiveInput, None]:
    frame = cv2.imread(str(work.frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise OSError(f"Could not load frame: {work.frame_path}")
    if frame.shape[:2] != render_shape:
        raise ValueError(
            f"Frame shape changed at {work.frame_path}: "
            f"expected {render_shape}, got {frame.shape[:2]}"
        )
    primitive = MetricStereoPrimitiveInput(
        metric=store.load(work.depth_path),
        virtual_baseline_mm=np.float64(virtual_baseline_mm),
        convergence_distance_m=np.float64(convergence_distance_m),
        max_disparity_percent=np.float64(max_disparity_percent),
        retained_crop_width=retained_crop_width,
    )
    return frame, primitive, None


class _StereoFilePipeline:
    """Own the bounded queues, lifecycle permits, and I/O worker threads."""

    def __init__(
        self,
        *,
        renderer: StereoRenderer,
        work_factory: Callable[[], Any],
        work_count: int,
        decode: _DecodeFrame,
        render_settings: StereoRenderSettings | StereoSplatSettings | QualityStereoControls,
        render_mode: Literal["fast", "quality"],
        geometry_mode: str,
        occlusion_fill: Literal["none", "background"],
        rgb_stage_fingerprint: str,
        keep_masks: bool,
        workers: int,
        capacity: int,
        completed: int,
        total_frames: int,
        report_progress: Callable[..., None],
    ) -> None:
        self.renderer = renderer
        self.work_factory = work_factory
        self.work_count = work_count
        self.decode = decode
        self.render_settings = render_settings
        self.render_mode = render_mode
        self.geometry_mode = geometry_mode
        self.occlusion_fill = occlusion_fill
        self.rgb_stage_fingerprint = rgb_stage_fingerprint
        self.keep_masks = keep_masks
        self.workers = workers
        self.completed = completed
        self.total_frames = total_frames
        self.report_progress = report_progress
        self.metrics = _PipelineMetrics(capacity)
        self.permits = _LifecyclePermits(capacity, self.metrics)
        self.decode_requests: queue.Queue[object] = queue.Queue(maxsize=capacity)
        self.decoded_frames: queue.Queue[object] = queue.Queue(maxsize=capacity)
        self.write_requests: queue.Queue[object] = queue.Queue(maxsize=capacity)
        self.write_completions: queue.Queue[object] = queue.Queue(maxsize=capacity)
        self.first_error: tuple[str, _FileWorkItem, Exception] | None = None
        self.feeder_error: BaseException | None = None
        self.feed_done = threading.Event()
        self.work_items_enqueued = 0
        self.writes_enqueued = 0
        self.completions_seen = 0
        self.successful_completions = 0
        self.feeder = threading.Thread(
            target=self._feed_decoder,
            name="stereo-decode-feeder",
            daemon=True,
        )
        self.decoder_threads = [
            threading.Thread(
                target=self._decode_worker,
                name=f"stereo-decode-{index}",
                daemon=True,
            )
            for index in range(workers)
        ]
        self.writer_threads = [
            threading.Thread(
                target=self._write_worker,
                name=f"stereo-write-{index}",
                daemon=True,
            )
            for index in range(workers)
        ]

    def run(self) -> StereoPipelineStats:
        started = time.perf_counter()
        for thread in self.decoder_threads + self.writer_threads:
            thread.start()
        self.feeder.start()

        decoded_seen = 0
        while True:
            try:
                message = self.decoded_frames.get(timeout=0.05)
            except queue.Empty:
                if self.feed_done.is_set() and decoded_seen >= self.work_items_enqueued:
                    break
                continue
            try:
                assert isinstance(message, _DecodedMessage)
                write_item = self._render_decoded(message)
                decoded_seen += 1
                del message
                if write_item is not None:
                    try:
                        _timed_put(self.write_requests, write_item, self.metrics)
                    except BaseException:
                        write_item.release_payload_and_permit()
                        raise
                    self.writes_enqueued += 1
                    del write_item
                self._drain_ready_completions()
            finally:
                self.decoded_frames.task_done()

        self._wait_for_writes()
        for _ in range(self.workers):
            _timed_put(self.write_requests, _QUEUE_SENTINEL, self.metrics)
        self.feeder.join()
        for thread in self.decoder_threads + self.writer_threads:
            thread.join()

        self.metrics.pipeline_finished(time.perf_counter() - started)
        stats = self.metrics.snapshot()
        if self.feeder_error is not None:
            raise RuntimeError(
                f"Stereo provider replay failed: {self.feeder_error}"
            ) from self.feeder_error
        if self.first_error is not None:
            stage, work, error = self.first_error
            raise RuntimeError(f"Stereo {stage} failed for {work.frame_name}: {error}") from error
        return stats

    def _feed_decoder(self) -> None:
        try:
            work_iterator = iter(self.work_factory())
            for _ in range(self.work_count):
                permit = self.permits.acquire()
                try:
                    work = next(work_iterator)
                except BaseException:
                    permit.release()
                    raise
                request = _PermittedWork(work=work, permit=permit)
                try:
                    _timed_put(self.decode_requests, request, self.metrics)
                except BaseException:
                    remaining = request.permit
                    request.permit = None
                    if remaining is not None:
                        remaining.release()
                    raise
                self.work_items_enqueued += 1
                del request, work, permit
        except BaseException as error:
            self.feeder_error = error.with_traceback(None)
        finally:
            for _ in range(self.workers):
                _timed_put(self.decode_requests, _QUEUE_SENTINEL, self.metrics)
            self.feed_done.set()

    def _decode_worker(self) -> None:
        while True:
            request = self.decode_requests.get()
            try:
                if request is _QUEUE_SENTINEL:
                    return
                assert isinstance(request, _PermittedWork)
                work = request.work
                permit = request.take_permit()
                message = self._decode(work, permit)
                _timed_put(self.decoded_frames, message, self.metrics)
                del message, permit, work
            finally:
                self.decode_requests.task_done()

    def _decode(
        self,
        work: _FileWorkItem,
        permit: _LifecyclePermitLease,
    ) -> _DecodedMessage:
        try:
            frame, render_input, projection_stats = self.decode(work)
        except Exception as error:
            error = error.with_traceback(None)
            permit.release()
            return _DecodedMessage(work=work, error=error)
        self.metrics.increment("decoded_frames")
        return _DecodedMessage(
            work=work,
            permit=permit,
            frame=frame,
            render_input=render_input,
            projection_stats=projection_stats,
        )

    def _render_decoded(self, message: _DecodedMessage) -> _WriteItem | None:  # noqa: C901
        if message.error is not None:
            self._record_error("decode", message.work, message.error)
            return None
        permit, frame, render_input, projection_stats = message.take_payload()
        work = message.work
        if self.first_error is not None:
            frame = None
            render_input = None
            projection_stats = None
            permit.release()
            return None
        canonical = None
        compact = None
        result = None
        geometry = None
        primitive = None
        metric = None
        quality_publication = None
        left_image = None
        right_image = None
        try:
            if isinstance(render_input, RelativeStereoPrimitiveInput):
                if not isinstance(self.render_settings, StereoRenderSettings):
                    raise TypeError("Relative stereo requires StereoRenderSettings")
                canonical = render_input.encoded_canonical.astype(np.float32)
                canonical *= np.float32(1.0 / render_input.encoding_scale)
                if self.render_settings.stereo_render_mode == "quality":
                    compact = render_relative_quality_compact(
                        renderer=self.renderer,
                        frame=frame,
                        canonical=canonical,
                        settings=self.render_settings,
                    )
                    result = None
                elif hasattr(self.renderer, "render_relative_fast_compact"):
                    compact = self.renderer.render_relative_fast_compact(
                        frame,
                        canonical,
                        self.render_settings,
                    )
                    result = None
                else:
                    result = self.renderer.render(
                        frame,
                        canonical,
                        self.render_settings,
                    )
                    compact = None
            elif isinstance(render_input, MetricStereoPrimitiveInput):
                primitive = render_input
                if isinstance(self.render_settings, QualityStereoControls):
                    compact = render_metric_primitives_compact(
                        renderer=self.renderer,
                        frame=frame,
                        primitives=primitive,
                        controls=self.render_settings,
                    )
                    projection_stats = compact.metric_stats
                    result = None
                elif isinstance(self.render_settings, StereoSplatSettings):
                    metric = primitive.metric
                    geometry, projection_stats = build_metric_geometry(
                        metric.inverse_depth,
                        metric.valid,
                        metric.focal_x_normalized,
                        (int(frame.shape[0]), int(frame.shape[1])),
                        virtual_baseline_mm=float(primitive.virtual_baseline_mm),
                        convergence_distance_m=float(primitive.convergence_distance_m),
                        max_disparity_percent=float(primitive.max_disparity_percent),
                        retained_crop_width=primitive.retained_crop_width,
                    )
                    if hasattr(self.renderer, "render_geometry_fast_compact"):
                        compact = self.renderer.render_geometry_fast_compact(
                            frame,
                            geometry,
                            self.render_settings,
                            metric_stats=projection_stats,
                        )
                        result = None
                    else:
                        result = self.renderer.render_geometry(
                            frame,
                            geometry,
                            self.render_settings,
                        )
                        compact = None
                else:
                    raise TypeError(
                        "Metric stereo requires StereoSplatSettings or QualityStereoControls"
                    )
            else:
                raise TypeError("Unsupported stereo primitive input")
        except Exception as error:
            error = error.with_traceback(None)
            canonical = None
            compact = None
            result = None
            geometry = None
            primitive = None
            metric = None
            quality_publication = None
            left_image = None
            right_image = None
            frame = None
            render_input = None
            projection_stats = None
            permit.release()
            self._record_error("render", work, error)
            return None
        try:
            self.metrics.increment("rendered_frames")
            if compact is not None:
                left_image = compact.left_image
                right_image = compact.right_image
                quality_publication = prepare_quality_frame(
                    QualityFramePublication(
                        job_root=work.left_path.parent.parent,
                        frame_name=work.frame_name,
                        left_rgb_path=work.left_path,
                        right_rgb_path=work.right_path,
                        compact=compact,
                        geometry_mode=(
                            "metric" if self.geometry_mode == "metric_camera" else "relative"
                        ),
                        occlusion_fill=self.occlusion_fill,
                        rgb_stage_fingerprint=self.rgb_stage_fingerprint,
                        keep_masks=self.keep_masks,
                        render_mode=self.render_mode,
                    )
                )
                del compact
            else:
                assert result is not None
                left_image = result.left_image
                right_image = result.right_image
                quality_publication = None
            return _WriteItem(
                work=work,
                left_image=left_image,
                right_image=right_image,
                permit=permit,
                projection_stats=projection_stats,
                quality_publication=quality_publication,
                geometry_mode=self.geometry_mode,
                occlusion_fill=self.occlusion_fill,
                rgb_stage_fingerprint=self.rgb_stage_fingerprint,
                keep_masks=self.keep_masks,
                render_mode=self.render_mode,
            )
        except Exception as error:
            error = error.with_traceback(None)
            canonical = None
            compact = None
            result = None
            geometry = None
            primitive = None
            metric = None
            quality_publication = None
            left_image = None
            right_image = None
            frame = None
            render_input = None
            projection_stats = None
            permit.release()
            self._record_error("render", work, error)
            return None

    def _write_worker(self) -> None:
        while True:
            item = self.write_requests.get()
            try:
                if item is _QUEUE_SENTINEL:
                    return
                assert isinstance(item, _WriteItem)
                completion = self._write(item)
                _timed_put(self.write_completions, completion, self.metrics)
            finally:
                self.write_requests.task_done()

    def _write(self, item: _WriteItem) -> _WriteCompletion:
        work = item.work
        error = None
        started = time.perf_counter()
        try:
            _write_pair(item)
        except Exception as write_error:
            error = write_error.with_traceback(None)
        else:
            self.metrics.increment("written_frames")
        finally:
            self.metrics.writer_busy(time.perf_counter() - started)
            item.release_payload_and_permit()
        return _WriteCompletion(work=work, error=error)

    def _drain_ready_completions(self) -> None:
        while True:
            try:
                completion = self.write_completions.get_nowait()
            except queue.Empty:
                return
            try:
                assert isinstance(completion, _WriteCompletion)
                self._handle_completion(completion)
            finally:
                self.write_completions.task_done()

    def _wait_for_writes(self) -> None:
        while self.completions_seen < self.writes_enqueued:
            completion = self.write_completions.get()
            try:
                assert isinstance(completion, _WriteCompletion)
                self._handle_completion(completion)
            finally:
                self.write_completions.task_done()

    def _handle_completion(self, completion: _WriteCompletion) -> None:
        self.completions_seen += 1
        if completion.error is not None:
            self._record_error("write", completion.work, completion.error)
            return
        self.successful_completions += 1
        try:
            self.report_progress(
                processed=self.completed + self.successful_completions,
                total=self.total_frames,
                work_index=completion.work.index,
                left_path=completion.work.left_path,
            )
        except Exception as error:
            self._record_error("progress", completion.work, error)

    def _record_error(
        self,
        stage: str,
        work: _FileWorkItem,
        error: Exception,
    ) -> None:
        if self.first_error is None:
            self.first_error = (stage, work, error)


class StereoPairGenerator:
    """Generate stereo pairs with main-thread rendering and bounded threaded I/O."""

    def __init__(
        self,
        verbose: bool = False,
        *,
        renderer: StereoRenderer | None = None,
    ) -> None:
        self.verbose = verbose
        self._renderer = renderer
        self.last_pipeline_stats: StereoPipelineStats | StreamingStereoPipelineStats | None = None
        self.last_metric_clamp_summary: dict[str, Any] | None = None

    @property
    def renderer(self) -> StereoRenderer:
        if self._renderer is None:
            self._renderer = StereoRenderer()
        return self._renderer

    @staticmethod
    def _occlusion_fill(settings: dict[str, Any]) -> Literal["none", "background"]:
        return cast(
            Literal["none", "background"],
            str(settings.get("occlusion_fill", "background")),
        )

    @staticmethod
    def _manifest_frame_names(manifest: StereoInputManifestV1, *, filenames: bool):
        suffix = ".png" if filenames else ""
        try:
            for record in manifest.replay_declared_records():
                yield f"{record.frame_name}{suffix}"
        except StereoControlStateError as error:
            raise StereoInputIdentityMismatch(str(error)) from error

    def _streamed_stage_identity(
        self,
        *,
        manifest: StereoInputManifestV1,
        provider,
        settings: dict[str, Any],
        render_mode: Literal["fast", "quality"],
        geometry_mode: Literal["relative", "metric"],
        occlusion_fill: Literal["none", "background"],
        quality_content: StreamedQualityInputManifest | None,
        retained_crop_width: int | None,
        effective_convergence: float | None,
    ) -> tuple[StreamedCanonicalObject | StreamedPrettyJsonValue, str]:
        render_shape = manifest.render_shape
        if render_mode == "quality":
            if quality_content is None:
                raise ValueError("Quality RGB identity requires its content manifest")
            if geometry_mode == "relative":
                projection: dict[str, Any] = {
                    "convergence": float(settings.get("convergence", 0.5)),
                    "kind": "relative",
                    "stereo_strength": float(settings.get("stereo_strength", 2.0)),
                }
            else:
                if retained_crop_width is None or effective_convergence is None:
                    raise ValueError("metric Quality projection is incomplete")
                requested = settings.get("metric_convergence_distance", "auto")
                projection = {
                    "center_crop_algorithm_version": CENTER_CROP_ALGORITHM_VERSION,
                    "effective_convergence_distance_m": float(effective_convergence),
                    "kind": "metric",
                    "max_disparity_percent": float(
                        settings.get("max_disparity_percent", 2.0)
                    ),
                    "projection_algorithm_version": METRIC_PROJECTION_ALGORITHM_VERSION,
                    "requested_convergence_distance": (
                        "auto" if requested == "auto" else float(requested)
                    ),
                    "retained_crop_width": retained_crop_width,
                    "sample_aspect_ratio": "1:1",
                    "source_width": render_shape[1],
                    "virtual_baseline_mm": float(settings.get("virtual_baseline_mm", 63.0)),
                }
            if occlusion_fill == "background":
                configured_limit = int(settings.get("occlusion_fill_max_px", 8))
                repair_policy: dict[str, Any] = {
                    "configured_limit_1080p": configured_limit,
                    "local_limit_formula": "min-scaled-safe-predicted-plus2-v1",
                    "predicted_gap_policy": "max-four-neighbour-eye-shift-v1",
                    "scaled_safe_limit_px": max(
                        1,
                        math.floor(configured_limit * render_shape[0] / 1080.0 + 0.5),
                    ),
                }
            else:
                repair_policy = {
                    "configured_limit_1080p": None,
                    "local_limit_formula": None,
                    "predicted_gap_policy": None,
                    "scaled_safe_limit_px": None,
                }
            document = build_streamed_canonical_object(
                static_fields={
                    "algorithm_version": QUALITY_STEREO_ALGORITHM_VERSION,
                    "encoding": "uint8_png",
                    "geometry_mode": geometry_mode,
                    "native_geometry_fingerprint": (
                        quality_content.native_geometry_fingerprint
                    ),
                    "occlusion_fill": occlusion_fill,
                    "projection": projection,
                    "quality_input_manifest_sha256": quality_content.raw_sha256,
                    "render_shape": [render_shape[0], render_shape[1]],
                    "repair_policy": repair_policy,
                    "schema_version": 2,
                    "source_guide_fingerprint": quality_content.source_guide_fingerprint,
                },
                array_key="frame_names",
                items=lambda: self._manifest_frame_names(manifest, filenames=False),
                fingerprint_key="fingerprint",
            )
            assert document.fingerprint is not None
            return document, document.fingerprint

        device = getattr(self.renderer, "device", None)
        device_type = getattr(device, "type", None) or "custom"
        static: dict[str, Any] = {
            "algorithm_version": STEREO_STAGE_ALGORITHM_VERSION,
            "encoding": "uint8_png",
            "geometry_mode": "metric_camera" if geometry_mode == "metric" else "relative",
            "occlusion_fill": occlusion_fill,
            "render_shape": [render_shape[0], render_shape[1]],
            "renderer_device_type": str(device_type),
            "schema_version": STEREO_STAGE_SCHEMA_VERSION,
        }
        if geometry_mode == "relative":
            static.update(
                {
                    "render_settings": {
                        "convergence": float(settings.get("convergence", 0.5)),
                        "occlusion_fill": occlusion_fill,
                        "stereo_strength": float(settings.get("stereo_strength", 2.0)),
                    },
                    "source_canonical_fingerprint": manifest.source_geometry_fingerprint,
                }
            )
        else:
            if retained_crop_width is None or effective_convergence is None:
                raise ValueError("metric Fast projection is incomplete")
            requested = settings.get("metric_convergence_distance", "auto")
            static.update(
                {
                    "center_crop_algorithm_version": CENTER_CROP_ALGORITHM_VERSION,
                    "effective_convergence_distance_m": float(effective_convergence),
                    "max_disparity_percent": float(
                        settings.get("max_disparity_percent", 2.0)
                    ),
                    "projection_algorithm_version": METRIC_PROJECTION_ALGORITHM_VERSION,
                    "requested_convergence_distance": (
                        "auto" if requested == "auto" else float(requested)
                    ),
                    "retained_crop_width": retained_crop_width,
                    "sample_aspect_ratio": "1:1",
                    "source_metric_fingerprint": manifest.source_geometry_fingerprint,
                    "source_width": render_shape[1],
                    "virtual_baseline_mm": float(settings.get("virtual_baseline_mm", 63.0)),
                }
            )
        canonical = build_streamed_canonical_object(
            static_fields=static,
            array_key="frame_names",
            items=lambda: self._manifest_frame_names(manifest, filenames=True),
            fingerprint_key="fingerprint",
        )
        assert canonical.fingerprint is not None
        pretty_value = dict(static)
        pretty_value["fingerprint"] = canonical.fingerprint
        pretty_value["frame_names"] = ReplayableJsonArray(
            lambda: self._manifest_frame_names(manifest, filenames=True)
        )
        return build_streamed_pretty_json_value(pretty_value), canonical.fingerprint

    @staticmethod
    def _streaming_diagnostics_context(
        *,
        manifest: StereoInputManifestV1,
        provider,
        render_mode: Literal["fast", "quality"],
        geometry_mode: Literal["relative", "metric"],
        occlusion_fill: Literal["none", "background"],
        keep_masks: bool,
        rgb_stage_fingerprint: str,
        quality_content: StreamedQualityInputManifest | None,
        left_dir: Path,
        right_dir: Path,
    ) -> StreamingDiagnosticsStageContext:
        if quality_content is None:
            source_guide_fingerprint = manifest.source_geometry_fingerprint
            native_geometry_fingerprint = None
            content_sha = None
        else:
            source_guide_fingerprint = quality_content.source_guide_fingerprint
            native_geometry_fingerprint = quality_content.native_geometry_fingerprint
            content_sha = quality_content.raw_sha256
        def replay_source_names():
            count = 0
            try:
                for item in provider.replay():
                    if item.source_index != count:
                        raise StereoInputIdentityMismatch(
                            "diagnostics source replay order changed"
                        )
                    count += 1
                    yield item.frame_name
                if count != manifest.frame_count:
                    raise StereoInputIdentityMismatch(
                        "diagnostics source replay ended before its frame count"
                    )
            except StereoInputIdentityMismatch:
                raise
            except (StereoControlError, OSError, TypeError, ValueError) as error:
                raise StereoInputIdentityMismatch(str(error)) from error

        return StreamingDiagnosticsStageContext(
            job_root=manifest.job_root,
            frame_count=manifest.frame_count,
            frame_names=replay_source_names,
            render_shape=manifest.render_shape,
            render_mode=render_mode,
            geometry_mode=geometry_mode,
            occlusion_fill=occlusion_fill,
            mask_payloads_enabled=keep_masks,
            rgb_stage_fingerprint=rgb_stage_fingerprint,
            source_guide_fingerprint=source_guide_fingerprint,
            native_geometry_fingerprint=native_geometry_fingerprint,
            quality_input_manifest_sha256=content_sha,
            native_geometry_sample_count=(
                manifest.native_shape[0] * manifest.native_shape[1]
                if render_mode == "quality"
                else None
            ),
            left_rgb_parent=left_dir,
            right_rgb_parent=right_dir,
        )

    def create_stereo_pairs_from_manifest(
        self,
        manifest: StereoInputManifestV1,
        directories: dict[str, Path],
        settings: dict[str, Any],
        progress_tracker=None,
    ) -> bool:
        """Restart transient source changes only while the stage is read-only."""

        manifest_path = manifest.path
        job_root = manifest.job_root
        last_change: BaseException | None = None
        for _attempt in range(STEREO_READ_ONLY_RESTART_LIMIT):
            current = load_stereo_input_manifest_v1(
                manifest_path,
                job_root=job_root,
                validate_payloads=False,
            )
            try:
                try:
                    return self._create_stereo_pairs_from_manifest_once(
                        current,
                        directories,
                        settings,
                        progress_tracker,
                    )
                except StereoInputIdentityMismatch as error:
                    raise StereoProviderChangedError(str(error)) from error
            except StereoProviderChangedError as error:
                last_change = error
        assert last_change is not None
        raise StereoProviderChangedError(
            "stereo inputs did not stabilize during read-only audit"
        ) from last_change

    def _create_stereo_pairs_from_manifest_once(
        self,
        manifest: StereoInputManifestV1,
        directories: dict[str, Path],
        settings: dict[str, Any],
        progress_tracker=None,
    ) -> bool:
        """Render one validated durable upstream stereo sequence."""

        self.last_pipeline_stats = None
        self.last_metric_clamp_summary = None
        adopted = load_stereo_input_manifest_v1(
            manifest.path,
            job_root=manifest.job_root,
            validate_payloads=False,
        )
        if adopted != manifest:
            raise StereoProviderChangedError("stereo input manifest descriptor changed")
        if Path(directories["base"]).resolve() != adopted.job_root:
            raise ValueError("stereo input manifest belongs to a different job root")
        geometry_mode = str(settings.get("stereo_geometry_mode", "relative"))
        expected_manifest_mode = "metric" if geometry_mode == "metric_camera" else "relative"
        if adopted.geometry_mode != expected_manifest_mode:
            raise ValueError("stereo geometry mode differs from its upstream manifest")
        render_mode = cast(
            Literal["fast", "quality"],
            str(settings.get("stereo_render_mode", "fast")),
        )
        if render_mode not in {"fast", "quality"}:
            raise ValueError("stereo render mode must be Fast or Quality")

        left_dir = Path(directories["left_frames"])
        right_dir = Path(directories["right_frames"])
        provider = adopted.open_provider(left_dir=left_dir, right_dir=right_dir)
        render_shape = adopted.render_shape
        occlusion_fill = self._occlusion_fill(settings)
        normalized_geometry = cast(
            Literal["relative", "metric"],
            "metric" if geometry_mode == "metric_camera" else "relative",
        )
        keep_masks = bool(settings.get("keep_intermediates", True))

        retained_crop_width: int | None = None
        effective_convergence: float | None = None
        if normalized_geometry == "metric":
            retained_crop_width, _height = calculate_center_crop_dimensions(
                render_shape[1],
                render_shape[0],
                float(settings.get("crop_factor", 1.0)),
            )
            requested = settings.get("metric_convergence_distance", "auto")
            if requested == "auto":
                if adopted.resolved_auto_distance_m is None:
                    raise ValueError(METRIC_STEREO_COMPLETION_ERROR)
                effective_convergence = adopted.resolved_auto_distance_m
            else:
                effective_convergence = float(requested)

        quality_content: StreamedQualityInputManifest | None
        if render_mode == "quality":
            quality_content = build_streamed_quality_input_manifest(
                adopted,
                validate_payloads=False,
            )
        else:
            quality_content = None
        stage_document, stage_fingerprint = self._streamed_stage_identity(
            manifest=adopted,
            provider=provider,
            settings=settings,
            render_mode=render_mode,
            geometry_mode=normalized_geometry,
            occlusion_fill=occlusion_fill,
            quality_content=quality_content,
            retained_crop_width=retained_crop_width,
            effective_convergence=effective_convergence,
        )
        diagnostics_context = self._streaming_diagnostics_context(
            manifest=adopted,
            provider=provider,
            render_mode=render_mode,
            geometry_mode=normalized_geometry,
            occlusion_fill=occlusion_fill,
            keep_masks=keep_masks,
            rgb_stage_fingerprint=stage_fingerprint,
            quality_content=quality_content,
            left_dir=left_dir,
            right_dir=right_dir,
        )
        metadata_path = left_dir / "metadata.json"
        content_path = adopted.job_root / "04_stereo_diagnostics" / "quality_input_manifest.json"
        with AcquiredRoot(adopted.job_root) as acquired:
            stage_changed = not stage_document.matches(metadata_path, acquired=acquired)
            if quality_content is not None:
                stage_changed = stage_changed or not quality_content.matches(
                    content_path,
                    acquired=acquired,
                )
        diagnostics_root_state = (
            "unclaimed"
            if stage_changed
            else classify_streaming_diagnostics_root(diagnostics_context)
        )
        reset_diagnostics_root = (
            stage_changed or diagnostics_root_state == "unclaimed"
        )

        def classify(item: StereoProviderItem) -> StereoAction:
            if stage_changed or diagnostics_root_state == "unclaimed":
                return StereoAction.RENDER
            action = classify_streaming_diagnostics_frame(
                diagnostics_context,
                item.frame_name,
            )
            return {
                "reuse": StereoAction.REUSE,
                "migrate": StereoAction.MIGRATE,
                "render": StereoAction.RENDER,
            }[action]

        control = audit_stereo_provider(provider, classify)
        aggregates_valid = (
            control.render_count == 0
            and control.migrate_count == 0
            and diagnostics_root_state == "complete"
            and streaming_diagnostics_aggregates_match(diagnostics_context)
        )
        mutation_required = (
            stage_changed
            or control.render_count > 0
            or control.migrate_count > 0
            or not aggregates_valid
        )

        def revalidate_output_actions() -> None:
            with AcquiredRoot(adopted.job_root) as acquired:
                current_stage_changed = not stage_document.matches(
                    metadata_path,
                    acquired=acquired,
                )
                if quality_content is not None:
                    current_stage_changed = current_stage_changed or not quality_content.matches(
                        content_path,
                        acquired=acquired,
                    )
            if current_stage_changed != stage_changed:
                raise StereoProviderChangedError(
                    "stereo stage identity changed during read-only audit"
                )
            current_root_state = (
                "unclaimed"
                if current_stage_changed
                else classify_streaming_diagnostics_root(diagnostics_context)
            )
            if current_root_state != diagnostics_root_state:
                raise StereoProviderChangedError(
                    "stereo diagnostics root changed during read-only audit"
                )

            def current_classify(item: StereoProviderItem) -> StereoAction:
                if current_stage_changed or current_root_state == "unclaimed":
                    return StereoAction.RENDER
                action = classify_streaming_diagnostics_frame(
                    diagnostics_context,
                    item.frame_name,
                )
                return {
                    "reuse": StereoAction.REUSE,
                    "migrate": StereoAction.MIGRATE,
                    "render": StereoAction.RENDER,
                }[action]

            validate_stereo_action_replay(provider, control, current_classify)
            current_aggregates_valid = (
                control.render_count == 0
                and control.migrate_count == 0
                and current_root_state == "complete"
                and streaming_diagnostics_aggregates_match(diagnostics_context)
            )
            current_mutation_required = (
                current_stage_changed
                or control.render_count > 0
                or control.migrate_count > 0
                or not current_aggregates_valid
            )
            if current_mutation_required != mutation_required:
                raise StereoProviderChangedError(
                    "stereo retained aggregate action changed during read-only audit"
                )

        if render_mode == "quality":
            _quality_stage_memory_preflight(
                control=control,
                render_shape=render_shape,
                native_shape=adopted.native_shape,
                geometry_mode=normalized_geometry,
                occlusion_fill=occlusion_fill,
                consolidation_planned=mutation_required,
            )
            adopted.audit_payloads()
        if not mutation_required:
            validate_stereo_provider_replay(
                provider,
                control,
                mutation_started=False,
            )
            revalidate_output_actions()
            return True

        disk_plan = _build_stereo_stage_disk_plan(
            provider=provider,
            control=control,
            job_root=adopted.job_root,
            render_shape=render_shape,
            render_mode=render_mode,
            geometry_mode=normalized_geometry,
            keep_masks=keep_masks,
            left_dir=left_dir,
            stage_document=stage_document,
            quality_content=quality_content,
        )
        admit_stereo_stage_disk(disk_plan)

        def revalidate_disk_admission() -> None:
            revalidate_output_actions()
            revalidated = _build_stereo_stage_disk_plan(
                provider=provider,
                control=control,
                job_root=adopted.job_root,
                render_shape=render_shape,
                render_mode=render_mode,
                geometry_mode=normalized_geometry,
                keep_masks=keep_masks,
                left_dir=left_dir,
                stage_document=stage_document,
                quality_content=quality_content,
            )
            if revalidated != disk_plan:
                raise StereoProviderChangedError(
                    "stereo disk plan changed before the first mutation"
                )
            admit_stereo_stage_disk(revalidated)

        configured_workers = validate_stereo_io_workers(
            settings.get("stereo_io_workers", _default_stereo_io_workers())
        )
        if render_mode == "quality":
            capacity_value, effective_workers, _thread_bytes = quality_pipeline_capacity(
                control.render_count
            )
        else:
            maximum_item = (
                control.maximum_fast_metric_item_bytes
                if normalized_geometry == "metric"
                else control.maximum_fast_relative_item_bytes
            )
            selection = select_fast_pipeline_capacity(
                configured_workers=configured_workers,
                pending_render_count=control.render_count,
                fast_active_scratch=(
                    272 * (render_shape[1] + 1)
                    + 1024 * 1024
                    + STEREO_CONTROL_OVERHEAD
                ),
                fast_slot_bytes=maximum_item + HOST_SLOT_OVERHEAD,
            )
            capacity_value = selection.capacity
            effective_workers = selection.effective_workers
        mutation_started = False

        def begin_mutation() -> None:
            nonlocal mutation_started
            left_dir.mkdir(parents=True, exist_ok=True)
            right_dir.mkdir(parents=True, exist_ok=True)
            diagnostics_root = adopted.job_root / "04_stereo_diagnostics"
            diagnostics_frames = diagnostics_root / "frames"
            downstream = self._downstream_stage_directories(directories)
            authorized_roots = [left_dir, right_dir, diagnostics_root, *downstream]
            if not reset_diagnostics_root:
                authorized_roots.append(diagnostics_frames)
            with SafeTreeMutation(
                adopted.job_root,
                tuple(authorized_roots),
            ) as mutation:
                mutation_started = True
                if control.render_count > 0:
                    for directory in downstream:
                        mutation.clear_root(directory)
                root_targets = [
                    left_dir / "metadata.json",
                    diagnostics_root / "metadata.json",
                    diagnostics_root / "stereo_coverage_frames.jsonl",
                    diagnostics_root / "stereo_coverage_summary.json",
                ]
                if normalized_geometry == "metric":
                    root_targets.append(left_dir / "clamp_summary.json")
                if quality_content is not None:
                    root_targets.append(diagnostics_root / "quality_input_manifest.json")
                for target in root_targets:
                    temporary = publication_temporary_path(target)
                    evidence = mutation.capture_regular_child(
                        target.parent,
                        temporary,
                    )
                    mutation.unlink_child(
                        target.parent,
                        temporary,
                        evidence=evidence,
                    )
                if stage_changed:
                    mutation.clear_root(left_dir)
                    mutation.clear_root(right_dir)
                if reset_diagnostics_root:
                    mutation.remove_root(diagnostics_root)
                else:
                    for item in iter_stereo_action(provider, control, StereoAction.RENDER):
                        left_evidence = mutation.capture_regular_child(left_dir, item.left_path)
                        mutation.unlink_child(
                            left_dir,
                            item.left_path,
                            evidence=left_evidence,
                        )
                        left_temporary = publication_temporary_path(item.left_path)
                        left_temporary_evidence = mutation.capture_regular_child(
                            left_dir,
                            left_temporary,
                        )
                        mutation.unlink_child(
                            left_dir,
                            left_temporary,
                            evidence=left_temporary_evidence,
                        )
                        right_evidence = mutation.capture_regular_child(
                            right_dir,
                            item.right_path,
                        )
                        mutation.unlink_child(
                            right_dir,
                            item.right_path,
                            evidence=right_evidence,
                        )
                        right_temporary = publication_temporary_path(item.right_path)
                        right_temporary_evidence = mutation.capture_regular_child(
                            right_dir,
                            right_temporary,
                        )
                        mutation.unlink_child(
                            right_dir,
                            right_temporary,
                            evidence=right_temporary_evidence,
                        )
                        frame_tree = diagnostics_frames / item.frame_name
                        frame_tree_evidence = mutation.capture_child_tree(
                            diagnostics_frames,
                            frame_tree,
                        )
                        mutation.remove_child_tree(
                            diagnostics_frames,
                            frame_tree,
                            evidence=frame_tree_evidence,
                        )
                    for item in iter_stereo_action(
                        provider,
                        control,
                        StereoAction.MIGRATE,
                    ):
                        manifest = diagnostics_frames / item.frame_name / "manifest.json"
                        manifest_temporary = publication_temporary_path(manifest)
                        evidence = mutation.capture_regular_child(
                            diagnostics_frames,
                            manifest_temporary,
                        )
                        mutation.unlink_descendant(
                            diagnostics_frames,
                            manifest_temporary,
                            evidence=evidence,
                        )
            if quality_content is not None:
                quality_content.publish(content_path)
            stage_document.publish(metadata_path)
            begin_streaming_diagnostics_stage(diagnostics_context)

        def decode(item: StereoProviderItem):
            frame = _decode_verified_png(
                item.guide_path,
                job_root=adopted.job_root,
                expected_sha256=item.guide_sha256,
                expected_byte_count=item.guide_byte_count,
                flags=cv2.IMREAD_COLOR,
            )
            if frame.shape != (*item.guide_shape, 3) or frame.dtype != np.uint8:
                raise StereoInputIdentityMismatch(
                    f"guide PNG shape changed: {item.guide_path}"
                )
            if normalized_geometry == "relative":
                encoded = _decode_verified_png(
                    item.geometry_path,
                    job_root=adopted.job_root,
                    expected_sha256=item.geometry_sha256,
                    expected_byte_count=item.geometry_byte_count,
                    flags=cv2.IMREAD_UNCHANGED,
                )
                if encoded.shape != item.geometry_shape or encoded.dtype != np.uint16:
                    raise StereoInputIdentityMismatch(
                        f"geometry PNG shape changed: {item.geometry_path}"
                    )
                encoding_scale = adopted.relative_encoding_scale
                assert encoding_scale is not None
                primitive: RelativeStereoPrimitiveInput | MetricStereoPrimitiveInput = (
                    RelativeStereoPrimitiveInput(
                        encoded_canonical=encoded,
                        encoding_scale=encoding_scale,
                    )
                )
            else:
                try:
                    audited = metric_npz.audit_metric_npz(
                        item.geometry_path,
                        expected_shape=item.geometry_shape,
                        job_root=adopted.job_root,
                    )
                except (OSError, TypeError, ValueError) as error:
                    raise StereoInputIdentityMismatch(
                        f"metric geometry changed before decode: {item.geometry_path}"
                    ) from error
                if (
                    audited.sha256 != item.geometry_sha256
                    or audited.byte_count != item.geometry_byte_count
                ):
                    raise StereoInputIdentityMismatch(
                        f"metric geometry content changed before decode: {item.geometry_path}"
                    )
                metric = MetricGeometryFrame._from_owned(
                    audited.inverse_depth,
                    audited.valid,
                    audited.focal_x_normalized,
                )
                assert retained_crop_width is not None and effective_convergence is not None
                primitive = MetricStereoPrimitiveInput(
                    metric=metric,
                    virtual_baseline_mm=np.float64(
                        settings.get("virtual_baseline_mm", 63.0)
                    ),
                    convergence_distance_m=np.float64(effective_convergence),
                    max_disparity_percent=np.float64(
                        settings.get("max_disparity_percent", 2.0)
                    ),
                    retained_crop_width=retained_crop_width,
                )
            return _ManifestDecodedFrame(item, frame, primitive)

        if normalized_geometry == "relative":
            render_settings: StereoRenderSettings | StereoSplatSettings | QualityStereoControls = (
                StereoRenderSettings(
                    stereo_strength=float(settings.get("stereo_strength", 2.0)),
                    convergence=float(settings.get("convergence", 0.5)),
                    occlusion_fill=occlusion_fill,
                    stereo_render_mode=render_mode,
                    occlusion_fill_max_px=settings.get("occlusion_fill_max_px", 8),
                )
            )
        elif render_mode == "quality":
            render_settings = QualityStereoControls(
                occlusion_fill=occlusion_fill,
                occlusion_fill_max_px=settings.get("occlusion_fill_max_px", 8),
            )
        else:
            assert retained_crop_width is not None
            render_settings = StereoSplatSettings(
                max_eye_shift_fraction=(
                    float(settings.get("max_disparity_percent", 2.0))
                    / 100.0
                    * (retained_crop_width / render_shape[1])
                    / 2.0
                ),
                occlusion_fill=occlusion_fill,
            )

        def render(decoded):
            if not isinstance(decoded, _ManifestDecodedFrame):
                raise TypeError("manifest decoder returned an invalid ownership payload")
            work = decoded.work
            if normalized_geometry == "relative":
                assert isinstance(render_settings, StereoRenderSettings)
                if render_mode == "quality":
                    compact = render_relative_quality_compact(
                        renderer=self.renderer,
                        frame=decoded.take_frame(),
                        canonical=_take_relative_canonical(decoded),
                        settings=render_settings,
                    )
                else:
                    compact = self.renderer.render_relative_fast_compact(
                        decoded.take_frame(),
                        _take_relative_canonical(decoded),
                        render_settings,
                    )
            else:
                if isinstance(render_settings, QualityStereoControls):
                    compact = render_metric_primitives_compact(
                        renderer=self.renderer,
                        frame=decoded.take_frame(),
                        primitives=_take_metric_primitive(decoded),
                        controls=render_settings,
                    )
                else:
                    assert isinstance(render_settings, StereoSplatSettings)
                    compact = self.renderer.render_metric_primitives_fast_compact(
                        decoded.take_frame(),
                        _take_metric_primitive(decoded),
                        render_settings,
                    )
            if decoded.frame is not None or decoded.primitive is not None:
                raise RuntimeError("manifest render did not consume decoded ownership")
            del decoded
            publication = prepare_quality_frame(
                QualityFramePublication(
                    job_root=adopted.job_root,
                    frame_name=work.frame_name,
                    left_rgb_path=work.left_path,
                    right_rgb_path=work.right_path,
                    compact=compact,
                    geometry_mode=normalized_geometry,
                    occlusion_fill=occlusion_fill,
                    rgb_stage_fingerprint=stage_fingerprint,
                    keep_masks=keep_masks,
                    render_mode=render_mode,
                )
            )
            return _WriteItem(
                work=work,
                left_image=compact.left_image,
                right_image=compact.right_image,
                projection_stats=compact.metric_stats,
                quality_publication=publication,
                geometry_mode=normalized_geometry,
                occlusion_fill=occlusion_fill,
                rgb_stage_fingerprint=stage_fingerprint,
                keep_masks=keep_masks,
                render_mode=render_mode,
            )

        committed = 0
        progress_total = control.render_count + control.migrate_count
        progress_lock = threading.Lock()

        def write(write_item: _WriteItem) -> None:
            nonlocal committed
            _write_pair(write_item)
            with progress_lock:
                committed += 1
                self._report_progress(
                    progress_tracker,
                    processed=committed,
                    total=progress_total,
                    work_index=write_item.work.index,
                    left_path=write_item.work.left_path,
                )

        self.last_pipeline_stats = run_streaming_stereo_pipeline(
            provider=provider,
            plan=control,
            capacity=capacity_value,
            effective_workers=effective_workers,
            decode=decode,
            render=render,
            write=write,
            mutation=begin_mutation,
            pre_mutation=revalidate_disk_admission,
            configured_workers=configured_workers,
        )
        try:
            validate_stereo_provider_replay(
                provider,
                control,
                mutation_started=mutation_started,
            )
            for item in iter_stereo_action(provider, control, StereoAction.MIGRATE):
                migrate_streaming_frame_manifest(
                    diagnostics_context,
                    item.frame_name,
                    source_index=item.source_index,
                )
                with progress_lock:
                    committed += 1
                    self._report_progress(
                        progress_tracker,
                        processed=committed,
                        total=progress_total,
                        work_index=item.source_index,
                        left_path=item.left_path,
                    )
            if not diagnostics_context.mask_payloads_enabled:
                prune_streaming_unreferenced_masks(diagnostics_context)
        except StereoInputIdentityMismatch as error:
            raise StereoStageInputChangedError(str(error)) from error
        try:
            if quality_content is not None:
                revalidated = build_streamed_quality_input_manifest(adopted)
                if (
                    revalidated.raw_sha256 != quality_content.raw_sha256
                    or revalidated.fingerprint != quality_content.fingerprint
                ):
                    raise ValueError("Quality input content changed after mutation")
        except (StereoInputIdentityMismatch, OSError, TypeError, ValueError) as error:
            raise StereoStageInputChangedError(str(error)) from error
        validate_stereo_provider_replay(
            provider,
            control,
            mutation_started=True,
        )
        try:
            completion = complete_streaming_diagnostics_stage(
                diagnostics_context,
                before_complete_publish=lambda: validate_stereo_provider_replay(
                    provider,
                    control,
                    mutation_started=True,
                ),
            )
        except StereoInputIdentityMismatch as error:
            raise StereoStageInputChangedError(str(error)) from error
        if normalized_geometry == "metric":
            summary = completion.summary.value
            assert isinstance(summary, dict)
            metric_summary = summary["metric_projection"]
            assert isinstance(metric_summary, dict)
            assert completion.metric_clamp_summary is not None
            self.last_metric_clamp_summary = {
                "schema_version": 1,
                "affected_frame_count": metric_summary["affected_frame_count"],
                "mean_clamped_fraction": metric_summary["mean_clamped_fraction"],
                "max_clamped_fraction": metric_summary["max_clamped_fraction"],
                "compatibility_summary_sha256": (
                    completion.metric_clamp_summary.raw_sha256
                ),
            }
        return True

    def _build_stage_plan(
        self,
        depth_files: list[Path],
        frame_files: list[Path],
        render_shape: tuple[int, int],
        settings: dict[str, Any],
        occlusion_fill: Literal["none", "background"],
    ) -> _StereoStagePlan:
        geometry_mode = str(settings.get("stereo_geometry_mode", "relative"))
        if geometry_mode == "relative":
            return self._build_relative_plan(
                depth_files,
                frame_files,
                render_shape,
                settings,
                occlusion_fill,
            )
        if geometry_mode == "metric_camera":
            return self._build_metric_plan(
                depth_files,
                frame_files,
                render_shape,
                settings,
                occlusion_fill,
            )
        raise ValueError(f"Unsupported stereo geometry mode: {geometry_mode}")

    def _build_relative_plan(
        self,
        depth_files: list[Path],
        frame_files: list[Path],
        render_shape: tuple[int, int],
        settings: dict[str, Any],
        occlusion_fill: Literal["none", "background"],
    ) -> _StereoStagePlan:
        metadata = self._get_canonical_metadata(depth_files, frame_files)
        stereo_strength = float(settings.get("stereo_strength", 2.0))
        convergence = float(settings.get("convergence", 0.5))

        def decode(work: _FileWorkItem):
            return _decode_relative_work_item(
                work,
                encoding_scale=float(metadata["encoding_scale"]),
                render_shape=render_shape,
            )

        render_settings = StereoRenderSettings(
            stereo_strength=stereo_strength,
            convergence=convergence,
            occlusion_fill=occlusion_fill,
            stereo_render_mode=cast(
                Literal["fast", "quality"],
                str(settings.get("stereo_render_mode", "fast")),
            ),
            occlusion_fill_max_px=settings.get("occlusion_fill_max_px", 8),
        )
        plan_type = (
            _QualityRelativePlan
            if render_settings.stereo_render_mode == "quality"
            else _FastRelativePlan
        )
        return plan_type(
            source_metadata=metadata,
            decode=decode,
            settings=render_settings,
        )

    def _build_metric_plan(
        self,
        depth_files: list[Path],
        frame_files: list[Path],
        render_shape: tuple[int, int],
        settings: dict[str, Any],
        occlusion_fill: Literal["none", "background"],
    ) -> _StereoStagePlan:
        store, metadata = self._get_metric_store(depth_files, frame_files)
        requested_convergence = settings.get("metric_convergence_distance", "auto")
        effective_convergence = (
            float(metadata["convergence"]["resolved_auto_distance_m"])
            if requested_convergence == "auto"
            else float(requested_convergence)
        )
        retained_crop_width, _retained_crop_height = calculate_center_crop_dimensions(
            render_shape[1],
            render_shape[0],
            float(settings.get("crop_factor", 1.0)),
        )
        virtual_baseline_mm = float(settings.get("virtual_baseline_mm", 63.0))
        max_disparity_percent = float(settings.get("max_disparity_percent", 2.0))
        render_mode = cast(
            Literal["fast", "quality"],
            str(settings.get("stereo_render_mode", "fast")),
        )

        def decode(work: _FileWorkItem):
            return _decode_metric_work_item(
                work,
                store=store,
                render_shape=render_shape,
                virtual_baseline_mm=virtual_baseline_mm,
                convergence_distance_m=effective_convergence,
                max_disparity_percent=max_disparity_percent,
                retained_crop_width=retained_crop_width,
            )

        render_settings: StereoSplatSettings | QualityStereoControls
        if render_mode == "quality":
            render_settings = QualityStereoControls(
                occlusion_fill=occlusion_fill,
                occlusion_fill_max_px=settings.get("occlusion_fill_max_px", 8),
            )
        else:
            render_settings = StereoSplatSettings(
                max_eye_shift_fraction=(
                    max_disparity_percent / 100.0 * (retained_crop_width / render_shape[1]) / 2.0
                ),
                occlusion_fill=occlusion_fill,
            )

        if render_mode == "quality":
            assert isinstance(render_settings, QualityStereoControls)
            return _QualityMetricPlan(
                source_metadata=metadata,
                decode=decode,
                settings=render_settings,
                retained_crop_width=retained_crop_width,
                effective_convergence=effective_convergence,
            )
        assert isinstance(render_settings, StereoSplatSettings)
        return _FastMetricPlan(
            source_metadata=metadata,
            decode=decode,
            settings=render_settings,
            retained_crop_width=retained_crop_width,
            effective_convergence=effective_convergence,
        )

    def create_stereo_pairs_from_files(  # noqa: C901
        self,
        frame_files: list[Path],
        depth_files: list[Path],
        directories: dict[str, Path],
        settings: dict[str, Any],
        progress_tracker=None,
    ) -> bool:
        """Decode, render, and atomically write file-backed stereo pairs."""

        self.last_pipeline_stats = None
        self.last_metric_clamp_summary = None
        geometry_mode = str(settings.get("stereo_geometry_mode", "relative"))
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
        try:
            left_dir = directories["left_frames"]
            right_dir = directories["right_frames"]
            left_dir.mkdir(parents=True, exist_ok=True)
            right_dir.mkdir(parents=True, exist_ok=True)
            render_shape = self._read_render_shape(frame_files[0])
            occlusion_fill = self._occlusion_fill(settings)
            plan = self._build_stage_plan(
                depth_files,
                frame_files,
                render_shape,
                settings,
                occlusion_fill,
            )
            quality_context: DiagnosticsStageContext | None = None
            if isinstance(plan, (_QualityRelativePlan, _QualityMetricPlan)):
                content = self._build_quality_content_manifest(
                    job_root=left_dir.parent,
                    frame_files=frame_files,
                    depth_files=depth_files,
                    plan=plan,
                )
                stage_metadata = self._quality_stereo_stage_metadata(
                    plan=plan,
                    content=content,
                    render_shape=render_shape,
                    settings=settings,
                    occlusion_fill=occlusion_fill,
                )
                stage_changed = self._prepare_quality_stereo_stage(
                    left_dir,
                    right_dir,
                    stage_metadata,
                    content,
                )
                quality_context = self._quality_diagnostics_context(
                    job_root=left_dir.parent,
                    metadata=stage_metadata,
                    content=content,
                    plan=plan,
                    occlusion_fill=occlusion_fill,
                    keep_masks=bool(settings.get("keep_intermediates", True)),
                    left_rgb_parent=left_dir,
                    right_rgb_parent=right_dir,
                )
                diagnostic_audit = audit_diagnostics_stage(quality_context)
                if diagnostic_audit.disposition != "complete":
                    begin_diagnostics_stage(quality_context)
                work_plan = self._build_quality_work_items(
                    frame_files,
                    depth_files,
                    left_dir,
                    right_dir,
                    diagnostic_audit.reusable_frame_names,
                    diagnostic_audit.migrate_frame_names,
                )
            else:
                stage_metadata = self._stereo_stage_metadata(
                    plan.source_metadata,
                    frame_files,
                    render_shape,
                    settings,
                    geometry_mode=plan.geometry_mode,
                    occlusion_fill=occlusion_fill,
                    retained_crop_width=plan.retained_crop_width,
                    effective_convergence=plan.effective_convergence,
                )
                stage_changed = self._prepare_stereo_stage(left_dir, right_dir, stage_metadata)
                compact_fast = (
                    plan.geometry_mode == "relative"
                    and hasattr(self.renderer, "render_relative_fast_compact")
                ) or (
                    plan.geometry_mode == "metric_camera"
                    and hasattr(self.renderer, "render_geometry_fast_compact")
                )
                if compact_fast:
                    if stage_changed:
                        shutil.rmtree(
                            left_dir.parent / "04_stereo_diagnostics",
                            ignore_errors=True,
                        )
                    quality_context = self._fast_diagnostics_context(
                        job_root=left_dir.parent,
                        metadata=stage_metadata,
                        frame_files=frame_files,
                        render_shape=render_shape,
                        geometry_mode=plan.geometry_mode,
                        occlusion_fill=occlusion_fill,
                        keep_masks=bool(settings.get("keep_intermediates", True)),
                        left_rgb_parent=left_dir,
                        right_rgb_parent=right_dir,
                    )
                    diagnostic_audit = audit_diagnostics_stage(quality_context)
                    if diagnostic_audit.disposition != "complete":
                        begin_diagnostics_stage(quality_context)
                    work_plan = self._build_quality_work_items(
                        frame_files,
                        depth_files,
                        left_dir,
                        right_dir,
                        diagnostic_audit.reusable_frame_names,
                        diagnostic_audit.migrate_frame_names,
                    )
                else:
                    work_plan = self._build_file_work_items(
                        frame_files,
                        depth_files,
                        left_dir,
                        right_dir,
                        render_shape,
                        metric_mode=plan.geometry_mode == "metric_camera",
                    )
            if stage_changed:
                self._reset_downstream_stages(directories)
            if work_plan.repaired_outputs and not stage_changed:
                self._reset_downstream_stages(directories)
            if work_plan.render_count == 0:
                print(f"  Reusing {work_plan.reuse_count} existing stereo pairs")
                if quality_context is not None:
                    if work_plan.migrate_count:
                        migrate_frame_manifests_sync(
                            quality_context,
                            lambda: work_plan.iter_frame_names(StereoAction.MIGRATE),
                        )
                    if diagnostic_audit.disposition != "complete":
                        complete_diagnostics_stage(quality_context)
                    if plan.geometry_mode == "metric_camera":
                        self._load_quality_metric_summary(left_dir)
                elif plan.geometry_mode == "metric_camera":
                    self._summarize_metric_clamps(frame_files, left_dir)
                return True

            workers = validate_stereo_io_workers(
                settings.get("stereo_io_workers", _default_stereo_io_workers())
            )
            capacity = calculate_stereo_pipeline_capacity(
                render_shape[1],
                render_shape[0],
                workers,
            )
            if plan.render_mode == "quality":
                capacity = 1
                workers = 1
            print(
                f"  Using {workers} stereo I/O workers with " f"{capacity} bounded frame slots..."
            )
            self._run_file_pipeline(
                work_plan,
                decode=plan.decode,
                render_settings=plan.render_settings,
                render_mode=plan.render_mode,
                geometry_mode=plan.geometry_mode,
                occlusion_fill=occlusion_fill,
                rgb_stage_fingerprint=str(stage_metadata["fingerprint"]),
                keep_masks=bool(settings.get("keep_intermediates", True)),
                workers=workers,
                capacity=capacity,
                completed=work_plan.reuse_count,
                total_frames=len(frame_files),
                progress_tracker=progress_tracker,
            )
            if quality_context is not None:
                if work_plan.migrate_count:
                    migrate_frame_manifests_sync(
                        quality_context,
                        lambda: work_plan.iter_frame_names(StereoAction.MIGRATE),
                    )
                complete_diagnostics_stage(quality_context)
                if plan.geometry_mode == "metric_camera":
                    self._load_quality_metric_summary(left_dir)
            elif plan.geometry_mode == "metric_camera":
                self._summarize_metric_clamps(frame_files, left_dir)
            return True
        except Exception as error:
            return _stereo_creation_failed(error)

    def _stereo_stage_metadata(
        self,
        canonical_metadata: dict[str, Any],
        frame_files: list[Path],
        render_shape: tuple[int, int],
        settings: dict[str, Any],
        *,
        geometry_mode: str,
        occlusion_fill: Literal["none", "background"],
        retained_crop_width: int | None,
        effective_convergence: float | None,
    ) -> dict[str, Any]:
        device = getattr(self.renderer, "device", None)
        device_type = getattr(device, "type", None) or "custom"
        metadata = {
            "schema_version": STEREO_STAGE_SCHEMA_VERSION,
            "algorithm_version": STEREO_STAGE_ALGORITHM_VERSION,
            "geometry_mode": geometry_mode,
            "frame_names": [path.name for path in frame_files],
            "render_shape": [int(render_shape[0]), int(render_shape[1])],
            "occlusion_fill": occlusion_fill,
            "renderer_device_type": str(device_type),
            "encoding": "uint8_png",
        }
        render_mode = str(settings.get("stereo_render_mode", "fast"))
        if render_mode == "quality":
            metadata["render_mode"] = "quality"
            metadata["occlusion_fill_max_px"] = int(settings.get("occlusion_fill_max_px", 8))
        if geometry_mode == "relative":
            metadata.update(
                {
                    "source_canonical_fingerprint": canonical_metadata["fingerprint"],
                    "render_settings": {
                        "stereo_strength": float(settings.get("stereo_strength", 2.0)),
                        "convergence": float(settings.get("convergence", 0.5)),
                        "occlusion_fill": occlusion_fill,
                    },
                }
            )
        else:
            assert retained_crop_width is not None and effective_convergence is not None
            requested_convergence = settings.get("metric_convergence_distance", "auto")
            metadata.update(
                {
                    "source_metric_fingerprint": canonical_metadata["fingerprint"],
                    "projection_algorithm_version": METRIC_PROJECTION_ALGORITHM_VERSION,
                    "source_width": int(render_shape[1]),
                    "retained_crop_width": int(retained_crop_width),
                    "center_crop_algorithm_version": CENTER_CROP_ALGORITHM_VERSION,
                    "sample_aspect_ratio": "1:1",
                    "virtual_baseline_mm": float(settings.get("virtual_baseline_mm", 63.0)),
                    "requested_convergence_distance": (
                        "auto" if requested_convergence == "auto" else float(requested_convergence)
                    ),
                    "effective_convergence_distance_m": float(effective_convergence),
                    "max_disparity_percent": float(settings.get("max_disparity_percent", 2.0)),
                }
            )
        metadata["fingerprint"] = canonical_json_hash(metadata)
        return metadata

    @staticmethod
    def _build_quality_content_manifest(
        *,
        job_root: Path,
        frame_files: list[Path],
        depth_files: list[Path],
        plan: _QualityRelativePlan | _QualityMetricPlan,
    ):
        from ...io.content_manifest import build_quality_input_manifest

        if isinstance(plan, _QualityRelativePlan):
            return build_quality_input_manifest(
                job_root=job_root,
                frame_files=frame_files,
                geometry_files=depth_files,
                geometry_mode="relative",
                relative_encoding_scale=np.float32(plan.source_metadata["encoding_scale"]),
                metric_native_shape=None,
            )
        return build_quality_input_manifest(
            job_root=job_root,
            frame_files=frame_files,
            geometry_files=depth_files,
            geometry_mode="metric",
            relative_encoding_scale=None,
            metric_native_shape=tuple(plan.source_metadata["native_shape"]),
        )

    @staticmethod
    def _quality_stereo_stage_metadata(
        *,
        plan: _QualityRelativePlan | _QualityMetricPlan,
        content: Any,
        render_shape: tuple[int, int],
        settings: dict[str, Any],
        occlusion_fill: Literal["none", "background"],
    ) -> dict[str, Any]:
        content_payload = content.payload
        if isinstance(plan, _QualityRelativePlan):
            projection: dict[str, Any] = {
                "kind": "relative",
                "stereo_strength": float(plan.settings.stereo_strength),
                "convergence": float(plan.settings.convergence),
            }
            geometry_mode = "relative"
        else:
            requested = settings.get("metric_convergence_distance", "auto")
            projection = {
                "kind": "metric",
                "projection_algorithm_version": METRIC_PROJECTION_ALGORITHM_VERSION,
                "source_width": int(render_shape[1]),
                "retained_crop_width": int(plan.retained_crop_width),
                "center_crop_algorithm_version": CENTER_CROP_ALGORITHM_VERSION,
                "sample_aspect_ratio": "1:1",
                "virtual_baseline_mm": float(settings.get("virtual_baseline_mm", 63.0)),
                "requested_convergence_distance": (
                    "auto" if requested == "auto" else float(requested)
                ),
                "effective_convergence_distance_m": float(plan.effective_convergence),
                "max_disparity_percent": float(settings.get("max_disparity_percent", 2.0)),
            }
            geometry_mode = "metric"
        if occlusion_fill == "background":
            configured_limit = int(settings.get("occlusion_fill_max_px", 8))
            repair_policy: dict[str, Any] = {
                "configured_limit_1080p": configured_limit,
                "scaled_safe_limit_px": max(
                    1,
                    math.floor(configured_limit * render_shape[0] / 1080.0 + 0.5),
                ),
                "predicted_gap_policy": "max-four-neighbour-eye-shift-v1",
                "local_limit_formula": "min-scaled-safe-predicted-plus2-v1",
            }
        else:
            repair_policy = {
                "configured_limit_1080p": None,
                "scaled_safe_limit_px": None,
                "predicted_gap_policy": None,
                "local_limit_formula": None,
            }
        metadata: dict[str, Any] = {
            "schema_version": 2,
            "algorithm_version": QUALITY_STEREO_ALGORITHM_VERSION,
            "geometry_mode": geometry_mode,
            "frame_names": list(content_payload["frame_names"]),
            "render_shape": [int(render_shape[0]), int(render_shape[1])],
            "occlusion_fill": occlusion_fill,
            "encoding": "uint8_png",
            "source_guide_fingerprint": content_payload["source_guide_fingerprint"],
            "native_geometry_fingerprint": content_payload["native_geometry_fingerprint"],
            "quality_input_manifest_sha256": content.raw_sha256,
            "projection": projection,
            "repair_policy": repair_policy,
        }
        metadata["fingerprint"] = canonical_json_hash(metadata)
        return metadata

    @staticmethod
    def _prepare_quality_stereo_stage(
        left_dir: Path,
        right_dir: Path,
        expected_metadata: dict[str, Any],
        content: Any,
    ) -> bool:
        from ...io.content_manifest import publish_quality_input_manifest

        job_root = left_dir.parent
        diagnostics_root = job_root / "04_stereo_diagnostics"
        metadata_path = left_dir / "metadata.json"
        content_path = diagnostics_root / "quality_input_manifest.json"
        try:
            existing_metadata_raw = metadata_path.read_bytes()
            existing_metadata = json.loads(existing_metadata_raw.decode("ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            existing_metadata = None
            existing_metadata_raw = None
        try:
            existing_content_raw = content_path.read_bytes()
        except OSError:
            existing_content_raw = None
        expected_metadata_raw = canonical_json_bytes(expected_metadata)
        stage_changed = (
            existing_metadata != expected_metadata
            or existing_metadata_raw != expected_metadata_raw
            or existing_content_raw != content.raw
        )
        if stage_changed:
            for directory in (left_dir, right_dir):
                for path in directory.iterdir():
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
            shutil.rmtree(diagnostics_root, ignore_errors=True)
        publish_quality_input_manifest(content_path, content)
        _atomic_write_canonical_json(metadata_path, expected_metadata)
        return stage_changed

    @staticmethod
    def _quality_diagnostics_context(
        *,
        job_root: Path,
        metadata: dict[str, Any],
        content: Any,
        plan: _QualityRelativePlan | _QualityMetricPlan,
        occlusion_fill: Literal["none", "background"],
        keep_masks: bool,
        left_rgb_parent: Path,
        right_rgb_parent: Path,
    ) -> DiagnosticsStageContext:
        return DiagnosticsStageContext(
            job_root=job_root,
            frame_names=tuple(content.payload["frame_names"]),
            render_shape=tuple(metadata["render_shape"]),
            render_mode="quality",
            geometry_mode=("relative" if isinstance(plan, _QualityRelativePlan) else "metric"),
            occlusion_fill=occlusion_fill,
            mask_payloads_enabled=keep_masks,
            rgb_stage_fingerprint=str(metadata["fingerprint"]),
            source_guide_fingerprint=str(metadata["source_guide_fingerprint"]),
            native_geometry_fingerprint=str(metadata["native_geometry_fingerprint"]),
            quality_input_manifest_sha256=content.raw_sha256,
            native_geometry_sample_count=(
                int(plan.source_metadata["native_shape"][0])
                * int(plan.source_metadata["native_shape"][1])
            ),
            left_rgb_parent=left_rgb_parent,
            right_rgb_parent=right_rgb_parent,
        )

    @staticmethod
    def _fast_diagnostics_context(
        *,
        job_root: Path,
        metadata: dict[str, Any],
        frame_files: list[Path],
        render_shape: tuple[int, int],
        geometry_mode: str,
        occlusion_fill: Literal["none", "background"],
        keep_masks: bool,
        left_rgb_parent: Path,
        right_rgb_parent: Path,
    ) -> DiagnosticsStageContext:
        source_provenance = metadata.get(
            "source_canonical_fingerprint",
            metadata.get("source_metric_fingerprint", metadata["fingerprint"]),
        )
        return DiagnosticsStageContext(
            job_root=job_root,
            frame_names=tuple(path.stem for path in frame_files),
            render_shape=render_shape,
            render_mode="fast",
            geometry_mode=("metric" if geometry_mode == "metric_camera" else "relative"),
            occlusion_fill=occlusion_fill,
            mask_payloads_enabled=keep_masks,
            rgb_stage_fingerprint=str(metadata["fingerprint"]),
            source_guide_fingerprint=str(source_provenance),
            native_geometry_fingerprint=None,
            quality_input_manifest_sha256=None,
            left_rgb_parent=left_rgb_parent,
            right_rgb_parent=right_rgb_parent,
        )

    @staticmethod
    def _build_quality_work_items(
        frame_files: list[Path],
        depth_files: list[Path],
        left_dir: Path,
        right_dir: Path,
        reusable_frame_names: tuple[str, ...],
        migrate_frame_names: tuple[str, ...],
    ) -> _FileWorkPlan:
        reusable = set(reusable_frame_names)
        migrate = set(migrate_frame_names)
        if reusable & migrate:
            raise ValueError("diagnostics reuse and migration actions overlap")
        actions = PackedStereoActions(len(frame_files))
        render_count = 0
        repaired_outputs = False
        for index, (frame_file, depth_file) in enumerate(zip(frame_files, depth_files)):
            frame_name = frame_file.stem
            if frame_name in reusable:
                actions.set(index, StereoAction.REUSE)
                continue
            if frame_name in migrate:
                actions.set(index, StereoAction.MIGRATE)
                continue
            actions.set(index, StereoAction.RENDER)
            render_count += 1
            left_path = left_dir / f"{frame_name}.png"
            right_path = right_dir / f"{frame_name}.png"
            diagnostic_dir = left_dir.parent / "04_stereo_diagnostics" / "frames" / frame_name
            repaired_outputs = repaired_outputs or any(
                path.exists() for path in (left_path, right_path, diagnostic_dir)
            )
            left_path.unlink(missing_ok=True)
            right_path.unlink(missing_ok=True)
            shutil.rmtree(diagnostic_dir, ignore_errors=True)
        if len(reusable) + len(migrate) + render_count != len(frame_files):
            raise ValueError("diagnostics action cardinality differs from the frame sequence")
        return _FileWorkPlan(
            frame_files=frame_files,
            depth_files=depth_files,
            left_dir=left_dir,
            right_dir=right_dir,
            actions=actions,
            render_count=render_count,
            reuse_count=len(reusable),
            migrate_count=len(migrate),
            repaired_outputs=repaired_outputs,
            metric_mode=False,
        )

    def _load_quality_metric_summary(self, left_dir: Path) -> None:
        path = left_dir / "clamp_summary.json"
        payload = json.loads(path.read_text(encoding="ascii"))
        self.last_metric_clamp_summary = payload
        fractions = payload["clamped_fractions"]
        frame_names = payload["frame_names"]
        for frame_name, fraction in zip(frame_names, fractions):
            if float(fraction) > 0.05:
                print(
                    "Warning: metric disparity was clamped above 5% of valid pixels "
                    f"starting at source frame {frame_name}"
                )
                break

    @staticmethod
    def _prepare_stereo_stage(
        left_dir: Path,
        right_dir: Path,
        expected_metadata: dict[str, Any],
    ) -> bool:
        metadata_path = left_dir / "metadata.json"
        existing_metadata = None
        try:
            existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

        has_outputs = any(left_dir.iterdir()) or any(right_dir.iterdir())
        stage_changed = existing_metadata is not None or has_outputs
        stage_changed = stage_changed and existing_metadata != expected_metadata
        if stage_changed:
            for directory in (left_dir, right_dir):
                for path in directory.iterdir():
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
            existing_metadata = None
        if existing_metadata != expected_metadata:
            _atomic_write_json(metadata_path, expected_metadata)
        return stage_changed

    @staticmethod
    def _downstream_stage_directories(directories: dict[str, Path]) -> tuple[Path, ...]:
        return tuple(
            Path(directories[name])
            for name in (
                "left_distorted",
                "right_distorted",
                "left_cropped",
                "right_cropped",
                "left_upscaled",
                "right_upscaled",
                "vr_frames",
            )
            if directories.get(name) is not None
        )

    @classmethod
    def _reset_downstream_stages(cls, directories: dict[str, Path]) -> None:
        downstream = cls._downstream_stage_directories(directories)
        if not downstream:
            return
        job_root = Path(directories.get("base", downstream[0].parent))
        with SafeTreeMutation(job_root, downstream) as mutation:
            for directory in downstream:
                mutation.clear_root(directory)

    def _run_file_pipeline(
        self,
        work_plan: _FileWorkPlan,
        *,
        decode: _DecodeFrame,
        render_settings: StereoRenderSettings | StereoSplatSettings | QualityStereoControls,
        render_mode: Literal["fast", "quality"],
        geometry_mode: str,
        occlusion_fill: Literal["none", "background"],
        rgb_stage_fingerprint: str,
        keep_masks: bool,
        workers: int,
        capacity: int,
        completed: int,
        total_frames: int,
        progress_tracker,
    ) -> None:
        pipeline = _StereoFilePipeline(
            renderer=self.renderer,
            work_factory=work_plan.iter_render,
            work_count=work_plan.render_count,
            decode=decode,
            render_settings=render_settings,
            render_mode=render_mode,
            geometry_mode=geometry_mode,
            occlusion_fill=occlusion_fill,
            rgb_stage_fingerprint=rgb_stage_fingerprint,
            keep_masks=keep_masks,
            workers=workers,
            capacity=capacity,
            completed=completed,
            total_frames=total_frames,
            report_progress=lambda **values: self._report_progress(
                progress_tracker,
                **values,
            ),
        )
        try:
            self.last_pipeline_stats = pipeline.run()
        finally:
            self.last_pipeline_stats = pipeline.metrics.snapshot()

    @staticmethod
    def _report_progress(
        progress_tracker,
        *,
        processed: int,
        total: int,
        work_index: int,
        left_path: Path | None,
    ) -> None:
        if progress_tracker and (processed % 5 == 0 or processed == total):
            progress_tracker.update_progress(
                "Creating stereo pairs",
                phase="stereo_generation",
                frame_num=processed,
                step_name="Stereo Pair Creation",
                step_progress=processed,
                step_total=total,
            )
        if (
            progress_tracker
            and left_path is not None
            and hasattr(progress_tracker, "send_preview_frame")
            and (work_index % PREVIEW_FRAME_SAMPLE_RATE == 0 or processed == total)
        ):
            progress_tracker.send_preview_frame(left_path, "stereo_left", processed)

    @staticmethod
    def _read_render_shape(frame_path: Path) -> tuple[int, int]:
        header = read_png_header(frame_path)
        if header is None:
            raise OSError(f"Could not read PNG header: {frame_path}")
        return header.height, header.width

    @staticmethod
    def _build_file_work_items(
        frame_files: list[Path],
        depth_files: list[Path],
        left_dir: Path,
        right_dir: Path,
        render_shape: tuple[int, int],
        *,
        metric_mode: bool,
    ) -> _FileWorkPlan:
        actions = PackedStereoActions(len(frame_files))
        render_count = 0
        completed = 0
        repaired_outputs = False
        stat_dir = left_dir / "clamp_stats"
        for directory in (left_dir, right_dir, stat_dir):
            if directory.is_dir():
                for temporary in directory.glob("*.tmp"):
                    temporary.unlink(missing_ok=True)
        if metric_mode:
            stat_dir.mkdir(parents=True, exist_ok=True)
        for index, (frame_file, depth_file) in enumerate(zip(frame_files, depth_files)):
            frame_name = frame_file.stem
            left_path = left_dir / f"{frame_name}.png"
            right_path = right_dir / f"{frame_name}.png"
            stat_path = stat_dir / f"{frame_name}.json" if metric_mode else None
            if StereoPairGenerator._stereo_outputs_are_valid(
                left_path,
                right_path,
                stat_path,
                frame_name,
                render_shape,
            ):
                actions.set(index, StereoAction.REUSE)
                completed += 1
                continue
            actions.set(index, StereoAction.RENDER)
            render_count += 1
            output_paths = [left_path, right_path]
            if stat_path is not None:
                output_paths.append(stat_path)
            repaired_outputs = repaired_outputs or any(path.exists() for path in output_paths)
            for path in output_paths:
                path.unlink(missing_ok=True)
        return _FileWorkPlan(
            frame_files=frame_files,
            depth_files=depth_files,
            left_dir=left_dir,
            right_dir=right_dir,
            actions=actions,
            render_count=render_count,
            reuse_count=completed,
            migrate_count=0,
            repaired_outputs=repaired_outputs,
            metric_mode=metric_mode,
        )

    @staticmethod
    def _stereo_outputs_are_valid(
        left_path: Path,
        right_path: Path,
        stat_path: Path | None,
        frame_name: str,
        render_shape: tuple[int, int],
    ) -> bool:
        for path in (left_path, right_path):
            if not png_header_matches(path, shape=(*render_shape, 3), bit_depth=8):
                return False
        return (
            stat_path is None
            or StereoPairGenerator._read_metric_stat_sidecar(
                stat_path,
                frame_name,
            )
            is not None
        )

    @staticmethod
    def _read_metric_stat_sidecar(
        path: Path,
        frame_name: str,
    ) -> MetricProjectionStats | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "frame_name",
            "valid_pixel_count",
            "clamped_pixel_count",
            "clamped_fraction",
        }:
            return None
        if (
            payload.get("schema_version") != METRIC_CLAMP_STATS_SCHEMA_VERSION
            or payload.get("frame_name") != frame_name
        ):
            return None
        try:
            return MetricProjectionStats(
                payload["valid_pixel_count"],
                payload["clamped_pixel_count"],
                payload["clamped_fraction"],
            )
        except (TypeError, ValueError):
            return None

    def _summarize_metric_clamps(self, frame_files: list[Path], left_dir: Path) -> None:
        fractions: list[float] = []
        earliest_warning: str | None = None
        for frame_file in frame_files:
            frame_name = frame_file.stem
            stats = self._read_metric_stat_sidecar(
                left_dir / "clamp_stats" / f"{frame_name}.json",
                frame_name,
            )
            if stats is None:
                raise ValueError(f"Metric clamp statistics are invalid for {frame_name}")
            fractions.append(stats.clamped_fraction)
            if earliest_warning is None and stats.clamped_fraction > 0.05:
                earliest_warning = frame_name

        summary: dict[str, Any] = {
            "schema_version": METRIC_CLAMP_STATS_SCHEMA_VERSION,
            "frame_names": [path.stem for path in frame_files],
            "clamped_fractions": fractions,
            "affected_frame_count": sum(fraction > 0.0 for fraction in fractions),
            "mean_clamped_fraction": float(np.mean(fractions)) if fractions else 0.0,
            "max_clamped_fraction": max(fractions, default=0.0),
        }
        _atomic_write_json(left_dir / "clamp_summary.json", summary)
        self.last_metric_clamp_summary = summary
        if earliest_warning is not None:
            print(
                "Warning: metric disparity was clamped above 5% of valid pixels "
                f"starting at source frame {earliest_warning}"
            )

    @staticmethod
    def _get_canonical_metadata(
        depth_files: list[Path],
        frame_files: list[Path],
    ) -> dict[str, Any]:
        """Load either strict base or stabilized render-disparity metadata."""

        artifact = validate_render_disparity_input(depth_files, frame_files)
        metadata = dict(artifact.metadata)
        metadata["fingerprint"] = artifact.fingerprint
        return metadata

    @staticmethod
    def _get_metric_store(
        depth_files: list[Path],
        frame_files: list[Path],
    ) -> tuple[MetricGeometryStore, dict[str, Any]]:
        try:
            if not depth_files:
                raise ValueError("Metric geometry files are required")
            metadata = MetricGeometryStore.read_metadata(depth_files[0].parent)
            if metadata is None:
                raise ValueError("Metric geometry metadata is missing or malformed")
            store = MetricGeometryStore.open_existing(
                depth_files[0].parent,
                frame_names=[path.name for path in frame_files],
                source_raw_fingerprint=cast(str, metadata.get("source_raw_fingerprint")),
                source_frame_fingerprint=cast(str, metadata.get("source_frame_fingerprint")),
                candidate_scene_fingerprint=cast(
                    str,
                    metadata.get("candidate_scene_fingerprint"),
                ),
            )
            expected_paths = [path.resolve() for path in store.complete_files]
            if [path.resolve() for path in depth_files] != expected_paths:
                raise ValueError("Metric geometry files do not match the metadata path manifest")
            return store, store.metadata
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise ValueError(METRIC_STEREO_COMPLETION_ERROR) from error
