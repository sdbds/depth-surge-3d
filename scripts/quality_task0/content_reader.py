"""Bounded one-frame metric content ownership for the Task 0 audit gate."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import struct
from typing import Iterable, Iterator

import numpy as np

from . import metric_npz


QUALITY_CONTENT_STREAM_BYTES = metric_npz.METRIC_NPZ_WRITE_STREAM_BYTES
QUALITY_METRIC_AUDIT_PHASE = "metric-content-audit"


class QualityHostBudgetError(MemoryError):
    """A phase could not acquire its explicitly budgeted host ownership."""

    def __init__(self, phase: str, path: Path, requested_bytes: int) -> None:
        self.phase = phase
        self.path = Path(path)
        self.requested_bytes = int(requested_bytes)
        super().__init__(
            f"{phase} host allocation failed for {self.path} "
            f"({self.requested_bytes} requested bytes)"
        )


@dataclass(frozen=True)
class MetricPayloadIdentity:
    """Scalar identity retained after an owned audit frame is released."""

    relative_path: str
    sha256: str
    byte_count: int
    archive_form: metric_npz.ArchiveForm
    native_shape: tuple[int, int]
    focal_x_normalized_float32_bits: str


@dataclass
class OwnedMetricAuditFrame:
    """Exactly one inverse array plus one validity array for one audited file."""

    path: Path
    inverse_depth: np.ndarray | None
    valid: np.ndarray | None
    focal_x_normalized: np.float32
    archive_form: metric_npz.ArchiveForm
    sha256: str
    byte_count: int
    maximum_stream_bytes: int
    stream_workspace_peak_bytes: int
    value_scratch_peak_bytes: int
    _released: bool = False

    @property
    def released(self) -> bool:
        return self._released

    @property
    def owned_payload_bytes(self) -> int:
        if self.inverse_depth is None or self.valid is None:
            return 0
        return int(self.inverse_depth.nbytes + self.valid.nbytes)

    @property
    def native_shape(self) -> tuple[int, int]:
        if self.inverse_depth is None:
            raise RuntimeError("metric audit frame is released")
        return int(self.inverse_depth.shape[0]), int(self.inverse_depth.shape[1])

    def identity(self) -> MetricPayloadIdentity:
        if self._released:
            raise RuntimeError("metric audit frame is released")
        focal_bits = struct.unpack(
            "<I",
            struct.pack("<f", float(self.focal_x_normalized)),
        )[0]
        return MetricPayloadIdentity(
            relative_path=self.path.as_posix(),
            sha256=self.sha256,
            byte_count=self.byte_count,
            archive_form=self.archive_form,
            native_shape=self.native_shape,
            focal_x_normalized_float32_bits=f"{focal_bits:08x}",
        )

    def release(self) -> None:
        if self._released:
            return
        self.inverse_depth = None
        self.valid = None
        self._released = True

    def __enter__(self) -> OwnedMetricAuditFrame:
        if self._released:
            raise RuntimeError("metric audit frame is released")
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


@dataclass
class MetricSequenceAuditTrace:
    """Lifetime trace proving a capacity-one metric audit phase."""

    events: list[str] = field(default_factory=list)
    active_frames_peak: int = 0
    active_payload_bytes_peak: int = 0
    maximum_stream_bytes: int = 0
    stream_workspace_peak_bytes: int = 0
    maximum_value_scratch_bytes: int = 0


@dataclass(frozen=True)
class MetricSequenceAuditResult:
    identities: tuple[MetricPayloadIdentity, ...]
    trace: MetricSequenceAuditTrace


def _default_allocator(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    return np.empty(shape, dtype=dtype, order="C")


def audit_metric_payload(
    path: Path,
    *,
    expected_shape: tuple[int, int],
    allocator=_default_allocator,
) -> OwnedMetricAuditFrame:
    """Audit one regular payload without ``zipfile``, ``np.load``, or a copy constructor."""

    path = Path(path)
    requested_bytes = 5 * int(expected_shape[0]) * int(expected_shape[1])
    try:
        audit = metric_npz.audit_metric_npz(
            path,
            expected_shape=expected_shape,
            allocator=allocator,
        )
    except metric_npz.MetricNpzAllocationError as error:
        raise QualityHostBudgetError(
            QUALITY_METRIC_AUDIT_PHASE,
            path,
            requested_bytes,
        ) from error
    return OwnedMetricAuditFrame(
        path=path,
        inverse_depth=audit.inverse_depth,
        valid=audit.valid,
        focal_x_normalized=audit.focal_x_normalized,
        archive_form=audit.layout.archive_form,
        sha256=audit.sha256,
        byte_count=audit.byte_count,
        maximum_stream_bytes=audit.maximum_stream_bytes,
        stream_workspace_peak_bytes=audit.stream_workspace_peak_bytes,
        value_scratch_peak_bytes=audit.value_scratch_peak_bytes,
    )


def _iter_paths(paths: Iterable[Path]) -> Iterator[Path]:
    for path in paths:
        yield Path(path)


def audit_metric_sequence(
    paths: Iterable[Path],
    *,
    expected_shape: tuple[int, int],
) -> MetricSequenceAuditResult:
    """Audit in source order and drop every 5*G ownership before the next open."""

    identities: list[MetricPayloadIdentity] = []
    trace = MetricSequenceAuditTrace()
    active_frames = 0
    for index, path in enumerate(_iter_paths(paths)):
        trace.events.append(f"open:{index}")
        frame = audit_metric_payload(path, expected_shape=expected_shape)
        active_frames += 1
        trace.active_frames_peak = max(trace.active_frames_peak, active_frames)
        trace.active_payload_bytes_peak = max(
            trace.active_payload_bytes_peak,
            frame.owned_payload_bytes,
        )
        trace.maximum_stream_bytes = max(
            trace.maximum_stream_bytes,
            frame.maximum_stream_bytes,
        )
        trace.stream_workspace_peak_bytes = max(
            trace.stream_workspace_peak_bytes,
            frame.stream_workspace_peak_bytes,
        )
        trace.maximum_value_scratch_bytes = max(
            trace.maximum_value_scratch_bytes,
            frame.value_scratch_peak_bytes,
        )
        try:
            identities.append(frame.identity())
        finally:
            frame.release()
            active_frames -= 1
            trace.events.append(f"release:{index}")
    return MetricSequenceAuditResult(tuple(identities), trace)
