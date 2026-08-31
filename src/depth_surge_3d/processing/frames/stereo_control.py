"""Replayable O(1)-item stereo control plane and parked-thread implementation."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import ctypes
from enum import IntEnum
import hashlib
import os
from pathlib import Path
import queue
import re
import secrets
import struct
import sys
import threading
import time
from typing import Callable, Iterable, Iterator, Protocol, cast

from ...io.stereo_contract import (
    STEREO_CONTROL_FRAME_CAP,
    StereoControlBudgetError,
    StereoControlError,
)

STEREO_CONTROL_ACTION_BYTES = 1 * 1024 * 1024
STEREO_CONTROL_OVERHEAD = 4 * 1024 * 1024
STEREO_HOST_BUDGET = 512 * 1024 * 1024
STEREO_IO_THREAD_BYTES = 2 * 1024 * 1024
STEREO_REQUESTED_STACK_BYTES = 512 * 1024
QUALITY_PNG_ENCODER_MAX_WORKERS = 4
QUALITY_PNG_ENCODER_STACK_RESERVE_BYTES = 2 * 1024 * 1024
# Standard executor stacks need their own reserve plus thread/TLS runtime slack.
QUALITY_PNG_ENCODER_THREAD_BYTES = 4 * 1024 * 1024
STEREO_CONTROL_MANIFEST_STREAM_BYTES = 1024 * 1024
STEREO_CONTROL_JSON_STATE_BYTES = 256 * 1024
STEREO_CONTROL_DIAGNOSTICS_STATE_BYTES = 512 * 1024
STEREO_CONTROL_PROGRESS_STATE_BYTES = 128 * 1024
STEREO_CONTROL_PROVIDER_STATE_BYTES = 256 * 1024
STEREO_CONTROL_ALLOCATOR_SLACK_BYTES = 512 * 1024

_HEX_32 = re.compile(r"^[0-9a-f]{32}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_FRAME_STEM = re.compile(r"^frame_(?:[0-9]{6}|[1-9][0-9]{6,})$")
_STEREO_THREAD_CREATION_LOCK = threading.Lock()


class StereoControlStateError(StereoControlError):
    """A packed action or replay state is invalid."""


class StereoPipelineCapacityError(StereoControlBudgetError):
    """A stereo lifecycle cannot fit the fixed host-memory budget."""


class QualityPipelineCapacityError(StereoPipelineCapacityError):
    """The pre-mutation Quality lifecycle cannot fit the host-memory budget."""


class StereoProviderChangedError(StereoControlError):
    """A pre-mutation replay changed and must restart from read-only audit."""


class StereoStageInputChangedError(StereoControlError):
    """A post-mutation replay changed and cannot restart in the current call."""


class StereoInputIdentityMismatch(StereoControlError):
    """Decoded source bytes no longer match the frozen provider identity."""


class StereoThreadBudgetError(StereoControlError):
    """Thread creation, attestation, restoration, or teardown failed."""

    def __init__(self, stage: str, detail: str, started_threads: int, joined_threads: int) -> None:
        self.stage = stage
        self.started_threads = int(started_threads)
        self.joined_threads = int(joined_threads)
        super().__init__(
            f"stereo thread {stage} failed: {detail}; "
            f"started={started_threads}, joined={joined_threads}"
        )


class StereoAction(IntEnum):
    REUSE = 0
    RENDER = 1
    MIGRATE = 2
    INVALID = 3


class PackedStereoActions:
    """Exactly two action bits per source position."""

    __slots__ = ("frame_count", "_storage")

    def __init__(self, frame_count: int) -> None:
        if isinstance(frame_count, bool) or not isinstance(frame_count, int):
            raise TypeError("stereo frame count must be an integer")
        if frame_count <= 0:
            raise StereoControlBudgetError("stereo frame count must be positive")
        if frame_count > STEREO_CONTROL_FRAME_CAP:
            raise StereoControlBudgetError(
                f"stereo frame count {frame_count} exceeds {STEREO_CONTROL_FRAME_CAP}"
            )
        self.frame_count = frame_count
        self._storage = bytearray((frame_count + 3) // 4)
        if len(self._storage) > STEREO_CONTROL_ACTION_BYTES:
            raise StereoControlBudgetError("packed stereo action allocation exceeds one MiB")

    @property
    def storage(self) -> bytearray:
        return self._storage

    @property
    def storage_nbytes(self) -> int:
        return len(self._storage)

    def _position(self, index: int) -> tuple[int, int]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("stereo action index must be an integer")
        if index < 0 or index >= self.frame_count:
            raise IndexError("stereo action index is outside the source sequence")
        return index >> 2, (index & 3) << 1

    def set(self, index: int, action: StereoAction) -> None:
        try:
            normalized = StereoAction(action)
        except (TypeError, ValueError) as error:
            raise StereoControlStateError("unknown stereo action") from error
        if normalized is StereoAction.INVALID:
            raise StereoControlStateError("invalid packed stereo action cannot be emitted")
        byte_index, shift = self._position(index)
        mask = 0b11 << shift
        self._storage[byte_index] = (self._storage[byte_index] & ~mask) | (int(normalized) << shift)

    def get(self, index: int) -> StereoAction:
        byte_index, shift = self._position(index)
        action = StereoAction((self._storage[byte_index] >> shift) & 0b11)
        if action is StereoAction.INVALID:
            raise StereoControlStateError("invalid packed stereo action encountered on replay")
        return action

    def __iter__(self) -> Iterator[StereoAction]:
        for index in range(self.frame_count):
            yield self.get(index)


@dataclass(frozen=True)
class StereoProviderGeneration:
    audit_generation_id: str
    input_manifest_fingerprint: str
    frame_count: int

    def __post_init__(self) -> None:
        if not _HEX_32.fullmatch(self.audit_generation_id):
            raise ValueError("audit generation ID must be 32 lowercase hexadecimal digits")
        if not _HEX_64.fullmatch(self.input_manifest_fingerprint):
            raise ValueError("input manifest fingerprint must be 64 lowercase hexadecimal digits")
        if (
            isinstance(self.frame_count, bool)
            or not isinstance(self.frame_count, int)
            or self.frame_count <= 0
            or self.frame_count > STEREO_CONTROL_FRAME_CAP
        ):
            raise StereoControlBudgetError("provider frame count is outside the stereo control cap")

    @classmethod
    def freeze(cls, canonical_manifest_bytes: bytes, frame_count: int) -> StereoProviderGeneration:
        return cls(
            audit_generation_id=secrets.token_hex(16),
            input_manifest_fingerprint=hashlib.sha256(canonical_manifest_bytes).hexdigest(),
            frame_count=frame_count,
        )


@dataclass(frozen=True)
class StereoProviderItem:
    audit_generation_id: str
    input_manifest_fingerprint: str
    source_index: int
    frame_name: str
    guide_path: Path
    geometry_path: Path
    left_path: Path
    right_path: Path
    guide_sha256: str
    geometry_sha256: str
    guide_byte_count: int
    geometry_byte_count: int
    guide_shape: tuple[int, int]
    geometry_shape: tuple[int, int]
    frame_json_bound: int

    @property
    def index(self) -> int:
        return self.source_index

    @property
    def frame_path(self) -> Path:
        return self.guide_path

    @property
    def depth_path(self) -> Path:
        return self.geometry_path

    @property
    def stat_path(self) -> None:
        return None


class _ReplayFactory(Protocol):
    def __call__(self) -> Iterable[StereoProviderItem]: ...


class ReplayableStereoProvider:
    """One generation plus a factory which reopens its source-order manifest."""

    __slots__ = ("generation", "_replay_factory")

    def __init__(
        self,
        generation: StereoProviderGeneration,
        replay_factory: _ReplayFactory,
    ) -> None:
        if not isinstance(generation, StereoProviderGeneration):
            raise TypeError("stereo provider requires a frozen generation")
        if not callable(replay_factory):
            raise TypeError("stereo provider replay factory must be callable")
        self.generation = generation
        self._replay_factory = replay_factory

    def replay(self) -> Iterator[StereoProviderItem]:
        return iter(self._replay_factory())


@dataclass(frozen=True)
class StereoControlAuditPlan:
    generation: StereoProviderGeneration
    actions: PackedStereoActions
    render_count: int
    migrate_count: int
    item_stream_sha256: str
    maximum_native_pixels: int
    maximum_compressed_payload_bytes: int
    maximum_source_native_pixels: int
    maximum_source_compressed_payload_bytes: int
    maximum_frame_json_bound: int
    frame_name_json_array_bytes: int
    maximum_fast_relative_item_bytes: int
    maximum_fast_metric_item_bytes: int
    has_consecutive_render_input: bool
    control_residency: "StereoControlResidencyEvidence"

    @property
    def persistent_control_bytes(self) -> int:
        return self.control_residency.total_bytes


@dataclass(frozen=True)
class StereoControlResidencyEvidence:
    action_bytes: int
    manifest_stream_bytes: int
    json_state_bytes: int
    diagnostics_state_bytes: int
    progress_state_bytes: int
    provider_state_bytes: int
    allocator_slack_bytes: int
    scalar_object_bytes: int

    @property
    def total_bytes(self) -> int:
        return sum(
            (
                self.action_bytes,
                self.manifest_stream_bytes,
                self.json_state_bytes,
                self.diagnostics_state_bytes,
                self.progress_state_bytes,
                self.provider_state_bytes,
                self.allocator_slack_bytes,
                self.scalar_object_bytes,
            )
        )


def _control_residency_evidence(
    generation: StereoProviderGeneration,
    actions: PackedStereoActions,
    item_stream_sha256: str,
) -> StereoControlResidencyEvidence:
    scalar_bytes = (
        sys.getsizeof(generation)
        + sys.getsizeof(actions)
        + sys.getsizeof(item_stream_sha256)
        + 64 * 1024
    )
    return StereoControlResidencyEvidence(
        action_bytes=sys.getsizeof(actions.storage),
        manifest_stream_bytes=STEREO_CONTROL_MANIFEST_STREAM_BYTES,
        json_state_bytes=STEREO_CONTROL_JSON_STATE_BYTES,
        diagnostics_state_bytes=STEREO_CONTROL_DIAGNOSTICS_STATE_BYTES,
        progress_state_bytes=STEREO_CONTROL_PROGRESS_STATE_BYTES,
        provider_state_bytes=STEREO_CONTROL_PROVIDER_STATE_BYTES,
        allocator_slack_bytes=STEREO_CONTROL_ALLOCATOR_SLACK_BYTES,
        scalar_object_bytes=scalar_bytes,
    )


def canonical_frame_stem(index: int) -> str:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("canonical frame index must be a nonnegative integer")
    decimal = str(index)
    return "frame_" + decimal.zfill(max(6, len(decimal)))


def _update_length_prefixed(digest: hashlib._Hash, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(struct.pack("<Q", len(encoded)))
    digest.update(encoded)


def _validate_provider_item(
    item: StereoProviderItem,
    generation: StereoProviderGeneration,
    expected_index: int,
) -> None:
    if not isinstance(item, StereoProviderItem):
        raise StereoControlStateError("provider replay yielded a non-item object")
    if item.audit_generation_id != generation.audit_generation_id:
        raise StereoControlStateError("provider audit generation changed")
    if item.input_manifest_fingerprint != generation.input_manifest_fingerprint:
        raise StereoControlStateError("provider input manifest fingerprint changed")
    if item.source_index != expected_index:
        raise StereoControlStateError("provider source index is noncontiguous")
    stem = item.frame_name
    if not _CANONICAL_FRAME_STEM.fullmatch(stem):
        raise StereoControlStateError("provider frame name is noncanonical")
    if not _HEX_64.fullmatch(item.guide_sha256) or not _HEX_64.fullmatch(item.geometry_sha256):
        raise StereoControlStateError("provider content hash is noncanonical")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (
            item.guide_byte_count,
            item.geometry_byte_count,
            item.frame_json_bound,
        )
    ):
        raise StereoControlStateError("provider byte bounds are invalid")
    for shape in (item.guide_shape, item.geometry_shape):
        if (
            not isinstance(shape, tuple)
            or len(shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in shape
            )
        ):
            raise StereoControlStateError("provider image shape is invalid")
    expected_guide = f"{stem}.png"
    if item.guide_path.name != expected_guide:
        raise StereoControlStateError("provider guide path does not match the canonical stem")
    if item.geometry_path.stem != stem or item.geometry_path.suffix not in {".png", ".npz"}:
        raise StereoControlStateError("provider geometry path does not match the canonical stem")
    if item.left_path.name != expected_guide or item.right_path.name != expected_guide:
        raise StereoControlStateError("provider target path does not match the canonical stem")


def _update_item_digest(digest: hashlib._Hash, item: StereoProviderItem) -> None:
    digest.update(struct.pack("<Q", item.source_index))
    for value in (
        item.audit_generation_id,
        item.input_manifest_fingerprint,
        item.frame_name,
        item.guide_path.as_posix(),
        item.geometry_path.as_posix(),
        item.left_path.as_posix(),
        item.right_path.as_posix(),
        item.guide_sha256,
        item.geometry_sha256,
    ):
        _update_length_prefixed(digest, value)
    digest.update(
        struct.pack(
            "<QQQQQ",
            item.guide_byte_count,
            item.geometry_byte_count,
            item.guide_shape[0],
            item.guide_shape[1],
            item.frame_json_bound,
        )
    )
    digest.update(struct.pack("<QQ", item.geometry_shape[0], item.geometry_shape[1]))


def audit_stereo_provider(
    provider: ReplayableStereoProvider,
    classify: Callable[[StereoProviderItem], StereoAction],
) -> StereoControlAuditPlan:
    """Build only the packed action bytes and scalar replay evidence."""

    generation = provider.generation
    actions = PackedStereoActions(generation.frame_count)
    digest = hashlib.sha256()
    render_count = 0
    migrate_count = 0
    maximum_native_pixels = 0
    maximum_compressed_payload_bytes = 0
    maximum_source_native_pixels = 0
    maximum_source_compressed_payload_bytes = 0
    maximum_frame_json_bound = 0
    frame_name_json_array_bytes = 2
    maximum_fast_relative_item_bytes = 0
    maximum_fast_metric_item_bytes = 0
    previous_render_input: tuple[object, ...] | None = None
    has_consecutive_render_input = False
    expected_index = 0
    for item in provider.replay():
        if expected_index >= generation.frame_count:
            raise StereoControlStateError("provider replay exceeds its frozen frame count")
        _validate_provider_item(item, generation, expected_index)
        _update_item_digest(digest, item)
        try:
            action = StereoAction(classify(item))
        except (TypeError, ValueError) as error:
            raise StereoControlStateError("classifier returned an unknown stereo action") from error
        actions.set(expected_index, action)
        render_count += int(action is StereoAction.RENDER)
        migrate_count += int(action is StereoAction.MIGRATE)
        maximum_frame_json_bound = max(maximum_frame_json_bound, item.frame_json_bound)
        source_native_pixels = item.geometry_shape[0] * item.geometry_shape[1]
        source_compressed_bytes = item.guide_byte_count + item.geometry_byte_count
        maximum_source_native_pixels = max(
            maximum_source_native_pixels,
            source_native_pixels,
        )
        maximum_source_compressed_payload_bytes = max(
            maximum_source_compressed_payload_bytes,
            source_compressed_bytes,
        )
        frame_name_json_array_bytes += len(item.frame_name) + 2 + int(expected_index > 0)
        if action is StereoAction.RENDER:
            render_input = (
                item.guide_sha256,
                item.guide_byte_count,
                item.guide_shape,
                item.geometry_sha256,
                item.geometry_byte_count,
                item.geometry_shape,
            )
            has_consecutive_render_input |= previous_render_input == render_input
            previous_render_input = render_input
            output_pixels = item.guide_shape[0] * item.guide_shape[1]
            native_pixels = item.geometry_shape[0] * item.geometry_shape[1]
            compressed = item.guide_byte_count + item.geometry_byte_count
            maximum_native_pixels = max(maximum_native_pixels, native_pixels)
            maximum_compressed_payload_bytes = max(
                maximum_compressed_payload_bytes,
                compressed,
            )
            maximum_fast_relative_item_bytes = max(
                maximum_fast_relative_item_bytes,
                3 * output_pixels + 2 * native_pixels + compressed,
                7 * output_pixels + 6 * native_pixels,
                25 * output_pixels,
                23 * output_pixels + item.frame_json_bound,
            )
            maximum_fast_metric_item_bytes = max(
                maximum_fast_metric_item_bytes,
                3 * output_pixels + 5 * native_pixels + compressed,
                11 * output_pixels + 9 * native_pixels,
                25 * output_pixels,
                23 * output_pixels + item.frame_json_bound,
            )
        expected_index += 1
    if expected_index != generation.frame_count:
        raise StereoControlStateError("provider replay ended before its frozen frame count")
    item_stream_sha256 = digest.hexdigest()
    residency = _control_residency_evidence(generation, actions, item_stream_sha256)
    plan = StereoControlAuditPlan(
        generation=generation,
        actions=actions,
        render_count=render_count,
        migrate_count=migrate_count,
        item_stream_sha256=item_stream_sha256,
        maximum_native_pixels=maximum_native_pixels,
        maximum_compressed_payload_bytes=maximum_compressed_payload_bytes,
        maximum_source_native_pixels=maximum_source_native_pixels,
        maximum_source_compressed_payload_bytes=maximum_source_compressed_payload_bytes,
        maximum_frame_json_bound=maximum_frame_json_bound,
        frame_name_json_array_bytes=frame_name_json_array_bytes,
        maximum_fast_relative_item_bytes=maximum_fast_relative_item_bytes,
        maximum_fast_metric_item_bytes=maximum_fast_metric_item_bytes,
        has_consecutive_render_input=has_consecutive_render_input,
        control_residency=residency,
    )
    if plan.persistent_control_bytes > STEREO_CONTROL_OVERHEAD:
        raise StereoControlBudgetError("persistent stereo control plane exceeds four MiB")
    return plan


def _replay_digest(
    provider: ReplayableStereoProvider,
    generation: StereoProviderGeneration,
) -> str:
    if provider.generation != generation:
        raise StereoControlStateError("provider generation or count changed")
    digest = hashlib.sha256()
    expected_index = 0
    for item in provider.replay():
        if expected_index >= generation.frame_count:
            raise StereoControlStateError("provider replay exceeds its frozen frame count")
        _validate_provider_item(item, generation, expected_index)
        _update_item_digest(digest, item)
        expected_index += 1
    if expected_index != generation.frame_count:
        raise StereoControlStateError("provider replay ended before its frozen frame count")
    return digest.hexdigest()


def _raise_provider_changed(detail: str, mutation_started: bool) -> None:
    error_type = StereoStageInputChangedError if mutation_started else StereoProviderChangedError
    raise error_type(detail)


def validate_stereo_provider_replay(
    provider: ReplayableStereoProvider,
    plan: StereoControlAuditPlan,
    *,
    mutation_started: bool,
) -> None:
    try:
        replay_hash = _replay_digest(provider, plan.generation)
    except (StereoControlError, TypeError, ValueError, OSError) as error:
        _raise_provider_changed(str(error), mutation_started)
    if replay_hash != plan.item_stream_sha256:
        _raise_provider_changed("provider item identity changed", mutation_started)


def validate_stereo_action_replay(
    provider: ReplayableStereoProvider,
    plan: StereoControlAuditPlan,
    classify: Callable[[StereoProviderItem], StereoAction],
) -> None:
    """Reclassify retained outputs without allocating a second action vector."""

    if provider.generation != plan.generation:
        raise StereoProviderChangedError("provider generation changed")
    digest = hashlib.sha256()
    expected_index = 0
    render_count = 0
    migrate_count = 0
    try:
        replay = provider.replay()
        for item in replay:
            if expected_index >= plan.generation.frame_count:
                raise StereoControlStateError("provider replay exceeds its frozen frame count")
            _validate_provider_item(item, plan.generation, expected_index)
            _update_item_digest(digest, item)
            try:
                action = StereoAction(classify(item))
            except (TypeError, ValueError) as error:
                raise StereoControlStateError(
                    "classifier returned an unknown stereo action"
                ) from error
            if action is not plan.actions.get(expected_index):
                raise StereoProviderChangedError(
                    "stereo retained-output action changed before mutation"
                )
            render_count += int(action is StereoAction.RENDER)
            migrate_count += int(action is StereoAction.MIGRATE)
            expected_index += 1
    except StereoProviderChangedError:
        raise
    except (StereoControlError, OSError, TypeError, ValueError) as error:
        raise StereoProviderChangedError(str(error)) from error
    if expected_index != plan.generation.frame_count:
        raise StereoProviderChangedError("provider replay ended before its frozen frame count")
    if digest.hexdigest() != plan.item_stream_sha256:
        raise StereoProviderChangedError("provider item identity changed")
    if render_count != plan.render_count or migrate_count != plan.migrate_count:
        raise StereoProviderChangedError("stereo retained-output action counts changed")


def iter_stereo_action(
    provider: ReplayableStereoProvider,
    plan: StereoControlAuditPlan,
    selected_action: StereoAction,
    *,
    mutation_started: bool = True,
) -> Iterator[StereoProviderItem]:
    """Replay lazily and yield only one action without constructing a resident set."""

    selected_action = StereoAction(selected_action)
    if selected_action is StereoAction.INVALID:
        raise StereoControlStateError("invalid action cannot be selected")
    if provider.generation != plan.generation:
        _raise_provider_changed("provider generation changed", mutation_started)
    digest = hashlib.sha256()
    expected_index = 0
    try:
        replay = provider.replay()
    except (StereoControlError, OSError, TypeError, ValueError) as error:
        _raise_provider_changed(str(error), mutation_started)
    while True:
        try:
            item = next(replay)
        except StopIteration:
            break
        except (StereoControlError, OSError, TypeError, ValueError) as error:
            _raise_provider_changed(str(error), mutation_started)
        if expected_index >= plan.generation.frame_count:
            _raise_provider_changed(
                "provider replay exceeds its frozen frame count",
                mutation_started,
            )
        try:
            _validate_provider_item(item, plan.generation, expected_index)
            action = plan.actions.get(expected_index)
        except StereoControlError as error:
            _raise_provider_changed(str(error), mutation_started)
        _update_item_digest(digest, item)
        expected_index += 1
        if action is selected_action:
            yield item
            del item
    if expected_index != plan.generation.frame_count:
        _raise_provider_changed(
            "provider replay ended before its frozen frame count",
            mutation_started,
        )
    if digest.hexdigest() != plan.item_stream_sha256:
        _raise_provider_changed("provider item identity changed", mutation_started)


@dataclass(frozen=True)
class FastPipelineCapacity:
    capacity: int
    effective_workers: int
    thread_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class QualityPipelineCapacity:
    capacity: int
    effective_workers: int
    writer_workers: int
    png_encoder_workers: int
    thread_bytes: int
    total_bytes: int


def quality_pipeline_capacity(pending_render_count: int) -> tuple[int, int, int]:
    if isinstance(pending_render_count, bool) or not isinstance(pending_render_count, int):
        raise TypeError("pending render count must be an integer")
    if pending_render_count < 0:
        raise ValueError("pending render count must be nonnegative")
    if pending_render_count == 0:
        return 0, 0, 0
    return 1, 1, 3 * STEREO_IO_THREAD_BYTES


def select_quality_pipeline_capacity(  # noqa: C901
    *,
    configured_workers: int,
    pending_render_count: int,
    quality_active_scratch: int,
    quality_slot_bytes: int,
    retained_bytes: int = 0,
    png_payloads_per_item: int = 0,
) -> QualityPipelineCapacity:
    """Overlap Quality decode, render, and write under the fixed host budget."""

    if isinstance(configured_workers, bool) or not isinstance(configured_workers, int):
        raise TypeError("configured stereo workers must be an integer")
    if configured_workers < 1 or configured_workers > 16:
        raise ValueError("configured stereo workers must be in 1..16")
    if isinstance(pending_render_count, bool) or not isinstance(pending_render_count, int):
        raise TypeError("pending render count must be an integer")
    if pending_render_count < 0:
        raise ValueError("pending render count must be nonnegative")
    if isinstance(png_payloads_per_item, bool) or not isinstance(png_payloads_per_item, int):
        raise TypeError("Quality PNG payload count must be an integer")
    if png_payloads_per_item < 0 or png_payloads_per_item > 6:
        raise ValueError("Quality PNG payload count must be in 0..6")
    if quality_active_scratch < 0 or quality_slot_bytes <= 0 or retained_bytes < 0:
        raise ValueError("Quality scratch and slot byte counts are invalid")
    if pending_render_count == 0:
        return QualityPipelineCapacity(
            capacity=0,
            effective_workers=0,
            writer_workers=0,
            png_encoder_workers=0,
            thread_bytes=0,
            total_bytes=quality_active_scratch + retained_bytes,
        )

    selected: QualityPipelineCapacity | None = None
    candidate_max = min(4, configured_workers, pending_render_count)
    for capacity in range(1, candidate_max + 1):
        effective_workers = min(configured_workers, capacity)
        writer_workers = min(3, effective_workers)
        pipeline_thread_count = 1 + effective_workers + writer_workers
        base_thread_bytes = pipeline_thread_count * STEREO_IO_THREAD_BYTES
        base_total = (
            quality_active_scratch
            + retained_bytes
            + (capacity - 1) * quality_slot_bytes
            + base_thread_bytes
        )
        available_png_threads = max(
            0,
            (STEREO_HOST_BUDGET - base_total) // QUALITY_PNG_ENCODER_THREAD_BYTES,
        )
        png_encoder_workers = min(
            QUALITY_PNG_ENCODER_MAX_WORKERS,
            configured_workers,
            pending_render_count * png_payloads_per_item,
            capacity * png_payloads_per_item,
            available_png_threads,
        )
        if png_encoder_workers < 2:
            png_encoder_workers = 0
        png_thread_bytes = png_encoder_workers * QUALITY_PNG_ENCODER_THREAD_BYTES
        thread_bytes = base_thread_bytes + png_thread_bytes
        total = base_total + png_thread_bytes
        if total <= STEREO_HOST_BUDGET:
            selected = QualityPipelineCapacity(
                capacity=capacity,
                effective_workers=effective_workers,
                writer_workers=writer_workers,
                png_encoder_workers=png_encoder_workers,
                thread_bytes=thread_bytes,
                total_bytes=total,
            )
    if selected is None:
        raise QualityPipelineCapacityError("no Quality lifecycle capacity fits the host budget")
    return selected


def select_fast_pipeline_capacity(
    *,
    configured_workers: int,
    pending_render_count: int,
    fast_active_scratch: int,
    fast_slot_bytes: int,
    retained_bytes: int = 0,
) -> FastPipelineCapacity:
    """Choose the unique largest eligible capacity from the frozen formula."""

    if isinstance(configured_workers, bool) or not isinstance(configured_workers, int):
        raise TypeError("configured stereo workers must be an integer")
    if configured_workers < 1 or configured_workers > 16:
        raise ValueError("configured stereo workers must be in 1..16")
    if isinstance(pending_render_count, bool) or not isinstance(pending_render_count, int):
        raise TypeError("pending render count must be an integer")
    if pending_render_count < 0:
        raise ValueError("pending render count must be nonnegative")
    if fast_active_scratch < 0 or fast_slot_bytes <= 0 or retained_bytes < 0:
        raise ValueError("Fast scratch, slot, and retained byte counts are invalid")
    if pending_render_count == 0:
        return FastPipelineCapacity(0, 0, 0, fast_active_scratch + retained_bytes)
    candidate_max = min(2 * configured_workers, pending_render_count)
    selected: FastPipelineCapacity | None = None
    for capacity in range(1, candidate_max + 1):
        effective_workers = min(configured_workers, capacity)
        thread_bytes = (1 + 2 * effective_workers) * STEREO_IO_THREAD_BYTES
        total = fast_active_scratch + retained_bytes + thread_bytes + capacity * fast_slot_bytes
        if total <= STEREO_HOST_BUDGET:
            selected = FastPipelineCapacity(capacity, effective_workers, thread_bytes, total)
    if selected is None:
        raise StereoPipelineCapacityError("no Fast lifecycle capacity fits the host budget")
    return selected


class StereoPipelineQueues:
    """The four fixed handoff queues, all with one selected capacity."""

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("stereo pipeline queue capacity must be positive")
        self.decode_requests: queue.Queue[object] = queue.Queue(maxsize=capacity)
        self.decoded_frames: queue.Queue[object] = queue.Queue(maxsize=capacity)
        self.write_requests: queue.Queue[object] = queue.Queue(maxsize=capacity)
        self.write_completions: queue.Queue[object] = queue.Queue(maxsize=capacity)

    @property
    def capacities(self) -> tuple[int, int, int, int]:
        return (
            self.decode_requests.maxsize,
            self.decoded_frames.maxsize,
            self.write_requests.maxsize,
            self.write_completions.maxsize,
        )


@dataclass(frozen=True)
class StackReservationEvidence:
    adapter: str
    reserve_bytes: int
    committed_bytes: int
    granularity_bytes: int


@dataclass(frozen=True)
class StereoThreadCreationStats:
    thread_names: tuple[str, ...]
    requested_stack_bytes: int
    previous_stack_bytes: int
    restored_stack_bytes: int
    stack_evidence: tuple[StackReservationEvidence, ...]


class _ThreadLike(Protocol):
    name: str
    ident: int | None

    def start(self) -> None: ...

    def join(self, timeout: float | None = None) -> None: ...

    def is_alive(self) -> bool: ...


class _WindowsReservedThread:
    """Minimal joinable CRT thread with reserve semantics, not commit semantics."""

    def __init__(
        self,
        *,
        target: Callable[..., None],
        args: tuple[object, ...],
        name: str,
    ) -> None:
        self.name = name
        self.ident: int | None = None
        self._target: Callable[..., None] | None = target
        self._args = args
        self._handle: int | None = None
        self._callback: object | None = None
        self._closed = False
        self.error: BaseException | None = None

    def start(self) -> None:
        if self._handle is not None or self._closed:
            raise RuntimeError("reserved thread can only be started once")
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_uint, ctypes.c_void_p)

        @callback_type
        def entry(_argument: int) -> int:
            try:
                target = self._target
                if target is None:
                    raise RuntimeError("reserved thread target was released before start")
                target(*self._args)
            except BaseException as error:
                self.error = error
            return 0

        runtime = ctypes.CDLL("ucrtbase.dll", use_errno=True)
        begin = runtime._beginthreadex
        begin.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            callback_type,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_uint),
        ]
        begin.restype = ctypes.c_size_t
        thread_id = ctypes.c_uint()
        handle = begin(
            None,
            STEREO_REQUESTED_STACK_BYTES,
            entry,
            None,
            0x00010000,
            ctypes.byref(thread_id),
        )
        if handle == 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, "_beginthreadex failed")
        self._callback = entry
        self._handle = int(handle)
        self.ident = int(thread_id.value)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        set_description = getattr(kernel32, "SetThreadDescription", None)
        if set_description is not None:
            set_description.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
            set_description.restype = ctypes.c_long
            set_description(ctypes.c_void_p(self._handle), self.name)

    def is_alive(self) -> bool:
        if self._handle is None or self._closed:
            return False
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        result = kernel32.WaitForSingleObject(ctypes.c_void_p(self._handle), 0)
        if result == 0x00000102:
            return True
        if result == 0xFFFFFFFF:
            raise OSError(ctypes.get_last_error(), "WaitForSingleObject failed")
        return False

    def join(self, timeout: float | None = None) -> None:
        if self._handle is None:
            raise RuntimeError("cannot join an unstarted reserved thread")
        if self._closed:
            return
        if timeout is None:
            milliseconds = 0xFFFFFFFF
        else:
            milliseconds = max(0, min(0xFFFFFFFE, int(timeout * 1000)))
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        result = kernel32.WaitForSingleObject(ctypes.c_void_p(self._handle), milliseconds)
        if result == 0x00000102:
            return
        if result != 0:
            raise OSError(ctypes.get_last_error(), "WaitForSingleObject failed")
        if not kernel32.CloseHandle(ctypes.c_void_p(self._handle)):
            raise OSError(ctypes.get_last_error(), "CloseHandle failed")
        self._closed = True
        self._callback = None
        self._target = None
        self._args = ()


def _new_stereo_thread(
    *,
    target: Callable[..., None],
    args: tuple[object, ...],
    name: str,
) -> _ThreadLike:
    if sys.platform == "win32":
        return _WindowsReservedThread(target=target, args=args, name=name)
    return threading.Thread(target=target, args=args, name=name, daemon=True)


class ParkedStereoWorkers:
    """A fully attested worker set parked before any work or mutation."""

    def __init__(
        self,
        threads: list[_ThreadLike],
        start_gate: threading.Event,
        cancel_event: threading.Event,
        stats: StereoThreadCreationStats,
    ) -> None:
        self._threads = threads
        self._start_gate = start_gate
        self._cancel_event = cancel_event
        self.stats = stats

    @property
    def live_thread_count(self) -> int:
        return sum(int(thread.is_alive()) for thread in self._threads)

    def release(self) -> None:
        self._start_gate.set()

    def join(self) -> None:
        for thread in self._threads:
            thread.join()
        errors = [getattr(thread, "error", None) for thread in self._threads]
        error = next((item for item in errors if item is not None), None)
        if error is not None:
            raise StereoThreadBudgetError(
                "worker",
                str(error),
                len(self._threads),
                len(self._threads),
            ) from error

    def release_and_join(self) -> None:
        self.release()
        self.join()

    def cancel_and_join(self) -> None:
        self._cancel_event.set()
        self._start_gate.set()
        for thread in self._threads:
            thread.join()


def _empty_worker_group() -> ParkedStereoWorkers:
    return ParkedStereoWorkers(
        [],
        threading.Event(),
        threading.Event(),
        StereoThreadCreationStats((), 0, 0, 0, ()),
    )


def _thread_names(
    effective_workers: int,
    writer_workers: int,
) -> list[tuple[str, str, int]]:
    names = [("stereo-feeder-0", "feeder", 0)]
    names.extend(
        (f"stereo-decoder-{index}", "decoder", index) for index in range(effective_workers)
    )
    names.extend((f"stereo-writer-{index}", "writer", index) for index in range(writer_workers))
    return names


def _attest_stack(evidence: StackReservationEvidence) -> None:
    if not isinstance(evidence, StackReservationEvidence):
        raise StereoThreadBudgetError("attestation", "adapter returned no evidence", 0, 0)
    if evidence.granularity_bytes <= 0:
        raise StereoThreadBudgetError("attestation", "invalid platform granularity", 0, 0)
    admitted = (
        (STEREO_REQUESTED_STACK_BYTES + evidence.granularity_bytes - 1)
        // evidence.granularity_bytes
    ) * evidence.granularity_bytes
    if evidence.reserve_bytes <= 0 or evidence.reserve_bytes > admitted:
        raise StereoThreadBudgetError(
            "attestation",
            f"reserved stack {evidence.reserve_bytes} exceeds {admitted}",
            0,
            0,
        )
    if evidence.committed_bytes < 0 or evidence.committed_bytes > evidence.reserve_bytes:
        raise StereoThreadBudgetError("attestation", "invalid committed stack bytes", 0, 0)


def _attest_executor_stack(evidence: StackReservationEvidence) -> None:
    if not isinstance(evidence, StackReservationEvidence):
        raise StereoThreadBudgetError("attestation", "adapter returned no evidence", 0, 0)
    if evidence.granularity_bytes <= 0:
        raise StereoThreadBudgetError("attestation", "invalid platform granularity", 0, 0)
    if (
        evidence.reserve_bytes <= 0
        or evidence.reserve_bytes > QUALITY_PNG_ENCODER_STACK_RESERVE_BYTES
    ):
        raise StereoThreadBudgetError(
            "attestation",
            f"executor stack reserve {evidence.reserve_bytes} exceeds "
            f"the stack limit {QUALITY_PNG_ENCODER_STACK_RESERVE_BYTES}",
            0,
            0,
        )
    if evidence.committed_bytes < 0 or evidence.committed_bytes > evidence.reserve_bytes:
        raise StereoThreadBudgetError("attestation", "invalid committed stack bytes", 0, 0)


def create_attested_stereo_executor(  # noqa: C901
    *,
    worker_count: int,
    thread_name_prefix: str,
    stack_adapter: Callable[[], StackReservationEvidence] | None = None,
    stack_size_api: Callable[..., int] = threading.stack_size,
    attestation_timeout_seconds: float = 10.0,
) -> tuple[ThreadPoolExecutor, StereoThreadCreationStats]:
    """Prime a fixed executor and prove every worker fits its charged stack budget."""

    if isinstance(worker_count, bool) or not isinstance(worker_count, int):
        raise TypeError("executor worker count must be an integer")
    if worker_count < 1 or worker_count > 16:
        raise ValueError("executor worker count must be in 1..16")
    if not isinstance(thread_name_prefix, str) or not thread_name_prefix:
        raise ValueError("executor thread name prefix must be nonempty")
    if stack_adapter is None:
        stack_adapter = attest_current_thread_stack

    release = threading.Event()
    reports: queue.Queue[tuple[str, StackReservationEvidence | BaseException]] = queue.Queue(
        maxsize=worker_count
    )
    executor: ThreadPoolExecutor | None = None
    parked: list[Future[None]] = []
    previous_stack = 0
    restored_stack = 0
    failure_stage = "creation"
    failure: BaseException | None = None
    evidence_by_name: dict[str, StackReservationEvidence] = {}

    def attest_and_park() -> None:
        name = threading.current_thread().name
        try:
            report: StackReservationEvidence | BaseException = stack_adapter()
        except BaseException as error:
            report = error
        reports.put((name, report))
        release.wait()

    with _STEREO_THREAD_CREATION_LOCK:
        try:
            previous_stack = stack_size_api()
            failure_stage = "stack setting"
            stack_size_api(STEREO_REQUESTED_STACK_BYTES)
            failure_stage = "construction"
            executor = ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix=thread_name_prefix,
            )
            for _ in range(worker_count):
                parked.append(executor.submit(attest_and_park))
            failure_stage = "attestation"
            for _ in range(worker_count):
                try:
                    name, report = reports.get(timeout=attestation_timeout_seconds)
                except queue.Empty as error:
                    raise TimeoutError("executor worker attestation timed out") from error
                if isinstance(report, BaseException):
                    raise report
                _attest_executor_stack(report)
                if name in evidence_by_name:
                    raise RuntimeError("executor did not prime distinct worker threads")
                evidence_by_name[name] = report
        except BaseException as error:
            failure = error
        finally:
            try:
                failure_stage = "restoration" if failure is None else failure_stage
                stack_size_api(previous_stack)
                restored_stack = previous_stack
            except BaseException as restore_error:
                if failure is None:
                    failure = restore_error
                    failure_stage = "restoration"

    try:
        release.set()
        if failure is not None:
            raise StereoThreadBudgetError(
                failure_stage,
                str(failure),
                len(parked),
                len(parked),
            ) from failure

        assert executor is not None
        for future in parked:
            future.result()
        names = tuple(evidence_by_name)
        return executor, StereoThreadCreationStats(
            thread_names=names,
            requested_stack_bytes=STEREO_REQUESTED_STACK_BYTES,
            previous_stack_bytes=previous_stack,
            restored_stack_bytes=restored_stack,
            stack_evidence=tuple(evidence_by_name[name] for name in names),
        )
    except BaseException:
        release.set()
        for future in parked:
            future.cancel()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        raise


def create_parked_stereo_workers(  # noqa: C901
    *,
    pending_render_count: int,
    effective_workers: int,
    writer_workers: int | None = None,
    worker: Callable[[str, int], None],
    stack_adapter: Callable[[], StackReservationEvidence] | None = None,
    stack_size_api: Callable[..., int] = threading.stack_size,
    attestation_timeout_seconds: float = 10.0,
) -> ParkedStereoWorkers:
    """Create in fixed order, attest while parked, and restore global stack state."""

    if pending_render_count == 0:
        if effective_workers != 0 or writer_workers not in (None, 0):
            raise ValueError("P=0 requires zero effective workers")
        return _empty_worker_group()
    if pending_render_count < 0 or effective_workers < 1 or effective_workers > 16:
        raise ValueError("nonzero stereo work requires 1..16 effective workers")
    if writer_workers is None:
        writer_workers = effective_workers
    if (
        isinstance(writer_workers, bool)
        or not isinstance(writer_workers, int)
        or writer_workers < 1
        or writer_workers > effective_workers
    ):
        raise ValueError("writer workers must be in 1..effective workers")
    if stack_adapter is None:
        stack_adapter = attest_current_thread_stack
    names = _thread_names(effective_workers, writer_workers)
    start_gate = threading.Event()
    cancel_event = threading.Event()
    reports: queue.Queue[tuple[str, StackReservationEvidence | BaseException]] = queue.Queue(
        maxsize=len(names)
    )
    threads: list[_ThreadLike] = []
    started_threads: list[_ThreadLike] = []
    previous_stack = 0
    restored_stack = 0
    failure_stage = "creation"
    failure: BaseException | None = None
    evidence_by_name: dict[str, StackReservationEvidence] = {}

    def parked_target(name: str, role: str, index: int) -> None:
        try:
            evidence = stack_adapter()
            reports.put((name, evidence))
        except BaseException as error:
            reports.put((name, error))
        start_gate.wait()
        if not cancel_event.is_set():
            worker(role, index)

    with _STEREO_THREAD_CREATION_LOCK:
        try:
            previous_stack = stack_size_api()
            failure_stage = "stack setting"
            stack_size_api(STEREO_REQUESTED_STACK_BYTES)
            failure_stage = "construction"
            for name, role, index in names:
                thread = _new_stereo_thread(
                    target=parked_target,
                    args=(name, role, index),
                    name=name,
                )
                threads.append(thread)
                failure_stage = "start"
                thread.start()
                started_threads.append(thread)
                failure_stage = "construction"
            failure_stage = "attestation"
            for _ in names:
                try:
                    name, report = reports.get(timeout=attestation_timeout_seconds)
                except queue.Empty as error:
                    raise TimeoutError("parked worker attestation timed out") from error
                if isinstance(report, BaseException):
                    raise report
                _attest_stack(report)
                evidence_by_name[name] = report
        except BaseException as error:
            failure = error
        finally:
            try:
                failure_stage = "restoration" if failure is None else failure_stage
                stack_size_api(previous_stack)
                restored_stack = previous_stack
            except BaseException as restore_error:
                if failure is None:
                    failure = restore_error
                    failure_stage = "restoration"

    if failure is not None:
        cancel_event.set()
        start_gate.set()
        joined = 0
        for thread in started_threads:
            thread.join()
            joined += 1
        detail = str(failure)
        if isinstance(failure, StereoThreadBudgetError):
            detail = str(failure)
        raise StereoThreadBudgetError(
            failure_stage,
            detail,
            len(started_threads),
            joined,
        ) from failure

    stats = StereoThreadCreationStats(
        thread_names=tuple(name for name, _role, _index in names),
        requested_stack_bytes=STEREO_REQUESTED_STACK_BYTES,
        previous_stack_bytes=previous_stack,
        restored_stack_bytes=restored_stack,
        stack_evidence=tuple(evidence_by_name[name] for name, _role, _index in names),
    )
    return ParkedStereoWorkers(started_threads, start_gate, cancel_event, stats)


@dataclass(frozen=True)
class StreamingStereoPipelineStats:
    configured_capacity: int
    effective_workers: int
    actual_thread_count: int
    queue_capacities: tuple[int, int, int, int]
    requested_stack_bytes: int
    decoded: int
    rendered: int
    written: int
    maximum_live_items: int
    decode_render_overlap: bool
    render_write_overlap: bool
    decoder_writer_overlap: bool
    work_item_copies: int
    parked_committed_delta_bytes: int
    live_threads_after_join: int
    pipeline_wall_seconds: float
    decode_busy_seconds: float
    render_busy_seconds: float
    write_busy_seconds: float
    configured_workers: int = 0
    feeder_thread_count: int = 0
    decoder_thread_count: int = 0
    writer_thread_count: int = 0
    calculated_thread_bytes: int = 0
    previous_stack_bytes: int = 0
    restored_stack_bytes: int = 0
    stack_granularity_bytes: tuple[int, ...] = ()
    reported_stack_reserve_bytes: tuple[int, ...] = ()
    png_encoder_thread_count: int = 0


@dataclass(frozen=True)
class StereoControlExecutionStats:
    pipeline: StreamingStereoPipelineStats
    migrated: int
    thread_count_during_migration: int


class _StreamingTrace:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active_decode = 0
        self.active_render = 0
        self.active_write = 0
        self.live_items = 0
        self.maximum_live_items = 0
        self.decoded = 0
        self.rendered = 0
        self.written = 0
        self.decode_render_overlap = False
        self.render_write_overlap = False
        self.decoder_writer_overlap = False
        self.busy_seconds = {"decode": 0.0, "render": 0.0, "write": 0.0}

    def item_started(self) -> None:
        with self.lock:
            self.live_items += 1
            self.maximum_live_items = max(self.maximum_live_items, self.live_items)

    def item_finished(self) -> None:
        with self.lock:
            self.live_items -= 1

    def begin(self, stage: str) -> float:
        started = time.perf_counter()
        with self.lock:
            setattr(self, f"active_{stage}", getattr(self, f"active_{stage}") + 1)
            self.decode_render_overlap |= self.active_decode > 0 and self.active_render > 0
            self.render_write_overlap |= self.active_render > 0 and self.active_write > 0
            self.decoder_writer_overlap |= self.active_decode > 0 and self.active_write > 0
        return started

    def end(self, stage: str, started: float) -> None:
        elapsed = time.perf_counter() - started
        with self.lock:
            setattr(self, f"active_{stage}", getattr(self, f"active_{stage}") - 1)
            counter = {"decode": "decoded", "render": "rendered", "write": "written"}[stage]
            setattr(self, counter, getattr(self, counter) + 1)
            self.busy_seconds[stage] += elapsed


_PIPELINE_SENTINEL = object()
_PIPELINE_COMPLETION = object()


def run_streaming_stereo_pipeline(  # noqa: C901
    *,
    provider: ReplayableStereoProvider,
    plan: StereoControlAuditPlan,
    capacity: int,
    effective_workers: int,
    writer_workers: int | None = None,
    decode: Callable[[StereoProviderItem], object],
    render: Callable[[object], object],
    write: Callable[[object], object],
    mutation: Callable[[], None] | None = None,
    pre_mutation: Callable[[], None] | None = None,
    configured_workers: int | None = None,
    ordered_render: bool = False,
) -> StreamingStereoPipelineStats:
    """Drive a capacity-bounded synthetic pipeline without a resident work-item list."""

    pipeline_started = time.perf_counter()
    if not isinstance(ordered_render, bool):
        raise TypeError("ordered_render must be boolean")
    if plan.render_count == 0:
        if capacity != 0 or effective_workers != 0 or writer_workers not in (None, 0):
            raise ValueError("P=0 requires zero pipeline capacity and workers")
        if configured_workers is None:
            configured_workers = 0
        if configured_workers < 0 or configured_workers > 16:
            raise ValueError("configured workers must be between zero and sixteen")
        validate_stereo_provider_replay(provider, plan, mutation_started=False)
        if pre_mutation is not None:
            pre_mutation()
        if mutation is not None:
            try:
                mutation()
            except StereoInputIdentityMismatch as error:
                _raise_provider_changed(str(error), mutation_started=True)
        return StreamingStereoPipelineStats(
            configured_capacity=0,
            effective_workers=0,
            actual_thread_count=0,
            queue_capacities=(0, 0, 0, 0),
            requested_stack_bytes=0,
            decoded=0,
            rendered=0,
            written=0,
            maximum_live_items=0,
            decode_render_overlap=False,
            render_write_overlap=False,
            decoder_writer_overlap=False,
            work_item_copies=0,
            parked_committed_delta_bytes=0,
            live_threads_after_join=0,
            pipeline_wall_seconds=time.perf_counter() - pipeline_started,
            decode_busy_seconds=0.0,
            render_busy_seconds=0.0,
            write_busy_seconds=0.0,
            configured_workers=configured_workers,
        )
    if capacity <= 0 or effective_workers <= 0:
        raise ValueError("nonzero render work requires positive capacity and workers")
    if writer_workers is None:
        writer_workers = effective_workers
    if (
        isinstance(writer_workers, bool)
        or not isinstance(writer_workers, int)
        or writer_workers < 1
        or writer_workers > effective_workers
    ):
        raise ValueError("writer workers must be in 1..effective workers")
    if configured_workers is None:
        configured_workers = effective_workers
    if configured_workers < effective_workers or configured_workers > 16:
        raise ValueError("configured workers must contain the effective worker count")
    validate_stereo_provider_replay(provider, plan, mutation_started=False)
    before_committed = current_process_committed_bytes()
    queues = StereoPipelineQueues(capacity)
    permits = threading.BoundedSemaphore(capacity)
    trace = _StreamingTrace()
    pipeline_cancel = threading.Event()
    feeder_complete = threading.Event()
    worker_errors: queue.Queue[tuple[str, BaseException]] = queue.Queue(maxsize=1)
    mutation_started = False

    def validate_handoff(item: StereoProviderItem) -> None:
        _validate_provider_item(item, plan.generation, item.source_index)

    def put_or_cancel(target: queue.Queue[object], value: object) -> bool:
        while not pipeline_cancel.is_set():
            try:
                target.put(value, timeout=0.05)
            except queue.Full:
                continue
            return True
        return False

    def record_worker_error(role: str, error: BaseException) -> None:
        try:
            worker_errors.put_nowait((role, error))
        except queue.Full:
            pass
        pipeline_cancel.set()

    def raise_worker_error() -> None:
        try:
            role, error = worker_errors.get_nowait()
        except queue.Empty:
            raise StereoControlStateError("stereo pipeline cancelled without an error")
        worker_errors.task_done()
        if isinstance(error, StereoInputIdentityMismatch):
            _raise_provider_changed(str(error), mutation_started)
        if role == "feeder" and isinstance(
            error,
            (StereoControlError, OSError, TypeError, ValueError),
        ):
            _raise_provider_changed(str(error), mutation_started)
        raise error

    def coordinator_get(target: queue.Queue[object]) -> object:
        while True:
            if pipeline_cancel.is_set():
                raise_worker_error()
            try:
                return target.get(timeout=0.05)
            except queue.Empty:
                continue

    def release_item() -> None:
        trace.item_finished()
        permits.release()

    def drain_owned_queues() -> None:
        for target, owns in (
            (queues.decode_requests, lambda value: value is not _PIPELINE_SENTINEL),
            (
                queues.decoded_frames,
                lambda value: isinstance(value, tuple)
                and len(value) == 2
                and isinstance(value[0], StereoProviderItem),
            ),
            (
                queues.write_requests,
                lambda value: value is not _PIPELINE_SENTINEL
                and isinstance(value, tuple)
                and len(value) == 2
                and isinstance(value[0], StereoProviderItem),
            ),
            (queues.write_completions, lambda _value: False),
        ):
            while True:
                try:
                    message = target.get_nowait()
                except queue.Empty:
                    break
                try:
                    if owns(message):
                        release_item()
                finally:
                    target.task_done()
                    message = None

    def worker_body(role: str) -> None:
        if role == "feeder":
            try:
                actions = iter(iter_stereo_action(provider, plan, StereoAction.RENDER))
                while True:
                    while not pipeline_cancel.is_set() and not permits.acquire(timeout=0.05):
                        pass
                    if pipeline_cancel.is_set():
                        return
                    try:
                        item = next(actions)
                    except StopIteration:
                        permits.release()
                        feeder_complete.set()
                        break
                    except BaseException:
                        permits.release()
                        raise
                    trace.item_started()
                    if not put_or_cancel(queues.decode_requests, item):
                        release_item()
                        return
                    del item
            finally:
                for _ in range(effective_workers):
                    if not put_or_cancel(queues.decode_requests, _PIPELINE_SENTINEL):
                        break
            return
        if role == "decoder":
            while True:
                try:
                    queued_item = queues.decode_requests.get(timeout=0.05)
                except queue.Empty:
                    if pipeline_cancel.is_set():
                        return
                    continue
                transferred = False
                owns_item = queued_item is not _PIPELINE_SENTINEL
                decode_item: StereoProviderItem | None = None
                decoded_payload: object | None = None
                try:
                    if queued_item is _PIPELINE_SENTINEL:
                        return
                    if not isinstance(queued_item, StereoProviderItem):
                        raise StereoControlStateError("decoder received a non-item")
                    decode_item = queued_item
                    validate_handoff(decode_item)
                    stage_started = trace.begin("decode")
                    try:
                        decoded_payload = decode(decode_item)
                    finally:
                        trace.end("decode", stage_started)
                    transferred = put_or_cancel(
                        queues.decoded_frames,
                        (decode_item, decoded_payload),
                    )
                    if not transferred:
                        return
                finally:
                    queues.decode_requests.task_done()
                    queued_item = None
                    decode_item = None
                    decoded_payload = None
                    if owns_item and not transferred:
                        release_item()
        if role == "writer":
            while True:
                try:
                    message = queues.write_requests.get(timeout=0.05)
                except queue.Empty:
                    if pipeline_cancel.is_set():
                        return
                    continue
                owns_item = message is not _PIPELINE_SENTINEL
                write_item: StereoProviderItem | None = None
                write_payload: object | None = None
                try:
                    if message is _PIPELINE_SENTINEL:
                        return
                    if not isinstance(message, tuple) or len(message) != 2:
                        raise StereoControlStateError("writer received a malformed message")
                    write_item, write_payload = message
                    if not isinstance(write_item, StereoProviderItem):
                        raise StereoControlStateError("writer received a non-item")
                    validate_handoff(write_item)
                    stage_started = trace.begin("write")
                    try:
                        write(write_payload)
                    finally:
                        trace.end("write", stage_started)
                    if not put_or_cancel(queues.write_completions, _PIPELINE_COMPLETION):
                        return
                finally:
                    queues.write_requests.task_done()
                    message = None
                    write_item = None
                    write_payload = None
                    if owns_item:
                        release_item()

    def worker(role: str, _worker_index: int) -> None:
        try:
            worker_body(role)
        except BaseException as error:
            record_worker_error(role, error)

    group = create_parked_stereo_workers(
        pending_render_count=plan.render_count,
        effective_workers=effective_workers,
        writer_workers=writer_workers,
        worker=worker,
    )
    actual_threads = group.live_thread_count
    parked_committed_delta = max(0, current_process_committed_bytes() - before_committed)
    rendered = 0
    completions = 0
    pending_decoded: dict[int, tuple[object, object]] = {}
    expected_render_indexes = iter(
        index for index, action in enumerate(plan.actions) if action == StereoAction.RENDER
    )
    next_render_index = next(expected_render_indexes, None)
    try:
        validate_stereo_provider_replay(
            provider,
            plan,
            mutation_started=False,
        )
        if pre_mutation is not None:
            pre_mutation()
        if mutation is not None:
            mutation_started = True
            mutation()
        group.release()
        while rendered < plan.render_count:
            decoded_message: object | None = None
            if ordered_render and next_render_index in pending_decoded:
                assert next_render_index is not None
                decoded_message = pending_decoded.pop(next_render_index)
            else:
                decoded_message = coordinator_get(queues.decoded_frames)
                queues.decoded_frames.task_done()
                if ordered_render:
                    if not isinstance(decoded_message, tuple) or len(decoded_message) != 2:
                        release_item()
                        decoded_message = None
                        raise StereoControlStateError("renderer received a malformed message")
                    pending_item = decoded_message[0]
                    if not isinstance(pending_item, StereoProviderItem):
                        release_item()
                        decoded_message = None
                        raise StereoControlStateError("renderer received a non-item")
                    source_index = pending_item.source_index
                    if (
                        source_index in pending_decoded
                        or plan.actions.get(source_index) != StereoAction.RENDER
                    ):
                        release_item()
                        decoded_message = None
                        raise StereoControlStateError(
                            "ordered renderer received an unexpected source index"
                        )
                    pending_decoded[source_index] = decoded_message
                    decoded_message = None
                    if next_render_index not in pending_decoded:
                        continue
                    assert next_render_index is not None
                    decoded_message = pending_decoded.pop(next_render_index)
            handed_to_writer = False
            render_item: StereoProviderItem | None = None
            payload: object | None = None
            rendered_payload: object | None = None
            try:
                if not isinstance(decoded_message, tuple) or len(decoded_message) != 2:
                    raise StereoControlStateError("renderer received a malformed message")
                render_item = decoded_message[0]
                if not isinstance(render_item, StereoProviderItem):
                    raise StereoControlStateError("renderer received a non-item")
                payload = decoded_message[1]
                validate_handoff(render_item)
                stage_started = trace.begin("render")
                try:
                    rendered_payload = render(payload)
                finally:
                    trace.end("render", stage_started)
                handed_to_writer = put_or_cancel(
                    queues.write_requests,
                    (render_item, rendered_payload),
                )
                if not handed_to_writer:
                    raise_worker_error()
                rendered += 1
                if ordered_render:
                    next_render_index = next(expected_render_indexes, None)
                while True:
                    try:
                        queues.write_completions.get_nowait()
                    except queue.Empty:
                        break
                    else:
                        queues.write_completions.task_done()
                        completions += 1
            finally:
                decoded_message = None
                render_item = None
                payload = None
                rendered_payload = None
                if not handed_to_writer:
                    release_item()
        for _ in range(writer_workers):
            if not put_or_cancel(queues.write_requests, _PIPELINE_SENTINEL):
                raise_worker_error()
        while completions < plan.render_count:
            coordinator_get(queues.write_completions)
            queues.write_completions.task_done()
            completions += 1
        group.join()
        live_threads_after_join = group.live_thread_count
        if pipeline_cancel.is_set() or not worker_errors.empty():
            raise_worker_error()
        if not feeder_complete.is_set():
            raise StereoControlStateError(
                "stereo feeder ended without completing its authoritative replay"
            )
    except BaseException as error:
        pipeline_cancel.set()
        group.cancel_and_join()
        for _message in pending_decoded.values():
            release_item()
        pending_decoded.clear()
        worker_failure: tuple[str, BaseException] | None = None
        try:
            worker_failure = worker_errors.get_nowait()
            worker_errors.task_done()
        except queue.Empty:
            pass
        drain_owned_queues()
        while True:
            try:
                worker_errors.get_nowait()
            except queue.Empty:
                break
            worker_errors.task_done()
        if worker_failure is not None:
            role, worker_error = worker_failure
            worker_failure = None
            if isinstance(worker_error, StereoInputIdentityMismatch):
                _raise_provider_changed(str(worker_error), mutation_started)
            if role == "feeder" and isinstance(
                worker_error,
                (StereoControlError, OSError, TypeError, ValueError),
            ):
                _raise_provider_changed(str(worker_error), mutation_started)
            raise worker_error
        if isinstance(error, StereoInputIdentityMismatch):
            _raise_provider_changed(str(error), mutation_started)
        raise error
    return StreamingStereoPipelineStats(
        configured_capacity=capacity,
        effective_workers=effective_workers,
        actual_thread_count=actual_threads,
        queue_capacities=queues.capacities,
        requested_stack_bytes=group.stats.requested_stack_bytes,
        decoded=trace.decoded,
        rendered=trace.rendered,
        written=trace.written,
        maximum_live_items=trace.maximum_live_items,
        decode_render_overlap=trace.decode_render_overlap,
        render_write_overlap=trace.render_write_overlap,
        decoder_writer_overlap=trace.decoder_writer_overlap,
        work_item_copies=0,
        parked_committed_delta_bytes=parked_committed_delta,
        live_threads_after_join=live_threads_after_join,
        pipeline_wall_seconds=time.perf_counter() - pipeline_started,
        decode_busy_seconds=trace.busy_seconds["decode"],
        render_busy_seconds=trace.busy_seconds["render"],
        write_busy_seconds=trace.busy_seconds["write"],
        configured_workers=configured_workers,
        feeder_thread_count=1,
        decoder_thread_count=effective_workers,
        writer_thread_count=writer_workers,
        calculated_thread_bytes=actual_threads * STEREO_IO_THREAD_BYTES,
        previous_stack_bytes=group.stats.previous_stack_bytes,
        restored_stack_bytes=group.stats.restored_stack_bytes,
        stack_granularity_bytes=tuple(
            evidence.granularity_bytes for evidence in group.stats.stack_evidence
        ),
        reported_stack_reserve_bytes=tuple(
            evidence.reserve_bytes for evidence in group.stats.stack_evidence
        ),
    )


def execute_stereo_control_plan(
    *,
    provider: ReplayableStereoProvider,
    plan: StereoControlAuditPlan,
    capacity: int,
    effective_workers: int,
    decode: Callable[[StereoProviderItem], object],
    render: Callable[[object], object],
    write: Callable[[object], object],
    migrate: Callable[[StereoProviderItem], None],
) -> StereoControlExecutionStats:
    """Run P, join all threads, then replay R synchronously."""

    pipeline = run_streaming_stereo_pipeline(
        provider=provider,
        plan=plan,
        capacity=capacity,
        effective_workers=effective_workers,
        decode=decode,
        render=render,
        write=write,
    )
    migration_threads = current_process_thread_count()
    if pipeline.live_threads_after_join != 0:
        raise StereoThreadBudgetError(
            "teardown",
            "pipeline threads remain live before migration",
            pipeline.actual_thread_count,
            pipeline.actual_thread_count - pipeline.live_threads_after_join,
        )
    migrated = 0
    for item in iter_stereo_action(provider, plan, StereoAction.MIGRATE):
        migrate(item)
        migrated += 1
    if migrated != plan.migrate_count:
        raise StereoControlStateError("migration replay count disagrees with the action vector")
    return StereoControlExecutionStats(pipeline, migrated, migration_threads)


def _windows_stack_evidence() -> StackReservationEvidence:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    low = ctypes.c_size_t()
    high = ctypes.c_size_t()
    get_limits = kernel32.GetCurrentThreadStackLimits
    get_limits.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    get_limits.restype = None
    get_limits(ctypes.byref(low), ctypes.byref(high))
    if high.value <= low.value:
        raise OSError("GetCurrentThreadStackLimits returned invalid bounds")

    class _SystemInfo(ctypes.Structure):
        _fields_ = [
            ("processor_architecture", ctypes.c_ushort),
            ("reserved", ctypes.c_ushort),
            ("page_size", ctypes.c_uint32),
            ("minimum_application_address", ctypes.c_void_p),
            ("maximum_application_address", ctypes.c_void_p),
            ("active_processor_mask", ctypes.c_size_t),
            ("number_of_processors", ctypes.c_uint32),
            ("processor_type", ctypes.c_uint32),
            ("allocation_granularity", ctypes.c_uint32),
            ("processor_level", ctypes.c_ushort),
            ("processor_revision", ctypes.c_ushort),
        ]

    class _MemoryBasicInformation(ctypes.Structure):
        _fields_ = [
            ("base_address", ctypes.c_void_p),
            ("allocation_base", ctypes.c_void_p),
            ("allocation_protect", ctypes.c_uint32),
            ("partition_id", ctypes.c_ushort),
            ("region_size", ctypes.c_size_t),
            ("state", ctypes.c_uint32),
            ("protect", ctypes.c_uint32),
            ("kind", ctypes.c_uint32),
        ]

    system_info = _SystemInfo()
    kernel32.GetSystemInfo(ctypes.byref(system_info))
    virtual_query = kernel32.VirtualQuery
    virtual_query.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_MemoryBasicInformation),
        ctypes.c_size_t,
    ]
    virtual_query.restype = ctypes.c_size_t
    committed = 0
    address = low.value
    while address < high.value:
        info = _MemoryBasicInformation()
        result = virtual_query(
            ctypes.c_void_p(address),
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if result != ctypes.sizeof(info) or info.region_size <= 0:
            raise OSError(ctypes.get_last_error(), "VirtualQuery failed for stack bounds")
        region_end = min(address + int(info.region_size), high.value)
        if info.state == 0x1000:
            committed += region_end - address
        address = region_end
    return StackReservationEvidence(
        adapter="windows-get-stack-limits-v1",
        reserve_bytes=int(high.value - low.value),
        committed_bytes=int(committed),
        granularity_bytes=int(system_info.allocation_granularity),
    )


def _linux_stack_evidence() -> StackReservationEvidence:
    libc = ctypes.CDLL(None, use_errno=True)
    pthread_self = libc.pthread_self
    pthread_self.restype = ctypes.c_ulong
    pthread_getattr_np = libc.pthread_getattr_np
    pthread_getattr_np.argtypes = [ctypes.c_ulong, ctypes.c_void_p]
    pthread_getattr_np.restype = ctypes.c_int
    pthread_attr_getstack = libc.pthread_attr_getstack
    pthread_attr_getstack.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    pthread_attr_getstack.restype = ctypes.c_int
    pthread_attr_destroy = libc.pthread_attr_destroy
    pthread_attr_destroy.argtypes = [ctypes.c_void_p]
    attr = ctypes.create_string_buffer(256)
    result = pthread_getattr_np(pthread_self(), ctypes.byref(attr))
    if result != 0:
        raise OSError(result, "pthread_getattr_np failed")
    try:
        address = ctypes.c_void_p()
        size = ctypes.c_size_t()
        result = pthread_attr_getstack(
            ctypes.byref(attr), ctypes.byref(address), ctypes.byref(size)
        )
        if result != 0:
            raise OSError(result, "pthread_attr_getstack failed")
    finally:
        pthread_attr_destroy(ctypes.byref(attr))
    return StackReservationEvidence(
        adapter="linux-pthread-getattr-v1",
        reserve_bytes=int(size.value),
        committed_bytes=0,
        granularity_bytes=int(cast(Callable[[str], int], getattr(os, "sysconf"))("SC_PAGE_SIZE")),
    )


def attest_current_thread_stack() -> StackReservationEvidence:
    """Return current worker reserve facts through the frozen platform adapter."""

    if sys.platform == "win32":
        return _windows_stack_evidence()
    if sys.platform.startswith("linux"):
        return _linux_stack_evidence()
    raise OSError(f"unsupported stereo stack-attestation platform: {sys.platform}")


def _windows_process_thread_count() -> int:
    class _ThreadEntry32(ctypes.Structure):
        _fields_ = [
            ("size", ctypes.c_uint32),
            ("usage", ctypes.c_uint32),
            ("thread_id", ctypes.c_uint32),
            ("owner_process_id", ctypes.c_uint32),
            ("base_priority", ctypes.c_long),
            ("priority_delta", ctypes.c_long),
            ("flags", ctypes.c_uint32),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    snapshot_api = kernel32.CreateToolhelp32Snapshot
    snapshot_api.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    snapshot_api.restype = ctypes.c_void_p
    snapshot = snapshot_api(0x00000004, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if snapshot == invalid_handle:
        raise OSError(ctypes.get_last_error(), "CreateToolhelp32Snapshot failed")
    entry = _ThreadEntry32(size=ctypes.sizeof(_ThreadEntry32))
    first = kernel32.Thread32First
    next_entry = kernel32.Thread32Next
    first.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ThreadEntry32)]
    next_entry.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ThreadEntry32)]
    first.restype = ctypes.c_int
    next_entry.restype = ctypes.c_int
    process_id = os.getpid()
    count = 0
    try:
        success = first(snapshot, ctypes.byref(entry))
        while success:
            count += int(entry.owner_process_id == process_id)
            entry.size = ctypes.sizeof(_ThreadEntry32)
            success = next_entry(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    if count <= 0:
        raise OSError("thread snapshot did not contain the current process")
    return count


def current_process_thread_count() -> int:
    """Return the OS-visible thread count, including foreign CRT workers."""

    if sys.platform == "win32":
        return _windows_process_thread_count()
    if sys.platform.startswith("linux"):
        with os.scandir("/proc/self/task") as entries:
            return sum(1 for entry in entries if entry.name.isdecimal())
    raise OSError(f"unsupported thread-count platform: {sys.platform}")


def _windows_process_committed_bytes() -> int:
    class _ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("size", ctypes.c_uint32),
            ("page_fault_count", ctypes.c_uint32),
            ("peak_working_set_size", ctypes.c_size_t),
            ("working_set_size", ctypes.c_size_t),
            ("quota_peak_paged_pool_usage", ctypes.c_size_t),
            ("quota_paged_pool_usage", ctypes.c_size_t),
            ("quota_peak_nonpaged_pool_usage", ctypes.c_size_t),
            ("quota_nonpaged_pool_usage", ctypes.c_size_t),
            ("pagefile_usage", ctypes.c_size_t),
            ("peak_pagefile_usage", ctypes.c_size_t),
            ("private_usage", ctypes.c_size_t),
        ]

    counters = _ProcessMemoryCountersEx(size=ctypes.sizeof(_ProcessMemoryCountersEx))
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    get_memory = psapi.GetProcessMemoryInfo
    get_memory.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    get_memory.restype = ctypes.c_int
    process = kernel32.GetCurrentProcess()
    if not get_memory(process, ctypes.byref(counters), counters.size):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return int(counters.private_usage)


def current_process_committed_bytes() -> int:
    """Return process-private committed bytes for thread-envelope measurements."""

    if sys.platform == "win32":
        return _windows_process_committed_bytes()
    if sys.platform.startswith("linux"):
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
        raise OSError("/proc/self/status has no VmRSS field")
    raise OSError(f"unsupported committed-memory platform: {sys.platform}")
