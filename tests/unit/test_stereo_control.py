"""Production contracts for the replayable stereo control plane and parked workers."""

from __future__ import annotations

from dataclasses import replace
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import threading
import time
import weakref

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.processing.frames import stereo_control, stereo_input_manifest
from src.depth_surge_3d.processing.frames.metric_geometry import (
    ClipConvergence,
    MetricGeometryFrame,
    MetricGeometryStore,
)


def _generation(frame_count: int) -> stereo_control.StereoProviderGeneration:
    return stereo_control.StereoProviderGeneration(
        audit_generation_id="0123456789abcdef0123456789abcdef",
        input_manifest_fingerprint="ab" * 32,
        frame_count=frame_count,
    )


def _item(
    generation: stereo_control.StereoProviderGeneration,
    index: int,
) -> stereo_control.StereoProviderItem:
    stem = stereo_control.canonical_frame_stem(index)
    return stereo_control.StereoProviderItem(
        audit_generation_id=generation.audit_generation_id,
        input_manifest_fingerprint=generation.input_manifest_fingerprint,
        source_index=index,
        frame_name=stem,
        guide_path=Path("guides") / f"{stem}.png",
        geometry_path=Path("geometry") / f"{stem}.npz",
        left_path=Path("left") / f"{stem}.png",
        right_path=Path("right") / f"{stem}.png",
        guide_sha256=f"{index:064x}"[-64:],
        geometry_sha256=f"{index + 1:064x}"[-64:],
        guide_byte_count=4096,
        geometry_byte_count=2048,
        guide_shape=(1080, 1920),
        geometry_shape=(540, 960),
        frame_json_bound=1024 * 1024,
    )


def _provider(frame_count: int) -> stereo_control.ReplayableStereoProvider:
    generation = _generation(frame_count)

    def replay():
        for index in range(frame_count):
            yield _item(generation, index)

    return stereo_control.ReplayableStereoProvider(generation, replay)


@pytest.mark.parametrize(
    ("frame_count", "expected_bytes"),
    [(1, 1), (4, 1), (5, 2), (10_000, 2_500), (4_194_304, 1_048_576)],
)
def test_packed_action_vector_has_the_exact_two_bit_allocation(
    frame_count: int,
    expected_bytes: int,
) -> None:
    actions = stereo_control.PackedStereoActions(frame_count)

    assert actions.storage_nbytes == expected_bytes
    assert len(actions.storage) == expected_bytes
    assert sys.getsizeof(actions.storage) + sys.getsizeof(actions) < 4 * 1024 * 1024


def test_action_vector_tracks_p_zero_r_n_and_mixed_counts() -> None:
    all_migrate = stereo_control.audit_stereo_provider(
        _provider(11),
        lambda _item: stereo_control.StereoAction.MIGRATE,
    )
    mixed = stereo_control.audit_stereo_provider(
        _provider(12),
        lambda item: stereo_control.StereoAction(item.source_index % 3),
    )

    assert (all_migrate.render_count, all_migrate.migrate_count) == (0, 11)
    assert list(all_migrate.actions) == [stereo_control.StereoAction.MIGRATE] * 11
    assert (mixed.render_count, mixed.migrate_count) == (4, 4)
    assert list(mixed.actions) == [stereo_control.StereoAction(index % 3) for index in range(12)]


def test_invalid_action_is_never_admitted_or_replayed() -> None:
    actions = stereo_control.PackedStereoActions(2)

    with pytest.raises(stereo_control.StereoControlStateError, match="invalid"):
        actions.set(0, stereo_control.StereoAction.INVALID)

    actions._storage[0] = 0b11
    with pytest.raises(stereo_control.StereoControlStateError, match="invalid"):
        actions.get(0)


def test_provider_audit_is_lazy_and_freezes_one_generation() -> None:
    provider = _provider(10_000)
    yielded = 0

    def replay():
        nonlocal yielded
        for item in provider.replay():
            yielded += 1
            yield item

    wrapped = stereo_control.ReplayableStereoProvider(provider.generation, replay)
    plan = stereo_control.audit_stereo_provider(
        wrapped,
        lambda item: (
            stereo_control.StereoAction.RENDER
            if item.source_index in {0, 9999}
            else stereo_control.StereoAction.REUSE
        ),
    )

    assert yielded == 10_000
    assert plan.generation == provider.generation
    assert plan.render_count == 2
    assert plan.migrate_count == 0
    assert plan.persistent_control_bytes < stereo_control.STEREO_CONTROL_OVERHEAD
    assert not hasattr(plan, "items")


def _published_relative_provider(tmp_path: Path, count: int = 2):
    guides = tmp_path / "00_original_frames"
    geometry = tmp_path / "03_disparity_maps"
    left = tmp_path / "04_left_frames"
    right = tmp_path / "04_right_frames"
    for directory in (guides, geometry, left, right):
        directory.mkdir()
    guide_paths: list[Path] = []
    geometry_paths: list[Path] = []
    for source_index in range(count):
        frame_index = source_index + 1
        stem = f"frame_{frame_index:06d}"
        guide = guides / f"{stem}.png"
        disparity = geometry / f"{stem}.png"
        assert cv2.imwrite(
            str(guide),
            np.full((4, 6, 3), 20 + source_index, dtype=np.uint8),
        )
        assert cv2.imwrite(
            str(disparity),
            np.full((2, 3), 100 + source_index, dtype=np.uint16),
        )
        guide_paths.append(guide)
        geometry_paths.append(disparity)
    descriptor = stereo_input_manifest.publish_stereo_input_manifest_v1(
        manifest_path=geometry / "stereo_input_manifest.jsonl",
        job_root=tmp_path,
        guide_files=iter(guide_paths),
        geometry_files=iter(geometry_paths),
        frame_count=count,
        geometry_mode="relative",
        source_metadata={
            "fingerprint": "ab" * 32,
            "encoding_scale": 65535.0,
        },
    )
    provider = descriptor.open_provider(left_dir=left, right_dir=right)
    return descriptor, provider, guide_paths, geometry_paths


def test_canonical_upstream_manifest_owns_provider_identity_without_input_lists(
    tmp_path: Path,
) -> None:
    descriptor, provider, guide_paths, geometry_paths = _published_relative_provider(tmp_path)
    guide_paths.clear()
    geometry_paths.clear()

    items = list(provider.replay())

    assert descriptor.raw_sha256 == hashlib.sha256(descriptor.path.read_bytes()).hexdigest()
    assert provider.generation.input_manifest_fingerprint == descriptor.raw_sha256
    assert [item.frame_name for item in items] == ["frame_000001", "frame_000002"]
    assert [item.source_index for item in items] == [0, 1]
    assert not hasattr(descriptor, "items")
    assert not hasattr(provider, "items")

    adopted = stereo_input_manifest.load_stereo_input_manifest_v1(
        descriptor.path,
        job_root=tmp_path,
    )
    assert adopted == descriptor


def test_every_provider_replay_rehashes_actual_source_payload(tmp_path: Path) -> None:
    descriptor, provider, guide_paths, _geometry_paths = _published_relative_provider(
        tmp_path,
        count=1,
    )
    plan = stereo_control.audit_stereo_provider(
        provider,
        lambda _item: stereo_control.StereoAction.REUSE,
    )
    assert cv2.imwrite(str(guide_paths[0]), np.full((4, 6, 3), 99, dtype=np.uint8))

    with pytest.raises(stereo_control.StereoProviderChangedError, match="payload changed"):
        stereo_control.validate_stereo_provider_replay(
            provider,
            plan,
            mutation_started=False,
        )


def test_relative_provider_replay_streams_each_declared_payload_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _descriptor, provider, _guide_paths, _geometry_paths = _published_relative_provider(
        tmp_path,
        count=2,
    )
    original = stereo_input_manifest._stream_identity
    calls = 0

    def counted(path: Path, **kwargs):
        nonlocal calls
        calls += 1
        return original(path, **kwargs)

    monkeypatch.setattr(stereo_input_manifest, "_stream_identity", counted)

    assert len(list(provider.replay())) == 2
    assert calls == 4


def test_provider_replay_buffers_manifest_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor, provider, _guide_paths, _geometry_paths = _published_relative_provider(
        tmp_path,
        count=2,
    )
    original_regular_file = stereo_input_manifest._regular_file
    line_reads = 0
    chunk_reads = 0

    class CountingRaw(io.RawIOBase):
        def __init__(self, handle) -> None:
            self._handle = handle

        def readable(self) -> bool:
            return True

        def fileno(self) -> int:
            return self._handle.fileno()

        def readinto(self, target) -> int:
            nonlocal chunk_reads
            chunk_reads += 1
            return self._handle.readinto(target)

        def readline(self, size: int = -1) -> bytes:
            nonlocal line_reads
            line_reads += 1
            return self._handle.readline(size)

        def close(self) -> None:
            self._handle.close()
            super().close()

    def counted_regular_file(path: Path, **kwargs):
        handle, metadata = original_regular_file(path, **kwargs)
        if Path(path) == descriptor.path:
            return CountingRaw(handle), metadata
        return handle, metadata

    monkeypatch.setattr(
        stereo_input_manifest,
        "_regular_file",
        counted_regular_file,
    )

    assert len(list(provider.replay())) == 2
    assert line_reads == 0
    assert chunk_reads <= 2


def test_provider_replay_uses_one_acquired_root_instance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _descriptor, provider, _guide_paths, _geometry_paths = _published_relative_provider(
        tmp_path,
        count=2,
    )
    acquired_root = stereo_input_manifest.AcquiredRoot
    instances = 0

    class CountedAcquiredRoot(acquired_root):
        def __init__(self, root: Path) -> None:
            nonlocal instances
            instances += 1
            super().__init__(root)

    monkeypatch.setattr(
        stereo_input_manifest,
        "AcquiredRoot",
        CountedAcquiredRoot,
    )

    assert len(list(provider.replay())) == 2
    assert instances == 1


@pytest.mark.skipif(os.name == "nt", reason="Windows keeps the acquired directory named")
def test_provider_replay_keeps_one_acquired_root_for_manifest_and_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_root = tmp_path / "job"
    job_root.mkdir()
    _descriptor, provider, _guide_paths, _geometry_paths = _published_relative_provider(
        job_root,
        count=1,
    )
    displaced = tmp_path / "job-displaced"
    original_read_line = stereo_input_manifest._read_line
    line_count = 0

    def swap_root_after_header(handle):
        nonlocal line_count
        raw = original_read_line(handle)
        line_count += 1
        if line_count == 1:
            job_root.rename(displaced)
            job_root.mkdir()
        return raw

    monkeypatch.setattr(stereo_input_manifest, "_read_line", swap_root_after_header)

    items = list(provider.replay())

    assert [item.frame_name for item in items] == ["frame_000001"]
    assert (displaced / "00_original_frames" / "frame_000001.png").is_file()
    assert list(job_root.iterdir()) == []


def test_upstream_completion_publishes_validated_relative_manifest(tmp_path: Path) -> None:
    guides = tmp_path / "00_original_frames"
    geometry = tmp_path / "03_disparity_maps"
    guides.mkdir()
    geometry.mkdir()
    guide = guides / "frame_000001.png"
    disparity = geometry / "frame_000001.png"
    assert cv2.imwrite(str(guide), np.full((4, 6, 3), 20, dtype=np.uint8))
    assert cv2.imwrite(str(disparity), np.full((2, 3), 100, dtype=np.uint16))
    metadata = {
        "schema_version": 1,
        "algorithm_version": "scene-percentile-v1",
        "representation": "relative_disparity",
        "near_value": 1.0,
        "far_value": 0.0,
        "encoding": "uint16_png",
        "encoding_scale": 65535.0,
        "num_frames": 1,
        "frame_names": [guide.name],
        "native_shape": [2, 3],
        "source_raw_fingerprint": "raw",
        "source_model_fingerprint": "model",
        "scene_manifest_fingerprint": "scene",
        "depth_bounds_fingerprint": "bounds",
    }
    from src.depth_surge_3d.processing.frames.depth_storage import canonical_json_hash

    metadata["fingerprint"] = canonical_json_hash(metadata)
    (geometry / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    descriptor = stereo_input_manifest.publish_validated_stereo_input_manifest_v1(
        job_root=tmp_path,
        guide_files=[guide],
        geometry_files=[disparity],
        geometry_mode="relative",
    )

    assert descriptor.path == geometry / "stereo_input_manifest.jsonl"
    assert descriptor.frame_count == 1
    assert descriptor.source_geometry_fingerprint == metadata["fingerprint"]


def test_upstream_completion_publishes_validated_metric_manifest(tmp_path: Path) -> None:
    guides = tmp_path / "00_original_frames"
    geometry = tmp_path / "03_metric_geometry"
    guides.mkdir()
    guide = guides / "frame_000001.png"
    assert cv2.imwrite(str(guide), np.full((4, 6, 3), 20, dtype=np.uint8))
    store = MetricGeometryStore.open_or_create(
        geometry,
        frame_names=[guide.name],
        native_shape=(2, 3),
        source_raw_fingerprint="raw-metric",
        source_frame_fingerprint="frames",
        candidate_scene_fingerprint="scenes",
        preflight_required_bytes=0,
    )
    store.write_frame(
        guide.name,
        MetricGeometryFrame(
            np.ones((2, 3), dtype=np.float32),
            np.ones((2, 3), dtype=np.bool_),
            np.float32(0.5),
        ),
    )
    store.finalize(ClipConvergence(np.float32(2.0), (0,), 6))

    descriptor = stereo_input_manifest.publish_validated_stereo_input_manifest_v1(
        job_root=tmp_path,
        guide_files=[guide],
        geometry_files=list(store.complete_files),
        geometry_mode="metric_camera",
    )

    assert descriptor.geometry_mode == "metric"
    assert descriptor.native_shape == (2, 3)
    assert descriptor.resolved_auto_distance_m == pytest.approx(2.0)
    assert descriptor.source_geometry_fingerprint == store.metadata["fingerprint"]


@pytest.mark.parametrize(
    "field",
    [
        "audit_generation_id",
        "input_manifest_fingerprint",
        "source_index",
        "frame_name",
        "guide_path",
        "geometry_path",
        "left_path",
        "right_path",
        "guide_sha256",
        "geometry_sha256",
        "count",
    ],
)
def test_every_provider_identity_change_is_detected_on_replay(field: str) -> None:
    original = _provider(3)
    plan = stereo_control.audit_stereo_provider(
        original,
        lambda _item: stereo_control.StereoAction.REUSE,
    )
    generation = original.generation
    replacement_generation = generation
    replacement_count = generation.frame_count
    if field == "count":
        replacement_count += 1
        replacement_generation = replace(generation, frame_count=replacement_count)

    def changed_replay():
        for index in range(replacement_count):
            item = _item(replacement_generation, index)
            if index == 1 and field != "count":
                if field == "source_index":
                    item = replace(item, source_index=2)
                elif field in {"guide_path", "geometry_path", "left_path", "right_path"}:
                    item = replace(item, **{field: Path("changed") / getattr(item, field).name})
                elif field in {"audit_generation_id", "input_manifest_fingerprint"}:
                    item = replace(
                        item, **{field: "cd" * (16 if field.startswith("audit") else 32)}
                    )
                else:
                    item = replace(item, **{field: "cd" * 32})
            yield item

    changed = stereo_control.ReplayableStereoProvider(replacement_generation, changed_replay)

    with pytest.raises(stereo_control.StereoProviderChangedError):
        stereo_control.validate_stereo_provider_replay(changed, plan, mutation_started=False)
    with pytest.raises(stereo_control.StereoStageInputChangedError):
        stereo_control.validate_stereo_provider_replay(changed, plan, mutation_started=True)


def test_replay_yields_only_the_selected_action_without_resident_sets() -> None:
    provider = _provider(20)
    plan = stereo_control.audit_stereo_provider(
        provider,
        lambda item: (
            stereo_control.StereoAction.RENDER
            if item.source_index % 5 == 0
            else stereo_control.StereoAction.REUSE
        ),
    )

    selected = list(
        stereo_control.iter_stereo_action(
            provider,
            plan,
            stereo_control.StereoAction.RENDER,
        )
    )

    assert [item.source_index for item in selected] == [0, 5, 10, 15]


@pytest.mark.parametrize("failure_position", ["before_first", "middle", "after_last"])
def test_action_replay_wraps_iterator_failures_as_post_mutation_input_change(
    failure_position: str,
) -> None:
    original = _provider(3)
    plan = stereo_control.audit_stereo_provider(
        original,
        lambda _item: stereo_control.StereoAction.RENDER,
    )

    def replay():
        if failure_position == "before_first":
            raise OSError("injected provider iteration failure")
        for item in original.replay():
            yield item
            if failure_position == "middle" and item.source_index == 0:
                raise ValueError("injected provider iteration failure")
        if failure_position == "after_last":
            raise stereo_control.StereoControlStateError("injected provider iteration failure")

    changed = stereo_control.ReplayableStereoProvider(original.generation, replay)

    with pytest.raises(
        stereo_control.StereoStageInputChangedError,
        match="injected provider iteration failure",
    ):
        list(
            stereo_control.iter_stereo_action(
                changed,
                plan,
                stereo_control.StereoAction.RENDER,
            )
        )


def test_quality_capacity_is_exactly_one_and_p_zero_creates_no_control() -> None:
    assert stereo_control.quality_pipeline_capacity(0) == (0, 0, 0)
    assert stereo_control.quality_pipeline_capacity(1) == (1, 1, 6 * 1024 * 1024)
    assert stereo_control.quality_pipeline_capacity(1_000_000) == (
        1,
        1,
        6 * 1024 * 1024,
    )


def test_quality_capacity_overlaps_decode_render_and_write_when_memory_allows() -> None:
    selection = stereo_control.select_quality_pipeline_capacity(
        configured_workers=8,
        pending_render_count=60,
        quality_active_scratch=240 * 1024 * 1024,
        quality_slot_bytes=72 * 1024 * 1024,
    )

    assert selection.capacity == 4
    assert selection.effective_workers == 4
    assert selection.writer_workers == 3
    assert selection.thread_bytes == 16 * 1024 * 1024
    assert selection.total_bytes == 472 * 1024 * 1024


def test_quality_capacity_charges_bounded_parallel_png_encoders() -> None:
    selection = stereo_control.select_quality_pipeline_capacity(
        configured_workers=8,
        pending_render_count=60,
        quality_active_scratch=240 * 1024 * 1024,
        quality_slot_bytes=72 * 1024 * 1024,
        png_payloads_per_item=6,
    )

    assert selection.capacity == 4
    assert selection.effective_workers == 4
    assert selection.writer_workers == 3
    assert selection.png_encoder_workers == 4
    assert selection.thread_bytes == 32 * 1024 * 1024
    assert selection.total_bytes == 488 * 1024 * 1024


@pytest.mark.parametrize(
    ("remaining_mib", "png_workers"), [(0, 0), (4, 0), (8, 2), (12, 3), (16, 4)]
)
def test_quality_capacity_preserves_lifecycle_slots_and_png_runtime_slack(
    remaining_mib: int, png_workers: int
) -> None:
    selection = stereo_control.select_quality_pipeline_capacity(
        configured_workers=8,
        pending_render_count=60,
        quality_active_scratch=(280 - remaining_mib) * 1024 * 1024,
        quality_slot_bytes=72 * 1024 * 1024,
        png_payloads_per_item=6,
    )

    assert selection.capacity == 4
    assert selection.png_encoder_workers == png_workers
    assert selection.thread_bytes == (16 + png_workers * 4) * 1024 * 1024
    assert selection.total_bytes == (512 - remaining_mib + png_workers * 4) * 1024 * 1024


def test_quality_capacity_charges_a_retained_duplicate_cache() -> None:
    selection = stereo_control.select_quality_pipeline_capacity(
        configured_workers=8,
        pending_render_count=60,
        quality_active_scratch=240 * 1024 * 1024,
        quality_slot_bytes=72 * 1024 * 1024,
        retained_bytes=72 * 1024 * 1024,
    )

    assert selection.capacity == 3
    assert selection.effective_workers == 3
    assert selection.writer_workers == 3
    assert selection.thread_bytes == 14 * 1024 * 1024
    assert selection.total_bytes == 470 * 1024 * 1024


def test_quality_capacity_falls_back_to_two_when_third_slot_exceeds_budget() -> None:
    selection = stereo_control.select_quality_pipeline_capacity(
        configured_workers=8,
        pending_render_count=60,
        quality_active_scratch=384 * 1024 * 1024,
        quality_slot_bytes=60 * 1024 * 1024,
    )

    assert selection.capacity == 2
    assert selection.effective_workers == 2
    assert selection.writer_workers == 2
    assert selection.thread_bytes == 10 * 1024 * 1024
    assert selection.total_bytes == 454 * 1024 * 1024


def test_quality_capacity_falls_back_to_one_when_overlap_exceeds_budget() -> None:
    selection = stereo_control.select_quality_pipeline_capacity(
        configured_workers=8,
        pending_render_count=60,
        quality_active_scratch=450 * 1024 * 1024,
        quality_slot_bytes=72 * 1024 * 1024,
    )

    assert selection.capacity == 1
    assert selection.effective_workers == 1
    assert selection.writer_workers == 1
    assert selection.total_bytes == 456 * 1024 * 1024


def test_fast_capacity_uses_the_unique_largest_eligible_integer() -> None:
    selection = stereo_control.select_fast_pipeline_capacity(
        configured_workers=4,
        pending_render_count=20,
        fast_active_scratch=8 * 1024 * 1024,
        fast_slot_bytes=96 * 1024 * 1024,
    )

    assert selection.capacity == 5
    assert selection.effective_workers == 4
    assert selection.thread_bytes == 18 * 1024 * 1024
    assert selection.total_bytes == 506 * 1024 * 1024
    assert selection.total_bytes <= stereo_control.STEREO_HOST_BUDGET


def test_fast_capacity_charges_retained_consumer_memory() -> None:
    selection = stereo_control.select_fast_pipeline_capacity(
        configured_workers=4,
        pending_render_count=20,
        fast_active_scratch=8 * 1024 * 1024,
        fast_slot_bytes=96 * 1024 * 1024,
        retained_bytes=72 * 1024 * 1024,
    )

    assert selection.capacity == 4
    assert selection.effective_workers == 4
    assert selection.thread_bytes == 18 * 1024 * 1024
    assert selection.total_bytes == 482 * 1024 * 1024
    assert selection.total_bytes <= stereo_control.STEREO_HOST_BUDGET


def test_fast_capacity_rejection_uses_shared_pipeline_error() -> None:
    with pytest.raises(
        stereo_control.StereoPipelineCapacityError,
        match="Fast lifecycle capacity",
    ):
        stereo_control.select_fast_pipeline_capacity(
            configured_workers=1,
            pending_render_count=1,
            fast_active_scratch=500 * 1024 * 1024,
            fast_slot_bytes=96 * 1024 * 1024,
        )


def _fixed_stack_evidence() -> stereo_control.StackReservationEvidence:
    return stereo_control.StackReservationEvidence(
        adapter="test",
        reserve_bytes=512 * 1024,
        committed_bytes=64 * 1024,
        granularity_bytes=64 * 1024,
    )


def test_parked_workers_restore_stack_setting_before_work_release() -> None:
    calls: list[int | None] = []
    setting = 0

    def stack_size(value: int | None = None) -> int:
        nonlocal setting
        calls.append(value)
        if value is None:
            return setting
        previous = setting
        setting = value
        return previous

    ran: list[str] = []
    group = stereo_control.create_parked_stereo_workers(
        pending_render_count=5,
        effective_workers=2,
        worker=lambda role, index: ran.append(f"{role}:{index}"),
        stack_adapter=_fixed_stack_evidence,
        stack_size_api=stack_size,
    )

    assert group.stats.thread_names == (
        "stereo-feeder-0",
        "stereo-decoder-0",
        "stereo-decoder-1",
        "stereo-writer-0",
        "stereo-writer-1",
    )
    assert group.stats.requested_stack_bytes == 512 * 1024
    assert group.stats.previous_stack_bytes == 0
    assert group.stats.restored_stack_bytes == 0
    assert setting == 0
    assert ran == []
    assert group.live_thread_count == 5

    group.release_and_join()

    assert sorted(ran) == sorted(["feeder:0", "decoder:0", "decoder:1", "writer:0", "writer:1"])
    assert group.live_thread_count == 0
    assert calls == [None, 512 * 1024, 0]


def test_p_zero_does_not_touch_global_stack_or_create_threads() -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("P=0 must not touch thread creation")

    group = stereo_control.create_parked_stereo_workers(
        pending_render_count=0,
        effective_workers=0,
        worker=forbidden,
        stack_adapter=forbidden,
        stack_size_api=forbidden,
    )

    assert group.live_thread_count == 0
    assert group.stats.thread_names == ()


def test_p_zero_pipeline_replays_and_crosses_mutation_without_threads() -> None:
    provider = _provider(2)
    plan = stereo_control.audit_stereo_provider(
        provider,
        lambda _item: stereo_control.StereoAction.REUSE,
    )
    mutations: list[str] = []

    stats = stereo_control.run_streaming_stereo_pipeline(
        provider=provider,
        plan=plan,
        capacity=0,
        effective_workers=0,
        configured_workers=4,
        decode=lambda _item: (_ for _ in ()).throw(AssertionError("decode ran")),
        render=lambda _item: (_ for _ in ()).throw(AssertionError("render ran")),
        write=lambda _item: (_ for _ in ()).throw(AssertionError("write ran")),
        mutation=lambda: mutations.append("started"),
    )

    assert mutations == ["started"]
    assert stats.configured_capacity == 0
    assert stats.configured_workers == 4
    assert stats.actual_thread_count == 0
    assert stats.live_threads_after_join == 0


def test_attestation_failure_cancels_and_joins_every_started_worker() -> None:
    calls = 0

    def adapter() -> stereo_control.StackReservationEvidence:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected adapter failure")
        return _fixed_stack_evidence()

    with pytest.raises(stereo_control.StereoThreadBudgetError, match="attestation") as captured:
        stereo_control.create_parked_stereo_workers(
            pending_render_count=2,
            effective_workers=2,
            worker=lambda _role, _index: None,
            stack_adapter=adapter,
        )

    assert captured.value.started_threads == captured.value.joined_threads


def test_attested_executor_primes_and_joins_every_bounded_worker() -> None:
    executor, stats = stereo_control.create_attested_stereo_executor(
        worker_count=3,
        thread_name_prefix="quality-png-test",
    )
    try:
        assert len(stats.thread_names) == 3
        assert len(set(stats.thread_names)) == 3
        assert len(stats.stack_evidence) == 3
        assert all(
            evidence.reserve_bytes <= stereo_control.QUALITY_PNG_ENCODER_STACK_RESERVE_BYTES
            for evidence in stats.stack_evidence
        )
        assert list(executor.map(lambda value: value + 1, range(3))) == [1, 2, 3]
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    assert not [
        thread.name
        for thread in threading.enumerate()
        if thread.name.startswith("quality-png-test")
    ]


def test_attested_executor_failure_joins_every_primed_worker() -> None:
    calls = 0

    def fail_one_attestation() -> stereo_control.StackReservationEvidence:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected executor attestation failure")
        return _fixed_stack_evidence()

    with pytest.raises(
        stereo_control.StereoThreadBudgetError,
        match="attestation",
    ) as captured:
        stereo_control.create_attested_stereo_executor(
            worker_count=3,
            thread_name_prefix="quality-png-failed-test",
            stack_adapter=fail_one_attestation,
        )

    assert captured.value.started_threads == captured.value.joined_threads == 3
    assert not [
        thread.name
        for thread in threading.enumerate()
        if thread.name.startswith("quality-png-failed-test")
    ]


def test_attested_executor_rejects_stack_using_thread_runtime_slack() -> None:
    def oversized_stack() -> stereo_control.StackReservationEvidence:
        return replace(_fixed_stack_evidence(), reserve_bytes=3 * 1024 * 1024)

    with pytest.raises(stereo_control.StereoThreadBudgetError, match="stack reserve") as captured:
        stereo_control.create_attested_stereo_executor(
            worker_count=2,
            thread_name_prefix="quality-png-oversized-test",
            stack_adapter=oversized_stack,
        )

    assert captured.value.started_threads == captured.value.joined_threads == 2
    assert not [
        thread.name
        for thread in threading.enumerate()
        if thread.name.startswith("quality-png-oversized-test")
    ]


@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, RuntimeError])
def test_attested_executor_joins_workers_when_priming_result_is_interrupted(
    monkeypatch: pytest.MonkeyPatch, failure_type: type[BaseException]
) -> None:
    real_executor = stereo_control.ThreadPoolExecutor
    real_result = stereo_control.Future.result
    created = []

    def capture_executor(**kwargs):
        executor = real_executor(**kwargs)
        created.append(executor)
        return executor

    def interrupt_result(_future, timeout=None):
        monkeypatch.setattr(stereo_control.Future, "result", real_result)
        raise failure_type("injected priming interruption")

    monkeypatch.setattr(stereo_control, "ThreadPoolExecutor", capture_executor)
    monkeypatch.setattr(stereo_control.Future, "result", interrupt_result)
    try:
        with pytest.raises(failure_type, match="injected priming interruption"):
            stereo_control.create_attested_stereo_executor(
                worker_count=2,
                thread_name_prefix="quality-png-interrupted-test",
                stack_adapter=_fixed_stack_evidence,
            )

        assert not [
            thread.name
            for thread in threading.enumerate()
            if thread.name.startswith("quality-png-interrupted-test")
        ]
    finally:
        for executor in created:
            executor.shutdown(wait=True, cancel_futures=True)


@pytest.mark.parametrize("failure", ["setting", "restoration"])
def test_stack_setting_failures_never_release_parked_work(failure: str) -> None:
    setting = 0
    calls = 0
    ran: list[str] = []

    def stack_size(value: int | None = None) -> int:
        nonlocal setting, calls
        if value is None:
            return setting
        calls += 1
        if (failure == "setting" and value == 512 * 1024) or (
            failure == "restoration" and value == 0
        ):
            raise RuntimeError(f"injected {failure}")
        previous = setting
        setting = value
        return previous

    with pytest.raises(stereo_control.StereoThreadBudgetError, match=failure):
        stereo_control.create_parked_stereo_workers(
            pending_render_count=1,
            effective_workers=1,
            worker=lambda role, _index: ran.append(role),
            stack_adapter=_fixed_stack_evidence,
            stack_size_api=stack_size,
        )

    assert ran == []
    assert calls >= 2


def test_stack_creation_lock_serializes_concurrent_jobs() -> None:
    intervals: list[tuple[str, str, float]] = []
    interval_lock = threading.Lock()

    def run_job(label: str) -> None:
        def stack_size(value: int | None = None) -> int:
            with interval_lock:
                intervals.append(
                    (label, "get" if value is None else f"set:{value}", time.perf_counter())
                )
            if value == 512 * 1024:
                time.sleep(0.03)
            return 0

        group = stereo_control.create_parked_stereo_workers(
            pending_render_count=1,
            effective_workers=1,
            worker=lambda _role, _index: None,
            stack_adapter=_fixed_stack_evidence,
            stack_size_api=stack_size,
        )
        group.cancel_and_join()

    threads = [threading.Thread(target=run_job, args=(label,)) for label in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    labels = [label for label, operation, _time in intervals if operation == "get"]
    assert labels in (["a", "b"], ["b", "a"])
    first, second = labels
    first_restore = next(
        timestamp
        for label, operation, timestamp in intervals
        if label == first and operation == "set:0"
    )
    second_get = next(
        timestamp
        for label, operation, timestamp in intervals
        if label == second and operation == "get"
    )
    assert first_restore <= second_get


def test_four_pipeline_queues_share_the_selected_capacity() -> None:
    queues = stereo_control.StereoPipelineQueues(7)

    assert queues.capacities == (7, 7, 7, 7)


@pytest.mark.parametrize("workers", [1, 4, 16])
def test_live_worker_handles_match_feeder_decoders_and_writers(workers: int) -> None:
    group = stereo_control.create_parked_stereo_workers(
        pending_render_count=100,
        effective_workers=workers,
        worker=lambda _role, _index: None,
    )
    try:
        assert group.live_thread_count == 1 + 2 * workers
        assert len(group.stats.stack_evidence) == 1 + 2 * workers
    finally:
        group.cancel_and_join()
    assert group.live_thread_count == 0


@pytest.mark.parametrize("workers", [1, 4, 16])
def test_streaming_pipeline_forces_bounded_decode_render_write_overlap(workers: int) -> None:
    provider = _provider(96)
    plan = stereo_control.audit_stereo_provider(
        provider,
        lambda _item: stereo_control.StereoAction.RENDER,
    )

    def delayed(value):
        time.sleep(0.0005)
        return value

    stats = stereo_control.run_streaming_stereo_pipeline(
        provider=provider,
        plan=plan,
        capacity=2 * workers,
        effective_workers=workers,
        decode=delayed,
        render=delayed,
        write=lambda value: delayed(value),
    )

    assert (stats.decoded, stats.rendered, stats.written) == (96, 96, 96)
    assert stats.maximum_live_items <= 2 * workers
    assert stats.queue_capacities == (2 * workers,) * 4
    assert stats.actual_thread_count == 1 + 2 * workers
    assert stats.configured_workers == workers
    assert stats.feeder_thread_count == 1
    assert stats.decoder_thread_count == workers
    assert stats.writer_thread_count == workers
    assert stats.calculated_thread_bytes == (
        stats.actual_thread_count * stereo_control.STEREO_IO_THREAD_BYTES
    )
    assert stats.previous_stack_bytes == stats.restored_stack_bytes
    assert len(stats.stack_granularity_bytes) == stats.actual_thread_count
    assert len(stats.reported_stack_reserve_bytes) == stats.actual_thread_count
    assert stats.requested_stack_bytes == 512 * 1024
    assert stats.parked_committed_delta_bytes <= (
        stats.actual_thread_count * stereo_control.STEREO_IO_THREAD_BYTES
    )
    assert stats.decode_render_overlap
    assert stats.render_write_overlap
    assert stats.work_item_copies == 0
    assert stats.pipeline_wall_seconds > 0.0
    assert stats.decode_busy_seconds > 0.0
    assert stats.render_busy_seconds > 0.0
    assert stats.write_busy_seconds > 0.0


def test_streaming_pipeline_can_serialize_writes_without_reducing_decoders() -> None:
    provider = _provider(24)
    plan = stereo_control.audit_stereo_provider(
        provider,
        lambda _item: stereo_control.StereoAction.RENDER,
    )

    stats = stereo_control.run_streaming_stereo_pipeline(
        provider=provider,
        plan=plan,
        capacity=3,
        effective_workers=3,
        writer_workers=1,
        decode=lambda item: item,
        render=lambda item: item,
        write=lambda _item: None,
    )

    assert (stats.decoded, stats.rendered, stats.written) == (24, 24, 24)
    assert stats.actual_thread_count == 5
    assert stats.decoder_thread_count == 3
    assert stats.writer_thread_count == 1
    assert stats.calculated_thread_bytes == 10 * 1024 * 1024
    assert stats.live_threads_after_join == 0


def test_streaming_pipeline_can_restore_source_order_before_rendering() -> None:
    """A slow first decode must not let later frames cross a stream sink."""

    provider = _provider(6)
    plan = stereo_control.audit_stereo_provider(
        provider,
        lambda _item: stereo_control.StereoAction.RENDER,
    )
    later_decode_completed = threading.Event()
    rendered: list[int] = []
    written: list[int] = []

    def decode(item: stereo_control.StereoProviderItem):
        if item.source_index == 0:
            assert later_decode_completed.wait(timeout=5.0)
        elif item.source_index == 1:
            later_decode_completed.set()
        return item

    def render(item: stereo_control.StereoProviderItem):
        rendered.append(item.source_index)
        return item

    def write(item: stereo_control.StereoProviderItem) -> None:
        written.append(item.source_index)

    stats = stereo_control.run_streaming_stereo_pipeline(
        provider=provider,
        plan=plan,
        capacity=3,
        effective_workers=3,
        writer_workers=1,
        decode=decode,
        render=render,
        write=write,
        ordered_render=True,
    )

    assert rendered == list(range(6))
    assert written == list(range(6))
    assert stats.maximum_live_items <= 3


def test_capacity_permit_is_released_only_after_large_payload_references_are_gone() -> None:
    provider = _provider(2)
    plan = stereo_control.audit_stereo_provider(
        provider,
        lambda _item: stereo_control.StereoAction.RENDER,
    )
    decoded_refs: list[weakref.ReferenceType[_Payload]] = []
    rendered_refs: list[weakref.ReferenceType[_Payload]] = []

    def decode(item: stereo_control.StereoProviderItem) -> _Payload:
        if item.source_index == 1:
            gc.collect()
            assert decoded_refs[0]() is None
            assert rendered_refs[0]() is None
        payload = _Payload(item.source_index)
        decoded_refs.append(weakref.ref(payload))
        return payload

    def render(payload: _Payload) -> _Payload:
        rendered = _Payload(payload.index)
        rendered_refs.append(weakref.ref(rendered))
        return rendered

    stereo_control.run_streaming_stereo_pipeline(
        provider=provider,
        plan=plan,
        capacity=1,
        effective_workers=1,
        decode=decode,
        render=render,
        write=lambda _payload: None,
    )


def test_feeder_acquires_capacity_before_requesting_the_next_provider_item() -> None:
    base = _provider(2)
    replay_number = 0
    second_feeder_item_requested = threading.Event()

    def replay():
        nonlocal replay_number
        replay_number += 1
        current = replay_number
        for item in base.replay():
            if current == 4 and item.source_index == 1:
                second_feeder_item_requested.set()
            yield item

    provider = stereo_control.ReplayableStereoProvider(base.generation, replay)
    plan = stereo_control.audit_stereo_provider(
        provider,
        lambda _item: stereo_control.StereoAction.RENDER,
    )
    writer_entered = threading.Event()
    release_writer = threading.Event()
    result: list[BaseException | None] = []

    def write(_payload: object) -> None:
        writer_entered.set()
        assert release_writer.wait(timeout=5.0)

    def run() -> None:
        try:
            stereo_control.run_streaming_stereo_pipeline(
                provider=provider,
                plan=plan,
                capacity=1,
                effective_workers=1,
                decode=lambda item: item,
                render=lambda item: item,
                write=write,
            )
        except BaseException as error:  # noqa: BLE001 - asserted below
            result.append(error)
        else:
            result.append(None)

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    assert writer_entered.wait(timeout=5.0)
    assert not second_feeder_item_requested.wait(timeout=0.2)
    release_writer.set()
    runner.join(timeout=5.0)
    assert not runner.is_alive()
    assert result == [None]


def test_completion_acknowledgement_does_not_retain_a_finished_work_item() -> None:
    generation = _generation(2)
    replay_number = 0
    first_feeder_item: list[weakref.ReferenceType[stereo_control.StereoProviderItem]] = []

    def replay():
        nonlocal replay_number
        replay_number += 1
        current = replay_number
        for index in range(2):
            if current == 4 and index == 1:
                gc.collect()
                assert first_feeder_item[0]() is None
            item = _item(generation, index)
            if current == 4 and index == 0:
                first_feeder_item.append(weakref.ref(item))
            yield item
            item = None

    provider = stereo_control.ReplayableStereoProvider(generation, replay)
    plan = stereo_control.audit_stereo_provider(
        provider,
        lambda _item: stereo_control.StereoAction.RENDER,
    )

    stereo_control.run_streaming_stereo_pipeline(
        provider=provider,
        plan=plan,
        capacity=1,
        effective_workers=1,
        decode=lambda item: item.source_index,
        render=lambda item: item,
        write=lambda _item: None,
    )


class _Payload:
    __slots__ = ("data", "index", "__weakref__")

    def __init__(self, index: int) -> None:
        self.index = index
        self.data = bytearray(1024 * 1024)


def test_downstream_mutation_failure_cancels_workers_before_gate_release() -> None:
    provider = _provider(4)
    plan = stereo_control.audit_stereo_provider(
        provider,
        lambda _item: stereo_control.StereoAction.RENDER,
    )
    baseline = stereo_control.current_process_thread_count()
    touched: list[str] = []

    with pytest.raises(RuntimeError, match="injected mutation"):
        stereo_control.run_streaming_stereo_pipeline(
            provider=provider,
            plan=plan,
            capacity=2,
            effective_workers=1,
            decode=lambda item: touched.append("decode") or item,
            render=lambda item: touched.append("render") or item,
            write=lambda item: touched.append("write") or item,
            mutation=lambda: (_ for _ in ()).throw(RuntimeError("injected mutation")),
        )

    assert touched == []
    assert stereo_control.current_process_thread_count() == baseline


def test_parked_workers_get_a_final_provider_replay_before_mutation() -> None:
    base = _provider(2)
    replay_count = 0

    def replay():
        nonlocal replay_count
        replay_count += 1
        if replay_count == 3:
            raise stereo_control.StereoControlStateError("changed while workers parked")
        yield from base.replay()

    provider = stereo_control.ReplayableStereoProvider(base.generation, replay)
    plan = stereo_control.audit_stereo_provider(
        provider,
        lambda _item: stereo_control.StereoAction.RENDER,
    )
    mutations: list[str] = []
    baseline = stereo_control.current_process_thread_count()

    with pytest.raises(
        stereo_control.StereoProviderChangedError,
        match="changed while workers parked",
    ):
        stereo_control.run_streaming_stereo_pipeline(
            provider=provider,
            plan=plan,
            capacity=1,
            effective_workers=1,
            decode=lambda item: item,
            render=lambda item: item,
            write=lambda _item: None,
            mutation=lambda: mutations.append("started"),
        )

    assert mutations == []
    assert stereo_control.current_process_thread_count() == baseline


def test_downstream_mutation_input_change_is_typed_after_the_boundary() -> None:
    provider = _provider(2)
    plan = stereo_control.audit_stereo_provider(
        provider,
        lambda _item: stereo_control.StereoAction.RENDER,
    )

    with pytest.raises(
        stereo_control.StereoStageInputChangedError,
        match="injected mutation replay change",
    ):
        stereo_control.run_streaming_stereo_pipeline(
            provider=provider,
            plan=plan,
            capacity=1,
            effective_workers=1,
            decode=lambda item: item,
            render=lambda item: item,
            write=lambda _item: None,
            mutation=lambda: (_ for _ in ()).throw(
                stereo_control.StereoInputIdentityMismatch("injected mutation replay change")
            ),
        )


@pytest.mark.parametrize("failure_stage", ["feeder", "decode", "write"])
def test_streaming_pipeline_propagates_first_worker_error_without_deadlock(
    failure_stage: str,
) -> None:
    base = _provider(6)
    replay_count = 0

    def replay():
        nonlocal replay_count
        replay_count += 1
        if failure_stage == "feeder" and replay_count == 4:
            raise stereo_control.StereoControlStateError("injected feeder change")
        yield from base.replay()

    provider = stereo_control.ReplayableStereoProvider(base.generation, replay)
    plan = stereo_control.audit_stereo_provider(
        provider,
        lambda _item: stereo_control.StereoAction.RENDER,
    )
    result: list[BaseException | None] = []

    def run() -> None:
        try:
            stereo_control.run_streaming_stereo_pipeline(
                provider=provider,
                plan=plan,
                capacity=1,
                effective_workers=1,
                decode=lambda item: (
                    (_ for _ in ()).throw(RuntimeError("injected decode failure"))
                    if failure_stage == "decode" and item.source_index == 0
                    else item
                ),
                render=lambda item: item,
                write=lambda item: (
                    (_ for _ in ()).throw(RuntimeError("injected write failure"))
                    if failure_stage == "write" and item.source_index == 0
                    else None
                ),
                mutation=lambda: None,
            )
        except BaseException as error:  # noqa: BLE001 - asserted below
            result.append(error)
        else:
            result.append(None)

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    runner.join(timeout=5.0)

    assert not runner.is_alive(), f"{failure_stage} failure deadlocked the pipeline"
    assert len(result) == 1
    if failure_stage == "feeder":
        assert isinstance(result[0], stereo_control.StereoStageInputChangedError)
        assert "injected feeder change" in str(result[0])
    else:
        assert isinstance(result[0], RuntimeError)
        assert f"injected {failure_stage} failure" in str(result[0])


def test_late_feeder_error_after_last_render_cannot_return_success() -> None:
    base = _provider(4)
    replay_count = 0

    def replay():
        nonlocal replay_count
        replay_count += 1
        current = replay_count
        yield from base.replay()
        if current == 4:
            time.sleep(0.1)
            raise stereo_control.StereoControlStateError("late feeder corruption")

    provider = stereo_control.ReplayableStereoProvider(base.generation, replay)
    plan = stereo_control.audit_stereo_provider(
        provider,
        lambda item: (
            stereo_control.StereoAction.RENDER
            if item.source_index == 0
            else stereo_control.StereoAction.REUSE
        ),
    )
    baseline = stereo_control.current_process_thread_count()

    with pytest.raises(
        stereo_control.StereoStageInputChangedError,
        match="late feeder corruption",
    ):
        stereo_control.run_streaming_stereo_pipeline(
            provider=provider,
            plan=plan,
            capacity=1,
            effective_workers=1,
            decode=lambda item: bytearray(1024 * 1024),
            render=lambda payload: payload,
            write=lambda _payload: None,
            mutation=lambda: None,
        )

    assert stereo_control.current_process_thread_count() == baseline


def test_decoder_input_identity_mismatch_is_fatal_after_mutation() -> None:
    provider = _provider(2)
    plan = stereo_control.audit_stereo_provider(
        provider,
        lambda _item: stereo_control.StereoAction.RENDER,
    )

    with pytest.raises(
        stereo_control.StereoStageInputChangedError,
        match="injected input mismatch",
    ):
        stereo_control.run_streaming_stereo_pipeline(
            provider=provider,
            plan=plan,
            capacity=1,
            effective_workers=1,
            decode=lambda _item: (_ for _ in ()).throw(
                stereo_control.StereoInputIdentityMismatch("injected input mismatch")
            ),
            render=lambda item: item,
            write=lambda _item: None,
            mutation=lambda: None,
        )


def test_migration_replay_starts_only_after_pipeline_thread_teardown() -> None:
    provider = _provider(20)
    plan = stereo_control.audit_stereo_provider(
        provider,
        lambda item: (
            stereo_control.StereoAction.RENDER
            if item.source_index % 2 == 0
            else stereo_control.StereoAction.MIGRATE
        ),
    )
    migrated: list[int] = []

    def migrate(item: stereo_control.StereoProviderItem) -> None:
        migrated.append(item.source_index)

    result = stereo_control.execute_stereo_control_plan(
        provider=provider,
        plan=plan,
        capacity=2,
        effective_workers=1,
        decode=lambda item: item,
        render=lambda item: item,
        write=lambda _item: None,
        migrate=migrate,
    )

    assert result.migrated == 10
    assert migrated == list(range(1, 20, 2))
    assert result.pipeline.live_threads_after_join == 0
    assert result.thread_count_during_migration > 0


def test_process_thread_count_is_diagnostic_not_pipeline_leak_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(2)
    plan = stereo_control.audit_stereo_provider(
        provider,
        lambda item: (
            stereo_control.StereoAction.RENDER
            if item.source_index == 0
            else stereo_control.StereoAction.MIGRATE
        ),
    )
    monkeypatch.setattr(
        stereo_control,
        "current_process_thread_count",
        lambda: 50,
    )

    result = stereo_control.execute_stereo_control_plan(
        provider=provider,
        plan=plan,
        capacity=1,
        effective_workers=1,
        decode=lambda item: item,
        render=lambda item: item,
        write=lambda _item: None,
        migrate=lambda _item: None,
    )

    assert result.pipeline.live_threads_after_join == 0
    assert result.thread_count_during_migration == 50


@pytest.mark.skipif(sys.platform != "win32", reason="Windows reserve adapter fixture")
def test_windows_stack_adapter_reports_current_reserved_bounds() -> None:
    evidence = stereo_control.attest_current_thread_stack()

    assert evidence.adapter == "windows-get-stack-limits-v1"
    assert evidence.reserve_bytes > 0
    assert evidence.granularity_bytes > 0
    assert hashlib.sha256(str(evidence).encode()).digest()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows reserved-thread fixture")
def test_windows_worker_backend_uses_512_kib_as_the_actual_reserve() -> None:
    group = stereo_control.create_parked_stereo_workers(
        pending_render_count=1,
        effective_workers=1,
        worker=lambda _role, _index: None,
    )
    try:
        assert {item.reserve_bytes for item in group.stats.stack_evidence} == {512 * 1024}
        assert group.live_thread_count == 3
    finally:
        group.cancel_and_join()
