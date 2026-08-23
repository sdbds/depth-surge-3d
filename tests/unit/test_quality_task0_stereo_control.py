"""Task 0 contracts for the replayable stereo control plane and parked workers."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import threading
import time

import pytest

from scripts.quality_task0 import stereo_control


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


def test_quality_capacity_is_exactly_one_and_p_zero_creates_no_control() -> None:
    assert stereo_control.quality_pipeline_capacity(0) == (0, 0, 0)
    assert stereo_control.quality_pipeline_capacity(1) == (1, 1, 6 * 1024 * 1024)
    assert stereo_control.quality_pipeline_capacity(1_000_000) == (
        1,
        1,
        6 * 1024 * 1024,
    )


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
def test_live_os_thread_delta_matches_feeder_decoders_and_writers(workers: int) -> None:
    before = stereo_control.current_process_thread_count()
    group = stereo_control.create_parked_stereo_workers(
        pending_render_count=100,
        effective_workers=workers,
        worker=lambda _role, _index: None,
    )
    try:
        assert stereo_control.current_process_thread_count() - before == 1 + 2 * workers
        assert group.live_thread_count == 1 + 2 * workers
        assert len(group.stats.stack_evidence) == 1 + 2 * workers
    finally:
        group.cancel_and_join()
    assert stereo_control.current_process_thread_count() == before


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
    assert stats.requested_stack_bytes == 512 * 1024
    assert stats.parked_committed_delta_bytes <= (
        stats.actual_thread_count * stereo_control.STEREO_IO_THREAD_BYTES
    )
    assert stats.decode_render_overlap
    assert stats.render_write_overlap
    assert stats.work_item_copies == 0


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
    baseline = stereo_control.current_process_thread_count()
    migrated: list[int] = []

    def migrate(item: stereo_control.StereoProviderItem) -> None:
        assert stereo_control.current_process_thread_count() == baseline
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
    assert result.thread_count_during_migration == baseline


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
