"""Dynamic host RAM budget and streaming stereo pipeline control tests."""

from __future__ import annotations

import pytest

from src.depth_surge_3d.processing.frames.stereo_control import (
    MIN_STEREO_HOST_BUDGET_BYTES,
    available_host_memory_bytes,
    dynamic_host_budget_bytes,
    run_streaming_stereo_pipeline,
    select_fast_pipeline_capacity,
    select_quality_pipeline_capacity,
)


def test_dynamic_host_budget_unblocks_4k_quality() -> None:
    budget = dynamic_host_budget_bytes()
    assert budget >= MIN_STEREO_HOST_BUDGET_BYTES
    assert available_host_memory_bytes() is None or available_host_memory_bytes() > 0
    assert select_fast_pipeline_capacity((2160, 3840)) >= 1
    assert select_quality_pipeline_capacity((2160, 3840)) >= 1


def test_streaming_pipeline_preserves_order_and_bounds_in_flight() -> None:
    written: list[int] = []
    completed: list[int] = []

    stats = run_streaming_stereo_pipeline(
        range(12),
        decode_fn=lambda i: i * 10,
        render_fn=lambda i, val: val + 1,
        write_fn=lambda i, out: written.append(out),
        capacity=3,
        decode_workers=2,
        write_workers=2,
        ordered_writes=True,
        on_completed=lambda item, n: completed.append(n),
    )

    assert written == [i * 10 + 1 for i in range(12)]
    assert completed == list(range(1, 13))
    assert stats.frame_count == 12
    assert stats.render_count == 12


def test_streaming_pipeline_propagates_cancellation_and_errors() -> None:
    with pytest.raises(InterruptedError):
        run_streaming_stereo_pipeline(
            range(8),
            decode_fn=lambda i: i,
            render_fn=lambda i, d: d,
            write_fn=lambda i, r: None,
            capacity=2,
            decode_workers=2,
            write_workers=2,
            is_cancelled=lambda: True,
        )

    def failing_render(i: int, d: int) -> int:
        if i == 2:
            raise ValueError("boom")
        return d

    with pytest.raises(ValueError, match="boom"):
        run_streaming_stereo_pipeline(
            range(6),
            decode_fn=lambda i: i,
            render_fn=failing_render,
            write_fn=lambda i, r: None,
            capacity=2,
            decode_workers=2,
            write_workers=2,
        )


def test_four_stage_pipeline_preserves_order_and_propagates_postprocess_errors() -> None:
    import time

    written: list[int] = []
    completed: list[int] = []

    def slow_post(item: int, rendered: int) -> int:
        # Earlier items sleep slightly longer to verify post_to_write restores exact order.
        time.sleep(0.01 if item % 2 == 0 else 0.001)
        return rendered * 2

    stats = run_streaming_stereo_pipeline(
        range(10),
        decode_fn=lambda i: i + 1,
        render_fn=lambda i, d: d * 10,
        postprocess_fn=slow_post,
        postprocess_workers=4,
        write_fn=lambda i, fin: written.append(fin),
        capacity=6,
        decode_workers=4,
        write_workers=1,
        ordered_writes=True,
        on_completed=lambda _item, n: completed.append(n),
    )

    assert written == [(i + 1) * 20 for i in range(10)]
    assert completed == list(range(1, 11))
    assert stats.peak_in_flight <= 6

    def failing_post(i: int, r: int) -> int:
        if i == 3:
            raise ValueError("repair failed")
        return r

    with pytest.raises(ValueError, match="repair failed"):
        run_streaming_stereo_pipeline(
            range(8),
            decode_fn=lambda i: i,
            render_fn=lambda i, d: d,
            postprocess_fn=failing_post,
            postprocess_workers=3,
            write_fn=lambda i, f: None,
            capacity=4,
            decode_workers=2,
            write_workers=1,
            ordered_writes=True,
        )
