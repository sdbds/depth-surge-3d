"""Tests for bounded CPU frame-stage worker selection."""

from __future__ import annotations

import threading
import time

import pytest

from src.depth_surge_3d.processing.frames import frame_stage_parallelism as parallelism
from src.depth_surge_3d.processing.frames.frame_stage_parallelism import (
    calculate_frame_stage_workers,
    max_png_frame_pair_pixels,
    png_headers_match,
    run_ordered_frame_tasks,
    uniform_png_frame_pair_header,
)
from src.depth_surge_3d.utils.imaging.png_header import PngHeader


@pytest.mark.parametrize(
    ("cpu_count", "frame_count", "item_bytes", "expected"),
    [
        (16, 100, 1, 8),
        (6, 100, 1, 4),
        (16, 3, 1, 3),
        (1, 100, 1, 1),
        (None, 100, 1, 1),
        (16, 100, 600_000_000, 1),
        (16, 100, 2_000_000_000, 1),
    ],
)
def test_worker_count_obeys_cpu_frame_and_memory_caps(
    monkeypatch: pytest.MonkeyPatch,
    cpu_count: int | None,
    frame_count: int,
    item_bytes: int,
    expected: int,
) -> None:
    monkeypatch.setattr("os.cpu_count", lambda: cpu_count)

    assert calculate_frame_stage_workers(frame_count, item_bytes) == expected


@pytest.mark.parametrize(
    ("frame_count", "item_bytes"),
    [(0, 1), (-1, 1), (1, 0), (1, -1)],
)
def test_worker_count_rejects_non_positive_work(frame_count: int, item_bytes: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        calculate_frame_stage_workers(frame_count, item_bytes)


def test_worker_count_honors_explicit_policy_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("os.cpu_count", lambda: 64)

    assert (
        calculate_frame_stage_workers(
            100,
            100,
            max_workers=3,
            memory_budget_bytes=250,
        )
        == 2
    )


def test_ordered_tasks_report_on_the_caller_thread_despite_out_of_order_completion():
    caller_thread = threading.get_ident()
    reported = []

    def worker(index: int) -> str:
        time.sleep((3 - index) * 0.02)
        return f"result-{index}"

    run_ordered_frame_tasks(
        range(3),
        worker,
        worker_count=3,
        on_ordered_result=lambda index, result: reported.append(
            (index, result, threading.get_ident())
        ),
    )

    assert reported == [
        (0, "result-0", caller_thread),
        (1, "result-1", caller_thread),
        (2, "result-2", caller_thread),
    ]


def test_later_failure_is_detected_before_slow_earlier_work_releases():
    first_started = threading.Event()
    failure_raised = threading.Event()
    release_first = threading.Event()
    executed = []
    outcome = {}

    def worker(index: int) -> int:
        executed.append(index)
        if index == 0:
            first_started.set()
            assert release_first.wait(5)
            return index
        if index == 1:
            assert first_started.wait(2)
            failure_raised.set()
            raise RuntimeError("later frame failed")
        return index

    def run_tasks() -> None:
        try:
            run_ordered_frame_tasks(range(20), worker, worker_count=2)
        except Exception as error:  # noqa: BLE001 - assertion captures the worker error
            outcome["error"] = error

    thread = threading.Thread(target=run_tasks)
    thread.start()
    try:
        assert failure_raised.wait(2)
        time.sleep(0.05)
        assert thread.is_alive()
        assert len(executed) == 2
        assert set(executed) == {0, 1}
    finally:
        release_first.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert isinstance(outcome["error"], RuntimeError)
    assert str(outcome["error"]) == "later frame failed"


def test_uniform_png_pair_header_scans_pairs_concurrently(monkeypatch):
    expected = PngHeader(width=64, height=48, bit_depth=8, color_type=2, channels=3)
    active = 0
    max_active = 0
    lock = threading.Lock()

    def delayed_header(_path):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return expected

    monkeypatch.setattr(parallelism, "read_png_header", delayed_header)
    monkeypatch.setattr("os.cpu_count", lambda: 4)

    actual = uniform_png_frame_pair_header(
        [f"left-{index}.png" for index in range(4)],
        [f"right-{index}.png" for index in range(4)],
    )

    assert actual == expected
    assert max_active >= 2


def test_max_png_pair_pixels_checks_every_source_header(monkeypatch):
    headers = {
        "left-0.png": PngHeader(64, 48, 8, 2, 3),
        "right-0.png": PngHeader(64, 48, 8, 2, 3),
        "left-1.png": PngHeader(200, 100, 8, 2, 3),
        "right-1.png": PngHeader(80, 60, 8, 2, 3),
    }
    monkeypatch.setattr(parallelism, "read_png_header", headers.get)

    assert (
        max_png_frame_pair_pixels(
            ["left-0.png", "left-1.png"],
            ["right-0.png", "right-1.png"],
        )
        == 20_000
    )


def test_png_header_matching_scans_outputs_concurrently(monkeypatch):
    active = 0
    max_active = 0
    lock = threading.Lock()

    def delayed_match(_path, *, shape, bit_depth):
        nonlocal active, max_active
        assert shape == (48, 64, 3)
        assert bit_depth == 8
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return True

    monkeypatch.setattr(parallelism, "png_header_matches", delayed_match)
    monkeypatch.setattr("os.cpu_count", lambda: 4)

    assert png_headers_match(
        [f"frame-{index}.png" for index in range(4)],
        shape=(48, 64, 3),
        bit_depth=8,
    )
    assert max_active >= 2
