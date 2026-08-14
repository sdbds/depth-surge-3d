"""Bounded worker selection and execution for independent frame-stage work."""

from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable, Generic, Iterable, Sequence, TypeVar

from ...utils.imaging.png_header import PngHeader, png_header_matches, read_png_header


MAX_FRAME_STAGE_WORKERS = 8
FRAME_STAGE_MEMORY_BUDGET_BYTES = 1024**3
FRAME_STAGE_CPU_RESERVE = 2
_PNG_PAIR_HEADER_ESTIMATE_BYTES = 1024

_Item = TypeVar("_Item")
_Result = TypeVar("_Result")


def calculate_frame_stage_workers(
    frame_count: int,
    estimated_bytes_per_item: int,
    *,
    max_workers: int = MAX_FRAME_STAGE_WORKERS,
    memory_budget_bytes: int = FRAME_STAGE_MEMORY_BUDGET_BYTES,
) -> int:
    """Choose a worker count bounded by work, CPU reserve, and memory."""

    if frame_count <= 0 or estimated_bytes_per_item <= 0:
        raise ValueError("Frame work and item memory must be positive")

    logical_cpus = os.cpu_count() or 1
    cpu_limit = max(1, logical_cpus - FRAME_STAGE_CPU_RESERVE)
    memory_limit = max(1, memory_budget_bytes // estimated_bytes_per_item)
    return max(1, min(frame_count, max_workers, cpu_limit, memory_limit))


def run_ordered_frame_tasks(
    items: Iterable[_Item],
    worker: Callable[[_Item], _Result],
    *,
    worker_count: int,
    on_ordered_result: Callable[[int, _Result], None] | None = None,
) -> None:
    """Run bounded work, detect failures promptly, and report in input order."""

    if worker_count <= 0:
        raise ValueError("Worker count must be positive")

    _OrderedFrameTaskRunner(
        items,
        worker,
        worker_count=worker_count,
        on_ordered_result=on_ordered_result,
    ).run()


class _InvalidPngHeader(ValueError):
    pass


def _header_scan_ranges(frame_count: int, worker_count: int) -> list[range]:
    chunk_size = (frame_count + worker_count - 1) // worker_count
    return [
        range(start, min(start + chunk_size, frame_count))
        for start in range(0, frame_count, chunk_size)
    ]


def uniform_png_frame_pair_header(
    left_files: Sequence[Path | str], right_files: Sequence[Path | str]
) -> PngHeader | None:
    """Validate every source header concurrently and return their shared shape."""

    if not left_files or len(left_files) != len(right_files):
        return None
    expected = read_png_header(left_files[0])
    if expected is None:
        return None
    worker_count = calculate_frame_stage_workers(len(left_files), _PNG_PAIR_HEADER_ESTIMATE_BYTES)

    def validate_range(indices: range) -> None:
        for index in indices:
            left_header = expected if index == 0 else read_png_header(left_files[index])
            right_header = read_png_header(right_files[index])
            if left_header != expected or right_header != expected:
                raise _InvalidPngHeader

    try:
        run_ordered_frame_tasks(
            _header_scan_ranges(len(left_files), worker_count),
            validate_range,
            worker_count=worker_count,
        )
    except _InvalidPngHeader:
        return None
    return expected


def max_png_frame_pair_pixels(
    left_files: Sequence[Path | str], right_files: Sequence[Path | str]
) -> int | None:
    """Read every source header concurrently and return the largest eye image."""

    if not left_files or len(left_files) != len(right_files):
        return None
    largest = 0
    worker_count = calculate_frame_stage_workers(len(left_files), _PNG_PAIR_HEADER_ESTIMATE_BYTES)

    def inspect_range(indices: range) -> int:
        local_largest = 0
        for index in indices:
            left_header = read_png_header(left_files[index])
            right_header = read_png_header(right_files[index])
            if left_header is None or right_header is None:
                raise _InvalidPngHeader
            local_largest = max(
                local_largest,
                left_header.width * left_header.height,
                right_header.width * right_header.height,
            )
        return local_largest

    def record_maximum(_index: int, pixels: int) -> None:
        nonlocal largest
        largest = max(largest, pixels)

    try:
        run_ordered_frame_tasks(
            _header_scan_ranges(len(left_files), worker_count),
            inspect_range,
            worker_count=worker_count,
            on_ordered_result=record_maximum,
        )
    except _InvalidPngHeader:
        return None
    return largest


def png_headers_match(files: Sequence[Path | str], *, shape: Sequence[int], bit_depth: int) -> bool:
    """Validate a generated PNG collection concurrently without decoding pixels."""

    if not files:
        return False
    worker_count = calculate_frame_stage_workers(len(files), _PNG_PAIR_HEADER_ESTIMATE_BYTES)

    def validate_range(indices: range) -> None:
        for index in indices:
            if not png_header_matches(files[index], shape=shape, bit_depth=bit_depth):
                raise _InvalidPngHeader

    try:
        run_ordered_frame_tasks(
            _header_scan_ranges(len(files), worker_count),
            validate_range,
            worker_count=worker_count,
        )
    except _InvalidPngHeader:
        return False
    return True


class _OrderedFrameTaskRunner(Generic[_Item, _Result]):
    def __init__(
        self,
        items: Iterable[_Item],
        worker: Callable[[_Item], _Result],
        *,
        worker_count: int,
        on_ordered_result: Callable[[int, _Result], None] | None,
    ) -> None:
        self._items = iter(items)
        self._worker = worker
        self._worker_count = worker_count
        self._on_ordered_result = on_ordered_result
        self._executor = ThreadPoolExecutor(max_workers=worker_count)
        self._in_flight: dict[Future[_Result], int] = {}
        self._ready: dict[int, _Result] = {}
        self._next_submit_index = 0
        self._next_report_index = 0

    def run(self) -> None:
        try:
            self._fill_worker_slots()
            while self._in_flight:
                self._collect_completed()
                self._fill_worker_slots()
                self._report_ready_in_order()
        except Exception:
            self._cancel_and_wait()
            raise
        else:
            self._executor.shutdown(wait=True)

    def _submit_next(self) -> bool:
        try:
            item = next(self._items)
        except StopIteration:
            return False
        future = self._executor.submit(self._worker, item)
        self._in_flight[future] = self._next_submit_index
        self._next_submit_index += 1
        return True

    def _fill_worker_slots(self) -> None:
        while len(self._in_flight) < self._worker_count and self._submit_next():
            pass

    def _collect_completed(self) -> None:
        done, _ = wait(tuple(self._in_flight), return_when=FIRST_COMPLETED)
        completed_batch = []
        for future in done:
            index = self._in_flight.pop(future)
            completed_batch.append((index, future.result()))
        self._ready.update(completed_batch)

    def _report_ready_in_order(self) -> None:
        while self._next_report_index in self._ready:
            result = self._ready.pop(self._next_report_index)
            if self._on_ordered_result is not None:
                self._on_ordered_result(self._next_report_index, result)
            self._next_report_index += 1

    def _cancel_and_wait(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
