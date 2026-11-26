"""Dynamic RAM-aware streaming stereo pipeline controller."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
import ctypes
from dataclasses import dataclass
import os
import queue
import threading
import time
from typing import TypeVar


MIN_STEREO_HOST_BUDGET_BYTES = 1024 * 1024 * 1024  # 1 GiB
MAX_STEREO_HOST_BUDGET_BYTES = 8 * 1024 * 1024 * 1024  # 8 GiB
DEFAULT_STEREO_HOST_BUDGET_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB fallback
STEREO_CONTROL_OVERHEAD = 16 * 1024 * 1024
MAX_PIPELINE_CAPACITY = 16

TItem = TypeVar("TItem")
TDecoded = TypeVar("TDecoded")
TRendered = TypeVar("TRendered")


class StereoControlError(RuntimeError):
    """Base error for stereo pipeline control."""


class StereoPipelineCapacityError(StereoControlError):
    """Raised when a single stereo frame cannot fit in the available host memory budget."""


class StereoControlStateError(StereoControlError):
    """Raised when stereo stage inputs or metadata are inconsistent."""


@dataclass(frozen=True)
class StreamingStereoPipelineStats:
    """Summary metrics recorded by a streaming stereo render run."""

    frame_count: int
    render_count: int
    skip_count: int
    capacity: int
    decode_workers: int
    write_workers: int
    peak_in_flight: int
    elapsed_seconds: float


def available_host_memory_bytes() -> int | None:  # noqa: C901
    """Query currently available host RAM across Windows, Linux, and macOS."""

    if os.name == "nt":
        try:

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                avail = int(status.ullAvailPhys)
                return avail if avail > 0 else None
        except Exception:
            return None
        return None

    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb * 1024 if kb > 0 else None
    except OSError:
        pass

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        avail_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        if page_size > 0 and avail_pages > 0:
            return page_size * avail_pages
    except (AttributeError, OSError, ValueError):
        pass
    return None


def dynamic_host_budget_bytes(
    *,
    fraction: float = 0.35,
    minimum: int = MIN_STEREO_HOST_BUDGET_BYTES,
    maximum: int = MAX_STEREO_HOST_BUDGET_BYTES,
    fallback: int = DEFAULT_STEREO_HOST_BUDGET_BYTES,
) -> int:
    """Compute a dynamic host RAM budget scaled to available physical memory."""

    avail = available_host_memory_bytes()
    if avail is None:
        return fallback
    target = int(avail * fraction)
    return max(minimum, min(maximum, target))


def select_fast_pipeline_capacity(
    render_shape: tuple[int, int],
    *,
    host_budget_bytes: int | None = None,
    extra_per_frame_bytes: int = 0,
    max_capacity: int = MAX_PIPELINE_CAPACITY,
) -> int:
    """Choose in-flight queue capacity for Fast stereo rendering."""

    height, width = int(render_shape[0]), int(render_shape[1])
    pixels = max(1, height * width)
    per_slot = pixels * 24 + max(0, int(extra_per_frame_bytes)) + 2 * 1024 * 1024
    budget = host_budget_bytes if host_budget_bytes is not None else dynamic_host_budget_bytes()
    usable = max(per_slot, budget - STEREO_CONTROL_OVERHEAD)
    return max(1, min(max_capacity, usable // per_slot))


def select_quality_pipeline_capacity(
    render_shape: tuple[int, int],
    *,
    host_budget_bytes: int | None = None,
    extra_per_frame_bytes: int = 0,
    max_capacity: int = 8,
) -> int:
    """Choose in-flight queue capacity for Quality stereo rendering."""

    height, width = int(render_shape[0]), int(render_shape[1])
    pixels = max(1, height * width)
    per_slot = pixels * 96 + max(0, int(extra_per_frame_bytes)) + 16 * 1024 * 1024
    budget = host_budget_bytes if host_budget_bytes is not None else dynamic_host_budget_bytes()
    usable = budget - STEREO_CONTROL_OVERHEAD
    if usable < per_slot:
        return 1
    return max(1, min(max_capacity, usable // per_slot))


def run_streaming_stereo_pipeline(  # noqa: C901
    items: Iterable[TItem],
    *,
    decode_fn: Callable[[TItem], TDecoded],
    render_fn: Callable[[TItem, TDecoded], TRendered],
    write_fn: Callable[[TItem, TRendered], None],
    capacity: int,
    decode_workers: int,
    write_workers: int,
    ordered_writes: bool = False,
    on_completed: Callable[[TItem, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> StreamingStereoPipelineStats:
    """Execute bounded decode -> main-thread render -> async/ordered write pipeline."""

    started = time.perf_counter()
    work_list = list(items)
    total = len(work_list)
    if total == 0:
        return StreamingStereoPipelineStats(
            frame_count=0,
            render_count=0,
            skip_count=0,
            capacity=max(1, capacity),
            decode_workers=max(1, decode_workers),
            write_workers=max(1, write_workers),
            peak_in_flight=0,
            elapsed_seconds=0.0,
        )

    bounded_capacity = max(1, int(capacity))
    d_workers = max(1, min(int(decode_workers), total))
    w_workers = 1 if ordered_writes else max(1, min(int(write_workers), total))

    stop_event = threading.Event()
    in_flight_lock = threading.Lock()
    in_flight = 0
    peak_in_flight = 0
    completed_count = 0

    def check_cancel() -> None:
        if is_cancelled is not None and is_cancelled():
            stop_event.set()
            raise InterruptedError("Stereo processing was cancelled")

    def mark_enter() -> None:
        nonlocal in_flight, peak_in_flight
        with in_flight_lock:
            in_flight += 1
            if in_flight > peak_in_flight:
                peak_in_flight = in_flight

    def mark_exit() -> None:
        nonlocal in_flight
        with in_flight_lock:
            in_flight = max(0, in_flight - 1)

    # Single-threaded fast path when capacity == 1 and minimal workers
    if bounded_capacity == 1 and d_workers == 1 and w_workers == 1:
        for index, item in enumerate(work_list, 1):
            check_cancel()
            decoded = decode_fn(item)
            rendered = render_fn(item, decoded)
            del decoded
            write_fn(item, rendered)
            del rendered
            if on_completed is not None:
                on_completed(item, index)
        return StreamingStereoPipelineStats(
            frame_count=total,
            render_count=total,
            skip_count=0,
            capacity=1,
            decode_workers=1,
            write_workers=1,
            peak_in_flight=1,
            elapsed_seconds=time.perf_counter() - started,
        )

    decode_queue: queue.Queue[tuple[TItem, Future[TDecoded]] | None] = queue.Queue(
        maxsize=bounded_capacity
    )
    write_semaphore = threading.Semaphore(bounded_capacity)
    write_futures: list[tuple[TItem, Future[None]]] = []
    first_error: list[BaseException] = []

    def decode_task(item: TItem) -> TDecoded:
        if stop_event.is_set():
            raise InterruptedError("cancelled")
        return decode_fn(item)

    def write_task(item: TItem, rendered: TRendered) -> None:
        try:
            if stop_event.is_set():
                return
            write_fn(item, rendered)
        finally:
            del rendered
            mark_exit()
            write_semaphore.release()

    with (
        ThreadPoolExecutor(max_workers=d_workers, thread_name_prefix="stereo-dec") as dec_pool,
        ThreadPoolExecutor(max_workers=w_workers, thread_name_prefix="stereo-enc") as wr_pool,
    ):

        def feeder() -> None:
            try:
                for item in work_list:
                    if stop_event.is_set():
                        break
                    mark_enter()
                    fut = dec_pool.submit(decode_task, item)
                    decode_queue.put((item, fut))
            except BaseException as exc:
                first_error.append(exc)
                stop_event.set()
            finally:
                decode_queue.put(None)

        feeder_thread = threading.Thread(target=feeder, daemon=True)
        feeder_thread.start()

        try:
            while True:
                entry = decode_queue.get()
                if entry is None:
                    break
                item, dec_fut = entry
                check_cancel()
                decoded = dec_fut.result()
                rendered = render_fn(item, decoded)
                del decoded

                # Drain already-finished write futures before acquiring another slot
                while write_futures and write_futures[0][1].done():
                    done_item, done_fut = write_futures.pop(0)
                    done_fut.result()
                    completed_count += 1
                    if on_completed is not None:
                        on_completed(done_item, completed_count)

                write_semaphore.acquire()
                wr_fut = wr_pool.submit(write_task, item, rendered)
                del rendered
                write_futures.append((item, wr_fut))

            for done_item, done_fut in write_futures:
                check_cancel()
                done_fut.result()
                completed_count += 1
                if on_completed is not None:
                    on_completed(done_item, completed_count)

            if first_error:
                raise first_error[0]
        except BaseException:
            stop_event.set()
            # Drain decode_queue so feeder never blocks on put()
            while True:
                try:
                    if decode_queue.get_nowait() is None:
                        break
                except queue.Empty:
                    if not feeder_thread.is_alive():
                        break
                    time.sleep(0.005)
            raise
        finally:
            feeder_thread.join(timeout=5.0)

    return StreamingStereoPipelineStats(
        frame_count=total,
        render_count=total,
        skip_count=0,
        capacity=bounded_capacity,
        decode_workers=d_workers,
        write_workers=w_workers,
        peak_in_flight=peak_in_flight,
        elapsed_seconds=time.perf_counter() - started,
    )
