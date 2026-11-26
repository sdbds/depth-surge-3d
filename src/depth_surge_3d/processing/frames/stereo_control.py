"""Dynamic RAM-aware streaming stereo pipeline controller."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import queue
import threading
import time
from typing import Any, TypeVar

from ...core.host_memory import available_host_memory_bytes


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

    @property
    def pipeline_wall_seconds(self) -> float:
        return self.elapsed_seconds

    @property
    def permit_count(self) -> int:
        return self.capacity

    @property
    def written_frames(self) -> int:
        return self.render_count

    @property
    def writer_busy_seconds(self) -> float:
        return 0.0

    @property
    def queue_wait_seconds(self) -> float:
        return 0.0

    @property
    def permit_wait_seconds(self) -> float:
        return 0.0


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
    max_capacity: int = 24,
) -> int:
    """Choose in-flight queue capacity for Quality stereo rendering."""

    height, width = int(render_shape[0]), int(render_shape[1])
    pixels = max(1, height * width)
    per_slot = pixels * 64 + max(0, int(extra_per_frame_bytes)) + 16 * 1024 * 1024
    budget = host_budget_bytes if host_budget_bytes is not None else dynamic_host_budget_bytes()
    usable = budget - STEREO_CONTROL_OVERHEAD
    if usable < per_slot:
        return 1
    return max(1, min(max_capacity, usable // per_slot))


def _run_four_stage_pipeline(  # noqa: C901
    work_list: list[TItem],
    *,
    decode_fn: Callable[[TItem], TDecoded],
    render_fn: Callable[[TItem, TDecoded], TRendered],
    postprocess_fn: Callable[[TItem, TRendered], Any],
    write_fn: Callable[[TItem, Any], None],
    bounded_capacity: int,
    d_workers: int,
    p_workers: int,
    w_workers: int,
    on_completed: Callable[[TItem, int], None] | None,
    is_cancelled: Callable[[], bool] | None,
    started: float,
) -> StreamingStereoPipelineStats:
    """Execute bounded 4-stage decode/prep -> GPU splat -> CPU repair -> write pipeline."""

    total = len(work_list)
    stop_event = threading.Event()
    slot_semaphore = threading.Semaphore(bounded_capacity)
    in_flight_lock = threading.Lock()
    error_lock = threading.Lock()
    in_flight = 0
    peak_in_flight = 0
    first_error: list[BaseException] = []

    def record_error(exc: BaseException) -> None:
        with error_lock:
            if not first_error:
                first_error.append(exc)
        stop_event.set()

    def check_cancel() -> None:
        if is_cancelled is not None and is_cancelled():
            record_error(InterruptedError("Stereo processing was cancelled"))
            raise InterruptedError("Stereo processing was cancelled")

    def mark_enter() -> None:
        nonlocal in_flight, peak_in_flight
        with in_flight_lock:
            in_flight += 1
            if in_flight > peak_in_flight:
                peak_in_flight = in_flight

    def release_slot() -> None:
        nonlocal in_flight
        with in_flight_lock:
            in_flight = max(0, in_flight - 1)
        slot_semaphore.release()

    decode_queue: queue.Queue[tuple[TItem, Future[TDecoded]] | None] = queue.Queue()
    post_queue: queue.Queue[tuple[TItem, Future[Any]] | None] = queue.Queue()

    def decode_task(item: TItem) -> TDecoded:
        if stop_event.is_set():
            raise InterruptedError("cancelled")
        return decode_fn(item)

    def post_task(item: TItem, rendered: TRendered) -> Any:
        try:
            if stop_event.is_set():
                raise InterruptedError("cancelled")
            return postprocess_fn(item, rendered)
        finally:
            del rendered

    def write_task(item: TItem, finished: Any) -> None:
        try:
            if stop_event.is_set():
                return
            write_fn(item, finished)
        finally:
            del finished
            release_slot()

    with (
        ThreadPoolExecutor(max_workers=d_workers, thread_name_prefix="stereo-prep") as dec_pool,
        ThreadPoolExecutor(max_workers=p_workers, thread_name_prefix="stereo-repair") as post_pool,
        ThreadPoolExecutor(max_workers=w_workers, thread_name_prefix="stereo-write") as wr_pool,
    ):

        def feeder() -> None:
            try:
                for item in work_list:
                    if stop_event.is_set():
                        break
                    slot_semaphore.acquire()
                    if stop_event.is_set():
                        slot_semaphore.release()
                        break
                    mark_enter()
                    fut = dec_pool.submit(decode_task, item)
                    decode_queue.put((item, fut))
            except BaseException as exc:
                record_error(exc)
            finally:
                decode_queue.put(None)

        def post_to_write_pump() -> None:
            write_futures: list[tuple[TItem, Future[None]]] = []
            completed_count = 0

            def drain_done(wait_all: bool) -> None:
                nonlocal completed_count
                while write_futures and (wait_all or write_futures[0][1].done()):
                    done_item, done_fut = write_futures.pop(0)
                    done_fut.result()
                    completed_count += 1
                    if on_completed is not None and not stop_event.is_set():
                        on_completed(done_item, completed_count)

            try:
                while True:
                    entry = post_queue.get()
                    if entry is None:
                        break
                    item, post_fut = entry
                    if stop_event.is_set():
                        release_slot()
                        continue
                    try:
                        finished = post_fut.result()
                    except BaseException as exc:
                        release_slot()
                        if not stop_event.is_set():
                            record_error(exc)
                        continue
                    if stop_event.is_set():
                        del finished
                        release_slot()
                        continue
                    wr_fut = wr_pool.submit(write_task, item, finished)
                    del finished
                    write_futures.append((item, wr_fut))
                    drain_done(wait_all=False)
                drain_done(wait_all=True)
            except BaseException as exc:
                record_error(exc)
                # Ensure any remaining items still pushed to post_queue release their slots.
                while True:
                    try:
                        leftover = post_queue.get_nowait()
                    except queue.Empty:
                        break
                    if leftover is None:
                        break
                    release_slot()
                for _item, rem_fut in write_futures:
                    try:
                        rem_fut.result()
                    except BaseException:
                        pass

        feeder_thread = threading.Thread(target=feeder, daemon=True)
        pump_thread = threading.Thread(target=post_to_write_pump, daemon=True)
        feeder_thread.start()
        pump_thread.start()

        try:
            while True:
                entry = decode_queue.get()
                if entry is None:
                    break
                item, dec_fut = entry
                if stop_event.is_set():
                    release_slot()
                    continue
                try:
                    check_cancel()
                    decoded = dec_fut.result()
                    rendered = render_fn(item, decoded)
                    del decoded
                except BaseException:
                    release_slot()
                    raise
                post_fut = post_pool.submit(post_task, item, rendered)
                del rendered
                post_queue.put((item, post_fut))
        except BaseException as exc:
            record_error(exc)
            # Drain decode_queue so feeder never blocks on slot_semaphore.acquire().
            while True:
                entry = decode_queue.get()
                if entry is None:
                    break
                release_slot()
        finally:
            post_queue.put(None)
            feeder_thread.join(timeout=10.0)
            pump_thread.join(timeout=30.0)

    if first_error:
        raise first_error[0]

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


def run_streaming_stereo_pipeline(  # noqa: C901
    items: Iterable[TItem],
    *,
    decode_fn: Callable[[TItem], TDecoded],
    render_fn: Callable[[TItem, TDecoded], TRendered],
    write_fn: Callable[[TItem, Any], None],
    capacity: int,
    decode_workers: int,
    write_workers: int,
    postprocess_fn: Callable[[TItem, TRendered], Any] | None = None,
    postprocess_workers: int | None = None,
    ordered_writes: bool = False,
    on_completed: Callable[[TItem, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> StreamingStereoPipelineStats:
    """Execute bounded decode -> main-thread render -> [optional repair] -> write pipeline."""

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
    p_workers = (
        max(1, min(int(postprocess_workers or d_workers), total))
        if postprocess_fn is not None
        else 1
    )

    # Single-threaded fast path when capacity == 1 and minimal workers
    if bounded_capacity == 1 and d_workers == 1 and w_workers == 1 and p_workers == 1:
        for index, item in enumerate(work_list, 1):
            if is_cancelled is not None and is_cancelled():
                raise InterruptedError("Stereo processing was cancelled")
            decoded = decode_fn(item)
            rendered = render_fn(item, decoded)
            del decoded
            finished: Any = (
                postprocess_fn(item, rendered) if postprocess_fn is not None else rendered
            )
            del rendered
            write_fn(item, finished)
            del finished
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

    if postprocess_fn is not None:
        return _run_four_stage_pipeline(
            work_list,
            decode_fn=decode_fn,
            render_fn=render_fn,
            postprocess_fn=postprocess_fn,
            write_fn=write_fn,
            bounded_capacity=bounded_capacity,
            d_workers=d_workers,
            p_workers=p_workers,
            w_workers=w_workers,
            on_completed=on_completed,
            is_cancelled=is_cancelled,
            started=started,
        )

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
