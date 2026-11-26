"""Caller-owned fixed host arenas for one Quality repair eye."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .repair_records import (
    QUALITY_REPAIR_RECORD_CAP,
    REPAIR_RECORD_DTYPE,
    QualityRepairBudgetError,
)


QUALITY_RECORD_ARENA_BYTES = 64 * 1024 * 1024
QUALITY_GRAPH_ARENA_BYTES = 64 * 1024 * 1024
QUALITY_REPAIR_ARENA_BYTES = 64 * 1024 * 1024


class QualityArenaAllocationError(QualityRepairBudgetError):
    """A fixed arena or one of its frozen partitions cannot represent the work."""


@dataclass
class QualityByteArena:
    """A resettable aligned bump allocator backed by one fixed NumPy byte array."""

    storage: np.ndarray
    label: str
    cursor: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.storage, np.ndarray)
            or self.storage.dtype != np.uint8
            or self.storage.ndim != 1
            or not self.storage.flags.c_contiguous
        ):
            raise TypeError("arena storage must be a contiguous uint8 vector")

    @property
    def capacity(self) -> int:
        return int(self.storage.nbytes)

    def reset(self) -> None:
        self.cursor = 0

    def mark(self) -> int:
        return self.cursor

    def rewind(self, mark: int) -> None:
        if isinstance(mark, bool) or not isinstance(mark, int) or not 0 <= mark <= self.cursor:
            raise ValueError("arena rewind mark is invalid")
        self.cursor = mark

    def allocate(
        self,
        shape: tuple[int, ...] | int,
        dtype: np.dtype | type,
        *,
        zero: bool = False,
        alignment: int = 64,
    ) -> np.ndarray:
        dimensions = (shape,) if isinstance(shape, int) else shape
        if not dimensions or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in dimensions
        ):
            raise ValueError("arena allocation shape must contain nonnegative integers")
        item_dtype = np.dtype(dtype)
        item_count = 1
        for value in dimensions:
            item_count *= value
        byte_count = item_count * item_dtype.itemsize
        start = (self.cursor + alignment - 1) // alignment * alignment
        end = start + byte_count
        if end > self.capacity:
            raise QualityArenaAllocationError(
                f"{self.label} fixed arena exceeded: requested {byte_count} bytes with "
                f"{self.capacity - start} bytes remaining"
            )
        result = np.ndarray(
            dimensions,
            dtype=item_dtype,
            buffer=self.storage,
            offset=start,
            order="C",
        )
        self.cursor = end
        if zero:
            result.fill(0)
        return result


@dataclass
class QualityFixedArenas:
    """The three atomically preallocated Quality repair arenas."""

    record_storage: np.ndarray
    graph: QualityByteArena | None
    repair: QualityByteArena | None

    @staticmethod
    def _allocate_planning_scratch() -> tuple[QualityByteArena, QualityByteArena]:
        try:
            graph_storage = np.empty(QUALITY_GRAPH_ARENA_BYTES, dtype=np.uint8)
            repair_storage = np.empty(QUALITY_REPAIR_ARENA_BYTES, dtype=np.uint8)
        except MemoryError as error:
            raise QualityArenaAllocationError(
                "Quality planning arena preallocation failed before Pass A"
            ) from error
        return (
            QualityByteArena(graph_storage, "Quality graph"),
            QualityByteArena(repair_storage, "Quality repair"),
        )

    @classmethod
    def allocate(cls) -> QualityFixedArenas:
        try:
            record = np.empty(QUALITY_RECORD_ARENA_BYTES, dtype=np.uint8)
        except MemoryError as error:
            raise QualityArenaAllocationError(
                "Quality fixed arena preallocation failed before Pass A"
            ) from error
        try:
            graph, repair = cls._allocate_planning_scratch()
        except QualityArenaAllocationError as error:
            raise QualityArenaAllocationError(
                "Quality fixed arena preallocation failed before Pass A"
            ) from error
        return cls(record, graph, repair)

    def reset_eye(self) -> None:
        if self.graph is None or self.repair is None:
            if self.graph is not None or self.repair is not None:
                raise AssertionError("Quality planning arenas have split ownership")
            self.graph, self.repair = self._allocate_planning_scratch()
        self.graph.reset()
        self.repair.reset()

    def release_planning_scratch(self) -> None:
        if self.graph is None or self.repair is None:
            raise AssertionError("Quality planning arenas are not both live")
        self.graph = None
        self.repair = None

    def records(self) -> np.ndarray:
        result = self.record_storage.view(REPAIR_RECORD_DTYPE)
        if result.size != QUALITY_REPAIR_RECORD_CAP:
            raise AssertionError("record arena size differs from the canonical cap")
        return result
