"""Fixed-capacity indexed heap for the Task 0 geodesic prototype."""

from __future__ import annotations

from typing import SupportsInt

import numpy as np


class QualityRegionQueueBudgetError(RuntimeError):
    """The fixed heap contract or its dense position state was violated."""


class IndexedRegionHeap:
    """One live heap entry per unsettled raster pixel."""

    def __init__(
        self,
        distance: np.ndarray,
        owner: np.ndarray,
        region_rank_bits: np.ndarray,
        *,
        capacity: int,
    ) -> None:
        self._validate_arrays(distance, owner, region_rank_bits)
        if isinstance(capacity, (bool, np.bool_)) or not isinstance(capacity, (int, np.integer)):
            raise TypeError("capacity must be an integer")
        if not 0 < int(capacity) <= distance.size:
            raise ValueError("capacity must be within 1..raster pixel count")
        self.distance = distance
        self.owner = owner
        self.region_rank_bits = region_rank_bits
        self.capacity = int(capacity)
        self.heap_pixel = np.empty(self.capacity, dtype=np.uint32)
        self.position = np.full(distance.shape, -1, dtype=np.int32)
        self._length = 0
        self.max_live_entries = 0
        self.insertions = 0
        self.decrease_keys = 0
        self.pop_count = 0

    def __len__(self) -> int:
        return self._length

    def push_or_decrease(self, pixel: SupportsInt) -> None:
        sample = self._validate_pixel(pixel)
        slot = int(self.position.flat[sample])
        if slot == -2:
            raise QualityRegionQueueBudgetError("cannot decrease a settled heap pixel")
        if slot == -1:
            self._insert(sample)
            return
        if slot < 0 or slot >= self._length or int(self.heap_pixel[slot]) != sample:
            raise QualityRegionQueueBudgetError("heap position state is inconsistent")
        self.decrease_keys += 1
        self._sift_up(slot)

    def pop(self) -> int:
        if self._length == 0:
            raise IndexError("cannot pop an empty indexed heap")
        result = int(self.heap_pixel[0])
        self.pop_count += 1
        self.position.flat[result] = np.int32(-2)
        self._length -= 1
        if self._length:
            replacement = int(self.heap_pixel[self._length])
            self.heap_pixel[0] = np.uint32(replacement)
            self.position.flat[replacement] = np.int32(0)
            self._sift_down(0)
        return result

    @staticmethod
    def _validate_arrays(
        distance: np.ndarray,
        owner: np.ndarray,
        region_rank_bits: np.ndarray,
    ) -> None:
        if not isinstance(distance, np.ndarray) or distance.dtype != np.uint64:
            raise TypeError("distance must be a uint64 NumPy array")
        if distance.ndim != 2 or not distance.flags.c_contiguous:
            raise ValueError("distance must be a C-contiguous 2D raster")
        if (
            not isinstance(owner, np.ndarray)
            or owner.dtype != np.uint32
            or owner.shape != distance.shape
            or not owner.flags.c_contiguous
        ):
            raise TypeError("owner must be a matching C-contiguous uint32 raster")
        if (
            not isinstance(region_rank_bits, np.ndarray)
            or region_rank_bits.dtype != np.uint32
            or region_rank_bits.ndim != 1
            or region_rank_bits.size == 0
            or not region_rank_bits.flags.c_contiguous
        ):
            raise TypeError("region_rank_bits must be a non-empty uint32 vector")

    def _validate_pixel(self, pixel: SupportsInt) -> int:
        if isinstance(pixel, (bool, np.bool_)) or not isinstance(pixel, (int, np.integer)):
            raise TypeError("heap pixel must be an integer")
        sample = int(pixel)
        if not 0 <= sample < self.distance.size:
            raise ValueError("heap pixel lies outside the raster")
        region = int(self.owner.flat[sample])
        if not 0 < region < self.region_rank_bits.size:
            raise ValueError("queued heap pixel must have a known canonical region")
        return sample

    def _insert(self, pixel: int) -> None:
        if self._length >= self.capacity:
            raise QualityRegionQueueBudgetError("indexed heap capacity exceeded")
        slot = self._length
        self.heap_pixel[slot] = np.uint32(pixel)
        self.position.flat[pixel] = np.int32(slot)
        self._length += 1
        self.insertions += 1
        self.max_live_entries = max(self.max_live_entries, self._length)
        self._sift_up(slot)

    def _priority(self, pixel: int) -> tuple[int, int, int, int, int]:
        width = self.distance.shape[1]
        y, x = divmod(pixel, width)
        region = int(self.owner.flat[pixel])
        return (
            int(self.distance.flat[pixel]),
            -int(self.region_rank_bits[region]),
            region,
            y,
            x,
        )

    def _less(self, left_slot: int, right_slot: int) -> bool:
        return self._priority(int(self.heap_pixel[left_slot])) < self._priority(
            int(self.heap_pixel[right_slot])
        )

    def _swap(self, left_slot: int, right_slot: int) -> None:
        left_pixel = int(self.heap_pixel[left_slot])
        right_pixel = int(self.heap_pixel[right_slot])
        self.heap_pixel[left_slot] = np.uint32(right_pixel)
        self.heap_pixel[right_slot] = np.uint32(left_pixel)
        self.position.flat[left_pixel] = np.int32(right_slot)
        self.position.flat[right_pixel] = np.int32(left_slot)

    def _sift_up(self, slot: int) -> None:
        while slot:
            parent = (slot - 1) // 2
            if not self._less(slot, parent):
                return
            self._swap(slot, parent)
            slot = parent

    def _sift_down(self, slot: int) -> None:
        while True:
            left = slot * 2 + 1
            if left >= self._length:
                return
            right = left + 1
            child = right if right < self._length and self._less(right, left) else left
            if not self._less(child, slot):
                return
            self._swap(slot, child)
            slot = child
