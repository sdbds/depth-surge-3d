"""Per-volume read-only disk admission for the stereo stage."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Callable, Hashable, Iterable

from .metric_geometry import filesystem_allocation_unit


class StereoStageDiskBudgetError(OSError):
    """A target volume cannot hold its complete pending stereo transaction."""

    def __init__(
        self,
        path: Path,
        required_bytes: int,
        free_bytes: int,
        *,
        first_mutation_reserve: int,
        reclaimable_bytes: int,
    ) -> None:
        self.path = Path(path)
        self.required_bytes = int(required_bytes)
        self.free_bytes = int(free_bytes)
        self.first_mutation_reserve = int(first_mutation_reserve)
        self.reclaimable_bytes = int(reclaimable_bytes)
        super().__init__(
            f"stereo stage requires {self.required_bytes} bytes on {self.path}; "
            f"first-mutation reserve is {self.first_mutation_reserve} bytes, "
            f"free space is {self.free_bytes} bytes, and audited reclaimable "
            f"space is {self.reclaimable_bytes} bytes"
        )


@dataclass(frozen=True, slots=True)
class StereoVolumeRequirement:
    key: Hashable
    representative_path: Path
    allocation_unit: int
    required_bytes: int
    first_mutation_reserve: int = 0


@dataclass(frozen=True, slots=True)
class StereoStageDiskPlan:
    volumes: tuple[StereoVolumeRequirement, ...]

    @property
    def required_bytes(self) -> int:
        return sum(volume.required_bytes for volume in self.volumes)


def _existing_anchor(path: Path) -> Path:
    candidate = Path(os.path.abspath(os.fspath(path)))
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise OSError(f"stereo target has no existing volume anchor: {path}")
        candidate = parent
    return candidate


def _default_volume_key(path: Path) -> Hashable:
    anchor = _existing_anchor(path)
    metadata = os.stat(anchor)
    return (int(metadata.st_dev), anchor.anchor.casefold())


def _default_allocation_unit(path: Path) -> int:
    return max(4096, int(filesystem_allocation_unit(_existing_anchor(path))))


class StereoStageDiskPlanBuilder:
    """Accumulate exact allocated bytes and one-transaction overlap per volume."""

    def __init__(
        self,
        *,
        volume_key: Callable[[Path], Hashable] = _default_volume_key,
        allocation_unit: Callable[[Path], int] = _default_allocation_unit,
    ) -> None:
        self._volume_key = volume_key
        self._allocation_unit = allocation_unit
        self._paths: dict[Hashable, Path] = {}
        self._units: dict[Hashable, int] = {}
        self._required: dict[Hashable, int] = {}
        self._maximum_overlap: dict[Hashable, int] = {}
        self._maximum_overlap_entries: dict[Hashable, int] = {}
        self._directory_entries: dict[Hashable, int] = {}
        self._first_mutation: dict[Hashable, int] = {}

    def _target(self, path: Path) -> tuple[Hashable, int]:
        path = Path(path)
        key = self._volume_key(path)
        unit = max(4096, int(self._allocation_unit(path)))
        if unit <= 0:
            raise ValueError("stereo allocation unit must be positive")
        previous = self._units.setdefault(key, unit)
        if previous != unit:
            raise ValueError("one volume reported inconsistent allocation units")
        self._paths.setdefault(key, path)
        self._required.setdefault(key, 0)
        self._directory_entries.setdefault(key, 0)
        self._maximum_overlap.setdefault(key, 0)
        self._maximum_overlap_entries.setdefault(key, 0)
        self._first_mutation.setdefault(key, 0)
        return key, unit

    @staticmethod
    def _allocated(logical_bytes: int, unit: int) -> int:
        if isinstance(logical_bytes, bool) or not isinstance(logical_bytes, int):
            raise TypeError("stereo logical byte count must be an integer")
        if logical_bytes < 0:
            raise ValueError("stereo logical byte count must be nonnegative")
        return 0 if logical_bytes == 0 else ((logical_bytes + unit - 1) // unit) * unit

    def add_payload(self, path: Path, logical_bytes: int, *, count: int = 1) -> None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("stereo payload count must be a nonnegative integer")
        key, unit = self._target(path)
        self._required[key] += count * self._allocated(logical_bytes, unit)

    def add_directory_entries(self, path: Path, count: int) -> None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("stereo directory-entry count must be nonnegative")
        key, _unit = self._target(path)
        self._directory_entries[key] += count

    def add_transaction(self, entries: Iterable[tuple[Path, int]]) -> None:
        """Fold one atomic transaction into per-volume scalar maxima."""

        allocated: dict[Hashable, int] = {}
        entry_counts: dict[Hashable, int] = {}
        for path, logical_bytes in entries:
            key, unit = self._target(path)
            allocated[key] = allocated.get(key, 0) + self._allocated(logical_bytes, unit)
            entry_counts[key] = entry_counts.get(key, 0) + 1
        for key, value in allocated.items():
            self._maximum_overlap[key] = max(self._maximum_overlap[key], value)
            self._maximum_overlap_entries[key] = max(
                self._maximum_overlap_entries[key],
                entry_counts[key],
            )

    def add_first_mutation_transaction(self, path: Path, logical_bytes: int) -> None:
        """Charge one create/replace transaction without reclaim credit."""

        key, unit = self._target(path)
        self._first_mutation[key] += self._allocated(logical_bytes, unit) + unit

    def build(self) -> StereoStageDiskPlan:
        requirements: list[StereoVolumeRequirement] = []
        for key, path in self._paths.items():
            unit = self._units[key]
            required = (
                self._required[key]
                + self._maximum_overlap[key]
                + unit
                * (
                    self._directory_entries[key]
                    + self._maximum_overlap_entries[key]
                )
            )
            requirements.append(
                StereoVolumeRequirement(
                    key=key,
                    representative_path=path,
                    allocation_unit=unit,
                    required_bytes=required,
                    first_mutation_reserve=self._first_mutation[key],
                )
            )
        requirements.sort(key=lambda value: repr(value.key))
        return StereoStageDiskPlan(tuple(requirements))


def admit_stereo_stage_disk(
    plan: StereoStageDiskPlan,
    *,
    free_bytes: Callable[[Path], int] | None = None,
) -> None:
    """Apply both no-reclaim and complete-plan inequalities per volume."""

    if not isinstance(plan, StereoStageDiskPlan):
        raise TypeError("stereo disk admission requires a StereoStageDiskPlan")
    query = free_bytes or (lambda path: int(shutil.disk_usage(_existing_anchor(path)).free))
    for volume in plan.volumes:
        available = int(query(volume.representative_path))
        if available < volume.first_mutation_reserve or available < volume.required_bytes:
            raise StereoStageDiskBudgetError(
                volume.representative_path,
                volume.required_bytes,
                available,
                first_mutation_reserve=volume.first_mutation_reserve,
                reclaimable_bytes=0,
            )
