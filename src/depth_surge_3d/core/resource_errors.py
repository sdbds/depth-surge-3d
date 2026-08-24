"""Typed resource failures shared by bounded processing phases."""

from __future__ import annotations

from pathlib import Path


class QualityHostBudgetError(MemoryError):
    """A Quality phase could not fit or acquire its host-memory ownership."""

    def __init__(self, phase: str, path: Path, requested_bytes: int) -> None:
        self.phase = phase
        self.path: Path | None = Path(path)
        self.requested_bytes = int(requested_bytes)
        self.required_bytes = int(requested_bytes)
        self.budget_bytes: int | None = None
        super().__init__(
            f"{phase} host allocation failed for {self.path} "
            f"({self.requested_bytes} requested bytes)"
        )

    @classmethod
    def for_limit(
        cls,
        phase: str,
        required_bytes: int,
        budget_bytes: int,
    ) -> "QualityHostBudgetError":
        error = cls.__new__(cls)
        error.phase = phase
        error.path = None
        error.requested_bytes = int(required_bytes)
        error.required_bytes = int(required_bytes)
        error.budget_bytes = int(budget_bytes)
        MemoryError.__init__(
            error,
            f"{phase} requires {required_bytes} host bytes; budget is {budget_bytes}",
        )
        return error
