"""Pure artifact-first decisions for lazy video model loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal


RenderSource = Literal["base", "stabilized"]


@dataclass(frozen=True)
class PipelineExecutionPlan:
    """Neural work required to produce the requested render input."""

    can_run_cache_only: bool
    needs_base_depth_model: bool
    needs_vdpp_model: bool
    selected_render_source: RenderSource


def build_artifact_execution_plan(
    *,
    temporal_postprocessor: str,
    base_artifact_valid: Callable[[], bool],
    stabilized_artifact_valid: Callable[[], bool],
) -> PipelineExecutionPlan:
    """Audit the newest requested artifact first and avoid eager dependencies."""

    if temporal_postprocessor == "off":
        base_valid = base_artifact_valid()
        return PipelineExecutionPlan(
            can_run_cache_only=base_valid,
            needs_base_depth_model=not base_valid,
            needs_vdpp_model=False,
            selected_render_source="base",
        )
    if temporal_postprocessor != "vdpp":
        raise ValueError("temporal_postprocessor must be off or vdpp")

    stabilized_valid = stabilized_artifact_valid()
    if stabilized_valid:
        return PipelineExecutionPlan(
            can_run_cache_only=True,
            needs_base_depth_model=False,
            needs_vdpp_model=False,
            selected_render_source="stabilized",
        )

    base_valid = base_artifact_valid()
    return PipelineExecutionPlan(
        can_run_cache_only=False,
        needs_base_depth_model=not base_valid,
        needs_vdpp_model=True,
        selected_render_source="stabilized",
    )
