"""Artifact-first model-loading plan tests."""

from __future__ import annotations

from src.depth_surge_3d.processing.orchestration.execution_plan import (
    build_artifact_execution_plan,
)


def test_complete_stabilized_artifact_skips_base_audit_and_both_models() -> None:
    calls: list[str] = []

    def stable() -> bool:
        calls.append("stabilized")
        return True

    def base() -> bool:
        calls.append("base")
        raise AssertionError("base audit is unnecessary for a content-addressed artifact")

    plan = build_artifact_execution_plan(
        temporal_postprocessor="vdpp",
        base_artifact_valid=base,
        stabilized_artifact_valid=stable,
    )

    assert calls == ["stabilized"]
    assert plan.can_run_cache_only is True
    assert plan.needs_base_depth_model is False
    assert plan.needs_vdpp_model is False
    assert plan.selected_render_source == "stabilized"


def test_vdpp_with_only_base_artifact_loads_only_vdpp() -> None:
    plan = build_artifact_execution_plan(
        temporal_postprocessor="vdpp",
        base_artifact_valid=lambda: True,
        stabilized_artifact_valid=lambda: False,
    )

    assert plan.can_run_cache_only is False
    assert plan.needs_base_depth_model is False
    assert plan.needs_vdpp_model is True
    assert plan.selected_render_source == "stabilized"


def test_vdpp_with_no_artifacts_loads_models_in_sequence() -> None:
    plan = build_artifact_execution_plan(
        temporal_postprocessor="vdpp",
        base_artifact_valid=lambda: False,
        stabilized_artifact_valid=lambda: False,
    )

    assert plan.can_run_cache_only is False
    assert plan.needs_base_depth_model is True
    assert plan.needs_vdpp_model is True


def test_off_mode_never_audits_or_loads_vdpp() -> None:
    plan = build_artifact_execution_plan(
        temporal_postprocessor="off",
        base_artifact_valid=lambda: True,
        stabilized_artifact_valid=lambda: (_ for _ in ()).throw(
            AssertionError("off mode must not inspect the dormant stage")
        ),
    )

    assert plan.can_run_cache_only is True
    assert plan.needs_base_depth_model is False
    assert plan.needs_vdpp_model is False
    assert plan.selected_render_source == "base"
