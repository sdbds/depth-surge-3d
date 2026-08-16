"""Tests for the centralized depth backend registry."""

import importlib.util
from unittest.mock import MagicMock

import pytest

from src.depth_surge_3d.inference.depth.backend_registry import (
    BackendAvailability,
    BackendCapabilities,
    EstimatorRequest,
    backend_availability,
    create_registered_depth_estimator,
    get_backend_spec,
)


@pytest.mark.parametrize("backend_id", ["v2", "v3", "see_through", "moge2"])
def test_registry_contains_every_supported_backend(backend_id: str) -> None:
    """Every documented backend resolves to its own registered specification."""
    assert get_backend_spec(backend_id).backend_id == backend_id


def test_unknown_backend_is_not_a_v3_fallback() -> None:
    """Typos are rejected rather than silently selecting a different backend."""
    with pytest.raises(ValueError, match="Unknown depth backend: typo"):
        get_backend_spec("typo")


def test_moge_variants_and_pins_are_exact() -> None:
    """MoGe selection exposes the pinned artifacts used to reproduce a run."""
    spec = get_backend_spec("moge2")
    assert spec.default_model_size == "vitb"
    assert {
        key: (value.repo_id, value.revision, value.parameters_millions)
        for key, value in spec.variants.items()
    } == {
        "vits": (
            "Ruicheng/moge-2-vits-normal",
            "679230677b4d282c6f304189a93e98e14f085902",
            35,
        ),
        "vitb": (
            "Ruicheng/moge-2-vitb-normal",
            "54ad3a693e61907ea4633d13dec6ee682fa09419",
            104,
        ),
        "vitl": (
            "Ruicheng/moge-2-vitl",
            "39c4d5e957afe587e04eec59dc2bcc3be5ecd968",
            326,
        ),
    }
    assert spec.capabilities == BackendCapabilities(
        metric_depth=True,
        pinhole_fx=True,
        stereo_geometry_modes=frozenset({"relative", "metric_camera"}),
    )


def test_missing_moge_extra_reports_only_supported_command(monkeypatch) -> None:
    """A missing optional dependency gives users the project-supported install path."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    availability = backend_availability("moge2")
    assert availability == BackendAvailability(
        available=False,
        reason="MoGe-2 optional dependency is not installed",
        install_command="uv sync --extra moge2",
    )


def test_selected_moge_factory_receives_immutable_variant(monkeypatch) -> None:
    """The selected registry variant, not a floating alias, reaches the adapter."""
    import src.depth_surge_3d.inference.depth.backend_registry as registry
    import src.depth_surge_3d.inference.depth.video_depth_estimator_moge2 as adapter

    create = MagicMock(return_value=object())
    monkeypatch.setattr(
        registry,
        "_moge_availability",
        lambda: BackendAvailability(available=True),
    )
    monkeypatch.setattr(adapter, "create_video_depth_estimator_moge2", create)
    request = EstimatorRequest(
        model_path=None,
        model_size="vits",
        device="cuda:1",
        metric=True,
        temporal_window_overlap=8,
    )

    result = create_registered_depth_estimator("moge2", request)

    assert result is create.return_value
    create.assert_called_once_with(
        model_size="vits",
        model_path=None,
        repo_id="Ruicheng/moge-2-vits-normal",
        revision="679230677b4d282c6f304189a93e98e14f085902",
        device="cuda:1",
    )


def test_selected_moge_factory_reports_install_command(monkeypatch) -> None:
    import src.depth_surge_3d.inference.depth.backend_registry as registry

    monkeypatch.setattr(
        registry,
        "_moge_availability",
        lambda: BackendAvailability(
            available=False,
            reason="MoGe-2 optional dependency is not installed",
            install_command="uv sync --extra moge2",
        ),
    )
    request = EstimatorRequest(None, "vitb", "cpu", True, 8)
    with pytest.raises(RuntimeError, match=r"Install with: uv sync --extra moge2"):
        create_registered_depth_estimator("moge2", request)
