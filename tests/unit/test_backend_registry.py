"""Tests for the centralized depth backend registry."""

import importlib.util

import pytest

from src.depth_surge_3d.inference.depth.backend_registry import (
    BackendAvailability,
    BackendCapabilities,
    backend_availability,
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
