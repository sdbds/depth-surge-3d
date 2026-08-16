"""Tests for the final shared processing-settings schema."""

from __future__ import annotations

import os

import pytest

from src.depth_surge_3d.core.settings import (
    PROCESSING_SETTINGS_SCHEMA_VERSION,
    REMOVED_SETTING_NAMES,
    validate_settings,
)
from src.depth_surge_3d.core.constants import DEFAULT_SETTINGS


def test_direct_vr_encode_defaults_off() -> None:
    assert DEFAULT_SETTINGS["direct_vr_encode"] is False
    assert validate_settings({}, source="legacy_disk")["direct_vr_encode"] is False


@pytest.mark.parametrize("value", [0, 1, "false", "true", None])
def test_direct_vr_encode_rejects_non_booleans(value: object) -> None:
    with pytest.raises(ValueError, match="direct_vr_encode"):
        validate_settings({"direct_vr_encode": value}, source="explicit")


@pytest.mark.parametrize("value", [False, True])
def test_direct_vr_encode_accepts_booleans(value: bool) -> None:
    assert (
        validate_settings({"direct_vr_encode": value}, source="explicit")["direct_vr_encode"]
        is value
    )


def test_metric_geometry_settings_bump_settings_schema() -> None:
    assert PROCESSING_SETTINGS_SCHEMA_VERSION == 3


def test_metric_geometry_defaults_are_explicit() -> None:
    settings = validate_settings({}, source="explicit")
    assert settings["stereo_geometry_mode"] == "relative"
    assert settings["virtual_baseline_mm"] == 63.0
    assert settings["metric_convergence_distance"] == "auto"
    assert settings["max_disparity_percent"] == 2.0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("virtual_baseline_mm", -0.01),
        ("virtual_baseline_mm", 100.01),
        ("virtual_baseline_mm", float("nan")),
        ("metric_convergence_distance", 0.09),
        ("metric_convergence_distance", 1000.01),
        ("metric_convergence_distance", float("inf")),
        ("max_disparity_percent", -0.01),
        ("max_disparity_percent", 5.01),
    ],
)
def test_metric_settings_reject_out_of_contract_values(name, value) -> None:
    with pytest.raises(ValueError, match=name):
        validate_settings({name: value}, source="explicit")


def test_final_defaults_cover_depth_dibr_and_migration_controls() -> None:
    settings = validate_settings({}, source="explicit")

    assert settings["stereo_strength"] == 2.0
    assert settings["convergence"] == 0.5
    assert settings["occlusion_fill"] == "background"
    assert settings["scene_detection"] is True
    assert settings["scene_cut_threshold"] == 0.55
    assert settings["min_scene_frames"] == 8
    assert settings["raw_storage_dtype"] == "auto"
    assert settings["stereo_io_workers"] == min(4, max(1, (os.cpu_count() or 1) - 2))
    assert settings["migrate_legacy"] == "archive"
    assert REMOVED_SETTING_NAMES == {
        "baseline",
        "focal_length",
        "hole_fill_quality",
        "processing_mode",
    }
    assert REMOVED_SETTING_NAMES.isdisjoint(settings)


@pytest.mark.parametrize(
    "values",
    [
        {"stereo_strength": -0.01},
        {"stereo_strength": 5.01},
        {"convergence": -0.01},
        {"convergence": 1.01},
        {"scene_cut_threshold": -0.01},
        {"scene_cut_threshold": 1.01},
        {"min_scene_frames": 0},
        {"stereo_io_workers": 0},
        {"stereo_io_workers": 17},
    ],
)
def test_numeric_ranges_are_enforced(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        validate_settings(values, source="explicit")


@pytest.mark.parametrize(
    "values",
    [
        {"occlusion_fill": "telea"},
        {"raw_storage_dtype": "float64"},
        {"migrate_legacy": "prompt"},
    ],
)
def test_option_sets_are_enforced(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        validate_settings(values, source="explicit")


@pytest.mark.parametrize("name", sorted(REMOVED_SETTING_NAMES))
def test_removed_explicit_names_fail_instead_of_being_ignored(name: str) -> None:
    with pytest.raises(ValueError, match=f"removed setting.*{name}"):
        validate_settings({name: 1}, source="explicit")


def test_unknown_explicit_name_fails() -> None:
    with pytest.raises(ValueError, match="unknown setting.*mystery"):
        validate_settings({"mystery": 1}, source="explicit")


def test_legacy_disk_strips_removed_and_unknown_names_without_mutating_input() -> None:
    stored = {
        "baseline": 0.065,
        "focal_length": 1000,
        "hole_fill_quality": "fast",
        "mystery_v1": True,
        "stereo_strength": 3.0,
    }

    migrated = validate_settings(stored, source="legacy_disk")

    assert migrated["stereo_strength"] == 3.0
    assert "baseline" not in migrated
    assert "focal_length" not in migrated
    assert "hole_fill_quality" not in migrated
    assert "mystery_v1" not in migrated
    assert stored["baseline"] == 0.065


def test_boolean_fields_do_not_accept_integer_surrogates() -> None:
    with pytest.raises(ValueError, match="scene_detection"):
        validate_settings({"scene_detection": 1}, source="explicit")


def test_invalid_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="source"):
        validate_settings({}, source="resume")


def test_moge2_is_a_valid_persisted_depth_backend() -> None:
    settings = validate_settings(
        {
            "depth_model_version": "moge2",
            "model_size": "vitb",
            "model_path": None,
            "depth_resolution": "auto",
            "use_metric_depth": True,
        },
        source="explicit",
    )

    assert settings["depth_model_version"] == "moge2"
