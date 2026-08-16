"""CLI coverage for selecting the optional MoGe-2 backend."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from src.depth_surge_3d.inference.depth.backend_registry import BackendAvailability


@pytest.fixture
def cli_module():
    project_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "depth_surge_3d_cli_moge2_test",
        project_root / "depth_surge_3d.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_accepts_moge_and_normalizes_default_variant(cli_module) -> None:
    parser = cli_module.create_argument_parser()
    args = parser.parse_args(["clip.mp4", "--depth-model-version", "moge2"])
    settings = cli_module._build_processing_settings(args)
    assert settings["depth_model_version"] == "moge2"
    assert settings["model_size"] == "vitb"
    assert settings["model_path"] is None
    assert settings["use_metric_depth"] is True


def test_cli_model_override_and_size_are_mutually_exclusive(cli_module) -> None:
    parser = cli_module.create_argument_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "clip.mp4",
                "--depth-model-version",
                "moge2",
                "--model",
                "owner/repo@abc",
                "--model-size",
                "vits",
            ]
        )


def test_cli_missing_extra_fails_before_projector_creation(cli_module, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_module,
        "backend_availability",
        lambda _backend: BackendAvailability(
            False,
            "MoGe-2 optional dependency is not installed",
            "uv sync --extra moge2",
        ),
    )
    monkeypatch.setattr(cli_module, "validate_video_file", lambda _path: True)
    args = cli_module.create_argument_parser().parse_args(
        ["clip.mp4", "--depth-model-version", "moge2"]
    )
    assert cli_module.validate_arguments(args) is None
    assert "Install with: uv sync --extra moge2" in capsys.readouterr().out


def test_cli_normalizes_custom_model_and_depth_resolution(cli_module) -> None:
    args = cli_module.create_argument_parser().parse_args(
        [
            "clip.mp4",
            "--depth-model-version",
            "moge2",
            "--model",
            "owner/repo@abc",
            "--depth-resolution",
            "720",
        ]
    )

    settings = cli_module._build_processing_settings(args)

    assert settings["model_size"] == "custom"
    assert settings["model_path"] == "owner/repo@abc"
    assert settings["depth_resolution"] == "720"
    assert settings["use_metric_depth"] is True


def test_cli_disables_metric_inference_for_non_metric_backend(cli_module) -> None:
    args = cli_module.create_argument_parser().parse_args(
        ["clip.mp4", "--depth-model-version", "see_through", "--metric"]
    )

    settings = cli_module._build_processing_settings(args)

    assert settings["use_metric_depth"] is False


def test_cli_normalizes_metric_geometry_options(cli_module) -> None:
    args = cli_module.create_argument_parser().parse_args(
        [
            "clip.mp4",
            "--depth-model-version",
            "moge2",
            "--model-size",
            "vits",
            "--stereo-geometry-mode",
            "metric_camera",
            "--virtual-baseline-mm",
            "64.0",
            "--metric-convergence-distance",
            "2.5",
            "--max-disparity-percent",
            "1.5",
            "--format",
            "side_by_side",
            "--no-distortion",
        ]
    )

    settings = cli_module._build_processing_settings(args)

    assert settings["stereo_geometry_mode"] == "metric_camera"
    assert settings["virtual_baseline_mm"] == 64.0
    assert settings["metric_convergence_distance"] == 2.5
    assert settings["max_disparity_percent"] == 1.5


def test_cli_accepts_auto_metric_convergence(cli_module) -> None:
    args = cli_module.create_argument_parser().parse_args(
        ["clip.mp4", "--metric-convergence-distance", "auto"]
    )

    assert cli_module._build_processing_settings(args)["metric_convergence_distance"] == "auto"


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_cli_rejects_nonfinite_metric_convergence(cli_module, value: str) -> None:
    parser = cli_module.create_argument_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["clip.mp4", "--metric-convergence-distance", value])

    assert not math.isfinite(float(value))


def test_cli_has_no_public_moge_resolution_level_flag(cli_module) -> None:
    help_text = cli_module.create_argument_parser().format_help()

    assert "moge-resolution" not in help_text
    assert "resolution-level" not in help_text
