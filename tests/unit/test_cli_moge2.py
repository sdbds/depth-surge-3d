"""CLI coverage for selecting the optional MoGe-2 backend."""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.depth_surge_3d.inference.depth.backend_registry import BackendAvailability


@pytest.fixture
def cli_module():
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root / "src"))
    try:
        return importlib.import_module("depth_surge_3d.cli")
    finally:
        sys.path.pop(0)


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


def test_cli_help_describes_backend_temporal_behavior(cli_module) -> None:
    help_text = cli_module.create_argument_parser().format_help()

    assert "V2 uses fixed shot-aware temporal inference" in help_text
    assert "V3, See-Through, and MoGe-2 infer frames independently" in help_text


@pytest.mark.parametrize(
    ("invalid_settings", "sample_aspect_ratio"),
    [
        ({}, (4, 3)),
        ({"vr_format": "over_under"}, (1, 1)),
        ({"apply_distortion": True}, (1, 1)),
    ],
)
def test_cli_resume_rejects_metric_constraints_before_load_or_migration(
    cli_module, monkeypatch, tmp_path, invalid_settings, sample_aspect_ratio
) -> None:
    import importlib

    projector_module = importlib.import_module("depth_surge_3d.rendering.stereo_projector")
    estimator = MagicMock()
    estimator.get_model_size.return_value = "vitb"
    estimator.device = "cpu"
    estimator.metric = True
    estimator.repo_id = "Ruicheng/moge-2-vitb-normal"
    estimator.revision = "54ad3a693e61907ea4633d13dec6ee682fa09419"
    estimator.inference_precision = "float32"
    monkeypatch.setattr(
        projector_module,
        "create_registered_depth_estimator",
        lambda *_args, **_kwargs: estimator,
    )
    monkeypatch.setattr(projector_module, "validate_video_file", lambda _path: True)
    monkeypatch.setattr(
        projector_module,
        "get_video_properties",
        lambda _path: {
            "width": 1920,
            "height": 1080,
            "fps": 24.0,
            "frame_count": 1,
            "sample_aspect_ratio_numerator": sample_aspect_ratio[0],
            "sample_aspect_ratio_denominator": sample_aspect_ratio[1],
        },
    )
    projector = projector_module.StereoProjector(
        device="cpu",
        metric=True,
        depth_model_version="moge2",
        model_size="vitb",
    )
    settings = {
        "depth_model_version": "moge2",
        "model_size": "vitb",
        "stereo_geometry_mode": "metric_camera",
        "vr_format": "side_by_side",
        "apply_distortion": False,
        **invalid_settings,
    }
    resume_info = {
        "can_resume": True,
        "batch_name": "metric",
        "status": "in_progress",
        "progress_info": None,
        "recommendations": [],
        "settings_file": tmp_path / "job-settings.json",
    }
    build_resume = MagicMock()
    build_fingerprint = MagicMock()
    migrate = MagicMock()
    monkeypatch.setattr(cli_module.sys, "argv", ["depth_surge_3d.py", "--resume", str(tmp_path)])
    monkeypatch.setattr(cli_module, "can_resume_processing", lambda _path: resume_info)
    monkeypatch.setattr(
        cli_module,
        "load_processing_settings",
        lambda _path: {
            "metadata": {"source_video": "source.mp4"},
            "processing_settings": settings,
        },
    )
    monkeypatch.setattr(cli_module, "_backend_is_available", lambda _backend: True)
    monkeypatch.setattr(cli_module, "create_stereo_projector", lambda **_kwargs: projector)
    monkeypatch.setattr(cli_module, "build_resume_report", build_resume)
    monkeypatch.setattr(cli_module, "build_current_model_fingerprint", build_fingerprint)
    monkeypatch.setattr(cli_module, "apply_legacy_migration", migrate)

    result = cli_module.main()

    assert result == 1
    estimator.load_model.assert_not_called()
    build_fingerprint.assert_not_called()
    build_resume.assert_not_called()
    migrate.assert_not_called()


def test_cli_resume_rebuilds_numeric_preflight_from_migrated_settings(
    cli_module, monkeypatch, tmp_path
) -> None:
    initial_preflight = MagicMock()
    initial_preflight.settings = {
        "depth_model_version": "v3",
        "model_size": "vitl",
        "model_path": None,
        "depth_resolution": 1080,
        "keep_intermediates": False,
    }
    migrated_settings = {**initial_preflight.settings, "keep_intermediates": True}
    execution_preflight = MagicMock()
    projector = MagicMock()
    projector.preflight_video.return_value = initial_preflight
    projector.load_model.return_value = True
    projector.revalidate_video_preflight.return_value = execution_preflight
    projector.execute_video.return_value = True
    report = MagicMock(migrated_settings=migrated_settings, stages=(), removed_settings=())
    resume_info = {
        "can_resume": True,
        "batch_name": "resume",
        "status": "in_progress",
        "progress_info": None,
        "recommendations": [],
        "settings_file": tmp_path / "job-settings.json",
    }
    fingerprint = {"backend": "loaded.Estimator"}
    monkeypatch.setattr(cli_module.sys, "argv", ["depth_surge_3d.py", "--resume", str(tmp_path)])
    monkeypatch.setattr(cli_module, "can_resume_processing", lambda _path: resume_info)
    monkeypatch.setattr(
        cli_module,
        "load_processing_settings",
        lambda _path: {
            "metadata": {"source_video": "source.mp4"},
            "processing_settings": {
                "depth_model_version": "v3",
                "depth_resolution": "auto",
            },
        },
    )
    monkeypatch.setattr(cli_module, "_backend_is_available", lambda _backend: True)
    monkeypatch.setattr(cli_module, "create_stereo_projector", lambda **_kwargs: projector)
    build_fingerprint = MagicMock(return_value=fingerprint)
    build_report = MagicMock(return_value=report)
    monkeypatch.setattr(cli_module, "build_current_model_fingerprint", build_fingerprint)
    monkeypatch.setattr(cli_module, "build_resume_report", build_report)
    monkeypatch.setattr(cli_module, "apply_legacy_migration", MagicMock())

    result = cli_module.main()

    assert result == 0
    build_fingerprint.assert_called_once_with(
        projector.depth_estimator,
        initial_preflight.settings,
    )
    assert build_report.call_args.args[1]["depth_resolution"] == 1080
    projector.revalidate_video_preflight.assert_called_once_with(
        initial_preflight,
        migrated_settings,
    )
    projector.execute_video.assert_called_once_with(execution_preflight)
    projector.process_video.assert_not_called()
