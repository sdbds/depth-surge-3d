"""Tests for See-Through model selection through public entry points."""

from __future__ import annotations

import importlib.util
import json
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import MagicMock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _DepthModelOptionParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_depth_select = False
        self.values: list[str] = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "select" and attributes.get("id") == "depthModelVersion":
            self.in_depth_select = True
        elif tag == "option" and self.in_depth_select:
            value = attributes.get("value")
            if value:
                self.values.append(value)

    def handle_endtag(self, tag):
        if tag == "select" and self.in_depth_select:
            self.in_depth_select = False


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "depth_surge_3d_cli_for_test", PROJECT_ROOT / "depth_surge_3d.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_accepts_see_through_model_version():
    cli = _load_cli_module()

    args = cli.create_argument_parser().parse_args(
        ["input.mp4", "--depth-model-version", "see_through"]
    )

    assert args.depth_model_version == "see_through"


def test_web_ui_exposes_see_through_model_option():
    parser = _DepthModelOptionParser()
    parser.feed((PROJECT_ROOT / "templates" / "index.html").read_text(encoding="utf-8"))

    assert "see_through" in parser.values


def test_web_runner_uses_relative_see_through_repo(tmp_path):
    import app as web_app
    from src.depth_surge_3d.inference.depth.video_depth_estimator_see_through import (
        DEFAULT_SEE_THROUGH_REPO,
    )

    projector = MagicMock()
    projector.depth_estimator.load_model.return_value = False

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch.object(
            web_app, "create_stereo_projector", return_value=projector
        ) as create_projector,
    ):
        web_app.process_video_async(
            "test-session",
            tmp_path / "input.mp4",
            {
                "depth_model_version": "see_through",
                "device": "cuda",
                "use_metric_depth": True,
            },
            tmp_path / "output",
        )

    create_projector.assert_called_once_with(
        DEFAULT_SEE_THROUGH_REPO,
        "cuda",
        metric=False,
        depth_model_version="see_through",
    )


def test_see_through_factory_is_public():
    from src.depth_surge_3d.inference import create_see_through_depth_estimator

    assert callable(create_see_through_depth_estimator)


def test_web_resume_restores_saved_processing_settings(tmp_path):
    """Resume must not silently replace See-Through and stereo settings with defaults."""
    import app as web_app

    saved_settings = {
        "depth_model_version": "see_through",
        "depth_resolution": "768",
        "stereo_strength": 3.0,
        "convergence": 0.4,
        "baseline": 0.04,
        "focal_length": 800,
        "apply_distortion": False,
        "keep_intermediates": False,
    }
    (tmp_path / "job-settings.json").write_text(
        json.dumps({"processing_settings": saved_settings}),
        encoding="utf-8",
    )

    result = web_app.detect_resume_settings(tmp_path)

    for key, value in saved_settings.items():
        if key in {"baseline", "focal_length"}:
            continue
        assert result[key] == value
    assert "baseline" not in result
    assert "focal_length" not in result


def test_web_process_validates_final_settings_before_starting(tmp_path):
    import app as web_app

    output_root = tmp_path / "output"
    output_dir = output_root / "job"
    output_dir.mkdir(parents=True)
    source_video = output_dir / "source.mp4"
    source_video.touch()
    web_app.app.config["OUTPUT_FOLDER"] = str(output_root)
    web_app.current_processing.update(
        {"active": False, "session_id": None, "thread": None, "stop_requested": False}
    )
    settings = {
        "stereo_strength": 3.0,
        "convergence": 0.4,
        "occlusion_fill": "none",
        "scene_detection": False,
        "scene_cut_threshold": 0.7,
        "min_scene_frames": 12,
        "raw_storage_dtype": "float32",
        "stereo_io_workers": 3,
        "migrate_legacy": "archive",
    }

    with (
        patch.object(web_app, "find_source_video", return_value=source_video),
        patch.object(web_app.socketio, "start_background_task", return_value=MagicMock()) as start,
    ):
        response = web_app.app.test_client().post(
            "/process",
            json={"output_dir": str(output_dir), "settings": settings},
        )

    assert response.status_code == 200
    validated = start.call_args.args[3]
    for key, value in settings.items():
        assert validated[key] == value


def test_web_process_rejects_removed_explicit_settings(tmp_path):
    import app as web_app

    output_root = tmp_path / "output"
    output_dir = output_root / "job"
    output_dir.mkdir(parents=True)
    source_video = output_dir / "source.mp4"
    source_video.touch()
    web_app.app.config["OUTPUT_FOLDER"] = str(output_root)
    web_app.current_processing.update(
        {"active": False, "session_id": None, "thread": None, "stop_requested": False}
    )

    with (
        patch.object(web_app, "find_source_video", return_value=source_video),
        patch.object(web_app.socketio, "start_background_task") as start,
    ):
        response = web_app.app.test_client().post(
            "/process",
            json={"output_dir": str(output_dir), "settings": {"baseline": 0.065}},
        )

    assert response.status_code == 400
    assert "removed setting" in response.get_json()["error"]
    start.assert_not_called()


def test_web_resume_requires_current_request_to_authorize_legacy_delete(tmp_path):
    import app as web_app

    output_dir = tmp_path / "job"
    output_dir.mkdir()
    source_video = output_dir / "source.mp4"
    source_video.touch()
    web_app.current_processing.update(
        {"active": False, "session_id": None, "thread": None, "stop_requested": False}
    )
    report = MagicMock()
    report.migrated_settings = {"migrate_legacy": "archive"}
    report.to_dict.return_value = {"preserved_stages": ["frames"]}

    with (
        patch.object(web_app, "find_source_video", return_value=source_video),
        patch.object(
            web_app,
            "detect_resume_settings",
            return_value={"migrate_legacy": "delete"},
        ),
        patch.object(web_app, "build_resume_report", return_value=report) as build,
        patch.object(web_app, "apply_legacy_migration") as migrate,
        patch.object(web_app.socketio, "start_background_task", return_value=MagicMock()),
    ):
        response = web_app.app.test_client().post(
            "/resume",
            json={"output_dir": str(output_dir)},
        )

    assert response.status_code == 200
    assert build.call_args.args[1]["migrate_legacy"] == "archive"
    migrate.assert_called_once_with(report, "archive")


def test_cli_resume_restores_depth_backend_without_forwarding_cache_metadata(tmp_path, monkeypatch):
    """CLI resume must rebuild the saved estimator and pass only process-video options."""
    cli = _load_cli_module()
    projector = MagicMock()
    projector.process_video.return_value = True
    saved_settings = {
        "depth_model_version": "see_through",
        "model_path": "24yearsold/custom-marigold",
        "model_size": "large",
        "depth_resolution": "768",
        "use_metric_depth": False,
        "device": "cuda",
        "denoising_steps": 4,
        "seed": 1026,
        "stereo_strength": 3.0,
        "keep_intermediates": False,
        "vr_format": "side_by_side",
    }
    resume_info = {
        "can_resume": True,
        "batch_name": "anime",
        "status": "in_progress",
        "progress_info": None,
        "recommendations": [],
        "settings_file": tmp_path / "job-settings.json",
    }
    monkeypatch.setattr(cli.sys, "argv", ["depth_surge_3d.py", "--resume", str(tmp_path)])

    with (
        patch.object(cli, "can_resume_processing", return_value=resume_info),
        patch.object(
            cli,
            "load_processing_settings",
            return_value={
                "metadata": {"source_video": "source.mkv"},
                "processing_settings": saved_settings,
            },
        ),
        patch.object(cli, "create_stereo_projector", return_value=projector) as create_projector,
    ):
        result = cli.main()

    assert result == 0
    create_projector.assert_called_once_with(
        model_path="24yearsold/custom-marigold",
        device="cuda",
        metric=False,
        depth_model_version="see_through",
    )
    resume_kwargs = projector.process_video.call_args.kwargs
    assert resume_kwargs["video_path"] == "source.mkv"
    assert resume_kwargs["output_dir"] == str(tmp_path)
    assert resume_kwargs["stereo_strength"] == 3.0
    assert resume_kwargs["keep_intermediates"] is False
    for metadata_key in (
        "depth_model_version",
        "model_path",
        "model_size",
        "depth_resolution",
        "use_metric_depth",
        "device",
        "denoising_steps",
        "seed",
    ):
        assert metadata_key not in resume_kwargs
