"""Tests for See-Through model selection through public entry points."""

from __future__ import annotations

import importlib.util
import json
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from src.depth_surge_3d.core.file_identity import (
    FILE_IDENTITY_ALGORITHM_VERSION,
    file_sample_fingerprint,
)


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


def test_web_ui_does_not_force_see_through_depth_resolution():
    html = (PROJECT_ROOT / "templates" / "index.html").read_text(encoding="utf-8")

    assert "document.getElementById('depthResolution').value = '768';" not in html
    assert "See-Through Native - 768x768 (Square)" in html
    assert "depthModelVersion: document.getElementById('depthModelVersion').value" in html
    assert "depthResolution: document.getElementById('depthResolution').value" in html


def test_web_runner_uses_relative_see_through_repo(tmp_path):
    import app as web_app
    from src.depth_surge_3d.inference.depth.video_depth_estimator_see_through import (
        DEFAULT_SEE_THROUGH_REPO,
    )

    projector = MagicMock()
    projector.load_model.return_value = False

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_name", return_value="Test CUDA device"),
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


def test_web_resume_infers_see_through_for_legacy_settings(tmp_path):
    """Legacy settings without a backend field must recover See-Through from its repo."""
    import app as web_app

    (tmp_path / "job-settings.json").write_text(
        json.dumps(
            {
                "processing_settings": {
                    "model_path": "24yearsold/seethroughv0.0.1_marigold",
                    "model_size": "large",
                }
            }
        ),
        encoding="utf-8",
    )

    result = web_app.detect_resume_settings(tmp_path)

    assert result["depth_model_version"] == "see_through"


def test_web_resume_preserves_direct_vr_setting_and_defaults_older_jobs_off(tmp_path):
    import app as web_app

    direct_dir = tmp_path / "direct"
    legacy_dir = tmp_path / "legacy"
    direct_dir.mkdir()
    legacy_dir.mkdir()
    direct_settings = direct_dir / "job-settings.json"
    legacy_settings = legacy_dir / "job-settings.json"
    direct_settings.write_text(
        json.dumps({"processing_settings": {"direct_vr_encode": True}}),
        encoding="utf-8",
    )
    legacy_settings.write_text(
        json.dumps({"processing_settings": {}}),
        encoding="utf-8",
    )

    assert (
        web_app.detect_resume_settings(direct_dir, settings_file=direct_settings)[
            "direct_vr_encode"
        ]
        is True
    )
    assert (
        web_app.detect_resume_settings(legacy_dir, settings_file=legacy_settings)[
            "direct_vr_encode"
        ]
        is False
    )


def test_resume_depth_model_inference_uses_raw_metadata_and_preserves_explicit_value(tmp_path):
    from src.depth_surge_3d.io.resume import resolve_resume_depth_model_version

    assert resolve_resume_depth_model_version({}, tmp_path, default="v2") == "v2"

    raw_dir = tmp_path / "02_depth_raw"
    raw_dir.mkdir()
    (raw_dir / "metadata.json").write_text(
        json.dumps(
            {
                "semantic_fingerprint": {
                    "backend": (
                        "depth_surge_3d.inference.depth."
                        "video_depth_estimator_see_through.SeeThroughDepthEstimator"
                    ),
                    "model_info": {"model_version": "See-Through Marigold"},
                }
            }
        ),
        encoding="utf-8",
    )

    assert resolve_resume_depth_model_version({}, tmp_path, default="v3") == "see_through"
    assert (
        resolve_resume_depth_model_version(
            {"depth_model_version": "v2"},
            tmp_path,
            default="v3",
        )
        == "v2"
    )


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


def test_web_resume_requires_current_request_to_authorize_legacy_delete(tmp_path, monkeypatch):
    import app as web_app

    output_dir = tmp_path / "job"
    output_dir.mkdir()
    source_video = output_dir / "source.mp4"
    source_video.write_bytes(b"source-video")
    settings_file = output_dir / "job-settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "metadata": {
                    "source_video": str(source_video),
                    "source_video_name": source_video.name,
                    "source_video_fingerprint_algorithm": FILE_IDENTITY_ALGORITHM_VERSION,
                    "source_video_fingerprint": file_sample_fingerprint(source_video),
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(web_app.app.config, "OUTPUT_FOLDER", str(tmp_path))
    web_app.current_processing.update(
        {"active": False, "session_id": None, "thread": None, "stop_requested": False}
    )
    report = MagicMock()
    report.migrated_settings = {"migrate_legacy": "archive"}
    report.settings_file = settings_file
    report.to_dict.return_value = {"preserved_stages": ["frames"]}

    with (
        patch.object(
            web_app,
            "detect_resume_settings",
            return_value={"migrate_legacy": "delete"},
        ),
        patch.object(web_app, "build_resume_report", return_value=report) as build,
        patch.object(web_app, "apply_legacy_migration") as migrate,
        patch.object(web_app.socketio, "start_background_task", return_value=MagicMock()) as start,
    ):
        response = web_app.app.test_client().post(
            "/resume",
            json={
                "output_dir": str(output_dir),
                "raw_storage_dtype": "float32",
            },
        )

    assert response.status_code == 200
    assert build.call_args.args[1]["migrate_legacy"] == "archive"
    assert build.call_args.args[1]["raw_storage_dtype"] == "float32"
    assert build.call_args.kwargs["source_video"] == source_video
    assert build.call_args.kwargs["settings_file"] == settings_file
    migrate.assert_not_called()
    resume_context = start.call_args.args[5]
    assert resume_context == {
        "migration_mode": "archive",
        "settings_file": settings_file,
    }


def test_web_background_resume_validates_loaded_model_before_migration(tmp_path):
    import app as web_app

    output_dir = tmp_path / "job"
    output_dir.mkdir()
    source_video = output_dir / "source.mp4"
    source_video.touch()
    settings_file = output_dir / "job-settings.json"
    settings_file.write_text("{}", encoding="utf-8")
    settings = {
        "depth_model_version": "v3",
        "model_size": "large",
        "device": "cpu",
        "use_metric_depth": False,
    }
    projector = MagicMock()
    projector.load_model.return_value = True
    fingerprint = {"backend": "loaded.Estimator"}
    report = MagicMock()
    report.migrated_settings = settings

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch.object(web_app, "create_stereo_projector", return_value=projector),
        patch.object(
            web_app,
            "build_current_model_fingerprint",
            return_value=fingerprint,
        ) as build_fingerprint,
        patch.object(web_app, "build_resume_report", return_value=report) as build_report,
        patch.object(web_app, "apply_legacy_migration") as migrate,
        patch.object(web_app, "get_video_info", return_value=None),
    ):
        web_app.process_video_async(
            "test-session",
            source_video,
            settings,
            output_dir,
            {"migration_mode": "archive", "settings_file": settings_file},
        )

    build_fingerprint.assert_called_once_with(projector.depth_estimator, settings)
    assert build_report.call_count == 2
    build_report.assert_any_call(
        output_dir,
        settings,
        source_video=source_video,
        settings_file=settings_file,
    )
    build_report.assert_called_with(
        output_dir,
        settings,
        source_video=source_video,
        model_fingerprint=fingerprint,
        settings_file=settings_file,
    )
    migrate.assert_called_once_with(report, "archive")


def test_web_complete_stabilized_resume_skips_cuda_and_base_model(tmp_path):
    import app as web_app

    output_dir = tmp_path / "job"
    output_dir.mkdir()
    source_video = output_dir / "source.mp4"
    source_video.touch()
    settings = web_app.validate_settings(
        {
            "temporal_postprocessor": "vdpp",
            "device": "cpu",
            "vr_format": "side_by_side",
            "vr_resolution": "16x9-1080p",
        },
        source="legacy_disk",
    )
    report = MagicMock()
    report.migrated_settings = settings
    processor = MagicMock()
    lock_states = []

    def observe_lock(**kwargs):
        lock_states.append(kwargs["job_lock"].is_acquired)
        return True

    processor.process.side_effect = observe_lock
    stable_files = [output_dir / "03_disparity_stabilized" / "frame_000001.png"]

    with (
        patch.object(web_app, "build_resume_report", return_value=report),
        patch.object(
            web_app,
            "_preserved_render_artifact",
            return_value=(stable_files, "stabilized"),
        ),
        patch.object(
            web_app,
            "create_stereo_projector",
            side_effect=AssertionError("base model must stay lazy"),
        ) as create_projector,
        patch("torch.cuda.is_available", side_effect=AssertionError("CUDA must not be probed")),
        patch.object(
            web_app,
            "get_video_info",
            return_value={"fps": 24.0, "frame_count": 1, "width": 64, "height": 48},
        ),
        patch.object(web_app, "VideoProcessor", return_value=processor) as processor_class,
        patch.object(web_app, "apply_legacy_migration"),
        patch.object(web_app.socketio, "emit"),
        patch.object(web_app.socketio, "sleep"),
    ):
        web_app.process_video_async(
            "test-session",
            source_video,
            settings,
            output_dir,
            {"migration_mode": "archive", "settings_file": output_dir / "settings.json"},
        )

    create_projector.assert_not_called()
    processor_class.assert_called_once()
    assert processor_class.call_args.args[0] is None
    assert lock_states == [True]


def test_web_resume_rejects_output_directory_outside_managed_root(tmp_path, monkeypatch):
    import app as web_app

    output_root = tmp_path / "managed"
    output_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    web_app.current_processing.update(
        {"active": False, "session_id": None, "thread": None, "stop_requested": False}
    )
    monkeypatch.setitem(web_app.app.config, "OUTPUT_FOLDER", str(output_root))

    with (
        patch.object(web_app, "find_source_video") as find_source,
        patch.object(web_app, "build_resume_report") as build,
        patch.object(web_app, "apply_legacy_migration") as migrate,
        patch.object(web_app.socketio, "start_background_task") as start,
    ):
        response = web_app.app.test_client().post(
            "/resume",
            json={"output_dir": str(outside), "migrate_legacy": "delete"},
        )

    assert response.status_code == 403
    find_source.assert_not_called()
    build.assert_not_called()
    migrate.assert_not_called()
    start.assert_not_called()


def test_web_resume_accepts_legacy_settings_without_source_fingerprint(tmp_path, monkeypatch):
    import json

    import app as web_app

    output_root = tmp_path / "managed"
    output_dir = output_root / "job"
    output_dir.mkdir(parents=True)
    correct_source = output_dir / "source.avi"
    correct_source.write_bytes(b"correct-source")
    (output_dir / "distractor.mp4").write_bytes(b"unrelated-video")
    settings_file = output_dir / "job-settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "metadata": {
                    "source_video": str(correct_source),
                    "source_video_name": correct_source.name,
                    "settings_schema_version": 2,
                },
                "processing_settings": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(web_app.app.config, "OUTPUT_FOLDER", str(output_root))
    web_app.current_processing.update(
        {"active": False, "session_id": None, "thread": None, "stop_requested": False}
    )
    report = MagicMock()
    report.migrated_settings = {}
    report.to_dict.return_value = {}

    with (
        patch.object(web_app, "build_resume_report", return_value=report),
        patch.object(web_app.socketio, "start_background_task", return_value=MagicMock()) as start,
    ):
        response = web_app.app.test_client().post(
            "/resume",
            json={"output_dir": str(output_dir)},
        )

    assert response.status_code == 200
    assert start.call_args.args[2] == correct_source.resolve()


def test_web_resume_rejects_saved_source_fingerprint_mismatch(tmp_path, monkeypatch):
    import json

    import app as web_app

    output_root = tmp_path / "managed"
    output_dir = output_root / "job"
    output_dir.mkdir(parents=True)
    source_video = output_dir / "source.avi"
    source_video.write_bytes(b"changed-source")
    (output_dir / "distractor.mp4").write_bytes(b"unrelated-video")
    (output_dir / "job-settings.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "source_video": str(source_video),
                    "source_video_name": source_video.name,
                    "source_video_fingerprint_algorithm": FILE_IDENTITY_ALGORITHM_VERSION,
                    "source_video_fingerprint": "0" * 64,
                    "settings_schema_version": 2,
                },
                "processing_settings": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(web_app.app.config, "OUTPUT_FOLDER", str(output_root))
    web_app.current_processing.update(
        {"active": False, "session_id": None, "thread": None, "stop_requested": False}
    )

    with patch.object(web_app.socketio, "start_background_task") as start:
        response = web_app.app.test_client().post(
            "/resume",
            json={"output_dir": str(output_dir)},
        )

    assert response.status_code == 409
    assert "source video" in response.get_json()["error"].lower()
    start.assert_not_called()


def test_cli_resume_restores_depth_backend_without_forwarding_cache_metadata(tmp_path, monkeypatch):
    """CLI resume must rebuild the saved estimator and pass only process-video options."""
    cli = _load_cli_module()
    projector = MagicMock()
    projector.load_model.return_value = True
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
    report = MagicMock()
    report.stages = ()
    report.removed_settings = ()
    report.migrated_settings = cli.validate_settings(saved_settings, source="legacy_disk")
    model_fingerprint = {"backend": "loaded.Estimator"}
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
        patch.object(
            cli,
            "build_current_model_fingerprint",
            return_value=model_fingerprint,
        ) as build_fingerprint,
        patch.object(cli, "build_resume_report", return_value=report) as build_report,
        patch.object(cli, "apply_legacy_migration") as migrate,
    ):
        result = cli.main()

    assert result == 0
    create_projector.assert_called_once_with(
        model_path="24yearsold/custom-marigold",
        device="cuda",
        metric=False,
        depth_model_version="see_through",
    )
    projector.load_model.assert_called_once_with()
    preflight_settings = {**report.migrated_settings, "verbose": False}
    build_fingerprint.assert_called_once_with(projector.depth_estimator, preflight_settings)
    assert build_report.call_args_list == [
        call(
            tmp_path,
            preflight_settings,
            source_video="source.mkv",
            settings_file=resume_info["settings_file"],
        ),
        call(
            tmp_path,
            preflight_settings,
            source_video="source.mkv",
            model_fingerprint=model_fingerprint,
            settings_file=resume_info["settings_file"],
        ),
    ]
    migrate.assert_called_once_with(report, "archive")
    resume_kwargs = projector.process_video.call_args.kwargs
    assert resume_kwargs["video_path"] == "source.mkv"
    assert resume_kwargs["output_dir"] == str(tmp_path)
    assert resume_kwargs["settings"]["stereo_strength"] == 3.0
    assert resume_kwargs["settings"]["keep_intermediates"] is False
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


def test_cli_resume_infers_see_through_for_legacy_settings(tmp_path, monkeypatch):
    """CLI resume must select See-Through when old settings only contain its repo."""
    cli = _load_cli_module()
    projector = MagicMock()
    projector.load_model.return_value = True
    projector.process_video.return_value = True
    saved_settings = {
        "model_path": "24yearsold/seethroughv0.0.1_marigold",
        "model_size": "large",
        "use_metric_depth": False,
        "device": "cuda",
        "stereo_strength": 3.0,
        "keep_intermediates": False,
    }
    resume_info = {
        "can_resume": True,
        "batch_name": "anime",
        "status": "in_progress",
        "progress_info": None,
        "recommendations": [],
        "settings_file": tmp_path / "job-settings.json",
    }
    report = MagicMock()
    report.stages = ()
    report.removed_settings = ()
    report.migrated_settings = cli.validate_settings(
        {**saved_settings, "depth_model_version": "see_through"},
        source="legacy_disk",
    )
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
        patch.object(cli, "build_current_model_fingerprint", return_value={"backend": "loaded"}),
        patch.object(cli, "build_resume_report", return_value=report),
        patch.object(cli, "apply_legacy_migration"),
    ):
        result = cli.main()

    assert result == 0
    create_projector.assert_called_once_with(
        model_path="24yearsold/seethroughv0.0.1_marigold",
        device="cuda",
        metric=False,
        depth_model_version="see_through",
    )


def test_cli_complete_stabilized_resume_skips_cuda_and_base_model(tmp_path, monkeypatch):
    cli = _load_cli_module()
    source_video = tmp_path / "source.mp4"
    source_video.touch()
    saved_settings = {
        "temporal_postprocessor": "vdpp",
        "device": "cpu",
        "vr_format": "side_by_side",
        "vr_resolution": "16x9-1080p",
    }
    resume_info = {
        "can_resume": True,
        "batch_name": "cached",
        "status": "in_progress",
        "progress_info": None,
        "recommendations": [],
        "settings_file": tmp_path / "job-settings.json",
    }
    report = MagicMock()
    report.stages = ()
    report.removed_settings = ()
    report.migrated_settings = cli.validate_settings(saved_settings, source="legacy_disk")
    processor = MagicMock()
    observed_locks = []

    def observe_lock(**kwargs):
        observed_locks.append(kwargs["job_lock"].is_acquired)
        return True

    processor.process.side_effect = observe_lock
    stable_files = [tmp_path / "03_disparity_stabilized" / "frame_000001.png"]
    monkeypatch.setattr(cli.sys, "argv", ["depth_surge_3d.py", "--resume", str(tmp_path)])

    with (
        patch.object(cli, "can_resume_processing", return_value=resume_info),
        patch.object(
            cli,
            "load_processing_settings",
            return_value={
                "metadata": {"source_video": str(source_video)},
                "processing_settings": saved_settings,
                "video_properties": {"fps": 24.0, "frame_count": 1},
            },
        ),
        patch.object(cli, "build_resume_report", return_value=report),
        patch.object(
            cli,
            "_preserved_render_artifact",
            return_value=(stable_files, "stabilized"),
        ),
        patch.object(
            cli,
            "create_stereo_projector",
            side_effect=AssertionError("base model must stay lazy"),
        ) as create_projector,
        patch("torch.cuda.is_available", side_effect=AssertionError("CUDA must not be probed")),
        patch.object(cli, "apply_legacy_migration"),
        patch(
            "depth_surge_3d.processing.orchestration.video_processor.VideoProcessor",
            return_value=processor,
        ) as processor_class,
    ):
        assert cli.main() == 0

    create_projector.assert_not_called()
    processor_class.assert_called_once()
    assert processor_class.call_args.args[0] is None
    assert observed_locks == [True]
