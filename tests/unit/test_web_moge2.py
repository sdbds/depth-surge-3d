"""Web UI coverage for selecting the optional MoGe-2 backend."""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from src.depth_surge_3d.inference.depth.backend_registry import BackendAvailability


def _video_properties(*, sar=(1, 1)):
    return {
        "width": 1920,
        "height": 1080,
        "fps": 24.0,
        "frame_count": 48,
        "duration": 2.0,
        "sample_aspect_ratio_numerator": sar[0],
        "sample_aspect_ratio_denominator": sar[1],
    }


def _prepare_process_request(web_app, monkeypatch, tmp_path):
    output_dir = tmp_path / "job"
    output_dir.mkdir()
    (output_dir / "source.mp4").write_bytes(b"test")
    web_app.current_processing.update(
        {"active": False, "session_id": None, "thread": None, "stop_requested": False}
    )
    monkeypatch.setitem(web_app.app.config, "OUTPUT_FOLDER", str(tmp_path))
    monkeypatch.setattr(web_app, "backend_availability", lambda _backend: BackendAvailability(True))
    monkeypatch.setattr(
        web_app,
        "get_video_properties",
        lambda _path: _video_properties(),
        raising=False,
    )
    return output_dir


def _metric_payload(output_dir, **overrides):
    settings = {
        "depth_model_version": "moge2",
        "model_size": "vitb",
        "stereo_geometry_mode": "metric_camera",
        "virtual_baseline_mm": 63.0,
        "metric_convergence_distance": "auto",
        "max_disparity_percent": 2.0,
        "vr_format": "side_by_side",
        "apply_distortion": False,
    }
    settings.update(overrides)
    return {"output_dir": str(output_dir), "settings": settings}


@pytest.fixture
def client():
    import app as web_app

    web_app.app.config["TESTING"] = True
    return web_app.app.test_client()


def test_index_marks_unavailable_moge_option_disabled(client, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.backend_availability",
        lambda backend_id: (
            BackendAvailability(False, "missing", "uv sync --extra moge2")
            if backend_id == "moge2"
            else BackendAvailability(True)
        ),
    )
    response = client.get("/")
    html = response.get_data(as_text=True)
    assert 'option value="moge2" disabled' in html
    assert "uv sync --extra moge2" in html


def test_moge_model_sizes_use_vits_vitb_vitl_values(client, monkeypatch) -> None:
    monkeypatch.setattr("app.backend_availability", lambda _backend: BackendAvailability(True))
    html = client.get("/").get_data(as_text=True)
    assert 'data-backend="moge2"' in html
    assert 'data-model-size="vits"' in html
    assert 'data-model-size="vitb"' in html
    assert 'data-model-size="vitl"' in html


def test_saved_disabled_backend_falls_back_before_refreshing_controls(client, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.backend_availability",
        lambda backend_id: BackendAvailability(backend_id != "moge2"),
    )

    html = client.get("/").get_data(as_text=True)
    load_settings = html[
        html.index("function loadSettings()") : html.index("function resetToDefaults()")
    ]

    assert "ensureAvailableDepthBackend();" in load_settings
    assert "option[selected]:not(:disabled)" in html
    assert "option:not(:disabled)" in html
    assert load_settings.index("ensureAvailableDepthBackend();") < load_settings.index(
        "updateDepthModelControls();"
    )


def test_process_rejects_unavailable_moge_before_background_start(
    client, monkeypatch, tmp_path
) -> None:
    import app as web_app

    web_app.current_processing.update(
        {"active": False, "session_id": None, "thread": None, "stop_requested": False}
    )
    monkeypatch.setitem(web_app.app.config, "OUTPUT_FOLDER", str(tmp_path))
    monkeypatch.setattr(
        web_app,
        "backend_availability",
        lambda backend_id: (
            BackendAvailability(
                False,
                "MoGe-2 optional dependency is not installed",
                "uv sync --extra moge2",
            )
            if backend_id == "moge2"
            else BackendAvailability(True)
        ),
    )
    start = MagicMock()
    create_projector = MagicMock()
    monkeypatch.setattr(web_app.socketio, "start_background_task", start)
    monkeypatch.setattr(web_app, "create_stereo_projector", create_projector)

    response = client.post(
        "/process",
        json={
            "output_dir": str(tmp_path / "job"),
            "settings": {"depth_model_version": "moge2"},
        },
    )

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "MoGe-2 optional dependency is not installed. Install with: uv sync --extra moge2"
    }
    start.assert_not_called()
    create_projector.assert_not_called()


def test_web_normalizes_moge_defaults_and_forces_metric_inference() -> None:
    import app as web_app

    settings = web_app._normalize_depth_backend_settings(
        {"depth_model_version": "moge2", "use_metric_depth": False}
    )

    assert settings["depth_model_version"] == "moge2"
    assert settings["model_size"] == "vitb"
    assert settings["model_path"] is None
    assert settings["depth_resolution"] == "auto"
    assert settings["use_metric_depth"] is True


def test_web_normalizes_legacy_v3_variant_name() -> None:
    import app as web_app

    settings = web_app._normalize_depth_backend_settings(
        {"depth_model_version": "v3", "model_size": "large"}
    )

    assert settings["model_size"] == "vitl"


def test_web_runner_passes_normalized_moge_variant_to_projector(tmp_path) -> None:
    import app as web_app

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
                "depth_model_version": "moge2",
                "device": "cuda",
                "use_metric_depth": False,
            },
            tmp_path / "output",
            None,
            _video_properties(),
        )

    create_projector.assert_called_once_with(
        None,
        "cuda",
        metric=True,
        depth_model_version="moge2",
        model_size="vitb",
    )


def test_web_metric_controls_and_payload_use_public_names(client, monkeypatch) -> None:
    monkeypatch.setattr("app.backend_availability", lambda _backend: BackendAvailability(True))

    html = client.get("/").get_data(as_text=True)

    for element_id in (
        "stereoGeometryMode",
        "relativeGeometrySettings",
        "metricGeometrySettings",
        "virtualBaselineMm",
        "metricConvergenceMode",
        "metricConvergenceDistance",
        "maxDisparityPercent",
        "metricExperimentalWarning",
        "effectiveProcessingConfig",
    ):
        assert f'id="{element_id}"' in html
    for setting_name in (
        "stereo_geometry_mode",
        "virtual_baseline_mm",
        "metric_convergence_distance",
        "max_disparity_percent",
    ):
        assert f"{setting_name}:" in html
    assert "MOGE_RESOLUTION_LEVEL" not in html
    assert "moge_resolution_level" not in html


def test_forged_unsupported_metric_backend_is_rejected_before_background_start(
    client, monkeypatch, tmp_path
) -> None:
    import app as web_app

    output_dir = _prepare_process_request(web_app, monkeypatch, tmp_path)
    start = MagicMock()
    create_projector = MagicMock()
    monkeypatch.setattr(web_app.socketio, "start_background_task", start)
    monkeypatch.setattr(web_app, "create_stereo_projector", create_projector)

    response = client.post(
        "/process",
        json=_metric_payload(output_dir, depth_model_version="v3"),
    )

    assert response.status_code == 400
    assert "does not support" in response.get_json()["error"]
    start.assert_not_called()
    create_projector.assert_not_called()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"vr_format": "over_under"}, "vr_format=side_by_side"),
        ({"apply_distortion": True}, "apply_distortion=false"),
    ],
)
def test_forged_metric_projection_constraints_are_rejected_before_background_start(
    client, monkeypatch, tmp_path, overrides, message
) -> None:
    import app as web_app

    output_dir = _prepare_process_request(web_app, monkeypatch, tmp_path)
    start = MagicMock()
    monkeypatch.setattr(web_app.socketio, "start_background_task", start)

    response = client.post("/process", json=_metric_payload(output_dir, **overrides))

    assert response.status_code == 400
    assert message in response.get_json()["error"]
    start.assert_not_called()


def test_non_square_sar_is_rejected_before_background_start(client, monkeypatch, tmp_path) -> None:
    import app as web_app

    output_dir = _prepare_process_request(web_app, monkeypatch, tmp_path)
    monkeypatch.setattr(
        web_app,
        "get_video_properties",
        lambda _path: _video_properties(sar=(4, 3)),
        raising=False,
    )
    start = MagicMock()
    monkeypatch.setattr(web_app.socketio, "start_background_task", start)

    response = client.post("/process", json=_metric_payload(output_dir))

    assert response.status_code == 400
    assert "square-pixel" in response.get_json()["error"]
    start.assert_not_called()


def test_process_resolves_auto_depth_size_and_passes_probed_properties_once(
    client, monkeypatch, tmp_path
) -> None:
    import app as web_app

    output_dir = _prepare_process_request(web_app, monkeypatch, tmp_path)
    probe = MagicMock(return_value=_video_properties())
    start = MagicMock(return_value=object())
    monkeypatch.setattr(web_app, "get_video_properties", probe, raising=False)
    monkeypatch.setattr(web_app.socketio, "start_background_task", start)

    response = client.post("/process", json=_metric_payload(output_dir, depth_resolution="auto"))

    assert response.status_code == 200
    assert probe.call_count == 1
    assert start.call_args.args[3]["depth_resolution"] == 1080
    assert start.call_args.args[6] == _video_properties()


def test_web_resume_restores_moge_variant_and_metric_values(tmp_path) -> None:
    import app as web_app

    saved = {
        "depth_model_version": "moge2",
        "model_size": "vitl",
        "stereo_geometry_mode": "metric_camera",
        "virtual_baseline_mm": 70.0,
        "metric_convergence_distance": 3.0,
        "max_disparity_percent": 1.25,
        "vr_format": "side_by_side",
        "apply_distortion": False,
    }
    settings_file = tmp_path / "job-settings.json"
    settings_file.write_text(
        json.dumps({"metadata": {"source_video": "source.mp4"}, "processing_settings": saved}),
        encoding="utf-8",
    )

    restored = web_app.detect_resume_settings(tmp_path, settings_file=settings_file)

    assert {key: restored[key] for key in saved} == saved


def test_web_emits_configuration_before_model_load(monkeypatch, tmp_path) -> None:
    import app as web_app

    events: list[str] = []
    estimator = MagicMock()
    estimator.repo_id = "Ruicheng/moge-2-vitb-normal"
    estimator.revision = "54ad3a693e61907ea4633d13dec6ee682fa09419"
    estimator.device = "cpu"
    estimator.inference_precision = "float32"
    estimator.resolution_level = 9
    projector = MagicMock(depth_estimator=estimator)
    projector.load_model.side_effect = lambda: events.append("load_model") or False
    monkeypatch.setattr(web_app, "create_stereo_projector", lambda *_args, **_kwargs: projector)
    monkeypatch.setattr(
        web_app.socketio,
        "emit",
        lambda event, *_args, **_kwargs: events.append(event),
    )
    monkeypatch.setattr(web_app.socketio, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        web_app, "get_video_properties", lambda _path: _video_properties(), raising=False
    )

    with patch("torch.cuda.is_available", return_value=False):
        web_app.process_video_async(
            "test-session",
            tmp_path / "input.mp4",
            _metric_payload(tmp_path)["settings"],
            tmp_path / "output",
            None,
            _video_properties(),
        )

    assert events.index("processing_configuration") < events.index("load_model")


def test_join_after_process_response_replays_exact_processing_configuration(
    client, monkeypatch, tmp_path
) -> None:
    import app as web_app

    output_dir = _prepare_process_request(web_app, monkeypatch, tmp_path)
    estimator = MagicMock()
    estimator.repo_id = "Ruicheng/moge-2-vitb-normal"
    estimator.revision = "54ad3a693e61907ea4633d13dec6ee682fa09419"
    estimator.device = "cpu"
    estimator.inference_precision = "float32"
    estimator.resolution_level = 9
    projector = MagicMock(depth_estimator=estimator)
    projector.load_model.return_value = False
    monkeypatch.setattr(web_app, "create_stereo_projector", lambda *_args, **_kwargs: projector)

    def run_synchronously(target, *args):
        target(*args)
        return object()

    monkeypatch.setattr(web_app.socketio, "start_background_task", run_synchronously)
    response = client.post("/process", json=_metric_payload(output_dir))
    session_id = response.get_json()["session_id"]
    socket_client = web_app.socketio.test_client(web_app.app, flask_test_client=client)
    try:
        socket_client.emit("join_session", {"session_id": session_id})
        configurations = [
            event["args"][0]
            for event in socket_client.get_received()
            if event["name"] == "processing_configuration"
        ]
    finally:
        socket_client.disconnect()

    assert configurations == [
        {
            "backend": "moge2",
            "model_size": "vitb",
            "repository": "Ruicheng/moge-2-vitb-normal",
            "revision": "54ad3a693e61907ea4633d13dec6ee682fa09419",
            "device": "cpu",
            "precision": "float32",
            "depth_resolution": 1080,
            "adapter_resolution_level": 9,
            "camera_capability": "pinhole_fx",
            "geometry_mode": "metric_camera",
            "projection": {
                "virtual_baseline_mm": 63.0,
                "metric_convergence_distance": "auto",
                "max_disparity_percent": 2.0,
            },
        }
    ]


def test_new_runs_clear_stale_effective_configuration(client) -> None:
    html = client.get("/").get_data(as_text=True)

    assert "function clearEffectiveProcessingConfig()" in html
    process_handler = html[html.index("// Start processing") : html.index("// Resume processing")]
    resume_handler = html[html.index("// Resume processing") : html.index("// Stop processing")]
    assert "clearEffectiveProcessingConfig();" in process_handler
    assert "clearEffectiveProcessingConfig();" in resume_handler
    assert "values.replaceChildren();" in html
    assert "container.hidden = true;" in html
