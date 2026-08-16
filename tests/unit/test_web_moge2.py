"""Web UI coverage for selecting the optional MoGe-2 backend."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.depth_surge_3d.inference.depth.backend_registry import BackendAvailability


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
        )

    create_projector.assert_called_once_with(
        None,
        "cuda",
        metric=True,
        depth_model_version="moge2",
        model_size="vitb",
    )
