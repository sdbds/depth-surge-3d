from __future__ import annotations

import pytest


@pytest.fixture
def web_app():
    import app

    app.app.config["TESTING"] = True
    return app


def test_web_quality_controls_are_visible_and_submitted(web_app) -> None:
    html = web_app.app.test_client().get("/").get_data(as_text=True)

    assert 'id="stereoRenderMode"' in html
    assert '<option value="quality"' in html
    assert '<option value="fast"' in html
    assert 'id="occlusionFillMaxPx"' in html
    assert 'min="1" max="32"' in html
    assert 'id="stereoQualityCpuWarning"' in html
    assert "stereo_render_mode: document.getElementById('stereoRenderMode').value" in html
    assert "occlusion_fill_max_px: parseInt(" in html
    assert "let stereoRenderModeExplicit = false;" in html
    assert "document.getElementById('stereoRenderMode').value = 'fast'" in html
    assert "stereoRenderModeExplicit = true;" in html


@pytest.mark.parametrize(("device", "expected"), [("cuda", "fast"), ("cpu", "fast")])
def test_web_omitted_mode_resolves_from_effective_device(
    web_app, monkeypatch: pytest.MonkeyPatch, device: str, expected: str
) -> None:
    monkeypatch.setattr(web_app, "_effective_web_renderer_device", lambda _settings: device)

    settings = web_app._resolve_new_web_stereo_settings({"device": "auto"})

    assert settings["stereo_render_mode"] == expected
    assert settings["occlusion_fill_max_px"] == 8


def test_web_explicit_quality_mode_wins_on_cpu(web_app, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_app, "_effective_web_renderer_device", lambda _settings: "cpu")

    settings = web_app._resolve_new_web_stereo_settings(
        {
            "device": "auto",
            "stereo_render_mode": "quality",
            "occlusion_fill_max_px": 9,
        }
    )

    assert settings["stereo_render_mode"] == "quality"
    assert settings["occlusion_fill_max_px"] == 9


def test_web_resume_override_parser_preserves_omission(web_app, tmp_path) -> None:
    web_app.app.config["OUTPUT_FOLDER"] = str(tmp_path)
    job = tmp_path / "job"
    job.mkdir()

    omitted = web_app._prepare_resume_request({"output_dir": str(job)})
    explicit = web_app._prepare_resume_request(
        {
            "output_dir": str(job),
            "stereo_render_mode": "fast",
            "occlusion_fill_max_px": 7,
        }
    )

    assert omitted[-2:] == (None, None)
    assert explicit[-2:] == ("fast", 7)


def test_web_resume_stereo_resolution_preserves_saved_and_migrates_legacy(
    web_app, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(web_app, "_effective_web_renderer_device", lambda _settings: "cuda")

    preserved = web_app._resolve_resume_web_stereo_settings(
        {"stereo_render_mode": "quality", "occlusion_fill_max_px": 11},
        stereo_override=None,
        fill_limit_override=None,
    )
    legacy = web_app._resolve_resume_web_stereo_settings(
        {}, stereo_override=None, fill_limit_override=None
    )
    overridden = web_app._resolve_resume_web_stereo_settings(
        {"stereo_render_mode": "quality", "occlusion_fill_max_px": 11},
        stereo_override="fast",
        fill_limit_override=6,
    )

    assert (preserved["stereo_render_mode"], preserved["occlusion_fill_max_px"]) == (
        "quality",
        11,
    )
    assert (legacy["stereo_render_mode"], legacy["occlusion_fill_max_px"]) == (
        "fast",
        8,
    )
    assert (overridden["stereo_render_mode"], overridden["occlusion_fill_max_px"]) == (
        "fast",
        6,
    )
