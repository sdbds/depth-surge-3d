from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


def test_publication_resume_never_repairs_a_job_owned_by_another_writer(
    cli_module, tmp_path, monkeypatch
):
    settings_file = tmp_path / "processing-settings-v5.json"
    info = {"settings_file": settings_file, "status": "completed"}
    monkeypatch.setattr(
        cli_module,
        "recover_processing_settings_artifact",
        lambda *_: pytest.fail("recovered without the job lock"),
    )
    with cli_module.JobWriterLock(tmp_path):
        assert cli_module._finish_saved_publication(tmp_path, info) is False


@pytest.mark.parametrize("authenticated", [False, True])
def test_completed_resume_requires_positive_media_audit(
    cli_module, tmp_path, monkeypatch, authenticated
):
    settings_file = tmp_path / "processing-settings-v5.json"
    info = {"settings_file": settings_file, "status": "completed"}
    monkeypatch.setattr(
        cli_module,
        "recover_processing_settings_artifact",
        lambda _: SimpleNamespace(status="completed"),
    )
    monkeypatch.setattr(
        cli_module,
        "audit_final_media_v1",
        lambda *_: SimpleNamespace(final_video_authenticated=authenticated),
    )
    assert cli_module._finish_saved_publication(tmp_path, info) is authenticated


def test_cli_accepts_explicit_quality_controls(cli_module) -> None:
    args = cli_module.create_argument_parser().parse_args(
        [
            "input.mp4",
            "--stereo-render-mode",
            "quality",
            "--occlusion-fill-max-px",
            "10",
            "--device",
            "cpu",
        ]
    )

    settings = cli_module._build_processing_settings(args)

    assert settings["stereo_render_mode"] == "quality"
    assert settings["occlusion_fill_max_px"] == 10


def test_cli_refuses_unimplemented_motion_interpolation(cli_module, monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["depth-surge-3d", "input.mp4", "--experimental-frame-interpolation"]
    )
    with pytest.raises(SystemExit) as error:
        cli_module.main()
    assert error.value.code == 2
    assert "motion interpolation" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_cli_defaults_omitted_mode_to_fast_on_every_device(
    monkeypatch: pytest.MonkeyPatch,
    cli_module,
    device: str,
) -> None:
    monkeypatch.setattr(cli_module, "_effective_vdpp_device", lambda _requested: device)
    args = cli_module.create_argument_parser().parse_args(["input.mp4"])

    settings = cli_module._build_processing_settings(args)

    assert settings["stereo_render_mode"] == "fast"


def test_cli_rejects_invalid_quality_limit_before_processing(cli_module) -> None:
    parser = cli_module.create_argument_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["input.mp4", "--occlusion-fill-max-px", "0"])


def test_cli_omitted_quality_limit_uses_new_job_default(cli_module) -> None:
    args = cli_module.create_argument_parser().parse_args(["input.mp4"])

    settings = cli_module._build_processing_settings(args)

    assert settings["occlusion_fill_max_px"] == 8


def test_cli_defaults_to_cleanup_intermediates(cli_module) -> None:
    parser = cli_module.create_argument_parser()

    omitted = cli_module._build_processing_settings(parser.parse_args(["input.mp4"]))
    kept = cli_module._build_processing_settings(
        parser.parse_args(["input.mp4", "--keep-intermediates"])
    )
    discarded = cli_module._build_processing_settings(
        parser.parse_args(["input.mp4", "--no-intermediates"])
    )

    assert omitted["keep_intermediates"] is False
    assert kept["keep_intermediates"] is True
    assert discarded["keep_intermediates"] is False


def test_cli_direct_vr_encoding_defaults_on_and_can_be_disabled(cli_module) -> None:
    parser = cli_module.create_argument_parser()

    omitted = cli_module._build_processing_settings(parser.parse_args(["input.mp4"]))
    direct = cli_module._build_processing_settings(
        parser.parse_args(["input.mp4", "--direct-vr-encode"])
    )
    assembled = cli_module._build_processing_settings(
        parser.parse_args(["input.mp4", "--no-direct-vr-encode"])
    )

    assert omitted["direct_vr_encode"] is True
    assert direct["direct_vr_encode"] is True
    assert assembled["direct_vr_encode"] is False


def test_cli_resume_stereo_overrides_are_presence_aware(cli_module) -> None:
    parser = cli_module.create_argument_parser()
    saved = {"stereo_render_mode": "quality", "occlusion_fill_max_px": 11}

    omitted = cli_module._apply_resume_stereo_settings(
        saved,
        parser.parse_args(["--resume", "job"]),
    )
    explicit = cli_module._apply_resume_stereo_settings(
        saved,
        parser.parse_args(
            [
                "--resume",
                "job",
                "--stereo-render-mode",
                "fast",
                "--occlusion-fill-max-px",
                "6",
            ]
        ),
    )

    assert omitted["stereo_render_mode"] == "quality"
    assert omitted["occlusion_fill_max_px"] == 11
    assert explicit["stereo_render_mode"] == "fast"
    assert explicit["occlusion_fill_max_px"] == 6


def test_cli_warns_when_quality_is_explicitly_selected_on_cpu(cli_module, capsys) -> None:
    cli_module._warn_quality_on_cpu({"stereo_render_mode": "quality"}, renderer_device_type="cpu")

    assert "Quality stereo on CPU" in capsys.readouterr().out
