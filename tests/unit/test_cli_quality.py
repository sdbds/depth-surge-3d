from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture
def cli_module():
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root / "src"))
    try:
        yield importlib.import_module("depth_surge_3d.cli")
    finally:
        sys.path.pop(0)


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
