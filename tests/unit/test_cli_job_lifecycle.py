"""The console owns job selection and exit status; the projector owns rendering."""

import datetime
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def command(cli_module, monkeypatch):
    def run(*args):
        monkeypatch.setattr(sys, "argv", ["depth-surge-3d", *map(str, args)])
        return cli_module.main()

    return run


@pytest.fixture
def projector(cli_module, monkeypatch):
    backend = Mock()
    monkeypatch.setattr(cli_module, "create_stereo_projector", lambda *args, **kwargs: backend)
    monkeypatch.setattr(cli_module, "validate_video_file", lambda path: Path(path).is_file())
    monkeypatch.setattr(cli_module, "_backend_is_available", lambda version: True)
    return backend


def test_new_conversions_never_reuse_the_other_job_directory(
    command, projector, tmp_path, monkeypatch
):
    source = tmp_path / "clip.mp4"
    source.touch()
    fixed = datetime.datetime(2026, 1, 2, 3, 4, 5)
    monkeypatch.setattr(datetime, "datetime", type("Frozen", (), {"now": lambda: fixed}))
    outputs = []

    def render(*, video_path, output_dir, settings):
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        assert video_path == str(source) and settings["target_fps"] == 60
        outputs.append(target)
        (target / "result").write_text(str(len(outputs)))
        return True

    projector.process_video.side_effect = render
    for _ in range(2):
        assert (
            command(source, "--device", "cpu", "--target-fps", 60, "--output-dir", tmp_path / "out")
            == 0
        )
    assert outputs[0] != outputs[1]
    assert [p.parent for p in outputs] == [tmp_path / "out"] * 2
    assert [p.joinpath("result").read_text() for p in outputs] == ["1", "2"]
    assert projector.unload_model.call_count == 2


@pytest.mark.parametrize("outcome", [False, KeyboardInterrupt(), OSError("decoder stopped")])
def test_failed_or_interrupted_render_returns_failure_and_releases_model(
    command, projector, tmp_path, outcome
):
    source = tmp_path / "clip.mp4"
    source.touch()
    if isinstance(outcome, BaseException):
        projector.process_video.side_effect = outcome
    else:
        projector.process_video.return_value = outcome
    assert command(source, "--device", "cpu", "--verbose", "--output-dir", tmp_path / "out") == 1
    projector.unload_model.assert_called_once_with()


def test_model_information_does_not_create_a_job(command, projector, tmp_path, capsys):
    source = tmp_path / "clip.mp4"
    source.touch()
    projector.get_model_info.return_value = {"encoder": "small"}
    assert command(source, "--model-info", "--device", "cpu", "--output-dir", tmp_path / "out") == 0
    assert "encoder: small" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()
    projector.process_video.assert_not_called()
    projector.unload_model.assert_called_once_with()


def test_cache_commands_operate_on_the_selected_cache_only(
    command, cli_module, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(
        cli_module, "create_stereo_projector", lambda *_: pytest.fail("unexpected inference")
    )
    entry = tmp_path / "depth-surge-3d" / "depth_cache" / "job"
    entry.mkdir(parents=True)
    (entry / "depth_000000.png").write_bytes(b"fixture")
    neighbor = tmp_path / "other-application"
    neighbor.mkdir()
    assert command("--cache-info") == 0
    assert "1" in capsys.readouterr().out
    assert command("--cache-clear") == 0
    assert neighbor.is_dir() and not entry.exists()
    assert command("--list-resolutions") == 0
    assert "custom:WxH" in capsys.readouterr().out


@pytest.mark.parametrize(
    "mode, preserved, expected",
    [
        ("vdpp", {"disparity_maps", "disparity_stabilized"}, "stabilized"),
        ("vdpp", {"disparity_maps"}, "base"),
        ("off", {"disparity_maps"}, "base"),
        ("off", {"disparity_stabilized"}, None),
        ("vdpp", set(), None),
    ],
)
def test_resume_uses_only_preserved_frames_in_manifest_order(
    cli_module, tmp_path, mode, preserved, expected
):
    stages = {}
    names = ["frame_000002", "frame_000001"]
    for name in ("disparity_maps", "disparity_stabilized"):
        directory = tmp_path / name
        directory.mkdir()
        obj = {"frame_names": names}
        if name == "disparity_stabilized":
            obj = {"semantic_identity": obj}
        (directory / "metadata.json").write_text(json.dumps(obj))
        stages[name] = SimpleNamespace(
            disposition="preserve" if name in preserved else "redo", paths=[directory]
        )
    result = cli_module._preserved_render_artifact(SimpleNamespace(stage=stages.__getitem__), mode)
    if expected is None:
        assert result is None
    else:
        directory = stages[
            "disparity_stabilized" if expected == "stabilized" else "disparity_maps"
        ].paths[0]
        assert result == ([directory / (name + ".png") for name in names], expected)


@pytest.mark.parametrize("payload", ["invalid", "[]", '{"frame_names":[]}', '{"frame_names":[2]}'])
def test_incomplete_metadata_does_not_select_a_cached_render(cli_module, tmp_path, payload):
    (tmp_path / "metadata.json").write_text(payload)
    report = SimpleNamespace(
        stage=lambda _: SimpleNamespace(disposition="preserve", paths=[tmp_path])
    )
    assert cli_module._preserved_render_artifact(report, "off") is None


@pytest.mark.parametrize("failure", ["no source", "invalid video", "bad fps", "missing backend"])
def test_invalid_new_job_never_loads_a_model_or_creates_output(
    cli_module, command, monkeypatch, tmp_path, failure
):
    source = tmp_path / "clip.mp4"
    source.touch()
    factory = Mock(side_effect=AssertionError("validation must precede projector creation"))
    monkeypatch.setattr(cli_module, "create_stereo_projector", factory)
    monkeypatch.setattr(cli_module, "validate_video_file", lambda _: failure != "invalid video")
    monkeypatch.setattr(cli_module, "_backend_is_available", lambda _: failure != "missing backend")
    args = [] if failure == "no source" else [source]
    if failure == "bad fps":
        args += ["--target-fps", "0"]
    assert command(*args, "--output-dir", tmp_path / "out") == 1
    factory.assert_not_called()
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("outcome", ["verified", "mismatch", "unreadable"])
def test_completed_cli_resume_authenticates_without_opening_the_original_media(
    command, cli_module, monkeypatch, tmp_path, outcome, capsys
):
    control = tmp_path / "processing-settings-v5.json"
    control.touch()
    movie = tmp_path / "finished.mp4"
    movie.write_bytes(b"previous movie")
    monkeypatch.setattr(
        cli_module,
        "can_resume_processing",
        lambda _: {"settings_file": control, "status": "completed"},
    )

    def open_settings(path):
        assert path == control
        if outcome == "unreadable":
            raise OSError("storage unavailable")
        return SimpleNamespace(
            status="completed",
            to_object=lambda: {"output_info": {"expected_final_relative_path": movie.name}},
        )

    monkeypatch.setattr(cli_module, "recover_processing_settings_artifact", open_settings)
    monkeypatch.setattr(
        cli_module,
        "verify_existing_final_video",
        lambda *_: outcome == "verified",
    )
    projector = Mock(side_effect=AssertionError("completed jobs must never render again"))
    monkeypatch.setattr(cli_module, "create_stereo_projector", projector)
    assert command("--resume", tmp_path) == (0 if outcome == "verified" else 1)
    if outcome == "mismatch":
        assert "left untouched" in capsys.readouterr().out
    assert movie.read_bytes() == b"previous movie"
    projector.assert_not_called()
