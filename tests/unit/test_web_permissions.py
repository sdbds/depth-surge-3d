"""Local desktop transport, managed jobs, and non-destructive Web exports."""

import io
import json
import runpy
import shutil
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

import app as web


@pytest.fixture
def desktop(tmp_path, monkeypatch):
    root = tmp_path / "output"
    job = root / "session"
    job.mkdir(parents=True)
    monkeypatch.setitem(web.app.config, "OUTPUT_FOLDER", str(root))
    monkeypatch.setitem(web.current_processing, "active", False)
    monkeypatch.setattr(web, "cleanup_processes", lambda: None)
    opener = Mock()
    monkeypatch.setattr(web, "platform", SimpleNamespace(system=lambda: "Windows"))
    monkeypatch.setattr(web.os, "startfile", opener, raising=False)
    return web.app.test_client(), job, opener


@pytest.mark.parametrize("args, allowed", [([], True), (["--host", "0.0.0.0"], False)])
def test_launcher_only_binds_loopback(args, allowed, tmp_path, monkeypatch):
    run = Mock()
    monkeypatch.setattr(web.SocketIO, "run", run)
    monkeypatch.setattr(signal, "signal", lambda *_: None)
    monkeypatch.setattr(sys, "argv", ["app.py", *args])
    monkeypatch.chdir(tmp_path)
    if not allowed:
        with pytest.raises(SystemExit) as rejected:
            runpy.run_path(web.__file__, run_name="__main__")
        assert rejected.value.code == 2
        run.assert_not_called()
        return
    runpy.run_path(web.__file__, run_name="__main__")
    assert run.call_args.kwargs["host"] == "127.0.0.1"


@pytest.mark.parametrize("route", ["/detect_resumable", "/socket.io/?EIO=4&transport=polling"])
def test_browser_requests_must_come_from_the_local_ui(desktop, route):
    client, _, _ = desktop
    url = "http://localhost:5050"
    for origin, status in ((url, 200), ("http://localhost:5051", 403)):
        request = client.options if route.startswith("/socket.io/") else client.get
        response = request(route, base_url=url, headers={"Origin": origin})
        assert response.status_code == status


def test_only_local_hosts_and_peers_are_served(desktop):
    client, _, _ = desktop
    assert client.get("/detect_resumable", base_url="http://workstation.example").status_code == 403
    response = client.get("/detect_resumable", environ_overrides={"REMOTE_ADDR": "192.0.2.1"})
    assert response.status_code == 403


def test_status_polling_never_triggers_process_cleanup(monkeypatch):
    monkeypatch.setattr(web, "SHUTDOWN_FLAG", False)
    monkeypatch.setattr(web, "ACTIVE_PROCESSES", set())
    monkeypatch.setitem(web.current_processing, "thread", None)
    external_command = Mock()
    monkeypatch.setattr(web.subprocess, "run", external_command)
    client = web.app.test_client()
    for _ in range(3):
        assert client.get("/status").status_code == 200
    assert not web.SHUTDOWN_FLAG
    external_command.assert_not_called()


def test_shutdown_stops_only_the_desktop_services_own_media_child(monkeypatch):
    monkeypatch.setattr(web, "SHUTDOWN_FLAG", False)
    monkeypatch.setattr(web, "ACTIVE_PROCESSES", set())
    external_command = Mock()
    monkeypatch.setattr(web.subprocess, "run", external_command)
    waiting = Event()
    run_media = web._run_web_media
    sleeper = [sys.executable, "-c", "import time; time.sleep(15)"]
    other_job = subprocess.Popen(sleeper)

    class MediaChild(subprocess.Popen):
        def communicate(self, *args, **kwargs):
            waiting.set()
            return super().communicate(*args, **kwargs)

    monkeypatch.setattr(web.subprocess, "Popen", MediaChild)
    try:
        with ThreadPoolExecutor(1) as pool:
            own_job = pool.submit(run_media, sleeper, timeout=10)
            assert waiting.wait(timeout=5)
            web.cleanup_processes()
            assert own_job.result(timeout=5).returncode != 0
        assert other_job.poll() is None
        assert not web.ACTIVE_PROCESSES
        external_command.assert_not_called()
    finally:
        other_job.terminate()
        other_job.wait(timeout=5)


@pytest.mark.parametrize(
    "route, field",
    [
        ("/open_directory", "path"),
        ("/analyze_batch", "batch_dir"),
        ("/stitch_video", "batch_dir"),
        ("/process", "output_dir"),
        ("/resume", "output_dir"),
    ],
)
def test_job_operations_require_a_managed_job_not_the_collection(desktop, route, field):
    client, job, opener = desktop
    response = client.post(route, json={field: str(job.parent), "settings": {"device": "cpu"}})
    assert response.status_code == 403
    opener.assert_not_called()


def test_open_only_dispatches_a_managed_directory(desktop, tmp_path):
    client, job, opener = desktop
    other = tmp_path / "personal"
    other.mkdir()
    regular_file = job.parent / "notes.txt"
    regular_file.write_text("not a folder")
    for path in (other, regular_file):
        response = client.post("/open_directory", json={"path": str(path)})
        assert 400 <= response.status_code < 500
        assert response.get_json()["success"] is False
    opener.assert_not_called()
    assert client.post("/open_directory", json={"path": str(job)}).get_json()["success"]
    assert Path(opener.call_args.args[0]) == job


def test_job_operations_do_not_follow_linked_frame_directories(desktop, tmp_path):
    client, job, _ = desktop
    shared_frames = tmp_path / "shared-frames"
    shared_frames.mkdir()
    stage = job / web.INTERMEDIATE_DIRS["vr_frames"]
    try:
        stage.symlink_to(shared_frames, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Platform does not permit directory symlinks: {exc}")
    response = client.post("/analyze_batch", json={"batch_dir": str(job)})
    assert response.status_code == 403
    assert response.get_json()["success"] is False


@pytest.fixture
def finished_frames(desktop):
    _, job, _ = desktop
    frames = job / web.INTERMEDIATE_DIRS["vr_frames"]
    frames.mkdir()
    assert cv2.imwrite(str(frames / "frame_000001.png"), np.zeros((16, 32, 3), dtype=np.uint8))
    return job


def test_analyze_and_stitch_a_managed_job(desktop, finished_frames):
    if not shutil.which("ffmpeg"):
        pytest.skip("Real stitching requires FFmpeg")
    client, job, _ = desktop
    payload = {"batch_dir": str(job), "output_filename": "my edit.mp4"}
    analysis = client.post("/analyze_batch", json=payload).get_json()
    assert analysis["success"] and analysis["analysis"]["frame_count"] == 1
    result = client.post("/stitch_video", json=payload).get_json()
    assert result["success"], result
    assert Path(result["output_path"]) == job / "my edit.mp4"
    assert (job / "my edit.mp4").stat().st_size > 0


def test_stitch_accepts_a_filename_not_another_destination(desktop, finished_frames, tmp_path):
    client, job, _ = desktop
    selected = tmp_path / "another edit.mp4"
    result = client.post(
        "/stitch_video", json={"batch_dir": str(job), "output_filename": str(selected)}
    )
    assert result.status_code == 400
    assert not selected.exists()


def test_stitch_preserves_an_existing_file(desktop, finished_frames):
    client, job, _ = desktop
    original = job / "my edit.mp4"
    original.write_bytes(b"already saved")
    result = client.post(
        "/stitch_video", json={"batch_dir": str(job), "output_filename": original.name}
    )
    assert result.status_code == 409
    assert original.read_bytes() == b"already saved"


def test_stitch_keeps_an_edit_saved_while_the_encoder_was_working(
    desktop, finished_frames, monkeypatch
):
    client, job, _ = desktop
    destination = job / "edit.mp4"

    def encode(command, *, timeout):
        destination.write_bytes(b"another completed edit")
        Path(command[-1]).write_bytes(b"just encoded")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(web, "_run_web_media", encode)
    result = client.post(
        "/stitch_video", json={"batch_dir": str(job), "output_filename": destination.name}
    )
    assert result.status_code == 409
    assert destination.read_bytes() == b"another completed edit"


def test_stitch_leaves_the_active_writer_in_control(desktop, finished_frames):
    client, job, _ = desktop
    with web.JobWriterLock(job):
        response = client.post(
            "/stitch_video", json={"batch_dir": str(job), "output_filename": "new.mp4"}
        )
    assert response.status_code == 409
    assert not (job / "new.mp4").exists()


@pytest.mark.parametrize("setting", ["model_path", "video_path"])
def test_new_jobs_only_take_media_and_model_selections_from_the_ui(desktop, setting):
    client, job, _ = desktop
    payload = {"output_dir": str(job), "settings": {"device": "cpu", setting: "local-file"}}
    response = client.post("/process", json=payload)
    assert response.status_code == 400
    assert setting in response.get_json()["error"]


@pytest.fixture
def completed_job(desktop):
    _, job, _ = desktop
    source = job / "source.mp4"
    source.touch()
    settings_file = job / "job-settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "metadata": {"source_video_name": source.name, "processing_status": "completed"},
                "processing_settings": {},
            }
        ),
        encoding="utf-8",
    )
    return source, settings_file


def test_completed_job_cannot_be_resumed(desktop, completed_job):
    client, job, _ = desktop
    _, settings_file = completed_job
    final = job / "result.mp4"
    final.write_bytes(b"published")
    record = settings_file.read_bytes()
    response = client.post("/resume", json={"output_dir": str(job)})
    assert response.status_code == 409
    assert "completed" in response.get_json()["error"].lower()
    assert settings_file.read_bytes() == record
    assert final.read_bytes() == b"published"


def test_background_resume_rechecks_completion_before_cleanup(completed_job, monkeypatch):
    source, settings_file = completed_job
    job = source.parent
    work = job / web.INTERMEDIATE_DIRS["disparity_stabilized"] / ".vdpp-work"
    work.mkdir(parents=True)
    file = work / "shot_0.raw.f32.mmap"
    file.write_bytes(b"unchanged")
    monkeypatch.setattr(web, "get_video_properties", lambda _: None)
    monkeypatch.setattr(web, "get_video_info", lambda _: None)
    web.process_video_async(
        "session", "requester", source, {"device": "cpu"}, job, {"settings_file": settings_file}
    )
    assert file.is_file()
    assert file.read_bytes() == b"unchanged"


def test_upload_accepts_only_source_video_filenames(desktop):
    client, job, _ = desktop
    response = client.post("/upload", data={"video": (io.BytesIO(b"notes"), "notes.txt")})
    assert response.status_code == 400
    assert list(job.parent.iterdir()) == [job]


def test_same_name_uploads_always_get_distinct_job_directories(desktop, monkeypatch):
    client, _, _ = desktop
    monkeypatch.setattr(web, "time", SimpleNamespace(time=lambda: 100))
    monkeypatch.setattr(web, "datetime", SimpleNamespace(now=lambda: datetime(2025, 1, 1)))
    monkeypatch.setattr(web, "get_video_info", lambda _: {"fps": 24})
    monkeypatch.setattr(web, "_run_web_media", lambda *_a, **_k: SimpleNamespace(stdout=""))
    jobs = []
    for content in (b"clip one", b"clip two"):
        response = client.post("/upload", data={"video": (io.BytesIO(content), "input.mp4")})
        assert response.get_json()["success"]
        jobs.append(Path(response.get_json()["output_dir"]))
    assert jobs[0] != jobs[1]
    assert [(job / "input.mp4").read_bytes() for job in jobs] == [b"clip one", b"clip two"]
