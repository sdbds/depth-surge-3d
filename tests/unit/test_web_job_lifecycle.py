"""Normal desktop job admission, status polling, and background handoff."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Thread, current_thread, main_thread
from types import SimpleNamespace

import pytest

import app as web


@pytest.fixture
def job(tmp_path, monkeypatch):
    folder = tmp_path / "job"
    folder.mkdir()
    video = folder / "source.mp4"
    video.touch()
    monkeypatch.setitem(web.app.config, "OUTPUT_FOLDER", str(tmp_path))
    monkeypatch.setitem(web.app.config, "TESTING", False)
    monkeypatch.setattr(
        web,
        "current_processing",
        dict(
            web.current_processing, active=False, session_id=None, thread=None, stop_requested=False
        ),
    )
    monkeypatch.setattr(
        web,
        "get_video_properties",
        lambda _: dict(
            width=640,
            height=480,
            fps=24,
            frame_count=10,
            sample_aspect_ratio_numerator=1,
            sample_aspect_ratio_denominator=1,
        ),
    )
    monkeypatch.setattr(web, "backend_availability", lambda _: SimpleNamespace(available=True))
    monkeypatch.setattr(web, "detect_resume_settings", lambda *a, **kw: {"device": "cpu"})
    monkeypatch.setattr(web, "resolve_resume_source_video", lambda *a, **kw: video)
    monkeypatch.setattr(
        web,
        "build_resume_report",
        lambda *a, **kw: SimpleNamespace(migrated_settings={}, to_dict=lambda: {}),
    )
    sock = web.socketio.test_client(web.app)
    sid = web.socketio.server.manager.sid_from_eio_sid(sock.eio_sid, "/")
    try:
        yield {"output_dir": str(folder), "socket_id": sid, "settings": {"device": "cpu"}}
    finally:
        sock.disconnect()


@pytest.fixture
def parked(monkeypatch):
    finish = Event()
    threads = []

    def launch(*_):
        thread = Thread(target=finish.wait, daemon=True)
        thread.start()
        threads.append(thread)
        return thread

    monkeypatch.setattr(web.socketio, "start_background_task", launch)
    try:
        yield launch
    finally:
        finish.set()
        for thread in threads:
            thread.join(5)


def test_active_status_serializes_progress_without_the_worker_handle(job, parked):
    client = web.app.test_client()
    response = client.post("/process", json=job)
    assert response.status_code == 200
    web.current_processing["worker_ready_event"] = Event()
    status = client.get("/status")
    assert status.status_code == 200
    state = status.get_json()
    assert state["active"] is True
    assert state["session_id"] == response.get_json()["session_id"]
    assert state["progress"] == 0
    assert "thread" not in state and "worker_ready_event" not in state


def test_resume_listing_shows_only_unfinished_jobs_with_source_frames(job, tmp_path):
    selected = Path(job["output_dir"])
    (selected / web.INTERMEDIATE_DIRS["frames"]).mkdir()
    for name in ("complete", "missing_source"):
        other = tmp_path / name
        (other / web.INTERMEDIATE_DIRS["frames"]).mkdir(parents=True)
        if name == "complete":
            (other / "source.mp4").touch()
            (other / "source_3D_finished.mp4").touch()
    response = web.app.test_client().get("/detect_resumable").get_json()
    assert response["success"] and [row["name"] for row in response["jobs"]] == [selected.name]


@pytest.mark.parametrize("first,second", [("/process", "/resume"), ("/resume", "/process")])
def test_overlapping_valid_submissions_admit_only_one_job(job, parked, monkeypatch, first, second):
    checking, release = Event(), Event()
    validate_socket = web._active_requester_socket_id

    def pause_first(data):
        sid = validate_socket(data)
        if current_thread() is not main_thread():
            checking.set()
            assert release.wait(10)
        return sid

    monkeypatch.setattr(web, "_active_requester_socket_id", pause_first)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(web.app.test_client().post, first, json=job)
        try:
            assert checking.wait(10)
            second_response = web.app.test_client().post(second, json=job)
        finally:
            release.set()
        first_response = pending.result(10)
    assert sorted([first_response.status_code, second_response.status_code]) == [200, 400]
    winner = next(r for r in (first_response, second_response) if r.status_code == 200)
    assert web.current_processing["session_id"] == winner.get_json()["session_id"]


def test_stop_requested_before_worker_start_survives_handoff(job, monkeypatch):
    calls, events = [], []
    monkeypatch.setattr(web.socketio, "start_background_task", lambda *args: calls.append(args))
    monkeypatch.setattr(web.socketio, "emit", lambda event, *_a, **_kw: events.append(event))
    monkeypatch.setattr(web.socketio, "sleep", lambda _: None)

    def unexpected_work(*_):
        raise AssertionError("already-cancelled job started video work")

    monkeypatch.setattr(web, "JobWriterLock", unexpected_work)
    client = web.app.test_client()
    session_id = client.post("/process", json=job).get_json()["session_id"]
    assert client.post("/stop", json={"session_id": session_id}).get_json()["success"]
    target, *args = calls.pop()
    target(*args)
    assert "processing_stopped" in events
    assert "processing_error" not in events
    assert client.get("/status").get_json()["active"] is False


@pytest.mark.parametrize("late_error", [False, True])
def test_late_launcher_result_cannot_replace_or_release_next_job(
    job, parked, monkeypatch, late_error
):
    client = web.app.test_client()
    next_job = {}

    def no_work(*_):
        raise RuntimeError("test worker finished early")

    def finish_then_admit_next(target, *args):
        worker = Thread(target=target, args=args, daemon=True)
        worker.start()
        worker.join(10)
        assert not worker.is_alive()
        monkeypatch.setattr(web.socketio, "start_background_task", parked)
        response = web.app.test_client().post("/process", json=job)
        next_job["id"] = response.get_json()["session_id"]
        next_job["thread"] = web.current_processing["thread"]
        if late_error:
            raise RuntimeError("late launcher failure")
        return worker

    monkeypatch.setattr(web, "JobWriterLock", no_work)
    monkeypatch.setattr(web.socketio, "sleep", lambda _: None)
    monkeypatch.setattr(web.socketio, "start_background_task", finish_then_admit_next)
    response = client.post("/process", json=job)
    assert response.status_code == (500 if late_error else 200)
    assert web.current_processing["session_id"] == next_job["id"]
    assert web.current_processing["thread"] is next_job["thread"]


def test_worker_from_failed_launch_cannot_claim_or_finish_a_new_job(job, parked, monkeypatch):
    client = web.app.test_client()
    run_late = []

    def failed_launch(target, *args):
        run_late.append((target, args))
        raise RuntimeError("background launcher failed")

    def unexpected_work(*_):
        raise AssertionError("retired worker started processing")

    monkeypatch.setattr(web, "JobWriterLock", unexpected_work)
    monkeypatch.setattr(web.socketio, "sleep", lambda _: None)
    monkeypatch.setattr(web.socketio, "start_background_task", failed_launch)
    assert client.post("/process", json=job).status_code == 500
    monkeypatch.setattr(web.socketio, "start_background_task", parked)
    session_id = client.post("/resume", json=job).get_json()["session_id"]
    assert client.post("/stop", json={"session_id": session_id}).get_json()["success"]
    target, args = run_late.pop()
    target(*args)
    state = web.current_processing
    assert state["session_id"] == session_id
    assert state["active"] and state["stop_requested"]


def test_launcher_failure_during_work_keeps_admission_closed_until_cleanup(job, monkeypatch):
    started, finish = Event(), Event()
    workers = []

    def run(*_args):
        started.set()
        assert finish.wait(10)

    def launch_then_fail(target, *args):
        worker = Thread(target=target, args=args, daemon=True)
        workers.append(worker)
        worker.start()
        assert started.wait(10)
        raise RuntimeError("launcher failed after dispatch")

    monkeypatch.setattr(web, "process_video_async", run)
    monkeypatch.setattr(web.socketio, "start_background_task", launch_then_fail)
    client = web.app.test_client()
    try:
        assert client.post("/process", json=job).status_code == 500
        status = client.get("/status").get_json()
        assert status["active"] and status["stop_requested"]
        assert client.post("/resume", json=job).status_code == 400
    finally:
        finish.set()
        for worker in workers:
            worker.join(10)
    assert client.get("/status").get_json()["active"] is False
