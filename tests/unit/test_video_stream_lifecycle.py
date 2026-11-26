"""Exercise stream ownership and recovery with real files, pipes and a local child."""

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from src.depth_surge_3d.io.reservation_protocol import (
    ReservationConflictError,
    ReservationIncompleteError,
)
from src.depth_surge_3d.processing.video import video_encoder
from src.depth_surge_3d.processing.video.streamed_vr_work import (
    StreamedVrWorkIndexV1,
    StreamedVrWorkReadyV1,
)
from tests.unit.test_video_encoder import _streamed_schema5_job


@pytest.fixture
def open_stream(tmp_path, monkeypatch):
    source, job, settings, settings_file = _streamed_schema5_job(tmp_path)
    baseline = settings_file.read_bytes()
    encoder = video_encoder.VideoEncoder()
    options = dict(
        output_dir=job, original_video=str(source), settings=settings, frame_count=1, first_index=0
    )
    monkeypatch.setattr(video_encoder, "verify_ffmpeg_installation", lambda: True)
    real_start = video_encoder.FfmpegInputPipeProcessV1.start
    sessions = []

    def open_one(script="import sys;sys.stdin.buffer.read();print('progress=end')"):
        monkeypatch.setattr(
            video_encoder.FfmpegInputPipeProcessV1,
            "start",
            lambda pipe, command: real_start(pipe, [sys.executable, "-c", script]),
        )
        session = encoder.open_streamed_vr_session(**options)
        sessions.append(session)
        return session

    yield open_one
    for session in reversed(sessions):
        session.abort()
    assert settings_file.read_bytes() == baseline
    assert not list(job.glob(".depth-surge-streamed-vr-*"))


@pytest.mark.parametrize(
    "termination", ["clean-fallback", "abort-retry", "fallback-retry", "exit-error", "empty-output"]
)
def test_prepublication_encoder_failures_never_commit_partial_frames(
    open_stream, monkeypatch, termination
):
    script = "import sys; sys.stdin.buffer.read(); print('progress=end')"
    if termination == "exit-error":
        script += ";sys.stderr.write('invalid input pixels');sys.exit(3)"
    session = open_stream(script)
    work = session.work
    original_index = work.index_path.read_bytes()
    original_ready = work.ready_path.read_bytes()
    if termination in {"exit-error", "empty-output"}:
        with pytest.raises(ValueError, match="BGR8"):
            session.write("frame_000000", np.zeros((4, 8), np.uint8))
        with pytest.raises(ValueError, match="shape differs"):
            session.write("frame_000000", np.zeros((4, 9, 3), np.uint8))
        session.write("frame_000000", np.full((4, 8, 3), 57, np.uint8))
        assert session.finish_and_publish() is False
    else:
        if termination.endswith("retry"):
            abort = session.process.abort

            def refused_abort():
                raise OSError("pipe cleanup unavailable")

            monkeypatch.setattr(session.process, "abort", refused_abort)
            action = (
                session.abort
                if termination == "abort-retry"
                else session.discard_for_file_backed_fallback
            )
            with pytest.raises(ReservationIncompleteError, match="retryable"):
                action()
            assert work.index_path.read_bytes() == original_index
            assert work.ready_path.read_bytes() == original_ready
            assert all(p.is_file() for p in work.validate_ready_work_files())
            monkeypatch.setattr(session.process, "abort", abort)
        session.discard_for_file_backed_fallback()

    assert not work.index_path.exists() and not work.ready_path.exists()
    assert not work.output_path.exists()
    assert session.process._process.poll() is not None
    with pytest.raises(RuntimeError, match="closed"):
        session.write("frame_000000", np.zeros((4, 8, 3), np.uint8))
    with pytest.raises(RuntimeError, match="launch phase"):
        session.discard_for_file_backed_fallback()


def test_work_handoff_requires_exactly_one_manifest_and_the_owned_output(open_stream):
    session = open_stream()
    work = session.work
    manifest, output = work.validate_ready_work_files()
    with pytest.raises(ReservationConflictError, match="old name"):
        work.retire_handoff(final_sibling_path=output)
    second = work.manifest_target_path
    second.write_bytes(b"independent temporary content")
    try:
        with pytest.raises(ReservationConflictError, match="two manifest generations"):
            work.validate_ready_work_files()
        with pytest.raises(ReservationConflictError, match="two manifest generations"):
            work.discard()
        assert second.read_bytes() == b"independent temporary content"
        assert manifest.is_file() and output.is_file() and work.ready_path.is_file()
    finally:
        second.unlink()
    replacement = output.with_name("saved-original")
    output.rename(replacement)
    output.write_bytes(b"replacement")
    try:
        with pytest.raises(ReservationConflictError, match="identity changed"):
            work.validate_ready_work_files()
        assert output.read_bytes() == b"replacement"
    finally:
        output.unlink()
        replacement.rename(output)
    assert work.validate_ready_work_files() == (manifest, output)


def test_work_evidence_rejects_conflicting_authority_fields_without_touching_work_files(
    open_stream,
):
    session = open_stream()
    work = session.work
    index = work.index.to_object()
    ready = work.ready.to_object()
    for update, reason in [
        ({"unknown": True}, "schema"),
        ({"schema_version": 2}, "version"),
        ({"work_generation": "a"}, "generation"),
        ({"settings_revision_at_preflight": False}, "revision"),
        ({"expected_final_relative_path": "../neighbor.mp4"}, "path"),
        ({"output_relative_path": "other.mp4"}, "noncanonical"),
        ({"fingerprint": "0" * 64}, "fingerprint"),
        ({"publication_generation": "9" * 32}, "authority"),
        ({"job_root_identity": {}}, "evidence"),
    ]:
        with pytest.raises(ReservationConflictError, match=reason):
            StreamedVrWorkIndexV1.from_object({**index, **update})
    for update in ({"schema_version": 0}, {"fingerprint": "a" * 64}, {"output_identity": {}}):
        with pytest.raises(ReservationConflictError):
            StreamedVrWorkReadyV1.from_object({**ready, **update})
    assert StreamedVrWorkIndexV1.from_object(index) == work.index
    assert StreamedVrWorkReadyV1.from_object(ready) == work.ready
    assert all(path.is_file() for path in work.validate_ready_work_files())


@pytest.mark.parametrize(("samples", "bound"), [(None, 0), (1024, 3), (1025, 4), (48000, 49)])
def test_container_validation_freezes_dimensions_rate_and_aac_packet_ceiling(
    tmp_path, monkeypatch, samples, bound
):
    observed = []
    settings = {"vr_output_width": 12, "vr_output_height": 4}
    cancelled = (lambda: False) if samples is not None else None

    def probe(path, expectation, **kwargs):
        observed.append((path, expectation, kwargs))
        return True

    monkeypatch.setattr(video_encoder, "validate_final_container_v4", probe)
    check = video_encoder.VideoEncoder()._container_validator_for_invocation(
        settings, 5, "24000/1000", "libx264", samples, cancelled
    )
    settings["vr_output_height"] = 1000
    candidate = tmp_path / "encoded.mp4"
    assert check(candidate) is True
    [(path, expectation, kwargs)] = observed
    assert path == candidate
    assert (expectation.width, expectation.height, expectation.fps) == (12, 4, "24/1")
    assert (expectation.audio_expected, expectation.audio_packet_upper_bound) == (
        samples is not None,
        bound,
    )
    assert kwargs == ({} if cancelled is None else {"cancelled": cancelled})


@pytest.mark.parametrize(
    ("stderr", "code", "expected"), [("", 0, 0), ("codec warning", 0, 1), ("failed", 7, 7)]
)
def test_schema5_runner_uses_real_final_progress_and_never_succeeds_with_diagnostics(
    stderr, code, expected
):
    updates = []
    tracker = SimpleNamespace(
        is_cancelled=lambda: False, update_progress=lambda message, **kwargs: updates.append(kwargs)
    )
    script = (
        "import sys;print('frame=9');print('progress=end');"
        f"sys.stderr.write({stderr!r});sys.exit({code})"
    )
    status, messages = video_encoder.VideoEncoder()._run_schema5_ffmpeg_with_progress(
        [sys.executable, "-c", script], 4, tracker
    )
    assert status == expected
    assert messages == ((stderr,) if stderr else ())
    assert [update["frame_num"] for update in updates] == ([] if stderr else [4])


@pytest.mark.parametrize("bad", ["input", "output"])
def test_unbound_runtime_paths_are_not_committed_to_publication_arguments(tmp_path, bad):
    from tests.unit.test_video_encoder import _disabled_audio

    candidate = tmp_path / "temporary.mp4"
    command = ["ffmpeg", "-i", "left-sequence", str(candidate)]
    command[2 if bad == "input" else 3] = str(tmp_path / "different")
    with pytest.raises(ValueError, match="unbound input|output path changed"):
        video_encoder.VideoEncoder._normalize_publication_arguments(
            command=command,
            image_inputs={"left-sequence": "left"},
            selected_audio=_disabled_audio(),
            runtime_output=candidate,
            input_stage_fingerprint="a" * 64,
            final_relative_path="done.mp4",
        )


@pytest.mark.parametrize(
    ("layout", "reason"),
    [
        ("outside", "outside the job root"),
        ("none", "has no frames"),
        ("gap", "not contiguous"),
        ("case", "frame name is noncanonical"),
        ("padding", "padding is noncanonical"),
        ("directory", "not an ordinary file"),
        ("cancelled", "cancelled"),
    ],
)
def test_assembled_frame_preflight_leaves_settings_and_source_untouched(tmp_path, layout, reason):
    _, job, settings, settings_file = _streamed_schema5_job(tmp_path)
    stable = settings_file.read_bytes()
    frames = (tmp_path if layout == "outside" else job) / "assembled"
    frames.mkdir()
    names = {
        "gap": ["frame_000000.png", "frame_000002.png"],
        "case": ["FRAME_000000.PNG"],
        "padding": ["frame_0000000.png"],
        "cancelled": ["frame_000000.png"],
    }
    for name in names.get(layout, []):
        (frames / name).write_bytes(b"frame")
    if layout == "directory":
        (frames / "frame_000000.png").mkdir()
    with pytest.raises(InterruptedError if layout == "cancelled" else ValueError, match=reason):
        video_encoder.VideoEncoder()._prepare_schema5_assembled_context(
            vr_frames_dir=frames,
            output_dir=job,
            settings=settings,
            cancelled=lambda: layout == "cancelled",
        )
    assert settings_file.read_bytes() == stable
    assert sorted(p.name for p in frames.iterdir()) == sorted(
        names.get(layout, ["frame_000000.png"] if layout == "directory" else [])
    )
    assert not (job / video_encoder.FINAL_ENCODING_RESERVATION_RELATIVE_PATH).exists()


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("legacy", "schema-5"),
        ("frame-count", "selected count"),
        ("disabled", "unavailable"),
        ("settings", "differ from durable settings"),
        ("active", "active final reservation"),
    ],
)
def test_stream_preflight_never_launches_over_incompatible_job_state(
    tmp_path, monkeypatch, fault, reason
):
    source, job, settings, settings_file = _streamed_schema5_job(tmp_path)
    if fault == "legacy":
        job = tmp_path / "legacy"
        job.mkdir()
    if fault == "settings":
        settings = {**settings, "vr_output_width": 12}
    active = job / video_encoder.FINAL_ENCODING_RESERVATION_RELATIVE_PATH
    if fault == "active":
        video_encoder.prepare_final_encoding_reservation_v1(
            job_root=job,
            settings=video_encoder.ProcessingSettingsArtifactV5.parse(settings_file.read_bytes()),
            publication_generation="a1" * 16,
            input_manifest_raw_max=4096,
            final_manifest_raw_max=4096,
        )
        prior = active.read_bytes()
    monkeypatch.setattr(video_encoder, "verify_ffmpeg_installation", lambda: fault != "disabled")
    with pytest.raises((RuntimeError, ValueError), match=reason):
        video_encoder.VideoEncoder().open_streamed_vr_session(
            output_dir=job,
            original_video=str(source),
            settings=settings,
            frame_count=2 if fault == "frame-count" else 1,
        )
    assert not list(job.glob(".depth-surge-streamed-vr-*"))
    if fault == "active":
        assert active.read_bytes() == prior


@pytest.mark.parametrize(
    "replacement", [None, b"", bytes(512 * 1024 + 1)], ids=["missing", "empty", "oversized"]
)
def test_work_discard_waits_for_readable_settings_authority(open_stream, replacement):
    work = open_stream().work
    path = work.job_root / "processing-settings-v5.json"
    held = path.with_suffix(".held")
    path.rename(held)
    try:
        if replacement is not None:
            path.write_bytes(replacement)
        with pytest.raises(ReservationConflictError, match="unavailable|bound"):
            work.discard()
        assert work.index_path.exists() and work.ready_path.exists()
        assert all(p.exists() for p in work.validate_ready_work_files())
    finally:
        path.unlink(missing_ok=True)
        held.rename(path)
    assert work.validate_current_authority().revision == work.index.settings_revision_at_preflight


@pytest.mark.parametrize("available", [False, True])
def test_legacy_scalar_stereo_source_resolves_only_its_requested_frame_range(
    tmp_path, monkeypatch, available
):
    from tests.unit.test_video_encoder import _direct_settings, _write_eye_sequence

    left = _write_eye_sequence(tmp_path / "left", [7, 8])
    right = _write_eye_sequence(tmp_path / "right", [7, 8])
    settings = _direct_settings(target_fps="24", source_fps=24)
    encoder = video_encoder.VideoEncoder()
    launched = []

    def encode(command, total, _tracker):
        launched.append(command)
        assert total == 2
        assert [
            command[i + 1] for i in range(len(command) - 1) if command[i] == "-start_number"
        ] == ["7", "7"]
        assert [
            Path(command[i + 1]).parent for i in range(len(command) - 1) if command[i] == "-i"
        ] == [left[0].parent, right[0].parent]
        Path(command[-1]).write_bytes(b"legacy final payload")
        return 0, ()

    monkeypatch.setattr(encoder, "_run_ffmpeg_with_progress", encode)
    monkeypatch.setattr(video_encoder, "verify_ffmpeg_installation", lambda: available)
    source = video_encoder.DirectStereoSource(left[0].parent, right[0].parent, 7, 2)
    options = dict(
        output_dir=tmp_path, original_video=str(tmp_path / "source.mp4"), settings=settings
    )
    assert not encoder.create_video_from_stereo_source(source, total_frames=1, **options)
    assert encoder.create_video_from_stereo_source(source, total_frames=2, **options) is available
    assert len(launched) == int(available)
    assert all(path.is_file() for path in [*left, *right])
    outputs = list(tmp_path.glob("*.mp4"))
    assert [path.read_bytes() for path in outputs] == (
        [b"legacy final payload"] if available else []
    )


@pytest.mark.parametrize("fault", ["missing-binary", "decode-failed"])
def test_failed_extraction_creates_stage_but_never_attests_a_partial_sequence(
    tmp_path, monkeypatch, fault
):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original")
    calls = []

    def unavailable(command, **kwargs):
        calls.append(command)
        if fault == "missing-binary":
            raise FileNotFoundError(command[0])
        return SimpleNamespace(returncode=1, stderr="incomplete bitstream")

    monkeypatch.setattr(video_encoder.subprocess, "run", unavailable)
    result = video_encoder.VideoEncoder().extract_frames(
        str(source),
        {"base": tmp_path},
        {"frame_count": 2, "fps": 24, "frame_rate": "24000/1001"},
        {},
    )
    assert result == []
    stage = tmp_path / video_encoder.INTERMEDIATE_DIRS["frames"]
    assert stage.is_dir() and not (stage / "metadata.json").exists()
    assert source.read_bytes() == b"original"
    assert len(calls) == (1 if fault == "missing-binary" else 2)
    if len(calls) == 2:
        assert "-hwaccel" in calls[0] and "-hwaccel" not in calls[1]
        assert all(c[c.index("-fps_mode") + 1] == "passthrough" for c in calls)
