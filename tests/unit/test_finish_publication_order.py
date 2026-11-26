"""A failed publication checkpoint cannot authorize the next destructive step."""

from types import SimpleNamespace

import pytest

from src.depth_surge_3d.io.reservation_protocol import FinalPublicationRecovery
from src.depth_surge_3d.processing.orchestration import pipeline_orchestrator as pipeline


STEPS = (
    "resume_stream",
    "validate",
    "adopt",
    "prune",
    "mark_pending",
    "cleanup",
    "terminal",
    "retire",
    "audit",
)


@pytest.mark.parametrize(
    "changes,manifest,expected",
    [
        ({}, {}, True),
        ({"vr_format": "over_under", "vr_output_width": 16, "vr_output_height": 16}, {}, True),
        ({"vr_output_width": 16}, {}, False),
        ({"per_eye_width": True}, {}, False),
        ({"apply_distortion": True}, {"render_shape": (32, 48)}, True),
        ({"apply_distortion": None}, {}, False),
        ({"crop_factor": 0.8}, {}, False),
        ({"crop_factor": "bad"}, {}, False),
        ({}, {"frame_count": 0}, False),
        ({}, {"render_shape": (4, 8)}, False),
        ({"upscale_model": "x2"}, {}, False),
        ({"keep_intermediates": True}, {}, False),
    ],
)
def test_streaming_only_replaces_the_file_pipeline_for_equivalent_geometry(
    changes, manifest, expected
):
    settings = dict(
        vr_format="side_by_side",
        direct_vr_encode=True,
        keep_intermediates=False,
        per_eye_width=16,
        per_eye_height=8,
        vr_output_width=32,
        vr_output_height=8,
        apply_distortion=False,
    )
    inputs = SimpleNamespace(
        **(dict(geometry_mode="relative", render_shape=(8, 16), frame_count=3) | manifest)
    )
    eligible = pipeline.ProcessingOrchestrator._streamed_stereo_settings_eligible
    assert (
        eligible(settings | changes, inputs, schema5_settings=True, active_reservation=False)
        is expected
    )
    assert not eligible(settings | changes, inputs, schema5_settings=True, active_reservation=True)


@pytest.mark.parametrize(
    "names,first",
    [
        ([], None),
        (["clip_000001"], None),
        (["frame_1"], None),
        (["frame_0000001"], None),
        (["frame_"], None),
        (["frame_000004", "frame_000005"], 4),
        (["frame_1000000"], 1000000),
    ],
)
def test_streamed_start_index_closes_the_replay_before_opening_ffmpeg(names, first):
    closed = []

    def replay():
        try:
            for name in names:
                yield SimpleNamespace(frame_name=name)
        finally:
            closed.append(True)

    inputs = SimpleNamespace(replay_declared_records=replay)
    start = pipeline.ProcessingOrchestrator._streamed_first_frame_index
    if first is None:
        with pytest.raises(ValueError, match="no frames|noncanonical"):
            start(inputs)
    else:
        assert start(inputs) == first
    assert closed == [True]


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "resume_stream",
        "validate",
        "prune",
        "mark_pending",
        "cleanup",
        "terminal",
        "retire",
        "audit",
    ],
)
def test_cleanup_advances_only_as_far_as_verified_state(tmp_path, monkeypatch, failure):
    control = tmp_path / "processing-settings-v5.json"
    control.touch()
    index = tmp_path / pipeline.FINAL_ENCODING_RESERVATION_RELATIVE_PATH
    index.parent.mkdir()
    index.touch()
    checkpoints = []

    def step(name):
        checkpoints.append(name)
        if name == failure and name not in {"terminal", "resume_stream", "audit"}:
            raise OSError(f"{name} unavailable")
        return name != failure

    artifact = SimpleNamespace(
        status="in_progress",
        selected_frame_count=3,
        to_object=lambda: {
            "processing_settings": {"keep_intermediates": False},
            "metadata": {"cleanup_status": "not_requested"},
            "runtime_info": {"metric_clamp_summary": {"clamped": 1}},
        },
    )
    prepared = SimpleNamespace(
        requires_explicit_streamed_recovery=lambda: "resume_stream" not in checkpoints,
        recover_publication_prefix=lambda: FinalPublicationRecovery(
            "publication_authenticated", "continue-finalization"
        ),
    )
    capability = SimpleNamespace(validate_authenticated_publication=lambda: step("validate"))
    encoder = SimpleNamespace(
        resume_streamed_publication=lambda *_: step("resume_stream"),
        adopt_final_encoding_reservation=lambda actual: (actual is capability and step("adopt")),
        retire_final_encoding_reservation=lambda *, success: success and step("retire"),
    )

    def terminal(file, status, info):
        assert file == control and status == "completed"
        assert info == {
            "frames_processed": 3,
            "cleanup_status": "complete",
            "metric_clamp_summary": {"clamped": 1},
        }
        return step("terminal")

    monkeypatch.setattr(pipeline, "recover_processing_settings_artifact", lambda _: artifact)
    monkeypatch.setattr(pipeline, "load_final_encoding_reservation_v1", lambda *_: prepared)
    monkeypatch.setattr(
        pipeline.FinalEncodingReservationCapabilityV1, "from_prepared", lambda _: capability
    )
    monkeypatch.setattr(pipeline, "commit_payload_pruned", lambda *_args, **_kw: step("prune"))
    monkeypatch.setattr(
        pipeline, "commit_cleanup_pending_status", lambda *_args, **_kw: step("mark_pending")
    )
    monkeypatch.setattr(pipeline, "run_authorized_cleanup", lambda _: step("cleanup"))
    monkeypatch.setattr(pipeline, "update_processing_status", terminal)
    monkeypatch.setattr(
        pipeline,
        "audit_final_media_v1",
        lambda *_: SimpleNamespace(final_video_authenticated=step("audit")),
    )

    assert pipeline.ProcessingOrchestrator.finish_publication_only(tmp_path, encoder) is (
        failure is None
    )
    prefix = len(STEPS) if failure is None else STEPS.index(failure) + 1
    assert checkpoints == list(STEPS[:prefix])


@pytest.mark.parametrize("status", ["paused", "completed", "failed"])
def test_non_running_or_unreadable_job_does_not_enter_cleanup(tmp_path, monkeypatch, status):
    finish = pipeline.ProcessingOrchestrator.finish_publication_only
    (tmp_path / "processing-settings-v5.json").write_text("{}")
    decoder = object()
    monkeypatch.setattr(
        pipeline, "recover_processing_settings_artifact", lambda _: SimpleNamespace(status=status)
    )
    assert finish(tmp_path, decoder) is None

    def unreadable(_):
        raise OSError("settings are unavailable")

    monkeypatch.setattr(pipeline, "recover_processing_settings_artifact", unreadable)
    assert finish(tmp_path, decoder) is False
