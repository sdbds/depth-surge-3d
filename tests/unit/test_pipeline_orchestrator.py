"""Unit tests for pipeline orchestrator."""

import gc
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import weakref

import numpy as np
import pytest

from src.depth_surge_3d.processing.orchestration.pipeline_orchestrator import (
    ProcessingOrchestrator,
)
from src.depth_surge_3d.io.job_lock import JobWriterLock
from src.depth_surge_3d.io.reservation_protocol import ReservationIncompleteError
from src.depth_surge_3d.io.resume import LegacyFastAdoptionV1
from src.depth_surge_3d.processing.frames.stereo_control import (
    StereoPipelineCapacityError,
)


class TestProcessingOrchestratorInit:
    """Test ProcessingOrchestrator initialization."""

    def test_init_with_all_processors(self):
        """Test initialization with all required processors."""
        depth_proc = Mock()
        stereo_gen = Mock()
        distortion_proc = Mock()
        upscaler = Mock()
        vr_assembler = Mock()
        video_encoder = Mock()

        orchestrator = ProcessingOrchestrator(
            depth_proc,
            stereo_gen,
            distortion_proc,
            upscaler,
            vr_assembler,
            video_encoder,
        )

        assert orchestrator.depth_processor == depth_proc
        assert orchestrator.stereo_generator == stereo_gen
        assert orchestrator.distortion_processor == distortion_proc
        assert orchestrator.upscaler == upscaler
        assert orchestrator.vr_assembler == vr_assembler
        assert orchestrator.video_encoder == video_encoder
        assert video_encoder.defer_final_reservation_retirement is True
        assert orchestrator.verbose is False
        assert orchestrator._start_time == 0.0
        assert not hasattr(orchestrator, "_total_steps")
        assert "_get_total_steps" not in vars(ProcessingOrchestrator)
        assert "_print_step_complete" not in vars(ProcessingOrchestrator)

    def test_init_with_verbose(self):
        """Test initialization with verbose flag."""
        orchestrator = ProcessingOrchestrator(
            Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), verbose=True
        )
        assert orchestrator.verbose is True

    @pytest.mark.parametrize(
        ("files", "source"),
        [
            (None, "base"),
            ([Path("frame_000001.png")], None),
            ([Path("frame_000001.png")], "unknown"),
        ],
    )
    def test_preselected_files_and_producer_must_be_supplied_together(self, files, source):
        with pytest.raises(ValueError, match="preselected"):
            ProcessingOrchestrator(
                Mock(),
                Mock(),
                Mock(),
                Mock(),
                Mock(),
                Mock(),
                preselected_depth_files=files,
                preselected_render_source=source,
            )


class TestFormatProcessingTime:
    """Test _format_processing_time pure function."""

    def test_seconds_only(self):
        """Test formatting seconds only."""
        assert ProcessingOrchestrator._format_processing_time(45) == "45s"

    def test_minutes_and_seconds(self):
        """Test formatting minutes and seconds."""
        assert ProcessingOrchestrator._format_processing_time(125) == "2m 5s"

    def test_hours_minutes_seconds(self):
        """Test formatting hours, minutes, and seconds."""
        assert ProcessingOrchestrator._format_processing_time(5025) == "1h 23m 45s"

    def test_exact_hour(self):
        """Test formatting exact hour."""
        assert ProcessingOrchestrator._format_processing_time(3600) == "1h"

    def test_exact_minute(self):
        """Test formatting exact minute."""
        assert ProcessingOrchestrator._format_processing_time(120) == "2m"

    def test_zero_seconds(self):
        """Test formatting zero seconds."""
        assert ProcessingOrchestrator._format_processing_time(0) == "0s"

    def test_large_duration(self):
        """Test formatting large duration."""
        # 10 hours, 30 minutes, 15 seconds
        assert ProcessingOrchestrator._format_processing_time(37815) == "10h 30m 15s"


class TestSetupProcessing:
    """Test _setup_processing method."""

    @patch(
        "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.create_output_directories"
    )
    @patch(
        "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.save_processing_settings"
    )
    @patch("src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.time.time")
    def test_setup_processing(self, mock_time, mock_save_settings, mock_create_dirs):
        """Test setup processing creates directories and settings file."""
        mock_time.return_value = 1234567890
        mock_create_dirs.return_value = {"base": Path("/output/dir")}
        mock_save_settings.return_value = Path("/output/dir/settings.json")

        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())

        video_props = {"fps": 30, "frame_count": 100}
        settings = {"keep_intermediates": True}

        output_path, directories, settings_file = orchestrator._setup_processing(
            "/input/video.mp4", "/output/dir", settings, video_props
        )

        assert output_path == Path("/output/dir")
        assert directories == {"base": Path("/output/dir")}
        assert settings_file == Path("/output/dir/settings.json")
        mock_create_dirs.assert_called_once_with(
            Path("/output/dir"),
            True,
            omitted_intermediates={"disparity_stabilized"},
        )

    def test_setup_processing_omits_vr_directory_in_direct_mode(self):
        """Direct encoding does not create the unused assembled-frame directory."""
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())
        with (
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.create_output_directories",
                return_value={"base": Path("/output/dir")},
            ) as create_dirs,
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.save_processing_settings"
            ),
        ):
            orchestrator._setup_processing(
                "/input/video.mp4",
                "/output/dir",
                {"keep_intermediates": True, "direct_vr_encode": True},
                {"fps": 30, "frame_count": 1},
            )

        create_dirs.assert_called_once_with(
            Path("/output/dir"),
            True,
            omitted_intermediates={"disparity_stabilized", "vr_frames"},
        )

    def test_setup_processing_omits_distortion_directories_for_fused_stage(self):
        """Fused distortion writes Stage 6 directly, so Stage 5 is not allocated."""
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())
        with (
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.create_output_directories",
                return_value={"base": Path("/output/dir")},
            ) as create_dirs,
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.save_processing_settings"
            ),
        ):
            orchestrator._setup_processing(
                "/input/video.mp4",
                "/output/dir",
                {
                    "keep_intermediates": True,
                    "apply_distortion": True,
                    "direct_vr_encode": True,
                },
                {"fps": 30, "frame_count": 1},
            )

        create_dirs.assert_called_once_with(
            Path("/output/dir"),
            True,
            omitted_intermediates={
                "disparity_stabilized",
                "left_distorted",
                "right_distorted",
                "vr_frames",
            },
        )

    def test_setup_processing_keeps_legacy_directory_call_when_direct_mode_is_false(
        self,
    ):
        """Explicitly disabling direct mode preserves the legacy directory setup."""
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())
        with (
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.create_output_directories",
                return_value={"base": Path("/output/dir")},
            ) as create_dirs,
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.save_processing_settings"
            ),
        ):
            orchestrator._setup_processing(
                "/input/video.mp4",
                "/output/dir",
                {"keep_intermediates": True, "direct_vr_encode": False},
                {"fps": 30, "frame_count": 1},
            )

        create_dirs.assert_called_once_with(
            Path("/output/dir"),
            True,
            omitted_intermediates={"disparity_stabilized"},
        )

    def test_setup_processing_creates_stabilized_directory_only_when_requested(self):
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())
        with (
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.create_output_directories",
                return_value={"base": Path("/output/dir")},
            ) as create_dirs,
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.save_processing_settings"
            ),
        ):
            orchestrator._setup_processing(
                "/input/video.mp4",
                "/output/dir",
                {
                    "keep_intermediates": True,
                    "direct_vr_encode": False,
                    "temporal_postprocessor": "vdpp",
                },
                {"fps": 30, "frame_count": 1},
            )

        create_dirs.assert_called_once_with(Path("/output/dir"), True)


class TestFinalizeProcessing:
    """Test _finalize_processing method."""

    @pytest.mark.parametrize("audit_result", [True, False, "error"])
    def test_completion_time_includes_cleanup_and_terminal_audit(
        self, tmp_path, monkeypatch, capsys, audit_result
    ):
        from src.depth_surge_3d.processing.orchestration import pipeline_orchestrator as module

        clock = SimpleNamespace(now=1010.0)

        def advance(seconds, result=None):
            clock.now += seconds
            return result

        reservation = SimpleNamespace(
            recover_publication_prefix=lambda: SimpleNamespace(phase="publication_authenticated"),
            validate_authenticated_publication=lambda: advance(1),
        )
        encoder = SimpleNamespace(
            pending_final_encoding_reservation=reservation,
            retire_final_encoding_reservation=lambda **_kwargs: advance(1),
        )
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), encoder)
        orchestrator._start_time = 1000.0
        orchestrator._settings_file = tmp_path / "processing-settings-v5.json"
        terminal_updates = []

        def commit_terminal(*args):
            terminal_updates.append(args)
            return advance(1, True)

        def audit(*_args):
            advance(2)
            if audit_result == "error":
                raise ValueError("audit failed")
            return SimpleNamespace(final_video_authenticated=audit_result)

        monkeypatch.setattr(module.time, "time", lambda: 1010.0)
        monkeypatch.setattr(module.time, "perf_counter", lambda: clock.now)
        monkeypatch.setattr(module, "load_processing_settings", lambda _path: None)
        monkeypatch.setattr(module, "commit_payload_pruned", lambda *_args, **_kwargs: {})
        monkeypatch.setattr(module, "commit_cleanup_pending_status", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            module, "run_authorized_cleanup", lambda *_args, **_kwargs: advance(5, 3)
        )
        monkeypatch.setattr(module, "update_processing_status", commit_terminal)
        monkeypatch.setattr(module, "recover_processing_settings_artifact", lambda _path: object())
        monkeypatch.setattr(module, "audit_final_media_v1", audit)

        result = orchestrator._finalize_processing(
            True,
            tmp_path,
            "source.mp4",
            {
                "vr_format": "side_by_side",
                "vr_resolution": "16x9-1080p",
                "keep_intermediates": False,
            },
            3,
        )

        output = capsys.readouterr().out
        assert result is (audit_result is True)
        assert len(terminal_updates) == 1
        if audit_result is True:
            assert "PROCESSING COMPLETE" in output
            # 10 pipeline + 1 validation + 5 cleanup + 1 commit + 1 retire + 2 audit.
            assert "20s" in output
        else:
            assert "PROCESSING COMPLETE" not in output

    @patch("src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.completion_banner")
    @patch(
        "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.update_processing_status"
    )
    @patch(
        "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.generate_output_filename"
    )
    @patch("src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.time.perf_counter")
    def test_finalize_processing_success(
        self, mock_time, mock_gen_filename, mock_update_status, mock_banner
    ):
        """Test finalize processing on success."""
        mock_time.return_value = 1500.0  # Current time when finalize is called
        mock_gen_filename.return_value = "video_3D_side_by_side.mp4"

        video_encoder = Mock()
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), video_encoder)
        orchestrator._start_time = 1000.0  # Started 500 seconds ago
        orchestrator._settings_file = Path("/output/settings.json")

        settings = {"vr_format": "side_by_side", "vr_resolution": "16x9-1080p"}

        orchestrator._finalize_processing(
            success=True,
            output_path=Path("/output"),
            video_path="/input/video.mp4",
            settings=settings,
            num_frames=300,
        )

        # Check completion banner was called
        mock_banner.assert_called_once()
        args = mock_banner.call_args[1]
        assert args["output_file"] == str(Path("/output/video_3D_side_by_side.mp4"))
        assert args["processing_time"] == "8m 20s"  # 500 seconds = 8m 20s
        assert args["num_frames"] == 300
        assert args["vr_format"] == "side_by_side"

        # Check status was updated
        mock_update_status.assert_called_once()
        video_encoder.retire_final_encoding_reservation.assert_called_once_with(success=True)

    @patch(
        "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.update_processing_status"
    )
    def test_finalize_processing_failure(self, mock_update_status):
        """Test finalize processing on failure."""
        video_encoder = Mock()
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), video_encoder)
        orchestrator._settings_file = Path("/output/settings.json")

        settings = {"vr_format": "side_by_side", "vr_resolution": "16x9-1080p"}

        orchestrator._finalize_processing(
            success=False,
            output_path=Path("/output"),
            video_path="/input/video.mp4",
            settings=settings,
            num_frames=300,
        )

        mock_update_status.assert_called_once_with(
            Path("/output/settings.json"), "failed", {"error": "Video creation failed"}
        )
        video_encoder.retire_final_encoding_reservation.assert_called_once_with(success=False)

    @patch(
        "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.cleanup_intermediate_files"
    )
    @patch("src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.completion_banner")
    @patch(
        "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.update_processing_status"
    )
    def test_finalize_success_cleans_temporary_files_when_not_retained(
        self, _update_status, _banner, cleanup
    ):
        """Intermediates are deleted only after final video creation succeeds."""
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())
        orchestrator._start_time = 0.0
        orchestrator._settings_file = Path("/output/settings.json")

        with patch(
            "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.time.perf_counter",
            return_value=1.0,
        ):
            orchestrator._finalize_processing(
                True,
                Path("/output"),
                "/input/video.mp4",
                {
                    "vr_format": "side_by_side",
                    "vr_resolution": "16x9-1080p",
                    "keep_intermediates": False,
                },
                3,
            )

        cleanup.assert_called_once_with(Path("/output"))

    @patch(
        "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.cleanup_intermediate_files"
    )
    @patch(
        "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.update_processing_status"
    )
    def test_failed_job_retains_all_committed_checkpoints(self, update_status, cleanup, tmp_path):
        committed = {}
        for name in (
            "00_original_frames",
            "02_depth_raw",
            "03_disparity_maps",
            "03_metric_geometry",
        ):
            directory = tmp_path / name
            directory.mkdir()
            payload = directory / "committed.bin"
            payload.write_bytes(name.encode("ascii"))
            committed[payload] = payload.read_bytes()
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())
        orchestrator._settings_file = tmp_path / "job-settings.json"

        orchestrator._finalize_processing(
            False,
            tmp_path,
            str(tmp_path / "source.mp4"),
            {"keep_intermediates": False},
            2,
        )

        cleanup.assert_not_called()
        assert {path: path.read_bytes() for path in committed} == committed
        update_status.assert_called_once_with(
            orchestrator._settings_file,
            "failed",
            {"error": "Video creation failed"},
        )

    @patch("src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.completion_banner")
    @patch(
        "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.update_processing_status"
    )
    def test_metric_clamp_summary_is_persisted_in_completed_runtime_info(
        self, update_status, _banner, tmp_path
    ):
        summary = {
            "affected_frame_count": 2,
            "mean_clamped_fraction": 0.125,
            "max_clamped_fraction": 0.25,
        }
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())
        orchestrator._settings_file = tmp_path / "job-settings.json"
        orchestrator._metric_clamp_summary = summary

        with patch(
            "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.time.time",
            return_value=1.0,
        ):
            orchestrator._finalize_processing(
                True,
                tmp_path,
                str(tmp_path / "source.mp4"),
                {
                    "vr_format": "side_by_side",
                    "vr_resolution": "custom",
                    "keep_intermediates": True,
                },
                2,
            )

        additional_info = update_status.call_args.args[2]
        assert additional_info["metric_clamp_summary"] == summary

    def test_schema_five_cleanup_commits_authorization_and_pending_before_deletion(self, tmp_path):
        events = []
        video_encoder = Mock()
        reservation = Mock()
        reservation.recover_publication_prefix.return_value = Mock(
            phase="publication_authenticated"
        )
        video_encoder.pending_final_encoding_reservation = reservation
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), video_encoder)
        orchestrator._start_time = 0.0
        orchestrator._settings_file = tmp_path / "processing-settings-v5.json"
        authenticated = Mock(final_video_authenticated=True)

        with (
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "recover_processing_settings_artifact",
                side_effect=lambda _path: events.append("recover") or Mock(),
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "audit_final_media_v1",
                side_effect=lambda *_args, **_kwargs: events.append("audit") or authenticated,
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "commit_payload_pruned",
                side_effect=lambda _path, **_kwargs: events.append("payload_pruned")
                or {"status": "payload_pruned"},
            ) as commit_pruned,
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "commit_cleanup_pending_status",
                side_effect=lambda _path, **_kwargs: events.append("pending") or Mock(),
            ) as commit_pending,
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "run_authorized_cleanup",
                side_effect=lambda *_args, **_kwargs: events.append("delete") or 7,
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "update_processing_status",
                side_effect=lambda *_args, **_kwargs: events.append("terminal") or True,
            ) as update_status,
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.completion_banner"
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.time.time",
                return_value=1.0,
            ),
        ):
            assert True is orchestrator._finalize_processing(
                True,
                tmp_path,
                str(tmp_path / "source.mp4"),
                {
                    "vr_format": "side_by_side",
                    "vr_resolution": "16x9-1080p",
                    "keep_intermediates": False,
                },
                3,
            )

        assert events == [
            "payload_pruned",
            "pending",
            "delete",
            "terminal",
            "recover",
            "audit",
        ]
        assert update_status.call_args.args[2]["cleanup_status"] == "complete"
        assert commit_pruned.call_args.kwargs == {"reservation": reservation}
        assert commit_pending.call_args.kwargs == {"reservation": reservation}
        video_encoder.retire_final_encoding_reservation.assert_called_once_with(success=True)

    def test_schema_five_cleanup_identity_failure_preserves_successful_media(self, tmp_path):
        from src.depth_surge_3d.rendering.quality.cleanup_transaction import (
            CleanupIdentityMismatchError,
        )

        encoder = Mock()
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), encoder)
        orchestrator._start_time = 0.0
        orchestrator._settings_file = tmp_path / "processing-settings-v5.json"

        with (
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "recover_processing_settings_artifact",
                return_value=Mock(),
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "audit_final_media_v1",
                return_value=Mock(final_video_authenticated=True),
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "commit_payload_pruned",
                return_value={"status": "payload_pruned"},
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "commit_cleanup_pending_status",
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "run_authorized_cleanup",
                side_effect=CleanupIdentityMismatchError("changed"),
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "update_processing_status",
                return_value=True,
            ) as update_status,
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.completion_banner"
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.time.time",
                return_value=1.0,
            ),
        ):
            assert not orchestrator._finalize_processing(
                True,
                tmp_path,
                str(tmp_path / "source.mp4"),
                {
                    "vr_format": "side_by_side",
                    "vr_resolution": "16x9-1080p",
                    "keep_intermediates": False,
                },
                3,
            )

        update_status.assert_not_called()
        encoder.retire_final_encoding_reservation.assert_not_called()


class TestCleanupOnlyResume:
    @pytest.mark.parametrize("failure_stage", ["model", "lock"])
    def test_process_records_elapsed_time_when_owned_teardown_raises(
        self, tmp_path, monkeypatch, failure_stage
    ):
        from src.depth_surge_3d.processing.orchestration import pipeline_orchestrator as module

        clock = SimpleNamespace(now=1000.0)
        events = []

        def release_model():
            events.append("model")
            clock.now += 2
            if failure_stage == "model":
                raise RuntimeError("model teardown failed")

        class TimedJobLock(JobWriterLock):
            def release(self):
                events.append("lock")
                super().release()
                clock.now += 3
                if failure_stage == "lock":
                    raise RuntimeError("lock teardown failed")

        orchestrator = ProcessingOrchestrator(
            Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), release_depth_model=release_model
        )

        def resume(_output_dir):
            events.append("resume")
            clock.now += 10
            return True

        monkeypatch.setattr(module.time, "perf_counter", lambda: clock.now)
        monkeypatch.setattr(module, "JobWriterLock", TimedJobLock)
        monkeypatch.setattr(orchestrator, "_resume_cleanup_only", resume)

        with pytest.raises(RuntimeError, match=f"{failure_stage} teardown failed"):
            orchestrator.process(tmp_path / "source.mp4", tmp_path, {}, {})

        assert events == ["resume", "model", "lock"]
        assert orchestrator.process_elapsed_seconds == 15.0
        with JobWriterLock(tmp_path):
            pass

    @pytest.mark.parametrize("resume_result", [True, False, "interrupted"])
    def test_process_elapsed_time_includes_owned_teardown_on_cleanup_only_resume(
        self, tmp_path, monkeypatch, resume_result
    ):
        from src.depth_surge_3d.processing.orchestration import pipeline_orchestrator as module

        clock = SimpleNamespace(now=1000.0)

        def release_model():
            clock.now += 2

        class TimedJobLock(JobWriterLock):
            def release(self):
                super().release()
                clock.now += 3

        orchestrator = ProcessingOrchestrator(
            Mock(), Mock(), Mock(), Mock(), Mock(), Mock(), release_depth_model=release_model
        )

        def resume(_output_dir):
            clock.now += 10
            if resume_result == "interrupted":
                raise InterruptedError("cancelled")
            return resume_result

        monkeypatch.setattr(module.time, "time", lambda: -500.0)
        monkeypatch.setattr(module.time, "perf_counter", lambda: clock.now)
        monkeypatch.setattr(module, "JobWriterLock", TimedJobLock)
        monkeypatch.setattr(orchestrator, "_resume_cleanup_only", resume)

        if resume_result == "interrupted":
            with pytest.raises(InterruptedError, match="cancelled"):
                orchestrator.process(tmp_path / "source.mp4", tmp_path, {}, {})
        else:
            assert orchestrator.process(tmp_path / "source.mp4", tmp_path, {}, {}) is resume_result

        assert orchestrator.process_elapsed_seconds == 15.0
        with JobWriterLock(tmp_path):
            pass

    def test_process_returns_after_cleanup_only_resume(self, tmp_path):
        encoder = Mock()
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), encoder)

        with (
            patch.object(orchestrator, "_resume_cleanup_only", return_value=True) as resume,
            patch.object(orchestrator, "_setup_processing") as setup,
        ):
            result = orchestrator.process(
                tmp_path / "source.mp4",
                tmp_path,
                {"fps": 24.0, "frame_count": 3},
                {"keep_intermediates": False},
            )

        assert result is True
        resume.assert_called_once_with(tmp_path)
        setup.assert_not_called()

    def test_payload_pruned_resume_without_final_index_fails_without_mutation(self, tmp_path):
        settings_file = tmp_path / "processing-settings-v5.json"
        settings_file.write_bytes(b"settings")
        diagnostics = tmp_path / "04_stereo_diagnostics"
        diagnostics.mkdir()
        (diagnostics / "metadata.json").write_text('{"status":"payload_pruned"}', encoding="ascii")
        artifact = Mock(status="in_progress", selected_frame_count=3)
        artifact.to_object.return_value = {
            "metadata": {"cleanup_status": "not_requested"},
            "processing_settings": {"keep_intermediates": False},
            "video_properties": {"frame_count": 3},
            "output_info": {"expected_final_relative_path": "result.mp4"},
            "runtime_info": {"metric_clamp_summary": None},
        }
        events = []
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())
        diagnostics_projection = Mock(status="payload_pruned")
        diagnostics_projection.validate_state_contract.return_value = diagnostics_projection

        with (
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "recover_processing_settings_artifact",
                side_effect=lambda _path: events.append("recover") or artifact,
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "project_diagnostics_metadata",
                return_value=diagnostics_projection,
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "audit_final_media_v1",
                side_effect=lambda *_args, **_kwargs: events.append("audit")
                or Mock(final_video_authenticated=True),
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "commit_cleanup_pending_status",
                side_effect=lambda _path, **_kwargs: events.append("pending") or artifact,
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "run_authorized_cleanup",
                side_effect=lambda _path: events.append("delete") or 2,
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "update_processing_status",
                side_effect=lambda *_args, **_kwargs: events.append("terminal") or True,
            ) as update,
        ):
            assert orchestrator._resume_cleanup_only(tmp_path) is False

        assert events == ["recover"]
        update.assert_not_called()

    def test_authenticated_active_index_finishes_before_pipeline_setup(self, tmp_path):
        settings_file = tmp_path / "processing-settings-v5.json"
        settings_file.write_bytes(b"settings")
        active_index = tmp_path / ".depth-surge-reservations-v1/final-encoding-reservation-v1.json"
        active_index.parent.mkdir()
        active_index.write_bytes(b"index")
        artifact = Mock(status="in_progress", selected_frame_count=450)
        artifact.to_object.return_value = {
            "metadata": {"cleanup_status": "not_requested"},
            "processing_settings": {"keep_intermediates": True},
            "video_properties": {"frame_count": 473},
            "output_info": {"expected_final_relative_path": "result.mp4"},
            "runtime_info": {"metric_clamp_summary": None},
        }
        reservation = Mock()
        reservation.requires_explicit_streamed_recovery.return_value = False
        reservation.recover_publication_prefix.return_value = Mock(
            phase="publication_authenticated"
        )
        capability = Mock()
        encoder = Mock()
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), encoder)
        events = []
        capability.validate_authenticated_publication.side_effect = lambda: events.append(
            "active-validate"
        )

        def retire(*, success):
            assert success is True
            active_index.unlink()
            events.append("retire")

        encoder.retire_final_encoding_reservation.side_effect = retire

        with (
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "recover_processing_settings_artifact",
                side_effect=lambda _path: events.append("settings") or artifact,
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "load_final_encoding_reservation_v1",
                side_effect=lambda *_args: events.append("index") or reservation,
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "FinalEncodingReservationCapabilityV1.from_prepared",
                return_value=capability,
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "audit_final_media_v1",
                side_effect=lambda *_args: (
                    (_ for _ in ()).throw(AssertionError("audit ran before index retirement"))
                    if active_index.exists()
                    else events.append("audit") or Mock(final_video_authenticated=True)
                ),
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "update_processing_status",
                side_effect=lambda *_args, **_kwargs: events.append("terminal") or True,
            ) as update,
        ):
            assert orchestrator._resume_cleanup_only(tmp_path) is True

        assert events == [
            "settings",
            "settings",
            "index",
            "active-validate",
            "terminal",
            "settings",
            "retire",
            "settings",
            "audit",
        ]
        capability.validate_authenticated_publication.assert_called_once_with()
        assert update.call_args.args[2]["frames_processed"] == 450
        encoder.adopt_final_encoding_reservation.assert_called_once_with(capability)
        encoder.retire_final_encoding_reservation.assert_called_once_with(success=True)

    def test_cleanup_failure_stays_pending_and_a_later_resume_retries(self, tmp_path):
        from src.depth_surge_3d.rendering.quality.cleanup_transaction import (
            CleanupIdentityMismatchError,
        )

        settings_file = tmp_path / "processing-settings-v5.json"
        settings_file.write_bytes(b"settings")
        active_index = tmp_path / ".depth-surge-reservations-v1/final-encoding-reservation-v1.json"
        active_index.parent.mkdir()
        active_index.write_bytes(b"index")
        artifact = Mock(status="in_progress", selected_frame_count=3)
        artifact.to_object.return_value = {
            "metadata": {"cleanup_status": "pending"},
            "processing_settings": {"keep_intermediates": False},
            "video_properties": {"frame_count": 3},
            "output_info": {"expected_final_relative_path": "result.mp4"},
            "runtime_info": {"metric_clamp_summary": None},
        }
        reservation = Mock()
        reservation.requires_explicit_streamed_recovery.return_value = False
        reservation.recover_publication_prefix.return_value = Mock(
            phase="publication_authenticated"
        )
        capability = Mock()
        diagnostics_projection = Mock(status="payload_pruned")
        diagnostics_projection.validate_state_contract.return_value = diagnostics_projection
        encoder = Mock()
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), encoder)

        with (
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "recover_processing_settings_artifact",
                return_value=artifact,
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "load_final_encoding_reservation_v1",
                return_value=reservation,
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "FinalEncodingReservationCapabilityV1.from_prepared",
                return_value=capability,
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "project_diagnostics_metadata",
                return_value=diagnostics_projection,
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "audit_final_media_v1",
                return_value=Mock(final_video_authenticated=True),
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "run_authorized_cleanup",
                side_effect=[CleanupIdentityMismatchError("changed"), 2],
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "update_processing_status",
                return_value=True,
            ) as update,
        ):
            assert orchestrator._resume_cleanup_only(tmp_path) is False
            update.assert_not_called()
            encoder.retire_final_encoding_reservation.assert_not_called()

            assert orchestrator._resume_cleanup_only(tmp_path) is True

        assert update.call_args.args[1] == "completed"
        assert update.call_args.args[2]["cleanup_status"] == "complete"
        encoder.retire_final_encoding_reservation.assert_called_once_with(success=True)

    def test_cleanup_projection_releases_settings_and_prepared_reservation(
        self,
        tmp_path: Path,
    ) -> None:
        settings_file = tmp_path / "processing-settings-v5.json"
        settings_file.write_bytes(b"settings")
        active_index = tmp_path / ".depth-surge-reservations-v1/final-encoding-reservation-v1.json"
        active_index.parent.mkdir()
        active_index.write_bytes(b"index")
        references: dict[str, weakref.ReferenceType[object]] = {}

        class Artifact:
            status = "in_progress"
            selected_frame_count = 3

            @staticmethod
            def to_object():
                return {
                    "metadata": {"cleanup_status": "not_requested"},
                    "processing_settings": {"keep_intermediates": True},
                    "video_properties": {"frame_count": 3},
                    "runtime_info": {"metric_clamp_summary": None},
                }

        class Prepared:
            @staticmethod
            def requires_explicit_streamed_recovery():
                return False

            @staticmethod
            def recover_publication_prefix():
                from src.depth_surge_3d.io.reservation_protocol import FinalPublicationRecovery

                return FinalPublicationRecovery(
                    "publication_authenticated", "continue-finalization"
                )

        capability = Mock()
        encoder = Mock()
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), encoder)

        def recover(_path):
            return Artifact()

        def load(_root, settings):
            prepared = Prepared()
            references["settings"] = weakref.ref(settings)
            references["reservation"] = weakref.ref(prepared)
            return prepared

        def project(_path):
            gc.collect()
            assert references["settings"]() is None
            assert references["reservation"]() is None
            projection = Mock(status="complete")
            projection.validate_state_contract.return_value = projection
            return projection

        def retire(*, success):
            assert success is True
            active_index.unlink()

        encoder.retire_final_encoding_reservation.side_effect = retire
        with (
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "recover_processing_settings_artifact",
                new=recover,
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "load_final_encoding_reservation_v1",
                new=load,
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "FinalEncodingReservationCapabilityV1.from_prepared",
                new=lambda _prepared: capability,
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "project_diagnostics_metadata",
                new=project,
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "update_processing_status",
                return_value=True,
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "audit_final_media_v1",
                return_value=Mock(final_video_authenticated=True),
            ),
        ):
            assert orchestrator._resume_cleanup_only(tmp_path) is True

        capability.validate_authenticated_publication.assert_called_once_with()
        encoder.adopt_final_encoding_reservation.assert_called_once_with(capability)


class TestExecutePipeline:
    @staticmethod
    def _streamed_stereo_settings(**overrides):
        settings = {
            "apply_distortion": True,
            "crop_factor": 1.0,
            "direct_vr_encode": True,
            "fisheye_projection": "equisolid",
            "keep_intermediates": False,
            "per_eye_height": 6,
            "per_eye_width": 8,
            "stereo_geometry_mode": "relative",
            "stereo_render_mode": "fast",
            "upscale_model": "none",
            "vr_format": "side_by_side",
            "vr_output_height": 6,
            "vr_output_width": 16,
            "vr_resolution": "custom",
        }
        settings.update(overrides)
        return settings

    @pytest.mark.parametrize(
        ("settings_overrides", "manifest_overrides", "schema5", "active_reservation"),
        [
            ({"stereo_render_mode": "draft"}, {}, True, False),
            ({"stereo_geometry_mode": "metric_camera"}, {}, True, False),
            ({}, {"geometry_mode": "metric"}, True, False),
            ({"direct_vr_encode": False}, {}, True, False),
            ({"keep_intermediates": True}, {}, True, False),
            ({"upscale_model": "realesrgan"}, {}, True, False),
            ({"vr_format": "unknown"}, {}, True, False),
            ({"vr_output_width": 15}, {}, True, False),
            ({"apply_distortion": False, "crop_factor": 0.9}, {}, True, False),
            (
                {"apply_distortion": False, "crop_factor": 1.0},
                {"render_shape": (7, 8)},
                True,
                False,
            ),
            ({}, {}, False, False),
            ({}, {}, True, True),
        ],
    )
    def test_streamed_stereo_eligibility_rejects_every_unsupported_gate(
        self,
        settings_overrides,
        manifest_overrides,
        schema5,
        active_reservation,
    ):
        settings = self._streamed_stereo_settings(**settings_overrides)
        manifest_values = {
            "frame_count": 2,
            "geometry_mode": "relative",
            "render_shape": (6, 8),
        }
        manifest_values.update(manifest_overrides)

        assert not ProcessingOrchestrator._streamed_stereo_settings_eligible(
            settings,
            SimpleNamespace(**manifest_values),
            schema5_settings=schema5,
            active_reservation=active_reservation,
        )

    @pytest.mark.parametrize("render_mode", ["fast", "quality"])
    def test_streamed_stereo_eligibility_accepts_supported_render_modes(
        self,
        render_mode,
    ):
        manifest = SimpleNamespace(
            frame_count=2,
            geometry_mode="relative",
            render_shape=(6, 8),
        )
        assert ProcessingOrchestrator._streamed_stereo_settings_eligible(
            self._streamed_stereo_settings(stereo_render_mode=render_mode),
            manifest,
            schema5_settings=True,
            active_reservation=False,
        )
        assert ProcessingOrchestrator._streamed_stereo_settings_eligible(
            self._streamed_stereo_settings(
                stereo_render_mode=render_mode,
                apply_distortion=False,
                crop_factor=1.0,
            ),
            manifest,
            schema5_settings=True,
            active_reservation=False,
        )

    def test_streamed_stereo_consumer_bound_charges_maps_transform_and_pipe_copies(self):
        settings = self._streamed_stereo_settings(
            per_eye_width=1920,
            per_eye_height=1080,
            vr_output_width=3840,
            vr_output_height=1080,
        )
        source_pixels = 1920 * 1080
        target_pixels = 1920 * 1080
        final_pixels = 3840 * 1080
        map_bytes = 8 * source_pixels
        transform_peak = 3 * source_pixels + 6 * target_pixels
        pipe_peak = 6 * target_pixels + 9 * final_pixels

        assert ProcessingOrchestrator._streamed_stereo_consumer_memory_bound(
            (1080, 1920),
            settings,
        ) == map_bytes + max(transform_peak, pipe_peak)

    def test_streamed_stereo_session_precedes_render_and_publishes_after_generation(self, tmp_path):
        events: list[str] = []
        session = Mock()

        def write(frame_name, frame):
            events.append("write")
            assert frame_name == "frame_000001"
            assert frame.shape == (6, 16, 3)
            assert np.all(frame[:, :8] == 17)
            assert np.all(frame[:, 8:] == 29)

        session.write.side_effect = write
        session.finish_and_publish.side_effect = lambda: events.append("publish") or True
        encoder = Mock()
        encoder.pending_final_encoding_reservation = None
        encoder.open_streamed_vr_session.side_effect = (
            lambda **_kwargs: events.append("open") or session
        )
        stereo = Mock()

        def render(*_args, frame_consumer=None, frame_consumer_memory_bytes=0):
            events.append("render")
            assert frame_consumer is not None
            assert frame_consumer_memory_bytes == (
                ProcessingOrchestrator._streamed_stereo_consumer_memory_bound(
                    (6, 8),
                    settings,
                )
            )
            frame_consumer(
                "frame_000001",
                np.full((6, 8, 3), 17, np.uint8),
                np.full((6, 8, 3), 29, np.uint8),
            )
            events.append("render-complete")
            return True

        stereo.create_stereo_pairs_from_manifest.side_effect = render
        distortion = Mock()
        distortion.prepare_stream_transform.side_effect = lambda **_kwargs: events.append(
            "prepare"
        ) or (lambda left, right: (left, right))
        orchestrator = ProcessingOrchestrator(Mock(), stereo, distortion, Mock(), Mock(), encoder)
        orchestrator._settings_file = tmp_path / "processing-settings-v5.json"
        settings = self._streamed_stereo_settings()
        directories = {
            "base": tmp_path,
            "left_frames": tmp_path / "04_left_frames",
            "right_frames": tmp_path / "04_right_frames",
        }
        manifest = SimpleNamespace(
            frame_count=1,
            geometry_mode="relative",
            render_shape=(6, 8),
            replay_declared_records=lambda: iter((SimpleNamespace(frame_name="frame_000001"),)),
        )
        progress = Mock()

        with (
            patch.object(orchestrator, "_execute_post_stereo_steps") as legacy_post,
            patch.object(orchestrator, "_finalize_processing", return_value=True) as finalize,
        ):
            assert orchestrator._execute_remaining_steps(
                directories,
                settings,
                manifest,
                "source.mp4",
                tmp_path,
                progress,
            )

        assert events == [
            "prepare",
            "open",
            "render",
            "write",
            "render-complete",
            "publish",
        ]
        session.abort.assert_not_called()
        legacy_post.assert_not_called()
        encoder.create_video.assert_not_called()
        encoder.create_video_from_stereo_source.assert_not_called()
        encoder.open_streamed_vr_session.assert_called_once_with(
            output_dir=tmp_path,
            original_video="source.mp4",
            settings=settings,
            frame_count=1,
            first_index=1,
            progress_tracker=progress,
        )
        finalize.assert_called_once_with(True, tmp_path, "source.mp4", settings, 1)
        progress.finish.assert_called_once_with("Video processing complete")

    def test_streamed_stereo_aborts_session_when_generation_fails(self, tmp_path):
        events = []
        session = Mock()
        session.abort.side_effect = lambda: events.append("abort")
        encoder = Mock()
        encoder.pending_final_encoding_reservation = None
        encoder.open_streamed_vr_session.return_value = session
        stereo = Mock()
        stereo.create_stereo_pairs_from_manifest.return_value = False
        distortion = Mock()
        distortion.prepare_stream_transform.return_value = lambda left, right: (left, right)
        orchestrator = ProcessingOrchestrator(Mock(), stereo, distortion, Mock(), Mock(), encoder)
        orchestrator._settings_file = tmp_path / "processing-settings-v5.json"
        manifest = SimpleNamespace(
            frame_count=1,
            geometry_mode="relative",
            render_shape=(6, 8),
            replay_declared_records=lambda: iter((SimpleNamespace(frame_name="frame_000001"),)),
        )

        with patch.object(
            orchestrator,
            "_handle_step_error",
            side_effect=lambda message: events.append(("failed", message)) or False,
        ) as failure:
            assert not orchestrator._execute_remaining_steps(
                {
                    "base": tmp_path,
                    "left_frames": tmp_path / "04_left_frames",
                    "right_frames": tmp_path / "04_right_frames",
                },
                self._streamed_stereo_settings(),
                manifest,
                "source.mp4",
                tmp_path,
            )

        session.abort.assert_called_once_with()
        session.finish_and_publish.assert_not_called()
        failure.assert_called_once_with("Stereo pair creation failed")
        assert events == ["abort", ("failed", "Stereo pair creation failed")]

    def test_streamed_stereo_does_not_terminalize_when_abort_is_incomplete(self, tmp_path):
        session = Mock()
        session.abort.side_effect = ReservationIncompleteError("streamed cleanup remains retryable")
        encoder = Mock()
        encoder.pending_final_encoding_reservation = None
        encoder.open_streamed_vr_session.return_value = session
        stereo = Mock()
        stereo.create_stereo_pairs_from_manifest.return_value = False
        distortion = Mock()
        distortion.prepare_stream_transform.return_value = lambda left, right: (left, right)
        orchestrator = ProcessingOrchestrator(Mock(), stereo, distortion, Mock(), Mock(), encoder)
        orchestrator._settings_file = tmp_path / "processing-settings-v5.json"
        manifest = SimpleNamespace(
            frame_count=1,
            geometry_mode="relative",
            render_shape=(6, 8),
            replay_declared_records=lambda: iter((SimpleNamespace(frame_name="frame_000001"),)),
        )

        with patch.object(orchestrator, "_handle_step_error") as failure:
            with pytest.raises(ReservationIncompleteError, match="retryable"):
                orchestrator._execute_remaining_steps(
                    {
                        "base": tmp_path,
                        "left_frames": tmp_path / "04_left_frames",
                        "right_frames": tmp_path / "04_right_frames",
                    },
                    self._streamed_stereo_settings(),
                    manifest,
                    "source.mp4",
                    tmp_path,
                )

        session.abort.assert_called_once_with()
        session.finish_and_publish.assert_not_called()
        failure.assert_not_called()

    def test_streamed_stereo_capacity_rejection_discards_session_and_falls_back(self, tmp_path):
        session = Mock()
        encoder = Mock()
        encoder.pending_final_encoding_reservation = None
        encoder.open_streamed_vr_session.return_value = session
        stereo = Mock()

        def render(*_args, frame_consumer=None, **_kwargs):
            if frame_consumer is not None:
                raise StereoPipelineCapacityError(
                    "stream consumer leaves no stereo lifecycle capacity"
                )
            return True

        stereo.create_stereo_pairs_from_manifest.side_effect = render
        distortion = Mock()
        distortion.prepare_stream_transform.return_value = lambda left, right: (left, right)
        orchestrator = ProcessingOrchestrator(Mock(), stereo, distortion, Mock(), Mock(), encoder)
        orchestrator._settings_file = tmp_path / "processing-settings-v5.json"
        manifest = SimpleNamespace(
            frame_count=1,
            geometry_mode="relative",
            render_shape=(6, 8),
            replay_declared_records=lambda: iter((SimpleNamespace(frame_name="frame_000001"),)),
        )

        with patch.object(orchestrator, "_execute_post_stereo_steps", return_value=True) as post:
            assert orchestrator._execute_remaining_steps(
                {
                    "base": tmp_path,
                    "left_frames": tmp_path / "04_left_frames",
                    "right_frames": tmp_path / "04_right_frames",
                },
                self._streamed_stereo_settings(),
                manifest,
                "source.mp4",
                tmp_path,
            )

        session.discard_for_file_backed_fallback.assert_called_once_with()
        session.abort.assert_not_called()
        assert stereo.create_stereo_pairs_from_manifest.call_count == 2
        post.assert_called_once()

    def test_upstream_path_lists_are_released_before_stereo_preflight(self, tmp_path):
        """The bounded stereo phase cannot inherit either upstream O(N) graph."""

        class TrackedPaths(list):
            pass

        references = {}

        class VideoEncoder:
            defer_final_reservation_retirement = False

            @staticmethod
            def extract_frames(*_args):
                paths = TrackedPaths([tmp_path / "frame_000001.png"])
                references["frames"] = weakref.ref(paths)
                return paths

        class DepthProcessor:
            @staticmethod
            def generate_depth_map_files(*_args):
                paths = TrackedPaths([tmp_path / "depth_000001.png"])
                references["geometry"] = weakref.ref(paths)
                return paths

        class StereoGenerator:
            @staticmethod
            def create_stereo_pairs_from_manifest(*_args):
                gc.collect()
                assert references["frames"]() is None
                assert references["geometry"]() is None
                return False

        def publish_manifest(**_kwargs):
            return SimpleNamespace(frame_count=1)

        orchestrator = ProcessingOrchestrator(
            DepthProcessor(),
            StereoGenerator(),
            Mock(),
            Mock(),
            Mock(),
            VideoEncoder(),
        )

        with patch(
            "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
            "publish_validated_stereo_input_manifest_v1",
            new=publish_manifest,
        ):
            assert not orchestrator._execute_pipeline(
                "source.mp4",
                tmp_path,
                {"base": tmp_path, "frames": tmp_path / "frames"},
                {"fps": 30.0, "frame_count": 1},
                {},
            )


@pytest.mark.parametrize("fps_settings", [{}, {"target_fps": None}, {"target_fps": "original"}])
def test_legacy_fast_adoption_skips_upstream_manifest_and_stereo_provider(
    tmp_path: Path,
    fps_settings,
) -> None:
    adoption = LegacyFastAdoptionV1(
        job_root=tmp_path,
        source_settings_schema_version=4,
        settings_fingerprint="1" * 64,
        frame_count=2,
        render_shape=(4, 6),
        geometry_mode="relative",
        occlusion_fill="background",
        rgb_stage_fingerprint="2" * 64,
        source_guide_fingerprint="3" * 64,
        stage_metadata_sha256="4" * 64,
        payload_sequence_sha256="5" * 64,
        metric_clamp_summary_sha256=None,
        metric_clamp_evidence_sha256=None,
        metric_valid_pixel_count=None,
        metric_clamped_pixel_count=None,
        metric_affected_frame_count=None,
        metric_mean_clamped_fraction=None,
        metric_max_clamped_fraction=None,
    )
    encoder = Mock()
    stereo = Mock()
    orchestrator = ProcessingOrchestrator(
        Mock(),
        stereo,
        Mock(),
        Mock(),
        Mock(),
        encoder,
        legacy_fast_adoption=adoption,
    )
    settings_file = tmp_path / "processing-settings-v5.json"
    directories = {
        "base": tmp_path,
        "left_frames": tmp_path / "04_left_frames",
        "right_frames": tmp_path / "04_right_frames",
    }
    lock = SimpleNamespace(is_acquired=True, output_dir=tmp_path)
    artifact = object()

    with (
        patch(
            "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.recover_processing_control_prefix_v1"
        ),
        patch(
            "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.cleanup_vdpp_private_work"
        ),
        patch.object(orchestrator, "_resume_cleanup_only", return_value=None),
        patch.object(
            orchestrator,
            "_setup_processing",
            return_value=(tmp_path, directories, settings_file),
        ),
        patch(
            "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.recover_processing_settings_artifact",
            return_value=artifact,
        ),
        patch(
            "src.depth_surge_3d.io.resume.revalidate_legacy_fast_adoption_v1",
            return_value=adoption,
        ) as revalidate,
        patch(
            "src.depth_surge_3d.io.resume.iter_legacy_fast_frame_names",
            side_effect=lambda _adoption: iter(("frame_000000", "frame_000001")),
        ),
        patch(
            "src.depth_surge_3d.io.resume.legacy_fast_metric_projection",
            return_value=None,
        ),
        patch(
            "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.publish_legacy_fast_unavailable"
        ) as publish,
        patch.object(
            orchestrator,
            "_execute_post_stereo_steps",
            return_value=True,
            create=True,
        ) as post_stereo,
        patch.object(orchestrator, "_execute_pipeline", return_value=False) as pipeline,
    ):
        assert orchestrator.process(
            tmp_path / "source.mp4",
            tmp_path,
            {"fps": 30.0},
            {"keep_intermediates": False, **fps_settings},
            job_lock=lock,
        )

    revalidate.assert_called_once_with(
        adoption,
        artifact,
        allow_diagnostics_prefix=True,
    )
    publish.assert_called_once()
    post_stereo.assert_called_once()
    pipeline.assert_not_called()
    encoder.extract_frames.assert_not_called()
    stereo.create_stereo_pairs_from_manifest.assert_not_called()


class TestPipelineStages:
    def test_releases_depth_model_before_starting_step_3(self, tmp_path):
        """The depth model leaves memory as soon as its pipeline stage is complete."""

        class LoadedDepthModel:
            def __init__(self):
                self.loaded = True

            def unload_model(self):
                self.loaded = False

        class DepthProcessor:
            @staticmethod
            def generate_depth_map_files(*_args):
                return [tmp_path / "depth_000001.png"]

        class StereoGenerator:
            def __init__(self, model):
                self.model = model
                self.model_loaded_at_step_3 = None

            def create_stereo_pairs_from_manifest(self, *_args):
                self.model_loaded_at_step_3 = self.model.loaded
                return False

        model = LoadedDepthModel()
        stereo_generator = StereoGenerator(model)
        video_encoder = Mock()
        video_encoder.extract_frames.return_value = [tmp_path / "frame_000001.png"]
        orchestrator = ProcessingOrchestrator(
            DepthProcessor(),
            stereo_generator,
            Mock(),
            Mock(),
            Mock(),
            video_encoder,
            release_depth_model=model.unload_model,
        )

        with patch(
            "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
            "publish_validated_stereo_input_manifest_v1",
            return_value=Mock(frame_count=1),
        ):
            result = orchestrator._execute_pipeline(
                "source.mp4",
                tmp_path,
                {"base": tmp_path, "frames": tmp_path / "frames"},
                {"fps": 30.0, "frame_count": 1},
                {},
            )

        assert result is False
        assert model.loaded is False
        assert stereo_generator.model_loaded_at_step_3 is False

    def test_vdpp_runs_after_owner_release_and_before_stereo(self, tmp_path):
        events = []
        frame = tmp_path / "frame_000001.png"
        base = tmp_path / "base_000001.png"
        stable = tmp_path / "stable_000001.png"

        depth_processor = Mock()
        depth_processor.generate_depth_map_files.return_value = base_files = [base]
        temporal = Mock()

        def stabilize(*args):
            events.append("vdpp")
            assert events == ["release", "vdpp"]
            assert args[0] == base_files
            return [stable]

        temporal.generate_files.side_effect = stabilize
        stereo = Mock()

        def render(_manifest, *_args):
            events.append("stereo")
            return False

        stereo.create_stereo_pairs_from_manifest.side_effect = render
        encoder = Mock()
        encoder.extract_frames.return_value = [frame]
        orchestrator = ProcessingOrchestrator(
            depth_processor,
            stereo,
            Mock(),
            Mock(),
            Mock(),
            encoder,
            release_depth_model=lambda: events.append("release"),
            temporal_stabilizer=temporal,
        )

        with patch(
            "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
            "publish_validated_stereo_input_manifest_v1",
            return_value=Mock(frame_count=1),
        ) as publish_stereo_input:
            assert not orchestrator._execute_pipeline(
                "source.mp4",
                tmp_path,
                {"base": tmp_path, "frames": tmp_path / "frames"},
                {"fps": 30.0, "frame_count": 1},
                {"temporal_postprocessor": "vdpp"},
            )

        assert events == ["release", "vdpp", "stereo"]
        assert publish_stereo_input.call_args.kwargs["geometry_files"] == [stable]

    def test_vdpp_failure_never_starts_stereo(self, tmp_path):
        depth_processor = Mock()
        depth_processor.generate_depth_map_files.return_value = [tmp_path / "base.png"]
        temporal = Mock()
        temporal.generate_files.side_effect = RuntimeError("VDPP failed")
        stereo = Mock()
        encoder = Mock()
        encoder.extract_frames.return_value = [tmp_path / "frame.png"]
        orchestrator = ProcessingOrchestrator(
            depth_processor,
            stereo,
            Mock(),
            Mock(),
            Mock(),
            encoder,
            release_depth_model=Mock(),
            temporal_stabilizer=temporal,
        )

        with pytest.raises(RuntimeError, match="VDPP failed"):
            orchestrator._execute_pipeline(
                "source.mp4",
                tmp_path,
                {"base": tmp_path, "frames": tmp_path / "frames"},
                {"fps": 30.0, "frame_count": 1},
                {"temporal_postprocessor": "vdpp"},
            )

        stereo.create_stereo_pairs_from_files.assert_not_called()

    @staticmethod
    def _direct_encoding_dependencies(tmp_path):
        frame = tmp_path / "frame_000001.png"
        depth = tmp_path / "depth_000001.png"
        left = tmp_path / "left_cropped" / "frame_000001.png"
        right = tmp_path / "right_cropped" / "frame_000001.png"
        directories = {
            "base": tmp_path,
            "left_frames": tmp_path / "left",
            "right_frames": tmp_path / "right",
            "left_cropped": tmp_path / "left_cropped",
            "right_cropped": tmp_path / "right_cropped",
        }
        settings = {
            "apply_distortion": False,
            "upscale_model": "none",
            "per_eye_width": 8,
            "per_eye_height": 8,
            "vr_format": "side_by_side",
            "vr_output_width": 16,
            "vr_output_height": 8,
            "vr_resolution": "custom",
            "keep_intermediates": True,
            "direct_vr_encode": True,
        }
        stereo_generator = Mock()
        stereo_generator.create_stereo_pairs_from_files.return_value = True
        stereo_generator.create_stereo_pairs_from_manifest.return_value = True
        distortion_processor = Mock()
        distortion_processor.crop_frames.return_value = True
        vr_assembler = Mock()
        video_encoder = Mock()
        orchestrator = ProcessingOrchestrator(
            Mock(),
            stereo_generator,
            distortion_processor,
            Mock(),
            vr_assembler,
            video_encoder,
        )
        return (
            orchestrator,
            directories,
            settings,
            frame,
            depth,
            left,
            right,
            vr_assembler,
            video_encoder,
        )

    def test_direct_encoding_uses_resolved_stereo_sources(self, tmp_path, capsys):
        """Direct mode sends resolved pairs to FFmpeg without assembling VR frames."""
        (
            orchestrator,
            directories,
            settings,
            frame,
            depth,
            left,
            right,
            vr_assembler,
            video_encoder,
        ) = self._direct_encoding_dependencies(tmp_path)
        progress_tracker = Mock()
        direct_source = Mock()
        vr_assembler.resolve_direct_vr_source.return_value = direct_source
        video_encoder.create_video_from_stereo_source.return_value = True

        with patch.object(orchestrator, "_finalize_processing"):
            assert orchestrator._execute_remaining_steps(
                directories,
                settings,
                Mock(frame_count=1),
                "source.mp4",
                directories["base"],
                progress_tracker,
            )

        vr_assembler.assemble_vr_frames.assert_not_called()
        vr_assembler.resolve_direct_vr_source.assert_called_once_with(directories, settings, 1)
        video_encoder.create_video.assert_not_called()
        video_encoder.create_video_from_stereo_source.assert_called_once_with(
            direct_source,
            directories["base"],
            "source.mp4",
            settings,
            total_frames=1,
            progress_tracker=progress_tracker,
        )
        assert "VR frames" not in capsys.readouterr().out

    def test_metric_stereo_prints_exactly_one_clamp_summary_line(self, tmp_path, capsys):
        (
            orchestrator,
            directories,
            settings,
            frame,
            depth,
            left,
            right,
            vr_assembler,
            _video_encoder,
        ) = self._direct_encoding_dependencies(tmp_path)
        settings["stereo_geometry_mode"] = "metric_camera"
        summary = {
            "affected_frame_count": 2,
            "mean_clamped_fraction": 0.125,
            "max_clamped_fraction": 0.25,
        }
        orchestrator.stereo_generator.last_metric_clamp_summary = summary
        vr_assembler.resolve_direct_vr_source.return_value = Mock()

        with patch.object(orchestrator, "_finalize_processing"):
            assert orchestrator._execute_remaining_steps(
                directories,
                settings,
                Mock(frame_count=1),
                "source.mp4",
                directories["base"],
            )

        expected = (
            "Metric disparity clamp summary: affected_frames=2, " "mean=12.5000%, max=25.0000%"
        )
        assert capsys.readouterr().out.count(expected) == 1
        assert orchestrator._metric_clamp_summary == summary

    def test_relative_stereo_does_not_report_stale_metric_clamp_summary(self, tmp_path, capsys):
        (
            orchestrator,
            directories,
            settings,
            frame,
            depth,
            left,
            right,
            vr_assembler,
            _video_encoder,
        ) = self._direct_encoding_dependencies(tmp_path)
        orchestrator.stereo_generator.last_metric_clamp_summary = {
            "affected_frame_count": 1,
            "mean_clamped_fraction": 1.0,
            "max_clamped_fraction": 1.0,
        }
        vr_assembler.resolve_direct_vr_source.return_value = Mock()

        with patch.object(orchestrator, "_finalize_processing"):
            assert orchestrator._execute_remaining_steps(
                directories,
                settings,
                Mock(frame_count=1),
                "source.mp4",
                directories["base"],
            )

        assert "Metric disparity clamp summary:" not in capsys.readouterr().out
        assert orchestrator._metric_clamp_summary is None

    def test_direct_encoding_stops_when_stereo_sources_are_invalid(self, tmp_path):
        """Direct source validation failure never starts either video encoder path."""
        (
            orchestrator,
            directories,
            settings,
            frame,
            depth,
            _left,
            _right,
            vr_assembler,
            video_encoder,
        ) = self._direct_encoding_dependencies(tmp_path)
        vr_assembler.resolve_direct_vr_source.return_value = None

        assert not orchestrator._execute_remaining_steps(
            directories,
            settings,
            Mock(frame_count=1),
            "source.mp4",
            directories["base"],
        )

        vr_assembler.assemble_vr_frames.assert_not_called()
        video_encoder.create_video.assert_not_called()
        video_encoder.create_video_from_stereo_source.assert_not_called()

    def test_direct_encoding_failure_finalizes_without_cleanup(self, tmp_path):
        """A failed direct FFmpeg encode uses failure finalization and keeps intermediates."""
        (
            orchestrator,
            directories,
            settings,
            frame,
            depth,
            left,
            right,
            vr_assembler,
            video_encoder,
        ) = self._direct_encoding_dependencies(tmp_path)
        settings["keep_intermediates"] = False
        progress_tracker = Mock()
        vr_assembler.resolve_direct_vr_source.return_value = Mock()
        video_encoder.create_video_from_stereo_source.return_value = False

        with (
            patch.object(
                orchestrator,
                "_finalize_processing",
                wraps=orchestrator._finalize_processing,
            ) as finalize,
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.cleanup_intermediate_files"
            ) as cleanup,
        ):
            assert not orchestrator._execute_remaining_steps(
                directories,
                settings,
                Mock(frame_count=1),
                "source.mp4",
                directories["base"],
                progress_tracker,
            )

        finalize.assert_called_once_with(
            False,
            directories["base"],
            "source.mp4",
            settings,
            1,
        )
        progress_tracker.finish.assert_not_called()
        cleanup.assert_not_called()

    def test_required_distortion_stage_fails_when_stereo_outputs_are_missing(self, tmp_path):
        stereo_generator = Mock()
        stereo_generator.create_stereo_pairs_from_manifest.return_value = True
        distortion_processor = Mock()
        orchestrator = ProcessingOrchestrator(
            Mock(), stereo_generator, distortion_processor, Mock(), Mock(), Mock()
        )
        left_dir = tmp_path / "left"
        right_dir = tmp_path / "right"
        left_dir.mkdir()
        right_dir.mkdir()

        result = orchestrator._execute_remaining_steps(
            {
                "base": tmp_path,
                "left_frames": left_dir,
                "right_frames": right_dir,
            },
            {"apply_distortion": True},
            Mock(frame_count=1),
            "source.mp4",
            tmp_path,
        )

        assert result is False
        distortion_processor.apply_distortion.assert_not_called()

    @pytest.mark.parametrize(
        ("geometry_mode", "stage_key", "suffix", "saved_label"),
        [
            ("relative", "disparity_maps", ".png", "Canonical disparity maps"),
            ("metric_camera", "metric_geometry", ".npz", "Metric geometry frames"),
        ],
    )
    def test_pipeline_uses_disk_backed_depth_and_stereo_apis(
        self, tmp_path, geometry_mode, stage_key, suffix, saved_label
    ):
        """The video path never builds whole-video frame or depth arrays."""
        frame_files = [tmp_path / "frame_000001.png"]
        geometry_files = [tmp_path / f"depth_000001{suffix}"]
        depth_processor = Mock()
        depth_processor.generate_depth_map_files.return_value = geometry_files
        depth_processor.generate_depth_maps.side_effect = AssertionError("array depth API used")
        stereo_generator = Mock()
        stereo_generator.create_stereo_pairs_from_files.return_value = True
        stereo_generator.create_stereo_pairs.side_effect = AssertionError("array stereo API used")
        distortion_processor = Mock()
        distortion_processor.crop_frames.return_value = True
        vr_assembler = Mock()
        vr_assembler.assemble_vr_frames.return_value = True
        video_encoder = Mock()
        video_encoder.extract_frames.return_value = frame_files
        video_encoder.create_video.return_value = True
        directories = {
            "base": tmp_path,
            "frames": tmp_path / "frames",
            "disparity_maps": tmp_path / "disparity",
            "metric_geometry": tmp_path / "metric_geometry",
            "left_frames": tmp_path / "left",
            "right_frames": tmp_path / "right",
            "left_cropped": tmp_path / "left_cropped",
            "right_cropped": tmp_path / "right_cropped",
            "vr_frames": tmp_path / "vr",
        }
        settings = {
            "apply_distortion": False,
            "upscale_model": "none",
            "per_eye_width": 8,
            "per_eye_height": 8,
            "vr_format": "side_by_side",
            "vr_output_width": 16,
            "vr_output_height": 8,
            "vr_resolution": "custom",
            "keep_intermediates": True,
            "stereo_geometry_mode": geometry_mode,
        }
        orchestrator = ProcessingOrchestrator(
            depth_processor,
            stereo_generator,
            distortion_processor,
            Mock(),
            vr_assembler,
            video_encoder,
        )

        stereo_input_manifest = Mock(frame_count=1)
        with (
            patch.object(orchestrator, "_finalize_processing"),
            patch.object(orchestrator, "_print_saved_to") as saved_to,
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "publish_validated_stereo_input_manifest_v1",
                return_value=stereo_input_manifest,
                create=True,
            ) as publish_stereo_input,
        ):
            result = orchestrator._execute_pipeline(
                "source.mp4",
                tmp_path,
                directories,
                {"fps": 30.0, "frame_count": 1},
                settings,
            )

        assert result is True
        depth_processor.generate_depth_map_files.assert_called_once()
        publish_stereo_input.assert_called_once_with(
            job_root=directories["base"],
            guide_files=frame_files,
            geometry_files=geometry_files,
            geometry_mode=geometry_mode,
        )
        stereo_generator.create_stereo_pairs_from_manifest.assert_called_once_with(
            stereo_input_manifest,
            directories,
            settings,
            None,
        )
        stereo_generator.create_stereo_pairs_from_files.assert_not_called()
        video_encoder.create_video_from_stereo_source.assert_not_called()
        saved_to.assert_any_call(directories[stage_key], saved_label)


def test_failed_direct_encoding_does_not_finish_progress(tmp_path: Path) -> None:
    stereo_generator = Mock()
    stereo_generator.create_stereo_pairs_from_manifest.return_value = True
    distortion_processor = Mock()
    distortion_processor.crop_frames.return_value = True
    vr_assembler = Mock()
    vr_assembler.resolve_direct_vr_source.return_value = Mock()
    video_encoder = Mock()
    video_encoder.create_video_from_stereo_source.return_value = False
    orchestrator = ProcessingOrchestrator(
        Mock(),
        stereo_generator,
        distortion_processor,
        Mock(),
        vr_assembler,
        video_encoder,
    )
    directories = {
        "base": tmp_path,
        "left_frames": tmp_path / "left",
        "right_frames": tmp_path / "right",
        "left_cropped": tmp_path / "left_cropped",
        "right_cropped": tmp_path / "right_cropped",
    }
    settings = {
        "apply_distortion": False,
        "upscale_model": "none",
        "per_eye_width": 8,
        "per_eye_height": 8,
        "vr_format": "side_by_side",
        "vr_output_width": 16,
        "vr_output_height": 8,
        "vr_resolution": "custom",
        "keep_intermediates": True,
        "direct_vr_encode": True,
    }
    progress_tracker = Mock()

    with patch.object(orchestrator, "_finalize_processing"):
        result = orchestrator._execute_remaining_steps(
            directories,
            settings,
            Mock(frame_count=1),
            "source.mp4",
            tmp_path,
            progress_tracker,
        )

    assert result is False
    progress_tracker.finish.assert_not_called()


def test_distortion_pipeline_fuses_stage_five_and_crop(tmp_path: Path) -> None:
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_cropped = tmp_path / "left_cropped"
    right_cropped = tmp_path / "right_cropped"
    for directory in (left_dir, right_dir, left_cropped, right_cropped):
        directory.mkdir()
    left_file = left_dir / "frame_000001.png"
    right_file = right_dir / "frame_000001.png"
    left_file.touch()
    right_file.touch()

    distortion_processor = Mock()
    distortion_processor.apply_distortion_and_crop.return_value = True
    distortion_processor.apply_distortion.return_value = True
    distortion_processor.crop_frames.return_value = True
    vr_assembler = Mock()
    direct_source = Mock()
    vr_assembler.resolve_direct_vr_source.return_value = direct_source
    video_encoder = Mock()
    video_encoder.create_video_from_stereo_source.return_value = True
    orchestrator = ProcessingOrchestrator(
        Mock(), Mock(), distortion_processor, Mock(), vr_assembler, video_encoder
    )
    directories = {
        "base": tmp_path,
        "left_frames": left_dir,
        "right_frames": right_dir,
        "left_cropped": left_cropped,
        "right_cropped": right_cropped,
    }
    settings = {
        "apply_distortion": True,
        "fisheye_projection": "equisolid",
        "upscale_model": "none",
        "per_eye_width": 8,
        "per_eye_height": 8,
        "direct_vr_encode": True,
        "vr_format": "side_by_side",
        "vr_resolution": "custom",
    }

    with patch.object(orchestrator, "_finalize_processing"):
        assert orchestrator._execute_post_stereo_steps(
            directories,
            settings,
            1,
            "source.mp4",
            tmp_path,
        )

    distortion_processor.apply_distortion_and_crop.assert_called_once_with(
        [left_file], [right_file], directories, settings, None
    )
    distortion_processor.apply_distortion.assert_not_called()
    distortion_processor.crop_frames.assert_not_called()


class TestHandleStepError:
    """Test _handle_step_error method."""

    @patch(
        "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.update_processing_status"
    )
    def test_handle_step_error(self, mock_update_status):
        """Test handle step error updates settings file."""
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())
        orchestrator._settings_file = Path("/output/settings.json")

        result = orchestrator._handle_step_error("Test error message")

        assert result is False
        mock_update_status.assert_called_once_with(
            Path("/output/settings.json"), "failed", {"error": "Test error message"}
        )

    def test_handle_step_error_no_settings_file(self):
        """Test handle step error without settings file."""
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())
        orchestrator._settings_file = None

        result = orchestrator._handle_step_error("Test error message")

        assert result is False


class TestPrintMethods:
    """Test console output helper methods."""

    def test_print_saved_to_with_directory(self, capsys):
        """Test _print_saved_to with valid directory."""
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())

        orchestrator._print_saved_to(Path("/output/dir"), "Test prefix")

        captured = capsys.readouterr()
        assert f"Test prefix: {Path('/output/dir')}" in captured.out

    def test_print_saved_to_with_none(self, capsys):
        """Test _print_saved_to with None directory."""
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())

        orchestrator._print_saved_to(None, "Test prefix")

        captured = capsys.readouterr()
        assert captured.out == ""


class TestProcessMethod:
    """Test main process method."""

    def test_process_releases_depth_model_when_step_1_raises(self, tmp_path):
        """A failure before depth inference must not strand a preloaded model."""

        class LoadedDepthModel:
            def __init__(self):
                self.loaded = True
                self.unload_count = 0

            def unload_model(self):
                self.loaded = False
                self.unload_count += 1

        model = LoadedDepthModel()
        video_encoder = Mock()
        video_encoder.extract_frames.side_effect = OSError("frame extraction failed")
        orchestrator = ProcessingOrchestrator(
            Mock(),
            Mock(),
            Mock(),
            Mock(),
            Mock(),
            video_encoder,
            release_depth_model=model.unload_model,
        )

        with patch.object(
            orchestrator,
            "_setup_processing",
            return_value=(tmp_path, {"base": tmp_path}, tmp_path / "settings.json"),
        ):
            result = orchestrator.process(
                video_path=tmp_path / "source.mp4",
                output_dir=tmp_path,
                video_properties={"fps": 30, "frame_count": 1},
                settings={"keep_intermediates": True},
            )

        assert result is False
        assert model.loaded is False
        assert model.unload_count == 1

    def test_process_recovers_root_control_before_vdpp_or_ordinary_audit(self, tmp_path):
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())
        order: list[str] = []
        lock = JobWriterLock(tmp_path).acquire()
        try:
            with (
                patch(
                    "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                    "recover_processing_control_prefix_v1",
                    side_effect=lambda _root: order.append("control"),
                ),
                patch(
                    "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                    "cleanup_vdpp_private_work",
                    side_effect=lambda _root, _lock: order.append("vdpp"),
                ),
                patch.object(
                    orchestrator,
                    "_resume_cleanup_only",
                    side_effect=lambda _root: order.append("audit") or True,
                ),
            ):
                assert orchestrator.process(
                    video_path=tmp_path / "source.mp4",
                    output_dir=tmp_path,
                    video_properties={"fps": 30, "frame_count": 1},
                    settings={"keep_intermediates": True},
                    job_lock=lock,
                )
        finally:
            lock.release()

        assert order == ["control", "vdpp", "audit"]

    def test_process_reuses_preacquired_writer_lock_without_releasing_it(self, tmp_path):
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())
        lock = JobWriterLock(tmp_path).acquire()
        try:
            with (
                patch.object(
                    orchestrator,
                    "_setup_processing",
                    return_value=(tmp_path, {"base": tmp_path}, None),
                ),
                patch.object(orchestrator, "_execute_pipeline", return_value=True),
                patch(
                    "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.JobWriterLock",
                    side_effect=AssertionError("must reuse the authoritative lock"),
                ),
            ):
                assert orchestrator.process(
                    video_path=tmp_path / "source.mp4",
                    output_dir=tmp_path,
                    video_properties={"fps": 30, "frame_count": 1},
                    settings={"keep_intermediates": True},
                    job_lock=lock,
                )

            assert lock.is_acquired is True
        finally:
            lock.release()

    def test_process_removes_stale_vdpp_work_before_setup_with_supplied_lock(
        self,
        tmp_path,
    ):
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())
        work = tmp_path / "03_disparity_stabilized/.vdpp-work"
        work.mkdir(parents=True)
        (work / "shot_0.raw.f32.mmap").write_bytes(b"stale")
        lock = JobWriterLock(tmp_path).acquire()

        def observe_setup(*_args, **_kwargs):
            assert not work.exists()
            return tmp_path, {"base": tmp_path}, None

        try:
            with (
                patch.object(orchestrator, "_setup_processing", side_effect=observe_setup),
                patch.object(orchestrator, "_execute_pipeline", return_value=True),
            ):
                assert orchestrator.process(
                    video_path=tmp_path / "source.mp4",
                    output_dir=tmp_path,
                    video_properties={"fps": 30, "frame_count": 1},
                    settings={"keep_intermediates": True},
                    job_lock=lock,
                )
        finally:
            lock.release()

    @patch("src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.time.time")
    def test_process_exception_handling(self, mock_time):
        """Test process method handles exceptions."""
        mock_time.return_value = 1000.0

        depth_proc = Mock()
        video_encoder = Mock()
        video_encoder.extract_frames.side_effect = Exception("Test error")

        orchestrator = ProcessingOrchestrator(
            depth_proc, Mock(), Mock(), Mock(), Mock(), video_encoder
        )

        with patch.object(orchestrator, "_setup_processing") as mock_setup:
            mock_setup.return_value = (
                Path("/output"),
                {"base": Path("/output")},
                Path("/output/settings.json"),
            )

            result = orchestrator.process(
                video_path=Path("/input/video.mp4"),
                output_dir=Path("/output"),
                video_properties={"fps": 30, "frame_count": 100},
                settings={"keep_intermediates": True},
            )

            assert result is False

    def test_process_propagates_interrupted_error(self, tmp_path):
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), Mock())

        with (
            patch.object(
                orchestrator,
                "_setup_processing",
                return_value=(tmp_path, {"base": tmp_path}, tmp_path / "settings.json"),
            ),
            patch.object(
                orchestrator,
                "_execute_pipeline",
                side_effect=InterruptedError("Processing stopped by user request"),
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator.update_processing_status"
            ) as update_status,
        ):
            with pytest.raises(InterruptedError, match="stopped by user request"):
                orchestrator.process(
                    video_path=tmp_path / "source.mp4",
                    output_dir=tmp_path,
                    video_properties={"fps": 30, "frame_count": 1},
                    settings={"keep_intermediates": True, "direct_vr_encode": True},
                )

        update_status.assert_not_called()

    def test_process_keeps_reservation_incomplete_retryable_without_terminalizing(self, tmp_path):
        from src.depth_surge_3d.io.reservation_protocol import (
            ReservationIncompleteError,
        )

        video_encoder = Mock()
        orchestrator = ProcessingOrchestrator(Mock(), Mock(), Mock(), Mock(), Mock(), video_encoder)
        settings_file = tmp_path / "processing-settings-v5.json"

        with (
            patch.object(
                orchestrator,
                "_setup_processing",
                return_value=(tmp_path, {"base": tmp_path}, settings_file),
            ),
            patch.object(
                orchestrator,
                "_execute_pipeline",
                side_effect=ReservationIncompleteError("reservation materialization ENOSPC"),
            ),
            patch(
                "src.depth_surge_3d.processing.orchestration.pipeline_orchestrator."
                "update_processing_status"
            ) as update_status,
        ):
            assert (
                orchestrator.process(
                    video_path=tmp_path / "source.mp4",
                    output_dir=tmp_path,
                    video_properties={"fps": 30, "frame_count": 1},
                    settings={"keep_intermediates": False, "direct_vr_encode": True},
                )
                is False
            )

        update_status.assert_not_called()
        video_encoder.retire_final_encoding_reservation.assert_not_called()
