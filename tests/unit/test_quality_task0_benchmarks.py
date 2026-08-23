"""Task 0 scale-matrix, subprocess RSS, and gate-decision contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.quality_task0 import benchmarks, reporting


def test_geometry_scale_matrix_is_the_closed_two_by_four_by_three_product() -> None:
    cases = benchmarks.build_geometry_cases(("cpu",))

    assert len(cases) == 24
    assert len({case.run_id for case in cases}) == 24
    assert {case.parameter("resolution") for case in cases} == {"1080p", "4k"}
    assert {case.parameter("geometry_ratio") for case in cases} == {
        "0.5",
        "1",
        "2",
        "4",
    }
    assert {case.parameter("edge_percent") for case in cases} == {25, 50, 100}


def test_registry_has_every_required_gate_and_no_duplicate_run_id() -> None:
    registry = benchmarks.build_case_registry(
        device_contexts=("cpu",),
        fixture_settings=Path("fixture-settings.json"),
    )

    assert set(benchmarks.available_gate_names()) == {
        "geometry-scale",
        "region-scale",
        "fallback-scale",
        "content-scale",
        "control-scale",
        "encoding-scale",
        "fault-matrix",
        "runtime-matrix",
        "real-fixture",
    }
    assert len(registry) == len(set(registry))
    selected = ("geometry-scale", "fallback-scale")
    required = benchmarks.required_run_ids(selected, registry)
    assert required
    assert all(registry[run_id].gate in selected for run_id in required)


def test_geometry_memory_preflight_rejects_over_budget_without_allocation() -> None:
    estimate = benchmarks.estimate_geometry_case_bytes(
        width=3840,
        height=2160,
        geometry_ratio="4",
        edge_percent=100,
    )

    assert estimate.geometry_pixels == 4 * 3840 * 2160
    assert estimate.query_pixels == 3840 * 2160
    assert estimate.predicted_phase_rss_bytes > benchmarks.HOST_RSS_CAP_BYTES
    with pytest.raises(benchmarks.BenchmarkCapacityError, match="pre-mutation"):
        benchmarks.require_geometry_capacity(estimate)


def test_visit_p95_uses_exact_nearest_rank_without_float_percentile_workspace() -> None:
    visits = np.asarray([0, 1, 1, 2, 4, 4, 4, 9, 9, 20], dtype=np.uint32)

    assert benchmarks.visit_p95_nearest_rank(visits) == 20
    assert benchmarks.visit_p95_nearest_rank(np.zeros(100, dtype=np.uint32)) == 0
    with pytest.raises(ValueError):
        benchmarks.visit_p95_nearest_rank(np.asarray([], dtype=np.uint32))


def test_full_grid_oracle_resolves_half_pixel_ties_to_lower_coordinates() -> None:
    target_y = np.asarray([0.5, 1.5, 1.5000000000000002], dtype=np.float64)
    target_x = np.asarray([0.5, 0.0, 0.5], dtype=np.float64)

    sample_indices = benchmarks.full_grid_nearest_sample_indices(
        target_y,
        target_x,
        height=4,
        width=3,
    )

    assert sample_indices.tolist() == [0, 3, 6]


def test_4k_fallback_stripes_leave_a_full_safe_donor_support() -> None:
    period, hole_width = benchmarks.fallback_stripe_geometry(
        edge_percent=25,
        safe_radius=2,
    )

    assert hole_width * 4 == period
    assert period - hole_width >= 2 * 2 + 1


def test_tiny_geometry_worker_matches_closed_form_oracle_and_reports_visits() -> None:
    case = benchmarks.BenchmarkCase.create(
        run_id="geometry.cpu.tiny.r0_5.edge100",
        gate="geometry-scale",
        case="tiny-r0_5-edge100",
        device_context="cpu",
        operation="geometry",
        parameters={
            "width": 32,
            "height": 18,
            "resolution": "tiny",
            "geometry_ratio": "0.5",
            "edge_percent": 100,
        },
    )

    result = benchmarks.execute_worker_case(case)

    assert result.status == "passed"
    assert result.facts["oracle"] == "full-grid-closed-form-v1"
    assert result.equality_sha256 != "00" * 32
    if result.facts["capacity_rejected"]:
        assert result.visit_total == 0
        assert result.facts["pre_mutation"] is True
    else:
        assert result.visit_total > 0
        assert 0 < result.visit_p95 <= result.visit_max


def test_subprocess_runner_measures_rss_and_builds_a_closed_descriptor(tmp_path: Path) -> None:
    case = benchmarks.BenchmarkCase.create(
        run_id="geometry.cpu.tiny.r1.edge25",
        gate="geometry-scale",
        case="tiny-r1-edge25",
        device_context="cpu",
        operation="geometry",
        parameters={
            "width": 64,
            "height": 36,
            "resolution": "tiny",
            "geometry_ratio": "1",
            "edge_percent": 25,
        },
    )
    case_file = tmp_path / "case.json"
    case_file.write_bytes(reporting.canonical_json_bytes(case.to_object()))

    descriptor = benchmarks.run_case_subprocess(case, case_file=case_file)

    assert descriptor.run_id == case.run_id
    assert descriptor.status == "passed"
    assert 0 < descriptor.peak_rss_bytes <= benchmarks.HOST_RSS_CAP_BYTES
    assert descriptor.facts["process_baseline_rss_bytes"] > 0
    assert descriptor.facts["process_tree_peak_rss_bytes"] > 32 * 1024 * 1024
    assert descriptor.peak_rss_bytes == (
        descriptor.facts["process_tree_peak_rss_bytes"]
        - descriptor.facts["process_baseline_rss_bytes"]
    )
    assert descriptor.wall_time_ns > 0
    assert descriptor.visit_total > 0


def test_over_cap_worker_evidence_is_failed_instead_of_mislabeled_passed() -> None:
    case = benchmarks.BenchmarkCase.create(
        run_id="runtime.cpu.overcap",
        gate="runtime-matrix",
        case="overcap",
        device_context="cpu",
        operation="runtime",
        parameters={},
    )
    result = benchmarks.WorkerResult(
        status="passed",
        arena_bytes=0,
        visit_total=0,
        visit_max=0,
        visit_p95=0,
        equality_sha256="01" * 32,
        facts={"phase": "runtime"},
        baseline_rss_bytes=100,
        phase_wall_time_ns=1,
    )

    descriptor = benchmarks.finalize_worker_descriptor(
        case,
        result,
        process_wall_time_ns=2,
        process_tree_peak_rss_bytes=benchmarks.HOST_RSS_CAP_BYTES + 101,
    )

    assert descriptor.status == "failed"
    assert descriptor.facts["rss_cap_exceeded"] is True


def test_runtime_baseline_is_not_charged_to_the_quality_phase_cap() -> None:
    case = benchmarks.BenchmarkCase.create(
        run_id="runtime.cpu.baseline",
        gate="runtime-matrix",
        case="baseline",
        device_context="cpu",
        operation="runtime",
        parameters={},
    )
    baseline = benchmarks.HOST_RSS_CAP_BYTES + 32 * 1024 * 1024
    result = benchmarks.WorkerResult(
        status="passed",
        arena_bytes=0,
        visit_total=0,
        visit_max=0,
        visit_p95=0,
        equality_sha256="01" * 32,
        facts={"phase": "runtime"},
        baseline_rss_bytes=baseline,
        phase_wall_time_ns=123,
    )

    descriptor = benchmarks.finalize_worker_descriptor(
        case,
        result,
        process_wall_time_ns=999,
        process_tree_peak_rss_bytes=baseline + 8 * 1024 * 1024,
    )

    assert descriptor.status == "passed"
    assert descriptor.wall_time_ns == 123
    assert descriptor.facts["process_wall_time_ns"] == 999
    assert descriptor.peak_rss_bytes == 8 * 1024 * 1024
    assert descriptor.facts["rss_measurement"] == "process-tree-peak-minus-baseline-v1"
    assert descriptor.facts["rss_cap_exceeded"] is False


def test_geometry_phase_over_thirty_seconds_is_failed() -> None:
    case = benchmarks.BenchmarkCase.create(
        run_id="geometry.cpu.slow",
        gate="geometry-scale",
        case="slow",
        device_context="cpu",
        operation="geometry",
        parameters={},
    )
    result = benchmarks.WorkerResult(
        status="passed",
        arena_bytes=0,
        visit_total=0,
        visit_max=0,
        visit_p95=0,
        equality_sha256="01" * 32,
        facts={},
        baseline_rss_bytes=100,
        phase_wall_time_ns=30_000_000_001,
    )

    descriptor = benchmarks.finalize_worker_descriptor(
        case,
        result,
        process_wall_time_ns=31_000_000_000,
        process_tree_peak_rss_bytes=101,
    )

    assert descriptor.status == "failed"
    assert descriptor.facts["wall_cap_ns"] == 30_000_000_000
    assert descriptor.facts["wall_cap_exceeded"] is True


@pytest.mark.parametrize(
    ("operation", "parameters"),
    [
        (
            "region",
            {
                "width": 64,
                "height": 36,
                "resolution": "tiny",
                "edge_percent": 50,
            },
        ),
        ("fallback", {"width": 64, "height": 36, "edge_percent": 25}),
        (
            "content",
            {
                "width": 64,
                "height": 36,
                "resolution": "tiny",
                "geometry_ratio": "1",
            },
        ),
        ("control", {"frame_count": 100}),
        ("encoding", {"frame_count": 100, "mode": "direct"}),
        ("runtime", {}),
    ],
)
def test_each_nonfixture_worker_emits_closed_success_evidence(
    operation: str,
    parameters: dict,
) -> None:
    case = benchmarks.BenchmarkCase.create(
        run_id=f"{operation}.cpu.tiny",
        gate="runtime-matrix",
        case=f"{operation}-tiny",
        device_context="cpu",
        operation=operation,
        parameters=parameters,
    )

    result = benchmarks.execute_worker_case(case)

    assert result.status == "passed"
    assert result.equality_sha256 != "00" * 32
    assert result.facts


def test_fixture_worker_streams_the_reported_job_without_adopting_other_settings(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    final = tmp_path / "final.mp4"
    source.write_bytes(b"source")
    final.write_bytes(b"final")
    for directory in ("00_original_frames", "04_left_frames", "04_right_frames"):
        path = tmp_path / directory
        path.mkdir()
        (path / "frame_000000.png").write_bytes(b"png")
    settings = tmp_path / "job-settings.json"
    settings.write_text(
        """{
          "metadata": {"source_video": "SOURCE", "processing_status": "completed"},
          "video_properties": {"width": 1920, "height": 1080, "frame_count": 1},
          "processing_settings": {"stereo_strength": 1.5, "occlusion_fill": "background"},
          "runtime_info": {"final_output": "FINAL"}
        }""".replace(
            "SOURCE", str(source).replace("\\", "\\\\")
        ).replace(
            "FINAL", str(final).replace("\\", "\\\\")
        ),
        encoding="utf-8",
    )
    (tmp_path / "decoy-settings.json").write_text("{}", encoding="ascii")
    case = benchmarks.BenchmarkCase.create(
        run_id="fixture.cpu.tiny",
        gate="real-fixture",
        case="tiny",
        device_context="cpu",
        operation="fixture",
        parameters={"settings_path": str(settings)},
    )

    result = benchmarks.execute_worker_case(case)

    assert result.status == "passed"
    assert result.facts["settings_path"] == str(settings)
    assert result.facts["frame_counts"] == {
        "00_original_frames": 1,
        "04_left_frames": 1,
        "04_right_frames": 1,
    }


def test_gate_suite_is_fail_closed_and_binds_source_identity(tmp_path: Path) -> None:
    case = benchmarks.BenchmarkCase.create(
        run_id="fault.cpu.aggregate",
        gate="fault-matrix",
        case="aggregate",
        device_context="cpu",
        operation="faults",
        parameters={},
    )
    registry = {case.run_id: case}
    source = reporting.SourceIdentity("01" * 20, "02" * 32)
    environment = reporting.EnvironmentIdentity("3.11", "2", "2", "4", "test-platform")

    report = benchmarks.assemble_gate_report(
        selected_gates=("fault-matrix",),
        registry=registry,
        source=source,
        environment=environment,
        run=lambda _case: reporting.GateRunDescriptor.create(
            run_id=case.run_id,
            gate=case.gate,
            case=case.case,
            device_context=case.device_context,
            status="passed",
            wall_time_ns=1,
            peak_rss_bytes=1,
            rss_cap_bytes=benchmarks.HOST_RSS_CAP_BYTES,
            arena_bytes=0,
            visit_total=0,
            visit_max=0,
            visit_p95=0,
            equality_sha256=hashlib.sha256(b"faults").hexdigest(),
            facts={},
        ),
    )

    assert report.source == source
    assert report.required_run_ids == (case.run_id,)
    assert report.gate_decision == "prebuilt_extension"
