"""Task 0 canonical gate-report and fail-closed runner contracts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from scripts.quality_task0 import reporting
from scripts import run_quality_task0


def _source() -> reporting.SourceIdentity:
    return reporting.SourceIdentity(
        commit="01" * 20,
        dirty_diff_sha256="02" * 32,
    )


def _environment() -> reporting.EnvironmentIdentity:
    return reporting.EnvironmentIdentity(
        python_version="3.11.11",
        numpy_version="2.2.6",
        torch_version="2.7.1+cu128",
        opencv_version="4.12.0",
        platform="Windows-11",
    )


def _run(run_id: str = "geometry.cpu.1080p.edge25") -> reporting.GateRunDescriptor:
    return reporting.GateRunDescriptor.create(
        run_id=run_id,
        gate="geometry-scale",
        case="1080p-edge25",
        device_context="cpu",
        status="passed",
        wall_time_ns=123,
        peak_rss_bytes=456,
        rss_cap_bytes=1024,
        arena_bytes=64,
        visit_total=10,
        visit_max=4,
        visit_p95=3,
        equality_sha256="03" * 32,
        facts={"pixel_count": 1920 * 1080},
    )


def test_canonical_json_and_checked_u64_are_closed() -> None:
    value = {"z": 1, "text": "\u53ef", "a": [True, None]}

    assert reporting.canonical_json_bytes(value) == (b'{"a":[true,null],"text":"\\u53ef","z":1}')
    assert reporting.checked_u64(0, "counter") == 0
    assert reporting.checked_u64((1 << 64) - 1, "counter") == (1 << 64) - 1
    for invalid in (-1, 1 << 64, True, 1.0):
        with pytest.raises(reporting.ReportSchemaError):
            reporting.checked_u64(invalid, "counter")


def test_run_descriptor_fingerprint_covers_all_exact_fields() -> None:
    run = _run()
    value = run.to_object()
    without_fingerprint = dict(value)
    without_fingerprint.pop("fingerprint")

    assert (
        value["fingerprint"]
        == hashlib.sha256(reporting.canonical_json_bytes(without_fingerprint)).hexdigest()
    )
    assert reporting.GateRunDescriptor.from_object(value) == run

    value["peak_rss_bytes"] += 1
    with pytest.raises(reporting.ReportSchemaError, match="fingerprint"):
        reporting.GateRunDescriptor.from_object(value)


def test_run_descriptor_rejects_cap_failure_labeled_as_success() -> None:
    with pytest.raises(reporting.ReportSchemaError, match="RSS cap"):
        reporting.GateRunDescriptor.create(
            run_id="rss.cpu",
            gate="rss",
            case="over-cap",
            device_context="cpu",
            status="passed",
            wall_time_ns=1,
            peak_rss_bytes=1025,
            rss_cap_bytes=1024,
            arena_bytes=0,
            visit_total=0,
            visit_max=0,
            visit_p95=0,
            equality_sha256="00" * 32,
            facts={},
        )


def test_report_fails_closed_when_required_runs_are_missing_or_duplicated() -> None:
    run = _run()

    with pytest.raises(reporting.IncompleteTask0ReportError, match="required"):
        reporting.Task0Report.create(
            source=_source(),
            environment=_environment(),
            required_run_ids=(run.run_id, "missing.cpu"),
            runs=(run,),
            gate_decision="existing_dependencies",
        )
    with pytest.raises(reporting.IncompleteTask0ReportError, match="duplicate"):
        reporting.Task0Report.create(
            source=_source(),
            environment=_environment(),
            required_run_ids=(run.run_id,),
            runs=(run, run),
            gate_decision="existing_dependencies",
        )


def test_atomic_report_round_trip_rejects_noncanonical_and_wrong_source(
    tmp_path: Path,
) -> None:
    run = _run()
    report = reporting.Task0Report.create(
        source=_source(),
        environment=_environment(),
        required_run_ids=(run.run_id,),
        runs=(run,),
        gate_decision="existing_dependencies",
    )
    path = tmp_path / "report.json"

    reporting.publish_report(path, report)

    assert reporting.load_report(path, expected_source=_source()) == report
    assert not (tmp_path / ".report.json.tmp").exists()
    pretty = json.dumps(report.to_object(), indent=2).encode("ascii")
    path.write_bytes(pretty)
    with pytest.raises(reporting.ReportSchemaError, match="canonical"):
        reporting.load_report(path)
    reporting.publish_report(path, report)
    with pytest.raises(reporting.StaleTask0ReportError):
        reporting.load_report(
            path,
            expected_source=reporting.SourceIdentity(
                commit=_source().commit,
                dirty_diff_sha256="ff" * 32,
            ),
        )


def test_gate_selection_rejects_unknown_and_duplicate_names() -> None:
    available = ("geometry", "content", "encoding")

    assert reporting.select_gate_names([], available) == available
    assert reporting.select_gate_names(["encoding", "geometry"], available) == (
        "encoding",
        "geometry",
    )
    with pytest.raises(reporting.GateSelectionError, match="duplicate"):
        reporting.select_gate_names(["geometry", "geometry"], available)
    with pytest.raises(reporting.GateSelectionError, match="unknown"):
        reporting.select_gate_names(["missing"], available)


def test_cli_selection_supports_individual_and_complete_gate_sets() -> None:
    available = ("geometry", "content", "encoding")

    assert run_quality_task0.parse_gate_selection([], available) == available
    assert run_quality_task0.parse_gate_selection(
        ["--gate", "encoding", "--gate", "content"], available
    ) == ("encoding", "content")
    with pytest.raises(reporting.GateSelectionError):
        run_quality_task0.parse_gate_selection(
            ["--gate", "geometry", "--gate", "geometry"], available
        )
    with pytest.raises(reporting.GateSelectionError):
        run_quality_task0.parse_gate_selection(["--gate", "unknown"], available)


def test_process_rss_and_device_context_labels_are_bounded_scalars() -> None:
    rss = reporting.read_process_rss_bytes(os.getpid())

    assert 0 < rss < (1 << 64)
    assert reporting.device_context_label("cpu") == "cpu"
    assert reporting.device_context_label("cuda", index=0, name="RTX 5090") == ("cuda:0:RTX_5090")
    with pytest.raises(reporting.ReportSchemaError):
        reporting.device_context_label("cuda", index=-1, name="bad")


def test_environment_identity_records_runtime_versions() -> None:
    environment = reporting.capture_environment_identity()

    assert environment.python_version
    assert environment.numpy_version
    assert environment.torch_version
    assert environment.opencv_version
    assert environment.platform


def test_source_identity_hashes_tracked_and_untracked_bytes(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "task0@example.invalid"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Task Zero"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("one\n", encoding="ascii")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)

    clean = reporting.capture_source_identity(tmp_path)
    tracked.write_text("two\n", encoding="ascii")
    modified = reporting.capture_source_identity(tmp_path)
    (tmp_path / "untracked.bin").write_bytes(b"three")
    untracked = reporting.capture_source_identity(tmp_path)

    assert clean.commit == modified.commit == untracked.commit
    assert (
        len({clean.dirty_diff_sha256, modified.dirty_diff_sha256, untracked.dirty_diff_sha256}) == 3
    )
