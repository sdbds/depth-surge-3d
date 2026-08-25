"""Tests for the hash-bound stereo Quality production verifier."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import scripts.verify_stereo_quality_fixture as verifier
from scripts.verify_stereo_quality_fixture import (
    FixtureFrame,
    ReviewRegion,
    VerificationConfig,
    run_verification,
)
from src.depth_surge_3d.rendering.stereo_renderer import StereoRenderer, StereoRenderSettings


@pytest.fixture(autouse=True)
def _candidate_native_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    source_sha256 = verifier._sha256(
        verifier.PROJECT_ROOT / "native" / "quality_kd_native.cpp"
    )
    monkeypatch.setattr(
        verifier,
        "native_binary_evidence",
        lambda: {
            "path": "fixture-native.pyd",
            "identity": {"fixture": True},
            "sha256": "1" * 64,
            "byte_count": 1,
            "build_info": {"source_sha256": source_sha256},
        },
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def _git_repository(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=path, check=True)
    tracked = path / "tracked.txt"
    tracked.write_text("base", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=path, check=True)
    return path


def _write_fixture(root: Path, *, compatible_fast: bool = True) -> VerificationConfig:
    height, width = 16, 32
    rows, columns = np.indices((height, width))
    source = np.stack(
        (
            (columns * 7 + rows * 3) % 256,
            (columns * 5 + 40) % 256,
            (rows * 11 + 80) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    canonical = np.full((8, 16), 6554, dtype=np.uint16)
    canonical[:, 5:11] = np.uint16(32768)
    canonical[:, 11:] = np.uint16(58982)
    frame_name = "frame_000001.png"
    source_path = root / "00_original_frames" / frame_name
    canonical_path = root / "03_disparity_maps" / frame_name
    left_path = root / "04_left_frames" / frame_name
    right_path = root / "04_right_frames" / frame_name
    _write_png(source_path, source)
    _write_png(canonical_path, canonical)

    renderer = StereoRenderer(device="cpu")
    settings = StereoRenderSettings(
        stereo_strength=1.5,
        convergence=0.5,
        occlusion_fill="background",
        stereo_render_mode="fast",
        occlusion_fill_max_px=8,
    )
    rendered = renderer.render(source, canonical.astype(np.float32) / np.float32(65535.0), settings)
    if compatible_fast:
        _write_png(left_path, rendered.left_image)
        _write_png(right_path, rendered.right_image)
    else:
        _write_png(left_path, np.zeros_like(source))
        _write_png(right_path, np.zeros_like(source))

    processing = {
        "stereo_strength": 1.5,
        "convergence": 0.5,
        "stereo_geometry_mode": "relative",
        "occlusion_fill": "background",
    }
    settings_path = root / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "metadata": {"settings_schema_version": 4},
                "video_properties": {"width": width, "height": height},
                "processing_settings": processing,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    source_metadata = root / "00_original_frames" / "metadata.json"
    canonical_metadata = root / "03_disparity_maps" / "metadata.json"
    source_metadata.write_text(json.dumps({"fingerprint": "a" * 64}), encoding="utf-8")
    canonical_metadata.write_text(json.dumps({"fingerprint": "b" * 64}), encoding="utf-8")

    frame = FixtureFrame(
        number=1,
        source_sha256=_sha256(source_path),
        canonical_sha256=_sha256(canonical_path),
        fast_left_sha256=_sha256(left_path),
        fast_right_sha256=_sha256(right_path),
        regions=(ReviewRegion("edge", 4, 2, 28, 14),),
    )
    return VerificationConfig(
        settings_relative_path="settings.json",
        settings_raw_sha256=_sha256(settings_path),
        settings_payload_sha256=_canonical_hash(processing),
        source_stage_fingerprint="a" * 64,
        source_frame_fingerprint=None,
        canonical_stage_fingerprint="b" * 64,
        ordered_manifest_sha256=None,
        frames=(frame,),
        render_width=width,
        render_height=height,
        stereo_strength=1.5,
        convergence=0.5,
        occlusion_fill_max_px=8,
        performance_gate_enabled=False,
    )


def test_half_pixel_nearest_diagnostic_has_frozen_mapping() -> None:
    source = np.arange(6, dtype=np.float32).reshape(2, 3)

    actual = verifier._nearest_half_pixel_upsample(source, (4, 6))

    assert np.array_equal(
        actual,
        np.asarray(
            [
                [0, 0, 1, 1, 2, 2],
                [0, 0, 1, 1, 2, 2],
                [3, 3, 4, 4, 5, 5],
                [3, 3, 4, 4, 5, 5],
            ],
            dtype=np.float32,
        ),
    )


def test_cpu_affinity_parser_expands_a_canonical_processor_set() -> None:
    assert verifier._parse_cpu_affinity("0-3,6,8-9") == (0, 1, 2, 3, 6, 8, 9)

    with pytest.raises(ValueError, match="duplicate"):
        verifier._parse_cpu_affinity("0-2,2")
    with pytest.raises(ValueError, match="ascending"):
        verifier._parse_cpu_affinity("3-1")
    with pytest.raises(ValueError, match="nonnegative"):
        verifier._parse_cpu_affinity("-1")


def test_windows_topology_parser_counts_only_processor_core_records() -> None:
    raw = b"".join(
        (
            (0).to_bytes(4, "little") + (8).to_bytes(4, "little"),
            (1).to_bytes(4, "little") + (12).to_bytes(4, "little") + b"xxxx",
            (0).to_bytes(4, "little") + (8).to_bytes(4, "little"),
        )
    )

    assert verifier._count_windows_processor_core_records(raw) == 2


def test_windows_physical_core_count_uses_native_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(verifier, "_optional_psutil", lambda: None)
    monkeypatch.setattr(verifier.os, "name", "nt")
    monkeypatch.setattr(verifier, "_windows_physical_core_count", lambda: 24)

    assert verifier._physical_core_count() == 24


def test_windows_cpu_model_comes_from_the_processor_name_registry_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Key:
        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    def open_key(root, path):
        assert root == "HKLM"
        assert path == r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        return Key()

    def query_value(_key, name):
        assert name == "ProcessorNameString"
        return "  Example   CPU 9000  ", 1

    fake = SimpleNamespace(
        HKEY_LOCAL_MACHINE="HKLM",
        OpenKey=open_key,
        QueryValueEx=query_value,
    )
    monkeypatch.setitem(sys.modules, "winreg", fake)

    assert verifier._windows_cpu_model() == "Example CPU 9000"


def test_windows_affinity_evidence_queries_every_enumerated_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(verifier.os, "name", "nt")
    monkeypatch.setattr(verifier, "_windows_processor_group_count", lambda: 1)
    monkeypatch.setattr(verifier, "_current_thread_cpu_affinity", lambda: (2, 3))
    monkeypatch.setattr(
        verifier,
        "_windows_live_thread_affinities",
        lambda: ((101, 0, (2, 3)), (202, 0, (2, 3))),
    )

    assert verifier._affinity_evidence((2, 3)) == {
        "requested": [2, 3],
        "effective": [2, 3],
        "windows_thread_group_affinities": None,
        "all_threads_verified": True,
        "verified_thread_count": 2,
    }


def test_windows_affinity_evidence_rejects_one_escaped_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(verifier.os, "name", "nt")
    monkeypatch.setattr(verifier, "_windows_processor_group_count", lambda: 1)
    monkeypatch.setattr(verifier, "_current_thread_cpu_affinity", lambda: (2, 3))
    monkeypatch.setattr(
        verifier,
        "_windows_live_thread_affinities",
        lambda: ((101, 0, (2, 3)), (202, 0, (3,))),
    )

    with pytest.raises(RuntimeError, match="thread 202 escaped"):
        verifier._affinity_evidence((2, 3))


def test_multi_group_windows_runtime_fails_closed_without_cross_group_cpu_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(verifier.os, "name", "nt")
    monkeypatch.setattr(verifier, "_windows_processor_group_count", lambda: 2)
    monkeypatch.setattr(
        verifier,
        "_windows_live_thread_affinities",
        lambda: ((101, 0, (0, 1)), (202, 1, (0, 1))),
    )
    monkeypatch.setattr(verifier, "_physical_core_count", lambda: 64)
    monkeypatch.setattr(verifier, "_physical_memory_bytes", lambda: 128 * 1024**3)
    monkeypatch.setattr(verifier, "_command_first_line", lambda _argv: None)

    with pytest.raises(RuntimeError, match="cross-group"):
        verifier._runtime_metadata("cpu", None)


def test_multi_group_windows_rejects_only_explicit_flat_affinity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(verifier.os, "name", "nt")
    monkeypatch.setattr(verifier, "_windows_processor_group_count", lambda: 2)

    with pytest.raises(RuntimeError, match="cross-group"):
        verifier._affinity_evidence((0, 1))


def test_quality_native_evidence_rejects_a_stale_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        verifier,
        "native_binary_evidence",
        lambda: {
            "path": "stale-native.pyd",
            "sha256": "1" * 64,
            "build_info": {"source_sha256": "2" * 64},
        },
    )
    monkeypatch.setattr(verifier, "_sha256", lambda _path: "3" * 64)

    with pytest.raises(RuntimeError, match="candidate source"):
        verifier._quality_native_runtime_evidence()


def test_git_state_hashes_dirty_tracked_and_untracked_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _git_repository(tmp_path / "repo")
    monkeypatch.setattr(verifier, "PROJECT_ROOT", repository)

    assert verifier._git_state()["dirty"] is False
    tracked = repository / "tracked.txt"
    tracked.write_text("first", encoding="utf-8")
    first_tracked = verifier._git_state()["dirty_fingerprint"]
    tracked.write_text("second", encoding="utf-8")
    second_tracked = verifier._git_state()["dirty_fingerprint"]
    subprocess.run(["git", "checkout", "--", "tracked.txt"], cwd=repository, check=True)
    untracked = repository / "untracked.txt"
    untracked.write_text("first", encoding="utf-8")
    first_untracked = verifier._git_state()["dirty_fingerprint"]
    untracked.write_text("second", encoding="utf-8")
    second_untracked = verifier._git_state()["dirty_fingerprint"]

    assert first_tracked != second_tracked
    assert first_untracked != second_untracked


def test_git_state_rejects_an_over_budget_untracked_name_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _git_repository(tmp_path / "repo")
    (repository / "untracked.txt").write_text("content", encoding="utf-8")
    monkeypatch.setattr(verifier, "PROJECT_ROOT", repository)
    monkeypatch.setattr(verifier, "SOURCE_IDENTITY_NAME_BYTES", 1)

    with pytest.raises(RuntimeError, match="name list"):
        verifier._git_state()


def test_untracked_name_budget_stops_at_cap_plus_one_and_reaps_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Stream(io.BytesIO):
        maximum_request = 0

        def read(self, size: int = -1) -> bytes:
            self.maximum_request = max(self.maximum_request, size)
            return super().read(size)

    class Process:
        def __init__(self) -> None:
            self.stdout = Stream(b"x" * 9)
            self.terminated = False
            self.waited = False

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            raise AssertionError("responsive fake child should not need kill")

        def wait(self, timeout=None) -> int:
            self.waited = True
            return 0

    process = Process()
    monkeypatch.setattr(verifier, "SOURCE_IDENTITY_NAME_BYTES", 8)
    monkeypatch.setattr(verifier.subprocess, "Popen", lambda *_args, **_kwargs: process)

    with pytest.raises(RuntimeError, match="name list"):
        verifier._hash_untracked_git_files(hashlib.sha256())

    assert process.stdout.tell() == 9
    assert process.stdout.maximum_request == 9
    assert process.terminated is True
    assert process.waited is True


def test_git_state_frames_each_untracked_file_content_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _git_repository(tmp_path / "repo")
    monkeypatch.setattr(verifier, "PROJECT_ROOT", repository)
    first = repository / "a"
    second = repository / "b"

    first.write_bytes((1).to_bytes(8, "little") + b"b" + b"X")
    one_file_state = verifier._git_state()["dirty_fingerprint"]

    first.write_bytes(b"")
    second.write_bytes(b"X")
    two_file_state = verifier._git_state()["dirty_fingerprint"]

    assert one_file_state != two_file_state


def test_release_cleanliness_ignores_only_generated_evidence_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _git_repository(tmp_path / "repo")
    monkeypatch.setattr(verifier, "PROJECT_ROOT", repository)
    evidence = repository / verifier._TASK0_REPORT_RELATIVE_PATH
    evidence.parent.mkdir(parents=True)
    evidence.write_text("generated", encoding="ascii")

    evidence_only = verifier._git_state()

    assert evidence_only["dirty"] is True
    assert evidence_only["release_tree_clean"] is True

    (repository / "ordinary-untracked.py").write_text("changed", encoding="ascii")

    assert verifier._git_state()["release_tree_clean"] is False


def test_task0_source_accepts_only_evidence_only_descendant_commits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _git_repository(tmp_path / "repo")
    monkeypatch.setattr(verifier, "PROJECT_ROOT", repository)
    source_before_evidence = verifier.task0_reporting.capture_source_identity(
        repository,
        excluded_relative_paths=verifier._RELEASE_EVIDENCE_RELATIVE_PATHS,
    )
    evidence = repository / verifier._TASK0_REPORT_RELATIVE_PATH
    evidence.parent.mkdir(parents=True)
    evidence.write_text("generated", encoding="ascii")
    subprocess.run(["git", "add", evidence.as_posix()], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "evidence"], cwd=repository, check=True
    )
    evidence_descendant = verifier.task0_reporting.capture_source_identity(
        repository,
        excluded_relative_paths=verifier._RELEASE_EVIDENCE_RELATIVE_PATHS,
    )

    assert verifier._task0_source_matches_candidate_v1(
        source_before_evidence,
        evidence_descendant,
    )

    (repository / "tracked.txt").write_text("source change", encoding="ascii")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "source"], cwd=repository, check=True
    )
    changed_source = verifier.task0_reporting.capture_source_identity(
        repository,
        excluded_relative_paths=verifier._RELEASE_EVIDENCE_RELATIVE_PATHS,
    )

    assert not verifier._task0_source_matches_candidate_v1(
        source_before_evidence,
        changed_source,
    )


def test_release_gate_cannot_pass_a_dirty_source_tree() -> None:
    checks = [
        {
            "fast_left_pixels_equal": True,
            "fast_right_pixels_equal": True,
            "fast_left_png_equal": True,
            "fast_right_png_equal": True,
            "quality_final_unresolved_pixels_zero": True,
            "quality_without_fill_exposes_geometry_holes": True,
            "nearest_diagnostic_is_distinct": True,
            "all_output_shapes_and_dtypes_valid": True,
        }
    ]
    benchmark = {
        "status": "disabled",
        "passed": False,
    }
    task0 = {
        "candidate_source_match": True,
        "native_build_match": True,
        "resource_bounds": True,
        "gates": {name: True for name in verifier._TASK0_RELEASE_RUN_GROUPS},
    }

    gates = verifier._aggregate_gates(
        checks,
        benchmark,
        temporal_radius=0,
        release_required=True,
        source={"dirty": True},
        task0=task0,
    )

    assert gates["source_tree_clean"] is False


@pytest.mark.parametrize("failed_gate", verifier._BENCHMARK_RELEASE_GATE_NAMES)
def test_release_gate_requires_each_independent_benchmark_gate(
    failed_gate: str,
) -> None:
    checks = [
        {
            "fast_left_pixels_equal": True,
            "fast_right_pixels_equal": True,
            "fast_left_png_equal": True,
            "fast_right_png_equal": True,
            "quality_final_unresolved_pixels_zero": True,
            "quality_without_fill_exposes_geometry_holes": True,
            "nearest_diagnostic_is_distinct": True,
            "all_output_shapes_and_dtypes_valid": True,
        }
    ]
    benchmark_gates = {
        name: name != failed_gate for name in verifier._BENCHMARK_RELEASE_GATE_NAMES
    }
    task0 = {
        "candidate_source_match": True,
        "native_build_match": True,
        "resource_bounds": True,
        "gates": {name: True for name in verifier._TASK0_RELEASE_RUN_GROUPS},
    }

    gates = verifier._aggregate_gates(
        checks,
        {"status": "failed", "gates": benchmark_gates},
        temporal_radius=0,
        release_required=True,
        source={"dirty": False, "release_tree_clean": True},
        task0=task0,
    )

    assert gates[failed_gate] is False
    assert all(
        value for name, value in gates.items() if name != failed_gate
    )


def test_full_pipeline_benchmark_publishes_production_diagnostics(
    tmp_path: Path,
) -> None:
    config = _write_fixture(tmp_path / "fixture")
    loaded = verifier._load_frame(tmp_path / "fixture", config.frames[0], config)
    renderer = StereoRenderer(device="cpu")
    settings = verifier._settings(config, mode="quality", fill="background")
    benchmark_root = tmp_path / "benchmark"

    result = verifier._benchmark_quality_pipeline_mode(
        lambda: verifier.render_relative_quality_compact(
            renderer=renderer,
            frame=loaded.source,
            canonical=loaded.canonical,
            settings=settings,
        ),
        device="cpu",
        warmups=0,
        measured=2,
        expected_shape=loaded.source.shape,
        benchmark_root=benchmark_root,
    )

    assert result["deterministic_output"] is True
    assert result["output_shape_valid"] is True
    assert len(set(result["published_diagnostic_byte_counts"])) == 1
    assert result["published_diagnostic_byte_counts"][0] > 0
    assert (benchmark_root / "04_left_frames" / "benchmark.png").is_file()
    assert (
        benchmark_root
        / "04_stereo_diagnostics"
        / "frames"
        / "benchmark"
        / "manifest.json"
    ).is_file()


def test_missing_hash_bound_fixture_writes_nonzero_report(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    missing = config.frames[0]
    (tmp_path / "00_original_frames" / missing.file_name).unlink()
    report_path = tmp_path / "report.json"

    exit_code = run_verification(
        tmp_path,
        report_path,
        tmp_path / "artifacts",
        device="cpu",
        config=config,
        benchmark_warmups=0,
        benchmark_measured=0,
        temporal_radius=0,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report["status"] == "fixture_unavailable"
    assert report["inputs"][missing.stem]["source"]["status"] == "missing"


def test_tiny_fixture_writes_four_modes_diagnostics_and_review_crops(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    report_path = tmp_path / "report.json"
    artifacts = tmp_path / "artifacts"

    exit_code = run_verification(
        tmp_path,
        report_path,
        artifacts,
        device="cpu",
        config=config,
        benchmark_warmups=0,
        benchmark_measured=0,
        temporal_radius=0,
        benchmark_command=("python", "verify.py", "--device", "cpu"),
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    frame = report["frames"][0]
    assert exit_code == 0
    assert report["status"] == "machine_pass_human_review_required"
    assert report["gates"]["fast_byte_compatibility"] is True
    assert report["gates"]["quality_final_unresolved_lanes_zero"] is True
    assert report["benchmark_commands"] == [
        {
            "argv": ["python", "verify.py", "--device", "cpu"],
            "cwd": str(Path.cwd().resolve()),
        }
    ]
    assert set(frame["modes"]) == {"fast", "quality", "quality_without_fill", "nearest"}
    assert Path(frame["diagnostics"]["coverage_path"]).is_file()
    assert Path(frame["diagnostics"]["repair_path"]).is_file()
    crop = frame["regions"][0]["artifacts"]["quality_left"]
    crop_image = cv2.imread(crop, cv2.IMREAD_COLOR)
    assert crop_image.shape[:2] == (12 * 4, 24 * 4)


def test_fast_pixel_mismatch_is_a_numeric_failure(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path, compatible_fast=False)
    report_path = tmp_path / "report.json"

    exit_code = run_verification(
        tmp_path,
        report_path,
        tmp_path / "artifacts",
        device="cpu",
        config=config,
        benchmark_warmups=0,
        benchmark_measured=0,
        temporal_radius=0,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert report["status"] == "machine_gate_failure"
    assert report["gates"]["fast_byte_compatibility"] is False
