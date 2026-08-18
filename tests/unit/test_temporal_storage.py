"""Shot-atomic VDPP stabilized-depth storage tests."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest
import numpy as np

from src.depth_surge_3d.core.depth_contract import canonical_json_hash
from src.depth_surge_3d.core.render_disparity import validate_render_disparity_input
from src.depth_surge_3d.processing.frames.temporal_storage import (
    StabilizedDepthStore,
    build_final_shot_plan,
)


def _frame_files(count: int) -> list[Path]:
    return [Path(f"frame_{index + 1:06d}.png") for index in range(count)]


def _semantic(frame_files: list[Path], shot_plan: list[dict[str, int]]) -> dict:
    return {
        "frame_names": [path.name for path in frame_files],
        "native_shape": [3, 5],
        "source_canonical_fingerprint": "base-fingerprint",
        "scene_manifest_fingerprint": "scene-fingerprint",
        "postprocessor_settings": {"temporal_postprocessor": "vdpp"},
        "model_identity": {"name": "vdpp", "vendor_port_version": 1},
        "execution_plan": {"window_size": 32, "overlap": 4, "stride": 28},
        "shot_plan": shot_plan,
    }


def _store(
    root: Path,
    *,
    count: int = 4,
    cuts: list[int] | None = None,
    runtime: str = "runtime-a",
    source: str = "base-fingerprint",
) -> StabilizedDepthStore:
    frame_files = _frame_files(count)
    plan = build_final_shot_plan(count, cuts or [])
    semantic = _semantic(frame_files, plan)
    semantic["source_canonical_fingerprint"] = source
    return StabilizedDepthStore(
        root,
        frame_files=frame_files,
        semantic_identity=semantic,
        runtime_identity={"runtime": runtime},
        execution_provenance={"runtime": runtime, "preflight_max_memory_allocated": 0},
    )


def _outputs(length: int, offset: int = 0):
    for index in range(length):
        yield index, np.full((3, 5), offset + index * 1000, dtype=np.uint16)


def _copy_calibration(length: int) -> dict[str, object]:
    return {
        "mode": "base_fallback",
        "pair_count": 0,
        "midpoint_count": 0,
        "midpoint_fraction": 0.0,
        "flat_frame_count": length,
        "source_mean": None,
        "source_variance": None,
        "source_std": None,
        "raw_mean": None,
        "raw_variance": None,
        "raw_std": None,
        "covariance": None,
        "correlation": None,
        "scale": None,
        "shift": None,
        "candidate_mean": None,
        "candidate_std": None,
        "postclip_contrast_ratio": None,
        "postclip_mean_drift": None,
        "preclip_low_fraction": None,
        "preclip_high_fraction": None,
        "fallback_reason": "source_no_range",
    }


def _commit(store: StabilizedDepthStore, shot_id: int, outputs, length: int) -> None:
    store.commit_shot(
        shot_id,
        outputs,
        calibration=_copy_calibration(length),
    )


def test_final_shot_plan_uses_strict_half_open_ranges() -> None:
    assert build_final_shot_plan(7, [2, 5]) == [
        {"shot_id": 0, "start": 0, "end": 2},
        {"shot_id": 1, "start": 2, "end": 5},
        {"shot_id": 2, "start": 5, "end": 7},
    ]

    with pytest.raises(ValueError, match="final cuts"):
        build_final_shot_plan(7, [5, 2])
    with pytest.raises(ValueError, match="final cuts"):
        build_final_shot_plan(7, [0, 5])


def test_committed_shots_finalize_to_a_valid_content_addressed_artifact(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "03_disparity_stabilized", count=4, cuts=[2])
    audit = store.audit()
    store.prepare(audit)

    _commit(store, 0, _outputs(2), 2)
    first_metadata = json.loads(store.metadata_path.read_text(encoding="utf-8"))
    first_state = first_metadata["state_fingerprint"]
    _commit(store, 1, _outputs(2, offset=2000), 2)
    files = store.finalize()

    metadata = json.loads(store.metadata_path.read_text(encoding="utf-8"))
    artifact = validate_render_disparity_input(files, _frame_files(4))
    assert metadata["status"] == "complete"
    assert metadata["state_fingerprint"] != first_state
    assert metadata["payload_fingerprint"]
    assert artifact.fingerprint == metadata["artifact_fingerprint"]


def test_interrupted_shot_is_not_committed_and_partial_payloads_are_removed(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "stable", count=3)
    store.prepare(store.audit())

    def interrupted():
        yield 0, np.zeros((3, 5), dtype=np.uint16)
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        _commit(store, 0, interrupted(), 3)

    metadata = json.loads(store.metadata_path.read_text(encoding="utf-8"))
    assert metadata["completed_shots"] == []
    assert list(store.root.glob("*.png")) == []
    assert list(store.manifest_dir.glob("*.json")) == []


def test_corrupt_completed_shot_is_repaired_without_discarding_later_shots(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stable"
    store = _store(root, count=4, cuts=[2])
    store.prepare(store.audit())
    _commit(store, 0, _outputs(2), 2)
    _commit(store, 1, _outputs(2, offset=2000), 2)
    store.finalize()
    later_bytes = (root / "frame_000003.png").read_bytes()
    (root / "frame_000001.png").write_bytes(b"changed")

    resumed = _store(root, count=4, cuts=[2])
    audit = resumed.audit()

    assert audit.complete is False
    assert audit.reusable_shot_ids == (1,)
    assert audit.invalid_shot_ids == (0,)
    resumed.prepare(audit)
    assert not (root / "frame_000001.png").exists()
    assert (root / "frame_000003.png").read_bytes() == later_bytes


def test_partial_runtime_change_requires_whole_unfinished_stage_reset(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stable"
    original = _store(root, count=4, cuts=[2], runtime="runtime-a")
    original.prepare(original.audit())
    _commit(original, 0, _outputs(2), 2)

    resumed = _store(root, count=4, cuts=[2], runtime="runtime-b")
    audit = resumed.audit()

    assert audit.reset_required is True
    assert "runtime" in audit.reason
    resumed.prepare(audit)
    assert list(root.glob("*.png")) == []
    assert json.loads(resumed.metadata_path.read_text())["completed_shots"] == []


def test_partial_v1_stage_requires_reset_without_touching_base_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stable"
    original = _store(root, count=2)
    original.prepare(original.audit())
    metadata = json.loads(original.metadata_path.read_text(encoding="utf-8"))
    metadata["algorithm_version"] = "vdpp-canonical-shot-v1"
    metadata["metadata_fingerprint"] = canonical_json_hash(
        {key: value for key, value in metadata.items() if key != "metadata_fingerprint"}
    )
    original.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    audit = _store(root, count=2).audit()

    assert audit.reset_required is True
    assert "semantic identity" in audit.reason


def test_complete_artifact_ignores_runtime_change_but_not_semantic_change(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stable"
    original = _store(root, count=2, runtime="runtime-a")
    original.prepare(original.audit())
    _commit(original, 0, _outputs(2), 2)
    original.finalize()

    runtime_changed = _store(root, count=2, runtime="runtime-b")
    assert runtime_changed.audit().complete is True

    semantic_changed = _store(root, count=2, runtime="runtime-b", source="other-base")
    changed_audit = semantic_changed.audit()
    assert changed_audit.complete is False
    assert changed_audit.reset_required is True
    assert "semantic" in changed_audit.reason


def test_output_validation_rejects_wrong_index_shape_and_non_uint16_values(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "stable", count=1)
    store.prepare(store.audit())

    with pytest.raises(ValueError, match="ordered"):
        _commit(store, 0, [(1, np.zeros((3, 5), dtype=np.uint16))], 1)
    with pytest.raises(ValueError, match="shape"):
        _commit(store, 0, [(0, np.zeros((2, 5), dtype=np.uint16))], 1)
    with pytest.raises(TypeError, match="uint16"):
        _commit(store, 0, [(0, np.zeros((3, 5), dtype=np.float32))], 1)


def test_commit_rejects_negative_shot_id(tmp_path: Path) -> None:
    store = _store(tmp_path / "stable", count=2, cuts=[1])
    store.prepare(store.audit())

    with pytest.raises(ValueError, match="Unknown stabilized shot"):
        _commit(store, -1, _outputs(1), 1)


def test_failed_metadata_publish_does_not_poison_in_memory_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path / "stable", count=1)
    store.prepare(store.audit())
    original_write = store._atomic_write_json
    calls = 0

    def fail_metadata_once(path: Path, payload: dict) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("metadata publish failed")
        original_write(path, payload)

    monkeypatch.setattr(store, "_atomic_write_json", fail_metadata_once)
    with pytest.raises(OSError, match="metadata publish failed"):
        _commit(store, 0, _outputs(1), 1)

    monkeypatch.setattr(store, "_atomic_write_json", original_write)
    _commit(store, 0, _outputs(1), 1)
    assert len(store.finalize()) == 1


def test_commit_persists_exact_uint16_payload_and_canonical_diagnostics(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "stable", count=1)
    store.prepare(store.audit())
    values = np.arange(15, dtype=np.uint16).reshape(3, 5)

    _commit(store, 0, [(0, values)], 1)

    written = cv2.imread(str(store.depth_files[0]), cv2.IMREAD_UNCHANGED)
    manifest = json.loads((store.manifest_dir / "shot_000000.json").read_text())
    assert np.array_equal(written, values)
    assert manifest["schema_version"] == 2
    assert manifest["calibration"] == _copy_calibration(1)


def test_commit_canonicalizes_diagnostics_before_writing_payload(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "stable", count=1)
    store.prepare(store.audit())
    invalid = _copy_calibration(1)
    invalid["midpoint_fraction"] = -0.0

    store.commit_shot(0, _outputs(1), calibration=invalid)

    manifest = json.loads((store.manifest_dir / "shot_000000.json").read_text())
    assert manifest["calibration"]["midpoint_fraction"].hex() == "0x0.0p+0"
