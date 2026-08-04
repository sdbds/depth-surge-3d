"""Native raw-depth storage and resume contract tests."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.depth_surge_3d.inference.depth.types import DepthRepresentation
from src.depth_surge_3d.processing.frames.depth_storage import (
    RawDepthFingerprintError,
    RawDepthOverflowError,
    RawDepthStore,
    build_model_fingerprint,
    estimate_depth_disk_bytes,
    require_disk_space,
)


class _Estimator:
    def __init__(self, model_path: Path) -> None:
        self.model_path = str(model_path)
        self.metric = False
        self.device = "cpu"

    def get_model_info(self) -> dict:
        return {"model_name": "fixture", "revision": "abc123", "precision": "fp32"}


def _open(
    directory: Path,
    first_values: np.ndarray,
    *,
    requested_dtype: str = "auto",
    semantic: dict | None = None,
) -> RawDepthStore:
    return RawDepthStore.open_or_create(
        directory,
        frame_names=["frame_000000.png", "frame_000001.png"],
        representation=DepthRepresentation.RELATIVE_DEPTH,
        semantic_fingerprint=semantic or {"model": "fixture", "revision": "1"},
        requested_dtype=requested_dtype,
        first_values=first_values,
    )


def test_model_fingerprint_hashes_local_weights_and_is_deterministic(tmp_path: Path) -> None:
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"stable weights")
    estimator = _Estimator(weights)
    settings = {"depth_resolution": "720", "fp32": True}

    first = build_model_fingerprint(estimator, settings)
    second = build_model_fingerprint(estimator, dict(reversed(list(settings.items()))))

    assert first == second
    assert first["weight_sha256"]
    assert first["model_info"]["revision"] == "abc123"


def test_model_fingerprint_uses_loaded_remote_artifact_identity() -> None:
    class RemoteEstimator:
        def __init__(self, artifact_identity: str) -> None:
            self.artifact_identity = artifact_identity

        def get_model_info(self) -> dict:
            return {
                "model_name": "owner/model",
                "artifact_identity": self.artifact_identity,
            }

    first = build_model_fingerprint(RemoteEstimator("hf:owner/model@aaa"), {})
    second = build_model_fingerprint(RemoteEstimator("hf:owner/model@bbb"), {})

    assert first["artifact_identity"] == "hf:owner/model@aaa"
    assert second["artifact_identity"] == "hf:owner/model@bbb"
    assert first != second


def test_auto_store_selects_float16_and_writes_atomic_compressed_files(tmp_path: Path) -> None:
    first = np.array([[[0.0, 0.5], [1.0, np.nan]]], dtype=np.float32)
    store = _open(tmp_path / "raw", first)

    paths = store.write_batch(["frame_000000.png"], first)

    assert store.metadata["selected_dtype"] == "float16"
    assert store.metadata["storage_status"] == "ready"
    assert paths == [tmp_path / "raw" / "frame_000000.npz"]
    with np.load(paths[0], allow_pickle=False) as payload:
        assert payload["values"].dtype == np.float16
    assert store.load(paths[0]).dtype == np.float32
    assert not list((tmp_path / "raw").glob("*.tmp"))


def test_explicit_float16_rejects_unrepresentable_first_chunk_before_metadata(
    tmp_path: Path,
) -> None:
    values = np.array([[[70000.0]]], dtype=np.float32)

    with pytest.raises(RawDepthOverflowError, match="float16"):
        _open(tmp_path / "raw", values, requested_dtype="float16")

    assert not (tmp_path / "raw" / "metadata.json").exists()


def test_auto_store_promotes_completed_files_without_reinference(tmp_path: Path) -> None:
    first = np.array([[[0.1, 0.2]]], dtype=np.float32)
    second = np.array([[[70000.0, 2.0]]], dtype=np.float32)
    store = _open(tmp_path / "raw", first)
    first_path = store.write_batch(["frame_000000.png"], first)[0]

    store.write_batch(["frame_000001.png"], second)

    assert store.metadata["selected_dtype"] == "float32"
    assert store.metadata["storage_provenance"] == "promoted_float16_to_float32"
    with np.load(first_path, allow_pickle=False) as payload:
        assert payload["values"].dtype == np.float32
        np.testing.assert_array_equal(
            payload["values"], first.astype(np.float16).astype(np.float32)[0]
        )


def test_explicit_float16_can_resume_as_float32_without_deleting_completed_files(
    tmp_path: Path,
) -> None:
    first = np.array([[[0.1, 0.2]]], dtype=np.float32)
    store = _open(tmp_path / "raw", first, requested_dtype="float16")
    first_path = store.write_batch(["frame_000000.png"], first)[0]
    with pytest.raises(RawDepthOverflowError):
        store.write_batch(["frame_000001.png"], np.array([[[70000.0, 2.0]]], np.float32))

    resumed = _open(tmp_path / "raw", first, requested_dtype="float32")

    assert first_path.exists()
    assert resumed.metadata["selected_dtype"] == "float32"
    assert resumed.complete_files == [first_path]


def test_promotion_resumes_after_atomic_rewrite_crash(tmp_path: Path, monkeypatch) -> None:
    values = np.array([[[0.1]], [[0.2]]], dtype=np.float32)
    store = _open(tmp_path / "raw", values, requested_dtype="float16")
    store.write_batch(["frame_000000.png", "frame_000001.png"], values)
    original = store._rewrite_file_as_float32
    calls = 0

    def crash_after_first(path: Path) -> None:
        nonlocal calls
        original(path)
        calls += 1
        if calls == 1:
            raise OSError("simulated crash")

    monkeypatch.setattr(store, "_rewrite_file_as_float32", crash_after_first)
    with pytest.raises(OSError, match="simulated crash"):
        store.promote_to_float32(requested_dtype="float32")
    metadata = json.loads((tmp_path / "raw" / "metadata.json").read_text())
    assert metadata["storage_status"] == "promoting"

    resumed = _open(tmp_path / "raw", values, requested_dtype="float32")

    assert resumed.metadata["storage_status"] == "ready"
    for path in resumed.complete_files:
        with np.load(path, allow_pickle=False) as payload:
            assert payload["values"].dtype == np.float32


def test_semantic_fingerprint_mismatch_rejects_partial_raw_directory(tmp_path: Path) -> None:
    values = np.array([[[0.1]]], dtype=np.float32)
    _open(tmp_path / "raw", values, semantic={"model": "a"})

    with pytest.raises(RawDepthFingerprintError, match="semantic fingerprint"):
        _open(tmp_path / "raw", values, semantic={"model": "b"})


def test_existing_corrupt_raw_payload_is_rejected_before_resume(tmp_path: Path) -> None:
    values = np.array([[[0.1, 0.2]]], dtype=np.float32)
    store = _open(tmp_path / "raw", values)
    payload = store.write_batch(["frame_000000.png"], values)[0]
    payload.write_bytes(b"not-an-npz")

    with pytest.raises(RawDepthFingerprintError, match="payload"):
        _open(tmp_path / "raw", values)


def test_requested_dtype_change_is_rejected_unless_it_promotes_float16(
    tmp_path: Path,
) -> None:
    values = np.array([[[0.1]]], dtype=np.float32)
    _open(tmp_path / "raw", values, requested_dtype="float16")

    with pytest.raises(RawDepthFingerprintError, match="requested dtype"):
        _open(tmp_path / "raw", values, requested_dtype="auto")


def test_disk_budget_uses_uncompressed_raw_and_canonical_payloads(tmp_path: Path) -> None:
    required = estimate_depth_disk_bytes(
        frame_count=10,
        native_width=20,
        native_height=30,
        storage_bytes=2,
        keep_intermediates=True,
    )

    assert required == int(10 * 20 * 30 * 2 * 1.25 + 10 * 20 * 30 * 2 * 1.10)
    require_disk_space(tmp_path, required, available_bytes=required)
    with pytest.raises(OSError, match="required"):
        require_disk_space(tmp_path, required, available_bytes=required - 1)
