"""Native raw-depth storage and resume contract tests."""

import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from src.depth_surge_3d.inference.depth.types import (
    DepthBatch,
    DepthRepresentation,
    PinholeCameraBatch,
)
from src.depth_surge_3d.processing.frames import depth_storage as depth_storage_module
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
    fingerprint = dict(semantic or {"model": "fixture", "revision": "1"})
    fingerprint.setdefault("camera_model", "none")
    if (directory / "metadata.json").is_file():
        return RawDepthStore.open_existing(
            directory,
            frame_names=["frame_000000.png", "frame_000001.png"],
            semantic_fingerprint=fingerprint,
            requested_dtype=requested_dtype,
        )
    return RawDepthStore.create(
        directory,
        frame_names=["frame_000000.png", "frame_000001.png"],
        semantic_fingerprint=fingerprint,
        requested_dtype=requested_dtype,
        first_batch=DepthBatch(first_values, DepthRepresentation.RELATIVE_DEPTH),
    )


def _batch(values: np.ndarray) -> DepthBatch:
    return DepthBatch(values, DepthRepresentation.RELATIVE_DEPTH)


VALID_VALUES = np.ones((2, 3), dtype=np.float32)
VALID_FOCAL = np.array(0.8, dtype=np.float32)


def metric_batch_with_focal(value: float) -> DepthBatch:
    return DepthBatch(
        np.ones((1, 2, 3), dtype=np.float32),
        DepthRepresentation.METRIC_DEPTH,
        camera=PinholeCameraBatch(np.array([value], dtype=np.float32)),
    )


def overwrite_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


@pytest.fixture
def v2_raw_store(tmp_path: Path) -> RawDepthStore:
    directory = tmp_path / "v2_raw"
    directory.mkdir()
    semantic = {"model": "fixture", "revision": "1"}
    metadata = {
        "schema_version": 2,
        "storage_status": "ready",
        "representation": "relative_depth",
        "frame_names": ["frame_000001.png"],
        "native_shape": [2, 3],
        "requested_dtype": "float32",
        "selected_dtype": "float32",
        "storage_provenance": "native_float32",
        "compression": "npz_deflate",
        "semantic_fingerprint": semantic,
        "completed_count": 1,
        "promoted_frame_count": 0,
    }
    metadata["fingerprint"] = RawDepthStore._fingerprint(metadata)
    metadata_path = directory / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    overwrite_npz(directory / "frame_000001.npz", {"values": VALID_VALUES})
    return RawDepthStore(directory, metadata)


@pytest.fixture
def v3_pinhole_store(tmp_path: Path) -> RawDepthStore:
    batch = metric_batch_with_focal(0.8)
    store = RawDepthStore.create(
        tmp_path / "v3_pinhole",
        frame_names=["frame_000001.png"],
        semantic_fingerprint={"camera_model": "pinhole_fx"},
        requested_dtype="float32",
        first_batch=batch,
    )
    store.write_batch(["frame_000001.png"], batch)
    return store


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


def test_model_fingerprint_tracks_inference_batch_size() -> None:
    class BatchedEstimator:
        def __init__(self, batch_size: int) -> None:
            self.batch_size = batch_size

        def get_model_info(self) -> dict:
            return {
                "model_name": "owner/model",
                "revision": "immutable-revision",
                "inference_batch_size": self.batch_size,
            }

    single = build_model_fingerprint(BatchedEstimator(1), {})
    batched = build_model_fingerprint(BatchedEstimator(4), {})

    assert single["model_info"]["inference_batch_size"] == 1
    assert batched["model_info"]["inference_batch_size"] == 4
    assert single != batched


def test_model_fingerprint_ignores_loaded_runtime_state() -> None:
    class StatefulEstimator:
        def __init__(self) -> None:
            self.loaded = False

        def get_model_info(self) -> dict:
            return {
                "model_name": "owner/model",
                "revision": "immutable-revision",
                "device": "cuda",
                "loaded": self.loaded,
                "memory_efficient": True,
            }

    estimator = StatefulEstimator()
    before_load = build_model_fingerprint(estimator, {"device": "cuda"})
    estimator.loaded = True
    after_load = build_model_fingerprint(estimator, {"device": "cuda"})

    assert before_load == after_load
    assert before_load["model_info"] == {
        "model_name": "owner/model",
        "revision": "immutable-revision",
        "device": "cuda",
    }


def test_model_fingerprint_rejects_unclassified_model_info_fields() -> None:
    class FutureEstimator:
        def get_model_info(self) -> dict:
            return {
                "model_name": "owner/model",
                "tile_size": 512,
            }

    with pytest.raises(ValueError, match=r"Unclassified model info fields: tile_size"):
        build_model_fingerprint(FutureEstimator(), {})


def test_unreadable_raw_schema_reports_schema_mismatch(tmp_path: Path) -> None:
    values = np.array([[[0.1, 0.2]]], dtype=np.float32)
    store = _open(tmp_path / "raw", values)
    assert store.metadata["schema_version"] == 3
    store.metadata["schema_version"] = 1
    store.metadata_path.write_text(json.dumps(store.metadata), encoding="utf-8")

    with pytest.raises(RawDepthFingerprintError, match="schema version"):
        _open(tmp_path / "raw", values)


def test_new_depth_only_store_writes_v3_values_only(tmp_path: Path) -> None:
    batch = DepthBatch(
        np.ones((1, 2, 3), dtype=np.float32),
        DepthRepresentation.RELATIVE_DEPTH,
    )
    store = RawDepthStore.create(
        tmp_path,
        frame_names=["frame_000001.png"],
        semantic_fingerprint={"camera_model": "none"},
        requested_dtype="float32",
        first_batch=batch,
    )
    store.write_batch(["frame_000001.png"], batch)
    assert store.metadata["schema_version"] == 3
    assert store.metadata["camera_model"] == "none"
    with zipfile.ZipFile(store.path_for("frame_000001.png")) as payload:
        assert payload.namelist() == ["values.npy"]


def test_v3_pinhole_payload_commits_depth_and_focal_together(tmp_path: Path) -> None:
    batch = metric_batch_with_focal(0.8)
    store = RawDepthStore.create(
        tmp_path,
        frame_names=["frame_000001.png"],
        semantic_fingerprint={"camera_model": "pinhole_fx"},
        requested_dtype="float32",
        first_batch=batch,
    )
    path = store.write_batch(["frame_000001.png"], batch)[0]
    with zipfile.ZipFile(path) as payload:
        assert payload.namelist() == ["values.npy", "focal_x_normalized.npy"]
    loaded = store.load_batch([path])
    assert loaded.camera is not None
    assert loaded.camera.focal_x_normalized.tolist() == pytest.approx([0.8])


def test_schema_v2_values_only_store_remains_reusable(v2_raw_store: RawDepthStore) -> None:
    before_metadata = v2_raw_store.metadata_path.read_bytes()
    before_payloads = [path.read_bytes() for path in v2_raw_store.complete_files]
    reopened = RawDepthStore.open_existing(
        v2_raw_store.directory,
        frame_names=["frame_000001.png"],
        semantic_fingerprint={
            **v2_raw_store.metadata["semantic_fingerprint"],
            "camera_model": "none",
        },
        requested_dtype="float32",
    )
    assert reopened.metadata["schema_version"] == 2
    assert reopened.camera_model == "none"
    assert reopened.load_batch(reopened.complete_files).camera is None
    assert v2_raw_store.metadata_path.read_bytes() == before_metadata
    assert [path.read_bytes() for path in v2_raw_store.complete_files] == before_payloads


def test_schema_v2_float16_store_is_not_promoted_or_rewritten(
    v2_raw_store: RawDepthStore,
) -> None:
    v2_raw_store.metadata.update(
        {
            "requested_dtype": "auto",
            "selected_dtype": "float16",
            "storage_provenance": "native_float16",
        }
    )
    v2_raw_store.metadata["fingerprint"] = RawDepthStore._fingerprint(v2_raw_store.metadata)
    v2_raw_store.metadata_path.write_text(
        json.dumps(v2_raw_store.metadata, indent=2), encoding="utf-8"
    )
    overwrite_npz(
        v2_raw_store.complete_files[0],
        {"values": VALID_VALUES.astype(np.float16)},
    )
    before_metadata = v2_raw_store.metadata_path.read_bytes()
    before_payload = v2_raw_store.complete_files[0].read_bytes()

    reopened = RawDepthStore.open_existing(
        v2_raw_store.directory,
        frame_names=["frame_000001.png"],
        semantic_fingerprint={
            **v2_raw_store.metadata["semantic_fingerprint"],
            "camera_model": "none",
        },
        requested_dtype="float32",
    )

    assert reopened.metadata["selected_dtype"] == "float16"
    assert v2_raw_store.metadata_path.read_bytes() == before_metadata
    assert v2_raw_store.complete_files[0].read_bytes() == before_payload


def test_schema_v2_interrupted_promotion_is_rejected(v2_raw_store: RawDepthStore) -> None:
    v2_raw_store.metadata["storage_status"] = "promoting"
    v2_raw_store.metadata_path.write_text(json.dumps(v2_raw_store.metadata), encoding="utf-8")

    with pytest.raises(RawDepthFingerprintError, match="transaction status"):
        RawDepthStore.open_existing(
            v2_raw_store.directory,
            frame_names=["frame_000001.png"],
            semantic_fingerprint={
                **v2_raw_store.metadata["semantic_fingerprint"],
                "camera_model": "none",
            },
            requested_dtype="float32",
        )


@pytest.mark.parametrize(
    ("arrays", "message"),
    [
        ({"values": VALID_VALUES}, "focal_x_normalized"),
        (
            {
                "values": VALID_VALUES,
                "focal_x_normalized": VALID_FOCAL,
                "extra": np.array(1),
            },
            "exact members",
        ),
        (
            {"values": VALID_VALUES, "focal_x_normalized": np.array([0.8], np.float32)},
            "scalar",
        ),
        (
            {"values": VALID_VALUES, "focal_x_normalized": np.array(0.8, np.float64)},
            "float32",
        ),
        (
            {"values": VALID_VALUES, "focal_x_normalized": np.array(0.0, np.float32)},
            "positive",
        ),
        (
            {"values": VALID_VALUES, "focal_x_normalized": np.array(np.nan, np.float32)},
            "finite",
        ),
    ],
)
def test_corrupt_v3_pinhole_payload_is_rejected(
    v3_pinhole_store: RawDepthStore,
    arrays: dict[str, np.ndarray],
    message: str,
) -> None:
    overwrite_npz(v3_pinhole_store.complete_files[0], arrays)
    with pytest.raises(ValueError, match=message):
        v3_pinhole_store.validate_payloads()


def test_v3_camera_model_must_match_semantic_fingerprint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="camera_model"):
        RawDepthStore.create(
            tmp_path,
            frame_names=["frame_000001.png"],
            semantic_fingerprint={"camera_model": "none"},
            requested_dtype="float32",
            first_batch=metric_batch_with_focal(0.8),
        )


def test_interrupted_npz_temporary_is_not_a_committed_frame(
    v3_pinhole_store: RawDepthStore,
) -> None:
    temporary = v3_pinhole_store.directory / "frame_000002.npz.tmp"
    temporary.write_bytes(b"incomplete")
    assert v3_pinhole_store.validate_payloads() == 1
    assert not temporary.exists()


def test_failed_npz_write_leaves_neither_commit_nor_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _batch(np.ones((1, 2, 3), dtype=np.float32))
    store = RawDepthStore.create(
        tmp_path,
        frame_names=["frame_000001.png"],
        semantic_fingerprint={"camera_model": "none"},
        requested_dtype="float32",
        first_batch=batch,
    )

    def fail_save(handle, **_arrays) -> None:
        handle.write(b"partial")
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(depth_storage_module.np, "savez_compressed", fail_save)
    with pytest.raises(OSError, match="interrupted"):
        store.write_batch(["frame_000001.png"], batch)

    assert not store.path_for("frame_000001.png").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_read_metadata_does_not_require_payloads_and_rejects_bad_json(tmp_path: Path) -> None:
    assert RawDepthStore.read_metadata(tmp_path) is None
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text('{"schema_version": 3}', encoding="utf-8")
    assert RawDepthStore.read_metadata(tmp_path) == {"schema_version": 3}
    metadata_path.write_text("[]", encoding="utf-8")
    assert RawDepthStore.read_metadata(tmp_path) is None
    metadata_path.write_text("{", encoding="utf-8")
    assert RawDepthStore.read_metadata(tmp_path) is None


def test_auto_store_selects_float16_and_writes_atomic_compressed_files(tmp_path: Path) -> None:
    first = np.array([[[0.0, 0.5], [1.0, np.nan]]], dtype=np.float32)
    store = _open(tmp_path / "raw", first)

    paths = store.write_batch(["frame_000000.png"], _batch(first))

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
    first_path = store.write_batch(["frame_000000.png"], _batch(first))[0]

    store.write_batch(["frame_000001.png"], _batch(second))

    assert store.metadata["selected_dtype"] == "float32"
    assert store.metadata["storage_provenance"] == "promoted_float16_to_float32"
    assert store.metadata["promoted_frame_count"] == 1
    with np.load(first_path, allow_pickle=False) as payload:
        assert payload["values"].dtype == np.float32
        np.testing.assert_array_equal(
            payload["values"], first.astype(np.float16).astype(np.float32)[0]
        )


def test_pinhole_promotion_reemits_original_scalar_focal(tmp_path: Path) -> None:
    first = metric_batch_with_focal(0.8)
    store = RawDepthStore.create(
        tmp_path,
        frame_names=["frame_000001.png", "frame_000002.png"],
        semantic_fingerprint={"camera_model": "pinhole_fx"},
        requested_dtype="auto",
        first_batch=first,
    )
    first_path = store.write_batch(["frame_000001.png"], first)[0]
    second = DepthBatch(
        np.full((1, 2, 3), 70000.0, dtype=np.float32),
        DepthRepresentation.METRIC_DEPTH,
        camera=PinholeCameraBatch(np.array([0.9], dtype=np.float32)),
    )

    store.write_batch(["frame_000002.png"], second)

    with np.load(first_path, allow_pickle=False) as payload:
        assert payload.files == ["values", "focal_x_normalized"]
        assert payload["values"].dtype == np.float32
        assert payload["focal_x_normalized"].shape == ()
        assert payload["focal_x_normalized"].dtype == np.float32
        assert payload["focal_x_normalized"].item() == pytest.approx(0.8)


def test_explicit_float16_can_resume_as_float32_without_deleting_completed_files(
    tmp_path: Path,
) -> None:
    first = np.array([[[0.1, 0.2]]], dtype=np.float32)
    store = _open(tmp_path / "raw", first, requested_dtype="float16")
    first_path = store.write_batch(["frame_000000.png"], _batch(first))[0]
    with pytest.raises(RawDepthOverflowError):
        store.write_batch(
            ["frame_000001.png"],
            _batch(np.array([[[70000.0, 2.0]]], np.float32)),
        )

    resumed = _open(tmp_path / "raw", first, requested_dtype="float32")

    assert first_path.exists()
    assert resumed.metadata["selected_dtype"] == "float32"
    assert resumed.complete_files == [first_path]


def test_promotion_resumes_after_atomic_rewrite_crash(tmp_path: Path, monkeypatch) -> None:
    values = np.array([[[0.1]], [[0.2]]], dtype=np.float32)
    store = _open(tmp_path / "raw", values, requested_dtype="float16")
    store.write_batch(["frame_000000.png", "frame_000001.png"], _batch(values))
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
    payload = store.write_batch(["frame_000000.png"], _batch(values))[0]
    payload.write_bytes(b"not-an-npz")

    with pytest.raises(RawDepthFingerprintError, match="payload"):
        _open(tmp_path / "raw", values)


def test_payload_validation_reads_npz_headers_without_materializing_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = np.array([[[0.1, 0.2]]], dtype=np.float32)
    store = _open(tmp_path / "raw", values)
    store.write_batch(["frame_000000.png"], _batch(values))

    def reject_materialization(*_args, **_kwargs):
        raise AssertionError("payload data was materialized")

    monkeypatch.setattr(depth_storage_module.np, "asarray", reject_materialization)

    assert store.validate_payloads() == 1


def test_payload_validation_compares_names_without_resolving_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = np.array([[[0.1, 0.2]]], dtype=np.float32)
    store = _open(tmp_path / "raw", values)
    store.write_batch(["frame_000000.png"], _batch(values))
    original_resolve = Path.resolve
    resolve_calls = 0

    def counted_resolve(path: Path, *args, **kwargs) -> Path:
        nonlocal resolve_calls
        resolve_calls += 1
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", counted_resolve)

    assert store.validate_payloads() == 1
    assert resolve_calls == 0


def test_write_batch_defers_metadata_rewrite_until_stage_flush(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = np.array([[[0.1, 0.2]]], dtype=np.float32)
    store = _open(tmp_path / "raw", values)
    writes: list[dict] = []

    monkeypatch.setattr(
        depth_storage_module,
        "_atomic_write_json",
        lambda _path, payload: writes.append(dict(payload)),
    )

    store.write_batch(["frame_000000.png"], _batch(values))

    assert store.metadata["completed_count"] == 1
    assert writes == []

    store.flush_metadata()

    assert len(writes) == 1
    assert writes[0]["completed_count"] == 1


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
