"""Fast raw compression preserves the NPY contract and atomic resume boundary."""

import io
import struct
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest

from src.depth_surge_3d.inference.depth.types import (
    DepthBatch,
    DepthRepresentation,
    PinholeCameraBatch,
)
from src.depth_surge_3d.processing.frames.depth_storage import RawDepthStore, _atomic_save_npz


def _compressed_member(path, name):
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(name)
    data = path.read_bytes()
    header = struct.Struct("<4s5H3I2H")
    fields = header.unpack_from(data, info.header_offset)
    assert fields[0] == b"PK\x03\x04"
    start = info.header_offset + header.size + fields[-2] + fields[-1]
    return data[start : start + info.compress_size]


def test_raw_payload_uses_fast_lossless_deflate_stream(tmp_path):
    values = (np.arange(16384, dtype=np.float32) % 37).reshape(128, 128)
    destination = tmp_path / "frame.npz"
    _atomic_save_npz(destination, {"values": values})

    expected_npy = io.BytesIO()
    np.save(expected_npy, values, allow_pickle=False)
    compressor = zlib.compressobj(level=1, wbits=-15)
    expected_stream = compressor.compress(expected_npy.getvalue()) + compressor.flush()
    actual_stream = _compressed_member(destination, "values.npy")
    assert zlib.decompress(actual_stream, wbits=-15) == expected_npy.getvalue()
    assert actual_stream == expected_stream


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.dtype(">f4")])
@pytest.mark.parametrize("layout", ["c", "fortran", "strided", "empty"])
def test_fast_archive_preserves_exact_numpy_members_and_input(tmp_path, dtype, layout):
    values = np.array([[0.0, -0.0, np.nan], [np.inf, -np.inf, 0.125]], dtype=dtype)
    if layout == "fortran":
        values = np.asfortranarray(values)
    elif layout == "strided":
        values = values[::-1, ::-1]
    elif layout == "empty":
        values = values[:0]
    values.setflags(write=False)
    arrays = {"values": values, "focal_x_normalized": np.asarray(0.8, dtype=np.float32)}
    before = values.tobytes()
    reference = io.BytesIO()
    np.savez_compressed(reference, **arrays)
    destination = tmp_path / "frame.npz"

    _atomic_save_npz(destination, arrays)

    with zipfile.ZipFile(reference) as old, zipfile.ZipFile(destination) as new:
        assert new.namelist() == old.namelist() == ["values.npy", "focal_x_normalized.npy"]
        assert new.testzip() is None
        for name in old.namelist():
            assert new.getinfo(name).compress_type == zipfile.ZIP_DEFLATED
            assert new.read(name) == old.read(name)
    with np.load(destination, allow_pickle=False) as payload:
        for name, original in arrays.items():
            assert payload[name].dtype == original.dtype
            assert payload[name].shape == original.shape
            assert payload[name].tobytes() == original.tobytes()
    assert values.tobytes() == before
    assert not values.flags.writeable
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_archive_replacement_preserves_old_commit(tmp_path, monkeypatch):
    destination = tmp_path / "frame.npz"
    _atomic_save_npz(destination, {"values": np.ones((2, 3), np.float32)})
    original_bytes = destination.read_bytes()
    original_replace = Path.replace

    def fail_replace(path, target):
        if path == destination.with_name("frame.npz.tmp"):
            raise OSError("replacement failed")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failed"):
        _atomic_save_npz(destination, {"values": np.zeros((2, 3), np.float32)})
    assert destination.read_bytes() == original_bytes
    assert not list(tmp_path.glob("*.tmp"))


def test_mixed_compression_levels_remain_resumable(tmp_path):
    names = ["frame_000001.png", "frame_000002.png"]
    values = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    batch = DepthBatch(values, DepthRepresentation.INVERSE_DEPTH)
    fingerprint = {"camera_model": "none"}
    store = RawDepthStore.create(
        tmp_path,
        frame_names=names,
        semantic_fingerprint=fingerprint,
        requested_dtype="float32",
        first_batch=batch,
    )
    with store.path_for(names[0]).open("wb") as handle:
        np.savez_compressed(handle, values=values[0])
    first_bytes = store.path_for(names[0]).read_bytes()
    store.write_batch(names, batch)
    store.flush_metadata()

    resumed = RawDepthStore.open_existing(
        tmp_path,
        frame_names=names,
        semantic_fingerprint=fingerprint,
        requested_dtype="float32",
    )
    assert resumed.metadata["completed_count"] == 2
    assert store.path_for(names[0]).read_bytes() == first_bytes
    assert resumed.load_batch(resumed.complete_files).values.tobytes() == values.tobytes()


def test_failed_second_frame_keeps_prior_frame_and_can_resume(tmp_path, monkeypatch):
    names = ["frame_000001.png", "frame_000002.png"]
    values = np.stack([np.ones((2, 3), np.float32), np.full((2, 3), 2, np.float32)])
    batch = DepthBatch(values, DepthRepresentation.INVERSE_DEPTH)
    fingerprint = {"camera_model": "none"}
    store = RawDepthStore.create(
        tmp_path,
        frame_names=names,
        semantic_fingerprint=fingerprint,
        requested_dtype="float32",
        first_batch=batch,
    )
    original_write = np.lib.format.write_array

    def fail_second(handle, array, *args, **kwargs):
        if array.ndim == 2 and array[0, 0] == 2:
            handle.write(b"partial NPY")
            raise OSError("disk write failed")
        return original_write(handle, array, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(np.lib.format, "write_array", fail_second)
        with pytest.raises(OSError, match="disk write failed"):
            store.write_batch(names, batch)
    assert store.path_for(names[0]).is_file()
    assert not store.path_for(names[1]).exists()
    assert not list(tmp_path.glob("*.tmp"))
    resumed = RawDepthStore.open_existing(
        tmp_path,
        frame_names=names,
        semantic_fingerprint=fingerprint,
        requested_dtype="float32",
    )
    assert resumed.metadata["completed_count"] == 1
    resumed.write_batch(names, batch)
    resumed.flush_metadata()
    assert resumed.load_batch(resumed.complete_files).values.tobytes() == values.tobytes()


def test_failed_camera_member_does_not_publish_partial_frame(tmp_path, monkeypatch):
    names = ["frame_000001.png"]
    batch = DepthBatch(
        np.ones((1, 2, 3), np.float32),
        DepthRepresentation.METRIC_DEPTH,
        camera=PinholeCameraBatch(np.array([0.8], np.float32)),
    )
    store = RawDepthStore.create(
        tmp_path,
        frame_names=names,
        semantic_fingerprint={"camera_model": "pinhole_fx"},
        requested_dtype="float32",
        first_batch=batch,
    )
    original_write = np.lib.format.write_array

    def fail_focal(handle, array, *args, **kwargs):
        if array.ndim == 0:
            handle.write(b"partial camera")
            raise OSError("camera member failed")
        return original_write(handle, array, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(np.lib.format, "write_array", fail_focal)
        with pytest.raises(OSError, match="camera member failed"):
            store.write_batch(names, batch)
    assert not store.path_for(names[0]).exists()
    assert not list(tmp_path.glob("*.tmp"))
    assert store.metadata["completed_count"] == 0
    store.write_batch(names, batch)
    assert store.validate_payloads() == 1
    loaded = store.load_batch(store.complete_files)
    assert loaded.camera.focal_x_normalized.tobytes() == batch.camera.focal_x_normalized.tobytes()


def test_streamed_members_can_cross_zip64_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 32)
    values = np.arange(64, dtype=np.float32).reshape(8, 8)
    destination = tmp_path / "frame.npz"
    _atomic_save_npz(destination, {"values": values})
    with np.load(destination, allow_pickle=False) as payload:
        assert payload["values"].tobytes() == values.tobytes()
