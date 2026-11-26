"""Production contracts for the closed metric NPZ grammar and owned writer."""

from __future__ import annotations

import errno
import hashlib
import io
import os
from pathlib import Path
import struct
import zlib

import numpy as np
import pytest

from src.depth_surge_3d.io import metric_npz
from src.depth_surge_3d.io.durable_publish import (
    OwnedTemporaryCollisionError,
    OwnedTemporaryIdentityError,
)


def _metric_values() -> tuple[np.ndarray, np.ndarray, np.float32]:
    inverse = np.array(
        [[0.0, 0.5, 1.0], [2.0, 4.0, 8.0]],
        dtype=np.float32,
    )
    valid = np.array(
        [[False, True, True], [True, True, True]],
        dtype=np.bool_,
    )
    return inverse, valid, np.float32(1.25)


def _write_owned(path: Path) -> metric_npz.MetricNpzWriteTrace:
    inverse, valid, focal = _metric_values()
    return metric_npz.write_metric_npz(path, inverse, valid, focal)


def _numpy_history_bytes() -> bytes:
    inverse, valid, focal = _metric_values()
    target = io.BytesIO()
    np.savez_compressed(
        target,
        inverse_depth=inverse,
        valid=valid,
        focal_x_normalized=np.asarray(focal, dtype=np.float32),
    )
    return target.getvalue()


def _central_offsets(payload: bytes) -> list[int]:
    eocd = payload.rfind(b"PK\x05\x06")
    assert eocd >= 0
    central = struct.unpack_from("<IHHHHIIH", payload, eocd)[6]
    offsets: list[int] = []
    cursor = central
    for _ in range(3):
        offsets.append(cursor)
        fields = struct.unpack_from("<IHHHHHHIIIHHHHHII", payload, cursor)
        cursor += 46 + fields[10] + fields[11] + fields[12]
    assert cursor == eocd
    return offsets


def _local_offsets(payload: bytes) -> list[int]:
    offsets: list[int] = []
    cursor = 0
    for _ in range(3):
        offsets.append(cursor)
        fields = struct.unpack_from("<IHHHHHIIIHH", payload, cursor)
        extra = payload[cursor + 30 + fields[9] : cursor + 30 + fields[9] + fields[10]]
        if fields[7] == 0xFFFFFFFF:
            assert extra[:4] == b"\x01\x00\x10\x00"
            compressed = struct.unpack_from("<Q", extra, 12)[0]
        else:
            compressed = fields[7]
        cursor += 30 + fields[9] + fields[10] + compressed
    return offsets


def _history_as_u32(payload: bytes) -> bytes:
    transformed = bytearray(payload)
    for offset in _local_offsets(payload):
        extra_offset = offset + 30 + struct.unpack_from("<H", payload, offset + 26)[0]
        uncompressed, compressed = struct.unpack_from("<QQ", payload, extra_offset + 4)
        struct.pack_into("<H", transformed, offset + 4, 20)
        struct.pack_into("<II", transformed, offset + 18, compressed, uncompressed)
    for offset in _central_offsets(payload):
        made_by = struct.unpack_from("<H", payload, offset + 4)[0]
        struct.pack_into("<H", transformed, offset + 4, (made_by & 0xFF00) | 20)
        struct.pack_into("<H", transformed, offset + 6, 20)
    return bytes(transformed)


def _history_as_sentinel(payload: bytes) -> bytes:
    transformed = bytearray(payload)
    for offset in _local_offsets(payload):
        struct.pack_into("<H", transformed, offset + 4, 45)
        struct.pack_into("<II", transformed, offset + 18, 0xFFFFFFFF, 0xFFFFFFFF)
    for offset in _central_offsets(payload):
        made_by = struct.unpack_from("<H", payload, offset + 4)[0]
        struct.pack_into("<H", transformed, offset + 4, (made_by & 0xFF00) | 45)
        struct.pack_into("<H", transformed, offset + 6, 45)
    return bytes(transformed)


def _canonical_npy(
    name: str,
    data: bytes,
    *,
    shape: tuple[int, ...],
    descr: str | None = None,
    fortran: str = "False",
    version: bytes = b"\x01\x00",
) -> bytes:
    if name == "inverse_depth.npy":
        dtype = descr or "<f4"
        shape_text = f"({shape[0]}, {shape[1]})"
    elif name == "valid.npy":
        dtype = descr or "|b1"
        shape_text = f"({shape[0]}, {shape[1]})"
    else:
        dtype = descr or "<f4"
        shape_text = "()"
    body = (
        f"{{'descr': '{dtype}', 'fortran_order': {fortran}, " f"'shape': {shape_text}, }}"
    ).encode("ascii")
    spaces = (-(10 + len(body) + 1)) % 64 or 64
    header = body + b" " * spaces + b"\n"
    return b"\x93NUMPY" + version + struct.pack("<H", len(header)) + header + data


def _owned_archive(
    members: list[tuple[str, bytes]],
    *,
    crc_delta: int = 0,
) -> bytes:
    output = bytearray()
    records: list[tuple[str, int, int, int, int]] = []
    for name, raw in members:
        compressor = zlib.compressobj(level=9, wbits=-zlib.MAX_WBITS)
        compressed = compressor.compress(raw) + compressor.flush()
        crc = (zlib.crc32(raw) + crc_delta) & 0xFFFFFFFF
        local_offset = len(output)
        encoded = name.encode("ascii")
        output.extend(
            struct.pack(
                "<IHHHHHIIIHH",
                0x04034B50,
                20,
                0,
                8,
                0,
                33,
                crc,
                len(compressed),
                len(raw),
                len(encoded),
                0,
            )
        )
        output.extend(encoded)
        output.extend(compressed)
        records.append((name, local_offset, crc, len(compressed), len(raw)))
    central_offset = len(output)
    for name, local_offset, crc, compressed, raw in records:
        encoded = name.encode("ascii")
        output.extend(
            struct.pack(
                "<IHHHHHHIIIHHHHHII",
                0x02014B50,
                20,
                20,
                0,
                8,
                0,
                33,
                crc,
                compressed,
                raw,
                len(encoded),
                0,
                0,
                0,
                0,
                0,
                local_offset,
            )
        )
        output.extend(encoded)
    central_size = len(output) - central_offset
    output.extend(
        struct.pack(
            "<IHHHHIIH",
            0x06054B50,
            0,
            0,
            3,
            3,
            central_size,
            central_offset,
            0,
        )
    )
    return bytes(output)


def _raw_members() -> list[tuple[str, bytes]]:
    inverse, valid, focal = _metric_values()
    return [
        (
            "inverse_depth.npy",
            _canonical_npy("inverse_depth.npy", inverse.tobytes(), shape=inverse.shape),
        ),
        ("valid.npy", _canonical_npy("valid.npy", valid.tobytes(), shape=valid.shape)),
        (
            "focal_x_normalized.npy",
            _canonical_npy(
                "focal_x_normalized.npy",
                struct.pack("<f", float(focal)),
                shape=(),
            ),
        ),
    ]


def test_owned_writer_emits_the_frozen_zip32_layout(tmp_path: Path) -> None:
    path = tmp_path / "frame_000000.npz"
    trace = _write_owned(path)

    audit = metric_npz.audit_metric_npz(path, expected_shape=(2, 3))

    assert audit.layout.archive_form == "owned-zip32-v1"
    assert [member.name for member in audit.layout.members] == [
        "inverse_depth.npy",
        "valid.npy",
        "focal_x_normalized.npy",
    ]
    assert audit.inverse_depth.tolist() == [[0.0, 0.5, 1.0], [2.0, 4.0, 8.0]]
    assert audit.valid.tolist() == [[False, True, True], [True, True, True]]
    assert audit.focal_x_normalized == np.float32(1.25)
    assert audit.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert audit.stream_workspace_peak_bytes <= metric_npz.METRIC_NPZ_WRITE_STREAM_BYTES
    assert trace.complete_raw_member_copies == 0
    assert trace.complete_compressed_member_copies == 0
    assert trace.maximum_input_chunk_bytes <= metric_npz.METRIC_NPZ_WRITE_INPUT_CHUNK_BYTES
    assert trace.maximum_output_block_bytes <= metric_npz.METRIC_NPZ_WRITE_STREAM_BYTES


def test_identity_audit_hashes_and_scans_headers_without_allocating_metric_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "frame_000000.npz"
    _write_owned(path)

    monkeypatch.setattr(
        metric_npz,
        "_allocate_arrays",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("identity replay must not allocate metric arrays")
        ),
    )

    identity = metric_npz.audit_metric_npz_identity(path, expected_shape=(2, 3))

    assert identity.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert identity.byte_count == path.stat().st_size
    assert identity.layout.archive_form == "owned-zip32-v1"
    assert identity.member_shapes == ((2, 3), (2, 3), ())
    assert identity.maximum_stream_bytes <= metric_npz.METRIC_NPZ_WRITE_STREAM_BYTES
    payload = path.read_bytes()
    for offset in _local_offsets(payload):
        local = struct.unpack_from("<IHHHHHIIIHH", payload, offset)
        assert (local[1], local[2], local[3], local[4], local[5], local[10]) == (
            20,
            0,
            8,
            0,
            33,
            0,
        )
    for offset in _central_offsets(payload):
        central = struct.unpack_from("<IHHHHHHIIIHHHHHII", payload, offset)
        assert (central[1], central[2], central[3], central[4]) == (20, 20, 0, 8)
        assert (central[14], central[15]) == (0, 0)


def test_current_numpy_payload_classifies_as_the_frozen_historical_form(tmp_path: Path) -> None:
    path = tmp_path / "current.npz"
    path.write_bytes(_numpy_history_bytes())

    audit = metric_npz.audit_metric_npz(path, expected_shape=(2, 3))

    assert audit.layout.archive_form in {
        "numpy-force-zip64-local-u32-v1",
        "numpy-force-zip64-local-sentinel-v1",
    }


def test_current_atomic_save_npz_output_uses_owned_zip32_form(
    tmp_path: Path,
) -> None:
    from src.depth_surge_3d.processing.frames.metric_geometry import (
        MetricGeometryFrame,
        _atomic_save_npz,
    )

    inverse, valid, focal = _metric_values()
    frame = MetricGeometryFrame(inverse, valid, focal)
    path = tmp_path / "current-atomic.npz"

    _atomic_save_npz(path, frame, required_bytes=0)
    audit = metric_npz.audit_metric_npz(path, expected_shape=(2, 3))

    assert audit.layout.archive_form == "owned-zip32-v1"
    assert np.array_equal(audit.inverse_depth, inverse)
    assert np.array_equal(audit.valid, valid)
    assert audit.focal_x_normalized == focal


def test_version_20_historical_golden_classifies_without_runtime_branch(tmp_path: Path) -> None:
    path = tmp_path / "cpython-3.10.11.npz"
    path.write_bytes(_history_as_u32(_numpy_history_bytes()))

    audit = metric_npz.audit_metric_npz(path, expected_shape=(2, 3))

    assert audit.layout.archive_form == "numpy-force-zip64-local-u32-v1"


def test_version_45_historical_golden_classifies_without_runtime_branch(tmp_path: Path) -> None:
    path = tmp_path / "cpython-3.12.1.npz"
    path.write_bytes(_history_as_sentinel(_numpy_history_bytes()))

    audit = metric_npz.audit_metric_npz(path, expected_shape=(2, 3))

    assert audit.layout.archive_form == "numpy-force-zip64-local-sentinel-v1"


def test_audit_allocator_receives_only_the_two_final_arrays(tmp_path: Path) -> None:
    path = tmp_path / "frame_000000.npz"
    _write_owned(path)
    allocations: list[tuple[tuple[int, ...], np.dtype, int]] = []

    def allocate(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        result = np.empty(shape, dtype=dtype)
        allocations.append((shape, np.dtype(dtype), result.nbytes))
        return result

    audit = metric_npz.audit_metric_npz(
        path,
        expected_shape=(2, 3),
        allocator=allocate,
    )

    assert allocations == [
        ((2, 3), np.dtype(np.float32), 24),
        ((2, 3), np.dtype(np.bool_), 6),
    ]
    assert audit.stream_workspace_peak_bytes <= metric_npz.METRIC_NPZ_WRITE_STREAM_BYTES


@pytest.mark.parametrize(
    ("label", "expected", "mutate"),
    [
        ("flag", "flag bits", lambda data: struct.pack_into("<H", data, 6, 8)),
        ("method", "method", lambda data: struct.pack_into("<H", data, 8, 0)),
        ("local-name", "local name", lambda data: data.__setitem__(30, ord("I"))),
        (
            "local-offset",
            "local offset",
            lambda data: struct.pack_into("<I", data, _central_offsets(data)[0] + 42, 1),
        ),
        ("crc", "record disagreement", lambda data: struct.pack_into("<I", data, 14, 0)),
        ("trailing", "trailing archive data", lambda data: data.extend(b"x")),
    ],
)
def test_closed_archive_grammar_rejects_malformed_records(
    tmp_path: Path,
    label: str,
    expected: str,
    mutate,
) -> None:
    path = tmp_path / "malformed.npz"
    _write_owned(path)
    payload = bytearray(path.read_bytes())
    mutate(payload)
    path.write_bytes(payload)

    with pytest.raises(metric_npz.MetricNpzFormatError, match=expected):
        metric_npz.audit_metric_npz(path, expected_shape=(2, 3))


@pytest.mark.parametrize(
    ("label", "expected", "member_index", "replacement"),
    [
        (
            "npy version",
            "descriptor/version",
            0,
            lambda name, raw: _canonical_npy(
                name,
                raw[-24:],
                shape=(2, 3),
                version=b"\x02\x00",
            ),
        ),
        (
            "dtype",
            "dtype/shape grammar",
            0,
            lambda name, raw: _canonical_npy(
                name,
                raw[-24:],
                shape=(2, 3),
                descr=">f4",
            ),
        ),
        (
            "fortran",
            "dtype/shape grammar",
            1,
            lambda name, raw: _canonical_npy(
                name,
                raw[-6:],
                shape=(2, 3),
                fortran="True",
            ),
        ),
        (
            "shape mismatch",
            "shape mismatch",
            0,
            lambda name, raw: _canonical_npy(name, raw[-24:], shape=(3, 2)),
        ),
        ("overlong", "declared uncompressed size", 1, lambda _name, raw: raw + b"\x00"),
        ("short", "declared uncompressed size", 1, lambda _name, raw: raw[:-1]),
    ],
)
def test_closed_npy_grammar_rejects_malformed_members(
    tmp_path: Path,
    label: str,
    expected: str,
    member_index: int,
    replacement,
) -> None:
    members = _raw_members()
    name, raw = members[member_index]
    members[member_index] = (name, replacement(name, raw))
    path = tmp_path / "malformed-npy.npz"
    path.write_bytes(_owned_archive(members))

    with pytest.raises(metric_npz.MetricNpzError, match=expected):
        metric_npz.audit_metric_npz(path, expected_shape=(2, 3))


def test_crc_mismatch_is_checked_against_the_inflated_member(tmp_path: Path) -> None:
    path = tmp_path / "crc.npz"
    path.write_bytes(_owned_archive(_raw_members(), crc_delta=1))

    with pytest.raises(metric_npz.MetricNpzFormatError, match="crc"):
        metric_npz.audit_metric_npz(path, expected_shape=(2, 3))


def test_corrupt_deflate_stream_cannot_fall_back_to_a_generic_decoder(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.npz"
    _write_owned(path)
    payload = bytearray(path.read_bytes())
    first = _local_offsets(payload)[0]
    fields = struct.unpack_from("<IHHHHHIIIHH", payload, first)
    data_offset = first + 30 + fields[9] + fields[10]
    payload[data_offset + fields[7] - 2] ^= 0x80
    path.write_bytes(payload)

    with pytest.raises(metric_npz.MetricNpzFormatError, match="DEFLATE|crc"):
        metric_npz.audit_metric_npz(path, expected_shape=(2, 3))


def test_valid_member_rejects_non_boolean_storage_byte(tmp_path: Path) -> None:
    members = _raw_members()
    name, raw = members[1]
    malformed = bytearray(raw)
    malformed[-1] = 2
    members[1] = (name, bytes(malformed))
    path = tmp_path / "invalid-bool.npz"
    path.write_bytes(_owned_archive(members))

    with pytest.raises(metric_npz.MetricNpzContentError, match="0 or 1"):
        metric_npz.audit_metric_npz(path, expected_shape=(2, 3))


@pytest.mark.parametrize("cross", ["sentinel-as-20", "u32-as-45"])
def test_crossed_historical_forms_are_rejected(tmp_path: Path, cross: str) -> None:
    current = _numpy_history_bytes()
    payload = (
        _history_as_sentinel(current) if cross == "sentinel-as-20" else _history_as_u32(current)
    )
    transformed = bytearray(payload)
    version = 20 if cross == "sentinel-as-20" else 45
    for offset in _local_offsets(payload):
        struct.pack_into("<H", transformed, offset + 4, version)
    for offset in _central_offsets(payload):
        made_by = struct.unpack_from("<H", payload, offset + 4)[0]
        struct.pack_into("<H", transformed, offset + 4, (made_by & 0xFF00) | version)
        struct.pack_into("<H", transformed, offset + 6, version)
    path = tmp_path / f"{cross}.npz"
    path.write_bytes(transformed)

    with pytest.raises(metric_npz.MetricNpzFormatError):
        metric_npz.audit_metric_npz(path, expected_shape=(2, 3))


def test_historical_external_attributes_are_not_widened(tmp_path: Path) -> None:
    payload = bytearray(_numpy_history_bytes())
    for offset in _central_offsets(payload):
        struct.pack_into("<I", payload, offset + 38, 0)
    path = tmp_path / "bad-attributes.npz"
    path.write_bytes(payload)

    with pytest.raises(metric_npz.MetricNpzFormatError, match="attributes"):
        metric_npz.audit_metric_npz(path, expected_shape=(2, 3))


@pytest.mark.parametrize(
    "failure_stage",
    ["compression", "seek", "patch", "sync", "replace", "disk-full"],
)
def test_owned_writer_preserves_old_bytes_before_replace_failure(
    tmp_path: Path,
    failure_stage: str,
) -> None:
    path = tmp_path / "frame_000000.npz"
    old = b"certified old payload"
    path.write_bytes(old)

    def fail(stage: str) -> None:
        if stage != failure_stage:
            return
        if stage == "disk-full":
            raise OSError(errno.ENOSPC, "injected disk full")
        raise RuntimeError(f"injected {stage}")

    inverse, valid, focal = _metric_values()
    expected = metric_npz.MetricNpzDiskFullError if failure_stage == "disk-full" else RuntimeError
    with pytest.raises(expected):
        metric_npz.write_metric_npz(path, inverse, valid, focal, fault=fail)

    assert path.read_bytes() == old
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("collision", ["regular", "hardlink", "symlink"])
def test_owned_writer_preserves_an_unowned_temporary_collision(
    tmp_path: Path,
    collision: str,
) -> None:
    path = tmp_path / "frame_000000.npz"
    temporary = path.with_name(f".{path.name}.metric-npz.tmp")
    sentinel = tmp_path / "outside.bin"
    sentinel.write_bytes(b"must survive")
    if collision == "regular":
        temporary.write_bytes(b"unowned")
    elif collision == "hardlink":
        os.link(sentinel, temporary)
    else:
        try:
            temporary.symlink_to(sentinel)
        except OSError as error:
            pytest.skip(f"symlink creation unavailable: {error}")

    with pytest.raises(OwnedTemporaryCollisionError, match="already exists"):
        _write_owned(path)

    assert sentinel.read_bytes() == b"must survive"
    assert temporary.exists() or temporary.is_symlink()


def test_owned_writer_preserves_a_post_sync_name_replacement(tmp_path: Path) -> None:
    path = tmp_path / "frame_000000.npz"
    temporary = path.with_name(f".{path.name}.metric-npz.tmp")
    parked = tmp_path / "parked.tmp"

    def replace_temporary(stage: str) -> None:
        if stage == "after-sync":
            temporary.rename(parked)
            temporary.write_bytes(b"replacement")
            raise RuntimeError("simulated crash")

    inverse, valid, focal = _metric_values()
    with pytest.raises(OwnedTemporaryIdentityError, match="changed"):
        metric_npz.write_metric_npz(
            path,
            inverse,
            valid,
            focal,
            fault=replace_temporary,
        )

    assert parked.is_file()
    assert temporary.read_bytes() == b"replacement"
    assert not path.exists()


def test_size_bound_is_checked_before_opening_temporary(tmp_path: Path) -> None:
    calls: list[int] = []

    def reserve(_path: Path, byte_count: int) -> None:
        calls.append(byte_count)

    bound = metric_npz.metric_npz_temp_logical_bound((2160, 3840))

    assert bound > 5 * 2160 * 3840
    assert calls == []


def test_zip32_overflow_is_rejected_without_a_filesystem_mutation(tmp_path: Path) -> None:
    with pytest.raises(metric_npz.MetricNpzSizeError, match="ZIP32"):
        metric_npz.metric_npz_temp_logical_bound((65536, 65536))

    assert list(tmp_path.iterdir()) == []


def test_reserve_check_precedes_temporary_open_and_keeps_old_target(tmp_path: Path) -> None:
    path = tmp_path / "frame_000000.npz"
    old = b"old target remains allocated"
    path.write_bytes(old)
    observations: list[tuple[int, bytes, list[Path]]] = []

    def reserve(_directory: Path, byte_count: int) -> None:
        observations.append((byte_count, path.read_bytes(), list(tmp_path.glob("*.tmp"))))

    inverse, valid, focal = _metric_values()
    trace = metric_npz.write_metric_npz(path, inverse, valid, focal, reserve=reserve)

    assert observations == [(trace.logical_bound_bytes, old, [])]


def test_checked_write_rejects_a_short_return() -> None:
    class ShortHandle:
        def write(self, payload: bytes) -> int:
            return len(payload) - 1

    with pytest.raises(metric_npz.MetricNpzWriteError, match="short"):
        metric_npz._write_checked(ShortHandle(), b"payload")


def test_parent_sync_failure_occurs_after_irreversible_replace(tmp_path: Path) -> None:
    path = tmp_path / "frame_000000.npz"
    old = b"old"
    path.write_bytes(old)

    def fail(stage: str) -> None:
        if stage == "parent-sync":
            raise RuntimeError("injected parent sync")

    inverse, valid, focal = _metric_values()
    with pytest.raises(RuntimeError, match="parent sync"):
        metric_npz.write_metric_npz(path, inverse, valid, focal, fault=fail)

    assert path.read_bytes() != old
    audit = metric_npz.audit_metric_npz(path, expected_shape=(2, 3))
    assert audit.layout.archive_form == "owned-zip32-v1"


def test_multiblock_depth_frame_has_one_identity_and_roundtrips_without_aliasing(tmp_path):
    from src.depth_surge_3d.io.nofollow_io import AcquiredRoot

    shape = (512, 768)
    row = np.linspace(0.125, 4.0, shape[1], dtype=np.float32)
    inverse = np.tile(row, (shape[0], 1))
    valid = inverse < 3.0
    inverse[~valid] = 0.0
    path = tmp_path / "frame.npz"
    trace = metric_npz.write_metric_npz(path, inverse, valid, np.float32(0.75))
    with AcquiredRoot(tmp_path) as acquired:
        identity = metric_npz.audit_metric_npz_identity(
            path, expected_shape=shape, acquired=acquired
        )
        full = metric_npz.audit_metric_npz(path, expected_shape=shape, acquired=acquired)
    assert identity.sha256 == full.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert identity.byte_count == path.stat().st_size
    assert identity.member_shapes == (shape, shape, ())
    assert 0 < full.stream_workspace_peak_bytes <= metric_npz.METRIC_NPZ_WRITE_STREAM_BYTES
    assert 0 < trace.maximum_input_chunk_bytes < inverse.nbytes
    np.testing.assert_array_equal(full.inverse_depth, inverse)
    np.testing.assert_array_equal(full.valid, valid)
    assert full.focal_x_normalized == np.float32(0.75)
    assert not full.inverse_depth.flags.writeable and not full.valid.flags.writeable
    inverse[0, 0] = 99
    assert full.inverse_depth[0, 0] == row[0]


@pytest.mark.parametrize(
    ("bad_value", "reason"),
    [
        ("nan-depth", "finite"),
        ("negative-zero", "positive zero"),
        ("no-depth", "must be positive"),
        ("negative-focal", "focal value"),
    ],
)
def test_legacy_archive_identity_is_not_proof_of_valid_metric_values(tmp_path, bad_value, reason):
    inverse, valid, focal = _metric_values()
    if bad_value == "nan-depth":
        inverse[0, 1] = np.nan
    elif bad_value == "negative-zero":
        inverse[0, 0] = -0.0
    elif bad_value == "no-depth":
        inverse[0, 1] = 0.0
    else:
        focal = np.float32(-0.5)
    path = tmp_path / "old-cache.npz"
    np.savez_compressed(path, inverse_depth=inverse, valid=valid, focal_x_normalized=focal)
    identity = metric_npz.audit_metric_npz_identity(path, expected_shape=(2, 3), job_root=tmp_path)
    with pytest.raises(metric_npz.MetricNpzContentError, match=reason):
        metric_npz.audit_metric_npz(path, expected_shape=(2, 3), job_root=tmp_path)
    assert identity.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("failure", ["memory", "wrong-depth-dtype", "readonly-mask"])
def test_array_allocation_failure_leaves_cached_depth_retryable(tmp_path, failure):
    path = tmp_path / "frame.npz"
    _write_owned(path)
    original = path.read_bytes()

    def unavailable(shape, dtype):
        if failure == "memory":
            raise MemoryError("worker has insufficient memory")
        if failure == "wrong-depth-dtype" and dtype == np.float32:
            return np.empty(shape, dtype=np.float64)
        result = np.empty(shape, dtype=dtype)
        if dtype == np.bool_:
            result.flags.writeable = False
        return result

    with pytest.raises(metric_npz.MetricNpzAllocationError):
        metric_npz.audit_metric_npz(path, allocator=unavailable)
    assert path.read_bytes() == original
    np.testing.assert_array_equal(metric_npz.audit_metric_npz(path).valid, _metric_values()[1])


@pytest.mark.parametrize(
    "invalid", ["dtype", "transposed", "mask-dtype", "mask-shape", "focal-type"]
)
def test_mismatched_frame_cannot_replace_the_existing_metric_cache(tmp_path, invalid):
    path = tmp_path / "known-good.npz"
    _write_owned(path)
    old = path.read_bytes()
    inverse, valid, focal = _metric_values()
    if invalid == "dtype":
        inverse = inverse.astype(np.float64)
    elif invalid == "transposed":
        inverse = inverse.T
    elif invalid == "mask-dtype":
        valid = valid.astype(np.uint8)
    elif invalid == "mask-shape":
        valid = valid[:1]
    else:
        focal = float(focal)
    with pytest.raises((TypeError, ValueError)):
        metric_npz.write_metric_npz(path, inverse, valid, focal)
    assert path.read_bytes() == old
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("checkpoint", ["start", "compressed-data", "central-record", "footer"])
def test_truncated_metric_archive_is_not_a_resume_source(tmp_path, checkpoint):
    target = tmp_path / "valid.npz"
    _write_owned(target)
    expected = target.read_bytes()
    identity = metric_npz.audit_metric_npz_identity(target)
    cutoff = {
        "start": 0,
        "compressed-data": identity.layout.members[0].data_offset + 1,
        "central-record": identity.layout.central_offset + 5,
        "footer": len(expected) - 3,
    }[checkpoint]
    partial = tmp_path / "incomplete.npz"
    partial.write_bytes(expected[:cutoff])
    for audit in (metric_npz.audit_metric_npz, metric_npz.audit_metric_npz_identity):
        with pytest.raises(metric_npz.MetricNpzFormatError):
            audit(partial, expected_shape=(2, 3))
    assert target.read_bytes() == expected
    assert partial.read_bytes() == expected[:cutoff]


def test_identity_audit_requires_the_original_job_and_both_array_shapes(tmp_path):
    from src.depth_surge_3d.io.nofollow_io import AcquiredRoot

    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    path = first / "metric.npz"
    inverse, valid, focal = _metric_values()
    np.savez_compressed(path, inverse_depth=inverse, valid=valid[:1], focal_x_normalized=focal)
    with AcquiredRoot(first) as acquired:
        with pytest.raises(metric_npz.MetricNpzFormatError, match="wrong job root"):
            metric_npz.audit_metric_npz_identity(path, job_root=second, acquired=acquired)
        with pytest.raises(metric_npz.MetricNpzFormatError, match="shapes differ"):
            metric_npz.audit_metric_npz_identity(path, job_root=first, acquired=acquired)
    _write_owned(path)
    with pytest.raises(metric_npz.MetricNpzFormatError, match="shape mismatch"):
        metric_npz.audit_metric_npz_identity(path, expected_shape=(3, 2), job_root=first)
    assert metric_npz.audit_metric_npz_identity(path, expected_shape=(2, 3)).member_shapes[0] == (
        2,
        3,
    )
