"""Portable containment and no-symlink filesystem I/O tests."""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from src.depth_surge_3d.io.file_identity import (
    capture_directory_identity,
    capture_file_identity,
)
from src.depth_surge_3d.io.nofollow_io import (
    AcquiredRoot,
    NoFollowPathError,
    SafeTreeMutation,
    open_regular_beneath,
)


def test_acquired_root_reads_writes_and_lists_regular_files(tmp_path: Path) -> None:
    sub = tmp_path / "stage"
    sub.mkdir()
    target = sub / "item.bin"

    with AcquiredRoot(tmp_path) as acquired:
        handle, created_id = acquired.create_regular_exclusive(target)
        with handle:
            handle.write(b"hello-world")
        assert acquired.regular_identity(target) is not None
        assert acquired.regular_identity(sub / "missing.bin", missing_ok=True) is None
        with acquired.open_regular(target) as reader:
            assert reader.read() == b"hello-world"
        entries = list(acquired.iter_directory(sub))
        assert [e.name for e in entries] == ["item.bin"]
        assert entries[0].is_regular is True
        assert acquired.unlink_owned_regular(target, expected_identity=created_id) is True
        assert not target.exists()


def test_acquired_root_rejects_escape_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "job"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_bytes(b"outside")

    with AcquiredRoot(root) as acquired:
        with pytest.raises(NoFollowPathError):
            acquired.open_regular(outside)
        with pytest.raises(NoFollowPathError):
            acquired.unlink_owned_regular(outside)


def test_open_regular_beneath_rejects_symlink_when_supported(tmp_path: Path) -> None:
    root = tmp_path / "job"
    root.mkdir()
    target = root / "real.txt"
    target.write_bytes(b"real")
    link = root / "alias.txt"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable in this environment")

    with pytest.raises(NoFollowPathError):
        open_regular_beneath(root, link)


def test_file_and_directory_identity_roundtrip(tmp_path: Path) -> None:
    f = tmp_path / "data.bin"
    f.write_bytes(b"123456")
    fid = capture_file_identity(f)
    did = capture_directory_identity(tmp_path)
    assert fid.size == 6
    assert fid.from_object(fid.to_object()) == fid
    assert did.from_object(did.to_object()) == did

    mut = SafeTreeMutation(tmp_path)
    nested = mut.ensure_dir(tmp_path / "a" / "b")
    assert nested.is_dir()
    assert mut.unlink_regular(f) is True
    assert not f.exists()
