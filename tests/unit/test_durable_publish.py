"""Atomic same-directory file publication tests."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.depth_surge_3d.io.durable_publish import (
    OwnedSiblingTemporary,
    OwnedTemporaryCollisionError,
    atomic_write_bytes,
    atomic_write_json,
    durable_replace,
)


def test_atomic_write_bytes_and_json_replace_cleanly(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    atomic_write_json(target, {"step": 1, "ok": True}, sync=False)
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True, "step": 1}

    atomic_write_bytes(target, b'{"step":2}', sync=True)
    assert target.read_bytes() == b'{"step":2}'
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_owned_sibling_temporary_cleans_up_on_error(tmp_path: Path) -> None:
    target = tmp_path / "frame.png"
    owned = OwnedSiblingTemporary.create(target)
    temp_path = owned.path
    assert temp_path.exists()
    owned.handle.write(b"partial")
    owned.cleanup()
    assert not temp_path.exists()
    assert not target.exists()


def test_owned_sibling_temporary_collision_and_cross_dir_replace(tmp_path: Path) -> None:
    target = tmp_path / "out.bin"
    fixed_tmp = tmp_path / ".out.bin.tmp"
    fixed_tmp.write_bytes(b"existing")
    with pytest.raises(OwnedTemporaryCollisionError):
        OwnedSiblingTemporary.create(target, temporary=fixed_tmp)

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    with pytest.raises(ValueError, match="same-directory"):
        durable_replace(fixed_tmp, other_dir / "out.bin")
