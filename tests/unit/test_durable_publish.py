from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.depth_surge_3d.io.durable_publish import (
    OwnedSiblingTemporary,
    OwnedTemporaryCollisionError,
    OwnedTemporaryIdentityError,
    publication_temporary_path,
)


@pytest.mark.parametrize("collision", ["regular", "hardlink", "symlink"])
def test_owned_temporary_rejects_a_preexisting_unowned_name(
    tmp_path: Path,
    collision: str,
) -> None:
    target = tmp_path / "target.json"
    temporary = publication_temporary_path(target)
    sentinel = tmp_path / "sentinel.bin"
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
        OwnedSiblingTemporary.create(target)

    assert sentinel.read_bytes() == b"must survive"
    assert temporary.exists() or temporary.is_symlink()


def test_owned_temporary_cleanup_preserves_a_name_swap_replacement(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    owned = OwnedSiblingTemporary.create(target)
    owned.handle.write(b"owned")
    owned.sync()
    owned.close()
    parked = tmp_path / "parked.tmp"
    owned.path.rename(parked)
    owned.path.write_bytes(b"replacement")

    with pytest.raises(OwnedTemporaryIdentityError, match="changed"):
        owned.cleanup()

    assert parked.read_bytes() == b"owned"
    assert owned.path.read_bytes() == b"replacement"


def test_owned_temporary_replaces_its_target_and_survives_post_replace_fault(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"old")
    owned = OwnedSiblingTemporary.create(target)
    owned.handle.write(b"new")
    owned.sync()

    with pytest.raises(RuntimeError, match="simulated crash"):
        owned.replace(after_replace=lambda: (_ for _ in ()).throw(RuntimeError("simulated crash")))
    owned.cleanup()

    assert target.read_bytes() == b"new"
    assert not owned.path.exists()


def test_owned_temporary_publishes_when_target_was_absent(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    owned = OwnedSiblingTemporary.create(target)
    owned.handle.write(b"new")
    owned.sync()

    owned.replace()
    owned.cleanup()

    assert target.read_bytes() == b"new"
    assert not owned.path.exists()
