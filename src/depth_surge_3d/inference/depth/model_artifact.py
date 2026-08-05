"""Resolve model repositories to immutable local artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def _hash_directory(directory: Path) -> str:
    hasher = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        hasher.update(path.relative_to(directory).as_posix().encode("utf-8"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                hasher.update(chunk)
    return hasher.hexdigest()


def resolve_hf_snapshot(repo_id: str, *, cache_dir: Path | str | None = None) -> tuple[str, str]:
    """Return an immutable local load path and content-bound artifact identity."""

    local_path = Path(repo_id).expanduser()
    if local_path.is_dir():
        resolved = local_path.resolve()
        return str(resolved), f"local:{_hash_directory(resolved)}"

    from huggingface_hub import snapshot_download

    kwargs: dict[str, Any] = {"repo_id": repo_id}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    try:
        snapshot = Path(snapshot_download(**kwargs)).resolve()
    except Exception as online_error:
        try:
            snapshot = Path(snapshot_download(**kwargs, local_files_only=True)).resolve()
        except Exception:
            raise online_error

    revision = snapshot.name if snapshot.parent.name == "snapshots" else _hash_directory(snapshot)
    return str(snapshot), f"hf:{repo_id}@{revision}"
