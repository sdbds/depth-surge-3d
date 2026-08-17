"""Immutable model artifact resolution tests."""

from pathlib import Path
from unittest.mock import MagicMock

from src.depth_surge_3d.inference.depth.model_artifact import resolve_hf_snapshot


def test_remote_model_identity_uses_resolved_snapshot_revision(tmp_path, monkeypatch):
    revision = "a" * 40
    snapshot = tmp_path / "hub" / "models--owner--model" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    download = MagicMock(return_value=str(snapshot))
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)

    load_path, identity = resolve_hf_snapshot("owner/model", cache_dir=tmp_path / "cache")

    assert load_path == str(snapshot.resolve())
    assert identity == f"hf:owner/model@{revision}"
    download.assert_called_once_with(
        repo_id="owner/model",
        cache_dir=str(tmp_path / "cache"),
    )


def test_resolve_hf_snapshot_forwards_immutable_revision(monkeypatch, tmp_path):
    snapshot = tmp_path / "models--owner--repo" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)

    path, identity = resolve_hf_snapshot("owner/repo", revision="pinned123")

    assert Path(path) == snapshot.resolve()
    assert calls == [{"repo_id": "owner/repo", "revision": "pinned123"}]
    assert identity == "hf:owner/repo@abc123"


def test_resolve_hf_snapshot_keeps_revision_during_offline_retry(monkeypatch, tmp_path):
    snapshot = tmp_path / "models--owner--repo" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        if not kwargs.get("local_files_only"):
            raise ConnectionError("offline")
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    resolve_hf_snapshot("owner/repo", revision="pinned123")

    assert calls[1] == {
        "repo_id": "owner/repo",
        "revision": "pinned123",
        "local_files_only": True,
    }


def test_local_model_identity_changes_with_artifact_bytes(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weights = model_dir / "model.safetensors"
    weights.write_bytes(b"first")

    first_path, first_identity = resolve_hf_snapshot(str(model_dir))
    weights.write_bytes(b"second")
    second_path, second_identity = resolve_hf_snapshot(str(model_dir))

    assert first_path == second_path == str(model_dir.resolve())
    assert first_identity.startswith("local:")
    assert first_identity != second_identity
