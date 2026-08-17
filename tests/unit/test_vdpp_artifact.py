"""Pinned VDPP checkpoint resolution tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.depth_surge_3d.inference.depth import vdpp_artifact


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, size: int) -> bytes:
        chunk = self.payload[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


def _pin(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    monkeypatch.setattr(vdpp_artifact, "VDPP_CHECKPOINT_SIZE", len(payload))
    monkeypatch.setattr(
        vdpp_artifact,
        "VDPP_CHECKPOINT_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )


def test_existing_verified_checkpoint_never_opens_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"verified-weights"
    _pin(monkeypatch, payload)
    path = tmp_path / "VDPP" / "vdpp.pth"
    path.parent.mkdir()
    path.write_bytes(payload)

    resolved = vdpp_artifact.ensure_vdpp_checkpoint(
        tmp_path,
        opener=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("network should not be used")
        ),
    )

    assert resolved == path


def test_download_streams_reports_progress_and_atomically_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"checkpoint-payload"
    _pin(monkeypatch, payload)
    progress: list[tuple[int, int]] = []

    path = vdpp_artifact.ensure_vdpp_checkpoint(
        tmp_path,
        opener=lambda *_args, **_kwargs: _Response(payload),
        progress_callback=lambda current, total: progress.append((current, total)),
        chunk_size=4,
    )

    assert path.read_bytes() == payload
    assert not path.with_suffix(".pth.part").exists()
    assert progress[-1] == (len(payload), len(payload))


def test_wrong_download_digest_is_never_loaded_or_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = b"expected"
    _pin(monkeypatch, expected)
    wrong = b"wrong___"

    with pytest.raises(ValueError, match="SHA-256"):
        vdpp_artifact.ensure_vdpp_checkpoint(
            tmp_path,
            opener=lambda *_args, **_kwargs: _Response(wrong),
        )

    final = tmp_path / "VDPP" / "vdpp.pth"
    assert not final.exists()
    assert not final.with_suffix(".pth.part").exists()


def test_wrong_existing_file_is_replaced_only_after_verified_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"new-verified"
    _pin(monkeypatch, payload)
    path = tmp_path / "VDPP" / "vdpp.pth"
    path.parent.mkdir()
    path.write_bytes(b"old-corrupt")

    resolved = vdpp_artifact.ensure_vdpp_checkpoint(
        tmp_path,
        opener=lambda *_args, **_kwargs: _Response(payload),
    )

    assert resolved.read_bytes() == payload


def test_pinned_public_checkpoint_identity() -> None:
    assert vdpp_artifact.VDPP_UPSTREAM_RELEASE == "v1.0"
    assert vdpp_artifact.VDPP_UPSTREAM_REVISION == "73cc2b4dc6b3b5cfb2e37f51e452461e03fe26f5"
    assert vdpp_artifact.VDPP_CHECKPOINT_SIZE == 116485370
    assert (
        vdpp_artifact.VDPP_CHECKPOINT_SHA256
        == "7368315b126093f0335147f42a1920f255d529613bfffc5c6cf4ef832deb73a7"
    )
