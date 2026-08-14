"""Performance-first source frame identity tests."""

from __future__ import annotations

import os
from pathlib import Path

from src.depth_surge_3d.core import file_identity
from src.depth_surge_3d.processing.frames.source_frame_manifest import (
    SOURCE_FRAME_ALGORITHM_VERSION,
    SOURCE_FRAME_SCHEMA_VERSION,
    frame_sequence_fingerprint,
)


def test_frame_fingerprint_uses_filesystem_metadata_not_payload_reads(
    tmp_path, monkeypatch
) -> None:
    frame = tmp_path / "frame_000001.png"
    frame.write_bytes(b"encoded frame payload")

    def fail_open(self: Path, *args, **kwargs):
        raise AssertionError(f"fingerprinting read payload bytes from {self}")

    monkeypatch.setattr(Path, "open", fail_open)

    assert frame_sequence_fingerprint([frame])


def test_source_frame_contract_marks_the_lightweight_algorithm() -> None:
    assert SOURCE_FRAME_SCHEMA_VERSION == 2
    assert SOURCE_FRAME_ALGORITHM_VERSION == "png-stat-v1"


def test_source_video_identity_is_sampled_content_not_path_metadata(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"a" * (3 * 1024 * 1024))
    fingerprint = getattr(file_identity, "file_sample_fingerprint", None)

    assert callable(fingerprint)
    assert file_identity.FILE_IDENTITY_ALGORITHM_VERSION == "file-sample-blake2b-v1"
    original = fingerprint(source)

    renamed = source.with_name("renamed-source.mp4")
    source.rename(renamed)
    stat = renamed.stat()
    os.utime(renamed, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert fingerprint(renamed) == original

    preserved = renamed.stat()
    with renamed.open("r+b") as handle:
        handle.write(b"b")
    os.utime(renamed, ns=(preserved.st_atime_ns, preserved.st_mtime_ns))
    assert fingerprint(renamed) != original
