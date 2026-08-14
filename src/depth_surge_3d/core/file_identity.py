"""Bounded-cost content identity for large source files."""

from __future__ import annotations

import hashlib
from pathlib import Path


FILE_IDENTITY_ALGORITHM_VERSION = "file-sample-blake2b-v1"
FILE_IDENTITY_SAMPLE_SIZE = 1024 * 1024


def file_sample_fingerprint(path: Path | str) -> str:
    """Hash at most the first and last MiB, independent of name and timestamps."""

    candidate = Path(path)
    file_size = candidate.stat().st_size
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(file_size.to_bytes(16, byteorder="big", signed=False))
    with candidate.open("rb") as handle:
        if file_size <= FILE_IDENTITY_SAMPLE_SIZE * 2:
            hasher.update(handle.read())
        else:
            hasher.update(handle.read(FILE_IDENTITY_SAMPLE_SIZE))
            handle.seek(-FILE_IDENTITY_SAMPLE_SIZE, 2)
            hasher.update(handle.read(FILE_IDENTITY_SAMPLE_SIZE))
    return hasher.hexdigest()
