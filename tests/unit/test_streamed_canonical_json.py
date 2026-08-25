"""Tests for canonical JSON with one replayable variable-cardinality array."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from src.depth_surge_3d.io.durable_publish import (
    OwnedTemporaryCollisionError,
    OwnedTemporaryIdentityError,
    publication_temporary_path,
)
from src.depth_surge_3d.io.streamed_canonical_json import (
    ReplayableJsonArray,
    build_streamed_canonical_object,
    build_streamed_canonical_object_arrays,
    build_streamed_canonical_value,
)


def test_streamed_canonical_object_matches_regular_encoder_without_a_resident_list(
    tmp_path: Path,
) -> None:
    replays = 0

    def items():
        nonlocal replays
        replays += 1
        for index in range(3):
            yield f"frame_{index + 1:06d}.png"

    streamed = build_streamed_canonical_object(
        static_fields={"algorithm_version": "test-v1", "schema_version": 1},
        array_key="frame_names",
        items=items,
        fingerprint_key="fingerprint",
    )
    path = tmp_path / "metadata.json"
    streamed.publish(path)

    without_fingerprint = {
        "algorithm_version": "test-v1",
        "frame_names": [
            "frame_000001.png",
            "frame_000002.png",
            "frame_000003.png",
        ],
        "schema_version": 1,
    }
    expected_fingerprint = hashlib.sha256(
        json.dumps(
            without_fingerprint,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    expected = dict(without_fingerprint, fingerprint=expected_fingerprint)
    expected_raw = json.dumps(
        expected,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")

    assert path.read_bytes() == expected_raw
    assert streamed.fingerprint == expected_fingerprint
    assert streamed.raw_sha256 == hashlib.sha256(expected_raw).hexdigest()
    assert streamed.raw_byte_count == len(expected_raw)
    assert streamed.matches(path)
    assert replays >= 3


def test_streamed_match_rejects_equal_bytes_through_a_link(tmp_path: Path) -> None:
    from src.depth_surge_3d.io.nofollow_io import AcquiredRoot

    root = tmp_path / "job"
    root.mkdir()
    external = tmp_path / "external.json"
    document = build_streamed_canonical_value({"schema_version": 1})
    external.write_bytes(b"".join(document.chunks()))
    linked = root / "metadata.json"
    try:
        os.symlink(external, linked)
    except OSError as error:
        pytest.skip(f"file symlink creation is unavailable: {error}")

    with AcquiredRoot(root) as acquired:
        assert not document.matches(linked, acquired=acquired)


def test_multiple_streamed_arrays_preserve_sorted_canonical_key_order(tmp_path: Path) -> None:
    streamed = build_streamed_canonical_object_arrays(
        static_fields={"middle": 1},
        array_items={
            "alpha": lambda: iter(("a", "b")),
            "zulu": lambda: iter((1, 2)),
        },
    )
    path = tmp_path / "multiple.json"
    streamed.publish(path)

    assert path.read_bytes() == b'{"alpha":["a","b"],"middle":1,"zulu":[1,2]}'


def test_nested_replayable_array_is_streamed_in_canonical_object(tmp_path: Path) -> None:
    streamed = build_streamed_canonical_value(
        {
            "metric": {"fractions": ReplayableJsonArray(lambda: iter((0.0, 0.5)))},
            "names": ReplayableJsonArray(lambda: iter(("a", "b"))),
        }
    )
    path = tmp_path / "nested.json"
    streamed.publish(path)

    assert path.read_bytes() == b'{"metric":{"fractions":[0.0,0.5]},"names":["a","b"]}'


def test_replayable_array_matches_legacy_pretty_json_bytes(tmp_path: Path) -> None:
    from src.depth_surge_3d.io.streamed_canonical_json import (
        build_streamed_pretty_json_value,
    )

    payload = {
        "frame_names": ["frame_000000.png", "frame_000001.png"],
        "nested": {"enabled": True, "values": [1, 2]},
        "schema_version": 1,
    }
    streamed_value = dict(payload)
    streamed_value["frame_names"] = ReplayableJsonArray(
        lambda: iter(payload["frame_names"])
    )
    document = build_streamed_pretty_json_value(streamed_value)
    path = tmp_path / "metadata.json"

    document.publish(path)

    assert path.read_bytes() == json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")


def test_streamed_publish_preserves_a_preexisting_deterministic_temporary(
    tmp_path: Path,
) -> None:
    document = build_streamed_canonical_value({"schema_version": 1})
    path = tmp_path / "metadata.json"
    temporary = publication_temporary_path(path)
    temporary.write_bytes(b"unowned")

    with pytest.raises(OwnedTemporaryCollisionError, match="already exists"):
        document.publish(path)

    assert temporary.read_bytes() == b"unowned"
    assert not path.exists()


def test_streamed_publish_preserves_a_post_sync_name_replacement(tmp_path: Path) -> None:
    document = build_streamed_canonical_value({"schema_version": 1})
    path = tmp_path / "metadata.json"
    temporary = publication_temporary_path(path)
    parked = tmp_path / "parked.tmp"

    def replace_temporary(stage: str) -> None:
        if stage == "after_canonical_json_temporary_sync":
            temporary.rename(parked)
            temporary.write_bytes(b"replacement")
            raise RuntimeError("simulated crash")

    with pytest.raises(OwnedTemporaryIdentityError, match="changed"):
        document.publish(path, fault=replace_temporary)

    assert parked.is_file()
    assert temporary.read_bytes() == b"replacement"
    assert not path.exists()
