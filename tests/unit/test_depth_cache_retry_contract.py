"""A bad optional depth cache must be a miss, not a processing failure."""

import json

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.utils.domain import depth_cache
from tests.unit.test_depth_cache import canonical_metadata, canonical_settings


@pytest.fixture
def depth_files(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source fingerprint")
    frame = tmp_path / "frame_000012.png"
    assert cv2.imwrite(str(frame), np.arange(16, dtype=np.uint16).reshape(4, 4))
    metadata = canonical_metadata([frame.name])
    config = canonical_settings()
    cached = depth_cache.get_cache_dir() / depth_cache.compute_cache_key(str(source), config)
    return str(source), frame, metadata, config, cached


@pytest.mark.parametrize("invalid", ["[]", "null", "1"])
@pytest.mark.parametrize("action", ["read", "save"])
def test_valid_json_with_wrong_document_type_never_breaks_a_cache_miss_or_save(
    depth_files, invalid, action
):
    source, frame, _, config, cached = depth_files
    cached.mkdir()
    (cached / "metadata.json").write_text(invalid)
    if action == "read":
        assert depth_cache.get_cached_depth_map_files(source, config, 1) is None
    else:
        (frame.parent / "metadata.json").write_text(invalid)
        assert depth_cache.save_depth_map_files_to_cache(source, config, [frame]) is False
    assert (cached / "metadata.json").read_text() == invalid


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("schema_version", 0),
        ("algorithm_version", "other"),
        ("representation", "inverse_depth"),
        ("near_value", 0.0),
        ("far_value", 1.0),
        ("encoding", "uint8_png"),
        ("encoding_scale", 255.0),
        ("num_frames", 2),
        ("frame_names", []),
        ("scene_manifest_fingerprint", None),
    ],
)
def test_self_consistent_but_incompatible_cache_metadata_is_never_reused(depth_files, field, bad):
    source, frame, metadata, config, cached = depth_files
    cached.mkdir()
    depth_path = cached / "depth_000000.png"
    depth_path.write_bytes(frame.read_bytes())
    changed = {**metadata, field: bad}
    if bad is None:
        del changed[field]
    changed.pop("fingerprint")
    changed["fingerprint"] = depth_cache.canonical_json_hash(changed)
    (cached / "metadata.json").write_text(json.dumps(changed))
    assert depth_cache.get_cached_depth_map_files(source, config, 1) is None
    (cached / "metadata.json").write_text(json.dumps(metadata))
    assert depth_cache.get_cached_depth_map_files(source, config, 1) == [depth_path]
    np.testing.assert_array_equal(
        cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED),
        np.arange(16, dtype=np.uint16).reshape(4, 4),
    )


def test_failed_cache_metadata_install_cleans_partial_marker_before_retry(monkeypatch, depth_files):
    from pathlib import Path

    source, frame, metadata, config, cached = depth_files
    manifest = frame.parent / "metadata.json"
    manifest.write_text("truncated")
    assert not depth_cache.save_depth_map_files_to_cache(source, config, [frame])
    manifest.write_text(json.dumps(metadata))
    assert not depth_cache.save_depth_map_files_to_cache(source, config, [])
    install = Path.replace
    with monkeypatch.context() as patch:

        def unavailable(self, target):
            if self == cached / "metadata.json.tmp":
                raise OSError("cannot publish metadata")
            return install(self, target)

        patch.setattr(Path, "replace", unavailable)
        assert not depth_cache.save_depth_map_files_to_cache(source, config, [frame])
    assert not (cached / "metadata.json.tmp").exists()
    assert depth_cache.get_cached_depth_map_files(source, config, 1) is None
    assert depth_cache.save_depth_map_files_to_cache(source, config, [frame])
    assert depth_cache.get_cached_depth_map_files(source, config, 1) == [
        cached / "depth_000000.png"
    ]
