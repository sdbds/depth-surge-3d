"""Unit tests for canonical depth cache utilities."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.depth_surge_3d.processing.frames.depth_storage import canonical_json_hash
from src.depth_surge_3d.utils.domain import depth_cache
from src.depth_surge_3d.utils.domain.depth_cache import (
    clear_cache,
    compute_cache_key,
    get_cache_dir,
    get_cache_size,
)


MODEL_FINGERPRINT = "model-fingerprint"


def canonical_metadata(frame_names: list[str]) -> dict:
    metadata = {
        "schema_version": 1,
        "algorithm_version": "scene-percentile-v1",
        "representation": "relative_disparity",
        "near_value": 1.0,
        "far_value": 0.0,
        "encoding": "uint16_png",
        "encoding_scale": 65535.0,
        "num_frames": len(frame_names),
        "frame_names": frame_names,
        "native_shape": [4, 4],
        "source_raw_fingerprint": "raw-fingerprint",
        "source_model_fingerprint": MODEL_FINGERPRINT,
        "scene_manifest_fingerprint": "scene-fingerprint",
        "depth_bounds_fingerprint": "bounds-fingerprint",
    }
    metadata["fingerprint"] = canonical_json_hash(metadata)
    return metadata


def canonical_settings(**overrides) -> dict:
    return {
        "depth_model_version": "v3",
        "model_fingerprint": MODEL_FINGERPRINT,
        **overrides,
    }


def create_cache_entry(cache_dir: Path, video_file: Path, settings: dict, metadata: dict):
    entry = cache_dir / compute_cache_key(str(video_file), settings)
    entry.mkdir()
    (entry / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for i in range(metadata.get("num_frames", 0)):
        (entry / f"depth_{i:06d}.png").write_bytes(b"png")
    return entry


class TestGetCacheDir:
    @patch.dict("os.environ", {"XDG_CACHE_HOME": "/tmp/xdg_cache"})
    def test_cache_dir_with_xdg(self):
        cache_dir = get_cache_dir()
        assert cache_dir == Path("/tmp/xdg_cache") / "depth-surge-3d" / "depth_cache"

    @patch.dict("os.environ", {}, clear=True)
    def test_cache_dir_without_xdg(self, tmp_path):
        with patch("pathlib.Path.home", return_value=tmp_path):
            cache_dir = get_cache_dir()
            assert cache_dir == tmp_path / ".cache" / "depth-surge-3d" / "depth_cache"
            assert cache_dir.exists()


class TestComputeCacheKey:
    def test_cache_key_is_stable_hex(self, tmp_path):
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test video content" * 1000)
        settings = canonical_settings(model_size="base", depth_resolution=518)

        first = compute_cache_key(str(video_file), settings)
        second = compute_cache_key(str(video_file), settings)

        assert first == second
        assert len(first) == 32
        assert all(character in "0123456789abcdef" for character in first)

    def test_cache_key_changes_with_video(self, tmp_path):
        video1 = tmp_path / "video1.mp4"
        video1.write_bytes(b"content1" * 1000)
        video2 = tmp_path / "video2.mp4"
        video2.write_bytes(b"content2" * 1000)

        assert compute_cache_key(str(video1), canonical_settings()) != compute_cache_key(
            str(video2), canonical_settings()
        )

    def test_cache_key_ignores_path_and_timestamp_for_identical_content(self, tmp_path):
        first = tmp_path / "first.mp4"
        second = tmp_path / "renamed.mp4"
        payload = b"same video content" * 1000
        first.write_bytes(payload)
        second.write_bytes(payload)
        stat = second.stat()
        os.utime(second, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

        settings = canonical_settings()
        assert compute_cache_key(str(first), settings) == compute_cache_key(str(second), settings)

    def test_cache_key_changes_with_inference_inputs(self, tmp_path):
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test content" * 1000)
        base = canonical_settings(
            model_path="24yearsold/seethroughv0.0.1_marigold",
            start_time="03:15",
            end_time="03:22",
            super_sample="none",
        )
        base_key = compute_cache_key(str(video_file), base)

        for change in (
            {"model_fingerprint": "another-model"},
            {"model_path": "24yearsold/a-different-checkpoint"},
            {"start_time": "10:00", "end_time": "10:07"},
            {"super_sample": "auto"},
        ):
            assert compute_cache_key(str(video_file), {**base, **change}) != base_key

    def test_cache_key_handles_large_file(self, tmp_path):
        video_file = tmp_path / "large.mp4"
        video_file.write_bytes(b"x" * (3 * 1024 * 1024))

        key = compute_cache_key(str(video_file), canonical_settings())

        assert len(key) == 32


class TestGetCachedDepthMapFiles:
    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_cache_miss_without_metadata(self, mock_cache_dir, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test")

        assert (
            depth_cache.get_cached_depth_map_files(
                str(video_file), canonical_settings(), num_frames=1
            )
            is None
        )

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_valid_canonical_cache_returns_paths_without_decoding(self, mock_cache_dir, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test")
        settings = canonical_settings()
        entry = create_cache_entry(
            cache_dir,
            video_file,
            settings,
            canonical_metadata(["frame_000000.png", "frame_000001.png"]),
        )

        result = depth_cache.get_cached_depth_map_files(str(video_file), settings, 2)

        assert result == [entry / "depth_000000.png", entry / "depth_000001.png"]

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_lookup_rejects_missing_or_mismatched_model_fingerprint(self, mock_cache_dir, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test")

        missing = {"depth_model_version": "v3"}
        create_cache_entry(cache_dir, video_file, missing, canonical_metadata(["frame_000000.png"]))
        mismatch = canonical_settings(model_fingerprint="different-model")
        create_cache_entry(
            cache_dir, video_file, mismatch, canonical_metadata(["frame_000000.png"])
        )

        assert depth_cache.get_cached_depth_map_files(str(video_file), missing, 1) is None
        assert depth_cache.get_cached_depth_map_files(str(video_file), mismatch, 1) is None

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_lookup_rejects_invalid_metadata_fingerprint(self, mock_cache_dir, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test")
        settings = canonical_settings()
        metadata = canonical_metadata(["frame_000000.png"])
        metadata["source_raw_fingerprint"] = "tampered"
        create_cache_entry(cache_dir, video_file, settings, metadata)

        assert depth_cache.get_cached_depth_map_files(str(video_file), settings, 1) is None

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_lookup_rejects_previous_algorithm(self, mock_cache_dir, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test")
        settings = canonical_settings()
        metadata = canonical_metadata(["frame_000000.png"])
        metadata["algorithm_version"] = "scene-percentile-v0"
        metadata.pop("fingerprint")
        metadata["fingerprint"] = canonical_json_hash(metadata)
        create_cache_entry(cache_dir, video_file, settings, metadata)

        assert depth_cache.get_cached_depth_map_files(str(video_file), settings, 1) is None

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_lookup_rejects_missing_depth_file(self, mock_cache_dir, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test")
        settings = canonical_settings()
        entry = create_cache_entry(
            cache_dir,
            video_file,
            settings,
            canonical_metadata(["frame_000000.png", "frame_000001.png"]),
        )
        (entry / "depth_000001.png").unlink()

        assert depth_cache.get_cached_depth_map_files(str(video_file), settings, 2) is None

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_lookup_rejects_invalid_json(self, mock_cache_dir, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test")
        settings = canonical_settings()
        entry = cache_dir / compute_cache_key(str(video_file), settings)
        entry.mkdir()
        (entry / "metadata.json").write_text("{ invalid json", encoding="utf-8")

        assert depth_cache.get_cached_depth_map_files(str(video_file), settings, 1) is None


class TestSaveDepthMapFiles:
    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_save_copies_canonical_files_and_exact_metadata(self, mock_cache_dir, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test")
        settings = canonical_settings()
        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()
        depth_files = []
        for i in range(2):
            path = depth_dir / f"frame_{i:06d}.png"
            assert cv2.imwrite(str(path), np.full((4, 4), i * 65535, dtype=np.uint16))
            depth_files.append(path)
        metadata = canonical_metadata([path.name for path in depth_files])
        (depth_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

        assert depth_cache.save_depth_map_files_to_cache(str(video_file), settings, depth_files)

        entry = cache_dir / compute_cache_key(str(video_file), settings)
        assert (entry / "depth_000000.png").exists()
        assert (entry / "depth_000001.png").exists()
        assert json.loads((entry / "metadata.json").read_text(encoding="utf-8")) == metadata

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_save_rejects_missing_or_mismatched_model_fingerprint(self, mock_cache_dir, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test")
        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()
        depth_file = depth_dir / "frame_000000.png"
        assert cv2.imwrite(str(depth_file), np.zeros((4, 4), dtype=np.uint16))
        (depth_dir / "metadata.json").write_text(
            json.dumps(canonical_metadata([depth_file.name])), encoding="utf-8"
        )

        assert not depth_cache.save_depth_map_files_to_cache(str(video_file), {}, [depth_file])
        assert not depth_cache.save_depth_map_files_to_cache(
            str(video_file), canonical_settings(model_fingerprint="different-model"), [depth_file]
        )

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_save_rejects_frame_names_that_do_not_match_sources(self, mock_cache_dir, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test")
        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()
        depth_file = depth_dir / "frame_000000.png"
        assert cv2.imwrite(str(depth_file), np.zeros((4, 4), dtype=np.uint16))
        (depth_dir / "metadata.json").write_text(
            json.dumps(canonical_metadata(["another_frame.png"])), encoding="utf-8"
        )

        assert not depth_cache.save_depth_map_files_to_cache(
            str(video_file), canonical_settings(), [depth_file]
        )


class TestCacheMaintenance:
    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_clear_cache_counts_removed_entries(self, mock_cache_dir, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir
        for name in ("entry1", "entry2"):
            entry = cache_dir / name
            entry.mkdir()
            (entry / "file.txt").write_text("test", encoding="utf-8")

        assert clear_cache() == 2
        assert not any(cache_dir.iterdir())

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_clear_cache_ignores_failed_entry(self, mock_cache_dir, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir
        (cache_dir / "entry1").mkdir()
        (cache_dir / "entry2").mkdir()

        with patch("shutil.rmtree", side_effect=[OSError("failed"), None]):
            assert clear_cache() == 1

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_cache_size_reports_entries_and_bytes(self, mock_cache_dir, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir
        entry1 = cache_dir / "entry1"
        entry1.mkdir()
        (entry1 / "one.bin").write_bytes(b"x" * 100)
        entry2 = cache_dir / "entry2"
        entry2.mkdir()
        (entry2 / "two.bin").write_bytes(b"x" * 300)

        assert get_cache_size() == (2, 400)

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_cache_size_for_missing_directory_is_zero(self, mock_cache_dir, tmp_path):
        mock_cache_dir.return_value = tmp_path / "cache"

        assert get_cache_size() == (0, 0)
