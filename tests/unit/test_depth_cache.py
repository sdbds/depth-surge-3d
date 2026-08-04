"""
Unit tests for depth cache utilities.
"""

import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.depth_surge_3d.processing.frames.depth_storage import canonical_json_hash
from src.depth_surge_3d.utils.domain import depth_cache

from src.depth_surge_3d.utils.domain.depth_cache import (
    get_cache_dir,
    compute_cache_key,
    get_cached_depth_maps,
    save_depth_maps_to_cache,
    clear_cache,
    get_cache_size,
)


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
        "source_model_fingerprint": "model-fingerprint",
        "scene_manifest_fingerprint": "scene-fingerprint",
        "depth_bounds_fingerprint": "bounds-fingerprint",
    }
    metadata["fingerprint"] = canonical_json_hash(metadata)
    return metadata


class TestGetCacheDir:
    """Test get_cache_dir function."""

    @patch.dict("os.environ", {"XDG_CACHE_HOME": "/tmp/xdg_cache"})
    def test_cache_dir_with_xdg(self):
        """Test cache dir uses XDG_CACHE_HOME when set."""
        cache_dir = get_cache_dir()
        assert cache_dir == Path("/tmp/xdg_cache") / "depth-surge-3d" / "depth_cache"
        assert "depth-surge-3d" in str(cache_dir)
        assert "depth_cache" in str(cache_dir)

    @patch.dict("os.environ", {}, clear=True)
    def test_cache_dir_without_xdg(self, tmp_path):
        """Test cache dir uses ~/.cache when XDG_CACHE_HOME not set."""
        # Use tmp_path to avoid permission issues with mocking
        with patch("pathlib.Path.home", return_value=tmp_path):
            cache_dir = get_cache_dir()
            expected = tmp_path / ".cache" / "depth-surge-3d" / "depth_cache"
            assert cache_dir == expected
            assert cache_dir.exists()  # Should be created


class TestComputeCacheKey:
    """Test compute_cache_key function."""

    def test_cache_key_format(self, tmp_path):
        """Test cache key is 32 hex characters."""
        # Create test video file
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test video content" * 1000)

        settings = {
            "depth_model_version": "v3",
            "model_size": "base",
            "depth_resolution": 518,
        }

        cache_key = compute_cache_key(str(video_file), settings)
        assert len(cache_key) == 32
        assert all(c in "0123456789abcdef" for c in cache_key)

    def test_cache_key_different_videos(self, tmp_path):
        """Test different videos produce different cache keys."""
        video1 = tmp_path / "video1.mp4"
        video1.write_bytes(b"content1" * 1000)

        video2 = tmp_path / "video2.mp4"
        video2.write_bytes(b"content2" * 1000)

        settings = {"depth_model_version": "v3"}

        key1 = compute_cache_key(str(video1), settings)
        key2 = compute_cache_key(str(video2), settings)

        assert key1 != key2

    def test_cache_key_different_settings(self, tmp_path):
        """Test different settings produce different cache keys."""
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test content" * 1000)

        settings1 = {"depth_model_version": "v3", "model_size": "small"}
        settings2 = {"depth_model_version": "v3", "model_size": "large"}

        key1 = compute_cache_key(str(video_file), settings1)
        key2 = compute_cache_key(str(video_file), settings2)

        assert key1 != key2

    def test_cache_key_same_video_same_settings(self, tmp_path):
        """Test same video and settings produce same cache key."""
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test content" * 1000)

        settings = {"depth_model_version": "v3", "model_size": "base"}

        key1 = compute_cache_key(str(video_file), settings)
        key2 = compute_cache_key(str(video_file), settings)

        assert key1 == key2

    def test_cache_key_changes_for_inference_source_and_frame_range(self, tmp_path):
        """A cache entry cannot be reused for another checkpoint, clip, or input scale."""
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test content" * 1000)
        base_settings = {
            "depth_model_version": "see_through",
            "model_path": "24yearsold/seethroughv0.0.1_marigold",
            "start_time": "03:15",
            "end_time": "03:22",
            "super_sample": "none",
        }
        base_key = compute_cache_key(str(video_file), base_settings)

        changes = (
            {"model_path": "24yearsold/a-different-checkpoint"},
            {"start_time": "10:00", "end_time": "10:07"},
            {"super_sample": "auto"},
        )
        for change in changes:
            changed_settings = {**base_settings, **change}
            assert compute_cache_key(str(video_file), changed_settings) != base_key


class TestGetCachedDepthMaps:
    """Test get_cached_depth_maps function."""

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_cache_miss_no_directory(self, mock_cache_dir, tmp_path):
        """Test cache miss when directory doesn't exist."""
        mock_cache_dir.return_value = tmp_path / "cache"

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test")

        settings = {"depth_model_version": "v3"}
        result = get_cached_depth_maps(str(video_file), settings, 10)

        assert result is None

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_cache_miss_no_metadata(self, mock_cache_dir, tmp_path):
        """Test cache miss when metadata file missing."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test")

        settings = {"depth_model_version": "v3"}
        cache_key = compute_cache_key(str(video_file), settings)

        # Create cache entry dir but no metadata
        (cache_dir / cache_key).mkdir()

        result = get_cached_depth_maps(str(video_file), settings, 10)

        assert result is None

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_cache_miss_frame_count_mismatch(self, mock_cache_dir, tmp_path):
        """Test cache miss when frame count doesn't match."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test")

        settings = {"depth_model_version": "v3"}
        cache_key = compute_cache_key(str(video_file), settings)

        cache_entry = cache_dir / cache_key
        cache_entry.mkdir()

        # Create metadata with different frame count
        metadata = {"num_frames": 5}
        with open(cache_entry / "metadata.json", "w") as f:
            json.dump(metadata, f)

        result = get_cached_depth_maps(str(video_file), settings, 10)

        assert result is None

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_cache_hit_success(self, mock_cache_dir, tmp_path):
        """Test successful cache hit."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test")

        settings = {"depth_model_version": "v3"}
        cache_key = compute_cache_key(str(video_file), settings)

        cache_entry = cache_dir / cache_key
        cache_entry.mkdir()

        # Create metadata
        num_frames = 3
        metadata = {"num_frames": num_frames}
        with open(cache_entry / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Create depth map files
        expected_depths = []
        for i in range(num_frames):
            depth_data = np.random.default_rng(i).random((100, 100)).astype(np.float32)
            depth_uint16 = (depth_data * 1000.0).astype(np.uint16)
            cv2.imwrite(str(cache_entry / f"depth_{i:06d}.png"), depth_uint16)
            expected_depths.append(depth_uint16.astype(np.float32) / 1000.0)

        result = get_cached_depth_maps(str(video_file), settings, num_frames)

        assert result is not None
        assert len(result) == num_frames
        assert result.shape == (num_frames, 100, 100)
        np.testing.assert_allclose(result, np.asarray(expected_depths), atol=0.0)

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_cached_file_lookup_does_not_decode_all_depth_maps(self, mock_cache_dir, tmp_path):
        """File-based callers can reuse a cache without allocating an array stack."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test")
        settings = {"depth_model_version": "v3"}
        cache_entry = cache_dir / compute_cache_key(str(video_file), settings)
        cache_entry.mkdir()
        (cache_entry / "metadata.json").write_text(
            json.dumps(canonical_metadata(["frame_000000.png", "frame_000001.png"]))
        )
        expected = [cache_entry / f"depth_{i:06d}.png" for i in range(2)]
        for path in expected:
            path.write_bytes(b"png")

        with patch("src.depth_surge_3d.utils.domain.depth_cache.cv2.imread") as imread:
            result = depth_cache.get_cached_depth_map_files(str(video_file), settings, 2)

        assert result == expected
        imread.assert_not_called()

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_cached_file_lookup_rejects_invalid_canonical_fingerprint(
        self, mock_cache_dir, tmp_path
    ):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test")
        settings = {"depth_model_version": "v3"}
        cache_entry = cache_dir / compute_cache_key(str(video_file), settings)
        cache_entry.mkdir()
        metadata = canonical_metadata(["frame_000000.png"])
        metadata["source_raw_fingerprint"] = "tampered"
        (cache_entry / "metadata.json").write_text(json.dumps(metadata))
        (cache_entry / "depth_000000.png").write_bytes(b"png")

        assert depth_cache.get_cached_depth_map_files(str(video_file), settings, 1) is None

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_cached_file_lookup_rejects_model_fingerprint_mismatch(self, mock_cache_dir, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test")
        settings = {"depth_model_version": "v3"}
        cache_entry = cache_dir / compute_cache_key(str(video_file), settings)
        cache_entry.mkdir()
        metadata = canonical_metadata(["frame_000000.png"])
        (cache_entry / "metadata.json").write_text(json.dumps(metadata))
        (cache_entry / "depth_000000.png").write_bytes(b"png")

        result = depth_cache.get_cached_depth_map_files(
            str(video_file),
            settings,
            1,
            expected_model_fingerprint="different-model",
        )

        assert result is None

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_cached_file_lookup_rejects_previous_canonical_algorithm(
        self, mock_cache_dir, tmp_path
    ):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test")
        settings = {"depth_model_version": "v3"}
        cache_entry = cache_dir / compute_cache_key(str(video_file), settings)
        cache_entry.mkdir()
        metadata = canonical_metadata(["frame_000000.png"])
        metadata["algorithm_version"] = "scene-percentile-v0"
        metadata.pop("fingerprint")
        metadata["fingerprint"] = canonical_json_hash(metadata)
        (cache_entry / "metadata.json").write_text(json.dumps(metadata))
        (cache_entry / "depth_000000.png").write_bytes(b"png")

        assert depth_cache.get_cached_depth_map_files(str(video_file), settings, 1) is None


class TestSaveDepthMapsToCache:
    """Test save_depth_maps_to_cache function."""

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_save_success(self, mock_cache_dir, tmp_path):
        """Test successful save to cache."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test content" * 1000)

        settings = {
            "depth_model_version": "v3",
            "model_size": "base",
            "depth_resolution": 518,
        }

        depth_maps = np.random.rand(3, 100, 100).astype(np.float32)

        success = save_depth_maps_to_cache(str(video_file), settings, depth_maps)

        assert success is True

        # Verify files were created
        cache_key = compute_cache_key(str(video_file), settings)
        cache_entry = cache_dir / cache_key

        assert cache_entry.exists()
        assert (cache_entry / "metadata.json").exists()
        assert (cache_entry / "depth_000000.png").exists()
        assert (cache_entry / "depth_000001.png").exists()
        assert (cache_entry / "depth_000002.png").exists()

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_save_metadata_content(self, mock_cache_dir, tmp_path):
        """Test metadata content is correct."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test")

        settings = {
            "depth_model_version": "v3",
            "model_size": "base",
            "depth_resolution": 518,
            "use_metric_depth": False,
        }

        depth_maps = np.random.rand(2, 50, 50).astype(np.float32)

        save_depth_maps_to_cache(str(video_file), settings, depth_maps)

        cache_key = compute_cache_key(str(video_file), settings)
        metadata_file = cache_dir / cache_key / "metadata.json"

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        assert metadata["num_frames"] == 2
        assert metadata["cache_version"] == "1.0"
        assert metadata["depth_settings"]["depth_model_version"] == "v3"
        assert metadata["depth_settings"]["model_size"] == "base"

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_save_depth_map_files_streams_pngs_to_cache(self, mock_cache_dir, tmp_path):
        """Disk-backed depth maps are cached one file at a time."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test")
        settings = {"depth_model_version": "v3"}
        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()
        depth_files = []
        for i in range(2):
            depth_file = depth_dir / f"frame_{i:06d}.png"
            cv2.imwrite(
                str(depth_file),
                np.full((4, 4), i * 65535, dtype=np.uint16),
            )
            depth_files.append(depth_file)
        source_metadata = canonical_metadata([path.name for path in depth_files])
        (depth_dir / "metadata.json").write_text(json.dumps(source_metadata))

        assert depth_cache.save_depth_map_files_to_cache(str(video_file), settings, depth_files)

        cache_entry = cache_dir / compute_cache_key(str(video_file), settings)
        cached_files = [cache_entry / f"depth_{i:06d}.png" for i in range(2)]
        assert all(path.exists() for path in cached_files)
        metadata = json.loads((cache_entry / "metadata.json").read_text())
        assert metadata == source_metadata


class TestClearCache:
    """Test clear_cache function."""

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_clear_empty_cache(self, mock_cache_dir, tmp_path):
        """Test clearing empty cache."""
        cache_dir = tmp_path / "cache"
        mock_cache_dir.return_value = cache_dir

        count = clear_cache()

        assert count == 0

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_clear_cache_with_entries(self, mock_cache_dir, tmp_path):
        """Test clearing cache with entries."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir

        # Create some cache entries
        (cache_dir / "entry1").mkdir()
        (cache_dir / "entry1" / "file.txt").write_text("test")
        (cache_dir / "entry2").mkdir()
        (cache_dir / "entry2" / "file.txt").write_text("test")

        count = clear_cache()

        assert count == 2
        assert not (cache_dir / "entry1").exists()
        assert not (cache_dir / "entry2").exists()


class TestGetCacheSize:
    """Test get_cache_size function."""

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_cache_size_empty(self, mock_cache_dir, tmp_path):
        """Test cache size for empty cache."""
        cache_dir = tmp_path / "cache"
        mock_cache_dir.return_value = cache_dir

        num_entries, total_size = get_cache_size()

        assert num_entries == 0
        assert total_size == 0

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_cache_size_with_entries(self, mock_cache_dir, tmp_path):
        """Test cache size with entries."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir

        # Create cache entries
        entry1 = cache_dir / "entry1"
        entry1.mkdir()
        (entry1 / "file1.txt").write_bytes(b"x" * 100)
        (entry1 / "file2.txt").write_bytes(b"x" * 200)

        entry2 = cache_dir / "entry2"
        entry2.mkdir()
        (entry2 / "file3.txt").write_bytes(b"x" * 300)

        num_entries, total_size = get_cache_size()

        assert num_entries == 2
        assert total_size == 600


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_cache_key_large_file(self, tmp_path):
        """Test cache key computation with large file (>2MB)."""
        # Create a file larger than 2MB to trigger seek from end
        video_file = tmp_path / "large.mp4"
        video_file.write_bytes(b"x" * (3 * 1024 * 1024))  # 3MB

        settings = {"depth_model_version": "v3"}
        cache_key = compute_cache_key(str(video_file), settings)

        # Should still produce valid 32-char hex key
        assert len(cache_key) == 32
        assert all(c in "0123456789abcdef" for c in cache_key)

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_cache_hit_missing_depth_file(self, mock_cache_dir, tmp_path):
        """Test cache miss when depth file is missing."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test")

        settings = {"depth_model_version": "v3"}
        cache_key = compute_cache_key(str(video_file), settings)

        cache_entry = cache_dir / cache_key
        cache_entry.mkdir()

        # Create metadata
        num_frames = 3
        metadata = {"num_frames": num_frames}
        with open(cache_entry / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Create only first 2 depth files (missing 3rd)
        for i in range(2):
            depth_data = np.random.rand(100, 100).astype(np.float32)
            depth_uint16 = (depth_data * 1000.0).astype(np.uint16)
            cv2.imwrite(str(cache_entry / f"depth_{i:06d}.png"), depth_uint16)

        result = get_cached_depth_maps(str(video_file), settings, num_frames)

        # Should return None (cache miss)
        assert result is None

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_cache_hit_corrupted_depth_file(self, mock_cache_dir, tmp_path):
        """Test cache miss when depth file is corrupted."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test")

        settings = {"depth_model_version": "v3"}
        cache_key = compute_cache_key(str(video_file), settings)

        cache_entry = cache_dir / cache_key
        cache_entry.mkdir()

        # Create metadata
        num_frames = 2
        metadata = {"num_frames": num_frames}
        with open(cache_entry / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Create first valid depth file
        depth_data = np.random.rand(100, 100).astype(np.float32)
        depth_uint16 = (depth_data * 1000.0).astype(np.uint16)
        cv2.imwrite(str(cache_entry / "depth_000000.png"), depth_uint16)

        # Create corrupted second depth file
        (cache_entry / "depth_000001.png").write_text("corrupted")

        result = get_cached_depth_maps(str(video_file), settings, num_frames)

        # Should return None (cache miss due to corrupted file)
        assert result is None

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    def test_cache_hit_invalid_metadata(self, mock_cache_dir, tmp_path):
        """Test cache miss when metadata is invalid JSON."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test")

        settings = {"depth_model_version": "v3"}
        cache_key = compute_cache_key(str(video_file), settings)

        cache_entry = cache_dir / cache_key
        cache_entry.mkdir()

        # Create invalid metadata
        (cache_entry / "metadata.json").write_text("{ invalid json")

        result = get_cached_depth_maps(str(video_file), settings, 10)

        # Should return None (cache miss)
        assert result is None

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    @patch("cv2.imwrite")
    def test_save_failure_exception(self, mock_imwrite, mock_cache_dir, tmp_path):
        """Test save_depth_maps_to_cache handles exceptions gracefully."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"test")

        settings = {"depth_model_version": "v3"}
        depth_maps = np.random.rand(2, 50, 50).astype(np.float32)

        # Make cv2.imwrite raise an exception
        mock_imwrite.side_effect = Exception("Write failed")

        success = save_depth_maps_to_cache(str(video_file), settings, depth_maps)

        # Should return False (graceful failure)
        assert success is False

    @patch("src.depth_surge_3d.utils.domain.depth_cache.get_cache_dir")
    @patch("shutil.rmtree")
    def test_clear_cache_exception(self, mock_rmtree, mock_cache_dir, tmp_path):
        """Test clear_cache handles exceptions gracefully."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        mock_cache_dir.return_value = cache_dir

        # Create cache entries
        (cache_dir / "entry1").mkdir()
        (cache_dir / "entry2").mkdir()

        # Make rmtree raise exception for first entry but succeed for second
        mock_rmtree.side_effect = [Exception("Delete failed"), None]

        count = clear_cache()

        # Should skip failed entry and count only successful one
        assert count == 1
