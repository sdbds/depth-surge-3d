"""Unit tests for VideoDepthEstimator (V2)."""

import sys
import types

import numpy as np
import pytest
import torch
from unittest.mock import patch, MagicMock

from src.depth_surge_3d.inference.depth.video_depth_estimator import (
    VDA_INFER_LEN,
    VDA_INTERP_LEN,
    VDA_KEYFRAMES,
    VDA_OVERLAP,
    VideoDepthEstimator,
    create_video_depth_estimator,
)
from src.depth_surge_3d.inference.depth.types import DepthRepresentation
from src.depth_surge_3d.core.constants import DEFAULT_MODEL_PATH


class TestVideoDepthEstimator:
    """Test VideoDepthEstimator class."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH)
        assert estimator.model_path == DEFAULT_MODEL_PATH
        assert estimator.device in ["cuda", "cpu", "mps"]
        assert estimator.metric is False
        assert estimator.model is None

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        custom_path = "models/custom.pth"
        estimator = VideoDepthEstimator(custom_path, device="cpu", metric=True)
        assert estimator.model_path == custom_path
        assert estimator.device == "cpu"
        assert estimator.metric is True

    @pytest.mark.parametrize(
        ("device", "precision"),
        [("cpu", "float32"), ("cuda", "float16")],
    )
    def test_inference_precision_matches_fp32_false_execution(self, device, precision):
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device=device)

        assert estimator.inference_precision == precision

    def test_determine_device_auto_with_cuda(self):
        """Test device determination when CUDA is available."""
        with patch("torch.cuda.is_available", return_value=True):
            estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="auto")
            assert estimator.device == "cuda"

    def test_determine_device_auto_without_cuda(self):
        """Test device determination when CUDA is not available."""
        with patch("torch.cuda.is_available", return_value=False):
            estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="auto")
            assert estimator.device in ["cpu", "mps"]

    def test_determine_device_auto_mps(self):
        """Test device determination when MPS is available (macOS)."""
        with patch("torch.cuda.is_available", return_value=False):
            # Mock MPS availability
            mock_mps = MagicMock()
            mock_mps.is_available.return_value = True
            with patch("torch.backends.mps", mock_mps, create=True):
                estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="auto")
                assert estimator.device == "mps"

    def test_get_model_type_from_path(self):
        """Test model type detection from path."""
        estimator = VideoDepthEstimator("models/video_depth_anything_vits.pth")
        assert estimator._get_model_type(estimator.model_path) == "vits"

        estimator = VideoDepthEstimator("models/video_depth_anything_vitb.pth")
        assert estimator._get_model_type(estimator.model_path) == "vitb"

        estimator = VideoDepthEstimator("models/video_depth_anything_vitl.pth")
        assert estimator._get_model_type(estimator.model_path) == "vitl"

    def test_get_model_type_fallback(self):
        """Test model type detection falls back to vitl."""
        estimator = VideoDepthEstimator("models/unknown_model.pth")
        assert estimator._get_model_type(estimator.model_path) == "vitl"

    def test_estimator_has_no_per_frame_normalizer(self):
        """Estimator adapters must preserve raw model scale."""
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
        assert not hasattr(estimator, "_normalize_depths")

    def test_get_model_info_not_loaded(self):
        """Test model info when model is not loaded."""
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
        info = estimator.get_model_info()

        assert info == {"inference_algorithm": "vda-offline-shot-v1"}

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.empty_cache")
    def test_unload_model_with_cuda(self, mock_empty_cache, mock_cuda_available):
        """Test model unloading with CUDA cleanup."""
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cuda")
        estimator.model = MagicMock()  # Simulate loaded model
        estimator.unload_model()

        assert estimator.model is None
        mock_empty_cache.assert_called_once()

    def test_unload_model_without_loading(self):
        """Test unload when model was never loaded."""
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
        estimator.unload_model()  # Should not raise error
        assert estimator.model is None


class TestCreateVideoDepthEstimator:
    """Test factory function for VideoDepthEstimator."""

    def test_create_with_defaults(self):
        """Test factory function with defaults."""
        estimator = create_video_depth_estimator()
        assert isinstance(estimator, VideoDepthEstimator)
        assert estimator.model_path == DEFAULT_MODEL_PATH

    def test_create_with_custom_params(self):
        """Test factory function with custom parameters."""
        custom_path = "models/custom.pth"
        estimator = create_video_depth_estimator(model_path=custom_path, device="cpu", metric=True)
        assert isinstance(estimator, VideoDepthEstimator)
        assert estimator.model_path == custom_path
        assert estimator.device == "cpu"
        assert estimator.metric is True

    def test_create_with_none_model_path(self):
        """Test factory function with None model path uses default."""
        estimator = create_video_depth_estimator(model_path=None)
        assert estimator.model_path == DEFAULT_MODEL_PATH


class TestEnsureDependencies:
    """Test _ensure_dependencies method."""

    def test_ensure_dependencies_success(self):
        """Test successful dependency check."""
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")

        # Assume dependencies are available in test environment
        result = estimator._ensure_dependencies()

        # Should return True if repo exists
        assert isinstance(result, bool)

    def test_ensure_dependencies_repo_not_exists(self):
        """Test dependency check when repo doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
            result = estimator._ensure_dependencies()

            # Should return False when repo missing
            assert result is False

    def test_ensure_dependencies_model_not_found_auto_download_fails(self):
        """Test when model file doesn't exist and auto-download fails."""
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")

        # Mock os.path.exists to return False for model file
        with patch("os.path.exists", return_value=False):
            with patch.object(estimator, "_auto_download_model", return_value=False):
                result = estimator._ensure_dependencies()

                assert result is False


class TestSuppressModelOutput:
    """Test _suppress_model_output context manager."""

    def test_suppress_model_output_context(self):
        """Test that output suppression works as context manager."""
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")

        # Should not raise error
        with estimator._suppress_model_output():
            pass  # Context manager should work

    def test_suppress_model_output_restores_stdout(self):
        """Test that stdout is restored after suppression."""
        import sys

        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")

        original_stdout = sys.stdout

        with estimator._suppress_model_output():
            # stdout should be redirected
            assert sys.stdout != original_stdout

        # stdout should be restored
        assert sys.stdout == original_stdout


class TestLoadModel:
    """Test load_model method."""

    def test_load_model_dependencies_fail(self):
        """Test load_model when dependencies are missing."""
        with patch.object(VideoDepthEstimator, "_ensure_dependencies", return_value=False):
            estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
            result = estimator.load_model()

            assert result is False
            assert estimator.model is None

    def test_load_model_invalid_model_type(self):
        """Test load_model with invalid model type."""
        with patch.object(VideoDepthEstimator, "_ensure_dependencies", return_value=True):
            with patch.object(VideoDepthEstimator, "_get_model_type", return_value=None):
                estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
                result = estimator.load_model()

                assert result is False
                assert estimator.model is None

    def test_load_model_exception_handling(self):
        """Test load_model handles exceptions gracefully."""
        with patch.object(VideoDepthEstimator, "_ensure_dependencies", return_value=True):
            with patch.object(VideoDepthEstimator, "_get_model_type", return_value="vitl"):
                # Make sys.path.insert raise an exception
                with patch("sys.path", side_effect=RuntimeError("Test error")):
                    estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
                    result = estimator.load_model()

                    assert result is False


class TestAutoDownloadModel:
    """Test _auto_download_model method."""

    def test_auto_download_model_file_exists(self):
        """Test auto download when model file already exists."""
        with patch("pathlib.Path.exists", return_value=True):
            estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
            result = estimator._auto_download_model()

            # Should return True if file exists
            assert result is True

    def test_auto_download_model_no_url(self):
        """Test auto download when no URL configured."""
        with patch("pathlib.Path.exists", return_value=False):
            estimator = VideoDepthEstimator("models/unknown_model.pth", device="cpu")

            # Mock MODEL_DOWNLOAD_URLS to not have this model
            with patch(
                "src.depth_surge_3d.inference.depth.video_depth_estimator.MODEL_DOWNLOAD_URLS", {}
            ):
                result = estimator._auto_download_model()

                # Should return False when no URL available
                assert result is False

    def test_auto_download_model_download_exception(self):
        """Test auto download when download fails with exception."""
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")

        with patch("pathlib.Path.exists", return_value=False):
            with patch.object(estimator, "_get_model_type", return_value="vitl"):
                # Mock urllib.request.urlretrieve to raise an exception
                with patch("urllib.request.urlretrieve", side_effect=Exception("Network error")):
                    result = estimator._auto_download_model()

                    assert result is False


class TestGetModelInfo:
    """Test get_model_info when model is loaded."""

    def test_get_model_info_with_loaded_model(self):
        """Test model info when model is loaded."""
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")

        # Simulate loaded model with all required config keys
        estimator.model = MagicMock()
        estimator.model_config = {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
            "num_frames": 32,
        }

        info = estimator.get_model_info()

        assert "loaded" in info
        assert info["loaded"] is True
        assert "encoder" in info
        assert info["encoder"] == "vitl"
        assert "features" in info
        assert info["features"] == 256
        assert "temporal_consistency" in info
        assert info["temporal_consistency"] is True
        assert info["inference_algorithm"] == "vda-offline-shot-v1"


class TestEstimateDepthBatchCompatibility:
    """The in-memory API delegates one complete array to upstream VDA."""

    def test_estimate_depth_batch_model_not_loaded(self):
        """Test batch estimation when model not loaded."""
        import numpy as np
        import pytest

        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
        frames = np.random.rand(10, 480, 640, 3)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            estimator.estimate_depth_batch(frames)

    def test_large_batch_calls_upstream_once(self):
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
        estimator.model = MagicMock()
        frames = np.zeros((70, 2, 3, 3), dtype=np.uint8)
        with patch.object(estimator, "_estimate_depth_single_batch") as mock_single:
            mock_single.return_value = np.zeros((70, 2, 3), dtype=np.float32)
            result = estimator.estimate_depth_batch(frames)

            mock_single.assert_called_once()
            assert result.values.shape == (70, 2, 3)
            assert result.representation is DepthRepresentation.INVERSE_DEPTH


class TestEstimateDepthSingleBatch:
    """Test _estimate_depth_single_batch method."""

    def test_estimate_depth_single_batch_success(self):
        """Test single batch depth estimation."""
        import numpy as np

        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
        estimator.model = MagicMock()

        # Mock infer_video_depth to return depth maps
        mock_depths = np.random.rand(10, 480, 640)
        estimator.model.infer_video_depth.return_value = (mock_depths, None)

        frames = np.random.rand(10, 480, 640, 3).astype(np.uint8)

        with patch("builtins.open", MagicMock()):
            result = estimator._estimate_depth_single_batch(frames, 30, 518, False)

        assert result.shape == (10, 480, 640)
        estimator.model.infer_video_depth.assert_called_once()

    def test_estimate_depth_single_batch_exception(self):
        """Test single batch with model exception."""
        import numpy as np
        import pytest

        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
        estimator.model = MagicMock()
        estimator.model.infer_video_depth.side_effect = RuntimeError("CUDA OOM")

        frames = np.random.rand(10, 480, 640, 3).astype(np.uint8)

        with patch("builtins.open", MagicMock()):
            with pytest.raises(RuntimeError, match="Video depth estimation failed"):
                estimator._estimate_depth_single_batch(frames, 30, 518, False)


def _frame_loader(frame_count, requests):
    def load(indexes):
        requested = list(indexes)
        requests.append(requested)
        frames = np.zeros((len(requested), 2, 3, 3), dtype=np.uint8)
        for offset, index in enumerate(requested):
            frames[offset].fill(index)
        return frames

    return load


def _install_fake_fixed_forward(estimator, monkeypatch, *, fail_calls=(), observer=None):
    calls = []
    failures = set(fail_calls)

    def infer_fixed_window(
        frames,
        *,
        input_size,
        fp32,
        carried_input=None,
        padding_input=None,
        transform=None,
        output_shape=None,
    ):
        del input_size, fp32
        call_number = len(calls) + 1
        decoded = [int(frame[0, 0, 0]) for frame in frames]
        calls.append(decoded)
        if call_number in failures:
            failures.remove(call_number)
            raise torch.cuda.OutOfMemoryError("test window OOM")

        if carried_input is None:
            markers = list(decoded)
            markers.extend([markers[-1]] * (32 - len(markers)))
        else:
            markers = [int(value) for value in carried_input.flatten().tolist()]
            markers.extend(decoded)
            pad_value = decoded[-1] if decoded else int(padding_input.flatten()[0].item())
            markers.extend([pad_value] * (32 - len(markers)))
        current_input = torch.tensor(markers, dtype=torch.float32).reshape(1, 32, 1, 1, 1)
        depths = np.asarray(markers, dtype=np.float32).reshape(32, 1, 1)
        depths = np.broadcast_to(depths, (32, *output_shape)).copy()
        if observer is not None:
            observer(frames, carried_input, current_input, depths)
        return depths, current_input, transform or object()

    monkeypatch.setattr(estimator, "_infer_fixed_window", infer_fixed_window, raising=False)
    monkeypatch.setattr(
        estimator,
        "_interpolate_depths",
        lambda previous, current: (np.asarray(previous) + np.asarray(current)) / 2,
        raising=False,
    )
    return calls


def _install_fake_vda_transform_modules(monkeypatch, observed):
    transform_module = types.ModuleType("video_depth_anything.util.transform")

    class Resize:
        def __init__(self, **kwargs):
            observed.update(kwargs)

    class Passthrough:
        def __init__(self, **_kwargs):
            pass

    transform_module.Resize = Resize
    transform_module.NormalizeImage = Passthrough
    transform_module.PrepareForNet = Passthrough
    package = types.ModuleType("video_depth_anything")
    util_package = types.ModuleType("video_depth_anything.util")
    monkeypatch.setitem(sys.modules, "video_depth_anything", package)
    monkeypatch.setitem(sys.modules, "video_depth_anything.util", util_package)
    monkeypatch.setitem(sys.modules, "video_depth_anything.util.transform", transform_module)


def _install_upstream_numeric_helpers(monkeypatch, calls):
    util_module = types.ModuleType("utils.util")

    def compute_scale_and_shift(prediction, target, mask):
        calls["scale"] += 1
        prediction = np.asarray(prediction, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        a_00 = np.sum(mask * prediction * prediction)
        a_01 = np.sum(mask * prediction)
        a_11 = np.sum(mask)
        b_0 = np.sum(mask * prediction * target)
        b_1 = np.sum(mask * target)
        determinant = a_00 * a_11 - a_01 * a_01
        if determinant == 0:
            return 1.0, 0.0
        return (
            (a_11 * b_0 - a_01 * b_1) / determinant,
            (-a_01 * b_0 + a_00 * b_1) / determinant,
        )

    def get_interpolate_frames(previous, current):
        calls["interpolate"] += 1
        weights = np.linspace(0.0, 1.0, len(previous), dtype=np.float32)
        return [
            previous[index] * (1.0 - weight) + current[index] * weight
            for index, weight in enumerate(weights)
        ]

    util_module.compute_scale_and_shift = compute_scale_and_shift
    util_module.get_interpolate_frames = get_interpolate_frames
    utils_package = types.ModuleType("utils")
    utils_package.util = util_module
    monkeypatch.setitem(sys.modules, "utils", utils_package)
    monkeypatch.setitem(sys.modules, "utils.util", util_module)
    return compute_scale_and_shift, get_interpolate_frames


class _DeterministicWindowModel:
    def __init__(self) -> None:
        self.calls = 0

    def forward(self, current_input):
        call = self.calls
        self.calls += 1
        values = current_input[:, :, 0].float()
        temporal = torch.arange(VDA_INFER_LEN, dtype=torch.float32).reshape(1, -1, 1, 1)
        height, width = values.shape[-2:]
        spatial = torch.arange(height * width, dtype=torch.float32).reshape(1, 1, height, width)
        return (
            values * (1.0 + 0.35 * call)
            + 3.0 * call
            + temporal * 0.07 * call
            + spatial * 0.03 * call
        )


class _UpstreamOfflineReference:
    """Small direct transcription of upstream infer_video_depth orchestration."""

    def __init__(self, model, metric, compute_scale_and_shift, get_interpolate_frames):
        self.model = model
        self.metric = metric
        self.compute_scale_and_shift = compute_scale_and_shift
        self.get_interpolate_frames = get_interpolate_frames

    def infer_video_depth(self, frames, target_fps, **_kwargs):
        frame_list = [
            torch.from_numpy(frame[..., 0].astype(np.float32)).reshape(1, 1, 1, 2, 3)
            for frame in frames
        ]
        frame_step = VDA_INFER_LEN - VDA_OVERLAP
        original_length = len(frame_list)
        append_length = (frame_step - (original_length % frame_step)) % frame_step + (
            VDA_INFER_LEN - frame_step
        )
        frame_list.extend([frame_list[-1].clone()] * append_length)
        depth_list = []
        previous_input = None
        for frame_start in range(0, original_length, frame_step):
            current_input = torch.cat(
                frame_list[frame_start : frame_start + VDA_INFER_LEN],
                dim=1,
            )
            if previous_input is not None:
                current_input[:, :VDA_OVERLAP] = previous_input[:, list(VDA_KEYFRAMES)]
            depth = self.model.forward(current_input)[0]
            depth_list.extend(depth[index].numpy() for index in range(VDA_INFER_LEN))
            previous_input = current_input

        aligned = []
        references = []
        alignment_length = VDA_OVERLAP - VDA_INTERP_LEN
        keyframe_alignment = VDA_KEYFRAMES[:alignment_length]
        for frame_start in range(0, len(depth_list), VDA_INFER_LEN):
            if not aligned:
                aligned.extend(depth_list[:VDA_INFER_LEN])
                references.extend(depth_list[frame_start + index] for index in keyframe_alignment)
                continue
            current_references = [
                depth_list[frame_start + index] for index in range(len(keyframe_alignment))
            ]
            if self.metric:
                scale, shift = 1.0, 0.0
            else:
                scale, shift = self.compute_scale_and_shift(
                    np.concatenate(current_references),
                    np.concatenate(references),
                    np.concatenate(np.ones_like(references) == 1),
                )
            previous_overlap = aligned[-VDA_INTERP_LEN:]
            current_overlap = depth_list[frame_start + alignment_length : frame_start + VDA_OVERLAP]
            current_overlap = [np.maximum(value * scale + shift, 0) for value in current_overlap]
            aligned[-VDA_INTERP_LEN:] = self.get_interpolate_frames(
                previous_overlap,
                current_overlap,
            )
            for index in range(VDA_OVERLAP, VDA_INFER_LEN):
                aligned.append(np.maximum(depth_list[frame_start + index] * scale + shift, 0))
            references = references[:1]
            references.extend(
                np.maximum(depth_list[frame_start + index] * scale + shift, 0)
                for index in keyframe_alignment[1:]
            )
        return np.stack(aligned[:original_length], axis=0), target_fps


@pytest.mark.parametrize("input_size", [2160, 1080, 720, 640, 518, 384])
def test_vda_transform_preserves_requested_size_at_16_by_9(
    input_size,
    monkeypatch,
):
    observed = {}
    _install_fake_vda_transform_modules(monkeypatch, observed)

    VideoDepthEstimator._build_vda_transform(1080, 1920, input_size)

    assert observed["width"] == input_size
    assert observed["height"] == input_size


def test_vda_transform_rounds_only_after_ultrawide_ratio_reduction(monkeypatch):
    observed = {}
    _install_fake_vda_transform_modules(monkeypatch, observed)

    VideoDepthEstimator._build_vda_transform(1000, 2000, 518)

    expected = round(int(518 * 1.777 / 2.0) / 14) * 14
    assert observed["width"] == expected
    assert observed["height"] == expected


class TestSequenceDepth:
    @pytest.mark.parametrize("frame_count", [1, 22, 23, 24, 25, 31, 32, 33, 53, 54, 55, 100])
    def test_sequence_yields_every_source_frame_once(self, frame_count, monkeypatch):
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu", metric=True)
        estimator.model = object()
        requests = []
        _install_fake_fixed_forward(estimator, monkeypatch)

        yielded = list(
            estimator.iter_sequence_depth(
                frame_count,
                _frame_loader(frame_count, requests),
                target_fps=30,
                input_size=518,
                fp32=False,
            )
        )

        values = np.concatenate([batch.values for _start, batch in yielded], axis=0)
        starts = [start for start, _batch in yielded]
        expected_starts = []
        cursor = 0
        for start, batch in yielded:
            expected_starts.append(cursor)
            cursor += len(batch.values)
            assert start == expected_starts[-1]
            assert batch.representation is DepthRepresentation.METRIC_DEPTH
        assert starts == expected_starts
        assert cursor == frame_count
        assert values[:, 0, 0].tolist() == list(range(frame_count))
        assert max(map(len, requests)) <= 32
        assert all(request == sorted(set(request)) for request in requests)

    def test_sequence_requests_only_new_shot_local_indexes(self, monkeypatch):
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu", metric=True)
        estimator.model = object()
        requests = []
        _install_fake_fixed_forward(estimator, monkeypatch)

        list(
            estimator.iter_sequence_depth(
                55,
                _frame_loader(55, requests),
                target_fps=30,
                input_size=518,
                fp32=False,
            )
        )

        assert requests == [list(range(32)), list(range(32, 54)), [54]]

    def test_failed_window_can_be_retried_without_advancing_state(self, monkeypatch):
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu", metric=True)
        estimator.model = object()
        requests = []
        calls = _install_fake_fixed_forward(estimator, monkeypatch, fail_calls={2})
        iterator = estimator.iter_sequence_depth(
            55,
            _frame_loader(55, requests),
            target_fps=30,
            input_size=518,
            fp32=False,
        )

        first_start, first = next(iterator)
        with pytest.raises(torch.cuda.OutOfMemoryError, match="test window OOM"):
            next(iterator)
        second_start, second = next(iterator)

        assert first_start == 0
        assert first.values[:, 0, 0].tolist() == list(range(24))
        assert second_start == 24
        assert second.values[:, 0, 0].tolist() == list(range(24, 46))
        assert calls[1] == calls[2] == list(range(32, 54))
        assert requests[1] == requests[2] == list(range(32, 54))

    def test_later_windows_reuse_the_ten_frame_device_state_buffer(self, monkeypatch):
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu", metric=True)
        estimator.model = object()
        _install_fake_fixed_forward(estimator, monkeypatch)
        iterator = estimator.iter_sequence_depth(
            55,
            _frame_loader(55, []),
            target_fps=30,
            input_size=518,
            fp32=False,
        )

        next(iterator)
        assert iterator._retained_input is not None
        retained_pointer = iterator._retained_input.data_ptr()
        next(iterator)

        assert iterator._retained_input is not None
        assert iterator._retained_input.data_ptr() == retained_pointer

    def test_close_releases_retained_device_and_host_state(self, monkeypatch):
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu", metric=True)
        estimator.model = object()
        _install_fake_fixed_forward(estimator, monkeypatch)
        iterator = estimator.iter_sequence_depth(
            55,
            _frame_loader(55, []),
            target_fps=30,
            input_size=518,
            fp32=False,
        )
        next(iterator)
        assert iterator._retained_input is not None

        iterator.close()

        assert iterator._retained_input is None
        assert iterator._alignment_refs is None
        assert iterator._pending_depth is None
        with pytest.raises(StopIteration):
            next(iterator)

    @pytest.mark.parametrize(
        ("bad_frames", "message"),
        [
            (np.zeros((0, 2, 3, 3), dtype=np.uint8), "count"),
            (np.zeros((1, 2, 3, 3), dtype=np.float32), "uint8"),
            (np.zeros((1, 2, 3), dtype=np.uint8), "rank"),
            (np.zeros((1, 2, 3, 4), dtype=np.uint8), "channels"),
        ],
    )
    def test_sequence_rejects_invalid_loader_arrays(self, bad_frames, message):
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
        estimator.model = object()

        iterator = estimator.iter_sequence_depth(
            1,
            lambda _indexes: bad_frames,
            target_fps=30,
            input_size=518,
            fp32=False,
        )

        with pytest.raises((TypeError, ValueError), match=message):
            next(iterator)

    def test_sequence_rejects_geometry_change(self, monkeypatch):
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu", metric=True)
        estimator.model = object()
        _install_fake_fixed_forward(estimator, monkeypatch)
        call_count = 0

        def load(indexes):
            nonlocal call_count
            call_count += 1
            height = 2 if call_count == 1 else 4
            return np.zeros((len(indexes), height, 3, 3), dtype=np.uint8)

        iterator = estimator.iter_sequence_depth(
            33,
            load,
            target_fps=30,
            input_size=518,
            fp32=False,
        )
        next(iterator)

        with pytest.raises(ValueError, match="geometry"):
            next(iterator)

    def test_sequence_propagates_loader_exception(self):
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu")
        estimator.model = object()

        def load(_indexes):
            raise OSError("missing source.png")

        iterator = estimator.iter_sequence_depth(
            1,
            load,
            target_fps=30,
            input_size=518,
            fp32=False,
        )

        with pytest.raises(OSError, match="missing source.png"):
            next(iterator)

    def test_sequence_retains_only_fixed_temporal_state(self, monkeypatch):
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu", metric=True)
        estimator.model = object()
        _install_fake_fixed_forward(estimator, monkeypatch)
        iterator = estimator.iter_sequence_depth(
            100,
            _frame_loader(100, []),
            target_fps=30,
            input_size=518,
            fp32=False,
        )

        next(iterator)

        assert iterator._retained_input.shape[1] == 10
        assert iterator._alignment_refs.shape[0] == 2
        assert iterator._pending_depth.shape[0] == 8
        assert not hasattr(iterator, "_all_depths")

    @pytest.mark.parametrize("metric", [False, True])
    def test_sequence_is_numerically_equivalent_to_upstream_offline_inference(
        self,
        metric,
        monkeypatch,
    ):
        frame_count = 55
        frames = np.empty((frame_count, 2, 3, 3), dtype=np.uint8)
        spatial = np.arange(6, dtype=np.uint8).reshape(2, 3)
        for index in range(frame_count):
            frames[index] = (index + spatial[..., None]) % 255
        helper_calls = {"scale": 0, "interpolate": 0}
        compute_scale_and_shift, get_interpolate_frames = _install_upstream_numeric_helpers(
            monkeypatch,
            helper_calls,
        )
        reference = _UpstreamOfflineReference(
            _DeterministicWindowModel(),
            metric,
            compute_scale_and_shift,
            get_interpolate_frames,
        )
        expected, _fps = reference.infer_video_depth(
            frames,
            30,
            input_size=518,
            device="cpu",
            fp32=True,
        )
        helper_calls.update(scale=0, interpolate=0)
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu", metric=metric)
        estimator.model = _DeterministicWindowModel()
        monkeypatch.setattr(estimator, "_build_vda_transform", lambda *_args: object())
        monkeypatch.setattr(
            estimator,
            "_transform_vda_frame",
            lambda frame, _transform: torch.from_numpy(frame[..., 0].astype(np.float32)).reshape(
                1, 1, 1, 2, 3
            ),
        )

        yielded = list(
            estimator.iter_sequence_depth(
                frame_count,
                lambda indexes: frames[list(indexes)].copy(),
                target_fps=30,
                input_size=518,
                fp32=True,
            )
        )
        actual = np.concatenate([batch.values for _start, batch in yielded], axis=0)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        assert helper_calls["scale"] == (0 if metric else 2)
        assert helper_calls["interpolate"] == 2

    @pytest.mark.parametrize("frame_count", [55, 220])
    def test_sequence_enforces_documented_working_memory_bounds(
        self,
        frame_count,
        monkeypatch,
    ):
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu", metric=True)
        estimator.model = object()
        observed = {"decoded": 0, "host_depth": 0, "device_input": 0}

        def observe_window(frames, carried_input, current_input, depths):
            observed["decoded"] = max(observed["decoded"], len(frames))
            carried_count = 0 if carried_input is None else carried_input.shape[1]
            observed["device_input"] = max(
                observed["device_input"],
                current_input.shape[1] + carried_count,
            )
            assert len(depths) == VDA_INFER_LEN

        _install_fake_fixed_forward(estimator, monkeypatch, observer=observe_window)

        def observe_interpolation(previous, current):
            blended = [
                previous[index] * 0.5 + current[index] * 0.5 for index in range(VDA_INTERP_LEN)
            ]
            observed["host_depth"] = max(
                observed["host_depth"],
                len(previous) + VDA_INFER_LEN + 2 + len(blended),
            )
            return blended

        monkeypatch.setattr(estimator, "_interpolate_depths", observe_interpolation)

        list(
            estimator.iter_sequence_depth(
                frame_count,
                _frame_loader(frame_count, []),
                target_fps=30,
                input_size=518,
                fp32=False,
            )
        )

        assert observed == {"decoded": 32, "host_depth": 50, "device_input": 42}

    def test_finalize_window_reuses_current_storage_for_finalized_output(self, monkeypatch):
        estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu", metric=True)
        current = np.arange(VDA_INFER_LEN * 6, dtype=np.float32).reshape(
            VDA_INFER_LEN,
            2,
            3,
        )
        pending = np.arange(VDA_INTERP_LEN * 6, dtype=np.float32).reshape(
            VDA_INTERP_LEN,
            2,
            3,
        )
        references = np.zeros((2, 2, 3), dtype=np.float32)
        monkeypatch.setattr(
            estimator,
            "_interpolate_depths",
            lambda previous, post: [value.copy() for value in previous],
        )

        finalized, next_pending, next_reference = estimator._finalize_window(
            pending,
            current,
            references,
        )

        assert np.shares_memory(finalized, current)
        assert not np.shares_memory(next_pending, current)
        assert np.shares_memory(next_reference, current)
        assert references[:, 0, 0].tolist() == [0.0, 0.0]


def test_finalize_window_aligns_only_non_keyframe_outputs(monkeypatch):
    estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu", metric=False)
    current = np.arange(32, dtype=np.float32).reshape(32, 1, 1)
    pending = np.arange(100, 108, dtype=np.float32).reshape(8, 1, 1)
    references = np.asarray([[[10.0]], [[20.0]]], dtype=np.float32)
    observed = {}

    def compute(current_refs, retained_refs):
        observed["current_refs"] = current_refs.copy()
        observed["retained_refs"] = retained_refs.copy()
        return 2.0, 3.0

    def interpolate(previous, post):
        observed["post"] = np.asarray(post).copy()
        return np.asarray(previous)

    monkeypatch.setattr(estimator, "_compute_scale_and_shift", compute, raising=False)
    monkeypatch.setattr(estimator, "_interpolate_depths", interpolate, raising=False)

    finalized, next_pending, next_reference = estimator._finalize_window(
        pending, current, references
    )

    assert observed["current_refs"][:, 0, 0].tolist() == [0.0, 1.0]
    assert observed["retained_refs"][:, 0, 0].tolist() == [10.0, 20.0]
    assert observed["post"][:, 0, 0].tolist() == list(range(7, 23, 2))
    assert finalized[:8, 0, 0].tolist() == list(range(100, 108))
    assert finalized[8:, 0, 0].tolist() == list(range(23, 51, 2))
    assert next_pending[:, 0, 0].tolist() == list(range(51, 67, 2))
    assert next_reference[0, 0] == 27.0
    assert references[:, 0, 0].tolist() == [10.0, 20.0]


def test_fixed_window_pads_from_latest_new_source_frame(monkeypatch):
    class EchoModel:
        @staticmethod
        def forward(values):
            return values[:, :, 0]

    estimator = VideoDepthEstimator(DEFAULT_MODEL_PATH, device="cpu", metric=True)
    estimator.model = EchoModel()
    frames = np.zeros((2, 1, 1, 3), dtype=np.uint8)
    frames[0].fill(50)
    frames[1].fill(51)
    carried = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1, 1, 1)
    monkeypatch.setattr(
        estimator,
        "_transform_vda_frame",
        lambda frame, _transform: torch.tensor(float(frame[0, 0, 0]), dtype=torch.float32).reshape(
            1, 1, 1, 1, 1
        ),
    )

    depths, current_input, _transform = estimator._infer_fixed_window(
        frames,
        input_size=518,
        fp32=True,
        carried_input=carried,
        padding_input=carried[:, -1:],
        transform=object(),
        output_shape=(1, 1),
    )

    expected = list(range(10)) + [50, 51] + [51] * 20
    assert current_input.flatten().tolist() == expected
    assert depths[:, 0, 0].tolist() == expected
