"""Depth estimator output contract tests."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.depth_surge_3d.inference.depth.types import (
    DepthBatch,
    DepthRepresentation,
    PinholeCameraBatch,
)
from src.depth_surge_3d.inference.depth.video_depth_estimator import VideoDepthEstimator
from src.depth_surge_3d.inference.depth.video_depth_estimator_da3 import VideoDepthEstimatorDA3
from src.depth_surge_3d.inference.depth.video_depth_estimator_see_through import (
    SeeThroughDepthEstimator,
)


class _VideoModel:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def infer_video_depth(self, *_args, **_kwargs):
        return self.values.copy(), None


class _DA3Model:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def inference(self, *_args, **_kwargs):
        return SimpleNamespace(depth=self.values.copy())


class _SeeThroughModel:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def __call__(self, **_kwargs):
        return SimpleNamespace(depth_np=self.values.copy())


def test_depth_batch_requires_rank_three_float32_values() -> None:
    values = np.zeros((2, 3, 4), dtype=np.float32)

    batch = DepthBatch(values, DepthRepresentation.INVERSE_DEPTH)

    assert batch.values is values
    with pytest.raises(TypeError, match="float32"):
        DepthBatch(values.astype(np.float64), DepthRepresentation.INVERSE_DEPTH)
    with pytest.raises(ValueError, match=r"\[N,H,W\]"):
        DepthBatch(values[0], DepthRepresentation.INVERSE_DEPTH)


def test_existing_depth_batch_remains_camera_free() -> None:
    batch = DepthBatch(
        np.ones((2, 3, 4), dtype=np.float32),
        DepthRepresentation.RELATIVE_DEPTH,
    )
    assert batch.camera is None


def test_pinhole_camera_requires_positive_finite_float32_values() -> None:
    with pytest.raises(TypeError, match="float32"):
        PinholeCameraBatch(np.array([1.0], dtype=np.float64))
    with pytest.raises(ValueError, match="positive"):
        PinholeCameraBatch(np.array([0.0], dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        PinholeCameraBatch(np.array([np.nan], dtype=np.float32))


def test_depth_batch_camera_count_matches_depth_count() -> None:
    with pytest.raises(ValueError, match="batch length"):
        DepthBatch(
            np.ones((2, 3, 4), dtype=np.float32),
            DepthRepresentation.METRIC_DEPTH,
            camera=PinholeCameraBatch(np.array([0.8], dtype=np.float32)),
        )


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        (False, DepthRepresentation.INVERSE_DEPTH),
        (True, DepthRepresentation.METRIC_DEPTH),
    ],
)
def test_video_depth_anything_declares_representation_without_normalizing(
    metric: bool, expected: DepthRepresentation
) -> None:
    raw = np.array([[[2.0, 4.0], [8.0, 16.0]]], dtype=np.float32)
    estimator = VideoDepthEstimator("unused.pth", device="cpu", metric=metric)
    estimator.model = _VideoModel(raw)
    frames = np.zeros((1, 2, 2, 3), dtype=np.uint8)

    result = estimator.estimate_depth_batch(frames, input_size=2)

    assert isinstance(result, DepthBatch)
    assert result.representation is expected
    np.testing.assert_array_equal(result.values, raw)


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        (False, DepthRepresentation.RELATIVE_DEPTH),
        (True, DepthRepresentation.METRIC_DEPTH),
    ],
)
def test_da3_returns_native_resolution_without_normalizing(
    metric: bool, expected: DepthRepresentation
) -> None:
    raw = np.array([[[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]]], dtype=np.float32)
    estimator = VideoDepthEstimatorDA3(model_name="base", device="cpu", metric=metric)
    estimator.model = _DA3Model(raw)
    frames = np.zeros((1, 4, 6, 3), dtype=np.uint8)

    result = estimator.estimate_depth_batch(frames, input_size=3)

    assert isinstance(result, DepthBatch)
    assert result.representation is expected
    assert result.values.shape == (1, 2, 3)
    np.testing.assert_array_equal(result.values, raw)


def test_see_through_returns_zero_as_valid_native_relative_depth() -> None:
    raw = np.array([[[0.0, 0.25], [0.5, 1.0]]], dtype=np.float32)
    estimator = SeeThroughDepthEstimator(
        device="cpu",
        processing_resolution=2,
        pipeline_loader=lambda **_kwargs: None,
    )
    estimator.model = _SeeThroughModel(raw)
    frames = np.zeros((1, 4, 6, 3), dtype=np.uint8)

    result = estimator.estimate_depth_batch(frames, input_size=2)

    assert isinstance(result, DepthBatch)
    assert result.representation is DepthRepresentation.RELATIVE_DEPTH
    assert result.values.shape == (1, 2, 2)
    np.testing.assert_array_equal(result.values, raw)
