"""Inference contract tests for the optional MoGe-2 adapter."""

from __future__ import annotations

import hashlib
import sys
from types import ModuleType
from unittest.mock import MagicMock, Mock

import cv2
import numpy as np
import pytest
import torch

import src.depth_surge_3d.inference.depth as depth_package
import src.depth_surge_3d.inference.depth.video_depth_estimator_moge2 as moge_module
from src.depth_surge_3d.inference.depth.types import DepthRepresentation
from src.depth_surge_3d.inference.depth.video_depth_estimator_moge2 import (
    MOGE_PREPROCESSING_ALGORITHM,
    MOGE_RESOLUTION_LEVEL,
    MOGE_SOURCE_REVISION,
    VideoDepthEstimatorMoGe2,
    create_video_depth_estimator_moge2,
)


class FakeMoGeModel:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, dict[str, object]]] = []
        self.output = self._valid_output()

    @staticmethod
    def _valid_output() -> dict[str, torch.Tensor]:
        depth = torch.full((1, 4, 8), 2.0, dtype=torch.float32)
        mask = torch.ones((1, 4, 8), dtype=torch.bool)
        mask[:, 0, 0] = False
        intrinsics = torch.eye(3, dtype=torch.float32).unsqueeze(0)
        intrinsics[:, 0, 0] = 0.75
        return {
            "depth": depth,
            "mask": mask,
            "intrinsics": intrinsics,
            "points": torch.zeros((1, 4, 8, 3)),
            "normal": torch.zeros((1, 4, 8, 3)),
        }

    def infer(self, image: torch.Tensor, **kwargs):
        self.calls.append((image.detach().cpu(), kwargs))
        if image.shape[-2:] == (4, 8):
            return self.output
        height, width = image.shape[-2:]
        output = self._valid_output()
        output["depth"] = torch.full((1, height, width), 2.0, dtype=torch.float32)
        output["mask"] = torch.ones((1, height, width), dtype=torch.bool)
        output["mask"][:, 0, 0] = False
        return output


def one_frame() -> np.ndarray:
    return np.zeros((1, 4, 8, 3), dtype=np.uint8)


def loaded_fake_moge(*, device: str = "cpu", model_size: str = "vitb") -> VideoDepthEstimatorMoGe2:
    estimator = VideoDepthEstimatorMoGe2(model_size=model_size, device=device)
    estimator.model = FakeMoGeModel()
    return estimator


def test_moge_preprocessing_is_rgb_area_no_upscale_and_forwards_fixed_options(
    monkeypatch,
) -> None:
    resize_calls = []
    real_resize = cv2.resize

    def recording_resize(image, size, *, interpolation):
        resize_calls.append((size, interpolation))
        return real_resize(image, size, interpolation=interpolation)

    monkeypatch.setattr(cv2, "resize", recording_resize)
    estimator = VideoDepthEstimatorMoGe2(model_size="vitb", device="cpu")
    estimator.model = FakeMoGeModel()
    bgr = np.zeros((1, 4, 8, 3), dtype=np.uint8)
    bgr[0, :, :, 0] = 255

    result = estimator.estimate_depth_batch(bgr, input_size=4, fp32=False)

    image, kwargs = estimator.model.calls[0]
    assert image.shape == (1, 3, 2, 4)
    assert image.dtype == torch.float32
    assert torch.all(image[:, 2] == 1.0)
    assert resize_calls == [((4, 2), cv2.INTER_AREA)]
    assert kwargs == {
        "force_projection": False,
        "apply_mask": True,
        "resolution_level": 9,
        "use_fp16": False,
    }
    assert result.representation is DepthRepresentation.METRIC_DEPTH
    assert np.isinf(result.values[0, 0, 0])
    assert result.camera is not None
    np.testing.assert_array_equal(
        result.camera.focal_x_normalized,
        np.array([0.75], dtype=np.float32),
    )


def test_moge_never_upscales_and_accepts_only_one_frame() -> None:
    estimator = loaded_fake_moge()
    estimator.estimate_depth_batch(np.zeros((1, 3, 5, 3), dtype=np.uint8), input_size=20)
    image, _kwargs = estimator.model.calls[0]
    assert image.shape[-2:] == (3, 5)
    with pytest.raises(ValueError, match="one frame"):
        estimator.estimate_depth_batch(np.zeros((2, 3, 5, 3), dtype=np.uint8))


def test_moge_output_shape_is_aspect_preserving_and_rejects_invalid_values() -> None:
    estimator = VideoDepthEstimatorMoGe2(model_size="vitb", device="cpu")
    assert estimator.estimate_output_shape(8, 4, 4) == (2, 4)
    assert estimator.estimate_output_shape(5, 3, 20) == (3, 5)
    with pytest.raises(ValueError, match="positive"):
        estimator.estimate_output_shape(0, 3, 20)
    with pytest.raises(ValueError, match="positive"):
        estimator.estimate_output_shape(5, 3, 0)


@pytest.mark.parametrize("missing", ["depth", "mask", "intrinsics"])
def test_moge_requires_every_output_key(missing: str) -> None:
    estimator = loaded_fake_moge()
    estimator.model.output.pop(missing)
    with pytest.raises(ValueError, match=missing):
        estimator.estimate_depth_batch(one_frame())


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("depth", torch.ones((1, 4, 8), dtype=torch.float64), "depth.*float32"),
        ("depth", torch.ones((4, 8), dtype=torch.float32), "depth.*rank"),
        ("depth", torch.ones((2, 4, 8), dtype=torch.float32), "frame count"),
        ("mask", torch.ones((1, 4, 7), dtype=torch.bool), "spatial shape"),
        ("mask", torch.ones((4, 8), dtype=torch.bool), "mask.*rank"),
        ("intrinsics", torch.eye(3, dtype=torch.float32), "intrinsics.*rank"),
    ],
)
def test_moge_rejects_malformed_outputs(field, replacement, message) -> None:
    estimator = loaded_fake_moge()
    estimator.model.output[field] = replacement
    with pytest.raises((TypeError, ValueError), match=message):
        estimator.estimate_depth_batch(one_frame())


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("mask", torch.ones((1, 4, 8), dtype=torch.uint8), "mask.*bool"),
        (
            "intrinsics",
            torch.eye(3, dtype=torch.float64).unsqueeze(0),
            "intrinsics.*float32",
        ),
        (
            "intrinsics",
            torch.ones((1, 2, 3), dtype=torch.float32),
            "intrinsics.*shape",
        ),
    ],
)
def test_moge_rejects_wrong_mask_and_intrinsics_contracts(field, replacement, message) -> None:
    estimator = loaded_fake_moge()
    estimator.model.output[field] = replacement
    with pytest.raises((TypeError, ValueError), match=message):
        estimator.estimate_depth_batch(one_frame())


def test_moge_rejects_non_tensor_outputs() -> None:
    estimator = loaded_fake_moge()
    estimator.model.output["depth"] = np.ones((1, 4, 8), dtype=np.float32)
    with pytest.raises(TypeError, match="depth.*tensor"):
        estimator.estimate_depth_batch(one_frame())


@pytest.mark.parametrize("focal", [0.0, float("nan")])
def test_moge_rejects_invalid_normalized_focal(focal: float) -> None:
    estimator = loaded_fake_moge()
    estimator.model.output["intrinsics"][:, 0, 0] = focal
    with pytest.raises(ValueError, match="focal"):
        estimator.estimate_depth_batch(one_frame())


@pytest.mark.parametrize(
    ("device", "expected_fp16", "expected_precision"),
    [("cpu", False, "float32"), ("cuda", True, "float16")],
)
def test_moge_precision_is_device_fixed(device, expected_fp16, expected_precision) -> None:
    estimator = loaded_fake_moge(device=device)
    estimator.estimate_depth_batch(one_frame())
    assert estimator.model.calls[0][1]["use_fp16"] is expected_fp16
    assert estimator.inference_precision == expected_precision


def test_moge_cuda_honors_explicit_fp32_without_changing_device() -> None:
    estimator = loaded_fake_moge(device="cuda")
    estimator.estimate_depth_batch(one_frame(), fp32=True)
    assert estimator.model.calls[0][1]["use_fp16"] is False
    assert estimator.device == "cuda"


def test_moge_cuda_oom_reports_fixed_inputs() -> None:
    estimator = loaded_fake_moge(device="cuda", model_size="vitl")
    estimator.model.infer = Mock(side_effect=torch.cuda.OutOfMemoryError("oom"))
    with pytest.raises(
        RuntimeError,
        match=(
            r"model_size=vitl.*input=8x4.*resolution_level=9.*" r"precision=float16.*device=cuda"
        ),
    ):
        estimator.estimate_depth_batch(one_frame())


def test_moge_custom_artifacts_are_normalized_at_construction(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"local-weights")
    local = VideoDepthEstimatorMoGe2(model_path=str(tmp_path), device="cpu")
    assert local.repo_id == str(checkpoint.resolve())
    assert local.revision is None

    remote = VideoDepthEstimatorMoGe2(model_path="owner/custom@0123456789abcdef", device="cpu")
    assert remote.repo_id == "owner/custom"
    assert remote.revision == "0123456789abcdef"

    invalid = tmp_path / "weights.pt"
    invalid.write_bytes(b"wrong-name")
    with pytest.raises(ValueError, match="model.pt"):
        VideoDepthEstimatorMoGe2(model_path=str(invalid), device="cpu")
    with pytest.raises(ValueError, match="repo_id@revision"):
        VideoDepthEstimatorMoGe2(model_path="owner/floating", device="cpu")


def test_moge_internal_remote_artifact_requires_a_complete_pair() -> None:
    with pytest.raises(ValueError, match="repository and revision"):
        VideoDepthEstimatorMoGe2(repo_id="owner/custom", revision=None, device="cpu")
    with pytest.raises(ValueError, match="repository and revision"):
        VideoDepthEstimatorMoGe2(repo_id=None, revision="abc", device="cpu")


def _install_fake_moge(monkeypatch, model_class: MagicMock) -> None:
    moge = ModuleType("moge")
    model = ModuleType("moge.model")
    v2 = ModuleType("moge.model.v2")
    v2.MoGeModel = model_class
    monkeypatch.setitem(sys.modules, "moge", moge)
    monkeypatch.setitem(sys.modules, "moge.model", model)
    monkeypatch.setitem(sys.modules, "moge.model.v2", v2)


def test_moge_loads_pinned_snapshot_model_file(monkeypatch, tmp_path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    checkpoint = snapshot / "model.pt"
    checkpoint.write_bytes(b"remote-weights")
    identity = "hf:owner/model@resolved"
    resolve = MagicMock(return_value=(str(snapshot), identity))
    monkeypatch.setattr(moge_module, "resolve_hf_snapshot", resolve)
    loaded_model = MagicMock()
    loaded_model.to.return_value = loaded_model
    model_class = MagicMock()
    model_class.from_pretrained.return_value = loaded_model
    _install_fake_moge(monkeypatch, model_class)
    estimator = VideoDepthEstimatorMoGe2(
        model_size="vitb",
        repo_id="owner/model",
        revision="0123456789abcdef",
        device="cpu",
    )

    assert estimator.load_model() is True

    resolve.assert_called_once_with("owner/model", revision="0123456789abcdef")
    model_class.from_pretrained.assert_called_once_with(str(checkpoint))
    loaded_model.to.assert_called_once_with(device="cpu")
    loaded_model.eval.assert_called_once_with()
    assert estimator.artifact_identity == identity
    assert estimator.checkpoint_path == checkpoint.resolve()


def test_moge_local_load_hashes_only_model_file(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"local-weights")
    loaded_model = MagicMock()
    loaded_model.to.return_value = loaded_model
    model_class = MagicMock()
    model_class.from_pretrained.return_value = loaded_model
    _install_fake_moge(monkeypatch, model_class)
    estimator = VideoDepthEstimatorMoGe2(model_path=str(checkpoint), device="cpu")

    assert estimator.load_model() is True

    expected_hash = hashlib.sha256(b"local-weights").hexdigest()
    assert estimator.artifact_identity == f"local:{expected_hash}"
    model_class.from_pretrained.assert_called_once_with(str(checkpoint.resolve()))


def test_moge_remote_snapshot_requires_model_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        moge_module,
        "resolve_hf_snapshot",
        MagicMock(return_value=(str(tmp_path), "hf:owner/model@resolved")),
    )
    with pytest.raises(FileNotFoundError, match="model.pt"):
        VideoDepthEstimatorMoGe2(
            repo_id="owner/model", revision="0123456789abcdef", device="cpu"
        ).load_model()


def test_moge_reports_only_stable_classified_model_info(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    estimator = VideoDepthEstimatorMoGe2(
        model_size="vitl", model_path=str(checkpoint), device="cuda:1"
    )
    estimator.artifact_identity = "local:abc"
    estimator.inference_precision = "float16"

    assert estimator.get_model_size() == "vitl"
    assert estimator.get_model_info() == {
        "family": "moge",
        "model_name": "vitl",
        "model_version": "MoGe-2",
        "repository": str(checkpoint.resolve()),
        "revision": None,
        "source_revision": MOGE_SOURCE_REVISION,
        "artifact_identity": "local:abc",
        "metric": True,
        "device": "cuda:1",
        "precision": "float16",
        "resolution_level": MOGE_RESOLUTION_LEVEL,
        "preprocessing_algorithm": MOGE_PREPROCESSING_ALGORITHM,
        "camera_model": "pinhole_fx",
        "inference_batch_size": 1,
    }


def test_moge_factory_constructs_without_loading_optional_dependency() -> None:
    estimator = create_video_depth_estimator_moge2(
        model_size="vits",
        repo_id="owner/model",
        revision="0123456789abcdef",
        device="cpu",
    )
    assert isinstance(estimator, VideoDepthEstimatorMoGe2)
    assert estimator.model is None
    assert "moge" not in estimator.__class__.__module__.split(".")[:-1]


def test_depth_package_exports_typed_camera_and_moge_adapter() -> None:
    assert depth_package.PinholeCameraBatch.__name__ == "PinholeCameraBatch"
    assert depth_package.VideoDepthEstimatorMoGe2 is VideoDepthEstimatorMoGe2
    assert depth_package.create_video_depth_estimator_moge2 is create_video_depth_estimator_moge2


def test_moge_unload_releases_model_and_cuda_cache(monkeypatch) -> None:
    estimator = loaded_fake_moge(device="cuda")
    empty_cache = MagicMock()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)

    estimator.unload_model()

    assert estimator.model is None
    empty_cache.assert_called_once_with()
