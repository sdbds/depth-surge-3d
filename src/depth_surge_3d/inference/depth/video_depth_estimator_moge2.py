"""Optional MoGe-2 metric-depth adapter."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .model_artifact import resolve_hf_snapshot
from .types import DepthBatch, DepthRepresentation, PinholeCameraBatch


MOGE_SOURCE_REVISION = "925b8ed835a7a9cdb7578ba15c658a0afc969030"
MOGE_RESOLUTION_LEVEL = 9
MOGE_PREPROCESSING_ALGORITHM = "moge2-rgb-area-max-edge-v1"
MOGE_INSTALL_COMMAND = "uv sync --extra moge2"

_DEFAULT_MODEL_ARTIFACTS = {
    "vits": (
        "Ruicheng/moge-2-vits-normal",
        "679230677b4d282c6f304189a93e98e14f085902",
    ),
    "vitb": (
        "Ruicheng/moge-2-vitb-normal",
        "54ad3a693e61907ea4633d13dec6ee682fa09419",
    ),
    "vitl": (
        "Ruicheng/moge-2-vitl",
        "39c4d5e957afe587e04eec59dc2bcc3be5ecd968",
    ),
}


def _scaled_shape(width: int, height: int, max_edge: int) -> tuple[int, int]:
    if width < 1 or height < 1 or max_edge < 1:
        raise ValueError("MoGe dimensions and depth resolution must be positive")
    scale = min(1.0, max_edge / max(width, height))
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    if (
        target_width > width
        or target_height > height
        or max(target_width, target_height) > max_edge
    ):
        raise ValueError("MoGe preprocessing dimensions violate the single-scale contract")
    return target_height, target_width


def _split_remote_artifact(value: str) -> tuple[str, str]:
    repo_id, separator, revision = value.rpartition("@")
    if not separator or not repo_id or not revision:
        raise ValueError("Remote MoGe model must use repo_id@revision")
    return repo_id, revision


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


class VideoDepthEstimatorMoGe2:
    """Estimate one metric depth frame and its normalized horizontal focal."""

    max_batch_size = 1
    camera_model = "pinhole_fx"

    def __init__(
        self,
        model_size: str = "vitb",
        model_path: str | None = None,
        repo_id: str | None = None,
        revision: str | None = None,
        device: str = "auto",
    ) -> None:
        self.model_size = model_size
        self.device = self._determine_device(device)
        self.metric = True
        self.model: Any | None = None
        self.artifact_identity: str | None = None
        self.checkpoint_path: Path | None = None
        self.inference_precision = "float16" if self.device.startswith("cuda") else "float32"

        if model_path is not None:
            self.repo_id, self.revision = self._normalize_custom_artifact(model_path)
        else:
            if (repo_id is None) != (revision is None):
                raise ValueError("MoGe repository and revision must be provided together")
            if repo_id is None:
                try:
                    default_repo, default_revision = _DEFAULT_MODEL_ARTIFACTS[model_size]
                except KeyError as error:
                    raise ValueError(f"Unknown MoGe-2 model size: {model_size}") from error
                repo_id = default_repo
                revision = default_revision
            self.repo_id = repo_id
            self.revision = revision

    @staticmethod
    def _determine_device(device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _normalize_custom_artifact(value: str) -> tuple[str, str | None]:
        candidate = Path(value).expanduser()
        if candidate.exists():
            checkpoint = candidate / "model.pt" if candidate.is_dir() else candidate
            if checkpoint.name != "model.pt":
                raise ValueError("Local MoGe model must be a model.pt file")
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Local MoGe checkpoint is missing: {checkpoint}")
            return str(checkpoint.resolve()), None
        return _split_remote_artifact(value)

    def estimate_output_shape(
        self,
        frame_width: int,
        frame_height: int,
        input_size: int,
    ) -> tuple[int, int]:
        return _scaled_shape(frame_width, frame_height, input_size)

    def _resolve_checkpoint(self) -> tuple[Path, str]:
        if self.revision is None:
            checkpoint = Path(self.repo_id).resolve()
            if checkpoint.name != "model.pt" or not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Local MoGe checkpoint must resolve to model.pt: {checkpoint}"
                )
            return checkpoint, f"local:{_hash_file(checkpoint)}"

        snapshot_path, identity = resolve_hf_snapshot(self.repo_id, revision=self.revision)
        checkpoint = Path(snapshot_path).resolve() / "model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"MoGe snapshot is missing model.pt: {checkpoint}")
        return checkpoint, identity

    def load_model(self) -> bool:
        """Resolve immutable weights and load the optional upstream model."""
        checkpoint_path, artifact_identity = self._resolve_checkpoint()
        try:
            from moge.model.v2 import MoGeModel
        except ImportError as error:
            raise RuntimeError(
                f"MoGe-2 optional dependency is not installed. "
                f"Install with: {MOGE_INSTALL_COMMAND}"
            ) from error

        self.checkpoint_path = checkpoint_path
        self.artifact_identity = artifact_identity
        self.model = MoGeModel.from_pretrained(str(checkpoint_path))
        self.model = self.model.to(device=self.device)
        self.model.eval()
        return True

    @staticmethod
    def _require_tensor(output: Mapping[str, Any], field: str, dtype: torch.dtype) -> torch.Tensor:
        if field not in output:
            raise ValueError(f"MoGe output is missing {field}")
        value = output[field]
        if not torch.is_tensor(value):
            raise TypeError(f"MoGe {field} output must be a tensor")
        if value.dtype != dtype:
            raise TypeError(f"MoGe {field} output must use {dtype}")
        return value

    @staticmethod
    def _validate_rank_and_frame_count(value: torch.Tensor, field: str) -> None:
        if value.ndim != 3:
            raise ValueError(f"MoGe {field} output rank must be 3")
        if value.shape[0] != 1:
            raise ValueError(f"MoGe {field} output frame count must be one")

    @classmethod
    def _validate_output(
        cls,
        output: Any,
        expected_height: int,
        expected_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(output, Mapping):
            raise TypeError("MoGe inference output must be a mapping")
        depth = cls._require_tensor(output, "depth", torch.float32)
        mask = cls._require_tensor(output, "mask", torch.bool)
        intrinsics = cls._require_tensor(output, "intrinsics", torch.float32)

        cls._validate_rank_and_frame_count(depth, "depth")
        cls._validate_rank_and_frame_count(mask, "mask")
        cls._validate_rank_and_frame_count(intrinsics, "intrinsics")
        if depth.shape[1:] != (expected_height, expected_width):
            raise ValueError("MoGe depth output has an unexpected spatial shape")
        if mask.shape[1:] != depth.shape[1:]:
            raise ValueError("MoGe mask output has an unexpected spatial shape")
        if intrinsics.shape != (1, 3, 3):
            raise ValueError("MoGe intrinsics output must have shape [1,3,3]")

        focal = intrinsics[:, 0, 0]
        if not torch.isfinite(focal).all() or torch.any(focal <= 0):
            raise ValueError("MoGe normalized focal must be finite and positive")
        return depth, mask, focal

    def _preprocess(self, frame: np.ndarray, max_edge: int) -> torch.Tensor:
        height, width = frame.shape[:2]
        target_height, target_width = _scaled_shape(width, height, max_edge)
        rgb = np.ascontiguousarray(frame[..., ::-1], dtype=np.float32)
        rgb /= 255.0
        if (target_height, target_width) != (height, width):
            rgb = cv2.resize(
                rgb,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )
        image = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        return image.unsqueeze(0)

    def estimate_depth_batch(
        self,
        frames: np.ndarray,
        target_fps: int = 30,
        input_size: int | None = None,
        fp32: bool = False,
    ) -> DepthBatch:
        """Infer a single frame without adaptive fallback or retained point maps."""
        del target_fps
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if not isinstance(frames, np.ndarray):
            raise TypeError("frames must be a numpy array")
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("frames must have shape [N, H, W, 3]")
        if frames.dtype != np.uint8:
            raise TypeError("MoGe frames must use uint8 BGR values")
        if len(frames) != 1:
            raise ValueError("MoGe-2 accepts exactly one frame")

        height, width = int(frames.shape[1]), int(frames.shape[2])
        max_edge = max(height, width) if input_size is None else int(input_size)
        image = self._preprocess(frames[0], max_edge)
        output_height, output_width = image.shape[-2:]
        use_fp16 = self.device.startswith("cuda") and not fp32
        precision = "float16" if use_fp16 else "float32"
        self.inference_precision = precision
        try:
            with torch.inference_mode():
                output = self.model.infer(
                    image,
                    force_projection=False,
                    apply_mask=True,
                    resolution_level=MOGE_RESOLUTION_LEVEL,
                    use_fp16=use_fp16,
                )
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError(
                "MoGe-2 CUDA inference ran out of memory: "
                f"model_size={self.model_size}, input={width}x{height}, "
                f"resolution_level={MOGE_RESOLUTION_LEVEL}, "
                f"precision={precision}, device={self.device}"
            ) from error

        depth, mask, focal = self._validate_output(output, int(output_height), int(output_width))
        depth_values = depth.detach().cpu().numpy().copy()
        mask_values = mask.detach().cpu().numpy()
        depth_values[~mask_values] = np.inf
        focal_values = focal.detach().cpu().numpy().copy()
        return DepthBatch(
            depth_values,
            DepthRepresentation.METRIC_DEPTH,
            camera=PinholeCameraBatch(focal_values),
        )

    def get_model_size(self) -> str:
        return self.model_size

    def get_model_info(self) -> dict[str, Any]:
        return {
            "family": "moge",
            "model_name": self.model_size,
            "model_version": "MoGe-2",
            "repository": self.repo_id,
            "revision": self.revision,
            "source_revision": MOGE_SOURCE_REVISION,
            "artifact_identity": self.artifact_identity,
            "metric": True,
            "device": self.device,
            "precision": self.inference_precision,
            "resolution_level": MOGE_RESOLUTION_LEVEL,
            "preprocessing_algorithm": MOGE_PREPROCESSING_ALGORITHM,
            "camera_model": self.camera_model,
            "inference_batch_size": self.max_batch_size,
        }

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_video_depth_estimator_moge2(
    model_size: str = "vitb",
    model_path: str | None = None,
    repo_id: str | None = None,
    revision: str | None = None,
    device: str = "auto",
) -> VideoDepthEstimatorMoGe2:
    return VideoDepthEstimatorMoGe2(
        model_size=model_size,
        model_path=model_path,
        repo_id=repo_id,
        revision=revision,
        device=device,
    )
