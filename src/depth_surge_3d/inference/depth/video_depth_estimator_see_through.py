"""Anime-focused depth estimation using the See-Through Marigold model."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .model_artifact import resolve_hf_snapshot
from .types import DepthBatch, DepthRepresentation


DEFAULT_SEE_THROUGH_REPO = "24yearsold/seethroughv0.0.1_marigold"
DEFAULT_PROCESSING_RESOLUTION = 768
DEFAULT_DENOISING_STEPS = 4
DEFAULT_SEED = 1026
VAE_SCALE_FACTOR = 8
QINGLONG_CAPTIONS_ENV = "QINGLONG_CAPTIONS_ROOT"

_VENDOR_MARKER = Path("module/see_through/vendor/modules/marigold/marigold_depth_pipeline.py")

PipelineLoader = Callable[..., Any]
BatchRunner = Callable[..., np.ndarray]


def _run_opaque_marigold_batch(
    *,
    pipeline: Any,
    rgb_frames: np.ndarray,
    denoising_steps: int,
    seed: int,
) -> np.ndarray:
    """Run independent opaque frames as one CUDA batch."""
    if rgb_frames.ndim != 4 or rgb_frames.shape[-1] != 3:
        raise ValueError("rgb_frames must have shape [N, H, W, 3]")

    with torch.no_grad():
        rgb_values = np.asarray(rgb_frames, dtype=np.float32) / np.float32(255.0)
        rgb_tensor = torch.from_numpy(rgb_values).permute(0, 3, 1, 2)
        rgb_tensor = rgb_tensor.to(device=pipeline.vae.device, dtype=pipeline.vae.dtype)
        rgb_latent = pipeline.encode_rgb(rgb_tensor * 2.0 - 1.0)

        # The sole opaque layer and full-page condition are the same image.
        condition_latent = torch.cat((rgb_latent, rgb_latent), dim=1)
        batch_size, _, latent_height, latent_width = condition_latent.shape
        device = pipeline.unet.device
        generator = torch.Generator(device=device).manual_seed(seed)
        initial_noise = torch.randn(
            (1, 4, latent_height, latent_width),
            device=device,
            dtype=pipeline.unet.dtype,
            generator=generator,
        )
        target_latent = initial_noise.expand(batch_size, -1, -1, -1).clone()

        pipeline.scheduler.set_timesteps(denoising_steps, device=device)
        if pipeline.empty_text_embed is None:
            pipeline.encode_empty_text()
        text_embedding = pipeline.empty_text_embed.repeat((batch_size, 1, 1)).to(
            device=device,
            dtype=target_latent.dtype,
        )

        for timestep in pipeline.scheduler.timesteps:
            unet_input = torch.cat((condition_latent, target_latent), dim=1).unsqueeze(1)
            noise_prediction = pipeline.unet(
                unet_input,
                timestep,
                encoder_hidden_states=text_embedding,
            ).sample[:, 0]
            target_latent = pipeline.scheduler.step(
                noise_prediction,
                timestep,
                target_latent,
                generator=generator,
            ).prev_sample

        depth = torch.cat(
            [pipeline.decode_depth(value.unsqueeze(0)) for value in target_latent],
            dim=0,
        )
        depth = ((torch.clip(depth, -1.0, 1.0) + 1.0) / 2.0).squeeze(1)
        return depth.to(device="cpu", dtype=torch.float32).numpy()


def _is_qinglong_captions_root(path: Path) -> bool:
    return (path / _VENDOR_MARKER).is_file()


def resolve_qinglong_captions_root(source_root: str | Path | None = None) -> Path:
    """Locate the qinglong-captions checkout that owns the See-Through adapter."""
    if source_root is not None:
        explicit = Path(source_root).expanduser().resolve()
        if _is_qinglong_captions_root(explicit):
            return explicit
        raise FileNotFoundError(
            f"See-Through source not found under {explicit}. " f"Expected {_VENDOR_MARKER}."
        )

    env_root = os.environ.get(QINGLONG_CAPTIONS_ENV)
    if env_root:
        configured = Path(env_root).expanduser().resolve()
        if _is_qinglong_captions_root(configured):
            return configured
        raise FileNotFoundError(
            f"{QINGLONG_CAPTIONS_ENV} points to {configured}, but {_VENDOR_MARKER} is missing."
        )

    project_root = Path(__file__).resolve().parents[4]
    candidates = [
        project_root.parent / "qinglong-captions",
        Path.cwd().resolve().parent / "qinglong-captions",
    ]
    for candidate in candidates:
        candidate = candidate.resolve()
        if _is_qinglong_captions_root(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not locate qinglong-captions See-Through sources. "
        f"Set {QINGLONG_CAPTIONS_ENV} to the qinglong-captions repository root."
    )


def _load_see_through_pipeline(
    *,
    repo_id: str,
    source_root: Path,
    cache_dir: Path,
    device: str,
    dtype: torch.dtype,
) -> Any:
    source_path = str(source_root)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)

    from module.see_through.vendor.modules.layerdiffuse.layerdiff3d import (
        UNetFrameConditionModel,
    )
    from module.see_through.vendor.modules.marigold import MarigoldDepthPipeline

    resolved_repo, artifact_identity = resolve_hf_snapshot(repo_id, cache_dir=cache_dir)
    common_kwargs: dict[str, Any] = {
        "cache_dir": str(cache_dir),
        "torch_dtype": dtype,
    }
    unet = UNetFrameConditionModel.from_pretrained(
        resolved_repo,
        subfolder="unet",
        **common_kwargs,
    )
    pipeline = MarigoldDepthPipeline.from_pretrained(
        resolved_repo,
        unet=unet,
        **common_kwargs,
    )
    pipeline.to(device=device, dtype=dtype)
    if hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(disable=True)
    pipeline.cache_tag_embeds()
    pipeline._depth_surge_artifact_identity = artifact_identity
    return pipeline


class SeeThroughDepthEstimator:
    """Wrap the See-Through Marigold checkpoint as a frame depth backend."""

    model_type = "see_through"
    max_batch_size = 1

    def __init__(
        self,
        repo_id: str = DEFAULT_SEE_THROUGH_REPO,
        device: str = "auto",
        metric: bool = False,
        processing_resolution: int = DEFAULT_PROCESSING_RESOLUTION,
        denoising_steps: int = DEFAULT_DENOISING_STEPS,
        seed: int = DEFAULT_SEED,
        source_root: str | Path | None = None,
        pipeline_loader: PipelineLoader | None = None,
        verbose: bool = False,
    ) -> None:
        self.repo_id = repo_id
        self.device = self._determine_device(device)
        self.metric = False
        self.processing_resolution = int(processing_resolution)
        self.denoising_steps = int(denoising_steps)
        self.seed = int(seed)
        self.source_root = source_root
        self.pipeline_loader = pipeline_loader or _load_see_through_pipeline
        self.verbose = verbose
        self.model: Any | None = None
        self.artifact_identity: str | None = None
        self.max_batch_size = self._recommended_batch_size()
        self.batch_runner: BatchRunner | None = (
            _run_opaque_marigold_batch if self.device.startswith("cuda") else None
        )

    @property
    def inference_precision(self) -> str:
        """Report the same dtype selected by the model loader."""
        return str(self._resolve_dtype()).removeprefix("torch.")

    def _recommended_batch_size(self) -> int:
        if not self.device.startswith("cuda"):
            return 1
        try:
            total_gib = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
        except (AssertionError, RuntimeError, TypeError):
            return 1
        if total_gib >= 16:
            return 4
        if total_gib >= 10:
            return 2
        return 1

    @staticmethod
    def _determine_device(device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dtype(self) -> torch.dtype:
        if self.device.startswith("cuda"):
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    def estimate_output_shape(
        self,
        frame_width: int,
        frame_height: int,
        input_size: int,
    ) -> tuple[int, int]:
        """Return native 768 square geometry or an aspect-preserving aligned shape."""

        if frame_width < 1 or frame_height < 1:
            raise ValueError("Frame dimensions must be positive")
        requested_edge = max(int(input_size), 1)
        if requested_edge == DEFAULT_PROCESSING_RESOLUTION:
            return DEFAULT_PROCESSING_RESOLUTION, DEFAULT_PROCESSING_RESOLUTION
        processing_edge = max(
            VAE_SCALE_FACTOR,
            int(round(requested_edge / VAE_SCALE_FACTOR)) * VAE_SCALE_FACTOR,
        )

        if frame_width >= frame_height:
            processing_width = processing_edge
            scaled_height = processing_edge * frame_height / frame_width
            processing_height = max(
                VAE_SCALE_FACTOR,
                int(round(scaled_height / VAE_SCALE_FACTOR)) * VAE_SCALE_FACTOR,
            )
        else:
            processing_height = processing_edge
            scaled_width = processing_edge * frame_width / frame_height
            processing_width = max(
                VAE_SCALE_FACTOR,
                int(round(scaled_width / VAE_SCALE_FACTOR)) * VAE_SCALE_FACTOR,
            )
        return processing_height, processing_width

    def load_model(self) -> bool:
        """Load the specialized Marigold pipeline and its frame-conditioned UNet."""
        try:
            source_root = resolve_qinglong_captions_root(self.source_root)
            cache_dir = (source_root / "huggingface" / "hub").resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
            dtype = self._resolve_dtype()
            print(f"Loading See-Through Marigold model: {self.repo_id}")
            self.model = self.pipeline_loader(
                repo_id=self.repo_id,
                source_root=source_root,
                cache_dir=cache_dir,
                device=self.device,
                dtype=dtype,
            )
            artifact_identity = getattr(self.model, "_depth_surge_artifact_identity", None)
            if isinstance(artifact_identity, str):
                self.artifact_identity = artifact_identity
            print(
                "Loaded See-Through Marigold "
                f"on {self.device} ({dtype}, default {self.processing_resolution}px, "
                f"{self.denoising_steps} denoising steps)"
            )
            print("PyTorch SDPA enabled for See-Through")
            return True
        except ImportError as exc:
            print(f"Error: See-Through dependencies are not installed: {exc}")
            print("Install diffusers>=0.35.1 and accelerate, then retry.")
            return False
        except Exception as exc:
            print(f"Error loading See-Through Marigold: {exc}")
            return False

    def _make_generator(self) -> torch.Generator:
        generator_device = "cuda" if self.device.startswith("cuda") else "cpu"
        return torch.Generator(device=generator_device).manual_seed(self.seed)

    def _microbatch_size(self, processing_height: int, processing_width: int) -> int:
        reference_pixels = 1080 * 608
        pixel_ratio = max(1.0, (processing_height * processing_width) / reference_pixels)
        return max(1, min(self.max_batch_size, int(self.max_batch_size / pixel_ratio)))

    @staticmethod
    def _resize_rgb_frames(
        frames: np.ndarray,
        processing_height: int,
        processing_width: int,
    ) -> np.ndarray:
        import cv2

        resized_frames = []
        for frame in frames:
            rgb_frame = np.ascontiguousarray(frame[..., ::-1])
            interpolation = (
                cv2.INTER_AREA
                if max(frame.shape[:2]) > max(processing_height, processing_width)
                else cv2.INTER_LINEAR
            )
            resized_frames.append(
                cv2.resize(
                    rgb_frame,
                    (processing_width, processing_height),
                    interpolation=interpolation,
                )
            )
        return np.stack(resized_frames)

    def _estimate_cuda_batches(
        self,
        frames: np.ndarray,
        processing_height: int,
        processing_width: int,
    ) -> np.ndarray:
        if self.model is None or self.batch_runner is None:
            raise RuntimeError("CUDA batch runner is unavailable")

        batch_size = self._microbatch_size(processing_height, processing_width)
        depth_batches = []
        for offset in range(0, len(frames), batch_size):
            rgb_frames = self._resize_rgb_frames(
                frames[offset : offset + batch_size],
                processing_height,
                processing_width,
            )
            depth_batches.append(
                self.batch_runner(
                    pipeline=self.model,
                    rgb_frames=rgb_frames,
                    denoising_steps=self.denoising_steps,
                    seed=self.seed,
                )
            )
        return np.concatenate(depth_batches, axis=0).astype(np.float32, copy=False)

    def _estimate_sequential(
        self,
        frames: np.ndarray,
        processing_height: int,
        processing_width: int,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        depth_maps: list[np.ndarray] = []
        rgb_frames = self._resize_rgb_frames(frames, processing_height, processing_width)
        for index, resized_rgb in enumerate(rgb_frames):
            alpha = np.full((processing_height, processing_width, 1), 255, dtype=np.uint8)
            full_frame_rgba = np.concatenate((resized_rgb, alpha), axis=2)
            output = self.model(
                img_list=[full_frame_rgba],
                denoising_steps=self.denoising_steps,
                ensemble_size=1,
                match_input_res=True,
                generator=self._make_generator(),
                color_map=None,
                show_progress_bar=False,
            )
            depth_value = getattr(output, "depth_tensor", None)
            if depth_value is None:
                depth_value = getattr(output, "depth_np", None)
            if depth_value is None:
                raise RuntimeError("See-Through pipeline returned no depth map")
            if torch.is_tensor(depth_value):
                depth = depth_value.detach().to(device="cpu", dtype=torch.float32).numpy()
            else:
                depth = np.asarray(depth_value, dtype=np.float32)
            depth = np.squeeze(depth)
            if depth.ndim != 2:
                raise RuntimeError(f"Unexpected See-Through depth shape: {depth.shape}")
            depth_maps.append(depth.astype(np.float32, copy=False))
            if self.verbose:
                print(f"  See-Through depth frame {index + 1}/{len(frames)}")
        return np.stack(depth_maps)

    def estimate_depth_batch(
        self,
        frames: np.ndarray,
        target_fps: int = 30,
        input_size: int | None = None,
        fp32: bool = False,
    ) -> DepthBatch:
        """Estimate relative depth with deterministic per-frame diffusion noise."""
        del target_fps, fp32
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("frames must have shape [N, H, W, 3]")

        processing_res = (
            self.processing_resolution if input_size is None else max(int(input_size), 1)
        )
        processing_height, processing_width = self.estimate_output_shape(
            int(frames.shape[2]),
            int(frames.shape[1]),
            processing_res,
        )
        if self.batch_runner is not None:
            depth_maps = self._estimate_cuda_batches(
                frames,
                processing_height,
                processing_width,
            )
        else:
            depth_maps = self._estimate_sequential(
                frames,
                processing_height,
                processing_width,
            )
        return DepthBatch(depth_maps, DepthRepresentation.RELATIVE_DEPTH)

    def get_model_size(self) -> str:
        return "large"

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_name": self.repo_id,
            "model_version": "See-Through Marigold",
            "device": self.device,
            "metric": False,
            "loaded": self.model is not None,
            "temporal_consistency": False,
            "processing_resolution": self.processing_resolution,
            "denoising_steps": self.denoising_steps,
            "seed": self.seed,
            "inference_batch_size": self.max_batch_size,
            "source": "qinglong-captions/module/see_through",
            "artifact_identity": self.artifact_identity,
        }

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_see_through_depth_estimator(
    model_path: str | None = None,
    device: str = "auto",
    metric: bool = False,
) -> SeeThroughDepthEstimator:
    """Create the experimental anime-focused See-Through estimator."""
    return SeeThroughDepthEstimator(
        repo_id=model_path or DEFAULT_SEE_THROUGH_REPO,
        device=device,
        metric=metric,
    )
