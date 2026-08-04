"""Anime-focused depth estimation using the See-Through Marigold model."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .attention_backend import install_diffusers_flash_attention
from .types import DepthBatch, DepthRepresentation


DEFAULT_SEE_THROUGH_REPO = "24yearsold/seethroughv0.0.1_marigold"
DEFAULT_PROCESSING_RESOLUTION = 768
DEFAULT_DENOISING_STEPS = 4
DEFAULT_SEED = 1026
QINGLONG_CAPTIONS_ENV = "QINGLONG_CAPTIONS_ROOT"

_VENDOR_MARKER = Path("module/see_through/vendor/modules/marigold/marigold_depth_pipeline.py")

PipelineLoader = Callable[..., Any]


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

    common_kwargs: dict[str, Any] = {
        "cache_dir": str(cache_dir),
        "torch_dtype": dtype,
    }
    unet = UNetFrameConditionModel.from_pretrained(
        repo_id,
        subfolder="unet",
        **common_kwargs,
    )
    pipeline = MarigoldDepthPipeline.from_pretrained(
        repo_id,
        unet=unet,
        **common_kwargs,
    )
    pipeline.to(device=device, dtype=dtype)
    if hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(disable=True)
    pipeline.cache_tag_embeds()
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

    def load_model(self) -> bool:
        """Load the specialized Marigold pipeline and its frame-conditioned UNet."""
        try:
            source_root = resolve_qinglong_captions_root(self.source_root)
            cache_dir = (source_root / "huggingface" / "hub").resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
            dtype = self._resolve_dtype()
            flash_attention_modules = install_diffusers_flash_attention()
            print(f"Loading See-Through Marigold model: {self.repo_id}")
            self.model = self.pipeline_loader(
                repo_id=self.repo_id,
                source_root=source_root,
                cache_dir=cache_dir,
                device=self.device,
                dtype=dtype,
            )
            print(
                "Loaded See-Through Marigold "
                f"on {self.device} ({dtype}, {self.processing_resolution}px, "
                f"{self.denoising_steps} denoising steps)"
            )
            if flash_attention_modules:
                print("FlashAttention 2 enabled for See-Through (SDPA fallback active)")
            else:
                print("FlashAttention 2 unavailable for See-Through - using PyTorch SDPA")
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

    def estimate_depth_batch(
        self,
        frames: np.ndarray,
        target_fps: int = 30,
        input_size: int = DEFAULT_PROCESSING_RESOLUTION,
        fp32: bool = False,
    ) -> DepthBatch:
        """Estimate relative depth sequentially with deterministic diffusion noise."""
        del target_fps, fp32
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("frames must have shape [N, H, W, 3]")

        processing_res = min(max(int(input_size), 1), self.processing_resolution)
        depth_maps: list[np.ndarray] = []
        for index, frame in enumerate(frames):
            import cv2

            rgb_frame = np.ascontiguousarray(frame[..., ::-1])
            interpolation = (
                cv2.INTER_AREA if max(frame.shape[:2]) > processing_res else cv2.INTER_LINEAR
            )
            resized_rgb = cv2.resize(
                rgb_frame,
                (processing_res, processing_res),
                interpolation=interpolation,
            )
            alpha = np.full((processing_res, processing_res, 1), 255, dtype=np.uint8)
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

        return DepthBatch(np.stack(depth_maps), DepthRepresentation.RELATIVE_DEPTH)

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
            "source": "qinglong-captions/module/see_through",
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
