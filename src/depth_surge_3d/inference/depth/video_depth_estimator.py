"""Video depth estimation model management using Video-Depth-Anything.

This module handles loading and interfacing with the Video-Depth-Anything model,
which provides temporal consistency for video depth estimation.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any
import cv2
import torch
import torch.nn.functional as F
import numpy as np

from ...utils.system.console import success as console_success, error as console_error
from .types import DepthBatch, DepthRepresentation
from .v2_temporal_contract import (
    VDA_FRAME_STEP,
    VDA_INFERENCE_ALGORITHM,
    VDA_INFER_LEN,
    VDA_INTERP_LEN,
    VDA_KEYFRAMES,
    VDA_OVERLAP,
)
from ...core.constants import (
    DEFAULT_MODEL_PATH,
    VIDEO_DEPTH_ANYTHING_REPO_DIR,
    MODEL_CONFIGS,
    MODEL_DOWNLOAD_URLS,
    DEPTH_MODEL_INPUT_SIZE,
    DEPTH_MODEL_DEFAULT_FPS,
)


class _VDASequenceDepthIterator(Iterator[tuple[int, DepthBatch]]):
    """One retryable, bounded implementation of VDA's offline window loop."""

    def __init__(
        self,
        estimator: "VideoDepthEstimator",
        frame_count: int,
        load_frames: Callable[[Sequence[int]], np.ndarray],
        *,
        target_fps: int,
        input_size: int,
        fp32: bool,
    ) -> None:
        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 0:
            raise ValueError("frame_count must be a non-negative integer")
        if not callable(load_frames):
            raise TypeError("load_frames must be callable")
        self._estimator = estimator
        self._frame_count = frame_count
        self._load_frames = load_frames
        self._target_fps = target_fps
        self._input_size = input_size
        self._fp32 = fp32
        self._representation = (
            DepthRepresentation.METRIC_DEPTH
            if estimator.metric
            else DepthRepresentation.INVERSE_DEPTH
        )
        self._source_geometry: tuple[int, int] | None = None
        self._transform: Any = None
        self._retained_input: torch.Tensor | None = None
        self._alignment_refs: np.ndarray | None = None
        self._pending_depth: np.ndarray | None = None
        self._pending_start = 0
        self._next_window_start = 0
        self._initialized = False
        self._pending_emitted = False
        self._finished = False

    def __iter__(self) -> "_VDASequenceDepthIterator":
        return self

    def __next__(self) -> tuple[int, DepthBatch]:
        if self._finished or self._frame_count == 0:
            self._finished = True
            raise StopIteration

        while True:
            if not self._initialized:
                return self._advance_first_window()
            if self._next_window_start < self._frame_count:
                finalized = self._advance_later_window(self._next_window_start)
                if finalized is not None:
                    return finalized
                continue
            if not self._pending_emitted:
                self._pending_emitted = True
                visible_count = max(
                    0,
                    min(VDA_INTERP_LEN, self._frame_count - self._pending_start),
                )
                if visible_count:
                    assert self._pending_depth is not None
                    return self._batch(
                        self._pending_start,
                        self._pending_depth[:visible_count],
                    )
            self.close()
            raise StopIteration

    def close(self) -> None:
        """Release retained temporal state when iteration ends or is abandoned."""

        self._finished = True
        self._release_state()

    def _load(self, indexes: Sequence[int]) -> np.ndarray:
        values = self._load_frames(indexes)
        if not isinstance(values, np.ndarray):
            raise TypeError("Loader must return a numpy array")
        if len(values) != len(indexes):
            raise ValueError("Loader returned an unexpected frame count")
        if values.dtype != np.uint8:
            raise TypeError("Loader must return uint8 BGR frames")
        if values.ndim != 4:
            raise ValueError("Loader must return a rank-4 frame array")
        if values.shape[-1] != 3:
            raise ValueError("Loader frames must have exactly 3 channels")
        geometry = (int(values.shape[1]), int(values.shape[2]))
        if self._source_geometry is None:
            self._source_geometry = geometry
        elif geometry != self._source_geometry:
            raise ValueError("Loader frame geometry changed within one shot")
        return values

    def _empty_frames(self) -> np.ndarray:
        assert self._source_geometry is not None
        height, width = self._source_geometry
        return np.empty((0, height, width, 3), dtype=np.uint8)

    def _run_window(
        self,
        frames: np.ndarray,
        *,
        carried_input: torch.Tensor | None,
        padding_input: torch.Tensor | None,
    ) -> tuple[np.ndarray, torch.Tensor, Any]:
        assert self._source_geometry is not None
        result = self._estimator._infer_fixed_window(
            frames,
            input_size=self._input_size,
            fp32=self._fp32,
            carried_input=carried_input,
            padding_input=padding_input,
            transform=self._transform,
            output_shape=self._source_geometry,
        )
        if not isinstance(result, tuple) or len(result) != 3:
            raise TypeError("Fixed VDA window must return depth, input state, and transform")
        depths, current_input, transform = result
        values = np.asarray(depths, dtype=np.float32)
        expected_shape = (VDA_INFER_LEN, *self._source_geometry)
        if values.shape != expected_shape:
            raise ValueError(
                f"Fixed VDA window returned shape {values.shape}, expected {expected_shape}"
            )
        if not isinstance(current_input, torch.Tensor) or current_input.ndim != 5:
            raise TypeError("Fixed VDA window input state must be a rank-5 tensor")
        if current_input.shape[0] != 1 or current_input.shape[1] != VDA_INFER_LEN:
            raise ValueError("Fixed VDA window input state must have shape [1,32,C,H,W]")
        return values, current_input, transform

    @staticmethod
    def _select_retained_input(current_input: torch.Tensor) -> torch.Tensor:
        return current_input[:, list(VDA_KEYFRAMES), ...].detach()

    @staticmethod
    def _refresh_retained_input(
        current_input: torch.Tensor,
        retained_input: torch.Tensor,
    ) -> torch.Tensor:
        """Reuse the ten-frame state buffer instead of allocating a second one."""
        retained_input[:, 0].copy_(current_input[:, VDA_KEYFRAMES[0]])
        retained_input[:, 1].copy_(current_input[:, VDA_KEYFRAMES[1]])
        retained_input[:, 2:].copy_(current_input[:, VDA_KEYFRAMES[2] : VDA_KEYFRAMES[-1] + 1])
        return retained_input.detach()

    def _advance_first_window(self) -> tuple[int, DepthBatch]:
        indexes = list(range(min(VDA_INFER_LEN, self._frame_count)))
        frames = self._load(indexes)
        depths, current_input, transform = self._run_window(
            frames,
            carried_input=None,
            padding_input=None,
        )

        retained = self._select_retained_input(current_input)
        references = depths[[VDA_KEYFRAMES[0], VDA_KEYFRAMES[1]]].copy()
        pending = depths[VDA_INFER_LEN - VDA_INTERP_LEN :].copy()
        self._transform = transform
        self._retained_input = retained
        self._alignment_refs = references
        self._pending_depth = pending
        self._pending_start = VDA_INFER_LEN - VDA_INTERP_LEN
        self._next_window_start = VDA_FRAME_STEP
        self._initialized = True

        visible_count = min(VDA_INFER_LEN - VDA_INTERP_LEN, self._frame_count)
        return self._batch(0, depths[:visible_count])

    def _advance_later_window(self, window_start: int) -> tuple[int, DepthBatch] | None:
        new_start = window_start + VDA_OVERLAP
        indexes = list(range(new_start, min(window_start + VDA_INFER_LEN, self._frame_count)))
        frames = self._load(indexes) if indexes else self._empty_frames()
        assert self._retained_input is not None
        carried_input = self._retained_input
        padding_input = carried_input[:, -1:, ...]
        depths, current_input, transform = self._run_window(
            frames,
            carried_input=carried_input,
            padding_input=padding_input,
        )
        assert self._pending_depth is not None
        assert self._alignment_refs is not None
        finalized, next_pending, next_reference = self._estimator._finalize_window(
            self._pending_depth,
            depths,
            self._alignment_refs,
        )

        retained = self._refresh_retained_input(current_input, carried_input)
        self._alignment_refs[1] = next_reference
        self._transform = transform
        self._retained_input = retained
        self._pending_depth = next_pending
        self._pending_start = window_start + VDA_INFER_LEN - VDA_INTERP_LEN
        self._next_window_start = window_start + VDA_FRAME_STEP

        finalized_start = window_start + (VDA_OVERLAP - VDA_INTERP_LEN)
        visible_count = max(
            0,
            min(len(finalized), self._frame_count - finalized_start),
        )
        if not visible_count:
            return None
        return self._batch(finalized_start, finalized[:visible_count])

    def _batch(self, start: int, values: np.ndarray) -> tuple[int, DepthBatch]:
        return start, DepthBatch(np.asarray(values, dtype=np.float32), self._representation)

    def _release_state(self) -> None:
        self._transform = None
        self._retained_input = None
        self._alignment_refs = None
        self._pending_depth = None


class VideoDepthEstimator:
    """Handles video depth estimation using Video-Depth-Anything models."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        metric: bool = False,
        temporal_window_overlap: int = 10,
    ):
        self.model_path = model_path
        self.device = self._determine_device(device)
        self.metric = metric
        self.temporal_window_overlap = temporal_window_overlap
        self.model = None
        self.model_config: dict[str, object] | None = None

    @property
    def inference_precision(self) -> str:
        """Report the dtype used by the fp32=False inference path."""
        return "float16" if self.device == "cuda" else "float32"

    def _determine_device(self, device: str) -> str:
        """Determine the best device to use for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def load_model(self) -> bool:
        """
        Load the video depth estimation model.

        Returns:
            True if model loaded successfully
        """
        try:
            # Ensure dependencies are available
            if not self._ensure_dependencies():
                return False

            # Determine model type from path
            model_type = self._get_model_type(self.model_path)
            if not model_type:
                print(f"Cannot determine model type from path: {self.model_path}")
                return False

            model_config = MODEL_CONFIGS[model_type]
            self.model_config = model_config

            # Import and load model
            repo_path = Path(VIDEO_DEPTH_ANYTHING_REPO_DIR)
            if str(repo_path) not in sys.path:
                sys.path.insert(0, str(repo_path))

            from video_depth_anything.video_depth import VideoDepthAnything

            self.model = VideoDepthAnything(**model_config, metric=self.metric)

            # Load state dict and fix key names if needed
            state_dict = torch.load(self.model_path, map_location="cpu")

            # Remap depth_head.* to head.* for compatibility
            fixed_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("depth_head."):
                    new_key = key.replace("depth_head.", "head.")
                    fixed_state_dict[new_key] = value
                else:
                    fixed_state_dict[key] = value

            self.model.load_state_dict(fixed_state_dict, strict=True)  # type: ignore[attr-defined]
            self.model = self.model.to(self.device).eval()  # type: ignore[attr-defined]

            model_variant = "Metric-" if self.metric else ""
            print(f"Loaded {model_variant}Video-Depth-Anything ({model_type}) on {self.device}")
            return True

        except Exception as e:
            print(f"Error loading video model: {e}")
            print("Try downloading the model manually from:")
            print(f"  {MODEL_DOWNLOAD_URLS.get(model_type, 'Unknown')}")  # type: ignore[arg-type]
            return False

    def _ensure_dependencies(self) -> bool:
        """Ensure model file and repository are available."""
        # Check model file
        if not os.path.exists(self.model_path):
            print(f"Model not found at {self.model_path}")
            if not self._auto_download_model():
                return False

        # Check repository
        repo_path = Path(VIDEO_DEPTH_ANYTHING_REPO_DIR)
        if not repo_path.exists():
            print("Video-Depth-Anything repository not found")
            print("Please ensure the repository is cloned to the vendor directory:")
            print(
                f"  mkdir -p vendor && git clone https://github.com/DepthAnything/Video-Depth-Anything.git {VIDEO_DEPTH_ANYTHING_REPO_DIR}"
            )
            return False

        return True

    def _auto_download_model(self) -> bool:
        """Auto-download the model if missing."""
        model_path = Path(self.model_path)
        if model_path.exists():
            return True

        print("Attempting to download video model automatically...")

        # Create model directory
        model_dir = model_path.parent
        model_dir.mkdir(parents=True, exist_ok=True)

        # Determine download URL
        model_type = self._get_model_type(self.model_path)
        if not model_type or model_type not in MODEL_DOWNLOAD_URLS:
            print("Cannot determine model download URL")
            return False

        download_url = MODEL_DOWNLOAD_URLS[model_type]

        try:
            print(f"Downloading video model to {self.model_path}...")
            urllib.request.urlretrieve(download_url, self.model_path)
            print(console_success("Video model downloaded successfully"))
            return True
        except Exception as e:
            print(console_error(f"Auto-download failed: {e}"))
            print(f"Please download manually from: {download_url}")
            return False

    def _get_model_type(self, model_path: str) -> str | None:
        """Determine model type from file path."""
        path_str = str(model_path).lower()

        if "vits" in path_str:
            return "vits"
        elif "vitb" in path_str:
            return "vitb"
        elif "vitl" in path_str:
            return "vitl"

        # Fallback to large model
        return "vitl"

    def estimate_depth_batch(
        self,
        frames: np.ndarray,
        target_fps: int = DEPTH_MODEL_DEFAULT_FPS,
        input_size: int = DEPTH_MODEL_INPUT_SIZE,
        fp32: bool = False,
    ) -> DepthBatch:
        """
        Estimate depth for one complete in-memory array supplied by the caller.

        Args:
            frames: Input frames array (shape: [N, H, W, 3], BGR format)
            target_fps: Target frame rate for processing
            input_size: Input size for the model (default: DEPTH_MODEL_INPUT_SIZE)
            fp32: Use FP32 instead of FP16 (slower but more accurate)

        Returns:
            Native-resolution float32 depth values with explicit representation
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        values = self._estimate_depth_single_batch(frames, target_fps, input_size, fp32)
        representation = (
            DepthRepresentation.METRIC_DEPTH if self.metric else DepthRepresentation.INVERSE_DEPTH
        )
        return DepthBatch(np.asarray(values, dtype=np.float32), representation)

    def iter_sequence_depth(
        self,
        frame_count: int,
        load_frames: Callable[[Sequence[int]], np.ndarray],
        *,
        target_fps: int,
        input_size: int,
        fp32: bool,
    ) -> Iterator[tuple[int, DepthBatch]]:
        """Return bounded VDA offline inference for one shot-local frame sequence."""

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return _VDASequenceDepthIterator(
            self,
            frame_count,
            load_frames,
            target_fps=target_fps,
            input_size=input_size,
            fp32=fp32,
        )

    def _estimate_depth_single_batch(
        self, frames: np.ndarray, target_fps: int, input_size: int, fp32: bool
    ) -> np.ndarray:
        """Process all frames in a single batch."""
        try:
            # Convert BGR to RGB
            frames_rgb = frames[..., ::-1].copy()

            # Suppress tqdm output from Video-Depth-Anything
            import sys
            import os

            old_stdout = sys.stdout
            old_stderr = sys.stderr
            try:
                sys.stdout = open(os.devnull, "w")
                sys.stderr = open(os.devnull, "w")

                # Call the video depth inference method
                depths, _ = self.model.infer_video_depth(  # type: ignore[attr-defined]
                    frames_rgb,
                    target_fps,
                    input_size=input_size,
                    device=self.device,
                    fp32=fp32,
                )
            finally:
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            return np.asarray(depths, dtype=np.float32)

        except Exception as e:
            raise RuntimeError(f"Video depth estimation failed: {e}")

    def _suppress_model_output(self):
        """Context manager to suppress model output streams."""
        import sys
        import os
        import contextlib

        @contextlib.contextmanager
        def suppress():
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            try:
                sys.stdout = open(os.devnull, "w")
                sys.stderr = open(os.devnull, "w")
                yield
            finally:
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        return suppress()

    @staticmethod
    def _build_vda_transform(frame_height: int, frame_width: int, input_size: int):
        from torchvision.transforms import Compose  # type: ignore[import-untyped]
        from video_depth_anything.util.transform import (  # type: ignore[import-not-found]
            Resize,
            NormalizeImage,
            PrepareForNet,
        )

        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14
        return Compose(
            [
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

    @staticmethod
    def _transform_vda_frame(frame_bgr: np.ndarray, transform) -> torch.Tensor:
        frame_rgb = frame_bgr[..., ::-1]
        transformed = transform({"image": frame_rgb.astype(np.float32) / 255.0})["image"]
        return torch.from_numpy(np.asarray(transformed, dtype=np.float32)).unsqueeze(0).unsqueeze(0)

    def _assemble_vda_input(
        self,
        frames: np.ndarray,
        transform,
        carried_input: torch.Tensor | None,
        padding_input: torch.Tensor | None,
    ) -> torch.Tensor:
        first_transformed = self._transform_vda_frame(frames[0], transform) if len(frames) else None
        template = carried_input[:, :1] if carried_input is not None else first_transformed
        if template is None:
            raise ValueError("A VDA window requires source or carried input")
        current_input = torch.empty(
            (1, VDA_INFER_LEN, *template.shape[2:]),
            dtype=template.dtype,
            device=self.device,
        )

        write_index = 0
        if carried_input is not None:
            if carried_input.shape[:2] != (1, VDA_OVERLAP):
                raise ValueError("Carried VDA input must have shape [1,10,C,H,W]")
            current_input[:, :VDA_OVERLAP].copy_(carried_input)
            write_index = VDA_OVERLAP

        for frame_index, frame in enumerate(frames):
            transformed = (
                first_transformed
                if frame_index == 0 and first_transformed is not None
                else self._transform_vda_frame(frame, transform)
            )
            current_input[:, write_index].copy_(transformed[:, 0])
            write_index += 1

        if write_index == VDA_INFER_LEN:
            return current_input
        if len(frames):
            pad_value = current_input[:, write_index - 1 : write_index]
        elif padding_input is not None:
            pad_value = padding_input
        else:
            raise ValueError("Cannot pad an empty VDA input window")
        repeated = pad_value.clone().expand(-1, VDA_INFER_LEN - write_index, -1, -1, -1)
        current_input[:, write_index:].copy_(repeated)
        return current_input

    def _infer_fixed_window(
        self,
        frames: np.ndarray,
        *,
        input_size: int,
        fp32: bool,
        carried_input: torch.Tensor | None,
        padding_input: torch.Tensor | None,
        transform,
        output_shape: tuple[int, int],
    ) -> tuple[np.ndarray, torch.Tensor, Any]:
        """Run exactly one 32-frame VDA forward without nested segmentation."""

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if transform is None:
            if not len(frames):
                raise ValueError("The first VDA window requires at least one source frame")
            transform = self._build_vda_transform(*output_shape, input_size)

        current_input = self._assemble_vda_input(
            frames,
            transform,
            carried_input,
            padding_input,
        )

        device_type = self.device.split(":", maxsplit=1)[0]
        with torch.no_grad():
            with torch.autocast(device_type=device_type, enabled=not fp32):
                depth = self.model.forward(current_input)  # type: ignore[union-attr]
        if not isinstance(depth, torch.Tensor) or depth.shape[:2] != (1, VDA_INFER_LEN):
            raise ValueError("VDA forward must return a tensor with shape [1,32,H,W]")
        depth = depth.to(current_input.dtype)
        depth = F.interpolate(
            depth.flatten(0, 1).unsqueeze(1),
            size=output_shape,
            mode="bilinear",
            align_corners=True,
        )
        values = depth[:, 0].detach().cpu().numpy().astype(np.float32, copy=False)
        return values, current_input, transform

    @staticmethod
    def _compute_scale_and_shift(
        current_references: np.ndarray,
        retained_references: np.ndarray,
    ) -> tuple[float, float]:
        from utils.util import compute_scale_and_shift  # type: ignore[import-not-found]

        current = np.concatenate(list(current_references), axis=0)
        retained = np.concatenate(list(retained_references), axis=0)
        mask = np.ones_like(retained, dtype=bool)
        scale, shift = compute_scale_and_shift(current, retained, mask)
        return float(scale), float(shift)

    @staticmethod
    def _interpolate_depths(previous: np.ndarray, current: np.ndarray) -> list[np.ndarray]:
        from utils.util import get_interpolate_frames  # type: ignore[import-not-found]

        return list(get_interpolate_frames(list(previous), list(current)))

    def _finalize_window(
        self,
        previous_pending: np.ndarray,
        current_depth: np.ndarray,
        retained_references: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Align one later window, blend its overlap, and split finalized state."""

        previous = np.asarray(previous_pending, dtype=np.float32)
        current = np.asarray(current_depth, dtype=np.float32)
        references = np.asarray(retained_references, dtype=np.float32)
        if previous.shape[0] != VDA_INTERP_LEN or current.shape[0] != VDA_INFER_LEN:
            raise ValueError("VDA finalization requires 8 pending and 32 current depth maps")
        if references.shape[0] != VDA_OVERLAP - VDA_INTERP_LEN:
            raise ValueError("VDA finalization requires two alignment references")
        if not current.flags.writeable:
            raise ValueError("VDA current-window depth must be writable")

        if self.metric:
            scale, shift = 1.0, 0.0
        else:
            scale, shift = self._compute_scale_and_shift(current[:2], references)
        aligned = current[2:]
        aligned *= scale
        aligned += shift
        np.maximum(aligned, 0, out=aligned)

        blended = self._interpolate_depths(previous, aligned[:VDA_INTERP_LEN])
        if len(blended) != VDA_INTERP_LEN:
            raise ValueError("VDA overlap interpolation returned an unexpected shape")
        for index, value in enumerate(blended):
            frame = np.asarray(value, dtype=np.float32)
            if frame.shape != previous.shape[1:]:
                raise ValueError("VDA overlap interpolation returned an unexpected shape")
            current[index] = frame

        finalized_tail_length = VDA_FRAME_STEP - VDA_INTERP_LEN
        for index in range(finalized_tail_length):
            current[VDA_INTERP_LEN + index] = current[VDA_OVERLAP + index]
        next_pending = current[VDA_INFER_LEN - VDA_INTERP_LEN :].copy()
        next_reference = current[VDA_KEYFRAMES[1] - (VDA_OVERLAP - VDA_INTERP_LEN)]
        return current[:VDA_FRAME_STEP], next_pending, next_reference

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        if not self.model_config:
            return {"inference_algorithm": VDA_INFERENCE_ALGORITHM}

        return {
            "encoder": self.model_config["encoder"],
            "features": self.model_config["features"],
            "out_channels": self.model_config["out_channels"],
            "num_frames": self.model_config["num_frames"],
            "device": self.device,
            "metric": self.metric,
            "model_path": self.model_path,
            "loaded": self.model is not None,
            "temporal_consistency": True,  # Key feature of video model
            "inference_algorithm": VDA_INFERENCE_ALGORITHM,
        }

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None

            # Clear GPU cache if using CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()


def create_video_depth_estimator(
    model_path: str | None = None,
    device: str = "auto",
    metric: bool = False,
    temporal_window_overlap: int = 10,
) -> VideoDepthEstimator:
    """
    Factory function to create a video depth estimator.

    Args:
        model_path: Path to model file (uses default if None)
        device: Device to use for inference
        metric: Use metric depth model (true depth values)
        temporal_window_overlap: Number of frames to overlap between chunks (V2 only)

    Returns:
        Configured VideoDepthEstimator instance
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    return VideoDepthEstimator(model_path, device, metric, temporal_window_overlap)
