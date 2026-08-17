"""Bounded shot iterator for the pinned VDPP v1.0 recurrence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Generator

import numpy as np
import torch
import torch.nn.functional as F

from ..._vendor.vdpp.vdpp_model import VDPP, compute_scale_and_shift
from .vdpp_contract import (
    VDPP_DOWNSIZE,
    VDPP_MODEL_CONFIG,
    VDPP_OVERLAP,
    VDPP_STRIDE,
    VDPP_WINDOW_SIZE,
    build_vdpp_execution_plan,
    ceil_multiple,
    vdpp_model_identity,
)


class VDPPTemporalPostprocessor:
    """Own the VDPP model, padded-space recurrence, and compact device state."""

    def __init__(
        self,
        *,
        checkpoint_path: Path | str | None = None,
        device: str | torch.device = "cuda",
        model: Any | None = None,
        model_factory: Callable[[], Any] | None = None,
        allow_unavailable_cuda_for_tests: bool = False,
    ) -> None:
        self.device = torch.device(device)
        if model is None and self.device.type != "cuda" and not allow_unavailable_cuda_for_tests:
            raise RuntimeError("VDPP generation requires a CUDA device")
        if (
            model is None
            and self.device.type == "cuda"
            and not torch.cuda.is_available()
            and not allow_unavailable_cuda_for_tests
        ):
            raise RuntimeError("VDPP generation requires an available CUDA device")

        self._active_retained: torch.Tensor | None = None
        self._model: Any | None = model
        if self._model is None:
            if checkpoint_path is None:
                raise ValueError("VDPP checkpoint path is required")
            factory = model_factory or (lambda: VDPP(**VDPP_MODEL_CONFIG))
            loaded_model = factory()
            self._register_release_checkpoint_state(loaded_model)
            state_dict = torch.load(
                Path(checkpoint_path),
                map_location="cpu",
                weights_only=True,
            )
            loaded_model.load_state_dict(state_dict, strict=True)
            self._model = loaded_model.to(self.device).eval()
        elif hasattr(self._model, "eval"):
            evaluated = self._model.eval()
            if evaluated is not None:
                self._model = evaluated

    @staticmethod
    def _register_release_checkpoint_state(model: Any) -> None:
        """Register the released checkpoint's inert key absent from public source."""

        if hasattr(model, "shift_head"):
            raise RuntimeError("Pinned VDPP source unexpectedly defines shift_head")
        model.shift_head = torch.nn.Sequential(torch.nn.Conv2d(1, 1, kernel_size=1))

    @property
    def retained_frame_count(self) -> int:
        retained = self._active_retained
        return 0 if retained is None else int(retained.shape[1])

    def model_identity(self) -> dict[str, Any]:
        return vdpp_model_identity()

    def execution_plan(self, native_shape: tuple[int, int]) -> dict[str, Any]:
        return build_vdpp_execution_plan(native_shape)

    @staticmethod
    def _validate_loaded_window(
        values: object,
        *,
        start: int,
        end: int,
        native_shape: tuple[int, int],
    ) -> np.ndarray:
        if not isinstance(values, np.ndarray):
            raise TypeError("VDPP load_window must return a NumPy float32 array")
        if values.dtype != np.float32:
            raise TypeError("VDPP load_window must return float32 values")
        expected_shape = (end - start, *native_shape)
        if values.shape != expected_shape:
            raise ValueError(
                f"VDPP load_window must return exactly {expected_shape}; got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("VDPP load_window values must be finite")
        if values.size and (float(values.min()) < 0.0 or float(values.max()) > 1.0):
            raise ValueError("VDPP load_window values must be in [0, 1]")
        return values

    def _load_padded(
        self,
        load_window: Callable[[int, int], np.ndarray],
        *,
        start: int,
        end: int,
        native_shape: tuple[int, int],
        padded_shape: tuple[int, int],
    ) -> torch.Tensor:
        loaded = self._validate_loaded_window(
            load_window(start, end),
            start=start,
            end=end,
            native_shape=native_shape,
        )
        return self._to_padded(
            loaded,
            padded_shape=padded_shape,
        )

    def _to_padded(
        self,
        loaded: np.ndarray,
        *,
        padded_shape: tuple[int, int],
    ) -> torch.Tensor:
        tensor = (
            torch.from_numpy(np.ascontiguousarray(loaded))
            .unsqueeze(0)
            .to(
                self.device,
                dtype=torch.float32,
            )
        )
        if tuple(loaded.shape[-2:]) != padded_shape:
            tensor = (
                F.interpolate(
                    tensor.flatten(0, 1).unsqueeze(1),
                    size=padded_shape,
                    mode="bilinear",
                    align_corners=True,
                )
                .squeeze(1)
                .unflatten(0, (1, loaded.shape[0]))
            )
        return tensor

    def _forward_window(self, inputs: torch.Tensor) -> torch.Tensor:
        model = self._model
        if model is None:
            raise RuntimeError("VDPP model has been released")
        with torch.inference_mode():
            outputs = model(inputs, downsize=VDPP_DOWNSIZE)
        if not isinstance(outputs, torch.Tensor) or outputs.shape != inputs.shape:
            shape = getattr(outputs, "shape", None)
            raise ValueError(
                f"VDPP model output shape {shape} does not match input {tuple(inputs.shape)}"
            )
        outputs = outputs.to(dtype=torch.float32)
        if not bool(torch.isfinite(outputs).all().item()):
            raise FloatingPointError("VDPP model output contains non-finite values")
        return outputs

    @staticmethod
    def _align_continuation(
        current: torch.Tensor,
        retained: torch.Tensor,
    ) -> torch.Tensor:
        with torch.inference_mode():
            prediction = current[:, :VDPP_OVERLAP].flatten(1, 2)
            target = retained.flatten(1, 2)
            mask = torch.ones_like(target)
            a_00 = torch.sum(mask * prediction * prediction, (1, 2))
            a_01 = torch.sum(mask * prediction, (1, 2))
            a_11 = torch.sum(mask, (1, 2))
            determinant = a_00 * a_11 - a_01 * a_01
            if not bool(torch.isfinite(determinant).all().item()):
                raise FloatingPointError(
                    f"VDPP affine determinant is non-finite: {determinant.detach().cpu().tolist()}"
                )
            scale, shift = compute_scale_and_shift(prediction, target, mask)
            if not bool(torch.isfinite(scale).all().item()) or not bool(
                torch.isfinite(shift).all().item()
            ):
                raise FloatingPointError(
                    "VDPP affine parameters are non-finite: "
                    f"scale={scale.detach().cpu().tolist()}, "
                    f"shift={shift.detach().cpu().tolist()}"
                )
            aligned = current * scale.view(-1, 1, 1, 1) + shift.view(-1, 1, 1, 1)
            if not bool(torch.isfinite(aligned).all().item()):
                raise FloatingPointError("VDPP aligned output contains non-finite values")
            return aligned

    @staticmethod
    def _to_native(frame: torch.Tensor, native_shape: tuple[int, int]) -> np.ndarray:
        with torch.inference_mode():
            if tuple(frame.shape[-2:]) != native_shape:
                frame = F.interpolate(
                    frame.view(1, 1, *frame.shape[-2:]),
                    size=native_shape,
                    mode="bilinear",
                    align_corners=True,
                ).view(*native_shape)
            return frame.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().copy()

    def process_shot(  # noqa: C901
        self,
        frame_count: int,
        load_window: Callable[[int, int], np.ndarray],
    ) -> Generator[tuple[int, np.ndarray], None, None]:
        """Yield ordered native-shape outputs before the storage-range clip."""

        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1:
            raise ValueError("VDPP frame_count must be a positive integer")
        if not callable(load_window):
            raise TypeError("VDPP load_window must be callable")

        retained: torch.Tensor | None = None
        current: torch.Tensor | None = None
        native_shape: tuple[int, int] | None = None
        try:
            start = 0
            end = min(VDPP_WINDOW_SIZE, frame_count)
            first_values = load_window(start, end)
            if not isinstance(first_values, np.ndarray) or first_values.ndim != 3:
                raise ValueError("VDPP load_window must return shape [frames, height, width]")
            native_shape = (int(first_values.shape[1]), int(first_values.shape[2]))
            validated = self._validate_loaded_window(
                first_values,
                start=start,
                end=end,
                native_shape=native_shape,
            )
            padded_shape = (
                ceil_multiple(native_shape[0], 14),
                ceil_multiple(native_shape[1], 14),
            )

            inputs = self._to_padded(
                validated,
                padded_shape=padded_shape,
            )
            del validated, first_values
            current = self._forward_window(inputs)
            del inputs
            if end < frame_count:
                retained = current[:, -VDPP_OVERLAP:].detach().clone()
                self._active_retained = retained
            for local_index in range(current.shape[1]):
                yield local_index, self._to_native(current[0, local_index], native_shape)
            del current
            current = None

            while end < frame_count:
                start += VDPP_STRIDE
                end = min(start + VDPP_WINDOW_SIZE, frame_count)
                if retained is None:
                    raise RuntimeError("VDPP continuation state is missing")
                inputs = self._load_padded(
                    load_window,
                    start=start,
                    end=end,
                    native_shape=native_shape,
                    padded_shape=padded_shape,
                )
                current = self._forward_window(inputs)
                del inputs
                aligned = self._align_continuation(current, retained)
                del current
                current = aligned
                retained = None
                self._active_retained = None

                emit_start = VDPP_OVERLAP
                for local_index in range(emit_start, current.shape[1]):
                    global_index = start + local_index
                    yield global_index, self._to_native(current[0, local_index], native_shape)
                if end < frame_count:
                    retained = current[:, -VDPP_OVERLAP:].detach().clone()
                    self._active_retained = retained
                del current
                current = None
        finally:
            self._active_retained = None
            retained = None
            current = None

    def preflight(
        self,
        pending_shot_length: int,
        native_shape: tuple[int, int],
    ) -> dict[str, Any]:
        """Exercise the real first/continuation allocation path before mutation."""

        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("VDPP CUDA preflight requires an available CUDA device")
        if pending_shot_length < 1:
            raise ValueError("VDPP preflight shot length must be positive")
        second_length = min(VDPP_WINDOW_SIZE, pending_shot_length - VDPP_STRIDE)
        trial_length = (
            min(pending_shot_length, VDPP_WINDOW_SIZE)
            if pending_shot_length <= VDPP_WINDOW_SIZE
            else VDPP_STRIDE + second_length
        )
        trial = np.full((VDPP_WINDOW_SIZE, *native_shape), 0.5, dtype=np.float32)
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        iterator = self.process_shot(
            trial_length,
            lambda start, end: trial[: end - start],
        )
        try:
            for _index, _values in iterator:
                pass
        finally:
            iterator.close()
        torch.cuda.synchronize(self.device)
        result: dict[str, Any] = {
            "preflight_window_lengths": [min(VDPP_WINDOW_SIZE, trial_length)],
            "preflight_max_memory_allocated": int(torch.cuda.max_memory_allocated(self.device)),
            "preflight_max_memory_reserved": int(torch.cuda.max_memory_reserved(self.device)),
        }
        if trial_length > VDPP_WINDOW_SIZE:
            result["preflight_window_lengths"].append(second_length)
        return result

    def release(self) -> None:
        """Release model and abandoned recurrence state; safe to call repeatedly."""

        self._active_retained = None
        self._model = None
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return
        error: BaseException | None = None
        try:
            torch.cuda.synchronize(self.device)
        except BaseException as exc:
            error = exc
        try:
            torch.cuda.empty_cache()
        except BaseException as exc:
            if error is None:
                error = exc
        if error is not None:
            raise error
