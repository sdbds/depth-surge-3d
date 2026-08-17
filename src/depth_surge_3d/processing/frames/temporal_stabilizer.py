"""File-backed coordinator for optional VDPP temporal stabilization."""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from ...core.depth_contract import canonical_json_hash
from ...core.render_disparity import validate_render_disparity_input
from ...inference.depth.vdpp_artifact import ensure_vdpp_checkpoint
from ...inference.depth.vdpp_contract import (
    build_vdpp_execution_plan,
    vdpp_model_identity,
)
from .depth_normalizer import decode_canonical_png
from .temporal_storage import StabilizedDepthStore, build_final_shot_plan


def _default_cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def _default_postprocessor_factory(checkpoint_path: Path, device: str):
    from ...inference.depth.vdpp_temporal_postprocessor import (
        VDPPTemporalPostprocessor,
    )

    return VDPPTemporalPostprocessor(
        checkpoint_path=checkpoint_path,
        device=device,
    )


def collect_vdpp_runtime_identity(
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collect only after artifact-first planning proves generation is needed."""

    import torch

    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("VDPP generation requires an available CUDA device")
    index = selected.index if selected.index is not None else torch.cuda.current_device()
    selected = torch.device("cuda", index)
    try:
        xformers_version = importlib.metadata.version("xformers")
        attention_backend = "xformers"
    except importlib.metadata.PackageNotFoundError:
        xformers_version = None
        attention_backend = "torch"
    driver_version = None
    get_driver_version = getattr(torch._C, "_cuda_getDriverVersion", None)
    if callable(get_driver_version):
        try:
            driver_version = str(get_driver_version())
        except RuntimeError:
            pass
    identity = {
        "torch_version": str(torch.__version__),
        "xformers_version": xformers_version,
        "attention_backend": attention_backend,
        "attention_operator": "auto",
        "cuda_runtime": str(torch.version.cuda),
        "cuda_driver": driver_version,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_name": torch.cuda.get_device_name(selected),
        "compute_capability": list(torch.cuda.get_device_capability(selected)),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    return identity, dict(identity)


class TemporalDepthStabilizer:
    """Own validation, artifacts, shot resume, model laziness, and progress."""

    def __init__(
        self,
        *,
        effective_device: str,
        models_dir: Path | str,
        cuda_available: Callable[[], bool] = _default_cuda_available,
        disk_usage: Callable[[Path], Any] = shutil.disk_usage,
        checkpoint_resolver: Callable[..., Path] = ensure_vdpp_checkpoint,
        postprocessor_factory: Callable[[Path, str], Any] = _default_postprocessor_factory,
        runtime_identity_provider: Callable[
            [str], tuple[dict[str, Any], dict[str, Any]]
        ] = collect_vdpp_runtime_identity,
    ) -> None:
        self.effective_device = str(effective_device)
        self.models_dir = Path(models_dir)
        self._cuda_available = cuda_available
        self._disk_usage = disk_usage
        self._checkpoint_resolver = checkpoint_resolver
        self._postprocessor_factory = postprocessor_factory
        self._runtime_identity_provider = runtime_identity_provider

    @staticmethod
    def _read_final_scene_manifest(
        scene_dir: Path,
        *,
        frame_names: list[str],
        canonical_scene_fingerprint: str,
    ) -> dict[str, Any]:
        path = scene_dir / "scene_manifest.json"
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Final scene manifest is invalid: {path}") from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("status") != "final"
            or manifest.get("frame_names") != frame_names
            or canonical_json_hash(manifest) != canonical_scene_fingerprint
        ):
            raise ValueError("Final scene manifest does not match canonical disparity")
        return manifest

    @staticmethod
    def _semantic_identity(
        *,
        frame_names: list[str],
        native_shape: tuple[int, int],
        source_fingerprint: str,
        scene_fingerprint: str,
        shot_plan: list[dict[str, int]],
    ) -> dict[str, Any]:
        return {
            "frame_names": frame_names,
            "native_shape": list(native_shape),
            "source_canonical_fingerprint": source_fingerprint,
            "scene_manifest_fingerprint": scene_fingerprint,
            "postprocessor_settings": {"temporal_postprocessor": "vdpp"},
            "model_identity": vdpp_model_identity(),
            "execution_plan": build_vdpp_execution_plan(native_shape),
            "shot_plan": shot_plan,
        }

    @staticmethod
    def _report_progress(
        progress_tracker: Any,
        message: str,
        *,
        completed: int,
        total: int,
    ) -> None:
        if progress_tracker is None:
            return
        progress_tracker.update_progress(
            message,
            phase="temporal_stabilization",
            frame_num=completed,
            step_name="Temporal Depth Stabilization",
            step_progress=completed,
            step_total=total,
        )

    @staticmethod
    def _required_disk_bytes(num_frames: int, native_shape: tuple[int, int]) -> int:
        frame_bytes = native_shape[0] * native_shape[1] * 2
        return int(num_frames * frame_bytes * 1.10) + frame_bytes

    def _preflight_disk(
        self,
        stable_root: Path,
        *,
        num_frames: int,
        native_shape: tuple[int, int],
    ) -> None:
        required = self._required_disk_bytes(num_frames, native_shape)
        try:
            free = int(self._disk_usage(stable_root.parent).free)
        except OSError as exc:
            raise OSError(f"Could not inspect disk space for VDPP output: {stable_root}") from exc
        if free < required:
            raise OSError(
                f"Insufficient disk space for VDPP output: need {required} bytes, "
                f"only {free} bytes are free"
            )

    @staticmethod
    def _window_loader(
        base_files: list[Path],
        *,
        shot_start: int,
        shot_length: int,
        native_shape: tuple[int, int],
    ) -> Callable[[int, int], np.ndarray]:
        def load_window(start: int, end: int) -> np.ndarray:
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end <= start
                or end > shot_length
            ):
                raise ValueError(f"Invalid VDPP shot-local range [{start}, {end})")
            output = np.empty((end - start, *native_shape), dtype=np.float32)
            for output_index, local_index in enumerate(range(start, end)):
                path = base_files[shot_start + local_index]
                decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if decoded is None:
                    raise OSError(f"Could not decode canonical disparity: {path}")
                if decoded.dtype != np.uint16 or decoded.ndim != 2:
                    raise TypeError(f"Canonical disparity must be uint16 grayscale: {path}")
                if decoded.shape != native_shape:
                    raise ValueError(
                        f"Canonical disparity shape {decoded.shape} does not match "
                        f"{native_shape}: {path}"
                    )
                output[output_index] = decode_canonical_png(decoded)
            return output

        return load_window

    def generate_files(  # noqa: C901
        self,
        base_files: list[Path],
        settings: dict[str, Any],
        directories: dict[str, Path],
        progress_tracker: Any = None,
    ) -> list[Path]:
        """Return a complete stabilized artifact or fail without passthrough."""

        if settings.get("temporal_postprocessor") != "vdpp":
            raise ValueError(
                "TemporalDepthStabilizer may only be invoked when temporal_postprocessor=vdpp"
            )
        if not base_files:
            raise ValueError("VDPP requires canonical disparity files")

        metadata_path = base_files[0].parent / "metadata.json"
        try:
            base_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Canonical disparity metadata is invalid: {metadata_path}") from exc
        if not isinstance(base_metadata, dict):
            raise ValueError(f"Canonical disparity metadata is invalid: {metadata_path}")
        frame_names = base_metadata.get("frame_names")
        if not isinstance(frame_names, list) or not all(
            isinstance(name, str) for name in frame_names
        ):
            raise ValueError("Canonical disparity frame manifest is invalid")
        frame_files = [Path(name) for name in frame_names]
        base_artifact = validate_render_disparity_input(base_files, frame_files)
        if base_artifact.producer != "base":
            raise ValueError("VDPP input must be base canonical disparity")
        scene_fingerprint = base_artifact.metadata.get("scene_manifest_fingerprint")
        if not isinstance(scene_fingerprint, str):
            raise ValueError("Canonical disparity is missing its scene fingerprint")
        manifest = self._read_final_scene_manifest(
            Path(directories["scene_data"]),
            frame_names=frame_names,
            canonical_scene_fingerprint=scene_fingerprint,
        )
        cuts = manifest.get("final_cuts")
        if not isinstance(cuts, list) or not all(
            not isinstance(cut, bool) and isinstance(cut, int) for cut in cuts
        ):
            raise ValueError("Final scene manifest has invalid cut indexes")
        shot_plan = build_final_shot_plan(len(base_files), cuts)
        semantic_identity = self._semantic_identity(
            frame_names=frame_names,
            native_shape=base_artifact.native_shape,
            source_fingerprint=base_artifact.fingerprint,
            scene_fingerprint=scene_fingerprint,
            shot_plan=shot_plan,
        )
        stable_root = Path(directories["disparity_stabilized"])

        lightweight_store = StabilizedDepthStore(
            stable_root,
            frame_files=frame_files,
            semantic_identity=semantic_identity,
            runtime_identity={},
            execution_provenance={},
        )
        lightweight_audit = lightweight_store.audit()
        if lightweight_audit.complete:
            self._report_progress(
                progress_tracker,
                f"Reused {len(base_files)} stabilized depth maps",
                completed=len(base_files),
                total=len(base_files),
            )
            return lightweight_store.depth_files

        if not self.effective_device.startswith("cuda") or not self._cuda_available():
            raise RuntimeError(
                "VDPP generation requires CUDA; a complete cached stabilized artifact "
                "can still be rendered without CUDA"
            )
        self._preflight_disk(
            stable_root,
            num_frames=len(base_files),
            native_shape=base_artifact.native_shape,
        )
        runtime_identity, execution_provenance = self._runtime_identity_provider(
            self.effective_device
        )
        store = StabilizedDepthStore(
            stable_root,
            frame_files=frame_files,
            semantic_identity=semantic_identity,
            runtime_identity=runtime_identity,
            execution_provenance=execution_provenance,
        )
        audit = store.audit()
        if audit.complete:
            return store.depth_files

        required_ids = set(audit.invalid_shot_ids) | set(audit.pending_shot_ids)
        if audit.reset_required:
            required_ids = {shot["shot_id"] for shot in shot_plan}
        if not required_ids:
            store.prepare(audit)
            completed = store.finalize()
            self._report_progress(
                progress_tracker,
                f"Reused {len(completed)} stabilized depth maps",
                completed=len(completed),
                total=len(completed),
            )
            return completed

        checkpoint = self._checkpoint_resolver(
            self.models_dir,
            progress_callback=lambda current, total: self._report_progress(
                progress_tracker,
                f"Downloading VDPP checkpoint {current}/{total} bytes",
                completed=0,
                total=len(base_files),
            ),
        )
        postprocessor = None
        try:
            postprocessor = self._postprocessor_factory(
                Path(checkpoint),
                self.effective_device,
            )
            if postprocessor.model_identity() != semantic_identity["model_identity"]:
                raise RuntimeError("VDPP model identity does not match the persisted plan")
            if (
                postprocessor.execution_plan(base_artifact.native_shape)
                != semantic_identity["execution_plan"]
            ):
                raise RuntimeError("VDPP execution arguments do not match the persisted plan")

            longest_pending = max(
                shot["end"] - shot["start"] for shot in shot_plan if shot["shot_id"] in required_ids
            )
            preflight = postprocessor.preflight(
                longest_pending,
                base_artifact.native_shape,
            )
            store.execution_provenance.update(preflight)
            store.prepare(audit)

            completed_frames = sum(
                shot["end"] - shot["start"]
                for shot in shot_plan
                if shot["shot_id"] in audit.reusable_shot_ids
            )
            self._report_progress(
                progress_tracker,
                "Stabilizing temporal depth",
                completed=completed_frames,
                total=len(base_files),
            )
            for shot in shot_plan:
                shot_id = shot["shot_id"]
                if shot_id in audit.reusable_shot_ids and not audit.reset_required:
                    continue
                shot_length = shot["end"] - shot["start"]
                loader = self._window_loader(
                    base_files,
                    shot_start=shot["start"],
                    shot_length=shot_length,
                    native_shape=base_artifact.native_shape,
                )
                iterator = postprocessor.process_shot(shot_length, loader)
                try:
                    store.commit_shot(shot_id, iterator)
                finally:
                    close = getattr(iterator, "close", None)
                    if callable(close):
                        close()
                completed_frames += shot_length
                self._report_progress(
                    progress_tracker,
                    f"Stabilized {completed_frames}/{len(base_files)} depth maps",
                    completed=completed_frames,
                    total=len(base_files),
                )
            return store.finalize()
        except BaseException as exc:
            print(f"VDPP temporal stabilization failed: {exc}", file=sys.stderr)
            raise
        finally:
            if postprocessor is not None:
                postprocessor.release()
