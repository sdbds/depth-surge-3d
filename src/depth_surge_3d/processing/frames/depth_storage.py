"""Atomic native-resolution raw-depth storage and resume validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from ...inference.depth.types import DepthRepresentation


RAW_DEPTH_SCHEMA_VERSION = 1
RAW_STORAGE_DTYPES = {"auto", "float16", "float32"}


class RawDepthFingerprintError(ValueError):
    """Raised when persisted raw depth belongs to another inference contract."""


class RawDepthOverflowError(ValueError):
    """Raised when explicit float16 storage cannot represent a finite value."""


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)


def canonical_json_hash(payload: Any) -> str:
    """Hash JSON-compatible metadata independently of dictionary insertion order."""

    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_model_fingerprint(estimator: Any, settings: dict[str, Any]) -> dict[str, Any]:
    """Build a stable semantic fingerprint for one estimator configuration."""

    model_info = _jsonable(
        estimator.get_model_info() if hasattr(estimator, "get_model_info") else {}
    )
    model_path_value = getattr(estimator, "model_path", None)
    model_path = Path(model_path_value) if model_path_value else None
    weight_sha256 = _hash_file(model_path) if model_path and model_path.is_file() else None
    artifact_identity = (
        weight_sha256
        or model_info.get("weight_sha256")
        or model_info.get("revision")
        or canonical_json_hash(model_info)
    )
    return {
        "backend": f"{type(estimator).__module__}.{type(estimator).__qualname__}",
        "model_info": model_info,
        "depth_settings": _jsonable(settings),
        "weight_sha256": weight_sha256,
        "artifact_identity": artifact_identity,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _atomic_save_npz(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, values=values)
    temporary.replace(path)


def _fits_float16(values: np.ndarray) -> bool:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return True
    limit = np.finfo(np.float16).max
    return bool(finite.min() >= -limit and finite.max() <= limit)


def estimate_depth_disk_bytes(
    *,
    frame_count: int,
    native_width: int,
    native_height: int,
    storage_bytes: int,
    keep_intermediates: bool,
) -> int:
    """Return conservative uncompressed peak bytes for raw and canonical depth."""

    raw_payload = frame_count * native_width * native_height * storage_bytes
    canonical_payload = frame_count * native_width * native_height * 2
    raw_allowance = raw_payload * 1.25
    canonical_allowance = canonical_payload * 1.10
    if keep_intermediates:
        return int(raw_allowance + canonical_allowance)
    atomic_overlap = 2 * max(
        native_width * native_height * storage_bytes, native_width * native_height * 2
    )
    return int(max(raw_allowance, canonical_allowance) + atomic_overlap)


def require_disk_space(
    directory: Path, required_bytes: int, *, available_bytes: int | None = None
) -> None:
    """Fail before expensive work when the output filesystem lacks space."""

    directory.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(directory).free if available_bytes is None else available_bytes
    if available < required_bytes:
        raise OSError(
            f"Insufficient disk space: required {required_bytes} bytes, available {available} bytes"
        )


class RawDepthStore:
    """One schema-validated directory of native-resolution raw depth maps."""

    def __init__(self, directory: Path, metadata: dict[str, Any]) -> None:
        self.directory = directory
        self.metadata_path = directory / "metadata.json"
        self.metadata = metadata

    @classmethod
    def open_or_create(
        cls,
        directory: Path,
        *,
        frame_names: list[str],
        representation: DepthRepresentation,
        semantic_fingerprint: dict[str, Any],
        requested_dtype: str,
        first_values: np.ndarray | None = None,
    ) -> "RawDepthStore":
        if requested_dtype not in RAW_STORAGE_DTYPES:
            raise ValueError(f"Unsupported raw storage dtype: {requested_dtype}")
        directory.mkdir(parents=True, exist_ok=True)
        metadata_path = directory / "metadata.json"
        semantic = _jsonable(semantic_fingerprint)

        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            store = cls(directory, metadata)
            store._validate_existing(
                frame_names,
                representation,
                semantic,
                requested_dtype,
                first_values,
            )
            if metadata.get("storage_status") == "promoting":
                store.promote_to_float32(requested_dtype=requested_dtype)
            elif metadata.get("selected_dtype") == "float16" and requested_dtype == "float32":
                store.promote_to_float32(requested_dtype=requested_dtype)
            elif requested_dtype == "float16" and metadata.get("selected_dtype") == "float32":
                raise RawDepthFingerprintError("Cannot narrow an existing float32 raw directory")
            return store

        if first_values is None:
            raise ValueError("First raw-depth chunk is required to create storage metadata")
        values = np.asarray(first_values, dtype=np.float32)
        if values.ndim != 3:
            raise ValueError("First raw-depth chunk must have shape [N,H,W]")
        selected_dtype = requested_dtype
        if requested_dtype == "auto":
            selected_dtype = "float16" if _fits_float16(values) else "float32"
        elif requested_dtype == "float16" and not _fits_float16(values):
            raise RawDepthOverflowError(
                "First raw-depth chunk exceeds float16 range; use raw_storage_dtype=float32"
            )

        provenance = f"native_{selected_dtype}"
        metadata = {
            "schema_version": RAW_DEPTH_SCHEMA_VERSION,
            "storage_status": "ready",
            "representation": representation.value,
            "frame_names": list(frame_names),
            "native_shape": [int(values.shape[1]), int(values.shape[2])],
            "requested_dtype": requested_dtype,
            "selected_dtype": selected_dtype,
            "storage_provenance": provenance,
            "compression": "npz_deflate",
            "semantic_fingerprint": semantic,
            "completed_count": 0,
        }
        metadata["fingerprint"] = cls._fingerprint(metadata)
        _atomic_write_json(metadata_path, metadata)
        return cls(directory, metadata)

    @staticmethod
    def _fingerprint(metadata: dict[str, Any]) -> str:
        keys = (
            "schema_version",
            "representation",
            "frame_names",
            "native_shape",
            "requested_dtype",
            "selected_dtype",
            "storage_provenance",
            "compression",
            "semantic_fingerprint",
        )
        return canonical_json_hash({key: metadata.get(key) for key in keys})

    def _validate_existing(
        self,
        frame_names: list[str],
        representation: DepthRepresentation,
        semantic_fingerprint: dict[str, Any],
        requested_dtype: str,
        first_values: np.ndarray | None,
    ) -> None:
        if self.metadata.get("schema_version") != RAW_DEPTH_SCHEMA_VERSION:
            raise RawDepthFingerprintError("Raw-depth schema version mismatch")
        if self.metadata.get("semantic_fingerprint") != semantic_fingerprint:
            raise RawDepthFingerprintError("Raw-depth semantic fingerprint mismatch")
        if self.metadata.get("representation") != representation.value:
            raise RawDepthFingerprintError("Raw-depth representation mismatch")
        if self.metadata.get("frame_names") != list(frame_names):
            raise RawDepthFingerprintError("Raw-depth source frame manifest mismatch")
        persisted_request = self.metadata.get("requested_dtype")
        promotes_float16 = (
            self.metadata.get("selected_dtype") == "float16" and requested_dtype == "float32"
        )
        if persisted_request != requested_dtype and not promotes_float16:
            raise RawDepthFingerprintError("Raw-depth requested dtype mismatch")
        if first_values is not None:
            values = np.asarray(first_values)
            if values.ndim != 3 or list(values.shape[1:]) != self.metadata.get("native_shape"):
                raise RawDepthFingerprintError("Raw-depth native shape mismatch")
        if self.metadata.get("storage_status") == "ready":
            expected = self._fingerprint(self.metadata)
            if self.metadata.get("fingerprint") != expected:
                raise RawDepthFingerprintError("Raw-depth storage fingerprint mismatch")

    def path_for(self, frame_name: str) -> Path:
        return self.directory / f"{Path(frame_name).stem}.npz"

    @property
    def complete_files(self) -> list[Path]:
        return [
            path
            for frame_name in self.metadata["frame_names"]
            if (path := self.path_for(frame_name)).is_file()
        ]

    def write_batch(self, frame_names: list[str], values: np.ndarray) -> list[Path]:
        batch = np.asarray(values, dtype=np.float32)
        if batch.ndim != 3 or len(frame_names) != len(batch):
            raise ValueError("Raw-depth batch names and [N,H,W] values must match")
        if list(batch.shape[1:]) != self.metadata["native_shape"]:
            raise RawDepthFingerprintError("Raw-depth native shape changed during inference")
        unknown = set(frame_names).difference(self.metadata["frame_names"])
        if unknown:
            raise RawDepthFingerprintError(f"Unknown raw-depth frame names: {sorted(unknown)}")

        if self.metadata["selected_dtype"] == "float16" and not _fits_float16(batch):
            if self.metadata["requested_dtype"] == "auto":
                self.promote_to_float32(requested_dtype="auto")
            else:
                raise RawDepthOverflowError(
                    "Raw-depth values exceed float16 range; resume with raw_storage_dtype=float32"
                )

        dtype = np.float16 if self.metadata["selected_dtype"] == "float16" else np.float32
        paths: list[Path] = []
        for frame_name, frame_values in zip(frame_names, batch):
            path = self.path_for(frame_name)
            if not path.is_file():
                _atomic_save_npz(path, frame_values.astype(dtype))
            paths.append(path)
        self.metadata["completed_count"] = len(self.complete_files)
        _atomic_write_json(self.metadata_path, self.metadata)
        return paths

    def load(self, path: Path) -> np.ndarray:
        with np.load(path, allow_pickle=False) as payload:
            values = np.asarray(payload["values"])
        if values.ndim != 2:
            raise ValueError(f"Raw-depth file must contain one [H,W] map: {path}")
        if self.metadata.get("storage_status") == "ready":
            expected = np.dtype(self.metadata["selected_dtype"])
            if values.dtype != expected:
                raise RawDepthFingerprintError(f"Raw-depth file dtype mismatch: {path}")
        return values.astype(np.float32)

    def _rewrite_file_as_float32(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as payload:
            values = np.asarray(payload["values"])
        if values.dtype == np.float32:
            return
        if values.dtype != np.float16:
            raise RawDepthFingerprintError(f"Cannot promote unexpected raw dtype at {path}")
        _atomic_save_npz(path, values.astype(np.float32))

    def promote_to_float32(self, *, requested_dtype: str) -> None:
        """Atomically widen completed float16 payloads without estimator calls."""

        if requested_dtype not in {"auto", "float32"}:
            raise ValueError("Float32 promotion requires auto or float32 storage")
        if (
            self.metadata.get("selected_dtype") == "float32"
            and self.metadata.get("storage_status") == "ready"
        ):
            return

        if self.metadata.get("storage_status") != "promoting":
            frame_pixels = int(np.prod(self.metadata["native_shape"]))
            extra_bytes = len(self.complete_files) * frame_pixels * 2 + frame_pixels * 4
            require_disk_space(self.directory, extra_bytes)
            self.metadata["storage_status"] = "promoting"
            self.metadata["previous_fingerprint"] = self.metadata.get("fingerprint")
            self.metadata["target_dtype"] = "float32"
            self.metadata["target_provenance"] = "promoted_float16_to_float32"
            self.metadata["requested_dtype"] = requested_dtype
            _atomic_write_json(self.metadata_path, self.metadata)

        for path in self.complete_files:
            self._rewrite_file_as_float32(path)

        for path in self.complete_files:
            with np.load(path, allow_pickle=False) as payload:
                if payload["values"].dtype != np.float32:
                    raise RawDepthFingerprintError(f"Raw-depth promotion incomplete at {path}")

        self.metadata.update(
            {
                "storage_status": "ready",
                "selected_dtype": "float32",
                "storage_provenance": "promoted_float16_to_float32",
            }
        )
        self.metadata.pop("target_dtype", None)
        self.metadata.pop("target_provenance", None)
        self.metadata["fingerprint"] = self._fingerprint(self.metadata)
        _atomic_write_json(self.metadata_path, self.metadata)
