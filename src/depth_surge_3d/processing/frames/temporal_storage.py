"""Shot-atomic storage and resume state for stabilized disparity."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from ...core.depth_contract import canonical_json_hash
from ...core.render_disparity import (
    STABILIZED_DEPTH_ALGORITHM_VERSION,
    STABILIZED_DEPTH_SCHEMA_VERSION,
    validate_render_disparity_input,
)
from ...utils.imaging.png_header import png_header_matches
from .depth_normalizer import encode_canonical_png


def build_final_shot_plan(num_frames: int, final_cuts: list[int]) -> list[dict[str, int]]:
    """Convert strict finalized cuts into complete half-open shot ranges."""

    if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames < 1:
        raise ValueError("num_frames must be a positive integer")
    if not isinstance(final_cuts, list):
        raise ValueError("final cuts must be a list")
    if any(isinstance(cut, bool) or not isinstance(cut, int) for cut in final_cuts):
        raise ValueError("final cuts must contain integer frame indexes")
    if final_cuts != sorted(set(final_cuts)) or any(
        cut <= 0 or cut >= num_frames for cut in final_cuts
    ):
        raise ValueError("final cuts must be strictly increasing indexes inside the video")

    boundaries = [0, *final_cuts, num_frames]
    return [
        {"shot_id": shot_id, "start": start, "end": end}
        for shot_id, (start, end) in enumerate(zip(boundaries, boundaries[1:]))
    ]


@dataclass(frozen=True)
class StabilizedStageAudit:
    """Read-only decision about an existing stabilized directory."""

    complete: bool
    reset_required: bool
    reusable_shot_ids: tuple[int, ...]
    invalid_shot_ids: tuple[int, ...]
    pending_shot_ids: tuple[int, ...]
    reason: str
    metadata: dict[str, Any] | None


class StabilizedDepthStore:
    """Persist one immutable shot at a time and publish only a full artifact."""

    def __init__(
        self,
        root: Path | str,
        *,
        frame_files: list[Path],
        semantic_identity: dict[str, Any],
        runtime_identity: dict[str, Any],
        execution_provenance: dict[str, Any],
    ) -> None:
        self.root = Path(root)
        self.metadata_path = self.root / "metadata.json"
        self.manifest_dir = self.root / "shot_manifests"
        self.frame_files = list(frame_files)
        self.semantic_identity = dict(semantic_identity)
        self.semantic_fingerprint = canonical_json_hash(self.semantic_identity)
        self.runtime_identity = dict(runtime_identity)
        self.runtime_fingerprint = canonical_json_hash(self.runtime_identity)
        self.execution_provenance = dict(execution_provenance)
        self.native_shape = self._validate_identity()
        self.shot_plan = [dict(shot) for shot in self.semantic_identity["shot_plan"]]
        self._metadata: dict[str, Any] | None = None

    def _validate_identity(self) -> tuple[int, int]:
        names = self.semantic_identity.get("frame_names")
        if names != [path.name for path in self.frame_files] or not names:
            raise ValueError("Stabilized semantic frame manifest is invalid")
        shape = self.semantic_identity.get("native_shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in shape
            )
        ):
            raise ValueError("Stabilized semantic native shape is invalid")
        plan = self.semantic_identity.get("shot_plan")
        expected = build_final_shot_plan(
            len(self.frame_files),
            [shot["end"] for shot in plan[:-1]] if isinstance(plan, list) and plan else [],
        )
        if plan != expected:
            raise ValueError("Stabilized semantic shot plan is invalid")
        return int(shape[0]), int(shape[1])

    @property
    def depth_files(self) -> list[Path]:
        return [self.root / f"{path.stem}.png" for path in self.frame_files]

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _refresh_metadata_fingerprints(metadata: dict[str, Any]) -> None:
        metadata["state_fingerprint"] = canonical_json_hash(
            {
                "status": metadata["status"],
                "completed_shots": metadata["completed_shots"],
            }
        )
        metadata.pop("metadata_fingerprint", None)
        metadata["metadata_fingerprint"] = canonical_json_hash(metadata)

    def _new_metadata(self, completed_shots: list[dict[str, Any]]) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "schema_version": STABILIZED_DEPTH_SCHEMA_VERSION,
            "algorithm_version": STABILIZED_DEPTH_ALGORITHM_VERSION,
            "status": "building",
            "representation": "relative_disparity",
            "near_value": 1.0,
            "far_value": 0.0,
            "encoding": "uint16_png",
            "encoding_scale": 65535.0,
            "num_frames": len(self.frame_files),
            "semantic_identity": self.semantic_identity,
            "semantic_fingerprint": self.semantic_fingerprint,
            "execution_provenance": self.execution_provenance,
            "partial_resume_runtime_fingerprint": self.runtime_fingerprint,
            "completed_shots": completed_shots,
            "state_fingerprint": "",
            "payload_fingerprint": None,
            "artifact_fingerprint": None,
        }
        self._refresh_metadata_fingerprints(metadata)
        return metadata

    @staticmethod
    def _read_metadata(path: Path) -> dict[str, Any] | None:
        try:
            decoded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        return decoded if isinstance(decoded, dict) else None

    @staticmethod
    def _metadata_self_hash_valid(metadata: dict[str, Any]) -> bool:
        fingerprint = metadata.get("metadata_fingerprint")
        unhashed = {key: value for key, value in metadata.items() if key != "metadata_fingerprint"}
        return isinstance(fingerprint, str) and fingerprint == canonical_json_hash(unhashed)

    def _shot_record_valid(
        self,
        shot: dict[str, int],
        record: object,
    ) -> bool:
        if not isinstance(record, dict) or record.get("shot_id") != shot["shot_id"]:
            return False
        relative = f"shot_manifests/shot_{shot['shot_id']:06d}.json"
        if record.get("manifest") != relative:
            return False
        manifest_path = self.root / relative
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        if not isinstance(manifest, dict):
            return False
        if record.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest():
            return False
        fingerprint = manifest.get("manifest_fingerprint")
        unhashed = {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
        if not isinstance(fingerprint, str) or fingerprint != canonical_json_hash(unhashed):
            return False
        if (
            manifest.get("schema_version") != 1
            or manifest.get("shot_id") != shot["shot_id"]
            or manifest.get("start") != shot["start"]
            or manifest.get("end") != shot["end"]
        ):
            return False

        expected_names = [path.name for path in self.frame_files[shot["start"] : shot["end"]]]
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != len(expected_names):
            return False
        normalized: list[dict[str, str]] = []
        try:
            for name, file_record in zip(expected_names, files):
                if not isinstance(file_record, dict) or file_record.get("name") != name:
                    return False
                path = self.root / f"{Path(name).stem}.png"
                if not png_header_matches(path, shape=self.native_shape, bit_depth=16):
                    return False
                digest = self._sha256(path)
                if file_record.get("sha256") != digest:
                    return False
                normalized.append({"name": name, "sha256": digest})
        except OSError:
            return False
        shot_payload = canonical_json_hash(normalized)
        return (
            manifest.get("shot_payload_sha256") == shot_payload
            and record.get("shot_payload_sha256") == shot_payload
        )

    def audit(self) -> StabilizedStageAudit:
        """Inspect existing state without changing a byte."""

        all_ids = tuple(shot["shot_id"] for shot in self.shot_plan)
        if not self.metadata_path.is_file():
            return StabilizedStageAudit(
                complete=False,
                reset_required=False,
                reusable_shot_ids=(),
                invalid_shot_ids=(),
                pending_shot_ids=all_ids,
                reason="stabilized metadata is missing",
                metadata=None,
            )
        metadata = self._read_metadata(self.metadata_path)
        if metadata is None or not self._metadata_self_hash_valid(metadata):
            return StabilizedStageAudit(
                complete=False,
                reset_required=True,
                reusable_shot_ids=(),
                invalid_shot_ids=(),
                pending_shot_ids=all_ids,
                reason="stabilized metadata fingerprint is invalid",
                metadata=metadata,
            )
        if (
            metadata.get("schema_version") != STABILIZED_DEPTH_SCHEMA_VERSION
            or metadata.get("algorithm_version") != STABILIZED_DEPTH_ALGORITHM_VERSION
            or metadata.get("semantic_fingerprint") != self.semantic_fingerprint
            or metadata.get("semantic_identity") != self.semantic_identity
        ):
            return StabilizedStageAudit(
                complete=False,
                reset_required=True,
                reusable_shot_ids=(),
                invalid_shot_ids=(),
                pending_shot_ids=all_ids,
                reason="stabilized semantic identity changed",
                metadata=metadata,
            )

        if metadata.get("status") == "complete":
            try:
                validate_render_disparity_input(self.depth_files, self.frame_files)
            except ValueError:
                pass
            else:
                return StabilizedStageAudit(
                    complete=True,
                    reset_required=False,
                    reusable_shot_ids=all_ids,
                    invalid_shot_ids=(),
                    pending_shot_ids=(),
                    reason="complete stabilized artifact is valid",
                    metadata=metadata,
                )

        if metadata.get("partial_resume_runtime_fingerprint") != self.runtime_fingerprint:
            return StabilizedStageAudit(
                complete=False,
                reset_required=True,
                reusable_shot_ids=(),
                invalid_shot_ids=(),
                pending_shot_ids=all_ids,
                reason="partial stabilized runtime identity changed",
                metadata=metadata,
            )

        records = metadata.get("completed_shots")
        records_by_id = (
            {
                record.get("shot_id"): record
                for record in records
                if isinstance(record, dict) and isinstance(record.get("shot_id"), int)
            }
            if isinstance(records, list)
            else {}
        )
        reusable: list[int] = []
        invalid: list[int] = []
        pending: list[int] = []
        for shot in self.shot_plan:
            shot_id = shot["shot_id"]
            record = records_by_id.get(shot_id)
            if record is None:
                pending.append(shot_id)
            elif self._shot_record_valid(shot, record):
                reusable.append(shot_id)
            else:
                invalid.append(shot_id)
        return StabilizedStageAudit(
            complete=False,
            reset_required=False,
            reusable_shot_ids=tuple(reusable),
            invalid_shot_ids=tuple(invalid),
            pending_shot_ids=tuple(pending),
            reason="stabilized stage requires shot generation",
            metadata=metadata,
        )

    def _delete_shot_payloads(self, shot: dict[str, int]) -> None:
        for frame in self.frame_files[shot["start"] : shot["end"]]:
            (self.root / f"{frame.stem}.png").unlink(missing_ok=True)
            (self.root / f"{frame.stem}.tmp.png").unlink(missing_ok=True)
        (self.manifest_dir / f"shot_{shot['shot_id']:06d}.json").unlink(missing_ok=True)
        (self.manifest_dir / f"shot_{shot['shot_id']:06d}.json.tmp").unlink(missing_ok=True)

    def prepare(self, audit: StabilizedStageAudit) -> None:
        """Apply an audit only after orchestration preflight authorizes mutation."""

        if audit.complete:
            self._metadata = audit.metadata
            return
        if audit.reset_required and self.root.is_dir():
            for path in self.root.iterdir():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

        reusable = set() if audit.reset_required else set(audit.reusable_shot_ids)
        for shot in self.shot_plan:
            if shot["shot_id"] not in reusable:
                self._delete_shot_payloads(shot)
        previous_records = (
            audit.metadata.get("completed_shots", []) if audit.metadata is not None else []
        )
        records_by_id = {
            record.get("shot_id"): record for record in previous_records if isinstance(record, dict)
        }
        completed = [records_by_id[shot_id] for shot_id in sorted(reusable)]
        self._metadata = self._new_metadata(completed)
        self._atomic_write_json(self.metadata_path, self._metadata)

    def _shot(self, shot_id: int) -> dict[str, int]:
        if isinstance(shot_id, bool) or not isinstance(shot_id, int):
            raise ValueError("shot_id must be an integer")
        try:
            return self.shot_plan[shot_id]
        except IndexError as exc:
            raise ValueError(f"Unknown stabilized shot: {shot_id}") from exc

    @staticmethod
    def _atomic_write_png(path: Path, values: np.ndarray) -> None:
        temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
        if not cv2.imwrite(str(temporary), values):
            raise OSError(f"Could not write stabilized disparity: {path}")
        os.replace(temporary, path)

    def commit_shot(
        self,
        shot_id: int,
        outputs: Iterable[tuple[int, np.ndarray]],
    ) -> None:
        """Commit one complete shot; incomplete windows never enter metadata."""

        if self._metadata is None:
            raise RuntimeError("Stabilized store must be prepared before writing")
        shot = self._shot(shot_id)
        completed_ids = {record["shot_id"] for record in self._metadata.get("completed_shots", [])}
        if shot_id in completed_ids:
            return
        self._delete_shot_payloads(shot)
        expected_length = shot["end"] - shot["start"]
        file_records: list[dict[str, str]] = []
        consumed = 0
        try:
            for expected_local, item in enumerate(outputs):
                if not isinstance(item, tuple) or len(item) != 2:
                    raise ValueError("Stabilized shot outputs must be ordered index/value pairs")
                local_index, values = item
                if local_index != expected_local:
                    raise ValueError("Stabilized shot outputs are not ordered and contiguous")
                if expected_local >= expected_length:
                    raise ValueError("Stabilized shot emitted too many frames")
                array = np.asarray(values, dtype=np.float32)
                if array.shape != self.native_shape:
                    raise ValueError(
                        f"Stabilized output shape {array.shape} does not match {self.native_shape}"
                    )
                if not np.isfinite(array).all():
                    raise ValueError("Stabilized output must be finite")
                frame = self.frame_files[shot["start"] + expected_local]
                output_path = self.root / f"{frame.stem}.png"
                self._atomic_write_png(output_path, encode_canonical_png(array))
                file_records.append({"name": frame.name, "sha256": self._sha256(output_path)})
                consumed += 1
            if consumed != expected_length:
                raise ValueError(
                    f"Stabilized shot emitted {consumed} frames; expected {expected_length}"
                )

            shot_payload = canonical_json_hash(file_records)
            manifest: dict[str, Any] = {
                "schema_version": 1,
                "shot_id": shot_id,
                "start": shot["start"],
                "end": shot["end"],
                "files": file_records,
                "shot_payload_sha256": shot_payload,
            }
            manifest["manifest_fingerprint"] = canonical_json_hash(manifest)
            manifest_path = self.manifest_dir / f"shot_{shot_id:06d}.json"
            self._atomic_write_json(manifest_path, manifest)
            record = {
                "shot_id": shot_id,
                "manifest": f"shot_manifests/shot_{shot_id:06d}.json",
                "manifest_sha256": self._sha256(manifest_path),
                "shot_payload_sha256": shot_payload,
            }
            self._metadata["completed_shots"].append(record)
            self._metadata["completed_shots"].sort(key=lambda value: value["shot_id"])
            self._refresh_metadata_fingerprints(self._metadata)
            self._atomic_write_json(self.metadata_path, self._metadata)
        except BaseException:
            self._delete_shot_payloads(shot)
            raise

    def finalize(self) -> list[Path]:
        """Publish the immutable artifact only after every shot validates."""

        if self._metadata is None:
            raise RuntimeError("Stabilized store must be prepared before finalization")
        completed = self._metadata.get("completed_shots", [])
        completed_ids = [record.get("shot_id") for record in completed]
        expected_ids = [shot["shot_id"] for shot in self.shot_plan]
        if completed_ids != expected_ids:
            raise RuntimeError("Cannot finalize stabilized disparity before every shot completes")
        payload_records = [
            {
                "shot_id": record["shot_id"],
                "shot_payload_sha256": record["shot_payload_sha256"],
            }
            for record in completed
        ]
        self._metadata["status"] = "complete"
        self._metadata["payload_fingerprint"] = canonical_json_hash(payload_records)
        self._metadata["artifact_fingerprint"] = canonical_json_hash(
            {
                "semantic_fingerprint": self.semantic_fingerprint,
                "payload_fingerprint": self._metadata["payload_fingerprint"],
            }
        )
        self._refresh_metadata_fingerprints(self._metadata)
        self._atomic_write_json(self.metadata_path, self._metadata)
        validate_render_disparity_input(self.depth_files, self.frame_files)
        return self.depth_files
