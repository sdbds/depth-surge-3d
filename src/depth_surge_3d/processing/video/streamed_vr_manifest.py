"""Canonical input evidence for assembled PNG frames sent through image2pipe."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import BinaryIO

from ...io.durable_publish import OwnedSiblingTemporary
from ...io.file_identity import PersistedFileIdentityV1
from ...io.reservation_protocol import ReservedArtifactTransaction
from .encoding_provider import (
    FINAL_ENCODING_JSON_STREAM_BYTES,
    EncodingManifestTrace,
    EncodingSequenceItem,
    EncodingSequenceProvider,
    StagedEncodingInputManifest,
    _parse_png_header,
    _validated_items,
    audit_encoding_provider,
    canonical_json_bytes,
)
from .streamed_vr_evidence import (
    StreamedVrInvocationV1,
    StreamedVrOutputEvidenceV1,
)


@dataclass(frozen=True, slots=True)
class CompletedStreamedVrInputManifestV1:
    path: Path
    staged: StagedEncodingInputManifest


class StreamedVrInputManifestV1:
    """Write one bounded manifest while exact PNG payloads are piped to FFmpeg."""

    def __init__(
        self,
        *,
        job_root: Path,
        provider: EncodingSequenceProvider,
        expected_shape: tuple[int, int],
        reservation_transaction: ReservedArtifactTransaction | None = None,
        invocation: StreamedVrInvocationV1 | None = None,
        owned_target_path: Path | None = None,
        owned_temporary_path: Path | None = None,
    ) -> None:
        provider.validated()
        if provider.mode != "assembled_vr" or provider.frames_parent is None:
            raise ValueError("streamed VR manifest requires an assembled provider")
        width, height = expected_shape
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or isinstance(height, bool)
            or not isinstance(height, int)
            or width <= 0
            or height <= 0
        ):
            raise ValueError("streamed VR manifest shape must be positive integers")

        self.provider = provider
        self.expected_shape = expected_shape
        self.invocation = invocation
        self.provider_audit = audit_encoding_provider(provider)
        root = Path(job_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        self._reserved_transaction = reservation_transaction
        self._reserved_context = None
        self._handle: BinaryIO | None = None
        if (owned_target_path is None) != (owned_temporary_path is None):
            raise ValueError("streamed VR owned paths must be supplied together")
        if reservation_transaction is not None and owned_target_path is not None:
            raise ValueError("streamed VR manifest cannot have two storage authorities")
        if reservation_transaction is None:
            target = (
                root / f".depth-surge-streamed-input-{provider.audit_generation_id}.json"
                if owned_target_path is None
                else Path(owned_target_path).resolve()
            )
            temporary = (
                None if owned_temporary_path is None else Path(owned_temporary_path).resolve()
            )
            if target.parent != root or (temporary is not None and temporary.parent != root):
                raise ValueError("streamed VR owned paths escaped the job root")
            self._owned: OwnedSiblingTemporary | None = OwnedSiblingTemporary.create(
                target,
                temporary=temporary,
            )
            self._handle = self._owned.handle
        else:
            if reservation_transaction.job_root.resolve() != root:
                raise ValueError("streamed VR reservation belongs to a different job root")
            self._owned = None
            self._reserved_context = reservation_transaction.open_authenticated_source(
                writable=True
            )
            self._handle = self._reserved_context.__enter__()
            self._handle.seek(0)
        self._items = iter(_validated_items(provider))
        self._next_item: EncodingSequenceItem | None = next(self._items, None)
        self._self_digest = hashlib.sha256()
        self._trace = EncodingManifestTrace()
        self._count = 0
        self._finished = False
        self._completed: CompletedStreamedVrInputManifestV1 | None = None
        self._completed_output_evidence: StreamedVrOutputEvidenceV1 | None = None
        self._fingerprint_offset = 0

        try:
            self._write_header()
        except BaseException:
            self.abort()
            raise

    @property
    def path(self) -> Path:
        if self._reserved_transaction is not None:
            return self._reserved_transaction.source_path
        owned = self._owned
        if owned is None:
            raise ValueError("streamed VR manifest is already cleaned")
        return owned.target

    def _write(self, payload: bytes, *, fingerprinted: bool = True) -> None:
        handle = self._handle
        if handle is None:
            raise ValueError("streamed VR manifest is already cleaned")
        if handle.write(payload) != len(payload):
            raise OSError("short streamed VR manifest write")
        if fingerprinted:
            self._self_digest.update(payload)

    def _write_header(self) -> None:
        placeholder = b"0" * 64
        algorithm = (
            b"encoding-input-sequence-v3"
            if self.invocation is not None
            else b"encoding-input-sequence-v2"
        )
        prefix = b'{"algorithm_version":"' + algorithm + b'","fingerprint":"'
        self._write(
            prefix + placeholder + b'"',
            fingerprinted=False,
        )
        self._fingerprint_offset = len(prefix)
        self._self_digest.update(b'{"algorithm_version":"' + algorithm + b'"')
        self._write(b',"frame_names":[')
        for source_index, item in enumerate(_validated_items(self.provider)):
            if source_index:
                self._write(b",")
            self._write(canonical_json_bytes(item.frame_name))
        self._write(b"]")
        if self.invocation is not None:
            self._write(
                b',"frame_stream_invocation":' + canonical_json_bytes(self.invocation.to_object())
            )
        self._write(b',"frames":[')
        if self.invocation is not None:
            handle = self._handle
            assert handle is not None
            handle.flush()
            os.fsync(handle.fileno())

    def append(self, frame_name: str, png_payload: bytes) -> None:
        if self._finished or self._handle is None:
            raise ValueError("streamed VR manifest is already closed")
        item = self._next_item
        if item is None or frame_name != item.frame_name:
            raise ValueError("streamed VR frames are not in source order")
        if not isinstance(png_payload, bytes):
            raise TypeError("streamed VR PNG payload must be immutable bytes")
        relative_path = item.frame_relative_path
        assert relative_path is not None
        header = _parse_png_header(png_payload[:33], Path(relative_path.as_posix()))
        if (header.width, header.height) != self.expected_shape:
            raise ValueError("streamed VR PNG shape changed")
        identity = {
            "relative_path": relative_path.as_posix(),
            "sha256": hashlib.sha256(png_payload).hexdigest(),
            "byte_count": len(png_payload),
            "png_header": {
                "width": header.width,
                "height": header.height,
                "bit_depth": header.bit_depth,
                "color_type": header.color_type,
            },
        }
        if self._count:
            self._write(b",")
        self._write(canonical_json_bytes(identity))
        self._trace.maximum_live_identities = 1
        self._trace.maximum_stream_bytes = max(
            self._trace.maximum_stream_bytes,
            len(png_payload),
        )
        self._count += 1
        self._next_item = next(self._items, None)

    def complete(
        self,
        *,
        output_evidence: StreamedVrOutputEvidenceV1 | None = None,
    ) -> CompletedStreamedVrInputManifestV1:
        if self._completed is not None:
            if output_evidence != self._completed_output_evidence:
                raise ValueError("streamed VR completion evidence changed")
            return self._completed
        if self._finished or self._handle is None:
            raise ValueError("streamed VR manifest is already closed")
        if self._count != self.provider.frame_count or self._next_item is not None:
            raise ValueError("streamed VR manifest frame count is incomplete")
        handle = self._handle
        assert handle is not None
        try:
            if self.invocation is None:
                if output_evidence is not None:
                    raise ValueError("legacy streamed manifest cannot carry output evidence")
                footer = b'],"left":null,"mode":"assembled_vr","right":null,' b'"schema_version":1}'
            else:
                if output_evidence is None:
                    raise ValueError("recoverable streamed manifest lacks output evidence")
                footer = (
                    b'],"left":null,"mode":"assembled_vr","right":null,'
                    b'"schema_version":2,"streamed_output":'
                    + canonical_json_bytes(output_evidence.to_object())
                    + b"}"
                )
            self._write(footer)
            end = handle.tell()
            fingerprint = self._self_digest.hexdigest()
            handle.seek(self._fingerprint_offset)
            if handle.write(fingerprint.encode("ascii")) != 64:
                raise OSError("short streamed VR manifest fingerprint write")
            handle.seek(0)
            raw_digest = hashlib.sha256()
            remaining = end
            while remaining:
                chunk = handle.read(min(FINAL_ENCODING_JSON_STREAM_BYTES, remaining))
                if not chunk:
                    raise OSError("streamed VR manifest ended before its written extent")
                raw_digest.update(chunk)
                remaining -= len(chunk)
            handle.seek(end)
            handle.flush()
            os.fsync(handle.fileno())
            if self._reserved_transaction is None:
                assert self._owned is not None
                self._owned.replace()
                self._handle = None
            else:
                self._close_reserved_source()
                phase = self._reserved_transaction.validate_streamed_physical_phase(
                    prefix_byte_count=end,
                    prefix_sha256=raw_digest.hexdigest(),
                )
                if phase != "full-source":
                    raise OSError("streamed VR reservation did not retain its full source")
            staged = StagedEncodingInputManifest(
                provider_audit=self.provider_audit,
                fingerprint=fingerprint,
                raw_sha256=raw_digest.hexdigest(),
                raw_byte_count=end,
                first_headers=(self.expected_shape,),
                scale_required=False,
                trace=self._trace,
            )
            self._finished = True
            self._completed = CompletedStreamedVrInputManifestV1(
                path=self.path,
                staged=staged,
            )
            self._completed_output_evidence = output_evidence
            return self._completed
        except BaseException:
            self.abort()
            raise

    @property
    def owned_source_identity(self) -> PersistedFileIdentityV1:
        owned = self._owned
        if owned is None:
            raise ValueError("streamed VR manifest is not an owned work file")
        return owned.source_identity

    def cleanup(self) -> None:
        if self._reserved_transaction is not None:
            self._close_reserved_source()
            self._reserved_transaction = None
            self._handle = None
            return
        owned = self._owned
        if owned is None:
            return
        if self._completed is not None:
            owned.remove_published_target()
        owned.cleanup()
        self._owned = None
        self._handle = None

    def abort(self) -> None:
        if self._reserved_transaction is not None:
            transaction = self._reserved_transaction
            self._close_reserved_source()
            transaction.reset_authenticated_source_to_zero_full()
            self._reserved_transaction = None
            self._handle = None
            self._finished = True
            return
        owned = self._owned
        if owned is None:
            return
        if self._completed is not None:
            owned.remove_published_target()
        owned.cleanup()
        self._owned = None
        self._handle = None
        self._finished = True

    def _close_reserved_source(self) -> None:
        context = self._reserved_context
        if context is None:
            return
        self._reserved_context = None
        self._handle = None
        context.__exit__(None, None, None)
