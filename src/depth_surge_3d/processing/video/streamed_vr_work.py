"""Durable prelaunch ownership for streamed VR work files."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re
from typing import Any, cast

from ...io.file_identity import (
    DirectoryIdentityV1,
    PersistedFileIdentityV1,
    capture_directory_identity,
    file_identity_from_open_handle,
)
from ...io.nofollow_io import AcquiredRoot, NoFollowPathError
from ...io.reservation_protocol import (
    ReservationConflictError,
    canonical_json_bytes,
    load_job_control,
    publish_write_ahead_intent,
    recover_control_artifact_pair_v1,
    sha256_canonical,
)
from ...io.settings_artifact import ProcessingSettingsArtifactV5
from .streamed_vr_evidence import StreamedVrInvocationV1


STREAMED_VR_WORK_INDEX_RELATIVE_PATH = Path(".depth-surge-streamed-vr-work-v1.json")
STREAMED_VR_WORK_READY_RELATIVE_PATH = Path(".depth-surge-streamed-vr-work-ready-v1.json")
_HEX32 = re.compile(r"^[0-9a-f]{32}$", flags=re.ASCII)
_HEX64 = re.compile(r"^[0-9a-f]{64}$", flags=re.ASCII)


def _closed_dict(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ReservationConflictError(f"{label} schema is not closed")
    return cast(dict[str, Any], value)


def _hex(value: object, pattern: re.Pattern[str], label: str) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise ReservationConflictError(f"{label} is invalid")
    return value


def _direct_child(value: object, label: str) -> str:
    if type(value) is not str:
        raise ReservationConflictError(f"{label} path is invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or len(path.parts) != 1
        or path.parts[0] in {"", ".", ".."}
        or "\\" in value
    ):
        raise ReservationConflictError(f"{label} path is invalid")
    return value


@dataclass(frozen=True, slots=True)
class StreamedVrWorkIndexV1:
    job_id: str
    job_root_identity: DirectoryIdentityV1
    settings_identity_fingerprint: str
    settings_revision_at_preflight: int
    settings_raw_sha256_at_preflight: str
    publication_generation: str
    work_generation: str
    expected_final_relative_path: str
    manifest_target_relative_path: str
    manifest_temporary_relative_path: str
    output_relative_path: str
    ready_relative_path: str
    invocation: StreamedVrInvocationV1
    fingerprint: str

    def _base_object(self) -> dict[str, Any]:
        return {
            "algorithm_version": "streamed-vr-work-reservation-v1",
            "expected_final_relative_path": self.expected_final_relative_path,
            "invocation": self.invocation.to_object(),
            "job_id": self.job_id,
            "job_root_identity": self.job_root_identity.to_object(),
            "manifest_target_relative_path": self.manifest_target_relative_path,
            "manifest_temporary_relative_path": self.manifest_temporary_relative_path,
            "output_relative_path": self.output_relative_path,
            "publication_generation": self.publication_generation,
            "ready_relative_path": self.ready_relative_path,
            "schema_version": 1,
            "settings_identity_fingerprint": self.settings_identity_fingerprint,
            "settings_raw_sha256_at_preflight": self.settings_raw_sha256_at_preflight,
            "settings_revision_at_preflight": self.settings_revision_at_preflight,
            "work_generation": self.work_generation,
        }

    def to_object(self) -> dict[str, Any]:
        return {**self._base_object(), "fingerprint": self.fingerprint}

    def validated(self) -> "StreamedVrWorkIndexV1":
        _hex(self.job_id, _HEX32, "stream work job id")
        _hex(
            self.settings_identity_fingerprint,
            _HEX64,
            "stream work settings identity",
        )
        _hex(
            self.settings_raw_sha256_at_preflight,
            _HEX64,
            "stream work settings hash",
        )
        _hex(self.publication_generation, _HEX32, "stream work publication generation")
        _hex(self.work_generation, _HEX32, "stream work generation")
        _hex(self.fingerprint, _HEX64, "stream work fingerprint")
        if not isinstance(self.job_root_identity, DirectoryIdentityV1):
            raise ReservationConflictError("stream work root identity is invalid")
        if (
            type(self.settings_revision_at_preflight) is not int
            or not 0 <= self.settings_revision_at_preflight < 1 << 64
        ):
            raise ReservationConflictError("stream work settings revision is invalid")
        expected = {
            "manifest_target_relative_path": (
                f".depth-surge-streamed-vr-{self.work_generation}-input.json"
            ),
            "manifest_temporary_relative_path": (
                f".depth-surge-streamed-vr-{self.work_generation}-input.part"
            ),
            "output_relative_path": (
                f".depth-surge-streamed-vr-{self.work_generation}-output.tmp.mp4"
            ),
            "ready_relative_path": STREAMED_VR_WORK_READY_RELATIVE_PATH.as_posix(),
        }
        for field, required in expected.items():
            value = _direct_child(getattr(self, field), field)
            if value != required:
                raise ReservationConflictError(f"stream work {field} is noncanonical")
        _direct_child(self.expected_final_relative_path, "stream work final output")
        if (
            self.invocation.job_id != self.job_id
            or self.invocation.settings_identity_fingerprint != self.settings_identity_fingerprint
            or self.invocation.publication_generation != self.publication_generation
        ):
            raise ReservationConflictError("stream work invocation crosses its authority")
        if self.fingerprint != sha256_canonical(self._base_object()):
            raise ReservationConflictError("stream work fingerprint differs")
        return self

    @classmethod
    def build(
        cls,
        *,
        job_id: str,
        job_root_identity: DirectoryIdentityV1,
        settings_identity_fingerprint: str,
        settings_revision_at_preflight: int,
        settings_raw_sha256_at_preflight: str,
        publication_generation: str,
        work_generation: str,
        expected_final_relative_path: str,
        invocation: StreamedVrInvocationV1,
    ) -> "StreamedVrWorkIndexV1":
        manifest_target = f".depth-surge-streamed-vr-{work_generation}-input.json"
        manifest_temporary = f".depth-surge-streamed-vr-{work_generation}-input.part"
        output = f".depth-surge-streamed-vr-{work_generation}-output.tmp.mp4"
        ready = STREAMED_VR_WORK_READY_RELATIVE_PATH.as_posix()
        provisional = cls(
            job_id=job_id,
            job_root_identity=job_root_identity,
            settings_identity_fingerprint=settings_identity_fingerprint,
            settings_revision_at_preflight=settings_revision_at_preflight,
            settings_raw_sha256_at_preflight=settings_raw_sha256_at_preflight,
            publication_generation=publication_generation,
            work_generation=work_generation,
            expected_final_relative_path=expected_final_relative_path,
            manifest_target_relative_path=manifest_target,
            manifest_temporary_relative_path=manifest_temporary,
            output_relative_path=output,
            ready_relative_path=ready,
            invocation=invocation,
            fingerprint="",
        )
        return cls(
            job_id=job_id,
            job_root_identity=job_root_identity,
            settings_identity_fingerprint=settings_identity_fingerprint,
            settings_revision_at_preflight=settings_revision_at_preflight,
            settings_raw_sha256_at_preflight=settings_raw_sha256_at_preflight,
            publication_generation=publication_generation,
            work_generation=work_generation,
            expected_final_relative_path=expected_final_relative_path,
            manifest_target_relative_path=manifest_target,
            manifest_temporary_relative_path=manifest_temporary,
            output_relative_path=output,
            ready_relative_path=ready,
            invocation=invocation,
            fingerprint=sha256_canonical(provisional._base_object()),
        ).validated()

    @classmethod
    def from_object(cls, value: object) -> "StreamedVrWorkIndexV1":
        item = _closed_dict(
            value,
            {
                "algorithm_version",
                "expected_final_relative_path",
                "fingerprint",
                "invocation",
                "job_id",
                "job_root_identity",
                "manifest_target_relative_path",
                "manifest_temporary_relative_path",
                "output_relative_path",
                "publication_generation",
                "ready_relative_path",
                "schema_version",
                "settings_identity_fingerprint",
                "settings_raw_sha256_at_preflight",
                "settings_revision_at_preflight",
                "work_generation",
            },
            "stream work index",
        )
        if (
            item["algorithm_version"] != "streamed-vr-work-reservation-v1"
            or item["schema_version"] != 1
        ):
            raise ReservationConflictError("stream work index version is invalid")
        try:
            invocation = StreamedVrInvocationV1.from_object(item["invocation"])
            root_identity = DirectoryIdentityV1.from_object(item["job_root_identity"])
        except (TypeError, ValueError) as error:
            raise ReservationConflictError("stream work nested evidence is invalid") from error
        return cls(
            job_id=item["job_id"],
            job_root_identity=root_identity,
            settings_identity_fingerprint=item["settings_identity_fingerprint"],
            settings_revision_at_preflight=item["settings_revision_at_preflight"],
            settings_raw_sha256_at_preflight=item["settings_raw_sha256_at_preflight"],
            publication_generation=item["publication_generation"],
            work_generation=item["work_generation"],
            expected_final_relative_path=item["expected_final_relative_path"],
            manifest_target_relative_path=item["manifest_target_relative_path"],
            manifest_temporary_relative_path=item["manifest_temporary_relative_path"],
            output_relative_path=item["output_relative_path"],
            ready_relative_path=item["ready_relative_path"],
            invocation=invocation,
            fingerprint=item["fingerprint"],
        ).validated()


@dataclass(frozen=True, slots=True)
class StreamedVrWorkReadyV1:
    index_fingerprint: str
    manifest_identity: PersistedFileIdentityV1
    output_identity: PersistedFileIdentityV1
    fingerprint: str

    def _base_object(self) -> dict[str, Any]:
        return {
            "algorithm_version": "streamed-vr-work-ready-v1",
            "index_fingerprint": self.index_fingerprint,
            "manifest_identity": self.manifest_identity.to_object(),
            "output_identity": self.output_identity.to_object(),
            "schema_version": 1,
        }

    def to_object(self) -> dict[str, Any]:
        return {**self._base_object(), "fingerprint": self.fingerprint}

    def validated(self) -> "StreamedVrWorkReadyV1":
        _hex(self.index_fingerprint, _HEX64, "stream work ready index fingerprint")
        _hex(self.fingerprint, _HEX64, "stream work ready fingerprint")
        for identity in (self.manifest_identity, self.output_identity):
            if not isinstance(identity, PersistedFileIdentityV1) or identity.link_count != 1:
                raise ReservationConflictError("stream work ready identity is invalid")
        if self.fingerprint != sha256_canonical(self._base_object()):
            raise ReservationConflictError("stream work ready fingerprint differs")
        return self

    @classmethod
    def build(
        cls,
        *,
        index_fingerprint: str,
        manifest_identity: PersistedFileIdentityV1,
        output_identity: PersistedFileIdentityV1,
    ) -> "StreamedVrWorkReadyV1":
        provisional = cls(
            index_fingerprint=index_fingerprint,
            manifest_identity=manifest_identity,
            output_identity=output_identity,
            fingerprint="",
        )
        return cls(
            index_fingerprint=index_fingerprint,
            manifest_identity=manifest_identity,
            output_identity=output_identity,
            fingerprint=sha256_canonical(provisional._base_object()),
        ).validated()

    @classmethod
    def from_object(cls, value: object) -> "StreamedVrWorkReadyV1":
        item = _closed_dict(
            value,
            {
                "algorithm_version",
                "fingerprint",
                "index_fingerprint",
                "manifest_identity",
                "output_identity",
                "schema_version",
            },
            "stream work ready descriptor",
        )
        if item["algorithm_version"] != "streamed-vr-work-ready-v1" or item["schema_version"] != 1:
            raise ReservationConflictError("stream work ready version is invalid")
        try:
            manifest_identity = PersistedFileIdentityV1.from_object(item["manifest_identity"])
            output_identity = PersistedFileIdentityV1.from_object(item["output_identity"])
        except (TypeError, ValueError) as error:
            raise ReservationConflictError("stream work ready identity is invalid") from error
        return cls(
            index_fingerprint=item["index_fingerprint"],
            manifest_identity=manifest_identity,
            output_identity=output_identity,
            fingerprint=item["fingerprint"],
        ).validated()


class StreamedVrWorkReservationV1:
    """Index-first owner for pre-diagnostics manifest and FFmpeg work output."""

    def __init__(
        self,
        *,
        job_root: Path,
        index: StreamedVrWorkIndexV1,
        ready: StreamedVrWorkReadyV1 | None = None,
    ) -> None:
        self.job_root = Path(job_root).resolve()
        self.index = index.validated()
        self.ready = ready.validated() if ready is not None else None
        if self.ready is not None and self.ready.index_fingerprint != self.index.fingerprint:
            raise ReservationConflictError("stream work ready belongs to another index")

    @property
    def index_path(self) -> Path:
        return self.job_root / STREAMED_VR_WORK_INDEX_RELATIVE_PATH

    @property
    def ready_path(self) -> Path:
        return self.job_root / self.index.ready_relative_path

    @property
    def manifest_target_path(self) -> Path:
        return self.job_root / self.index.manifest_target_relative_path

    @property
    def manifest_temporary_path(self) -> Path:
        return self.job_root / self.index.manifest_temporary_relative_path

    @property
    def output_path(self) -> Path:
        return self.job_root / self.index.output_relative_path

    @staticmethod
    def _settings_raw(
        job_root: Path,
        *,
        expected_root_identity: DirectoryIdentityV1 | None = None,
    ) -> bytes:
        root = Path(job_root).resolve()
        path = root / "processing-settings-v5.json"
        try:
            with AcquiredRoot(root) as acquired:
                if (
                    expected_root_identity is not None
                    and acquired.persistent_identity != expected_root_identity
                ):
                    raise ReservationConflictError("stream work job-root identity changed")
                with acquired.open_regular(path) as handle:
                    identity = file_identity_from_open_handle(handle)
                    raw = handle.read(512 * 1024 + 1)
                    if (
                        file_identity_from_open_handle(handle) != identity
                        or acquired.persistent_regular_identity(path) != identity
                    ):
                        raise ReservationConflictError(
                            "stream work settings identity changed during read"
                        )
                if (
                    expected_root_identity is not None
                    and acquired.persistent_identity != expected_root_identity
                ):
                    raise ReservationConflictError("stream work job-root identity changed")
        except (NoFollowPathError, FileNotFoundError, OSError) as error:
            raise ReservationConflictError("stream work settings are unavailable") from error
        if not 0 < len(raw) <= 512 * 1024:
            raise ReservationConflictError("stream work settings exceed their bound")
        return raw

    def validate_current_authority(self) -> ProcessingSettingsArtifactV5:
        if capture_directory_identity(self.job_root) != self.index.job_root_identity:
            raise ReservationConflictError("stream work job-root identity changed")
        control = load_job_control(self.job_root)
        if (
            control.job_id != self.index.job_id
            or control.settings_identity_fingerprint != self.index.settings_identity_fingerprint
            or control.job_root_identity != self.index.job_root_identity
        ):
            raise ReservationConflictError("stream work differs from job control")
        raw = self._settings_raw(
            self.job_root,
            expected_root_identity=self.index.job_root_identity,
        )
        settings = ProcessingSettingsArtifactV5.parse(raw)
        values = settings.to_object()
        metadata = values["metadata"]
        if (
            metadata["job_id"] != self.index.job_id
            or metadata["settings_identity_fingerprint"] != self.index.settings_identity_fingerprint
            or values["output_info"]["expected_final_relative_path"]
            != self.index.expected_final_relative_path
        ):
            raise ReservationConflictError("stream work differs from current settings")
        precommit = (
            settings.revision == self.index.settings_revision_at_preflight
            and hashlib.sha256(raw).hexdigest() == self.index.settings_raw_sha256_at_preflight
        )
        producer_committed = (
            settings.revision == self.index.settings_revision_at_preflight + 1
            and metadata["final_media_producer_contract_version"] == 4
            and metadata["final_media_publication_generation"] == self.index.publication_generation
        )
        if not precommit and not producer_committed:
            raise ReservationConflictError("stream work settings authority advanced illegally")
        return settings

    @classmethod
    def prepare(
        cls,
        *,
        job_root: Path,
        settings: ProcessingSettingsArtifactV5,
        publication_generation: str,
        work_generation: str,
        expected_final_relative_path: str,
        invocation: StreamedVrInvocationV1,
    ) -> "StreamedVrWorkReservationV1":
        root = Path(job_root).resolve()
        if cls.load(root) is not None:
            raise ReservationConflictError("an active streamed work reservation already exists")
        root_identity = capture_directory_identity(root)
        settings_raw = cls._settings_raw(
            root,
            expected_root_identity=root_identity,
        )
        if ProcessingSettingsArtifactV5.parse(settings_raw) != settings:
            raise ReservationConflictError("stream work settings changed at preflight")
        metadata = settings.to_object()["metadata"]
        index = StreamedVrWorkIndexV1.build(
            job_id=metadata["job_id"],
            job_root_identity=root_identity,
            settings_identity_fingerprint=metadata["settings_identity_fingerprint"],
            settings_revision_at_preflight=settings.revision,
            settings_raw_sha256_at_preflight=hashlib.sha256(settings_raw).hexdigest(),
            publication_generation=publication_generation,
            work_generation=work_generation,
            expected_final_relative_path=expected_final_relative_path,
            invocation=invocation,
        )
        candidates = (
            root / index.manifest_target_relative_path,
            root / index.manifest_temporary_relative_path,
            root / index.output_relative_path,
            root / index.ready_relative_path,
        )
        if any(path.exists() or path.is_symlink() for path in candidates):
            raise ReservationConflictError("stream work path collides at preflight")
        publish_write_ahead_intent(
            root / STREAMED_VR_WORK_INDEX_RELATIVE_PATH,
            index.to_object(),
            expected_parent_identity=index.job_root_identity,
        )
        reservation = cls(job_root=root, index=index)
        reservation.validate_current_authority()
        return reservation

    @classmethod
    def load(cls, job_root: Path) -> "StreamedVrWorkReservationV1 | None":
        root = Path(job_root).resolve()
        index_path = root / STREAMED_VR_WORK_INDEX_RELATIVE_PATH

        def parse_index(raw: bytes) -> StreamedVrWorkIndexV1:
            import json

            value = json.loads(raw)
            if canonical_json_bytes(value) != raw:
                raise ReservationConflictError("stream work index is noncanonical")
            return StreamedVrWorkIndexV1.from_object(value)

        def reject_malformed_temporary() -> None:
            raise ReservationConflictError("malformed stream work intent is retained fail-closed")

        def validate_index(value: object) -> None:
            cls(
                job_root=root,
                index=cast(StreamedVrWorkIndexV1, value),
            ).validate_current_authority()

        recovered = recover_control_artifact_pair_v1(
            index_path,
            max_bytes=256 * 1024,
            parser=parse_index,
            prepublish_validate=validate_index,
            preunlink_validate=reject_malformed_temporary,
        )
        if recovered is None:
            return None
        if not isinstance(recovered, StreamedVrWorkIndexV1):
            raise ReservationConflictError("stream work index type changed")
        reservation = cls(job_root=root, index=recovered)
        reservation.validate_current_authority()
        reservation.ready = reservation._load_ready()
        return reservation

    def _load_ready(self) -> StreamedVrWorkReadyV1 | None:
        def parse_ready(raw: bytes) -> StreamedVrWorkReadyV1:
            import json

            value = json.loads(raw)
            if canonical_json_bytes(value) != raw:
                raise ReservationConflictError("stream work ready is noncanonical")
            ready = StreamedVrWorkReadyV1.from_object(value)
            if ready.index_fingerprint != self.index.fingerprint:
                raise ReservationConflictError("stream work ready crosses its index")
            return ready

        def validate_ready(_value: object) -> None:
            self.validate_current_authority()

        def validate_ready_temporary_removal() -> None:
            # The published work index owns this exact ready name.  A malformed
            # create-new temporary has not crossed the ready publication boundary.
            self.validate_current_authority()

        recovered = recover_control_artifact_pair_v1(
            self.ready_path,
            max_bytes=64 * 1024,
            parser=parse_ready,
            # Completion moves the manifest to its target name and handoff moves
            # the output to the final reservation sibling.  The coordinator that
            # knows the durable phase validates those locations after parsing.
            prepublish_validate=validate_ready,
            preunlink_validate=validate_ready_temporary_removal,
            expected_parent_identity=self.index.job_root_identity,
        )
        if recovered is None:
            return None
        if not isinstance(recovered, StreamedVrWorkReadyV1):
            raise ReservationConflictError("stream work ready type changed")
        return recovered

    def publish_ready(
        self,
        *,
        manifest_identity: PersistedFileIdentityV1,
        output_identity: PersistedFileIdentityV1,
    ) -> StreamedVrWorkReadyV1:
        self.validate_current_authority()
        ready = StreamedVrWorkReadyV1.build(
            index_fingerprint=self.index.fingerprint,
            manifest_identity=manifest_identity,
            output_identity=output_identity,
        )
        self._validate_work_identity(self.manifest_temporary_path, manifest_identity)
        self._validate_work_identity(self.output_path, output_identity)
        publish_write_ahead_intent(
            self.ready_path,
            ready.to_object(),
            expected_parent_identity=self.index.job_root_identity,
        )
        self.ready = ready
        return ready

    def _single_manifest_location(
        self,
        expected: PersistedFileIdentityV1,
        *,
        allow_absent: bool,
    ) -> Path | None:
        candidates = tuple(
            path
            for path in (self.manifest_temporary_path, self.manifest_target_path)
            if path.exists() or path.is_symlink()
        )
        if len(candidates) > 1:
            raise ReservationConflictError("stream work has two manifest generations")
        if not candidates:
            if allow_absent:
                return None
            raise ReservationConflictError("stream work manifest is unavailable")
        self._validate_work_identity(candidates[0], expected)
        return candidates[0]

    def validate_ready_work_files(self) -> tuple[Path, Path]:
        """Authenticate the exact manifest and output names before handoff."""

        ready = self.ready if self.ready is not None else self._load_ready()
        if ready is None:
            raise ReservationConflictError("stream work lacks ready evidence")
        manifest = self._single_manifest_location(
            ready.manifest_identity,
            allow_absent=False,
        )
        assert manifest is not None
        self._validate_work_identity(self.output_path, ready.output_identity)
        return manifest, self.output_path

    def _validate_work_identity(
        self,
        path: Path,
        expected: PersistedFileIdentityV1,
    ) -> None:
        try:
            with AcquiredRoot(self.job_root) as acquired:
                if acquired.persistent_identity != self.index.job_root_identity:
                    raise ReservationConflictError("stream work root identity changed")
                actual = acquired.persistent_regular_identity(path)
        except (NoFollowPathError, FileNotFoundError, OSError) as error:
            raise ReservationConflictError("stream work file is unavailable") from error
        if actual != expected or actual.link_count != 1:
            raise ReservationConflictError("stream work file identity changed")

    def _unlink_work_file(
        self,
        path: Path,
        *,
        expected: PersistedFileIdentityV1 | None,
        missing_ok: bool = True,
    ) -> None:
        try:
            with AcquiredRoot(self.job_root) as acquired:
                if acquired.persistent_identity != self.index.job_root_identity:
                    raise ReservationConflictError("stream work root identity changed")
                transient = acquired.regular_identity(path, missing_ok=missing_ok)
                if transient is None:
                    return
                persistent = acquired.persistent_regular_identity(path)
                if not isinstance(persistent, PersistedFileIdentityV1):
                    raise ReservationConflictError("stream work cleanup identity is invalid")
                if persistent.link_count != 1 or (expected is not None and persistent != expected):
                    raise ReservationConflictError("stream work cleanup identity changed")
                acquired.unlink_owned_regular(
                    path,
                    expected_identity=transient,
                    missing_ok=missing_ok,
                )
                if acquired.regular_identity(path, missing_ok=True) is not None:
                    raise ReservationConflictError("stream work cleanup did not persist")
        except (NoFollowPathError, FileNotFoundError, OSError) as error:
            raise ReservationConflictError("stream work cleanup failed") from error

    def _unlink_control(self, path: Path, expected_raw: bytes) -> None:
        try:
            with AcquiredRoot(self.job_root) as acquired:
                if acquired.persistent_identity != self.index.job_root_identity:
                    raise ReservationConflictError("stream work control root changed")
                with acquired.open_regular(path) as handle:
                    if handle.read(len(expected_raw) + 1) != expected_raw:
                        raise ReservationConflictError("stream work control bytes changed")
                transient = acquired.regular_identity(path)
                persistent = acquired.persistent_regular_identity(path)
                if not isinstance(persistent, PersistedFileIdentityV1):
                    raise ReservationConflictError("stream work control identity is invalid")
                if persistent.link_count != 1:
                    raise ReservationConflictError("stream work control is not single-link")
                acquired.unlink_owned_regular(path, expected_identity=transient)
        except (NoFollowPathError, FileNotFoundError, OSError) as error:
            raise ReservationConflictError("stream work control cleanup failed") from error

    def discard(self) -> None:
        """Remove only files owned by this durable prelaunch generation."""

        self.validate_current_authority()
        observed_ready = self._load_ready()
        if self.ready is not None and observed_ready is not None and self.ready != observed_ready:
            raise ReservationConflictError("stream work ready evidence changed")
        ready = self.ready if self.ready is not None else observed_ready
        expected_manifest = ready.manifest_identity if ready is not None else None
        expected_output = ready.output_identity if ready is not None else None
        present_manifest_paths = [
            path
            for path in (self.manifest_target_path, self.manifest_temporary_path)
            if path.exists() or path.is_symlink()
        ]
        if len(present_manifest_paths) > 1:
            raise ReservationConflictError("stream work has two manifest generations")
        for path in present_manifest_paths:
            self._unlink_work_file(path, expected=expected_manifest)
        self._unlink_work_file(self.output_path, expected=expected_output)
        if ready is not None and (self.ready_path.exists() or self.ready_path.is_symlink()):
            self._unlink_control(self.ready_path, canonical_json_bytes(ready.to_object()))
        self._unlink_control(self.index_path, canonical_json_bytes(self.index.to_object()))
        self.ready = None

    def retire_handoff(
        self,
        *,
        final_sibling_path: Path,
        output_evidence: PersistedFileIdentityV1 | None = None,
    ) -> None:
        """Retire work authority only after the final index owns the same output inode."""

        self.validate_current_authority()
        ready = self.ready if self.ready is not None else self._load_ready()
        if self.output_path.exists() or self.output_path.is_symlink():
            raise ReservationConflictError("stream work output still occupies its old name")
        expected_output = ready.output_identity if ready is not None else output_evidence
        if expected_output is None:
            raise ReservationConflictError("stream work handoff lacks output evidence")
        if output_evidence is not None and output_evidence != expected_output:
            raise ReservationConflictError("stream work handoff output evidence differs")
        self._validate_work_identity(Path(final_sibling_path), expected_output)
        if ready is None:
            if any(
                path.exists() or path.is_symlink()
                for path in (self.manifest_target_path, self.manifest_temporary_path)
            ):
                raise ReservationConflictError(
                    "stream work manifest remains after ready retirement"
                )
        else:
            manifest = self._single_manifest_location(
                ready.manifest_identity,
                allow_absent=True,
            )
            if manifest is not None:
                self._unlink_work_file(manifest, expected=ready.manifest_identity)
            self._unlink_control(self.ready_path, canonical_json_bytes(ready.to_object()))
        self._unlink_control(self.index_path, canonical_json_bytes(self.index.to_object()))
        self.ready = None
