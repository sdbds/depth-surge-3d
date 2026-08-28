"""
Video encoding module.

Handles FFmpeg-based video encoding with hardware acceleration support (NVENC).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import os
import re
import secrets
import subprocess
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, BinaryIO, Iterator, Literal, cast

from ...io.operations import (
    get_frame_files,
    get_video_info_ffprobe,
    recover_processing_control_prefix_v1,
    verify_ffmpeg_installation,
)
from ...io.nofollow_io import AcquiredRoot, NoFollowPathError
from ...io.file_identity import (
    DirectoryIdentityV1,
    PersistedFileIdentityV1,
    file_identity_from_open_handle,
)
from ...io.reservation_protocol import (
    SETTINGS_ARTIFACT_RELATIVE_PATH,
    ReservationConflictError,
    ReservationIncompleteError,
)
from ...io.settings_artifact import ProcessingSettingsArtifactV5
from ...utils.path_utils import (
    calculate_frame_range,
    format_ffmpeg_time,
    generate_output_filename,
    parse_time_string,
)
from ...core.constants import (
    INTERMEDIATE_DIRS,
    DEFAULT_FALLBACK_FPS,
)
from ..frames.source_frame_manifest import (
    read_source_frame_manifest,
    source_frame_manifest_mismatch_reason,
    write_source_frame_manifest,
)
from ..frames.vr_assembler import DirectStereoSource
from ...core.file_identity import file_sample_fingerprint
from ...utils.imaging.png_header import PngHeader, read_png_header
from .encoding_provider import (
    FINAL_DECODE_DIAGNOSTIC_BYTES,
    EncodingProviderChangedError,
    EncodingProviderError,
    FINAL_ENCODING_LINE_BYTES,
    EncodingSequenceProvider,
    FfmpegProgressParser,
    FinalContainerExpectation,
    StagedEncodingInputManifest,
    audit_encoding_input_manifest,
    stage_encoding_input_manifest,
    validate_encoding_provider,
    validate_encoding_postpass,
)
from .final_publication import (
    ENCODING_INPUT_MANIFEST_RELATIVE_PATH,
    EncodingInputPublicationSource,
    FINAL_ENCODING_RESERVATION_RELATIVE_PATH,
    FinalEncodingReservationCapabilityV1,
    FinalPublicationTransactionV4,
    PreparedFinalEncodingReservationV1,
    validate_final_container_evidence_v1,
    prepare_final_encoding_reservation_v1,
    load_final_encoding_reservation_v1,
)
from .bounded_process import (
    BoundedProcessCaptureV1,
    BoundedProcessError,
    run_ffmpeg_progress_v1,
)
from .final_container_validation import validate_final_container_v4
from .final_audio import SelectedAudioIdentityV1, prepare_selected_audio_v1


_DIRECT_FRAME_NAME = re.compile(r"^frame_(\d{6,})\.png$")


@dataclass(frozen=True)
class _DirectStereoSequence:
    """Validated image2 manifests and the only headers needed for direct encoding."""

    left_pattern: Path
    right_pattern: Path
    start_number: int
    frame_count: int
    left_header: PngHeader
    right_header: PngHeader


@dataclass(frozen=True)
class _Schema5EncodingContext:
    provider: EncodingSequenceProvider
    staged_manifest: StagedEncodingInputManifest | None
    manifest_source: EncodingInputPublicationSource | None
    expected_relative_path: str
    sibling_path: Path
    reservation: FinalEncodingReservationCapabilityV1
    resolved_fps: str
    already_authenticated: bool = False


@dataclass(frozen=True, slots=True)
class _EncodingManifestExtentAuthority:
    """Scalar manifest location and bounds retained outside reservation phases."""

    job_root: Path | None
    job_root_identity: DirectoryIdentityV1 | None
    source_path: Path
    source_parent_identity: DirectoryIdentityV1 | None
    source_identity: PersistedFileIdentityV1 | None
    payload_raw_max: int | None
    logical_byte_count: int | None
    indexed_source: bool


@contextmanager
def _open_encoding_manifest_extent_authority(
    authority: _EncodingManifestExtentAuthority,
    *,
    writable: bool,
) -> Iterator[BinaryIO]:
    if (
        not authority.indexed_source
        or authority.job_root is None
        or authority.job_root_identity is None
        or authority.source_parent_identity is None
        or authority.source_identity is None
    ):
        raise ReservationConflictError(
            "indexed encoding manifest lacks scalar source authority"
        )
    try:
        with AcquiredRoot(authority.job_root) as acquired:
            if acquired.persistent_identity != authority.job_root_identity:
                raise ReservationConflictError(
                    "encoding manifest job-root identity changed"
                )
            if (
                acquired.directory_identity(authority.source_path.parent)
                != authority.source_parent_identity
            ):
                raise ReservationConflictError(
                    "encoding manifest source parent identity changed"
                )
            opener = acquired.open_regular_update if writable else acquired.open_regular
            with opener(authority.source_path) as handle:
                if (
                    file_identity_from_open_handle(handle)
                    != authority.source_identity
                ):
                    raise ReservationConflictError(
                        "encoding manifest source identity changed"
                    )
                yield handle
                if (
                    file_identity_from_open_handle(handle)
                    != authority.source_identity
                    or acquired.persistent_regular_identity(
                        authority.source_path
                    )
                    != authority.source_identity
                ):
                    raise ReservationConflictError(
                        "encoding manifest source identity changed during access"
                    )
            if (
                acquired.directory_identity(authority.source_path.parent)
                != authority.source_parent_identity
                or acquired.persistent_identity != authority.job_root_identity
            ):
                raise ReservationConflictError(
                    "encoding manifest source namespace changed during access"
                )
    except (NoFollowPathError, FileNotFoundError) as error:
        raise ReservationConflictError(
            "encoding manifest source cannot be opened without following links"
        ) from error


class VideoEncoder:
    """
    Handles video encoding using FFmpeg.

    Responsibilities:
    - Video creation with FFmpeg
    - Hardware encoder detection (NVENC)
    - Frame extraction from videos
    - Audio integration
    """

    def __init__(
        self,
        verbose: bool = False,
        *,
        defer_final_reservation_retirement: bool = False,
    ):
        """
        Initialize video encoder.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.defer_final_reservation_retirement = defer_final_reservation_retirement
        self._pending_final_reservation: FinalEncodingReservationCapabilityV1 | None = (
            None
        )

    @property
    def pending_final_encoding_reservation(
        self,
    ) -> FinalEncodingReservationCapabilityV1 | None:
        return self._pending_final_reservation

    def adopt_final_encoding_reservation(
        self,
        reservation: (
            PreparedFinalEncodingReservationV1 | FinalEncodingReservationCapabilityV1
        ),
    ) -> None:
        """Attach an authenticated recovery transaction for index-last retirement."""

        self._pending_final_reservation = (
            reservation
            if isinstance(reservation, FinalEncodingReservationCapabilityV1)
            else FinalEncodingReservationCapabilityV1.from_prepared(reservation)
        )

    def retire_final_encoding_reservation(self, *, success: bool) -> None:
        reservation = self._pending_final_reservation
        if reservation is None:
            return
        current = recover_processing_control_prefix_v1(reservation.job_root)
        expected_status = "completed" if success else "failed"
        if current is None or current.status != expected_status:
            raise ReservationConflictError(
                f"final reservation retirement requires durable {expected_status} "
                "terminal settings"
            )
        index_temporary = reservation.index_path.with_name(
            f".{reservation.index_path.name}.create-new.tmp"
        )
        if not any(
            path.exists() or path.is_symlink()
            for path in (reservation.index_path, index_temporary)
        ):
            self._pending_final_reservation = None
            return
        if success:
            reservation.retire()
        else:
            reservation.retire_terminal_failure()
        self._pending_final_reservation = None

    def create_video(  # noqa: C901
        self,
        vr_frames_dir: Path,
        output_dir: Path,
        original_video: str,
        settings: dict[str, Any],
        *,
        progress_tracker=None,
    ) -> bool:
        """
        Create final video with audio and encoding.

        Args:
            vr_frames_dir: Directory containing VR frames
            output_dir: Output directory
            original_video: Source video path for audio extraction
            settings: Processing settings

        Returns:
            True if successful, False otherwise

        Side effects:
            - Executes FFmpeg subprocess
            - Writes video file to disk
        """
        cancelled = self._cancellation_predicate(progress_tracker)
        try:
            recovered_settings = recover_processing_control_prefix_v1(output_dir)
        except InterruptedError:
            raise
        except ReservationIncompleteError:
            raise
        except Exception as error:
            print(f"Error recovering schema-5 video publication: {error}")
            return False
        schema5_present = recovered_settings is not None
        del recovered_settings
        active_index = output_dir / FINAL_ENCODING_RESERVATION_RELATIVE_PATH
        recover_before_dependency_gate = active_index.is_file()
        if not recover_before_dependency_gate and not verify_ffmpeg_installation():
            print("Error: FFmpeg not found. Cannot create output video.")
            return False
        resolved_encoder_name: str | None = None
        if schema5_present and not recover_before_dependency_gate:
            resolved_encoder_name = self._resolve_encoder_name(
                settings.get("video_encoder", "auto"),
                cancelled=cancelled,
            )

        try:
            schema5_context = self._prepare_schema5_assembled_context(
                vr_frames_dir=vr_frames_dir,
                output_dir=output_dir,
                settings=settings,
                cancelled=cancelled,
            )
        except InterruptedError:
            raise
        except ReservationIncompleteError:
            raise
        except Exception as error:
            print(f"Error preparing schema-5 video publication: {error}")
            return False
        if schema5_context is not None:
            self._pending_final_reservation = schema5_context.reservation
        if schema5_context is not None and schema5_context.already_authenticated:
            return True
        if recover_before_dependency_gate and not verify_ffmpeg_installation():
            print("Error: FFmpeg not found. Cannot create output video.")
            return False
        if schema5_context is not None and resolved_encoder_name is None:
            resolved_encoder_name = self._resolve_encoder_name(
                settings.get("video_encoder", "auto"),
                cancelled=cancelled,
            )
        if schema5_context is None:
            output_filename = generate_output_filename(
                Path(original_video).name,
                settings["vr_format"],
                settings["vr_resolution"],
            )
            output_path = output_dir / output_filename
        else:
            output_path = schema5_context.sibling_path
        # Build base FFmpeg command
        base_fps = (
            schema5_context.resolved_fps
            if schema5_context is not None
            else self._resolve_output_fps(original_video, settings)
        )

        cmd = ["ffmpeg", "-y"]
        if schema5_context is not None:
            cmd.extend(["-v", "error", "-xerror", "-nostats", "-progress", "pipe:1"])
        cmd.extend(["-framerate", str(base_fps)])
        if schema5_context is not None:
            cmd.extend(["-start_number", str(schema5_context.provider.first_index)])
        if schema5_context is None:
            assembled_pattern = vr_frames_dir / "frame_%06d.png"
        else:
            frames_parent = schema5_context.provider.frames_parent
            if frames_parent is None:
                raise ValueError("assembled schema-5 provider has no frames parent")
            assembled_pattern = (
                Path(os.path.abspath(os.fspath(output_dir))).joinpath(
                    *frames_parent.parts
                )
                / "frame_%06d.png"
            )
        cmd.extend(["-i", str(assembled_pattern)])

        selected_audio: SelectedAudioIdentityV1 | None = None
        # Legacy jobs retain their historical optional-map behavior. Schema 5
        # freezes an exact audio identity before constructing the command.
        audio_source: str | Path | None = None
        if schema5_context is not None:
            audio_kwargs: dict[str, object] = {
                "output_dir": output_dir,
                "original_video": original_video,
                "settings": settings,
                "frame_count": schema5_context.provider.frame_count,
                "fps": str(base_fps),
            }
            if cancelled is not None:
                audio_kwargs["cancelled"] = cancelled
            selected_audio = prepare_selected_audio_v1(**audio_kwargs)
            cmd.extend(selected_audio.input_arguments())
            cmd.extend(["-vf", "setsar=1", "-map", "0:v:0"])
            cmd.extend(selected_audio.output_arguments(1))
            cmd.extend(["-frames:v", str(schema5_context.provider.frame_count)])
        elif settings.get("preserve_audio", True):
            audio_file = output_dir / "original_audio.flac"
            print(f"Looking for pre-extracted audio at: {audio_file}")
            if audio_file.exists():
                print(f"Using pre-extracted audio: {audio_file}")
                audio_source = audio_file
            else:
                print(f"Warning: Pre-extracted audio not found at {audio_file}")
                print(f"Extracting audio from original video: {original_video}")
                audio_source = original_video
            assert audio_source is not None
            cmd.extend(self._build_audio_input_args(audio_source, settings))
            cmd.extend(["-c:a", "aac", "-shortest"])

        # Add video encoding settings
        encoder = settings.get("video_encoder", "auto")
        encoder_args, _ = self._build_encoder_cmd(
            encoder,
            output_path,
            resolved_encoder=resolved_encoder_name,
            cancelled=cancelled,
        )
        resolved_encoder = encoder_args[encoder_args.index("-c:v") + 1]
        cmd.extend(encoder_args)

        try:
            if schema5_context is not None:
                if output_path.exists() or output_path.is_symlink():
                    raise ReservationConflictError(
                        "indexed final-video sibling was not retired before launch"
                    )
                self._commit_schema5_launch_boundary(
                    schema5_context,
                    job_root=output_dir,
                    settings=settings,
                    cancelled=cancelled,
                )
                returncode, diagnostics = self._run_schema5_ffmpeg_with_progress(
                    cmd,
                    schema5_context.provider.frame_count,
                    progress_tracker,
                )
                if returncode != 0:
                    self._print_direct_ffmpeg_failure(returncode, diagnostics)
                    return False
                del diagnostics
            else:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"FFmpeg error: {result.stderr}")
                    return False
            if schema5_context is not None:
                assert schema5_context.staged_manifest is not None
                assert schema5_context.manifest_source is not None
                validate_encoding_postpass(
                    schema5_context.provider,
                    job_root=output_dir,
                    staged=schema5_context.staged_manifest,
                    per_eye_size=self._per_eye_size(settings),
                    vr_format=settings["vr_format"],
                    cancelled=cancelled,
                )
                assert selected_audio is not None
                selected_audio.revalidate(cancelled=cancelled)
                normalized = self._normalize_publication_arguments(
                    command=cmd,
                    image_inputs={
                        str(assembled_pattern): "frames",
                    },
                    selected_audio=selected_audio,
                    runtime_output=output_path,
                    input_stage_fingerprint=(
                        schema5_context.staged_manifest.fingerprint
                    ),
                    final_relative_path=schema5_context.expected_relative_path,
                )
                container_validator = self._container_validator_for_invocation(
                    settings,
                    schema5_context.provider.frame_count,
                    str(base_fps),
                    resolved_encoder,
                    selected_audio.audio_end_sample,
                    cancelled,
                )
                validated_container = validate_final_container_evidence_v1(
                    output_dir,
                    output_path,
                    container_validator,
                )
                publication_settings = recover_processing_control_prefix_v1(output_dir)
                if publication_settings is None:
                    raise ReservationConflictError(
                        "final publication lost its settings authority"
                    )
                publication = FinalPublicationTransactionV4(
                    job_root=output_dir,
                    settings=publication_settings,
                    sibling_relative_path=output_path.name,
                    encoding_input_source=schema5_context.manifest_source,
                    input_stage_fingerprint=(
                        schema5_context.staged_manifest.fingerprint
                    ),
                    resolved_encoding_arguments=normalized,
                    container_validator=None,
                    expected_sibling_identity=validated_container.identity,
                    expected_sibling_sha256=validated_container.raw_sha256,
                    expected_sibling_byte_count=validated_container.byte_count,
                    reservation=schema5_context.reservation,
                    retire_reservation=False,
                )
                del validated_container
                del publication_settings
                publication.publish()
                self._pending_final_reservation = schema5_context.reservation
            return True
        except InterruptedError:
            raise
        except ReservationIncompleteError:
            raise
        except Exception as e:
            print(f"Error creating output video: {e}")
            return False
        finally:
            if schema5_context is not None:
                self._discard_schema5_sibling_after_attempt(schema5_context)

    @staticmethod
    def _per_eye_size(settings: dict[str, Any]) -> tuple[int, int]:
        try:
            width = int(settings["per_eye_width"])
            height = int(settings["per_eye_height"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("schema-5 encoding requires per-eye dimensions") from error
        if width <= 0 or height <= 0:
            raise ValueError("schema-5 per-eye dimensions must be positive")
        return width, height

    @staticmethod
    def _discard_schema5_sibling_after_attempt(
        context: _Schema5EncodingContext,
    ) -> None:
        try:
            context.reservation.discard_nonsemantic_sibling()
        except Exception as error:
            print(f"Warning: Preserving indexed sibling for recovery: {error}")

    def _prepare_schema5_assembled_context(
        self,
        *,
        vr_frames_dir: Path,
        output_dir: Path,
        settings: dict[str, Any],
        cancelled: Callable[[], bool] | None = None,
    ) -> _Schema5EncodingContext | None:
        artifact = recover_processing_control_prefix_v1(output_dir)
        if artifact is None:
            return None
        del artifact
        root = Path(os.path.abspath(os.fspath(output_dir)))
        lexical_parent = Path(os.path.abspath(os.fspath(vr_frames_dir)))
        try:
            parent = lexical_parent.relative_to(root)
        except ValueError as error:
            raise ValueError(
                "assembled frame parent is outside the job root"
            ) from error
        frame_count = 0
        first_index: int | None = None
        last_index: int | None = None
        try:
            with AcquiredRoot(root) as acquired:
                for entry in acquired.iter_directory(lexical_parent):
                    if cancelled is not None and cancelled():
                        raise InterruptedError(
                            "final encoding input audit was cancelled"
                        )
                    match = _DIRECT_FRAME_NAME.fullmatch(entry.name)
                    if match is None:
                        folded = entry.name.casefold()
                        if folded.startswith("frame_") and folded.endswith(".png"):
                            raise ValueError("assembled frame name is noncanonical")
                        continue
                    if not entry.is_regular:
                        raise ValueError("assembled frame is not an ordinary file")
                    index = int(match.group(1))
                    if entry.name != f"frame_{index:06d}.png":
                        raise ValueError("assembled frame padding is noncanonical")
                    frame_count += 1
                    first_index = (
                        index if first_index is None else min(first_index, index)
                    )
                    last_index = index if last_index is None else max(last_index, index)
        except NoFollowPathError as error:
            raise ValueError(str(error)) from error
        if frame_count == 0 or first_index is None or last_index is None:
            raise ValueError("assembled schema-5 encoding has no frames")
        if frame_count != last_index - first_index + 1:
            raise ValueError("assembled schema-5 frame sequence is not contiguous")
        provider = EncodingSequenceProvider.freeze(
            mode="assembled_vr",
            frame_count=frame_count,
            first_index=first_index,
            frames_parent=PurePosixPath(parent.as_posix()),
        )
        return self._prepare_schema5_publication_context(
            provider=provider,
            output_dir=output_dir,
            settings=settings,
            cancelled=cancelled,
        )

    def _prepare_schema5_direct_context(
        self,
        *,
        sequence: _DirectStereoSequence,
        output_dir: Path,
        settings: dict[str, Any],
        cancelled: Callable[[], bool] | None = None,
    ) -> _Schema5EncodingContext | None:
        root = Path(os.path.abspath(os.fspath(output_dir)))
        left_parent = Path(
            os.path.abspath(os.fspath(sequence.left_pattern.parent))
        ).relative_to(root)
        right_parent = Path(
            os.path.abspath(os.fspath(sequence.right_pattern.parent))
        ).relative_to(root)
        provider = EncodingSequenceProvider.freeze(
            mode="direct_stereo",
            frame_count=sequence.frame_count,
            first_index=sequence.start_number,
            left_parent=PurePosixPath(left_parent.as_posix()),
            right_parent=PurePosixPath(right_parent.as_posix()),
        )
        return self._prepare_schema5_direct_provider_context(
            provider=provider,
            output_dir=output_dir,
            settings=settings,
            cancelled=cancelled,
        )

    def _prepare_schema5_direct_provider_context(
        self,
        *,
        provider: EncodingSequenceProvider,
        output_dir: Path,
        settings: dict[str, Any],
        cancelled: Callable[[], bool] | None = None,
    ) -> _Schema5EncodingContext | None:
        artifact = recover_processing_control_prefix_v1(output_dir)
        if artifact is None:
            return None
        del artifact
        if provider.mode != "direct_stereo":
            raise ValueError("direct schema-5 encoding requires a direct provider")
        return self._prepare_schema5_publication_context(
            provider=provider,
            output_dir=output_dir,
            settings=settings,
            cancelled=cancelled,
        )

    def _prepare_schema5_publication_context(
        self,
        *,
        provider: EncodingSequenceProvider,
        output_dir: Path,
        settings: dict[str, Any],
        cancelled: Callable[[], bool] | None = None,
        _allow_prelaunch_rebuild: bool = True,
    ) -> _Schema5EncodingContext:
        artifact = recover_processing_control_prefix_v1(output_dir)
        if artifact is None:
            raise ReservationConflictError(
                "schema-5 publication lost its settings authority"
            )
        artifact_value = artifact.to_object()
        frame_count = artifact.selected_frame_count
        if provider.frame_count != frame_count:
            raise ValueError(
                "encoding provider frame count differs from durable settings"
            )
        metadata = artifact_value["metadata"]
        publication_generation = metadata["final_media_publication_generation"]
        expected_relative_path = artifact_value["output_info"][
            "expected_final_relative_path"
        ]
        resolved_fps = self._resolve_schema5_output_fps_value(artifact_value)
        del metadata, artifact_value, artifact
        active_index = output_dir / FINAL_ENCODING_RESERVATION_RELATIVE_PATH
        allow_prelaunch_rebuild = _allow_prelaunch_rebuild
        while True:
            if active_index.is_file():
                phase_settings = recover_processing_control_prefix_v1(output_dir)
                if phase_settings is None:
                    raise ReservationConflictError(
                        "active final reservation lost its settings authority"
                    )
                reservation = load_final_encoding_reservation_v1(
                    output_dir, phase_settings
                )
                del phase_settings
                capability = FinalEncodingReservationCapabilityV1.from_prepared(
                    reservation
                )
                self._pending_final_reservation = capability
                recovery = reservation.recover_publication_prefix()
                recovery_phase = recovery.phase
                del recovery
                if recovery_phase == "publication_authenticated":
                    assert publication_generation is not None
                    del reservation
                    return _Schema5EncodingContext(
                        provider=provider,
                        staged_manifest=None,
                        manifest_source=None,
                        expected_relative_path=expected_relative_path,
                        sibling_path=capability.sibling_path,
                        reservation=capability,
                        resolved_fps=resolved_fps,
                        already_authenticated=True,
                    )
                if recovery_phase not in {
                    "ready_pre_video",
                    "input_manifest_committed_no_final_intent",
                    "post_boundary_retry_sibling_present",
                }:
                    raise ValueError(
                        f"unsupported final encoding recovery phase: {recovery_phase}"
                    )
                if (
                    reservation.sibling_path.exists()
                    or reservation.sibling_path.is_symlink()
                ):
                    if not reservation.discard_nonsemantic_sibling():
                        raise ValueError(
                            "indexed sibling is not discardable in this phase"
                        )
                reservation.reset_incomplete_manifest_sources()
                authority = self._encoding_manifest_extent_authority(reservation)
                publication_generation = reservation.index.publication_generation
                del reservation
                manifest_source = self._recover_encoding_manifest_source(
                    authority,
                    expected_frame_count=frame_count,
                )
                if manifest_source is None:
                    staged, manifest_source = self._stage_encoding_manifest_file(
                        provider=provider,
                        output_dir=output_dir,
                        settings=settings,
                        reservation=capability,
                        authority=authority,
                        cancelled=cancelled,
                    )
                else:
                    try:
                        staged = audit_encoding_input_manifest(
                            provider,
                            job_root=output_dir,
                            expected_fingerprint=manifest_source.fingerprint,
                            expected_raw_sha256=manifest_source.raw_sha256,
                            expected_raw_byte_count=manifest_source.raw_byte_count,
                            per_eye_size=self._per_eye_size(settings),
                            vr_format=settings["vr_format"],
                            cancelled=cancelled,
                        )
                    except (EncodingProviderError, OSError, ValueError):
                        if (
                            recovery_phase != "ready_pre_video"
                            or not allow_prelaunch_rebuild
                        ):
                            raise
                        capability.discard_prelaunch_invocation()
                        allow_prelaunch_rebuild = False
                        del authority, capability, manifest_source
                        continue
            else:
                if publication_generation is None:
                    publication_generation = secrets.token_hex(16)
                phase_settings = recover_processing_control_prefix_v1(output_dir)
                if phase_settings is None:
                    raise ReservationConflictError(
                        "final reservation preflight lost its settings authority"
                    )
                try:
                    reservation = prepare_final_encoding_reservation_v1(
                        job_root=output_dir,
                        settings=phase_settings,
                        publication_generation=publication_generation,
                        input_manifest_raw_max=self._encoding_manifest_raw_bound(
                            provider
                        ),
                        final_manifest_raw_max=1024 * 1024,
                        cancelled=cancelled,
                    )
                except ReservationIncompleteError:
                    raise
                except BaseException:
                    if active_index.is_file():
                        recovery_settings = recover_processing_control_prefix_v1(
                            output_dir
                        )
                        if recovery_settings is not None:
                            recovered = load_final_encoding_reservation_v1(
                                output_dir, recovery_settings
                            )
                            self._pending_final_reservation = (
                                FinalEncodingReservationCapabilityV1.from_prepared(
                                    recovered
                                )
                            )
                            del recovered, recovery_settings
                    raise
                finally:
                    del phase_settings
                capability = FinalEncodingReservationCapabilityV1.from_prepared(
                    reservation
                )
                self._pending_final_reservation = capability
                authority = self._encoding_manifest_extent_authority(reservation)
                del reservation
                staged, manifest_source = self._stage_encoding_manifest_file(
                    provider=provider,
                    output_dir=output_dir,
                    settings=settings,
                    reservation=capability,
                    authority=authority,
                    cancelled=cancelled,
                )
            return _Schema5EncodingContext(
                provider=provider,
                staged_manifest=staged,
                manifest_source=manifest_source,
                expected_relative_path=expected_relative_path,
                sibling_path=capability.sibling_path,
                reservation=capability,
                resolved_fps=resolved_fps,
            )

    @staticmethod
    def _encoding_manifest_extent_authority(
        reservation: PreparedFinalEncodingReservationV1,
    ) -> _EncodingManifestExtentAuthority:
        transaction = reservation.input_manifest_transaction
        if transaction is None:
            return _EncodingManifestExtentAuthority(
                job_root=None,
                job_root_identity=None,
                source_path=(
                    reservation.job_root / ENCODING_INPUT_MANIFEST_RELATIVE_PATH
                ),
                source_parent_identity=None,
                source_identity=None,
                payload_raw_max=None,
                logical_byte_count=None,
                indexed_source=False,
            )
        descriptor = transaction.descriptor
        root = Path(os.path.abspath(os.fspath(reservation.job_root)))
        source_path = transaction.source_path
        try:
            with AcquiredRoot(root) as acquired:
                root_identity = cast(
                    DirectoryIdentityV1,
                    acquired.persistent_identity,
                )
                source_parent_identity = cast(
                    DirectoryIdentityV1,
                    acquired.directory_identity(source_path.parent),
                )
                with acquired.open_regular(source_path) as handle:
                    source_identity = file_identity_from_open_handle(handle)
        except (NoFollowPathError, FileNotFoundError) as error:
            raise ReservationConflictError(
                "encoding manifest source cannot be authorized without following links"
            ) from error
        if (
            root_identity != reservation.index.sibling_video_parent_identity
            or source_parent_identity != descriptor.target_parent_identity
            or source_identity != descriptor.filesystem_identity
        ):
            raise ReservationConflictError(
                "encoding manifest scalar source authority differs from its index"
            )
        return _EncodingManifestExtentAuthority(
            job_root=root,
            job_root_identity=root_identity,
            source_path=source_path,
            source_parent_identity=source_parent_identity,
            source_identity=source_identity,
            payload_raw_max=descriptor.payload_raw_max,
            logical_byte_count=descriptor.logical_byte_count,
            indexed_source=True,
        )

    @staticmethod
    def _recover_encoding_manifest_source(
        authority: _EncodingManifestExtentAuthority,
        *,
        expected_frame_count: int,
    ) -> EncodingInputPublicationSource | None:
        if not authority.indexed_source:
            return EncodingInputPublicationSource.from_path(
                authority.source_path,
                expected_frame_count=expected_frame_count,
            )
        with _open_encoding_manifest_extent_authority(
            authority,
            writable=False,
        ) as handle:
            prefix = (
                PreparedFinalEncodingReservationV1._canonical_prefix_byte_count_handle(
                    handle
                )
            )
            if prefix is None:
                raise ReservationConflictError(
                    "indexed encoding manifest has a noncanonical payload prefix"
                )
            if prefix == 0:
                return None
            logical_byte_count = (
                authority.logical_byte_count
                if os.fstat(handle.fileno()).st_size == authority.logical_byte_count
                else None
            )
            assert authority.job_root is not None
            assert authority.job_root_identity is not None
            assert authority.source_parent_identity is not None
            assert authority.source_identity is not None
            return EncodingInputPublicationSource.from_authenticated_handle(
                path=authority.source_path,
                handle=handle,
                job_root=authority.job_root,
                job_root_identity=authority.job_root_identity,
                source_parent_identity=authority.source_parent_identity,
                source_identity=authority.source_identity,
                raw_byte_count=prefix,
                logical_byte_count=logical_byte_count,
                expected_frame_count=expected_frame_count,
            )

    @staticmethod
    def _commit_schema5_launch_boundary(
        context: _Schema5EncodingContext,
        *,
        job_root: Path,
        settings: dict[str, Any],
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        staged = context.staged_manifest
        if staged is None:
            raise ValueError("schema-5 launch has no staged encoding manifest")
        context.reservation.commit_producer_settings()
        try:
            validate_encoding_provider(
                context.provider,
                staged.provider_audit,
                launch_started=False,
                cancelled=cancelled,
            )
            audit_encoding_input_manifest(
                context.provider,
                job_root=job_root,
                expected_fingerprint=staged.fingerprint,
                expected_raw_sha256=staged.raw_sha256,
                expected_raw_byte_count=staged.raw_byte_count,
                per_eye_size=VideoEncoder._per_eye_size(settings),
                vr_format=settings["vr_format"],
                cancelled=cancelled,
            )
        except InterruptedError:
            raise
        except (EncodingProviderError, OSError, ValueError) as error:
            context.reservation.discard_prelaunch_invocation()
            if isinstance(error, EncodingProviderChangedError):
                raise
            raise EncodingProviderChangedError(
                "encoding image content changed at the launch boundary"
            ) from error

    @staticmethod
    def _encoding_manifest_raw_bound(provider: EncodingSequenceProvider) -> int:
        parents = tuple(
            parent
            for parent in (
                provider.frames_parent,
                provider.left_parent,
                provider.right_parent,
            )
            if parent is not None
        )
        view_count = 1 if provider.mode == "assembled_vr" else 2
        maximum_parent_scalars = max(len(parent.as_posix()) for parent in parents)
        maximum_index_digits = len(str(provider.last_index))
        maximum_path_scalars = maximum_parent_scalars + maximum_index_digits + 18
        per_frame_bound = 96 + view_count * (6 * maximum_path_scalars + 512)
        raw_bound = 4096 + provider.frame_count * per_frame_bound
        if raw_bound > (1 << 64) - 1:
            raise ValueError("encoding input manifest raw bound exceeds u64")
        return raw_bound

    def _stage_encoding_manifest_file(
        self,
        *,
        provider: EncodingSequenceProvider,
        output_dir: Path,
        settings: dict[str, Any],
        reservation: FinalEncodingReservationCapabilityV1,
        authority: _EncodingManifestExtentAuthority,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[StagedEncodingInputManifest, EncodingInputPublicationSource]:
        if (
            not authority.indexed_source
            or authority.payload_raw_max is None
            or authority.logical_byte_count is None
        ):
            raise ValueError("encoding input reservation is already consumed")
        path = authority.source_path
        with _open_encoding_manifest_extent_authority(
            authority,
            writable=True,
        ) as sink:
            sink.seek(0)
            staged = stage_encoding_input_manifest(
                provider,
                job_root=output_dir,
                sink=sink,
                per_eye_size=self._per_eye_size(settings),
                vr_format=settings["vr_format"],
                cancelled=cancelled,
            )
            if staged.raw_byte_count > authority.payload_raw_max:
                raise ValueError("encoding manifest exceeded its reserved raw bound")
            sink.flush()
            os.fsync(sink.fileno())
        assert authority.job_root is not None
        assert authority.job_root_identity is not None
        assert authority.source_parent_identity is not None
        assert authority.source_identity is not None
        source = EncodingInputPublicationSource(
            path=path,
            fingerprint=staged.fingerprint,
            raw_sha256=staged.raw_sha256,
            raw_byte_count=staged.raw_byte_count,
            logical_byte_count=authority.logical_byte_count,
            job_root=authority.job_root,
            job_root_identity=authority.job_root_identity,
            source_parent_identity=authority.source_parent_identity,
            source_identity=authority.source_identity,
        ).validated(expected_frame_count=provider.frame_count)
        materialized = reservation.materialize()
        try:
            transaction = materialized.input_manifest_transaction
            if transaction is None:
                raise ReservationConflictError(
                    "encoding input reservation was consumed during staging"
                )
            descriptor = transaction.descriptor
            if (
                transaction.source_path != authority.source_path
                or descriptor.payload_raw_max != authority.payload_raw_max
                or descriptor.logical_byte_count != authority.logical_byte_count
            ):
                raise ReservationConflictError(
                    "encoding input extent authority changed during staging"
                )
            transaction.validate_streamed_physical_phase(
                prefix_byte_count=staged.raw_byte_count,
                prefix_sha256=staged.raw_sha256,
            )
        finally:
            del materialized
        return staged, source

    @staticmethod
    def _selected_audio_source(
        output_dir: Path,
        original_video: str,
        settings: dict[str, Any],
    ) -> str | Path | None:
        if not settings.get("preserve_audio", True):
            return None
        audio_file = output_dir / "original_audio.flac"
        return audio_file if audio_file.exists() else original_video

    @staticmethod
    def _hash_input(path: Path) -> tuple[str, int]:
        digest = hashlib.sha256()
        count = 0
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                count += len(chunk)
        return digest.hexdigest(), count

    @classmethod
    def _normalize_publication_arguments(
        cls,
        *,
        command: list[str],
        image_inputs: dict[str, str],
        selected_audio: SelectedAudioIdentityV1,
        runtime_output: Path,
        input_stage_fingerprint: str,
        final_relative_path: str,
    ) -> tuple[str, ...]:
        normalized: list[str] = []
        audio_token = selected_audio.normalized_input_token()
        audio_text = (
            str(selected_audio.selected_source_path) if selected_audio.usable else None
        )
        for index, argument in enumerate(command):
            text = str(argument)
            if index > 0 and command[index - 1] == "-i":
                view = image_inputs.get(text)
                if view is not None:
                    text = (
                        f"@image-sequence:sha256={input_stage_fingerprint}:view={view}"
                    )
                elif audio_text is not None and text == audio_text:
                    assert audio_token is not None
                    text = audio_token
                else:
                    raise ValueError("runtime argv contains an unbound input path")
            if index == len(command) - 1:
                if text != str(runtime_output):
                    raise ValueError("runtime argv final output path changed")
                text = "@output:path64=" + FinalPublicationTransactionV4.path64(
                    final_relative_path
                )
            normalized.append(text)
        return tuple(normalized)

    @staticmethod
    def _validate_final_container(
        path: Path,
        settings: dict[str, Any],
        frame_count: int,
        fps: str,
        resolved_encoder: str,
        audio_end_sample: int | None,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> bool:
        width = int(settings["vr_output_width"])
        height = int(settings["vr_output_height"])
        rate = Fraction(fps)
        canonical_fps = f"{rate.numerator}/{rate.denominator}"
        if audio_end_sample is not None:
            packet_bound = (audio_end_sample + 1023) // 1024 + 2
        else:
            packet_bound = 0
        expectation = FinalContainerExpectation(
            width=width,
            height=height,
            fps=canonical_fps,
            frame_count=frame_count,
            resolved_encoder=cast(
                Literal["libx264", "libx265", "hevc_nvenc"], resolved_encoder
            ),
            audio_expected=audio_end_sample is not None,
            audio_packet_upper_bound=packet_bound,
        )
        if cancelled is None:
            return validate_final_container_v4(path, expectation)
        return validate_final_container_v4(path, expectation, cancelled=cancelled)

    @staticmethod
    def _cancellation_predicate(
        progress_tracker,
    ) -> Callable[[], bool] | None:
        """Adapt an explicit tracker cancellation method without probing dynamic mocks."""

        if progress_tracker is None:
            return None
        checker = getattr(type(progress_tracker), "is_cancelled", None)
        if callable(checker):
            return lambda: bool(progress_tracker.is_cancelled())
        try:
            checker = vars(progress_tracker).get("is_cancelled")
        except TypeError:
            checker = None
        if callable(checker):
            return lambda: bool(checker())
        return None

    def _validate_final_container_for_invocation(
        self,
        path: Path,
        settings: dict[str, Any],
        frame_count: int,
        fps: str,
        resolved_encoder: str,
        audio_end_sample: int | None,
        cancelled: Callable[[], bool] | None,
    ) -> bool:
        arguments = (
            path,
            settings,
            frame_count,
            fps,
            resolved_encoder,
            audio_end_sample,
        )
        if cancelled is None:
            return self._validate_final_container(*arguments)
        return self._validate_final_container(*arguments, cancelled=cancelled)

    def _container_validator_for_invocation(
        self,
        settings: dict[str, Any],
        frame_count: int,
        fps: str,
        resolved_encoder: str,
        audio_end_sample: int | None,
        cancelled: Callable[[], bool] | None,
    ) -> Callable[[Path], bool]:
        scalar_settings = {
            "vr_output_width": int(settings["vr_output_width"]),
            "vr_output_height": int(settings["vr_output_height"]),
        }

        def validate(path: Path) -> bool:
            return self._validate_final_container_for_invocation(
                path,
                scalar_settings,
                frame_count,
                fps,
                resolved_encoder,
                audio_end_sample,
                cancelled,
            )

        return validate

    @staticmethod
    def _build_audio_input_args(
        audio_source: str | Path,
        settings: dict[str, Any],
    ) -> list[str]:
        """Build an audio input whose local timeline matches the selected video clip."""

        start_value = settings.get("start_time")
        end_value = settings.get("end_time")
        start_seconds = parse_time_string(start_value) if start_value else None
        end_seconds = parse_time_string(end_value) if end_value else None
        clip_start = max(start_seconds or 0.0, 0.0)

        args: list[str] = []
        if clip_start > 0:
            args.extend(["-ss", format_ffmpeg_time(clip_start)])
        if end_seconds is not None and end_seconds > clip_start:
            args.extend(["-t", format_ffmpeg_time(end_seconds - clip_start)])
        args.extend(["-i", str(audio_source)])
        return args

    @staticmethod
    def _normalize_frame_rate(value: Any) -> str | None:
        """Return a positive FFmpeg frame-rate value, preserving exact fractions."""
        if value is None:
            return None

        text = str(value).strip()
        if not text or text.lower() in {"none", "original"}:
            return None

        try:
            if Fraction(text) <= 0:
                return None
        except (ValueError, ZeroDivisionError):
            return None

        return text

    @staticmethod
    def _resolve_schema5_output_fps(artifact: ProcessingSettingsArtifactV5) -> str:
        """Resolve one canonical rational exclusively from durable schema-5 scalars."""

        return VideoEncoder._resolve_schema5_output_fps_value(artifact.to_object())

    @staticmethod
    def _resolve_schema5_output_fps_value(values: dict[str, Any]) -> str:
        """Resolve the schema-5 rate from one already-validated settings value."""

        settings = values["processing_settings"]
        target = settings["target_fps"]
        if type(target) is int:
            rate = Fraction(target, 1)
        else:
            source = settings["source_fps"]
            if source is None or source <= 0:
                source = values["video_properties"]["fps"]
            rate = Fraction(str(source))
        if rate <= 0:
            raise ValueError("schema-5 output frame rate must be positive")
        return f"{rate.numerator}/{rate.denominator}"

    def _resolve_output_fps(self, original_video: str, settings: dict[str, Any]) -> str:
        """Resolve explicit FPS or preserve the source video's exact stream rate."""
        target_fps = self._normalize_frame_rate(settings.get("target_fps"))
        if target_fps is not None:
            return target_fps

        video_info = get_video_info_ffprobe(original_video)
        for stream in video_info.get("streams", []):
            if stream.get("codec_type") != "video":
                continue
            for rate_key in ("avg_frame_rate", "r_frame_rate"):
                source_rate = self._normalize_frame_rate(stream.get(rate_key))
                if source_rate is not None:
                    return source_rate

        source_fps = self._normalize_frame_rate(settings.get("source_fps"))
        if source_fps is not None:
            return source_fps

        return str(DEFAULT_FALLBACK_FPS)

    @staticmethod
    def _direct_frame_index(path: Path) -> int:
        """Return a canonical image2 frame index or reject a non-image2 filename."""

        match = _DIRECT_FRAME_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Noncanonical direct frame name: {path.name}")
        index = int(match.group(1))
        if path.name != f"frame_{index:06d}.png":
            raise ValueError(f"Noncanonical direct frame padding: {path.name}")
        return index

    def _validate_direct_stereo_sequence(
        self,
        left_files: Sequence[Path],
        right_files: Sequence[Path],
        total_frames: int,
    ) -> _DirectStereoSequence:
        """Validate corresponding image2 eye sequences before FFmpeg can start."""

        if not left_files or not right_files:
            raise ValueError(
                "Direct stereo sequences must contain frames for both eyes"
            )
        if len(left_files) != len(right_files):
            raise ValueError("Direct stereo sequences must have identical frame counts")
        frame_count = len(left_files)
        if total_frames > 0 and frame_count != total_frames:
            raise ValueError("Direct stereo sequence count does not match total frames")
        left_parent = Path(left_files[0]).parent
        right_parent = Path(right_files[0]).parent
        minimum: int | None = None
        maximum: int | None = None
        for position in range(frame_count):
            left = Path(left_files[position])
            right = Path(right_files[position])
            if left.parent != left_parent:
                raise ValueError(
                    "Direct left-eye frames must have one parent directory"
                )
            if right.parent != right_parent:
                raise ValueError(
                    "Direct right-eye frames must have one parent directory"
                )
            left_index = self._direct_frame_index(left)
            right_index = self._direct_frame_index(right)
            if left_index != right_index:
                raise ValueError(
                    "Direct stereo sequences must have identical frame names"
                )
            minimum = left_index if minimum is None else min(minimum, left_index)
            maximum = left_index if maximum is None else max(maximum, left_index)
        assert minimum is not None and maximum is not None
        if maximum - minimum + 1 != frame_count:
            raise ValueError("Direct stereo sequence frame indices must be gap-free")
        seen = bytearray((frame_count + 7) // 8)
        for path in left_files:
            relative_index = self._direct_frame_index(Path(path)) - minimum
            byte_index, bit_index = divmod(relative_index, 8)
            mask = 1 << bit_index
            if seen[byte_index] & mask:
                raise ValueError("Direct stereo sequence frame indices must be unique")
            seen[byte_index] |= mask

        first_left = left_parent / f"frame_{minimum:06d}.png"
        first_right = right_parent / f"frame_{minimum:06d}.png"
        left_header = read_png_header(first_left)
        right_header = read_png_header(first_right)
        if left_header is None or right_header is None:
            raise ValueError("Direct stereo sequence has an unreadable first PNG")

        return _DirectStereoSequence(
            left_pattern=left_parent / "frame_%06d.png",
            right_pattern=right_parent / "frame_%06d.png",
            start_number=minimum,
            frame_count=frame_count,
            left_header=left_header,
            right_header=right_header,
        )

    def _build_direct_stereo_command(  # noqa: C901
        self,
        sequence: _DirectStereoSequence,
        temporary_output: Path,
        original_video: str,
        settings: dict[str, Any],
        *,
        selected_audio: SelectedAudioIdentityV1 | None = None,
        resolved_fps: str | None = None,
    ) -> list[str]:
        """Build an FFmpeg command that stacks validated left and right image2 inputs."""

        try:
            per_eye_width = int(settings["per_eye_width"])
            per_eye_height = int(settings["per_eye_height"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Direct stereo encoding requires per-eye dimensions"
            ) from error
        if per_eye_width <= 0 or per_eye_height <= 0:
            raise ValueError("Direct stereo per-eye dimensions must be positive")

        vr_format = settings.get("vr_format")
        if not isinstance(vr_format, str):
            raise ValueError(f"Unsupported direct VR format: {vr_format}")
        stack_filter = {
            "side_by_side": "hstack",
            "over_under": "vstack",
        }.get(vr_format)
        if stack_filter is None:
            raise ValueError(f"Unsupported direct VR format: {vr_format}")

        fps = resolved_fps or self._resolve_output_fps(original_video, settings)
        start_number = str(sequence.start_number)
        command = ["ffmpeg", "-y"]
        if selected_audio is not None:
            command.extend(["-v", "error", "-xerror", "-nostats"])
        else:
            command.extend(["-loglevel", "error", "-nostats"])
        command.extend(
            [
                "-framerate",
                fps,
                "-start_number",
                start_number,
                "-i",
                str(sequence.left_pattern),
                "-framerate",
                fps,
                "-start_number",
                start_number,
                "-i",
                str(sequence.right_pattern),
            ]
        )

        preserve_audio = settings.get("preserve_audio", True)
        if selected_audio is not None:
            command.extend(selected_audio.input_arguments())
        elif preserve_audio:
            audio_file = temporary_output.parent / "original_audio.flac"
            audio_source: str | Path = (
                audio_file if audio_file.exists() else original_video
            )
            command.extend(self._build_audio_input_args(audio_source, settings))

        left_matches_target = (
            sequence.left_header.width == per_eye_width
            and sequence.left_header.height == per_eye_height
        )
        right_matches_target = (
            sequence.right_header.width == per_eye_width
            and sequence.right_header.height == per_eye_height
        )
        if left_matches_target and right_matches_target:
            filter_graph = (
                "[0:v]setsar=1[left];[1:v]setsar=1[right];"
                f"[left][right]{stack_filter}=inputs=2:shortest=1[stacked];"
                "[stacked]setsar=1[vr]"
            )
        else:
            filter_graph = (
                f"[0:v]scale={per_eye_width}:{per_eye_height}:"
                "flags=bicubic+accurate_rnd,setsar=1[left];"
                f"[1:v]scale={per_eye_width}:{per_eye_height}:"
                "flags=bicubic+accurate_rnd,setsar=1[right];"
                f"[left][right]{stack_filter}=inputs=2:shortest=1[stacked];"
                "[stacked]setsar=1[vr]"
            )

        command.extend(["-filter_complex", filter_graph, "-map", "[vr]"])
        if selected_audio is not None:
            command.extend(selected_audio.output_arguments(2))
        elif preserve_audio:
            command.extend(["-map", "2:a:0?", "-c:a", "aac", "-shortest"])
        command.extend(["-frames:v", str(sequence.frame_count), "-progress", "pipe:1"])
        encoder_args, _ = self._build_encoder_cmd(
            settings.get("video_encoder", "auto"), temporary_output
        )
        command.extend(encoder_args)
        return command

    def _build_schema5_direct_provider_command(
        self,
        provider: EncodingSequenceProvider,
        first_headers: tuple[tuple[int, int], ...],
        temporary_output: Path,
        original_video: str,
        settings: dict[str, Any],
        selected_audio: SelectedAudioIdentityV1,
        fps: str,
        resolved_encoder: str,
    ) -> list[str]:
        """Build the production direct command only from provider scalar state."""

        provider.validated()
        if (
            provider.mode != "direct_stereo"
            or provider.left_parent is None
            or provider.right_parent is None
            or len(first_headers) != 2
        ):
            raise ValueError("schema-5 direct provider/header state is invalid")
        per_eye_width, per_eye_height = self._per_eye_size(settings)
        vr_format = settings.get("vr_format")
        stack_filter = {
            "side_by_side": "hstack",
            "over_under": "vstack",
        }.get(vr_format)
        if stack_filter is None:
            raise ValueError(f"Unsupported direct VR format: {vr_format}")
        del original_video
        root = temporary_output.parent
        left_pattern = root.joinpath(*provider.left_parent.parts) / "frame_%06d.png"
        right_pattern = root.joinpath(*provider.right_parent.parts) / "frame_%06d.png"
        common = ["-framerate", fps, "-start_number", str(provider.first_index)]
        command = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-xerror",
            "-nostats",
            *common,
            "-i",
            str(left_pattern),
            *common,
            "-i",
            str(right_pattern),
            *selected_audio.input_arguments(),
        ]
        target = (per_eye_width, per_eye_height)
        if first_headers == (target, target):
            filter_graph = (
                "[0:v]setsar=1[left];[1:v]setsar=1[right];"
                f"[left][right]{stack_filter}=inputs=2:shortest=1[stacked];"
                "[stacked]setsar=1[vr]"
            )
        else:
            scale = (
                f"scale={per_eye_width}:{per_eye_height}:"
                "flags=bicubic+accurate_rnd,setsar=1"
            )
            filter_graph = (
                f"[0:v]{scale}[left];[1:v]{scale}[right];"
                f"[left][right]{stack_filter}=inputs=2:shortest=1[stacked];"
                "[stacked]setsar=1[vr]"
            )
        command.extend(["-filter_complex", filter_graph, "-map", "[vr]"])
        command.extend(selected_audio.output_arguments(2))
        command.extend(["-frames:v", str(provider.frame_count), "-progress", "pipe:1"])
        encoder_args, _ = self._build_encoder_cmd(
            settings.get("video_encoder", "auto"),
            temporary_output,
            resolved_encoder=resolved_encoder,
        )
        command.extend(encoder_args)
        return command

    def _run_ffmpeg_with_progress(
        self,
        command: list[str],
        total_frames: int,
        progress_tracker,
    ) -> tuple[int, tuple[str, ...]]:
        """Run FFmpeg with one merged progress stream and bounded diagnostics."""

        diagnostics: deque[str] = deque(maxlen=50)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        try:
            if process.stdout is None:
                raise OSError("FFmpeg progress stream is unavailable")
            for raw_line in process.stdout:
                self._consume_direct_ffmpeg_line(
                    raw_line, total_frames, progress_tracker, diagnostics
                )
            return process.wait(), tuple(diagnostics)
        except Exception:
            process.terminate()
            try:
                remaining, _ = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                remaining, _ = process.communicate()
            for raw_line in (remaining or "").splitlines():
                self._consume_direct_ffmpeg_line(
                    raw_line, total_frames, progress_tracker, diagnostics
                )
            self._print_direct_ffmpeg_diagnostics(tuple(diagnostics))
            raise

    def _run_schema5_ffmpeg_with_progress(
        self,
        command: list[str],
        total_frames: int,
        progress_tracker=None,
    ) -> tuple[int, tuple[str, ...]]:
        """Run final encoding with bounded binary progress and diagnostic pipes."""

        parser = FfmpegProgressParser()
        result = run_ffmpeg_progress_v1(
            command,
            parser=parser,
            diagnostic_cap=FINAL_DECODE_DIAGNOSTIC_BYTES,
            line_cap=FINAL_ENCODING_LINE_BYTES,
            stall_seconds=120.0,
            cancelled=self._cancellation_predicate(progress_tracker),
        )
        diagnostics = tuple(
            line
            for line in result.stderr.decode("utf-8", errors="replace").splitlines()
            if line
        )
        if result.stderr:
            return result.returncode if result.returncode != 0 else 1, diagnostics
        if progress_tracker is not None and parser.maximums["frame"] >= 0:
            self._consume_direct_ffmpeg_line(
                f"frame={parser.maximums['frame']}",
                total_frames,
                progress_tracker,
                deque(maxlen=1),
            )
        return result.returncode, diagnostics

    @staticmethod
    def _consume_direct_ffmpeg_line(
        raw_line: str,
        total_frames: int,
        progress_tracker,
        diagnostics: deque[str],
    ) -> None:
        """Parse one progress line without allowing tracker failures to escape."""

        line = raw_line.strip()
        if not line:
            return
        if not line.startswith("frame="):
            diagnostics.append(line)
            return
        try:
            frame = min(max(int(line.partition("=")[2]), 0), total_frames)
        except ValueError:
            diagnostics.append(line)
            return
        try:
            if progress_tracker is not None:
                progress_tracker.update_progress(
                    f"Encoding VR frame {frame}/{total_frames}",
                    phase="video_encoding",
                    frame_num=frame,
                    step_name="Direct VR Encoding",
                    step_progress=frame,
                    step_total=total_frames,
                )
        except InterruptedError:
            raise
        except Exception as error:
            print(f"Warning: Direct encoding progress update failed: {error}")

    @staticmethod
    def _print_direct_ffmpeg_diagnostics(diagnostics: tuple[str, ...]) -> None:
        """Print a bounded FFmpeg diagnostic tail when one is available."""

        if diagnostics:
            print("FFmpeg diagnostic tail:")
            for line in diagnostics:
                print(f"  {line}")

    @staticmethod
    def _print_direct_ffmpeg_failure(
        returncode: int, diagnostics: tuple[str, ...]
    ) -> None:
        """Print FFmpeg's bounded diagnostic tail for a failed direct encode."""

        print(f"Error: Direct FFmpeg encoding failed with return code {returncode}.")
        VideoEncoder._print_direct_ffmpeg_diagnostics(diagnostics)

    @staticmethod
    def _print_direct_ffmpeg_output_validation_failure(
        diagnostics: tuple[str, ...],
    ) -> None:
        """Report a zero-exit encode that did not produce a publishable file."""

        print(
            "Error: Direct FFmpeg output validation failed: "
            "temporary output is missing or empty."
        )
        VideoEncoder._print_direct_ffmpeg_diagnostics(diagnostics)

    def create_video_from_stereo_source(  # noqa: C901
        self,
        source: DirectStereoSource,
        output_dir: Path,
        original_video: str,
        settings: dict[str, Any],
        *,
        total_frames: int,
        progress_tracker=None,
    ) -> bool:
        """Encode schema-5 direct VR from parent/range scalars, never a Path graph."""

        cancelled = self._cancellation_predicate(progress_tracker)
        if source.frame_count != total_frames or source.frame_count <= 0:
            print("Error creating direct VR output video: source frame count changed")
            return False
        try:
            recovered_settings = recover_processing_control_prefix_v1(output_dir)
        except InterruptedError:
            raise
        except ReservationIncompleteError:
            raise
        except Exception as error:
            print(f"Error recovering schema-5 video publication: {error}")
            return False
        schema5_present = recovered_settings is not None
        del recovered_settings
        if not schema5_present:
            # Schema 1-4 compatibility is intentionally isolated on the legacy API.
            left_files = [
                source.left_parent / f"frame_{source.first_index + index:06d}.png"
                for index in range(source.frame_count)
            ]
            right_files = [
                source.right_parent / f"frame_{source.first_index + index:06d}.png"
                for index in range(source.frame_count)
            ]
            return self.create_video_from_stereo_sequences(
                left_files,
                right_files,
                output_dir,
                original_video,
                settings,
                total_frames=total_frames,
                progress_tracker=progress_tracker,
            )

        schema5_context: _Schema5EncodingContext | None = None
        temporary: Path | None = None
        try:
            root = Path(os.path.abspath(os.fspath(output_dir)))
            left_parent = Path(
                os.path.abspath(os.fspath(source.left_parent))
            ).relative_to(root)
            right_parent = Path(
                os.path.abspath(os.fspath(source.right_parent))
            ).relative_to(root)
            provider = EncodingSequenceProvider.freeze(
                mode="direct_stereo",
                frame_count=source.frame_count,
                first_index=source.first_index,
                left_parent=PurePosixPath(left_parent.as_posix()),
                right_parent=PurePosixPath(right_parent.as_posix()),
            )
            active_index = output_dir / FINAL_ENCODING_RESERVATION_RELATIVE_PATH
            recover_before_dependency_gate = active_index.is_file()
            if not recover_before_dependency_gate and not verify_ffmpeg_installation():
                print("Error: FFmpeg not found. Cannot create output video.")
                return False
            resolved_encoder_name: str | None = None
            if not recover_before_dependency_gate:
                resolved_encoder_name = self._resolve_encoder_name(
                    settings.get("video_encoder", "auto"),
                    cancelled=cancelled,
                )
            schema5_context = self._prepare_schema5_direct_provider_context(
                provider=provider,
                output_dir=output_dir,
                settings=settings,
                cancelled=cancelled,
            )
            if schema5_context is None:
                raise ValueError("schema-5 direct source lost its settings artifact")
            self._pending_final_reservation = schema5_context.reservation
            if schema5_context.already_authenticated:
                return True
            if recover_before_dependency_gate and not verify_ffmpeg_installation():
                print("Error: FFmpeg not found. Cannot create output video.")
                return False
            if resolved_encoder_name is None:
                resolved_encoder_name = self._resolve_encoder_name(
                    settings.get("video_encoder", "auto"),
                    cancelled=cancelled,
                )
            staged = schema5_context.staged_manifest
            manifest_source = schema5_context.manifest_source
            if staged is None or manifest_source is None:
                raise ValueError("schema-5 direct source has no staged input manifest")
            resolved_fps = schema5_context.resolved_fps
            audio_kwargs = {
                "output_dir": output_dir,
                "original_video": original_video,
                "settings": settings,
                "frame_count": provider.frame_count,
                "fps": resolved_fps,
            }
            if cancelled is not None:
                audio_kwargs["cancelled"] = cancelled
            selected_audio = prepare_selected_audio_v1(**audio_kwargs)
            temporary = schema5_context.sibling_path
            if temporary.exists() or temporary.is_symlink():
                raise ReservationConflictError(
                    "indexed final-video sibling was not retired before launch"
                )
            command = self._build_schema5_direct_provider_command(
                provider,
                staged.first_headers,
                temporary,
                original_video,
                settings,
                selected_audio,
                resolved_fps,
                resolved_encoder_name,
            )
            self._commit_schema5_launch_boundary(
                schema5_context,
                job_root=output_dir,
                settings=settings,
                cancelled=cancelled,
            )
            returncode, diagnostics = self._run_schema5_ffmpeg_with_progress(
                command,
                provider.frame_count,
                progress_tracker,
            )
            if returncode != 0:
                self._print_direct_ffmpeg_failure(returncode, diagnostics)
                return False
            if not temporary.is_file() or temporary.stat().st_size == 0:
                self._print_direct_ffmpeg_output_validation_failure(diagnostics)
                return False
            del diagnostics
            validate_encoding_postpass(
                provider,
                job_root=output_dir,
                staged=staged,
                per_eye_size=self._per_eye_size(settings),
                vr_format=settings["vr_format"],
                cancelled=cancelled,
            )
            selected_audio.revalidate(cancelled=cancelled)
            assert (
                provider.left_parent is not None and provider.right_parent is not None
            )
            left_pattern = root.joinpath(*provider.left_parent.parts) / "frame_%06d.png"
            right_pattern = (
                root.joinpath(*provider.right_parent.parts) / "frame_%06d.png"
            )
            normalized = self._normalize_publication_arguments(
                command=command,
                image_inputs={str(left_pattern): "left", str(right_pattern): "right"},
                selected_audio=selected_audio,
                runtime_output=temporary,
                input_stage_fingerprint=staged.fingerprint,
                final_relative_path=schema5_context.expected_relative_path,
            )
            container_validator = self._container_validator_for_invocation(
                settings,
                provider.frame_count,
                resolved_fps,
                resolved_encoder_name,
                selected_audio.audio_end_sample,
                cancelled,
            )
            validated_container = validate_final_container_evidence_v1(
                output_dir,
                temporary,
                container_validator,
            )
            publication_settings = recover_processing_control_prefix_v1(output_dir)
            if publication_settings is None:
                raise ReservationConflictError(
                    "final publication lost its settings authority"
                )
            publication = FinalPublicationTransactionV4(
                job_root=output_dir,
                settings=publication_settings,
                sibling_relative_path=temporary.name,
                encoding_input_source=manifest_source,
                input_stage_fingerprint=staged.fingerprint,
                resolved_encoding_arguments=normalized,
                container_validator=None,
                expected_sibling_identity=validated_container.identity,
                expected_sibling_sha256=validated_container.raw_sha256,
                expected_sibling_byte_count=validated_container.byte_count,
                reservation=schema5_context.reservation,
                retire_reservation=False,
            )
            del validated_container
            del publication_settings
            publication.publish()
            self._pending_final_reservation = schema5_context.reservation
            return True
        except InterruptedError:
            raise
        except ReservationIncompleteError:
            raise
        except Exception as error:
            print(f"Error creating direct VR output video: {error}")
            return False
        finally:
            if schema5_context is not None:
                self._discard_schema5_sibling_after_attempt(schema5_context)
            elif temporary is not None:
                temporary.unlink(missing_ok=True)

    def create_video_from_stereo_sequences(  # noqa: C901
        self,
        left_files: Sequence[Path],
        right_files: Sequence[Path],
        output_dir: Path,
        original_video: str,
        settings: dict[str, Any],
        *,
        total_frames: int,
        progress_tracker=None,
    ) -> bool:
        """Encode validated eye sequences and atomically publish the final video."""

        cancelled = self._cancellation_predicate(progress_tracker)
        try:
            recovered_settings = recover_processing_control_prefix_v1(output_dir)
        except InterruptedError:
            raise
        except ReservationIncompleteError:
            raise
        except Exception as error:
            print(f"Error recovering schema-5 video publication: {error}")
            return False
        schema5_present = recovered_settings is not None
        del recovered_settings
        if schema5_present:
            print(
                "Error creating direct VR output video: schema-5 requires the scalar "
                "create_video_from_stereo_source API"
            )
            return False
        temporary: Path | None = None
        schema5_context: _Schema5EncodingContext | None = None
        legacy_output_path: Path | None = None
        if not (output_dir / SETTINGS_ARTIFACT_RELATIVE_PATH).is_file():
            output_filename = generate_output_filename(
                Path(original_video).name,
                settings["vr_format"],
                settings["vr_resolution"],
            )
            legacy_output_path = output_dir / output_filename
            temporary = legacy_output_path.with_name(
                f".{legacy_output_path.stem}.direct.tmp.mp4"
            )
            temporary.unlink(missing_ok=True)
        try:
            active_index = output_dir / FINAL_ENCODING_RESERVATION_RELATIVE_PATH
            recover_before_dependency_gate = active_index.is_file()
            if not recover_before_dependency_gate and not verify_ffmpeg_installation():
                print("Error: FFmpeg not found. Cannot create output video.")
                return False
            sequence = self._validate_direct_stereo_sequence(
                left_files,
                right_files,
                total_frames,
            )
            schema5_context = self._prepare_schema5_direct_context(
                sequence=sequence,
                output_dir=output_dir,
                settings=settings,
                cancelled=cancelled,
            )
            if schema5_context is not None:
                self._pending_final_reservation = schema5_context.reservation
            if schema5_context is not None and schema5_context.already_authenticated:
                return True
            if recover_before_dependency_gate and not verify_ffmpeg_installation():
                print("Error: FFmpeg not found. Cannot create output video.")
                return False
            if schema5_context is not None:
                resolved_fps = schema5_context.resolved_fps
                audio_kwargs = {
                    "output_dir": output_dir,
                    "original_video": original_video,
                    "settings": settings,
                    "frame_count": sequence.frame_count,
                    "fps": resolved_fps,
                }
                if cancelled is not None:
                    audio_kwargs["cancelled"] = cancelled
                selected_audio = prepare_selected_audio_v1(**audio_kwargs)
            else:
                selected_audio = None
                resolved_fps = None
            if schema5_context is None:
                assert legacy_output_path is not None and temporary is not None
                output_path = legacy_output_path
            else:
                output_path = output_dir / schema5_context.expected_relative_path
                temporary = schema5_context.sibling_path
            if schema5_context is None:
                temporary.unlink(missing_ok=True)
            elif temporary.exists() or temporary.is_symlink():
                raise ReservationConflictError(
                    "indexed final-video sibling was not retired before launch"
                )
            command = self._build_direct_stereo_command(
                sequence,
                temporary,
                original_video,
                settings,
                selected_audio=selected_audio,
                resolved_fps=resolved_fps,
            )
            if schema5_context is not None:
                assert schema5_context.staged_manifest is not None
                validate_encoding_provider(
                    schema5_context.provider,
                    schema5_context.staged_manifest.provider_audit,
                    launch_started=False,
                    cancelled=cancelled,
                )
            if schema5_context is None:
                returncode, diagnostics = self._run_ffmpeg_with_progress(
                    command, sequence.frame_count, progress_tracker
                )
            else:
                returncode, diagnostics = self._run_schema5_ffmpeg_with_progress(
                    command, sequence.frame_count, progress_tracker
                )
            if returncode != 0:
                self._print_direct_ffmpeg_failure(returncode, diagnostics)
                return False
            if not temporary.is_file() or temporary.stat().st_size == 0:
                self._print_direct_ffmpeg_output_validation_failure(diagnostics)
                return False
            del diagnostics
            if schema5_context is None:
                temporary.replace(output_path)
            else:
                assert schema5_context.staged_manifest is not None
                assert schema5_context.manifest_source is not None
                validate_encoding_postpass(
                    schema5_context.provider,
                    job_root=output_dir,
                    staged=schema5_context.staged_manifest,
                    per_eye_size=self._per_eye_size(settings),
                    vr_format=settings["vr_format"],
                    cancelled=cancelled,
                )
                assert selected_audio is not None
                selected_audio.revalidate(cancelled=cancelled)
                normalized = self._normalize_publication_arguments(
                    command=command,
                    image_inputs={
                        str(sequence.left_pattern): "left",
                        str(sequence.right_pattern): "right",
                    },
                    selected_audio=selected_audio,
                    runtime_output=temporary,
                    input_stage_fingerprint=(
                        schema5_context.staged_manifest.fingerprint
                    ),
                    final_relative_path=schema5_context.expected_relative_path,
                )
                container_validator = self._container_validator_for_invocation(
                    settings,
                    schema5_context.provider.frame_count,
                    schema5_context.resolved_fps,
                    command[command.index("-c:v") + 1],
                    selected_audio.audio_end_sample,
                    cancelled,
                )
                validated_container = validate_final_container_evidence_v1(
                    output_dir,
                    temporary,
                    container_validator,
                )
                publication_settings = recover_processing_control_prefix_v1(output_dir)
                if publication_settings is None:
                    raise ReservationConflictError(
                        "final publication lost its settings authority"
                    )
                publication = FinalPublicationTransactionV4(
                    job_root=output_dir,
                    settings=publication_settings,
                    sibling_relative_path=temporary.name,
                    encoding_input_source=schema5_context.manifest_source,
                    input_stage_fingerprint=(
                        schema5_context.staged_manifest.fingerprint
                    ),
                    resolved_encoding_arguments=normalized,
                    container_validator=None,
                    expected_sibling_identity=validated_container.identity,
                    expected_sibling_sha256=validated_container.raw_sha256,
                    expected_sibling_byte_count=validated_container.byte_count,
                    reservation=schema5_context.reservation,
                    retire_reservation=False,
                )
                del validated_container
                del publication_settings
                publication.publish()
                self._pending_final_reservation = schema5_context.reservation
            return True
        except InterruptedError:
            raise
        except ReservationIncompleteError:
            raise
        except Exception as error:
            print(f"Error creating direct VR output video: {error}")
            return False
        finally:
            if schema5_context is not None:
                self._discard_schema5_sibling_after_attempt(schema5_context)
            elif temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as error:
                    print(
                        f"Warning: Could not clean direct VR temporary output: {error}"
                    )

    def extract_frames(
        self,
        video_path: str,
        directories: dict[str, Path],
        video_properties: dict[str, Any],
        settings: dict[str, Any],
    ) -> list[Path]:
        """
        Extract frames from video using FFmpeg.

        Args:
            video_path: Input video path
            directories: Dictionary of processing directories
            video_properties: Video metadata (frame_count, fps)
            settings: Processing settings with frame range info

        Returns:
            List of extracted frame file paths

        Side effects:
            - Executes FFmpeg subprocess
            - Writes frame images to disk
        """
        frames_dir = directories.get("frames")
        if not frames_dir:
            frames_dir = directories["base"] / INTERMEDIATE_DIRS["frames"]
            frames_dir.mkdir(exist_ok=True)

        # Calculate frame range and timing
        total_frames = video_properties["frame_count"]
        fps = video_properties["fps"]
        start_frame, end_frame = calculate_frame_range(
            total_frames, fps, settings.get("start_time"), settings.get("end_time")
        )
        expected_frames = end_frame - start_frame

        existing_frames = sorted(frames_dir.glob("frame_*.png"))
        frame_count_tolerance = max(2, (expected_frames + 999) // 1000)
        exact_count = len(existing_frames) == expected_frames
        tolerated_vfr_drift = (
            bool(existing_frames)
            and abs(len(existing_frames) - expected_frames) <= frame_count_tolerance
            and self._has_downstream_progress(directories)
        )
        source_path = Path(video_path)
        reusable_manifest = False
        if source_path.is_file():
            source_fingerprint = file_sample_fingerprint(source_path)
            reusable_manifest = (
                source_frame_manifest_mismatch_reason(
                    read_source_frame_manifest(frames_dir),
                    existing_frames,
                    source_fingerprint,
                )
                is None
            )
        if (
            expected_frames > 0
            and (exact_count or tolerated_vfr_drift)
            and reusable_manifest
        ):
            print(f"  Reusing {len(existing_frames)} already extracted frames")
            return existing_frames

        # FFmpeg overwrites matching names but does not remove stale tail frames.
        # Clear only this stage's exact output pattern before a full re-extraction.
        for frame_file in existing_frames:
            frame_file.unlink(missing_ok=True)
        (frames_dir / "metadata.json").unlink(missing_ok=True)

        # Convert frame numbers to timestamps for more efficient seeking
        start_time = start_frame / fps if fps > 0 else 0
        duration = expected_frames / fps if fps > 0 else 0

        # Try CUDA hardware decoding with optimized seeking
        cmd_cuda = [
            "ffmpeg",
            "-y",
            "-hwaccel",
            "cuda",
            "-ss",
            str(start_time),  # Seek before decoding (much faster)
            "-i",
            video_path,
            "-t",
            str(duration),  # Duration limit (more efficient than select filter)
            "-pix_fmt",
            "rgb24",
            "-fps_mode",
            "passthrough",  # Pass through original timestamps
            str(frames_dir / "frame_%06d.png"),
        ]

        try:
            result = subprocess.run(cmd_cuda, capture_output=True, text=True)
            if result.returncode != 0:
                # CUDA failed, try CPU fallback with optimized seeking
                print("  CUDA frame extraction failed, falling back to CPU")
                cmd_cpu = [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(start_time),  # Seek before decoding
                    "-i",
                    video_path,
                    "-t",
                    str(duration),  # Duration limit
                    "-pix_fmt",
                    "rgb24",
                    "-fps_mode",
                    "passthrough",
                    "-threads",
                    "0",  # Auto-detect optimal thread count
                    str(frames_dir / "frame_%06d.png"),
                ]
                result_cpu = subprocess.run(cmd_cpu, capture_output=True, text=True)
                if result_cpu.returncode != 0:
                    print(f"FFmpeg error: {result_cpu.stderr}")
                    return []
        except Exception as e:
            print(f"Error extracting frames: {e}")
            return []

        frame_files = get_frame_files(frames_dir)
        if frame_files:
            write_source_frame_manifest(frame_files, video_path)
        return frame_files

    @staticmethod
    def _has_downstream_progress(directories: dict[str, Path]) -> bool:
        """A later-stage file proves that FFmpeg completed before processing advanced."""
        for stage_name in INTERMEDIATE_DIRS:
            if stage_name == "frames":
                continue
            stage_dir = directories.get(stage_name)
            if stage_dir is not None:
                if next(stage_dir.glob("*.png"), None) is not None:
                    return True
                if next(stage_dir.glob("*.npz"), None) is not None:
                    return True
        return False

    def _check_nvenc_available(
        self,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> bool:
        """
        Check if NVIDIA NVENC hardware encoder is available.

        Returns:
            True if NVENC is available, False otherwise

        Side effects:
            - Executes FFmpeg subprocess for encoder detection
        """
        try:
            test_result = BoundedProcessCaptureV1(
                stdout_cap=1024 * 1024,
                stderr_cap=64 * 1024,
                timeout_seconds=10.0,
                cancelled=cancelled,
            ).run(("ffmpeg", "-hide_banner", "-encoders"))
        except InterruptedError:
            raise
        except (BoundedProcessError, OSError):
            return False
        return (
            test_result.returncode == 0
            and not test_result.stderr
            and b"hevc_nvenc" in test_result.stdout
        )

    def _resolve_encoder_name(
        self,
        encoder: str,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> Literal["libx264", "libx265", "hevc_nvenc"]:
        if encoder in {"auto", "nvenc"}:
            available = (
                self._check_nvenc_available()
                if cancelled is None
                else self._check_nvenc_available(cancelled=cancelled)
            )
            if available:
                return "hevc_nvenc"
        if encoder in {"libx264", "libx265"}:
            return cast(Literal["libx264", "libx265"], encoder)
        return "libx264"

    def _build_encoder_cmd(
        self,
        encoder: str,
        output_path: Path,
        *,
        resolved_encoder: str | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[list[str], bool]:
        """
        Build FFmpeg encoder arguments.

        Args:
            encoder: Encoder name (auto, nvenc, libx264, libx265)
            output_path: Output video file path

        Returns:
            Tuple of (encoder_args, is_nvenc_used)

        Side effects:
            - May print console output for encoder selection
            - Checks NVENC availability
        """
        codec = resolved_encoder or self._resolve_encoder_name(
            encoder,
            cancelled=cancelled,
        )
        if codec not in {"libx264", "libx265", "hevc_nvenc"}:
            raise ValueError("resolved video encoder is unsupported")
        if codec == "hevc_nvenc":
            print("  Using NVENC hardware encoding (H.265)")
            return (
                [
                    "-c:v",
                    "hevc_nvenc",
                    "-pix_fmt",
                    "yuv420p",
                    "-preset",
                    "p7",
                    "-tune",
                    "hq",
                    "-rc",
                    "vbr",
                    "-cq",
                    "18",
                    "-b:v",
                    "0",
                    str(output_path),
                ],
                True,
            )
        if encoder == "nvenc":
            print("  Warning: NVENC not available, falling back to software encoding")
        elif encoder not in {"auto", "libx264", "libx265"}:
            print(f"  Warning: Unknown encoder '{encoder}', using libx264")

        print(f"  Using software encoding ({codec})")
        return (
            [
                "-c:v",
                codec,
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",  # High quality
                "-preset",
                "medium",
                str(output_path),
            ],
            False,
        )
