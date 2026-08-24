"""
Video encoding module.

Handles FFmpeg-based video encoding with hardware acceleration support (NVENC).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import os
import re
import secrets
import subprocess
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Literal, cast

from ...io.operations import (
    commit_final_media_producer_marker,
    get_frame_files,
    get_video_info_ffprobe,
    verify_ffmpeg_installation,
)
from ...io.reservation_protocol import SETTINGS_ARTIFACT_RELATIVE_PATH
from ...io.settings_artifact import ProcessingSettingsArtifactV5
from ...utils.path_utils import (
    calculate_frame_range,
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
from ...core.file_identity import file_sample_fingerprint
from ...utils.imaging.png_header import PngHeader, read_png_header
from .encoding_provider import (
    FINAL_DECODE_DIAGNOSTIC_BYTES,
    FINAL_ENCODING_LINE_BYTES,
    EncodingSequenceProvider,
    FfmpegProgressParser,
    FinalContainerExpectation,
    StagedEncodingInputManifest,
    audit_encoding_input_manifest,
    stage_encoding_input_manifest,
    validate_encoding_postpass,
)
from .final_publication import (
    EncodingInputPublicationSource,
    FINAL_ENCODING_RESERVATION_RELATIVE_PATH,
    FinalPublicationTransactionV4,
    PreparedFinalEncodingReservationV1,
    prepare_final_encoding_reservation_v1,
    load_final_encoding_reservation_v1,
)
from .bounded_process import run_ffmpeg_progress_v1
from .final_container_validation import validate_final_container_v4
from .final_audio import SelectedAudioIdentityV1, prepare_selected_audio_v1


_DIRECT_FRAME_NAME = re.compile(r"^frame_(\d{6,})\.png$")


@dataclass(frozen=True)
class _DirectStereoSequence:
    """Validated image2 manifests and the only headers needed for direct encoding."""

    left_files: tuple[Path, ...]
    right_files: tuple[Path, ...]
    left_pattern: Path
    right_pattern: Path
    start_number: int
    frame_count: int
    left_header: PngHeader
    right_header: PngHeader


@dataclass(frozen=True)
class _Schema5EncodingContext:
    settings: ProcessingSettingsArtifactV5
    provider: EncodingSequenceProvider
    staged_manifest: StagedEncodingInputManifest | None
    manifest_source: EncodingInputPublicationSource | None
    expected_relative_path: str
    sibling_path: Path
    reservation: PreparedFinalEncodingReservationV1
    already_authenticated: bool = False


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
        self._pending_final_reservation: PreparedFinalEncodingReservationV1 | None = None

    def retire_final_encoding_reservation(self, *, success: bool) -> None:
        reservation = self._pending_final_reservation
        if reservation is None:
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
        if not verify_ffmpeg_installation():
            print("Error: FFmpeg not found. Cannot create output video.")
            return False

        schema5_context = self._prepare_schema5_assembled_context(
            vr_frames_dir=vr_frames_dir,
            output_dir=output_dir,
            settings=settings,
        )
        if schema5_context is not None and schema5_context.already_authenticated:
            if self.defer_final_reservation_retirement:
                self._pending_final_reservation = schema5_context.reservation
            else:
                schema5_context.reservation.retire()
            return True
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
        base_fps = self._resolve_output_fps(original_video, settings)

        cmd = ["ffmpeg", "-y"]
        if schema5_context is not None:
            cmd.extend(["-v", "error", "-xerror", "-nostats", "-progress", "pipe:1"])
        cmd.extend(
            [
                "-framerate",
                str(base_fps),
                "-i",
                str(vr_frames_dir / "frame_%06d.png"),
            ]
        )

        selected_audio: SelectedAudioIdentityV1 | None = None
        # Legacy jobs retain their historical optional-map behavior. Schema 5
        # freezes an exact audio identity before constructing the command.
        audio_source: str | Path | None = None
        if schema5_context is not None:
            selected_audio = prepare_selected_audio_v1(
                output_dir=output_dir,
                original_video=original_video,
                settings=settings,
                frame_count=schema5_context.provider.frame_count,
                fps=str(base_fps),
            )
            cmd.extend(selected_audio.input_arguments())
            cmd.extend(["-map", "0:v:0"])
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
        encoder_args, _ = self._build_encoder_cmd(encoder, output_path)
        resolved_encoder = encoder_args[encoder_args.index("-c:v") + 1]
        cmd.extend(encoder_args)

        try:
            if schema5_context is not None:
                returncode, diagnostics = self._run_schema5_ffmpeg_with_progress(
                    cmd,
                    schema5_context.provider.frame_count,
                )
                if returncode != 0:
                    self._print_direct_ffmpeg_failure(returncode, diagnostics)
                    return False
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
                )
                assert selected_audio is not None
                selected_audio.revalidate()
                normalized = self._normalize_publication_arguments(
                    command=cmd,
                    image_inputs={
                        str(vr_frames_dir / "frame_%06d.png"): "frames",
                    },
                    selected_audio=selected_audio,
                    runtime_output=output_path,
                    input_stage_fingerprint=(schema5_context.staged_manifest.fingerprint),
                    final_relative_path=schema5_context.expected_relative_path,
                )
                FinalPublicationTransactionV4(
                    job_root=output_dir,
                    settings=schema5_context.settings,
                    sibling_relative_path=output_path.name,
                    encoding_input_source=schema5_context.manifest_source,
                    input_stage_fingerprint=(schema5_context.staged_manifest.fingerprint),
                    resolved_encoding_arguments=normalized,
                    container_validator=lambda path: self._validate_final_container(
                        path,
                        settings,
                        schema5_context.provider.frame_count,
                        str(base_fps),
                        resolved_encoder,
                        selected_audio.audio_end_sample,
                    ),
                    reservation=schema5_context.reservation,
                    retire_reservation=not self.defer_final_reservation_retirement,
                ).publish()
                if self.defer_final_reservation_retirement:
                    self._pending_final_reservation = schema5_context.reservation
            return True
        except Exception as e:
            print(f"Error creating output video: {e}")
            return False
        finally:
            if schema5_context is not None:
                try:
                    schema5_context.sibling_path.unlink(missing_ok=True)
                except OSError:
                    pass

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

    def _prepare_schema5_assembled_context(
        self,
        *,
        vr_frames_dir: Path,
        output_dir: Path,
        settings: dict[str, Any],
    ) -> _Schema5EncodingContext | None:
        settings_path = output_dir / SETTINGS_ARTIFACT_RELATIVE_PATH
        if not settings_path.is_file():
            return None
        artifact = ProcessingSettingsArtifactV5.parse(settings_path.read_bytes())
        frame_count = 0
        first_index: int | None = None
        last_index: int | None = None
        for path in vr_frames_dir.iterdir():
            match = _DIRECT_FRAME_NAME.fullmatch(path.name)
            if match is not None and path.is_file() and not path.is_symlink():
                index = int(match.group(1))
                if path.name != f"frame_{index:06d}.png":
                    raise ValueError("assembled frame padding is noncanonical")
                frame_count += 1
                first_index = index if first_index is None else min(first_index, index)
                last_index = index if last_index is None else max(last_index, index)
        if frame_count == 0 or first_index is None or last_index is None:
            raise ValueError("assembled schema-5 encoding has no frames")
        if frame_count != last_index - first_index + 1:
            raise ValueError("assembled schema-5 frame sequence is not contiguous")
        root = output_dir.resolve()
        parent = vr_frames_dir.resolve().relative_to(root)
        provider = EncodingSequenceProvider.freeze(
            mode="assembled_vr",
            frame_count=frame_count,
            first_index=first_index,
            frames_parent=PurePosixPath(parent.as_posix()),
        )
        return self._prepare_schema5_publication_context(
            provider=provider,
            output_dir=output_dir,
            artifact=artifact,
            settings_path=settings_path,
            settings=settings,
        )

    def _prepare_schema5_direct_context(
        self,
        *,
        sequence: _DirectStereoSequence,
        output_dir: Path,
        settings: dict[str, Any],
    ) -> _Schema5EncodingContext | None:
        settings_path = output_dir / SETTINGS_ARTIFACT_RELATIVE_PATH
        if not settings_path.is_file():
            return None
        artifact = ProcessingSettingsArtifactV5.parse(settings_path.read_bytes())
        root = output_dir.resolve()
        left_parent = sequence.left_pattern.parent.resolve().relative_to(root)
        right_parent = sequence.right_pattern.parent.resolve().relative_to(root)
        provider = EncodingSequenceProvider.freeze(
            mode="direct_stereo",
            frame_count=sequence.frame_count,
            first_index=sequence.start_number,
            left_parent=PurePosixPath(left_parent.as_posix()),
            right_parent=PurePosixPath(right_parent.as_posix()),
        )
        return self._prepare_schema5_publication_context(
            provider=provider,
            output_dir=output_dir,
            artifact=artifact,
            settings_path=settings_path,
            settings=settings,
        )

    def _prepare_schema5_publication_context(
        self,
        *,
        provider: EncodingSequenceProvider,
        output_dir: Path,
        artifact: ProcessingSettingsArtifactV5,
        settings_path: Path,
        settings: dict[str, Any],
    ) -> _Schema5EncodingContext:
        metadata = artifact.to_object()["metadata"]
        publication_generation = metadata["final_media_publication_generation"]
        active_index = output_dir / FINAL_ENCODING_RESERVATION_RELATIVE_PATH
        if active_index.is_file():
            reservation = load_final_encoding_reservation_v1(output_dir, artifact)
            recovery = reservation.observe()
            if recovery.phase == "video_replaced_input_intent":
                reservation.recover_input_manifest_intent()
                recovery = reservation.observe()
            if recovery.phase == "final_manifest_intent":
                reservation.recover_final_manifest_intent()
                recovery = reservation.observe()
            if recovery.phase == "publication_authenticated":
                assert publication_generation is not None
                return _Schema5EncodingContext(
                    settings=artifact,
                    provider=provider,
                    staged_manifest=None,
                    manifest_source=None,
                    expected_relative_path=artifact.to_object()["output_info"][
                        "expected_final_relative_path"
                    ],
                    sibling_path=reservation.sibling_path,
                    reservation=reservation,
                    already_authenticated=True,
                )
            if recovery.phase not in {
                "ready_pre_video",
                "input_manifest_committed_no_final_intent",
                "post_boundary_retry_sibling_present",
            }:
                raise ValueError(f"unsupported final encoding recovery phase: {recovery.phase}")
            if reservation.sibling_path.exists() or reservation.sibling_path.is_symlink():
                if not reservation.sibling_path.is_file() or reservation.sibling_path.is_symlink():
                    raise ValueError("indexed retry sibling is unsafe")
                reservation.sibling_path.unlink()
            reservation.reset_incomplete_manifest_sources()
            manifest_source = reservation.input_publication_source()
            if manifest_source is None:
                staged, manifest_source = self._stage_encoding_manifest_file(
                    provider=provider,
                    output_dir=output_dir,
                    settings=settings,
                    reservation=reservation,
                )
            else:
                staged = audit_encoding_input_manifest(
                    provider,
                    job_root=output_dir,
                    expected_fingerprint=manifest_source.fingerprint,
                    expected_raw_sha256=manifest_source.raw_sha256,
                    expected_raw_byte_count=manifest_source.raw_byte_count,
                    per_eye_size=self._per_eye_size(settings),
                    vr_format=settings["vr_format"],
                )
            publication_generation = reservation.index.publication_generation
        else:
            if publication_generation is None:
                publication_generation = secrets.token_hex(16)
            reservation = prepare_final_encoding_reservation_v1(
                job_root=output_dir,
                settings=artifact,
                publication_generation=publication_generation,
                input_manifest_raw_max=self._encoding_manifest_raw_bound(provider),
                final_manifest_raw_max=1024 * 1024,
            )
            staged, manifest_source = self._stage_encoding_manifest_file(
                provider=provider,
                output_dir=output_dir,
                settings=settings,
                reservation=reservation,
            )
        artifact = commit_final_media_producer_marker(
            settings_path,
            generation=publication_generation,
        )
        return _Schema5EncodingContext(
            settings=artifact,
            provider=provider,
            staged_manifest=staged,
            manifest_source=manifest_source,
            expected_relative_path=artifact.to_object()["output_info"][
                "expected_final_relative_path"
            ],
            sibling_path=reservation.sibling_path,
            reservation=reservation,
        )

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
        if raw_bound > 512 * 1024 * 1024:
            raise ValueError("encoding input manifest exceeds its 512 MiB raw bound")
        return raw_bound

    def _stage_encoding_manifest_file(
        self,
        *,
        provider: EncodingSequenceProvider,
        output_dir: Path,
        settings: dict[str, Any],
        reservation: PreparedFinalEncodingReservationV1,
    ) -> tuple[StagedEncodingInputManifest, EncodingInputPublicationSource]:
        transaction = reservation.input_manifest_transaction
        if transaction is None:
            raise ValueError("encoding input reservation is already consumed")
        path = transaction.source_path
        try:
            with path.open("r+b", buffering=0) as sink:
                sink.seek(0)
                staged = stage_encoding_input_manifest(
                    provider,
                    job_root=output_dir,
                    sink=sink,
                    per_eye_size=self._per_eye_size(settings),
                    vr_format=settings["vr_format"],
                )
                if staged.raw_byte_count > transaction.descriptor.payload_raw_max:
                    raise ValueError("encoding manifest exceeded its reserved raw bound")
                sink.flush()
                os.fsync(sink.fileno())
            source = EncodingInputPublicationSource(
                path=path,
                fingerprint=staged.fingerprint,
                raw_sha256=staged.raw_sha256,
                raw_byte_count=staged.raw_byte_count,
                logical_byte_count=transaction.descriptor.logical_byte_count,
            ).validated()
            transaction.validate_streamed_physical_phase(
                prefix_byte_count=staged.raw_byte_count,
                prefix_sha256=staged.raw_sha256,
            )
            return staged, source
        except BaseException:
            raise

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
        audio_text = str(selected_audio.selected_source_path) if selected_audio.usable else None
        for index, argument in enumerate(command):
            text = str(argument)
            if index > 0 and command[index - 1] == "-i":
                view = image_inputs.get(text)
                if view is not None:
                    text = "@image-sequence:sha256=" f"{input_stage_fingerprint}:view={view}"
                elif audio_text is not None and text == audio_text:
                    assert audio_token is not None
                    text = audio_token
                else:
                    raise ValueError("runtime argv contains an unbound input path")
            if index == len(command) - 1:
                if text != str(runtime_output):
                    raise ValueError("runtime argv final output path changed")
                text = "@output:path64=" + FinalPublicationTransactionV4.path64(final_relative_path)
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
            resolved_encoder=cast(Literal["libx264", "libx265", "hevc_nvenc"], resolved_encoder),
            audio_expected=audio_end_sample is not None,
            audio_packet_upper_bound=packet_bound,
        )
        return validate_final_container_v4(path, expectation)

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
            args.extend(["-ss", f"{clip_start:g}"])
        if end_seconds is not None and end_seconds > clip_start:
            args.extend(["-t", f"{end_seconds - clip_start:g}"])
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
        left_files: list[Path],
        right_files: list[Path],
        total_frames: int,
    ) -> _DirectStereoSequence:
        """Validate corresponding image2 eye sequences before FFmpeg can start."""

        if not left_files or not right_files:
            raise ValueError("Direct stereo sequences must contain frames for both eyes")
        if len({path.parent for path in left_files}) != 1:
            raise ValueError("Direct left-eye frames must have one parent directory")
        if len({path.parent for path in right_files}) != 1:
            raise ValueError("Direct right-eye frames must have one parent directory")

        sorted_left = tuple(sorted(left_files, key=self._direct_frame_index))
        sorted_right = tuple(sorted(right_files, key=self._direct_frame_index))
        left_names = tuple(path.name for path in sorted_left)
        right_names = tuple(path.name for path in sorted_right)
        if left_names != right_names:
            raise ValueError("Direct stereo sequences must have identical frame names")
        if total_frames > 0 and len(sorted_left) != total_frames:
            raise ValueError("Direct stereo sequence count does not match total frames")

        indices = tuple(self._direct_frame_index(path) for path in sorted_left)
        if any(current != previous + 1 for previous, current in zip(indices, indices[1:])):
            raise ValueError("Direct stereo sequence frame indices must be gap-free")

        left_header = read_png_header(sorted_left[0])
        right_header = read_png_header(sorted_right[0])
        if left_header is None or right_header is None:
            raise ValueError("Direct stereo sequence has an unreadable first PNG")

        return _DirectStereoSequence(
            left_files=sorted_left,
            right_files=sorted_right,
            left_pattern=sorted_left[0].parent / "frame_%06d.png",
            right_pattern=sorted_right[0].parent / "frame_%06d.png",
            start_number=indices[0],
            frame_count=len(sorted_left),
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
    ) -> list[str]:
        """Build an FFmpeg command that stacks validated left and right image2 inputs."""

        try:
            per_eye_width = int(settings["per_eye_width"])
            per_eye_height = int(settings["per_eye_height"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Direct stereo encoding requires per-eye dimensions") from error
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

        fps = self._resolve_output_fps(original_video, settings)
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
            audio_source: str | Path = audio_file if audio_file.exists() else original_video
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
            filter_graph = f"[0:v][1:v]{stack_filter}=inputs=2:shortest=1[vr]"
        else:
            filter_graph = (
                f"[0:v]scale={per_eye_width}:{per_eye_height}:flags=bicubic+accurate_rnd[left];"
                f"[1:v]scale={per_eye_width}:{per_eye_height}:flags=bicubic+accurate_rnd[right];"
                f"[left][right]{stack_filter}=inputs=2:shortest=1[vr]"
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
        )
        diagnostics = tuple(
            line for line in result.stderr.decode("utf-8", errors="replace").splitlines() if line
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
    def _print_direct_ffmpeg_failure(returncode: int, diagnostics: tuple[str, ...]) -> None:
        """Print FFmpeg's bounded diagnostic tail for a failed direct encode."""

        print(f"Error: Direct FFmpeg encoding failed with return code {returncode}.")
        VideoEncoder._print_direct_ffmpeg_diagnostics(diagnostics)

    @staticmethod
    def _print_direct_ffmpeg_output_validation_failure(diagnostics: tuple[str, ...]) -> None:
        """Report a zero-exit encode that did not produce a publishable file."""

        print(
            "Error: Direct FFmpeg output validation failed: "
            "temporary output is missing or empty."
        )
        VideoEncoder._print_direct_ffmpeg_diagnostics(diagnostics)

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

        temporary: Path | None = None
        legacy_output_path: Path | None = None
        if not (output_dir / SETTINGS_ARTIFACT_RELATIVE_PATH).is_file():
            output_filename = generate_output_filename(
                Path(original_video).name,
                settings["vr_format"],
                settings["vr_resolution"],
            )
            legacy_output_path = output_dir / output_filename
            temporary = legacy_output_path.with_name(f".{legacy_output_path.stem}.direct.tmp.mp4")
            temporary.unlink(missing_ok=True)
        try:
            if not verify_ffmpeg_installation():
                print("Error: FFmpeg not found. Cannot create output video.")
                return False
            sequence = self._validate_direct_stereo_sequence(
                list(left_files), list(right_files), total_frames
            )
            schema5_context = self._prepare_schema5_direct_context(
                sequence=sequence,
                output_dir=output_dir,
                settings=settings,
            )
            if schema5_context is not None and schema5_context.already_authenticated:
                if self.defer_final_reservation_retirement:
                    self._pending_final_reservation = schema5_context.reservation
                else:
                    schema5_context.reservation.retire()
                return True
            selected_audio = (
                prepare_selected_audio_v1(
                    output_dir=output_dir,
                    original_video=original_video,
                    settings=settings,
                    frame_count=sequence.frame_count,
                    fps=self._resolve_output_fps(original_video, settings),
                )
                if schema5_context is not None
                else None
            )
            if schema5_context is None:
                assert legacy_output_path is not None and temporary is not None
                output_path = legacy_output_path
            else:
                output_path = output_dir / schema5_context.expected_relative_path
                temporary = schema5_context.sibling_path
            temporary.unlink(missing_ok=True)
            command = self._build_direct_stereo_command(
                sequence,
                temporary,
                original_video,
                settings,
                selected_audio=selected_audio,
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
                )
                assert selected_audio is not None
                selected_audio.revalidate()
                normalized = self._normalize_publication_arguments(
                    command=command,
                    image_inputs={
                        str(sequence.left_pattern): "left",
                        str(sequence.right_pattern): "right",
                    },
                    selected_audio=selected_audio,
                    runtime_output=temporary,
                    input_stage_fingerprint=(schema5_context.staged_manifest.fingerprint),
                    final_relative_path=schema5_context.expected_relative_path,
                )
                FinalPublicationTransactionV4(
                    job_root=output_dir,
                    settings=schema5_context.settings,
                    sibling_relative_path=temporary.name,
                    encoding_input_source=schema5_context.manifest_source,
                    input_stage_fingerprint=(schema5_context.staged_manifest.fingerprint),
                    resolved_encoding_arguments=normalized,
                    container_validator=lambda path: self._validate_final_container(
                        path,
                        settings,
                        schema5_context.provider.frame_count,
                        self._resolve_output_fps(original_video, settings),
                        command[command.index("-c:v") + 1],
                        selected_audio.audio_end_sample,
                    ),
                    reservation=schema5_context.reservation,
                    retire_reservation=not self.defer_final_reservation_retirement,
                ).publish()
                if self.defer_final_reservation_retirement:
                    self._pending_final_reservation = schema5_context.reservation
            return True
        except InterruptedError:
            raise
        except Exception as error:
            print(f"Error creating direct VR output video: {error}")
            return False
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as error:
                    print(f"Warning: Could not clean direct VR temporary output: {error}")

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
        if expected_frames > 0 and (exact_count or tolerated_vfr_drift) and reusable_manifest:
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
            "-vsync",
            "0",  # Pass through original timestamps
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
                    "-vsync",
                    "0",
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

    def _check_nvenc_available(self) -> bool:
        """
        Check if NVIDIA NVENC hardware encoder is available.

        Returns:
            True if NVENC is available, False otherwise

        Side effects:
            - Executes FFmpeg subprocess for encoder detection
        """
        test_result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True
        )
        return "hevc_nvenc" in test_result.stdout

    def _build_encoder_cmd(self, encoder: str, output_path: Path) -> tuple[list[str], bool]:
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
        # Try NVENC for auto or explicit nvenc
        if encoder in ["auto", "nvenc"]:
            if self._check_nvenc_available():
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
                        str(output_path),
                    ],
                    True,
                )
            elif encoder == "nvenc":
                print("  Warning: NVENC not available, falling back to software encoding")
            # Fall through to software encoding

        # Software encoding (default or explicit)
        if encoder in ["libx264", "libx265"]:
            codec = encoder
        else:
            # Unknown encoder or auto fallback
            codec = "libx264"
            if encoder not in ["auto", "nvenc"]:
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
