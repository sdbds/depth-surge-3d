"""
Video encoding module.

Handles FFmpeg-based video encoding with hardware acceleration (NVENC),
zero-copy raw BGR24 streaming, lightweight FFprobe validation, and atomic publication.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
import os
from pathlib import Path
import re
import secrets
import subprocess
import threading
from typing import Any, Literal, cast

import numpy as np

from ...core.constants import DEFAULT_FALLBACK_FPS, INTERMEDIATE_DIRS
from ...core.file_identity import file_sample_fingerprint
from ...core.video_timing import EncodingTimeline, probe_frame_rate, rate_text, selected_frame_range
from ...io.operations import (
    get_frame_files,
    get_video_info_ffprobe,
    recover_processing_control_prefix_v1,
    verify_ffmpeg_installation,
)
from ...utils.imaging.png_header import PngHeader, read_png_header
from ...utils.path_utils import (
    ffmpeg_frame_interval,
    format_ffmpeg_time,
    generate_output_filename,
    parse_time_string,
)
from ..frames.source_frame_manifest import (
    read_source_frame_manifest,
    source_frame_manifest_mismatch_reason,
    write_source_frame_manifest,
)
from ..frames.video_frame_source import VideoFrameSource, build_video_frame_source
from ..frames.vr_assembler import DirectStereoSource
from .final_audio import SelectedAudioIdentityV1, prepare_selected_audio_v1
from .final_container_validation import (
    FinalContainerExpectation,
    validate_final_container_v4,
)


_DIRECT_FRAME_NAME = re.compile(r"^frame_(\d{6,})\.png$")


@dataclass(frozen=True)
class _DirectStereoSequence:
    """Validated image2 stereo sequence headers for direct FFmpeg encoding."""

    left_pattern: Path
    right_pattern: Path
    start_number: int
    frame_count: int
    left_header: PngHeader
    right_header: PngHeader


class StreamedVrEncodingSession:
    """Zero-copy raw BGR24 FFmpeg streaming session with atomic MP4 publication."""

    def __init__(
        self,
        *,
        encoder: "VideoEncoder",
        output_dir: Path,
        final_output_path: Path,
        temp_output_path: Path,
        settings: dict[str, Any],
        frame_count: int,
        width: int,
        height: int,
        timeline: EncodingTimeline,
        resolved_encoder: Literal["libx264", "libx265", "hevc_nvenc"],
        selected_audio: SelectedAudioIdentityV1,
        process: subprocess.Popen[bytes],
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        self.encoder = encoder
        self.output_dir = Path(output_dir)
        self.final_output_path = Path(final_output_path)
        self.temp_output_path = Path(temp_output_path)
        self.settings = settings
        self.frame_count = frame_count
        self.width = width
        self.height = height
        self.timeline = timeline
        self.resolved_encoder = resolved_encoder
        self.selected_audio = selected_audio
        self.process = process
        self.cancelled = cancelled

        self._written_frames = 0
        self._closed = False
        self._published = False
        self._stderr_lines: deque[str] = deque(maxlen=50)
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stdout_thread = threading.Thread(target=self._drain_stdout, daemon=True)
        self._stderr_thread.start()
        self._stdout_thread.start()

    def _drain_stderr(self) -> None:
        stream = self.process.stderr
        if stream is None:
            return
        try:
            for raw_line in stream:
                text = raw_line.decode("utf-8", errors="replace").strip()
                if text:
                    self._stderr_lines.append(text)
        except OSError:
            pass

    def _drain_stdout(self) -> None:
        stream = self.process.stdout
        if stream is None:
            return
        try:
            for _raw_line in stream:
                pass
        except OSError:
            pass

    def write_vr_frame(self, frame_name: str, vr_frame: np.ndarray) -> None:
        """Write one contiguous uint8 BGR frame directly to FFmpeg stdin (zero copy)."""

        _ = frame_name
        if self._closed:
            raise RuntimeError("streamed VR encoding session is closed")
        if self.cancelled is not None and self.cancelled():
            raise InterruptedError("streamed VR encoding was cancelled")
        if not isinstance(vr_frame, np.ndarray) or vr_frame.dtype != np.uint8:
            raise TypeError("streamed VR frame must be a uint8 numpy array")
        if vr_frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"streamed VR frame shape {vr_frame.shape} differs from "
                f"expected ({self.height}, {self.width}, 3)"
            )
        stdin = self.process.stdin
        if stdin is None:
            raise RuntimeError("FFmpeg stdin pipe is unavailable")
        contiguous = np.ascontiguousarray(vr_frame)
        try:
            stdin.write(contiguous.data)
        except BrokenPipeError as error:
            self._wait_process()
            diagnostics = tuple(self._stderr_lines)
            self.encoder._print_direct_ffmpeg_failure(self.process.returncode or 1, diagnostics)
            raise RuntimeError("FFmpeg closed input pipe unexpectedly") from error
        self._written_frames += 1

    write = write_vr_frame

    def _wait_process(self) -> int:
        if self.process.stdin is not None and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        returncode = self.process.wait()
        self._stderr_thread.join(timeout=5.0)
        self._stdout_thread.join(timeout=5.0)
        return returncode

    def finish_and_publish(self) -> bool:
        """Close stdin, validate the encoded MP4 via FFprobe, and publish atomically."""

        if self._closed:
            raise RuntimeError("streamed VR encoding session is closed")
        try:
            if self._written_frames != self.frame_count:
                raise ValueError(
                    f"streamed VR wrote {self._written_frames} frames, expected {self.frame_count}"
                )
            returncode = self._wait_process()
            diagnostics = tuple(self._stderr_lines)
            if returncode != 0:
                self.encoder._print_direct_ffmpeg_failure(returncode, diagnostics)
                return False
            if not self.temp_output_path.is_file() or self.temp_output_path.stat().st_size == 0:
                self.encoder._print_direct_ffmpeg_output_validation_failure(diagnostics)
                return False
            self.encoder._validate_final_container(
                self.temp_output_path,
                self.settings,
                self.timeline.output_frames(self.frame_count),
                rate_text(self.timeline.output),
                self.resolved_encoder,
                self.selected_audio.usable,
            )
            os.replace(self.temp_output_path, self.final_output_path)
            self._published = True
            return True
        except InterruptedError:
            raise
        except Exception as error:
            print(f"Error creating streamed VR output video: {error}")
            return False
        finally:
            self._cleanup()

    def abort(self) -> None:
        self._cleanup()

    def discard_for_file_backed_fallback(self) -> None:
        self._cleanup()

    def _cleanup(self) -> None:  # noqa: C901
        if self._closed:
            return
        self._closed = True
        if self.process.poll() is None:
            if self.process.stdin is not None and not self.process.stdin.closed:
                try:
                    self.process.stdin.close()
                except OSError:
                    pass
            try:
                self.process.terminate()
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5.0)
            except OSError:
                pass
        self._stderr_thread.join(timeout=2.0)
        self._stdout_thread.join(timeout=2.0)
        if not self._published:
            try:
                self.temp_output_path.unlink(missing_ok=True)
            except OSError:
                pass


class VideoEncoder:
    """Handles FFmpeg frame extraction, hardware/software VR encoding, and atomic publish."""

    def __init__(self, verbose: bool = False, **_ignored: object) -> None:
        self.verbose = verbose

    @staticmethod
    def _cancellation_predicate(progress_tracker: Any) -> Callable[[], bool] | None:
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

    @staticmethod
    def _resolve_output_and_timeline(
        output_dir: Path,
        original_video: str,
        settings: dict[str, Any],
    ) -> tuple[Path, EncodingTimeline]:
        artifact = recover_processing_control_prefix_v1(output_dir)
        if artifact is not None:
            values = artifact.to_object()
            rel = values["output_info"]["expected_final_relative_path"]
            return output_dir / rel, EncodingTimeline.from_artifact(values)
        output_filename = generate_output_filename(
            Path(original_video).name,
            settings["vr_format"],
            settings["vr_resolution"],
        )
        return output_dir / output_filename, VideoEncoder._legacy_timeline(original_video, settings)

    @staticmethod
    def _legacy_timeline(original_video: str, settings: dict[str, Any]) -> EncodingTimeline:
        def optional_rate(value: object) -> Fraction | None:
            try:
                return EncodingTimeline.resolve(value, None).source
            except (ValueError, ZeroDivisionError):
                return None

        video_info = get_video_info_ffprobe(original_video)
        stream: dict[str, Any] = next(
            (s for s in video_info.get("streams", []) if s.get("codec_type") == "video"), {}
        )
        target = optional_rate(settings.get("target_fps"))
        source = optional_rate(probe_frame_rate(stream)) or optional_rate(
            settings.get("source_fps")
        )
        source = source or target or Fraction(DEFAULT_FALLBACK_FPS)
        return EncodingTimeline(source, target or source)

    @staticmethod
    def _temp_mp4_path(final_output_path: Path) -> Path:
        token = secrets.token_hex(4)
        return final_output_path.with_name(f".{final_output_path.stem}.{token}.tmp.mp4")

    @staticmethod
    def _prepare_timeline_audio(
        *,
        output_dir: Path,
        original_video: str,
        settings: dict[str, Any],
        frame_count: int,
        timeline: EncodingTimeline,
    ) -> SelectedAudioIdentityV1:
        source_rate = rate_text(timeline.source)
        kwargs: dict[str, Any] = {}
        if timeline.clip_start_frame is not None:
            kwargs["clip_start_frame"] = timeline.clip_start_frame
        return prepare_selected_audio_v1(
            output_dir=output_dir,
            original_video=original_video,
            settings={**settings, "source_fps": source_rate},
            frame_count=frame_count,
            fps=source_rate,
            **kwargs,
        )

    @staticmethod
    def _validate_final_container(
        path: Path,
        settings: dict[str, Any],
        frame_count: int,
        fps: str,
        resolved_encoder: Literal["libx264", "libx265", "hevc_nvenc"],
        audio_expected: bool,
    ) -> bool:
        width = int(settings["vr_output_width"])
        height = int(settings["vr_output_height"])
        rate = Fraction(fps)
        expectation = FinalContainerExpectation(
            width=width,
            height=height,
            fps=f"{rate.numerator}/{rate.denominator}",
            frame_count=frame_count,
            resolved_encoder=resolved_encoder,
            audio_expected=audio_expected,
        )
        return validate_final_container_v4(path, expectation)

    def open_streamed_vr_session(
        self,
        *,
        output_dir: Path,
        original_video: str,
        settings: dict[str, Any],
        frame_count: int,
        first_index: int = 0,
        progress_tracker: Any = None,
    ) -> StreamedVrEncodingSession:
        """Open a zero-copy raw BGR24 FFmpeg streaming session."""

        _ = first_index
        if not verify_ffmpeg_installation():
            raise RuntimeError("FFmpeg is unavailable for streamed VR encoding")
        root = Path(output_dir).resolve()
        final_output, timeline = self._resolve_output_and_timeline(root, original_video, settings)
        width = int(settings["vr_output_width"])
        height = int(settings["vr_output_height"])
        if width <= 0 or height <= 0:
            raise ValueError("streamed VR output dimensions must be positive")
        cancelled = self._cancellation_predicate(progress_tracker)
        resolved_encoder = self._resolve_encoder_name(
            settings.get("video_encoder", "auto"),
            cancelled=cancelled,
        )
        selected_audio = self._prepare_timeline_audio(
            output_dir=root,
            original_video=original_video,
            settings=settings,
            frame_count=frame_count,
            timeline=timeline,
        )
        temp_output = self._temp_mp4_path(final_output)
        video_filter, limit = timeline.ffmpeg_video_options(
            frame_count,
            drain_audio=selected_audio.usable,
        )
        command = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-xerror",
            "-nostats",
            "-framerate",
            rate_text(timeline.source),
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{width}x{height}",
            "-i",
            "pipe:0",
            *selected_audio.input_arguments(),
            "-vf",
            video_filter,
            "-map",
            "0:v:0",
            *selected_audio.output_arguments(1),
            *limit,
            "-progress",
            "pipe:1",
        ]
        encoder_args, _ = self._build_encoder_cmd(
            settings.get("video_encoder", "auto"),
            temp_output,
            resolved_encoder=resolved_encoder,
        )
        command.extend(encoder_args)
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return StreamedVrEncodingSession(
            encoder=self,
            output_dir=root,
            final_output_path=final_output,
            temp_output_path=temp_output,
            settings=settings,
            frame_count=frame_count,
            width=width,
            height=height,
            timeline=timeline,
            resolved_encoder=resolved_encoder,
            selected_audio=selected_audio,
            process=process,
            cancelled=cancelled,
        )

    def create_video_from_stereo_source(
        self,
        source: DirectStereoSource,
        output_dir: Path,
        original_video: str,
        settings: dict[str, Any],
        *,
        total_frames: int,
        progress_tracker: Any = None,
    ) -> bool:
        """Encode stereo left/right PNG sequences directly into VR MP4."""

        if source.frame_count != total_frames or source.frame_count <= 0:
            print("Error creating direct VR output video: source frame count changed")
            return False
        first_left = source.left_parent / f"frame_{source.first_index:06d}.png"
        first_right = source.right_parent / f"frame_{source.first_index:06d}.png"
        left_header = read_png_header(first_left)
        right_header = read_png_header(first_right)
        if left_header is None or right_header is None:
            print("Error creating direct VR output video: first stereo PNG unreadable")
            return False
        sequence = _DirectStereoSequence(
            left_pattern=source.left_parent / "frame_%06d.png",
            right_pattern=source.right_parent / "frame_%06d.png",
            start_number=source.first_index,
            frame_count=source.frame_count,
            left_header=left_header,
            right_header=right_header,
        )
        return self._encode_direct_sequence(
            sequence=sequence,
            output_dir=Path(output_dir),
            original_video=original_video,
            settings=settings,
            progress_tracker=progress_tracker,
        )

    def create_video_from_stereo_sequences(
        self,
        left_files: Sequence[Path],
        right_files: Sequence[Path],
        output_dir: Path,
        original_video: str,
        settings: dict[str, Any],
        *,
        total_frames: int,
        progress_tracker: Any = None,
    ) -> bool:
        """Validate eye file sequences and encode directly into VR MP4."""

        try:
            sequence = self._validate_direct_stereo_sequence(left_files, right_files, total_frames)
        except Exception as error:
            print(f"Error creating direct VR output video: {error}")
            return False
        return self._encode_direct_sequence(
            sequence=sequence,
            output_dir=Path(output_dir),
            original_video=original_video,
            settings=settings,
            progress_tracker=progress_tracker,
        )

    def _encode_direct_sequence(
        self,
        *,
        sequence: _DirectStereoSequence,
        output_dir: Path,
        original_video: str,
        settings: dict[str, Any],
        progress_tracker: Any = None,
    ) -> bool:
        if not verify_ffmpeg_installation():
            print("Error: FFmpeg not found. Cannot create output video.")
            return False
        cancelled = self._cancellation_predicate(progress_tracker)
        final_output, timeline = self._resolve_output_and_timeline(
            output_dir, original_video, settings
        )
        temp_output = self._temp_mp4_path(final_output)
        try:
            resolved_encoder = self._resolve_encoder_name(
                settings.get("video_encoder", "auto"),
                cancelled=cancelled,
            )
            selected_audio = self._prepare_timeline_audio(
                output_dir=output_dir,
                original_video=original_video,
                settings=settings,
                frame_count=sequence.frame_count,
                timeline=timeline,
            )
            command = self._build_direct_stereo_command(
                sequence=sequence,
                temporary_output=temp_output,
                settings=settings,
                selected_audio=selected_audio,
                timeline=timeline,
                resolved_encoder=resolved_encoder,
            )
            return self._run_and_publish(
                command=command,
                temp_output=temp_output,
                final_output=final_output,
                settings=settings,
                output_frame_count=timeline.output_frames(sequence.frame_count),
                output_fps=rate_text(timeline.output),
                resolved_encoder=resolved_encoder,
                audio_expected=selected_audio.usable,
                progress_tracker=progress_tracker,
                cancelled=cancelled,
            )
        except InterruptedError:
            raise
        except Exception as error:
            print(f"Error creating direct VR output video: {error}")
            return False
        finally:
            temp_output.unlink(missing_ok=True)

    def create_video(
        self,
        vr_frames_dir: Path,
        output_dir: Path,
        original_video: str,
        settings: dict[str, Any],
        progress_tracker: Any = None,
    ) -> bool:
        """Encode pre-assembled VR PNG frames into the final MP4."""

        if not verify_ffmpeg_installation():
            print("Error: FFmpeg not found. Cannot create output video.")
            return False
        vr_frames_dir = Path(vr_frames_dir)
        output_dir = Path(output_dir)
        frame_files = get_frame_files(vr_frames_dir)
        if not frame_files:
            print(f"Error: No VR frames found in {vr_frames_dir}")
            return False

        cancelled = self._cancellation_predicate(progress_tracker)
        final_output, timeline = self._resolve_output_and_timeline(
            output_dir, original_video, settings
        )
        temp_output = self._temp_mp4_path(final_output)
        try:
            first_index = self._direct_frame_index(frame_files[0])
            frame_count = len(frame_files)
            resolved_encoder = self._resolve_encoder_name(
                settings.get("video_encoder", "auto"),
                cancelled=cancelled,
            )
            selected_audio = self._prepare_timeline_audio(
                output_dir=output_dir,
                original_video=original_video,
                settings=settings,
                frame_count=frame_count,
                timeline=timeline,
            )
            video_filter, limit = timeline.ffmpeg_video_options(
                frame_count,
                drain_audio=selected_audio.usable,
            )
            pattern = vr_frames_dir / "frame_%06d.png"
            command = [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-xerror",
                "-nostats",
                "-framerate",
                rate_text(timeline.source),
                "-start_number",
                str(first_index),
                "-i",
                str(pattern),
                *selected_audio.input_arguments(),
                "-vf",
                video_filter,
                "-map",
                "0:v:0",
                *selected_audio.output_arguments(1),
                *limit,
                "-progress",
                "pipe:1",
            ]
            encoder_args, _ = self._build_encoder_cmd(
                settings.get("video_encoder", "auto"),
                temp_output,
                resolved_encoder=resolved_encoder,
            )
            command.extend(encoder_args)
            return self._run_and_publish(
                command=command,
                temp_output=temp_output,
                final_output=final_output,
                settings=settings,
                output_frame_count=timeline.output_frames(frame_count),
                output_fps=rate_text(timeline.output),
                resolved_encoder=resolved_encoder,
                audio_expected=selected_audio.usable,
                progress_tracker=progress_tracker,
                cancelled=cancelled,
            )
        except InterruptedError:
            raise
        except Exception as error:
            print(f"Error creating output video: {error}")
            return False
        finally:
            temp_output.unlink(missing_ok=True)

    def _run_and_publish(
        self,
        *,
        command: list[str],
        temp_output: Path,
        final_output: Path,
        settings: dict[str, Any],
        output_frame_count: int,
        output_fps: str,
        resolved_encoder: Literal["libx264", "libx265", "hevc_nvenc"],
        audio_expected: bool,
        progress_tracker: Any,
        cancelled: Callable[[], bool] | None,
    ) -> bool:
        returncode, diagnostics = self._run_ffmpeg_with_progress(
            command,
            output_frame_count,
            progress_tracker,
            cancelled=cancelled,
        )
        if returncode != 0:
            self._print_direct_ffmpeg_failure(returncode, diagnostics)
            return False
        if not temp_output.is_file() or temp_output.stat().st_size == 0:
            self._print_direct_ffmpeg_output_validation_failure(diagnostics)
            return False
        if "vr_output_width" in settings and "vr_output_height" in settings:
            self._validate_final_container(
                temp_output,
                settings,
                output_frame_count,
                output_fps,
                resolved_encoder,
                audio_expected,
            )
        os.replace(temp_output, final_output)
        return True

    @staticmethod
    def _direct_frame_index(path: Path) -> int:
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
        if not left_files or not right_files:
            raise ValueError("Direct stereo sequences must contain frames for both eyes")
        if len(left_files) != len(right_files):
            raise ValueError("Direct stereo sequences must have identical frame counts")
        frame_count = len(left_files)
        if total_frames > 0 and frame_count != total_frames:
            raise ValueError("Direct stereo sequence count does not match total frames")
        left_parent = Path(left_files[0]).parent
        right_parent = Path(right_files[0]).parent
        seen: set[int] = set()
        for left_raw, right_raw in zip(left_files, right_files):
            left = Path(left_raw)
            right = Path(right_raw)
            if left.parent != left_parent or right.parent != right_parent:
                raise ValueError("Direct stereo frames must share parent directories")
            left_idx = self._direct_frame_index(left)
            right_idx = self._direct_frame_index(right)
            if left_idx != right_idx:
                raise ValueError("Direct stereo sequences must have identical frame names")
            if left_idx in seen:
                raise ValueError("Direct stereo sequence frame indices must be unique")
            seen.add(left_idx)
        minimum = min(seen)
        maximum = max(seen)
        if maximum - minimum + 1 != frame_count:
            raise ValueError("Direct stereo sequence frame indices must be gap-free")

        left_header = read_png_header(left_parent / f"frame_{minimum:06d}.png")
        right_header = read_png_header(right_parent / f"frame_{minimum:06d}.png")
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

    def _build_direct_stereo_command(
        self,
        *,
        sequence: _DirectStereoSequence,
        temporary_output: Path,
        settings: dict[str, Any],
        selected_audio: SelectedAudioIdentityV1,
        timeline: EncodingTimeline,
        resolved_encoder: Literal["libx264", "libx265", "hevc_nvenc"],
    ) -> list[str]:
        per_eye_width = int(settings["per_eye_width"])
        per_eye_height = int(settings["per_eye_height"])
        if per_eye_width <= 0 or per_eye_height <= 0:
            raise ValueError("Direct stereo per-eye dimensions must be positive")
        vr_format = settings.get("vr_format")
        stack_filter = {"side_by_side": "hstack", "over_under": "vstack"}.get(str(vr_format))
        if stack_filter is None:
            raise ValueError(f"Unsupported direct VR format: {vr_format}")

        fps = rate_text(timeline.source)
        start_number = str(sequence.start_number)
        video_filter, limit = timeline.ffmpeg_video_options(
            sequence.frame_count,
            drain_audio=selected_audio.usable,
        )
        left_ok = (
            sequence.left_header.width == per_eye_width
            and sequence.left_header.height == per_eye_height
        )
        right_ok = (
            sequence.right_header.width == per_eye_width
            and sequence.right_header.height == per_eye_height
        )
        if left_ok and right_ok:
            filter_graph = (
                "[0:v]setsar=1[left];[1:v]setsar=1[right];"
                f"[left][right]{stack_filter}=inputs=2:shortest=1[stacked];"
                f"[stacked]{video_filter}[vr]"
            )
        else:
            scale = f"scale={per_eye_width}:{per_eye_height}:flags=bicubic+accurate_rnd,setsar=1"
            filter_graph = (
                f"[0:v]{scale}[left];[1:v]{scale}[right];"
                f"[left][right]{stack_filter}=inputs=2:shortest=1[stacked];"
                f"[stacked]{video_filter}[vr]"
            )
        command = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-xerror",
            "-nostats",
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
            *selected_audio.input_arguments(),
            "-filter_complex",
            filter_graph,
            "-map",
            "[vr]",
            *selected_audio.output_arguments(2),
            *limit,
            "-progress",
            "pipe:1",
        ]
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
        progress_tracker: Any,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[int, tuple[str, ...]]:
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
                if cancelled is not None and cancelled():
                    raise InterruptedError("FFmpeg encoding was cancelled")
                self._consume_direct_ffmpeg_line(
                    raw_line, total_frames, progress_tracker, diagnostics
                )
            return process.wait(), tuple(diagnostics)
        except BaseException:
            process.terminate()
            try:
                remaining, _ = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                remaining, _ = process.communicate()
            for raw_line in (remaining or "").splitlines():
                if not raw_line.startswith("frame="):
                    diagnostics.append(raw_line.strip())
            self._print_direct_ffmpeg_diagnostics(tuple(diagnostics))
            raise

    @staticmethod
    def _consume_direct_ffmpeg_line(
        raw_line: str,
        total_frames: int,
        progress_tracker: Any,
        diagnostics: deque[str],
    ) -> None:
        line = raw_line.strip()
        if not line:
            return
        if not line.startswith("frame="):
            if "=" not in line:
                diagnostics.append(line)
            return
        try:
            frame = min(max(int(line.partition("=")[2]), 0), total_frames)
        except ValueError:
            return
        if progress_tracker is not None:
            try:
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
        if diagnostics:
            print("FFmpeg diagnostic tail:")
            for line in diagnostics:
                print(f"  {line}")

    @staticmethod
    def _print_direct_ffmpeg_failure(returncode: int, diagnostics: tuple[str, ...]) -> None:
        print(f"Error: Direct FFmpeg encoding failed with return code {returncode}.")
        VideoEncoder._print_direct_ffmpeg_diagnostics(diagnostics)

    @staticmethod
    def _print_direct_ffmpeg_output_validation_failure(diagnostics: tuple[str, ...]) -> None:
        print(
            "Error: Direct FFmpeg output validation failed: "
            "temporary output is missing or empty."
        )
        VideoEncoder._print_direct_ffmpeg_diagnostics(diagnostics)

    @staticmethod
    def _build_audio_input_args(
        audio_source: str | Path,
        settings: dict[str, Any],
    ) -> list[str]:
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

    def prepare_streaming_frame_source(
        self,
        video_path: str,
        directories: dict[str, Path],
        video_properties: dict[str, Any],
        settings: dict[str, Any],
        *,
        reuse_index: bool = False,
    ) -> VideoFrameSource | None:
        """Build a zero-PNG streaming VideoFrameSource unless reusable PNGs already exist."""

        if not verify_ffmpeg_installation():
            return None
        frames_dir = directories.get("frames")
        if not frames_dir:
            frames_dir = directories["base"] / INTERMEDIATE_DIRS["frames"]
        if frames_dir.is_dir() and next(frames_dir.glob("frame_*.png"), None) is not None:
            return None
        return build_video_frame_source(
            video_path,
            frames_dir,
            video_properties,
            settings,
            compute_histograms=bool(settings.get("scene_detection", True)),
            reuse_index=reuse_index,
        )

    def extract_frames(
        self,
        video_path: str,
        directories: dict[str, Path],
        video_properties: dict[str, Any],
        settings: dict[str, Any],
    ) -> list[Path]:
        """Extract frames from video using FFmpeg (CUDA with CPU fallback)."""

        frames_dir = directories.get("frames")
        if not frames_dir:
            frames_dir = directories["base"] / INTERMEDIATE_DIRS["frames"]
            frames_dir.mkdir(exist_ok=True)

        fps = video_properties["fps"]
        frame_range = selected_frame_range(video_properties, settings)
        start_frame, end_frame = frame_range
        expected_frames = end_frame - start_frame

        existing_frames = sorted(frames_dir.glob("frame_*.png"))
        if self._extracted_frames_are_reusable(
            directories, video_path, frames_dir, existing_frames, frame_range
        ):
            print(f"  Reusing {len(existing_frames)} already extracted frames")
            return existing_frames

        for frame_file in existing_frames:
            frame_file.unlink(missing_ok=True)
        (frames_dir / "metadata.json").unlink(missing_ok=True)

        if video_properties.get("frame_rate"):
            start_time, duration = ffmpeg_frame_interval(
                start_frame, end_frame, Fraction(video_properties["frame_rate"])
            )
        else:
            start_time = str(start_frame / fps if fps > 0 else 0)
            duration = str(expected_frames / fps if fps > 0 else 0)

        cmd_cuda = [
            "ffmpeg",
            "-y",
            "-hwaccel",
            "cuda",
            "-ss",
            str(start_time),
            "-i",
            video_path,
            "-t",
            str(duration),
            "-pix_fmt",
            "rgb24",
            "-fps_mode",
            "passthrough",
            str(frames_dir / "frame_%06d.png"),
        ]
        try:
            result = subprocess.run(cmd_cuda, capture_output=True, text=True)
            if result.returncode != 0:
                print("  CUDA frame extraction failed, falling back to CPU")
                cmd_cpu = [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(start_time),
                    "-i",
                    video_path,
                    "-t",
                    str(duration),
                    "-pix_fmt",
                    "rgb24",
                    "-fps_mode",
                    "passthrough",
                    "-threads",
                    "0",
                    str(frames_dir / "frame_%06d.png"),
                ]
                result_cpu = subprocess.run(cmd_cpu, capture_output=True, text=True)
                if result_cpu.returncode != 0:
                    print(f"FFmpeg error: {result_cpu.stderr}")
                    return []
        except Exception as error:
            print(f"Error extracting frames: {error}")
            return []

        frame_files = get_frame_files(frames_dir)
        if frame_files:
            write_source_frame_manifest(frame_files, video_path, frame_range)
        return frame_files

    def _extracted_frames_are_reusable(
        self,
        directories: dict[str, Path],
        video_path: str,
        frames_dir: Path,
        existing_frames: list[Path],
        frame_range: tuple[int, int],
    ) -> bool:
        expected_frames = frame_range[1] - frame_range[0]
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
                    frame_range,
                )
                is None
            )
        return expected_frames > 0 and (exact_count or tolerated_vfr_drift) and reusable_manifest

    @staticmethod
    def _has_downstream_progress(directories: dict[str, Path]) -> bool:
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
        if cancelled is not None and cancelled():
            raise InterruptedError("encoder probe was cancelled")
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=size=256x256:rate=1",
                    "-frames:v",
                    "1",
                    "-c:v",
                    "hevc_nvenc",
                    "-f",
                    "hevc",
                    "pipe:1",
                ],
                capture_output=True,
                timeout=10.0,
                check=False,
            )
        except (subprocess.SubprocessError, OSError):
            return False
        return result.returncode == 0 and not result.stderr and bool(result.stdout)

    def _resolve_encoder_name(
        self,
        encoder: str,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> Literal["libx264", "libx265", "hevc_nvenc"]:
        if encoder in {"auto", "nvenc"}:
            if self._check_nvenc_available(cancelled=cancelled):
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
        codec = resolved_encoder or self._resolve_encoder_name(encoder, cancelled=cancelled)
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
                "18",
                "-preset",
                "medium",
                str(output_path),
            ],
            False,
        )
