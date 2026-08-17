"""
Video encoding module.

Handles FFmpeg-based video encoding with hardware acceleration support (NVENC).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
import re
import subprocess
from pathlib import Path
from typing import Any

from ...io.operations import (
    get_frame_files,
    get_video_info_ffprobe,
    verify_ffmpeg_installation,
)
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


class VideoEncoder:
    """
    Handles video encoding using FFmpeg.

    Responsibilities:
    - Video creation with FFmpeg
    - Hardware encoder detection (NVENC)
    - Frame extraction from videos
    - Audio integration
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize video encoder.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose

    def create_video(
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

        # Generate output filename
        output_filename = generate_output_filename(
            Path(original_video).name, settings["vr_format"], settings["vr_resolution"]
        )
        output_path = output_dir / output_filename

        # Build base FFmpeg command
        base_fps = self._resolve_output_fps(original_video, settings)

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(base_fps),
            "-i",
            str(vr_frames_dir / "frame_%06d.png"),
        ]

        # Add audio if preserving - use pre-extracted FLAC file
        if settings.get("preserve_audio", True):
            audio_file = output_dir / "original_audio.flac"
            print(f"Looking for pre-extracted audio at: {audio_file}")
            if audio_file.exists():
                print(f"Using pre-extracted audio: {audio_file}")
                audio_source: str | Path = audio_file
            else:
                print(f"Warning: Pre-extracted audio not found at {audio_file}")
                print(f"Extracting audio from original video: {original_video}")
                audio_source = original_video
            cmd.extend(self._build_audio_input_args(audio_source, settings))
            cmd.extend(["-c:a", "aac", "-shortest"])

        # Add video encoding settings
        encoder = settings.get("video_encoder", "auto")
        encoder_args, _ = self._build_encoder_cmd(encoder, output_path)
        cmd.extend(encoder_args)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                return False
            return True
        except Exception as e:
            print(f"Error creating output video: {e}")
            return False

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

    def _build_direct_stereo_command(
        self,
        sequence: _DirectStereoSequence,
        temporary_output: Path,
        original_video: str,
        settings: dict[str, Any],
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
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
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
        ]

        preserve_audio = settings.get("preserve_audio", True)
        if preserve_audio:
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
        if preserve_audio:
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

    def create_video_from_stereo_sequences(
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

        output_filename = generate_output_filename(
            Path(original_video).name,
            settings["vr_format"],
            settings["vr_resolution"],
        )
        output_path = output_dir / output_filename
        temporary = output_path.with_name(f".{output_path.stem}.direct.tmp.mp4")
        try:
            temporary.unlink(missing_ok=True)
            if not verify_ffmpeg_installation():
                print("Error: FFmpeg not found. Cannot create output video.")
                return False
            sequence = self._validate_direct_stereo_sequence(
                list(left_files), list(right_files), total_frames
            )
            command = self._build_direct_stereo_command(
                sequence, temporary, original_video, settings
            )
            returncode, diagnostics = self._run_ffmpeg_with_progress(
                command, sequence.frame_count, progress_tracker
            )
            if returncode != 0:
                self._print_direct_ffmpeg_failure(returncode, diagnostics)
                return False
            if not temporary.is_file() or temporary.stat().st_size == 0:
                self._print_direct_ffmpeg_output_validation_failure(diagnostics)
                return False
            temporary.replace(output_path)
            return True
        except InterruptedError:
            raise
        except Exception as error:
            print(f"Error creating direct VR output video: {error}")
            return False
        finally:
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
