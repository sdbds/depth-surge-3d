#!/usr/bin/env python3
"""
Depth Surge 3D Web UI
Flask application for converting 2D videos to immersive 3D VR format
"""

from __future__ import annotations

import os
import time
import uuid
import subprocess
import platform
import argparse
import sys
import signal
import base64
import shutil
import stat
import tempfile
from dataclasses import asdict
import json
from datetime import datetime
from ipaddress import ip_address
from pathlib import Path, PureWindowsPath
from threading import RLock, current_thread
from typing import Any
import cv2
import numpy as np

# Use one package namespace so module globals and caches are never duplicated.
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set PyTorch memory allocator config BEFORE importing torch
# This helps prevent memory fragmentation on GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Suppress warnings from dependencies
import warnings  # noqa: E402

warnings.filterwarnings("ignore", category=SyntaxWarning)  # moviepy old regex patterns

# Import our constants and utilities
from depth_surge_3d.core.constants import (  # noqa: E402
    DEFAULT_SETTINGS,
    MODEL_PATHS,
    MODEL_PATHS_METRIC,
    INTERMEDIATE_DIRS,
    SOCKETIO_PING_TIMEOUT,
    SOCKETIO_PING_INTERVAL,
    SOCKETIO_SLEEP_YIELD,
    INITIAL_PROCESSING_DELAY,
    BYTES_TO_GB_DIVISOR,
    FPS_ROUND_DIGITS,
    DURATION_ROUND_DIGITS,
    ASPECT_RATIO_ROUND_DIGITS,
    ASPECT_RATIO_SBS_THRESHOLD,
    ASPECT_RATIO_OU_THRESHOLD,
    SESSION_ID_DISPLAY_LENGTH,
    PROGRESS_UPDATE_INTERVAL,
    PROGRESS_DECIMAL_PLACES,
    PROGRESS_STEP_WEIGHTS,
    VDPP_PROGRESS_STEP_WEIGHTS,
    PREVIEW_UPDATE_INTERVAL,
    PREVIEW_DOWNSCALE_WIDTH,
    PREVIEW_JPEG_QUALITY,
    SUPPORTED_VIDEO_FORMATS,
    FFMPEG_CRF_HIGH_QUALITY,
    FFMPEG_CRF_MEDIUM_QUALITY,
    FFMPEG_CRF_FAST_QUALITY,
    FFMPEG_DEFAULT_PRESET,
    FFMPEG_PIX_FORMAT,
    VIDEO_CREATION_TIMEOUT,
    DEFAULT_FALLBACK_FPS,
    DEFAULT_SERVER_PORT,
    DEFAULT_SERVER_HOST,
    SIGNAL_SHUTDOWN_TIMEOUT,
)
from depth_surge_3d.utils.system.console import warning as console_warning  # noqa: E402
from depth_surge_3d.inference.depth.backend_registry import (  # noqa: E402
    TEMPORAL_STABILITY_WARNING,
    backend_availability,
    build_effective_depth_run_report,
    get_backend_spec,
    list_backend_specs,
    normalize_model_size,
    validate_backend_geometry_request,
)
from depth_surge_3d.core.settings import (  # noqa: E402
    parse_saved_processing_settings,
    resolve_stereo_render_mode,
    resolve_temporal_postprocessor,
    validate_settings,
)
from depth_surge_3d.io.operations import (  # noqa: E402
    find_settings_file,
    get_video_info_ffprobe,
    get_video_properties,
    load_processing_settings,
)
from depth_surge_3d.core.video_timing import (  # noqa: E402
    MOTION_INTERPOLATION_UNAVAILABLE,
    EncodingTimeline,
    canonical_frame_rate,
    probe_frame_rate,
    rate_text,
    selected_frame_range,
)
from depth_surge_3d.utils.path_utils import (  # noqa: E402
    calculate_start_frame,
    ffmpeg_frame_interval,
)
from depth_surge_3d.io.settings_artifact import validate_portable_final_component_v1  # noqa: E402
from depth_surge_3d.utils.domain.resolution import resolve_depth_input_size  # noqa: E402

from flask import Flask, render_template, request, jsonify  # noqa: E402
from flask_socketio import SocketIO  # noqa: E402
from werkzeug.wrappers import Request, Response  # noqa: E402
from socketio.exceptions import (  # type: ignore[import-untyped]  # noqa: E402
    TimeoutError as SocketIOTimeoutError,
)

# NOTE: torch is imported later (line ~960) to avoid CUDA initialization issues

from depth_surge_3d.processing import VideoProcessor  # noqa: E402
from depth_surge_3d.io.resume import (  # noqa: E402
    apply_legacy_migration,
    build_resume_report,
    resolve_resume_depth_model_version,
    resolve_resume_source_video,
)
from depth_surge_3d.io.job_lock import JobAlreadyLockedError, JobWriterLock  # noqa: E402
from depth_surge_3d.vdpp_work import cleanup_vdpp_private_work  # noqa: E402
from depth_surge_3d.processing.frames.depth_storage import (  # noqa: E402
    build_current_model_fingerprint,
)
from depth_surge_3d.processing.orchestration.execution_plan import (  # noqa: E402
    build_artifact_execution_plan,
)

# Global flags and state
VERBOSE = False
SHUTDOWN_FLAG = False
ACTIVE_PROCESSES: set[subprocess.Popen] = set()
_MEDIA_PROCESS_LOCK = RLock()
PROCESSING_CONFIGURATION_ACK_TIMEOUT = 5.0


def create_stereo_projector(*args, **kwargs):
    """Import depth backends only after artifact planning requires them."""

    from depth_surge_3d.rendering import create_stereo_projector as factory

    return factory(*args, **kwargs)


def _preserved_render_artifact(report, mode: str) -> tuple[list[Path], str] | None:  # noqa: C901
    stage_name = "disparity_stabilized" if mode == "vdpp" else "disparity_maps"
    try:
        stage = report.stage(stage_name)
    except KeyError:
        return None
    if stage.disposition != "preserve":
        if mode != "vdpp":
            return None
        try:
            stage = report.stage("disparity_maps")
        except KeyError:
            return None
        if stage.disposition != "preserve":
            return None
        stage_name = "disparity_maps"
    directory = stage.paths[0]
    try:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    if stage_name == "disparity_stabilized":
        semantic = metadata.get("semantic_identity", {})
        names = semantic.get("frame_names") if isinstance(semantic, dict) else None
        source = "stabilized"
    else:
        names = metadata.get("frame_names")
        source = "base"
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        return None
    return [directory / f"{Path(name).stem}.png" for name in names], source


def _validate_web_settings(settings: dict[str, Any], *, source: str) -> dict[str, Any]:
    """Validate Web settings while preserving null as source-frame FPS."""
    if not isinstance(settings, dict):
        raise TypeError("settings values must be a dictionary")
    preserve_source_fps = settings.get("target_fps", object()) is None
    values = dict(settings)
    if preserve_source_fps:
        values.pop("target_fps")
    validated = validate_settings(values, source=source)
    if preserve_source_fps:
        validated["target_fps"] = None
    return validated


def _effective_web_renderer_device(settings: dict[str, Any]) -> str:
    requested = str(settings.get("device", "auto"))
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_new_web_stereo_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Resolve omitted new-job controls without changing library defaults."""

    values = dict(settings)
    if values.get("experimental_frame_interpolation") is True:
        raise ValueError(MOTION_INTERPOLATION_UNAVAILABLE)
    values["stereo_render_mode"] = resolve_stereo_render_mode(
        persisted=None,
        override=values.get("stereo_render_mode"),
        is_resume=False,
        renderer_device_type=_effective_web_renderer_device(values),
    )
    values.setdefault("occlusion_fill_max_px", DEFAULT_SETTINGS["occlusion_fill_max_px"])
    return values


def _resolve_resume_web_stereo_settings(
    persisted: dict[str, Any],
    *,
    stereo_override: str | None,
    fill_limit_override: int | None,
) -> dict[str, Any]:
    """Preserve schema-5 stereo identity and give legacy jobs Fast/8."""

    values = dict(persisted)
    values["stereo_render_mode"] = resolve_stereo_render_mode(
        persisted=values.get("stereo_render_mode", "fast"),
        override=stereo_override,
        is_resume=True,
        renderer_device_type=_effective_web_renderer_device(values),
    )
    if fill_limit_override is not None:
        values["occlusion_fill_max_px"] = fill_limit_override
    else:
        values.setdefault("occlusion_fill_max_px", DEFAULT_SETTINGS["occlusion_fill_max_px"])
    return values


def _active_requester_socket_id(data: Any) -> str:
    """Return the connected Socket.IO client that owns an HTTP run request."""
    socket_id = data.get("socket_id") if isinstance(data, dict) else None
    if not isinstance(socket_id, str) or not socketio.server.manager.is_connected(socket_id, "/"):
        raise ValueError("A connected Socket.IO client is required")
    return socket_id


def _normalize_depth_backend_settings(
    settings: dict[str, Any], *, default_backend: str = "v3"
) -> dict[str, Any]:
    """Resolve product-facing depth selections into the shared settings schema."""
    normalized = dict(settings)
    backend_id = normalized.get("depth_model_version", default_backend)
    spec = get_backend_spec(backend_id)
    model_path = normalized.get("model_path")
    normalized["depth_model_version"] = backend_id
    normalized["model_size"] = normalize_model_size(
        backend_id,
        model_path=model_path,
        model_size=normalized.get("model_size"),
    )
    normalized["model_path"] = model_path
    normalized["depth_resolution"] = normalized.get("depth_resolution", "auto")
    requested_metric = bool(normalized.get("use_metric_depth", False))
    normalized["use_metric_depth"] = (
        True if backend_id == "moge2" else requested_metric and spec.capabilities.metric_depth
    )
    return normalized


def _normalize_available_depth_backend_settings(settings: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_depth_backend_settings(settings)
    availability = backend_availability(normalized["depth_model_version"])
    if availability.available:
        return normalized
    error = (
        availability.reason or f"Depth backend {normalized['depth_model_version']} is unavailable"
    )
    if availability.install_command:
        error = f"{error}. Install with: {availability.install_command}"
    raise ValueError(error)


def _check_web_model_selection(settings: dict[str, Any]) -> None:
    """The desktop UI selects installed presets, never a client-specified checkpoint."""
    model_path = settings.get("model_path")
    if model_path is None:
        return
    spec = get_backend_spec(settings.get("depth_model_version", "v3"))
    allowed = {value for v in spec.variants.values() for value in (v.backend_value, v.repo_id)}
    if spec.backend_id == "v2":
        allowed.update(MODEL_PATHS.values())
        allowed.update(MODEL_PATHS_METRIC.values())
    if not isinstance(model_path, str) or model_path not in allowed:
        raise _WebRequestError("Custom model_path is only supported by the CLI", 400)


def _depth_backend_options() -> list[dict[str, Any]]:
    options = []
    for spec in list_backend_specs():
        availability = backend_availability(spec.backend_id)
        options.append(
            {
                "id": spec.backend_id,
                "label": spec.display_name,
                "default_model_size": spec.default_model_size,
                "variants": [asdict(value) for value in spec.variants.values()],
                "capabilities": {
                    "metric_depth": spec.capabilities.metric_depth,
                    "pinhole_fx": spec.capabilities.pinhole_fx,
                    "stereo_geometry_modes": sorted(spec.capabilities.stereo_geometry_modes),
                },
                "availability": asdict(availability),
            }
        )
    return options


def vprint(*args: Any, **kwargs: Any) -> None:
    """Print only if verbose mode is enabled"""
    if VERBOSE:
        print(*args, **kwargs)


def _get_version_info() -> tuple[str, str]:
    """Get version and git commit ID"""
    try:
        from depth_surge_3d import __version__

        version = __version__
    except ImportError:
        version = "dev"

    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=1,
        ).stdout.strip()
        if not git_commit:
            git_commit = "unknown"
    except Exception:
        git_commit = "unknown"

    return version, git_commit


def _rgb_to_ansi256(r: int, g: int, b: int) -> int:
    """Convert RGB (0-5 range) to ANSI 256 color code"""
    return 16 + 36 * r + 6 * g + b


def _get_lime_color(position: float) -> str:
    """Lerp between dark green and brand lime (#39ff14) for smooth gradient"""
    # Brand lime color from CSS: #39ff14 = RGB(57, 255, 20)
    # Converted to ANSI 0-5 scale: (1, 5, 0)
    # Create gradient from dark green to brand lime
    start_rgb = (0, 3, 0)  # Dark green
    end_rgb = (1, 5, 0)  # Brand lime (#39ff14)

    # Lerp each RGB component with rounding for smoothness
    r = round(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * position)
    g = round(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * position)
    b = round(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * position)

    # Clamp to valid range (0-5)
    r = max(0, min(5, r))
    g = max(0, min(5, g))
    b = max(0, min(5, b))

    # Convert to ANSI 256 color code
    color_code = _rgb_to_ansi256(r, g, b)
    return f"\033[38;5;{color_code}m"


def _print_banner_border(border_width: int, char: str, row_offset: int, total_rows: int) -> None:
    """Print border with diagonal gradient matching text"""
    margin = 0.0  # No margin - let gradient span full banner
    max_diagonal = total_rows + border_width

    border = " " + _get_lime_color(0.0 - margin) + "█"  # Leading space
    for i in range(border_width):
        # Calculate diagonal position with stretched gradient
        raw_pos = (row_offset + i) / max_diagonal
        stretched_pos = (raw_pos * (1 + 2 * margin)) - margin
        border += f"{_get_lime_color(stretched_pos)}{char}"
    border += _get_lime_color(1.0 + margin) + "█\033[0m"
    print(border)


def _print_banner_line(line: str, row_idx: int, num_rows: int, max_diagonal: int) -> None:
    """Print single banner line with diagonal gradient"""
    reset = "\033[0m"
    margin = 0.0  # No margin - let gradient span full banner

    # Left border with leading space
    left_pos = row_idx / (num_rows + 1)
    colored_line = f" {_get_lime_color(left_pos)}█{reset} "  # Leading space

    # Apply diagonal gradient to entire text
    for col_idx, char in enumerate(line):
        raw_pos = (row_idx + col_idx) / max_diagonal
        stretched_pos = (raw_pos * (1 + 2 * margin)) - margin
        colored_line += f"{_get_lime_color(stretched_pos)}{char}"

    # Right border
    right_pos = (row_idx + 1) / (num_rows + 1)
    colored_line += f"{reset} {_get_lime_color(right_pos)}█{reset}"
    print(colored_line)


def print_banner() -> None:
    """Print Depth Surge 3D banner with diagonal lime gradient"""
    version, git_commit = _get_version_info()

    blue_accent = "\033[38;5;39m"
    bright_blue = "\033[38;5;51m"
    reset = "\033[0m"

    banner_lines = [
        "░█▀▄░█▀▀░█▀█░▀█▀░█░█░░░█▀▀░█░█░█▀▄░█▀▀░█▀▀░░░▀▀█░█▀▄░",
        "░█░█░█▀▀░█▀▀░░█░░█▀█░░░▀▀█░█░█░█▀▄░█░█░█▀▀░░░░▀▄░█░█░",
        "░▀▀░░▀▀▀░▀░░░░▀░░▀░▀░░░▀▀▀░▀▀▀░▀░▀░▀▀▀░▀▀▀░░░▀▀░░▀▀░░",
    ]

    print()

    # Calculate dimensions
    border_width = len(banner_lines[0]) + 2
    num_rows = len(banner_lines)
    line_length = len(banner_lines[0])
    max_diagonal = num_rows + line_length - 2

    # Print banner with borders (diagonal gradient on borders too)
    total_rows = num_rows + 2  # +2 for top and bottom borders
    _print_banner_border(border_width, "▀", row_offset=0, total_rows=total_rows)
    for row_idx, line in enumerate(banner_lines):
        _print_banner_line(
            line, row_idx + 1, num_rows, max_diagonal
        )  # +1 to account for top border
    _print_banner_border(border_width, "▄", row_offset=num_rows + 1, total_rows=total_rows)

    # GitHub repo link (with leading space for alignment)
    repo_link = "https://github.com/Tok/depth-surge-3d"
    padding = (border_width - len(repo_link)) // 2 + 1  # +1 for leading space
    print(f"{' ' * padding}{blue_accent}{repo_link}{reset}")

    # Version and commit info (with leading space for alignment)
    version_info = f"v{version} [{git_commit}]"
    version_padding = (border_width - len(version_info)) // 2 + 1  # +1 for leading space
    print(
        f"{' ' * version_padding}v{bright_blue}{version}{reset} [{bright_blue}{git_commit}{reset}]"
    )

    print()


def _run_web_media(command: list[str], *, timeout: float) -> subprocess.CompletedProcess:
    """Keep a handle to this server's child until its output is drained."""
    with _MEDIA_PROCESS_LOCK:
        if SHUTDOWN_FLAG:
            raise RuntimeError("The desktop UI is shutting down")
        proc = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
        )
        ACTIVE_PROCESSES.add(proc)
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except BaseException:
            proc.kill()
            proc.communicate()
            raise
        return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)
    finally:
        with _MEDIA_PROCESS_LOCK:
            ACTIVE_PROCESSES.discard(proc)


def cleanup_processes() -> None:
    """Stop only media children owned by this server, on explicit shutdown."""
    global SHUTDOWN_FLAG
    with _MEDIA_PROCESS_LOCK:
        SHUTDOWN_FLAG = True
        children = tuple(ACTIVE_PROCESSES)
    for child in children:
        try:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=SIGNAL_SHUTDOWN_TIMEOUT)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=SIGNAL_SHUTDOWN_TIMEOUT)
        except (OSError, subprocess.TimeoutExpired) as exc:
            vprint(f"Could not stop a desktop media process: {exc}")


def signal_handler(signum: int, frame: Any) -> None:
    """Handle shutdown signals"""
    print(f"\nReceived signal {signum}, shutting down gracefully...")

    # Stop any active processing
    with _PROCESSING_LOCK:
        active = current_processing["active"]
        worker = current_processing["thread"]
        if active:
            current_processing["stop_requested"] = True
    if active:
        print("   Stopping active video processing...")
        if worker is not None and worker is not current_thread() and worker.is_alive():
            worker.join(timeout=SIGNAL_SHUTDOWN_TIMEOUT)

    # Clean up all processes
    cleanup_processes()
    sys.exit(0)


def _desktop_transport(application):
    """Apply the same local peer, Host and Origin checks before HTTP or Socket.IO."""

    def serve(environ, start_response):
        incoming = Request(environ, shallow=True)
        try:
            local_peer = ip_address(incoming.remote_addr or "").is_loopback
        except ValueError:
            local_peer = False
        scheme = incoming.scheme
        port = environ.get("SERVER_PORT")
        suffix = "" if (scheme, port) in {("http", "80"), ("https", "443")} else f":{port}"
        host = incoming.host.lower()
        allowed_hosts = {name + suffix for name in ("localhost", "127.0.0.1", "[::1]")}
        origin = incoming.headers.get("Origin")
        sec_fetch_site = incoming.headers.get("Sec-Fetch-Site")
        cross_site_request = sec_fetch_site == "cross-site" and (
            incoming.method == "POST"
            or incoming.path in {"/detect_resumable", "/system_info", "/status"}
        )
        if (
            not local_peer
            or host not in allowed_hosts
            or (origin is not None and origin.lower() != f"{scheme}://{host}")
            or cross_site_request
        ):
            denied = Response("Only local, same-origin desktop requests are accepted.", status=403)
            return denied(environ, start_response)

        def start_response_with_headers(status, headers, exc_info=None):
            existing = {name.lower() for name, _ in headers}
            extra = [
                ("X-Frame-Options", "DENY"),
                ("X-Content-Type-Options", "nosniff"),
                ("Content-Security-Policy", "frame-ancestors 'none'"),
            ]
            merged = list(headers) + [
                (name, value) for name, value in extra if name.lower() not in existing
            ]
            return start_response(status, merged, exc_info)

        return application(environ, start_response_with_headers)

    return serve


app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(32)
app.config["OUTPUT_FOLDER"] = str(Path("output").resolve())
# Use threading async_mode and disable ping timeout for long-running tasks
socketio = SocketIO(
    app,
    logger=False,
    engineio_logger=False,
    async_mode="threading",
    ping_timeout=SOCKETIO_PING_TIMEOUT,
    ping_interval=SOCKETIO_PING_INTERVAL,
)
app.wsgi_app = _desktop_transport(app.wsgi_app)


# Global variables for processing state
_INITIAL_JOB_STATE = {
    "active": False,
    "progress": 0,
    "stage": "",
    "total_frames": 0,
    "current_frame": 0,
    "session_id": None,
    "stop_requested": False,
    "thread": None,
}
_PROCESSING_LOCK = RLock()
current_processing: dict[str, Any] = dict(_INITIAL_JOB_STATE)
_PUBLIC_JOB_FIELDS = (
    "active",
    "session_id",
    "stop_requested",
    "progress",
    "stage",
    "current_frame",
    "total_frames",
    "phase",
    "step_name",
    "step_progress",
    "step_total",
    "step_index",
    "total_steps",
    "eta",
)


def _processing_status() -> dict[str, Any]:
    with _PROCESSING_LOCK:
        return {
            key: current_processing[key] for key in _PUBLIC_JOB_FIELDS if key in current_processing
        }


def _release_processing_job(session_id: str) -> None:
    with _PROCESSING_LOCK:
        if current_processing["session_id"] == session_id:
            current_processing.update(
                active=False, session_id=None, stop_requested=False, thread=None
            )


def _run_claimed_job(session_id: str, *args: Any) -> None:
    with _PROCESSING_LOCK:
        if current_processing["session_id"] != session_id:
            return
        current_processing["thread"] = current_thread()
    try:
        process_video_async(session_id, *args)
    finally:
        _release_processing_job(session_id)


def _start_processing_job(*args: Any) -> str:
    session_id = str(uuid.uuid4())
    with _PROCESSING_LOCK:
        if current_processing["active"]:
            raise _WebRequestError("Processing already in progress", 400)
        current_processing.clear()
        current_processing.update(_INITIAL_JOB_STATE, active=True, session_id=session_id)
    try:
        worker = socketio.start_background_task(_run_claimed_job, session_id, *args)
    except BaseException:
        with _PROCESSING_LOCK:
            if current_processing["session_id"] == session_id:
                if current_processing["thread"] is None:
                    _release_processing_job(session_id)
                else:
                    # The launcher may have started its worker before reporting failure.
                    current_processing["stop_requested"] = True
        raise
    with _PROCESSING_LOCK:
        if current_processing["session_id"] == session_id:
            current_processing["thread"] = worker
    return session_id


def _active_session_snapshot(session_id: object) -> dict[str, Any] | None:
    """Return the issuing job's state when joining its canonical UUID room."""
    if not isinstance(session_id, str):
        return None
    try:
        parsed = uuid.UUID(session_id)
    except ValueError:
        return None
    if parsed.version != 4 or str(parsed) != session_id:
        return None
    status = _processing_status()
    return status if status["active"] and status["session_id"] == session_id else None


def _acknowledge_processing_configuration(requester_socket_id: str, report: dict[str, Any]) -> None:
    """Deliver configuration privately and require a bounded positive acknowledgement."""
    if not socketio.server.manager.is_connected(requester_socket_id, "/"):
        raise RuntimeError("Processing configuration requester disconnected")
    try:
        acknowledgement = socketio.call(
            "processing_configuration",
            report,
            to=requester_socket_id,
            timeout=PROCESSING_CONFIGURATION_ACK_TIMEOUT,
        )
    except SocketIOTimeoutError as error:
        raise RuntimeError("Processing configuration acknowledgement timed out") from error
    if (
        type(acknowledgement) is not dict
        or set(acknowledgement) != {"accepted"}
        or acknowledgement["accepted"] is not True
    ):
        raise RuntimeError("Processing configuration was not positively acknowledged")


def ensure_directories() -> None:
    """Ensure output directory exists"""
    Path(app.config["OUTPUT_FOLDER"]).mkdir(parents=True, exist_ok=True)


class _WebRequestError(ValueError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@app.errorhandler(_WebRequestError)
def _invalid_web_request(error):
    return jsonify({"success": False, "error": str(error)}), error.status_code


def _json_object() -> dict[str, Any]:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise _WebRequestError("Request body must be a JSON object", 400)
    return data


def _web_video_filename(value: object) -> str:
    try:
        name = validate_portable_final_component_v1(value)
    except ValueError as exc:
        raise _WebRequestError("Use a video filename without a directory path", 400) from exc
    if (
        PureWindowsPath(name).is_reserved()
        or Path(name).suffix.lower() not in SUPPORTED_VIDEO_FORMATS
    ):
        raise _WebRequestError("Use a filename with a supported video extension", 400)
    return name


def _is_reparse_point(info: os.stat_result) -> bool:
    return bool(getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _check_job_contents(job: Path) -> None:
    """Independent readers and encoders must not be handed a redirected job tree."""
    directories = [os.fspath(job)]
    while directories:
        with os.scandir(directories.pop()) as entries:
            for entry in entries:
                if entry.is_symlink() or (
                    os.name == "nt" and _is_reparse_point(entry.stat(follow_symlinks=False))
                ):
                    raise _WebRequestError("Web jobs cannot contain linked files or folders", 403)
                if entry.is_dir(follow_symlinks=False):
                    directories.append(entry.path)
                elif not entry.is_file(follow_symlinks=False):
                    raise _WebRequestError(
                        "Web jobs can contain only regular files and folders", 403
                    )


def _managed_job_directory(value: object, *, inspect_contents: bool = True) -> Path:
    """Accept existing, immediate job directories, not the collection or its aliases."""
    if not isinstance(value, (str, Path)) or not str(value).strip() or "\x00" in str(value):
        raise _WebRequestError("A job directory path is required", 400)
    configured_root = Path(os.path.abspath(app.config["OUTPUT_FOLDER"]))
    root = configured_root.resolve()
    roots = (configured_root, root)
    supplied = Path(value)
    job = Path(os.path.abspath(supplied))
    if ".." in supplied.parts or job in roots or job.parent not in roots:
        raise _WebRequestError("Select a job directly inside the managed output folder", 403)
    job = root / job.name
    try:
        info = job.lstat()
        if stat.S_ISLNK(info.st_mode) or _is_reparse_point(info):
            raise _WebRequestError("Linked job directories are not supported", 403)
        if not stat.S_ISDIR(info.st_mode):
            raise _WebRequestError("Select a job directory, not a file", 400)
        if inspect_contents:
            _check_job_contents(job)
    except FileNotFoundError as exc:
        raise _WebRequestError("Job directory or one of its files no longer exists", 404) from exc
    except OSError as exc:
        raise _WebRequestError("Could not inspect the managed job directory", 403) from exc
    return job


def _load_incomplete_web_settings(settings_file: Path | None) -> dict[str, Any] | None:
    if settings_file is None:
        return None
    saved = load_processing_settings(settings_file)
    if not isinstance(saved, dict) or not isinstance(saved.get("metadata", {}), dict):
        raise _WebRequestError("Could not read resume metadata", 409)
    if saved.get("metadata", {}).get("processing_status") == "completed":
        raise _WebRequestError("This job is already completed. Start a new job instead.", 409)
    return saved


def get_video_info(video_path: str | Path) -> dict[str, Any] | None:
    """Extract video information using OpenCV"""
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        # Calculate aspect ratio
        aspect_ratio = width / height if height > 0 else 1.0

        return {
            "width": width,
            "height": height,
            "fps": round(fps, FPS_ROUND_DIGITS),
            "duration": round(duration, DURATION_ROUND_DIGITS),
            "frame_count": frame_count,
            "aspect_ratio": round(aspect_ratio, ASPECT_RATIO_ROUND_DIGITS),
            "aspect_ratio_text": f"{width}:{height}",
        }
    finally:
        cap.release()


def find_source_video(directory: Path) -> Path | None:
    """
    Find the source video file in an output directory.

    Looks for video files excluding processed outputs (those with _3D_ in name).

    Args:
        directory: Directory to search

    Returns:
        Path to source video, or None if not found
    """
    files = list(directory.iterdir())
    for ext in SUPPORTED_VIDEO_FORMATS:
        for video_file in files:
            if (
                video_file.suffix.lower() == ext
                and "_3D_" not in video_file.name
                and video_file.is_file()
            ):
                return video_file

    return None


def get_system_info() -> dict[str, Any]:
    """Get system information including GPU details"""
    import torch  # Import here to avoid early CUDA initialization

    info = {
        "gpu_device": "CPU",
        "vram_usage": "N/A",
        "device_mode": "CPU",
        "cuda_available": False,
    }

    try:
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["gpu_device"] = torch.cuda.get_device_name(0)
            info["device_mode"] = "GPU"

            # Get VRAM usage
            try:
                allocated = torch.cuda.memory_allocated() / BYTES_TO_GB_DIVISOR
                total_memory = (
                    torch.cuda.get_device_properties(0).total_memory / BYTES_TO_GB_DIVISOR
                )
                info["vram_usage"] = f"{allocated:.1f}GB / {total_memory:.1f}GB"
            except Exception as vram_error:
                vprint(f"Error getting VRAM details: {vram_error}")
                info["vram_usage"] = "N/A"
    except Exception as e:
        vprint(f"Error getting GPU info: {e}")

    return info


class ProgressCallback:
    """Enhanced callback class for frame and stage progress."""

    def __init__(
        self,
        session_id: str,
        total_frames: int,
        enable_live_preview: bool = True,
        preview_update_interval: float = PREVIEW_UPDATE_INTERVAL,
        temporal_postprocessor: str = "off",
        settings: dict[str, Any] | None = None,
    ) -> None:
        self.session_id = session_id
        with _PROCESSING_LOCK:
            self._owned_job = current_processing["session_id"] == session_id
        self.total_frames = total_frames
        self.current_frame = 0
        self.last_update_time = 0
        self.current_phase = "extraction"  # extraction, processing, video
        self.step_start_times = {}  # Track start time for each step
        self.current_step_name = None
        self.start_time = time.time()  # For ETA calculation
        self.step_frame_times = []  # Track recent frame times for current step
        self.step_start_progress = 0  # Track progress when step started
        self._max_progress = 0.0

        # Preview tracking
        self.enable_live_preview = enable_live_preview
        self.last_preview_time = 0
        self.preview_interval = preview_update_interval
        self.preview_downscale_width = PREVIEW_DOWNSCALE_WIDTH

        # Step tracking (used for all modes)
        # Note: These must match the step names sent from video_processor.py
        self.steps = [
            "Frame Extraction",  # Step 1: FFmpeg extracts frames
            "Depth Map Generation",  # Step 2: AI generates depth maps
            "Stereo Pair Creation",  # Step 3: Create L/R stereo pairs (includes frame loading)
            "Fisheye Distortion",  # Step 4: Apply distortion (optional)
            "Crop Frames",  # Step 5: Crop frames for VR
            "AI Upscaling",  # Step 6: AI upscaling (optional)
            "VR Assembly",  # Step 7: Assemble final VR frames
            "Video Creation",  # Step 8: FFmpeg creates video
        ]
        if temporal_postprocessor == "vdpp":
            self.steps.insert(2, "Temporal Depth Stabilization")
        elif temporal_postprocessor != "off":
            raise ValueError("temporal_postprocessor must be off or vdpp")
        self.step_aliases = {
            "Stereo Pair Generation": "Stereo Pair Creation",
            "Direct VR Encoding": "Video Creation",
            "Frame Cropping": "Crop Frames",
        }
        base_weights = list(
            VDPP_PROGRESS_STEP_WEIGHTS
            if temporal_postprocessor == "vdpp"
            else PROGRESS_STEP_WEIGHTS
        )
        self.step_weights = (
            self._build_dynamic_weights(base_weights, settings)
            if settings is not None
            else base_weights
        )
        self.current_step_index = 0
        self.step_progress = 0
        self.step_total = 0

    def _build_dynamic_weights(
        self,
        base_weights: list[float],
        settings: dict[str, Any],
    ) -> list[float]:
        """Zero out disabled stages and scale active weights to measured runtime shares."""
        weights = dict(zip(self.steps, base_weights, strict=True))
        apply_distortion = bool(settings.get("apply_distortion", True))
        raw_crop = settings.get("crop_factor", 1.0)
        crop_factor = float(raw_crop) if isinstance(raw_crop, (int, float)) else 1.0
        upscale_model = str(settings.get("upscale_model") or "none")
        keep_intermediates = bool(settings.get("keep_intermediates", False))
        direct_vr_encode = bool(settings.get("direct_vr_encode", True))
        streamed_stereo = (
            direct_vr_encode
            and not keep_intermediates
            and upscale_model == "none"
            and (apply_distortion or crop_factor == 1.0)
        )
        is_quality = settings.get("stereo_render_mode") == "quality"
        if str(settings.get("depth_model_version") or "v2") == "see_through":
            weights["Depth Map Generation"] = 0.65
            weights["Stereo Pair Creation"] = 0.22 if is_quality else 0.10
        elif is_quality:
            weights["Stereo Pair Creation"] = max(weights["Stereo Pair Creation"], 0.30)
        disabled = {
            "Fisheye Distortion": not apply_distortion or streamed_stereo,
            "Crop Frames": (not apply_distortion and crop_factor == 1.0) or streamed_stereo,
            "AI Upscaling": upscale_model == "none",
            "VR Assembly": direct_vr_encode and not keep_intermediates,
        }
        for name, is_disabled in disabled.items():
            if is_disabled:
                weights[name] = 0.0
        if streamed_stereo:
            weights["Video Creation"] = 0.01
        total = sum(weights.values())
        return [weights[name] / total for name in self.steps] if total > 0.0 else base_weights

    def is_cancelled(self) -> bool:
        """Return the shared user-stop state without emitting progress."""

        with _PROCESSING_LOCK:
            owner = current_processing["session_id"]
            if self._owned_job and owner != self.session_id:
                return True
            return owner in (None, self.session_id) and current_processing["stop_requested"]

    def _emit_preview_array(
        self,
        frame_array: np.ndarray,
        frame_type: str,
        frame_number: int,
    ) -> None:
        height, width = frame_array.shape[:2]
        if width <= 0 or height <= 0:
            vprint(f"Preview: Invalid dimensions {width}x{height}")
            return
        max_input_dim = 8192
        if width > max_input_dim or height > max_input_dim:
            vprint(f"Preview: Frame too large ({width}x{height}), exceeds max {max_input_dim}px")
            return
        scale = self.preview_downscale_width / width
        new_width = self.preview_downscale_width
        new_height = max(1, int(height * scale))
        frame_small = cv2.resize(
            frame_array,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )
        ok, buffer = cv2.imencode(
            ".jpg",
            frame_small,
            [int(cv2.IMWRITE_JPEG_QUALITY), PREVIEW_JPEG_QUALITY],
        )
        if not ok:
            return
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        if len(img_base64) / 1024 > 500:
            vprint(f"Preview: Payload too large, skipping {frame_type} frame {frame_number}")
            return
        preview_data = {
            "frame_type": frame_type,
            "frame_number": frame_number,
            "image_data": f"data:image/jpeg;base64,{img_base64}",
            "dimensions": {"width": new_width, "height": new_height},
        }
        socketio.emit("frame_preview", preview_data, room=self.session_id)

    def send_preview_frame(
        self,
        frame_path: Path,
        frame_type: str,
        frame_number: int,
    ) -> None:
        """Send preview frame via websocket."""
        if not self.enable_live_preview:
            return
        current_time = time.time()
        if current_time - self.last_preview_time < self.preview_interval:
            return
        try:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                vprint(f"Preview: Failed to read frame {frame_path}")
                return
            self._emit_preview_array(frame, frame_type, frame_number)
            self.last_preview_time = current_time
        except Exception as e:
            vprint(
                f"Preview: Error sending {frame_type} frame {frame_number}: {type(e).__name__}: {e}"
            )

    def send_preview_frame_from_array(
        self,
        frame_array: np.ndarray,
        frame_type: str,
        frame_number: int,
    ) -> None:
        """Send preview frame directly from numpy array (no disk I/O)."""
        if not self.enable_live_preview:
            return
        current_time = time.time()
        if current_time - self.last_preview_time < self.preview_interval:
            return
        try:
            self._emit_preview_array(frame_array, frame_type, frame_number)
            self.last_preview_time = current_time
        except Exception as e:
            vprint(
                f"Preview: Error sending {frame_type} frame {frame_number}: {type(e).__name__}: {e}"
            )

    def _calculate_eta(self, current_progress: float) -> str | None:
        """
        Calculate estimated time remaining using adaptive per-step timing.

        This uses actual measured time-per-frame for the current step and
        weights for remaining steps, providing much more accurate ETAs
        especially for slow steps like ESRGAN upscaling.

        Args:
            current_progress: Current progress percentage (0-100)

        Returns:
            Formatted ETA string (e.g., "5m 23s") or None if not enough data
        """
        if current_progress <= 0:
            return None

        current_time = time.time()
        elapsed = current_time - self.start_time

        # Need at least 3 seconds of data for reasonable estimate
        if elapsed < 3:
            return None

        # Strategy: Estimate remaining time for current step + active remaining steps
        remaining_time = 0.0
        has_step_estimate = False

        # 1. Estimate remaining time for current step from frames completed since step start
        if self.step_total > 0 and self.step_progress > 0:
            step_start_time = self.step_start_times.get(self.current_step_name, self.start_time)
            step_elapsed = current_time - step_start_time
            frames_completed = self.step_progress - self.step_start_progress
            frames_remaining = max(0, self.step_total - self.step_progress)
            if frames_remaining == 0:
                has_step_estimate = True
            elif frames_completed >= 2 and step_elapsed >= 0.5:
                time_per_frame = step_elapsed / frames_completed
                remaining_time += time_per_frame * frames_remaining
                has_step_estimate = True
            else:
                # Never extrapolate a brand-new counted step from a prior step's speed
                # (e.g. slow diffusion depth -> fast parallel stereo); wait ~0.5s for real rate.
                return None

        # 2. Add estimated time for active future steps only (disabled steps have weight 0)
        future_weight = (
            sum(self.step_weights[self.current_step_index + 1 :]) * 100.0
            if self.current_step_index < len(self.steps) - 1
            else 0.0
        )
        if future_weight > 0.0 and current_progress > 5:
            avg_time_per_percent = elapsed / current_progress
            if has_step_estimate:
                remaining_time += avg_time_per_percent * future_weight
            else:
                remaining_time = avg_time_per_percent * (100.0 - current_progress)

        # Fallback: use overall progress rate if current step has no rate sample yet
        if not has_step_estimate and remaining_time <= 0 and current_progress > 0:
            progress_ratio = current_progress / 100.0
            estimated_total_time = elapsed / progress_ratio
            remaining_time = estimated_total_time - elapsed

        if remaining_time < 0:
            return None

        return self._format_time(remaining_time)

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time (e.g., '5m 23s' or '2h 15m')."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:  # Less than 1 hour
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:  # 1 hour or more
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def update_progress(  # noqa: C901
        self,
        stage: str,
        frame_num: int | None = None,
        phase: str | None = None,
        step_name: str | None = None,
        step_progress: int | None = None,
        step_total: int | None = None,
    ) -> None:
        import time

        # Check if stop has been requested
        if self.is_cancelled():
            raise InterruptedError("Processing stopped by user request")

        # Throttle updates without dropping step transitions or 100% completions
        current_time = time.time()
        step_changed = bool(step_name and step_name != self.current_step_name)
        step_finished = (
            step_progress is not None
            and step_total is not None
            and step_total > 0
            and step_progress >= step_total
        )
        if (
            not step_changed
            and not step_finished
            and current_time - self.last_update_time < PROGRESS_UPDATE_INTERVAL
        ):
            return
        self.last_update_time = current_time

        # Update phase if provided
        if phase:
            self.current_phase = phase

        # Track step changes and timing
        if step_name and step_name != self.current_step_name:
            # New step started
            self.current_step_name = step_name
            self.step_start_times[step_name] = current_time
            self.step_frame_times = []  # Reset frame times for new step
            self.step_start_progress = (
                step_progress if (step_progress is not None and step_progress > 0) else 0
            )

            # Update step index
            weighted_step_name = self.step_aliases.get(step_name, step_name)
            if weighted_step_name in self.steps:
                self.current_step_index = self.steps.index(weighted_step_name)

        # Update step progress
        if step_progress is not None:
            self.step_progress = step_progress
        if step_total is not None:
            self.step_total = step_total

        if frame_num is not None:
            self.current_frame = frame_num

        # Calculate overall progress using weighted steps
        step_progress_ratio = (
            (self.step_progress / max(self.step_total, 1)) if self.step_total > 0 else 0
        )

        # Sum weights of all completed steps
        cumulative_weight = sum(self.step_weights[: self.current_step_index])

        # Add weighted progress of current step
        if self.current_step_index < len(self.step_weights):
            cumulative_weight += step_progress_ratio * self.step_weights[self.current_step_index]

        progress = max(self._max_progress, round(cumulative_weight * 100, PROGRESS_DECIMAL_PLACES))
        self._max_progress = progress

        # Calculate ETA
        eta_str = self._calculate_eta(progress)

        # Emit progress update (always include step data for UI)
        progress_data = {
            "progress": progress,
            "stage": stage,
            "current_frame": self.current_frame,
            "total_frames": self.total_frames,
            "phase": self.current_phase,
            "step_name": self.current_step_name or "",
            "step_progress": self.step_progress,
            "step_total": self.step_total,
            "step_index": self.current_step_index,
            "total_steps": len(self.steps),
            "eta": eta_str,  # Add ETA
        }

        with _PROCESSING_LOCK:
            owner = current_processing["session_id"]
            if owner == self.session_id or (owner is None and not self._owned_job):
                current_processing.update(progress_data)

        # Console output - show both overall and step progress
        step_percent = (
            (self.step_progress / max(self.step_total, 1)) * 100 if self.step_total > 0 else 0
        )
        eta_suffix = f" | ETA: {eta_str}" if eta_str else ""
        progress_msg = (
            f"Overall: {progress:05.1f}% | "
            f"Step: {step_percent:03.0f}% ({self.step_progress:04d}/{self.step_total:04d}) | "
            f"{stage}{eta_suffix}"
        )
        print(progress_msg)

        try:
            # Emit progress (socketio.start_background_task handles context automatically)
            socketio.emit("progress_update", progress_data, room=self.session_id)
            # Yield control to allow message to be sent immediately (fixes buffering issue)
            socketio.sleep(SOCKETIO_SLEEP_YIELD)
        except Exception as e:
            print(console_warning(f"Error emitting progress: {e}"))
            import traceback

            traceback.print_exc()

    def get_step_duration(self):
        """Get duration of current step in seconds."""
        import time

        if self.current_step_name and self.current_step_name in self.step_start_times:
            return time.time() - self.step_start_times[self.current_step_name]
        return 0

    def finish(self, message: str = "Processing complete"):
        """Finish progress tracking and notify the active web session."""
        print(f"{message}")
        try:
            socketio.emit(
                "processing_complete",
                {"success": True, "message": message},
                room=self.session_id,
            )
            # Allow time for message to be sent
            socketio.sleep(SOCKETIO_SLEEP_YIELD)
        except Exception as e:
            print(console_warning(f"Error emitting completion: {e}"))


def process_video_async(  # noqa: C901
    session_id: str,
    requester_socket_id: str,
    video_path: str | Path,
    settings: dict[str, Any],
    output_dir: str | Path,
    resume_context: dict[str, Any] | None = None,
    video_properties: dict[str, Any] | None = None,
) -> None:
    """Process video in background thread"""
    projector = None
    job_lock = None
    try:
        with _PROCESSING_LOCK:
            if (
                current_processing["session_id"] == session_id
                and current_processing["stop_requested"]
            ):
                raise InterruptedError("Processing stopped by user request")
        resolved_output_dir = Path(output_dir).resolve()
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        job_lock = JobWriterLock(resolved_output_dir).acquire()
        existing_settings = None
        if resume_context is not None:
            # Another writer may have finished between the HTTP check and this lock.
            existing_settings = _load_incomplete_web_settings(
                find_settings_file(resolved_output_dir)
            )
        elif (fixed := resolved_output_dir / "processing-settings-v5.json").is_file():
            existing_settings = load_processing_settings(fixed)
        cleanup_vdpp_private_work(resolved_output_dir, job_lock)

        selected_artifact = None
        resume_report = None
        if resume_context is not None:
            settings = dict(settings)
            settings["depth_model_version"] = resolve_resume_depth_model_version(
                settings,
                output_dir,
                default="v3",
            )
        else:
            settings = _resolve_new_web_stereo_settings(settings)
        settings = _validate_web_settings(
            _normalize_depth_backend_settings(settings),
            source="explicit",
        )
        depth_model_version = settings["depth_model_version"]
        model_size = settings["model_size"]
        model_path = settings["model_path"]
        use_metric = settings["use_metric_depth"]

        if video_properties is None:
            video_properties = get_video_properties(str(video_path))
            if not video_properties:
                video_properties = get_video_info(video_path)
        if not video_properties:
            raise Exception("Could not read video file")
        settings["depth_resolution"] = resolve_depth_input_size(
            int(video_properties["width"]),
            int(video_properties["height"]),
            settings.get("depth_resolution", "auto"),
        )
        validate_backend_geometry_request(settings, video_properties)

        if (
            settings["stereo_geometry_mode"] == "metric_camera"
            and settings.get("temporal_postprocessor", "off") != "off"
        ):
            raise ValueError("VDPP is available only with relative stereo geometry")

        if resume_context is not None:
            resume_report = build_resume_report(
                output_dir,
                settings,
                source_video=video_path,
                settings_file=resume_context.get("settings_file"),
            )
            if settings["stereo_geometry_mode"] == "relative":
                selected_artifact = _preserved_render_artifact(
                    resume_report,
                    settings.get("temporal_postprocessor", "off"),
                )

        depth_model_version = settings.get("depth_model_version", "v3")  # Default to V3
        model_size = settings.get("model_size", "vitb")  # Default to Base for 16GB GPUs
        use_metric = settings.get("use_metric_depth", True)  # Default to metric depth
        selected_source = selected_artifact[1] if selected_artifact is not None else None
        execution_plan = build_artifact_execution_plan(
            temporal_postprocessor=settings.get("temporal_postprocessor", "off"),
            base_artifact_valid=lambda: selected_source == "base",
            stabilized_artifact_valid=lambda: selected_source == "stabilized",
        )
        needs_base_model = execution_plan.needs_base_depth_model
        needs_vdpp_generation = execution_plan.needs_vdpp_model

        device = str(settings.get("device", "auto"))
        cuda_available = False
        if needs_base_model or needs_vdpp_generation:
            import torch  # Delay CUDA/runtime initialization until planning requires it

            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                print(f"CUDA detected: {torch.cuda.get_device_name(0)}")
            else:
                print("CUDA not available, using CPU")
            if device == "auto":
                device = "cuda" if cuda_available else "cpu"
            if device.startswith("cuda") and not cuda_available:
                raise RuntimeError(
                    "GPU (CUDA) requested but not available. Please select Auto or CPU."
                )
            if needs_vdpp_generation and not device.startswith("cuda"):
                raise RuntimeError("VDPP generation requires CUDA")

        if needs_base_model:
            # Resolve the model only after a model-free artifact audit misses.
            print(
                f"Loading {get_backend_spec(depth_model_version).display_name}: "
                f"{model_size} (metric inference: {use_metric})"
            )
            print(f"Using device: {device.upper()}")
            projector = create_stereo_projector(
                model_path,
                device,
                metric=use_metric,
                depth_model_version=depth_model_version,
                model_size=model_size,
            )
            report = build_effective_depth_run_report(settings, projector.depth_estimator)
            _acknowledge_processing_configuration(requester_socket_id, report)
            if settings["stereo_geometry_mode"] == "metric_camera":
                print(TEMPORAL_STABILITY_WARNING)
            if not projector.load_model():
                raise RuntimeError("Failed to load depth estimation model")

            if resume_context is not None:
                model_fingerprint = build_current_model_fingerprint(
                    projector.depth_estimator,
                    settings,
                )
                resume_report = build_resume_report(
                    output_dir,
                    settings,
                    source_video=video_path,
                    model_fingerprint=model_fingerprint,
                    settings_file=resume_context.get("settings_file"),
                )

        if resume_context is not None:
            assert resume_report is not None
            migration_mode = resume_context.get("migration_mode", "archive")
            apply_legacy_migration(resume_report, migration_mode)
            resolved_temporal = settings.get("temporal_postprocessor", "off")
            settings = dict(resume_report.migrated_settings)
            settings.setdefault("temporal_postprocessor", resolved_temporal)
            if "video_path" in settings:
                settings["video_path"] = str(video_path)

        video_info = video_properties
        saved_video = (existing_settings or {}).get("video_properties")
        if isinstance(saved_video, dict) and "frame_rate" not in saved_video:
            video_info = dict(video_info)
            video_info.pop("frame_rate", None)
        start_frame, end_frame = selected_frame_range(video_info, settings)
        expected_frames = end_frame - start_frame

        enable_live_preview = settings.get("enable_live_preview", True)
        preview_update_interval = settings.get("preview_update_interval", PREVIEW_UPDATE_INTERVAL)
        callback = ProgressCallback(
            session_id,
            expected_frames,
            enable_live_preview,
            preview_update_interval,
            temporal_postprocessor=settings.get("temporal_postprocessor", "off"),
            settings=settings,
        )

        # Give client time to join the session room before starting processing
        socketio.sleep(INITIAL_PROCESSING_DELAY)

        temporal_stabilizer = None
        if needs_vdpp_generation:
            from depth_surge_3d.processing.frames.temporal_stabilizer import (
                TemporalDepthStabilizer,
            )

            temporal_stabilizer = TemporalDepthStabilizer(
                effective_device=device,
                models_dir=Path("models"),
            )
        processor_kwargs: dict[str, Any] = {
            "temporal_stabilizer": temporal_stabilizer,
        }
        if selected_artifact is not None:
            selected_files, selected_source = selected_artifact
            processor = VideoProcessor(
                None,
                preselected_depth_files=selected_files,
                preselected_render_source=selected_source,
                **processor_kwargs,
            )
            selected_artifact = None
            selected_files = None
            processor_kwargs.clear()
        else:
            assert projector is not None
            processor = VideoProcessor(
                projector.depth_estimator,
                release_depth_model=projector.unload_model,
                **processor_kwargs,
            )
            processor_kwargs.clear()

        # Calculate resolution settings that VideoProcessor expects
        from depth_surge_3d.utils.domain.resolution import (
            get_resolution_dimensions,
            calculate_vr_output_dimensions,
            auto_detect_resolution,
        )

        # Resolve VR resolution if auto
        vr_resolution = settings.get("vr_resolution", "auto")
        if vr_resolution == "auto":
            vr_resolution = auto_detect_resolution(
                video_info["width"],
                video_info["height"],
                settings.get("vr_format", "side_by_side"),
            )

        # Get resolution dimensions
        per_eye_width, per_eye_height = get_resolution_dimensions(vr_resolution)
        vr_output_width, vr_output_height = calculate_vr_output_dimensions(
            per_eye_width, per_eye_height, settings.get("vr_format", "side_by_side")
        )

        # Add calculated dimensions to settings
        settings.update(
            {
                "per_eye_width": per_eye_width,
                "per_eye_height": per_eye_height,
                "vr_output_width": vr_output_width,
                "vr_output_height": vr_output_height,
                "source_width": video_info["width"],
                "source_height": video_info["height"],
                "source_fps": video_info["fps"],
            }
        )

        success = processor.process(
            video_path=video_path,
            output_dir=output_dir,
            video_properties=video_info,
            settings=settings,
            progress_callback=callback,
            job_lock=job_lock,
        )

        if not success:
            raise Exception("Video processing failed")

        # Processing complete
        try:
            socketio.emit(
                "processing_complete",
                {
                    "success": True,
                    "output_dir": str(output_dir),
                    "message": "Video processing completed successfully!",
                },
                room=session_id,
            )
            # Allow time for message to be sent before thread terminates
            socketio.sleep(SOCKETIO_SLEEP_YIELD)
        except Exception as e:
            print(console_warning(f"Error emitting completion: {e}"))

    except InterruptedError as e:
        # Handle user-requested stop
        try:
            socketio.emit(
                "processing_stopped",
                {"success": True, "message": str(e)},
                room=session_id,
            )
            # Allow time for message to be sent before thread terminates
            socketio.sleep(SOCKETIO_SLEEP_YIELD)
        except Exception as emit_error:
            print(console_warning(f"Error emitting stop: {emit_error}"))

    except Exception as e:
        try:
            socketio.emit("processing_error", {"success": False, "error": str(e)}, room=session_id)
            # Allow time for message to be sent before thread terminates
            socketio.sleep(SOCKETIO_SLEEP_YIELD)
        except Exception as emit_error:
            print(console_warning(f"Error emitting error: {emit_error}"))
        print(f"Processing error: {e}")  # Always print errors

    finally:
        if projector is not None:
            try:
                projector.unload_model()
            except Exception as cleanup_error:
                print(console_warning(f"Error unloading depth model: {cleanup_error}"))
        if job_lock is not None:
            job_lock.release()


@app.route("/")
def index():
    """Main page"""
    return render_template("index.html", depth_backends=_depth_backend_options())


@app.route("/upload", methods=["POST"])
def upload_video() -> tuple[dict[str, Any], int] | tuple[Any, int]:
    """Handle video upload - saves directly to output directory with audio extraction"""
    from depth_surge_3d.utils.path_utils import sanitize_filename

    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    # Create timestamped output directory with sanitized filename
    original_filename = _web_video_filename(sanitize_filename(file.filename))
    video_name = Path(original_filename).stem[:48]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_directories()
    output_dir = Path(
        tempfile.mkdtemp(
            prefix=f"{int(time.time())}_{video_name}_",
            suffix=f"_{timestamp}",
            dir=app.config["OUTPUT_FOLDER"],
        )
    )
    vprint(f"Created output directory: {output_dir}")
    vprint(f"  Directory exists: {output_dir.exists()}")

    # Save video to output directory preserving original filename
    video_path = output_dir / original_filename
    vprint(f"Saving video as: {video_path.name}")
    try:
        file.save(video_path)
        vprint(f"  Video saved: {video_path.exists()}")
        video_info = get_video_info(video_path)
        if not video_info:
            shutil.rmtree(output_dir, ignore_errors=True)
            return jsonify({"error": "Invalid video file"}), 400
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise

    # Detect audio stream presence without blocking upload on a full FLAC transcode
    has_audio = False
    try:
        probe_result = _run_web_media(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            timeout=10,
        )
        has_audio = probe_result.returncode == 0 and probe_result.stdout.strip() == "audio"
    except Exception as e:
        print(f"Warning: Audio probe error: {e}")

    return jsonify(
        {
            "success": True,
            "filename": video_path.name,
            "output_dir": str(output_dir.resolve()),  # Ensure absolute path
            "video_info": video_info,
            "has_audio": has_audio,
        }
    )


@app.route("/process", methods=["POST"])
def start_processing() -> tuple[dict[str, Any], int] | tuple[Any, int]:  # noqa: C901
    """Start video processing"""
    if _processing_status()["active"]:
        return jsonify({"error": "Processing already in progress"}), 400

    data = _json_object()
    output_dir_str = data.get("output_dir")
    settings = data.get("settings", {})

    try:
        if not isinstance(settings, dict):
            raise ValueError("settings must be a JSON object")
        if settings.get("video_path") is not None:
            raise ValueError("video_path is selected from the managed job, not the request")
        _check_web_model_selection(settings)
        settings = _resolve_new_web_stereo_settings(settings)
        settings = _normalize_available_depth_backend_settings(settings)
        settings = _validate_web_settings(settings, source="explicit")
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    if settings.get("temporal_postprocessor") == "vdpp":
        import torch

        if settings.get("device") == "cpu" or not torch.cuda.is_available():
            return jsonify({"error": "VDPP generation requires CUDA"}), 400

    output_dir = _managed_job_directory(output_dir_str)
    vprint(f"Processing request for output directory: {output_dir}")

    # Find the source video file in output directory
    video_path = find_source_video(output_dir)

    if not video_path:
        return (
            jsonify({"error": "Source video file not found in output directory"}),
            404,
        )

    try:
        video_properties = get_video_properties(str(video_path))
        if not video_properties:
            raise ValueError("Could not read video file")
        settings["depth_resolution"] = resolve_depth_input_size(
            int(video_properties["width"]),
            int(video_properties["height"]),
            settings.get("depth_resolution", "auto"),
        )
        validate_backend_geometry_request(settings, video_properties)
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        requester_socket_id = _active_requester_socket_id(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    session_id = _start_processing_job(
        requester_socket_id, video_path, settings, output_dir, None, video_properties
    )

    return jsonify({"success": True, "session_id": session_id, "output_dir": str(output_dir)})


@app.route("/stop", methods=["POST"])
def stop_processing() -> tuple[Any, int] | Any:
    """Stop current processing"""
    data = _json_object()
    session_id = data.get("session_id")

    with _PROCESSING_LOCK:
        if not current_processing["active"]:
            return jsonify({"success": False, "error": "No processing currently active"}), 409
        if session_id != current_processing["session_id"]:
            return jsonify({"success": False, "error": "Invalid session ID"}), 400
        current_processing["stop_requested"] = True

    return jsonify({"success": True, "message": "Stop request sent"})


def _prepare_resume_request(  # noqa: C901
    data: Any,
) -> tuple[Path, str, str | None, str | None, str | None, int | None]:
    """Validate resume overrides and constrain the requested output directory."""

    if not isinstance(data, dict):
        raise _WebRequestError("Request body must be a JSON object", 400)

    output_dir = data.get("output_dir")
    migrate_legacy = data.get("migrate_legacy", "archive")
    raw_storage_dtype = data.get("raw_storage_dtype")
    temporal_present = "temporal_postprocessor" in data
    temporal_override = data.get("temporal_postprocessor") if temporal_present else None
    stereo_present = "stereo_render_mode" in data
    stereo_override = data.get("stereo_render_mode") if stereo_present else None
    fill_limit_present = "occlusion_fill_max_px" in data
    fill_limit_override = data.get("occlusion_fill_max_px") if fill_limit_present else None
    resume_overrides = {"migrate_legacy": migrate_legacy}
    if raw_storage_dtype is not None:
        resume_overrides["raw_storage_dtype"] = raw_storage_dtype
    if temporal_present:
        resume_overrides["temporal_postprocessor"] = temporal_override
    if stereo_present:
        resume_overrides["stereo_render_mode"] = stereo_override
    if fill_limit_present:
        resume_overrides["occlusion_fill_max_px"] = fill_limit_override
    try:
        validated = validate_settings(resume_overrides, source="explicit")
    except (TypeError, ValueError) as exc:
        raise _WebRequestError(str(exc), 400) from exc

    output_path = _managed_job_directory(output_dir)

    selected_dtype = validated.get("raw_storage_dtype") if raw_storage_dtype is not None else None
    selected_temporal = validated["temporal_postprocessor"] if temporal_present else None
    selected_stereo = validated["stereo_render_mode"] if stereo_present else None
    selected_fill_limit = validated["occlusion_fill_max_px"] if fill_limit_present else None
    return (
        output_path,
        validated["migrate_legacy"],
        selected_dtype,
        selected_temporal,
        selected_stereo,
        selected_fill_limit,
    )


@app.route("/resume", methods=["POST"])
def resume_processing():  # noqa: C901
    """Resume processing from a previous interrupted batch"""
    if _processing_status()["active"]:
        return jsonify({"error": "Processing already in progress"}), 400

    (
        output_path,
        migrate_legacy,
        raw_storage_dtype,
        temporal_override,
        stereo_override,
        fill_limit_override,
    ) = _prepare_resume_request(_json_object())

    settings_file = find_settings_file(output_path)
    try:
        settings = detect_resume_settings(output_path, settings_file=settings_file)
    except _WebRequestError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        return jsonify({"error": f"Could not read resume settings: {exc}"}), 409
    try:
        source_video = resolve_resume_source_video(
            output_path,
            settings_file=settings_file,
        )
    except (OSError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 409
    settings["depth_model_version"] = resolve_resume_depth_model_version(
        settings,
        output_path,
        default="v3",
    )
    _check_web_model_selection(settings)
    settings["migrate_legacy"] = migrate_legacy
    if raw_storage_dtype is not None:
        settings["raw_storage_dtype"] = raw_storage_dtype
    settings["temporal_postprocessor"] = resolve_temporal_postprocessor(
        persisted=settings.get("temporal_postprocessor"),
        override=temporal_override,
        is_resume=True,
    )
    settings = _resolve_resume_web_stereo_settings(
        settings,
        stereo_override=stereo_override,
        fill_limit_override=fill_limit_override,
    )
    resolved_temporal = settings["temporal_postprocessor"]
    resolved_stereo = settings["stereo_render_mode"]
    resolved_fill_limit = settings["occlusion_fill_max_px"]

    try:
        resume_report = build_resume_report(
            output_path,
            settings,
            source_video=source_video,
            settings_file=settings_file,
        )
        settings = dict(resume_report.migrated_settings)
        settings.setdefault("temporal_postprocessor", resolved_temporal)
        settings["stereo_render_mode"] = resolved_stereo
        settings["occlusion_fill_max_px"] = resolved_fill_limit
    except (OSError, TypeError, ValueError) as exc:
        return jsonify({"error": f"Could not migrate resume data: {exc}"}), 409

    try:
        requester_socket_id = _active_requester_socket_id(request.json)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    session_id = _start_processing_job(
        requester_socket_id,
        source_video,
        settings,
        output_path,
        {"migration_mode": migrate_legacy, "settings_file": settings_file},
    )

    return jsonify(
        {
            "success": True,
            "session_id": session_id,
            "output_dir": str(output_path),
            "resume_report": resume_report.to_dict(),
            "settings": {
                "temporal_postprocessor": settings["temporal_postprocessor"],
                "stereo_render_mode": settings["stereo_render_mode"],
                "occlusion_fill_max_px": settings["occlusion_fill_max_px"],
            },
        }
    )


def detect_resume_settings(output_path, *, settings_file=None):
    """Detect processing settings from existing output directory"""
    settings = _validate_web_settings(
        {"device": "auto", "target_fps": None},
        source="explicit",
    )

    # A resume must use the exact model and geometry from the interrupted run.
    # Directory-shape inference is only a fallback for legacy jobs without metadata.
    selected = settings_file or find_settings_file(output_path)
    settings_files = [selected] if selected is not None else []
    for settings_file in settings_files:
        saved_data = _load_incomplete_web_settings(settings_file)
        saved_settings = saved_data.get("processing_settings") if saved_data else None
        if isinstance(saved_settings, dict):
            metadata = saved_data.get("metadata", {})
            saved_version = (
                metadata.get("settings_schema_version") if isinstance(metadata, dict) else None
            )
            settings = parse_saved_processing_settings(
                saved_settings,
                saved_version=saved_version,
            ).settings
            settings["depth_model_version"] = resolve_resume_depth_model_version(
                settings,
                output_path,
                default="v3",
            )
            return settings

    # Try to detect VR format from directory structure
    if (output_path / INTERMEDIATE_DIRS["vr_frames"]).exists():
        # Check if there are side-by-side or over-under frames
        vr_files = list((output_path / INTERMEDIATE_DIRS["vr_frames"]).glob("*.png"))
        if vr_files:
            sample_frame = cv2.imread(str(vr_files[0]))
            if sample_frame is not None:
                height, width = sample_frame.shape[:2]
                if width > height * ASPECT_RATIO_SBS_THRESHOLD:  # Likely side-by-side
                    settings["vr_format"] = "side_by_side"
                elif height > width * ASPECT_RATIO_OU_THRESHOLD:  # Likely over-under
                    settings["vr_format"] = "over_under"

    settings["depth_model_version"] = resolve_resume_depth_model_version(
        settings,
        output_path,
        default="v3",
    )
    return settings


@app.route("/status")
def get_status():
    """Get current processing status"""
    return jsonify(_processing_status())


@app.route("/system_info")
def get_system_info_endpoint():
    """Get system information"""
    return jsonify(get_system_info())


def _resumable_job_summary(candidate: Path) -> dict[str, Any] | None:
    try:
        batch_dir = _managed_job_directory(candidate, inspect_contents=False)
        frames_dir = batch_dir / INTERMEDIATE_DIRS["frames"]
        if not os.path.lexists(frames_dir) or any(batch_dir.glob("*_3D_*.mp4")):
            return None
        frames_info = frames_dir.lstat()
        if stat.S_ISLNK(frames_info.st_mode) or _is_reparse_point(frames_info):
            return None
    except (_WebRequestError, OSError):
        return None

    # Check for source video file
    source_video = find_source_video(batch_dir)

    if not source_video:
        return None

    analysis = analyze_batch_directory(batch_dir)
    job = {
        "path": str(batch_dir),
        "name": batch_dir.name,
        "highest_stage": analysis.get("highest_stage", "unknown"),
        "frame_count": analysis.get("frame_count", 0),
        "vr_format": analysis.get("vr_format", "unknown"),
        "resolution": analysis.get("resolution", "unknown"),
    }
    try:
        resume_settings = detect_resume_settings(batch_dir)
        job["temporal_postprocessor"] = resume_settings.get("temporal_postprocessor", "off")
    except _WebRequestError:
        return None
    except (OSError, TypeError, ValueError):
        pass
    return job


@app.route("/detect_resumable")
def detect_resumable_jobs():
    """Detect incomplete processing jobs that can be resumed"""
    output_folder = Path(app.config["OUTPUT_FOLDER"])
    if not output_folder.exists():
        return jsonify({"success": True, "jobs": []})

    resumable_jobs = []

    try:
        # Scan output directories for incomplete jobs
        for candidate in output_folder.iterdir():
            job = _resumable_job_summary(candidate)
            if job is None:
                continue
            resumable_jobs.append(job)

        # Sort by modification time (most recent first), tolerating concurrent deletions
        def _safe_mtime(job_entry: dict[str, Any]) -> float:
            try:
                return Path(job_entry["path"]).stat().st_mtime
            except OSError:
                return 0.0

        resumable_jobs.sort(key=_safe_mtime, reverse=True)

        return jsonify({"success": True, "jobs": resumable_jobs})

    except Exception as e:
        vprint(f"Error detecting resumable jobs: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/open_directory", methods=["POST"])
def open_directory():
    """Open directory in file explorer"""
    directory_path = str(_managed_job_directory(_json_object().get("path"), inspect_contents=False))

    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(directory_path)
        elif system == "Darwin":  # macOS
            subprocess.Popen(["open", directory_path])
        else:  # Linux and others
            subprocess.Popen(["xdg-open", directory_path])

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


def analyze_batch_directory(batch_path):  # noqa: C901
    """Analyze batch directory to determine available processing stages and settings"""
    analysis = {
        "frame_count": 0,
        "vr_format": "unknown",
        "resolution": "unknown",
        "highest_stage": "none",
        "has_audio": False,
        "settings_summary": "unknown",
    }

    # Check for different processing stages
    stages = {
        INTERMEDIATE_DIRS["vr_frames"]: "Final VR frames",
        INTERMEDIATE_DIRS["left_upscaled"]: "Upscaled left frames",
        INTERMEDIATE_DIRS["right_upscaled"]: "Upscaled right frames",
        INTERMEDIATE_DIRS["left_distorted"]: "Distorted left frames",
        INTERMEDIATE_DIRS["right_distorted"]: "Distorted right frames",
        INTERMEDIATE_DIRS["left_cropped"]: "Cropped left frames",
        INTERMEDIATE_DIRS["right_cropped"]: "Cropped right frames",
        INTERMEDIATE_DIRS["left_frames"]: "Stereo left frames",
        INTERMEDIATE_DIRS["right_frames"]: "Stereo right frames",
        INTERMEDIATE_DIRS["disparity_maps"]: "Canonical disparity maps",
        INTERMEDIATE_DIRS["frames"]: "Original frames",
    }

    highest_stage_num = 0
    for stage_dir, stage_name in stages.items():
        stage_path = batch_path / stage_dir
        if stage_path.is_dir():
            with os.scandir(stage_path) as entries:
                frame_count = sum(
                    1
                    for entry in entries
                    if entry.name.lower().endswith((".png", ".jpg"))
                    and entry.is_file(follow_symlinks=False)
                )
            if frame_count > 0:
                stage_num = int(stage_dir.split("_")[0])
                if stage_num > highest_stage_num:
                    highest_stage_num = stage_num
                    analysis["highest_stage"] = stage_name
                    analysis["frame_count"] = frame_count

    # Detect VR format and resolution from highest stage
    if highest_stage_num > 0:
        sample_frame_dirs = [
            d
            for d in [
                INTERMEDIATE_DIRS["vr_frames"],
                INTERMEDIATE_DIRS["left_upscaled"],
                INTERMEDIATE_DIRS["right_upscaled"],
                INTERMEDIATE_DIRS["left_cropped"],
                INTERMEDIATE_DIRS["right_cropped"],
                INTERMEDIATE_DIRS["left_distorted"],
                INTERMEDIATE_DIRS["right_distorted"],
                INTERMEDIATE_DIRS["left_frames"],
                INTERMEDIATE_DIRS["right_frames"],
                INTERMEDIATE_DIRS["frames"],
            ]
            if (batch_path / d).exists()
        ]
        from depth_surge_3d.utils.imaging.png_header import read_png_header

        for frame_dir in sample_frame_dirs:
            frame_path = batch_path / frame_dir
            first_png = None
            with os.scandir(frame_path) as entries:
                for entry in entries:
                    if entry.name.lower().endswith(".png") and entry.is_file(follow_symlinks=False):
                        first_png = Path(entry.path)
                        break
            if first_png is not None:
                try:
                    header = read_png_header(first_png)
                    if header is not None:
                        w, h = header.width, header.height
                        analysis["resolution"] = f"{w}x{h}"

                        # Detect format based on aspect ratio
                        if frame_dir == INTERMEDIATE_DIRS["vr_frames"]:
                            if w > h * ASPECT_RATIO_SBS_THRESHOLD:
                                analysis["vr_format"] = "side_by_side"
                            elif h > w * ASPECT_RATIO_OU_THRESHOLD:
                                analysis["vr_format"] = "over_under"
                            else:
                                analysis["vr_format"] = "square"
                        break
                except Exception:
                    pass

    # Check for pre-extracted audio file or source video in batch directory
    audio_file = batch_path / "original_audio.flac"
    if audio_file.exists():
        analysis["has_audio"] = True
    else:
        # Check for source video file
        source_video = find_source_video(batch_path)
        if source_video:
            analysis["has_audio"] = True

    # Generate settings summary
    if analysis["vr_format"] != "unknown" and analysis["resolution"] != "unknown":
        analysis["settings_summary"] = f"{analysis['vr_format']}, {analysis['resolution']}"

    return analysis


_STITCHED_VIDEO_COMMENT = "Depth Surge 3D manual frame export"


def _stitch_source(batch_path, settings_file, metadata):
    """Prefer the recorded local source; never mistake our previous export for it."""
    if metadata.get("source_video") or metadata.get("source_video_name"):
        try:
            source = resolve_resume_source_video(batch_path, settings_file=settings_file)
        except ValueError:
            return None, {}
        return source, get_video_info_ffprobe(str(source))

    candidates = []
    for path in batch_path.iterdir():
        name = path.name.lower()
        if (
            not path.is_file()
            or path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS
            or "_3d_" in name
            or "_stitched_" in name
        ):
            continue
        info = get_video_info_ffprobe(str(path))
        tags = info.get("format", {}).get("tags", {})
        if any(k.lower() == "comment" and v == _STITCHED_VIDEO_COMMENT for k, v in tags.items()):
            continue
        candidates.append((path, info))
        if len(candidates) > 1:
            raise ValueError("Cannot identify the original video among multiple files in this job")
    return candidates[0] if candidates else (None, {})


def _stitch_media_inputs(batch_path, target, include_audio, frame_count):
    settings_file = find_settings_file(batch_path)
    saved = load_processing_settings(settings_file) if settings_file is not None else {}
    if not isinstance(saved, dict):
        raise ValueError("Could not read the saved job settings")
    properties = saved.get("video_properties", {})
    processing = saved.get("processing_settings", {})
    source_rate = canonical_frame_rate(properties.get("frame_rate")) or canonical_frame_rate(
        processing.get("source_fps")
    )
    audio = batch_path / "original_audio.flac" if include_audio else None
    if audio is not None and not audio.is_file():
        audio = None
    if source_rate is None or (include_audio and audio is None):
        source, info = _stitch_source(batch_path, settings_file, saved.get("metadata", {}))
        stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
        source_rate = (
            source_rate
            or probe_frame_rate(stream)
            or processing.get("source_fps")
            or properties.get("fps")
        )
        if source_rate is None:
            if settings_file is not None or source is not None:
                raise ValueError("Cannot determine the source frame rate for this job")
            source_rate = DEFAULT_FALLBACK_FPS if target in (None, "original") else target
        if (
            include_audio
            and audio is None
            and any(s.get("codec_type") == "audio" for s in info.get("streams", []))
        ):
            audio = source
    timeline = EncodingTimeline.resolve(source_rate, target)
    audio_args = []
    if audio is not None:
        clock = (
            timeline.source
            if properties.get("frame_rate") or not properties.get("fps")
            else float(properties["fps"])
        )
        start_frame = calculate_start_frame(
            properties.get("frame_count"), clock, processing.get("start_time")
        )
        start, duration = ffmpeg_frame_interval(
            start_frame, start_frame + frame_count, timeline.source
        )
        if start_frame:
            audio_args.extend(["-ss", start])
        audio_args.extend(["-t", duration, "-i", str(audio)])
    return timeline, audio_args


def create_video_from_batch(batch_path, settings):  # noqa: C901
    """Create video from batch frames using FFmpeg"""
    frame_source = settings.get("frame_source", "auto")
    quality = settings.get("quality", "medium")
    fps = settings.get("fps", "original")
    include_audio = settings.get("include_audio", False)
    output_filename = settings.get("output_filename")

    # Determine frame directory to use
    if frame_source not in {"auto", "vr_frames"}:
        raise ValueError(f"Unsupported frame source: {frame_source}")
    frame_dir = batch_path / INTERMEDIATE_DIRS["vr_frames"]

    if not frame_dir or not frame_dir.exists():
        raise Exception(f"Frame directory not found: {frame_dir}")

    frame_names = {path.name for path in frame_dir.glob("frame_*.png")}
    if not frame_names:
        raise Exception(f"No frames found in {frame_dir}")
    first_index = 0 if "frame_000000.png" in frame_names else 1
    indices = range(first_index, first_index + len(frame_names))
    if any(f"frame_{n:06d}.png" not in frame_names for n in indices):
        raise ValueError("Expected a contiguous sequence of numbered VR frames")

    # Determine output filename
    if not output_filename:
        batch_name = batch_path.name
        timestamp = datetime.now().strftime("%H%M%S")
        output_filename = f"{batch_name}_stitched_{timestamp}.mp4"

    output_path = batch_path / _web_video_filename(output_filename)
    if os.path.lexists(output_path):
        raise _WebRequestError("That filename already exists. Choose a new name.", 409)

    timeline, audio_args = _stitch_media_inputs(batch_path, fps, include_audio, len(frame_names))
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-n",
        "-framerate",
        rate_text(timeline.source),
        "-start_number",
        str(first_index),
        "-i",
        str(frame_dir / "frame_%06d.png"),
        *audio_args,
        "-map",
        "0:v:0",
        "-vf",
        timeline.video_filter,
    ]
    if audio_args:
        cmd.extend(["-map", "1:a:0?", "-c:a", "aac"])

    # Quality settings
    quality_map = {
        "high": FFMPEG_CRF_HIGH_QUALITY,
        "medium": FFMPEG_CRF_MEDIUM_QUALITY,
        "fast": FFMPEG_CRF_FAST_QUALITY,
    }
    crf = quality_map.get(quality, FFMPEG_CRF_MEDIUM_QUALITY)

    cmd.extend(["-c:v", "libx264", "-crf", crf, "-preset", FFMPEG_DEFAULT_PRESET])
    cmd.extend(["-pix_fmt", FFMPEG_PIX_FORMAT])  # For compatibility

    cmd.extend(["-metadata", f"comment={_STITCHED_VIDEO_COMMENT}"])

    # Publish a complete movie without replacing a destination created during encoding.
    with tempfile.TemporaryDirectory(prefix=".stitch-", dir=batch_path) as staging:
        encoded = Path(staging) / ("video" + output_path.suffix.lower())
        try:
            result = _run_web_media([*cmd, str(encoded)], timeout=VIDEO_CREATION_TIMEOUT)
            if result.returncode != 0:
                raise ValueError(f"FFmpeg error: {result.stderr}")
            if os.name == "nt":
                encoded.rename(output_path)  # Windows rename never replaces an existing file.
            else:
                os.link(encoded, output_path)
        except FileExistsError as exc:
            raise _WebRequestError("That filename already exists. Choose a new name.", 409) from exc
        except subprocess.TimeoutExpired as exc:
            raise ValueError("Video creation timed out") from exc

    return output_path


@app.route("/analyze_batch", methods=["POST"])
def analyze_batch():
    """Analyze a batch directory for video stitching"""
    batch_path = _managed_job_directory(_json_object().get("batch_dir"))

    try:
        analysis = analyze_batch_directory(batch_path)
        return jsonify({"success": True, "analysis": analysis})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/stitch_video", methods=["POST"])
def stitch_video():
    """Create video from batch frames"""
    data = _json_object()
    batch_path = _managed_job_directory(data.get("batch_dir"))

    try:
        with JobWriterLock(batch_path):
            output_path = create_video_from_batch(batch_path, data)
        return jsonify({"success": True, "output_path": str(output_path)})
    except _WebRequestError:
        raise
    except JobAlreadyLockedError as exc:
        return jsonify({"success": False, "error": str(exc)}), 409
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@socketio.on("connect")
def handle_connect():
    vprint(f"Client connected: {request.sid}")


@socketio.on("disconnect")
def handle_disconnect():
    vprint(f"Client disconnected: {request.sid}")

    # Only stop processing if there are no other connected clients
    # Don't immediately stop on disconnect since users might refresh the page
    # The processing will continue and can be rejoined


@socketio.on("join_session")
def handle_join_session(data):
    session_id = data.get("session_id") if isinstance(data, dict) else None
    status = _active_session_snapshot(session_id)
    if status is not None:
        # Join the session room for progress updates
        from flask_socketio import join_room

        join_room(session_id)
        vprint(f"Client {request.sid} joined session {session_id[:SESSION_ID_DISPLAY_LENGTH]}...")

        # Send initial status to joined client
        try:
            initial_data = {
                "progress": status.get("progress", 0),
                "stage": status.get("stage", "Initializing..."),
                "current_frame": status.get("current_frame", 0),
                "total_frames": status.get("total_frames", 0),
                "phase": status.get("phase", "extraction"),
                "step_name": status.get("step_name", ""),
                "step_index": status.get("step_index", 0),
                "total_steps": status.get("total_steps", 8),
                "step_progress": status.get("step_progress", 0),
                "step_total": status.get("step_total", 0),
                "eta": status.get("eta"),
            }
            socketio.emit("progress_update", initial_data, to=request.sid)
        except Exception as e:
            print(console_warning(f"Error emitting initial progress: {e}"))
    else:
        socketio.emit(
            "session_join_error",
            {"error": "Invalid or inactive processing session"},
            room=request.sid,
        )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Depth Surge 3D Web UI")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (shows GET/SET requests and client details)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help=f"Port to run the server on (default: {DEFAULT_SERVER_PORT})",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_SERVER_HOST,
        choices=("127.0.0.1", "localhost", "::1"),
        help=f"Loopback address only; LAN access is not supported (default: {DEFAULT_SERVER_HOST})",
    )
    args = parser.parse_args()

    # Set global verbose flag
    VERBOSE = args.verbose

    ensure_directories()

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Always print banner on startup
    print_banner()

    # Only print additional startup message if not already printed by run_ui.sh
    if not os.environ.get("DEPTH_SURGE_UI_SCRIPT"):
        print("Starting Depth Surge 3D Web UI...")
        print(f"Navigate to http://localhost:{args.port}")

    # Suppress Flask/Werkzeug production warnings for desktop application
    if not args.verbose:
        import logging

        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)
        # Suppress the specific production deployment warning
        import warnings

        warnings.filterwarnings("ignore", message=".*Werkzeug.*production.*")

    try:
        socketio.run(
            app,
            host=DEFAULT_SERVER_HOST if args.host == "localhost" else args.host,
            port=args.port,
            debug=False,
            allow_unsafe_werkzeug=True,
        )
    finally:
        cleanup_processes()
