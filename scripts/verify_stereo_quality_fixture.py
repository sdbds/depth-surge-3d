"""Verify the hash-bound seven-frame stereo Quality production fixture."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Callable, Literal


_AFFINITY_BOOTSTRAP_ENV = "DEPTH_SURGE_QUALITY_AFFINITY_V1"


def _parse_cpu_affinity(value: str) -> tuple[int, ...]:  # noqa: C901
    """Parse a comma-separated processor list with inclusive ranges."""

    if not value.strip():
        raise ValueError("CPU affinity must not be empty")
    processors: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError("CPU affinity contains an empty processor")
        if part.startswith("-"):
            raise ValueError("CPU affinity processors must be nonnegative")
        if "-" in part:
            if part.count("-") != 1:
                raise ValueError(f"invalid CPU affinity range: {part!r}")
            start_text, end_text = part.split("-", maxsplit=1)
            if not start_text.isdecimal() or not end_text.isdecimal():
                raise ValueError(f"invalid CPU affinity range: {part!r}")
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError("CPU affinity ranges must be ascending")
            expanded = tuple(range(start, end + 1))
        else:
            if not part.isdecimal():
                raise ValueError(f"invalid CPU affinity processor: {part!r}")
            expanded = (int(part),)
        for processor in expanded:
            if processor in processors:
                raise ValueError(f"duplicate CPU affinity processor: {processor}")
            processors.add(processor)
    return tuple(sorted(processors))


def _windows_processor_group_count() -> int:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    getter = getattr(kernel32, "GetActiveProcessorGroupCount", None)
    if getter is None:
        raise RuntimeError("Windows processor-group identity is unavailable")
    getter.restype = ctypes.c_ushort
    count = int(getter())
    if count <= 0:
        raise RuntimeError("Windows processor-group identity is invalid")
    return count


def _windows_single_processor_group() -> None:
    if _windows_processor_group_count() != 1:
        raise RuntimeError(
            "flat --cpu-affinity requires one Windows processor group; "
            "group-qualified affinity syntax is not supported"
        )


def _count_windows_processor_core_records(raw: bytes) -> int:
    offset = 0
    count = 0
    while offset < len(raw):
        if len(raw) - offset < 8:
            raise RuntimeError("Windows processor topology record is truncated")
        relationship = int.from_bytes(raw[offset : offset + 4], "little")
        size = int.from_bytes(raw[offset + 4 : offset + 8], "little")
        if size < 8 or size > len(raw) - offset:
            raise RuntimeError("Windows processor topology record size is invalid")
        if relationship == 0:  # RelationProcessorCore
            count += 1
        offset += size
    if offset != len(raw) or count <= 0:
        raise RuntimeError("Windows physical-core topology is unavailable")
    return count


def _windows_physical_core_count() -> int:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    query = kernel32.GetLogicalProcessorInformationEx
    query.argtypes = (wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD))
    query.restype = wintypes.BOOL
    required = wintypes.DWORD(0)
    if query(0, None, ctypes.byref(required)):
        raise RuntimeError("Windows processor topology sizing unexpectedly succeeded")
    if ctypes.get_last_error() != 122 or required.value < 8:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = (ctypes.c_ubyte * required.value)()
    returned = wintypes.DWORD(required.value)
    if not query(0, ctypes.byref(buffer), ctypes.byref(returned)):
        raise ctypes.WinError(ctypes.get_last_error())
    if returned.value > required.value:
        raise RuntimeError("Windows processor topology exceeded its sized buffer")
    return _count_windows_processor_core_records(bytes(buffer[: returned.value]))


def _windows_cpu_model() -> str | None:
    """Read the stable Windows marketing name without conflating family/model IDs."""

    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            value, _kind = winreg.QueryValueEx(key, "ProcessorNameString")
    except (ImportError, OSError):
        return None
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split())
    return normalized or None


def _windows_live_thread_affinities() -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    """Snapshot this process's live threads and query each group-affinity mask."""

    from ctypes import wintypes

    class ThreadEntry32(ctypes.Structure):
        _fields_ = [
            ("size", wintypes.DWORD),
            ("usage", wintypes.DWORD),
            ("thread_id", wintypes.DWORD),
            ("owner_process_id", wintypes.DWORD),
            ("base_priority", wintypes.LONG),
            ("delta_priority", wintypes.LONG),
            ("flags", wintypes.DWORD),
        ]

    class GroupAffinity(ctypes.Structure):
        _fields_ = [
            ("mask", ctypes.c_size_t),
            ("group", wintypes.WORD),
            ("reserved", wintypes.WORD * 3),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_snapshot = kernel32.CreateToolhelp32Snapshot
    create_snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
    create_snapshot.restype = wintypes.HANDLE
    first = kernel32.Thread32First
    first.argtypes = (wintypes.HANDLE, ctypes.POINTER(ThreadEntry32))
    first.restype = wintypes.BOOL
    next_entry = kernel32.Thread32Next
    next_entry.argtypes = (wintypes.HANDLE, ctypes.POINTER(ThreadEntry32))
    next_entry.restype = wintypes.BOOL
    open_thread = kernel32.OpenThread
    open_thread.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    open_thread.restype = wintypes.HANDLE
    get_affinity = kernel32.GetThreadGroupAffinity
    get_affinity.argtypes = (wintypes.HANDLE, ctypes.POINTER(GroupAffinity))
    get_affinity.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    get_process_id = kernel32.GetCurrentProcessId
    get_process_id.restype = wintypes.DWORD

    invalid_handle = ctypes.c_void_p(-1).value
    snapshot = create_snapshot(0x00000004, 0)  # TH32CS_SNAPTHREAD
    if not snapshot or ctypes.cast(snapshot, ctypes.c_void_p).value == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    evidence: list[tuple[int, int, tuple[int, ...]]] = []
    try:
        entry = ThreadEntry32()
        entry.size = ctypes.sizeof(entry)
        if not first(snapshot, ctypes.byref(entry)):
            raise ctypes.WinError(ctypes.get_last_error())
        process_id = int(get_process_id())
        while True:
            if int(entry.owner_process_id) == process_id:
                thread = open_thread(0x0800, False, entry.thread_id)
                if thread:
                    try:
                        affinity = GroupAffinity()
                        if not get_affinity(thread, ctypes.byref(affinity)):
                            raise ctypes.WinError(ctypes.get_last_error())
                        processors = tuple(
                            index
                            for index in range(ctypes.sizeof(ctypes.c_size_t) * 8)
                            if int(affinity.mask) & (1 << index)
                        )
                        if not processors:
                            raise RuntimeError("Windows thread affinity mask is empty")
                        evidence.append(
                            (int(entry.thread_id), int(affinity.group), processors)
                        )
                    finally:
                        close_handle(thread)
                elif ctypes.get_last_error() != 87:  # thread exited after the snapshot
                    raise ctypes.WinError(ctypes.get_last_error())
            entry.size = ctypes.sizeof(entry)
            if next_entry(snapshot, ctypes.byref(entry)):
                continue
            if ctypes.get_last_error() != 18:  # ERROR_NO_MORE_FILES
                raise ctypes.WinError(ctypes.get_last_error())
            break
    finally:
        close_handle(snapshot)
    if not evidence:
        raise RuntimeError("no live Windows thread could be affinity-verified")
    return tuple(evidence)


def _current_thread_cpu_affinity() -> tuple[int, ...]:
    if os.name == "nt":
        _windows_single_processor_group()
        process_mask = ctypes.c_size_t()
        system_mask = ctypes.c_size_t()
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.GetProcessAffinityMask.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        kernel32.GetProcessAffinityMask.restype = ctypes.c_int
        if not kernel32.GetProcessAffinityMask(
            kernel32.GetCurrentProcess(),
            ctypes.byref(process_mask),
            ctypes.byref(system_mask),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return tuple(
            index
            for index in range(ctypes.sizeof(ctypes.c_size_t) * 8)
            if process_mask.value & (1 << index)
        )
    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:
        raise RuntimeError("CPU affinity inspection is unsupported on this platform")
    return tuple(sorted(int(processor) for processor in getter(0)))


def _set_process_cpu_affinity(processors: tuple[int, ...]) -> None:
    if not processors:
        raise ValueError("CPU affinity must not be empty")
    if os.name == "nt":
        _windows_single_processor_group()
        pointer_bits = ctypes.sizeof(ctypes.c_size_t) * 8
        if processors[-1] >= pointer_bits:
            raise ValueError(
                f"processor {processors[-1]} exceeds the {pointer_bits}-bit affinity mask"
            )
        mask = sum(1 << processor for processor in processors)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        if not kernel32.SetProcessAffinityMask(
            kernel32.GetCurrentProcess(),
            ctypes.c_size_t(mask),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        setter = getattr(os, "sched_setaffinity", None)
        if setter is None:
            raise RuntimeError("CPU affinity control is unsupported on this platform")
        setter(0, set(processors))
    if _current_thread_cpu_affinity() != processors:
        raise RuntimeError("CPU affinity reread differs from the requested processor set")


def _requested_affinity_argument(argv: list[str]) -> tuple[int, ...] | None:
    values: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--cpu-affinity":
            if index + 1 == len(argv):
                raise ValueError("--cpu-affinity requires a value")
            values.append(argv[index + 1])
            index += 2
            continue
        if argument.startswith("--cpu-affinity="):
            values.append(argument.split("=", maxsplit=1)[1])
        index += 1
    if not values:
        return None
    if len(values) != 1:
        raise ValueError("--cpu-affinity may be specified only once")
    return _parse_cpu_affinity(values[0])


def _bootstrap_cpu_affinity_before_native_imports() -> None:
    requested = _requested_affinity_argument(sys.argv[1:])
    if requested is None:
        return
    marker = ",".join(str(processor) for processor in requested)
    if os.environ.get(_AFFINITY_BOOTSTRAP_ENV) == marker:
        if _current_thread_cpu_affinity() != requested:
            raise RuntimeError("bootstrapped CPU affinity was not inherited")
        return
    _set_process_cpu_affinity(requested)
    os.environ[_AFFINITY_BOOTSTRAP_ENV] = marker
    os.execv(
        sys.executable,
        (sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]),
    )


if __name__ == "__main__":
    _bootstrap_cpu_affinity_before_native_imports()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.depth_surge_3d.processing.frames.stereo_generator import (  # noqa: E402
    STEREO_STAGE_ALGORITHM_VERSION,
)
from src.depth_surge_3d.rendering.forward_splat import HORIZONTAL_SUBPIXELS  # noqa: E402
from src.depth_surge_3d.rendering.quality.diagnostics import (  # noqa: E402
    build_quality_frame_stats,
)
from src.depth_surge_3d.rendering.quality.diagnostic_transaction import (  # noqa: E402
    QualityFramePublication,
    canonical_json_bytes as diagnostics_canonical_json_bytes,
    prepare_quality_frame,
    publish_quality_frame,
)
from src.depth_surge_3d.rendering.quality.renderer import (  # noqa: E402
    render_relative_quality_compact,
)
from src.depth_surge_3d.rendering.quality.native_kd import (  # noqa: E402
    native_binary_evidence,
)
from src.depth_surge_3d.rendering.quality.types import (  # noqa: E402
    CompactStereoRenderResult,
)
from src.depth_surge_3d.rendering.stereo_renderer import (  # noqa: E402
    StereoRenderer,
    StereoRenderSettings,
)
from scripts.quality_task0 import reporting as task0_reporting  # noqa: E402


REPORT_SCHEMA_VERSION = 1
REPORT_ALGORITHM_VERSION = "stereo-quality-production-gate-v1"
DEFAULT_FIXTURE_ROOT = Path(
    r"H:\3dtest\1787051840_f908a5f038277cf447b8a6a9b5072311_20260818_191720"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "docs/superpowers/reports/2026-08-24-stereo-quality-production.json"
)
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "output/stereo-quality-production-gate"
SOURCE_IDENTITY_NAME_BYTES = 8 * 1024 * 1024
SOURCE_IDENTITY_STREAM_BYTES = 64 * 1024
_TASK0_REPORT_RELATIVE_PATH = Path(
    "docs/superpowers/reports/2026-08-24-stereo-quality-task0.json"
)
_RELEASE_EVIDENCE_RELATIVE_PATHS = (
    "docs/superpowers/reports/2026-08-23-stereo-quality-task0.json",
    _TASK0_REPORT_RELATIVE_PATH.as_posix(),
    "docs/superpowers/reports/2026-08-24-stereo-quality-production.json",
)
_TASK0_RELEASE_RUN_GROUPS: dict[str, tuple[str, ...]] = {
    "task0_content_4k": tuple(
        f"content.cpu.4k.r{ratio}" for ratio in ("0_5", "1", "2", "4")
    ),
    "task0_geometry_4k_edge25": tuple(
        f"geometry.{device}.4k.r{ratio}.edge25"
        for device in ("cpu", "cuda0")
        for ratio in ("0_5", "1", "2", "4")
    ),
    "task0_region_4k_edge25": (
        "region.cpu.4k.edge25",
        "region.cuda0.4k.edge25",
    ),
    "task0_fallback_4k_edge25": (
        "fallback.cpu.4k.edge25",
        "fallback.cuda0.4k.edge25",
    ),
    "task0_capacity": ("control.cpu.n4194304",),
    "task0_encoding_phases": (
        "encoding.cpu.assembled.n4194304",
        "encoding.cpu.direct.n4194304",
    ),
    "task0_faults": ("fault.cpu.aggregate",),
    "task0_fixture": ("fixture.cpu.reported",),
    "task0_runtime": ("runtime.cpu.matrix",),
}
_BENCHMARK_RELEASE_GATE_NAMES = (
    "benchmark_sample_count",
    "renderer_1080_timing",
    "renderer_1080_ratio",
    "renderer_1080_determinism",
    "renderer_4k_timing",
    "renderer_4k_ratio",
    "renderer_4k_determinism",
    "full_pipeline_1080_timing",
    "full_pipeline_1080_determinism",
    "full_pipeline_4k_timing",
    "full_pipeline_4k_determinism",
    "gpu_allocation_with_headroom",
)


@dataclass(frozen=True)
class ReviewRegion:
    """One half-open full-resolution crop used for contour review."""

    name: str
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class FixtureFrame:
    """One hash-pinned source/canonical/Fast tuple."""

    number: int
    source_sha256: str
    canonical_sha256: str
    fast_left_sha256: str
    fast_right_sha256: str
    regions: tuple[ReviewRegion, ...]

    @property
    def stem(self) -> str:
        return f"frame_{self.number:06d}"

    @property
    def file_name(self) -> str:
        return f"{self.stem}.png"


@dataclass(frozen=True)
class VerificationConfig:
    """Injectable immutable fixture and release-gate settings."""

    settings_relative_path: str
    settings_raw_sha256: str
    settings_payload_sha256: str
    source_stage_fingerprint: str
    source_frame_fingerprint: str | None
    canonical_stage_fingerprint: str
    ordered_manifest_sha256: str | None
    frames: tuple[FixtureFrame, ...]
    render_width: int
    render_height: int
    stereo_strength: float
    convergence: float
    occlusion_fill_max_px: int
    performance_gate_enabled: bool = True
    quality_renderer_p95_limit_seconds: float = 5.0
    quality_to_fast_p95_limit: float = 12.0
    quality_renderer_4k_p95_limit_seconds: float = 20.0
    quality_pipeline_1080_p95_limit_seconds: float = 6.0
    quality_pipeline_4k_p95_limit_seconds: float = 24.0
    quality_gpu_bytes_per_pixel_limit: int = 1536


DEFAULT_CONFIG = VerificationConfig(
    settings_relative_path=("f908a5f038277cf447b8a6a9b5072311_1787051942-settings.json"),
    settings_raw_sha256="13b66e657877c8227b100f10265114f1a88162faff4f6a950d8b2d7234d41fbe",
    settings_payload_sha256=("861ba59c027f57c62b94460b23906d6ddcb7c0fde50c96e002f72ce9544180da"),
    source_stage_fingerprint="29ccafdddc83b6c408708b29c41e5d2fabc392028d7d4149a11bdb9b9fbc60c3",
    source_frame_fingerprint="af573cc60b900ab279466c11e53c9e64daca173b1818e30509d562e8485d78e4",
    canonical_stage_fingerprint=(
        "bfc9dca3f7a4ee61816df04e97b1fc56b89d54a536cb5b434862883d0a7ec7fa"
    ),
    ordered_manifest_sha256="ee9b8b6126667baa995e6c8055340e8494b38e4d6f9d65c64554d4389127c979",
    frames=(
        FixtureFrame(
            89,
            "cca5b9dd367ab23d4931c73ec98b1d091c431c69385e5d1077608f4ec3fd060b",
            "66d524dd394d82ad9b96c1fda89b725b1b9ff954a25b26fcd0ea0eed8138f77c",
            "9f2b60cbc1aee589ed00afa6d854eaaa009ae708acc62534a47dddcb68473bfd",
            "be77f8fc1f8552c4d8240350d1e1a5c6f1b7d89efca059b260eed5490426534a",
            (
                ReviewRegion("boot_edge", 1370, 210, 1650, 570),
                ReviewRegion("thin_ribbon", 930, 430, 1320, 720),
            ),
        ),
        FixtureFrame(
            111,
            "6e5213f560e651495c8df6f98663b8d01190403ca55fa51d811415ce8131a11c",
            "cf63208d7f948274ecdbd14445b19b557efe13cf329bfb35879a01bfc22d5281",
            "306c73362fe4e6df290a7beff0330078f53ca152c2a736b1a4aa42192efb724b",
            "35f773da7ad214c918963144c9acf529ad0641141119007bf8c1c4a7299ab48f",
            (
                ReviewRegion("hat_feather", 0, 320, 340, 700),
                ReviewRegion("hair_curl", 0, 650, 480, 930),
            ),
        ),
        FixtureFrame(
            171,
            "6a3f46b3d952ab44bb6dcff85364e97b40e320168ed9c52272fd087f32b310cb",
            "deb8512a101beee29fe08b53db2ac779ea864b47a1e2d21da355fafd22aa6fff",
            "4b57b0ae58ad6acf6484f7a66319d3a3e6fcbc7db82110e46041600a08095a0f",
            "b9c20330df16b3a47ddef7992a76b81ae431701598db1c2c37dd105d09b8fcf0",
            (
                ReviewRegion("hair_and_feather", 720, 140, 1050, 560),
                ReviewRegion("gun_boundary", 980, 280, 1450, 800),
            ),
        ),
        FixtureFrame(
            176,
            "154969bd5ff228c94bfd5f1b0b0f7c66531d55758abcdbd03de53d363192d4bb",
            "48e12f34b40345b81dcca7104e216cf5ae8a71614d8faed6a6a1f8061d55c1eb",
            "890cd5db85b17dfde53e2bcc466a179a25b6b2f9b4b786413a6f4fcd8f52979c",
            "c741c8bc7356ee8decd8c3178df46a3673dcee0813e2ef7a0697a2151e6f9c60",
            (
                ReviewRegion("hair_curl", 580, 250, 900, 540),
                ReviewRegion("gun_muzzle", 890, 280, 1370, 760),
                ReviewRegion("chest_ribbon", 450, 330, 900, 620),
            ),
        ),
        FixtureFrame(
            231,
            "be5f4a523a88386d0a9f781332a02cd6d9b716dc49e0504a052f91949f99a73b",
            "1fcf5555f1679e564dd73aaa324ee0c873468e2f0ba6165ccbf5a7dfb3bb8d35",
            "e48e5043538abb16f01d8eab2686bf62dfa3ffbaecee3a297f12403bd605f9c4",
            "e7efaa5096bb28d5763d398f690ec9d1493a299ea16e72d171574e9365b23db5",
            (
                ReviewRegion("long_gun_barrel", 500, 480, 1060, 800),
                ReviewRegion("twin_curls", 1120, 380, 1580, 730),
                ReviewRegion("chest_ribbon", 1240, 650, 1600, 900),
            ),
        ),
        FixtureFrame(
            301,
            "0d9182b0aba89afdfafd5adce9a7d2ca0db5742e8bc5b2db04d07f4f1884c932",
            "296511484ffddee5b6dd74ad8bc7b51e7da2757a7dc4444ac08fee9ae9f2745b",
            "23031b568c3868e3f22487cce73a3784ddc67b41c9457dcf627cff5f84ae5896",
            "eecf711d3a603d1e82c523df455e178a59ec47c10f9f6d2bde0f50c53fc275ea",
            (
                ReviewRegion("crossed_gun_barrels", 430, 160, 1300, 780),
                ReviewRegion("waist_ribbon", 940, 500, 1300, 850),
            ),
        ),
        FixtureFrame(
            401,
            "fdf65df83cfe7d647f66b651eeb9cabe0a53c83578a66aad8f7b40662293ba3a",
            "4cecaffe00821a84ff81f1422ebc2ef6b50269a0541eb4f23593a1fc80d20347",
            "2a554442e95657ca9b94c0524c4a65a732ea64f26610860c779d47eac6f2b065",
            "93b39254b0e685ac44ce91272ab723e076c3afe4dab4f0346acafd1a577ed1ca",
            (ReviewRegion("ring_guns_and_silhouette", 730, 170, 1190, 600),),
        ),
    ),
    render_width=1920,
    render_height=1080,
    stereo_strength=1.5,
    convergence=0.5,
    occlusion_fill_max_px=8,
)


@dataclass(frozen=True)
class _LoadedFrame:
    source: np.ndarray
    canonical: np.ndarray
    fast_left: np.ndarray
    fast_right: np.ndarray


@dataclass(frozen=True)
class _RenderedMode:
    compact: CompactStereoRenderResult
    seconds: float


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_report(path: Path, report: dict[str, Any]) -> None:
    report.pop("fingerprint", None)
    report["fingerprint"] = _canonical_hash(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_png(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write PNG: {path}")
    return _sha256(path)


def _read_colour(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode RGB PNG: {path}")
    return np.ascontiguousarray(image)


def _read_canonical(path: Path) -> np.ndarray:
    encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if encoded is None or encoded.dtype != np.uint16 or encoded.ndim != 2:
        raise ValueError(f"Canonical PNG must decode as uint16 [H,W]: {path}")
    result = np.ascontiguousarray(encoded.astype(np.float32))
    result *= np.float32(1.0 / 65535.0)
    return result


def _file_status(path: Path, expected_hash: str) -> dict[str, Any]:
    actual = _sha256(path) if path.is_file() else None
    status = (
        "missing" if actual is None else "verified" if actual == expected_hash else "hash_mismatch"
    )
    return {
        "path": str(path.resolve()),
        "expected_sha256": expected_hash,
        "actual_sha256": actual,
        "status": status,
    }


def _metadata_status(path: Path, field: str, expected: str | None) -> dict[str, Any]:
    try:
        value = _read_json(path)
        actual = value.get(field)
    except (OSError, json.JSONDecodeError, ValueError):
        actual = None
    verified = expected is None or actual == expected
    return {
        "path": str(path.resolve()),
        "field": field,
        "expected": expected,
        "actual": actual,
        "status": "verified" if verified else "metadata_mismatch",
    }


def _settings_status(fixture_root: Path, config: VerificationConfig) -> dict[str, Any]:
    path = fixture_root / config.settings_relative_path
    raw = _file_status(path, config.settings_raw_sha256)
    try:
        settings = _read_json(path)
        processing = settings["processing_settings"]
        properties = settings["video_properties"]
        payload_hash = _canonical_hash(processing)
        semantics_match = (
            isinstance(processing, dict)
            and isinstance(properties, dict)
            and payload_hash == config.settings_payload_sha256
            and properties.get("width") == config.render_width
            and properties.get("height") == config.render_height
            and processing.get("stereo_strength") == config.stereo_strength
            and processing.get("convergence") == config.convergence
            and processing.get("stereo_geometry_mode") == "relative"
            and processing.get("occlusion_fill") == "background"
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        payload_hash = None
        semantics_match = False
    raw["expected_payload_sha256"] = config.settings_payload_sha256
    raw["actual_payload_sha256"] = payload_hash
    raw["semantics_match"] = semantics_match
    if raw["status"] == "verified" and not semantics_match:
        raw["status"] = "settings_semantics_mismatch"
    return raw


def _frame_paths(fixture_root: Path, frame: FixtureFrame) -> dict[str, Path]:
    return {
        "source": fixture_root / "00_original_frames" / frame.file_name,
        "canonical": fixture_root / "03_disparity_maps" / frame.file_name,
        "fast_left": fixture_root / "04_left_frames" / frame.file_name,
        "fast_right": fixture_root / "04_right_frames" / frame.file_name,
    }


def _inspect_inputs(
    fixture_root: Path,
    config: VerificationConfig,
) -> tuple[dict[str, Any], bool]:
    settings = _settings_status(fixture_root, config)
    stages = {
        "source_stage": _metadata_status(
            fixture_root / "00_original_frames/metadata.json",
            "fingerprint",
            config.source_stage_fingerprint,
        ),
        "source_frames": _metadata_status(
            fixture_root / "00_original_frames/metadata.json",
            "source_frame_fingerprint",
            config.source_frame_fingerprint,
        ),
        "canonical_stage": _metadata_status(
            fixture_root / "03_disparity_maps/metadata.json",
            "fingerprint",
            config.canonical_stage_fingerprint,
        ),
    }
    frames: dict[str, Any] = {}
    for frame in config.frames:
        paths = _frame_paths(fixture_root, frame)
        frames[frame.stem] = {
            "source": _file_status(paths["source"], frame.source_sha256),
            "canonical": _file_status(paths["canonical"], frame.canonical_sha256),
            "fast_left": _file_status(paths["fast_left"], frame.fast_left_sha256),
            "fast_right": _file_status(paths["fast_right"], frame.fast_right_sha256),
        }
    statuses = [settings["status"]]
    statuses.extend(value["status"] for value in stages.values())
    statuses.extend(
        value["status"] for frame_values in frames.values() for value in frame_values.values()
    )
    inputs = {
        "settings": settings,
        "stages": stages,
        **frames,
        "spec_declared_ordered_manifest_sha256": config.ordered_manifest_sha256,
        "embedded_manifest_sha256": _canonical_hash(
            [
                {
                    "frame_name": frame.stem,
                    "source_sha256": frame.source_sha256,
                    "canonical_sha256": frame.canonical_sha256,
                    "fast_left_sha256": frame.fast_left_sha256,
                    "fast_right_sha256": frame.fast_right_sha256,
                }
                for frame in config.frames
            ]
        ),
    }
    return inputs, all(value == "verified" for value in statuses)


def _load_frame(
    fixture_root: Path, frame: FixtureFrame, config: VerificationConfig
) -> _LoadedFrame:
    paths = _frame_paths(fixture_root, frame)
    loaded = _LoadedFrame(
        source=_read_colour(paths["source"]),
        canonical=_read_canonical(paths["canonical"]),
        fast_left=_read_colour(paths["fast_left"]),
        fast_right=_read_colour(paths["fast_right"]),
    )
    expected_rgb = (config.render_height, config.render_width, 3)
    for name, value in (
        ("source", loaded.source),
        ("fast_left", loaded.fast_left),
        ("fast_right", loaded.fast_right),
    ):
        if value.dtype != np.uint8 or value.shape != expected_rgb:
            raise ValueError(
                f"{frame.stem} {name} shape/dtype changed: {value.shape}/{value.dtype}"
            )
    if loaded.canonical.dtype != np.float32 or loaded.canonical.ndim != 2:
        raise ValueError(f"{frame.stem} canonical shape/dtype changed")
    return loaded


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA verification requested but CUDA is unavailable")
    return device


def _synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _time_render(
    operation: Callable[[], CompactStereoRenderResult],
    device: str,
) -> _RenderedMode:
    _synchronize(device)
    started = time.perf_counter()
    compact = operation()
    _synchronize(device)
    return _RenderedMode(compact=compact, seconds=time.perf_counter() - started)


def _nearest_axis(source_size: int, destination_size: int) -> np.ndarray:
    coordinates = np.arange(destination_size, dtype=np.float64)
    coordinates += np.float64(0.5)
    coordinates *= np.float64(source_size)
    coordinates /= np.float64(destination_size)
    coordinates -= np.float64(0.5)
    indexes = np.floor(coordinates + np.float64(0.5)).astype(np.int64)
    return np.clip(indexes, 0, source_size - 1)


def _nearest_half_pixel_upsample(
    source: np.ndarray,
    render_shape: tuple[int, int],
) -> np.ndarray:
    """Apply the explicit half-pixel nearest diagnostic without a library resize."""

    if source.ndim != 2 or source.shape[0] <= 0 or source.shape[1] <= 0:
        raise ValueError("nearest diagnostic source must be a non-empty 2D raster")
    rows = _nearest_axis(source.shape[0], render_shape[0])
    columns = _nearest_axis(source.shape[1], render_shape[1])
    return np.ascontiguousarray(source[rows[:, None], columns[None, :]])


def _settings(
    config: VerificationConfig,
    *,
    mode: Literal["fast", "quality"],
    fill: Literal["none", "background"],
) -> StereoRenderSettings:
    return StereoRenderSettings(
        stereo_strength=config.stereo_strength,
        convergence=config.convergence,
        occlusion_fill=fill,
        stereo_render_mode=mode,
        occlusion_fill_max_px=config.occlusion_fill_max_px,
    )


def _render_modes(
    renderer: StereoRenderer,
    loaded: _LoadedFrame,
    config: VerificationConfig,
    device: str,
) -> dict[str, _RenderedMode]:
    fast_settings = _settings(config, mode="fast", fill="background")
    quality_settings = _settings(config, mode="quality", fill="background")
    no_fill_settings = _settings(config, mode="quality", fill="none")
    nearest = _nearest_half_pixel_upsample(loaded.canonical, loaded.source.shape[:2])
    with torch.inference_mode():
        modes = {
            "fast": _time_render(
                lambda: renderer.render_relative_fast_compact(
                    loaded.source, loaded.canonical, fast_settings
                ),
                device,
            ),
            "quality": _time_render(
                lambda: render_relative_quality_compact(
                    renderer=renderer,
                    frame=loaded.source,
                    canonical=loaded.canonical,
                    settings=quality_settings,
                ),
                device,
            ),
            "quality_without_fill": _time_render(
                lambda: render_relative_quality_compact(
                    renderer=renderer,
                    frame=loaded.source,
                    canonical=loaded.canonical,
                    settings=no_fill_settings,
                ),
                device,
            ),
            "nearest": _time_render(
                lambda: renderer.render_relative_fast_compact(
                    loaded.source, nearest, fast_settings
                ),
                device,
            ),
        }
    return modes


def _coverage_visual(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    def colourize(values: np.ndarray) -> np.ndarray:
        output = np.full((*values.shape, 3), (32, 32, 32), dtype=np.uint8)
        partial = (values > 0) & (values < HORIZONTAL_SUBPIXELS)
        output[partial] = (0, 180, 255)
        output[values == 0] = (255, 0, 255)
        return output

    return np.concatenate((colourize(left), colourize(right)), axis=1)


def _repair_visual(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    def colourize(bits: np.ndarray) -> np.ndarray:
        output = np.full((*bits.shape, 3), (32, 32, 32), dtype=np.uint8)
        output[(bits & np.uint8(0x01)) != 0] = (0, 180, 255)
        output[(bits & np.uint8(0x02)) != 0] = (0, 96, 255)
        output[(bits & np.uint8(0x04)) != 0] = (0, 220, 0)
        output[(bits & np.uint8(0x10)) != 0] = (220, 220, 0)
        output[(bits & np.uint8(0x20)) != 0] = (220, 0, 220)
        output[(bits & np.uint8(0x40)) != 0] = (0, 0, 255)
        return output

    return np.concatenate((colourize(left), colourize(right)), axis=1)


def _crop_400(image: np.ndarray, region: ReviewRegion) -> np.ndarray:
    crop = image[region.y0 : region.y1, region.x0 : region.x1]
    if crop.size == 0:
        raise ValueError(f"Review region is empty: {region.name}")
    return cv2.resize(crop, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST)


def _mode_stats(
    name: str,
    rendered: _RenderedMode,
    frame_name: str,
) -> dict[str, Any]:
    fill = "none" if name == "quality_without_fill" else "background"
    render_mode = "quality" if name.startswith("quality") else "fast"
    compact = rendered.compact
    stats = build_quality_frame_stats(
        compact,
        frame_name=frame_name,
        geometry_mode="relative",
        occlusion_fill=fill,
        render_mode=render_mode,
    )
    serialized_stats = json.loads(diagnostics_canonical_json_bytes(stats))
    return {
        "render_seconds": rendered.seconds,
        "left_pixel_sha256": _array_sha256(compact.left_image),
        "right_pixel_sha256": _array_sha256(compact.right_image),
        "diagnostics": serialized_stats,
    }


def _write_mode_artifacts(
    artifacts_dir: Path,
    frame: FixtureFrame,
    modes: dict[str, _RenderedMode],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for name, rendered in modes.items():
        compact = rendered.compact
        eyes: dict[str, Any] = {}
        for eye, image in (("left", compact.left_image), ("right", compact.right_image)):
            path = (artifacts_dir / "full" / f"{frame.stem}_{name}_{eye}.png").resolve()
            eyes[eye] = {"path": str(path), "png_sha256": _write_png(path, image)}
        output[name] = eyes
    return output


def _region_report(
    artifacts_dir: Path,
    frame: FixtureFrame,
    modes: dict[str, _RenderedMode],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for region in frame.regions:
        paths: dict[str, str] = {}
        quality = modes["quality"].compact
        fast = modes["fast"].compact
        for mode_name, rendered in modes.items():
            for eye, image in (
                ("left", rendered.compact.left_image),
                ("right", rendered.compact.right_image),
            ):
                key = f"{mode_name}_{eye}"
                path = (
                    artifacts_dir / "crops" / f"{frame.stem}_{region.name}_{key}_400pct.png"
                ).resolve()
                _write_png(path, _crop_400(image, region))
                paths[key] = str(path)
        ys = slice(region.y0, region.y1)
        xs = slice(region.x0, region.x1)
        left_difference = np.abs(
            quality.left_image[ys, xs].astype(np.int16) - fast.left_image[ys, xs].astype(np.int16)
        )
        right_difference = np.abs(
            quality.right_image[ys, xs].astype(np.int16) - fast.right_image[ys, xs].astype(np.int16)
        )
        reports.append(
            {
                "name": region.name,
                "bounds": [region.x0, region.y0, region.x1, region.y1],
                "artifacts": paths,
                "quality_vs_fast_mae": {
                    "left": float(left_difference.mean()),
                    "right": float(right_difference.mean()),
                },
                "quality_repaired_pixel_count": {
                    "left": int(np.count_nonzero(quality.left_repair_bits[ys, xs] & 0x34)),
                    "right": int(np.count_nonzero(quality.right_repair_bits[ys, xs] & 0x34)),
                },
            }
        )
    return reports


def _write_diagnostics(
    artifacts_dir: Path,
    frame: FixtureFrame,
    quality: CompactStereoRenderResult,
) -> dict[str, str]:
    coverage_path = (artifacts_dir / "diagnostics" / f"{frame.stem}_coverage.png").resolve()
    repair_path = (artifacts_dir / "diagnostics" / f"{frame.stem}_repair.png").resolve()
    _write_png(
        coverage_path,
        _coverage_visual(quality.left_coverage_count, quality.right_coverage_count),
    )
    _write_png(
        repair_path,
        _repair_visual(quality.left_repair_bits, quality.right_repair_bits),
    )
    return {"coverage_path": str(coverage_path), "repair_path": str(repair_path)}


def _frame_machine_checks(
    loaded: _LoadedFrame,
    modes: dict[str, _RenderedMode],
    artifacts: dict[str, dict[str, Any]],
    frame: FixtureFrame,
) -> dict[str, bool]:
    fast = modes["fast"].compact
    quality = modes["quality"].compact
    no_fill = modes["quality_without_fill"].compact
    expected_shape = loaded.source.shape
    all_images = [
        image
        for rendered in modes.values()
        for image in (rendered.compact.left_image, rendered.compact.right_image)
    ]
    return {
        "fast_left_pixels_equal": np.array_equal(fast.left_image, loaded.fast_left),
        "fast_right_pixels_equal": np.array_equal(fast.right_image, loaded.fast_right),
        "fast_left_png_equal": artifacts["fast"]["left"]["png_sha256"] == frame.fast_left_sha256,
        "fast_right_png_equal": artifacts["fast"]["right"]["png_sha256"] == frame.fast_right_sha256,
        "all_output_shapes_and_dtypes_valid": all(
            value.shape == expected_shape and value.dtype == np.uint8 for value in all_images
        ),
        "quality_final_unresolved_pixels_zero": not bool(
            np.any(quality.left_repair_bits & np.uint8(0x40))
            or np.any(quality.right_repair_bits & np.uint8(0x40))
        ),
        "quality_without_fill_exposes_geometry_holes": bool(
            np.any(no_fill.left_repair_bits & np.uint8(0x40))
            or np.any(no_fill.right_repair_bits & np.uint8(0x40))
        ),
        "nearest_diagnostic_is_distinct": bool(
            not np.array_equal(modes["nearest"].compact.left_image, fast.left_image)
            or not np.array_equal(modes["nearest"].compact.right_image, fast.right_image)
        ),
    }


def _evaluate_frame(
    fixture_root: Path,
    artifacts_dir: Path,
    renderer: StereoRenderer,
    frame: FixtureFrame,
    config: VerificationConfig,
    device: str,
) -> tuple[dict[str, Any], dict[str, bool]]:
    loaded = _load_frame(fixture_root, frame, config)
    modes = _render_modes(renderer, loaded, config, device)
    artifacts = _write_mode_artifacts(artifacts_dir, frame, modes)
    checks = _frame_machine_checks(loaded, modes, artifacts, frame)
    report = {
        "frame_name": frame.stem,
        "modes": {
            name: {**_mode_stats(name, rendered, frame.stem), "artifacts": artifacts[name]}
            for name, rendered in modes.items()
        },
        "diagnostics": _write_diagnostics(artifacts_dir, frame, modes["quality"].compact),
        "regions": _region_report(artifacts_dir, frame, modes),
        "machine_checks": checks,
    }
    del loaded, modes
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return report, checks


def _linear_p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("value must be positive")
    return parsed


def _benchmark_mode(
    operation: Callable[[], CompactStereoRenderResult],
    *,
    device: str,
    warmups: int,
    measured: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        _time_render(operation, device)
    timings: list[float] = []
    identities: set[tuple[str, str]] = set()
    for _ in range(measured):
        result = _time_render(operation, device)
        timings.append(result.seconds)
        identities.add(
            (
                _array_sha256(result.compact.left_image),
                _array_sha256(result.compact.right_image),
            )
        )
    return {
        "warmup_count": warmups,
        "measured_count": measured,
        "seconds": timings,
        "p95_seconds": _linear_p95(timings),
        "deterministic_output": len(identities) <= 1,
        "output_identities": [list(value) for value in sorted(identities)],
    }


def _benchmark_quality_pipeline_mode(
    operation: Callable[[], CompactStereoRenderResult],
    *,
    device: str,
    warmups: int,
    measured: int,
    expected_shape: tuple[int, int, int],
    benchmark_root: Path,
) -> dict[str, Any]:
    """Measure render plus the production diagnostics publication transaction."""

    benchmark_root.mkdir(parents=True, exist_ok=True)
    left_path = benchmark_root / "04_left_frames" / "benchmark.png"
    right_path = benchmark_root / "04_right_frames" / "benchmark.png"

    def iteration() -> tuple[float, str, int, bool, int]:
        _synchronize(device)
        gpu_baseline = 0
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            gpu_baseline = int(torch.cuda.memory_allocated())
        started = time.perf_counter()
        compact = operation()
        output_shape_valid = all(
            value.shape == (expected_shape if value.ndim == 3 else expected_shape[:2])
            for value in (
                compact.left_image,
                compact.right_image,
                compact.left_coverage_count,
                compact.right_coverage_count,
                compact.left_repair_bits,
                compact.right_repair_bits,
            )
        )
        prepared = prepare_quality_frame(
            QualityFramePublication(
                job_root=benchmark_root,
                frame_name="benchmark",
                left_rgb_path=left_path,
                right_rgb_path=right_path,
                compact=compact,
                geometry_mode="relative",
                occlusion_fill="background",
                rgb_stage_fingerprint="0" * 64,
                keep_masks=True,
            )
        )
        manifest = publish_quality_frame(prepared)
        manifest_raw = diagnostics_canonical_json_bytes(manifest)
        digest = hashlib.sha256(manifest_raw)
        published_byte_count = len(manifest_raw) + sum(
            int(identity["byte_count"])
            for identity in manifest["payloads"].values()
            if identity is not None
        )
        digest.update(_array_sha256(compact.left_image).encode("ascii"))
        digest.update(_array_sha256(compact.right_image).encode("ascii"))
        _synchronize(device)
        seconds = time.perf_counter() - started
        gpu_peak = (
            max(0, int(torch.cuda.max_memory_allocated()) - gpu_baseline)
            if device == "cuda"
            else 0
        )
        identity = digest.hexdigest()
        del compact, prepared, manifest
        return seconds, identity, gpu_peak, output_shape_valid, published_byte_count

    for _ in range(warmups):
        iteration()
    timings: list[float] = []
    identities: set[str] = set()
    gpu_peaks: list[int] = []
    published_byte_counts: list[int] = []
    shapes_valid = True
    for _ in range(measured):
        seconds, identity, gpu_peak, shape_valid, published_byte_count = iteration()
        timings.append(seconds)
        identities.add(identity)
        gpu_peaks.append(gpu_peak)
        published_byte_counts.append(published_byte_count)
        shapes_valid = shapes_valid and shape_valid
    return {
        "warmup_count": warmups,
        "measured_count": measured,
        "seconds": timings,
        "p95_seconds": _linear_p95(timings),
        "deterministic_output": len(identities) <= 1,
        "output_identities": sorted(identities),
        "gpu_peak_allocated_bytes": max(gpu_peaks, default=0),
        "output_shape_valid": shapes_valid,
        "published_diagnostic_byte_counts": published_byte_counts,
    }


def _run_benchmark(
    fixture_root: Path,
    artifacts_dir: Path,
    renderer: StereoRenderer,
    config: VerificationConfig,
    device: str,
    warmups: int,
    measured: int,
) -> dict[str, Any]:
    if measured == 0:
        return {
            "status": "disabled",
            "performance_gate_enabled": config.performance_gate_enabled,
            "passed": not config.performance_gate_enabled,
            "gates": {
                name: not config.performance_gate_enabled
                for name in _BENCHMARK_RELEASE_GATE_NAMES
            },
        }
    loaded = _load_frame(fixture_root, config.frames[0], config)
    fast_settings = _settings(config, mode="fast", fill="background")
    quality_settings = _settings(config, mode="quality", fill="background")
    source_4k = cv2.resize(
        loaded.source,
        (3840, 2160),
        interpolation=cv2.INTER_NEAREST,
    )
    canonical_4k = cv2.resize(
        loaded.canonical,
        (loaded.canonical.shape[1] * 2, loaded.canonical.shape[0] * 2),
        interpolation=cv2.INTER_NEAREST,
    )
    source_4k = np.ascontiguousarray(source_4k)
    canonical_4k = np.ascontiguousarray(canonical_4k, dtype=np.float32)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".stereo-quality-benchmark-",
        dir=artifacts_dir,
    ) as temporary_root, torch.inference_mode():
        benchmark_root = Path(temporary_root)
        fast_1080 = _benchmark_mode(
            lambda: renderer.render_relative_fast_compact(
                loaded.source, loaded.canonical, fast_settings
            ),
            device=device,
            warmups=warmups,
            measured=measured,
        )
        quality_1080 = _benchmark_mode(
            lambda: render_relative_quality_compact(
                renderer=renderer,
                frame=loaded.source,
                canonical=loaded.canonical,
                settings=quality_settings,
            ),
            device=device,
            warmups=warmups,
            measured=measured,
        )
        pipeline_1080 = _benchmark_quality_pipeline_mode(
            lambda: render_relative_quality_compact(
                renderer=renderer,
                frame=loaded.source,
                canonical=loaded.canonical,
                settings=quality_settings,
            ),
            device=device,
            warmups=warmups,
            measured=measured,
            expected_shape=loaded.source.shape,
            benchmark_root=benchmark_root / "1080p",
        )
        fast_4k = _benchmark_mode(
            lambda: renderer.render_relative_fast_compact(
                source_4k, canonical_4k, fast_settings
            ),
            device=device,
            warmups=warmups,
            measured=measured,
        )
        quality_4k = _benchmark_mode(
            lambda: render_relative_quality_compact(
                renderer=renderer,
                frame=source_4k,
                canonical=canonical_4k,
                settings=quality_settings,
            ),
            device=device,
            warmups=warmups,
            measured=measured,
        )
        pipeline_4k = _benchmark_quality_pipeline_mode(
            lambda: render_relative_quality_compact(
                renderer=renderer,
                frame=source_4k,
                canonical=canonical_4k,
                settings=quality_settings,
            ),
            device=device,
            warmups=warmups,
            measured=measured,
            expected_shape=source_4k.shape,
            benchmark_root=benchmark_root / "4k",
        )
    ratio_1080 = (
        quality_1080["p95_seconds"] / fast_1080["p95_seconds"]
        if fast_1080["p95_seconds"]
        else None
    )
    ratio_4k = (
        quality_4k["p95_seconds"] / fast_4k["p95_seconds"]
        if fast_4k["p95_seconds"]
        else None
    )
    count_valid = warmups >= 5 and measured >= 30
    pixel_count_4k = 3840 * 2160
    gpu_peak = max(
        pipeline_1080["gpu_peak_allocated_bytes"],
        pipeline_4k["gpu_peak_allocated_bytes"],
    )
    gpu_with_headroom = (gpu_peak * 5 + 3) // 4
    gates = {
        "benchmark_sample_count": count_valid,
        "renderer_1080_timing": (
            quality_1080["p95_seconds"]
            <= config.quality_renderer_p95_limit_seconds
        ),
        "renderer_1080_ratio": (
            ratio_1080 is not None and ratio_1080 <= config.quality_to_fast_p95_limit
        ),
        "renderer_1080_determinism": bool(
            fast_1080["deterministic_output"] and quality_1080["deterministic_output"]
        ),
        "renderer_4k_timing": (
            quality_4k["p95_seconds"]
            <= config.quality_renderer_4k_p95_limit_seconds
        ),
        "renderer_4k_ratio": (
            ratio_4k is not None and ratio_4k <= config.quality_to_fast_p95_limit
        ),
        "renderer_4k_determinism": bool(
            fast_4k["deterministic_output"] and quality_4k["deterministic_output"]
        ),
        "full_pipeline_1080_timing": (
            pipeline_1080["p95_seconds"]
            <= config.quality_pipeline_1080_p95_limit_seconds
        ),
        "full_pipeline_1080_determinism": bool(
            pipeline_1080["deterministic_output"]
            and pipeline_1080["output_shape_valid"]
        ),
        "full_pipeline_4k_timing": (
            pipeline_4k["p95_seconds"]
            <= config.quality_pipeline_4k_p95_limit_seconds
        ),
        "full_pipeline_4k_determinism": bool(
            pipeline_4k["deterministic_output"]
            and pipeline_4k["output_shape_valid"]
        ),
        "gpu_allocation_with_headroom": bool(
            device == "cuda"
            and gpu_with_headroom
            <= config.quality_gpu_bytes_per_pixel_limit * pixel_count_4k
        ),
    }
    passed = all(gates.values())
    return {
        "status": "passed" if passed else "failed",
        "performance_gate_enabled": config.performance_gate_enabled,
        "resolutions": {
            "1080p": {
                "fast_renderer": fast_1080,
                "quality_renderer": quality_1080,
                "quality_full_pipeline": pipeline_1080,
                "quality_to_fast_p95_ratio": ratio_1080,
            },
            "4k": {
                "fast_renderer": fast_4k,
                "quality_renderer": quality_4k,
                "quality_full_pipeline": pipeline_4k,
                "quality_to_fast_p95_ratio": ratio_4k,
            },
        },
        "gpu_peak_allocated_bytes": gpu_peak,
        "gpu_peak_with_25_percent_headroom_bytes": gpu_with_headroom,
        "limits": {
            "minimum_warmups": 5,
            "minimum_measured": 30,
            "quality_renderer_1080_p95_seconds": config.quality_renderer_p95_limit_seconds,
            "quality_renderer_4k_p95_seconds": config.quality_renderer_4k_p95_limit_seconds,
            "quality_full_pipeline_1080_p95_seconds": (
                config.quality_pipeline_1080_p95_limit_seconds
            ),
            "quality_full_pipeline_4k_p95_seconds": (
                config.quality_pipeline_4k_p95_limit_seconds
            ),
            "quality_to_fast_p95_ratio": config.quality_to_fast_p95_limit,
            "quality_gpu_bytes_per_pixel": config.quality_gpu_bytes_per_pixel_limit,
        },
        "gates": gates,
        "passed": passed or not config.performance_gate_enabled,
    }


def _load_unpinned_pair(
    fixture_root: Path, number: int
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    name = f"frame_{number:06d}.png"
    source_path = fixture_root / "00_original_frames" / name
    canonical_path = fixture_root / "03_disparity_maps" / name
    return (
        _read_colour(source_path),
        _read_canonical(canonical_path),
        {"source_sha256": _sha256(source_path), "canonical_sha256": _sha256(canonical_path)},
    )


def _temporal_window(
    fixture_root: Path,
    artifacts_dir: Path,
    renderer: StereoRenderer,
    frame: FixtureFrame,
    config: VerificationConfig,
    device: str,
    radius: int,
) -> dict[str, Any]:
    settings = _settings(config, mode="quality", fill="background")
    crops: dict[tuple[str, str], list[np.ndarray]] = {
        (region.name, eye): [] for region in frame.regions for eye in ("left", "right")
    }
    entries: list[dict[str, Any]] = []
    for number in range(frame.number - radius, frame.number + radius + 1):
        source, canonical, hashes = _load_unpinned_pair(fixture_root, number)
        if number == frame.number:
            left = _read_colour(artifacts_dir / "full" / f"{frame.stem}_quality_left.png")
            right = _read_colour(artifacts_dir / "full" / f"{frame.stem}_quality_right.png")
            seconds = None
        else:
            rendered = _time_render(
                lambda: render_relative_quality_compact(
                    renderer=renderer,
                    frame=source,
                    canonical=canonical,
                    settings=settings,
                ),
                device,
            )
            left = rendered.compact.left_image
            right = rendered.compact.right_image
            seconds = rendered.seconds
        for region in frame.regions:
            crops[(region.name, "left")].append(_crop_400(left, region))
            crops[(region.name, "right")].append(_crop_400(right, region))
        entries.append({"frame_name": f"frame_{number:06d}", **hashes, "render_seconds": seconds})
        del left, right
    artifacts: dict[str, str] = {}
    for (region_name, eye), sequence in crops.items():
        path = (
            artifacts_dir
            / "temporal"
            / f"{frame.stem}_{region_name}_quality_{eye}_window_400pct.png"
        ).resolve()
        _write_png(path, np.concatenate(sequence, axis=1))
        artifacts[f"{region_name}_{eye}"] = str(path)
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"radius": radius, "frames": entries, "artifacts": artifacts}


def _run_temporal_windows(
    fixture_root: Path,
    artifacts_dir: Path,
    renderer: StereoRenderer,
    config: VerificationConfig,
    device: str,
    radius: int,
) -> list[dict[str, Any]]:
    if radius == 0:
        return []
    return [
        {
            "target_frame": frame.stem,
            **_temporal_window(
                fixture_root,
                artifacts_dir,
                renderer,
                frame,
                config,
                device,
                radius,
            ),
        }
        for frame in config.frames
    ]


def _hash_git_stdout(command: tuple[str, ...], digest: Any) -> int:
    try:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError as error:
        raise RuntimeError(f"source identity command could not start: {command[1]}") from error
    stream = process.stdout
    assert stream is not None
    byte_count = 0
    for chunk in iter(lambda: stream.read(SOURCE_IDENTITY_STREAM_BYTES), b""):
        byte_count += len(chunk)
        digest.update(chunk)
    if process.wait() != 0:
        raise RuntimeError(f"source identity command failed: {command[1]}")
    return byte_count


def _terminate_and_reap(process: subprocess.Popen[bytes]) -> None:
    try:
        process.terminate()
    except OSError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        finally:
            process.wait()


def _hash_untracked_git_files(digest: Any) -> tuple[int, int]:
    """Stream Git's NUL-framed untracked names directly into file hashing."""

    try:
        process = subprocess.Popen(
            ("git", "ls-files", "--others", "--exclude-standard", "-z"),
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError as error:
        raise RuntimeError("untracked source files could not be enumerated") from error
    stream = process.stdout
    assert stream is not None
    pending = bytearray()
    total_bytes = 0
    file_count = 0
    try:
        while True:
            remaining_with_sentinel = SOURCE_IDENTITY_NAME_BYTES + 1 - total_bytes
            if remaining_with_sentinel <= 0:
                raise RuntimeError("untracked source-name list exceeds its byte budget")
            chunk = stream.read(min(SOURCE_IDENTITY_STREAM_BYTES, remaining_with_sentinel))
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > SOURCE_IDENTITY_NAME_BYTES:
                raise RuntimeError("untracked source-name list exceeds its byte budget")
            pending.extend(chunk)
            consumed = 0
            while True:
                boundary = pending.find(0, consumed)
                if boundary < 0:
                    if consumed:
                        del pending[:consumed]
                    break
                raw_name = bytes(pending[consumed:boundary])
                consumed = boundary + 1
                if raw_name:
                    _hash_untracked_file(raw_name, digest)
                    file_count += 1
        if pending:
            raise RuntimeError("untracked source-name stream is not NUL-terminated")
        if process.wait() != 0:
            raise RuntimeError("untracked source files could not be enumerated")
    except BaseException:
        _terminate_and_reap(process)
        raise
    return file_count, total_bytes


def _hash_untracked_file(raw_name: bytes, digest: Any) -> None:
    try:
        decoded = raw_name.decode("utf-8", errors="surrogateescape")
    except UnicodeDecodeError as error:
        raise RuntimeError("untracked source name cannot be decoded") from error
    pure = PurePosixPath(decoded)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise RuntimeError("untracked source name is not repository-contained")
    path = PROJECT_ROOT.joinpath(*pure.parts)
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"untracked source entry is not a regular file: {decoded}")
        content_digest = hashlib.sha256()
        byte_count = 0
        with path.open("rb", buffering=0) as handle:
            opened = os.fstat(handle.fileno())
            if not stat.S_ISREG(opened.st_mode) or (
                int(opened.st_dev), int(opened.st_ino)
            ) != (int(before.st_dev), int(before.st_ino)):
                raise RuntimeError(f"untracked source changed before hashing: {decoded}")
            for chunk in iter(lambda: handle.read(SOURCE_IDENTITY_STREAM_BYTES), b""):
                byte_count += len(chunk)
                if byte_count > 0xFFFF_FFFF_FFFF_FFFF:
                    raise RuntimeError(f"untracked source is too large to frame: {decoded}")
                content_digest.update(chunk)
            after = os.fstat(handle.fileno())
        after_path = path.lstat()
        if (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_size),
            int(after.st_mtime_ns),
        ) != (
            int(opened.st_dev),
            int(opened.st_ino),
            int(opened.st_size),
            int(opened.st_mtime_ns),
        ):
            raise RuntimeError(f"untracked source changed while hashing: {decoded}")
        if byte_count != int(after.st_size) or (
            int(after_path.st_dev),
            int(after_path.st_ino),
            int(after_path.st_size),
            int(after_path.st_mtime_ns),
        ) != (
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_size),
            int(after.st_mtime_ns),
        ):
            raise RuntimeError(f"untracked source changed after hashing: {decoded}")
        digest.update(len(raw_name).to_bytes(8, "little"))
        digest.update(raw_name)
        digest.update(byte_count.to_bytes(8, "little"))
        digest.update(content_digest.digest())
    except OSError as error:
        raise RuntimeError(f"untracked source could not be hashed: {decoded}") from error


def _git_state() -> dict[str, Any]:
    try:
        commit_raw = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=PROJECT_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        commit = commit_raw.decode("ascii", errors="strict").strip()
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as error:
        raise RuntimeError("source commit could not be resolved") from error
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("source commit is not a lowercase SHA-1 identity")

    digest = hashlib.sha256(b"stereo-quality-source-v1\x00tracked\x00")
    tracked_bytes = _hash_git_stdout(
        ("git", "diff", "--binary", "--no-ext-diff", "HEAD", "--", "."),
        digest,
    )
    digest.update(b"\x00untracked\x00")
    untracked_count, _untracked_name_bytes = _hash_untracked_git_files(digest)
    dirty = tracked_bytes != 0 or untracked_count != 0
    return {
        "commit": commit,
        "dirty": dirty,
        "dirty_fingerprint": digest.hexdigest() if dirty else None,
        "release_tree_clean": _release_source_tree_clean_v1(),
    }


def _release_pathspec_v1() -> tuple[str, ...]:
    return tuple(
        f":(exclude){path}" for path in sorted(_RELEASE_EVIDENCE_RELATIVE_PATHS)
    )


def _release_source_tree_clean_v1() -> bool:
    """Ignore only the generated evidence files that necessarily follow a run."""

    try:
        tracked = subprocess.run(
            (
                "git",
                "diff",
                "--quiet",
                "HEAD",
                "--",
                ".",
                *_release_pathspec_v1(),
            ),
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError as error:
        raise RuntimeError("release source cleanliness could not be inspected") from error
    if tracked.returncode == 1:
        return False
    if tracked.returncode != 0:
        raise RuntimeError("release source tracked-state inspection failed")
    try:
        process = subprocess.Popen(
            (
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                ".",
                *_release_pathspec_v1(),
            ),
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError as error:
        raise RuntimeError("release source untracked-state inspection failed") from error
    stream = process.stdout
    assert stream is not None
    first = stream.read(1)
    if first:
        _terminate_and_reap(process)
        return False
    if process.wait() != 0:
        raise RuntimeError("release source untracked-state inspection failed")
    return True


def _task0_source_matches_candidate_v1(
    report_source: task0_reporting.SourceIdentity,
    current_source: task0_reporting.SourceIdentity,
) -> bool:
    """Allow a descendant commit only when it changes generated evidence files."""

    if report_source == current_source:
        return True
    if report_source.dirty_diff_sha256 != current_source.dirty_diff_sha256:
        return False
    try:
        ancestor = subprocess.run(
            (
                "git",
                "merge-base",
                "--is-ancestor",
                report_source.commit,
                current_source.commit,
            ),
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if ancestor.returncode == 1:
            return False
        if ancestor.returncode != 0:
            raise RuntimeError("Task 0 source ancestry inspection failed")
        source_diff = subprocess.run(
            (
                "git",
                "diff",
                "--quiet",
                report_source.commit,
                current_source.commit,
                "--",
                ".",
                *_release_pathspec_v1(),
            ),
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError as error:
        raise RuntimeError("Task 0 source ancestry could not be inspected") from error
    if source_diff.returncode not in {0, 1}:
        raise RuntimeError("Task 0 source descendant diff inspection failed")
    return source_diff.returncode == 0


def _physical_memory_bytes() -> int | None:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.total_physical)
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        line = _command_first_line(["sysctl", "-n", "hw.memsize"])
        return int(line) if line is not None and line.isdecimal() else None
    total = pages * page_size
    return total if pages > 0 and page_size > 0 and total > 0 else None


def _optional_psutil() -> Any | None:
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return None
    return psutil


def _physical_core_count() -> int | None:
    psutil = _optional_psutil()
    if psutil is not None:
        count = psutil.cpu_count(logical=False)
        if count is not None and int(count) > 0:
            return int(count)
    if os.name == "nt":
        try:
            return _windows_physical_core_count()
        except (OSError, RuntimeError):
            return None
    topology_root = Path("/sys/devices/system/cpu")
    identities: set[tuple[str, str]] = set()
    try:
        for cpu in topology_root.glob("cpu[0-9]*"):
            package = (cpu / "topology" / "physical_package_id").read_text().strip()
            core = (cpu / "topology" / "core_id").read_text().strip()
            identities.add((package, core))
    except OSError:
        identities.clear()
    if identities:
        return len(identities)
    line = _command_first_line(["sysctl", "-n", "hw.physicalcpu"])
    return int(line) if line is not None and line.isdecimal() and int(line) > 0 else None


def _process_cpu_affinity() -> list[int] | None:
    try:
        return list(_current_thread_cpu_affinity())
    except (OSError, RuntimeError):
        return None


def _quality_native_runtime_evidence() -> dict[str, object]:
    evidence = native_binary_evidence()
    native_source = PROJECT_ROOT / "native" / "quality_kd_native.cpp"
    candidate_source_sha256 = _sha256(native_source)
    build_info = evidence.get("build_info")
    if (
        not isinstance(build_info, dict)
        or build_info.get("source_sha256") != candidate_source_sha256
    ):
        raise RuntimeError(
            "loaded Quality native binary was not built from the candidate source"
        )
    evidence["candidate_source_path"] = str(native_source.resolve())
    evidence["candidate_source_sha256"] = candidate_source_sha256
    return evidence


def _affinity_evidence(requested: tuple[int, ...] | None) -> dict[str, Any]:
    thread_count = 0
    all_threads_verified = False
    effective: tuple[int, ...] | None
    windows_groups: list[dict[str, Any]] | None = None
    if os.name == "nt":
        group_count = _windows_processor_group_count()
        if group_count != 1:
            raise RuntimeError(
                "cross-group Windows thread affinity cannot be fully verified; "
                "CPU Sets inspection is required"
            )
        threads = _windows_live_thread_affinities()
        effective = _current_thread_cpu_affinity()
        target = effective if requested is None else requested
        if requested is not None and effective != requested:
            raise RuntimeError("effective CPU affinity differs from the requested set")
        for thread_id, group, actual in threads:
            if group != 0 or actual != target:
                raise RuntimeError(
                    f"thread {thread_id} escaped the effective CPU affinity"
                )
        thread_count = len(threads)
        all_threads_verified = True
    else:
        effective = _current_thread_cpu_affinity()
        target = effective if requested is None else requested
        if requested is not None and effective != requested:
            raise RuntimeError("effective CPU affinity differs from the requested set")
    if os.name != "nt" and Path("/proc/self/task").is_dir():
        getter = getattr(os, "sched_getaffinity", None)
        if getter is None:
            raise RuntimeError("per-thread CPU affinity inspection is unavailable")
        verified = 0
        for task in Path("/proc/self/task").iterdir():
            if not task.name.isdecimal():
                continue
            try:
                actual = tuple(sorted(int(cpu) for cpu in getter(int(task.name))))
            except ProcessLookupError:
                continue
            if actual != target:
                raise RuntimeError(
                    f"thread {task.name} escaped the effective CPU affinity"
                )
            verified += 1
        if verified == 0:
            raise RuntimeError("no live thread could be affinity-verified")
        thread_count = verified
        all_threads_verified = True
    return {
        "requested": None if requested is None else list(requested),
        "effective": None if effective is None else list(effective),
        "windows_thread_group_affinities": windows_groups,
        "all_threads_verified": all_threads_verified,
        "verified_thread_count": thread_count,
    }


def _command_first_line(argv: list[str]) -> str | None:
    try:
        result = subprocess.run(argv, check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    output = result.stdout.strip() or result.stderr.strip()
    return output.splitlines()[0] if output else None


def _runtime_metadata(
    device: str,
    requested_affinity: tuple[int, ...] | None,
) -> dict[str, Any]:
    gpu = torch.cuda.get_device_name() if device == "cuda" else None
    cpu_family_model = os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor()
    cpu_model = _windows_cpu_model() if os.name == "nt" else platform.processor()
    if not cpu_model and Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu_model = line.split(":", maxsplit=1)[1].strip()
                break
    physical_cores = _physical_core_count()
    logical_cores = os.cpu_count()
    physical_memory = _physical_memory_bytes()
    if (
        not cpu_model
        or physical_cores is None
        or logical_cores is None
        or logical_cores <= 0
        or physical_memory is None
        or physical_memory <= 0
    ):
        raise RuntimeError("required host runtime identity is unavailable")
    affinity_pre = _affinity_evidence(requested_affinity)
    native_evidence = _quality_native_runtime_evidence()
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_model": cpu_model,
        "cpu_family_model": cpu_family_model or None,
        "physical_core_count": physical_cores,
        "logical_core_count": logical_cores,
        "cpu_affinity_pre": affinity_pre,
        "physical_memory_bytes": physical_memory,
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "torch_version": torch.__version__,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "quality_native": native_evidence,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime_version": torch.version.cuda,
        "device": device,
        "gpu_model": gpu,
        "gpu_driver": _command_first_line(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ),
        "ffmpeg": _command_first_line(["ffmpeg", "-version"]),
        "frame_worker_count": 0,
    }


def _task0_release_evidence(
    native_evidence: dict[str, object],
    *,
    required: bool,
) -> dict[str, Any]:
    task0_path = PROJECT_ROOT / _TASK0_REPORT_RELATIVE_PATH
    if not required:
        return {
            "required": False,
            "valid": True,
            "path": str(task0_path.resolve()),
            "sha256": _sha256(task0_path) if task0_path.is_file() else None,
            "candidate_source_match": True,
            "native_build_match": True,
            "gates": {name: True for name in _TASK0_RELEASE_RUN_GROUPS},
            "resource_bounds": True,
            "error": None,
        }
    try:
        report = task0_reporting.load_report(task0_path)
        current_source = task0_reporting.capture_source_identity(
            PROJECT_ROOT,
            excluded_relative_paths=_RELEASE_EVIDENCE_RELATIVE_PATHS,
        )
        runs = {run.run_id: run for run in report.runs}
        group_gates = {
            name: all(
                run_id in runs and runs[run_id].status == "passed"
                for run_id in run_ids
            )
            for name, run_ids in _TASK0_RELEASE_RUN_GROUPS.items()
        }
        resource_bounds = all(
            run.status == "passed"
            and run.peak_rss_bytes <= run.rss_cap_bytes
            and run.facts.get("rss_cap_exceeded") is False
            and run.facts.get("arena_cap_exceeded") is False
            and run.facts.get("wall_cap_exceeded") is False
            for run in report.runs
        )
        build_info = native_evidence.get("build_info")
        expected_native_build = (
            {
                key: build_info[key]
                for key in ("algorithm", "kd_stack_capacity", "runtime_jit")
            }
            if isinstance(build_info, dict)
            and all(
                key in build_info
                for key in ("algorithm", "kd_stack_capacity", "runtime_jit")
            )
            else None
        )
        task0_native_builds = [
            run.facts["native_build"]
            for run in report.runs
            if isinstance(run.facts.get("native_build"), dict)
        ]
        native_build_match = bool(
            expected_native_build is not None
            and task0_native_builds
            and all(value == expected_native_build for value in task0_native_builds)
            and isinstance(build_info, dict)
            and build_info.get("source_sha256")
            == native_evidence.get("candidate_source_sha256")
        )
        candidate_source_match = _task0_source_matches_candidate_v1(
            report.source,
            current_source,
        )
        valid = (
            candidate_source_match
            and native_build_match
            and resource_bounds
            and all(group_gates.values())
        )
        return {
            "required": True,
            "valid": valid,
            "path": str(task0_path.resolve()),
            "sha256": _sha256(task0_path),
            "report_fingerprint": report.fingerprint,
            "candidate_source_match": candidate_source_match,
            "native_build_match": native_build_match,
            "gates": group_gates,
            "resource_bounds": resource_bounds,
            "error": None,
        }
    except (OSError, RuntimeError, ValueError) as error:
        return {
            "required": True,
            "valid": False,
            "path": str(task0_path.resolve()),
            "sha256": _sha256(task0_path) if task0_path.is_file() else None,
            "candidate_source_match": False,
            "native_build_match": False,
            "gates": {name: False for name in _TASK0_RELEASE_RUN_GROUPS},
            "resource_bounds": False,
            "error": f"{type(error).__name__}: {error}",
        }


def _base_report(
    fixture_root: Path,
    artifacts_dir: Path,
    inputs: dict[str, Any],
    config: VerificationConfig,
    device: str,
    benchmark_command: tuple[str, ...] | None,
    requested_affinity: tuple[int, ...] | None,
) -> dict[str, Any]:
    source = _git_state()
    runtime = _runtime_metadata(device, requested_affinity)
    task0 = _task0_release_evidence(
        runtime["quality_native"],
        required=config.performance_gate_enabled,
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "algorithm_version": REPORT_ALGORITHM_VERSION,
        "stereo_stage_algorithm_version": STEREO_STAGE_ALGORITHM_VERSION,
        "source": source,
        "runtime": runtime,
        "benchmark_commands": (
            [{"argv": list(benchmark_command), "cwd": str(Path.cwd().resolve())}]
            if benchmark_command is not None
            else []
        ),
        "fixture_root": str(fixture_root.resolve()),
        "artifacts_dir": str(artifacts_dir.resolve()),
        "inputs": inputs,
        "settings": {
            "render_shape": [config.render_height, config.render_width],
            "stereo_strength": config.stereo_strength,
            "convergence": config.convergence,
            "occlusion_fill": "background",
            "occlusion_fill_max_px": config.occlusion_fill_max_px,
            "stereo_render_modes": ["fast", "quality"],
            "diagnostic_modes": ["quality_without_fill", "nearest"],
        },
        "task0_report": task0,
    }


def _aggregate_gates(
    frame_checks: list[dict[str, bool]],
    benchmark: dict[str, Any],
    temporal_radius: int,
    *,
    release_required: bool,
    source: dict[str, Any],
    task0: dict[str, Any],
) -> dict[str, bool]:
    gates = {
        "fast_byte_compatibility": all(
            check["fast_left_pixels_equal"]
            and check["fast_right_pixels_equal"]
            and check["fast_left_png_equal"]
            and check["fast_right_png_equal"]
            for check in frame_checks
        ),
        "quality_final_unresolved_lanes_zero": all(
            check["quality_final_unresolved_pixels_zero"] for check in frame_checks
        ),
        "quality_without_fill_diagnostic_observed": all(
            check["quality_without_fill_exposes_geometry_holes"] for check in frame_checks
        ),
        "nearest_diagnostic_distinct": all(
            check["nearest_diagnostic_is_distinct"] for check in frame_checks
        ),
        "output_contract": all(
            check["all_output_shapes_and_dtypes_valid"] for check in frame_checks
        ),
        "source_tree_clean": bool(
            not release_required or source.get("release_tree_clean") is True
        ),
        "task0_candidate_source": bool(
            not release_required or task0.get("candidate_source_match") is True
        ),
        "task0_native_build": bool(
            not release_required or task0.get("native_build_match") is True
        ),
        "task0_resource_bounds": bool(
            not release_required or task0.get("resource_bounds") is True
        ),
        "temporal_windows_exported": temporal_radius >= 0,
    }
    benchmark_gates = benchmark.get("gates")
    for name in _BENCHMARK_RELEASE_GATE_NAMES:
        gates[name] = bool(
            not release_required
            or isinstance(benchmark_gates, dict)
            and benchmark_gates.get(name) is True
        )
    task0_gates = task0.get("gates")
    for name in _TASK0_RELEASE_RUN_GROUPS:
        gates[name] = bool(
            not release_required
            or isinstance(task0_gates, dict)
            and task0_gates.get(name) is True
        )
    return gates


def run_verification(  # noqa: C901
    fixture_root: Path,
    report_json: Path,
    artifacts_dir: Path,
    *,
    device: str = "auto",
    config: VerificationConfig = DEFAULT_CONFIG,
    benchmark_warmups: int = 5,
    benchmark_measured: int = 30,
    temporal_radius: int = 1,
    benchmark_command: tuple[str, ...] | None = None,
    cpu_affinity: tuple[int, ...] | None = None,
) -> int:
    """Run the release gate and always leave a machine-readable report."""

    if benchmark_warmups < 0 or benchmark_measured < 0 or temporal_radius < 0:
        raise ValueError("benchmark counts and temporal radius must be nonnegative")
    try:
        resolved_device = _resolve_device(device)
        inputs, available = _inspect_inputs(fixture_root, config)
        report = _base_report(
            fixture_root,
            artifacts_dir,
            inputs,
            config,
            resolved_device,
            benchmark_command,
            cpu_affinity,
        )
    except Exception as error:
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "algorithm_version": REPORT_ALGORITHM_VERSION,
            "status": "evaluation_error",
            "fixture_root": str(Path(fixture_root).resolve()),
            "artifacts_dir": str(Path(artifacts_dir).resolve()),
            "error": f"{type(error).__name__}: {error}",
        }
        _write_report(report_json, report)
        return 3
    if not available:
        report["status"] = "fixture_unavailable"
        _write_report(report_json, report)
        return 1

    try:
        renderer = StereoRenderer(device=resolved_device)
        frame_reports: list[dict[str, Any]] = []
        frame_checks: list[dict[str, bool]] = []
        for frame in config.frames:
            frame_report, checks = _evaluate_frame(
                fixture_root,
                artifacts_dir,
                renderer,
                frame,
                config,
                resolved_device,
            )
            frame_reports.append(frame_report)
            frame_checks.append(checks)
        benchmark = _run_benchmark(
            fixture_root,
            artifacts_dir,
            renderer,
            config,
            resolved_device,
            benchmark_warmups,
            benchmark_measured,
        )
        temporal = _run_temporal_windows(
            fixture_root,
            artifacts_dir,
            renderer,
            config,
            resolved_device,
            temporal_radius,
        )
        report["runtime"]["cpu_affinity_post"] = _affinity_evidence(cpu_affinity)
        gates = _aggregate_gates(
            frame_checks,
            benchmark,
            temporal_radius,
            release_required=config.performance_gate_enabled,
            source=report["source"],
            task0=report["task0_report"],
        )
    except Exception as error:  # The standalone gate must preserve failure evidence.
        report["status"] = "evaluation_error"
        report["error"] = f"{type(error).__name__}: {error}"
        _write_report(report_json, report)
        return 3

    report.update(
        {
            "frames": frame_reports,
            "benchmark": benchmark,
            "temporal_windows": temporal,
            "gates": gates,
            "human_review": {
                "agent_visual_review": "pending",
                "user_approval": "pending",
                "rejection_criteria": [
                    "background_stretch",
                    "soft_halo_wider_than_one_output_pixel",
                    "nearest_style_stair_step",
                    "missing_thin_structure",
                    "copied_foreground_streak",
                    "temporal_flicker",
                    "black_hole",
                    "cross_depth_donor_leakage",
                ],
            },
        }
    )
    if all(gates.values()):
        report["status"] = "machine_pass_human_review_required"
        exit_code = 0
    else:
        report["status"] = "machine_gate_failure"
        exit_code = 2
    _write_report(report_json, report)
    return exit_code


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", type=Path, default=DEFAULT_FIXTURE_ROOT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--benchmark-warmups", type=int, default=5)
    parser.add_argument("--benchmark-measured", type=int, default=30)
    parser.add_argument("--temporal-radius", type=int, default=1)
    parser.add_argument(
        "--cpu-affinity",
        type=_parse_cpu_affinity,
        help="Comma-separated logical processors or inclusive ranges (for example 0-15)",
    )
    parser.add_argument(
        "--torch-intraop-threads",
        type=_positive_int,
        help="Explicit Torch intra-op thread count for a reproducible reference run",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.cpu_affinity is not None:
        _set_process_cpu_affinity(args.cpu_affinity)
    if args.torch_intraop_threads is not None:
        torch.set_num_threads(args.torch_intraop_threads)
    exit_code = run_verification(
        args.fixture_root,
        args.report_json,
        args.artifacts_dir,
        device=args.device,
        benchmark_warmups=args.benchmark_warmups,
        benchmark_measured=args.benchmark_measured,
        temporal_radius=args.temporal_radius,
        benchmark_command=(
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ),
        cpu_affinity=args.cpu_affinity,
    )
    print(args.report_json.resolve())
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
