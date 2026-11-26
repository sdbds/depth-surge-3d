"""Cross-platform host RAM detection and dynamic memory budget scaling."""

from __future__ import annotations

import ctypes
import os


def _windows_available_memory() -> int | None:
    try:

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            avail = int(status.ullAvailPhys)
            return avail if avail > 0 else None
    except Exception:
        return None
    return None


def _proc_meminfo_available() -> int | None:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb * 1024 if kb > 0 else None
    except (OSError, ValueError):
        return None
    return None


def _sysconf_available() -> int | None:
    sysconf = getattr(os, "sysconf", None)
    if not callable(sysconf):
        return None
    try:
        page_size = int(sysconf("SC_PAGE_SIZE"))
        avail_pages = int(sysconf("SC_AVPHYS_PAGES"))
        if page_size > 0 and avail_pages > 0:
            return page_size * avail_pages
    except (OSError, ValueError, TypeError):
        return None
    return None


def available_host_memory_bytes() -> int | None:
    """Return available physical RAM in bytes on Windows, Linux, or macOS."""

    if os.name == "nt":
        return _windows_available_memory()
    avail = _proc_meminfo_available()
    if avail is not None:
        return avail
    return _sysconf_available()


def dynamic_host_budget_bytes(
    *,
    fraction: float = 0.35,
    minimum: int = 512 * 1024 * 1024,
    maximum: int = 8 * 1024 * 1024 * 1024,
    fallback: int = 2 * 1024 * 1024 * 1024,
) -> int:
    """Compute a practical host memory budget scaled to available physical RAM."""

    avail = available_host_memory_bytes()
    if avail is None:
        return max(minimum, min(maximum, fallback))
    scaled = int(avail * fraction)
    return max(minimum, min(maximum, scaled))
