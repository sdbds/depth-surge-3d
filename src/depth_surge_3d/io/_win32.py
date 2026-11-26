"""Access the Windows-only ctypes and CRT operations used by handle-based I/O."""

import ctypes
import sys


def load_library(name: str, *, use_last_error: bool = False) -> ctypes.CDLL:
    if sys.platform != "win32":
        raise OSError("Windows DLLs are unavailable on this platform")
    if use_last_error:
        return ctypes.WinDLL(name, use_last_error=True)
    return ctypes.WinDLL(name)


def last_error() -> int:
    if sys.platform != "win32":
        raise OSError("Windows last-error state is unavailable on this platform")
    return ctypes.get_last_error()


def set_last_error(value: int) -> None:
    if sys.platform != "win32":
        raise OSError("Windows last-error state is unavailable on this platform")
    ctypes.set_last_error(value)


def win_error(code: int) -> OSError:
    if sys.platform != "win32":
        raise OSError("Windows error messages are unavailable on this platform")
    return ctypes.WinError(code)


def get_osfhandle(fd: int) -> int:
    if sys.platform != "win32":
        raise OSError("Windows file handles are unavailable on this platform")
    import msvcrt

    return msvcrt.get_osfhandle(fd)


def open_osfhandle(handle: int, flags: int) -> int:
    if sys.platform != "win32":
        raise OSError("Windows file handles are unavailable on this platform")
    import msvcrt

    return msvcrt.open_osfhandle(handle, flags)
