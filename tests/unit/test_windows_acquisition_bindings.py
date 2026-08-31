"""Reuse native declarations without reusing acquired handles or file evidence."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import ctypes
import os
import threading

import pytest

from src.depth_surge_3d.io import file_identity as identity_module
from src.depth_surge_3d.io import nofollow_io as nofollow
from src.depth_surge_3d.io import reservation_protocol as reservation


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows native acquisition")


@pytest.fixture(autouse=True)
def cold_acquisition_bindings():
    factories = (
        (nofollow, "_windows_root_bindings"),
        (reservation, "_windows_relative_file_bindings"),
    )
    for module, name in factories:
        factory = getattr(module, name, None)
        if factory is not None:
            factory.cache_clear()
    identity_module._windows_identity_bindings.cache_clear()
    identity_module._windows_identity_bindings()
    yield
    for module, name in factories:
        factory = getattr(module, name, None)
        if factory is not None:
            factory.cache_clear()
    identity_module._windows_identity_bindings.cache_clear()


@contextmanager
def acquired_file(kind, path):
    if kind == "control":
        with reservation._open_relative_regular_file(path, writable=False, create=False) as (
            handle,
            _identity,
            _parent,
        ):
            yield handle
    else:
        with nofollow.AcquiredRoot(path.parent) as acquired:
            with acquired.open_regular(path) as handle:
                yield handle


def read_acquired(kind, path):
    with acquired_file(kind, path) as handle:
        return handle.read()


@pytest.mark.parametrize("kind", ["control", "root"])
def test_repeated_real_acquisitions_reuse_native_bindings(tmp_path, monkeypatch, kind):
    path = tmp_path / "control.bin"
    path.write_bytes(b"freshly acquired each time")
    original = ctypes.WinDLL
    loads = []

    def observe_load(*args, **kwargs):
        loads.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(ctypes, "WinDLL", observe_load)
    for _ in range(8):
        assert read_acquired(kind, path) == b"freshly acquired each time"
    assert loads == [(("kernel32",), {"use_last_error": True}), (("ntdll",), {})]


@pytest.mark.parametrize("kind", ["control", "root"])
def test_warmed_acquisitions_do_not_accumulate_ctypes_pointer_types(tmp_path, kind):
    path = tmp_path / "control.bin"
    path.write_bytes(b"bounded declaration lifetime")
    assert read_acquired(kind, path) == b"bounded declaration lifetime"
    before = set(ctypes._pointer_type_cache)
    for _ in range(12):
        assert read_acquired(kind, path) == b"bounded declaration lifetime"
    assert set(ctypes._pointer_type_cache) == before


@pytest.mark.parametrize("kind", ["control", "root"])
def test_shared_types_still_allocate_fresh_native_request_buffers(tmp_path, monkeypatch, kind):
    path = tmp_path / "control.bin"
    path.write_bytes(b"separate native requests")
    original_byref = ctypes.byref
    original_pointer = ctypes.pointer
    requests = defaultdict(list)

    def observe_byref(value, *args):
        name = type(value).__name__
        if name in {"IoStatusBlock", "ObjectAttributes"}:
            if name == "IoStatusBlock":
                assert bytes(value) == bytes(ctypes.sizeof(value))
            requests[name].append(value)
        return original_byref(value, *args)

    def observe_pointer(value):
        if type(value).__name__ == "UnicodeString":
            requests["UnicodeString"].append(value)
        return original_pointer(value)

    monkeypatch.setattr(ctypes, "byref", observe_byref)
    monkeypatch.setattr(ctypes, "pointer", observe_pointer)
    for _ in range(6):
        assert read_acquired(kind, path) == b"separate native requests"
    assert set(requests) == {"IoStatusBlock", "ObjectAttributes", "UnicodeString"}
    for values in requests.values():
        assert len(values) == len({id(value) for value in values}) == 6
        assert len({type(value) for value in values}) == 1


@pytest.mark.parametrize("kind", ["control", "root"])
def test_warmed_bindings_read_replacement_bytes_and_identity(tmp_path, kind):
    path = tmp_path / "control.bin"
    replacement = tmp_path / "replacement.bin"
    path.write_bytes(b"original bytes")
    assert read_acquired(kind, path) == b"original bytes"
    first = identity_module.capture_file_identity(path)
    replacement.write_bytes(b"new bytes from a different file")
    expected = identity_module.capture_file_identity(replacement)
    os.replace(replacement, path)
    with acquired_file(kind, path) as handle:
        assert handle.read() == b"new bytes from a different file"
        actual = identity_module.file_identity_from_open_handle(handle)
    assert actual == expected and actual.file_identity != first.file_identity


@pytest.mark.parametrize("kind", ["control", "root"])
@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
def test_concurrent_acquisitions_do_not_share_paths_handles_or_buffers(tmp_path, kind, warm):
    if warm:
        path = tmp_path / "warmup.bin"
        path.write_bytes(b"warmup")
        assert read_acquired(kind, path) == b"warmup"
    paths = [tmp_path / f"\u6d4b\u8bd5-\U0001f600-{index}.bin" for index in range(8)]
    for index, item in enumerate(paths):
        item.write_bytes(bytes([index]) * 257)
    barrier = threading.Barrier(4)

    def read(item):
        barrier.wait(timeout=10)
        with acquired_file(kind, item) as handle:
            barrier.wait(timeout=10)
            return handle.read()

    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(read, paths))
    assert actual == [bytes([index]) * 257 for index in range(8)]


@pytest.mark.parametrize("kind", ["control", "root"])
def test_failed_binding_initialization_remains_retryable(tmp_path, monkeypatch, kind):
    path = tmp_path / "control.bin"
    path.write_bytes(b"retry without retaining a failed acquisition")
    original = ctypes.WinDLL
    failure = OSError("injected unavailable ntdll")
    failed = False

    def load(name, *args, **kwargs):
        nonlocal failed
        if name == "ntdll" and not failed:
            failed = True
            raise failure
        return original(name, *args, **kwargs)

    monkeypatch.setattr(ctypes, "WinDLL", load)
    with pytest.raises(OSError) as captured:
        read_acquired(kind, path)
    assert captured.value is failure
    for _ in range(3):
        assert read_acquired(kind, path) == b"retry without retaining a failed acquisition"


@pytest.mark.parametrize("kind", ["control", "root"])
def test_warmed_acquisition_failure_does_not_cache_absence(tmp_path, kind):
    path = tmp_path / "missing.bin"
    error_type = OSError if kind == "control" else nofollow.NoFollowPathError
    with pytest.raises(error_type):
        read_acquired(kind, path)
    path.write_bytes(b"created after the failed acquisition")
    assert read_acquired(kind, path) == b"created after the failed acquisition"
    path.unlink()
    path.mkdir()
    with pytest.raises(error_type):
        read_acquired(kind, path)


@pytest.mark.parametrize("kind", ["control", "root"])
def test_shared_bindings_keep_concurrent_error_codes_separate(tmp_path, kind):
    path = tmp_path / "warmup.bin"
    path.write_bytes(b"warmup")
    assert read_acquired(kind, path) == b"warmup"
    missing_file = tmp_path / "missing.bin"
    missing_parent = tmp_path / "missing-parent" / "missing-child" / "missing.bin"
    barrier = threading.Barrier(4)

    def fail(item):
        barrier.wait(timeout=10)
        if kind == "control":
            with pytest.raises(OSError) as captured:
                read_acquired(kind, item)
            return captured.value.winerror
        with pytest.raises(nofollow.NoFollowPathError) as captured:
            read_acquired(kind, item)
        return int(str(captured.value).rsplit("Windows error ", 1)[1])

    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(fail, [missing_file, missing_parent] * 4))
    assert actual == [2, 3] * 4
    assert read_acquired(kind, path) == b"warmup"


@pytest.mark.parametrize("kind", ["control", "root"])
def test_failed_acquisitions_release_all_native_handles(tmp_path, kind):
    from ctypes import wintypes

    count_handles = ctypes.WinDLL("kernel32", use_last_error=True).GetProcessHandleCount
    count_handles.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    count_handles.restype = wintypes.BOOL

    def current_handles():
        result = wintypes.DWORD()
        assert count_handles(wintypes.HANDLE(-1), ctypes.byref(result))
        return result.value

    missing = tmp_path / "missing.bin"
    error_type = OSError if kind == "control" else nofollow.NoFollowPathError
    with pytest.raises(error_type):
        read_acquired(kind, missing)
    before = current_handles()
    for _ in range(32):
        with pytest.raises(error_type):
            read_acquired(kind, missing)
    assert current_handles() == before


def test_control_bindings_revalidate_the_current_parent(tmp_path):
    parent = tmp_path / "current"
    parent.mkdir()
    path = parent / "control.bin"
    path.write_bytes(b"original parent")
    assert read_acquired("control", path) == b"original parent"
    expected_parent = identity_module.capture_directory_identity(parent)
    parent.rename(tmp_path / "previous")
    parent.mkdir()
    path.write_bytes(b"different parent")
    with pytest.raises(reservation.ReservationConflictError, match="parent identity changed"):
        with reservation._open_relative_regular_file(
            path, writable=False, create=False, expected_parent_identity=expected_parent
        ):
            pytest.fail("a replaced parent must not be accepted")
    assert path.read_bytes() == b"different parent"


def test_control_bindings_revalidate_single_link_ownership(tmp_path):
    path = tmp_path / "control.bin"
    alias = tmp_path / "alias.bin"
    path.write_bytes(b"one link")
    assert read_acquired("control", path) == b"one link"
    os.link(path, alias)
    with pytest.raises(reservation.ReservationConflictError, match="not single-link"):
        read_acquired("control", path)
    alias.unlink()
    assert read_acquired("control", path) == b"one link"


def test_root_bindings_retain_locked_read_rename_exclusion(tmp_path):
    path = tmp_path / "control.bin"
    replacement = tmp_path / "replacement.bin"
    path.write_bytes(b"locked generation")
    replacement.write_bytes(b"next generation")
    assert read_acquired("root", path) == b"locked generation"
    with nofollow.AcquiredRoot(tmp_path) as acquired:
        with acquired.open_regular_locked(path) as handle:
            with pytest.raises(PermissionError):
                os.replace(replacement, path)
            assert handle.read() == b"locked generation"
    os.replace(replacement, path)
    assert read_acquired("root", path) == b"next generation"


def test_root_bindings_retain_exclusive_creation(tmp_path):
    path = tmp_path / "control.bin"
    path.write_bytes(b"do not replace")
    assert read_acquired("root", path) == b"do not replace"
    with nofollow.AcquiredRoot(tmp_path) as acquired:
        with pytest.raises(FileExistsError):
            acquired.create_regular_exclusive(path)
    assert path.read_bytes() == b"do not replace"


def test_control_bindings_retain_exclusive_creation(tmp_path):
    path = tmp_path / "control.bin"
    path.write_bytes(b"do not replace")
    assert read_acquired("control", path) == b"do not replace"
    with pytest.raises(OSError) as captured:
        with reservation._open_relative_regular_file(
            path, writable=True, create=True, exclusive=True
        ):
            pytest.fail("exclusive creation must not replace an existing control file")
    assert captured.value.winerror in {80, 183}
    assert path.read_bytes() == b"do not replace"
