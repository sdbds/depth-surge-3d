"""Reuse Windows ABI declarations without reusing any file identity or buffer."""

from concurrent.futures import ThreadPoolExecutor
import ctypes
import os
import time
from types import SimpleNamespace

import pytest

from src.depth_surge_3d.io import file_identity as identity_module


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows handle identities")


@pytest.fixture(autouse=True)
def cold_bindings():
    bindings = getattr(identity_module, "_windows_identity_bindings", None)
    if bindings is not None:
        bindings.cache_clear()
    yield
    if bindings is not None:
        bindings.cache_clear()


def test_repeated_real_handle_queries_reuse_the_dll_binding(tmp_path, monkeypatch):
    import msvcrt

    path = tmp_path / "identity.bin"
    path.write_bytes(b"same live file")
    original = ctypes.WinDLL
    loads = []

    def observe_load(*args, **kwargs):
        loads.append((args, kwargs))
        return original(*args, **kwargs)

    with path.open("rb") as handle:
        native = msvcrt.get_osfhandle(handle.fileno())
        monkeypatch.setattr(ctypes, "WinDLL", observe_load)
        results = [
            identity_module.windows_identity_from_handle(native, require_directory=False)
            for _ in range(8)
        ]
    assert all(result == results[0] for result in results)
    assert results[0][1] == 1
    assert loads == [(("kernel32",), {"use_last_error": True})]


class IdentityQuery:
    def __init__(self):
        self.records = {}
        self.calls = []
        self.failure_class = None
        self.failure_codes = {}
        self.delay = 0.0

    def __call__(self, handle, info_class, pointer, size):
        native = handle.value
        target = pointer._obj
        assert size == {18: 24, 1: 24, 9: 8}[info_class]
        assert bytes(target) == bytes(size)
        self.calls.append((native, info_class, target))
        if info_class == self.failure_class:
            ctypes.set_last_error(self.failure_codes.get(native, 5))
            time.sleep(self.delay)
            return 0
        volume, file_id, links, directory, attributes = self.records[native]
        if info_class == 18:
            target.volume_serial = volume
            target.file_id.identifier[:] = file_id
        elif info_class == 1:
            target.number_of_links = links
            target.directory = directory
        else:
            target.file_attributes = attributes
        time.sleep(self.delay)
        return 1


@pytest.fixture
def identity_api(monkeypatch):
    query = IdentityQuery()
    query.records[17] = (0x0123456789ABCDEF, bytes(range(16)), 1, False, 0x20)
    loads = []

    def load(name, *, use_last_error):
        assert name == "kernel32" and use_last_error is True
        loads.append(name)
        return SimpleNamespace(GetFileInformationByHandleEx=query)

    monkeypatch.setattr(ctypes, "WinDLL", load)
    return query, loads


def test_each_query_reads_fresh_identity_links_and_attributes(identity_api):
    query, loads = identity_api
    first = identity_module.windows_identity_from_handle(17, require_directory=False)
    query.records[17] = (0xFEDCBA9876543210, bytes(range(16, 32)), 7, False, 0x21)
    second = identity_module.windows_identity_from_handle(17, require_directory=False)

    assert first[0].to_object() == {
        "volume_serial": "0123456789abcdef",
        "file_id": "000102030405060708090a0b0c0d0e0f",
    }
    assert first[1:] == (1, 0x20)
    assert second[0].to_object() == {
        "volume_serial": "fedcba9876543210",
        "file_id": "101112131415161718191a1b1c1d1e1f",
    }
    assert second[1:] == (7, 0x21)
    assert [(handle, kind) for handle, kind, _ in query.calls] == [(17, 18), (17, 1), (17, 9)] * 2
    assert len({id(target) for _, _, target in query.calls}) == 6
    assert len({type(target) for _, kind, target in query.calls if kind == 18}) == 1
    assert loads == ["kernel32"]


@pytest.mark.parametrize(
    "info_class,label",
    [
        (18, "FileIdInfo"),
        (1, "FileStandardInfo"),
        (9, "FileAttributeTagInfo"),
    ],
)
def test_failure_reports_current_error_and_does_not_cache_a_result(identity_api, info_class, label):
    query, _ = identity_api
    query.failure_class = info_class
    query.failure_codes[17] = 32
    with pytest.raises(identity_module.PersistentIdentityCapabilityError) as captured:
        identity_module.windows_identity_from_handle(17, require_directory=False)
    assert str(captured.value) == f"GetFileInformationByHandleEx({label}) failed: Windows error 32"
    expected = [18, 1, 9]
    assert [kind for _, kind, _ in query.calls] == expected[: expected.index(info_class) + 1]
    query.failure_class = None
    actual, links, attributes = identity_module.windows_identity_from_handle(
        17, require_directory=False
    )
    assert actual.file_id == "000102030405060708090a0b0c0d0e0f"
    assert (links, attributes) == (1, 0x20)


@pytest.mark.parametrize(
    "directory,required,attributes,links,message",
    [
        (True, False, 0x10, 1, "identity handle type changed"),
        (False, True, 0x20, 1, "identity handle type changed"),
        (False, False, 0x420, 1, "identity handle is a reparse point"),
        (True, True, 0x410, 1, "identity handle is a reparse point"),
        (False, False, 0x20, 0, "identity handle has no links"),
        (True, True, 0x10, 0, "identity handle has no links"),
    ],
)
def test_warmed_bindings_still_reject_changed_handle_metadata(
    identity_api, directory, required, attributes, links, message
):
    query, _ = identity_api
    identity_module.windows_identity_from_handle(17, require_directory=False)
    query.records[17] = (1, bytes(16), links, directory, attributes)
    with pytest.raises(identity_module.PersistentIdentitySchemaError, match=message):
        identity_module.windows_identity_from_handle(17, require_directory=required)


@pytest.mark.parametrize("directory,attributes", [(False, 0x420), (True, 0x410)])
def test_explicit_reparse_permission_remains_per_call(identity_api, directory, attributes):
    query, _ = identity_api
    query.records[17] = (1, bytes(16), 2, directory, attributes)
    actual, links, flags = identity_module.windows_identity_from_handle(
        17, require_directory=directory, allow_reparse=True
    )
    assert actual.volume_serial == "0000000000000001"
    assert (links, flags) == (2, attributes)
    with pytest.raises(identity_module.PersistentIdentitySchemaError, match="reparse point"):
        identity_module.windows_identity_from_handle(17, require_directory=directory)


def test_concurrent_queries_keep_separate_result_buffers(identity_api):
    query, loads = identity_api
    identity_module.windows_identity_from_handle(17, require_directory=False)
    for handle in range(20, 28):
        query.records[handle] = (handle, bytes([handle]) * 16, handle, False, 0x20)
    query.delay = 0.001

    def capture(handle):
        return identity_module.windows_identity_from_handle(handle, require_directory=False)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(capture, range(20, 28)))
    for handle, (actual, links, attributes) in zip(range(20, 28), results):
        assert actual.volume_serial == f"{handle:016x}"
        assert actual.file_id == bytes([handle] * 16).hex()
        assert (links, attributes) == (handle, 0x20)
    assert len({id(target) for _, _, target in query.calls}) == 27
    assert loads == ["kernel32"]


def test_concurrent_failures_keep_thread_local_last_error(identity_api):
    query, _ = identity_api
    identity_module.windows_identity_from_handle(17, require_directory=False)
    query.failure_class = 18
    query.failure_codes = {17: 5, 18: 32}
    query.delay = 0.001

    def fail(handle):
        with pytest.raises(identity_module.PersistentIdentityCapabilityError) as captured:
            identity_module.windows_identity_from_handle(handle, require_directory=False)
        return str(captured.value)

    with ThreadPoolExecutor(max_workers=2) as executor:
        messages = list(executor.map(fail, [17, 18]))
    assert messages == [
        "GetFileInformationByHandleEx(FileIdInfo) failed: Windows error 5",
        "GetFileInformationByHandleEx(FileIdInfo) failed: Windows error 32",
    ]


def test_failed_binding_initialization_can_be_retried(monkeypatch):
    query = IdentityQuery()
    query.records[17] = (1, bytes(16), 1, False, 0x20)
    attempts = []
    failure = OSError("injected unavailable Windows binding")

    def load(*_args, **_kwargs):
        attempts.append(None)
        if len(attempts) == 1:
            raise failure
        return SimpleNamespace(GetFileInformationByHandleEx=query)

    monkeypatch.setattr(ctypes, "WinDLL", load)
    with pytest.raises(OSError) as captured:
        identity_module.windows_identity_from_handle(17, require_directory=False)
    assert captured.value is failure
    for _ in range(2):
        actual, links, flags = identity_module.windows_identity_from_handle(
            17, require_directory=False
        )
        assert actual.volume_serial == "0000000000000001" and (links, flags) == (1, 0x20)
    assert len(attempts) == 2


def test_platform_guard_precedes_even_warmed_bindings(identity_api, monkeypatch):
    query, loads = identity_api
    identity_module.windows_identity_from_handle(17, require_directory=False)
    monkeypatch.setattr(identity_module, "os", SimpleNamespace(name="posix"))
    with pytest.raises(
        identity_module.PersistentIdentityCapabilityError, match="Windows FileIdInfo is unavailable"
    ):
        identity_module.windows_identity_from_handle(17, require_directory=False)
    assert len(query.calls) == 3 and loads == ["kernel32"]


def test_real_identity_detects_replacement_and_link_count_changes(tmp_path):
    path = tmp_path / "identity.bin"
    alias = tmp_path / "alias.bin"
    replacement = tmp_path / "replacement.bin"
    path.write_bytes(b"original")
    original = identity_module.capture_file_identity(path)
    assert original.link_count == 1
    os.link(path, alias)
    linked = identity_module.capture_file_identity(path)
    assert linked.file_identity == original.file_identity and linked.link_count == 2
    alias.unlink()
    assert identity_module.capture_file_identity(path).link_count == 1
    replacement.write_bytes(b"replacement")
    expected = identity_module.capture_file_identity(replacement)
    os.replace(replacement, path)
    observed = identity_module.capture_file_identity(path)
    assert observed == expected and observed.file_identity != original.file_identity


def test_real_handle_rejects_wrong_object_kind_and_closed_handle(tmp_path):
    import msvcrt

    path = tmp_path / "identity.bin"
    path.write_bytes(b"regular file")
    with path.open("rb") as handle:
        native = msvcrt.get_osfhandle(handle.fileno())
        with pytest.raises(
            identity_module.PersistentIdentitySchemaError, match="identity handle type changed"
        ):
            identity_module.windows_identity_from_handle(native, require_directory=True)
    with pytest.raises(identity_module.PersistentIdentityCapabilityError, match="Windows error 6"):
        identity_module.windows_identity_from_handle(-1, require_directory=False)
