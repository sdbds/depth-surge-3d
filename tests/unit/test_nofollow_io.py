"""Handle-relative, no-follow input opening contracts."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.depth_surge_3d.io import nofollow_io
from src.depth_surge_3d.io.nofollow_io import (
    AcquiredRoot,
    FIXED_FILE_HASH_STREAM_BYTES,
    NoFollowPathError,
    SafeTreeMutation,
    open_regular_beneath,
    regular_allocation_evidence,
    scan_fixed_regular_v1,
)


def _symlink_or_skip(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        os.symlink(target, link, target_is_directory=directory)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")


def test_open_regular_beneath_reads_from_the_acquired_regular_handle(tmp_path: Path) -> None:
    payload = tmp_path / "inputs" / "frame.bin"
    payload.parent.mkdir()
    payload.write_bytes(b"original")

    with open_regular_beneath(tmp_path, payload) as handle:
        payload.replace(payload.with_suffix(".old"))
        payload.write_bytes(b"replacement")
        assert handle.read() == b"original"


def test_open_regular_beneath_rejects_a_final_file_link(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"payload")
    link = tmp_path / "linked.bin"
    _symlink_or_skip(link, target)

    with pytest.raises(NoFollowPathError, match="link|reparse"):
        open_regular_beneath(tmp_path, link)


def test_open_regular_beneath_rejects_a_linked_parent_component(tmp_path: Path) -> None:
    target = tmp_path / "real"
    target.mkdir()
    (target / "frame.bin").write_bytes(b"payload")
    linked_parent = tmp_path / "linked"
    _symlink_or_skip(linked_parent, target, directory=True)

    with pytest.raises(NoFollowPathError, match="link|reparse"):
        open_regular_beneath(tmp_path, linked_parent / "frame.bin")


def test_acquired_root_keeps_relative_reads_bound_to_original_directory(
    tmp_path: Path,
) -> None:
    job_root = tmp_path / "job"
    selected = job_root / "frames"
    selected.mkdir(parents=True)
    (selected / "frame.bin").write_bytes(b"original")

    with AcquiredRoot(job_root) as acquired:
        moved = tmp_path / "moved"
        try:
            job_root.rename(moved)
        except OSError as error:
            pytest.skip(f"renaming an acquired directory is unavailable: {error}")
        replacement = job_root / "frames"
        replacement.mkdir(parents=True)
        (replacement / "frame.bin").write_bytes(b"replacement")

        with acquired.open_regular(job_root / "frames" / "frame.bin") as handle:
            assert handle.read() == b"original"


def test_acquired_directory_enumeration_rejects_reparse_entries(tmp_path: Path) -> None:
    selected = tmp_path / "frames"
    selected.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"sentinel")
    _symlink_or_skip(selected / "frame_000001.png", outside)

    with AcquiredRoot(tmp_path) as acquired:
        with pytest.raises(NoFollowPathError, match="link|reparse"):
            list(acquired.iter_directory(selected))
    assert outside.read_bytes() == b"sentinel"


def test_fixed_file_scan_rejects_hard_links_before_hashing(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    alias = tmp_path / "alias.bin"
    try:
        os.link(source, alias)
    except OSError as error:
        pytest.skip(f"hard-link creation is unavailable: {error}")

    with pytest.raises(NoFollowPathError, match="link-count-one"):
        scan_fixed_regular_v1(tmp_path, source)

    assert source.read_bytes() == b"payload"
    assert alias.read_bytes() == b"payload"


def test_fixed_file_scan_is_cancellable_between_fixed_buffer_reads(
    tmp_path: Path,
) -> None:
    assert FIXED_FILE_HASH_STREAM_BYTES == 1024 * 1024
    source = tmp_path / "source.bin"
    source.write_bytes(b"x" * (2 * FIXED_FILE_HASH_STREAM_BYTES + 1))
    polls = 0

    def cancelled() -> bool:
        nonlocal polls
        polls += 1
        return polls >= 3

    with pytest.raises(InterruptedError, match="cancelled"):
        scan_fixed_regular_v1(tmp_path, source, cancelled=cancelled)

    assert polls == 3


def test_fixed_file_scan_rejects_name_replacement_during_scan(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    parked = tmp_path / "parked.bin"
    source.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    polls = 0

    def replace_name_without_cancelling() -> bool:
        nonlocal polls
        polls += 1
        if polls == 3:
            source.rename(parked)
            source.write_bytes(b"replacement")
        return False

    with pytest.raises(NoFollowPathError, match="changed"):
        scan_fixed_regular_v1(
            tmp_path,
            source,
            cancelled=replace_name_without_cancelling,
        )

    assert parked.stat().st_size == 2 * 1024 * 1024 + 1
    assert source.read_bytes() == b"replacement"


def test_fixed_file_scan_rejects_truncation_during_scan(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    polls = 0

    def truncate_without_cancelling() -> bool:
        nonlocal polls
        polls += 1
        if polls == 3:
            with source.open("r+b") as handle:
                handle.truncate(1024 * 1024)
        return False

    with pytest.raises(NoFollowPathError, match="ended"):
        scan_fixed_regular_v1(
            tmp_path,
            source,
            cancelled=truncate_without_cancelling,
        )


def test_fixed_file_scan_rejects_extension_during_scan(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    polls = 0

    def extend_without_cancelling() -> bool:
        nonlocal polls
        polls += 1
        if polls == 3:
            with source.open("ab") as handle:
                handle.write(b"extra")
        return False

    with pytest.raises(NoFollowPathError, match="exceeds"):
        scan_fixed_regular_v1(
            tmp_path,
            source,
            cancelled=extend_without_cancelling,
        )


def test_safe_tree_mutation_clears_and_removes_only_authorized_children(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    diagnostics = tmp_path / "diagnostics"
    (outputs / "nested").mkdir(parents=True)
    (outputs / "nested" / "old.bin").write_bytes(b"old")
    diagnostics.mkdir()
    (diagnostics / "frame-a").mkdir()
    (diagnostics / "frame-a" / "stats.json").write_bytes(b"{}")
    (diagnostics / "keep.bin").write_bytes(b"keep")

    with SafeTreeMutation(tmp_path, (outputs, diagnostics)) as mutation:
        mutation.clear_root(outputs)
        frame = diagnostics / "frame-a"
        evidence = mutation.capture_child_tree(diagnostics, frame)
        mutation.remove_child_tree(diagnostics, frame, evidence=evidence)

    assert outputs.is_dir()
    assert list(outputs.iterdir()) == []
    assert (diagnostics / "keep.bin").read_bytes() == b"keep"
    assert not (diagnostics / "frame-a").exists()


def test_safe_tree_mutation_rejects_unsafe_tree_before_any_deletion(
    tmp_path: Path,
) -> None:
    safe = tmp_path / "safe"
    unsafe = tmp_path / "unsafe"
    outside = tmp_path / "outside"
    safe.mkdir()
    unsafe.mkdir()
    outside.mkdir()
    (safe / "must-stay.bin").write_bytes(b"safe")
    (outside / "sentinel.bin").write_bytes(b"outside")
    _symlink_or_skip(unsafe / "linked", outside, directory=True)

    with pytest.raises(NoFollowPathError, match="link|reparse"):
        SafeTreeMutation(tmp_path, (safe, unsafe))

    assert (safe / "must-stay.bin").read_bytes() == b"safe"
    assert (outside / "sentinel.bin").read_bytes() == b"outside"


def test_safe_tree_mutation_rejects_authorized_root_replacement(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "old.bin").write_bytes(b"old")

    with SafeTreeMutation(tmp_path, (outputs,)) as mutation:
        parked = tmp_path / "parked"
        outputs.rename(parked)
        outputs.mkdir()
        (outputs / "sentinel.bin").write_bytes(b"replacement")
        with pytest.raises(NoFollowPathError, match="identity changed"):
            mutation.clear_root(outputs)

    assert (parked / "old.bin").read_bytes() == b"old"
    assert (outputs / "sentinel.bin").read_bytes() == b"replacement"


@pytest.mark.skipif(os.name != "nt", reason="Windows enumerated identity fixture")
@pytest.mark.parametrize("directory", [False, True])
def test_windows_tree_mutation_rejects_name_swap_before_child_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directory: bool,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    victim = root / "victim"
    parked = root / "parked"
    if directory:
        victim.mkdir()
        (victim / "authorized.bin").write_bytes(b"authorized")
    else:
        victim.write_bytes(b"authorized")

    with SafeTreeMutation(tmp_path, (root,)) as mutation:
        implementation = mutation._acquired._impl
        original_open = implementation._open_relative
        injected = False

        def swap_before_open(parent, name, *, directory, **kwargs):
            nonlocal injected
            if not injected and name == victim.name and not kwargs.get("create", False):
                injected = True
                victim.rename(parked)
                if directory:
                    victim.mkdir()
                    (victim / "replacement.bin").write_bytes(b"replacement")
                else:
                    victim.write_bytes(b"replacement")
            return original_open(parent, name, directory=directory, **kwargs)

        monkeypatch.setattr(implementation, "_open_relative", swap_before_open)
        with pytest.raises(NoFollowPathError, match="changed before acquisition"):
            mutation.clear_root(root)

    assert injected
    assert parked.exists()
    assert victim.exists()
    if directory:
        assert (parked / "authorized.bin").read_bytes() == b"authorized"
        assert (victim / "replacement.bin").read_bytes() == b"replacement"
    else:
        assert parked.read_bytes() == b"authorized"
        assert victim.read_bytes() == b"replacement"


@pytest.mark.skipif(os.name == "nt", reason="POSIX enumerated identity fixture")
@pytest.mark.parametrize("directory", [False, True])
def test_posix_tree_mutation_rejects_name_swap_before_child_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directory: bool,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    victim = root / "victim"
    parked = root / "parked"
    if directory:
        victim.mkdir()
        (victim / "authorized.bin").write_bytes(b"authorized")
    else:
        victim.write_bytes(b"authorized")

    with SafeTreeMutation(tmp_path, (root,)) as mutation:
        original_open = nofollow_io.os.open
        injected = False

        def swap_before_open(path, flags, mode=0o777, *, dir_fd=None):
            nonlocal injected
            if not injected and path == victim.name and dir_fd is not None:
                injected = True
                os.rename(
                    victim.name,
                    parked.name,
                    src_dir_fd=dir_fd,
                    dst_dir_fd=dir_fd,
                )
                if directory:
                    os.mkdir(victim.name, dir_fd=dir_fd)
                else:
                    replacement = original_open(
                        victim.name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=dir_fd,
                    )
                    os.write(replacement, b"replacement")
                    os.close(replacement)
            return original_open(path, flags, mode, dir_fd=dir_fd)

        monkeypatch.setattr(nofollow_io.os, "open", swap_before_open)
        with pytest.raises(NoFollowPathError, match="changed before acquisition"):
            mutation.clear_root(root)

    assert injected
    assert parked.exists()
    assert victim.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX quarantine race fixture")
@pytest.mark.parametrize("directory", [False, True])
def test_posix_quarantine_never_deletes_a_last_moment_name_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    directory: bool,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    victim = root / "victim"
    parked = root / "parked"
    if directory:
        victim.mkdir()
    else:
        victim.write_bytes(b"authorized")
    original = nofollow_io._posix_quarantine_noreplace
    injected = False

    def replace_before_quarantine(source_parent, source_name, target_parent, target_name):
        nonlocal injected
        if not injected and source_name == victim.name:
            injected = True
            os.rename(victim.name, parked.name, src_dir_fd=source_parent, dst_dir_fd=source_parent)
            if directory:
                os.mkdir(victim.name, dir_fd=source_parent)
            else:
                descriptor = os.open(
                    victim.name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=source_parent,
                )
                os.write(descriptor, b"replacement")
                os.close(descriptor)
        return original(source_parent, source_name, target_parent, target_name)

    monkeypatch.setattr(nofollow_io, "_posix_quarantine_noreplace", replace_before_quarantine)
    with SafeTreeMutation(tmp_path, (root,)) as mutation:
        evidence = (
            mutation.capture_child_tree(root, victim)
            if directory
            else mutation.capture_regular_child(root, victim)
        )
        with pytest.raises(NoFollowPathError, match="changed before quarantine"):
            if directory:
                mutation.remove_child_tree(root, victim, evidence=evidence)
            else:
                mutation.unlink_child(root, victim, evidence=evidence)

    assert parked.exists()
    assert victim.exists()
    if not directory:
        assert parked.read_bytes() == b"authorized"
        assert victim.read_bytes() == b"replacement"


@pytest.mark.skipif(os.name == "nt", reason="POSIX quarantine recovery fixture")
@pytest.mark.parametrize("directory", [False, True])
def test_posix_quarantine_resumes_after_the_public_name_was_moved(
    tmp_path: Path,
    directory: bool,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    victim = root / "victim"
    if directory:
        victim.mkdir()
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    else:
        victim.write_bytes(b"authorized")
        flags = os.O_PATH | os.O_NOFOLLOW
    parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    opened = os.open(victim.name, flags, dir_fd=parent)
    quarantine = ""

    def crash_after_move(stage: str, quarantine_name: str) -> None:
        nonlocal quarantine
        if stage == "after_quarantine_rename":
            quarantine = quarantine_name
            raise RuntimeError("simulated crash")

    try:
        with pytest.raises(RuntimeError, match="simulated crash"):
            nofollow_io.posix_remove_verified_name(
                parent,
                victim.name,
                opened,
                directory=directory,
                fault=crash_after_move,
            )
        assert quarantine
        assert not victim.exists()
        quarantine_fd = os.open(quarantine, flags, dir_fd=parent)
        try:
            nofollow_io.posix_remove_verified_name(
                parent,
                quarantine,
                quarantine_fd,
                directory=directory,
                recovering=True,
            )
        finally:
            os.close(quarantine_fd)
    finally:
        os.close(opened)
        os.close(parent)

    assert not (root / quarantine).exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX quarantine identity fixture")
@pytest.mark.parametrize("directory", [False, True])
def test_posix_public_quarantine_prefix_name_is_requarantined(
    tmp_path: Path,
    directory: bool,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    victim = root / ".depth-surge-quarantine-v1-public"
    if directory:
        victim.mkdir()
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    else:
        victim.write_bytes(b"authorized")
        flags = os.O_PATH | os.O_NOFOLLOW
    parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    opened = os.open(victim.name, flags, dir_fd=parent)
    moved_name = ""

    def replace_public_name(stage: str, quarantine_name: str) -> None:
        nonlocal moved_name
        if stage != "after_quarantine_rename":
            return
        moved_name = quarantine_name
        if directory:
            victim.mkdir()
        else:
            victim.write_bytes(b"replacement")

    try:
        with pytest.raises(nofollow_io.NoFollowPathError, match="reappeared"):
            nofollow_io.posix_remove_verified_name(
                parent,
                victim.name,
                opened,
                directory=directory,
                fault=replace_public_name,
            )
    finally:
        os.close(opened)
        os.close(parent)

    assert moved_name != victim.name
    assert victim.exists()
    if not directory:
        assert victim.read_bytes() == b"replacement"


def test_safe_tree_mutation_unlinks_only_a_regular_nested_descendant(tmp_path: Path) -> None:
    root = tmp_path / "frames"
    frame = root / "frame_000001"
    frame.mkdir(parents=True)
    residue = frame / ".manifest.tmp"
    residue.write_bytes(b"complete residue")

    with SafeTreeMutation(tmp_path, (root,)) as mutation:
        evidence = mutation.capture_regular_child(root, residue)
        mutation.unlink_descendant(root, residue, evidence=evidence)

    assert not residue.exists()


def test_safe_tree_mutation_rejects_a_wrong_type_nested_descendant(tmp_path: Path) -> None:
    root = tmp_path / "frames"
    residue = root / "frame_000001" / ".manifest.tmp"
    residue.mkdir(parents=True)

    with SafeTreeMutation(tmp_path, (root,)) as mutation:
        with pytest.raises(NoFollowPathError, match="wrong type|regular|reparse|invalid component"):
            evidence = mutation.capture_regular_child(root, residue)
            mutation.unlink_descendant(root, residue, evidence=evidence)

    assert residue.is_dir()


def test_safe_tree_mutation_rejects_a_linked_nested_descendant(tmp_path: Path) -> None:
    root = tmp_path / "frames"
    frame = root / "frame_000001"
    frame.mkdir(parents=True)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    residue = frame / ".manifest.tmp"
    _symlink_or_skip(residue, outside)

    with pytest.raises(NoFollowPathError, match="link|reparse|invalid component"):
        with SafeTreeMutation(tmp_path, (root,)) as mutation:
            evidence = mutation.capture_regular_child(root, residue)
            mutation.unlink_descendant(root, residue, evidence=evidence)

    assert outside.read_bytes() == b"outside"


@pytest.mark.parametrize("relative", (Path("missing"), Path("missing") / "nested"))
def test_safe_tree_mutation_accepts_an_authorized_root_that_remains_missing(
    tmp_path: Path,
    relative: Path,
) -> None:
    root = tmp_path / relative
    with SafeTreeMutation(tmp_path, (root,)) as mutation:
        mutation.clear_root(root)
        mutation.remove_root(root)


def test_safe_tree_mutation_rejects_absent_root_that_appears(tmp_path: Path) -> None:
    root = tmp_path / "diagnostics" / "frames"
    with SafeTreeMutation(tmp_path, (root,)) as mutation:
        root.mkdir(parents=True)
        (root / "sentinel.bin").write_bytes(b"replacement")
        with pytest.raises(NoFollowPathError, match="appeared"):
            mutation.clear_root(root)
    assert (root / "sentinel.bin").read_bytes() == b"replacement"


def test_safe_tree_mutation_rejects_absent_child_that_appears(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    child = root / "late.bin"
    with SafeTreeMutation(tmp_path, (root,)) as mutation:
        evidence = mutation.capture_regular_child(root, child)
        child.write_bytes(b"new")
        with pytest.raises(NoFollowPathError, match="appeared"):
            mutation.unlink_child(root, child, evidence=evidence)
    assert child.read_bytes() == b"new"


def test_safe_tree_mutation_preserves_child_replacement_after_preflight(
    tmp_path: Path,
) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    child = root / "result.bin"
    child.write_bytes(b"old")
    with SafeTreeMutation(tmp_path, (root,)) as mutation:
        evidence = mutation.capture_regular_child(root, child)
        child.rename(root / "parked.bin")
        child.write_bytes(b"replacement")
        with pytest.raises(NoFollowPathError, match="identity|changed"):
            mutation.unlink_child(root, child, evidence=evidence)
    assert child.read_bytes() == b"replacement"


def test_safe_tree_mutation_rejects_absent_root_that_becomes_a_link(
    tmp_path: Path,
) -> None:
    root = tmp_path / "diagnostics"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel.bin").write_bytes(b"outside")
    with SafeTreeMutation(tmp_path, (root,)) as mutation:
        _symlink_or_skip(root, outside, directory=True)
        with pytest.raises(NoFollowPathError, match="link|reparse"):
            mutation.remove_root(root)
    assert (outside / "sentinel.bin").read_bytes() == b"outside"


def test_regular_allocation_evidence_requires_one_owned_link(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"x" * 5000)
    with AcquiredRoot(tmp_path) as acquired:
        with acquired.open_regular(payload) as handle:
            evidence = regular_allocation_evidence(handle)
    if evidence is None:
        pytest.skip("the test filesystem has no conservative allocation adapter")
    assert evidence.allocated_bytes >= payload.stat().st_size

    alias = tmp_path / "alias.bin"
    os.link(payload, alias)
    with AcquiredRoot(tmp_path) as acquired:
        with acquired.open_regular(payload) as handle:
            assert regular_allocation_evidence(handle) is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor accounting only")
def test_posix_verification_failure_closes_the_parent_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    before = len(list(Path("/proc/self/fd").iterdir()))

    with AcquiredRoot(tmp_path) as acquired:
        implementation = acquired._impl

        def reject(*_args: object, **_kwargs: object) -> None:
            raise NoFollowPathError("changed mount")

        monkeypatch.setattr(implementation, "_verify", reject)
        with pytest.raises(NoFollowPathError, match="changed mount"):
            list(acquired.iter_directory(child))

    assert len(list(Path("/proc/self/fd").iterdir())) == before
