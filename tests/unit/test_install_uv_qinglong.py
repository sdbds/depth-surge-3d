"""Run the Windows entry point without downloading the runtime or touching a real project."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


INSTALL_SCRIPT = Path(__file__).parents[2] / "1.install-uv-qinglong.ps1"
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")
pytestmark = pytest.mark.skipif(
    os.name != "nt" or not POWERSHELL, reason="requires Windows PowerShell"
)


@pytest.fixture
def install(tmp_path, monkeypatch):
    checkout = tmp_path / "test checkout"
    checkout.mkdir()
    script = checkout / INSTALL_SCRIPT.name
    shutil.copyfile(INSTALL_SCRIPT, script)
    (checkout / "uv.lock").touch()
    commands = tmp_path / "commands"
    commands.mkdir()
    # Stop at the real process boundary, before fetching the multi-GB runtime.
    (commands / "uv.cmd").write_text(
        '@echo off\nif "%1"=="pip" goto check\n'
        '> "%CALL_LOG%" echo %UV_PROJECT_ENVIRONMENT%\n>> "%CALL_LOG%" echo %*\n'
        'if "%STAGE_CLI%"=="1" (\nmkdir "%UV_PROJECT_ENVIRONMENT%\\Scripts"\n'
        'type nul > "%UV_PROJECT_ENVIRONMENT%\\Scripts\\depth-surge-3d.exe"\n)\n'
        'exit /b %SYNC_STATUS%\n:check\n>> "%CALL_LOG%" echo %*\nexit /b 31\n'
    )
    for name in ("git", "ffmpeg", "ffprobe"):
        (commands / f"{name}.cmd").write_text("@exit /b 0\n")
    monkeypatch.setenv("PATH", str(commands) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("INSTALL_SCRIPT", str(script))
    monkeypatch.setenv("CALL_LOG", str(tmp_path / "uv-call.txt"))
    monkeypatch.setenv("SYNC_STATUS", "29")
    monkeypatch.delenv("STAGE_CLI", raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("UV_PROJECT_ENVIRONMENT", raising=False)

    def invoke(arguments):
        # Never execute a legacy bootstrapper that might install tools globally.
        cmd = (
            "$installer = Get-Command $env:INSTALL_SCRIPT; "
            "if (-not $installer.Parameters.ContainsKey('EnvironmentPath')) { "
            "throw 'Installer has no isolated environment option' }; "
            "& $installer " + arguments
        )
        result = subprocess.run(
            [
                POWERSHELL,
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                cmd,
            ],
            cwd=tmp_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        log = tmp_path / "uv-call.txt"
        return result, log.read_text().splitlines() if log.exists() else []

    return invoke


@pytest.mark.parametrize("extra", [False, True])
def test_locked_wheel_install_targets_only_selected_environment(install, tmp_path, extra):
    target = tmp_path / "fresh runtime"
    result, call = install("-EnvironmentPath './fresh runtime'" + (" -WithMoGe2" if extra else ""))

    assert result.returncode != 0  # uv failed; no interactive prompt or false success.
    assert len(call) == 2, result.stdout
    assert Path(call[0]) == target
    assert str(tmp_path / "test checkout") in call[1]
    assert "sync" in call[1].split()
    assert {"--locked", "--no-editable", "--no-dev", "--inexact", "3.11"} <= set(call[1].split())
    assert ("--extra moge2" in call[1]) is extra
    assert "29" in result.stdout
    assert not (tmp_path / "test checkout" / ".venv").exists()


def test_refuses_existing_non_environment_without_running_uv(install, tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    sentinel = target / "keep.txt"
    sentinel.write_text("untouched")
    result, call = install("-EnvironmentPath ./existing")

    assert result.returncode != 0
    assert "existing" in result.stdout.lower()
    assert not call
    assert sentinel.read_text() == "untouched"


@pytest.mark.skipif(sys.version_info[:2] != (3, 11), reason="needs a local CPython 3.11")
def test_reuses_the_exact_existing_311_interpreter(install, tmp_path):
    target = tmp_path / "my environment"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(target)], check=True)
    sentinel = target / "keep.txt"
    sentinel.touch()
    result, call = install("-EnvironmentPath './my environment'")

    assert len(call) == 2, result.stdout
    assert str(target / "Scripts" / "python.exe") in call[1]
    assert sentinel.is_file()


def test_uses_requested_uv_environment_not_the_active_one(install, tmp_path, monkeypatch):
    target = tmp_path / "uv target"
    active = tmp_path / "other environment"
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(target))
    monkeypatch.setenv("VIRTUAL_ENV", str(active))
    result, call = install("")

    assert len(call) == 2, result.stdout
    assert Path(call[0]) == target
    assert not active.exists()


def test_uv_success_without_installed_cli_is_an_error(install, tmp_path, monkeypatch):
    monkeypatch.setenv("SYNC_STATUS", "0")
    result, call = install("")

    assert len(call) == 2, result.stdout
    assert Path(call[0]) == tmp_path / "test checkout" / ".venv"
    assert result.returncode != 0
    assert "Missing installed CLI" in result.stdout


def test_fails_a_new_install_if_runtime_dependencies_are_incomplete(install, tmp_path, monkeypatch):
    monkeypatch.setenv("SYNC_STATUS", "0")
    monkeypatch.setenv("STAGE_CLI", "1")
    result, call = install("-EnvironmentPath './new runtime'")

    assert result.returncode != 0
    assert len(call) == 3, result.stdout
    assert "pip check --python" in call[2]
    assert str(tmp_path / "new runtime" / "Scripts" / "python.exe") in call[2]
    assert "31" in result.stdout


@pytest.mark.skipif(sys.version_info[:2] != (3, 11), reason="needs a local CPython 3.11")
def test_does_not_accept_a_checkout_exposed_by_an_old_editable_install(
    install, tmp_path, monkeypatch
):
    target = tmp_path / "old-runtime"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(target)], check=True)
    site = target / "Lib" / "site-packages"
    stale_source = tmp_path / "old-checkout"
    (stale_source / "depth_surge_3d").mkdir(parents=True)
    (stale_source / "depth_surge_3d" / "__init__.py").touch()
    (stale_source / "_quality_kd_native.py").touch()
    (site / "old-checkout.pth").write_text(str(stale_source) + "\n")
    metadata = site / "depth_surge_3d-0.1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: depth-surge-3d\nVersion: 0.1\n")
    (metadata / "RECORD").write_text("old-checkout.pth,,\n")
    (metadata / "direct_url.json").write_text(
        '{"url":"file:///old-checkout","dir_info":{"editable":true}}'
    )
    (target / "Scripts" / "depth-surge-3d.exe").touch()
    monkeypatch.setenv("SYNC_STATUS", "0")
    result, call = install("-EnvironmentPath './old-runtime'")

    assert len(call) == 2, result.stdout
    assert result.returncode != 0, result.stdout
    assert "editable" in result.stdout.lower()
