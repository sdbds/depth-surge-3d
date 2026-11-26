"""The Unix launcher must forward the app's status and never sweep the host."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("app_status", [0, 37])
def test_shell_launcher_preserves_exit_status_and_owns_only_its_app(tmp_path, app_status):
    if os.name == "nt":
        git = shutil.which("git")
        bash = Path(git).resolve().parents[1] / "bin" / "bash.exe" if git else None
    else:
        bash = shutil.which("bash")
    if not bash or not Path(bash).is_file():
        pytest.skip("Requires Bash")

    source = Path(__file__).resolve().parents[2] / "run_ui.sh"
    (tmp_path / "run_ui.sh").write_bytes(source.read_text(encoding="utf-8").encode("utf-8"))
    bin_dir = tmp_path / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "activate").touch()
    stubs = {
        "mkdir": "exit 0",
        "python": (
            'if [ "$1" = app.py ]; then exit "$APP_STATUS"; fi\n'
            'printf "%s\\n" "$*" >> "$PWD/unexpected-python"'
        ),
        "pkill": 'printf "unexpected host sweep" >> "$PWD/host-sweep"',
    }
    for name, body in stubs.items():
        stub = bin_dir / name
        stub.write_bytes(f"#!/bin/bash\n{body}\n".encode("ascii"))
        stub.chmod(0o755)

    result = subprocess.run(
        [bash, "-c", 'export PATH="$PWD/.venv/bin"; source ./run_ui.sh'],
        cwd=tmp_path,
        env={**os.environ, "APP_STATUS": str(app_status)},
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == app_status, result.stderr
    assert not (tmp_path / "host-sweep").exists()
    assert not (tmp_path / "unexpected-python").exists()
