"""Reject wheels that look installed but misrepresent their native module or CLI."""

import hashlib
from importlib.util import find_spec
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

from packaging.tags import sys_tags
import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def isolated_python(tmp_path_factory):
    env = tmp_path_factory.mktemp("provenance") / "venv"
    subprocess.run([sys.executable, "-I", "-m", "venv", env], cwd=env.parent, check=True)
    return env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("python shim", "not a compiled extension"),
        ("universal wheel", "not a platform wheel"),
        ("wrong entry point", "wrong console entry point"),
        ("unowned launcher", "launcher is not owned by this distribution"),
    ],
)
def test_native_gate_rejects_mispackaged_installation(fault, reason, isolated_python, tmp_path):
    tag = "py3-none-any" if fault == "universal wheel" else str(next(sys_tags()))
    info = "depth_surge_3d-1.dist-info"
    target = "wrong" if fault == "wrong entry point" else "main"
    files = {
        "depth_surge_3d/__init__.py": "",
        "depth_surge_3d/cli.py": "def main(): pass\ndef wrong(): pass\n",
        f"{info}/METADATA": "Metadata-Version: 2.1\nName: depth-surge-3d\nVersion: 1\n",
        f"{info}/WHEEL": f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: {tag}\n",
        f"{info}/entry_points.txt": f"[console_scripts]\ndepth-surge-3d=depth_surge_3d.cli:{target}\n",
    }
    if fault == "python shim":
        digest = hashlib.sha256((ROOT / "native/quality_kd_native.cpp").read_bytes()).hexdigest()
        files["_quality_kd_native.py"] = (
            f"def build_info(): return dict(source_sha256='{digest}', "
            "runtime_jit=False, kd_stack_capacity=64)\n"
        )
    else:
        binary = Path(find_spec("_quality_kd_native").origin)
        files[binary.name] = binary.read_bytes()
    record = f"{info}/RECORD"
    files[record] = "".join(f"{name},,\n" for name in [*files, record])
    wheel = tmp_path / f"depth_surge_3d-1-{tag}.whl"
    with ZipFile(wheel, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    subprocess.run(
        [
            isolated_python,
            "-I",
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--force-reinstall",
            wheel,
        ],
        cwd=tmp_path,
        check=True,
    )
    if fault == "unowned launcher":
        script = "import sysconfig;print(sysconfig.get_path('platlib'))"
        site = Path(
            subprocess.check_output([isolated_python, "-I", "-c", script], text=True).strip()
        )
        (site / record).write_text(files[record], encoding="utf-8")
    result = subprocess.run(
        [
            isolated_python,
            "-I",
            ROOT / "scripts/verify_installed_quality.py",
            "--native-only",
            "--wheel",
            wheel,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, result.stdout
    assert reason in result.stderr, result.stderr
