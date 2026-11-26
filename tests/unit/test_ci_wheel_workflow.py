"""CI must exercise the wheel it publishes, independently of style checks."""

from __future__ import annotations

import itertools
from importlib import metadata, util
import os
from pathlib import Path
import shlex
import subprocess
import sys
from zipfile import ZipFile

from coverage import CoverageData
import pytest
import yaml


WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


def _jobs(name):
    return yaml.load((WORKFLOWS / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)["jobs"]


def test_ci_builds_every_supported_native_wheel_cell():
    job = _jobs("quality-native-wheels.yml")["build-native-wheel"]
    matrix = job["strategy"]["matrix"]
    assert set(itertools.product(matrix["runner"], matrix["python"])) == set(
        itertools.product(
            ("windows-latest", "ubuntu-24.04", "ubuntu-24.04-arm"), ("3.10", "3.11", "3.12")
        )
    )
    assert not matrix.get("exclude")
    assert job["strategy"]["fail-fast"] == "false"
    (installer_step,) = (
        step for step in job["steps"] if "test_install_uv_qinglong.py" in step.get("run", "")
    )
    assert "runner.os == 'Windows'" in installer_step["if"]
    assert "matrix.python == '3.11'" in installer_step["if"]


def test_functional_job_consumes_matrix_artifact_without_waiting_for_static_checks():
    ci = _jobs("ci.yml")
    test = ci["test"]
    build_id = test["needs"]
    assert ci[build_id]["uses"] == "./.github/workflows/quality-native-wheels.yml"
    assert not ci[build_id].get("needs")

    build = _jobs("quality-native-wheels.yml")["build-native-wheel"]
    upload = next(step for step in build["steps"] if "upload-artifact" in step.get("uses", ""))
    download = next(step for step in test["steps"] if "download-artifact" in step.get("uses", ""))
    expected_name = upload["with"]["name"]
    for expression, value in (
        ("runner.os", "Linux"),
        ("runner.arch", "X64"),
        ("matrix.python", "3.11"),
    ):
        expected_name = expected_name.replace("${{ " + expression + " }}", value)
    assert download["with"]["name"] == expected_name


def test_coverage_gate_requires_the_unit_data_and_rejects_below_85_percent(tmp_path):
    steps = _jobs("ci.yml")["test"]["steps"]
    unit = next(step for step in steps if "--cov-report=" in step.get("run", ""))
    gate = next(step for step in steps if "-m coverage report" in step.get("run", ""))
    assert unit["env"]["COVERAGE_FILE"] == gate["env"]["COVERAGE_FILE"]
    assert gate.get("continue-on-error", "false") == "false"
    data_path = tmp_path / "unit.coverage"
    source = tmp_path / "sample.py"
    source.write_text("".join(f"value = {line}\n" for line in range(20)), encoding="utf-8")
    command = shlex.split(gate["run"])
    assert command[0] == "python"

    def run_gate():
        return subprocess.run(
            [sys.executable, *command[1:]],
            cwd=tmp_path,
            env={**os.environ, "COVERAGE_FILE": str(data_path)},
            capture_output=True,
        ).returncode

    assert run_gate() != 0
    data = CoverageData(basename=str(data_path))
    data.add_lines({str(source): set(range(1, 17))})
    data.write()
    assert run_gate() != 0  # 16/20
    data.add_lines({str(source): {17}})
    data.write()
    assert run_gate() == 0  # 17/20


@pytest.mark.skipif(os.environ.get("GITHUB_ACTIONS") != "true", reason="hosted wheel test suite")
def test_legacy_and_standard_imports_use_real_installed_files_in_ci():
    checkout = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    assert not Path.cwd().resolve().is_relative_to(checkout), "fixtures must not shadow the wheel"
    site = Path(metadata.distribution("depth-surge-3d").locate_file("")).resolve()
    assert site.is_relative_to(Path(sys.prefix).resolve())
    for name in ("depth_surge_3d", "src.depth_surge_3d"):
        assert Path(util.find_spec(name).origin).resolve() == site / "depth_surge_3d/__init__.py"
    assert Path(util.find_spec("_quality_kd_native").origin).resolve().parent == site


def test_native_wheel_step_uses_fresh_install_even_with_polluted_pythonpath(tmp_path):
    checkout = tmp_path / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / "ci_probe.py").write_text("origin = 'checkout'\n", encoding="utf-8")
    verifier = checkout / "scripts" / "verify_installed_quality.py"
    verifier.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "from importlib.metadata import distribution\n"
        "import ci_probe\n"
        "assert ci_probe.origin == 'wheel'\n"
        "assert sys.flags.isolated and sys.prefix != sys.base_prefix\n"
        "assert Path(ci_probe.__file__).is_relative_to(Path(sys.prefix))\n"
        "assert not Path.cwd().is_relative_to(Path(__file__).resolve().parents[1])\n"
        "wheel = Path(sys.argv[sys.argv.index('--wheel') + 1])\n"
        "assert wheel.is_absolute() and wheel.is_file() and '--native-only' in sys.argv\n"
        "assert distribution('ci-probe').read_text('direct_url.json')\n"
        "print('verified installed probe')\n",
        encoding="utf-8",
    )
    runner = tmp_path / "runner"
    wheelhouse = runner / "wheelhouse"
    wheelhouse.mkdir(parents=True)
    files = {
        "ci_probe.py": "origin = 'wheel'\n",
        "ci_probe-1.dist-info/METADATA": "Metadata-Version: 2.1\nName: ci-probe\nVersion: 1\n",
        "ci_probe-1.dist-info/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = "ci_probe-1.dist-info/RECORD"
    files[record] = "".join(f"{name},,\n" for name in [*files, record])
    with ZipFile(wheelhouse / "ci_probe-1-py3-none-any.whl", "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)

    steps = _jobs("quality-native-wheels.yml")["build-native-wheel"]["steps"]
    checks = [step for step in steps if "--native-only" in step.get("run", "")]
    assert len(checks) == 1, "each matrix cell must verify the installed wheel"
    check = checks[0]
    assert check["working-directory"] == "${{ runner.temp }}"
    assert check["shell"] == "python"

    def run_check():
        return subprocess.run(
            [sys.executable, "-c", check["run"]],
            cwd=runner,
            env={**os.environ, "GITHUB_WORKSPACE": str(checkout), "PYTHONPATH": str(checkout)},
            capture_output=True,
            text=True,
        )

    result = run_check()
    assert result.returncode == 0, result.stderr
    assert "verified installed probe" in result.stdout
    other = wheelhouse / "ci_probe-1-2-py3-none-any.whl"
    other.write_bytes((wheelhouse / "ci_probe-1-py3-none-any.whl").read_bytes())
    assert run_check().returncode != 0, "a stale second wheel must not be silently selected"
