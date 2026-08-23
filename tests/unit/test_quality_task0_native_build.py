"""Packaging contract for build-time native Quality artifacts."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_pyproject_registers_non_jit_native_wheel_build_hook() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert project["tool"]["hatch"]["build"]["hooks"]["custom"] == {"path": "hatch_build.py"}
    assert {"setuptools>=84", "numpy>=1.24,<2", "torch==2.13.0"}.issubset(
        project["build-system"]["requires"]
    )


def test_ci_builds_every_supported_native_wheel_cell() -> None:
    workflow = Path(".github/workflows/quality-native-wheels.yml").read_text(encoding="utf-8")

    for runner in ("windows-latest", "ubuntu-24.04", "ubuntu-24.04-arm"):
        assert runner in workflow
    for version in ('"3.10"', '"3.11"', '"3.12"'):
        assert version in workflow
    assert "python -m hatchling build -t wheel" in workflow
    assert "runtime_jit" in workflow
    assert "actions/upload-artifact@v4" in workflow
