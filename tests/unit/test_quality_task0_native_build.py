"""Packaging contract for build-time native Quality artifacts."""

from __future__ import annotations

import tomllib
from pathlib import Path
import sys

import hatch_build
from packaging.requirements import Requirement


def test_pyproject_registers_non_jit_native_wheel_build_hook() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert project["tool"]["hatch"]["build"]["hooks"]["custom"] == {"path": "hatch_build.py"}
    build_deps = {Requirement(req).name for req in project["build-system"]["requires"]}
    assert "pybind11" in build_deps
    assert build_deps.isdisjoint({"numpy", "torch"})


def test_hatch_hook_loads_builder_without_project_root_on_sys_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = Path(hatch_build.__file__).resolve().parent
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "path",
        [entry for entry in sys.path if Path(entry or ".").resolve() != project_root],
    )

    builder = hatch_build._load_native_builder()

    assert builder.ROOT == project_root
    assert builder.SOURCE == project_root / "native" / "quality_kd_native.cpp"
