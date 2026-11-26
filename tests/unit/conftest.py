"""Common unit-test fixtures for the installed CLI and the Quality contracts."""

from importlib import import_module
from pathlib import Path

import pytest


@pytest.fixture
def cli_module(monkeypatch):
    # In wheel CI the checkout's src symlink points at the installed package.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "src"))
    return import_module("depth_surge_3d.cli")


def _quality_module(name: str):
    @pytest.fixture(scope="module")
    def module():
        return import_module(f"src.depth_surge_3d.rendering.quality.{name}")

    return module


kd = _quality_module("implicit_kd")
indexed_heap = _quality_module("indexed_heap")
region_solver = _quality_module("region_solver")
quality_geometry = _quality_module("geometry")
native_kd = _quality_module("native_kd")
