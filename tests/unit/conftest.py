"""Common unit-test fixtures for the installed CLI and the Quality contracts."""

from importlib import import_module
from pathlib import Path

import pytest


@pytest.fixture
def cli_module(monkeypatch):
    # In wheel CI the checkout's src symlink points at the installed package.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "src"))
    return import_module("depth_surge_3d.cli")


@pytest.fixture(
    scope="module",
    params=["scripts.quality_task0", "src.depth_surge_3d.rendering.quality"],
    ids=["prototype", "production"],
)
def quality_contract_package(request):
    return request.param


def _quality_module(name, *, prototype_name=None):
    @pytest.fixture(scope="module")
    def module(quality_contract_package):
        basename = (
            prototype_name
            if prototype_name and quality_contract_package.startswith("scripts.")
            else name
        )
        return import_module(f"{quality_contract_package}.{basename}")

    return module


kd = _quality_module("implicit_kd")
indexed_heap = _quality_module("indexed_heap")
region_solver = _quality_module("region_solver")
quality_geometry = _quality_module("geometry", prototype_name="quality_geometry")
repair_records = _quality_module("repair_records")
local_strip = _quality_module("local_strip")
native_repair = _quality_module("native_repair")
native_kd = _quality_module("native_kd")
fallback = _quality_module("fallback")
exemplar = _quality_module("exemplar")
