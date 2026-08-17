"""Pinned VDPP source-vendoring integrity tests."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path


def test_vendored_manifest_binds_every_inference_file() -> None:
    package = Path("src/depth_surge_3d/_vendor/vdpp")
    manifest = json.loads((package / "UPSTREAM.json").read_text(encoding="utf-8"))

    assert manifest["repository"] == "https://github.com/injun-baek/VDPP"
    assert manifest["release"] == "v1.0"
    assert manifest["revision"] == "73cc2b4dc6b3b5cfb2e37f51e452461e03fe26f5"
    assert manifest["vendor_port_version"] == 1
    assert manifest["files"]
    for record in manifest["files"]:
        path = Path("src/depth_surge_3d/_vendor") / record["vendored_path"]
        assert path.is_file(), record
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["vendored_sha256"]


def test_vendored_subset_excludes_demo_and_assets_and_ships_license() -> None:
    vendor_root = Path("src/depth_surge_3d/_vendor")

    assert (vendor_root / "vdpp/LICENSE").is_file()
    assert (vendor_root / "vdpp/NOTICE.md").is_file()
    assert not (vendor_root / "run_video.py").exists()
    assert not (vendor_root / "assets").exists()
    assert not (vendor_root / "external").exists()


def test_vendored_imports_are_package_relative_and_model_imports() -> None:
    vendor_root = Path("src/depth_surge_3d/_vendor")
    sources = "\n".join(path.read_text(encoding="utf-8") for path in vendor_root.rglob("*.py"))

    assert "from vdpp" not in sources
    assert "from utils.normal_utils" not in sources
    module = importlib.import_module("src.depth_surge_3d._vendor.vdpp.vdpp_model")
    assert hasattr(module, "VDPP")
    assert hasattr(module, "compute_scale_and_shift")
