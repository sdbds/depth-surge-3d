"""Hatch wheel hook for the precompiled Quality native extension."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def _load_native_builder() -> ModuleType:
    """Load the repository build helper without relying on the caller's sys.path."""

    path = Path(__file__).resolve().parent / "scripts" / "build_quality_task0_native.py"
    spec = importlib.util.spec_from_file_location("_depth_surge_quality_native_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load the Quality native builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CustomBuildHook(BuildHookInterface):
    """Compile during wheel creation and mark the wheel platform-specific."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        del version
        if self.target_name != "wheel":
            return

        builder = _load_native_builder()
        artifact = builder.build_native_extension(
            output_dir=builder.BUILD_OUTPUT,
            build_temp=builder.BUILD_TEMP,
        )
        build_data["force_include"][str(artifact)] = artifact.name
        build_data["pure_python"] = False
        build_data["infer_tag"] = True

    def clean(self, versions: list[str]) -> None:
        del versions
        _load_native_builder().clean_native_extension()


def _artifact_target(artifact: Path) -> str:
    """Expose the wheel-root target for focused packaging tests."""

    return artifact.name
