"""Hatch wheel hook for the precompiled Quality native extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Compile during wheel creation and mark the wheel platform-specific."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        del version
        if self.target_name != "wheel":
            return

        from scripts.build_quality_task0_native import (
            BUILD_OUTPUT,
            BUILD_TEMP,
            build_native_extension,
        )

        artifact = build_native_extension(
            output_dir=BUILD_OUTPUT,
            build_temp=BUILD_TEMP,
        )
        build_data["force_include"][str(artifact)] = artifact.name
        build_data["pure_python"] = False
        build_data["infer_tag"] = True

    def clean(self, versions: list[str]) -> None:
        del versions
        from scripts.build_quality_task0_native import clean_native_extension

        clean_native_extension()


def _artifact_target(artifact: Path) -> str:
    """Expose the wheel-root target for focused packaging tests."""

    return artifact.name
