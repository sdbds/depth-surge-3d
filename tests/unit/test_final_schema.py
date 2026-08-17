"""Repository-level checks for the intentional final schema break."""

from __future__ import annotations

import json
from pathlib import Path

from src.depth_surge_3d.core.constants import INTERMEDIATE_DIRS
from src.depth_surge_3d.utils.domain import depth_cache


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_intermediate_directory_schema_has_no_runtime_legacy_aliases():
    assert "depth_maps" not in INTERMEDIATE_DIRS
    assert "supersampled" not in INTERMEDIATE_DIRS
    assert "left_final" not in INTERMEDIATE_DIRS
    assert "right_final" not in INTERMEDIATE_DIRS
    assert INTERMEDIATE_DIRS["disparity_maps"] == "03_disparity_maps"


def test_noncanonical_array_cache_api_is_removed():
    assert not hasattr(depth_cache, "get_cached_depth_maps")
    assert not hasattr(depth_cache, "save_depth_maps_to_cache")


def test_dead_frame_by_frame_video_processing_module_is_removed():
    assert not (
        PROJECT_ROOT / "src" / "depth_surge_3d" / "utils" / "imaging" / "video_processing.py"
    ).exists()


def test_current_examples_and_docs_do_not_advertise_removed_stereo_controls():
    settings = json.loads((PROJECT_ROOT / "example_settings.json").read_text(encoding="utf-8"))[
        "processing_settings"
    ]
    for name in ("baseline", "focal_length", "hole_fill_quality"):
        assert name not in settings

    current_docs = [
        PROJECT_ROOT / "docs" / "PARAMETERS.md",
        PROJECT_ROOT / "docs" / "USAGE.md",
        PROJECT_ROOT / "docs" / "ARCHITECTURE.md",
        PROJECT_ROOT / "docs" / "CLAUDE.md",
    ]
    removed_controls = (
        "--baseline",
        "`baseline`",
        "--focal-length",
        "`focal_length`",
        "--hole-fill-quality",
        "`hole_fill_quality`",
    )

    for path in current_docs:
        text = path.read_text(encoding="utf-8").lower()
        for control in removed_controls:
            assert control not in text, f"{path.name} still contains {control}"
