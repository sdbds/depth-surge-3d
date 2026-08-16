"""Mechanical contracts for the public MoGe-2 documentation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


WARNING = (
    "MoGe-2 performs per-frame depth and focal estimation. Temporal stability "
    "on video is not guaranteed; depth or focal drift may be visible across frames."
)

DOC_PATHS = (
    Path("README.md"),
    Path("docs/INSTALLATION.md"),
    Path("docs/PARAMETERS.md"),
    Path("docs/ARCHITECTURE.md"),
    Path("docs/USAGE.md"),
    Path("docs/TROUBLESHOOTING.md"),
    Path("CHANGELOG.md"),
)

MOGE_BASIC_COMMAND = "uv run depth-surge-3d clip.mp4 --depth-model-version moge2 --model-size vitb"
MOGE_METRIC_COMMAND = """uv run depth-surge-3d clip.mp4 --depth-model-version moge2 --model-size vitb \\
  --stereo-geometry-mode metric_camera --format side_by_side --no-distortion \\
  --virtual-baseline-mm 63 --metric-convergence-distance auto \\
  --max-disparity-percent 2"""


@pytest.mark.parametrize(
    "path",
    [Path("README.md"), Path("docs/PARAMETERS.md"), Path("docs/TROUBLESHOOTING.md")],
)
def test_temporal_warning_is_verbatim(path: Path) -> None:
    assert WARNING in path.read_text(encoding="utf-8")


def test_installation_documents_only_supported_optional_command() -> None:
    text = Path("docs/INSTALLATION.md").read_text(encoding="utf-8")
    assert "uv sync --extra moge2" in text
    assert "uv sync --extra requirements-moge2" not in text
    assert "uv sync -r requirements-moge2.txt" not in text


def test_public_docs_do_not_expose_internal_resolution_control() -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in DOC_PATHS)
    assert "MOGE_RESOLUTION_LEVEL" not in text
    assert "moge_resolution_level" not in text


def test_architecture_names_both_independent_stage3_formats() -> None:
    text = Path("docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "03_disparity_maps" in text
    assert "03_metric_geometry" in text
    assert "raw schema v3" in text


def test_architecture_documents_stage_and_cleanup_barriers() -> None:
    text = Path("docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    for contract in (
        "only the selected missing stage is generated",
        "valid inactive\nstage is preserved",
        "validated selected-Stage-3 barrier",
        "completed Stage 3 for resume",
        "Post-preflight\n`ENOSPC`",
        "cleanup does not run after a failed job",
    ):
        assert contract in text


def test_notices_retain_both_upstream_licenses() -> None:
    text = Path("THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "Microsoft MoGe" in text and "MIT License" in text
    assert "DINOv2" in text and "Apache License 2.0" in text
    assert "925b8ed835a7a9cdb7578ba15c658a0afc969030" in text


def test_example_settings_use_safe_metric_defaults_without_internal_level() -> None:
    document = json.loads(Path("example_settings.json").read_text(encoding="utf-8"))
    settings = document["processing_settings"]
    assert settings["stereo_geometry_mode"] == "relative"
    assert settings["virtual_baseline_mm"] == 63.0
    assert settings["metric_convergence_distance"] == "auto"
    assert settings["max_disparity_percent"] == 2.0
    assert "moge_resolution_level" not in settings


@pytest.mark.parametrize("path", [Path("docs/INSTALLATION.md"), Path("docs/PARAMETERS.md")])
def test_documentation_lists_all_immutable_weight_pins(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for repository, revision in (
        ("Ruicheng/moge-2-vits-normal", "679230677b4d282c6f304189a93e98e14f085902"),
        ("Ruicheng/moge-2-vitb-normal", "54ad3a693e61907ea4633d13dec6ee682fa09419"),
        ("Ruicheng/moge-2-vitl", "39c4d5e957afe587e04eec59dc2bcc3be5ecd968"),
    ):
        assert repository in text
        assert revision in text


def test_usage_has_both_complete_cli_examples_and_all_public_flags() -> None:
    text = Path("docs/USAGE.md").read_text(encoding="utf-8")
    assert MOGE_BASIC_COMMAND in text
    assert MOGE_METRIC_COMMAND in text
    for flag in (
        "--depth-model-version moge2",
        "--model-size",
        "--stereo-geometry-mode",
        "--virtual-baseline-mm",
        "--metric-convergence-distance",
        "--max-disparity-percent",
    ):
        assert flag in text


def test_parameters_document_exact_metric_defaults_and_bounds() -> None:
    text = Path("docs/PARAMETERS.md").read_text(encoding="utf-8")
    for contract in (
        "Default: `relative`",
        "finite `0` to `100`. Default: `63.0`",
        "from `0.1` to `1000` metres. Default: `auto`",
        "finite `0` to `5`. Default: `2.0`",
        "total\n  left-to-right disparity in retained final-output coordinates",
    ):
        assert contract in text


def test_docs_do_not_make_prohibited_affirmative_guarantees() -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in DOC_PATHS)
    prohibited = (
        r"guarantees? calibrated physical scale",
        r"guarantees? physically correct reconstruction",
        r"guarantees? improved stereo quality",
        r"guarantees? temporal stability",
        r"guarantees? viewing comfort",
        r"guarantees? viewing safety",
        r"superior to relative mode",
    )
    for pattern in prohibited:
        assert re.search(pattern, text, flags=re.IGNORECASE) is None


def test_example_settings_is_valid_json() -> None:
    json.loads(Path("example_settings.json").read_text(encoding="utf-8"))
