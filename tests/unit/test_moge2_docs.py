"""Mechanical contracts for the public MoGe-2 documentation."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
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

EXPECTED_VARIANTS = (
    (
        "Small / `vits`",
        "35M",
        "`Ruicheng/moge-2-vits-normal`",
        "`679230677b4d282c6f304189a93e98e14f085902`",
    ),
    (
        "Base / `vitb` default",
        "104M",
        "`Ruicheng/moge-2-vitb-normal`",
        "`54ad3a693e61907ea4633d13dec6ee682fa09419`",
    ),
    (
        "Large / `vitl`",
        "326M",
        "`Ruicheng/moge-2-vitl`",
        "`39c4d5e957afe587e04eec59dc2bcc3be5ecd968`",
    ),
)

MOGE_COMMIT = "925b8ed835a7a9cdb7578ba15c658a0afc969030"
MOGE_SOURCE_URL = f"https://github.com/microsoft/MoGe/tree/{MOGE_COMMIT}"
MOGE_LICENSE_URL = f"https://github.com/microsoft/MoGe/blob/{MOGE_COMMIT}/LICENSE"
DINOV2_BUNDLED_URL = f"{MOGE_SOURCE_URL}/moge/model/dinov2"
DINOV2_SOURCE_URL = "https://github.com/facebookresearch/dinov2"
DINOV2_LICENSE_URL = "https://github.com/facebookresearch/dinov2/blob/main/LICENSE"
EXPECTED_MOGE_NOTICE_LINKS = (
    ("Microsoft MoGe at the pinned commit", MOGE_SOURCE_URL),
    ("MIT License in the pinned source", MOGE_LICENSE_URL),
)
EXPECTED_DINOV2_NOTICE_LINKS = (
    ("DINOv2 directory in pinned MoGe", DINOV2_BUNDLED_URL),
    ("Meta Platforms DINOv2", DINOV2_SOURCE_URL),
    ("Apache License 2.0", DINOV2_LICENSE_URL),
)


def _variant_contract_matches(text: str) -> bool:
    """Keep every variant fact bound to its complete Markdown table row."""
    header = "| UI/setting | Parameters | Repository | Revision |"
    lines = text.splitlines()
    try:
        header_index = next(index for index, line in enumerate(lines) if line.strip() == header)
    except StopIteration:
        return False

    rows = []
    for line in lines[header_index + 1 :]:
        if not line.strip().startswith("|"):
            break
        cells = tuple(cell.strip() for cell in line.strip().strip("|").split("|"))
        if len(cells) == 4 and not all(set(cell) <= {"-", ":"} for cell in cells):
            rows.append(cells)
    return tuple(rows) == EXPECTED_VARIANTS


def _markdown_section(text: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\s*$\n(?P<body>.*?)(?=^## |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    return match.group("body") if match else ""


def _notice_contract_matches(text: str) -> bool:
    """Bind every upstream link label and target to its notice section."""
    moge = _markdown_section(text, "Microsoft MoGe")
    dinov2 = _markdown_section(text, "DINOv2")
    link_pattern = re.compile(r"\[([^\]\n]+)\]\(([^)\n]+)\)")
    return (
        tuple(link_pattern.findall(moge)) == EXPECTED_MOGE_NOTICE_LINKS
        and tuple(link_pattern.findall(dinov2)) == EXPECTED_DINOV2_NOTICE_LINKS
    )


def _find_prohibited_claims(text: str) -> list[str]:
    """Find affirmative claims while allowing explicit nearby negation."""
    prohibited = (
        r"\b(?:provides?|guarantees?|ensures?)\s+(?:a\s+)?calibrated physical scale\b",
        r"\b(?:is|provides?|guarantees?)\s+(?:a\s+)?physically correct(?: reconstruction)?\b",
        r"\b(?:improves?|guarantees?|ensures?)\s+(?:the\s+)?(?:improved\s+)?stereo quality\b",
        r"\b(?:is|guarantees?|ensures?)\s+temporally stable\b",
        r"\b(?:provides?|guarantees?|ensures?)\s+temporal stability\b",
        r"\b(?:guarantees?|ensures?|provides?)\s+viewing (?:comfort|safety)\b",
        r"\bis\s+(?:better than|superior to)\s+(?:the\s+)?relative mode\b",
    )
    negation = re.compile(
        r"(?:\b(?:do|does|did|is|are|was|were|can|could|will|would)\s+not"
        r"|\b(?:don't|doesn't|isn't|aren't|wasn't|weren't|can't|cannot|couldn't)\b)"
        r"(?:\s+(?:actually|always|necessarily|reliably))?\s*$",
        re.IGNORECASE,
    )
    normalized = re.sub(r"\s+", " ", text)
    matches = []
    for pattern in prohibited:
        for match in re.finditer(pattern, normalized, re.IGNORECASE):
            if not negation.search(normalized[max(0, match.start() - 80) : match.start()]):
                matches.append(match.group(0))
    return matches


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
    assert _notice_contract_matches(text)
    assert "925b8ed835a7a9cdb7578ba15c658a0afc969030" in text


def test_notice_contract_rejects_swapped_license_associations() -> None:
    swapped = """## Microsoft MoGe

- Project/source: https://github.com/microsoft/MoGe
- License: Apache License 2.0

## DINOv2

- Project/source: https://github.com/facebookresearch/dinov2
- License: MIT License
"""
    assert not _notice_contract_matches(swapped)


def test_notice_contract_requires_pinned_dinov2_bundled_source_link() -> None:
    notices = Path("THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    without_bundled_source = notices.replace(
        f"- Bundled source: [DINOv2 directory in pinned MoGe]({DINOV2_BUNDLED_URL})\n",
        "",
    )

    assert without_bundled_source != notices
    assert not _notice_contract_matches(without_bundled_source)


def test_notice_contract_rejects_swapped_license_link_targets() -> None:
    notices = Path("THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    swapped = notices.replace(MOGE_LICENSE_URL, "NOTICE_LICENSE_PLACEHOLDER")
    swapped = swapped.replace(DINOV2_LICENSE_URL, MOGE_LICENSE_URL)
    swapped = swapped.replace("NOTICE_LICENSE_PLACEHOLDER", DINOV2_LICENSE_URL)

    assert not _notice_contract_matches(swapped)


def test_notice_contract_rejects_unpinned_moge_source_url() -> None:
    notices = Path("THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    unpinned = notices.replace(MOGE_SOURCE_URL, "https://github.com/microsoft/MoGe", 1)

    assert unpinned != notices
    assert not _notice_contract_matches(unpinned)


def test_notice_contract_rejects_link_moved_to_wrong_section() -> None:
    notices = Path("THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    bundled_line = f"- Bundled source: [DINOv2 directory in pinned MoGe]({DINOV2_BUNDLED_URL})\n"
    moved = notices.replace(bundled_line, "")
    moved = moved.replace("## DINOv2", f"{bundled_line}\n## DINOv2")

    assert moved != notices
    assert not _notice_contract_matches(moved)


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
    assert _variant_contract_matches(text)


def test_variant_contract_rejects_swapped_revision_relationships() -> None:
    swapped = """| UI/setting | Parameters | Repository | Revision |
| --- | ---: | --- | --- |
| Small / `vits` | 35M | `Ruicheng/moge-2-vits-normal` | `54ad3a693e61907ea4633d13dec6ee682fa09419` |
| Base / `vitb` default | 104M | `Ruicheng/moge-2-vitb-normal` | `679230677b4d282c6f304189a93e98e14f085902` |
| Large / `vitl` | 326M | `Ruicheng/moge-2-vitl` | `39c4d5e957afe587e04eec59dc2bcc3be5ecd968` |
"""
    assert not _variant_contract_matches(swapped)


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
    assert _find_prohibited_claims(text) == []


@pytest.mark.parametrize(
    "claim",
    (
        "Metric mode provides calibrated physical scale.",
        "The reconstruction is physically correct.",
        "This mode improves stereo quality.",
        "The per-frame result is temporally stable.",
        "The disparity limit ensures viewing comfort.",
        "Metric mode is better than relative mode.",
        "Metric mode guarantees calibrated physical scale.",
        "Metric mode guarantees physically correct reconstruction.",
        "Metric mode guarantees improved stereo quality.",
        "Metric mode guarantees temporal stability.",
        "Metric mode guarantees viewing safety.",
        "Metric mode is superior to relative mode.",
    ),
)
def test_prohibited_claim_detector_rejects_common_affirmative_forms(claim: str) -> None:
    assert _find_prohibited_claims(claim), claim


@pytest.mark.parametrize(
    "limitation",
    (
        "Metric mode does not provide calibrated physical scale.",
        "The reconstruction is not physically correct.",
        "Improved stereo quality is not guaranteed.",
        "Temporal stability is not guaranteed.",
        "The disparity limit does not ensure viewing comfort or safety.",
        "Metric mode is not better than relative mode.",
    ),
)
def test_prohibited_claim_detector_allows_explicit_limitations(limitation: str) -> None:
    assert _find_prohibited_claims(limitation) == []


def test_packaged_cli_module_and_script_target_exist(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import depth_surge_3d.cli as cli; assert callable(cli.main)",
        ],
        capture_output=True,
        check=False,
        cwd=tmp_path,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert 'depth-surge-3d = "depth_surge_3d.cli:main"' in Path("pyproject.toml").read_text(
        encoding="utf-8"
    )


def test_packaged_cli_help_succeeds(tmp_path: Path) -> None:
    executable = shutil.which("depth-surge-3d")
    assert executable is not None
    result = subprocess.run(
        [executable, "--help"],
        capture_output=True,
        check=False,
        cwd=tmp_path,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Convert 2D videos to immersive 3D VR format" in result.stdout


def test_example_settings_is_valid_json() -> None:
    json.loads(Path("example_settings.json").read_text(encoding="utf-8"))
