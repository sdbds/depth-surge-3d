"""Contracts for the Qinglong uv installer dependency profile."""

from __future__ import annotations

import re
from pathlib import Path


INSTALL_SCRIPT = Path(__file__).parents[2] / "1.install-uv-qinglong.ps1"


def test_project_install_commands_include_moge2_by_default() -> None:
    text = INSTALL_SCRIPT.read_text(encoding="utf-8")
    commands = [
        line.strip()
        for line in text.splitlines()
        if line.lstrip().startswith("~/.local/bin/uv pip install")
        and "-r pyproject.toml" in line
    ]

    assert len(commands) == 2
    assert all(re.search(r"(?:^|\s)--extra\s+moge2(?:\s|$)", command) for command in commands)
    assert "dependency profile: base+moge2" in text
    assert "dependency profile: base-only" not in text
    assert 'Check "Install base and MoGe-2 requirements failed"' in text
