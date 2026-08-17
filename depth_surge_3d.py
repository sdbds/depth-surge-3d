#!/usr/bin/env python3
"""Backward-compatible launcher for the packaged Depth Surge 3D CLI."""

from __future__ import annotations

import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parent / "src"
source_root = str(SOURCE_ROOT)
if source_root in sys.path:
    sys.path.remove(source_root)
sys.path.insert(0, source_root)

from depth_surge_3d.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
