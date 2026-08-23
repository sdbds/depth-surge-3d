"""Command-line entry point for the fail-closed stereo Quality Task 0 gates."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.quality_task0 import reporting


def parse_gate_selection(argv: list[str], available: tuple[str, ...]) -> tuple[str, ...]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gate", action="append", default=[])
    arguments = parser.parse_args(argv)
    return reporting.select_gate_names(arguments.gate, available)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", action="append", default=[])
    parser.add_argument("--list", action="store_true", dest="list_gates")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/superpowers/reports/2026-08-23-stereo-quality-task0.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from scripts.quality_task0 import benchmarks

    arguments = _parser().parse_args(argv)
    available = benchmarks.available_gate_names()
    if arguments.list_gates:
        for name in available:
            print(name)
        return 0
    selected = reporting.select_gate_names(arguments.gate, available)
    report = benchmarks.run_task0_gate_suite(selected, report_path=arguments.report)
    reporting.publish_report(arguments.report, report)
    print(arguments.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
