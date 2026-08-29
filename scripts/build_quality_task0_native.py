"""Build the Task 0 native extension explicitly at package-build time."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import sysconfig
from pathlib import Path

import numpy as np
from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import include_paths


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native" / "quality_kd_native.cpp"
BUILD_TEMP = ROOT / "build" / "quality-task0-native" / sysconfig.get_platform()
BUILD_OUTPUT = ROOT / "build" / "quality-task0-native-artifacts" / sysconfig.get_platform()


def _compile_arguments() -> list[str]:
    if os.name == "nt":
        return ["/O2", "/std:c++17", "/EHsc", "/permissive-", "/fp:strict"]
    return ["-O3", "-std=c++17", "-fvisibility=hidden", "-ffp-contract=off", "-pthread"]


def _link_arguments() -> list[str]:
    return [] if os.name == "nt" else ["-pthread"]


def build_native_extension(
    *,
    output_dir: Path = ROOT,
    build_temp: Path = BUILD_TEMP,
) -> Path:
    """Compile to the repository root; importing never invokes this function."""

    build_temp.mkdir(parents=True, exist_ok=True)
    source_sha256 = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    identity_header = build_temp / "quality_kd_build_identity.h"
    identity_header.write_text(
        "#pragma once\n" f'#define QUALITY_KD_SOURCE_SHA256 "{source_sha256}"\n',
        encoding="ascii",
    )
    extension = Extension(
        "_quality_kd_native",
        sources=[str(SOURCE)],
        include_dirs=[str(build_temp), np.get_include(), *include_paths()],
        language="c++",
        extra_compile_args=_compile_arguments(),
        extra_link_args=_link_arguments(),
    )
    distribution = Distribution(
        {
            "name": "depth-surge-quality-task0-native",
            "ext_modules": [extension],
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_ext(distribution)
    command.build_lib = str(output_dir)
    command.build_temp = str(build_temp)
    command.ensure_finalized()
    command.run()
    artifact = output_dir / command.get_ext_filename(extension.name)
    if not artifact.is_file():
        raise RuntimeError(f"Native build did not produce the expected artifact: {artifact}")
    return artifact


def clean_native_extension() -> None:
    """Remove only artifacts whose exact extension suffix matches this module."""

    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not isinstance(suffix, str) or not suffix:
        raise RuntimeError("Python did not report a native extension suffix")
    for output_dir in (ROOT, BUILD_OUTPUT):
        artifact = output_dir / f"_quality_kd_native{suffix}"
        if artifact.is_file():
            artifact.unlink()
    for directory in (BUILD_TEMP, BUILD_OUTPUT):
        if directory.is_dir():
            shutil.rmtree(directory)


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--clean":
        clean_native_extension()
        return 0
    if len(sys.argv) != 1:
        raise SystemExit("usage: build_quality_task0_native.py [--clean]")
    print(build_native_extension())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
