"""Lightweight contracts for resumable generated frame stages."""

from __future__ import annotations

import cv2
import numpy as np

from src.depth_surge_3d.processing.frames.stage_manifest import (
    build_stage_identity,
    complete_stage,
    stage_is_reusable,
)


def _write_rgb(path, shape=(4, 6, 3)) -> None:
    assert cv2.imwrite(str(path), np.zeros(shape, dtype=np.uint8))


def test_completed_pair_stage_is_reusable_without_decoding(tmp_path) -> None:
    source_left = tmp_path / "source_left"
    source_right = tmp_path / "source_right"
    output_left = tmp_path / "output_left"
    output_right = tmp_path / "output_right"
    for directory in (source_left, source_right, output_left, output_right):
        directory.mkdir()
    left = source_left / "frame_000001.png"
    right = source_right / "frame_000001.png"
    _write_rgb(left)
    _write_rgb(right)
    _write_rgb(output_left / left.name)
    _write_rgb(output_right / right.name)
    identity = build_stage_identity(
        stage="test_pair",
        algorithm_version="test-v1",
        frame_names=[left.name],
        source_files=[left, right],
        settings={"mode": "fast"},
    )

    assert complete_stage(identity, (output_left, output_right), shape=(4, 6, 3))
    assert stage_is_reusable(identity, (output_left, output_right))


def test_source_metadata_change_invalidates_completed_stage(tmp_path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "output"
    output.mkdir()
    _write_rgb(source)
    _write_rgb(output / source.name)
    identity = build_stage_identity(
        stage="test_single",
        algorithm_version="test-v1",
        frame_names=[source.name],
        source_files=[source],
        settings={},
    )
    assert complete_stage(identity, (output,), shape=(4, 6, 3))

    source.touch()
    changed = build_stage_identity(
        stage="test_single",
        algorithm_version="test-v1",
        frame_names=[source.name],
        source_files=[source],
        settings={},
    )

    assert not stage_is_reusable(changed, (output,))


def test_structurally_invalid_output_is_not_reusable(tmp_path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "output"
    output.mkdir()
    _write_rgb(source)
    output_file = output / source.name
    _write_rgb(output_file)
    identity = build_stage_identity(
        stage="test_single",
        algorithm_version="test-v1",
        frame_names=[source.name],
        source_files=[source],
        settings={},
    )
    assert complete_stage(identity, (output,), shape=(4, 6, 3))
    output_file.write_bytes(b"corrupt")

    assert not stage_is_reusable(identity, (output,))
