"""Fast structural checks for pipeline PNG payloads."""

from __future__ import annotations

import cv2
import numpy as np

from src.depth_surge_3d.utils.imaging.png_header import (
    PngHeader,
    png_header_matches,
    read_png_header,
)


def test_header_dataclass_remains_compatible_with_python_39() -> None:
    assert "__slots__" not in vars(PngHeader)


def test_reads_rgb8_ihdr_without_decoding_pixels(tmp_path) -> None:
    path = tmp_path / "rgb.png"
    assert cv2.imwrite(str(path), np.zeros((7, 11, 3), dtype=np.uint8))

    header = read_png_header(path)

    assert header is not None
    assert (header.width, header.height) == (11, 7)
    assert header.bit_depth == 8
    assert header.channels == 3
    assert png_header_matches(path, shape=(7, 11, 3), bit_depth=8)


def test_reads_grayscale_uint16_ihdr(tmp_path) -> None:
    path = tmp_path / "depth.png"
    assert cv2.imwrite(str(path), np.zeros((5, 9), dtype=np.uint16))

    header = read_png_header(path)

    assert header is not None
    assert (header.width, header.height) == (9, 5)
    assert header.bit_depth == 16
    assert header.channels == 1
    assert png_header_matches(path, shape=(5, 9), bit_depth=16)


def test_rejects_truncated_or_non_png_payloads(tmp_path) -> None:
    truncated = tmp_path / "truncated.png"
    truncated.write_bytes(b"\x89PNG\r\n\x1a\n")
    unrelated = tmp_path / "unrelated.png"
    unrelated.write_bytes(b"not a png")

    assert read_png_header(truncated) is None
    assert read_png_header(unrelated) is None
    assert not png_header_matches(truncated, shape=(1, 1, 3), bit_depth=8)
