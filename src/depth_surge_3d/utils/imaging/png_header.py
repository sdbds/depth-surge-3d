"""Constant-time structural validation for PNG pipeline payloads."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_IHDR_SIZE = 33
_CHANNELS_BY_COLOR_TYPE = {
    0: 1,
    2: 3,
    3: 1,
    4: 2,
    6: 4,
}


@dataclass(frozen=True)
class PngHeader:
    width: int
    height: int
    bit_depth: int
    color_type: int
    channels: int


def read_png_header(path: Path | str) -> PngHeader | None:
    """Read only the signature and IHDR; pixel data is intentionally untouched."""

    try:
        with Path(path).open("rb") as handle:
            payload = handle.read(_IHDR_SIZE)
        if (
            len(payload) != _IHDR_SIZE
            or payload[:8] != _PNG_SIGNATURE
            or payload[8:12] != b"\x00\x00\x00\r"
            or payload[12:16] != b"IHDR"
        ):
            return None
        width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
            ">IIBBBBB", payload[16:29]
        )
        channels = _CHANNELS_BY_COLOR_TYPE.get(color_type)
        if (
            width < 1
            or height < 1
            or channels is None
            or compression != 0
            or filtering != 0
            or interlace not in {0, 1}
        ):
            return None
        return PngHeader(width, height, bit_depth, color_type, channels)
    except (OSError, struct.error):
        return None


def png_header_matches(
    path: Path | str,
    *,
    shape: Sequence[int],
    bit_depth: int,
) -> bool:
    """Check the dimensions, channel count, and sample width encoded in IHDR."""

    if len(shape) not in {2, 3}:
        return False
    header = read_png_header(path)
    expected_channels = 1 if len(shape) == 2 else int(shape[2])
    return bool(
        header
        and header.height == int(shape[0])
        and header.width == int(shape[1])
        and header.channels == expected_channels
        and header.bit_depth == bit_depth
    )
