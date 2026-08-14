"""Real FFmpeg evidence for direct stereo sequence encoding."""

import json
import math
import shutil
import subprocess
import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.processing.video.video_encoder import VideoEncoder


pytestmark = pytest.mark.integration

_FRAME_COUNT = 3
_FRAME_RATE = "24000/1001"
_SCALE_FLAGS = "bicubic+accurate_rnd"


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and ffprobe are required")


def _write_stereo_fixture(root: Path, frame_count: int = _FRAME_COUNT):
    left_dir = root / "06_left_cropped"
    right_dir = root / "06_right_cropped"
    left_dir.mkdir()
    right_dir.mkdir()
    left_files = []
    right_files = []
    for index in range(1, frame_count + 1):
        left = np.zeros((48, 64, 3), np.uint8)
        left[:, :] = (20 * index, 40, 220)
        cv2.putText(left, "L", (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        right = np.zeros((48, 64, 3), np.uint8)
        right[:, :] = (220, 40, 20 * index)
        cv2.putText(right, "R", (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        left_path = left_dir / f"frame_{index:06d}.png"
        right_path = right_dir / f"frame_{index:06d}.png"
        assert cv2.imwrite(str(left_path), left)
        assert cv2.imwrite(str(right_path), right)
        left_files.append(left_path)
        right_files.append(right_path)
    return left_files, right_files


def _direct_settings(vr_format: str, *, preserve_audio: bool = False) -> dict:
    return {
        "target_fps": _FRAME_RATE,
        "vr_format": vr_format,
        "vr_resolution": "custom",
        "per_eye_width": 64,
        "per_eye_height": 48,
        "preserve_audio": preserve_audio,
        "video_encoder": "libx264",
    }


def _probe(path: Path, *, count_frames: bool = False) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        *(("-count_frames",) if count_frames else ()),
        "-show_entries",
        "stream=index,codec_type,codec_name,width,height,avg_frame_rate,nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    return json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)


def _read_first_frame(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        decoded, frame = capture.read()
    finally:
        capture.release()
    assert decoded
    assert frame is not None
    return frame


def _letter_mask(letter: str) -> np.ndarray:
    image = np.zeros((48, 64), np.uint8)
    cv2.putText(image, letter, (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    return image > 200


def _assert_eye_content(eye: np.ndarray, *, letter: str, dominant_channel: int) -> None:
    background = eye[38:46, 42:60].mean(axis=(0, 1))
    other_channels = np.delete(background, dominant_channel)
    assert background[dominant_channel] > float(other_channels.max()) + 80

    visible_text = np.min(eye, axis=2) > 150
    expected_mask = _letter_mask(letter)
    other_mask = _letter_mask("R" if letter == "L" else "L")
    expected_coverage = float(visible_text[expected_mask].mean())
    other_coverage = float(visible_text[other_mask].mean())
    assert expected_coverage > 0.60
    assert expected_coverage > other_coverage + 0.10


def _assert_no_direct_intermediates(root: Path, output: Path) -> None:
    assert output.is_file()
    assert output.stat().st_size > 0
    assert not (root / "99_vr_frames").exists()
    assert not (root / "99_vr_frames" / "metadata.json").exists()
    assert list(root.glob(".*.direct.tmp.mp4")) == []


@pytest.mark.parametrize(
    ("vr_format", "expected_name", "expected_width", "expected_height"),
    [
        ("side_by_side", "source_3D_side-by-side_custom.mp4", 128, 48),
        ("over_under", "source_3D_over-under_custom.mp4", 64, 96),
    ],
)
def test_libx264_direct_encode_preserves_layout_rate_and_count(
    tmp_path, vr_format, expected_name, expected_width, expected_height
):
    _require_ffmpeg()
    left_files, right_files = _write_stereo_fixture(tmp_path)
    output = tmp_path / expected_name

    assert VideoEncoder().create_video_from_stereo_sequences(
        left_files,
        right_files,
        tmp_path,
        str(tmp_path / "source.mp4"),
        _direct_settings(vr_format),
        total_frames=_FRAME_COUNT,
    )

    _assert_no_direct_intermediates(tmp_path, output)
    streams = _probe(output, count_frames=True)["streams"]
    assert len(streams) == 1
    stream = streams[0]
    assert stream["codec_type"] == "video"
    assert stream["codec_name"] == "h264"
    assert (stream["width"], stream["height"]) == (expected_width, expected_height)
    assert stream["avg_frame_rate"] == _FRAME_RATE
    assert stream["nb_read_frames"] == str(_FRAME_COUNT)

    frame = _read_first_frame(output)
    assert frame.shape == (expected_height, expected_width, 3)
    if vr_format == "side_by_side":
        left_eye, right_eye = frame[:, :64], frame[:, 64:]
    else:
        left_eye, right_eye = frame[:48], frame[48:]
    _assert_eye_content(left_eye, letter="L", dominant_channel=2)
    _assert_eye_content(right_eye, letter="R", dominant_channel=0)


def test_direct_encode_maps_generated_flac_to_one_aac_stream(tmp_path):
    _require_ffmpeg()
    left_files, right_files = _write_stereo_fixture(tmp_path)
    audio = tmp_path / "original_audio.flac"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:a",
            "flac",
            str(audio),
        ],
        check=True,
    )
    output = tmp_path / "source_3D_side-by-side_custom.mp4"

    assert VideoEncoder().create_video_from_stereo_sequences(
        left_files,
        right_files,
        tmp_path,
        str(tmp_path / "source.mp4"),
        _direct_settings("side_by_side", preserve_audio=True),
        total_frames=_FRAME_COUNT,
    )

    _assert_no_direct_intermediates(tmp_path, output)
    streams = _probe(output)["streams"]
    video_streams = [stream for stream in streams if stream["codec_type"] == "video"]
    audio_streams = [stream for stream in streams if stream["codec_type"] == "audio"]
    assert len(video_streams) == 1
    assert video_streams[0]["codec_name"] == "h264"
    assert len(audio_streams) == 1
    assert audio_streams[0]["codec_name"] == "aac"


def _write_structured_resize_fixture(path: Path) -> np.ndarray:
    height, width = 55, 73
    y, x = np.indices((height, width), dtype=np.uint16)
    image = np.empty((height, width, 3), np.uint8)
    image[:, :, 0] = ((3 * x + 2 * y) % 256).astype(np.uint8)
    image[:, :, 1] = ((x + 4 * y) % 256).astype(np.uint8)
    image[:, :, 2] = ((5 * x + y) % 256).astype(np.uint8)
    cv2.rectangle(image, (5, 6), (31, 25), (12, 238, 61), -1)
    cv2.rectangle(image, (39, 9), (68, 31), (241, 31, 183), 2)
    image[34:49, 7:37] = np.where(
        ((x[34:49, 7:37] + y[34:49, 7:37]) % 2)[..., None] == 0,
        np.array((230, 35, 210), np.uint8),
        np.array((20, 220, 45), np.uint8),
    )
    cv2.line(image, (42, 38), (68, 38), (255, 255, 255), 2)
    cv2.line(image, (42, 38), (42, 50), (255, 255, 255), 2)
    cv2.line(image, (48, 44), (68, 44), (8, 8, 8), 1)
    cv2.line(image, (48, 50), (64, 50), (8, 8, 8), 1)
    assert cv2.imwrite(str(path), image)
    return image


def _psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    mse = np.mean((reference.astype(np.float64) - candidate.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 20 * math.log10(255.0 / math.sqrt(mse))


def _ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    values = []
    for channel in range(3):
        x = reference[:, :, channel].astype(np.float64)
        y = candidate[:, :, channel].astype(np.float64)
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
        sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
        numerator = (2 * mu_x * mu_y + 6.5025) * (2 * sigma_xy + 58.5225)
        denominator = (mu_x * mu_x + mu_y * mu_y + 6.5025) * (sigma_x + sigma_y + 58.5225)
        values.append(float(np.mean(numerator / denominator)))
    return float(np.mean(values))


def test_ffmpeg_bicubic_resize_reports_precompression_quality(tmp_path):
    _require_ffmpeg()
    source_path = tmp_path / "structured_source.png"
    ffmpeg_path = tmp_path / "ffmpeg_bicubic.png"
    source = _write_structured_resize_fixture(source_path)
    target_width, target_height = 64, 48
    legacy = cv2.resize(source, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    scale_filter = f"scale={target_width}:{target_height}:flags={_SCALE_FLAGS}"
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-vf",
        scale_filter,
        "-frames:v",
        "1",
        "-c:v",
        "png",
        str(ffmpeg_path),
    ]

    assert scale_filter == "scale=64:48:flags=bicubic+accurate_rnd"
    subprocess.run(command, check=True)
    candidate = cv2.imread(str(ffmpeg_path), cv2.IMREAD_COLOR)
    assert candidate is not None
    assert candidate.shape == (target_height, target_width, 3)

    psnr = _psnr(legacy, candidate)
    ssim = _ssim(legacy, candidate)
    print(f"Resize diagnostic: PSNR={psnr:.4f} dB, SSIM={ssim:.6f}")
    if psnr < 30.0:
        warnings.warn(f"Resize PSNR {psnr:.4f} dB is below 30 dB", stacklevel=2)
    if ssim < 0.95:
        warnings.warn(f"Resize SSIM {ssim:.6f} is below 0.95", stacklevel=2)
