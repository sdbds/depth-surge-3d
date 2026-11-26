"""Real decoder coverage for bounded final-container probe parallelism."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from src.depth_surge_3d.processing.video import final_container_validation as validation
from src.depth_surge_3d.processing.video.encoding_provider import (
    FinalContainerExpectation,
    FinalContainerValidationError,
)
from src.depth_surge_3d.processing.video.final_publication import (
    validate_final_container_evidence_v1,
)


pytestmark = pytest.mark.integration


def _run(command: list[str]) -> bytes:
    result = subprocess.run(command, capture_output=True, check=True, timeout=30)
    assert not result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def final_media(tmp_path_factory):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and ffprobe are required")
    root = tmp_path_factory.mktemp("final-container-probe")
    encoder_rows = _run(["ffmpeg", "-v", "error", "-hide_banner", "-encoders"])
    encoders = {
        fields[1]
        for line in encoder_rows.decode("utf-8").splitlines()
        if len(fields := line.split()) >= 2
    }
    media = {}

    def load(codec, audio):
        required = {codec, "aac"} if audio else {codec}
        if missing := required - encoders:
            pytest.skip(f"FFmpeg encoders are unavailable: {', '.join(sorted(missing))}")
        if (codec, audio) in media:
            return media[(codec, audio)]
        path = root / f"{codec}-audio{int(audio)}.mp4"
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-n",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x96:rate=24000/1001",
        ]
        if audio:
            command += ["-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000"]
        command += [
            "-frames:v",
            "19",
            "-t",
            "0.792458333",
            "-c:v",
            codec,
            "-preset",
            "medium",
            "-threads:v",
            "1",
            "-g",
            "12",
            "-bf",
            "3",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "setsar=1",
        ]
        if codec == "libx265":
            command += ["-x265-params", "log-level=error:pools=none:frame-threads=1"]
        if audio:
            command += ["-c:a", "aac", "-b:a", "96k"]
        command += ["-movflags", "+faststart", str(path)]
        _run(command)
        expectation = FinalContainerExpectation(
            width=160,
            height=96,
            fps="24000/1001",
            frame_count=19,
            resolved_encoder=codec,
            audio_expected=audio,
            audio_packet_upper_bound=64 if audio else 0,
        )
        media[(codec, audio)] = (path, expectation)
        return path, expectation

    return load


def test_publication_validates_the_opened_movie_with_real_ffmpeg(final_media):
    path, expectation = final_media("libx264", False)
    evidence = validate_final_container_evidence_v1(
        path.parent,
        path,
        lambda opened: validation.validate_final_container_v4(opened, expectation),
    )
    assert evidence.byte_count == path.stat().st_size
    assert evidence.raw_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(sys.platform != "linux", reason="procfs file descriptors are Linux-specific")
def test_unbound_descriptor_is_not_an_ordinary_movie_path(final_media):
    path, expectation = final_media("libx264", False)
    with path.open("rb") as handle:
        with pytest.raises(validation.FinalContainerProcessError, match="not an ordinary file"):
            validation.validate_final_container_v4(
                Path(f"/proc/{os.getpid()}/fd/{handle.fileno()}"), expectation
            )


@pytest.mark.parametrize("codec", ["libx264", "libx265"])
@pytest.mark.parametrize("audio", [False, True])
def test_threaded_final_probe_preserves_complete_counts_and_media(
    final_media, monkeypatch, codec, audio
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    path, expectation = final_media(codec, audio)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    probes = []
    for baseline in (True, False):

        def finite_runner(arguments, **limits):
            command = list(arguments)
            index = command.index("-threads:v")
            assert command[index + 1] == "8"
            if baseline:
                del command[index : index + 2]
            result = validation._run_finite(command, **limits)
            probes.append(result.stdout)
            return result

        assert validation.validate_final_container_v4(
            path, expectation, finite_runner=finite_runner
        )
    assert probes[0] == probes[1]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("damage", ["truncated", "video_packet", "audio_packet"])
@pytest.mark.parametrize("codec", ["libx264", "libx265"])
def test_threaded_final_validation_still_rejects_damaged_media(
    final_media, monkeypatch, tmp_path, damage, codec
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    source, expectation = final_media(codec, True)
    payload = bytearray(source.read_bytes())
    if damage == "truncated":
        del payload[len(payload) // 2 :]
    else:
        stream = "v:0" if damage == "video_packet" else "a:0"
        packets = json.loads(
            _run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    stream,
                    "-show_packets",
                    "-show_entries",
                    "packet=pos,size",
                    "-of",
                    "json",
                    str(source),
                ]
            )
        )["packets"]
        start, size = int(packets[0]["pos"]), int(packets[0]["size"])
        assert 4 < size and start + size <= len(payload)
        if damage == "video_packet":
            payload[start : start + 4] = b"\x7f\xff\xff\xff"
        else:
            payload[start : start + size] = b"\xff" * size
    candidate = tmp_path / "damaged.mp4"
    candidate.write_bytes(payload)
    with pytest.raises((validation.FinalContainerProcessError, FinalContainerValidationError)):
        validation.validate_final_container_v4(candidate, expectation)


@pytest.mark.parametrize("mismatch", ["frame_count", "dimensions", "missing_audio"])
def test_threaded_final_validation_keeps_exact_stream_expectations(
    final_media, monkeypatch, mismatch
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    path, expectation = final_media("libx264", False)
    changes = {
        "frame_count": {"frame_count": 18},
        "dimensions": {"width": 162},
        "missing_audio": {"audio_expected": True, "audio_packet_upper_bound": 64},
    }
    with pytest.raises(FinalContainerValidationError):
        validation.validate_final_container_v4(path, replace(expectation, **changes[mismatch]))
