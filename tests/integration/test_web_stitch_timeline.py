"""End-to-end timing of manual exports from the managed Web job directory."""

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pytest

import app as web


pytestmark = pytest.mark.integration


def _run(*args):
    return subprocess.run(args, capture_output=True, check=True).stdout


def _streams(path):
    probe = _run("ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", path)
    return {stream["codec_type"]: stream for stream in json.loads(probe)["streams"]}


@pytest.fixture
def stitch_job(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe are required")
    job = tmp_path / "managed" / "job"
    frames = job / web.INTERMEDIATE_DIRS["vr_frames"]
    frames.mkdir(parents=True)
    for i in range(1, 25):
        image = np.full((16, 32, 3), 8 * i, dtype=np.uint8)
        assert cv2.imwrite(str(frames / f"frame_{i:06d}.png"), image)
    monkeypatch.setitem(web.app.config, "OUTPUT_FOLDER", str(job.parent))
    client = web.app.test_client()

    def export(name, **settings):
        result = client.post(
            "/stitch_video", json={"batch_dir": str(job), "output_filename": name, **settings}
        ).get_json()
        assert result["success"], result
        return Path(result["output_path"])

    return job, export


@pytest.mark.parametrize("recorded_only", [True, False])
def test_original_preserves_fractional_clock_from_job_or_source(stitch_job, recorded_only):
    job, export = stitch_job
    source = job / "camera.mkv"
    properties = {"fps": 23.98, "frame_count": 24}
    if recorded_only:
        properties["frame_rate"] = "24000/1001"
    else:
        _run(
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=s=32x16:r=24000/1001",
            "-frames:v",
            "24",
            "-c:v",
            "ffv1",
            source,
        )
    (job / "job-settings.json").write_text(
        json.dumps(
            {
                "metadata": {"source_video_name": source.name},
                "video_properties": properties,
                "processing_settings": {},
            }
        ),
        encoding="utf-8",
    )

    video = _streams(export("original.mp4", fps="original"))["video"]

    assert Fraction(video["avg_frame_rate"]) == Fraction(24000, 1001)
    assert int(video["nb_read_frames"]) == 24


@pytest.mark.parametrize("audio", ["off", "flac", "source"])
def test_target_fps_keeps_selected_second_and_audio(stitch_job, audio):
    job, export = stitch_job
    source = job / "source.mkv"
    _run(
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=s=32x16:r=24:d=3",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=600:sample_rate=48000:duration=3",
        "-af",
        "volume=0:enable='lt(t,1)+gte(t,2)'",
        "-c:v",
        "ffv1",
        "-c:a",
        "pcm_s16le",
        source,
    )
    if audio == "flac":
        _run(
            "ffmpeg",
            "-v",
            "error",
            "-i",
            source,
            "-vn",
            "-c:a",
            "flac",
            job / "original_audio.flac",
        )
        source.unlink()
    (job / "job-settings.json").write_text(
        json.dumps(
            {
                "metadata": {"source_video_name": source.name},
                "video_properties": {"fps": 24, "frame_count": 72},
                "processing_settings": {"start_time": "0:01", "end_time": "0:02"},
            }
        ),
        encoding="utf-8",
    )

    output = export("converted.mp4", fps=60, include_audio=audio != "off")
    streams = _streams(output)

    assert Fraction(streams["video"]["avg_frame_rate"]) == 60
    assert int(streams["video"]["nb_read_frames"]) == 60
    assert float(streams["video"]["duration"]) == pytest.approx(1, abs=0.001)
    if audio == "off":
        assert "audio" not in streams
    else:
        assert float(streams["audio"]["duration"]) == pytest.approx(1, abs=0.025)
        samples = _run("ffmpeg", "-v", "error", "-i", output, "-map", "0:a:0", "-f", "s16le", "-")
        middle = np.frombuffer(samples, dtype="<i2")[12000:36000]
        assert np.mean(np.abs(middle)) > 1000  # The first and third source seconds were silent.


def test_bare_sequence_does_not_inherit_clock_from_its_previous_export(stitch_job):
    _, export = stitch_job
    first = _streams(export("presentation.mp4", fps=60))["video"]
    second = _streams(export("another edit.mp4", fps="original"))["video"]

    assert int(first["nb_read_frames"]) == int(second["nb_read_frames"]) == 24
    assert Fraction(first["avg_frame_rate"]) == 60
    assert Fraction(second["avg_frame_rate"]) == 30
