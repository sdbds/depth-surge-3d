"""Verify the user's timeline against real source and published media, not argv."""

from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.core.settings import validate_settings
from src.depth_surge_3d.core.video_timing import EncodingTimeline
from src.depth_surge_3d.io.operations import get_video_properties, save_processing_settings
from src.depth_surge_3d.processing.frames.vr_assembler import DirectStereoSource
from src.depth_surge_3d.processing.video.final_publication import (
    FinalPublicationTransactionV4,
    InjectedFinalPublicationCrash,
)
from src.depth_surge_3d.processing.video.video_encoder import VideoEncoder


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="FFmpeg required"
    ),
]


def _job(root: Path, *, rate: str, target: int | None, mode: str, audio: bool, old=False):
    source = root / "source.mov"
    inputs = ["-f", "lavfi", "-i", f"testsrc2=size=64x48:rate={rate}"]
    if audio:
        inputs.extend(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"])
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            *inputs,
            "-t",
            "1.001" if "/" in rate else "1",
            "-c:v",
            "libx264",
            *(["-c:a", "pcm_s16le"] if audio else []),
            str(source),
        ],
        check=True,
        timeout=30,
    )
    props = get_video_properties(str(source))
    if old:
        del props["frame_rate"]
    count = props["frame_count"]
    settings = validate_settings(
        dict(
            device="cpu",
            source_width=props["width"],
            source_height=props["height"],
            target_fps=target,
            source_fps=props["fps"],
            vr_format="side_by_side",
            vr_resolution="custom:64x48",
            per_eye_width=64,
            per_eye_height=48,
            vr_output_width=128,
            vr_output_height=48,
            preserve_audio=audio,
            direct_vr_encode=mode != "assembled",
            keep_intermediates=True,
            video_encoder="libx264",
        ),
        source="explicit",
    )
    job = root / "job"
    settings_file = save_processing_settings(
        job, "job", settings, props, str(source), transactional_v5=True
    )
    return source, job, count, settings, settings_file


def _publish(root: Path, *, rate: str, target: int | None, mode: str, audio: bool):
    source, job, count, settings, settings_file = _job(
        root, rate=rate, target=target, mode=mode, audio=audio
    )
    encoder = VideoEncoder()

    # A sharp change at the midpoint also detects output padded with the first or last frame.
    def frames():
        for i in range(count):
            yield np.full((48, 128, 3), 20 if i < count // 2 else 220, np.uint8)

    if mode == "streamed":
        session = encoder.open_streamed_vr_session(
            output_dir=job,
            original_video=str(source),
            settings=settings,
            frame_count=count,
            first_index=1,
        )
        try:
            for i, frame in enumerate(frames(), 1):
                session.write(f"frame_{i:06d}", frame)
            assert session.finish_and_publish()
        finally:
            session.abort()
    elif mode == "assembled":
        vr = job / "99_vr_frames"
        vr.mkdir()
        for i, frame in enumerate(frames(), 1):
            assert cv2.imwrite(str(vr / f"frame_{i:06d}.png"), frame)
        assert encoder.create_video(vr, job, str(source), settings)
    else:
        left, right = job / "06_left_cropped", job / "06_right_cropped"
        left.mkdir()
        right.mkdir()
        for i, frame in enumerate(frames(), 1):
            assert cv2.imwrite(str(left / f"frame_{i:06d}.png"), frame[:, :64])
            assert cv2.imwrite(str(right / f"frame_{i:06d}.png"), frame[:, 64:])
        assert encoder.create_video_from_stereo_source(
            DirectStereoSource(left, right, 1, count),
            job,
            str(source),
            settings,
            total_frames=count,
        )
    saved = json.loads(settings_file.read_bytes())
    assert saved["video_properties"]["frame_rate"] == (rate if "/" in rate else f"{rate}/1")
    assert encoder.pending_final_encoding_reservation is not None
    encoder.pending_final_encoding_reservation.validate_authenticated_publication()
    relative = saved["output_info"]["expected_final_relative_path"]
    output = job / relative
    return output, _probe(output)


def _probe(output):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(output)],
        capture_output=True,
        check=True,
        timeout=30,
    )
    return {s["codec_type"]: s for s in json.loads(probe.stdout)["streams"]}


@pytest.mark.parametrize(("rate", "count"), [("24000/1001", 24), ("30000/1001", 30)])
def test_keep_source_rate_publishes_exact_fraction(tmp_path, rate, count):
    _, streams = _publish(tmp_path, rate=rate, target=None, mode="streamed", audio=False)
    assert streams["video"]["avg_frame_rate"] == rate
    assert int(streams["video"]["nb_read_frames"]) == count


def test_fractional_downconversion_keeps_the_last_partial_output_interval(tmp_path):
    _, streams = _publish(tmp_path, rate="30000/1001", target=24, mode="streamed", audio=True)
    video = streams["video"]
    assert video["avg_frame_rate"] == "24/1"
    assert int(video["nb_read_frames"]) == 25  # The final output tick is at 1.000 s of 1.001 s.
    audio = streams["audio"]
    assert abs(
        Fraction(audio["duration_ts"]) * Fraction(audio["time_base"]) - Fraction(1001, 1000)
    ) < Fraction(1, 1000)


@pytest.mark.parametrize("mode", ["streamed", "direct", "assembled"])
def test_24_to_60_preserves_one_second_of_video_and_audio(tmp_path, mode):
    output, streams = _publish(tmp_path, rate="24", target=60, mode=mode, audio=True)
    assert streams["video"]["avg_frame_rate"] == "60/1"
    assert int(streams["video"]["nb_read_frames"]) == 60
    for stream in (streams["video"], streams["audio"]):
        assert Fraction(stream["duration_ts"]) * Fraction(stream["time_base"]) == 1
    cap = cv2.VideoCapture(str(output))
    try:
        for index, bright in ((12, False), (45, True)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            assert ok and (float(frame.mean()) > 128) == bright
    finally:
        cap.release()


def test_resume_finishes_a_pre_upgrade_publication_without_retiming_its_media(
    tmp_path, monkeypatch
):
    source, job, count, settings, settings_file = _job(
        tmp_path, rate="24", target=60, mode="streamed", audio=True, old=True
    )
    publish = FinalPublicationTransactionV4.publish
    limits = EncodingTimeline.ffmpeg_video_options
    with monkeypatch.context() as before_upgrade:
        # Freeze the old one-clock command and interrupt publication; FFmpeg and disk are real.
        before_upgrade.setattr(
            EncodingTimeline,
            "from_artifact",
            classmethod(lambda *_: EncodingTimeline.resolve(60, 60)),
        )
        before_upgrade.setattr(
            EncodingTimeline,
            "ffmpeg_video_options",
            lambda clock, n, *, drain_audio: limits(clock, n, drain_audio=False),
        )
        before_upgrade.setattr(
            FinalPublicationTransactionV4,
            "publish",
            lambda self: publish(self, crash_at="after-video"),
        )
        session = VideoEncoder().open_streamed_vr_session(
            output_dir=job,
            original_video=str(source),
            settings=settings,
            frame_count=count,
            first_index=1,
        )
        assert session.command[session.command.index("-frames:v") + 1] == str(count)
        for i in range(1, count + 1):
            session.write(f"frame_{i:06d}", np.zeros((48, 128, 3), dtype=np.uint8))
        with pytest.raises(InjectedFinalPublicationCrash):
            session._finish_and_publish()

    output = (
        job / json.loads(settings_file.read_bytes())["output_info"]["expected_final_relative_path"]
    )
    frozen = output.read_bytes()
    before = _probe(output)
    assert before["video"]["avg_frame_rate"] == "60/1"
    assert int(before["video"]["nb_read_frames"]) == 24
    encoder = VideoEncoder()
    assert encoder.create_video_from_stereo_source(
        DirectStereoSource(job / "no-left-frames", job / "no-right-frames", 1, count),
        job,
        str(source),
        settings,
        total_frames=count,
    )
    assert output.read_bytes() == frozen
    assert encoder.pending_final_encoding_reservation is not None
    encoder.pending_final_encoding_reservation.validate_authenticated_publication()
