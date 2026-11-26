"""Frame and audio content at an exact NTSC transition, including historical jobs."""

import json
from fractions import Fraction
import shutil
import subprocess

import cv2
import numpy as np
import pytest

import app as web
from src.depth_surge_3d.core.settings import validate_settings
from src.depth_surge_3d.io.operations import get_video_properties, save_processing_settings
from src.depth_surge_3d.io.resume import build_resume_report
from src.depth_surge_3d.io.settings_artifact import ProcessingSettingsArtifactV5
from src.depth_surge_3d.processing.frames.vr_assembler import DirectStereoSource
from src.depth_surge_3d.processing.orchestration.pipeline_orchestrator import ProcessingOrchestrator
from src.depth_surge_3d.processing.video.video_encoder import VideoEncoder


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="FFmpeg required"
    ),
]


@pytest.fixture
def transition(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    source = job / "transition.mov"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=red:s=32x16:r=30000/1001:d=2.002",
            "-f",
            "lavfi",
            "-i",
            r"aevalsrc=if(lt(t\,1.001)\,0\,0.3*sin(2*PI*650*t)):s=48000:d=2.002",
            "-vf",
            "drawbox=c=blue:t=fill:enable='eq(n,30)',drawbox=c=lime:t=fill:enable='gt(n,30)'",
            "-c:v",
            "libx264",
            "-g",
            "1",
            "-video_track_timescale",
            "30000",
            "-c:a",
            "pcm_s16le",
            str(source),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    props = get_video_properties(str(source))
    assert props["frame_rate"] == "30000/1001" and props["frame_count"] == 60
    settings = validate_settings(
        dict(
            start_time="0:01.001",
            end_time="0:01.035",
            target_fps=None,
            preserve_audio=True,
            video_encoder="libx264",
            direct_vr_encode=False,
            vr_resolution="custom:16x16",
            per_eye_width=16,
            per_eye_height=16,
            vr_output_width=32,
            vr_output_height=16,
            source_width=32,
            source_height=16,
            source_fps=props["fps"],
            keep_intermediates=True,
        ),
        source="explicit",
    )
    return job, source, props, settings


@pytest.mark.parametrize("exact", [True, False])
def test_extraction_resume_and_saved_count_select_the_same_ntsc_frames(
    transition, exact, monkeypatch
):
    job, source, props, settings = transition
    saved_props = dict(props)
    if not exact:
        del saved_props["frame_rate"]  # Historical schema-5 jobs froze their binary64 calculation.
    settings_file = save_processing_settings(
        job, "job", settings, saved_props, str(source), transactional_v5=True
    )
    encoder = VideoEncoder()
    orchestrator = ProcessingOrchestrator(None, None, None, None, None, encoder)
    extracted = []

    def only_extract(video_path, _root, directories, runtime_props, resolved, _progress):
        extracted.extend(encoder.extract_frames(video_path, directories, runtime_props, resolved))
        return True

    monkeypatch.setattr(orchestrator, "_execute_pipeline", only_extract)
    assert orchestrator.process(source, job, props, settings)
    assert props["frame_rate"] == "30000/1001"  # The original re-probe remains unchanged.

    first = cv2.imread(str(extracted[0]))[0, 0].tolist()
    assert (first[0] > first[2]) is exact, f"first pixel BGR: {first}"
    expected = 1 if exact else 2
    assert len(extracted) == expected
    assert (
        ProcessingSettingsArtifactV5.parse(settings_file.read_bytes()).selected_frame_count
        == expected
    )
    assert build_resume_report(job, settings).stage("frames").disposition == "preserve"


def test_noninteger_microsecond_pts_is_not_skipped_or_included_twice(transition):
    job, source, props, settings = transition
    frames = job / "00_original_frames"
    frames.mkdir()
    extracted = VideoEncoder().extract_frames(
        str(source),
        {"frames": frames},
        props,
        {**settings, "start_time": "0:01.035", "end_time": "0:01.068"},
    )
    assert len(extracted) == 1
    b, g, r = cv2.imread(str(extracted[0]))[0, 0].tolist()
    assert g > 200 and b < 10 and r < 10  # Green frame 31 starts at 1.034366666... s.


@pytest.mark.parametrize("export", ["web", "pipeline", "streamed", "direct"])
def test_ntsc_audio_starts_on_the_first_blue_frame(transition, export):
    job, source, props, settings = transition
    settings = {**settings, "direct_vr_encode": export in {"streamed", "direct"}}
    settings_file = save_processing_settings(
        job, "job", settings, props, str(source), transactional_v5=True
    )
    vr = job / "99_vr_frames"
    vr.mkdir()
    image = np.zeros((16, 32, 3), np.uint8)
    image[:, :, 0] = 255
    assert cv2.imwrite(str(vr / "frame_000001.png"), image)
    if export == "web":
        output = web.create_video_from_batch(
            job, {"include_audio": True, "output_filename": "edit.mp4"}
        )
    else:
        encoder = VideoEncoder()
        if export == "streamed":
            session = encoder.open_streamed_vr_session(
                output_dir=job,
                original_video=str(source),
                settings=settings,
                frame_count=1,
                first_index=1,
            )
            try:
                session.write("frame_000001", image)
                assert session.finish_and_publish()
            finally:
                session.abort()
        elif export == "direct":
            left, right = job / "06_left_cropped", job / "06_right_cropped"
            for directory in (left, right):
                directory.mkdir()
                assert cv2.imwrite(str(directory / "frame_000001.png"), image[:, :16])
            assert encoder.create_video_from_stereo_source(
                DirectStereoSource(left, right, 1, 1), job, str(source), settings, total_frames=1
            )
        else:
            assert encoder.create_video(vr, job, str(source), settings)
        assert encoder.pending_final_encoding_reservation is not None
        encoder.pending_final_encoding_reservation.validate_authenticated_publication()
        output = (
            job
            / json.loads(settings_file.read_bytes())["output_info"]["expected_final_relative_path"]
        )

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    streams = {s["codec_type"]: s for s in json.loads(probe.stdout)["streams"]}
    assert int(streams["video"]["nb_frames"]) == 1
    for stream in streams.values():
        duration = Fraction(stream["duration_ts"]) * Fraction(stream["time_base"])
        assert abs(duration - Fraction(1001, 30000)) < Fraction(1, 1000)
    audio = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(output), "-map", "0:a:0", "-f", "s16le", "-"],
        check=True,
        capture_output=True,
        timeout=30,
    )
    samples = np.frombuffer(audio.stdout, dtype="<i2")
    assert len(samples) >= 1500
    assert np.mean(np.abs(samples[250:1250])) > 3000  # Source was silent through frame 29.
