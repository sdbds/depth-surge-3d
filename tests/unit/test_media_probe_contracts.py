"""Only a complete, matching FFprobe record can certify an encoded movie."""

from dataclasses import replace
import hashlib
import json
from pathlib import PurePosixPath

import pytest

from src.depth_surge_3d.processing.video import encoding_provider as media


VIDEO = dict(
    codec_type="video",
    codec_name="h264",
    width=16,
    height=8,
    pix_fmt="yuv420p",
    sample_aspect_ratio="1:1",
    display_aspect_ratio="2:1",
    avg_frame_rate="24000/1001",
    nb_read_frames="24",
)
AUDIO = dict(codec_type="audio", codec_name="aac", nb_read_frames="48", avg_frame_rate="0/0")
EXPECTED = media.FinalContainerExpectation(16, 8, "24000/1001", 24, "libx264", True, 48)


def probe(*streams, **wrapper):
    return json.dumps(dict(streams=list(streams), **wrapper)).encode()


@pytest.mark.parametrize(
    "encoder,codec", [("libx264", "h264"), ("libx265", "hevc"), ("hevc_nvenc", "hevc")]
)
def test_probe_accepts_codec_streams_in_either_order(encoder, codec):
    video = dict(VIDEO, codec_name=codec)
    for order in ((video, AUDIO), (AUDIO, video)):
        streams = media.parse_final_ffprobe_json(probe(*order, stream_groups=[], programs=[]))
        media.validate_final_probe_streams(streams, replace(EXPECTED, resolved_encoder=encoder))


@pytest.mark.parametrize(
    "stream,field,value,reason",
    [
        ("video", "codec_type", "subtitle", "topology"),
        ("video", "codec_name", "hevc", "codec"),
        ("video", "width", 8, "dimensions"),
        ("video", "height", True, "dimensions"),
        ("video", "pix_fmt", "yuv444p", "pixel format"),
        ("video", "sample_aspect_ratio", "3:2", "sample aspect"),
        ("video", "display_aspect_ratio", "1:1", "display aspect"),
        ("video", "avg_frame_rate", "24/1", "frame rate"),
        ("video", "nb_read_frames", "23", "frame count"),
        ("video", "nb_read_frames", None, "incomplete"),
        ("audio", "nb_read_frames", "49", "packet"),
        ("audio", "codec_name", "flac", "audio codec"),
        ("audio", "codec_name", None, "incomplete"),
        ("audio", "avg_frame_rate", "24/1", "audio frame rate"),
    ],
)
def test_probe_rejects_mismatched_stream_without_publishing(stream, field, value, reason):
    video, audio = dict(VIDEO), dict(AUDIO)
    altered = video if stream == "video" else audio
    if value is None:
        del altered[field]
    else:
        altered[field] = value
    with pytest.raises(media.FinalContainerValidationError, match=reason):
        media.validate_final_probe_streams(
            media.parse_final_ffprobe_json(probe(video, audio)), EXPECTED
        )


def test_audio_presence_is_part_of_the_contract():
    silent = replace(EXPECTED, audio_expected=False, audio_packet_upper_bound=0)
    media.validate_final_probe_streams(media.parse_final_ffprobe_json(probe(VIDEO)), silent)
    with pytest.raises(media.FinalContainerValidationError, match="missing"):
        media.validate_final_probe_streams(media.parse_final_ffprobe_json(probe(VIDEO)), EXPECTED)
    with pytest.raises(media.FinalContainerValidationError, match="audio"):
        media.validate_final_probe_streams(
            media.parse_final_ffprobe_json(probe(VIDEO, AUDIO)), silent
        )


@pytest.mark.parametrize(
    "source,resolution,expected",
    [
        ("trailer.final.mov", "auto", "trailer.final_3D_over-under_auto.mp4"),
        ("clip?", "custom:64x32", "clip_3D_over-under_custom-64x32.mp4"),
        ("", "auto", "output_3D_over-under_auto.mp4"),
        (".sample", "auto", ".sample_3D_over-under_auto.mp4"),
    ],
)
def test_final_media_name_is_stable_for_resume(source, resolution, expected):
    assert (
        media.job_output_name_v1(
            source_video_name=source, vr_format="over_under", resolution=resolution
        )
        == expected
    )


def test_distinct_long_unicode_movie_names_keep_distinct_portable_destinations():
    head = "\N{CJK UNIFIED IDEOGRAPH-5F71}" * 90
    names = []
    for suffix in ("a", "b"):
        raw = head + suffix + "_3D_side-by-side_auto.mp4"
        digest = hashlib.sha256(raw.encode()).hexdigest()[:32]
        name = media.job_output_name_v1(
            source_video_name=head + suffix + ".mov", vr_format="side_by_side", resolution="auto"
        )
        assert name.endswith(f"~{digest}_3D_side-by-side_auto.mp4")
        assert len(name.encode()) <= 240 and media.is_portable_final_component_v1(name)
        names.append(name)
    assert names[0] != names[1]


@pytest.mark.parametrize(
    "overrides",
    [
        {"source_video_name": 12},
        {"source_video_name": "clip\x00.mp4"},
        {"resolution": "custom:10001x480"},
        {"resolution": "640x480"},
        {"vr_format": "mono"},
    ],
)
def test_invalid_job_metadata_cannot_select_an_output_filename(overrides):
    args = dict(source_video_name="clip.mp4", resolution="auto", vr_format="side_by_side")
    with pytest.raises(media.FinalMediaTargetMetadataError):
        media.job_output_name_v1(**(args | overrides))


@pytest.mark.parametrize("mode", ["assembled_vr", "direct_stereo"])
def test_image2_plan_keeps_first_frame_count_and_fractional_rate_on_all_inputs(mode):
    direct = mode == "direct_stereo"
    parents = (
        dict(left_parent=PurePosixPath("left"), right_parent=PurePosixPath("right"))
        if direct
        else dict(frames_parent=PurePosixPath("vr"))
    )
    provider = media.EncodingSequenceProvider.freeze(
        mode=mode, first_index=7, frame_count=3, **parents
    )
    params = dict(
        ffmpeg="ffmpeg",
        fps="24000/1001",
        vr_format="over_under",
        per_eye_size=(16, 8),
        first_headers=((16, 8), (16, 8)) if direct else ((16, 16),),
        output_path="work.mp4",
    )
    argv = media.build_image_encoding_argv(provider, **params)
    assert [argv[i + 1] for i, x in enumerate(argv) if x == "-framerate"] == ["24000/1001"] * len(
        parents
    )
    assert [argv[i + 1] for i, x in enumerate(argv) if x == "-start_number"] == ["7"] * len(parents)
    assert argv[-3:] == ("-frames:v", "3", "work.mp4")
    assert [argv[i + 1] for i, x in enumerate(argv) if x == "-i"] == [
        str(p / "frame_%06d.png") for p in parents.values()
    ]
    if direct:
        scaled = media.build_image_encoding_argv(
            provider, **(params | {"first_headers": ((8, 4), (16, 8))})
        )
        assert "scale=16:8" in scaled[scaled.index("-filter_complex") + 1]
        assert "vstack=inputs=2:shortest=1" in scaled[scaled.index("-filter_complex") + 1]
    for invalid in (
        {"fps": "0/1"},
        {"per_eye_size": (0, 8)},
        {"vr_format": "mono"},
        {"first_headers": ()},
    ):
        with pytest.raises((ValueError, media.EncodingProviderError)):
            media.build_image_encoding_argv(provider, **(params | invalid))
    if not direct:
        with pytest.raises(media.EncodingProviderError, match="shape"):
            media.build_image_encoding_argv(provider, **(params | {"first_headers": ((16, 8),)}))
    for invalid in (
        {"last_index": 3},
        {"mode": "other"},
        {"generation_fingerprint": "broken"},
        {"audit_generation_id": "broken"},
        {"generation_fingerprint": "0" * 64},
        {"left_parent": None} if direct else {"left_parent": PurePosixPath("left")},
    ):
        with pytest.raises(media.EncodingProviderError):
            list(replace(provider, **invalid).replay())


@pytest.mark.parametrize(
    "change",
    [
        dict(width=0),
        dict(frame_count=0),
        dict(fps="0/1"),
        dict(resolved_encoder="copy"),
        dict(audio_expected=1),
        dict(audio_packet_upper_bound=2.5),
        dict(audio_packet_upper_bound=0),
        dict(audio_expected=False, audio_packet_upper_bound=1),
    ],
)
def test_impossible_probe_contract_is_rejected_before_running_ffmpeg(change):
    with pytest.raises(media.FinalContainerValidationError):
        replace(EXPECTED, **change)


@pytest.mark.parametrize("sample_rate,channels", [("48000", 1), ("44100", 2), ("96000", 6)])
def test_source_audio_probe_skips_program_metadata_not_stream_measurements(sample_rate, channels):
    raw = probe(
        dict(codec_type="audio", sample_rate=sample_rate, channels=channels),
        programs=[{"tags": {"title": "Concert"}, "ids": [1, None, False]}],
        stream_groups=[],
    )
    assert media.parse_audio_ffprobe_json(raw) == media.AudioProbeStream(
        "audio", int(sample_rate), channels
    )
    assert media.parse_audio_ffprobe_json(probe()) is None


@pytest.mark.parametrize(
    "fields",
    [
        {"codec_type": "video", "sample_rate": "48000", "channels": 2},
        {"codec_type": "audio", "channels": 2},
        {"codec_type": "audio", "sample_rate": 48000, "channels": 2},
        {"codec_type": "audio", "sample_rate": "0", "channels": 2},
        {"codec_type": "audio", "sample_rate": "4294967296", "channels": 2},
        {"codec_type": "audio", "sample_rate": "48000", "channels": True},
        {"codec_type": "audio", "sample_rate": "48000", "channels": 0},
    ],
)
def test_invalid_audio_measurement_is_not_confused_with_no_audio(fields):
    with pytest.raises(media.FfprobeJsonError):
        media.parse_audio_ffprobe_json(probe(fields))


@pytest.mark.parametrize("reader", [media.parse_audio_ffprobe_json, media.parse_final_ffprobe_json])
@pytest.mark.parametrize(
    "raw",
    [
        b"{}",
        b'{"unrelated":[]}',
        b'{"streams":[{}',
        b'{"streams":[],"streams":[]}',
        b'{"programs":[null],"streams":[]}',
        b'{"streams":[{"unknown":3}]}',
    ],
)
def test_incomplete_or_ambiguous_probe_output_fails_closed(reader, raw):
    with pytest.raises(media.FfprobeJsonError):
        reader(raw)


@pytest.mark.parametrize(
    "parser_type", [media.FfmpegProgressParser, media.FinalDecodeProgressParser]
)
def test_progress_requires_complete_events_and_measures_forward_motion(parser_type):
    for bad in (b"frame=1", b"incomplete\n", b"frame=\x00\n", b"progress=other\n"):
        with pytest.raises(media.EncodingProgressError):
            parser_type().feed_line(bad)
    parser = parser_type()
    assert parser.feed_line(b"frame=2\r\n")
    assert not parser.feed_line(b"frame=1\n")
    assert not parser.feed_line(b"out_time_us=N/A\n")
    assert not parser.feed_line(b"progress=continue\n")
    with pytest.raises(media.EncodingProgressError):
        parser.finish()
    parser.feed_line(b"progress=end\n")
    parser.finish()
    with pytest.raises(media.EncodingProgressError):
        parser.feed_line(b"frame=3\n")
