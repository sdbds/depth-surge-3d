"""Production contracts for replayable final-encoding inputs and bounded parsers."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import sys
from typing import Literal

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.processing.video import encoding_provider


def _provider(
    *,
    mode: Literal["direct_stereo", "assembled_vr"] = "assembled_vr",
    frame_count: int = 3,
    first_index: int = 0,
) -> encoding_provider.EncodingSequenceProvider:
    if mode == "assembled_vr":
        return encoding_provider.EncodingSequenceProvider.freeze(
            mode=mode,
            frame_count=frame_count,
            first_index=first_index,
            frames_parent=PurePosixPath("99_vr_frames"),
        )
    return encoding_provider.EncodingSequenceProvider.freeze(
        mode=mode,
        frame_count=frame_count,
        first_index=first_index,
        left_parent=PurePosixPath("07_left_upscaled"),
        right_parent=PurePosixPath("07_right_upscaled"),
    )


def test_provider_generation_fingerprint_binds_only_scalar_generation_state() -> None:
    provider = _provider(mode="direct_stereo", frame_count=10, first_index=89)

    expected = encoding_provider.canonical_json_bytes(
        {
            "audit_generation_id": provider.audit_generation_id,
            "mode": "direct_stereo",
            "frame_count": 10,
            "first_index": 89,
            "last_index": 98,
            "left_parent": "07_left_upscaled",
            "right_parent": "07_right_upscaled",
            "frames_parent": None,
        }
    )

    assert provider.generation_fingerprint == hashlib.sha256(expected).hexdigest()
    assert not hasattr(provider, "items")
    assert sys.getsizeof(provider) < 1024


def test_replay_constructs_canonical_items_without_an_eager_path_list() -> None:
    provider = _provider(mode="direct_stereo", frame_count=3, first_index=999_999)

    items = list(provider.replay())

    assert [item.source_index for item in items] == [0, 1, 2]
    assert [item.frame_index for item in items] == [999_999, 1_000_000, 1_000_001]
    assert [item.frame_name for item in items] == [
        "frame_999999",
        "frame_1000000",
        "frame_1000001",
    ]
    assert items[0].left_relative_path == PurePosixPath("07_left_upscaled/frame_999999.png")
    assert items[0].right_relative_path == PurePosixPath("07_right_upscaled/frame_999999.png")
    assert items[0].frame_relative_path is None


@pytest.mark.parametrize(
    ("first_index", "frame_count"),
    [(2_147_483_646, 2), (2_147_483_647, 1)],
)
def test_image2_signed_endpoint_is_rejected_before_provider_construction(
    first_index: int,
    frame_count: int,
) -> None:
    with pytest.raises(encoding_provider.EncodingImage2RangeError):
        _provider(frame_count=frame_count, first_index=first_index)


def test_image2_two_frame_upper_boundary_is_admitted() -> None:
    provider = _provider(frame_count=2, first_index=2_147_483_645)

    assert provider.last_index == 2_147_483_646
    assert [item.frame_index for item in provider.replay()] == [2_147_483_645, 2_147_483_646]


def test_provider_replay_change_uses_pre_and_post_launch_error_types() -> None:
    provider = _provider(frame_count=3)
    evidence = encoding_provider.audit_encoding_provider(provider)
    changed = encoding_provider.EncodingSequenceProvider(
        mode=provider.mode,
        frame_count=provider.frame_count,
        first_index=provider.first_index,
        last_index=provider.last_index,
        left_parent=provider.left_parent,
        right_parent=provider.right_parent,
        frames_parent=PurePosixPath("changed"),
        audit_generation_id=provider.audit_generation_id,
        generation_fingerprint=provider.generation_fingerprint,
    )

    with pytest.raises(encoding_provider.EncodingProviderChangedError):
        encoding_provider.validate_encoding_provider(changed, evidence, launch_started=False)
    with pytest.raises(encoding_provider.FinalEncodingInputChangedError):
        encoding_provider.validate_encoding_provider(changed, evidence, launch_started=True)


def _write_png(path: Path, width: int, height: int, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((height, width, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_manifest_prepass_streams_one_identity_at_a_time_and_postpass_matches(
    tmp_path: Path,
) -> None:
    provider = _provider(frame_count=3)
    for item in provider.replay():
        assert item.frame_relative_path is not None
        _write_png(tmp_path / item.frame_relative_path, 8, 4, item.source_index)
    sink = io.BytesIO()

    staged = encoding_provider.stage_encoding_input_manifest(
        provider,
        job_root=tmp_path,
        sink=sink,
        per_eye_size=(4, 4),
        vr_format="side_by_side",
    )

    payload = json.loads(sink.getvalue())
    assert payload["fingerprint"] == staged.fingerprint
    without_fingerprint = dict(payload)
    without_fingerprint.pop("fingerprint")
    assert (
        staged.fingerprint
        == hashlib.sha256(encoding_provider.canonical_json_bytes(without_fingerprint)).hexdigest()
    )
    assert staged.raw_sha256 == hashlib.sha256(sink.getvalue()).hexdigest()
    assert staged.trace.maximum_live_identities == 1
    assert staged.trace.retained_item_lists == 0
    encoding_provider.validate_encoding_postpass(
        provider,
        job_root=tmp_path,
        staged=staged,
        per_eye_size=(4, 4),
        vr_format="side_by_side",
    )


def test_manifest_postpass_detects_any_image_byte_change(tmp_path: Path) -> None:
    provider = _provider(frame_count=1)
    item = next(provider.replay())
    assert item.frame_relative_path is not None
    path = tmp_path / item.frame_relative_path
    _write_png(path, 8, 4, 10)
    staged = encoding_provider.stage_encoding_input_manifest(
        provider,
        job_root=tmp_path,
        sink=io.BytesIO(),
        per_eye_size=(4, 4),
        vr_format="side_by_side",
    )
    _write_png(path, 8, 4, 11)

    with pytest.raises(encoding_provider.FinalEncodingInputChangedError):
        encoding_provider.validate_encoding_postpass(
            provider,
            job_root=tmp_path,
            staged=staged,
            per_eye_size=(4, 4),
            vr_format="side_by_side",
        )


def test_manifest_prepass_cancels_during_image_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(frame_count=1)
    item = next(provider.replay())
    assert item.frame_relative_path is not None
    _write_png(tmp_path / item.frame_relative_path, 64, 64, 10)
    monkeypatch.setattr(encoding_provider, "FINAL_ENCODING_IMAGE_STREAM_BYTES", 16)
    polls = 0

    def cancelled() -> bool:
        nonlocal polls
        polls += 1
        return polls >= 7

    with pytest.raises(InterruptedError, match="cancelled"):
        encoding_provider.stage_encoding_input_manifest(
            provider,
            job_root=tmp_path,
            sink=io.BytesIO(),
            per_eye_size=(32, 64),
            vr_format="side_by_side",
            cancelled=cancelled,
        )


def test_manifest_resume_and_postpass_are_cancellable(tmp_path: Path) -> None:
    provider = _provider(frame_count=1)
    item = next(provider.replay())
    assert item.frame_relative_path is not None
    _write_png(tmp_path / item.frame_relative_path, 8, 4, 10)
    staged = encoding_provider.stage_encoding_input_manifest(
        provider,
        job_root=tmp_path,
        sink=io.BytesIO(),
        per_eye_size=(4, 4),
        vr_format="side_by_side",
    )

    with pytest.raises(InterruptedError, match="cancelled"):
        encoding_provider.audit_encoding_input_manifest(
            provider,
            job_root=tmp_path,
            expected_fingerprint=staged.fingerprint,
            expected_raw_sha256=staged.raw_sha256,
            expected_raw_byte_count=staged.raw_byte_count,
            per_eye_size=(4, 4),
            vr_format="side_by_side",
            cancelled=lambda: True,
        )
    with pytest.raises(InterruptedError, match="cancelled"):
        encoding_provider.validate_encoding_postpass(
            provider,
            job_root=tmp_path,
            staged=staged,
            per_eye_size=(4, 4),
            vr_format="side_by_side",
            cancelled=lambda: True,
        )


def test_directory_prepass_rejects_an_unmanifested_trailing_image(tmp_path: Path) -> None:
    provider = _provider(frame_count=1)
    item = next(provider.replay())
    assert item.frame_relative_path is not None
    _write_png(tmp_path / item.frame_relative_path, 8, 4, 1)
    _write_png(tmp_path / "99_vr_frames/frame_000001.png", 8, 4, 2)

    with pytest.raises(encoding_provider.EncodingDirectoryError, match="count|range"):
        encoding_provider.stage_encoding_input_manifest(
            provider,
            job_root=tmp_path,
            sink=io.BytesIO(),
            per_eye_size=(4, 4),
            vr_format="side_by_side",
        )


@pytest.mark.parametrize(
    "name",
    [
        "frame_bad.png",
        "frame_000000.PNG",
        "FRAME_000000.png",
        "frame_+000000.png",
    ],
)
def test_directory_prepass_rejects_every_malformed_frame_domain_name(
    tmp_path: Path,
    name: str,
) -> None:
    provider = _provider(frame_count=1)
    item = next(provider.replay())
    assert item.frame_relative_path is not None
    if name.casefold() != "frame_000000.png":
        _write_png(tmp_path / item.frame_relative_path, 8, 4, 1)
    _write_png(tmp_path / "99_vr_frames" / name, 8, 4, 2)

    with pytest.raises(encoding_provider.EncodingDirectoryError, match="noncanonical"):
        encoding_provider.stage_encoding_input_manifest(
            provider,
            job_root=tmp_path,
            sink=io.BytesIO(),
            per_eye_size=(4, 4),
            vr_format="side_by_side",
        )


def test_directory_prepass_rejects_a_linked_parent(tmp_path: Path) -> None:
    real_parent = tmp_path / "real_frames"
    real_parent.mkdir()
    _write_png(real_parent / "frame_000000.png", 8, 4, 1)
    alias = tmp_path / "99_vr_frames"
    try:
        os.symlink(real_parent, alias, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this host")
    provider = _provider(frame_count=1)

    with pytest.raises(encoding_provider.EncodingDirectoryError):
        encoding_provider.stage_encoding_input_manifest(
            provider,
            job_root=tmp_path,
            sink=io.BytesIO(),
            per_eye_size=(4, 4),
            vr_format="side_by_side",
        )


def test_direct_argv_uses_scalar_patterns_and_exact_no_scale_filter() -> None:
    provider = _provider(mode="direct_stereo", frame_count=17, first_index=42)

    argv = encoding_provider.build_image_encoding_argv(
        provider,
        ffmpeg="ffmpeg",
        fps="24000/1001",
        vr_format="side_by_side",
        per_eye_size=(1920, 1080),
        first_headers=((1920, 1080), (1920, 1080)),
        output_path=".final-video.safe.mp4",
    )

    assert argv.count("-start_number") == 2
    assert argv.count("-frames:v") == 1
    assert "42" in argv and "17" in argv
    assert "07_left_upscaled/frame_%06d.png" in argv
    assert "07_right_upscaled/frame_%06d.png" in argv
    graph = argv[argv.index("-filter_complex") + 1]
    assert graph == (
        "[0:v]setsar=1[left];[1:v]setsar=1[right];"
        "[left][right]hstack=inputs=2:shortest=1[stacked];[stacked]setsar=1[vr]"
    )


def test_direct_argv_scales_both_eyes_when_either_header_differs() -> None:
    provider = _provider(mode="direct_stereo")

    argv = encoding_provider.build_image_encoding_argv(
        provider,
        ffmpeg="ffmpeg",
        fps="30/1",
        vr_format="over_under",
        per_eye_size=(1280, 720),
        first_headers=((1280, 720), (1279, 720)),
        output_path=".final-video.safe.mp4",
    )

    graph = argv[argv.index("-filter_complex") + 1]
    assert graph == (
        "[0:v]scale=1280:720:flags=bicubic+accurate_rnd,setsar=1[left];"
        "[1:v]scale=1280:720:flags=bicubic+accurate_rnd,setsar=1[right];"
        "[left][right]vstack=inputs=2:shortest=1[stacked];[stacked]setsar=1[vr]"
    )


def test_assembled_argv_has_explicit_range_and_square_pixel_filter() -> None:
    provider = _provider(frame_count=4, first_index=7)

    argv = encoding_provider.build_image_encoding_argv(
        provider,
        ffmpeg="ffmpeg",
        fps="60/1",
        vr_format="side_by_side",
        per_eye_size=(1920, 1080),
        first_headers=((3840, 1080),),
        output_path=".final-video.safe.mp4",
    )

    assert argv[argv.index("-start_number") + 1] == "7"
    assert argv[argv.index("-frames:v") + 1] == "4"
    assert argv[argv.index("-vf") + 1] == "setsar=1"
    assert "99_vr_frames/frame_%06d.png" in argv


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "CON.foo.mp4",
            "CON~afc0ef4e07a53c5302a14d2d1fd02201_3D_side-by-side_16x9-1080p.mp4",
        ),
        (
            "COM¹.foo.mp4",
            "COM¹~f98034cae9fd53d32245fa532dab5a16_3D_side-by-side_16x9-1080p.mp4",
        ),
        (
            "LPT³.part.mp4",
            "LPT³~7a788f8c065d09115bb8cddbe4b0b354_3D_side-by-side_16x9-1080p.mp4",
        ),
        ("COM³.mp4", "COM³_3D_side-by-side_16x9-1080p.mp4"),
        ("LPT³.mp4", "LPT³_3D_side-by-side_16x9-1080p.mp4"),
    ],
)
def test_job_output_name_goldens_cover_superscript_windows_devices(
    source: str,
    expected: str,
) -> None:
    result = encoding_provider.job_output_name_v1(
        source_video_name=source,
        vr_format="side_by_side",
        resolution="16x9-1080p",
    )

    assert result == expected
    assert encoding_provider.is_portable_final_component_v1(result)


@pytest.mark.parametrize(
    "component",
    [
        "CON.mp4",
        "com1.part.mp4",
        "LPT².foo.mp4",
        "bad?.mp4",
        "trailing. ",
        "\x00.mp4",
    ],
)
def test_portable_component_rejects_every_closed_predicate(component: str) -> None:
    assert not encoding_provider.is_portable_final_component_v1(component)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"streams":[],"programs":[],"stream_groups":[]}',
        b'{"programs":[{"id":1,"id":2}],"streams":[],"stream_groups":[]}',
        b'{"stream_groups":[],"streams":[{"codec_type":"audio","sample_rate":"48000","channels":2}]}',
    ],
)
def test_audio_ffprobe_parser_accepts_legal_ffmpeg_wrapper_shapes(raw: bytes) -> None:
    result = encoding_provider.parse_audio_ffprobe_json(raw)

    assert result is None or result.codec_type == "audio"


@pytest.mark.parametrize(
    "raw",
    [
        b'{"streams":[],"streams":[]}',
        b'{"streams":[{"codec_type":"audio","codec_type":"audio","sample_rate":"48000","channels":2}]}',
        b'{"streams":[],"unknown":[]}',
    ],
)
def test_audio_ffprobe_parser_rejects_duplicate_authority_and_unknown_roots(raw: bytes) -> None:
    with pytest.raises(encoding_provider.FfprobeJsonError):
        encoding_provider.parse_audio_ffprobe_json(raw)


def test_audio_ffprobe_parser_enforces_ignored_wrapper_state_caps() -> None:
    accepted = b'{"streams":[],"programs":[' + b",".join([b"{}"] * 256) + b"]}"
    rejected_containers = (
        b'{"streams":[],"programs":[' + b",".join([b"{}"] * 257) + b"]}"
    )
    accepted_scalar = (
        b'{"streams":[],"programs":[{"value":"' + (b"x" * 128) + b'"}]}'
    )
    rejected_scalar = (
        b'{"streams":[],"programs":[{"value":"' + (b"x" * 129) + b'"}]}'
    )

    assert encoding_provider.parse_audio_ffprobe_json(accepted) is None
    assert encoding_provider.parse_audio_ffprobe_json(accepted_scalar) is None
    with pytest.raises(encoding_provider.FfprobeJsonError, match="container"):
        encoding_provider.parse_audio_ffprobe_json(rejected_containers)
    with pytest.raises(encoding_provider.FfprobeJsonError, match="scalar"):
        encoding_provider.parse_audio_ffprobe_json(rejected_scalar)


def test_audio_ffprobe_parser_enforces_depth_without_recursive_decoding() -> None:
    depth_eight = b'{"streams":[],"programs":[{"v":[[[[[]]]]]}]}'
    depth_nine = b'{"streams":[],"programs":[{"v":[[[[[[]]]]]]}]}'

    assert encoding_provider.parse_audio_ffprobe_json(depth_eight) is None
    with pytest.raises(encoding_provider.FfprobeJsonError, match="depth"):
        encoding_provider.parse_audio_ffprobe_json(depth_nine)


def test_final_ffprobe_parser_allows_only_empty_optional_wrappers() -> None:
    raw = json.dumps(
        {
            "programs": [],
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "pix_fmt": "yuv420p",
                    "sample_aspect_ratio": "1:1",
                    "display_aspect_ratio": "16:9",
                    "avg_frame_rate": "24000/1001",
                    "nb_read_frames": "100",
                }
            ],
            "stream_groups": [],
        },
        separators=(",", ":"),
    ).encode()

    streams = encoding_provider.parse_final_ffprobe_json(raw)

    assert len(streams) == 1
    with pytest.raises(encoding_provider.FfprobeJsonError, match="empty"):
        encoding_provider.parse_final_ffprobe_json(
            raw.replace(b'"programs":[]', b'"programs":[{}]')
        )


def test_final_ffprobe_parser_enforces_scalar_token_cap() -> None:
    accepted = (
        b'{"streams":[{"codec_type":"' + (b"x" * 128) + b'"}],"programs":[]}'
    )
    rejected = (
        b'{"streams":[{"codec_type":"' + (b"x" * 129) + b'"}],"programs":[]}'
    )

    assert encoding_provider.parse_final_ffprobe_json(accepted)[0].fields["codec_type"] == (
        "x" * 128
    )
    with pytest.raises(encoding_provider.FfprobeJsonError, match="scalar"):
        encoding_provider.parse_final_ffprobe_json(rejected)


def test_final_probe_contract_validates_exact_generated_video_and_audio() -> None:
    streams = (
        encoding_provider.FinalProbeStream(
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "pix_fmt": "yuv420p",
                "sample_aspect_ratio": "1:1",
                "display_aspect_ratio": "16:9",
                "avg_frame_rate": "24000/1001",
                "nb_read_frames": "101",
            }
        ),
        encoding_provider.FinalProbeStream(
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "nb_read_frames": "47",
            }
        ),
    )
    expectation = encoding_provider.FinalContainerExpectation(
        width=1920,
        height=1080,
        fps="24000/1001",
        frame_count=101,
        resolved_encoder="libx264",
        audio_expected=True,
        audio_packet_upper_bound=48,
    )

    encoding_provider.validate_final_probe_streams(streams, expectation)


def test_final_probe_contract_accepts_ffprobe_8_audio_frame_rate_placeholder() -> None:
    streams = (
        encoding_provider.FinalProbeStream(
            {
                "codec_type": "video",
                "codec_name": "hevc",
                "width": 3840,
                "height": 1080,
                "pix_fmt": "yuv420p",
                "sample_aspect_ratio": "1:1",
                "display_aspect_ratio": "32:9",
                "avg_frame_rate": "30/1",
                "nb_read_frames": "450",
            }
        ),
        encoding_provider.FinalProbeStream(
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "avg_frame_rate": "0/0",
                "nb_read_frames": "646",
            }
        ),
    )
    expectation = encoding_provider.FinalContainerExpectation(
        width=3840,
        height=1080,
        fps="30/1",
        frame_count=450,
        resolved_encoder="hevc_nvenc",
        audio_expected=True,
        audio_packet_upper_bound=647,
    )

    encoding_provider.validate_final_probe_streams(streams, expectation)


def test_final_probe_contract_rejects_audio_frame_rate_other_than_placeholder() -> None:
    streams = (
        encoding_provider.FinalProbeStream(
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 8,
                "height": 4,
                "pix_fmt": "yuv420p",
                "sample_aspect_ratio": "1:1",
                "display_aspect_ratio": "2:1",
                "avg_frame_rate": "24/1",
                "nb_read_frames": "1",
            }
        ),
        encoding_provider.FinalProbeStream(
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "avg_frame_rate": "1/1",
                "nb_read_frames": "1",
            }
        ),
    )
    expectation = encoding_provider.FinalContainerExpectation(
        width=8,
        height=4,
        fps="24/1",
        frame_count=1,
        resolved_encoder="libx264",
        audio_expected=True,
        audio_packet_upper_bound=1,
    )

    with pytest.raises(
        encoding_provider.FinalContainerValidationError,
        match="audio frame rate",
    ):
        encoding_provider.validate_final_probe_streams(streams, expectation)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("codec_name", "hevc"),
        ("pix_fmt", "yuv444p"),
        ("sample_aspect_ratio", "2:2"),
        ("display_aspect_ratio", "1920:1080"),
        ("avg_frame_rate", "48000/2002"),
        ("nb_read_frames", "100"),
    ],
)
def test_final_probe_contract_rejects_noncanonical_video_fields(field, value) -> None:
    fields = {
        "codec_type": "video",
        "codec_name": "h264",
        "width": 1920,
        "height": 1080,
        "pix_fmt": "yuv420p",
        "sample_aspect_ratio": "1:1",
        "display_aspect_ratio": "16:9",
        "avg_frame_rate": "24000/1001",
        "nb_read_frames": "101",
    }
    fields[field] = value
    expectation = encoding_provider.FinalContainerExpectation(
        width=1920,
        height=1080,
        fps="24000/1001",
        frame_count=101,
        resolved_encoder="libx264",
        audio_expected=False,
        audio_packet_upper_bound=0,
    )

    with pytest.raises(encoding_provider.FinalContainerValidationError):
        encoding_provider.validate_final_probe_streams(
            (encoding_provider.FinalProbeStream(fields),), expectation
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_aspect_ratio", "1/1"),
        ("display_aspect_ratio", "16/9"),
        ("avg_frame_rate", "24000:1001"),
    ],
)
def test_final_probe_contract_rejects_ratio_separator_substitution(field, value) -> None:
    fields = {
        "codec_type": "video",
        "codec_name": "h264",
        "width": 1920,
        "height": 1080,
        "pix_fmt": "yuv420p",
        "sample_aspect_ratio": "1:1",
        "display_aspect_ratio": "16:9",
        "avg_frame_rate": "24000/1001",
        "nb_read_frames": "101",
    }
    fields[field] = value
    expectation = encoding_provider.FinalContainerExpectation(
        width=1920,
        height=1080,
        fps="24000/1001",
        frame_count=101,
        resolved_encoder="libx264",
        audio_expected=False,
        audio_packet_upper_bound=0,
    )

    with pytest.raises(encoding_provider.FinalContainerValidationError):
        encoding_provider.validate_final_probe_streams(
            (encoding_provider.FinalProbeStream(fields),), expectation
        )


def test_final_probe_audio_packet_bound_has_a_distinct_error() -> None:
    video = encoding_provider.FinalProbeStream(
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": 8,
            "height": 4,
            "pix_fmt": "yuv420p",
            "sample_aspect_ratio": "1:1",
            "display_aspect_ratio": "2:1",
            "avg_frame_rate": "24/1",
            "nb_read_frames": "1",
        }
    )
    audio = encoding_provider.FinalProbeStream(
        {"codec_type": "audio", "codec_name": "aac", "nb_read_frames": "4"}
    )
    expectation = encoding_provider.FinalContainerExpectation(
        width=8,
        height=4,
        fps="24/1",
        frame_count=1,
        resolved_encoder="libx264",
        audio_expected=True,
        audio_packet_upper_bound=3,
    )

    with pytest.raises(encoding_provider.FinalContainerAudioPacketBoundError):
        encoding_provider.validate_final_probe_streams((video, audio), expectation)


def test_progress_parser_resets_only_on_strict_numeric_increase() -> None:
    parser = encoding_provider.FfmpegProgressParser()

    assert parser.feed_line(b"frame=1\n")
    assert not parser.feed_line(b"frame=1\r\n")
    assert not parser.feed_line(b"fps=60\n")
    assert parser.feed_line(b"out_time_us=2\n")
    assert not parser.feed_line(b"progress=continue\n")
    assert not parser.feed_line(b"progress=end\n")
    parser.finish()

    with pytest.raises(encoding_provider.EncodingProgressError):
        parser.feed_line(b"frame=2\n")


def test_progress_parser_accepts_negative_ffmpeg_output_times() -> None:
    parser = encoding_provider.FfmpegProgressParser()

    assert parser.feed_line(b"frame=1\n")
    assert parser.feed_line(b"out_time_us=-66667\n")
    assert parser.feed_line(b"out_time_ms=-66667\n")
    assert not parser.feed_line(b"out_time_us=-66667\n")
    assert parser.feed_line(b"out_time_us=0\n")
    assert not parser.feed_line(b"progress=end\n")
    parser.finish()


def test_progress_parser_accepts_temporarily_unavailable_ffmpeg_measurements() -> None:
    parser = encoding_provider.FfmpegProgressParser()

    assert parser.feed_line(b"frame=0\n")
    assert not parser.feed_line(b"total_size=N/A\n")
    assert not parser.feed_line(b"out_time_us=N/A\n")
    assert not parser.feed_line(b"out_time_ms=N/A\n")
    assert parser.feed_line(b"out_time_us=0\n")
    assert not parser.feed_line(b"progress=end\n")
    parser.finish()


def test_final_decode_progress_accepts_temporarily_unavailable_output_time() -> None:
    parser = encoding_provider.FinalDecodeProgressParser()

    assert not parser.feed_line(b"out_time_us=N/A\n")
    assert not parser.feed_line(b"out_time_ms=N/A\n")
    assert parser.feed_line(b"frame=1\n")
    assert not parser.feed_line(b"progress=end\n")
    parser.finish()


@pytest.mark.parametrize(
    "line",
    [
        b"frame=-1\n",
        b"total_size=-1\n",
        b"out_time_us=-0\n",
        b"out_time_ms=+1\n",
        b"out_time_us=-9223372036854775809\n",
        b"out_time_ms=9223372036854775808\n",
    ],
)
def test_progress_parser_rejects_noncanonical_or_out_of_range_counters(
    line: bytes,
) -> None:
    with pytest.raises(encoding_provider.EncodingProgressError):
        encoding_provider.FfmpegProgressParser().feed_line(line)


def test_final_decode_progress_ignores_total_size_for_stall_progress() -> None:
    parser = encoding_provider.FinalDecodeProgressParser()

    assert not parser.feed_line(b"total_size=1\n")
    assert parser.feed_line(b"frame=1\n")
    assert not parser.feed_line(b"total_size=2\n")
    assert not parser.feed_line(b"progress=end\n")
    parser.finish()


def test_final_decode_progress_rejects_values_over_256_bytes() -> None:
    parser = encoding_provider.FinalDecodeProgressParser()

    with pytest.raises(encoding_provider.EncodingProgressError, match="value"):
        parser.feed_line(b"unknown=" + b"x" * 257 + b"\n")


def test_all_encoding_coordinator_phase_formulas_fit_the_eight_mib_budget() -> None:
    peaks = encoding_provider.encoding_control_phase_peaks()

    assert len(peaks) == 15
    assert max(peaks.values()) == 6 * 1024 * 1024
    assert max(peaks.values()) <= encoding_provider.FINAL_ENCODING_CONTROL_BUDGET
