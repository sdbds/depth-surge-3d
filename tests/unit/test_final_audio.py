from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.depth_surge_3d.processing.video import final_audio
from src.depth_surge_3d.processing.video.encoding_provider import AudioProbeStream


def _settings(**overrides):
    values = {
        "preserve_audio": True,
        "start_time": None,
        "end_time": None,
    }
    values.update(overrides)
    return values


def test_disabled_audio_does_not_touch_any_source(tmp_path: Path) -> None:
    with patch.object(final_audio, "_hash_open_regular_file") as hash_file:
        selected = final_audio.prepare_selected_audio_v1(
            output_dir=tmp_path,
            original_video=tmp_path / "missing.mp4",
            settings=_settings(preserve_audio=False),
            frame_count=12,
            fps="24000/1001",
        )

    assert selected.selection == "disabled"
    assert selected.disposition == "disabled"
    assert selected.input_arguments() == ()
    assert selected.output_arguments(1) == ()
    hash_file.assert_not_called()


def test_audio_selects_regular_flac_and_freezes_sample_trim(tmp_path: Path) -> None:
    original = tmp_path / "source.mp4"
    original.write_bytes(b"video-audio")
    flac = tmp_path / "original_audio.flac"
    flac.write_bytes(b"flac-audio")

    with (
        patch.object(
            final_audio,
            "_probe_first_audio_stream",
            return_value=AudioProbeStream("audio", 48_000, 2),
        ) as probe,
        patch.object(final_audio, "_decode_pcm_byte_count", return_value=48_000 * 2 * 4),
    ):
        selected = final_audio.prepare_selected_audio_v1(
            output_dir=tmp_path,
            original_video=original,
            settings=_settings(start_time="00:00:01", end_time="00:00:04"),
            frame_count=24,
            fps="24000/1001",
        )

    assert selected.selection == "original_audio_flac"
    assert selected.selected_source_path == flac.resolve()
    assert selected.resolved_clip_arguments == ("-ss", "1", "-t", "3")
    assert selected.disposition == "usable"
    assert selected.decoded_pcm_bytes == 48_000 * 2 * 4
    assert selected.audio_end_sample == 48_048
    assert selected.input_arguments() == (
        "-ss",
        "1",
        "-t",
        "3",
        "-i",
        str(flac.resolve()),
    )
    assert selected.output_arguments(2) == (
        "-map",
        "2:a:0",
        "-filter:a",
        "atrim=end_sample=48048,asetpts=PTS-STARTPTS",
        "-c:a",
        "aac",
    )
    assert "-shortest" not in selected.output_arguments(2)
    probe.assert_called_once_with(flac.resolve(), selected.resolved_clip_arguments)


def test_audio_symlink_flac_falls_back_to_original_video(tmp_path: Path) -> None:
    original = tmp_path / "source.mp4"
    original.write_bytes(b"source")
    external = tmp_path / "external.flac"
    external.write_bytes(b"audio")
    flac = tmp_path / "original_audio.flac"
    try:
        flac.symlink_to(external)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    with patch.object(final_audio, "_probe_first_audio_stream", return_value=None):
        selected = final_audio.prepare_selected_audio_v1(
            output_dir=tmp_path,
            original_video=original,
            settings=_settings(),
            frame_count=1,
            fps="24",
        )

    assert selected.selection == "original_video"
    assert selected.selected_source_path == original.resolve()
    assert selected.disposition == "no_a0"
    assert selected.input_arguments() == ()


def test_selected_original_video_symlink_is_fatal(tmp_path: Path) -> None:
    target = tmp_path / "actual.mp4"
    target.write_bytes(b"source")
    source = tmp_path / "source.mp4"
    try:
        source.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(final_audio.FinalAudioInputChangedError, match="non-link"):
        final_audio.prepare_selected_audio_v1(
            output_dir=tmp_path,
            original_video=source,
            settings=_settings(),
            frame_count=1,
            fps="24",
        )


def test_zero_decoded_samples_omits_audio_from_encoder(tmp_path: Path) -> None:
    original = tmp_path / "source.mp4"
    original.write_bytes(b"source")
    with (
        patch.object(
            final_audio,
            "_probe_first_audio_stream",
            return_value=AudioProbeStream("audio", 44_100, 1),
        ),
        patch.object(final_audio, "_decode_pcm_byte_count", return_value=0),
    ):
        selected = final_audio.prepare_selected_audio_v1(
            output_dir=tmp_path,
            original_video=original,
            settings=_settings(),
            frame_count=10,
            fps="30",
        )

    assert selected.disposition == "zero_decoded_samples"
    assert selected.sample_rate == 44_100
    assert selected.decoded_pcm_bytes == 0
    assert selected.audio_end_sample is None
    assert selected.input_arguments() == ()


def test_pcm_count_must_end_on_a_complete_sample_frame(tmp_path: Path) -> None:
    original = tmp_path / "source.mp4"
    original.write_bytes(b"source")
    with (
        patch.object(
            final_audio,
            "_probe_first_audio_stream",
            return_value=AudioProbeStream("audio", 48_000, 2),
        ),
        patch.object(final_audio, "_decode_pcm_byte_count", return_value=7),
    ):
        with pytest.raises(final_audio.FinalAudioDecodeError, match="PCM frame"):
            final_audio.prepare_selected_audio_v1(
                output_dir=tmp_path,
                original_video=original,
                settings=_settings(),
                frame_count=1,
                fps="24",
            )


def test_postpass_rejects_changed_selected_source(tmp_path: Path) -> None:
    original = tmp_path / "source.mp4"
    original.write_bytes(b"source")
    with patch.object(final_audio, "_probe_first_audio_stream", return_value=None):
        selected = final_audio.prepare_selected_audio_v1(
            output_dir=tmp_path,
            original_video=original,
            settings=_settings(),
            frame_count=1,
            fps="24",
        )

    original.write_bytes(b"changed")
    with pytest.raises(final_audio.FinalAudioInputChangedError):
        selected.revalidate()


def test_audio_probe_and_decode_commands_are_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    clip = ("-ss", "1.25", "-t", "2")

    assert final_audio.audio_probe_command_v1(source) == (
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type,sample_rate,channels",
        "-of",
        "json",
        str(source),
    )
    assert final_audio.audio_decode_command_v1(source, clip) == (
        "ffmpeg",
        "-v",
        "error",
        "-xerror",
        "-ss",
        "1.25",
        "-t",
        "2",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-c:a",
        "pcm_s32le",
        "-f",
        "s32le",
        "pipe:1",
    )
