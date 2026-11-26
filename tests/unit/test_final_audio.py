from __future__ import annotations

import os
from pathlib import Path
import sys
import wave
from unittest.mock import patch

import pytest

from src.depth_surge_3d.processing.video import final_audio
from src.depth_surge_3d.processing.video.bounded_process import BoundedProcessResult
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
        patch.object(final_audio, "_decode_pcm_byte_count", return_value=144_144 * 2 * 4),
    ):
        selected = final_audio.prepare_selected_audio_v1(
            output_dir=tmp_path,
            original_video=original,
            settings=_settings(start_time="00:00:01", end_time="00:00:04"),
            frame_count=72,
            fps="24000/1001",
        )

    assert selected.selection == "original_audio_flac"
    assert selected.selected_source_path == flac.resolve()
    assert selected.resolved_clip_arguments == ("-ss", "0.959", "-t", "3.003")
    assert selected.disposition == "usable"
    assert selected.decoded_pcm_bytes == 144_144 * 2 * 4
    assert selected.audio_end_sample == 144_144
    assert selected.input_arguments() == (
        "-ss",
        "0.959",
        "-t",
        "3.003",
        "-i",
        str(flac.resolve()),
    )
    assert selected.output_arguments(2) == (
        "-map",
        "2:a:0",
        "-filter:a",
        "atrim=end_sample=144144,asetpts=PTS-STARTPTS",
        "-c:a",
        "aac",
    )
    assert "-shortest" not in selected.output_arguments(2)
    probe.assert_called_once_with(flac.resolve(), selected.resolved_clip_arguments)


def test_audio_clip_arguments_align_milliseconds_to_selected_video_frames(
    tmp_path: Path,
) -> None:
    original = tmp_path / "source.mp4"
    original.write_bytes(b"video-audio")

    with patch.object(final_audio, "_probe_first_audio_stream", return_value=None):
        selected = final_audio.prepare_selected_audio_v1(
            output_dir=tmp_path,
            original_video=original,
            settings=_settings(start_time="00:00.125", end_time="00:01.375"),
            frame_count=38,
            fps="30",
        )

    assert selected.resolved_clip_arguments == ("-ss", "0.1", "-t", "1.267")


def test_audio_clip_arguments_align_milliseconds_after_one_hour(
    tmp_path: Path,
) -> None:
    original = tmp_path / "source.mp4"
    original.write_bytes(b"video-audio")

    with patch.object(final_audio, "_probe_first_audio_stream", return_value=None):
        selected = final_audio.prepare_selected_audio_v1(
            output_dir=tmp_path,
            original_video=original,
            settings=_settings(
                start_time="01:02:03.004",
                end_time="01:02:04.129",
            ),
            frame_count=33,
            fps="30",
        )

    assert selected.resolved_clip_arguments == (
        "-ss",
        "3723",
        "-t",
        "1.1",
    )


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


@pytest.mark.skipif(os.name != "nt", reason="Windows FileIdInfo contract")
def test_windows_audio_identity_comes_from_the_open_file_handle(tmp_path: Path) -> None:
    import msvcrt

    from src.depth_surge_3d.io.file_identity import windows_identity_from_handle

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    hashed = final_audio._hash_open_regular_file(source)
    with source.open("rb", buffering=0) as handle:
        identity, _links, _attributes = windows_identity_from_handle(
            int(msvcrt.get_osfhandle(handle.fileno())),
            require_directory=False,
        )

    assert hashed.identity == final_audio.OpenFileIdentityV1(
        "windows",
        identity.volume_serial,
        identity.file_id,
    )


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


def test_audio_probe_parser_state_overflow_maps_to_budget_error(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    overlong = b'{"streams":[],"programs":[{"value":"' + b"x" * 129 + b'"}]}'
    with patch.object(final_audio, "BoundedProcessCaptureV1") as capture:
        capture.return_value.run.return_value = BoundedProcessResult(overlong, b"", 0)

        with pytest.raises(final_audio.FinalAudioProbeBudgetError, match="parser-state"):
            final_audio._probe_first_audio_stream(source, ())


def _child(script):
    return (sys.executable, "-c", script)


@pytest.mark.parametrize(
    ("stage", "script", "error"),
    [
        ("probe", "print('not json')", final_audio.FinalAudioProbeError),
        ("probe", "import sys;sys.exit(2)", final_audio.FinalAudioProbeError),
        ("probe", "print('x'*300)", final_audio.FinalAudioProbeBudgetError),
        ("probe", "import time;time.sleep(15)", final_audio.FinalAudioProbeTimeoutError),
        ("decode", "import sys;sys.stderr.write('bad packet')", final_audio.FinalAudioDecodeError),
        ("decode", "import sys;sys.stderr.write('x'*300)", final_audio.FinalAudioDecodeBudgetError),
        ("decode", "import time;time.sleep(15)", final_audio.FinalAudioDecodeTimeoutError),
    ],
)
def test_external_audio_failures_are_classified_before_a_source_can_be_selected(
    monkeypatch, tmp_path, stage, script, error
):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source bytes")
    probe = """print('{"streams":[{"codec_type":"audio","sample_rate":"48000","channels":2}]}')"""
    monkeypatch.setattr(
        final_audio,
        "audio_probe_command_v1",
        lambda path: _child(script if stage == "probe" else probe),
    )
    monkeypatch.setattr(final_audio, "audio_decode_command_v1", lambda path, clip: _child(script))
    monkeypatch.setattr(final_audio, "FINAL_AUDIO_PROBE_STDOUT_BYTES", 128)
    monkeypatch.setattr(final_audio, "FINAL_AUDIO_DECODE_DIAGNOSTIC_BYTES", 128)
    if "sleep" in script:
        timeout = (
            "FINAL_AUDIO_PROBE_TIMEOUT_SECONDS"
            if stage == "probe"
            else "FINAL_AUDIO_DECODE_STALL_SECONDS"
        )
        monkeypatch.setattr(final_audio, timeout, 0.2)
    with pytest.raises(error) as caught:
        final_audio.prepare_selected_audio_v1(
            output_dir=tmp_path,
            original_video=source,
            settings=_settings(),
            frame_count=1,
            fps="24",
        )
    assert type(caught.value) is error
    # All input handles have been released even on timeout or a broken external utility.
    assert source.rename(tmp_path / "released.mp4").read_bytes() == b"source bytes"


def test_cancel_during_chunked_audio_hash_closes_source_before_reuse(monkeypatch, tmp_path):
    source = tmp_path / "selected.flac"
    source.write_bytes(b"aabbccddeeff")
    monkeypatch.setattr(final_audio, "FINAL_AUDIO_HASH_STREAM_BYTES", 2)
    states = iter((False, False, True))
    with pytest.raises(InterruptedError, match="hashing"):
        final_audio._hash_open_regular_file(source, cancelled=lambda: next(states))
    new = source.rename(tmp_path / "usable.flac")
    assert final_audio._hash_open_regular_file(new).byte_count == 12


@pytest.mark.parametrize(
    ("frame_count", "fps", "settings", "message"),
    [
        (0, "24", {}, "frame count"),
        (1, "x", {}, "frame rate"),
        (1, "-1", {}, "frame rate"),
        (1, "24", {"source_fps": "x"}, "source frame rate"),
        (1, "24", {"source_fps": 0}, "source frame rate"),
        (2**64 - 1, "1", {}, "sample numerator"),
        (2, str(2**64 - 1), {}, "end-sample numerator"),
    ],
)
def test_invalid_or_unrepresentable_audio_clock_is_not_rounded_into_a_valid_trim(
    monkeypatch, tmp_path, frame_count, fps, settings, message
):
    source = tmp_path / "selected.wav"
    source.write_bytes(b"audio")
    monkeypatch.setattr(
        final_audio, "_probe_first_audio_stream", lambda *_: AudioProbeStream("audio", 48_000, 2)
    )
    monkeypatch.setattr(final_audio, "_decode_pcm_byte_count", lambda *_: 8)
    with pytest.raises(final_audio.FinalAudioPcmOverflowError, match=message):
        final_audio.prepare_selected_audio_v1(
            output_dir=tmp_path,
            original_video=source,
            settings=_settings(**settings),
            frame_count=frame_count,
            fps=fps,
        )


def test_first_pcm_stream_counts_complete_stereo_frames_through_actual_pipes(tmp_path, monkeypatch):
    source = tmp_path / "stereo.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setparams((2, 4, 48_000, 0, "NONE", "not compressed"))
        audio.writeframes(bytes(range(8)) * 4_100)
    probe = (
        "import wave,sys,json; w=wave.open(sys.argv[1]);"
        "print(json.dumps({'streams':[{'codec_type':'audio','sample_rate':str(w.getframerate()),"
        "'channels':w.getnchannels()}]}))"
    )
    decode = "import wave,sys; w=wave.open(sys.argv[1]); sys.stdout.buffer.write(w.readframes(w.getnframes()))"
    monkeypatch.setattr(
        final_audio, "audio_probe_command_v1", lambda path: (*_child(probe), str(path))
    )
    monkeypatch.setattr(
        final_audio, "audio_decode_command_v1", lambda path, clip: (*_child(decode), str(path))
    )
    monkeypatch.setattr(final_audio, "FINAL_AUDIO_STREAM_BYTES", 29)
    selected = final_audio.prepare_selected_audio_v1(
        output_dir=tmp_path,
        original_video=source,
        settings=_settings(),
        frame_count=2,
        fps="24000/1001",
        clip_start_frame=0,
        cancelled=lambda: False,
    )
    assert selected.usable and selected.selection == "original_video"
    assert selected.input_arguments() == ("-i", str(source))
    assert selected.decoded_pcm_bytes == 4100 * 2 * 4
    assert selected.audio_end_sample == 4004
    assert "atrim=end_sample=4004,asetpts=PTS-STARTPTS" in selected.output_arguments(3)
    with pytest.raises(ValueError, match="index"):
        selected.output_arguments(True)
    selected.revalidate()
