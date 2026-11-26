"""Kill an encoder process at durable publication boundaries, then restart it."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.io.settings_artifact import ProcessingSettingsArtifactV5
from src.depth_surge_3d.processing.video import video_encoder
from src.depth_surge_3d.processing.video.final_publication import load_final_encoding_reservation_v1
from .test_output_timeline import _job, _probe


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(sys.platform != "linux", reason="Linux process-death contract"),
    pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg"),
]


def _prepare(root, mode):
    source, job, count, _, control = _job(root, rate="4", target=None, mode=mode, audio=False)
    folders = ["99_vr_frames"] if mode == "assembled" else ["06_left_cropped", "06_right_cropped"]
    for eye, folder in enumerate(folders):
        directory = job / folder
        directory.mkdir()
        for index in range(1, count + 1):
            width = 128 if mode == "assembled" else 64
            pixels = np.full((48, width, 3), index * 35 + eye * 24, np.uint8)
            assert cv2.imwrite(str(directory / f"frame_{index:06d}.png"), pixels)
    target = job / json.loads(control.read_bytes())["output_info"]["expected_final_relative_path"]
    return source, job, target, count


def _environment(root):
    # Use the exact package under test (site-packages in the wheel CI).
    env = {**os.environ, "PYTHONPATH": str(Path(video_encoder.__file__).resolve().parents[3])}
    tools = root / "bin"
    tools.mkdir(exist_ok=True)
    recorder = tools / "ffmpeg"
    if not recorder.exists():
        recorder.write_text(
            "#!/usr/bin/env python3\nimport json, os, sys\n"
            "with open(os.environ['FFMPEG_TRACE'], 'a') as f: print(json.dumps(sys.argv), file=f)\n"
            "os.execv(os.environ['REAL_FFMPEG'], ['ffmpeg', *sys.argv[1:]])\n"
        )
        recorder.chmod(0o755)
    env.update(
        PATH=f"{tools}:{env['PATH']}",
        REAL_FFMPEG=shutil.which("ffmpeg"),
        FFMPEG_TRACE=str(root / "ffmpeg.jsonl"),
        HF_HUB_OFFLINE="1",
    )
    return env


def _run(job, source, count, mode, boundary=""):
    worker = Path(__file__).with_name("publication_restart_worker.py")
    return subprocess.run(
        [sys.executable, str(worker), str(job), str(source), str(count), mode, boundary],
        cwd=job.parent,
        env=_environment(job.parent),
        capture_output=True,
        text=True,
        timeout=90,
    )


def _encoder_calls(job):
    commands = [json.loads(line) for line in (job.parent / "ffmpeg.jsonl").read_text().splitlines()]
    return [cmd for cmd in commands if "-c:v" in cmd and cmd[-1].endswith(".mp4")]


@pytest.mark.parametrize("mode", ["assembled", "direct", "streamed"])
@pytest.mark.parametrize(
    "boundary",
    [
        "encoded-output:synced",
        "video:published",
        "input-manifest:published",
        "final-manifest:published",
    ],
)
def test_process_death_reuses_encoded_inode_without_another_ffmpeg_encode(tmp_path, mode, boundary):
    source, job, target, count = _prepare(tmp_path, mode)
    interrupted = _run(job, source, count, mode, boundary)
    assert interrupted.returncode == 86, interrupted.stdout + interrupted.stderr
    assert len(_encoder_calls(job)) == 1
    if boundary == "encoded-output:synced" and mode == "streamed":
        (candidate,) = job.glob(".depth-surge-final-v4-*.tmp.mp4")
    else:
        candidate = (
            Path(_encoder_calls(job)[0][-1]) if boundary == "encoded-output:synced" else target
        )
    original_inode = candidate.stat().st_ino
    digest = hashlib.sha256(candidate.read_bytes()).digest()

    # Publishing the bytes already on disk must not require the input frames again.
    for png in job.glob("*/*.png"):
        png.unlink()
    for _ in range(2):
        restart = _run(job, source, count, mode)
        assert restart.returncode == 0, restart.stdout + restart.stderr
        assert len(_encoder_calls(job)) == 1
        assert target.stat().st_ino == original_inode
        assert hashlib.sha256(target.read_bytes()).digest() == digest
    assert int(_probe(target)["video"]["nb_read_frames"]) == count


@pytest.mark.parametrize(
    "boundary", ["encoded-output:synced", "video:published", "input-manifest:published"]
)
def test_resume_preserves_an_existing_movie_even_if_its_bytes_match(tmp_path, boundary):
    source, job, target, count = _prepare(tmp_path, "assembled")
    interrupted = _run(job, source, count, "assembled", boundary)
    assert interrupted.returncode == 86, interrupted.stdout + interrupted.stderr
    candidate = Path(_encoder_calls(job)[0][-1]) if boundary == "encoded-output:synced" else target
    original = candidate.read_bytes()
    replacement = tmp_path / "existing.mp4"
    replacement.write_bytes(original)
    inode = replacement.stat().st_ino
    replacement.replace(target)
    for _ in range(2):
        restart = _run(job, source, count, "assembled")
        assert restart.returncode == 1, restart.stdout + restart.stderr
        assert target.stat().st_ino == inode and target.read_bytes() == original
        assert len(_encoder_calls(job)) == 1


@pytest.mark.parametrize("mode", ["assembled", "streamed"])
def test_installed_cli_resumes_publication_without_models_or_frames(tmp_path, mode):
    source, job, target, count = _prepare(tmp_path, mode)
    interrupted = _run(job, source, count, mode, "video:published")
    assert interrupted.returncode == 86, interrupted.stdout + interrupted.stderr
    original = target.read_bytes()
    inode = target.stat().st_ino
    for png in job.glob("*/*.png"):
        png.unlink()
    cli = Path(sys.executable).parent / "depth-surge-3d"
    assert cli.is_file(), "install the project CLI before running the integration suite"

    def resume():
        return subprocess.run(
            [str(cli), "--resume", str(job)],
            cwd=tmp_path,
            env=_environment(tmp_path),
            capture_output=True,
            text=True,
            timeout=90,
        )

    for _ in range(2):
        resumed = resume()
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert (
        json.loads((job / "processing-settings-v5.json").read_bytes())["metadata"][
            "processing_status"
        ]
        == "completed"
    )
    assert len(_encoder_calls(job)) == 1
    assert target.stat().st_ino == inode and target.read_bytes() == original
    other = tmp_path / "other.mp4"
    other.write_bytes(original)
    other_inode = other.stat().st_ino
    other.replace(target)
    assert resume().returncode == 1
    assert target.stat().st_ino == other_inode and target.read_bytes() == original
    assert len(_encoder_calls(job)) == 1


def test_pre_upgrade_file_backed_job_preserves_video_when_there_is_no_durable_proof(tmp_path):
    source, job, output, count = _prepare(tmp_path, "assembled")
    interrupted = _run(job, source, count, "assembled", "video:published")
    assert interrupted.returncode == 86, interrupted.stdout + interrupted.stderr
    saved = ProcessingSettingsArtifactV5.parse((job / "processing-settings-v5.json").read_bytes())
    prepared = load_final_encoding_reservation_v1(job, saved)
    # Earlier file-backed publishers left this reserved extent zero until after both renames.
    with prepared.final_manifest_transaction.open_authenticated_source(writable=True) as handle:
        handle.write(bytes(os.fstat(handle.fileno()).st_size))
        os.fsync(handle.fileno())
    original = output.read_bytes()
    inode = output.stat().st_ino
    restart = _run(job, source, count, "assembled")
    assert restart.returncode == 1, restart.stdout + restart.stderr
    assert output.stat().st_ino == inode and output.read_bytes() == original
    cli = Path(sys.executable).parent / "depth-surge-3d"
    result = subprocess.run(
        [str(cli), "--resume", str(job)],
        cwd=tmp_path,
        env=_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 1
    assert "without re-encoding" in result.stdout, result.stdout + result.stderr
    assert len(_encoder_calls(job)) == 1


def test_torn_write_requires_complete_evidence_before_moving_the_encoded_video(tmp_path):
    source, job, output, count = _prepare(tmp_path, "assembled")
    assert _run(job, source, count, "assembled", "encoded-output:synced").returncode == 86
    candidate = Path(_encoder_calls(job)[0][-1])
    digest = hashlib.sha256(candidate.read_bytes()).digest()
    inode = candidate.stat().st_ino
    saved = ProcessingSettingsArtifactV5.parse((job / "processing-settings-v5.json").read_bytes())
    prepared = load_final_encoding_reservation_v1(job, saved)
    extent = prepared.final_manifest_transaction
    with extent.open_authenticated_source(writable=True) as handle:
        whole = handle.read()
        handle.seek(64)
        handle.write(bytes(len(whole) - 64))
        os.fsync(handle.fileno())
    failed = _run(job, source, count, "assembled")
    assert failed.returncode == 1
    assert "incomplete encoded-output evidence" in failed.stdout + failed.stderr
    assert output.stat().st_size == 0 and candidate.stat().st_ino == inode
    assert hashlib.sha256(candidate.read_bytes()).digest() == digest
    with extent.open_authenticated_source(writable=True) as handle:
        handle.write(whole)
        os.fsync(handle.fileno())
    retry = _run(job, source, count, "assembled")
    assert retry.returncode == 0, retry.stdout + retry.stderr
    assert output.stat().st_ino == inode and hashlib.sha256(output.read_bytes()).digest() == digest
    assert len(_encoder_calls(job)) == 1
