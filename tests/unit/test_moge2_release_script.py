from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.verify_moge2_release import (
    CANONICAL_CLIP_IDS,
    RELEASE_VARIANT_PINS,
    AtomicPublisher,
    ClipRender,
    FixedDepth,
    ProductionVariantSession,
    RawClip,
    ReleaseDependencies,
    ReleaseRunFailed,
    ReleaseRunner,
    compute_clip_measurements,
    create_argument_parser,
    load_corpus_config,
    map_source_roi,
    release_variants,
    render_markdown_report,
    validate_fixed_image_artifact,
    write_fixed_image_artifact,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inputs(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    fixed = tmp_path / "fixed.png"
    clips = [tmp_path / f"{clip_id}.mp4" for clip_id in CANONICAL_CLIP_IDS]
    fixed.write_bytes(b"fixed")
    for index, path in enumerate(clips):
        path.write_bytes(f"clip-{index}".encode())
    payload = {
        "fixed_image": {"path": fixed.name, "sha256": _sha(fixed)},
        "clips": [
            {
                "id": clip_id,
                "path": path.name,
                "sha256": _sha(path),
                "static_roi_xywh": [1, 1, 2, 2],
            }
            for clip_id, path in zip(CANONICAL_CLIP_IDS, clips)
        ],
    }
    config = tmp_path / "corpus.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    return config, payload


def _image_probe(_path: Path) -> tuple[int, int]:
    return 6, 4


def _video_probe(_path: Path) -> dict[str, Any]:
    return {
        "width": 6,
        "height": 4,
        "fps": 24.0,
        "frame_count": 2,
        "sample_aspect_ratio": "1:1",
    }


def _output_media_probe(_path: Path) -> dict[str, Any]:
    return {
        "width": 12,
        "height": 4,
        "fps": 24.0,
        "frame_count": 2,
        "duration": 2 / 24.0,
    }


def _load(config: Path):
    return load_corpus_config(config, image_probe=_image_probe, video_probe=_video_probe)


def test_parser_has_only_release_controls_and_positive_depth_resolution() -> None:
    parser = create_argument_parser()
    assert {action.dest for action in parser._actions} == {
        "help",
        "corpus_config",
        "output_dir",
        "device",
        "depth_resolution",
    }
    parsed = parser.parse_args(["--corpus-config", "c.json", "--output-dir", "out"])
    assert parsed.device == "cuda"
    assert parsed.depth_resolution == 1080
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--corpus-config", "c.json", "--output-dir", "out", "--depth-resolution", "0"]
        )
    for escape_hatch in (
        "--variant",
        "--model-size",
        "--moge-resolution-level",
        "--source-revision",
        "--skip-video",
        "--threshold",
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["--corpus-config", "c.json", "--output-dir", "out", escape_hatch, "x"]
            )


def test_static_help_does_not_import_optional_moge_or_create_output(tmp_path: Path) -> None:
    script = Path(__file__).parents[2] / "scripts" / "verify_moge2_release.py"
    blocker = tmp_path / "moge.py"
    blocker.write_text("raise RuntimeError('moge imported by help')\n", encoding="utf-8")
    output = tmp_path / "never-created"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--help",
            "--output-dir",
            str(output),
        ],
        cwd=script.parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--corpus-config" in completed.stdout
    assert not output.exists()


def test_load_config_resolves_paths_hashes_and_returns_immutable_records(tmp_path: Path) -> None:
    config, _payload = _write_inputs(tmp_path)
    loaded = _load(config)
    assert loaded.fixed_image.path == (tmp_path / "fixed.png").resolve()
    assert [clip.clip_id for clip in loaded.clips] == list(CANONICAL_CLIP_IDS)
    assert tuple(loaded.clips[0].static_roi_xywh) == (1, 1, 2, 2)
    with pytest.raises((AttributeError, TypeError)):
        loaded.clips[0].clip_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda p: p.update(extra=True), "top-level keys"),
        (lambda p: p.pop("clips"), "top-level keys"),
        (lambda p: p["fixed_image"].update(extra=True), "fixed_image keys"),
        (lambda p: p["clips"][0].pop("static_roi_xywh"), "clip keys"),
        (lambda p: p["clips"].pop(), "exactly three clips"),
        (lambda p: p["clips"].append(copy.deepcopy(p["clips"][0])), "exactly three clips"),
        (lambda p: p["clips"].reverse(), "canonical order"),
        (lambda p: p["clips"][1].update(id="indoor-near"), "canonical order"),
        (lambda p: p["clips"][0].update(id="Indoor-Near"), "canonical order"),
        (lambda p: p["fixed_image"].update(sha256="A" * 64), "lowercase SHA-256"),
        (lambda p: p["clips"][0].update(sha256="a" * 63), "lowercase SHA-256"),
        (lambda p: p["clips"][0].update(static_roi_xywh=[True, 1, 2, 2]), "integers"),
        (lambda p: p["clips"][0].update(static_roi_xywh=[-1, 1, 2, 2]), "nonnegative"),
        (lambda p: p["clips"][0].update(static_roi_xywh=[1, 1, 0, 2]), "positive"),
        (lambda p: p["clips"][0].update(static_roi_xywh=[5, 1, 2, 2]), "bounds"),
    ],
)
def test_load_config_rejects_exact_schema_id_hash_and_roi_errors(
    tmp_path: Path, mutate, match: str
) -> None:
    config, payload = _write_inputs(tmp_path)
    mutate(payload)
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        _load(config)


@pytest.mark.parametrize("text", ["[1, 2]", "not-json"])
def test_load_config_rejects_malformed_json_and_non_object_roots(tmp_path: Path, text: str) -> None:
    config = tmp_path / "corpus.json"
    config.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="corpus JSON"):
        _load(config)


def test_load_config_rejects_missing_and_tampered_regular_files(tmp_path: Path) -> None:
    config, payload = _write_inputs(tmp_path)
    (tmp_path / "fixed.png").unlink()
    with pytest.raises(ValueError, match="regular file"):
        _load(config)
    (tmp_path / "fixed.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _load(config)
    payload["fixed_image"]["path"] = "."
    payload["fixed_image"]["sha256"] = SHA_A
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="regular file"):
        _load(config)


@pytest.mark.parametrize(
    "sar, match",
    [
        (None, None),
        ("N/A", None),
        ("1:1", None),
        ("2:2", None),
        ("0:1", "components"),
        ("1:0", "components"),
        ("2147483648:2147483648", "components"),
        ("1.0:1", "unsigned"),
        ("4:3", "square-pixel"),
    ],
)
def test_load_config_normalizes_and_rejects_sar(
    tmp_path: Path, sar: object, match: str | None
) -> None:
    config, _payload = _write_inputs(tmp_path)

    def probe(_path: Path) -> dict[str, Any]:
        result = _video_probe(_path)
        result["sample_aspect_ratio"] = sar
        return result

    if match is None:
        loaded = load_corpus_config(config, image_probe=_image_probe, video_probe=probe)
        assert all((clip.sar_numerator, clip.sar_denominator) == (1, 1) for clip in loaded.clips)
    else:
        with pytest.raises(ValueError, match=match):
            load_corpus_config(config, image_probe=_image_probe, video_probe=probe)


def test_load_config_rejects_explicit_non_square_numeric_sar_without_text(
    tmp_path: Path,
) -> None:
    config, _payload = _write_inputs(tmp_path)

    def probe(_path: Path) -> dict[str, Any]:
        result = _video_probe(_path)
        result.pop("sample_aspect_ratio")
        result["sample_aspect_ratio_numerator"] = 4
        result["sample_aspect_ratio_denominator"] = 3
        return result

    with pytest.raises(ValueError, match="square-pixel"):
        load_corpus_config(config, image_probe=_image_probe, video_probe=probe)


def test_release_variants_are_registry_pins_in_required_order() -> None:
    assert (
        release_variants()
        == RELEASE_VARIANT_PINS
        == (
            (
                "vits",
                "Ruicheng/moge-2-vits-normal",
                "679230677b4d282c6f304189a93e98e14f085902",
            ),
            (
                "vitb",
                "Ruicheng/moge-2-vitb-normal",
                "54ad3a693e61907ea4633d13dec6ee682fa09419",
            ),
            (
                "vitl",
                "Ruicheng/moge-2-vitl",
                "39c4d5e957afe587e04eec59dc2bcc3be5ecd968",
            ),
        )
    )


def test_release_variants_reject_registry_drift() -> None:
    class Variant:
        def __init__(self, setting: str) -> None:
            self.setting = setting
            self.repo_id = "unexpected/repo"
            self.revision = "0" * 40

    class Spec:
        variants = {name: Variant(name) for name in ("vits", "vitb", "vitl")}

    with pytest.raises(RuntimeError, match="registry drift"):
        release_variants(lambda _backend: Spec())


def test_fixed_image_artifact_exact_npz_contract_and_invalid_rejection(tmp_path: Path) -> None:
    path = tmp_path / "fixed.npz"
    depth = np.array([[1.0, np.inf], [2.0, -1.0]], dtype=np.float32)
    valid = np.array([[True, False], [True, False]], dtype=np.bool_)
    write_fixed_image_artifact(path, FixedDepth(depth, valid, np.float32(1.25)))
    assert validate_fixed_image_artifact(path) == {
        "native_shape": [2, 2],
        "focal_x_normalized": 1.25,
        "valid_metric_pixels": 2,
    }
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == [
            "depth.npy",
            "valid.npy",
            "focal_x_normalized.npy",
        ]
    with np.load(path, allow_pickle=False) as payload:
        assert payload["depth"].dtype == np.float32
        assert payload["valid"].dtype == np.bool_
        assert payload["focal_x_normalized"].shape == ()
        assert payload["depth"].tolist() == [[1.0, 0.0], [2.0, 0.0]]

    with pytest.raises(ValueError, match="valid metric"):
        write_fixed_image_artifact(
            tmp_path / "bad.npz",
            FixedDepth(np.zeros((2, 2), np.float32), np.zeros((2, 2), bool), np.float32(1)),
        )
    with pytest.raises(ValueError, match="focal"):
        write_fixed_image_artifact(
            tmp_path / "bad-focal.npz",
            FixedDepth(np.ones((2, 2), np.float32), np.ones((2, 2), bool), np.float32(0)),
        )


def test_fixed_image_validator_rejects_extra_member_and_object_array(tmp_path: Path) -> None:
    extra = tmp_path / "extra.npz"
    np.savez(
        extra,
        depth=np.ones((1, 1), np.float32),
        valid=np.ones((1, 1), bool),
        focal_x_normalized=np.asarray(1, np.float32),
        extra=np.ones(1),
    )
    with pytest.raises(ValueError, match="exact members"):
        validate_fixed_image_artifact(extra)
    objects = tmp_path / "objects.npz"
    np.savez(
        objects,
        depth=np.asarray([[object()]], dtype=object),
        valid=np.ones((1, 1), bool),
        focal_x_normalized=np.asarray(1, np.float32),
    )
    with pytest.raises(ValueError, match="dtype"):
        validate_fixed_image_artifact(objects)


def _metric_fixture(tmp_path: Path) -> tuple[RawClip, ClipRender, ClipRender]:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "storage_status": "ready",
                "representation": "metric_depth",
                "camera_model": "pinhole_fx",
                "frame_names": ["frame_000000.png", "frame_000001.png"],
                "completed_count": 2,
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "frame_000000.npz").write_bytes(b"raw-0")
    (raw_dir / "frame_000001.npz").write_bytes(b"raw-1")
    raw = RawClip(
        directory=raw_dir,
        depth=np.asarray(
            [
                [[1, 2, 3], [2, 4, 6]],
                [[2, 4, 6], [4, 8, 12]],
            ],
            dtype=np.float32,
        ),
        valid=np.ones((2, 2, 3), dtype=np.bool_),
        focal_x_normalized=np.asarray([1.0, 1.2], dtype=np.float32),
        frame_names=("frame_000000", "frame_000001"),
        inference_calls=2,
        inferred_frame_count=2,
    )
    relative_path = tmp_path / "relative.tmp.mp4"
    metric_path = tmp_path / "metric.tmp.mp4"
    relative_path.write_bytes(b"relative")
    metric_path.write_bytes(b"metric")
    sidecars = []
    for index, fraction in enumerate((0.25, 0.5)):
        sidecar = tmp_path / f"frame_{index:06d}.json"
        sidecar.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "frame_name": f"frame_{index:06d}",
                    "valid_pixel_count": 4,
                    "clamped_pixel_count": int(fraction * 4),
                    "clamped_fraction": fraction,
                }
            ),
            encoding="utf-8",
        )
        sidecars.append(sidecar)
    relative = ClipRender(
        mode="relative",
        output_path=relative_path,
        hole_mask=np.asarray([[[False, True]], [[False, False]]], dtype=np.bool_),
        output_shape=(2, 6),
    )
    metric = ClipRender(
        mode="metric_camera",
        output_path=metric_path,
        hole_mask=np.asarray([[[True, True]], [[False, False]]], dtype=np.bool_),
        total_disparity_pixels=np.asarray(
            [
                [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
                [[2.0, 4.0, 6.0], [2.0, 4.0, 6.0]],
            ],
            dtype=np.float64,
        ),
        retained_source_xyxy=(0, 0, 6, 4),
        clamp_sidecars=tuple(sidecars),
        disparity_valid_mask=np.ones((2, 2, 3), dtype=np.bool_),
        output_shape=(2, 6),
    )
    return raw, relative, metric


def test_roi_mapping_uses_floor_start_ceil_end_and_retained_intersection() -> None:
    assert map_source_roi((1, 1, 3, 2), (4, 6), (2, 3), (0, 0, 6, 4)) == (0, 0, 2, 2)
    assert map_source_roi((2, 1, 3, 2), (4, 6), (2, 2), (1, 0, 5, 4)) == (0, 0, 2, 2)
    with pytest.raises(ValueError, match="no retained samples"):
        map_source_roi((0, 0, 1, 1), (4, 6), (2, 2), (2, 0, 6, 4))


def test_metric_formulas_statistics_holes_and_clamps_are_numeric(tmp_path: Path) -> None:
    raw, relative, metric = _metric_fixture(tmp_path)
    measured = compute_clip_measurements(
        raw,
        relative,
        metric,
        source_shape=(4, 6),
        source_roi_xywh=(1, 1, 2, 2),
        inference_seconds=4.0,
    )
    assert measured["inference_seconds_per_frame"] == 2.0
    assert measured["focal_min"] == 1.0
    assert measured["focal_max"] == pytest.approx(1.2)
    assert measured["focal_stddev"] == pytest.approx(0.1)
    assert measured["roi_metric_depth_mean_per_frame"] == [2.25, 4.5]
    assert measured["roi_metric_depth_stddev"] == 1.125
    assert measured["roi_output_disparity_mean_per_frame"] == [1.5, 3.0]
    assert measured["roi_output_disparity_stddev"] == 0.75
    assert measured["relative_hole_fraction"] == 0.25
    assert measured["metric_hole_fraction"] == 0.5
    assert measured["metric_clamped_fraction_per_frame"] == [0.25, 0.5]
    assert all(type(value) in {int, float, list} for value in measured.values())


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda raw, rel, met: replace(raw, inference_calls=3), "call count"),
        (lambda raw, rel, met: replace(raw, inferred_frame_count=1), "frame count"),
        (
            lambda raw, rel, met: replace(
                raw, focal_x_normalized=np.asarray([1.0, np.nan], np.float32)
            ),
            "focal",
        ),
        (
            lambda raw, rel, met: replace(met, hole_mask=np.asarray([[[2]]], np.int8)),
            "hole mask",
        ),
        (
            lambda raw, rel, met: replace(met, total_disparity_pixels=np.full((2, 2, 3), np.nan)),
            "disparity",
        ),
        (
            lambda raw, rel, met: replace(met, disparity_valid_mask=np.zeros((2, 2, 3), np.bool_)),
            "valid output-disparity",
        ),
    ],
)
def test_measurement_structural_validation(tmp_path: Path, change, match: str) -> None:
    raw, relative, metric = _metric_fixture(tmp_path)
    changed = change(raw, relative, metric)
    if isinstance(changed, RawClip):
        raw = changed
    elif changed.mode == "relative":
        relative = changed
    else:
        metric = changed
    with pytest.raises((TypeError, ValueError), match=match):
        compute_clip_measurements(
            raw,
            relative,
            metric,
            source_shape=(4, 6),
            source_roi_xywh=(1, 1, 2, 2),
            inference_seconds=1.0,
        )


class FakeCuda:
    def __init__(self, events: list[str], peaks: list[int] | None = None) -> None:
        self.events = events
        self.peaks = iter(peaks) if peaks is not None else None

    def synchronize(self) -> None:
        self.events.append("cuda.sync")

    def reset_peak_memory_stats(self) -> None:
        self.events.append("cuda.reset")

    def max_memory_allocated(self) -> int:
        self.events.append("cuda.max")
        return 4096 if self.peaks is None else next(self.peaks)


class FakeSession:
    def __init__(self, model_size: str, root: Path, events: list[str], fail: str | None) -> None:
        self.model_size = model_size
        self.root = root
        self.events = events
        self.fail = fail
        self.raw_by_clip: dict[str, RawClip] = {}

    def load(self) -> None:
        self.events.append(f"{self.model_size}.load")
        if self.fail == f"{self.model_size}.load":
            raise RuntimeError("load boom")

    def infer_fixed(self, _path: Path, _resolution: int) -> FixedDepth:
        self.events.append(f"{self.model_size}.fixed")
        self.events.append(f"{self.model_size}.fixed.path={_path}")
        if self.fail == f"{self.model_size}.fixed":
            raise RuntimeError("fixed boom")
        return FixedDepth(
            np.asarray([[1.0, np.inf], [2.0, 3.0]], np.float32),
            np.asarray([[True, False], [True, True]], bool),
            np.float32(1.5),
        )

    def infer_clip(self, clip, _resolution: int, workspace: Path) -> RawClip:
        self.events.append(f"{self.model_size}.{clip.clip_id}.infer")
        self.events.append(f"{self.model_size}.{clip.clip_id}.path={clip.path}")
        if self.fail == f"{self.model_size}.{clip.clip_id}.infer":
            raise RuntimeError("clip inference boom")
        raw_dir = workspace / "02_depth_raw"
        raw_dir.mkdir(parents=True)
        (raw_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "storage_status": "ready",
                    "representation": "metric_depth",
                    "camera_model": "pinhole_fx",
                    "frame_names": ["frame_000000.png", "frame_000001.png"],
                    "completed_count": 2,
                }
            ),
            encoding="utf-8",
        )
        for index in range(2):
            (raw_dir / f"frame_{index:06d}.npz").write_bytes(f"raw-{index}".encode())
        raw = RawClip(
            raw_dir,
            np.asarray(
                [
                    [[1, 2, 3], [2, 4, 6]],
                    [[2, 4, 6], [4, 8, 12]],
                ],
                np.float32,
            ),
            np.ones((2, 2, 3), bool),
            np.asarray([1.0, 1.2], np.float32),
            ("frame_000000", "frame_000001"),
            2,
            2,
        )
        self.raw_by_clip[clip.clip_id] = raw
        return raw

    def render_clip(
        self, clip, raw: RawClip, mode: str, output_path: Path, _settings: dict[str, Any]
    ) -> ClipRender:
        self.events.append(f"{self.model_size}.{clip.clip_id}.render.{mode}")
        if self.fail == f"{self.model_size}.{clip.clip_id}.{mode}":
            raise RuntimeError("render boom")
        output_path.write_bytes(f"{self.model_size}-{clip.clip_id}-{mode}".encode())
        if mode == "relative":
            return ClipRender(
                "relative",
                output_path,
                np.zeros((2, 2, 3), bool),
                output_shape=(4, 12),
            )
        sidecars = []
        sidecar_directory = raw.directory.parent / "clamp_stats"
        sidecar_directory.mkdir()
        for index, fraction in enumerate((0.0, 0.25)):
            sidecar = sidecar_directory / f"{clip.clip_id}-{index}.json"
            sidecar.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "frame_name": f"frame_{index:06d}",
                        "valid_pixel_count": 4,
                        "clamped_pixel_count": int(fraction * 4),
                        "clamped_fraction": fraction,
                    }
                ),
                encoding="utf-8",
            )
            sidecars.append(sidecar)
        return ClipRender(
            "metric_camera",
            output_path,
            np.zeros((2, 2, 3), bool),
            np.ones((2, 2, 3), np.float64),
            (0, 0, 6, 4),
            tuple(sidecars),
            np.ones((2, 2, 3), np.bool_),
            (4, 12),
        )

    def unload(self) -> None:
        self.events.append(f"{self.model_size}.unload")


def _runner(
    tmp_path: Path,
    *,
    fail: str | None = None,
    replace_fn=os.replace,
    media_probe=_output_media_probe,
    peaks: list[int] | None = None,
):
    events: list[str] = []

    def factory(model_size: str, _repository: str, _revision: str, _device: str):
        return FakeSession(model_size, tmp_path, events, fail)

    ticks = iter(float(index) for index in range(200))
    dependencies = ReleaseDependencies(
        session_factory=factory,
        perf_counter=lambda: next(ticks),
        utc_now=lambda: "2026-08-17T12:00:00Z",
        system_probe=lambda _device: {
            "os": "test-os",
            "python": "3.12-test",
            "pytorch": "2-test",
            "cuda": "12-test",
            "gpu": "fake-gpu",
        },
        git_probe=lambda: ("f" * 40, False),
        media_probe=media_probe,
        cuda=FakeCuda(events, peaks),
        publisher=AtomicPublisher(replace_fn=replace_fn, token_factory=lambda: "token"),
    )
    return ReleaseRunner(dependencies), events


def test_production_settings_keep_fractional_source_fps_as_original(tmp_path: Path) -> None:
    config_path, _payload = _write_inputs(tmp_path)
    clip = replace(_load(config_path).clips[0], fps=24000 / 1001)
    session = object.__new__(ProductionVariantSession)
    session.model_size = "vits"
    session.device = "cpu"

    settings = session._resolved_settings(clip, 1080, "relative")

    assert settings["target_fps"] == "original"


@pytest.mark.parametrize(
    "failure_kind, expected_stage",
    [
        ("git", "git_probe"),
        ("system", "system_probe"),
        ("cuda", "load_model"),
        ("construct", "construct_model"),
        ("vits.load", "load_model"),
        ("vits.fixed", "fixed_image_inference"),
        ("vits.indoor-near.infer", "clip_inference"),
        ("vits.indoor-near.relative", "render_relative"),
    ],
)
def test_every_runner_stage_failure_publishes_partial_safe_incomplete_reports(
    tmp_path: Path,
    failure_kind: str,
    expected_stage: str,
) -> None:
    config_path, _payload = _write_inputs(tmp_path)
    runner, _events = _runner(tmp_path, fail=failure_kind)
    if failure_kind == "git":
        runner.dependencies = replace(
            runner.dependencies,
            git_probe=lambda: (_ for _ in ()).throw(RuntimeError("git boom")),
        )
    elif failure_kind == "system":
        runner.dependencies = replace(
            runner.dependencies,
            system_probe=lambda _device: (_ for _ in ()).throw(RuntimeError("system boom")),
        )
    elif failure_kind == "cuda":

        class FailingCuda:
            def synchronize(self) -> None:
                raise RuntimeError("cuda probe boom")

            def reset_peak_memory_stats(self) -> None:
                raise AssertionError("reset must follow synchronize")

            def max_memory_allocated(self) -> int:
                raise AssertionError("max must follow synchronize")

        runner.dependencies = replace(runner.dependencies, cuda=FailingCuda())
    elif failure_kind == "construct":
        runner.dependencies = replace(
            runner.dependencies,
            session_factory=lambda *_args: (_ for _ in ()).throw(RuntimeError("construct boom")),
        )
    output = tmp_path / f"evidence-{failure_kind.replace('.', '-')}"

    with pytest.raises(ReleaseRunFailed) as caught:
        runner.run(_load(config_path), output, "cuda" if failure_kind == "cuda" else "cpu", 1080)

    assert caught.value.report["status"] == "incomplete"
    assert caught.value.report["failures"][0]["stage"] == expected_stage
    assert (
        json.loads((output / "report.json").read_text(encoding="utf-8"))["status"] == "incomplete"
    )
    assert "Status: `incomplete`" in (output / "report.md").read_text(encoding="utf-8")


def test_markdown_renders_a_variant_before_fixed_image_fields_exist() -> None:
    report = {
        "tool_schema_version": 1,
        "timestamp_utc": None,
        "status": "incomplete",
        "project_git": {"commit": None, "dirty": None},
        "system": {"os": None, "python": None, "pytorch": None, "cuda": None, "gpu": None},
        "moge_source_commit": "a" * 40,
        "adapter_resolution_level": 9,
        "requested_depth_resolution": 1080,
        "inputs": {"corpus_config": "corpus.json", "fixed_image": {}, "clips": []},
        "settings": {"device": "cpu"},
        "variants": [
            {
                "model_size": "vits",
                "repository": "repository",
                "revision": "b" * 40,
                "clips": [],
            }
        ],
        "failures": [],
    }

    markdown = render_markdown_report(report)

    assert "Status: `incomplete`" in markdown
    assert "### vits" in markdown


def test_fake_runner_loads_once_reuses_raw_for_modes_and_unloads_in_order(tmp_path: Path) -> None:
    config_path, _payload = _write_inputs(tmp_path)
    runner, events = _runner(tmp_path)
    report = runner.run(_load(config_path), tmp_path / "evidence", "cuda", 1080)
    for model_size in ("vits", "vitb", "vitl"):
        assert events.count(f"{model_size}.load") == 1
        assert events.count(f"{model_size}.unload") == 1
        for clip_id in CANONICAL_CLIP_IDS:
            assert events.count(f"{model_size}.{clip_id}.infer") == 1
            relative = events.index(f"{model_size}.{clip_id}.render.relative")
            metric = events.index(f"{model_size}.{clip_id}.render.metric_camera")
            assert relative < metric
    assert events.index("vits.unload") < events.index("vitb.load") < events.index("vitl.load")
    assert report["status"] == "complete"


def test_models_receive_authenticated_snapshots_and_private_workspace_is_cleaned(
    tmp_path: Path,
) -> None:
    config_path, _payload = _write_inputs(tmp_path)
    corpus = _load(config_path)
    originals = {corpus.fixed_image.path, *(clip.path for clip in corpus.clips)}
    runner, events = _runner(tmp_path)
    output = tmp_path / "evidence"

    report = runner.run(corpus, output, "cpu", 1080)

    seen_paths = {Path(event.split(".path=", 1)[1]) for event in events if ".path=" in event}
    assert seen_paths
    assert seen_paths.isdisjoint(originals)
    assert all(path.suffix in {".png", ".mp4"} for path in seen_paths)
    assert all(not path.exists() for path in seen_paths)
    assert all(output not in path.parents for path in seen_paths)
    serialized = json.dumps(report)
    assert all(str(path) not in serialized for path in seen_paths)
    assert Path(report["inputs"]["fixed_image"]["path"]) == corpus.fixed_image.path
    assert {Path(item["path"]) for item in report["inputs"]["clips"]} == {
        clip.path for clip in corpus.clips
    }


def test_snapshot_copy_is_rehashed_before_any_model_work(tmp_path: Path) -> None:
    config_path, _payload = _write_inputs(tmp_path)
    runner, events = _runner(tmp_path)

    def corrupting_copy(source: Path, destination: Path) -> None:
        shutil.copyfile(source, destination)
        if source.name == "outdoor-far.mp4":
            destination.write_bytes(b"changed while copying")

    runner.dependencies = replace(runner.dependencies, snapshot_copy=corrupting_copy)
    output = tmp_path / "evidence"

    with pytest.raises(ReleaseRunFailed, match="snapshot checksum mismatch") as caught:
        runner.run(_load(config_path), output, "cpu", 1080)

    assert caught.value.report["failures"][0]["stage"] == "snapshot_inputs"
    assert not any(event.endswith(".load") for event in events)
    assert json.loads((output / "report.json").read_text())["status"] == "incomplete"


def test_original_mutation_after_snapshot_does_not_change_model_inputs(tmp_path: Path) -> None:
    config_path, _payload = _write_inputs(tmp_path)
    corpus = _load(config_path)
    original_hashes = {
        corpus.fixed_image.path: corpus.fixed_image.sha256,
        **{clip.path: clip.sha256 for clip in corpus.clips},
    }
    runner, _events = _runner(tmp_path)
    copied = 0

    def mutate_after_last_copy(source: Path, destination: Path) -> None:
        nonlocal copied
        shutil.copyfile(source, destination)
        copied += 1
        if copied == 4:
            for original in original_hashes:
                original.write_bytes(b"changed after authenticated snapshot")

    runner.dependencies = replace(runner.dependencies, snapshot_copy=mutate_after_last_copy)

    report = runner.run(corpus, tmp_path / "evidence", "cpu", 1080)

    assert report["status"] == "complete"
    assert report["inputs"]["fixed_image"]["sha256"] == original_hashes[corpus.fixed_image.path]
    assert [item["sha256"] for item in report["inputs"]["clips"]] == [
        original_hashes[clip.path] for clip in corpus.clips
    ]


def test_cuda_measurement_call_order_and_cpu_zero_peak(tmp_path: Path) -> None:
    config_path, _payload = _write_inputs(tmp_path)
    runner, events = _runner(tmp_path)
    report = runner.run(_load(config_path), tmp_path / "cuda-evidence", "cuda", 1080)
    load_index = events.index("vits.load")
    assert events[load_index - 3 : load_index] == ["cuda.sync", "cuda.reset", "cuda.sync"]
    assert events[load_index + 1 : load_index + 3] == ["cuda.sync", "cuda.max"]
    assert report["variants"][0]["peak_vram_bytes"] == 4096
    assert events.count("cuda.max") == 15

    runner, cpu_events = _runner(tmp_path)
    cpu_report = runner.run(_load(config_path), tmp_path / "cpu-evidence", "cpu", 1080)
    assert not any(event.startswith("cuda.") for event in cpu_events)
    assert cpu_report["variants"][0]["peak_vram_bytes"] == 0
    assert cpu_report["variants"][0]["load_peak_vram_bytes"] == 0
    assert cpu_report["variants"][0]["fixed_image_peak_vram_bytes"] == 0
    assert all(
        clip["inference_peak_vram_bytes"] == 0 for clip in cpu_report["variants"][0]["clips"]
    )


def test_variant_peak_vram_is_max_of_load_fixed_and_every_clip(tmp_path: Path) -> None:
    config_path, _payload = _write_inputs(tmp_path)
    peaks = [10, 20, 30, 50, 40] + [1] * 10
    runner, events = _runner(tmp_path, peaks=peaks)

    report = runner.run(_load(config_path), tmp_path / "evidence", "cuda", 1080)

    first = report["variants"][0]
    assert first["load_peak_vram_bytes"] == 10
    assert first["fixed_image_peak_vram_bytes"] == 20
    assert [clip["inference_peak_vram_bytes"] for clip in first["clips"]] == [30, 50, 40]
    assert first["peak_vram_bytes"] == 50
    assert events.count("cuda.reset") == 15


def test_complete_report_schema_numeric_values_and_all_21_hashes(tmp_path: Path) -> None:
    config_path, _payload = _write_inputs(tmp_path)
    runner, _events = _runner(tmp_path)
    output = tmp_path / "evidence"
    report = runner.run(_load(config_path), output, "cuda", 1080)
    assert {
        "tool_schema_version",
        "timestamp_utc",
        "status",
        "project_git",
        "system",
        "moge_source_commit",
        "adapter_resolution_level",
        "requested_depth_resolution",
        "inputs",
        "settings",
        "variants",
        "failures",
    } <= report.keys()
    assert [item["model_size"] for item in report["variants"]] == ["vits", "vitb", "vitl"]
    required_variant = {
        "repository",
        "revision",
        "load_seconds",
        "load_peak_vram_bytes",
        "peak_vram_bytes",
        "fixed_image_inference_seconds",
        "fixed_image_peak_vram_bytes",
        "fixed_image_native_shape",
        "fixed_image_focal_x_normalized",
        "fixed_image_valid_metric_pixels",
        "fixed_image_output",
        "clips",
    }
    required_clip = {
        "id",
        "input_sha256",
        "inference_seconds_per_frame",
        "focal_min",
        "focal_max",
        "focal_stddev",
        "roi_metric_depth_mean_per_frame",
        "roi_metric_depth_stddev",
        "roi_output_disparity_mean_per_frame",
        "roi_output_disparity_stddev",
        "relative_hole_fraction",
        "metric_hole_fraction",
        "metric_clamped_fraction_per_frame",
        "relative_output",
        "metric_output",
        "raw_stage_sha256",
        "input_fps",
        "input_frame_count",
        "adapter_inference_call_count",
        "inferred_frame_count",
        "inference_peak_vram_bytes",
        "output_shape",
    }
    records = []
    for variant in report["variants"]:
        assert required_variant <= variant.keys()
        assert type(variant["load_seconds"]) is float
        assert type(variant["peak_vram_bytes"]) is int
        records.append(variant["fixed_image_output"])
        for clip in variant["clips"]:
            assert required_clip <= clip.keys()
            records.extend((clip["relative_output"], clip["metric_output"]))
            assert clip["adapter_inference_call_count"] == clip["inferred_frame_count"] == 2
    assert len(records) == 21
    for record in records:
        path = output / record["path"]
        assert path.is_file()
        assert record["sha256"] == _sha(path)
        assert not Path(record["path"]).is_absolute()
        if path.suffix == ".mp4":
            assert record["media"] == _output_media_probe(path)
    assert report["inputs"]["corpus_config_sha256"] == _sha(config_path)
    assert report["inputs"]["authenticated_private_snapshot"] is True
    assert json.loads((output / "report.json").read_text(encoding="utf-8")) == report
    assert not list(output.rglob("*.tmp*"))
    assert {path.name for path in output.iterdir()} == {
        "report.json",
        "report.md",
        "vits",
        "vitb",
        "vitl",
    }
    assert all(len(list((output / model).iterdir())) == 7 for model in ("vits", "vitb", "vitl"))


def test_markdown_is_deterministic_complete_and_has_unchecked_inspection_rows(
    tmp_path: Path,
) -> None:
    config_path, _payload = _write_inputs(tmp_path)
    runner, _events = _runner(tmp_path)
    report = runner.run(_load(config_path), tmp_path / "evidence", "cpu", 1080)
    first = render_markdown_report(report)
    assert first == render_markdown_report(copy.deepcopy(report))
    labels = (
        "edge tearing",
        "foreground sign",
        "scale pumping",
        "focal breathing",
        "convergence placement",
        "viewing discomfort",
    )
    assert first.count("| [ ] |") == 54
    for variant in ("vits", "vitb", "vitl"):
        for clip_id in CANONICAL_CLIP_IDS:
            assert first.count(f"| [ ] | {variant}/{clip_id} |") == len(labels)
    for label in labels:
        assert label in first
    assert "not portable thresholds" in first
    assert "do not establish physical calibration" in first
    assert "temporal stability" in first
    assert "6x4" in first
    assert "1:1" in first
    assert report["inputs"]["corpus_config_sha256"] in first
    for variant in report["variants"]:
        for clip in variant["clips"]:
            assert clip["raw_stage_sha256"] in first
            assert str(clip["adapter_inference_call_count"]) in first
            assert str(clip["inferred_frame_count"]) in first
            assert str(clip["input_fps"]) in first
            assert str(clip["input_frame_count"]) in first
            for field in ("relative_output", "metric_output"):
                assert clip[field]["sha256"] in first
                for measured in clip[field]["media"].values():
                    assert str(measured) in first


def test_runner_rejects_non_v3_or_incomplete_raw_stage_before_render(tmp_path: Path) -> None:
    config_path, _payload = _write_inputs(tmp_path)
    runner, events = _runner(tmp_path)
    original_factory = runner.dependencies.session_factory

    class BadRawSession:
        def __init__(self, inner) -> None:
            self.inner = inner

        def __getattr__(self, name: str):
            return getattr(self.inner, name)

        def infer_clip(self, clip, resolution, workspace):
            raw = self.inner.infer_clip(clip, resolution, workspace)
            (raw.directory / "metadata.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "storage_status": "ready",
                        "representation": "metric_depth",
                        "camera_model": "pinhole_fx",
                        "frame_names": [f"{name}.png" for name in raw.frame_names],
                        "completed_count": len(raw.frame_names),
                    }
                ),
                encoding="utf-8",
            )
            return raw

    runner.dependencies = replace(
        runner.dependencies,
        session_factory=lambda *args: BadRawSession(original_factory(*args)),
    )
    with pytest.raises(ReleaseRunFailed, match="schema v3"):
        runner.run(_load(config_path), tmp_path / "bad-raw", "cpu", 1080)
    assert not any("render" in event for event in events)


def test_failure_unloads_writes_noncomplete_reports_and_preserves_earlier_evidence(
    tmp_path: Path,
) -> None:
    config_path, _payload = _write_inputs(tmp_path)
    runner, events = _runner(tmp_path, fail="vitb.outdoor-far.metric_camera")
    output = tmp_path / "evidence"
    with pytest.raises(ReleaseRunFailed) as caught:
        runner.run(_load(config_path), output, "cuda", 1080)
    report = caught.value.report
    assert report["status"] == "incomplete"
    assert report["failures"] == [
        {
            "variant": "vitb",
            "clip": "outdoor-far",
            "stage": "render_metric_camera",
            "error_type": "RuntimeError",
            "message": "render boom",
        }
    ]
    assert "vitb.unload" in events
    assert "vitl.load" not in events
    assert (output / "vits" / "fixed-image-depth.npz").is_file()
    assert (output / "vitb" / "indoor-near-relative.mp4").is_file()
    assert not (output / "vitb" / "outdoor-far-metric-camera.mp4").exists()
    assert json.loads((output / "report.json").read_text())["status"] != "complete"
    assert "Status: `incomplete`" in (output / "report.md").read_text(encoding="utf-8")


def test_fixed_npz_validation_and_promotion_failures_keep_atomic_reports(
    tmp_path: Path,
) -> None:
    config_path, _payload = _write_inputs(tmp_path)
    runner, _events = _runner(tmp_path)
    original_factory = runner.dependencies.session_factory

    class InvalidFixedSession:
        def __init__(self, inner) -> None:
            self.inner = inner

        def __getattr__(self, name: str):
            return getattr(self.inner, name)

        def infer_fixed(self, path, resolution):
            self.inner.infer_fixed(path, resolution)
            return FixedDepth(
                np.zeros((2, 2), np.float32),
                np.zeros((2, 2), np.bool_),
                np.float32(1.0),
            )

    runner.dependencies = replace(
        runner.dependencies,
        session_factory=lambda *args: InvalidFixedSession(original_factory(*args)),
    )
    invalid_output = tmp_path / "invalid-fixed"
    with pytest.raises(ReleaseRunFailed) as invalid:
        runner.run(_load(config_path), invalid_output, "cpu", 1080)
    assert invalid.value.report["failures"][0]["stage"] == "fixed_image_npz"
    assert json.loads((invalid_output / "report.json").read_text())["status"] == "incomplete"

    def reject_fixed_promotion(source: Path, target: Path) -> None:
        if Path(target).name == "fixed-image-depth.npz":
            raise OSError("fixed promotion boom")
        os.replace(source, target)

    runner, _events = _runner(tmp_path, replace_fn=reject_fixed_promotion)
    promotion_output = tmp_path / "promotion-fixed"
    with pytest.raises(ReleaseRunFailed, match="fixed promotion boom") as promotion:
        runner.run(_load(config_path), promotion_output, "cpu", 1080)
    assert promotion.value.report["failures"][0]["stage"] == "fixed_image_promotion"
    assert not (promotion_output / "vits" / "fixed-image-depth.npz").exists()
    assert json.loads((promotion_output / "report.json").read_text())["status"] == "incomplete"


def test_report_publication_failure_is_recorded_and_retried_atomically(tmp_path: Path) -> None:
    config_path, _payload = _write_inputs(tmp_path)
    failed = False

    def fail_once(source: Path, target: Path) -> None:
        nonlocal failed
        if Path(target).name == "report.json" and not failed:
            failed = True
            raise OSError("report boom")
        os.replace(source, target)

    runner, _events = _runner(tmp_path, replace_fn=fail_once)
    output = tmp_path / "evidence"

    with pytest.raises(ReleaseRunFailed, match="report boom") as caught:
        runner.run(_load(config_path), output, "cpu", 1080)

    assert caught.value.report["failures"][0]["stage"] == "initial_report"
    assert json.loads((output / "report.json").read_text())["status"] == "incomplete"
    assert "Status: `incomplete`" in (output / "report.md").read_text(encoding="utf-8")


def test_complete_report_failure_restores_both_reports_to_incomplete(tmp_path: Path) -> None:
    config_path, _payload = _write_inputs(tmp_path)
    json_publications = 0

    def fail_complete_json(source: Path, target: Path) -> None:
        nonlocal json_publications
        if Path(target).name == "report.json":
            json_publications += 1
            if json_publications == 3:
                raise OSError("complete report boom")
        os.replace(source, target)

    runner, _events = _runner(tmp_path, replace_fn=fail_complete_json)
    output = tmp_path / "evidence"

    with pytest.raises(ReleaseRunFailed, match="complete report boom") as caught:
        runner.run(_load(config_path), output, "cpu", 1080)

    assert caught.value.report["failures"][0]["stage"] == "publish_complete_reports"
    assert json.loads((output / "report.json").read_text())["status"] == "incomplete"
    assert "Status: `incomplete`" in (output / "report.md").read_text(encoding="utf-8")


def test_last_report_publication_error_is_surfaced_without_destroying_old_reports(
    tmp_path: Path,
) -> None:
    config_path, _payload = _write_inputs(tmp_path)
    output = tmp_path / "evidence"
    output.mkdir()
    (output / "report.json").write_text('{"status":"incomplete"}\n', encoding="utf-8")
    (output / "report.md").write_text("Status: `incomplete`\n", encoding="utf-8")

    def reject_reports(source: Path, target: Path) -> None:
        if Path(target).name in {"report.json", "report.md"}:
            raise OSError("reports unavailable")
        os.replace(source, target)

    runner, _events = _runner(tmp_path, replace_fn=reject_reports)

    with pytest.raises(ReleaseRunFailed, match="publication also failed") as caught:
        runner.run(_load(config_path), output, "cpu", 1080)

    assert isinstance(caught.value.report_error, OSError)
    assert json.loads((output / "report.json").read_text())["status"] == "incomplete"
    assert "Status: `incomplete`" in (output / "report.md").read_text(encoding="utf-8")


def test_atomic_publisher_uses_sibling_temp_cleanup_and_preserves_destination_on_failure(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "result.bin"
    destination.write_bytes(b"old")
    seen: list[tuple[Path, Path]] = []

    def broken_replace(source: Path, target: Path) -> None:
        seen.append((Path(source), Path(target)))
        raise OSError("replace failed")

    publisher = AtomicPublisher(replace_fn=broken_replace, token_factory=lambda: "unique")
    with pytest.raises(OSError, match="replace failed"):
        publisher.write_bytes(destination, b"new")
    assert destination.read_bytes() == b"old"
    assert seen[0][0].parent == destination.parent
    assert seen[0][0] != destination
    assert not seen[0][0].exists()


def test_stale_unrecorded_output_and_raw_tampering_are_rejected(tmp_path: Path) -> None:
    config_path, _payload = _write_inputs(tmp_path)
    output = tmp_path / "evidence"
    stale = output / "vits" / "fixed-image-depth.npz"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")
    runner, events = _runner(tmp_path)
    with pytest.raises(ReleaseRunFailed, match="stale") as caught:
        runner.run(_load(config_path), output, "cpu", 1080)
    assert caught.value.report["status"] == "incomplete"
    assert "vits.load" not in events
    assert stale.read_bytes() == b"stale"

    clean_output = tmp_path / "tamper-evidence"
    runner, _events = _runner(tmp_path)
    original_factory = runner.dependencies.session_factory

    class TamperingSession:
        def __init__(self, inner) -> None:
            self.inner = inner

        def __getattr__(self, name: str):
            return getattr(self.inner, name)

        def render_clip(self, clip, raw, mode, output_path, settings):
            result = self.inner.render_clip(clip, raw, mode, output_path, settings)
            if mode == "relative":
                (raw.directory / "frame_000000.npz").write_bytes(b"tampered")
            return result

    runner.dependencies = replace(
        runner.dependencies,
        session_factory=lambda *args: TamperingSession(original_factory(*args)),
    )
    with pytest.raises(ReleaseRunFailed, match="raw stage changed"):
        runner.run(_load(config_path), clean_output, "cpu", 1080)


def test_unexpected_nonreport_file_is_rejected_but_failure_reports_can_be_updated(
    tmp_path: Path,
) -> None:
    config_path, _payload = _write_inputs(tmp_path)
    output = tmp_path / "evidence"
    output.mkdir()
    (output / "report.json").write_text('{"status":"incomplete"}', encoding="utf-8")
    (output / "report.md").write_text("old failure report", encoding="utf-8")
    junk = output / "notes.txt"
    junk.write_text("unrecorded", encoding="utf-8")
    runner, events = _runner(tmp_path)
    with pytest.raises(ReleaseRunFailed, match="stale or unrecorded"):
        runner.run(_load(config_path), output, "cpu", 1080)
    assert "vits.load" not in events
    assert junk.read_text(encoding="utf-8") == "unrecorded"
    assert json.loads((output / "report.json").read_text())["status"] == "incomplete"


def test_stale_empty_directory_is_rejected_before_model_work(tmp_path: Path) -> None:
    config_path, _payload = _write_inputs(tmp_path)
    output = tmp_path / "evidence"
    (output / "empty").mkdir(parents=True)
    runner, events = _runner(tmp_path)

    with pytest.raises(ReleaseRunFailed, match="stale or unrecorded"):
        runner.run(_load(config_path), output, "cpu", 1080)

    assert not any(event.endswith(".load") for event in events)
    assert (output / "empty").is_dir()


@pytest.mark.parametrize("extra_kind", ["file", "directory", "nested"])
def test_complete_tree_rejects_every_unrecorded_entry(tmp_path: Path, extra_kind: str) -> None:
    config_path, _payload = _write_inputs(tmp_path)
    published = 0

    def add_extra_after_last_artifact(source: Path, target: Path) -> None:
        nonlocal published
        os.replace(source, target)
        if Path(target).suffix in {".mp4", ".npz"}:
            published += 1
            if published == 21:
                root = Path(target).parents[1]
                if extra_kind == "file":
                    (root / "notes.txt").write_text("extra", encoding="utf-8")
                elif extra_kind == "directory":
                    (root / "empty").mkdir()
                else:
                    nested = root / "vits" / "nested"
                    nested.mkdir()
                    (nested / "extra.bin").write_bytes(b"extra")

    runner, _events = _runner(tmp_path, replace_fn=add_extra_after_last_artifact)
    output = tmp_path / f"evidence-{extra_kind}"

    with pytest.raises(ReleaseRunFailed, match="missing or unrecorded") as caught:
        runner.run(_load(config_path), output, "cpu", 1080)

    assert caught.value.report["status"] == "incomplete"
    assert json.loads((output / "report.json").read_text())["status"] == "incomplete"


def test_atomic_publisher_rejects_escape_and_linked_parent(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    publisher = AtomicPublisher(token_factory=lambda: "token")
    with pytest.raises(ValueError, match="escaped"):
        publisher.write_bytes(tmp_path / "outside.bin", b"bad", root=root)

    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = root / "vits"
    try:
        os.symlink(outside, linked_parent, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this Windows host")
    with pytest.raises(ValueError, match="symlink or reparse"):
        publisher.write_bytes(linked_parent / "escaped.bin", b"bad", root=root)
    assert not (outside / "escaped.bin").exists()


def test_runner_rejects_symlink_output_root_where_supported(tmp_path: Path) -> None:
    config_path, _payload = _write_inputs(tmp_path)
    real_output = tmp_path / "real-output"
    real_output.mkdir()
    linked_output = tmp_path / "linked-output"
    try:
        os.symlink(real_output, linked_output, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this Windows host")
    runner, _events = _runner(tmp_path)

    with pytest.raises(ValueError, match="symlink or reparse"):
        runner.run(_load(config_path), linked_output, "cpu", 1080)


def test_complete_status_requires_revalidation_of_all_recorded_hashes(tmp_path: Path) -> None:
    config_path, _payload = _write_inputs(tmp_path)
    replaced_media = 0

    def tampering_replace(source: Path, target: Path) -> None:
        nonlocal replaced_media
        os.replace(source, target)
        if Path(target).suffix in {".mp4", ".npz"}:
            replaced_media += 1
            if replaced_media == 21:
                first = Path(target).parents[1] / "vits" / "indoor-near-relative.mp4"
                first.write_bytes(b"late tamper")

    runner, _events = _runner(tmp_path, replace_fn=tampering_replace)
    with pytest.raises(ReleaseRunFailed, match="recorded hash mismatch") as caught:
        runner.run(_load(config_path), tmp_path / "evidence", "cpu", 1080)
    assert caught.value.report["status"] == "incomplete"


def test_media_is_probed_before_atomic_promotion(tmp_path: Path) -> None:
    config_path, _payload = _write_inputs(tmp_path)

    def wrong_width(_path: Path) -> dict[str, Any]:
        return {**_output_media_probe(_path), "width": 11}

    runner, _events = _runner(tmp_path, media_probe=wrong_width)
    output = tmp_path / "evidence"

    with pytest.raises(ReleaseRunFailed, match="packed output") as caught:
        runner.run(_load(config_path), output, "cpu", 1080)

    assert caught.value.report["failures"][0]["stage"] == "validate_relative_media"
    assert not (output / "vits" / "indoor-near-relative.mp4").exists()
    assert json.loads((output / "report.json").read_text())["status"] == "incomplete"


@pytest.mark.parametrize(
    "field, changed, message",
    [
        ("width", 11, "packed output"),
        ("height", 3, "packed output"),
        ("frame_count", 1, "frame count"),
        ("fps", 23.0, "fps"),
        ("duration", 1.0, "duration"),
    ],
)
def test_complete_revalidation_reprobes_every_media_property(
    tmp_path: Path,
    field: str,
    changed: int | float,
    message: str,
) -> None:
    config_path, _payload = _write_inputs(tmp_path)

    def changes_after_publication(path: Path) -> dict[str, Any]:
        properties = _output_media_probe(path)
        if path.name == "indoor-near-relative.mp4":
            properties[field] = changed
        return properties

    runner, _events = _runner(tmp_path, media_probe=changes_after_publication)
    output = tmp_path / f"evidence-{field}"

    with pytest.raises(ReleaseRunFailed, match=message) as caught:
        runner.run(_load(config_path), output, "cpu", 1080)

    assert caught.value.report["status"] == "incomplete"
    assert caught.value.report["failures"][0]["stage"] == "complete_revalidation"
    assert json.loads((output / "report.json").read_text())["status"] == "incomplete"
