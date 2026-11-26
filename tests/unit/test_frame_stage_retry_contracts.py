"""File-backed frame stages cannot attest broken PNG pairs, and must recover on retry."""

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.depth_surge_3d.processing.frames import distortion_processor, vr_assembler


@pytest.fixture
def stereo_pair(tmp_path):
    dirs = {
        key: tmp_path / key
        for key in (
            "left_frames",
            "right_frames",
            "left_distorted",
            "right_distorted",
            "left_cropped",
            "right_cropped",
            "vr_frames",
        )
    }
    for directory in dirs.values():
        directory.mkdir()
    left = np.full((8, 10, 3), (19, 70, 220), np.uint8)
    right = np.full((8, 10, 3), (200, 45, 7), np.uint8)
    name = "frame_000007.png"
    for key, eye in (("left_frames", left), ("right_frames", right)):
        assert cv2.imwrite(str(dirs[key] / name), eye)
    settings = dict(
        apply_distortion=True,
        fisheye_fov=90.0,
        fisheye_projection="equidistant",
        fisheye_crop_factor=1.0,
        crop_factor=0.8,
        per_eye_width=6,
        per_eye_height=4,
        vr_format="side_by_side",
        upscale_model="none",
    )
    return dirs, settings, name, left, right


@pytest.mark.parametrize("stage", ["distortion", "fused", "crop", "assembly"])
@pytest.mark.parametrize("payload", ["invalid-header", "header-without-pixels"])
def test_file_stages_do_not_attest_undecodable_pair_and_publish_exact_eyes_on_retry(
    stereo_pair, stage, payload
):
    dirs, settings, name, left, right = stereo_pair
    processor = distortion_processor.DistortionProcessor()
    assembly = vr_assembler.VRFrameAssembler()
    left_file, right_file = dirs["left_frames"] / name, dirs["right_frames"] / name
    outputs = [dirs["left_cropped"], dirs["right_cropped"]]
    messages = []
    tracker = SimpleNamespace(update_progress=lambda message, **fields: messages.append(fields))
    if stage == "distortion":
        outputs = [dirs["left_distorted"], dirs["right_distorted"]]

        def action():
            return processor.apply_distortion([left_file], [right_file], dirs, settings, tracker)

    elif stage == "fused":

        def action():
            return processor.apply_distortion_and_crop(
                [left_file], [right_file], dirs, settings, tracker
            )

    elif stage == "crop":
        settings["apply_distortion"] = False

        def action():
            return processor.crop_frames(dirs, settings, tracker, total_frames=1)

    else:
        dirs["left_cropped"], dirs["right_cropped"] = left_file.parent, right_file.parent
        outputs = [dirs["vr_frames"]]

        def action():
            return assembly.assemble_vr_frames(dirs, settings, tracker, total_frames=1)

    good = right_file.read_bytes()
    right_file.write_bytes(b"incomplete" if payload == "invalid-header" else good[:33])
    assert not action()
    assert not any((output / "metadata.json").exists() for output in outputs)
    right_file.write_bytes(good)
    assert action()
    marker = json.loads((outputs[0] / "metadata.json").read_bytes())
    assert marker["frame_names"] == [name]
    if stage == "assembly":
        result = cv2.imread(str(outputs[0] / name))
        assert result.shape == (4, 12, 3)
        np.testing.assert_array_equal(result[:, :6], np.broadcast_to(left[0, 0], (4, 6, 3)))
        np.testing.assert_array_equal(result[:, 6:], np.broadcast_to(right[0, 0], (4, 6, 3)))
    else:
        for output, source in zip(outputs, (left, right), strict=True):
            result = cv2.imread(str(output / name))
            np.testing.assert_array_equal(
                result[result.shape[0] // 2, result.shape[1] // 2], source[0, 0]
            )
    assert messages[-1]["step_total"] == messages[-1]["step_progress"] == 1
    # Stage reuse must leave verified bytes and progress unchanged.
    messages.clear()
    assert action()
    assert not messages


def test_vr_source_range_checks_both_eyes_ignoring_only_unrelated_entries(stereo_pair):
    dirs, settings, name, left, right = stereo_pair
    vr = vr_assembler.VRFrameAssembler()
    assert vr.resolve_direct_vr_source(dirs, settings, 0) is None
    l, r = dirs["left_cropped"], dirs["right_cropped"]
    assert cv2.imwrite(str(l / name), left)
    assert vr.resolve_direct_vr_source(dirs, settings, 0) is None
    assert cv2.imwrite(str(r / name), right)
    for directory in (l, r):
        (directory / "notes.txt").write_text("not part of the sequence")
    assert vr.resolve_direct_vr_source(dirs, settings, 1).first_index == 7
    assert vr.resolve_direct_vr_source(dirs, settings, 2) is None
    (l / "frame_000008.png").mkdir()
    assert vr.resolve_direct_vr_source(dirs, settings, 0) is None
    (l / "frame_000008.png").rmdir()
    assert cv2.imwrite(str(l / "frame_000009.png"), left)
    assert vr.resolve_direct_vr_source(dirs, settings, 0) is None
    assert vr._streaming_frame_range(l / "absent") is None


def test_stream_transform_rejects_geometry_drift_before_emitting_a_pair(stereo_pair, monkeypatch):
    _, settings, _, left, right = stereo_pair
    processor = distortion_processor.DistortionProcessor()
    for shape in ((0, 10), (True, 10), (8,)):
        with pytest.raises(ValueError, match="source shape"):
            processor.prepare_stream_transform(source_shape=shape, settings=settings)
    with pytest.raises(ValueError, match="target shape"):
        processor.prepare_stream_transform(
            source_shape=left.shape[:2], settings={**settings, "per_eye_height": 0}
        )
    transform = processor.prepare_stream_transform(source_shape=left.shape[:2], settings=settings)
    with pytest.raises(ValueError, match="pair shape"):
        transform(left, right[:, :-1])
    with monkeypatch.context() as patch:
        patch.setattr(distortion_processor, "apply_fisheye_square_crop", lambda raster, *_: raster)
        with pytest.raises(ValueError, match="wrong target shape"):
            transform(left, right)
    transformed = transform(left, right)
    assert all(raster.shape == (4, 6, 3) for raster in transformed)


def test_missing_output_or_failed_materialization_never_stamps_crop_or_assembly(
    stereo_pair, monkeypatch
):
    dirs, settings, name, left, right = stereo_pair
    processor = distortion_processor.DistortionProcessor()
    l, r = dirs["left_frames"] / name, dirs["right_frames"] / name
    assert not processor.apply_distortion_and_crop(
        [l], [r], dirs, {**settings, "apply_distortion": False}
    )
    assert not processor.apply_distortion_and_crop([l], [], dirs, settings)
    assert not processor.apply_distortion_and_crop(
        [l], [r], {**dirs, "left_cropped": None}, settings
    )
    assert not processor.apply_distortion([l], [r], {**dirs, "left_distorted": l}, settings)
    plain = {**settings, "apply_distortion": False, "crop_factor": 1.0}
    assert not processor.crop_frames({**dirs, "right_cropped": None}, plain)

    def cannot_copy(*args, **kwargs):
        raise PermissionError("directory cannot accept a new frame")

    with monkeypatch.context() as patch:
        patch.setattr(distortion_processor.os, "link", cannot_copy)
        patch.setattr(distortion_processor.shutil, "copy2", cannot_copy)
        assert not processor.crop_frames(dirs, plain)
    assert not (dirs["left_cropped"] / "metadata.json").exists()
    assert processor.crop_frames(dirs, plain)
    vr = vr_assembler.VRFrameAssembler()
    assert not vr.assemble_vr_frames({**dirs, "vr_frames": None}, settings, total_frames=1)
    with monkeypatch.context() as patch:
        patch.setattr(cv2, "imwrite", lambda *args: False)
        assert not vr.assemble_vr_frames(dirs, settings, total_frames=1)
    assert not (dirs["vr_frames"] / "metadata.json").exists()
    assert vr.assemble_vr_frames(dirs, settings, total_frames=1)
