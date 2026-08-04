"""Regression tests for resume controls in the web interface."""

from pathlib import Path


TEMPLATE_PATH = Path(__file__).parents[2] / "templates" / "index.html"


def test_resumable_job_path_is_not_embedded_in_inline_javascript():
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert "onclick=\"resumeJob('${job.path}')\"" not in template
    assert "resumeButton.addEventListener('click', () => resumeJob(job.path));" in template


def test_resumable_job_description_is_inserted_as_text_not_html():
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert '<span class="resumable-job-message"></span>' in template
    assert "messageElement.textContent = message;" in template
    assert "${message}" not in template


def test_web_form_exposes_final_depth_and_stereo_controls_only():
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    for element_id in (
        "stereoStrength",
        "convergence",
        "occlusionFill",
        "sceneDetection",
        "sceneCutThreshold",
        "minSceneFrames",
        "rawStorageDtype",
        "stereoIoWorkers",
        "migrateLegacy",
    ):
        assert f'id="{element_id}"' in template

    for removed_id in ("baseline", "focalLength", "holeFillQuality"):
        assert f'id="{removed_id}"' not in template


def test_web_payload_uses_final_depth_and_stereo_setting_names():
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    for setting_name in (
        "stereo_strength",
        "convergence",
        "occlusion_fill",
        "scene_detection",
        "scene_cut_threshold",
        "min_scene_frames",
        "raw_storage_dtype",
        "stereo_io_workers",
        "migrate_legacy",
    ):
        assert f"{setting_name}:" in template

    assert "baseline:" not in template
    assert "focal_length:" not in template
    assert "hole_fill_quality:" not in template
