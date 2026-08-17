"""Regression tests for resume controls in the web interface."""

from pathlib import Path


TEMPLATE_PATH = Path(__file__).parents[2] / "templates" / "index.html"


def test_resumable_job_path_is_not_embedded_in_inline_javascript():
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert "onclick=\"resumeJob('${job.path}')\"" not in template
    assert "resumeButton.addEventListener('click', () => resumeJob(job));" in template


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
        "temporalPostprocessorOff",
        "temporalPostprocessorVdpp",
        "resumeTemporalPostprocessor",
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
        "temporal_postprocessor",
        "stereo_io_workers",
        "migrate_legacy",
    ):
        assert f"{setting_name}:" in template

    assert "baseline:" not in template
    assert "focal_length:" not in template
    assert "hole_fill_quality:" not in template


def test_vdpp_web_control_is_binary_and_exposes_no_internal_tuning_knobs():
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert 'name="temporalPostprocessor"' in template
    assert 'value="off"' in template
    assert 'value="vdpp"' in template
    assert "CUDA" in template
    assert "download" in template.lower()
    assert 'id="vdppWindowSize"' not in template
    assert 'id="vdppOverlap"' not in template


def test_resume_temporal_override_is_presence_aware():
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert '<option value="saved" selected>Use saved setting</option>' in template
    assert "if (resumeTemporalPostprocessor !== 'saved')" in template
    assert "resumePayload.temporal_postprocessor = resumeTemporalPostprocessor;" in template
    assert "resumeTemporalSelect.value = job.temporal_postprocessor;" in template
