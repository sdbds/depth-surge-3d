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
