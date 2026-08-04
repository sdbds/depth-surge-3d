"""Tests for headset preset controls in the web interface."""

from pathlib import Path


TEMPLATE_PATH = Path(__file__).parents[2] / "templates" / "index.html"


def test_pimax_crystal_light_is_available_in_headset_selector():
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert '<option value="pimax-crystal-light">Pimax Crystal Light' in template
    assert "'pimax-crystal-light': { width: 2880, height: 2880, fov: 110 }" in template


def test_headset_presets_populate_existing_custom_resolution_fields():
    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    assert "document.getElementById('customWidth').value = settings.width" in template
    assert "document.getElementById('customHeight').value = settings.height" in template
    assert "document.getElementById('customResolution')" not in template
