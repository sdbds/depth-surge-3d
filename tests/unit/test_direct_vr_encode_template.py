from pathlib import Path


TEMPLATE = Path("templates/index.html").read_text(encoding="utf-8")


def test_direct_vr_encode_checkbox_defaults_off():
    assert 'id="directVrEncode"' in TEMPLATE
    checkbox = TEMPLATE.split('id="directVrEncode"', 1)[1].split(">", 1)[0]
    assert "checked" not in checkbox
    assert "Direct FFmpeg VR Encoding" in TEMPLATE


def test_direct_vr_encode_is_sent_and_persisted():
    assert "direct_vr_encode: document.getElementById('directVrEncode').checked" in TEMPLATE
    assert "directVrEncode: document.getElementById('directVrEncode').checked" in TEMPLATE
    assert "directVrEncode: false" in TEMPLATE
    assert "'directVrEncode'" in TEMPLATE


def test_direct_vr_encode_does_not_force_browser_settings_reset():
    assert TEMPLATE.count("const SETTINGS_VERSION = 3") == 2
