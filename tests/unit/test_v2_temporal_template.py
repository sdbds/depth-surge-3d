"""Web contract tests for fixed shot-aware V2 temporal inference."""

from pathlib import Path


TEMPLATE = (Path(__file__).parents[2] / "templates" / "index.html").read_text(encoding="utf-8")


def test_web_does_not_offer_or_submit_ignored_v2_temporal_controls() -> None:
    assert 'id="temporalWindowSettings"' not in TEMPLATE
    assert 'id="temporalWindowSize"' not in TEMPLATE
    assert 'id="temporalWindowOverlap"' not in TEMPLATE
    assert "temporal_window_size:" not in TEMPLATE
    assert "temporal_window_overlap:" not in TEMPLATE


def test_web_describes_v2_consistency_as_shot_aware() -> None:
    assert "fixed 32-frame windows with 10-frame overlap" in TEMPLATE
    assert "within detected shots" in TEMPLATE
    assert "state resets at cuts" in TEMPLATE
