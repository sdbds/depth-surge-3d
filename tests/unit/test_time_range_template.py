from pathlib import Path


TEMPLATE = Path("templates/index.html").read_text(encoding="utf-8")


def test_time_range_controls_advertise_millisecond_precision() -> None:
    assert "Start Time (mm:ss.SSS or hh:mm:ss.SSS)" in TEMPLATE
    assert "End Time (mm:ss.SSS or hh:mm:ss.SSS)" in TEMPLATE
    assert 'id="startTime" placeholder="00:00.000"' in TEMPLATE
    assert 'id="endTime" placeholder="Video end"' in TEMPLATE


def test_blank_end_time_means_the_full_source_extent() -> None:
    assert TEMPLATE.count("function updateRangeInfo()") == 1
    assert "const endSeconds = parsedEnd ?? currentVideoInfo.duration;" in TEMPLATE
    assert "parsedEnd === null\n                ? totalFrames" in TEMPLATE
    assert "document.getElementById('endTime').value = '';" in TEMPLATE


def test_range_preview_matches_source_frame_selection() -> None:
    assert "Math.floor(startSeconds * fps)" in TEMPLATE
    assert "Math.floor(endSeconds * fps)" in TEMPLATE
    assert "const estimatedFrames = endFrame - startFrame;" in TEMPLATE
    assert "duration.toFixed(3)" in TEMPLATE
