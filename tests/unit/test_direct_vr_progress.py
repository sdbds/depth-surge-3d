"""Real web progress and stop-handler coverage for direct VR encoding."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import app as web_app


def test_direct_vr_progress_aliases_to_final_weighted_step_without_regressing():
    callback = web_app.ProgressCallback("test-session", total_frames=2)
    web_app.current_processing["stop_requested"] = False

    with (
        patch.object(web_app.socketio, "emit"),
        patch.object(web_app.socketio, "sleep"),
    ):
        callback.last_update_time = 0
        callback.update_progress(
            "Cropping frame 2/2",
            step_name="Crop Frames",
            step_progress=2,
            step_total=2,
        )
        crop_progress = web_app.current_processing["progress"]

        callback.last_update_time = 0
        callback.update_progress(
            "Encoding VR frame 1/2",
            phase="video_encoding",
            frame_num=1,
            step_name="Direct VR Encoding",
            step_progress=1,
            step_total=2,
        )

    assert len(callback.steps) == len(callback.step_weights) == 8
    assert callback.current_step_name == "Direct VR Encoding"
    assert callback.current_step_index == callback.steps.index("Video Creation") == 7
    assert web_app.current_processing["step_name"] == "Direct VR Encoding"
    assert web_app.current_processing["progress"] >= crop_progress


@pytest.mark.parametrize("source_kind", ["file", "array"])
def test_preview_downsampling_uses_inter_area(tmp_path, source_kind):
    callback = web_app.ProgressCallback(
        "test-session",
        total_frames=1,
        preview_update_interval=0,
    )
    callback.preview_downscale_width = 6
    frame = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    original_resize = web_app.cv2.resize

    with (
        patch.object(web_app.cv2, "resize", wraps=original_resize) as resize_spy,
        patch.object(web_app.socketio, "emit"),
    ):
        if source_kind == "file":
            frame_path = tmp_path / "frame.png"
            assert web_app.cv2.imwrite(str(frame_path), frame)
            callback.send_preview_frame(frame_path, "stereo_left", 1)
        else:
            callback.send_preview_frame_from_array(frame, "stereo_left", 1)

    assert resize_spy.called
    assert resize_spy.call_args.kwargs.get("interpolation") == web_app.cv2.INTER_AREA


def test_app_stop_handler_emits_stopped_instead_of_error(tmp_path):
    projector = MagicMock()
    projector.load_model.return_value = True
    processor = MagicMock()
    processor.process.side_effect = InterruptedError("Processing stopped by user request")

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch.object(web_app, "create_stereo_projector", return_value=projector),
        patch.object(
            web_app,
            "get_video_info",
            return_value={"fps": 24.0, "frame_count": 1, "width": 64, "height": 48},
        ),
        patch.object(web_app, "VideoProcessor", return_value=processor),
        patch.object(web_app.socketio, "emit") as emit,
        patch.object(web_app.socketio, "sleep"),
    ):
        web_app.process_video_async(
            "test-session",
            tmp_path / "source.mp4",
            {
                "device": "cpu",
                "vr_format": "side_by_side",
                "vr_resolution": "16x9-1080p",
            },
            tmp_path,
        )

    events = [call.args[0] for call in emit.call_args_list]
    assert "processing_stopped" in events
    assert "processing_error" not in events
    assert web_app.current_processing["active"] is False
    assert web_app.current_processing["stop_requested"] is False
