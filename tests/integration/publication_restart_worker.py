"""One encoder process for the Linux hard-exit/restart integration test."""

import json
import os
from pathlib import Path
import sys

import numpy as np

from depth_surge_3d.processing.frames.vr_assembler import DirectStereoSource
from depth_surge_3d.processing.video.final_publication import FinalPublicationTransactionV4
from depth_surge_3d.processing.video.video_encoder import VideoEncoder


def main():
    job, source = map(Path, sys.argv[1:3])
    count, mode, boundary = int(sys.argv[3]), sys.argv[4], sys.argv[5]
    settings = json.loads((job / "processing-settings-v5.json").read_bytes())["processing_settings"]

    if boundary:
        emit = FinalPublicationTransactionV4._emit

        def hard_exit_at(self, event):
            emit(self, event)
            if event == boundary:
                os._exit(86)  # Do not run the encoder's ordinary error cleanup.

        FinalPublicationTransactionV4._emit = hard_exit_at
        if mode == "streamed" and boundary == "encoded-output:synced":
            # The streamed writer has already synced its output evidence and moved its
            # working video into the indexed sibling when it enters publish().
            def exit_before_publish(self):
                os._exit(86)

            FinalPublicationTransactionV4.publish = exit_before_publish

    encoder = VideoEncoder()
    if mode == "streamed" and boundary:
        session = encoder.open_streamed_vr_session(
            output_dir=job,
            original_video=str(source),
            settings=settings,
            frame_count=count,
            first_index=1,
        )
        for i in range(1, count + 1):
            session.write(f"frame_{i:06d}", np.full((48, 128, 3), i * 35, np.uint8))
        success = session.finish_and_publish()
    elif mode == "assembled":
        success = encoder.create_video(job / "99_vr_frames", job, str(source), settings)
    else:
        success = encoder.create_video_from_stereo_source(
            DirectStereoSource(job / "06_left_cropped", job / "06_right_cropped", 1, count),
            job,
            str(source),
            settings,
            total_frames=count,
        )
    if success:
        encoder.pending_final_encoding_reservation.validate_authenticated_publication()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
