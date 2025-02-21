# test/video_utils/video_generator.py

import cv2
import numpy as np
from pathlib import Path
import logging


def generate_test_video(output_path: str, duration: int = 5, fps: int = 30):
    """Generate a test video with moving shapes and patterns"""
    logger = logging.getLogger(__name__)

    try:
        # Video settings
        frame_size = (640, 480)
        total_frames = duration * fps

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        # Generate frames
        for i in range(total_frames):
            # Create frame
            frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

            # Add moving circle
            radius = 50
            center_x = int(frame_size[0] / 2 + np.sin(i / 30) * 100)
            center_y = int(frame_size[1] / 2 + np.cos(i / 30) * 100)
            cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), -1)

            # Add rectangle
            rect_size = 80
            rect_x = int(frame_size[0] / 2 - rect_size / 2 + np.cos(i / 20) * 150)
            rect_y = int(frame_size[1] / 2 - rect_size / 2 + np.sin(i / 20) * 150)
            cv2.rectangle(frame, (rect_x, rect_y),
                          (rect_x + rect_size, rect_y + rect_size),
                          (255, 0, 0), -1)

            # Add text
            text = f"Frame: {i}/{total_frames}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2)

            # Write frame
            out.write(frame)

        # Release video writer
        out.release()
        logger.info(f"Test video generated: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to generate test video: {str(e)}")
        return False


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Generate test video
    video_path = "test_video.mp4"
    generate_test_video(video_path)