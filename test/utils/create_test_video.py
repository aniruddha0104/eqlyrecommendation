# test/video_utils/create_test_video.py

import cv2
import numpy as np
from pathlib import Path
import logging
import os


def create_test_video():
    """Create a test video with various features"""
    # Get project root directory (2 levels up from video_utils)
    root_dir = Path(__file__).parent.parent.parent
    output_path = root_dir / 'test_data' / 'test_video.mp4'

    # Create directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Creating test video at: {output_path.absolute()}")

    # Video settings
    width, height = 640, 480
    fps = 30
    duration = 5  # seconds

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Create frames
    frames = duration * fps
    for i in range(frames):
        # Create frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Add moving shapes
        # Circle
        cx = int(width / 2 + 100 * np.sin(i * 2 * np.pi / frames))
        cy = int(height / 2 + 100 * np.cos(i * 2 * np.pi / frames))
        cv2.circle(frame, (cx, cy), 50, (0, 255, 0), -1)

        # Rectangle
        rx = int(width / 2 + 150 * np.cos(i * 2 * np.pi / frames))
        ry = int(height / 2)
        cv2.rectangle(frame, (rx - 30, ry - 30), (rx + 30, ry + 30), (0, 0, 255), -1)

        # Face-like pattern
        fx = int(width / 2)
        fy = int(height / 2)
        cv2.ellipse(frame, (fx, fy), (60, 80), 0, 0, 360, (255, 255, 255), -1)

        # Write frame
        out.write(frame)

    # Release everything
    out.release()
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    video_path = create_test_video()
    print(f"Test video created successfully at: {video_path.absolute()}")