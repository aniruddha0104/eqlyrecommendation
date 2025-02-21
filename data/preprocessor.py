import cv2
import numpy as np
import torch
from typing import Tuple, List


class VideoPreprocessor:
    def __init__(self, frame_size: Tuple[int, int] = (224, 224), sample_rate: int = 30):
        self.frame_size = frame_size
        self.sample_rate = sample_rate

    def process_video(self, video_path: str) -> torch.Tensor:
        frames = self._extract_frames(video_path)
        processed_frames = [self._process_frame(frame) for frame in frames]
        return torch.tensor(np.array(processed_frames))

    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        frames = []
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % self.sample_rate == 0:
                frames.append(frame)
            frame_count += 1

        cap.release()
        return frames

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, self.frame_size)
        normalized = resized / 255.0
        return normalized

    def _extract_audio_features(self, video_path: str) -> np.ndarray:
        # TODO: Implement audio feature extraction
        return np.array([])
