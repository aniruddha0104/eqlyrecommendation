import cv2
import torch
import numpy as np
from typing import Dict, List
from transformers import AutoFeatureExtractor, AutoModelForVideoClassification


class VideoProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/videomae-base")
        self.model = AutoModelForVideoClassification.from_pretrained("microsoft/videomae-base").to(self.device)

    async def extract_features(self, video_path: str) -> Dict[str, torch.Tensor]:
        frames = self._extract_frames(video_path)
        visual_features = await self._process_frames(frames)
        motion_features = self._analyze_motion(frames)
        gestures = self._detect_gestures(frames)

        return {
            'visual': visual_features,
            'motion': motion_features,
            'gestures': gestures
        }

    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        frames = []
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        return frames