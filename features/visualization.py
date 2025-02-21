# features/visualization.py
import cv2
import numpy as np
from typing import Dict, List


class DebugVisualizer:
    def __init__(self):
        self.colors = {
            'face': (0, 255, 0),
            'eyes': (255, 0, 0),
            'mouth': (0, 0, 255),
            'text': (255, 255, 255)
        }

    def draw_landmarks(self, frame: np.ndarray, landmarks, type='face') -> np.ndarray:
        debug_frame = frame.copy()
        if landmarks:
            h, w = frame.shape[:2]
            for landmark in landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(debug_frame, (x, y), 2, self.colors[type], -1)
        return debug_frame

    def draw_metrics(self, frame: np.ndarray, metrics: Dict) -> np.ndarray:
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX

        for category, values in metrics.items():
            if isinstance(values, dict):
                cv2.putText(frame, f"{category}:", (10, y_offset),
                            font, 0.7, self.colors['text'], 2)
                y_offset += 25

                for key, value in values.items():
                    if isinstance(value, (int, float)):
                        text = f"  {key}: {value:.2f}"
                        cv2.putText(frame, text, (10, y_offset),
                                    font, 0.5, self.colors['text'], 1)
                        y_offset += 20
            y_offset += 10

        return frame

    def draw_attention_target(self, frame: np.ndarray,
                              gaze_direction: Dict[str, float]) -> np.ndarray:
        h, w = frame.shape[:2]
        center_x = w // 2
        center_y = h // 2

        # Draw gaze direction
        if gaze_direction['confidence'] > 0:
            end_x = int(center_x + gaze_direction['horizontal'] * 100)
            end_y = int(center_y + gaze_direction['vertical'] * 100)
            cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y),
                            self.colors['eyes'], 2)