import cv2
import numpy as np
from fer import FER
import logging
from typing import Dict, Optional
from pathlib import Path
from collections import deque

class EmotionDetector:
    def __init__(self, config: Dict = None):
        """
        Initialize the EmotionDetector with optional configuration.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO)
        self.config = config or {}
        try:
            # Initialize the FER detector with MTCNN or alternative face detector
            self.emotion_detector = FER(mtcnn=True)
            self.last_emotions = deque(maxlen=5)  # Store last N emotions for smoothing
            self.debug_dir = Path(self.config.get('debug_dir', 'debug_output/emotions'))
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            self.frame_count = 0
        except Exception as e:
            self.logger.error(f"Emotion detector initialization failed: {str(e)}")
            raise

    def detect_emotions(self, frame: np.ndarray) -> Dict:
        """
        Detect emotions in a single frame.
        Returns a dictionary containing dominant emotion, confidence, and bounding box.
        """
        try:
            # Convert frame to RGB for processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize frame for faster processing
            resized_frame = self._resize_frame(rgb_frame, target_size=(480, 640))
            # Detect emotions
            emotions = self.emotion_detector.detect_emotions(resized_frame)
            if not emotions:
                return self._get_default_emotion()
            # Process detected emotions
            emotion_data = emotions[0]
            dominant_emotion, confidence = max(emotion_data['emotions'].items(), key=lambda x: x[1])
            result = {
                'dominant_emotion': dominant_emotion,
                'emotion_scores': emotion_data['emotions'],
                'confidence': confidence,
                'box': emotion_data.get('box', None)
            }
            # Smooth emotions using a moving average
            self._smooth_emotions(result)
            return self.last_emotions[-1]  # Return the latest smoothed emotion
        except Exception as e:
            self.logger.error(f"Emotion detection failed: {str(e)}")
            return self._get_default_emotion()

    def _resize_frame(self, frame: np.ndarray, target_size: tuple) -> np.ndarray:
        """
        Resize the frame to a target size for faster processing.
        """
        return cv2.resize(frame, target_size)

    def _smooth_emotions(self, current_emotion: Dict) -> None:
        """
        Smooth emotions using a moving average over the last N detections.
        """
        self.last_emotions.append(current_emotion)
        if len(self.last_emotions) > 1:
            avg_confidence = np.mean([e['confidence'] for e in self.last_emotions])
            dominant_emotions = [e['dominant_emotion'] for e in self.last_emotions]
            most_frequent_emotion = max(set(dominant_emotions), key=dominant_emotions.count)
            self.last_emotions[-1]['confidence'] = avg_confidence
            self.last_emotions[-1]['dominant_emotion'] = most_frequent_emotion

    def _get_default_emotion(self) -> Dict:
        """
        Return a default emotion dictionary when no emotions are detected.
        """
        return {
            'dominant_emotion': self.last_emotions[-1]['dominant_emotion']
            if self.last_emotions else 'neutral',
            'emotion_scores': {},
            'confidence': 0.0,
            'box': None
        }

    def save_debug_frame(self, frame: np.ndarray, emotion_data: Dict) -> None:
        """
        Save the frame with emotion overlay for debugging purposes.
        """
        try:
            if not self.config.get('save_debug', False):
                return
            debug_frame = frame.copy()
            if emotion_data['box']:
                x, y, w, h = emotion_data['box']
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = f"{emotion_data['dominant_emotion']}: {emotion_data['confidence']:.2f}"
                cv2.putText(debug_frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            debug_path = self.debug_dir / f"emotion_frame_{self.frame_count:04d}.jpg"
            cv2.imwrite(str(debug_path), debug_frame)
            self.frame_count += 1
        except Exception as e:
            self.logger.error(f"Saving debug frame failed: {str(e)}")

# Example usage
if __name__ == "__main__":
    config = {
        'debug_dir': 'debug_output/emotions',
        'save_debug': True
    }
    detector = EmotionDetector(config=config)
    cap = cv2.VideoCapture(0)  # Open webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        emotion_data = detector.detect_emotions(frame)
        detector.save_debug_frame(frame, emotion_data)
        # Display the dominant emotion on the frame
        cv2.putText(frame, emotion_data['dominant_emotion'], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()