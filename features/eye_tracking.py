# features/eye_tracking.py
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Any, Optional, Tuple


class EyeTracker:
    def __init__(self):
        # Initialize MediaPipe Face Mesh with specific configurations for better eye tracking
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,  # Set to False for video processing
            max_num_faces=1,  # We only need to track one face
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Define eye landmarks indices
        # These points are carefully selected to track eye movements
        self.LEFT_EYE_INDICES = [33, 133, 157, 158, 159, 160]  # Left eye perimeter
        self.RIGHT_EYE_INDICES = [362, 385, 386, 387, 388, 390]  # Right eye perimeter
        self.LEFT_IRIS = [468, 469, 470, 471, 472]  # Left iris landmarks
        self.RIGHT_IRIS = [473, 474, 475, 476, 477]  # Right iris landmarks

    def track_eyes(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame for eye tracking.

        Args:
            frame: BGR image from OpenCV
        Returns:
            Dictionary containing eye tracking metrics
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width = frame.shape[:2]

        # Process the frame
        results = self.mp_face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return self._empty_eye_result()

        face_landmarks = results.multi_face_landmarks[0]

        # Extract eye landmarks
        left_eye = self._get_eye_landmarks(face_landmarks, 'left')
        right_eye = self._get_eye_landmarks(face_landmarks, 'right')

        # Calculate metrics
        gaze_direction = self._calculate_gaze_direction(left_eye, right_eye)
        blink_state = self._detect_blink(left_eye, right_eye)
        eye_openness = self._calculate_eye_openness(left_eye, right_eye)

        return {
            'current_state': {
                'gaze_direction': gaze_direction,
                'blink_state': blink_state,
                'eye_openness': eye_openness,
                'attention_target': self._estimate_attention_target(gaze_direction)
            },
            'attention_metrics': self._calculate_attention_metrics(),
            'engagement_indicators': {
                'engagement_score': self._calculate_engagement_score(
                    gaze_direction, blink_state, eye_openness
                )
            },
            'teaching_effectiveness': {
                'effectiveness_score': self._calculate_teaching_effectiveness()
            }
        }

    def _get_eye_landmarks(self, landmarks, side: str) -> np.ndarray:
        """
        Extract eye landmarks from face landmarks.

        Args:
            landmarks: MediaPipe face landmarks
            side: 'left' or 'right'
        Returns:
            Array of eye landmark coordinates
        """
        indices = self.LEFT_EYE_INDICES if side == 'left' else self.RIGHT_EYE_INDICES
        points = []
        for idx in indices:
            point = landmarks.landmark[idx]
            points.append([point.x, point.y, point.z])
        return np.array(points)

    def _calculate_gaze_direction(self, left_eye: np.ndarray, right_eye: np.ndarray) -> Dict[str, float]:
        """
        Calculate gaze direction based on eye positions.

        Args:
            left_eye: Left eye landmarks
            right_eye: Right eye landmarks
        Returns:
            Dictionary containing horizontal and vertical gaze angles
        """
        # Calculate eye centers
        left_center = left_eye.mean(axis=0)
        right_center = right_eye.mean(axis=0)

        # Calculate gaze vector
        gaze_vector = right_center - left_center

        # Convert to angles
        horizontal_angle = np.arctan2(gaze_vector[0], gaze_vector[2])
        vertical_angle = np.arctan2(gaze_vector[1], gaze_vector[2])

        return {
            'horizontal': float(horizontal_angle),
            'vertical': float(vertical_angle),
            'confidence': self._calculate_gaze_confidence(left_eye, right_eye)
        }

    def _detect_blink(self, left_eye: np.ndarray, right_eye: np.ndarray) -> Dict[str, float]:
        """
        Detect blink state based on eye aspect ratio.
        """
        left_ear = self._calculate_eye_aspect_ratio(left_eye)
        right_ear = self._calculate_eye_aspect_ratio(right_eye)

        return {
            'is_blink': (left_ear + right_ear) / 2 < 0.2,
            'left_openness': left_ear,
            'right_openness': right_ear
        }

    def _calculate_eye_aspect_ratio(self, eye_points: np.ndarray) -> float:
        """
        Calculate eye aspect ratio for blink detection.
        """
        if len(eye_points) < 6:
            return 0.0

        # Vertical eye landmarks
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])

        # Horizontal eye landmarks
        h = np.linalg.norm(eye_points[0] - eye_points[3])

        # Calculate EAR
        return (v1 + v2) / (2.0 * h + 1e-6)

    def _calculate_eye_openness(self, left_eye: np.ndarray, right_eye: np.ndarray) -> float:
        """Calculate average eye openness."""
        left_ratio = self._calculate_eye_aspect_ratio(left_eye)
        right_ratio = self._calculate_eye_aspect_ratio(right_eye)
        return (left_ratio + right_ratio) / 2.0

    def _empty_eye_result(self) -> Dict[str, Any]:
        """Return default values when no face is detected."""
        return {
            'current_state': {
                'gaze_direction': {'horizontal': 0.0, 'vertical': 0.0, 'confidence': 0.0},
                'blink_state': {'is_blink': False, 'left_openness': 0.0, 'right_openness': 0.0},
                'attention_target': 'unknown',
                'eye_openness': 0.0
            },
            'attention_metrics': {'attention_score': 0.0},
            'engagement_indicators': {'engagement_score': 0.0},
            'teaching_effectiveness': {'effectiveness_score': 0.0}
        }

    # Add remaining utility methods...
    def _calculate_gaze_confidence(self, left_eye: np.ndarray, right_eye: np.ndarray) -> float:
        """Calculate confidence score for gaze detection."""
        return 1.0 - min(np.std(left_eye, axis=0).mean() + np.std(right_eye, axis=0).mean(), 1.0)

    def _estimate_attention_target(self, gaze_direction: Dict[str, float]) -> str:
        """Estimate where the person is looking based on gaze direction."""
        horizontal = abs(gaze_direction['horizontal'])
        vertical = abs(gaze_direction['vertical'])

        if horizontal < 0.2 and vertical < 0.2:
            return 'center'
        elif horizontal > vertical:
            return 'left' if gaze_direction['horizontal'] < 0 else 'right'
        else:
            return 'up' if gaze_direction['vertical'] < 0 else 'down'

    def _calculate_attention_metrics(self) -> Dict[str, float]:
        """Calculate attention-related metrics."""
        return {'attention_score': 0.5}  # Placeholder implementation

    def _calculate_engagement_score(
            self,
            gaze_direction: Dict[str, float],
            blink_state: Dict[str, float],
            eye_openness: float
    ) -> float:
        """Calculate overall engagement score."""
        return 0.5  # Placeholder implementation

    def _calculate_teaching_effectiveness(self) -> float:
        """Calculate teaching effectiveness score."""
        return 0.5  # Placeholder implementation