"""Face analysis module for video assessment platform.

This module provides facial expression and eye contact analysis
for the video assessment platform.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FaceAnalysisError(Exception):
    """Custom exception for face analysis errors"""
    pass


class FaceAnalyzer:
    """Analyzer for facial expressions, eye contact and related metrics."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the face analyzer with optional configuration.

        Args:
            config: Optional configuration dictionary with parameters
        """
        self.config = {
            'min_confidence': 0.7,
            'face_detection_model': 'opencv',
            'emotion_threshold': 0.5,
            'eye_contact_threshold': 0.65,
            'enable_age_gender': False,
            'enable_race': False,
        }
        if config:
            self.config.update(config)

        self.initialize_detectors()

    def initialize_detectors(self):
        """Initialize face detection and analysis models."""
        try:
            # Initialize face detector
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            # Check if detector loaded successfully
            if self.face_detector.empty():
                raise FaceAnalysisError("Failed to load face detector model")

            logger.info("Face analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Face detector initialization failed: {str(e)}")
            raise FaceAnalysisError("Failed to initialize face analysis system")

    def analyze_face(self, frame: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive face analysis on a single frame.

        Args:
            frame: Input frame as numpy array (BGR format)

        Returns:
            Dictionary containing face analysis results
        """
        try:
            # Convert to RGB for DeepFace
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces for performance check
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            if len(faces) == 0:
                logger.debug("No faces detected in frame")
                return self.get_default_analysis()

            # Run parallel analysis for emotions and eye tracking
            with ThreadPoolExecutor(max_workers=2) as executor:
                emotion_future = executor.submit(self.analyze_emotions, rgb_frame)
                eye_contact_future = executor.submit(self.analyze_eye_contact, frame, faces)

                emotion_data = emotion_future.result()
                eye_contact_data = eye_contact_future.result()

            face_count = len(faces)
            face_area_ratio = self.calculate_face_area_ratio(faces, frame.shape)

            return {
                'emotion': emotion_data,
                'eye_contact': eye_contact_data,
                'face_count': face_count,
                'face_area_ratio': face_area_ratio,
                'confidence': self._get_confidence(emotion_data, eye_contact_data),
                'frame_has_face': True
            }

        except Exception as e:
            logger.error(f"Face analysis failed: {str(e)}")
            return self.get_default_analysis()

    def analyze_emotions(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze emotions in the given frame.

        Args:
            frame: RGB frame as numpy array

        Returns:
            Dictionary with emotion analysis results
        """
        try:
            result = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend=self.config['face_detection_model']
            )

            if isinstance(result, list):
                result = result[0]

            emotions = result.get('emotion', {})
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else 'unknown'

            # Calculate engagement based on emotions
            engagement_score = self._calculate_emotion_engagement(emotions)

            return {
                'dominant_emotion': dominant_emotion,
                'emotion_scores': emotions,
                'engagement_score': engagement_score,
                'confidence': result.get('confidence', 0.0) if isinstance(result, dict) else 0.0
            }

        except Exception as e:
            logger.error(f"Emotion analysis failed: {str(e)}")
            return {
                'dominant_emotion': 'unknown',
                'emotion_scores': {},
                'engagement_score': 0.0,
                'confidence': 0.0
            }

    def analyze_eye_contact(self, frame: np.ndarray, faces: List[Tuple[int, int, int, int]]) -> Dict[str, float]:
        """Analyze eye contact quality in the frame.

        Args:
            frame: BGR frame as numpy array
            faces: List of detected face rectangles (x, y, w, h)

        Returns:
            Dictionary with eye contact metrics
        """
        try:
            if not faces:
                return {'score': 0.0, 'confidence': 0.0, 'duration': 0.0}

            # Use the largest face for analysis
            main_face = max(faces, key=lambda face: face[2] * face[3])
            x, y, w, h = main_face

            # Extract face region
            face_region = frame[y:y + h, x:x + w]
            if face_region.size == 0:
                return {'score': 0.0, 'confidence': 0.0, 'duration': 0.0}

            # Convert to grayscale for eye detection
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

            # Detect eyes
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(
                gray_face,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(15, 15)
            )

            if len(eyes) < 2:
                return {'score': 0.0, 'confidence': 0.0, 'duration': 0.0}

            # Calculate eye position relative to face
            # Center eyes should indicate good eye contact
            eye_positions = []
            for (ex, ey, ew, eh) in eyes[:2]:  # Use at most 2 eyes
                eye_center_x = ex + ew / 2
                eye_center_y = ey + eh / 2
                # Normalize positions
                norm_x = eye_center_x / w
                norm_y = eye_center_y / h
                eye_positions.append((norm_x, norm_y))

            # Calculate eye contact score based on position
            # Perfect eye contact has eyes in the middle-upper part of face
            eye_contact_score = 0.0
            for norm_x, norm_y in eye_positions:
                # Ideal x: 0.5 (center), Ideal y: around 0.3-0.4 (upper third)
                x_score = 1.0 - abs(norm_x - 0.5) * 2  # 1.0 at center, 0.0 at edges
                y_score = 1.0 - abs(norm_y - 0.35) * 3  # 1.0 at ideal height

                # Clamp scores to [0,1]
                x_score = max(0.0, min(1.0, x_score))
                y_score = max(0.0, min(1.0, y_score))

                # Average the scores
                eye_contact_score += (x_score * 0.7 + y_score * 0.3)

            # Average across eyes
            eye_contact_score /= len(eye_positions) if eye_positions else 1

            # Confidence based on number of eyes detected
            confidence = min(1.0, len(eyes) / 2.0)

            return {
                'score': float(eye_contact_score),
                'confidence': float(confidence),
                'duration': 0.0  # To be updated by tracking over time
            }

        except Exception as e:
            logger.error(f"Eye contact analysis failed: {str(e)}")
            return {'score': 0.0, 'confidence': 0.0, 'duration': 0.0}

    def calculate_face_area_ratio(self, faces: List[Tuple[int, int, int, int]],
                                  frame_shape: Tuple[int, int, int]) -> float:
        """Calculate the ratio of face area to frame area.

        Args:
            faces: List of detected face rectangles (x, y, w, h)
            frame_shape: Shape of the frame (height, width, channels)

        Returns:
            Ratio of face area to frame area
        """
        if not faces:
            return 0.0

        frame_area = frame_shape[0] * frame_shape[1]

        # Calculate combined face area
        face_area = sum(w * h for _, _, w, h in faces)

        # Calculate ratio, capped at 1.0
        ratio = min(1.0, face_area / frame_area)

        return float(ratio)

    def _calculate_emotion_engagement(self, emotions: Dict[str, float]) -> float:
        """Calculate engagement score based on emotions.

        Args:
            emotions: Dictionary of emotion scores

        Returns:
            Engagement score between 0 and 1
        """
        if not emotions:
            return 0.0

        # Positive emotions contribute to higher engagement
        positive_emotions = ['happy', 'surprise']
        negative_emotions = ['angry', 'disgust', 'fear', 'sad']
        neutral_score = emotions.get('neutral', 0.0)

        # Calculate positive and negative contributions
        positive_score = sum(emotions.get(emotion, 0.0) for emotion in positive_emotions)
        negative_score = sum(emotions.get(emotion, 0.0) for emotion in negative_emotions)

        # Neutral is considered moderate engagement
        neutral_contribution = neutral_score * 0.5

        # Calculate engagement score (positive increases, negative decreases)
        engagement = positive_score + neutral_contribution - (negative_score * 0.5)

        # Normalize to [0,1]
        normalized_engagement = max(0.0, min(1.0, engagement / 100.0))

        return float(normalized_engagement)

    def _get_confidence(self, emotion_data: Dict[str, Any], eye_contact_data: Dict[str, float]) -> float:
        """Calculate overall confidence score.

        Args:
            emotion_data: Emotion analysis results
            eye_contact_data: Eye contact analysis results

        Returns:
            Overall confidence score
        """
        emotion_confidence = emotion_data.get('confidence', 0.0)
        eye_contact_confidence = eye_contact_data.get('confidence', 0.0)

        # Weighted average
        confidence = emotion_confidence * 0.6 + eye_contact_confidence * 0.4

        return float(confidence)

    def get_default_analysis(self) -> Dict[str, Any]:
        """Return default analysis result when face detection fails."""
        return {
            'emotion': {
                'dominant_emotion': 'unknown',
                'emotion_scores': {},
                'engagement_score': 0.0,
                'confidence': 0.0
            },
            'eye_contact': {
                'score': 0.0,
                'confidence': 0.0,
                'duration': 0.0
            },
            'face_count': 0,
            'face_area_ratio': 0.0,
            'confidence': 0.0,
            'frame_has_face': False
        }