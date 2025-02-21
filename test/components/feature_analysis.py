# test/components/feature_analysis.py

import cv2
import numpy as np
import librosa
import torch
from typing import Dict, Any, List
import logging
from pathlib import Path
import mediapipe as mp


class AdvancedFeatureExtractor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup_extractors()

    def setup_extractors(self):
        """Setup all feature extractors"""
        try:
            # Visual feature extractors
            self.sift = cv2.SIFT_create()
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            # Mediapipe face mesh for detailed facial analysis
            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            # Mediapipe pose detection
            self.mp_pose = mp.solutions.pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            # Eye cascade classifiers
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize feature extractors: {str(e)}")
            raise

    def extract_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extract comprehensive features from frame"""
        try:
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            features = {
                'visual': self._extract_visual_features(frame, gray),
                'facial': self._extract_facial_features(frame_rgb, gray),
                'pose': self._extract_pose_features(frame_rgb),
                'motion': self._extract_motion_features(gray),
                'engagement': self._analyze_engagement(frame_rgb)
            }

            return features

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            return {}

    def _extract_visual_features(self, frame: np.ndarray, gray: np.ndarray) -> Dict[str, Any]:
        """Extract visual features"""
        # SIFT features
        keypoints = self.sift.detect(gray, None)

        # Edge detection
        edges = cv2.Canny(gray, 100, 200)

        # Color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_stats = {
            'hue_mean': float(np.mean(hsv[:, :, 0])),
            'saturation_mean': float(np.mean(hsv[:, :, 1])),
            'value_mean': float(np.mean(hsv[:, :, 2]))
        }

        return {
            'keypoint_count': len(keypoints),
            'edge_density': float(np.sum(edges > 0)) / (edges.shape[0] * edges.shape[1]),
            'color_stats': color_stats
        }

    def _extract_facial_features(self, frame_rgb: np.ndarray, gray: np.ndarray) -> Dict[str, Any]:
        """Extract facial features"""
        # Face detection with confidence
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

        # Face mesh analysis
        face_mesh_results = self.mp_face_mesh.process(frame_rgb)

        facial_features = {
            'face_count': len(faces),
            'face_locations': [[int(x) for x in face] for face in faces],
            'face_landmarks': [],
            'eye_stats': [],
            'expression_analysis': {}
        }

        if face_mesh_results.multi_face_landmarks:
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                # Extract facial landmarks
                landmarks = [(int(l.x * frame_rgb.shape[1]), int(l.y * frame_rgb.shape[0]))
                             for l in face_landmarks.landmark]
                facial_features['face_landmarks'].append(landmarks)

                # Analyze facial expression
                facial_features['expression_analysis'] = self._analyze_expression(landmarks)

        # Eye detection
        eyes = self.eye_cascade.detectMultiScale(gray)
        facial_features['eye_stats'] = {
            'eye_count': len(eyes),
            'eye_locations': [[int(x) for x in eye] for eye in eyes]
        }

        return facial_features

    def _analyze_expression(self, landmarks: List[tuple]) -> Dict[str, float]:
        """Analyze facial expression from landmarks"""
        try:
            # Calculate various facial metrics
            return {
                'eye_aspect_ratio': self._calculate_eye_aspect_ratio(landmarks),
                'mouth_aspect_ratio': self._calculate_mouth_aspect_ratio(landmarks),
                'eyebrow_position': self._calculate_eyebrow_position(landmarks)
            }
        except Exception as e:
            self.logger.error(f"Expression analysis failed: {str(e)}")
            return {}

    def _extract_pose_features(self, frame_rgb: np.ndarray) -> Dict[str, Any]:
        """Extract pose features"""
        pose_results = self.mp_pose.process(frame_rgb)

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark

            # Calculate pose metrics
            pose_metrics = {
                'head_position': self._calculate_head_position(landmarks),
                'shoulder_alignment': self._calculate_shoulder_alignment(landmarks),
                'body_orientation': self._calculate_body_orientation(landmarks)
            }

            return pose_metrics

        return {}

    def _extract_motion_features(self, gray: np.ndarray) -> Dict[str, Any]:
        """Extract motion features"""
        if not hasattr(self, 'prev_gray'):
            self.prev_gray = gray
            return {}

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Update previous frame
        self.prev_gray = gray

        # Calculate motion metrics
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        return {
            'motion_magnitude': float(np.mean(magnitude)),
            'motion_direction': float(np.mean(angle)),
            'motion_stability': float(np.std(magnitude))
        }

    def _analyze_engagement(self, frame_rgb: np.ndarray) -> Dict[str, float]:
        """Analyze engagement metrics"""
        engagement_metrics = {
            'attention_score': 0.0,
            'interaction_level': 0.0,
            'confidence_score': 0.0
        }

        try:
            # Combine facial, pose, and motion features for engagement analysis
            face_results = self._extract_facial_features(frame_rgb, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY))
            pose_results = self._extract_pose_features(frame_rgb)

            if face_results.get('face_count', 0) > 0:
                # Calculate attention score based on eye contact and head position
                engagement_metrics['attention_score'] = self._calculate_attention_score(
                    face_results, pose_results
                )

                # Calculate interaction level based on facial expressions and motion
                engagement_metrics['interaction_level'] = self._calculate_interaction_level(
                    face_results
                )

                # Calculate confidence score based on pose and expression
                engagement_metrics['confidence_score'] = self._calculate_confidence_score(
                    face_results, pose_results
                )

        except Exception as e:
            self.logger.error(f"Engagement analysis failed: {str(e)}")

        return engagement_metrics

    def _calculate_attention_score(self, face_results: Dict[str, Any],
                                   pose_results: Dict[str, Any]) -> float:
        """Calculate attention score"""
        attention_score = 0.0

        # Factor in eye aspect ratio
        if 'expression_analysis' in face_results:
            eye_ratio = face_results['expression_analysis'].get('eye_aspect_ratio', 0)
            attention_score += eye_ratio * 0.4

        # Factor in head position
        if 'head_position' in pose_results:
            head_score = 1.0 - min(abs(pose_results['head_position']), 1.0)
            attention_score += head_score * 0.6

        return min(attention_score, 1.0)

    def _calculate_interaction_level(self, face_results: Dict[str, Any]) -> float:
        """Calculate interaction level"""
        interaction_score = 0.0

        if 'expression_analysis' in face_results:
            expr = face_results['expression_analysis']

            # Factor in mouth movement
            mouth_ratio = expr.get('mouth_aspect_ratio', 0)
            interaction_score += mouth_ratio * 0.5

            # Factor in eyebrow position
            eyebrow_pos = expr.get('eyebrow_position', 0)
            interaction_score += eyebrow_pos * 0.5

        return min(interaction_score, 1.0)

    def _calculate_confidence_score(self, face_results: Dict[str, Any],
                                    pose_results: Dict[str, Any]) -> float:
        """Calculate confidence score"""
        confidence_score = 0.0

        # Factor in shoulder alignment
        if 'shoulder_alignment' in pose_results:
            alignment_score = 1.0 - min(abs(pose_results['shoulder_alignment']), 1.0)
            confidence_score += alignment_score * 0.4

        # Factor in body orientation
        if 'body_orientation' in pose_results:
            orientation_score = 1.0 - min(abs(pose_results['body_orientation']), 1.0)
            confidence_score += orientation_score * 0.3

        # Factor in facial expression
        if 'expression_analysis' in face_results:
            expr = face_results['expression_analysis']
            expr_score = expr.get('mouth_aspect_ratio', 0) + expr.get('eyebrow_position', 0)
            confidence_score += min(expr_score / 2, 0.3)

        return min(confidence_score, 1.0)

    # Helper geometric calculation methods
    def _calculate_eye_aspect_ratio(self, landmarks: List[tuple]) -> float:
        """Calculate eye aspect ratio"""
        # Use specific landmark indices for eyes
        return 0.5  # Placeholder

    def _calculate_mouth_aspect_ratio(self, landmarks: List[tuple]) -> float:
        """Calculate mouth aspect ratio"""
        # Use specific landmark indices for mouth
        return 0.5  # Placeholder

    def _calculate_eyebrow_position(self, landmarks: List[tuple]) -> float:
        """Calculate eyebrow position"""
        # Use specific landmark indices for eyebrows
        return 0.5  # Placeholder

    def _calculate_head_position(self, landmarks: List) -> float:
        """Calculate head position"""
        # Use nose and eyes landmarks
        return 0.5  # Placeholder

    def _calculate_shoulder_alignment(self, landmarks: List) -> float:
        """Calculate shoulder alignment"""
        # Use shoulder landmarks
        return 0.5  # Placeholder

    def _calculate_body_orientation(self, landmarks: List) -> float:
        """Calculate body orientation"""
        # Use multiple body landmarks
        return 0.5  # Placeholder