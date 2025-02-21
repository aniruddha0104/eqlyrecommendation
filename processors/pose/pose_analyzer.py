"""Pose analysis module for video assessment platform.

This module provides body language and gesture analysis capabilities
for the video assessment platform using MediaPipe pose estimation.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import time
import mediapipe as mp
from collections import deque
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PoseAnalysisError(Exception):
    """Custom exception for pose analysis errors"""
    pass


class GestureType(Enum):
    """Enum for different types of gestures"""
    POINTING = "pointing"
    HAND_WAVE = "hand_wave"
    HANDS_UP = "hands_up"
    ARMS_CROSSED = "arms_crossed"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class PostureType(Enum):
    """Enum for different types of postures"""
    STRAIGHT = "straight"
    LEANING_FORWARD = "leaning_forward"
    LEANING_BACK = "leaning_back"
    SLOUCHED = "slouched"
    SHIFTING = "shifting"
    UNKNOWN = "unknown"


class PoseAnalyzer:
    """Analyzer for body language, gestures and posture from video frames."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the pose analyzer with optional configuration.

        Args:
            config: Optional configuration dictionary with parameters
        """
        self.config = {
            'min_detection_confidence': 0.7,
            'min_tracking_confidence': 0.5,
            'model_complexity': 1,
            'gesture_detection_enabled': True,
            'posture_analysis_enabled': True,
            'motion_analysis_enabled': True,
            'history_size': 30,  # Frame history for motion analysis
            'confidence_threshold': 0.6,
            'hand_detection_enabled': False  # Enable if needed
        }
        if config:
            self.config.update(config)

        self.initialize_pose_detector()

        # Initialize state tracking
        self.pose_history = deque(maxlen=self.config['history_size'])
        self.gesture_history = deque(maxlen=10)  # Last 10 detected gestures
        self.posture_history = deque(maxlen=5)  # Last 5 detected postures
        self.current_gesture = GestureType.UNKNOWN
        self.current_posture = PostureType.UNKNOWN
        self.motion_level = 0.0
        self.stability_score = 0.0

    def initialize_pose_detector(self):
        """Initialize the MediaPipe pose detector."""
        try:
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.mp_pose = mp.solutions.pose

            # Initialize pose detector with specified configuration
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.config['model_complexity'],
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=self.config['min_detection_confidence'],
                min_tracking_confidence=self.config['min_tracking_confidence']
            )

            if self.config['hand_detection_enabled']:
                self.mp_hands = mp.solutions.hands
                self.hand_detector = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=self.config['min_detection_confidence'],
                    min_tracking_confidence=self.config['min_tracking_confidence']
                )

            logger.info("Pose detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pose detector: {str(e)}")
            raise PoseAnalysisError(f"Failed to initialize pose detector: {str(e)}")

    def analyze_pose(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze pose in a video frame.

        Args:
            frame: Input frame as numpy array (BGR format)

        Returns:
            Dictionary containing pose analysis results
        """
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with pose detector
            results = self.pose_detector.process(rgb_frame)

            # Initialize default result
            analysis_result = self._get_default_analysis()

            # If no pose detected, return default results
            if not results.pose_landmarks:
                return analysis_result

            # Extract and normalize landmarks
            landmarks = self._extract_normalized_landmarks(results.pose_landmarks, frame.shape)

            # Update pose history
            self.pose_history.append(landmarks)

            # Analyze different aspects of the pose
            confidence = self._calculate_confidence(results.pose_landmarks)

            if confidence >= self.config['confidence_threshold']:
                gesture = self._detect_gesture(landmarks) if self.config[
                    'gesture_detection_enabled'] else GestureType.UNKNOWN
                posture = self._analyze_posture(landmarks) if self.config[
                    'posture_analysis_enabled'] else PostureType.UNKNOWN
                motion_metrics = self._analyze_motion() if self.config['motion_analysis_enabled'] else {
                    'motion_level': 0.0, 'stability': 0.0}

                # Update state
                self.current_gesture = gesture
                self.current_posture = posture
                self.gesture_history.append(gesture)
                self.posture_history.append(posture)
                self.motion_level = motion_metrics['motion_level']
                self.stability_score = motion_metrics['stability']

                # Compile analysis result
                analysis_result = {
                    'landmarks': landmarks,
                    'confidence': confidence,
                    'gesture': {
                        'type': gesture.value,
                        'confidence': self._get_gesture_confidence(gesture),
                    },
                    'posture': {
                        'type': posture.value,
                        'confidence': self._get_posture_confidence(posture),
                    },
                    'motion': motion_metrics,
                    'engagement_indicators': self._calculate_engagement_indicators(landmarks, gesture, posture),
                    'has_pose': True
                }

            return analysis_result

        except Exception as e:
            logger.error(f"Pose analysis failed: {str(e)}")
            return self._get_default_analysis()

    def _extract_normalized_landmarks(self, pose_landmarks, frame_shape) -> List[List[float]]:
        """Extract and normalize landmarks from pose detection.

        Args:
            pose_landmarks: MediaPipe pose landmarks
            frame_shape: Shape of the input frame

        Returns:
            List of normalized landmark coordinates [x, y, z, visibility]
        """
        landmarks = []
        height, width = frame_shape[0], frame_shape[1]

        for idx, landmark in enumerate(pose_landmarks.landmark):
            # Normalize coordinates to frame dimensions
            x = landmark.x
            y = landmark.y
            z = landmark.z
            visibility = landmark.visibility

            landmarks.append([x, y, z, visibility])

        return landmarks

    def _calculate_confidence(self, pose_landmarks) -> float:
        """Calculate overall confidence of pose detection.

        Args:
            pose_landmarks: MediaPipe pose landmarks

        Returns:
            Confidence score between 0 and 1
        """
        # Average visibility of key landmarks
        key_indices = [
            0,  # nose
            11, 12,  # shoulders
            23, 24,  # hips
            13, 14,  # elbows
            15, 16  # wrists
        ]

        visibility_sum = sum(pose_landmarks.landmark[i].visibility for i in key_indices)
        confidence = visibility_sum / len(key_indices)

        return float(min(1.0, max(0.0, confidence)))

    def _detect_gesture(self, landmarks: List[List[float]]) -> GestureType:
        """Detect gesture from pose landmarks.

        Args:
            landmarks: Normalized pose landmarks

        Returns:
            Detected gesture type
        """
        # Extract key landmarks for gesture detection
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_elbow = landmarks[13]
        right_elbow = landmarks[14]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        left_hip = landmarks[23]
        right_hip = landmarks[24]

        # Check for hands up gesture
        if (left_wrist[1] < left_shoulder[1] - 0.1 and
                right_wrist[1] < right_shoulder[1] - 0.1):
            return GestureType.HANDS_UP

        # Check for arms crossed
        if (left_wrist[0] > nose[0] and right_wrist[0] < nose[0] and
                abs(left_wrist[1] - right_wrist[1]) < 0.1):
            return GestureType.ARMS_CROSSED

        # Check for pointing gesture (either hand)
        if (left_wrist[1] < left_elbow[1] or right_wrist[1] < right_elbow[1]):
            # Check if one arm is extended
            left_arm_extended = abs(left_wrist[0] - left_shoulder[0]) > 0.2
            right_arm_extended = abs(right_wrist[0] - right_shoulder[0]) > 0.2

            if left_arm_extended or right_arm_extended:
                return GestureType.POINTING

        # Check for hand wave
        if len(self.pose_history) >= 5:
            # Calculate wrist movement over last frames
            left_wrist_positions = [frame[15][0:2] for frame in self.pose_history[-5:]]
            right_wrist_positions = [frame[16][0:2] for frame in self.pose_history[-5:]]

            left_wrist_movement = self._calculate_position_variance(left_wrist_positions)
            right_wrist_movement = self._calculate_position_variance(right_wrist_positions)

            if (left_wrist_movement > 0.005 or right_wrist_movement > 0.005) and left_wrist[1] < left_shoulder[1]:
                return GestureType.HAND_WAVE

        # If no specific gesture detected, return neutral
        if all(lm[3] > 0.7 for lm in [left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist]):
            return GestureType.NEUTRAL

        return GestureType.UNKNOWN

    def _analyze_posture(self, landmarks: List[List[float]]) -> PostureType:
        """Analyze posture from pose landmarks.

        Args:
            landmarks: Normalized pose landmarks

        Returns:
            Detected posture type
        """
        # Extract key landmarks for posture analysis
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]

        # Calculate shoulder midpoint
        shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2
        shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2

        # Calculate hip midpoint
        hip_mid_x = (left_hip[0] + right_hip[0]) / 2
        hip_mid_y = (left_hip[1] + right_hip[1]) / 2

        # Calculate the angle of the spine (vertical alignment)
        dx = shoulder_mid_x - hip_mid_x
        dy = shoulder_mid_y - hip_mid_y
        spine_angle = np.arctan2(dx, dy) * 180 / np.pi

        # Calculate the distance of nose from shoulder midpoint (forward/backward lean)
        nose_offset_x = nose[0] - shoulder_mid_x

        # Check for shifting posture using history
        if len(self.pose_history) >= 10:
            hip_positions = [frame[24][0:2] for frame in self.pose_history[-10:]]
            hip_movement = self._calculate_position_variance(hip_positions)
            if hip_movement > 0.0008:  # Threshold for significant movement
                return PostureType.SHIFTING

        # Determine posture based on spine angle and nose position
        if abs(spine_angle) < 5:
            # Near vertical spine
            if nose_offset_x > 0.05:
                return PostureType.LEANING_FORWARD
            elif nose_offset_x < -0.05:
                return PostureType.LEANING_BACK
            else:
                return PostureType.STRAIGHT
        elif abs(spine_angle) < 15:
            # Slightly angled spine
            if spine_angle > 0:
                return PostureType.LEANING_FORWARD
            else:
                return PostureType.LEANING_BACK
        else:
            # Significantly angled spine - likely slouched
            return PostureType.SLOUCHED

    def _analyze_motion(self) -> Dict[str, float]:
        """Analyze motion patterns from pose history.

        Returns:
            Dictionary with motion metrics
        """
        if len(self.pose_history) < 5:
            return {'motion_level': 0.0, 'stability': 1.0}

        # Calculate motion by tracking key points across frames
        motion_points = [0, 11, 12, 13, 14, 15, 16]  # nose, shoulders, elbows, wrists

        # Calculate average motion
        total_motion = 0.0
        point_count = 0

        for point_idx in motion_points:
            point_positions = []
            for frame in self.pose_history[-5:]:
                # Skip if visibility is low
                if frame[point_idx][3] < 0.5:
                    continue
                point_positions.append(frame[point_idx][0:2])

            if len(point_positions) >= 3:
                variance = self._calculate_position_variance(point_positions)
                total_motion += variance
                point_count += 1

        # Normalize motion level to a 0-1 scale
        # Higher values indicate more movement
        if point_count > 0:
            avg_motion = total_motion / point_count
            motion_level = min(1.0, avg_motion * 100)
        else:
            motion_level = 0.0

        # Calculate stability (inverse of motion)
        stability = max(0.0, 1.0 - motion_level)

        return {
            'motion_level': float(motion_level),
            'stability': float(stability)
        }

    def _calculate_position_variance(self, positions: List[List[float]]) -> float:
        """Calculate variance of positions to measure movement.

        Args:
            positions: List of position coordinates

        Returns:
            Variance as a measure of movement
        """
        if len(positions) < 2:
            return 0.0

        # Convert to numpy array for easier calculation
        pos_array = np.array(positions)

        # Calculate variance in both x and y
        var_x = np.var(pos_array[:, 0])
        var_y = np.var(pos_array[:, 1])

        # Combined variance
        combined_var = var_x + var_y

        return float(combined_var)

    def _calculate_engagement_indicators(self, landmarks: List[List[float]],
                                         gesture: GestureType,
                                         posture: PostureType) -> Dict[str, float]:
        """Calculate engagement indicators from pose analysis.

        Args:
            landmarks: Normalized pose landmarks
            gesture: Detected gesture
            posture: Detected posture

        Returns:
            Dictionary with engagement metrics
        """
        # Base engagement score
        engagement_score = 0.5

        # Adjust based on gesture
        gesture_engagement = {
            GestureType.POINTING: 0.8,
            GestureType.HAND_WAVE: 0.7,
            GestureType.HANDS_UP: 0.6,
            GestureType.ARMS_CROSSED: 0.3,
            GestureType.NEUTRAL: 0.5,
            GestureType.UNKNOWN: 0.4
        }

        # Adjust based on posture
        posture_engagement = {
            PostureType.STRAIGHT: 0.8,
            PostureType.LEANING_FORWARD: 0.9,
            PostureType.LEANING_BACK: 0.4,
            PostureType.SLOUCHED: 0.2,
            PostureType.SHIFTING: 0.5,
            PostureType.UNKNOWN: 0.4
        }

        # Calculate final engagement score
        gesture_factor = gesture_engagement.get(gesture, 0.5)
        posture_factor = posture_engagement.get(posture, 0.5)

        # Weighted combination
        engagement_score = gesture_factor * 0.4 + posture_factor * 0.6

        # Additional factor: stability - too much movement can be distracting
        # but some movement is good for engagement
        optimal_motion = 0.3  # Some movement is good
        motion_factor = 1.0 - abs(self.motion_level - optimal_motion)

        # Calculate teaching confidence from posture and stability
        teaching_confidence = posture_factor * 0.7 + motion_factor * 0.3

        # Calculate expressiveness from gesture and motion
        expressiveness = gesture_factor * 0.6 + self.motion_level * 0.4

        return {
            'engagement_score': float(engagement_score),
            'teaching_confidence': float(teaching_confidence),
            'expressiveness': float(expressiveness),
            'motion_appropriateness': float(motion_factor)
        }

    def _get_gesture_confidence(self, gesture: GestureType) -> float:
        """Calculate confidence in gesture detection.

        Args:
            gesture: Detected gesture

        Returns:
            Confidence score for the gesture
        """
        # Higher confidence if same gesture detected multiple times
        if len(self.gesture_history) < 3:
            return 0.5

        # Count occurrences of current gesture in history
        recent_gestures = list(self.gesture_history)
        gesture_count = recent_gestures.count(gesture)

        # Calculate confidence based on consistency
        confidence = gesture_count / len(recent_gestures)

        return float(confidence)

    def _get_posture_confidence(self, posture: PostureType) -> float:
        """Calculate confidence in posture detection.

        Args:
            posture: Detected posture

        Returns:
            Confidence score for the posture
        """
        # Higher confidence if same posture detected multiple times
        if len(self.posture_history) < 3:
            return 0.6

        # Count occurrences of current posture in history
        recent_postures = list(self.posture_history)
        posture_count = recent_postures.count(posture)

        # Calculate confidence based on consistency
        confidence = 0.6 + (posture_count / len(recent_postures)) * 0.4

        return float(confidence)

    def _get_default_analysis(self) -> Dict[str, Any]:
        """Get default analysis result when no pose is detected."""
        return {
            'landmarks': [],
            'confidence': 0.0,
            'gesture': {
                'type': GestureType.UNKNOWN.value,
                'confidence': 0.0,
            },
            'posture': {
                'type': PostureType.UNKNOWN.value,
                'confidence': 0.0,
            },
            'motion': {
                'motion_level': 0.0,
                'stability': 0.0
            },
            'engagement_indicators': {
                'engagement_score': 0.0,
                'teaching_confidence': 0.0,
                'expressiveness': 0.0,
                'motion_appropriateness': 0.0
            },
            'has_pose': False
        }

    def visualize_pose(self, frame: np.ndarray, analysis_result: Dict[str, Any]) -> np.ndarray:
        """Visualize pose landmarks and analysis on frame.

        Args:
            frame: Input frame
            analysis_result: Analysis result from analyze_pose

        Returns:
            Frame with visualization overlaid
        """
        if not analysis_result['has_pose'] or not analysis_result['landmarks']:
            return frame

        # Draw pose landmarks
        mp_drawing = self.mp_drawing
        mp_pose = self.mp_pose

        # Create a copy to avoid modifying the original
        output_frame = frame.copy()

        # Convert landmarks to MediaPipe format
        landmarks_proto = mp_pose.PoseLandmark()
        landmarks = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()

        for idx, lm in enumerate(analysis_result['landmarks']):
            landmark = landmarks.landmark.add()
            landmark.x = lm[0]
            landmark.y = lm[1]
            landmark.z = lm[2]
            landmark.visibility = lm[3]

        # Draw the pose landmarks
        mp_drawing.draw_landmarks(
            output_frame,
            landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # Add text for gesture and posture
        h, w, _ = output_frame.shape

        # Draw gesture info
        gesture_text = f"Gesture: {analysis_result['gesture']['type']} ({analysis_result['gesture']['confidence']:.2f})"
        cv2.putText(output_frame, gesture_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw posture info
        posture_text = f"Posture: {analysis_result['posture']['type']} ({analysis_result['posture']['confidence']:.2f})"
        cv2.putText(output_frame, posture_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw engagement score
        engagement_text = f"Engagement: {analysis_result['engagement_indicators']['engagement_score']:.2f}"
        cv2.putText(output_frame, engagement_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return output_frame

    async def analyze_frames(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """
        Analyze body language and gestures across multiple frames.

        Args:
            frames: List of video frames as numpy arrays

        Returns:
            Dictionary containing aggregated pose analysis metrics
        """
        logger.info(f"Analyzing pose metrics across {len(frames)} frames")

        if not frames:
            logger.warning("No frames provided for pose analysis")
            return self._get_empty_metrics()

        try:
            # Sample frames if there are too many (process every Nth frame)
            sample_rate = max(1, len(frames) // 30)
            sampled_frames = frames[::sample_rate]

            # Analyze each frame
            frame_results = []
            for frame in sampled_frames:
                result = self.analyze_pose(frame)
                if result['has_pose']:
                    frame_results.append(result)

            # If no valid pose detected in any frame, return empty metrics
            if not frame_results:
                logger.warning("No valid pose detected in any frame")
                return self._get_empty_metrics()

            # Aggregate metrics across frames
            aggregated_metrics = self._aggregate_frame_results(frame_results)

            # Calculate teaching-specific metrics
            teaching_metrics = self._calculate_teaching_metrics(frame_results)

            # Combine all metrics
            final_metrics = {
                **aggregated_metrics,
                **teaching_metrics
            }

            logger.info(f"Pose analysis complete. Gesture effectiveness: {final_metrics['gesture_effectiveness']:.2f}, "
                        f"Posture confidence: {final_metrics['posture_confidence']:.2f}")

            return final_metrics

        except Exception as e:
            logger.error(f"Error in pose frame analysis: {str(e)}")
            return self._get_empty_metrics()

    def _aggregate_frame_results(self, frame_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Aggregate pose analysis results across multiple frames.

        Args:
            frame_results: List of per-frame analysis results

        Returns:
            Dictionary with aggregated metrics
        """
        # Count occurrences of each gesture and posture type
        gesture_counts = {}
        posture_counts = {}

        # Track engagement metrics across frames
        engagement_scores = []
        teaching_confidence_scores = []
        expressiveness_scores = []
        motion_appropriateness_scores = []
        confidence_scores = []

        # Collect metrics from each frame
        for result in frame_results:
            # Count gestures and postures
            gesture_type = result['gesture']['type']
            posture_type = result['posture']['type']

            gesture_counts[gesture_type] = gesture_counts.get(gesture_type, 0) + 1
            posture_counts[posture_type] = posture_counts.get(posture_type, 0) + 1

            # Collect engagement metrics
            engagement_scores.append(result['engagement_indicators']['engagement_score'])
            teaching_confidence_scores.append(result['engagement_indicators']['teaching_confidence'])
            expressiveness_scores.append(result['engagement_indicators']['expressiveness'])
            motion_appropriateness_scores.append(result['engagement_indicators']['motion_appropriateness'])
            confidence_scores.append(result['confidence'])

        # Find most common gesture and posture
        dominant_gesture = max(gesture_counts.items(), key=lambda x: x[1])[
            0] if gesture_counts else GestureType.UNKNOWN.value
        dominant_posture = max(posture_counts.items(), key=lambda x: x[1])[
            0] if posture_counts else PostureType.UNKNOWN.value

        # Calculate gesture diversity (number of different gestures used)
        gesture_diversity = len([g for g, count in gesture_counts.items()
                                 if g not in [GestureType.UNKNOWN.value] and count > len(frame_results) * 0.05])

        # Calculate frequency of positive/effective gestures
        positive_gestures = [GestureType.POINTING.value, GestureType.HAND_WAVE.value]
        negative_gestures = [GestureType.ARMS_CROSSED.value]

        positive_gesture_count = sum(gesture_counts.get(g, 0) for g in positive_gestures)
        negative_gesture_count = sum(gesture_counts.get(g, 0) for g in negative_gestures)
        total_gestures = sum(gesture_counts.values())

        positive_gesture_ratio = positive_gesture_count / total_gestures if total_gestures > 0 else 0
        negative_gesture_ratio = negative_gesture_count / total_gestures if total_gestures > 0 else 0

        # Calculate posture stability (consistency of posture)
        total_postures = sum(posture_counts.values())
        dominant_posture_ratio = posture_counts.get(dominant_posture, 0) / total_postures if total_postures > 0 else 0
        posture_stability = min(1.0, dominant_posture_ratio * 1.5)  # Scale up, max at 1.0

        # Calculate average metrics
        avg_engagement = sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0
        avg_teaching_confidence = sum(teaching_confidence_scores) / len(
            teaching_confidence_scores) if teaching_confidence_scores else 0
        avg_expressiveness = sum(expressiveness_scores) / len(expressiveness_scores) if expressiveness_scores else 0
        avg_motion_appropriateness = sum(motion_appropriateness_scores) / len(
            motion_appropriateness_scores) if motion_appropriateness_scores else 0
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

        return {
            'dominant_gesture': dominant_gesture,
            'dominant_posture': dominant_posture,
            'gesture_diversity': float(gesture_diversity),
            'positive_gesture_ratio': float(positive_gesture_ratio),
            'negative_gesture_ratio': float(negative_gesture_ratio),
            'posture_stability': float(posture_stability),
            'average_engagement': float(avg_engagement),
            'average_confidence': float(avg_confidence),
            'average_expressiveness': float(avg_expressiveness),
            'average_motion_appropriateness': float(avg_motion_appropriateness)
        }

    def _calculate_teaching_metrics(self, frame_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate teaching-specific metrics from pose analysis.

        Args:
            frame_results: List of per-frame analysis results

        Returns:
            Dictionary with teaching-specific metrics
        """
        # Extract key metrics from aggregated results
        aggregated = self._aggregate_frame_results(frame_results)

        # Calculate gesture effectiveness
        # Positive factors: diversity, positive_ratio, expressiveness
        # Negative factors: negative_ratio
        gesture_effectiveness = (
                0.3 * min(1.0, aggregated['gesture_diversity'] / 3) +  # Ideal: using at least 3 different gestures
                0.4 * aggregated['positive_gesture_ratio'] +
                0.3 * aggregated['average_expressiveness'] -
                0.2 * aggregated['negative_gesture_ratio']  # Penalty for negative gestures
        )
        gesture_effectiveness = max(0.0, min(1.0, gesture_effectiveness))

        # Calculate posture confidence
        # Good teaching posture: STRAIGHT or LEANING_FORWARD,
        # stability is important
        posture_score = 0.0
        if aggregated['dominant_posture'] == PostureType.STRAIGHT.value:
            posture_score = 0.9
        elif aggregated['dominant_posture'] == PostureType.LEANING_FORWARD.value:
            posture_score = 1.0  # Optimal engaged posture
        elif aggregated['dominant_posture'] == PostureType.SHIFTING.value:
            posture_score = 0.6  # Some movement is okay
        elif aggregated['dominant_posture'] == PostureType.LEANING_BACK.value:
            posture_score = 0.4
        elif aggregated['dominant_posture'] == PostureType.SLOUCHED.value:
            posture_score = 0.2
        else:  # UNKNOWN
            posture_score = 0.3

        posture_confidence = (
                0.7 * posture_score +
                0.3 * aggregated['posture_stability']
        )

        # Calculate movement appropriateness
        # Some movement is good, but too much can be distracting
        movement_appropriateness = aggregated['average_motion_appropriateness']

        # Overall teaching body language score
        teaching_body_language = (
                0.4 * gesture_effectiveness +
                0.4 * posture_confidence +
                0.2 * movement_appropriateness
        )

        return {
            'gesture_effectiveness': float(gesture_effectiveness),
            'posture_confidence': float(posture_confidence),
            'movement_appropriateness': float(movement_appropriateness),
            'teaching_body_language_score': float(teaching_body_language)
        }

    def _get_empty_metrics(self) -> Dict[str, float]:
        """
        Get empty metrics when analysis fails or no pose is detected.

        Returns:
            Dictionary with zero values for all metrics
        """
        return {
            'dominant_gesture': GestureType.UNKNOWN.value,
            'dominant_posture': PostureType.UNKNOWN.value,
            'gesture_diversity': 0.0,
            'positive_gesture_ratio': 0.0,
            'negative_gesture_ratio': 0.0,
            'posture_stability': 0.0,
            'average_engagement': 0.0,
            'average_confidence': 0.0,
            'average_expressiveness': 0.0,
            'average_motion_appropriateness': 0.0,
            'gesture_effectiveness': 0.0,
            'posture_confidence': 0.0,
            'movement_appropriateness': 0.0,
            'teaching_body_language_score': 0.0
        }