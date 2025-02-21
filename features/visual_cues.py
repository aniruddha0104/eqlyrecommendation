# eqly_assessment/features/visual_cues.py
import torch
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Any
import cv2


class VisualCuesAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh()
        self.mp_pose = mp.solutions.pose.Pose()
        self.mp_hands = mp.solutions.hands.Hands()

        # Landmark indices
        self.MOUTH_INDICES = [61, 291]  # Mouth corners
        self.EYEBROW_INDICES = [70, 300]  # Eyebrows
        self.EYE_INDICES = [33, 133, 362, 385]  # Eyes
        self.SHOULDER_INDICES = [11, 12]  # Shoulders
        self.SPINE_INDICES = [11, 12, 23, 24]  # Spine points

    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_landmarks = self._get_face_landmarks(rgb_frame)
        pose_landmarks = self._get_pose_landmarks(rgb_frame)
        hand_landmarks = self._get_hand_landmarks(rgb_frame)

        return {
            'facial_expressions': self._analyze_facial_expressions(face_landmarks),
            'eye_contact': self._analyze_eye_contact(face_landmarks),
            'body_language': self._analyze_body_language(pose_landmarks),
            'gestures': self._analyze_gestures(hand_landmarks, pose_landmarks),
            'engagement_signals': self._analyze_engagement(face_landmarks, pose_landmarks, hand_landmarks)
        }

    def _get_face_landmarks(self, frame: np.ndarray) -> List:
        results = self.mp_face_mesh.process(frame)
        return results.multi_face_landmarks[0] if results.multi_face_landmarks else None

    def _get_pose_landmarks(self, frame: np.ndarray) -> List:
        results = self.mp_pose.process(frame)
        return results.pose_landmarks if results.pose_landmarks else None

    def _get_hand_landmarks(self, frame: np.ndarray) -> List:
        results = self.mp_hands.process(frame)
        return results.multi_hand_landmarks if results.multi_hand_landmarks else None

    def _get_points(self, landmarks: List, indices: List) -> np.ndarray:
        return np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y, landmarks.landmark[idx].z)
                         for idx in indices])

    def _calculate_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return np.linalg.norm(p1 - p2)

    def _analyze_facial_expressions(self, landmarks: List) -> Dict[str, float]:
        if not landmarks:
            return {'neutral': 1.0}

        mouth_points = self._get_points(landmarks, self.MOUTH_INDICES)
        eye_points = self._get_points(landmarks, self.EYE_INDICES)
        eyebrow_points = self._get_points(landmarks, self.EYEBROW_INDICES)

        smile_ratio = self._calculate_distance(mouth_points[0], mouth_points[1])
        eye_ratio = np.mean([self._calculate_distance(eye_points[i], eye_points[i + 1])
                             for i in range(0, len(eye_points), 2)])
        eyebrow_ratio = self._calculate_distance(eyebrow_points[0], eyebrow_points[1])

        return {
            'smile': min(smile_ratio * 2, 1.0),
            'eye_openness': min(eye_ratio * 3, 1.0),
            'eyebrow_raise': min(eyebrow_ratio * 2, 1.0)
        }

    def _analyze_eye_contact(self, landmarks: List) -> Dict[str, float]:
        if not landmarks:
            return {'eye_contact_score': 0.0}

        eye_points = self._get_points(landmarks, self.EYE_INDICES)
        left_eye_center = np.mean(eye_points[0:2], axis=0)
        right_eye_center = np.mean(eye_points[2:4], axis=0)

        gaze_direction = right_eye_center - left_eye_center
        gaze_angle = np.arctan2(gaze_direction[1], gaze_direction[0])

        contact_score = 1.0 - min(abs(gaze_angle) / np.pi, 1.0)

        return {
            'eye_contact_score': contact_score,
            'gaze_angle': float(gaze_angle),
            'gaze_stability': self._calculate_gaze_stability(eye_points)
        }

    def _analyze_body_language(self, landmarks: List) -> Dict[str, float]:
        if not landmarks:
            return {'confidence_score': 0.0}

        shoulder_points = self._get_points(landmarks, self.SHOULDER_INDICES)
        spine_points = self._get_points(landmarks, self.SPINE_INDICES)

        posture_score = self._calculate_posture_score(spine_points)
        shoulder_alignment = self._calculate_shoulder_alignment(shoulder_points)

        return {
            'posture_score': posture_score,
            'shoulder_alignment': shoulder_alignment,
            'confidence_score': (posture_score + shoulder_alignment) / 2
        }

    def _analyze_gestures(self, hand_landmarks: List, pose_landmarks: List) -> Dict[str, float]:
        if not hand_landmarks or not pose_landmarks:
            return {'gesture_score': 0.0}

        hand_positions = [np.array([lm.x, lm.y, lm.z]) for lm in hand_landmarks[0].landmark]
        gesture_speed = self._calculate_gesture_speed(hand_positions)
        gesture_range = self._calculate_gesture_range(hand_positions)

        return {
            'gesture_speed': gesture_speed,
            'gesture_range': gesture_range,
            'expressiveness': (gesture_speed + gesture_range) / 2
        }

    def _analyze_engagement(self, face_landmarks: List, pose_landmarks: List, hand_landmarks: List) -> Dict[str, float]:
        facial_score = 0.0
        if face_landmarks:
            expressions = self._analyze_facial_expressions(face_landmarks)
            facial_score = np.mean(list(expressions.values()))

        posture_score = 0.0
        if pose_landmarks:
            body_metrics = self._analyze_body_language(pose_landmarks)
            posture_score = body_metrics['confidence_score']

        gesture_score = 0.0
        if hand_landmarks and pose_landmarks:
            gesture_metrics = self._analyze_gestures(hand_landmarks, pose_landmarks)
            gesture_score = gesture_metrics['expressiveness']

        overall_engagement = (facial_score + posture_score + gesture_score) / 3

        return {
            'facial_engagement': facial_score,
            'posture_engagement': posture_score,
            'gesture_engagement': gesture_score,
            'overall_engagement': overall_engagement
        }

    def _calculate_posture_score(self, spine_points: np.ndarray) -> float:
        spine_angles = []
        for i in range(len(spine_points) - 2):
            v1 = spine_points[i + 1] - spine_points[i]
            v2 = spine_points[i + 2] - spine_points[i + 1]
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            spine_angles.append(angle)
        return 1.0 - min(np.mean(spine_angles) / np.pi, 1.0)

    def _calculate_shoulder_alignment(self, shoulder_points: np.ndarray) -> float:
        shoulder_vector = shoulder_points[1] - shoulder_points[0]
        horizontal = np.array([1.0, 0.0, 0.0])
        angle = np.arccos(np.dot(shoulder_vector, horizontal) / np.linalg.norm(shoulder_vector))
        return 1.0 - min(abs(angle) / np.pi, 1.0)

    def _calculate_gaze_stability(self, eye_points: np.ndarray) -> float:
        eye_centers = np.mean(eye_points.reshape(-1, 2, 3), axis=1)
        gaze_vectors = np.diff(eye_centers, axis=0)
        stability = 1.0 - min(np.std(gaze_vectors) * 10, 1.0)
        return float(stability)

    def _calculate_gesture_speed(self, hand_positions: List[np.ndarray]) -> float:
        velocities = np.diff(hand_positions, axis=0)
        speed = np.mean(np.linalg.norm(velocities, axis=1))
        return min(speed * 10, 1.0)

    def _calculate_gesture_range(self, hand_positions: List[np.ndarray]) -> float:
        positions = np.array(hand_positions)
        range_of_motion = np.max(positions, axis=0) - np.min(positions, axis=0)
        return min(np.mean(range_of_motion) * 5, 1.0)

    def _get_mouth_points(self, landmarks: List) -> np.ndarray:
        MOUTH_LANDMARKS = [0, 17, 61, 291, 39, 269, 270, 409]
        return np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) for idx in MOUTH_LANDMARKS])

    def _calculate_smile_ratio(self, mouth_points: np.ndarray) -> float:
        mouth_width = np.linalg.norm(mouth_points[2] - mouth_points[3])
        mouth_height = np.linalg.norm(mouth_points[4] - mouth_points[6])
        return mouth_width / (mouth_height + 1e-6)

    def _get_eyebrow_points(self, landmarks: List) -> np.ndarray:
        EYEBROW_LANDMARKS = [63, 293, 105, 334]
        return np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) for idx in EYEBROW_LANDMARKS])

    def _calculate_eyebrow_raise_ratio(self, eyebrow_points: np.ndarray) -> float:
        left_raise = eyebrow_points[0][1] - eyebrow_points[1][1]
        right_raise = eyebrow_points[2][1] - eyebrow_points[3][1]
        return np.mean([left_raise, right_raise])

    def _get_eye_points(self, landmarks: List) -> np.ndarray:
        EYE_LANDMARKS = [33, 159, 133, 145, 362, 386, 374, 263]
        return np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) for idx in EYE_LANDMARKS])

    def _calculate_eye_squint_ratio(self, eye_points: np.ndarray) -> float:
        left_eye_height = np.linalg.norm(eye_points[0] - eye_points[1])
        right_eye_height = np.linalg.norm(eye_points[4] - eye_points[5])
        return 1.0 - np.mean([left_eye_height, right_eye_height])

    def _calculate_mouth_open_ratio(self, mouth_points: np.ndarray) -> float:
        mouth_height = np.linalg.norm(mouth_points[4] - mouth_points[6])
        return min(mouth_height * 3.0, 1.0)

    def _calculate_movement_speed(self, landmarks: List) -> float:
        points = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
        velocities = np.diff(points, axis=0)
        return np.mean(np.linalg.norm(velocities, axis=1))

    def _calculate_movement_smoothness(self, landmarks: List) -> float:
        points = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
        accelerations = np.diff(points, n=2, axis=0)
        jerk = np.mean(np.linalg.norm(accelerations, axis=1))
        return 1.0 - min(jerk * 10.0, 1.0)

    def _calculate_facial_engagement(self, landmarks: List) -> float:
        if not landmarks:
            return 0.0
        expressions = self._analyze_facial_expressions(landmarks)
        return np.mean(list(expressions.values()))

    def _calculate_body_engagement(self, landmarks: List) -> float:
        if not landmarks:
            return 0.0
        body_metrics = self._analyze_body_language(landmarks)
        return body_metrics['confidence_score']

    def _calculate_gesture_engagement(self, landmarks: List) -> float:
        if not landmarks:
            return 0.0
        return 0.5