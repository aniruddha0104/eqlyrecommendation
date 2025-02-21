import cv2
import numpy as np
import logging
from pathlib import Path
import time
import mediapipe as mp
from tqdm import tqdm
import json
from typing import Dict, List, Optional, Union


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif str(type(obj)) == "":
            return None
        return super(NumpyEncoder, self).default(obj)


class OptimizedProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.frame_count = 0
        self.timestamp = 0
        self.consecutive_detections = 0

        # Create output directories
        self.output_dir = Path(self.config.get('output_dir', 'analysis_results'))
        self.debug_dir = self.output_dir / 'debug'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.setup_components()

    def setup_components(self):
        try:
            self.mp_face = mp.solutions.face_detection
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles

            self.face_detector = self.mp_face.FaceDetection(
                model_selection=1,
                min_detection_confidence=self.config['thresholds']['min_face_detection']
            )
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                smooth_landmarks=True,
                min_detection_confidence=self.config['thresholds']['min_pose_detection'],
                min_tracking_confidence=0.2,
                enable_segmentation=True
            )
            self.cascade_classifier = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            # Resolution settings
            self.face_target_size = (640, 480)
            self.pose_target_size = (1280, 720)
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}", exc_info=True)
            raise

    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        faces = []
        try:
            enhanced = cv2.resize(frame, self.face_target_size)
            rgb_frame = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

            # MediaPipe detection
            results = self.face_detector.process(rgb_frame)
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = enhanced.shape
                    x = max(0, int(bbox.xmin * w))
                    y = max(0, int(bbox.ymin * h))
                    width = min(int(bbox.width * w), w - x)
                    height = min(int(bbox.height * h), h - y)
                    faces.append({
                        'bbox': {'x': x, 'y': y, 'w': width, 'h': height},
                        'confidence': float(detection.score[0]),
                        'method': 'mediapipe'
                    })

            # Cascade classifier backup
            if not faces:
                gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
                cascade_faces = self.cascade_classifier.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
                )
                for (x, y, w, h) in cascade_faces:
                    faces.append({
                        'bbox': {'x': x, 'y': y, 'w': w, 'h': h},
                        'confidence': 0.5,
                        'method': 'cascade'
                    })
            return faces
        except Exception as e:
            self.logger.error(f"Face detection failed: {str(e)}")
            return []

    def preprocess_pose_frame(self, frame: np.ndarray) -> np.ndarray:
        try:
            frame = cv2.resize(frame, self.pose_target_size)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)
            return enhanced
        except Exception as e:
            self.logger.error(f"Pose preprocessing failed: {str(e)}")
            return frame

    def process_single_frame(self, frame: np.ndarray) -> Dict:
        try:
            faces = self.detect_faces(frame)
            pose_frame = self.preprocess_pose_frame(frame)
            rgb_frame = cv2.cvtColor(pose_frame, cv2.COLOR_BGR2RGB)
            pose_results = self.pose_detector.process(rgb_frame)

            if self.config.get('debug', False):
                self.save_debug_frame(pose_frame, faces, pose_results)

            pose_detected = False
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                key_points = [
                    self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                    self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    self.mp_pose.PoseLandmark.LEFT_HIP,
                    self.mp_pose.PoseLandmark.RIGHT_HIP,
                    self.mp_pose.PoseLandmark.NOSE
                ]
                visibility_threshold = 0.2
                pose_detected = any(landmarks[point].visibility > visibility_threshold
                                    for point in key_points)

            quality_metrics = self.calculate_quality_metrics(frame)
            engagement_score = self.calculate_engagement(
                pose_results.pose_landmarks if pose_detected else None,
                faces
            )

            return {
                'frame_index': self.frame_count,
                'faces': faces,
                'pose_detected': pose_detected,
                'quality': quality_metrics,
                'engagement': engagement_score
            }
        except Exception as e:
            self.logger.error(f"Frame processing failed: {str(e)}")
            return {'frame_index': self.frame_count, 'error': str(e)}

    def save_debug_frame(self, frame: np.ndarray, faces: List[Dict], pose_results) -> None:
        try:
            debug_frame = frame.copy()
            for face in faces:
                bbox = face['bbox']
                color = (0, 255, 0) if face['method'] == 'mediapipe' else (0, 165, 255)
                cv2.rectangle(
                    debug_frame,
                    (bbox['x'], bbox['y']),
                    (bbox['x'] + bbox['w'], bbox['y'] + bbox['h']),
                    color,
                    2
                )
                cv2.putText(
                    debug_frame,
                    f"{face['method']}: {face['confidence']:.2f}",
                    (bbox['x'], bbox['y'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
            if pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    debug_frame,
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
            debug_path = self.debug_dir / f"frame_{self.frame_count:04d}.jpg"
            cv2.imwrite(str(debug_path), debug_frame)
        except Exception as e:
            self.logger.error(f"Debug frame save failed: {str(e)}")

    def calculate_quality_metrics(self, frame: np.ndarray) -> Dict:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return {
                'clarity': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                'brightness': float(np.mean(gray)),
                'contrast': float(gray.std())
            }
        except Exception as e:
            self.logger.error(f"Quality metrics calculation failed: {str(e)}")
            return {'clarity': 0.0, 'brightness': 0.0, 'contrast': 0.0}

    def calculate_engagement(self, pose_landmarks, faces: List[Dict]) -> float:
        try:
            engagement_score = 0.0
            if faces:
                face_scores = []
                for face in faces:
                    x = face['bbox']['x']
                    w = face['bbox']['w']
                    center_x = x + w / 2
                    center_score = 1 - (2 * abs(center_x / self.face_target_size[0] - 0.5))
                    face_scores.append(center_score * face['confidence'])
                engagement_score += max(face_scores) * 0.5

            if pose_landmarks:
                nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
                left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
                head_tilt = abs(nose.y - shoulder_center_y)
                posture_score = max(0, 1.0 - (head_tilt * 2))
                engagement_score += posture_score * 0.5

            return engagement_score
        except Exception as e:
            self.logger.error(f"Engagement calculation failed: {str(e)}")
            return 0.0

    def analyze_video(self, video_path: str) -> Dict:
        cap = None
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise IOError(f"Cannot open video: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            results = []
            times = []

            with tqdm(total=total_frames, desc="Processing video") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if self.frame_count % 2 == 0:  # Process every other frame
                        start_time = time.time()
                        result = self.process_single_frame(frame)
                        processing_time = time.time() - start_time
                        results.append(result)
                        times.append(processing_time)

                    self.frame_count += 1
                    pbar.update(1)

            # Safeguard against empty results
            if not results:
                raise ValueError("No valid frames were processed.")

            summary = {
                'performance': {
                    'total_frames': len(results),
                    'average_fps': 1.0 / (sum(times) / len(times)) if times else 0,
                    'processing_time': sum(times)
                },
                'detection_rates': {
                    'face_detection_rate': sum(1 for r in results if r.get('faces')) / len(results),
                    'pose_detection_rate': sum(1 for r in results if r.get('pose_detected')) / len(results)
                },
                'results': results  # Ensure results are included in the summary
            }

            return {
                'summary': summary,
                'recommendations': self.generate_recommendations(summary),
                'results': results
            }
        except Exception as e:
            self.logger.error(f"Video analysis failed: {str(e)}")
            raise
        finally:
            if cap is not None:
                cap.release()

    def generate_recommendations(self, summary: Dict) -> List[Dict]:
        recommendations = []

        # Performance recommendations
        avg_fps = summary['performance']['average_fps']
        if avg_fps < self.config['thresholds']['min_fps']:
            recommendations.append({
                'category': 'performance',
                'issue': 'Low processing speed',
                'recommendation': 'Consider reducing video resolution or improving hardware.',
                'severity': 'high' if avg_fps < 5 else 'medium'
            })

        # Face detection recommendations
        face_rate = summary['detection_rates']['face_detection_rate']
        if face_rate < self.config['thresholds']['min_face_detection']:
            recommendations.append({
                'category': 'face_detection',
                'issue': 'Low face detection rate',
                'recommendation': 'Improve lighting and ensure faces are clearly visible.',
                'severity': 'high' if face_rate < 0.2 else 'medium'
            })

        # Pose detection recommendations
        pose_rate = summary['detection_rates']['pose_detection_rate']
        if pose_rate < self.config['thresholds']['min_pose_detection']:
            recommendations.append({
                'category': 'pose_detection',
                'issue': 'Low pose detection rate',
                'recommendation': 'Ensure full upper body visibility and reduce occlusions.',
                'severity': 'high' if pose_rate < 0.1 else 'medium'
            })

        # Quality metric recommendations
        avg_clarity = np.mean([r['quality']['clarity'] for r in summary['results']])
        if avg_clarity < self.config['thresholds']['min_clarity']:
            recommendations.append({
                'category': 'quality',
                'issue': 'Low video clarity',
                'recommendation': 'Use a higher-resolution camera or improve focus.',
                'severity': 'medium'
            })

        return recommendations

    def run(self, video_path: str):
        try:
            video_path = str(Path(video_path).resolve())
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")

            results = self.analyze_video(video_path)

            print("\n=== Analysis Results ===")
            print(f"Total Frames: {results['summary']['performance']['total_frames']}")
            print(f"Average FPS: {results['summary']['performance']['average_fps']:.1f}")
            print(f"Face Detection Rate: {results['summary']['detection_rates']['face_detection_rate']:.1%}")
            print(f"Pose Detection Rate: {results['summary']['detection_rates']['pose_detection_rate']:.1%}")

            print("\nRecommendations:")
            for rec in results['recommendations']:
                severity_color = '\033[91m' if rec['severity'] == 'high' else '\033[93m'
                print(f"{severity_color}[{rec['severity'].upper()}]\033[0m {rec['category']}: {rec['recommendation']}")

            results_path = self.output_dir / 'analysis_results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, cls=NumpyEncoder, indent=2)
            print(f"\nResults saved to: {results_path}")
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}", exc_info=True)
            print(f"Error: {str(e)}")


def main():
    config = {
        'output_dir': 'analysis_results',
        'debug': True,
        'thresholds': {
            'min_fps': 15,
            'min_face_detection': 0.5,
            'min_pose_detection': 0.2,
            'min_clarity': 100,
            'min_brightness': 80,
            'min_contrast': 20
        }
    }

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('video_analysis.log'),
            logging.StreamHandler()
        ]
    )

    try:
        processor = OptimizedProcessor(config)
        processor.run('test_data/test_video.mp4')  # Update with your video path
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()