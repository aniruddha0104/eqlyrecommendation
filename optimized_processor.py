import cv2
import numpy as np
import logging
from pathlib import Path
import time
import mediapipe as mp
from tqdm import tqdm
import threading
from queue import Queue
import json


class UltimateProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.frame_queue = Queue(maxsize=30)
        self.result_queue = Queue()
        self.frame_count = 0
        self.setup_components()

    def setup_components(self):
        """Initialize components with ultra-optimized parameters"""
        try:
            # Initialize MediaPipe components with dedicated instances
            self.mp_face = mp.solutions.face_detection
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils

            # Initialize multiple face detectors for redundancy
            self.face_detectors = [
                self.mp_face.FaceDetection(
                    model_selection=1,  # Full-range model
                    min_detection_confidence=conf
                ) for conf in [0.3, 0.5]  # Multiple confidence thresholds
            ]

            # Initialize pose detector with optimal settings
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,  # Balanced setting
                smooth_landmarks=True,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )

            # Initialize cascade classifier as backup
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

            # Configure enhanced processing parameters
            self.target_size = (640, 480)
            self.debug_dir = Path(self.config.get('output_dir', 'debug_output'))
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        except Exception as e:
            self.logger.error(f"Component initialization failed: {str(e)}", exc_info=True)
            raise

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Advanced frame enhancement pipeline"""
        try:
            # Initial resize for consistent processing
            frame = cv2.resize(frame, (1280, 720))

            # Convert to LAB color space for better enhancement
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to luminance channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced_l = clahe.apply(l)

            # Merge channels back
            enhanced_lab = cv2.merge([enhanced_l, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            # Apply additional enhancements
            enhanced = cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)

            # Final resize to target size
            enhanced = cv2.resize(enhanced, self.target_size)

            return enhanced
        except Exception as e:
            self.logger.error(f"Frame enhancement failed: {str(e)}")
            return cv2.resize(frame, self.target_size)

    def detect_faces(self, frame: np.ndarray) -> list:
        """Multi-stage face detection with fallback"""
        faces = []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Try MediaPipe detectors
        for detector in self.face_detectors:
            try:
                results = detector.process(frame_rgb)
                if results.detections:
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        confidence = detection.score[0]

                        faces.append({
                            'bbox': (x, y, width, height),
                            'confidence': float(confidence),
                            'source': 'mediapipe'
                        })
                    break  # Stop if faces found
            except Exception as e:
                self.logger.warning(f"MediaPipe face detection failed: {str(e)}")

        # Fallback to Cascade Classifier if no faces found
        if not faces:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cascade_faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )

                for (x, y, w, h) in cascade_faces:
                    faces.append({
                        'bbox': (x, y, w, h),
                        'confidence': 0.5,  # Default confidence for cascade
                        'source': 'cascade'
                    })
            except Exception as e:
                self.logger.warning(f"Cascade face detection failed: {str(e)}")

        return faces

    def process_frame(self, frame: np.ndarray) -> dict:
        """Process single frame with comprehensive analysis"""
        try:
            # Enhance frame
            enhanced = self.enhance_frame(frame)

            # Detect faces with multi-stage approach
            faces = self.detect_faces(enhanced)

            # Detect pose
            pose_results = self.pose.process(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))

            # Calculate quality metrics
            quality = self.calculate_quality(enhanced)

            # Save debug visualization if enabled
            if self.config.get('save_debug', False):
                self.save_debug_frame(enhanced, faces, pose_results)

            return {
                'frame_index': self.frame_count,
                'face_detection': {
                    'detected': len(faces) > 0,
                    'count': len(faces),
                    'detections': faces
                },
                'pose_detection': {
                    'detected': pose_results.pose_landmarks is not None,
                    'landmarks': pose_results.pose_landmarks
                },
                'quality': quality
            }

        except Exception as e:
            self.logger.error(f"Frame processing failed: {str(e)}")
            return {'frame_index': self.frame_count, 'error': str(e)}

    def calculate_quality(self, frame: np.ndarray) -> dict:
        """Calculate comprehensive quality metrics"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Basic metrics
            brightness = np.mean(gray)
            contrast = np.std(gray)
            clarity = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Edge detection for detail measurement
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])

            # Noise estimation
            noise = estimate_noise(gray)

            return {
                'brightness': float(brightness),
                'contrast': float(contrast),
                'clarity': float(clarity),
                'edge_density': float(edge_density),
                'noise_level': float(noise)
            }
        except Exception as e:
            self.logger.error(f"Quality calculation failed: {str(e)}")
            return {}

    def save_debug_frame(self, frame: np.ndarray, faces: list, pose_results) -> None:
        """Save annotated frame for debugging"""
        try:
            debug_frame = frame.copy()

            # Draw face detections
            for face in faces:
                x, y, w, h = face['bbox']
                color = (0, 255, 0) if face['source'] == 'mediapipe' else (0, 165, 255)
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(debug_frame,
                            f"{face['source']}: {face['confidence']:.2f}",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2)

            # Draw pose landmarks
            if pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    debug_frame,
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )

            # Save frame
            cv2.imwrite(
                str(self.debug_dir / f"debug_frame_{self.frame_count:04d}.jpg"),
                debug_frame
            )

        except Exception as e:
            self.logger.error(f"Debug frame save failed: {str(e)}")

    def analyze_video(self, video_path: str) -> dict:
        """Process video with comprehensive analysis"""
        cap = None
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise IOError(f"Could not open video: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            results = {
                'frames': [],
                'metrics': {
                    'face_detections': [],
                    'pose_detections': [],
                    'quality_metrics': []
                }
            }

            with tqdm(total=total_frames, desc="Processing video") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Process frame
                    start_time = time.time()
                    frame_result = self.process_frame(frame)
                    processing_time = time.time() - start_time

                    # Update metrics
                    results['frames'].append({
                        'index': self.frame_count,
                        'processing_time': processing_time,
                        **frame_result
                    })

                    self.frame_count += 1
                    pbar.update(1)

            # Generate summary
            summary = self.generate_summary(results)
            results['summary'] = summary

            return results

        except Exception as e:
            self.logger.error(f"Video analysis failed: {str(e)}")
            raise
        finally:
            if cap is not None:
                cap.release()

    def generate_summary(self, results: dict) -> dict:
        """Generate comprehensive analysis summary"""
        try:
            frames = results['frames']

            # Calculate detection rates
            face_detections = [f['face_detection']['detected'] for f in frames]
            pose_detections = [f['pose_detection']['detected'] for f in frames]

            # Calculate average processing time
            processing_times = [f['processing_time'] for f in frames]

            return {
                'performance': {
                    'total_frames': len(frames),
                    'average_fps': 1.0 / (sum(processing_times) / len(processing_times)),
                    'max_processing_time': max(processing_times),
                    'min_processing_time': min(processing_times)
                },
                'detection_rates': {
                    'face_detection_rate': sum(face_detections) / len(face_detections),
                    'pose_detection_rate': sum(pose_detections) / len(pose_detections)
                }
            }

        except Exception as e:
            self.logger.error(f"Summary generation failed: {str(e)}")
            return {}


def estimate_noise(gray_img):
    """Estimate image noise level"""
    H, W = gray_img.shape
    M = [[1, -2, 1],
         [-2, 4, -2],
         [1, -2, 1]]
    sigma = np.sum(np.sum(np.absolute(cv2.filter2D(gray_img, -1, np.array(M)))))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W - 2) * (H - 2))
    return sigma


def main():
    """Main execution with optimal configuration"""
    config = {
        'output_dir': 'analysis_results',
        'save_debug': True,
        'thresholds': {
            'min_face_detection': 0.3,
            'min_pose_detection': 0.3,
            'min_brightness': 40,
            'min_contrast': 30
        }
    }

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('video_analysis.log'),
            logging.StreamHandler()
        ]
    )

    try:
        processor = UltimateProcessor(config)
        results = processor.analyze_video('test_data/test_video.mp4')

        print("\n=== Analysis Results ===")
        print(f"Total Frames: {results['summary']['performance']['total_frames']}")
        print(f"Average FPS: {results['summary']['performance']['average_fps']:.1f}")
        print(f"Face Detection Rate: {results['summary']['detection_rates']['face_detection_rate']:.1%}")
        print(f"Pose Detection Rate: {results['summary']['detection_rates']['pose_detection_rate']:.1%}")

        # Save detailed results
        with open('analysis_results/detailed_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()