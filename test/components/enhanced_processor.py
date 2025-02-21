# components/enhanced_processor.py

import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from pytesseract import image_to_string
from tqdm import tqdm
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from mediapipe import solutions
from deepface import DeepFace
import queue
import threading
from datetime import datetime
import os
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoAnalysisError(Exception):
    """Custom exception for video analysis errors"""
    pass


class OptimizedVideoAnalyzer:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize analyzer with optimized settings"""
        self.config = {
            'frame_skip': 10,  # Process every Nth frame
            'target_size': (640, 480),  # Target frame size
            'num_workers': 4,  # Number of worker threads
            'queue_size': 30,  # Frame queue size
            'batch_size': 8,  # Batch size for processing
            'min_detection_confidence': 0.7,
            'min_tracking_confidence': 0.5,
            'model_complexity': 1,
            'min_confidence': 0.5  # Confidence threshold
        }
        if config:
            self.config.update(config)

        self.initialize_system()

    def initialize_system(self):
        """Initialize the analysis system"""
        try:
            # Initialize queues
            self.frame_queue = queue.Queue(maxsize=self.config['queue_size'])
            self.results_queue = queue.Queue()

            # Initialize detectors
            self.initialize_detectors()

            logger.info("System initialized successfully")
        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}")
            raise VideoAnalysisError("Failed to initialize analysis system")

    def initialize_detectors(self):
        """Initialize detection models"""
        try:
            # Initialize MediaPipe pose detector
            self.pose_detector = solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=self.config['model_complexity'],
                min_detection_confidence=self.config['min_detection_confidence'],
                min_tracking_confidence=self.config['min_tracking_confidence']
            )

            # Initialize face detector
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            logger.info("All detectors initialized successfully")
        except Exception as e:
            logger.error(f"Detector initialization failed: {str(e)}")
            raise VideoAnalysisError("Failed to initialize detectors")

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Analyze video with optimized processing"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        try:
            # Get video metadata
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            duration = total_frames / fps

            print(f"\nAnalyzing Video: {os.path.basename(video_path)}")
            print(f"Duration: {duration:.1f}s | FPS: {fps} | Frames: {total_frames}")

            start_time = time.time()

            # Process frames with progress bar
            with tqdm(total=total_frames, desc="Processing Video", unit="frames") as pbar:
                producer_thread = threading.Thread(
                    target=self.frame_producer,
                    args=(cap, total_frames, pbar)
                )
                producer_thread.start()

                workers = []
                for _ in range(self.config['num_workers']):
                    worker = threading.Thread(target=self.process_frames_worker)
                    worker.start()
                    workers.append(worker)

                # Collect results
                results = {}
                processed_frames = 0
                expected_frames = total_frames // self.config['frame_skip']

                while processed_frames < expected_frames:
                    try:
                        frame_idx, result = self.results_queue.get(timeout=1)
                        results[frame_idx] = result
                        processed_frames += 1
                        self.results_queue.task_done()
                    except queue.Empty:
                        if not any(worker.is_alive() for worker in workers):
                            break

                producer_thread.join()
                for worker in workers:
                    worker.join()

            processing_time = time.time() - start_time

            print(f"\nProcessing completed in {processing_time:.2f} seconds")
            print("Generating analysis...")

            # Generate final analysis
            frame_data = [results[k] for k in sorted(results.keys())]
            analysis = self.generate_final_analysis(frame_data, {
                'duration': duration,
                'fps': fps,
                'total_frames': total_frames,
                'processed_frames': len(frame_data),
                'processing_time': processing_time
            })

            return analysis

        finally:
            cap.release()

    def frame_producer(self, cap: cv2.VideoCapture, total_frames: int, pbar: tqdm):
        """Producer thread for frame reading"""
        try:
            frame_count = 0
            last_timestamp = -1  # Track last valid timestamp

            while frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % self.config['frame_skip'] == 0:
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)  # Get frame timestamp in milliseconds

                    # Correct timestamp if necessary
                    if timestamp <= last_timestamp:
                        timestamp = last_timestamp + 1

                    last_timestamp = timestamp
                    resized_frame = cv2.resize(frame, self.config['target_size'])
                    self.frame_queue.put((frame_count, resized_frame))

                frame_count += 1
                pbar.update(1)

        except Exception as e:
            logger.error(f"Frame producer failed: {str(e)}")
            raise VideoAnalysisError("Frame producer encountered an error")

    def process_frames_worker(self):
        """Worker thread for processing frames"""
        while True:
            try:
                frame_idx, frame = self.frame_queue.get(timeout=1)
                result = self.process_single_frame(frame)
                self.results_queue.put((frame_idx, result))
                self.frame_queue.task_done()
            except queue.Empty:
                break

    def process_single_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a single frame"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Analyze pose and emotions in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                pose_future = executor.submit(self.analyze_pose, rgb_frame)
                emotion_future = executor.submit(self.analyze_emotions, rgb_frame)

                pose_data = pose_future.result()
                emotion_data = emotion_future.result()

            attention_score = self.calculate_attention(pose_data)

            return {
                'pose': pose_data,
                'emotion': emotion_data,
                'attention': attention_score,
                'timestamp': datetime.now().isoformat(),
                'text': self.extract_text(frame)  # New: Extract text
            }

        except Exception as e:
            logger.error(f"Frame processing failed: {str(e)}")
            return self.get_default_frame_data()

    def extract_text(self, frame: np.ndarray) -> str:
        """Extract text from a frame using OCR"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            text = image_to_string(gray)
            return text.strip()
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
            return ""

    def analyze_pose(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze pose in frame"""
        try:
            results = self.pose_detector.process(frame)

            if results.pose_landmarks:
                key_landmarks = [0, 1, 2, 3, 4, 5, 6, 7]  # Face and upper body
                landmarks = [[
                    results.pose_landmarks.landmark[i].x,
                    results.pose_landmarks.landmark[i].y,
                    results.pose_landmarks.landmark[i].z
                ] for i in key_landmarks]

                confidence = float(results.pose_landmarks.landmark[0].visibility)

                return {
                    'landmarks': landmarks,
                    'confidence': confidence
                }

            return self.get_default_pose_data()

        except Exception as e:
            logger.error(f"Pose analysis failed: {str(e)}")
            return self.get_default_pose_data()

    def analyze_emotions(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze emotions in frame"""
        try:
            result = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )

            if isinstance(result, list):
                result = result[0]

            emotions = result.get('emotion', {})
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else 'unknown'

            return {
                'dominant_emotion': dominant_emotion,
                'emotion_scores': emotions
            }

        except Exception as e:
            logger.error(f"Emotion analysis failed: {str(e)}")
            return self.get_default_emotion_data()

    def calculate_attention(self, pose_data: Dict[str, Any]) -> float:
        """Calculate attention score"""
        try:
            if not pose_data['landmarks']:
                return 0.0

            landmarks = pose_data['landmarks']

            if len(landmarks) >= 4:
                nose = landmarks[0]
                left_eye = landmarks[2]
                right_eye = landmarks[3]

                eye_center = [
                    (left_eye[0] + right_eye[0]) / 2,
                    (left_eye[1] + right_eye[1]) / 2
                ]

                angle = abs(np.arctan2(nose[1] - eye_center[1], nose[0] - eye_center[0]))
                attention = 1.0 - min(angle / np.pi, 1.0)
                return attention * pose_data['confidence']

            return 0.0

        except Exception as e:
            logger.error(f"Attention calculation failed: {str(e)}")
            return 0.0

    def generate_final_analysis(self, frame_data: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive final analysis"""
        try:
            attention_scores = [frame['attention'] for frame in frame_data]
            emotions = [frame['emotion']['dominant_emotion'] for frame in frame_data]

            emotion_counts = Counter(emotions)
            total_frames = len(frame_data)
            emotion_distribution = {
                emotion: round((count / total_frames) * 100, 1)
                for emotion, count in emotion_counts.items()
            }

            analysis = {
                "video_info": {
                    "duration_seconds": round(metadata['duration'], 2),
                    "total_frames": metadata['total_frames'],
                    "processed_frames": metadata['processed_frames'],
                    "fps": metadata['fps'],
                    "processing_time": round(metadata['processing_time'], 2)
                },
                "attention_analysis": {
                    "average_score": round(np.mean(attention_scores) * 100, 1),
                    "stability": round((1 - np.std(attention_scores)) * 100, 1),
                    "peak_attention": round(max(attention_scores) * 100, 1)
                },
                "emotional_analysis": {
                    "dominant_emotion": emotion_counts.most_common(1)[0][0],
                    "distribution": emotion_distribution,
                    "emotional_range": len(emotion_counts)
                },
                "engagement_metrics": self.calculate_engagement_metrics(frame_data)
            }

            self.print_analysis(analysis)
            return analysis

        except Exception as e:
            logger.error(f"Final analysis generation failed: {str(e)}")
            return {}

    def calculate_engagement_metrics(self, frame_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate detailed engagement metrics"""
        try:
            attention_scores = [f['attention'] for f in frame_results]
            emotions = [f['emotion']['dominant_emotion'] for f in frame_results]

            sustained_attention = sum(1 for score in attention_scores if score > 0.7) / len(attention_scores)
            emotional_engagement = len(set(emotions)) / 7  # Normalize by total possible emotions

            return {
                "sustained_attention": round(sustained_attention * 100, 1),
                "emotional_engagement": round(emotional_engagement * 100, 1),
                "overall_engagement": round((sustained_attention + emotional_engagement) * 50, 1)
            }

        except Exception as e:
            logger.error(f"Engagement metrics calculation failed: {str(e)}")
            return {
                "sustained_attention": 0.0,
                "emotional_engagement": 0.0,
                "overall_engagement": 0.0
            }

    def print_analysis(self, analysis: Dict[str, Any]):
        """Print comprehensive analysis results"""
        print("\n" + "=" * 50)
        print("ðŸ“Š VIDEO ANALYSIS RESULTS")
        print("=" * 50)

        print(f"\nðŸ“¹ Video Information:")
        print(f"Duration: {analysis['video_info']['duration_seconds']}s")
        print(f"Processed Frames: {analysis['video_info']['processed_frames']}")

        print(f"\nðŸŽ¯ Attention Analysis:")
        print(f"Average Score: {analysis['attention_analysis']['average_score']}%")
        print(f"Attention Stability: {analysis['attention_analysis']['stability']}%")
        print(f"Peak Attention: {analysis['attention_analysis']['peak_attention']}%")

        print(f"\nðŸ˜Š Emotional Analysis:")
        print(f"Dominant Emotion: {analysis['emotional_analysis']['dominant_emotion']}")
        print(f"Emotional Range: {analysis['emotional_analysis']['emotional_range']} emotions")
        print("\nEmotion Distribution:")
        for emotion, percentage in analysis['emotional_analysis']['distribution'].items():
            print(f"  â€¢ {emotion}: {percentage}%")

        print(f"\nðŸ“ˆ Engagement Metrics:")
        print(f"Sustained Attention: {analysis['engagement_metrics']['sustained_attention']}%")
        print(f"Emotional Engagement: {analysis['engagement_metrics']['emotional_engagement']}%")
        print(f"Overall Engagement: {analysis['engagement_metrics']['overall_engagement']}%")

    def get_default_frame_data(self) -> Dict[str, Any]:
        return {
            'pose': self.get_default_pose_data(),
            'emotion': self.get_default_emotion_data(),
            'attention': 0.0,
            'timestamp': datetime.now().isoformat()
        }

    def get_default_pose_data(self) -> Dict[str, Any]:
        return {
            'landmarks': [],
            'confidence': 0.0
        }

    def get_default_emotion_data(self) -> Dict[str, Any]:
        return {
            'dominant_emotion': 'unknown',
            'emotion_scores': {}
        }


def main():
    """Main execution function"""
    try:
        start_time = time.time()

        # Initialize analyzer
        analyzer = OptimizedVideoAnalyzer()

        # Set video path
        video_path = 'test_data/test_video.mp4'

        # Run analysis
        results = analyzer.analyze_video(video_path)

        # Save results
        output_dir = 'analysis_results'
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, 'video_analysis.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

        end_time = time.time()
        processing_time = end_time - start_time

        print(f"\nProcessing completed in {processing_time:.2f} seconds")
        print(f"Results saved to: {output_file}")

    except Exception as e:
        print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()