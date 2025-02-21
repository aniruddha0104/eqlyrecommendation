# test_teaching_assessment.py
import json
from typing import Dict, Any

import cv2
import numpy as np
import time
import logging
from datetime import datetime
from pathlib import Path

import psutil as psutil
from assessment.teaching_assessment import TeachingAssessmentSystem

# Configure logging for detailed debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('teaching_assessment_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TeachingAssessmentTester:
    """
    Comprehensive testing suite for the Teaching Assessment System.

    This tester performs systematic validation of:
    1. Video processing capabilities
    2. Audio analysis functionality
    3. Real-time assessment metrics
    4. System performance and reliability
    5. Visualization accuracy
    """

    def __init__(self):
        """Initialize the testing environment with necessary configurations."""
        self.config = {
            'video_config': {
                'frame_width': 1280,
                'frame_height': 720,
                'fps': 30
            },
            'audio_config': {
                'sample_rate': 16000,
                'channels': 1,
                'chunk_size': 1024
            },
            'assessment_config': {
                'metrics_update_rate': 500,  # ms
                'visualization_enabled': True,
                'debug_mode': True
            }
        }

        # Initialize test metrics storage
        self.test_results = {
            'video_metrics': [],
            'audio_metrics': [],
            'performance_metrics': [],
            'system_stats': []
        }

        # Create output directory for test artifacts
        self.output_dir = Path('test_results') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # test_teaching_assessment.py
    def run_comprehensive_test(self, duration_seconds: int = 30):
        """Run a comprehensive system test."""
        logger.info(f"Starting comprehensive test for {duration_seconds} seconds")

        cap = None
        try:
            # Initialize the assessment system
            system = TeachingAssessmentSystem(self.config)
            logger.info("System initialized successfully")

            # Start webcam capture
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Failed to open webcam")

            # Configure camera
            self._configure_camera(cap)

            # Run test loop
            self._run_test_loop(system, cap, duration_seconds)

        except Exception as e:
            logger.error(f"Test failed with error: {str(e)}", exc_info=True)
        finally:
            if cap is not None and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            logger.info("Test completed")

    def _configure_camera(self, cap):
        """Configure camera settings."""
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['video_config']['frame_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['video_config']['frame_height'])
        cap.set(cv2.CAP_PROP_FPS, self.config['video_config']['fps'])
        logger.info("Camera configured successfully")

    def _run_test_loop(self, system, cap, duration_seconds):
        """Run the main test loop."""
        start_time = time.time()
        frame_count = 0

        while (time.time() - start_time) < duration_seconds:
            # Process frame
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                continue

            # Process frame and collect metrics
            try:
                frame_metrics = self._process_frame(system, frame)
                if frame_metrics:
                    self._collect_metrics(frame_metrics)
                    self._display_debug_info(frame, frame_metrics)
            except Exception as e:
                logger.error(f"Frame processing error: {str(e)}")
                continue

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Test interrupted by user")
                break

        # Analyze results
        self._analyze_test_results(frame_count, duration_seconds)

    def _process_frame(self, system: TeachingAssessmentSystem,
                       frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame and collect detailed metrics.

        Args:
            system: Teaching assessment system instance
            frame: Input video frame

        Returns:
            Dictionary containing frame processing metrics
        """
        try:
            # Start timing
            start_time = time.perf_counter()

            # Process through assessment system
            assessment = system.process_frame(frame)

            # Calculate processing time
            processing_time = time.perf_counter() - start_time

            return {
                'timestamp': time.time(),
                'assessment': assessment,
                'processing_time': processing_time,
                'frame_shape': frame.shape,
                'system_metrics': {
                    'cpu_usage': psutil.cpu_percent(),
                    'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024
                }
            }

        except Exception as e:
            logger.error(f"Frame processing failed: {str(e)}", exc_info=True)
            return None

    def _display_debug_info(self, frame: np.ndarray, metrics: Dict[str, Any]):
        """
        Display real-time debug information on frame.

        Args:
            frame: Current video frame
            metrics: Frame processing metrics
        """
        if not metrics:
            return frame

        # Create debug overlay
        debug_frame = frame.copy()

        # Draw performance metrics
        cv2.putText(
            debug_frame,
            f"Processing Time: {metrics['processing_time'] * 1000:.1f}ms",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # Draw assessment metrics
        y_offset = 60
        for metric_name, value in metrics['assessment'].items():
            if isinstance(value, (int, float)):
                cv2.putText(
                    debug_frame,
                    f"{metric_name}: {value:.2f}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                y_offset += 30

        cv2.imshow('Teaching Assessment Test', debug_frame)

    def _collect_metrics(self, metrics: Dict[str, Any]):
        """
        Collect and store frame metrics for analysis.

        Args:
            metrics: Frame processing metrics
        """
        if not metrics:
            return

        # Store relevant metrics
        self.test_results['video_metrics'].append({
            'timestamp': metrics['timestamp'],
            'processing_time': metrics['processing_time']
        })

        self.test_results['system_stats'].append(metrics['system_metrics'])

        if 'assessment' in metrics:
            self.test_results['performance_metrics'].append(metrics['assessment'])

    def _analyze_test_results(self, frame_count: int, duration: float):
        """
        Analyze and save test results.

        Args:
            frame_count: Total number of processed frames
            duration: Test duration in seconds
        """
        # Calculate summary statistics
        processing_times = [m['processing_time']
                            for m in self.test_results['video_metrics']]

        summary = {
            'total_frames': frame_count,
            'average_fps': frame_count / duration,
            'average_processing_time': np.mean(processing_times),
            'max_processing_time': np.max(processing_times),
            'min_processing_time': np.min(processing_times),
            'assessment_metrics_collected': len(self.test_results['performance_metrics'])
        }

        # Save results
        results_file = self.output_dir / 'test_summary.json'
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=4)

        logger.info(f"Test results saved to {results_file}")

        # Log summary
        logger.info("\nTest Summary:")
        for key, value in summary.items():
            logger.info(f"{key}: {value}")


if __name__ == "__main__":
    # Run comprehensive test
    tester = TeachingAssessmentTester()
    tester.run_comprehensive_test(duration_seconds=30)