"""Video processing module for the assessment platform.

This module handles video loading, frame extraction, and core video processing
for the assessment platform.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Generator, Iterator
import logging
import time
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
from datetime import datetime
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoProcessingError(Exception):
    """Custom exception for video processing errors"""
    pass


class VideoProcessor:
    """Core video processing module for handling video data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the video processor with optional configuration.

        Args:
            config: Optional configuration dictionary with parameters
        """
        self.config = {
            'frame_skip': 5,  # Process every Nth frame
            'target_size': (640, 480),  # Target frame size for processing
            'queue_size': 30,  # Frame queue size for parallel processing
            'num_workers': 4,  # Number of worker threads
            'batch_size': 8,  # Batch size for processing
            'min_video_duration': 3.0,  # Minimum video duration in seconds
            'max_video_duration': 600.0  # Maximum video duration in seconds
        }
        if config:
            self.config.update(config)

        self.frame_queue = None
        self.results_queue = None
        self.total_frames = 0
        self.fps = 0
        self.duration = 0
        self.video_info = {}

    def load_video(self, video_path: str) -> bool:
        """Load and validate a video file.

        Args:
            video_path: Path to the video file

        Returns:
            True if video loaded successfully, False otherwise

        Raises:
            FileNotFoundError: If video file not found
            VideoProcessingError: If video cannot be processed
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoProcessingError(f"Failed to open video: {video_path}")

        try:
            # Get video metadata
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.duration = self.total_frames / max(1.0, self.fps)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Validate video
            if self.duration < self.config['min_video_duration']:
                logger.error(f"Video too short: {self.duration:.2f}s")
                return False

            if self.duration > self.config['max_video_duration']:
                logger.error(f"Video too long: {self.duration:.2f}s")
                return False

            if self.fps <= 0:
                logger.error(f"Invalid FPS: {self.fps}")
                return False

            # Store video information
            self.video_info = {
                'path': video_path,
                'filename': os.path.basename(video_path),
                'total_frames': self.total_frames,
                'fps': self.fps,
                'duration': self.duration,
                'width': width,
                'height': height,
                'aspect_ratio': width / height if height > 0 else 0,
                'target_size': self.config['target_size']
            }

            logger.info(f"Video loaded: {video_path}")
            logger.info(f"Duration: {self.duration:.2f}s, FPS: {self.fps:.1f}, Frames: {self.total_frames}")
            return True

        finally:
            cap.release()

    def get_video_info(self) -> Dict[str, Any]:
        """Get video information dictionary.

        Returns:
            Dictionary containing video metadata
        """
        return self.video_info

    def extract_frames(self, video_path: str) -> Generator[Tuple[int, np.ndarray, float], None, None]:
        """Extract frames from the video with frame index and timestamp.

        Args:
            video_path: Path to the video file

        Yields:
            Tuple containing (frame_index, frame_data, timestamp_ms)

        Raises:
            VideoProcessingError: If frame extraction fails
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoProcessingError(f"Failed to open video for frame extraction: {video_path}")

        try:
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.config['frame_skip'] == 0:
                    # Get timestamp in milliseconds
                    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

                    # Resize frame to target size
                    resized_frame = cv2.resize(frame, self.config['target_size'])

                    yield frame_idx, resized_frame, timestamp_ms

                frame_idx += 1

        except Exception as e:
            logger.error(f"Frame extraction failed: {str(e)}")
            raise VideoProcessingError(f"Failed to extract frames: {str(e)}")

        finally:
            cap.release()

    def process_video(self, video_path: str,
                      frame_processor_func: callable,
                      progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Process video with parallel frame processing.

        Args:
            video_path: Path to the video file
            frame_processor_func: Function to process each frame
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary containing processed video results

        Raises:
            VideoProcessingError: If video processing fails
        """
        if not self.load_video(video_path):
            raise VideoProcessingError(f"Failed to load video: {video_path}")

        start_time = time.time()

        try:
            # Initialize queues
            self.frame_queue = queue.Queue(maxsize=self.config['queue_size'])
            self.results_queue = queue.Queue()

            # Setup progress tracking
            processed_frames = 0
            expected_frames = self.total_frames // self.config['frame_skip']

            with tqdm(total=self.total_frames, desc="Processing Video", unit="frames") as pbar:
                # Start frame producer thread
                producer_thread = threading.Thread(
                    target=self._frame_producer,
                    args=(video_path, pbar)
                )
                producer_thread.start()

                # Start worker threads
                workers = []
                for _ in range(self.config['num_workers']):
                    worker = threading.Thread(
                        target=self._process_frames_worker,
                        args=(frame_processor_func,)
                    )
                    worker.daemon = True
                    worker.start()
                    workers.append(worker)

                # Collect results
                results = {}

                while processed_frames < expected_frames:
                    try:
                        frame_idx, timestamp, result = self.results_queue.get(timeout=1)
                        results[frame_idx] = (timestamp, result)
                        processed_frames += 1

                        if progress_callback and processed_frames % 10 == 0:
                            progress_value = min(0.99, processed_frames / expected_frames)
                            progress_callback(progress_value)

                        self.results_queue.task_done()
                    except queue.Empty:
                        if not any(worker.is_alive() for worker in workers) and producer_thread.is_alive() is False:
                            break

                producer_thread.join()
                for worker in workers:
                    worker.join(timeout=1.0)

            processing_time = time.time() - start_time

            # Final progress update
            if progress_callback:
                progress_callback(1.0)

            # Convert results to chronological list with timestamps
            frame_results = []
            for frame_idx in sorted(results.keys()):
                timestamp, result = results[frame_idx]
                result['frame_idx'] = frame_idx
                result['timestamp_ms'] = timestamp
                frame_results.append(result)

            # Compile processing summary
            summary = {
                'video_info': self.video_info,
                'processing_info': {
                    'processed_frames': processed_frames,
                    'expected_frames': expected_frames,
                    'processing_time': processing_time,
                    'frames_per_second': processed_frames / processing_time if processing_time > 0 else 0
                },
                'frame_results': frame_results
            }

            logger.info(f"Video processing completed in {processing_time:.2f}s")
            logger.info(f"Processed {processed_frames} frames")

            return summary

        except Exception as e:
            logger.error(f"Video processing failed: {str(e)}")
            raise VideoProcessingError(f"Failed to process video: {str(e)}")

    def _frame_producer(self, video_path: str, pbar: Optional[tqdm] = None):
        """Producer thread for frame extraction.

        Args:
            video_path: Path to the video file
            pbar: Optional progress bar
        """
        try:
            for frame_idx, frame, timestamp_ms in self.extract_frames(video_path):
                self.frame_queue.put((frame_idx, frame, timestamp_ms))
                if pbar:
                    pbar.update(self.config['frame_skip'])

            # Signal completion
            for _ in range(self.config['num_workers']):
                self.frame_queue.put((None, None, None))

        except Exception as e:
            logger.error(f"Frame producer failed: {str(e)}")

    def _process_frames_worker(self, frame_processor_func: callable):
        """Worker thread for processing frames.

        Args:
            frame_processor_func: Function to process each frame
        """
        while True:
            try:
                frame_idx, frame, timestamp_ms = self.frame_queue.get(timeout=5)

                # Check for end signal
                if frame_idx is None:
                    self.frame_queue.task_done()
                    break

                # Process frame and put result in results queue
                try:
                    result = frame_processor_func(frame)
                    self.results_queue.put((frame_idx, timestamp_ms, result))
                except Exception as e:
                    logger.error(f"Frame processing error on frame {frame_idx}: {str(e)}")
                    # Put empty result to maintain frame count
                    self.results_queue.put((frame_idx, timestamp_ms, {}))

                self.frame_queue.task_done()

            except queue.Empty:
                break

    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        """Extract audio from video file.

        Args:
            video_path: Path to the video file
            output_path: Optional output path for audio file

        Returns:
            Path to extracted audio file

        Raises:
            VideoProcessingError: If audio extraction fails
        """
        try:
            if output_path is None:
                # Generate output path
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_dir = os.path.dirname(video_path)
                output_path = os.path.join(output_dir, f"{base_name}_audio.wav")

            # Use ffmpeg for audio extraction
            import subprocess

            command = [
                'ffmpeg',
                '-i', video_path,
                '-q:a', '0',
                '-map', 'a',
                '-y',  # Overwrite output file
                output_path
            ]

            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if process.returncode != 0:
                error_message = process.stderr.decode()
                logger.error(f"Audio extraction failed: {error_message}")
                raise VideoProcessingError(f"Failed to extract audio: {error_message}")

            if not os.path.exists(output_path):
                raise VideoProcessingError("Audio extraction failed: Output file not created")

            logger.info(f"Audio extracted to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Audio extraction failed: {str(e)}")
            raise VideoProcessingError(f"Failed to extract audio: {str(e)}")

    def generate_thumbnail(self, video_path: str, output_path: Optional[str] = None,
                           time_offset: float = 1.0) -> str:
        """Generate thumbnail from video.

        Args:
            video_path: Path to the video file
            output_path: Optional output path for thumbnail
            time_offset: Time in seconds to capture thumbnail from

        Returns:
            Path to thumbnail image

        Raises:
            VideoProcessingError: If thumbnail generation fails
        """
        try:
            if output_path is None:
                # Generate output path
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_dir = os.path.dirname(video_path)
                output_path = os.path.join(output_dir, f"{base_name}_thumbnail.jpg")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise VideoProcessingError(f"Failed to open video for thumbnail: {video_path}")

            # Seek to specified time
            cap.set(cv2.CAP_PROP_POS_MSEC, time_offset * 1000)

            # Read frame
            ret, frame = cap.read()
            if not ret:
                # Try first frame if seeking failed
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    raise VideoProcessingError("Failed to read frame for thumbnail")

            # Resize to standard size
            thumbnail = cv2.resize(frame, (640, 360))

            # Save thumbnail
            cv2.imwrite(output_path, thumbnail)

            cap.release()

            if not os.path.exists(output_path):
                raise VideoProcessingError("Thumbnail generation failed: Output file not created")

            logger.info(f"Thumbnail generated: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Thumbnail generation failed: {str(e)}")
            raise VideoProcessingError(f"Failed to generate thumbnail: {str(e)}")