"""Utility functions for video processing.

This module provides helper functions for video manipulation,
metadata extraction, and format conversions.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import os
import json
import time
import hashlib
from datetime import datetime
import subprocess
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoUtilsError(Exception):
    """Custom exception for video utilities errors"""
    pass


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    """Extract comprehensive metadata from video file.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary containing video metadata

    Raises:
        FileNotFoundError: If video file not found
        VideoUtilsError: If metadata extraction fails
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        # Basic metadata using OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoUtilsError(f"Failed to open video: {video_path}")

        # Extract basic properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        cap.release()

        # File metadata
        file_size = os.path.getsize(video_path)
        file_created = os.path.getctime(video_path)
        file_modified = os.path.getmtime(video_path)
        file_extension = os.path.splitext(video_path)[1].lower()

        # Try to get codec info
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = ''.join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

        # Calculate aspect ratio
        aspect_ratio = width / height if height > 0 else 0

        # Advanced metadata using ffprobe if available
        advanced_metadata = {}
        try:
            advanced_metadata = get_ffprobe_metadata(video_path)
        except Exception as e:
            logger.warning(f"Failed to get advanced metadata with ffprobe: {str(e)}")

        # Combine all metadata
        metadata = {
            'basic': {
                'width': width,
                'height': height,
                'fps': float(fps),
                'frame_count': frame_count,
                'duration': float(duration),
                'aspect_ratio': float(aspect_ratio),
                'codec': fourcc
            },
            'file': {
                'path': video_path,
                'size': file_size,
                'size_mb': round(file_size / (1024 * 1024), 2),
                'created': datetime.fromtimestamp(file_created).isoformat(),
                'modified': datetime.fromtimestamp(file_modified).isoformat(),
                'extension': file_extension
            },
            'advanced': advanced_metadata,
            'hash': calculate_file_hash(video_path)
        }

        return metadata

    except Exception as e:
        logger.error(f"Metadata extraction failed: {str(e)}")
        raise VideoUtilsError(f"Failed to extract video metadata: {str(e)}")


def get_ffprobe_metadata(video_path: str) -> Dict[str, Any]:
    """Get advanced metadata using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary of advanced metadata

    Raises:
        VideoUtilsError: If ffprobe fails
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            raise VideoUtilsError(f"ffprobe failed: {result.stderr}")

        probe_data = json.loads(result.stdout)

        # Extract relevant metadata
        metadata = {}

        # Format metadata
        if 'format' in probe_data:
            format_data = probe_data['format']
            metadata['format'] = {
                'format_name': format_data.get('format_name', ''),
                'format_long_name': format_data.get('format_long_name', ''),
                'duration': float(format_data.get('duration', 0)),
                'bit_rate': int(format_data.get('bit_rate', 0)),
                'tags': format_data.get('tags', {})
            }

        # Stream metadata
        if 'streams' in probe_data:
            video_streams = []
            audio_streams = []

            for stream in probe_data['streams']:
                stream_type = stream.get('codec_type', '')

                if stream_type == 'video':
                    video_streams.append({
                        'codec': stream.get('codec_name', ''),
                        'codec_long_name': stream.get('codec_long_name', ''),
                        'width': stream.get('width', 0),
                        'height': stream.get('height', 0),
                        'bit_rate': stream.get('bit_rate', 0),
                        'fps': eval_fraction(stream.get('r_frame_rate', '0/1')),
                        'pixel_format': stream.get('pix_fmt', ''),
                        'profile': stream.get('profile', ''),
                        'level': stream.get('level', 0),
                        'tags': stream.get('tags', {})
                    })
                elif stream_type == 'audio':
                    audio_streams.append({
                        'codec': stream.get('codec_name', ''),
                        'codec_long_name': stream.get('codec_long_name', ''),
                        'sample_rate': stream.get('sample_rate', ''),
                        'channels': stream.get('channels', 0),
                        'channel_layout': stream.get('channel_layout', ''),
                        'bit_rate': stream.get('bit_rate', 0),
                        'tags': stream.get('tags', {})
                    })

            metadata['video_streams'] = video_streams
            metadata['audio_streams'] = audio_streams

        return metadata

    except Exception as e:
        logger.error(f"ffprobe metadata extraction failed: {str(e)}")
        return {}


def eval_fraction(fraction_str: str) -> float:
    """Evaluate fraction string (e.g., '30000/1001') to float.

    Args:
        fraction_str: Fraction as string

    Returns:
        Evaluated float value
    """
    try:
        if '/' in fraction_str:
            num, den = map(int, fraction_str.split('/'))
            return num / den if den != 0 else 0
        else:
            return float(fraction_str)
    except:
        return 0.0


def calculate_file_hash(file_path: str, algorithm: str = 'sha256', chunk_size: int = 8192) -> str:
    """Calculate hash of a file.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        chunk_size: Size of chunks to read

    Returns:
        Hex digest of hash
    """
    if algorithm == 'md5':
        hasher = hashlib.md5()
    elif algorithm == 'sha1':
        hasher = hashlib.sha1()
    else:
        hasher = hashlib.sha256()

    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            hasher.update(chunk)

    return hasher.hexdigest()


def extract_frames(video_path: str, output_dir: str,
                   frame_interval: int = 1,
                   max_frames: Optional[int] = None) -> List[str]:
    """Extract frames from video at specified interval.

    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every nth frame
        max_frames: Maximum number of frames to extract (None for all)

    Returns:
        List of paths to saved frame images

    Raises:
        VideoUtilsError: If frame extraction fails
    """
    if not os.path.exists(video_path):
        raise VideoUtilsError(f"Video file not found: {video_path}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoUtilsError(f"Failed to open video: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate number of frames to extract
        if max_frames is not None:
            extract_count = min(frame_count // frame_interval, max_frames)
        else:
            extract_count = frame_count // frame_interval

        extracted_paths = []
        processed_frames = 0
        frame_idx = 0

        # Extract frames
        while processed_frames < extract_count:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                extracted_paths.append(frame_path)
                processed_frames += 1

            frame_idx += 1

        cap.release()

        if not extracted_paths:
            raise VideoUtilsError("No frames were extracted")

        logger.info(f"Extracted {len(extracted_paths)} frames from {video_path}")
        return extracted_paths

    except Exception as e:
        logger.error(f"Frame extraction failed: {str(e)}")
        raise VideoUtilsError(f"Failed to extract frames: {str(e)}")


def get_video_frame_at_time(video_path: str, time_seconds: float) -> np.ndarray:
    """Extract a specific frame at given time.

    Args:
        video_path: Path to video file
        time_seconds: Time in seconds to extract frame from

    Returns:
        Frame as numpy array

    Raises:
        VideoUtilsError: If frame extraction fails
    """
    if not os.path.exists(video_path):
        raise VideoUtilsError(f"Video file not found: {video_path}")

    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoUtilsError(f"Failed to open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

        # Validate time
        if time_seconds < 0 or time_seconds >= duration:
            time_seconds = min(max(0, time_seconds), duration - 0.1)

        # Calculate frame number
        frame_number = int(time_seconds * fps)

        # Set position to requested frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise VideoUtilsError(f"Failed to read frame at time {time_seconds}s")

        return frame

    except Exception as e:
        logger.error(f"Frame extraction at time failed: {str(e)}")
        raise VideoUtilsError(f"Failed to extract frame at time {time_seconds}s: {str(e)}")


def create_video_from_frames(frame_paths: List[str],
                             output_path: str,
                             fps: float = 30.0,
                             codec: str = 'mp4v') -> str:
    """Create video from list of frame image paths.

    Args:
        frame_paths: List of paths to frame images
        output_path: Path to save output video
        fps: Frames per second for output video
        codec: FourCC codec code

    Returns:
        Path to created video

    Raises:
        VideoUtilsError: If video creation fails
    """
    if not frame_paths:
        raise VideoUtilsError("No frame paths provided")

    # Check if first frame exists
    if not os.path.exists(frame_paths[0]):
        raise VideoUtilsError(f"Frame file not found: {frame_paths[0]}")

    try:
        # Read first frame to get dimensions
        first_frame = cv2.imread(frame_paths[0])
        if first_frame is None:
            raise VideoUtilsError(f"Failed to read frame: {frame_paths[0]}")

        height, width, _ = first_frame.shape

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Write frames to video
        for frame_path in frame_paths:
            if not os.path.exists(frame_path):
                logger.warning(f"Frame not found, skipping: {frame_path}")
                continue

            frame = cv2.imread(frame_path)
            if frame is None:
                logger.warning(f"Failed to read frame, skipping: {frame_path}")
                continue

            # Ensure frame has correct dimensions
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))

            out.write(frame)

        out.release()

        if not os.path.exists(output_path):
            raise VideoUtilsError("Failed to create output video file")

        return output_path

    except Exception as e:
        logger.error(f"Video creation failed: {str(e)}")
        raise VideoUtilsError(f"Failed to create video from frames: {str(e)}")


def detect_scene_changes(video_path: str, threshold: float = 30.0,
                         min_scene_length: float = 0.5) -> List[float]:
    """Detect scene changes in a video.

    Args:
        video_path: Path to video file
        threshold: Difference threshold for scene change detection
        min_scene_length: Minimum scene length in seconds

    Returns:
        List of scene change timestamps in seconds

    Raises:
        VideoUtilsError: If scene detection fails
    """
    if not os.path.exists(video_path):
        raise VideoUtilsError(f"Video file not found: {video_path}")

    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoUtilsError(f"Failed to open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        min_frames_per_scene = int(min_scene_length * fps)

        # Initialize variables
        prev_frame = None
        scenes = []
        frame_count = 0
        last_scene_frame = -min_frames_per_scene  # Allow scene change at start

        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                # Calculate difference
                diff = cv2.absdiff(gray, prev_frame)
                mean_diff = cv2.mean(diff)[0]

                # Check if scene change
                if (mean_diff > threshold and
                        frame_count - last_scene_frame >= min_frames_per_scene):
                    time_sec = frame_count / fps
                    scenes.append(time_sec)
                    last_scene_frame = frame_count

            prev_frame = gray
            frame_count += 1

        cap.release()

        return scenes

    except Exception as e:
        logger.error(f"Scene detection failed: {str(e)}")
        raise VideoUtilsError(f"Failed to detect scenes: {str(e)}")


def compress_video(input_path: str, output_path: str,
                   target_size_mb: Optional[float] = None,
                   crf: int = 23,
                   preset: str = 'medium') -> str:
    """Compress video to reduce file size.

    Args:
        input_path: Path to input video
        output_path: Path to save compressed video
        target_size_mb: Target size in megabytes (approximate)
        crf: Constant Rate Factor (0-51, lower is better quality)
        preset: Encoding preset ('ultrafast', 'superfast', 'veryfast',
                'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow')

    Returns:
        Path to compressed video

    Raises:
        VideoUtilsError: If compression fails
    """
    if not os.path.exists(input_path):
        raise VideoUtilsError(f"Input video not found: {input_path}")

    try:
        # Get input video info
        input_size_mb = os.path.getsize(input_path) / (1024 * 1024)

        # Adjust CRF based on target size if specified
        if target_size_mb is not None:
            if target_size_mb >= input_size_mb:
                # No need to compress if target size is larger
                logger.info(
                    f"Target size ({target_size_mb:.2f}MB) >= input size ({input_size_mb:.2f}MB), using minimal compression")
                crf = 18
            else:
                # Calculate compression ratio
                ratio = input_size_mb / target_size_mb

                # Adjust CRF based on ratio (higher ratio = higher CRF)
                if ratio <= 1.5:
                    crf = 23
                elif ratio <= 2.5:
                    crf = 28
                elif ratio <= 4:
                    crf = 32
                else:
                    crf = min(38, int(23 + ratio / 2))

        # Ensure CRF is in valid range
        crf = max(0, min(51, crf))

        # Compress using ffmpeg
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-c:v', 'libx264',
            '-crf', str(crf),
            '-preset', preset,
            '-c:a', 'aac',
            '-b:a', '128k',
            '-y',  # Overwrite output
            output_path
        ]

        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if process.returncode != 0:
            raise VideoUtilsError(f"ffmpeg compression failed: {process.stderr}")

        if not os.path.exists(output_path):
            raise VideoUtilsError("Compression failed to create output file")

        # Report compression results
        output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        reduction_percent = 100 * (1 - output_size_mb / input_size_mb)

        logger.info(f"Compressed video: {input_size_mb:.2f}MB → {output_size_mb:.2f}MB " +
                    f"({reduction_percent:.1f}% reduction)")

        return output_path

    except Exception as e:
        logger.error(f"Video compression failed: {str(e)}")
        raise VideoUtilsError(f"Failed to compress video: {str(e)}")


def analyze_video_quality(video_path: str) -> Dict[str, Any]:
    """Analyze video quality metrics.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with quality metrics

    Raises:
        VideoUtilsError: If analysis fails
    """
    if not os.path.exists(video_path):
        raise VideoUtilsError(f"Video file not found: {video_path}")

    try:
        # Get basic metadata
        metadata = get_video_metadata(video_path)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoUtilsError(f"Failed to open video: {video_path}")

        # Initialize metrics
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_interval = max(1, frame_count // 50)  # Sample ~50 frames

        brightness_values = []
        contrast_values = []
        blur_values = []
        noise_values = []

        sampled_frames = 0
        frame_idx = 0

        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Brightness - mean pixel value
                brightness = cv2.mean(gray)[0]
                brightness_values.append(brightness)

                # Contrast - standard deviation of pixel values
                contrast = np.std(gray.astype(np.float32))
                contrast_values.append(contrast)

                # Blur - variance of Laplacian
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                blur = np.var(laplacian)
                blur_values.append(blur)

                # Noise - mean absolute deviation from median
                median_blurred = cv2.medianBlur(gray, 5)
                noise = np.mean(np.abs(gray.astype(np.float32) - median_blurred.astype(np.float32)))
                noise_values.append(noise)

                sampled_frames += 1

            frame_idx += 1

        cap.release()

        if sampled_frames == 0:
            raise VideoUtilsError("No frames were analyzed")

        # Calculate quality metrics
        avg_brightness = np.mean(brightness_values)
        avg_contrast = np.mean(contrast_values)
        avg_blur = np.mean(blur_values)
        avg_noise = np.mean(noise_values)

        # Normalized scores (0-100)
        brightness_score = min(100, max(0, 100 - abs(avg_brightness - 127) / 1.27))
        contrast_score = min(100, max(0, avg_contrast / 0.8))
        sharpness_score = min(100, max(0, avg_blur / 20))
        noise_score = min(100, max(0, 100 - avg_noise * 2))

        # Overall quality score
        quality_score = (brightness_score * 0.2 +
                         contrast_score * 0.3 +
                         sharpness_score * 0.3 +
                         noise_score * 0.2)

        return {
            'brightness': {
                'average': float(avg_brightness),
                'score': float(brightness_score),
                'samples': len(brightness_values)
            },
            'contrast': {
                'average': float(avg_contrast),
                'score': float(contrast_score),
                'samples': len(contrast_values)
            },
            'sharpness': {
                'average_laplacian_var': float(avg_blur),
                'score': float(sharpness_score),
                'samples': len(blur_values)
            },
            'noise': {
                'average': float(avg_noise),
                'score': float(noise_score),
                'samples': len(noise_values)
            },
            'overall_quality_score': float(quality_score),
            'sampling_info': {
                'frames_sampled': sampled_frames,
                'total_frames': frame_count,
                'sample_interval': sample_interval
            },
            'video_info': metadata['basic']
        }

    except Exception as e:
        logger.error(f"Video quality analysis failed: {str(e)}")
        raise VideoUtilsError(f"Failed to analyze video quality: {str(e)}")
