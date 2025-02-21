# test/components/data_processing.py

import cv2
import numpy as np
import torch
from pathlib import Path
import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from torch.utils.data import Dataset, DataLoader


class VideoDataset(Dataset):
    def __init__(self, video_path: str, config: Dict[str, Any]):
        self.video_path = video_path
        self.config = config
        self.frames = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self.load_video()

    def load_video(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise Exception(f"Failed to open video: {self.video_path}")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                self.frames.append(frame)
            cap.release()

        except Exception as e:
            self.logger.error(f"Failed to load video: {str(e)}")
            raise

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return frame


class FeatureExtractor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup_extractors()

    def setup_extractors(self):
        try:
            self.sift = cv2.SIFT_create()
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize feature extractors: {str(e)}")
            raise

    def extract_features(self, frame: np.ndarray) -> Dict[str, int]:
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Detect SIFT features
            keypoints = self.sift.detect(gray, None)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

            return {
                'keypoints': int(len(keypoints)),  # Ensure int type
                'faces': int(len(faces))  # Ensure int type
            }

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            return {'keypoints': 0, 'faces': 0}


class DataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.feature_extractor = FeatureExtractor(config)
        self.metrics = {
            'processing_times': [],
            'features': [],
            'total_frames': 0
        }
        self.is_processing = False

    def start(self):
        self.is_processing = True
        self.logger.info("Processing started")

    def stop(self):
        self.is_processing = False
        self.logger.info("Processing stopped")

    def process_data(self, frame: torch.Tensor) -> Dict[str, Any]:
        try:
            start_time = time.time()

            # Convert tensor to numpy
            if isinstance(frame, torch.Tensor):
                frame = (frame.numpy() * 255).astype(np.uint8)
                if frame.shape[0] == 3:  # CHW to HWC
                    frame = np.transpose(frame, (1, 2, 0))

            # Extract features
            features = self.feature_extractor.extract_features(frame)

            # Calculate processing time
            processing_time = float(time.time() - start_time)  # Ensure float type

            # Update metrics
            self.metrics['processing_times'].append(processing_time)
            self.metrics['features'].append(features)
            self.metrics['total_frames'] += 1

            return {
                'features': features,
                'processing_time': processing_time
            }

        except Exception as e:
            self.logger.error(f"Data processing failed: {str(e)}")
            return {'error': str(e)}

    def get_metrics(self) -> Dict[str, Any]:
        try:
            features_list = self.metrics['features']
            metrics = {
                'total_frames': int(self.metrics['total_frames']),
                'average_processing_time': float(np.mean(self.metrics['processing_times'])),
                'max_processing_time': float(np.max(self.metrics['processing_times'])),
                'average_keypoints': float(np.mean([f['keypoints'] for f in features_list])),
                'average_faces': float(np.mean([f['faces'] for f in features_list])),
                'max_keypoints': int(np.max([f['keypoints'] for f in features_list])),
                'max_faces': int(np.max([f['faces'] for f in features_list])),
                'total_features': int(sum(f['keypoints'] for f in features_list))
            }
            return metrics
        except Exception as e:
            self.logger.error(f"Failed to calculate metrics: {str(e)}")
            return {'error': str(e)}


def process_video(video_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process video file"""
    logger = logging.getLogger(__name__)

    try:
        # Initialize processor
        processor = DataProcessor(config)
        processor.start()

        # Create dataset and dataloader
        dataset = VideoDataset(video_path, config)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Process frames
        total_frames = len(dataset)
        logger.info(f"Starting to process {total_frames} frames")

        for i, frame in enumerate(dataloader):
            processor.process_data(frame.squeeze(0))
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{total_frames} frames")

        processor.stop()

        # Get metrics
        metrics = processor.get_metrics()
        return metrics

    except Exception as e:
        logger.error(f"Video processing failed: {str(e)}")
        return {'error': str(e)}


def save_metrics(metrics: Dict[str, Any], file_path: Path):
    """Save metrics with proper type conversion"""
    converted_metrics = {}

    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            converted_metrics[key] = value.item()  # Convert numpy types to Python types
        elif isinstance(value, np.ndarray):
            converted_metrics[key] = value.tolist()  # Convert arrays to lists
        else:
            converted_metrics[key] = value

    with open(file_path, 'w') as f:
        json.dump(converted_metrics, f, indent=2)


def main():
    """Main function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        # Get project root directory
        root_dir = Path(__file__).parent.parent.parent

        # Configuration
        config = {
            'target_size': (224, 224),
            'normalize': True,
            'batch_size': 1
        }

        # Setup paths
        video_path = root_dir / 'test_data' / 'test_video.mp4'
        results_dir = root_dir / 'results'
        results_dir.mkdir(exist_ok=True)

        if video_path.exists():
            logger.info(f"Processing video: {video_path}")
            metrics = process_video(str(video_path), config)
        else:
            logger.info(f"Video file not found at: {video_path}")
            logger.info("Processing dummy frames instead")

            # Process dummy frames
            processor = DataProcessor(config)
            processor.start()

            num_frames = 100
            for _ in range(num_frames):
                dummy_frame = torch.randn(3, 224, 224)
                processor.process_data(dummy_frame)

            processor.stop()
            metrics = processor.get_metrics()

        # Save results with proper type conversion
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        results_file = results_dir / f'processing_results_{timestamp}.json'
        save_metrics(metrics, results_file)

        logger.info(f"Results saved to: {results_file}")

        # Print summary
        print("\nProcessing Results:")
        print("=" * 50)
        if 'error' in metrics:
            print(f"Processing Error: {metrics['error']}")
        else:
            for key, value in metrics.items():
                if isinstance(value, (float, np.floating)):
                    print(f"{key}: {value:.2f}")
                else:
                    print(f"{key}: {value}")
        print("=" * 50)

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()