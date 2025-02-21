import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import cv2
import numpy as np
from .pipeline import DataPipeline
from ..data.augmentation import VideoAugmentation

class VideoDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir: str,
            labels_file: str,
            transform: Optional[VideoAugmentation] = None,
            split: str = 'train'
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        self.pipeline = DataPipeline({})

        # Load labels and metadata
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)

        # Filter videos by split
        self.video_paths = []
        self.video_labels = []
        for video_id, data in self.labels.items():
            if data['split'] == split:
                video_path = self.data_dir / f"{video_id}.mp4"
                if video_path.exists():
                    self.video_paths.append(video_path)
                    self.video_labels.append(data['labels'])

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_path = str(self.video_paths[idx])
        labels = self.video_labels[idx]

        # Extract features
        features = self.pipeline.process_video(video_path)

        # Apply augmentations if in training
        if self.split == 'train' and self.transform:
            features = self.transform(features)

        return {
            'features': features,
            'labels': labels
        }

    def get_label_distribution(self) -> Dict[str, float]:
        """Calculate distribution of labels in dataset"""
        label_counts = {}
        for labels in self.video_labels:
            for key, value in labels.items():
                if key not in label_counts:
                    label_counts[key] = []
                label_counts[key].append(value)

        return {
            key: {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
            for key, values in label_counts.items()
        }
