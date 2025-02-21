import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple
from .preprocessor import VideoPreprocessor

class VideoDataset(Dataset):
    def __init__(self, video_paths: List[str], labels: List[Dict[str, float]], config: Dict[str, Any]):
        self.video_paths = video_paths
        self.labels = labels
        self.preprocessor = VideoPreprocessor(
            frame_size=config.get('frame_size', (224, 224)),
            sample_rate=config.get('sample_rate', 30)
        )

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        features = self.preprocessor.process_video(self.video_paths[idx])
        return features, self.labels[idx]

def create_data_loaders(
    train_data: List[Tuple[str, Dict[str, float]]],
    val_data: List[Tuple[str, Dict[str, float]]],
    config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader]:
    train_paths, train_labels = zip(*train_data)
    val_paths, val_labels = zip(*val_data)

    train_dataset = VideoDataset(train_paths, train_labels, config)
    val_dataset = VideoDataset(val_paths, val_labels, config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        num_workers=config.get('num_workers', 4)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        num_workers=config.get('num_workers', 4)
    )

    return train_loader, val_loader