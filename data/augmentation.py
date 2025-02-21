import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Union
import random


class VideoAugmentation:
    def __init__(self, config: Dict):
        self.config = config
        self.enabled_augmentations = config.get('enabled_augmentations', [
            'random_flip',
            'random_rotate',
            'adjust_brightness',
            'adjust_contrast',
            'temporal_crop',
            'spatial_crop'
        ])

    def __call__(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        augmented_features = features.copy()

        for aug_name in self.enabled_augmentations:
            if random.random() < self.config.get(f'{aug_name}_prob', 0.5):
                aug_method = getattr(self, aug_name)
                augmented_features = aug_method(augmented_features)

        return augmented_features

    def random_flip(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Horizontal flip of video frames"""
        if 'frames' in features:
            features['frames'] = torch.flip(features['frames'], [-1])
            if 'face_features' in features:
                features['face_features'] = torch.flip(features['face_features'], [-1])
        return features

    def random_rotate(
            self,
            features: Dict[str, torch.Tensor],
            angle_range: float = 15.0
    ) -> Dict[str, torch.Tensor]:
        """Random rotation within angle range"""
        angle = random.uniform(-angle_range, angle_range)
        if 'frames' in features:
            features['frames'] = self._rotate_tensor(features['frames'], angle)
            if 'face_features' in features:
                features['face_features'] = self._rotate_tensor(features['face_features'], angle)
        return features

    def adjust_brightness(
            self,
            features: Dict[str, torch.Tensor],
            factor_range: Tuple[float, float] = (0.8, 1.2)
    ) -> Dict[str, torch.Tensor]:
        """Adjust brightness of video frames"""
        factor = random.uniform(factor_range[0], factor_range[1])
        if 'frames' in features:
            features['frames'] = torch.clamp(features['frames'] * factor, 0, 1)
        return features

    def adjust_contrast(
            self,
            features: Dict[str, torch.Tensor],
            factor_range: Tuple[float, float] = (0.8, 1.2)
    ) -> Dict[str, torch.Tensor]:
        """Adjust contrast of video frames"""
        factor = random.uniform(factor_range[0], factor_range[1])
        if 'frames' in features:
            mean = features['frames'].mean(dim=[2, 3], keepdim=True)
            features['frames'] = torch.clamp((features['frames'] - mean) * factor + mean, 0, 1)
        return features

    def temporal_crop(
            self,
            features: Dict[str, torch.Tensor],
            min_frames: int = 16
    ) -> Dict[str, torch.Tensor]:
        """Random temporal crop of video"""
        if 'frames' in features:
            num_frames = features['frames'].size(0)
            if num_frames > min_frames:
                start_idx = random.randint(0, num_frames - min_frames)
                end_idx = start_idx + min_frames
                features['frames'] = features['frames'][start_idx:end_idx]

                # Adjust other temporal features
                temporal_keys = ['face_features', 'body_features', 'eye_tracking']
                for key in temporal_keys:
                    if key in features:
                        features[key] = features[key][start_idx:end_idx]
        return features

    def spatial_crop(
            self,
            features: Dict[str, torch.Tensor],
            crop_size: Tuple[int, int] = (200, 200)
    ) -> Dict[str, torch.Tensor]:
        """Random spatial crop of video frames"""
        if 'frames' in features:
            _, h, w, _ = features['frames'].shape
            top = random.randint(0, h - crop_size[0])
            left = random.randint(0, w - crop_size[1])

            features['frames'] = features['frames'][:, top:top + crop_size[0], left:left + crop_size[1], :]

            if 'face_features' in features:
                features['face_features'] = F.interpolate(
                    features['face_features'],
                    size=crop_size,
                    mode='bilinear',
                    align_corners=True
                )
        return features

    @staticmethod
    def _rotate_tensor(x: torch.Tensor, angle: float) -> torch.Tensor:
        """Rotate tensor by angle degrees"""
        return F.rotate(x, angle)