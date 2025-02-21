import asyncio
from typing import Dict, Any, AsyncGenerator, List
import numpy as np
import cv2
import torch
from ..features.enterprise_features import EnterpriseFeatureExtractor
from ..evaluation.enterprise_metrics import EnterpriseMetrics


class RealtimeAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_extractor = EnterpriseFeatureExtractor(config)
        self.metrics = EnterpriseMetrics()
        self.buffer_size = config.get('buffer_size', 30)
        self.min_confidence = config.get('min_confidence', 0.7)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    async def process_stream(self, video_stream) -> AsyncGenerator[Dict[str, Any], None]:
        buffer = []
        metrics_history = []

        async for frame in video_stream:
            buffer.append(frame)

            if len(buffer) >= self.buffer_size:
                features = await self._process_buffer(buffer)
                scores = await self._evaluate_segment(features)
                metrics = self.metrics.evaluate_teaching_quality(scores)

                if self._is_confident(metrics):
                    metrics_history.append(metrics)
                    smoothed_metrics = self._smooth_metrics(metrics_history)
                    yield {
                        'current': metrics,
                        'trend': smoothed_metrics,
                        'warnings': self._generate_warnings(metrics),
                        'suggestions': self._generate_realtime_suggestions(metrics)
                    }

                buffer = buffer[self.buffer_size // 2:]

    async def _process_buffer(self, buffer: List[np.ndarray]) -> Dict[str, torch.Tensor]:
        features = []
        for frame in buffer:
            frame_features = await self._process_frame(frame)
            features.append(frame_features)

        return {
            'visual': torch.stack([f['visual'] for f in features]),
            'audio': torch.stack([f['audio'] for f in features]),
            'facial': torch.stack([f['facial'] for f in features])
        }

    async def _process_frame(self, frame: np.ndarray) -> Dict[str, torch.Tensor]:
        return await asyncio.gather(
            self._extract_visual_features(frame),
            self._extract_audio_features(frame),
            self._extract_facial_features(frame)
        )

    async def _evaluate_segment(self, features: Dict[str, torch.Tensor]) -> Dict[str, float]:
        with torch.no_grad():
            visual_scores = self.feature_extractor.evaluate_visual(features['visual'])
            audio_scores = self.feature_extractor.evaluate_audio(features['audio'])
            facial_scores = self.feature_extractor.evaluate_facial(features['facial'])

            return {
                **visual_scores,
                **audio_scores,
                **facial_scores
            }

    def _is_confident(self, metrics: Dict[str, Any]) -> bool:
        confidence_scores = [
            score for score in metrics.values()
            if isinstance(score, (int, float))
        ]
        return np.mean(confidence_scores) >= self.min_confidence

    def _smooth_metrics(self, metrics_history: List[Dict[str, Any]]) -> Dict[str, float]:
        if len(metrics_history) < 2:
            return metrics_history[-1]

        window_size = min(5, len(metrics_history))
        recent_metrics = metrics_history[-window_size:]

        smoothed = {}
        for key in recent_metrics[0].keys():
            if isinstance(recent_metrics[0][key], (int, float)):
                values = [m[key] for m in recent_metrics]
                smoothed[key] = np.mean(values)

        return smoothed

    def _generate_warnings(self, metrics: Dict[str, Any]) -> List[str]:
        warnings = []
        thresholds = self.config.get('warning_thresholds', {})

        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                threshold = thresholds.get(metric, 0.6)
                if value < threshold:
                    warnings.append(f"Low {metric.replace('_', ' ')}: {value:.2f}")

        return warnings

    def _generate_realtime_suggestions(self, metrics: Dict[str, Any]) -> List[str]:
        suggestions = []
        if metrics.get('engagement', 0) < 0.7:
            suggestions.append("Try increasing interaction with audience")
        if metrics.get('clarity', 0) < 0.7:
            suggestions.append("Speak more slowly and clearly")
        if metrics.get('visual_engagement', 0) < 0.7:
            suggestions.append("Maintain more eye contact")
        return suggestions[:3]