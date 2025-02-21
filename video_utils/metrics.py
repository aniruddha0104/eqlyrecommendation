import torch
import numpy as np
from typing import Dict, Any

def calculate_understanding_ratio(
    teacher_scores: Dict[str, float],
    learner_scores: Dict[str, float]
) -> float:
    teacher_avg = np.mean(list(teacher_scores.values()))
    learner_avg = np.mean(list(learner_scores.values()))
    return learner_avg / teacher_avg if teacher_avg > 0 else 0

def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    return {
        'mse': torch.mean((predictions - targets) ** 2).item(),
        'mae': torch.mean(torch.abs(predictions - targets)).item(),
        'accuracy': torch.mean((torch.abs(predictions - targets) < 0.1).float()).item()
    }

def log_metrics(metrics: Dict[str, float], step: int, prefix: str = '') -> None:
    metrics_str = ' | '.join(f'{k}: {v:.4f}' for k, v in metrics.items())
    print(f'{prefix} Step {step}: {metrics_str}')