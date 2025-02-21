# eqly_assessment/evaluation/metrics.py
import torch
import numpy as np
from typing import Dict, List
from sklearn.metrics import roc_auc_score, precision_recall_curve
from torch.utils.data import DataLoader


class EvaluationMetrics:
    def __init__(self, config: Dict):
        self.config = config
        self.metrics_history = []

    def calculate_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        metrics = {
            'teaching_metrics': self._calculate_teaching_metrics(predictions, targets),
            'engagement_metrics': self._calculate_engagement_metrics(predictions, targets),
            'overall_metrics': self._calculate_overall_metrics(predictions, targets)
        }
        self.metrics_history.append(metrics)
        return metrics

    def _calculate_teaching_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        return {
            'content_accuracy': self._compute_accuracy(
                predictions['content'],
                targets['content']
            ),
            'explanation_clarity': self._compute_clarity_score(
                predictions['explanation'],
                targets['explanation']
            ),
            'structure_score': self._compute_structure_score(
                predictions['structure'],
                targets['structure']
            )
        }

    def _calculate_engagement_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        return {
            'visual_engagement': self._compute_engagement_score(
                predictions['visual'],
                targets['visual']
            ),
            'verbal_engagement': self._compute_engagement_score(
                predictions['verbal'],
                targets['verbal']
            ),
            'interaction_quality': self._compute_interaction_score(
                predictions['interaction'],
                targets['interaction']
            )
        }

    def _calculate_overall_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        return {
            'teaching_effectiveness': self._compute_effectiveness_score(
                predictions,
                targets
            ),
            'knowledge_transfer': self._compute_transfer_score(
                predictions,
                targets
            )
        }

# eqly_assessment/evaluation/evaluator.py
class ModelEvaluator:
    def __init__(self, model, config: Dict):
        self.model = model
        self.config = config
        self.metrics = EvaluationMetrics(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in eval_loader:
                features = batch['features'].to(self.device)
                targets = batch['labels'].to(self.device)

                predictions = self.model(features)
                all_predictions.append(predictions)
                all_targets.append(targets)

        return self.metrics.calculate_metrics(
            self._concatenate_predictions(all_predictions),
            self._concatenate_predictions(all_targets)
        )

    def _concatenate_predictions(self, predictions: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        concatenated = {}
        for key in predictions[0].keys():
            concatenated[key] = torch.cat([p[key] for p in predictions])
        return concatenated

# eqly_assessment/evaluation/progress_tracker.py
class ProgressTracker:
    def __init__(self, config: Dict):
        self.config = config
        self.history = []
        self.best_metrics = {}

    def update(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """Update tracking and return True if model improved"""
        self.history.append({
            'epoch': epoch,
            'metrics': metrics
        })

        improved = False
        for metric_name, value in metrics.items():
            if metric_name not in self.best_metrics or self._is_better(metric_name, value):
                self.best_metrics[metric_name] = value
                improved = True

        return improved

    def _is_better(self, metric_name: str, value: float) -> bool:
        """Check if new value is better than current best"""
        if metric_name not in self.best_metrics:
            return True
        # Lower is better for loss metrics
        if 'loss' in metric_name:
            return value < self.best_metrics[metric_name]
        # Higher is better for other metrics
        return value > self.best_metrics[metric_name]

    def get_summary(self) -> Dict:
        """Get training progress summary"""
        return {
            'best_metrics': self.best_metrics,
            'latest_metrics': self.history[-1]['metrics'] if self.history else None,
            'num_epochs': len(self.history),
            'improvement_trend': self._calculate_improvement_trend()
        }

    def _calculate_improvement_trend(self) -> Dict[str, float]:
        """Calculate improvement trends for each metric"""
        if len(self.history) < 2:
            return {}

        trends = {}
        for metric in self.history[0]['metrics'].keys():
            values = [h['metrics'][metric] for h in self.history]
            trends[metric] = (values[-1] - values[0]) / len(values)

        return trends