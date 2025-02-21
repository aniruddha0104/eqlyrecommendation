# eqly_assessment/core/metrics.py
from typing import Dict, List, Optional
import numpy as np
from typing import Dict, List, Optional, Any
from ..models.enterprise import EnterpriseConfig


class EnterpriseMetricsCalculator:
    def __init__(self, config: EnterpriseConfig):
        self.config = config
        self.weights = config.assessment.teacher_weights

    def calculate_metrics(
            self,
            audio_features: Dict,
            video_features: Dict
    ) -> Dict[str, float]:
        teaching_scores = self._calculate_teaching_scores(audio_features, video_features)
        engagement_scores = self._calculate_engagement_scores(audio_features, video_features)
        effectiveness_scores = self._calculate_effectiveness_scores(teaching_scores, engagement_scores)

        return {
            'teaching': teaching_scores,
            'engagement': engagement_scores,
            'effectiveness': effectiveness_scores,
            'overall': self._calculate_overall_score(teaching_scores, engagement_scores)
        }

    def _calculate_teaching_scores(
            self,
            audio_features: Dict,
            video_features: Dict
    ) -> Dict[str, float]:
        return {
            'content_mastery': self._evaluate_content_mastery(audio_features),
            'explanation_clarity': self._evaluate_explanation_clarity(audio_features, video_features),
            'structure': self._evaluate_structure(audio_features),
            'technical_accuracy': self._evaluate_technical_accuracy(audio_features)
        }

    def _calculate_engagement_scores(
            self,
            audio_features: Dict,
            video_features: Dict
    ) -> Dict[str, float]:
        return {
            'vocal_engagement': self._evaluate_vocal_engagement(audio_features),
            'visual_engagement': self._evaluate_visual_engagement(video_features),
            'interaction_quality': self._evaluate_interaction_quality(audio_features, video_features),
            'audience_connection': self._evaluate_audience_connection(video_features)
        }

    def _calculate_effectiveness_scores(
            self,
            teaching_scores: Dict[str, float],
            engagement_scores: Dict[str, float]
    ) -> Dict[str, float]:
        return {
            'knowledge_transfer': self._evaluate_knowledge_transfer(teaching_scores),
            'retention_potential': self._evaluate_retention_potential(teaching_scores, engagement_scores),
            'practical_application': self._evaluate_practical_application(teaching_scores)
        }

    def _calculate_overall_score(
            self,
            teaching_scores: Dict[str, float],
            engagement_scores: Dict[str, float]
    ) -> float:
        teaching_avg = np.mean(list(teaching_scores.values()))
        engagement_avg = np.mean(list(engagement_scores.values()))
        return 0.7 * teaching_avg + 0.3 * engagement_avg


# eqly_assessment/core/benchmarks.py
class EnterpriseBenchmarks:
    def __init__(self, config: EnterpriseConfig):
        self.config = config
        self.industry_benchmarks = self._load_industry_benchmarks()
        self.percentile_thresholds = self._load_percentile_thresholds()

    def compare_to_benchmarks(
            self,
            metrics: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        return {
            'industry_comparison': self._compare_to_industry(metrics),
            'percentile_ranking': self._calculate_percentiles(metrics),
            'improvement_potential': self._calculate_improvement_potential(metrics)
        }

    def _compare_to_industry(
            self,
            metrics: Dict[str, float]
    ) -> Dict[str, float]:
        return {
            metric: (value - self.industry_benchmarks.get(metric, 0))
            for metric, value in metrics.items()
        }

    def _calculate_percentiles(
            self,
            metrics: Dict[str, float]
    ) -> Dict[str, float]:
        return {
            metric: self._get_percentile_rank(metric, value)
            for metric, value in metrics.items()
        }


# eqly_assessment/core/reporting.py
class EnterpriseReporting:
    def __init__(self, config: EnterpriseConfig):
        self.config = config
        self.metrics_calculator = EnterpriseMetricsCalculator(config)
        self.benchmarks = EnterpriseBenchmarks(config)

    def generate_report(
            self,
            assessment_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        metrics = self.metrics_calculator.calculate_metrics(
            assessment_results['audio_features'],
            assessment_results['video_features']
        )

        benchmarks = self.benchmarks.compare_to_benchmarks(metrics)

        return {
            'metrics': metrics,
            'benchmarks': benchmarks,
            'insights': self._generate_insights(metrics, benchmarks),
            'recommendations': self._generate_recommendations(metrics, benchmarks),
            'improvement_plan': self._create_improvement_plan(metrics, benchmarks)
        }

    def _generate_insights(
            self,
            metrics: Dict[str, float],
            benchmarks: Dict[str, Dict[str, float]]
    ) -> List[str]:
        insights = []

        if metrics['teaching']['content_mastery'] > 0.8:
            insights.append("Demonstrates strong subject matter expertise")

        if metrics['engagement']['audience_connection'] < 0.6:
            insights.append("Opportunity to improve audience engagement")

        return insights

    def _generate_recommendations(
            self,
            metrics: Dict[str, float],
            benchmarks: Dict[str, Dict[str, float]]
    ) -> List[Dict[str, str]]:
        recommendations = []

        for metric, score in metrics['teaching'].items():
            if score < 0.7:
                recommendations.append({
                    'area': metric,
                    'recommendation': self._get_recommendation_for_metric(metric),
                    'priority': 'high' if score < 0.5 else 'medium'
                })

        return recommendations