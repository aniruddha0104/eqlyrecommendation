from typing import Dict, Any, List
import numpy as np
import json
from pathlib import Path


class EnterpriseMetrics:
    def __init__(self, config_path: str = "config/enterprise_benchmarks.json"):
        self.confidence_threshold = 0.85
        self.benchmarks = self._load_benchmarks(config_path)
        self.industry_weights = {
            'technical': 0.4,
            'communication': 0.3,
            'engagement': 0.2,
            'adaptability': 0.1
        }

    def _load_benchmarks(self, config_path: str) -> Dict[str, Any]:
        with open(config_path) as f:
            return json.load(f)

    def evaluate_teaching_quality(self, scores: Dict[str, float]) -> Dict[str, Any]:
        technical_scores = self._evaluate_technical_depth(scores)
        communication_scores = self._evaluate_communication(scores)
        engagement_scores = self._evaluate_engagement(scores)
        comparison = self._compare_to_benchmarks(scores)

        overall_score = (
                technical_scores['aggregate'] * self.industry_weights['technical'] +
                communication_scores['aggregate'] * self.industry_weights['communication'] +
                engagement_scores['aggregate'] * self.industry_weights['engagement']
        )

        return {
            'technical_proficiency': technical_scores,
            'communication_skills': communication_scores,
            'engagement_metrics': engagement_scores,
            'industry_comparison': comparison,
            'overall_score': overall_score,
            'recommendation': self._generate_recommendation(scores, overall_score),
            'improvement_areas': self._identify_improvement_areas(scores)
        }

    def _evaluate_technical_depth(self, scores: Dict[str, float]) -> Dict[str, float]:
        technical_metrics = {
            'concept_mastery': scores['content_mastery'],
            'explanation_clarity': scores['explanation_clarity'],
            'technical_accuracy': scores.get('accuracy', 0.0),
            'depth_of_knowledge': scores.get('knowledge_depth', 0.0),
            'problem_solving': scores.get('problem_solving', 0.0)
        }

        return {
            **technical_metrics,
            'aggregate': np.mean(list(technical_metrics.values()))
        }

    def _evaluate_communication(self, scores: Dict[str, float]) -> Dict[str, float]:
        communication_metrics = {
            'clarity': scores.get('clarity', 0.0),
            'structure': scores.get('structure', 0.0),
            'pace': scores.get('speaking_pace', 0.0),
            'articulation': scores.get('articulation', 0.0),
            'audience_adaptation': scores.get('adaptation', 0.0)
        }

        return {
            **communication_metrics,
            'aggregate': np.mean(list(communication_metrics.values()))
        }

    def _evaluate_engagement(self, scores: Dict[str, float]) -> Dict[str, float]:
        engagement_metrics = {
            'student_interaction': scores.get('interaction', 0.0),
            'enthusiasm': scores.get('enthusiasm', 0.0),
            'attention_retention': scores.get('attention', 0.0),
            'visual_engagement': scores.get('visual_engagement', 0.0),
            'emotional_connection': scores.get('emotional_connection', 0.0)
        }

        return {
            **engagement_metrics,
            'aggregate': np.mean(list(engagement_metrics.values()))
        }

    def _compare_to_benchmarks(self, scores: Dict[str, float]) -> Dict[str, Any]:
        industry_avg = self.benchmarks['industry_average']
        top_performers = self.benchmarks['top_performers']

        return {
            'percentile': self._calculate_percentile(scores),
            'industry_gap': {
                k: scores.get(k, 0.0) - industry_avg.get(k, 0.0)
                for k in scores.keys()
            },
            'top_performer_gap': {
                k: scores.get(k, 0.0) - top_performers.get(k, 0.0)
                for k in scores.keys()
            }
        }

    def _calculate_percentile(self, scores: Dict[str, float]) -> float:
        overall = np.mean(list(scores.values()))
        distribution = self.benchmarks['score_distribution']
        return np.searchsorted(distribution, overall) / len(distribution) * 100

    def _generate_recommendation(self, scores: Dict[str, float], overall_score: float) -> Dict[str, Any]:
        improvements = self._identify_improvement_areas(scores)

        return {
            'summary': self._get_recommendation_summary(overall_score),
            'priority_areas': improvements[:3],
            'suggested_resources': self._get_learning_resources(improvements),
            'next_steps': self._generate_action_plan(improvements)
        }

    def _identify_improvement_areas(self, scores: Dict[str, float]) -> List[Dict[str, Any]]:
        areas = []
        for metric, score in scores.items():
            if score < self.benchmarks['industry_average'].get(metric, 0.8):
                areas.append({
                    'metric': metric,
                    'current_score': score,
                    'gap': self.benchmarks['industry_average'].get(metric, 0.8) - score,
                    'priority': 'high' if score < 0.6 else 'medium'
                })

        return sorted(areas, key=lambda x: x['gap'], reverse=True)

    def _get_recommendation_summary(self, overall_score: float) -> str:
        if overall_score >= 0.9:
            return "Expert level performance - Focus on innovation and mentoring"
        elif overall_score >= 0.8:
            return "Strong performance - Target specific areas for excellence"
        elif overall_score >= 0.7:
            return "Good performance - Address key improvement areas"
        else:
            return "Development needed - Focus on fundamental improvements"

    def _get_learning_resources(self, improvements: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        resources = []
        for area in improvements:
            metric = area['metric']
            if metric in self.benchmarks['learning_resources']:
                resources.extend(self.benchmarks['learning_resources'][metric])
        return resources[:5]

    def _generate_action_plan(self, improvements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        action_plan = []
        for area in improvements:
            action_plan.append({
                'focus_area': area['metric'],
                'target_score': min(area['current_score'] + 0.2, 1.0),
                'timeframe': '2 weeks' if area['priority'] == 'high' else '4 weeks',
                'action_items': self.benchmarks['improvement_actions'].get(area['metric'], [])
            })
        return action_plan
