# test/components/metrics_reporter.py

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import logging
from datetime import datetime


class MetricsReporter:
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.results_dir = Path(config.get('results_dir', 'test_results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, metrics: Dict[str, Any], tag: str = None):
        """Generate comprehensive test report"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_dir = self.results_dir / f"report_{timestamp}"
            report_dir.mkdir(parents=True, exist_ok=True)

            # Save raw metrics
            self._save_raw_metrics(metrics, report_dir)

            # Generate and save summary
            summary = self._generate_summary(metrics)
            self._save_summary(summary, report_dir)

            # Save detailed analysis
            analysis = self._generate_analysis(metrics)
            self._save_analysis(analysis, report_dir)

            return str(report_dir)

        except Exception as e:
            self.logger.error(f"Failed to generate report: {str(e)}")
            raise

    def _save_raw_metrics(self, metrics: Dict[str, Any], report_dir: Path):
        """Save raw metrics data"""
        metrics_file = report_dir / 'raw_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

    def _generate_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of metrics"""
        return {
            'performance_summary': self._summarize_performance(metrics.get('performance', {})),
            'resource_summary': self._summarize_resources(metrics.get('resources', {})),
            'model_summary': self._summarize_model_metrics(metrics.get('model_metrics', {}))
        }

    def _summarize_performance(self, perf_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize performance metrics"""
        return {
            'fps': perf_metrics.get('fps', 0),
            'average_latency': perf_metrics.get('average_latency', 0),
            'throughput': perf_metrics.get('throughput', 0)
        }

    def _summarize_resources(self, resource_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize resource metrics"""
        return {
            'cpu_usage': resource_metrics.get('cpu_usage', 0),
            'memory_usage': resource_metrics.get('memory_usage', 0),
            'gpu_usage': resource_metrics.get('gpu_usage', 0)
        }

    def _summarize_model_metrics(self, model_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize model metrics"""
        if not model_metrics:
            return {}

        teacher_metrics = model_metrics.get('teacher_model', {}).get('scores', {})
        learner_metrics = model_metrics.get('learner_model', {}).get('scores', {})

        return {
            'teacher_model': {
                'average_score': np.mean(list(teacher_metrics.values())) if teacher_metrics else 0
            },
            'learner_model': {
                'average_score': np.mean(list(learner_metrics.values())) if learner_metrics else 0
            }
        }

    def _save_summary(self, summary: Dict[str, Any], report_dir: Path):
        """Save summary report"""
        summary_file = report_dir / 'summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

    def _generate_analysis(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed analysis"""
        return {
            'performance_analysis': self._analyze_performance(metrics.get('performance', {})),
            'resource_analysis': self._analyze_resources(metrics.get('resources', {})),
            'model_analysis': self._analyze_model_metrics(metrics.get('model_metrics', {}))
        }

    def _analyze_performance(self, perf_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance metrics"""
        fps = perf_metrics.get('fps', 0)
        target_fps = self.config.get('target_fps', 30)

        return {
            'fps_analysis': {
                'current': fps,
                'target': target_fps,
                'achievement': (fps / target_fps * 100) if target_fps > 0 else 0,
                'status': 'good' if fps >= target_fps else 'needs_improvement'
            },
            'latency_analysis': {
                'current': perf_metrics.get('average_latency', 0),
                'status': self._get_latency_status(perf_metrics.get('average_latency', 0))
            }
        }

    def _analyze_resources(self, resource_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resource metrics"""
        return {
            'cpu': {
                'usage': resource_metrics.get('cpu_usage', 0),
                'status': self._get_resource_status(
                    resource_metrics.get('cpu_usage', 0),
                    'cpu'
                )
            },
            'memory': {
                'usage': resource_metrics.get('memory_usage', 0),
                'status': self._get_resource_status(
                    resource_metrics.get('memory_usage', 0),
                    'memory'
                )
            },
            'gpu': {
                'usage': resource_metrics.get('gpu_usage', 0),
                'status': self._get_resource_status(
                    resource_metrics.get('gpu_usage', 0),
                    'gpu'
                )
            } if resource_metrics.get('gpu_usage') is not None else None
        }

    def _analyze_model_metrics(self, model_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model metrics"""
        if not model_metrics:
            return {}

        return {
            'teacher_model': self._analyze_model_performance(
                model_metrics.get('teacher_model', {})
            ),
            'learner_model': self._analyze_model_performance(
                model_metrics.get('learner_model', {})
            )
        }

    def _analyze_model_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze individual model performance"""
        scores = metrics.get('scores', {})
        if not scores:
            return {}

        avg_score = np.mean(list(scores.values()))
        return {
            'average_score': avg_score,
            'status': 'good' if avg_score >= 0.8 else 'needs_improvement',
            'metrics': scores
        }

    def _get_latency_status(self, latency: float) -> str:
        """Get status based on latency"""
        if latency <= 30:
            return 'good'
        elif latency <= 50:
            return 'acceptable'
        else:
            return 'needs_improvement'

    def _get_resource_status(self, usage: float, resource_type: str) -> str:
        """Get status based on resource usage"""
        thresholds = {
            'cpu': {'warning': 80, 'critical': 90},
            'memory': {'warning': 85, 'critical': 95},
            'gpu': {'warning': 85, 'critical': 95}
        }

        threshold = thresholds.get(resource_type, {'warning': 80, 'critical': 90})

        if usage >= threshold['critical']:
            return 'critical'
        elif usage >= threshold['warning']:
            return 'warning'
        else:
            return 'good'

    def _save_analysis(self, analysis: Dict[str, Any], report_dir: Path):
        """Save detailed analysis"""
        analysis_file = report_dir / 'detailed_analysis.json'
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)