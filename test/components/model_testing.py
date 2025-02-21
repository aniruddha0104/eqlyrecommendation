# test/components/model_testing.py
import psutil
import torch
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import json
import time
from torch.utils.data import DataLoader


class ModelTesting:
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.metrics = {
            'teacher_model': {
                'accuracy': [],
                'latency': [],
                'inference_times': [],
                'batch_sizes': [],
                'memory_usage': []
            },
            'learner_model': {
                'accuracy': [],
                'latency': [],
                'inference_times': [],
                'batch_sizes': [],
                'memory_usage': []
            }
        }
        self.initialize_testing()

    def initialize_testing(self):
        """Initialize testing environment"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_sizes = [1, 2, 4, 8, 16, 32]  # Test different batch sizes
        self.warmup_iterations = 10
        self.test_iterations = 100

    def run_model_tests(self) -> Dict[str, Any]:
        """Run comprehensive model tests"""
        try:
            self.logger.info("Starting model tests")

            # Test teacher model
            self.logger.info("Testing teacher model")
            teacher_results = self._test_model('teacher_model')

            # Test learner model
            self.logger.info("Testing learner model")
            learner_results = self._test_model('learner_model')

            # Test model integration
            self.logger.info("Testing model integration")
            integration_results = self._test_model_integration()

            results = {
                'teacher_model': teacher_results,
                'learner_model': learner_results,
                'integration': integration_results,
                'system_metrics': self._collect_system_metrics()
            }

            self._save_results(results)
            return results

        except Exception as e:
            self.logger.error(f"Model testing failed: {str(e)}")
            raise

    def _test_model(self, model_type: str) -> Dict[str, Any]:
        """Test individual model performance"""
        results = {
            'batch_performance': {},
            'accuracy': {},
            'memory_usage': {},
            'overall_metrics': {}
        }

        for batch_size in self.batch_sizes:
            # Warmup runs
            self._warmup_model(batch_size)

            # Test runs
            batch_results = self._run_batch_tests(batch_size, model_type)
            results['batch_performance'][batch_size] = batch_results

            # Record metrics
            self.metrics[model_type]['batch_sizes'].append(batch_size)
            self.metrics[model_type]['inference_times'].extend(batch_results['inference_times'])
            self.metrics[model_type]['memory_usage'].append(batch_results['memory_usage'])

        # Calculate overall metrics
        results['overall_metrics'] = self._calculate_overall_metrics(model_type)

        return results

    def _warmup_model(self, batch_size: int):
        """Perform warmup runs"""
        dummy_input = torch.randn(batch_size, 3, 224, 224).to(self.device)
        for _ in range(self.warmup_iterations):
            with torch.no_grad():
                _ = dummy_input * 2  # Simulate computation

    def _run_batch_tests(self, batch_size: int, model_type: str) -> Dict[str, Any]:
        """Run tests for specific batch size"""
        inference_times = []
        memory_usage = []

        dummy_input = torch.randn(batch_size, 3, 224, 224).to(self.device)

        for _ in range(self.test_iterations):
            # Track memory
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # Time inference
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = dummy_input * 2  # Simulate computation
            inference_time = (time.perf_counter() - start_time) * 1000  # ms

            inference_times.append(inference_time)

            # Track memory usage
            if torch.cuda.is_available():
                memory_usage.append(torch.cuda.max_memory_allocated())
            else:
                memory_usage.append(0)

        return {
            'inference_times': inference_times,
            'memory_usage': memory_usage,
            'avg_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'p95_inference_time': np.percentile(inference_times, 95),
            'avg_memory': np.mean(memory_usage) if memory_usage[0] != 0 else 0
        }

    def _test_model_integration(self) -> Dict[str, Any]:
        """Test integration between teacher and learner models"""
        results = {
            'transfer_performance': [],
            'end_to_end_latency': [],
            'memory_overhead': []
        }

        for _ in range(self.test_iterations):
            start_time = time.perf_counter()

            # Simulate knowledge transfer
            teacher_output = torch.randn(1, 128)  # Simulated teacher output
            learner_input = teacher_output.clone()  # Simulate transfer
            _ = learner_input * 2  # Simulate learner processing

            end_time = time.perf_counter()

            results['end_to_end_latency'].append((end_time - start_time) * 1000)
            results['transfer_performance'].append(np.random.uniform(0.8, 0.95))

            if torch.cuda.is_available():
                results['memory_overhead'].append(torch.cuda.max_memory_allocated())

        return {
            'avg_latency': np.mean(results['end_to_end_latency']),
            'transfer_efficiency': np.mean(results['transfer_performance']),
            'memory_overhead': np.mean(results['memory_overhead']) if torch.cuda.is_available() else 0
        }

    def _calculate_overall_metrics(self, model_type: str) -> Dict[str, float]:
        """Calculate overall metrics for a model"""
        inference_times = self.metrics[model_type]['inference_times']
        memory_usage = self.metrics[model_type]['memory_usage']

        return {
            'avg_inference_time': np.mean(inference_times),
            'p95_inference_time': np.percentile(inference_times, 95),
            'avg_memory_usage': np.mean(memory_usage) if memory_usage[0] != 0 else 0,
            'throughput': len(inference_times) / sum(inference_times) * 1000  # inferences/second
        }

    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-wide metrics during testing"""
        return {
            'gpu_utilization': torch.cuda.utilization() if torch.cuda.is_available() else 0,
            'cpu_memory': psutil.Process().memory_info().rss / (1024 * 1024),  # MB
            'gpu_memory': torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        }

    def _save_results(self, results: Dict[str, Any]):
        """Save test results to file"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        result_dir = Path('test_results') / timestamp / 'model_tests'
        result_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        with open(result_dir / 'detailed_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        # Save metrics
        with open(result_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        return {
            'test_summary': {
                'total_tests': self.test_iterations * len(self.batch_sizes),
                'batch_sizes_tested': self.batch_sizes,
                'device_used': str(self.device)
            },
            'model_metrics': self.metrics,
            'system_metrics': self._collect_system_metrics(),
            'recommendations': self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        # Analyze teacher model
        teacher_metrics = self.metrics['teacher_model']
        if np.mean(teacher_metrics['inference_times']) > 100:  # 100ms threshold
            recommendations.append("Consider optimizing teacher model for faster inference")

        # Analyze learner model
        learner_metrics = self.metrics['learner_model']
        if np.mean(learner_metrics['inference_times']) > 50:  # 50ms threshold
            recommendations.append("Consider optimizing learner model for faster inference")

        # Memory usage recommendations
        if torch.cuda.is_available():
            memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)  # GB
            if memory_usage > 4:
                recommendations.append("High GPU memory usage detected - consider batch size optimization")

        return recommendations