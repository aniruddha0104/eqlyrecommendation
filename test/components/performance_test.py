# test/components/performance_test.py

import time
import psutil
import torch
import numpy as np
from typing import Dict, List, Any
import logging
from pathlib import Path
import json


class PerformanceMonitor:
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.metrics = {
            'frame_times': [],
            'fps_history': [],
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [] if torch.cuda.is_available() else None,
            'latency': [],
            'throughput': []
        }
        self.start_time = None
        self.frame_count = 0
        self.is_monitoring = False

    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.frame_count = 0
        self.is_monitoring = True
        self._collect_initial_metrics()

    def _collect_initial_metrics(self):
        """Collect initial system metrics"""
        self.metrics['cpu_usage'].append(psutil.cpu_percent(interval=0.1))
        self.metrics['memory_usage'].append(psutil.virtual_memory().percent)
        if self.metrics['gpu_usage'] is not None:
            self.metrics['gpu_usage'].append(
                torch.cuda.memory_allocated() / max(1, torch.cuda.max_memory_allocated())
            )

    def update_metrics(self, batch_size: int = 1):
        """Update performance metrics"""
        if not self.is_monitoring:
            return

        # Record timing
        current_time = time.time()
        frame_time = (current_time - self.start_time) / max(1, self.frame_count)
        self.metrics['frame_times'].append(frame_time)

        # Update counts
        self.frame_count += batch_size

        # Calculate FPS
        elapsed = current_time - self.start_time
        fps = self.frame_count / max(elapsed, 0.001)
        self.metrics['fps_history'].append(fps)

        # Update system metrics
        self.metrics['cpu_usage'].append(psutil.cpu_percent())
        self.metrics['memory_usage'].append(psutil.virtual_memory().percent)

        if self.metrics['gpu_usage'] is not None:
            self.metrics['gpu_usage'].append(
                torch.cuda.memory_allocated() / max(1, torch.cuda.max_memory_allocated())
            )

        # Calculate latency and throughput
        if self.metrics['frame_times']:
            latency = self.metrics['frame_times'][-1] * 1000  # Convert to ms
            self.metrics['latency'].append(latency)
            self.metrics['throughput'].append(batch_size / max(frame_time, 0.001))

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        self.update_metrics()  # Update before returning

        fps = self.calculate_fps()
        cpu = np.mean(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0
        memory = np.mean(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0

        metrics = {
            'fps': fps,
            'cpu_usage': cpu,
            'memory_usage': memory,
            'latency': np.mean(self.metrics['latency']) if self.metrics['latency'] else 0,
            'throughput': np.mean(self.metrics['throughput']) if self.metrics['throughput'] else 0
        }

        if self.metrics['gpu_usage']:
            metrics['gpu_usage'] = np.mean(self.metrics['gpu_usage'])

        return metrics

    def calculate_fps(self) -> float:
        """Calculate current FPS"""
        if not self.start_time:
            return 0.0

        elapsed_time = time.time() - self.start_time
        return self.frame_count / max(elapsed_time, 0.001)

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        current_metrics = self.get_current_metrics()

        report = {
            'overall_performance': {
                'average_fps': current_metrics['fps'],
                'target_fps': self.config.get('target_fps', 30),
                'fps_achievement': (current_metrics['fps'] / self.config.get('target_fps', 30)) * 100,
                'average_latency': current_metrics['latency'],
                'average_throughput': current_metrics['throughput']
            },
            'resource_usage': {
                'cpu': {
                    'current': current_metrics['cpu_usage'],
                    'average': np.mean(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0,
                    'peak': np.max(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0
                },
                'memory': {
                    'current': current_metrics['memory_usage'],
                    'average': np.mean(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0,
                    'peak': np.max(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0
                }
            },
            'stability_metrics': {
                'fps_stability': self._calculate_stability(self.metrics['fps_history']),
                'latency_stability': self._calculate_stability(self.metrics['latency']),
                'performance_rating': self._calculate_performance_rating(current_metrics)
            }
        }

        if self.metrics['gpu_usage']:
            report['resource_usage']['gpu'] = {
                'current': current_metrics.get('gpu_usage', 0),
                'average': np.mean(self.metrics['gpu_usage']),
                'peak': np.max(self.metrics['gpu_usage'])
            }

        return report

    def _calculate_stability(self, data: List[float]) -> float:
        """Calculate stability score (0-1)"""
        if not data:
            return 0.0

        std = np.std(data)
        mean = np.mean(data)

        if mean == 0:
            return 1.0

        cv = std / max(mean, 0.001)  # Coefficient of variation
        stability = 1 / (1 + cv)

        return float(min(max(stability, 0), 1))

    def _calculate_performance_rating(self, metrics: Dict[str, float]) -> str:
        """Calculate overall performance rating"""
        target_fps = self.config.get('target_fps', 30)
        fps_ratio = metrics['fps'] / target_fps if target_fps > 0 else 0

        if fps_ratio >= 0.95:
            return 'excellent'
        elif fps_ratio >= 0.8:
            return 'good'
        elif fps_ratio >= 0.6:
            return 'fair'
        else:
            return 'needs_improvement'

    def save_metrics(self, save_path: Path):
        """Save performance metrics to file"""
        try:
            metrics_file = save_path / 'performance_metrics.json'

            with open(metrics_file, 'w') as f:
                json.dump({
                    'raw_metrics': {
                        k: v if not isinstance(v, np.ndarray) else v.tolist()
                        for k, v in self.metrics.items()
                    },
                    'analysis': self.generate_performance_report()
                }, f, indent=2)

            self.logger.info(f"Performance metrics saved to {metrics_file}")

        except Exception as e:
            self.logger.error(f"Failed to save metrics: {str(e)}")

    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics = {
            'frame_times': [],
            'fps_history': [],
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [] if torch.cuda.is_available() else None,
            'latency': [],
            'throughput': []
        }
        self.frame_count = 0
        self.start_time = time.time()