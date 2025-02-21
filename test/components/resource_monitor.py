# test/components/resource_monitor.py

import psutil
import torch
import time
import numpy as np
from typing import Dict, Any, List
import logging
from pathlib import Path
import json


class ResourceMonitor:
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.initialize_metrics()
        self.start_monitoring()

    def initialize_metrics(self):
        """Initialize resource monitoring metrics"""
        self.metrics = {
            'system': {
                'cpu_usage': [],
                'memory_usage': [],
                'disk_io': [],
                'network_io': []
            },
            'gpu': {
                'memory_usage': [],
                'utilization': [],
                'temperature': []
            } if torch.cuda.is_available() else None,
            'process': {
                'cpu_percent': [],
                'memory_percent': [],
                'threads': [],
                'handles': []
            }
        }
        self.process = psutil.Process()

    def start_monitoring(self):
        """Start collecting initial metrics"""
        self.collect_metrics()  # Collect initial metrics

    def collect_metrics(self):
        """Collect current resource metrics"""
        try:
            # System metrics
            self.metrics['system']['cpu_usage'].append(psutil.cpu_percent(interval=0.1))
            self.metrics['system']['memory_usage'].append(psutil.virtual_memory().percent)

            # Process metrics
            self.metrics['process']['cpu_percent'].append(self.process.cpu_percent())
            self.metrics['process']['memory_percent'].append(self.process.memory_percent())
            self.metrics['process']['threads'].append(self.process.num_threads())

            # GPU metrics if available
            if self.metrics['gpu']:
                self.collect_gpu_metrics()

        except Exception as e:
            self.logger.error(f"Error collecting metrics: {str(e)}")

    def collect_gpu_metrics(self):
        """Collect GPU-specific metrics"""
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    memory_usage = torch.cuda.memory_allocated(i) / torch.cuda.max_memory_allocated(i)
                    self.metrics['gpu']['memory_usage'].append(memory_usage)
        except Exception as e:
            self.logger.error(f"Error collecting GPU metrics: {str(e)}")

    def analyze_resource_usage(self) -> Dict[str, Any]:
        """Analyze resource usage patterns"""
        self.collect_metrics()  # Collect metrics before analysis

        analysis = {
            'system': self._analyze_system_metrics(),
            'process': self._analyze_process_metrics()
        }

        if self.metrics['gpu']:
            analysis['gpu'] = self._analyze_gpu_metrics()

        return analysis

    def _analyze_system_metrics(self) -> Dict[str, Any]:
        """Analyze system-level metrics"""
        cpu_usage = self.metrics['system']['cpu_usage']
        memory_usage = self.metrics['system']['memory_usage']

        return {
            'cpu': {
                'current': cpu_usage[-1] if cpu_usage else 0,
                'average': np.mean(cpu_usage) if cpu_usage else 0,
                'peak': np.max(cpu_usage) if cpu_usage else 0,
                'variability': np.std(cpu_usage) if len(cpu_usage) > 1 else 0
            },
            'memory': {
                'current': memory_usage[-1] if memory_usage else 0,
                'average': np.mean(memory_usage) if memory_usage else 0,
                'peak': np.max(memory_usage) if memory_usage else 0,
                'trend': self._calculate_trend(memory_usage)
            }
        }

    def _analyze_process_metrics(self) -> Dict[str, Any]:
        """Analyze process-specific metrics"""
        cpu_percent = self.metrics['process']['cpu_percent']
        memory_percent = self.metrics['process']['memory_percent']

        return {
            'cpu_usage': {
                'current': cpu_percent[-1] if cpu_percent else 0,
                'average': np.mean(cpu_percent) if cpu_percent else 0,
                'peak': np.max(cpu_percent) if cpu_percent else 0
            },
            'memory_usage': {
                'current': memory_percent[-1] if memory_percent else 0,
                'average': np.mean(memory_percent) if memory_percent else 0,
                'peak': np.max(memory_percent) if memory_percent else 0
            }
        }

    def _analyze_gpu_metrics(self) -> Dict[str, Any]:
        """Analyze GPU metrics"""
        if not self.metrics['gpu']:
            return {}

        gpu_memory = self.metrics['gpu']['memory_usage']

        return {
            'memory': {
                'current': gpu_memory[-1] if gpu_memory else 0,
                'average': np.mean(gpu_memory) if gpu_memory else 0,
                'peak': np.max(gpu_memory) if gpu_memory else 0
            }
        }

    def _calculate_trend(self, data: List[float]) -> str:
        """Calculate trend direction"""
        if len(data) < 2:
            return "insufficient_data"

        try:
            slope = np.polyfit(range(len(data)), data, 1)[0]

            if abs(slope) < 0.01:
                return "stable"
            elif slope > 0:
                return "increasing"
            else:
                return "decreasing"
        except Exception:
            return "insufficient_data"

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current resource metrics"""
        self.collect_metrics()  # Ensure we have current data

        metrics = {
            'cpu_usage': self.metrics['system']['cpu_usage'][-1] if self.metrics['system']['cpu_usage'] else 0,
            'memory_usage': self.metrics['system']['memory_usage'][-1] if self.metrics['system']['memory_usage'] else 0,
        }

        if self.metrics['gpu']:
            metrics['gpu_usage'] = self.metrics['gpu']['memory_usage'][-1] if self.metrics['gpu']['memory_usage'] else 0

        return metrics

    def generate_report(self) -> Dict[str, Any]:
        """Generate resource usage report"""
        return {
            'current_usage': self.get_current_metrics(),
            'analysis': self.analyze_resource_usage(),
            'warnings': self._generate_warnings()
        }

    def _generate_warnings(self) -> List[str]:
        """Generate resource usage warnings"""
        warnings = []
        current_metrics = self.get_current_metrics()

        if current_metrics['cpu_usage'] > self.config.get('resource_thresholds', {}).get('cpu_warning', 80):
            warnings.append(f"High CPU usage: {current_metrics['cpu_usage']:.1f}%")

        if current_metrics['memory_usage'] > self.config.get('resource_thresholds', {}).get('memory_warning', 85):
            warnings.append(f"High memory usage: {current_metrics['memory_usage']:.1f}%")

        return warnings