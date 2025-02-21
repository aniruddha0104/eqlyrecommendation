import cv2
import numpy as np
import time
import logging
import json
from pathlib import Path
from datetime import datetime
import psutil
import torch
from typing import Dict, List, Any
import sys


class RobustTestSystem:
    def __init__(self):
        # Initialize core components
        self.start_time = None
        self.is_running = False
        self.error_occurred = False
        self.processed_frames = 0

        # Setup all systems
        self._initialize_base_config()
        self.setup_logging()
        self.detect_environment_capabilities()
        self.initialize_dynamic_config()
        self.setup_storage()
        self.setup_performance_tracking()

    def _initialize_base_config(self):
        """Initialize base configuration settings"""
        self.base_config = {
            'video': {
                'resolution': (1280, 720),
                'fps': 30,
                'format': 'BGR'
            },
            'metrics': {
                'collection_interval': 0.1,
                'save_interval': 30,
                'max_errors': 5
            }
        }

    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.handlers = []

        # Create logs directory
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)

        # Setup formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(
            log_dir / f'test_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.logger.setLevel(logging.DEBUG)

    def detect_environment_capabilities(self):
        """Detect system capabilities"""
        try:
            self.env_capabilities = {
                'gui_support': self._check_gui_support(),
                'cuda_support': torch.cuda.is_available(),
                'system_memory': psutil.virtual_memory().total / (1024 * 1024 * 1024),
                'cpu_count': psutil.cpu_count()
            }
            self.logger.info(f"Environment capabilities: {self.env_capabilities}")
        except Exception as e:
            self.logger.error(f"Environment detection failed: {str(e)}")
            self.env_capabilities = {
                'gui_support': False,
                'cuda_support': False,
                'system_memory': 4.0,
                'cpu_count': 2
            }

    def _check_gui_support(self) -> bool:
        """Check if GUI is supported"""
        try:
            if hasattr(cv2, 'imshow'):
                return True
            return False
        except Exception as e:
            self.logger.debug(f"GUI support check failed: {str(e)}")
            return False

    def initialize_dynamic_config(self):
        """Initialize dynamic configuration"""
        self.config = {
            'video': self.base_config['video'],
            'processing': {
                'batch_size': self._calculate_optimal_batch_size(),
                'use_gpu': self.env_capabilities['cuda_support'],
                'num_workers': self._calculate_optimal_workers(),
                'max_errors': self.base_config['metrics']['max_errors']
            },
            'metrics': self.base_config['metrics']
        }

    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of worker threads"""
        return max(1, min(int(self.env_capabilities['cpu_count'] * 0.75), 8))

    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size"""
        if self.env_capabilities['cuda_support']:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            return min(16, int(gpu_memory / (1024 * 1024 * 1024)))
        else:
            return min(4, max(1, int(self.env_capabilities['system_memory'] / 4)))

    def setup_storage(self):
        """Setup storage directories"""
        self.results_dir = Path('test_results') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.metrics_dir = self.results_dir / 'metrics'

        for directory in [self.results_dir, self.metrics_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        self.results_storage = {
            'processed_frames': 0,
            'saved_metrics': 0,
            'error_logs': []
        }

    def setup_performance_tracking(self):
        """Setup performance tracking metrics"""
        self.performance_metrics = {
            'processing': {
                'frame_times': [],
                'fps_history': [],
                'latency_history': []
            },
            'resources': {
                'cpu_usage': [],
                'memory_usage': [],
                'gpu_usage': [] if self.env_capabilities['cuda_support'] else None
            }
        }

    def run_test_suite(self, duration: int = 30):
        """Run the test suite"""
        self.logger.info(f"Starting test suite (duration: {duration}s)")
        self.start_time = time.time()
        self.is_running = True

        try:
            self._run_simulation_loop(duration)
        except Exception as e:
            self.logger.error(f"Test suite failed: {str(e)}")
            self.error_occurred = True
        finally:
            self.is_running = False
            self._cleanup()

    def _run_simulation_loop(self, duration: int):
        """Run simulation loop with visible progress"""
        end_time = time.time() + duration
        frame_count = 0
        total_frames = duration * 30  # 30 FPS

        print("\nStarting Test Suite...")
        print("=" * 50)

        while time.time() < end_time and not self.error_occurred:
            try:
                frame_start = time.perf_counter()

                # Simulate processing
                time.sleep(1 / 30)  # Simulate 30 FPS
                frame_count += 1

                # Update metrics
                processing_time = time.perf_counter() - frame_start
                self._update_metrics(processing_time)

                # Show progress
                self._display_progress(frame_count, total_frames)

                # Save metrics periodically
                if frame_count % self.config['metrics']['save_interval'] == 0:
                    self._save_metrics()
                    self._display_current_metrics()

            except Exception as e:
                self.logger.error(f"Processing error: {str(e)}")
                if len(self.results_storage['error_logs']) > self.config['processing']['max_errors']:
                    break

    def _display_progress(self, current: int, total: int):
        """Display progress bar"""
        progress = int((current / total) * 50)
        sys.stdout.write('\r')
        sys.stdout.write(f"Progress: [{'=' * progress}{' ' * (50 - progress)}] {current}/{total} frames")
        sys.stdout.flush()

    def _update_metrics(self, processing_time: float):
        """Update performance metrics"""
        self.performance_metrics['processing']['frame_times'].append(processing_time)
        self.processed_frames += 1

        # Update resource usage
        self.performance_metrics['resources']['cpu_usage'].append(psutil.cpu_percent())
        self.performance_metrics['resources']['memory_usage'].append(psutil.virtual_memory().percent)

        if self.env_capabilities['cuda_support']:
            self.performance_metrics['resources']['gpu_usage'].append(
                torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            )

    def _display_current_metrics(self):
        """Display current performance metrics"""
        print("\n\nCurrent Metrics:")
        print("-" * 30)

        # Calculate current FPS
        recent_times = self.performance_metrics['processing']['frame_times'][-30:]
        current_fps = len(recent_times) / sum(recent_times)

        # Display performance metrics
        print(f"FPS: {current_fps:.2f}")
        print(f"CPU Usage: {psutil.cpu_percent()}%")
        print(f"Memory Usage: {psutil.virtual_memory().percent}%")

        if self.env_capabilities['cuda_support']:
            gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            print(f"GPU Usage: {gpu_usage * 100:.2f}%")

        print(f"Processed Frames: {self.processed_frames}")
        print("-" * 30)
        print("\n")  # Add space for next progress bar

    def _save_metrics(self):
        """Save current metrics to file"""
        try:
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            metrics_file = self.metrics_dir / f'metrics_{current_time}.json'

            metrics_data = {
                'timestamp': current_time,
                'performance': self.performance_metrics,
                'resources': {
                    'cpu_usage': psutil.cpu_percent(),
                    'memory_usage': psutil.virtual_memory().percent,
                    'gpu_usage': torch.cuda.memory_allocated() if self.env_capabilities['cuda_support'] else None
                }
            }

            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save metrics: {str(e)}")

    def _generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        report = {
            'test_duration': time.time() - self.start_time,
            'frames_processed': self.processed_frames,
            'average_fps': len(self.performance_metrics['processing']['frame_times']) /
                           (time.time() - self.start_time),
            'processing_stats': {
                'mean_time': np.mean(self.performance_metrics['processing']['frame_times']),
                'std_time': np.std(self.performance_metrics['processing']['frame_times']),
                'min_time': np.min(self.performance_metrics['processing']['frame_times']),
                'max_time': np.max(self.performance_metrics['processing']['frame_times'])
            },
            'resource_usage': {
                'mean_cpu': np.mean(self.performance_metrics['resources']['cpu_usage']),
                'mean_memory': np.mean(self.performance_metrics['resources']['memory_usage']),
                'mean_gpu': np.mean(self.performance_metrics['resources']['gpu_usage'])
                if self.env_capabilities['cuda_support'] else None
            },
            'error_count': len(self.results_storage['error_logs'])
        }

        # Print summary
        print("\n" + "=" * 50)
        print("Test Summary:")
        print(f"Duration: {report['test_duration']:.2f} seconds")
        print(f"Frames Processed: {report['frames_processed']}")
        print(f"Average FPS: {report['average_fps']:.2f}")
        print(f"Mean CPU Usage: {report['resource_usage']['mean_cpu']:.2f}%")
        print(f"Mean Memory Usage: {report['resource_usage']['mean_memory']:.2f}%")
        if report['resource_usage']['mean_gpu']:
            print(f"Mean GPU Usage: {report['resource_usage']['mean_gpu']:.2f}%")
        print("=" * 50)

        return report

    def _cleanup(self):
        """Cleanup and generate final report"""
        try:
            # Generate and save final report
            final_report = self._generate_summary_report()
            report_file = self.results_dir / 'final_report.json'

            with open(report_file, 'w') as f:
                json.dump(final_report, f, indent=2)

            print(f"\nFinal report saved to: {report_file}")
            self.logger.info("Cleanup completed successfully")

        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")


def main():
    """Main entry point"""
    try:
        test_system = RobustTestSystem()
        test_system.run_test_suite(duration=30)
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        logging.error("Test system failed", exc_info=True)
    finally:
        print("\nTest completed.")


if __name__ == "__main__":
    main()