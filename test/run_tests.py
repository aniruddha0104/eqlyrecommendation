# test/run_tests.py

import logging
from pathlib import Path
from datetime import datetime
import time
from components.model_tester import ModelTester
from components.performance_test import PerformanceMonitor
from components.resource_monitor import ResourceMonitor
from components.metrics_reporter import MetricsReporter

# Default test configuration
DEFAULT_CONFIG = {
    'system': {
        'results_dir': 'test_results',
        'log_dir': 'logs',
        'test_duration': 30,
        'batch_size': 1
    },
    'model': {
        'model_path': 'weights',
        'performance_thresholds': {
            'min_accuracy': 0.8,
            'max_inference_time': 50
        }
    },
    'performance': {
        'target_fps': 30,
        'metrics_interval': 0.1,
        'save_interval': 30,
        'max_errors': 5,
        'performance_thresholds': {
            'min_fps': 25,
            'max_latency': 100,
            'max_cpu_usage': 80,
            'max_memory_usage': 85,
            'max_gpu_usage': 90
        }
    },
    'resources': {
        'monitor_gpu': True,
        'monitor_network': True,
        'resource_thresholds': {
            'cpu_warning': 80,
            'memory_warning': 85,
            'gpu_warning': 90
        }
    },
    'reporting': {
        'generate_plots': True,
        'save_raw_data': True,
        'results_dir': 'test_results'
    }
}


def setup_logging():
    """Setup logging configuration"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(
                log_dir / f'test_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            ),
            logging.StreamHandler()
        ]
    )


def simulate_processing(duration: int, perf_monitor: PerformanceMonitor):
    """Simulate processing for testing"""
    end_time = time.time() + duration
    frame_count = 0

    while time.time() < end_time:
        start = time.perf_counter()

        # Simulate processing
        time.sleep(1 / 30)  # Simulate 30 FPS processing
        frame_count += 1

        # Update metrics
        processing_time = time.perf_counter() - start
        perf_monitor.update_metrics(batch_size=1)


def run_tests(config=None):
    """Main test execution function"""
    if config is None:
        config = DEFAULT_CONFIG

    logger = logging.getLogger('TestRunner')
    logger.info("Starting test suite")

    try:
        # Initialize components
        model_tester = ModelTester(config['model'])
        perf_monitor = PerformanceMonitor(config['performance'])
        resource_monitor = ResourceMonitor(config['resources'])
        metrics_reporter = MetricsReporter(config['reporting'])

        # Start monitoring
        logger.info("Starting performance and resource monitoring")
        perf_monitor.start_monitoring()

        # Run model tests and simulate processing
        logger.info("Running model tests")
        test_data = {
            'duration': config['system']['test_duration']
        }
        model_results = model_tester.test_models(test_data)

        # Simulate processing for metrics
        simulate_processing(config['system']['test_duration'], perf_monitor)

        # Collect metrics
        logger.info("Collecting metrics")
        performance_metrics = perf_monitor.get_current_metrics()
        resource_metrics = resource_monitor.analyze_resource_usage()

        # Generate reports
        logger.info("Generating test reports")
        metrics = {
            'model_metrics': model_results,
            'performance': performance_metrics,
            'resources': resource_metrics
        }

        report_path = metrics_reporter.generate_report(metrics)

        # Print summary
        print("\nTest Results Summary:")
        print("=" * 50)
        print(f"Model Tests: {'Completed' if model_results else 'Failed'}")
        print(f"Average FPS: {performance_metrics['fps']:.2f}")
        print(f"CPU Usage: {performance_metrics['cpu_usage']:.1f}%")
        print(f"Memory Usage: {performance_metrics['memory_usage']:.1f}%")
        if 'gpu_usage' in performance_metrics:
            print(f"GPU Usage: {performance_metrics['gpu_usage']:.1f}%")
        print(f"Processing Latency: {performance_metrics.get('latency', 0):.2f}ms")
        print("=" * 50)

        return {
            'status': 'success',
            'report_path': report_path,
            'metrics': metrics
        }

    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}", exc_info=True)
        return {
            'status': 'failed',
            'error': str(e)
        }


def main():
    """Main entry point"""
    # Setup logging
    setup_logging()
    logger = logging.getLogger('TestRunner')

    try:
        # Create required directories
        Path('test_results').mkdir(exist_ok=True)
        Path('logs').mkdir(exist_ok=True)
        Path('weights').mkdir(exist_ok=True)

        # Run tests with default configuration
        results = run_tests()

        # Check results
        if results['status'] == 'success':
            logger.info("Test suite completed successfully")
            logger.info(f"Report available at: {results['report_path']}")
        else:
            logger.error(f"Test suite failed: {results['error']}")

    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}", exc_info=True)
    finally:
        logger.info("Test execution completed")


if __name__ == "__main__":
    main()