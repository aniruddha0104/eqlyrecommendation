# test/config/test_config.py

config = {
    'system': {
        'results_dir': 'test_results',
        'log_dir': 'logs',
        'target_fps': 30,
        'test_duration': 30,
        'batch_size': 1
    },
    'performance': {
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
    'model': {
        'teacher_model_path': 'weights/teacher_model.pth',
        'learner_model_path': 'weights/learner_model.pth',
        'batch_size': 32,
        'performance_thresholds': {
            'min_accuracy': 0.8,
            'max_inference_time': 50
        }
    },
    'resources': {
        'monitor_gpu': True,
        'monitor_network': True,
        'monitor_disk': True,
        'resource_thresholds': {
            'cpu_warning': 80,
            'memory_warning': 85,
            'gpu_warning': 90,
            'temperature_warning': 80
        }
    },
    'reporting': {
        'generate_plots': True,
        'save_raw_data': True,
        'create_html': True,
        'plot_style': 'seaborn',
        'plot_formats': ['png', 'pdf']
    }
}