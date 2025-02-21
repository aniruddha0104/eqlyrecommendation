# test/components/metrics_processor.py
import numpy as np


class MetricsProcessor:
    def __init__(self):
        self.metrics = {
            'frame_metrics': [],
            'performance_metrics': {
                'processing_times': [],
                'memory_usage': [],
                'cpu_usage': []
            },
            'feature_metrics': {
                'keypoints': [],
                'faces': [],
                'motion': [],
                'engagement': []
            },
            'quality_metrics': {
                'clarity_scores': [],
                'lighting_scores': [],
                'stability_scores': []
            }
        }

    def process_frame_metrics(self, frame_data):
        """Process individual frame metrics"""
        frame_metrics = {
            'content_metrics': {
                'keypoint_count': frame_data.get('keypoints', 0),
                'face_count': frame_data.get('faces', 0),
                'face_confidence': frame_data.get('face_confidence', 0),
                'motion_score': frame_data.get('motion_score', 0)
            },
            'quality_metrics': {
                'clarity': self._calculate_clarity(frame_data),
                'lighting': self._calculate_lighting(frame_data),
                'stability': self._calculate_stability(frame_data)
            },
            'engagement_metrics': {
                'attention_score': frame_data.get('attention', 0),
                'expression_score': frame_data.get('expression', 0),
                'interaction_score': frame_data.get('interaction', 0)
            },
            'audio_metrics': {  # If audio is available
                'speech_confidence': frame_data.get('speech_confidence', 0),
                'tone_metrics': frame_data.get('tone_metrics', {}),
                'speech_rate': frame_data.get('speech_rate', 0)
            }
        }

        self.metrics['frame_metrics'].append(frame_metrics)
        self._update_aggregate_metrics(frame_metrics)
        return frame_metrics

    def _update_aggregate_metrics(self, frame_metrics):
        """Update running aggregates"""
        # Feature metrics
        self.metrics['feature_metrics']['keypoints'].append(
            frame_metrics['content_metrics']['keypoint_count']
        )
        self.metrics['feature_metrics']['faces'].append(
            frame_metrics['content_metrics']['face_count']
        )

        # Quality metrics
        self.metrics['quality_metrics']['clarity_scores'].append(
            frame_metrics['quality_metrics']['clarity']
        )
        self.metrics['quality_metrics']['lighting_scores'].append(
            frame_metrics['quality_metrics']['lighting']
        )
        self.metrics['quality_metrics']['stability_scores'].append(
            frame_metrics['quality_metrics']['stability']
        )

        # Engagement metrics
        engagement_score = (
                                   frame_metrics['engagement_metrics']['attention_score'] +
                                   frame_metrics['engagement_metrics']['expression_score'] +
                                   frame_metrics['engagement_metrics']['interaction_score']
                           ) / 3
        self.metrics['feature_metrics']['engagement'].append(engagement_score)

    def generate_summary(self):
        """Generate comprehensive summary metrics"""
        summary = {
            'content_analysis': {
                'total_frames': len(self.metrics['frame_metrics']),
                'keypoint_stats': {
                    'average': np.mean(self.metrics['feature_metrics']['keypoints']),
                    'max': np.max(self.metrics['feature_metrics']['keypoints']),
                    'stability': np.std(self.metrics['feature_metrics']['keypoints'])
                },
                'face_detection': {
                    'average_faces': np.mean(self.metrics['feature_metrics']['faces']),
                    'max_faces': np.max(self.metrics['feature_metrics']['faces']),
                    'detection_rate': len([f for f in self.metrics['feature_metrics']['faces'] if f > 0]) /
                                      len(self.metrics['feature_metrics']['faces'])
                }
            },
            'quality_analysis': {
                'clarity': {
                    'average': np.mean(self.metrics['quality_metrics']['clarity_scores']),
                    'consistency': 1 - np.std(self.metrics['quality_metrics']['clarity_scores'])
                },
                'lighting': {
                    'average': np.mean(self.metrics['quality_metrics']['lighting_scores']),
                    'consistency': 1 - np.std(self.metrics['quality_metrics']['lighting_scores'])
                },
                'stability': {
                    'average': np.mean(self.metrics['quality_metrics']['stability_scores']),
                    'consistency': 1 - np.std(self.metrics['quality_metrics']['stability_scores'])
                }
            },
            'engagement_analysis': {
                'average_engagement': np.mean(self.metrics['feature_metrics']['engagement']),
                'engagement_trend': self._calculate_trend(self.metrics['feature_metrics']['engagement']),
                'peak_engagement': np.max(self.metrics['feature_metrics']['engagement']),
                'engagement_stability': 1 - np.std(self.metrics['feature_metrics']['engagement'])
            },
            'performance_metrics': {
                'average_processing_time': np.mean(self.metrics['performance_metrics']['processing_times']),
                'max_processing_time': np.max(self.metrics['performance_metrics']['processing_times']),
                'processing_stability': 1 - np.std(self.metrics['performance_metrics']['processing_times'])
            }
        }

        # Add recommendations
        summary['recommendations'] = self._generate_recommendations(summary)

        return summary

    def _generate_recommendations(self, summary):
        """Generate improvement recommendations"""
        recommendations = []

        # Content recommendations
        if summary['content_analysis']['face_detection']['detection_rate'] < 0.8:
            recommendations.append("Consider improving face visibility and camera angle")

        # Quality recommendations
        quality = summary['quality_analysis']
        if quality['clarity']['average'] < 0.7:
            recommendations.append("Consider improving video clarity/resolution")
        if quality['lighting']['average'] < 0.7:
            recommendations.append("Consider improving lighting conditions")
        if quality['stability']['average'] < 0.7:
            recommendations.append("Consider using a more stable camera setup")

        # Engagement recommendations
        engagement = summary['engagement_analysis']
        if engagement['average_engagement'] < 0.6:
            recommendations.append("Consider techniques to improve engagement")
        if engagement['engagement_stability'] < 0.7:
            recommendations.append("Work on maintaining consistent engagement levels")

        return recommendations

    def _calculate_trend(self, values):
        """Calculate trend direction and magnitude"""
        if len(values) < 2:
            return "insufficient_data"

        slope = np.polyfit(range(len(values)), values, 1)[0]

        if abs(slope) < 0.01:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"

    def _calculate_clarity(self, frame_data):
        """Calculate frame clarity score"""
        # Implement clarity calculation based on features
        return 0.8  # Placeholder

    def _calculate_lighting(self, frame_data):
        """Calculate lighting quality score"""
        # Implement lighting calculation based on features
        return 0.8  # Placeholder

    def _calculate_stability(self, frame_data):
        """Calculate frame stability score"""
        # Implement stability calculation based on features
        return 0.8  # Placeholder