# test/components/visualization.py

import cv2
import numpy as np
from typing import Dict, Any, List
import matplotlib.pyplot as plt
from pathlib import Path


class FeatureVisualizer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Drawing settings
        self.colors = {
            'face': (255, 0, 0),  # Red
            'eyes': (0, 255, 0),  # Green
            'mouth': (0, 0, 255),  # Blue
            'keypoints': (255, 255, 0),  # Yellow
            'pose': (0, 255, 255)  # Cyan
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def visualize_frame_features(self, frame: np.ndarray, features: Dict[str, Any],
                                 frame_number: int) -> np.ndarray:
        """Visualize features on frame"""
        vis_frame = frame.copy()

        # Draw facial features
        if 'facial' in features:
            self._draw_facial_features(vis_frame, features['facial'])

        # Draw visual features
        if 'visual' in features:
            self._draw_visual_features(vis_frame, features['visual'])

        # Draw pose features
        if 'pose' in features:
            self._draw_pose_features(vis_frame, features['pose'])

        # Draw motion features
        if 'motion' in features:
            self._draw_motion_features(vis_frame, features['motion'])

        # Draw engagement metrics
        if 'engagement' in features:
            self._draw_engagement_metrics(vis_frame, features['engagement'])

        return vis_frame

        # test/components/visualization.py (continued)

        def _draw_facial_features(self, frame: np.ndarray, facial_features: Dict[str, Any]):
            """Draw facial features"""
            # Draw face rectangles and metrics
            y_offset = 30
            for face_loc in facial_features.get('face_locations', []):
                x, y, w, h = face_loc
                cv2.rectangle(frame, (x, y), (x + w, y + h), self.colors['face'], 2)

                # Draw expression metrics
                expr = facial_features.get('expression_analysis', {})
                metrics_text = [
                    f"Eye AR: {expr.get('eye_aspect_ratio', 0):.2f}",
                    f"Mouth AR: {expr.get('mouth_aspect_ratio', 0):.2f}",
                    f"Brow Pos: {expr.get('eyebrow_position', 0):.2f}"
                ]

                for text in metrics_text:
                    cv2.putText(frame, text, (x, y + y_offset),
                                self.font, 0.5, self.colors['face'], 1)
                    y_offset += 20

        def _draw_visual_features(self, frame: np.ndarray, visual_features: Dict[str, Any]):
            """Draw visual features"""
            # Draw keypoints
            keypoints = visual_features.get('keypoints', [])
            for kp in keypoints:
                x, y = map(int, kp.pt)
                cv2.circle(frame, (x, y), 3, self.colors['keypoints'], -1)

            # Draw color analysis
            color_stats = visual_features.get('color_stats', {})
            color_text = [
                f"Hue: {color_stats.get('hue_mean', 0):.1f}",
                f"Sat: {color_stats.get('saturation_mean', 0):.1f}",
                f"Val: {color_stats.get('value_mean', 0):.1f}"
            ]

            for i, text in enumerate(color_text):
                cv2.putText(frame, text, (10, 20 + i * 20),
                            self.font, 0.5, self.colors['keypoints'], 1)

        def _draw_pose_features(self, frame: np.ndarray, pose_features: Dict[str, Any]):
            """Draw pose features"""
            pose_text = [
                f"Head Pos: {pose_features.get('head_position', 0):.2f}",
                f"Shoulder Align: {pose_features.get('shoulder_alignment', 0):.2f}",
                f"Body Orient: {pose_features.get('body_orientation', 0):.2f}"
            ]

            for i, text in enumerate(pose_text):
                cv2.putText(frame, text, (frame.shape[1] - 200, 20 + i * 20),
                            self.font, 0.5, self.colors['pose'], 1)

        def _draw_engagement_metrics(self, frame: np.ndarray, engagement: Dict[str, Any]):
            """Draw engagement metrics"""
            metrics_text = [
                f"Attention: {engagement.get('attention_score', 0):.2f}",
                f"Interaction: {engagement.get('interaction_level', 0):.2f}",
                f"Confidence: {engagement.get('confidence_score', 0):.2f}"
            ]

            y_base = frame.shape[0] - 80
            for i, text in enumerate(metrics_text):
                cv2.putText(frame, text, (10, y_base + i * 20),
                            self.font, 0.5, (255, 255, 255), 1)

        def save_visualization(self, frame: np.ndarray, frame_number: int):
            """Save visualization frame"""
            output_path = self.output_dir / f"frame_{frame_number:04d}.jpg"
            cv2.imwrite(str(output_path), frame)

        def generate_summary_plots(self, all_features: List[Dict[str, Any]],
                                   save_prefix: str = "analysis"):
            """Generate summary plots of features over time"""
            # Create figure for metrics over time
            plt.figure(figsize=(15, 10))

            # Plot engagement metrics
            engagement_scores = {
                'attention': [],
                'interaction': [],
                'confidence': []
            }

            for features in all_features:
                if 'engagement' in features:
                    engagement = features['engagement']
                    engagement_scores['attention'].append(engagement.get('attention_score', 0))
                    engagement_scores['interaction'].append(engagement.get('interaction_level', 0))
                    engagement_scores['confidence'].append(engagement.get('confidence_score', 0))

            frames = range(len(all_features))

            plt.subplot(2, 1, 1)
            for metric, scores in engagement_scores.items():
                plt.plot(frames, scores, label=metric.capitalize())
            plt.title('Engagement Metrics Over Time')
            plt.xlabel('Frame')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)

            # Plot feature counts
            plt.subplot(2, 1, 2)
            keypoints = [f['visual']['keypoint_count'] for f in all_features if 'visual' in f]
            faces = [len(f['facial']['face_locations']) for f in all_features if 'facial' in f]

            plt.plot(frames, keypoints, label='Keypoints')
            plt.plot(frames, faces, label='Faces')
            plt.title('Feature Detection Over Time')
            plt.xlabel('Frame')
            plt.ylabel('Count')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(str(self.output_dir / f"{save_prefix}_summary.png"))
            plt.close()

    # Complete the data processing code by integrating the new analysis and visualization components
    class EnhancedDataProcessor:
        def __init__(self, config: Dict[str, Any]):
            self.config = config
            self.logger = logging.getLogger(self.__class__.__name__)

            # Initialize components
            self.feature_extractor = AdvancedFeatureExtractor(config)
            self.visualizer = FeatureVisualizer(Path(config.get('output_dir', 'results')))

            # Store features for analysis
            self.all_features = []
            self.processed_frames = 0

        def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
            """Process a single frame"""
            try:
                # Extract features
                features = self.feature_extractor.extract_features(frame)

                # Create visualization
                vis_frame = self.visualizer.visualize_frame_features(
                    frame, features, self.processed_frames
                )

                # Save visualization
                self.visualizer.save_visualization(vis_frame, self.processed_frames)

                # Store features
                self.all_features.append(features)
                self.processed_frames += 1

                return features

            except Exception as e:
                self.logger.error(f"Frame processing failed: {str(e)}")
                return {}

        def generate_analysis(self) -> Dict[str, Any]:
            """Generate final analysis"""
            try:
                # Generate summary plots
                self.visualizer.generate_summary_plots(self.all_features)

                # Calculate overall metrics
                metrics = self._calculate_overall_metrics()

                # Save metrics
                self._save_metrics(metrics)

                return metrics

            except Exception as e:
                self.logger.error(f"Analysis generation failed: {str(e)}")
                return {'error': str(e)}

        def _calculate_overall_metrics(self) -> Dict[str, Any]:
            """Calculate overall metrics from all processed frames"""
            metrics = {
                'total_frames': self.processed_frames,
                'engagement': {
                    'average_attention': 0.0,
                    'average_interaction': 0.0,
                    'average_confidence': 0.0
                },
                'features': {
                    'average_keypoints': 0.0,
                    'average_faces': 0.0
                }
            }

            try:
                # Calculate averages
                attention_scores = []
                interaction_scores = []
                confidence_scores = []
                keypoint_counts = []
                face_counts = []

                for features in self.all_features:
                    if 'engagement' in features:
                        attention_scores.append(features['engagement'].get('attention_score', 0))
                        interaction_scores.append(features['engagement'].get('interaction_level', 0))
                        confidence_scores.append(features['engagement'].get('confidence_score', 0))

                    if 'visual' in features:
                        keypoint_counts.append(features['visual'].get('keypoint_count', 0))

                    if 'facial' in features:
                        face_counts.append(len(features['facial'].get('face_locations', [])))

                metrics['engagement']['average_attention'] = np.mean(attention_scores)
                metrics['engagement']['average_interaction'] = np.mean(interaction_scores)
                metrics['engagement']['average_confidence'] = np.mean(confidence_scores)
                metrics['features']['average_keypoints'] = np.mean(keypoint_counts)
                metrics['features']['average_faces'] = np.mean(face_counts)

            except Exception as e:
                self.logger.error(f"Metrics calculation failed: {str(e)}")

            return metrics

        def _save_metrics(self, metrics: Dict[str, Any]):
            """Save metrics to file"""
            try:
                output_path = Path(self.config['output_dir']) / 'analysis_metrics.json'
                with open(output_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
            except Exception as e:
                self.logger.error(f"Failed to save metrics: {str(e)}")

    def main():
        """Main function"""
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
        )
        logger = logging.getLogger(__name__)

        try:
            # Configuration
            config = {
                'target_size': (224, 224),
                'normalize': True,
                'output_dir': 'results/analysis'
            }

            # Initialize processor
            processor = EnhancedDataProcessor(config)

            # Process video
            video_path = "test_data/test_video.mp4"
            if Path(video_path).exists():
                cap = cv2.VideoCapture(video_path)
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    processor.process_frame(frame)
                    frame_count += 1

                    if frame_count % 10 == 0:
                        logger.info(f"Processed {frame_count} frames")

                cap.release()

            # Generate final analysis
            metrics = processor.generate_analysis()

            logger.info("Processing completed")
            logger.info(f"Results saved in: {config['output_dir']}")

            # Print summary
            print("\nAnalysis Results:")
            print("=" * 50)
            for category, values in metrics.items():
                print(f"\n{category.upper()}:")
                if isinstance(values, dict):
                    for metric, value in values.items():
                        print(f"  {metric}: {value:.2f}")
                else:
                    print(f"  {values}")
            print("=" * 50)

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}", exc_info=True)

    if __name__ == "__main__":
        main()