import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


class VisualizationUtils:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.output_dir = Path(self.config.get('output_dir', 'visualization_output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Predefine colors for faster access
        self.colors = {
            'face': (0, 255, 0),      # Green
            'pose': (255, 0, 0),      # Blue
            'emotion': (0, 165, 255),  # Orange
            'text': (255, 255, 255)    # White
        }

        # Use a thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)

    def draw_frame_analysis(
        self,
        frame: np.ndarray,
        faces: List[Dict],
        pose_data: Optional[Dict] = None,
        emotion_data: Optional[Dict] = None
    ) -> np.ndarray:
        """Draw all detections on frame"""
        try:
            vis_frame = frame.copy()

            # Parallelize drawing tasks
            futures = []
            if faces:
                futures.append(self.executor.submit(self._draw_faces, vis_frame, faces))
            if pose_data and pose_data.get('pose_landmarks'):
                futures.append(self.executor.submit(self._draw_pose, vis_frame, pose_data['pose_landmarks']))
            if emotion_data:
                futures.append(self.executor.submit(self._draw_emotions, vis_frame, emotion_data))

            # Wait for all tasks to complete
            for future in futures:
                future.result()

            return vis_frame

        except Exception as e:
            print(f"Visualization failed: {str(e)}")
            return frame

    def _draw_faces(self, frame: np.ndarray, faces: List[Dict]) -> np.ndarray:
        """Draw face detections"""
        for face in faces:
            bbox = face['bbox']
            confidence = face.get('confidence', 0.0)

            # Draw bounding box
            cv2.rectangle(
                frame,
                (bbox['x'], bbox['y']),
                (bbox['x'] + bbox['w'], bbox['y'] + bbox['h']),
                self.colors['face'],
                2
            )

            # Draw confidence
            text = f"Face: {confidence:.2f}"
            cv2.putText(
                frame,
                text,
                (bbox['x'], bbox['y'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.colors['text'],
                2
            )

        return frame

    def _draw_emotions(self, frame: np.ndarray, emotion_data: Dict) -> np.ndarray:
        """Draw emotion data"""
        if emotion_data.get('box'):
            x, y, w, h = emotion_data['box']

            # Draw emotion box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                self.colors['emotion'],
                2
            )

            # Draw emotion label
            text = f"{emotion_data['dominant_emotion']}: {emotion_data['confidence']:.2f}"
            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.colors['emotion'],
                2
            )

        return frame

    def create_analysis_plots(self, analysis_results: Dict, frame_count: int) -> None:
        """Create and save analysis plots"""
        try:
            # Use multithreading for plot generation
            self.executor.submit(self._plot_engagement, analysis_results.get('engagement_scores', []), frame_count)
            self.executor.submit(self._plot_emotion_distribution, analysis_results.get('emotions', {}))
            if 'voice_features' in analysis_results:
                self.executor.submit(self._plot_voice_analysis, analysis_results['voice_features'])

        except Exception as e:
            print(f"Plot creation failed: {str(e)}")

    def _plot_engagement(self, engagement_scores: List[float], frame_count: int) -> None:
        """Plot engagement over time"""
        plt.figure(figsize=(10, 5))
        plt.plot(engagement_scores)
        plt.title('Engagement Score Over Time')
        plt.xlabel('Frame Number')
        plt.ylabel('Engagement Score')
        plt.grid(True)
        plt.savefig(str(self.output_dir / 'engagement_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_emotion_distribution(self, emotion_data: Dict) -> None:
        """Plot emotion distribution"""
        if emotion_data and 'emotion_scores' in emotion_data:
            emotions = list(emotion_data['emotion_scores'].keys())
            scores = list(emotion_data['emotion_scores'].values())

            plt.figure(figsize=(10, 5))
            plt.bar(emotions, scores)
            plt.title('Emotion Distribution')
            plt.xlabel('Emotions')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(str(self.output_dir / 'emotion_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_voice_analysis(self, voice_features: Dict) -> None:
        """Plot voice analysis features"""
        plt.figure(figsize=(12, 6))

        features = voice_features.get('speech_features', {})
        if features:
            plt.subplot(131)
            plt.plot(features.get('mfcc_mean', []))
            plt.title('MFCC Features')
            plt.xlabel('Coefficient')
            plt.ylabel('Value')

        prosody = voice_features.get('prosody', {})
        if prosody:
            plt.subplot(132)
            plt.plot(prosody.get('pitch_mean', 0))
            plt.title('Pitch Analysis')
            plt.xlabel('Time')
            plt.ylabel('Pitch')

        plt.tight_layout()
        plt.savefig(str(self.output_dir / 'voice_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()