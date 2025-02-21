import asyncio
import numpy as np
import cv2
import json
from teacher_evaluator import TeacherEvaluator


async def evaluate_teaching_video(video_path, transcript_path, domain=None):
    """
    Evaluate a teaching video using the TeacherEvaluator

    Args:
        video_path: Path to the teaching video file
        transcript_path: Path to the transcript text file
        domain: Optional domain for specialized analysis

    Returns:
        Evaluation results dictionary
    """
    # Load configuration
    config = {
        'knowledge_base': {'domain_path': 'data/domains'},
        'scoring_weights': {
            'visual': 0.25,
            'audio': 0.25,
            'content': 0.4,
            'authenticity': 0.1
        }
    }

    # Initialize evaluator
    evaluator = TeacherEvaluator(config)

    # Read transcript
    with open(transcript_path, 'r') as f:
        transcript = f.read()

    # Process video to extract frames and audio
    video_frames, audio_data = await process_video(video_path)

    # Prepare session data
    session_data = {
        'video_frames': video_frames,
        'audio_data': audio_data,
        'transcript': transcript,
        'domain': domain
    }

    # Evaluate teaching session
    results = await evaluator.evaluate_teaching_session(session_data)

    # Print summary
    print_evaluation_summary(results)

    return results


async def process_video(video_path):
    """Extract frames and audio from video file"""
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Extract frames (sample every second)
    frames = []
    for i in range(0, frame_count, int(fps)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    # In a real implementation, we would extract audio here
    # For this example, we'll use dummy audio data
    audio_data = np.random.randn(16000 * (frame_count // fps))

    cap.release()
    return frames, audio_data


def print_evaluation_summary(results):
    """Print a summary of evaluation results"""
    print("\n===== TEACHING EVALUATION SUMMARY =====")
    print(f"Overall Score: {results['overall_score']:.2f}")
    print(f"Understanding Ratio: {results['understanding_ratio']:.2f}")

    print("\nContent Analysis:")
    print(f"  Accuracy: {results['accuracy']:.2f}")
    print(f"  Density: {results['content_metrics']['density']:.2f}")
    print(f"  Structure: {results['content_metrics']['structure_score']:.2f}")

    print("\nDelivery Analysis:")
    print(f"  Engagement: {results['engagement']:.2f}")
    print(f"  Authenticity: {results['authenticity_score']:.2f}")
    print(f"  Eye Contact: {results['visual_metrics']['eye_contact']:.2f}")
    print(f"  Speech Clarity: {results['audio_metrics']['clarity']:.2f}")

    print("\nFeedback Summary:")
    print(results['feedback']['summary'])


async def main():
    # Example usage
    results = await evaluate_teaching_video(
        video_path='teaching_videos/neural_networks_lecture.mp4',
        transcript_path='transcripts/neural_networks_lecture.txt',
        domain='data-science'
    )

    # Save detailed results to file
    with open('evaluation_detailed_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())