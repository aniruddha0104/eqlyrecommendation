# test_script.py
import cv2
import os
from features.eye_tracking import EyeTracker
from features.visual_cues import VisualCuesAnalyzer


def test_single_video():
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)  # Use webcam

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    print("Initializing trackers...")
    eye_tracker = EyeTracker()
    visual_analyzer = VisualCuesAnalyzer({})

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Process frame
        eye_metrics = eye_tracker.track_eyes(frame)
        visual_metrics = visual_analyzer.analyze_frame(frame)

        # Draw some basic visualization
        if eye_metrics['current_state']['attention_target'] != 'unknown':
            cv2.putText(
                frame,
                f"Looking: {eye_metrics['current_state']['attention_target']}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        # Display frame
        cv2.imshow('Eye Tracking', frame)

        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_single_video()