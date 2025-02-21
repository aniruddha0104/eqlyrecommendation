import cv2
import numpy as np
import logging
from pathlib import Path
import time
import mediapipe as mp
from tqdm import tqdm
import json
from typing import Dict, List, Optional, Union


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif str(type(obj)) == "<class 'mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList'>":
            return None
        return super(NumpyEncoder, self).default(obj)


class OptimizedProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.frame_count = 0
        self.timestamp = 0
        self.consecutive_detections = 0

        self.output_dir = Path(self.config.get('output_dir', 'analysis_results'))
        self.debug_dir = self.output_dir / 'debug'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.setup_components()

    def setup_components(self):
        try:
            self.mp_face = mp.solutions.face_detection
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles

            self.face_detector = self.mp_face.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.3
            )

            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                smooth_landmarks=True,
                min_detection_confidence=0.2,
                min_tracking_confidence=0.2,
                enable_segmentation=True
            )

            self.cascade_classifier = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            self.face_target_size = (640, 480)
            self.pose_target_size = (1280, 720)

        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}", exc_info=True)
            raise

    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        faces = []
        try:
            enhanced = cv2.resize(frame, self.face_target_size)
            rgb_frame = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

            results = self.face_detector.process(rgb_frame)
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = enhanced.shape
                    x = max(0, int(bbox.xmin * w))
                    y = max(0