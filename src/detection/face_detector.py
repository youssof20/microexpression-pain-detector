"""
Face detection and landmark extraction using MediaPipe Tasks FaceLandmarker.
Works with mediapipe 0.10.31+ (no legacy solutions API).
Returns 468 landmarks in pixel coordinates.
"""

import os
import cv2
import numpy as np
from typing import Optional
import urllib.request
import mediapipe as mp

# MediaPipe Tasks API (mediapipe 0.10.31+)
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# Model URL and local cache path
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "face_landmarker.task")


def _ensure_model() -> str:
    """Download face_landmarker.task if missing. Returns path to .task file."""
    os.makedirs(_MODEL_DIR, exist_ok=True)
    if not os.path.isfile(_MODEL_PATH):
        urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


class FaceDetector:
    """
    MediaPipe Face Landmarker (Tasks API).
    Returns face landmarks as (N, 3) array in pixel coordinates (same 468 topology as before).
    """

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        model_path = _ensure_model()
        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self.baseline_landmarks = None

    def detect_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face and return landmarks as (N, 3) array in pixel coordinates.

        Args:
            frame: BGR image (e.g. from webcam)

        Returns:
            landmarks (N, 3) or None if no face detected
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        # MediaPipe Tasks expect mp.Image (same package as Tasks in 0.10.31+)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return None

        # face_landmarks[0] is a list of NormalizedLandmark with .x, .y, .z (normalized 0-1)
        lm_list = result.face_landmarks[0]
        landmarks = np.array(
            [[p.x * w, p.y * h, p.z] for p in lm_list],
            dtype=np.float64,
        )
        return landmarks

    def set_baseline(self, landmarks: np.ndarray) -> None:
        """Store baseline landmarks (e.g. median over 30 frames)."""
        self.baseline_landmarks = landmarks.copy()

    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray, show_all: bool = False) -> np.ndarray:
        """Draw landmark points on frame. If show_all=False, draw only key FACS points."""
        out = frame.copy()
        if landmarks is None:
            return out
        n = len(landmarks)
        if show_all:
            for i in range(n):
                x, y = int(landmarks[i, 0]), int(landmarks[i, 1])
                if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
                    cv2.circle(out, (x, y), 1, (0, 255, 0), -1)
        else:
            from src.utils.config import MediaPipeIndices
            key_indices = [
                MediaPipeIndices.LEFT_BROW_INNER, MediaPipeIndices.RIGHT_BROW_INNER,
                MediaPipeIndices.LEFT_EYE_TOP, MediaPipeIndices.LEFT_EYE_BOTTOM,
                MediaPipeIndices.RIGHT_EYE_TOP, MediaPipeIndices.RIGHT_EYE_BOTTOM,
                MediaPipeIndices.NOSE_TIP, MediaPipeIndices.NOSE_ROOT,
                MediaPipeIndices.MOUTH_LEFT, MediaPipeIndices.MOUTH_RIGHT,
                MediaPipeIndices.LEFT_CHEEK, MediaPipeIndices.RIGHT_CHEEK,
            ]
            for idx in key_indices:
                if idx < n:
                    x, y = int(landmarks[idx, 0]), int(landmarks[idx, 1])
                    if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
                        cv2.circle(out, (x, y), 3, (0, 255, 0), -1)
        return out

    def cleanup(self) -> None:
        """Release resources. FaceLandmarker has no close() in Tasks API; no-op."""
        pass
