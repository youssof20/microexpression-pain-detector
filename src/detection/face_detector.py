"""
Face detection and landmark extraction using MediaPipe Face Mesh.
Returns 468 landmarks in pixel coordinates.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple

# MediaPipe returns normalized coords; we need pixel coords
def _normalized_to_pixel(landmark, frame_width: int, frame_height: int) -> np.ndarray:
    return np.array([
        landmark.x * frame_width,
        landmark.y * frame_height,
        landmark.z  # keep z for depth if needed
    ])


class FaceDetector:
    """
    MediaPipe Face Mesh face detector.
    Returns 468 facial landmarks in pixel coordinates.
    """

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.baseline_landmarks = None

    def detect_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face and return 468 landmarks as (468, 3) array in pixel coordinates.

        Args:
            frame: BGR image (e.g. from webcam)

        Returns:
            landmarks (468, 3) or None if no face detected
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0]
        landmarks = np.array([
            _normalized_to_pixel(p, w, h) for p in lm.landmark
        ], dtype=np.float64)
        return landmarks

    def set_baseline(self, landmarks: np.ndarray) -> None:
        """Store baseline landmarks (e.g. median over 30 frames)."""
        self.baseline_landmarks = landmarks.copy()

    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray, show_all: bool = False) -> np.ndarray:
        """Draw landmark points on frame. If show_all=False, draw only key FACS points."""
        out = frame.copy()
        if landmarks is None:
            return out
        if show_all:
            for i in range(len(landmarks)):
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
                if idx < landmarks.shape[0]:
                    x, y = int(landmarks[idx, 0]), int(landmarks[idx, 1])
                    if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
                        cv2.circle(out, (x, y), 3, (0, 255, 0), -1)
        return out

    def cleanup(self) -> None:
        self.face_mesh.close()
