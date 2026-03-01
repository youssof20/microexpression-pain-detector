"""
FACS-based feature extraction for pain detection.
Extracts landmark distances for AU4, AU9, AU46, AU20, AU6 using MediaPipe Face Mesh.
"""

import numpy as np
from typing import Dict, Optional
from src.utils.config import MediaPipeIndices


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def _norm_by_face(landmarks: np.ndarray, value: float) -> float:
    """Normalize a distance by face size (inter-ocular or face bbox)."""
    left_eye = landmarks[MediaPipeIndices.LEFT_EYE_INNER]
    right_eye = landmarks[MediaPipeIndices.RIGHT_EYE_INNER]
    ref = _dist(left_eye, right_eye)
    if ref <= 1e-6:
        return 0.0
    return value / ref


class FeatureExtractor:
    """
    Extracts FACS-relevant distances from MediaPipe 468 landmarks.
    Returns raw distance values (for baseline) and optional normalized values.
    """

    def __init__(self):
        self.baseline = None  # dict of median values: au4, au9, au46, au20, au6

    def extract_distances(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Extract the 5 FACS-relevant distances (raw, in pixels or normalized by face).
        Used for baseline and for current frame.
        """
        out = {}
        # AU4: brow lowerer — distance between brow and eye (vertical gap)
        left_brow = landmarks[MediaPipeIndices.LEFT_BROW_INNER]
        left_eye_top = landmarks[MediaPipeIndices.LEFT_EYE_TOP]
        right_brow = landmarks[MediaPipeIndices.RIGHT_BROW_INNER]
        right_eye_top = landmarks[MediaPipeIndices.RIGHT_EYE_TOP]
        au4_left = abs(left_brow[1] - left_eye_top[1])
        au4_right = abs(right_brow[1] - right_eye_top[1])
        out["au4"] = (au4_left + au4_right) / 2.0

        # AU9: nose wrinkler — upper nose bridge "width" (e.g. distance between nose sides or nose length)
        nose_root = landmarks[MediaPipeIndices.NOSE_ROOT]
        nose_left = landmarks[MediaPipeIndices.NOSE_LEFT]
        nose_right = landmarks[MediaPipeIndices.NOSE_RIGHT]
        out["au9"] = _dist(nose_left, nose_right)

        # AU46: eye tightener — eye aperture height (average of left and right)
        left_h = abs(landmarks[MediaPipeIndices.LEFT_EYE_TOP][1] - landmarks[MediaPipeIndices.LEFT_EYE_BOTTOM][1])
        right_h = abs(landmarks[MediaPipeIndices.RIGHT_EYE_TOP][1] - landmarks[MediaPipeIndices.RIGHT_EYE_BOTTOM][1])
        out["au46"] = (left_h + right_h) / 2.0

        # AU20: lip corner — horizontal lip stretch (mouth corner distance)
        mouth_left = landmarks[MediaPipeIndices.MOUTH_LEFT]
        mouth_right = landmarks[MediaPipeIndices.MOUTH_RIGHT]
        out["au20"] = _dist(mouth_left, mouth_right)

        # AU6: cheek raiser — cheek-to-eye distance (average left and right)
        left_cheek = landmarks[MediaPipeIndices.LEFT_CHEEK]
        left_eye_c = (landmarks[MediaPipeIndices.LEFT_EYE_TOP] + landmarks[MediaPipeIndices.LEFT_EYE_BOTTOM]) / 2
        right_cheek = landmarks[MediaPipeIndices.RIGHT_CHEEK]
        right_eye_c = (landmarks[MediaPipeIndices.RIGHT_EYE_TOP] + landmarks[MediaPipeIndices.RIGHT_EYE_BOTTOM]) / 2
        out["au6"] = (_dist(left_cheek, left_eye_c) + _dist(right_cheek, right_eye_c)) / 2.0

        return out

    def set_baseline(self, baseline: Dict[str, float]) -> None:
        """Set baseline (median of 30 frames) for each AU distance."""
        self.baseline = dict(baseline)

    def get_normalized_deviations(self, landmarks: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Get current distances and compute 0-1 normalized deviation from baseline.
        Positive deviation = more "pain-like" (brow lower, nose wrinkle, eye squint, lip stretch, cheek raise).
        """
        current = self.extract_distances(landmarks)
        if self.baseline is None:
            return None
        deviations = {}
        for key in ["au4", "au9", "au46", "au20", "au6"]:
            b = self.baseline.get(key)
            c = current[key]
            if b is None or b <= 1e-9:
                deviations[key] = 0.0
                continue
            # AU4: brow lowerer — smaller gap = pain. dev = (baseline - current) / baseline
            if key == "au4":
                dev = (b - c) / b
            # AU9: nose wrinkler — narrower width = pain. dev = (baseline - current) / baseline
            elif key == "au9":
                dev = (b - c) / b
            # AU46: eye tightener — smaller aperture = pain. dev = (baseline - current) / baseline
            elif key == "au46":
                dev = (b - c) / b
            # AU20: lip stretch — wider = pain. dev = (current - baseline) / baseline
            elif key == "au20":
                dev = (c - b) / b
            # AU6: cheek raiser — smaller distance = pain. dev = (baseline - current) / baseline
            elif key == "au6":
                dev = (b - c) / b
            else:
                dev = 0.0
            deviations[key] = max(0.0, min(1.0, dev))
        return deviations
