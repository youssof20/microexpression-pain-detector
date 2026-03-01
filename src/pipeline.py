"""
Single-frame pipeline: detect face -> extract FACS distances -> baseline or score -> draw overlay.
Shared state for baseline collection and current score (used by webrtc callback and app).
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
import threading

from src.detection.face_detector import FaceDetector
from src.detection.feature_extractor import FeatureExtractor
from src.detection.pain_classifier import PainClassifier
from src.visualization.annotator import draw_overlay


# Shared state for webrtc callback (thread-safe for baseline buffer)
_lock = threading.Lock()
_state = {
    "collecting_baseline": False,
    "baseline_buffer": [],
    "baseline_set": False,
    "current_score": 0.0,
    "current_category": "Neutral",
    "current_detailed": {},
}
# Detector instances set by app so callback can run without session_state
_detectors = None


def set_detectors(face_detector: FaceDetector, feature_extractor: FeatureExtractor, pain_classifier: PainClassifier) -> None:
    global _detectors
    _detectors = (face_detector, feature_extractor, pain_classifier)


def get_state() -> Dict[str, Any]:
    with _lock:
        return dict(_state)


def set_collecting_baseline(value: bool) -> None:
    with _lock:
        _state["collecting_baseline"] = value
        if value:
            _state["baseline_buffer"] = []


def set_baseline_set(value: bool) -> None:
    """Mark baseline as set (e.g. after setting from sample video)."""
    with _lock:
        _state["baseline_set"] = value


def process_frame(frame: np.ndarray, face_detector: FaceDetector,
                  feature_extractor: FeatureExtractor,
                  pain_classifier: PainClassifier) -> Tuple[np.ndarray, float, str, Dict]:
    """
    Process one BGR frame: detect face, extract features, update baseline or compute score, draw overlay.
    Returns (annotated_frame, score_0_100, category, detailed_scores).
    """
    out = frame.copy()
    h, w = frame.shape[:2]

    landmarks = face_detector.detect_face(frame)
    if landmarks is None:
        cv2.putText(out, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        with _lock:
            _state["current_score"] = 0.0
            _state["current_category"] = "No face"
            _state["current_detailed"] = {}
        return out, 0.0, "No face", {}

    distances = feature_extractor.extract_distances(landmarks)

    with _lock:
        collecting = _state["collecting_baseline"]
        baseline_set = _state["baseline_set"]

    if collecting:
        with _lock:
            _state["baseline_buffer"].append(distances)
            buf = list(_state["baseline_buffer"])
        if len(buf) >= 30:
            # Compute median baseline
            keys = ["au4", "au9", "au46", "au20", "au6"]
            baseline = {k: float(np.median([d[k] for d in buf])) for k in keys}
            feature_extractor.set_baseline(baseline)
            pain_classifier.set_baseline(baseline)
            with _lock:
                _state["collecting_baseline"] = False
                _state["baseline_set"] = True
                _state["baseline_buffer"] = []

    if not baseline_set:
        cv2.putText(out, "Set baseline (neutral expression)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        out = face_detector.draw_landmarks(out, landmarks, show_all=False)
        with _lock:
            _state["current_score"] = 0.0
            _state["current_category"] = "No baseline"
            _state["current_detailed"] = {}
        return out, 0.0, "No baseline", {}

    deviations = feature_extractor.get_normalized_deviations(landmarks)
    score_100, category, detailed = pain_classifier.classify_pain(deviations)

    with _lock:
        _state["current_score"] = score_100
        _state["current_category"] = category
        _state["current_detailed"] = detailed

    out = draw_overlay(out, landmarks, score_100, category, show_landmarks=True)
    return out, score_100, category, detailed
