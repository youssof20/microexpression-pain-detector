"""
Rule-based pain scoring from FACS feature deviations.
Baseline = median of 30 frames; score 0-100 = weighted sum of deviations.
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque
from src.utils.config import AUWeights, PainThresholds


class PainClassifier:
    """
    Rule-based pain score 0-100 from normalized FACS deviations.
    No CNN. Weights: AU4 0.35, AU9 0.20, AU6 0.20, AU46 0.15, AU20 0.10.
    """

    def __init__(self):
        self.baseline_buffer: List[Dict[str, float]] = []
        self.baseline: Optional[Dict[str, float]] = None
        self.baseline_frames = 30

    def add_baseline_sample(self, distances: Dict[str, float]) -> None:
        """Add one frame's distances to baseline buffer."""
        self.baseline_buffer.append(dict(distances))

    def is_baseline_ready(self) -> bool:
        return len(self.baseline_buffer) >= self.baseline_frames

    def set_baseline_from_buffer(self) -> None:
        """Compute median of buffered distances and set as baseline. Clear buffer."""
        if not self.is_baseline_ready():
            return
        keys = ["au4", "au9", "au46", "au20", "au6"]
        self.baseline = {}
        for k in keys:
            values = [d[k] for d in self.baseline_buffer if k in d]
            self.baseline[k] = float(np.median(values))
        self.baseline_buffer.clear()

    def set_baseline(self, baseline: Dict[str, float]) -> None:
        """Set baseline directly (e.g. from external median)."""
        self.baseline = dict(baseline)

    def get_baseline_buffer_count(self) -> int:
        return len(self.baseline_buffer)

    def classify_pain(self, deviations: Optional[Dict[str, float]]) -> tuple:
        """
        Compute pain score 0-100 and category from normalized deviations.
        Returns (score_0_100, category, detailed_scores_dict).
        """
        if deviations is None or self.baseline is None:
            return 0.0, "Neutral", {}

        score = (
            deviations.get("au4", 0.0) * AUWeights.AU4_BROW_LOWERER
            + deviations.get("au9", 0.0) * AUWeights.AU9_NOSE_WRINKLER
            + deviations.get("au6", 0.0) * AUWeights.AU6_CHEEK_RAISER
            + deviations.get("au46", 0.0) * AUWeights.AU46_EYE_TIGHTENER
            + deviations.get("au20", 0.0) * AUWeights.AU20_LIP_STRETCH
        )
        # Scale to 0-100
        score_100 = min(100.0, max(0.0, score * 100.0))

        if score_100 < PainThresholds.NEUTRAL:
            category = "Neutral"
        elif score_100 < PainThresholds.MILD:
            category = "Mild discomfort"
        else:
            category = "High pain indicators"

        detailed = {
            "au4": deviations.get("au4", 0.0),
            "au9": deviations.get("au9", 0.0),
            "au6": deviations.get("au6", 0.0),
            "au46": deviations.get("au46", 0.0),
            "au20": deviations.get("au20", 0.0),
        }
        return score_100, category, detailed
