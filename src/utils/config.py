"""
Configuration constants for the Pain Detector project.
Uses MediaPipe Face Mesh 468-landmark indices for FACS.
"""

# MediaPipe Face Mesh landmark indices (468 points)
# https://developers.google.com/mediapipe/solutions/vision/face_landmarker
class MediaPipeIndices:
    # Nose
    NOSE_TIP = 1
    NOSE_BOTTOM = 2
    NOSE_ROOT = 168
    NOSE_LEFT = 98
    NOSE_RIGHT = 327

    # Left eye
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    LEFT_EYE_INNER = 33
    LEFT_EYE_OUTER = 133

    # Right eye
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263

    # Brows (AU4 brow lowerer)
    LEFT_BROW_INNER = 63
    LEFT_BROW_OUTER = 70
    RIGHT_BROW_INNER = 336
    RIGHT_BROW_OUTER = 300

    # Mouth (AU20 lip corner)
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291

    # Cheeks (AU6 cheek raiser)
    LEFT_CHEEK = 234
    RIGHT_CHEEK = 454


# Pain score thresholds (0-100 scale)
class PainThresholds:
    NEUTRAL = 30
    MILD = 60
    HIGH = 100


# FACS weights for pain score: AU4 0.35, AU9 0.20, AU6 0.20, AU46 0.15, AU20 0.10
class AUWeights:
    AU4_BROW_LOWERER = 0.35
    AU9_NOSE_WRINKLER = 0.20
    AU6_CHEEK_RAISER = 0.20
    AU46_EYE_TIGHTENER = 0.15
    AU20_LIP_STRETCH = 0.10


# Visualization colors (BGR for OpenCV)
class Colors:
    GREEN = (0, 255, 0)
    YELLOW = (0, 255, 255)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)


class StreamlitConfig:
    PAGE_TITLE = "Microexpression Pain Detector"
    PAGE_ICON = "🔬"
    LAYOUT = "wide"
