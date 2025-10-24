"""
Configuration constants for the Pain Detector project.
Centralizes all magic numbers and settings.
"""

# Simplified landmark indices for FACS Action Units
# Based on 9-point simplified face model
class LandmarkIndices:
    # Brow landmarks
    LEFT_INNER_BROW = 0
    RIGHT_INNER_BROW = 1
    
    # Eye landmarks
    LEFT_EYE_TOP = 2
    LEFT_EYE_BOTTOM = 3
    RIGHT_EYE_TOP = 4
    RIGHT_EYE_BOTTOM = 5
    
    # Nose landmarks
    NOSE_TIP = 6
    
    # Mouth landmarks
    MOUTH_TOP = 7
    MOUTH_BOTTOM = 8

# Pain score thresholds
class PainThresholds:
    NONE = 0.3
    MILD = 0.6
    MODERATE = 1.0

# Visualization colors (BGR format for OpenCV)
class Colors:
    GREEN = (0, 255, 0)      # No pain
    YELLOW = (0, 255, 255)   # Mild pain
    RED = (0, 0, 255)        # Moderate pain
    BLUE = (255, 0, 0)       # Landmarks
    WHITE = (255, 255, 255)  # Text
    BLACK = (0, 0, 0)        # Background

# FACS Action Unit weights for pain scoring
class AUWeights:
    AU4_BROW_LOWERER = 0.3
    AU6_ORBITAL_TIGHTENING = 0.25
    AU7_LID_TIGHTENER = 0.2
    AU9_NOSE_WRINKLER = 0.15
    AU10_UPPER_LIP_RAISER = 0.1

# Model configuration
class ModelConfig:
    FACE_CROP_SIZE = 224
    CONFIDENCE_THRESHOLD = 0.5
    TEMPORAL_SMOOTHING_FACTOR = 0.7
    MIN_FACE_SIZE = 50  # Minimum face size in pixels

# Streamlit configuration
class StreamlitConfig:
    PAGE_TITLE = "Microexpression Pain Detector"
    PAGE_ICON = "😷"
    LAYOUT = "wide"
    SIDEBAR_STATE = "expanded"

# File paths
class Paths:
    MODELS_DIR = "src/models/pretrained"
    SAMPLE_VIDEOS_DIR = "data/sample_videos"
    DATASET_INSTRUCTIONS = "data/dataset_instructions.md"
