"""
Face detection and landmark extraction using OpenCV.
Provides basic face detection with simplified landmark estimation.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from src.utils.config import LandmarkIndices, ModelConfig


class FaceDetector:
    """
    OpenCV-based face detector with simplified landmark estimation.
    Optimized for real-time performance on CPU.
    """
    
    def __init__(self):
        """Initialize OpenCV face detection."""
        # Load Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Cache for face normalization
        self.baseline_face_size = None
        self.baseline_landmarks = None
        
    def detect_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face and extract simplified landmarks from frame.
        
        Args:
            frame: Input BGR image frame
            
        Returns:
            Simplified landmarks array (9, 3) or None if no face detected
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with more sensitive parameters
        faces = self.face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(30, 30))
        
        if len(faces) > 0:
            # Get the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            # Check if face is large enough
            if w >= ModelConfig.MIN_FACE_SIZE and h >= ModelConfig.MIN_FACE_SIZE:
                # Create simplified landmarks based on face bounding box
                landmarks = self._create_simplified_landmarks(x, y, w, h)
                return landmarks
                
        return None
    
    def _create_simplified_landmarks(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """
        Create simplified landmarks based on face bounding box.
        
        Args:
            x, y, w, h: Face bounding box coordinates
            
        Returns:
            Simplified landmarks array (9, 3)
        """
        # Create 9 key landmarks based on face proportions
        landmarks = np.zeros((9, 3))
        
        # Face center
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Inner brows (estimated positions)
        landmarks[0] = [center_x - w * 0.15, y + h * 0.25, 0]  # Left inner brow
        landmarks[1] = [center_x + w * 0.15, y + h * 0.25, 0]     # Right inner brow
        
        # Eyes (estimated positions)
        landmarks[2] = [center_x - w * 0.2, y + h * 0.35, 0]    # Left eye top
        landmarks[3] = [center_x - w * 0.2, y + h * 0.4, 0]     # Left eye bottom
        landmarks[4] = [center_x + w * 0.2, y + h * 0.35, 0]    # Right eye top
        landmarks[5] = [center_x + w * 0.2, y + h * 0.4, 0]     # Right eye bottom
        
        # Nose tip
        landmarks[6] = [center_x, y + h * 0.5, 0]
        
        # Mouth
        landmarks[7] = [center_x, y + h * 0.7, 0]               # Mouth top
        landmarks[8] = [center_x, y + h * 0.75, 0]              # Mouth bottom
        
        return landmarks
    
    def _is_face_large_enough(self, landmarks: np.ndarray, width: int, height: int) -> bool:
        """
        Check if detected face meets minimum size requirements.
        
        Args:
            landmarks: Face landmarks array
            width: Frame width
            height: Frame height
            
        Returns:
            True if face is large enough for reliable analysis
        """
        # Calculate face bounding box
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        
        face_width = np.max(x_coords) - np.min(x_coords)
        face_height = np.max(y_coords) - np.min(y_coords)
        
        # Check minimum size relative to frame
        min_width = width * 0.1  # 10% of frame width
        min_height = height * 0.1  # 10% of frame height
        
        return face_width >= min_width and face_height >= min_height
    
    def get_face_crop(self, frame: np.ndarray, landmarks: np.ndarray, 
                     padding: float = 0.2) -> Optional[np.ndarray]:
        """
        Extract face crop from frame for CNN processing.
        
        Args:
            frame: Input frame
            landmarks: Face landmarks
            padding: Padding around face bounding box (0.0-1.0)
            
        Returns:
            Cropped face image or None if crop is invalid
        """
        try:
            # Calculate bounding box
            x_coords = landmarks[:, 0]
            y_coords = landmarks[:, 1]
            
            x_min = int(np.min(x_coords))
            x_max = int(np.max(x_coords))
            y_min = int(np.min(y_coords))
            y_max = int(np.max(y_coords))
            
            # Add padding
            width = x_max - x_min
            height = y_max - y_min
            
            pad_x = int(width * padding)
            pad_y = int(height * padding)
            
            x_min = max(0, x_min - pad_x)
            x_max = min(frame.shape[1], x_max + pad_x)
            y_min = max(0, y_min - pad_y)
            y_max = min(frame.shape[0], y_max + pad_y)
            
            # Extract crop
            face_crop = frame[y_min:y_max, x_min:x_max]
            
            # Resize to model input size
            if face_crop.size > 0:
                face_crop = cv2.resize(face_crop, (ModelConfig.FACE_CROP_SIZE, ModelConfig.FACE_CROP_SIZE))
                return face_crop
                
        except Exception as e:
            print(f"Error cropping face: {e}")
            
        return None
    
    def set_baseline(self, landmarks: np.ndarray):
        """
        Set baseline landmarks for normalization.
        Should be called when subject has neutral expression.
        
        Args:
            landmarks: Neutral expression landmarks
        """
        self.baseline_landmarks = landmarks.copy()
        
        # Calculate baseline face size for normalization
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        self.baseline_face_size = np.sqrt(
            (np.max(x_coords) - np.min(x_coords)) * 
            (np.max(y_coords) - np.min(y_coords))
        )
    
    def get_normalized_landmarks(self, landmarks: np.ndarray) -> Optional[np.ndarray]:
        """
        Normalize landmarks relative to baseline.
        
        Args:
            landmarks: Current landmarks
            
        Returns:
            Normalized landmarks or None if no baseline set
        """
        if self.baseline_landmarks is None:
            return landmarks
            
        # Simple normalization: subtract baseline
        normalized = landmarks - self.baseline_landmarks
        
        # Scale by face size if available
        if self.baseline_face_size is not None:
            current_face_size = np.sqrt(
                (np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])) * 
                (np.max(landmarks[:, 1]) - np.min(landmarks[:, 1]))
            )
            scale_factor = self.baseline_face_size / current_face_size
            normalized *= scale_factor
            
        return normalized
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray, 
                      show_all: bool = False) -> np.ndarray:
        """
        Draw landmarks on frame for visualization.
        
        Args:
            frame: Input frame
            landmarks: Face landmarks
            show_all: If True, show all landmarks; if False, show only key landmarks
            
        Returns:
            Frame with landmarks drawn
        """
        frame_copy = frame.copy()
        
        if show_all:
            # Draw all landmarks as small circles
            for landmark in landmarks:
                x, y = int(landmark[0]), int(landmark[1])
                cv2.circle(frame_copy, (x, y), 1, (0, 255, 0), -1)
        else:
            # Draw only key landmarks for pain detection
            key_indices = [
                LandmarkIndices.LEFT_INNER_BROW,
                LandmarkIndices.RIGHT_INNER_BROW,
                LandmarkIndices.LEFT_EYE_TOP,
                LandmarkIndices.LEFT_EYE_BOTTOM,
                LandmarkIndices.RIGHT_EYE_TOP,
                LandmarkIndices.RIGHT_EYE_BOTTOM,
                LandmarkIndices.NOSE_TIP,
                LandmarkIndices.MOUTH_TOP,
                LandmarkIndices.MOUTH_BOTTOM
            ]
            
            for idx in key_indices:
                if idx < len(landmarks):
                    x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                    cv2.circle(frame_copy, (x, y), 3, (0, 255, 0), -1)
        
        return frame_copy
    
    def cleanup(self):
        """Clean up resources."""
        # OpenCV doesn't need explicit cleanup
        pass
