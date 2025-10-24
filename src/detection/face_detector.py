"""
Face detection and landmark extraction using MediaPipe.
Provides robust real-time face detection with 468 facial landmarks.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List
from src.utils.config import LandmarkIndices, ModelConfig


class FaceDetector:
    """
    MediaPipe-based face detector for extracting facial landmarks.
    Optimized for real-time performance on CPU.
    """
    
    def __init__(self):
        """Initialize MediaPipe face mesh solution."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize face mesh with optimized settings
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,  # Focus on single face for pain detection
            refine_landmarks=True,  # Use refined landmarks for better accuracy
            min_detection_confidence=ModelConfig.CONFIDENCE_THRESHOLD,
            min_tracking_confidence=0.5
        )
        
        # Cache for face normalization
        self.baseline_face_size = None
        self.baseline_landmarks = None
        
    def detect_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face and extract landmarks from frame.
        
        Args:
            frame: Input BGR image frame
            
        Returns:
            Normalized landmarks array (468, 3) or None if no face detected
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            # Get first (and only) face
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert to numpy array
            landmarks = np.array([
                [landmark.x, landmark.y, landmark.z] 
                for landmark in face_landmarks.landmark
            ])
            
            # Normalize landmarks to frame dimensions
            height, width = frame.shape[:2]
            landmarks[:, 0] *= width   # x coordinates
            landmarks[:, 1] *= height   # y coordinates
            
            # Check if face is large enough
            if self._is_face_large_enough(landmarks, width, height):
                return landmarks
                
        return None
    
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
        """Clean up MediaPipe resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
