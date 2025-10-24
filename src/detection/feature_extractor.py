"""
FACS-based feature extraction for pain detection.
Implements Facial Action Coding System (FACS) Action Units relevant to pain expressions.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from src.utils.config import LandmarkIndices


class FeatureExtractor:
    """
    Extracts FACS-based features from facial landmarks for pain detection.
    Focuses on Action Units commonly associated with pain expressions.
    """
    
    def __init__(self):
        """Initialize feature extractor with FACS Action Unit mappings."""
        self.au_features = {}
        self.baseline_features = None
        
    def extract_all_features(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Extract all pain-relevant FACS features from landmarks.
        
        Args:
            landmarks: Face landmarks array (468, 3)
            
        Returns:
            Dictionary of feature names and values
        """
        features = {}
        
        # Extract individual Action Units
        features['au4'] = self._extract_au4_brow_lowerer(landmarks)
        features['au6'] = self._extract_au6_orbital_tightening(landmarks)
        features['au7'] = self._extract_au7_lid_tightener(landmarks)
        features['au9'] = self._extract_au9_nose_wrinkler(landmarks)
        features['au10'] = self._extract_au10_upper_lip_raiser(landmarks)
        features['au43'] = self._extract_au43_eye_closure(landmarks)
        
        # Extract composite features
        features['eye_tightening'] = self._extract_eye_tightening_composite(landmarks)
        features['mouth_tension'] = self._extract_mouth_tension(landmarks)
        features['overall_tension'] = self._extract_overall_tension(landmarks)
        
        return features
    
    def _extract_au4_brow_lowerer(self, landmarks: np.ndarray) -> float:
        """
        Extract AU4 (Brow Lowerer) - inner brow lowering.
        Pain indicator: brows pulled down and together.
        
        Args:
            landmarks: Face landmarks
            
        Returns:
            AU4 intensity (0.0-1.0)
        """
        try:
            # Get inner brow landmarks
            left_inner = landmarks[LandmarkIndices.LEFT_INNER_BROW]
            right_inner = landmarks[LandmarkIndices.RIGHT_INNER_BROW]
            
            # Calculate vertical distance between inner brows
            brow_distance = abs(left_inner[1] - right_inner[1])
            
            # Calculate horizontal distance (should decrease when brows lower)
            brow_horizontal = abs(left_inner[0] - right_inner[0])
            
            # Normalize by face width (approximate)
            face_width = self._get_face_width(landmarks)
            normalized_distance = brow_distance / face_width if face_width > 0 else 0
            
            # AU4 is stronger when brows are closer together and lower
            au4_score = min(1.0, normalized_distance * 5)  # Increased sensitivity
            
            return au4_score
            
        except (IndexError, ValueError):
            return 0.0
    
    def _extract_au6_orbital_tightening(self, landmarks: np.ndarray) -> float:
        """
        Extract AU6 (Orbital Tightening) - eye aperture reduction.
        Pain indicator: eyes squinting or tightening.
        
        Args:
            landmarks: Face landmarks
            
        Returns:
            AU6 intensity (0.0-1.0)
        """
        try:
            # Get eye landmarks
            left_eye_top = landmarks[LandmarkIndices.LEFT_EYE_TOP]
            left_eye_bottom = landmarks[LandmarkIndices.LEFT_EYE_BOTTOM]
            right_eye_top = landmarks[LandmarkIndices.RIGHT_EYE_TOP]
            right_eye_bottom = landmarks[LandmarkIndices.RIGHT_EYE_BOTTOM]
            
            # Calculate eye aperture (vertical distance)
            left_aperture = abs(left_eye_top[1] - left_eye_bottom[1])
            right_aperture = abs(right_eye_top[1] - right_eye_bottom[1])
            
            # Average aperture
            avg_aperture = (left_aperture + right_aperture) / 2
            
            # Normalize by face height
            face_height = self._get_face_height(landmarks)
            normalized_aperture = avg_aperture / face_height if face_height > 0 else 0
            
            # AU6 is stronger when aperture is smaller (eyes more closed)
            # Invert so smaller aperture = higher score
            au6_score = max(0.0, 1.0 - normalized_aperture * 5)  # Increased sensitivity
            
            return au6_score
            
        except (IndexError, ValueError):
            return 0.0
    
    def _extract_au7_lid_tightener(self, landmarks: np.ndarray) -> float:
        """
        Extract AU7 (Lid Tightener) - upper eyelid tightening.
        Pain indicator: upper eyelids pulled down.
        
        Args:
            landmarks: Face landmarks
            
        Returns:
            AU7 intensity (0.0-1.0)
        """
        try:
            # Get eyelid landmarks (approximate using eye landmarks)
            left_eye_top = landmarks[LandmarkIndices.LEFT_EYE_TOP]
            left_eye_bottom = landmarks[LandmarkIndices.LEFT_EYE_BOTTOM]
            right_eye_top = landmarks[LandmarkIndices.RIGHT_EYE_TOP]
            right_eye_bottom = landmarks[LandmarkIndices.RIGHT_EYE_BOTTOM]
            
            # Calculate eye opening ratio
            left_opening = abs(left_eye_top[1] - left_eye_bottom[1])
            right_opening = abs(right_eye_top[1] - right_eye_bottom[1])
            
            # AU7 is similar to AU6 but focuses on upper lid
            avg_opening = (left_opening + right_opening) / 2
            face_height = self._get_face_height(landmarks)
            normalized_opening = avg_opening / face_height if face_height > 0 else 0
            
            au7_score = max(0.0, 1.0 - normalized_opening * 2.5)
            
            return au7_score
            
        except (IndexError, ValueError):
            return 0.0
    
    def _extract_au9_nose_wrinkler(self, landmarks: np.ndarray) -> float:
        """
        Extract AU9 (Nose Wrinkler) - nose wrinkling.
        Pain indicator: nose bridge wrinkling.
        
        Args:
            landmarks: Face landmarks
            
        Returns:
            AU9 intensity (0.0-1.0)
        """
        try:
            # Get nose landmarks
            nose_tip = landmarks[LandmarkIndices.NOSE_TIP]
            nose_bridge = landmarks[LandmarkIndices.NOSE_BRIDGE]
            
            # Calculate nose bridge to tip distance
            nose_length = abs(nose_tip[1] - nose_bridge[1])
            
            # Normalize by face height
            face_height = self._get_face_height(landmarks)
            normalized_length = nose_length / face_height if face_height > 0 else 0
            
            # AU9 is detected by changes in nose bridge area
            # This is a simplified implementation
            au9_score = min(1.0, normalized_length * 2)
            
            return au9_score
            
        except (IndexError, ValueError):
            return 0.0
    
    def _extract_au10_upper_lip_raiser(self, landmarks: np.ndarray) -> float:
        """
        Extract AU10 (Upper Lip Raiser) - upper lip raising.
        Pain indicator: upper lip pulled up.
        
        Args:
            landmarks: Face landmarks
            
        Returns:
            AU10 intensity (0.0-1.0)
        """
        try:
            # Get mouth landmarks (simplified)
            mouth_top = landmarks[LandmarkIndices.MOUTH_TOP]
            mouth_bottom = landmarks[LandmarkIndices.MOUTH_BOTTOM]
            
            # Calculate mouth height
            mouth_height = abs(mouth_top[1] - mouth_bottom[1])
            
            # Calculate distance from mouth to nose
            nose_tip = landmarks[LandmarkIndices.NOSE_TIP]
            mouth_nose_distance = abs(mouth_top[1] - nose_tip[1])
            
            # Normalize
            face_height = self._get_face_height(landmarks)
            normalized_height = mouth_height / face_height if face_height > 0 else 0
            normalized_distance = mouth_nose_distance / face_height if face_height > 0 else 0
            
            # AU10 is stronger when upper lip is raised (smaller mouth-nose distance)
            au10_score = max(0.0, 1.0 - normalized_distance * 1.5)
            
            return au10_score
            
        except (IndexError, ValueError):
            return 0.0
    
    def _extract_au43_eye_closure(self, landmarks: np.ndarray) -> float:
        """
        Extract AU43 (Eye Closure) - eye closing.
        Pain indicator: eyes closing or squinting.
        
        Args:
            landmarks: Face landmarks
            
        Returns:
            AU43 intensity (0.0-1.0)
        """
        try:
            # Similar to AU6/AU7 but focused on complete closure
            left_eye_top = landmarks[LandmarkIndices.LEFT_EYE_TOP]
            left_eye_bottom = landmarks[LandmarkIndices.LEFT_EYE_BOTTOM]
            right_eye_top = landmarks[LandmarkIndices.RIGHT_EYE_TOP]
            right_eye_bottom = landmarks[LandmarkIndices.RIGHT_EYE_BOTTOM]
            
            # Calculate eye opening
            left_opening = abs(left_eye_top[1] - left_eye_bottom[1])
            right_opening = abs(right_eye_top[1] - right_eye_bottom[1])
            
            avg_opening = (left_opening + right_opening) / 2
            face_height = self._get_face_height(landmarks)
            normalized_opening = avg_opening / face_height if face_height > 0 else 0
            
            # AU43 is strongest when eyes are nearly closed
            au43_score = max(0.0, 1.0 - normalized_opening * 4)
            
            return au43_score
            
        except (IndexError, ValueError):
            return 0.0
    
    def _extract_eye_tightening_composite(self, landmarks: np.ndarray) -> float:
        """
        Extract composite eye tightening feature combining AU6, AU7, AU43.
        
        Args:
            landmarks: Face landmarks
            
        Returns:
            Composite eye tightening score (0.0-1.0)
        """
        au6 = self._extract_au6_orbital_tightening(landmarks)
        au7 = self._extract_au7_lid_tightener(landmarks)
        au43 = self._extract_au43_eye_closure(landmarks)
        
        # Weighted combination
        composite_score = (au6 * 0.4 + au7 * 0.3 + au43 * 0.3)
        return min(1.0, composite_score)
    
    def _extract_mouth_tension(self, landmarks: np.ndarray) -> float:
        """
        Extract mouth tension features.
        
        Args:
            landmarks: Face landmarks
            
        Returns:
            Mouth tension score (0.0-1.0)
        """
        try:
            # Get mouth landmarks (simplified)
            mouth_top = landmarks[LandmarkIndices.MOUTH_TOP]
            mouth_bottom = landmarks[LandmarkIndices.MOUTH_BOTTOM]
            
            # Calculate mouth dimensions
            mouth_height = abs(mouth_top[1] - mouth_bottom[1])
            
            # Estimate mouth width from face proportions
            face_width = self._get_face_width(landmarks)
            estimated_mouth_width = face_width * 0.4  # Rough estimate
            
            # Normalize
            face_height = self._get_face_height(landmarks)
            normalized_height = mouth_height / face_height if face_height > 0 else 0
            normalized_width = estimated_mouth_width / face_width if face_width > 0 else 0
            
            # Mouth tension can be detected by changes in mouth shape
            tension_score = min(1.0, (normalized_height + normalized_width) * 0.5)
            
            return tension_score
            
        except (IndexError, ValueError):
            return 0.0
    
    def _extract_overall_tension(self, landmarks: np.ndarray) -> float:
        """
        Extract overall facial tension score.
        
        Args:
            landmarks: Face landmarks
            
        Returns:
            Overall tension score (0.0-1.0)
        """
        # Extract individual features directly to avoid recursion
        au4 = self._extract_au4_brow_lowerer(landmarks)
        au6 = self._extract_au6_orbital_tightening(landmarks)
        au7 = self._extract_au7_lid_tightener(landmarks)
        au9 = self._extract_au9_nose_wrinkler(landmarks)
        au10 = self._extract_au10_upper_lip_raiser(landmarks)
        eye_tightening = (au6 * 0.4 + au7 * 0.3 + self._extract_au43_eye_closure(landmarks) * 0.3)
        mouth_tension = self._extract_mouth_tension(landmarks)
        
        # Weighted combination of key pain indicators
        tension_score = (
            au4 * 0.25 +
            eye_tightening * 0.35 +
            au9 * 0.15 +
            au10 * 0.15 +
            mouth_tension * 0.10
        )
        
        return min(1.0, tension_score)
    
    def _get_face_width(self, landmarks: np.ndarray) -> float:
        """Calculate approximate face width from landmarks."""
        try:
            x_coords = landmarks[:, 0]
            return np.max(x_coords) - np.min(x_coords)
        except:
            return 100.0  # Default fallback
    
    def _get_face_height(self, landmarks: np.ndarray) -> float:
        """Calculate approximate face height from landmarks."""
        try:
            y_coords = landmarks[:, 1]
            return np.max(y_coords) - np.min(y_coords)
        except:
            return 100.0  # Default fallback
    
    def set_baseline(self, landmarks: np.ndarray):
        """
        Set baseline features for normalization.
        
        Args:
            landmarks: Neutral expression landmarks
        """
        self.baseline_features = self.extract_all_features(landmarks)
    
    def get_normalized_features(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Get features normalized relative to baseline.
        
        Args:
            landmarks: Current landmarks
            
        Returns:
            Normalized features
        """
        current_features = self.extract_all_features(landmarks)
        
        if self.baseline_features is None:
            return current_features
        
        # Simple normalization: subtract baseline
        normalized = {}
        for key in current_features:
            if key in self.baseline_features:
                normalized[key] = max(0.0, current_features[key] - self.baseline_features[key])
            else:
                normalized[key] = current_features[key]
        
        return normalized
