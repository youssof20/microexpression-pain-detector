"""
Pain classification using rule-based scoring and optional CNN model.
Combines FACS features to determine pain level and category.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from collections import deque
import torch
import torch.nn as nn
from src.utils.config import PainThresholds, AUWeights, ModelConfig


class PainClassifier:
    """
    Pain classifier combining rule-based FACS scoring with optional CNN model.
    Provides real-time pain level assessment with temporal smoothing.
    """
    
    def __init__(self, use_cnn: bool = False, model_path: Optional[str] = None):
        """
        Initialize pain classifier.
        
        Args:
            use_cnn: Whether to use CNN model for enhanced predictions
            model_path: Path to pre-trained CNN model
        """
        self.use_cnn = use_cnn
        self.cnn_model = None
        
        # Temporal smoothing buffer
        self.score_history = deque(maxlen=10)  # Keep last 10 scores
        self.smoothed_score = 0.0
        
        # Load CNN model if requested
        if use_cnn and model_path:
            self._load_cnn_model(model_path)
    
    def classify_pain(self, features: Dict[str, float], 
                     face_crop: Optional[np.ndarray] = None) -> Tuple[float, str, Dict[str, float]]:
        """
        Classify pain level from features and optional face crop.
        
        Args:
            features: FACS features dictionary
            face_crop: Optional face crop for CNN processing
            
        Returns:
            Tuple of (pain_score, category, detailed_scores)
        """
        # Get rule-based score
        rule_score = self._calculate_rule_based_score(features)
        
        # Get CNN score if available
        cnn_score = 0.0
        if self.use_cnn and self.cnn_model is not None and face_crop is not None:
            cnn_score = self._predict_cnn_score(face_crop)
        
        # Combine scores
        if self.use_cnn and cnn_score > 0:
            # Weighted combination: 70% rule-based, 30% CNN
            combined_score = rule_score * 0.7 + cnn_score * 0.3
        else:
            combined_score = rule_score
        
        # Apply temporal smoothing
        smoothed_score = self._apply_temporal_smoothing(combined_score)
        
        # Determine category
        category = self._score_to_category(smoothed_score)
        
        # Prepare detailed scores
        detailed_scores = {
            'rule_based': rule_score,
            'cnn': cnn_score,
            'combined': combined_score,
            'smoothed': smoothed_score,
            'category': category
        }
        
        return smoothed_score, category, detailed_scores
    
    def _calculate_rule_based_score(self, features: Dict[str, float]) -> float:
        """
        Calculate pain score using rule-based FACS weighting.
        
        Args:
            features: FACS features dictionary
            
        Returns:
            Pain score (0.0-1.0)
        """
        # Weighted combination of FACS Action Units
        score = (
            features.get('au4', 0.0) * AUWeights.AU4_BROW_LOWERER +
            features.get('au6', 0.0) * AUWeights.AU6_ORBITAL_TIGHTENING +
            features.get('au7', 0.0) * AUWeights.AU7_LID_TIGHTENER +
            features.get('au9', 0.0) * AUWeights.AU9_NOSE_WRINKLER +
            features.get('au10', 0.0) * AUWeights.AU10_UPPER_LIP_RAISER
        )
        
        # Add composite features
        score += features.get('eye_tightening', 0.0) * 0.2
        score += features.get('overall_tension', 0.0) * 0.1
        
        # Normalize to 0-1 range
        return min(1.0, max(0.0, score))
    
    def _predict_cnn_score(self, face_crop: np.ndarray) -> float:
        """
        Predict pain score using CNN model.
        
        Args:
            face_crop: Face crop image (224x224x3)
            
        Returns:
            CNN pain score (0.0-1.0)
        """
        try:
            if self.cnn_model is None:
                return 0.0
            
            # Preprocess image
            processed_image = self._preprocess_image(face_crop)
            
            # Predict
            with torch.no_grad():
                prediction = self.cnn_model(processed_image)
                score = torch.sigmoid(prediction).item()
            
            return score
            
        except Exception as e:
            print(f"CNN prediction error: {e}")
            return 0.0
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for CNN input.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        image_rgb = image[:, :, ::-1]
        
        # Normalize to [0, 1]
        image_normalized = image_rgb.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1)
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def _apply_temporal_smoothing(self, score: float) -> float:
        """
        Apply temporal smoothing to reduce jitter.
        
        Args:
            score: Current pain score
            
        Returns:
            Smoothed score
        """
        # Add current score to history
        self.score_history.append(score)
        
        # Calculate smoothed score using exponential moving average
        if len(self.score_history) == 1:
            self.smoothed_score = score
        else:
            alpha = ModelConfig.TEMPORAL_SMOOTHING_FACTOR
            self.smoothed_score = alpha * score + (1 - alpha) * self.smoothed_score
        
        return self.smoothed_score
    
    def _score_to_category(self, score: float) -> str:
        """
        Convert pain score to category.
        
        Args:
            score: Pain score (0.0-1.0)
            
        Returns:
            Pain category string
        """
        if score < PainThresholds.NONE:
            return "No Pain"
        elif score < PainThresholds.MILD:
            return "Mild Pain"
        elif score < PainThresholds.MODERATE:
            return "Moderate Pain"
        else:
            return "Severe Pain"
    
    def _load_cnn_model(self, model_path: str):
        """
        Load pre-trained CNN model.
        
        Args:
            model_path: Path to model file
        """
        try:
            # Define model architecture (will be implemented in cnn_model.py)
            from src.models.cnn_model import PainCNN
            
            self.cnn_model = PainCNN()
            
            # Load weights
            checkpoint = torch.load(model_path, map_location='cpu')
            self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
            self.cnn_model.eval()
            
            print(f"CNN model loaded from {model_path}")
            
        except Exception as e:
            print(f"Failed to load CNN model: {e}")
            self.cnn_model = None
            self.use_cnn = False
    
    def get_session_summary(self) -> Dict[str, float]:
        """
        Get summary statistics for current session.
        
        Returns:
            Dictionary with session statistics
        """
        if not self.score_history:
            return {
                'avg_score': 0.0,
                'max_score': 0.0,
                'min_score': 0.0,
                'total_frames': 0,
                'pain_episodes': 0
            }
        
        scores = list(self.score_history)
        
        # Calculate statistics
        avg_score = np.mean(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        total_frames = len(scores)
        
        # Count pain episodes (consecutive frames above threshold)
        pain_episodes = 0
        in_pain_episode = False
        pain_threshold = PainThresholds.NONE
        
        for score in scores:
            if score > pain_threshold:
                if not in_pain_episode:
                    pain_episodes += 1
                    in_pain_episode = True
            else:
                in_pain_episode = False
        
        return {
            'avg_score': avg_score,
            'max_score': max_score,
            'min_score': min_score,
            'total_frames': total_frames,
            'pain_episodes': pain_episodes
        }
    
    def reset_session(self):
        """Reset session data for new analysis."""
        self.score_history.clear()
        self.smoothed_score = 0.0


class PainCNN(nn.Module):
    """
    Lightweight CNN model for pain detection.
    Based on MobileNetV2 architecture for efficient inference.
    """
    
    def __init__(self, num_classes: int = 1):
        """
        Initialize CNN model.
        
        Args:
            num_classes: Number of output classes (1 for regression)
        """
        super(PainCNN, self).__init__()
        
        # Use MobileNetV2 as backbone
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        
        # Replace classifier for pain regression
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.last_channel, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        return self.backbone(x)
