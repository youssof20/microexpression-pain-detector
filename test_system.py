"""
Simple test script to verify the pain detection system works.
"""

import cv2
import numpy as np
from src.detection.face_detector import FaceDetector
from src.detection.feature_extractor import FeatureExtractor
from src.detection.pain_classifier import PainClassifier

def test_system():
    """Test the pain detection system with a simple image."""
    print("Testing Pain Detection System...")
    
    # Initialize components
    face_detector = FaceDetector()
    feature_extractor = FeatureExtractor()
    pain_classifier = PainClassifier()
    
    # Create a test image (black image with a white rectangle representing a face)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw a simple face-like rectangle
    cv2.rectangle(test_image, (200, 150), (440, 350), (255, 255, 255), -1)
    
    print("Test image created")
    
    # Test face detection
    landmarks = face_detector.detect_face(test_image)
    
    if landmarks is not None:
        print(f"Face detected! Landmarks shape: {landmarks.shape}")
        
        # Test feature extraction
        features = feature_extractor.extract_all_features(landmarks)
        print(f"Features extracted: {list(features.keys())}")
        
        # Test pain classification
        pain_score, category, detailed_scores = pain_classifier.classify_pain(features)
        print(f"Pain Score: {pain_score:.2f}")
        print(f"Category: {category}")
        print(f"Detailed Scores: {detailed_scores}")
        
        print("System test completed successfully!")
        
    else:
        print("No face detected in test image")
    
    # Cleanup
    face_detector.cleanup()

if __name__ == "__main__":
    test_system()
