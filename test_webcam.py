"""
Test script to verify webcam functionality works.
"""

import cv2
import numpy as np
from src.detection.face_detector import FaceDetector

def test_webcam():
    """Test basic webcam functionality."""
    print("Testing webcam functionality...")
    
    # Initialize face detector
    face_detector = FaceDetector()
    
    # Try to open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return False
    
    print("✅ Webcam opened successfully!")
    
    # Try to read a frame
    ret, frame = cap.read()
    if ret:
        print("✅ Successfully read frame from webcam")
        print(f"Frame shape: {frame.shape}")
        
        # Test face detection
        landmarks = face_detector.detect_face(frame)
        if landmarks is not None:
            print("✅ Face detection working!")
            print(f"Detected landmarks shape: {landmarks.shape}")
        else:
            print("⚠️ No face detected (this is normal if no face is visible)")
        
    else:
        print("❌ Error: Could not read frame from webcam")
        cap.release()
        return False
    
    cap.release()
    face_detector.cleanup()
    print("✅ Webcam test completed successfully!")
    return True

if __name__ == "__main__":
    test_webcam()
