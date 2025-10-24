"""
Simple demo script for testing the pain detection system.
This script provides a basic command-line interface for testing.
"""

import cv2
import numpy as np
import time
from src.detection.face_detector import FaceDetector
from src.detection.feature_extractor import FeatureExtractor
from src.detection.pain_classifier import PainClassifier
from src.visualization.annotator import PainAnnotator


def demo_webcam():
    """Run a simple webcam demo."""
    print("Starting Pain Detection Demo...")
    print("Press 'q' to quit, 'b' to set baseline")
    
    # Initialize components
    face_detector = FaceDetector()
    feature_extractor = FeatureExtractor()
    pain_classifier = PainClassifier()
    annotator = PainAnnotator()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam initialized successfully!")
    print("Make sure your face is visible in the camera")
    
    baseline_set = False
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Detect face and landmarks
        landmarks = face_detector.detect_face(frame)
        
        if landmarks is not None:
            # Set baseline on first detection
            if not baseline_set:
                face_detector.set_baseline(landmarks)
                feature_extractor.set_baseline(landmarks)
                baseline_set = True
                print("Baseline set! Now try different facial expressions.")
            
            # Extract features
            features = feature_extractor.extract_all_features(landmarks)
            
            # Get face crop for CNN (optional)
            face_crop = face_detector.get_face_crop(frame, landmarks)
            
            # Classify pain
            pain_score, category, detailed_scores = pain_classifier.classify_pain(
                features, face_crop
            )
            
            # Annotate frame
            annotated_frame = annotator.annotate_frame(
                frame, landmarks, pain_score, category, detailed_scores
            )
            
            # Display results in console every 30 frames
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: Pain Score = {pain_score:.2f}, Category = {category}")
            
        else:
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "No Face Detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: No face detected")
        
        # Display frame
        cv2.imshow("Pain Detection Demo", annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            if landmarks is not None:
                face_detector.set_baseline(landmarks)
                feature_extractor.set_baseline(landmarks)
                baseline_set = True
                print("Baseline reset!")
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    face_detector.cleanup()
    print("Demo completed!")


def demo_image(image_path):
    """Test pain detection on a single image."""
    print(f"Testing pain detection on image: {image_path}")
    
    # Initialize components
    face_detector = FaceDetector()
    feature_extractor = FeatureExtractor()
    pain_classifier = PainClassifier()
    annotator = PainAnnotator()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Detect face and landmarks
    landmarks = face_detector.detect_face(image)
    
    if landmarks is not None:
        # Extract features
        features = feature_extractor.extract_all_features(landmarks)
        
        # Get face crop for CNN (optional)
        face_crop = face_detector.get_face_crop(image, landmarks)
        
        # Classify pain
        pain_score, category, detailed_scores = pain_classifier.classify_pain(
            features, face_crop
        )
        
        # Annotate frame
        annotated_image = annotator.annotate_frame(
            image, landmarks, pain_score, category, detailed_scores
        )
        
        # Display results
        print(f"Pain Score: {pain_score:.2f}")
        print(f"Category: {category}")
        print(f"Detailed Scores: {detailed_scores}")
        
        # Save annotated image
        output_path = f"annotated_{image_path}"
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved as {output_path}")
        
        # Display image
        cv2.imshow("Pain Detection Result", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        print("No face detected in image")
    
    face_detector.cleanup()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test with image
        demo_image(sys.argv[1])
    else:
        # Run webcam demo
        demo_webcam()
