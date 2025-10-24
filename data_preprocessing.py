# Sample Data Preprocessing Script

This script demonstrates how to preprocess video data for training the CNN model.

```python
"""
Data preprocessing script for pain detection training.
Extracts face crops and creates training data from video files.
"""

import cv2
import os
import numpy as np
from pathlib import Path
import argparse
from src.detection.face_detector import FaceDetector


def preprocess_dataset(input_dir: str, output_dir: str, fps: int = 1):
    """
    Preprocess dataset for training.
    
    Args:
        input_dir: Directory containing raw videos/images
        output_dir: Directory to save processed data
        fps: Frames per second to extract
    """
    face_detector = FaceDetector()
    
    # Create output directories
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)
    
    # Process each video
    video_files = list(Path(input_dir).glob("*.avi")) + list(Path(input_dir).glob("*.mp4"))
    
    for video_path in video_files:
        print(f"Processing {video_path.name}...")
        process_video(video_path, face_detector, output_dir, fps)


def process_video(video_path: Path, face_detector: FaceDetector, output_dir: str, fps: int):
    """Process single video file."""
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)  # Extract every nth frame
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract face every frame_interval frames
        if frame_count % frame_interval == 0:
            landmarks = face_detector.detect_face(frame)
            if landmarks is not None:
                face_crop = face_detector.get_face_crop(frame, landmarks)
                if face_crop is not None:
                    # Save face crop
                    output_path = f"{output_dir}/images/{video_path.stem}_{extracted_count:06d}.jpg"
                    cv2.imwrite(output_path, face_crop)
                    
                    # Save label (placeholder - replace with actual pain scores)
                    label_path = f"{output_dir}/labels/{video_path.stem}_{extracted_count:06d}.txt"
                    with open(label_path, 'w') as f:
                        f.write("0.5")  # Placeholder - replace with actual pain score
                    
                    extracted_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {extracted_count} frames from {video_path.name}")


def create_sample_data():
    """Create sample data for testing."""
    print("Creating sample data structure...")
    
    # Create sample directories
    os.makedirs("data/sample_videos", exist_ok=True)
    
    # Create a sample video (placeholder)
    sample_video_path = "data/sample_videos/sample_pain_demo.mp4"
    
    if not os.path.exists(sample_video_path):
        print(f"Please add a sample video to {sample_video_path}")
        print("You can record a short video of facial expressions for testing.")
    
    print("Sample data structure created!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset for training")
    parser.add_argument("--input_dir", type=str, help="Input directory with videos")
    parser.add_argument("--output_dir", type=str, default="processed_data", help="Output directory")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second to extract")
    parser.add_argument("--create_sample", action="store_true", help="Create sample data structure")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_data()
    elif args.input_dir:
        preprocess_dataset(args.input_dir, args.output_dir, args.fps)
    else:
        print("Please specify --input_dir or use --create_sample")
```
