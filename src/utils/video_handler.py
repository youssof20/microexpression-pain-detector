"""
Video handling utilities for webcam and file processing.
Manages video capture, frame processing, and video file analysis.
"""

import cv2
import numpy as np
from typing import Optional, Generator, Tuple, Dict
import os
from pathlib import Path


class VideoHandler:
    """
    Handles video input from webcam or video files.
    Provides frame-by-frame processing capabilities.
    """
    
    def __init__(self):
        """Initialize video handler."""
        self.cap = None
        self.is_webcam = False
        self.current_frame = None
        
    def initialize_webcam(self, camera_index: int = 0) -> bool:
        """
        Initialize webcam capture.
        
        Args:
            camera_index: Camera index (0 for default)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(camera_index)
            
            # Set webcam properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            if self.cap.isOpened():
                self.is_webcam = True
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Error initializing webcam: {e}")
            return False
    
    def initialize_video_file(self, video_path: str) -> bool:
        """
        Initialize video file capture.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}")
                return False
            
            self.cap = cv2.VideoCapture(video_path)
            
            if self.cap.isOpened():
                self.is_webcam = False
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Error initializing video file: {e}")
            return False
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read next frame from video source.
        
        Returns:
            Frame as numpy array or None if no frame available
        """
        if self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        
        if ret:
            self.current_frame = frame
            return frame
        else:
            return None
    
    def get_frame_properties(self) -> Tuple[int, int, float]:
        """
        Get video properties.
        
        Returns:
            Tuple of (width, height, fps)
        """
        if self.cap is None:
            return 0, 0, 0.0
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        return width, height, fps
    
    def is_opened(self) -> bool:
        """
        Check if video source is opened.
        
        Returns:
            True if opened, False otherwise
        """
        return self.cap is not None and self.cap.isOpened()
    
    def release(self):
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_webcam = False
        self.current_frame = None
    
    def process_video_file(self, video_path: str, 
                          frame_callback, 
                          progress_callback=None) -> bool:
        """
        Process entire video file with callback function.
        
        Args:
            video_path: Path to video file
            frame_callback: Function to call for each frame
            progress_callback: Optional progress callback function
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialize_video_file(video_path):
            return False
        
        try:
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = 0
            
            while True:
                frame = self.read_frame()
                if frame is None:
                    break
                
                # Process frame
                frame_callback(frame, current_frame)
                
                current_frame += 1
                
                # Update progress
                if progress_callback and total_frames > 0:
                    progress = current_frame / total_frames
                    progress_callback(progress)
            
            return True
            
        except Exception as e:
            print(f"Error processing video: {e}")
            return False
        
        finally:
            self.release()
    
    def save_frame(self, frame: np.ndarray, filename: str) -> bool:
        """
        Save frame as image file.
        
        Args:
            frame: Frame to save
            filename: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cv2.imwrite(filename, frame)
            return True
        except Exception as e:
            print(f"Error saving frame: {e}")
            return False
    
    def create_video_writer(self, output_path: str, fps: float = 30.0) -> Optional[cv2.VideoWriter]:
        """
        Create video writer for saving processed video.
        
        Args:
            output_path: Output video path
            fps: Frames per second
            
        Returns:
            VideoWriter object or None if failed
        """
        try:
            if self.current_frame is None:
                return None
            
            height, width = self.current_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            return writer
            
        except Exception as e:
            print(f"Error creating video writer: {e}")
            return None


class VideoProcessor:
    """
    High-level video processing class for pain detection.
    Combines video handling with pain detection pipeline.
    """
    
    def __init__(self, face_detector, feature_extractor, pain_classifier, annotator):
        """
        Initialize video processor.
        
        Args:
            face_detector: Face detection instance
            feature_extractor: Feature extraction instance
            pain_classifier: Pain classification instance
            annotator: Visualization annotator instance
        """
        self.face_detector = face_detector
        self.feature_extractor = feature_extractor
        self.pain_classifier = pain_classifier
        self.annotator = annotator
        self.video_handler = VideoHandler()
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float, str, Dict]:
        """
        Process single frame for pain detection.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (annotated_frame, pain_score, category, detailed_scores)
        """
        # Detect face and landmarks
        landmarks = self.face_detector.detect_face(frame)
        
        if landmarks is None:
            # No face detected
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "No Face Detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return annotated_frame, 0.0, "No Face", {}
        
        # Extract features
        features = self.feature_extractor.extract_all_features(landmarks)
        
        # Get face crop for CNN (optional)
        face_crop = self.face_detector.get_face_crop(frame, landmarks)
        
        # Classify pain
        pain_score, category, detailed_scores = self.pain_classifier.classify_pain(
            features, face_crop
        )
        
        # Annotate frame
        annotated_frame = self.annotator.annotate_frame(
            frame, landmarks, pain_score, category, detailed_scores
        )
        
        return annotated_frame, pain_score, category, detailed_scores
    
    def start_webcam_processing(self, camera_index: int = 0) -> bool:
        """
        Start webcam processing.
        
        Args:
            camera_index: Camera index
            
        Returns:
            True if successful, False otherwise
        """
        return self.video_handler.initialize_webcam(camera_index)
    
    def process_webcam_frame(self) -> Optional[Tuple[np.ndarray, float, str, Dict]]:
        """
        Process next webcam frame.
        
        Returns:
            Tuple of (annotated_frame, pain_score, category, detailed_scores) or None
        """
        frame = self.video_handler.read_frame()
        if frame is None:
            return None
        
        return self.process_frame(frame)
    
    def stop_webcam(self):
        """Stop webcam processing."""
        self.video_handler.release()
    
    def process_video_file(self, video_path: str, output_path: Optional[str] = None) -> bool:
        """
        Process entire video file.
        
        Args:
            video_path: Input video path
            output_path: Optional output video path
            
        Returns:
            True if successful, False otherwise
        """
        def frame_callback(frame, frame_number):
            annotated_frame, pain_score, category, detailed_scores = self.process_frame(frame)
            
            if output_path and frame_number == 0:
                # Initialize video writer on first frame
                self.output_writer = self.video_handler.create_video_writer(output_path)
            
            if hasattr(self, 'output_writer') and self.output_writer is not None:
                self.output_writer.write(annotated_frame)
        
        success = self.video_handler.process_video_file(video_path, frame_callback)
        
        # Clean up video writer
        if hasattr(self, 'output_writer') and self.output_writer is not None:
            self.output_writer.release()
            delattr(self, 'output_writer')
        
        return success
