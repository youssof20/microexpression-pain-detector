"""
Main Streamlit application for Microexpression-Based Pain Detector.
Provides webcam interface, video upload, and real-time analysis.
"""

import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
from pathlib import Path
import os

# Import our modules
from src.detection.face_detector import FaceDetector
from src.detection.feature_extractor import FeatureExtractor
from src.detection.pain_classifier import PainClassifier
from src.visualization.annotator import PainAnnotator
from src.visualization.charts import PainChartGenerator, SessionTracker
from src.utils.video_handler import VideoProcessor
from src.utils.config import StreamlitConfig, Colors


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'face_detector' not in st.session_state:
        st.session_state.face_detector = FaceDetector()
    
    if 'feature_extractor' not in st.session_state:
        st.session_state.feature_extractor = FeatureExtractor()
    
    if 'pain_classifier' not in st.session_state:
        st.session_state.pain_classifier = PainClassifier()
    
    if 'annotator' not in st.session_state:
        st.session_state.annotator = PainAnnotator()
    
    if 'chart_generator' not in st.session_state:
        st.session_state.chart_generator = PainChartGenerator()
    
    if 'session_tracker' not in st.session_state:
        st.session_state.session_tracker = SessionTracker()
    
    if 'video_processor' not in st.session_state:
        st.session_state.video_processor = VideoProcessor(
            st.session_state.face_detector,
            st.session_state.feature_extractor,
            st.session_state.pain_classifier,
            st.session_state.annotator
        )
    
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False
    
    if 'baseline_set' not in st.session_state:
        st.session_state.baseline_set = False


def process_frame_for_display(frame):
    """Process frame and return annotated result."""
    try:
        annotated_frame, pain_score, category, detailed_scores = st.session_state.video_processor.process_frame(frame)
        
        # Add to session tracker
        st.session_state.session_tracker.add_data_point(pain_score, category, detailed_scores)
        
        return annotated_frame, pain_score, category, detailed_scores
    except Exception as e:
        st.error(f"Error processing frame: {e}")
        return frame, 0.0, "Error", {}


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title=StreamlitConfig.PAGE_TITLE,
        page_icon=StreamlitConfig.PAGE_ICON,
        layout=StreamlitConfig.LAYOUT
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("Microexpression-Based Pain Detector")
    st.markdown("**Demo Application - Not for Clinical Use**")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Sensitivity slider
    sensitivity = st.sidebar.slider("Detection Sensitivity", 0.1, 1.0, 0.5, 0.1)
    
    # Visualization options
    show_landmarks = st.sidebar.checkbox("Show Landmarks", value=True)
    show_regions = st.sidebar.checkbox("Highlight Pain Regions", value=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Live Detection", "Video Upload", "Session History", "About"])
    
    with tab1:
        st.header("Live Webcam Detection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Webcam controls
            if not st.session_state.webcam_active:
                if st.button("Start Webcam", type="primary"):
                    st.session_state.webcam_active = True
                    st.success("Webcam interface activated!")
            else:
                if st.button("Stop Webcam"):
                    st.session_state.webcam_active = False
                    st.success("Webcam interface deactivated!")
            
            # Baseline setting
            if st.session_state.webcam_active and not st.session_state.baseline_set:
                st.info("📸 Take a photo with a neutral expression to set the baseline")
                baseline_camera = st.camera_input("Set Baseline - Neutral Expression")
                
                if baseline_camera is not None:
                    # Convert to OpenCV format
                    bytes_data = baseline_camera.getvalue()
                    cv_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    
                    # Detect face and set baseline
                    landmarks = st.session_state.face_detector.detect_face(cv_image)
                    if landmarks is not None:
                        try:
                            st.session_state.face_detector.set_baseline(landmarks)
                            st.session_state.feature_extractor.set_baseline(landmarks)
                            st.session_state.baseline_set = True
                            st.success("✅ Baseline set successfully!")
                        except Exception as e:
                            st.error(f"Error setting baseline: {str(e)}")
                    else:
                        st.error("No face detected. Please try again with better lighting and make sure your face is clearly visible.")
        
        with col2:
            # Current status
            if st.session_state.webcam_active:
                st.metric("Status", "Active", delta="Live")
            else:
                st.metric("Status", "Inactive", delta="Stopped")
        
        # Webcam feed using Streamlit's camera input
        if st.session_state.webcam_active and st.session_state.baseline_set:
            st.info("📹 Webcam is active! Take a photo to analyze your facial expression.")
            
            # Use Streamlit's camera input
            camera_input = st.camera_input("Take a photo for pain detection analysis")
            
            if camera_input is not None:
                # Convert uploaded image to OpenCV format
                bytes_data = camera_input.getvalue()
                cv_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                
                # Process the captured frame
                try:
                    annotated_frame, pain_score, category, detailed_scores = st.session_state.video_processor.process_frame(cv_image)
                except Exception as e:
                    st.error(f"Error processing frame: {str(e)}")
                    return
                
                # Convert BGR to RGB for display
                display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Display the processed frame
                st.image(display_frame, channels="RGB", use_column_width=True, caption="Pain Detection Analysis")
                
                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pain Score", f"{pain_score:.2f}")
                with col2:
                    st.metric("Category", category)
                with col3:
                    st.metric("Analysis", "Complete")
                
                # Display detailed FACS scores
                if detailed_scores:
                    st.subheader("FACS Action Unit Breakdown")
                    fas_data = []
                    for au_name, score in detailed_scores.items():
                        if au_name in ['au4', 'au6', 'au7', 'au9', 'au10', 'eye_tightening']:
                            fas_data.append({"Action Unit": au_name.upper(), "Intensity": f"{score:.2f}"})
                    
                    if fas_data:
                        st.table(fas_data)
                
                # Add to session tracker
                st.session_state.session_tracker.add_data_point(pain_score, category, detailed_scores)
    
    with tab2:
        st.header("Video File Analysis")
        
        # File upload
        uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
        
        if uploaded_file is not None:
            # Save uploaded file
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process video
            if st.button("Analyze Video"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process video file
                output_path = f"analyzed_{uploaded_file.name}"
                
                def progress_callback(progress):
                    progress_bar.progress(progress)
                    status_text.text(f"Processing... {progress*100:.1f}%")
                
                success = st.session_state.video_processor.process_video_file(
                    temp_path, output_path
                )
                
                if success:
                    st.success("Video analysis completed!")
                    
                    # Display results
                    st.video(output_path)
                    
                    # Clean up temp files
                    os.remove(temp_path)
                    if os.path.exists(output_path):
                        os.remove(output_path)
                else:
                    st.error("Failed to process video")
    
    with tab3:
        st.header("Session History & Analytics")
        
        # Session summary
        session_summary = st.session_state.session_tracker.get_session_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Frames", session_summary['total_frames'])
        with col2:
            st.metric("Avg Pain Score", f"{session_summary['avg_score']:.2f}")
        with col3:
            st.metric("Max Pain Score", f"{session_summary['max_score']:.2f}")
        with col4:
            st.metric("Pain Episodes", session_summary['pain_episodes'])
        
        # Charts
        session_data = st.session_state.session_tracker.get_session_data()
        
        if session_data['timestamps']:
            # Real-time chart
            fig = st.session_state.chart_generator.create_realtime_chart(
                session_data['timestamps'],
                session_data['pain_scores']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # FACS breakdown
            if session_data['fas_scores']:
                # Average FACS scores
                avg_fas_scores = {}
                for fas_dict in session_data['fas_scores']:
                    for key, value in fas_dict.items():
                        if key not in avg_fas_scores:
                            avg_fas_scores[key] = []
                        avg_fas_scores[key].append(value)
                
                # Calculate averages
                for key in avg_fas_scores:
                    avg_fas_scores[key] = np.mean(avg_fas_scores[key])
                
                # Create FACS chart
                fas_fig = st.session_state.chart_generator.create_facs_breakdown(avg_fas_scores)
                st.plotly_chart(fas_fig, use_container_width=True)
        
        # Export data
        if st.button("Export Session Data"):
            filename = f"pain_session_{int(time.time())}.csv"
            if st.session_state.session_tracker.export_to_csv(filename):
                st.success(f"Data exported to {filename}")
                
                # Provide download link
                with open(filename, 'rb') as f:
                    st.download_button(
                        label="Download CSV",
                        data=f.read(),
                        file_name=filename,
                        mime="text/csv"
                    )
                
                # Clean up
                os.remove(filename)
            else:
                st.error("Failed to export data")
    
    with tab4:
        st.header("About This Application")
        
        st.markdown("""
        ## Microexpression-Based Pain Detector
        
        This is a **demonstration application** for detecting subtle facial microexpressions 
        indicative of pain or discomfort using computer vision and machine learning techniques.
        
        ### Methodology
        
        The system uses the **Facial Action Coding System (FACS)** to detect pain-related 
        Action Units (AUs):
        
        - **AU4 (Brow Lowerer)**: Inner brow lowering
        - **AU6/AU7 (Orbital Tightening)**: Eye aperture reduction
        - **AU9 (Nose Wrinkler)**: Nose bridge wrinkling
        - **AU10 (Upper Lip Raiser)**: Upper lip raising
        - **AU43 (Eye Closure)**: Eye closing/squinting
        
        ### Technical Implementation
        
        - **Face Detection**: MediaPipe for robust facial landmark extraction
        - **Feature Extraction**: FACS-based geometric analysis of facial landmarks
        - **Classification**: Rule-based scoring with optional CNN enhancement
        - **Visualization**: Real-time annotation and interactive charts
        
        ### Important Disclaimers
        
        ⚠️ **This application is for demonstration purposes only and is NOT intended for clinical use.**
        
        - Results should not be used for medical diagnosis or treatment decisions
        - The system has not been validated for clinical accuracy
        - Pain detection is complex and requires professional medical assessment
        - This tool is designed for educational and research purposes only
        
        ### Limitations
        
        - Performance may vary based on lighting conditions and camera quality
        - Individual differences in facial expressions may affect accuracy
        - The system requires clear facial visibility and good lighting
        - Results are not validated against clinical pain assessment tools
        
        ### Future Improvements
        
        - Integration with clinical pain assessment datasets
        - Improved CNN models trained on larger datasets
        - Multi-modal analysis combining facial expressions with other signals
        - Real-time performance optimization for mobile devices
        
        ### Contact
        
        For questions about this project or to report issues, please contact the development team.
        """)
        
        # Technical details
        with st.expander("Technical Details"):
            st.code("""
            # Key Dependencies
            - streamlit>=1.28.0
            - opencv-python>=4.8.0
            - mediapipe>=0.10.0
            - numpy>=1.24.0
            - plotly>=5.17.0
            - torch>=2.0.0 (optional)
            """, language="python")
            
            st.markdown("""
            **Performance**: The system is optimized for real-time processing on standard laptop CPUs.
            Target performance: >15 FPS with webcam input.
            
            **Model Size**: Rule-based approach requires minimal computational resources.
            Optional CNN model: ~3M parameters for efficient inference.
            """)


if __name__ == "__main__":
    main()
