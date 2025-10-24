"""
Visualization utilities for pain detection.
Handles drawing landmarks, pain indicators, and real-time charts.
"""

import cv2
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional, Tuple
from src.utils.config import Colors, LandmarkIndices


class PainAnnotator:
    """
    Annotates video frames with pain detection results.
    Draws landmarks, pain indicators, and text overlays.
    """
    
    def __init__(self):
        """Initialize annotator with default settings."""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_thickness = 2
        
    def annotate_frame(self, frame: np.ndarray, landmarks: np.ndarray, 
                      pain_score: float, category: str, 
                      detailed_scores: Dict[str, float],
                      show_landmarks: bool = True,
                      show_regions: bool = True) -> np.ndarray:
        """
        Annotate frame with pain detection results.
        
        Args:
            frame: Input video frame
            landmarks: Face landmarks
            pain_score: Current pain score
            category: Pain category
            detailed_scores: Detailed scoring breakdown
            show_landmarks: Whether to show facial landmarks
            show_regions: Whether to highlight pain regions
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw landmarks if requested
        if show_landmarks:
            annotated_frame = self._draw_landmarks(annotated_frame, landmarks)
        
        # Highlight pain regions if requested
        if show_regions:
            annotated_frame = self._highlight_pain_regions(annotated_frame, landmarks, pain_score)
        
        # Draw pain score and category
        annotated_frame = self._draw_pain_info(annotated_frame, pain_score, category)
        
        # Draw detailed scores
        annotated_frame = self._draw_detailed_scores(annotated_frame, detailed_scores)
        
        return annotated_frame
    
    def _draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Draw facial landmarks on frame.
        
        Args:
            frame: Input frame
            landmarks: Face landmarks
            
        Returns:
            Frame with landmarks drawn
        """
        # Draw key landmarks for pain detection
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
                cv2.circle(frame, (x, y), 3, Colors.BLUE, -1)
        
        return frame
    
    def _highlight_pain_regions(self, frame: np.ndarray, landmarks: np.ndarray, 
                              pain_score: float) -> np.ndarray:
        """
        Highlight regions showing pain indicators.
        
        Args:
            frame: Input frame
            landmarks: Face landmarks
            pain_score: Current pain score
            
        Returns:
            Frame with highlighted regions
        """
        # Determine color based on pain level
        if pain_score < 0.3:
            color = Colors.GREEN
        elif pain_score < 0.6:
            color = Colors.YELLOW
        else:
            color = Colors.RED
        
        # Highlight eyebrow region
        try:
            left_brow = landmarks[LandmarkIndices.LEFT_INNER_BROW]
            right_brow = landmarks[LandmarkIndices.RIGHT_INNER_BROW]
            
            brow_center_x = int((left_brow[0] + right_brow[0]) / 2)
            brow_center_y = int((left_brow[1] + right_brow[1]) / 2)
            
            cv2.circle(frame, (brow_center_x, brow_center_y), 20, color, 2)
            
        except (IndexError, ValueError):
            pass
        
        # Highlight eye region
        try:
            left_eye_top = landmarks[LandmarkIndices.LEFT_EYE_TOP]
            left_eye_bottom = landmarks[LandmarkIndices.LEFT_EYE_BOTTOM]
            right_eye_top = landmarks[LandmarkIndices.RIGHT_EYE_TOP]
            right_eye_bottom = landmarks[LandmarkIndices.RIGHT_EYE_BOTTOM]
            
            # Left eye
            left_eye_center_x = int(left_eye_top[0])
            left_eye_center_y = int((left_eye_top[1] + left_eye_bottom[1]) / 2)
            cv2.circle(frame, (left_eye_center_x, left_eye_center_y), 15, color, 2)
            
            # Right eye
            right_eye_center_x = int(right_eye_top[0])
            right_eye_center_y = int((right_eye_top[1] + right_eye_bottom[1]) / 2)
            cv2.circle(frame, (right_eye_center_x, right_eye_center_y), 15, color, 2)
            
        except (IndexError, ValueError):
            pass
        
        return frame
    
    def _draw_pain_info(self, frame: np.ndarray, pain_score: float, category: str) -> np.ndarray:
        """
        Draw pain score and category on frame.
        
        Args:
            frame: Input frame
            pain_score: Pain score
            category: Pain category
            
        Returns:
            Frame with pain info drawn
        """
        # Determine text color based on pain level
        if pain_score < 0.3:
            text_color = Colors.GREEN
        elif pain_score < 0.6:
            text_color = Colors.YELLOW
        else:
            text_color = Colors.RED
        
        # Draw pain score
        score_text = f"Pain Score: {pain_score:.2f}"
        cv2.putText(frame, score_text, (10, 30), self.font, self.font_scale, text_color, self.font_thickness)
        
        # Draw category
        category_text = f"Category: {category}"
        cv2.putText(frame, category_text, (10, 60), self.font, self.font_scale, text_color, self.font_thickness)
        
        return frame
    
    def _draw_detailed_scores(self, frame: np.ndarray, detailed_scores: Dict[str, float]) -> np.ndarray:
        """
        Draw detailed FACS scores on frame.
        
        Args:
            frame: Input frame
            detailed_scores: Detailed scoring breakdown
            
        Returns:
            Frame with detailed scores drawn
        """
        y_offset = 90
        
        # Draw key FACS scores
        fas_scores = ['au4', 'au6', 'au7', 'au9', 'au10', 'eye_tightening']
        
        for score_name in fas_scores:
            if score_name in detailed_scores:
                score_value = detailed_scores[score_name]
                text = f"{score_name.upper()}: {score_value:.2f}"
                cv2.putText(frame, text, (10, y_offset), self.font, 0.5, Colors.WHITE, 1)
                y_offset += 20
        
        return frame


class PainChartGenerator:
    """
    Generates real-time charts for pain detection visualization.
    Creates line charts, histograms, and session summaries.
    """
    
    def __init__(self):
        """Initialize chart generator."""
        self.time_window = 60  # seconds
        self.max_points = 300  # maximum points to display
        
    def create_realtime_chart(self, timestamps: List[float], scores: List[float]) -> go.Figure:
        """
        Create real-time pain score chart.
        
        Args:
            timestamps: List of timestamps
            scores: List of pain scores
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add pain score line
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=scores,
            mode='lines',
            name='Pain Score',
            line=dict(color='red', width=2),
            hovertemplate='Time: %{x}<br>Pain Score: %{y:.2f}<extra></extra>'
        ))
        
        # Add threshold lines
        fig.add_hline(y=0.3, line_dash="dash", line_color="yellow", 
                     annotation_text="Mild Pain Threshold")
        fig.add_hline(y=0.6, line_dash="dash", line_color="orange", 
                     annotation_text="Moderate Pain Threshold")
        
        # Update layout
        fig.update_layout(
            title="Real-time Pain Detection",
            xaxis_title="Time (seconds)",
            yaxis_title="Pain Score",
            yaxis=dict(range=[0, 1]),
            height=400,
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
    
    def create_session_summary(self, session_data: Dict[str, float]) -> go.Figure:
        """
        Create session summary chart.
        
        Args:
            session_data: Session statistics
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Pain Score', 'Pain Episodes', 'Score Distribution', 'Session Stats'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "histogram"}, {"type": "table"}]]
        )
        
        # Average pain score gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=session_data.get('avg_score', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Average Pain Score"},
            gauge={'axis': {'range': [None, 1]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 0.3], 'color': "lightgray"},
                            {'range': [0.3, 0.6], 'color': "yellow"},
                            {'range': [0.6, 1], 'color': "red"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 0.6}}
        ), row=1, col=1)
        
        # Pain episodes counter
        fig.add_trace(go.Indicator(
            mode="number",
            value=session_data.get('pain_episodes', 0),
            title={'text': "Pain Episodes"},
            number={'font': {'size': 50}}
        ), row=1, col=2)
        
        # Score distribution histogram (placeholder)
        fig.add_trace(go.Histogram(
            x=[0.1, 0.2, 0.3, 0.4, 0.5],  # Placeholder data
            name="Score Distribution"
        ), row=2, col=1)
        
        # Session stats table
        stats_data = [
            ['Metric', 'Value'],
            ['Total Frames', str(session_data.get('total_frames', 0))],
            ['Max Score', f"{session_data.get('max_score', 0):.2f}"],
            ['Min Score', f"{session_data.get('min_score', 0):.2f}"],
            ['Pain Episodes', str(session_data.get('pain_episodes', 0))]
        ]
        
        fig.add_trace(go.Table(
            header=dict(values=stats_data[0], fill_color='paleturquoise'),
            cells=dict(values=list(zip(*stats_data[1:])), fill_color='lavender')
        ), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False, title_text="Session Summary")
        
        return fig
    
    def create_facs_breakdown(self, fas_scores: Dict[str, float]) -> go.Figure:
        """
        Create FACS Action Unit breakdown chart.
        
        Args:
            fas_scores: FACS scores dictionary
            
        Returns:
            Plotly figure
        """
        # Prepare data
        au_names = list(fas_scores.keys())
        au_values = list(fas_scores.values())
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(x=au_names, y=au_values, marker_color='lightblue')
        ])
        
        fig.update_layout(
            title="FACS Action Unit Breakdown",
            xaxis_title="Action Units",
            yaxis_title="Intensity",
            height=400,
            template="plotly_white"
        )
        
        return fig
    
    def create_comparison_chart(self, rule_scores: List[float], cnn_scores: List[float], 
                              timestamps: List[float]) -> go.Figure:
        """
        Create comparison chart between rule-based and CNN scores.
        
        Args:
            rule_scores: Rule-based scores
            cnn_scores: CNN scores
            timestamps: Timestamps
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add rule-based scores
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=rule_scores,
            mode='lines',
            name='Rule-based',
            line=dict(color='blue', width=2)
        ))
        
        # Add CNN scores
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=cnn_scores,
            mode='lines',
            name='CNN',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title="Rule-based vs CNN Pain Detection",
            xaxis_title="Time (seconds)",
            yaxis_title="Pain Score",
            yaxis=dict(range=[0, 1]),
            height=400,
            template="plotly_white"
        )
        
        return fig
