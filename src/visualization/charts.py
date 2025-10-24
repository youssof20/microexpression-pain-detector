"""
Chart generation utilities for pain detection visualization.
Creates interactive charts using Plotly for real-time and session analysis.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import time


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
        
        # Score distribution histogram
        if session_data.get('total_frames', 0) > 0:
            # Create histogram from actual session data
            scores = session_data.get('pain_scores', [])
            if scores:
                fig.add_trace(go.Histogram(
                    x=scores,
                    name="Score Distribution",
                    nbinsx=10
                ), row=2, col=1)
            else:
                fig.add_trace(go.Histogram(
                    x=[0],  # Empty histogram
                    name="Score Distribution"
                ), row=2, col=1)
        else:
            fig.add_trace(go.Histogram(
                x=[0],  # Empty histogram
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


class SessionTracker:
    """
    Tracks session data for pain detection analysis.
    Maintains history of scores, timestamps, and statistics.
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize session tracker.
        
        Args:
            max_history: Maximum number of data points to keep
        """
        self.max_history = max_history
        self.timestamps = deque(maxlen=max_history)
        self.pain_scores = deque(maxlen=max_history)
        self.categories = deque(maxlen=max_history)
        self.fas_scores = deque(maxlen=max_history)
        self.start_time = time.time()
        
    def add_data_point(self, pain_score: float, category: str, fas_scores: Dict[str, float]):
        """
        Add new data point to session.
        
        Args:
            pain_score: Current pain score
            category: Pain category
            fas_scores: FACS scores dictionary
        """
        current_time = time.time() - self.start_time
        
        self.timestamps.append(current_time)
        self.pain_scores.append(pain_score)
        self.categories.append(category)
        self.fas_scores.append(fas_scores.copy())
    
    def get_session_data(self) -> Dict[str, List]:
        """
        Get all session data.
        
        Returns:
            Dictionary containing all session data
        """
        return {
            'timestamps': list(self.timestamps),
            'pain_scores': list(self.pain_scores),
            'categories': list(self.categories),
            'fas_scores': list(self.fas_scores)
        }
    
    def get_session_summary(self) -> Dict[str, float]:
        """
        Get session summary statistics.
        
        Returns:
            Dictionary with session statistics
        """
        if not self.pain_scores:
            return {
                'avg_score': 0.0,
                'max_score': 0.0,
                'min_score': 0.0,
                'total_frames': 0,
                'pain_episodes': 0,
                'session_duration': 0.0
            }
        
        scores = list(self.pain_scores)
        categories = list(self.categories)
        
        # Calculate statistics
        avg_score = np.mean(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        total_frames = len(scores)
        session_duration = time.time() - self.start_time
        
        # Count pain episodes
        pain_episodes = 0
        in_pain_episode = False
        
        for category in categories:
            if category != "No Pain":
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
            'pain_episodes': pain_episodes,
            'session_duration': session_duration
        }
    
    def export_to_csv(self, filename: str) -> bool:
        """
        Export session data to CSV file.
        
        Args:
            filename: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = self.get_session_data()
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            return True
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False
    
    def reset_session(self):
        """Reset session data."""
        self.timestamps.clear()
        self.pain_scores.clear()
        self.categories.clear()
        self.fas_scores.clear()
        self.start_time = time.time()
