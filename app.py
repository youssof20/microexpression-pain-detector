"""
Microexpression Pain Detector — Streamlit app with live webcam via streamlit-webrtc.
Uses MediaPipe Face Mesh and FACS-based rule scoring. Set neutral baseline on button press.
"""

import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer

from src.detection.face_detector import FaceDetector
from src.detection.feature_extractor import FeatureExtractor
from src.detection.pain_classifier import PainClassifier
from src.pipeline import process_frame, set_collecting_baseline, get_state
from src.utils.config import StreamlitConfig, PainThresholds

# Module-level refs for webrtc callback (callback may run in another thread)
_detectors = {"fd": None, "fe": None, "pc": None}


def init_session_state():
    if "face_detector" not in st.session_state:
        st.session_state.face_detector = FaceDetector()
    if "feature_extractor" not in st.session_state:
        st.session_state.feature_extractor = FeatureExtractor()
    if "pain_classifier" not in st.session_state:
        st.session_state.pain_classifier = PainClassifier()
    _detectors["fd"] = st.session_state.face_detector
    _detectors["fe"] = st.session_state.feature_extractor
    _detectors["pc"] = st.session_state.pain_classifier


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    fd, fe, pc = _detectors["fd"], _detectors["fe"], _detectors["pc"]
    if fd is None or fe is None or pc is None:
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    out, score, category, detailed = process_frame(img, fd, fe, pc)
    return av.VideoFrame.from_ndarray(out, format="bgr24")


def main():
    st.set_page_config(
        page_title=StreamlitConfig.PAGE_TITLE,
        page_icon=StreamlitConfig.PAGE_ICON,
        layout=StreamlitConfig.LAYOUT,
    )
    init_session_state()

    st.title("Microexpression Pain Detector")
    st.caption("FACS-based pain detection from webcam. Not for clinical use.")

    # Baseline button: start collecting 30 frames for neutral baseline
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Set baseline (neutral expression)", type="primary"):
            set_collecting_baseline(True)
            st.info("Hold a neutral expression for ~1 second. Baseline will be set automatically after 30 frames.")
    with col2:
        state = get_state()
        if state.get("baseline_set"):
            st.success("Baseline set. Pain score is relative to your neutral expression.")
        elif state.get("collecting_baseline"):
            n = len(state.get("baseline_buffer", []))
            st.warning(f"Collecting baseline... {n}/30 frames.")

    # Live webcam with streamlit-webrtc
    ctx = webrtc_streamer(
        key="pain-cam",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

    # Gauge: 0–30 neutral, 30–60 mild discomfort, 60–100 high pain
    state = get_state()
    score = state.get("current_score", 0.0)
    category = state.get("current_category", "Neutral")

    st.subheader("Pain score (0–100)")
    # Progress bar with three zones
    st.progress(min(1.0, score / 100.0))
    if score < PainThresholds.NEUTRAL:
        zone = "Neutral (0–30)"
        color = "green"
    elif score < PainThresholds.MILD:
        zone = "Mild discomfort (30–60)"
        color = "orange"
    else:
        zone = "High pain indicators (60–100)"
        color = "red"
    st.markdown(f"**{score:.0f}** — {zone} — *{category}*")

    with st.expander("FACS weights"):
        st.markdown("""
        - **AU4** Brow lowerer (0.35)
        - **AU9** Nose wrinkler (0.20)
        - **AU6** Cheek raiser (0.20)
        - **AU46** Eye tightener (0.15)
        - **AU20** Lip corner stretch (0.10)
        """)


if __name__ == "__main__":
    main()
