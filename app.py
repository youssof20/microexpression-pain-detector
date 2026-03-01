"""
Microexpression Pain Detector — Streamlit app with live webcam or sample video.
Uses MediaPipe Face Mesh and FACS-based rule scoring. Set neutral baseline on button press.
"""

import os
import tempfile
import streamlit as st
import av
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer

from src.detection.face_detector import FaceDetector
from src.detection.feature_extractor import FeatureExtractor
from src.detection.pain_classifier import PainClassifier
from src.pipeline import process_frame, set_collecting_baseline, set_baseline_set, get_state
from src.utils.config import StreamlitConfig, PainThresholds

# Sample video path (relative to project root)
SAMPLE_VIDEO_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "stock-footage-a-man-is-holding-his-head-in-pain-showing-signs-of-a-severe-headache-or-migraine.webm",
)

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


def set_baseline_from_video(video_path: str, fd, fe, pc, num_frames: int = 30) -> bool:
    """Collect baseline from first num_frames (with face) of a video file. Returns True if baseline set."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    buffer = []
    try:
        while len(buffer) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            landmarks = fd.detect_face(frame)
            if landmarks is None:
                continue
            distances = fe.extract_distances(landmarks)
            buffer.append(distances)
        cap.release()
    except Exception:
        cap.release()
        return False
    if len(buffer) < num_frames:
        return False
    keys = ["au4", "au9", "au46", "au20", "au6"]
    baseline = {k: float(np.median([d[k] for d in buffer])) for k in keys}
    fe.set_baseline(baseline)
    pc.set_baseline(baseline)
    set_baseline_set(True)
    return True


def process_video_file(video_path: str, fd, fe, pc, progress_callback=None) -> str | None:
    """Process video with pain overlay. Returns path to temp output video or None."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    out_path = os.path.join(tempfile.gettempdir(), "pain_detection_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    n = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out_frame, _, _, _ = process_frame(frame, fd, fe, pc)
            writer.write(out_frame)
            n += 1
            if progress_callback and total > 0:
                progress_callback(n / total)
        writer.release()
        cap.release()
    except Exception:
        writer.release()
        cap.release()
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        return None
    return out_path if n > 0 else None


def main():
    st.set_page_config(
        page_title=StreamlitConfig.PAGE_TITLE,
        page_icon=StreamlitConfig.PAGE_ICON,
        layout=StreamlitConfig.LAYOUT,
    )
    init_session_state()

    st.title("Microexpression Pain Detector")
    st.caption("FACS-based pain detection from webcam or sample video. Not for clinical use.")

    # Input source: Webcam or Sample video
    input_source = st.radio(
        "Input source",
        options=["Webcam", "Sample video"],
        index=0,
        horizontal=True,
    )

    fd = st.session_state.face_detector
    fe = st.session_state.feature_extractor
    pc = st.session_state.pain_classifier

    # Baseline: same for both sources
    col1, col2 = st.columns([1, 2])
    with col1:
        if input_source == "Webcam":
            if st.button("Set baseline (neutral expression)", type="primary"):
                set_collecting_baseline(True)
                st.info("Hold a neutral expression for ~1 second. Baseline will be set automatically after 30 frames.")
        else:
            if st.button("Set baseline from sample video", type="primary"):
                if os.path.isfile(SAMPLE_VIDEO_PATH):
                    if set_baseline_from_video(SAMPLE_VIDEO_PATH, fd, fe, pc):
                        st.success("Baseline set from first 30 frames of sample video.")
                    else:
                        st.error("Could not set baseline (no face in first 30 frames or read error).")
                else:
                    st.error(f"Sample video not found: {SAMPLE_VIDEO_PATH}")
    with col2:
        state = get_state()
        if state.get("baseline_set"):
            st.success("Baseline set. Pain score is relative to your neutral expression.")
        elif state.get("collecting_baseline") and input_source == "Webcam":
            n = len(state.get("baseline_buffer", []))
            st.warning(f"Collecting baseline... {n}/30 frames.")

    if input_source == "Webcam":
        # Live webcam with streamlit-webrtc
        webrtc_streamer(
            key="pain-cam",
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )
    else:
        # Sample video
        if not os.path.isfile(SAMPLE_VIDEO_PATH):
            st.warning(f"Sample video not found at `{SAMPLE_VIDEO_PATH}`. Add the file to use this option.")
        else:
            st.video(SAMPLE_VIDEO_PATH)
            st.caption("Sample: stock footage (headache/pain). Set baseline above, then run detection below.")
            if st.button("Run pain detection on sample video"):
                progress = st.progress(0.0)
                status = st.empty()
                status.text("Processing video...")
                out_path = process_video_file(
                    SAMPLE_VIDEO_PATH, fd, fe, pc,
                    progress_callback=lambda p: progress.progress(p),
                )
                progress.progress(1.0)
                if out_path and os.path.isfile(out_path):
                    status.text("Done. Processed video with pain overlay:")
                    st.video(out_path)
                    try:
                        os.remove(out_path)
                    except OSError:
                        pass
                else:
                    status.text("")
                    st.error("Failed to process video.")

    # Gauge: 0–30 neutral, 30–60 mild discomfort, 60–100 high pain
    state = get_state()
    score = state.get("current_score", 0.0)
    category = state.get("current_category", "Neutral")

    st.subheader("Pain score (0–100)")
    st.progress(min(1.0, score / 100.0))
    if score < PainThresholds.NEUTRAL:
        zone = "Neutral (0–30)"
    elif score < PainThresholds.MILD:
        zone = "Mild discomfort (30–60)"
    else:
        zone = "High pain indicators (60–100)"
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
