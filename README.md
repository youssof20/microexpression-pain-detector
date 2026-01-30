# Microexpression Pain Detector

A webcam app that tries to detect pain from facial expressions using computer vision.

## What it does

Uses your webcam to analyze facial movements and estimate if someone might be experiencing pain. Based on the Facial Action Coding System (FACS) - basically looking for specific facial muscle movements like brow furrowing, eye squinting, etc.

**Disclaimer**: This is just a student project for learning purposes, not a real medical tool.

## Setup

```bash
pip install -r requirements.txt
python -m streamlit run app.py --server.port 8502
```

Then go to `http://localhost:8502`

## How to use

1. Click "Start Webcam"
2. Let it detect your face
3. Click "Set Baseline" with a neutral expression
4. Make different expressions to see the pain detection in action

## What's inside

- `src/detection/` - Face detection and feature extraction
- `src/models/` - CNN model (optional, can use rule-based instead)
- `src/visualization/` - Drawing stuff on the video feed
- `app.py` - Main Streamlit app

## Requirements

- Python 3.8+
- Webcam
- The packages in requirements.txt

## Limitations

- Not accurate enough for real use
- Lighting matters a lot
- Works differently for different people
- Just a proof of concept

## Notes

Made this to learn about computer vision and pain detection research. If you want to train the CNN model yourself, check out `train_model.py` but the rule-based approach works okay for testing.
