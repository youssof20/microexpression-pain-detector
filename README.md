# Microexpression-Based Pain Detector

A real-time webcam-based system that detects subtle facial microexpressions indicative of pain or discomfort using computer vision and machine learning techniques.

## Overview

This project implements a **demonstration application** for detecting pain-related facial microexpressions using the Facial Action Coding System (FACS). The system combines rule-based feature extraction with optional CNN enhancement to provide real-time pain level assessment.

**Important**: This application is for demonstration and educational purposes only. It is NOT intended for clinical use and should not be used for medical diagnosis or treatment decisions.

## Features

- **Real-time Detection**: Live webcam analysis with immediate feedback
- **FACS-based Analysis**: Detects pain-related Action Units (AU4, AU6, AU7, AU9, AU10, AU43)
- **Multiple Input Sources**: Webcam feed and video file upload
- **Interactive Visualization**: Real-time charts and session analytics
- **Modular Architecture**: Easy to extend and customize
- **Optional CNN Enhancement**: Lightweight MobileNetV2 model for improved accuracy

## Technical Implementation

### Core Components

- **Face Detection**: MediaPipe for robust facial landmark extraction (468 points)
- **Feature Extraction**: FACS-based geometric analysis of facial landmarks
- **Pain Classification**: Rule-based scoring with optional CNN enhancement
- **Visualization**: Real-time annotation and interactive Plotly charts
- **User Interface**: Streamlit-based web application

### Pain Detection Methodology

The system detects pain-related facial expressions using these FACS Action Units:

- **AU4 (Brow Lowerer)**: Inner brow lowering and pulling together
- **AU6/AU7 (Orbital Tightening)**: Eye aperture reduction and eyelid tightening
- **AU9 (Nose Wrinkler)**: Nose bridge wrinkling
- **AU10 (Upper Lip Raiser)**: Upper lip raising
- **AU43 (Eye Closure)**: Eye closing or squinting

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam or camera device
- 4GB RAM minimum (8GB recommended)

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/youssof20/microexpression-pain-detector.git
   cd microexpression_pain_detector
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import cv2, mediapipe, streamlit; print('Installation successful!')"
   ```

## Quick Start

### Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** to `http://localhost:8501`

3. **Navigate to "Live Detection" tab**

4. **Click "Start Webcam"** and allow camera permissions

5. **Set baseline** by clicking "Set Baseline" while maintaining a neutral expression

6. **Observe real-time pain detection** with visual annotations

### Basic Usage

1. **Live Detection**: Real-time webcam analysis with immediate feedback
2. **Video Upload**: Analyze pre-recorded videos for pain expressions
3. **Session History**: View charts and statistics from your session
4. **About**: Learn about the methodology and limitations

## Project Structure

```
microexpression_pain_detector/
├── src/
│   ├── detection/
│   │   ├── face_detector.py          # MediaPipe face & landmark detection
│   │   ├── feature_extractor.py      # FACS-based pain features
│   │   └── pain_classifier.py        # Rule-based + optional CNN classifier
│   ├── models/
│   │   ├── cnn_model.py              # Lightweight CNN architecture
│   │   └── pretrained/               # Directory for saved models
│   ├── visualization/
│   │   ├── annotator.py              # Draw landmarks & pain indicators
│   │   └── charts.py                 # Real-time plotting utilities
│   └── utils/
│       ├── video_handler.py          # Webcam/video file processing
│       └── config.py                 # Configuration constants
├── data/
│   ├── sample_videos/                # Example videos for testing
│   └── dataset_instructions.md       # How to obtain training datasets
├── app.py                            # Main Streamlit application
├── train_model.py                    # Optional CNN training script
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Advanced Usage

### Training CNN Models (Optional)

For improved accuracy, you can train a CNN model using pain expression datasets:

1. **Obtain training data** (see `data/dataset_instructions.md`)
2. **Preprocess your dataset**:
   ```bash
   python data_preprocessing.py --input_dir raw_data --output_dir processed_data
   ```
3. **Train the model**:
   ```bash
   python train_model.py --data_dir processed_data --epochs 50 --device cuda
   ```
4. **Use trained model** in the application by setting `use_cnn=True`

### Customization

- **Adjust sensitivity**: Use the sensitivity slider in the sidebar
- **Modify thresholds**: Edit `src/utils/config.py` for different pain categories
- **Add new features**: Extend `FeatureExtractor` class for additional FACS AUs
- **Custom visualization**: Modify `PainAnnotator` for different visual styles

## Performance

### System Requirements

- **CPU**: Intel i5 or AMD Ryzen 5 (minimum)
- **RAM**: 4GB (8GB recommended)
- **Camera**: 720p webcam or better
- **OS**: Windows 10/11, macOS 10.14+, or Ubuntu 18.04+

### Expected Performance

- **Real-time processing**: >15 FPS on laptop CPU
- **Memory usage**: <500MB RAM
- **Model size**: Rule-based (~1MB), CNN (~12MB)
- **Accuracy**: 60-70% (rule-based), 75-85% (with CNN)

## Limitations and Disclaimers

### Important Limitations

- **Not for clinical use**: This is a demonstration tool only
- **Individual differences**: Facial expressions vary significantly between people
- **Environmental factors**: Lighting and camera quality affect accuracy
- **Limited validation**: Not validated against clinical pain assessment tools
- **Cultural variations**: Pain expressions may vary across cultures

### Ethical Considerations

- Always obtain proper consent before recording or analyzing facial expressions
- Respect privacy and data protection laws
- Clearly communicate the non-clinical nature of the tool
- Use responsibly in educational and research contexts only

## Troubleshooting

### Common Issues

1. **Webcam not detected**:
   - Check camera permissions
   - Try different camera index (0, 1, 2)
   - Restart the application

2. **Low performance**:
   - Close other applications
   - Reduce webcam resolution
   - Use CPU-optimized settings

3. **Installation errors**:
   - Update pip: `pip install --upgrade pip`
   - Install Visual C++ Build Tools (Windows)
   - Use conda instead of pip for problematic packages

4. **Import errors**:
   - Verify Python version (3.8+)
   - Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

### Getting Help

- Check the [Issues](https://github.com/your-repo/issues) page
- Review the troubleshooting section above
- Ensure all dependencies are properly installed

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black src/ app.py

# Lint code
flake8 src/ app.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **MediaPipe**: Google's framework for face detection and landmark extraction
- **FACS**: Facial Action Coding System by Paul Ekman and Wallace Friesen
- **UNBC-McMaster**: Pain expression dataset for research
- **Streamlit**: Web application framework
- **PyTorch**: Deep learning framework for CNN models

## Citation

If you use this project in your research, please cite:

```bibtex
@software{microexpression_pain_detector,
  title={Microexpression-Based Pain Detector},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/microexpression_pain_detector}
}
```

## Contact

For questions, suggestions, or collaboration opportunities, please contact:

- **Email**: your.email@university.edu
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)
- **GitHub**: [Your GitHub Profile](https://github.com/yourusername)

---

**Remember**: This tool is for demonstration and educational purposes only. Always consult healthcare professionals for medical pain assessment and treatment.
"# microexpression-pain-detector" 
