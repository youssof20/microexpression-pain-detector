# Dataset Instructions for Pain Detection Training

This document provides instructions for obtaining and using datasets to train the CNN model for enhanced pain detection accuracy.

## Available Datasets

### 1. UNBC-McMaster Shoulder Pain Expression Archive

**Description**: The most comprehensive dataset for pain expression analysis, containing video sequences of shoulder pain patients.

**Access**:
- Website: [UNBC-McMaster Pain Archive](http://www.paindetection.org/)
- Registration required (free for academic use)
- Contains 200 video sequences with pain intensity ratings (0-10 scale)

**Dataset Structure**:
```
UNBC-McMaster/
├── Videos/
│   ├── Subject_001/
│   │   ├── video_001.avi
│   │   └── video_002.avi
│   └── Subject_002/
│       └── ...
├── Labels/
│   ├── Subject_001_labels.txt
│   └── Subject_002_labels.txt
└── README.txt
```

**Preprocessing Steps**:
1. Extract frames from videos at 1 FPS
2. Detect and crop faces using MediaPipe
3. Resize to 224x224 pixels
4. Create label files with pain scores (0-1 normalized)

### 2. CK+ (Extended Cohn-Kanade) Dataset

**Description**: Facial expression dataset that can be adapted for pain detection training.

**Access**:
- Website: [CK+ Dataset](https://www.pitt.edu/~emotion/ck-spread.htm)
- Free download after registration
- Contains 593 sequences from 123 subjects

**Adaptation for Pain Detection**:
- Use "disgust" and "fear" expressions as pain indicators
- Combine with other datasets for better pain representation

### 3. BP4D-Spontaneous Dataset

**Description**: Spontaneous facial expression dataset with FACS annotations.

**Access**:
- Available through research collaboration
- Contains 328 videos with FACS coding
- Useful for training FACS-based features

## Data Preparation Script

Create a preprocessing script to prepare your dataset:

```python
# data_preprocessing.py
import cv2
import os
import numpy as np
from pathlib import Path
from src.detection.face_detector import FaceDetector

def preprocess_dataset(input_dir, output_dir):
    """
    Preprocess dataset for training.
    
    Args:
        input_dir: Directory containing raw videos/images
        output_dir: Directory to save processed data
    """
    face_detector = FaceDetector()
    
    # Create output directories
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)
    
    # Process each video/image
    for video_path in Path(input_dir).glob("*.avi"):
        process_video(video_path, face_detector, output_dir)

def process_video(video_path, face_detector, output_dir):
    """Process single video file."""
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract face every 30 frames (1 FPS)
        if frame_count % 30 == 0:
            landmarks = face_detector.detect_face(frame)
            if landmarks is not None:
                face_crop = face_detector.get_face_crop(frame, landmarks)
                if face_crop is not None:
                    # Save face crop
                    output_path = f"{output_dir}/images/{video_path.stem}_{frame_count:06d}.jpg"
                    cv2.imwrite(output_path, face_crop)
                    
                    # Save label (you'll need to map this to actual pain scores)
                    label_path = f"{output_dir}/labels/{video_path.stem}_{frame_count:06d}.txt"
                    with open(label_path, 'w') as f:
                        f.write("0.5")  # Placeholder - replace with actual pain score
        
        frame_count += 1
    
    cap.release()
```

## Training Instructions

### 1. Prepare Your Dataset

1. Download one of the datasets above
2. Run the preprocessing script to extract face crops
3. Organize data in the following structure:
```
training_data/
├── image_001.jpg
├── image_001.txt
├── image_002.jpg
├── image_002.txt
└── ...
```

### 2. Train the Model

```bash
# Basic training
python train_model.py --data_dir training_data --epochs 50

# Advanced training with custom parameters
python train_model.py \
    --data_dir training_data \
    --output_dir src/models/pretrained \
    --model_type mobilenet \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --device cuda
```

### 3. Model Evaluation

After training, evaluate your model:

```python
# evaluate_model.py
import torch
from src.models.cnn_model import load_model
from src.detection.pain_classifier import PainClassifier

# Load trained model
model = load_model("src/models/pretrained/pain_detector_mobilenet.pth")

# Test on validation set
# ... evaluation code ...
```

## Expected Results

With proper training data, you should expect:

- **Rule-based only**: 60-70% accuracy on pain detection
- **With CNN enhancement**: 75-85% accuracy on pain detection
- **Real-time performance**: >15 FPS on laptop CPU

## Troubleshooting

### Common Issues

1. **Low accuracy**: Ensure diverse training data with proper pain score annotations
2. **Overfitting**: Use data augmentation and regularization
3. **Slow training**: Reduce batch size or use GPU acceleration
4. **Memory issues**: Process data in smaller batches

### Data Quality Tips

- Ensure good lighting in training videos
- Include diverse subjects (age, gender, ethnicity)
- Balance pain/no-pain samples
- Validate pain scores with clinical experts

## Legal and Ethical Considerations

- Ensure proper consent for all training data
- Follow institutional review board (IRB) guidelines
- Respect patient privacy and data protection laws
- Clearly label outputs as non-clinical/demo only

## Additional Resources

- [FACS Manual](https://www.paulekman.com/facs/)
- [Pain Expression Research Papers](https://scholar.google.com/scholar?q=pain+facial+expression+detection)
- [MediaPipe Documentation](https://mediapipe.dev/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
