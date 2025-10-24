"""
CNN model architecture for pain detection.
Lightweight MobileNetV2-based model for real-time inference.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class PainCNN(nn.Module):
    """
    Lightweight CNN model for pain detection.
    Based on MobileNetV2 architecture for efficient inference.
    """
    
    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        """
        Initialize CNN model.
        
        Args:
            num_classes: Number of output classes (1 for regression)
            pretrained: Whether to use pretrained weights
        """
        super(PainCNN, self).__init__()
        
        # Use MobileNetV2 as backbone
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        
        # Get the number of input features for the classifier
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier for pain regression
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights for the new layers
        self._initialize_weights()
    
    def forward(self, x):
        """Forward pass."""
        return self.backbone(x)
    
    def _initialize_weights(self):
        """Initialize weights for new layers."""
        for m in self.backbone.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class PainDetectorCNN(nn.Module):
    """
    Alternative CNN architecture specifically designed for pain detection.
    Uses custom architecture optimized for facial expression analysis.
    """
    
    def __init__(self, input_size: int = 224, num_classes: int = 1):
        """
        Initialize custom CNN model.
        
        Args:
            input_size: Input image size
            num_classes: Number of output classes
        """
        super(PainDetectorCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fifth convolutional block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_model(model_type: str = "mobilenet", num_classes: int = 1, 
                pretrained: bool = True) -> nn.Module:
    """
    Create pain detection model.
    
    Args:
        model_type: Type of model ("mobilenet" or "custom")
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        PyTorch model
    """
    if model_type == "mobilenet":
        return PainCNN(num_classes=num_classes, pretrained=pretrained)
    elif model_type == "custom":
        return PainDetectorCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_model(model_path: str, model_type: str = "mobilenet", 
              device: str = "cpu") -> Optional[nn.Module]:
    """
    Load pre-trained model from file.
    
    Args:
        model_path: Path to model file
        model_type: Type of model
        device: Device to load model on
        
    Returns:
        Loaded model or None if failed
    """
    try:
        # Create model
        model = create_model(model_type=model_type)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def save_model(model: nn.Module, model_path: str, 
              additional_info: Optional[dict] = None):
    """
    Save model to file.
    
    Args:
        model: Model to save
        model_path: Path to save model
        additional_info: Additional information to save
    """
    try:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_type': type(model).__name__
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, model_path)
        print(f"Model saved to {model_path}")
        
    except Exception as e:
        print(f"Error saving model: {e}")


def get_model_info(model: nn.Module) -> dict:
    """
    Get model information.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'model_type': type(model).__name__
    }
