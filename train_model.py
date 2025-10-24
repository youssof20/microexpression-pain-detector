"""
Training script for CNN pain detection model.
Handles data loading, training, and model saving.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import json
from typing import List, Tuple, Optional
import argparse
from pathlib import Path

from src.models.cnn_model import create_model, save_model, get_model_info


class PainDataset(Dataset):
    """
    Dataset class for pain detection training.
    """
    
    def __init__(self, data_dir: str, transform=None):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing training data
            transform: Image transformations
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Tuple[str, float]]:
        """
        Load dataset samples.
        
        Returns:
            List of (image_path, pain_score) tuples
        """
        samples = []
        
        # Look for images and corresponding labels
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, file)
                    
                    # Look for corresponding label file
                    label_path = image_path.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
                    
                    if os.path.exists(label_path):
                        try:
                            with open(label_path, 'r') as f:
                                pain_score = float(f.read().strip())
                            samples.append((image_path, pain_score))
                        except:
                            continue
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, pain_score = self.samples[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(pain_score, dtype=torch.float32)


def get_transforms(train: bool = True) -> transforms.Compose:
    """
    Get image transformations.
    
    Args:
        train: Whether this is for training (includes augmentation)
        
    Returns:
        Composed transforms
    """
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                num_epochs: int = 50, learning_rate: float = 0.001,
                device: str = "cpu") -> List[float]:
    """
    Train the model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        
    Returns:
        List of validation losses
    """
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output.squeeze(), target)
                val_loss += loss.item()
        
        val_losses.append(val_loss / len(val_loader))
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
        
        scheduler.step()
    
    return val_losses


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train pain detection CNN')
    parser.add_argument('--data_dir', type=str, required=True, help='Training data directory')
    parser.add_argument('--output_dir', type=str, default='src/models/pretrained', help='Output directory')
    parser.add_argument('--model_type', type=str, default='mobilenet', choices=['mobilenet', 'custom'], help='Model type')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device to train on')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model
    model = create_model(model_type=args.model_type)
    print(f"Created model: {get_model_info(model)}")
    
    # Create datasets
    train_dataset = PainDataset(args.data_dir, transform=get_transforms(train=True))
    val_dataset = PainDataset(args.data_dir, transform=get_transforms(train=False))
    
    # Split dataset (80% train, 20% val)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Train model
    val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    # Save model
    model_path = os.path.join(args.output_dir, f'pain_detector_{args.model_type}.pth')
    save_model(model, model_path, {
        'val_losses': val_losses,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate
    })
    
    print(f"Training completed. Model saved to {model_path}")


if __name__ == "__main__":
    main()
