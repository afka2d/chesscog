#!/usr/bin/env python3
"""
Quick Training Script for Chess Recognition Models

This script provides a simplified training approach that works with your custom dataset.
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
import argparse
from PIL import Image
import chess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt

class ChessDataset(Dataset):
    """Simple dataset for chess piece recognition"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        """Load all samples from the data directory"""
        for json_file in self.data_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                annotation = json.load(f)
            
            img_file = json_file.with_suffix('.png')
            if img_file.exists():
                self.samples.append({
                    'image_path': img_file,
                    'annotation': annotation
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Create simple labels based on FEN
        fen = sample['annotation']['fen']
        board = chess.Board(fen)
        
        # Count pieces for simple classification
        piece_count = len(board.piece_map())
        
        # Simple classification: 0=empty, 1=few pieces, 2=many pieces
        if piece_count == 0:
            label = 0
        elif piece_count <= 8:
            label = 1
        else:
            label = 2
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_simple_model(num_classes=3):
    """Create a simple ResNet model for chess piece classification"""
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_model(model, train_loader, val_loader, num_epochs=10, device='cpu'):
    """Train the model"""
    print(f"🚀 Training on {device}...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_losses[-1]:.4f}, "
              f"Val Loss: {val_losses[-1]:.4f}, "
              f"Val Accuracy: {accuracy:.2f}%")
    
    return train_losses, val_losses

def main():
    parser = argparse.ArgumentParser(description="Quick training for chess recognition")
    parser.add_argument("--data_dir", default="training_output", help="Directory containing training data")
    parser.add_argument("--output_dir", default="quick_trained_models", help="Output directory for trained models")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    
    args = parser.parse_args()
    
    print("🎯 Quick Chess Recognition Model Training")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dir = Path(args.data_dir) / "train"
    val_dir = Path(args.data_dir) / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        print("❌ Training data not found. Please run simple_train.py first.")
        sys.exit(1)
    
    train_dataset = ChessDataset(train_dir)
    val_dataset = ChessDataset(val_dir)
    
    print(f"📊 Dataset sizes: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Create model
    model = create_simple_model()
    model = model.to(device)
    
    print(f"📈 Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, args.epochs, device)
    
    # Save model
    model_path = output_dir / "chess_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved to {model_path}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png")
    print(f"📊 Training curves saved to {output_dir / 'training_curves.png'}")
    
    print(f"\n🎉 Training completed! Model saved in {output_dir}")
    print("📁 You can now use this model for chess piece recognition")

if __name__ == "__main__":
    main() 