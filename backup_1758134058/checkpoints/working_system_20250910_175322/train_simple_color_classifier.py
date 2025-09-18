#!/usr/bin/env python3
"""
Simple, fast training of a color classifier for chess pieces.
This focuses only on distinguishing white vs black pieces, which should be very reliable.
"""

import logging
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
from collections import Counter
import time
from sklearn.metrics import classification_report
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleColorDataset(Dataset):
    def __init__(self, data_dir, transform=None, max_samples_per_color=500):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.max_samples_per_color = max_samples_per_color
        self.samples = []
        self.labels = []
        
        self.load_samples()
        
    def load_samples(self):
        """Load piece images with balanced white/black sampling"""
        # Get all white pieces
        white_pieces = ['white_pawn', 'white_rook', 'white_knight', 'white_bishop', 'white_queen', 'white_king']
        # Get all black pieces  
        black_pieces = ['black_pawn', 'black_rook', 'black_knight', 'black_bishop', 'black_queen', 'black_king']
        
        # Load white pieces (label = 0)
        for piece_class in white_pieces:
            piece_dir = self.data_dir / piece_class
            if piece_dir.exists():
                all_images = list(piece_dir.glob("*.png"))
                if len(all_images) > self.max_samples_per_color // len(white_pieces):
                    all_images = random.sample(all_images, self.max_samples_per_color // len(white_pieces))
                
                for img_path in all_images:
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            self.samples.append(str(img_path))
                            self.labels.append(0)  # White = 0
                    except Exception as e:
                        logger.warning(f"Failed to load {img_path}: {e}")
        
        # Load black pieces (label = 1)
        for piece_class in black_pieces:
            piece_dir = self.data_dir / piece_class
            if piece_dir.exists():
                all_images = list(piece_dir.glob("*.png"))
                if len(all_images) > self.max_samples_per_color // len(black_pieces):
                    all_images = random.sample(all_images, self.max_samples_per_color // len(black_pieces))
                
                for img_path in all_images:
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            self.samples.append(str(img_path))
                            self.labels.append(1)  # Black = 1
                    except Exception as e:
                        logger.warning(f"Failed to load {img_path}: {e}")
        
        logger.info(f"Loaded {len(self.samples)} samples for color classification")
        logger.info(f"White pieces: {self.labels.count(0)}, Black pieces: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load and preprocess image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))  # Small size for fast training
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

class SimpleColorClassifier:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        
        # Simple transforms - minimal augmentation to prevent overfitting
        self.train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def create_simple_model(self):
        """Create a very simple model for color classification"""
        model = models.mobilenet_v2(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.last_channel, 2)  # Only 2 classes: white/black
        )
        return model
    
    def train_color_classifier(self, epochs=5, batch_size=64, learning_rate=0.001):
        """Train the color classifier"""
        logger.info("Training simple color classifier...")
        
        # Create dataset
        dataset = SimpleColorDataset(self.data_dir, self.train_transform, max_samples_per_color=400)
        
        # Split data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Create model
        model = self.create_simple_model()
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Simple training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training with early stopping
        best_val_acc = 0
        patience = 2
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1)
                    val_correct += (pred == target).sum().item()
                    val_total += target.size(0)
            
            val_acc = val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            
            logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                torch.save(model.state_dict(), "models/color_classifier_simple.pt")
                logger.info(f"New best validation accuracy: {val_acc:.4f}")
            else:
                patience += 1
                if patience >= 2:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            scheduler.step()
        
        # Load best model
        model.load_state_dict(torch.load("models/color_classifier_simple.pt"))
        
        # Final evaluation
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Print results
        report = classification_report(all_targets, all_preds, target_names=['white', 'black'], zero_division=0)
        logger.info(f"\nColor Classification Report:\n{report}")
        
        return model, best_val_acc

def main():
    # Check if dataset exists
    data_dir = Path("grey_background_dataset/pieces/train")
    if not data_dir.exists():
        logger.error(f"Dataset not found at {data_dir}")
        return
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    # Create and train color classifier
    classifier = SimpleColorClassifier(data_dir)
    model, accuracy = classifier.train_color_classifier()
    
    logger.info(f"\n🎯 COLOR CLASSIFIER RESULTS:")
    logger.info(f"Accuracy: {accuracy:.1%}")
    logger.info(f"Model saved to models/color_classifier_simple.pt")
    
    if accuracy >= 0.9:
        logger.info("✅ SUCCESS: High accuracy achieved!")
    elif accuracy >= 0.8:
        logger.info("✅ GOOD: Reasonable accuracy achieved")
    else:
        logger.info("⚠️  WARNING: Lower than expected accuracy")

if __name__ == "__main__":
    main()
