#!/usr/bin/env python3
"""
Quick training of a robust two-stage piece classifier with strong anti-overfitting measures.
This version is optimized for speed while maintaining high real-world accuracy.
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
from collections import defaultdict, Counter
import time
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuickChessPieceDataset(Dataset):
    def __init__(self, data_dir, transform=None, stage="color", max_samples_per_class=200):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.stage = stage
        self.max_samples_per_class = max_samples_per_class
        self.samples = []
        self.labels = []
        
        self.load_samples()
        
    def load_samples(self):
        """Load piece images with balanced sampling"""
        piece_classes = [
            'white_pawn', 'white_rook', 'white_knight', 'white_bishop', 'white_queen', 'white_king',
            'black_pawn', 'black_rook', 'black_knight', 'black_bishop', 'black_queen', 'black_king'
        ]
        
        for piece_class in piece_classes:
            piece_dir = self.data_dir / piece_class
            if piece_dir.exists():
                # Get all images for this class
                all_images = list(piece_dir.glob("*.png"))
                
                # Sample up to max_samples_per_class to balance the dataset
                if len(all_images) > self.max_samples_per_class:
                    import random
                    all_images = random.sample(all_images, self.max_samples_per_class)
                
                for img_path in all_images:
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            self.samples.append(str(img_path))
                            
                            if self.stage == "color":
                                label = 0 if 'white' in piece_class else 1
                            else:
                                piece_type = piece_class.split('_')[1]
                                piece_type_map = {
                                    'pawn': 0, 'rook': 1, 'knight': 2, 
                                    'bishop': 3, 'queen': 4, 'king': 5
                                }
                                label = piece_type_map[piece_type]
                            
                            self.labels.append(label)
                    except Exception as e:
                        logger.warning(f"Failed to load {img_path}: {e}")
        
        logger.info(f"Loaded {len(self.samples)} samples for {self.stage} classification")
        logger.info(f"Label distribution: {Counter(self.labels)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load and preprocess image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

class QuickRobustClassifier:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        
        # Lightweight transforms for speed
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def create_lightweight_model(self, num_classes):
        """Create a lightweight but effective model"""
        model = models.mobilenet_v2(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, num_classes)
        )
        return model
    
    def train_stage_quick(self, stage, epochs=8, batch_size=64, learning_rate=0.001):
        """Quick training with strong anti-overfitting"""
        logger.info(f"Quick training {stage} classifier...")
        
        # Create balanced datasets
        train_dataset = QuickChessPieceDataset(self.data_dir, self.train_transform, stage, max_samples_per_class=150)
        val_dataset = QuickChessPieceDataset(self.data_dir, self.val_transform, stage, max_samples_per_class=50)
        
        # Split data
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Create lightweight model
        num_classes = 2 if stage == "color" else 6
        model = self.create_lightweight_model(num_classes)
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Strong anti-overfitting setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)  # Strong weight decay
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training with early stopping
        best_val_acc = 0
        patience = 3
        patience_counter = 0
        
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
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
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
                patience_counter = 0
                torch.save(model.state_dict(), f"models/{stage}_classifier_quick.pt")
                logger.info(f"New best validation accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            scheduler.step()
        
        # Load best model
        model.load_state_dict(torch.load(f"models/{stage}_classifier_quick.pt"))
        
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
        class_names = ['white', 'black'] if stage == "color" else ['pawn', 'rook', 'knight', 'bishop', 'queen', 'king']
        report = classification_report(all_targets, all_preds, target_names=class_names, zero_division=0)
        logger.info(f"\n{class_names} Classification Report:\n{report}")
        
        return model, best_val_acc
    
    def train_both_stages_quick(self):
        """Quick training of both stages"""
        logger.info("Starting quick two-stage classifier training...")
        
        # Create models directory
        Path("models").mkdir(exist_ok=True)
        
        # Train color classifier (should be very fast and accurate)
        logger.info("="*50)
        logger.info("TRAINING COLOR CLASSIFIER (Quick)")
        logger.info("="*50)
        color_model, color_acc = self.train_stage_quick("color", epochs=6, batch_size=64)
        
        # Train piece type classifier
        logger.info("="*50)
        logger.info("TRAINING PIECE TYPE CLASSIFIER (Quick)")
        logger.info("="*50)
        piece_model, piece_acc = self.train_stage_quick("piece_type", epochs=8, batch_size=64)
        
        # Calculate combined accuracy
        combined_acc = color_acc * piece_acc
        
        logger.info("="*50)
        logger.info("QUICK TRAINING COMPLETE")
        logger.info("="*50)
        logger.info(f"Color Classifier Accuracy: {color_acc:.3f}")
        logger.info(f"Piece Type Classifier Accuracy: {piece_acc:.3f}")
        logger.info(f"Combined Accuracy: {combined_acc:.3f}")
        
        return color_model, piece_model, combined_acc

def main():
    # Check if dataset exists
    data_dir = Path("grey_background_dataset/pieces/train")
    if not data_dir.exists():
        logger.error(f"Dataset not found at {data_dir}")
        return
    
    # Create and train quick robust classifier
    classifier = QuickRobustClassifier(data_dir)
    color_model, piece_model, combined_acc = classifier.train_both_stages_quick()
    
    logger.info(f"\n🎯 QUICK TRAINING RESULTS:")
    logger.info(f"Combined Accuracy: {combined_acc:.1%}")
    logger.info(f"Expected Real-World Performance: {combined_acc:.1%}")
    
    if combined_acc >= 0.7:
        logger.info("✅ SUCCESS: Achieved target accuracy of 70%+")
        logger.info("Models saved to models/color_classifier_quick.pt and models/piece_type_classifier_quick.pt")
    else:
        logger.info("⚠️  WARNING: Did not achieve target accuracy. Consider:")
        logger.info("   - Increasing max_samples_per_class")
        logger.info("   - More training epochs")
        logger.info("   - Different model architecture")

if __name__ == "__main__":
    main()
