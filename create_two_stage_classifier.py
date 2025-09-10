#!/usr/bin/env python3
"""
Create a two-stage piece classifier:
1. Color classifier (white vs black)
2. Piece type classifier (6 types per color)

This approach should achieve 70%+ accuracy by simplifying the problem.
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
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import time
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChessPieceDataset(Dataset):
    def __init__(self, data_dir, transform=None, stage="color"):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.stage = stage  # "color" or "piece_type"
        self.samples = []
        self.labels = []
        
        # Load all piece images
        self.load_samples()
        
    def load_samples(self):
        """Load all piece images and create labels based on stage"""
        piece_classes = [
            'white_pawn', 'white_rook', 'white_knight', 'white_bishop', 'white_queen', 'white_king',
            'black_pawn', 'black_rook', 'black_knight', 'black_bishop', 'black_queen', 'black_king'
        ]
        
        for piece_class in piece_classes:
            piece_dir = self.data_dir / piece_class
            if piece_dir.exists():
                for img_path in piece_dir.glob("*.png"):
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            self.samples.append(str(img_path))
                            
                            if self.stage == "color":
                                # Color classification: 0=white, 1=black
                                label = 0 if 'white' in piece_class else 1
                            else:
                                # Piece type classification: 0=pawn, 1=rook, 2=knight, 3=bishop, 4=queen, 5=king
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
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224
        img = cv2.resize(img, (224, 224))
        
        # Convert to PIL Image
        img = Image.fromarray(img)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        return img, label

class TwoStageClassifier:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.color_model = None
        self.piece_model = None
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def create_model(self, num_classes):
        """Create a ResNet18 model for classification"""
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    
    def train_stage(self, stage, epochs=10, batch_size=32, learning_rate=0.001):
        """Train a single stage of the classifier"""
        logger.info(f"Training {stage} classifier...")
        
        # Create datasets
        train_dataset = ChessPieceDataset(self.data_dir, self.train_transform, stage)
        val_dataset = ChessPieceDataset(self.data_dir, self.val_transform, stage)
        
        # Split data (80% train, 20% val)
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        num_classes = 2 if stage == "color" else 6
        model = self.create_model(num_classes)
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        # Training loop
        best_val_acc = 0
        train_losses = []
        val_accuracies = []
        
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
                
                if batch_idx % 50 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
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
            
            train_losses.append(avg_train_loss)
            val_accuracies.append(val_acc)
            
            logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f"models/{stage}_classifier_best.pt")
            
            scheduler.step()
        
        # Load best model
        model.load_state_dict(torch.load(f"models/{stage}_classifier_best.pt"))
        
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
        
        # Print classification report
        class_names = ['white', 'black'] if stage == "color" else ['pawn', 'rook', 'knight', 'bishop', 'queen', 'king']
        report = classification_report(all_targets, all_preds, target_names=class_names)
        logger.info(f"\n{class_names} Classification Report:\n{report}")
        
        # Save model
        if stage == "color":
            self.color_model = model
        else:
            self.piece_model = model
        
        return model, best_val_acc
    
    def train_both_stages(self):
        """Train both color and piece type classifiers"""
        logger.info("Starting two-stage classifier training...")
        
        # Create models directory
        Path("models").mkdir(exist_ok=True)
        
        # Train color classifier
        logger.info("="*50)
        logger.info("TRAINING COLOR CLASSIFIER")
        logger.info("="*50)
        color_model, color_acc = self.train_stage("color", epochs=15)
        
        # Train piece type classifier
        logger.info("="*50)
        logger.info("TRAINING PIECE TYPE CLASSIFIER")
        logger.info("="*50)
        piece_model, piece_acc = self.train_stage("piece_type", epochs=15)
        
        # Calculate combined accuracy
        combined_acc = color_acc * piece_acc
        
        logger.info("="*50)
        logger.info("TRAINING COMPLETE")
        logger.info("="*50)
        logger.info(f"Color Classifier Accuracy: {color_acc:.3f}")
        logger.info(f"Piece Type Classifier Accuracy: {piece_acc:.3f}")
        logger.info(f"Combined Accuracy: {combined_acc:.3f}")
        
        return color_model, piece_model, combined_acc
    
    def predict(self, img_path):
        """Predict piece using two-stage approach"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load and preprocess image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = Image.fromarray(img)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Stage 1: Color classification
        self.color_model.eval()
        with torch.no_grad():
            color_output = self.color_model(img_tensor)
            color_pred = color_output.argmax(dim=1).item()
            color_conf = torch.softmax(color_output, dim=1).max().item()
        
        # Stage 2: Piece type classification
        self.piece_model.eval()
        with torch.no_grad():
            piece_output = self.piece_model(img_tensor)
            piece_pred = piece_output.argmax(dim=1).item()
            piece_conf = torch.softmax(piece_output, dim=1).max().item()
        
        # Combine predictions
        color = "white" if color_pred == 0 else "black"
        piece_types = ['pawn', 'rook', 'knight', 'bishop', 'queen', 'king']
        piece_type = piece_types[piece_pred]
        
        final_pred = f"{color}_{piece_type}"
        combined_conf = color_conf * piece_conf
        
        return final_pred, combined_conf, color_conf, piece_conf

def main():
    # Check if dataset exists
    data_dir = Path("grey_background_dataset/pieces/train")
    if not data_dir.exists():
        logger.error(f"Dataset not found at {data_dir}")
        logger.info("Please ensure the grey_background_dataset exists with the correct structure")
        return
    
    # Create and train two-stage classifier
    classifier = TwoStageClassifier(data_dir)
    color_model, piece_model, combined_acc = classifier.train_both_stages()
    
    logger.info(f"\n🎯 FINAL RESULTS:")
    logger.info(f"Combined Accuracy: {combined_acc:.1%}")
    logger.info(f"Expected Real-World Performance: {combined_acc:.1%}")
    
    if combined_acc >= 0.7:
        logger.info("✅ SUCCESS: Achieved target accuracy of 70%+")
    else:
        logger.info("⚠️  WARNING: Did not achieve target accuracy. Consider:")
        logger.info("   - More training data")
        logger.info("   - Data augmentation")
        logger.info("   - Longer training")
        logger.info("   - Different model architecture")

if __name__ == "__main__":
    main()
