#!/usr/bin/env python3
"""
Simple piece classifier training script that focuses on preventing overfitting.
Trains a single model to predict 6 piece types: pawn, knight, bishop, rook, king, queen.
"""

import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from collections import Counter
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class PieceDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: Path, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        
        # Map piece types to indices (simplified - no color distinction)
        self.piece_type_labels = {
            "pawn": 0, "knight": 1, "bishop": 2, "rook": 3, "queen": 4, "king": 5
        }
        self.idx_to_piece_type = {v: k for k, v in self.piece_type_labels.items()}
        
        self._load_samples()

    def _load_samples(self):
        for piece_dir in self.data_dir.iterdir():
            if piece_dir.is_dir():
                piece_name = piece_dir.name
                # Extract piece type from directory name (e.g., "white_pawn" -> "pawn")
                if '_' in piece_name:
                    piece_type = piece_name.split('_')[1]
                else:
                    piece_type = piece_name
                
                if piece_type in self.piece_type_labels:
                    for img_path in piece_dir.glob("*.png"):
                        self.samples.append((img_path, piece_type))
                        self.labels.append(self.piece_type_labels[piece_type])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, piece_type = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        label = self.piece_type_labels[piece_type]
        return image, label

class SimplePieceClassifier:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Strong data augmentation to prevent overfitting
        self.transform = transforms.Compose([
            transforms.Resize(100),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.piece_type_labels = {
            "pawn": 0, "knight": 1, "bishop": 2, "rook": 3, "queen": 4, "king": 5
        }
        self.num_classes = len(self.piece_type_labels)
        self.idx_to_piece_type = {v: k for k, v in self.piece_type_labels.items()}
        
        self.dataset = PieceDataset(self.data_dir, self.transform)
        
        # Split dataset into training and validation
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)
        
        self.model = self._get_model(self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        
        # Use AdamW with weight decay to prevent overfitting
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )

    def _get_model(self, num_classes):
        # Use EfficientNet-B0 for efficiency and good performance
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        model = model.to(self.device)
        return model

    def train(self, epochs: int = 15):
        logger.info(f"Training simple piece classifier...")
        logger.info(f"Loaded {len(self.dataset)} samples for piece type classification")
        
        # Log label distribution
        train_labels = [self.dataset.labels[i] for i in self.train_dataset.indices]
        val_labels = [self.dataset.labels[i] for i in self.val_dataset.indices]
        
        train_counts = Counter([self.idx_to_piece_type[l] for l in train_labels])
        val_counts = Counter([self.idx_to_piece_type[l] for l in val_labels])
        
        logger.info(f"Train distribution: {dict(train_counts)}")
        logger.info(f"Validation distribution: {dict(val_counts)}")
        
        best_val_accuracy = 0.0
        epochs_no_improve = 0
        patience = 5  # Early stopping patience
        
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0
            
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(self.train_dataset)
            train_losses.append(epoch_loss)
            
            # Validation phase
            val_accuracy, val_report = self.evaluate()
            val_accuracies.append(val_accuracy)
            
            logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {epoch_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_accuracy)
            
            # Early stopping logic
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_no_improve = 0
                
                # Save the best model
                model_path = Path("models/piece_classifier_simple.pt")
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), str(model_path))
                logger.info(f"New best validation accuracy: {best_val_accuracy:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    logger.info("Early stopping triggered.")
                    break
        
        # Plot training progress
        self._plot_training_progress(train_losses, val_accuracies)
        
        logger.info(f"\n🎯 PIECE CLASSIFIER RESULTS:")
        logger.info(f"Best validation accuracy: {best_val_accuracy*100:.1f}%")
        logger.info(f"Model saved to models/piece_classifier_simple.pt")
        
        return best_val_accuracy

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        accuracy = correct / total
        
        # Generate classification report
        target_names = [self.idx_to_piece_type[i] for i in sorted(self.idx_to_piece_type.keys())]
        report = classification_report(all_labels, all_predictions, target_names=target_names, output_dict=False)
        
        return accuracy, report

    def _plot_training_progress(self, train_losses, val_accuracies):
        """Plot training progress to help identify overfitting"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot training loss
        ax1.plot(train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Plot validation accuracy
        ax2.plot(val_accuracies)
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        logger.info("Training progress plot saved as 'training_progress.png'")

def main():
    data_dir = Path("grey_background_dataset/pieces/train")
    if not data_dir.exists():
        logger.error(f"Dataset not found at {data_dir}")
        return
    
    # Check if we have enough data
    total_samples = 0
    for piece_dir in data_dir.iterdir():
        if piece_dir.is_dir():
            count = len(list(piece_dir.glob("*.png")))
            total_samples += count
            logger.info(f"{piece_dir.name}: {count} samples")
    
    logger.info(f"Total samples: {total_samples}")
    
    if total_samples < 100:
        logger.warning("Very few samples available. Consider collecting more data.")
    
    # Train the classifier
    classifier = SimplePieceClassifier(data_dir)
    best_accuracy = classifier.train(epochs=15)
    
    logger.info(f"\n✅ Training completed!")
    logger.info(f"Best validation accuracy: {best_accuracy*100:.1f}%")
    logger.info("Model saved to models/piece_classifier_simple.pt")

if __name__ == "__main__":
    main()