#!/usr/bin/env python3
"""
Enhanced Piece Classifier with advanced data augmentation and model architecture
to achieve 90% accuracy target.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import random
import os
from pathlib import Path
from chesscog.core.dataset.dataset import build_dataset, Datasets
from chesscog.corner_detection.detect_corners import CN
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class AdvancedChessPieceAugmentation:
    """Advanced data augmentation specifically for chess pieces."""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img):
        # Random rotation with chess piece constraints
        if random.random() < self.p:
            angle = random.uniform(-15, 15)  # Limited rotation for chess pieces
            img = TF.rotate(img, angle, fill=128)  # Gray fill
        
        # Random horizontal flip (chess pieces are symmetric)
        if random.random() < self.p:
            img = TF.hflip(img)
        
        # Color jittering (subtle for chess pieces)
        if random.random() < self.p:
            img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
            img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))
            img = TF.adjust_saturation(img, random.uniform(0.8, 1.2))
        
        # Random perspective (simulate different viewing angles)
        if random.random() < self.p:
            img = TF.perspective(img, self._get_perspective_transform(), fill=128)
        
        # Random noise
        if random.random() < self.p:
            img = self._add_noise(img)
        
        # Random blur (simulate focus issues)
        if random.random() < self.p * 0.3:  # Less frequent
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        
        return img
    
    def _get_perspective_transform(self):
        """Generate perspective transform matrix."""
        # Subtle perspective changes
        w, h = 224, 448  # Assuming standard size
        src_points = [(0, 0), (w, 0), (w, h), (0, h)]
        dst_points = [
            (random.uniform(-10, 10), random.uniform(-10, 10)),
            (w + random.uniform(-10, 10), random.uniform(-10, 10)),
            (w + random.uniform(-10, 10), h + random.uniform(-10, 10)),
            (random.uniform(-10, 10), h + random.uniform(-10, 10))
        ]
        return TF._get_perspective_coeffs(src_points, dst_points)
    
    def _add_noise(self, img):
        """Add random noise to image."""
        img_array = np.array(img)
        noise = np.random.normal(0, 5, img_array.shape).astype(np.uint8)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EnhancedChessPieceClassifier(nn.Module):
    """Enhanced classifier using ResNet50 with custom modifications."""
    
    def __init__(self, num_classes=12, dropout_rate=0.3):
        super(EnhancedChessPieceClassifier, self).__init__()
        
        # Use ResNet50 as base (better than ResNet18)
        self.backbone = models.resnet50(pretrained=True)
        
        # Modify the final layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class ChessPieceTrainer:
    """Enhanced trainer with advanced techniques."""
    
    def __init__(self, config_path, model_save_dir="enhanced_models"):
        self.config = CN.load_yaml_with_base(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True)
        
        # Enhanced transforms
        self.train_transforms = transforms.Compose([
            transforms.Resize((224, 448)),
            AdvancedChessPieceAugmentation(p=0.7),  # Aggressive augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transforms = transforms.Compose([
            transforms.Resize((224, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
    
    def setup_model(self):
        """Setup model, optimizer, and loss function."""
        self.model = EnhancedChessPieceClassifier(
            num_classes=len(self.config.DATASET.CLASSES),
            dropout_rate=0.3
        ).to(self.device)
        
        # Use AdamW optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        
        # Cosine annealing scheduler with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Use Focal Loss for class imbalance
        self.criterion = FocalLoss(alpha=1, gamma=2)
    
    def create_datasets(self):
        """Create datasets with enhanced transforms."""
        # Override the config transforms temporarily
        original_transforms = self.config.DATASET.TRANSFORMS
        
        # Set enhanced transforms
        self.config.DATASET.TRANSFORMS = {
            'RESIZE': [224, 448],
            'NORMALIZE': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        }
        
        train_dataset = build_dataset(self.config, Datasets.TRAIN)
        val_dataset = build_dataset(self.config, Datasets.VAL)
        test_dataset = build_dataset(self.config, Datasets.TEST)
        
        # Restore original transforms
        self.config.DATASET.TRANSFORMS = original_transforms
        
        return train_dataset, val_dataset, test_dataset
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        return total_loss / len(train_loader), 100. * correct / total
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return total_loss / len(val_loader), 100. * correct / total
    
    def train(self, epochs=50, batch_size=32):
        """Main training loop."""
        print("Setting up model and datasets...")
        self.setup_model()
        train_dataset, val_dataset, test_dataset = self.create_datasets()
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
        print(f"Using device: {self.device}")
        
        best_val_acc = 0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model, self.model_save_dir / "best_model.pt")
                print(f"New best validation accuracy: {val_acc:.2f}%")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, self.model_save_dir / f"checkpoint_epoch_{epoch+1}.pt")
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        # Final evaluation on test set
        print("\n" + "="*50)
        print("FINAL TEST EVALUATION")
        print("="*50)
        test_loss, test_acc = self.validate(test_loader)
        print(f"Test Accuracy: {test_acc:.2f}%")
        
        # Plot training curves
        self.plot_training_curves(train_losses, val_losses, train_accs, val_accs)
        
        return test_acc
    
    def plot_training_curves(self, train_losses, val_losses, train_accs, val_accs):
        """Plot training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy curves
        ax2.plot(train_accs, label='Train Acc')
        ax2.plot(val_accs, label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.model_save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to run enhanced training."""
    print("Enhanced Chess Piece Classifier Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = ChessPieceTrainer('config/piece_classifier/ResNet_uniform.yaml')
    
    # Train the model
    test_accuracy = trainer.train(epochs=50, batch_size=32)
    
    print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")
    print(f"Model saved to: {trainer.model_save_dir}")

if __name__ == "__main__":
    main() 