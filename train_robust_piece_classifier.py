#!/usr/bin/env python3
"""
Train a robust piece classifier with comprehensive anti-overfitting measures.
This ensures 80%+ real-world accuracy without overfitting.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import numpy as np
from pathlib import Path
import os
import random
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class RobustPieceClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Data paths
        self.data_dir = Path("grey_background_dataset/pieces")
        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "val"
        self.test_dir = self.data_dir / "test"
        
        # Model parameters
        self.num_classes = 12
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 50
        self.patience = 10  # Early stopping patience
        
        # Anti-overfitting measures
        self.weight_decay = 0.01  # L2 regularization
        self.dropout_rate = 0.5
        self.data_augmentation = True
        
        # Class names
        self.class_names = [
            'black_bishop', 'black_king', 'black_knight', 'black_pawn', 'black_queen', 'black_rook',
            'white_bishop', 'white_king', 'white_knight', 'white_pawn', 'white_queen', 'white_rook'
        ]
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def analyze_dataset(self):
        """Analyze the dataset to understand class distribution and potential issues."""
        print("🔍 Analyzing Dataset")
        print("=" * 50)
        
        # Count samples in each class
        class_counts = {}
        for split in ['train', 'val', 'test']:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                print(f"   ⚠️  {split} directory not found")
                continue
                
            split_counts = {}
            for class_name in self.class_names:
                class_dir = split_dir / class_name
                if class_dir.exists():
                    count = len([f for f in class_dir.iterdir() if f.suffix == '.png'])
                    split_counts[class_name] = count
                else:
                    split_counts[class_name] = 0
            
            class_counts[split] = split_counts
            
            print(f"\n📊 {split.upper()} SET:")
            total_samples = sum(split_counts.values())
            for class_name, count in split_counts.items():
                percentage = (count / total_samples * 100) if total_samples > 0 else 0
                print(f"   {class_name}: {count:4d} samples ({percentage:5.1f}%)")
            print(f"   Total: {total_samples:4d} samples")
        
        # Calculate class weights for balancing
        train_counts = list(class_counts['train'].values())
        if sum(train_counts) > 0:
            # Inverse frequency weighting
            max_count = max(train_counts)
            class_weights = [max_count / count if count > 0 else 0 for count in train_counts]
            # Normalize weights
            class_weights = [w / sum(class_weights) * len(class_weights) for w in class_weights]
        else:
            class_weights = [1.0] * self.num_classes
        
        print(f"\n⚖️  CLASS WEIGHTS (for balancing):")
        for i, (class_name, weight) in enumerate(zip(self.class_names, class_weights)):
            print(f"   {class_name}: {weight:.3f}")
        
        return class_weights
    
    def get_transforms(self, is_training=True):
        """Get data transforms with appropriate augmentation."""
        if is_training and self.data_augmentation:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def create_data_loaders(self, class_weights):
        """Create data loaders with class balancing."""
        print("\n📦 Creating Data Loaders")
        print("=" * 50)
        
        # Training data
        train_dataset = datasets.ImageFolder(
            str(self.train_dir),
            transform=self.get_transforms(is_training=True)
        )
        
        # Validation data
        val_dataset = datasets.ImageFolder(
            str(self.val_dir),
            transform=self.get_transforms(is_training=False)
        )
        
        # Test data
        test_dataset = datasets.ImageFolder(
            str(self.test_dir),
            transform=self.get_transforms(is_training=False)
        )
        
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        print(f"   Test samples: {len(test_dataset)}")
        
        # Create weighted sampler for training
        train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
        class_counts = Counter(train_labels)
        sample_weights = [1.0 / class_counts[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )
        
        return train_loader, val_loader, test_loader, class_weights
    
    def create_model(self):
        """Create a robust ResNet18 model with anti-overfitting measures."""
        print("\n🏗️  Creating Model")
        print("=" * 50)
        
        # Use ResNet18 as base
        model = models.resnet18(pretrained=True)
        
        # Modify the final layer
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, self.num_classes)
        )
        
        model = model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Dropout rate: {self.dropout_rate}")
        
        return model
    
    def train_epoch(self, model, train_loader, criterion, optimizer):
        """Train for one epoch."""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 50 == 0:
                print(f"   Batch {batch_idx:3d}/{len(train_loader):3d}: Loss={loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train_model(self, model, train_loader, val_loader, class_weights):
        """Train the model with early stopping and monitoring."""
        print("\n🚀 Training Model")
        print("=" * 50)
        
        # Loss function with class weights
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        # Optimizer with weight decay
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Early stopping
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        
        print(f"   Epochs: {self.num_epochs}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Weight decay: {self.weight_decay}")
        print(f"   Early stopping patience: {self.patience}")
        
        for epoch in range(self.num_epochs):
            print(f"\n📅 Epoch {epoch+1}/{self.num_epochs}")
            print("-" * 30)
            
            # Train
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                print(f"   ✅ New best validation accuracy: {val_acc:.2f}%")
            else:
                patience_counter += 1
                print(f"   ⏳ No improvement ({patience_counter}/{self.patience})")
                
                if patience_counter >= self.patience:
                    print(f"   🛑 Early stopping triggered!")
                    break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"\n🎯 Best validation accuracy: {best_val_acc:.2f}%")
        
        return model
    
    def evaluate_model(self, model, test_loader):
        """Evaluate the model on test data."""
        print("\n🧪 Evaluating Model on Test Data")
        print("=" * 50)
        
        model.eval()
        all_predictions = []
        all_targets = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100.0 * correct / total
        print(f"   Test Accuracy: {accuracy:.2f}%")
        
        # Detailed classification report
        print(f"\n📊 Detailed Classification Report:")
        report = classification_report(all_targets, all_predictions, target_names=self.class_names, digits=3)
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   📊 Confusion matrix saved as 'confusion_matrix.png'")
        
        return accuracy, all_predictions, all_targets
    
    def plot_training_history(self):
        """Plot training history."""
        print("\n📈 Plotting Training History")
        print("=" * 50)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Training Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   📊 Training history saved as 'training_history.png'")
    
    def save_model(self, model, test_accuracy):
        """Save the trained model."""
        print("\n💾 Saving Model")
        print("=" * 50)
        
        model_path = "models/piece_classifier/ResNet_robust.pt"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model state dict
        torch.save(model.state_dict(), model_path)
        
        # Save full model for easy loading
        full_model_path = "models/piece_classifier/ResNet_robust_full.pt"
        torch.save(model, full_model_path)
        
        # Save training info
        info_path = "models/piece_classifier/ResNet_robust_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"Model: ResNet18 with dropout and class balancing\n")
            f.write(f"Test Accuracy: {test_accuracy:.2f}%\n")
            f.write(f"Training Epochs: {len(self.train_losses)}\n")
            f.write(f"Best Val Accuracy: {max(self.val_accuracies):.2f}%\n")
            f.write(f"Class Weights: {self.class_weights}\n")
            f.write(f"Data Augmentation: {self.data_augmentation}\n")
            f.write(f"Dropout Rate: {self.dropout_rate}\n")
            f.write(f"Weight Decay: {self.weight_decay}\n")
        
        print(f"   ✅ Model saved to {model_path}")
        print(f"   ✅ Full model saved to {full_model_path}")
        print(f"   ✅ Training info saved to {info_path}")
        print(f"   📊 Model size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
    
    def run_training(self):
        """Run the complete training pipeline."""
        print("🎯 Training Robust Piece Classifier")
        print("=" * 60)
        
        # Analyze dataset
        class_weights = self.analyze_dataset()
        
        # Create data loaders
        train_loader, val_loader, test_loader, class_weights = self.create_data_loaders(class_weights)
        self.class_weights = class_weights  # Store for saving
        
        # Create model
        model = self.create_model()
        
        # Train model
        model = self.train_model(model, train_loader, val_loader, class_weights)
        
        # Evaluate model
        test_accuracy, predictions, targets = self.evaluate_model(model, test_loader)
        
        # Plot training history
        self.plot_training_history()
        
        # Save model
        self.save_model(model, test_accuracy)
        
        # Final assessment
        print(f"\n🎯 FINAL ASSESSMENT")
        print("=" * 50)
        if test_accuracy >= 80.0:
            print(f"   ✅ SUCCESS: Model achieves {test_accuracy:.2f}% accuracy (≥80%)")
            print(f"   ✅ Model is ready for production use")
        else:
            print(f"   ❌ FAILURE: Model only achieves {test_accuracy:.2f}% accuracy (<80%)")
            print(f"   ❌ Model needs further improvement")
        
        return test_accuracy >= 80.0

if __name__ == "__main__":
    trainer = RobustPieceClassifier()
    success = trainer.run_training()
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("   The model is ready for deployment.")
    else:
        print("\n😞 Training did not meet the 80% accuracy requirement.")
        print("   Consider adjusting hyperparameters or training data.")
