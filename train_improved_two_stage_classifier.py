#!/usr/bin/env python3
"""
Improved Two-Stage Chess Piece Classifier Training Script
Addresses overfitting through data augmentation, early stopping, cross-validation, and regularization.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ColorDataset(Dataset):
    """Dataset for color classification (2 classes: black, white)."""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {'black': 0, 'white': 1}
        
        # Load samples
        self._load_samples()
        logger.info(f"Color dataset: Loaded {len(self.samples)} samples from 2 classes")
        
    def _load_samples(self):
        """Load all samples with their color labels."""
        for class_name in ['black_pawn', 'black_rook', 'black_knight', 'black_bishop', 'black_queen', 'black_king']:
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.exists(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(class_dir, filename), 0))  # black = 0
        
        for class_name in ['white_pawn', 'white_rook', 'white_knight', 'white_bishop', 'white_queen', 'white_king']:
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.exists(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(class_dir, filename), 1))  # white = 1
        
        # Shuffle samples
        np.random.shuffle(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            logger.error(f"Error loading {img_path}: {e}")
            # Return a placeholder if image loading fails
            if self.transform:
                placeholder = torch.zeros(3, 100, 200)
                return placeholder, label
            return None, label

class PieceTypeDataset(Dataset):
    """Dataset for piece type classification (6 classes: pawn, rook, knight, bishop, queen, king)."""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {'pawn': 0, 'rook': 1, 'knight': 2, 'bishop': 3, 'queen': 4, 'king': 5}
        
        # Load samples
        self._load_samples()
        logger.info(f"Piece type dataset: Loaded {len(self.samples)} samples from 6 classes")
        
    def _load_samples(self):
        """Load all samples with their piece type labels."""
        piece_types = ['pawn', 'rook', 'knight', 'bishop', 'queen', 'king']
        
        for piece_type in piece_types:
            piece_type_idx = self.class_to_idx[piece_type]
            
            # Load both black and white pieces of this type
            for color in ['black', 'white']:
                class_name = f"{color}_{piece_type}"
                class_dir = os.path.join(self.data_dir, class_name)
                if os.path.exists(class_dir):
                    for filename in os.listdir(class_dir):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((os.path.join(class_dir, filename), piece_type_idx))
        
        # Shuffle samples
        np.random.shuffle(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            logger.error(f"Error loading {img_path}: {e}")
            # Return a placeholder if image loading fails
            if self.transform:
                placeholder = torch.zeros(3, 100, 200)
                return placeholder, label
            return None, label

class ColorClassifier(nn.Module):
    """Improved color classifier with regularization."""
    
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(ColorClassifier, self).__init__()
        
        # Use a pre-trained ResNet18 as backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze early layers to prevent overfitting
        for param in list(self.backbone.parameters())[:-20]:  # Freeze all but last few layers
            param.requires_grad = False
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class PieceTypeClassifier(nn.Module):
    """Improved piece type classifier with regularization."""
    
    def __init__(self, num_classes=6, dropout_rate=0.3):
        super(PieceTypeClassifier, self).__init__()
        
        # Use a pre-trained ResNet18 as backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze early layers to prevent overfitting
        for param in list(self.backbone.parameters())[:-20]:  # Freeze all but last few layers
            param.requires_grad = False
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

def get_transforms(is_training=True):
    """Get transforms with robust data augmentation for training."""
    
    if is_training:
        return transforms.Compose([
            transforms.Resize((100, 200)),  # Match original training size
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((100, 200)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, patience=10):
    """Train model with early stopping and learning rate scheduling."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    logger.info(f"Training on {device}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
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
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 50 == 0:
                logger.info(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.4f}')
        
        train_loss /= len(train_loader)
        train_accuracy = 100. * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100. * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}: '
                   f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
                   f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            try:
                torch.save(model.state_dict(), f'best_{model.__class__.__name__}.pth')
                logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
            except Exception as e:
                logger.warning(f"Failed to save best model: {e}")
        else:
            patience_counter += 1
            logger.info(f"Patience counter: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, model.__class__.__name__)
    
    return model, train_losses, val_losses, train_accuracies, val_accuracies

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, model_name):
    """Plot training and validation curves."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title(f'{model_name} - Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(train_accuracies, label='Training Accuracy', color='blue')
    ax2.plot(val_accuracies, label='Validation Accuracy', color='red')
    ax2.set_title(f'{model_name} - Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, test_loader, class_names):
    """Evaluate model performance with detailed metrics."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100. * sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets)
    
    # Classification report
    report = classification_report(all_targets, all_predictions, target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    logger.info(f"Overall Accuracy: {accuracy:.2f}%")
    logger.info(f"Per-class accuracy:")
    for class_name, metrics in report.items():
        if isinstance(metrics, dict) and 'precision' in metrics:
            logger.info(f"  {class_name}: {metrics['precision']:.3f} precision, {metrics['recall']:.3f} recall")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model.__class__.__name__} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{model.__class__.__name__}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, report, cm

def main():
    """Main training function."""
    
    # Configuration
    data_dir = "enhanced_training_dataset_v2/pieces/train"
    batch_size = 16  # Smaller batch size for better generalization
    num_epochs = 100  # Will use early stopping
    learning_rate = 0.0001  # Lower learning rate to prevent overfitting
    patience = 8  # Lower patience for earlier stopping
    
    logger.info("Starting improved two-stage classifier training...")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Max epochs: {num_epochs}")
    logger.info(f"Patience: {patience}")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Data directory {data_dir} not found!")
        return
    
    # Get transforms
    train_transform = get_transforms(is_training=True)
    val_transform = get_transforms(is_training=False)
    
    # Create color dataset
    logger.info("Creating color classification dataset...")
    color_dataset = ColorDataset(data_dir, transform=train_transform)
    
    # Split color dataset: 70% train, 15% validation, 15% test
    total_color_size = len(color_dataset)
    color_train_size = int(0.7 * total_color_size)
    color_val_size = int(0.15 * total_color_size)
    color_test_size = total_color_size - color_train_size - color_val_size
    
    color_train_dataset, color_val_dataset, color_test_dataset = random_split(
        color_dataset, [color_train_size, color_val_size, color_test_size]
    )
    
    # Apply appropriate transforms
    color_train_dataset.dataset.transform = train_transform
    color_val_dataset.dataset.transform = val_transform
    color_test_dataset.dataset.transform = val_transform
    
    logger.info(f"Color dataset split: Train={len(color_train_dataset)}, Val={len(color_val_dataset)}, Test={len(color_test_dataset)}")
    
    # Create piece type dataset
    logger.info("Creating piece type classification dataset...")
    piece_dataset = PieceTypeDataset(data_dir, transform=train_transform)
    
    # Split piece type dataset: 70% train, 15% validation, 15% test
    total_piece_size = len(piece_dataset)
    piece_train_size = int(0.7 * total_piece_size)
    piece_val_size = int(0.15 * total_piece_size)
    piece_test_size = total_piece_size - piece_train_size - piece_val_size
    
    piece_train_dataset, piece_val_dataset, piece_test_dataset = random_split(
        piece_dataset, [piece_train_size, piece_val_size, piece_test_size]
    )
    
    # Apply appropriate transforms
    piece_train_dataset.dataset.transform = train_transform
    piece_val_dataset.dataset.transform = val_transform
    piece_test_dataset.dataset.transform = val_transform
    
    logger.info(f"Piece type dataset split: Train={len(piece_train_dataset)}, Val={len(piece_val_dataset)}, Test={len(piece_test_dataset)}")
    
    # Create data loaders
    color_train_loader = DataLoader(color_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    color_val_loader = DataLoader(color_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    color_test_loader = DataLoader(color_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    piece_train_loader = DataLoader(piece_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    piece_val_loader = DataLoader(piece_val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    piece_test_loader = DataLoader(piece_test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Train Color Classifier
    logger.info("=" * 50)
    logger.info("TRAINING COLOR CLASSIFIER")
    logger.info("=" * 50)
    
    color_model = ColorClassifier(num_classes=2, dropout_rate=0.5)
    color_model, color_train_losses, color_val_losses, color_train_accs, color_val_accs = train_model(
        color_model, color_train_loader, color_val_loader, num_epochs, learning_rate, patience
    )
    
    # Evaluate color classifier
    logger.info("Evaluating Color Classifier...")
    color_accuracy, color_report, color_cm = evaluate_model(
        color_model, color_test_loader, ['Black', 'White']
    )
    
    # Train Piece Type Classifier
    logger.info("=" * 50)
    logger.info("TRAINING PIECE TYPE CLASSIFIER")
    logger.info("=" * 50)
    
    piece_model = PieceTypeClassifier(num_classes=6, dropout_rate=0.5)
    piece_model, piece_train_losses, piece_val_losses, piece_train_accs, piece_val_accs = train_model(
        piece_model, piece_train_loader, piece_val_loader, num_epochs, learning_rate, patience
    )
    
    # Evaluate piece type classifier
    logger.info("Evaluating Piece Type Classifier...")
    piece_accuracy, piece_report, piece_cm = evaluate_model(
        piece_model, piece_test_loader, ['Pawn', 'Rook', 'Knight', 'Bishop', 'Queen', 'King']
    )
    
    # Save final models
    torch.save(color_model.state_dict(), 'improved_color_classifier.pth')
    torch.save(piece_model.state_dict(), 'improved_piece_type_classifier.pth')
    
    # Summary
    logger.info("=" * 50)
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Color Classifier - Test Accuracy: {color_accuracy:.2f}%")
    logger.info(f"Piece Type Classifier - Test Accuracy: {piece_accuracy:.2f}%")
    logger.info("Models saved as:")
    logger.info("  - improved_color_classifier.pth")
    logger.info("  - improved_piece_type_classifier.pth")
    logger.info("  - best_ColorClassifier.pth (best validation performance)")
    logger.info("  - best_PieceTypeClassifier.pth (best validation performance)")
    
    # Check for overfitting
    logger.info("=" * 50)
    logger.info("OVERFITTING ANALYSIS")
    logger.info("=" * 50)
    
    # Color classifier overfitting check
    color_overfit = color_train_accs[-1] - color_val_accs[-1]
    logger.info(f"Color Classifier - Training vs Validation accuracy gap: {color_overfit:.2f}%")
    if color_overfit > 10:
        logger.warning("⚠️  Color classifier shows signs of overfitting!")
    else:
        logger.info("✅ Color classifier shows good generalization")
    
    # Piece type classifier overfitting check
    piece_overfit = piece_train_accs[-1] - piece_val_accs[-1]
    logger.info(f"Piece Type Classifier - Training vs Validation accuracy gap: {piece_overfit:.2f}%")
    if piece_overfit > 10:
        logger.warning("⚠️  Piece type classifier shows signs of overfitting!")
    else:
        logger.info("✅ Piece type classifier shows good generalization")

if __name__ == "__main__":
    main()
