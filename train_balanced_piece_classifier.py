#!/usr/bin/env python3
"""
Balanced Combined Piece Classifier Training Script

This script trains a piece classification model on BOTH Marshall and Grey background
datasets with proper balancing to prevent catastrophic forgetting.

Key Features:
1. Uses EfficientNet-B0 (proven architecture)
2. Balances batches with equal Marshall and Grey samples
3. Validates on BOTH datasets every epoch
4. Early stopping if either dataset degrades
5. Saves best model based on combined performance
6. Runs overnight without manual intervention
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from pathlib import Path
import cv2
import numpy as np
import json
import logging
from PIL import Image
import pillow_heif
from datetime import datetime
import sys

# Register HEIF support
pillow_heif.register_heif_opener()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'balanced_piece_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Piece labels
PIECE_LABELS = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']
PIECE_TO_IDX = {p: i for i, p in enumerate(PIECE_LABELS)}


class BalancedPieceDataset(Dataset):
    """Dataset that balances Marshall and Grey background data"""
    
    def __init__(self, marshall_data, grey_data, transform=None):
        """
        Args:
            marshall_data: List of (image, label, source='marshall') tuples
            grey_data: List of (image, label, source='grey') tuples
            transform: Torchvision transforms to apply
        """
        # Combine datasets with source tags
        self.data = []
        
        # Add Marshall data
        for img, label in marshall_data:
            self.data.append((img, label, 'marshall'))
        
        # Add Grey data
        for img, label in grey_data:
            self.data.append((img, label, 'grey'))
        
        self.transform = transform
        
        # Calculate weights for balanced sampling
        n_marshall = len(marshall_data)
        n_grey = len(grey_data)
        total = n_marshall + n_grey
        
        # Weight inversely proportional to dataset size
        marshall_weight = total / (2 * n_marshall) if n_marshall > 0 else 0
        grey_weight = total / (2 * n_grey) if n_grey > 0 else 0
        
        self.sample_weights = []
        for _, _, source in self.data:
            if source == 'marshall':
                self.sample_weights.append(marshall_weight)
            else:
                self.sample_weights.append(grey_weight)
        
        logger.info(f"Dataset created: {len(marshall_data)} Marshall + {len(grey_data)} Grey = {len(self.data)} total")
        logger.info(f"Sampling weights: Marshall={marshall_weight:.4f}, Grey={grey_weight:.4f}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label, source = self.data[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def load_grey_background_pieces(split='train'):
    """Load pre-extracted grey background piece images"""
    logger.info(f"Loading grey background {split} dataset...")
    
    pieces_dir = Path(f'grey_background_dataset/pieces/{split}')
    if not pieces_dir.exists():
        logger.warning(f"Grey background {split} directory not found: {pieces_dir}")
        return []
    
    data = []
    
    for piece_type in PIECE_LABELS:
        for color in ['black', 'white']:
            piece_dir = pieces_dir / f'{color}_{piece_type}'
            if not piece_dir.exists():
                continue
            
            # Load all PNG and JPG files
            for img_path in list(piece_dir.glob('*.png')) + list(piece_dir.glob('*.jpg')):
                try:
                    # Load image
                    img = Image.open(img_path).convert('RGB')
                    
                    # Get label (piece type only, not color)
                    label = PIECE_TO_IDX[piece_type]
                    
                    data.append((img, label))
                    
                except Exception as e:
                    logger.warning(f"Error loading {img_path}: {e}")
                    continue
    
    logger.info(f"Loaded {len(data)} grey background {split} samples")
    return data


def load_marshall_pieces():
    """Extract piece images from Marshall board images"""
    logger.info("Extracting pieces from Marshall board images...")
    
    # Load annotations
    annotations_path = Path('marshall_chess_annotations/annotations.json')
    if not annotations_path.exists():
        logger.error(f"Marshall annotations not found: {annotations_path}")
        return []
    
    with open(annotations_path) as f:
        data = json.load(f)
        annotations = data.get('annotations', data)
    
    # Marshall photos directory
    marshall_photos_dir = Path("/Users/tonyblum/Desktop/marshall photos")
    if not marshall_photos_dir.exists():
        logger.error(f"Marshall photos directory not found: {marshall_photos_dir}")
        return []
    
    data = []
    processed = 0
    errors = 0
    
    for image_name, annotation in annotations.items():
        if processed % 50 == 0 and processed > 0:
            logger.info(f"Processed {processed}/{len(annotations)} images, extracted {len(data)} pieces")
        
        image_path = marshall_photos_dir / image_name
        if not image_path.exists():
            errors += 1
            continue
        
        try:
            # Load HEIC image
            if image_path.suffix.lower() == '.heic':
                pil_image = Image.open(image_path)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            else:
                image = cv2.imread(str(image_path))
            
            if image is None:
                errors += 1
                continue
            
            # Get corners and FEN
            corners = annotation.get('corners', [])
            fen = annotation.get('fen', '')
            
            if len(corners) != 4 or not fen:
                continue
            
            # Warp board
            corners_array = np.array(corners, dtype=np.float32)
            size = 800
            dst_points = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(corners_array, dst_points)
            warped = cv2.warpPerspective(image, M, (size, size))
            
            # Extract pieces from FEN
            board_fen = fen.split()[0]  # Get just the board part
            ranks = board_fen.split('/')
            
            square_size = 100
            for rank_idx, rank_str in enumerate(ranks):
                file_idx = 0
                for char in rank_str:
                    if char.isdigit():
                        file_idx += int(char)
                    elif char.isalpha():
                        # Extract square
                        y = rank_idx * square_size
                        x = file_idx * square_size
                        square = warped[y:y+square_size, x:x+square_size]
                        
                        # Convert to PIL RGB
                        square_rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
                        square_pil = Image.fromarray(square_rgb)
                        
                        # Get piece type label
                        piece_char = char.lower()
                        piece_map = {'p': 'pawn', 'n': 'knight', 'b': 'bishop',
                                   'r': 'rook', 'q': 'queen', 'k': 'king'}
                        
                        if piece_char in piece_map:
                            piece_type = piece_map[piece_char]
                            label = PIECE_TO_IDX[piece_type]
                            data.append((square_pil, label))
                        
                        file_idx += 1
            
            processed += 1
            
        except Exception as e:
            logger.warning(f"Error processing {image_name}: {e}")
            errors += 1
            continue
    
    logger.info(f"Processed {processed} Marshall images, extracted {len(data)} pieces, {errors} errors")
    return data


def create_model(num_classes=6):
    """Create EfficientNet-B0 model (same as original working model)"""
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')  # Start with ImageNet pretrained
    
    # Replace classifier
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    return model


def validate_model(model, dataloader, device, dataset_name=""):
    """Validate model on a dataset"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total if total > 0 else 0
    logger.info(f"{dataset_name} Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy


def train_balanced_piece_classifier():
    """Train piece classifier with balanced data from both datasets"""
    
    logger.info("="*80)
    logger.info("BALANCED PIECE CLASSIFIER TRAINING")
    logger.info("="*80)
    logger.info("")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading training data...")
    grey_train_data = load_grey_background_pieces('train')
    grey_val_data = load_grey_background_pieces('val')
    marshall_data = load_marshall_pieces()
    
    if len(grey_train_data) == 0:
        logger.error("❌ No grey background training data found!")
        return False
    
    if len(marshall_data) == 0:
        logger.error("❌ No Marshall data found!")
        return False
    
    # Create validation splits for Marshall (use 10% for validation)
    np.random.seed(42)
    marshall_indices = np.random.permutation(len(marshall_data))
    val_size = len(marshall_data) // 10
    marshall_val_data = [marshall_data[i] for i in marshall_indices[:val_size]]
    marshall_train_data = [marshall_data[i] for i in marshall_indices[val_size:]]
    
    logger.info(f"")
    logger.info(f"Data split:")
    logger.info(f"  Grey Train: {len(grey_train_data)}")
    logger.info(f"  Grey Val: {len(grey_val_data)}")
    logger.info(f"  Marshall Train: {len(marshall_train_data)}")
    logger.info(f"  Marshall Val: {len(marshall_val_data)}")
    
    # Transforms (matching original piece classifier)
    train_transform = transforms.Compose([
        transforms.Resize(100),  # Original uses 100x100
        transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(100),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create balanced datasets
    train_dataset = BalancedPieceDataset(marshall_train_data, grey_train_data, transform=train_transform)
    
    # Create separate validation datasets for each source
    grey_val_dataset = BalancedPieceDataset([], grey_val_data, transform=val_transform)
    marshall_val_dataset = BalancedPieceDataset(marshall_val_data, [], transform=val_transform)
    
    # Create weighted sampler for balanced training
    sampler = WeightedRandomSampler(
        weights=train_dataset.sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=4)
    grey_val_loader = DataLoader(grey_val_dataset, batch_size=32, shuffle=False, num_workers=2)
    marshall_val_loader = DataLoader(marshall_val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Create model
    logger.info("")
    logger.info("Creating model...")
    model = create_model(num_classes=6)
    model = model.to(device)
    logger.info("✅ EfficientNet-B0 model created")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower LR for fine-tuning
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Training parameters
    max_epochs = 50
    patience = 7  # Early stopping patience
    best_combined_acc = 0
    best_grey_acc = 0
    epochs_without_improvement = 0
    min_grey_acc = 90.0  # Don't allow grey accuracy to drop below 90%
    
    logger.info("")
    logger.info("="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80)
    logger.info(f"Max epochs: {max_epochs}")
    logger.info(f"Early stopping patience: {patience}")
    logger.info(f"Minimum grey accuracy threshold: {min_grey_acc}%")
    logger.info("")
    
    # Training loop
    for epoch in range(max_epochs):
        logger.info(f"Epoch {epoch+1}/{max_epochs}")
        logger.info("-" * 40)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 50 == 0:
                batch_acc = 100 * correct / total
                logger.info(f"  Batch {i+1}/{len(train_loader)}: Loss={loss.item():.4f}, Acc={batch_acc:.2f}%")
        
        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        
        # Validation phase
        grey_val_acc = validate_model(model, grey_val_loader, device, "Grey Val")
        marshall_val_acc = validate_model(model, marshall_val_loader, device, "Marshall Val")
        combined_acc = (grey_val_acc + marshall_val_acc) / 2
        
        logger.info(f"")
        logger.info(f"Epoch {epoch+1} Results:")
        logger.info(f"  Train Loss: {avg_loss:.4f}")
        logger.info(f"  Train Acc: {train_acc:.2f}%")
        logger.info(f"  Grey Val Acc: {grey_val_acc:.2f}%")
        logger.info(f"  Marshall Val Acc: {marshall_val_acc:.2f}%")
        logger.info(f"  Combined Acc: {combined_acc:.2f}%")
        logger.info("")
        
        # Check for grey accuracy degradation
        if grey_val_acc < min_grey_acc:
            logger.warning(f"⚠️  Grey accuracy ({grey_val_acc:.2f}%) below minimum ({min_grey_acc}%)")
            logger.warning(f"   Stopping to prevent catastrophic forgetting!")
            break
        
        # Update learning rate
        scheduler.step(combined_acc)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"  Learning rate: {current_lr:.6f}")
        
        # Save best model
        if combined_acc > best_combined_acc:
            best_combined_acc = combined_acc
            best_grey_acc = grey_val_acc
            best_marshall_acc = marshall_val_acc
            epochs_without_improvement = 0
            
            # Save model
            output_dir = Path("models_marshall_improved")
            output_dir.mkdir(exist_ok=True)
            model_path = output_dir / "piece_classifier_balanced.pt"
            
            torch.save(model.state_dict(), str(model_path))
            
            logger.info(f"✅ New best model saved! Combined: {combined_acc:.2f}% (Grey: {grey_val_acc:.2f}%, Marshall: {marshall_val_acc:.2f}%)")
            logger.info(f"   Saved to: {model_path}")
        else:
            epochs_without_improvement += 1
            logger.info(f"  No improvement ({epochs_without_improvement}/{patience})")
        
        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(f"")
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        logger.info("")
    
    # Final summary
    logger.info("="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Best Combined Accuracy: {best_combined_acc:.2f}%")
    logger.info(f"  - Grey Val: {best_grey_acc:.2f}%")
    logger.info(f"  - Marshall Val: {best_marshall_acc:.2f}%")
    logger.info(f"Model saved to: models_marshall_improved/piece_classifier_balanced.pt")
    logger.info("="*80)
    
    return True


if __name__ == "__main__":
    try:
        logger.info("="*80)
        logger.info(f"BALANCED PIECE CLASSIFIER TRAINING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80)
        logger.info("")
        logger.info("This training will run overnight without manual intervention")
        logger.info("Using EfficientNet-B0 with balanced sampling from both datasets")
        logger.info("")
        
        success = train_balanced_piece_classifier()
        
        if success:
            logger.info("")
            logger.info("✅ Training completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Training failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Fatal error during training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
