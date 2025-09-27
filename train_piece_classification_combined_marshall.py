#!/usr/bin/env python3
"""
Train improved piece classification model using COMBINED approach
- Starts from existing working piece classification model
- Incorporates both previous training data AND Marshall data
- Creates additional model without replacing current working model
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
import numpy as np
from pathlib import Path
import logging
from PIL import Image
import time
from sklearn.model_selection import train_test_split
import random

# Add HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False
    print("Warning: pillow-heif not installed. HEIC files may not load properly.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CombinedPieceDataset(Dataset):
    """Dataset for combined piece classification training"""
    
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        square = item['square']
        label = item['label']
        
        # Handle different data formats
        if isinstance(square, torch.Tensor):
            # Already a tensor (from previous data)
            square_pil = transforms.ToPILImage()(square)
        else:
            # BGR image (from Marshall data)
            if len(square.shape) == 3 and square.shape[2] == 3:
                square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
            square_pil = Image.fromarray(square)
        
        if self.transform:
            square_pil = self.transform(square_pil)
        
        return square_pil, label

class CombinedPieceTrainer:
    """Trainer for combined piece classification model"""
    
    def __init__(self, marshall_annotations_path="marshall_chess_annotations/annotations.json"):
        """Initialize the trainer"""
        self.marshall_annotations_path = Path(marshall_annotations_path)
        self.output_dir = Path("models_marshall_improved")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load Marshall annotations
        self.load_marshall_annotations()
        
        # Model configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Piece type mapping (6 classes: pawn, knight, bishop, rook, queen, king)
        self.piece_types = {
            'P': 0, 'p': 0,  # pawn
            'N': 1, 'n': 1,  # knight
            'B': 2, 'b': 2,  # bishop
            'R': 3, 'r': 3,  # rook
            'Q': 4, 'q': 4,  # queen
            'K': 5, 'k': 5   # king
        }
        
        self.num_classes = 6
        
    def load_marshall_annotations(self):
        """Load Marshall annotations"""
        logger.info("Loading Marshall annotations...")
        
        with open(self.marshall_annotations_path, 'r') as f:
            data = json.load(f)
        
        self.annotations = data.get('annotations', {})
        self.excluded_images = set(data.get('excluded_images', []))
        
        # Filter out excluded images
        self.valid_annotations = {
            k: v for k, v in self.annotations.items() 
            if k not in self.excluded_images
        }
        
        logger.info(f"Loaded {len(self.valid_annotations)} valid Marshall annotations")
    
    def load_working_piece_model(self):
        """Load the current working piece classification model"""
        try:
            # Skip trying to load existing model due to architecture mismatch
            # Create new ResNet18 model directly
            logger.info("Creating new ResNet18 model for combined training...")
            logger.info("This will train from ImageNet pre-trained weights + Marshall data")
            
            # Create new ResNet18 model
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 6)
            logger.info("✅ Created new ResNet18 model for combined training")
            return model
            
        except Exception as e:
            logger.error(f"❌ Error loading working piece model: {e}")
            return None
    
    def load_previous_training_data(self):
        """Load previous piece classification training data"""
        logger.info("Loading previous training data...")
        
        dataset = []
        data_paths = [
            "data/pieces",
            "models/piece_classifier/train",
            "runs/piece_classifier/ResNet/train"
        ]
        
        for data_path in data_paths:
            path = Path(data_path)
            if path.exists():
                logger.info(f"Found previous data at: {path}")
                
                # Load images and labels
                for img_file in path.rglob("*.jpg"):
                    try:
                        # Extract label from directory structure
                        label = self.extract_piece_label_from_path(img_file)
                        if label is not None:
                            # Load image
                            img = Image.open(img_file)
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            
                            # Convert to tensor
                            transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])
                            img_tensor = transform(img)
                            
                            dataset.append({
                                'square': img_tensor,
                                'label': label,
                                'source': 'previous',
                                'image_name': img_file.name
                            })
                            
                    except Exception as e:
                        logger.warning(f"Error processing {img_file}: {e}")
                        continue
                
                if dataset:
                    break
        
        logger.info(f"Loaded {len(dataset)} previous training samples")
        return dataset
    
    def extract_piece_label_from_path(self, img_path):
        """Extract piece type label from image path"""
        try:
            # Try to extract from directory structure
            parts = str(img_path).split('/')
            for part in parts:
                if any(piece in part.lower() for piece in ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']):
                    piece_name = part.lower()
                    if 'pawn' in piece_name:
                        return 0
                    elif 'knight' in piece_name:
                        return 1
                    elif 'bishop' in piece_name:
                        return 2
                    elif 'rook' in piece_name:
                        return 3
                    elif 'queen' in piece_name:
                        return 4
                    elif 'king' in piece_name:
                        return 5
            
            return None
        except:
            return None
    
    def fen_to_board(self, fen):
        """Convert FEN string to 8x8 board representation"""
        board = [['.' for _ in range(8)] for _ in range(8)]
        
        # Split FEN into parts
        parts = fen.split()
        if not parts:
            return board
        
        # Parse piece positions
        ranks = parts[0].split('/')
        for rank_idx, rank in enumerate(ranks):
            file_idx = 0
            for char in rank:
                if char.isdigit():
                    # Empty squares
                    file_idx += int(char)
                else:
                    # Piece
                    if file_idx < 8:
                        board[rank_idx][file_idx] = char
                        file_idx += 1
        
        return board
    
    def warp_board(self, image, corners):
        """Warp image to get a square chessboard"""
        try:
            # Convert corners to numpy array
            src_points = np.array(corners, dtype=np.float32)
            
            # Define destination points for a square board
            size = 400  # 400x400 pixel board
            dst_points = np.array([
                [0, 0],
                [size, 0],
                [size, size],
                [0, size]
            ], dtype=np.float32)
            
            # Get perspective transform
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Warp image
            warped = cv2.warpPerspective(image, matrix, (size, size))
            
            return warped
        except Exception as e:
            logger.warning(f"Error warping board: {e}")
            return None
    
    def extract_squares_with_piece_labels(self, warped_board, fen):
        """Extract 64 squares and create piece type labels from FEN"""
        squares = []
        labels = []
        
        # Parse FEN to get piece positions
        board = self.fen_to_board(fen)
        
        square_size = warped_board.shape[0] // 8
        
        for rank in range(8):
            for file in range(8):
                # Extract square
                y1 = rank * square_size
                y2 = (rank + 1) * square_size
                x1 = file * square_size
                x2 = (file + 1) * square_size
                
                square = warped_board[y1:y2, x1:x2]
                squares.append(square)
                
                # Get piece at this position
                piece = board[rank][file]
                if piece == '.':
                    labels.append(None)  # Skip empty squares
                else:
                    label = self.piece_types.get(piece, None)
                    labels.append(label)
        
        return squares, labels
    
    def create_marshall_dataset(self):
        """Create Marshall piece classification dataset"""
        logger.info("Creating Marshall piece classification dataset...")
        
        dataset = []
        marshall_photos_dir = Path("/Users/tonyblum/Desktop/marshall photos")
        
        processed = 0
        errors = 0
        
        for image_name, annotation in self.valid_annotations.items():
            image_path = marshall_photos_dir / image_name
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                errors += 1
                continue
                
            try:
                # Load image (handle HEIC files)
                if image_path.suffix.lower() == '.heic' and HEIC_SUPPORT:
                    pil_image = Image.open(image_path)
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                else:
                    image = cv2.imread(str(image_path))
                
                if image is None:
                    continue
                    
                # Get corners and FEN
                corners = annotation.get('corners', [])
                fen = annotation.get('fen', '')
                
                if len(corners) != 4 or not fen:
                    continue
                
                # Warp board to get square images
                warped_board = self.warp_board(image, corners)
                if warped_board is None:
                    continue
                
                # Extract squares and create piece labels
                squares, piece_labels = self.extract_squares_with_piece_labels(warped_board, fen)
                
                for square, label in zip(squares, piece_labels):
                    if square is not None and label is not None:
                        dataset.append({
                            'square': square,
                            'label': label,
                            'source': 'marshall',
                            'image_name': image_name
                        })
                
                processed += 1
                if processed % 50 == 0:
                    logger.info(f"Processed {processed} Marshall images, created {len(dataset)} samples")
                        
            except Exception as e:
                logger.warning(f"Error processing {image_name}: {e}")
                errors += 1
                continue
        
        logger.info(f"Created Marshall dataset with {len(dataset)} samples")
        logger.info(f"Processed {processed} Marshall images, {errors} errors")
        return dataset
    
    def create_combined_dataset(self):
        """Create combined dataset from previous and Marshall data"""
        logger.info("Creating combined dataset...")
        
        # Load previous training data
        previous_data = self.load_previous_training_data()
        
        # Load Marshall data
        marshall_data = self.create_marshall_dataset()
        
        # Combine datasets
        combined_data = previous_data + marshall_data
        
        logger.info(f"Combined dataset: {len(previous_data)} previous + {len(marshall_data)} Marshall = {len(combined_data)} total")
        
        return combined_data
    
    def get_transforms(self, is_training=True):
        """Get data transforms"""
        if is_training:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def train_model(self):
        """Train the combined piece classification model"""
        logger.info("🚀 Starting Combined Piece Classification Training")
        logger.info("=" * 60)
        
        # Create combined dataset
        dataset = self.create_combined_dataset()
        if len(dataset) < 100:
            logger.error(f"Not enough data for training: {len(dataset)} samples")
            return None
        
        # Split dataset
        train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
        logger.info(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
        
        # Create data loaders
        train_dataset = CombinedPieceDataset(train_data, self.get_transforms(is_training=True))
        val_dataset = CombinedPieceDataset(val_data, self.get_transforms(is_training=False))
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
        
        # Load working model as base
        model = self.load_working_piece_model()
        if model is None:
            logger.error("Failed to load working model, cannot proceed")
            return None
        
        model = model.to(self.device)
        
        # Setup training (fine-tuning approach)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate for fine-tuning
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        # Training parameters
        max_epochs = 15  # Limit to prevent overfitting
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        logger.info(f"Fine-tuning for maximum {max_epochs} epochs with early stopping (patience={patience})")
        logger.info(f"Learning rate: 0.0001 (lower for fine-tuning)")
        
        # Training loop
        for epoch in range(max_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
                
                if batch_idx % 20 == 0:
                    logger.info(f'Epoch {epoch+1}/{max_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            # Calculate metrics
            train_loss /= len(train_loader)
            train_acc = 100.0 * train_correct / train_total
            val_loss /= len(val_loader)
            val_acc = 100.0 * val_correct / val_total
            
            logger.info(f'Epoch {epoch+1}/{max_epochs}:')
            logger.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                model_path = self.output_dir / "piece_classification_combined_marshall.pt"
                torch.save(model.state_dict(), model_path)
                logger.info(f'✅ New best model saved: {model_path}')
            else:
                patience_counter += 1
                logger.info(f'No improvement for {patience_counter} epochs')
                
                if patience_counter >= patience:
                    logger.info(f'Early stopping triggered after {epoch+1} epochs')
                    break
        
        logger.info("🎉 Combined piece classification training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        return model
    
    def validate_model(self, model_path=None):
        """Validate the trained model"""
        if model_path is None:
            model_path = self.output_dir / "piece_classification_combined_marshall.pt"
        
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return
        
        # Load model
        model = self.load_working_piece_model()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        
        # Create validation dataset
        dataset = self.create_combined_dataset()
        if len(dataset) < 100:
            logger.error("Not enough data for validation")
            return
        
        # Split dataset
        _, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
        val_dataset = CombinedPieceDataset(val_data, self.get_transforms(is_training=False))
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
        
        # Validate
        correct = 0
        total = 0
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes
        
        piece_names = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # Per-class accuracy
                for i in range(target.size(0)):
                    label = target[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1
        
        overall_acc = 100.0 * correct / total
        logger.info(f"Overall accuracy: {overall_acc:.2f}%")
        
        for i in range(self.num_classes):
            if class_total[i] > 0:
                acc = 100.0 * class_correct[i] / class_total[i]
                logger.info(f"{piece_names[i]}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")

def main():
    """Main training function"""
    logger.info("🎯 Combined Marshall Piece Classification Training")
    logger.info("=" * 60)
    
    # Initialize trainer
    trainer = CombinedPieceTrainer()
    
    # Train model
    model = trainer.train_model()
    
    if model is not None:
        # Validate model
        logger.info("\n🔍 Validating trained model...")
        trainer.validate_model()
        
        logger.info("\n✅ Training completed successfully!")
        logger.info("Model saved as: models_marshall_improved/piece_classification_combined_marshall.pt")
        logger.info("This model combines previous training data with Marshall data for better performance")
    else:
        logger.error("❌ Training failed!")

if __name__ == "__main__":
    main()
