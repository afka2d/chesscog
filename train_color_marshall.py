#!/usr/bin/env python3
"""
Train Marshall Improved Color Classification Model
Fine-tune existing working color model with Marshall data
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

# Add HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False
    print("Warning: pillow-heif not installed. HEIC files may not load properly.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ColorDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def preprocess_square_for_color(square):
    """Preprocess square for color classification"""
    # Resize to 224x224
    square = cv2.resize(square, (224, 224))
    # Convert BGR to RGB
    square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
    # Normalize
    square = square.astype(np.float32) / 255.0
    # Convert to tensor
    square = torch.from_numpy(square).permute(2, 0, 1)
    return square

def warp_board(image, corners):
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

def extract_square(warped_board, rank, file):
    """Extract a single square from the warped board"""
    square_size = warped_board.shape[0] // 8
    
    y1 = rank * square_size
    y2 = (rank + 1) * square_size
    x1 = file * square_size
    x2 = (file + 1) * square_size
    
    # Extract square
    square = warped_board[y1:y2, x1:x2]
    
    return square

def fen_to_board(fen):
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

def create_color_dataset():
    """Create dataset for color classification training"""
    logger.info("Creating color classification dataset...")
    
    # Load Marshall annotations
    with open("marshall_chess_annotations/annotations.json", 'r') as f:
        data = json.load(f)
    
    annotations = data.get('annotations', {})
    excluded_images = set(data.get('excluded_images', []))
    
    # Filter out excluded images
    valid_annotations = {
        k: v for k, v in annotations.items() 
        if k not in excluded_images
    }
    
    logger.info(f"Valid annotations for training: {len(valid_annotations)}")
    
    dataset = []
    marshall_photos_dir = Path("/Users/tonyblum/Desktop/marshall photos")
    
    processed = 0
    errors = 0
    
    for image_name, annotation in valid_annotations.items():
        image_path = marshall_photos_dir / image_name
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            errors += 1
            continue
            
        try:
            # Load image (handle HEIC files)
            if image_path.suffix.lower() == '.heic' and HEIC_SUPPORT:
                # Load HEIC with PIL and convert to OpenCV format
                pil_image = Image.open(image_path)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            else:
                image = cv2.imread(str(image_path))
            
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                errors += 1
                continue
                
            # Get corners and FEN
            corners = annotation.get('corners', [])
            fen = annotation.get('fen', '')
            
            if len(corners) != 4 or not fen:
                logger.warning(f"Invalid data for {image_name}")
                errors += 1
                continue
            
            # Warp board to get square images
            warped_board = warp_board(image, corners)
            if warped_board is None:
                logger.warning(f"Could not warp board for {image_name}")
                errors += 1
                continue
            
            # Extract squares and create labels
            squares, labels = extract_squares_with_color_labels(warped_board, fen)
            
            for square, label in zip(squares, labels):
                if square is not None and label is not None:
                    # Preprocess square
                    square_tensor = preprocess_square_for_color(square)
                    dataset.append({
                        'square': square_tensor,
                        'label': torch.tensor(label, dtype=torch.long),
                        'image_name': image_name
                    })
                    
        except Exception as e:
            logger.warning(f"Error processing {image_name}: {e}")
            errors += 1
            continue
        
        processed += 1
        if processed % 50 == 0:
            logger.info(f"Processed {processed} images...")
    
    logger.info(f"Created color classification dataset with {len(dataset)} samples")
    logger.info(f"Processed: {processed}, Errors: {errors}")
    return dataset

def extract_squares_with_color_labels(warped_board, fen):
    """Extract 64 squares and create color labels from FEN"""
    squares = []
    labels = []
    
    # Parse FEN to get piece positions
    board = fen_to_board(fen)
    
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
                # 0 for white (uppercase), 1 for black (lowercase)
                label = 0 if piece.isupper() else 1
                labels.append(label)
    
    return squares, labels

def load_working_color_model():
    """Load the existing working color model"""
    logger.info("Loading working color model...")
    
    # Load the working model architecture
    def _get_color_model_architecture(num_classes):
        model = models.mobilenet_v2(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        return model
    
    # Create model with same architecture
    model = _get_color_model_architecture(2)  # 2 classes: white/black
    
    # Load weights from working model
    model_path = Path("models/color_classifier_simple.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"Working color model not found at {model_path}")
    
    model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
    model.eval()
    
    logger.info("✅ Working color model loaded successfully")
    return model

def train_color_model():
    """Train improved color classification model"""
    logger.info("🚀 Starting Marshall Color Classification Model Training")
    logger.info("=" * 60)
    
    # Create dataset
    dataset = create_color_dataset()
    if len(dataset) < 100:
        logger.error("Not enough data for color classification training")
        return None
    
    # Split dataset
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    # Load working model as base
    model = load_working_color_model()
    
    # Set up training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Use a lower learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower LR for fine-tuning
    criterion = nn.CrossEntropyLoss()
    
    # Training loop with early stopping
    best_val_acc = 0
    patience = 5  # Reduced patience for faster training
    patience_counter = 0
    
    logger.info(f"Training on {len(train_data)} samples, validating on {len(val_data)} samples")
    logger.info(f"Using device: {device}")
    
    for epoch in range(20):  # Reduced epochs for faster training
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            squares = batch['square']
            labels = batch['label']
            
            squares = squares.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(squares)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                squares = batch['square']
                labels = batch['label']
                
                squares = squares.to(device)
                labels = labels.to(device)
                
                outputs = model(squares)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            output_dir = Path("models_marshall_improved")
            output_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), output_dir / "color_classification_marshall.pt")
            logger.info(f"💾 New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break
    
    logger.info("🎉 Color classification training completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    return model

if __name__ == "__main__":
    train_color_model()

