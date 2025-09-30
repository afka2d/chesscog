#!/usr/bin/env python3
"""
Train Marshall occupancy model using the SAME architecture as the original model.
This ensures compatibility with the existing API.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
import json
import logging
from PIL import Image
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class MarshallOccupancyDataset(Dataset):
    def __init__(self, annotations, image_dir, transform=None):
        self.annotations = annotations
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.valid_samples = []
        
        # Filter valid annotations
        for image_name, data in annotations.items():
            if 'corners' in data and 'fen' in data:
                image_path = self.image_dir / image_name
                if image_path.exists():
                    self.valid_samples.append((image_name, data))
        
        logger.info(f"Loaded {len(self.valid_samples)} valid samples")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        image_name, data = self.valid_samples[idx]
        
        # Load image
        image_path = self.image_dir / image_name
        image = cv2.imread(str(image_path))
        if image is None:
            # Try HEIC format
            try:
                from pillow_heif import register_heif_opener
                register_heif_opener()
                image = Image.open(image_path)
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            except:
                logger.warning(f"Could not load image: {image_path}")
                return None
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get corners and FEN
        corners = data['corners']
        fen = data['fen']
        
        # Warp board
        warped = self.warp_board(image, corners)
        if warped is None:
            return None
        
        # Extract squares and labels
        squares, labels = self.extract_squares_with_labels(warped, fen)
        
        # Convert to tensors
        square_tensors = []
        label_tensors = []
        
        for square, label in zip(squares, labels):
            if label is not None:  # Only occupied squares
                if self.transform:
                    square_tensor = self.transform(Image.fromarray(square))
                else:
                    square_tensor = torch.from_numpy(square).permute(2, 0, 1).float() / 255.0
                
                square_tensors.append(square_tensor)
                label_tensors.append(torch.tensor(label, dtype=torch.long))
        
        if not square_tensors:
            return None
        
        return {
            'squares': torch.stack(square_tensors),
            'labels': torch.stack(label_tensors)
        }
    
    def warp_board(self, image, corners):
        """Warp image to get a square chessboard"""
        try:
            src_points = np.array(corners, dtype=np.float32)
            size = 400
            dst_points = np.array([
                [0, 0], [size, 0], [size, size], [0, size]
            ], dtype=np.float32)
            
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            warped = cv2.warpPerspective(image, matrix, (size, size))
            return warped
        except Exception as e:
            logger.warning(f"Error warping board: {e}")
            return None
    
    def extract_squares_with_labels(self, warped_board, fen):
        """Extract 64 squares and create occupancy labels from FEN"""
        squares = []
        labels = []
        
        # Parse FEN to get piece positions
        board = self.fen_to_board(fen)
        
        square_size = warped_board.shape[0] // 8
        
        for rank in range(8):
            for file in range(8):
                y1 = rank * square_size
                y2 = (rank + 1) * square_size
                x1 = file * square_size
                x2 = (file + 1) * square_size
                
                square = warped_board[y1:y2, x1:x2]
                squares.append(square)
                
                piece = board[rank][file]
                if piece == '.':
                    labels.append(0)  # Empty
                else:
                    labels.append(1)  # Occupied
        
        return squares, labels
    
    def fen_to_board(self, fen):
        """Convert FEN string to board representation"""
        board = []
        rows = fen.split('/')
        for row in rows:
            board_row = []
            for char in row:
                if char.isdigit():
                    board_row.extend(['.'] * int(char))
                else:
                    board_row.append(char)
            board.append(board_row)
        return board

def load_original_occupancy_model():
    """Load the original occupancy model architecture"""
    try:
        original_path = Path("runs/occupancy_classifier/ResNet/ResNet.pt")
        if not original_path.exists():
            logger.error(f"Original occupancy model not found at {original_path}")
            return None
        
        model = torch.load(str(original_path), map_location='cpu', weights_only=False)
        logger.info("✅ Original occupancy model architecture loaded")
        return model
    except Exception as e:
        logger.error(f"Error loading original model: {e}")
        return None

def train_occupancy_model():
    """Train the Marshall occupancy model"""
    logger.info("🚀 Starting Marshall occupancy model training with correct architecture")
    
    # Load annotations
    annotations_path = Path("marshall_chess_annotations/annotations.json")
    if not annotations_path.exists():
        logger.error(f"Annotations not found at {annotations_path}")
        return
    
    with open(annotations_path, 'r') as f:
        data = json.load(f)
        annotations = data.get('annotations', data)
    
    logger.info(f"Loaded {len(annotations)} annotations")
    
    # Load original model architecture
    original_model = load_original_occupancy_model()
    if original_model is None:
        logger.error("Failed to load original model architecture")
        return
    
    # Create dataset
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = MarshallOccupancyDataset(annotations, "marshall_chess_annotations", transform)
    if len(dataset) == 0:
        logger.error("No valid samples found")
        return
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # Use the original model architecture
    model = original_model
    model.train()
    
    # Set up training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    num_epochs = 20
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            if batch is None:
                continue
            
            squares = batch['squares']
            labels = batch['labels']
            
            # Flatten for batch processing
            squares = squares.view(-1, squares.shape[-3], squares.shape[-2], squares.shape[-1])
            labels = labels.view(-1)
            
            optimizer.zero_grad()
            outputs = model(squares)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                
                # Save best model
                output_dir = Path("models_marshall_improved")
                output_dir.mkdir(exist_ok=True)
                torch.save(model.state_dict(), output_dir / "occupancy_marshall_correct_architecture.pt")
                logger.info(f"✅ Saved best model (loss: {best_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
    
    logger.info("🎉 Training completed!")
    logger.info(f"Best loss: {best_loss:.4f}")

if __name__ == "__main__":
    train_occupancy_model()
