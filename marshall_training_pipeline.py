#!/usr/bin/env python3
"""
Safe Marshall Training Pipeline
Creates improved models using Marshall data without affecting current working models
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
import shutil

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

class MarshallTrainingPipeline:
    def __init__(self, marshall_annotations_path="marshall_chess_annotations/annotations.json"):
        """Initialize the training pipeline with Marshall data"""
        self.marshall_annotations_path = Path(marshall_annotations_path)
        self.output_dir = Path("models_marshall_improved")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load Marshall annotations
        self.load_marshall_annotations()
        
        # Model configurations
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def load_marshall_annotations(self):
        """Load Marshall annotations and prepare training data"""
        logger.info("Loading Marshall annotations...")
        
        with open(self.marshall_annotations_path, 'r') as f:
            data = json.load(f)
        
        self.annotations = data.get('annotations', {})
        self.excluded_images = set(data.get('excluded_images', []))
        
        logger.info(f"Loaded {len(self.annotations)} Marshall annotations")
        logger.info(f"Excluded {len(self.excluded_images)} images")
        
        # Filter out excluded images
        self.valid_annotations = {
            k: v for k, v in self.annotations.items() 
            if k not in self.excluded_images
        }
        
        logger.info(f"Valid annotations for training: {len(self.valid_annotations)}")
    
    def create_corner_detection_dataset(self):
        """Create dataset for corner detection training"""
        logger.info("Creating corner detection dataset...")
        
        dataset = []
        marshall_photos_dir = Path("/Users/tonyblum/Desktop/marshall photos")
        
        logger.info(f"Looking for images in: {marshall_photos_dir}")
        logger.info(f"Total annotations to process: {len(self.valid_annotations)}")
        
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
                    
                # Get corners from annotation
                corners = annotation.get('corners', [])
                if len(corners) != 4:
                    logger.warning(f"Invalid corners for {image_name}: {len(corners)} corners")
                    errors += 1
                    continue
                    
                # Convert to numpy array and normalize
                corners_array = np.array(corners, dtype=np.float32)
                
                # Normalize corners to [0, 1] range
                h, w = image.shape[:2]
                corners_normalized = corners_array / np.array([w, h])
                
                # Flatten corners to [8] shape (4 corners × 2 coordinates)
                corners_flat = corners_normalized.flatten().astype(np.float32)
                
                # Preprocess image for corner detection
                processed_image = self.preprocess_image_for_corner_detection(image)
                
                dataset.append({
                    'image': processed_image,
                    'corners': corners_flat,
                    'image_name': image_name
                })
                
                processed += 1
                if processed % 50 == 0:
                    logger.info(f"Processed {processed} images...")
                
            except Exception as e:
                logger.warning(f"Error processing {image_name}: {e}")
                errors += 1
                continue
        
        logger.info(f"Created corner detection dataset with {len(dataset)} samples")
        logger.info(f"Processed: {processed}, Errors: {errors}")
        return dataset
    
    def create_occupancy_dataset(self):
        """Create dataset for occupancy detection training"""
        logger.info("Creating occupancy detection dataset...")
        
        dataset = []
        marshall_photos_dir = Path("/Users/tonyblum/Desktop/marshall photos")
        
        for image_name, annotation in self.valid_annotations.items():
            image_path = marshall_photos_dir / image_name
            if not image_path.exists():
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
                
                # Extract squares and create labels
                squares, labels = self.extract_squares_with_labels(warped_board, fen)
                
                for square, label in zip(squares, labels):
                    if square is not None:
                        dataset.append({
                            'square': square,
                            'label': label,
                            'image_name': image_name
                        })
                        
            except Exception as e:
                logger.warning(f"Error processing {image_name}: {e}")
                continue
        
        logger.info(f"Created occupancy dataset with {len(dataset)} samples")
        return dataset
    
    def create_color_classification_dataset(self):
        """Create dataset for color classification training"""
        logger.info("Creating color classification dataset...")
        
        dataset = []
        marshall_photos_dir = Path("/Users/tonyblum/Desktop/marshall photos")
        
        for image_name, annotation in self.valid_annotations.items():
            image_path = marshall_photos_dir / image_name
            if not image_path.exists():
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
                
                # Extract squares and create color labels
                squares, color_labels = self.extract_squares_with_color_labels(warped_board, fen)
                
                for square, label in zip(squares, color_labels):
                    if square is not None and label is not None:
                        dataset.append({
                            'square': square,
                            'label': label,
                            'image_name': image_name
                        })
                        
            except Exception as e:
                logger.warning(f"Error processing {image_name}: {e}")
                continue
        
        logger.info(f"Created color classification dataset with {len(dataset)} samples")
        return dataset
    
    def create_piece_classification_dataset(self):
        """Create dataset for piece classification training"""
        logger.info("Creating piece classification dataset...")
        
        dataset = []
        marshall_photos_dir = Path("/Users/tonyblum/Desktop/marshall photos")
        
        for image_name, annotation in self.valid_annotations.items():
            image_path = marshall_photos_dir / image_name
            if not image_path.exists():
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
                            'image_name': image_name
                        })
                        
            except Exception as e:
                logger.warning(f"Error processing {image_name}: {e}")
                continue
        
        logger.info(f"Created piece classification dataset with {len(dataset)} samples")
        return dataset
    
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
    
    def extract_squares_with_labels(self, warped_board, fen):
        """Extract 64 squares and create occupancy labels from FEN"""
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
                # 1 if occupied, 0 if empty
                label = 1 if piece != '.' else 0
                labels.append(label)
        
        return squares, labels
    
    def extract_squares_with_color_labels(self, warped_board, fen):
        """Extract 64 squares and create color labels from FEN"""
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
                    # 0 for white (uppercase), 1 for black (lowercase)
                    label = 0 if piece.isupper() else 1
                    labels.append(label)
        
        return squares, labels
    
    def extract_squares_with_piece_labels(self, warped_board, fen):
        """Extract 64 squares and create piece type labels from FEN"""
        squares = []
        labels = []
        
        # Parse FEN to get piece positions
        board = self.fen_to_board(fen)
        
        # Piece type mapping (excluding empty)
        piece_types = {
            'P': 0, 'p': 0,  # pawn
            'N': 1, 'n': 1,  # knight
            'B': 2, 'b': 2,  # bishop
            'R': 3, 'r': 3,  # rook
            'Q': 4, 'q': 4,  # queen
            'K': 5, 'k': 5   # king
        }
        
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
                    label = piece_types.get(piece, None)
                    labels.append(label)
        
        return squares, labels
    
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
    
    def train_corner_detection_model(self):
        """Train improved corner detection model"""
        logger.info("Training Marshall corner detection model...")
        
        # Create dataset
        dataset = self.create_corner_detection_dataset()
        if len(dataset) < 10:
            logger.error("Not enough data for corner detection training")
            return None
        
        # Split dataset
        train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
        
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
        
        # Create model
        model = CornerDetectionModel().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(100):
            # Training
            model.train()
            train_loss = 0
            for batch in train_loader:
                images = batch['image']
                corners = batch['corners']
                
                # Images are already preprocessed
                image_tensors = images.to(self.device)
                corners = corners.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(image_tensors)
                loss = criterion(outputs, corners)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image']
                    corners = batch['corners']
                    
                    # Images are already preprocessed
                    image_tensors = images.to(self.device)
                    corners = corners.to(self.device)
                    
                    outputs = model(image_tensors)
                    loss = criterion(outputs, corners)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), self.output_dir / "corner_detection_marshall.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered")
                    break
        
        logger.info("Corner detection training completed")
        return model
    
    def train_occupancy_model(self):
        """Train improved occupancy detection model"""
        logger.info("Training Marshall occupancy model...")
        
        # Create dataset
        dataset = self.create_occupancy_dataset()
        if len(dataset) < 100:
            logger.error("Not enough data for occupancy training")
            return None
        
        # Filter out None labels
        dataset = [d for d in dataset if d['label'] is not None]
        
        # Split dataset
        train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
        
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        
        # Create model
        model = OccupancyModel().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_acc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(50):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                squares = batch['square']
                labels = batch['label']
                
                # Convert squares to tensors
                square_tensors = []
                for square in squares:
                    square_tensor = self.preprocess_square_for_occupancy(square)
                    square_tensors.append(square_tensor)
                
                square_tensors = torch.stack(square_tensors).to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(square_tensors)
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
                    
                    square_tensors = []
                    for square in squares:
                        square_tensor = self.preprocess_square_for_occupancy(square)
                        square_tensors.append(square_tensor)
                    
                    square_tensors = torch.stack(square_tensors).to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = model(square_tensors)
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
                torch.save(model.state_dict(), self.output_dir / "occupancy_marshall.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered")
                    break
        
        logger.info("Occupancy training completed")
        return model
    
    def train_color_classification_model(self):
        """Train improved color classification model"""
        logger.info("Training Marshall color classification model...")
        
        # Create dataset
        dataset = self.create_color_classification_dataset()
        if len(dataset) < 100:
            logger.error("Not enough data for color classification training")
            return None
        
        # Filter out None labels
        dataset = [d for d in dataset if d['label'] is not None]
        
        # Split dataset
        train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
        
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        
        # Create model
        model = ColorModel().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_acc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(50):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                squares = batch['square']
                labels = batch['label']
                
                # Convert squares to tensors
                square_tensors = []
                for square in squares:
                    square_tensor = self.preprocess_square_for_color(square)
                    square_tensors.append(square_tensor)
                
                square_tensors = torch.stack(square_tensors).to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(square_tensors)
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
                    
                    square_tensors = []
                    for square in squares:
                        square_tensor = self.preprocess_square_for_color(square)
                        square_tensors.append(square_tensor)
                    
                    square_tensors = torch.stack(square_tensors).to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = model(square_tensors)
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
                torch.save(model.state_dict(), self.output_dir / "color_classification_marshall.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered")
                    break
        
        logger.info("Color classification training completed")
        return model
    
    def train_piece_classification_model(self):
        """Train improved piece classification model"""
        logger.info("Training Marshall piece classification model...")
        
        # Create dataset
        dataset = self.create_piece_classification_dataset()
        if len(dataset) < 100:
            logger.error("Not enough data for piece classification training")
            return None
        
        # Filter out None labels
        dataset = [d for d in dataset if d['label'] is not None]
        
        # Split dataset
        train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
        
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
        
        # Create model
        model = PieceModel().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_acc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(50):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                squares = batch['square']
                labels = batch['label']
                
                # Convert squares to tensors
                square_tensors = []
                for square in squares:
                    square_tensor = self.preprocess_square_for_piece(square)
                    square_tensors.append(square_tensor)
                
                square_tensors = torch.stack(square_tensors).to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(square_tensors)
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
                    
                    square_tensors = []
                    for square in squares:
                        square_tensor = self.preprocess_square_for_piece(square)
                        square_tensors.append(square_tensor)
                    
                    square_tensors = torch.stack(square_tensors).to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = model(square_tensors)
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
                torch.save(model.state_dict(), self.output_dir / "piece_classification_marshall.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered")
                    break
        
        logger.info("Piece classification training completed")
        return model
    
    def preprocess_image_for_corner_detection(self, image):
        """Preprocess image for corner detection"""
        # Resize to 224x224
        image = cv2.resize(image, (224, 224))
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Normalize
        image = image.astype(np.float32) / 255.0
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image
    
    def preprocess_square_for_occupancy(self, square):
        """Preprocess square for occupancy detection"""
        # Resize to 100x100
        square = cv2.resize(square, (100, 100))
        # Convert BGR to RGB
        square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        # Normalize
        square = square.astype(np.float32) / 255.0
        # Convert to tensor
        square = torch.from_numpy(square).permute(2, 0, 1)
        return square
    
    def preprocess_square_for_color(self, square):
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
    
    def preprocess_square_for_piece(self, square):
        """Preprocess square for piece classification"""
        # Resize to 224x224
        square = cv2.resize(square, (224, 224))
        # Convert BGR to RGB
        square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        # Normalize
        square = square.astype(np.float32) / 255.0
        # Convert to tensor
        square = torch.from_numpy(square).permute(2, 0, 1)
        return square
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        logger.info("🚀 Starting Marshall Training Pipeline")
        logger.info("=" * 60)
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Train all models
        logger.info("1. Training Corner Detection Model...")
        self.train_corner_detection_model()
        
        logger.info("2. Training Occupancy Detection Model...")
        self.train_occupancy_model()
        
        logger.info("3. Training Color Classification Model...")
        self.train_color_classification_model()
        
        logger.info("4. Training Piece Classification Model...")
        self.train_piece_classification_model()
        
        logger.info("🎉 Marshall Training Pipeline Completed!")
        logger.info(f"Models saved to: {self.output_dir}")
        
        # Create model info file
        self.create_model_info()
    
    def create_model_info(self):
        """Create info file about the trained models"""
        info = {
            "marshall_models": {
                "corner_detection_marshall.pt": "Improved corner detection using Marshall data",
                "occupancy_marshall.pt": "Improved occupancy detection using Marshall data", 
                "color_classification_marshall.pt": "Improved color classification using Marshall data",
                "piece_classification_marshall.pt": "Improved piece classification using Marshall data"
            },
            "training_data": {
                "total_annotations": len(self.valid_annotations),
                "excluded_images": len(self.excluded_images),
                "chess_set": "marshall"
            },
            "note": "These models are safe copies and do not affect existing working models"
        }
        
        with open(self.output_dir / "model_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info("Model info saved to model_info.json")

# Model definitions
class CornerDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        self.backbone.classifier = nn.Identity()
        self.regressor = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 8)  # 4 corners * 2 coordinates
        )
    
    def forward(self, x):
        features = self.backbone(x)
        corners = self.regressor(features)
        return corners

class OccupancyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(512, 2)  # occupied/empty
    
    def forward(self, x):
        return self.backbone(x)

class ColorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=None)
        self.backbone.classifier[1] = nn.Linear(1280, 2)  # white/black
    
    def forward(self, x):
        return self.backbone(x)

class PieceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        self.backbone.classifier[1] = nn.Linear(1280, 6)  # 6 piece types
    
    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    pipeline = MarshallTrainingPipeline()
    pipeline.run_training_pipeline()

