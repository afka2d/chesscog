#!/usr/bin/env python3
"""
Improved Two-Stage Chess Piece Classifier
Integrates improved color and piece type classifiers with better error handling.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import cv2
import chess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedColorClassifier(nn.Module):
    """Improved color classifier with regularization."""
    
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(ImprovedColorClassifier, self).__init__()
        
        # Use a pre-trained ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=False)  # Will load from saved weights
        
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

class ImprovedPieceTypeClassifier(nn.Module):
    """Improved piece type classifier with regularization."""
    
    def __init__(self, num_classes=6, dropout_rate=0.3):
        super(ImprovedPieceTypeClassifier, self).__init__()
        
        # Use a pre-trained ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=False)  # Will load from saved weights
        
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

class ImprovedTwoStagePieceClassifier:
    """
    Improved two-stage chess piece classifier that addresses overfitting
    and provides better confidence scoring.
    """
    
    def __init__(self, color_model_path=None, piece_model_path=None):
        self.color_classifier = None
        self.piece_type_classifier = None
        self.color_transforms = None
        self.piece_transforms = None
        
        # Load models if paths provided
        if color_model_path:
            self.load_color_classifier(color_model_path)
        if piece_model_path:
            self.load_piece_type_classifier(piece_model_path)
        
        # Set up transforms
        self._setup_transforms()
        
        logger.info("Improved Two-Stage Piece Classifier initialized")
    
    def _setup_transforms(self):
        """Set up transforms for both classifiers."""
        # Color classifier transforms
        self.color_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((100, 200)),  # Match training size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Piece type classifier transforms
        self.piece_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((100, 200)),  # Match training size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_color_classifier(self, model_path):
        """Load the color classifier model."""
        try:
            logger.info(f"Loading color classifier from {model_path}")
            self.color_classifier = ImprovedColorClassifier(num_classes=2, dropout_rate=0.4)
            
            # Load state dict
            state_dict = torch.load(model_path, map_location='cpu')
            self.color_classifier.load_state_dict(state_dict)
            self.color_classifier.eval()
            
            logger.info("Color classifier loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load color classifier: {e}")
            self.color_classifier = None
    
    def load_piece_type_classifier(self, model_path):
        """Load the piece type classifier model."""
        try:
            logger.info(f"Loading piece type classifier from {model_path}")
            self.piece_type_classifier = ImprovedPieceTypeClassifier(num_classes=6, dropout_rate=0.4)
            
            # Load state dict
            state_dict = torch.load(model_path, map_location='cpu')
            self.piece_type_classifier.load_state_dict(state_dict)
            self.piece_type_classifier.eval()
            
            logger.info("Piece type classifier loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load piece type classifier: {e}")
            self.piece_type_classifier = None
    
    def classify_piece(self, square_img, confidence_threshold=0.7):
        """
        Classify a single chess piece using the two-stage approach.
        
        Args:
            square_img: numpy array of the square image
            confidence_threshold: minimum confidence for predictions
            
        Returns:
            tuple: (piece_name, color_confidence, piece_confidence, overall_confidence)
        """
        if self.color_classifier is None or self.piece_type_classifier is None:
            logger.error("Models not loaded")
            return None, 0.0, 0.0, 0.0
        
        try:
            # Stage 1: Color classification
            color_input = self.color_transforms(square_img).unsqueeze(0)
            
            with torch.no_grad():
                color_output = self.color_classifier(color_input)
                color_probs = torch.softmax(color_output, dim=1)
                color_confidence, color_pred = torch.max(color_probs, dim=1)
                
                color = "white" if color_pred.item() == 1 else "black"
                color_conf = color_confidence.item()
            
            # Stage 2: Piece type classification
            piece_input = self.piece_transforms(square_img).unsqueeze(0)
            
            with torch.no_grad():
                piece_output = self.piece_type_classifier(piece_input)
                piece_probs = torch.softmax(piece_output, dim=1)
                piece_confidence, piece_pred = torch.max(piece_probs, dim=1)
                
                piece_types = ["pawn", "rook", "knight", "bishop", "queen", "king"]
                piece_type = piece_types[piece_pred.item()]
                piece_conf = piece_confidence.item()
            
            # Combine predictions
            full_piece_name = f"{color}_{piece_type}"
            overall_confidence = (color_conf + piece_conf) / 2
            
            # Check confidence threshold
            if overall_confidence < confidence_threshold:
                logger.warning(f"Low confidence prediction: {full_piece_name} (conf: {overall_confidence:.3f})")
            
            return full_piece_name, color_conf, piece_conf, overall_confidence
            
        except Exception as e:
            logger.error(f"Error during piece classification: {e}")
            return None, 0.0, 0.0, 0.0
    
    def classify_board(self, img, corners, occupancy):
        """
        Classify all pieces on the chess board.
        
        Args:
            img: full board image
            corners: board corner coordinates
            occupancy: 8x8 occupancy matrix (True for occupied squares)
            
        Returns:
            list: 8x8 list of chess.Piece objects
        """
        if self.color_classifier is None or self.piece_type_classifier is None:
            logger.error("Models not loaded")
            return None
        
        try:
            # Warp the image to get a square board
            warped_img = self._warp_board(img, corners)
            
            # Extract individual squares
            squares = self._extract_squares(warped_img)
            
            # Classify each occupied square
            board = [[None for _ in range(8)] for _ in range(8)]
            
            for rank in range(8):
                for file in range(8):
                    if occupancy[rank][file]:  # Square is occupied
                        square_img = squares[rank][file]
                        
                        piece_name, color_conf, piece_conf, overall_conf = self.classify_piece(square_img)
                        
                        if piece_name:
                            chess_piece = self._name_to_chess_piece(piece_name)
                            board[rank][file] = chess_piece
                            
                            logger.debug(f"Square {chr(97+file)}{8-rank}: {piece_name} "
                                       f"(color_conf: {color_conf:.3f}, piece_conf: {piece_conf:.3f})")
                        else:
                            logger.warning(f"Failed to classify piece at {chr(97+file)}{8-rank}")
            
            return board
            
        except Exception as e:
            logger.error(f"Error during board classification: {e}")
            return None
    
    def _warp_board(self, img, corners):
        """Warp the image to get a square board view."""
        # Define target board size
        target_size = 800
        
        # Define target corners (square board)
        target_corners = np.array([
            [0, 0],                    # Top-left
            [target_size, 0],          # Top-right
            [target_size, target_size], # Bottom-right
            [0, target_size]           # Bottom-left
        ], dtype=np.float32)
        
        # Convert corners to numpy array
        src_corners = np.array(corners, dtype=np.float32)
        
        # Calculate perspective transform
        transform_matrix = cv2.getPerspectiveTransform(src_corners, target_corners)
        
        # Apply perspective transform
        warped = cv2.warpPerspective(img, transform_matrix, (target_size, target_size))
        
        return warped
    
    def _extract_squares(self, warped_img):
        """Extract individual squares from the warped board image."""
        squares = []
        square_size = warped_img.shape[0] // 8
        
        for rank in range(8):
            rank_squares = []
            for file in range(8):
                # Extract square (note: chess coordinates are different from image coordinates)
                y1 = rank * square_size
                y2 = (rank + 1) * square_size
                x1 = file * square_size
                x2 = (file + 1) * square_size
                
                square = warped_img[y1:y2, x1:x2]
                rank_squares.append(square)
            
            squares.append(rank_squares)
        
        return squares
    
    def _name_to_chess_piece(self, piece_name):
        """Convert piece name string to chess.Piece object."""
        try:
            color_str, piece_type = piece_name.split('_')
            color = chess.WHITE if color_str == "white" else chess.BLACK
            
            piece_map = {
                "pawn": chess.PAWN,
                "rook": chess.ROOK,
                "knight": chess.KNIGHT,
                "bishop": chess.BISHOP,
                "queen": chess.QUEEN,
                "king": chess.KING
            }
            
            piece_type_enum = piece_map.get(piece_type)
            if piece_type_enum is None:
                logger.error(f"Unknown piece type: {piece_type}")
                return None
            
            return chess.Piece(piece_type_enum, color)
            
        except Exception as e:
            logger.error(f"Error converting piece name {piece_name}: {e}")
            return None
    
    def get_model_info(self):
        """Get information about loaded models."""
        info = {
            "color_classifier_loaded": self.color_classifier is not None,
            "piece_type_classifier_loaded": self.piece_type_classifier is not None,
            "models_ready": self.color_classifier is not None and self.piece_type_classifier is not None
        }
        
        if self.color_classifier is not None:
            info["color_classifier_params"] = sum(p.numel() for p in self.color_classifier.parameters())
        
        if self.piece_type_classifier is not None:
            info["piece_type_classifier_params"] = sum(p.numel() for p in self.piece_type_classifier.parameters())
        
        return info

def test_improved_classifier():
    """Test the improved classifier."""
    logger.info("Testing Improved Two-Stage Classifier...")
    
    # Initialize classifier
    classifier = ImprovedTwoStagePieceClassifier()
    
    # Check model info
    info = classifier.get_model_info()
    logger.info(f"Model info: {info}")
    
    # Test with sample data (you would need actual images here)
    logger.info("Classifier test completed")

if __name__ == "__main__":
    test_improved_classifier()
