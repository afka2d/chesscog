#!/usr/bin/env python3
"""
Manual Corner Chess Recognition Pipeline
Bypasses automatic corner detection and uses manual corners for board warping
"""

import cv2
import numpy as np
import torch
import chess
from pathlib import Path
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)

class ManualCornerRecognizer:
    def __init__(self, models_path="runs"):
        """Initialize with trained models."""
        self.models_path = Path(models_path)
        
        # Load occupancy classifier
        occupancy_model_path = self.models_path / "occupancy_classifier" / "ResNet" / "ResNet.pt"
        self.occupancy_model = torch.load(str(occupancy_model_path), map_location='cpu', weights_only=False)
        self.occupancy_model.eval()
        
        # Load piece classifier
        piece_model_path = self.models_path / "piece_classifier" / "ResNet" / "ResNet.pt"
        self.piece_model = torch.load(str(piece_model_path), map_location='cpu', weights_only=False)
        self.piece_model.eval()
        
        # Define transforms (must match training configuration)
        self.occupancy_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((100, 100)),  # Match training config
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.piece_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 448)),  # Match training config
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Piece class mapping
        self.piece_classes = [
            'black_bishop', 'black_king', 'black_knight', 'black_pawn', 
            'black_queen', 'black_rook', 'white_bishop', 'white_king', 
            'white_knight', 'white_pawn', 'white_queen', 'white_rook'
        ]
        
        logger.info("Manual corner recognizer initialized successfully")
    
    def sort_corner_points(self, corners):
        """Sort corners to ensure correct order: top-left, top-right, bottom-right, bottom-left."""
        # Convert to numpy array if needed
        corners = np.array(corners, dtype=np.float32)
        
        # Find center
        center = np.mean(corners, axis=0)
        
        # Sort by angle from center
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        
        # Reorder corners
        sorted_corners = corners[sorted_indices]
        
        # Ensure correct order: top-left, top-right, bottom-right, bottom-left
        # This is a simplified approach - you might need to adjust based on your specific needs
        return sorted_corners
    
    def warp_chessboard(self, img, corners):
        """Warp the chessboard using manual corners."""
        # Sort corners
        sorted_corners = self.sort_corner_points(corners)
        
        # Define target size (8x8 squares, each 224x224 pixels)
        target_size = (1792, 1792)  # 8 * 224 = 1792
        
        # Define target corners (top-left, top-right, bottom-right, bottom-left)
        target_corners = np.array([
            [0, 0],           # top-left
            [target_size[0], 0],  # top-right
            [target_size[0], target_size[1]],  # bottom-right
            [0, target_size[1]]   # bottom-left
        ], dtype=np.float32)
        
        # Calculate perspective transform
        transform_matrix = cv2.getPerspectiveTransform(sorted_corners, target_corners)
        
        # Warp the image
        warped = cv2.warpPerspective(img, transform_matrix, target_size)
        
        return warped
    
    def extract_square(self, warped_board, rank, file):
        """Extract a specific square from the warped board."""
        # Calculate square coordinates (224x224 pixels each)
        x1 = file * 224
        y1 = rank * 224
        x2 = x1 + 224
        y2 = y1 + 224
        
        # Extract square
        square = warped_board[y1:y2, x1:x2]
        
        return square
    
    def predict_occupancy(self, square_img):
        """Predict if a square is occupied."""
        with torch.no_grad():
            # Apply transform
            input_tensor = self.occupancy_transform(square_img).unsqueeze(0)
            
            # Get prediction
            output = self.occupancy_model(input_tensor)
            probs = torch.softmax(output, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()
            
            return prediction == 1, confidence  # 1 = occupied, 0 = empty
    
    def predict_piece(self, square_img):
        """Predict the piece type on an occupied square."""
        with torch.no_grad():
            # Apply transform
            input_tensor = self.piece_transform(square_img).unsqueeze(0)
            
            # Get prediction
            output = self.piece_model(input_tensor)
            probs = torch.softmax(output, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()
            
            return self.piece_classes[prediction], confidence
    
    def recognize_position(self, img, manual_corners):
        """Recognize chess position using manual corners."""
        logger.info(f"Starting manual corner recognition with corners: {manual_corners}")
        
        # Warp the chessboard
        warped_board = self.warp_chessboard(img, manual_corners)
        logger.info(f"Warped board shape: {warped_board.shape}")
        
        # Create chess board
        board = chess.Board()
        board.clear()  # Start with empty board
        
        pieces_found = 0
        
        # Process each square
        for rank in range(8):
            for file in range(8):
                # Extract square
                square_img = self.extract_square(warped_board, rank, file)
                
                # Check occupancy
                is_occupied, occupancy_conf = self.predict_occupancy(square_img)
                
                if is_occupied and occupancy_conf > 0.5:  # Confidence threshold
                    # Predict piece type
                    piece_type, piece_conf = self.predict_piece(square_img)
                    
                    if piece_conf > 0.3:  # Confidence threshold
                        # Convert to chess square
                        square = chess.square(file, 7 - rank)  # Convert to chess coordinates
                        
                        # Create piece
                        if piece_type.startswith('white_'):
                            color = chess.WHITE
                            piece_name = piece_type[6:]  # Remove 'white_' prefix
                        else:
                            color = chess.BLACK
                            piece_name = piece_type[6:]  # Remove 'black_' prefix
                        
                        # Map piece names to chess constants
                        piece_map = {
                            'pawn': chess.PAWN,
                            'knight': chess.KNIGHT,
                            'bishop': chess.BISHOP,
                            'rook': chess.ROOK,
                            'queen': chess.QUEEN,
                            'king': chess.KING
                        }
                        
                        piece = chess.Piece(piece_map[piece_name], color)
                        board.set_piece_at(square, piece)
                        pieces_found += 1
                        
                        logger.info(f"Square {chr(97+file)}{8-rank}: {piece_type} (conf: {piece_conf:.3f})")
        
        logger.info(f"Total pieces found: {pieces_found}")
        return board, pieces_found

def test_manual_recognizer():
    """Test the manual corner recognizer."""
    # Initialize recognizer
    recognizer = ManualCornerRecognizer()
    
    # Load test image and corners
    img_path = "grey_background_dataset/images/test/IMG_4752.JPG"
    img = cv2.imread(img_path)
    
    # Manual corners from annotation
    manual_corners = [[993, 2294], [2702, 2064], [2755, 3892], [542, 3864]]
    
    # Recognize position
    board, pieces_found = recognizer.recognize_position(img, manual_corners)
    
    print(f"FEN: {board.fen()}")
    print(f"Pieces found: {pieces_found}")
    
    # Convert to 2D representation
    board_2d = []
    for rank in range(8):
        row = []
        for file in range(8):
            square = chess.square(file, 7 - rank)
            piece = board.piece_at(square)
            if piece:
                row.append(piece.symbol())
            else:
                row.append(".")
        board_2d.append(row)
    
    print("Board 2D:")
    for row in board_2d:
        print(" ".join(row))

if __name__ == "__main__":
    test_manual_recognizer() 