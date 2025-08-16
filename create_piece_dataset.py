#!/usr/bin/env python3
"""
Create piece dataset with larger square images for better classification
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
import chess

def sort_corner_points(corners):
    """Sort corners to ensure correct order: top-left, top-right, bottom-right, bottom-left."""
    corners = np.array(corners, dtype=np.float32)
    
    # Find center
    center = np.mean(corners, axis=0)
    
    # Sort by angle from center
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    
    # Reorder corners
    sorted_corners = corners[sorted_indices]
    
    return sorted_corners

def warp_chessboard(img, corners):
    """Warp the chessboard using manual corners."""
    # Sort corners
    sorted_corners = sort_corner_points(corners)
    
    # Define target size (8x8 squares, each 224x224 pixels for higher resolution)
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

def extract_square(warped_board, rank, file):
    """Extract a specific square from the warped board."""
    # Calculate square coordinates (224x224 pixels each)
    x1 = file * 224
    y1 = rank * 224
    x2 = x1 + 224
    y2 = y1 + 224
    
    # Extract square
    square = warped_board[y1:y2, x1:x2]
    
    return square

def create_piece_dataset():
    """Create piece dataset with larger square images."""
    
    # Setup directories
    dataset_dir = Path("grey_background_dataset")
    pieces_dir = Path("grey_background_dataset/pieces")
    
    # Create piece directories
    piece_types = [
        'black_bishop', 'black_king', 'black_knight', 'black_pawn', 
        'black_queen', 'black_rook', 'white_bishop', 'white_king', 
        'white_knight', 'white_pawn', 'white_queen', 'white_rook'
    ]
    
    for subset in ['train', 'val', 'test']:
        for piece_type in piece_types:
            (pieces_dir / subset / piece_type).mkdir(parents=True, exist_ok=True)
    
    # Process each subset
    for subset in ['train', 'val', 'test']:
        print(f"\nProcessing {subset} set...")
        
        images_dir = dataset_dir / "images" / subset
        annotations_dir = dataset_dir / "annotations" / subset
        
        if not images_dir.exists():
            print(f"Images directory not found: {images_dir}")
            continue
            
        image_files = list(images_dir.glob("*.JPG"))
        print(f"Found {len(image_files)} images in {subset}")
        
        for i, img_file in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {img_file.name}")
            
            # Load image
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"Failed to load image: {img_file}")
                continue
            
            # Load annotation
            annotation_file = annotations_dir / f"{img_file.stem}.json"
            if not annotation_file.exists():
                print(f"Annotation not found: {annotation_file}")
                continue
                
            with open(annotation_file, 'r') as f:
                annotation = json.load(f)
            
            # Get corners and FEN
            corners = annotation.get('corners')
            fen = annotation.get('fen')
            
            if not corners or not fen:
                print(f"Missing corners or FEN in {annotation_file}")
                continue
            
            try:
                # Parse FEN
                board = chess.Board(fen)
                
                # Warp chessboard
                warped_board = warp_chessboard(img, corners)
                
                # Extract pieces
                for rank in range(8):
                    for file in range(8):
                        square = chess.square(file, 7 - rank)
                        piece = board.piece_at(square)
                        
                        if piece:
                            # Extract square image
                            square_img = extract_square(warped_board, rank, file)
                            
                            # Determine piece type
                            color = "white" if piece.color else "black"
                            piece_name = piece.symbol().lower()
                            
                            if piece_name == 'p':
                                piece_type = f"{color}_pawn"
                            elif piece_name == 'r':
                                piece_type = f"{color}_rook"
                            elif piece_name == 'n':
                                piece_type = f"{color}_knight"
                            elif piece_name == 'b':
                                piece_type = f"{color}_bishop"
                            elif piece_name == 'q':
                                piece_type = f"{color}_queen"
                            elif piece_name == 'k':
                                piece_type = f"{color}_king"
                            else:
                                continue
                            
                            # Save piece image
                            output_path = pieces_dir / subset / piece_type / f"{img_file.stem}_{chr(97+file)}{8-rank}.png"
                            cv2.imwrite(str(output_path), square_img)
                            
            except Exception as e:
                print(f"Error processing {img_file.name}: {e}")
                continue
    
    print("\n✅ Successfully created piece dataset with larger images!")

if __name__ == "__main__":
    create_piece_dataset() 