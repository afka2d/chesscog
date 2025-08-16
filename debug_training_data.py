#!/usr/bin/env python3
"""
Debug training data to see if there's a mismatch.
"""

import cv2
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

def debug_training_data():
    """Check if training data matches expected positions."""
    
    # Load the same test image
    img_path = "grey_background_dataset/images/test/IMG_4752.JPG"
    annotation_path = "grey_background_dataset/annotations/test/IMG_4752.json"
    
    # Load image and annotation
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    
    corners = np.array(annotation['corners'], dtype=np.float32)
    fen = annotation['fen']
    
    print(f"Image: {img_path}")
    print(f"FEN: {fen}")
    print(f"Corners: {corners}")
    
    # Warp the image using the same method as training
    target_size = (1792, 1792)  # 8 * 224
    target_corners = np.array([
        [0, 0],
        [target_size[0], 0],
        [target_size[0], target_size[1]],
        [0, target_size[1]]
    ], dtype=np.float32)
    
    transform_matrix = cv2.getPerspectiveTransform(corners, target_corners)
    warped = cv2.warpPerspective(img, transform_matrix, target_size)
    
    # Create visualization showing what we expect vs. what training data has
    fig, axes = plt.subplots(2, 8, figsize=(20, 10))
    fig.suptitle(f"Training Data Debug: {Path(img_path).name}", fontsize=16)
    
    # Expected pieces from FEN
    import chess
    board = chess.Board(fen)
    
    for rank in range(8):
        for file in range(8):
            # Extract square from warped image
            x1 = file * 224
            y1 = rank * 224
            x2 = x1 + 224
            y2 = y1 + 224
            square_img = warped[y1:y2, x1:x2]
            
            # Get expected piece from FEN
            chess_square = chess.square(file, 7 - rank)
            expected_piece = board.piece_at(chess_square)
            
            # Show extracted square
            axes[0, file].imshow(square_img)
            axes[0, file].set_title(f"{chr(97+file)}{8-rank}", fontsize=8)
            axes[0, file].axis('off')
            
            # Show expected piece
            axes[1, file].text(0.5, 0.5, 
                             expected_piece.symbol() if expected_piece else ".", 
                             ha='center', va='center', fontsize=20, 
                             transform=axes[1, file].transAxes)
            axes[1, file].set_title("Expected", fontsize=8)
            axes[1, file].axis('off')
    
    plt.tight_layout()
    plt.savefig('training_data_debug.png', dpi=150, bbox_inches='tight')
    print("✅ Debug visualization saved to training_data_debug.png")
    
    # Also check what pieces should be in the training data
    pieces_dir = Path("grey_background_dataset/pieces/test")
    print(f"\nExpected pieces in training data for {Path(img_path).stem}:")
    for rank in range(8):
        for file in range(8):
            chess_square = chess.square(file, 7 - rank)
            piece = board.piece_at(chess_square)
            if piece:
                square_name = f"{chr(97+file)}{8-rank}"
                color = "white" if piece.color else "black"
                piece_type = {
                    chess.PAWN: "pawn", chess.ROOK: "rook", chess.KNIGHT: "knight",
                    chess.BISHOP: "bishop", chess.QUEEN: "queen", chess.KING: "king"
                }[piece.piece_type]
                
                expected_file = pieces_dir / f"{color}_{piece_type}" / f"{Path(img_path).stem}_{square_name}.png"
                exists = expected_file.exists()
                print(f"  {square_name}: {color} {piece_type} -> {expected_file} {'✓' if exists else '✗'}")

if __name__ == "__main__":
    debug_training_data()