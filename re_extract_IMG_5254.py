#!/usr/bin/env python3
"""
Re-extract pieces for IMG_5254.JPG using the corrected corners.
This will replace the incorrectly extracted pieces with properly aligned ones.
"""

import os
import cv2
import numpy as np
import json
import chess
from pathlib import Path
import shutil

def re_extract_pieces_for_image(image_path, annotation_path, output_dir):
    """Re-extract pieces for a specific image using corrected corners."""
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    # Load annotation
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    
    corners = data['corners']
    fen = data['fen']
    image_name = os.path.splitext(data['image'])[0]
    
    print(f"🖼️  Re-extracting pieces for: {data['image']}")
    print(f"📍 Using corners: {corners}")
    print(f"♟️  FEN: {fen}")
    
    # Parse the chess position
    try:
        board = chess.Board(fen)
    except ValueError as e:
        print(f"❌ Invalid FEN: {e}")
        return
    
    # Convert corners to numpy array for perspective transform
    src_corners = np.array(corners, dtype=np.float32)
    
    # Define the destination corners (square board)
    board_size = 800
    dst_corners = np.array([
        [0, 0],                    # Top-left
        [board_size, 0],           # Top-right  
        [board_size, board_size],  # Bottom-right
        [0, board_size]            # Bottom-left
    ], dtype=np.float32)
    
    # Calculate perspective transform
    transform_matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)
    
    # Apply perspective transform
    warped = cv2.warpPerspective(img, transform_matrix, (board_size, board_size))
    
    # Save debug image
    cv2.imwrite(f"debug_warped_{image_name}.png", warped)
    print(f"💾 Debug warped board saved: debug_warped_{image_name}.png")
    
    # Extract individual squares
    square_size = board_size // 8
    pieces_extracted = 0
    
    # First, remove old pieces for this image
    old_pieces_removed = 0
    for piece_dir in Path(output_dir).glob("*/"):
        if piece_dir.is_dir():
            for old_piece in piece_dir.glob(f"{image_name}_*.png"):
                old_piece.unlink()
                old_pieces_removed += 1
    
    print(f"🗑️  Removed {old_pieces_removed} old pieces for {image_name}")
    
    # Extract new pieces with correct alignment
    for rank in range(8):
        for file in range(8):
            # Calculate square position (rank 0 = rank 8, rank 7 = rank 1)
            chess_rank = 8 - rank
            chess_file = chr(ord('a') + file)
            square_name = f"{chess_file}{chess_rank}"
            
            # Get the piece at this square
            square_index = chess.square(file, 7 - rank)  # Convert to python-chess indexing
            piece = board.piece_at(square_index)
            
            # Extract square image
            y1 = rank * square_size
            y2 = (rank + 1) * square_size
            x1 = file * square_size  
            x2 = (file + 1) * square_size
            
            square_img = warped[y1:y2, x1:x2]
            
            if piece is not None:
                # Determine piece type and color
                piece_color = "white" if piece.color == chess.WHITE else "black"
                piece_type = piece.piece_type
                piece_names = {
                    chess.PAWN: "pawn",
                    chess.ROOK: "rook", 
                    chess.KNIGHT: "knight",
                    chess.BISHOP: "bishop",
                    chess.QUEEN: "queen",
                    chess.KING: "king"
                }
                piece_name = piece_names[piece_type]
                
                # Create output directory
                piece_class = f"{piece_color}_{piece_name}"
                piece_dir = os.path.join(output_dir, piece_class)
                os.makedirs(piece_dir, exist_ok=True)
                
                # Save piece image
                piece_filename = f"{image_name}_{square_name}.png"
                piece_path = os.path.join(piece_dir, piece_filename)
                
                # Resize to match training format (100x200)
                resized_square = cv2.resize(square_img, (100, 200))
                cv2.imwrite(piece_path, resized_square)
                
                pieces_extracted += 1
                print(f"✅ Extracted {piece_class}: {piece_filename}")
    
    print(f"\n🎯 Re-extraction complete!")
    print(f"   📊 Total pieces extracted: {pieces_extracted}")
    print(f"   🗑️  Old pieces removed: {old_pieces_removed}")
    return pieces_extracted

if __name__ == "__main__":
    # Paths for IMG_5254.JPG
    image_path = "enhanced_training_dataset/images/IMG_5254.JPG"
    annotation_path = "enhanced_training_dataset/annotations/IMG_5254.json"
    output_dir = "enhanced_training_dataset/pieces/train"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
    elif not os.path.exists(annotation_path):
        print(f"❌ Annotation not found: {annotation_path}")
    else:
        re_extract_pieces_for_image(image_path, annotation_path, output_dir)

