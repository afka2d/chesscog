#!/usr/bin/env python3
"""
Simple script to fix IMG_4755 annotation and re-extract pieces.
"""

import os
import cv2
import numpy as np
import json
import chess

def fix_IMG_4755():
    """Fix IMG_4755 with correct corners and FEN."""
    
    # Current incorrect annotation
    current_annotation = {
        "image": "IMG_4755.JPG",
        "corners": [
            [897, 2193],    # a8 (top-left)
            [2731, 2140],   # h8 (top-right) 
            [2736, 4084],   # h1 (bottom-right)
            [451, 3921]     # a1 (bottom-left)
        ],
        "fen": "rnbqkbnr/1ppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1",
        "timestamp": "original"
    }
    
    print("🔧 Fixing IMG_4755 annotation")
    print("=" * 40)
    print(f"Current FEN: {current_annotation['fen']}")
    print(f"Current corners: {current_annotation['corners']}")
    print()
    
    # Get new corners
    print("Enter new corner coordinates (or press Enter to keep current):")
    new_corners = []
    
    corner_names = ["a8 (top-left)", "h8 (top-right)", "h1 (bottom-right)", "a1 (bottom-left)"]
    for i, (name, current) in enumerate(zip(corner_names, current_annotation['corners'])):
        user_input = input(f"Corner {i+1} ({name}) [{current[0]},{current[1]}]: ").strip()
        if user_input:
            try:
                x, y = map(int, user_input.split(','))
                new_corners.append([x, y])
            except:
                print(f"Invalid format, keeping current: {current}")
                new_corners.append(current)
        else:
            new_corners.append(current)
    
    # Get new FEN
    print(f"\nCurrent FEN: {current_annotation['fen']}")
    print("This FEN shows e8 as black king, but you said it's actually a pawn")
    new_fen = input("Enter correct FEN (or press Enter to keep current): ").strip()
    
    if not new_fen:
        new_fen = current_annotation['fen']
    
    # Validate FEN
    try:
        board = chess.Board(new_fen)
        print(f"✅ Valid FEN: {new_fen}")
    except ValueError as e:
        print(f"❌ Invalid FEN: {e}")
        return
    
    # Update annotation
    updated_annotation = {
        "image": "IMG_4755.JPG",
        "corners": new_corners,
        "fen": new_fen,
        "timestamp": "fixed"
    }
    
    # Save updated annotation
    annotation_path = "grey_background_dataset/annotations/train/IMG_4755.json"
    with open(annotation_path, 'w') as f:
        json.dump(updated_annotation, f, indent=2)
    
    print(f"\n💾 Annotation updated: {annotation_path}")
    
    # Re-extract pieces
    print("\n🔄 Re-extracting pieces...")
    pieces_extracted = extract_pieces_from_board(
        "grey_background_dataset/images/train/IMG_4755.JPG",
        new_corners, 
        new_fen, 
        "IMG_4755"
    )
    
    print(f"✅ Re-extraction complete! {pieces_extracted} pieces extracted")

def extract_pieces_from_board(image_path, corners, fen, image_name):
    """Extract individual pieces using the provided corners and FEN."""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Parse FEN to get piece positions
        board = chess.Board(fen)
        
        # Warp the chessboard to get a square grid
        warped = warp_chessboard(img, corners)
        
        # Calculate square size
        square_size = min(warped.shape[0], warped.shape[1]) // 8
        
        pieces_extracted = 0
        
        # Piece type mapping
        piece_mapping = {
            'P': 'white_pawn', 'R': 'white_rook', 'N': 'white_knight',
            'B': 'white_bishop', 'Q': 'white_queen', 'K': 'white_king',
            'p': 'black_pawn', 'r': 'black_rook', 'n': 'black_knight',
            'b': 'black_bishop', 'q': 'black_queen', 'k': 'black_king'
        }
        
        # Process each square
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, 7 - rank)  # Convert to chess coordinates
                piece = board.piece_at(square)
                
                if piece is not None:
                    # Extract square image
                    x1 = file * square_size
                    y1 = rank * square_size
                    x2 = x1 + square_size
                    y2 = y1 + square_size
                    
                    square_img = warped[y1:y2, x1:x2]
                    
                    if square_img.size > 0:
                        # Resize to standard size
                        square_resized = cv2.resize(square_img, (100, 200))
                        
                        # Determine piece type and color
                        piece_char = piece.symbol()
                        folder_name = piece_mapping[piece_char]
                        
                        # Create folder if it doesn't exist
                        piece_folder = os.path.join("grey_background_dataset", "pieces", "train", folder_name)
                        os.makedirs(piece_folder, exist_ok=True)
                        
                        # Generate filename
                        piece_filename = f"{image_name}_{chr(97+file)}{8-rank}.png"
                        piece_path = os.path.join(piece_folder, piece_filename)
                        
                        # Save piece image
                        cv2.imwrite(piece_path, square_resized)
                        pieces_extracted += 1
                        
                        print(f"   Extracted {piece_char} from {chr(97+file)}{8-rank} -> {folder_name}/{piece_filename}")
        
        return pieces_extracted
        
    except Exception as e:
        print(f"❌ Error extracting pieces: {e}")
        return 0

def warp_chessboard(img, corners):
    """Warp the chessboard to a square grid using the provided corners."""
    # Convert corners to numpy array
    src_corners = np.array(corners, dtype=np.float32)
    
    # Define destination corners (square grid)
    board_size = 400  # Size of warped board
    dst_corners = np.array([
        [0, 0],                    # Top-left
        [board_size, 0],           # Top-right
        [board_size, board_size],  # Bottom-right
        [0, board_size]            # Bottom-left
    ], dtype=np.float32)
    
    # Calculate perspective transform
    transform_matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)
    
    # Apply transform
    warped = cv2.warpPerspective(img, transform_matrix, (board_size, board_size))
    
    return warped

if __name__ == "__main__":
    fix_IMG_4755()

