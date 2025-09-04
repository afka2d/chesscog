#!/usr/bin/env python3
"""
Re-process IMG_4755 with the correct FEN and corners.
This will fix the incorrectly labeled piece.
"""

import os
import cv2
import numpy as np
import json
import chess
from pathlib import Path

def display_image_with_grid(image_path):
    """Display image with 8x8 grid overlay and get corner input."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Draw 8x8 grid
    grid_img = img_rgb.copy()
    
    # Vertical lines
    for i in range(9):
        x = int((width * i) / 8)
        cv2.line(grid_img, (x, 0), (x, height), (255, 0, 0), 2)
    
    # Horizontal lines
    for i in range(9):
        y = int((height * i) / 8)
        cv2.line(grid_img, (0, y), (width, y), (255, 0, 0), 2)
    
    # Add coordinate labels
    for rank in range(8):
        for file in range(8):
            x = int((width * file) / 8) + 20
            y = int((height * rank) / 8) + 30
            label = f"{chr(97+file)}{8-rank}"
            cv2.putText(grid_img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Display image
    cv2.imshow('IMG_4755 - Click corners in order: a8, h8, h1, a1', grid_img)
    cv2.waitKey(1)
    
    print(f"\n📸 Processing: IMG_4755.JPG")
    print("🔍 Click the four corners in this order:")
    print("   1. Top-left (a8) - White's perspective")
    print("   2. Top-right (h8)")
    print("   3. Bottom-right (h1)")
    print("   4. Bottom-left (a1)")
    print("   Click each corner, then press 'Enter' to continue")
    
    # Get corner clicks
    corners = []
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            corners.append((x, y))
            print(f"   Corner {len(corners)}: ({x}, {y})")
            cv2.circle(grid_img, (x, y), 10, (0, 255, 0), -1)
            cv2.imshow('IMG_4755 - Click corners in order: a8, h8, h1, a1', grid_img)
    
    cv2.setMouseCallback('IMG_4755 - Click corners in order: a8, h8, h1, a1', mouse_callback)
    
    input("Press Enter when all 4 corners are clicked...")
    cv2.destroyAllWindows()
    
    if len(corners) == 4:
        print(f"✅ All corners captured: {corners}")
        return img, corners
    else:
        raise ValueError(f"Expected 4 corners, got {len(corners)}")

def get_fen_input():
    """Get FEN input from user."""
    print(f"\n♟️  Enter the CORRECT FEN for IMG_4755:")
    print("   Format: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    print("   (This should be the actual position in the image)")
    
    fen = input("FEN: ").strip()
    
    # Validate FEN
    try:
        board = chess.Board(fen)
        print(f"✅ Valid FEN: {fen}")
        return fen
    except ValueError as e:
        print(f"❌ Invalid FEN: {e}")
        return get_fen_input()

def extract_pieces_from_board(img, corners, fen, image_name):
    """Extract individual pieces using the provided corners and FEN."""
    try:
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

def save_annotation(corners, fen):
    """Save annotation data."""
    annotation = {
        "image": "IMG_4755.JPG",
        "corners": corners,
        "fen": fen,
        "timestamp": "reprocessed"
    }
    
    annotation_path = "grey_background_dataset/annotations/train/IMG_4755.json"
    with open(annotation_path, 'w') as f:
        json.dump(annotation, f, indent=2)
    
    print(f"💾 Annotation updated: {annotation_path}")

def main():
    """Main function to re-process IMG_4755."""
    print("🔧 Re-processing IMG_4755 with correct FEN and corners")
    print("=" * 60)
    
    image_path = "grey_background_dataset/images/train/IMG_4755.JPG"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    try:
        # Get corners manually
        img, corners = display_image_with_grid(image_path)
        
        # Get FEN input
        fen = get_fen_input()
        
        # Save updated annotation
        save_annotation(corners, fen)
        
        # Extract pieces
        pieces_extracted = extract_pieces_from_board(img, corners, fen, "IMG_4755")
        
        print(f"\n✅ Re-processing complete!")
        print(f"♟️  Pieces extracted: {pieces_extracted}")
        print(f"📁 Pieces saved to: grey_background_dataset/pieces/train/")
        print(f"📁 Annotation updated: grey_background_dataset/annotations/train/IMG_4755.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

