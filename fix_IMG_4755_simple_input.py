#!/usr/bin/env python3
"""
Simple text-based fix for IMG_4755 - no visual display needed.
"""

import os
import cv2
import numpy as np
import json
import chess

def get_text_input():
    """Get corner coordinates and FEN from text input."""
    
    print("🔧 Fixing IMG_4755 annotation")
    print("=" * 50)
    print("Current annotation:")
    print("  FEN: rnbqkbnr/1ppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1")
    print("  Corners: [897,2193], [2731,2140], [2736,4084], [451,3921]")
    print()
    print("The current FEN shows e8 as black king, but you said it's actually a pawn.")
    print()
    
    # Get new corners
    print("Enter new corner coordinates (or press Enter to keep current):")
    print("Format: x,y (e.g., 897,2193)")
    print()
    
    new_corners = []
    corner_names = ["a8 (top-left)", "h8 (top-right)", "h1 (bottom-right)", "a1 (bottom-left)"]
    current_corners = [[897, 2193], [2731, 2140], [2736, 4084], [451, 3921]]
    
    for i, (name, current) in enumerate(zip(corner_names, current_corners)):
        while True:
            user_input = input(f"Corner {i+1} ({name}) [{current[0]},{current[1]}]: ").strip()
            if not user_input:
                # Keep current value
                new_corners.append(current)
                print(f"   Keeping current: {current}")
                break
            elif ',' in user_input:
                try:
                    x, y = map(int, user_input.split(','))
                    new_corners.append([x, y])
                    print(f"   New value: [{x}, {y}]")
                    break
                except ValueError:
                    print("   ❌ Invalid format. Please use: x,y")
            else:
                print("   ❌ Please use format: x,y")
    
    # Get new FEN
    print()
    print("Enter the CORRECT FEN for this position:")
    print("Current: rnbqkbnr/1ppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1")
    print("(This should show the actual pieces in the image)")
    
    while True:
        new_fen = input("New FEN: ").strip()
        if not new_fen:
            new_fen = "rnbqkbnr/1ppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w KQkq - 0 1"
            print("   Keeping current FEN")
            break
        
        # Validate FEN
        try:
            board = chess.Board(new_fen)
            print(f"   ✅ Valid FEN: {new_fen}")
            break
        except ValueError as e:
            print(f"   ❌ Invalid FEN: {e}")
            print("   Please try again")
    
    return new_corners, new_fen

def fix_IMG_4755():
    """Fix IMG_4755 with user input."""
    
    # Get input from user
    new_corners, new_fen = get_text_input()
    
    print()
    print("📋 Summary of changes:")
    print(f"  New corners: {new_corners}")
    print(f"  New FEN: {new_fen}")
    print()
    
    # Confirm before proceeding
    confirm = input("Proceed with these changes? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("❌ Cancelled")
        return
    
    # Update annotation
    updated_annotation = {
        "image": "IMG_4755.JPG",
        "corners": new_corners,
        "fen": new_fen,
        "timestamp": "fixed_simple_input"
    }
    
    # Save updated annotation
    annotation_path = "grey_background_dataset/annotations/train/IMG_4755.json"
    with open(annotation_path, 'w') as f:
        json.dump(updated_annotation, f, indent=2)
    
    print(f"💾 Annotation updated: {annotation_path}")
    
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

