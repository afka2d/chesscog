#!/usr/bin/env python3
"""
Continue debugging IMG_4755 using the captured corners.
"""

import os
import cv2
import numpy as np
import json
import chess
import matplotlib.pyplot as plt

def get_fen_input():
    """Get FEN input from user."""
    print(f"\n♟️  Enter the CORRECT FEN for IMG_4755:")
    print("   Format: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    print("   (This should be the actual position in the image)")
    print("   Current FEN shows e8 as black king, but you said it's actually a pawn")
    
    fen = input("FEN: ").strip()
    
    # Validate FEN
    try:
        board = chess.Board(fen)
        print(f"✅ Valid FEN: {fen}")
        return fen
    except ValueError as e:
        print(f"❌ Invalid FEN: {e}")
        return get_fen_input()

def show_warped_board(img, corners, fen):
    """Show the warped board to verify the transformation."""
    try:
        # Parse FEN to get piece positions
        board = chess.Board(fen)
        
        # Warp the chessboard to get a square grid
        warped = warp_chessboard(img, corners)
        
        # Save warped image for inspection
        warped_path = "debug_warped_IMG_4755.png"
        cv2.imwrite(warped_path, warped)
        print(f"💾 Warped board saved to: {warped_path}")
        
        # Show with matplotlib
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        plt.title('Warped Board - Verify squares align correctly')
        plt.axis('off')
        
        print("\n🔍 Verify the warped board:")
        print("   - Squares should be perfectly aligned")
        print("   - Each square should contain one piece")
        print("   - The grid should be 8x8")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Error showing warped board: {e}")

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

def analyze_fen_vs_image(fen, corners):
    """Analyze what the FEN says vs what should be extracted."""
    try:
        board = chess.Board(fen)
        print(f"\n📊 FEN Analysis:")
        print(f"   FEN: {fen}")
        print(f"   Corners: {corners}")
        
        # Show what each square should contain
        print(f"\n🔍 Expected piece positions:")
        for rank in range(8):
            row_pieces = []
            for file in range(8):
                square = chess.square(file, 7 - rank)
                piece = board.piece_at(square)
                if piece:
                    row_pieces.append(f"{piece.symbol()}")
                else:
                    row_pieces.append(".")
            print(f"   Rank {8-rank}: {' '.join(row_pieces)}")
        
        # Highlight the problematic square e8
        e8_square = chess.square(4, 0)  # e8 = file 4, rank 0
        e8_piece = board.piece_at(e8_square)
        if e8_piece:
            print(f"\n⚠️  Square e8 (file 4, rank 0): {e8_piece.symbol()} ({e8_piece.color})")
            print(f"   This should be extracted as: {e8_piece.symbol()}")
        else:
            print(f"\n⚠️  Square e8 (file 4, rank 0): empty")
        
    except Exception as e:
        print(f"❌ Error analyzing FEN: {e}")

def main():
    """Main function to continue debugging IMG_4755."""
    print("🔍 Continuing Debug of IMG_4755")
    print("=" * 50)
    
    # Use the corners you already captured
    corners = [
        [915, 2194],   # a8 (top-left)
        [2736, 2154],  # h8 (top-right)
        [2721, 4113],  # h1 (bottom-right)
        [453, 3927]    # a1 (bottom-left)
    ]
    
    print(f"✅ Using captured corners: {corners}")
    
    image_path = "grey_background_dataset/images/train/IMG_4755.JPG"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Get FEN input
        fen = get_fen_input()
        
        # Analyze the FEN and corners
        analyze_fen_vs_image(fen, corners)
        
        # Show the warped board for verification
        show_warped_board(img, corners, fen)
        
        print(f"\n✅ Debug complete!")
        print(f"📁 Warped board: debug_warped_IMG_4755.png")
        print(f"🔍 Review the warped image to see if:")
        print(f"   1. Squares align correctly with the board")
        print(f"   2. Each square contains the expected piece")
        print(f"   3. The FEN matches what you see in the image")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

