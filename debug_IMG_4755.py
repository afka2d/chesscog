#!/usr/bin/env python3
"""
Debug IMG_4755 by showing the image with grid overlay and manual input.
This will help identify what's causing the wrong piece labels.
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
        
        # Show the warped board
        cv2.imshow('Warped Board - Verify squares align correctly', warped)
        cv2.waitKey(1)
        
        print("\n🔍 Verify the warped board:")
        print("   - Squares should be perfectly aligned")
        print("   - Each square should contain one piece")
        print("   - The grid should be 8x8")
        
        input("Press Enter to continue...")
        cv2.destroyAllWindows()
        
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
    """Main function to debug IMG_4755."""
    print("🔍 Debugging IMG_4755 - Manual Corner and FEN Input")
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
        
        # Analyze the FEN and corners
        analyze_fen_vs_image(fen, corners)
        
        # Show the warped board for verification
        show_warped_board(img, corners, fen)
        
        print(f"\n✅ Debug complete!")
        print(f"📁 Warped board saved to: debug_warped_IMG_4755.png")
        print(f"🔍 Review the warped image to see if squares align correctly")
        print(f"📊 Check if the FEN matches what you see in the image")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

