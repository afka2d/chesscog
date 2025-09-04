#!/usr/bin/env python3
"""
Debug IMG_4762 by showing the image with grid overlay and manual corner input.
This will help fix the distorted piece extraction.
"""

import os
import cv2
import numpy as np
import json
import chess
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def display_image_with_grid_matplotlib(image_path):
    """Display image with 8x8 grid overlay using matplotlib."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_rgb)
    
    # Draw 8x8 grid
    for i in range(9):
        # Vertical lines
        x = (width * i) / 8
        ax.axvline(x=x, color='blue', linewidth=2)
        
        # Horizontal lines
        y = (height * i) / 8
        ax.axhline(y=y, color='blue', linewidth=2)
    
    # Add coordinate labels
    for rank in range(8):
        for file in range(8):
            x = (width * file) / 8 + width/16
            y = (height * rank) / 8 + height/16
            label = f"{chr(97+file)}{8-rank}"
            ax.text(x, y, label, color='green', fontsize=10, ha='center', va='center')
    
    # Set title
    ax.set_title('IMG_4762 - Click corners in order: a8, h8, h1, a1', fontsize=14)
    ax.axis('off')
    
    # Save the grid image
    grid_path = "debug_IMG_4762_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    print(f"💾 Grid image saved to: {grid_path}")
    
    # Get corner clicks
    corners = []
    
    def onclick(event):
        if event.inaxes != ax:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        corners.append([x, y])
        
        # Draw circle on clicked point
        circle = patches.Circle((x, y), 20, fill=False, color='red', linewidth=3)
        ax.add_patch(circle)
        
        # Add corner number
        ax.text(x+30, y+30, f"{len(corners)}", color='red', fontsize=12, weight='bold')
        
        print(f"   Corner {len(corners)}: ({x}, {y})")
        
        # Redraw
        plt.draw()
        
        if len(corners) == 4:
            print("✅ All 4 corners captured!")
    
    # Connect the click event
    fig.canvas.mpl_connect('button_press_event', onclick)
    
    print(f"\n📸 Processing: IMG_4762.JPG")
    print("🔍 Click the four corners in this order:")
    print("   1. Top-left (a8) - White's perspective")
    print("   2. Top-right (h8)")
    print("   3. Bottom-right (h1)")
    print("   4. Bottom-left (a1)")
    print("   Click each corner, then close the window when done")
    
    # Show the plot
    plt.show()
    
    if len(corners) == 4:
        print(f"✅ All corners captured: {corners}")
        return img, corners
    else:
        print(f"⚠️  Expected 4 corners, got {len(corners)}")
        return img, []

def get_fen_input():
    """Get FEN input from user."""
    print(f"\n♟️  Enter the CORRECT FEN for IMG_4762:")
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

def show_warped_board(img, corners, fen):
    """Show the warped board to verify the transformation."""
    try:
        # Parse FEN to get piece positions
        board = chess.Board(fen)
        
        # Warp the chessboard to get a square grid
        warped = warp_chessboard(img, corners)
        
        # Save warped image for inspection
        warped_path = "debug_warped_IMG_4762.png"
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
        
        # Highlight the problematic square b8
        b8_square = chess.square(1, 0)  # b8 = file 1, rank 0
        b8_piece = board.piece_at(b8_square)
        if b8_piece:
            print(f"\n⚠️  Square b8 (file 1, rank 0): {b8_piece.symbol()} ({b8_piece.color})")
            print(f"   This should be extracted as: {b8_piece.symbol()}")
            print(f"   Current test data shows this as a black knight, but you said it looks like a pawn")
        else:
            print(f"\n⚠️  Square b8 (file 1, rank 0): empty")
        
    except Exception as e:
        print(f"❌ Error analyzing FEN: {e}")

def main():
    """Main function to debug IMG_4762."""
    print("🔍 Debugging IMG_4762 - Manual Corner and FEN Input")
    print("=" * 60)
    
    image_path = "grey_background_dataset/images/train/IMG_4762.JPG"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    try:
        # Get corners by clicking
        img, corners = display_image_with_grid_matplotlib(image_path)
        
        if not corners:
            print("❌ No corners captured - cannot continue")
            return
        
        # Get FEN input
        fen = get_fen_input()
        
        # Analyze the FEN and corners
        analyze_fen_vs_image(fen, corners)
        
        # Show the warped board for verification
        show_warped_board(img, corners, fen)
        
        print(f"\n✅ Debug complete!")
        print(f"📁 Grid image: debug_IMG_4762_grid.png")
        print(f"📁 Warped board: debug_warped_IMG_4762.png")
        print(f"🔍 Review both images to see if:")
        print(f"   1. Grid overlay matches the actual board")
        print(f"   2. Warped squares align correctly")
        print(f"   3. FEN matches what you see in the image")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

