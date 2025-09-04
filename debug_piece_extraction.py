#!/usr/bin/env python3
"""
Debug the piece extraction process for NEW_20250805_135338_002 to find where it's going wrong.
"""

import cv2
import numpy as np
import os
import chess

def debug_piece_extraction():
    """Debug the piece extraction process step by step."""
    print("🔍 Debugging piece extraction process...")
    
    # Image path
    image_path = "grey_background_dataset/images/test/NEW_20250805_135338_002.JPG"
    
    # Current corners
    corners = [
        [536, 1894],   # a8 (top-left)
        [2726, 1818],  # h8 (top-right)
        [2866, 4130],  # h1 (bottom-right)
        [359, 4101]    # a1 (bottom-left)
    ]
    
    # FEN
    fen = "3r1r2/3b2pk/1p1b2q1/p1pN1p1p/Q1P1n1pP/1P1P1n2/1B4N1/1K1R1R1B w - - 0 1"
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not read image {image_path}")
        return
    
    print(f"📐 Original image: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Step 1: Convert corners to numpy array
    corners_np = np.array(corners, dtype=np.float32)
    print(f"📐 Corners: {corners}")
    
    # Step 2: Define target corners (perfect square)
    target_size = 400
    target_corners = np.array([
        [0, 0],                    # a8 (top-left)
        [target_size, 0],          # h8 (top-right)
        [target_size, target_size], # h1 (bottom-right)
        [0, target_size]           # a1 (bottom-left)
    ], dtype=np.float32)
    
    print(f"📐 Target size: {target_size}x{target_size}")
    
    # Step 3: Calculate perspective transform
    matrix = cv2.getPerspectiveTransform(corners_np, target_corners)
    print(f"✅ Perspective transform matrix calculated")
    
    # Step 4: Apply perspective transform
    warped = cv2.warpPerspective(image, matrix, (target_size, target_size))
    print(f"✅ Image warped to {warped.shape[1]}x{warped.shape[0]}")
    
    # Save warped image
    warped_path = "debug_outputs/NEW_20250805_135338_002_warped_debug.png"
    cv2.imwrite(warped_path, warped)
    print(f"💾 Warped image saved to: {warped_path}")
    
    # Step 5: Parse FEN
    board = chess.Board(fen)
    print(f"✅ FEN parsed: {fen}")
    
    # Step 6: Extract a specific piece for debugging (d8 - Black Rook)
    square_size = target_size // 8
    print(f"📐 Square size: {square_size}x{square_size} pixels")
    
    # Extract d8 (file 3, rank 0)
    file, rank = 3, 0  # d8
    x1 = file * square_size
    y1 = rank * square_size
    x2 = x1 + square_size
    y2 = y1 + square_size
    
    print(f"🔍 Extracting d8: file={file}, rank={rank}")
    print(f"   Square boundaries: ({x1},{y1}) to ({x2},{y2})")
    
    # Extract the square
    square_img = warped[y1:y2, x1:x2]
    print(f"✅ Square extracted: {square_img.shape[1]}x{square_img.shape[0]}")
    
    # Save the extracted square
    square_path = "debug_outputs/NEW_20250805_135338_002_d8_square_debug.png"
    cv2.imwrite(square_path, square_img)
    print(f"💾 Square image saved to: {square_path}")
    
    # Step 7: Show the warped board with grid lines
    warped_with_grid = warped.copy()
    
    # Draw grid lines
    for i in range(1, 8):
        # Vertical lines
        x = i * square_size
        cv2.line(warped_with_grid, (x, 0), (x, target_size), (0, 255, 0), 2)
        
        # Horizontal lines
        y = i * square_size
        cv2.line(warped_with_grid, (0, y), (target_size, y), (0, 255, 0), 2)
    
    # Highlight d8 square
    cv2.rectangle(warped_with_grid, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(warped_with_grid, "d8", (x1+5, y1+25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Save grid image
    grid_path = "debug_outputs/NEW_20250805_135338_002_grid_debug.png"
    cv2.imwrite(grid_path, warped_with_grid)
    print(f"💾 Grid image saved to: {grid_path}")
    
    # Step 8: Display images for verification
    print(f"\n🔍 Displaying debug images...")
    print(f"   Press any key to advance through each image...")
    
    # Show warped board
    cv2.imshow('Warped Board (Debug)', warped)
    cv2.waitKey(0)
    
    # Show warped board with grid
    cv2.imshow('Warped Board with Grid (Debug)', warped_with_grid)
    cv2.waitKey(0)
    
    # Show extracted square
    cv2.imshow('Extracted d8 Square (Debug)', square_img)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
    print(f"\n✅ Debug complete!")
    print(f"📁 Debug files saved:")
    print(f"   - Warped board: {warped_path}")
    print(f"   - Grid overlay: {grid_path}")
    print(f"   - d8 square: {square_path}")
    
    print(f"\n🔍 Analysis:")
    print(f"   - Does the warped board look like a proper chess board?")
    print(f"   - Are the grid lines evenly spaced?")
    print(f"   - Does the d8 square contain a black rook?")
    print(f"   - Is the square properly centered on the piece?")

if __name__ == "__main__":
    debug_piece_extraction()
