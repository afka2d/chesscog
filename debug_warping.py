#!/usr/bin/env python3
"""
Debug board warping process
"""

import cv2
import numpy as np
import json
from pathlib import Path
from chesscog.occupancy_classifier.create_dataset import warp_chessboard_image
from chesscog.core import sort_corner_points

def debug_warping():
    """Debug the board warping process."""
    
    # Load image
    img_path = "grey_background_dataset/images/test/IMG_4752.JPG"
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"❌ Could not load image: {img_path}")
        return
    
    print(f"✅ Loaded image: {img.shape}")
    
    # Load manual corners
    annotation_path = "grey_background_dataset/annotations/test/IMG_4752.json"
    with open(annotation_path, 'r') as f:
        data = json.load(f)
        corners = data['corners']
    
    print(f"✅ Manual corners: {corners}")
    
    # Convert to numpy array and sort
    corners_array = np.array(corners, dtype=np.float32)
    corners_array = sort_corner_points(corners_array)
    
    print(f"✅ Sorted corners: {corners_array}")
    
    # Warp the board
    warped_board = warp_chessboard_image(img, corners_array)
    
    print(f"✅ Warped board shape: {warped_board.shape}")
    
    # Save the warped board
    cv2.imwrite("debug_warped_board.png", warped_board)
    print("✅ Saved debug_warped_board.png")
    
    # Test occupancy on a few squares
    square_size = warped_board.shape[0] // 8
    
    # Test center squares (likely to have pieces)
    test_squares = [(3, 3), (4, 4), (3, 4), (4, 3)]  # e4, e5, d4, d5
    
    for rank, file in test_squares:
        x1 = file * square_size
        y1 = rank * square_size
        x2 = x1 + square_size
        y2 = y1 + square_size
        
        square_img = warped_board[y1:y2, x1:x2]
        
        # Save the square
        square_filename = f"debug_square_{chr(97+file)}{8-rank}.png"
        cv2.imwrite(square_filename, square_img)
        print(f"✅ Saved {square_filename} (shape: {square_img.shape})")
        
        # Calculate average brightness to see if it looks like a piece
        avg_brightness = np.mean(square_img)
        print(f"   Average brightness: {avg_brightness:.1f}")
        
        # Check if it's mostly white (empty) or has content (occupied)
        if avg_brightness > 200:
            print(f"   Likely: EMPTY (bright)")
        else:
            print(f"   Likely: OCCUPIED (dark)")

if __name__ == "__main__":
    debug_warping() 