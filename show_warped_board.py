#!/usr/bin/env python3
"""
Display the warped board with grid overlay for piece verification.
"""

import cv2
import numpy as np
import os

def show_warped_board():
    """Display the warped board with grid overlay."""
    print("🔍 Displaying warped board with grid overlay...")
    
    # Load the warped board
    warped_path = "debug_outputs/NEW_20250805_135338_002_warped_debug.png"
    
    if not os.path.exists(warped_path):
        print(f"❌ Warped board not found: {warped_path}")
        print(f"   Running debug script first...")
        
        # Run the debug script to generate the warped board
        import subprocess
        subprocess.run(["python3", "debug_piece_extraction.py"])
    
    # Load the warped board
    warped = cv2.imread(warped_path)
    if warped is None:
        print(f"❌ Could not load warped board: {warped_path}")
        return
    
    print(f"✅ Loaded warped board: {warped.shape[1]}x{warped.shape[0]} pixels")
    
    # Create grid overlay
    warped_with_grid = warped.copy()
    target_size = 400
    square_size = target_size // 8
    
    # Draw grid lines
    for i in range(1, 8):
        # Vertical lines
        x = i * square_size
        cv2.line(warped_with_grid, (x, 0), (x, target_size), (0, 255, 0), 2)
        
        # Horizontal lines
        y = i * square_size
        cv2.line(warped_with_grid, (0, y), (target_size, y), (0, 255, 0), 2)
    
    # Add square labels
    files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
    
    for file_idx, file in enumerate(files):
        for rank_idx, rank in enumerate(ranks):
            x = file_idx * square_size + square_size // 2
            y = rank_idx * square_size + square_size // 2
            
            # Add square coordinate label
            label = f"{file}{rank}"
            cv2.putText(warped_with_grid, label, (x-15, y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # Highlight specific squares for verification
    highlight_squares = [
        ('h7', (7, 1)),  # Should contain black king
        ('a4', (0, 4)),  # Should contain white queen
        ('d8', (3, 0)),  # Should contain black rook
        ('f8', (5, 0))   # Should contain black rook
    ]
    
    for square_name, (file_idx, rank_idx) in highlight_squares:
        x1 = file_idx * square_size
        y1 = rank_idx * square_size
        x2 = x1 + square_size
        y2 = y1 + square_size
        
        # Draw red rectangle around highlighted square
        cv2.rectangle(warped_with_grid, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # Add label above the square
        cv2.putText(warped_with_grid, f"CHECK: {square_name}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Save the grid image
    grid_path = "debug_outputs/NEW_20250805_135338_002_grid_with_labels.png"
    cv2.imwrite(grid_path, warped_with_grid)
    print(f"💾 Grid image with labels saved to: {grid_path}")
    
    # Display the images
    print(f"\n🔍 Displaying warped board...")
    print(f"   Green lines: Grid overlay")
    print(f"   Blue text: Square coordinates")
    print(f"   Red rectangles: Squares to verify")
    print(f"   Press any key to continue...")
    
    # Show warped board with grid
    cv2.imshow('NEW_20250805_135338_002 - Warped Board with Grid', warped_with_grid)
    cv2.waitKey(0)
    
    # Also show the original warped board without grid
    cv2.imshow('NEW_20250805_135338_002 - Warped Board (No Grid)', warped)
    print(f"   Now showing warped board without grid...")
    print(f"   Press any key to close...")
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
    print(f"\n✅ Display complete!")
    print(f"🔍 Verification checklist:")
    print(f"   - h7 (top-right area): Should contain a BLACK KING")
    print(f"   - a4 (left side, middle): Should contain a WHITE QUEEN")
    print(f"   - d8 (left side, top): Should contain a BLACK ROOK")
    print(f"   - f8 (right side, top): Should contain a BLACK ROOK")
    print(f"   - Does the board look like a proper chess board?")
    print(f"   - Are pieces clearly visible in their squares?")

if __name__ == "__main__":
    show_warped_board()
