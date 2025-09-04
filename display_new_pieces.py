#!/usr/bin/env python3
"""
Display the newly generated piece images from NEW_20250805_135338_002 for verification.
"""

import cv2
import numpy as np
import os
from pathlib import Path

def display_piece_grid(pieces_to_show):
    """Display a grid of piece images for verification."""
    print("🖼️  Displaying newly generated piece images for verification...")
    print("=" * 70)
    
    # Create a grid layout
    cols = 4
    rows = (len(pieces_to_show) + cols - 1) // cols
    
    # Calculate grid dimensions
    piece_size = 100
    grid_width = cols * piece_size
    grid_height = rows * piece_size
    
    # Create grid image
    grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 240  # Light gray background
    
    for i, (square, piece_type, filename) in enumerate(pieces_to_show):
        # Calculate position in grid
        row = i // cols
        col = i % cols
        
        # Load piece image
        piece_path = f"re_extracted_NEW_20250805_135338_002/{filename}"
        if os.path.exists(piece_path):
            piece_img = cv2.imread(piece_path)
            if piece_img is not None:
                # Resize to fit grid
                piece_resized = cv2.resize(piece_img, (piece_size, piece_size))
                
                # Calculate position
                y1 = row * piece_size
                y2 = y1 + piece_size
                x1 = col * piece_size
                x2 = x1 + piece_size
                
                # Place in grid
                grid[y1:y2, x1:x2] = piece_resized
                
                # Add label
                label = f"{square}\n{piece_type}"
                cv2.putText(grid, label, (x1 + 5, y1 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
                print(f"   ✅ {square}: {piece_type} ({filename})")
            else:
                print(f"   ❌ Could not load: {filename}")
        else:
            print(f"   ❌ File not found: {filename}")
    
    # Save grid image
    grid_path = "debug_outputs/NEW_20250805_135338_002_pieces_grid.png"
    os.makedirs("debug_outputs", exist_ok=True)
    cv2.imwrite(grid_path, grid)
    print(f"\n💾 Piece grid saved to: {grid_path}")
    
    # Display grid
    cv2.imshow('NEW_20250805_135338_002 - Newly Generated Pieces', grid)
    print("\n🔍 Review the piece images above.")
    print("   Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return grid_path

def display_warped_board():
    """Display the warped board for verification."""
    print("\n🔍 Displaying warped board for perspective verification...")
    
    warped_path = "debug_outputs/NEW_20250805_135338_002_warped.png"
    if os.path.exists(warped_path):
        warped_img = cv2.imread(warped_path)
        if warped_img is not None:
            # Resize for display if too large
            height, width = warped_img.shape[:2]
            if width > 800:
                scale = 800 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                warped_img = cv2.resize(warped_img, (new_width, new_height))
            
            cv2.imshow('NEW_20250805_135338_002 - Warped Board (Perspective Corrected)', warped_img)
            print("   ✅ Warped board displayed")
            print("   🔍 Verify that the board looks square and pieces are properly aligned")
            print("   Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("   ❌ Could not load warped board image")
    else:
        print("   ❌ Warped board image not found")

def main():
    """Main function to display new piece images."""
    print("🔍 Verification of NEW_20250805_135338_002 Piece Images")
    print("=" * 70)
    
    # Key pieces to display for verification
    pieces_to_show = [
        # Rank 8 (top row)
        ('d8', 'Black Rook', 'NEW_20250805_135338_002_d8.png'),
        ('f8', 'Black Rook', 'NEW_20250805_135338_002_f8.png'),
        
        # Rank 7
        ('d7', 'Black Bishop', 'NEW_20250805_135338_002_d7.png'),
        ('h7', 'Black King', 'NEW_20250805_135338_002_h7.png'),
        
        # Rank 6
        ('g6', 'Black Queen', 'NEW_20250805_135338_002_g6.png'),
        ('b6', 'Black Pawn', 'NEW_20250805_135338_002_b6.png'),
        
        # Rank 5
        ('d5', 'White Knight', 'NEW_20250805_135338_002_d5.png'),
        ('a5', 'Black Pawn', 'NEW_20250805_135338_002_a5.png'),
        
        # Rank 4
        ('a4', 'White Queen', 'NEW_20250805_135338_002_a4.png'),
        ('e4', 'Black Knight', 'NEW_20250805_135338_002_e4.png'),
        
        # Rank 3
        ('f3', 'Black Knight', 'NEW_20250805_135338_002_f3.png'),
        ('b3', 'White Pawn', 'NEW_20250805_135338_002_b3.png'),
        
        # Rank 2
        ('b2', 'White Bishop', 'NEW_20250805_135338_002_b2.png'),
        ('g2', 'White Knight', 'NEW_20250805_135338_002_g2.png'),
        
        # Rank 1 (bottom row)
        ('b1', 'White King', 'NEW_20250805_135338_002_b1.png'),
        ('d1', 'White Rook', 'NEW_20250805_135338_002_d1.png'),
        ('f1', 'White Rook', 'NEW_20250805_135338_002_f1.png'),
        ('h1', 'White Bishop', 'NEW_20250805_135338_002_h1.png')
    ]
    
    try:
        # Step 1: Display piece grid
        print("🔍 Step 1: Displaying piece images...")
        grid_path = display_piece_grid(pieces_to_show)
        
        # Step 2: Display warped board
        print("\n🔍 Step 2: Displaying warped board...")
        display_warped_board()
        
        print(f"\n✅ Verification complete!")
        print(f"🖼️  Piece grid saved to: {grid_path}")
        print(f"🔍 Warped board saved to: debug_outputs/NEW_20250805_135338_002_warped.png")
        
        print(f"\n🎯 Quality Assessment:")
        print(f"   - Each piece should be clearly visible and centered")
        print(f"   - No overlapping pieces or board edges")
        print(f"   - Pieces should be properly aligned to their squares")
        print(f"   - Warped board should look like a perfect square")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
