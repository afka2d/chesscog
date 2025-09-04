#!/usr/bin/env python3
"""
Verify the IMG_4752.JPG fix by showing the warped board and extracted pieces.
"""

import os
import cv2
import numpy as np

def show_warped_board():
    """Display the warped board with grid overlay."""
    print("🔍 Showing warped board...")
    
    warped_path = "debug_outputs/IMG_4752_warped.png"
    if not os.path.exists(warped_path):
        print(f"❌ Warped board not found: {warped_path}")
        return
    
    # Load warped board
    warped = cv2.imread(warped_path)
    if warped is None:
        print(f"❌ Could not load warped board")
        return
    
    # Create a copy for drawing
    display = warped.copy()
    
    # Add grid overlay
    height, width = display.shape[:2]
    square_size = height // 8
    
    # Draw vertical lines
    for i in range(9):
        x = i * square_size
        cv2.line(display, (x, 0), (x, height), (0, 255, 0), 2)
    
    # Draw horizontal lines
    for i in range(9):
        y = i * square_size
        cv2.line(display, (0, y), (width, y), (0, 255, 0), 2)
    
    # Add square labels
    for rank in range(8):
        for file in range(8):
            x = file * square_size + square_size // 2
            y = rank * square_size + square_size // 2
            
            # Calculate square name
            file_name = chr(ord('a') + file)
            rank_name = str(8 - rank)
            square_name = file_name + rank_name
            
            # Add label
            cv2.putText(display, square_name, (x-15, y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # Save grid overlay
    grid_path = "debug_outputs/IMG_4752_warped_grid.png"
    cv2.imwrite(grid_path, display)
    print(f"   💾 Saved grid overlay: {grid_path}")
    
    # Display the image
    cv2.imshow('IMG_4752 Warped Board with Grid', display)
    print(f"   📱 Displaying warped board with grid overlay")
    print(f"   Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_extracted_pieces():
    """Display the extracted piece images."""
    print(f"\n🔍 Showing extracted piece images...")
    
    pieces_dir = "re_extracted_IMG_4752"
    if not os.path.exists(pieces_dir):
        print(f"❌ Pieces directory not found: {pieces_dir}")
        return
    
    # Get all piece images
    piece_files = [f for f in os.listdir(pieces_dir) if f.endswith('.png')]
    piece_files.sort()  # Sort for consistent order
    
    if not piece_files:
        print(f"❌ No piece images found in {pieces_dir}")
        return
    
    print(f"   📁 Found {len(piece_files)} piece images")
    
    # Create a grid display
    grid_size = int(np.ceil(np.sqrt(len(piece_files))))
    cell_size = 100
    
    # Create grid image
    grid_width = grid_size * cell_size
    grid_height = grid_size * cell_size
    grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 200
    
    for i, piece_file in enumerate(piece_files):
        piece_path = os.path.join(pieces_dir, piece_file)
        piece_img = cv2.imread(piece_path)
        
        if piece_img is not None:
            # Resize piece to fit in grid cell
            piece_resized = cv2.resize(piece_img, (cell_size, cell_size))
            
            # Calculate grid position
            row = i // grid_size
            col = i % grid_size
            
            # Place piece in grid
            y1 = row * cell_size
            y2 = y1 + cell_size
            x1 = col * cell_size
            x2 = x1 + cell_size
            
            grid[y1:y2, x1:x2] = piece_resized
            
            # Add filename label
            label = piece_file.replace('.png', '')
            cv2.putText(grid, label, (x1+5, y1+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # Save grid
    grid_path = "debug_outputs/IMG_4752_pieces_grid.png"
    cv2.imwrite(grid_path, grid)
    print(f"   💾 Saved pieces grid: {grid_path}")
    
    # Display grid
    cv2.imshow('IMG_4752 Extracted Pieces Grid', grid)
    print(f"   📱 Displaying pieces grid")
    print(f"   Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_individual_pieces():
    """Show each piece image individually for detailed inspection."""
    print(f"\n🔍 Showing individual pieces...")
    
    pieces_dir = "re_extracted_IMG_4752"
    if not os.path.exists(pieces_dir):
        print(f"❌ Pieces directory not found: {pieces_dir}")
        return
    
    piece_files = [f for f in os.listdir(pieces_dir) if f.endswith('.png')]
    piece_files.sort()
    
    print(f"   📁 Will show {len(piece_files)} pieces individually")
    print(f"   Press any key to advance to next piece...")
    
    for i, piece_file in enumerate(piece_files):
        piece_path = os.path.join(pieces_dir, piece_file)
        piece_img = cv2.imread(piece_path)
        
        if piece_img is not None:
            # Create display with filename
            display = piece_img.copy()
            cv2.putText(display, piece_file, (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(f'Piece {i+1}/{len(piece_files)}', display)
            print(f"   📸 Showing {piece_file} ({i+1}/{len(piece_files)})")
            
            cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    print(f"   ✅ Finished showing all pieces")

def main():
    """Main function to verify the IMG_4752 fix."""
    print("🔍 IMG_4752.JPG Fix Verification")
    print("=" * 50)
    
    try:
        # Step 1: Show warped board
        print(f"\n🎯 STEP 1: Verify Warped Board")
        show_warped_board()
        
        # Step 2: Show pieces grid
        print(f"\n🎯 STEP 2: Verify Pieces Grid")
        show_extracted_pieces()
        
        # Step 3: Show individual pieces
        print(f"\n🎯 STEP 3: Verify Individual Pieces")
        show_individual_pieces()
        
        print(f"\n✅ Verification complete!")
        print(f"\n💡 If everything looks good:")
        print(f"   1. The warped board should be square and properly aligned")
        print(f"   2. The piece images should be clear and properly cropped")
        print(f"   3. You can proceed to replace the dataset pieces")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
