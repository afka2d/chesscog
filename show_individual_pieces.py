#!/usr/bin/env python3
"""
Display each individual piece image from NEW_20250805_135338_002 for proper verification.
"""

import cv2
import os
from pathlib import Path

def show_piece(piece_path, square, piece_type):
    """Display a single piece image."""
    if os.path.exists(piece_path):
        piece_img = cv2.imread(piece_path)
        if piece_img is not None:
            # Get original dimensions
            height, width = piece_img.shape[:2]
            
            # Display original size
            cv2.imshow(f'{square} - {piece_type} ({width}x{height})', piece_img)
            print(f"   📸 {square}: {piece_type} - {width}x{height} pixels")
            print(f"      Press any key to continue to next piece...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return True
        else:
            print(f"   ❌ Could not load image: {piece_path}")
            return False
    else:
        print(f"   ❌ File not found: {piece_path}")
        return False

def main():
    """Main function to display individual pieces."""
    print("🔍 Individual Piece Verification for NEW_20250805_135338_002")
    print("=" * 70)
    
    # All pieces to display
    pieces_to_show = [
        # Rank 8 (top row)
        ('d8', 'Black Rook', 'NEW_20250805_135338_002_d8.png'),
        ('f8', 'Black Rook', 'NEW_20250805_135338_002_f8.png'),
        
        # Rank 7
        ('d7', 'Black Bishop', 'NEW_20250805_135338_002_d7.png'),
        ('g7', 'Black Pawn', 'NEW_20250805_135338_002_g7.png'),
        ('h7', 'Black King', 'NEW_20250805_135338_002_h7.png'),
        
        # Rank 6
        ('b6', 'Black Pawn', 'NEW_20250805_135338_002_b6.png'),
        ('d6', 'Black Bishop', 'NEW_20250805_135338_002_d6.png'),
        ('g6', 'Black Queen', 'NEW_20250805_135338_002_g6.png'),
        
        # Rank 5
        ('a5', 'Black Pawn', 'NEW_20250805_135338_002_a5.png'),
        ('c5', 'Black Pawn', 'NEW_20250805_135338_002_c5.png'),
        ('d5', 'White Knight', 'NEW_20250805_135338_002_d5.png'),
        ('f5', 'Black Pawn', 'NEW_20250805_135338_002_f5.png'),
        ('h5', 'Black Pawn', 'NEW_20250805_135338_002_h5.png'),
        
        # Rank 4
        ('a4', 'White Queen', 'NEW_20250805_135338_002_a4.png'),
        ('c4', 'White Pawn', 'NEW_20250805_135338_002_c4.png'),
        ('e4', 'Black Knight', 'NEW_20250805_135338_002_e4.png'),
        ('g4', 'Black Pawn', 'NEW_20250805_135338_002_g4.png'),
        ('h4', 'White Pawn', 'NEW_20250805_135338_002_h4.png'),
        
        # Rank 3
        ('b3', 'White Pawn', 'NEW_20250805_135338_002_b3.png'),
        ('d3', 'White Pawn', 'NEW_20250805_135338_002_d3.png'),
        ('f3', 'Black Knight', 'NEW_20250805_135338_002_f3.png'),
        
        # Rank 2
        ('b2', 'White Bishop', 'NEW_20250805_135338_002_b2.png'),
        ('g2', 'White Knight', 'NEW_20250805_135338_002_g2.png'),
        
        # Rank 1 (bottom row)
        ('b1', 'White King', 'NEW_20250805_135338_002_b1.png'),
        ('d1', 'White Rook', 'NEW_20250805_135338_002_d1.png'),
        ('f1', 'White Rook', 'NEW_20250805_135338_002_f1.png'),
        ('h1', 'White Bishop', 'NEW_20250805_135338_002_h1.png')
    ]
    
    print(f"🔍 Will display {len(pieces_to_show)} individual pieces...")
    print("   Each piece will be shown at its original size")
    print("   Press any key to advance to the next piece")
    print("   Close any window to stop the process")
    print()
    
    try:
        displayed_count = 0
        
        for square, piece_type, filename in pieces_to_show:
            piece_path = f"re_extracted_NEW_20250805_135338_002/{filename}"
            
            print(f"🔍 Showing piece {displayed_count + 1}/{len(pieces_to_show)}...")
            if show_piece(piece_path, square, piece_type):
                displayed_count += 1
            else:
                print(f"   ⚠️  Skipped {square}")
            
            print()
        
        print(f"✅ Display complete!")
        print(f"🎯 Successfully displayed: {displayed_count}/{len(pieces_to_show)} pieces")
        
        if displayed_count == len(pieces_to_show):
            print(f"🎉 All pieces displayed successfully!")
        else:
            print(f"⚠️  Some pieces could not be displayed")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Display interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
