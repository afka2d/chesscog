#!/usr/bin/env python3
"""
Display the newly corrected pieces for verification.
"""

import cv2
import os

def show_corrected_pieces():
    """Show the newly corrected pieces for verification."""
    print("🔍 Displaying newly corrected pieces for verification...")
    
    # Directory with corrected pieces
    pieces_dir = "re_extracted_NEW_20250805_135338_002_corrected"
    
    if not os.path.exists(pieces_dir):
        print(f"❌ Corrected pieces directory not found: {pieces_dir}")
        return
    
    # Key pieces to verify
    key_pieces = [
        ('e8', 'Black King', 'NEW_20250805_135338_002_e8.png'),
        ('a4', 'White Queen', 'NEW_20250805_135338_002_a4.png'),
        ('a8', 'Black Rook', 'NEW_20250805_135338_002_a8.png'),
        ('e7', 'Black Rook', 'NEW_20250805_135338_002_e7.png'),
        ('g6', 'Black Queen', 'NEW_20250805_135338_002_g6.png'),
        ('e5', 'White Bishop', 'NEW_20250805_135338_002_e5.png'),
        ('f4', 'White Rook', 'NEW_20250805_135338_002_f4.png'),
        ('g2', 'White King', 'NEW_20250805_135338_002_g2.png')
    ]
    
    print(f"🔍 Will display {len(key_pieces)} key pieces...")
    print("   Each piece will be shown at its original size")
    print("   Press any key to advance to the next piece")
    print()
    
    displayed_count = 0
    
    for square, piece_type, filename in key_pieces:
        piece_path = os.path.join(pieces_dir, filename)
        
        if os.path.exists(piece_path):
            piece_img = cv2.imread(piece_path)
            if piece_img is not None:
                # Get dimensions
                height, width = piece_img.shape[:2]
                
                # Display piece
                cv2.imshow(f'{square} - {piece_type} ({width}x{height})', piece_img)
                print(f"   📸 {square}: {piece_type} - {width}x{height} pixels")
                print(f"      Press any key to continue...")
                
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                displayed_count += 1
            else:
                print(f"   ❌ Could not load: {filename}")
        else:
            print(f"   ❌ File not found: {filename}")
        
        print()
    
    print(f"✅ Display complete!")
    print(f"🎯 Successfully displayed: {displayed_count}/{len(key_pieces)} pieces")
    
    print(f"\n🔍 Quality Assessment:")
    print(f"   - Are pieces clearly visible and centered?")
    print(f"   - Do they match what you see on the board?")
    print(f"   - Are they properly cropped to their squares?")
    print(f"   - No overlapping pieces or board edges?")

if __name__ == "__main__":
    show_corrected_pieces()
