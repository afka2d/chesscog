#!/usr/bin/env python3
"""
Show the e8 piece specifically for verification.
"""

import matplotlib.pyplot as plt
import cv2

def show_e8_piece():
    """Show the e8 piece to verify if it's a king or pawn."""
    
    # Load the e8 piece image
    e8_path = "re_extracted_IMG_4755/IMG_4755_e8.png"
    
    try:
        img = cv2.imread(e8_path)
        if img is None:
            print(f"❌ Could not load image: {e8_path}")
            return
        
        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Display the piece
        plt.figure(figsize=(6, 12))
        plt.imshow(img_rgb)
        plt.title('IMG_4755 - Square e8\nAccording to FEN: Black King (k)\nIs this actually a king or a pawn?', fontsize=14)
        plt.axis('off')
        
        print("🔍 Review the e8 piece:")
        print("   - According to your FEN, e8 should contain a black king (k)")
        print("   - Look at the piece image - does it look like a king or a pawn?")
        print("   - If it looks like a pawn, the FEN needs correction")
        print("   - If it looks like a king, the original issue was elsewhere")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    show_e8_piece()

