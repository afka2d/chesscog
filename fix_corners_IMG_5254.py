#!/usr/bin/env python3
"""
Fix corners for IMG_5254.JPG specifically.
This script allows you to re-enter the correct corners for this image.
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path

def fix_corners_for_image(image_path, annotation_path):
    """Fix corners for a specific image."""
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    # Load current annotation
    current_corners = []
    current_fen = ""
    if os.path.exists(annotation_path):
        with open(annotation_path, 'r') as f:
            data = json.load(f)
            current_corners = data.get('corners', [])
            current_fen = data.get('fen', '')
    
    print(f"🖼️  Image: {os.path.basename(image_path)}")
    print(f"📏 Image size: {img.shape[1]} x {img.shape[0]}")
    print(f"📍 Current corners: {current_corners}")
    print(f"♟️  Current FEN: {current_fen}")
    print("\n" + "="*60)
    
    # Display image with grid
    display_img = img.copy()
    height, width = img.shape[:2]
    
    # Draw 8x8 grid
    for i in range(9):
        # Vertical lines
        x = int(width * i / 8)
        cv2.line(display_img, (x, 0), (x, height), (0, 255, 0), 2)
        # Horizontal lines
        y = int(height * i / 8)
        cv2.line(display_img, (0, y), (width, y), (0, 255, 0), 2)
    
    # Draw current corners if they exist
    for i, corner in enumerate(current_corners):
        if len(corner) == 2:
            cv2.circle(display_img, (corner[0], corner[1]), 20, (0, 0, 255), -1)
            cv2.putText(display_img, str(i+1), (corner[0]-10, corner[1]+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Show image
    cv2.imshow('IMG_5254.JPG - Click corners in order: TL, TR, BR, BL', display_img)
    cv2.setWindowProperty('IMG_5254.JPG - Click corners in order: TL, TR, BR, BL', cv2.WND_PROP_TOPMOST, 1)
    
    print("🎯 Click the 4 corners in this order:")
    print("   1. Top-Left (a8)")
    print("   2. Top-Right (h8)") 
    print("   3. Bottom-Right (h1)")
    print("   4. Bottom-Left (a1)")
    print("\n💡 Click on the image window, then press any key when done")
    
    # Mouse callback for corner selection
    corners = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            corners.append([x, y])
            print(f"📍 Corner {len(corners)}: ({x}, {y})")
            
            # Draw the corner
            cv2.circle(display_img, (x, y), 15, (255, 0, 0), -1)
            cv2.putText(display_img, str(len(corners)), (x-10, y+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow('IMG_5254.JPG - Click corners in order: TL, TR, BR, BL', display_img)
    
    cv2.setMouseCallback('IMG_5254.JPG - Click corners in order: TL, TR, BR, BL', mouse_callback)
    
    # Wait for 4 corners
    while len(corners) < 4:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            cv2.destroyAllWindows()
            print("❌ Cancelled by user")
            return
    
    cv2.destroyAllWindows()
    
    print(f"\n✅ All 4 corners captured:")
    for i, corner in enumerate(corners):
        print(f"   Corner {i+1}: {corner}")
    
    # Ask for FEN
    print(f"\n♟️  Current FEN: {current_fen}")
    new_fen = input("Enter new FEN (or press Enter to keep current): ").strip()
    if not new_fen:
        new_fen = current_fen
    
    # Save updated annotation
    updated_data = {
        "image": os.path.basename(image_path),
        "corners": corners,
        "fen": new_fen,
        "timestamp": "IMG_5254_FIXED"
    }
    
    with open(annotation_path, 'w') as f:
        json.dump(updated_data, f, indent=2)
    
    print(f"\n✅ Updated annotation saved to: {annotation_path}")
    print(f"📍 New corners: {corners}")
    print(f"♟️  New FEN: {new_fen}")

if __name__ == "__main__":
    # Paths for IMG_5254.JPG
    image_path = "enhanced_training_dataset/images/IMG_5254.JPG"
    annotation_path = "enhanced_training_dataset/annotations/IMG_5254.json"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("Please make sure the image is in the correct location")
    else:
        fix_corners_for_image(image_path, annotation_path)

