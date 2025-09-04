#!/usr/bin/env python3
"""
Diagnose the corner coordinates for NEW_20250805_135338_002.JPG
"""

import cv2
import numpy as np
import os

def diagnose_corners():
    """Display the image with current corners to diagnose the issue."""
    print("🔍 Diagnosing corner coordinates for NEW_20250805_135338_002...")
    
    # Image path
    image_path = "grey_background_dataset/images/test/NEW_20250805_135338_002.JPG"
    
    # Current corners from annotation
    current_corners = [
        [536, 1894],   # a8 (top-left)
        [2726, 1818],  # h8 (top-right)
        [2866, 4130],  # h1 (bottom-right)
        [359, 4101]    # a1 (bottom-left)
    ]
    
    # Original corners (before manual adjustment)
    original_corners = [
        [536, 1882],   # a8 (top-left)
        [2718, 1822],  # h8 (top-right)
        [2858, 4146],  # h1 (bottom-right)
        [356, 4088]    # a1 (bottom-left)
    ]
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not read image {image_path}")
        return
    
    # Get image dimensions
    height, width = image.shape[:2]
    print(f"📐 Image dimensions: {width}x{height} pixels")
    
    # Create a copy for drawing
    image_with_corners = image.copy()
    
    # Draw current corners (blue)
    for i, corner in enumerate(current_corners):
        x, y = corner
        cv2.circle(image_with_corners, (x, y), 30, (255, 0, 0), -1)  # Blue circle
        cv2.putText(image_with_corners, f"{i+1}", (x+35, y+35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
    
    # Draw original corners (red) for comparison
    for i, corner in enumerate(original_corners):
        x, y = corner
        cv2.circle(image_with_corners, (x, y), 20, (0, 0, 255), -1)  # Red circle
        cv2.putText(image_with_corners, f"O{i+1}", (x+25, y+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Add corner labels
    corner_labels = ['a8', 'h8', 'h1', 'a1']
    for i, (corner, label) in enumerate(zip(current_corners, corner_labels)):
        x, y = corner
        cv2.putText(image_with_corners, label, (x-50, y-50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    
    # Draw lines connecting corners to show the board outline
    for i in range(4):
        pt1 = tuple(current_corners[i])
        pt2 = tuple(current_corners[(i + 1) % 4])
        cv2.line(image_with_corners, pt1, pt2, (0, 255, 0), 3)  # Green lines
    
    # Save diagnostic image
    debug_path = "debug_outputs/NEW_20250805_135338_002_corners_diagnostic.png"
    os.makedirs("debug_outputs", exist_ok=True)
    cv2.imwrite(debug_path, image_with_corners)
    print(f"💾 Diagnostic image saved to: {debug_path}")
    
    # Display image
    print(f"\n🔍 Displaying image with corners...")
    print(f"   Blue circles: Current corners (manually adjusted)")
    print(f"   Red circles: Original corners (before adjustment)")
    print(f"   Green lines: Board outline from current corners")
    print(f"   Press any key to close...")
    
    # Resize if too large for display
    display_img = image_with_corners
    if width > 1200:
        scale = 1200 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        display_img = cv2.resize(image_with_corners, (new_width, new_height))
        print(f"   📏 Resized for display: {new_width}x{new_height}")
    
    cv2.imshow('NEW_20250805_135338_002 - Corner Diagnostic', display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Analysis
    print(f"\n📊 Corner Analysis:")
    print(f"   Current corners: {current_corners}")
    print(f"   Original corners: {original_corners}")
    
    # Calculate board dimensions
    board_width = max(current_corners[1][0], current_corners[2][0]) - min(current_corners[0][0], current_corners[3][0])
    board_height = max(current_corners[2][1], current_corners[3][1]) - min(current_corners[0][1], current_corners[1][1])
    print(f"   Board dimensions: {board_width}x{board_height} pixels")
    
    # Check if corners form a reasonable rectangle
    print(f"\n🔍 Visual Assessment:")
    print(f"   - Do the blue circles (current corners) look correct?")
    print(f"   - Does the green outline look like a proper chess board?")
    print(f"   - Are the corners at the actual board corners?")
    print(f"   - Is the board reasonably square-shaped?")

if __name__ == "__main__":
    diagnose_corners()
