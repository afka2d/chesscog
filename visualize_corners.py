#!/usr/bin/env python3
"""
Script to visualize corner coordinates on chess board images.
This helps you see exactly where the corner coordinates are located.
"""

import cv2
import json
import os
import argparse
from pathlib import Path

def visualize_corners(image_path, annotation_path=None, corners=None):
    """
    Visualize corner coordinates on a chess board image.
    
    Args:
        image_path: Path to the image file
        annotation_path: Path to the annotation JSON file (optional)
        corners: List of corner coordinates as [x1,y1 x2,y2 x3,y3 x4,y4] (optional)
    """
    
    # Load the image
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"📸 Loaded image: {image_path}")
    print(f"   Size: {img.shape[1]}x{img.shape[0]} pixels")
    
    # Get corner coordinates
    corner_coords = None
    
    if corners:
        # Parse corner coordinates from command line
        try:
            corner_parts = corners.split()
            if len(corner_parts) == 4:
                corner_coords = []
                for part in corner_parts:
                    x, y = map(int, part.split(','))
                    corner_coords.append([x, y])
            else:
                print("❌ Invalid corner format. Use: x1,y1 x2,y2 x3,y3 x4,y4")
                return
        except Exception as e:
            print(f"❌ Error parsing corners: {e}")
            return
    
    elif annotation_path:
        # Load from annotation file
        if not os.path.exists(annotation_path):
            print(f"❌ Annotation file not found: {annotation_path}")
            return
        
        try:
            with open(annotation_path, 'r') as f:
                annotation = json.load(f)
            
            if 'corners' in annotation:
                corner_coords = annotation['corners']
                print(f"📄 Loaded corners from annotation: {annotation_path}")
            else:
                print("❌ No corners found in annotation file")
                return
        except Exception as e:
            print(f"❌ Error loading annotation: {e}")
            return
    
    else:
        print("❌ No corner coordinates provided")
        return
    
    # Create a copy for visualization
    vis_img = img.copy()
    
    # Draw corner points and labels
    corner_names = ['Top-Left (a8)', 'Top-Right (h8)', 'Bottom-Right (h1)', 'Bottom-Left (a1)']
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # Green, Red, Blue, Yellow
    
    print("\n🎯 Corner Coordinates:")
    for i, (corner, name, color) in enumerate(zip(corner_coords, corner_names, colors)):
        x, y = corner
        print(f"   {i+1}. {name}: ({x}, {y})")
        
        # Draw circle at corner
        cv2.circle(vis_img, (x, y), 15, color, -1)
        cv2.circle(vis_img, (x, y), 15, (255, 255, 255), 2)  # White border
        
        # Draw label
        label = f"{i+1}: ({x},{y})"
        cv2.putText(vis_img, label, (x + 20, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Draw lines connecting corners to show the board outline
    for i in range(4):
        pt1 = tuple(corner_coords[i])
        pt2 = tuple(corner_coords[(i + 1) % 4])
        cv2.line(vis_img, pt1, pt2, (255, 255, 255), 2)
    
    # Resize image for display if it's too large
    height, width = vis_img.shape[:2]
    max_display_size = 1200
    
    if max(height, width) > max_display_size:
        scale = max_display_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        vis_img = cv2.resize(vis_img, (new_width, new_height))
        print(f"\n📏 Resized for display: {new_width}x{new_height}")
    
    # Save the visualization
    output_path = f"corner_visualization_{Path(image_path).stem}.jpg"
    cv2.imwrite(output_path, vis_img)
    print(f"\n💾 Saved visualization: {output_path}")
    
    # Show the image
    print("\n🖼️  Opening image viewer...")
    os.system(f"open {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Visualize corner coordinates on chess board images")
    parser.add_argument("image", help="Path to the chess board image")
    parser.add_argument("--annotation", help="Path to annotation JSON file")
    parser.add_argument("--corners", help="Corner coordinates as 'x1,y1 x2,y2 x3,y3 x4,y4'")
    
    args = parser.parse_args()
    
    if not args.annotation and not args.corners:
        print("❌ Please provide either --annotation or --corners")
        return
    
    visualize_corners(args.image, args.annotation, args.corners)

if __name__ == "__main__":
    main() 