#!/usr/bin/env python3
"""
Corner Coordinate Update Tool for Chess Recognition Training

This tool helps you update corner coordinates for chess board images.
It provides an interactive way to view annotated images and update JSON files.
"""

import os
import json
import argparse
from pathlib import Path
import cv2
import numpy as np

def show_image_info(image_path):
    """Show basic information about an image"""
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            height, width = img.shape[:2]
            print(f"📏 Image dimensions: {width} x {height} pixels")
            return width, height
    return None, None

def list_annotated_images():
    """List all annotated images available"""
    # Try multiple possible dataset directories
    possible_dirs = [
        "grey_background_dataset/annotations",
        "custom_training_data/annotations",
        "annotations"
    ]
    
    annotated_images = []
    for annotations_dir in possible_dirs:
        if os.path.exists(annotations_dir):
            # Look for JSON annotation files
            json_files = list(Path(annotations_dir).glob("*.json"))
            if json_files:
                annotated_images = json_files
                break
    
    if not annotated_images:
        print("❌ No annotated images found in any dataset directory")
        print("Available directories checked:")
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                print(f"  - {dir_path}")
        return []
    
    print("🖼️  Available annotated images:")
    for i, json_path in enumerate(annotated_images, 1):
        base_name = json_path.stem
        print(f"  {i:2d}. {base_name}")
    
    return annotated_images

def show_corner_guide():
    """Show a guide for corner coordinates"""
    print("""
🎯 CORNER COORDINATE GUIDE
==========================

The corners array should contain the 4 outer corners of the chess board (not the image corners).
The order should be: [top-left, top-right, bottom-right, bottom-left]

Example for a 1000x1000 image with a chess board in the center:
{
  "corners": [
    [200, 200],   // Top-left corner of chess board
    [800, 200],   // Top-right corner of chess board  
    [800, 800],   // Bottom-right corner of chess board
    [200, 800]    // Bottom-left corner of chess board
  ]
}

IMPORTANT NOTES:
- Use the annotated images (*_annotated.jpg) to see the current corner positions
- The red dots show the current corner coordinates
- Update the coordinates to match the actual chess board corners
- The chess board corners are the 4 outer corners of the 8x8 grid
- Don't use the image corners unless the chess board fills the entire image
""")

def update_corner_coordinates(image_name, new_corners):
    """Update corner coordinates for a specific image"""
    # Try multiple possible dataset directories
    possible_dirs = [
        "grey_background_dataset/annotations",
        "custom_training_data/annotations",
        "annotations"
    ]
    
    json_path = None
    for annotations_dir in possible_dirs:
        test_path = f"{annotations_dir}/{image_name}.json"
        if os.path.exists(test_path):
            json_path = test_path
            break
    
    if not json_path:
        print(f"❌ Annotation file not found for {image_name}")
        print("Checked directories:")
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                print(f"  - {dir_path}")
        return False
    
    # Load current annotation
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Update corners
    data["corners"] = new_corners
    data["notes"] = "Corner coordinates updated manually"
    
    # Save updated annotation
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Updated corner coordinates for {image_name}")
    return True

def validate_corners(corners, image_width, image_height):
    """Validate that corner coordinates are reasonable"""
    if len(corners) != 4:
        print("❌ Must have exactly 4 corners")
        return False
    
    for i, corner in enumerate(corners):
        if len(corner) != 2:
            print(f"❌ Corner {i} must have 2 coordinates (x, y)")
            return False
        
        x, y = corner
        if x < 0 or x > image_width or y < 0 or y > image_height:
            print(f"⚠️  Corner {i} ({x}, {y}) is outside image bounds ({image_width}x{image_height})")
    
    # Check if corners form a reasonable rectangle
    # (This is a basic check - you might want more sophisticated validation)
    print("✅ Corner coordinates look reasonable")
    return True

def interactive_update():
    """Interactive corner coordinate update"""
    print("🎯 Interactive Corner Coordinate Update")
    print("=" * 50)
    
    # List available images
    annotated_images = list_annotated_images()
    
    if not annotated_images:
        print("❌ No annotated images found")
        return
    
    # Get user selection
    try:
        selection = input("\nEnter the number of the image to update (or 'q' to quit): ")
        if selection.lower() == 'q':
            return
        
        selection_idx = int(selection) - 1
        if selection_idx < 0 or selection_idx >= len(annotated_images):
            print("❌ Invalid selection")
            return
        
        selected_json = annotated_images[selection_idx]
        base_name = selected_json.stem
        
        print(f"\n📸 Selected: {base_name}")
        
        # Show image info
        # Try multiple possible image directories
        possible_image_dirs = [
            "grey_background_dataset/images",
            "custom_training_data/images",
            "images"
        ]
        
        original_image_path = None
        for image_dir in possible_image_dirs:
            for ext in ['.jpg', '.jpeg', '.JPG', '.png', '.PNG']:
                test_path = f"{image_dir}/{base_name}{ext}"
                if os.path.exists(test_path):
                    original_image_path = test_path
                    break
            if original_image_path:
                break
        
        width, height = show_image_info(original_image_path)
        if width and height:
            print(f"📏 Image size: {width} x {height}")
        
        # Show current corners
        # Try multiple possible annotation directories
        possible_annotation_dirs = [
            "grey_background_dataset/annotations",
            "custom_training_data/annotations",
            "annotations"
        ]
        
        json_path = None
        for annotation_dir in possible_annotation_dirs:
            test_path = f"{annotation_dir}/{base_name}.json"
            if os.path.exists(test_path):
                json_path = test_path
                break
        
        if json_path and os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            print(f"\n📍 Current corner coordinates:")
            for i, corner in enumerate(data["corners"]):
                print(f"  Corner {i+1}: ({corner[0]}, {corner[1]})")
        
        # Get new corner coordinates
        print(f"\n🎯 Enter new corner coordinates for {base_name}")
        print("Format: x1,y1 x2,y2 x3,y3 x4,y4")
        print("Order: top-left, top-right, bottom-right, bottom-left")
        
        corners_input = input("New corners: ")
        try:
            # Parse corner coordinates
            corner_pairs = corners_input.strip().split()
            if len(corner_pairs) != 4:
                print("❌ Must provide exactly 4 corner coordinates")
                return
            
            new_corners = []
            for pair in corner_pairs:
                x, y = map(int, pair.split(','))
                new_corners.append([x, y])
            
            # Validate corners
            if validate_corners(new_corners, width, height):
                # Update the file
                if update_corner_coordinates(base_name, new_corners):
                    print(f"✅ Successfully updated {base_name}")
                else:
                    print(f"❌ Failed to update {base_name}")
        
        except ValueError as e:
            print(f"❌ Invalid coordinate format: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    except ValueError:
        print("❌ Invalid input")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

def batch_update_example():
    """Show example of batch updating corner coordinates"""
    print("""
📝 BATCH UPDATE EXAMPLE
=======================

You can update multiple corner coordinates at once by editing the JSON files directly.
Here's an example for IMG_4698.JPG:

Current (incorrect):
{
  "corners": [
    [0, 0],           // Image corner
    [3240, 0],        // Image corner  
    [3240, 5760],     // Image corner
    [0, 5760]         // Image corner
  ]
}

Updated (correct - example):
{
  "corners": [
    [500, 300],       // Top-left of chess board
    [2740, 300],      // Top-right of chess board
    [2740, 5460],     // Bottom-right of chess board
    [500, 5460]       // Bottom-left of chess board
  ]
}

To update multiple files:
1. Open each *_annotated.jpg file to see current corners
2. Identify the actual chess board corners
3. Update the corresponding .json file
4. Run validation to check your work
""")

def main():
    parser = argparse.ArgumentParser(description="Update corner coordinates for chess board images")
    parser.add_argument("--guide", action="store_true", help="Show corner coordinate guide")
    parser.add_argument("--list", action="store_true", help="List all annotated images")
    parser.add_argument("--interactive", action="store_true", help="Interactive corner update")
    parser.add_argument("--example", action="store_true", help="Show batch update example")
    parser.add_argument("--update", nargs=2, metavar=("IMAGE", "CORNERS"), 
                       help="Update corners for specific image (format: 'x1,y1 x2,y2 x3,y3 x4,y4')")
    
    args = parser.parse_args()
    
    if args.guide:
        show_corner_guide()
    elif args.list:
        list_annotated_images()
    elif args.interactive:
        interactive_update()
    elif args.example:
        batch_update_example()
    elif args.update:
        image_name, corners_str = args.update
        try:
            corner_pairs = corners_str.strip().split()
            new_corners = []
            for pair in corner_pairs:
                x, y = map(int, pair.split(','))
                new_corners.append([x, y])
            
            if update_corner_coordinates(image_name, new_corners):
                print(f"✅ Updated {image_name}")
            else:
                print(f"❌ Failed to update {image_name}")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("🎯 Corner Coordinate Update Tool")
        print("\nAvailable commands:")
        print("  --guide       Show corner coordinate guide")
        print("  --list        List all annotated images")
        print("  --interactive Interactive corner update")
        print("  --example     Show batch update example")
        print("  --update      Update specific image corners")
        print("\nExample usage:")
        print("  python update_corners.py --interactive")
        print("  python update_corners.py --update IMG_4698 '500,300 2740,300 2740,5460 500,5460'")

if __name__ == "__main__":
    main() 