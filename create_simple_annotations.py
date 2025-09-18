#!/usr/bin/env python3
"""
Simple manual annotation tool for creating ground truth data.
This creates a template that you can fill in manually.
"""

import json
import os
from pathlib import Path

def create_annotation_template(image_path):
    """Create a template annotation file for an image"""
    print(f"\nCreating annotation template for: {Path(image_path).name}")
    
    # Create empty board template
    squares = []
    for rank in range(8, 0, -1):  # 8 to 1
        for file in range(8):  # a to h
            square_name = f"{chr(97+file)}{rank}"
            squares.append(square_name)
    
    # Create template with all squares empty
    template = {}
    for square in squares:
        template[square] = {
            "occupied": False,
            "color": None,
            "piece": None
        }
    
    # Save template
    annotation_file = image_path.replace('.JPG', '.json').replace('.jpg', '.json')
    with open(annotation_file, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Template saved to: {annotation_file}")
    print("\nTo annotate this image:")
    print("1. Open the JSON file in a text editor")
    print("2. For each occupied square, change:")
    print("   - 'occupied': true")
    print("   - 'color': 'white' or 'black'")
    print("   - 'piece': 'pawn', 'rook', 'knight', 'bishop', 'queen', or 'king'")
    print("3. Save the file")
    
    return annotation_file

def main():
    """Main function"""
    print("Simple Ground Truth Annotation Tool")
    print("=" * 40)
    
    # Find images
    dataset_path = "my_chess_images/train/images"
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
        return
    
    # Find images
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    images = []
    for ext in image_extensions:
        images.extend(Path(dataset_path).glob(f"**/*{ext}"))
    
    if not images:
        print("No images found to annotate")
        return
    
    print(f"Found {len(images)} images")
    
    # Ask how many to process
    max_images = input("How many images to create templates for? (default: 3): ").strip()
    if not max_images:
        max_images = 3
    else:
        max_images = int(max_images)
    
    # Process images
    for i, image_path in enumerate(images[:max_images]):
        print(f"\n--- Image {i+1}/{min(max_images, len(images))} ---")
        create_annotation_template(str(image_path))
    
    print(f"\n✅ Created {min(max_images, len(images))} annotation templates")
    print("Next steps:")
    print("1. Edit the JSON files to add piece annotations")
    print("2. Run: python comprehensive_accuracy_evaluation.py")

if __name__ == "__main__":
    main()
