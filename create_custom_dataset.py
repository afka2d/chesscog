#!/usr/bin/env python3
"""
Custom dataset creation script for training chess recognition models.

This script helps you create training data from real chess board images.
You'll need to manually annotate the corners and FEN for each image.

Usage:
1. Place your chess board images in a folder
2. Run this script to create annotation files
3. Manually annotate the corners and FEN for each image
4. Run the training scripts
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import argparse
from PIL import Image, ImageDraw, ImageFont
import chess

def create_annotation_template(image_path, output_dir):
    """Create an annotation template for a chess board image."""
    
    # Load the image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    
    # Create annotation template
    annotation = {
        "image_path": str(image_path),
        "image_size": [width, height],
        "corners": [
            [0, 0],      # Top-left corner (you need to annotate this)
            [width, 0],  # Top-right corner
            [width, height],  # Bottom-right corner
            [0, height]  # Bottom-left corner
        ],
        "fen": "8/8/8/8/8/8/8/8 w - - 0 1",  # Empty board (you need to annotate this)
        "white_turn": True,
        "notes": "Please annotate the 4 corner points and provide the FEN notation"
    }
    
    # Save annotation template
    annotation_path = output_dir / f"{image_path.stem}.json"
    with open(annotation_path, 'w') as f:
        json.dump(annotation, f, indent=2)
    
    print(f"Created annotation template: {annotation_path}")
    print(f"Please edit this file to add the correct corner coordinates and FEN notation")
    
    return annotation_path

def create_corner_annotation_tool(image_path, annotation_path):
    """Create a simple tool to help annotate corners visually."""
    
    # Load image
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load annotation
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    
    # Create a copy for drawing
    img_draw = img.copy()
    
    # Draw corner points
    corners = annotation["corners"]
    for i, corner in enumerate(corners):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(img_draw, (x, y), 10, (255, 0, 0), -1)  # Blue circle
        cv2.putText(img_draw, str(i+1), (x+15, y+15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Draw lines connecting corners
    for i in range(4):
        pt1 = tuple(map(int, corners[i]))
        pt2 = tuple(map(int, corners[(i+1) % 4]))
        cv2.line(img_draw, pt1, pt2, (0, 255, 0), 2)  # Green lines
    
    # Save annotated image
    annotated_path = annotation_path.parent / f"{image_path.stem}_annotated.jpg"
    cv2.imwrite(str(annotated_path), cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))
    
    print(f"Created annotated image: {annotated_path}")
    print("Use this image to help you identify the correct corner coordinates")
    
    return annotated_path

def validate_annotation(annotation_path):
    """Validate that an annotation file has been properly filled out."""
    
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    
    # Check if corners are still default values
    corners = annotation["corners"]
    image_size = annotation["image_size"]
    
    # Check if corners are at image boundaries (likely not annotated)
    if (corners[0] == [0, 0] and 
        corners[1] == [image_size[0], 0] and
        corners[2] == [image_size[0], image_size[1]] and
        corners[3] == [0, image_size[1]]):
        return False, "Corners appear to be default values - please annotate them"
    
    # Check if FEN is still empty board
    if annotation["fen"] == "8/8/8/8/8/8/8/8 w - - 0 1":
        return False, "FEN is still empty board - please provide the actual position"
    
    # Validate FEN
    try:
        board = chess.Board(annotation["fen"])
    except ValueError as e:
        return False, f"Invalid FEN: {e}"
    
    return True, "Annotation is valid"

def create_dataset_structure(output_dir):
    """Create the directory structure for the custom dataset."""
    
    # Create main directories
    for subset in ["train", "val", "test"]:
        for subdir in ["images", "annotations"]:
            (output_dir / subset / subdir).mkdir(parents=True, exist_ok=True)
    
    print(f"Created dataset structure in: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Create custom chess dataset for training")
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="Directory containing chess board images")
    parser.add_argument("--output_dir", type=str, default="custom_dataset",
                       help="Output directory for the dataset")
    parser.add_argument("--create_templates", action="store_true",
                       help="Create annotation templates for all images")
    parser.add_argument("--validate", action="store_true",
                       help="Validate existing annotations")
    parser.add_argument("--create_annotated_images", action="store_true",
                       help="Create annotated images to help with corner detection")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        return
    
    # Create dataset structure
    create_dataset_structure(output_dir)
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} image files")
    
    if args.create_templates:
        print("\nCreating annotation templates...")
        for image_path in image_files:
            annotation_path = create_annotation_template(
                image_path, 
                output_dir / "annotations"
            )
    
    if args.create_annotated_images:
        print("\nCreating annotated images...")
        annotation_files = list(output_dir.glob("annotations/*.json"))
        for annotation_path in annotation_files:
            # Find corresponding image
            image_name = annotation_path.stem
            image_path = None
            for img_file in image_files:
                if img_file.stem == image_name:
                    image_path = img_file
                    break
            
            if image_path:
                create_corner_annotation_tool(image_path, annotation_path)
    
    if args.validate:
        print("\nValidating annotations...")
        annotation_files = list(output_dir.glob("annotations/*.json"))
        valid_count = 0
        
        for annotation_path in annotation_files:
            is_valid, message = validate_annotation(annotation_path)
            if is_valid:
                print(f"✓ {annotation_path.name}: {message}")
                valid_count += 1
            else:
                print(f"✗ {annotation_path.name}: {message}")
        
        print(f"\nValidation complete: {valid_count}/{len(annotation_files)} annotations are valid")
    
    print(f"\nDataset creation complete!")
    print(f"Next steps:")
    print(f"1. Edit the annotation files in {output_dir}/annotations/")
    print(f"2. Add corner coordinates and FEN notation for each image")
    print(f"3. Run validation again to ensure all annotations are correct")
    print(f"4. Use the training scripts to train the models")

if __name__ == "__main__":
    main() 