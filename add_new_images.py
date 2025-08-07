#!/usr/bin/env python3
"""
Script to add new images to the chess dataset
"""

import os
import json
import shutil
from pathlib import Path
import argparse
from datetime import datetime

def create_annotation_template(image_name, fen="8/8/8/8/8/8/8/8 w - - 0 1"):
    """Create a template annotation JSON."""
    return {
        "image": image_name,
        "corners": [
            [],  # Top-Left (a8)
            [],  # Top-Right (h8)
            [],  # Bottom-Right (h1)
            []   # Bottom-Left (a1)
        ],
        "fen": fen
    }

def add_images_to_dataset(new_images_dir, dataset_dir="grey_background_dataset", 
                         train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Add new images to the dataset with proper train/val/test split.
    
    Args:
        new_images_dir: Directory containing new images
        dataset_dir: Target dataset directory
        train_ratio: Percentage for training (default 70%)
        val_ratio: Percentage for validation (default 20%)
        test_ratio: Percentage for testing (default 10%)
    """
    
    dataset_path = Path(dataset_dir)
    new_images_path = Path(new_images_dir)
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        raise ValueError("Train, val, and test ratios must sum to 1.0")
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(new_images_path.glob(f"*{ext}"))
    
    if not image_files:
        print(f"No images found in {new_images_dir}")
        return
    
    print(f"Found {len(image_files)} images to add")
    
    # Shuffle and split images
    import random
    random.shuffle(image_files)
    
    total_images = len(image_files)
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)
    test_count = total_images - train_count - val_count
    
    train_images = image_files[:train_count]
    val_images = image_files[train_count:train_count + val_count]
    test_images = image_files[train_count + val_count:]
    
    print(f"Split: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    # Process each split
    splits = [
        ("train", train_images),
        ("val", val_images),
        ("test", test_images)
    ]
    
    for split_name, images in splits:
        if not images:
            continue
            
        print(f"\nProcessing {split_name} split...")
        
        # Create directories if they don't exist
        images_dir = dataset_path / "images" / split_name
        annotations_dir = dataset_path / "annotations" / split_name
        
        images_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images and create annotations
        for i, image_path in enumerate(images):
            # Generate new filename to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"NEW_{timestamp}_{i:03d}{image_path.suffix}"
            
            # Copy image
            dest_image_path = images_dir / new_filename
            shutil.copy2(image_path, dest_image_path)
            
            # Create annotation
            annotation = create_annotation_template(new_filename)
            annotation_path = annotations_dir / f"{new_filename.rsplit('.', 1)[0]}.json"
            
            with open(annotation_path, 'w') as f:
                json.dump(annotation, f, indent=2)
            
            print(f"  Added: {new_filename}")
    
    print(f"\nSuccessfully added {total_images} images to dataset!")
    print(f"Next steps:")
    print(f"1. Run the corner corrector: python interactive_corner_corrector.py")
    print(f"2. Set the correct FEN strings for each position")
    print(f"3. Verify corner coordinates are accurate")

def main():
    parser = argparse.ArgumentParser(description="Add new images to chess dataset")
    parser.add_argument("new_images_dir", help="Directory containing new images")
    parser.add_argument("--dataset", default="grey_background_dataset", 
                       help="Target dataset directory")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                       help="Training set ratio (default: 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                       help="Validation set ratio (default: 0.2)")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                       help="Test set ratio (default: 0.1)")
    
    args = parser.parse_args()
    
    add_images_to_dataset(
        args.new_images_dir,
        args.dataset,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    )

if __name__ == "__main__":
    main() 