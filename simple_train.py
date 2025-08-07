#!/usr/bin/env python3
"""
Simple Training Script for Chess Recognition Models

This script directly trains the models using your custom dataset without complex preprocessing.
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
import argparse
from PIL import Image
import chess

def load_annotations(annotations_dir):
    """Load all annotation files"""
    annotations = {}
    for json_file in Path(annotations_dir).glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            annotations[json_file.stem] = data
    return annotations

def validate_dataset(images_dir, annotations_dir):
    """Validate that the dataset is ready for training"""
    print("🔍 Validating dataset...")
    
    annotations = load_annotations(annotations_dir)
    image_files = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.jpeg")) + list(Path(images_dir).glob("*.JPG"))
    
    print(f"📊 Found {len(image_files)} images and {len(annotations)} annotations")
    
    # Check that each image has an annotation
    missing_annotations = []
    for img_file in image_files:
        if img_file.stem not in annotations:
            missing_annotations.append(img_file.name)
    
    if missing_annotations:
        print(f"⚠️  Missing annotations for: {missing_annotations}")
        return False
    
    # Check that annotations have required fields
    invalid_annotations = []
    for name, ann in annotations.items():
        if "corners" not in ann or "fen" not in ann:
            invalid_annotations.append(name)
    
    if invalid_annotations:
        print(f"⚠️  Invalid annotations for: {invalid_annotations}")
        return False
    
    print("✅ Dataset validation passed!")
    return True

def create_training_data(images_dir, annotations_dir, output_dir):
    """Create training data in the format expected by the training scripts"""
    print("🔄 Creating training data...")
    
    annotations = load_annotations(annotations_dir)
    
    # Create output directory structure
    train_dir = Path(output_dir) / "train"
    val_dir = Path(output_dir) / "val"
    test_dir = Path(output_dir) / "test"
    
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Split data (simple split: 70% train, 20% val, 10% test)
    image_files = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.jpeg")) + list(Path(images_dir).glob("*.JPG"))
    
    # Sort for reproducible splits
    image_files.sort()
    
    n_total = len(image_files)
    n_train = int(0.7 * n_total)
    n_val = int(0.2 * n_total)
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    print(f"📊 Data split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    # Process each split
    for split_name, files, split_dir in [("train", train_files, train_dir), 
                                        ("val", val_files, val_dir), 
                                        ("test", test_files, test_dir)]:
        print(f"📝 Processing {split_name} split...")
        
        for img_file in files:
            img_name = img_file.stem
            if img_name not in annotations:
                continue
                
            ann = annotations[img_name]
            
            # Copy image
            target_img = split_dir / f"{img_name}.png"
            try:
                img = Image.open(img_file)
                img.save(target_img)
            except Exception as e:
                print(f"⚠️  Error processing {img_name}: {e}")
                continue
            
            # Create annotation file
            target_ann = split_dir / f"{img_name}.json"
            with open(target_ann, 'w') as f:
                json.dump(ann, f, indent=2)
    
    print("✅ Training data created successfully!")
    return output_dir

def run_training(output_dir):
    """Run the actual training using the existing training scripts"""
    print("🚀 Starting training...")
    
    # Set up data paths for the training scripts
    os.environ["RENDERS_DIR"] = str(Path(output_dir))
    
    try:
        # Train occupancy classifier
        print("🎯 Training occupancy classifier...")
        occupancy_result = os.system("python -m chesscog.occupancy_classifier.train --config ResNet")
        
        if occupancy_result != 0:
            print("❌ Occupancy classifier training failed")
            return False
        
        # Train piece classifier
        print("🎯 Training piece classifier...")
        piece_result = os.system("python -m chesscog.piece_classifier.train --config ResNet")
        
        if piece_result != 0:
            print("❌ Piece classifier training failed")
            return False
        
        print("✅ Training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Simple training for chess recognition models")
    parser.add_argument("--images_dir", default="custom_training_data/images", help="Directory containing images")
    parser.add_argument("--annotations_dir", default="custom_training_data/annotations", help="Directory containing annotations")
    parser.add_argument("--output_dir", default="training_output", help="Output directory for training data")
    parser.add_argument("--skip_validation", action="store_true", help="Skip dataset validation")
    parser.add_argument("--skip_training", action="store_true", help="Skip model training")
    
    args = parser.parse_args()
    
    print("🎯 Simple Chess Recognition Model Training")
    print("=" * 50)
    
    # Validate dataset
    if not args.skip_validation:
        if not validate_dataset(args.images_dir, args.annotations_dir):
            print("❌ Dataset validation failed")
            sys.exit(1)
    
    # Create training data
    training_data_dir = create_training_data(args.images_dir, args.annotations_dir, args.output_dir)
    
    # Run training
    if not args.skip_training:
        success = run_training(training_data_dir)
        if not success:
            print("❌ Training failed")
            sys.exit(1)
    
    print(f"\n🎉 Process completed! Training data saved in {args.output_dir}")
    print("📁 You can now use the trained models with your chess recognition API")

if __name__ == "__main__":
    main() 