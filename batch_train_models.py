#!/usr/bin/env python3
"""
Batch Training Script for Chess Recognition Models

This script helps you quickly process multiple chess board images and train
the occupancy and piece classifiers on your custom dataset.
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
import cv2
import numpy as np

def create_sample_annotations():
    """Create sample annotations for common chess positions"""
    sample_positions = {
        "IMG_4698.JPG": {
            "fen": "8/8/8/4P3/3p4/8/8/8 w - - 0 1",
            "description": "Black pawn on d4, white pawn on e5"
        },
        "IMG_4540.jpeg": {
            "fen": "8/8/8/8/8/8/8/8 w - - 0 1", 
            "description": "Empty board - good for training empty squares"
        },
        # Add more sample positions as needed
    }
    return sample_positions

def update_annotation_with_fen(image_name, fen, description=""):
    """Update an annotation file with the correct FEN notation"""
    annotation_path = f"custom_training_data/annotations/{image_name.replace('.JPG', '.json').replace('.jpg', '.json').replace('.jpeg', '.json')}"
    
    if not os.path.exists(annotation_path):
        print(f"Warning: Annotation file {annotation_path} not found")
        return False
    
    try:
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        
        data["fen"] = fen
        data["notes"] = f"{description}. Corner coordinates need to be updated based on the annotated image."
        
        with open(annotation_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Updated {annotation_path} with FEN: {fen}")
        return True
    except Exception as e:
        print(f"❌ Error updating {annotation_path}: {e}")
        return False

def batch_update_annotations():
    """Update multiple annotation files with sample FEN notations"""
    sample_positions = create_sample_annotations()
    
    print("🔄 Updating annotation files with sample FEN notations...")
    
    for image_name, position_data in sample_positions.items():
        update_annotation_with_fen(
            image_name, 
            position_data["fen"], 
            position_data["description"]
        )
    
    print("\n📝 Next steps:")
    print("1. Review the annotated images in custom_training_data/annotations/")
    print("2. Update corner coordinates in the JSON files")
    print("3. Add FEN notations for the remaining images")
    print("4. Run validation and training")

def validate_annotations():
    """Validate all annotation files"""
    print("🔍 Validating annotations...")
    
    result = subprocess.run([
        "python", "create_custom_dataset.py",
        "--input_dir", "custom_training_data/images",
        "--output_dir", "custom_training_data",
        "--validate"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ All annotations are valid!")
        return True
    else:
        print("❌ Validation failed:")
        print(result.stderr)
        return False

def train_models():
    """Train the occupancy and piece classifiers"""
    print("🚀 Starting model training...")
    
    result = subprocess.run([
        "python", "train_custom_models.py",
        "--input_dir", "custom_training_data",
        "--output_dir", "custom_training_results"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Training completed successfully!")
        print("📊 Results saved in custom_training_results/")
        return True
    else:
        print("❌ Training failed:")
        print(result.stderr)
        return False

def show_dataset_summary():
    """Show a summary of the current dataset"""
    images_dir = Path("custom_training_data/images")
    annotations_dir = Path("custom_training_data/annotations")
    
    if not images_dir.exists():
        print("❌ Images directory not found")
        return
    
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.JPG"))
    annotation_files = list(annotations_dir.glob("*.json"))
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Images: {len(image_files)}")
    print(f"   Annotations: {len(annotation_files)}")
    
    if annotation_files:
        print(f"\n📝 Sample annotations:")
        for i, ann_file in enumerate(annotation_files[:5]):  # Show first 5
            try:
                with open(ann_file, 'r') as f:
                    data = json.load(f)
                print(f"   {ann_file.name}: {data.get('fen', 'No FEN')}")
            except:
                print(f"   {ann_file.name}: Error reading")
    
    print(f"\n🎯 Ready for training: {len(image_files) >= 5}")

def main():
    parser = argparse.ArgumentParser(description="Batch train chess recognition models")
    parser.add_argument("--update-fen", action="store_true", 
                       help="Update annotation files with sample FEN notations")
    parser.add_argument("--validate", action="store_true",
                       help="Validate all annotation files")
    parser.add_argument("--train", action="store_true",
                       help="Train the models")
    parser.add_argument("--summary", action="store_true",
                       help="Show dataset summary")
    parser.add_argument("--full-pipeline", action="store_true",
                       help="Run the complete pipeline: update FEN, validate, train")
    
    args = parser.parse_args()
    
    if args.full_pipeline:
        print("🔄 Running complete training pipeline...")
        batch_update_annotations()
        if validate_annotations():
            train_models()
        return
    
    if args.update_fen:
        batch_update_annotations()
    
    if args.validate:
        validate_annotations()
    
    if args.train:
        train_models()
    
    if args.summary:
        show_dataset_summary()
    
    if not any([args.update_fen, args.validate, args.train, args.summary, args.full_pipeline]):
        print("🎯 Chess Recognition Model Training")
        print("=" * 40)
        print("\nAvailable commands:")
        print("  --update-fen     Update annotations with sample FEN notations")
        print("  --validate       Validate all annotation files")
        print("  --train          Train the models")
        print("  --summary        Show dataset summary")
        print("  --full-pipeline  Run complete pipeline")
        print("\nQuick start:")
        print("  python batch_train_models.py --full-pipeline")

if __name__ == "__main__":
    main() 