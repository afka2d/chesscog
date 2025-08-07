#!/usr/bin/env python3
"""
Enhanced Batch Training Script for Chess Recognition Models

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
    """Create sample annotations for common chess positions based on image descriptions"""
    sample_positions = {
        "IMG_4698": {
            "fen": "8/8/8/4P3/3p4/8/8/8 w - - 0 1",
            "description": "Black pawn on d4, white pawn on e5"
        },
        "IMG_4540": {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 
            "description": "Standard starting position - all pieces"
        },
        "IMG_4545": {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "description": "Standard starting position - all pieces"
        },
        "IMG_4546": {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "description": "Standard starting position - all pieces"
        },
        "IMG_4547": {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "description": "Standard starting position - all pieces"
        },
        "IMG_4549": {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "description": "Standard starting position - all pieces"
        },
        "IMG_4558": {
            "fen": "8/8/8/4P3/3p4/8/8/8 w - - 0 1",
            "description": "Black pawn on d4, white pawn on e5"
        },
        "IMG_4565": {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "description": "Standard starting position - all pieces"
        },
        "IMG_4567": {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "description": "Standard starting position - all pieces"
        },
        "IMG_4572": {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "description": "Standard starting position - all pieces"
        },
        "IMG_4573": {
            "fen": "8/8/8/4P3/3p4/8/8/8 w - - 0 1",
            "description": "Black pawn on d4, white pawn on e5"
        },
        "IMG_4575": {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "description": "Standard starting position - all pieces"
        },
        "IMG_4579": {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "description": "Standard starting position - all pieces"
        },
        "IMG_4587": {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "description": "Standard starting position - all pieces"
        },
        "sample": {
            "fen": "8/8/6P1/8/4p3/p7/4Q1r1/8 w - - 0 1",
            "description": "Complex position with 5 pieces: white queen on e2, white pawn on g6, black pawns on a3 and e4, black rook on g2"
        }
    }
    return sample_positions

def update_annotations_with_fen():
    """Update annotation files with sample FEN notations"""
    print("🔄 Updating annotations with sample FEN notations...")
    
    sample_positions = create_sample_annotations()
    annotations_dir = "custom_training_data/annotations"
    
    updated_count = 0
    for json_file in Path(annotations_dir).glob("*.json"):
        if json_file.stem in sample_positions:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Update FEN
            data["fen"] = sample_positions[json_file.stem]["fen"]
            data["description"] = sample_positions[json_file.stem]["description"]
            data["notes"] = f"Sample FEN: {sample_positions[json_file.stem]['description']}. Corner coordinates still need to be updated."
            
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✅ Updated {json_file.name} with FEN: {sample_positions[json_file.stem]['fen']}")
            updated_count += 1
    
    print(f"📝 Updated {updated_count} annotation files with sample FEN notations")
    return updated_count

def validate_dataset():
    """Validate the current dataset"""
    print("🔍 Validating dataset...")
    
    annotations_dir = "custom_training_data/annotations"
    images_dir = "custom_training_data/images"
    
    # Count files
    annotation_files = list(Path(annotations_dir).glob("*.json"))
    image_files = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.jpeg")) + list(Path(images_dir).glob("*.JPG"))
    
    print(f"📊 Dataset Summary:")
    print(f"   - Total images: {len(image_files)}")
    print(f"   - Total annotations: {len(annotation_files)}")
    
    # Check for missing annotations
    annotated_images = {f.stem for f in annotation_files}
    image_names = {f.stem for f in image_files}
    missing_annotations = image_names - annotated_images
    
    if missing_annotations:
        print(f"   ⚠️  Missing annotations for: {missing_annotations}")
    else:
        print(f"   ✅ All images have annotations")
    
    # Check FEN completeness
    fen_count = 0
    for json_file in annotation_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            if data.get("fen") and data["fen"] != "8/8/8/8/8/8/8/8 w - - 0 1":
                fen_count += 1
    
    print(f"   - Images with FEN: {fen_count}/{len(annotation_files)}")
    
    return len(image_files), len(annotation_files), fen_count

def train_models():
    """Train the models using the current dataset"""
    print("🚀 Starting model training...")
    
    try:
        # Run the quick training script
        result = subprocess.run([
            "python", "quick_train.py", 
            "--epochs", "10",
            "--batch_size", "8",
            "--learning_rate", "0.001"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            print(result.stdout)
        else:
            print("❌ Training failed!")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error during training: {e}")

def create_training_summary():
    """Create a summary of the training dataset"""
    print("📋 Creating training dataset summary...")
    
    total_images, total_annotations, fen_count = validate_dataset()
    
    summary = f"""
# 🎯 Enhanced Chess Training Dataset Summary

## 📊 Dataset Statistics
- **Total Images:** {total_images}
- **Total Annotations:** {total_annotations}
- **Images with FEN:** {fen_count}
- **Coverage:** {fen_count/total_annotations*100:.1f}%

## 🖼️ Image Types
- **Starting Positions:** Multiple images of standard chess setup
- **Game Positions:** Various mid-game positions with pieces
- **Empty Boards:** Some images for training empty square detection

## 🚀 Next Steps
1. **Review Annotated Images:** Check `custom_training_data/annotations/*_annotated.jpg`
2. **Update Corner Coordinates:** Edit JSON files with precise corner coordinates
3. **Validate Annotations:** Run validation to ensure accuracy
4. **Train Models:** Use the training scripts to create improved models

## 📁 Key Files
- **Images:** `custom_training_data/images/`
- **Annotations:** `custom_training_data/annotations/`
- **Annotated Images:** `custom_training_data/annotations/*_annotated.jpg`
- **Training Script:** `quick_train.py`
- **Training Guide:** `TRAINING_GUIDE.md`
"""
    
    with open("ENHANCED_DATASET_SUMMARY.md", "w") as f:
        f.write(summary)
    
    print("✅ Created ENHANCED_DATASET_SUMMARY.md")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Batch Training for Chess Recognition Models")
    parser.add_argument("--update-fen", action="store_true", help="Update annotations with sample FEN notations")
    parser.add_argument("--validate", action="store_true", help="Validate the current dataset")
    parser.add_argument("--train", action="store_true", help="Train models with current dataset")
    parser.add_argument("--summary", action="store_true", help="Create training dataset summary")
    parser.add_argument("--full-pipeline", action="store_true", help="Run the complete pipeline")
    
    args = parser.parse_args()
    
    if args.full_pipeline:
        print("🎯 Running complete enhanced training pipeline...")
        update_annotations_with_fen()
        validate_dataset()
        create_training_summary()
        train_models()
    elif args.update_fen:
        update_annotations_with_fen()
    elif args.validate:
        validate_dataset()
    elif args.train:
        train_models()
    elif args.summary:
        create_training_summary()
    else:
        print("🎯 Enhanced Chess Training Dataset Manager")
        print("\nAvailable commands:")
        print("  --update-fen     Update annotations with sample FEN notations")
        print("  --validate       Validate the current dataset")
        print("  --train          Train models with current dataset")
        print("  --summary        Create training dataset summary")
        print("  --full-pipeline  Run the complete pipeline")
        print("\nCurrent dataset status:")
        validate_dataset()

if __name__ == "__main__":
    main() 