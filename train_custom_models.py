#!/usr/bin/env python3
"""
Custom Training Script for Chess Recognition Models

This script prepares a custom dataset and trains both occupancy and piece classifiers.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def prepare_custom_dataset(custom_data_dir, output_dir):
    """Convert user annotations and images into the format expected by existing training pipelines"""
    print("🔄 Preparing custom dataset...")
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy images to the expected location
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Copy all images
    source_images = os.path.join(custom_data_dir, "images")
    if os.path.exists(source_images):
        subprocess.run(["cp", "-r", f"{source_images}/*", images_dir], check=True)
        print(f"✅ Copied images to {images_dir}")
    
    # Copy annotations
    annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)
    
    source_annotations = os.path.join(custom_data_dir, "annotations")
    if os.path.exists(source_annotations):
        subprocess.run(["cp", "-r", f"{source_annotations}/*.json", annotations_dir], check=True)
        print(f"✅ Copied annotations to {annotations_dir}")
    
    return output_dir

def train_models(dataset_dir, output_dir):
    """Run dataset creation and training for occupancy and piece classifiers"""
    print("🚀 Starting model training...")
    
    # First, create the datasets using the existing scripts
    print("📊 Creating occupancy classifier dataset...")
    occupancy_result = subprocess.run([
        "python", "-m", "chesscog.occupancy_classifier.create_dataset",
        "--input_dir", dataset_dir,
        "--output_dir", os.path.join(output_dir, "occupancy_dataset")
    ], capture_output=True, text=True)
    
    if occupancy_result.returncode != 0:
        print("❌ Failed to create occupancy dataset:")
        print(occupancy_result.stderr)
        return False
    
    print("📊 Creating piece classifier dataset...")
    piece_result = subprocess.run([
        "python", "-m", "chesscog.piece_classifier.create_dataset",
        "--input_dir", dataset_dir,
        "--output_dir", os.path.join(output_dir, "piece_dataset")
    ], capture_output=True, text=True)
    
    if piece_result.returncode != 0:
        print("❌ Failed to create piece dataset:")
        print(piece_result.stderr)
        return False
    
    # Now train the models
    print("🎯 Training occupancy classifier...")
    occupancy_train_result = subprocess.run([
        "python", "-m", "chesscog.occupancy_classifier.train",
        "--config", "ResNet"  # Use ResNet for better performance
    ], capture_output=True, text=True)
    
    if occupancy_train_result.returncode != 0:
        print("❌ Failed to train occupancy classifier:")
        print(occupancy_train_result.stderr)
        return False
    
    print("🎯 Training piece classifier...")
    piece_train_result = subprocess.run([
        "python", "-m", "chesscog.piece_classifier.train",
        "--config", "ResNet"  # Use ResNet for better performance
    ], capture_output=True, text=True)
    
    if piece_train_result.returncode != 0:
        print("❌ Failed to train piece classifier:")
        print(piece_train_result.stderr)
        return False
    
    print("✅ Training completed successfully!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Train chess recognition models on custom dataset")
    parser.add_argument("--input_dir", required=True, help="Directory containing custom dataset")
    parser.add_argument("--output_dir", required=True, help="Directory to save training results")
    
    args = parser.parse_args()
    
    # Prepare the dataset
    prepared_dir = prepare_custom_dataset(args.input_dir, args.output_dir)
    
    # Train the models
    success = train_models(prepared_dir, args.output_dir)
    
    if success:
        print(f"\n🎉 Training completed! Results saved in {args.output_dir}")
        print("📁 You can now use the trained models with your chess recognition API")
    else:
        print("\n❌ Training failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 