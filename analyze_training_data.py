#!/usr/bin/env python3
"""
Analyze the training data distribution to understand potential issues.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_training_data():
    """Analyze the training data distribution."""
    print("🔍 Analyzing Training Data Distribution")
    print("=" * 50)
    
    # Check all datasets
    datasets = ['train', 'val', 'test']
    base_path = Path("grey_background_dataset/pieces")
    
    for dataset in datasets:
        dataset_path = base_path / dataset
        if not dataset_path.exists():
            print(f"❌ Dataset not found: {dataset_path}")
            continue
            
        print(f"\n📁 {dataset.upper()} Dataset:")
        print(f"   Path: {dataset_path}")
        
        # Count images per class
        class_counts = {}
        total_images = 0
        
        for class_dir in dataset_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                image_count = len(list(class_dir.glob("*.png")))
                class_counts[class_name] = image_count
                total_images += image_count
                print(f"   {class_name}: {image_count} images")
        
        print(f"   Total: {total_images} images")
        
        # Check for class imbalance
        if class_counts:
            min_count = min(class_counts.values())
            max_count = max(class_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            print(f"   Class imbalance ratio: {imbalance_ratio:.2f}")
            if imbalance_ratio > 3:
                print("   ⚠️  WARNING: Significant class imbalance detected!")
            
            # Check for knight bias in data
            knight_classes = [k for k in class_counts.keys() if 'knight' in k]
            knight_count = sum(class_counts[k] for k in knight_classes)
            knight_percentage = (knight_count / total_images) * 100 if total_images > 0 else 0
            
            print(f"   Knight images: {knight_count}/{total_images} ({knight_percentage:.1f}%)")
            
            if knight_percentage > 30:
                print("   🚨 WARNING: High percentage of knight images in training data!")
        
        # Check for empty classes
        empty_classes = [k for k, v in class_counts.items() if v == 0]
        if empty_classes:
            print(f"   ❌ Empty classes: {empty_classes}")

def check_image_quality():
    """Check if there are any obvious issues with the images."""
    print(f"\n🔍 Checking Image Quality")
    print("=" * 30)
    
    # Sample a few images from each class
    base_path = Path("grey_background_dataset/pieces/train")
    if not base_path.exists():
        print("❌ Training data not found")
        return
    
    for class_dir in base_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            images = list(class_dir.glob("*.png"))
            if images:
                sample_image = images[0]
                print(f"   {class_name}: Sample image: {sample_image.name}")

if __name__ == "__main__":
    analyze_training_data()
    check_image_quality()
