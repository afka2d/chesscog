#!/usr/bin/env python3
"""
Create a balanced test set for piece classifier evaluation
"""

import os
import shutil
from pathlib import Path
import random

def create_balanced_test_set():
    """Create a balanced test set with equal numbers of each piece type."""
    
    pieces_dir = Path("grey_background_dataset/pieces")
    train_dir = pieces_dir / "train"
    test_dir = pieces_dir / "test"
    
    # Create test directory
    test_dir.mkdir(exist_ok=True)
    
    # Number of test images per class (balanced)
    test_images_per_class = 20
    
    piece_classes = [
        'black_bishop', 'black_king', 'black_knight', 'black_pawn', 
        'black_queen', 'black_rook', 'white_bishop', 'white_king', 
        'white_knight', 'white_pawn', 'white_queen', 'white_rook'
    ]
    
    for piece_class in piece_classes:
        train_piece_dir = train_dir / piece_class
        test_piece_dir = test_dir / piece_class
        
        if not train_piece_dir.exists():
            print(f"Warning: {piece_class} directory not found")
            continue
            
        # Create test subdirectory
        test_piece_dir.mkdir(exist_ok=True)
        
        # Get all images in this class
        images = list(train_piece_dir.glob("*.png"))
        
        if len(images) < test_images_per_class:
            print(f"Warning: {piece_class} only has {len(images)} images, using all")
            test_images_per_class = len(images)
        
        # Randomly select test images
        test_images = random.sample(images, test_images_per_class)
        
        # Copy to test directory
        for img_path in test_images:
            shutil.copy2(img_path, test_piece_dir / img_path.name)
            
        print(f"Created test set for {piece_class}: {len(test_images)} images")

if __name__ == "__main__":
    create_balanced_test_set() 