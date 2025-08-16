#!/usr/bin/env python3
"""
Script to verify image preprocessing transforms match between training and inference.
"""

import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_and_preprocess_image(img_path, transform):
    """Load and preprocess an image."""
    # Load image
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor
    img_tensor = transforms.ToTensor()(img)
    
    # Apply transform
    processed = transform(img_tensor)
    
    return processed

def visualize_transforms(img_path, piece_transform, occupancy_transform):
    """Visualize the transforms applied to an image."""
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img)
    axes[0].set_title("Original")
    
    # Piece classifier transform
    piece_img = load_and_preprocess_image(img_path, piece_transform)
    piece_img = piece_img.permute(1, 2, 0).numpy()
    axes[1].imshow(piece_img)
    axes[1].set_title(f"Piece Transform\n{piece_img.shape}")
    
    # Occupancy classifier transform
    occ_img = load_and_preprocess_image(img_path, occupancy_transform)
    occ_img = occ_img.permute(1, 2, 0).numpy()
    axes[2].imshow(occ_img)
    axes[2].set_title(f"Occupancy Transform\n{occ_img.shape}")
    
    plt.tight_layout()
    plt.savefig("transform_visualization.png")
    print("✅ Transform visualization saved to transform_visualization.png")

def main():
    # Define transforms
    piece_transform = transforms.Compose([
        transforms.Resize((224, 448)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    occupancy_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Find a test image
    test_dir = Path("grey_background_dataset/images/test")
    test_images = list(test_dir.glob("*.JPG"))
    if test_images:
        test_img = test_images[0]
        print(f"Using test image: {test_img}")
        visualize_transforms(test_img, piece_transform, occupancy_transform)
    else:
        print("No test images found!")

if __name__ == "__main__":
    main()