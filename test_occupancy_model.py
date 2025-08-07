#!/usr/bin/env python3
"""
Test occupancy model with actual dataset samples
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from torchvision import transforms
import random

def test_occupancy_model():
    """Test the occupancy model with actual dataset samples."""
    
    # Load the occupancy model
    model_path = Path("runs/occupancy_classifier/ResNet/ResNet.pt")
    model = torch.load(str(model_path), map_location='cpu', weights_only=False)
    model.eval()
    
    print("✅ Occupancy model loaded successfully")
    
    # Define transforms (must match training configuration)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((100, 100)),  # Match training config
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test with actual dataset samples
    empty_dir = Path("data:/occupancy/train/empty")
    occupied_dir = Path("data:/occupancy/train/occupied")
    
    # Get some sample images
    empty_images = list(empty_dir.glob("*.png"))[:5]
    occupied_images = list(occupied_dir.glob("*.png"))[:5]
    
    print(f"\n🔍 Testing with {len(empty_images)} empty samples:")
    for img_path in empty_images:
        img = cv2.imread(str(img_path))
        if img is not None:
            with torch.no_grad():
                input_tensor = transform(img).unsqueeze(0)
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                prediction = torch.argmax(probs, dim=1).item()
                confidence = probs[0][prediction].item()
                
                print(f"  {img_path.name}: {prediction} (0=empty, 1=occupied), conf: {confidence:.3f}")
    
    print(f"\n🔍 Testing with {len(occupied_images)} occupied samples:")
    for img_path in occupied_images:
        img = cv2.imread(str(img_path))
        if img is not None:
            with torch.no_grad():
                input_tensor = transform(img).unsqueeze(0)
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                prediction = torch.argmax(probs, dim=1).item()
                confidence = probs[0][prediction].item()
                
                print(f"  {img_path.name}: {prediction} (0=empty, 1=occupied), conf: {confidence:.3f}")

if __name__ == "__main__":
    test_occupancy_model() 