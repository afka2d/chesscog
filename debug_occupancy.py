#!/usr/bin/env python3
"""
Debug occupancy classifier
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from torchvision import transforms

def test_occupancy_classifier():
    """Test the occupancy classifier directly."""
    
    # Load the occupancy model
    model_path = Path("runs/occupancy_classifier/ResNet/ResNet.pt")
    model = torch.load(str(model_path), map_location='cpu', weights_only=False)
    model.eval()
    
    print("✅ Occupancy model loaded successfully")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and test with a sample image
    img_path = "grey_background_dataset/images/test/IMG_4752.JPG"
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"❌ Could not load image: {img_path}")
        return
    
    print(f"✅ Loaded image: {img.shape}")
    
    # Create a simple test - take a small patch from the image
    # This should be a square that likely contains a piece
    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2
    patch_size = 100
    
    # Extract a patch from the center (likely to contain a piece)
    patch = img[center_y-patch_size//2:center_y+patch_size//2, 
                center_x-patch_size//2:center_x+patch_size//2]
    
    print(f"✅ Extracted patch: {patch.shape}")
    
    # Save the patch for inspection
    cv2.imwrite("debug_patch.png", patch)
    print("✅ Saved debug_patch.png")
    
    # Test the model
    with torch.no_grad():
        # Transform the patch
        input_tensor = transform(patch).unsqueeze(0)
        print(f"✅ Input tensor shape: {input_tensor.shape}")
        
        # Get prediction
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()
        
        print(f"✅ Prediction: {prediction} (0=empty, 1=occupied)")
        print(f"✅ Confidence: {confidence:.4f}")
        print(f"✅ Probabilities: empty={probs[0][0].item():.4f}, occupied={probs[0][1].item():.4f}")
    
    # Test with a completely white patch (should be empty)
    white_patch = np.ones((patch_size, patch_size, 3), dtype=np.uint8) * 255
    cv2.imwrite("debug_white_patch.png", white_patch)
    
    with torch.no_grad():
        input_tensor = transform(white_patch).unsqueeze(0)
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()
        
        print(f"✅ White patch prediction: {prediction} (0=empty, 1=occupied)")
        print(f"✅ White patch confidence: {confidence:.4f}")
        print(f"✅ White patch probabilities: empty={probs[0][0].item():.4f}, occupied={probs[0][1].item():.4f}")

if __name__ == "__main__":
    test_occupancy_classifier() 