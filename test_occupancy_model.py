#!/usr/bin/env python3
"""
Test the Marshall occupancy model to see if it's working correctly
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

def load_marshall_occupancy_model():
    """Load the Marshall occupancy model (architecture + state_dict)."""
    try:
        # Load original model architecture
        original_model_path = Path("runs/occupancy_classifier/ResNet/ResNet.pt")
        if not original_model_path.exists():
            print(f"❌ Original occupancy model not found at {original_model_path}")
            return None
        
        model = torch.load(str(original_model_path), map_location='cpu', weights_only=False)
        print("✅ Original occupancy model architecture loaded")
        
        # Load Marshall weights
        marshall_path = Path("models_marshall_improved/occupancy_marshall.pt")
        if not marshall_path.exists():
            print(f"❌ Marshall occupancy model not found at {marshall_path}")
            return None
        
        marshall_weights = torch.load(str(marshall_path), map_location='cpu', weights_only=True)
        print("✅ Marshall occupancy weights loaded")
        
        # Apply Marshall weights to original architecture
        model.load_state_dict(marshall_weights)
        print("✅ Marshall occupancy model loaded successfully")
        
        model.eval()
        return model
        
    except Exception as e:
        print(f"❌ Error loading Marshall occupancy model: {e}")
        return None

def test_occupancy_model():
    """Test the Marshall occupancy model"""
    print("🧪 Testing Marshall Occupancy Model...")
    print("=" * 60)
    
    # Load model
    model = load_marshall_occupancy_model()
    if model is None:
        return
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test on different types of squares
    test_cases = [
        ("Empty square (white)", np.ones((100, 100, 3), dtype=np.uint8) * 255),
        ("Empty square (black)", np.zeros((100, 100, 3), dtype=np.uint8)),
        ("Random noise", np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)),
        ("Gradient", np.linspace(0, 255, 100*100*3).reshape(100, 100, 3).astype(np.uint8)),
    ]
    
    print("\n🔍 Testing occupancy detection on different square types:")
    for name, square in test_cases:
        # Convert to PIL Image
        square_pil = Image.fromarray(square)
        
        # Apply transforms
        square_tensor = transform(square_pil).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            output = model(square_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred].item()
        
        occupied = "Occupied" if pred == 1 else "Empty"
        print(f"   {name:<25} | {occupied:<8} | Conf: {conf:.3f} | Raw: {output[0].tolist()}")
    
    print("\n" + "=" * 60)
    print("🎯 Occupancy Model Testing Complete!")

if __name__ == "__main__":
    test_occupancy_model()
