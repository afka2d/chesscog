#!/usr/bin/env python3
"""
Compare original vs Marshall occupancy models
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

def load_original_occupancy_model():
    """Load the original occupancy model"""
    try:
        model_path = Path("runs/occupancy_classifier/ResNet/ResNet.pt")
        if not model_path.exists():
            print(f"❌ Original occupancy model not found at {model_path}")
            return None
        
        model = torch.load(str(model_path), map_location='cpu', weights_only=False)
        model.eval()
        print("✅ Original occupancy model loaded successfully")
        return model
        
    except Exception as e:
        print(f"❌ Error loading original occupancy model: {e}")
        return None

def load_marshall_occupancy_model():
    """Load the Marshall occupancy model (architecture + state_dict)."""
    try:
        # Load original model architecture
        original_model_path = Path("runs/occupancy_classifier/ResNet/ResNet.pt")
        if not original_model_path.exists():
            print(f"❌ Original occupancy model not found at {original_model_path}")
            return None
        
        model = torch.load(str(original_model_path), map_location='cpu', weights_only=False)
        
        # Load Marshall weights
        marshall_path = Path("models_marshall_improved/occupancy_marshall.pt")
        if not marshall_path.exists():
            print(f"❌ Marshall occupancy model not found at {marshall_path}")
            return None
        
        marshall_weights = torch.load(str(marshall_path), map_location='cpu', weights_only=True)
        
        # Apply Marshall weights to original architecture
        model.load_state_dict(marshall_weights)
        model.eval()
        print("✅ Marshall occupancy model loaded successfully")
        return model
        
    except Exception as e:
        print(f"❌ Error loading Marshall occupancy model: {e}")
        return None

def compare_occupancy_models():
    """Compare original vs Marshall occupancy models"""
    print("🧪 Comparing Original vs Marshall Occupancy Models...")
    print("=" * 70)
    
    # Load models
    original_model = load_original_occupancy_model()
    marshall_model = load_marshall_occupancy_model()
    
    if original_model is None or marshall_model is None:
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
    
    print("\n🔍 Comparing occupancy detection on different square types:")
    print(f"{'Square Type':<25} | {'Original':<15} | {'Marshall':<15} | {'Match':<5}")
    print("-" * 70)
    
    for name, square in test_cases:
        # Convert to PIL Image
        square_pil = Image.fromarray(square)
        
        # Apply transforms
        square_tensor = transform(square_pil).unsqueeze(0)
        
        # Get original prediction
        with torch.no_grad():
            orig_output = original_model(square_tensor)
            orig_probs = torch.softmax(orig_output, dim=1)
            orig_pred = torch.argmax(orig_probs, dim=1).item()
            orig_conf = orig_probs[0][orig_pred].item()
        
        # Get Marshall prediction
        with torch.no_grad():
            marshall_output = marshall_model(square_tensor)
            marshall_probs = torch.softmax(marshall_output, dim=1)
            marshall_pred = torch.argmax(marshall_probs, dim=1).item()
            marshall_conf = marshall_probs[0][marshall_pred].item()
        
        orig_occupied = "Occupied" if orig_pred == 1 else "Empty"
        marshall_occupied = "Occupied" if marshall_pred == 1 else "Empty"
        match = "✓" if orig_pred == marshall_pred else "✗"
        
        print(f"{name:<25} | {orig_occupied:<8} ({orig_conf:.3f}) | {marshall_occupied:<8} ({marshall_conf:.3f}) | {match:<5}")
    
    print("\n" + "=" * 70)
    print("🎯 Occupancy Model Comparison Complete!")

if __name__ == "__main__":
    compare_occupancy_models()
