#!/usr/bin/env python3
"""
Debug script to test Marshall models directly
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Piece type labels
PIECE_TYPE_LABELS = {0: "pawn", 1: "knight", 2: "bishop", 3: "rook", 4: "queen", 5: "king"}
COLOR_LABELS = {0: "black", 1: "white"}

def _get_piece_type_model_architecture(num_classes):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def _get_color_model_architecture(num_classes):
    model = models.mobilenet_v2(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

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

def load_combined_piece_classifier():
    """Load the combined piece classification model."""
    try:
        # Load the combined piece classifier
        model_path = Path("models_marshall_improved/piece_classification_combined_marshall.pt")
        if not model_path.exists():
            print(f"❌ Combined piece classifier not found at {model_path}")
            return None
        
        # Create the model architecture first
        model = _get_piece_type_model_architecture(len(PIECE_TYPE_LABELS))
        print("✅ Combined piece classifier architecture created")
        
        # Load the state_dict
        state_dict = torch.load(str(model_path), map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        print("✅ Combined piece classifier weights loaded")
        
        model.eval()
        print("✅ Combined piece classifier loaded successfully")
        return model
        
    except Exception as e:
        print(f"❌ Error loading combined piece classifier: {e}")
        return None

def load_color_classifier():
    """Load the color classification model."""
    try:
        color_model_path = Path("models/color_classifier_simple.pt")
        if not color_model_path.exists():
            print(f"❌ Color classifier not found at {color_model_path}")
            return None
        
        model = _get_color_model_architecture(len(COLOR_LABELS))
        model.load_state_dict(torch.load(str(color_model_path), map_location='cpu'))
        model.eval()
        print("✅ Color classifier loaded successfully")
        return model
        
    except Exception as e:
        print(f"❌ Error loading color classifier: {e}")
        return None

def test_models():
    """Test all models with a dummy input"""
    print("🧪 Testing Marshall Models...")
    print("=" * 50)
    
    # Create dummy input (100x100 RGB image)
    dummy_input = torch.randn(1, 3, 100, 100)
    print(f"📸 Created dummy input: {dummy_input.shape}")
    
    # Test occupancy model
    print("\n1. Testing Marshall Occupancy Model:")
    occupancy_model = load_marshall_occupancy_model()
    if occupancy_model is not None:
        with torch.no_grad():
            output = occupancy_model(dummy_input)
            print(f"   ✅ Output shape: {output.shape}")
            print(f"   ✅ Output range: [{output.min():.3f}, {output.max():.3f}]")
            print(f"   ✅ Prediction: {'Occupied' if output[0][1] > output[0][0] else 'Empty'}")
    else:
        print("   ❌ Failed to load occupancy model")
    
    # Test color model
    print("\n2. Testing Color Classification Model:")
    color_model = load_color_classifier()
    if color_model is not None:
        with torch.no_grad():
            output = color_model(dummy_input)
            print(f"   ✅ Output shape: {output.shape}")
            print(f"   ✅ Output range: [{output.min():.3f}, {output.max():.3f}]")
            predicted_color = torch.argmax(output, dim=1).item()
            print(f"   ✅ Prediction: {COLOR_LABELS[predicted_color]}")
    else:
        print("   ❌ Failed to load color model")
    
    # Test piece classification model
    print("\n3. Testing Combined Piece Classification Model:")
    piece_model = load_combined_piece_classifier()
    if piece_model is not None:
        with torch.no_grad():
            output = piece_model(dummy_input)
            print(f"   ✅ Output shape: {output.shape}")
            print(f"   ✅ Output range: [{output.min():.3f}, {output.max():.3f}]")
            predicted_piece = torch.argmax(output, dim=1).item()
            print(f"   ✅ Prediction: {PIECE_TYPE_LABELS[predicted_piece]}")
    else:
        print("   ❌ Failed to load piece model")
    
    print("\n" + "=" * 50)
    print("🎯 Model Testing Complete!")

if __name__ == "__main__":
    test_models()
