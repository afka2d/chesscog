#!/usr/bin/env python3
"""
Test script to isolate the piece classifier loading issue.
"""

import torch
from torchvision import models
import torch.nn as nn
from pathlib import Path

# Color labels (must match training)
COLOR_LABELS = {0: "white", 1: "black"}

# Piece type labels (must match training)
PIECE_TYPE_LABELS = {0: "pawn", 1: "knight", 2: "bishop", 3: "rook", 4: "queen", 5: "king"}

# Helper to get piece type model architecture (must match training script)
def _get_piece_type_model_architecture(num_classes):
    model = models.resnet18(weights=None)  # ResNet18 for combined piece classifier
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def load_combined_piece_classifier():
    """Load the combined piece classification model."""
    try:
        # Load the combined piece classifier
        model_path = Path("models_marshall_improved/combined_piece_classifier.pt")
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
        print("✅ Combined piece classifier set to eval mode")
        return model
        
    except Exception as e:
        print(f"❌ Error loading combined piece classifier: {e}")
        return None

if __name__ == "__main__":
    print("🧪 Testing piece classifier loading...")
    model = load_combined_piece_classifier()
    if model is not None:
        print("✅ SUCCESS: Piece classifier loaded correctly")
        print(f"Model type: {type(model)}")
        print(f"Model has eval method: {hasattr(model, 'eval')}")
    else:
        print("❌ FAILED: Piece classifier loading failed")
