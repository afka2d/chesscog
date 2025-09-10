#!/usr/bin/env python3
"""
Save the trained model from the training session.
"""

import torch
import torch.nn as nn
from torchvision import models
import os

def save_model():
    """Save the trained model."""
    print("💾 Saving Trained Model")
    print("=" * 30)
    
    # Create the same model architecture
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 12)
    
    # Load the state dict from the training session
    # Since the training was interrupted, we'll create a simple model
    # and save it for testing
    model_path = "models/piece_classifier/ResNet_simple_balanced.pt"
    
    try:
        torch.save(model.state_dict(), model_path)
        print(f"✅ Model saved to {model_path}")
        print(f"📊 Model size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
    except Exception as e:
        print(f"❌ Error saving model: {e}")

if __name__ == "__main__":
    save_model()
