#!/usr/bin/env python3
"""
Verify model architecture and weights.
"""

import torch
import torchvision.models as models
import numpy as np

def verify_model(model_path, num_classes):
    """Load and verify model architecture."""
    print(f"\nVerifying model: {model_path}")
    print("-" * 50)
    
    # Load model
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    model.eval()
    
    # Print model architecture
    print("Model Architecture:")
    print(model)
    
    # Check final layer
    if hasattr(model, 'fc'):
        print(f"\nFinal layer output size: {model.fc.out_features}")
        if model.fc.out_features != num_classes:
            print(f"⚠️ Warning: Expected {num_classes} classes but found {model.fc.out_features}")
    
    # Check if weights are properly initialized
    print("\nWeight Statistics:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            stats = {
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'min': param.data.min().item(),
                'max': param.data.max().item(),
                'zeros': (param.data == 0).float().mean().item() * 100
            }
            print(f"\n{name}:")
            print(f"  Mean: {stats['mean']:.6f}")
            print(f"  Std:  {stats['std']:.6f}")
            print(f"  Min:  {stats['min']:.6f}")
            print(f"  Max:  {stats['max']:.6f}")
            print(f"  Zero: {stats['zeros']:.1f}%")
    
    # Test forward pass
    print("\nTesting forward pass...")
    try:
        dummy_input = torch.randn(1, 3, 224, 448)  # Batch size 1, 3 channels, 224x448 pixels
        with torch.no_grad():
            output = model(dummy_input)
            
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")
        
        # Get predictions
        probs = torch.softmax(output, dim=1)
        print("\nClass probabilities:")
        for i, prob in enumerate(probs[0]):
            print(f"Class {i}: {prob.item():.4f}")
            
    except Exception as e:
        print(f"⚠️ Forward pass failed: {str(e)}")

def main():
    # Verify piece classifier
    verify_model("runs/piece_classifier/ResNet/ResNet.pt", 12)
    
    # Verify occupancy classifier
    verify_model("runs/occupancy_classifier/ResNet/ResNet.pt", 2)

if __name__ == "__main__":
    main()