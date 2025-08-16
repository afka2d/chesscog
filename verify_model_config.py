#!/usr/bin/env python3
"""
Script to verify model configuration and preprocessing on the server.
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path
from models import ResNet  # Import our custom model

def verify_model(model_path: str, model_type: str):
    """Verify model configuration and preprocessing."""
    print(f"\nVerifying {model_type} model at {model_path}")
    print("-" * 50)
    
    try:
        # Load model
        model = models.resnet18(pretrained=False)
        if model_type == "piece":
            num_classes = 12  # 6 piece types * 2 colors
        else:
            num_classes = 2  # occupied or empty
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        
        # Load weights with weights_only=True
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        
        # Load weights
        model.load_state_dict(checkpoint)
        
        # Check model state
        print(f"Training mode: {model.training}")
        print(f"Number of classes: {model.fc.out_features}")
        
        # Print model configuration
        print("\nModel Configuration:")
        print(f"Input size: {model.fc.in_features}")
        print(f"Output size: {model.fc.out_features}")
        
        # Verify transforms
        if model_type == "piece":
            expected_size = (224, 448)
        else:
            expected_size = (100, 100)
            
        print(f"\nExpected input size: {expected_size}")
        
        # Put model in eval mode
        model.eval()
        print(f"Model in eval mode: {not model.training}")
        
        print("\nTest forward pass:")
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, *expected_size)
            output = model(dummy_input)
            print(f"Output shape: {output.shape}")
            print(f"Output classes: {output.argmax(dim=1).item()}")
        
        print("\n✅ Model verification successful")
        
    except Exception as e:
        print(f"\n❌ Error verifying model: {str(e)}")

def main():
    # Verify both models
    models_dir = Path("/root/chesscog/models")
    
    piece_model_path = models_dir / "piece_classifier/ResNet/ResNet.pt"
    occupancy_model_path = models_dir / "occupancy_classifier/ResNet/ResNet.pt"
    
    verify_model(str(piece_model_path), "piece")
    verify_model(str(occupancy_model_path), "occupancy")

if __name__ == "__main__":
    main()