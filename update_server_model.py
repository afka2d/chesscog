"""
Script to update the server's model loading code.
"""

import torch
import torchvision.models as models

def convert_model(input_path: str, output_path: str, num_classes: int):
    """Convert a custom model to standard ResNet format."""
    # Create standard ResNet model
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    # Load weights from custom model (trusted source)
    checkpoint = torch.load(input_path, map_location=torch.device('cpu'), weights_only=False)
    
    # Save in standard format
    torch.save(model.state_dict(), output_path)
    print(f"Converted model saved to {output_path}")

def main():
    # Convert piece classifier (12 classes)
    convert_model(
        "/root/chesscog/models/piece_classifier/ResNet/ResNet.pt",
        "/root/chesscog/models/piece_classifier/ResNet/ResNet_converted.pt",
        12
    )
    
    # Convert occupancy classifier (2 classes)
    convert_model(
        "/root/chesscog/models/occupancy_classifier/ResNet/ResNet.pt",
        "/root/chesscog/models/occupancy_classifier/ResNet/ResNet_converted.pt",
        2
    )

if __name__ == "__main__":
    main()