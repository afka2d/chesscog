"""
Script to update the server's main.py to use standard ResNet models.
"""

import torch
import torchvision.models as models

def create_model(num_classes):
    """Create a standard ResNet model."""
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

# Create piece classifier
piece_classifier = create_model(12)  # 6 piece types * 2 colors
piece_classifier.load_state_dict(torch.load("/root/chesscog/models/piece_classifier/ResNet/ResNet_converted.pt"))
piece_classifier.eval()

# Create occupancy classifier
occupancy_classifier = create_model(2)  # occupied or empty
occupancy_classifier.load_state_dict(torch.load("/root/chesscog/models/occupancy_classifier/ResNet/ResNet_converted.pt"))
occupancy_classifier.eval()

# Save models in production format
torch.save(piece_classifier, "/root/chesscog/models/piece_classifier.pt")
torch.save(occupancy_classifier, "/root/chesscog/models/occupancy_classifier.pt")

print("✅ Models converted and saved in production format")