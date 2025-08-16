"""
Custom ResNet model definitions for chess piece and occupancy classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

# Add the models to safe globals for loading
torch.serialization.add_safe_globals([
    'chesscog.piece_classifier.models.ResNet',
    'chesscog.occupancy_classifier.models.ResNet'
])