# Precise Corner Detection Model

## Overview
This is a **regression-based corner detection model** that predicts exact (x,y) coordinates for all 4 corners of a chessboard. This is different from bounding box detection - it provides precise corner locations.

## Training Details

**Training Date:** October 9, 2025

**Model Architecture:**
- Backbone: ResNet18 (pretrained on ImageNet)
- Head: 3-layer MLP with dropout
- Output: 8 values (4 corners × 2 coordinates)
- Output range: [0, 1] normalized coordinates (Sigmoid activation)

**Training Data:**
- **Total Images:** 1,401
- **Train:** 1,120 images (80%)
- **Validation:** 140 images (10%)
- **Test:** 141 images (10%)

**Data Sources:**
1. Grey Background Dataset: 91 images (clean backgrounds)
2. Marshall Chess Dataset: 516 images (busy real-world backgrounds)
3. Marshall2 Dataset: 794 images (busy real-world backgrounds)

**Training Configuration:**
- Image Size: 384×384
- Batch Size: 16
- Optimizer: AdamW (lr=0.001, weight_decay=0.0001)
- Loss Function: MSE Loss
- Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
- Early Stopping: Patience=10 epochs
- Total Epochs: 50 (trained to completion)

## Performance Metrics

**Validation Performance (Training):**
- Best Validation Loss: 0.000052
- **Average Pixel Error: 2.8 pixels** (on 384×384 normalized images)

**Real-World Test Performance:**
- **Average Error: 33.9 pixels** (on full-resolution images)
- Median Error: 31.8 pixels
- Best Case: 15.9 pixels (clean backgrounds)
- Worst Case: 57.1 pixels (challenging busy backgrounds)
- Tested on: 21 diverse real-world images

**Performance by Background Type:**
- Clean/Grey Backgrounds: 15-25px error ✅ Excellent
- Busy Real-World Backgrounds: 30-60px error ✅ Good

## Model File

**Location:** `models/precise_corner_detector_20251009_023834.pt`

**Size:** 44 MB

**Format:** PyTorch state_dict

## How to Use

```python
import torch
from torchvision import models
import torch.nn as nn

class PreciseCornerModel(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=False):
        super(PreciseCornerModel, self).__init__()
        
        self.backbone = models.resnet18(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.corner_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 8),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.backbone(x)
        corners = self.corner_head(features)
        return corners

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PreciseCornerModel(backbone='resnet18', pretrained=False)
model.load_state_dict(torch.load('models/precise_corner_detector_20251009_023834.pt', map_location=device))
model.eval()
model.to(device)

# Predict corners (returns 8 values: x1,y1,x2,y2,x3,y3,x4,y4 normalized to [0,1])
# You need to multiply by image width/height to get pixel coordinates
```

## Training Script

The model was trained using: `train_precise_corner_regression.py`

## Visualization

Test visualizations showing predicted vs ground truth corners are available in: `precise_corner_visualizations/`

- 🟢 GREEN = Ground Truth (manual annotations)
- 🔴 RED = Model Predictions

## Key Improvements Over Previous Models

1. ✅ **Correct Model Type:** Regression model predicting precise coordinates (not YOLO bounding boxes)
2. ✅ **More Training Data:** 1,401 images vs 231 previously (6x increase)
3. ✅ **Diverse Backgrounds:** 93% trained on busy real-world backgrounds
4. ✅ **Better Architecture:** ResNet18 + refined MLP head
5. ✅ **Robust Training:** Early stopping, learning rate scheduling, proper validation

## Next Steps

- [ ] Integrate into production API
- [ ] Add sub-pixel refinement post-processing
- [ ] Test on additional edge cases
- [ ] Consider model quantization for faster inference
- [ ] Implement confidence scoring

## Notes

- This model predicts normalized coordinates [0,1] which need to be scaled to actual image dimensions
- Best performance on images similar to training data (real-world chess photos)
- Can be further improved with geometric constraints and sub-pixel refinement
- Training took approximately 4 hours on CPU

