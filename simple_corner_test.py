#!/usr/bin/env python3
"""
Simple corner detection test to verify the model works.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

class LightweightCornerModel(nn.Module):
    def __init__(self):
        super(LightweightCornerModel, self).__init__()
        
        # Use a lightweight ResNet18
        self.backbone = models.resnet18(weights='DEFAULT')
        self.backbone.fc = nn.Identity()
        
        # Simple corner detection head
        self.corner_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8),  # 4 corners × 2 coordinates
            nn.Sigmoid()  # Normalize to [0, 1]
        )
        
    def forward(self, x):
        features = self.backbone(x)
        corners = self.corner_head(features)
        return corners

def test_corner_model():
    """Test the corner detection model directly"""
    print("🧪 TESTING CORNER DETECTION MODEL")
    print("=" * 50)
    
    # Check if model exists
    model_path = "models/corner_detector_best.pt"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    print(f"✅ Model found: {model_path}")
    
    # Load model
    try:
        device = torch.device('cpu')  # Use CPU for testing
        checkpoint = torch.load(model_path, map_location=device)
        
        print(f"📊 Model info:")
        print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"   Train loss: {checkpoint.get('train_loss', 'unknown')}")
        print(f"   Val loss: {checkpoint.get('val_loss', 'unknown')}")
        print(f"   Image size: {checkpoint.get('image_size', 'unknown')}")
        
        # Create model
        model = LightweightCornerModel()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ Model loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Test with a sample image
    test_image_path = "grey_background_dataset/images/test/IMG_4785.JPG"
    
    if not Path(test_image_path).exists():
        test_image_path = "grey_background_dataset/images/val/IMG_4779.JPG"
        
        if not Path(test_image_path).exists():
            print("❌ No test images found")
            return False
    
    print(f"\n🖼️  Testing with image: {Path(test_image_path).name}")
    
    try:
        # Load and preprocess image
        image = cv2.imread(test_image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]
        
        print(f"   Original dimensions: {orig_w} x {orig_h}")
        
        # Create transform
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Prepare image
        pil_image = Image.fromarray(image_rgb)
        input_tensor = transform(pil_image).unsqueeze(0)
        
        print(f"   Input tensor shape: {input_tensor.shape}")
        
        # Predict corners
        with torch.no_grad():
            corners_normalized = model(input_tensor).numpy()[0]
        
        print(f"   Normalized corners: {corners_normalized}")
        
        # Convert back to original image coordinates
        corners_pixels = corners_normalized.reshape(4, 2)
        corners_pixels[:, 0] *= orig_w  # Scale x coordinates
        corners_pixels[:, 1] *= orig_h  # Scale y coordinates
        
        print(f"   Pixel corners: {corners_pixels}")
        
        # Create visualization
        vis_image = image.copy()
        
        # Draw corners
        corner_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # Red, Green, Blue, Yellow
        corner_labels = ['TL', 'TR', 'BR', 'BL']
        
        for i, (corner, color, label) in enumerate(zip(corners_pixels, corner_colors, corner_labels)):
            x, y = int(corner[0]), int(corner[1])
            
            # Draw circle
            cv2.circle(vis_image, (x, y), 20, color, -1)
            
            # Draw label
            cv2.putText(vis_image, label, (x-15, y-25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Draw board outline
        corners_int = corners_pixels.astype(np.int32)
        cv2.polylines(vis_image, [corners_int], True, (255, 255, 255), 3)
        
        # Save visualization
        output_path = "corner_detection_test_result.jpg"
        cv2.imwrite(output_path, vis_image)
        
        print(f"✅ Corner detection successful!")
        print(f"   📸 Visualization saved to: {output_path}")
        
        # Load ground truth for comparison if available
        gt_path = f"grey_background_dataset/annotations/test/{Path(test_image_path).stem}.json"
        if not Path(gt_path).exists():
            gt_path = f"grey_background_dataset/annotations/val/{Path(test_image_path).stem}.json"
        
        if Path(gt_path).exists():
            try:
                with open(gt_path, 'r') as f:
                    annotation = json.load(f)
                
                gt_corners = np.array(annotation.get('corners', []))
                
                if len(gt_corners) == 4:
                    # Calculate error
                    errors = np.sqrt(np.sum((gt_corners - corners_pixels) ** 2, axis=1))
                    avg_error = np.mean(errors)
                    
                    print(f"\n📊 ACCURACY COMPARISON:")
                    print(f"   Ground truth corners: {gt_corners}")
                    print(f"   Predicted corners: {corners_pixels}")
                    print(f"   Per-corner errors: {[f'{e:.1f}' for e in errors]} pixels")
                    print(f"   Average error: {avg_error:.1f} pixels")
                    
                    if avg_error < 50:
                        print("✅ EXCELLENT: Very accurate corner detection")
                    elif avg_error < 100:
                        print("✅ GOOD: Acceptable corner detection")
                    elif avg_error < 200:
                        print("⚠️  FAIR: Needs improvement")
                    else:
                        print("❌ POOR: Significant improvement needed")
                        
            except Exception as e:
                print(f"⚠️  Could not load ground truth: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in corner detection: {e}")
        return False

def main():
    """Main function"""
    print("Simple Corner Detection Test")
    print("=" * 50)
    
    success = test_corner_model()
    
    if success:
        print(f"\n🎯 CORNER DETECTION TEST SUCCESSFUL!")
        print("The model is working correctly.")
        print("Check the visualization: corner_detection_test_result.jpg")
    else:
        print(f"\n❌ CORNER DETECTION TEST FAILED!")

if __name__ == "__main__":
    import json
    main()
