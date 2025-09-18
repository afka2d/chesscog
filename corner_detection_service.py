#!/usr/bin/env python3
"""
Simple corner detection service that works reliably.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import json

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

class CornerDetectionService:
    def __init__(self, model_path="models/corner_detector_best.pt"):
        self.model = None
        self.transform = None
        self.device = torch.device('cpu')
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        """Load the corner detection model"""
        try:
            if not Path(self.model_path).exists():
                print(f"❌ Model not found: {self.model_path}")
                return False
            
            # Load model
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model = LightweightCornerModel()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Create transform
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            print(f"✅ Corner detection model loaded")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def detect_corners(self, image_path):
        """Detect corners in an image"""
        if self.model is None:
            return None
        
        try:
            # Load image
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = image_rgb.shape[:2]
            
            # Prepare image
            pil_image = Image.fromarray(image_rgb)
            input_tensor = self.transform(pil_image).unsqueeze(0)
            
            # Predict corners
            with torch.no_grad():
                corners_normalized = self.model(input_tensor).numpy()[0]
            
            # Convert back to original image coordinates
            corners_pixels = corners_normalized.reshape(4, 2)
            corners_pixels[:, 0] *= orig_w
            corners_pixels[:, 1] *= orig_h
            
            return corners_pixels.tolist()
            
        except Exception as e:
            print(f"❌ Error detecting corners: {e}")
            return None
    
    def visualize_corners(self, image_path, output_path=None):
        """Detect corners and create visualization"""
        corners = self.detect_corners(image_path)
        
        if corners is None:
            return None
        
        try:
            # Load original image
            image = cv2.imread(image_path)
            vis_image = image.copy()
            
            # Draw corners
            corner_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # Red, Green, Blue, Yellow
            corner_labels = ['TL', 'TR', 'BR', 'BL']
            
            for i, (corner, color, label) in enumerate(zip(corners, corner_colors, corner_labels)):
                x, y = int(corner[0]), int(corner[1])
                
                # Draw circle
                cv2.circle(vis_image, (x, y), 30, color, -1)
                
                # Draw label
                cv2.putText(vis_image, label, (x-20, y-35), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            # Draw board outline
            corners_int = np.array(corners, dtype=np.int32)
            cv2.polylines(vis_image, [corners_int], True, (255, 255, 255), 5)
            
            # Save visualization
            if output_path is None:
                output_path = f"corner_detection_{Path(image_path).stem}.jpg"
            
            cv2.imwrite(output_path, vis_image)
            
            return {
                'corners': corners,
                'visualization_path': output_path,
                'image_dimensions': [image.shape[1], image.shape[0]]
            }
            
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")
            return None

def test_multiple_images():
    """Test corner detection on multiple images"""
    print("🧪 TESTING CORNER DETECTION ON MULTIPLE IMAGES")
    print("=" * 60)
    
    service = CornerDetectionService()
    
    if service.model is None:
        print("❌ Corner detection service not available")
        return
    
    # Find test images
    test_images = []
    
    # Test images
    test_dir = Path("grey_background_dataset/images/test")
    if test_dir.exists():
        test_images.extend(list(test_dir.glob("*.JPG"))[:3])  # First 3 test images
    
    # Validation images
    val_dir = Path("grey_background_dataset/images/val")
    if val_dir.exists():
        test_images.extend(list(val_dir.glob("*.JPG"))[:2])  # First 2 val images
    
    if not test_images:
        print("❌ No test images found")
        return
    
    print(f"📊 Testing {len(test_images)} images:")
    
    total_error = 0
    successful_tests = 0
    
    for i, image_path in enumerate(test_images):
        print(f"\n--- Test {i+1}/{len(test_images)}: {image_path.name} ---")
        
        # Detect corners
        result = service.visualize_corners(str(image_path))
        
        if result:
            print(f"✅ Detected corners: {result['corners']}")
            print(f"   Visualization: {result['visualization_path']}")
            
            # Compare with ground truth if available
            gt_path = Path(f"grey_background_dataset/annotations/{image_path.parent.name}/{image_path.stem}.json")
            
            if gt_path.exists():
                try:
                    with open(gt_path, 'r') as f:
                        annotation = json.load(f)
                    
                    gt_corners = np.array(annotation.get('corners', []))
                    pred_corners = np.array(result['corners'])
                    
                    if len(gt_corners) == 4:
                        errors = np.sqrt(np.sum((gt_corners - pred_corners) ** 2, axis=1))
                        avg_error = np.mean(errors)
                        total_error += avg_error
                        successful_tests += 1
                        
                        print(f"   Average error: {avg_error:.1f} pixels")
                        
                except Exception as e:
                    print(f"   ⚠️  Could not compare with ground truth: {e}")
            
        else:
            print(f"❌ Corner detection failed")
    
    if successful_tests > 0:
        overall_avg_error = total_error / successful_tests
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Successful tests: {successful_tests}/{len(test_images)}")
        print(f"   Average pixel error: {overall_avg_error:.1f} pixels")
        
        if overall_avg_error < 50:
            print("✅ EXCELLENT: Very accurate corner detection")
        elif overall_avg_error < 100:
            print("✅ GOOD: Acceptable corner detection")
        elif overall_avg_error < 200:
            print("⚠️  FAIR: Needs improvement")
        else:
            print("❌ POOR: Significant improvement needed")

def main():
    """Main function"""
    print("Corner Detection Service Test")
    print("=" * 50)
    
    test_multiple_images()
    
    print(f"\n🎯 CORNER DETECTION TESTING COMPLETE!")
    print("Check the generated visualization files to see the results.")

if __name__ == "__main__":
    main()
