#!/usr/bin/env python3
"""
Visualize the precise corner detection model on test images.
Shows both predicted corners and ground truth corners for comparison.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
import json
from pathlib import Path
import logging
import random
from PIL import Image
import pillow_heif

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PreciseCornerModel(nn.Module):
    """Regression model that predicts exact corner coordinates"""
    def __init__(self, backbone='resnet18', pretrained=True):
        super(PreciseCornerModel, self).__init__()
        
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
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

def load_model(model_path):
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PreciseCornerModel(backbone='resnet18', pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model, device

def predict_corners(model, device, image_path, image_size=384):
    """Predict corners for an image"""
    # Load image
    image_path = Path(image_path)
    
    if image_path.suffix.lower() in ['.heic']:
        pillow_heif.register_heif_opener()
        img = Image.open(image_path)
        image = np.array(img.convert('RGB'))
    else:
        image = cv2.imread(str(image_path))
        if image is None:
            return None, None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    original_h, original_w = image.shape[:2]
    
    # Prepare for model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    resized = cv2.resize(image, (image_size, image_size))
    image_tensor = transform(resized).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        corners_normalized = model(image_tensor).cpu().numpy()[0]
    
    # Denormalize to original image coordinates
    corners = corners_normalized.reshape(4, 2)
    corners[:, 0] *= original_w
    corners[:, 1] *= original_h
    
    return corners, image

def load_all_test_data():
    """Load all test images with ground truth"""
    all_data = []
    
    # Grey Background
    grey_annotations = Path('grey_background_dataset/annotations')
    grey_images = Path('grey_background_dataset/training images')
    
    for split in ['test', 'val']:  # Use test and val for visualization
        split_dir = grey_annotations / split
        if split_dir.exists():
            for json_file in split_dir.glob('*.json'):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    
                    image_name = data.get('image', json_file.stem + '.JPG')
                    image_path = grey_images / image_name
                    
                    if not image_path.exists():
                        for ext in ['.JPG', '.jpg']:
                            alt = grey_images / (json_file.stem + ext)
                            if alt.exists():
                                image_path = alt
                                break
                    
                    if image_path.exists() and 'corners' in data and len(data['corners']) == 4:
                        all_data.append({
                            'image_path': str(image_path),
                            'corners': data['corners'],
                            'source': 'grey_background'
                        })
                except Exception as e:
                    pass
    
    # Marshall Chess
    ann_file = Path('marshall_chess_annotations/annotations.json')
    marshall_photos = Path('/Users/tonyblum/Desktop/marshall photos')
    
    if ann_file.exists():
        with open(ann_file) as f:
            data = json.load(f)
        
        for image_name, ann_data in data.get('annotations', {}).items():
            if 'corners' in ann_data and len(ann_data['corners']) == 4:
                image_path = marshall_photos / image_name
                if image_path.exists():
                    all_data.append({
                        'image_path': str(image_path),
                        'corners': ann_data['corners'],
                        'source': 'marshall_chess'
                    })
    
    # Marshall2
    ann_dir = Path('marshall2_training_images/annotations')
    img_dir = Path('marshall2_training_images')
    
    if ann_dir.exists():
        for json_file in ann_dir.glob('*.json'):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                if (data.get('annotation_method') == 'manual_interactive' and 
                    'corners' in data and len(data['corners']) == 4):
                    
                    image_name = json_file.stem + '.jpg'
                    image_path = img_dir / image_name
                    
                    if image_path.exists():
                        all_data.append({
                            'image_path': str(image_path),
                            'corners': data['corners'],
                            'source': 'marshall2'
                        })
            except Exception as e:
                pass
    
    return all_data

def visualize_corners(image, predicted_corners, ground_truth_corners, title):
    """Draw corners on image"""
    vis_image = image.copy()
    
    # Draw ground truth in GREEN
    for i, corner in enumerate(ground_truth_corners):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(vis_image, (x, y), 15, (0, 255, 0), 3)
        cv2.putText(vis_image, f'GT{i+1}', (x+20, y+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    # Draw predictions in RED
    for i, corner in enumerate(predicted_corners):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(vis_image, (x, y), 12, (255, 0, 0), 3)
        cv2.putText(vis_image, f'P{i+1}', (x+20, y-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    
    # Draw lines connecting ground truth corners
    for i in range(4):
        pt1 = tuple(map(int, ground_truth_corners[i]))
        pt2 = tuple(map(int, ground_truth_corners[(i+1)%4]))
        cv2.line(vis_image, pt1, pt2, (0, 255, 0), 2)
    
    # Draw lines connecting predicted corners
    for i in range(4):
        pt1 = tuple(map(int, predicted_corners[i]))
        pt2 = tuple(map(int, predicted_corners[(i+1)%4]))
        cv2.line(vis_image, pt1, pt2, (255, 0, 0), 2)
    
    # Calculate error
    errors = []
    for pred, gt in zip(predicted_corners, ground_truth_corners):
        error = np.sqrt((pred[0] - gt[0])**2 + (pred[1] - gt[1])**2)
        errors.append(error)
    
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    
    # Add title with error
    cv2.putText(vis_image, title, (30, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(vis_image, f'Avg Error: {avg_error:.1f}px  Max: {max_error:.1f}px', 
               (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(vis_image, 'GREEN=Ground Truth, RED=Predicted', 
               (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    return vis_image, avg_error, max_error

def main():
    print("🎯 VISUALIZING PRECISE CORNER DETECTION")
    print("=" * 70)
    
    # Find latest model
    model_files = list(Path('models').glob('precise_corner_detector_*.pt'))
    if not model_files:
        print("❌ No trained model found!")
        return
    
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    print(f"📦 Loading model: {latest_model.name}")
    
    model, device = load_model(latest_model)
    print(f"✅ Model loaded on {device}")
    
    # Load test data
    print("\n📊 Loading test data...")
    test_data = load_all_test_data()
    print(f"   Found {len(test_data)} test images")
    
    # Sample 20 random images from different sources
    samples_per_source = {}
    for item in test_data:
        source = item['source']
        if source not in samples_per_source:
            samples_per_source[source] = []
        samples_per_source[source].append(item)
    
    selected_samples = []
    for source, items in samples_per_source.items():
        n_samples = min(7, len(items))
        selected_samples.extend(random.sample(items, n_samples))
    
    print(f"\n🎨 Creating visualizations for {len(selected_samples)} images...")
    
    # Create output directory
    output_dir = Path('precise_corner_visualizations')
    output_dir.mkdir(exist_ok=True)
    
    all_errors = []
    
    for idx, item in enumerate(selected_samples):
        print(f"   Processing {idx+1}/{len(selected_samples)}: {Path(item['image_path']).name}")
        
        # Predict corners
        predicted_corners, original_image = predict_corners(model, device, item['image_path'])
        
        if predicted_corners is None:
            print(f"      ⚠️  Failed to process")
            continue
        
        ground_truth = np.array(item['corners'])
        
        # Visualize
        vis_image, avg_error, max_error = visualize_corners(
            original_image,
            predicted_corners,
            ground_truth,
            f"{Path(item['image_path']).name} ({item['source']})"
        )
        
        all_errors.append(avg_error)
        
        # Save
        output_path = output_dir / f"{idx+1:02d}_{Path(item['image_path']).stem}_viz.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        print(f"      ✅ Avg error: {avg_error:.1f}px, Max: {max_error:.1f}px")
    
    print(f"\n{'='*70}")
    print(f"✅ VISUALIZATION COMPLETE!")
    print(f"\n📊 Overall Statistics:")
    print(f"   Images processed: {len(all_errors)}")
    print(f"   Average error: {np.mean(all_errors):.1f} pixels")
    print(f"   Median error: {np.median(all_errors):.1f} pixels")
    print(f"   Min error: {np.min(all_errors):.1f} pixels")
    print(f"   Max error: {np.max(all_errors):.1f} pixels")
    print(f"\n📁 Visualizations saved to: {output_dir}/")
    print(f"\nLegend:")
    print(f"   🟢 GREEN circles & lines = Ground Truth (manual annotations)")
    print(f"   🔴 RED circles & lines = Model Predictions")
    print(f"   Closer the red is to green = Better accuracy!")
    
if __name__ == '__main__':
    main()

