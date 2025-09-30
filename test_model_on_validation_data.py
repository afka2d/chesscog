#!/usr/bin/env python3
"""
Test the Marshall model on actual validation data to see if it works correctly
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import json

# Piece type labels
PIECE_TYPE_LABELS = {0: "pawn", 1: "knight", 2: "bishop", 3: "rook", 4: "queen", 5: "king"}

def _get_piece_type_model_architecture(num_classes):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def load_combined_piece_classifier():
    """Load the combined piece classification model."""
    try:
        model_path = Path("models_marshall_improved/piece_classification_combined_marshall.pt")
        if not model_path.exists():
            print(f"❌ Combined piece classifier not found at {model_path}")
            return None
        
        model = _get_piece_type_model_architecture(len(PIECE_TYPE_LABELS))
        state_dict = torch.load(str(model_path), map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ Combined piece classifier loaded successfully")
        return model
        
    except Exception as e:
        print(f"❌ Error loading combined piece classifier: {e}")
        return None

def test_model_on_grey_background_data():
    """Test the model on grey background validation data"""
    print("🧪 Testing Marshall Model on Grey Background Data...")
    print("=" * 60)
    
    # Load model
    model = load_combined_piece_classifier()
    if model is None:
        return
    
    # Load some grey background test data
    test_data_paths = [
        "grey_background_dataset/pieces/test",
        "grey_background_dataset/pieces/val",
        "grey_background_dataset/pieces/train"
    ]
    
    # Find test images
    test_images = []
    for path in test_data_paths:
        test_path = Path(path)
        if test_path.exists():
            for img_file in test_path.rglob("*.png"):
                test_images.append(img_file)
                if len(test_images) >= 20:  # Limit to 20 images for testing
                    break
            if len(test_images) >= 20:
                break
    
    if not test_images:
        print("❌ No test images found")
        return
    
    print(f"📸 Found {len(test_images)} test images")
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test predictions
    predictions = []
    confidences = []
    
    print("\n🔍 Testing model predictions:")
    for i, img_path in enumerate(test_images[:10]):  # Test first 10 images
        try:
            # Load image
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Apply transforms
            img_tensor = transform(img).unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                output = model(img_tensor)
                pred = torch.argmax(output, dim=1).item()
                conf = torch.softmax(output, dim=1)[0][pred].item()
            
            predictions.append(pred)
            confidences.append(conf)
            
            # Extract piece type from path for comparison
            piece_type = "unknown"
            path_str = str(img_path).lower()
            if 'pawn' in path_str:
                piece_type = "pawn"
            elif 'knight' in path_str:
                piece_type = "knight"
            elif 'bishop' in path_str:
                piece_type = "bishop"
            elif 'rook' in path_str:
                piece_type = "rook"
            elif 'queen' in path_str:
                piece_type = "queen"
            elif 'king' in path_str:
                piece_type = "king"
            
            predicted_piece = PIECE_TYPE_LABELS[pred]
            print(f"   {i+1:2d}. {img_path.name[:30]:<30} | True: {piece_type:<8} | Pred: {predicted_piece:<8} | Conf: {conf:.3f}")
            
        except Exception as e:
            print(f"   {i+1:2d}. Error processing {img_path.name}: {e}")
            continue
    
    # Analyze results
    print(f"\n📊 Analysis:")
    print(f"   Total predictions: {len(predictions)}")
    print(f"   Average confidence: {np.mean(confidences):.3f}")
    print(f"   Confidence std: {np.std(confidences):.3f}")
    print(f"   Min confidence: {np.min(confidences):.3f}")
    print(f"   Max confidence: {np.max(confidences):.3f}")
    
    # Check for diversity in predictions
    unique_predictions = set(predictions)
    print(f"   Unique predictions: {len(unique_predictions)}/6")
    print(f"   Prediction distribution: {[PIECE_TYPE_LABELS[p] for p in unique_predictions]}")
    
    print("\n" + "=" * 60)
    print("🎯 Model Testing Complete!")

if __name__ == "__main__":
    test_model_on_grey_background_data()
