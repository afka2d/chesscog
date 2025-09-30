#!/usr/bin/env python3
"""
Debug script to compare API preprocessing vs validation preprocessing
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

# Piece type labels
PIECE_TYPE_LABELS = {0: "pawn", 1: "knight", 2: "bishop", 3: "rook", 4: "queen", 5: "king"}
COLOR_LABELS = {0: "black", 1: "white"}

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

def test_preprocessing_differences():
    """Test different preprocessing approaches"""
    print("🧪 Testing Preprocessing Differences...")
    print("=" * 60)
    
    # Load a test image
    test_image_path = Path("yolo_detection_IMG_4763.jpg")
    if not test_image_path.exists():
        print("❌ Test image not found")
        return
    
    print(f"📸 Using test image: {test_image_path}")
    
    # Load image
    img = cv2.imread(str(test_image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"   Image shape: {img.shape}")
    
    # Create a test square
    square = cv2.resize(img, (224, 224))
    print(f"   Square shape: {square.shape}")
    
    # Test different preprocessing approaches
    print("\n1. API Preprocessing (ImageNet normalization):")
    api_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    square_pil = Image.fromarray(square)
    api_tensor = api_transform(square_pil)
    print(f"   ✅ API tensor shape: {api_tensor.shape}")
    print(f"   ✅ API tensor range: [{api_tensor.min():.3f}, {api_tensor.max():.3f}]")
    print(f"   ✅ API tensor mean: {api_tensor.mean():.3f}")
    print(f"   ✅ API tensor std: {api_tensor.std():.3f}")
    
    print("\n2. Validation Preprocessing (ImageNet normalization):")
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_tensor = val_transform(square_pil)
    print(f"   ✅ Validation tensor shape: {val_tensor.shape}")
    print(f"   ✅ Validation tensor range: [{val_tensor.min():.3f}, {val_tensor.max():.3f}]")
    print(f"   ✅ Validation tensor mean: {val_tensor.mean():.3f}")
    print(f"   ✅ Validation tensor std: {val_tensor.std():.3f}")
    
    print("\n3. Simple Preprocessing (divide by 255):")
    simple_tensor = torch.from_numpy(square).permute(2, 0, 1).float() / 255.0
    print(f"   ✅ Simple tensor shape: {simple_tensor.shape}")
    print(f"   ✅ Simple tensor range: [{simple_tensor.min():.3f}, {simple_tensor.max():.3f}]")
    print(f"   ✅ Simple tensor mean: {simple_tensor.mean():.3f}")
    print(f"   ✅ Simple tensor std: {simple_tensor.std():.3f}")
    
    # Test model predictions
    print("\n4. Model Predictions:")
    model = load_combined_piece_classifier()
    if model is not None:
        with torch.no_grad():
            # API preprocessing
            api_output = model(api_tensor.unsqueeze(0))
            api_pred = torch.argmax(api_output, dim=1).item()
            api_conf = torch.softmax(api_output, dim=1)[0][api_pred].item()
            print(f"   API prediction: {PIECE_TYPE_LABELS[api_pred]} (conf: {api_conf:.3f})")
            
            # Validation preprocessing
            val_output = model(val_tensor.unsqueeze(0))
            val_pred = torch.argmax(val_output, dim=1).item()
            val_conf = torch.softmax(val_output, dim=1)[0][val_pred].item()
            print(f"   Validation prediction: {PIECE_TYPE_LABELS[val_pred]} (conf: {val_conf:.3f})")
            
            # Simple preprocessing
            simple_output = model(simple_tensor.unsqueeze(0))
            simple_pred = torch.argmax(simple_output, dim=1).item()
            simple_conf = torch.softmax(simple_output, dim=1)[0][simple_pred].item()
            print(f"   Simple prediction: {PIECE_TYPE_LABELS[simple_pred]} (conf: {simple_conf:.3f})")
    
    print("\n" + "=" * 60)
    print("🎯 Preprocessing Comparison Complete!")

if __name__ == "__main__":
    test_preprocessing_differences()
