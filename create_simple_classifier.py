#!/usr/bin/env python3
"""
Create a simple piece classifier using the existing models but with better configuration.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from pathlib import Path
import os

def create_simple_resnet():
    """Create a simple ResNet18 model for piece classification."""
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 12)
    return model

def test_simple_model():
    """Test the simple model with proper transforms."""
    print("🧪 Testing Simple Model")
    print("=" * 30)
    
    # Create model
    model = create_simple_resnet()
    model.eval()
    
    # Define proper transforms (224x224 for ResNet18)
    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test with a few sample images
    test_images = [
        "grey_background_dataset/pieces/test/white_king/NEW_20250805_135338_005_b3.png",
        "grey_background_dataset/pieces/test/white_queen/NEW_20250805_135338_002_a4.png",
        "grey_background_dataset/pieces/test/white_rook/NEW_20250805_135338_011_h1.png",
        "grey_background_dataset/pieces/test/white_bishop/NEW_20250805_135338_009_b2.png",
        "grey_background_dataset/pieces/test/white_knight/NEW_20250805_135338_006_g2.png",
        "grey_background_dataset/pieces/test/white_pawn/NEW_20250805_135338_008_f3.png"
    ]
    
    class_names = [
        'black_bishop', 'black_king', 'black_knight', 'black_pawn', 'black_queen', 'black_rook',
        'white_bishop', 'white_king', 'white_knight', 'white_pawn', 'white_queen', 'white_rook'
    ]
    
    print(f"📋 Classes: {class_names}")
    
    predictions = []
    for i, image_path in enumerate(test_images):
        if not os.path.exists(image_path):
            print(f"   {i+1}. ❌ {os.path.basename(image_path)}: Not found")
            continue
            
        try:
            from PIL import Image
            img = Image.open(image_path).convert('RGB')
            img_tensor = transforms_test(img).unsqueeze(0)
            
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            predicted_name = class_names[predicted_class]
            expected_name = os.path.basename(os.path.dirname(image_path))
            
            correct = predicted_name == expected_name
            status = "✅" if correct else "❌"
            
            print(f"   {i+1}. {status} {os.path.basename(image_path)}: {predicted_name} (conf: {confidence:.3f})")
            if not correct:
                print(f"       Expected: {expected_name}")
            
            predictions.append({
                'image': os.path.basename(image_path),
                'expected': expected_name,
                'predicted': predicted_name,
                'confidence': confidence,
                'correct': correct
            })
            
        except Exception as e:
            print(f"   {i+1}. ❌ {os.path.basename(image_path)}: Error - {e}")
    
    # Summary
    if predictions:
        correct_count = sum(1 for p in predictions if p['correct'])
        total_count = len(predictions)
        accuracy = correct_count / total_count * 100
        
        print(f"\n📊 Summary:")
        print(f"   Correct: {correct_count}/{total_count} ({accuracy:.1f}%)")
        
        # Check for knight bias
        knight_predictions = sum(1 for p in predictions if 'knight' in p['predicted'])
        knight_percentage = knight_predictions / total_count * 100
        print(f"   Knight predictions: {knight_predictions}/{total_count} ({knight_percentage:.1f}%)")
        
        if knight_percentage > 50:
            print("   🚨 WARNING: Model has strong knight bias!")
        else:
            print("   ✅ Knight distribution looks normal")
    
    return model

def save_model_for_api(model):
    """Save the model for use in the API."""
    print(f"\n💾 Saving Model for API")
    print("=" * 30)
    
    model_path = "models/piece_classifier/ResNet_simple.pt"
    try:
        torch.save(model, model_path)
        print(f"✅ Model saved to {model_path}")
        print(f"📊 Model size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
        return model_path
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return None

if __name__ == "__main__":
    model = test_simple_model()
    if model:
        save_model_for_api(model)
