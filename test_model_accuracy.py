#!/usr/bin/env python3
"""
Test the actual accuracy of our piece classifier model on real data.
This will give us a definitive answer about whether we achieve 80%+ accuracy.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from pathlib import Path
import os
from PIL import Image
import random

def create_resnet_model():
    """Create the ResNet18 model architecture."""
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 12)
    return model

def test_model_accuracy():
    """Test the model accuracy on real piece images."""
    print("🧪 Testing Model Accuracy on Real Data")
    print("=" * 50)
    
    # Load the model we trained
    model_path = "models/piece_classifier/InceptionV3.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return False
    
    # Load model
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    model.eval()
    
    # Define transforms (299x299 for InceptionV3)
    transforms_test = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Class names
    class_names = [
        'black_bishop', 'black_king', 'black_knight', 'black_pawn', 'black_queen', 'black_rook',
        'white_bishop', 'white_king', 'white_knight', 'white_pawn', 'white_queen', 'white_rook'
    ]
    
    print(f"📋 Classes: {class_names}")
    
    # Test on multiple images from each class
    test_results = []
    total_tests = 0
    correct_tests = 0
    
    # Test each class
    for class_name in class_names:
        class_dir = f"grey_background_dataset/pieces/test/{class_name}"
        if not os.path.exists(class_dir):
            print(f"   ⚠️  {class_name}: Directory not found")
            continue
        
        # Get all images in this class
        image_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
        if not image_files:
            print(f"   ⚠️  {class_name}: No images found")
            continue
        
        # Test up to 10 random images from this class
        test_images = random.sample(image_files, min(10, len(image_files)))
        
        class_correct = 0
        class_total = len(test_images)
        
        print(f"\n🔍 Testing {class_name} ({class_total} images):")
        
        for i, image_file in enumerate(test_images):
            image_path = os.path.join(class_dir, image_file)
            
            try:
                # Load and preprocess image
                img = Image.open(image_path).convert('RGB')
                img_tensor = transforms_test(img).unsqueeze(0)
                
                # Get prediction
                with torch.no_grad():
                    output = model(img_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                
                predicted_name = class_names[predicted_class]
                correct = predicted_name == class_name
                
                if correct:
                    class_correct += 1
                    correct_tests += 1
                    status = "✅"
                else:
                    status = "❌"
                
                print(f"   {i+1:2d}. {status} {image_file}: {predicted_name} (conf: {confidence:.3f})")
                
                test_results.append({
                    'class': class_name,
                    'image': image_file,
                    'predicted': predicted_name,
                    'confidence': confidence,
                    'correct': correct
                })
                
            except Exception as e:
                print(f"   {i+1:2d}. ❌ {image_file}: Error - {e}")
        
        class_accuracy = class_correct / class_total * 100
        print(f"   📊 {class_name} accuracy: {class_correct}/{class_total} ({class_accuracy:.1f}%)")
        
        total_tests += class_total
    
    # Overall results
    overall_accuracy = correct_tests / total_tests * 100 if total_tests > 0 else 0
    
    print(f"\n📊 OVERALL RESULTS:")
    print(f"   Total tests: {total_tests}")
    print(f"   Correct: {correct_tests}")
    print(f"   Accuracy: {overall_accuracy:.1f}%")
    
    # Check for knight bias
    knight_predictions = sum(1 for r in test_results if 'knight' in r['predicted'])
    knight_percentage = knight_predictions / total_tests * 100 if total_tests > 0 else 0
    
    print(f"\n🔍 BIAS ANALYSIS:")
    print(f"   Knight predictions: {knight_predictions}/{total_tests} ({knight_percentage:.1f}%)")
    
    if knight_percentage > 30:
        print("   🚨 WARNING: Model has strong knight bias!")
    else:
        print("   ✅ Knight distribution looks normal")
    
    # Per-class accuracy breakdown
    print(f"\n📈 PER-CLASS ACCURACY:")
    class_stats = {}
    for result in test_results:
        class_name = result['class']
        if class_name not in class_stats:
            class_stats[class_name] = {'correct': 0, 'total': 0}
        class_stats[class_name]['total'] += 1
        if result['correct']:
            class_stats[class_name]['correct'] += 1
    
    for class_name, stats in class_stats.items():
        accuracy = stats['correct'] / stats['total'] * 100
        print(f"   {class_name}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")
    
    # Final verdict
    print(f"\n🎯 FINAL VERDICT:")
    if overall_accuracy >= 80:
        print(f"   ✅ SUCCESS: Model achieves {overall_accuracy:.1f}% accuracy (≥80%)")
        return True
    else:
        print(f"   ❌ FAILURE: Model only achieves {overall_accuracy:.1f}% accuracy (<80%)")
        return False

if __name__ == "__main__":
    success = test_model_accuracy()
    if success:
        print("\n🎉 Model meets the 80%+ accuracy requirement!")
    else:
        print("\n😞 Model does not meet the 80%+ accuracy requirement.")
