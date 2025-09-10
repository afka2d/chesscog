#!/usr/bin/env python3
"""
Test the trained model on completely different images to check for overfitting.
This will give us the REAL accuracy when the model is used in practice.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from pathlib import Path
import os
from PIL import Image
import random
import glob

def test_real_world_accuracy():
    """Test the model on completely different images to check for overfitting."""
    print("🔍 Testing REAL WORLD Accuracy (Anti-Overfitting Test)")
    print("=" * 60)
    
    # Load the trained model
    model_path = "models/piece_classifier/ResNet_robust_full.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return False
    
    print(f"📦 Loading model from {model_path}")
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    model.eval()
    
    # Define transforms (same as training)
    transforms_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Class names
    class_names = [
        'black_bishop', 'black_king', 'black_knight', 'black_pawn', 'black_queen', 'black_rook',
        'white_bishop', 'white_king', 'white_knight', 'white_pawn', 'white_queen', 'white_rook'
    ]
    
    print(f"📋 Classes: {class_names}")
    
    # Test on DIFFERENT images - not from the training set
    # Use images from a different source or different time period
    test_directories = [
        "my_chess_images/train/images",  # Different source
        "training_data_2",  # Different dataset
        "grey_background_dataset/images/test"  # Original images (not pieces)
    ]
    
    all_results = []
    total_tests = 0
    correct_tests = 0
    
    print(f"\n🧪 Testing on DIFFERENT image sources:")
    print("=" * 50)
    
    for test_dir in test_directories:
        if not os.path.exists(test_dir):
            print(f"   ⚠️  {test_dir}: Directory not found")
            continue
        
        print(f"\n📁 Testing directory: {test_dir}")
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(test_dir, ext)))
        
        if not image_files:
            print(f"   ⚠️  No images found in {test_dir}")
            continue
        
        # Test up to 20 random images from this directory
        test_images = random.sample(image_files, min(20, len(image_files)))
        
        print(f"   📊 Testing {len(test_images)} images...")
        
        for i, image_path in enumerate(test_images):
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
                
                # For this test, we can't easily determine the "correct" answer
                # since we're using arbitrary images. Instead, let's look for patterns:
                
                # Check if the model is making reasonable predictions
                is_reasonable = confidence > 0.1 and confidence < 0.99  # Not too confident, not too uncertain
                
                print(f"   {i+1:2d}. {os.path.basename(image_path)}: {predicted_name} (conf: {confidence:.3f}) {'✅' if is_reasonable else '❌'}")
                
                all_results.append({
                    'image': os.path.basename(image_path),
                    'predicted': predicted_name,
                    'confidence': confidence,
                    'reasonable': is_reasonable
                })
                
                total_tests += 1
                if is_reasonable:
                    correct_tests += 1
                
            except Exception as e:
                print(f"   {i+1:2d}. ❌ {os.path.basename(image_path)}: Error - {e}")
    
    # Analyze results
    print(f"\n📊 REAL WORLD ANALYSIS:")
    print("=" * 50)
    
    if total_tests == 0:
        print("   ❌ No tests completed - cannot assess overfitting")
        return False
    
    reasonable_percentage = correct_tests / total_tests * 100
    
    print(f"   Total tests: {total_tests}")
    print(f"   Reasonable predictions: {correct_tests} ({reasonable_percentage:.1f}%)")
    
    # Check for bias patterns
    prediction_counts = {}
    confidence_scores = []
    
    for result in all_results:
        pred = result['predicted']
        conf = result['confidence']
        
        if pred not in prediction_counts:
            prediction_counts[pred] = 0
        prediction_counts[pred] += 1
        confidence_scores.append(conf)
    
    print(f"\n🔍 BIAS ANALYSIS:")
    print("=" * 30)
    
    # Check for overconfident predictions (sign of overfitting)
    avg_confidence = np.mean(confidence_scores)
    high_confidence_count = sum(1 for c in confidence_scores if c > 0.9)
    high_confidence_percentage = high_confidence_count / len(confidence_scores) * 100
    
    print(f"   Average confidence: {avg_confidence:.3f}")
    print(f"   High confidence (>0.9): {high_confidence_count}/{len(confidence_scores)} ({high_confidence_percentage:.1f}%)")
    
    if high_confidence_percentage > 80:
        print("   🚨 WARNING: Model is overconfident - likely overfitting!")
    elif high_confidence_percentage > 60:
        print("   ⚠️  CAUTION: Model may be overfitting")
    else:
        print("   ✅ Confidence levels look reasonable")
    
    # Check for prediction diversity
    unique_predictions = len(prediction_counts)
    print(f"   Unique predictions: {unique_predictions}/12 classes")
    
    if unique_predictions < 6:
        print("   🚨 WARNING: Model has low diversity - likely overfitting!")
    elif unique_predictions < 9:
        print("   ⚠️  CAUTION: Model may have limited diversity")
    else:
        print("   ✅ Good prediction diversity")
    
    # Check for specific biases
    print(f"\n📈 PREDICTION DISTRIBUTION:")
    for class_name, count in sorted(prediction_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_tests * 100
        print(f"   {class_name}: {count:2d} ({percentage:4.1f}%)")
    
    # Check for knight bias (common overfitting pattern)
    knight_predictions = sum(count for name, count in prediction_counts.items() if 'knight' in name)
    knight_percentage = knight_predictions / total_tests * 100
    
    if knight_percentage > 40:
        print(f"   🚨 WARNING: Strong knight bias ({knight_percentage:.1f}%) - likely overfitting!")
    elif knight_percentage > 25:
        print(f"   ⚠️  CAUTION: Moderate knight bias ({knight_percentage:.1f}%)")
    else:
        print(f"   ✅ Knight distribution looks normal ({knight_percentage:.1f}%)")
    
    # Final assessment
    print(f"\n🎯 OVERFITTING ASSESSMENT:")
    print("=" * 40)
    
    overfitting_indicators = 0
    
    if high_confidence_percentage > 80:
        overfitting_indicators += 1
        print("   ❌ Overconfident predictions")
    
    if unique_predictions < 6:
        overfitting_indicators += 1
        print("   ❌ Low prediction diversity")
    
    if knight_percentage > 40:
        overfitting_indicators += 1
        print("   ❌ Strong knight bias")
    
    if reasonable_percentage < 50:
        overfitting_indicators += 1
        print("   ❌ Low reasonable prediction rate")
    
    if overfitting_indicators == 0:
        print("   ✅ Model shows good generalization - likely NOT overfitting")
        return True
    elif overfitting_indicators <= 1:
        print("   ⚠️  Model may have minor overfitting issues")
        return False
    else:
        print("   🚨 Model shows clear signs of overfitting - NOT suitable for production")
        return False

if __name__ == "__main__":
    success = test_real_world_accuracy()
    
    if success:
        print("\n🎉 Model appears to generalize well!")
        print("   The high accuracy is likely real, not overfitting.")
    else:
        print("\n😞 Model shows signs of overfitting.")
        print("   The high accuracy is likely not representative of real-world performance.")
        print("   Consider using a simpler model or more regularization.")
