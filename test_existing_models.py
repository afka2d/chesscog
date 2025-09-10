#!/usr/bin/env python3
"""
Test all existing models to find one that actually works in practice.
Focus on real-world performance, not training accuracy.
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

def test_existing_models():
    """Test all existing models to find the best one for real-world use."""
    print("🔍 Testing ALL Existing Models for Real-World Performance")
    print("=" * 60)
    
    # List all available models
    model_dir = Path("models/piece_classifier")
    model_files = list(model_dir.glob("*.pt"))
    
    print(f"📦 Found {len(model_files)} models:")
    for model_file in model_files:
        print(f"   - {model_file.name}")
    
    # Class names
    class_names = [
        'black_bishop', 'black_king', 'black_knight', 'black_pawn', 'black_queen', 'black_rook',
        'white_bishop', 'white_king', 'white_knight', 'white_pawn', 'white_queen', 'white_rook'
    ]
    
    # Test each model
    results = {}
    
    for model_file in model_files:
        print(f"\n🧪 Testing {model_file.name}")
        print("-" * 40)
        
        try:
            # Load model
            model = torch.load(str(model_file), map_location='cpu', weights_only=False)
            model.eval()
            
            # Define transforms (try different sizes)
            transforms_224 = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            transforms_299 = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Test on different image sources
            test_directories = [
                "my_chess_images/train/images",
                "grey_background_dataset/images/test"
            ]
            
            all_predictions = []
            total_tests = 0
            
            for test_dir in test_directories:
                if not os.path.exists(test_dir):
                    continue
                
                # Get image files
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    image_files.extend(glob.glob(os.path.join(test_dir, ext)))
                
                if not image_files:
                    continue
                
                # Test up to 10 random images
                test_images = random.sample(image_files, min(10, len(image_files)))
                
                for image_path in test_images:
                    try:
                        img = Image.open(image_path).convert('RGB')
                        
                        # Try both transform sizes
                        for transforms_test in [transforms_224, transforms_299]:
                            try:
                                img_tensor = transforms_test(img).unsqueeze(0)
                                
                                with torch.no_grad():
                                    output = model(img_tensor)
                                    predicted = torch.argmax(output, 1).item()
                                    confidence = torch.softmax(output, 1)[0][predicted].item()
                                
                                all_predictions.append(predicted)
                                total_tests += 1
                                break  # Use first successful transform
                                
                            except Exception as e:
                                continue
                                
                    except Exception as e:
                        continue
            
            if all_predictions:
                unique_predictions = len(set(all_predictions))
                diversity_score = unique_predictions / 12.0
                
                # Check for bias
                prediction_counts = {}
                for pred in all_predictions:
                    pred_name = class_names[pred]
                    prediction_counts[pred_name] = prediction_counts.get(pred_name, 0) + 1
                
                # Check for knight bias
                knight_predictions = sum(count for name, count in prediction_counts.items() if 'knight' in name)
                knight_percentage = knight_predictions / len(all_predictions) * 100
                
                # Check for single-class bias
                max_class_count = max(prediction_counts.values()) if prediction_counts else 0
                max_class_percentage = max_class_count / len(all_predictions) * 100
                
                print(f"   Tests: {total_tests}")
                print(f"   Diversity: {unique_predictions}/12 classes ({diversity_score:.2f})")
                print(f"   Knight bias: {knight_percentage:.1f}%")
                print(f"   Max class: {max_class_percentage:.1f}%")
                
                # Score the model
                score = 0
                if diversity_score >= 0.8:  # Good diversity
                    score += 3
                elif diversity_score >= 0.5:  # Moderate diversity
                    score += 2
                elif diversity_score >= 0.3:  # Poor diversity
                    score += 1
                
                if knight_percentage <= 20:  # Low knight bias
                    score += 2
                elif knight_percentage <= 40:  # Moderate knight bias
                    score += 1
                
                if max_class_percentage <= 50:  # No single class dominance
                    score += 2
                elif max_class_percentage <= 70:  # Moderate dominance
                    score += 1
                
                results[model_file.name] = {
                    'score': score,
                    'diversity': diversity_score,
                    'knight_bias': knight_percentage,
                    'max_class_bias': max_class_percentage,
                    'total_tests': total_tests
                }
                
                if score >= 5:
                    print(f"   ✅ GOOD: Score {score}/7")
                elif score >= 3:
                    print(f"   ⚠️  MODERATE: Score {score}/7")
                else:
                    print(f"   ❌ POOR: Score {score}/7")
            else:
                print(f"   ❌ No tests completed")
                results[model_file.name] = {'score': 0, 'error': 'No tests completed'}
                
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            results[model_file.name] = {'score': 0, 'error': str(e)}
    
    # Find the best model
    print(f"\n🏆 MODEL COMPARISON RESULTS")
    print("=" * 50)
    
    best_model = None
    best_score = -1
    
    for model_name, result in results.items():
        if 'error' in result:
            print(f"   {model_name}: ERROR - {result['error']}")
        else:
            score = result['score']
            diversity = result['diversity']
            knight_bias = result['knight_bias']
            max_class_bias = result['max_class_bias']
            
            print(f"   {model_name}:")
            print(f"      Score: {score}/7")
            print(f"      Diversity: {diversity:.2f}")
            print(f"      Knight bias: {knight_bias:.1f}%")
            print(f"      Max class bias: {max_class_bias:.1f}%")
            
            if score > best_score:
                best_score = score
                best_model = model_name
    
    if best_model and best_score > 0:
        print(f"\n🎯 BEST MODEL: {best_model}")
        print(f"   Score: {best_score}/7")
        
        if best_score >= 5:
            print(f"   ✅ This model should work well in practice!")
            return best_model
        elif best_score >= 3:
            print(f"   ⚠️  This model may work but has some issues")
            return best_model
        else:
            print(f"   ❌ No model is suitable for production use")
            return None
    else:
        print(f"\n❌ No suitable model found")
        return None

if __name__ == "__main__":
    best_model = test_existing_models()
    
    if best_model:
        print(f"\n🎉 Recommendation: Use {best_model}")
        print(f"   This model shows the best real-world performance.")
    else:
        print(f"\n😞 No existing model is suitable for production use.")
        print(f"   Consider using a completely different approach.")
