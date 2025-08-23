#!/usr/bin/env python3
"""
Test the original ResNet model to see if it performs better than the broken ResNet_uniform.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from torchvision import transforms
import time

def test_original_model():
    """Test the original ResNet model performance."""
    
    print("🧪 Testing Original ResNet Model")
    print("=" * 50)
    
    # Test both models
    models_to_test = [
        ("Original ResNet", "runs/piece_classifier/ResNet/ResNet.pt"),
        ("ResNet_uniform", "runs/piece_classifier/ResNet_uniform/ResNet_uniform.pt")
    ]
    
    for model_name, model_path in models_to_test:
        print(f"\n🔍 Testing {model_name}")
        print("-" * 40)
        
        if not Path(model_path).exists():
            print(f"❌ Model not found: {model_path}")
            continue
        
        # Load model
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = torch.load(str(model_path), map_location=device, weights_only=False)
            model.eval()
            print(f"✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            continue
        
        # Define transforms (use the correct size for each model)
        if "uniform" in model_path:
            transform_size = (224, 448)  # ResNet_uniform training size
        else:
            transform_size = (100, 200)  # Original ResNet training size
        
        piece_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(transform_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"📊 Using transform size: {transform_size}")
        
        # Piece classes
        piece_classes = [
            'black_bishop', 'black_king', 'black_knight', 'black_pawn', 
            'black_queen', 'black_rook', 'white_bishop', 'white_king', 
            'white_knight', 'white_pawn', 'white_queen', 'white_rook'
        ]
        
        # Test on a small sample
        test_dir = Path("grey_background_dataset/pieces/test")
        if not test_dir.exists():
            print(f"❌ Test directory not found")
            continue
        
        # Quick test with 5 images per class
        total_correct = 0
        total_tested = 0
        prediction_counts = {}
        
        start_time = time.time()
        
        for piece_class in piece_classes:
            class_dir = test_dir / piece_class
            if not class_dir.exists():
                continue
            
            # Test first 5 images
            test_images = list(class_dir.glob("*.png"))[:5]
            class_correct = 0
            
            for img_path in test_images:
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    input_tensor = piece_transform(img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                        probabilities = torch.softmax(output, dim=1)
                        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                        confidence = probabilities[0][predicted_class_idx].item()
                    
                    predicted_class = piece_classes[predicted_class_idx]
                    is_correct = predicted_class == piece_class
                    
                    if is_correct:
                        class_correct += 1
                        total_correct += 1
                    
                    total_tested += 1
                    
                    # Track predictions
                    if predicted_class not in prediction_counts:
                        prediction_counts[predicted_class] = 0
                    prediction_counts[predicted_class] += 1
                    
                except Exception as e:
                    continue
            
            if len(test_images) > 0:
                accuracy = class_correct / len(test_images) * 100
                print(f"  {piece_class:15}: {class_correct}/{len(test_images)} = {accuracy:.1f}%")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Overall results
        overall_accuracy = (total_correct / total_tested * 100) if total_tested > 0 else 0
        print(f"\n📊 Overall Accuracy: {overall_accuracy:.1f}% ({total_correct}/{total_tested})")
        print(f"⏱️  Processing Time: {processing_time:.2f} seconds")
        
        # Prediction distribution
        print(f"\n🎯 Prediction Distribution (Top 5):")
        sorted_predictions = sorted(prediction_counts.items(), key=lambda x: x[1], reverse=True)
        for predicted_class, count in sorted_predictions[:5]:
            percentage = (count / total_tested) * 100
            print(f"  {predicted_class:15}: {count:2d} ({percentage:5.1f}%)")
        
        # Check for model bias
        if prediction_counts:
            max_pred = max(prediction_counts.values())
            if max_pred > total_tested * 0.4:
                print(f"⚠️  WARNING: Model shows bias toward {max(prediction_counts, key=prediction_counts.get)}")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    test_original_model()

