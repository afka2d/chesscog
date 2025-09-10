#!/usr/bin/env python3
"""
Test the piece classification model directly to debug the knight issue.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

def test_model_directly():
    """Test the model with sample images from the dataset."""
    print("🔍 Testing Model Directly")
    print("=" * 50)
    
    # Load the model
    model_path = "models/piece_classifier/InceptionV3.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"📦 Loading model from {model_path}")
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    model.eval()
    
    # Define transforms (matching the InceptionV3 configuration)
    transforms_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Define class names
    piece_classes = [
        'black_bishop', 'black_king', 'black_knight', 'black_pawn', 'black_queen', 'black_rook',
        'white_bishop', 'white_king', 'white_knight', 'white_pawn', 'white_queen', 'white_rook'
    ]
    
    print(f"📋 Classes: {piece_classes}")
    
    # Test with sample images from each class
    test_images = [
        "grey_background_dataset/pieces/test/white_king/NEW_20250805_135338_005_b3.png",
        "grey_background_dataset/pieces/test/white_queen/NEW_20250805_135338_002_a4.png",
        "grey_background_dataset/pieces/test/white_rook/NEW_20250805_135338_011_h1.png",
        "grey_background_dataset/pieces/test/white_bishop/NEW_20250805_135338_009_b2.png",
        "grey_background_dataset/pieces/test/white_knight/NEW_20250805_135338_006_g2.png",
        "grey_background_dataset/pieces/test/white_pawn/NEW_20250805_135338_008_f3.png",
        "grey_background_dataset/pieces/test/black_king/IMG_4752_d7.png",
        "grey_background_dataset/pieces/test/black_queen/NEW_20250805_135338_008_g6.png",
        "grey_background_dataset/pieces/test/black_rook/NEW_20250805_135338_007_a8.png",
        "grey_background_dataset/pieces/test/black_bishop/NEW_20250805_135338_011_e5.png",
        "grey_background_dataset/pieces/test/black_knight/NEW_20250805_135338_003_d2.png",
        "grey_background_dataset/pieces/test/black_pawn/NEW_20250805_135338_005_c7.png"
    ]
    
    print(f"\n🧪 Testing with {len(test_images)} sample images:")
    
    predictions = []
    for i, image_path in enumerate(test_images):
        if not os.path.exists(image_path):
            print(f"   {i+1:2d}. ❌ {os.path.basename(image_path)}: Not found")
            continue
            
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            img_tensor = transforms_test(img_array).unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            predicted_name = piece_classes[predicted_class]
            expected_name = os.path.basename(os.path.dirname(image_path))
            
            correct = predicted_name == expected_name
            status = "✅" if correct else "❌"
            
            print(f"   {i+1:2d}. {status} {os.path.basename(image_path)}: {predicted_name} (conf: {confidence:.3f})")
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
            print(f"   {i+1:2d}. ❌ {os.path.basename(image_path)}: Error - {e}")
    
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
        
        # Show confidence distribution
        confidences = [p['confidence'] for p in predictions]
        avg_confidence = np.mean(confidences)
        print(f"   Average confidence: {avg_confidence:.3f}")
        
        # Show class distribution
        class_counts = {}
        for p in predictions:
            piece_type = p['predicted'].split('_')[1]  # Extract piece type
            class_counts[piece_type] = class_counts.get(piece_type, 0) + 1
        
        print(f"   Predicted piece types: {dict(class_counts)}")

if __name__ == "__main__":
    test_model_directly()
