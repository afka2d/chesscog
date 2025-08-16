#!/usr/bin/env python3
"""
Test the newly trained piece classifier with the balanced test set
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from torchvision import transforms

def test_new_model():
    """Test the new piece classifier with the balanced test set."""
    
    # Load piece classifier model
    piece_model_path = Path("runs/piece_classifier/ResNet/ResNet.pt")
    piece_model = torch.load(str(piece_model_path), map_location='cpu', weights_only=False)
    piece_model.eval()
    
    # Define transform
    piece_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((100, 200)),  # Match training config
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Piece class mapping
    piece_classes = [
        'black_bishop', 'black_king', 'black_knight', 'black_pawn', 
        'black_queen', 'black_rook', 'white_bishop', 'white_king', 
        'white_knight', 'white_pawn', 'white_queen', 'white_rook'
    ]
    
    # Test with the balanced test set
    test_dir = Path("grey_background_dataset/pieces/test")
    if not test_dir.exists():
        print("Test directory not found. Run create_balanced_test_set.py first.")
        return
    
    correct_predictions = 0
    total_predictions = 0
    
    for piece_class in piece_classes:
        test_piece_dir = test_dir / piece_class
        if not test_piece_dir.exists():
            continue
            
        print(f"\n=== Testing {piece_class} ===")
        class_correct = 0
        class_total = 0
        
        for img_file in test_piece_dir.iterdir():
            if img_file.suffix == '.png':
                img = cv2.imread(str(img_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                with torch.no_grad():
                    input_tensor = piece_transform(img).unsqueeze(0)
                    output = piece_model(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    
                    # Get prediction
                    prediction = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][prediction].item()
                    
                    predicted_class = piece_classes[prediction]
                    is_correct = predicted_class == piece_class
                    
                    if is_correct:
                        class_correct += 1
                        correct_predictions += 1
                    
                    class_total += 1
                    total_predictions += 1
                    
                    print(f"  {img_file.name}: {predicted_class} (conf: {confidence:.3f}) {'✅' if is_correct else '❌'}")
        
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        print(f"  {piece_class} accuracy: {class_accuracy:.3f} ({class_correct}/{class_total})")
    
    overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\n=== Overall Accuracy: {overall_accuracy:.3f} ({correct_predictions}/{total_predictions}) ===")

if __name__ == "__main__":
    test_new_model() 