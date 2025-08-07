#!/usr/bin/env python3
"""
Debug piece classifier to see what's wrong with piece classification
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from torchvision import transforms

def test_piece_classifier():
    """Test the piece classifier with multiple piece images."""
    
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
    
    # Test multiple piece types
    pieces_dir = Path("grey_background_dataset/pieces/train")
    if pieces_dir.exists():
        for piece_type in pieces_dir.iterdir():
            if piece_type.is_dir():
                print(f"\n=== Testing {piece_type.name} ===")
                for img_file in piece_type.iterdir():
                    if img_file.suffix == '.png':
                        print(f"Image: {img_file.name}")
                        img = cv2.imread(str(img_file))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        with torch.no_grad():
                            input_tensor = piece_transform(img).unsqueeze(0)
                            output = piece_model(input_tensor)
                            probs = torch.softmax(output, dim=1)
                            
                            # Get top 3 predictions
                            top_probs, top_indices = torch.topk(probs[0], 3)
                            
                            print(f"Expected: {piece_type.name}")
                            print("Top 3 predictions:")
                            for i in range(3):
                                print(f"  {piece_classes[top_indices[i]]}: {top_probs[i]:.3f}")
                            
                            # Check if correct prediction is in top 3
                            expected_idx = piece_classes.index(piece_type.name)
                            if expected_idx in top_indices:
                                rank = (top_indices == expected_idx).nonzero(as_tuple=True)[0][0]
                                print(f"✅ Correct prediction found at rank {rank + 1}")
                            else:
                                print("❌ Correct prediction not in top 3")
                        break  # Only test one image per piece type

if __name__ == "__main__":
    test_piece_classifier() 