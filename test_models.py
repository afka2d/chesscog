#!/usr/bin/env python3
"""
Test models directly on a single image to verify predictions.
"""

import cv2
import numpy as np
import torch
import chess
from pathlib import Path
from torchvision import transforms
import matplotlib.pyplot as plt

def test_models(img_path, corners):
    """Test piece and occupancy classifiers on a single image."""
    # Load image
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load models (trusted source)
    piece_model = torch.load("runs/piece_classifier/ResNet/ResNet.pt", map_location='cpu', weights_only=False)
    occupancy_model = torch.load("runs/occupancy_classifier/ResNet/ResNet.pt", map_location='cpu', weights_only=False)
    
    piece_model.eval()
    occupancy_model.eval()
    
    # Define transforms
    piece_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    occupancy_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Piece class mapping
    piece_classes = [
        'black_bishop', 'black_king', 'black_knight', 'black_pawn', 
        'black_queen', 'black_rook', 'white_bishop', 'white_king', 
        'white_knight', 'white_pawn', 'white_queen', 'white_rook'
    ]
    
    # Sort and convert corners
    corners = np.array(corners, dtype=np.float32)
    
    # Define target size
    target_size = (1792, 1792)  # 8 * 224
    
    # Define target corners
    target_corners = np.array([
        [0, 0],
        [target_size[0], 0],
        [target_size[0], target_size[1]],
        [0, target_size[1]]
    ], dtype=np.float32)
    
    # Calculate perspective transform
    transform_matrix = cv2.getPerspectiveTransform(corners, target_corners)
    
    # Warp the image
    warped = cv2.warpPerspective(img, transform_matrix, target_size)
    
    # Create visualization grid
    fig, axes = plt.subplots(8, 8, figsize=(20, 20))
    fig.suptitle("Square Classifications", fontsize=16)
    
    # Process each square
    results = []
    for rank in range(8):
        for file in range(8):
            # Extract square
            x1 = file * 224
            y1 = rank * 224
            x2 = x1 + 224
            y2 = y1 + 224
            square = warped[y1:y2, x1:x2]
            
            # Test occupancy
            with torch.no_grad():
                occ_input = occupancy_transform(square).unsqueeze(0)
                occ_output = occupancy_model(occ_input)
                occ_probs = torch.softmax(occ_output, dim=1)
                is_occupied = torch.argmax(occ_probs, dim=1).item() == 1
                occ_conf = occ_probs[0][1].item()  # Confidence for "occupied" class
            
            # Test piece classification if occupied
            piece_type = None
            piece_conf = 0
            if is_occupied:
                with torch.no_grad():
                    piece_input = piece_transform(square).unsqueeze(0)
                    piece_output = piece_model(piece_input)
                    piece_probs = torch.softmax(piece_output, dim=1)
                    piece_idx = torch.argmax(piece_probs, dim=1).item()
                    piece_type = piece_classes[piece_idx]
                    piece_conf = piece_probs[0][piece_idx].item()
            
            # Store results
            results.append({
                'square': f"{chr(97+file)}{8-rank}",
                'occupied': is_occupied,
                'occ_conf': occ_conf,
                'piece': piece_type,
                'piece_conf': piece_conf
            })
            
            # Visualize
            axes[rank, file].imshow(square)
            axes[rank, file].axis('off')
            title = f"{chr(97+file)}{8-rank}\n"
            if is_occupied:
                title += f"{piece_type}\n{piece_conf:.2f}"
            axes[rank, file].set_title(title, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('square_classifications.png')
    print("✅ Visualization saved to square_classifications.png")
    
    # Print results
    print("\nSquare Classifications:")
    print("-" * 50)
    for result in results:
        if result['occupied']:
            print(f"{result['square']}: {result['piece']} (conf: {result['piece_conf']:.2f})")
        else:
            print(f"{result['square']}: empty (conf: {1-result['occ_conf']:.2f})")

if __name__ == "__main__":
    # Test with a known image
    img_path = "grey_background_dataset/images/test/IMG_4752.JPG"
    corners = [[993, 2294], [2702, 2064], [2755, 3892], [542, 3864]]
    test_models(img_path, corners)