#!/usr/bin/env python3
"""
Debug manual corner recognizer step by step
"""

import cv2
import numpy as np
import torch
import json
from pathlib import Path
from torchvision import transforms
from chesscog.occupancy_classifier.create_dataset import warp_chessboard_image
from chesscog.core import sort_corner_points

def debug_manual_recognizer():
    """Debug the manual corner recognizer step by step."""
    
    # Load the occupancy model
    model_path = Path("runs/occupancy_classifier/ResNet/ResNet.pt")
    model = torch.load(str(model_path), map_location='cpu', weights_only=False)
    model.eval()
    
    print("✅ Occupancy model loaded successfully")
    
    # Define transforms (must match training configuration)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((100, 100)),  # Match training config
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load image
    img_path = "grey_background_dataset/images/test/IMG_4752.JPG"
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"❌ Could not load image: {img_path}")
        return
    
    print(f"✅ Loaded image: {img.shape}")
    
    # Load manual corners
    annotation_path = "grey_background_dataset/annotations/test/IMG_4752.json"
    with open(annotation_path, 'r') as f:
        data = json.load(f)
        corners = data['corners']
    
    print(f"✅ Manual corners: {corners}")
    
    # Convert to numpy array and sort
    corners_array = np.array(corners, dtype=np.float32)
    corners_array = sort_corner_points(corners_array)
    
    # Warp the board
    warped_board = warp_chessboard_image(img, corners_array)
    print(f"✅ Warped board shape: {warped_board.shape}")
    
    # Process each square
    square_size = warped_board.shape[0] // 8
    occupied_count = 0
    
    print("\n🔍 Testing each square:")
    
    for rank in range(8):
        for file in range(8):
            # Get square coordinates
            x1 = file * square_size
            y1 = rank * square_size
            x2 = x1 + square_size
            y2 = y1 + square_size
            
            square_img = warped_board[y1:y2, x1:x2]
            
            # Check if square is occupied
            with torch.no_grad():
                occupancy_input = transform(square_img).unsqueeze(0)
                occupancy_output = model(occupancy_input)
                occupancy_probs = torch.softmax(occupancy_output, dim=1)
                is_occupied = torch.argmax(occupancy_probs, dim=1).item() == 1
                confidence = occupancy_probs[0][1 if is_occupied else 0].item()
            
            # Calculate average brightness
            avg_brightness = np.mean(square_img)
            
            # Get square name
            square_name = f"{chr(97+file)}{8-rank}"
            
            if is_occupied:
                occupied_count += 1
                status = "OCCUPIED"
            else:
                status = "EMPTY"
            
            print(f"  {square_name}: {status} (conf: {confidence:.3f}, brightness: {avg_brightness:.1f})")
            
            # Save a few squares for inspection
            if rank in [3, 4] and file in [3, 4]:  # Center squares
                cv2.imwrite(f"debug_square_{square_name}_{status.lower()}.png", square_img)
    
    print(f"\n📊 Summary:")
    print(f"  Total squares: 64")
    print(f"  Occupied squares: {occupied_count}")
    print(f"  Empty squares: {64 - occupied_count}")
    print(f"  Occupancy rate: {occupied_count/64*100:.1f}%")

if __name__ == "__main__":
    debug_manual_recognizer() 