#!/usr/bin/env python3
"""
Debug the complete API pipeline to find where it differs from validation
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import json
import chess

# Piece type labels
PIECE_TYPE_LABELS = {0: "pawn", 1: "knight", 2: "bishop", 3: "rook", 4: "queen", 5: "king"}
COLOR_LABELS = {0: "black", 1: "white"}

def _get_piece_type_model_architecture(num_classes):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def load_combined_piece_classifier():
    """Load the combined piece classification model."""
    try:
        model_path = Path("models_marshall_improved/piece_classification_combined_marshall.pt")
        if not model_path.exists():
            print(f"❌ Combined piece classifier not found at {model_path}")
            return None
        
        model = _get_piece_type_model_architecture(len(PIECE_TYPE_LABELS))
        state_dict = torch.load(str(model_path), map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ Combined piece classifier loaded successfully")
        return model
        
    except Exception as e:
        print(f"❌ Error loading combined piece classifier: {e}")
        return None

def sort_corner_points(corners):
    """Sort corners to ensure correct order: top-left, top-right, bottom-right, bottom-left."""
    corners = np.array(corners, dtype=np.float32)
    center = np.mean(corners, axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    sorted_corners = corners[sorted_indices]
    sums = np.sum(sorted_corners, axis=1)
    top_left_idx = np.argmin(sums)
    reordered_corners = np.roll(sorted_corners, -top_left_idx, axis=0)
    return reordered_corners

def warp_chessboard(img_array, corners_array):
    """Warp chessboard using the exact logic from the working commit."""
    corners = sort_corner_points(corners_array)
    board_size = 800
    dst_points = np.array([
        [0, 0],                           # top-left
        [board_size - 1, 0],             # top-right
        [board_size - 1, board_size - 1], # bottom-right
        [0, board_size - 1]              # bottom-left
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(corners, dst_points)
    warped = cv2.warpPerspective(img_array, M, (board_size, board_size))
    return warped

def extract_square(warped_board, rank, file):
    """Extract a single square from the warped board using exact logic from working commit."""
    board_size = warped_board.shape[0]
    square_size = board_size // 8
    
    x1 = file * square_size
    y1 = rank * square_size
    x2 = x1 + square_size
    y2 = y1 + square_size
    
    square = warped_board[y1:y2, x1:x2]
    return square

def test_api_pipeline():
    """Test the complete API pipeline"""
    print("🧪 Testing Complete API Pipeline...")
    print("=" * 60)
    
    # Load a test image and annotation
    test_image_path = Path("yolo_detection_IMG_4763.jpg")
    if not test_image_path.exists():
        print("❌ Test image not found")
        return
    
    print(f"📸 Using test image: {test_image_path}")
    
    # Load image
    img = cv2.imread(str(test_image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"   Image shape: {img.shape}")
    
    # Use some test corners (you can replace with actual corners from annotations)
    test_corners = [[100, 100], [500, 100], [500, 500], [100, 500]]
    print(f"   Test corners: {test_corners}")
    
    # Warp chessboard
    warped_board = warp_chessboard(img, np.array(test_corners, dtype=np.float32))
    print(f"   Warped board shape: {warped_board.shape}")
    
    # Load model
    model = load_combined_piece_classifier()
    if model is None:
        return
    
    # Test piece classification on a few squares
    piece_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\n🔍 Testing piece classification on sample squares:")
    test_squares = [(0, 0), (0, 1), (1, 0), (1, 1), (7, 7)]  # Some sample squares
    
    for rank, file in test_squares:
        square = extract_square(warped_board, rank, file)
        square_name = f"{chr(97+file)}{8-rank}"
        
        # Convert to PIL Image
        square_pil = Image.fromarray(square)
        
        # Apply transforms
        square_tensor = piece_transform(square_pil).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            output = model(square_tensor)
            pred = torch.argmax(output, dim=1).item()
            conf = torch.softmax(output, dim=1)[0][pred].item()
        
        print(f"   Square {square_name} (rank={rank}, file={file}): {PIECE_TYPE_LABELS[pred]} (conf: {conf:.3f})")
    
    print("\n" + "=" * 60)
    print("🎯 API Pipeline Test Complete!")

if __name__ == "__main__":
    test_api_pipeline()
