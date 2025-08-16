#!/usr/bin/env python3
"""
Debug script to test the occupancy classifier with known good images.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import json
from chesscog.recognition.recognition import ChessRecognizer
from chesscog.occupancy_classifier.create_dataset import warp_chessboard_image, crop_square
from chesscog.core.dataset.dataset import build_transforms, Datasets
from chesscog.corner_detection.detect_corners import CN
import chess
from PIL import Image
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def test_occupancy_with_known_image():
    """Test occupancy classifier with a known good image."""
    
    print("=== Debugging Occupancy Classifier ===")
    
    # Test with a known image that should have pieces
    test_image_path = "grey_background_dataset/images/test/IMG_4752.JPG"
    annotation_path = "grey_background_dataset/annotations/test/IMG_4752.json"
    
    if not Path(test_image_path).exists():
        print(f"❌ Test image not found: {test_image_path}")
        return
    
    if not Path(annotation_path).exists():
        print(f"❌ Annotation not found: {annotation_path}")
        return
    
    # Load image and annotation
    img = cv2.imread(test_image_path)
    print(f"✅ Image loaded: {img.shape}")
    
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    
    corners = np.array(annotation['corners'], dtype=np.float32)
    expected_fen = annotation['fen']
    print(f"✅ Expected FEN: {expected_fen}")
    print(f"✅ Corners: {corners}")
    
    # Load occupancy classifier directly
    try:
        models_path = Path("models")
        recognizer = ChessRecognizer(models_path)
        print("✅ ChessRecognizer loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load recognizer: {e}")
        return
    
    # Test occupancy classification
    try:
        print("\n--- Testing Occupancy Classification ---")
        occupancy = recognizer._classify_occupancy(img, chess.WHITE, corners)
        occupied_count = np.sum(occupancy)
        print(f"Occupancy result: {occupied_count} occupied squares out of 64")
        
        if occupied_count > 0:
            occupied_squares = [chess.square_name(sq) for sq, occ in zip(chess.SQUARES, occupancy) if occ]
            print(f"Occupied squares: {occupied_squares}")
        else:
            print("❌ NO OCCUPIED SQUARES FOUND - This is the problem!")
            
            # Debug further - let's look at individual square predictions
            print("\n--- Debugging Individual Squares ---")
            
            # Warp the image
            warped = warp_chessboard_image(img, corners)
            print(f"Warped image shape: {warped.shape}")
            
            # Test a few squares manually
            test_squares = [chess.A1, chess.E1, chess.E8, chess.A8]  # Corner squares
            
            for square in test_squares:
                try:
                    square_img = crop_square(warped, square, turn=chess.WHITE)
                    square_pil = Image.fromarray(square_img)
                    
                    # Apply transforms
                    transformed = recognizer._occupancy_transforms(square_pil)
                    transformed = transformed.unsqueeze(0)
                    
                    # Predict
                    with torch.no_grad():
                        prediction = recognizer._occupancy_model(transformed)
                        probabilities = torch.softmax(prediction, dim=1)
                        predicted_class = torch.argmax(probabilities, dim=1).item()
                        confidence = probabilities[0][predicted_class].item()
                        
                        is_occupied = predicted_class == recognizer._occupancy_cfg.DATASET.CLASSES.index("occupied")
                        
                        print(f"Square {chess.square_name(square)}: {'occupied' if is_occupied else 'empty'} (confidence: {confidence:.3f})")
                        
                except Exception as square_error:
                    print(f"Error processing square {chess.square_name(square)}: {square_error}")
        
    except Exception as e:
        print(f"❌ Occupancy classification failed: {e}")
        import traceback
        traceback.print_exc()
        
    # Also test with the runs model for comparison
    print("\n--- Testing with Runs Model ---")
    try:
        runs_occupancy_path = Path("runs/occupancy_classifier/ResNet/ResNet.pt")
        if runs_occupancy_path.exists():
            runs_model = torch.load(runs_occupancy_path, map_location='cpu', weights_only=False)
            runs_model.eval()
            
            # Test the same image with runs model
            warped = warp_chessboard_image(img, corners)
            
            # Process a test square
            test_square = chess.E1
            square_img = crop_square(warped, test_square, turn=chess.WHITE)
            square_pil = Image.fromarray(square_img)
            transformed = recognizer._occupancy_transforms(square_pil)
            transformed = transformed.unsqueeze(0)
            
            with torch.no_grad():
                prediction = runs_model(transformed)
                probabilities = torch.softmax(prediction, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
                print(f"Runs model - Square {chess.square_name(test_square)}: class {predicted_class} (confidence: {confidence:.3f})")
                
        else:
            print("Runs occupancy model not found")
            
    except Exception as e:
        print(f"Error testing runs model: {e}")

if __name__ == "__main__":
    test_occupancy_with_known_image()