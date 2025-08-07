#!/usr/bin/env python3
"""
Debug script to test the visualization functions directly.
"""

import cv2
import numpy as np
import chess
from pathlib import Path
import sys
sys.path.append('.')

from chesscog.recognition.recognition import ChessRecognizer
from chesscog.corner_detection import find_corners
from chesscog.corner_detection.detect_corners import CN

def test_visualization():
    """Test the visualization functions directly."""
    
    print("=== Testing Visualization Functions ===")
    
    # Load configuration and recognizer
    cfg = CN.load_yaml_with_base("config/corner_detection.yaml")
    recognizer = ChessRecognizer(Path("models"))
    
    # Load test image
    img = cv2.imread("IMG_4587.jpg")
    if img is None:
        print("Failed to load test image")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Image loaded: {img.shape}")
    
    try:
        # Test the new predict_with_debug method
        print("Testing predict_with_debug...")
        board, corners, debug_images = recognizer.predict_with_debug(img, chess.WHITE)
        
        print(f"Board FEN: {board.fen()}")
        print(f"Corners shape: {corners.shape}")
        print(f"Debug images keys: {list(debug_images.keys())}")
        
        # Check each debug image
        for key, img_data in debug_images.items():
            if img_data is not None:
                print(f"✓ {key}: {img_data.shape if hasattr(img_data, 'shape') else type(img_data)}")
                # Save the image
                cv2.imwrite(f"debug_{key}.png", cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR))
                print(f"  Saved as debug_{key}.png")
            else:
                print(f"✗ {key}: None")
        
        # Test visualization functions directly
        print("\nTesting visualization functions directly...")
        
        # Get warped board
        from chesscog.occupancy_classifier.create_dataset import warp_chessboard_image
        warped_board = warp_chessboard_image(img, corners)
        print(f"Warped board shape: {warped_board.shape}")
        
        # Test occupancy classification
        occupancy = recognizer._classify_occupancy(img, chess.WHITE, corners)
        print(f"Occupancy shape: {occupancy.shape}")
        print(f"Occupancy sum: {np.sum(occupancy)}")
        
        # Test occupancy visualization
        occupancy_vis = recognizer._visualize_occupancy_map(warped_board, occupancy, chess.WHITE)
        print(f"Occupancy visualization shape: {occupancy_vis.shape}")
        cv2.imwrite("debug_occupancy_vis.png", cv2.cvtColor(occupancy_vis, cv2.COLOR_RGB2BGR))
        print("Saved occupancy visualization as debug_occupancy_vis.png")
        
        # Test piece classification
        pieces = recognizer._classify_pieces(img, chess.WHITE, corners, occupancy)
        print(f"Pieces array length: {len(pieces)}")
        print(f"Non-None pieces: {sum(1 for p in pieces if p is not None)}")
        
        # Test piece visualization
        piece_vis = recognizer._visualize_piece_map(warped_board, pieces, occupancy, chess.WHITE)
        print(f"Piece visualization shape: {piece_vis.shape}")
        cv2.imwrite("debug_piece_vis.png", cv2.cvtColor(piece_vis, cv2.COLOR_RGB2BGR))
        print("Saved piece visualization as debug_piece_vis.png")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_visualization() 