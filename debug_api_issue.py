#!/usr/bin/env python3
"""
Debug the API issue by testing the exact same flow.
"""

import numpy as np
import chess
from PIL import Image
from simple_piece_classifier import SimplePieceClassifier
from chesscog.recognition.recognition import ChessRecognizer
from pathlib import Path

def test_api_flow():
    """Test the exact API flow to find the bug."""
    print("🔍 Debugging API Flow")
    print("=" * 40)
    
    # Load image
    img = Image.open('grey_background_dataset/images/test/IMG_4763.JPG').convert('RGB')
    img_array = np.array(img)
    
    # Test corners
    corners_array = np.array([[724, 2064], [2692, 1886], [2784, 4104], [441, 3979]], dtype=np.float32)
    
    # Initialize recognizer
    recognizer = ChessRecognizer(Path("models"))
    
    # Get occupancy using original recognizer
    print("1. Getting occupancy...")
    board, detected_corners = recognizer.predict(img_array, chess.WHITE)
    
    # Get occupancy from the board
    occupancy = np.zeros(64, dtype=bool)
    for square in chess.SQUARES:
        if board.piece_at(square) is not None:
            occupancy[square] = True
    
    print(f"   Occupancy: {sum(occupancy)} occupied squares")
    
    # Convert occupancy to a simple Python list
    occupancy_list = occupancy.tolist()
    print(f"   Occupancy list type: {type(occupancy_list[0])}")
    
    # Test the custom piece classifier directly
    print("2. Testing custom piece classifier...")
    classifier = SimplePieceClassifier()
    
    try:
        pieces = classifier.classify_pieces(img_array, corners_array, occupancy_list, chess.WHITE)
        print(f"   Success! Detected {len([p for p in pieces if p is not None])} pieces")
        return True
    except Exception as e:
        print(f"   Error in piece classifier: {e}")
        return False

if __name__ == "__main__":
    success = test_api_flow()
    if success:
        print("\n✅ API flow works correctly!")
    else:
        print("\n❌ API flow has issues!")
