#!/usr/bin/env python3
"""
Comprehensive debug test to find the exact issue.
"""

import numpy as np
import chess
from PIL import Image
from simple_piece_classifier import SimplePieceClassifier
from chesscog.recognition.recognition import ChessRecognizer
from pathlib import Path

def debug_step_by_step():
    """Debug each step of the API flow."""
    print("🔍 Comprehensive Debug Test")
    print("=" * 50)
    
    # Load image
    img = Image.open('grey_background_dataset/images/test/IMG_4763.JPG').convert('RGB')
    img_array = np.array(img)
    print(f"✅ Image loaded: {img_array.shape}")
    
    # Test corners
    corners_array = np.array([[724, 2064], [2692, 1886], [2784, 4104], [441, 3979]], dtype=np.float32)
    print(f"✅ Corners: {corners_array.shape}")
    
    # Initialize recognizer
    recognizer = ChessRecognizer(Path("models"))
    print("✅ Recognizer initialized")
    
    # Get occupancy using original recognizer
    print("1. Getting occupancy...")
    board, detected_corners = recognizer.predict(img_array, chess.WHITE)
    print(f"   Board: {board}")
    
    # Get occupancy from the board
    occupancy = np.zeros(64, dtype=bool)
    for square in chess.SQUARES:
        if board.piece_at(square) is not None:
            occupancy[square] = True
    
    print(f"   Occupancy array: {sum(occupancy)} occupied squares")
    print(f"   Occupancy type: {type(occupancy)}")
    print(f"   Occupancy dtype: {occupancy.dtype}")
    
    # Convert occupancy to a simple Python list
    occupancy_list = occupancy.tolist()
    print(f"   Occupancy list type: {type(occupancy_list)}")
    print(f"   First few elements: {occupancy_list[:10]}")
    print(f"   Element types: {[type(x) for x in occupancy_list[:10]]}")
    
    # Test the custom piece classifier directly
    print("2. Testing custom piece classifier...")
    classifier = SimplePieceClassifier()
    
    try:
        print("   Calling classify_pieces...")
        pieces = classifier.classify_pieces(img_array, corners_array, occupancy_list, chess.WHITE)
        print(f"   Success! Detected {len([p for p in pieces if p is not None])} pieces")
        
        # Test the pieces
        piece_types = set(p for p in pieces if p is not None)
        print(f"   Piece types: {list(piece_types)}")
        print(f"   Diversity: {len(piece_types)}/12")
        
        return True
    except Exception as e:
        print(f"   Error in piece classifier: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_step_by_step()
    if success:
        print("\n🎉 All steps work correctly!")
    else:
        print("\n❌ Error found in the process!")
