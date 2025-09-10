#!/usr/bin/env python3
"""
Debug the occupancy detection and piece classification integration.
"""

import numpy as np
import chess
from PIL import Image
from simple_piece_classifier import SimplePieceClassifier
from chesscog.recognition.recognition import ChessRecognizer
from pathlib import Path

def debug_occupancy_integration():
    """Debug the occupancy detection and piece classification."""
    print("🔍 Debugging Occupancy Integration")
    print("=" * 50)
    
    # Load test image
    img = Image.open('grey_background_dataset/images/test/IMG_4763.JPG').convert('RGB')
    img_array = np.array(img)
    
    # Test corners
    corners_array = np.array([[724, 2064], [2692, 1886], [2784, 4104], [441, 3979]], dtype=np.float32)
    
    # Initialize recognizers
    piece_classifier = SimplePieceClassifier(Path("models"))
    occupancy_recognizer = ChessRecognizer(Path("models"))
    
    print("1. Testing occupancy detection...")
    try:
        board, detected_corners = occupancy_recognizer.predict(img_array, chess.WHITE)
        
        # Extract occupancy from the board
        occupancy = []
        for square in chess.SQUARES:
            occupancy.append(board.piece_at(square) is not None)
        
        occupied_count = sum(occupancy)
        print(f"   Occupancy detected: {occupied_count} occupied squares")
        print(f"   Occupancy type: {type(occupancy)}")
        print(f"   First 10 occupancy values: {occupancy[:10]}")
        
    except Exception as e:
        print(f"   Error in occupancy detection: {e}")
        return False
    
    print("\n2. Testing piece classification with real occupancy...")
    try:
        pieces_1d = piece_classifier.classify_pieces(img_array, corners_array, occupancy, chess.WHITE)
        
        occupied_pieces = [p for p in pieces_1d if p is not None]
        piece_types = set(occupied_pieces)
        
        print(f"   Pieces detected: {len(occupied_pieces)}")
        print(f"   Unique piece types: {len(piece_types)}")
        print(f"   Piece types: {list(piece_types)}")
        
        # Calculate diversity
        diversity = len(piece_types) / 12.0 if len(occupied_pieces) > 0 else 0
        print(f"   Diversity score: {diversity:.2f}")
        
        return True
        
    except Exception as e:
        print(f"   Error in piece classification: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_occupancy_formats():
    """Test different occupancy data formats."""
    print("\n" + "="*50)
    print("🧪 Testing Different Occupancy Formats")
    print("="*50)
    
    # Load test image
    img = Image.open('grey_background_dataset/images/test/IMG_4763.JPG').convert('RGB')
    img_array = np.array(img)
    
    # Test corners
    corners_array = np.array([[724, 2064], [2692, 1886], [2784, 4104], [441, 3979]], dtype=np.float32)
    
    piece_classifier = SimplePieceClassifier(Path("models"))
    
    # Test different occupancy formats
    test_formats = [
        {
            "name": "All True (list)",
            "occupancy": [True] * 64
        },
        {
            "name": "All True (numpy array)",
            "occupancy": np.array([True] * 64)
        },
        {
            "name": "Half True (list)",
            "occupancy": [True] * 32 + [False] * 32
        },
        {
            "name": "Half True (numpy array)",
            "occupancy": np.array([True] * 32 + [False] * 32)
        },
        {
            "name": "Few pieces (list)",
            "occupancy": [True] * 8 + [False] * 56
        },
        {
            "name": "Few pieces (numpy array)",
            "occupancy": np.array([True] * 8 + [False] * 56)
        }
    ]
    
    for test_format in test_formats:
        print(f"\n📊 Testing {test_format['name']}")
        try:
            pieces_1d = piece_classifier.classify_pieces(img_array, corners_array, test_format['occupancy'], chess.WHITE)
            
            occupied_pieces = [p for p in pieces_1d if p is not None]
            piece_types = set(occupied_pieces)
            
            print(f"   Pieces detected: {len(occupied_pieces)}")
            print(f"   Unique piece types: {len(piece_types)}")
            print(f"   Piece types: {list(piece_types)}")
            
        except Exception as e:
            print(f"   Error: {e}")

if __name__ == "__main__":
    success1 = debug_occupancy_integration()
    test_occupancy_formats()
    
    if success1:
        print("\n🎉 Occupancy integration is working!")
    else:
        print("\n❌ Occupancy integration has issues!")