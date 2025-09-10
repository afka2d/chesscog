#!/usr/bin/env python3
"""
Test the piece classifier directly to debug accuracy issues.
"""

import numpy as np
import chess
from PIL import Image
from simple_piece_classifier import SimplePieceClassifier
from pathlib import Path

def test_piece_classifier_direct():
    """Test piece classifier directly."""
    print("🔍 Testing Piece Classifier Directly")
    print("=" * 50)
    
    # Load test image
    img_path = "grey_background_dataset/images/test/IMG_4763.JPG"
    img = Image.open(img_path)
    img_array = np.array(img)
    
    # Test corners
    corners = [[724, 2064], [2692, 1886], [2784, 4104], [441, 3979]]
    corners_array = np.array(corners, dtype=np.float32)
    
    # Initialize piece classifier
    piece_classifier = SimplePieceClassifier(Path("models"))
    
    # Test with different occupancy patterns
    test_cases = [
        ("All occupied", [True] * 64),
        ("Realistic sparse", [True] * 6 + [False] * 58),
        ("Empty board", [False] * 64),
        ("Center pieces", [False] * 20 + [True] * 8 + [False] * 36)
    ]
    
    for name, occupancy in test_cases:
        print(f"\n📋 Test Case: {name}")
        print(f"   Occupied squares: {sum(occupancy)}/64")
        
        try:
            # Classify pieces
            pieces_1d = piece_classifier.classify_pieces(img_array, corners_array, occupancy, chess.WHITE)
            
            # Count piece types
            piece_counts = {}
            for piece in pieces_1d:
                if piece is not None:
                    piece_name = f"{'white' if piece.color else 'black'}_{piece.symbol().lower()}"
                    piece_counts[piece_name] = piece_counts.get(piece_name, 0) + 1
            
            total_pieces = sum(piece_counts.values())
            unique_types = len(piece_counts)
            
            print(f"   Pieces detected: {total_pieces}")
            print(f"   Unique types: {unique_types}")
            print(f"   Breakdown: {dict(piece_counts)}")
            
            # Check for biases
            if total_pieces > 0:
                total_pawns = piece_counts.get('white_p', 0) + piece_counts.get('black_p', 0)
                pawn_percentage = (total_pawns / total_pieces) * 100
                print(f"   Pawn percentage: {pawn_percentage:.1f}%")
                
                if pawn_percentage > 80:
                    print("   ⚠️  Severe pawn bias!")
                elif pawn_percentage > 60:
                    print("   ⚠️  Moderate pawn bias")
                else:
                    print("   ✅ Pawn distribution looks reasonable")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_piece_classifier_direct()
