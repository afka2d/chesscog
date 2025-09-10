#!/usr/bin/env python3
"""
Debug the piece classification issue with real occupancy data.
"""

import numpy as np
from PIL import Image
import chess
from simple_piece_classifier import SimplePieceClassifier
from chesscog.recognition.recognition import ChessRecognizer
from pathlib import Path

def debug_piece_classification():
    """Debug why piece classification is only detecting 4 types."""
    print("🔍 Debugging Piece Classification")
    print("=" * 50)
    
    # Load test image
    img_path = "grey_background_dataset/images/test/IMG_4763.JPG"
    img = Image.open(img_path)
    img_array = np.array(img)
    
    # Test corners
    corners = np.array([[724, 2064], [2692, 1886], [2784, 4104], [441, 3979]], dtype=np.float32)
    
    print(f"📸 Image loaded: {img_array.shape}")
    print(f"📐 Corners: {corners.shape}")
    
    # Initialize classifiers
    print("\n🔧 Initializing classifiers...")
    piece_classifier = SimplePieceClassifier(Path("models"))
    recognizer = ChessRecognizer(Path("models"))
    
    # Get real occupancy from ChessRecognizer
    print("\n🎯 Getting real occupancy...")
    board, detected_corners = recognizer.predict(img_array, chess.WHITE)
    
    # Convert board to occupancy array
    occupancy = np.zeros(64, dtype=bool)
    for square in chess.SQUARES:
        if board.piece_at(square) is not None:
            occupancy[square] = True
    
    occupied_count = np.sum(occupancy)
    print(f"   Occupied squares: {occupied_count}/64")
    
    # Test piece classification with real occupancy
    print("\n🎲 Testing piece classification...")
    try:
        pieces_1d = piece_classifier.classify_pieces(img_array, corners, occupancy.tolist(), chess.WHITE)
        print(f"   Pieces classified: {len(pieces_1d)}")
        
        # Analyze results
        piece_names = []
        for i, piece in enumerate(pieces_1d):
            if piece is not None:
                piece_name = f"{'white' if piece.color else 'black'}_{piece.symbol().lower()}"
                piece_names.append(piece_name)
                print(f"   Square {i}: {piece_name}")
        
        unique_types = set(piece_names)
        print(f"\n📊 ANALYSIS:")
        print(f"   Total pieces: {len(piece_names)}")
        print(f"   Unique types: {len(unique_types)}")
        print(f"   Types: {list(unique_types)}")
        
        # Test with all squares occupied (previous working version)
        print("\n🧪 Testing with all squares occupied...")
        all_occupied = [True] * 64
        pieces_all = piece_classifier.classify_pieces(img_array, corners, all_occupied, chess.WHITE)
        
        all_piece_names = []
        for i, piece in enumerate(pieces_all):
            if piece is not None:
                piece_name = f"{'white' if piece.color else 'black'}_{piece.symbol().lower()}"
                all_piece_names.append(piece_name)
        
        all_unique_types = set(all_piece_names)
        print(f"   All occupied pieces: {len(all_piece_names)}")
        print(f"   All occupied unique types: {len(all_unique_types)}")
        print(f"   All occupied types: {list(all_unique_types)}")
        
        # Compare results
        print(f"\n🔍 COMPARISON:")
        print(f"   Real occupancy: {len(unique_types)} types")
        print(f"   All occupied: {len(all_unique_types)} types")
        print(f"   Difference: {len(all_unique_types) - len(unique_types)} types")
        
        if len(unique_types) < len(all_unique_types):
            print("   ⚠️  Real occupancy is detecting fewer piece types!")
            print("   This suggests the occupancy data might be filtering out some pieces")
        else:
            print("   ✅ Both methods detect similar piece types")
            
    except Exception as e:
        print(f"❌ Error in piece classification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_piece_classification()