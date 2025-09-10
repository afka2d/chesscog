#!/usr/bin/env python3
"""
Debug the piece classification issue where all pieces are predicted as pawns.
"""

import numpy as np
from PIL import Image
import chess
from simple_piece_classifier import SimplePieceClassifier
from chesscog.recognition.recognition import ChessRecognizer
from pathlib import Path

def debug_piece_classification_issue():
    """Debug why all pieces are being predicted as pawns."""
    print("🔍 Debugging Piece Classification Issue")
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
        piece_objects = []
        for i, piece in enumerate(pieces_1d):
            if piece is not None:
                if isinstance(piece, str):
                    piece_names.append(piece)
                    print(f"   Square {i}: {piece} (string)")
                elif hasattr(piece, 'symbol'):
                    piece_name = f"{'white' if piece.color else 'black'}_{piece.symbol().lower()}"
                    piece_names.append(piece_name)
                    piece_objects.append(piece)
                    print(f"   Square {i}: {piece_name} (chess.Piece)")
                else:
                    piece_names.append(str(piece))
                    print(f"   Square {i}: {piece} (unknown type: {type(piece)})")
        
        unique_types = set(piece_names)
        print(f"\n📊 ANALYSIS:")
        print(f"   Total pieces: {len(piece_names)}")
        print(f"   Unique types: {len(unique_types)}")
        print(f"   Types: {list(unique_types)}")
        
        # Check if all pieces are pawns
        pawn_count = sum(1 for name in piece_names if 'p' in name.lower())
        print(f"   Pawn count: {pawn_count}/{len(piece_names)}")
        
        if pawn_count == len(piece_names) and len(piece_names) > 0:
            print("   ⚠️  ALL PIECES ARE PAWNS! This suggests a model issue.")
        elif pawn_count > len(piece_names) * 0.8:
            print("   ⚠️  MOSTLY PAWNS! This suggests a model bias.")
        else:
            print("   ✅ Good diversity in piece types")
            
        # Test the piece classifier's internal state
        print(f"\n🔍 PIECE CLASSIFIER INTERNAL STATE:")
        print(f"   Model loaded: {hasattr(piece_classifier, '_pieces_model')}")
        print(f"   Transforms loaded: {hasattr(piece_classifier, '_pieces_transforms')}")
        print(f"   Piece classes: {getattr(piece_classifier, '_piece_classes', 'Not loaded')}")
        
        if hasattr(piece_classifier, '_piece_classes'):
            print(f"   Piece classes shape: {piece_classifier._piece_classes.shape}")
            print(f"   Piece classes: {piece_classifier._piece_classes}")
            
    except Exception as e:
        print(f"❌ Error in piece classification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_piece_classification_issue()
