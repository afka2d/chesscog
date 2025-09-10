#!/usr/bin/env python3
"""
Simple debug of the piece extraction process.
"""

import numpy as np
from PIL import Image
import chess
from simple_piece_classifier import SimplePieceClassifier
from chesscog.recognition.recognition import ChessRecognizer
from pathlib import Path
import torch

def debug_simple_piece_extraction():
    """Debug the piece extraction process simply."""
    print("🔍 Debugging Simple Piece Extraction")
    print("=" * 40)
    
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
    
    # Test the actual piece classification
    print(f"\n🎲 TESTING ACTUAL PIECE CLASSIFICATION:")
    try:
        pieces_1d = piece_classifier.classify_pieces(img_array, corners, occupancy.tolist(), chess.WHITE)
        
        # Analyze results
        piece_names = []
        occupied_squares = []
        
        for i, piece in enumerate(pieces_1d):
            if piece is not None:
                rank, file = i // 8, i % 8
                occupied_squares.append((rank, file))
                
                if hasattr(piece, 'symbol'):
                    piece_name = f"{'white' if piece.color else 'black'}_{piece.symbol().lower()}"
                    piece_names.append(piece_name)
                    print(f"   Square {i} (rank {rank}, file {file}): {piece_name}")
                else:
                    piece_names.append(str(piece))
                    print(f"   Square {i} (rank {rank}, file {file}): {piece}")
        
        print(f"\n📊 CLASSIFICATION SUMMARY:")
        print(f"   Total pieces classified: {len(piece_names)}")
        print(f"   Occupied squares: {occupied_squares}")
        
        if piece_names:
            from collections import Counter
            piece_counts = Counter(piece_names)
            print(f"   Piece counts: {dict(piece_counts)}")
            
            pawn_count = sum(1 for name in piece_names if 'p' in name.lower())
            pawn_ratio = pawn_count / len(piece_names) if piece_names else 0
            print(f"   Pawn ratio: {pawn_count}/{len(piece_names)} ({pawn_ratio*100:.1f}%)")
            
            if pawn_ratio > 0.7:
                print("   ⚠️  STRONG PAWN BIAS DETECTED!")
            elif pawn_ratio > 0.5:
                print("   ⚠️  MODERATE PAWN BIAS DETECTED!")
            else:
                print("   ✅ No significant pawn bias")
        
        # Test with a different image to see if it's image-specific
        print(f"\n🔄 TESTING WITH DIFFERENT IMAGE:")
        try:
            # Create a simple test image
            test_img = Image.new('RGB', (1200, 1200), color='white')
            test_img_array = np.array(test_img)
            
            # Test with all squares occupied
            test_occupancy = [True] * 64
            test_pieces = piece_classifier.classify_pieces(test_img_array, corners, test_occupancy, chess.WHITE)
            
            test_piece_names = []
            for piece in test_pieces:
                if piece is not None and hasattr(piece, 'symbol'):
                    piece_name = f"{'white' if piece.color else 'black'}_{piece.symbol().lower()}"
                    test_piece_names.append(piece_name)
            
            if test_piece_names:
                test_piece_counts = Counter(test_piece_names)
                print(f"   Test image piece counts: {dict(test_piece_counts)}")
                
                test_pawn_count = sum(1 for name in test_piece_names if 'p' in name.lower())
                test_pawn_ratio = test_pawn_count / len(test_piece_names) if test_piece_names else 0
                print(f"   Test image pawn ratio: {test_pawn_count}/{len(test_piece_names)} ({test_pawn_ratio*100:.1f}%)")
            
        except Exception as e:
            print(f"   Error testing with different image: {e}")
        
    except Exception as e:
        print(f"   Error in piece classification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_simple_piece_extraction()
