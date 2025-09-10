#!/usr/bin/env python3
"""
Test piece classification directly to debug the issue.
"""

import requests
import json
from PIL import Image
import io
import chess
import numpy as np

def test_piece_classification_direct():
    """Test piece classification directly."""
    print("🔍 Testing Piece Classification Directly")
    print("=" * 50)
    
    # Load test image
    img_path = "grey_background_dataset/images/test/IMG_4763.JPG"
    with open(img_path, 'rb') as f:
        image_data = f.read()
    
    # Test corners
    corners = [[724, 2064], [2692, 1886], [2784, 4104], [441, 3979]]
    
    # Make multiple API requests to see consistency
    for i in range(3):
        print(f"\n🔄 Test {i+1}/3:")
        
        files = {'image': ('test.jpg', image_data, 'image/jpeg')}
        data = {
            'corners': json.dumps(corners),
            'color': 'white'
        }
        
        try:
            response = requests.post('http://localhost:8000/recognize_chess_position_with_corners', files=files, data=data)
            response.raise_for_status()
            result = response.json()
            
            # Parse results
            fen = result.get('fen', '')
            success = result.get('success', False)
            
            print(f"   FEN: {fen}")
            print(f"   Success: {success}")
            
            if success and fen:
                # Parse FEN to analyze pieces
                board = chess.Board(fen)
                piece_counts = {}
                
                for square in chess.SQUARES:
                    piece = board.piece_at(square)
                    if piece:
                        piece_name = f"{'white' if piece.color else 'black'}_{piece.symbol().lower()}"
                        piece_counts[piece_name] = piece_counts.get(piece_name, 0) + 1
                
                total_pieces = sum(piece_counts.values())
                unique_types = len(piece_counts)
                
                print(f"   Pieces: {total_pieces}, Types: {unique_types}")
                print(f"   Breakdown: {dict(piece_counts)}")
                
                # Check for pawn bias
                total_pawns = piece_counts.get('white_p', 0) + piece_counts.get('black_p', 0)
                pawn_percentage = (total_pawns / total_pieces) * 100 if total_pieces > 0 else 0
                print(f"   Pawns: {total_pawns}/{total_pieces} ({pawn_percentage:.1f}%)")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_piece_classification_direct()
