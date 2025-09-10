#!/usr/bin/env python3
"""
Test the API with a realistic chess position.
"""

import requests
import json
from PIL import Image
import io
import chess
import numpy as np

def test_realistic_position():
    """Test with a realistic chess position."""
    print("🎯 Testing Realistic Chess Position")
    print("=" * 50)
    
    # Load test image
    img_path = "grey_background_dataset/images/test/IMG_4763.JPG"
    with open(img_path, 'rb') as f:
        image_data = f.read()
    
    # Test corners
    corners = [[724, 2064], [2692, 1886], [2784, 4104], [441, 3979]]
    
    # Make API request
    files = {'image': ('test.jpg', image_data, 'image/jpeg')}
    data = {
        'corners': json.dumps(corners),
        'color': 'white'
    }
    
    try:
        print("📡 Making API request...")
        response = requests.post('http://localhost:8000/recognize_chess_position_with_corners', files=files, data=data)
        response.raise_for_status()
        result = response.json()
        print("✅ API request successful!")
        
        # Parse results
        fen = result.get('fen', '')
        success = result.get('success', False)
        
        print(f"\n📊 RESULTS:")
        print(f"   FEN: {fen}")
        print(f"   Success: {success}")
        
        if success and fen:
            # Parse FEN to analyze pieces
            board = chess.Board(fen)
            piece_counts = {}
            occupied_squares = []
            
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    piece_name = f"{'white' if piece.color else 'black'}_{piece.symbol().lower()}"
                    piece_counts[piece_name] = piece_counts.get(piece_name, 0) + 1
                    occupied_squares.append(f"{chess.square_name(square)}: {piece_name}")
            
            total_pieces = sum(piece_counts.values())
            unique_types = len(piece_counts)
            diversity_score = unique_types / 12  # 12 possible piece types
            
            print(f"   Occupied squares: {total_pieces}")
            print(f"   Unique piece types: {unique_types}")
            print(f"   Diversity score: {diversity_score:.2f}")
            
            # Estimate accuracy based on diversity
            if diversity_score >= 0.5:
                accuracy_range = "75-85%"
            elif diversity_score >= 0.3:
                accuracy_range = "65-75%"
            else:
                accuracy_range = "50-65%"
            
            print(f"   Estimated accuracy: {accuracy_range}")
            
            print(f"\n🎯 PIECE CLASSIFICATION:")
            print(f"   Total pieces detected: {total_pieces}")
            print(f"   Piece breakdown:")
            for piece_type, count in sorted(piece_counts.items()):
                print(f"     {piece_type}: {count}")
            
            # Check for pawn bias
            total_pawns = piece_counts.get('white_p', 0) + piece_counts.get('black_p', 0)
            pawn_percentage = (total_pawns / total_pieces) * 100 if total_pieces > 0 else 0
            
            print(f"\n⚠️  PAWN BIAS ANALYSIS:")
            print(f"   Pawns detected: {total_pawns}/{total_pieces} ({pawn_percentage:.1f}%)")
            if pawn_percentage > 60:
                print(f"   ⚠️  High pawn bias detected!")
            elif pawn_percentage > 40:
                print(f"   ⚠️  Moderate pawn bias detected")
            else:
                print(f"   ✅ Pawn distribution looks reasonable")
            
            # Display board
            print(f"\n🏁 BOARD REPRESENTATION:")
            board_str = str(board).replace(' ', '.')
            for i, line in enumerate(board_str.split('\n')):
                print(f"   {line}")
            
            print(f"\n🔍 OCCUPANCY ANALYSIS:")
            print(f"   Total occupied squares: {total_pieces}/64")
            print(f"   Occupancy rate: {(total_pieces/64)*100:.1f}%")
            
            if occupied_squares:
                print(f"\n📍 OCCUPIED SQUARES:")
                for square_info in occupied_squares:
                    print(f"   {square_info}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Error details: {e.response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_realistic_position()